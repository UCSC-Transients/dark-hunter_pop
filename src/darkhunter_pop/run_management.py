"""Stage registry, caching/skip logic, run-file I/O, and per-stage source hashing.

ARCHITECTURE.md §5. Every stage registers under its canonical name. Science stages are not
executed here yet — this module is the contract they must conform to.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from darkhunter_pop.config_loader import (
    assert_config_checksum,
    config_checksum,
    enabled_selection_content_fingerprint,
    repo_root,
)
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import (
    RunManifest,
    StageRecord,
    StageStatus,
    SyntheticStandIn,
)

_RUNS_DIR = repo_root() / "runs"


@dataclass(frozen=True)
class StageSpec:
    """Registry entry for one named pipeline stage."""

    name: str
    module: str
    inputs_from: tuple[str, ...]
    dependency_modules: tuple[str, ...]
    """Import paths whose source files feed ``source_hash`` (science-affecting only)."""

    config_fingerprint_keys: tuple[str, ...]
    """Dotted keys into the config JSON dump that parameterize this stage's artifact path."""

    uses_gaiamock: bool = False


# Canonical pipeline order (ARCHITECTURE.md §4 headers).
STAGE_ORDER: tuple[str, ...] = (
    "data_acquisition",
    "mass_derivation_bulk",
    "sample_selection",
    "mass_derivation_refined",
    "rv_astrometry_gate",
    "joint_orbit_fit",
    "companion_nature_likelihood",
    "triples",
    "selection_function_astrometric",
    "selection_function_followup",
    "population_model",
    "sensitivity_analysis",
    "inference",
    "diagnostics",
)


def _spec(
    name: str,
    module: str,
    *,
    inputs_from: Sequence[str] = (),
    deps: Sequence[str] | None = None,
    config_keys: Sequence[str] = (),
    uses_gaiamock: bool = False,
) -> StageSpec:
    dependency_modules = tuple(deps) if deps is not None else (module,)
    return StageSpec(
        name=name,
        module=module,
        inputs_from=tuple(inputs_from),
        dependency_modules=dependency_modules,
        config_fingerprint_keys=tuple(config_keys),
        uses_gaiamock=uses_gaiamock,
    )


STAGE_REGISTRY: dict[str, StageSpec] = {
    s.name: s
    for s in (
        _spec(
            "data_acquisition",
            "darkhunter_pop.data_acquisition",
            deps=(
                "darkhunter_pop.data_acquisition",
                "darkhunter_pop.nss_covariance",
            ),
            config_keys=(
                "active_dr_mode",
                "dr3.quality_cut_bins",
                "dr4.quality_cut_bins",
                "dr3.external_photometry_crossmatches",
                "dr4.external_photometry_crossmatches",
                "dr3.nss_table",
                "dr4.nss_table",
            ),
        ),
        _spec(
            "mass_derivation_bulk",
            "darkhunter_pop.mass_derivation",
            inputs_from=("data_acquisition",),
            deps=(
                "darkhunter_pop.mass_derivation",
                "darkhunter_pop.constants",
            ),
            config_keys=(
                "mass_calibration",
                "mass_derivation.dark_companion_flux_ratio",
                "classification.M_MIN_msun",
                "classification.n_sigma_mass_cut",
            ),
            uses_gaiamock=True,
        ),
        _spec(
            "sample_selection",
            "darkhunter_pop.sample_selection",
            inputs_from=("mass_derivation_bulk",),
            deps=(
                "darkhunter_pop.sample_selection",
                "darkhunter_pop.elbadry2026_m2_sigma",
                "darkhunter_pop.janssens_mass",
                "darkhunter_pop.elbadry2026_selection",
                "darkhunter_pop.mc_mass_function",
                "darkhunter_pop.physics_utils",
                "darkhunter_pop.sensitivity_analysis",
                "darkhunter_pop.data_acquisition",
            ),
            config_keys=("sample_selection", "mc_mass_function"),
        ),
        _spec(
            "mass_derivation_refined",
            "darkhunter_pop.mass_derivation",
            inputs_from=("mass_derivation_bulk",),
            deps=(
                "darkhunter_pop.mass_derivation",
                "darkhunter_pop.constants",
            ),
            config_keys=(
                "mass_calibration",
                "mass_derivation",
                "active_dr_mode",
            ),
        ),
        _spec(
            "rv_astrometry_gate",
            "darkhunter_pop.rv_consistency",
            inputs_from=("mass_derivation_refined",),
            config_keys=("rv_consistency",),
        ),
        _spec(
            "joint_orbit_fit",
            "darkhunter_pop.rv_consistency",
            inputs_from=("rv_astrometry_gate",),
            config_keys=("rv_consistency",),
        ),
        _spec(
            "companion_nature_likelihood",
            "darkhunter_pop.companion_nature",
            inputs_from=("joint_orbit_fit", "mass_derivation_refined"),
            deps=(
                "darkhunter_pop.companion_nature",
                "darkhunter_pop.phot_sed_adapter",
            ),
            config_keys=(
                "companion_nature",
                "mass_derivation.phot_sed_root",
                "mass_derivation.phot_sed_filename_template",
                "physics.cooling_tracks",
                "physics.cooling_atmosphere",
                "physics.cooling_tracks_path",
            ),
        ),
        _spec(
            "triples",
            "darkhunter_pop.triples",
            inputs_from=("companion_nature_likelihood",),
            deps=(
                "darkhunter_pop.triples",
                "darkhunter_pop.triples.tess_variability",
                "darkhunter_pop.triples.rotation_check",
            ),
            config_keys=("triples",),
        ),
        _spec(
            "selection_function_astrometric",
            "darkhunter_pop.forward_model",
            inputs_from=("data_acquisition",),
            config_keys=(
                "gaiamock",
                "active_dr_mode",
                "physics",
                "selection_function_astrometric",
                "dr3.selection_function_astrometric",
                "dr4.selection_function_astrometric",
            ),
            uses_gaiamock=True,
        ),
        _spec(
            "selection_function_followup",
            "darkhunter_pop.forward_model",
            inputs_from=("selection_function_astrometric",),
            config_keys=(
                "active_dr_mode",
                "physics",
                "selection_function_followup",
                "dr3.selection_function_followup",
                "dr4.selection_function_followup",
            ),
        ),
        _spec(
            "population_model",
            "darkhunter_pop.population_model",
            inputs_from=(
                "companion_nature_likelihood",
                "selection_function_astrometric",
                "selection_function_followup",
            ),
            deps=(
                "darkhunter_pop.population_model",
                "darkhunter_pop.constants",
            ),
            config_keys=(
                "population_model",
                "physics",
                "classification",
                "mass_calibration.delta_M_Ch_msun",
            ),
        ),
        _spec(
            "sensitivity_analysis",
            "darkhunter_pop.sensitivity_analysis",
            inputs_from=("population_model",),
            config_keys=(
                "physics.mc_noise_threshold",
                "sensitivity_analysis",
            ),
        ),
        _spec(
            "inference",
            "darkhunter_pop.inference",
            inputs_from=(
                "population_model",
                "selection_function_astrometric",
                "selection_function_followup",
                "sensitivity_analysis",
                "sample_selection",
            ),
            deps=(
                "darkhunter_pop.inference",
                "darkhunter_pop.physics_utils",
                "darkhunter_pop.population_model",
                "darkhunter_pop.sample_inclusion",
                "darkhunter_pop.spuriousness_model",
            ),
            config_keys=(
                "inference",
                "physics",
                "classification",
                "population_model",
                "mass_calibration.delta_M_Ch_msun",
                "sample_selection",
            ),
        ),
        _spec(
            "diagnostics",
            "darkhunter_pop.diagnostics",
            inputs_from=("inference",),
            deps=(
                "darkhunter_pop.diagnostics",
                "darkhunter_pop.sample_diagnostics",
                "darkhunter_pop.benchmarks",
                "darkhunter_pop.sbc",
            ),  # plotting excluded from hash
            config_keys=("paths.artifact_root", "diagnostics", "benchmarks"),
        ),
    )
}


class StageAction(str, Enum):
    RUN = "run"
    SKIP_CACHED = "skip_cached"
    SKIP_REASON = "skip_reason"
    #: A completed/cached record exists, but its ``source_hash`` or artifact
    #: fingerprint no longer matches what the current code/config produce. The
    #: artifact is stale: neither reused nor silently re-run (ARCHITECTURE.md §5).
    REFUSE_STALE = "refuse_stale"


class StaleStageCacheError(RuntimeError):
    """A cached stage artifact is stale w.r.t. current code/config.

    Raised instead of silently reusing the stale artifact or silently
    re-running the stage. Matches the run-management refusal convention used
    for a config-checksum or gaiamock-version mismatch: the operator must pass
    an explicit ``--force-rerun <stage>`` (which starts a new run file) to
    proceed.
    """


# Canonical skip detail when ``config.triples.enabled`` is false (ARCHITECTURE.md §4).
TRIPLES_DISABLED_SKIP_REASON = "triples.enabled=false"
# Canonical skip details for ``sample_selection`` (CONTINUATION_PLAN §12.3).
SAMPLE_SELECTION_DISABLED_SKIP_REASON = "sample_selection.enabled=false"
SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON = "sample_selection: no enabled samples"


@dataclass(frozen=True)
class StagePlanEntry:
    stage: str
    action: StageAction
    detail: str
    artifact_path: Path | None = None


def stage_default_skip_reason(
    spec: StageSpec, config: PipelineConfig
) -> str | None:
    """Config-driven skip reasons known to ``run_management`` (no science execution).

    Currently: the ``triples`` stage is off by default. Callers may still pass an
    explicit ``skip_reason`` to ``plan_stage`` (e.g. ``rv_astrometry_gate_failed``).
    """
    if spec.name == "triples" and not config.triples.enabled:
        return TRIPLES_DISABLED_SKIP_REASON
    if spec.name == "sample_selection":
        if not config.sample_selection.enabled:
            return SAMPLE_SELECTION_DISABLED_SKIP_REASON
        if not any(entry.enabled for entry in config.sample_selection.samples):
            return SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON
    return None


def runs_dir() -> Path:
    return _RUNS_DIR


def module_file_path(module_name: str) -> Path:
    """Resolve a dotted module to its ``.py`` file under ``src/``."""
    parts = module_name.split(".")
    if parts[0] != "darkhunter_pop":
        raise ValueError(f"expected darkhunter_pop.* module, got {module_name}")
    base = repo_root() / "src" / Path(*parts)
    py_file = base.with_suffix(".py")
    init_file = base / "__init__.py"
    if py_file.is_file():
        return py_file
    if init_file.is_file():
        return init_file
    raise FileNotFoundError(f"no source file for module {module_name}")


def compute_source_hash(spec: StageSpec) -> str:
    """SHA256 over the concatenation of dependency module source files."""
    digest = hashlib.sha256()
    for module_name in spec.dependency_modules:
        path = module_file_path(module_name)
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dig(mapping: Mapping[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def config_subset_for_stage(
    config: PipelineConfig, spec: StageSpec
) -> dict[str, Any]:
    dump = config.model_dump(mode="json")
    subset: dict[str, Any] = {}
    for key in spec.config_fingerprint_keys:
        subset[key] = _dig(dump, key)
    # Always include active mode so DR switches cannot reuse paths.
    subset["active_dr_mode"] = config.active_dr_mode.value
    # CONTINUATION_PLAN §12.3: artifact path keyed on the content hash of every
    # enabled selection file plus its mode, so a threshold edit cannot reuse cache.
    if spec.name == "sample_selection":
        subset["enabled_selection_content_sha256"] = (
            enabled_selection_content_fingerprint(config)
        )
    return subset


def config_subset_fingerprint(subset: Mapping[str, Any]) -> str:
    blob = yaml.safe_dump(dict(subset), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def stage_artifact_path(
    config: PipelineConfig,
    spec: StageSpec,
    *,
    run_id: str,
) -> Path:
    """One HDF5 per stage, parameterized by the stage's config subset."""
    subset = config_subset_for_stage(config, spec)
    fp = config_subset_fingerprint(subset)
    root = Path(config.paths.artifact_root)
    if not root.is_absolute():
        root = repo_root() / root
    return root / run_id / spec.name / f"{fp}.h5"


def short_git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root()), "rev-parse", "--short=7", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_run_id(*, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(tz=timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{short_git_commit()}"


def run_file_path(run_id: str, runs: Path | None = None) -> Path:
    return (runs or _RUNS_DIR) / f"{run_id}.yaml"


def load_run_manifest(path: Path) -> RunManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunManifest.model_validate(raw)


def save_run_manifest(manifest: RunManifest, path: Path | None = None) -> Path:
    out = path or run_file_path(manifest.run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def parse_run_id_timestamp(run_id: str) -> datetime:
    """Parse ``YYYYMMDD-HHMMSS`` prefix from ``run_id`` (not filesystem mtime)."""
    prefix = "-".join(run_id.split("-")[:2])
    return datetime.strptime(prefix, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


def list_run_manifests(runs: Path | None = None) -> list[RunManifest]:
    root = runs or _RUNS_DIR
    if not root.is_dir():
        return []
    manifests: list[RunManifest] = []
    for path in root.glob("*.yaml"):
        try:
            manifests.append(load_run_manifest(path))
        except Exception:
            continue
    manifests.sort(key=lambda m: parse_run_id_timestamp(m.run_id), reverse=True)
    return manifests


def list_incomplete_runs(runs: Path | None = None) -> list[RunManifest]:
    return [m for m in list_run_manifests(runs) if m.is_incomplete()]


def format_incomplete_runs_table(manifests: Sequence[RunManifest]) -> str:
    lines = [
        "run_id                         status     last_stage                 created_at                 checksum",
        "-" * 110,
    ]
    for m in manifests:
        last = _last_completed_stage(m) or "-"
        status = "incomplete"
        created = m.created_at.isoformat()
        lines.append(
            f"{m.run_id:<30} {status:<10} {last:<24} {created:<24} {m.config_checksum[:12]}"
        )
    return "\n".join(lines)


def _last_completed_stage(manifest: RunManifest) -> str | None:
    terminal = {StageStatus.COMPLETED, StageStatus.CACHED, StageStatus.SKIPPED}
    last: str | None = None
    for name in STAGE_ORDER:
        rec = manifest.stages.get(name)
        if rec is not None and rec.status in terminal:
            last = name
    return last


def create_run_manifest(
    config: PipelineConfig,
    *,
    parent_run_id: str | None = None,
    stages_seed: Mapping[str, StageRecord] | None = None,
    when: datetime | None = None,
    random_seeds: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    dry_run_label: str | None = None,
    synthetic_stand_ins: Sequence[SyntheticStandIn] = (),
) -> RunManifest:
    """Build a fresh ``RunManifest`` stamped with everything known at run birth.

    Parameters
    ----------
    config:
        Loaded pipeline config; supplies the checksum, active DR mode, artifact
        root, host profile and gaiamock version triple.
    parent_run_id:
        Set when copying prior stage records forward from a force-re-run.
    stages_seed:
        Stage records copied forward (``copy_stages_before``).
    when:
        Creation timestamp; also fixes the ``run_id`` prefix. Defaults to now.
    random_seeds:
        Seeds actually in effect for this run, recorded for accounting —
        ``dynesty`` is validated by multi-run posterior agreement, never by
        bitwise seed replay (ARCHITECTURE.md §4 ``inference``).
    dry_run / dry_run_label / synthetic_stand_ins:
        Dry-run labeling, applied **at birth** (issue #201). ``RunManifest``
        refuses ``dry_run=True`` without both a label and at least one declared
        stand-in, so a mislabeled dry run cannot be written at all.

    Limitations
    -----------
    Nothing here validates that ``random_seeds`` matches what the stages will
    actually consume, nor that ``synthetic_stand_ins`` is exhaustive — both are
    declarations the caller is responsible for.
    """
    now = when or datetime.now(tz=timezone.utc)
    run_id = make_run_id(when=now)
    return RunManifest(
        run_id=run_id,
        created_at=now,
        parent_run_id=parent_run_id,
        config_checksum=config_checksum(config),
        active_dr_mode=config.active_dr_mode,
        artifact_root=config.paths.artifact_root,
        host_profile=config.paths.host_profile,
        gaiamock_mod_release=config.gaiamock.mod_release,
        gaiamock_mod_sha256=config.gaiamock.mod_sha256,
        gaiamock_git_commit=config.gaiamock.git_commit,
        random_seeds=dict(random_seeds or {}),
        dry_run=dry_run,
        dry_run_label=dry_run_label,
        synthetic_stand_ins=[s.model_copy(deep=True) for s in synthetic_stand_ins],
        stages=dict(stages_seed or {}),
    )


def copy_stages_before(
    parent: RunManifest, stage_name: str
) -> dict[str, StageRecord]:
    """Copy completion records for stages strictly before ``stage_name`` in ``STAGE_ORDER``."""
    if stage_name not in STAGE_REGISTRY:
        raise KeyError(stage_name)
    idx = STAGE_ORDER.index(stage_name)
    prior = set(STAGE_ORDER[:idx])
    return {
        name: record.model_copy(deep=True)
        for name, record in parent.stages.items()
        if name in prior
    }


def new_run_for_force_rerun(
    parent: RunManifest,
    config: PipelineConfig,
    stage_name: str,
) -> RunManifest:
    """Force-re-run of a completed stage → new run file with prior stages copied.

    The parent's seeds and dry-run labeling are carried forward: a child of a
    dry run is still a dry run, with the same declared stand-ins (issue #201).
    """
    assert_config_checksum(config, parent.config_checksum)
    seed = copy_stages_before(parent, stage_name)
    return create_run_manifest(
        config,
        parent_run_id=parent.run_id,
        stages_seed=seed,
        random_seeds=parent.random_seeds,
        dry_run=parent.dry_run,
        dry_run_label=parent.dry_run_label,
        synthetic_stand_ins=parent.synthetic_stand_ins,
    )


def resolve_run_file(
    *,
    run_file: Path | None,
    config: PipelineConfig,
    runs: Path | None = None,
) -> tuple[RunManifest, Path, bool]:
    """Select or create a run file.

    Returns ``(manifest, path, created_new)``.

    If ``run_file`` is omitted and any incomplete runs exist, raises ``SystemExit``-style
    ``RuntimeError`` after formatting the incomplete-run table (caller prints / exits).
    """
    root = runs or _RUNS_DIR
    if run_file is not None:
        manifest = load_run_manifest(run_file)
        assert_config_checksum(config, manifest.config_checksum)
        return manifest, run_file, False

    incomplete = list_incomplete_runs(root)
    if incomplete:
        table = format_incomplete_runs_table(incomplete)
        raise RuntimeError(
            "one or more incomplete runs exist; pass --run-file explicitly.\n" + table
        )

    manifest = create_run_manifest(config)
    path = run_file_path(manifest.run_id, root)
    save_run_manifest(manifest, path)
    return manifest, path, True


def assert_stage_source_hash(
    spec: StageSpec, recorded: str | None, *, require_recorded: bool
) -> str:
    """Check only this stage's hash (not upstream). Returns the current hash."""
    current = compute_source_hash(spec)
    if require_recorded and recorded is not None and recorded != current:
        raise ValueError(
            f"source_hash mismatch for stage {spec.name}:\n"
            f"  current={current}\n"
            f"  recorded={recorded}\n"
            "Re-run this stage (amend if mid-pipeline resume) or start a new run."
        )
    return current


#: Placeholder printed in place of a stage record's ``source_hash`` when the
#: record carries none at all (pre-dating hash recording). Such a record is
#: refused exactly like a mismatched hash (issue #169, ARCHITECTURE.md §5).
MISSING_SOURCE_HASH_DETAIL = "<missing>"


def source_hash_stale_reason(spec: StageSpec, recorded: str | None) -> str | None:
    """Why ``recorded`` fails this stage's source-hash check, or ``None`` if it passes.

    A **missing** recorded hash (``None`` or empty) is treated identically to a
    **mismatched** one: both return a reason, so the caller refuses rather than
    reusing the artifact (issue #169; Ryan's decision of 2026-09-14). Rationale:
    a record with no hash cannot be shown to match the current dependency
    modules, and the run-management posture on uncertainty is to refuse and
    require an explicit ``--force-rerun``. The no-hash population is finite —
    only run files written before hash recording — because every freshly
    executed stage records a real hash (``mark_stage_started`` /
    ``mark_stage_finished``).

    Parameters
    ----------
    spec:
        Registered stage spec whose ``dependency_modules`` define the hash.
    recorded:
        The hash stored on the stage record, or ``None`` / ``""`` when absent.

    Returns
    -------
    str | None
        ``"source_hash recorded=... current=..."`` when stale (with
        ``MISSING_SOURCE_HASH_DETAIL`` standing in for an absent recorded
        hash), otherwise ``None``.

    Limitations
    -----------
    Only *this* stage's hash is considered — upstream stages are never checked
    here (ARCHITECTURE.md §5). Hashing the dependency modules requires their
    source files to be readable in the working tree; ``compute_source_hash``
    raises if one is missing.
    """
    current = compute_source_hash(spec)
    if recorded == current:
        return None
    shown = recorded if recorded else MISSING_SOURCE_HASH_DETAIL
    return f"source_hash recorded={shown} current={current}"


def stale_cache_detail(
    spec: StageSpec,
    record: StageRecord,
    current_artifact: Path,
) -> str | None:
    """Describe why a completed stage record is stale, or ``None`` if it is current.

    Two independent staleness conditions are checked against the **current**
    working tree and config, in the order they are reported:

    1. ``record.source_hash`` differs from ``compute_source_hash(spec)`` — a
       dependency module of this stage changed since the artifact was written.
       A record carrying **no** ``source_hash`` at all fails this condition too:
       missing is treated exactly like mismatched (issue #169).
    2. The recorded artifact's **file name** (the config-subset fingerprint)
       differs from ``current_artifact``'s — the stage's config subset now
       fingerprints differently, so the recorded artifact was produced under a
       different configuration. Only the file name is compared, never the full
       path: ``new_run_for_force_rerun`` legitimately copies prior stage records
       forward still pointing into the *parent* run's artifact directory, and
       that is not staleness.

    Parameters
    ----------
    spec:
        Registered stage spec whose ``dependency_modules`` define the hash.
    record:
        The manifest's completion record for this stage. Callers only pass
        records already known to be ``COMPLETED``/``CACHED`` with an existing
        artifact file.
    current_artifact:
        Artifact path the current config would produce for this run
        (``stage_artifact_path``).

    Limitations
    -----------
    A record carrying no ``source_hash`` (pre-dating hash recording) is reported
    stale on condition 1 — it cannot be shown to match, so it is refused and
    needs an explicit ``--force-rerun`` (issue #169). Consequence: run files
    written before hash recording are unresumable without that flag. Upstream
    stages are deliberately **not** checked here — per-stage hashing is scoped
    to the stage itself (ARCHITECTURE.md §5), and human judgment owns upstream
    re-runs.
    """
    reasons: list[str] = []
    hash_reason = source_hash_stale_reason(spec, record.source_hash)
    if hash_reason is not None:
        reasons.append(hash_reason)
    recorded_artifact = record.artifact_path
    if recorded_artifact and Path(recorded_artifact).name != current_artifact.name:
        reasons.append(
            f"artifact fingerprint recorded={Path(recorded_artifact).name} "
            f"current={current_artifact.name}"
        )
    if not reasons:
        return None
    return "; ".join(reasons)


def assert_plan_not_stale(entry: StagePlanEntry) -> None:
    """Raise ``StaleStageCacheError`` for a ``REFUSE_STALE`` plan entry.

    No-op for every other action, so callers can invoke it unconditionally right
    after ``plan_stage``.
    """
    if entry.action is StageAction.REFUSE_STALE:
        raise StaleStageCacheError(
            f"stale cached artifact for stage {entry.stage}: {entry.detail}\n"
            "Refusing to reuse it and refusing to silently re-run. Re-run "
            f"explicitly with --force-rerun {entry.stage} (which starts a new "
            "run file), or select a run whose record matches the current code "
            "and config."
        )


def plan_stage(
    spec: StageSpec,
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    force_rerun: bool = False,
    skip_reason: str | None = None,
    extra_note: str | None = None,
) -> StagePlanEntry:
    """Decide whether a stage should run, use cache, refuse, or skip.

    When ``skip_reason`` is omitted, ``stage_default_skip_reason`` may still skip
    (e.g. ``triples`` with ``enabled=false``). Explicit ``skip_reason`` wins.

    A completed/cached record is honored as ``SKIP_CACHED`` **only** when its
    recorded ``source_hash`` and artifact fingerprint still match what the
    current code and config produce; otherwise the entry is ``REFUSE_STALE``, so
    the run plan stays printable (``--dry-run`` reports the staleness) while
    execution refuses via ``assert_plan_not_stale``. A record with **no**
    recorded ``source_hash`` is refused identically to a mismatched one (#169).

    ``extra_note``, when given, is appended to the resulting entry's ``detail``
    (``"<detail> | <extra_note>"``) so a caller can surface a stage-specific,
    plan-time-known condition in the printed run plan without changing the
    cache/refuse/skip decision itself — e.g. ``mass_derivation_refined``'s
    degraded-SED-unavailable note (issue #181).
    """
    entry = _plan_stage_decision(spec, manifest, config, force_rerun=force_rerun, skip_reason=skip_reason)
    if extra_note:
        entry = dataclass_replace(entry, detail=f"{entry.detail} | {extra_note}")
    return entry


def _plan_stage_decision(
    spec: StageSpec,
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    force_rerun: bool = False,
    skip_reason: str | None = None,
) -> StagePlanEntry:
    """Core cache/refuse/skip/run decision for ``plan_stage`` (no note appending)."""
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)
    effective_skip = (
        skip_reason
        if skip_reason is not None
        else stage_default_skip_reason(spec, config)
    )
    if effective_skip:
        return StagePlanEntry(
            stage=spec.name,
            action=StageAction.SKIP_REASON,
            detail=effective_skip,
            artifact_path=artifact,
        )
    if force_rerun:
        return StagePlanEntry(
            stage=spec.name,
            action=StageAction.RUN,
            detail="running: force_rerun=True",
            artifact_path=artifact,
        )
    record = manifest.stages.get(spec.name)
    if (
        record is not None
        and record.status in {StageStatus.COMPLETED, StageStatus.CACHED}
        and record.artifact_path
        and Path(record.artifact_path).is_file()
    ):
        stale = stale_cache_detail(spec, record, artifact)
        if stale is not None:
            return StagePlanEntry(
                stage=spec.name,
                action=StageAction.REFUSE_STALE,
                detail=(
                    f"stale cache: recorded artifact {record.artifact_path} "
                    f"no longer matches current code/config ({stale}); "
                    f"re-run with --force-rerun {spec.name}"
                ),
                artifact_path=Path(record.artifact_path),
            )
        return StagePlanEntry(
            stage=spec.name,
            action=StageAction.SKIP_CACHED,
            detail=f"cached: output exists at {record.artifact_path}",
            artifact_path=Path(record.artifact_path),
        )
    if artifact.is_file():
        # Artifact sits at the current config fingerprint by construction, so
        # only the dependency hash can be stale here (and only if some record
        # exists to compare against). A record with no recorded hash is refused
        # exactly like a mismatched one (#169).
        if record is not None:
            hash_reason = source_hash_stale_reason(spec, record.source_hash)
            if hash_reason is not None:
                return StagePlanEntry(
                    stage=spec.name,
                    action=StageAction.REFUSE_STALE,
                    detail=(
                        f"stale cache: output exists at {artifact} but "
                        f"{hash_reason}; "
                        f"re-run with --force-rerun {spec.name}"
                    ),
                    artifact_path=artifact,
                )
        return StagePlanEntry(
            stage=spec.name,
            action=StageAction.SKIP_CACHED,
            detail=f"cached: output exists at {artifact}",
            artifact_path=artifact,
        )
    return StagePlanEntry(
        stage=spec.name,
        action=StageAction.RUN,
        detail="running: output missing",
        artifact_path=artifact,
    )


def format_run_plan(
    manifest: RunManifest,
    config: PipelineConfig,
    plan: Sequence[StagePlanEntry],
    *,
    run_path: Path,
    created_new: bool,
    dry_run: bool = False,
) -> str:
    """Fully legible run-plan screen output (exempt from caveman compression).

    When ``manifest.dry_run`` is set, the plan leads with the dry-run banner and
    prints **every** declared synthetic stand-in before the stage list, so an
    operator reading the plan before execution sees what was substituted without
    opening the run file (issue #201, EXECUTION_PLAN.md §7).

    Note the two independent senses of "dry run" here, deliberately kept
    distinct: the ``dry_run`` *argument* is ``--dry-run`` (plan and execute
    nothing), while ``manifest.dry_run`` is the run-level label meaning "built on
    documented substitutions, so not a science result".
    """
    if dry_run and created_new:
        run_status = "planned (not written — dry-run)"
    elif created_new:
        run_status = "created"
    else:
        run_status = "existing"
    lines = ["=== dark-hunter_pop run plan ==="]
    if manifest.dry_run:
        lines.append(f"*** {manifest.dry_run_label} ***")
    lines.extend(
        [
            f"run_file: {run_path} ({run_status})",
            f"run_id: {manifest.run_id}",
            f"active_dr_mode: {config.active_dr_mode.value}",
            f"config_checksum: {manifest.config_checksum}",
            f"artifact_root: {manifest.artifact_root}",
            f"host_profile: {manifest.host_profile or 'none (config.yaml as authored)'}",
            f"code_commit: {short_git_commit()}",
            f"gaiamock version triple: release={manifest.gaiamock_mod_release} "
            f"sha256={manifest.gaiamock_mod_sha256} "
            f"git_commit={manifest.gaiamock_git_commit}",
            f"random_seeds: {manifest.random_seeds or '(none recorded)'}",
            f"dry_run: {manifest.dry_run}",
        ]
    )
    lines.append(
        f"synthetic_stand_ins: {len(manifest.synthetic_stand_ins)} declared"
    )
    for stand_in in manifest.synthetic_stand_ins:
        lines.append(f"  - {stand_in.one_line()}")
        lines.append(f"      {stand_in.description}")
        if stand_in.config_keys:
            lines.append(
                "      config keys: " + ", ".join(stand_in.config_keys)
            )
        if stand_in.values:
            lines.append(f"      values: {stand_in.values}")
    lines.append("stages:")
    for entry in plan:
        subset = config_subset_for_stage(config, STAGE_REGISTRY[entry.stage])
        lines.append(f"  - {entry.stage}: {entry.detail}")
        lines.append(f"      config_subset: {subset}")
        if entry.artifact_path is not None:
            lines.append(f"      artifact: {entry.artifact_path}")
    lines.append("=== end run plan ===")
    return "\n".join(lines)


def mark_stage_started(
    manifest: RunManifest,
    spec: StageSpec,
    config: PipelineConfig,
    *,
    force_rerun: bool = False,
) -> RunManifest:
    now = datetime.now(tz=timezone.utc)
    stages = dict(manifest.stages)
    stages[spec.name] = StageRecord(
        stage_name=spec.name,
        status=StageStatus.RUNNING,
        started_at=now,
        source_hash=compute_source_hash(spec),
        config_subset=config_subset_for_stage(config, spec),
        artifact_path=str(stage_artifact_path(config, spec, run_id=manifest.run_id)),
        code_commit=short_git_commit(),
        force_rerun=force_rerun,
        gaiamock_mod_release=config.gaiamock.mod_release if spec.uses_gaiamock else None,
        gaiamock_mod_sha256=config.gaiamock.mod_sha256 if spec.uses_gaiamock else None,
        gaiamock_git_commit=config.gaiamock.git_commit if spec.uses_gaiamock else None,
    )
    return manifest.model_copy(update={"stages": stages})


def mark_stage_finished(
    manifest: RunManifest,
    spec: StageSpec,
    *,
    status: StageStatus,
    reason: str | None = None,
    artifact_path: Path | None = None,
) -> RunManifest:
    now = datetime.now(tz=timezone.utc)
    stages = dict(manifest.stages)
    prior = stages.get(spec.name)
    if prior is None:
        raise KeyError(f"stage {spec.name} was never started")
    stages[spec.name] = prior.model_copy(
        update={
            "status": status,
            "finished_at": now,
            "reason": reason,
            "artifact_path": (
                str(artifact_path) if artifact_path is not None else prior.artifact_path
            ),
            "source_hash": prior.source_hash or compute_source_hash(spec),
        }
    )
    return manifest.model_copy(update={"stages": stages})


def record_stage_resources(
    manifest: RunManifest,
    stage_name: str,
    *,
    wall_clock_seconds: float,
    rss_increase_bytes: int | None = None,
    rss_high_water_bytes: int | None = None,
) -> RunManifest:
    """Attach measured wall-clock / peak-RSS to an existing stage record.

    Kept separate from ``mark_stage_finished`` on purpose: measurement is the
    caller's business, not the stage protocol's, so none of the fourteen stage
    runners needs to know about it (issue #201, EXECUTION_PLAN.md §5.6).

    Parameters
    ----------
    manifest:
        Live run manifest; returned updated, never mutated in place.
    stage_name:
        Registered stage whose record receives the measurements. Must already
        exist on the manifest.
    wall_clock_seconds:
        Monotonic elapsed time spent inside the stage runner, in seconds. This
        is *not* ``finished_at - started_at``: it also covers the planning and
        cache-check work around the runner call.
    rss_increase_bytes:
        Increase in the process RSS high-water mark across the stage, in bytes,
        or ``None`` when unmeasured.
    rss_high_water_bytes:
        The process-lifetime RSS high-water mark at stage end, in bytes, or
        ``None``.

    Raises
    ------
    KeyError
        If ``stage_name`` has no record on the manifest.

    Limitations
    -----------
    Both figures derive from ``getrusage``'s monotonic high-water mark, so a
    stage peaking below an earlier stage's peak reports an increase of ``0``
    and cannot be ranked by these numbers alone. Neither covers shared pages,
    swap, or child processes.
    """
    stages = dict(manifest.stages)
    prior = stages.get(stage_name)
    if prior is None:
        raise KeyError(f"stage {stage_name} has no record to annotate")
    stages[stage_name] = prior.model_copy(
        update={
            "wall_clock_seconds": float(wall_clock_seconds),
            "rss_increase_bytes": (
                None if rss_increase_bytes is None else int(rss_increase_bytes)
            ),
            "rss_high_water_bytes": (
                None
                if rss_high_water_bytes is None
                else int(rss_high_water_bytes)
            ),
        }
    )
    return manifest.model_copy(update={"stages": stages})


@dataclass(frozen=True)
class StageGuardOutcome:
    """Result of ``plan_and_guard``: the plan entry plus what the runner must do.

    Attributes
    ----------
    plan:
        The ``StagePlanEntry`` produced by ``plan_stage`` for this stage.
    manifest:
        The manifest the caller must carry forward. Unchanged for
        ``SKIP_CACHED`` and ``RUN``; carries the ``skipped`` completion record
        for ``SKIP_REASON``.
    proceed:
        ``True`` only for ``StageAction.RUN``. A runner must return
        ``outcome.manifest`` immediately when this is ``False`` and perform no
        science work and no artifact write.
    """

    plan: StagePlanEntry
    manifest: RunManifest
    proceed: bool


def plan_and_guard(
    spec: StageSpec,
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    skip_reason: str | None = None,
) -> StageGuardOutcome:
    """Plan a stage and handle every non-``RUN`` ``StageAction`` exhaustively.

    The single entry point every stage runner uses instead of a bare
    ``plan_stage`` call, so the caching contract of ARCHITECTURE.md §5 is
    implemented once rather than re-derived per runner (issues #167, #172):

    * ``REFUSE_STALE`` → raises ``StaleStageCacheError`` via
      ``assert_plan_not_stale`` **before** any manifest mutation or artifact
      write, so a stale cached artifact is neither reused nor silently rebuilt.
    * ``SKIP_CACHED`` → ``proceed=False``, manifest untouched.
    * ``SKIP_REASON`` → records ``running`` then ``skipped`` (with
      ``plan.detail`` as the reason and no artifact) on the run file, and
      returns ``proceed=False``.
    * ``RUN`` → ``proceed=True``; the caller owns ``mark_stage_started`` and the
      rest of the stage protocol.

    Parameters
    ----------
    spec:
        Registry entry for the stage being planned (``STAGE_REGISTRY[name]``).
    manifest / config:
        Live run manifest and loaded pipeline config.
    run_path:
        Run-file path used to persist the ``SKIP_REASON`` records. Never written
        for any other action.
    force_rerun:
        Per-stage force-re-run override, forwarded to ``plan_stage``.
    skip_reason:
        Explicit skip detail (e.g. ``rv_astrometry_gate_failed``); wins over
        ``stage_default_skip_reason``.

    Limitations
    -----------
    This is a caching/run-file guard only: it performs no science and does not
    validate upstream artifacts. An unrecognized ``StageAction`` raises
    ``ValueError`` rather than defaulting to "run", so a future action added to
    the enum cannot silently reintroduce the fallthrough this helper exists to
    prevent.
    """
    plan = plan_stage(
        spec, manifest, config, force_rerun=force_rerun, skip_reason=skip_reason
    )
    assert_plan_not_stale(plan)

    if plan.action is StageAction.SKIP_CACHED:
        return StageGuardOutcome(plan=plan, manifest=manifest, proceed=False)

    if plan.action is StageAction.SKIP_REASON:
        updated = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
        save_run_manifest(updated, run_path)
        updated = mark_stage_finished(
            updated,
            spec,
            status=StageStatus.SKIPPED,
            reason=plan.detail,
            artifact_path=None,
        )
        save_run_manifest(updated, run_path)
        return StageGuardOutcome(plan=plan, manifest=updated, proceed=False)

    if plan.action is StageAction.RUN:
        return StageGuardOutcome(plan=plan, manifest=manifest, proceed=True)

    raise ValueError(
        f"unhandled StageAction {plan.action!r} for stage {plan.stage}"
    )


def wipe_stage_artifacts(manifest: RunManifest, stage_name: str) -> None:
    """Delete partial/complete artifact for a stage (mid-stage crash amend)."""
    record = manifest.stages.get(stage_name)
    if record is None or not record.artifact_path:
        return
    path = Path(record.artifact_path)
    if path.is_file():
        path.unlink()


def validate_registry_inputs_from() -> list[str]:
    """Return errors if ``inputs_from`` references unknown stages."""
    errors: list[str] = []
    for name, spec in STAGE_REGISTRY.items():
        for dep in spec.inputs_from:
            if dep not in STAGE_REGISTRY:
                errors.append(f"{name}.inputs_from references unknown stage {dep!r}")
    missing = [n for n in STAGE_ORDER if n not in STAGE_REGISTRY]
    extra = [n for n in STAGE_REGISTRY if n not in STAGE_ORDER]
    if missing:
        errors.append(f"STAGE_ORDER missing registry entries: {missing}")
    if extra:
        errors.append(f"registry has stages not in STAGE_ORDER: {extra}")
    return errors


def purge_run(
    run_path: Path,
    *,
    with_artifacts: bool = False,
    force: bool = False,
) -> None:
    """Delete a run YAML; optionally its recorded HDF5 artifacts."""
    manifest = load_run_manifest(run_path)
    if manifest.is_complete() and not force:
        raise ValueError(
            f"refusing to purge completed run {manifest.run_id}; pass force=True"
        )
    if with_artifacts:
        for record in manifest.stages.values():
            if record.artifact_path:
                path = Path(record.artifact_path)
                if path.is_file():
                    path.unlink()
    run_path.unlink()
