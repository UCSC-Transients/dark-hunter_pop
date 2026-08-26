"""Stage registry, caching/skip logic, run-file I/O, and per-stage source hashing.

ARCHITECTURE.md §5. Every stage registers under its canonical name. Science stages are not
executed here yet — this module is the contract they must conform to.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from darkhunter_pop.config_loader import (
    assert_config_checksum,
    config_checksum,
    repo_root,
)
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import (
    RunManifest,
    StageRecord,
    StageStatus,
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
            config_keys=("physics.cooling_tracks", "physics.cooling_atmosphere"),
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
            ),
            config_keys=("physics", "classification"),
        ),
        _spec(
            "diagnostics",
            "darkhunter_pop.diagnostics",
            inputs_from=("inference",),
            deps=("darkhunter_pop.diagnostics",),  # plotting excluded from hash
            config_keys=("paths.artifact_root", "diagnostics"),
        ),
    )
}


class StageAction(str, Enum):
    RUN = "run"
    SKIP_CACHED = "skip_cached"
    SKIP_REASON = "skip_reason"


# Canonical skip detail when ``config.triples.enabled`` is false (ARCHITECTURE.md §4).
TRIPLES_DISABLED_SKIP_REASON = "triples.enabled=false"


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
) -> RunManifest:
    now = when or datetime.now(tz=timezone.utc)
    run_id = make_run_id(when=now)
    return RunManifest(
        run_id=run_id,
        created_at=now,
        parent_run_id=parent_run_id,
        config_checksum=config_checksum(config),
        active_dr_mode=config.active_dr_mode,
        artifact_root=config.paths.artifact_root,
        gaiamock_mod_release=config.gaiamock.mod_release,
        gaiamock_mod_sha256=config.gaiamock.mod_sha256,
        gaiamock_git_commit=config.gaiamock.git_commit,
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
    """Force-re-run of a completed stage → new run file with prior stages copied."""
    assert_config_checksum(config, parent.config_checksum)
    seed = copy_stages_before(parent, stage_name)
    return create_run_manifest(
        config, parent_run_id=parent.run_id, stages_seed=seed
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


def plan_stage(
    spec: StageSpec,
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    force_rerun: bool = False,
    skip_reason: str | None = None,
) -> StagePlanEntry:
    """Decide whether a stage should run, use cache, or skip for another reason.

    When ``skip_reason`` is omitted, ``stage_default_skip_reason`` may still skip
    (e.g. ``triples`` with ``enabled=false``). Explicit ``skip_reason`` wins.
    """
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
        return StagePlanEntry(
            stage=spec.name,
            action=StageAction.SKIP_CACHED,
            detail=f"cached: output exists at {record.artifact_path}",
            artifact_path=Path(record.artifact_path),
        )
    if artifact.is_file():
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
) -> str:
    """Fully legible run-plan screen output (exempt from caveman compression)."""
    lines = [
        "=== dark-hunter_pop run plan ===",
        f"run_file: {run_path} ({'created' if created_new else 'existing'})",
        f"run_id: {manifest.run_id}",
        f"active_dr_mode: {config.active_dr_mode.value}",
        f"config_checksum: {manifest.config_checksum}",
        f"artifact_root: {manifest.artifact_root}",
        "stages:",
    ]
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
