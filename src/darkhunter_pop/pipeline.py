"""Main-program orchestration: plan + execute ``STAGE_ORDER`` via stage runners.

Roster #16 / issue #79. Science stays in per-stage modules; this module only resolves
run files, prints the run plan, and dispatches. ARCHITECTURE.md §5.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from darkhunter_pop.companion_nature import run_companion_nature_likelihood
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.data_acquisition import run_data_acquisition
from darkhunter_pop.diagnostics import run_diagnostics_stage
from darkhunter_pop.forward_model import (
    run_selection_function_astrometric,
    run_selection_function_followup,
)
from darkhunter_pop.inference import run_inference_stage
from darkhunter_pop.mass_derivation import (
    run_mass_derivation_bulk,
    run_mass_derivation_refined,
)
from darkhunter_pop.population_model import run_population_model_stage
from darkhunter_pop.run_management import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    StageAction,
    StagePlanEntry,
    assert_stage_source_hash,
    create_run_manifest,
    format_incomplete_runs_table,
    format_run_plan,
    list_incomplete_runs,
    mark_stage_finished,
    mark_stage_started,
    new_run_for_force_rerun,
    plan_stage,
    resolve_run_file,
    run_file_path,
    runs_dir,
    save_run_manifest,
    stage_artifact_path,
    wipe_stage_artifacts,
)
from darkhunter_pop.rv_consistency import (
    JOINT_ORBIT_SKIP_REASON,
    run_joint_orbit_fit,
    run_rv_astrometry_gate,
)
from darkhunter_pop.sample_selection import run_sample_selection_stage
from darkhunter_pop.schemas import RunManifest, StageStatus
from darkhunter_pop.sensitivity_analysis import run_sensitivity_analysis_stage
from darkhunter_pop.triples import run_triples_stage

# Callable signature shared by every registered stage runner.
StageRunner = Callable[..., RunManifest]


def _artifact_from_stage(manifest: RunManifest, stage_name: str) -> Path | None:
    rec = manifest.stages.get(stage_name)
    if rec is None or not rec.artifact_path:
        return None
    path = Path(rec.artifact_path)
    return path if path.is_file() else None


def _run_selection_function_astrometric_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
) -> RunManifest:
    """Wire ``run_selection_function_astrometric`` into the run-file protocol."""
    spec = STAGE_REGISTRY["selection_function_astrometric"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action is StageAction.SKIP_CACHED:
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    da = _artifact_from_stage(manifest, "data_acquisition")
    run_selection_function_astrometric(
        config, artifact, data_acquisition_artifact=da
    )

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest


def _run_selection_function_followup_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
) -> RunManifest:
    """Wire ``run_selection_function_followup`` into the run-file protocol."""
    spec = STAGE_REGISTRY["selection_function_followup"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action is StageAction.SKIP_CACHED:
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    astro = _artifact_from_stage(manifest, "selection_function_astrometric")
    run_selection_function_followup(config, artifact, astrometric_artifact=astro)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest


STAGE_RUNNERS: Mapping[str, StageRunner] = {
    "data_acquisition": run_data_acquisition,
    "mass_derivation_bulk": run_mass_derivation_bulk,
    "sample_selection": run_sample_selection_stage,
    "mass_derivation_refined": run_mass_derivation_refined,
    "rv_astrometry_gate": run_rv_astrometry_gate,
    "joint_orbit_fit": run_joint_orbit_fit,
    "companion_nature_likelihood": run_companion_nature_likelihood,
    "triples": run_triples_stage,
    "selection_function_astrometric": _run_selection_function_astrometric_stage,
    "selection_function_followup": _run_selection_function_followup_stage,
    "population_model": run_population_model_stage,
    "sensitivity_analysis": run_sensitivity_analysis_stage,
    "inference": run_inference_stage,
    "diagnostics": run_diagnostics_stage,
}


def joint_orbit_plan_skip_reason(manifest: RunManifest) -> str | None:
    """Return ``rv_astrometry_gate_failed`` when a prior joint-stage skip is recorded.

    Live all-fail detection after the gate runs is owned by ``run_joint_orbit_fit``.
    This helper covers resume/amend when the skip is already on the run file.
    """
    prior = manifest.stages.get("joint_orbit_fit")
    if (
        prior is not None
        and prior.status is StageStatus.SKIPPED
        and prior.reason == JOINT_ORBIT_SKIP_REASON
    ):
        return JOINT_ORBIT_SKIP_REASON
    return None


def build_stage_plan(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    force_rerun_stages: Sequence[str] = (),
    stage_subset: Sequence[str] | None = None,
) -> list[StagePlanEntry]:
    """Build the full run plan for every stage in ``STAGE_ORDER`` (or a subset)."""
    forced = set(force_rerun_stages)
    names = list(STAGE_ORDER) if stage_subset is None else list(stage_subset)
    unknown = [n for n in names if n not in STAGE_REGISTRY]
    if unknown:
        raise KeyError(f"unknown stage(s): {unknown}")

    plan: list[StagePlanEntry] = []
    for name in names:
        skip_reason: str | None = None
        if name == "joint_orbit_fit":
            skip_reason = joint_orbit_plan_skip_reason(manifest)
        plan.append(
            plan_stage(
                STAGE_REGISTRY[name],
                manifest,
                config,
                force_rerun=name in forced,
                skip_reason=skip_reason,
            )
        )
    return plan


def apply_force_rerun(
    manifest: RunManifest,
    run_path: Path,
    config: PipelineConfig,
    force_rerun_stages: Sequence[str],
) -> tuple[RunManifest, Path, bool]:
    """Apply amend vs new-run rules for ``--force-rerun`` (ARCHITECTURE.md §5).

    Force-re-run of a **completed/cached** stage → new run file with prior stages
    copied. Mid-stage crash / incomplete → wipe partials and amend the same file.
    When multiple stages are listed, the earliest in ``STAGE_ORDER`` sets the
    new-run boundary.
    """
    if not force_rerun_stages:
        return manifest, run_path, False

    unknown = [n for n in force_rerun_stages if n not in STAGE_REGISTRY]
    if unknown:
        raise KeyError(f"unknown force-rerun stage(s): {unknown}")

    earliest = min(force_rerun_stages, key=lambda n: STAGE_ORDER.index(n))
    rec = manifest.stages.get(earliest)

    if rec is not None and rec.status in {StageStatus.COMPLETED, StageStatus.CACHED}:
        child = new_run_for_force_rerun(manifest, config, earliest)
        child_path = run_file_path(child.run_id, run_path.parent)
        save_run_manifest(child, child_path)
        return child, child_path, True

    # Mid-crash / pending / running: wipe partials and amend.
    wipe_stage_artifacts(manifest, earliest)
    if rec is not None and rec.status is StageStatus.RUNNING:
        stages = dict(manifest.stages)
        stages[earliest] = rec.model_copy(
            update={"status": StageStatus.PENDING, "finished_at": None, "reason": None}
        )
        manifest = manifest.model_copy(update={"stages": stages})
        save_run_manifest(manifest, run_path)
    return manifest, run_path, False


def resolve_pipeline_run(
    config: PipelineConfig,
    *,
    run_file: Path | None,
    force_rerun_stages: Sequence[str] = (),
    runs: Path | None = None,
    dry_run: bool = False,
) -> tuple[RunManifest, Path, bool]:
    """Select / create a run, then apply force-rerun amend/new-run rules.

    On ``dry_run`` with no ``--run-file`` and no incomplete runs, build an
    in-memory manifest without writing a new YAML under ``runs/``.
    """
    root = runs if runs is not None else runs_dir()

    if dry_run and run_file is None:
        incomplete = list_incomplete_runs(root)
        if incomplete:
            table = format_incomplete_runs_table(incomplete)
            raise RuntimeError(
                "one or more incomplete runs exist; pass --run-file explicitly.\n"
                + table
            )
        manifest = create_run_manifest(config)
        path = run_file_path(manifest.run_id, root)
        created = True
    else:
        manifest, path, created = resolve_run_file(
            run_file=run_file, config=config, runs=root
        )

    if force_rerun_stages:
        manifest, path, force_created = apply_force_rerun(
            manifest, path, config, force_rerun_stages
        )
        created = created or force_created

    return manifest, path, created


def _check_source_hash_before_run(
    manifest: RunManifest, stage_name: str
) -> None:
    """Refuse resume when this stage's recorded source_hash no longer matches."""
    spec = STAGE_REGISTRY[stage_name]
    rec = manifest.stages.get(stage_name)
    recorded = rec.source_hash if rec is not None else None
    require = bool(recorded)
    assert_stage_source_hash(spec, recorded, require_recorded=require)


def execute_plan(
    manifest: RunManifest,
    config: PipelineConfig,
    plan: Sequence[StagePlanEntry],
    *,
    run_path: Path,
    force_rerun_stages: Sequence[str] = (),
    runners: Mapping[str, StageRunner] | None = None,
) -> RunManifest:
    """Execute planned stages; skip cached / skip-reason entries without science work.

    Per-stage start/end lines are printed to stdout (ARCHITECTURE.md §5).
    """
    registry = runners if runners is not None else STAGE_RUNNERS
    forced = set(force_rerun_stages)
    current = manifest

    for entry in plan:
        print(f"[stage] {entry.stage}: {entry.detail}", flush=True)

        if entry.action is StageAction.SKIP_CACHED:
            print(f"[stage] {entry.stage}: done (cached)", flush=True)
            continue

        if entry.action is StageAction.SKIP_REASON:
            spec = STAGE_REGISTRY[entry.stage]
            current = mark_stage_started(
                current, spec, config, force_rerun=entry.stage in forced
            )
            save_run_manifest(current, run_path)
            current = mark_stage_finished(
                current,
                spec,
                status=StageStatus.SKIPPED,
                reason=entry.detail,
                artifact_path=None,
            )
            save_run_manifest(current, run_path)
            print(f"[stage] {entry.stage}: done (skipped)", flush=True)
            continue

        # RUN
        runner = registry.get(entry.stage)
        if runner is None:
            raise KeyError(f"no stage runner registered for {entry.stage!r}")
        _check_source_hash_before_run(current, entry.stage)
        current = runner(
            current,
            config,
            run_path=run_path,
            force_rerun=entry.stage in forced,
        )
        print(f"[stage] {entry.stage}: done", flush=True)

    return current


def run_pipeline(
    *,
    config: PipelineConfig | None = None,
    config_path: Path | None = None,
    run_file: Path | None = None,
    force_rerun_stages: Sequence[str] = (),
    stage_subset: Sequence[str] | None = None,
    dry_run: bool = False,
    runs: Path | None = None,
    runners: Mapping[str, StageRunner] | None = None,
) -> tuple[RunManifest, Path, str]:
    """Load config, resolve run, print plan, optionally execute.

    Returns ``(manifest, run_path, plan_text)``. On ``dry_run``, stages are not
    executed and a new run YAML is not written when ``run_file`` is omitted.
    """
    cfg = config if config is not None else load_config(config_path)
    manifest, path, created = resolve_pipeline_run(
        cfg,
        run_file=run_file,
        force_rerun_stages=force_rerun_stages,
        runs=runs,
        dry_run=dry_run,
    )
    plan = build_stage_plan(
        manifest,
        cfg,
        force_rerun_stages=force_rerun_stages,
        stage_subset=stage_subset,
    )
    plan_text = format_run_plan(
        manifest, cfg, plan, run_path=path, created_new=created, dry_run=dry_run
    )
    print(plan_text, flush=True)

    if dry_run:
        return manifest, path, plan_text

    # Persist new dry-skipped creates when actually executing.
    if created and not path.is_file():
        save_run_manifest(manifest, path)

    updated = execute_plan(
        manifest,
        cfg,
        plan,
        run_path=path,
        force_rerun_stages=force_rerun_stages,
        runners=runners,
    )
    return updated, path, plan_text


def validate_stage_runners() -> list[str]:
    """Return errors if ``STAGE_ORDER`` lacks a runner mapping."""
    errors: list[str] = []
    for name in STAGE_ORDER:
        if name not in STAGE_RUNNERS:
            errors.append(f"STAGE_ORDER entry {name!r} has no STAGE_RUNNERS mapping")
    extra = [n for n in STAGE_RUNNERS if n not in STAGE_ORDER]
    for name in extra:
        errors.append(f"STAGE_RUNNERS has extra stage {name!r} not in STAGE_ORDER")
    return errors
