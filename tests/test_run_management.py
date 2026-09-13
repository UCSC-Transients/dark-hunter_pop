"""Tests for run_management registry, hashing, amend/new-run, and purge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import darkhunter_pop.run_management as run_management
from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.run_management import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    StageAction,
    assert_stage_source_hash,
    compute_source_hash,
    config_subset_for_stage,
    copy_stages_before,
    create_run_manifest,
    format_run_plan,
    list_incomplete_runs,
    mark_stage_finished,
    mark_stage_started,
    module_file_path,
    new_run_for_force_rerun,
    plan_stage,
    purge_run,
    resolve_run_file,
    save_run_manifest,
    stage_artifact_path,
    validate_registry_inputs_from,
)
from darkhunter_pop.schemas import StageRecord, StageStatus

pytestmark = pytest.mark.unit


def test_registry_complete_and_inputs_from_valid() -> None:
    assert validate_registry_inputs_from() == []
    assert "joint_orbit_fit" in STAGE_REGISTRY
    assert STAGE_REGISTRY["joint_orbit_fit"].module.endswith("rv_consistency")
    assert set(STAGE_ORDER) == set(STAGE_REGISTRY)


def test_source_hash_stable_and_stage_local(tmp_path: Path) -> None:
    spec = STAGE_REGISTRY["data_acquisition"]
    h1 = compute_source_hash(spec)
    h2 = compute_source_hash(spec)
    assert h1 == h2
    assert len(h1) == 64
    # Different stage → different dependency set / hash (almost surely).
    other = compute_source_hash(STAGE_REGISTRY["inference"])
    assert other != h1


def test_config_change_changes_artifact_path() -> None:
    cfg = load_config()
    spec = STAGE_REGISTRY["mass_derivation_bulk"]
    p1 = stage_artifact_path(cfg, spec, run_id="runA")
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_calibration.sigma_logM = 0.05
    p2 = stage_artifact_path(tweaked, spec, run_id="runA")
    assert p1 != p2
    assert p1.name != p2.name


def test_resolve_run_requires_flag_when_incomplete(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    # incomplete: has a pending stage
    from darkhunter_pop.schemas import StageRecord

    manifest = manifest.model_copy(
        update={
            "stages": {
                "data_acquisition": StageRecord(
                    stage_name="data_acquisition",
                    status=StageStatus.PENDING,
                )
            }
        }
    )
    path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, path)
    with pytest.raises(RuntimeError, match="incomplete runs"):
        resolve_run_file(run_file=None, config=cfg, runs=runs)


def test_resolve_creates_when_no_incomplete(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest, path, created = resolve_run_file(run_file=None, config=cfg, runs=runs)
    assert created is True
    assert path.is_file()
    assert manifest.config_checksum == config_checksum(cfg)


def test_cache_hit_and_force_rerun(tmp_path: Path) -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    spec = STAGE_REGISTRY["data_acquisition"]
    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"hdf5-placeholder")
    hit = plan_stage(spec, manifest, cfg)
    assert hit.action is StageAction.SKIP_CACHED
    forced = plan_stage(spec, manifest, cfg, force_rerun=True)
    assert forced.action is StageAction.RUN
    assert "force_rerun" in forced.detail


def test_force_rerun_copies_prior_stages(tmp_path: Path) -> None:
    cfg = load_config()
    parent = create_run_manifest(cfg)
    parent = mark_stage_started(
        parent, STAGE_REGISTRY["data_acquisition"], cfg
    )
    art = Path(parent.stages["data_acquisition"].artifact_path or "")
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"x")
    parent = mark_stage_finished(
        parent,
        STAGE_REGISTRY["data_acquisition"],
        status=StageStatus.COMPLETED,
        artifact_path=art,
    )
    parent = mark_stage_started(
        parent, STAGE_REGISTRY["mass_derivation_bulk"], cfg
    )
    parent = mark_stage_finished(
        parent,
        STAGE_REGISTRY["mass_derivation_bulk"],
        status=StageStatus.COMPLETED,
    )
    child = new_run_for_force_rerun(parent, cfg, "mass_derivation_bulk")
    assert child.parent_run_id == parent.run_id
    assert "data_acquisition" in child.stages
    assert "mass_derivation_bulk" not in child.stages
    assert copy_stages_before(parent, "mass_derivation_bulk").keys() == {
        "data_acquisition"
    }


def test_checksum_mismatch_refuses_resume(tmp_path: Path) -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, path)
    tweaked = cfg.model_copy(deep=True)
    tweaked.physics.mc_noise_threshold = 0.05
    with pytest.raises(ValueError, match="config checksum mismatch"):
        resolve_run_file(run_file=path, config=tweaked, runs=tmp_path)


def test_run_plan_text_is_legible() -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    plan = [
        plan_stage(STAGE_REGISTRY[name], manifest, cfg)
        for name in ("data_acquisition", "joint_orbit_fit")
    ]
    text = format_run_plan(
        manifest, cfg, plan, run_path=Path("runs/example.yaml"), created_new=True
    )
    assert "=== dark-hunter_pop run plan ===" in text
    assert "data_acquisition" in text
    assert "joint_orbit_fit" in text
    assert "config_subset" in text


def test_purge_refuses_completed_without_force(tmp_path: Path) -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    from darkhunter_pop.schemas import StageRecord

    manifest = manifest.model_copy(
        update={
            "stages": {
                "data_acquisition": StageRecord(
                    stage_name="data_acquisition",
                    status=StageStatus.COMPLETED,
                )
            }
        }
    )
    path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, path)
    art = tmp_path / "out.h5"
    art.write_bytes(b"data")
    manifest = manifest.model_copy(
        update={
            "stages": {
                "data_acquisition": StageRecord(
                    stage_name="data_acquisition",
                    status=StageStatus.COMPLETED,
                    artifact_path=str(art),
                )
            }
        }
    )
    save_run_manifest(manifest, path)
    with pytest.raises(ValueError, match="refusing to purge"):
        purge_run(path)
    purge_run(path, with_artifacts=True, force=True)
    assert not path.exists()
    assert not art.exists()


def test_assert_stage_source_hash_mismatch() -> None:
    spec = STAGE_REGISTRY["diagnostics"]
    current = compute_source_hash(spec)
    assert_stage_source_hash(spec, current, require_recorded=True)
    with pytest.raises(ValueError, match="source_hash mismatch"):
        assert_stage_source_hash(spec, "0" * 64, require_recorded=True)


def test_joint_orbit_fit_skip_reason() -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    entry = plan_stage(
        STAGE_REGISTRY["joint_orbit_fit"],
        manifest,
        cfg,
        skip_reason="rv_astrometry_gate_failed",
    )
    assert entry.action is StageAction.SKIP_REASON
    assert entry.detail == "rv_astrometry_gate_failed"


def test_triples_default_skip_in_plan_stage() -> None:
    """Off-by-default triples: plan_stage skips without an explicit skip_reason."""
    from darkhunter_pop.run_management import TRIPLES_DISABLED_SKIP_REASON

    cfg = load_config()
    manifest = create_run_manifest(cfg)
    entry = plan_stage(STAGE_REGISTRY["triples"], manifest, cfg)
    assert entry.action is StageAction.SKIP_REASON
    assert entry.detail == TRIPLES_DISABLED_SKIP_REASON


def _completed_record(
    stage_name: str,
    *,
    artifact: Path,
    source_hash: str,
) -> StageRecord:
    """Build a ``COMPLETED`` stage record pointing at ``artifact``."""
    return StageRecord(
        stage_name=stage_name,
        status=StageStatus.COMPLETED,
        source_hash=source_hash,
        artifact_path=str(artifact),
    )


def test_stale_source_hash_is_not_treated_as_cached(tmp_path: Path) -> None:
    """Regression for #157.

    A ``completed`` record whose artifact exists on disk but whose recorded
    ``source_hash`` predates a dependency-module change must NOT plan as
    ``SKIP_CACHED``. It refuses (``REFUSE_STALE``), and ``--force-rerun`` is
    the documented escape hatch.
    """
    cfg = load_config()
    cfg.paths.artifact_root = str(tmp_path / "artifacts")
    manifest = create_run_manifest(cfg)
    spec = STAGE_REGISTRY["sample_selection"]
    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"hdf5-placeholder")

    current = compute_source_hash(spec)
    fresh = manifest.model_copy(
        update={
            "stages": {
                spec.name: _completed_record(
                    spec.name, artifact=artifact, source_hash=current
                )
            }
        }
    )
    assert plan_stage(spec, fresh, cfg).action is StageAction.SKIP_CACHED

    stale = manifest.model_copy(
        update={
            "stages": {
                spec.name: _completed_record(
                    spec.name, artifact=artifact, source_hash="0" * 64
                )
            }
        }
    )
    entry = plan_stage(spec, stale, cfg)
    assert entry.action is StageAction.REFUSE_STALE
    assert entry.action is not StageAction.SKIP_CACHED
    assert "source_hash" in entry.detail
    assert "--force-rerun" in entry.detail
    with pytest.raises(run_management.StaleStageCacheError, match="stale cached"):
        run_management.assert_plan_not_stale(entry)

    # Explicit force-rerun is the documented way through.
    forced = plan_stage(spec, stale, cfg, force_rerun=True)
    assert forced.action is StageAction.RUN
    run_management.assert_plan_not_stale(forced)


def test_stale_artifact_fingerprint_is_not_treated_as_cached(tmp_path: Path) -> None:
    """Regression for #157: a config change that re-fingerprints the artifact
    must refuse rather than reuse the record's old artifact."""
    cfg = load_config()
    cfg.paths.artifact_root = str(tmp_path / "artifacts")
    manifest = create_run_manifest(cfg)
    spec = STAGE_REGISTRY["mass_derivation_bulk"]
    old_artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    old_artifact.parent.mkdir(parents=True, exist_ok=True)
    old_artifact.write_bytes(b"hdf5-placeholder")
    recorded = manifest.model_copy(
        update={
            "stages": {
                spec.name: _completed_record(
                    spec.name,
                    artifact=old_artifact,
                    source_hash=compute_source_hash(spec),
                )
            }
        }
    )
    assert plan_stage(spec, recorded, cfg).action is StageAction.SKIP_CACHED

    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_calibration.sigma_logM = 0.05
    assert (
        stage_artifact_path(tweaked, spec, run_id=manifest.run_id).name
        != old_artifact.name
    )
    entry = plan_stage(spec, recorded, tweaked)
    assert entry.action is StageAction.REFUSE_STALE
    assert "artifact fingerprint" in entry.detail


def test_copied_forward_record_from_parent_run_is_still_cached(
    tmp_path: Path,
) -> None:
    """A force-rerun child run copies prior records forward pointing into the
    PARENT run's artifact directory. Only the fingerprint (file name) is
    compared, so that must not read as stale (#157)."""
    cfg = load_config()
    cfg.paths.artifact_root = str(tmp_path / "artifacts")
    parent = create_run_manifest(cfg)
    spec = STAGE_REGISTRY["data_acquisition"]
    parent_artifact = stage_artifact_path(cfg, spec, run_id=parent.run_id)
    parent_artifact.parent.mkdir(parents=True, exist_ok=True)
    parent_artifact.write_bytes(b"hdf5-placeholder")
    parent = parent.model_copy(
        update={
            "stages": {
                spec.name: _completed_record(
                    spec.name,
                    artifact=parent_artifact,
                    source_hash=compute_source_hash(spec),
                )
            }
        }
    )
    child = create_run_manifest(
        cfg,
        parent_run_id=parent.run_id,
        stages_seed=copy_stages_before(parent, "mass_derivation_bulk"),
        when=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert child.run_id != parent.run_id
    # The copied-forward record still points into the parent's artifact dir.
    recorded = child.stages[spec.name].artifact_path or ""
    assert parent.run_id in recorded
    assert child.run_id not in recorded
    entry = plan_stage(spec, child, cfg)
    assert entry.action is StageAction.SKIP_CACHED


def test_execute_plan_refuses_stale_before_running_anything(tmp_path: Path) -> None:
    """#157: ``execute_plan`` must raise on a stale entry before any runner
    executes — including a stale stage late in ``STAGE_ORDER``."""
    from darkhunter_pop.pipeline import execute_plan
    from darkhunter_pop.run_management import StagePlanEntry

    cfg = load_config()
    manifest = create_run_manifest(cfg)
    called: list[str] = []

    def _runner(current, config, *, run_path, force_rerun):  # type: ignore[no-untyped-def]
        called.append("ran")
        return current

    plan = [
        StagePlanEntry(
            stage="data_acquisition",
            action=StageAction.RUN,
            detail="running: output missing",
        ),
        StagePlanEntry(
            stage="sample_selection",
            action=StageAction.REFUSE_STALE,
            detail="stale cache: source_hash recorded=0 current=1",
        ),
    ]
    with pytest.raises(run_management.StaleStageCacheError):
        execute_plan(
            manifest,
            cfg,
            plan,
            run_path=tmp_path / "run.yaml",
            runners={"data_acquisition": _runner},
        )
    assert called == []


def test_list_incomplete_sorted_by_run_id_not_mtime(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    older = create_run_manifest(
        cfg, when=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    newer = create_run_manifest(
        cfg, when=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    from darkhunter_pop.schemas import StageRecord

    for m in (older, newer):
        m2 = m.model_copy(
            update={
                "stages": {
                    "data_acquisition": StageRecord(
                        stage_name="data_acquisition",
                        status=StageStatus.RUNNING,
                    )
                }
            }
        )
        save_run_manifest(m2, runs / f"{m2.run_id}.yaml")
    incomplete = list_incomplete_runs(runs)
    assert incomplete[0].run_id == newer.run_id


def test_sample_selection_dependency_modules_cover_actual_imports() -> None:
    """Regression for #151/#158: sample_selection.py's code path pulls in
    elbadry2026_m2_sigma (Q12 sigma_m2_astrometric_msun derivation),
    elbadry2026_selection + janssens_mass (enrich_elbadry2026_row), and
    mc_mass_function (load_selection_rows_from_uncut_snapshot(attach_mc=True)
    / _attach_elbadry_m2_sigma_inplace). Per #158, the transitive imports
    those modules pull in are also declared: physics_utils (imported by
    elbadry2026_selection.py and mc_mass_function.py for the AMRF/a0/M-tilde-2
    chain), sensitivity_analysis (imported by mc_mass_function.py), and
    data_acquisition (imported directly by sample_selection.py). All eight
    must feed source_hash.

    elbadry2024_selection is deliberately excluded: sample_selection.py never
    imports it (only sample_diagnostics.py does, for a different stage).
    """
    spec = STAGE_REGISTRY["sample_selection"]
    assert set(spec.dependency_modules) == {
        "darkhunter_pop.sample_selection",
        "darkhunter_pop.elbadry2026_m2_sigma",
        "darkhunter_pop.janssens_mass",
        "darkhunter_pop.elbadry2026_selection",
        "darkhunter_pop.mc_mass_function",
        "darkhunter_pop.physics_utils",
        "darkhunter_pop.sensitivity_analysis",
        "darkhunter_pop.data_acquisition",
    }
    assert "darkhunter_pop.elbadry2024_selection" not in spec.dependency_modules
    # Every declared dependency must actually resolve to a source file.
    for module_name in spec.dependency_modules:
        assert module_file_path(module_name).is_file()


def test_sample_selection_config_fingerprint_covers_mc_mass_function() -> None:
    """Regression for #151: mc_mass_function.{n_draws,random_seed,eig_*_floor}
    change the science output (draw count, seed, covariance-eigenvalue floors
    used by the fixed-M1-tilde MC) and the parent-cache directory name
    (``+enrich+mc{n_draws}``), so both must move the stage's config
    fingerprint / artifact path.
    """
    spec = STAGE_REGISTRY["sample_selection"]
    assert "mc_mass_function" in spec.config_fingerprint_keys

    cfg = load_config()
    base_subset = config_subset_for_stage(cfg, spec)

    seed_tweaked = cfg.model_copy(deep=True)
    seed_tweaked.mc_mass_function.random_seed += 1
    assert config_subset_for_stage(seed_tweaked, spec) != base_subset

    draws_tweaked = cfg.model_copy(deep=True)
    draws_tweaked.mc_mass_function.n_draws += 1
    assert config_subset_for_stage(draws_tweaked, spec) != base_subset

    run_id = "runA"
    p_base = stage_artifact_path(cfg, spec, run_id=run_id)
    p_seed = stage_artifact_path(seed_tweaked, spec, run_id=run_id)
    p_draws = stage_artifact_path(draws_tweaked, spec, run_id=run_id)
    assert p_base != p_seed
    assert p_base != p_draws


def test_sample_selection_source_hash_sensitive_to_elbadry2026_m2_sigma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for #151: an edit to elbadry2026_m2_sigma.py must move the
    stage's source_hash, since sample_selection._attach_elbadry_m2_sigma_inplace
    imports ``sigma_m2_tilde_astrometric_msun`` from it. Simulated via a
    shadow copy of the real file redirected through ``module_file_path`` so no
    real source file is touched (per #151's "Do not touch sample_selection.py's
    science logic" and the general no-edit-to-prove-it constraint).
    """
    spec = STAGE_REGISTRY["sample_selection"]
    target = "darkhunter_pop.elbadry2026_m2_sigma"
    assert target in spec.dependency_modules

    real_resolve = run_management.module_file_path
    real_path = real_resolve(target)
    shadow = tmp_path / "elbadry2026_m2_sigma.py"
    shadow.write_bytes(real_path.read_bytes())

    def _resolve(module_name: str) -> Path:
        if module_name == target:
            return shadow
        return real_resolve(module_name)

    monkeypatch.setattr(run_management, "module_file_path", _resolve)

    h_before = compute_source_hash(spec)
    shadow.write_bytes(real_path.read_bytes() + b"\n# regression-test edit\n")
    h_after = compute_source_hash(spec)
    assert h_before != h_after
