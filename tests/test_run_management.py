"""Tests for run_management registry, hashing, amend/new-run, and purge."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

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
    new_run_for_force_rerun,
    plan_stage,
    purge_run,
    resolve_run_file,
    save_run_manifest,
    stage_artifact_path,
    validate_registry_inputs_from,
)
from darkhunter_pop.schemas import StageStatus

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
