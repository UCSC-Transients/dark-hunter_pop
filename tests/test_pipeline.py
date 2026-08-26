"""Tests for the main-program wiring (issue #79 / roster #16)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.pipeline import (
    STAGE_ORDER,
    STAGE_RUNNERS,
    apply_force_rerun,
    build_stage_plan,
    joint_orbit_plan_skip_reason,
    run_pipeline,
    validate_stage_runners,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    StageAction,
    TRIPLES_DISABLED_SKIP_REASON,
    create_run_manifest,
    mark_stage_finished,
    mark_stage_started,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.rv_consistency import JOINT_ORBIT_SKIP_REASON
from darkhunter_pop.schemas import RunManifest, StageRecord, StageStatus

pytestmark = [pytest.mark.unit, pytest.mark.api]


def test_stage_runners_cover_stage_order() -> None:
    assert validate_stage_runners() == []
    assert set(STAGE_RUNNERS) == set(STAGE_ORDER)


def test_build_stage_plan_includes_all_stages_and_triples_skip() -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    plan = build_stage_plan(manifest, cfg)
    assert [e.stage for e in plan] == list(STAGE_ORDER)
    triples = next(e for e in plan if e.stage == "triples")
    assert triples.action is StageAction.SKIP_REASON
    assert triples.detail == TRIPLES_DISABLED_SKIP_REASON


def test_joint_orbit_plan_skip_from_prior_record() -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    assert joint_orbit_plan_skip_reason(manifest) is None
    stages = dict(manifest.stages)
    stages["joint_orbit_fit"] = StageRecord(
        stage_name="joint_orbit_fit",
        status=StageStatus.SKIPPED,
        reason=JOINT_ORBIT_SKIP_REASON,
    )
    manifest = manifest.model_copy(update={"stages": stages})
    assert joint_orbit_plan_skip_reason(manifest) == JOINT_ORBIT_SKIP_REASON
    plan = build_stage_plan(manifest, cfg, stage_subset=["joint_orbit_fit"])
    assert plan[0].action is StageAction.SKIP_REASON
    assert plan[0].detail == JOINT_ORBIT_SKIP_REASON


def test_dry_run_prints_plan_without_writing_run_or_calling_runners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    runs = tmp_path / "runs"
    runs.mkdir()

    calls: list[str] = []

    def _spy(
        manifest: RunManifest,
        config: PipelineConfig,
        *,
        run_path: Path,
        force_rerun: bool = False,
    ) -> RunManifest:
        del config, run_path, force_rerun
        calls.append("ran")
        return manifest

    spies = {name: _spy for name in STAGE_ORDER}

    _manifest, path, text = run_pipeline(
        config=cfg,
        dry_run=True,
        runs=runs,
        runners=spies,
    )
    captured = capsys.readouterr()
    assert "=== dark-hunter_pop run plan ===" in text
    assert "=== dark-hunter_pop run plan ===" in captured.out
    assert "data_acquisition" in text
    assert "triples" in text
    assert TRIPLES_DISABLED_SKIP_REASON in text
    assert "running: output missing" in text
    assert calls == []
    assert not path.is_file()
    assert list(runs.glob("*.yaml")) == []


def test_dry_run_refuses_when_incomplete_without_run_file(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    incomplete = create_run_manifest(cfg)
    incomplete = incomplete.model_copy(
        update={
            "stages": {
                "data_acquisition": StageRecord(
                    stage_name="data_acquisition",
                    status=StageStatus.PENDING,
                )
            }
        }
    )
    save_run_manifest(incomplete, runs / f"{incomplete.run_id}.yaml")
    with pytest.raises(RuntimeError, match="incomplete runs"):
        run_pipeline(config=cfg, dry_run=True, runs=runs)


def test_execute_with_stub_runners_updates_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    runs = tmp_path / "runs"
    runs.mkdir()

    def _make_stub(stage_name: str):
        def runner(
            manifest: RunManifest,
            config: PipelineConfig,
            *,
            run_path: Path,
            force_rerun: bool = False,
        ) -> RunManifest:
            spec = STAGE_REGISTRY[stage_name]
            art = stage_artifact_path(config, spec, run_id=manifest.run_id)
            art.parent.mkdir(parents=True, exist_ok=True)
            art.write_bytes(b"stub")
            updated = mark_stage_started(
                manifest, spec, config, force_rerun=force_rerun
            )
            updated = mark_stage_finished(
                updated, spec, status=StageStatus.COMPLETED, artifact_path=art
            )
            save_run_manifest(updated, run_path)
            return updated

        return runner

    order = ["data_acquisition", "triples"]
    stubs = {name: _make_stub(name) for name in order}

    manifest, path, text = run_pipeline(
        config=cfg,
        stage_subset=order,
        runs=runs,
        runners=stubs,
        dry_run=False,
    )
    out = capsys.readouterr().out
    assert "=== dark-hunter_pop run plan ===" in text
    assert path.is_file()
    assert manifest.stages["data_acquisition"].status is StageStatus.COMPLETED
    assert manifest.stages["triples"].status is StageStatus.SKIPPED
    assert manifest.stages["triples"].reason == TRIPLES_DISABLED_SKIP_REASON
    assert "[stage] data_acquisition:" in out
    assert "[stage] triples: done (skipped)" in out


def test_force_rerun_completed_creates_new_run(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    runs = tmp_path / "runs"
    runs.mkdir()
    parent = create_run_manifest(
        cfg, when=datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    spec = STAGE_REGISTRY["data_acquisition"]
    parent = mark_stage_started(parent, spec, cfg)
    art = Path(parent.stages["data_acquisition"].artifact_path or "")
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"x")
    parent = mark_stage_finished(
        parent, spec, status=StageStatus.COMPLETED, artifact_path=art
    )
    parent_path = runs / f"{parent.run_id}.yaml"
    save_run_manifest(parent, parent_path)

    child, child_path, created = apply_force_rerun(
        parent, parent_path, cfg, ["data_acquisition"]
    )
    assert created is True
    assert child.parent_run_id == parent.run_id
    assert child_path != parent_path
    assert child_path.is_file()
    assert "data_acquisition" not in child.stages


def test_cli_dry_run_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_cli", cli_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline_cli"] = mod
    spec.loader.exec_module(mod)

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr("darkhunter_pop.pipeline.runs_dir", lambda: runs)
    monkeypatch.setattr("darkhunter_pop.run_management.runs_dir", lambda: runs)

    code = mod.main(["--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "=== dark-hunter_pop run plan ===" in out
    assert list(runs.glob("*.yaml")) == []
