"""Stage-runner caching-guard regressions (issue #172, follow-up to #167 / PR #170).

Every stage runner plans its own stage with ``plan_and_guard`` instead of a bare
``plan_stage`` call, so the four ``StageAction`` outcomes are handled in exactly
one place (``run_management.plan_and_guard``). These tests drive each registered
runner **directly** — not through ``pipeline.execute_plan``, whose upfront
``assert_plan_not_stale`` sweep would mask the defect — and assert:

* ``REFUSE_STALE`` raises ``StaleStageCacheError`` with no manifest mutation and
  no artifact write (a stale artifact is neither reused nor silently rebuilt,
  ARCHITECTURE.md §5);
* ``SKIP_REASON`` records ``skipped`` with the plan detail and runs no science;
* the ``--dry-run`` plan path still *reports* staleness without raising.

Parametrization walks ``STAGE_ORDER`` / ``pipeline.STAGE_RUNNERS`` rather than a
hand-written list, so a stage runner added later is covered automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from darkhunter_pop import pipeline, run_management
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.run_management import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    StageAction,
    StaleStageCacheError,
    create_run_manifest,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import RunManifest, StageRecord, StageStatus

pytestmark = pytest.mark.unit

STALE_SOURCE_HASH = "0" * 64
SKIP_SENTINEL = "test-injected skip reason"


def _explode_query(*args: Any, **kwargs: Any) -> Any:
    """Stand-in Gaia query proving ``data_acquisition`` never reached science."""
    raise AssertionError("stage science executed despite a non-RUN plan action")


# Per-stage keyword arguments that make a *fallthrough* fail loudly and offline
# instead of touching the Gaia archive.
EXTRA_RUNNER_KWARGS: Mapping[str, Mapping[str, Any]] = {
    "data_acquisition": {"query_fn": _explode_query},
}


def _config_for(stage: str, tmp_path: Path) -> PipelineConfig:
    """Load the canonical config, redirect artifacts to ``tmp_path``.

    ``triples`` is additionally enabled, because its config-driven
    ``SKIP_REASON`` outranks cache staleness in ``plan_stage`` and would hide the
    ``REFUSE_STALE`` path under test.
    """
    cfg = load_config()
    updates: dict[str, Any] = {
        "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})
    }
    if stage == "triples":
        updates["triples"] = cfg.triples.model_copy(update={"enabled": True})
    return cfg.model_copy(update=updates)


def _fresh_run(cfg: PipelineConfig, tmp_path: Path) -> tuple[RunManifest, Path]:
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)
    return manifest, run_path


@pytest.mark.parametrize("stage", list(STAGE_ORDER))
def test_every_stage_runner_refuses_stale_cache(stage: str, tmp_path: Path) -> None:
    """Regression for #172: no runner silently re-runs a ``REFUSE_STALE`` stage."""
    runner = pipeline.STAGE_RUNNERS[stage]
    cfg = _config_for(stage, tmp_path)
    spec = STAGE_REGISTRY[stage]
    manifest = create_run_manifest(cfg)

    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"hdf5-placeholder")

    stale_record = StageRecord(
        stage_name=stage,
        status=StageStatus.COMPLETED,
        source_hash=STALE_SOURCE_HASH,
        artifact_path=str(artifact),
    )
    manifest = manifest.model_copy(update={"stages": {stage: stale_record}})
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    # Confirm the fixture really reaches REFUSE_STALE before trusting the runner
    # assertion below (otherwise a SKIP_CACHED plan would pass vacuously).
    assert plan_stage(spec, manifest, cfg).action is StageAction.REFUSE_STALE

    with pytest.raises(StaleStageCacheError):
        runner(
            manifest,
            cfg,
            run_path=run_path,
            **EXTRA_RUNNER_KWARGS.get(stage, {}),
        )

    # Neither the run file nor the stale artifact may be touched.
    reloaded = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    assert reloaded["stages"][stage]["source_hash"] == STALE_SOURCE_HASH
    assert reloaded["stages"][stage]["status"] == "completed"
    assert artifact.read_bytes() == b"hdf5-placeholder"


@pytest.mark.parametrize("stage", list(STAGE_ORDER))
def test_every_stage_runner_honors_skip_reason(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for #172: a ``SKIP_REASON`` plan records ``skipped``, runs nothing.

    The reason is injected through ``stage_default_skip_reason`` because most
    stages have no config-driven skip of their own; the injected detail must
    reach the run file verbatim.
    """
    runner = pipeline.STAGE_RUNNERS[stage]
    cfg = _config_for(stage, tmp_path)
    spec = STAGE_REGISTRY[stage]

    def _fake_skip_reason(
        candidate_spec: run_management.StageSpec, config: PipelineConfig
    ) -> str | None:
        return SKIP_SENTINEL if candidate_spec.name == stage else None

    monkeypatch.setattr(
        run_management, "stage_default_skip_reason", _fake_skip_reason
    )

    manifest, run_path = _fresh_run(cfg, tmp_path)
    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)

    updated = runner(
        manifest,
        cfg,
        run_path=run_path,
        **EXTRA_RUNNER_KWARGS.get(stage, {}),
    )

    record = updated.stages[stage]
    assert record.status is StageStatus.SKIPPED
    assert record.reason == SKIP_SENTINEL
    assert not artifact.is_file()

    reloaded = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    assert reloaded["stages"][stage]["status"] == "skipped"
    assert reloaded["stages"][stage]["reason"] == SKIP_SENTINEL


def test_dry_run_plan_still_reports_stale_without_raising(tmp_path: Path) -> None:
    """``--dry-run`` must print a stale stage, not raise (#157 / #159 behavior)."""
    cfg = _config_for("data_acquisition", tmp_path)
    spec = STAGE_REGISTRY["data_acquisition"]
    manifest = create_run_manifest(cfg)
    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"hdf5-placeholder")
    manifest = manifest.model_copy(
        update={
            "stages": {
                spec.name: StageRecord(
                    stage_name=spec.name,
                    status=StageStatus.COMPLETED,
                    source_hash=STALE_SOURCE_HASH,
                    artifact_path=str(artifact),
                )
            }
        }
    )
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    _, _, plan_text = pipeline.run_pipeline(
        config=cfg, run_file=run_path, dry_run=True, runs=tmp_path
    )

    assert "stale cache" in plan_text
    plan = pipeline.build_stage_plan(manifest, cfg)
    assert plan[0].action is StageAction.REFUSE_STALE


def test_plan_and_guard_rejects_an_unhandled_action(tmp_path: Path) -> None:
    """A future ``StageAction`` must raise, never fall through to "run"."""

    class _FutureAction(str):
        pass

    cfg = _config_for("data_acquisition", tmp_path)
    spec = STAGE_REGISTRY["data_acquisition"]
    manifest, run_path = _fresh_run(cfg, tmp_path)

    def _fake_plan_stage(*args: Any, **kwargs: Any) -> run_management.StagePlanEntry:
        return run_management.StagePlanEntry(
            stage=spec.name,
            action=_FutureAction("future"),  # type: ignore[arg-type]
            detail="unknown action",
        )

    original = run_management.plan_stage
    run_management.plan_stage = _fake_plan_stage  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="unhandled StageAction"):
            run_management.plan_and_guard(
                spec, manifest, cfg, run_path=run_path
            )
    finally:
        run_management.plan_stage = original  # type: ignore[assignment]
