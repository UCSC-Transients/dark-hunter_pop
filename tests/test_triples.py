"""Tests for the off-by-default ``triples`` stub stage (issue #49)."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import yaml

from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    StaleStageCacheError,
    StageAction,
    TRIPLES_DISABLED_SKIP_REASON,
    create_run_manifest,
    format_run_plan,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
    stage_default_skip_reason,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    ParameterSet,
    StageRecord,
    StageStatus,
    TessBlock,
)
from darkhunter_pop.triples import (
    evaluate_candidate_triple_stub,
    read_triples_artifact,
    run_triples_stage,
    run_triples_stub,
    write_triples_artifact,
)
from darkhunter_pop.triples.rotation_check import evaluate_rotation_consistency
from darkhunter_pop.triples.tess_variability import evaluate_tess_variability

pytestmark = pytest.mark.unit


def test_config_defaults_triples_off() -> None:
    cfg = load_config()
    assert cfg.triples.enabled is False
    assert cfg.triples.tess_variability_channel is True
    assert cfg.triples.rotation_consistency_channel is True


def test_checksum_includes_triples() -> None:
    assert "triples" in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    altered.triples.enabled = True
    assert config_checksum(altered) != base


def test_registry_fingerprints_triples_config() -> None:
    spec = STAGE_REGISTRY["triples"]
    assert spec.module.endswith("triples")
    assert spec.inputs_from == ("companion_nature_likelihood",)
    assert "triples" in spec.config_fingerprint_keys
    assert "darkhunter_pop.triples.tess_variability" in spec.dependency_modules
    assert "darkhunter_pop.triples.rotation_check" in spec.dependency_modules


def test_plan_stage_skips_when_disabled() -> None:
    cfg = load_config()
    assert cfg.triples.enabled is False
    manifest = create_run_manifest(cfg)
    entry = plan_stage(STAGE_REGISTRY["triples"], manifest, cfg)
    assert entry.action is StageAction.SKIP_REASON
    assert entry.detail == TRIPLES_DISABLED_SKIP_REASON
    assert (
        stage_default_skip_reason(STAGE_REGISTRY["triples"], cfg)
        == TRIPLES_DISABLED_SKIP_REASON
    )


def test_plan_stage_runs_when_enabled() -> None:
    cfg = load_config()
    enabled = cfg.model_copy(deep=True)
    enabled.triples.enabled = True
    manifest = create_run_manifest(enabled)
    entry = plan_stage(STAGE_REGISTRY["triples"], manifest, enabled)
    assert entry.action is StageAction.RUN
    assert stage_default_skip_reason(STAGE_REGISTRY["triples"], enabled) is None


def test_run_plan_shows_triples_skip() -> None:
    cfg = load_config()
    manifest = create_run_manifest(cfg)
    plan = [plan_stage(STAGE_REGISTRY["triples"], manifest, cfg)]
    text = format_run_plan(
        manifest, cfg, plan, run_path=Path("runs/example.yaml"), created_new=True
    )
    assert "triples" in text
    assert TRIPLES_DISABLED_SKIP_REASON in text


def test_run_triples_stage_default_is_clean_skip(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    updated = run_triples_stage(manifest, cfg, run_path=run_path)
    record = updated.stages["triples"]
    assert record.status is StageStatus.SKIPPED
    assert record.reason == TRIPLES_DISABLED_SKIP_REASON
    # Planned path may be recorded, but no science artifact is written.
    art_dir = tmp_path / manifest.run_id / "triples"
    if art_dir.exists():
        assert list(art_dir.glob("*.h5")) == []
    if record.artifact_path:
        assert not Path(record.artifact_path).is_file()


def test_run_triples_stub_refuses_when_disabled() -> None:
    cfg = load_config()
    with pytest.raises(ValueError, match="triples.enabled=true"):
        run_triples_stub([], cfg)


def test_enable_path_writes_stub_artifact(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "triples": cfg.triples.model_copy(update={"enabled": True}),
        }
    )
    candidate = CandidateRecord(
        source_id=42,
        tess=TessBlock(
            period_day=2.5,
            amplitude=0.01,
            variability_flag=True,
            implied_v_rot_kms=15.0,
        ),
        m1=ParameterSet(
            names=["M1", "v_sin_i"],
            values=[1.0, 12.0],
            covariance=[[0.01, 0.0], [0.0, 1.0]],
            provenance="uberMS",
            units=["Msun", "km/s"],
        ),
    )
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    updated = run_triples_stage(
        manifest, cfg, run_path=run_path, candidates=[candidate]
    )
    record = updated.stages["triples"]
    assert record.status is StageStatus.COMPLETED
    assert record.artifact_path is not None
    path = Path(record.artifact_path)
    assert path.is_file()
    with h5py.File(path, "r") as handle:
        assert handle.attrs["stage"] == "triples"
        assert bool(handle.attrs["stub"]) is True
        assert int(handle.attrs["n_candidates"]) == 1
    payload = read_triples_artifact(path)
    assert payload["enabled"] is True
    assert payload["evidence"][0]["p_triple"] is None
    assert payload["evidence"][0]["tess"]["available"] is True
    assert payload["evidence"][0]["rotation"]["available"] is True
    assert payload["evidence"][0]["rotation"]["consistent"] is None


def test_run_triples_stage_refuses_stale_cache_instead_of_silently_rerunning(
    tmp_path: Path,
) -> None:
    """Regression for #167.

    ``run_triples_stage`` calls ``plan_stage`` directly (not via
    ``pipeline.execute_plan``), so it must itself refuse a ``REFUSE_STALE``
    plan entry rather than falling through its ``SKIP_CACHED`` / ``SKIP_REASON``
    checks into re-running the stage. Before the fix, a stale ``source_hash``
    fell straight into the enable-path run branch and silently
    re-ran/overwrote the artifact; the stage runner must instead raise
    ``StaleStageCacheError`` and must not mutate the run manifest (the stale
    record stays exactly as recorded).
    """
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "triples": cfg.triples.model_copy(update={"enabled": True}),
        }
    )
    manifest = create_run_manifest(cfg)
    spec = STAGE_REGISTRY["triples"]
    artifact = stage_artifact_path(cfg, spec, run_id=manifest.run_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"hdf5-placeholder")

    stale_record = StageRecord(
        stage_name=spec.name,
        status=StageStatus.COMPLETED,
        source_hash="0" * 64,
        artifact_path=str(artifact),
    )
    manifest = manifest.model_copy(update={"stages": {spec.name: stale_record}})
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    # Confirm the fixture actually reaches REFUSE_STALE before trusting the
    # stage-runner assertion below.
    assert plan_stage(spec, manifest, cfg).action is StageAction.REFUSE_STALE

    candidate = CandidateRecord(
        source_id=42,
        tess=TessBlock(
            period_day=2.5,
            amplitude=0.01,
            variability_flag=True,
            implied_v_rot_kms=15.0,
        ),
        m1=ParameterSet(
            names=["M1", "v_sin_i"],
            values=[1.0, 12.0],
            covariance=[[0.01, 0.0], [0.0, 1.0]],
            provenance="uberMS",
            units=["Msun", "km/s"],
        ),
    )

    with pytest.raises(StaleStageCacheError):
        run_triples_stage(
            manifest, cfg, run_path=run_path, candidates=[candidate]
        )

    # The manifest on disk must be untouched: still the stale COMPLETED record,
    # never overwritten by a silent re-run.
    reloaded = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    assert reloaded["stages"][spec.name]["source_hash"] == "0" * 64
    assert reloaded["stages"][spec.name]["status"] == "completed"


def test_evidence_channel_hooks_are_stubs() -> None:
    cfg = load_config()
    empty = CandidateRecord(source_id=1)
    tess = evaluate_tess_variability(empty, cfg)
    assert tess.available is False
    rot = evaluate_rotation_consistency(empty, cfg)
    assert rot.available is False
    assert rot.consistent is None

    with_data = CandidateRecord(
        source_id=2,
        tess=TessBlock(implied_v_rot_kms=10.0),
        extras={"v_sin_i_kms": 8.0},
    )
    stub = evaluate_candidate_triple_stub(with_data, cfg)
    assert stub.p_triple is None
    assert stub.rotation.available is True
    assert stub.rotation.consistent is None


def test_write_read_artifact_round_trip(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.triples.enabled = True
    result = run_triples_stub([CandidateRecord(source_id=7)], cfg)
    path = tmp_path / "triples.h5"
    write_triples_artifact(path, result)
    loaded = read_triples_artifact(path)
    assert loaded["n_candidates"] == 1
    assert loaded["evidence"][0]["source_id"] == 7
