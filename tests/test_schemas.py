"""Tests for Foundation schemas (ParameterSet, CandidateRecord, RunManifest, …)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from darkhunter_pop.schemas import (
    SAMPLES_HDF5_FORMAT,
    ActiveDRMode,
    CandidateRecord,
    FitTier,
    FollowUpRecord,
    MarginalView,
    OrbitTier,
    OutlierTestResult,
    ParameterSet,
    PhotometryPoint,
    RunManifest,
    ScalarEstimate,
    StageRecord,
    StageStatus,
    TessBlock,
    ThieleInnesElements,
)

pytestmark = pytest.mark.unit


def _toy_parameterset() -> ParameterSet:
    # Correlated 2-vector with known diagonal.
    cov = [[0.04, 0.01], [0.01, 0.09]]
    return ParameterSet(
        names=["logM", "logR"],
        values=[0.0, 0.1],
        covariance=cov,
        provenance="TAG10",
        units=["dex", "dex"],
    )


def test_parameterset_round_trip_json() -> None:
    original = _toy_parameterset()
    restored = ParameterSet.model_validate(original.model_dump())
    assert restored.names == original.names
    assert restored.values == original.values
    assert restored.covariance == original.covariance
    assert restored.provenance == "TAG10"


def test_parameterset_marginal_uses_covariance_diag() -> None:
    ps = _toy_parameterset()
    m = ps.marginal("logM")
    assert isinstance(m, MarginalView)
    assert m.value == pytest.approx(0.0)
    assert m.sigma == pytest.approx(0.2)
    assert m.unit == "dex"
    r = ps.marginal("logR")
    assert r.sigma == pytest.approx(0.3)


def test_parameterset_provenance_does_not_change_marginal() -> None:
    a = _toy_parameterset()
    b = a.model_copy(update={"provenance": "gspphot-fallback"})
    assert a.marginal("logM").sigma == b.marginal("logM").sigma
    assert a.marginal("logM").value == b.marginal("logM").value
    assert a.provenance != b.provenance


def test_parameterset_rejects_asymmetric_covariance() -> None:
    with pytest.raises(ValidationError):
        ParameterSet(
            names=["a", "b"],
            values=[0.0, 0.0],
            covariance=[[1.0, 0.5], [0.0, 1.0]],
            provenance="x",
        )


def test_parameterset_samples_stub() -> None:
    ps = _toy_parameterset()
    with pytest.raises(NotImplementedError, match="SAMPLES_HDF5_FORMAT|not implemented"):
        ps.get_posterior_samples()
    assert "samples" in SAMPLES_HDF5_FORMAT.lower()


def test_candidate_record_round_trip() -> None:
    rec = CandidateRecord(
        source_id=1234567890123456789,
        nss_solution_type="Orbital",
        ra_deg=10.0,
        dec_deg=-20.0,
        parallax_mas=5.0,
        thiele_innes=ThieleInnesElements(A=1.0, B=2.0, F=3.0, G=4.0),
        nss_orbital={"period_day": 100.0, "eccentricity": 0.1},
        rv_summary={"instruments": ["APF"], "n_epochs": 12},
        photometry=[PhotometryPoint(band="G", mag=12.3, mag_err=0.01)],
        tess=TessBlock(period_day=2.5, variability_flag=True, light_curve_path="/ext/lc.fits"),
        m1=_toy_parameterset(),
        orbit_tier=OrbitTier.ASTROMETRY_ONLY,
        fit_tier=FitTier.BULK_ESTIMATE,
    )
    restored = CandidateRecord.model_validate(rec.model_dump())
    assert restored.source_id == rec.source_id
    assert restored.m1 is not None
    assert restored.m1.provenance == "TAG10"
    assert restored.orbit_tier is OrbitTier.ASTROMETRY_ONLY
    assert restored.tess is not None
    assert restored.tess.light_curve_path == "/ext/lc.fits"


def test_outlier_and_followup_round_trip() -> None:
    gate = OutlierTestResult(
        source_id=1,
        chi2_dof=1.2,
        threshold=3.0,
        passed=True,
        instruments=[],
    )
    follow = FollowUpRecord(
        source_id=1,
        target_lists=["APF"],
        adoption_dates={"APF": "2024-01-15"},
        n_observations=4,
        brightness_g_mag=13.0,
        declination_deg=-10.0,
    )
    assert OutlierTestResult.model_validate(gate.model_dump()).passed is True
    assert FollowUpRecord.model_validate(follow.model_dump()).adoption_dates["APF"] == (
        "2024-01-15"
    )


def test_run_manifest_yaml_round_trip() -> None:
    now = datetime(2026, 8, 25, 20, 0, 0, tzinfo=timezone.utc)
    manifest = RunManifest(
        run_id="20260825-200000-abcdef0",
        created_at=now,
        parent_run_id=None,
        config_checksum="abc123",
        active_dr_mode=ActiveDRMode.DR3,
        artifact_root="output",
        gaiamock_mod_release="gaiamock-mod-v1",
        gaiamock_mod_sha256="71454d768f5eb5514555d1cee716a3eab89121815d55fb561ba13f013d7ef028",
        gaiamock_git_commit="dd30fdbf787eb96878734605ac077ac69bf28c84",
        stages={
            "data_acquisition": StageRecord(
                stage_name="data_acquisition",
                status=StageStatus.COMPLETED,
                started_at=now,
                finished_at=now,
                source_hash="deadbeef",
                config_subset={"quality_cuts": []},
                artifact_path="output/data_acquisition/run.h5",
                code_commit="abcdef0",
            ),
            "mass_derivation_bulk": StageRecord(
                stage_name="mass_derivation_bulk",
                status=StageStatus.PENDING,
            ),
        },
    )
    assert manifest.is_incomplete()
    payload = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False)
    loaded = RunManifest.model_validate(yaml.safe_load(payload))
    assert loaded.run_id == manifest.run_id
    assert loaded.active_dr_mode is ActiveDRMode.DR3
    assert loaded.stages["data_acquisition"].status is StageStatus.COMPLETED
    assert loaded.stages["mass_derivation_bulk"].status is StageStatus.PENDING
    assert loaded.is_incomplete()


def test_run_manifest_complete_when_all_terminal() -> None:
    now = datetime.now(tz=timezone.utc)
    manifest = RunManifest(
        run_id="20260825-200001-abcdef0",
        created_at=now,
        config_checksum="x",
        stages={
            "a": StageRecord(stage_name="a", status=StageStatus.COMPLETED),
            "b": StageRecord(
                stage_name="b",
                status=StageStatus.SKIPPED,
                reason="rv_astrometry_gate_failed",
            ),
        },
    )
    assert manifest.is_complete()
    assert not manifest.is_incomplete()


def test_scalar_estimate_exists_for_standalone_inputs() -> None:
    s = ScalarEstimate(value=1.2, sigma=0.1, unit="Msun", provenance="external")
    assert ScalarEstimate.model_validate(s.model_dump()).value == pytest.approx(1.2)


def test_enums() -> None:
    assert OrbitTier.JOINT_ASTROMETRY_RV.value == "joint_astrometry_rv"
    assert FitTier.FULL_UBERMS.value == "full_uberMS"
    assert ActiveDRMode.DR3.value == "dr3"
