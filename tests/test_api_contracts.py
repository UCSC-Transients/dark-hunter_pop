"""API-contract tests: registry inputs_from completeness and schema producer→consumer.

ARCHITECTURE.md §10. Catches cases where a stage output shape changes but a declared
consumer is not updated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_pop.run_management import STAGE_REGISTRY, validate_registry_inputs_from
from darkhunter_pop.schemas import (
    CandidateRecord,
    ParameterSet,
    RunManifest,
)

pytestmark = pytest.mark.api


def test_registry_inputs_from_complete() -> None:
    assert validate_registry_inputs_from() == []


def test_parameterset_consumer_accepts_producer_dump() -> None:
    """Producer writes ParameterSet; consumer re-validates the same payload."""
    produced = ParameterSet(
        names=["M1", "R1"],
        values=[1.0, 1.0],
        covariance=[[0.01, 0.0], [0.0, 0.01]],
        provenance="TAG10",
        units=["Msun", "Rsun"],
    )
    payload = produced.model_dump(mode="json")
    consumed = ParameterSet.model_validate(payload)
    assert consumed.marginal("M1").sigma == pytest.approx(0.1)


def test_candidate_record_accepts_embedded_parameterset() -> None:
    m1 = ParameterSet(
        names=["M1"],
        values=[1.2],
        covariance=[[0.04]],
        provenance="TAG10",
    )
    produced = CandidateRecord(source_id=1, m1=m1, fit_tier=None)
    consumed = CandidateRecord.model_validate(produced.model_dump(mode="json"))
    assert consumed.m1 is not None
    assert consumed.m1.provenance == "TAG10"


def test_data_acquisition_hdf5_producer_consumer_round_trip(tmp_path: Path) -> None:
    """Producer (stage HDF5) → consumer (read_stage_hdf5) contract."""
    from datetime import datetime, timezone

    from astropy.table import Table

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.data_acquisition import (
        FunnelCounts,
        SnapshotMeta,
        compute_stage_diagnostics,
        read_stage_hdf5,
        table_to_candidates,
        write_stage_hdf5,
    )

    table = Table(
        {
            "source_id": [1001],
            "nss_solution_type": ["Orbital"],
            "ra": [10.0],
            "dec": [-20.0],
            "parallax": [5.0],
            "goodness_of_fit": [4.0],
            "ruwe": [1.2],
            "period": [100.0],
            "eccentricity": [0.1],
            "phot_g_mean_mag": [12.5],
            "g_mag": [12.5],
        }
    )
    dr = load_config().dr3
    candidates = table_to_candidates(table, dr)
    snapshot = SnapshotMeta(
        snapshot_id="api_test",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="deadbeef",
        row_count=1,
        result_path=tmp_path / "q.ecsv",
        meta_path=tmp_path / "m.yaml",
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=1, after_quality_cut=1, candidates_written=1),
        quality_cut_bin_counts={},
    )
    path = tmp_path / "artifact.h5"
    write_stage_hdf5(path, candidates, snapshot=snapshot, diagnostics=diagnostics)
    loaded, meta = read_stage_hdf5(path)
    assert CandidateRecord.model_validate(loaded[0].model_dump(mode="json"))
    assert meta["n_candidates"] == 1


def test_every_stage_declares_dependency_modules() -> None:
    for name, spec in STAGE_REGISTRY.items():
        assert spec.dependency_modules, f"{name} missing dependency_modules"
        assert spec.module, f"{name} missing module"


def test_sensitivity_analysis_hdf5_producer_consumer_round_trip(tmp_path: Path) -> None:
    """Producer (sensitivity_analysis HDF5) → consumer (read payload) contract."""
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.sensitivity_analysis import (
        read_sensitivity_analysis_artifact,
        run_sensitivity_analysis,
        write_sensitivity_analysis_artifact,
    )

    result = run_sensitivity_analysis(load_config(), require_mc_pass=True)
    path = tmp_path / "sensitivity.h5"
    write_sensitivity_analysis_artifact(path, result)
    payload = read_sensitivity_analysis_artifact(path)
    assert payload["schema_version"] == 1
    assert "dimensionality" in payload
    assert "class_covariates" in payload
    assert "mc_noise_convergence" in payload


def test_run_manifest_round_trip_is_consumer_safe() -> None:
    from datetime import datetime, timezone

    from darkhunter_pop.schemas import ActiveDRMode, StageRecord, StageStatus

    produced = RunManifest(
        run_id="20260825-120000-abcdef0",
        created_at=datetime.now(tz=timezone.utc),
        config_checksum="abc",
        active_dr_mode=ActiveDRMode.DR3,
        stages={
            "data_acquisition": StageRecord(
                stage_name="data_acquisition",
                status=StageStatus.COMPLETED,
            )
        },
    )
    consumed = RunManifest.model_validate(produced.model_dump(mode="json"))
    assert consumed.is_complete()
