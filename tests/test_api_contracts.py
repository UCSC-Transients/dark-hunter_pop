"""API-contract tests: registry inputs_from completeness and schema producer→consumer.

ARCHITECTURE.md §10. Catches cases where a stage output shape changes but a declared
consumer is not updated.
"""

from __future__ import annotations

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


def test_every_stage_declares_dependency_modules() -> None:
    for name, spec in STAGE_REGISTRY.items():
        assert spec.dependency_modules, f"{name} missing dependency_modules"
        assert spec.module, f"{name} missing module"


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
