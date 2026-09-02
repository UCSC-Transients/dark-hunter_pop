"""Tests for dark-hunter_rv JSON summary attachment."""

from __future__ import annotations

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import ActiveDRMode
from darkhunter_pop.rv_adapter import attach_rv_summaries, resolve_rv_summary_path
from darkhunter_pop.rv_consistency import collect_rv_epochs
from darkhunter_pop.schemas import CandidateRecord

pytestmark = __import__("pytest").mark.unit


def test_resolve_rv_summary_path_fixture() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = "tests/fixtures/rv_summaries"
    path = resolve_rv_summary_path(tweaked, 424242)
    assert path is not None
    assert path.name == "424242.json"
    assert path.is_file()


def test_attach_rv_summaries_merges_json() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = "tests/fixtures/rv_summaries"
    candidates = [
        CandidateRecord(source_id=424242),
        CandidateRecord(source_id=999999),
    ]
    updated, stats = attach_rv_summaries(candidates, tweaked)
    assert stats["attached"] == 1
    assert stats["missing"] == 1
    assert updated[0].rv_summary.get("n_epochs") == 6
    assert len(collect_rv_epochs(updated[0].rv_summary)) >= 3
    assert updated[1].rv_summary == {}


def test_attach_disabled_when_root_null() -> None:
    cfg = load_config()
    assert cfg.dr3.rv_summary_root is None
    updated, stats = attach_rv_summaries([CandidateRecord(source_id=1)], cfg)
    assert stats["disabled"] == 1
    assert updated[0].rv_summary == {}


def test_dr4_has_independent_rv_root_key() -> None:
    cfg = load_config()
    assert cfg.active_dr_mode is ActiveDRMode.DR3
    assert hasattr(cfg.dr4, "rv_summary_root")


def test_attached_rv_summary_flows_through_gate() -> None:
    from darkhunter_pop.rv_consistency import run_gate_on_candidates

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = "tests/fixtures/rv_summaries"
    # Build a candidate that also has NSS orbit metadata for the gate.
    base = CandidateRecord(
        source_id=424242,
        nss_solution_type="Orbital",
        nss_orbital={
            "period": 200.0,
            "eccentricity": 0.2,
            "parallax": 5.0,
            "t_periastron": 100.0,
        },
    )
    updated, stats = attach_rv_summaries([base], tweaked)
    assert stats["attached"] == 1
    gated, diag = run_gate_on_candidates(updated, cfg)
    assert diag.n_scored >= 1
