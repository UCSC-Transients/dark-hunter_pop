"""Tests for dark-hunter_rv JSON summary attachment."""

from __future__ import annotations

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import ActiveDRMode
from darkhunter_pop.rv_adapter import attach_rv_summaries, resolve_rv_summary_path
from darkhunter_pop.rv_consistency import collect_rv_epochs, run_gate_on_candidates
from darkhunter_pop.schemas import CandidateRecord

pytestmark = __import__("pytest").mark.unit


def test_resolve_rv_summary_path_fixture() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = "tests/fixtures/rv_summaries"
    path = resolve_rv_summary_path(tweaked, 424242)
    assert path is not None
    assert path.name == "Gaia_DR3_424242_summary.json"
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
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = None
    updated, stats = attach_rv_summaries([CandidateRecord(source_id=1)], tweaked)
    assert stats["disabled"] == 1
    assert updated[0].rv_summary == {}


def test_default_config_sets_independent_rv_roots() -> None:
    from darkhunter_pop.config_schema import DRPathConfig

    cfg = load_config()
    assert cfg.active_dr_mode is ActiveDRMode.DR3
    assert cfg.dr3.rv_summary_root == "data/dr3/rv_summaries"
    assert cfg.dr4.rv_summary_root == "data/dr4/rv_summaries"
    assert cfg.dr3.rv_summary_filename_template == "Gaia_DR3_{source_id}_summary.json"
    assert cfg.dr4.rv_summary_filename_template == "Gaia_DR3_{source_id}_summary.json"
    # Keys remain independent even when templates match.
    assert "rv_summary_root" in DRPathConfig.model_fields


def test_attached_rv_summary_flows_through_gate() -> None:
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
    # Gate re-attaches even when DA left rv_summary empty.
    gated, diag = run_gate_on_candidates([base], tweaked)
    assert diag.rv_summary_attached == 1
    assert diag.n_scored >= 1
    assert gated[0].rv_summary.get("n_epochs") == 6


def test_gate_prefers_priority_then_recent_mtime(tmp_path) -> None:
    import os
    import time

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    root = tmp_path / "rv"
    root.mkdir()
    tweaked.dr3.rv_summary_root = str(root)
    tweaked.rv_consistency.priority_source_ids = [111]
    tweaked.rv_consistency.prefer_recent_summary_mtime = True
    tweaked.rv_consistency.prefer_public_external_rvs = False

    def _write(sid: int, mtime: float) -> None:
        path = root / f"Gaia_DR3_{sid}_summary.json"
        path.write_text(
            '{"schema_version":1,"source_id":%d,"n_epochs":3,'
            '"nss_orbital":{"period_day":100.0,"eccentricity":0.1,'
            '"t_periastron_day":10.0,"semi_amp_primary_kms":10.0,'
            '"arg_periastron_deg":30.0},'
            '"pipeline_epochs":['
            '{"mjd":58000.0,"rv_kms":1.0,"rv_err_kms":0.5,"instrument":"APF"},'
            '{"mjd":58010.0,"rv_kms":2.0,"rv_err_kms":0.5,"instrument":"APF"},'
            '{"mjd":58020.0,"rv_kms":1.5,"rv_err_kms":0.5,"instrument":"APF"}'
            "],"
            '"external_rvs":[]}' % sid,
            encoding="utf-8",
        )
        os.utime(path, (mtime, mtime))

    now = time.time()
    _write(222, now - 100)  # older
    _write(333, now - 10)  # newer non-priority
    _write(111, now - 50)  # priority calibrator

    cands = [
        CandidateRecord(
            source_id=sid,
            nss_orbital={"period": 100.0, "eccentricity": 0.1, "t_periastron": 10.0},
        )
        for sid in (222, 333, 111)
    ]
    gated, diag = run_gate_on_candidates(cands, tweaked)
    assert diag.n_scored >= 1
    # Priority id first, then newer mtime (333 before 222).
    assert [c.source_id for c in gated[:3]] == [111, 333, 222]


def test_kept_existing_rv_summary_not_overwritten() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.dr3.rv_summary_root = "tests/fixtures/rv_summaries"
    prior = CandidateRecord(
        source_id=424242,
        rv_summary={"n_epochs": 1, "pipeline_epochs": [], "source": "prior"},
    )
    updated, stats = attach_rv_summaries([prior], tweaked)
    assert stats["kept_existing"] == 1
    assert stats["attached"] == 0
    assert updated[0].rv_summary["source"] == "prior"
