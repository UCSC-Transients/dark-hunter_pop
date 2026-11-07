"""R5-T7 (#624): gate-5a target resolution and the recovery statistic. No data/ access, no gaiamock."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rung5_known_objects_5a", ROOT / "scripts" / "rung5_known_objects_5a.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)  # type: ignore[union-attr]
CFG = yaml.safe_load((ROOT / "config" / "benchmarks" / "rung5_known_objects_dr3.yaml").read_text())


@pytest.mark.unit
def test_targets_resolve() -> None:
    t = mod.load_targets(CFG, ROOT)
    groups = {g: [x for x in t if x["group"] == g] for g in CFG["groups"]}
    assert {x["name"] for x in groups["known_truth_bh"]} == {"Gaia-BH1", "Gaia-BH2"}
    assert len(groups["elbadry2024_ns_candidates"]) == 24  # Table 3 state_counts.good
    assert len({x["source_id"] for x in t}) == len(t)
    assert CFG["not_injected"][0]["source_id"] not in {x["source_id"] for x in t}


@pytest.mark.unit
def test_wilson_interval() -> None:
    p, lo, hi = mod.binomial_interval(25, 50)
    assert p == 0.5 and lo < 0.5 < hi and hi - lo == pytest.approx(0.139, abs=0.01)
    assert all(x != x for x in mod.binomial_interval(0, 0))


@pytest.mark.unit
def test_summarize_counts_c1_and_errors() -> None:
    recs = [
        {"source_id": 1, "realization": 0, "outcome": 12, "accepted": True, "period": 300.0},
        {"source_id": 1, "realization": 1, "outcome": 12, "accepted": True, "period": 900.0},   # above C1
        {"source_id": 1, "realization": 2, "outcome": 12, "accepted": False, "period": 300.0},
        {"source_id": 1, "realization": 3, "outcome": 9},
        {"source_id": 1, "realization": 4, "error": "boom"},
    ]
    s = mod.summarize(recs, 830.4)[1]
    assert s["n"] == 4 and s["n_error"] == 1
    assert s["recovered_c1"][0] == pytest.approx(0.25)
    assert s["acceleration"][0] == pytest.approx(0.25)
    s2 = mod.summarize(recs, 830.4, lambda r: False)[1]
    assert s2["acceleration"][0] == 0.0
