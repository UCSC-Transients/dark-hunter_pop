"""FL-T1 (#641): candidate cuts, GALEX epoch-propagated separation, per-host aggregation. No data/ access."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "fl_candidates_641.py"
_s = importlib.util.spec_from_file_location("fl_candidates_641", _P)
mod = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mod)  # type: ignore[union-attr]


@pytest.mark.unit
def test_select_candidates_fl_q2() -> None:
    m2 = np.array([1.2, 0.8, 0.8, 0.5, np.nan, 1.5])
    a = np.array([0.1, 0.6, 0.4, 0.9, 0.9, np.nan])
    ok = np.array([True, True, True, True, True, False])
    assert mod.select_candidates(m2, a, ok, 1.0, 0.5, 0.6).tolist() == [True, True, False, False, False, False]


@pytest.mark.unit
def test_propagated_sep() -> None:
    # 1000 mas/yr in dec for -9 yr moves the host 9 arcsec south; a source there has ~0 separation
    s = mod.propagated_sep_arcsec(np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([1000.0]),
                                  np.array([10.0]), np.array([-9.0 / 3600.0]), -9.0)
    assert s[0] == pytest.approx(0.0, abs=1e-6)
    s2 = mod.propagated_sep_arcsec(np.array([359.9999]), np.array([0.0]), np.array([np.nan]), np.array([np.nan]),
                                   np.array([0.0001]), np.array([0.0]), -9.0)
    assert s2[0] == pytest.approx(0.72, rel=1e-3)


@pytest.mark.unit
def test_by_sid_aggregations() -> None:
    sids = np.array([5, 7, 9])
    key = np.array([7, 7, 5, 11])
    assert mod._by_sid(sids, key, np.ones(4), "count").tolist() == [1, 2, 0]
    assert mod._by_sid(sids, key, np.array([1.0, 3.0, 2.0, 9.0]), "max")[:2].tolist() == [2.0, 3.0]
    assert mod._by_sid(sids, key, np.array([100, 100, 5, 1]), "nunique").tolist() == [1, 1, 0]
