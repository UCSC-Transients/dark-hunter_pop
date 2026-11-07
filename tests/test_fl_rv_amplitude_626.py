"""FL-T3 (#626): K1 prediction and the noise-floor summary. No data/ access."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "fl_rv_amplitude_check_626.py"
_s = importlib.util.spec_from_file_location("fl_rv_amplitude_check_626", _P)
mod = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mod)  # type: ignore[union-attr]


@pytest.mark.unit
def test_k1_earth_sun() -> None:
    # 1 AU orbit, 1 yr, edge-on, circular -> 29.78 km/s
    k = mod.k1_pred_kms(np.array([10.0]), np.array([10.0]), np.array([365.25]), np.array([0.0]), np.array([np.pi / 2]))
    assert k[0] == pytest.approx(29.78)
    k2 = mod.k1_pred_kms(np.array([10.0]), np.array([10.0]), np.array([365.25]), np.array([0.6]), np.array([np.pi / 6]))
    assert k2[0] == pytest.approx(29.78 * 0.5 / 0.8)


@pytest.mark.unit
def test_floor_by_bin() -> None:
    g = np.array([8.5, 8.6, 8.7, 9.5, 20.0])
    a = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    med, p90, n = mod.floor_by_bin(g, a, np.array([8.0, 9.0, 10.0, 11.0]))
    assert n.tolist() == [3, 1, 0]
    assert med[0] == 2.0 and med[1] == 4.0 and np.isnan(med[2])
