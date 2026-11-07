"""#621: distance helpers of scripts/build_nss_host_m1_table_621.py (rung-3 real-side convention). No data/ access."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "build_nss_host_m1_table_621.py"
_spec = importlib.util.spec_from_file_location("build_nss_host_m1_table_621", _P)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)  # type: ignore[union-attr]


@pytest.mark.unit
def test_zero_point_median_per_g_bin() -> None:
    g = np.array([10.0, 10.5, 12.0, 14.0, 16.0, 18.0, 18.5])
    obs = np.full(7, 2.0)
    truth = obs + np.array([0.01, 0.03, 0.02, 0.04, 0.05, 0.06, 0.08])
    parent = SimpleNamespace(columns={"phot_g_mean_mag": g, "parallax": obs, "parallax_error": np.full(7, 0.01)},
                             truth_parallax_mas=truth, usable=np.ones(7, bool))
    np.testing.assert_allclose(mod.parallax_zero_point(parent), [0.02, 0.02, 0.04, 0.05, 0.07])


@pytest.mark.unit
def test_row_distances_bounds_and_nan() -> None:
    zp = np.array([0.0, 0.0, 0.1, 0.0, 0.0])
    rr, rlo, rhi = mod.row_distances(np.array([14.0, 14.0, 14.0]), np.array([1.9, -0.2, 0.05]), np.array([0.1, 0.1, 0.2]), zp)
    assert rr[0] == pytest.approx(500.0) and rlo[0] == pytest.approx(1000 / 2.1) and rhi[0] == pytest.approx(1000 / 1.9)
    assert np.isnan(rr[1]) and np.isnan(rhi[2]) and np.isfinite(rr[2])
