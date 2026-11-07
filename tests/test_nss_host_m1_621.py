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


@pytest.mark.unit
def test_bj_distances_lookup_and_checksum(tmp_path: Path) -> None:
    import hashlib

    import yaml
    from astropy.table import Table

    t = Table({"source_id": np.array([5, 7], np.int64), "r_med_geo": [100.0, 200.0], "r_lo_geo": [90.0, 180.0],
               "r_hi_geo": [110.0, 220.0]})
    t.write(tmp_path / "table.ecsv", format="ascii.ecsv")
    sha = hashlib.sha256((tmp_path / "table.ecsv").read_bytes()).hexdigest()
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump({"table_ecsv_sha256": sha}))
    r, lo, hi = mod.bj_distances(tmp_path, np.array([7, 9, 5]))
    assert r[0] == 200.0 and np.isnan(r[1]) and r[2] == 100.0 and lo[2] == 90.0 and hi[0] == 220.0
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump({"table_ecsv_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="checksum"):
        mod.bj_distances(tmp_path, np.array([5]))
