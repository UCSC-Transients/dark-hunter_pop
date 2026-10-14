"""Spec §12.23 (#494): MP-Q27 IPD-pass weight and resolution envelope from wide binaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import resolved_companions as rc


def _cat(n: int = 40000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    th = 10 ** rng.uniform(np.log10(0.3), 1.0, n)  # flat in log θ (complete truth)
    keep = rng.uniform(size=n) < np.clip((th - 0.4) / 1.0, 0, 1)  # resolved fraction rises 0.4 -> 1.4"
    fail = rng.uniform(size=n) < np.where(th < 1.5, 0.9, 0.05)
    dg = rng.uniform(0, 2, n)
    return {"theta": th[keep], "R": np.zeros(keep.sum()), "Gmag1": np.full(keep.sum(), 15.0),
            "Gmag2": 15.0 + dg[keep], "IPDfmp1": np.where(fail[keep], 10.0, 0.0), "IPDgofha1": np.full(keep.sum(), 0.05)}


@pytest.mark.unit
def test_tables_and_weight() -> None:
    tab = rc.build_tables(_cat(), [0.3, 0.5, 0.7, 1.0, 1.5, 2, 3, 4, 6, 8, 10], [0, 2], r_chance_max=0.1, g2_max=19.5,
                          ref_theta_arcsec=(3.0, 10.0), ipd_fmp_max=2.0, ipd_gof_max=0.1, min_pairs=20)
    assert tab.completeness[-2, 0] == pytest.approx(1.0, abs=0.15)
    assert tab.completeness[1, 0] < 0.5
    assert tab.p_pass[3, 0] == pytest.approx(0.1 / 0.95, abs=0.05)
    assert tab.p_pass[-1, 0] == pytest.approx(1.0, abs=0.05)
    th = np.array([0.1, 0.6, 1.2, 5.0, 1.2])
    dg = np.array([1.0, 1.0, 1.0, 1.0, np.inf])  # last: dark companion
    none = rc.p_pass_ipd(th, dg, tab, below_envelope="none", hold_min_arcsec=0.1)
    hold = rc.p_pass_ipd(th, dg, tab, below_envelope="hold", hold_min_arcsec=0.05)
    assert none[0] == 1.0 and hold[0] < 0.5
    assert none[2] < 0.3 and none[3] == pytest.approx(1.0, abs=0.05) and none[4] == 1.0


@pytest.mark.unit
def test_load_checks_sha(tmp_path: Path) -> None:
    p = tmp_path / "c.csv"
    p.write_text("theta,R,Gmag1,Gmag2,IPDfmp1,IPDgofha1\n0.001,0.0,15,16,0,0.05\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    d = rc.load_elbadry2021(p, sha)
    assert d["theta"][0] == pytest.approx(3.6)
    with pytest.raises(ValueError):
        rc.load_elbadry2021(p, "0" * 64)


def _orb6_line(p: str, pu: str, a: str, au: str, i: str, t: str, tu: str, e: str, om: str, grade: str) -> str:
    buf = [" "] * 240
    for k, val in (("v1", "6.0"), ("v2", "7.0"), ("p", p), ("p_unit", pu), ("a", a), ("a_unit", au), ("i", i),
                   ("node", "10"), ("t", t), ("t_unit", tu), ("e", e), ("omega", om), ("grade", grade)):
        a0, b0 = rc.ORB6_SLICES[k]
        buf[a0:a0 + len(val)] = list(val)
    return "".join(buf)


@pytest.mark.unit
def test_parse_orb6_and_separation() -> None:
    lines = [_orb6_line("10.0", "y", "0.5", "a", "90.0", "2016.0", "y", "0.0", "0.0", "2"),
             _orb6_line("3652.5", "d", "500", "m", "0.0", "51544.5", "m", "0.0", "0.0", "4")]
    o = rc.parse_orb6(lines)
    np.testing.assert_allclose(o["p_yr"], [10.0, 10.0])
    np.testing.assert_allclose(o["a_as"], [0.5, 0.5])
    np.testing.assert_allclose(o["t_yr"], [2016.0, 2000.0], atol=1e-6)
    np.testing.assert_allclose(o["grade"], [2, 4])
    kep = lambda M, e: M  # circular orbits: E = M  # noqa: E731
    np.testing.assert_allclose(rc.orb6_separation(o, 2016.0, kep), [0.5, 0.5], atol=1e-9)  # edge-on at node; face-on
    np.testing.assert_allclose(rc.orb6_separation(o, 2018.5, kep)[0], 0.0, atol=1e-9)  # edge-on, quarter period


@pytest.mark.unit
def test_orb6_pass_table_normalization_and_fill() -> None:
    th = np.repeat([0.03, 0.2], 50)
    dg = np.zeros(100)
    ok = np.r_[np.ones(40), np.zeros(10), np.ones(5), np.zeros(45)]
    t = rc.orb6_pass_table(th, dg, ok, np.ones(100), [0.02, 0.05, 0.1, 0.3], [0, 1, 2, 6, 12], min_weight=8,
                           ref_theta_max=0.05, fill_dg_max=5.0)
    np.testing.assert_allclose(t.p_pass[0, 0], 1.0)
    np.testing.assert_allclose(t.p_pass[2, 0], 0.1 / 0.8)
    np.testing.assert_allclose(t.p_pass[2, 1], t.p_pass[2, 0])  # filled from lower ΔG
    np.testing.assert_allclose(t.p_pass[:, 3], 1.0)  # ΔG >= fill_dg_max
