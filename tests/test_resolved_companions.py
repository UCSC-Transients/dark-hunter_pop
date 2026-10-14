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
