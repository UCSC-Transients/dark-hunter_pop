"""#391 spec §12.31: observed-vbroad forward model (draft)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import vbroad_model as vm


def _one(**kw):
    base = dict(v1=[100.0], v2=[0.0], f_rvs=[0.0], k1=[0.0], k2=[0.0], ecc=[0.0], omega_rad=[0.0], teff1=[6000.0], grvs=[9.0])
    base.update({k: [v] for k, v in kw.items()})
    cfg = vm.VbroadModelConfig(rel_scatter=0.0)
    return vm.observed_vbroad(**base, rng=np.random.default_rng(1), cfg=cfg)[0]


@pytest.mark.unit
def test_single_cool_star_recovers_vsini_and_filters() -> None:
    assert _one() == pytest.approx(100.0)
    assert np.isnan(_one(v1=3.0))          # below the 5 km/s floor
    assert np.isnan(_one(grvs=12.5))       # fainter than G_RVS 12
    assert _one(teff1=9600.0) == pytest.approx(128.0)  # hot-star bias (Frémat §4)


@pytest.mark.unit
def test_no_sb1_smearing_and_sb2_flag() -> None:
    assert _one(k1=40.0, k2=0.0, f_rvs=0.0) == pytest.approx(100.0)   # dark companion: per-transit RV, no smearing
    assert np.isnan(_one(v1=10.0, v2=10.0, f_rvs=0.8, k1=80.0, k2=80.0))  # well-separated twin -> SB2-flagged
    assert _one(v1=60.0, v2=60.0, f_rvs=0.1, k1=5.0, k2=5.0) > 60.0    # unresolved blend broadens


@pytest.mark.unit
def test_k_semi_amplitudes_sun_jupiter_like() -> None:
    k1, k2 = vm.k_semi_amplitudes([1.0], [0.000954], [4332.6], [0.0], [90.0])
    assert k1[0] == pytest.approx(0.0125, rel=0.05)  # 12.5 m/s


@pytest.mark.unit
def test_rotation_prior_sampling(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    m = np.r_[rng.uniform(1.6, 2.0, 200), rng.uniform(2.5, 3.0, 200)]
    v = np.r_[rng.uniform(10, 60, 200), rng.uniform(150, 250, 200)]
    pr = vm.RotationPrior(m, rng.uniform(0, 1, 400), v, cool_vsini=np.full(50, 7.0))
    s = pr.sample([1.8, 2.8, 1.0, 2.8], [0.3, 0.3, 0.3, 0.3], [False, False, False, True], np.random.default_rng(1))
    assert 10 <= s[0] <= 60 and 150 <= s[1] <= 250 and s[2] == 7.0 and 1.0 <= s[3] <= 8.0


@pytest.mark.unit
def test_zorec_royer_reader(tmp_path: Path) -> None:
    p = tmp_path / "t.dat"
    p.write_text("     3 A1Vn            3.957 0.008  1.6387 0.0725 2.36 0.08 0.685 0.075 228  4 CP \n")
    pr = vm.RotationPrior.from_zorec_royer(p)
    assert pr.m[0] == pytest.approx(2.36) and pr.a[0] == pytest.approx(0.685) and pr.v[0] == 228.0
