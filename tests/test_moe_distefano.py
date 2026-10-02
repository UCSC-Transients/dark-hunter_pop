"""Moe & Di Stefano (2017) densities against the paper's own numbers (#391).

docs/MOCK_POPULATION_SPEC.md §2. Checks are against values MdS17 prints: Table 13
(F_twin, gamma_largeq, gamma_smallq at log P = 1, 3, 5, 7 for solar-type primaries; the
twin and slope anchors for other types), f_mult;q>0.3(1 Msun) = 0.36 (§9.4), the Fig. 2
normalization, continuity at q = 0.3 (Eq. 2), e_max (Eq. 3) and circularization.
"""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import moe_distefano as mds

pytestmark = pytest.mark.physics

LINEAR = "linear_m1"


@pytest.fixture(scope="module")
def table() -> mds.MdS17Table:
    return mds.load_mds17_table()


def test_multiplicity_frequency_solar(table: mds.MdS17Table) -> None:
    lp = np.linspace(0.2, 8.0, 200_001)
    fmult = np.trapz(mds.f_logp_q03(1.0, lp, table), lp)
    assert fmult == pytest.approx(0.36, abs=0.01)  # §9.4: 0.36 ± 0.03


def test_frequency_anchors_and_continuity(table: mds.MdS17Table) -> None:
    for m1 in (1.0, 3.5, 10.0):
        x = np.log10(m1)
        assert mds.f_logp_q03(m1, 0.5, table) == pytest.approx(0.020 + 0.04 * x + 0.07 * x * x)
        assert mds.f_logp_q03(m1, 2.7, table) == pytest.approx(0.039 + 0.07 * x + 0.01 * x * x)
        assert mds.f_logp_q03(m1, 5.5, table) == pytest.approx(0.078 - 0.05 * x + 0.04 * x * x)
        for edge in (1.0, 2.0, 3.4, 5.5):
            lo, hi = mds.f_logp_q03(m1, [edge - 1e-9, edge + 1e-9], table)
            assert lo == pytest.approx(hi, abs=1e-6)
    assert mds.f_logp_q03(1.0, [0.1, 8.0, 9.0], table).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "log_p,f_twin,g_large,g_small",
    [(1.0, 0.30, -0.5, 0.3), (3.0, 0.20, -0.5, 0.3), (5.0, 0.10, -0.5, 0.3), (7.0, 0.0, -1.1, 0.3)],
)
def test_table13_solar_type(table: mds.MdS17Table, log_p: float, f_twin: float, g_large: float, g_small: float) -> None:
    assert float(mds.f_twin(1.0, log_p, table)) == pytest.approx(f_twin, abs=1e-9)
    assert float(mds.gamma_largeq(1.0, log_p, table, m1_interpolation=LINEAR)) == pytest.approx(g_large)
    assert float(mds.gamma_smallq(1.0, log_p, table, m1_interpolation=LINEAR)) == pytest.approx(g_small)


def test_table13_anchor_types(table: mds.MdS17Table) -> None:
    lp = np.array([1.0, 3.0, 5.0, 7.0])
    # Eq. 10 at 3.5 Msun (A / late-B midpoint): Table 13 column -0.5, -0.9, -1.4, -2.0
    np.testing.assert_allclose(mds.gamma_largeq(3.5, lp, table, m1_interpolation=LINEAR), [-0.5, -0.9, -1.4, -2.0])
    # Eq. 11 / 15 above 6 Msun: Table 13 mid-B..O columns
    np.testing.assert_allclose(mds.gamma_largeq(10.0, lp, table, m1_interpolation=LINEAR), [-0.5, -1.7, -2.0, -2.0])
    np.testing.assert_allclose(mds.gamma_smallq(10.0, lp, table, m1_interpolation=LINEAR), [0.1, -0.2, -1.2, -1.5])
    # Eq. 6 short-period twin fraction for O-type (Table 13: 0.08) and early-B (0.14)
    assert float(mds.f_twin(28.0, 0.5, table)) == pytest.approx(0.30 - 0.15 * np.log10(28.0))
    assert float(mds.f_twin(12.0, 3.0, table)) == 0.0  # log P_twin = 1.5 above 6.5 Msun


def test_interpolation_between_anchors(table: mds.MdS17Table) -> None:
    mid = 0.5 * (1.2 + 3.5)
    lp = 3.0
    lin = float(mds.gamma_largeq(mid, lp, table, m1_interpolation=LINEAR))
    assert lin == pytest.approx(0.5 * (-0.5 + -0.9))
    log = float(mds.gamma_largeq(mid, lp, table, m1_interpolation="linear_log_m1"))
    assert log != pytest.approx(lin)


def test_q_density_normalization_and_continuity(table: mds.MdS17Table) -> None:
    for m1, lp in ((1.0, 2.0), (3.5, 4.0), (10.0, 1.2)):
        q = np.linspace(0.3, 1.0, 400_001)
        assert np.trapz(mds.q_density(q, m1, lp, table, m1_interpolation=LINEAR), q) == pytest.approx(1.0, abs=1e-4)
        below, above = mds.q_density([0.3 - 1e-9, 0.3], m1, lp, table, m1_interpolation=LINEAR)
        assert below == pytest.approx(above, rel=1e-6)
    assert mds.q_density([0.05, 1.01], 1.0, 2.0, table, m1_interpolation=LINEAR).tolist() == [0.0, 0.0]


def test_twin_excess_is_excess_fraction(table: mds.MdS17Table) -> None:
    # Fig. 2 definition: uniform gamma_largeq = 0 with F_twin -> density 1 - F + F/0.05 above 0.95.
    lp = 3.0
    ft = float(mds.f_twin(1.0, lp, table))
    gl = float(mds.gamma_largeq(1.0, lp, table, m1_interpolation=LINEAR))
    q_hi, q_lo = 0.97, 0.9
    d_hi, d_lo = mds.q_density([q_hi, q_lo], 1.0, lp, table, m1_interpolation=LINEAR)
    norm = (1.0 - 0.3 ** (gl + 1)) / (gl + 1)
    assert d_lo == pytest.approx((1 - ft) * q_lo**gl / norm)
    assert d_hi == pytest.approx((1 - ft) * q_hi**gl / norm + ft / 0.05)


def test_eccentricity(table: mds.MdS17Table) -> None:
    assert float(mds.e_max(2.0, table)) == 0.0
    assert float(mds.e_max(16.0, table)) == pytest.approx(1.0 - 8.0 ** (-2.0 / 3.0))
    assert mds.is_circular([1.5, 2.0, 2.1], table).tolist() == [True, True, False]
    e = np.linspace(0.0, 1.0, 400_001)
    for m1, p in ((1.0, 300.0), (8.0, 50.0)):
        dens = mds.e_density(e, m1, p, table, m1_interpolation=LINEAR, eta_floor=-0.9)
        assert np.trapz(dens, e) == pytest.approx(1.0, abs=2e-3)
    # Eq. 17 and Eq. 18 directly
    assert float(mds.eta(1.0, 2.0, table, m1_interpolation=LINEAR, eta_floor=-0.9)) == pytest.approx(0.6 - 0.7 / 1.5)
    assert float(mds.eta(10.0, 2.0, table, m1_interpolation=LINEAR, eta_floor=-0.9)) == pytest.approx(0.9 - 0.2 / 1.5)
    assert float(mds.eta(1.0, 0.8, table, m1_interpolation=LINEAR, eta_floor=-0.9)) == -0.9
    with pytest.raises(ValueError):
        mds.eta(1.0, 2.0, table, m1_interpolation=LINEAR, eta_floor=-1.0)


def test_low_mass_scale() -> None:
    s = mds.low_mass_frequency_scale([0.05, 0.08, np.sqrt(0.08 * 0.8), 0.8, 2.0], m1_anchor_msun=0.8, m1_zero_msun=0.08)
    np.testing.assert_allclose(s, [0.0, 0.0, 0.5, 1.0, 1.0])
