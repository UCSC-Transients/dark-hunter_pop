"""Unit tests for ``darkhunter_pop.constants`` (true constants only)."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from darkhunter_pop import constants as C

pytestmark = pytest.mark.unit


def test_astropy_reexports_are_quantities() -> None:
    assert C.c.unit.is_equivalent(u.m / u.s)
    assert C.G.unit.is_equivalent(u.m**3 / u.kg / u.s**2)
    assert C.M_sun.unit.is_equivalent(u.kg)


def test_m_ch_is_solar_mass_quantity() -> None:
    assert C.M_CH.unit.is_equivalent(u.Msun)
    assert C.M_CH.to_value(u.Msun) == pytest.approx(1.4)


def test_tag10_coefficient_tables() -> None:
    assert C.TAG10_A.shape == (7,)
    assert C.TAG10_B.shape == (7,)
    assert C.TAG10_A_ERR.shape == (7,)
    assert C.TAG10_B_ERR.shape == (7,)
    assert C.TAG10_X_OFFSET == pytest.approx(4.1)
    # Spot-check Table 1 first / last entries (Torres et al. 2010).
    assert C.TAG10_A[0] == pytest.approx(1.5689)
    assert C.TAG10_A[6] == pytest.approx(0.1010)
    assert C.TAG10_B[0] == pytest.approx(2.4427)
    assert C.TAG10_B[4] == pytest.approx(-0.21415)


def test_santos2013_coefficients() -> None:
    assert C.SANTOS2013_S2 == pytest.approx(0.791)
    assert C.SANTOS2013_S1 == pytest.approx(-0.575)
    assert C.SANTOS2013_S0 == pytest.approx(0.701)


def test_spectroscopic_mass_function_factor() -> None:
    assert C.SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS == pytest.approx(1.036149e-7, rel=1e-6)
    assert C.GAIA_J2016_MJD == pytest.approx(57388.5)


def test_no_choosable_thresholds_exported() -> None:
    """Regression: choosables must not appear as module-level constants."""
    banned = {
        "M_MIN",
        "M_TOV",
        "SIGMA_LOG_M",
        "sigma_logM",
        "N_SIGMA",
        "DELTA_BIC",
        "CHI2_DOF_THRESHOLD",
    }
    exported = set(dir(C))
    assert banned.isdisjoint(exported)
