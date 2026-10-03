"""True physical constants for ``darkhunter_pop``.

Only quantities that are **not choosable** live here:

* re-exports from :mod:`astropy.constants` (``c``, ``G``, solar units, …);
* literature-fixed calibration coefficient tables (TAG10, Santos et al. 2013);
* the Chandrasekhar mass ``M_Ch`` (optional ``Delta_M_Ch`` belongs in config).

Thresholds, method switches, priors, and published scatters that may be updated when a
calibration is swapped (``sigma_logM``, ``M_MIN``, ``M_TOV`` prior, Santos on/off, …) live in
``config.yaml`` — see ``docs/ARCHITECTURE.md`` §2 and §7.
"""

from __future__ import annotations

import math

from typing import Final

import numpy as np
from astropy import constants as const
from astropy import units as u
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Astropy re-exports (prefer these over hand-coded SI values)
# ---------------------------------------------------------------------------

c = const.c
G = const.G
h = const.h
k_B = const.k_B
sigma_sb = const.sigma_sb
M_sun = const.M_sun
R_sun = const.R_sun
L_sun = const.L_sun
au = const.au
pc = const.pc

# ---------------------------------------------------------------------------
# Compact-object mass scale (WD hard truncation). Choosable Delta_M_Ch → config.
# ---------------------------------------------------------------------------

# Nominal Chandrasekhar mass for a carbon-oxygen WD (standard literature value).
M_CH: Final[u.Quantity] = 1.4 * u.Msun

# ---------------------------------------------------------------------------
# Unit bookkeeping used by the astrometric mass function / absolute magnitudes
# ---------------------------------------------------------------------------

# Days per Julian year (IAU; astropy ``u.yr``): P_yr = P_day / JULIAN_YEAR_DAYS.
JULIAN_YEAR_DAYS: Final[float] = float((1.0 * u.yr).to_value(u.day))
# Parallax (mas) of a source at the 10 pc absolute-magnitude reference distance:
# M = m - 5 log10(PARALLAX_MAS_AT_10PC / parallax_mas).
# Written as 1000 mas arcsec^-1 / 10 pc (exactly 100.0; the astropy unit
# conversion rounds to 99.99999999999999).
PARALLAX_MAS_AT_10PC: Final[float] = 1000.0 / 10.0

# ---------------------------------------------------------------------------
# Torres, Andersen & Giménez (2010) — Table 1 coefficients
# https://doi.org/10.1007/s00159-009-0025-1  (arXiv:0908.2624)
#
#   log M = a1 + a2*X + a3*X^2 + a4*X^3 + a5*(log g)^2 + a6*(log g)^3 + a7*[Fe/H]
#   log R = b1 + b2*X + b3*X^2 + b4*X^3 + b5*(log g)^2 + b6*(log g)^3 + b7*[Fe/H]
#   X = log10(Teff) - TAG10_X_OFFSET
#
# Coefficient uncertainties are retained for analytic error propagation in
# mass_derivation; the published residual scatters sigma_logM / sigma_logR are
# config defaults (method-tied, swappable), not constants.
# ---------------------------------------------------------------------------

TAG10_X_OFFSET: Final[float] = 4.1

TAG10_A: Final[NDArray[np.floating]] = np.array(
    [1.5689, 1.3787, 0.4243, 1.139, -0.1425, 0.01969, 0.1010],
    dtype=np.float64,
)
TAG10_A_ERR: Final[NDArray[np.floating]] = np.array(
    [0.0580, 0.0290, 0.0290, 0.240, 0.0110, 0.00190, 0.0140],
    dtype=np.float64,
)
TAG10_B: Final[NDArray[np.floating]] = np.array(
    [2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173],
    dtype=np.float64,
)
TAG10_B_ERR: Final[NDArray[np.floating]] = np.array(
    [0.0380, 0.0160, 0.0270, 0.130, 0.00750, 0.00130, 0.00820],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Santos et al. (2013) quadratic TAG10→isochrone mass correction coefficients.
# M_corr = s2 * M_TAG10^2 + s1 * M_TAG10 + s0
# (as quoted in Mortier et al. 2013, A&A 558, A106, Eq. 1; ARCHITECTURE.md §4)
# Enable/disable via config; coefficients themselves are literature-fixed.
# ---------------------------------------------------------------------------

SANTOS2013_S2: Final[float] = 0.791
SANTOS2013_S1: Final[float] = -0.575
SANTOS2013_S0: Final[float] = 0.701

# ---------------------------------------------------------------------------
# Riello et al. (2021, A&A 649, A3) corrected BP/RP flux excess factor C* (Eq. 6, Table 2)
# and its 1-sigma scatter sigma_C*(G) (Eq. 18). Literature-fixed; used for the Halbwachs
# et al. (2023) §1.2 NSS input cut |C*| < 1.645 sigma_C* (docs/MOCK_POPULATION_SPEC.md §0.1).
# C* = C - (a0 + a1 x + a2 x^2 + a3 x^3), x = BP - RP, piecewise in x.
# ---------------------------------------------------------------------------

RIELLO2021_CSTAR_X_BREAKS: Final[tuple[float, float]] = (0.5, 4.0)
RIELLO2021_CSTAR_BLUE: Final[tuple[float, float, float, float]] = (1.154360, 0.033772, 0.032277, 0.0)
RIELLO2021_CSTAR_GREEN: Final[tuple[float, float, float, float]] = (1.162004, 0.011464, 0.049255, -0.005879)
RIELLO2021_CSTAR_RED: Final[tuple[float, float, float, float]] = (1.057572, 0.140537, 0.0, 0.0)
# sigma_C*(G) = s0 + s1 * G^s2
RIELLO2021_SIGMA_CSTAR: Final[tuple[float, float, float]] = (0.0059898, 8.817481e-12, 7.618399)

# ---------------------------------------------------------------------------
# Andrae et al. (2018, A&A 616, A8) Gaia G-band bolometric correction, Eq. 7 / Table 4:
# BC_G(Teff) = sum_i a_i (Teff - TEFF_SUN)^i, separate fits for 4000-8000 K and
# 3300-4000 K (a0 of the cool fit set for continuity at 4000 K). Their solar reference
# values (Table 3): Teff = 5772 K, M_bol = 4.74 mag. Used only for the CMD radius of
# giant primaries (docs/MOCK_POPULATION_SPEC.md §10, MP-Q28).
# ---------------------------------------------------------------------------

ANDRAE2018_BCG_WARM: Final[tuple[float, float, float, float, float]] = (
    6.000e-02, 6.731e-05, -6.647e-08, 2.859e-11, -7.197e-15,
)
ANDRAE2018_BCG_COOL: Final[tuple[float, float, float, float, float]] = (
    1.749e00, 1.977e-03, 3.737e-07, -8.966e-11, -4.183e-14,
)
ANDRAE2018_BCG_TEFF_RANGE_K: Final[tuple[float, float, float]] = (3300.0, 4000.0, 8000.0)
ANDRAE2018_TEFF_SUN_K: Final[float] = 5772.0
ANDRAE2018_MBOL_SUN: Final[float] = 4.74

#: Maximum G-band brightening a single luminous companion can add (an equal-light twin):
#: 2.5 log10(2) mag. Pure arithmetic, not a choice.
TWIN_BRIGHTENING_MAG: Final[float] = 2.5 * math.log10(2.0)

#: Eggleton (1983, ApJ 268, 368) Roche-lobe radius r_L / a = A q^(2/3) / (B q^(2/3) + ln(1 + q^(1/3))),
#: q = M_donor / M_accretor; accurate to 1% for all q.
EGGLETON1983_RL: Final[tuple[float, float]] = (0.49, 0.6)

# ---------------------------------------------------------------------------
# Spectroscopic binary mass-function conversion (P in days, K in km/s → Msun).
# f = SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS * K^3 * P * (1-e^2)^{3/2}
# Derived from G + Msun via astropy — not a choosable threshold.
# ---------------------------------------------------------------------------

SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS: Final[float] = float(
    (1.0 * u.day * (1.0 * u.km / u.s) ** 3 / (2.0 * np.pi * G)).to(u.Msun).value
)

# Gaia DR3 NSS ``t_periastron`` origin (days from J2016.0 → add to this MJD).
GAIA_J2016_MJD: Final[float] = 57388.5

# ---------------------------------------------------------------------------
# Kepler's third law / orbital velocity in (AU, day, Msun) units (joint
# astrometry + RV orbit fit, #347). Derived from astropy G, Msun, au — true
# constants, not choosables.
#   a_total[AU]^3 = KEPLER_AU3_PER_MSUN_DAY2 * (M1 + M2)[Msun] * P[day]^2
#   v[km/s] = AU_PER_DAY_KMS * v[AU/day]
# ---------------------------------------------------------------------------

KEPLER_AU3_PER_MSUN_DAY2: Final[float] = float(
    (G * const.M_sun * (1.0 * u.day) ** 2 / (4.0 * np.pi**2 * const.au**3))
    .decompose()
    .value
)
AU_PER_DAY_KMS: Final[float] = float((1.0 * u.au / u.day).to_value(u.km / u.s))
