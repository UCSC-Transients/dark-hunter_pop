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
