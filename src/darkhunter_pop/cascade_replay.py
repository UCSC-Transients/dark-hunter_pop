"""Replay #390 injection realizations and take apart gaiamock's cascade decision (#399, #398).

The #390 injection test stores only the *winning* branch of
``gaiamock_mod.fit_full_astrometric_cascade`` for each realization. To find out why a
published DR3 orbit came back as an accepted acceleration solution we need the
quantities the cascade compared and then threw away: the 9- and 7-parameter
significance ``s``, goodness of fit ``F2`` and parallax significance for **every**
realization, and the orbit fit the cascade never ran. This module recomputes them on
the **identical** epoch data, using only gaiamock_mod's own functions.

How the replay works
--------------------
Every #390 realization is seeded at the call boundary (``injection_test.injection_rng_seeds``
→ ``forward_model.seeded_global_rng``), so ``predict_astrometry_binary_in_terms_of_a0``
returns bit-for-bit the same epochs when called again with the same seeds. Calling it a
second time with the same seeds but ``parallax = pmra = pmdec = a0 = 0`` gives the same
transit selection, scan angles and noise draws with no signal, i.e. the realized noise
vector alone. The noise-free signal is the difference. With the two parts separated the
noise can be rescaled without touching gaiamock (:meth:`ReplayedEpochs.observations`).

Because every cascade statistic (``s``, ``F2``, ``a0/sigma_a0``, ``plx/sigma_plx``) is
invariant under a common rescaling of the observations and their stated errors,
``noise_scale = err_scale = k`` is exactly "the same truth observed with k times the
per-CCD noise", and ``k < 1`` is the same truth at higher S/N.

Limitations
-----------
* gaiamock hard-wires the acceleration thresholds inside
  ``fit_full_astrometric_cascade``. They are mirrored below **only** to label which
  condition failed; :func:`predicted_branch` is checked against gaiamock's own decision
  in the tests and, in ``scripts/diagnose_cascade_399.py``, on every stored realization.
  Nothing here changes or replaces the cascade.
* The noise-only replay assumes ``predict_astrometry_binary_in_terms_of_a0`` draws the
  same random numbers whatever the signal amplitude (true for gaiamock_mod
  ``gaiamock-mod-v1``: the 10% transit rejection, the per-source error jitter and the
  per-transit noise do not depend on a0, parallax or proper motion).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.forward_model import GlobalRNGSeeds, seeded_global_rng

#: Thresholds hard-wired in ``gaiamock_mod.fit_full_astrometric_cascade``
#: (El-Badry et al. 2024 Eqs. 12-13; Halbwachs et al. 2023 Sects. 2.2.2, 4.2, 4.3).
#: Mirrored only to label which condition decided a realization; never used to
#: accept or reject anything.
GAIAMOCK_ACCEL_SIGNIFICANCE_MIN: Final[float] = 12.0
GAIAMOCK_ACCEL_F2_MAX: Final[float] = 25.0
GAIAMOCK_ACCEL9_PLX_COEFF: Final[float] = 2.1
GAIAMOCK_ACCEL7_PLX_COEFF: Final[float] = 1.2
GAIAMOCK_ACCEL_PLX_EXPONENT: Final[float] = 1.05
#: ``fit_full_astrometric_cascade`` minimum visibility periods / observations.
GAIAMOCK_MIN_VISIBILITY_PERIODS: Final[int] = 12
GAIAMOCK_MIN_OBSERVATIONS: Final[int] = 13
#: Days between transits that start a new visibility period (gaiamock: ``> 4``).
GAIAMOCK_VISIBILITY_GAP_DAY: Final[float] = 4.0

#: Branch codes (same as ``injection_test.OUTCOME_*``).
BRANCH_INSUFFICIENT_VISIBILITY: Final[int] = 0
BRANCH_FIVE_PARAMETER: Final[int] = 5
BRANCH_SEVEN_PARAMETER: Final[int] = 7
BRANCH_NINE_PARAMETER: Final[int] = 9
BRANCH_ORBITAL: Final[int] = 12


@dataclass(frozen=True)
class ReplayedEpochs:
    """One realization's epoch astrometry split into signal and noise.

    Attributes
    ----------
    t_ast_yr, psi, plx_factor
        Transit times (yr from the DR reference epoch), scan angles (rad) and AL
        parallax factors, exactly as gaiamock returned them.
    observed
        AL displacements gaiamock returned (signal + noise), mas. Used unchanged when
        both scales are 1 so the baseline replay is bit-identical to the original.
    signal
        Noise-free AL displacement (barycentre motion + photocentre orbit), mas.
    noise
        The realized noise vector, mas (includes gaiamock's sky-dependent error term).
    ast_err
        Stated per-CCD AL error passed to the fitter, mas.
    """

    t_ast_yr: NDArray[np.float64]
    psi: NDArray[np.float64]
    plx_factor: NDArray[np.float64]
    observed: NDArray[np.float64]
    signal: NDArray[np.float64]
    noise: NDArray[np.float64]
    ast_err: NDArray[np.float64]

    def observations(
        self, *, noise_scale: float = 1.0, err_scale: float = 1.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(observations, stated errors)`` with the noise and stated errors rescaled.

        Parameters
        ----------
        noise_scale
            Multiplies the realized noise vector (> 0).
        err_scale
            Multiplies the stated per-CCD errors given to the fitter (> 0).

        Returns
        -------
        tuple
            ``signal + noise_scale * noise`` and ``err_scale * ast_err``. With both
            scales equal to 1 the original arrays are returned unchanged.
        """
        if not (noise_scale > 0 and err_scale > 0):
            raise ValueError(f"scales must be > 0, got {noise_scale=}, {err_scale=}")
        if noise_scale == 1.0 and err_scale == 1.0:
            return self.observed, self.ast_err
        return self.signal + noise_scale * self.noise, err_scale * self.ast_err


def _predict(
    gaiamock: ModuleType,
    c_funcs: Any,
    values: Mapping[str, float],
    *,
    data_release: str,
    with_signal: bool,
) -> tuple[NDArray[np.float64], ...]:
    scale = 1.0 if with_signal else 0.0
    out = gaiamock.predict_astrometry_binary_in_terms_of_a0(
        ra=float(values["ra"]),
        dec=float(values["dec"]),
        parallax=scale * float(values["parallax"]),
        pmra=scale * float(values["pmra"]),
        pmdec=scale * float(values["pmdec"]),
        period=float(values["period"]),
        Tp=float(values["t_periastron"]),
        ecc=float(values["eccentricity"]),
        omega=float(values["Omega_rad"]),
        inc=float(values["inc_rad"]),
        w=float(values["omega_rad"]),
        a0_mas=scale * float(values["a0_mas"]),
        phot_g_mean_mag=float(values["g_mag"]),
        data_release=data_release,
        c_funcs=c_funcs,
    )
    return tuple(np.asarray(x, dtype=np.float64) for x in out)


def replay_injection_epochs(
    gaiamock: ModuleType,
    c_funcs: Any,
    values: Mapping[str, float],
    seeds: GlobalRNGSeeds,
    *,
    data_release: str,
) -> ReplayedEpochs:
    """Regenerate one #390 realization's epochs and separate signal from noise.

    Parameters
    ----------
    gaiamock, c_funcs
        Imported ``gaiamock_mod`` and its C functions.
    values
        Injection truth (``injection_test.PublishedTruth.values`` or the
        ``systems/truth`` row of the #390 HDF5): ``ra``, ``dec``, ``parallax``,
        ``pmra``, ``pmdec``, ``period``, ``t_periastron``, ``eccentricity``,
        ``Omega_rad``, ``inc_rad``, ``omega_rad``, ``a0_mas``, ``g_mag``.
    seeds
        The realization's seeds (``injection_test.injection_rng_seeds``).
    data_release
        ``'dr3'`` for #390.

    Returns
    -------
    ReplayedEpochs

    Raises
    ------
    RuntimeError
        If the noise-only replay does not reproduce the same transits (the RNG
        stream diverged), which would make the signal/noise split invalid.
    """
    with seeded_global_rng(seeds, c_funcs):
        t, psi, pf, obs, err = _predict(
            gaiamock, c_funcs, values, data_release=data_release, with_signal=True
        )
    with seeded_global_rng(seeds, c_funcs):
        t0, psi0, pf0, noise, err0 = _predict(
            gaiamock, c_funcs, values, data_release=data_release, with_signal=False
        )
    if not (
        np.array_equal(t, t0)
        and np.array_equal(psi, psi0)
        and np.array_equal(pf, pf0)
        and np.array_equal(err, err0)
    ):
        raise RuntimeError("noise-only replay drew different transits; RNG stream diverged")
    return ReplayedEpochs(
        t_ast_yr=t,
        psi=psi,
        plx_factor=pf,
        observed=obs,
        signal=obs - noise,
        noise=noise,
        ast_err=err,
    )


def n_visibility_periods(t_ast_yr: NDArray[np.float64]) -> int:
    """Visibility periods as gaiamock counts them (gaps > 4 d start a new one)."""
    t = np.asarray(t_ast_yr, dtype=np.float64)
    return int(np.sum(np.diff(t * 365.25) > GAIAMOCK_VISIBILITY_GAP_DAY) + 1)


def chi2_from_f2(f2: float, nu: int) -> float:
    """Invert the Wilson-Hilferty ``F2`` (Halbwachs et al. 2023 Eq. 1) to chi^2.

    ``F2 = sqrt(9 nu / 2) [ (chi2/nu)^(1/3) + 2/(9 nu) - 1 ]``; returns chi^2. NaN when
    the inversion gives a negative cube (``F2`` below the transform's floor).
    """
    if nu <= 0 or not np.isfinite(f2):
        return math.nan
    cube = f2 * math.sqrt(2.0 / (9.0 * nu)) + 1.0 - 2.0 / (9.0 * nu)
    return nu * cube**3 if cube > 0 else math.nan


def inflation_factor_from_f2(
    f2: NDArray[np.float64] | float, nu: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """Goodness-of-fit error inflation factor ``c`` implied by a published/recovered ``F2``.

    Halbwachs et al. (2023) Eq. 2: ``c = sqrt( (chi2/nu) / (1 - 2/(9 nu))^3 )``, with
    ``chi2/nu`` from inverting Eq. 1. DR3 and gaiamock both multiply every parameter
    uncertainty by ``c`` (``get_uncertainties_at_best_fit_binary_solution``), so the
    ratio of two ``c`` values is the part of a sigma ratio that comes from the fit
    residuals rather than from the stated per-epoch errors or the number of epochs.

    Vectorized. NaN where ``nu <= 0`` or the inversion is undefined.
    """
    f2a = np.asarray(f2, dtype=np.float64)
    nua = np.asarray(nu, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = 2.0 / (9.0 * nua)
        cube = f2a * np.sqrt(w) + 1.0 - w
        chi2_red = np.where(cube > 0, cube**3, np.nan)
        c = np.sqrt(chi2_red / (1.0 - w) ** 3)
    return np.where(nua > 0, c, np.nan)


@dataclass(frozen=True)
class LinearCascadeStats:
    """What gaiamock's cascade computes before it tries an orbit.

    ``ruwe`` from ``check_ruwe``; ``s9, f2_9, plx_snr9`` from ``check_9par``;
    ``s7, f2_7, plx_snr7`` from ``check_7par``. ``n_obs`` and ``n_vis`` are the
    observation and visibility-period counts.
    """

    ruwe: float
    s9: float
    f2_9: float
    plx_snr9: float
    s7: float
    f2_7: float
    plx_snr7: float
    n_obs: int
    n_vis: int


def linear_cascade_statistics(
    gaiamock: ModuleType,
    t_ast_yr: NDArray[np.float64],
    psi: NDArray[np.float64],
    plx_factor: NDArray[np.float64],
    ast_obs: NDArray[np.float64],
    ast_err: NDArray[np.float64],
) -> LinearCascadeStats:
    """Run gaiamock's 5-, 9- and 7-parameter checks on one data set.

    All three are computed for every realization (the cascade itself stops at the
    first accepted model). Deterministic; no RNG is drawn.
    """
    ruwe, _, _ = gaiamock.check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    f2_9, s9, mu9, sig9 = gaiamock.check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    f2_7, s7, mu7, sig7 = gaiamock.check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    return LinearCascadeStats(
        ruwe=float(ruwe),
        s9=float(s9),
        f2_9=float(f2_9),
        plx_snr9=float(mu9[-1] / sig9[-1]),
        s7=float(s7),
        f2_7=float(f2_7),
        plx_snr7=float(mu7[-1] / sig7[-1]),
        n_obs=int(len(ast_obs)),
        n_vis=n_visibility_periods(t_ast_yr),
    )


def acceleration_conditions(stats: LinearCascadeStats) -> dict[str, bool]:
    """Each acceleration acceptance condition separately (El-Badry et al. 2024 Eqs. 12-13).

    Keys: ``s9``, ``f2_9``, ``plx9``, ``s7``, ``f2_7``, ``plx7``. A model is accepted
    when all three of its conditions hold.
    """
    return {
        "s9": stats.s9 > GAIAMOCK_ACCEL_SIGNIFICANCE_MIN,
        "f2_9": stats.f2_9 < GAIAMOCK_ACCEL_F2_MAX,
        "plx9": stats.plx_snr9
        > GAIAMOCK_ACCEL9_PLX_COEFF * stats.s9**GAIAMOCK_ACCEL_PLX_EXPONENT,
        "s7": stats.s7 > GAIAMOCK_ACCEL_SIGNIFICANCE_MIN,
        "f2_7": stats.f2_7 < GAIAMOCK_ACCEL_F2_MAX,
        "plx7": stats.plx_snr7
        > GAIAMOCK_ACCEL7_PLX_COEFF * stats.s7**GAIAMOCK_ACCEL_PLX_EXPONENT,
    }


def predicted_branch(stats: LinearCascadeStats, *, ruwe_min: float) -> int:
    """The branch gaiamock's cascade takes, from the linear statistics alone.

    Returns ``BRANCH_INSUFFICIENT_VISIBILITY``, ``BRANCH_FIVE_PARAMETER``,
    ``BRANCH_NINE_PARAMETER``, ``BRANCH_SEVEN_PARAMETER`` or ``BRANCH_ORBITAL`` (the
    cascade would fit an orbit). Labels only; verified against the cascade itself.
    """
    if stats.n_vis < GAIAMOCK_MIN_VISIBILITY_PERIODS or stats.n_obs < GAIAMOCK_MIN_OBSERVATIONS:
        return BRANCH_INSUFFICIENT_VISIBILITY
    if stats.ruwe < ruwe_min:
        return BRANCH_FIVE_PARAMETER
    c = acceleration_conditions(stats)
    if c["s9"] and c["f2_9"] and c["plx9"]:
        return BRANCH_NINE_PARAMETER
    if c["s7"] and c["f2_7"] and c["plx7"]:
        return BRANCH_SEVEN_PARAMETER
    return BRANCH_ORBITAL


def orbit_coverage(
    t_ast_yr: NDArray[np.float64], period_day: float, t_periastron_day: float
) -> tuple[float, float, bool]:
    """``(span_day, phase_span, periastron_in_window)`` for one realization.

    ``span_day`` is the time between the first and last transit; ``phase_span`` is
    ``span / P`` (fraction of one orbit covered, uncapped); ``periastron_in_window``
    is True when some periastron passage ``Tp + n P`` falls inside the observed span.
    Times follow gaiamock's convention (``t_ast_yr * 365.25`` days from the DR
    reference epoch, the epoch ``t_periastron`` is quoted at).
    """
    t_day = np.asarray(t_ast_yr, dtype=np.float64) * 365.25
    lo, hi = float(np.min(t_day)), float(np.max(t_day))
    span = hi - lo
    if not (period_day > 0):
        return span, math.nan, False
    n_first = math.ceil((lo - t_periastron_day) / period_day)
    return span, span / period_day, bool(t_periastron_day + n_first * period_day <= hi)


def forced_orbit_fit(
    gaiamock: ModuleType,
    c_funcs: Any,
    epochs: ReplayedEpochs,
    seeds: GlobalRNGSeeds,
    *,
    ruwe_min: float,
    noise_scale: float = 1.0,
    err_scale: float = 1.0,
) -> list[float]:
    """The orbit fit the cascade would have run had it not accepted an acceleration.

    Calls ``gaiamock.fit_full_astrometric_cascade(..., skip_acceleration=True)`` on the
    replayed observations, reseeding libc ``rand()`` with the realization's seed so
    the simulated-annealing search starts from the same stream as in #390 (where the
    orbit fit was the first ``rand()`` consumer after seeding).

    Returns
    -------
    list[float]
        The 23-element cascade vector (``injection_test.parse_cascade_result`` reads it).
    """
    obs, err = epochs.observations(noise_scale=noise_scale, err_scale=err_scale)
    with seeded_global_rng(seeds, c_funcs):
        res = gaiamock.fit_full_astrometric_cascade(
            t_ast_yr=epochs.t_ast_yr,
            psi=epochs.psi,
            plx_factor=epochs.plx_factor,
            ast_obs=obs,
            ast_err=err,
            c_funcs=c_funcs,
            verbose=False,
            show_residuals=False,
            ruwe_min=ruwe_min,
            skip_acceleration=True,
        )
    return [float(x) for x in res]
