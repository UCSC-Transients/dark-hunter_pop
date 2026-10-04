"""Step 1a injection test: re-inject published DR3 orbits through gaiamock_mod (#390).

The test validates ``gaiamock_mod`` independently of any population model. Each
published Gaia DR3 ``Orbital`` / ``AstroSpectroSB1`` solution is taken as the truth
(photocentre orbit, position, parallax, proper motion, G), its epoch astrometry is
simulated with ``gaiamock_mod.predict_astrometry_binary_in_terms_of_a0`` at the real
sky position (DR3 scanning law, unbinned CCD-level noise, sky-dependent error
rescaling), and the result is refit with ``gaiamock_mod.fit_full_astrometric_cascade``.
The recovered solution is then compared with the published one.

Design decisions
----------------
* **Photocentre a0 is injected directly.** ``gaiamock_mod`` exposes
  ``predict_astrometry_binary_in_terms_of_a0``, so no M1 / M2 / flux-ratio choice is
  needed. That function is the photocentre approximation; for ``f = 0`` the
  luminous-binary path (``al_bias_binary``) reduces to exactly the same linear
  photocentre shift, so the two are equivalent for dark or faint companions.
* **Campbell elements come from gaiamock.** Published Thiele-Innes (A, B, F, G) are
  converted with ``gaiamock_mod.get_Campbell_elements``; nothing here reimplements
  Kepler, the scanning law, the noise model or the cascade (``docs/GAIAMOCK_API.md``).
* **Published sigma_a0** is ``a0 / significance`` (DR3 defines ``significance`` as
  ``a0 / sigma_a0`` for astrometric orbits). ``sigma_cos_i`` is the linear
  propagation of the published **full** A/B/F/G covariance (``nss_covariance``);
  rows whose covariance cannot be rebuilt get NaN, never a diagonal substitute.
* **Seeding** reuses the #371 boundary (``forward_model.seeded_global_rng``): every
  (source, realization) pair gets numpy + libc ``rand()`` seeds from
  ``SeedSequence(entropy=base_seed, spawn_key=(INJECTION_RNG_STREAM, source_id,
  realization))``, so a realization replays bit-for-bit regardless of worker
  count, order, or which other systems ran.
* **Cuts** are ``forward_model.passes_orbital_solution_cuts`` with the active DR's
  ``orbital_solution_cuts`` (El-Badry et al. 2024 Eq. 18 + Eqs. 20-22).

Limitations
-----------
* The fit is astrometry-only, so for ``AstroSpectroSB1`` (published with RVs) the
  recovered uncertainties are expected to exceed the published ones.
* ``gaiamock_mod`` returns only diagonal uncertainties of the 12-parameter fit, so
  the recovered ``cos i`` has no recovered sigma; its pull uses the published
  (full-covariance) ``sigma_cos_i``.
* ``fit_full_astrometric_cascade`` bounds the period search to
  ``[P_min = 10 d, 10^4 d]`` and ``e < 0.99``; published orbits outside cannot be
  recovered by construction (flagged ``outside_fit_bounds``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_schema import OrbitalSolutionCutsConfig
from darkhunter_pop.forward_model import (
    GlobalRNGSeeds,
    passes_orbital_solution_cuts,
    seeded_global_rng,
)
from darkhunter_pop.nss_covariance import reconstruct_nss_covariance

#: ``SeedSequence`` spawn-key stream for injection realizations. Distinct from
#: ``forward_model.MOCK_RNG_STREAM_SKY`` (0) and ``MOCK_RNG_STREAM_REALIZATION`` (1).
INJECTION_RNG_STREAM: Final[int] = 2

#: Seeding scheme string recorded in the output HDF5.
INJECTION_RNG_SEED_SCHEME: Final[str] = (
    "numpy.random.SeedSequence(entropy=base_seed, spawn_key=("
    f"{INJECTION_RNG_STREAM}, source_id, realization)).generate_state(2, uint32) -> "
    "(np.random.seed, libc srand) via forward_model.seeded_global_rng around "
    "predict_astrometry_binary_in_terms_of_a0 + fit_full_astrometric_cascade (#371, #390)"
)

#: Bounds hard-wired inside ``gaiamock_mod.fit_full_astrometric_cascade`` (``P_min``
#: default) and ``fit_orbital_solution_nonlinear`` (``U``). Mirrored here only to
#: *flag* published orbits the fitter cannot reach; they are not used in any fit.
GAIAMOCK_FIT_PERIOD_MIN_DAY: Final[float] = 10.0
GAIAMOCK_FIT_PERIOD_MAX_DAY: Final[float] = 1.0e4
GAIAMOCK_FIT_ECC_MAX: Final[float] = 0.99

#: Outcome codes per realization.
OUTCOME_INSUFFICIENT_VISIBILITY: Final[int] = 0
OUTCOME_FIVE_PARAMETER: Final[int] = 5
OUTCOME_SEVEN_PARAMETER: Final[int] = 7
OUTCOME_NINE_PARAMETER: Final[int] = 9
OUTCOME_ORBITAL: Final[int] = 12

#: Days between two transits that start a new visibility period, as used by
#: ``gaiamock_mod.fit_full_astrometric_cascade`` (``np.diff(t*365.25) > 4``). Used
#: only to report ``N_visibility_periods`` for realizations that stop before the
#: orbital branch (gaiamock reports it only for orbital fits).
GAIAMOCK_VISIBILITY_GAP_DAY: Final[float] = 4.0

#: Columns of the published-truth table (all float64 except where noted).
TRUTH_COLUMNS: Final[tuple[str, ...]] = (
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "period",
    "period_error",
    "t_periastron",
    "eccentricity",
    "eccentricity_error",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "significance",
    "goodness_of_fit",
    "ruwe",
    "g_mag",
    "a0_mas",
    "sigma_a0_mas",
    "Omega_rad",
    "omega_rad",
    "inc_rad",
    "cos_i",
    "sigma_cos_i",
    "sigma_a0_cov_mas",
)

#: Recovered-solution columns per realization (float64).
RECOVERED_COLUMNS: Final[tuple[str, ...]] = (
    "parallax",
    "parallax_error",
    "period",
    "period_error",
    "eccentricity",
    "eccentricity_error",
    "a0_mas",
    "sigma_a0_mas",
    "inc_deg",
    "cos_i",
    "significance",
    "goodness_of_fit",
    "ruwe",
    "n_visibility_periods",
    "n_obs",
    "acceleration_significance",
)

#: Individual cut flags recorded per realization (same thresholds as
#: ``passes_orbital_solution_cuts``; the AND of these must equal ``accepted``).
CUT_FLAG_NAMES: Final[tuple[str, ...]] = (
    "cut_a0_over_err",
    "cut_f2",
    "cut_parallax_over_error",
    "cut_a0_over_err_sqrt_p",
    "cut_sigma_e",
)


@dataclass(frozen=True)
class PublishedTruth:
    """Truth vector for one published DR3 solution, ready for injection.

    Attributes
    ----------
    source_id, nss_solution_type
        DR3 identifiers.
    values
        Mapping over ``TRUTH_COLUMNS``. Angles in radians, a0 in mas, period and
        ``t_periastron`` in days (``t_periastron`` relative to J2016.0, the
        convention both DR3 and ``gaiamock_mod.rescale_times_astrometry`` use).
    """

    source_id: int
    nss_solution_type: str
    values: Mapping[str, float]


def injection_rng_seeds(base_seed: int, source_id: int, realization: int) -> GlobalRNGSeeds:
    """Deterministic numpy / libc seeds for one (source, realization) injection.

    Parameters
    ----------
    base_seed
        Base entropy (default: ``selection_function_astrometric.mock_population.random_seed``).
    source_id
        Gaia DR3 ``source_id`` (non-negative 64-bit integer).
    realization
        Noise-realization index, ``0 .. n_realizations - 1``.

    Returns
    -------
    GlobalRNGSeeds
        Two uint32 words from ``SeedSequence(base_seed, spawn_key=(INJECTION_RNG_STREAM,
        source_id, realization))``. Independent of order, worker count, or sample.
    """
    if base_seed < 0 or source_id < 0 or realization < 0:
        raise ValueError(
            f"seed components must be non-negative, got {base_seed=}, {source_id=}, "
            f"{realization=}"
        )
    words = np.random.SeedSequence(
        entropy=int(base_seed),
        spawn_key=(INJECTION_RNG_STREAM, int(source_id), int(realization)),
    ).generate_state(2, dtype=np.uint32)
    return GlobalRNGSeeds(numpy_seed=int(words[0]), c_rand_seed=int(words[1]))


def _finite_float(value: Any) -> float:
    """``float(value)`` with masked / empty / non-numeric values mapped to NaN."""
    if value is None:
        return math.nan
    try:
        if np.ma.is_masked(value):
            return math.nan
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def campbell_from_thiele_innes(
    gaiamock: ModuleType, a: float, b: float, f: float, g: float
) -> tuple[float, float, float, float]:
    """``(a0_mas, Omega_rad, omega_rad, inc_rad)`` via ``gaiamock.get_Campbell_elements``."""
    a0, big_omega, small_omega, inc = gaiamock.get_Campbell_elements(
        float(a), float(b), float(f), float(g)
    )
    return float(a0), float(big_omega), float(small_omega), float(inc)


def propagate_thiele_innes_sigmas(
    gaiamock: ModuleType,
    abfg: NDArray[np.float64],
    cov_abfg: NDArray[np.float64],
    *,
    rel_step: float = 1.0e-6,
) -> tuple[float, float]:
    """Linear propagation of the A/B/F/G covariance into ``(sigma_a0, sigma_cos_i)``.

    The Jacobian of ``(a0, cos i)`` with respect to ``(A, B, F, G)`` is taken by
    central finite differences of ``gaiamock.get_Campbell_elements`` (no Campbell
    algebra is reimplemented here).

    Parameters
    ----------
    abfg
        Published ``(A, B, F, G)`` in mas.
    cov_abfg
        Their 4x4 covariance (mas^2), from the full NSS covariance.
    rel_step
        Finite-difference step as a fraction of ``max |A,B,F,G|``.

    Returns
    -------
    tuple[float, float]
        ``(sigma_a0_mas, sigma_cos_i)``; NaN if the covariance is not finite.

    Limitations
    -----------
    First-order only: near face-on (|cos i| -> 1) cos i is non-linear in A/B/F/G and
    the linear sigma under-states the true spread.
    """
    x0 = np.asarray(abfg, dtype=np.float64)
    cov = np.asarray(cov_abfg, dtype=np.float64)
    if x0.shape != (4,) or cov.shape != (4, 4) or not np.all(np.isfinite(cov)):
        return math.nan, math.nan
    h = rel_step * max(float(np.max(np.abs(x0))), 1.0e-12)
    jac = np.zeros((2, 4), dtype=np.float64)
    for k in range(4):
        up = x0.copy()
        dn = x0.copy()
        up[k] += h
        dn[k] -= h
        a_up, _, _, i_up = campbell_from_thiele_innes(gaiamock, *up)
        a_dn, _, _, i_dn = campbell_from_thiele_innes(gaiamock, *dn)
        jac[0, k] = (a_up - a_dn) / (2.0 * h)
        jac[1, k] = (math.cos(i_up) - math.cos(i_dn)) / (2.0 * h)
    out_cov = jac @ cov @ jac.T
    var = np.diag(out_cov)
    sig = np.where(var > 0, np.sqrt(np.clip(var, 0.0, None)), np.nan)
    return float(sig[0]), float(sig[1])


_ABFG_NAMES: Final[tuple[str, ...]] = (
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
)


def published_truth_from_row(
    row: Mapping[str, Any], gaiamock: ModuleType
) -> PublishedTruth:
    """Build the injection truth for one merged DR3 row.

    Parameters
    ----------
    row
        Mapping with the published columns: ``source_id``, ``nss_solution_type``,
        ``ra``, ``dec``, ``parallax``, ``parallax_error``, ``pmra``, ``pmdec``,
        ``period``, ``period_error``, ``t_periastron``, ``eccentricity``,
        ``eccentricity_error``, ``a/b/f/g_thiele_innes`` (+ ``_error``),
        ``significance``, ``goodness_of_fit``, ``ruwe``, ``g_mag``, plus
        ``corr_vec`` / ``bit_index`` for the covariance.
    gaiamock
        Imported ``gaiamock_mod``.

    Returns
    -------
    PublishedTruth
        Truth values; ``sigma_cos_i`` / ``sigma_a0_cov_mas`` are NaN when the NSS
        covariance cannot be rebuilt (never replaced by a diagonal).
    """
    vals: dict[str, float] = {}
    for name in TRUTH_COLUMNS:
        if name in row:
            vals[name] = _finite_float(row[name])
    a, b, f, g = (vals[n] for n in _ABFG_NAMES)
    a0, big_omega, small_omega, inc = campbell_from_thiele_innes(gaiamock, a, b, f, g)
    vals["a0_mas"] = a0
    vals["Omega_rad"] = big_omega
    vals["omega_rad"] = small_omega
    vals["inc_rad"] = inc
    vals["cos_i"] = math.cos(inc)
    sig = vals.get("significance", math.nan)
    vals["sigma_a0_mas"] = a0 / sig if (np.isfinite(sig) and sig > 0) else math.nan

    sol_type = str(row.get("nss_solution_type", "")).strip('"')
    sigma_a0_cov = math.nan
    sigma_cos_i = math.nan
    cov_res = reconstruct_nss_covariance(row, nss_solution_type=sol_type)
    if cov_res.parameter_set is not None:
        names = list(cov_res.parameter_set.names)
        if all(n in names for n in _ABFG_NAMES):
            idx = [names.index(n) for n in _ABFG_NAMES]
            cov_full = np.asarray(cov_res.parameter_set.covariance, dtype=np.float64)
            cov4 = cov_full[np.ix_(idx, idx)]
            sigma_a0_cov, sigma_cos_i = propagate_thiele_innes_sigmas(
                gaiamock, np.array([a, b, f, g]), cov4
            )
    vals["sigma_a0_cov_mas"] = sigma_a0_cov
    vals["sigma_cos_i"] = sigma_cos_i
    for name in TRUTH_COLUMNS:
        vals.setdefault(name, math.nan)
    return PublishedTruth(
        source_id=int(row["source_id"]),
        nss_solution_type=sol_type,
        values=vals,
    )


def thiele_innes_roundtrip_error(values: Mapping[str, float]) -> float:
    """Max |A,B,F,G (rebuilt) - A,B,F,G (published)| in mas, a convention check.

    Rebuilds the Thiele-Innes constants from the Campbell elements returned by
    ``gaiamock.get_Campbell_elements`` with the same expressions
    ``gaiamock_mod.predict_astrometry_binary_in_terms_of_a0`` uses to place the
    photocentre, so a near-zero value proves the injected orbit **is** the
    published one (sign, quadrant and axis conventions included). Verification only;
    never used to simulate.
    """
    a0 = values["a0_mas"]
    big_omega, small_omega, inc = values["Omega_rad"], values["omega_rad"], values["inc_rad"]
    cw, sw = math.cos(small_omega), math.sin(small_omega)
    cO, sO = math.cos(big_omega), math.sin(big_omega)
    ci = math.cos(inc)
    rebuilt = (
        a0 * (cw * cO - sw * sO * ci),
        a0 * (cw * sO + sw * cO * ci),
        -a0 * (sw * cO + cw * sO * ci),
        -a0 * (sw * sO - cw * cO * ci),
    )
    published = tuple(values[n] for n in _ABFG_NAMES)
    return float(max(abs(r - p) for r, p in zip(rebuilt, published)))


def outside_fit_bounds(period_day: float, eccentricity: float) -> bool:
    """True when gaiamock's orbit search cannot reach this published (P, e)."""
    return bool(
        not (GAIAMOCK_FIT_PERIOD_MIN_DAY <= period_day <= GAIAMOCK_FIT_PERIOD_MAX_DAY)
        or not (eccentricity < GAIAMOCK_FIT_ECC_MAX)
    )


def individual_cut_flags(
    *,
    a0_over_err: float,
    parallax_over_error: float,
    period_days: float,
    sigma_ecc: float,
    goodness_of_fit_f2: float,
    cuts: OrbitalSolutionCutsConfig,
) -> dict[str, bool]:
    """Per-cut pass flags for the diagnostic waterfall (same thresholds as acceptance).

    Acceptance itself is always ``forward_model.passes_orbital_solution_cuts``; these
    flags only say *which* cut failed. ``period_days <= 0`` fails every cut.
    """
    if not (period_days > 0):
        return {name: False for name in CUT_FLAG_NAMES}
    return {
        "cut_a0_over_err": bool(a0_over_err > cuts.a0_over_err_min),
        "cut_f2": bool(goodness_of_fit_f2 < cuts.goodness_of_fit_f2_max),
        "cut_parallax_over_error": bool(
            parallax_over_error > cuts.parallax_over_error_times_period_min_days / period_days
        ),
        "cut_a0_over_err_sqrt_p": bool(
            a0_over_err > cuts.a0_over_err_times_sqrt_period_min / math.sqrt(period_days)
        ),
        "cut_sigma_e": bool(
            sigma_ecc
            < cuts.sigma_e_ln_period_slope * math.log(period_days) + cuts.sigma_e_intercept
        ),
    }


def parse_cascade_result(
    cascade: Sequence[float],
    *,
    n_visibility_periods: int,
    n_obs: int,
    cuts: OrbitalSolutionCutsConfig,
) -> dict[str, Any]:
    """Turn a ``fit_full_astrometric_cascade`` vector into a recovered-solution record.

    Return layout (gaiamock_mod, orbital branch): plx, sig_parallax, A, sig_A, B,
    sig_B, F, sig_F, G, sig_G, period, sig_period, phi_p, sig_phi_p, ecc, sig_ecc,
    inc_deg, a0_mas, sigma_a0_mas, N_visibility_periods, N_obs, F2, ruwe. Non-orbital
    branches are flagged by ``plx`` in {0, -1, -7, -9}; their RUWE sits at index 1
    (5-par), 8 (7-par) or 12 (9-par).

    Parameters
    ----------
    cascade
        The 23-element cascade vector.
    n_visibility_periods, n_obs
        Counts from the simulated epochs (used for non-orbital branches).
    cuts
        Active DR ``orbital_solution_cuts``.

    Returns
    -------
    dict
        ``outcome`` (int code), ``accepted`` (bool), the ``RECOVERED_COLUMNS``
        (NaN where the branch does not define them) and the ``CUT_FLAG_NAMES``.
    """
    res = [float(x) for x in cascade]
    if len(res) < 23:
        res = res + [0.0] * (23 - len(res))
    rec: dict[str, Any] = {name: math.nan for name in RECOVERED_COLUMNS}
    rec.update({name: False for name in CUT_FLAG_NAMES})
    rec["n_visibility_periods"] = float(n_visibility_periods)
    rec["n_obs"] = float(n_obs)
    rec["accepted"] = False
    plx = res[0]
    if plx == 0.0:
        rec["outcome"] = OUTCOME_INSUFFICIENT_VISIBILITY
        return rec
    if np.isclose(plx, -1.0):
        rec["outcome"] = OUTCOME_FIVE_PARAMETER
        rec["ruwe"] = res[1]
        rec["parallax"] = res[2]
        rec["parallax_error"] = res[3]
        return rec
    if np.isclose(plx, -7.0):
        rec["outcome"] = OUTCOME_SEVEN_PARAMETER
        rec["acceleration_significance"] = res[1]
        rec["parallax"], rec["parallax_error"] = res[2], res[3]
        rec["ruwe"] = res[8]
        rec["goodness_of_fit"] = res[9]
        return rec
    if np.isclose(plx, -9.0):
        rec["outcome"] = OUTCOME_NINE_PARAMETER
        rec["acceleration_significance"] = res[1]
        rec["parallax"], rec["parallax_error"] = res[2], res[3]
        rec["ruwe"] = res[12]
        rec["goodness_of_fit"] = res[13]
        return rec
    rec["outcome"] = OUTCOME_ORBITAL
    rec["parallax"], rec["parallax_error"] = res[0], res[1]
    rec["period"], rec["period_error"] = res[10], res[11]
    rec["eccentricity"], rec["eccentricity_error"] = res[14], res[15]
    rec["inc_deg"] = res[16]
    rec["cos_i"] = math.cos(math.radians(res[16])) if np.isfinite(res[16]) else math.nan
    rec["a0_mas"], rec["sigma_a0_mas"] = res[17], res[18]
    rec["n_visibility_periods"], rec["n_obs"] = res[19], res[20]
    rec["goodness_of_fit"], rec["ruwe"] = res[21], res[22]
    a0_over_err = res[17] / res[18] if res[18] > 0 else math.nan
    plx_over_err = res[0] / res[1] if res[1] > 0 else math.nan
    rec["significance"] = a0_over_err
    rec.update(
        individual_cut_flags(
            a0_over_err=a0_over_err,
            parallax_over_error=plx_over_err,
            period_days=res[10],
            sigma_ecc=res[15],
            goodness_of_fit_f2=res[21],
            cuts=cuts,
        )
    )
    rec["accepted"] = bool(
        res[10] > 0
        and res[1] > 0
        and res[18] > 0
        and passes_orbital_solution_cuts(
            a0_over_err=a0_over_err,
            parallax_over_error=plx_over_err,
            period_days=res[10],
            sigma_ecc=res[15],
            goodness_of_fit_f2=res[21],
            cuts=cuts,
        )
    )
    return rec


def inject_one_realization(
    gaiamock: ModuleType,
    c_funcs: Any,
    truth: PublishedTruth,
    *,
    realization: int,
    base_seed: int,
    cuts: OrbitalSolutionCutsConfig,
    data_release: str,
    ruwe_min: float,
    skip_acceleration: bool,
) -> dict[str, Any]:
    """Simulate and refit one noise realization of one published orbit.

    Calls ``gaiamock.predict_astrometry_binary_in_terms_of_a0`` (truth = published
    photocentre orbit at the real position, parallax, proper motion and G) and then
    ``gaiamock.fit_full_astrometric_cascade``, both inside
    ``forward_model.seeded_global_rng`` with :func:`injection_rng_seeds`.

    Returns
    -------
    dict
        :func:`parse_cascade_result` output plus ``source_id``, ``realization``,
        ``rng_seed_numpy``, ``rng_seed_c_rand``.
    """
    v = truth.values
    seeds = injection_rng_seeds(base_seed, truth.source_id, realization)
    with seeded_global_rng(seeds, c_funcs):
        t_ast_yr, psi, plx_factor, ast_obs, ast_err = (
            gaiamock.predict_astrometry_binary_in_terms_of_a0(
                ra=v["ra"],
                dec=v["dec"],
                parallax=v["parallax"],
                pmra=v["pmra"],
                pmdec=v["pmdec"],
                period=v["period"],
                Tp=v["t_periastron"],
                ecc=v["eccentricity"],
                omega=v["Omega_rad"],
                inc=v["inc_rad"],
                w=v["omega_rad"],
                a0_mas=v["a0_mas"],
                phot_g_mean_mag=v["g_mag"],
                data_release=data_release,
                c_funcs=c_funcs,
            )
        )
        n_vis = int(np.sum(np.diff(t_ast_yr * 365.25) > GAIAMOCK_VISIBILITY_GAP_DAY) + 1)
        cascade = gaiamock.fit_full_astrometric_cascade(
            t_ast_yr=t_ast_yr,
            psi=psi,
            plx_factor=plx_factor,
            ast_obs=ast_obs,
            ast_err=ast_err,
            c_funcs=c_funcs,
            verbose=False,
            show_residuals=False,
            ruwe_min=ruwe_min,
            skip_acceleration=skip_acceleration,
        )
    rec = parse_cascade_result(
        cascade, n_visibility_periods=n_vis, n_obs=len(t_ast_yr), cuts=cuts
    )
    rec["source_id"] = truth.source_id
    rec["realization"] = int(realization)
    rec["rng_seed_numpy"] = seeds.numpy_seed
    rec["rng_seed_c_rand"] = seeds.c_rand_seed
    return rec


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def stratified_sample_indices(
    strata_values: Mapping[str, NDArray[np.float64]],
    *,
    n_bins: int,
    n_total: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Equal-allocation stratified sample over quantile bins of several variables.

    Parameters
    ----------
    strata_values
        Name -> per-row values (same length). Rows with any non-finite value are
        never selected.
    n_bins
        Quantile bins per variable (cells = ``n_bins ** len(strata_values)``).
    n_total
        Requested sample size; each non-empty cell gets ``ceil(n_total / n_cells)``
        rows, or all of them if it has fewer. The result may differ slightly from
        ``n_total``.
    seed
        Within each cell the members are put in a fixed random order from
        ``default_rng(SeedSequence(seed, spawn_key=(cell,)))`` and the first ``k``
        are taken. A smaller ``n_total`` with the same seed and strata is therefore
        always a **subset** of a larger one (pilot rows are reused by the scale-up).

    Returns
    -------
    indices
        Selected row indices, sorted.
    weights
        Population weight per selected row: ``N_cell / n_cell_selected``, so a
        weighted mean over the sample estimates the population mean.
    edges
        Name -> quantile bin edges used.
    """
    names = list(strata_values)
    if not names:
        raise ValueError("need at least one stratification variable")
    arrays = [np.asarray(strata_values[n], dtype=np.float64) for n in names]
    n_rows = len(arrays[0])
    finite = np.ones(n_rows, dtype=bool)
    for arr in arrays:
        if len(arr) != n_rows:
            raise ValueError("strata arrays differ in length")
        finite &= np.isfinite(arr)
    edges: dict[str, NDArray[np.float64]] = {}
    cell = np.zeros(n_rows, dtype=np.int64)
    for name, arr in zip(names, arrays):
        e = np.quantile(arr[finite], np.linspace(0.0, 1.0, n_bins + 1))
        edges[name] = e
        b = np.clip(np.searchsorted(e, arr, side="right") - 1, 0, n_bins - 1)
        cell = cell * n_bins + b
    cell[~finite] = -1
    n_cells = n_bins ** len(names)
    per_cell = int(math.ceil(n_total / n_cells))
    chosen: list[NDArray[np.int64]] = []
    weights: list[NDArray[np.float64]] = []
    for c in range(n_cells):
        members = np.flatnonzero(cell == c)
        if len(members) == 0:
            continue
        k = min(per_cell, len(members))
        cell_rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(c,)))
        pick = cell_rng.permutation(members)[:k]
        chosen.append(pick.astype(np.int64))
        weights.append(np.full(k, len(members) / k, dtype=np.float64))
    idx = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    w = np.concatenate(weights) if weights else np.zeros(0, dtype=np.float64)
    order = np.argsort(idx)
    return idx[order], w[order], edges


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PullSummary:
    """Robust and moment summaries of one pull distribution.

    ``median`` and ``sigma_mad`` (1.4826 x MAD) are robust to period aliases;
    ``mean`` / ``std`` use only |pull| < ``clip``; ``frac_outlier`` is the fraction
    with |pull| >= ``clip``.
    """

    n: int
    median: float
    sigma_mad: float
    mean: float
    std: float
    frac_outlier: float
    clip: float


def summarize_pulls(pulls: Iterable[float], *, clip: float = 5.0) -> PullSummary:
    """Summaries of finite pulls; empty input gives NaNs and ``n = 0``."""
    x = np.asarray(list(pulls), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return PullSummary(0, math.nan, math.nan, math.nan, math.nan, math.nan, clip)
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    inl = x[np.abs(x) < clip]
    mean = float(np.mean(inl)) if len(inl) else math.nan
    std = float(np.std(inl, ddof=1)) if len(inl) > 1 else math.nan
    return PullSummary(
        n=int(len(x)),
        median=med,
        sigma_mad=mad,
        mean=mean,
        std=std,
        frac_outlier=float(np.mean(np.abs(x) >= clip)),
        clip=clip,
    )


def binomial_fraction_interval(k: int, n: int, *, z: float = 1.0) -> tuple[float, float, float]:
    """Wilson score interval ``(fraction, lo, hi)`` for k successes in n trials.

    ``z = 1`` gives a ~68% interval. ``n = 0`` returns NaNs.
    """
    if n <= 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)
