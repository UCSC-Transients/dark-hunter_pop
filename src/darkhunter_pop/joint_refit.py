"""Joint luminous + compact-object rung-3 refit helpers (#561; docs/MOCK_POPULATION_SPEC.md §12.37).

Reweight-only pieces of the #554 fixes and the compact-object expectation:

* :func:`flux_scatter_log_ratio` (F3): q-dependent flux-ratio scatter for coeval main-sequence pairs;
* :func:`inflate_low_ess_variance` (F2): MC variance inflated where a bin's ESS is below a floor;
* :func:`depletion_log_factor` (F6): looks-single depletion, exp(+Λ_ns) per parent (G, d) cell;
* :func:`nonsingle_flags` (F6): which mock systems a companion makes look non-single;
* :func:`twin_band_loglike` / :func:`twin_band_masks` (F4): binomial twin-band occupancy term;
* :func:`photocentre_size` (F1): A = (a0/ϖ) / P^(2/3);
* :func:`co_observables` (compact objects on the luminous fit axes).

Statistics only; no gaiamock call and no new draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

FloatArray = NDArray[np.float64]


class FluxScatterConfig(BaseModel):
    """F3: σ_f(q) = ``sigma_0`` for q ≤ ``q_taper_start``, linear to ``sigma_min`` at q = 1 (coeval MS pairs only)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    sigma_0: float = Field(..., gt=0)
    sigma_min: float = Field(..., gt=0)
    q_taper_start: float = Field(..., gt=0, lt=1)


class TwinBandConfig(BaseModel):
    """F4: system-CMD bands (ΔM from the MS ridge) and the distance bins of the twin-occupancy term."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    colour0_range: tuple[float, float]
    twin_dm: tuple[float, float]
    single_dm: tuple[float, float]
    distance_edges_kpc: tuple[float, ...]


class DepletionConfig(BaseModel):
    """F6: (G, d) cells of the looks-single depletion factor and the non-single criteria (§12.33 looks-single cuts)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    g_edges: tuple[float, ...]
    distance_edges_kpc: tuple[float, ...]
    ruwe_max: float = Field(..., gt=0)
    dm_min: float
    dm_colour_range: tuple[float, float]


def flux_scatter_sigma(q: ArrayLike, cfg: FluxScatterConfig) -> FloatArray:
    """σ_f(q) of the F3 law."""
    qq = np.asarray(q, float)
    t = np.clip((qq - cfg.q_taper_start) / (1.0 - cfg.q_taper_start), 0.0, 1.0)
    return cfg.sigma_0 + t * (cfg.sigma_min - cfg.sigma_0)


def flux_scatter_log_ratio(
    log10_f: ArrayLike, relation_log10_f: ArrayLike, q: ArrayLike, evolved: ArrayLike, cfg: FluxScatterConfig,
) -> FloatArray:
    """log [N(log f; rel, σ_f(q)) / N(log f; rel, σ_0)] per draw (F3); 0 for evolved systems, dark or undefined draws,
    and when disabled. Both densities are normalized in log f, so the companion count is unchanged."""
    lf = np.asarray(log10_f, float)
    rel = np.asarray(relation_log10_f, float)
    out = np.zeros(lf.size)
    if not cfg.enabled:
        return out
    ok = np.isfinite(lf) & np.isfinite(rel) & ~np.asarray(evolved, bool)
    s1 = flux_scatter_sigma(np.asarray(q, float), cfg)
    d = np.where(ok, lf - rel, 0.0)
    out = np.where(ok, -np.log(s1 / cfg.sigma_0) - 0.5 * d**2 * (1.0 / s1**2 - 1.0 / cfg.sigma_0**2), 0.0)
    return out


def inflate_low_ess_variance(mu: ArrayLike, var: ArrayLike, ess_min: float) -> FloatArray:
    """F2: var × max(1, ess_min / ESS_b) with ESS_b = μ² / var (bins with var = 0 unchanged)."""
    m, v = np.asarray(mu, float), np.asarray(var, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(v > 0, m**2 / v, np.inf)
    return v * np.maximum(1.0, ess_min / np.maximum(ess, 1e-300))


def nonsingle_flags(
    solution_type: ArrayLike, cascade: ArrayLike, accepted: ArrayLike, published_acceleration: ArrayLike,
    sys_mg0: ArrayLike, sys_colour0: ArrayLike, sys_evolved: ArrayLike, ridge: Mapping[str, Sequence[float]],
    cfg: DepletionConfig,
) -> NDArray[np.bool_]:
    """F6: systems a companion makes look non-single under the §12.33 cuts: a published orbit or acceleration, a
    5-parameter RUWE ≥ ``ruwe_max`` (gaiamock's res[1]), any later cascade stage (its RUWE passed ruwe_min > ruwe_max),
    or a non-evolved system with ΔM < ``dm_min`` in the ``dm_colour_range``. Insufficient visibility or out of parent: no."""
    st = np.asarray(solution_type).astype(str)
    casc = np.asarray(cascade, float)
    later = np.isin(st, ["seven_parameter", "nine_parameter", "orbital_failed_cuts", "twelve_parameter_orbital"])
    ruwe5 = np.where(st == "five_parameter", casc[:, 1], -np.inf)
    c0 = np.asarray(sys_colour0, float)
    dm = np.asarray(sys_mg0, float) - np.interp(c0, ridge["colour"], ridge["mag"], left=np.nan, right=np.nan)
    lo, hi = cfg.dm_colour_range
    with np.errstate(invalid="ignore"):
        cmd = ~np.asarray(sys_evolved, bool) & (c0 >= lo) & (c0 <= hi) & (dm < cfg.dm_min)
    return (np.asarray(accepted, bool) | np.asarray(published_acceleration, bool) | later | (ruwe5 >= cfg.ruwe_max) | cmd) & (st != "not_in_parent")


def depletion_log_factor(
    w_base: ArrayLike, nonsingle: ArrayLike, draw_rows: ArrayLike, row_g: ArrayLike, row_d_kpc: ArrayLike,
    row_usable: ArrayLike, row_weight: ArrayLike | None, scale_to_full: float, cfg: DepletionConfig,
) -> tuple[FloatArray, dict[str, Any]]:
    """F6: per parent row, Λ_ns = expected non-single companions per row in its (G, d) cell (base weights over all
    draws ÷ the cell's expanded row count), and log factor +Λ_ns (= −log P(looks single)). Returns the per-row log
    factor (0 outside the cells or when disabled) and a per-cell summary."""
    g, d = np.asarray(row_g, float), np.asarray(row_d_kpc, float)
    nrow = g.size
    out = np.zeros(nrow)
    summ: dict[str, Any] = {}
    if not cfg.enabled:
        return out, summ
    rw = np.ones(nrow) if row_weight is None else np.asarray(row_weight, float)
    use = np.asarray(row_usable, bool)
    gi = np.searchsorted(cfg.g_edges, g, side="right") - 1
    di = np.searchsorted(cfg.distance_edges_kpc, d, side="right") - 1
    ng, nd = len(cfg.g_edges) - 1, len(cfg.distance_edges_kpc) - 1
    inside = (gi >= 0) & (gi < ng) & (di >= 0) & (di < nd)
    cell = np.where(inside, gi * nd + np.clip(di, 0, nd - 1), -1)
    n_cells = ng * nd
    expanded = np.bincount(cell[use & inside], weights=rw[use & inside], minlength=n_cells) * float(scale_to_full)
    r = np.asarray(draw_rows, np.int64)
    dc = cell[r]
    s = (dc >= 0) & np.asarray(nonsingle, bool)
    lam_num = np.bincount(dc[s], weights=np.asarray(w_base, float)[s], minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(expanded > 0, lam_num / expanded, 0.0)
    out = np.where(cell >= 0, lam[np.clip(cell, 0, n_cells - 1)], 0.0)
    for k in range(n_cells):
        if expanded[k] > 0:
            summ[f"G{cfg.g_edges[k // nd]:g}-{cfg.g_edges[k // nd + 1]:g} d{cfg.distance_edges_kpc[k % nd]:g}-{cfg.distance_edges_kpc[k % nd + 1]:g}"] = {
                "lambda_ns": float(lam[k]), "p_looks_single": float(math.exp(-lam[k]))}
    return out, summ


def twin_band_masks(sys_colour0: ArrayLike, sys_mg0: ArrayLike, ridge: Mapping[str, Sequence[float]], cfg: TwinBandConfig) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(twin, single) system-CMD band membership (F4)."""
    c0 = np.asarray(sys_colour0, float)
    dm = np.asarray(sys_mg0, float) - np.interp(c0, ridge["colour"], ridge["mag"], left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore"):
        inc = (c0 >= cfg.colour0_range[0]) & (c0 <= cfg.colour0_range[1])
        tw = inc & (dm >= cfg.twin_dm[0]) & (dm < cfg.twin_dm[1])
        si = inc & (dm >= cfg.single_dm[0]) & (dm < cfg.single_dm[1])
    return tw, si


def twin_band_loglike(k_twin: ArrayLike, k_single: ArrayLike, mu_twin: ArrayLike, mu_single: ArrayLike) -> float:
    """F4: Σ over distance bins of the binomial log-likelihood of the real twin count among twin + single, with
    p = μ_twin / (μ_twin + μ_single) (constant terms dropped; bins with no mock or no real systems skipped)."""
    kt, ks = np.asarray(k_twin, float), np.asarray(k_single, float)
    mt, ms = np.asarray(mu_twin, float), np.asarray(mu_single, float)
    ok = (mt + ms > 0) & (kt + ks > 0)
    p = np.clip(np.where(ok, mt / np.where(ok, mt + ms, 1.0), 0.5), 1e-12, 1 - 1e-12)
    return float(np.sum(np.where(ok, kt * np.log(p) + ks * np.log1p(-p), 0.0)))


def photocentre_size(a0_mas: ArrayLike, parallax_mas: ArrayLike, period_days: ArrayLike) -> FloatArray:
    """F1: A = (a0 / ϖ) / P_yr^(2/3) in AU yr^(-2/3) (El-Badry et al. 2024's AMRF numerator at fixed mass)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a0_mas, float) / np.asarray(parallax_mas, float) / (np.asarray(period_days, float) / 365.25) ** (2.0 / 3.0)


def co_observables(truth: Mapping[str, NDArray[Any]], outcome: Mapping[str, NDArray[Any]], period_max_days: float) -> dict[str, NDArray[Any]]:
    """Compact-object draws on the luminous fit axes: system G, 1/ϖ_fit (orbit) or the 7/9-parameter parallax
    (acceleration), log P_fit, e_fit, system CMD class; ``in_orbit`` (accepted, P_fit ≤ C1) and ``in_accel``."""
    casc = np.asarray(outcome["cascade"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_o, d_a, lp = 1.0 / casc[:, 0], 1.0 / casc[:, 2], np.log10(casc[:, 10])
    orb = np.asarray(outcome["accepted_orbital"], bool) & (casc[:, 10] <= period_max_days)
    return {"g": np.asarray(truth["phot_g_mean_mag"], float), "d_orbit": d_o, "d_accel": d_a, "log_p": lp, "ecc": casc[:, 14],
            "cls": np.asarray(truth["sys_evolved"], bool).astype(float), "in_orbit": orb,
            "in_accel": np.asarray(outcome["published_acceleration"], bool)}


# ---------------------------------------------------------------------------
# #567: physical WD target (spec §12.38) — a parameterised target, not population synthesis
# ---------------------------------------------------------------------------


class WDTargetConfig(BaseModel):
    """Fixed shape numbers of the §12.38 WD target (fitted: ln R_wd, β, f_mt, f_He, α; see :func:`wd_target_log_density`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    m1_pivot_msun: float = Field(..., gt=0)
    ifmr_segments: tuple[tuple[float, float, float, float], ...]
    ms_lifetime_gyr_at_1msun: float = Field(..., gt=0)
    ms_lifetime_mass_exponent: float
    log_p_ni_min: float
    log_p_max: float
    m_wd_min: float = Field(..., gt=0)
    m_wd_max: float = Field(..., gt=0)
    he_max_msun: float = Field(..., gt=0)
    ts99_a: float = Field(..., gt=0)
    ts99_b_days: float = Field(..., gt=0)
    ts99_c_msun: float
    ts99_ref: str
    he_sigma_log_p: float = Field(..., gt=0)
    co_mt_mu_msun: float = Field(..., gt=0)
    co_mt_sigma_msun: float = Field(..., gt=0)
    co_mt_mu_log_p: float
    co_mt_sigma_log_p: float = Field(..., gt=0)
    co_mt_ref: str
    e_mt_max: float = Field(..., gt=0, le=1)
    circular_period_days: float = Field(..., gt=0)
    e_max_period_scale_days: float = Field(..., gt=0)
    e_max_exponent: float = Field(..., lt=0)


def _ln_norm_pdf(x: FloatArray, mu: ArrayLike, sig: float) -> FloatArray:
    return -0.5 * ((x - mu) / sig) ** 2 - math.log(sig * math.sqrt(2 * math.pi))


def ts99_log_period(m_wd: ArrayLike, cfg: WDTargetConfig) -> FloatArray:
    """log10 P (days) of the Tauris & Savonije (1999) relation M_WD = (P/b)^(1/a) + c; NaN for M_WD ≤ c."""
    m = np.asarray(m_wd, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m > cfg.ts99_c_msun, math.log10(cfg.ts99_b_days) + cfg.ts99_a * np.log10(m - cfg.ts99_c_msun), np.nan)


def _e_logpdf(e: FloatArray, p: FloatArray, kind: str, cfg: WDTargetConfig) -> FloatArray:
    circ = p <= cfg.circular_period_days
    if kind == "uniform_to_emax":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            em = np.clip(1.0 - (p / cfg.e_max_period_scale_days) ** cfg.e_max_exponent, 0.0, 1.0)
            le = np.where((e >= 0) & (e < em), -np.log(em), -np.inf)
    else:
        le = np.where((e >= 0) & (e < cfg.e_mt_max), -math.log(cfg.e_mt_max), -np.inf)
    return np.where(circ, np.where(e == 0.0, 0.0, -np.inf), le)


def wd_target_log_density(
    m2: ArrayLike, period_days: ArrayLike, ecc: ArrayLike, m1: ArrayLike, t_sys_gyr: ArrayLike,
    *, ln_r_wd: float, beta: float, f_mt: float, f_he: float, alpha: float, cfg: WDTargetConfig,
) -> FloatArray:
    """log λ_WD in the proposal's measure (per dex M2, per dex P, per unit e or the circular point mass), WITHOUT the
    cooling-age factor (it is the §13.2 law in every branch and cancels against the default target)."""
    m, p, e = (np.asarray(x, float) for x in (m2, period_days, ecc))
    m1_ = np.asarray(m1, float)
    lp = np.log10(p)
    ln10 = math.log(10.0)
    seg = np.asarray(cfg.ifmr_segments, float)
    mi_lo_tab, mi_hi_tab = float(seg[:, 0].min()), float(seg[:, 1].max())
    # (a) non-interacting: M_i ≥ max(M1, M_TO(t_sys)), p(M_i) ∝ M_i^-α on [lo, mi_hi]
    m_to = (np.asarray(t_sys_gyr, float) / cfg.ms_lifetime_gyr_at_1msun) ** (1.0 / cfg.ms_lifetime_mass_exponent)
    lo = np.maximum(np.maximum(m1_, np.where(np.isfinite(m_to), m_to, np.inf)), mi_lo_tab)
    ok_lo = lo < mi_hi_tab
    if abs(alpha - 1.0) < 1e-9:
        z = np.log(mi_hi_tab / np.where(ok_lo, lo, 1.0))
    else:
        z = (np.where(ok_lo, lo, 1.0) ** (1 - alpha) - mi_hi_tab ** (1 - alpha)) / (alpha - 1)
    from darkhunter_pop.compact_objects import ifmr_initial_mass

    mi = ifmr_initial_mass(m, seg)
    slope = np.full(m.size, np.nan)
    for (a_lo, a_hi, sl, ic) in seg:
        f_lo, f_hi = sl * a_lo + ic, sl * a_hi + ic
        slope = np.where(np.isnan(slope) & (m >= f_lo) & (m <= f_hi), sl, slope)
    inside = ok_lo & np.isfinite(slope) & (mi >= lo) & (mi <= mi_hi_tab)
    with np.errstate(divide="ignore", invalid="ignore"):
        lpm_ni = np.where(inside, -alpha * np.log(np.where(inside, mi, 1.0)) - np.log(np.where(inside, z, 1.0))
                          - np.log(np.where(inside, slope, 1.0)) + np.log(m * ln10), -np.inf)
    lpp_ni = np.where((lp >= cfg.log_p_ni_min) & (lp <= cfg.log_p_max), -math.log(cfg.log_p_max - cfg.log_p_ni_min), -np.inf)
    l_ni = lpm_ni + lpp_ni + _e_logpdf(e, p, "uniform_to_emax", cfg)
    # (b1) post-MT He WDs on the TS99 relation
    in_he = (m >= cfg.m_wd_min) & (m <= cfg.he_max_msun)
    lpt = ts99_log_period(m, cfg)
    l_he = np.where(in_he & np.isfinite(lpt), np.log(m * ln10 / (cfg.he_max_msun - cfg.m_wd_min)) + _ln_norm_pdf(lp, np.nan_to_num(lpt), cfg.he_sigma_log_p), -np.inf)
    l_he = l_he + _e_logpdf(e, p, "mt", cfg)
    # (b2) post-MT CO WDs (AU scales): truncated normal in M_WD on [he_max, m_wd_max]
    from scipy.special import ndtr

    zc = ndtr((cfg.m_wd_max - cfg.co_mt_mu_msun) / cfg.co_mt_sigma_msun) - ndtr((cfg.he_max_msun - cfg.co_mt_mu_msun) / cfg.co_mt_sigma_msun)
    in_co = (m > cfg.he_max_msun) & (m <= cfg.m_wd_max)
    l_co = np.where(in_co, _ln_norm_pdf(m, cfg.co_mt_mu_msun, cfg.co_mt_sigma_msun) - math.log(zc) + np.log(m * ln10)
                    + _ln_norm_pdf(lp, cfg.co_mt_mu_log_p, cfg.co_mt_sigma_log_p), -np.inf) + _e_logpdf(e, p, "mt", cfg)
    with np.errstate(divide="ignore"):
        mix = np.logaddexp(np.logaddexp(math.log(max(1 - f_mt, 1e-300)) + l_ni, math.log(max(f_mt * f_he, 1e-300)) + l_he),
                           math.log(max(f_mt * (1 - f_he), 1e-300)) + l_co)
    return ln_r_wd + beta * np.log(m1_ / cfg.m1_pivot_msun) + mix
