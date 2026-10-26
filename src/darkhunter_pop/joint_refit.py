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
