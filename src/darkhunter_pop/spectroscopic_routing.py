"""Bright spectroscopic routing of DR3 NSS solutions (#391 spec §12.34; statistical, calibrated).

Per mock system: G_RVS from G and BP−RP (Sartoretti et al. 2023, A&A 674, A6 cubic, linearly continued redward), the light-weighted RV
semi-amplitude K_obs = (K1 − f K2)/(1 + f), an epoch RV noise σ_RV(G_RVS, Teff) = σ0 10^{slope (G_RVS − 8)} (× hot
factor above ``teff_hot_k``), spectroscopic detection when G_RVS ≤ ``grvs_max`` and K_obs/σ_RV √(N_t/2) > S_min.
A system detected both spectroscopically and astrometrically is published astrometrically with probability
``p_route`` (else SB1-only, outside our sample). σ0 and S_min are degenerate; S_min and p_route are calibrated on the
bright parent's DR3 per-star solution-type rates. Reweight-level (stored truth only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoutingConfig:
    s_min: float = 5.0
    p_route: float = 1.0
    sigma0_kms: float = 0.5
    slope_per_mag: float = 0.25
    teff_hot_k: float = 7000.0
    hot_factor: float = 3.0
    n_transits: int = 20
    grvs_max: float = 12.0


def grvs_from_g(g: ArrayLike, bp_rp: ArrayLike) -> FloatArray:
    """G_RVS from G and (observed) BP−RP: the Sartoretti et al. (2023) cubic for BP−RP ≤ 1.2, continued linearly
    (matching value and slope) redward. An approximation; the routing calibration absorbs residual offsets."""
    g, c = np.asarray(g, float), np.asarray(bp_rp, float)
    cubic = lambda x: -0.0397 - 0.2852 * x - 0.0330 * x**2 - 0.0867 * x**3  # noqa: E731
    slope = -0.2852 - 2 * 0.0330 * 1.2 - 3 * 0.0867 * 1.2**2
    return g + np.where(c <= 1.2, cubic(c), cubic(1.2) + slope * (c - 1.2))


def k_observed(k1: ArrayLike, k2: ArrayLike, flux_ratio: ArrayLike) -> FloatArray:
    """Semi-amplitude of the light-weighted line centre (dark companion: K1)."""
    f = np.nan_to_num(np.asarray(flux_ratio, float), nan=0.0)
    return np.abs(np.asarray(k1, float) - f * np.asarray(k2, float)) / (1.0 + f)


def sigma_rv_epoch(grvs: ArrayLike, teff_k: ArrayLike, cfg: RoutingConfig) -> FloatArray:
    s = cfg.sigma0_kms * 10.0 ** (cfg.slope_per_mag * (np.asarray(grvs, float) - 8.0))
    return np.where(np.asarray(teff_k, float) > cfg.teff_hot_k, cfg.hot_factor * s, s)


def spectroscopic_detected(k_obs: ArrayLike, grvs: ArrayLike, teff_k: ArrayLike, cfg: RoutingConfig) -> NDArray[np.bool_]:
    snr = np.asarray(k_obs, float) / sigma_rv_epoch(grvs, teff_k, cfg) * np.sqrt(cfg.n_transits / 2.0)
    return (np.asarray(grvs, float) <= cfg.grvs_max) & (snr > cfg.s_min)


def astrometric_publication_factor(spec_detected: ArrayLike, cfg: RoutingConfig) -> FloatArray:
    """Per-draw factor on the astrometric-orbit weight: p_route where also SB-detected, else 1."""
    return np.where(np.asarray(spec_detected, bool), cfg.p_route, 1.0)
