"""Magnitude-limit (Malmquist / Öpik) conditioning for Gaia-star primaries (#405).

docs/MOCK_POPULATION_SPEC.md §9. The mock keeps every real parent row's observables (G,
Bailer-Jones geometric distance, sky position, TAG10 M1; spec §0.1 MP-Q4–Q6) and draws a
companion state ``c`` in place of the row's unknown one. The parent selection depends on
observables only, so it cancels, and the correct companion density for row ``s`` is

    p(c | o_s) = π(c | M̂1_s) × W_s(c),     W_s(c) = L_s(f_b(c)) / Z_s

    L_s(f) = N(ΔM_s + 2.5 log10(1 + f); 0, σ_s)
    ΔM_s   = G_s − μ(d̂_s) − A_G,s − M_G^J(M̂1_s) − δ_zp
    σ_s²   = σ_int² + σ_μ,s² + σ_A,s² + (∂M_G^J/∂log10 M1)² σ²_log M̂1
    Z_s    = (1 − F_lum(M̂1_s)) L_s(0) + ∫ λ_f(log10 f | M̂1_s) L_s(f) dlog10 f

``π = λ`` is the target intensity (here the MdS17 luminous target of
:func:`darkhunter_pop.proposal_set.mds17_luminous_log_intensity`), ``λ_f`` its marginal on
log10 f and ``F_lum`` its integral. Dark companions (f = 0) carry ``L_s(0)``, like singles.

**Hook for** ``proposal_set``: add :func:`log_conditioning_factor` (log W) to the target
log-intensity before :func:`darkhunter_pop.proposal_set.importance_weights`; for proposal-set
draws :func:`log_weight_for_draws` does the bookkeeping from the stored truth and the parent
snapshot. :func:`log_no_companion_probability` gives p(∅ | o_s) for statistics that count
systems. :func:`tabulate_selection_volume` and :func:`selection_volume` give V_S for the
volume-limited diagnostic (spec §9.5) and are never needed by the forward model.

Every choice the spec leaves open (MP-Q25–Q31) is a ``provisional_*`` config field.
Nothing here imports or reimplements gaiamock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import moe_distefano as mds
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.proposal_set import (
    MdS17TargetConfig,
    ParentSnapshot,
    janssens_absolute_g,
    relation_log10_flux_ratio,
)

FloatArray = NDArray[np.float64]

#: dμ / d ln d = 5 / ln 10 (converts a fractional distance spread to magnitudes).
_MU_PER_LN_D: float = 5.0 / math.log(10.0)


# ---------------------------------------------------------------------------
# Config (config/population/malmquist.yaml; not under config/fragments/)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FluxMarginalGridConfig(_Strict):
    """Quadrature grid for λ_f(log10 f | M1) and F_lum(M1) (spec §9.3)."""

    log_m1_min: float
    log_m1_max: float
    n_m1: int = Field(..., ge=2)
    n_log_q: int = Field(..., ge=8)
    n_log_p: int = Field(..., ge=8)
    n_hermite: int = Field(..., ge=3)
    log_f_min: float
    log_f_max: float
    n_log_f: int = Field(..., ge=8)

    @model_validator(mode="after")
    def _order(self) -> FluxMarginalGridConfig:
        if not (self.log_m1_min < self.log_m1_max and self.log_f_min < self.log_f_max):
            raise ValueError("grid bounds must be increasing")
        return self


class MalmquistConfig(_Strict):
    """Settings for the magnitude-limit conditioning weight (spec §9, MP-Q25–Q31)."""

    provisional_sigma_int_mag: float = Field(..., ge=0.0)  # MP-Q25
    provisional_mg_zero_point_mag: float  # MP-Q25 (δ_zp, added to M_G^J)
    provisional_sigma_log_m1_dex: float = Field(..., ge=0.0)  # MP-Q25 (TAG10 scatter)
    provisional_blending: Literal["all_unresolved"]  # MP-Q27
    provisional_giant_policy: Literal["unit_weight"]  # MP-Q28
    provisional_distance_marginalization: Literal["gaussian_mu", "split_normal_mu"]  # MP-Q30
    provisional_frequency_above_one: Literal["clip_single_probability_at_zero"]  # MP-Q31
    slope_step_dex: float = Field(..., gt=0.0)  # finite-difference step for ∂M_G/∂log M1
    grid: FluxMarginalGridConfig


def load_malmquist_config(path: str | Path, key: str = "malmquist") -> MalmquistConfig:
    """Validate the ``key`` section of a YAML file (relative paths from the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return MalmquistConfig.model_validate(yaml.safe_load(p.read_text())[key])


# ---------------------------------------------------------------------------
# Row quantities: ΔM_s and σ_s
# ---------------------------------------------------------------------------


def distance_modulus(distance_pc: ArrayLike) -> FloatArray:
    """μ = 5 log10(d / 10 pc)."""
    return 5.0 * np.log10(np.asarray(distance_pc, dtype=np.float64)) - 5.0


def sigma_mu_from_quantiles(r_lo: ArrayLike, r_med: ArrayLike, r_hi: ArrayLike) -> FloatArray:
    """Gaussian-in-μ width from 16th / 50th / 84th distance percentiles (spec §9.3 (A2)).

    σ_μ = (5 / ln 10) (r_hi − r_lo) / (2 r_med). Non-finite inputs give NaN.
    """
    lo = np.asarray(r_lo, dtype=np.float64)
    med = np.asarray(r_med, dtype=np.float64)
    hi = np.asarray(r_hi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _MU_PER_LN_D * (hi - lo) / (2.0 * med)


def mg_slope_per_dex(m1_msun: ArrayLike, step_dex: float) -> FloatArray:
    """∂M_G^J / ∂log10 M1 by a central difference of the Janssens relation."""
    lm = np.log10(np.asarray(m1_msun, dtype=np.float64))
    hi = janssens_absolute_g(10.0 ** (lm + step_dex))
    lo = janssens_absolute_g(10.0 ** (lm - step_dex))
    return (hi - lo) / (2.0 * step_dex)


def luminosity_excess(
    g_mag: ArrayLike,
    distance_pc: ArrayLike,
    a_g_mag: ArrayLike,
    m1_msun: ArrayLike,
    *,
    zero_point_mag: float,
) -> FloatArray:
    """ΔM = G − μ(d) − A_G − (M_G^J(M1) + δ_zp); negative when over-luminous for M1."""
    mg = janssens_absolute_g(m1_msun) + zero_point_mag
    return (
        np.asarray(g_mag, dtype=np.float64)
        - distance_modulus(distance_pc)
        - np.asarray(a_g_mag, dtype=np.float64)
        - mg
    )


def row_sigma(
    m1_msun: ArrayLike,
    sigma_mu_mag: ArrayLike,
    sigma_a_mag: ArrayLike,
    cfg: MalmquistConfig,
) -> FloatArray:
    """σ_s = sqrt(σ_int² + σ_μ² + σ_A² + (∂M_G/∂log M1)² σ²_log M1) (spec §9.3)."""
    slope = mg_slope_per_dex(m1_msun, cfg.slope_step_dex)
    var = (
        cfg.provisional_sigma_int_mag**2
        + np.asarray(sigma_mu_mag, dtype=np.float64) ** 2
        + np.asarray(sigma_a_mag, dtype=np.float64) ** 2
        + (slope * cfg.provisional_sigma_log_m1_dex) ** 2
    )
    return np.sqrt(var)


def log_light_likelihood(
    delta_m: ArrayLike,
    log10_f: ArrayLike,
    sigma: ArrayLike,
    *,
    sigma_mu_lo: ArrayLike | None = None,
    sigma_mu_hi: ArrayLike | None = None,
) -> FloatArray:
    """log L = log ∫ N(ΔM(μ) + 2.5 log10(1 + f); 0, σ) p(μ) dμ. ``log10_f = -inf`` means f = 0.

    Without ``sigma_mu_lo`` / ``sigma_mu_hi`` (MP-Q30 option ``gaussian_mu``) ``sigma`` is the
    total σ_s and p(μ) is folded in as a Gaussian. With them (option ``split_normal_mu``)
    ``sigma`` excludes the distance term and p(μ) is a split normal around μ(r_med) with
    widths μ(r_med) − μ(r_lo) and μ(r_hi) − μ(r_med); the convolution is closed-form:
    L = K Σ_side σ_side / sqrt(σ² + σ_side²) exp(−a² / 2(σ² + σ_side²)) Φ(±a σ_side / (σ sqrt(σ² + σ_side²))),
    K = 2 / (sqrt(2π)(σ_lo + σ_hi)), a = ΔM_med + 2.5 log10(1 + f). Equal widths give the Gaussian.

    Measured in the #405 closed loop (docs/gate405): this split normal, with its *mode* at
    μ(r_med), does **worse** than ``gaussian_mu`` (its median is shifted when the widths
    differ). A quantile-matched split normal is untested. MP-Q30 stays Ryan's choice.
    """
    from scipy.special import log_ndtr

    dm = np.asarray(delta_m, dtype=np.float64)
    lf = np.asarray(log10_f, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    with np.errstate(over="ignore"):
        boost = 2.5 * np.log10(1.0 + np.where(np.isfinite(lf), 10.0**lf, 0.0))
    a = dm + boost
    if sigma_mu_lo is None or sigma_mu_hi is None:
        resid = a / s
        return -0.5 * resid * resid - np.log(s) - 0.5 * math.log(2.0 * math.pi)
    lo = np.asarray(sigma_mu_lo, dtype=np.float64)
    hi = np.asarray(sigma_mu_hi, dtype=np.float64)
    log_k = math.log(2.0) - 0.5 * math.log(2.0 * math.pi) - np.log(lo + hi)
    # The distance spread y = μ − μ_med enters ΔM(μ) = ΔM_med − y. The y > 0 (far) side uses
    # the r_hi width and is weighted by Φ(+a σ_hi / ...); the near side by Φ(−a σ_lo / ...).
    v_hi = s * s + hi * hi
    v_lo = s * s + lo * lo
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.log(hi) - 0.5 * np.log(v_hi) - 0.5 * a * a / v_hi + log_ndtr(a * hi / (s * np.sqrt(v_hi)))
        t_lo = np.log(lo) - 0.5 * np.log(v_lo) - 0.5 * a * a / v_lo + log_ndtr(-a * lo / (s * np.sqrt(v_lo)))
    return log_k + np.logaddexp(t_hi, t_lo)


# ---------------------------------------------------------------------------
# λ_f(log10 f | M1) and F_lum(M1) for the MdS17 luminous target
# ---------------------------------------------------------------------------


def mds17_luminous_m2_p_intensity(
    m1_msun: ArrayLike, log_q: ArrayLike, log_p: ArrayLike, target: MdS17TargetConfig
) -> FloatArray:
    """Companions per dex q per dex P at fixed M1 for the rung-2 MdS17 luminous target.

    Exactly the M2 and P factors of
    :func:`darkhunter_pop.proposal_set.mds17_luminous_log_intensity` (same table, same
    provisional settings, same M1 clamp for the shapes); the e and f factors are densities
    that integrate to one there and are left out here. A test checks the two agree.
    """
    table = mds.load_mds17_table(target.table_path)
    m1, lq, lp = np.broadcast_arrays(
        np.asarray(m1_msun, float), np.asarray(log_q, float), np.asarray(log_p, float)
    )
    m1_shape = np.clip(m1, table.m1_range[0], table.m1_range[1])
    q = 10.0**lq
    freq = mds.f_logp_q03(m1_shape, lp, table) * mds.low_mass_frequency_scale(
        m1,
        m1_anchor_msun=target.provisional_low_mass_anchor_msun,
        m1_zero_msun=target.provisional_low_mass_zero_msun,
    )
    pq = mds.q_density(q, m1_shape, lp, table, m1_interpolation=target.provisional_m1_interpolation)
    return freq * pq * q * math.log(10.0)


@dataclass(frozen=True)
class FluxMarginalGrid:
    """λ_f on a grid: ``lam_f[i, b]`` = companions per primary of mass ``m1[i]`` with
    log10 f in bin ``b`` (centre ``log_f[b]``); ``f_lum[i]`` = Σ_b lam_f[i, b] plus the mass
    that fell outside the f bins (clipped into the end bins, so none is lost)."""

    log_m1: FloatArray
    log_f: FloatArray
    lam_f: FloatArray
    f_lum: FloatArray

    def interpolate(self, m1_msun: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(λ_f rows, F_lum) at arbitrary M1, linear in log10 M1, clamped to the grid ends."""
        x = np.clip(np.log10(np.asarray(m1_msun, float)), self.log_m1[0], self.log_m1[-1])
        j = np.clip(np.searchsorted(self.log_m1, x, side="right") - 1, 0, self.log_m1.size - 2)
        t = (x - self.log_m1[j]) / (self.log_m1[j + 1] - self.log_m1[j])
        lam = (1.0 - t)[..., None] * self.lam_f[j] + t[..., None] * self.lam_f[j + 1]
        return lam, (1.0 - t) * self.f_lum[j] + t * self.f_lum[j + 1]


def _log_q_edges(table: mds.MdS17Table, n: int) -> FloatArray:
    """log q cell edges that fall on the density's breaks (q_break, twin_q_min), so the
    midpoint rule never straddles a discontinuity of q_density."""
    lo, hi = math.log10(table.q_range[0]), math.log10(table.q_range[1])
    breaks = [lo, math.log10(table.q_break), math.log10(table.twin_q_min), hi]
    widths = np.diff(breaks)
    counts = np.maximum(np.round(n * widths / (hi - lo)).astype(int), 4)
    pieces = [np.linspace(a, b, k + 1)[:-1] for a, b, k in zip(breaks[:-1], breaks[1:], counts)]
    return np.concatenate(pieces + [np.array([hi])])


def build_flux_marginal(target: MdS17TargetConfig, grid: FluxMarginalGridConfig) -> FluxMarginalGrid:
    """Tabulate λ_f(log10 f | M1) by midpoint quadrature in (log q, log P) and Gauss–Hermite
    in log10 f ~ N(log10 f_J(M1, M2), σ_f) (spec §9.3). Pure numpy, a few seconds."""
    table = mds.load_mds17_table(target.table_path)
    plo, phi = table.log_p_range
    lq_edges = _log_q_edges(table, grid.n_log_q)
    lp_edges = np.linspace(plo, phi, grid.n_log_p + 1)
    lq = 0.5 * (lq_edges[1:] + lq_edges[:-1])
    lp = 0.5 * (lp_edges[1:] + lp_edges[:-1])
    LQ, LP = np.meshgrid(lq, lp, indexing="ij")
    cell = np.outer(np.diff(lq_edges), np.diff(lp_edges))
    nodes, hw = np.polynomial.hermite_e.hermegauss(grid.n_hermite)
    hw = hw / hw.sum()
    log_m1 = np.linspace(grid.log_m1_min, grid.log_m1_max, grid.n_m1)
    f_edges = np.linspace(grid.log_f_min, grid.log_f_max, grid.n_log_f + 1)
    log_f = 0.5 * (f_edges[1:] + f_edges[:-1])
    lam_f = np.zeros((grid.n_m1, grid.n_log_f))
    f_lum = np.zeros(grid.n_m1)
    sig = target.flux_sigma_dex
    for i, lm1 in enumerate(log_m1):
        m1 = 10.0**lm1
        dens = mds17_luminous_m2_p_intensity(m1, LQ, LP, target) * cell
        rel = relation_log10_flux_ratio(m1, m1 * 10.0**LQ)
        ok = np.isfinite(dens) & np.isfinite(rel) & (dens > 0)
        d, r = dens[ok], rel[ok]
        f_lum[i] = float(d.sum())
        lf = r[:, None] + sig * nodes[None, :]
        wt = d[:, None] * hw[None, :]
        b = np.clip(np.searchsorted(f_edges, lf.ravel(), side="right") - 1, 0, grid.n_log_f - 1)
        lam_f[i] = np.bincount(b, weights=wt.ravel(), minlength=grid.n_log_f)
    return FluxMarginalGrid(log_m1=log_m1, log_f=log_f, lam_f=lam_f, f_lum=f_lum)


# ---------------------------------------------------------------------------
# Z_s, log W and p(∅ | o)
# ---------------------------------------------------------------------------


def _log_terms(
    m1_msun: FloatArray,
    delta_m: FloatArray,
    sigma: FloatArray,
    grid: FluxMarginalGrid,
    chunk: int,
    mu_lo: FloatArray | None = None,
    mu_hi: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """(log of the single term (1 − F)L(0), log Z, F > 1 flag), chunked over rows."""
    n = m1_msun.size
    log_single = np.empty(n)
    log_z = np.empty(n)
    over = np.empty(n, dtype=bool)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        lam, f_lum = grid.interpolate(m1_msun[sl])
        if mu_lo is None or mu_hi is None:
            l0 = log_light_likelihood(delta_m[sl], -np.inf, sigma[sl])
            lf = log_light_likelihood(delta_m[sl, None], grid.log_f[None, :], sigma[sl, None])
        else:
            l0 = log_light_likelihood(delta_m[sl], -np.inf, sigma[sl], sigma_mu_lo=mu_lo[sl], sigma_mu_hi=mu_hi[sl])
            lf = log_light_likelihood(
                delta_m[sl, None], grid.log_f[None, :], sigma[sl, None],
                sigma_mu_lo=mu_lo[sl, None], sigma_mu_hi=mu_hi[sl, None],
            )
        # Σ_b λ_b exp(lf_b) in log space, stabilized by the per-row maximum.
        mx = np.maximum(lf.max(axis=1), l0)
        lum = np.sum(lam * np.exp(lf - mx[:, None]), axis=1)
        p_single = np.clip(1.0 - f_lum, 0.0, None)  # MP-Q31 provisional clip
        over[sl] = f_lum > 1.0
        with np.errstate(divide="ignore"):
            log_single[sl] = np.log(p_single) + l0
            log_z[sl] = mx + np.log(p_single * np.exp(l0 - mx) + lum)
    return log_single, log_z, over


def log_conditioning_factor(
    log10_f: ArrayLike,
    m1_msun: ArrayLike,
    delta_m: ArrayLike,
    sigma: ArrayLike,
    grid: FluxMarginalGrid,
    *,
    is_giant: ArrayLike | None = None,
    sigma_mu_lo: ArrayLike | None = None,
    sigma_mu_hi: ArrayLike | None = None,
    chunk: int = 20_000,
) -> FloatArray:
    """log W_s(c) = log L_s(f) − log Z_s per draw (spec §9.3).

    ``sigma_mu_lo`` / ``sigma_mu_hi`` select the split-normal distance marginalization
    (MP-Q30 option ``split_normal_mu``; ``sigma`` then excludes the distance term).

    Add the result to the target log-intensity before
    :func:`darkhunter_pop.proposal_set.importance_weights`. Rows with a non-finite ΔM or σ
    (no distance, no M_G^J) and giants under ``provisional_giant_policy: unit_weight``
    (MP-Q28) get log W = 0, i.e. the naive draw; callers report how many.
    """
    lf = np.asarray(log10_f, float)
    m1 = np.asarray(m1_msun, float)
    dm = np.asarray(delta_m, float)
    s = np.asarray(sigma, float)
    ok = np.isfinite(dm) & np.isfinite(s) & (s > 0) & np.isfinite(m1)
    if is_giant is not None:
        ok &= ~np.asarray(is_giant, bool)
    split = sigma_mu_lo is not None and sigma_mu_hi is not None
    if split:
        lo = np.broadcast_to(np.asarray(sigma_mu_lo, float), lf.shape)
        hi = np.broadcast_to(np.asarray(sigma_mu_hi, float), lf.shape)
        ok &= np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
    out = np.zeros(lf.shape)
    if ok.any():
        kw = {"sigma_mu_lo": lo[ok], "sigma_mu_hi": hi[ok]} if split else {}
        _, log_z, _ = _log_terms(m1[ok], dm[ok], s[ok], grid, chunk, kw.get("sigma_mu_lo"), kw.get("sigma_mu_hi"))
        out[ok] = log_light_likelihood(dm[ok], lf[ok], s[ok], **kw) - log_z
    return out


def log_no_companion_probability(
    m1_msun: ArrayLike,
    delta_m: ArrayLike,
    sigma: ArrayLike,
    grid: FluxMarginalGrid,
    *,
    sigma_mu_lo: ArrayLike | None = None,
    sigma_mu_hi: ArrayLike | None = None,
    chunk: int = 20_000,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """log p(∅ | o_s) = log[(1 − F_lum) L_s(0) / Z_s] and the MP-Q31 flag (F_lum > 1).

    Dark companions are not in this target (MP-Q17 provisional ``none``); with a compact
    mixture, (1 − F) would include them.
    """
    m1 = np.asarray(m1_msun, float)
    lo = None if sigma_mu_lo is None else np.broadcast_to(np.asarray(sigma_mu_lo, float), m1.shape)
    hi = None if sigma_mu_hi is None else np.broadcast_to(np.asarray(sigma_mu_hi, float), m1.shape)
    log_single, log_z, over = _log_terms(m1, np.asarray(delta_m, float), np.asarray(sigma, float), grid, chunk, lo, hi)
    return log_single - log_z, over


# ---------------------------------------------------------------------------
# Proposal-set bookkeeping (the hook used with proposal_set draws)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowConditioning:
    """Per parent row: ΔM_s, σ_s and the M1 the conditioning used.

    Under ``split_normal_mu`` (MP-Q30), ``sigma`` excludes distance and ``sigma_mu_lo`` /
    ``sigma_mu_hi`` carry the two-sided distance-modulus widths; under ``gaussian_mu`` they
    are None and ``sigma`` is the total σ_s.
    """

    delta_m: FloatArray
    sigma: FloatArray
    m1_msun: FloatArray
    is_giant: NDArray[np.bool_]
    sigma_mu_lo: FloatArray | None = None
    sigma_mu_hi: FloatArray | None = None

    def mu_kwargs(self, idx: Any = slice(None)) -> dict[str, FloatArray]:
        if self.sigma_mu_lo is None or self.sigma_mu_hi is None:
            return {}
        return {"sigma_mu_lo": self.sigma_mu_lo[idx], "sigma_mu_hi": self.sigma_mu_hi[idx]}


def row_conditioning(
    parent: ParentSnapshot,
    cfg: MalmquistConfig,
    *,
    a_g_mag: ArrayLike,
    sigma_a_mag: ArrayLike,
) -> RowConditioning:
    """ΔM_s and σ_s for every parent row (spec §9.3).

    Distance: d̂ = 1000 / ``truth_parallax_mas`` (Bailer-Jones ``r_med_geo`` under MP-Q4) and
    σ_μ from ``r_lo_geo`` / ``r_med_geo`` / ``r_hi_geo``. Extinction ``a_g_mag`` and its error
    are explicit inputs because the source is open (MP-Q14 / MP-Q29); pass zeros only for a
    synthetic parent without dust.
    """
    cols = parent.columns
    for name in ("r_lo_geo", "r_med_geo", "r_hi_geo"):
        if name not in cols:
            raise ValueError(f"parent snapshot lacks {name}; Bailer-Jones quantiles are required (spec §9.3 A2)")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1000.0 / np.asarray(parent.truth_parallax_mas, float)
    s_mu = sigma_mu_from_quantiles(cols["r_lo_geo"], cols["r_med_geo"], cols["r_hi_geo"])
    m1 = np.asarray(parent.m1_msun, float)
    dm = luminosity_excess(
        cols["phot_g_mean_mag"], d, a_g_mag, m1, zero_point_mag=cfg.provisional_mg_zero_point_mag
    )
    s_a = np.broadcast_to(np.asarray(sigma_a_mag, float), m1.shape)
    giant = np.asarray(parent.is_giant, bool)
    if cfg.provisional_distance_marginalization == "split_normal_mu":
        mu_med = distance_modulus(cols["r_med_geo"])
        return RowConditioning(
            delta_m=dm,
            sigma=row_sigma(m1, np.zeros_like(m1), s_a, cfg),
            m1_msun=m1,
            is_giant=giant,
            sigma_mu_lo=mu_med - distance_modulus(cols["r_lo_geo"]),
            sigma_mu_hi=distance_modulus(cols["r_hi_geo"]) - mu_med,
        )
    sig = row_sigma(m1, s_mu, s_a, cfg)
    return RowConditioning(delta_m=dm, sigma=sig, m1_msun=m1, is_giant=giant)


def log_weight_for_draws(
    truth: Mapping[str, NDArray[Any]],
    rows: RowConditioning,
    grid: FluxMarginalGrid,
    cfg: MalmquistConfig,
) -> FloatArray:
    """log W per proposal-set draw (``truth`` from :func:`proposal_set.sample_proposal`).

    Usage (the hook)::

        lt = ps.mds17_luminous_log_intensity(truth, target)
        lt = lt + malmquist.log_weight_for_draws(truth, rows, grid, mcfg)
        w = ps.importance_weights(lt, [log_q_j(truth) ...], [n_j ...], scale_to_full=...)

    ``provisional_blending: all_unresolved`` (MP-Q27): every drawn companion's light blends.
    """
    if cfg.provisional_blending != "all_unresolved":  # pragma: no cover - Literal guards it
        raise ValueError(cfg.provisional_blending)
    r = np.asarray(truth["parent_row"], dtype=np.int64)
    lf = np.where(np.asarray(truth["is_dark"], bool), -np.inf, np.asarray(truth["log10_flux_ratio"], float))
    return log_conditioning_factor(
        lf,
        rows.m1_msun[r],
        rows.delta_m[r],
        rows.sigma[r],
        grid,
        is_giant=rows.is_giant[r] if cfg.provisional_giant_policy == "unit_weight" else None,
        **rows.mu_kwargs(r),
    )


# ---------------------------------------------------------------------------
# Volume-limited diagnostic (spec §9.5): V_S(M_abs)
# ---------------------------------------------------------------------------


def tabulate_selection_volume(
    m_abs_grid: ArrayLike,
    distance_pc: ArrayLike,
    p_select: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    position_weight: ArrayLike | None = None,
    chunk: int = 4096,
) -> FloatArray:
    """V_S(M) ∝ E_{r ~ ρ_*}[P(S | G = M + μ(r), r)] by Monte Carlo over positions.

    ``distance_pc`` are draws from the spatial density ρ_* (with any sky dependence carried
    by ``p_select``'s closure over the same positions' order); ``p_select(g, idx)`` returns
    the selection probability for apparent magnitudes ``g`` (shape ``(n_m, n_pos)``) at the
    position indices ``idx``. Units are arbitrary (relative volumes), which is all the
    1/V_S inversion needs.
    """
    m = np.asarray(m_abs_grid, float)
    mu = distance_modulus(distance_pc)
    w = np.ones(mu.size) if position_weight is None else np.asarray(position_weight, float)
    out = np.zeros(m.size)
    for a in range(0, mu.size, chunk):
        idx = np.arange(a, min(a + chunk, mu.size))
        g = m[:, None] + mu[None, idx]
        out += (p_select(g, idx) * w[None, idx]).sum(axis=1)
    return out / w.sum()


def selection_volume(
    m_abs: ArrayLike, m_abs_grid: ArrayLike, v_table: ArrayLike, *, sigma_int_mag: float, n_hermite: int = 9
) -> FloatArray:
    """V_S at system absolute magnitude(s) ``m_abs`` convolved with the intrinsic scatter."""
    nodes, hw = np.polynomial.hermite_e.hermegauss(n_hermite)
    hw = hw / hw.sum()
    m = np.asarray(m_abs, float)
    vals = np.interp(m[..., None] + sigma_int_mag * nodes, np.asarray(m_abs_grid, float), np.asarray(v_table, float))
    return np.sum(vals * hw, axis=-1)
