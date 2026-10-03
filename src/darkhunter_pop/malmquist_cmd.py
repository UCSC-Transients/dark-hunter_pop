"""2-D CMD magnitude-limit (Malmquist / Öpik) weight (#418; docs/MOCK_POPULATION_SPEC.md §11.4).

Ryan (2026-10-03): "do the normal stellar-locus binary check, but a binary will not just shift
M_G, but also BP−RP. Both have to be taken into account along with extinction." For parent row
``s`` with dereddened system point ``y_s = (C_s, M_s)`` and a drawn luminous companion of mass
M2 and G-band flux ratio f = F2/F1, the companion's light is subtracted in G, BP and RP:

    x_G  = f / (1 + f)
    x_BP = x_G 10^{−0.4 [(BP−G)_2 − (BP−G)_s]},   x_RP = x_G 10^{+0.4 [(G−RP)_2 − (G−RP)_s]}
    C_1  = C_s − 2.5 log10(1 − x_BP) + 2.5 log10(1 − x_RP),   M_1 = M_s + 2.5 log10(1 + f)

with the companion's intrinsic colours from the MIST main sequence at M2 and the system's
(BP−G), (G−RP) from the MIST single-star colour–colour relation at C_s. The remaining primary
is compared with the measured single-star ridge R(C) (:class:`darkhunter_pop.giants.MSRidge`):

    L_s(c) = N(M_1 − R(C_1); 0, σ_s(C_1)),   L_s(∅) = N(M_s − R(C_s); 0, σ_s(C_s))
    σ_s(C)² = σ_R(C)² + σ_μ,s² + σ_A² (1 − R′(C) / k_s)²
    Z_s = (1 − F_lum(M̂1)) L_s(∅) + Σ_{q,f} λ_qf(M̂1) L_s(q, f),   W_s(c) = L_s(c) / Z_s

No M1 → M_G relation enters L. ``λ_qf`` is the rung-2 MdS17 luminous target
(:func:`darkhunter_pop.proposal_set.mds17_luminous_log_intensity`) marginalized over P and e
onto (log q, log f) bins; f is spread about the decided Janssens relation (MP-Q13).

Rows that get W = 1 (counted by the caller): CMD-evolved rows (§10.6), rows outside the ridge
colour range, rows without a CMD. Every open choice is a ``provisional_*`` config field
(MP-Q33–Q38). Nothing here imports or reimplements gaiamock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import moe_distefano as mds
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.giants import MSRidge
from darkhunter_pop.malmquist import _log_q_edges, mds17_luminous_m2_p_intensity
from darkhunter_pop.proposal_set import MdS17TargetConfig, relation_log10_flux_ratio

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Config (config/population/malmquist_cmd.yaml; not under config/fragments/)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QFGridConfig(_Strict):
    """Quadrature grid for λ_qf(log q, log f | M1) (numerical only)."""

    log_m1_min: float
    log_m1_max: float
    n_m1: int = Field(..., ge=2)
    n_log_q_fine: int = Field(..., ge=8)  # quadrature cells in log q (on MdS17's breaks)
    n_log_q_bins: int = Field(..., ge=4)  # output bins in log q
    n_log_p: int = Field(..., ge=8)
    n_hermite: int = Field(..., ge=3)
    log_f_min: float  # companions fainter than this are put in the first bin (boost < 1e-5 mag)
    log_f_max: float
    n_log_f: int = Field(..., ge=8)

    @model_validator(mode="after")
    def _order(self) -> QFGridConfig:
        if not (self.log_m1_min < self.log_m1_max and self.log_f_min < self.log_f_max):
            raise ValueError("grid bounds must be increasing")
        return self


class CompanionColourConfig(_Strict):
    """Companion and system colours from the MIST main sequence (MP-Q37)."""

    provisional_mode: Literal["fiducial_ms"] = "fiducial_ms"
    fiducial_feh_dex: float = -0.1
    fiducial_log_age: float = 9.6


class CmdMalmquistConfig(_Strict):
    """Settings for the 2-D weight (spec §11.4)."""

    provisional_giant_policy: Literal["unit_weight"] = "unit_weight"  # §10.6 (determined)
    provisional_outside_ridge: Literal["unit_weight"] = "unit_weight"  # MP-Q38
    provisional_blending: Literal["all_unresolved"] = "all_unresolved"  # MP-Q27
    sigma_a_mag: float = Field(0.0, ge=0.0)  # MP-Q29 decided: 0
    companion_colour: CompanionColourConfig = CompanionColourConfig()
    grid: QFGridConfig


def load_cmd_malmquist_config(path: str | Path, key: str = "malmquist_cmd") -> CmdMalmquistConfig:
    """Validate the ``key`` section of a YAML file (relative paths from the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return CmdMalmquistConfig.model_validate(yaml.safe_load(p.read_text())[key])


# ---------------------------------------------------------------------------
# MIST colours: companion at M2 and the single-star colour–colour relation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MsColours:
    """MIST main sequence at one ([Fe/H], age): colours vs mass and vs (BP−RP)."""

    mass: FloatArray
    colour: FloatArray
    bp_g: FloatArray
    g_rp: FloatArray
    mg: FloatArray

    def companion(self, m2: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(BP−G, G−RP) of a main-sequence star of mass ``m2`` (clamped to the MIST range)."""
        m = np.clip(np.asarray(m2, float), self.mass[0], self.mass[-1])
        return np.interp(m, self.mass, self.bp_g), np.interp(m, self.mass, self.g_rp)

    def from_colour(self, colour: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(BP−G, G−RP) of a single star of colour ``colour`` (BP−RP), clamped."""
        order = np.argsort(self.colour)
        c = np.asarray(colour, float)
        bpg = np.interp(c, self.colour[order], self.bp_g[order])
        return bpg, c - bpg


def ms_colours(grid: im.NativeGrid, feh: float, log_age: float) -> MsColours:
    """MIST MS relation at ([Fe/H], log age), linear in [Fe/H] between the bracketing files."""
    j = int(np.clip(np.searchsorted(grid.feh, feh) - 1, 0, grid.feh.size - 2))
    t = float(np.clip((feh - grid.feh[j]) / (grid.feh[j + 1] - grid.feh[j]), 0.0, 1.0))
    m_a, a = im.main_sequence_colour_relation(grid, float(grid.feh[j]), log_age)
    m_b, b = im.main_sequence_colour_relation(grid, float(grid.feh[j + 1]), log_age)
    lo, hi = max(m_a[0], m_b[0]), min(m_a[-1], m_b[-1])
    mass = np.geomspace(lo, hi, 400)
    out = {k: (1 - t) * np.interp(mass, m_a, a[k]) + t * np.interp(mass, m_b, b[k]) for k in ("mg", "bp_g", "g_rp", "colour")}
    return MsColours(mass=mass, colour=out["colour"], bp_g=out["bp_g"], g_rp=out["g_rp"], mg=out["mg"])


# ---------------------------------------------------------------------------
# Subtracting the companion and the ridge likelihood
# ---------------------------------------------------------------------------


def subtract_companion(
    colour_sys: ArrayLike,
    mg_sys: ArrayLike,
    log10_f: ArrayLike,
    m2_msun: ArrayLike,
    ms: MsColours,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Primary (C_1, M_1) after removing the companion's G, BP, RP light; ``ok`` False where
    the companion would outshine the system in BP or RP. ``log10_f = -inf`` returns the system."""
    cs = np.asarray(colour_sys, float)
    ms_ = np.asarray(mg_sys, float)
    lf = np.asarray(log10_f, float)
    with np.errstate(over="ignore", invalid="ignore"):
        f = np.where(np.isfinite(lf), 10.0**lf, 0.0)
    xg = f / (1.0 + f)
    bpg2, grp2 = ms.companion(m2_msun)
    bpgs, grps = ms.from_colour(cs)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xbp = xg * 10.0 ** (-0.4 * (bpg2 - bpgs))
        xrp = xg * 10.0 ** (0.4 * (grp2 - grps))
        ok = (xbp < 1.0) & (xrp < 1.0)
        c1 = cs - 2.5 * np.log10(np.where(ok, 1.0 - xbp, 1.0)) + 2.5 * np.log10(np.where(ok, 1.0 - xrp, 1.0))
        m1 = ms_ + 2.5 * np.log10(1.0 + f)
    return c1, m1, ok


@dataclass(frozen=True)
class RidgeTables:
    """Ridge R(C), width σ_R(C) and slope R′(C), clamped at the measured ends."""

    colour: FloatArray
    mag: FloatArray
    sigma: FloatArray
    slope: FloatArray

    @classmethod
    def from_ridge(cls, ridge: MSRidge) -> RidgeTables:
        return cls(
            colour=np.asarray(ridge.colour, float),
            mag=np.asarray(ridge.mag, float),
            sigma=np.asarray(ridge.sigma, float),
            slope=np.gradient(np.asarray(ridge.mag, float), np.asarray(ridge.colour, float)),
        )

    def at(self, c: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
        x = np.asarray(c, float)
        inr = np.isfinite(x) & (x >= self.colour[0]) & (x <= self.colour[-1])
        return (
            np.interp(x, self.colour, self.mag),
            np.interp(x, self.colour, self.sigma),
            np.interp(x, self.colour, self.slope),
            inr,
        )


def log_ridge_likelihood(
    colour1: ArrayLike,
    mg1: ArrayLike,
    sigma_mu: ArrayLike,
    ridge: RidgeTables,
    *,
    sigma_a_mag: float = 0.0,
    k_ag_over_ebprp: ArrayLike | float = 2.0,
) -> FloatArray:
    """log N(M_1 − R(C_1); 0, σ_s(C_1)) with σ_s² = σ_R² + σ_μ² + σ_A² (1 − R′/k)²."""
    r, s_r, slope, _ = ridge.at(colour1)
    smu = np.asarray(sigma_mu, float)
    k = np.asarray(k_ag_over_ebprp, float)
    var = s_r**2 + smu**2 + (sigma_a_mag * (1.0 - slope / k)) ** 2
    resid = np.asarray(mg1, float) - r
    return -0.5 * resid * resid / var - 0.5 * np.log(var) - _LOG_SQRT_2PI


# ---------------------------------------------------------------------------
# λ_qf(M1) grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QFGrid:
    """``lam[i, a, b]``: companions per primary of mass ``10**log_m1[i]`` with log q in bin
    ``a`` (centre ``log_q[a]``) and log f in bin ``b`` (centre ``log_f[b]``); ``f_lum[i]`` is
    the total (mass outside the f range is clipped into the end bins, none is lost)."""

    log_m1: FloatArray
    log_q: FloatArray
    log_f: FloatArray
    lam: FloatArray
    f_lum: FloatArray

    def interpolate(self, m1_msun: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(λ_qf, F_lum) at arbitrary M1, linear in log10 M1, clamped to the grid ends."""
        x = np.clip(np.log10(np.asarray(m1_msun, float)), self.log_m1[0], self.log_m1[-1])
        j = np.clip(np.searchsorted(self.log_m1, x, side="right") - 1, 0, self.log_m1.size - 2)
        t = (x - self.log_m1[j]) / (self.log_m1[j + 1] - self.log_m1[j])
        lam = (1.0 - t)[..., None, None] * self.lam[j] + t[..., None, None] * self.lam[j + 1]
        return lam, (1.0 - t) * self.f_lum[j] + t * self.f_lum[j + 1]


def build_qf_grid(target: MdS17TargetConfig, grid: QFGridConfig) -> QFGrid:
    """Tabulate λ_qf by midpoint quadrature in (log q, log P) and Gauss–Hermite in log f."""
    table = mds.load_mds17_table(target.table_path)
    plo, phi = table.log_p_range
    lq_edges = _log_q_edges(table, grid.n_log_q_fine)
    lp_edges = np.linspace(plo, phi, grid.n_log_p + 1)
    lq = 0.5 * (lq_edges[1:] + lq_edges[:-1])
    lp = 0.5 * (lp_edges[1:] + lp_edges[:-1])
    LQ, LP = np.meshgrid(lq, lp, indexing="ij")
    cell = np.outer(np.diff(lq_edges), np.diff(lp_edges))
    nodes, hw = np.polynomial.hermite_e.hermegauss(grid.n_hermite)
    hw = hw / hw.sum()
    log_m1 = np.linspace(grid.log_m1_min, grid.log_m1_max, grid.n_m1)
    q_out = np.linspace(lq_edges[0], lq_edges[-1], grid.n_log_q_bins + 1)
    f_edges = np.linspace(grid.log_f_min, grid.log_f_max, grid.n_log_f + 1)
    lam = np.zeros((grid.n_m1, grid.n_log_q_bins, grid.n_log_f))
    f_lum = np.zeros(grid.n_m1)
    qb_fine = np.clip(np.searchsorted(q_out, lq, side="right") - 1, 0, grid.n_log_q_bins - 1)
    for i, lm1 in enumerate(log_m1):
        m1 = 10.0**lm1
        dens = (mds17_luminous_m2_p_intensity(m1, LQ, LP, target) * cell).sum(axis=1)  # per fine log q
        rel = relation_log10_flux_ratio(m1, m1 * 10.0**lq)
        ok = np.isfinite(dens) & np.isfinite(rel) & (dens > 0)
        d, r, qb = dens[ok], rel[ok], qb_fine[ok]
        f_lum[i] = float(d.sum())
        lf = r[:, None] + target.flux_sigma_dex * nodes[None, :]
        wt = d[:, None] * hw[None, :]
        fb = np.clip(np.searchsorted(f_edges, lf, side="right") - 1, 0, grid.n_log_f - 1)
        flat = (np.broadcast_to(qb[:, None], fb.shape) * grid.n_log_f + fb).ravel()
        lam[i] = np.bincount(flat, weights=wt.ravel(), minlength=grid.n_log_q_bins * grid.n_log_f).reshape(
            grid.n_log_q_bins, grid.n_log_f
        )
    return QFGrid(
        log_m1=log_m1,
        log_q=0.5 * (q_out[1:] + q_out[:-1]),
        log_f=0.5 * (f_edges[1:] + f_edges[:-1]),
        lam=lam,
        f_lum=f_lum,
    )


# ---------------------------------------------------------------------------
# Rows, Z_s, log W
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmdRows:
    """Per parent row: dereddened system point, σ_μ, k = A_G / E(BP−RP), M̂1, and unit-weight flags."""

    colour0: FloatArray
    mg0: FloatArray
    sigma_mu: FloatArray
    k_ag_over_ebprp: FloatArray
    m1_msun: FloatArray
    unit_weight: BoolArray
    unit_reason: NDArray[np.str_]

    def counts(self) -> dict[str, int]:
        r, k = np.unique(self.unit_reason, return_counts=True)
        return {str(a): int(b) for a, b in zip(r, k)}


def cmd_rows(
    colour0: ArrayLike,
    mg0: ArrayLike,
    sigma_mu: ArrayLike,
    m1_msun: ArrayLike,
    ridge: RidgeTables,
    *,
    evolved: ArrayLike | None = None,
    a_g: ArrayLike | None = None,
    e_bp_rp: ArrayLike | None = None,
) -> CmdRows:
    """Assemble the per-row inputs and mark the unit-weight rows (reason per row)."""
    c = np.asarray(colour0, float)
    m = np.asarray(mg0, float)
    smu = np.asarray(sigma_mu, float)
    smu = np.where(np.isfinite(smu), smu, 0.0)
    n = c.size
    reason = np.full(n, "weighted", dtype="<U16")
    _, _, _, inr = ridge.at(c)
    reason[~inr] = "outside_ridge"
    reason[~(np.isfinite(c) & np.isfinite(m))] = "no_cmd"
    if evolved is not None:
        reason[np.asarray(evolved, bool) & (reason == "weighted")] = "evolved"
    m1 = np.asarray(m1_msun, float)
    reason[~np.isfinite(m1) & (reason == "weighted")] = "no_m1"
    if a_g is not None and e_bp_rp is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.asarray(a_g, float) / np.asarray(e_bp_rp, float)
        k = np.where(np.isfinite(k) & (k > 0), k, 2.0)
    else:
        k = np.full(n, 2.0)
    return CmdRows(
        colour0=c, mg0=m, sigma_mu=smu, k_ag_over_ebprp=k, m1_msun=m1,
        unit_weight=reason != "weighted", unit_reason=reason,
    )


def _row_log_terms(
    rows: CmdRows,
    idx: NDArray[np.int64],
    qf: QFGrid,
    ridge: RidgeTables,
    ms: MsColours,
    sigma_a: float,
    chunk: int,
) -> tuple[FloatArray, FloatArray]:
    """(log of (1 − F)L(∅), log Z) for rows ``idx`` (chunked)."""
    log_single = np.empty(idx.size)
    log_z = np.empty(idx.size)
    q = 10.0**qf.log_q
    for a in range(0, idx.size, chunk):
        ii = idx[a:a + chunk]
        c, m, smu, k, m1 = rows.colour0[ii], rows.mg0[ii], rows.sigma_mu[ii], rows.k_ag_over_ebprp[ii], rows.m1_msun[ii]
        lam, f_lum = qf.interpolate(m1)  # (n, nq, nf)
        l0 = log_ridge_likelihood(c, m, smu, ridge, sigma_a_mag=sigma_a, k_ag_over_ebprp=k)
        shape = (ii.size, q.size, qf.log_f.size)
        m2 = np.broadcast_to((m1[:, None] * q[None, :])[:, :, None], shape)
        lf = np.broadcast_to(qf.log_f[None, None, :], shape)
        c1, mg1, ok = subtract_companion(
            np.broadcast_to(c[:, None, None], shape), np.broadcast_to(m[:, None, None], shape), lf, m2, ms
        )
        ll = log_ridge_likelihood(
            c1, mg1, np.broadcast_to(smu[:, None, None], shape), ridge, sigma_a_mag=sigma_a,
            k_ag_over_ebprp=np.broadcast_to(k[:, None, None], shape),
        )
        ll = np.where(ok, ll, -np.inf)
        mx = np.maximum(ll.reshape(ii.size, -1).max(axis=1), l0)
        lum = np.sum(lam * np.exp(ll - mx[:, None, None]), axis=(1, 2))
        p_single = np.clip(1.0 - f_lum, 0.0, None)  # MP-Q31 provisional clip
        with np.errstate(divide="ignore"):
            log_single[a:a + chunk] = np.log(p_single) + l0
            log_z[a:a + chunk] = mx + np.log(p_single * np.exp(l0 - mx) + lum)
    return log_single, log_z


@dataclass(frozen=True)
class RowNormalization:
    """log Z_s and log p(∅ | o_s) per row (NaN for unit-weight rows)."""

    log_z: FloatArray
    log_p_single: FloatArray


def row_normalization(
    rows: CmdRows, qf: QFGrid, ridge: RidgeTables, ms: MsColours, cfg: CmdMalmquistConfig, *, chunk: int = 400
) -> RowNormalization:
    """Z_s for every weighted row (spec §11.4)."""
    n = rows.colour0.size
    log_z = np.full(n, np.nan)
    lps = np.full(n, np.nan)
    idx = np.flatnonzero(~rows.unit_weight)
    if idx.size:
        ls, lz = _row_log_terms(rows, idx, qf, ridge, ms, cfg.sigma_a_mag, chunk)
        log_z[idx] = lz
        lps[idx] = ls - lz
    return RowNormalization(log_z=log_z, log_p_single=lps)


def log_weight_for_draws(
    truth: Mapping[str, NDArray[Any]],
    rows: CmdRows,
    norm: RowNormalization,
    ridge: RidgeTables,
    ms: MsColours,
    cfg: CmdMalmquistConfig,
) -> FloatArray:
    """log W per proposal-set draw; 0 on unit-weight rows. Dark draws carry L_s(∅).

    ``truth`` needs ``parent_row``, ``is_dark``, ``log10_flux_ratio`` and ``m2_msun`` (as
    written by :func:`darkhunter_pop.proposal_set.sample_proposal`).
    """
    r = np.asarray(truth["parent_row"], np.int64)
    dark = np.asarray(truth["is_dark"], bool)
    lf = np.where(dark, -np.inf, np.asarray(truth["log10_flux_ratio"], float))
    m2 = np.asarray(truth["m2_msun"], float)
    out = np.zeros(r.size)
    w = ~rows.unit_weight[r]
    if w.any():
        rr = r[w]
        c1, mg1, ok = subtract_companion(rows.colour0[rr], rows.mg0[rr], lf[w], m2[w], ms)
        ll = log_ridge_likelihood(
            c1, mg1, rows.sigma_mu[rr], ridge, sigma_a_mag=cfg.sigma_a_mag, k_ag_over_ebprp=rows.k_ag_over_ebprp[rr]
        )
        out[w] = np.where(ok, ll, -np.inf) - norm.log_z[rr]
    return out


def ridge_residual(colour0: ArrayLike, mg0: ArrayLike, ridge: RidgeTables) -> FloatArray:
    """ΔM_2D = M_G0 − R(C0) (NaN outside the ridge colour range); the MP-Q25 replacement."""
    r, _, _, inr = ridge.at(colour0)
    return np.where(inr, np.asarray(mg0, float) - r, np.nan)
