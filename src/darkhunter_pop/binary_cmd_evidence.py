"""CMD evidence for a luminous companion: single star vs star + MdS17 companion (spec §12.22).

Under MP-Q6 a parent row's observed photometry is the *system*. The mock's drawn companion
enters through the deblended M1 (MP-Q36) but, before §12.22, the observed CMD did not
condition *whether* a row has a (bright) companion. This module supplies that conditioning
as a reweight, with the same prior, kernel and σ (incl. the isochrone fit's intrinsic floors)
as the single-star isochrone fit:

* ``E_S(x)``: the single-star prior-predictive density at the row's dereddened (C0, M_G0),
  i.e. the isochrone fit's evidence (:func:`darkhunter_pop.isochrone_mass.posterior_moments`).
* ``E_B(x | q)``: the same with every prior point replaced by the composite of that point and a
  coeval main-sequence companion of mass q·M (MIST photometry at the point's nearest native
  ([Fe/H], age)); the IMF/age/[Fe/H] prior on the primary is unchanged.
* ``LR(q) = E_B / E_S``.

With the row's MdS17 companion intensity Λ(q) (per q cell, integrated over log P in the
target domain) and F = Σ Λ, the per-draw weight factor is

    factor = LR(q_draw) · (1 + F) / (1 + Σ_q Λ(q) LR(q)),

the single-companion posterior P(companion at q | x) = Λ LR / (1 + Σ Λ LR), normalized so an
uninformative CMD (LR ≡ 1) reproduces the current weights exactly. The flux-ratio scatter of
MP-Q13 is not propagated into LR (the composite uses the MIST relation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop import isochrone_mass as im

FloatArray = NDArray[np.float64]


def _nearest(grid_axis: ArrayLike, x: ArrayLike) -> NDArray[np.int64]:
    """Index of the nearest value of the sorted ``grid_axis`` for each ``x``."""
    g = np.asarray(grid_axis, float)
    v = np.asarray(x, float)
    if g.size == 1:
        return np.zeros(v.size, np.int64)
    j = np.clip(np.searchsorted(g, v), 1, g.size - 1)
    return np.where(np.abs(v - g[j - 1]) <= np.abs(g[j] - v), j - 1, j).astype(np.int64)


def composite_points(pts: im.PriorPoints, grid: im.NativeGrid, q: float, *,
                     companion_shift: tuple[FloatArray, FloatArray, float, float] | None = None) -> im.PriorPoints:
    """Prior points with a coeval MS companion of mass ``q`` × the point's current mass added.

    The companion's (M_G, BP, RP) come from :func:`isochrone_mass.main_sequence_colour_relation`
    at the point's nearest native ([Fe/H], age); a companion below the MS mass range adds no
    light, one above it is clamped to the MS top. Only ``colour`` and ``mg`` change.
    """
    m = np.asarray(pts.values["star_mass"], float)
    fi = _nearest(grid.feh, pts.feh)
    ai = _nearest(grid.log_age, pts.log_age)
    key = fi * grid.log_age.size + ai
    mg = np.asarray(pts.mg, float)
    bp = mg + (np.asarray(pts.values["bp"], float) - np.asarray(pts.values["mg"], float))
    rp = mg - (np.asarray(pts.values["mg"], float) - np.asarray(pts.values["rp"], float))
    mg2 = np.full(m.size, np.inf)
    bp2 = np.full(m.size, np.inf)
    rp2 = np.full(m.size, np.inf)
    for k in np.unique(key):
        s = np.flatnonzero(key == k)
        mass, rel = im.main_sequence_colour_relation(grid, float(grid.feh[k // grid.log_age.size]),
                                                     float(grid.log_age[k % grid.log_age.size]))
        if mass.size < 2:
            continue
        m2 = q * m[s]
        lit = m2 >= mass[0]
        m2c = np.clip(m2, mass[0], mass[-1])
        g2 = np.interp(m2c, mass, rel["mg"])
        dc2 = 0.0
        if companion_shift is not None:  # §12.22 (b): the companion on the same empirical MS locus
            smg, sdc, jlo, jhi = companion_shift
            dc2 = mass_blend(m2c, jlo, jhi) * np.interp(g2, smg, sdc)
        mg2[s] = np.where(lit, g2, np.inf)
        bp2[s] = np.where(lit, g2 + np.interp(m2c, mass, rel["bp_g"]) + 0.5 * dc2, np.inf)
        rp2[s] = np.where(lit, g2 - np.interp(m2c, mass, rel["g_rp"]) - 0.5 * dc2, np.inf)
    fl = lambda x: 10.0 ** (-0.4 * x)  # noqa: E731
    mg_c = -2.5 * np.log10(fl(mg) + fl(mg2))
    col_c = -2.5 * np.log10(fl(bp) + fl(bp2)) + 2.5 * np.log10(fl(rp) + fl(rp2))
    return replace(pts, colour=col_c, mg=mg_c)


def log10_evidence(cmap: im.CmdMap, colour0: ArrayLike, mg0: ArrayLike, sigma_colour: ArrayLike,
                   sigma_mag: ArrayLike, lk: Any) -> FloatArray:
    """The isochrone fit's log10 prior-predictive density on ``cmap`` (NaN off grid)."""
    post = im.posterior_moments(colour0, mg0, sigma_colour, sigma_mag, cmap, lk)
    return np.asarray(post.log10_evidence, float)


def log10_lr_grid(
    pts: im.PriorPoints, grid: im.NativeGrid, cfg: im.IsochroneMassConfig, q_grid: Sequence[float],
    colour0: ArrayLike, mg0: ArrayLike, sigma_colour: ArrayLike, sigma_mag: ArrayLike, *,
    companion_shift: tuple[FloatArray, FloatArray, float, float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """(log10 E_S per row, log10 LR per row × q). LR is 0 (−inf) where the composite map has no
    support (below the evidence floor) but the single map does; NaN where E_S is undefined."""
    base = im.build_cmd_map(pts, cfg.cmd_map)
    les = log10_evidence(base, colour0, mg0, sigma_colour, sigma_mag, cfg.likelihood)
    out = np.full((les.size, len(q_grid)), np.nan)
    for j, q in enumerate(q_grid):
        cm = im.build_cmd_map(composite_points(pts, grid, float(q), companion_shift=companion_shift), cfg.cmd_map)
        leb = log10_evidence(cm, colour0, mg0, sigma_colour, sigma_mag, cfg.likelihood)
        out[:, j] = np.where(np.isfinite(leb), leb - les, -np.inf)
    out[~np.isfinite(les)] = np.nan
    return les, out


def row_factor(lam: ArrayLike, log10_lr: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Per row: (1 + F) / (1 + Σ Λ LR) and F = Σ Λ, from Λ (rows × q) and log10 LR (rows × q).
    Rows with an undefined LR get factor 1."""
    lam_ = np.asarray(lam, float)
    llr = np.asarray(log10_lr, float)
    F = lam_.sum(axis=1)
    G = np.nansum(lam_ * np.where(np.isfinite(llr) | np.isneginf(llr), 10.0 ** llr, 1.0), axis=1)
    bad = ~np.all(np.isfinite(llr) | np.isneginf(llr), axis=1)
    return np.where(bad, 1.0, (1.0 + F) / (1.0 + G)), F


def draw_factor(q_draw: ArrayLike, row_of_draw: ArrayLike, q_grid: Sequence[float], log10_lr: ArrayLike,
                row_norm: ArrayLike) -> FloatArray:
    """Per draw: LR(q_draw) (linear interpolation of log10 LR in q, clamped) × row normalization.
    Undefined rows (NaN LR) get 1."""
    qg = np.asarray(q_grid, float)
    llr = np.asarray(log10_lr, float)[np.asarray(row_of_draw, np.int64)]
    q = np.clip(np.asarray(q_draw, float), qg[0], qg[-1])
    j = np.clip(np.searchsorted(qg, q) - 1, 0, qg.size - 2)
    t = (q - qg[j]) / (qg[j + 1] - qg[j])
    a, b = llr[np.arange(q.size), j], llr[np.arange(q.size), j + 1]
    with np.errstate(invalid="ignore"):
        v = np.where(np.isneginf(a) | np.isneginf(b), -np.inf, (1 - t) * a + t * b)
    lr = np.where(np.isfinite(v) | np.isneginf(v), 10.0 ** v, np.nan)
    norm = np.asarray(row_norm, float)[np.asarray(row_of_draw, np.int64)]
    return np.where(np.isfinite(lr), lr * norm, 1.0)


def q_intensity_grid(m1: ArrayLike, q_edges: ArrayLike, target: Any, *, n_logp: int = 60) -> FloatArray:
    """Λ(q cell) per row: ∫ dlogP f_logP;q>0.3(M1, P) s_low(M1) ∫_cell p_q(q | M1, P) dq over the
    MdS17 log P domain; rows × cells (midpoint rule in q with 4 sub-points per cell)."""
    from darkhunter_pop import moe_distefano as mds

    table = mds.load_mds17_table(target.table_path)
    m = np.asarray(m1, float)
    ms = np.clip(m, table.m1_range[0], table.m1_range[1])
    lp = np.linspace(table.log_p_range[0], table.log_p_range[1], n_logp + 1)
    lpc = 0.5 * (lp[1:] + lp[:-1])
    dlp = np.diff(lp)
    qe = np.asarray(q_edges, float)
    out = np.zeros((m.size, qe.size - 1))
    slow = mds.low_mass_frequency_scale(m, m1_anchor_msun=target.provisional_low_mass_anchor_msun,
                                        m1_zero_msun=target.provisional_low_mass_zero_msun)
    for i, (x, dx) in enumerate(zip(lpc, dlp)):
        f = mds.f_logp_q03(ms, np.full(m.size, x), table) * slow
        for j in range(qe.size - 1):
            sub = qe[j] + (np.arange(4) + 0.5) / 4 * (qe[j + 1] - qe[j])
            pq = np.mean([mds.q_density(np.full(m.size, s), ms, np.full(m.size, x), table,
                                        m1_interpolation=target.provisional_m1_interpolation) for s in sub], axis=0)
            out[:, j] += f * pq * (qe[j + 1] - qe[j]) * dx
    return out


# ---------------------------------------------------------------------------
# θ-dependent normalization (spec §12.22 amendment, Ryan 2026-10-13)
# ---------------------------------------------------------------------------


def precompute_theta_grids(m1: ArrayLike, q_edges: ArrayLike, target: Any, *, n_logp: int = 30,
                           n_qnorm: int = 40, q_norm_min: float = 0.3) -> dict[str, Any]:
    """Baseline MdS17 pieces per row for :func:`theta_row_norm`.

    ``lam0[r, p, k]``: f_logP;q>0.3 · s_low · ∫_cell_k p_q dq · ΔlogP on the full MdS17 log P
    domain; ``pqn[r, p, j]``: p_q(qn_j) Δq on the MP-Q42b normalization grid over
    [``q_norm_min``, 1]; centres ``lpc``, ``qc``, ``qn``.
    """
    from darkhunter_pop import moe_distefano as mds

    table = mds.load_mds17_table(target.table_path)
    m = np.asarray(m1, float)
    ms = np.clip(m, table.m1_range[0], table.m1_range[1])
    lp = np.linspace(table.log_p_range[0], table.log_p_range[1], n_logp + 1)
    lpc, dlp = 0.5 * (lp[1:] + lp[:-1]), np.diff(lp)
    qe = np.asarray(q_edges, float)
    qne = np.linspace(q_norm_min, 1.0, n_qnorm + 1)
    qn, dqn = 0.5 * (qne[1:] + qne[:-1]), np.diff(qne)
    slow = mds.low_mass_frequency_scale(m, m1_anchor_msun=target.provisional_low_mass_anchor_msun,
                                        m1_zero_msun=target.provisional_low_mass_zero_msun)
    interp = target.provisional_m1_interpolation
    lam0 = np.zeros((m.size, lpc.size, qe.size - 1))
    pqn = np.zeros((m.size, lpc.size, qn.size))
    for p, (x, dx) in enumerate(zip(lpc, dlp)):
        xx = np.full(m.size, x)
        f = mds.f_logp_q03(ms, xx, table) * slow * dx
        for k in range(qe.size - 1):
            sub = qe[k] + (np.arange(4) + 0.5) / 4 * (qe[k + 1] - qe[k])
            pq = np.mean([mds.q_density(np.full(m.size, s), ms, xx, table, m1_interpolation=interp) for s in sub], axis=0)
            lam0[:, p, k] = f * pq * (qe[k + 1] - qe[k])
        for j, (qq, dq) in enumerate(zip(qn, dqn)):
            pqn[:, p, j] = mds.q_density(np.full(m.size, qq), ms, xx, table, m1_interpolation=interp) * dq
    return {"lam0": lam0, "pqn": pqn, "lpc": lpc, "qc": 0.5 * (qe[1:] + qe[:-1]), "qn": qn}


def theta_row_norm(grids: dict[str, Any], log10_lr: ArrayLike, log_s_row: ArrayLike, log_m_p: ArrayLike,
                   dgamma_q: float, ln_f_twin: float, twin_q_min: float) -> FloatArray:
    """Per row (1 + F(θ)) / (1 + Σ_q Λ(q; θ) LR(q)) with
    Λ(q; θ) = e^{log_s_row} Σ_P Λ⁰(P, q) e^{log_m_p(P)} g(q) / N(P), g = q^{Δγ_q} e^{ln F_tw [q ≥ q_tw]},
    N(P) = Σ_j p_q(qn_j) g(qn_j) Δq (MP-Q42b). Rows with an undefined LR get 1."""
    g = lambda q: np.exp(dgamma_q * np.log(q) + ln_f_twin * (q >= twin_q_min))  # noqa: E731
    gq, gn = g(grids["qc"]), g(grids["qn"])
    N = grids["pqn"] @ gn  # (R, P)
    lam = np.einsum("rpk,rp->rk", grids["lam0"], np.exp(np.asarray(log_m_p, float))[None, :] / N) * gq[None, :]
    lam *= np.exp(np.asarray(log_s_row, float))[:, None]
    llr = np.asarray(log10_lr, float)
    L = np.where(np.isneginf(llr), 0.0, 10.0 ** np.where(np.isfinite(llr), llr, 0.0))
    bad = ~np.all(np.isfinite(llr) | np.isneginf(llr), axis=1)
    out = (1.0 + lam.sum(axis=1)) / (1.0 + (lam * L).sum(axis=1))
    return np.where(bad, 1.0, out)


# ---------------------------------------------------------------------------
# Empirical single-star MS locus at low mass (spec §12.22 second amendment, (b))
# ---------------------------------------------------------------------------


def ridge_colour_shift(pts: im.PriorPoints, ridge_colour: ArrayLike, ridge_mag: ArrayLike, *, join_hi_msun: float,
                       mg_edges: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Δc(M_G) = c_ridge(M_G) − c_MIST(M_G) on ``mg_edges`` centres: c_MIST is the prior-weighted
    median colour of MIST MS points (phase < 0.5, M < ``join_hi_msun``) per M_G bin, c_ridge the
    measured ridge colour at that M_G (ridge inverted in M_G; NaN outside its range → Δc = 0)."""
    mg = np.asarray(pts.mg, float)
    ms = (np.asarray(pts.values["phase"], float) < 0.5) & (np.asarray(pts.values["star_mass"], float) < join_hi_msun)
    e = np.asarray(mg_edges, float)
    cen = 0.5 * (e[1:] + e[:-1])
    cm = np.full(cen.size, np.nan)
    for k, (a, b) in enumerate(zip(e[:-1], e[1:])):
        s = ms & (mg >= a) & (mg < b)
        if s.sum() >= 10:
            c, w = pts.colour[s], pts.weight[s]
            o = np.argsort(c)
            cm[k] = float(np.interp(0.5, np.cumsum(w[o]) / w[o].sum(), c[o]))
    rc, rm = np.asarray(ridge_colour, float), np.asarray(ridge_mag, float)
    o = np.argsort(rm)
    cr = np.interp(cen, rm[o], rc[o], left=np.nan, right=np.nan)
    d = np.where(np.isfinite(cr) & np.isfinite(cm), cr - cm, 0.0)
    return cen, d


def mass_blend(m: ArrayLike, join_lo_msun: float, join_hi_msun: float) -> FloatArray:
    """1 at M ≤ join_lo, linear to 0 at join_hi, 0 above."""
    return np.clip((join_hi_msun - np.asarray(m, float)) / (join_hi_msun - join_lo_msun), 0.0, 1.0)


def ridge_corrected_points(pts: im.PriorPoints, shift_mg: FloatArray, shift_dc: FloatArray, *, join_lo_msun: float,
                           join_hi_msun: float) -> im.PriorPoints:
    """MS points moved in colour by t(M) Δc(M_G) (G kept: BP += Δ/2, RP −= Δ/2)."""
    m = np.asarray(pts.values["star_mass"], float)
    ms = np.asarray(pts.values["phase"], float) < 0.5
    dc = np.where(ms, mass_blend(m, join_lo_msun, join_hi_msun) * np.interp(pts.mg, shift_mg, shift_dc), 0.0)
    vals = dict(pts.values)
    vals["bp"] = np.asarray(vals["bp"], float) + 0.5 * dc
    vals["rp"] = np.asarray(vals["rp"], float) - 0.5 * dc
    return replace(pts, colour=np.asarray(pts.colour, float) + dc, values=vals)
