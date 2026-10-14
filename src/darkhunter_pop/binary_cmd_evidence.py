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


def composite_points(pts: im.PriorPoints, grid: im.NativeGrid, q: float) -> im.PriorPoints:
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
        mg2[s] = np.where(lit, g2, np.inf)
        bp2[s] = np.where(lit, g2 + np.interp(m2c, mass, rel["bp_g"]), np.inf)
        rp2[s] = np.where(lit, g2 - np.interp(m2c, mass, rel["g_rp"]), np.inf)
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
    colour0: ArrayLike, mg0: ArrayLike, sigma_colour: ArrayLike, sigma_mag: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """(log10 E_S per row, log10 LR per row × q). LR is 0 (−inf) where the composite map has no
    support (below the evidence floor) but the single map does; NaN where E_S is undefined."""
    base = im.build_cmd_map(pts, cfg.cmd_map)
    les = log10_evidence(base, colour0, mg0, sigma_colour, sigma_mag, cfg.likelihood)
    out = np.full((les.size, len(q_grid)), np.nan)
    for j, q in enumerate(q_grid):
        cm = im.build_cmd_map(composite_points(pts, grid, float(q)), cfg.cmd_map)
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
