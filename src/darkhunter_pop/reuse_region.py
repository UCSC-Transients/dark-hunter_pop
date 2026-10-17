"""Spec §12.27: interim reuse of old proposal draws inside the unchanged region R.

A draw is in R when its generation-time inputs agree between two model versions (the one it was
generated with and the other), recomputed from the draw's own stored random content and seeds:
the deblended M1 (|Δ ln M1| < tol), the row-level coeval MIST flux-ratio relation (|Δ log10 f| <
tol), and no resolved companion under the current model. Old generations then enter the
deterministic mixture with their proposal restricted to R (:func:`restricted_mixture_weights`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Truth columns produced by posterior deblending that are dropped before re-deblending.
_DEBLEND_COLUMNS = ("m1_hat_msun", "iso_phase", "iso_feh", "iso_log_age", "deblend_chi2", "m1_row_msun",
                    "m2_proposal_msun", "m1_deblend_dark_msun")


def recompute_under(truth: Mapping[str, NDArray[Any]], prop: Any, parent_other: Any, cfg_other: Any,
                    native_other: Any) -> dict[str, FloatArray]:
    """The draw's deblended M1 and row-level flux relation under another model version, keeping its
    proposal q (= m2_proposal / m1_row), P, e and seeds. Returns ``m1`` and ``relation_row``."""
    from darkhunter_pop import proposal_set as ps

    r = np.asarray(truth["parent_row"], np.int64)
    q = np.asarray(truth["m2_proposal_msun"], float) / np.asarray(truth["m1_row_msun"], float)
    m1row = np.asarray(parent_other.isochrone["m1_mean"], float)[r]
    base = {k: np.asarray(v) for k, v in truth.items() if k not in _DEBLEND_COLUMNS}
    base["m1_msun"], base["m2_msun"] = m1row, q * m1row
    deb = ps.apply_posterior_deblending(base, parent_other, cfg_other, prop, native=native_other)
    rel = ps.mist_relation_for_draws({"parent_row": r, "m1_msun": m1row, "m2_msun": q * m1row}, parent_other, cfg_other,
                                     native=native_other)
    return {"m1": np.asarray(deb["m1_msun"], float), "relation_row": np.asarray(rel, float)}


def row_relation(truth: Mapping[str, NDArray[Any]], parent: Any, cfg: Any, native: Any) -> FloatArray:
    """The draw's row-level coeval MIST relation under its own model version."""
    from darkhunter_pop import proposal_set as ps

    r = np.asarray(truth["parent_row"], np.int64)
    return np.asarray(ps.mist_relation_for_draws({"parent_row": r, "m1_msun": np.asarray(truth["m1_row_msun"], float),
                                                  "m2_msun": np.asarray(truth["m2_proposal_msun"], float)}, parent, cfg, native=native), float)


def region_mask(dlnm1: ArrayLike, dlogf: ArrayLike, resolved: ArrayLike, is_dark: ArrayLike, *, tol_dlnm1: float,
                tol_dlogf: float) -> NDArray[np.bool_]:
    """1_R: |Δ ln M1| < tol, |Δ log f| < tol (dark draws: f unchanged), and not resolved."""
    dm = np.asarray(dlnm1, float)
    df = np.where(np.asarray(is_dark, bool), 0.0, np.asarray(dlogf, float))
    return np.isfinite(dm) & (np.abs(dm) < tol_dlnm1) & np.isfinite(df) & (np.abs(df) < tol_dlogf) & ~np.asarray(resolved, bool)


def restricted_mixture_weights(log_target: ArrayLike, log_q_components: Sequence[ArrayLike], n_per_generation: Sequence[int],
                               restricted: Sequence[bool], in_region: ArrayLike, draw_is_restricted: ArrayLike, *,
                               scale_to_full: float) -> FloatArray:
    """Deterministic-mixture weights with some generations' proposals restricted to R (spec §12.27).

    w(x) = scale λ(x) / [Σ_j n_j q_j(x) 1_R(x)^{restricted_j}]; draws from a restricted generation
    outside R get 0. ``draw_is_restricted`` marks draws that came from a restricted generation.
    """
    from darkhunter_pop import proposal_set as ps

    inr = np.asarray(in_region, bool)
    comps = [np.where(inr, np.asarray(c, float), -np.inf) if rj else np.asarray(c, float)
             for c, rj in zip(log_q_components, restricted)]
    lt = np.asarray(log_target, float)
    keep = ~(np.asarray(draw_is_restricted, bool) & ~inr)
    w = np.zeros(lt.size)
    if keep.any():
        w[keep] = ps.importance_weights(lt[keep], [c[keep] for c in comps], list(n_per_generation), scale_to_full=scale_to_full)
    return w


def needs_dark_deblend(gen_cfgs: Sequence[Any]) -> bool:
    """True if any generation's proposal uses ``evolved_rows_centre: evolved_relation_deblended``. Then every
    block of a combined cache needs ``m1_deblend_dark_msun`` at all of its draws, because each generation's
    log q is evaluated at every draw (deterministic-mixture cross-evaluation)."""
    return any(getattr(g.flux, "evolved_rows_centre", None) == "evolved_relation_deblended"
               or getattr(getattr(g, "m2", None), "evolved_tie", "row_m1") == "deblended_dark" for g in gen_cfgs)
