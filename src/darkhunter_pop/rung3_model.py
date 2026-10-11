"""Rung-3 fit-model pieces adopted on 2026-10-10 (#391; docs/MOCK_POPULATION_SPEC.md §12.15, §12.16).

* :func:`evolved_log_modifier` — MP-Q28 option A: λ → λ · f_evo on CMD-evolved parent rows (a reweightable target
  change; it never touches stored draws).
* :func:`spurious_probability` — π(x) of the §12.16 calibration (logistic in standardized covariates), with the
  MP-Q46 extrapolation choice made explicit: ``"clamp"`` (provisional default) or ``"logit"``.
* :func:`spurious_template` — the §12.16 shape T_b ∝ Σ_{real orbits i in b} π(x_i) and its total Σ π.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

Extrapolation = Literal["clamp", "logit"]


def evolved_log_modifier(is_evolved: ArrayLike, ln_f_evo: float) -> NDArray[np.float64]:
    """log of the §12.15 multiplier per draw: ``ln_f_evo`` on evolved rows, 0 elsewhere."""
    return np.where(np.asarray(is_evolved, bool), float(ln_f_evo), 0.0)


def spurious_probability(
    calibration: Mapping[str, Any],
    covariates: Mapping[str, ArrayLike],
    *,
    extrapolation: Extrapolation = "clamp",
    beta: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """π(x) from a ``scripts/calibrate_spurious_391.py`` calibration.

    ``calibration`` carries ``pi_keys``, ``pi_beta`` (intercept first), ``standardize_mean`` / ``standardize_sd`` and
    ``calibrated_range``. Rows with a non-finite selected covariate get the median π of the finite rows (never NaN).
    """
    keys = list(calibration["pi_keys"])
    b = np.asarray(calibration["pi_beta"] if beta is None else beta, float)
    n = len(np.asarray(next(iter(covariates.values()))))
    cols = []
    for k in keys:
        v = np.asarray(covariates[k], float)
        if extrapolation == "clamp":
            lo, hi = calibration["calibrated_range"][k]
            v = np.clip(v, lo, hi)
        elif extrapolation != "logit":
            raise ValueError(f"unknown extrapolation {extrapolation!r}")
        cols.append((v - calibration["standardize_mean"][k]) / calibration["standardize_sd"][k])
    if not cols:
        return np.full(n, float(expit(b[0])))
    Z = np.column_stack(cols)
    p = expit(b[0] + Z @ b[1:])
    ok = np.all(np.isfinite(Z), axis=1)
    fill = float(np.median(p[ok])) if ok.any() else float(expit(b[0]))
    return np.where(ok, p, fill)


def spurious_template(bin_of_orbit: ArrayLike, pi: ArrayLike, n_bins: int) -> tuple[NDArray[np.float64], float]:
    """(normalized T_b, Σ π) over real orbits with a valid bin index (``bin_of_orbit >= 0``)."""
    idx = np.asarray(bin_of_orbit, np.int64)
    p = np.asarray(pi, float)
    ok = idx >= 0
    t = np.bincount(idx[ok], weights=p[ok], minlength=n_bins).astype(float)
    total = float(t.sum())
    return (t / total if total > 0 else t), total
