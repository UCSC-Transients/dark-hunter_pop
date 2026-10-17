"""Spec §12.27: restricted-support deterministic mixture is unbiased; R mask logic."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from darkhunter_pop import reuse_region as rr


@pytest.mark.unit
def test_region_mask() -> None:
    m = rr.region_mask([0.005, 0.02, 0.0, 0.0, np.nan], [0.01, 0.0, 0.5, 0.0, 0.0], [False, False, False, True, False],
                       [False, False, True, False, False], tol_dlnm1=0.01, tol_dlogf=0.02)
    np.testing.assert_array_equal(m, [True, False, True, False, False])  # dark draw: f change ignored


@pytest.mark.unit
def test_restricted_mixture_unbiased() -> None:
    """Old proposal U(-3, 3) restricted to R = {x < 0}; new proposal N(0, 2). The combined estimator of
    E_target[g] over (-3, 3) (target N(0, 1)) is unbiased; old draws outside R get zero weight."""
    rng = np.random.default_rng(1)
    n_old, n_new = 40000, 20000
    xo = rng.uniform(-3, 3, n_old)
    xn = rng.normal(0, 2, n_new)
    xn = xn[np.abs(xn) < 3]
    x = np.concatenate([xo, xn])
    lt = norm.logpdf(x)
    lq_old = np.where(np.abs(x) < 3, np.log(1 / 6), -np.inf)
    lq_new = norm.logpdf(x, 0, 2) - np.log(norm.cdf(3, 0, 2) - norm.cdf(-3, 0, 2))
    inr = x < 0
    w = rr.restricted_mixture_weights(lt, [lq_old, lq_new], [n_old, xn.size], [True, False], inr,
                                      np.r_[np.ones(n_old, bool), np.zeros(xn.size, bool)], scale_to_full=1.0)
    assert np.all(w[:n_old][~inr[:n_old]] == 0)
    g = x**2
    est = np.sum(w * g)
    truth = norm.expect(lambda t: t**2, lb=-3, ub=3)
    assert est == pytest.approx(truth, rel=0.03)
    # the R-side alone also integrates correctly (old + new both contribute there)
    assert np.sum(w[inr] * g[inr]) == pytest.approx(truth / 2, rel=0.03)


@pytest.mark.unit
def test_needs_dark_deblend_any_generation() -> None:
    """The combined cache must deblend every block when ANY generation needs it (the old-only gen 28-style
    proposal is evaluated at new draws too) -- the 2026-10-16 combined-cache failure."""
    from types import SimpleNamespace as N

    g = lambda c: N(flux=N(evolved_rows_centre=c))  # noqa: E731
    assert rr.needs_dark_deblend([g("evolved_relation"), g("evolved_relation_deblended")])
    assert not rr.needs_dark_deblend([g("evolved_relation"), g("dwarf_relation")])
