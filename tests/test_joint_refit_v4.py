"""Rung 3 v4 population terms and the post-hoc routed type (#603, spec §12.42)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import betaln

from darkhunter_pop import joint_refit as jr

V4 = jr.V4TermsConfig(short_p_centre=2.0, short_p_width=0.1, ecc_m1_pivot_msun=1.0, cb_m1_pivot_msun=1.0)


@pytest.mark.physics
def test_ecc_shape_m1_term_keeps_the_beta_normalised() -> None:
    x = np.linspace(1e-6, 1 - 1e-6, 400_001)
    a, b = 1.6, 2.5
    base = np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b))
    for m1, beta in ((0.5, 1.3), (1.6, -0.7), (1.0, 2.0)):
        d = jr.ecc_shape_m1_log_ratio(np.full(x.size, a), np.full(x.size, b), np.log1p(-x), np.ones(x.size, bool), np.full(x.size, m1), beta, V4)
        assert np.trapz(base * np.exp(d), x) == pytest.approx(1.0, abs=1e-4)
    assert np.all(jr.ecc_shape_m1_log_ratio([a], [b], [-0.3], [True], [1.0], 2.0, V4) == 0.0)  # M1 at the pivot
    assert jr.ecc_shape_m1_log_ratio([a], [b], [-0.3], [False], [2.0], 2.0, V4)[0] == 0.0  # outside the Beta support
    # beta_e > 0 makes a high-M1 primary less eccentric: weight drops at large x, rises at small x
    hi, lo = jr.ecc_shape_m1_log_ratio([a, a], [b, b], np.log1p(-np.array([0.9, 0.1])), [True, True], [1.5, 1.5], 1.0, V4)
    assert hi < 0 < lo


@pytest.mark.physics
def test_twin_short_p_term_preserves_the_frequency_per_draw() -> None:
    rng = np.random.default_rng(1)
    qg = np.linspace(0.3, 1.0, 201)[:-1] + 0.35 / 200
    twin_g = qg >= 0.95
    pq = rng.random(qg.size)
    pq /= pq.sum()
    dq, lnT = 0.4, 0.3
    g = np.exp(dq * np.log(qg) + lnT * twin_g)
    p_old = pq * g / (pq * g).sum()  # q distribution of the existing model
    for s, d in ((1.0, -2.0), (0.5, 1.0), (0.0, -3.0)):
        a = (pq * np.exp(dq * np.log(qg)) * ~twin_g).sum()
        b = (pq * np.exp(dq * np.log(qg)) * twin_g).sum()
        dl = jr.twin_short_p_log_ratio(twin_g, np.full(qg.size, s), np.full(qg.size, a), np.full(qg.size, b), lnT, d)
        assert (p_old * np.exp(dl)).sum() == pytest.approx(1.0, rel=1e-12)
        if s == 0.0:
            assert np.allclose(dl, 0.0)


@pytest.mark.physics
def test_short_period_weight_and_close_binary_term() -> None:
    s = jr.short_period_weight([1.0, 2.0, 3.0], V4)
    assert s[0] == pytest.approx(1.0, abs=1e-4) and s[1] == pytest.approx(0.5) and s[2] == pytest.approx(0.0, abs=1e-4)
    f = jr.close_binary_log_factor([0.5, 1.0, 1.5], [1.0, 1.0, 1.0], -2.0, V4)
    assert f.tolist() == pytest.approx([-2.0 * np.log(0.5), 0.0, 0.0])
    assert np.all(jr.close_binary_log_factor([0.5], [0.0], -2.0, V4) == 0.0)


class _FakeST:
    def p_asb1(self, g, bp, p, e, k1):
        return 0.5


@pytest.mark.unit
def test_routed_asb1_skips_non_orbital_branches_and_is_deterministic() -> None:
    c = np.zeros((3, 23))
    c[0, [0, 10, 14, 16, 17]] = [2.0, 300.0, 0.3, 60.0, 1.0]
    c[1, 0] = -9.0  # acceleration branch
    c[2, 0] = 0.0  # insufficient visibility
    a = jr.routed_asb1([12.0] * 3, [1.0] * 3, c, [53] * 3, [10_000_001] * 3, 391, _FakeST())
    b = jr.routed_asb1([12.0] * 3, [1.0] * 3, c, [53] * 3, [10_000_001] * 3, 391, _FakeST())
    assert a[0] in (0.0, 1.0) and np.isnan(a[1]) and np.isnan(a[2])
    assert np.array_equal(a, b, equal_nan=True)
