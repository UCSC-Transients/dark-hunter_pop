"""Equivalence of the vectorized ``F = 0`` companion-mass solver with the old path (#283).

``invert_astrometric_companion_mass`` used to seed its Newton polish with a
per-element ``np.roots`` call. #283 replaced that seed, for ``F == 0`` only, with
a vectorized bracketed Newton. ``_reference_invert`` below is a verbatim copy of
the pre-#283 implementation (``main`` @ ``1cd27ce``), kept **only** as a test
oracle.

Tolerance: the issue's acceptance bound is <= 1e-12 relative. Both paths feed
the same Newton polish, so they differ only through the seed; in practice the
results agree to a few ulp (the tests also assert that the large majority are
bit-identical). 1e-12 leaves ~4 orders of headroom over the observed max while
still being ~10^4 tighter than anything a mass measurement could resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import physics_utils as P

pytestmark = pytest.mark.physics

REL_TOL = 1e-12


def _reference_invert(m1_msun, m_f_msun, flux_ratio=0.0, *, tol=1e-12, max_iter=80):
    """Pre-#283 implementation (per-element ``np.roots`` seed). Test oracle only."""
    m1 = np.asarray(m1_msun, dtype=np.float64)
    mf = np.asarray(m_f_msun, dtype=np.float64)
    flux = np.asarray(flux_ratio, dtype=np.float64)
    m1_b, mf_b, f_b = np.broadcast_arrays(m1, mf, flux)
    out_shape = m1_b.shape
    m1_r = np.ravel(m1_b)
    mf_r = np.ravel(mf_b)
    f_r = np.ravel(f_b)
    q = np.full(m1_r.shape, np.nan, dtype=np.float64)
    y = np.full(m1_r.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(m1_r)
        & np.isfinite(mf_r)
        & np.isfinite(f_r)
        & (m1_r > 0.0)
        & (mf_r > 0.0)
        & (f_r >= 0.0)
    )
    if not np.any(valid):
        return q.reshape(out_shape)
    y[valid] = mf_r[valid] / m1_r[valid]
    for idx in np.flatnonzero(valid):
        f_i = float(f_r[idx])
        a_i = float(y[idx]) * (1.0 + f_i) ** 3
        coeffs = (
            1.0,
            -3.0 * f_i - a_i,
            3.0 * f_i * f_i - 2.0 * a_i,
            -(f_i**3) - a_i,
        )
        roots = np.roots(coeffs)
        real = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)])
        real = real[real > f_i]
        if real.size == 0:
            continue
        q[idx] = float(np.min(real))
    one_f3 = (1.0 + f_r) ** 3
    for _ in range(max_iter):
        dq = q - f_r
        one_q = 1.0 + q
        resid = dq**3 - y * one_f3 * one_q**2
        deriv = 3.0 * dq**2 - y * one_f3 * 2.0 * one_q
        movable = (
            valid
            & np.isfinite(q)
            & np.isfinite(resid)
            & np.isfinite(deriv)
            & (np.abs(deriv) > 0.0)
        )
        step = np.zeros_like(q)
        step[movable] = resid[movable] / deriv[movable]
        trial = q - step
        trial = np.where(trial <= f_r, 0.5 * (q + f_r) + 0.5 * np.maximum(q - f_r, 0.1), trial)
        q = np.where(movable, trial, q)
        if np.all(~movable | (np.abs(resid) < tol)):
            break
    q = np.where(valid & np.isfinite(q) & (q > f_r), q, np.nan)
    return (q * m1_r).reshape(out_shape)


def _assert_equivalent(new, old, *, min_identical_frac=0.0):
    new = np.asarray(new)
    old = np.asarray(old)
    assert new.shape == old.shape
    np.testing.assert_array_equal(np.isnan(new), np.isnan(old))
    np.testing.assert_array_equal(np.isinf(new), np.isinf(old))
    fin = np.isfinite(old)
    if np.any(fin):
        rel = np.abs(new[fin] - old[fin]) / np.abs(old[fin])
        assert float(np.max(rel)) <= REL_TOL, float(np.max(rel))
        assert np.mean(new[fin] == old[fin]) >= min_identical_frac


def _silent(fn, *args, **kwargs):
    with np.errstate(all="ignore"):
        return fn(*args, **kwargs)


def test_random_wide_range_matches_reference() -> None:
    rng = np.random.default_rng(283)
    n = 20_000
    m1 = 10.0 ** rng.uniform(-2.0, 2.0, n)
    y = 10.0 ** rng.uniform(-12.0, 12.0, n)
    mf = y * m1
    new = _silent(P.invert_astrometric_companion_mass, m1, mf, 0.0)
    old = _silent(_reference_invert, m1, mf, 0.0)
    assert np.all(np.isfinite(old))
    _assert_equivalent(new, old, min_identical_frac=0.5)


def test_production_like_mc_ensemble_matches_reference() -> None:
    """Per-draw M1 and log-normal m_f around realistic Orbital values."""
    rng = np.random.default_rng(10_000)
    for m1_c, mf_c in [(1.0, 1e-3), (0.8, 0.3), (1.2, 5.0), (0.5, 1e-6), (2.0, 50.0)]:
        m1 = np.clip(rng.normal(m1_c, 0.1 * m1_c, 10_000), np.finfo(float).tiny, None)
        mf = mf_c * np.exp(rng.normal(0.0, 0.5, 10_000))
        new = _silent(P.invert_astrometric_companion_mass, m1, mf, 0.0)
        old = _silent(_reference_invert, m1, mf, 0.0)
        _assert_equivalent(new, old, min_identical_frac=0.5)


def test_extreme_finite_inputs_match_reference() -> None:
    tiny = np.finfo(np.float64).tiny
    ys = np.array(
        [5e-324, 1e-320, tiny, 1e-300, 1e-200, 1e-100, 1e-30, 1e-16, 0.2499999, 0.25,
         0.2500001, 1.0, 1e16, 1e100, 1e200, 1e300, 4e307, 8e307, 8.98e307]
    )
    new = _silent(P.invert_astrometric_companion_mass, 1.0, ys, 0.0)
    old = _silent(_reference_invert, 1.0, ys, 0.0)
    _assert_equivalent(new, old)
    m1s = np.array([1e-300, 1e-100, 1e100, 1e300])
    for mf in (1e-300, 1.0, 1e300):
        mfv = np.full(m1s.shape, mf)
        with np.errstate(all="ignore"):
            yv = mfv / m1s
        keep = np.isfinite(2.0 * yv)
        new = _silent(P.invert_astrometric_companion_mass, m1s[keep], mfv[keep], 0.0)
        old = _silent(_reference_invert, m1s[keep], mfv[keep], 0.0)
        _assert_equivalent(new, old)


def test_invalid_and_degenerate_inputs_give_same_markers() -> None:
    m1 = np.array([1.0, 0.0, -1.0, np.nan, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e300])
    mf = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0, -0.1, np.nan, np.inf, -np.inf, 0.1, 0.1, 1e-300])
    fl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, np.nan, 0.0])
    new = _silent(P.invert_astrometric_companion_mass, m1, mf, fl)
    old = _silent(_reference_invert, m1, mf, fl)
    _assert_equivalent(new, old)
    # m_f/M1 underflows to 0 (last element) -> NaN on both paths.
    assert np.isnan(new[-1]) and np.all(np.isnan(new[1:])) and np.isfinite(new[0])
    # all-invalid early return and negative-zero flux ratio
    assert np.isnan(P.invert_astrometric_companion_mass(-1.0, 1.0, 0.0))
    _assert_equivalent(
        P.invert_astrometric_companion_mass(1.3, 0.2, -0.0), _reference_invert(1.3, 0.2, -0.0)
    )


def test_overflowing_coefficient_raises_like_np_roots() -> None:
    for m1, mf in [(1e-300, 1e10), (1.0, 1.7e308)]:
        with pytest.raises(np.linalg.LinAlgError):
            _silent(_reference_invert, [1.0, m1], [0.1, mf], 0.0)
        with pytest.raises(np.linalg.LinAlgError):
            _silent(P.invert_astrometric_companion_mass, [1.0, m1], [0.1, mf], 0.0)


def test_shapes_and_broadcasting_match_reference() -> None:
    rng = np.random.default_rng(7)
    m1 = rng.uniform(0.3, 3.0, (4, 1, 5))
    mf = 10.0 ** rng.uniform(-6.0, 2.0, (1, 3, 5))
    new = P.invert_astrometric_companion_mass(m1, mf, 0.0)
    old = _reference_invert(m1, mf, 0.0)
    assert new.shape == (4, 3, 5)
    _assert_equivalent(new, old)
    assert np.ndim(P.invert_astrometric_companion_mass(1.0, 0.1, 0.0)) == 0


def test_mixed_flux_ratio_array_matches_reference() -> None:
    """F > 0 keeps the np.roots seed; mixing it with F == 0 changes nothing."""
    rng = np.random.default_rng(11)
    n = 2_000
    m1 = rng.uniform(0.3, 3.0, n)
    mf = 10.0 ** rng.uniform(-8.0, 3.0, n)
    fl = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.uniform(0.0, 0.3, n))
    new = _silent(P.invert_astrometric_companion_mass, m1, mf, fl)
    old = _silent(_reference_invert, m1, mf, fl)
    _assert_equivalent(new, old, min_identical_frac=0.5)


def test_guess_brackets_and_solves_cubic() -> None:
    a = 10.0 ** np.linspace(-300.0, 300.0, 6001)
    q = P._dark_companion_q_guess(a)
    lo = np.maximum(a, np.cbrt(a))
    hi = np.maximum(4.0 * a, np.cbrt(4.0 * a))
    assert np.all(q >= lo * (1 - 1e-15)) and np.all(q <= hi * (1 + 1e-15))
    # residual of q^3/(1+q)^2 = a, relative
    lhs = np.exp(3.0 * np.log(q) - 2.0 * np.log1p(q))
    assert np.max(np.abs(lhs / a - 1.0)) < 1e-12
    assert np.isnan(P._dark_companion_q_guess(np.array([0.0]))[0])
