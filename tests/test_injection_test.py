"""Step 1a injection test helpers (#390).

Unit tests use synthetic cascade vectors and a fake gaiamock; the ``gaiamock``
tests run the real overlay (Campbell round trip, seeded replay of one injection).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import injection_test as it
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.forward_model import (
    MOCK_RNG_STREAM_REALIZATION,
    MOCK_RNG_STREAM_SKY,
    passes_orbital_solution_cuts,
)
from darkhunter_pop.gaiamock_vendor import is_overlay_ready


@pytest.fixture(scope="module")
def cuts() -> Any:
    return load_config().active_dr().selection_function_astrometric.orbital_solution_cuts


# ---------------------------------------------------------------- seeding


@pytest.mark.unit
def test_injection_seeds_deterministic_and_distinct() -> None:
    a = it.injection_rng_seeds(42, 181947527763469824, 0)
    assert a == it.injection_rng_seeds(42, 181947527763469824, 0)
    others = {
        it.injection_rng_seeds(42, 181947527763469824, 1),
        it.injection_rng_seeds(43, 181947527763469824, 0),
        it.injection_rng_seeds(42, 181947527763469825, 0),
    }
    assert a not in others and len(others) == 3
    for s in (a, *others):
        assert 0 <= s.numpy_seed < 2**32 and 0 <= s.c_rand_seed < 2**32


@pytest.mark.unit
def test_injection_stream_distinct_from_mock_streams() -> None:
    assert it.INJECTION_RNG_STREAM not in (MOCK_RNG_STREAM_SKY, MOCK_RNG_STREAM_REALIZATION)


@pytest.mark.unit
def test_injection_seeds_reject_negative() -> None:
    with pytest.raises(ValueError):
        it.injection_rng_seeds(-1, 1, 0)
    with pytest.raises(ValueError):
        it.injection_rng_seeds(1, 1, -1)


# ---------------------------------------------------------------- sampling


@pytest.mark.unit
def test_stratified_sample_nested_and_weighted() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    strata = {
        "g": rng.normal(14, 2, n),
        "logp": rng.uniform(1, 4, n),
        "logs": rng.uniform(0.7, 3, n),
    }
    strata["g"][:10] = np.nan
    small, w_small, edges = it.stratified_sample_indices(strata, n_bins=3, n_total=54, seed=7)
    big, w_big, _ = it.stratified_sample_indices(strata, n_bins=3, n_total=270, seed=7)
    assert set(small.tolist()) <= set(big.tolist())
    assert not set(range(10)) & set(big.tolist())
    # Weights reconstruct the finite population size.
    assert math.isclose(float(w_big.sum()), n - 10, rel_tol=1e-9)
    assert math.isclose(float(w_small.sum()), n - 10, rel_tol=1e-9)
    assert set(edges) == {"g", "logp", "logs"}
    assert len(np.unique(big)) == len(big)


@pytest.mark.unit
def test_stratified_sample_small_cells_take_all() -> None:
    strata = {"x": np.arange(6, dtype=float)}
    idx, w, _ = it.stratified_sample_indices(strata, n_bins=3, n_total=300, seed=1)
    assert sorted(idx.tolist()) == list(range(6))
    assert np.allclose(w, 1.0)


# ---------------------------------------------------------------- cascade parsing


def _orbital_vector(**kw: float) -> list[float]:
    v = dict(
        plx=2.8, sig_plx=0.04, A=-1.8, sA=0.05, B=0.6, sB=0.05, F=0.77, sF=0.05,
        G=1.65, sG=0.05, P=720.0, sP=12.0, phi=1.0, sphi=0.1, e=0.4, se=0.05,
        inc=150.0, a0=2.0, sa0=0.08, nvis=18.0, nobs=530.0, F2=5.0, ruwe=6.9,
    )
    v.update(kw)
    return list(v.values())


@pytest.mark.unit
def test_parse_orbital_accepted_matches_forward_model(cuts: Any) -> None:
    rec = it.parse_cascade_result(_orbital_vector(), n_visibility_periods=1, n_obs=1, cuts=cuts)
    assert rec["outcome"] == it.OUTCOME_ORBITAL
    assert rec["accepted"] is True
    assert all(rec[c] for c in it.CUT_FLAG_NAMES)
    assert rec["n_visibility_periods"] == 18.0  # orbital branch reports gaiamock's own
    assert math.isclose(rec["significance"], 2.0 / 0.08)
    assert math.isclose(rec["cos_i"], math.cos(math.radians(150.0)))


@pytest.mark.unit
@pytest.mark.parametrize(
    "override,failing",
    [
        ({"F2": 30.0}, "cut_f2"),
        ({"sa0": 1.0}, "cut_a0_over_err"),
        ({"se": 0.9}, "cut_sigma_e"),
        ({"sig_plx": 2.0}, "cut_parallax_over_error"),
        ({"P": 20.0, "sa0": 0.1, "a0": 2.0, "sig_plx": 0.001, "se": 0.001}, "cut_a0_over_err_sqrt_p"),
    ],
)
def test_parse_orbital_cut_flags(cuts: Any, override: dict[str, float], failing: str) -> None:
    vec = _orbital_vector(**override)
    rec = it.parse_cascade_result(vec, n_visibility_periods=1, n_obs=1, cuts=cuts)
    assert rec[failing] is False
    expect = passes_orbital_solution_cuts(
        a0_over_err=vec[17] / vec[18],
        parallax_over_error=vec[0] / vec[1],
        period_days=vec[10],
        sigma_ecc=vec[15],
        goodness_of_fit_f2=vec[21],
        cuts=cuts,
    )
    assert rec["accepted"] is expect is False
    assert rec["accepted"] == all(rec[c] for c in it.CUT_FLAG_NAMES)


@pytest.mark.unit
def test_parse_non_orbital_branches(cuts: Any) -> None:
    rec0 = it.parse_cascade_result([0.0] * 23, n_visibility_periods=8, n_obs=90, cuts=cuts)
    assert rec0["outcome"] == it.OUTCOME_INSUFFICIENT_VISIBILITY and not rec0["accepted"]
    assert rec0["n_visibility_periods"] == 8.0

    v5 = [-1.0] * 23
    v5[1], v5[2], v5[3] = 1.1, 3.0, 0.05
    rec5 = it.parse_cascade_result(v5, n_visibility_periods=15, n_obs=300, cuts=cuts)
    assert rec5["outcome"] == it.OUTCOME_FIVE_PARAMETER
    assert rec5["ruwe"] == 1.1 and rec5["parallax"] == 3.0

    v7 = [-7.0] * 23
    v7[1], v7[8], v7[9] = 40.0, 3.3, 4.0
    rec7 = it.parse_cascade_result(v7, n_visibility_periods=15, n_obs=300, cuts=cuts)
    assert rec7["outcome"] == it.OUTCOME_SEVEN_PARAMETER
    assert rec7["ruwe"] == 3.3 and rec7["goodness_of_fit"] == 4.0
    assert rec7["acceleration_significance"] == 40.0

    v9 = [-9.0] * 23
    v9[1], v9[12], v9[13] = 30.0, 2.2, 1.0
    rec9 = it.parse_cascade_result(v9, n_visibility_periods=15, n_obs=300, cuts=cuts)
    assert rec9["outcome"] == it.OUTCOME_NINE_PARAMETER
    assert rec9["ruwe"] == 2.2 and rec9["goodness_of_fit"] == 1.0
    for rec in (rec5, rec7, rec9):
        assert not rec["accepted"] and math.isnan(rec["period"])


@pytest.mark.unit
def test_outside_fit_bounds() -> None:
    assert it.outside_fit_bounds(5.0, 0.1)
    assert it.outside_fit_bounds(2.0e4, 0.1)
    assert it.outside_fit_bounds(100.0, 0.995)
    assert not it.outside_fit_bounds(100.0, 0.3)


# ---------------------------------------------------------------- statistics


@pytest.mark.unit
def test_summarize_pulls_gaussian_and_outliers() -> None:
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1, 20000), [50.0] * 200, [np.nan] * 5])
    s = it.summarize_pulls(x, clip=5.0)
    assert s.n == 20200
    assert abs(s.median) < 0.03 and abs(s.sigma_mad - 1.0) < 0.05
    assert abs(s.std - 1.0) < 0.03
    assert math.isclose(s.frac_outlier, 200 / 20200, rel_tol=0.05)
    empty = it.summarize_pulls([np.nan])
    assert empty.n == 0 and math.isnan(empty.median)


@pytest.mark.unit
def test_binomial_interval() -> None:
    p, lo, hi = it.binomial_fraction_interval(50, 100)
    assert p == 0.5 and lo < 0.5 < hi and math.isclose(hi - 0.5, 0.5 - lo)
    p, lo, hi = it.binomial_fraction_interval(0, 10)
    assert p == 0.0 and lo == 0.0 and hi > 0.0
    assert all(math.isnan(v) for v in it.binomial_fraction_interval(0, 0))


# ---------------------------------------------------------------- truth construction


def _campbell_fake() -> SimpleNamespace:
    """Fake gaiamock whose Campbell conversion is a0 = |ABFG| and cos i = A/|ABFG|."""

    def get_Campbell_elements(a: float, b: float, f: float, g: float) -> tuple[float, ...]:
        r = math.sqrt(a * a + b * b + f * f + g * g)
        return r, 0.1, 0.2, math.acos(a / r)

    return SimpleNamespace(get_Campbell_elements=get_Campbell_elements)


@pytest.mark.unit
def test_propagate_sigmas_linear() -> None:
    fake = _campbell_fake()
    x = np.array([3.0, 0.0, 4.0, 0.0])
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    s_a0, s_ci = it.propagate_thiele_innes_sigmas(fake, x, cov)
    # d a0 / dx = x / r = (0.6, 0, 0.8, 0) -> var = 0.36*0.01 + 0.64*0.09
    assert math.isclose(s_a0, math.sqrt(0.36 * 0.01 + 0.64 * 0.09), rel_tol=1e-5)
    # cos i = A / r: d/dA = 1/r - A^2/r^3 = 0.128, d/dF = -A F / r^3 = -0.096
    assert math.isclose(s_ci, math.sqrt(0.128**2 * 0.01 + 0.096**2 * 0.09), rel_tol=1e-5)
    nan_a0, nan_ci = it.propagate_thiele_innes_sigmas(fake, x, np.full((4, 4), np.nan))
    assert math.isnan(nan_a0) and math.isnan(nan_ci)


@pytest.mark.unit
def test_published_truth_without_covariance_is_nan_not_diagonal() -> None:
    row = dict(
        source_id=1, nss_solution_type="Orbital", ra=10.0, dec=-5.0, parallax=2.0,
        parallax_error=0.05, pmra=1.0, pmdec=2.0, period=500.0, period_error=5.0,
        t_periastron=-10.0, eccentricity=0.2, eccentricity_error=0.02,
        a_thiele_innes=3.0, b_thiele_innes=0.0, f_thiele_innes=4.0, g_thiele_innes=0.0,
        a_thiele_innes_error=0.1, b_thiele_innes_error=0.1,
        f_thiele_innes_error=0.1, g_thiele_innes_error=0.1,
        significance=25.0, goodness_of_fit=3.0, ruwe=4.0, g_mag=13.0,
    )
    t = it.published_truth_from_row(row, _campbell_fake())
    assert math.isclose(t.values["a0_mas"], 5.0)
    assert math.isclose(t.values["sigma_a0_mas"], 5.0 / 25.0)
    assert math.isnan(t.values["sigma_cos_i"]) and math.isnan(t.values["sigma_a0_cov_mas"])
    assert set(it.TRUTH_COLUMNS) <= set(t.values)


# ---------------------------------------------------------------- real gaiamock

_PUBLISHED_ROW = dict(
    source_id=181947527763469824, nss_solution_type="Orbital", ra=78.88435885851148,
    dec=33.734906720683, parallax=2.804344092523763, parallax_error=0.0495,
    pmra=-11.388, pmdec=0.326, period=722.9159, period_error=19.6, t_periastron=-153.18,
    eccentricity=0.3907, eccentricity_error=0.0567, a_thiele_innes=-1.8243585277547643,
    b_thiele_innes=0.5938978326347945, f_thiele_innes=0.7738281210723258,
    g_thiele_innes=1.6547410723209703, significance=30.0, goodness_of_fit=13.97,
    ruwe=6.94, g_mag=11.895,
)


@pytest.mark.gaiamock
@pytest.mark.skipif(not is_overlay_ready(), reason="gaiamock_mod overlay not installed")
def test_campbell_roundtrip_reproduces_published_thiele_innes() -> None:
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    rng = np.random.default_rng(11)
    for _ in range(200):
        abfg = rng.normal(0, 2, 4)
        row = dict(_PUBLISHED_ROW)
        row.update(zip(("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"), abfg))
        t = it.published_truth_from_row(row, gm)
        assert it.thiele_innes_roundtrip_error(t.values) < 1e-9


@pytest.mark.gaiamock
@pytest.mark.skipif(not is_overlay_ready(), reason="gaiamock_mod overlay not installed")
def test_injection_replays_bit_for_bit(cuts: Any) -> None:
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    cf = gm.read_in_C_functions()
    t = it.published_truth_from_row(_PUBLISHED_ROW, gm)
    kw = dict(base_seed=42, cuts=cuts, data_release="dr3", ruwe_min=1.4, skip_acceleration=False)
    a = it.inject_one_realization(gm, cf, t, realization=0, **kw)
    np.random.seed(999)
    np.random.uniform(size=33)
    b = it.inject_one_realization(gm, cf, t, realization=0, **kw)
    assert a == b
    assert a["outcome"] == it.OUTCOME_ORBITAL
    assert abs(a["period"] - 722.9) / a["period_error"] < 5
