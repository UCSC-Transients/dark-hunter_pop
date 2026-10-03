"""Cascade replay helpers for #399 / #398.

Unit tests use synthetic statistics; the ``gaiamock`` tests run the real overlay and
check that the replay reproduces gaiamock's own epochs and cascade decision.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import cascade_replay as cr
from darkhunter_pop import injection_test as it
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.gaiamock_vendor import is_overlay_ready


def _stats(**kw: float) -> cr.LinearCascadeStats:
    base = dict(ruwe=3.0, s9=5.0, f2_9=3.0, plx_snr9=100.0, s7=5.0, f2_7=3.0, plx_snr7=100.0,
                n_obs=600, n_vis=25)
    base.update(kw)
    return cr.LinearCascadeStats(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_f2_inversion_roundtrip() -> None:
    nu = 500
    for chi2_red in (0.8, 1.0, 1.3, 3.0):
        f2 = math.sqrt(9 * nu / 2) * (chi2_red ** (1 / 3) + 2 / (9 * nu) - 1)
        assert cr.chi2_from_f2(f2, nu) == pytest.approx(chi2_red * nu, rel=1e-12)
    assert math.isnan(cr.chi2_from_f2(1.0, 0))


@pytest.mark.unit
def test_inflation_factor_matches_halbwachs_eq2() -> None:
    nu = np.array([300.0, 500.0, 800.0])
    # F2 = 0  ->  chi2/nu = (1 - 2/9nu)^3  ->  c = 1 exactly
    assert np.allclose(cr.inflation_factor_from_f2(np.zeros(3), nu), 1.0)
    chi2_red = 1.44
    f2 = np.sqrt(9 * nu / 2) * (chi2_red ** (1 / 3) + 2 / (9 * nu) - 1)
    c = cr.inflation_factor_from_f2(f2, nu)
    assert np.allclose(c, np.sqrt(chi2_red / (1 - 2 / (9 * nu)) ** 3))
    assert np.isnan(cr.inflation_factor_from_f2(1.0, -3.0))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kw", "branch"),
    [
        (dict(n_vis=11), cr.BRANCH_INSUFFICIENT_VISIBILITY),
        (dict(n_obs=12), cr.BRANCH_INSUFFICIENT_VISIBILITY),
        (dict(ruwe=1.2), cr.BRANCH_FIVE_PARAMETER),
        (dict(s9=20.0), cr.BRANCH_NINE_PARAMETER),
        (dict(s9=20.0, f2_9=30.0, s7=15.0), cr.BRANCH_SEVEN_PARAMETER),
        (dict(s9=20.0, plx_snr9=40.0, s7=15.0), cr.BRANCH_SEVEN_PARAMETER),  # 2.1*20^1.05 = 49.6
        (dict(s9=20.0, plx_snr9=40.0, s7=15.0, plx_snr7=15.0), cr.BRANCH_ORBITAL),
        (dict(s9=11.9, s7=11.9), cr.BRANCH_ORBITAL),
    ],
)
def test_predicted_branch(kw: dict[str, float], branch: int) -> None:
    assert cr.predicted_branch(_stats(**kw), ruwe_min=1.4) == branch


@pytest.mark.unit
def test_acceleration_conditions_keys() -> None:
    c = cr.acceleration_conditions(_stats(s9=20.0, plx_snr9=40.0))
    assert c == {"s9": True, "f2_9": True, "plx9": False, "s7": False, "f2_7": True, "plx7": True}


@pytest.mark.unit
def test_orbit_coverage() -> None:
    t_yr = np.linspace(-1.4, 1.4, 50)
    span, frac, peri = cr.orbit_coverage(t_yr, 2000.0, 0.0)
    assert span == pytest.approx(2.8 * 365.25)
    assert frac == pytest.approx(2.8 * 365.25 / 2000.0)
    assert peri
    _, _, peri2 = cr.orbit_coverage(t_yr, 5000.0, 2000.0)  # passages at 2000, -3000 d
    assert not peri2
    _, _, peri3 = cr.orbit_coverage(t_yr, 5000.0, -4700.0)  # passage at +300 d
    assert peri3


@pytest.mark.unit
def test_observations_scaling() -> None:
    rng = np.random.default_rng(1)
    sig, noi = rng.normal(size=20), rng.normal(size=20)
    ep = cr.ReplayedEpochs(
        t_ast_yr=np.arange(20.0), psi=np.zeros(20), plx_factor=np.zeros(20),
        observed=sig + noi, signal=sig, noise=noi, ast_err=np.full(20, 0.1),
    )
    o, e = ep.observations()
    assert o is ep.observed and e is ep.ast_err
    o2, e2 = ep.observations(noise_scale=2.0, err_scale=3.0)
    assert np.allclose(o2, sig + 2 * noi) and np.allclose(e2, 0.3)
    with pytest.raises(ValueError):
        ep.observations(noise_scale=0.0)


# ------------------------------------------------------------------ gaiamock

_ROW = dict(  # Gaia DR3 181947527763469824, the same public row as test_injection_test
    source_id=181947527763469824, nss_solution_type="Orbital", ra=78.88435885851148,
    dec=33.734906720683, parallax=2.804344092523763, parallax_error=0.0495,
    pmra=-11.388, pmdec=0.326, period=722.9159, period_error=19.6, t_periastron=-153.18,
    eccentricity=0.3907, eccentricity_error=0.0567, a_thiele_innes=-1.8243585277547643,
    b_thiele_innes=0.5938978326347945, f_thiele_innes=0.7738281210723258,
    g_thiele_innes=1.6547410723209703, significance=30.0, goodness_of_fit=13.97,
    ruwe=6.94, g_mag=11.895,
)


@pytest.fixture(scope="module")
def gm_cf() -> tuple[Any, Any]:
    if not is_overlay_ready():
        pytest.skip("gaiamock_mod overlay not installed")
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    return gm, gm.read_in_C_functions()


@pytest.mark.gaiamock
def test_replay_reproduces_gaiamock_epochs_and_decision(gm_cf: tuple[Any, Any]) -> None:
    gm, cf = gm_cf
    cuts = load_config().active_dr().selection_function_astrometric.orbital_solution_cuts
    # vary a0 so both acceleration and orbital outcomes occur
    branches = set()
    for scale in (0.05, 0.2, 1.0):
        row = dict(_ROW)
        for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"):
            row[k] = scale * float(_ROW[k])
        truth = it.published_truth_from_row(row, gm)
        for r in range(3):
            seeds = it.injection_rng_seeds(42, truth.source_id, r)
            ep = cr.replay_injection_epochs(gm, cf, truth.values, seeds, data_release="dr3")
            assert np.allclose(ep.signal + ep.noise, ep.observed, rtol=0, atol=1e-12)
            rec = it.inject_one_realization(
                gm, cf, truth, realization=r, base_seed=42, cuts=cuts,
                data_release="dr3", ruwe_min=1.4, skip_acceleration=False,
            )
            st = cr.linear_cascade_statistics(gm, ep.t_ast_yr, ep.psi, ep.plx_factor, ep.observed, ep.ast_err)
            b = cr.predicted_branch(st, ruwe_min=1.4)
            assert b == rec["outcome"]
            branches.add(b)
            if b in (cr.BRANCH_SEVEN_PARAMETER, cr.BRANCH_NINE_PARAMETER):
                s = st.s9 if b == cr.BRANCH_NINE_PARAMETER else st.s7
                assert s == pytest.approx(rec["acceleration_significance"], rel=1e-12)
            if b == cr.BRANCH_ORBITAL:
                vec = cr.forced_orbit_fit(gm, cf, ep, seeds, ruwe_min=1.4)
                assert vec[10] == pytest.approx(rec["period"], rel=1e-12)
    assert len(branches) >= 2


@pytest.mark.gaiamock
def test_common_rescaling_leaves_statistics_invariant(gm_cf: tuple[Any, Any]) -> None:
    """noise x k with errors x k == the same truth at S/N / k (all cascade stats invariant)."""
    gm, cf = gm_cf
    truth = it.published_truth_from_row(_ROW, gm)
    ep = cr.replay_injection_epochs(gm, cf, truth.values, it.injection_rng_seeds(42, truth.source_id, 0),
                                    data_release="dr3")
    k = 1.37
    a = cr.linear_cascade_statistics(gm, ep.t_ast_yr, ep.psi, ep.plx_factor, ep.observed, ep.ast_err)
    b = cr.linear_cascade_statistics(gm, ep.t_ast_yr, ep.psi, ep.plx_factor, k * ep.observed, k * ep.ast_err)
    for f in ("ruwe", "s9", "f2_9", "plx_snr9", "s7", "f2_7", "plx_snr7"):
        assert getattr(a, f) == pytest.approx(getattr(b, f), rel=1e-9)
