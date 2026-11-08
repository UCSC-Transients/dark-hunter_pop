"""R5-T2 (#620): one M2_app estimator for mock draws and real NSS rows. No data/ access."""

from __future__ import annotations

import math

import numpy as np
import pytest

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop import rung5_m2app as rm
from darkhunter_pop.schemas import CandidateRecord, ParameterSet

RUNG4 = "config/population/rung4_co_v1.yaml"
NSS_NAMES = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "eccentricity", "t_periastron",
             "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")


@pytest.fixture(scope="module")
def mr() -> cob.MassRecoveryConfig:
    return cob.load_rung4_config(RUNG4).mass_recovery.model_copy(update={"n_mc_draws": 2000})


def _orbit(m1: float = 1.0, m2: float = 5.0, p_day: float = 200.0, plx: float = 2.0, inc_deg: float = 60.0,
           rel_err: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    a_rel = ((p_day / 365.25) ** 2 * (m1 + m2)) ** (1.0 / 3.0)
    a0 = plx * a_rel * m2 / (m1 + m2)
    ci = math.cos(math.radians(inc_deg))
    p0 = np.array([0.0, 0.0, plx, 5.0, -3.0, p_day, 0.2, 0.1, a0, 0.0, 0.0, a0 * ci])
    sig = np.abs(p0) * rel_err + 1e-3
    return p0, np.diag(sig**2)


@pytest.mark.unit
def test_mock_wrapper_equals_core(mr: cob.MassRecoveryConfig) -> None:
    p0, cov = _orbit()
    m1 = np.full(mr.n_mc_draws, 1.0)
    a = cob.m2_posterior_draws(p0, cov, m1, n_draws=mr.n_mc_draws, random_seed=5, flux_ratio=0.0, prior="isotropic_log_m2", student_t_nu=3.0)
    b = cob.m2_posterior_draws_from_solution(cob.cascade_parameter_set(p0, cov), m1, n_draws=mr.n_mc_draws, random_seed=5,
                                             flux_ratio=0.0, prior="isotropic_log_m2", student_t_nu=3.0)
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)


@pytest.mark.unit
def test_nss_names_give_identical_result(mr: cob.MassRecoveryConfig) -> None:
    p0, cov = _orbit()
    mock = cob.cascade_parameter_set(p0, cov)
    real = ParameterSet(names=list(NSS_NAMES), values=list(p0), covariance=cov.tolist(), provenance="test_nss_row")
    kw = dict(log_m1_mean=0.0, log_m1_sigma=0.03, significance=40.0, cfg=mr, seed=123)
    s1, s2 = rm.m2_app_for_solution(mock, **kw), rm.m2_app_for_solution(real, **kw)
    assert s1 is not None and s2 is not None
    for k in ("m2_med", "m2_q16", "m2_q84", "m_f_med", "mc_ess", "student_t_nu"):
        assert s1[k] == s2[k]


@pytest.mark.unit
def test_parameter_order_does_not_matter(mr: cob.MassRecoveryConfig) -> None:
    p0, cov = _orbit()
    perm = np.random.default_rng(1).permutation(12)
    a = ParameterSet(names=list(NSS_NAMES), values=list(p0), covariance=cov.tolist(), provenance="t")
    b = ParameterSet(names=[NSS_NAMES[i] for i in perm], values=list(p0[perm]), covariance=cov[np.ix_(perm, perm)].tolist(), provenance="t")
    kw = dict(log_m1_mean=0.0, log_m1_sigma=0.03, significance=40.0, cfg=mr, seed=9)
    s1, s2 = rm.m2_app_for_solution(a, **kw), rm.m2_app_for_solution(b, **kw)
    assert s2["m2_med"] == pytest.approx(s1["m2_med"], rel=0.03)


@pytest.mark.unit
def test_recovers_dark_companion_mass(mr: cob.MassRecoveryConfig) -> None:
    p0, cov = _orbit(m1=1.0, m2=5.0)
    sol = ParameterSet(names=list(NSS_NAMES), values=list(p0), covariance=cov.tolist(), provenance="t")
    s = rm.m2_app_for_solution(sol, log_m1_mean=0.0, log_m1_sigma=0.01, significance=100.0, cfg=mr, seed=4)
    assert s is not None
    assert s["m2_med"] == pytest.approx(5.0, rel=0.1)
    assert s["m2_q16"] < s["m2_med"] < s["m2_q84"]
    assert s["student_t_nu"] == pytest.approx(cob.student_t_nu_for(100.0, mr))


@pytest.mark.unit
def test_lognormal_m1_shared_with_mock() -> None:
    rng_a, rng_b = np.random.default_rng([7, 1]), np.random.default_rng([7, 1])
    x = cob.lognormal_m1_draws(-0.1, 0.05, 100, rng_a)
    y = 10.0 ** (-0.1 + 0.05 * rng_b.standard_normal(100))
    np.testing.assert_array_equal(x, y)
    assert np.all(np.isnan(cob.lognormal_m1_draws(float("nan"), 0.1, 5, rng_a)))


def _cand(**kw) -> CandidateRecord:
    p0, cov = _orbit()
    base = dict(source_id=42, nss_solution_type="Orbital",
                nss_solution=ParameterSet(names=list(NSS_NAMES), values=list(p0), covariance=cov.tolist(), provenance="t"),
                nss_orbital={"significance": 40.0}, extras={"log_m1_mean": 0.0, "log_m1_sigma": 0.03})
    base.update(kw)
    return CandidateRecord(**base)


@pytest.mark.unit
@pytest.mark.parametrize("kw, reason", [
    ({}, "ok"),
    ({"nss_solution_type": "SB1"}, "not_c1_solution_type"),
    ({"nss_solution": None}, "no_nss_covariance"),
    ({"extras": {}}, "no_isochrone_m1"),
    ({"nss_orbital": {}}, "no_significance"),
])
def test_candidate_reasons(mr: cob.MassRecoveryConfig, kw: dict, reason: str) -> None:
    summ, why = rm.m2_app_for_candidate(_cand(**kw), mr, c1_period_max_days=830.4)
    assert why == reason
    assert (summ is not None) == (reason == "ok")


@pytest.mark.unit
def test_candidate_period_ceiling(mr: cob.MassRecoveryConfig) -> None:
    summ, why = rm.m2_app_for_candidate(_cand(), mr, c1_period_max_days=100.0)
    assert summ is None and why == "period_above_c1"


@pytest.mark.unit
def test_orbit_names_cover_nss_canonical_set() -> None:
    from darkhunter_pop import mc_mass_function as mcm

    need = {"a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes", "parallax", "period"}
    assert need <= {mcm.canonical_nss_name(n) for n in ps.ORBIT_PARAMETER_NAMES}
    assert need <= {mcm.canonical_nss_name(n) for n in NSS_NAMES}
