"""Apparent companion mass M2_app for real NSS rows with the mock's estimator (R5-T2, #620; RUNG5_SPEC §3).

M2_app is the weighted posterior median of M2 under the dark-companion inversion (flux ratio
``mass_recovery.assumed_flux_ratio``, 0), the primary's isochrone lognormal M1 posterior, the ``isotropic_log_m2``
prior and multivariate Student-t likelihood draws with ν set by a0/σ_a0 (#564). Mock draws get it from
:func:`compact_objects.recover_masses`; real rows from :func:`m2_app_for_candidate`. Both call
:func:`compact_objects.m2_posterior_draws_from_solution` and :func:`compact_objects.summarize_m2_posterior` with
the same :class:`compact_objects.MassRecoveryConfig`, so the axis is defined identically on both sides.

For real rows: the solution is the full NSS ``ParameterSet`` from ``corr_vec``/``bit_index`` (never a diagonal);
a0/σ_a0 is the DR3 ``significance``; M1 is the bulk-mass stage's MIST isochrone posterior (``log_m1_mean``,
``log_m1_sigma`` in the record's extras). Rows outside the C1 domain or lacking any input get a reason, never a value.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop.schemas import CandidateRecord

#: Solution types of the C1 orbit domain (RUNG5_SPEC §2.1; El-Badry et al. 2024 §4 set).
C1_SOLUTION_TYPES: tuple[str, ...] = ("Orbital", "AstroSpectroSB1")
ESTIMATOR = "rung5_m2app_v1"


def m2_app_for_solution(
    solution: Any, *, log_m1_mean: float, log_m1_sigma: float, significance: float, cfg: cob.MassRecoveryConfig,
    seed: int,
) -> dict[str, Any] | None:
    """M2 posterior summary of one orbital solution (mock or real) — the R5-T2 estimator."""
    seed_ = int(cfg.random_seed) ^ int(seed)
    m1 = cob.lognormal_m1_draws(log_m1_mean, log_m1_sigma, cfg.n_mc_draws, np.random.default_rng([seed_, 1]))
    if not np.all(np.isfinite(m1)):
        return None
    nu = cob.student_t_nu_for(float(significance), cfg)
    m_f, m2, w, method = cob.m2_posterior_draws_from_solution(
        solution, m1, n_draws=cfg.n_mc_draws, random_seed=seed_, flux_ratio=cfg.assumed_flux_ratio,
        prior=cfg.m2_posterior_prior, student_t_nu=nu)
    summ = cob.summarize_m2_posterior(m_f, m2, w)
    if summ is None:
        return None
    summ["student_t_nu"] = float("nan") if nu is None else float(nu)
    summ["factorization"] = str(getattr(method, "value", method))
    return summ


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def candidate_inputs(cand: CandidateRecord) -> Mapping[str, float]:
    """Significance (a0/σ_a0) and the isochrone log M1 moments of a record; NaN where absent."""
    ex = cand.extras or {}
    orb = cand.nss_orbital or {}
    sig = _num(orb.get("significance", ex.get("significance")))
    return {"significance": sig, "log_m1_mean": _num(ex.get("log_m1_mean")), "log_m1_sigma": _num(ex.get("log_m1_sigma"))}


def m2_app_for_candidate(
    cand: CandidateRecord, cfg: cob.MassRecoveryConfig, *, c1_period_max_days: float,
) -> tuple[dict[str, Any] | None, str]:
    """(summary, reason) for a real NSS row. ``reason`` is ``ok`` or why no value was computed."""
    if cand.nss_solution_type not in C1_SOLUTION_TYPES:
        return None, "not_c1_solution_type"
    sol = cand.nss_solution
    if sol is None:
        return None, "no_nss_covariance"
    names = list(sol.names)
    if "period" not in names:
        return None, "no_period"
    period = float(sol.values_array()[names.index("period")])
    if not (math.isfinite(period) and period <= c1_period_max_days):
        return None, "period_above_c1"
    x = candidate_inputs(cand)
    if not (math.isfinite(x["log_m1_mean"]) and math.isfinite(x["log_m1_sigma"])):
        return None, "no_isochrone_m1"
    if cfg.likelihood_tails == "student_t" and not (math.isfinite(x["significance"]) and x["significance"] > 0):
        return None, "no_significance"
    summ = m2_app_for_solution(sol, log_m1_mean=x["log_m1_mean"], log_m1_sigma=x["log_m1_sigma"],
                               significance=x["significance"], cfg=cfg, seed=int(cand.source_id))
    if summ is None:
        return None, "no_usable_draws"
    return summ, "ok"
