"""joint_orbit_fit (#347): joint NSS-astrometry + RV fit, convergence, bounds, covariance.

Known-truth recovery uses Gaia BH1, BH2 and BH3.

* Gaia BH1 / BH2: the **real** Gaia DR3 NSS solution vector + covariance (public
  catalogue values, ``tests/fixtures/gaia_bh_nss_solutions.json``) combined with
  **synthetic** RV epochs generated from the published joint orbit — labelled
  FIXTURE, not data. (The real APF epochs are not in the repo; a separate test
  uses the staged ``data/dr3/rv_summaries`` BH1 summary when present.)
* Gaia BH3: no DR3 orbital solution exists, so the NSS vector itself is
  **synthetic**: built from the published combined-solution Campbell elements,
  with a covariance obtained by Monte Carlo propagation of the published
  uncertainties (correlated, not diagonal). Labelled FIXTURE.

Published values:
* BH1 — El-Badry et al. 2023a (MNRAS 518, 1057), Table 1 joint fit.
  K = 66.7 km/s is the published RV semi-amplitude cited in #347.
* BH2 — El-Badry et al. 2023b (MNRAS 521, 4323), Table 2 joint Gaia+RV fit.
* BH3 — Gaia Collaboration, Panuzzo et al. 2024 (A&A 686, L2), Tables 2–3
  combined solution; K is derived here from a1, P, e, i (not published as such).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import constants
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.rv_consistency import (
    JOINT_FAIL_BOUND_HIT,
    JOINT_OUTPUT_NAMES,
    JOINT_PROVENANCE,
    JOINT_SKIP_MISSING_NSS_COVARIANCE,
    campbell_from_thiele_innes,
    collect_rv_epochs,
    count_rejected_epochs_bad_mjd,
    fit_joint_orbit,
    k_from_primary_orbit_kms,
    primary_orbit_au,
    run_gate_on_candidates,
    run_joint_on_candidates,
    rv_curve_kms,
    solve_m2_with_inclination_msun,
    thiele_innes_from_campbell,
)
from darkhunter_pop.schemas import CandidateRecord, OrbitTier, ParameterSet

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "gaia_bh_nss_solutions.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
BH1_ID = 4373465352415301632

# Published joint solutions (see module docstring for citations).
PUBLISHED: dict[str, dict[str, float]] = {
    "Gaia-BH1": {
        "P": 185.59, "e": 0.451, "i_deg": 126.6, "omega_deg": 12.8, "Omega_deg": 97.8,
        "Tp_mjd": 57388.5 - 1.1, "gamma": 46.6, "K": 66.7,
        "M1": 0.93, "M1_err": 0.05, "M2": 9.62, "M2_err": 0.18,
    },
    "Gaia-BH2": {
        "P": 1276.7, "e": 0.5176, "i_deg": 34.87, "omega_deg": 130.9, "Omega_deg": 266.9,
        "Tp_mjd": 57388.5 + 49.3, "gamma": -4.22, "K": 25.23,
        "M1": 1.07, "M1_err": 0.19, "M2": 8.94, "M2_err": 0.34,
    },
    "Gaia-BH3": {
        "P": 4253.1, "P_err": 98.5, "e": 0.7291, "e_err": 0.0048,
        "i_deg": 110.580, "i_err": 0.095, "omega_deg": 77.34, "omega_err": 0.76,
        "Omega_deg": 136.236, "Omega_err": 0.128,
        "Tp_mjd": 2458177.39 - 2400000.5, "Tp_err": 0.88,
        "a0_mas": 27.39, "a0_err": 0.49, "parallax": 1.6933, "parallax_err": 0.0164,
        "gamma": -357.31, "M1": 0.76, "M1_err": 0.05, "M2": 32.70, "M2_err": 0.82,
    },
}


def _cfg() -> PipelineConfig:
    return load_config()


def _m1(value: float, err: float) -> ParameterSet:
    return ParameterSet(
        names=["M1"], values=[value], covariance=[[err**2]], provenance="published", units=["Msun"]
    )


def _m2_upstream() -> ParameterSet:
    return ParameterSet(
        names=["M2"], values=[5.0], covariance=[[1.0]], provenance="upstream_stub", units=["Msun"]
    )


def _synthetic_rv_epochs(
    pub: dict[str, float], *, k_kms: float, n: int, noise: float, seed: int
) -> list[dict[str, Any]]:
    """FIXTURE: RVs drawn from the published orbit (not real measurements)."""
    rng = np.random.default_rng(seed)
    t = pub["Tp_mjd"] + np.sort(rng.uniform(0.0, 1.6 * pub["P"], size=n))
    y = rv_curve_kms(
        t,
        period_day=pub["P"],
        eccentricity=pub["e"],
        t_periastron_mjd=pub["Tp_mjd"],
        k_kms=k_kms,
        omega_rad=math.radians(pub["omega_deg"]),
        gamma_kms=pub["gamma"],
    ) + rng.normal(0.0, noise, size=n)
    return [
        {"mjd": float(t[i]), "rv_kms": float(y[i]), "rv_err_kms": noise, "telescope": "SYNTH"}
        for i in range(n)
    ]


def _gate_passed_extras() -> dict[str, Any]:
    return {"rv_astrometry_gate": {"passed": True, "skipped": False, "instruments": []}}


def _candidate(
    source_id: int,
    nss_solution: ParameterSet,
    epochs: list[dict[str, Any]],
    m1: ParameterSet,
) -> CandidateRecord:
    return CandidateRecord(
        source_id=source_id,
        nss_solution_type="Orbital",
        nss_solution=nss_solution,
        rv_summary={"pipeline_epochs": epochs, "external_rvs": []},
        m1=m1,
        m2=_m2_upstream(),
        orbit_tier=OrbitTier.ASTROMETRY_ONLY,
        extras=_gate_passed_extras(),
    )


def _real_nss(name: str) -> tuple[int, ParameterSet]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))["solutions"][name]
    return int(data["source_id"]), ParameterSet.model_validate(data["nss_solution"])


def _bh3_synthetic_nss(seed: int = 3) -> ParameterSet:
    """FIXTURE: BH3 NSS-like vector from published Campbell elements (MC covariance)."""
    pub = PUBLISHED["Gaia-BH3"]
    rng = np.random.default_rng(seed)
    n = 20000
    a0 = rng.normal(pub["a0_mas"], pub["a0_err"], n)
    om = np.radians(rng.normal(pub["omega_deg"], pub["omega_err"], n))
    node = np.radians(rng.normal(pub["Omega_deg"], pub["Omega_err"], n))
    inc = np.radians(rng.normal(pub["i_deg"], pub["i_err"], n))
    abfg = np.array([thiele_innes_from_campbell(*x) for x in zip(a0, om, node, inc)])
    draws = np.column_stack(
        [
            rng.normal(pub["parallax"], pub["parallax_err"], n),
            abfg,
            rng.normal(pub["e"], pub["e_err"], n),
            rng.normal(pub["P"], pub["P_err"], n),
            rng.normal(pub["Tp_mjd"] - constants.GAIA_J2016_MJD, pub["Tp_err"], n),
        ]
    )
    center = [
        pub["parallax"],
        *thiele_innes_from_campbell(
            pub["a0_mas"],
            math.radians(pub["omega_deg"]),
            math.radians(pub["Omega_deg"]),
            math.radians(pub["i_deg"]),
        ),
        pub["e"],
        pub["P"],
        pub["Tp_mjd"] - constants.GAIA_J2016_MJD,
    ]
    cov = np.cov(draws, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    return ParameterSet(
        names=[
            "parallax", "a_thiele_innes", "b_thiele_innes", "f_thiele_innes",
            "g_thiele_innes", "eccentricity", "period", "t_periastron",
        ],
        values=[float(v) for v in center],
        covariance=cov.tolist(),
        provenance="FIXTURE:synthetic_from_GaiaCollab2024_BH3_combined_solution",
    )


def _bh3_derived_k() -> float:
    pub = PUBLISHED["Gaia-BH3"]
    a1 = pub["a0_mas"] / pub["parallax"]
    return k_from_primary_orbit_kms(a1, pub["P"], pub["e"], math.radians(pub["i_deg"]))


def _orbit(cand: CandidateRecord) -> tuple[dict[str, float], dict[str, float]]:
    orbit = cand.extras["joint_orbit"]
    vals = dict(zip(orbit["names"], orbit["values"]))
    sig = {n: math.sqrt(orbit["covariance"][i][i]) for i, n in enumerate(orbit["names"])}
    return vals, sig


def _assert_recovers(name: str, cand: CandidateRecord, k_pub: float) -> None:
    pub = PUBLISHED[name]
    assert cand.orbit_tier is OrbitTier.JOINT_ASTROMETRY_RV, cand.extras.get(
        "joint_orbit_fit_status"
    )
    status = cand.extras["joint_orbit_fit_status"]
    assert status["converged"] is True
    assert status["bound_hits"] == []
    vals, sig = _orbit(cand)
    assert cand.m2 is not None and cand.m2.provenance == JOINT_PROVENANCE
    m2_tol = 3.0 * math.hypot(sig["M2"], pub["M2_err"])
    assert abs(vals["M2"] - pub["M2"]) <= m2_tol, (name, vals["M2"], sig["M2"])
    assert abs(vals["K"] - k_pub) <= 3.0 * sig["K"] + 0.02 * k_pub, (name, vals["K"], sig["K"])
    assert math.degrees(vals["inclination"]) == pytest.approx(pub["i_deg"], abs=5.0)


@pytest.mark.parametrize("name", ["Gaia-BH1", "Gaia-BH2"])
def test_known_truth_real_nss_synthetic_rvs(name: str) -> None:
    pub = PUBLISHED[name]
    sid, nss = _real_nss(name)
    epochs = _synthetic_rv_epochs(pub, k_kms=pub["K"], n=20, noise=0.1, seed=sid % 1000)
    cand = _candidate(sid, nss, epochs, _m1(pub["M1"], pub["M1_err"]))
    out = fit_joint_orbit(cand, _cfg().rv_consistency)
    _assert_recovers(name, out, pub["K"])


def test_known_truth_bh3_synthetic_fixture() -> None:
    pub = PUBLISHED["Gaia-BH3"]
    k_pub = _bh3_derived_k()
    epochs = _synthetic_rv_epochs(pub, k_kms=k_pub, n=20, noise=0.1, seed=33)
    cand = _candidate(4318465066420528000, _bh3_synthetic_nss(), epochs, _m1(pub["M1"], pub["M1_err"]))
    out = fit_joint_orbit(cand, _cfg().rv_consistency)
    _assert_recovers("Gaia-BH3", out, k_pub)


def test_bh1_real_staged_summary_regression() -> None:
    """Real APF epochs (staged ``data/``) + real NSS: BH1 must not reproduce 98 Msun."""
    path = REPO_ROOT / "data" / "dr3" / "rv_summaries" / f"Gaia_DR3_{BH1_ID}_summary.json"
    if not path.is_file():
        pytest.skip("staged BH1 RV summary not present (worktree without data/)")
    summary = json.loads(path.read_text(encoding="utf-8"))
    sid, nss = _real_nss("Gaia-BH1")
    pub = PUBLISHED["Gaia-BH1"]
    cand = CandidateRecord(
        source_id=sid,
        nss_solution_type="Orbital",
        nss_solution=nss,
        rv_summary=summary,
        m1=_m1(pub["M1"], pub["M1_err"]),
        m2=_m2_upstream(),
        orbit_tier=OrbitTier.ASTROMETRY_ONLY,
        extras=_gate_passed_extras(),
    )
    out = fit_joint_orbit(cand, _cfg().rv_consistency)
    vals, sig = _orbit(out)
    assert abs(vals["M2"] - pub["M2"]) <= 3.0 * math.hypot(sig["M2"], pub["M2_err"])
    assert abs(vals["K"] - pub["K"]) <= 3.0 * sig["K"]


# ---------------------------------------------------------------------------
# Defect regressions (#347)
# ---------------------------------------------------------------------------


def test_m2_solver_returns_none_outside_bracket() -> None:
    # f so large no M2 <= 500 satisfies it: must be None, never the 500 edge.
    assert solve_m2_with_inclination_msun(1.0e4, 1.0, 90.0, m2_min_msun=1e-3, m2_max_msun=500.0) is None
    # f so small the root is below the lower edge: None, never ~0.
    assert solve_m2_with_inclination_msun(1.0e-14, 1.0, 90.0, m2_min_msun=1e-3, m2_max_msun=500.0) is None
    m2 = solve_m2_with_inclination_msun(0.5, 1.0, 60.0, m2_min_msun=1e-3, m2_max_msun=500.0)
    assert m2 is not None
    s3 = math.sin(math.radians(60.0)) ** 3
    assert m2**3 * s3 / (1.0 + m2) ** 2 == pytest.approx(0.5, rel=1e-9)


def test_thiele_innes_campbell_roundtrip() -> None:
    a0, om, node, inc = 2.67, math.radians(12.8), math.radians(97.8), math.radians(126.6)
    abfg = thiele_innes_from_campbell(a0, om, node, inc)
    back = campbell_from_thiele_innes(*abfg)
    assert back[0] == pytest.approx(a0, rel=1e-9)
    assert back[3] == pytest.approx(inc, abs=1e-9)
    # (ω, Ω) is recovered up to the (ω+π, Ω+π) Thiele–Innes degeneracy.
    d_om = (back[1] - om) % math.pi
    d_node = (back[2] - node) % math.pi
    assert min(d_om, math.pi - d_om) < 1e-9
    assert min(d_node, math.pi - d_node) < 1e-9


def test_kepler_k_matches_spectroscopic_mass_function() -> None:
    m1, m2, p, e, inc = 0.93, 9.62, 185.59, 0.451, math.radians(126.6)
    k = k_from_primary_orbit_kms(primary_orbit_au(m1, m2, p), p, e, inc)
    f = constants.SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS * k**3 * p * (1 - e * e) ** 1.5
    assert f == pytest.approx((m2 * math.sin(inc)) ** 3 / (m1 + m2) ** 2, rel=1e-6)
    assert k == pytest.approx(66.7, abs=1.5)  # published BH1 K


def test_placeholder_mjd_epochs_rejected() -> None:
    summary = {
        "pipeline_epochs": [
            {"mjd": 60848.3, "rv_kms": 28.8, "rv_err_kms": 0.07, "telescope": "APF"},
            {"mjd": 0.0, "rv_kms": 59.6, "rv_err_kms": 0.12, "telescope": "APF"},
        ],
        "external_rvs": [{"mjd": 0.0, "rv_kms": 1.0, "rv_err_kms": 1.0, "telescope": "RAVE_DR6"}],
    }
    min_mjd = _cfg().rv_consistency.rv_epoch_min_mjd
    assert len(collect_rv_epochs(summary)) == 3
    kept = collect_rv_epochs(summary, min_mjd=min_mjd)
    assert [e.mjd for e in kept] == [60848.3]
    assert count_rejected_epochs_bad_mjd(summary, min_mjd=min_mjd) == 2


def test_bound_hit_flagged_and_counted_not_returned() -> None:
    pub = PUBLISHED["Gaia-BH1"]
    sid, nss = _real_nss("Gaia-BH1")
    epochs = _synthetic_rv_epochs(pub, k_kms=pub["K"], n=20, noise=0.1, seed=1)
    cand = _candidate(sid, nss, epochs, _m1(pub["M1"], pub["M1_err"]))
    cfg = _cfg().model_copy(deep=True)
    cfg.rv_consistency.joint_m2_bounds_msun = (1.0e-3, 5.0)  # true M2 ~9.6 is outside
    joint, diag = run_joint_on_candidates([cand], cfg)
    out = joint[0]
    assert out.extras["joint_orbit_fit_skip_reason"] == JOINT_FAIL_BOUND_HIT
    assert "M2" in out.extras["joint_orbit_fit_status"]["bound_hits"]
    assert out.orbit_tier is OrbitTier.ASTROMETRY_ONLY
    assert out.m2 is not None and out.m2.provenance == "upstream_stub"  # untouched
    assert "joint_orbit" not in out.extras
    assert diag.n_bound_hit == 1 and diag.n_fit == 0
    assert diag.bound_hit_parameters == {"M2": 1}


def test_covariance_is_fit_covariance_not_prior_widths() -> None:
    pub = PUBLISHED["Gaia-BH2"]
    sid, nss = _real_nss("Gaia-BH2")
    cfg = _cfg()
    sigmas = []
    for noise in (0.05, 2.0):
        epochs = _synthetic_rv_epochs(pub, k_kms=pub["K"], n=20, noise=noise, seed=7)
        out = fit_joint_orbit(_candidate(sid, nss, epochs, _m1(pub["M1"], pub["M1_err"])), cfg.rv_consistency)
        vals, sig = _orbit(out)
        cov = np.asarray(out.extras["joint_orbit"]["covariance"])
        assert np.all(np.linalg.eigvalsh(cov[:4, :4]) > 0)
        # Correlated, full matrix: K and M2 are strongly correlated.
        i_k, i_m2 = JOINT_OUTPUT_NAMES.index("K"), JOINT_OUTPUT_NAMES.index("M2")
        assert abs(cov[i_k, i_m2]) > 0.0
        assert out.m2 is not None
        assert out.m2.marginal("M2").sigma == pytest.approx(sig["M2"])
        sigmas.append(sig["K"])
    # Uncertainty responds to the data: noisier RVs → larger σ_K.
    assert sigmas[1] > 5.0 * sigmas[0]


def test_missing_nss_covariance_is_skipped_never_diagonal() -> None:
    pub = PUBLISHED["Gaia-BH1"]
    sid, nss = _real_nss("Gaia-BH1")
    epochs = _synthetic_rv_epochs(pub, k_kms=pub["K"], n=20, noise=0.1, seed=1)
    cand = _candidate(sid, nss, epochs, _m1(pub["M1"], pub["M1_err"]))
    cand = cand.model_copy(update={"nss_solution": None})
    out = fit_joint_orbit(cand, _cfg().rv_consistency)
    assert out.extras["joint_orbit_fit_skip_reason"] == JOINT_SKIP_MISSING_NSS_COVARIANCE
    assert out.orbit_tier is OrbitTier.ASTROMETRY_ONLY


def test_gate_reports_rejected_placeholder_epochs() -> None:
    pub = PUBLISHED["Gaia-BH1"]
    sid, nss = _real_nss("Gaia-BH1")
    epochs = _synthetic_rv_epochs(pub, k_kms=pub["K"], n=8, noise=0.1, seed=2)
    epochs.append({"mjd": 0.0, "rv_kms": 59.6, "rv_err_kms": 0.12, "telescope": "SYNTH"})
    cand = _candidate(sid, nss, epochs, _m1(pub["M1"], pub["M1_err"])).model_copy(
        update={"extras": {}, "nss_orbital": {"period_day": pub["P"], "eccentricity": pub["e"],
                                              "t_periastron_day": pub["Tp_mjd"] - 57388.5,
                                              "semi_amp_primary_kms": pub["K"],
                                              "arg_periastron_deg": pub["omega_deg"],
                                              "inclination_deg": pub["i_deg"]}}
    )
    cfg = _cfg().model_copy(deep=True)
    cfg.dr3.rv_summary_root = None
    _out, diag = run_gate_on_candidates([cand], cfg)
    assert diag.n_epochs_rejected_bad_mjd == 1
    assert diag.n_candidates_with_rejected_epochs == 1
