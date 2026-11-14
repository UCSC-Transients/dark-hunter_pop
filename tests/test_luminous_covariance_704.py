"""#704 (option B): luminous-generation orbit covariances by deterministic replay, the pre-filter and the adapter marks.

Synthetic only (no data/, no gaiamock): the dust maps are a constant-E(B-V) stub, the cascade a linear fake.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from darkhunter_pop import luminous_covariance as lc
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.andrews2022_atf import require_procedure
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.mock_atf_columns import mock_rows_atf_columns
from darkhunter_pop.mock_selection_rows import CASCADE_COLUMNS, forward_model_selection, mock_draws_to_selection_rows
from darkhunter_pop.sample_selection import CutOutcome, NotApplicable, SampleSelectionRegistry, evaluate_cut

LUMINOUS_FRAGMENT = "config/population/proposal_set_single59.yaml"
CO_FRAGMENT = "config/population/proposal_set_co_v1_smoke.yaml"
_SHORT = {"a_thiele_innes": "A", "b_thiele_innes": "B", "f_thiele_innes": "F", "g_thiele_innes": "G"}


class _ConstantEbv:
    def __init__(self, dust_cfg: Any) -> None:
        self.dust_cfg = dust_cfg

    def ebv_for(self, ids: Any, ra: Any, dec: Any, plx: Any) -> tuple[Any, Any, Any]:
        n = len(ids)
        return np.full(n, 0.02), np.zeros(n, dtype=int), np.zeros(n, dtype=int)


@pytest.fixture(scope="module")
def registry() -> SampleSelectionRegistry:
    return SampleSelectionRegistry(load_config(), repo=repo_root())


@pytest.fixture(scope="module")
def ccfg() -> lc.LuminousCovarianceConfig:
    return lc.load_luminous_covariance_config()


def _a0_mas(m1: float, m2: float, period_days: float, plx_mas: float) -> float:
    """Face-on dark-companion photocentre semi-major axis (f = 0)."""
    a_au = ((m1 + m2) * (period_days / 365.25) ** 2) ** (1 / 3)
    return a_au * m2 / (m1 + m2) * plx_mas


def _draws(m2: list[float], g: list[float], sol: list[str], period: float = 300.0, plx: float = 2.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """(truth, outcome) of n accepted orbits with dark-companion M2 at M1 = 1 (the Andrews pass-1 M1)."""
    n = len(m2)
    names = list(lc_names())
    p0 = np.zeros((n, 12))
    cov = np.zeros((n, 12, 12))
    cascade = np.zeros((n, len(CASCADE_COLUMNS)))
    for k in range(n):
        a0 = _a0_mas(1.0, m2[k], period, plx)
        v = dict.fromkeys(names, 0.0)
        # inclined (cos i = 0.6), Omega = omega = 0: A = a0, G = a0 cos i, B = F = 0 -> the Thiele-Innes a0 is exact and
        # the a0 gradient is regular (face-on is the u = v singular point)
        v.update(parallax=plx, period=period, eccentricity=0.1, phi_p=0.3, a_thiele_innes=a0, g_thiele_innes=0.6 * a0)
        p0[k] = [v[nm] for nm in names]
        sig = np.array([0.01, 0.01, 0.01 * plx, 0.01, 0.01, 0.5, 0.01, 0.01, 0.01 * a0, 0.01 * a0, 0.01 * a0, 0.01 * a0])
        cov[k] = np.diag(sig**2)
        s = dict(zip(names, sig, strict=True))
        for name, val in {"plx": plx, "sig_parallax": s["parallax"], "period": period, "sig_period": s["period"],
                          "ecc": 0.1, "sig_ecc": s["eccentricity"], "F2": 2.0, "ruwe": 1.6, "N_visibility_periods": 20.0,
                          "N_obs": 300.0, **{sh: v[lo] for lo, sh in _SHORT.items()},
                          **{f"sig_{sh}": s[lo] for lo, sh in _SHORT.items()}}.items():
            cascade[k, CASCADE_COLUMNS.index(name)] = val
    ids = np.arange(n, dtype=np.int64) + 77_000_000
    truth = {
        "draw_index": ids, "source_id": np.full(n, 4_000_000_000_000_000_000, dtype=np.int64),
        "ra_deg": np.full(n, 150.0), "dec_deg": np.full(n, 20.0), "phot_g_mean_mag": np.asarray(g, float),
        "sys_bp_rp": np.full(n, 0.85), "m1_row_msun": np.ones(n), "m1_msun": np.ones(n), "m2_msun": np.asarray(m2, float),
        "is_dark": np.zeros(n, dtype=bool), "co_component": np.zeros(n, dtype=np.int8), "period_days": np.full(n, period),
        "sys_evolved": np.zeros(n, dtype=bool), "parent_row": np.zeros(n, dtype=np.int64),
    }
    outcome = {"cascade": cascade, "accepted_orbital": np.ones(n, dtype=bool), "nss_solution_type": np.array(sol),
               "orbit_p0": p0, "orbit_cov": cov}
    return truth, outcome


def lc_names() -> tuple[str, ...]:
    return tuple(ps.ORBIT_PARAMETER_NAMES)


def _rows(truth: dict[str, Any], outcome: dict[str, Any], include_orbit: bool = False) -> list[dict[str, Any]]:
    n = truth["draw_index"].size
    return mock_draws_to_selection_rows(truth, outcome, pipeline_m1=(np.full(n, 0.9), np.full(n, 0.05)), include_orbit=include_orbit)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_step_equals_compact_object_step(ccfg: lc.LuminousCovarianceConfig) -> None:
    """Same Jacobian step as the CO path (every CO fragment that stores covariances)."""
    steps = set()
    for p in sorted((repo_root() / "config/population").glob("proposal_set_co_*.yaml")):
        co = (yaml.safe_load(p.read_text())["proposal"] or {}).get("compact_object") or {}
        if co.get("store_orbit_covariance"):
            steps.add(float(co["covariance_jacobian_step"]))
    assert steps == {ccfg.covariance_jacobian_step}


@pytest.mark.unit
def test_prefilter_strictly_looser_than_frozen(registry: SampleSelectionRegistry, ccfg: lc.LuminousCovarianceConfig) -> None:
    assert lc.check_prefilter_looser(registry, ccfg) == []
    b = lc.frozen_bounds(registry, ccfg)
    assert (b.andrews_m2_threshold_msun, b.andrews_pass1_m1_msun, b.andrews_solution_types) == (1.4, 1.0, ("Orbital",))
    assert (b.elbadry2026_m2_tilde_min_msun, b.elbadry2026_period_days_max, b.elbadry2026_g_mag_faint_limit) == (1.05, 1200.0, 15.0)
    assert set(b.elbadry2026_solution_types) == {"Orbital", "AstroSpectroSB1"}
    tight = ccfg.model_copy(update={"prefilter": ccfg.prefilter.model_copy(update={
        "andrews": ccfg.prefilter.andrews.model_copy(update={"m2_point_min_msun": 1.4}),
        "elbadry2026": ccfg.prefilter.elbadry2026.model_copy(update={"period_days_max": 900.0, "solution_types": ("Orbital",)})})})
    bad = lc.check_prefilter_looser(registry, tight)
    assert len(bad) == 3
    with pytest.raises(ValueError, match="not looser"):
        lc.evaluate_prefilter([], tight, registry=registry)


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_andrews_point_m2_recovers_dark_companion_mass(registry: SampleSelectionRegistry) -> None:
    proc = require_procedure(registry.resolved("andrews2022"))
    truth, outcome = _draws([0.5, 1.4, 8.0], [13.0] * 3, ["Orbital"] * 3)
    rows = _rows(truth, outcome)
    m2 = lc.andrews_point_m2(rows, proc)
    np.testing.assert_allclose(m2, [0.5, 1.4, 8.0], rtol=1e-9)
    rows[0]["parallax"] = -1.0
    assert np.isnan(lc.andrews_point_m2(rows, proc)[0])


@pytest.mark.unit
def test_prefilter_branches(registry: SampleSelectionRegistry, ccfg: lc.LuminousCovarianceConfig) -> None:
    #        Andrews-reachable  ASB1 heavy  faint low   faint heavy-Orbital  ASB1 light
    m2 = [3.0, 3.0, 0.4, 3.0, 0.3]
    g = [13.0, 13.0, 16.0, 16.0, 13.0]
    sol = ["Orbital", "AstroSpectroSB1", "Orbital", "Orbital", "AstroSpectroSB1"]
    truth, outcome = _draws(m2, g, sol)
    rows = _rows(truth, outcome)
    pf = lc.evaluate_prefilter(rows, ccfg, registry=registry, extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    assert pf.pass_andrews.tolist() == [True, False, False, True, False]
    assert pf.pass_elbadry2026[[2, 3]].tolist() == [False, False]  # G >= 15.5: the M~2 branch is not even enriched
    assert np.isnan(pf.m2_tilde_msun[[2, 3]]).all()
    assert pf.pass_any.tolist()[:4] == [True, bool(pf.pass_elbadry2026[1]), False, True]
    # the M~2 the filter uses is the evaluator's own
    sel = forward_model_selection(registry, "elbadry2026", extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    res = sel.evaluate(rows)
    for k, sid in enumerate(pf.source_id):
        if sid in res.tilde_masses_by_source and k not in (2, 3):
            assert pf.m2_tilde_msun[k] == pytest.approx(res.tilde_masses_by_source[int(sid)][1], rel=1e-12)
    assert np.isfinite(pf.m2_tilde_msun[[0, 1]]).all() and pf.pass_elbadry2026[1]
    assert not pf.pass_elbadry2026[4]


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def _sidecar(acc: np.ndarray, ok: np.ndarray, **attrs: Any) -> lc.OrbitCovSidecar:
    d = acc[ok]
    rng = np.random.default_rng(3)
    recs = {}
    for x in d:
        a = rng.normal(size=(12, 12))
        recs[int(x)] = {"orbit_p0": rng.normal(size=12).tolist(), "orbit_cov": (a @ a.T).ravel().tolist(),
                        "orbit_cov_sigma_check": 1e-6, "orbit_cov_cpu_seconds": 0.1}
    base = {"artifact": "g.h5", "primary_model": "pm", "fragment_fingerprint": "fp", "n_draws": 6, "prefilter_version": "lcf1"}
    base.update(attrs)
    return lc.sidecar_from_records(acc, ok, recs, base)


@pytest.mark.unit
def test_sidecar_roundtrip_merge_and_refusals(tmp_path: Path) -> None:
    di = np.arange(6, dtype=np.int64) + 500
    truth = {"draw_index": di}
    acc_mask = np.array([True, True, False, True, False, True])
    outcome: dict[str, Any] = {"accepted_orbital": acc_mask}
    acc = di[acc_mask]
    ok = np.array([True, False, True, False])
    sc = _sidecar(acc, ok)
    p = lc.write_orbit_cov_sidecar(tmp_path / "g.orbit_cov.h5", sc)
    with pytest.raises(FileExistsError):
        lc.write_orbit_cov_sidecar(p, sc)
    back = lc.read_orbit_cov_sidecar(p)
    np.testing.assert_array_equal(back.orbit_cov, sc.orbit_cov)
    attrs = {"primary_model": "pm", "fragment_fingerprint": "fp"}
    lc.merge_orbit_cov_sidecar(truth, outcome, attrs, back)
    assert outcome["orbit_cov_status"].tolist() == ["replayed", "prefilter_excluded:lcf1", "", "replayed", "", "prefilter_excluded:lcf1"]
    assert np.isfinite(outcome["orbit_cov"][[0, 3]]).all() and np.isnan(outcome["orbit_cov"][[1, 2, 4, 5]]).all()
    np.testing.assert_array_equal(outcome["orbit_p0"][3], sc.orbit_p0[1])
    with pytest.raises(ValueError, match="already carries"):
        lc.merge_orbit_cov_sidecar(truth, outcome, attrs, back)
    for bad_attrs, bad_acc, msg in (({"primary_model": "x", "fragment_fingerprint": "fp"}, acc_mask, "primary_model"),
                                    (attrs, ~acc_mask, "accepted-draw set")):
        with pytest.raises(ValueError, match=msg):
            lc.merge_orbit_cov_sidecar(truth, {"accepted_orbital": bad_acc}, bad_attrs, back)
    with pytest.raises(ValueError, match="n_draws"):
        lc.merge_orbit_cov_sidecar({"draw_index": di[:5]}, {"accepted_orbital": acc_mask[:5]}, attrs, back)
    with pytest.raises(ValueError, match="smoke"):
        lc.merge_orbit_cov_sidecar(truth, {"accepted_orbital": acc_mask}, attrs, _sidecar(acc, ok, smoke=True))
    one = {int(acc[0]): {"orbit_p0": [0.0] * 12, "orbit_cov": [0.0] * 144, "orbit_cov_sigma_check": 0.0, "orbit_cov_cpu_seconds": 0.0}}
    with pytest.raises(ValueError, match="exactly the accepted"):  # incomplete replay: refused on write
        lc.write_orbit_cov_sidecar(tmp_path / "bad.h5", lc.sidecar_from_records(acc, np.ones(4, bool), one, dict(sc.attrs)))


@pytest.mark.unit
def test_load_generation_draws_merges_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    di = np.arange(6, dtype=np.int64) + 500
    acc_mask = np.array([True, True, False, True, False, True])
    art = tmp_path / "single_gen99.h5"

    def fake_read(path: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return ({"draw_index": di.copy()}, {"accepted_orbital": acc_mask.copy(), "nss_solution_type": np.array(["Orbital"] * 6)},
                {"primary_model": "pm", "fragment_fingerprint": "fp"})

    monkeypatch.setattr(ps, "read_proposal_artifact", fake_read)
    _, o = ps.load_generation_draws([art], require_primary_model="pm")
    assert "orbit_cov" not in o
    with pytest.raises(FileNotFoundError, match="sidecar"):
        ps.load_generation_draws([art], require_primary_model="pm", orbit_cov_sidecar=True)
    lc.write_orbit_cov_sidecar(lc.orbit_cov_sidecar_path(art), _sidecar(di[acc_mask], np.array([True, False, True, False])))
    _, o = ps.load_generation_draws([art], require_primary_model="pm", orbit_cov_sidecar=True)
    assert o["orbit_cov"].shape == (6, 12, 12) and o["orbit_cov_status"][1] == "prefilter_excluded:lcf1"


# ---------------------------------------------------------------------------
# Adapter marks: excluded rows are recorded as not applicable with the pre-filter reason
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_excluded_rows_not_applicable_with_reason(registry: SampleSelectionRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    truth, outcome = _draws([3.0, 3.0], [13.0, 13.0], ["Orbital", "Orbital"])
    outcome["orbit_cov_status"] = np.array(["replayed", "prefilter_excluded:lcf1"])
    outcome["orbit_p0"][1] = np.nan
    outcome["orbit_cov"][1] = np.nan
    rows = _rows(truth, outcome, include_orbit=True)
    reason = "mock_prefilter_excluded:lcf1"
    assert rows[0]["mock_orbit_cov_status"] == "replayed" and rows[0]["mock_orbit_solution"] is not None
    assert "mock_orbit_p0" not in rows[1] and rows[1]["sigma_m2_astrometric_msun"] == NotApplicable(reason)
    # Andrews: the ATF columns of the excluded row stop the chain at atf_covariance_valid with the reason
    spec = registry.resolved("andrews2022")
    atf = mock_rows_atf_columns(rows, spec, m1_msun=np.full(2, 0.9), m1_sigma_msun=np.full(2, 0.05))
    ex = atf[int(rows[1]["source_id"])]
    assert ex["andrews_atf_covariance_ok"] == NotApplicable(reason) and ex["andrews_atf_covariance_failure"] == reason
    assert math.isfinite(ex["andrews_atf_abs_g_mag"]) and ex["andrews_atf_goodness_of_fit"] == 2.0
    assert atf[int(rows[0]["source_id"])]["andrews_atf_covariance_ok"] is True
    res = forward_model_selection(registry, "andrews2022", reproduction_columns=atf).evaluate(rows)
    first = res.outcomes_by_source[int(rows[1]["source_id"])][0]
    assert first[0] == "atf_covariance_valid" and first[1] is CutOutcome.NOT_APPLICABLE and first[2] == reason
    assert res.outcomes_by_source[int(rows[0]["source_id"])][0][1] is CutOutcome.PASSED
    # El-Badry 2026: the mark survives enrichment and the sigma attach, and m2_error records the reason
    sel = forward_model_selection(registry, "elbadry2026", extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    monkeypatch.setattr(sel, "_nss_enrichment_index", lambda: {})
    enriched = [dict(r) for r in sel._enrich_rows_for_spec(rows)]
    sel._attach_elbadry_m2_sigma_inplace(enriched)
    assert enriched[1]["sigma_m2_astrometric_msun"] == NotApplicable(reason)
    assert isinstance(enriched[0].get("sigma_m2_astrometric_msun"), float)
    cut = next(c for b in sel.spec.branches for s in b.subsamples if s.id == "sub_chandrasekhar" for c in s.cuts if c.id == "m2_error")
    assert evaluate_cut(cut, sel.bind_row(enriched[1])) == (CutOutcome.NOT_APPLICABLE, reason)


# ---------------------------------------------------------------------------
# simulate_one: the CO covariance path for a luminous proposal
# ---------------------------------------------------------------------------


class _LinearGaiamock:
    def __init__(self, x: np.ndarray, theta: np.ndarray) -> None:
        self.x, self.theta = x, theta

    def get_astrometric_residuals_12par(self, t, psi, pf, obs, err, theta, c_funcs):  # type: ignore[no-untyped-def]
        return (obs - self.x @ np.asarray(theta, float)) / err

    def get_astrometric_chi2(self, t, psi, pf, obs, err, period, phi_p, ecc, c_funcs):  # type: ignore[no-untyped-def]
        th = self.theta
        r = self.get_astrometric_residuals_12par(t, psi, pf, obs, err, th, c_funcs)
        return float(np.sum(r**2)), np.array([th[0], th[3], th[1], th[4], th[2], th[9], th[11], th[8], th[10]])


@pytest.mark.api
def test_simulate_one_luminous_orbit_covariance(monkeypatch: pytest.MonkeyPatch, ccfg: lc.LuminousCovarianceConfig) -> None:
    from darkhunter_pop import compact_objects as cob
    from darkhunter_pop import epoch_model as em

    frag = ps.load_proposal_set_fragment(LUMINOUS_FRAGMENT)
    assert frag.proposal.compact_object is None
    seen: list[dict[str, Any]] = []
    n = 300
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, 12))
    theta = np.r_[0.0, 0.0, 2.0, 1.0, -1.0, 300.0, 0.2, 0.5, 1.0, 0.1, -0.2, 0.9]
    err = np.full(n, 0.2)
    obs = x @ theta + err * rng.normal(size=n)
    gm = _LinearGaiamock(x, theta)
    epochs = (np.linspace(0, 3, n), np.zeros(n), np.zeros(n), obs, err)
    vec = [theta[2], 0.01, theta[8], 0.01, theta[9], 0.01, theta[10], 0.01, theta[11], 0.01, theta[5], 1.0, theta[7], 0.01,
           theta[6], 0.01, 40.0, 1.2, 0.01, 20, n, 1.0, 3.0]

    def fake_run_cascade(g, cf, predict, config, source, **kw):  # type: ignore[no-untyped-def]
        seen.append(kw)
        return em.CascadeRun(cascade=list(vec), n_obs=n, n_transits=n // 9, n_visibility_periods=20, ruwe_scale=1.0,
                             epochs=epochs if kw.get("keep_epochs") else None, nss_type="Orbital")

    monkeypatch.setattr(em, "run_cascade", fake_run_cascade)
    monkeypatch.setattr(ps, "seeded_global_rng", lambda seeds, cf: contextlib.nullcontext())
    cfg_pipe = load_config()
    cuts = cfg_pipe.active_dr().selection_function_astrometric.orbital_solution_cuts
    setup = ps.EpochSetup(config=dataclasses.replace(em.epoch_model_config_from_mapping(cfg_pipe.active_dr().epoch_model), enabled=True), gaps_jd=None)
    draw = {"generation": 59, "draw_index": 13_000_007, "ra_deg": 10.0, "dec_deg": -5.0, "parallax_mas": 2.0, "pmra_masyr": 0.0,
            "pmdec_masyr": 0.0, "m1_msun": 1.0, "m2_msun": 0.8, "period_days": 300.0, "Tp_days": 10.0, "eccentricity": 0.2,
            "Omega_rad": 1.0, "inc_deg": 60.0, "omega_rad": 2.0, "phot_g_mean_mag": 14.0, "flux_ratio": 0.3, "l_deg": 100.0, "b_deg": 20.0}
    plain = ps.simulate_one(draw, gaiamock=gm, c_funcs=None, cfg=frag.proposal, cuts=cuts, epoch=setup)
    assert "orbit_cov" not in plain and not seen[-1].get("keep_epochs")
    rec = ps.simulate_one(draw, gaiamock=gm, c_funcs=None, cfg=frag.proposal, cuts=cuts, epoch=setup,
                          orbit_covariance_step=ccfg.covariance_jacobian_step)
    assert seen[-1].get("keep_epochs") is True
    for k in ("cascade", "solution_type", "accepted_orbital", "nss_solution_type", "numpy_seed", "c_rand_seed"):
        assert rec[k] == plain[k], k  # keeping the epochs leaves the outcome unchanged
    want = cob.orbit_covariance_from_epochs(gm, None, epochs, rec["cascade"], step=ccfg.covariance_jacobian_step)
    np.testing.assert_array_equal(np.asarray(rec["orbit_cov"]).reshape(12, 12), want.covariance)
    np.testing.assert_array_equal(rec["orbit_p0"], want.p0)
    co = ps.load_proposal_set_fragment(CO_FRAGMENT).proposal
    with pytest.raises(ValueError, match="covariance_jacobian_step"):
        ps.simulate_one({**draw, "generation": 1000}, gaiamock=gm, c_funcs=None, cfg=co, cuts=cuts, epoch=setup, orbit_covariance_step=1e-6)
    with pytest.raises(ValueError, match="epoch model"):
        ps.simulate_one(draw, gaiamock=gm, c_funcs=None, cfg=frag.proposal.model_copy(update={"epoch_model": "off"}), cuts=cuts,
                        epoch=None, orbit_covariance_step=1e-8)


@pytest.mark.unit
def test_replay_check_flags_every_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(repo_root() / "scripts"))
    mod = importlib.import_module("replay_luminous_covariance_704")
    outcome = {"numpy_seed": np.array([5]), "c_rand_seed": np.array([6]), "solution_type": np.array(["twelve_parameter_orbital"]),
               "accepted_orbital": np.array([True]), "nss_solution_type": np.array(["Orbital"]), "cascade": np.ones((1, 23))}
    rec = {"numpy_seed": 5, "c_rand_seed": 6, "solution_type": "twelve_parameter_orbital", "accepted_orbital": True,
           "nss_solution_type": "Orbital", "cascade": [1.0] * 23, "orbit_cov": [0.0] * 144}
    assert mod.check_replayed(rec, 0, outcome) == []
    worse = {**rec, "numpy_seed": 4, "nss_solution_type": "AstroSpectroSB1", "cascade": [1.0 + 1e-6] * 23}
    worse.pop("orbit_cov")
    assert mod.check_replayed(worse, 0, outcome) == ["seeds", "nss_solution_type", "cascade", "no_orbit_cov"]


@pytest.mark.unit
def test_luminous_truth_without_co_component() -> None:
    """#706: luminous generations have no co_component truth column; rows carry -1, CO codes stay unchanged."""
    from darkhunter_pop.mock_selection_rows import LUMINOUS_CO_COMPONENT

    truth, outcome = _draws([3.0, 0.5], [13.0, 13.0], ["Orbital", "Orbital"])
    with_co = _rows(truth, outcome)
    assert [r["mock_co_component"] for r in with_co] == [0, 0]
    truth.pop("co_component")
    rows = _rows(truth, outcome)
    assert LUMINOUS_CO_COMPONENT == -1 and [r["mock_co_component"] for r in rows] == [-1, -1]
