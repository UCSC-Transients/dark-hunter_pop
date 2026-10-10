"""Proposal set: densities, sampler, weights, ESS, gaiamock call, artifact (#391).

docs/MOCK_POPULATION_SPEC.md §3. gaiamock is replaced by a fake that draws from numpy's
global RNG (as gaiamock_mod does), so seeding and plumbing are tested without the overlay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config

FRAGMENT = "config/population/proposal_set_pilot.yaml"
DECIDED = "config/population/proposal_set_decided_tune.yaml"


@pytest.fixture(scope="module")
def fragment() -> ps.ProposalSetFragment:
    return ps.load_proposal_set_fragment(FRAGMENT)


def _parent(n: int = 50, seed: int = 0) -> ps.ParentSnapshot:
    rng = np.random.default_rng(seed)
    cols = {
        "source_id": np.arange(n, dtype=np.int64) + 1000,
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-60, 60, n),
        "parallax": rng.uniform(0.3, 20.0, n),
        "pmra": rng.normal(0, 5, n),
        "pmdec": rng.normal(0, 5, n),
        "phot_g_mean_mag": rng.uniform(8, 18, n),
    }
    m1 = rng.uniform(0.7, 2.0, n)
    usable = np.ones(n, dtype=bool)
    usable[0] = False  # one unusable row must never be drawn
    return ps.ParentSnapshot(
        columns=cols,
        m1_msun=m1,
        m1_source=np.full(n, "MSC"),
        atmosphere_logg=rng.uniform(3.0, 5.0, n),
        truth_parallax_mas=cols["parallax"] * 1.01,
        is_giant=np.zeros(n, dtype=bool),
        flags={"all": usable},
        usable=usable,
        meta={"parent_h5_sha256": "x"},
        scale_to_full=1000.0,
        path=Path("."),
    )


@pytest.mark.unit
def test_fragment_loads_and_is_not_auto_merged(fragment: ps.ProposalSetFragment) -> None:
    assert fragment.proposal.parallax_floor_mas > 0
    # load_config merges config/fragments/*.yaml; the pilot config must stay out of it.
    cfg = load_config()
    assert not hasattr(cfg, "proposal")
    assert len(ps.fragment_fingerprint(fragment)) == 12


@pytest.mark.unit
def test_parent_adql_never_cuts_on_outcomes() -> None:
    q = ps.build_gaia_source_parent_adql(k=1_000_000, parallax_floor_mas=0.2, g_max=19.0)
    where = q.split("WHERE", 1)[1]
    assert "random_index < 1000000" in where
    assert "phot_g_mean_mag < 19.0" in where and "parallax > 0.2" in where
    for outcome in ("ruwe", "visibility_periods_used", "ipd_", "excess"):
        assert outcome not in where
    assert "gs.ruwe" in q and "ap.teff_msc1" in q
    with pytest.raises(ValueError):
        ps.build_gaia_source_parent_adql(k=0, parallax_floor_mas=0.2, g_max=19.0)


@pytest.mark.physics
def test_proposal_marginals_normalize(fragment: ps.ProposalSetFragment) -> None:
    p = fragment.proposal
    lp = np.linspace(-1, 9, 400_001)
    assert np.trapz(np.exp(ps.log_q_log_p(lp, p.period)), lp) == pytest.approx(1.0, abs=1e-4)
    lm2 = np.linspace(-3, 3, 600_001)
    assert np.trapz(np.exp(ps.log_q_log_m2(lm2, np.log10(1.3), p.m2)), lm2) == pytest.approx(1.0, abs=1e-4)
    lf = np.linspace(-10, 5, 600_001)
    for m1, m2 in ((1.0, 0.6), (1.0, 1e-3)):  # second: no Janssens value -> uniform only
        lum = np.exp(ps.log_q_flux(lf, np.zeros_like(lf, bool), np.full_like(lf, m1), np.full_like(lf, m2), p.flux))
        assert np.trapz(lum, lf) + p.flux.dark_fraction == pytest.approx(1.0, abs=1e-4)
    e = np.linspace(0, 1, 100_001)
    assert np.trapz(np.exp(ps.log_q_ecc(e, np.full_like(e, 100.0), p.eccentricity)), e) == pytest.approx(1.0, abs=1e-3)
    assert ps.log_q_ecc([0.0, 0.1], [1.5, 1.5], p.eccentricity).tolist() == [0.0, -np.inf]


@pytest.mark.unit
def test_sampler_reproducible_and_logq_consistent(fragment: ps.ProposalSetFragment) -> None:
    parent = _parent()
    cfg = fragment.proposal.model_copy(update={"n_draws": 400})
    a = ps.sample_proposal(parent, cfg)
    b = ps.sample_proposal(parent, cfg)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert not np.any(a["parent_row"] == 0)
    np.testing.assert_allclose(a["parallax_mas"], parent.truth_parallax_mas[a["parent_row"]])
    np.testing.assert_allclose(a["measured_parallax_mas"], parent.columns["parallax"][a["parent_row"]])
    assert np.all(a["flux_ratio"][a["is_dark"]] == 0.0)
    assert np.all(a["eccentricity"][a["period_days"] <= 2.0] == 0.0)
    np.testing.assert_allclose(ps.log_q_total_for(a, parent, cfg), a["log_q_total"])
    assert np.all(np.isfinite(a["log_q_total"]))
    other = ps.sample_proposal(parent, cfg.model_copy(update={"generation": 1}))
    assert not np.array_equal(other["period_days"], a["period_days"])


@pytest.mark.physics
def test_weights_recover_known_expectation(fragment: ps.ProposalSetFragment) -> None:
    # Target = proposal x 3 -> w = scale * 3 / N for every draw; Σw = 3 * scale.
    rng_q = np.random.default_rng(1).normal(size=1000)
    w = ps.importance_weights(rng_q + math.log(3.0), [rng_q], [1000], scale_to_full=10.0)
    np.testing.assert_allclose(w, 3.0 * 10.0 / 1000)
    assert ps.kish_ess(w) == pytest.approx(1000)
    # Deterministic mixture: two identical generations of n1, n2 equal one of n1 + n2.
    w2 = ps.importance_weights(rng_q, [rng_q, rng_q], [400, 600], scale_to_full=1.0)
    np.testing.assert_allclose(w2, ps.importance_weights(rng_q, [rng_q], [1000], scale_to_full=1.0))
    assert ps.importance_weights([-np.inf], [[0.0]], [1], scale_to_full=1.0).tolist() == [0.0]


@pytest.mark.physics
def test_mixture_estimator_unbiased() -> None:
    # E_p[1] over p = U(0,1) from two proposals U(0,2) and U(0,1): Σw ≈ 1.
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(0, 2, 20_000), rng.uniform(0, 1, 5_000)])
    lp = np.where(x < 1, 0.0, -np.inf)
    lq1 = np.full_like(x, math.log(0.5))
    lq2 = np.where(x < 1, 0.0, -np.inf)
    w = ps.importance_weights(lp, [lq1, lq2], [20_000, 5_000], scale_to_full=1.0)
    assert w.sum() == pytest.approx(1.0, rel=0.02)


@pytest.mark.unit
def test_ess_and_trust_criterion() -> None:
    assert ps.kish_ess([1, 1, 1, 1]) == pytest.approx(4)
    assert ps.kish_ess([1, 0, 0, 0]) == pytest.approx(1)
    assert ps.kish_ess([]) == 0.0
    vals = np.array([0.5, 0.5, 1.5, 1.5, 1.5])
    w = np.array([0.01, 0.01, 1.0, 1.0, 1.0])
    bw = ps.binned_weights(vals, w, [0, 1, 2])
    assert bw.n_draws.tolist() == [2, 3]
    np.testing.assert_allclose(bw.ess, [2.0, 3.0])
    # σ_MC/σ_Poisson < t  <=>  ESS_b >= N_b / t²
    np.testing.assert_allclose(bw.mc_to_poisson, np.sqrt(bw.sum_w / bw.ess))
    assert bw.trusted(0.1).tolist() == [False, False]
    assert bw.trusted(1.0).tolist() == [True, False]


@pytest.mark.unit
def test_published_acceleration(fragment: ps.ProposalSetFragment) -> None:
    cfg = fragment.proposal.acceleration_publication
    nine = [-9.0] * 23
    nine[1] = 25.0
    assert ps.published_acceleration(nine, cfg)
    seven = [-7.0] * 23
    seven[1], seven[9] = 25.0, 23.0
    assert not ps.published_acceleration(seven, cfg)
    seven[9] = 10.0
    assert ps.published_acceleration(seven, cfg)
    assert not ps.published_acceleration([-1.0] * 23, cfg)


class _FakeGaiamock:
    """Draws from numpy's global RNG, like gaiamock_mod's epoch noise."""

    def run_full_astrometric_cascade(self, **kw: Any) -> list[float]:
        noise = np.random.normal()
        if kw["period"] > 1000:
            return [-1.0] * 23
        res = [0.0] * 23
        res[0], res[1] = kw["parallax"] + 0.01 * noise, 0.01
        res[10], res[11] = kw["period"], 1.0
        res[14], res[15] = kw["ecc"], 0.01
        res[16], res[17], res[18] = kw["inc_deg"], 2.0, 0.05
        res[21], res[22] = 0.0, 3.0
        return res


@pytest.mark.api
def test_simulate_and_artifact_roundtrip(tmp_path: Path, fragment: ps.ProposalSetFragment) -> None:
    cfg_pipe = load_config()
    cuts = cfg_pipe.active_dr().selection_function_astrometric.orbital_solution_cuts
    parent = _parent()
    prop = fragment.proposal.model_copy(update={"n_draws": 12})
    truth = ps.sample_proposal(parent, prop)
    keys = [k for k, v in truth.items() if np.asarray(v).ndim == 1]
    draws = [{k: truth[k][i].item() for k in keys} for i in range(12)]
    gm = _FakeGaiamock()
    out1 = [ps.simulate_one(d, gaiamock=gm, c_funcs=None, cfg=prop, cuts=cuts) for d in draws]
    out2 = [ps.simulate_one(d, gaiamock=gm, c_funcs=None, cfg=prop, cuts=cuts) for d in reversed(draws)]
    by_idx = {o["draw_index"]: o for o in out2}
    for o in out1:
        assert o["cascade"] == by_idx[o["draw_index"]]["cascade"]  # order-independent seeds
    path = ps.write_proposal_artifact(
        tmp_path / "a.h5", truth, out1, fragment=fragment.model_copy(update={"proposal": prop}),
        parent=parent, provenance={"label": "test"},
    )
    t, o, attrs = ps.read_proposal_artifact(path)
    np.testing.assert_array_equal(t["draw_index"], truth["draw_index"])
    assert o["cascade"].shape == (12, ps.CASCADE_VECTOR_LENGTH)
    assert set(o["solution_type"]) <= {"five_parameter", "twelve_parameter_orbital", "orbital_failed_cuts"}
    assert float(attrs["scale_to_full"]) == 1000.0
    with pytest.raises(ValueError):
        ps.write_proposal_artifact(tmp_path / "b.h5", truth, out1[:-1], fragment=fragment, parent=parent, provenance={})


@pytest.mark.physics
def test_mds17_target_support(fragment: ps.ProposalSetFragment) -> None:
    tgt = fragment.target_mds17
    truth = {
        "m1_msun": np.array([1.0, 1.0, 1.0, 1.0, 0.05]),
        "m2_msun": np.array([0.5, 0.5, 0.05, 0.5, 0.03]),
        "period_days": np.array([300.0, 300.0, 300.0, 1e9, 300.0]),
        "eccentricity": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        "is_dark": np.array([False, True, False, False, False]),
        "log10_flux_ratio": ps.relation_log10_flux_ratio([1.0, 1.0, 1.0, 1.0, 0.05], [0.5, 0.5, 0.05, 0.5, 0.03]),
    }
    lam = ps.mds17_luminous_log_intensity(truth, tgt)
    assert np.isfinite(lam[0])
    # dark, q < 0.1, log P > 8 and M1 below the low-mass zero point get zero intensity
    assert np.all(np.isneginf(lam[1:]))


@pytest.mark.unit
def test_janssens_flux_ratio() -> None:
    assert float(ps.relation_log10_flux_ratio(1.0, 1.0)) == pytest.approx(0.0)
    assert float(ps.relation_log10_flux_ratio(1.0, 0.5)) < -1.0
    assert np.isnan(ps.janssens_absolute_g(1e-3))


@pytest.mark.unit
def test_decided_config_records_decisions() -> None:
    d = ps.load_proposal_set_fragment(DECIDED).proposal
    assert "5963152741" in d.decision_ref
    assert d.parallax_floor_mas == 0.2
    assert d.halbwachs_ipd_cstar_cuts == "applied_star_values"
    assert d.truth_distance == "bailer_jones2021_geometric"
    assert d.light_split == "observed_g_is_system_total"
    tgt = ps.load_proposal_set_fragment(DECIDED).target_mds17
    assert tgt.mass_luminosity == "janssens2022" and tgt.flux_sigma_dex == 0.1


@pytest.mark.unit
def test_bailer_jones_join() -> None:
    q = ps.build_gaia_source_parent_adql(k=10, parallax_floor_mas=0.2, g_max=19.0, include_bailer_jones=True)
    assert "LEFT JOIN external.gaiaedr3_distance AS bj ON gs.source_id = bj.source_id" in q
    assert "bj.r_med_geo" in q and "bj.r_hi_geo" in q
    assert "bj." not in ps.build_gaia_source_parent_adql(k=10, parallax_floor_mas=0.2, g_max=19.0)


@pytest.mark.physics
def test_cstar_matches_riello2021() -> None:
    # Riello et al. (2021) Eq. 6 polynomial at its three colour regimes, and Eq. 18.
    x = np.array([0.0, 1.0, 5.0])
    c = np.array([1.2, 1.3, 1.9])
    expect = c - np.array([1.154360, 1.162004 + 0.011464 + 0.049255 - 0.005879, 1.057572 + 0.140537 * 5.0])
    np.testing.assert_allclose(ps.corrected_flux_excess(x, c), expect)
    assert float(ps.sigma_cstar(10.0)) == pytest.approx(0.0059898 + 8.817481e-12 * 10.0**7.618399)
    assert np.isnan(ps.corrected_flux_excess(np.nan, 1.2))


@pytest.mark.unit
def test_halbwachs_flags() -> None:
    cuts = ps.load_proposal_set_fragment(DECIDED).proposal.halbwachs_cuts
    cols = {
        "ipd_frac_multi_peak": np.array([0.0, 2.0, 3.0, 0.0, np.nan]),
        "ipd_gof_harmonic_amplitude": np.array([0.05, 0.05, 0.05, 0.1, 0.05]),
        "bp_rp": np.array([1.0, 1.0, 1.0, 1.0, np.nan]),
        "phot_bp_rp_excess_factor": np.full(5, 1.162004 + 0.011464 + 0.049255 - 0.005879),
        "phot_g_mean_mag": np.full(5, 12.0),
    }
    f = ps.halbwachs_input_flags(cols, cuts)
    assert f["halbwachs_ipd"].tolist() == [True, True, False, False, False]
    assert f["halbwachs_cstar"].tolist() == [True, True, True, True, False]  # no colour fails


@pytest.mark.api
def test_load_parent_snapshot_applies_decided_filters(tmp_path: Path) -> None:
    import h5py
    import yaml

    n = 6
    cols: dict[str, np.ndarray] = {c: np.full(n, np.nan) for c in ps.GAIA_SOURCE_PARENT_COLUMNS}
    cols["source_id"] = np.arange(n, dtype=np.int64)
    cols["random_index"] = np.arange(n, dtype=np.int64)
    for c, v in (("ra", 10.0), ("dec", 5.0), ("pmra", 1.0), ("pmdec", 1.0), ("phot_g_mean_mag", 12.0)):
        cols[c] = np.full(n, v)
    cols["parallax"] = np.array([5.0, 5.0, 0.1, 5.0, 5.0, 5.0])  # row 2 below the floor
    cols["teff_msc1"], cols["logg_msc1"], cols["mh_msc"] = np.full(n, 5800.0), np.full(n, 4.4), np.zeros(n)
    cols["logg_msc1"][4] = 3.0  # giant: kept, flagged
    cols["teff_msc1"][5] = np.nan  # no atmosphere (no GSP-Phot either): dropped
    cols["ipd_frac_multi_peak"], cols["ipd_gof_harmonic_amplitude"] = np.zeros(n), np.full(n, 0.01)
    cols["ipd_frac_multi_peak"][3] = 10.0  # fails Halbwachs (b)
    cols["bp_rp"] = np.full(n, 0.8)
    cols["phot_bp_rp_excess_factor"] = ps.corrected_flux_excess(0.8, 0.0) * -1.0
    cols["r_med_geo"] = np.array([200.0, np.nan, 200.0, 200.0, 250.0, 200.0])  # row 1: no BJ
    cols["r_lo_geo"], cols["r_hi_geo"] = cols["r_med_geo"] * 0.9, cols["r_med_geo"] * 1.1
    d = tmp_path / "snap"
    d.mkdir()
    with h5py.File(d / "parent.h5", "w") as h:
        for k, v in cols.items():
            h.create_dataset(k, data=v)
    meta = {"parent_h5_sha256": ps._sha256(d / "parent.h5"), "random_index_max_exclusive": 1000,
            "gaia_source_total_rows": ps.GAIA_SOURCE_TOTAL_ROWS}
    (d / "meta.yaml").write_text(yaml.safe_dump(meta))
    prop = ps.load_proposal_set_fragment(DECIDED).proposal
    parent = ps.load_parent_snapshot(d, load_config(), prop, m1_cache=False)
    assert parent.usable.tolist() == [True, False, False, False, True, False]
    assert parent.is_giant.tolist() == [False, False, False, False, True, False]
    assert parent.truth_parallax_mas[0] == pytest.approx(5.0)  # 1000 / 200 pc
    assert parent.truth_parallax_mas[4] == pytest.approx(4.0)  # BJ, not the measured 5.0
    att = parent.attrition()
    assert att["snapshot_rows"] == 6 and att[list(parent.flags)[-1]] == 2


@pytest.mark.unit
def test_real_comparison_keep() -> None:
    prop = ps.load_proposal_set_fragment(DECIDED).proposal
    cols = {
        "parallax": np.array([1.0, 0.1, 1.0]),
        "teff_msc1": np.array([5800.0, 5800.0, np.nan]),
        "logg_msc1": np.array([4.4, 4.4, np.nan]),
        "mh_msc": np.array([0.0, 0.0, np.nan]),
    }
    keep, counts = ps.real_comparison_keep(cols, prop)
    assert keep.tolist() == [True, False, False]
    assert counts == {"rows": 3, "parallax_floor": 2, "parallax_floor_and_atmosphere": 1}


@pytest.mark.physics
def test_weighted_ks() -> None:
    rng = np.random.default_rng(5)
    real = rng.normal(size=4000)
    mock = rng.uniform(-4, 4, 4000)
    # weights turning the uniform mock into the real normal distribution: small D
    w = np.exp(-0.5 * mock**2)
    d_good, n_eff, p_good = ps.weighted_ks(real, mock, w)
    d_bad, _, p_bad = ps.weighted_ks(real, mock, np.ones_like(mock))
    assert d_good < 0.05 < d_bad
    assert p_good > 0.01 > p_bad
    assert 0 < n_eff < 4000


# ---------------------------------------------------------------------------
# #409 / #410: bounded MdS17-support eccentricity proposal
# ---------------------------------------------------------------------------

RESTART = "config/population/proposal_set_restart_smoke.yaml"


def _ecc_cfg() -> ps.EccentricityProposalConfig:
    return ps.load_proposal_set_fragment(RESTART).proposal.eccentricity


@pytest.mark.physics
def test_bounded_ecc_proposal_covers_mds17_support_and_normalizes() -> None:
    from darkhunter_pop import moe_distefano as mds

    cfg = _ecc_cfg()
    table = mds.load_mds17_table()
    for p in (3.0, 30.0, 500.0, 5000.0, 1e6):
        emax = float(mds.e_max(p, table))
        assert float(cfg.e_max(p)) == pytest.approx(emax)
        e = np.linspace(1e-9, emax * (1 - 1e-9), 2001)
        assert np.all(np.isfinite(ps.log_q_ecc(e, np.full_like(e, p), cfg)))  # q > 0 on the whole support
        if emax > 0.95:  # the region #409 found uncovered
            hi = e[e > 0.95]
            assert hi.size and np.all(np.isfinite(ps.log_q_ecc(hi, np.full_like(hi, p), cfg)))
        from scipy.integrate import quad

        def dens(x: float, period: float = p) -> float:
            return float(np.exp(ps.log_q_ecc(np.array([x]), np.array([period]), cfg))[0])

        brk = sorted({emax, cfg.defensive_e_max})
        total = sum(quad(dens, a0, b0, limit=400)[0] for a0, b0 in zip([0.0] + brk[:-1], brk))
        assert total == pytest.approx(1.0, abs=1e-4)


@pytest.mark.physics
def test_bounded_ecc_weights_are_bounded() -> None:
    from darkhunter_pop import moe_distefano as mds

    cfg = _ecc_cfg()
    tgt = ps.load_proposal_set_fragment(RESTART).target_mds17
    table = mds.load_mds17_table()
    floor = tgt.provisional_eta_floor
    worst = 0.0
    for m1 in (0.8, 1.0, 3.0, 8.0):
        for p in (2.5, 4.0, 10.0, 100.0, 1e3, 1e5):
            emax = float(mds.e_max(p, table))
            e = np.geomspace(1e-12, emax * (1 - 1e-12), 4001)
            pe = mds.e_density(e, m1, p, table, m1_interpolation="linear_m1", eta_floor=floor)
            q = np.exp(ps.log_q_ecc(e, np.full_like(e, p), cfg))
            worst = max(worst, float(np.max(pe / q)))
    # analytic bound: (eta+1)/((eta_q+1) floor_weight) near e = 0, (eta+1)/support_weight elsewhere
    bound = 2.0 / ((cfg.floor_eta + 1.0) * cfg.floor_weight)
    assert worst < bound


@pytest.mark.physics
def test_bounded_ecc_importance_sampling_matches_quadrature() -> None:
    # #410 acceptance: IS of ∫ p_e de = 1 at fixed M1 with a few draws per row, eta < -0.5 included.
    from darkhunter_pop import moe_distefano as mds

    cfg = _ecc_cfg()
    table = mds.load_mds17_table()
    rng = np.random.default_rng(410)
    periods = 10.0 ** rng.uniform(0.35, 4.0, 20_000)  # includes log P < 1.4 where eta < -0.5
    e = ps.sample_ecc(periods, cfg, rng)
    w = mds.e_density(e, 1.0, periods, table, m1_interpolation="linear_m1", eta_floor=-0.9) / np.exp(
        ps.log_q_ecc(e, periods, cfg)
    )
    mean, err = w.mean(), w.std() / np.sqrt(w.size)
    assert abs(mean - 1.0) < 4 * err + 1e-3
    assert err < 0.01  # finite, small variance


@pytest.mark.unit
def test_check_eccentricity_bounded() -> None:
    cfg = _ecc_cfg()
    ps.check_eccentricity_bounded(cfg, -0.9)
    with pytest.raises(ValueError):
        ps.check_eccentricity_bounded(cfg, -0.95)  # target floor below the proposal power law
    old = ps.load_proposal_set_fragment(DECIDED).proposal.eccentricity
    assert old.shape == "uniform"
    with pytest.raises(ValueError, match="#409"):
        ps.check_eccentricity_bounded(old, -0.9)


@pytest.mark.unit
def test_uniform_ecc_sampling_unchanged_for_old_artifacts() -> None:
    old = ps.load_proposal_set_fragment(DECIDED).proposal.eccentricity
    p = np.array([1.0, 50.0, 500.0])
    a = ps.sample_ecc(p, old, np.random.default_rng(1))
    b = np.where(p <= old.circular_period_days, 0.0, np.random.default_rng(1).uniform(0.0, old.e_cap, size=3))
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_epoch_setup_must_match_config() -> None:
    prop = ps.load_proposal_set_fragment(RESTART).proposal
    assert prop.epoch_model == "dr3_config"
    with pytest.raises(ValueError, match="epoch is missing"):
        ps.simulate_one({"generation": 20, "draw_index": 0}, gaiamock=_FakeGaiamock(), c_funcs=None,
                        cfg=prop, cuts=None, epoch=None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_real_comparison_keep_applies_input_cuts() -> None:
    prop = ps.load_proposal_set_fragment(DECIDED).proposal
    green = 1.162004 + 0.011464 * 0.8 + 0.049255 * 0.64 - 0.005879 * 0.512
    cols = {
        "source_id": np.array([1, 2, 3, 4]),
        "parallax": np.array([1.0, 1.0, 1.0, 1.0]),
        "teff_msc1": np.full(4, 5800.0), "logg_msc1": np.full(4, 4.4), "mh_msc": np.zeros(4),
    }
    inp = {
        "source_id": np.array([3, 1, 2]),  # row 4 missing -> fails
        "ipd_frac_multi_peak": np.array([0.0, 0.0, 5.0]),
        "ipd_gof_harmonic_amplitude": np.array([0.01, 0.01, 0.01]),
        "bp_rp": np.full(3, 0.8), "phot_bp_rp_excess_factor": np.full(3, green),
        "phot_g_mean_mag": np.full(3, 12.0),
    }
    keep, counts = ps.real_comparison_keep(cols, prop, inp)
    assert keep.tolist() == [True, False, True, False]
    assert counts["missing_input_columns"] == 1 and counts["and_halbwachs_ipd_cstar"] == 2


def _synthetic_parent_for_malmquist(n: int = 4000, zp: float = 0.3, sig: float = 0.2, seed: int = 9) -> ps.ParentSnapshot:
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(0.9, 1.4, n)
    r = rng.uniform(100.0, 400.0, n)
    mg = ps.janssens_absolute_g(m1) + zp + rng.normal(0.0, sig, n)
    g = mg + 5 * np.log10(r) - 5
    cols = {
        "source_id": np.arange(n), "phot_g_mean_mag": g, "ruwe": np.where(rng.uniform(size=n) < 0.9, 1.0, 2.0),
        "r_med_geo": r, "r_lo_geo": r * 0.999, "r_hi_geo": r * 1.001, "l": np.zeros(n), "b": np.full(n, 30.0),
        "parallax": 1000.0 / r, "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n),
        "pmra": np.zeros(n), "pmdec": np.zeros(n),
    }
    giant = np.zeros(n, bool)
    giant[:10] = True
    return ps.ParentSnapshot(
        columns=cols, m1_msun=m1, m1_source=np.full(n, "MSC"), atmosphere_logg=np.full(n, 4.4),
        truth_parallax_mas=1000.0 / r, is_giant=giant, flags={"all": np.ones(n, bool)}, usable=np.ones(n, bool),
        meta={"parent_h5_sha256": "x"}, scale_to_full=1.0, path=Path("."),
    )


@pytest.mark.physics
def test_zero_point_fit_recovers_known_values() -> None:
    import yaml

    from darkhunter_pop.malmquist import MalmquistConfig

    raw = yaml.safe_load(Path("config/population/malmquist_decided.yaml").read_text())
    mcfg = MalmquistConfig.model_validate(raw["malmquist"] | {"provisional_sigma_log_m1_dex": 0.0})
    ext = ps.ParentExtinctionConfig.model_validate(raw["extinction"])
    fit = ps.ZeroPointFitConfig.model_validate(raw["zero_point_fit"] | {"n_bootstrap": 20})
    parent = _synthetic_parent_for_malmquist()
    res = ps.fit_mg_zero_point(parent, np.zeros(parent.n_rows), mcfg, fit, ext)
    assert res.zero_point_mag == pytest.approx(0.3, abs=4 * res.zero_point_err_mag + 0.005)
    assert res.sigma_int_mag == pytest.approx(0.2, abs=4 * res.sigma_int_err_mag + 0.005)
    assert 0 < res.zero_point_err_mag < 0.02
    assert res.n_stars < parent.n_rows  # RUWE >= 1.4 and giants excluded


@pytest.mark.api
def test_malmquist_weight_assembly() -> None:
    import yaml

    from darkhunter_pop.malmquist import MalmquistConfig

    raw = yaml.safe_load(Path("config/population/malmquist_decided.yaml").read_text())
    small = raw["malmquist"] | {"grid": raw["malmquist"]["grid"] | {"n_m1": 8, "n_log_q": 16, "n_log_p": 16, "n_log_f": 40},
                                "provisional_sigma_int_mag": 0.2, "provisional_mg_zero_point_mag": 0.3}
    mcfg = MalmquistConfig.model_validate(small)
    ext = ps.ParentExtinctionConfig.model_validate(raw["extinction"])
    parent = _synthetic_parent_for_malmquist(n=300)
    prop = ps.load_proposal_set_fragment(RESTART).proposal.model_copy(update={"n_draws": 400})
    tgt = ps.load_proposal_set_fragment(RESTART).target_mds17
    truth = ps.sample_proposal(parent, prop)
    lw, counts = ps.malmquist_log_weight(truth, parent, tgt, mcfg, np.zeros(parent.n_rows), ext)
    assert lw.shape == (400,) and np.all(np.isfinite(lw))
    giant_draws = parent.is_giant[truth["parent_row"]]
    assert np.all(lw[giant_draws] == 0.0)  # MP-Q28 hook: giants unit weight
    assert counts["unit_weight_giant"] == int(giant_draws.sum())
    with pytest.raises(ValueError, match="MP-Q30"):
        ps.malmquist_log_weight(truth, parent, tgt, mcfg.model_copy(update={"provisional_distance_marginalization": "split_normal_mu"}),
                                np.zeros(parent.n_rows), ext)


@pytest.mark.unit
def test_proposal_draws_go_through_epoch_model_run_cascade(monkeypatch: pytest.MonkeyPatch) -> None:
    """#418 / #421 / PR #422: with the epoch model on, every draw runs through
    ``epoch_model.run_cascade`` (epochs + per-CCD noise + RUWE normalization), with G and the
    draw's galactic (l, b) in the source context and the draw-keyed Generators; the runner
    builds that setup from ``dr3.epoch_model``."""
    import importlib.util

    from darkhunter_pop import epoch_model as em
    from darkhunter_pop.config_loader import load_config

    spec = importlib.util.spec_from_file_location("run_proposal_pilot", Path("scripts/run_proposal_pilot.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    frag = ps.load_proposal_set_fragment(RESTART)
    setup = mod.build_epoch_setup(load_config(), frag.proposal)
    assert isinstance(setup, ps.EpochSetup) and setup.config.enabled
    seen: dict[str, object] = {}

    def fake_run_cascade(gm, cf, predict, config, source, **kw):  # type: ignore[no-untyped-def]
        seen.update(source=source, config=config, **kw)
        return em.CascadeRun(cascade=[0.0] * ps.CASCADE_VECTOR_LENGTH, n_obs=0, n_transits=0, n_visibility_periods=0, ruwe_scale=1.0)

    monkeypatch.setattr(em, "run_cascade", fake_run_cascade)
    monkeypatch.setattr(ps, "seeded_global_rng", lambda seeds, cf: __import__("contextlib").nullcontext())
    draw = {"generation": 20, "draw_index": 3, "ra_deg": 10.0, "dec_deg": -5.0, "parallax_mas": 2.0, "pmra_masyr": 0.0,
            "pmdec_masyr": 0.0, "m1_msun": 1.0, "m2_msun": 0.5, "period_days": 300.0, "Tp_days": 10.0, "eccentricity": 0.1,
            "Omega_rad": 1.0, "inc_deg": 60.0, "omega_rad": 2.0, "phot_g_mean_mag": 15.0, "flux_ratio": 0.01,
            "l_deg": 120.0, "b_deg": -30.0}
    cuts = load_config().active_dr().selection_function_astrometric.orbital_solution_cuts
    rec = ps.simulate_one(draw, gaiamock=_FakeGaiamock(), c_funcs=None, cfg=frag.proposal, cuts=cuts, epoch=setup)
    src = seen["source"]
    assert (src.g_mag, src.l_deg, src.b_deg) == (15.0, 120.0, -30.0)  # type: ignore[attr-defined]
    assert src.beta_deg is not None and np.isfinite(src.beta_deg)  # type: ignore[attr-defined]  # #442
    assert seen["config"] is setup.config
    assert seen["ruwe_min"] == frag.proposal.ruwe_min
    assert rec["solution_type"] is not None and not rec["accepted_orbital"]


@pytest.mark.unit
def test_epoch_source_context_feeds_visibility_period_loss() -> None:
    """#442: with dr3.epoch_model.visibility_period_loss on, the draw's source context must
    carry beta and b, or every simulate_one draw raises in thin_gost_mask."""
    from darkhunter_pop import epoch_model as em
    from darkhunter_pop.config_loader import load_config

    cfg = em.epoch_model_config_from_mapping(load_config().dr3.epoch_model)
    assert cfg.vp_loss is not None
    for draw in ({"ra_deg": 10.0, "dec_deg": -5.0, "phot_g_mean_mag": 18.0},
                 {"ra_deg": 266.4, "dec_deg": -28.9, "phot_g_mean_mag": 12.0, "l_deg": np.nan, "b_deg": None}):
        src = ps.epoch_source_context(draw)
        assert src.beta_deg is not None and src.b_deg is not None and src.l_deg is not None
        jd = 2457000.0 + np.repeat(np.arange(0.0, 900.0, 3.0), 10) + np.tile(np.arange(10) * 5.0 / 86400.0, 300)
        keep = em.thin_gost_mask(jd, cfg, em.gap_intervals_jd(cfg), g_mag=src.g_mag, rng=np.random.default_rng(0),
                                 l_deg=src.l_deg, b_deg=src.b_deg, beta_deg=src.beta_deg)
        assert keep.shape == jd.shape and keep.any()


# ---------------------------------------------------------------------------
# #391 efficiency (2026-10-08): MdS17-shaped (q, P), evolved top-up, MIST-centred flux
# ---------------------------------------------------------------------------

TUNE2 = "config/population/proposal_set_restart_tune2.yaml"


def _tune2() -> ps.ProposalConfig:
    return ps.load_proposal_set_fragment(TUNE2).proposal


@pytest.mark.physics
def test_m2_p_shape_table_normalized_and_joint_density_normalizes() -> None:
    cfg = _tune2()
    tab = ps.m2_period_shape_table(cfg.m2_p_shape)
    np.testing.assert_allclose(tab.prob.sum(axis=(1, 2)), 1.0, rtol=1e-12)
    for m1 in (0.5, 1.0, 2.5):
        lm1 = np.log10(m1)
        lm2 = np.linspace(-2.0, 2.0, 1601)
        lp = np.linspace(-1.0, 9.0, 2001)
        LM2, LP = np.meshgrid(lm2, lp, indexing="ij")
        d = np.exp(ps.log_q_m2_p_joint(LM2, LP, np.full_like(LM2, lm1), cfg))
        total = np.trapz(np.trapz(d, lp, axis=1), lm2)
        assert total == pytest.approx(1.0, abs=5e-3)


@pytest.mark.physics
def test_m2_p_shape_weights_bounded() -> None:
    from darkhunter_pop import malmquist as mq

    cfg = _tune2()
    tgt = ps.load_proposal_set_fragment(TUNE2).target_mds17
    worst = 0.0
    worst_box = 0.0
    for m1 in (0.6, 0.9, 1.3, 3.0):
        lq = np.linspace(-1.0, 0.0, 401)[1:-1]
        lp = np.linspace(0.21, 7.99, 401)
        LQ, LP = np.meshgrid(lq, lp, indexing="ij")
        target = mq.mds17_luminous_m2_p_intensity(m1, LQ, LP, tgt)
        target = target / np.trapz(np.trapz(target, lp, axis=1), lq)  # normalized shape
        q = np.exp(ps.log_q_m2_p_joint(np.log10(m1) + LQ, LP, np.full_like(LQ, np.log10(m1)), cfg))
        r = target / q
        box = (LP >= cfg.m2_p_shape.log_p_min) & (LP <= cfg.m2_p_shape.log_p_max)
        worst_box = max(worst_box, float(np.max(r[box])))
        worst = max(worst, float(np.max(r)))
    # Bounded everywhere by target max / defensive floor (finite variance); inside the shape
    # box the ratio is ~ 1 / weight up to cell discretization (the twin step at q = 0.95).
    assert np.isfinite(worst) and worst < 500.0
    assert worst_box < 5.0


@pytest.mark.unit
def test_parent_evolved_component() -> None:
    plx = np.array([1.0, 2.0, 5.0, 0.5])
    usable = np.array([True, True, True, False])
    evolved = np.array([False, True, False, True])
    base_cfg = ps.ParentProposalConfig(uniform_fraction=0.3, parallax_power=0.75, parallax_cap_mas=10.0)
    old = ps.parent_proposal_probabilities(plx, usable, base_cfg)
    same = ps.parent_proposal_probabilities(plx, usable, base_cfg, evolved)
    np.testing.assert_array_equal(old, same)  # evolved_weight 0 is bit-identical
    cfg = base_cfg.model_copy(update={"evolved_weight": 0.1})
    q = ps.parent_proposal_probabilities(plx, usable, cfg, evolved)
    assert q.sum() == pytest.approx(1.0)
    assert q[3] == 0.0  # unusable evolved row never drawn
    assert q[1] == pytest.approx(0.9 * old[1] + 0.1)
    assert np.all(q[usable] >= 0.9 * 0.3 / usable.sum())  # floor keeps 1/q bounded
    with pytest.raises(ValueError):
        ps.parent_proposal_probabilities(plx, usable, cfg)


@pytest.mark.unit
def test_mist_centre_requires_config() -> None:
    fc = _tune2().flux
    with pytest.raises(ValueError, match="PipelineConfig"):
        ps.proposal_relation_centre(np.array([1.0]), np.array([0.5]), np.array([0]), None, fc)


@pytest.mark.unit
def test_shape_sampler_reproducible_and_logq_consistent(fragment: ps.ProposalSetFragment) -> None:
    parent = _parent()
    shape = _tune2().m2_p_shape
    cfg = fragment.proposal.model_copy(update={"n_draws": 600, "m2_p_shape": shape})
    a = ps.sample_proposal(parent, cfg)
    b = ps.sample_proposal(parent, cfg)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert "log_q_m2_p" in a and "log_q_log_m2" not in a
    np.testing.assert_allclose(ps.log_q_total_for(a, parent, cfg), a["log_q_total"])
    assert np.all(np.isfinite(a["log_q_total"]))
    # draws inside the shape box are concentrated there (most come from the shape component)
    lp = np.log10(a["period_days"])
    assert np.mean((lp >= 1.0) & (lp <= 4.0)) > 0.75
    # the main stream is untouched by the shape component (its own Generator)
    old = ps.sample_proposal(parent, fragment.proposal.model_copy(update={"n_draws": 600}))
    np.testing.assert_array_equal(old["parent_row"], a["parent_row"])


# ---------------------------------------------------------------------------
# #391 option (i): evolved-row flux centred on the dark-companion deblended M2
# ---------------------------------------------------------------------------

EVO27 = "config/population/proposal_set_restart_evolved27.yaml"


class _EvoParent:
    """Minimal isochrone-mode stand-in: two rows, the second CMD-evolved."""

    def __init__(self) -> None:
        self.is_giant = np.array([False, True])
        self.cmd = {"mg0": np.array([4.0, 1.0])}


@pytest.mark.unit
def test_deblended_evolved_centre_uses_q_times_dark_m1() -> None:
    from darkhunter_pop.giants import evolved_log10_flux_ratio

    fc = ps.load_proposal_set_fragment(EVO27).proposal.flux.model_copy(update={"relation": "janssens2022"})
    par = _EvoParent()
    rows = np.array([0, 1, 1])
    m1 = np.array([1.0, 1.4, 1.4])
    m2 = np.array([0.5, 0.7, 0.7])
    dark = np.array([np.nan, 0.9, np.nan])  # third: no posterior -> falls back to proposal M2
    c = ps.proposal_relation_centre(m1, m2, rows, par, fc, None, dark)
    assert c[0] == pytest.approx(float(ps.relation_log10_flux_ratio(1.0, 0.5)))
    assert c[1] == pytest.approx(float(evolved_log10_flux_ratio(0.5 * 0.9, 1.0)))
    assert c[2] == pytest.approx(float(evolved_log10_flux_ratio(0.7, 1.0)))
    with pytest.raises(ValueError, match="m1_deblend_dark"):
        ps.proposal_relation_centre(m1, m2, rows, par, fc, None, None)
    sig, wr = ps.flux_width_and_weight(rows, par, fc)
    np.testing.assert_allclose(sig, [fc.relation_sigma_dex, 0.15, 0.15])
    np.testing.assert_allclose(wr, [fc.relation_weight, 0.8, 0.8])


@pytest.mark.physics
def test_evolved_flux_weights_bounded_and_finite_variance() -> None:
    # Target N(c_t, 0.1) per dex f vs the evolved proposal 0.8 N(c_q, 0.15) + 0.2 U[-7, 0.5]
    # for any centre offset: the ratio is bounded by the defensive floor, so E_q[w^2] < inf;
    # with the centre matched the IS mean recovers 1 with small error.
    fc = ps.load_proposal_set_fragment(EVO27).proposal.flux
    lf = np.linspace(fc.log_f_min, fc.log_f_max, 200001)
    for offset in (0.0, 0.3, 1.5):
        q = np.exp(ps.log_q_flux(lf, np.zeros_like(lf, bool), np.ones_like(lf), np.full_like(lf, 0.5), fc,
                                 centre=np.full_like(lf, -2.0 + offset), sigma=np.full_like(lf, 0.15),
                                 relation_weight=np.full_like(lf, 0.8))) / (1.0 - fc.dark_fraction)
        p = np.exp(-0.5 * ((lf + 2.0) / 0.1) ** 2) / (0.1 * np.sqrt(2 * np.pi))
        r = p / q
        bound = (1.0 / (0.1 * np.sqrt(2 * np.pi))) / (0.2 / (fc.log_f_max - fc.log_f_min))
        assert np.max(r) <= bound * (1 + 1e-9)
        second_moment = np.trapz(p * r, lf)  # E_q[w^2] = ∫ p^2 / q
        assert np.isfinite(second_moment)
        if offset == 0.0:
            assert second_moment < 2.0  # matched centre: ESS fraction > 50%


@pytest.mark.unit
def test_evolved_only_parent_component() -> None:
    plx = np.array([1.0, 2.0, 5.0])
    usable = np.array([True, True, True])
    evolved = np.array([False, True, True])
    cfg = ps.ParentProposalConfig(uniform_fraction=0.3, parallax_power=0.75, parallax_cap_mas=10.0, evolved_weight=1.0)
    q = ps.parent_proposal_probabilities(plx, usable, cfg, evolved)
    np.testing.assert_allclose(q, [0.0, 0.5, 0.5])


@pytest.mark.unit
def test_evolved27_config_loads() -> None:
    p = ps.load_proposal_set_fragment(EVO27).proposal
    assert p.generation == 27 and p.parent.evolved_weight == 1.0
    assert p.flux.evolved_rows_centre == "evolved_relation_deblended"


# ---------------------------------------------------------------------------
# #391 rung 3: re-centred shape component (generation 28)
# ---------------------------------------------------------------------------

RECENTRED28 = "config/population/proposal_set_restart_recentred28.yaml"


@pytest.mark.unit
def test_shape_modifier_defaults_are_identity() -> None:
    cfg = _tune2().m2_p_shape
    q = np.linspace(0.1, 1.0, 50)
    lp = np.linspace(1.0, 4.0, 50)
    np.testing.assert_array_equal(ps.m2_p_shape_modifier(q, lp, cfg), np.ones(50))


@pytest.mark.physics
def test_recentred_shape_table_follows_modifier_and_normalizes() -> None:
    rec = ps.load_proposal_set_fragment(RECENTRED28).proposal
    base = rec.m2_p_shape.model_copy(update={"dgamma_q": 0.0, "ln_f_twin": 0.0, "gamma_p": 0.0, "ln_long_p": 0.0})
    t1, t0 = ps.m2_period_shape_table(rec.m2_p_shape), ps.m2_period_shape_table(base)
    np.testing.assert_allclose(t1.prob.sum(axis=(1, 2)), 1.0, rtol=1e-12)
    lq_c = 0.5 * (t1.lq_edges[1:] + t1.lq_edges[:-1])
    lp_c = 0.5 * (t1.lp_edges[1:] + t1.lp_edges[:-1])
    LQ, LP = np.meshgrid(lq_c, lp_c, indexing="ij")
    mod = ps.m2_p_shape_modifier(10.0**LQ, LP, rec.m2_p_shape)
    for k in (0, 10, 30):
        expect = t0.prob[k] * mod
        np.testing.assert_allclose(t1.prob[k], expect / expect.sum(), rtol=1e-10, atol=1e-15)
    for m1 in (0.5, 1.0, 2.5):  # the full joint (shape + defensive) still integrates to one
        lm1 = np.log10(m1)
        lm2 = np.linspace(-2.0, 2.0, 1601)
        lp = np.linspace(-1.0, 9.0, 2001)
        LM2, LPP = np.meshgrid(lm2, lp, indexing="ij")
        d = np.exp(ps.log_q_m2_p_joint(LM2, LPP, np.full_like(LM2, lm1), rec))
        assert np.trapz(np.trapz(d, lp, axis=1), lm2) == pytest.approx(1.0, abs=5e-3)


@pytest.mark.physics
def test_recentred_proposal_weights_bounded_for_old_and_refit_targets() -> None:
    """Gen-28 q(log q, log P) against both the MdS17 target and the rung-3 best-fit target."""
    from darkhunter_pop import malmquist as mq

    frag = ps.load_proposal_set_fragment(RECENTRED28)
    cfg, tgt = frag.proposal, frag.target_mds17
    ps.check_eccentricity_bounded(cfg.eccentricity, tgt.provisional_eta_floor)
    for modified in (False, True):
        worst = 0.0
        for m1 in (0.6, 0.9, 1.3, 3.0):
            lq = np.linspace(-1.0, 0.0, 401)[1:-1]
            lp = np.linspace(0.21, 7.99, 401)
            LQ, LP = np.meshgrid(lq, lp, indexing="ij")
            target = mq.mds17_luminous_m2_p_intensity(m1, LQ, LP, tgt)
            if modified:
                target = target * ps.m2_p_shape_modifier(10.0**LQ, LP, cfg.m2_p_shape)
            target = target / np.trapz(np.trapz(target, lp, axis=1), lq)
            q = np.exp(ps.log_q_m2_p_joint(np.log10(m1) + LQ, LP, np.full_like(LQ, np.log10(m1)), cfg))
            worst = max(worst, float(np.max(target / q)))
        assert np.isfinite(worst) and worst < 500.0


# ---------------------------------------------------------------------------
# #391 rung 3 staged top-up: targeted parent component (generation 29)
# ---------------------------------------------------------------------------

TOPUP29 = "config/population/proposal_set_restart_topup29.yaml"


def _parent_rows(n: int = 5000, seed: int = 3) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    plx = rng.uniform(0.2, 5.0, n)
    g = rng.uniform(6.0, 19.0, n)
    m1 = rng.uniform(0.2, 2.5, n)
    usable = rng.uniform(size=n) > 0.1
    evolved = rng.uniform(size=n) > 0.9
    return plx, g, m1, usable, evolved


@pytest.mark.unit
def test_parent_target_component_zero_weight_is_identity() -> None:
    plx, g, m1, usable, evolved = _parent_rows()
    cfg = ps.load_proposal_set_fragment(TOPUP29).proposal.parent
    off = cfg.model_copy(update={"target_weight": 0.0})
    base = ps.parent_proposal_probabilities(plx, usable, off, evolved)
    np.testing.assert_array_equal(base, ps.parent_proposal_probabilities(plx, usable, off, evolved, g, m1))


@pytest.mark.unit
def test_parent_target_component_normalized_bounded_and_targeted() -> None:
    plx, g, m1, usable, evolved = _parent_rows()
    cfg = ps.load_proposal_set_fragment(TOPUP29).proposal.parent
    base = ps.parent_proposal_probabilities(plx, usable, cfg.model_copy(update={"target_weight": 0.0}), evolved)
    q = ps.parent_proposal_probabilities(plx, usable, cfg, evolved, g, m1)
    assert q.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(q[~usable] == 0.0)
    # any target that the old mixture covered stays covered: p/q <= (p/base) / (1 - eps)
    ok = base > 0
    assert np.max(base[ok] / q[ok]) <= 1.0 / (1.0 - cfg.target_weight) + 1e-12
    (g0, g1), (p0, p1) = cfg.target_g_range, cfg.target_parallax_range_mas
    box = usable & (g >= g0) & (g < g1) & (plx >= p0) & (plx < p1)
    assert q[box].sum() > base[box].sum() + 0.5 * cfg.target_weight
    hi = box & (m1 > 1.6)
    lo = box & (m1 < 0.8)
    assert np.mean(q[hi] - (1 - cfg.target_weight) * base[hi]) > np.mean(q[lo] - (1 - cfg.target_weight) * base[lo])


@pytest.mark.unit
def test_parent_target_requires_rows_and_fields() -> None:
    plx, g, m1, usable, evolved = _parent_rows()
    cfg = ps.load_proposal_set_fragment(TOPUP29).proposal.parent
    with pytest.raises(ValueError):
        ps.parent_proposal_probabilities(plx, usable, cfg, evolved)
    with pytest.raises(ValueError):
        cfg.model_copy(update={"target_g_range": None}).model_validate(cfg.model_dump() | {"target_g_range": None})


@pytest.mark.physics
def test_topup29_shape_and_ecc_bounded() -> None:
    from darkhunter_pop import malmquist as mq

    frag = ps.load_proposal_set_fragment(TOPUP29)
    cfg, tgt = frag.proposal, frag.target_mds17
    ps.check_eccentricity_bounded(cfg.eccentricity, tgt.provisional_eta_floor)
    for m1 in (0.5, 1.0, 2.5):  # joint (shape + defensive) density normalizes
        lm1 = np.log10(m1)
        lm2 = np.linspace(-2.0, 2.0, 1601)
        lp = np.linspace(-1.0, 9.0, 2001)
        LM2, LPP = np.meshgrid(lm2, lp, indexing="ij")
        d = np.exp(ps.log_q_m2_p_joint(LM2, LPP, np.full_like(LM2, lm1), cfg))
        assert np.trapz(np.trapz(d, lp, axis=1), lm2) == pytest.approx(1.0, abs=5e-3)
    for modified in (False, True):
        worst = 0.0
        for m1 in (0.6, 0.9, 1.3, 3.0):
            lq = np.linspace(-1.0, 0.0, 401)[1:-1]
            lp = np.linspace(0.21, 7.99, 401)
            LQ, LP = np.meshgrid(lq, lp, indexing="ij")
            target = mq.mds17_luminous_m2_p_intensity(m1, LQ, LP, tgt)
            if modified:
                target = target * ps.m2_p_shape_modifier(10.0**LQ, LP, cfg.m2_p_shape)
            target = target / np.trapz(np.trapz(target, lp, axis=1), lq)
            q = np.exp(ps.log_q_m2_p_joint(np.log10(m1) + LQ, LP, np.full_like(LQ, np.log10(m1)), cfg))
            worst = max(worst, float(np.max(target / q)))
        assert np.isfinite(worst) and worst < 500.0
