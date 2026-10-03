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
