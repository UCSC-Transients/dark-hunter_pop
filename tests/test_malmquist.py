"""Magnitude-limit (Malmquist / Öpik) conditioning weight and its closed-loop proof (#405).

docs/MOCK_POPULATION_SPEC.md §9. Unit tests for the weight's pieces, an analytic Öpik-limit
check, and the closed loop (small in the required gate, large under ``slow``). No gaiamock.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import malmquist as mq
from darkhunter_pop import moe_distefano as mds
from darkhunter_pop import proposal_set as ps

DECIDED = "config/population/proposal_set_decided_tune.yaml"
MCFG = "config/population/malmquist.yaml"


@pytest.fixture(scope="module")
def frag() -> ps.ProposalSetFragment:
    return ps.load_proposal_set_fragment(DECIDED)


@pytest.fixture(scope="module")
def mcfg() -> mq.MalmquistConfig:
    return mq.load_malmquist_config(MCFG)


@pytest.fixture(scope="module")
def grid(frag: ps.ProposalSetFragment, mcfg: mq.MalmquistConfig) -> mq.FluxMarginalGrid:
    return mq.build_flux_marginal(frag.target_mds17, mcfg.grid)


@pytest.mark.unit
def test_config_loads_with_provisional_keys(mcfg: mq.MalmquistConfig) -> None:
    open_keys = [k for k in mq.MalmquistConfig.model_fields if k.startswith("provisional_")]
    assert len(open_keys) == 7  # MP-Q25 (x3), Q27, Q28, Q30, Q31


@pytest.mark.unit
def test_sigma_mu_and_luminosity_excess() -> None:
    s = mq.sigma_mu_from_quantiles(900.0, 1000.0, 1100.0)
    assert s == pytest.approx(5.0 / math.log(10.0) * 0.1)
    m1 = np.array([0.5, 1.0, 2.0])
    d = np.array([100.0, 1000.0, 3000.0])
    g = ps.janssens_absolute_g(m1) + mq.distance_modulus(d)
    np.testing.assert_allclose(mq.luminosity_excess(g, d, 0.0, m1, zero_point_mag=0.0), 0.0, atol=1e-12)
    # A twin (f = 1) is 2.5 log10 2 = 0.753 mag over-luminous; its light likelihood peaks there.
    dm = -2.5 * math.log10(2.0)
    assert mq.log_light_likelihood(dm, 0.0, 0.1) > mq.log_light_likelihood(dm, -np.inf, 0.1) + 20


@pytest.mark.unit
def test_mg_slope_matches_janssens_segment() -> None:
    slope = mq.mg_slope_per_dex(np.array([1.0]), 0.001)
    seg = next(s for s in ps.segments_from_table() if s.m_low < 1.0 < s.m_up)
    assert slope[0] == pytest.approx(seg.a, rel=1e-6)


@pytest.mark.physics
def test_m2_p_intensity_is_the_target_without_e_and_f(frag: ps.ProposalSetFragment) -> None:
    """Identity: target log λ = log λ_{q,P} + log p_e + log p_f (so Z cannot drift from it)."""
    t = frag.target_mds17
    table = mds.load_mds17_table(t.table_path)
    rng = np.random.default_rng(1)
    n = 4000
    m1 = 10 ** rng.uniform(-0.5, 0.6, n)
    lq = rng.uniform(-1.0, 0.0, n)
    lp = rng.uniform(0.2, 8.0, n)
    p = 10**lp
    e = np.where(p <= table.circular_period_days, 0.0, rng.uniform(0, 1, n) * mds.e_max(p, table))
    rel = ps.relation_log10_flux_ratio(m1, m1 * 10**lq)
    lf = rel + rng.normal(0, 0.1, n)
    truth = {"m1_msun": m1, "m2_msun": m1 * 10**lq, "period_days": p, "eccentricity": e, "log10_flux_ratio": lf, "is_dark": np.zeros(n, bool)}
    lt = ps.mds17_luminous_log_intensity(truth, t)
    m1s = np.clip(m1, *table.m1_range)
    pe = np.where(
        mds.is_circular(p, table),
        1.0,
        mds.e_density(e, m1s, p, table, m1_interpolation=t.provisional_m1_interpolation, eta_floor=t.provisional_eta_floor),
    )
    pf = np.exp(-0.5 * ((lf - rel) / t.flux_sigma_dex) ** 2) / (t.flux_sigma_dex * math.sqrt(2 * math.pi))
    mine = mq.mds17_luminous_m2_p_intensity(m1, lq, lp, t) * pe * pf
    ok = np.isfinite(lt) & (mine > 0)
    assert ok.sum() > 0.9 * n
    np.testing.assert_allclose(lt[ok], np.log(mine[ok]), rtol=1e-10, atol=1e-10)


@pytest.mark.physics
def test_flux_marginal_converged_and_mds17_check_value(frag: ps.ProposalSetFragment, mcfg: mq.MalmquistConfig, grid: mq.FluxMarginalGrid) -> None:
    fine = mq.build_flux_marginal(frag.target_mds17, mcfg.grid.model_copy(update={"n_log_q": 400, "n_log_p": 500, "n_m1": 4, "log_m1_min": -0.4, "log_m1_max": 0.3}))
    m = 10**fine.log_m1
    np.testing.assert_allclose(grid.interpolate(m)[1], fine.f_lum, rtol=2e-3)
    np.testing.assert_allclose(grid.lam_f.sum(axis=1), grid.f_lum, rtol=1e-12)
    # MdS17 §9.4 check value: f_mult;q>0.3 = 0.36 at 1 Msun. Here q > 0.1, so it is larger.
    assert 0.36 < float(grid.interpolate(np.array([1.0]))[1][0]) < 0.6


@pytest.mark.physics
def test_weights_normalize_and_reduce_to_naive(grid: mq.FluxMarginalGrid) -> None:
    m1 = np.full(5, 1.0)
    dm = np.array([0.3, 0.0, -0.3, -0.75, -0.2])
    sig = np.array([0.15, 0.15, 0.15, 0.15, 1e4])
    lp0, over = mq.log_no_companion_probability(m1, dm, sig, grid)
    assert not over.any()
    lam, f_lum = grid.interpolate(m1)
    nf = grid.log_f.size
    lw = mq.log_conditioning_factor(
        np.tile(grid.log_f, 5), np.repeat(m1, nf), np.repeat(dm, nf), np.repeat(sig, nf), grid
    ).reshape(5, nf)
    total = np.exp(lp0) + np.sum(lam * np.exp(lw), axis=1)
    np.testing.assert_allclose(total, 1.0, rtol=1e-9)  # p(∅|o) + ∫ π W = 1
    # σ → ∞: G carries no information, W → 1, p(∅) → 1 − F (the naive draw).
    assert np.exp(lp0[-1]) == pytest.approx(1.0 - f_lum[-1], rel=1e-4)
    # More over-luminous rows are more likely binaries.
    assert np.all(np.diff(np.exp(lp0[:4])) < 0)


@pytest.mark.physics
def test_opik_limit_in_a_homogeneous_euclidean_volume() -> None:
    """σ → 0, pure G limit, no floor: the parent's twin fraction is the (1+f)^{3/2} boost,
    and the per-row W reproduces it, while the naive draw stays at the volume-limited value."""
    rng = np.random.default_rng(7)
    n = 400_000
    pi_twin, mg, glim, sig = 0.2, 5.0, 15.0, 0.05
    r = 4000.0 * rng.uniform(size=n) ** (1 / 3)
    twin = rng.uniform(size=n) < pi_twin
    g = mg + sig * rng.standard_normal(n) - 2.5 * np.log10(np.where(twin, 2.0, 1.0)) + mq.distance_modulus(r)
    sel = g < glim
    boost = 2.0**1.5
    expect = pi_twin * boost / (1 - pi_twin + pi_twin * boost)
    assert twin[sel].mean() == pytest.approx(expect, abs=0.01)
    lam = np.zeros((2, 3))
    lam[:, 2] = pi_twin  # all companion mass at log10 f = 0 (f = 1)
    toy = mq.FluxMarginalGrid(log_m1=np.array([-1.0, 1.0]), log_f=np.array([-10.0, -5.0, 0.0]), lam_f=lam, f_lum=np.full(2, pi_twin))
    dm = g[sel] - mq.distance_modulus(r[sel]) - mg
    lp0, _ = mq.log_no_companion_probability(np.ones(sel.sum()), dm, np.full(sel.sum(), sig), toy)
    assert 1.0 - np.exp(lp0).mean() == pytest.approx(twin[sel].mean(), abs=0.005)


@pytest.mark.physics
def test_selection_volume_euclidean_ratio() -> None:
    rng = np.random.default_rng(3)
    d = 5000.0 * rng.uniform(size=100_000) ** (1 / 3)
    tab_m = np.linspace(0, 10, 201)

    def p_sel(gm: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return (gm < 15.0).astype(float)

    v = mq.tabulate_selection_volume(tab_m, d, p_sel)
    ratio = mq.selection_volume(5.0 - 2.5 * math.log10(2), tab_m, v, sigma_int_mag=0.0) / mq.selection_volume(5.0, tab_m, v, sigma_int_mag=0.0)
    assert ratio == pytest.approx(2.0**1.5, rel=0.03)


def _bj_parent(n: int = 40) -> ps.ParentSnapshot:
    rng = np.random.default_rng(0)
    r = rng.uniform(100, 2000, n)
    cols = {
        "source_id": np.arange(n), "ra": np.zeros(n), "dec": np.zeros(n), "parallax": 1000 / r,
        "pmra": np.zeros(n), "pmdec": np.zeros(n), "phot_g_mean_mag": rng.uniform(10, 18, n),
        "r_lo_geo": 0.9 * r, "r_med_geo": r, "r_hi_geo": 1.1 * r,
    }
    giant = np.zeros(n, bool)
    giant[:5] = True
    return ps.ParentSnapshot(columns=cols, m1_msun=rng.uniform(0.7, 2, n), m1_source=np.full(n, "MSC"),
                             atmosphere_logg=np.full(n, 4.4), truth_parallax_mas=1000 / r, is_giant=giant,
                             flags={"all": np.ones(n, bool)}, usable=np.ones(n, bool), meta={}, scale_to_full=1.0, path=Path("."))


@pytest.mark.api
def test_hook_for_proposal_set_draws(frag: ps.ProposalSetFragment, mcfg: mq.MalmquistConfig, grid: mq.FluxMarginalGrid) -> None:
    parent = _bj_parent()
    prop = frag.proposal.model_copy(update={"n_draws": 500})
    truth = ps.sample_proposal(parent, prop)
    rows = mq.row_conditioning(parent, mcfg, a_g_mag=np.zeros(parent.n_rows), sigma_a_mag=0.0)
    lw = mq.log_weight_for_draws(truth, rows, grid, mcfg)
    assert lw.shape == truth["draw_index"].shape and np.all(np.isfinite(lw))
    giant_draws = rows.is_giant[truth["parent_row"]]
    assert np.all(lw[giant_draws] == 0.0)  # MP-Q28 provisional: unit weight
    w = ps.importance_weights(ps.mds17_luminous_log_intensity(truth, frag.target_mds17) + lw, [truth["log_q_total"]], [500], scale_to_full=1.0)
    assert np.all(w >= 0)
    bad = _bj_parent()
    cols = dict(bad.columns)
    del cols["r_lo_geo"]
    with pytest.raises(ValueError, match="r_lo_geo"):
        mq.row_conditioning(ps.ParentSnapshot(**{**bad.__dict__, "columns": cols}), mcfg, a_g_mag=0.0, sigma_a_mag=0.0)


# ---------------------------------------------------------------------------
# Closed loop (spec §9.6)
# ---------------------------------------------------------------------------


def _check_closed_loop(size: str, naive_min_pull: float) -> None:
    from darkhunter_pop import malmquist_closed_loop as cl

    res, _ = cl.run_closed_loop(size)
    t = res.parent_binary_total
    assert abs(t["closure_corrected"] - 1.0) < 0.01
    corr = [res.parent_binary_by_g.pull_corrected, res.parent_binary_by_m1.pull_corrected, res.alpha0.pull_corrected]
    corr += [c.pull_corrected for c in res.parent_shapes.values()]
    assert abs(t["pull_corrected"]) < 3.0
    assert max(float(np.max(np.abs(p))) for p in corr) < 3.5, "with W the mock must match the parent"
    naive = [c.pull_naive for c in res.parent_shapes.values()]
    assert max(float(np.max(np.abs(p))) for p in naive) > naive_min_pull, "without W the test must have power"
    v = res.volume
    zc = (v.corrected_fraction - v.truth_fraction) / np.hypot(v.truth_fraction_err, v.corrected_fraction_err)
    zn = (v.naive_fraction - v.truth_fraction) / np.hypot(v.truth_fraction_err, v.naive_fraction_err)
    assert np.max(np.abs(zc)) < 3.5
    assert np.max(np.abs(zn)) > naive_min_pull
    q = v.shapes["q"]
    zq = (q["corrected"] - q["truth"]) / np.hypot(q["truth_err"], q["corrected_err"])
    assert np.max(np.abs(zq)) < 3.5


@pytest.mark.physics
def test_closed_loop_small() -> None:
    _check_closed_loop("small", naive_min_pull=5.0)


@pytest.mark.slow
def test_closed_loop_large() -> None:
    _check_closed_loop("large", naive_min_pull=10.0)
