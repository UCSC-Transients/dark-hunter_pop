"""MIST isochrone M1 and the 2-D CMD Malmquist weight (#418; spec §11).

Unit tests run on toy grids (no MIST files). Tests that need the real MIST grid at
``isochrone_mass.mist_root`` skip when it is absent (CI); the closed loop with MIST photometry
is ``slow`` (docs/gate418 has the numbers).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import malmquist as mq
from darkhunter_pop import malmquist_cmd as mc
from darkhunter_pop import proposal_set as ps

DECIDED = "config/population/proposal_set_decided_tune.yaml"
CMCFG = "config/population/malmquist_cmd.yaml"


def _mist_available() -> bool:
    from darkhunter_pop.config_loader import load_config

    root = load_config().isochrone_mass.mist_root
    return root is not None and (Path(root).expanduser() / "mist").is_dir()


needs_mist = pytest.mark.skipif(not _mist_available(), reason="MIST grid not on this host (isochrone_mass.mist_root)")


# ---------------------------------------------------------------------------
# isochrone_mass on toy inputs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feh_tokens_match_mist_file_names() -> None:
    assert im._feh_token(-0.25, "iso") == "m0.25"
    assert im._feh_token(0.0, "iso") == "p0.00"
    assert im._feh_token(0.5, "bc") == "p050"
    assert im._feh_token(-1.75, "bc") == "m175"


@pytest.mark.unit
def test_imf_is_continuous_with_the_configured_slopes() -> None:
    cfg = im.ImfPriorConfig()
    m = np.array([0.08 - 1e-7, 0.08 + 1e-7, 0.5 - 1e-7, 0.5 + 1e-7, 1.0, 2.0])
    xi = im.imf_density(m, cfg)
    assert xi[0] == pytest.approx(xi[1], rel=1e-5)
    assert xi[2] == pytest.approx(xi[3], rel=1e-5)
    assert xi[4] / xi[5] == pytest.approx(2.0**2.3, rel=1e-9)
    assert im.imf_density(np.array([0.0, -1.0]), cfg).tolist() == [0.0, 0.0]


@pytest.mark.unit
def test_r_max_past_runs_along_isochrone_and_across_ages() -> None:
    m_init = np.array([[1.0, 1.1, 1.2, np.nan], [1.0, 1.05, 1.1, 1.12]])
    log_r = np.log10(np.array([[1.0, 50.0, 10.0, np.nan], [1.0, 2.0, 3.0, 4.0]]))
    r = im._r_max_past(np.array([9.0, 9.1]), m_init, log_r)
    assert r[0, 2] == pytest.approx(50.0)  # clump point remembers the RGB tip on its own isochrone
    assert r[1, 2] >= 50.0 - 1e-9  # M_init 1.1 had R = 50 at the earlier age
    assert np.isnan(r[0, 3])


def _toy_points(n: int = 20000) -> im.PriorPoints:
    """A toy main sequence: C0 = 0.3 + 2.2 (1 - M), M_G = 2 + 3.5 C0, mass 0.1-1."""
    m = np.linspace(0.1, 1.0, n)
    c = 0.3 + 2.2 * (1.0 - m)
    mg = 2.0 + 3.5 * c
    vals = {
        "initial_mass": m, "star_mass": m, "log_r": np.log10(m), "log_g": np.full(n, 4.5),
        "log_teff": np.full(n, 3.7), "phase": np.zeros(n), "mg": mg, "bp": mg + 0.3, "rp": mg - 0.3 - c + 0.6,
        "r_max_past": m,
    }
    return im.PriorPoints(weight=np.full(n, 1.0 / n), colour=c, mg=mg, feh=np.zeros(n), log_age=np.full(n, 9.5), values=vals)


@pytest.mark.unit
def test_posterior_moments_recover_the_toy_mass_and_flag_off_grid() -> None:
    cmap = im.build_cmd_map(_toy_points(), im.CmdMapConfig(colour_min=0.0, colour_max=3.0, mag_min=0.0, mag_max=14.0))
    lk = im.IsochroneLikelihoodConfig()
    truth = np.array([0.3, 0.6, 0.9])
    c = 0.3 + 2.2 * (1.0 - truth)
    m = 2.0 + 3.5 * c
    c = np.append(c, 0.5)  # far off the toy sequence
    m = np.append(m, 12.0)
    post = im.posterior_moments(c, m, np.full(4, 0.02), np.full(4, 0.05), cmap, lk)
    assert post.ok.tolist() == [True, True, True, False]
    assert post.reason[3] == "off_grid"
    np.testing.assert_allclose(post.m1_mean[:3], truth, atol=0.01)
    assert np.all(post.m1_sigma[:3] < 0.03)
    assert np.all(post.p_evolved[:3] == 0.0)
    nan = im.posterior_moments(np.array([np.nan]), np.array([5.0]), np.array([0.02]), np.array([0.05]), cmap, lk)
    assert nan.reason[0] == "no_cmd"


@pytest.mark.unit
def test_chunking_does_not_change_the_answer() -> None:
    cmap = im.build_cmd_map(_toy_points(), im.CmdMapConfig(colour_min=0.0, colour_max=3.0, mag_min=0.0, mag_max=14.0))
    rng = np.random.default_rng(0)
    c = rng.uniform(0.4, 2.2, 500)
    m = 2.0 + 3.5 * c + rng.normal(0, 0.05, 500)
    a = im.posterior_moments(c, m, np.full(500, 0.02), np.full(500, 0.05), cmap, im.IsochroneLikelihoodConfig(chunk_rows=37))
    b = im.posterior_moments(c, m, np.full(500, 0.02), np.full(500, 0.05), cmap, im.IsochroneLikelihoodConfig(chunk_rows=500))
    np.testing.assert_allclose(a.m1_mean, b.m1_mean, rtol=1e-10, equal_nan=True)


@pytest.mark.unit
def test_likelihood_sigmas_project_the_extinction_vector() -> None:
    lk = im.IsochroneLikelihoodConfig(provisional_ebv_fractional_sigma=0.1)
    sc, sm = im.likelihood_sigmas(np.array([0.1]), np.array([0.5]), np.array([2.7]), np.array([1.35]), lk)
    assert sc[0] == pytest.approx(math.hypot(0.02, 0.05 * 1.35))
    assert sm[0] == pytest.approx(math.sqrt(0.1**2 + 0.05**2 + (0.05 * 2.7) ** 2))


# ---------------------------------------------------------------------------
# malmquist_cmd on toy colours and ridge
# ---------------------------------------------------------------------------


def _toy_ms(same_colour: bool = False) -> mc.MsColours:
    mass = np.geomspace(0.1, 2.0, 200)
    colour = 0.3 + 2.2 * (1.0 - np.clip(mass, 0.1, 1.0))
    bp_g = 0.4 * colour if not same_colour else np.full(mass.size, 0.4)
    g_rp = colour - bp_g if not same_colour else np.full(mass.size, 0.6)
    if same_colour:
        colour = np.full(mass.size, 1.0)
    return mc.MsColours(mass=mass, colour=colour + 1e-9 * np.arange(mass.size), bp_g=bp_g, g_rp=g_rp, mg=2.0 + 3.5 * colour)


def _toy_ridge() -> mc.RidgeTables:
    c = np.linspace(0.3, 2.8, 26)
    return mc.RidgeTables.from_ridge(giants.MSRidge(colour=c, mag=2.0 + 3.5 * c, sigma=np.full(c.size, 0.25), n_rows=np.full(c.size, 1000)))


@pytest.mark.unit
def test_subtracting_a_companion_moves_the_primary_faint_and_blue() -> None:
    ms = _toy_ms()
    c1, m1, ok = mc.subtract_companion(np.array([1.0, 1.0, 1.0]), np.array([5.5, 5.5, 5.5]), np.array([-np.inf, -0.3, 2.0]),
                                       np.array([0.5, 0.5, 0.5]), ms)
    assert ok.tolist() == [True, True, False]
    assert c1[0] == pytest.approx(1.0) and m1[0] == pytest.approx(5.5)
    assert m1[1] == pytest.approx(5.5 + 2.5 * math.log10(1 + 10**-0.3))
    assert c1[1] < 1.0  # a redder companion leaves a bluer primary


@pytest.mark.unit
def test_same_colour_companion_reduces_to_the_1d_weight() -> None:
    """Spec §11.4 limit: C_1 = C_s, so L = N(ΔM + 2.5 log10(1 + f); 0, σ) with ΔM = M_s − R(C_s)."""
    ms = _toy_ms(same_colour=True)
    rt = _toy_ridge()
    cs, msys, lf = 1.0, 5.0, -0.4
    c1, m1, _ = mc.subtract_companion(np.array([cs]), np.array([msys]), np.array([lf]), np.array([0.6]), ms)
    assert c1[0] == pytest.approx(cs, abs=1e-6)
    l2 = mc.log_ridge_likelihood(c1, m1, np.array([0.1]), rt)
    dm = msys - (2.0 + 3.5 * cs)
    l1 = mq.log_light_likelihood(np.array([dm]), np.array([lf]), np.array([math.hypot(0.25, 0.1)]))
    assert l2[0] == pytest.approx(l1[0], abs=1e-9)


@pytest.mark.unit
def test_extinction_error_projects_onto_the_ridge_residual() -> None:
    rt = _toy_ridge()  # slope 3.5
    a = mc.log_ridge_likelihood(np.array([1.0]), np.array([5.5]), np.array([0.0]), rt, sigma_a_mag=0.0)
    b = mc.log_ridge_likelihood(np.array([1.0]), np.array([5.5]), np.array([0.0]), rt, sigma_a_mag=0.2, k_ag_over_ebprp=3.5)
    assert a[0] == pytest.approx(b[0])  # reddening vector parallel to the ridge: no information lost
    c = mc.log_ridge_likelihood(np.array([1.0]), np.array([5.5]), np.array([0.0]), rt, sigma_a_mag=0.2, k_ag_over_ebprp=2.0)
    assert c[0] != pytest.approx(a[0])


@pytest.fixture(scope="module")
def frag() -> ps.ProposalSetFragment:
    return ps.load_proposal_set_fragment(DECIDED)


@pytest.fixture(scope="module")
def small_qf(frag: ps.ProposalSetFragment) -> mc.QFGrid:
    g = mc.QFGridConfig(log_m1_min=-0.6, log_m1_max=0.3, n_m1=6, n_log_q_fine=60, n_log_q_bins=12, n_log_p=60,
                        n_hermite=7, log_f_min=-5.0, log_f_max=0.5, n_log_f=55)
    return mc.build_qf_grid(frag.target_mds17, g)


@pytest.mark.physics
def test_qf_grid_marginal_matches_the_1d_flux_grid(frag: ps.ProposalSetFragment, small_qf: mc.QFGrid) -> None:
    """Summed over log q, λ_qf is the §9 λ_f on the same f bins, and F_lum agrees."""
    g1 = mq.FluxMarginalGridConfig(log_m1_min=-0.6, log_m1_max=0.3, n_m1=6, n_log_q=60, n_log_p=60, n_hermite=7,
                                   log_f_min=-5.0, log_f_max=0.5, n_log_f=55)
    one = mq.build_flux_marginal(frag.target_mds17, g1)
    np.testing.assert_allclose(small_qf.f_lum, one.f_lum, rtol=1e-10)
    np.testing.assert_allclose(small_qf.lam.sum(axis=1), one.lam_f, rtol=1e-8, atol=1e-14)


@pytest.mark.physics
def test_weights_normalize_over_the_companion_grid(small_qf: mc.QFGrid) -> None:
    """(1 − F) W(∅) + Σ_bins λ W(bin) = 1 per row when draws sit at the bin centres."""
    ms, rt = _toy_ms(), _toy_ridge()
    cfgw = mc.CmdMalmquistConfig(single_star_density=mc.SingleStarDensityConfig(provisional_model="gaussian_ridge"),
                                 grid=mc.QFGridConfig(log_m1_min=-0.6, log_m1_max=0.3, n_m1=6, n_log_q_fine=60, n_log_q_bins=12,
                                                      n_log_p=60, n_hermite=7, log_f_min=-5.0, log_f_max=0.5, n_log_f=55))
    c = np.array([0.9, 1.2, 1.6])
    m = 2.0 + 3.5 * c - np.array([0.0, 0.5, 0.9])  # on the ridge, mildly and strongly over-luminous
    m1 = np.array([0.8, 0.7, 0.6])
    rows = mc.cmd_rows(c, m, np.full(3, 0.1), m1, rt)
    norm = mc.row_normalization(rows, small_qf, rt, ms, cfgw)
    q = 10.0**small_qf.log_q
    for s in range(3):
        lam, f_lum = small_qf.interpolate(m1[s])
        lq, lf = np.meshgrid(small_qf.log_q, small_qf.log_f, indexing="ij")
        truth = {"parent_row": np.full(lq.size, s), "is_dark": np.zeros(lq.size, bool), "log10_flux_ratio": lf.ravel(),
                 "m2_msun": (m1[s] * 10.0**lq).ravel()}
        lw = mc.log_weight_for_draws(truth, rows, norm, rt, ms, cfgw)
        dark = {"parent_row": np.array([s]), "is_dark": np.array([True]), "log10_flux_ratio": np.array([0.0]), "m2_msun": np.array([1.0])}
        w0 = math.exp(mc.log_weight_for_draws(dark, rows, norm, rt, ms, cfgw)[0])
        total = (1.0 - f_lum) * w0 + float(np.sum(lam.ravel() * np.exp(lw)))
        assert total == pytest.approx(1.0, rel=1e-9)
        assert math.exp(norm.log_p_single[s]) == pytest.approx((1.0 - f_lum) * w0, rel=1e-9)
    assert q.size == small_qf.lam.shape[1]
    # the over-luminous row is more likely a binary
    assert norm.log_p_single[2] < norm.log_p_single[0]


@pytest.mark.unit
def test_colour_jacobian_is_one_without_a_companion_and_compresses_with_one() -> None:
    ms = _toy_ms()
    assert mc.colour_jacobian(np.array([1.0]), 0.0, 0.0, ms)[0] == pytest.approx(1.0)
    assert mc.colour_jacobian(np.array([1.0]), 0.5, 0.5, ms)[0] == pytest.approx(0.5)


@pytest.mark.unit
def test_mist_density_lookup_and_ridge_anchor() -> None:
    """The anchored density peaks on the ridge and integrates to ~1 over M at fixed colour."""
    cmap = im.build_cmd_map(_toy_points(), im.CmdMapConfig(colour_min=0.0, colour_max=3.0, mag_min=0.0, mag_max=14.0))
    c = np.linspace(0.4, 2.2, 10)
    ridge = giants.MSRidge(colour=c, mag=2.3 + 3.5 * c, sigma=np.full(c.size, 0.2), n_rows=np.full(c.size, 500))
    dcfg = mc.SingleStarDensityConfig()
    dens = mc.build_single_star_density(cmap, dcfg, ridge)
    np.testing.assert_allclose(dens.shift(c), 0.3, atol=0.03)  # toy MS is 0.3 mag brighter than the ridge
    mg = np.linspace(0.0, 14.0, 2801)
    for cc in (0.8, 1.5):
        ld = dens.log_density(np.full(mg.size, cc), mg, np.full(mg.size, 0.1))
        assert abs(mg[np.argmax(ld)] - (2.3 + 3.5 * cc)) < 0.05
    cfg_m = mc.CmdMalmquistConfig(grid=mc.QFGridConfig(log_m1_min=-0.6, log_m1_max=0.3, n_m1=6, n_log_q_fine=60, n_log_q_bins=12,
                                                       n_log_p=60, n_hermite=7, log_f_min=-5.0, log_f_max=0.5, n_log_f=55))
    with pytest.raises(ValueError, match="SingleStarDensity"):
        mc.log_primary_likelihood(np.array([1.0]), np.array([5.0]), np.array([0.1]), _toy_ridge(), cfg_m)


@pytest.mark.unit
def test_unit_weight_rows_are_flagged_with_reasons() -> None:
    rt = _toy_ridge()
    rows = mc.cmd_rows(np.array([1.0, 0.1, np.nan, 1.0, 1.0]), np.array([5.5, 2.0, 5.0, 5.5, 5.5]), np.zeros(5),
                       np.array([0.8, 0.8, 0.8, 0.8, np.nan]), rt, evolved=np.array([False, False, False, True, False]))
    assert rows.unit_reason.tolist() == ["weighted", "outside_ridge", "no_cmd", "evolved", "no_m1"]


@pytest.mark.unit
def test_ridge_ruwe_cut_requires_ruwe() -> None:
    cfg = giants.load_giants_config("config/population/giants.yaml")
    assert cfg.ridge.ruwe_max == 1.4
    rng = np.random.default_rng(1)
    c = rng.uniform(0.4, 2.0, 50000)
    m = 2.0 + 3.5 * c + np.abs(rng.normal(0, 0.25, c.size))
    with pytest.raises(ValueError, match="ruwe"):
        giants.fit_ms_ridge(m, c, np.full(c.size, 50.0), cfg.ridge)
    ruwe = np.where(rng.uniform(size=c.size) < 0.5, 1.0, 2.0)
    r = giants.fit_ms_ridge(m, c, np.full(c.size, 50.0), cfg.ridge, ruwe=ruwe)
    assert r.n_rows.sum() < 0.6 * c.size


@pytest.mark.unit
def test_evolved_flux_relation_leaves_dwarf_rows_unchanged(frag: ps.ProposalSetFragment) -> None:
    rng = np.random.default_rng(3)
    n = 200
    m1 = rng.uniform(0.8, 1.5, n)
    m2 = m1 * rng.uniform(0.2, 0.9, n)
    truth = {"m1_msun": m1, "m2_msun": m2, "period_days": 10 ** rng.uniform(1, 4, n), "eccentricity": rng.uniform(0, 0.5, n),
             "log10_flux_ratio": ps.relation_log10_flux_ratio(m1, m2) + rng.normal(0, 0.05, n), "is_dark": np.zeros(n, bool)}
    base = ps.mds17_luminous_log_intensity(truth, frag.target_mds17)
    same = ps.mds17_luminous_log_intensity(truth, frag.target_mds17, evolved_mg0_system=np.full(n, np.nan))
    np.testing.assert_array_equal(base, same)
    mg0 = np.where(np.arange(n) < 10, 0.5, np.nan)
    evo = ps.mds17_luminous_log_intensity(truth, frag.target_mds17, evolved_mg0_system=mg0)
    np.testing.assert_array_equal(evo[10:], base[10:])
    assert np.all(evo[:10] < base[:10])  # dwarf-relation f is ~2 dex too bright for a giant


# ---------------------------------------------------------------------------
# With the real MIST grid (skipped where it is absent)
# ---------------------------------------------------------------------------


@needs_mist
@pytest.mark.physics
def test_mist_sun_and_m_dwarf_masses() -> None:
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    post = model.fit(np.array([0.82, 2.5]), np.array([4.67, 10.0]), sigma_mu=np.array([0.02, 0.02]))
    assert 0.85 < post.m1_mean[0] < 1.1
    assert 0.3 < post.m1_mean[1] < 0.5
    assert np.all(post.p_evolved < 0.05)


@needs_mist
@pytest.mark.slow
def test_cmd_closed_loop_small() -> None:
    """Spec §11.5 on the small universe. The §11.5 target (every 2-D pull ≤ 3) is NOT met yet
    (docs/gate418: residual 4-7σ in single bins). This pins what is measured: no W fails visibly,
    the 2-D W removes most of it, and the twin bin closes."""
    from darkhunter_pop import malmquist_cmd_closed_loop as cl

    res, _ = cl.run_cmd_closed_loop("small")
    assert cl.max_abs_pull(res, "none") > 8.0
    assert cl.max_abs_pull(res, "two_d") < 0.75 * cl.max_abs_pull(res, "none")
    assert abs(res.tables["q"]["pull_two_d"][-1]) < 3.0


@pytest.mark.unit
def test_ms_colour_bank_groups_rows_and_fiducial_mode() -> None:
    """MP-Q37 coeval: rows get their own isochrone's colours; fiducial mode is one group."""
    from types import SimpleNamespace

    feh = np.array([-0.5, -0.5, 0.2, np.nan])
    age = np.array([9.5, 9.5, 9.0, np.nan])
    fake = SimpleNamespace(feh=np.array([-1.0, 0.0, 0.5]), log_age=np.array([9.0, 9.5, 10.0]))
    calls: list[tuple[float, float]] = []

    def fake_ms(grid, f, a):  # type: ignore[no-untyped-def]
        calls.append((round(f, 3), round(a, 3)))
        return _toy_ms()

    import darkhunter_pop.malmquist_cmd as mcm

    orig = mcm.ms_colours
    mcm.ms_colours = fake_ms  # type: ignore[assignment]
    try:
        bank = mc.ms_colour_bank(fake, feh, age, mc.CompanionColourConfig())  # type: ignore[arg-type]
        assert bank.row_group[0] == bank.row_group[1] != bank.row_group[2]
        assert (-0.5, 9.5) in calls and (0.2, 9.0) in calls and (-0.06, 9.5) in calls  # NaN row: fiducial
        one = mc.ms_colour_bank(fake, feh, age, mc.CompanionColourConfig(mode="fiducial_ms"))  # type: ignore[arg-type]
        assert len(one.groups) == 1 and np.all(one.row_group == 0)
    finally:
        mcm.ms_colours = orig  # type: ignore[assignment]


@pytest.mark.unit
def test_evolved_flux_proposal_centre_is_used_by_sampler_and_density() -> None:
    """MP-Q28d: evolved rows' relation centre is the §10.4 relation, identically in the
    sampler and in log_q_total_for (so the importance weights stay exact)."""
    from dataclasses import replace

    from darkhunter_pop import giants as gi

    frag = ps.load_proposal_set_fragment("config/population/proposal_set_isochrone_smoke.yaml")
    prop = frag.proposal.model_copy(update={"n_draws": 400})
    assert prop.flux.evolved_rows_centre == "evolved_relation"
    n = 50
    rng = np.random.default_rng(5)
    cols = {"source_id": np.arange(n), "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n),
            "l": rng.uniform(0, 360, n), "b": rng.uniform(-60, 60, n), "parallax": rng.uniform(0.5, 5, n),
            "pmra": np.zeros(n), "pmdec": np.zeros(n), "phot_g_mean_mag": rng.uniform(10, 18, n)}
    giant = np.arange(n) < 20
    parent = ps.ParentSnapshot(columns=cols, m1_msun=rng.uniform(0.8, 1.5, n), m1_source=np.full(n, "MIST"),
                               atmosphere_logg=np.full(n, 4.0), truth_parallax_mas=cols["parallax"], is_giant=giant,
                               flags={"ok": np.ones(n, bool)}, usable=np.ones(n, bool), meta={}, scale_to_full=1.0,
                               path=Path("synthetic"), cmd={"mg0": np.where(giant, 0.5, 5.0)})
    truth = ps.sample_proposal(parent, prop)
    again = ps.log_q_total_for(truth, parent, prop)
    np.testing.assert_allclose(again, truth["log_q_total"], rtol=1e-12)
    r = truth["parent_row"]
    sel = giant[r] & ~truth["is_dark"]
    centre = gi.evolved_log10_flux_ratio(truth["m2_msun"][sel], 0.5)
    near = np.abs(truth["log10_flux_ratio"][sel] - centre) < 0.6
    assert near.mean() > 0.7  # most evolved-row luminous draws sit on the evolved relation
    _ = replace


def _toy_iso() -> dict[str, np.ndarray]:
    m = np.linspace(0.1, 1.2, 400)
    c = 0.3 + 2.2 * (1.0 - np.clip(m, 0.1, 1.0))
    mg = 2.0 + 3.5 * c
    return {"star_mass": m, "mg": mg, "bp": mg + 0.4 * c, "rp": mg - 0.6 * c, "phase": np.zeros(m.size)}


@pytest.mark.unit
def test_deblend_recovers_the_primary_of_a_synthetic_binary() -> None:
    """MP-Q36: build a system from primary M1 = 0.8 and a coeval companion (q, f); the
    deblending on the same isochrone returns M1 (single stars return their own mass)."""
    iso = _toy_iso()
    fl = lambda x: 10.0 ** (-0.4 * x)  # noqa: E731
    m1, q, f = 0.8, 0.7, 0.4
    i1 = int(np.argmin(np.abs(iso["star_mass"] - m1)))
    i2 = int(np.argmin(np.abs(iso["star_mass"] - q * iso["star_mass"][i1])))
    g1, bp1, rp1 = iso["mg"][i1], iso["bp"][i1], iso["rp"][i1]
    g2 = g1 - 2.5 * np.log10(f)
    bp2 = g2 + (iso["bp"][i2] - iso["mg"][i2])
    rp2 = g2 - (iso["mg"][i2] - iso["rp"][i2])
    gs = -2.5 * np.log10(fl(g1) + fl(g2))
    cs = -2.5 * np.log10(fl(bp1) + fl(bp2)) + 2.5 * np.log10(fl(rp1) + fl(rp2))
    mm, chi2 = im.deblend_primary_mass(iso, np.array([cs, iso["bp"][i1] - iso["rp"][i1]]), np.array([gs, g1]),
                                       np.full(2, 0.02), np.full(2, 0.05), np.array([q, 0.5]),
                                       np.array([np.log10(f), -np.inf]))
    assert mm[0] == pytest.approx(iso["star_mass"][i1], abs=0.01)
    assert mm[1] == pytest.approx(iso["star_mass"][i1], abs=0.01)
    assert chi2[0] < 1.0 and chi2[1] < 1.0


@pytest.mark.unit
def test_posterior_sampler_draws_near_the_star() -> None:
    pts = _toy_points()
    cmapcfg = im.CmdMapConfig(colour_min=0.0, colour_max=3.0, mag_min=0.0, mag_max=14.0)
    smp = im.PosteriorSampler.build(pts, cmapcfg)
    c = np.full(2000, 0.3 + 2.2 * 0.4)  # true mass 0.6
    m = 2.0 + 3.5 * c
    d = smp.sample(c, m, np.full(c.size, 0.02), np.full(c.size, 0.05), np.random.default_rng(3))
    assert np.all(np.isfinite(d["m1"]))
    assert np.median(d["m1"]) == pytest.approx(0.6, abs=0.01)
    assert 0.003 < np.std(d["m1"]) < 0.03


@pytest.mark.physics
def test_mist_coeval_q_grid_normalizes_weights(frag: ps.ProposalSetFragment) -> None:
    """MP-Q40: with the per-row MIST relation, (1 − F) W(∅) + Σ λ_q Σ_h w_h W(q, f_h) = 1."""
    g = mc.QFGridConfig(log_m1_min=-0.6, log_m1_max=0.3, n_m1=6, n_log_q_fine=60, n_log_q_bins=12, n_log_p=60,
                        n_hermite=7, log_f_min=-5.0, log_f_max=0.5, n_log_f=55)
    tgt = frag.target_mds17.model_copy(update={"mass_luminosity": "mist_coeval"})
    qg = mc.build_q_grid(tgt, g)
    assert np.all(qg.f_lum > 0) and qg.weights.sum() == pytest.approx(1.0)
    ms, rt = _toy_ms(), _toy_ridge()
    cfgw = mc.CmdMalmquistConfig(single_star_density=mc.SingleStarDensityConfig(provisional_model="gaussian_ridge"), grid=g)
    c = np.array([1.0, 1.4])
    m = 2.0 + 3.5 * c - np.array([0.0, 0.6])
    m1 = np.array([0.75, 0.6])
    rows = mc.cmd_rows(c, m, np.full(2, 0.1), m1, rt)
    norm = mc.row_normalization(rows, qg, rt, ms, cfgw)
    for s in range(2):
        lam, f_lum = qg.interpolate(m1[s])
        q = 10.0**qg.log_q
        rel = ms.log10_flux_ratio(m1[s], m1[s] * q)
        lf = rel[:, None] + qg.sigma_f_dex * qg.nodes[None, :]
        n = lf.size
        truth = {"parent_row": np.full(n, s), "is_dark": np.zeros(n, bool), "log10_flux_ratio": lf.ravel(),
                 "m2_msun": np.repeat(m1[s] * q, qg.nodes.size)}
        lw = mc.log_weight_for_draws(truth, rows, norm, rt, ms, cfgw)
        dark = {"parent_row": np.array([s]), "is_dark": np.array([True]), "log10_flux_ratio": np.array([0.0]), "m2_msun": np.array([1.0])}
        w0 = math.exp(mc.log_weight_for_draws(dark, rows, norm, rt, ms, cfgw)[0])
        tot = (1 - f_lum) * w0 + float(np.sum((lam[:, None] * qg.weights[None, :]).ravel() * np.exp(lw)))
        assert tot == pytest.approx(1.0, rel=1e-9)


@pytest.mark.unit
def test_mist_coeval_target_needs_the_relation(frag: ps.ProposalSetFragment) -> None:
    tgt = frag.target_mds17.model_copy(update={"mass_luminosity": "mist_coeval"})
    truth = {"m1_msun": np.array([1.0]), "m2_msun": np.array([0.6]), "period_days": np.array([300.0]),
             "eccentricity": np.array([0.2]), "log10_flux_ratio": np.array([-1.0]), "is_dark": np.array([False])}
    with pytest.raises(ValueError, match="relation_log10_f"):
        ps.mds17_luminous_log_intensity(truth, tgt)
    a = ps.mds17_luminous_log_intensity(truth, tgt, relation_log10_f=np.array([-1.0]))
    b = ps.mds17_luminous_log_intensity(truth, tgt, relation_log10_f=np.array([-1.3]))
    assert a[0] - b[0] == pytest.approx(0.5 * (0.3 / tgt.flux_sigma_dex) ** 2)


@pytest.mark.unit
def test_feh_likelihood_selects_the_matching_layer() -> None:
    """MP-Q33: a [Fe/H] measurement picks the layer whose isochrone it matches."""
    a = _toy_points(4000)
    b = _toy_points(4000)
    b.values["star_mass"] = b.values["star_mass"] * 1.2  # same CMD, heavier at the other [Fe/H]
    pts = im.PriorPoints(weight=np.concatenate([a.weight, b.weight]) / 2, colour=np.concatenate([a.colour, b.colour]),
                         mg=np.concatenate([a.mg, b.mg]), feh=np.concatenate([np.full(4000, -0.5), np.full(4000, 0.3)]),
                         log_age=np.concatenate([a.log_age, b.log_age]),
                         values={k: np.concatenate([a.values[k], b.values[k]]) for k in a.values})
    lm = im.build_layered_cmd_map(pts, im.CmdMapConfig(colour_min=0.0, colour_max=3.0, mag_min=0.0, mag_max=14.0), [-0.5, 0.3])
    c = np.array([1.2, 1.2, 1.2])
    m = 2.0 + 3.5 * c
    post = im.posterior_moments_with_feh(c, m, np.full(3, 0.02), np.full(3, 0.05), np.array([-0.5, 0.3, np.nan]),
                                         np.full(3, 0.1), lm, im.IsochroneLikelihoodConfig())
    base = 1.0 - (1.2 - 0.3) / 2.2
    assert post.m1_mean[0] == pytest.approx(base, abs=0.02)
    assert post.m1_mean[1] == pytest.approx(1.2 * base, abs=0.02)
    assert post.m1_mean[0] < post.m1_mean[2] < post.m1_mean[1]  # no measurement: the prior mixes both


@pytest.mark.unit
def test_gdr3apcal_reliability_cut() -> None:
    pytest.importorskip("gdr3apcal")
    from darkhunter_pop.config_schema import FehLikelihoodConfig

    cols = {"teff_gspphot": np.array([5500.0, 7000.0, 5200.0]), "logg_gspphot": np.array([4.4, 4.0, 4.5]),
            "mh_gspphot": np.array([-0.5, 0.0, -0.3]), "azero_gspphot": np.array([0.1, 0.1, 0.1]),
            "ebpminrp_gspphot": np.array([0.05, 0.04, 0.05]), "ag_gspphot": np.array([0.08, 0.08, 0.08]),
            "mg_gspphot": np.array([4.6, 2.5, 5.0]), "libname_gspphot": np.array(["MARCS", "MARCS", "A"])}
    f, ok = im.calibrated_gspphot_feh(cols, np.array([30.0, 40.0, 20.0]), np.array([20.0, 30.0, 25.0]), FehLikelihoodConfig())
    assert ok.tolist() == [True, False, False]  # Teff > 6500 K and the A library are not reliable
    assert np.isfinite(f[0]) and np.isnan(f[1])


def _tiny_native() -> im.NativeGrid:
    feh = np.array([-0.5, 0.0])
    lage = np.array([9.0, 10.0])
    eep = np.array([200, 300, 400], dtype=np.int64)
    v = np.zeros((2, 2, 3, len(im.NATIVE_QUANTITIES)), dtype=np.float32)
    k = im.NATIVE_QUANTITIES.index
    v[..., k("initial_mass")] = np.array([0.8, 1.0, 1.01], dtype=np.float32)
    v[..., k("phase")] = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    v[..., k("mg")] = np.array([5.0, 3.0, 0.5], dtype=np.float32)
    v[..., k("bp")] = np.array([1.0, 0.8, 1.2], dtype=np.float32)
    return im.NativeGrid(feh=feh, log_age=lage, eep=eep, values=v)


@pytest.mark.unit
def test_age_power_and_cheb_weight_reweight_the_prior() -> None:
    """#418 APOKASC-3 knobs: (age/1 Gyr)^gamma on the age weight, rho on core-He-burning points."""
    grid = _tiny_native()
    base = im.IsochroneMassConfig()
    base = base.model_copy(update={"grid": base.grid.model_copy(update={"n_feh_substeps": 1, "n_age_substeps": 1})})
    p0 = im.prior_points(grid, base)
    p1 = im.prior_points(grid, base.model_copy(update={"age": base.age.model_copy(update={"provisional_age_power": 1.0})}))
    p2 = im.prior_points(grid, base.model_copy(update={"provisional_cheb_weight": 2.0}))
    old0, old1 = p0.log_age > 9.5, p1.log_age > 9.5
    # gamma = 1 multiplies the 10 Gyr weights by 10 relative to the 1 Gyr ones.
    r0 = p0.weight[old0].sum() / p0.weight[~old0].sum()
    r1 = p1.weight[old1].sum() / p1.weight[~old1].sum()
    assert r1 / r0 == pytest.approx(10.0, rel=1e-9)
    ch0, ch2 = p0.values["phase"] >= 2.5, p2.values["phase"] >= 2.5
    s0 = p0.weight[ch0].sum() / p0.weight[~ch0].sum()
    s2 = p2.weight[ch2].sum() / p2.weight[~ch2].sum()
    assert s2 / s0 == pytest.approx(2.0, rel=1e-9)
