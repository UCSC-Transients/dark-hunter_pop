"""Statistical epoch model around gaiamock's GOST list (#400, #398).

Unit tests use synthetic transit lists and a fake gaiamock module; the ``gaiamock``
tests run the real overlay and check that the wrapper changes nothing but the
transit list.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import epoch_model as em
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.gaiamock_vendor import is_overlay_ready

REPO = Path(__file__).resolve().parents[1]


def _cfg(**kw: Any) -> em.EpochModelConfig:
    base = load_config().dr3.epoch_model
    assert base is not None
    out = dataclasses.replace(em.epoch_model_config_from_mapping(base), enabled=True)
    if any(k_.startswith("transit_loss") for k_ in kw):
        # binned-model tests: switch off the continuous / clustered / visibility-period parts
        kw = {"continuous": None, "clustered": None, "vp_loss": None, **kw}
    elif "clustered" in kw:
        # episode-model tests (E4): the #432 visibility-period loss replaces episodes when set
        kw = {"vp_loss": None, **kw}
    return dataclasses.replace(out, **kw) if kw else out


def _transits(n: int, *, start: float = 2457000.0, step: float = 3.0, rows: int = 10) -> np.ndarray:
    """``n`` synthetic FoV transits of ``rows`` CCD rows 5 s apart, ``step`` days apart."""
    t0 = start + step * np.arange(n)
    return (t0[:, None] + (5.0 / 86400.0) * np.arange(rows)[None, :]).ravel()


@pytest.mark.unit
def test_obmt_relation_matches_lindegren_window() -> None:
    cfg = _cfg()
    kw = dict(reference_rev=cfg.obmt_reference_rev, reference_jd_tcb=cfg.obmt_reference_jd_tcb,
              rev_per_day=cfg.obmt_rev_per_day)
    # Lindegren et al. (2021) Sect. 2.2: OBMT 1192.13-5230.09 = J2014.64032-J2017.40415.
    jd = em.obmt_to_jd_tcb(np.array([1192.13, 5230.09]), **kw)
    julian_year = 2451545.0 + (np.array([2014.64032, 2017.40415]) - 2000.0) * 365.25
    np.testing.assert_allclose(jd, julian_year, atol=0.01)


@pytest.mark.unit
def test_gap_table_loads_and_matches_checksum() -> None:
    cfg = _cfg()
    s, e, desc = em.load_gap_table(cfg.gap_table)
    assert s.size == 138 and np.all(e >= s)
    assert any("Decontamination" in d for d in desc)
    assert em._sha256(cfg.gap_table) == load_config().dr3.epoch_model.gap_table_sha256


@pytest.mark.unit
def test_checksum_mismatch_refuses(tmp_path: Path) -> None:
    section = load_config().dr3.epoch_model.model_dump()
    bad = tmp_path / "gaps.csv"
    bad.write_text("start,end,duration [rev],description\n1.0,2.0,1.0,x\n")
    section["gap_table"] = str(bad)
    with pytest.raises(ValueError, match="sha256"):
        em.epoch_model_config_from_mapping(section)


@pytest.mark.unit
def test_window_added_as_open_gaps() -> None:
    gaps = em.gap_intervals_jd(_cfg())
    assert gaps.shape == (140, 2)
    assert np.isneginf(gaps[0, 0]) and np.isposinf(gaps[-1, 1])
    no_window = em.gap_intervals_jd(_cfg(agis_window_obmt_rev=None))
    assert no_window.shape == (138, 2)


@pytest.mark.unit
def test_in_gaps_handles_overlaps_and_edges() -> None:
    gaps = np.array([[10.0, 20.0], [15.0, 16.0], [30.0, 31.0]])
    t = np.array([5.0, 10.0, 17.0, 20.0, 20.5, 30.5, 40.0])
    np.testing.assert_array_equal(em.in_gaps(t, gaps), [False, True, True, True, False, True, False])
    assert not em.in_gaps(t, np.zeros((0, 2))).any()


@pytest.mark.unit
def test_fov_transit_ids_unsorted_input() -> None:
    jd = _transits(4)
    perm = np.random.default_rng(0).permutation(jd.size)
    ids = em.fov_transit_ids(jd[perm], 0.01)
    np.testing.assert_array_equal(ids, np.repeat(np.arange(4), 10)[perm])


@pytest.mark.unit
def test_visibility_periods_definition() -> None:
    assert em.n_visibility_periods_from_days(np.array([0.0, 1.0, 5.5, 6.0, 20.0]), 4.0) == 3
    assert em.n_visibility_periods_from_days(np.array([]), 4.0) == 0


@pytest.mark.unit
def test_transit_loss_probability_bins_and_density() -> None:
    cfg = _cfg(transit_loss_g_edges=(3.0, 13.0, 19.0), transit_loss_prob=(0.02, 0.07))
    assert em.transit_loss_probability(2.0, cfg) == 0.02  # nearest bin below the edges
    assert em.transit_loss_probability(12.9, cfg) == 0.02
    assert em.transit_loss_probability(13.0, cfg) == 0.07
    assert em.transit_loss_probability(21.0, cfg) == 0.07  # nearest bin above
    dens = dataclasses.replace(cfg, transit_loss_density_slope=-0.03, transit_loss_density_ref=1e4)
    assert em.transit_loss_probability(15.0, dens, density_per_deg2=1e5) == pytest.approx(0.04)
    assert em.transit_loss_probability(15.0, dens) == 0.07  # no density given: slope unused


@pytest.mark.unit
def test_config_validation() -> None:
    with pytest.raises(ValueError):
        _cfg(transit_loss_prob=(0.1,))
    with pytest.raises(ValueError):
        _cfg(transit_loss_g_edges=(3.0, 13.0, 19.0), transit_loss_prob=(0.1, 1.0))


@pytest.mark.unit
def test_thin_mask_removes_gaps_and_whole_transits() -> None:
    cfg = _cfg(transit_loss_g_edges=(0.0, 30.0), transit_loss_prob=(0.3,), agis_window_obmt_rev=None)
    jd = _transits(2000, step=0.3)
    gaps = np.array([[jd[100] - 0.001, jd[400] + 0.001]])
    keep = em.thin_gost_mask(jd, cfg, gaps, g_mag=15.0, rng=np.random.default_rng(1))
    assert not keep[100:410].any()
    per_transit = keep.reshape(-1, 10)
    assert np.all(per_transit.all(axis=1) | ~per_transit.any(axis=1))  # all or nothing
    outside = np.r_[per_transit[:10, 0], per_transit[41:, 0]]
    assert outside.mean() == pytest.approx(0.7, abs=0.03)
    again = em.thin_gost_mask(jd, cfg, gaps, g_mag=15.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(keep, again)


@pytest.mark.unit
def test_thin_mask_zero_loss_no_gaps_keeps_everything() -> None:
    cfg = _cfg(transit_loss_g_edges=(0.0, 30.0), transit_loss_prob=(0.0,), apply_gaps=False)
    jd = _transits(50)
    assert em.thin_gost_mask(jd, cfg, np.zeros((0, 2)), g_mag=10.0, rng=np.random.default_rng(3)).all()


def _fake_gaiamock(jd: np.ndarray) -> SimpleNamespace:
    def get_gost_one_position(ra: float, dec: float, data_release: str) -> Any:
        return {em.GOST_TIME_COLUMN: jd}

    class _Tab(dict):
        def __getitem__(self, key: Any) -> Any:
            if isinstance(key, np.ndarray):
                return _Tab({k: v[key] for k, v in self.items()})
            return dict.__getitem__(self, key)

    mod = SimpleNamespace()
    mod.get_gost_one_position = lambda ra, dec, data_release: _Tab(get_gost_one_position(ra, dec, data_release))
    return mod


@pytest.mark.unit
def test_wrapper_patches_and_restores() -> None:
    jd = _transits(100)
    mod = _fake_gaiamock(jd)
    original = mod.get_gost_one_position
    cfg = _cfg(transit_loss_g_edges=(0.0, 30.0), transit_loss_prob=(0.5,), apply_gaps=False)
    with em.gost_epoch_model(mod, cfg, em.SourceEpochContext(g_mag=12.0), np.random.default_rng(0)):
        thinned = mod.get_gost_one_position(0.0, 0.0, data_release="dr3")[em.GOST_TIME_COLUMN]
        assert 300 < thinned.size < 700 and thinned.size % 10 == 0
    assert mod.get_gost_one_position is original
    with pytest.raises(RuntimeError):
        with em.gost_epoch_model(mod, cfg, em.SourceEpochContext(g_mag=12.0), np.random.default_rng(0)):
            raise RuntimeError("boom")
    assert mod.get_gost_one_position is original
    off = dataclasses.replace(cfg, enabled=False)
    with em.gost_epoch_model(mod, off, em.SourceEpochContext(g_mag=12.0), np.random.default_rng(0)):
        assert mod.get_gost_one_position is original


@pytest.mark.unit
def test_rng_streams_reproducible_and_disjoint() -> None:
    a = em.epoch_model_rng(42, 2, 123, 0).uniform(size=5)
    b = em.epoch_model_rng(42, 2, 123, 0).uniform(size=5)
    c = em.epoch_model_rng(42, 2, 123, 0, tag=em.EXCESS_NOISE_RNG_TAG).uniform(size=5)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    with pytest.raises(ValueError):
        em.epoch_model_rng(42, -1)


@pytest.mark.unit
def test_bright_star_excess_noise() -> None:
    t_yr = _transits(30) / 365.25
    off, sigma = em.bright_star_excess_noise(
        t_yr, 14.0, np.random.default_rng(0), g_max=13.0, sigma_max_mas=0.04, split_day=0.01
    )
    assert sigma == 0.0 and not off.any()
    off, sigma = em.bright_star_excess_noise(
        t_yr, 11.0, np.random.default_rng(0), g_max=13.0, sigma_max_mas=0.04, split_day=0.01
    )
    assert 0.0 <= sigma <= 0.04
    per = off.reshape(-1, 10)
    assert np.all(per == per[:, :1])  # common to every CCD row of a transit
    assert np.unique(per[:, 0]).size == 30


@pytest.mark.unit
def test_config_paths_dr3_set_dr4_null() -> None:
    cfg = load_config()
    assert cfg.dr3.epoch_model is not None and cfg.dr4.epoch_model is None
    assert cfg.dr3.epoch_model.enabled is True  # Ryan 2026-10-03, #400 E1
    assert cfg.dr3.epoch_model.transit_loss.model == "continuous"
    # N2d adopted (Ryan 2026-10-03): bright per-CCD noise + RUWE = UWE / u0_mock(G)
    assert cfg.dr3.epoch_model.bright_excess_noise is not None and cfg.dr3.epoch_model.bright_excess_noise.enabled
    assert cfg.dr3.epoch_model.ruwe_u0 is not None and cfg.dr3.epoch_model.ruwe_u0.enabled
    loaded = em.epoch_model_config_from_mapping(cfg.dr3.epoch_model)
    assert loaded.excess_noise is not None and loaded.ruwe_u0 is not None
    tl = cfg.dr3.epoch_model.transit_loss
    assert len(tl.prob) == len(tl.g_edges) - 1


@pytest.mark.gaiamock
@pytest.mark.skipif(not is_overlay_ready(), reason="gaiamock_mod overlay not installed")
def test_real_gaiamock_wrapper_only_changes_transits() -> None:
    from darkhunter_pop.forward_model import GlobalRNGSeeds, seeded_global_rng
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    cf = gm.read_in_C_functions()
    seeds = GlobalRNGSeeds(numpy_seed=7, c_rand_seed=8)
    args = dict(ra=150.0, dec=-20.0, parallax=2.0, pmra=5.0, pmdec=-3.0, period=500.0, Tp=10.0,
                ecc=0.3, omega=1.0, inc=1.0, w=0.5, a0_mas=0.5, phot_g_mean_mag=12.0,
                data_release="dr3", c_funcs=cf)
    with seeded_global_rng(seeds, cf):
        bare = gm.predict_astrometry_binary_in_terms_of_a0(**args)
    null = _cfg(transit_loss_g_edges=(0.0, 30.0), transit_loss_prob=(0.0,), apply_gaps=False)
    with seeded_global_rng(seeds, cf), em.gost_epoch_model(
        gm, null, em.SourceEpochContext(12.0), np.random.default_rng(0)
    ):
        same = gm.predict_astrometry_binary_in_terms_of_a0(**args)
    for x, y in zip(bare, same):
        np.testing.assert_array_equal(x, y)
    full = _cfg()
    gaps = em.gap_intervals_jd(full)
    with seeded_global_rng(seeds, cf), em.gost_epoch_model(
        gm, full, em.SourceEpochContext(12.0, l_deg=250.0, b_deg=-30.0, beta_deg=-20.0), np.random.default_rng(0)
    ):
        thin = gm.predict_astrometry_binary_in_terms_of_a0(**args)
    jd = thin[0] * 365.25 + 2457389.0  # gaiamock rescale_times_astrometry for DR3
    assert not em.in_gaps(jd, gaps).any()
    assert len(thin[0]) < len(bare[0])


# --------------------------------------------------------------------------------------
# v2 (Ryan 2026-10-03): continuous + sky keep, clustered loss, per-CCD excess noise
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_real_harmonics_orthonormal() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, 200_000)
    l_deg, b_deg = rng.uniform(0, 360, z.size), np.degrees(np.arcsin(z))
    Y = em.real_sph_harm_galactic(l_deg, b_deg, 2)
    gram = 4 * np.pi * (Y.T @ Y) / z.size
    np.testing.assert_allclose(gram, np.eye(8), atol=0.03)


@pytest.mark.unit
def test_continuous_keep_matches_formula_and_caps() -> None:
    cfg = _cfg()
    cont = cfg.continuous
    assert cont is not None
    g, l_deg, b_deg = 14.7, 100.0, 20.0
    x = (np.clip(g, *cont.g_clip) - cont.g_ref) / cont.g_scale
    eta = sum(c * x**i for i, c in enumerate(cont.coef_g)) + float(
        em.real_sph_harm_galactic(l_deg, b_deg, cont.sky_lmax)[0] @ np.asarray(cont.coef_sky))
    assert em.keep_probability(g, cfg, l_deg=l_deg, b_deg=b_deg) == pytest.approx(min(1.0, np.exp(eta)))
    with pytest.raises(ValueError, match="sky term"):
        em.keep_probability(g, cfg)
    big = dataclasses.replace(cfg, continuous=dataclasses.replace(cont, coef_g=(1.0,) + cont.coef_g[1:]))
    assert em.keep_probability(g, big, l_deg=l_deg, b_deg=b_deg) == 1.0


@pytest.mark.unit
def test_clustered_fraction_ramp() -> None:
    cfg = _cfg(clustered=em.ClusteredLossConfig(g_start=14.5, g_full=16.5, frac_max=0.5, tau_day=2.0))
    assert em.clustered_fraction(14.0, cfg) == 0.0
    assert em.clustered_fraction(15.5, cfg) == pytest.approx(0.25)
    assert em.clustered_fraction(19.0, cfg) == pytest.approx(0.5)
    assert em.clustered_fraction(19.0, _cfg(clustered=None)) == 0.0


@pytest.mark.unit
def test_episode_mask_covers_expected_fraction() -> None:
    t = np.linspace(0.0, 1000.0, 200_001)
    rng = np.random.default_rng(5)
    frac = np.mean([em.loss_episode_mask(t, 0.1, 2.0, rng).mean() for _ in range(40)])
    assert frac == pytest.approx(0.1, abs=0.01)
    assert not em.loss_episode_mask(t, 0.0, 2.0, rng).any()


@pytest.mark.unit
def test_thin_mask_total_keep_with_clustering() -> None:
    cfg = _cfg(clustered=em.ClusteredLossConfig(g_start=14.5, g_full=16.5, frac_max=0.5, tau_day=2.0),
               apply_gaps=False)
    jd = _transits(20_000, step=0.2)
    p = em.keep_probability(18.0, cfg, l_deg=10.0, b_deg=-5.0)
    keeps = [em.thin_gost_mask(jd, cfg, np.zeros((0, 2)), g_mag=18.0, rng=np.random.default_rng(s),
                               l_deg=10.0, b_deg=-5.0).reshape(-1, 10)[:, 0].mean() for s in range(10)]
    assert np.mean(keeps) == pytest.approx(p, abs=0.006)


@pytest.mark.unit
def test_excess_noise_and_ruwe_rescale() -> None:
    nz = em.PerCcdExcessNoiseConfig(g_max=13.0, knots_g=(10.0, 12.0), knots_r2=(0.8, 0.4))
    assert em.excess_noise_r2(11.0, nz) == pytest.approx(0.6)
    assert em.excess_noise_r2(9.0, nz) == pytest.approx(0.8)
    assert em.excess_noise_r2(13.0, nz) == 0.0
    err = np.full(100_000, 0.13)
    extra, k = em.per_ccd_excess_noise(err, 11.0, nz, np.random.default_rng(1))
    assert k == pytest.approx(np.sqrt(1.6))
    assert np.std(extra) == pytest.approx(np.sqrt(0.6) * 0.13, rel=0.01)
    zero, k0 = em.per_ccd_excess_noise(err, 14.0, nz, np.random.default_rng(1))
    assert k0 == 1.0 and not zero.any()
    vec5 = [-1.0, 2.0] + [0.0] * 21
    assert em.rescale_cascade_ruwe(vec5, 2.0)[1] == 1.0
    vec7 = [-7.0] + [0.0] * 7 + [3.0] + [0.0] * 14
    assert em.rescale_cascade_ruwe(vec7, 1.5)[8] == pytest.approx(2.0)
    vecorb = [1.0] * 22 + [4.0]
    assert em.rescale_cascade_ruwe(vecorb, 2.0)[22] == 2.0
    assert em.rescale_cascade_ruwe([0.0] * 23, 2.0) == [0.0] * 23


@pytest.mark.gaiamock
@pytest.mark.skipif(not is_overlay_ready(), reason="gaiamock_mod overlay not installed")
def test_run_cascade_disabled_equals_bare_gaiamock() -> None:
    from darkhunter_pop.forward_model import GlobalRNGSeeds, seeded_global_rng
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    cf = gm.read_in_C_functions()
    seeds = GlobalRNGSeeds(numpy_seed=11, c_rand_seed=12)
    kw = dict(ra=40.0, dec=10.0, parallax=3.0, pmra=1.0, pmdec=2.0, period=300.0, Tp=5.0, ecc=0.2,
              omega=0.3, inc=1.2, w=0.4, a0_mas=0.05, phot_g_mean_mag=11.0, data_release="dr3", c_funcs=cf)
    with seeded_global_rng(seeds, cf):
        t, psi, pf, obs, err = gm.predict_astrometry_binary_in_terms_of_a0(**kw)
        bare = gm.fit_full_astrometric_cascade(t, psi, pf, obs, err, cf, ruwe_min=1.4)
    off = dataclasses.replace(_cfg(), enabled=False)
    with seeded_global_rng(seeds, cf):
        run = em.run_cascade(gm, cf, lambda: gm.predict_astrometry_binary_in_terms_of_a0(**kw), off,
                             em.SourceEpochContext(11.0, l_deg=1.0, b_deg=2.0),
                             epoch_rng=np.random.default_rng(0), noise_rng=np.random.default_rng(1),
                             ruwe_min=1.4, skip_acceleration=False)
    np.testing.assert_array_equal(np.asarray(run.cascade, dtype=float), np.asarray(bare, dtype=float))
    assert run.ruwe_scale == 1.0 and run.n_obs == len(t)


@pytest.mark.unit
def test_u0_table_loads_and_interpolates(tmp_path: Path) -> None:
    sec = load_config().dr3.epoch_model
    assert sec is not None and sec.ruwe_u0 is not None
    tab = em.load_u0_table(REPO / sec.ruwe_u0.table, sec.ruwe_u0.table_sha256)
    assert 1.1 < tab(11.0) < 1.45 and 0.85 < tab(17.0) < 1.05
    assert tab(2.0) == tab.u0[0] and tab(25.0) == tab.u0[-1]
    bad = tmp_path / "u0.csv"
    bad.write_text("# x\ng,u0\n10,1.0\n12,1.2\n")
    t2 = em.load_u0_table(bad)
    assert t2(11.0) == pytest.approx(1.1)
    with pytest.raises(ValueError, match="sha256"):
        em.load_u0_table(bad, "0" * 64)


@pytest.mark.unit
def test_run_cascade_k_uses_u0_when_configured() -> None:
    cfg = _cfg()
    tab = em.RuweU0Table(g=(10.0, 14.0), u0=(1.3, 0.95))
    on = dataclasses.replace(cfg, ruwe_u0=tab)
    assert em.ruwe_scale_u0(12.0, on) == pytest.approx(1.125)
    assert em.ruwe_scale_u0(12.0, dataclasses.replace(cfg, ruwe_u0=None)) == 1.0


def _vp_cfg(**kw: float) -> em.VisibilityPeriodLossConfig:
    base = dict(g_clip=(6.0, 19.0), g_ref=14.0, g_scale=4.0, c0=-3.0, c1=1.0, c2=0.0, c_beta=-1.0,
                c_b=-1.0, e0=0.0, d0=-4.0, d1=0.5)
    base.update(kw)
    return em.VisibilityPeriodLossConfig(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_vp_loss_probabilities() -> None:
    vp = _vp_cfg()
    assert vp.degraded_probability(18.0, 0.0, 0.0) > vp.degraded_probability(18.0, 60.0, 0.0)
    assert vp.degraded_probability(18.0, 0.0, 0.0) > vp.degraded_probability(12.0, 0.0, 0.0)
    assert vp.q_degraded() == pytest.approx(0.5)


@pytest.mark.unit
def test_vp_mode_drops_whole_visibility_periods_and_keeps_mean() -> None:
    # 40 visibility periods of 5 transits (1 d apart), separated by 10 d
    t0 = np.concatenate([1000.0 + 15.0 * k + np.arange(5.0) for k in range(40)])
    jd = (t0[:, None] + (5.0 / 86400.0) * np.arange(10)[None, :]).ravel()
    vp = _vp_cfg(c0=50.0, e0=0.0)  # always degraded, q_bad = 0.5
    cfg = _cfg(apply_gaps=False, clustered=None)
    cfg = dataclasses.replace(cfg, vp_loss=vp)
    p = em.keep_probability(15.0, cfg, l_deg=10.0, b_deg=5.0)
    keeps, nv = [], []
    for s_ in range(200):
        k = em.thin_gost_mask(jd, cfg, np.zeros((0, 2)), g_mag=15.0, rng=np.random.default_rng(s_),
                              l_deg=10.0, b_deg=5.0, beta_deg=0.0)
        per_tr = k.reshape(-1, 10)[:, 0].reshape(40, 5)
        keeps.append(per_tr.mean())
        nv.append(int(per_tr.any(axis=1).sum()))
    # dropout (0.5) exceeds the total loss, so p_ind clips at 0 and keep = 1 - q_bad
    assert np.mean(keeps) == pytest.approx(0.5, abs=0.02)
    assert np.mean(nv) == pytest.approx(20.0, abs=0.6)  # half the visibility periods
    # realistic: rare degraded stars -> expected kept fraction is the calibrated p_keep
    cfg2 = dataclasses.replace(cfg, vp_loss=_vp_cfg(c0=-4.0, e0=-1.0, d0=-6.0))
    k2 = [em.thin_gost_mask(jd, cfg2, np.zeros((0, 2)), g_mag=15.0, rng=np.random.default_rng(s_),
                            l_deg=10.0, b_deg=5.0, beta_deg=0.0).mean() for s_ in range(400)]
    assert np.mean(k2) == pytest.approx(p, abs=0.004)
    with pytest.raises(ValueError, match="beta_deg"):
        em.thin_gost_mask(jd, cfg, np.zeros((0, 2)), g_mag=15.0, rng=np.random.default_rng(0), l_deg=1.0, b_deg=1.0)


@pytest.mark.unit
def test_source_context_coordinates() -> None:
    sc = em.source_context(266.405, -28.936, 15.0)  # near the Galactic centre
    assert abs(sc.b_deg) < 1.0 and abs(sc.beta_deg) < 7.0
