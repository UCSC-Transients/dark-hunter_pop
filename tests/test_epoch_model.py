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
    assert cfg.dr3.epoch_model.enabled is False
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
        gm, full, em.SourceEpochContext(12.0), np.random.default_rng(0)
    ):
        thin = gm.predict_astrometry_binary_in_terms_of_a0(**args)
    jd = thin[0] * 365.25 + 2457389.0  # gaiamock rescale_times_astrometry for DR3
    assert not em.in_gaps(jd, gaps).any()
    assert len(thin[0]) < len(bare[0])
