"""Joint luminous + compact-object refit helpers (#561, spec §12.37)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from darkhunter_pop import joint_refit as jr

FS = jr.FluxScatterConfig(enabled=True, sigma_0=0.1, sigma_min=0.03, q_taper_start=0.8)
RIDGE = {"colour": [0.5, 1.0, 2.0, 3.0], "mag": [3.0, 5.0, 9.0, 12.0]}


@pytest.mark.physics
def test_flux_scatter_ratio_is_a_normalized_reweight() -> None:
    assert jr.flux_scatter_sigma([0.5, 0.8, 0.9, 1.0], FS).tolist() == pytest.approx([0.1, 0.1, 0.065, 0.03])
    x = np.linspace(-1.5, 1.5, 300_001)
    for q in (0.5, 0.9, 1.0):
        lr = jr.flux_scatter_log_ratio(x, np.zeros_like(x), np.full_like(x, q), np.zeros_like(x, bool), FS)
        base = np.exp(-0.5 * (x / 0.1) ** 2) / (0.1 * math.sqrt(2 * math.pi))
        assert np.trapz(base * np.exp(lr), x) == pytest.approx(1.0, abs=1e-6)  # the companion count is unchanged
    r = jr.flux_scatter_log_ratio([0.2, 0.2, np.nan, 0.2], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.5], [False, True, False, False], FS)
    assert r[0] < -5 and r[1] == 0.0 and r[2] == 0.0 and r[3] == 0.0
    assert np.all(jr.flux_scatter_log_ratio([0.2], [0.0], [1.0], [False], FS.model_copy(update={"enabled": False})) == 0.0)


@pytest.mark.unit
def test_variance_inflation_only_below_the_floor() -> None:
    mu, var = np.array([10.0, 10.0, 0.0]), np.array([1.0, 25.0, 0.0])  # ESS 100, 4, (empty)
    v = jr.inflate_low_ess_variance(mu, var, 10.0)
    assert v[0] == 1.0 and v[1] == pytest.approx(25.0 * 10 / 4) and v[2] == 0.0
    assert mu[1] ** 2 / v[1] == pytest.approx(4.0**2 / 10.0)  # leverage of a low-ESS bin shrinks as ESS^2 / ESS_min


@pytest.mark.unit
def test_nonsingle_flags_follow_the_looks_single_cuts() -> None:
    cfg = jr.DepletionConfig(enabled=True, g_edges=(11, 16), distance_edges_kpc=(0, 1.5), ruwe_max=1.25, dm_min=-0.4, dm_colour_range=(1.05, 2.75))
    st = np.array(["five_parameter", "five_parameter", "seven_parameter", "five_parameter", "five_parameter", "not_in_parent", "insufficient_visibility"])
    casc = np.zeros((7, 23))
    casc[:, 1] = [1.0, 1.3, 30.0, 1.0, 1.0, 9.0, 0.0]
    acc = np.array([False, False, False, False, True, False, False])
    c0 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    mg = np.array([7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 7.0])  # ridge at 1.5 is 7.0 -> ΔM 0 or −1
    f = jr.nonsingle_flags(st, casc, acc, np.zeros(7, bool), mg, c0, np.zeros(7, bool), RIDGE, cfg)
    assert f.tolist() == [False, True, True, True, True, False, False]


@pytest.mark.physics
def test_depletion_factor_is_minus_log_p_looks_single() -> None:
    cfg = jr.DepletionConfig(enabled=True, g_edges=(10, 13, 16), distance_edges_kpc=(0, 1.0), ruwe_max=1.25, dm_min=-0.4, dm_colour_range=(1.05, 2.75))
    rows_g = np.array([11.0, 12.0, 14.0, 20.0])
    rows_d = np.array([0.5, 0.5, 0.5, 0.5])
    # cell 0 holds rows 0,1 (2 rows x scale 10 = 20 expanded); non-single weight 4 -> Λ = 0.2
    lf, summ = jr.depletion_log_factor([3.0, 1.0, 5.0, 2.0], [True, True, False, True], [0, 1, 2, 3], rows_g, rows_d,
                                       np.ones(4, bool), None, 10.0, cfg)
    assert lf.tolist() == pytest.approx([0.2, 0.2, 0.0, 0.0])
    assert summ["G10-13 d0-1"]["p_looks_single"] == pytest.approx(math.exp(-0.2))


@pytest.mark.physics
def test_twin_band_loglike_peaks_at_the_real_fraction() -> None:
    k_t, k_s = np.array([30.0, 5.0]), np.array([70.0, 95.0])
    best = max(np.linspace(0.05, 0.6, 56), key=lambda p: jr.twin_band_loglike(k_t[:1], k_s[:1], [p], [1 - p]))
    assert best == pytest.approx(0.3, abs=0.011)
    assert jr.twin_band_loglike(k_t, k_s, [0.3, 0.0], [0.7, 0.0]) == pytest.approx(jr.twin_band_loglike(k_t[:1], k_s[:1], [0.3], [0.7]))
    cfg = jr.TwinBandConfig(enabled=True, colour0_range=(0.85, 2.0), twin_dm=(-0.95, -0.55), single_dm=(-0.25, 0.25), distance_edges_kpc=(0, 0.2))
    tw, si = jr.twin_band_masks([1.5, 1.5, 1.5, 2.5], [6.2, 7.0, 6.6, 9.0], RIDGE, cfg)
    assert tw.tolist() == [True, False, False, False] and si.tolist() == [False, True, False, False]


@pytest.mark.unit
def test_photocentre_size_and_co_observables() -> None:
    assert jr.photocentre_size([2.0], [1.0], [365.25])[0] == pytest.approx(2.0)
    casc = np.zeros((2, 23))
    casc[:, 0], casc[:, 2], casc[:, 10], casc[:, 14] = [2.0, 0.0], [0.0, 4.0], [500.0, 900.0], [0.3, 0.0]
    o = jr.co_observables({"phot_g_mean_mag": np.array([12.0, 13.0]), "sys_evolved": np.array([False, True])},
                          {"cascade": casc, "accepted_orbital": np.array([True, True]), "published_acceleration": np.array([False, True])}, 830.4)
    assert o["in_orbit"].tolist() == [True, False] and o["in_accel"].tolist() == [False, True]
    assert o["d_orbit"][0] == 0.5 and o["d_accel"][1] == 0.25 and o["cls"].tolist() == [0.0, 1.0]


WDT = jr.WDTargetConfig(
    enabled=True, m1_pivot_msun=1.0, ifmr_segments=((0.83, 2.85, 0.080, 0.489), (2.85, 3.60, 0.187, 0.184), (3.60, 7.20, 0.107, 0.471)),
    ms_lifetime_gyr_at_1msun=10.0, ms_lifetime_mass_exponent=-2.5, log_p_ni_min=3.0, log_p_max=8.0, m_wd_min=0.15, m_wd_max=1.4,
    he_max_msun=0.46, ts99_a=4.5, ts99_b_days=1.2e5, ts99_c_msun=0.12, ts99_ref="TS99", he_sigma_log_p=0.2, co_mt_mu_msun=0.6,
    co_mt_sigma_msun=0.1, co_mt_mu_log_p=2.5, co_mt_sigma_log_p=0.35, co_mt_ref="El-Badry+2024", e_mt_max=0.2,
    circular_period_days=2.0, e_max_period_scale_days=2.0, e_max_exponent=-2.0 / 3.0)


@pytest.mark.physics
def test_wd_target_branches_normalize_and_respect_the_ifmr_floor() -> None:
    lm = np.linspace(math.log10(0.15), math.log10(1.4), 241)
    lpg = np.linspace(0.2, 8.0, 391)
    eg = np.linspace(0.0, 1.0, 201)[:-1] + 0.0025
    M, LP, E = np.meshgrid(10**lm, lpg, eg, indexing="ij")
    tots = {}
    for f_mt, f_he in ((0.0, 0.5), (1.0, 1.0), (1.0, 0.0), (0.4, 0.3)):
        l = jr.wd_target_log_density(M.ravel(), 10 ** LP.ravel(), E.ravel(), np.full(M.size, 0.9), np.full(M.size, 8.0),
                                     ln_r_wd=0.0, beta=0.0, f_mt=f_mt, f_he=f_he, alpha=2.35, cfg=WDT).reshape(M.shape)
        tots[(f_mt, f_he)] = np.trapz(np.trapz(np.trapz(np.exp(l), eg, axis=2), lpg, axis=1), lm)
    assert tots[(0.0, 0.5)] == pytest.approx(1.0, abs=0.03) and tots[(1.0, 0.0)] == pytest.approx(1.0, abs=0.03)
    # He WDs below ~0.2 Msun lie at P < 10^0.2 d on TS99 (outside the astrometric domain): that part of the branch is cut
    assert 0.75 < tots[(1.0, 1.0)] < 0.85
    assert tots[(0.4, 0.3)] == pytest.approx(0.6 * tots[(0.0, 0.5)] + 0.4 * (0.3 * tots[(1.0, 1.0)] + 0.7 * tots[(1.0, 0.0)]), rel=1e-6)
    # IFMR floor: an older primary (M1 2.0, young system) forbids WDs below IFMR(2.0) = 0.649 on the non-interacting branch
    l = jr.wd_target_log_density([0.6, 0.7], [1e4, 1e4], [0.3, 0.3], [2.0, 2.0], [1.0, 1.0], ln_r_wd=0.0, beta=0.0, f_mt=0.0, f_he=0.5, alpha=2.35, cfg=WDT)
    assert l[0] == -np.inf and np.isfinite(l[1])
    # He WDs follow TS99: 0.30 Msun sits near 54 d
    assert 10 ** jr.ts99_log_period(0.30, WDT) == pytest.approx(1.2e5 * 0.18**4.5, rel=1e-9)
    l = jr.wd_target_log_density([0.3, 0.3], [54.0, 5000.0], [0.05, 0.05], [1.0, 1.0], [8.0, 8.0], ln_r_wd=0.0, beta=0.0, f_mt=1.0, f_he=1.0, alpha=2.35, cfg=WDT)
    assert l[0] > l[1] + 10
    assert jr.wd_target_log_density([0.6], [1e4], [0.3], [2.0], [5.0], ln_r_wd=0.5, beta=1.0, f_mt=0.0, f_he=0.5, alpha=2.35, cfg=WDT)[0] == pytest.approx(
        0.5 + math.log(2.0) + jr.wd_target_log_density([0.6], [1e4], [0.3], [2.0], [5.0], ln_r_wd=0.0, beta=0.0, f_mt=0.0, f_he=0.5, alpha=2.35, cfg=WDT)[0])
