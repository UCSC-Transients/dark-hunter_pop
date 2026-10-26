"""Rung 4 compact-object machinery (#552, docs/MOCK_POPULATION_SPEC.md §13).

Proposal / target / weights / SBC partition / version guard, WD light, selection-function table, cascade
covariance and mass recovery. gaiamock is replaced by small fakes; nothing here runs the overlay or reads data/.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps

CO_FRAGMENT = "config/population/proposal_set_co_v1_smoke.yaml"
LUMINOUS = "config/population/proposal_set_single45.yaml"
RUNG4 = "config/population/rung4_co_v1.yaml"
RECENTRED = "config/population/proposal_set_co_v1_recentred1001.yaml"


@pytest.fixture(scope="module")
def frag() -> ps.ProposalSetFragment:
    return ps.load_proposal_set_fragment(CO_FRAGMENT)


def _parent(n: int = 60, seed: int = 0) -> ps.ParentSnapshot:
    rng = np.random.default_rng(seed)
    cols = {
        "source_id": np.arange(n, dtype=np.int64) + 7000,
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 60, n),
        "parallax": rng.uniform(0.3, 20.0, n), "pmra": rng.normal(0, 5, n), "pmdec": rng.normal(0, 5, n),
        "phot_g_mean_mag": rng.uniform(9, 18, n),
    }
    usable = np.ones(n, dtype=bool)
    usable[0] = False
    m1 = rng.uniform(0.6, 1.6, n)
    iso = {"log_m1_mean": np.log10(m1), "log_m1_sigma": np.full(n, 0.002)}
    return ps.ParentSnapshot(
        columns=cols, m1_msun=m1, m1_source=np.full(n, "MIST"), atmosphere_logg=np.full(n, 4.4),
        truth_parallax_mas=cols["parallax"].copy(), is_giant=rng.uniform(size=n) < 0.2, flags={"all": usable},
        usable=usable, meta={"parent_h5_sha256": "x"}, scale_to_full=500.0, path=Path("."), isochrone=iso,
    )


# ---------------------------------------------------------------------------
# Config, version guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_co_fragment_version_and_mixing_guard(frag: ps.ProposalSetFragment) -> None:
    lum = ps.load_proposal_set_fragment(LUMINOUS).proposal
    v = ps.primary_model_version(frag.proposal)
    assert v == ps.primary_model_version(lum) + ":co_v1"
    assert ps.is_compact_object_version(v) and not ps.is_compact_object_version(ps.primary_model_version(lum))
    assert ps.check_single_primary_model([frag.proposal, frag.proposal]) == v
    for allow in (False, True):  # spec §13.2: never mixed, not even as a labelled diagnostic
        with pytest.raises(ValueError, match="compact-object and luminous"):
            ps.check_single_primary_model([frag.proposal, lum], allow_mixed=allow)
    other = frag.proposal.model_copy(update={"compact_object": frag.proposal.compact_object.model_copy(update={"version": "co_v2"})})  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        ps.check_single_primary_model([frag.proposal, other])
    with pytest.raises(ValueError):
        cob.check_co_generations([lum])


@pytest.mark.unit
def test_luminous_fingerprints_unchanged() -> None:
    """Adding ``compact_object`` must not move any luminous generation's artifact key."""
    assert ps.fragment_fingerprint(ps.load_proposal_set_fragment(LUMINOUS)) == "6654f9b916af"
    assert ps.fragment_fingerprint(ps.load_proposal_set_fragment("config/population/proposal_set_pilot.yaml")) == "8424ca42465a"


@pytest.mark.unit
def test_co_config_validation(frag: ps.ProposalSetFragment) -> None:
    raw = frag.model_dump(mode="json")
    bad = {**raw, "target_co": {**raw["target_co"], "version": "co_v9"}}
    with pytest.raises(ValueError, match="version"):
        ps.ProposalSetFragment.model_validate(bad)
    with pytest.raises(ValueError, match="target_co is required"):
        ps.ProposalSetFragment.model_validate({**raw, "target_co": None})
    co = raw["proposal"]["compact_object"]
    with pytest.raises(ValueError, match="M_Ch"):
        ps.CompactObjectProposalConfig.model_validate({**co, "wd": {**co["wd"], "m2_max_msun": 1.5}})
    with pytest.raises(ValueError, match="M_Ch"):
        ps.CompactObjectTargetConfig.model_validate({**raw["target_co"], "wd": {**raw["target_co"]["wd"], "m2_max_msun": 1.45}})
    with pytest.raises(ValueError, match="weight"):
        ps.CompactObjectProposalConfig.model_validate({**co, "wd": {**co["wd"], "weight": 0.5}})
    shape = ps.load_proposal_set_fragment(LUMINOUS).proposal.m2_p_shape
    with pytest.raises(ValueError, match="m2_p_shape"):
        ps.ProposalConfig.model_validate({**raw["proposal"], "m2_p_shape": shape.model_dump(mode="json")})  # type: ignore[union-attr]
    assert cob.load_rung4_config(RUNG4).version == frag.proposal.compact_object.version  # type: ignore[union-attr]


@pytest.mark.unit
def test_co_fragment_carries_decided_settings() -> None:
    from darkhunter_pop.decided_settings import check_launch

    assert check_launch(CO_FRAGMENT) == []


# ---------------------------------------------------------------------------
# Proposal, SBC partition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_co_sampler_reproducible_ranges_and_logq(frag: ps.ProposalSetFragment) -> None:
    parent = _parent()
    cfg = frag.proposal.model_copy(update={"n_draws": 3000})
    a = cob.sample_co_proposal(parent, cfg, draw_index_offset=100_000_000)
    b = cob.sample_co_proposal(parent, cfg, draw_index_offset=100_000_000)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["draw_index"][0] == 100_000_000 and not np.any(a["parent_row"] == 0)
    co = cfg.compact_object
    wd = a["co_component"] == cob.COMPONENT_CODES["wd"]
    assert 0.35 < wd.mean() < 0.45
    assert np.all((a["m2_msun"][wd] >= co.wd.m2_min_msun) & (a["m2_msun"][wd] <= co.wd.m2_max_msun))  # type: ignore[union-attr]
    assert np.all((a["m2_msun"][~wd] >= co.dark.m2_min_msun) & (a["m2_msun"][~wd] <= co.dark.m2_max_msun))  # type: ignore[union-attr]
    assert np.all(a["flux_ratio"][~wd] == 0.0) and np.all(a["is_dark"] == ~wd) and np.all(np.isnan(a["flux_ratio"][wd]))
    assert np.all(np.isfinite(a["wd_cooling_age_gyr"][wd])) and np.all(np.isnan(a["wd_cooling_age_gyr"][~wd]))
    assert np.all(a["eccentricity"][a["period_days"] <= 2.0] == 0.0)
    assert np.all(np.isfinite(a["log_q_total"]))
    np.testing.assert_allclose(cob.log_q_co_total_for(a, parent, cfg), a["log_q_total"])
    core = (np.log10(a["period_days"]) > 1.5) & (np.log10(a["period_days"]) < 3.5)
    assert core.mean() > 0.6  # core 1.5-3.5 at weight 0.5 + the defensive part's share


@pytest.mark.physics
def test_co_proposal_marginals_normalize(frag: ps.ProposalSetFragment) -> None:
    cfg = frag.proposal
    lm = np.linspace(-1.5, 2.0, 400_001)
    for name, code in cob.COMPONENT_CODES.items():
        q = np.exp(cob.log_q_co_m2(np.full(lm.size, code), 10.0**lm, cfg))
        assert np.trapz(q, lm) == pytest.approx(getattr(cfg.compact_object, name).weight, abs=1e-4)
    lt = np.linspace(-4, 2, 400_001)
    q = np.exp(cob.log_q_co_cooling_age(np.zeros(lt.size), 10.0**lt, cfg))
    assert np.trapz(q, lt) == pytest.approx(1.0, abs=1e-4)
    assert np.all(cob.log_q_co_cooling_age([1, 1], [np.nan, 3.0], cfg) == 0.0)  # dark draws: no such dimension


@pytest.mark.unit
def test_holdout_mask_fraction_and_determinism() -> None:
    idx = np.arange(100_000_000, 100_200_000)
    m = cob.holdout_mask(idx, 0.2, 552)
    assert m.mean() == pytest.approx(0.2, abs=0.005)
    np.testing.assert_array_equal(cob.holdout_mask(idx[::-1], 0.2, 552), m[::-1])  # per draw, order-free
    assert not np.array_equal(cob.holdout_mask(idx, 0.2, 553), m)
    assert np.all(cob.holdout_mask(idx, 0.1, 552) <= m)  # nested in the fraction


# ---------------------------------------------------------------------------
# Target and weights
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_co_target_support(frag: ps.ProposalSetFragment) -> None:
    tg = frag.target_co
    base = {"co_component": np.array([1, 1, 1, 1, 0, 0]), "m2_msun": np.array([5.0, 50.0, 5.0, 5.0, 0.3, 0.6]),
            "period_days": np.array([300.0, 300.0, 1.8, 300.0, 300.0, 300.0]), "eccentricity": np.array([0.2, 0.2, 0.0, 0.999, 0.2, 0.2]),
            "m1_msun": np.ones(6), "wd_cooling_age_gyr": np.array([np.nan] * 4 + [2.0, 9.9]), "iso_log_age": np.full(6, 10.0)}
    lt = cob.co_log_target(base, tg)  # type: ignore[arg-type]
    assert np.isfinite(lt[0]) and np.isfinite(lt[2]) and np.isfinite(lt[4])
    assert lt[1] == -np.inf  # dark M2 above 40
    assert lt[3] == -np.inf  # e above e_max(P)
    assert lt[5] == -np.inf  # 0.6 Msun WD: progenitor ~1.4 Msun lives ~4 Gyr, so t_cool 9.9 > 10 - 4 Gyr


@pytest.mark.physics
def test_wd_cooling_age_target_normalizes(frag: ps.ProposalSetFragment) -> None:
    tc = frag.target_co.wd_cooling_age  # type: ignore[union-attr]
    lt = np.linspace(-6, 1.5, 300_001)
    for m_wd in (0.3, 0.7, 1.2):
        d = {"wd_cooling_age_gyr": 10.0**lt, "iso_log_age": np.full(lt.size, 10.0), "m2_msun": np.full(lt.size, m_wd)}
        assert np.trapz(np.exp(cob.wd_log_cooling_age_target(d, tc)), lt) == pytest.approx(1.0, abs=1e-3)
    mi = cob.ifmr_initial_mass([0.3, 0.717, 1.3], tc.ifmr_segments)
    assert mi[1] == pytest.approx(2.85, abs=0.01) and mi[2] == pytest.approx(7.2)


@pytest.mark.physics
def test_co_weights_recover_target_normalization(frag: ps.ProposalSetFragment) -> None:
    """Σ w over dark draws estimates scale × Σ_rows ∫ λ = scale × n_usable × rate (all target factors normalized),
    for the training and held-out partitions alike (spec §3.4, §3.7)."""
    parent = _parent(n=40, seed=3)
    cfg = frag.proposal.model_copy(update={"n_draws": 40_000})
    truth = cob.sample_co_proposal(parent, cfg, draw_index_offset=100_000_000)
    truth["iso_log_age"] = np.full(truth["draw_index"].size, 10.0)
    tg = frag.target_co
    expect = parent.scale_to_full * int(parent.usable.sum()) * tg.dark.rate_per_primary  # type: ignore[union-attr]
    dark = truth["co_component"] == cob.COMPONENT_CODES["dark"]
    for part in ("train", "holdout", "all"):
        w = cob.co_importance_weights(truth, parent, [cfg], tg, partition=part)  # type: ignore[arg-type]
        assert np.sum(w[dark]) == pytest.approx(expect, rel=0.05)
    w2 = cob.co_importance_weights(truth, parent, [cfg, cfg.model_copy(update={"generation": 1001})], tg, partition="all")  # type: ignore[arg-type]
    np.testing.assert_allclose(w2, cob.co_importance_weights(truth, parent, [cfg], tg, partition="all"))  # type: ignore[arg-type]  # n_j = 0 for an absent generation
    with pytest.raises(ValueError):
        cob.co_importance_weights(truth, parent, [cfg], tg.model_copy(update={"version": "co_v2"}))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# WD light
# ---------------------------------------------------------------------------


def _tiny_grid(tmp_path: Path) -> tuple[Path, str]:
    rows = ["mass_msun,age_gyr,Mg,Mbp,Mrp,teff_k,logg"]
    for m, off in ((0.4, 0.0), (0.8, 1.0)):
        for age, mg in ((0.0, 9.0), (0.01, 10.0), (1.0, 12.0), (10.0, 15.0)):
            rows.append(f"{m},{age},{mg + off},{mg + off + 0.1},{mg + off - 0.3},{20000 / (1 + age)},8.0")
    d = tmp_path / "wd"
    d.mkdir()
    (d / "tracks.csv").write_text("\n".join(rows) + "\n")
    return d, hashlib.sha256((d / "tracks.csv").read_bytes()).hexdigest()


@pytest.mark.unit
def test_wd_cooling_grid_interpolation(tmp_path: Path) -> None:
    d, sha = _tiny_grid(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        cob.load_wd_cooling_grid(d, "0" * 64)
    g = cob.load_wd_cooling_grid(d, sha)
    out = g.abs_mags([0.4, 0.8, 0.6, 0.1, 1.4, 0.4, np.nan], [1.0, 1.0, 1.0, 1.0, 1.0, math.sqrt(0.01 * 1.0), 1.0], 1e5)
    assert out[0, 0] == pytest.approx(12.0) and out[1, 0] == pytest.approx(13.0) and out[2, 0] == pytest.approx(12.5)
    assert out[3, 0] == pytest.approx(12.0) and out[4, 0] == pytest.approx(13.0)  # masses clamped to the grid
    assert out[5, 0] == pytest.approx(11.0)  # linear in log age between 10 Myr and 1 Gyr
    assert out[0, 1] - out[0, 2] == pytest.approx(0.4) and np.all(np.isnan(out[6]))
    young = g.abs_mags([0.6], [1e-9], 1e5)  # below the age floor -> evaluated at the 1e5 yr floor
    assert young[0, 0] == pytest.approx(9.0 + 5.0 / 7.0 + 0.5)  # age 0 sits at log age 0; 10 Myr at 7


@pytest.mark.unit
def test_wd_companion_light_sets_flux_ratio_and_colour() -> None:
    from darkhunter_pop import isochrone_mass as im

    n_iso = 50
    mass = np.linspace(0.3, 1.5, n_iso)
    iso = {"star_mass": mass, "mg": 10.0 - 4.0 * (mass - 0.3), "bp": 10.6 - 4.4 * (mass - 0.3), "rp": 9.4 - 3.6 * (mass - 0.3),
           "phase": np.zeros(n_iso), "log_teff": np.linspace(3.5, 3.9, n_iso)}
    c0, g0 = np.array([0.8, 0.8, 0.8]), np.array([6.0, 6.0, 6.0])
    one = np.ones(3)
    cam = (np.array([11.0, np.nan, 3.0]), np.array([10.9, np.nan, 2.8]), np.array([11.3, np.nan, 3.5]))
    pr = im.primary_and_system_photometry(iso, c0, g0, 0.05 * one, 0.05 * one, np.array([0.5, 0.5, 0.5]),
                                         np.array([-np.inf, -np.inf, -np.inf]), np.zeros(3, bool), companion_abs_mags=cam)
    assert pr["log10_f"][0] == pytest.approx(-0.4 * (11.0 - pr["mg1"][0]))
    assert pr["log10_f"][1] == -np.inf and pr["mg_sys"][1] == pytest.approx(pr["mg1"][1])  # dark: primary alone
    assert pr["mg_sys"][2] < pr["mg1"][2] - 2.0  # a hot young WD outshines a faint primary: not dark in G
    assert pr["c_sys"][2] < pr["c1"][2]  # and pulls the system blue
    base = im.primary_and_system_photometry(iso, c0, g0, 0.05 * one, 0.05 * one, np.full(3, 0.5), np.full(3, -1.0), np.zeros(3, bool))
    assert np.all(base["log10_f"] == pytest.approx(-1.0))  # luminous path unchanged without companion_abs_mags


# ---------------------------------------------------------------------------
# Selection-function table
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_selection_table_recovers_known_detection_probability(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 200_000
    x = rng.uniform(0, 2, n)
    y = rng.uniform(0, 3, n)
    p_true = 0.1 + 0.2 * np.floor(x) + 0.1 * np.floor(y)
    w = rng.lognormal(0, 0.5, n)
    d = rng.uniform(size=n) < p_true
    t = cob.build_selection_table({"x": x, "y": y}, w, d, {"x": [0, 1, 2], "y": [0, 1, 2, 3]}, meta={"k": 1})
    expect = 0.1 + 0.2 * np.arange(2)[:, None] + 0.1 * np.arange(3)[None, :]
    sd = np.sqrt(t.var_p_det)
    assert np.all(np.abs(t.p_det - expect) < 4 * sd)
    assert t.n_draws.sum() == n and np.all(t.ess <= t.n_draws) and np.all(t.ess > 0.5 * t.n_draws)
    np.testing.assert_allclose(sd, np.sqrt(expect * (1 - expect) / t.ess), rtol=0.1)  # ratio-estimator variance
    assert np.all(t.mc_poisson_ratio > 0)
    t2 = cob.build_selection_table({"x": x - 1, "y": y}, w, d, {"x": [0, 1], "y": [0, 3]})
    assert t2.n_draws.sum() == int(np.sum((x >= 1) & (x < 2)))  # outside the edges: dropped
    path = cob.write_selection_tables(tmp_path / "t.h5", {"dark": t}, attrs={"version": "co_v1", "artifacts": ["a"]})
    back = cob.read_selection_tables(path)["dark"]
    np.testing.assert_array_equal(back.sum_w_det, t.sum_w_det)
    assert back.axes == ("x", "y") and back.meta == {"k": 1}


@pytest.mark.unit
def test_detection_flags_and_axis_values() -> None:
    o = {"accepted_orbital": np.array([True, False, False]), "published_acceleration": np.array([False, True, False])}
    assert cob.detected_flags(o, "orbit_or_acceleration").tolist() == [True, True, False]
    assert cob.detected_flags(o, "orbit").tolist() == [True, False, False]
    v = cob.selection_axis_values({"m2_msun": np.array([10.0]), "period_days": np.array([100.0]), "eccentricity": np.array([0.3]),
                                   "phot_g_mean_mag": np.array([12.0]), "parallax_mas": np.array([2.0]), "m1_msun": np.array([1.1])})
    assert v["log10_m2_msun"][0] == 1.0 and v["distance_kpc"][0] == 0.5 and v["log10_period_days"][0] == 2.0


# ---------------------------------------------------------------------------
# Cascade covariance and mass recovery
# ---------------------------------------------------------------------------


class _LinearGaiamock:
    """Residuals linear in all 12 parameters: r = (obs − X θ) / err; get_astrometric_chi2 returns the fitted
    linear parameters in gaiamock's order (ra_off, pmra, dec_off, pmdec, plx, B, G, A, F)."""

    def __init__(self, x: np.ndarray, theta: np.ndarray) -> None:
        self.x, self.theta = x, theta

    def get_astrometric_residuals_12par(self, t, psi, pf, obs, err, theta, c_funcs):  # type: ignore[no-untyped-def]
        return (obs - self.x @ np.asarray(theta, float)) / err

    def get_astrometric_chi2(self, t, psi, pf, obs, err, period, phi_p, ecc, c_funcs):  # type: ignore[no-untyped-def]
        th = self.theta
        r = self.get_astrometric_residuals_12par(t, psi, pf, obs, err, th, c_funcs)
        return float(np.sum(r**2)), np.array([th[0], th[3], th[1], th[4], th[2], th[9], th[11], th[8], th[10]])


@pytest.mark.physics
def test_orbit_covariance_matches_linear_least_squares() -> None:
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=(n, 12))
    theta = rng.normal(size=12)
    err = np.full(n, 0.3)
    obs = x @ theta + err * rng.normal(size=n) * 1.3
    gm = _LinearGaiamock(x, theta)
    chi2 = float(np.sum(((obs - x @ theta) / err) ** 2))
    nu = n - 12
    cc = math.sqrt((chi2 / nu) / (1 - 2 / (9 * nu)) ** 3)
    cov_true = np.linalg.inv((x / err[:, None]).T @ (x / err[:, None])) * cc**2
    sig = np.sqrt(np.diag(cov_true))
    cascade = [0.0] * ps.CASCADE_VECTOR_LENGTH
    for k, j in (("parallax", 2), ("A", 8), ("B", 9), ("F", 10), ("G", 11), ("period", 5), ("phi_p", 7), ("ecc", 6)):
        cascade[cob._CASCADE_INDEX[k]] = theta[j]
        cascade[cob._CASCADE_INDEX["sig_" + k]] = sig[j]
    epochs = (np.linspace(0, 3, n), np.zeros(n), np.zeros(n), obs, err)
    oc = cob.orbit_covariance_from_epochs(gm, None, epochs, cascade, step=1e-6)
    np.testing.assert_allclose(oc.p0, theta)
    np.testing.assert_allclose(oc.covariance, cov_true, rtol=1e-5, atol=1e-12)
    assert oc.max_sigma_mismatch < 1e-5
    ps_ = cob.cascade_parameter_set(oc.p0, oc.covariance)
    assert ps_.names == list(ps.ORBIT_PARAMETER_NAMES) and ps_.provenance == cob.CASCADE_COVARIANCE_PROVENANCE
    with pytest.raises(ValueError, match="no diagonal fallback"):
        cob.cascade_parameter_set(oc.p0, np.full((12, 12), np.nan))


def _face_on_a0_mas(m1: float, m2: float, period_days: float, plx_mas: float) -> float:
    a_au = ((m1 + m2) * (period_days / 365.25) ** 2) ** (1 / 3)
    return a_au * m2 / (m1 + m2) * plx_mas


@pytest.mark.physics
def test_recover_masses_unbiased_for_precise_dark_orbit() -> None:
    parent = _parent(n=5)
    row, m2, period, plx = 2, 6.0, 500.0, 3.0
    m1 = float(parent.m1_msun[row])
    a0 = _face_on_a0_mas(m1, m2, period, plx)
    p0 = np.zeros(12)
    p0[[2, 5, 8, 11]] = [plx, period, a0, a0]  # face-on: A = G = a0, B = F = 0
    cov = np.diag(np.r_[np.full(5, 1e-6), 1e-4, 1e-6, 1e-6, np.full(4, (1e-4 * a0) ** 2)])
    cov[2, 2] = (1e-4 * plx) ** 2
    truth = {"draw_index": np.array([10, 11]), "parent_row": np.array([row, row]), "m2_msun": np.array([m2, m2])}
    outcome = {"orbit_p0": np.stack([p0, p0]), "orbit_cov": np.stack([cov, np.full((12, 12), np.nan)])}
    rc = cob.load_rung4_config(RUNG4).mass_recovery
    rec = cob.recover_masses(truth, outcome, parent, rc, select=np.array([True, True]))
    assert rec["m2_med"][0] == pytest.approx(m2, rel=0.01)
    assert 0.05 < rec["m2_pit"][0] < 0.95 and rec["factorization"][0] == 0
    assert np.isnan(rec["m2_med"][1]) and rec["factorization"][1] == -1  # singular: counted, never diagonalized


@pytest.mark.physics
def test_recovery_summary_coverage_for_calibrated_pits() -> None:
    rng = np.random.default_rng(8)
    n = 40_000
    m2 = 10 ** rng.uniform(0.0, 1.6, n)
    rec = {"m2_med": m2 * (1 + 0.01 * rng.normal(size=n)), "m2_pit": rng.uniform(size=n)}
    rc = cob.load_rung4_config(RUNG4).mass_recovery
    rows = cob.recovery_summary(m2, rec, np.ones(n), rc)
    filled = [r for r in rows if r["n"] > 1000]
    assert filled
    for r in filled:
        assert abs(r["median_bias"]) < 0.005
        assert r["coverage_0.68"] == pytest.approx(0.68, abs=0.03) and r["coverage_0.95"] == pytest.approx(0.95, abs=0.02)


@pytest.mark.api
def test_simulate_one_stores_orbit_covariance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frag: ps.ProposalSetFragment) -> None:
    """A compact-object draw keeps its epochs (``run_cascade(keep_epochs=True)``) and stores the full covariance of
    an orbital fit; the artifact carries it."""
    import contextlib
    import dataclasses

    from darkhunter_pop import epoch_model as em
    from darkhunter_pop.config_loader import load_config

    seen: dict[str, Any] = {}
    n = 300
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, 12))
    theta = np.r_[0.0, 0.0, 2.0, 1.0, -1.0, 300.0, 0.2, 0.5, 1.0, 0.1, -0.2, 0.9]
    err = np.full(n, 0.2)
    obs = x @ theta + err * rng.normal(size=n)
    gm = _LinearGaiamock(x, theta)

    def fake_run_cascade(g, cf, predict, config, source, **kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        vec = [theta[2], 0.01, theta[8], 0.01, theta[9], 0.01, theta[10], 0.01, theta[11], 0.01, theta[5], 1.0, theta[7], 0.01,
               theta[6], 0.01, 40.0, 1.2, 0.01, 20, n, 1.0, 3.0]
        return em.CascadeRun(cascade=vec, n_obs=n, n_transits=n // 9, n_visibility_periods=20, ruwe_scale=1.0,
                             epochs=(np.linspace(0, 3, n), np.zeros(n), np.zeros(n), obs, err) if kw.get("keep_epochs") else None)

    monkeypatch.setattr(em, "run_cascade", fake_run_cascade)
    monkeypatch.setattr(ps, "seeded_global_rng", lambda seeds, cf: contextlib.nullcontext())
    cfg_pipe = load_config()
    cuts = cfg_pipe.active_dr().selection_function_astrometric.orbital_solution_cuts
    setup = ps.EpochSetup(config=dataclasses.replace(em.epoch_model_config_from_mapping(cfg_pipe.active_dr().epoch_model), enabled=True), gaps_jd=None)
    draw = {"generation": 1000, "draw_index": 100_000_007, "ra_deg": 10.0, "dec_deg": -5.0, "parallax_mas": 2.0, "pmra_masyr": 0.0,
            "pmdec_masyr": 0.0, "m1_msun": 1.0, "m2_msun": 8.0, "period_days": 300.0, "Tp_days": 10.0, "eccentricity": 0.2,
            "Omega_rad": 1.0, "inc_deg": 60.0, "omega_rad": 2.0, "phot_g_mean_mag": 14.0, "flux_ratio": 0.0, "l_deg": 100.0, "b_deg": 20.0}
    rec = ps.simulate_one(draw, gaiamock=gm, c_funcs=None, cfg=frag.proposal, cuts=cuts, epoch=setup)
    assert seen.get("keep_epochs") is True
    assert rec["solution_type"] in ("twelve_parameter_orbital", "orbital_failed_cuts")
    cov = np.asarray(rec["orbit_cov"]).reshape(12, 12)
    assert np.all(np.isfinite(cov)) and np.allclose(cov, cov.T)
    np.testing.assert_allclose(rec["orbit_p0"], theta)
    skip = ps.simulate_one({**draw, "system_in_cut": False}, gaiamock=gm, c_funcs=None, cfg=frag.proposal, cuts=cuts, epoch=setup)
    assert skip["solution_type"] == "not_in_parent" and np.all(np.isnan(skip["orbit_cov"]))
    truth = {k: np.array([v, v]) for k, v in draw.items()}
    truth["draw_index"] = np.array([100_000_007, 100_000_008])
    skip["draw_index"] = 100_000_008
    path = ps.write_proposal_artifact(tmp_path / "co.h5", truth, [rec, skip], fragment=frag, parent=_parent(), provenance={})
    _, o, attrs = ps.read_proposal_artifact(path)
    assert o["orbit_cov"].shape == (2, 12, 12) and str(attrs["primary_model"]).endswith(":co_v1")


@pytest.mark.unit
def test_contamination_is_a_declared_stand_in(tmp_path: Path) -> None:
    from darkhunter_pop.run_validity import read_stand_ins

    si = cob.contamination_stand_in("co_v1")
    path = cob.write_selection_tables(tmp_path / "t.h5", {}, attrs={"version": "co_v1"})
    from darkhunter_pop.run_validity import write_stand_ins

    write_stand_ins(path, [si])
    got = read_stand_ins(path)
    assert got is not None and [s.name for s in got] == ["rung4_contamination_not_computed"]


# ---------------------------------------------------------------------------
# Re-centred proposal (#552, Ryan 2026-10-24; generation 1001)
# ---------------------------------------------------------------------------


def _parent_rw(n: int = 60, seed: int = 0) -> ps.ParentSnapshot:
    import dataclasses

    p = _parent(n, seed)
    rng = np.random.default_rng(seed + 100)
    iso = {**p.isochrone, "log_age_mean": rng.uniform(9.3, 10.1, n)}  # type: ignore[dict-item]
    rw = np.where(p.columns["phot_g_mean_mag"] < 11.0, 0.05, 1.0)  # §12.30 bright stratum
    return dataclasses.replace(p, isochrone=iso, row_weight=rw)


@pytest.fixture(scope="module")
def rec() -> ps.ProposalSetFragment:
    return ps.load_proposal_set_fragment(RECENTRED)


@pytest.mark.unit
def test_recentred_config_same_version_and_old_fingerprint(frag: ps.ProposalSetFragment, rec: ps.ProposalSetFragment) -> None:
    assert ps.primary_model_version(rec.proposal) == ps.primary_model_version(frag.proposal)
    assert cob.check_co_generations([frag.proposal, rec.proposal]).version == "co_v1"
    assert ps.fragment_fingerprint(frag) == "beba9ac1f609"  # gen 1000 keeps its key
    assert ps.fragment_fingerprint(rec) != ps.fragment_fingerprint(frag)
    assert rec.proposal.compact_object.wd_cooling_age_matched.shape == rec.target_co.wd_cooling_age  # type: ignore[union-attr]
    from darkhunter_pop.decided_settings import check_launch

    assert check_launch(RECENTRED) == []


@pytest.mark.physics
def test_row_weight_share_normalizes_and_matches_target(rec: ps.ProposalSetFragment) -> None:
    par = _parent_rw()
    cfg = rec.proposal.parent
    q = ps.parent_proposal_probabilities(par.columns["parallax"], par.usable, cfg, par.is_giant, par.columns["phot_g_mean_mag"],
                                         par.m1_msun, par.row_weight)
    assert q.sum() == pytest.approx(1.0) and q[0] == 0.0
    base = ps.parent_proposal_probabilities(par.columns["parallax"], par.usable, cfg.model_copy(update={"row_weight_fraction": 0.0}),
                                            par.is_giant, par.columns["phot_g_mean_mag"], par.m1_msun)
    tw = np.where(par.usable, par.row_weight, 0.0)
    np.testing.assert_allclose(q, 0.3 * base + 0.7 * tw / tw.sum())
    ratio = np.where(par.usable, tw / tw.sum() / q, 0.0)  # target row law / q: bounded by 1 / 0.7
    assert ratio.max() <= 1 / 0.7 + 1e-12


@pytest.mark.physics
def test_matched_cooling_age_density_normalizes(rec: ps.ProposalSetFragment) -> None:
    cfg = rec.proposal
    lt = np.linspace(-7, 2, 400_001)
    t = 10.0**lt
    for m2, tsys in ((0.3, 5.0), (0.7, 8.0), (1.2, 0.5), (0.6, 1.0)):  # last: avail <= 0 -> log-uniform only
        lq = cob.log_q_co_cooling_age(np.zeros(lt.size), t, cfg, m2_msun=np.full(lt.size, m2), t_sys_row_gyr=np.full(lt.size, tsys))
        assert np.trapz(np.exp(lq), lt) == pytest.approx(1.0, abs=2e-3)
    with pytest.raises(ValueError, match="m2_msun"):
        cob.log_q_co_cooling_age([0], [1.0], cfg)


@pytest.mark.unit
def test_recentred_sampler_logq_consistent(rec: ps.ProposalSetFragment, frag: ps.ProposalSetFragment) -> None:
    par = _parent_rw()
    cfg = rec.proposal.model_copy(update={"n_draws": 3000})
    a = cob.sample_co_proposal(par, cfg, draw_index_offset=100_002_000)
    np.testing.assert_array_equal(a["m2_msun"], cob.sample_co_proposal(par, cfg, draw_index_offset=100_002_000)["m2_msun"])
    np.testing.assert_allclose(cob.log_q_co_total_for(a, par, cfg), a["log_q_total"])
    assert np.all(np.isfinite(a["log_q_total"]))
    wd = a["co_component"] == 0
    av = cob.wd_available_time_gyr(a["m2_msun"], cob.row_system_age_gyr(par, a["parent_row"]), cfg.compact_object.wd_cooling_age_matched.shape)  # type: ignore[union-attr]
    assert np.mean(a["wd_cooling_age_gyr"][wd] < np.maximum(av[wd], 0)) > 0.6  # the matched share lands inside [0, avail]
    # gen 1000 evaluated at gen-1001 draws (and vice versa) is finite where its support covers them
    g0 = frag.proposal.model_copy(update={"n_draws": 3000})
    b = cob.sample_co_proposal(par, g0, draw_index_offset=100_000_000)
    assert np.all(np.isfinite(cob.log_q_co_total_for(b, par, cfg)))


@pytest.mark.physics
def test_mixture_weights_across_gens_1000_1001(frag: ps.ProposalSetFragment, rec: ps.ProposalSetFragment) -> None:
    """Deterministic mixture over gens 1000 + 1001 keeps Σw unbiased, and the re-centred draws carry a higher ESS
    fraction for WDs than the gen-1000 draws (the cooling-age and row laws now match the target)."""
    par = _parent_rw(n=40, seed=3)
    g0 = frag.proposal.model_copy(update={"n_draws": 20_000})
    g1 = rec.proposal.model_copy(update={"n_draws": 20_000})
    a = cob.sample_co_proposal(par, g0, draw_index_offset=100_000_000)
    b = cob.sample_co_proposal(par, g1, draw_index_offset=100_020_000)
    truth = {k: np.concatenate([a[k], b[k]]) for k in a if k in b}
    truth["iso_log_age"] = np.asarray(par.isochrone["log_age_mean"], float)[truth["parent_row"]]  # type: ignore[index]
    tg = frag.target_co
    w = cob.co_importance_weights(truth, par, [g0, g1], tg, partition="all")  # type: ignore[arg-type]
    dark = truth["co_component"] == 1
    expect = par.scale_to_full * float(np.sum(np.where(par.usable, par.row_weight, 0.0))) * tg.dark.rate_per_primary  # type: ignore[union-attr]
    assert np.sum(w[dark]) == pytest.approx(expect, rel=0.05)
    gen = truth["generation"]
    ess_frac = {}
    for g, cfg in ((1000, g0), (1001, g1)):
        sub = {k: v[gen == g] for k, v in truth.items()}
        ws = cob.co_importance_weights(sub, par, [cfg], tg, partition="all")  # type: ignore[arg-type]
        wd = sub["co_component"] == 0
        ess_frac[g] = ps.kish_ess(ws[wd]) / wd.sum()
    assert ess_frac[1001] > 1.5 * ess_frac[1000]


# ---------------------------------------------------------------------------
# #557: M2 posterior prior and gate 4b
# ---------------------------------------------------------------------------


def _ti(a: float, w: float, om: float, i: float) -> np.ndarray:
    c = math.cos(i)
    return np.array([a * (math.cos(w) * math.cos(om) - math.sin(w) * math.sin(om) * c),
                     a * (math.cos(w) * math.sin(om) + math.sin(w) * math.cos(om) * c),
                     -a * (math.sin(w) * math.cos(om) + math.cos(w) * math.sin(om) * c),
                     -a * (math.sin(w) * math.sin(om) - math.cos(w) * math.cos(om) * c)])


@pytest.mark.physics
def test_thiele_innes_jacobian_is_a_cubed_sin_cubed() -> None:
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = np.array([rng.uniform(0.5, 3), rng.uniform(0, 6), rng.uniform(0, 6), rng.uniform(0.1, 3)])
        jac = np.zeros((4, 4))
        for k in range(4):
            d = np.zeros(4)
            d[k] = 1e-6
            jac[:, k] = (_ti(*(p + d)) - _ti(*(p - d))) / 2e-6
        assert abs(np.linalg.det(jac)) == pytest.approx(p[0] ** 3 * math.sin(p[3]) ** 3, rel=1e-5)


@pytest.mark.physics
def test_isotropic_log_m2_prior_calibrates_coverage_flat_does_not() -> None:
    """SBC on the M2 posterior: truths from isotropic orientation and log-uniform M2, Gaussian TI noise at a0/σ ≈ 8.
    The prior-weighted posterior covers at the nominal rate and is unbiased in PIT; the flat-TI one is pushed high."""
    rng = np.random.default_rng(7)
    m1, period, plx = 1.0, 400.0, 2.0
    names = list(ps.ORBIT_PARAMETER_NAMES)
    pits = {"flat_thiele_innes": [], "isotropic_log_m2": []}
    for _ in range(300):
        m2 = 10 ** rng.uniform(0.3, 1.3)
        a0 = ((m1 + m2) * (period / 365.25) ** 2) ** (1 / 3) * m2 / (m1 + m2) * plx
        ti = _ti(a0, rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), math.acos(rng.uniform(-1, 1)))
        sig = a0 / 8.0
        p0 = np.zeros(12)
        p0[names.index("parallax")], p0[names.index("period")] = plx, period
        for j, k in enumerate(("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")):
            p0[names.index(k)] = ti[j] + sig * rng.standard_normal()
        cov = np.diag(np.full(12, 1e-10))
        for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"):
            cov[names.index(k), names.index(k)] = sig**2
        seed = int(rng.integers(2**31))
        for prior in pits:
            _, m2d, wq, _ = cob.m2_posterior_draws(p0, cov, np.full(4000, m1), n_draws=4000, random_seed=seed, flux_ratio=0.0, prior=prior)
            ok = np.isfinite(m2d) & (wq > 0)
            pits[prior].append(np.sum(wq[ok][m2d[ok] < m2]) / wq[ok].sum())
    iso = np.asarray(pits["isotropic_log_m2"])
    flat = np.asarray(pits["flat_thiele_innes"])
    assert np.mean(np.abs(iso - 0.5) < 0.34) == pytest.approx(0.68, abs=0.07)
    assert np.mean(np.abs(iso - 0.5) < 0.475) == pytest.approx(0.95, abs=0.04)
    assert abs(np.mean(iso < 0.16) - np.mean(iso > 0.84)) < 0.08
    assert np.mean(flat < 0.16) - np.mean(flat > 0.84) > 0.08  # flat TI: truth in the lower tail too often
    with pytest.raises(ValueError, match="unknown"):
        cob.m2_posterior_draws(p0, cov, np.ones(10), n_draws=10, random_seed=1, flux_ratio=0.0, prior="nope")


@pytest.mark.unit
def test_gate_4b_pointwise_and_chi2() -> None:
    rc = cob.load_rung4_config(RUNG4).mass_recovery
    assert rc.m2_posterior_prior == "isotropic_log_m2" and rc.gate_period_max_days == pytest.approx(830.4)
    good = [{"ess": 400.0, "median_bias": 0.01, "coverage_0.68": 0.69, "coverage_0.68_mc_sigma": 0.023,
             "coverage_0.95": 0.94, "coverage_0.95_mc_sigma": 0.011}] * 3
    g = cob.gate_4b(good, rc, min_ess=30)
    assert g["passed_pointwise"] and g["bins_used"] == 3 and g["coverage_0.68"]["p_value"] > 0.5
    bad = good + [{**good[0], "coverage_0.68": 0.55}]
    assert not cob.gate_4b(bad, rc, min_ess=30)["passed_pointwise"]
    assert cob.gate_4b(bad, rc, min_ess=500)["bins_used"] == 0 and not cob.gate_4b(bad, rc, min_ess=500)["passed_pointwise"]


# ---------------------------------------------------------------------------
# #556: matched WD cooling age at the draw's posterior-draw age (gen 1003)
# ---------------------------------------------------------------------------

WD1003 = "config/population/proposal_set_co_v1_wd1003.yaml"


@pytest.mark.unit
def test_wd1003_config_and_older_fingerprints() -> None:
    f3 = ps.load_proposal_set_fragment(WD1003)
    assert f3.proposal.compact_object.wd_cooling_age_matched.age_source == "draw"  # type: ignore[union-attr]
    assert ps.fragment_fingerprint(ps.load_proposal_set_fragment(RECENTRED)) == "f075282844af"
    assert ps.fragment_fingerprint(ps.load_proposal_set_fragment("config/population/proposal_set_co_v1_batch1002.yaml")) == "217b5182ebc1"
    assert cob.check_co_generations([ps.load_proposal_set_fragment(CO_FRAGMENT).proposal, f3.proposal]).version == "co_v1"
    from darkhunter_pop.decided_settings import check_launch

    assert check_launch(WD1003) == []


@pytest.mark.unit
def test_draw_age_sampler_matches_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampler draws the MP-Q35 posterior age first (as apply_system_photometry will) and its stored log q equals the
    density re-evaluated at the stored ``iso_log_age``; a row-mean generation evaluated at the same draws ignores it."""
    par = _parent_rw()
    f3 = ps.load_proposal_set_fragment(WD1003).proposal.model_copy(update={"n_draws": 3000})
    ages = {}

    def fake_posterior_draws(rows, parent, config, cfg, **kw):  # type: ignore[no-untyped-def]
        r = np.asarray(rows)
        la = 9.0 + 0.9 * np.sin(r * 1.7 + cfg.generation)  # deterministic per (row, generation), like the real one
        ages["log_age"] = la
        return {"log_age": la}, None, None, None, None

    monkeypatch.setattr(ps, "posterior_draws", fake_posterior_draws)
    with pytest.raises(ValueError, match="PipelineConfig"):
        cob.sample_co_proposal(par, f3)
    a = cob.sample_co_proposal(par, f3, draw_index_offset=100_410_000, config=object())
    a["iso_log_age"] = ages["log_age"]
    np.testing.assert_allclose(cob.log_q_co_total_for(a, par, f3), a["log_q_total"])
    wd = a["co_component"] == 0
    assert 0.65 < wd.mean() < 0.75
    av = cob.wd_available_time_gyr(a["m2_msun"], 10 ** a["iso_log_age"] / 1e9, f3.compact_object.wd_cooling_age_matched.shape)  # type: ignore[union-attr]
    inside = wd & (a["wd_cooling_age_gyr"] < np.maximum(av, 0))
    assert inside.sum() / wd.sum() > 0.6
    with pytest.raises(ValueError, match="iso_log_age"):
        cob.log_q_co_total_for({k: v for k, v in a.items() if k != "iso_log_age"}, par, f3)
    g1 = ps.load_proposal_set_fragment(RECENTRED).proposal
    np.testing.assert_allclose(cob.log_q_co_total_for(a, par, g1), cob.log_q_co_total_for({**a, "iso_log_age": a["iso_log_age"] + 0.3}, par, g1))


@pytest.mark.physics
def test_draw_age_matching_bounds_wd_cooling_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the target's own age in the matched share, the WD cooling-age weight ratio p/q is bounded by 1/0.7 where
    the target has support (the gen-1002 tail came from t > t_avail at the row-mean age)."""
    par = _parent_rw()
    f3 = ps.load_proposal_set_fragment(WD1003)
    monkeypatch.setattr(ps, "posterior_draws", lambda rows, parent, config, cfg, **kw: ({"log_age": 9.0 + 0.9 * np.cos(np.asarray(rows) * 0.3)}, None, None, None, None))
    cfg = f3.proposal.model_copy(update={"n_draws": 5000})
    a = cob.sample_co_proposal(par, cfg, draw_index_offset=100_410_000, config=object())
    a["iso_log_age"] = 9.0 + 0.9 * np.cos(a["parent_row"] * 0.3)
    wd = a["co_component"] == 0
    lt = cob.wd_log_cooling_age_target({k: a[k][wd] for k in ("wd_cooling_age_gyr", "iso_log_age", "m2_msun")}, f3.target_co.wd_cooling_age)  # type: ignore[union-attr]
    lq = a["log_q_co_cooling_age"][wd]
    ok = np.isfinite(lt)
    assert ok.mean() > 0.5 and np.max(lt[ok] - lq[ok]) <= math.log(1 / 0.7) + 1e-9
