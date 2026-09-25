"""Monte Carlo mass-function propagation (CONTINUATION_PLAN §5, §11, §13)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.mc_mass_function import (
    CovarianceFactorization,
    MassFunctionDraws,
    UnhandledCovarianceModeError,
    ensemble_row_quantities,
    factorize_covariance,
    format_m2_posterior_convergence_report,
    manifest_seed_record,
    propagate_nss_solution,
    record_seeds_on_manifest,
    run_m2_posterior_convergence,
    sample_multivariate_normal,
    sample_primary_mass_draws,
    synthetic_orbital_solution,
)
from darkhunter_pop.physics_utils import invert_astrometric_companion_mass
from darkhunter_pop.schemas import ActiveDRMode, ParameterSet, RunManifest

pytestmark = pytest.mark.physics


def test_cholesky_factor_recovers_covariance() -> None:
    cov = np.array([[4.0, 1.2], [1.2, 1.0]])
    factor = factorize_covariance(cov, eig_rel_floor=1e-12, eig_abs_floor=1e-18)
    assert factor.method is CovarianceFactorization.CHOLESKY
    reconstructed = factor.factor @ factor.factor.T
    assert reconstructed == pytest.approx(cov)


def test_rank_deficient_uses_eigen_clip() -> None:
    cov = np.diag([1.0, 0.5, 0.0, 0.2])
    factor = factorize_covariance(cov, eig_rel_floor=1e-12, eig_abs_floor=1e-18)
    assert factor.method in {
        CovarianceFactorization.CHOLESKY_NUGGET,
        CovarianceFactorization.EIGEN_CLIP,
    }
    rng = np.random.default_rng(0)
    draws = sample_multivariate_normal(np.zeros(4), factor, 2000, rng)
    empirical = np.cov(draws, rowvar=False)
    assert empirical[2, 2] == pytest.approx(0.0, abs=0.02)
    assert empirical[0, 0] == pytest.approx(1.0, rel=0.15)


def test_propagate_matches_mean_inversion() -> None:
    solution = synthetic_orbital_solution(relative_error=1e-8, seed=1)
    draws = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=32, random_seed=7, source_id=11
    )
    assert draws.factorization is CovarianceFactorization.CHOLESKY
    mean_m2 = invert_astrometric_companion_mass(
        1.0, float(np.mean(draws.m_f_msun[np.isfinite(draws.m_f_msun)])), 0.0
    )
    assert draws.m2_mean() == pytest.approx(mean_m2, rel=0.05)
    assert draws.n_valid == draws.n_draws


def test_probability_and_snr_are_ensemble_not_linear() -> None:
    solution = synthetic_orbital_solution(relative_error=0.04, seed=2)
    draws = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=400, random_seed=3
    )
    row = ensemble_row_quantities(draws, m2_threshold_msun=1.4)
    p = draws.probability_m2_above(1.4)
    assert row["p_m2_above"] == pytest.approx(p)
    assert row["m2_snr"] == pytest.approx(draws.m2_mean() / draws.m2_std())
    assert 0.0 <= p <= 1.0


def test_rank_deficient_12x12_propagates() -> None:
    solution = synthetic_orbital_solution(relative_error=0.03, rank_deficient=True, seed=4)
    evals = np.linalg.eigvalsh(solution.covariance_array())
    assert float(np.min(evals)) <= 1.0e-10
    draws = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=64, random_seed=5, source_id=1
    )
    assert draws.factorization in {
        CovarianceFactorization.CHOLESKY,
        CovarianceFactorization.CHOLESKY_NUGGET,
        CovarianceFactorization.EIGEN_CLIP,
    }
    assert draws.n_valid > 0


def test_rejects_diagonal_covariance_mode() -> None:
    solution = synthetic_orbital_solution()
    with pytest.raises(UnhandledCovarianceModeError):
        propagate_nss_solution(
            solution,
            m1_msun=1.0,
            n_draws=8,
            random_seed=0,
            covariance_mode="diagonal",
        )


def test_manifest_records_seed() -> None:
    now = datetime(2026, 9, 2, tzinfo=timezone.utc)
    manifest = RunManifest(
        run_id="20260902-000000-abcdef0",
        created_at=now,
        config_checksum="abc",
        active_dr_mode=ActiveDRMode.DR3,
    )
    updated = record_seeds_on_manifest(
        manifest, manifest_seed_record(random_seed=19, n_draws=10000, factorization="cholesky")
    )
    assert updated.random_seeds["mc_mass_function"]["random_seed"] == 19
    assert updated.random_seeds["mc_mass_function"]["n_draws"] == 10000


def test_convergence_reuses_mc_noise_threshold() -> None:
    cfg = load_config()
    solution = synthetic_orbital_solution(relative_error=0.02, seed=8)
    draws = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=400, random_seed=cfg.mc_mass_function.random_seed
    )
    diagnostic = run_m2_posterior_convergence(
        [draws],
        m2_threshold_msun=1.4,
        probability_cut=0.95,
        mc_noise_threshold=cfg.physics.mc_noise_threshold,
        boundary_n_sigma=cfg.mc_mass_function.boundary_n_sigma,
    )
    p = draws.probability_m2_above(1.4)
    if p == 0.0:
        assert diagnostic.max_mc_poisson_ratio == pytest.approx(0.0)
    else:
        assert diagnostic.max_mc_poisson_ratio == pytest.approx(1.0 / np.sqrt(400.0))
    assert diagnostic.all_systems_subdominant
    assert diagnostic.n_systems == 1
    report = format_m2_posterior_convergence_report(diagnostic)
    assert "n_within_mc_of_boundary" in report
    assert str(diagnostic.n_within_mc_of_boundary) in report


def test_boundary_count_flags_near_cut() -> None:
    n_draws = 400
    p_hat = 0.95
    n_above = int(round(p_hat * n_draws))
    near = np.concatenate(
        [np.full(n_above, 2.0), np.full(n_draws - n_above, 1.0)]
    )
    far = np.full(n_draws, 3.0)

    def _wrap(m2: np.ndarray, source_id: int) -> MassFunctionDraws:
        return MassFunctionDraws(
            source_id=source_id,
            n_draws=n_draws,
            random_seed=0,
            factorization=CovarianceFactorization.CHOLESKY,
            a0_mas=np.ones(n_draws),
            m_f_msun=np.ones(n_draws),
            m2_msun=m2,
            m1_msun=1.0,
            flux_ratio=0.0,
        )

    diagnostic = run_m2_posterior_convergence(
        [_wrap(near, 1), _wrap(far, 2)],
        m2_threshold_msun=1.4,
        probability_cut=0.95,
        mc_noise_threshold=0.1,
        boundary_n_sigma=1.0,
    )
    assert diagnostic.n_within_mc_of_boundary >= 1
    by_id = {row["source_id"]: row for row in diagnostic.per_system}
    assert by_id[2]["within_mc_of_boundary"] is False


def test_config_fragment_and_checksum() -> None:
    assert "mc_mass_function" in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    assert cfg.mc_mass_function.n_draws == 10000
    assert cfg.mc_mass_function.covariance == "full_12x12"
    assert cfg.diagnostics.hooks.m2_posterior_convergence is True
    base = config_checksum(cfg)
    flipped = cfg.model_copy(deep=True)
    flipped.mc_mass_function.random_seed = cfg.mc_mass_function.random_seed + 1
    assert config_checksum(flipped) != base


def test_missing_nss_names_fail() -> None:
    ps = ParameterSet(
        names=["parallax", "period"],
        values=[1.0, 365.25],
        covariance=[[0.01, 0.0], [0.0, 1.0]],
        provenance="incomplete",
    )
    with pytest.raises(ValueError, match="missing required names"):
        propagate_nss_solution(ps, m1_msun=1.0, n_draws=4, random_seed=0)


@pytest.mark.slow
def test_1e4_draws_mc_noise_subdominant() -> None:
    cfg = load_config()
    n_draws = cfg.mc_mass_function.n_draws
    solution = synthetic_orbital_solution(relative_error=0.03, seed=11)
    deficient = synthetic_orbital_solution(
        relative_error=0.03, rank_deficient=True, seed=12
    )
    ensembles = [
        propagate_nss_solution(
            solution,
            m1_msun=1.0,
            n_draws=n_draws,
            random_seed=cfg.mc_mass_function.random_seed,
            source_id=1,
            eig_rel_floor=cfg.mc_mass_function.eig_rel_floor,
            eig_abs_floor=cfg.mc_mass_function.eig_abs_floor,
        ),
        propagate_nss_solution(
            deficient,
            m1_msun=1.0,
            n_draws=n_draws,
            random_seed=cfg.mc_mass_function.random_seed + 1,
            source_id=2,
            eig_rel_floor=cfg.mc_mass_function.eig_rel_floor,
            eig_abs_floor=cfg.mc_mass_function.eig_abs_floor,
        ),
    ]
    diagnostic = run_m2_posterior_convergence(
        ensembles,
        m2_threshold_msun=1.4,
        probability_cut=0.95,
        mc_noise_threshold=cfg.physics.mc_noise_threshold,
        boundary_n_sigma=cfg.mc_mass_function.boundary_n_sigma,
    )
    assert diagnostic.n_draws == 10000
    assert diagnostic.all_systems_subdominant
    assert diagnostic.max_mc_poisson_ratio < cfg.physics.mc_noise_threshold
    assert diagnostic.mc_noise.all_bins_passed
    # Binomial SE at p=0.95, N=1e4 is ~0.0022, well below the 0.05 remaining mass.
    assert diagnostic.max_probability_sigma < 0.01
    assert diagnostic.n_within_mc_of_boundary >= 0


# --- Issue #257: Andrews et al. (2022) per-draw M1 (Gaia Apsis FLAME mass /
# uniform fallback) --------------------------------------------------------


def test_sample_primary_mass_draws_gaussian_around_flame_value() -> None:
    rng = np.random.default_rng(42)
    draws = sample_primary_mass_draws(
        flame_value_msun=0.85,
        n_draws=20000,
        rng=rng,
        flame_fixed_error_msun=0.1,
        uniform_low_msun=0.63,
        uniform_high_msun=1.0,
    )
    assert draws.shape == (20000,)
    assert np.all(draws > 0.0)
    assert float(np.mean(draws)) == pytest.approx(0.85, abs=0.01)
    assert float(np.std(draws)) == pytest.approx(0.1, rel=0.05)


def test_sample_primary_mass_draws_uniform_fallback_when_flame_missing() -> None:
    rng = np.random.default_rng(7)
    draws = sample_primary_mass_draws(
        flame_value_msun=None,
        n_draws=20000,
        rng=rng,
        flame_fixed_error_msun=0.1,
        uniform_low_msun=0.63,
        uniform_high_msun=1.0,
    )
    assert draws.shape == (20000,)
    assert float(np.min(draws)) >= 0.63
    assert float(np.max(draws)) <= 1.0
    assert float(np.mean(draws)) == pytest.approx(0.815, abs=0.02)


def test_sample_primary_mass_draws_uniform_fallback_on_nonfinite_or_nonpositive() -> None:
    rng = np.random.default_rng(9)
    for bad_value in (float("nan"), -1.0, 0.0):
        draws = sample_primary_mass_draws(
            flame_value_msun=bad_value,
            n_draws=64,
            rng=rng,
            flame_fixed_error_msun=0.1,
            uniform_low_msun=0.63,
            uniform_high_msun=1.0,
        )
        assert np.all(draws >= 0.63) and np.all(draws <= 1.0)


def test_sample_primary_mass_draws_rejects_bad_config() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="flame_fixed_error_msun"):
        sample_primary_mass_draws(
            flame_value_msun=1.0,
            n_draws=10,
            rng=rng,
            flame_fixed_error_msun=0.0,
            uniform_low_msun=0.63,
            uniform_high_msun=1.0,
        )
    with pytest.raises(ValueError, match="uniform_high_msun"):
        sample_primary_mass_draws(
            flame_value_msun=None,
            n_draws=10,
            rng=rng,
            flame_fixed_error_msun=0.1,
            uniform_low_msun=1.0,
            uniform_high_msun=0.63,
        )


def test_mass_function_draws_broadcasts_scalar_m1_to_per_draw_array() -> None:
    solution = synthetic_orbital_solution(relative_error=0.03, seed=4)
    draws = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=50, random_seed=11, source_id=5
    )
    assert isinstance(draws.m1_msun, np.ndarray)
    assert draws.m1_msun.shape == (50,)
    assert np.all(draws.m1_msun == 1.0)
    assert draws.m1_mean() == pytest.approx(1.0)
    assert draws.m1_std() == pytest.approx(0.0)


def test_mass_function_draws_propagates_per_draw_m1_array() -> None:
    """A genuinely per-draw M1 ensemble (#257) must flow through to M2, not
    collapse to a single shared value — the core interface change this issue
    requires.
    """
    solution = synthetic_orbital_solution(relative_error=1e-8, seed=6)
    n_draws = 5000
    rng = np.random.default_rng(123)
    m1_draws = sample_primary_mass_draws(
        flame_value_msun=0.9,
        n_draws=n_draws,
        rng=rng,
        flame_fixed_error_msun=0.1,
        uniform_low_msun=0.63,
        uniform_high_msun=1.0,
    )
    draws = propagate_nss_solution(
        solution, m1_msun=m1_draws, n_draws=n_draws, random_seed=17, source_id=8
    )
    assert draws.m1_msun.shape == (n_draws,)
    np.testing.assert_allclose(draws.m1_msun, m1_draws)
    assert draws.m1_std() > 0.0
    # M2 = f(m_f, M1, F); a genuinely varying M1 ensemble must show up as
    # nonzero M2 spread beyond what a fixed M1 alone would give.
    fixed_draws = propagate_nss_solution(
        solution, m1_msun=0.9, n_draws=n_draws, random_seed=17, source_id=8
    )
    assert draws.m2_std() > fixed_draws.m2_std()


def test_ensemble_row_quantities_reports_m1_mc_stats() -> None:
    solution = synthetic_orbital_solution(relative_error=0.03, seed=10)
    rng = np.random.default_rng(3)
    m1_draws = sample_primary_mass_draws(
        flame_value_msun=None,
        n_draws=200,
        rng=rng,
        flame_fixed_error_msun=0.1,
        uniform_low_msun=0.63,
        uniform_high_msun=1.0,
    )
    draws = propagate_nss_solution(
        solution, m1_msun=m1_draws, n_draws=200, random_seed=21, source_id=2
    )
    quantities = ensemble_row_quantities(draws, m2_threshold_msun=1.4)
    assert quantities["m1_msun_mc_mean"] == pytest.approx(draws.m1_mean())
    assert quantities["m1_msun_mc_sigma"] == pytest.approx(draws.m1_std())
    assert quantities["m1_msun_mc_sigma"] > 0.0


def test_mass_function_draws_rejects_mismatched_m1_shape() -> None:
    with pytest.raises(ValueError, match="m1_msun"):
        MassFunctionDraws(
            source_id=1,
            n_draws=10,
            random_seed=0,
            factorization=CovarianceFactorization.CHOLESKY,
            a0_mas=np.ones(10),
            m_f_msun=np.ones(10),
            m2_msun=np.ones(10),
            m1_msun=np.ones(5),
            flux_ratio=0.0,
        )
