"""El-Badry 2026 ``σ_M̃2`` (fixed M̃1, astrometric cov only)."""

from __future__ import annotations

import pytest

from darkhunter_pop.elbadry2026_m2_sigma import (
    clear_non_elbadry_m2_astrometric_sigma,
    sigma_m2_tilde_astrometric_msun,
)
from darkhunter_pop.mc_mass_function import propagate_nss_solution, synthetic_orbital_solution

pytestmark = pytest.mark.physics


def test_sigma_m2_tilde_differs_from_andrews_m1() -> None:
    """σ_M̃2 at Janssens M̃1 must not equal Andrews σ at M1=1 by construction."""
    solution = synthetic_orbital_solution(relative_error=0.03, seed=11)
    andrews = propagate_nss_solution(
        solution, m1_msun=1.0, n_draws=800, random_seed=5, source_id=1
    )
    sigma_tilde = sigma_m2_tilde_astrometric_msun(
        solution,
        m1_tilde_msun=1.35,
        n_draws=800,
        random_seed=5,
        eig_rel_floor=1e-12,
        eig_abs_floor=1e-18,
        source_id=1,
    )
    assert sigma_tilde is not None
    # Same seed + different M1 → different companion-mass scatter scale.
    assert sigma_tilde != pytest.approx(andrews.m2_std(), rel=1e-6)


def test_clear_strips_unprovenance_sigma() -> None:
    cleared = clear_non_elbadry_m2_astrometric_sigma(
        {"sigma_m2_astrometric_msun": 0.01, "sigma_m2_msun": 0.01}
    )
    assert "sigma_m2_astrometric_msun" not in cleared
    kept = clear_non_elbadry_m2_astrometric_sigma(
        {
            "sigma_m2_astrometric_msun": 0.05,
            "_sigma_m2_astrometric_provenance": "elbadry2026_m1_tilde_fixed",
        }
    )
    assert kept["sigma_m2_astrometric_msun"] == pytest.approx(0.05)
