"""Closed-form tests for the spectroscopic SB1 mass function (not astrometric)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from darkhunter_pop import constants
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SpectroscopicMassFunctionConfig
from darkhunter_pop.data_acquisition import classify_sb1_reproduction_route
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    spectroscopic_mass_function,
    spectroscopic_minimum_companion_mass,
)

pytestmark = pytest.mark.physics


def test_spectroscopic_mass_function_circular_closed_form() -> None:
    period_day = 365.25
    k1 = 10.0
    f_m = float(spectroscopic_mass_function(period_day, k1, 0.0))
    expected = constants.SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS * (k1**3) * period_day
    assert f_m == pytest.approx(expected)


def test_spectroscopic_mass_function_eccentricity_factor() -> None:
    period_day = 80.0
    k1 = 40.0
    ecc = 0.5
    f_circ = float(spectroscopic_mass_function(period_day, k1, 0.0))
    f_ecc = float(spectroscopic_mass_function(period_day, k1, ecc))
    assert f_ecc == pytest.approx(f_circ * (1.0 - ecc**2) ** 1.5)


def test_spectroscopic_mass_function_invalid_is_nan() -> None:
    assert np.isnan(spectroscopic_mass_function(0.0, 10.0, 0.0))
    assert np.isnan(spectroscopic_mass_function(10.0, -1.0, 0.0))
    assert np.isnan(spectroscopic_mass_function(10.0, 10.0, 1.0))
    assert np.isnan(spectroscopic_mass_function(10.0, 10.0, -0.1))


def test_spectroscopic_and_astrometric_mass_functions_are_distinct() -> None:
    # Same period, unrelated inputs: the two formulae must not be interchangeable.
    period = 365.25
    spec = float(spectroscopic_mass_function(period, 20.0, 0.1))
    astro = float(astrometric_mass_function(1.0, 2.0, period))
    assert spec != pytest.approx(astro)
    assert spec > 0.0
    assert astro == pytest.approx(0.125)


def test_spectroscopic_minimum_companion_mass_closed_form() -> None:
    # f_m = M2^3 / (M1+M2)^2 at i=90°. Pick M1=1, M2=2 → f = 8/9.
    m1, m2 = 1.0, 2.0
    f_m = (m2**3) / (m1 + m2) ** 2
    recovered = float(
        spectroscopic_minimum_companion_mass(
            f_m, m1, max_m2_msun=500.0, n_bisection=200
        )
    )
    assert recovered == pytest.approx(m2, rel=1e-8)

    # Equal-mass: M1=M2=1 → f=1/4.
    recovered_eq = float(
        spectroscopic_minimum_companion_mass(
            0.25, 1.0, max_m2_msun=500.0, n_bisection=200
        )
    )
    assert recovered_eq == pytest.approx(1.0, rel=1e-8)


def test_spectroscopic_minimum_companion_mass_vectorized() -> None:
    m1 = np.array([0.8, 1.2, 2.0])
    m2 = np.array([1.5, 1.5, 3.0])
    f_m = m2**3 / (m1 + m2) ** 2
    recovered = spectroscopic_minimum_companion_mass(
        f_m, m1, max_m2_msun=500.0, n_bisection=200
    )
    np.testing.assert_allclose(recovered, m2, rtol=1e-8)


def test_spectroscopic_minimum_companion_mass_does_not_use_sini() -> None:
    """Edge-on inversion is sin i = 1; it is not inclination marginalization."""
    m1, m2 = 1.0, 3.0
    f_edge = m2**3 / (m1 + m2) ** 2
    f_60 = (m2 * np.sin(np.deg2rad(60.0))) ** 3 / (m1 + m2) ** 2
    recovered_edge = float(
        spectroscopic_minimum_companion_mass(
            f_edge, m1, max_m2_msun=500.0, n_bisection=200
        )
    )
    recovered_60 = float(
        spectroscopic_minimum_companion_mass(
            f_60, m1, max_m2_msun=500.0, n_bisection=200
        )
    )
    assert recovered_edge == pytest.approx(m2, rel=1e-8)
    assert recovered_60 < m2 - 0.5


def test_v1_config_refuses_inference_flag() -> None:
    with pytest.raises(ValidationError, match="v1_inference_eligible"):
        SpectroscopicMassFunctionConfig(v1_inference_eligible=True)


def test_published_table8_inversion_and_route_split() -> None:
    cfg = load_config().spectroscopic_mass_function
    path = Path(cfg.published_table_path)
    if not path.is_absolute():
        path = repo_root() / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = raw["data"]
    assert len(rows) == cfg.expected_union_n

    n_ms = 0
    n_fm = 0
    n_both = 0
    n_union = 0
    for row in rows:
        m1 = row.get("m1_msun")
        m2_min = row.get("m2_min_msun")
        f_m = float(row["fm_msun"])
        if m1 is not None and m2_min is not None:
            recovered = float(
                spectroscopic_minimum_companion_mass(
                    f_m,
                    float(m1),
                    max_m2_msun=cfg.inversion_max_m2_msun,
                    n_bisection=cfg.inversion_n_bisection,
                )
            )
            assert recovered == pytest.approx(float(m2_min), rel=2e-3, abs=0.02)
        route = classify_sb1_reproduction_route(
            k1_significance=float(row["significance"]),
            f_m_msun=f_m,
            m1_msun=None if m1 is None else float(m1),
            m2_min_msun=None if m2_min is None else float(m2_min),
            spectroscopic=cfg,
        )
        n_ms += int(route.main_sequence_min_companion_mass)
        n_fm += int(route.high_mass_function)
        n_both += int(
            route.main_sequence_min_companion_mass and route.high_mass_function
        )
        n_union += int(route.in_union)

    assert n_ms == cfg.expected_n_ms_min_companion
    assert n_fm == cfg.expected_n_high_fm
    assert n_both == cfg.expected_n_both_routes
    assert n_union == cfg.expected_union_n
    assert n_ms + n_fm - n_both == n_union


def test_population_model_does_not_consume_sb1_path() -> None:
    from darkhunter_pop import population_model as pop
    from darkhunter_pop import inference as inf

    for module in (pop, inf):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        banned = (
            "darkhunter_pop.physics_utils.spectroscopic_minimum_companion_mass",
            "darkhunter_pop.physics_utils.invert_spectroscopic_minimum_companion_mass",
        )
        for name in imported:
            assert name not in banned
        src = Path(module.__file__).read_text(encoding="utf-8")
        assert "spectroscopic_minimum_companion_mass" not in src
        assert "invert_spectroscopic_minimum_companion_mass" not in src
        assert "spectroscopic_mass_function(" not in src
