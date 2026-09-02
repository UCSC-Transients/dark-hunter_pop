"""Physics tests for ``physics_utils`` (analytic solutions) and scope guards."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import physics_utils as P

pytestmark = pytest.mark.physics


def test_mas_rad_round_trip() -> None:
    mas = np.array([1.0, 1000.0])
    assert P.rad_to_mas(P.mas_to_rad(mas)) == pytest.approx(mas)


def test_parallax_distance_and_au_mas() -> None:
    # 1 mas → 1000 pc; 1 AU at 1000 pc → 1 mas.
    assert P.parallax_mas_to_distance_pc(1.0) == pytest.approx(1000.0)
    assert P.au_to_mas(1.0, 1000.0) == pytest.approx(1.0)
    assert P.mas_to_au(1.0, 1000.0) == pytest.approx(1.0)


def test_homogeneous_poisson_log_likelihood_analytic() -> None:
    # Domain [0, 1], λ = 2 constant → Λ = 2. Two events: log L = 2*log(2) − 2.
    log_l = P.poisson_log_likelihood_inhomogeneous(
        [math.log(2.0), math.log(2.0)], integrated_intensity=2.0
    )
    assert log_l == pytest.approx(2.0 * math.log(2.0) - 2.0)


def test_zero_events_likelihood_is_minus_integral() -> None:
    assert P.poisson_log_likelihood_inhomogeneous([], 3.5) == pytest.approx(-3.5)


def test_integrate_constant_intensity() -> None:
    x = np.linspace(0.0, 2.0, 21)
    lam = np.full_like(x, 3.0)
    assert P.integrate_intensity_trapezoid(x, lam) == pytest.approx(6.0)


def test_poisson_upper_limit_95() -> None:
    ul = P.poisson_upper_limit_zero_events(0.95)
    assert ul == pytest.approx(-math.log(0.05))
    assert ul == pytest.approx(2.99573227355)


def test_binned_poisson_empty_and_occupied() -> None:
    # One empty bin μ=1 → contrib −1; one bin n=2, μ=2 → 2*log2 − 2 − log(2!)
    ll = P.binned_poisson_log_likelihood([0.0, 2.0], [1.0, 2.0])
    expected = -1.0 + (2.0 * math.log(2.0) - 2.0 - math.lgamma(3.0))
    assert ll == pytest.approx(expected)


def test_impossible_counts_when_mu_zero() -> None:
    assert P.binned_poisson_log_likelihood([1.0], [0.0]) == float("-inf")


def test_thiele_innes_to_campbell_round_trip() -> None:
    # Campbell elements → TI (Wikipedia) → recover a0.
    a0 = 17.57  # mas
    omega = np.deg2rad(231.65)
    inc = np.deg2rad(79.205)
    node = np.deg2rad(204.85)
    a = a0
    A = a * (np.cos(omega) * np.cos(node) - np.sin(omega) * np.sin(node) * np.cos(inc))
    B = a * (np.cos(omega) * np.sin(node) + np.sin(omega) * np.cos(node) * np.cos(inc))
    F = a * (-np.sin(omega) * np.cos(node) - np.cos(omega) * np.sin(node) * np.cos(inc))
    G = a * (-np.sin(omega) * np.sin(node) + np.cos(omega) * np.cos(node) * np.cos(inc))
    a0_out, omega_out, inc_out = P.thiele_innes_to_campbell(A, B, F, G)
    assert a0_out == pytest.approx(a0, rel=1e-4)
    assert inc_out == pytest.approx(inc, abs=0.05)
    assert np.isfinite(omega_out)


def test_astrometric_mass_function_basic() -> None:
    fm = P.astrometric_mass_function(1.0, 2.0, 365.25)
    assert fm == pytest.approx(0.125)
    assert np.isnan(P.astrometric_mass_function(1.0, 0.0, 365.25))


def test_physics_utils_does_not_import_gaiamock() -> None:
    """Scope guard: residual module must not import orbital gaiamock APIs."""
    src_path = Path(P.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned_prefixes = ("gaiamock", "gaiamock_mod", "darkhunter_pop.gaiamock_vendor")
    for name in imported:
        assert not any(
            name == b or name.startswith(b + ".") for b in banned_prefixes
        ), f"unexpected import {name}"
