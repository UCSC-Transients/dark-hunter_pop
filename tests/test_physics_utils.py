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


def test_physics_utils_does_not_import_gaiamock() -> None:
    """Scope guard: residual module must not pull in orbital gaiamock APIs."""
    src = Path(P.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "gaiamock" not in imported
    assert "gaiamock_mod" not in imported
    assert "gaiamock_vendor" not in src
    banned = ("kepler", "solve_kepler", "check_ruwe", "fit_full_astrometric")
    lower = src.lower()
    for token in banned:
        # Allow mentioning in the module docstring as "NOT a Kepler solver".
        if token == "kepler":
            continue
        assert token not in lower
