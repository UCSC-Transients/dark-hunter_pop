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


def test_photocenter_a0_halbwachs_formula() -> None:
    a0 = 17.57
    omega = np.deg2rad(231.65)
    inc = np.deg2rad(79.205)
    node = np.deg2rad(204.85)
    A = a0 * (np.cos(omega) * np.cos(node) - np.sin(omega) * np.sin(node) * np.cos(inc))
    B = a0 * (np.cos(omega) * np.sin(node) + np.sin(omega) * np.cos(node) * np.cos(inc))
    F = a0 * (-np.sin(omega) * np.cos(node) - np.cos(omega) * np.sin(node) * np.cos(inc))
    G = a0 * (-np.sin(omega) * np.sin(node) + np.cos(omega) * np.cos(node) * np.cos(inc))
    u = (A**2 + B**2 + F**2 + G**2) / 2.0
    v = A * G - B * F
    expected = np.sqrt(u + np.sqrt(u**2 - v**2))
    assert P.photocenter_a0_from_thiele_innes(A, B, F, G) == pytest.approx(expected)
    a0_c, _, _ = P.thiele_innes_to_campbell(A, B, F, G)
    assert a0_c == pytest.approx(expected)


def test_astrometric_mass_function_basic() -> None:
    fm = P.astrometric_mass_function(1.0, 2.0, 365.25)
    assert fm == pytest.approx(0.125)
    assert np.isnan(P.astrometric_mass_function(1.0, 0.0, 365.25))


def test_dark_companion_inversion_round_trip() -> None:
    m1 = 1.0
    m2 = 1.6
    q = m2 / m1
    mf = m1 * (q**3 / (1.0 + q) ** 2)
    recovered = P.invert_astrometric_companion_mass(m1, mf, 0.0)
    assert recovered == pytest.approx(m2, rel=1e-8)


def test_luminous_companion_uses_one_plus_f_not_f1() -> None:
    q = 1.2
    flux = 0.25
    y = P.luminous_companion_mass_function_over_m1(q, flux)
    expected = (q - flux) ** 3 / ((1.0 + flux) ** 3 * (1.0 + q) ** 2)
    wrong_f1 = (q - flux) ** 3 / ((1.0 + 1.0) ** 3 * (1.0 + q) ** 2)
    assert y == pytest.approx(expected)
    assert y != pytest.approx(wrong_f1, rel=1e-3)
    m1 = 1.1
    mf = y * m1
    m2 = P.invert_astrometric_companion_mass(m1, mf, flux)
    assert m2 == pytest.approx(q * m1, rel=1e-6)


def test_spectroscopic_mass_function_and_independent_inversion() -> None:
    p, k, e, m1 = 10.0, 50.0, 0.1, 1.2
    fm = float(P.spectroscopic_mass_function(p, k, e))
    assert fm > 0.0
    m2 = float(P.invert_spectroscopic_minimum_companion_mass(fm, m1))
    recovered = m2**3 / (m1 + m2) ** 2
    assert recovered == pytest.approx(fm, rel=1e-6)
    assert "invert_astrometric_companion_mass" not in (
        P.invert_spectroscopic_minimum_companion_mass.__code__.co_names
    )


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
