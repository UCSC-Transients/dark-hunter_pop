"""Unit conversions and inhomogeneous Poisson point-process primitives.

ARCHITECTURE.md §2; residual scope from ``docs/GAIAMOCK_API.md``. Deliberately **not** a
Kepler solver, RUWE predictor, or astrometric cascade — those come from ``gaiamock_mod`` via
``darkhunter_pop.gaiamock_vendor.import_gaiamock_mod``.
"""

from __future__ import annotations

import math

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop import constants

# ---------------------------------------------------------------------------
# Unit conversions (astropy-backed)
# ---------------------------------------------------------------------------


def mas_to_rad(mas: ArrayLike) -> NDArray[np.floating]:
    """Convert milliarcseconds to radians."""
    return (np.asarray(mas, dtype=np.float64) * u.mas).to_value(u.rad)


def rad_to_mas(rad: ArrayLike) -> NDArray[np.floating]:
    """Convert radians to milliarcseconds."""
    return (np.asarray(rad, dtype=np.float64) * u.rad).to_value(u.mas)


def deg_to_rad(deg: ArrayLike) -> NDArray[np.floating]:
    return np.deg2rad(np.asarray(deg, dtype=np.float64))


def rad_to_deg(rad: ArrayLike) -> NDArray[np.floating]:
    return np.rad2deg(np.asarray(rad, dtype=np.float64))


def parallax_mas_to_distance_pc(parallax_mas: ArrayLike) -> NDArray[np.floating]:
    """Distance in parsecs from parallax in mas (``d = 1000 / ϖ``)."""
    plx = np.asarray(parallax_mas, dtype=np.float64)
    if np.any(plx <= 0):
        raise ValueError("parallax_mas must be positive")
    return 1000.0 / plx


def au_to_mas(separation_au: ArrayLike, distance_pc: ArrayLike) -> NDArray[np.floating]:
    """Angular size in mas for a physical separation in AU at ``distance_pc``."""
    sep = np.asarray(separation_au, dtype=np.float64)
    dist = np.asarray(distance_pc, dtype=np.float64)
    if np.any(dist <= 0):
        raise ValueError("distance_pc must be positive")
    # θ[mas] = a[AU] / d[pc] * 1000? No: 1 AU at 1 pc subtends 1 arcsec = 1000 mas.
    # θ[arcsec] = a[AU] / d[pc] → θ[mas] = 1000 * a / d.
    return 1000.0 * sep / dist


def mas_to_au(angle_mas: ArrayLike, distance_pc: ArrayLike) -> NDArray[np.floating]:
    """Physical size in AU for an angular size in mas at ``distance_pc``."""
    ang = np.asarray(angle_mas, dtype=np.float64)
    dist = np.asarray(distance_pc, dtype=np.float64)
    if np.any(dist <= 0):
        raise ValueError("distance_pc must be positive")
    return ang * dist / 1000.0


def solar_masses_to_kg(mass_msun: ArrayLike) -> NDArray[np.floating]:
    return np.asarray(mass_msun, dtype=np.float64) * constants.M_sun.to_value(u.kg)


# ---------------------------------------------------------------------------
# Thiele–Innes geometry (no gaiamock dependency)
# ---------------------------------------------------------------------------


def photocenter_a0_from_thiele_innes(
    A: ArrayLike,
    B: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
) -> NDArray[np.floating]:
    """Photocenter angular semi-major axis ``a0`` (mas) from Thiele–Innes (mas).

    Halbwachs et al. (2022); CONTINUATION_PLAN §5.1:

    ``u = (A² + B² + F² + G²) / 2``, ``v = A G − B F``,
    ``a0 = √(u + √(u² − v²))``.
    """
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    f = np.asarray(F, dtype=np.float64)
    g = np.asarray(G, dtype=np.float64)
    u = (a**2 + b**2 + f**2 + g**2) / 2.0
    v = a * g - b * f
    radicand = np.maximum(u * u - v * v, 0.0)
    return np.sqrt(np.maximum(u + np.sqrt(radicand), 0.0))


def thiele_innes_to_campbell(
    A: ArrayLike,
    B: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert Thiele–Innes elements (mas) to Campbell ``(a0, omega, inc)`` in radians.

    Follows Halbwachs et al. (2023) / NSSTools, matching ``gaiamock_mod.get_Campbell_elements``.
    ``a0`` is the §5.1 photocenter semi-major axis (mas).
    """
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    f = np.asarray(F, dtype=np.float64)
    g = np.asarray(G, dtype=np.float64)

    wp_minus_omega = np.arctan2(b - f, a + g)
    wm_minus_omega = np.arctan2(-b - f, a - g)
    omega = (wp_minus_omega + wm_minus_omega) / 2.0
    omega_node = (wp_minus_omega - wm_minus_omega) / 2.0

    adjust = omega_node < 0
    omega = np.where(adjust, omega + np.pi, omega)
    omega_node = np.where(adjust, omega_node + np.pi, omega_node)

    tan2_i_ag = np.abs((a + g) * np.cos(wm_minus_omega))
    tan2_i_bf = np.abs((f - b) * np.sin(wm_minus_omega))
    use_ag = tan2_i_ag > tan2_i_bf
    inc = np.where(
        use_ag,
        2.0
        * np.arctan2(
            np.sqrt(np.abs((a - g) * np.cos(wp_minus_omega))),
            np.sqrt(np.maximum(tan2_i_ag, 0.0)),
        ),
        2.0
        * np.arctan2(
            np.sqrt(np.abs((b + f) * np.sin(wp_minus_omega))),
            np.sqrt(np.maximum(tan2_i_bf, 0.0)),
        ),
    )

    a0 = photocenter_a0_from_thiele_innes(a, b, f, g)
    omega = np.mod(omega, 2.0 * np.pi)
    return a0, omega, inc


def astrometric_mass_function(
    a0_mas: ArrayLike,
    parallax_mas: ArrayLike,
    period_day: ArrayLike,
) -> NDArray[np.floating]:
    """Observational mass function ``f(m) = (a0/ϖ)³ / P_yr²`` (no M1 assumption)."""
    a0 = np.asarray(a0_mas, dtype=np.float64)
    plx = np.asarray(parallax_mas, dtype=np.float64)
    period = np.asarray(period_day, dtype=np.float64)
    out = np.full(a0.shape, np.nan, dtype=np.float64)
    valid = (plx > 0.0) & (period > 0.0) & np.isfinite(a0) & (a0 > 0.0)
    if np.any(valid):
        p_yr = period[valid] / 365.25
        out[valid] = (a0[valid] / plx[valid]) ** 3 / p_yr**2
    return out


def luminous_companion_mass_function_over_m1(
    q: ArrayLike,
    flux_ratio: ArrayLike,
) -> NDArray[np.floating]:
    """``m_f / M1`` for a luminous companion (CONTINUATION_PLAN §5.4).

    With ``q = M2/M1`` and ``F = F2/F1``:

    ``m_f / M1 = (q − F)³ / [ (1 + F)³ (1 + q)² ]``

    The dark-companion limit (§5.3) is ``F = 0``. The denominator uses
    ``(1 + F)³``, not ``(1 + F1)³`` (transcription correction in §5.4).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    flux = np.asarray(flux_ratio, dtype=np.float64)
    out = np.full(np.broadcast(q_arr, flux).shape, np.nan, dtype=np.float64)
    q_b, f_b = np.broadcast_arrays(q_arr, flux)
    valid = np.isfinite(q_b) & np.isfinite(f_b) & (f_b >= 0.0) & (q_b > f_b)
    if np.any(valid):
        qv = q_b[valid]
        fv = f_b[valid]
        out[valid] = (qv - fv) ** 3 / ((1.0 + fv) ** 3 * (1.0 + qv) ** 2)
    return out


def invert_astrometric_companion_mass(
    m1_msun: ArrayLike,
    m_f_msun: ArrayLike,
    flux_ratio: ArrayLike = 0.0,
    *,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> NDArray[np.floating]:
    """Solve §5.3/§5.4 for ``M2`` given ``M1``, ``m_f``, and ``F = F2/F1``.

    Newton's method on ``f(q) = (q − F)³ − (m_f/M1) (1+F)³ (1+q)²`` with
    ``q = M2/M1``. ``F = 0`` is the dark-companion (minimum-M2) limit.
    """
    m1 = np.asarray(m1_msun, dtype=np.float64)
    mf = np.asarray(m_f_msun, dtype=np.float64)
    flux = np.asarray(flux_ratio, dtype=np.float64)
    m1_b, mf_b, f_b = np.broadcast_arrays(m1, mf, flux)
    out_shape = m1_b.shape
    m1_r = np.ravel(m1_b)
    mf_r = np.ravel(mf_b)
    f_r = np.ravel(f_b)
    q = np.full(m1_r.shape, np.nan, dtype=np.float64)
    y = np.full(m1_r.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(m1_r)
        & np.isfinite(mf_r)
        & np.isfinite(f_r)
        & (m1_r > 0.0)
        & (mf_r > 0.0)
        & (f_r >= 0.0)
    )
    if not np.any(valid):
        return q.reshape(out_shape)
    y[valid] = mf_r[valid] / m1_r[valid]
    # Cubic (q-F)^3 - y (1+F)^3 (1+q)^2 = 0, expanded in q.
    # q^3 + (-3F - a) q^2 + (3 F^2 - 2a) q + (-F^3 - a) = 0 with a = y (1+F)^3.
    for idx in np.flatnonzero(valid):
        f_i = float(f_r[idx])
        a_i = float(y[idx]) * (1.0 + f_i) ** 3
        coeffs = (
            1.0,
            -3.0 * f_i - a_i,
            3.0 * f_i * f_i - 2.0 * a_i,
            -(f_i**3) - a_i,
        )
        roots = np.roots(coeffs)
        real = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)])
        real = real[real > f_i]
        if real.size == 0:
            continue
        q[idx] = float(np.min(real))
    one_f3 = (1.0 + f_r) ** 3
    for _ in range(max_iter):
        dq = q - f_r
        one_q = 1.0 + q
        resid = dq**3 - y * one_f3 * one_q**2
        deriv = 3.0 * dq**2 - y * one_f3 * 2.0 * one_q
        movable = (
            valid
            & np.isfinite(q)
            & np.isfinite(resid)
            & np.isfinite(deriv)
            & (np.abs(deriv) > 0.0)
        )
        step = np.zeros_like(q)
        step[movable] = resid[movable] / deriv[movable]
        trial = q - step
        trial = np.where(trial <= f_r, 0.5 * (q + f_r) + 0.5 * np.maximum(q - f_r, 0.1), trial)
        q = np.where(movable, trial, q)
        if np.all(~movable | (np.abs(resid) < tol)):
            break
    q = np.where(valid & np.isfinite(q) & (q > f_r), q, np.nan)
    return (q * m1_r).reshape(out_shape)


# ---------------------------------------------------------------------------
# Inhomogeneous Poisson point-process primitives
# ---------------------------------------------------------------------------


def poisson_log_likelihood_inhomogeneous(
    log_intensity_at_events: ArrayLike,
    integrated_intensity: float,
) -> float:
    """Log-likelihood for an inhomogeneous Poisson point process.

    For observed points ``x_i`` with intensity ``λ(x)``:

    ``log L = Σ_i log λ(x_i) − ∫ λ(x) dx``

    Parameters
    ----------
    log_intensity_at_events:
        ``log λ(x_i)`` at each observed event (natural log).
    integrated_intensity:
        ``Λ = ∫ λ(x) dx`` over the observation domain (must be ≥ 0).
    """
    logs = np.asarray(log_intensity_at_events, dtype=np.float64)
    if integrated_intensity < 0:
        raise ValueError("integrated_intensity must be >= 0")
    if logs.size == 0:
        return float(-integrated_intensity)
    if not np.all(np.isfinite(logs)):
        raise ValueError("log_intensity_at_events must be finite")
    return float(np.sum(logs) - integrated_intensity)


def integrate_intensity_trapezoid(
    x: ArrayLike,
    intensity: ArrayLike,
) -> float:
    """Trapezoidal integral of ``λ(x)`` on a 1-D grid (analytic tests use exact cases)."""
    x_arr = np.asarray(x, dtype=np.float64)
    lam = np.asarray(intensity, dtype=np.float64)
    if x_arr.ndim != 1 or lam.shape != x_arr.shape:
        raise ValueError("x and intensity must be 1-D and the same shape")
    if x_arr.size < 2:
        raise ValueError("need at least two samples to integrate")
    if np.any(np.diff(x_arr) <= 0):
        raise ValueError("x must be strictly increasing")
    return float(np.trapz(lam, x_arr))


def poisson_mean_count(integrated_intensity: float) -> float:
    """Expected event count ``⟨N⟩ = Λ`` for a Poisson process."""
    if integrated_intensity < 0:
        raise ValueError("integrated_intensity must be >= 0")
    return float(integrated_intensity)


def poisson_upper_limit_zero_events(
    confidence: float = 0.95,
) -> float:
    """Upper limit on mean count ``μ`` given zero observed events.

    Solves ``P(N=0 | μ) = e^{-μ} = 1 − confidence``, i.e.
    ``μ_ul = −ln(1 − confidence)``. For 95%, ``μ_ul ≈ 2.9957``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return float(-math.log(1.0 - confidence))


def binned_poisson_log_likelihood(
    counts: ArrayLike,
    expected: ArrayLike,
) -> float:
    """Sum of independent Poisson log-likelihoods over bins.

    ``log L = Σ_k [ n_k log μ_k − μ_k − log(n_k!) ]`` with ``0 log 0`` treated as 0 when
    ``n_k = 0``. Empty bins (``n_k = 0``) contribute ``−μ_k``.
    """
    n = np.asarray(counts, dtype=np.float64)
    mu = np.asarray(expected, dtype=np.float64)
    if n.shape != mu.shape:
        raise ValueError("counts and expected must share shape")
    if np.any(n < 0) or np.any(mu < 0):
        raise ValueError("counts and expected must be non-negative")
    total = 0.0
    for n_k, mu_k in zip(n.ravel(), mu.ravel(), strict=True):
        if mu_k == 0.0:
            if n_k > 0:
                return float("-inf")
            continue
        total += n_k * math.log(mu_k) - mu_k - math.lgamma(n_k + 1.0)
    return float(total)
