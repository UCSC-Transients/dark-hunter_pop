"""El-Badry 2026 ``σ_M̃2`` (§8.2 / §15 Q12 / #284).

Two config-switchable propagation methods (``sigma_m2_tilde.method`` in
``config/selections/elbadry2026.yaml``):

- ``analytic`` — first-order Jacobian of ``M̃2(A, B, F, G, ϖ, P; M̃1)``, the
  nsstools-style estimate (#199: nsstools' ``σ_a0`` is the analytic Jacobian of
  ``a0(A, B, F, G)``). Adopted for the reproduction path by the PI decision on
  #284;
- ``monte_carlo`` — the full-12×12 NSS draw ensemble (pre-#284 default).

By default Janssens ``M̃1`` is held fixed (``propagate_fit_uncertainty: false``,
Q12). When the switch is on, the Janssens Table 1 fit uncertainty on
``log10 M̃1`` enters too (analytically, or as per-draw ``M̃1`` in the MC).

The provenance tag on ``_sigma_m2_astrometric_provenance`` names the method and
the ``M̃1`` treatment (:func:`sigma_provenance_tag`). This is **not** Andrews
``σ_M2`` at fixed ``M1 = 1.0`` — do not alias the two.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from darkhunter_pop.mc_mass_function import (
    M1_DRAW_SEED_SALT,
    canonical_nss_name,
    propagate_nss_solution,
)
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    invert_astrometric_companion_mass,
    photocenter_a0_from_thiele_innes,
)
from darkhunter_pop.schemas import ParameterSet

#: Legacy (pre-#284) tag: MC, fixed Janssens M̃1. Still recognized on read.
LEGACY_MC_FIXED_PROVENANCE = "elbadry2026_m1_tilde_fixed"
_PROVENANCE_PREFIX = "elbadry2026_"

_ANALYTIC_NAMES: tuple[str, ...] = (
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "parallax",
    "period",
)


def sigma_provenance_tag(
    method: str, *, m1_uncertainty: bool, analytic_covariance: str | None = None
) -> str:
    """Provenance naming how ``sigma_m2_astrometric_msun`` was made (#284).

    ``elbadry2026_{analytic_full|analytic_nsstools_blocks|mc}_m1_tilde_{fixed|janssens}``.
    """
    if method == "analytic":
        label = f"analytic_{analytic_covariance or 'full'}"
    elif method == "monte_carlo":
        label = "mc"
    else:
        raise ValueError(f"unhandled sigma_m2_tilde method {method!r}")
    m1 = "janssens" if m1_uncertainty else "fixed"
    return f"{_PROVENANCE_PREFIX}{label}_m1_tilde_{m1}"


def is_elbadry_sigma_provenance(tag: Any) -> bool:
    """True for any tag written by this module (current or legacy)."""
    return isinstance(tag, str) and tag.startswith(_PROVENANCE_PREFIX)


def _photocenter_a0_gradient(
    a: float, b: float, f: float, g: float
) -> tuple[float, np.ndarray] | None:
    """``a0`` and ``∂a0/∂(A, B, F, G)``; ``None`` at the ``u² = v²`` singularity."""
    u = 0.5 * (a * a + b * b + f * f + g * g)
    v = a * g - b * f
    w2 = u * u - v * v
    if not (np.isfinite(w2) and w2 > 0.0):
        return None
    w = float(np.sqrt(w2))
    a0 = float(photocenter_a0_from_thiele_innes(a, b, f, g))
    if not (np.isfinite(a0) and a0 > 0.0):
        return None
    du = np.array([a, b, f, g], dtype=np.float64)
    dv = np.array([g, -f, -b, a], dtype=np.float64)
    d_a0_sq = du + (u * du - v * dv) / w
    return a0, d_a0_sq / (2.0 * a0)


def dark_companion_m2_derivatives(
    m1_msun: float, m2_msun: float
) -> tuple[float, float]:
    """``(∂M2/∂ln m_f, ∂M2/∂M1)`` for ``m_f = M2³/(M1+M2)²`` (dark companion)."""
    total = m1_msun + m2_msun
    denom = 3.0 / m2_msun - 2.0 / total
    return 1.0 / denom, (2.0 / total) / denom


def sigma_m2_tilde_analytic_msun(
    nss_solution: ParameterSet,
    *,
    m1_tilde_msun: float,
    analytic_covariance: str = "full",
    sigma_log10_m1: float = 0.0,
) -> float | None:
    """First-order ``σ_M̃2`` (dark companion) from the NSS covariance (#284).

    ``ln m_f = 3 ln a0 − 3 ln ϖ − 2 ln P`` and ``M̃2`` solves
    ``m_f = M̃2³/(M̃1+M̃2)²``. ``analytic_covariance="full"`` propagates the
    gradient through the ``(A, B, F, G, ϖ, P)`` block of the full covariance
    (cross terms kept); ``"nsstools_blocks"`` takes nsstools' ``σ_a0`` from the
    ``(A, B, F, G)`` block and adds ``ϖ`` and ``P`` in quadrature.
    ``sigma_log10_m1 > 0`` adds the Janssens ``M̃1`` term, independent of the
    astrometry. ``None`` when undefined (singular ``a0``, non-positive ϖ/P, …).
    """
    m1 = float(m1_tilde_msun)
    if not (np.isfinite(m1) and m1 > 0.0):
        return None
    names = [canonical_nss_name(n) for n in nss_solution.names]
    try:
        idx = [names.index(n) for n in _ANALYTIC_NAMES]
    except ValueError:
        return None
    values = nss_solution.values_array()[idx]
    cov = nss_solution.covariance_array()[np.ix_(idx, idx)]
    if not np.all(np.isfinite(cov)):
        return None
    a, b, f, g, plx, period = (float(x) for x in values)
    if not (plx > 0.0 and period > 0.0):
        return None
    grad = _photocenter_a0_gradient(a, b, f, g)
    if grad is None:
        return None
    a0, d_a0 = grad
    if analytic_covariance == "full":
        g_lnmf = np.concatenate([3.0 * d_a0 / a0, [-3.0 / plx, -2.0 / period]])
        var_lnmf = float(g_lnmf @ cov @ g_lnmf)
    elif analytic_covariance == "nsstools_blocks":
        var_a0 = float(d_a0 @ cov[:4, :4] @ d_a0)
        var_lnmf = (
            9.0 * var_a0 / a0**2
            + 9.0 * float(cov[4, 4]) / plx**2
            + 4.0 * float(cov[5, 5]) / period**2
        )
    else:
        raise ValueError(f"unhandled analytic_covariance {analytic_covariance!r}")
    if not (np.isfinite(var_lnmf) and var_lnmf >= 0.0):
        return None
    m_f = float(astrometric_mass_function(a0, plx, period))
    m2 = float(invert_astrometric_companion_mass(m1, m_f, 0.0))
    if not (np.isfinite(m2) and m2 > 0.0):
        return None
    d_lnmf, d_m1 = dark_companion_m2_derivatives(m1, m2)
    sigma_m1 = m1 * np.log(10.0) * float(sigma_log10_m1)
    var = d_lnmf**2 * var_lnmf + d_m1**2 * sigma_m1**2
    sigma = float(np.sqrt(var))
    return sigma if np.isfinite(sigma) else None


def sigma_m2_tilde_astrometric_msun(
    nss_solution: ParameterSet,
    *,
    m1_tilde_msun: float,
    n_draws: int,
    random_seed: int,
    eig_rel_floor: float,
    eig_abs_floor: float,
    source_id: int | None = None,
    sigma_log10_m1: float = 0.0,
) -> float | None:
    """Return ensemble ``std(M̃2)``, or ``None`` on failure.

    ``M̃1`` is held fixed unless ``sigma_log10_m1 > 0``, in which case each draw
    gets ``M̃1 · 10^N(0, σ_log10M1)`` from an RNG stream salted off the
    covariance stream (Janssens fit uncertainty, ``propagate_fit_uncertainty``).
    """
    if not np.isfinite(m1_tilde_msun) or float(m1_tilde_msun) <= 0.0:
        return None
    m1_draws: Any = float(m1_tilde_msun)
    if sigma_log10_m1 > 0.0:
        rng_m1 = np.random.default_rng(int(random_seed) ^ M1_DRAW_SEED_SALT)
        m1_draws = float(m1_tilde_msun) * 10.0 ** rng_m1.normal(
            0.0, float(sigma_log10_m1), size=int(n_draws)
        )
    try:
        draws = propagate_nss_solution(
            nss_solution,
            m1_msun=m1_draws,
            n_draws=int(n_draws),
            random_seed=int(random_seed),
            eig_rel_floor=float(eig_rel_floor),
            eig_abs_floor=float(eig_abs_floor),
            source_id=source_id,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    sigma = float(draws.m2_std())
    if not np.isfinite(sigma):
        return None
    return sigma


def _looks_like_andrews_alias(row: Mapping[str, Any]) -> bool:
    """True when ``sigma_m2_astrometric_msun`` is a copy of Andrews ``sigma_m2_msun``.

    The pre-reconciliation alias wrote the *same object* into both keys, so
    numeric equality of the two columns is the alias fingerprint. A row that
    carries only one of the two, or two genuinely different values, is left
    alone.
    """
    astrometric = row.get("sigma_m2_astrometric_msun")
    andrews = row.get("sigma_m2_msun")
    if astrometric is None or andrews is None:
        return False
    try:
        return bool(np.isclose(float(astrometric), float(andrews), rtol=0.0, atol=0.0))
    except (TypeError, ValueError):
        return False


def clear_non_elbadry_m2_astrometric_sigma(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``sigma_m2_astrometric_msun`` when it is an Andrews-σ alias.

    Parent caches built before the alias was removed carry Andrews
    ``sigma_m2_msun`` (fixed ``M1 = 1.0``) copied into the astrometric column;
    that value must never feed ``sub_chandrasekhar``, which needs σ_M̃2 at the
    fixed Janssens ``M̃1``. Column ownership is strict (CLAUDE.md Gotchas).

    Kept untouched:

    - rows whose ``_sigma_m2_astrometric_provenance`` is an El-Badry tag
      (:func:`is_elbadry_sigma_provenance`; produced by this module);
    - rows that supply ``sigma_m2_astrometric_msun`` in their own right — a
      published catalog column or a test fixture — recognized by the absence of
      an equal ``sigma_m2_msun`` beside it.

    Limitation: the alias test is numeric equality, so a genuine σ_M̃2 that
    happens to equal Andrews' σ to the bit is dropped and then recomputed by
    the configured σ path. That is conservative, not lossy.
    """
    out = dict(row)
    if is_elbadry_sigma_provenance(out.get("_sigma_m2_astrometric_provenance")):
        return out
    if _looks_like_andrews_alias(out):
        out.pop("sigma_m2_astrometric_msun", None)
        out.pop("_sigma_m2_astrometric_provenance", None)
    return out
