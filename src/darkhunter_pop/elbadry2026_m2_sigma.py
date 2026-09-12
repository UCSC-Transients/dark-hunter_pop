"""El-Badry 2026 ``σ_M̃2`` from astrometric covariance only (§8.2 / §15 Q12).

Holds Janssens ``M̃1`` fixed (``propagate_fit_uncertainty: false``) and draws the
NSS full covariance through photocenter → mass function → dark-companion ``M̃2``.
This is **not** Andrews ``σ_M2`` at fixed ``M1 = 1.0`` — do not alias the two.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from darkhunter_pop.mc_mass_function import propagate_nss_solution
from darkhunter_pop.schemas import ParameterSet


def sigma_m2_tilde_astrometric_msun(
    nss_solution: ParameterSet,
    *,
    m1_tilde_msun: float,
    n_draws: int,
    random_seed: int,
    eig_rel_floor: float,
    eig_abs_floor: float,
    source_id: int | None = None,
) -> float | None:
    """Return ensemble ``std(M̃2)`` with ``M̃1`` held fixed, or ``None`` on failure."""
    if not np.isfinite(m1_tilde_msun) or float(m1_tilde_msun) <= 0.0:
        return None
    try:
        draws = propagate_nss_solution(
            nss_solution,
            m1_msun=float(m1_tilde_msun),
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

    - rows tagged ``_sigma_m2_astrometric_provenance ==
      "elbadry2026_m1_tilde_fixed"`` (produced by the MC path here);
    - rows that supply ``sigma_m2_astrometric_msun`` in their own right — a
      published catalog column or a test fixture — recognized by the absence of
      an equal ``sigma_m2_msun`` beside it.

    Limitation: the alias test is numeric equality, so a genuine σ_M̃2 that
    happens to equal Andrews' σ to the bit is dropped and then recomputed by
    the MC path. That is conservative, not lossy.
    """
    out = dict(row)
    if out.get("_sigma_m2_astrometric_provenance") == "elbadry2026_m1_tilde_fixed":
        return out
    if _looks_like_andrews_alias(out):
        out.pop("sigma_m2_astrometric_msun", None)
        out.pop("_sigma_m2_astrometric_provenance", None)
    return out
