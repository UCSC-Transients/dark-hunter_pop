"""Monte Carlo astrometric mass-function propagation (CONTINUATION_PLAN §11).

Draws the full NSS fitted-parameter covariance (typically 12×12 Orbital),
propagates each draw through §5.1 → §5.2 → §5.3, and returns a per-system
posterior over ``(m_f, M2)``. Probability cuts are ensemble fractions;
``M2 / σ_M2`` uses the ensemble mean and standard deviation.

Near-singular covariances: try Cholesky of the symmetrized matrix, then
Cholesky of a tiny-nugget ridge, then an eigenvalue clip (negative and
relatively tiny eigenvalues zeroed). There is no diagonal-only fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    invert_astrometric_companion_mass,
    photocenter_a0_from_thiele_innes,
)
from darkhunter_pop.schemas import ParameterSet, RunManifest
from darkhunter_pop.sensitivity_analysis import (
    MCNoiseConvergenceDiagnostic,
    evaluate_mc_noise_at_n,
    run_mc_noise_convergence,
)

# Gaia NSS / Thiele–Innes aliases → canonical ParameterSet names (#18).
_PARAM_ALIASES: dict[str, str] = {
    "A": "a_thiele_innes",
    "B": "b_thiele_innes",
    "F": "f_thiele_innes",
    "G": "g_thiele_innes",
    "a_thiele_innes": "a_thiele_innes",
    "b_thiele_innes": "b_thiele_innes",
    "f_thiele_innes": "f_thiele_innes",
    "g_thiele_innes": "g_thiele_innes",
    "parallax": "parallax",
    "period": "period",
    "period_day": "period",
}

_REQUIRED_CANONICAL: tuple[str, ...] = (
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "parallax",
    "period",
)

MANIFEST_SEED_KEY: str = "mc_mass_function"
FULL_COVARIANCE_MODE: str = "full_12x12"


class CovarianceFactorization(str, Enum):
    """How a draw factor ``L`` with ``x = μ + L z`` was obtained."""

    CHOLESKY = "cholesky"
    CHOLESKY_NUGGET = "cholesky_nugget"
    EIGEN_CLIP = "eigen_clip"


class UnhandledCovarianceModeError(ValueError):
    """Raised when a Monte Carlo covariance mode is not dispatched."""


@dataclass(frozen=True)
class CovarianceFactor:
    """Square factor for ``mean + factor @ z`` sampling."""

    factor: NDArray[np.floating]
    method: CovarianceFactorization
    n_clipped: int = 0
    nugget: float = 0.0


@dataclass
class MassFunctionDraws:
    """One system's Monte Carlo ensemble over ``(m_f, M2)``."""

    source_id: int | None
    n_draws: int
    random_seed: int
    factorization: CovarianceFactorization
    a0_mas: NDArray[np.floating]
    m_f_msun: NDArray[np.floating]
    m2_msun: NDArray[np.floating]
    m1_msun: float
    flux_ratio: float
    n_clipped_eigenvalues: int = 0

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.m2_msun)))

    def probability_m2_above(self, threshold_msun: float) -> float:
        """``P(M2 > threshold)`` as an ensemble fraction of all draws.

        Non-finite inversions count as not exceeding the threshold (conservative).
        """
        if self.n_draws < 1:
            return float("nan")
        return float(np.count_nonzero(self.m2_msun > threshold_msun) / self.n_draws)

    def probability_mc_sigma(self, threshold_msun: float) -> float:
        """Binomial standard error of :meth:`probability_m2_above`."""
        p = self.probability_m2_above(threshold_msun)
        if not np.isfinite(p) or self.n_draws < 1:
            return float("nan")
        return float(np.sqrt(p * (1.0 - p) / self.n_draws))

    def m2_mean(self) -> float:
        finite = self.m2_msun[np.isfinite(self.m2_msun)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def m2_std(self) -> float:
        finite = self.m2_msun[np.isfinite(self.m2_msun)]
        if finite.size < 2:
            return float("nan")
        return float(np.std(finite, ddof=1))

    def m2_snr(self) -> float:
        """``M2 / σ_M2`` from the ensemble, not linearized error propagation."""
        mean = self.m2_mean()
        std = self.m2_std()
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            return float("nan")
        return float(mean / std)


@dataclass
class M2PosteriorConvergenceDiagnostic:
    """§13 ``m2_posterior_convergence``: MC noise on ``P(M2 > threshold)``."""

    n_draws: int
    random_seed: int
    probability_cut: float
    m2_threshold_msun: float
    mc_noise_threshold: float
    boundary_n_sigma: float
    n_systems: int
    n_within_mc_of_boundary: int
    max_probability_sigma: float
    max_mc_poisson_ratio: float
    all_systems_subdominant: bool
    factorization_counts: dict[str, int]
    mc_noise: MCNoiseConvergenceDiagnostic
    per_system: tuple[dict[str, Any], ...] = ()
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        def _finite(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        per_system = []
        for row in self.per_system:
            cleaned = dict(row)
            for key in ("p_m2_above", "p_sigma", "m2_snr"):
                val = cleaned.get(key)
                if isinstance(val, float) and not np.isfinite(val):
                    cleaned[key] = None
            per_system.append(cleaned)
        return {
            "n_draws": self.n_draws,
            "random_seed": self.random_seed,
            "probability_cut": self.probability_cut,
            "m2_threshold_msun": self.m2_threshold_msun,
            "mc_noise_threshold": self.mc_noise_threshold,
            "boundary_n_sigma": self.boundary_n_sigma,
            "n_systems": self.n_systems,
            "n_within_mc_of_boundary": self.n_within_mc_of_boundary,
            "max_probability_sigma": _finite(self.max_probability_sigma),
            "max_mc_poisson_ratio": _finite(self.max_mc_poisson_ratio),
            "all_systems_subdominant": self.all_systems_subdominant,
            "factorization_counts": dict(self.factorization_counts),
            "mc_noise": self.mc_noise.as_dict(),
            "per_system": per_system,
            "message": self.message,
        }


def canonical_nss_name(name: str) -> str:
    """Map a ParameterSet name onto the Gaia NSS canonical stem when known."""
    return _PARAM_ALIASES.get(name, name)


def _index_map(names: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i, raw in enumerate(names):
        mapping[canonical_nss_name(raw)] = i
    return mapping


def factorize_covariance(
    covariance: NDArray[np.floating],
    *,
    eig_rel_floor: float,
    eig_abs_floor: float,
) -> CovarianceFactor:
    """Build a sampling factor for a symmetric, possibly rank-deficient covariance.

    Order: Cholesky → Cholesky with ``eig_abs_floor`` nugget → clipped eigendecomposition.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    symmetric = 0.5 * (cov + cov.T)
    try:
        factor = np.linalg.cholesky(symmetric)
        return CovarianceFactor(factor=factor, method=CovarianceFactorization.CHOLESKY)
    except np.linalg.LinAlgError:
        pass

    nugget = float(max(eig_abs_floor, 0.0))
    if nugget > 0.0:
        try:
            factor = np.linalg.cholesky(symmetric + nugget * np.eye(symmetric.shape[0]))
            return CovarianceFactor(
                factor=factor,
                method=CovarianceFactorization.CHOLESKY_NUGGET,
                nugget=nugget,
            )
        except np.linalg.LinAlgError:
            pass

    evals, evecs = np.linalg.eigh(symmetric)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    floor = max(float(eig_abs_floor), float(eig_rel_floor) * max(abs(max_eval), 0.0))
    clipped = np.where(evals < floor, 0.0, evals)
    n_clipped = int(np.count_nonzero(clipped != evals))
    factor = evecs * np.sqrt(np.maximum(clipped, 0.0))
    return CovarianceFactor(
        factor=factor,
        method=CovarianceFactorization.EIGEN_CLIP,
        n_clipped=n_clipped,
        nugget=nugget,
    )


def sample_multivariate_normal(
    mean: NDArray[np.floating],
    factor: CovarianceFactor,
    n_draws: int,
    rng: Generator,
) -> NDArray[np.floating]:
    """Draw ``n_draws`` from ``N(mean, Σ)`` given ``Σ ≈ factor @ factor.T`` (Cholesky)
    or ``factor @ factor.T`` with ``factor = V √Λ`` (eigen clip).
    """
    z = rng.standard_normal((n_draws, mean.size))
    return mean + z @ factor.factor.T


def propagate_thiele_innes_draws(
    a_mas: ArrayLike,
    b_mas: ArrayLike,
    f_mas: ArrayLike,
    g_mas: ArrayLike,
    parallax_mas: ArrayLike,
    period_day: ArrayLike,
    m1_msun: ArrayLike,
    *,
    flux_ratio: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """§5.1 → §5.2 → §5.3/§5.4 for already-drawn orbital parameters."""
    a0 = photocenter_a0_from_thiele_innes(a_mas, b_mas, f_mas, g_mas)
    m_f = astrometric_mass_function(a0, parallax_mas, period_day)
    m2 = invert_astrometric_companion_mass(m1_msun, m_f, flux_ratio)
    return a0, m_f, m2


def propagate_nss_solution(
    solution: ParameterSet,
    *,
    m1_msun: float,
    n_draws: int,
    random_seed: int,
    flux_ratio: float = 0.0,
    eig_rel_floor: float = 1e-12,
    eig_abs_floor: float = 1e-18,
    source_id: int | None = None,
    covariance_mode: str = FULL_COVARIANCE_MODE,
) -> MassFunctionDraws:
    """Draw the NSS covariance and propagate to ``(m_f, M2)``."""
    if covariance_mode != FULL_COVARIANCE_MODE:
        raise UnhandledCovarianceModeError(
            f"unhandled covariance mode {covariance_mode!r}; only "
            f"{FULL_COVARIANCE_MODE!r} is supported (no diagonal fallback)"
        )
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    names = list(solution.names)
    index = _index_map(names)
    missing = [name for name in _REQUIRED_CANONICAL if name not in index]
    if missing:
        raise ValueError(f"NSS ParameterSet missing required names: {missing}")
    mean = solution.values_array()
    cov = solution.covariance_array()
    factor = factorize_covariance(
        cov, eig_rel_floor=eig_rel_floor, eig_abs_floor=eig_abs_floor
    )
    rng = np.random.default_rng(int(random_seed))
    draws = sample_multivariate_normal(mean, factor, n_draws, rng)
    a = draws[:, index["a_thiele_innes"]]
    b = draws[:, index["b_thiele_innes"]]
    f_ti = draws[:, index["f_thiele_innes"]]
    g = draws[:, index["g_thiele_innes"]]
    plx = draws[:, index["parallax"]]
    period = draws[:, index["period"]]
    a0, m_f, m2 = propagate_thiele_innes_draws(
        a, b, f_ti, g, plx, period, m1_msun, flux_ratio=flux_ratio
    )
    return MassFunctionDraws(
        source_id=source_id,
        n_draws=n_draws,
        random_seed=int(random_seed),
        factorization=factor.method,
        a0_mas=a0,
        m_f_msun=m_f,
        m2_msun=m2,
        m1_msun=float(m1_msun),
        flux_ratio=float(flux_ratio),
        n_clipped_eigenvalues=factor.n_clipped,
    )


def ensemble_row_quantities(
    draws: MassFunctionDraws,
    *,
    m2_threshold_msun: float,
) -> dict[str, Any]:
    """Columns consumed by sample-selection probability / SNR cuts (§6.4, §11)."""
    return {
        "p_m2_above": draws.probability_m2_above(m2_threshold_msun),
        "p_m2_gt_threshold": draws.probability_m2_above(m2_threshold_msun),
        "m2_msun": draws.m2_mean(),
        "sigma_m2_msun": draws.m2_std(),
        "m2_snr": draws.m2_snr(),
        "m_f_msun": float(np.nanmean(draws.m_f_msun)),
        "n_mc_draws": draws.n_draws,
        "n_mc_valid": draws.n_valid,
        "mc_factorization": draws.factorization.value,
        "mc_random_seed": draws.random_seed,
    }


def manifest_seed_record(
    *,
    random_seed: int,
    n_draws: int,
    factorization: str | None = None,
) -> dict[str, Any]:
    """Payload stored on ``RunManifest.random_seeds`` (accounting, not bitwise replay)."""
    record: dict[str, Any] = {
        "random_seed": int(random_seed),
        "n_draws": int(n_draws),
    }
    if factorization is not None:
        record["factorization"] = factorization
    return record


def record_seeds_on_manifest(
    manifest: RunManifest,
    record: Mapping[str, Any],
    *,
    key: str = MANIFEST_SEED_KEY,
) -> RunManifest:
    """Return a copy of ``manifest`` with ``random_seeds[key]`` set."""
    seeds = dict(manifest.random_seeds)
    seeds[key] = dict(record)
    return manifest.model_copy(update={"random_seeds": seeds})


def run_m2_posterior_convergence(
    ensembles: Sequence[MassFunctionDraws],
    *,
    m2_threshold_msun: float,
    probability_cut: float,
    mc_noise_threshold: float,
    boundary_n_sigma: float,
    n_mock_start: int | None = None,
    n_mock_max: int | None = None,
    growth_factor: float = 2.0,
) -> M2PosteriorConvergenceDiagnostic:
    """MC noise on ``P(M2 > threshold)`` plus count near the probability cut.

    Reuses ``evaluate_mc_noise_at_n`` / ``run_mc_noise_convergence``: each system's
    expected success count is ``n_draws * P``, so ``σ_MC / σ_Poisson = 1/√n_draws``
    independent of ``P`` (same identity as the mock-injection guardrail).
    """
    if not 0.0 < probability_cut < 1.0:
        raise ValueError("probability_cut must be in (0, 1)")
    if boundary_n_sigma <= 0.0:
        raise ValueError("boundary_n_sigma must be > 0")
    rows: list[dict[str, Any]] = []
    n_boundary = 0
    sigmas: list[float] = []
    expected_counts: list[float] = []
    n_draws = 0
    seed = 0
    method_counts: dict[str, int] = {}
    for draws in ensembles:
        n_draws = draws.n_draws
        seed = draws.random_seed
        method_counts[draws.factorization.value] = (
            method_counts.get(draws.factorization.value, 0) + 1
        )
        p = draws.probability_m2_above(m2_threshold_msun)
        sigma = draws.probability_mc_sigma(m2_threshold_msun)
        sigmas.append(sigma)
        expected_counts.append(float(draws.n_draws) * p if np.isfinite(p) else 0.0)
        within = bool(
            np.isfinite(p)
            and np.isfinite(sigma)
            and abs(p - probability_cut) <= boundary_n_sigma * sigma
        )
        if within:
            n_boundary += 1
        rows.append(
            {
                "source_id": draws.source_id,
                "p_m2_above": p,
                "p_sigma": sigma,
                "within_mc_of_boundary": within,
                "n_valid": draws.n_valid,
                "m2_snr": draws.m2_snr(),
                "factorization": draws.factorization.value,
            }
        )

    n_eval = n_draws if n_draws > 0 else 1
    start = n_mock_start if n_mock_start is not None else max(10, n_eval // 16 or 1)
    maximum = n_mock_max if n_mock_max is not None else max(n_eval, start)
    # Representative expected counts near the cut, plus the observed ensemble means.
    probe_counts = [float(n_eval) * probability_cut, *expected_counts[:8]]
    mc_noise = run_mc_noise_convergence(
        probe_counts,
        threshold=mc_noise_threshold,
        n_mock_start=start,
        n_mock_max=maximum,
        growth_factor=growth_factor,
    )
    at_final = evaluate_mc_noise_at_n(
        expected_counts or [float(n_eval) * probability_cut],
        n_eval,
        mc_noise_threshold,
    )
    max_ratio = max((row.ratio for row in at_final), default=0.0)
    max_sigma = max((s for s in sigmas if np.isfinite(s)), default=float("nan"))
    all_ok = bool(at_final) and all(row.passed for row in at_final)
    message = (
        f"m2_posterior_convergence: n_draws={n_eval}, n_systems={len(ensembles)}, "
        f"max σ_P={max_sigma:.6g}, max σ_MC/σ_Poisson={max_ratio:.6g} "
        f"(threshold={mc_noise_threshold}), "
        f"{n_boundary} system(s) within {boundary_n_sigma:g} σ_MC of "
        f"P={probability_cut:g}. "
        f"{mc_noise.message}"
    )
    return M2PosteriorConvergenceDiagnostic(
        n_draws=n_eval,
        random_seed=seed,
        probability_cut=probability_cut,
        m2_threshold_msun=float(m2_threshold_msun),
        mc_noise_threshold=float(mc_noise_threshold),
        boundary_n_sigma=float(boundary_n_sigma),
        n_systems=len(ensembles),
        n_within_mc_of_boundary=n_boundary,
        max_probability_sigma=float(max_sigma) if np.isfinite(max_sigma) else float("nan"),
        max_mc_poisson_ratio=float(max_ratio),
        all_systems_subdominant=all_ok,
        factorization_counts=method_counts,
        mc_noise=mc_noise,
        per_system=tuple(rows),
        message=message,
    )


def format_m2_posterior_convergence_report(
    diagnostic: M2PosteriorConvergenceDiagnostic,
) -> str:
    """Full-detail §13 report (caveman exemption)."""
    lines = [
        "m2_posterior_convergence",
        f"  n_draws: {diagnostic.n_draws}",
        f"  random_seed: {diagnostic.random_seed}",
        f"  M2 threshold (Msun): {diagnostic.m2_threshold_msun}",
        f"  probability cut: {diagnostic.probability_cut}",
        f"  physics.mc_noise_threshold: {diagnostic.mc_noise_threshold}",
        f"  boundary_n_sigma: {diagnostic.boundary_n_sigma}",
        f"  n_systems: {diagnostic.n_systems}",
        f"  n_within_mc_of_boundary: {diagnostic.n_within_mc_of_boundary}",
        f"  max_probability_sigma: {diagnostic.max_probability_sigma}",
        f"  max_mc_poisson_ratio: {diagnostic.max_mc_poisson_ratio}",
        f"  all_systems_subdominant: {diagnostic.all_systems_subdominant}",
        f"  factorization_counts: {diagnostic.factorization_counts}",
        f"  mc_noise_guardrail_passed: {diagnostic.mc_noise.all_bins_passed}",
        f"  message: {diagnostic.message}",
        "  per_system:",
    ]
    if not diagnostic.per_system:
        lines.append("    (none)")
    for row in diagnostic.per_system:
        lines.append(
            "    source_id={source_id} P={p_m2_above:.6f} σ_P={p_sigma:.6g} "
            "boundary={within_mc_of_boundary} n_valid={n_valid} "
            "M2/σ_M2={m2_snr} factorization={factorization}".format(**row)
        )
    return "\n".join(lines)


def _default_orbital_mean() -> tuple[list[str], NDArray[np.floating]]:
    names = [
        "ra",
        "dec",
        "parallax",
        "pmra",
        "pmdec",
        "a_thiele_innes",
        "b_thiele_innes",
        "f_thiele_innes",
        "g_thiele_innes",
        "eccentricity",
        "period",
        "t_periastron",
    ]
    a0 = 1.2
    omega = 0.4
    inc = 1.1
    node = 0.7
    a_ti = a0 * (np.cos(omega) * np.cos(node) - np.sin(omega) * np.sin(node) * np.cos(inc))
    b_ti = a0 * (np.cos(omega) * np.sin(node) + np.sin(omega) * np.cos(node) * np.cos(inc))
    f_ti = a0 * (
        -np.sin(omega) * np.cos(node) - np.cos(omega) * np.sin(node) * np.cos(inc)
    )
    g_ti = a0 * (
        -np.sin(omega) * np.sin(node) + np.cos(omega) * np.cos(node) * np.cos(inc)
    )
    mean = np.array(
        [0.0, 0.0, 2.0, 0.0, 0.0, a_ti, b_ti, f_ti, g_ti, 0.2, 365.25, 0.0],
        dtype=np.float64,
    )
    return names, mean


def synthetic_orbital_solution(
    *,
    relative_error: float = 0.02,
    rank_deficient: bool = False,
    seed: int = 0,
) -> ParameterSet:
    """Well-conditioned (or rank-deficient) 12-parameter Orbital-like ParameterSet."""
    names, mean = _default_orbital_mean()
    rng = np.random.default_rng(seed)
    scales = np.maximum(np.abs(mean) * relative_error, 1.0e-3)
    scales[0:2] = 1.0e-4
    scales[3:5] = 0.05
    raw = rng.normal(size=(len(names), len(names)))
    cov = raw @ raw.T
    # Rescale to the requested diagonal.
    current = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    factor = scales / current
    cov = cov * np.outer(factor, factor)
    if rank_deficient:
        evals, evecs = np.linalg.eigh(cov)
        evals[:2] = 0.0
        cov = (evecs * evals) @ evecs.T
        cov = 0.5 * (cov + cov.T)
    return ParameterSet(
        names=names,
        values=[float(v) for v in mean],
        covariance=cov.tolist(),
        provenance="synthetic_nss_orbital",
        units=[
            "deg",
            "deg",
            "mas",
            "mas/yr",
            "mas/yr",
            "mas",
            "mas",
            "mas",
            "mas",
            "1",
            "day",
            "day",
        ],
    )
