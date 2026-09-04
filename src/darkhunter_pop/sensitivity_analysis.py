"""Stage: ``sensitivity_analysis``.

Unified module for (a) joint N-D vs collapse to 1D dN/dM and (b) per-class covariates
beyond mass (ARCHITECTURE.md §4). Enforces mock-injection
``sigma_MC / sigma_Poisson < physics.mc_noise_threshold`` with a required convergence
diagnostic. Emits recommendation artifacts for ``population_model`` / ``inference`` —
those modules' defaults are never rewritten here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_schema import PipelineConfig, SensitivityAnalysisConfig
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import RunManifest, StageStatus

PopulationClass = Literal["BH", "NS", "WD", "other", "outlier"]
DimensionalityChoice = Literal["1d_dndm", "joint_nd"]
LikelihoodForm = Literal["unbinned", "binned"]


@dataclass(frozen=True)
class BinMCNoiseResult:
    """Per-bin Monte-Carlo vs Poisson noise comparison."""

    bin_index: int
    expected_count: float
    n_mock: int
    sigma_mc: float
    sigma_poisson: float
    ratio: float
    passed: bool


@dataclass(frozen=True)
class MCNoiseConvergenceDiagnostic:
    """Required convergence diagnostic for the mock-injection noise budget."""

    threshold: float
    n_mock_final: int
    all_bins_passed: bool
    per_bin: tuple[BinMCNoiseResult, ...]
    schedule_n_mock: tuple[int, ...]
    schedule_max_ratio: tuple[float, ...]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_mock_final": self.n_mock_final,
            "all_bins_passed": self.all_bins_passed,
            "schedule_n_mock": list(self.schedule_n_mock),
            "schedule_max_ratio": list(self.schedule_max_ratio),
            "message": self.message,
            "per_bin": [
                {
                    "bin_index": b.bin_index,
                    "expected_count": b.expected_count,
                    "n_mock": b.n_mock,
                    "sigma_mc": b.sigma_mc,
                    "sigma_poisson": b.sigma_poisson,
                    "ratio": b.ratio,
                    "passed": b.passed,
                }
                for b in self.per_bin
            ],
        }


@dataclass(frozen=True)
class DimensionalityRecommendation:
    """Consumable by ``inference``: joint N-D vs 1D dN/dM (+ binned vs unbinned)."""

    preferred_model: DimensionalityChoice
    preferred_likelihood: LikelihoodForm
    delta_bic: float
    bic_1d: float
    bic_nd: float
    mean_count_per_nd_bin: float
    dimension_names: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "preferred_model": self.preferred_model,
            "preferred_likelihood": self.preferred_likelihood,
            "delta_bic": self.delta_bic,
            "bic_1d": self.bic_1d,
            "bic_nd": self.bic_nd,
            "mean_count_per_nd_bin": self.mean_count_per_nd_bin,
            "dimension_names": list(self.dimension_names),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ClassCovariateRecommendation:
    """Consumable by ``population_model``: covariates beyond mass for one class."""

    population_class: str
    selected_covariates: tuple[str, ...]
    tested_covariates: tuple[str, ...]
    delta_bic_by_covariate: dict[str, float]
    mass_always_included: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "population_class": self.population_class,
            "selected_covariates": list(self.selected_covariates),
            "tested_covariates": list(self.tested_covariates),
            "delta_bic_by_covariate": dict(self.delta_bic_by_covariate),
            "mass_always_included": self.mass_always_included,
        }


@dataclass
class SensitivityAnalysisResult:
    """Full stage output before HDF5 serialization."""

    mc_noise: MCNoiseConvergenceDiagnostic
    dimensionality: DimensionalityRecommendation
    class_covariates: list[ClassCovariateRecommendation] = field(default_factory=list)
    mc_noise_threshold: float = 0.1
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def recommendation_payload(self) -> dict[str, Any]:
        """Stable JSON-serializable record for downstream stages (read-only contract)."""
        return {
            "schema_version": 1,
            "stage": "sensitivity_analysis",
            "mc_noise_threshold": self.mc_noise_threshold,
            "mc_noise_convergence": self.mc_noise.as_dict(),
            "dimensionality": self.dimensionality.as_dict(),
            "class_covariates": [c.as_dict() for c in self.class_covariates],
            "notes": (
                "Recommendations only. population_model and inference must opt in "
                "explicitly; this stage does not rewrite their defaults."
            ),
        }


def sigma_mc_poisson_ratio(expected_count: float, n_mock: int) -> tuple[float, float, float]:
    """Return ``(sigma_mc, sigma_poisson, ratio)`` for one bin.

    For a Poisson mean ``μ`` estimated from ``n_mock`` independent mock draws,
    ``σ_MC = √(μ / n_mock)`` and ``σ_Poisson = √μ``, so
    ``ratio = σ_MC / σ_Poisson = 1 / √n_mock`` when ``μ > 0``.
    """
    if n_mock < 1:
        raise ValueError("n_mock must be >= 1")
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")
    if expected_count == 0.0:
        return 0.0, 0.0, 0.0
    sigma_poisson = math.sqrt(expected_count)
    sigma_mc = math.sqrt(expected_count / float(n_mock))
    return sigma_mc, sigma_poisson, sigma_mc / sigma_poisson


def minimum_n_mock_for_threshold(threshold: float) -> int:
    """Smallest ``n_mock`` with ``1/√n_mock < threshold`` (strict)."""
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    # n > 1/threshold^2; float edges (e.g. 0.1**2) need a strict check.
    n = max(1, math.floor(1.0 / (threshold * threshold)))
    while (1.0 / math.sqrt(n)) >= threshold:
        n += 1
    return n


def evaluate_mc_noise_at_n(
    expected_counts: Sequence[float],
    n_mock: int,
    threshold: float,
) -> tuple[BinMCNoiseResult, ...]:
    """Evaluate the MC/Poisson ratio at a fixed mock volume for every bin."""
    results: list[BinMCNoiseResult] = []
    for index, mu in enumerate(expected_counts):
        sigma_mc, sigma_poisson, ratio = sigma_mc_poisson_ratio(float(mu), n_mock)
        results.append(
            BinMCNoiseResult(
                bin_index=index,
                expected_count=float(mu),
                n_mock=n_mock,
                sigma_mc=sigma_mc,
                sigma_poisson=sigma_poisson,
                ratio=ratio,
                passed=ratio < threshold,
            )
        )
    return tuple(results)


def run_mc_noise_convergence(
    expected_counts: Sequence[float],
    *,
    threshold: float,
    n_mock_start: int,
    n_mock_max: int,
    growth_factor: float,
) -> MCNoiseConvergenceDiagnostic:
    """Grow mock volume until every bin satisfies the noise budget, or fail.

    Records the full ``(n_mock, max_ratio)`` schedule as the required convergence
    diagnostic (ARCHITECTURE.md §4; dark-hunter-pop-workflow §7).
    """
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be > 1")
    if n_mock_start < 1 or n_mock_max < n_mock_start:
        raise ValueError("invalid n_mock_start / n_mock_max")

    schedule_n: list[int] = []
    schedule_max: list[float] = []
    n_mock = n_mock_start
    last_bins: tuple[BinMCNoiseResult, ...] = ()

    while n_mock <= n_mock_max:
        bins = evaluate_mc_noise_at_n(expected_counts, n_mock, threshold)
        last_bins = bins
        max_ratio = max((b.ratio for b in bins), default=0.0)
        schedule_n.append(n_mock)
        schedule_max.append(max_ratio)
        if all(b.passed for b in bins):
            msg = (
                f"MC noise budget met: all {len(bins)} bins have "
                f"sigma_MC/sigma_Poisson < {threshold} at n_mock={n_mock} "
                f"(max_ratio={max_ratio:.6g})."
            )
            return MCNoiseConvergenceDiagnostic(
                threshold=threshold,
                n_mock_final=n_mock,
                all_bins_passed=True,
                per_bin=bins,
                schedule_n_mock=tuple(schedule_n),
                schedule_max_ratio=tuple(schedule_max),
                message=msg,
            )
        next_n = int(math.ceil(n_mock * growth_factor))
        if next_n <= n_mock:
            next_n = n_mock + 1
        n_mock = next_n

    max_ratio = max((b.ratio for b in last_bins), default=float("inf"))
    msg = (
        f"MC noise budget NOT met within n_mock_max={n_mock_max}: "
        f"max_ratio={max_ratio:.6g} still >= threshold={threshold}."
    )
    return MCNoiseConvergenceDiagnostic(
        threshold=threshold,
        n_mock_final=schedule_n[-1] if schedule_n else n_mock_max,
        all_bins_passed=False,
        per_bin=last_bins,
        schedule_n_mock=tuple(schedule_n),
        schedule_max_ratio=tuple(schedule_max),
        message=msg,
    )


def _poisson_bic(log_likelihood: float, n_params: int, n_events: int) -> float:
    """BIC = k ln n − 2 ln L (n clamped to ≥ 1 for empty catalogs)."""
    n_eff = max(int(n_events), 1)
    return float(n_params * math.log(n_eff) - 2.0 * log_likelihood)


def _joint_powerlaw_log_likelihood(
    mass: NDArray[np.floating],
    aux: NDArray[np.floating],
    *,
    mass_lo: float,
    mass_hi: float,
    aux_lo: float,
    aux_hi: float,
    mass_index: float,
    aux_index: float,
) -> float:
    """Unbinned PPP log-likelihood on a rectangular (mass, aux) domain.

    Intensity ``λ ∝ m^{α} · aux^{β}``. The 1D dN/dM collapse uses ``β = −1``
    (log-flat auxiliary coordinate — the standard orbital-period null).
    """
    m = np.asarray(mass, dtype=np.float64)
    a = np.asarray(aux, dtype=np.float64)
    mask = (
        (m >= mass_lo)
        & (m <= mass_hi)
        & (a >= aux_lo)
        & (a <= aux_hi)
    )
    m = m[mask]
    a = a[mask]
    n = int(m.size)
    if n == 0:
        return 0.0

    def _mass_z(alpha: float) -> float:
        if abs(alpha + 1.0) < 1e-12:
            return math.log(mass_hi / mass_lo)
        return (mass_hi ** (alpha + 1.0) - mass_lo ** (alpha + 1.0)) / (alpha + 1.0)

    def _aux_z(beta: float) -> float:
        if abs(beta + 1.0) < 1e-12:
            return math.log(aux_hi / aux_lo)
        return (aux_hi ** (beta + 1.0) - aux_lo ** (beta + 1.0)) / (beta + 1.0)

    alpha = mass_index
    beta = aux_index
    z_m = _mass_z(alpha)
    z_a = _aux_z(beta)
    log_a = math.log(n) - math.log(z_m * z_a)
    return float(
        np.sum(alpha * np.log(m) + beta * np.log(np.clip(a, 1e-300, None)))
        + n * log_a
        - n
    )


def _grid_best_joint(
    mass: NDArray[np.floating],
    aux: NDArray[np.floating],
    *,
    mass_lo: float,
    mass_hi: float,
    aux_lo: float,
    aux_hi: float,
    fix_aux_index: float | None,
) -> tuple[float, float, float]:
    """Grid-search indices; returns ``(mass_index, aux_index, logL)``.

    When ``fix_aux_index`` is set, only the mass index is free (1D collapse with
    that fixed auxiliary index, typically ``−1`` for log-flat period).
    """
    mass_grid = np.linspace(-3.0, 1.0, 41)
    if fix_aux_index is not None:
        best = (-1.0, float(fix_aux_index), -float("inf"))
        for alpha in mass_grid:
            ll = _joint_powerlaw_log_likelihood(
                mass,
                aux,
                mass_lo=mass_lo,
                mass_hi=mass_hi,
                aux_lo=aux_lo,
                aux_hi=aux_hi,
                mass_index=float(alpha),
                aux_index=float(fix_aux_index),
            )
            if ll > best[2]:
                best = (float(alpha), float(fix_aux_index), ll)
        return best

    aux_grid = np.linspace(-3.0, 1.0, 41)
    best_nd = (-1.0, -1.0, -float("inf"))
    for alpha in mass_grid:
        for beta in aux_grid:
            ll = _joint_powerlaw_log_likelihood(
                mass,
                aux,
                mass_lo=mass_lo,
                mass_hi=mass_hi,
                aux_lo=aux_lo,
                aux_hi=aux_hi,
                mass_index=float(alpha),
                aux_index=float(beta),
            )
            if ll > best_nd[2]:
                best_nd = (float(alpha), float(beta), ll)
    return best_nd


def recommend_dimensionality(
    catalog: Mapping[str, NDArray[np.floating]],
    cfg: SensitivityAnalysisConfig,
) -> DimensionalityRecommendation:
    """Compare 1D mass-only vs joint mass×aux intensity models via BIC.

    Both models are inhomogeneous Poisson intensities on the same (mass, aux)
    domain. The 1D model fixes a log-flat auxiliary index (``β = −1``); the joint
    model frees ``β``. Prefers unbinned likelihood when a hypothetical N-D
    histogram would be sparse.
    """
    dims = tuple(cfg.joint_dimensions)
    mass = np.asarray(catalog["mass_msun"], dtype=np.float64)
    n_events = int(mass.size)
    aux_name = dims[1] if len(dims) > 1 else None
    if aux_name is None or aux_name not in catalog:
        raise KeyError("joint_dimensions must include an auxiliary column after mass_msun")
    aux = np.asarray(catalog[aux_name], dtype=np.float64)
    aux = np.clip(aux, 1e-6, None)
    aux_lo = float(np.min(aux))
    aux_hi = float(np.max(aux))
    if aux_hi <= aux_lo:
        aux_hi = aux_lo * 1.1

    _, _, log_l_1d = _grid_best_joint(
        mass,
        aux,
        mass_lo=cfg.mass_min_msun,
        mass_hi=cfg.mass_max_msun,
        aux_lo=aux_lo,
        aux_hi=aux_hi,
        fix_aux_index=-1.0,
    )
    bic_1d = _poisson_bic(log_l_1d, n_params=2, n_events=n_events)  # A, α

    _, _, log_l_nd = _grid_best_joint(
        mass,
        aux,
        mass_lo=cfg.mass_min_msun,
        mass_hi=cfg.mass_max_msun,
        aux_lo=aux_lo,
        aux_hi=aux_hi,
        fix_aux_index=None,
    )
    bic_nd = _poisson_bic(log_l_nd, n_params=3, n_events=n_events)  # A, α, β

    delta_bic = bic_1d - bic_nd  # positive ⇒ joint preferred

    n_aux_bins = 4
    n_params_hist = cfg.n_mass_bins * (n_aux_bins ** max(len(dims) - 1, 0))
    mean_per_bin = float(n_events) / float(max(n_params_hist, 1))

    if delta_bic >= cfg.bic_delta_prefer_joint_nd:
        preferred_model: DimensionalityChoice = "joint_nd"
        model_reason = (
            f"ΔBIC={delta_bic:.3f} >= {cfg.bic_delta_prefer_joint_nd} "
            f"favours joint N-D over 1D dN/dM."
        )
    else:
        preferred_model = "1d_dndm"
        model_reason = (
            f"ΔBIC={delta_bic:.3f} < {cfg.bic_delta_prefer_joint_nd} "
            f"— collapse to 1D dN/dM."
        )

    if mean_per_bin < cfg.mean_count_per_bin_unbinned_preference:
        preferred_likelihood: LikelihoodForm = "unbinned"
        like_reason = (
            f"mean count/bin={mean_per_bin:.3f} < "
            f"{cfg.mean_count_per_bin_unbinned_preference} → prefer unbinned."
        )
    else:
        preferred_likelihood = "binned"
        like_reason = (
            f"mean count/bin={mean_per_bin:.3f} >= "
            f"{cfg.mean_count_per_bin_unbinned_preference} → binned acceptable."
        )

    return DimensionalityRecommendation(
        preferred_model=preferred_model,
        preferred_likelihood=preferred_likelihood,
        delta_bic=delta_bic,
        bic_1d=bic_1d,
        bic_nd=bic_nd,
        mean_count_per_nd_bin=mean_per_bin,
        dimension_names=dims,
        rationale=f"{model_reason} {like_reason}",
    )


def _logistic_log_likelihood(
    y: NDArray[np.floating],
    logits: NDArray[np.floating],
) -> float:
    """Bernoulli log-likelihood with numerically stable softplus."""
    # log p = y log σ(z) + (1-y) log(1-σ(z)) = y z − softplus(z)
    softplus = np.logaddexp(0.0, logits)
    return float(np.sum(y * logits - softplus))


def _fit_logistic_mle(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    n_steps: int = 80,
    lr: float = 0.15,
) -> tuple[float, NDArray[np.floating], float]:
    """Gradient-ascent MLE for ``logit p = β0 + x · β``; returns ``(β0, β, logL)``."""
    n_features = x.shape[1]
    beta0 = 0.0
    beta = np.zeros(n_features, dtype=np.float64)
    for _ in range(n_steps):
        logits = beta0 + x @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        resid = y - p
        beta0 += lr * float(np.mean(resid))
        beta += lr * (x.T @ resid) / max(len(y), 1)
    logits = beta0 + x @ beta
    return beta0, beta, _logistic_log_likelihood(y, logits)


@dataclass(frozen=True)
class BinaryCovariateRecommendation:
    """ΔBIC covariate selection for a binary outcome (e.g. spuriousness).

    Consumable by ``spuriousness_model``: retain only covariates the module
    justifies. Complete-case per covariate — rows with a missing value for that
    candidate are excluded from that candidate's ΔBIC, not dropped globally.
    """

    selected_covariates: tuple[str, ...]
    tested_covariates: tuple[str, ...]
    dropped_covariates: tuple[str, ...]
    delta_bic_by_covariate: dict[str, float]
    n_complete_by_covariate: dict[str, int]
    n_events: int
    n_positives: int
    bic_delta_threshold: float
    drop_reasons: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_covariates": list(self.selected_covariates),
            "tested_covariates": list(self.tested_covariates),
            "dropped_covariates": list(self.dropped_covariates),
            "delta_bic_by_covariate": dict(self.delta_bic_by_covariate),
            "n_complete_by_covariate": dict(self.n_complete_by_covariate),
            "n_events": self.n_events,
            "n_positives": self.n_positives,
            "bic_delta_threshold": self.bic_delta_threshold,
            "drop_reasons": dict(self.drop_reasons),
        }


def recommend_binary_outcome_covariates(
    design: Mapping[str, NDArray[np.floating]],
    y: NDArray[np.floating],
    candidate_covariates: Sequence[str],
    *,
    bic_delta_include: float,
    min_complete_rows: int = 20,
) -> BinaryCovariateRecommendation:
    """Select covariates for a binary outcome via complete-case ΔBIC.

    Baseline is intercept-only logistic. A candidate is retained when
    ``BIC(intercept) − BIC(intercept + covariate) >= bic_delta_include`` on
    rows where that covariate is finite. Candidates with too few finite rows
    are dropped with an explicit reason — never silently.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n_events = int(y_arr.size)
    n_positives = int(np.sum(y_arr > 0.5))
    delta_by_cov: dict[str, float] = {}
    n_complete: dict[str, int] = {}
    drop_reasons: dict[str, str] = {}
    selected: list[str] = []

    for cov_name in candidate_covariates:
        if cov_name not in design:
            delta_by_cov[cov_name] = float("nan")
            n_complete[cov_name] = 0
            drop_reasons[cov_name] = "absent_from_design"
            continue
        col = np.asarray(design[cov_name], dtype=np.float64).reshape(-1)
        if col.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"covariate {cov_name!r} length {col.shape[0]} != y length {y_arr.shape[0]}"
            )
        mask = np.isfinite(col) & np.isfinite(y_arr)
        n_ok = int(np.sum(mask))
        n_complete[cov_name] = n_ok
        if n_ok < min_complete_rows:
            delta_by_cov[cov_name] = float("nan")
            drop_reasons[cov_name] = (
                f"insufficient_complete_rows ({n_ok} < {min_complete_rows})"
            )
            continue
        y_m = y_arr[mask]
        if float(np.sum(y_m > 0.5)) < 1.0 or float(np.sum(y_m <= 0.5)) < 1.0:
            delta_by_cov[cov_name] = float("nan")
            drop_reasons[cov_name] = "complete_cases_lack_both_classes"
            continue
        # Intercept-only on the same complete-case subset.
        x0 = np.zeros((n_ok, 0), dtype=np.float64)
        _, _, log_l0 = _fit_logistic_mle(x0, y_m)
        bic0 = _poisson_bic(log_l0, n_params=1, n_events=n_ok)
        scale = float(np.std(col[mask])) or 1.0
        cov_z = ((col[mask] - float(np.mean(col[mask]))) / scale).reshape(-1, 1)
        _, _, log_l1 = _fit_logistic_mle(cov_z, y_m)
        bic1 = _poisson_bic(log_l1, n_params=2, n_events=n_ok)
        delta = bic0 - bic1
        delta_by_cov[cov_name] = float(delta)
        if delta >= bic_delta_include:
            selected.append(cov_name)
        else:
            drop_reasons[cov_name] = (
                f"delta_bic={delta:.3f} < threshold={bic_delta_include}"
            )

    tested = tuple(candidate_covariates)
    selected_t = tuple(selected)
    dropped = tuple(c for c in tested if c not in selected_t)
    return BinaryCovariateRecommendation(
        selected_covariates=selected_t,
        tested_covariates=tested,
        dropped_covariates=dropped,
        delta_bic_by_covariate=delta_by_cov,
        n_complete_by_covariate=n_complete,
        n_events=n_events,
        n_positives=n_positives,
        bic_delta_threshold=float(bic_delta_include),
        drop_reasons=drop_reasons,
    )


def recommend_class_covariates(
    catalog: Mapping[str, NDArray[np.floating]],
    class_labels: NDArray[np.str_],
    cfg: SensitivityAnalysisConfig,
) -> list[ClassCovariateRecommendation]:
    """Per type-mixture class, select covariates beyond mass via ΔBIC.

    Mass is always included. A candidate covariate is selected when
    ``BIC(mass-only) − BIC(mass+covariate) >= bic_delta_include_covariate``.
    """
    mass = np.asarray(catalog["mass_msun"], dtype=np.float64)
    log_mass = np.log(np.clip(mass, 1e-6, None)).reshape(-1, 1)
    recommendations: list[ClassCovariateRecommendation] = []

    for class_name in cfg.population_classes:
        y = (class_labels == class_name).astype(np.float64)
        n_events = int(y.size)
        _, _, log_l_mass = _fit_logistic_mle(log_mass, y)
        bic_mass = _poisson_bic(log_l_mass, n_params=2, n_events=n_events)  # β0 + β_M

        delta_by_cov: dict[str, float] = {}
        selected: list[str] = []
        for cov_name in cfg.candidate_covariates:
            if cov_name not in catalog:
                raise KeyError(f"covariate {cov_name!r} missing from catalog")
            cov = np.asarray(catalog[cov_name], dtype=np.float64).reshape(-1, 1)
            # Standardize covariate for stable MLE.
            scale = float(np.std(cov)) or 1.0
            cov_z = (cov - float(np.mean(cov))) / scale
            x_ext = np.hstack([log_mass, cov_z])
            _, _, log_l_ext = _fit_logistic_mle(x_ext, y)
            bic_ext = _poisson_bic(log_l_ext, n_params=3, n_events=n_events)
            delta = bic_mass - bic_ext
            delta_by_cov[cov_name] = delta
            if delta >= cfg.bic_delta_include_covariate:
                selected.append(cov_name)

        recommendations.append(
            ClassCovariateRecommendation(
                population_class=class_name,
                selected_covariates=tuple(selected),
                tested_covariates=tuple(cfg.candidate_covariates),
                delta_bic_by_covariate=delta_by_cov,
                mass_always_included=True,
            )
        )
    return recommendations


def generate_fiducial_catalog(
    cfg: SensitivityAnalysisConfig,
    *,
    rng: np.random.Generator | None = None,
    nd_signal: bool = False,
    covariate_signals: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, NDArray[np.floating]], NDArray[np.str_]]:
    """Synthetic catalog for sensitivity tests when ``population_model`` is absent.

    Parameters
    ----------
    nd_signal
        When True, detection density depends on period as well as mass (favours N-D).
    covariate_signals
        Optional ``{class_name: [covariate, ...]}`` injecting class←covariate dependence
        so ΔBIC can select those covariates in tests.
    """
    rng = rng or np.random.default_rng(cfg.random_seed)
    n = cfg.n_synthetic_systems
    # Log-uniform masses in the configured window.
    mass = np.exp(
        rng.uniform(math.log(cfg.mass_min_msun), math.log(cfg.mass_max_msun), size=n)
    )
    period = 10.0 ** rng.uniform(1.0, 3.5, size=n)  # days
    eccentricity = rng.beta(1.5, 4.0, size=n)
    ruwe = 1.0 + rng.exponential(0.3, size=n)

    if nd_signal:
        # Down-weight long-period systems so period enters the joint density.
        keep_prob = 1.0 / (1.0 + (period / 300.0) ** 2)
        keep = rng.random(n) < keep_prob
        # Ensure we retain enough systems.
        if keep.sum() < max(n // 4, 20):
            keep = np.ones(n, dtype=bool)
        mass, period, eccentricity, ruwe = (
            mass[keep],
            period[keep],
            eccentricity[keep],
            ruwe[keep],
        )
        n = int(mass.size)

    classes = np.asarray(cfg.population_classes, dtype=np.str_)
    # Base class draws; WD preference at lower mass.
    logits = np.zeros((n, len(classes)), dtype=np.float64)
    for i, name in enumerate(classes):
        if name == "WD":
            logits[:, i] = 1.5 - np.log(mass)
        elif name == "NS":
            logits[:, i] = -0.2 * (np.log(mass) - math.log(1.4)) ** 2
        elif name == "BH":
            logits[:, i] = -0.3 * (np.log(mass) - math.log(10.0)) ** 2
        elif name == "outlier":
            logits[:, i] = -1.5 + 0.5 * (ruwe - 1.0)
        else:
            logits[:, i] = -0.5

    signals = covariate_signals or {}
    for class_name, cov_names in signals.items():
        if class_name not in classes:
            continue
        idx = int(np.where(classes == class_name)[0][0])
        for cov_name in cov_names:
            if cov_name == "ruwe":
                logits[:, idx] += 2.5 * (ruwe - 1.0)
            elif cov_name == "eccentricity":
                logits[:, idx] += 2.0 * eccentricity
            elif cov_name == "period_day":
                logits[:, idx] += 0.8 * (np.log10(period) - 2.0)

    # Softmax sample.
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    draws = np.array(
        [classes[rng.choice(len(classes), p=probs[i])] for i in range(n)],
        dtype=np.str_,
    )

    catalog: dict[str, NDArray[np.floating]] = {
        "mass_msun": mass.astype(np.float64),
        "period_day": period.astype(np.float64),
        "eccentricity": eccentricity.astype(np.float64),
        "ruwe": ruwe.astype(np.float64),
    }
    return catalog, draws


def run_sensitivity_analysis(
    config: PipelineConfig,
    *,
    catalog: Mapping[str, NDArray[np.floating]] | None = None,
    class_labels: NDArray[np.str_] | None = None,
    nd_signal: bool = False,
    covariate_signals: Mapping[str, Sequence[str]] | None = None,
    require_mc_pass: bool = True,
) -> SensitivityAnalysisResult:
    """Execute sensitivity analysis and return recommendation records.

    When ``catalog`` is omitted, a fiducial synthetic catalog is generated so this
    stage can run before ``population_model`` lands (Phase 4). Downstream stages
    must read the HDF5 recommendations explicitly — defaults are not patched.
    """
    sa = config.sensitivity_analysis
    threshold = float(config.physics.mc_noise_threshold)

    mc = run_mc_noise_convergence(
        sa.fiducial_expected_counts,
        threshold=threshold,
        n_mock_start=sa.n_mock_start,
        n_mock_max=sa.n_mock_max,
        growth_factor=sa.n_mock_growth_factor,
    )
    if require_mc_pass and not mc.all_bins_passed:
        raise RuntimeError(mc.message)

    if catalog is None or class_labels is None:
        catalog, class_labels = generate_fiducial_catalog(
            sa,
            rng=np.random.default_rng(sa.random_seed),
            nd_signal=nd_signal,
            covariate_signals=covariate_signals,
        )

    dimensionality = recommend_dimensionality(catalog, sa)
    class_cov = recommend_class_covariates(catalog, class_labels, sa)

    return SensitivityAnalysisResult(
        mc_noise=mc,
        dimensionality=dimensionality,
        class_covariates=class_cov,
        mc_noise_threshold=threshold,
        config_snapshot=sa.model_dump(mode="json"),
    )


def write_sensitivity_analysis_artifact(
    path: Path,
    result: SensitivityAnalysisResult,
) -> None:
    """Persist one HDF5 artifact with recommendation records + MC diagnostic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.recommendation_payload()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "sensitivity_analysis"
        handle.attrs["schema_version"] = 1
        handle.attrs["mc_noise_threshold"] = result.mc_noise_threshold
        handle.attrs["mc_noise_passed"] = result.mc_noise.all_bins_passed
        handle.attrs["n_mock_final"] = result.mc_noise.n_mock_final
        handle.attrs["preferred_model"] = result.dimensionality.preferred_model
        handle.attrs["preferred_likelihood"] = result.dimensionality.preferred_likelihood

        rec = handle.create_group("recommendations")
        rec.create_dataset(
            "payload_json",
            data=np.array(
                json.dumps(payload, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )

        mc_grp = handle.create_group("mc_noise_convergence")
        mc_grp.attrs["threshold"] = result.mc_noise.threshold
        mc_grp.attrs["all_bins_passed"] = result.mc_noise.all_bins_passed
        mc_grp.attrs["message"] = result.mc_noise.message
        mc_grp.create_dataset(
            "schedule_n_mock",
            data=np.asarray(result.mc_noise.schedule_n_mock, dtype=np.int64),
        )
        mc_grp.create_dataset(
            "schedule_max_ratio",
            data=np.asarray(result.mc_noise.schedule_max_ratio, dtype=np.float64),
        )
        if result.mc_noise.per_bin:
            mc_grp.create_dataset(
                "bin_expected_count",
                data=np.asarray(
                    [b.expected_count for b in result.mc_noise.per_bin],
                    dtype=np.float64,
                ),
            )
            mc_grp.create_dataset(
                "bin_ratio",
                data=np.asarray(
                    [b.ratio for b in result.mc_noise.per_bin],
                    dtype=np.float64,
                ),
            )
            mc_grp.create_dataset(
                "bin_passed",
                data=np.asarray(
                    [b.passed for b in result.mc_noise.per_bin],
                    dtype=bool,
                ),
            )

        dim = handle.create_group("dimensionality")
        for key, value in result.dimensionality.as_dict().items():
            if isinstance(value, list):
                dim.create_dataset(
                    key,
                    data=np.array(value, dtype=h5py.string_dtype("utf-8")),
                )
            elif isinstance(value, str):
                dim.attrs[key] = value
            else:
                dim.attrs[key] = value

        cov_grp = handle.create_group("class_covariates")
        for rec_c in result.class_covariates:
            g = cov_grp.create_group(rec_c.population_class)
            g.attrs["mass_always_included"] = rec_c.mass_always_included
            g.create_dataset(
                "selected_covariates",
                data=np.array(
                    list(rec_c.selected_covariates),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            g.create_dataset(
                "tested_covariates",
                data=np.array(
                    list(rec_c.tested_covariates),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            for name, delta in rec_c.delta_bic_by_covariate.items():
                g.attrs[f"delta_bic_{name}"] = delta


def read_sensitivity_analysis_artifact(path: Path) -> dict[str, Any]:
    """Load the recommendation payload from a stage HDF5 (consumer contract)."""
    with h5py.File(path, "r") as handle:
        raw = handle["recommendations"]["payload_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        return json.loads(text)


def format_sensitivity_report(result: SensitivityAnalysisResult) -> str:
    """Fully legible diagnostic report (exempt from caveman compression)."""
    lines = [
        "=== sensitivity_analysis report ===",
        f"mc_noise_threshold: {result.mc_noise_threshold}",
        result.mc_noise.message,
        (
            f"dimensionality: preferred_model={result.dimensionality.preferred_model} "
            f"preferred_likelihood={result.dimensionality.preferred_likelihood} "
            f"ΔBIC={result.dimensionality.delta_bic:.3f}"
        ),
        f"  rationale: {result.dimensionality.rationale}",
        "class covariates (mass always included):",
    ]
    for rec in result.class_covariates:
        selected = ",".join(rec.selected_covariates) or "(none)"
        lines.append(f"  {rec.population_class}: selected=[{selected}]")
        for name, delta in sorted(rec.delta_bic_by_covariate.items()):
            lines.append(f"    ΔBIC({name})={delta:.3f}")
    lines.append(
        "note: recommendations are artifacts only; "
        "population_model / inference defaults are not rewritten."
    )
    lines.append("=== end sensitivity_analysis report ===")
    return "\n".join(lines)


def run_sensitivity_analysis_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    require_mc_pass: bool = True,
) -> RunManifest:
    """Execute ``sensitivity_analysis``: MC gate, recommendations, HDF5, manifest."""
    spec = STAGE_REGISTRY["sensitivity_analysis"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_sensitivity_analysis(config, require_mc_pass=require_mc_pass)
    write_sensitivity_analysis_artifact(artifact, result)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
