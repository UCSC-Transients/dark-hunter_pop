"""Stage: ``population_model``.

Hierarchical multiplicity → type mixture (BH / NS / WD / other / outlier) over a
non-parametric compact-object mass function (ARCHITECTURE.md §4, issue #57).

v1 multiplicity forces ``P(single)=0``, ``P(triple)=0`` (everything NSS = binary
generative branch); the latent multiplicity layer stays generic for a later triples
branch. Type-class rates are functions of mass plus only those covariates
``sensitivity_analysis`` recommends. WD hard-truncates at ``M_Ch``; NS soft/marginalizes
``M_TOV``; outlier is exempt from both. External compact-object mass functions are
never used as priors.

Weight contract with ``companion_nature_likelihood`` (#56): see
:data:`darkhunter_pop.schemas.COMPANION_NATURE_WEIGHT_KEYS`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.config_loader import effective_M_Ch_msun
from darkhunter_pop.config_schema import PipelineConfig, PopulationModelConfig
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    COMPANION_NATURE_WEIGHT_SCHEMA_VERSION,
    CandidateRecord,
    RunManifest,
    StageStatus,
)
from darkhunter_pop.sensitivity_analysis import read_sensitivity_analysis_artifact

SCHEMA_VERSION = 1
PopulationClass = Literal["BH", "NS", "WD", "other", "outlier"]
MassFunctionModelName = Literal["free_height_bins", "gp_log_dndm"]


# ---------------------------------------------------------------------------
# Companion-nature weight schema (#56 ↔ #57)
# ---------------------------------------------------------------------------


def validate_companion_nature_weights(weights: Mapping[str, float]) -> None:
    """Require exactly the five class keys with finite non-negative values."""
    keys = set(weights)
    expected = set(COMPANION_NATURE_WEIGHT_KEYS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            "companion_nature_weights keys must be exactly "
            f"{list(COMPANION_NATURE_WEIGHT_KEYS)}; "
            f"missing={missing} extra={extra}"
        )
    for key, value in weights.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"companion_nature_weights[{key!r}] must be finite and >= 0, got {value}"
            )


def normalize_companion_nature_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Validate and normalize responsibilities to sum to 1.

    A zero-sum vector (all evidence absent) becomes a uniform distribution so the
    system is retained with equal class contamination — never discarded.
    """
    validate_companion_nature_weights(weights)
    raw = {k: float(weights[k]) for k in COMPANION_NATURE_WEIGHT_KEYS}
    total = sum(raw.values())
    if total <= 0.0:
        n = len(COMPANION_NATURE_WEIGHT_KEYS)
        return {k: 1.0 / n for k in COMPANION_NATURE_WEIGHT_KEYS}
    return {k: raw[k] / total for k in COMPANION_NATURE_WEIGHT_KEYS}


def companion_nature_weight_schema() -> dict[str, Any]:
    """Stable JSON contract for #56 producers and #57 consumers."""
    return {
        "schema_version": COMPANION_NATURE_WEIGHT_SCHEMA_VERSION,
        "keys": list(COMPANION_NATURE_WEIGHT_KEYS),
        "normalization": "non_negative_responsibilities_sum_to_one",
        "pre_filter": False,
        "notes": (
            "population_model consumes these as fixed empirical-Bayes plug-in weights; "
            "systems are never discarded for looking non-compact."
        ),
    }


# ---------------------------------------------------------------------------
# Multiplicity + bin edges + truncation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiplicityLayer:
    """Latent multiplicity branch probabilities (generative, not a pre-class)."""

    p_single: float
    p_binary: float
    p_triple: float
    v1_forced_binary: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_single": self.p_single,
            "p_binary": self.p_binary,
            "p_triple": self.p_triple,
            "v1_forced_binary": self.v1_forced_binary,
            "notes": (
                "v1 documents P(single)=P(triple)=0; NSS catalog treated as binary "
                "generative branch. Triple branch reserved for later topology mixture."
            ),
        }


def v1_multiplicity_layer(cfg: PopulationModelConfig) -> MultiplicityLayer:
    """Build the multiplicity layer; refuse non-v1 values silently applied."""
    if abs(cfg.p_single) > 1e-12 or abs(cfg.p_triple) > 1e-12:
        raise ValueError(
            "v1 population_model requires p_single=0 and p_triple=0 "
            f"(got p_single={cfg.p_single}, p_triple={cfg.p_triple})"
        )
    if abs(cfg.p_binary - 1.0) > 1e-12:
        raise ValueError(
            f"v1 population_model requires p_binary=1 (got {cfg.p_binary})"
        )
    return MultiplicityLayer(
        p_single=0.0,
        p_binary=1.0,
        p_triple=0.0,
        v1_forced_binary=True,
    )


def build_mass_bin_edges(
    cfg: PopulationModelConfig,
) -> NDArray[np.floating]:
    """Fix mass-bin edges from the fiducial design — never from real detections.

    ``equal_log_m``: geometric edges over ``[mass_min, mass_max]``.
    ``equal_fiducial_count``: remap equal-log fine grid so cumulative fiducial
    expected counts are equalized across bins (still uses only the design vector).
    """
    n = cfg.n_mass_bins
    m_lo = float(cfg.mass_min_msun)
    m_hi = float(cfg.mass_max_msun)
    if cfg.bin_edge_policy == "equal_log_m":
        return np.geomspace(m_lo, m_hi, n + 1)
    if cfg.bin_edge_policy == "equal_fiducial_count":
        fine = np.geomspace(m_lo, m_hi, 512)
        # Piecewise-constant density on equal-log coarse bins → CDF on fine grid.
        coarse = np.geomspace(m_lo, m_hi, n + 1)
        counts = np.asarray(cfg.fiducial_expected_counts, dtype=np.float64)
        if counts.sum() <= 0:
            return coarse
        dens = counts / np.maximum(np.diff(np.log(coarse)), 1e-30)
        log_m = np.log(fine)
        log_c = np.log(coarse)
        dens_fine = np.zeros_like(fine)
        for i in range(n):
            mask = (log_m >= log_c[i]) & (log_m < log_c[i + 1])
            if i == n - 1:
                mask = (log_m >= log_c[i]) & (log_m <= log_c[i + 1])
            dens_fine[mask] = dens[i]
        # Integrate dens d(log M) ≈ expected count measure.
        dlog = np.diff(log_m)
        mid = 0.5 * (dens_fine[:-1] + dens_fine[1:])
        cdf = np.concatenate([[0.0], np.cumsum(mid * dlog)])
        cdf /= cdf[-1]
        targets = np.linspace(0.0, 1.0, n + 1)
        log_edges = np.interp(targets, cdf, log_m)
        return np.exp(log_edges)
    raise ValueError(f"unknown bin_edge_policy: {cfg.bin_edge_policy!r}")


def wd_hard_truncation_weight(
    mass_msun: ArrayLike,
    m_ch_msun: float,
) -> NDArray[np.floating]:
    """Hard WD cutoff at ``M_Ch`` (no support above)."""
    m = np.asarray(mass_msun, dtype=np.float64)
    return (m <= float(m_ch_msun)).astype(np.float64)


def ns_soft_truncation_weight(
    mass_msun: ArrayLike,
    m_tov_msun: float,
    width_msun: float,
) -> NDArray[np.floating]:
    """Soft logistic survival ``σ(-(M−M_TOV)/w)`` for a fixed ``M_TOV``."""
    if width_msun <= 0:
        raise ValueError("width_msun must be > 0")
    m = np.asarray(mass_msun, dtype=np.float64)
    z = (m - float(m_tov_msun)) / float(width_msun)
    # Stable logistic: 1 / (1 + exp(z))
    return np.where(z >= 0, np.exp(-z) / (1.0 + np.exp(-z)), 1.0 / (1.0 + np.exp(z)))


def ns_soft_truncation_marginalized(
    mass_msun: ArrayLike,
    *,
    m_tov_mean_msun: float,
    m_tov_sigma_msun: float,
    width_msun: float,
    n_quad: int,
) -> NDArray[np.floating]:
    """Marginalize soft NS truncation over a Gaussian ``M_TOV`` prior."""
    if n_quad < 5:
        raise ValueError("n_quad must be >= 5")
    # Gauss-Hermite nodes for N(μ, σ^2): x = μ + σ√2 x_i, w ∝ w_i / √π
    x_i, w_i = np.polynomial.hermite.hermgauss(n_quad)
    mtov = float(m_tov_mean_msun) + float(m_tov_sigma_msun) * math.sqrt(2.0) * x_i
    weights = w_i / math.sqrt(math.pi)
    m = np.asarray(mass_msun, dtype=np.float64)
    acc = np.zeros_like(m, dtype=np.float64)
    for mt, wt in zip(mtov, weights, strict=True):
        if mt <= 0:
            continue
        acc += float(wt) * ns_soft_truncation_weight(m, float(mt), width_msun)
    return acc


def class_truncation_weight(
    population_class: str,
    mass_msun: ArrayLike,
    *,
    m_ch_msun: float,
    m_tov_mean_msun: float,
    m_tov_sigma_msun: float,
    m_tov_width_msun: float,
    m_tov_n_quad: int,
    apply_m_tov: bool,
) -> NDArray[np.floating]:
    """Per-class mass truncation mask / soft weight.

    * WD: hard ``M_Ch``.
    * NS: soft/marginalized ``M_TOV`` when ``apply_m_tov`` else unity (tier-1 raw CO).
    * outlier / other / BH: no ``M_Ch`` / ``M_TOV`` truncation.
    """
    m = np.asarray(mass_msun, dtype=np.float64)
    ones = np.ones_like(m, dtype=np.float64)
    if population_class == "WD":
        return wd_hard_truncation_weight(m, m_ch_msun)
    if population_class == "NS":
        if not apply_m_tov:
            return ones
        return ns_soft_truncation_marginalized(
            m,
            m_tov_mean_msun=m_tov_mean_msun,
            m_tov_sigma_msun=m_tov_sigma_msun,
            width_msun=m_tov_width_msun,
            n_quad=m_tov_n_quad,
        )
    if population_class in {"BH", "other", "outlier"}:
        return ones
    raise ValueError(f"unknown population_class: {population_class!r}")


# ---------------------------------------------------------------------------
# Non-parametric mass function (free-height bins + GP-on-log swap)
# ---------------------------------------------------------------------------


def default_bin_heights(cfg: PopulationModelConfig) -> NDArray[np.floating]:
    """Fiducial free-height vector (proportional to design expected counts)."""
    counts = np.asarray(cfg.fiducial_expected_counts, dtype=np.float64)
    if counts.sum() <= 0:
        return np.ones(cfg.n_mass_bins, dtype=np.float64)
    return counts.copy()


def free_height_dndm(
    mass_msun: ArrayLike,
    bin_edges: NDArray[np.floating],
    heights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Piecewise-constant ``dN/dM`` from free bin heights (expected counts / ΔM)."""
    m = np.asarray(mass_msun, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    h = np.asarray(heights, dtype=np.float64)
    if h.size != edges.size - 1:
        raise ValueError("heights length must equal n_bins")
    delta_m = np.diff(edges)
    dens = h / np.maximum(delta_m, 1e-30)
    idx = np.searchsorted(edges, m, side="right") - 1
    out = np.zeros_like(m, dtype=np.float64)
    valid = (idx >= 0) & (idx < dens.size) & (m >= edges[0]) & (m <= edges[-1])
    out[valid] = dens[idx[valid]]
    return out


def gp_log_dndm(
    mass_msun: ArrayLike,
    bin_edges: NDArray[np.floating],
    heights: NDArray[np.floating],
    *,
    length_scale_log_m: float,
    variance: float,
) -> NDArray[np.floating]:
    """GP-on-log(dN/dM) via RBF smoothing of free-height log-density.

    Bin centers carry ``log(height/ΔM)``; an RBF kernel with the configured length
    scale / variance interpolates ``log(dN/dM)`` at query masses. Swappable second
    model for inference model comparison — not an external CO MF prior.
    """
    m = np.asarray(mass_msun, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    h = np.asarray(heights, dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])  # log-mid
    delta_m = np.diff(edges)
    log_dens = np.log(np.maximum(h / np.maximum(delta_m, 1e-30), 1e-30))
    log_m = np.log(np.maximum(m, 1e-30))
    log_c = np.log(centers)
    ell2 = float(length_scale_log_m) ** 2
    diff_cc = log_c[:, None] - log_c[None, :]
    k_cc = float(variance) * np.exp(-0.5 * (diff_cc**2) / ell2)
    k_cc = k_cc + 1e-8 * np.eye(k_cc.shape[0])
    diff_qc = log_m[:, None] - log_c[None, :]
    k_qc = float(variance) * np.exp(-0.5 * (diff_qc**2) / ell2)
    alpha = np.linalg.solve(k_cc, log_dens)
    log_pred = k_qc @ alpha
    return np.exp(log_pred)


def evaluate_mass_function(
    mass_msun: ArrayLike,
    *,
    model: MassFunctionModelName,
    bin_edges: NDArray[np.floating],
    heights: NDArray[np.floating],
    cfg: PopulationModelConfig,
) -> NDArray[np.floating]:
    """Dispatch free-height vs GP-on-log mass-function evaluation."""
    if model == "free_height_bins":
        return free_height_dndm(mass_msun, bin_edges, heights)
    if model == "gp_log_dndm":
        return gp_log_dndm(
            mass_msun,
            bin_edges,
            heights,
            length_scale_log_m=cfg.gp_length_scale_log_m,
            variance=cfg.gp_variance,
        )
    raise ValueError(f"unknown mass_function_model: {model!r}")


# ---------------------------------------------------------------------------
# Auxiliary parametric families (hooks for inference)
# ---------------------------------------------------------------------------


def m1_pdf(mass_msun: ArrayLike, family: str) -> NDArray[np.floating]:
    """Primary-mass PDF hook (Kroupa broken power-law, IMF comparison only)."""
    m = np.asarray(mass_msun, dtype=np.float64)
    if family != "kroupa":
        raise ValueError(f"unsupported m1_family: {family!r}")
    # Kroupa (2001) high-mass slope α=2.3 for M>0.5; unnormalized on support.
    out = np.zeros_like(m)
    low = (m >= 0.08) & (m < 0.5)
    high = m >= 0.5
    out[low] = m[low] ** (-1.3)
    out[high] = m[high] ** (-2.3)
    return out


def period_pdf(period_day: ArrayLike, family: str) -> NDArray[np.floating]:
    """Orbital-period PDF hook (flat-in-log-P or Moe & Di Stefano label)."""
    p = np.asarray(period_day, dtype=np.float64)
    if family == "flat_log_p":
        return np.where(p > 0, 1.0 / p, 0.0)
    if family == "moe_di_stefano":
        # Named comparison hook: same flat-log placeholder shape; inference swaps.
        return np.where(p > 0, 1.0 / p, 0.0)
    raise ValueError(f"unsupported period_family: {family!r}")


def eccentricity_pdf(eccentricity: ArrayLike, family: str) -> NDArray[np.floating]:
    """Eccentricity PDF hook (thermal or SN-kick-informed label)."""
    e = np.asarray(eccentricity, dtype=np.float64)
    if family == "thermal":
        return np.where((e >= 0) & (e <= 1), 2.0 * e, 0.0)
    if family == "sn_kick":
        return np.where((e >= 0) & (e <= 1), 3.0 * e**2, 0.0)
    raise ValueError(f"unsupported eccentricity_family: {family!r}")


# ---------------------------------------------------------------------------
# Covariate opt-in from sensitivity_analysis
# ---------------------------------------------------------------------------


def covariates_from_sensitivity_payload(
    payload: Mapping[str, Any] | None,
    *,
    population_classes: Sequence[str],
    apply: bool,
) -> dict[str, tuple[str, ...]]:
    """Per-class covariates beyond mass; mass-only when artifact absent or opted out."""
    base = {name: tuple() for name in population_classes}
    if not apply or payload is None:
        return base
    rows = payload.get("class_covariates") or []
    for row in rows:
        name = str(row["population_class"])
        if name not in base:
            continue
        selected = tuple(str(x) for x in row.get("selected_covariates", []))
        base[name] = selected
    return base


# ---------------------------------------------------------------------------
# Rate functions + two-tier dN/dM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassRateSpec:
    """Rate-function specification for one population class."""

    population_class: str
    covariates_beyond_mass: tuple[str, ...]
    mass_always_included: bool = True
    applies_m_ch: bool = False
    applies_m_tov: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "population_class": self.population_class,
            "covariates_beyond_mass": list(self.covariates_beyond_mass),
            "mass_always_included": self.mass_always_included,
            "applies_m_ch": self.applies_m_ch,
            "applies_m_tov": self.applies_m_tov,
        }


def build_class_rate_specs(
    population_classes: Sequence[str],
    covariates: Mapping[str, tuple[str, ...]],
) -> list[ClassRateSpec]:
    """Construct per-class rate specs with truncation flags."""
    specs: list[ClassRateSpec] = []
    for name in population_classes:
        specs.append(
            ClassRateSpec(
                population_class=name,
                covariates_beyond_mass=tuple(covariates.get(name, ())),
                applies_m_ch=(name == "WD"),
                applies_m_tov=(name == "NS"),
            )
        )
    return specs


def class_rate_density(
    mass_msun: ArrayLike,
    population_class: str,
    *,
    bin_edges: NDArray[np.floating],
    heights: NDArray[np.floating],
    cfg: PopulationModelConfig,
    m_ch_msun: float,
    m_tov_mean_msun: float,
    apply_m_tov: bool,
    class_fraction: float = 1.0,
) -> NDArray[np.floating]:
    """``rate_k(M2) = class_fraction × MF(M) × truncation_k(M)``."""
    mf = evaluate_mass_function(
        mass_msun,
        model=cfg.mass_function_model,
        bin_edges=bin_edges,
        heights=heights,
        cfg=cfg,
    )
    trunc = class_truncation_weight(
        population_class,
        mass_msun,
        m_ch_msun=m_ch_msun,
        m_tov_mean_msun=m_tov_mean_msun,
        m_tov_sigma_msun=cfg.m_tov_prior_sigma_msun,
        m_tov_width_msun=cfg.m_tov_soft_width_msun,
        m_tov_n_quad=cfg.m_tov_prior_n_quad,
        apply_m_tov=apply_m_tov,
    )
    return float(class_fraction) * mf * trunc


@dataclass
class TwoTierDnDm:
    """Two-tier compact-object mass-function outputs (ARCHITECTURE.md §4)."""

    mass_grid_msun: NDArray[np.floating]
    # Tier 1: raw total CO dN/dM, classification-independent, no M_TOV assumption.
    raw_total_co_dndm: NDArray[np.floating]
    # Tier 2: species-classified dN/dM with M_TOV marginalized for NS.
    classified_dndm: dict[str, NDArray[np.floating]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mass_grid_msun": self.mass_grid_msun.tolist(),
            "raw_total_co_dndm": self.raw_total_co_dndm.tolist(),
            "classified_dndm": {k: v.tolist() for k, v in self.classified_dndm.items()},
        }


def evaluate_two_tier_dndm(
    mass_grid_msun: ArrayLike,
    *,
    bin_edges: NDArray[np.floating],
    heights: NDArray[np.floating],
    cfg: PopulationModelConfig,
    m_ch_msun: float,
    m_tov_mean_msun: float,
    class_fractions: Mapping[str, float] | None = None,
) -> TwoTierDnDm:
    """Evaluate tier-1 raw CO and tier-2 classified ``dN/dM`` on a mass grid."""
    grid = np.asarray(mass_grid_msun, dtype=np.float64)
    fracs = {name: 1.0 for name in cfg.population_classes}
    if class_fractions is not None:
        fracs.update({k: float(v) for k, v in class_fractions.items()})

    classified: dict[str, NDArray[np.floating]] = {}
    for name in cfg.population_classes:
        classified[name] = class_rate_density(
            grid,
            name,
            bin_edges=bin_edges,
            heights=heights,
            cfg=cfg,
            m_ch_msun=m_ch_msun,
            m_tov_mean_msun=m_tov_mean_msun,
            apply_m_tov=True,
            class_fraction=fracs.get(name, 1.0),
        )

    raw = np.zeros_like(grid, dtype=np.float64)
    for name in cfg.compact_object_classes:
        raw = raw + class_rate_density(
            grid,
            name,
            bin_edges=bin_edges,
            heights=heights,
            cfg=cfg,
            m_ch_msun=m_ch_msun,
            m_tov_mean_msun=m_tov_mean_msun,
            apply_m_tov=False,
            class_fraction=fracs.get(name, 1.0),
        )
    return TwoTierDnDm(
        mass_grid_msun=grid,
        raw_total_co_dndm=raw,
        classified_dndm=classified,
    )


# ---------------------------------------------------------------------------
# Per-system plug-in weights + stage result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemPopulationWeight:
    """Empirical-Bayes plug-in row for one system (never a discard flag)."""

    source_id: int
    m2_msun: float | None
    weights: dict[str, float]
    responsibilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "m2_msun": self.m2_msun,
            "weights": dict(self.weights),
            "responsibilities": dict(self.responsibilities),
        }


def _m2_point(candidate: CandidateRecord) -> float | None:
    if candidate.m2 is None or not candidate.m2.names:
        return None
    if "M2" in candidate.m2.names:
        idx = list(candidate.m2.names).index("M2")
        return float(candidate.m2.values[idx])
    return float(candidate.m2.values[0])


def collect_system_weights(
    candidates: Sequence[CandidateRecord],
) -> list[SystemPopulationWeight]:
    """Extract / normalize companion_nature weights; retain every system."""
    rows: list[SystemPopulationWeight] = []
    for cand in candidates:
        if cand.companion_nature_weights is None:
            raw = {k: 1.0 for k in COMPANION_NATURE_WEIGHT_KEYS}
        else:
            validate_companion_nature_weights(cand.companion_nature_weights)
            raw = {
                k: float(cand.companion_nature_weights[k])
                for k in COMPANION_NATURE_WEIGHT_KEYS
            }
        resp = normalize_companion_nature_weights(raw)
        rows.append(
            SystemPopulationWeight(
                source_id=int(cand.source_id),
                m2_msun=_m2_point(cand),
                weights=raw,
                responsibilities=resp,
            )
        )
    return rows


def assert_no_external_co_mf_priors(cfg: PopulationModelConfig) -> None:
    """Hard rule: pulsar / LIGO / literature CO MFs are comparison-only."""
    if cfg.allow_external_co_mf_priors is not False:
        raise ValueError(
            "Hard rule violated: external compact-object mass functions must not "
            "be used as inference priors (ARCHITECTURE.md §4)."
        )


@dataclass
class PopulationModelResult:
    """Full stage output before HDF5 serialization."""

    schema_version: int
    multiplicity: MultiplicityLayer
    class_rates: list[ClassRateSpec]
    mass_function_model: str
    bin_edges_msun: NDArray[np.floating]
    bin_heights: NDArray[np.floating]
    m_ch_msun: float
    m_tov_msun: float
    two_tier: TwoTierDnDm
    system_weights: list[SystemPopulationWeight] = field(default_factory=list)
    covariates_applied: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sensitivity_artifact_used: bool = False
    aux_families: dict[str, str] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    weight_schema: dict[str, Any] = field(default_factory=companion_nature_weight_schema)

    def recommendation_payload(self) -> dict[str, Any]:
        """Stable consumer contract for ``inference`` / diagnostics."""
        return {
            "schema_version": self.schema_version,
            "stage": "population_model",
            "weight_schema": self.weight_schema,
            "multiplicity": self.multiplicity.as_dict(),
            "class_rates": [c.as_dict() for c in self.class_rates],
            "mass_function_model": self.mass_function_model,
            "bin_edges_msun": self.bin_edges_msun.tolist(),
            "bin_heights": self.bin_heights.tolist(),
            "m_ch_msun": self.m_ch_msun,
            "m_tov_msun": self.m_tov_msun,
            "two_tier_dndm": self.two_tier.as_dict(),
            "covariates_applied": {
                k: list(v) for k, v in self.covariates_applied.items()
            },
            "sensitivity_artifact_used": self.sensitivity_artifact_used,
            "aux_families": dict(self.aux_families),
            "n_systems": len(self.system_weights),
            "external_co_mf_priors": False,
            "notes": (
                "Staged-but-connected: companion_nature weights are fixed plug-ins. "
                "External CO mass functions are comparison-only, never priors."
            ),
        }


def run_population_model(
    config: PipelineConfig,
    *,
    candidates: Sequence[CandidateRecord] | None = None,
    sensitivity_payload: Mapping[str, Any] | None = None,
    sensitivity_artifact_path: Path | None = None,
    mass_grid: ArrayLike | None = None,
) -> PopulationModelResult:
    """Build the hierarchical population model and two-tier ``dN/dM`` outputs."""
    cfg = config.population_model
    assert_no_external_co_mf_priors(cfg)
    multiplicity = v1_multiplicity_layer(cfg)
    m_ch = effective_M_Ch_msun(config)
    m_tov = float(config.classification.M_TOV_msun)

    payload = sensitivity_payload
    used_sa = False
    if payload is None and sensitivity_artifact_path is not None:
        payload = read_sensitivity_analysis_artifact(sensitivity_artifact_path)
        used_sa = True
    elif payload is not None:
        used_sa = True

    covariates = covariates_from_sensitivity_payload(
        payload,
        population_classes=cfg.population_classes,
        apply=cfg.apply_sensitivity_covariates,
    )
    if payload is None:
        used_sa = False

    class_rates = build_class_rate_specs(cfg.population_classes, covariates)
    edges = build_mass_bin_edges(cfg)
    heights = default_bin_heights(cfg)

    if mass_grid is None:
        grid = np.sqrt(edges[:-1] * edges[1:])
    else:
        grid = np.asarray(mass_grid, dtype=np.float64)

    systems = collect_system_weights(candidates or ())
    if systems:
        mean_frac = {
            k: float(np.mean([s.responsibilities[k] for s in systems]))
            for k in COMPANION_NATURE_WEIGHT_KEYS
        }
    else:
        n = len(COMPANION_NATURE_WEIGHT_KEYS)
        mean_frac = {k: 1.0 / n for k in COMPANION_NATURE_WEIGHT_KEYS}

    two_tier = evaluate_two_tier_dndm(
        grid,
        bin_edges=edges,
        heights=heights,
        cfg=cfg,
        m_ch_msun=m_ch,
        m_tov_mean_msun=m_tov,
        class_fractions=mean_frac,
    )

    return PopulationModelResult(
        schema_version=SCHEMA_VERSION,
        multiplicity=multiplicity,
        class_rates=class_rates,
        mass_function_model=cfg.mass_function_model,
        bin_edges_msun=edges,
        bin_heights=heights,
        m_ch_msun=m_ch,
        m_tov_msun=m_tov,
        two_tier=two_tier,
        system_weights=systems,
        covariates_applied=covariates,
        sensitivity_artifact_used=used_sa and cfg.apply_sensitivity_covariates,
        aux_families={
            "m1": cfg.m1_family,
            "period": cfg.period_family,
            "eccentricity": cfg.eccentricity_family,
        },
        config_snapshot=cfg.model_dump(mode="json"),
    )


def write_population_model_artifact(path: Path, result: PopulationModelResult) -> None:
    """Write stage HDF5 with model spec, two-tier dN/dM, and plug-in weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.recommendation_payload()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "population_model"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["mass_function_model"] = result.mass_function_model
        handle.attrs["m_ch_msun"] = result.m_ch_msun
        handle.attrs["m_tov_msun"] = result.m_tov_msun
        handle.attrs["sensitivity_artifact_used"] = result.sensitivity_artifact_used
        handle.attrs["external_co_mf_priors"] = False
        handle.attrs["p_single"] = result.multiplicity.p_single
        handle.attrs["p_binary"] = result.multiplicity.p_binary
        handle.attrs["p_triple"] = result.multiplicity.p_triple

        rec = handle.create_group("model")
        rec.create_dataset(
            "payload_json",
            data=np.array(
                json.dumps(payload, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        rec.create_dataset("bin_edges_msun", data=result.bin_edges_msun)
        rec.create_dataset("bin_heights", data=result.bin_heights)

        tier = handle.create_group("two_tier_dndm")
        tier.create_dataset("mass_grid_msun", data=result.two_tier.mass_grid_msun)
        tier.create_dataset(
            "raw_total_co_dndm", data=result.two_tier.raw_total_co_dndm
        )
        cls_grp = tier.create_group("classified")
        for name, arr in result.two_tier.classified_dndm.items():
            cls_grp.create_dataset(name, data=arr)

        wt = handle.create_group("system_weights")
        wt.attrs["schema_version"] = COMPANION_NATURE_WEIGHT_SCHEMA_VERSION
        wt.attrs["n_systems"] = len(result.system_weights)
        if result.system_weights:
            wt.create_dataset(
                "source_id",
                data=np.asarray(
                    [s.source_id for s in result.system_weights], dtype=np.int64
                ),
            )
            m2 = np.asarray(
                [
                    np.nan if s.m2_msun is None else s.m2_msun
                    for s in result.system_weights
                ],
                dtype=np.float64,
            )
            wt.create_dataset("m2_msun", data=m2)
            for key in COMPANION_NATURE_WEIGHT_KEYS:
                wt.create_dataset(
                    f"responsibility_{key}",
                    data=np.asarray(
                        [s.responsibilities[key] for s in result.system_weights],
                        dtype=np.float64,
                    ),
                )


def read_population_model_artifact(path: Path) -> dict[str, Any]:
    """Load the model payload JSON from a stage HDF5 (consumer contract)."""
    with h5py.File(path, "r") as handle:
        raw = handle["model"]["payload_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        return json.loads(text)


def format_population_model_report(result: PopulationModelResult) -> str:
    """Fully legible diagnostic report (exempt from caveman compression)."""
    lines = [
        "=== population_model report ===",
        (
            f"multiplicity: P(single)={result.multiplicity.p_single} "
            f"P(binary)={result.multiplicity.p_binary} "
            f"P(triple)={result.multiplicity.p_triple} "
            f"(v1_forced_binary={result.multiplicity.v1_forced_binary})"
        ),
        f"mass_function_model: {result.mass_function_model}",
        f"n_bins: {result.bin_edges_msun.size - 1}",
        f"M_Ch_msun: {result.m_ch_msun}",
        f"M_TOV_msun: {result.m_tov_msun}",
        f"n_systems: {len(result.system_weights)}",
        f"sensitivity_artifact_used: {result.sensitivity_artifact_used}",
        f"aux_families: {result.aux_families}",
        "class rates (mass always included):",
    ]
    for spec in result.class_rates:
        cov = ",".join(spec.covariates_beyond_mass) or "(none)"
        lines.append(
            f"  {spec.population_class}: covariates=[{cov}] "
            f"M_Ch={spec.applies_m_ch} M_TOV={spec.applies_m_tov}"
        )
    lines.append(
        "two-tier dN/dM: (1) raw total CO without M_TOV; "
        "(2) species-classified with M_TOV marginalized for NS."
    )
    lines.append(
        "hard rule: external CO mass functions are comparison-only, never priors."
    )
    lines.append("=== end population_model report ===")
    return "\n".join(lines)


def run_population_model_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
    sensitivity_artifact_path: Path | None = None,
) -> RunManifest:
    """Execute ``population_model``: build model, write HDF5, update manifest."""
    spec = STAGE_REGISTRY["population_model"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_population_model(
        config,
        candidates=candidates,
        sensitivity_artifact_path=sensitivity_artifact_path,
    )
    write_population_model_artifact(artifact, result)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
