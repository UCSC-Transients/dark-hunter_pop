"""Multi-sample Poisson inclusion operator (CONTINUATION_PLAN §4.7, §15 Q1).

Resolves the three-way sample-overlap double-counting question for v1 inference.

Q1 decision — unified inclusion-indicator (not separate Poisson processes)
---------------------------------------------------------------------------
The inference samples overlap three ways, not pairwise:

* El-Badry 2024 drew candidates from Andrews et al. (2022).
* El-Badry 2026 astrometric subsample 3 *is* the Andrews selection restricted
  to ``G < 15`` (§8.7).
* Gaia BH1 / Table E1 membership further couples the three catalogs.

Separate Poisson processes with ``Σ_s log L(rate_s)`` (or ``Λ = Σ_s Λ_s``) count
the same physical system once per sample that admits it. That is wrong when the
observed catalogs are overlapping subsets of one parent population.

The unified inclusion-indicator formulation treats membership as a mark on a
single point process. For sample survival probabilities ``p_s(x)``:

    p_any(x) = 1 − Π_s (1 − p_s(x))

    λ(x|θ) = population_model(θ) × SF_astro × SF_followup × p_any(x)

Each unique ``source_id`` enters the data term once (staged-but-connected plug-in
weights). Nested / overlapping cuts never inflate ``Λ``. Separate per-sample
Poisson processes remain valid only for *isolated* single-sample analyses; they
are rejected for the joint v1 likelihood.

v1 stays staged-but-connected (workflow §7): per-system nature evidence is a
fixed weight. Outcome-dependent criteria (§7.3, §8.5) are applied identically to
mocks via this operator and never as silent filters on real systems.
``accel_jerk`` stays disabled (§9). Inference uses ``andrews2022_modified``
(not frozen ``andrews2022``) and only El-Badry 2026's astrometric branch
(§6.6, §8.4.1). Per-sample spurious fractions under ``validation_targets`` are
never read — ``not_spurious`` calls the shared ``P(spurious | ·)`` model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    InclusionOperatorSpec,
    OutcomeDependentCriterionSpec,
    PipelineConfig,
    SampleSelectionFile,
)
from darkhunter_pop.sample_selection import (
    load_sample_selection_file,
    resolve_inherits,
)
from darkhunter_pop.sensitivity_analysis import BinaryCovariateRecommendation
from darkhunter_pop.spuriousness_model import (
    FittedSpuriousnessModel,
    PerSampleSpuriousRateReadError,
    assert_no_per_sample_spurious_rate_input,
    predict_p_spurious,
)

SCHEMA_VERSION = 1

OverlapFormulation = Literal[
    "unified_inclusion_indicator",
    "separate_poisson",
]

# Inference-facing samples (CONTINUATION_PLAN Slot J). Frozen andrews2022 and
# elbadry2026 SB1 are deliberately absent; accel_jerk is blocked (§9).
DEFAULT_INFERENCE_SAMPLE_NAMES: tuple[str, ...] = (
    "andrews2022_modified",
    "elbadry2024",
    "elbadry2026",
)


def intercept_only_spurious_model(
    *,
    p_spurious: float = 0.25,
) -> FittedSpuriousnessModel:
    """Build Φ(intercept)=``p_spurious`` for CI/SBC when no #111 fit is loaded.

    Not a per-sample rate. Production should prefer a fitted artifact from
    ``spuriousness_model.fit_spuriousness_model``.
    """
    p = float(p_spurious)
    if not 0.0 < p < 1.0:
        raise ValueError("p_spurious must lie in (0, 1)")
    intercept = float(norm.ppf(p))
    empty_sens = BinaryCovariateRecommendation(
        selected_covariates=(),
        tested_covariates=(),
        dropped_covariates=(),
        delta_bic_by_covariate={},
        n_complete_by_covariate={},
        n_events=0,
        n_positives=0,
        bic_delta_threshold=0.0,
        drop_reasons={},
    )
    return FittedSpuriousnessModel(
        schema_version=1,
        link_function="probit_intercept",
        censoring="none",
        retained_covariates=(),
        outcome_feature_names=(),
        selection_feature_names=(),
        covariate_means={},
        covariate_scales={},
        outcome_intercept=intercept,
        outcome_coef={},
        selection_intercept=0.0,
        selection_coef={},
        rho=0.0,
        rho_fixed=True,
        regularization_l2=0.0,
        sensitivity=empty_sens,
        literature_interaction={},
        n_labeled=0,
        n_spurious=0,
        n_genuine=0,
        n_undetermined=0,
        log_likelihood=0.0,
        identified=True,
        identification_notes="intercept_only test double for inclusion operator",
    )

Q1_JUSTIFICATION = __doc__ or ""


@dataclass(frozen=True)
class MockRealization:
    """One mock system for inclusion / sample-SF evaluation."""

    source_id: int
    mass_msun: float
    g_mag: float | None = None
    period_days: float | None = None
    m2_joint_fit_msun: float | None = None
    orbit_coverage_fraction: float | None = None
    goodness_of_fit: float | None = None
    significance: float | None = None
    implied_companion_mass_msun: float | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def as_covariate_map(self) -> dict[str, float]:
        """Covariates for ``predict_p_spurious`` (NaN → missing)."""
        out: dict[str, float] = {
            "period_days": _nan_if_none(self.period_days),
            "phot_g_mean_mag": _nan_if_none(self.g_mag),
            "goodness_of_fit": _nan_if_none(self.goodness_of_fit),
            "significance": _nan_if_none(self.significance),
            "implied_companion_mass_msun": _nan_if_none(
                self.implied_companion_mass_msun
                if self.implied_companion_mass_msun is not None
                else self.mass_msun
            ),
        }
        for key, value in self.extras.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                out[str(key)] = float(value)
        return out


def _nan_if_none(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class SampleInclusionConfig:
    """Resolved multi-sample inclusion settings for one inference run."""

    formulation: OverlapFormulation
    sample_names: tuple[str, ...]
    default_catalog_sf: Mapping[str, float]
    spurious_model: FittedSpuriousnessModel | None
    selection_files: Mapping[str, SampleSelectionFile]


@dataclass
class SampleOverlapMatrix:
    """Pairwise and three-way source-ID overlap (§13 ``sample_overlap_matrix``)."""

    sample_names: tuple[str, ...]
    pairwise_counts: dict[tuple[str, str], int]
    pairwise_ids: dict[tuple[str, str], frozenset[int]]
    three_way_count: int
    three_way_ids: frozenset[int]
    per_sample_counts: dict[str, int]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "diagnostic": "sample_overlap_matrix",
            "sample_names": list(self.sample_names),
            "per_sample_counts": dict(self.per_sample_counts),
            "pairwise_counts": {
                f"{a}|{b}": n for (a, b), n in sorted(self.pairwise_counts.items())
            },
            "three_way_count": self.three_way_count,
            "three_way_source_ids": sorted(self.three_way_ids),
            "notes": self.notes
            or (
                "Three-way overlap is required: El-Badry 2024 ⊂ Andrews candidates; "
                "El-Badry 2026 subsample 3 = Andrews ∩ (G < 15). "
                "Unified inclusion-indicator avoids double-counting."
            ),
        }


def build_sample_overlap_matrix(
    membership: Mapping[str, Sequence[int]],
) -> SampleOverlapMatrix:
    """Emit pairwise + three-way overlap from per-sample source-ID sets."""
    names = tuple(sorted(membership.keys()))
    sets = {name: frozenset(int(s) for s in membership[name]) for name in names}
    pairwise_counts: dict[tuple[str, str], int] = {}
    pairwise_ids: dict[tuple[str, str], frozenset[int]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sets[a] & sets[b]
            pairwise_counts[(a, b)] = len(inter)
            pairwise_ids[(a, b)] = inter
    three_way_ids: frozenset[int] = frozenset()
    if len(names) >= 3:
        three_way_ids = sets[names[0]]
        for name in names[1:]:
            three_way_ids &= sets[name]
    return SampleOverlapMatrix(
        sample_names=names,
        pairwise_counts=pairwise_counts,
        pairwise_ids=pairwise_ids,
        three_way_count=len(three_way_ids),
        three_way_ids=three_way_ids,
        per_sample_counts={n: len(sets[n]) for n in names},
    )


def _criterion_probability(
    criterion: OutcomeDependentCriterionSpec,
    mock: MockRealization,
    *,
    spurious_model: FittedSpuriousnessModel | None,
) -> float:
    """P(criterion satisfied | mock). Never reads validation_targets rates."""
    kind = criterion.kind
    if kind == "m2_joint_fit":
        threshold = criterion.m2_min_msun
        if threshold is None:
            raise ValueError(f"criterion {criterion.id!r}: m2_min_msun required")
        m2 = mock.m2_joint_fit_msun
        if m2 is None:
            # Unknown joint-fit mass → use true mock mass as the forward-model proxy.
            m2 = mock.mass_msun
        return 1.0 if float(m2) > float(threshold) else 0.0
    if kind == "not_spurious":
        if spurious_model is None:
            raise ValueError(
                f"criterion {criterion.id!r}: shared spuriousness model required"
            )
        p_spur = float(predict_p_spurious(spurious_model, mock.as_covariate_map())[0])
        return float(max(0.0, min(1.0, 1.0 - p_spur)))
    if kind == "orbit_coverage":
        frac_min = criterion.orbit_coverage_min_fraction
        if frac_min is None:
            raise ValueError(
                f"criterion {criterion.id!r}: orbit_coverage_min_fraction required"
            )
        cov = mock.orbit_coverage_fraction
        if cov is None:
            # Forward-model default: full coverage when mock does not carry the
            # follow-up observable (catalog-level mocks).
            return 1.0
        return 1.0 if float(cov) >= float(frac_min) else 0.0
    if kind == "followup_characterized":
        # Soft prior on being characterized in the follow-up subset (§8.5).
        # Probability is config-owned when supplied via extras; else pass.
        raw = mock.extras.get("p_followup_characterized")
        if raw is None:
            return 1.0
        return float(max(0.0, min(1.0, float(raw))))
    # Exhaustive for Literal kinds.
    raise ValueError(f"unhandled outcome-dependent criterion kind: {kind!r}")


def outcome_dependent_inclusion_probability(
    spec: InclusionOperatorSpec | None,
    mock: MockRealization,
    *,
    spurious_model: FittedSpuriousnessModel | None,
) -> float:
    """Product of §7.3 / §8.5 outcome-dependent terms for one mock."""
    if spec is None or not spec.outcome_dependent_criteria:
        return 1.0
    prob = 1.0
    for criterion in spec.outcome_dependent_criteria:
        prob *= _criterion_probability(
            criterion, mock, spurious_model=spurious_model
        )
    return float(prob)


def sample_inclusion_probability(
    catalog_sf: float,
    inclusion_spec: InclusionOperatorSpec | None,
    mock: MockRealization,
    *,
    spurious_model: FittedSpuriousnessModel | None,
) -> float:
    """``p_s(x) = S_catalog_s(x) × Π outcome criteria``."""
    p_out = outcome_dependent_inclusion_probability(
        inclusion_spec, mock, spurious_model=spurious_model
    )
    return float(max(0.0, min(1.0, float(catalog_sf) * p_out)))


def unified_any_sample_probability(
    per_sample_p: Sequence[float],
) -> float:
    """``p_any = 1 − Π_s (1 − p_s)`` — unified inclusion-indicator kernel."""
    survival = 1.0
    for p in per_sample_p:
        survival *= 1.0 - float(max(0.0, min(1.0, p)))
    return float(1.0 - survival)


def separate_poisson_sum_probability(
    per_sample_p: Sequence[float],
) -> float:
    """Naive ``Σ p_s`` (can exceed 1) — retained only to demonstrate double-counting."""
    return float(sum(float(max(0.0, p)) for p in per_sample_p))


def combine_sample_probabilities(
    per_sample_p: Sequence[float],
    *,
    formulation: OverlapFormulation,
) -> float:
    """Combine per-sample inclusion probs under the chosen Q1 formulation."""
    if formulation == "unified_inclusion_indicator":
        return unified_any_sample_probability(per_sample_p)
    if formulation == "separate_poisson":
        return separate_poisson_sum_probability(per_sample_p)
    raise ValueError(f"unknown overlap formulation: {formulation!r}")


def mean_sample_selection_multiplier(
    *,
    formulation: OverlapFormulation,
    catalog_sfs: Mapping[str, float],
    sample_names: Sequence[str],
    inclusion_specs: Mapping[str, InclusionOperatorSpec | None],
    mocks: Sequence[MockRealization],
    spurious_model: FittedSpuriousnessModel | None,
) -> float:
    """Monte Carlo estimate of ``E[p_any(x)]`` (or naive sum) over mocks.

    When ``mocks`` is empty, fall back to combining the catalog SF scalars alone
    with outcome terms at their default (pass) values via a unit mock.
    """
    names = list(sample_names)
    if not names:
        return 1.0
    if not mocks:
        unit = MockRealization(source_id=0, mass_msun=1.0)
        per = [
            sample_inclusion_probability(
                float(catalog_sfs.get(name, 1.0)),
                inclusion_specs.get(name),
                unit,
                spurious_model=spurious_model,
            )
            for name in names
        ]
        return combine_sample_probabilities(per, formulation=formulation)

    vals = np.empty(len(mocks), dtype=np.float64)
    for i, mock in enumerate(mocks):
        per = [
            sample_inclusion_probability(
                float(catalog_sfs.get(name, 1.0)),
                inclusion_specs.get(name),
                mock,
                spurious_model=spurious_model,
            )
            for name in names
        ]
        vals[i] = combine_sample_probabilities(per, formulation=formulation)
    return float(np.mean(vals))


def load_inference_selection_files(
    config: PipelineConfig,
    sample_names: Sequence[str],
    *,
    root: Path | None = None,
) -> dict[str, SampleSelectionFile]:
    """Load frozen selection YAMLs for inference sample names."""
    root = root or repo_root()
    by_name = {entry.name: entry for entry in config.sample_selection.samples}
    # Load every registry file so inherits / depends_on resolve.
    all_files: dict[str, SampleSelectionFile] = {}
    for entry in config.sample_selection.samples:
        path = Path(entry.path)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            all_files[entry.name] = load_sample_selection_file(path)
    out: dict[str, SampleSelectionFile] = {}
    for name in sample_names:
        if name not in by_name:
            raise KeyError(
                f"inference sample {name!r} missing from sample_selection registry"
            )
        if name not in all_files:
            raise FileNotFoundError(
                f"selection file for inference sample {name!r} not found "
                f"({by_name[name].path})"
            )
        out[name] = resolve_inherits(all_files[name], files_by_name=all_files)
    return out


def inclusion_specs_from_files(
    files: Mapping[str, SampleSelectionFile],
) -> dict[str, InclusionOperatorSpec | None]:
    return {name: f.inclusion_operator for name, f in files.items()}


def resolve_inference_sample_names(config: PipelineConfig) -> tuple[str, ...]:
    """Inference samples from config; drop accel_jerk; never use frozen andrews2022."""
    names = list(config.inference.multi_sample.sample_names)
    cleaned: list[str] = []
    for name in names:
        if name == "accel_jerk":
            continue
        if name == "andrews2022":
            raise ValueError(
                "inference must use andrews2022_modified (N=25), not frozen andrews2022 (§6.6)"
            )
        cleaned.append(name)
    return tuple(cleaned)


def elbadry2026_inference_branch_ok(selection: SampleSelectionFile) -> None:
    """Guard §8.4.1: only the astrometric branch feeds the likelihood."""
    branches = selection.inference_branches
    if not branches:
        return
    if branches != ["astrometric"]:
        raise ValueError(
            "elbadry2026 v1 inference_branches must be exactly ['astrometric'] "
            f"(§8.4.1); got {branches!r}"
        )


def build_sample_inclusion_context(
    config: PipelineConfig,
    *,
    spurious_model: FittedSpuriousnessModel | None = None,
    catalog_sf_overrides: Mapping[str, float] | None = None,
) -> SampleInclusionConfig:
    """Assemble inclusion context; refuse per-sample spurious-rate inputs."""
    assert_no_per_sample_spurious_rate_input(config)
    names = resolve_inference_sample_names(config)
    files = load_inference_selection_files(config, names)
    if "elbadry2026" in files:
        elbadry2026_inference_branch_ok(files["elbadry2026"])
    defaults = dict(config.inference.multi_sample.default_catalog_sf)
    if catalog_sf_overrides:
        defaults.update({k: float(v) for k, v in catalog_sf_overrides.items()})
    return SampleInclusionConfig(
        formulation=config.inference.multi_sample.formulation,
        sample_names=names,
        default_catalog_sf=defaults,
        spurious_model=spurious_model,
        selection_files=files,
    )


def estimate_sample_selection_sf(
    context: SampleInclusionConfig,
    *,
    mocks: Sequence[MockRealization] | None = None,
) -> float:
    """Scalar ``sample_selection_function`` multiplier for the Poisson rate."""
    if context.formulation == "separate_poisson":
        # Allowed only as a diagnostic contrast; production config must use unified.
        pass
    specs = inclusion_specs_from_files(context.selection_files)
    return mean_sample_selection_multiplier(
        formulation=context.formulation,
        catalog_sfs=context.default_catalog_sf,
        sample_names=context.sample_names,
        inclusion_specs=specs,
        mocks=list(mocks or ()),
        spurious_model=context.spurious_model,
    )


def format_q1_justification_report(
    *,
    formulation: OverlapFormulation,
    overlap: SampleOverlapMatrix | None = None,
    unified_sf: float | None = None,
    separate_sf: float | None = None,
) -> str:
    """Full-detail diagnostic report (caveman exemption)."""
    lines = [
        "=== sample_inclusion Q1 resolution (CONTINUATION_PLAN §15) ===",
        f"formulation: {formulation}",
        "",
        "Decision: unified_inclusion_indicator",
        "Rejected: separate_poisson for joint multi-sample inference",
        "",
        "Overlap structure (three-way, not pairwise):",
        "  - El-Badry 2024 drew from Andrews candidates",
        "  - El-Badry 2026 astrometric subsample 3 = Andrews ∩ (G < 15)",
        "  - Naive Σ rate_s double-counts systems in intersections",
        "",
        "Unified kernel: p_any = 1 - Π_s (1 - p_s); each source_id once in data term.",
        "v1 remains staged-but-connected (no fully joint upgrade).",
        "Inference samples: andrews2022_modified, elbadry2024, elbadry2026(astrometric).",
        "accel_jerk: disabled (§9).",
    ]
    if overlap is not None:
        lines.append("")
        lines.append("sample_overlap_matrix:")
        for name, n in overlap.per_sample_counts.items():
            lines.append(f"  {name}: N={n}")
        for (a, b), n in sorted(overlap.pairwise_counts.items()):
            lines.append(f"  {a} ∩ {b}: {n}")
        lines.append(f"  three-way ∩: {overlap.three_way_count}")
    if unified_sf is not None:
        lines.append(f"unified E[p_any]: {unified_sf:.6g}")
    if separate_sf is not None:
        lines.append(
            f"naive Σ p_s (double-counting diagnostic): {separate_sf:.6g}"
        )
    return "\n".join(lines) + "\n"


def refuse_validation_target_spurious_read(payload: Mapping[str, Any]) -> None:
    """Raise if a caller tries to feed validation_targets spurious rates in."""
    vt = payload.get("validation_targets")
    if not isinstance(vt, Mapping):
        return
    if "spurious_fraction" in vt or "spurious_rate" in vt:
        raise PerSampleSpuriousRateReadError(
            "validation_targets spurious rates are outputs only (§4.8); "
            "use shared P(spurious | ·) for inclusion"
        )


# Re-export helpers used by inference wiring / tests.
__all__ = [
    "DEFAULT_INFERENCE_SAMPLE_NAMES",
    "MockRealization",
    "Q1_JUSTIFICATION",
    "SampleInclusionConfig",
    "SampleOverlapMatrix",
    "build_sample_inclusion_context",
    "build_sample_overlap_matrix",
    "combine_sample_probabilities",
    "estimate_sample_selection_sf",
    "format_q1_justification_report",
    "intercept_only_spurious_model",
    "load_inference_selection_files",
    "mean_sample_selection_multiplier",
    "outcome_dependent_inclusion_probability",
    "refuse_validation_target_spurious_read",
    "resolve_inference_sample_names",
    "sample_inclusion_probability",
    "separate_poisson_sum_probability",
    "unified_any_sample_probability",
]
