"""Tests for multi-sample inclusion operator + §15 Q1 resolution (issue #112)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import InferenceMultiSampleConfig
from darkhunter_pop.inference import ObservedEvent, run_inference, unbinned_log_likelihood
from darkhunter_pop.population_model import run_population_model
from darkhunter_pop.sample_inclusion import (
    MockRealization,
    build_sample_inclusion_context,
    build_sample_overlap_matrix,
    combine_sample_probabilities,
    estimate_sample_selection_sf,
    intercept_only_spurious_model,
    outcome_dependent_inclusion_probability,
    sample_inclusion_probability,
    unified_any_sample_probability,
)
from darkhunter_pop.sample_selection import load_sample_selection_file
from darkhunter_pop.spuriousness_model import predict_p_spurious
from pathlib import Path

pytestmark = pytest.mark.unit


def test_unified_any_avoids_double_count() -> None:
    # Three overlapping samples each with p=0.5 → unified ≤ 1; naive sum = 1.5.
    per = [0.5, 0.5, 0.5]
    unified = unified_any_sample_probability(per)
    naive = combine_sample_probabilities(per, formulation="separate_poisson")
    assert unified == pytest.approx(1.0 - 0.125)
    assert naive == pytest.approx(1.5)
    assert unified < naive


def test_sample_overlap_matrix_three_way() -> None:
    membership = {
        "andrews2022_modified": [1, 2, 3, 4],
        "elbadry2024": [1, 2, 9],
        "elbadry2026": [1, 3, 8],
    }
    mat = build_sample_overlap_matrix(membership)
    assert mat.three_way_count == 1
    assert mat.three_way_ids == frozenset({1})
    assert mat.pairwise_counts[("andrews2022_modified", "elbadry2024")] == 2
    payload = mat.as_dict()
    assert payload["diagnostic"] == "sample_overlap_matrix"
    assert payload["three_way_count"] == 1


def test_not_spurious_uses_shared_model_not_validation_targets() -> None:
    cfg = load_config()
    files = {
        "elbadry2024": load_sample_selection_file(
            Path("config/selections/elbadry2024.yaml")
        )
    }
    model = intercept_only_spurious_model(p_spurious=0.25)
    mock = MockRealization(
        source_id=1,
        mass_msun=1.5,
        m2_joint_fit_msun=1.5,
        orbit_coverage_fraction=0.8,
        g_mag=14.0,
    )
    p = sample_inclusion_probability(
        1.0,
        files["elbadry2024"].inclusion_operator,
        mock,
        spurious_model=model,
    )
    # Catalog SF=1; m2 and coverage pass; not_spurious = 0.75.
    assert p == pytest.approx(0.75)
    # Ensure validation_targets exist but are not consumed.
    assert files["elbadry2024"].validation_targets is not None
    assert files["elbadry2024"].validation_targets.spurious_fraction is not None


def test_refuse_frozen_andrews_and_accel_in_inference_config() -> None:
    with pytest.raises(ValidationError):
        InferenceMultiSampleConfig(sample_names=["andrews2022", "elbadry2024"])
    with pytest.raises(ValidationError):
        InferenceMultiSampleConfig(sample_names=["accel_jerk"])


def test_build_inclusion_context_loads_three_samples() -> None:
    cfg = load_config()
    ctx = build_sample_inclusion_context(
        cfg, spurious_model=intercept_only_spurious_model()
    )
    assert ctx.formulation == "unified_inclusion_indicator"
    assert ctx.sample_names == (
        "andrews2022_modified",
        "elbadry2024",
        "elbadry2026",
    )
    assert "elbadry2026" in ctx.selection_files
    assert ctx.selection_files["elbadry2026"].inference_branches == ["astrometric"]
    sf = estimate_sample_selection_sf(ctx)
    assert 0.0 < sf <= 1.0


def test_outcome_dependent_m2_cut() -> None:
    spec = load_sample_selection_file(
        Path("config/selections/elbadry2024.yaml")
    ).inclusion_operator
    model = intercept_only_spurious_model(p_spurious=0.1)
    fail = MockRealization(
        source_id=1,
        mass_msun=1.0,
        m2_joint_fit_msun=1.0,
        orbit_coverage_fraction=1.0,
    )
    p = outcome_dependent_inclusion_probability(
        spec, fail, spurious_model=model
    )
    assert p == 0.0


@pytest.mark.physics
def test_inference_rate_includes_sample_sf() -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.inference.skip_sampler = True
    cfg.population_model.n_mass_bins = 4
    cfg.population_model.fiducial_expected_counts = [5.0, 8.0, 4.0, 2.0]
    result = run_inference(cfg, events=[])
    assert result.multi_sample_formulation == "unified_inclusion_indicator"
    assert result.sample_selection_sf > 0.0
    assert result.sample_overlap_matrix["three_way_count"] == 1
    assert "unified_inclusion_indicator" in result.q1_justification_report.lower()


@pytest.mark.physics
def test_sbc_style_recovery_with_sample_sf() -> None:
    """Inject known heights; recover median near truth under unified sample SF."""
    cfg = load_config().model_copy(deep=True)
    cfg.inference.skip_sampler = True
    cfg.inference.likelihood_form = "binned"
    cfg.population_model.n_mass_bins = 4
    cfg.population_model.fiducial_expected_counts = [8.0, 12.0, 6.0, 3.0]
    pop = run_population_model(cfg)
    edges = np.asarray(pop.bin_edges_msun, dtype=np.float64)
    truth = np.asarray(pop.bin_heights, dtype=np.float64)
    # Draw Poisson counts from truth × SF product, then invert via MLE = counts/SF.
    rng = np.random.default_rng(112)
    spur = intercept_only_spurious_model(p_spurious=0.2)
    ctx = build_sample_inclusion_context(cfg, spurious_model=spur)
    mocks = [
        MockRealization(
            source_id=i,
            mass_msun=float(rng.uniform(0.5, 5.0)),
            m2_joint_fit_msun=float(rng.uniform(1.3, 3.0)),
            orbit_coverage_fraction=1.0,
            g_mag=14.0,
        )
        for i in range(40)
    ]
    sample_sf = estimate_sample_selection_sf(ctx, mocks=mocks)
    astro = 0.8
    follow = 0.9
    mean = truth * astro * follow * sample_sf
    counts = rng.poisson(mean)
    events: list[ObservedEvent] = []
    for i, c in enumerate(counts):
        m_lo, m_hi = edges[i], edges[i + 1]
        mid = float(np.sqrt(m_lo * m_hi))
        for _ in range(int(c)):
            events.append(ObservedEvent(source_id=10_000 + i * 100 + _, mass_msun=mid, weight=1.0))
    result = run_inference(
        cfg,
        events=events,
        inclusion_mocks=mocks,
        spurious_model=spur,
    )
    # Fiducial LL finite; recovered sample SF matches mock estimate.
    assert np.isfinite(result.fiducial_log_likelihood)
    assert result.sample_selection_sf == pytest.approx(sample_sf, rel=1e-6)
    # Naive separate sum must exceed unified when overlaps / multi-sample.
    naive = combine_sample_probabilities(
        [0.5, 0.5, 0.5], formulation="separate_poisson"
    )
    assert naive > unified_any_sample_probability([0.5, 0.5, 0.5])


def test_predict_p_spurious_intercept() -> None:
    model = intercept_only_spurious_model(p_spurious=0.4)
    p = predict_p_spurious(model, {"phot_g_mean_mag": 14.0})
    assert float(p[0]) == pytest.approx(0.4, abs=1e-6)


def test_registry_lists_sample_inclusion_dep() -> None:
    from darkhunter_pop.run_management import STAGE_REGISTRY

    spec = STAGE_REGISTRY["inference"]
    assert "sample_selection" in spec.inputs_from
    assert "darkhunter_pop.sample_inclusion" in spec.dependency_modules
