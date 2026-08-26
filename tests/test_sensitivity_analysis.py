"""Tests for sensitivity_analysis stage (issue #38)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import SensitivityAnalysisConfig
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.sensitivity_analysis import (
    evaluate_mc_noise_at_n,
    format_sensitivity_report,
    generate_fiducial_catalog,
    minimum_n_mock_for_threshold,
    read_sensitivity_analysis_artifact,
    recommend_class_covariates,
    recommend_dimensionality,
    run_mc_noise_convergence,
    run_sensitivity_analysis,
    run_sensitivity_analysis_stage,
    sigma_mc_poisson_ratio,
    write_sensitivity_analysis_artifact,
)

pytestmark = pytest.mark.unit


def test_config_loads_sensitivity_analysis_fragment() -> None:
    cfg = load_config()
    sa = cfg.sensitivity_analysis
    assert sa.n_mass_bins == 8
    assert len(sa.fiducial_expected_counts) == 8
    assert sa.joint_dimensions[0] == "mass_msun"
    assert cfg.physics.mc_noise_threshold == pytest.approx(0.1)
    assert "ruwe" in sa.candidate_covariates
    assert set(sa.population_classes) == {"BH", "NS", "WD", "other", "outlier"}


def test_registry_fingerprints_sensitivity_config() -> None:
    spec = STAGE_REGISTRY["sensitivity_analysis"]
    assert spec.module.endswith("sensitivity_analysis")
    assert "physics.mc_noise_threshold" in spec.config_fingerprint_keys
    assert "sensitivity_analysis" in spec.config_fingerprint_keys
    assert spec.inputs_from == ("population_model",)


def test_sigma_mc_poisson_ratio_identity() -> None:
    sigma_mc, sigma_poisson, ratio = sigma_mc_poisson_ratio(25.0, n_mock=100)
    assert sigma_poisson == pytest.approx(5.0)
    assert sigma_mc == pytest.approx(0.5)
    assert ratio == pytest.approx(0.1)
    assert ratio == pytest.approx(1.0 / np.sqrt(100.0))


def test_minimum_n_mock_for_default_threshold() -> None:
    # 1/sqrt(n) < 0.1 ⇒ n > 100 ⇒ n_min = 101
    assert minimum_n_mock_for_threshold(0.1) == 101


@pytest.mark.physics
def test_mc_noise_convergence_meets_threshold() -> None:
    cfg = load_config()
    expected = cfg.sensitivity_analysis.fiducial_expected_counts
    threshold = cfg.physics.mc_noise_threshold
    diag = run_mc_noise_convergence(
        expected,
        threshold=threshold,
        n_mock_start=cfg.sensitivity_analysis.n_mock_start,
        n_mock_max=cfg.sensitivity_analysis.n_mock_max,
        growth_factor=cfg.sensitivity_analysis.n_mock_growth_factor,
    )
    assert diag.all_bins_passed
    assert diag.n_mock_final >= minimum_n_mock_for_threshold(threshold)
    assert len(diag.schedule_n_mock) >= 2
    assert len(diag.schedule_n_mock) == len(diag.schedule_max_ratio)
    # Schedule must be monotonically non-increasing in max_ratio.
    assert all(
        diag.schedule_max_ratio[i] >= diag.schedule_max_ratio[i + 1] - 1e-12
        for i in range(len(diag.schedule_max_ratio) - 1)
    )
    assert all(b.passed for b in diag.per_bin)
    assert "MC noise budget met" in diag.message


def test_mc_noise_convergence_fails_when_max_too_small() -> None:
    diag = run_mc_noise_convergence(
        [10.0, 20.0],
        threshold=0.1,
        n_mock_start=10,
        n_mock_max=50,
        growth_factor=2.0,
    )
    assert not diag.all_bins_passed
    assert "NOT met" in diag.message
    bins = evaluate_mc_noise_at_n([10.0, 20.0], n_mock=50, threshold=0.1)
    assert all(not b.passed for b in bins)


def test_dimensionality_prefers_1d_without_nd_signal() -> None:
    cfg = load_config().sensitivity_analysis
    catalog, _ = generate_fiducial_catalog(
        cfg, rng=np.random.default_rng(0), nd_signal=False
    )
    rec = recommend_dimensionality(catalog, cfg)
    assert rec.preferred_model == "1d_dndm"
    assert rec.dimension_names[0] == "mass_msun"
    assert rec.preferred_likelihood in {"unbinned", "binned"}


def test_dimensionality_prefers_joint_nd_with_period_signal() -> None:
    cfg = load_config().sensitivity_analysis.model_copy(
        update={
            "bic_delta_prefer_joint_nd": 1.0,
            "n_synthetic_systems": 1200,
            "mean_count_per_bin_unbinned_preference": 100.0,
        }
    )
    catalog, _ = generate_fiducial_catalog(
        cfg, rng=np.random.default_rng(7), nd_signal=True
    )
    rec = recommend_dimensionality(catalog, cfg)
    assert rec.delta_bic > 0
    assert rec.preferred_model == "joint_nd"
    assert "joint N-D" in rec.rationale or "joint" in rec.rationale.lower()


def test_class_covariates_select_injected_ruwe_for_outlier() -> None:
    cfg = load_config().sensitivity_analysis.model_copy(
        update={
            "bic_delta_include_covariate": 2.0,
            "n_synthetic_systems": 600,
        }
    )
    catalog, labels = generate_fiducial_catalog(
        cfg,
        rng=np.random.default_rng(11),
        covariate_signals={"outlier": ["ruwe"]},
    )
    recs = recommend_class_covariates(catalog, labels, cfg)
    by_class = {r.population_class: r for r in recs}
    assert "outlier" in by_class
    outlier = by_class["outlier"]
    assert outlier.mass_always_included
    assert "ruwe" in outlier.tested_covariates
    assert outlier.delta_bic_by_covariate["ruwe"] > 0
    assert "ruwe" in outlier.selected_covariates


def test_hdf5_round_trip_and_consumer_payload(tmp_path: Path) -> None:
    cfg = load_config()
    result = run_sensitivity_analysis(cfg, require_mc_pass=True)
    path = tmp_path / "sensitivity.h5"
    write_sensitivity_analysis_artifact(path, result)
    assert path.is_file()

    payload = read_sensitivity_analysis_artifact(path)
    assert payload["schema_version"] == 1
    assert payload["stage"] == "sensitivity_analysis"
    assert "population_model and inference must opt in" in payload["notes"]
    assert payload["dimensionality"]["preferred_model"] in {"1d_dndm", "joint_nd"}
    assert len(payload["class_covariates"]) == len(
        cfg.sensitivity_analysis.population_classes
    )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["stage"] == "sensitivity_analysis"
        assert handle.attrs["mc_noise_passed"]
        assert "schedule_n_mock" in handle["mc_noise_convergence"]
        assert "WD" in handle["class_covariates"]


def test_report_is_fully_legible() -> None:
    result = run_sensitivity_analysis(load_config(), require_mc_pass=True)
    report = format_sensitivity_report(result)
    assert "sensitivity_analysis report" in report
    assert "mc_noise_threshold" in report
    assert "defaults are not rewritten" in report


def test_stage_runner_writes_artifact_and_updates_manifest(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    # Keep checksum consistent with mutated paths by rebuilding via dump/load pattern.
    from darkhunter_pop.config_schema import PipelineConfig

    raw = cfg.model_dump(mode="json")
    cfg = PipelineConfig.model_validate(raw)

    run_path = tmp_path / "run.yaml"
    manifest = create_run_manifest(cfg)
    # create_run_manifest uses live checksum — fine.
    save_run_manifest(manifest, run_path)

    updated = run_sensitivity_analysis_stage(
        manifest, cfg, run_path=run_path, force_rerun=True
    )
    rec = updated.stages["sensitivity_analysis"]
    assert rec.status.value == "completed"
    assert rec.artifact_path is not None
    artifact = Path(rec.artifact_path)
    assert artifact.is_file()
    payload = read_sensitivity_analysis_artifact(artifact)
    assert payload["mc_noise_convergence"]["all_bins_passed"] is True

    # Cached skip on second call.
    again = run_sensitivity_analysis_stage(updated, cfg, run_path=run_path)
    assert again.stages["sensitivity_analysis"].artifact_path == rec.artifact_path


def test_sensitivity_config_validation() -> None:
    with pytest.raises(Exception):
        SensitivityAnalysisConfig(
            n_mass_bins=3,
            fiducial_expected_counts=[1.0, 2.0],
        )
    with pytest.raises(Exception):
        SensitivityAnalysisConfig(joint_dimensions=["period_day", "mass_msun"])


def test_checksum_includes_sensitivity_section() -> None:
    cfg = load_config()
    base = config_checksum(cfg)
    tweaked = cfg.model_copy(deep=True)
    tweaked.sensitivity_analysis.bic_delta_include_covariate = 99.0
    assert config_checksum(tweaked) != base


def test_stage_artifact_path_changes_with_sensitivity_config(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "out")
    from darkhunter_pop.config_schema import PipelineConfig

    cfg = PipelineConfig.model_validate(cfg.model_dump(mode="json"))
    spec = STAGE_REGISTRY["sensitivity_analysis"]
    p1 = stage_artifact_path(cfg, spec, run_id="runA")
    tweaked = cfg.model_copy(deep=True)
    tweaked.sensitivity_analysis.n_synthetic_systems = 401
    tweaked = PipelineConfig.model_validate(tweaked.model_dump(mode="json"))
    p2 = stage_artifact_path(tweaked, spec, run_id="runA")
    assert p1 != p2
