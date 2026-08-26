"""Tests for population_model stage (issue #57)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS, PipelineConfig
from darkhunter_pop.population_model import (
    build_mass_bin_edges,
    companion_nature_weight_schema,
    evaluate_mass_function,
    format_population_model_report,
    normalize_companion_nature_weights,
    ns_soft_truncation_marginalized,
    ns_soft_truncation_weight,
    read_population_model_artifact,
    run_population_model,
    run_population_model_stage,
    validate_companion_nature_weights,
    wd_hard_truncation_weight,
    write_population_model_artifact,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    CandidateRecord,
    ParameterSet,
)

pytestmark = pytest.mark.unit


def _uniform_weights(**overrides: float) -> dict[str, float]:
    base = {k: 0.0 for k in COMPANION_NATURE_WEIGHT_KEYS}
    base.update(overrides)
    if sum(base.values()) == 0:
        return {k: 1.0 for k in COMPANION_NATURE_WEIGHT_KEYS}
    return base


def test_config_loads_population_model_fragment() -> None:
    cfg = load_config()
    pm = cfg.population_model
    assert pm.p_single == pytest.approx(0.0)
    assert pm.p_binary == pytest.approx(1.0)
    assert pm.p_triple == pytest.approx(0.0)
    assert pm.mass_function_model == "free_height_bins"
    assert pm.n_mass_bins == 8
    assert len(pm.fiducial_expected_counts) == 8
    assert pm.allow_external_co_mf_priors is False
    assert set(pm.population_classes) == set(COMPANION_NATURE_WEIGHT_KEYS)


def test_checksum_includes_population_model() -> None:
    assert "population_model" in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    altered.population_model.n_mass_bins = cfg.population_model.n_mass_bins + 1
    altered.population_model.fiducial_expected_counts = [
        1.0
    ] * altered.population_model.n_mass_bins
    assert config_checksum(altered) != base


def test_registry_fingerprints_population_model() -> None:
    spec = STAGE_REGISTRY["population_model"]
    assert spec.module.endswith("population_model")
    assert "population_model" in spec.config_fingerprint_keys
    assert "classification" in spec.config_fingerprint_keys
    assert "mass_calibration.delta_M_Ch_msun" in spec.config_fingerprint_keys
    assert spec.inputs_from == (
        "companion_nature_likelihood",
        "selection_function_astrometric",
        "selection_function_followup",
    )


def test_weight_schema_contract() -> None:
    schema = companion_nature_weight_schema()
    assert schema["keys"] == list(COMPANION_NATURE_WEIGHT_KEYS)
    assert schema["pre_filter"] is False
    validate_companion_nature_weights(_uniform_weights(WD=1.0, NS=1.0))
    with pytest.raises(ValueError, match="exactly"):
        validate_companion_nature_weights({"BH": 1.0})
    norm = normalize_companion_nature_weights(_uniform_weights(WD=3.0, NS=1.0))
    assert sum(norm.values()) == pytest.approx(1.0)
    assert norm["WD"] == pytest.approx(0.75)
    zero_raw = {k: 0.0 for k in COMPANION_NATURE_WEIGHT_KEYS}
    zero = normalize_companion_nature_weights(zero_raw)
    assert all(v == pytest.approx(0.2) for v in zero.values())


def test_v1_multiplicity_and_bin_edges() -> None:
    cfg = load_config()
    edges = build_mass_bin_edges(cfg.population_model)
    assert edges.size == cfg.population_model.n_mass_bins + 1
    assert edges[0] == pytest.approx(cfg.population_model.mass_min_msun)
    assert edges[-1] == pytest.approx(cfg.population_model.mass_max_msun)
    assert np.all(np.diff(edges) > 0)
    bad = cfg.model_copy(deep=True)
    bad.population_model.p_single = 0.1
    bad.population_model.p_binary = 0.9
    with pytest.raises(ValueError, match="p_single=0"):
        run_population_model(bad)


def test_wd_hard_and_ns_soft_truncation() -> None:
    m = np.array([1.0, 1.4, 1.5, 2.5, 10.0])
    wd = wd_hard_truncation_weight(m, m_ch_msun=1.4)
    assert wd.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    soft = ns_soft_truncation_weight(m, m_tov_msun=2.2, width_msun=0.05)
    assert soft[0] > 0.9
    assert soft[-1] < 0.1
    marg = ns_soft_truncation_marginalized(
        m,
        m_tov_mean_msun=2.2,
        m_tov_sigma_msun=0.2,
        width_msun=0.05,
        n_quad=21,
    )
    assert marg.shape == m.shape
    assert marg[0] > marg[-1]
    result = run_population_model(load_config(), mass_grid=m)
    assert np.all(result.two_tier.classified_dndm["outlier"] >= 0)
    assert result.two_tier.classified_dndm["WD"][2] == pytest.approx(0.0)
    assert result.two_tier.raw_total_co_dndm[-1] > 0


@pytest.mark.physics
def test_two_tier_and_gp_swap() -> None:
    cfg = load_config()
    free = run_population_model(cfg)
    assert free.multiplicity.p_single == 0.0
    assert free.multiplicity.p_triple == 0.0
    assert set(free.two_tier.classified_dndm) == set(
        cfg.population_model.population_classes
    )
    assert free.two_tier.raw_total_co_dndm.shape == free.two_tier.mass_grid_msun.shape

    gp_cfg = cfg.model_copy(deep=True)
    gp_cfg.population_model.mass_function_model = "gp_log_dndm"
    gp = run_population_model(gp_cfg)
    assert gp.mass_function_model == "gp_log_dndm"
    assert np.all(gp.two_tier.raw_total_co_dndm > 0)

    edges = free.bin_edges_msun
    heights = free.bin_heights
    mid = np.sqrt(edges[:-1] * edges[1:])
    dens = evaluate_mass_function(
        mid,
        model="free_height_bins",
        bin_edges=edges,
        heights=heights,
        cfg=cfg.population_model,
    )
    assert dens.shape == mid.shape
    assert np.all(dens > 0)


def test_consumes_sensitivity_covariates_and_weights() -> None:
    cfg = load_config()
    payload = {
        "class_covariates": [
            {
                "population_class": "outlier",
                "selected_covariates": ["ruwe"],
                "tested_covariates": ["ruwe", "eccentricity"],
                "delta_bic_by_covariate": {"ruwe": 12.0},
                "mass_always_included": True,
            },
            {
                "population_class": "WD",
                "selected_covariates": [],
                "tested_covariates": ["ruwe"],
                "delta_bic_by_covariate": {"ruwe": 1.0},
                "mass_always_included": True,
            },
        ]
    }
    cand = CandidateRecord(
        source_id=42,
        m2=ParameterSet(
            names=["M2"],
            values=[1.2],
            covariance=[[0.01]],
            provenance="test",
        ),
        companion_nature_weights=_uniform_weights(WD=4.0, NS=1.0),
    )
    result = run_population_model(
        cfg,
        candidates=[cand],
        sensitivity_payload=payload,
    )
    assert result.sensitivity_artifact_used is True
    assert result.covariates_applied["outlier"] == ("ruwe",)
    assert result.covariates_applied["WD"] == ()
    assert len(result.system_weights) == 1
    assert result.system_weights[0].responsibilities["WD"] == pytest.approx(0.8)

    opted = cfg.model_copy(deep=True)
    opted.population_model.apply_sensitivity_covariates = False
    result2 = run_population_model(
        opted, candidates=[cand], sensitivity_payload=payload
    )
    assert result2.sensitivity_artifact_used is False
    assert result2.covariates_applied["outlier"] == ()


def test_hdf5_round_trip_and_stage(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "artifacts")
    cand = CandidateRecord(
        source_id=7,
        companion_nature_weights=_uniform_weights(BH=1.0),
    )
    result = run_population_model(cfg, candidates=[cand])
    path = tmp_path / "pop.h5"
    write_population_model_artifact(path, result)
    payload = read_population_model_artifact(path)
    assert payload["stage"] == "population_model"
    assert payload["external_co_mf_priors"] is False
    assert payload["weight_schema"]["keys"] == list(COMPANION_NATURE_WEIGHT_KEYS)
    assert "raw_total_co_dndm" in payload["two_tier_dndm"]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["stage"] == "population_model"
        assert handle.attrs["p_triple"] == 0.0
        assert "responsibility_WD" in handle["system_weights"]

    report = format_population_model_report(result)
    assert "population_model report" in report
    assert "P(triple)=0.0" in report

    run_path = tmp_path / "run.yaml"
    manifest = create_run_manifest(cfg)
    save_run_manifest(manifest, run_path)
    updated = run_population_model_stage(
        manifest, cfg, run_path=run_path, candidates=[cand]
    )
    rec = updated.stages["population_model"]
    assert rec.status.value == "completed"
    artifact = stage_artifact_path(
        cfg, STAGE_REGISTRY["population_model"], run_id=manifest.run_id
    )
    assert Path(rec.artifact_path).resolve() == artifact.resolve()
    again = run_population_model_stage(updated, cfg, run_path=run_path)
    assert again.stages["population_model"].artifact_path == rec.artifact_path


def test_forbid_external_co_mf_priors_in_schema() -> None:
    cfg = load_config().model_dump(mode="json")
    cfg["population_model"]["allow_external_co_mf_priors"] = True
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(cfg)
