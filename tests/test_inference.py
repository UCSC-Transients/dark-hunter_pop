"""Tests for inference stage (issue #63)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.inference import (
    ROBUSTNESS_PROTOCOL,
    ObservedEvent,
    binned_log_likelihood,
    events_from_population_result,
    format_inference_report,
    mass_function_intensity,
    posterior_vs_prior_overlap,
    read_astrometric_sf_scalar,
    read_followup_sf_scalar,
    read_inference_artifact,
    resolve_likelihood_form,
    run_inference,
    run_inference_stage,
    unbinned_log_likelihood,
    write_inference_artifact,
    zero_count_poisson_upper_limits,
)
from darkhunter_pop.physics_utils import poisson_upper_limit_zero_events
from darkhunter_pop.population_model import run_population_model
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


def _smoke_config(tmp_path: Path | None = None):
    cfg = load_config().model_copy(deep=True)
    cfg.inference.skip_sampler = True
    cfg.inference.nlive = 5
    cfg.inference.maxcall = 80
    cfg.inference.dlogz = 1.0
    cfg.inference.n_mass_grid = 16
    cfg.population_model.n_mass_bins = 4
    cfg.population_model.fiducial_expected_counts = [5.0, 8.0, 4.0, 2.0]
    if tmp_path is not None:
        cfg.paths.artifact_root = str(tmp_path / "artifacts")
    return cfg


def test_config_loads_inference_fragment() -> None:
    cfg = load_config()
    inf = cfg.inference
    assert inf.apply_sensitivity_dimensionality is True
    assert inf.likelihood_form == "auto"
    assert inf.eccentricity_hypothesis == "thermal"
    assert inf.circular_implies_wd is False
    assert inf.nlive == 20
    assert inf.n_robustness_runs == 1
    assert inf.zero_count_ul_confidence == pytest.approx(0.95)
    assert "seed" in ROBUSTNESS_PROTOCOL.lower()


def test_inference_not_yet_in_shared_checksum() -> None:
    """Review #64 may add ``inference`` to SHARED_CHECKSUM_SECTIONS + freeze §3.

    First PR fingerprints via registry only — ask before freeze/checksum edits.
    """
    assert "inference" not in SHARED_CHECKSUM_SECTIONS
    assert "diagnostics" not in SHARED_CHECKSUM_SECTIONS


def test_registry_fingerprints_inference() -> None:
    spec = STAGE_REGISTRY["inference"]
    assert spec.module.endswith("inference")
    assert "inference" in spec.config_fingerprint_keys
    assert "population_model" in spec.config_fingerprint_keys
    assert "darkhunter_pop.physics_utils" in spec.dependency_modules
    assert spec.inputs_from == (
        "population_model",
        "selection_function_astrometric",
        "selection_function_followup",
        "sensitivity_analysis",
    )


def test_resolve_likelihood_form_honors_sa_opt_in() -> None:
    cfg = load_config().inference
    assert resolve_likelihood_form(cfg, None) == "unbinned"
    sa = {"dimensionality": {"preferred_likelihood": "binned"}}
    assert resolve_likelihood_form(cfg, sa) == "binned"
    opted_out = cfg.model_copy(update={"apply_sensitivity_dimensionality": False})
    assert resolve_likelihood_form(opted_out, sa) == "unbinned"
    forced = cfg.model_copy(update={"likelihood_form": "unbinned"})
    assert resolve_likelihood_form(forced, sa) == "unbinned"


@pytest.mark.physics
def test_poisson_sf_likelihood_assembly() -> None:
    cfg = _smoke_config()
    pop = run_population_model(cfg)
    edges = pop.bin_edges_msun
    heights = pop.bin_heights
    grid = np.geomspace(edges[0], edges[-1], 32)
    lam = mass_function_intensity(
        grid,
        heights,
        bin_edges=edges,
        population_cfg=cfg.population_model,
        astrometric_sf=0.5,
        followup_sf=0.8,
    )
    assert lam.shape == grid.shape
    assert np.all(lam >= 0)
    lam2 = mass_function_intensity(
        grid,
        heights,
        bin_edges=edges,
        population_cfg=cfg.population_model,
        astrometric_sf=0.25,
        followup_sf=0.8,
    )
    assert np.allclose(lam2, 0.5 * lam)

    masses = np.array([1.0, 2.0])
    weights = np.array([1.0, 1.0])
    ll_u = unbinned_log_likelihood(
        heights,
        event_masses=masses,
        event_weights=weights,
        mass_grid=grid,
        bin_edges=edges,
        population_cfg=cfg.population_model,
        astrometric_sf=0.5,
        followup_sf=0.8,
    )
    assert np.isfinite(ll_u)
    ll_b = binned_log_likelihood(
        heights,
        event_masses=masses,
        event_weights=weights,
        bin_edges=edges,
        population_cfg=cfg.population_model,
        astrometric_sf=0.5,
        followup_sf=0.8,
    )
    assert np.isfinite(ll_b)


def test_zero_count_upper_limits_and_overlap() -> None:
    cfg = load_config().inference
    counts = np.array([0.0, 2.0, 0.0, 1.0])
    uls = zero_count_poisson_upper_limits(
        counts, confidence=cfg.zero_count_ul_confidence
    )
    assert len(uls) == 2
    expected = poisson_upper_limit_zero_events(cfg.zero_count_ul_confidence)
    assert uls[0]["mu_upper"] == pytest.approx(expected)
    assert uls[0]["bin_index"] == 0
    assert uls[1]["bin_index"] == 2

    prior_low = np.array([0.0, 0.0])
    prior_high = np.array([12.0, 12.0])
    samples = np.random.default_rng(0).uniform(0.0, 12.0, size=(500, 2))
    ov = posterior_vs_prior_overlap(
        samples,
        prior_low=prior_low,
        prior_high=prior_high,
        threshold=0.5,
    )
    assert ov["mean_width_ratio"] > 0.5
    assert ov["prior_dominated"] is True


def test_model_comparison_switches_change_weights() -> None:
    cfg = _smoke_config()
    cand = CandidateRecord(
        source_id=11,
        m2=ParameterSet(
            names=["M2"],
            values=[1.1],
            covariance=[[0.01]],
            provenance="test",
        ),
        companion_nature_weights=_uniform_weights(WD=1.0, NS=1.0),
    )
    thermal = cfg.model_copy(deep=True)
    thermal.inference.eccentricity_hypothesis = "thermal"
    thermal.inference.circular_implies_wd = False
    sn = cfg.model_copy(deep=True)
    sn.inference.eccentricity_hypothesis = "sn_kick"
    sn.inference.circular_implies_wd = True
    sn.inference.circular_e_threshold = 0.1

    pop = run_population_model(cfg, candidates=[cand])
    ev_th = events_from_population_result(
        pop, cfg=thermal.inference, eccentricities={11: 0.02}
    )
    ev_sn = events_from_population_result(
        pop, cfg=sn.inference, eccentricities={11: 0.02}
    )
    assert len(ev_th) == 1 and len(ev_sn) == 1
    assert ev_th[0].weight != pytest.approx(ev_sn[0].weight)


def test_sf_scalar_readers(tmp_path: Path) -> None:
    astro = tmp_path / "astro.h5"
    with h5py.File(astro, "w") as handle:
        handle.attrs["detection_fraction"] = 0.37
    assert read_astrometric_sf_scalar(astro, default=1.0) == pytest.approx(0.37)
    assert read_astrometric_sf_scalar(None, default=0.9) == pytest.approx(0.9)

    follow = tmp_path / "follow.h5"
    with h5py.File(follow, "w") as handle:
        grp = handle.create_group("followup_catalog")
        grp.create_dataset("probability", data=np.array([0.2, 0.4, 0.6]))
    assert read_followup_sf_scalar(follow, default=1.0) == pytest.approx(0.4)


def test_hdf5_round_trip_and_stage(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    result = run_inference(cfg)
    assert result.likelihood_form == "unbinned"
    assert result.n_events == 0
    assert len(result.zero_count_upper_limits) == cfg.population_model.n_mass_bins
    assert result.logz is None
    report = format_inference_report(result)
    assert "staged-but-connected" in report

    path = tmp_path / "inf.h5"
    write_inference_artifact(path, result)
    payload = read_inference_artifact(path)
    assert payload["stage"] == "inference"
    assert payload["v1_staged_but_connected"] is True
    assert payload["v2_fully_joint"] is False

    run_path = tmp_path / "run.yaml"
    manifest = create_run_manifest(cfg)
    save_run_manifest(manifest, run_path)
    manifest = run_inference_stage(manifest, cfg, run_path=run_path)
    assert "inference" in manifest.stages
    assert manifest.stages["inference"].status.value in {"completed", "cached"}
    art = stage_artifact_path(
        cfg, STAGE_REGISTRY["inference"], run_id=manifest.run_id
    )
    assert art.is_file()

    manifest2 = run_inference_stage(manifest, cfg, run_path=run_path)
    assert manifest2.stages["inference"].status.value in {"completed", "cached"}


@pytest.mark.api
def test_dynesty_smoke_short() -> None:
    """Bounded dynesty smoke for CI (tiny nlive/maxcall). Cluster recipe ≠ CI."""
    pytest.importorskip("dynesty")
    cfg = load_config().model_copy(deep=True)
    cfg.inference.skip_sampler = False
    cfg.inference.nlive = 8
    cfg.inference.maxcall = 80
    cfg.inference.dlogz = 2.0
    cfg.inference.n_robustness_runs = 1
    cfg.inference.n_mass_grid = 12
    cfg.population_model.n_mass_bins = 3
    cfg.population_model.fiducial_expected_counts = [4.0, 6.0, 3.0]
    events = [
        ObservedEvent(source_id=1, mass_msun=1.0, weight=1.0),
        ObservedEvent(source_id=2, mass_msun=2.5, weight=0.8),
    ]
    result = run_inference(cfg, events=events)
    assert result.logz is not None
    assert np.isfinite(result.logz)
    assert len(result.posterior_median_heights) == 3
    assert "prior_dominated" in result.posterior_prior_overlap
    assert result.sampler_runs[0]["nlive"] == 8
