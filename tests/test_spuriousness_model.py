"""Tests for the shared spuriousness model (issue #111 / roster #27)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS, SpuriousnessModelFile
from darkhunter_pop.sensitivity_analysis import recommend_binary_outcome_covariates
from darkhunter_pop.spuriousness_model import (
    acceptance_targets_passed,
    assert_no_per_sample_spurious_rate_input,
    build_candidate_design,
    fit_spuriousness_model,
    format_censoring_report,
    format_covariate_sensitivity_report,
    format_labeled_set_performance_report,
    format_rate_reproduction_report,
    harmonic_distance,
    load_labeled_sources,
    load_spuriousness_model_file,
    predict_p_spurious,
    write_spuriousness_reports,
)

pytestmark = pytest.mark.unit

GAIA_BH1 = 4373465352415301632


def test_config_loads_spuriousness_outside_selections() -> None:
    cfg = load_config()
    assert cfg.spuriousness_model.enabled is True
    assert cfg.spuriousness_model.path == "config/spuriousness_model.yaml"
    assert not cfg.spuriousness_model.path.startswith("config/selections/")
    assert "spuriousness_model" in SHARED_CHECKSUM_SECTIONS
    assert_no_per_sample_spurious_rate_input(cfg)


def test_checksum_includes_spuriousness_model() -> None:
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    altered.spuriousness_model.enabled = False
    assert config_checksum(altered) != base


def test_spuriousness_model_file_validates() -> None:
    spec = load_spuriousness_model_file()
    assert isinstance(spec, SpuriousnessModelFile)
    assert spec.censoring == "joint"
    assert spec.link_function == "probit"
    assert len(spec.labeled_sets) == 4
    assert "f2_x_g_break" in spec.candidate_covariates
    assert spec.validation.elbadry2026_sb1.role == "advisory"


def test_labeled_set_counts_match_fixtures() -> None:
    spec = load_spuriousness_model_file()
    sources = load_labeled_sources(spec)
    assert len(sources) == 293
    n_spur = sum(1 for s in sources if s.verdict == "spurious")
    n_gen = sum(1 for s in sources if s.verdict == "genuine")
    n_und = sum(1 for s in sources if s.verdict == "undetermined")
    assert n_spur == 63
    assert n_gen == 197
    assert n_und == 33
    # Two label axes: ultramassive WD is genuine, not spurious.
    wd = [s for s in sources if s.nature == "massive_white_dwarf"]
    assert wd
    assert all(s.verdict == "genuine" for s in wd)
    # Gaia BH1 genuine.
    bh1 = [s for s in sources if s.source_id == GAIA_BH1]
    assert bh1
    assert all(s.verdict == "genuine" for s in bh1)


def test_harmonic_distance_gaia_bh1_near_third_harmonic() -> None:
    # P = 185.8 d ≈ 3 × 62 d.
    d = harmonic_distance(185.8, fundamental_period_days=62.0, multiples_max=20)
    assert d < 0.01


def test_sensitivity_module_drives_covariate_retention() -> None:
    spec = load_spuriousness_model_file()
    sources = load_labeled_sources(spec)
    design = build_candidate_design(sources, spec)
    result = fit_spuriousness_model(spec)
    sens = result.model.sensitivity
    # Every retained covariate must appear in the sensitivity selected set.
    for name in result.model.retained_covariates:
        assert name in sens.selected_covariates
    # Dropped covariates carry an explicit reason.
    for name in sens.dropped_covariates:
        assert name in sens.drop_reasons
    # Absent fixture columns are dropped, not hand-kept.
    assert "parallax_snr" in sens.dropped_covariates
    assert "visibility_periods_used" in sens.dropped_covariates
    report = format_covariate_sensitivity_report(result)
    assert "literature_interaction" in report
    assert "f2_x_g_break" in report


def test_recommend_binary_outcome_covariates_selects_signal() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x_signal = rng.normal(size=n)
    x_noise = rng.normal(size=n)
    logits = -0.5 + 1.5 * x_signal
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(np.float64)
    rec = recommend_binary_outcome_covariates(
        {"signal": x_signal, "noise": x_noise},
        y,
        ["signal", "noise"],
        bic_delta_include=6.0,
        min_complete_rows=20,
    )
    assert "signal" in rec.selected_covariates
    assert "noise" in rec.dropped_covariates


@pytest.mark.physics
def test_acceptance_rate_reproduction_one_parameter_set() -> None:
    result = fit_spuriousness_model()
    assert acceptance_targets_passed(result)
    by_name = {r.sample: r for r in result.rate_reproduction}
    assert by_name["elbadry2024"].passed is True
    assert by_name["elbadry2026_astrometric"].passed is True
    assert by_name["elbadry2026_sb1"].role == "advisory"
    assert by_name["elbadry2026_sb1"].passed is None
    # Fixture recoverability of the published denominators.
    assert by_name["elbadry2024"].n == 48
    assert by_name["elbadry2026_astrometric"].n == 76
    report = format_rate_reproduction_report(result)
    assert "ESCALATE" not in report
    assert result.sb1_q15_report["status"] == "escalated_pending_human_signoff"
    assert result.sb1_q15_report["spurious_solution_fraction_all_151"] == pytest.approx(
        24 / 151
    )


def test_predict_p_spurious_evaluable_on_mocks() -> None:
    result = fit_spuriousness_model()
    model = result.model
    # Mock-like dict with only physical/observational fields — no sample rate.
    n = 5
    cov = {
        "goodness_of_fit": np.linspace(0.0, 10.0, n),
        "a0_snr": np.linspace(5.0, 40.0, n),
        "log_implied_companion_mass": np.linspace(0.0, 1.5, n),
        "f2_x_g_break": np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        "phot_g_mean_mag": np.linspace(10.0, 15.0, n),
    }
    p = predict_p_spurious(model, cov)
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # Higher F2×G / lower S/N should raise spuriousness on average.
    assert float(p[-1]) >= float(p[0]) - 1e-9


def test_undetermined_not_dropped_censoring_report() -> None:
    result = fit_spuriousness_model()
    assert result.model.n_undetermined == 33
    report = format_censoring_report(result)
    assert "n_undetermined=33" in report
    assert "rate_shift" in report
    # Every undetermined row appears.
    for s in result.design.sources:
        if s.verdict == "undetermined":
            assert str(s.source_id) in report


def test_gaia_bh1_scored_explicitly() -> None:
    result = fit_spuriousness_model()
    report = format_labeled_set_performance_report(result)
    assert f"gaia_bh1 source_id={GAIA_BH1}" in report
    assert "verdict=genuine" in report
    assert "andrews_exclusion=wrong_per_Q14" in report
    bh1_rows = [
        s for s in result.design.sources if s.source_id == GAIA_BH1
    ]
    assert bh1_rows
    assert all(s.verdict == "genuine" for s in bh1_rows)
    # Extreme spurious E1 masses (122 / 119 Msun) must outrank BH1 on P(spurious).
    e1_bh1 = next(
        i
        for i, s in enumerate(result.design.sources)
        if s.source_id == GAIA_BH1 and s.table == "elbadry2023_table_e1"
    )
    extreme_spur = [
        i
        for i, s in enumerate(result.design.sources)
        if s.table == "elbadry2023_table_e1"
        and s.verdict == "spurious"
        and s.implied_companion_mass_msun is not None
        and s.implied_companion_mass_msun > 50.0
    ]
    assert extreme_spur
    assert float(result.p_spurious[e1_bh1]) < float(
        np.min(result.p_spurious[extreme_spur])
    )


def test_no_per_sample_rate_in_model_inputs() -> None:
    spec = load_spuriousness_model_file()
    dumped = spec.model_dump(mode="json")
    # validation block exists as outputs; fitting must not require reading
    # per-sample selection validation_targets.
    assert "validation" in dumped
    result = fit_spuriousness_model(spec)
    # Sanity: config snapshot keeps validation as acceptance metadata only.
    assert result.config_snapshot["validation"]["elbadry2024"][
        "target_spurious_fraction"
    ] == pytest.approx(0.25)


def test_write_reports(tmp_path: Path) -> None:
    result = fit_spuriousness_model()
    paths = write_spuriousness_reports(result, tmp_path)
    assert set(paths) == {
        "spuriousness_rate_reproduction",
        "spuriousness_covariate_sensitivity",
        "spuriousness_labeled_set_performance",
        "spuriousness_censoring_report",
    }
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0
