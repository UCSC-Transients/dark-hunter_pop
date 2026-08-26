"""Tests for simulation-based calibration recovery + coverage (issue #69)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import InjectedMassFunctionProfile, SBCConfig
from darkhunter_pop.diagnostics import (
    emit_sbc_recovery,
    list_diagnostic_helpers,
    resolve_diagnostic_dirs,
    run_diagnostics_scaffolding,
)
from darkhunter_pop.sbc import (
    assert_sbc_coverage,
    build_injected_truths,
    equal_tailed_credible_interval,
    format_sbc_report,
    inject_binned_counts,
    inject_unbinned_events,
    read_sbc_artifact,
    recover_heights_analytic,
    run_sbc_suite,
    scale_injected_heights,
    truth_in_interval,
    write_sbc_artifact,
)


def _sbc_config(**overrides: object) -> SBCConfig:
    base = load_config().diagnostics.sbc.model_copy(deep=True)
    return base.model_copy(update=overrides)


@pytest.mark.unit
def test_config_loads_sbc_fragment() -> None:
    cfg = load_config()
    sbc = cfg.diagnostics.sbc
    assert sbc.enabled is True
    assert sbc.run_in_stage is False
    assert sbc.recovery_backend == "analytic_binned"
    assert sbc.credible_interval_level == pytest.approx(0.68)
    assert sbc.coverage_abs_tolerance == pytest.approx(0.20)
    assert sbc.n_mass_bins == 4
    assert len(sbc.injected_profiles) == 4
    names = {p.name for p in sbc.injected_profiles}
    assert names == {"flat", "rising", "falling", "peaked"}
    assert cfg.diagnostics.hooks.sbc_recovery is True


@pytest.mark.unit
def test_sbc_helper_registered() -> None:
    names = list_diagnostic_helpers()
    assert "emit_sbc_recovery" in names
    assert "format_sbc_report" in names


@pytest.mark.unit
def test_scale_and_inject_truths() -> None:
    heights = scale_injected_heights([1.0, 3.0], expected_total_rate=20.0)
    assert heights.sum() == pytest.approx(20.0)
    assert heights[1] == pytest.approx(3.0 * heights[0])

    cfg = load_config().model_copy(deep=True)
    sbc = _sbc_config(n_mass_bins=4, expected_total_rate=40.0)
    truths = build_injected_truths(cfg, sbc=sbc)
    assert len(truths) == 4
    assert len({t.profile_name for t in truths}) == 4
    for t in truths:
        assert t.heights.sum() == pytest.approx(40.0)
        assert t.bin_edges_msun.size == 5


@pytest.mark.unit
@pytest.mark.physics
def test_analytic_binned_coverage_calibrated() -> None:
    """Gamma CI coverage matches the configured level within tolerance."""
    cfg = load_config().model_copy(deep=True)
    sbc = _sbc_config(
        n_mass_bins=3,
        n_repeats=40,
        expected_total_rate=60.0,
        coverage_abs_tolerance=0.12,
        credible_interval_level=0.68,
        n_posterior_samples=1500,
        injected_profiles=[
            InjectedMassFunctionProfile(
                name="flat", relative_heights=[1.0, 1.0, 1.0]
            ),
            InjectedMassFunctionProfile(
                name="rising", relative_heights=[0.5, 1.0, 2.0]
            ),
            InjectedMassFunctionProfile(
                name="falling", relative_heights=[2.0, 1.0, 0.5]
            ),
        ],
        recovery_backend="analytic_binned",
        random_seed=69,
    )
    result = run_sbc_suite(cfg, sbc=sbc)
    report = format_sbc_report(result)
    assert "=== simulation-based calibration (SBC) ===" in report
    assert "overall_empirical_coverage" in report
    assert result.overall_passed is True
    assert_sbc_coverage(result)
    for summary in result.profile_summaries:
        assert summary.passed is True
        assert summary.abs_error <= sbc.coverage_abs_tolerance


@pytest.mark.unit
def test_credible_interval_helpers() -> None:
    samples = np.linspace(0.0, 1.0, 1001)
    lo, hi = equal_tailed_credible_interval(samples, level=0.68)
    assert lo < 0.5 < hi
    assert truth_in_interval(0.5, lo, hi) is True
    assert truth_in_interval(-1.0, lo, hi) is False


@pytest.mark.unit
def test_inject_events_and_analytic_recover() -> None:
    rng = np.random.default_rng(7)
    edges = np.array([0.5, 1.0, 2.0, 4.0])
    heights = np.array([10.0, 20.0, 10.0])
    counts = inject_binned_counts(
        heights, astrometric_sf=1.0, followup_sf=0.5, rng=rng
    )
    assert counts.shape == (3,)
    assert np.all(counts >= 0)
    events = inject_unbinned_events(
        heights,
        bin_edges=edges,
        astrometric_sf=1.0,
        followup_sf=0.5,
        rng=np.random.default_rng(8),
    )
    assert all(e.weight == 1.0 for e in events)
    sbc = _sbc_config(astrometric_sf=1.0, followup_sf=0.5, n_posterior_samples=500)
    samples = recover_heights_analytic(counts, sbc=sbc, rng=np.random.default_rng(9))
    assert samples.shape == (500, 3)
    assert np.all(samples > 0.0)


@pytest.mark.unit
def test_sbc_artifact_and_emit_hook(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path)
    sbc = _sbc_config(
        n_mass_bins=2,
        n_repeats=6,
        expected_total_rate=30.0,
        coverage_abs_tolerance=0.35,
        injected_profiles=[
            InjectedMassFunctionProfile(name="a", relative_heights=[1.0, 2.0]),
            InjectedMassFunctionProfile(name="b", relative_heights=[2.0, 1.0]),
        ],
        random_seed=11,
    )
    cfg.diagnostics.sbc = sbc
    result = run_sbc_suite(cfg, sbc=sbc)
    path = tmp_path / "sbc.h5"
    write_sbc_artifact(path, result)
    payload = read_sbc_artifact(path)
    assert payload["stage"] == "diagnostics_sbc"
    assert payload["n_records"] == len(result.records)

    dirs = resolve_diagnostic_dirs(cfg, run_id="sbc_hook")
    emission = emit_sbc_recovery(cfg, dirs, sbc=sbc)
    assert emission.skipped_reason is None
    assert any(p.name == "sbc_recovery.txt" for p in emission.reports)
    assert any(p.suffix == ".h5" for p in emission.reports)


@pytest.mark.unit
def test_diagnostics_stage_skips_sbc_by_default(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path)
    assert cfg.diagnostics.sbc.run_in_stage is False
    result = run_diagnostics_scaffolding(cfg, run_id="no_sbc")
    assert result.sbc_payload is None
    report = result.as_dict()["notes"]
    assert "SBC" in report


@pytest.mark.unit
def test_diagnostics_stage_can_run_sbc(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path)
    cfg.diagnostics.sbc = _sbc_config(
        run_in_stage=True,
        n_mass_bins=2,
        n_repeats=5,
        expected_total_rate=25.0,
        coverage_abs_tolerance=0.40,
        injected_profiles=[
            InjectedMassFunctionProfile(name="x", relative_heights=[1.0, 1.0]),
        ],
        random_seed=3,
    )
    result = run_diagnostics_scaffolding(cfg, run_id="with_sbc", run_sbc=True)
    assert result.sbc_payload is not None
    assert "overall_empirical_coverage" in result.sbc_payload
    assert any(h.hook_name == "sbc_recovery" for h in result.hooks_run)


@pytest.mark.unit
@pytest.mark.api
def test_assert_sbc_coverage_failure_message() -> None:
    cfg = load_config().model_copy(deep=True)
    # Extremely tight tolerance forces a failure on finite Monte Carlo.
    sbc = _sbc_config(
        n_mass_bins=2,
        n_repeats=4,
        expected_total_rate=8.0,
        coverage_abs_tolerance=1.0e-12,
        injected_profiles=[
            InjectedMassFunctionProfile(name="tiny", relative_heights=[1.0, 1.0]),
        ],
        random_seed=1,
    )
    result = run_sbc_suite(cfg, sbc=sbc)
    if result.overall_passed and all(p.passed for p in result.profile_summaries):
        pytest.skip("degenerate exact match at pathological tolerance")
    with pytest.raises(AssertionError, match="SBC credible-interval coverage failed"):
        assert_sbc_coverage(result)


@pytest.mark.slow
def test_sbc_multi_injection_analytic_coverage_suite() -> None:
    """Fuller multi-profile analytic SBC with coverage assert (not required CI)."""
    cfg = load_config().model_copy(deep=True)
    sbc = _sbc_config(
        recovery_backend="analytic_binned",
        n_mass_bins=4,
        n_repeats=30,
        expected_total_rate=80.0,
        coverage_abs_tolerance=0.10,
        credible_interval_level=0.68,
        n_posterior_samples=2500,
        injected_profiles=[
            InjectedMassFunctionProfile(
                name="flat", relative_heights=[1.0, 1.0, 1.0, 1.0]
            ),
            InjectedMassFunctionProfile(
                name="rising", relative_heights=[0.5, 1.0, 2.0, 3.0]
            ),
            InjectedMassFunctionProfile(
                name="falling", relative_heights=[3.0, 2.0, 1.0, 0.5]
            ),
            InjectedMassFunctionProfile(
                name="peaked", relative_heights=[0.5, 2.5, 2.5, 0.5]
            ),
        ],
        random_seed=69001,
    )
    result = run_sbc_suite(cfg, sbc=sbc)
    assert len(result.profile_summaries) == 4
    assert len(result.records) == 4 * 30 * 4
    assert_sbc_coverage(result)


@pytest.mark.slow
def test_sbc_multi_injection_dynesty_coverage() -> None:
    """Fuller multi-profile dynesty recovery suite (not required CI)."""
    pytest.importorskip("dynesty")
    cfg = load_config().model_copy(deep=True)
    sbc = _sbc_config(
        recovery_backend="dynesty",
        n_mass_bins=3,
        n_repeats=3,
        expected_total_rate=50.0,
        coverage_abs_tolerance=0.55,
        credible_interval_level=0.68,
        inference_nlive=15,
        inference_maxcall=400,
        inference_dlogz=1.5,
        inference_n_mass_grid=20,
        injected_profiles=[
            InjectedMassFunctionProfile(
                name="flat", relative_heights=[1.0, 1.0, 1.0]
            ),
            InjectedMassFunctionProfile(
                name="rising", relative_heights=[0.5, 1.0, 2.5]
            ),
            InjectedMassFunctionProfile(
                name="peaked", relative_heights=[0.5, 2.0, 0.5]
            ),
        ],
        random_seed=69,
    )
    result = run_sbc_suite(cfg, sbc=sbc, recovery_backend="dynesty")
    assert result.recovery_backend == "dynesty"
    assert len(result.profile_summaries) == 3
    assert result.n_repeats == 3
    assert len(result.records) == 3 * 3 * 3
    report = format_sbc_report(result)
    assert "recovery_backend: dynesty" in report
    assert np.isfinite(result.overall_empirical_coverage)
    # Tiny nested-sampling budgets are not science-grade; require the suite to
    # complete with finite coverage and a full-detail report, not tight CI match.
    assert 0.0 <= result.overall_empirical_coverage <= 1.0
    for rec in result.records:
        assert rec.ci_low <= rec.ci_high
        assert np.isfinite(rec.posterior_median)
