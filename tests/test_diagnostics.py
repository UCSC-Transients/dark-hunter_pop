"""Tests for plotting primitives and the Phase 6 diagnostic suite (issues #39 / #71)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.companion_nature import age_bin_diagnostic
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.diagnostics import (
    DEFAULT_ELBADRY_PANEL_ORDER,
    DIAGNOSTICS_SCHEMA_VERSION,
    assess_sampler_consistency,
    clear_diagnostic_helpers,
    count_fit_tiers,
    emit_age_stratified_wd,
    emit_elbadry_six_panel,
    emit_fit_tier_coverage,
    emit_funnel_sky,
    emit_gate_pass_rate,
    emit_info_gain_followup,
    emit_mc_noise_convergence,
    emit_sampler_consistency,
    emit_solution_type_fractions,
    emit_triples_robustness,
    format_diagnostics_stage_report,
    format_fit_tier_coverage_report,
    format_funnel_report,
    format_gate_pass_rate_report,
    get_diagnostic_helper,
    list_diagnostic_helpers,
    rank_information_gain,
    read_diagnostics_artifact,
    register_diagnostic_helper,
    resolve_diagnostic_dirs,
    run_diagnostic_suite,
    run_diagnostics_scaffolding,
    run_diagnostics_stage,
    write_diagnostics_artifact,
    _ensure_builtin_helpers_registered,
    _hydrate_diagnostics_from_manifest,
)
from darkhunter_pop.forward_model import (
    SOLUTION_TYPE_LABELS,
    SolutionTypeFractionResult,
    run_solution_type_validation,
)
from darkhunter_pop.plotting import (
    apply_axes_style,
    matplotlib_available,
    plot_categorical_bars,
    plot_grouped_bars,
    plot_histogram,
    plot_line_with_threshold,
    plot_overlay_histograms,
    plot_six_panel_grid,
    plot_sky_mollweide,
    require_pyplot,
    resolve_histogram_bins,
    series_style,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    TRIPLES_DISABLED_SKIP_REASON,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    CandidateRecord,
    FitTier,
    ParameterSet,
    StageRecord,
    StageStatus,
)
from darkhunter_pop.sensitivity_analysis import run_mc_noise_convergence

pytestmark = pytest.mark.unit


def _m1(value: float, sigma: float) -> ParameterSet:
    return ParameterSet(
        names=["M1"],
        values=[value],
        covariance=[[sigma**2]],
        provenance="test",
        units=["Msun"],
    )


def _candidate(
    *,
    source_id: int,
    fit_tier: FitTier | None = FitTier.BULK_ESTIMATE,
    m1_value: float = 1.0,
    m1_sigma: float = 0.1,
    age_gyr: float | None = 2.0,
    weights: dict[str, float] | None = None,
) -> CandidateRecord:
    extras: dict = {}
    if age_gyr is not None:
        extras["age_gyr"] = age_gyr
    if weights is None:
        weights = {k: 0.2 for k in COMPANION_NATURE_WEIGHT_KEYS}
    return CandidateRecord(
        source_id=source_id,
        fit_tier=fit_tier,
        m1=_m1(m1_value, m1_sigma),
        companion_nature_weights=weights,
        extras=extras,
    )


def test_config_loads_diagnostics_fragment() -> None:
    cfg = load_config()
    assert cfg.diagnostics.figure_dpi == 120
    assert cfg.diagnostics.figures_subdir == "figures"
    assert cfg.diagnostics.reports_subdir == "reports"
    assert cfg.diagnostics.write_figures is True
    assert cfg.diagnostics.histogram_max_bins == 80
    assert cfg.diagnostics.sky_map_point_size == pytest.approx(0.1)
    assert cfg.diagnostics.sky_map_alpha == pytest.approx(0.25)
    assert cfg.diagnostics.info_gain_top_n == 20
    assert cfg.diagnostics.sampler_logz_sigma_tol == 3.0
    assert cfg.diagnostics.hooks.funnel_sky is True
    assert cfg.diagnostics.hooks.elbadry_six_panel is True
    assert cfg.diagnostics.hooks.age_stratified_wd is True
    assert cfg.diagnostics.hooks.triples_robustness is True
    assert cfg.diagnostics.hooks.info_gain_followup is True
    assert cfg.diagnostics.hooks.sampler_consistency is True
    assert cfg.diagnostics.hooks.mc_noise_convergence is True
    assert cfg.diagnostics.hooks.solution_type_fractions is True
    assert cfg.diagnostics.hooks.known_truth_benchmarks is True
    assert cfg.diagnostics.hooks.comparison_catalogs is True
    assert cfg.diagnostics.hooks.sbc_recovery is True
    assert cfg.diagnostics.hooks.m2_posterior_convergence is True
    assert cfg.diagnostics.sbc.enabled is True
    assert cfg.diagnostics.sbc.run_in_stage is False


def test_config_loads_plotting_style_fragment() -> None:
    cfg = load_config()
    assert cfg.plotting.font_family == "serif"
    assert cfg.plotting.axes_label_fontsize == pytest.approx(18.0)
    assert cfg.plotting.tick_label_fontsize == pytest.approx(14.0)
    assert cfg.plotting.tick_direction == "in"
    assert cfg.plotting.tick_width == pytest.approx(2.0)
    assert cfg.plotting.tick_major_length == pytest.approx(8.0)
    assert cfg.plotting.tick_minor_length == pytest.approx(4.0)
    assert cfg.plotting.line_width == pytest.approx(2.0)
    assert "#0072B2" in cfg.plotting.color_cycle
    assert cfg.plotting.figsize_portrait == (5.0, 7.0)


def test_registry_fingerprints_diagnostics_config() -> None:
    spec = STAGE_REGISTRY["diagnostics"]
    assert spec.module.endswith("diagnostics")
    assert "paths.artifact_root" in spec.config_fingerprint_keys
    assert "diagnostics" in spec.config_fingerprint_keys
    assert "benchmarks" in spec.config_fingerprint_keys
    assert spec.inputs_from == ("inference",)
    assert "darkhunter_pop.plotting" not in spec.dependency_modules
    assert "darkhunter_pop.benchmarks" in spec.dependency_modules
    assert "darkhunter_pop.sbc" in spec.dependency_modules


def test_builtin_helpers_registered() -> None:
    clear_diagnostic_helpers()
    _ensure_builtin_helpers_registered()
    names = list_diagnostic_helpers()
    assert "emit_funnel_sky" in names
    assert "emit_elbadry_six_panel" in names
    assert "emit_fit_tier_coverage" in names
    assert "emit_gate_pass_rate" in names
    assert "emit_age_stratified_wd" in names
    assert "emit_triples_robustness" in names
    assert "emit_info_gain_followup" in names
    assert "emit_sampler_consistency" in names
    assert "emit_mc_noise_convergence" in names
    assert "emit_solution_type_fractions" in names
    assert "emit_known_truth_benchmarks" in names
    assert "emit_comparison_catalogs" in names
    assert "emit_sbc_recovery" in names
    assert "emit_m2_posterior_convergence" in names
    assert callable(get_diagnostic_helper("emit_funnel_sky"))


def test_register_custom_helper() -> None:
    clear_diagnostic_helpers()
    _ensure_builtin_helpers_registered()

    def _probe() -> str:
        return "ok"

    register_diagnostic_helper("custom_probe", _probe)
    assert get_diagnostic_helper("custom_probe")() == "ok"
    clear_diagnostic_helpers()
    _ensure_builtin_helpers_registered()


def test_resolve_diagnostic_dirs_layout(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="run_test")
    assert dirs.root == tmp_path / "run_test" / "diagnostics"
    assert dirs.figures == dirs.root / "figures"
    assert dirs.reports == dirs.root / "reports"
    assert dirs.figures.is_dir()
    assert dirs.reports.is_dir()

    artifact = tmp_path / "run_test" / "data_acquisition" / "abc.h5"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"")
    beside = resolve_diagnostic_dirs(cfg, run_id="run_test", beside_artifact=artifact)
    assert beside.root == artifact.parent / "abc_diagnostics"


def test_funnel_and_gate_reports_are_full_detail() -> None:
    funnel = format_funnel_report(
        {"queried": 10, "after_quality_cut": 8, "candidates_written": 7},
        quality_cut_bin_counts={"bin0": 5, "bin1": 3},
    )
    assert "=== data_acquisition funnel ===" in funnel
    assert "queried: 10" in funnel
    assert "bin0: 5" in funnel

    gate = format_gate_pass_rate_report(
        {"passed": 4, "failed": 1, "skipped": 2},
        gate_name="rv_astrometry_gate",
        chi2_dof_values=[1.0, 2.0, 3.0],
        chi2_dof_threshold=5.0,
    )
    assert "pass_rate_among_scored: 0.8000" in gate
    assert "chi2/dof threshold is config-owned" in gate
    assert "chi2_dof_threshold" in gate

    tiers = format_fit_tier_coverage_report(
        {
            FitTier.BULK_ESTIMATE.value: 3,
            FitTier.FULL_UBERMS.value: 1,
        }
    )
    assert "bulk_estimate: count=3" in tiers
    assert "full_uberMS: count=1" in tiers


def test_count_fit_tiers_and_info_gain_rank() -> None:
    cfg = load_config()
    cands = [
        _candidate(source_id=1, fit_tier=FitTier.BULK_ESTIMATE, m1_sigma=0.05),
        _candidate(source_id=2, fit_tier=FitTier.FULL_UBERMS, m1_sigma=0.4),
        _candidate(source_id=3, fit_tier=None, m1_sigma=0.1),
    ]
    counts = count_fit_tiers(cands)
    assert counts[FitTier.BULK_ESTIMATE.value] == 1
    assert counts[FitTier.FULL_UBERMS.value] == 1
    assert counts["unset"] == 1
    ranked = rank_information_gain(cands, cfg)
    assert ranked[0].source_id == 2
    assert ranked[0].rank == 1


def test_sampler_consistency_multi_run() -> None:
    ok = assess_sampler_consistency(
        [
            {"logz": 10.0, "logz_err": 0.5},
            {"logz": 10.2, "logz_err": 0.5},
            {"logz": 9.9, "logz_err": 0.5},
        ],
        logz_sigma_tol=3.0,
    )
    assert ok.n_runs == 3
    assert ok.consistent is True
    bad = assess_sampler_consistency(
        [
            {"logz": 0.0, "logz_err": 0.01},
            {"logz": 10.0, "logz_err": 0.01},
        ],
        logz_sigma_tol=3.0,
    )
    assert bad.consistent is False


def test_emit_hooks_write_reports(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="hooks")
    funnel = emit_funnel_sky(
        cfg,
        dirs,
        funnel_counts={"queried": 3, "after_quality_cut": 2, "candidates_written": 2},
        ruwe=[1.1, 1.2],
        ra_deg=[10.0, 20.0],
        dec_deg=[-5.0, 5.0],
    )
    assert funnel.skipped_reason is None
    assert len(funnel.reports) == 1
    assert funnel.reports[0].is_file()

    panels = {
        name: {"mock": np.linspace(0, 1, 20), "real": np.linspace(0.1, 1.1, 15)}
        for name in DEFAULT_ELBADRY_PANEL_ORDER
    }
    six = emit_elbadry_six_panel(cfg, dirs, panels=panels)
    assert six.reports and six.reports[0].is_file()
    assert "P_orb_days" in six.reports[0].read_text(encoding="utf-8")

    tiers = emit_fit_tier_coverage(
        cfg,
        dirs,
        counts={FitTier.BULK_ESTIMATE.value: 2, FitTier.FULL_UBERMS.value: 1},
    )
    assert tiers.reports and "fit-tier coverage" in tiers.reports[0].read_text(
        encoding="utf-8"
    )

    gate = emit_gate_pass_rate(
        cfg,
        dirs,
        counts={"passed": 1, "failed": 0, "skipped": 0},
        chi2_dof_values=[0.8, 1.1],
    )
    assert gate.reports and gate.reports[0].is_file()
    assert "chi2_dof:" in gate.reports[0].read_text(encoding="utf-8")


def test_required_suite_hooks_write_reports(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="suite")
    cands = [
        _candidate(
            source_id=10,
            age_gyr=1.0,
            weights={
                "BH": 0.05,
                "NS": 0.05,
                "WD": 0.6,
                "other": 0.2,
                "outlier": 0.1,
            },
        ),
        _candidate(
            source_id=11,
            age_gyr=8.0,
            fit_tier=FitTier.FULL_UBERMS,
            m1_sigma=0.5,
            weights={
                "BH": 0.05,
                "NS": 0.05,
                "WD": 0.55,
                "other": 0.25,
                "outlier": 0.1,
            },
        ),
    ]

    age = emit_age_stratified_wd(cfg, dirs, candidates=cands)
    assert age.skipped_reason is None
    text = age.reports[0].read_text(encoding="utf-8")
    assert "age-stratified WD-debiasing" in text
    assert "age_independence_ok" in text

    triples = emit_triples_robustness(cfg, dirs)
    assert triples.skipped_reason is None
    ttext = triples.reports[0].read_text(encoding="utf-8")
    assert "stub-safe skip" in ttext
    assert TRIPLES_DISABLED_SKIP_REASON in ttext

    info = emit_info_gain_followup(cfg, dirs, candidates=cands)
    assert "follow-up priority" in info.reports[0].read_text(encoding="utf-8")

    samp = emit_sampler_consistency(
        cfg,
        dirs,
        sampler_runs=[
            {"logz": 1.0, "logz_err": 0.2, "seed": 1},
            {"logz": 1.1, "logz_err": 0.2, "seed": 18},
            {"logz": 0.95, "logz_err": 0.2, "seed": 35},
        ],
    )
    assert "CONSISTENT" in samp.reports[0].read_text(encoding="utf-8")

    mc = run_mc_noise_convergence(
        [8.0, 4.0],
        threshold=float(cfg.physics.mc_noise_threshold),
        n_mock_start=16,
        n_mock_max=256,
        growth_factor=2.0,
    )
    mc_hook = emit_mc_noise_convergence(cfg, dirs, diagnostic=mc)
    assert "Poisson-negligibility" in mc_hook.reports[0].read_text(encoding="utf-8")

    frac = {label: 1.0 / len(SOLUTION_TYPE_LABELS) for label in SOLUTION_TYPE_LABELS}
    st = SolutionTypeFractionResult(
        mock_fractions=dict(frac),
        real_fractions=dict(frac),
        max_abs_delta=0.0,
        passed=True,
    )
    st_hook = emit_solution_type_fractions(cfg, dirs, result=st)
    assert st_hook.payload["passed"] is True
    assert "solution-type-fraction" in st_hook.reports[0].read_text(encoding="utf-8")


def test_triples_enabled_comparison(tmp_path: Path) -> None:
    cfg = load_config()
    enabled = cfg.model_copy(deep=True)
    enabled.triples.enabled = True
    enabled = enabled.model_copy(
        update={
            "paths": enabled.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": enabled.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    dirs = resolve_diagnostic_dirs(enabled, run_id="triples_on")
    hook = emit_triples_robustness(
        enabled,
        dirs,
        flagged_source_ids=[101, 102],
        metrics_with_flagged={"n_events": 10.0, "mean_wd": 0.4},
        metrics_without_flagged={"n_events": 8.0, "mean_wd": 0.35},
    )
    text = hook.reports[0].read_text(encoding="utf-8")
    assert "comparison active" in text
    assert "absolute_deltas" in text


def test_hook_disable_skips_emission(tmp_path: Path) -> None:
    cfg = load_config()
    hooks = cfg.diagnostics.hooks.model_copy(update={"funnel_sky": False})
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"hooks": hooks}),
        }
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="skip")
    result = emit_funnel_sky(cfg, dirs, funnel_counts={"queried": 1})
    assert result.skipped_reason is not None
    assert "funnel_sky=false" in result.skipped_reason


def test_solution_type_validation_gate_still_green() -> None:
    """Existing SF solution-type gate helper remains available and passes when matched."""
    from darkhunter_pop.forward_model import MockRealizationRecord, SolutionType

    records = [
        MockRealizationRecord(
            solution_type=SolutionType.TWELVE_PARAMETER_ORBITAL,
            accepted_orbital=True,
        )
        for _ in range(10)
    ] + [
        MockRealizationRecord(
            solution_type=SolutionType.FIVE_PARAMETER,
            accepted_orbital=False,
        )
        for _ in range(10)
    ]
    real = {
        label: 0.0 for label in SOLUTION_TYPE_LABELS
    }
    real[SolutionType.TWELVE_PARAMETER_ORBITAL.value] = 0.5
    real[SolutionType.FIVE_PARAMETER.value] = 0.5
    result = run_solution_type_validation(records, real, max_abs_delta=0.05)
    assert result.passed is True


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_apply_axes_style_inward_ticks_and_serif() -> None:
    cfg = load_config().plotting
    plt = require_pyplot()
    fig, axis = plt.subplots()
    apply_axes_style(
        axis,
        cfg,
        xlabel=r"Right Ascension (deg)",
        ylabel=r"Declination (deg)",
        title="sky",
    )
    family = axis.xaxis.get_label().get_fontfamily()
    if isinstance(family, str):
        assert family == cfg.font_family
    else:
        assert cfg.font_family in family
    assert axis.xaxis.get_label().get_fontsize() == pytest.approx(cfg.axes_label_fontsize)
    assert axis.xaxis.majorTicks[0].tick1line.get_markeredgewidth() == pytest.approx(
        cfg.tick_width
    )
    sty = series_style(1, cfg)
    assert sty["color"] == cfg.color_cycle[1]
    assert sty["linestyle"] == cfg.linestyle_cycle[1 % len(cfg.linestyle_cycle)]
    plt.close(fig)


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_plotting_primitives_write_pngs(tmp_path: Path) -> None:
    dpi = 80
    hist = plot_histogram(
        np.array([1.0, 2.0, 2.5, 3.0]),
        tmp_path / "hist.png",
        xlabel="x",
        title="hist",
        dpi=dpi,
    )
    assert hist is not None and hist.is_file()

    sky = plot_sky_mollweide(
        np.array([10.0, 200.0]),
        np.array([-20.0, 40.0]),
        tmp_path / "sky.png",
        dpi=dpi,
    )
    assert sky is not None and sky.is_file()

    overlay = plot_overlay_histograms(
        {"mock": np.linspace(0, 1, 50), "real": np.linspace(0.2, 1.2, 40)},
        tmp_path / "overlay.png",
        xlabel="x",
        title="overlay",
        dpi=dpi,
    )
    assert overlay is not None and overlay.is_file()

    panels = {
        name: {"mock": np.random.default_rng(0).normal(size=30)}
        for name in DEFAULT_ELBADRY_PANEL_ORDER
    }
    grid = plot_six_panel_grid(
        panels,
        tmp_path / "six.png",
        panel_order=DEFAULT_ELBADRY_PANEL_ORDER,
        dpi=dpi,
    )
    assert grid is not None and grid.is_file()

    bars = plot_categorical_bars(
        ["a", "b"],
        [1.0, 2.0],
        tmp_path / "bars.png",
        xlabel="cat",
        ylabel="n",
        title="bars",
        dpi=dpi,
    )
    assert bars is not None and bars.is_file()

    grouped = plot_grouped_bars(
        ["x", "y"],
        {"mock": [0.4, 0.6], "real": [0.45, 0.55]},
        tmp_path / "grouped.png",
        xlabel="bin",
        ylabel="frac",
        title="grouped",
        dpi=dpi,
    )
    assert grouped is not None and grouped.is_file()

    line = plot_line_with_threshold(
        [10, 20, 40, 80],
        [0.3, 0.2, 0.12, 0.08],
        tmp_path / "conv.png",
        xlabel="n_mock",
        ylabel="ratio",
        title="conv",
        dpi=dpi,
        threshold=0.1,
        log_x=True,
    )
    assert line is not None and line.is_file()


def test_resolve_histogram_bins_caps_auto_for_heavy_tails() -> None:
    """Heavy-tailed large-N samples must not keep unbounded auto bin counts (#96)."""
    rng = np.random.default_rng(96)
    values = np.concatenate(
        [rng.exponential(1.5, size=50_000) + 1.0, rng.uniform(40.0, 80.0, size=200)]
    )
    uncapped_counts, _ = np.histogram(values, bins="auto")
    assert len(uncapped_counts) > 80
    resolved = resolve_histogram_bins(values, "auto", max_bins=80)
    assert resolved == 80
    capped_counts, _ = np.histogram(values, bins=resolved)
    assert len(capped_counts) == 80
    assert capped_counts.max() > 0


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_plot_histogram_drops_nonfinite_values(tmp_path: Path) -> None:
    """NaN/inf must not abort bins=auto (eccentricity has missing NSS rows)."""
    values = np.array([0.0, 0.1, np.nan, 0.2, np.inf, 0.3, -np.inf], dtype=np.float64)
    path = plot_histogram(
        values,
        tmp_path / "ecc_nan.png",
        xlabel="eccentricity",
        title="eccentricity",
        dpi=80,
        max_bins=80,
    )
    assert path is not None and path.is_file()
    assert plot_histogram(
        np.array([np.nan, np.inf]),
        tmp_path / "all_nan.png",
        xlabel="x",
        title="empty",
        dpi=80,
        max_bins=80,
    ) is None


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_plot_histogram_xlim_filters_and_clips_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.array([1.0, 2.0, 5.0, 50_000.0], dtype=np.float64)
    titles: list[str] = []

    def _capture_title(axis: object, cfg: object, **kwargs: object) -> None:
        titles.append(str(kwargs.get("title", "")))
        apply_axes_style(axis, cfg, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("darkhunter_pop.plotting.apply_axes_style", _capture_title)
    path = plot_histogram(
        values,
        tmp_path / "m2.png",
        xlabel="mass",
        title="M2",
        dpi=80,
        max_bins=10,
        xlim=(0.0, 30.0),
        log_y=True,
    )
    assert path is not None and path.is_file()
    assert titles == ["M2 (1 > 30 M$_\\odot$ omitted)"]


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_plot_histogram_max_bins_keeps_visible_bars(tmp_path: Path) -> None:
    rng = np.random.default_rng(96)
    values = np.concatenate(
        [
            rng.exponential(200.0, size=20_000) + 0.2,
            rng.uniform(4000.0, 9000.0, size=50),
        ]
    )
    path = plot_histogram(
        values,
        tmp_path / "period_capped.png",
        xlabel="period (day)",
        title="period_day",
        dpi=80,
        max_bins=80,
    )
    assert path is not None and path.is_file()
    uncapped, _ = np.histogram(values, bins="auto")
    assert len(uncapped) > 80


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_emit_funnel_sky_uses_config_hist_and_sky_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """funnel_sky must pass diagnostics histogram/sky rendering knobs (#96, #97)."""
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path / "art")}),
        }
    )
    assert cfg.diagnostics.histogram_max_bins == 80
    assert cfg.diagnostics.sky_map_point_size == pytest.approx(0.1)
    assert cfg.diagnostics.sky_map_alpha == pytest.approx(0.25)

    seen: dict[str, object] = {}

    def _fake_hist(values, path, **kwargs):  # type: ignore[no-untyped-def]
        seen["hist_max_bins"] = kwargs.get("max_bins")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        return Path(path)

    def _fake_sky(ra, dec, path, **kwargs):  # type: ignore[no-untyped-def]
        seen["sky_point_size"] = kwargs.get("point_size")
        seen["sky_alpha"] = kwargs.get("alpha")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        return Path(path)

    monkeypatch.setattr("darkhunter_pop.diagnostics.plot_histogram", _fake_hist)
    monkeypatch.setattr("darkhunter_pop.diagnostics.plot_sky_mollweide", _fake_sky)

    dirs = resolve_diagnostic_dirs(cfg, run_id="t", beside_artifact=tmp_path / "x.h5")
    emit_funnel_sky(
        cfg,
        dirs,
        funnel_counts={"queried": 3, "candidates_written": 3},
        ruwe=[1.1, 1.2, 40.0],
        period_day=[10.0, 400.0, 8000.0],
        eccentricity=[0.1, 0.2, 0.3],
        ra_deg=[10.0, 20.0, 30.0],
        dec_deg=[-10.0, 0.0, 10.0],
    )
    assert seen["hist_max_bins"] == 80
    assert seen["sky_point_size"] == pytest.approx(0.1)
    assert seen["sky_alpha"] == pytest.approx(0.25)


def test_scaffolding_stage_writes_hdf5_and_report(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    result = run_diagnostics_scaffolding(cfg, run_id="scaffold_run", demo_hooks=True)
    assert result.schema_version == DIAGNOSTICS_SCHEMA_VERSION
    assert result.dirs.root.is_dir()
    assert "emit_funnel_sky" in result.helpers_registered
    assert len(result.hooks_run) >= 8
    hook_names = {h.hook_name for h in result.hooks_run}
    assert "fit_tier_coverage" in hook_names
    assert "triples_robustness" in hook_names
    assert "sampler_consistency" in hook_names
    assert "known_truth_benchmarks" in hook_names
    assert "comparison_catalogs" in hook_names

    artifact = tmp_path / "out.h5"
    write_diagnostics_artifact(artifact, result)
    payload = read_diagnostics_artifact(artifact)
    assert payload["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert "helpers_registered" in payload
    with h5py.File(artifact, "r") as handle:
        assert handle.attrs["stage"] == "diagnostics"

    report = format_diagnostics_stage_report(result)
    assert "diagnostics stage (full suite)" in report
    assert "Phase 6" in report or "SBC recovery" in report or "known-truth" in report
    assert "sbc_config:" in report


def test_hydrate_diagnostics_from_companion_nature(tmp_path: Path) -> None:
    from darkhunter_pop.companion_nature import write_stage_hdf5 as write_cn_hdf5

    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path / "out")}),
        }
    )
    manifest = create_run_manifest(cfg)
    cn_path = tmp_path / "companion_nature.h5"
    write_cn_hdf5(
        cn_path,
        [_candidate(source_id=1), _candidate(source_id=2)],
        diagnostics={"n_input": 2, "n_weighted": 2},
    )
    manifest = manifest.model_copy(
        update={
            "stages": {
                **manifest.stages,
                "companion_nature_likelihood": StageRecord(
                    stage_name="companion_nature_likelihood",
                    status=StageStatus.COMPLETED,
                    artifact_path=str(cn_path),
                ),
            }
        }
    )
    hydrated = _hydrate_diagnostics_from_manifest(manifest, cfg)
    assert hydrated["candidates"] is not None
    assert len(hydrated["candidates"]) == 2


def test_run_diagnostics_stage_updates_manifest(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path / "out")}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    updated = run_diagnostics_stage(
        manifest, cfg, run_path=run_path, force_rerun=True, demo_hooks=True
    )
    assert updated.stages["diagnostics"].status is StageStatus.COMPLETED
    artifact = stage_artifact_path(
        cfg, STAGE_REGISTRY["diagnostics"], run_id=manifest.run_id
    )
    assert artifact.is_file()
    payload = read_diagnostics_artifact(artifact)
    assert "Phase 6 diagnostic suite" in payload["notes"]
    assert "issue #70" in payload["notes"] or "known-truth" in payload["notes"]
    assert "issue #69" in payload["notes"] or "SBC" in payload["notes"]
    assert "Phase 6 diagnostic suite" in payload["notes"]
    assert "issue #70" in payload["notes"] or "known-truth" in payload["notes"]


def test_age_bin_diagnostic_ties_to_companion_nature() -> None:
    cfg = load_config()
    cands = [
        _candidate(source_id=1, age_gyr=0.5),
        _candidate(source_id=2, age_gyr=5.0),
    ]
    diagnostic = age_bin_diagnostic(cands, cfg.companion_nature)
    assert diagnostic.bin_counts
    assert "Age-bin diagnostic" in diagnostic.message


@pytest.mark.slow
def test_slow_full_suite_with_figures(tmp_path: Path) -> None:
    """Longer end-to-end suite emission (figures + multi-run + MC schedule)."""
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
        }
    )
    rng = np.random.default_rng(71)
    cands = [
        _candidate(
            source_id=1000 + i,
            fit_tier=FitTier.BULK_ESTIMATE if i % 3 else FitTier.FULL_UBERMS,
            m1_sigma=float(0.05 + 0.4 * rng.random()),
            age_gyr=float(rng.uniform(0.2, 10.0)),
            weights={
                "BH": 0.05,
                "NS": 0.1,
                "WD": float(0.4 + 0.2 * rng.random()),
                "other": 0.25,
                "outlier": 0.1,
            },
        )
        for i in range(40)
    ]
    # Renormalize weights.
    for cand in cands:
        w = cand.companion_nature_weights
        assert w is not None
        s = sum(w.values())
        cand.companion_nature_weights = {k: v / s for k, v in w.items()}

    mc = run_mc_noise_convergence(
        [12.0, 6.0, 3.0, 1.5],
        threshold=float(cfg.physics.mc_noise_threshold),
        n_mock_start=8,
        n_mock_max=512,
        growth_factor=1.5,
    )
    frac = {label: 1.0 / len(SOLUTION_TYPE_LABELS) for label in SOLUTION_TYPE_LABELS}
    st = SolutionTypeFractionResult(
        mock_fractions=dict(frac),
        real_fractions=dict(frac),
        max_abs_delta=0.0,
        passed=True,
    )
    sampler_runs = [
        {"logz": 5.0 + 0.05 * k, "logz_err": 0.2, "seed": 71 + 17 * k, "nlive": 50 + 10 * k}
        for k in range(5)
    ]
    result = run_diagnostic_suite(
        cfg,
        run_id="slow_suite",
        candidates=cands,
        funnel_counts={"queried": 100, "after_quality_cut": 80, "candidates_written": 40},
        gate_counts={"passed": 30, "failed": 5, "skipped": 5},
        chi2_dof_values=list(rng.uniform(0.5, 4.0, size=35)),
        sampler_runs=sampler_runs,
        mc_noise=mc,
        solution_types=st,
        demo_missing=False,
    )
    names = {h.hook_name for h in result.hooks_run}
    required = {
        "fit_tier_coverage",
        "gate_pass_rate",
        "age_stratified_wd",
        "triples_robustness",
        "info_gain_followup",
        "sampler_consistency",
        "mc_noise_convergence",
        "solution_type_fractions",
    }
    assert required.issubset(names)
    for hook in result.hooks_run:
        if hook.hook_name in required:
            assert hook.skipped_reason is None
            assert hook.reports
    assert any(h.hook_name == "sampler_consistency" and h.payload.get("consistent") for h in result.hooks_run)
    assert mc.all_bins_passed or mc.n_mock_final >= 8
    assert "known_truth_benchmarks" in names
    assert "comparison_catalogs" in names
