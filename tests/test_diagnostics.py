"""Tests for plotting primitives and diagnostics scaffolding (issue #39)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.diagnostics import (
    DEFAULT_ELBADRY_PANEL_ORDER,
    clear_diagnostic_helpers,
    emit_elbadry_six_panel,
    emit_fit_tier_coverage,
    emit_funnel_sky,
    emit_gate_pass_rate,
    format_diagnostics_stage_report,
    format_fit_tier_coverage_report,
    format_funnel_report,
    format_gate_pass_rate_report,
    get_diagnostic_helper,
    list_diagnostic_helpers,
    read_diagnostics_artifact,
    register_diagnostic_helper,
    resolve_diagnostic_dirs,
    run_diagnostics_scaffolding,
    run_diagnostics_stage,
    write_diagnostics_artifact,
    _ensure_builtin_helpers_registered,
)
from darkhunter_pop.plotting import (
    matplotlib_available,
    plot_categorical_bars,
    plot_histogram,
    plot_overlay_histograms,
    plot_six_panel_grid,
    plot_sky_mollweide,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import FitTier, StageStatus

pytestmark = pytest.mark.unit


def test_config_loads_diagnostics_fragment() -> None:
    cfg = load_config()
    assert cfg.diagnostics.figure_dpi == 120
    assert cfg.diagnostics.figures_subdir == "figures"
    assert cfg.diagnostics.reports_subdir == "reports"
    assert cfg.diagnostics.write_figures is True
    assert cfg.diagnostics.hooks.funnel_sky is True
    assert cfg.diagnostics.hooks.elbadry_six_panel is True
    assert cfg.diagnostics.hooks.known_truth_benchmarks is True
    assert cfg.diagnostics.hooks.comparison_catalogs is True


def test_registry_fingerprints_diagnostics_config() -> None:
    spec = STAGE_REGISTRY["diagnostics"]
    assert spec.module.endswith("diagnostics")
    assert "paths.artifact_root" in spec.config_fingerprint_keys
    assert "diagnostics" in spec.config_fingerprint_keys
    assert "benchmarks" in spec.config_fingerprint_keys
    assert spec.inputs_from == ("inference",)
    assert "darkhunter_pop.plotting" not in spec.dependency_modules
    assert "darkhunter_pop.benchmarks" in spec.dependency_modules


def test_builtin_helpers_registered() -> None:
    clear_diagnostic_helpers()
    _ensure_builtin_helpers_registered()
    names = list_diagnostic_helpers()
    assert "emit_funnel_sky" in names
    assert "emit_elbadry_six_panel" in names
    assert "emit_fit_tier_coverage" in names
    assert "emit_gate_pass_rate" in names
    assert "emit_known_truth_benchmarks" in names
    assert "emit_comparison_catalogs" in names
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
    )
    assert "pass_rate_among_scored: 0.8000" in gate
    assert "chi2/dof threshold is config-owned" in gate

    tiers = format_fit_tier_coverage_report(
        {
            FitTier.BULK_ESTIMATE.value: 3,
            FitTier.FULL_UBERMS.value: 1,
        }
    )
    assert "bulk_estimate: count=3" in tiers
    assert "full_uberMS: count=1" in tiers


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
    assert tiers.reports and "fit-tier coverage" in tiers.reports[0].read_text(encoding="utf-8")

    gate = emit_gate_pass_rate(cfg, dirs, counts={"passed": 1, "failed": 0, "skipped": 0})
    assert gate.reports and gate.reports[0].is_file()


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


def test_scaffolding_stage_writes_hdf5_and_report(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    result = run_diagnostics_scaffolding(cfg, run_id="scaffold_run", demo_hooks=True)
    assert result.schema_version == 1
    assert result.dirs.root.is_dir()
    assert "emit_funnel_sky" in result.helpers_registered
    assert len(result.hooks_run) == 4

    artifact = tmp_path / "out.h5"
    write_diagnostics_artifact(artifact, result)
    payload = read_diagnostics_artifact(artifact)
    assert payload["schema_version"] == 1
    assert "helpers_registered" in payload
    with h5py.File(artifact, "r") as handle:
        assert handle.attrs["stage"] == "diagnostics"

    report = format_diagnostics_stage_report(result)
    assert "diagnostics stage" in report
    assert "issue #70" in report or "known-truth" in report


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
    assert "known-truth" in payload["notes"] or "issue #70" in payload["notes"]
