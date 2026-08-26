"""Stage: ``diagnostics`` — scaffolding, known-truth, and comparison catalogs.

``plotting.py`` provides shared rendering primitives; this module decides what to
check and emit. Known-truth Gaia BH benchmarks and comparison-only literature /
external mass-function catalogs are fixture-driven (issue #70). Simulation-based
calibration recovery is issue #69 (separate).

Diagnostic reports and plot captions stay full-detail (caveman exemption).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.benchmarks import (
    assert_required_catalogs_present,
    check_known_truth_expectations,
    format_comparison_catalog_report,
    format_known_truth_report,
    load_all_comparison_catalogs,
    load_known_truth_table_from_config,
    synthetic_observed_from_truth,
    validate_benchmarks_config,
)
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.plotting import (
    MatplotlibUnavailableError,
    matplotlib_available,
    plot_categorical_bars,
    plot_histogram,
    plot_overlay_histograms,
    plot_six_panel_grid,
    plot_sky_mollweide,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import FitTier, RunManifest, StageStatus

# Default El-Badry et al. (2024) panel axis names (ARCHITECTURE.md §4 / forward_model).
DEFAULT_ELBADRY_PANEL_ORDER: tuple[str, ...] = (
    "P_orb_days",
    "G_mag",
    "inv_parallax_mas_inv",
    "eccentricity",
    "f_m_msun",
    "cos_inclination",
)

DiagnosticHelper = Callable[..., Any]


@dataclass(frozen=True)
class DiagnosticDirs:
    """Resolved output directories for one diagnostics emission site."""

    root: Path
    figures: Path
    reports: Path


@dataclass
class HookEmissionResult:
    """Paths written by one diagnostic hook."""

    hook_name: str
    figures: list[Path] = field(default_factory=list)
    reports: list[Path] = field(default_factory=list)
    skipped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook_name": self.hook_name,
            "figures": [str(p) for p in self.figures],
            "reports": [str(p) for p in self.reports],
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class DiagnosticsStageResult:
    """Diagnostics stage output (scaffolding + known-truth + comparison catalogs)."""

    schema_version: int
    dirs: DiagnosticDirs
    hooks_run: list[HookEmissionResult]
    helpers_registered: tuple[str, ...]
    matplotlib_available: bool
    config_snapshot: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root": str(self.dirs.root),
            "figures_dir": str(self.dirs.figures),
            "reports_dir": str(self.dirs.reports),
            "matplotlib_available": self.matplotlib_available,
            "helpers_registered": list(self.helpers_registered),
            "hooks_run": [h.as_dict() for h in self.hooks_run],
            "config_snapshot": self.config_snapshot,
            "notes": (
                "Diagnostics scaffolding plus known-truth Gaia BH benchmarks and "
                "comparison-only catalog reports (issue #70). Simulation-based "
                "calibration recovery is issue #69."
            ),
        }


_HELPER_REGISTRY: dict[str, DiagnosticHelper] = {}


def register_diagnostic_helper(name: str, fn: DiagnosticHelper) -> None:
    """Register a named diagnostic helper for stage/product reuse."""
    if not name:
        raise ValueError("diagnostic helper name must be non-empty")
    _HELPER_REGISTRY[name] = fn


def list_diagnostic_helpers() -> tuple[str, ...]:
    """Return registered helper names in sorted order."""
    return tuple(sorted(_HELPER_REGISTRY))


def get_diagnostic_helper(name: str) -> DiagnosticHelper:
    """Look up a registered helper or raise ``KeyError``."""
    return _HELPER_REGISTRY[name]


def clear_diagnostic_helpers() -> None:
    """Remove all registered helpers (tests only)."""
    _HELPER_REGISTRY.clear()


def resolve_artifact_root(config: PipelineConfig) -> Path:
    """Resolve ``paths.artifact_root`` relative to the repo when not absolute."""
    root = Path(config.paths.artifact_root)
    if not root.is_absolute():
        root = repo_root() / root
    return root


def resolve_diagnostic_dirs(
    config: PipelineConfig,
    *,
    run_id: str,
    beside_artifact: Path | None = None,
) -> DiagnosticDirs:
    """Resolve figure/report directories under config paths.

    Default layout: ``{artifact_root}/{run_id}/diagnostics/{figures,reports}/``.
    When ``beside_artifact`` is set (stage-local emission), directories sit next to
    that HDF5 as ``{stem}_diagnostics/{figures,reports}/``.
    """
    diag = config.diagnostics
    if beside_artifact is not None:
        root = Path(beside_artifact).parent / f"{Path(beside_artifact).stem}_diagnostics"
    else:
        root = resolve_artifact_root(config) / run_id / "diagnostics"
    figures = root / diag.figures_subdir
    reports = root / diag.reports_subdir
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return DiagnosticDirs(root=root, figures=figures, reports=reports)


def write_report(path: Path, text: str) -> Path:
    """Write a full-detail diagnostic text report (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def format_funnel_report(
    funnel_counts: Mapping[str, int],
    *,
    quality_cut_bin_counts: Mapping[str, int] | None = None,
    stage_name: str = "data_acquisition",
) -> str:
    """Full-detail funnel table for stage diagnostics."""
    lines = [
        f"=== {stage_name} funnel ===",
    ]
    for key, value in funnel_counts.items():
        lines.append(f"  {key}: {value}")
    if quality_cut_bin_counts:
        lines.append("  quality_cut_bins:")
        for key, count in sorted(quality_cut_bin_counts.items()):
            lines.append(f"    {key}: {count}")
    lines.append(f"=== end {stage_name} funnel ===")
    return "\n".join(lines)


def format_elbadry_panel_report(
    panels: Mapping[str, Mapping[str, Sequence[float] | NDArray[np.floating]]],
    *,
    panel_order: Sequence[str] = DEFAULT_ELBADRY_PANEL_ORDER,
    title: str = "El-Badry-style six-panel comparison",
) -> str:
    """Full-detail caption/summary for a six-panel emission (no KS thresholds here)."""
    lines = [
        f"=== {title} ===",
        "note: science KS thresholds live in selection_function_astrometric.validation_gate; "
        "this report only summarizes series lengths for plotting hooks.",
    ]
    for name in panel_order:
        series = panels.get(name, {})
        if not series:
            lines.append(f"  {name}: (no series)")
            continue
        parts = []
        for label, values in series.items():
            n = int(np.asarray(values).size)
            parts.append(f"{label}=n={n}")
        lines.append(f"  {name}: " + ", ".join(parts))
    lines.append(f"=== end {title} ===")
    return "\n".join(lines)


def format_fit_tier_coverage_report(counts: Mapping[str, int]) -> str:
    """Full-detail fit-tier coverage summary."""
    lines = ["=== fit-tier coverage ==="]
    total = sum(int(v) for v in counts.values())
    lines.append(f"  total_candidates: {total}")
    for tier in FitTier:
        n = int(counts.get(tier.value, counts.get(tier.name, 0)))
        frac = (n / total) if total else 0.0
        lines.append(f"  {tier.value}: count={n} fraction={frac:.4f}")
    extras = set(counts) - {t.value for t in FitTier} - {t.name for t in FitTier}
    for key in sorted(extras):
        lines.append(f"  {key}: count={int(counts[key])}")
    lines.append("=== end fit-tier coverage ===")
    return "\n".join(lines)


def format_gate_pass_rate_report(
    counts: Mapping[str, int],
    *,
    gate_name: str = "rv_astrometry_gate",
) -> str:
    """Full-detail gate pass/fail stub report (threshold values stay in stage config)."""
    passed = int(counts.get("passed", 0))
    failed = int(counts.get("failed", 0))
    skipped = int(counts.get("skipped", 0))
    total = passed + failed + skipped
    rate = (passed / (passed + failed)) if (passed + failed) else float("nan")
    lines = [
        f"=== {gate_name} pass-rate diagnostic ===",
        f"  passed: {passed}",
        f"  failed: {failed}",
        f"  skipped: {skipped}",
        f"  total: {total}",
        f"  pass_rate_among_scored: {rate:.4f}"
        if passed + failed
        else "  pass_rate_among_scored: undefined (no scored systems)",
        "  note: chi2/dof threshold is config-owned by rv_astrometry_gate; "
        "this hook only records counts.",
        f"=== end {gate_name} pass-rate diagnostic ===",
    ]
    return "\n".join(lines)


def _maybe_plot(write_figures: bool, fn: Callable[[], Path | None]) -> Path | None:
    if not write_figures:
        return None
    try:
        return fn()
    except MatplotlibUnavailableError:
        return None


def emit_funnel_sky(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    funnel_counts: Mapping[str, int],
    quality_cut_bin_counts: Mapping[str, int] | None = None,
    ruwe: NDArray[np.floating] | Sequence[float] | None = None,
    period_day: NDArray[np.floating] | Sequence[float] | None = None,
    eccentricity: NDArray[np.floating] | Sequence[float] | None = None,
    ra_deg: NDArray[np.floating] | Sequence[float] | None = None,
    dec_deg: NDArray[np.floating] | Sequence[float] | None = None,
    stage_name: str = "data_acquisition",
) -> HookEmissionResult:
    """Hook: funnel table + RUWE/period/ecc histograms + sky map (data_acquisition)."""
    diag = config.diagnostics
    if not diag.hooks.funnel_sky:
        return HookEmissionResult(
            hook_name="funnel_sky",
            skipped_reason="diagnostics.hooks.funnel_sky=false",
        )
    result = HookEmissionResult(hook_name="funnel_sky")
    dpi = diag.figure_dpi

    if diag.write_reports:
        report = write_report(
            dirs.reports / f"{stage_name}_funnel.txt",
            format_funnel_report(
                funnel_counts,
                quality_cut_bin_counts=quality_cut_bin_counts,
                stage_name=stage_name,
            ),
        )
        result.reports.append(report)

    if diag.write_figures:
        for name, values, xlabel in (
            ("ruwe", ruwe, "RUWE"),
            ("period_day", period_day, "period [day]"),
            ("eccentricity", eccentricity, "eccentricity"),
        ):
            path = _maybe_plot(
                True,
                lambda values=values, name=name, xlabel=xlabel: plot_histogram(
                    values,
                    dirs.figures / f"{name}.png",
                    xlabel=xlabel,
                    title=name,
                    dpi=dpi,
                ),
            )
            if path is not None:
                result.figures.append(path)
        sky = _maybe_plot(
            True,
            lambda: plot_sky_mollweide(
                ra_deg,
                dec_deg,
                dirs.figures / "sky_map.png",
                title="sky coverage",
                dpi=dpi,
            ),
        )
        if sky is not None:
            result.figures.append(sky)
        if funnel_counts:
            labels = list(funnel_counts.keys())
            values = [float(funnel_counts[k]) for k in labels]
            bars = _maybe_plot(
                True,
                lambda: plot_categorical_bars(
                    labels,
                    values,
                    dirs.figures / "funnel_bars.png",
                    xlabel="step",
                    ylabel="count",
                    title=f"{stage_name} funnel",
                    dpi=dpi,
                ),
            )
            if bars is not None:
                result.figures.append(bars)

    return result


def emit_elbadry_six_panel(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    panels: Mapping[str, Mapping[str, Sequence[float] | NDArray[np.floating]]],
    panel_order: Sequence[str] = DEFAULT_ELBADRY_PANEL_ORDER,
    title: str = "El-Badry-style six-panel comparison",
) -> HookEmissionResult:
    """Hook: El-Badry-style six-panel overlay figure + length report (SF validation)."""
    diag = config.diagnostics
    if not diag.hooks.elbadry_six_panel:
        return HookEmissionResult(
            hook_name="elbadry_six_panel",
            skipped_reason="diagnostics.hooks.elbadry_six_panel=false",
        )
    result = HookEmissionResult(hook_name="elbadry_six_panel")
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "elbadry_six_panel.txt",
                format_elbadry_panel_report(
                    panels, panel_order=panel_order, title=title
                ),
            )
        )
    if diag.write_figures:
        grid = _maybe_plot(
            True,
            lambda: plot_six_panel_grid(
                panels,
                dirs.figures / "elbadry_six_panel.png",
                panel_order=panel_order,
                dpi=diag.figure_dpi,
                title=title,
            ),
        )
        if grid is not None:
            result.figures.append(grid)
        # Also emit first panel as a standalone overlay when present.
        if panel_order:
            first = panel_order[0]
            series = panels.get(first, {})
            if series:
                single = _maybe_plot(
                    True,
                    lambda: plot_overlay_histograms(
                        series,
                        dirs.figures / f"elbadry_panel_{first}.png",
                        xlabel=first,
                        title=first,
                        dpi=diag.figure_dpi,
                    ),
                )
                if single is not None:
                    result.figures.append(single)
    return result


def emit_fit_tier_coverage(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    counts: Mapping[str, int],
) -> HookEmissionResult:
    """Hook stub: fit-tier coverage map (bulk_estimate vs full_uberMS)."""
    diag = config.diagnostics
    if not diag.hooks.fit_tier_coverage:
        return HookEmissionResult(
            hook_name="fit_tier_coverage",
            skipped_reason="diagnostics.hooks.fit_tier_coverage=false",
        )
    result = HookEmissionResult(hook_name="fit_tier_coverage")
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "fit_tier_coverage.txt",
                format_fit_tier_coverage_report(counts),
            )
        )
    if diag.write_figures and counts:
        labels = list(counts.keys())
        values = [float(counts[k]) for k in labels]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                values,
                dirs.figures / "fit_tier_coverage.png",
                xlabel="fit_tier",
                ylabel="count",
                title="fit-tier coverage",
                dpi=diag.figure_dpi,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_gate_pass_rate(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    counts: Mapping[str, int],
    gate_name: str = "rv_astrometry_gate",
) -> HookEmissionResult:
    """Hook stub: RV/astrometry gate pass-rate bars (threshold stays in stage config)."""
    diag = config.diagnostics
    if not diag.hooks.gate_pass_rate:
        return HookEmissionResult(
            hook_name="gate_pass_rate",
            skipped_reason="diagnostics.hooks.gate_pass_rate=false",
        )
    result = HookEmissionResult(hook_name="gate_pass_rate")
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / f"{gate_name}_pass_rate.txt",
                format_gate_pass_rate_report(counts, gate_name=gate_name),
            )
        )
    if diag.write_figures and counts:
        labels = list(counts.keys())
        values = [float(counts[k]) for k in labels]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                values,
                dirs.figures / f"{gate_name}_pass_rate.png",
                xlabel="outcome",
                ylabel="count",
                title=f"{gate_name} outcomes",
                dpi=diag.figure_dpi,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_known_truth_benchmarks(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    observed: Mapping[int, Any] | Sequence[Any] | None = None,
) -> HookEmissionResult:
    """Hook: Gaia BH known-truth expectation report (fixture/config-driven)."""
    diag = config.diagnostics
    if not diag.hooks.known_truth_benchmarks:
        return HookEmissionResult(
            hook_name="known_truth_benchmarks",
            skipped_reason="diagnostics.hooks.known_truth_benchmarks=false",
        )
    validate_benchmarks_config(config)
    table = load_known_truth_table_from_config(config)
    obs = observed if observed is not None else synthetic_observed_from_truth(table)
    results = check_known_truth_expectations(
        table,
        obs,
        ruwe_match_tolerance=float(config.benchmarks.ruwe_match_tolerance),
    )
    result = HookEmissionResult(hook_name="known_truth_benchmarks")
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "known_truth_gaia_bh.txt",
                format_known_truth_report(
                    table,
                    results,
                    ruwe_match_tolerance=float(config.benchmarks.ruwe_match_tolerance),
                ),
            )
        )
    return result


def emit_comparison_catalogs(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
) -> HookEmissionResult:
    """Hook: comparison-only catalog loaders + caveated report (never priors)."""
    diag = config.diagnostics
    if not diag.hooks.comparison_catalogs:
        return HookEmissionResult(
            hook_name="comparison_catalogs",
            skipped_reason="diagnostics.hooks.comparison_catalogs=false",
        )
    validate_benchmarks_config(config)
    catalogs = load_all_comparison_catalogs(config)
    assert_required_catalogs_present(catalogs)
    result = HookEmissionResult(hook_name="comparison_catalogs")
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "comparison_catalogs.txt",
                format_comparison_catalog_report(catalogs),
            )
        )
    return result


def _ensure_builtin_helpers_registered() -> None:
    """Idempotently register the infrastructure hook helpers."""
    builtins: dict[str, DiagnosticHelper] = {
        "emit_funnel_sky": emit_funnel_sky,
        "emit_elbadry_six_panel": emit_elbadry_six_panel,
        "emit_fit_tier_coverage": emit_fit_tier_coverage,
        "emit_gate_pass_rate": emit_gate_pass_rate,
        "emit_known_truth_benchmarks": emit_known_truth_benchmarks,
        "emit_comparison_catalogs": emit_comparison_catalogs,
        "format_funnel_report": format_funnel_report,
        "format_elbadry_panel_report": format_elbadry_panel_report,
        "format_fit_tier_coverage_report": format_fit_tier_coverage_report,
        "format_gate_pass_rate_report": format_gate_pass_rate_report,
        "format_known_truth_report": format_known_truth_report,
        "format_comparison_catalog_report": format_comparison_catalog_report,
    }
    for name, fn in builtins.items():
        if name not in _HELPER_REGISTRY:
            register_diagnostic_helper(name, fn)


def run_diagnostics_scaffolding(
    config: PipelineConfig,
    *,
    run_id: str,
    demo_hooks: bool = True,
) -> DiagnosticsStageResult:
    """Build directories, register helpers, optionally emit stub + benchmark hooks.

    When ``demo_hooks`` is True, emits empty-count fit-tier/gate stubs plus
    known-truth and comparison-catalog reports from fixtures.
    """
    _ensure_builtin_helpers_registered()
    dirs = resolve_diagnostic_dirs(config, run_id=run_id)
    hooks: list[HookEmissionResult] = []
    if demo_hooks:
        hooks.append(
            emit_fit_tier_coverage(
                config,
                dirs,
                counts={
                    FitTier.BULK_ESTIMATE.value: 0,
                    FitTier.FULL_UBERMS.value: 0,
                },
            )
        )
        hooks.append(
            emit_gate_pass_rate(
                config,
                dirs,
                counts={"passed": 0, "failed": 0, "skipped": 0},
            )
        )
        hooks.append(emit_known_truth_benchmarks(config, dirs))
        hooks.append(emit_comparison_catalogs(config, dirs))
    return DiagnosticsStageResult(
        schema_version=1,
        dirs=dirs,
        hooks_run=hooks,
        helpers_registered=list_diagnostic_helpers(),
        matplotlib_available=matplotlib_available(),
        config_snapshot={
            "diagnostics": config.diagnostics.model_dump(mode="json"),
            "benchmarks": config.benchmarks.model_dump(mode="json"),
        },
    )


def write_diagnostics_artifact(path: Path, result: DiagnosticsStageResult) -> None:
    """Persist scaffolding metadata to the stage HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "diagnostics"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["matplotlib_available"] = result.matplotlib_available
        handle.attrs["root"] = str(result.dirs.root)
        handle.create_dataset(
            "payload_json",
            data=np.array(
                json.dumps(payload, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        helpers = handle.create_group("helpers")
        helpers.create_dataset(
            "registered",
            data=np.array(
                list(result.helpers_registered),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        hooks = handle.create_group("hooks")
        for emission in result.hooks_run:
            g = hooks.create_group(emission.hook_name)
            if emission.skipped_reason is not None:
                g.attrs["skipped_reason"] = emission.skipped_reason
            g.create_dataset(
                "figures",
                data=np.array(
                    [str(p) for p in emission.figures],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            g.create_dataset(
                "reports",
                data=np.array(
                    [str(p) for p in emission.reports],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )


def read_diagnostics_artifact(path: Path) -> dict[str, Any]:
    """Load the scaffolding payload from a diagnostics stage HDF5."""
    with h5py.File(path, "r") as handle:
        raw = handle["payload_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        return json.loads(text)


def format_diagnostics_stage_report(result: DiagnosticsStageResult) -> str:
    """Fully legible diagnostics-stage summary (exempt from caveman compression)."""
    lines = [
        "=== diagnostics stage ===",
        f"schema_version: {result.schema_version}",
        f"root: {result.dirs.root}",
        f"figures_dir: {result.dirs.figures}",
        f"reports_dir: {result.dirs.reports}",
        f"matplotlib_available: {result.matplotlib_available}",
        f"figure_dpi: {(result.config_snapshot.get('diagnostics') or result.config_snapshot).get('figure_dpi')}",
        f"helpers_registered ({len(result.helpers_registered)}):",
    ]
    for name in result.helpers_registered:
        lines.append(f"  - {name}")
    lines.append("hooks_run:")
    if not result.hooks_run:
        lines.append("  (none)")
    for emission in result.hooks_run:
        if emission.skipped_reason:
            lines.append(f"  {emission.hook_name}: skipped ({emission.skipped_reason})")
        else:
            lines.append(
                f"  {emission.hook_name}: "
                f"figures={len(emission.figures)} reports={len(emission.reports)}"
            )
            for path in emission.reports:
                lines.append(f"    report: {path}")
            for path in emission.figures:
                lines.append(f"    figure: {path}")
    lines.append(
        "scope_note: known-truth Gaia BH benchmarks and comparison-only catalogs "
        "are fixture-driven (issue #70). SBC recovery is issue #69."
    )
    lines.append("=== end diagnostics stage ===")
    return "\n".join(lines)


def run_diagnostics_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    demo_hooks: bool = True,
) -> RunManifest:
    """Execute the ``diagnostics`` scaffolding stage and update the run manifest."""
    spec = STAGE_REGISTRY["diagnostics"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_diagnostics_scaffolding(
        config, run_id=manifest.run_id, demo_hooks=demo_hooks
    )
    write_diagnostics_artifact(artifact, result)
    write_report(
        result.dirs.reports / "diagnostics_stage.txt",
        format_diagnostics_stage_report(result),
    )

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest


# Register builtins at import so list_diagnostic_helpers() is useful immediately.
_ensure_builtin_helpers_registered()
