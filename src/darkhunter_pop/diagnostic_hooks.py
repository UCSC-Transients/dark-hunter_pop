"""Low-level diagnostic hook primitives shared by pipeline stages (#182).

Early stages (``data_acquisition``, ``rv_consistency``) emit stage-local diagnostic
reports and figures. They used to import these hooks from the ``diagnostics``
*stage* module, which is stage 14 and at module scope reaches most of the package,
so every early stage's real transitive dependency set degenerated to "everything"
(a layering inversion that made ``STAGE_REGISTRY.dependency_modules`` meaningless).

This module holds only what those early-stage call sites need, and depends only on
cross-cutting infrastructure (``config_loader``, ``config_schema``, ``plotting``):

- output-directory resolution and report writing (``DiagnosticDirs``,
  ``resolve_diagnostic_dirs``, ``write_report``, ``HookEmissionResult``);
- the ``funnel_sky`` hook (``data_acquisition``) and ``gate_pass_rate`` hook
  (``rv_astrometry_gate``);
- the NSS comparison-panel / solution-type label vocabularies shared by
  ``data_acquisition`` (real NSS panels), ``forward_model`` (mock panels) and
  ``diagnostics``.

``diagnostics`` and ``forward_model`` re-export every name defined here, so
existing imports keep working unchanged. It must never import a stage module.
Diagnostic reports and plot captions stay full-detail (caveman exemption).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.plotting import (
    MatplotlibUnavailableError,
    axis_label,
    plot_categorical_bars,
    plot_histogram,
    plot_sky_mollweide,
)

# Six El-Badry et al. (2024) comparison panels (ARCHITECTURE.md §4).
SIX_PANEL_NAMES: tuple[str, ...] = (
    "P_orb_days",
    "G_mag",
    "inv_parallax_mas_inv",
    "eccentricity",
    "f_m_msun",
    "cos_inclination",
)

SOLUTION_TYPE_LABELS: tuple[str, ...] = (
    "insufficient_visibility",
    "five_parameter",
    "seven_parameter",
    "nine_parameter",
    "twelve_parameter_orbital",
    "orbital_failed_cuts",
)


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
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook_name": self.hook_name,
            "figures": [str(p) for p in self.figures],
            "reports": [str(p) for p in self.reports],
            "skipped_reason": self.skipped_reason,
            "payload": self.payload,
        }


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


def format_gate_pass_rate_report(
    counts: Mapping[str, int],
    *,
    gate_name: str = "rv_astrometry_gate",
    chi2_dof_values: Sequence[float] | None = None,
    chi2_dof_threshold: float | None = None,
) -> str:
    """Full-detail gate pass/fail report (threshold values stay in stage config)."""
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
    ]
    if chi2_dof_threshold is not None:
        lines.append(
            f"  chi2_dof_threshold (from rv_consistency config): {chi2_dof_threshold}"
        )
    if chi2_dof_values:
        arr = np.asarray(list(chi2_dof_values), dtype=np.float64)
        lines.append(
            f"  chi2_dof: n={arr.size} median={float(np.median(arr)):.4f} "
            f"p90={float(np.percentile(arr, 90)):.4f} "
            f"max={float(np.max(arr)):.4f}"
        )
    lines.append(
        "  note: chi2/dof threshold is config-owned by rv_astrometry_gate; "
        "this hook records pass/fail counts and optional chi2/dof distribution."
    )
    lines.append(f"=== end {gate_name} pass-rate diagnostic ===")
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
        max_bins = int(diag.histogram_max_bins)
        for name, values, title in (
            ("ruwe", ruwe, f"{stage_name}: RUWE"),
            ("period_day", period_day, f"{stage_name}: orbital period"),
            ("eccentricity", eccentricity, f"{stage_name}: eccentricity"),
        ):
            path = _maybe_plot(
                True,
                lambda values=values, name=name, title=title: plot_histogram(
                    values,
                    dirs.figures / f"{name}.png",
                    xlabel=axis_label(name),
                    title=title,
                    dpi=dpi,
                    max_bins=max_bins,
                    style=config.plotting,
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
                point_size=float(diag.sky_map_point_size),
                alpha=float(diag.sky_map_alpha),
                style=config.plotting,
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
                    xlabel="funnel step",
                    ylabel="count",
                    title=f"{stage_name} funnel",
                    dpi=dpi,
                    style=config.plotting,
                ),
            )
            if bars is not None:
                result.figures.append(bars)

    return result


def emit_gate_pass_rate(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    counts: Mapping[str, int],
    gate_name: str = "rv_astrometry_gate",
    chi2_dof_values: Sequence[float] | None = None,
    chi2_dof_threshold: float | None = None,
) -> HookEmissionResult:
    """Hook: RV/astrometry gate pass-rate + optional chi2/dof distribution."""
    diag = config.diagnostics
    if not diag.hooks.gate_pass_rate:
        return HookEmissionResult(
            hook_name="gate_pass_rate",
            skipped_reason="diagnostics.hooks.gate_pass_rate=false",
        )
    threshold = chi2_dof_threshold
    if threshold is None:
        threshold = float(config.rv_consistency.chi2_dof_threshold)
    result = HookEmissionResult(
        hook_name="gate_pass_rate",
        payload={
            "counts": {k: int(v) for k, v in counts.items()},
            "chi2_dof_threshold": threshold,
            "n_chi2_dof": len(chi2_dof_values) if chi2_dof_values else 0,
        },
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / f"{gate_name}_pass_rate.txt",
                format_gate_pass_rate_report(
                    counts,
                    gate_name=gate_name,
                    chi2_dof_values=chi2_dof_values,
                    chi2_dof_threshold=threshold,
                ),
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
                style=config.plotting,
            ),
        )
        if path is not None:
            result.figures.append(path)
        if chi2_dof_values:
            hist = _maybe_plot(
                True,
                lambda: plot_histogram(
                    chi2_dof_values,
                    dirs.figures / f"{gate_name}_chi2_dof.png",
                    xlabel=axis_label("chi2_dof"),
                    title=f"{gate_name}: $\\chi^2$ per degree of freedom",
                    dpi=diag.figure_dpi,
                    max_bins=int(diag.histogram_max_bins),
                    style=config.plotting,
                ),
            )
            if hist is not None:
                result.figures.append(hist)
    return result
