"""Stage: ``diagnostics`` — required ARCHITECTURE.md §4 diagnostic suite.

``plotting.py`` provides shared rendering primitives; this module decides what to
check and emits full-detail reports + figures. Phase 2 owned the scaffolding
hooks (funnel/sky, El-Badry six-panel, fit-tier, gate pass-rate). Phase 6 (#71)
completes the remaining required diagnostics:

- fit-tier coverage map
- age-stratified WD-debiasing check (companion_nature age diagnostic)
- with/without-flagged-triples robustness (stub-safe when ``triples.enabled=false``)
- information-gain / follow-up-priority report (per-system + population)
- sampler multi-run consistency (inference robustness protocol)
- mock-injection Poisson-negligibility convergence plot
- gaiamock solution-type-fraction validation wrapper
- RV chi2/dof gate pass-rate diagnostic

Known-truth / comparison catalogs (#70) are also emitted here when enabled.
SBC recovery (#69) is wired here via ``emit_sbc_recovery`` / ``diagnostics.sbc``.
Diagnostic reports and plot captions stay full-detail (caveman exemption).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from darkhunter_pop.config_schema import PipelineConfig, SBCConfig
from darkhunter_pop.forward_model import (
    SOLUTION_TYPE_LABELS,
    SolutionTypeFractionResult,
)
from darkhunter_pop.plotting import (
    MatplotlibUnavailableError,
    matplotlib_available,
    plot_categorical_bars,
    plot_grouped_bars,
    plot_histogram,
    plot_line_with_threshold,
    plot_overlay_histograms,
    plot_six_panel_grid,
    plot_sky_mollweide,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    TRIPLES_DISABLED_SKIP_REASON,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.sbc import (
    format_sbc_report,
    read_sbc_artifact,
    run_sbc_suite,
    write_sbc_artifact,
)
from darkhunter_pop.schemas import CandidateRecord, FitTier, RunManifest, StageStatus
from darkhunter_pop.sensitivity_analysis import (
    MCNoiseConvergenceDiagnostic,
    run_mc_noise_convergence,
)

# Circular-import note: ``data_acquisition`` imports diagnostics emitters at module
# load, and ``companion_nature`` / ``mass_derivation`` sit on that import path.
# Those two modules are therefore imported only inside the helpers that need them.

# Default El-Badry et al. (2024) panel axis names (ARCHITECTURE.md §4 / forward_model).
DEFAULT_ELBADRY_PANEL_ORDER: tuple[str, ...] = (
    "P_orb_days",
    "G_mag",
    "inv_parallax_mas_inv",
    "eccentricity",
    "f_m_msun",
    "cos_inclination",
)

# Axis labels with units (docs/PLOTS.md); keys match ``DEFAULT_ELBADRY_PANEL_ORDER``.
ELBADRY_PANEL_XLABELS: dict[str, str] = {
    "P_orb_days": "orbital period (day)",
    "G_mag": r"$G$ magnitude (mag)",
    "inv_parallax_mas_inv": r"inverse parallax (mas$^{-1}$)",
    "eccentricity": "eccentricity",
    "f_m_msun": r"companion mass fraction ($M_\odot$)",
    "cos_inclination": r"$\cos i$",
}

DiagnosticHelper = Callable[..., Any]
DIAGNOSTICS_SCHEMA_VERSION = 2


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


@dataclass
class SamplerConsistencyResult:
    """Agreement summary across independent nested-sampling robustness runs."""

    n_runs: int
    logz_values: list[float]
    logz_errs: list[float]
    max_abs_logz_delta: float
    max_abs_logz_sigma: float
    consistent: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "logz_values": list(self.logz_values),
            "logz_errs": list(self.logz_errs),
            "max_abs_logz_delta": self.max_abs_logz_delta,
            "max_abs_logz_sigma": self.max_abs_logz_sigma,
            "consistent": self.consistent,
            "message": self.message,
        }


@dataclass
class InfoGainSystemRow:
    """One system's information-gain / follow-up priority entry."""

    source_id: int
    score: float
    rank: int
    fit_tier: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "score": self.score,
            "rank": self.rank,
            "fit_tier": self.fit_tier,
        }


@dataclass
class DiagnosticsStageResult:
    """Full diagnostic-suite stage output (ARCHITECTURE.md §4 + #69 / #70 / #71)."""

    schema_version: int
    dirs: DiagnosticDirs
    hooks_run: list[HookEmissionResult]
    helpers_registered: tuple[str, ...]
    matplotlib_available: bool
    config_snapshot: dict[str, Any]
    sbc_payload: dict[str, Any] | None = None

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
            "sbc_payload": self.sbc_payload,
            "notes": (
                "Phase 6 diagnostic suite (#71): fit-tier coverage, age-stratified "
                "WD check, triples robustness (stub-safe), info-gain/follow-up "
                "priority, sampler multi-run consistency, MC Poisson-negligibility "
                "convergence, gaiamock solution-type fractions, RV gate pass-rate. "
                "Also emits known-truth Gaia BH benchmarks and comparison-only "
                "catalog reports (issue #70). Optional SBC recovery is issue #69."
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


def count_fit_tiers(candidates: Sequence[CandidateRecord]) -> dict[str, int]:
    """Count candidates by ``FitTier`` (unknown / missing → ``unset``)."""
    counts: dict[str, int] = {tier.value: 0 for tier in FitTier}
    counts["unset"] = 0
    for cand in candidates:
        if cand.fit_tier is None:
            counts["unset"] += 1
        else:
            key = (
                cand.fit_tier.value
                if isinstance(cand.fit_tier, FitTier)
                else str(cand.fit_tier)
            )
            counts[key] = counts.get(key, 0) + 1
    return counts


def assess_sampler_consistency(
    sampler_runs: Sequence[Mapping[str, Any]],
    *,
    logz_sigma_tol: float,
) -> SamplerConsistencyResult:
    """Compare independent dynesty robustness runs (not bitwise seed identity)."""
    n = len(sampler_runs)
    logz = [float(r.get("logz", float("nan"))) for r in sampler_runs]
    logz_err = [float(r.get("logz_err", float("nan"))) for r in sampler_runs]
    if n == 0:
        return SamplerConsistencyResult(
            n_runs=0,
            logz_values=[],
            logz_errs=[],
            max_abs_logz_delta=float("nan"),
            max_abs_logz_sigma=float("nan"),
            consistent=False,
            message="No sampler runs provided; cannot assess multi-run consistency.",
        )
    if n == 1:
        return SamplerConsistencyResult(
            n_runs=1,
            logz_values=logz,
            logz_errs=logz_err,
            max_abs_logz_delta=0.0,
            max_abs_logz_sigma=0.0,
            consistent=True,
            message=(
                "Single sampler run only (CI smoke / n_robustness_runs=1). "
                "Science runs require inference.n_robustness_runs >= 3."
            ),
        )
    max_delta = 0.0
    max_sigma = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(logz[i] - logz[j])
            err_i = logz_err[i] if math.isfinite(logz_err[i]) else 0.0
            err_j = logz_err[j] if math.isfinite(logz_err[j]) else 0.0
            err = math.hypot(err_i, err_j)
            sigma = delta / err if err > 0 else (0.0 if delta == 0 else float("inf"))
            max_delta = max(max_delta, delta)
            max_sigma = max(max_sigma, sigma)
    ok = max_sigma <= float(logz_sigma_tol)
    msg = (
        f"Sampler multi-run consistency: n_runs={n}, "
        f"max_|ΔlogZ|={max_delta:.4g}, max_|ΔlogZ|/σ={max_sigma:.4g} "
        f"(tol={logz_sigma_tol:.4g}); "
        f"{'CONSISTENT' if ok else 'INCONSISTENT'}."
    )
    return SamplerConsistencyResult(
        n_runs=n,
        logz_values=logz,
        logz_errs=logz_err,
        max_abs_logz_delta=max_delta,
        max_abs_logz_sigma=max_sigma,
        consistent=ok,
        message=msg,
    )


def rank_information_gain(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> list[InfoGainSystemRow]:
    """Rank systems by information-gain score (higher = higher follow-up priority)."""
    # Deferred import: see module-level circular-import note.
    from darkhunter_pop.mass_derivation import information_gain_stub

    scored: list[tuple[float, CandidateRecord]] = []
    for cand in candidates:
        scored.append((information_gain_stub(cand, config), cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    rows: list[InfoGainSystemRow] = []
    for rank, (score, cand) in enumerate(scored, start=1):
        tier = cand.fit_tier.value if cand.fit_tier is not None else None
        rows.append(
            InfoGainSystemRow(
                source_id=int(cand.source_id),
                score=float(score),
                rank=rank,
                fit_tier=tier,
            )
        )
    return rows


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
    known = {t.value for t in FitTier} | {t.name for t in FitTier}
    extras = set(counts) - known
    for key in sorted(extras):
        n = int(counts[key])
        frac = (n / total) if total else 0.0
        lines.append(f"  {key}: count={n} fraction={frac:.4f}")
    lines.append("=== end fit-tier coverage ===")
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


def format_age_stratified_wd_report(diagnostic: Any) -> str:
    """Full-detail age-stratified WD-debiasing / age-independence report.

    ``diagnostic`` is a ``companion_nature.AgeBinDiagnostic`` (typed as Any to
    avoid a circular import with ``data_acquisition`` → diagnostics).
    """
    lines = [
        "=== age-stratified WD-debiasing check ===",
        diagnostic.message,
        f"  age_independence_ok: {diagnostic.age_independence_ok}",
        f"  max_abs_mean_weight_delta: {diagnostic.max_abs_mean_weight_delta:.6g}",
        f"  bin_edges_gyr: {list(diagnostic.bin_edges_gyr)}",
        f"  bin_counts: {list(diagnostic.bin_counts)}",
        "  global_mean_weights:",
    ]
    for key, value in diagnostic.global_mean_weights.items():
        lines.append(f"    {key}: {value:.6g}")
    lines.append("  mean_weights_by_bin (WD emphasis):")
    for i, mean in enumerate(diagnostic.mean_weights_by_bin):
        wd = mean.get("WD", float("nan"))
        lines.append(
            f"    bin[{i}] n={diagnostic.bin_counts[i]} WD={wd:.6g} full={mean}"
        )
    lines.append("=== end age-stratified WD-debiasing check ===")
    return "\n".join(lines)


def format_triples_robustness_report(
    *,
    triples_enabled: bool,
    n_flagged: int,
    with_flags: Mapping[str, float] | None,
    without_flags: Mapping[str, float] | None,
    skip_reason: str | None = None,
) -> str:
    """Full-detail with/without-flagged-triples robustness report."""
    lines = [
        "=== triples on/off robustness ===",
        f"  triples.enabled: {triples_enabled}",
        f"  n_flagged_triples: {n_flagged}",
    ]
    if not triples_enabled:
        reason = skip_reason or TRIPLES_DISABLED_SKIP_REASON
        lines.append(f"  status: stub-safe skip ({reason})")
        lines.append(
            "  note: v1 forces P(triple)=0 in population_model. When triples is "
            "enabled later, this hook compares population metrics with vs without "
            "flagged outer companions."
        )
    else:
        lines.append("  status: comparison active")
        lines.append("  metrics_with_flagged_included:")
        for key, value in (with_flags or {}).items():
            lines.append(f"    {key}: {value}")
        lines.append("  metrics_with_flagged_excluded:")
        for key, value in (without_flags or {}).items():
            lines.append(f"    {key}: {value}")
        if with_flags and without_flags:
            lines.append("  absolute_deltas:")
            for key in sorted(set(with_flags) | set(without_flags)):
                a = float(with_flags.get(key, float("nan")))
                b = float(without_flags.get(key, float("nan")))
                lines.append(f"    {key}: {abs(a - b):.6g}")
    lines.append("=== end triples on/off robustness ===")
    return "\n".join(lines)


def format_info_gain_report(
    rows: Sequence[InfoGainSystemRow],
    *,
    top_n: int,
) -> str:
    """Full-detail per-system + population information-gain / follow-up report."""
    scores = np.asarray([r.score for r in rows], dtype=np.float64)
    lines = [
        "=== information-gain / follow-up priority ===",
        f"  n_systems: {len(rows)}",
        f"  top_n_listed: {top_n}",
    ]
    if scores.size:
        lines.append(
            f"  score_summary: min={float(np.min(scores)):.6g} "
            f"median={float(np.median(scores)):.6g} "
            f"max={float(np.max(scores)):.6g} "
            f"mean={float(np.mean(scores)):.6g}"
        )
        lines.append(
            "  population_note: higher score = larger bulk M1 relative uncertainty "
            "(priority for refined uberMS / RV follow-up queue)."
        )
    else:
        lines.append("  score_summary: (no systems)")
    lines.append("  top_priority_systems:")
    for row in list(rows)[: max(0, top_n)]:
        lines.append(
            f"    rank={row.rank} source_id={row.source_id} "
            f"score={row.score:.6g} fit_tier={row.fit_tier}"
        )
    lines.append("=== end information-gain / follow-up priority ===")
    return "\n".join(lines)


def format_sampler_consistency_report(result: SamplerConsistencyResult) -> str:
    """Full-detail sampler multi-run consistency report."""
    lines = [
        "=== sampler multi-run consistency ===",
        result.message,
        f"  n_runs: {result.n_runs}",
        f"  consistent: {result.consistent}",
        f"  max_abs_logz_delta: {result.max_abs_logz_delta}",
        f"  max_abs_logz_sigma: {result.max_abs_logz_sigma}",
        "  runs:",
    ]
    for i, (lz, err) in enumerate(zip(result.logz_values, result.logz_errs)):
        lines.append(f"    run[{i}]: logZ={lz} logZ_err={err}")
    lines.append(
        "  note: reproducibility is the multi-run robustness protocol "
        "(inference.ROBUSTNESS_PROTOCOL), not bitwise seed identity."
    )
    lines.append("=== end sampler multi-run consistency ===")
    return "\n".join(lines)


def format_mc_noise_convergence_report(diagnostic: MCNoiseConvergenceDiagnostic) -> str:
    """Full-detail mock-injection Poisson-negligibility convergence report."""
    lines = [
        "=== mock-injection Poisson-negligibility convergence ===",
        diagnostic.message,
        f"  threshold (physics.mc_noise_threshold): {diagnostic.threshold}",
        f"  n_mock_final: {diagnostic.n_mock_final}",
        f"  all_bins_passed: {diagnostic.all_bins_passed}",
        "  schedule (n_mock, max_ratio):",
    ]
    for n_mock, ratio in zip(diagnostic.schedule_n_mock, diagnostic.schedule_max_ratio):
        lines.append(f"    n_mock={n_mock} max_ratio={ratio:.6g}")
    if diagnostic.per_bin:
        lines.append("  per_bin at final n_mock:")
        for bin_row in diagnostic.per_bin:
            lines.append(
                f"    bin[{bin_row.bin_index}] mu={bin_row.expected_count:.6g} "
                f"ratio={bin_row.ratio:.6g} passed={bin_row.passed}"
            )
    lines.append("=== end mock-injection Poisson-negligibility convergence ===")
    return "\n".join(lines)


def format_solution_type_fraction_report(
    result: SolutionTypeFractionResult,
    *,
    max_abs_delta_config: float | None = None,
) -> str:
    """Full-detail gaiamock solution-type-fraction validation report."""
    lines = [
        "=== gaiamock solution-type-fraction validation ===",
        f"  passed: {result.passed}",
        f"  max_abs_delta: {result.max_abs_delta:.6g}",
    ]
    if max_abs_delta_config is not None:
        lines.append(
            "  config_max_abs_delta "
            f"(selection_function_astrometric.validation_gate): {max_abs_delta_config}"
        )
    lines.append("  fractions (mock vs real):")
    for label in SOLUTION_TYPE_LABELS:
        m = result.mock_fractions.get(label, 0.0)
        r = result.real_fractions.get(label, 0.0)
        lines.append(
            f"    {label}: mock={m:.6g} real={r:.6g} |delta|={abs(m - r):.6g}"
        )
    lines.append(
        "  note: this diagnostic wraps the existing SF validation-gate outputs; "
        "KS six-panel science thresholds remain in selection_function_astrometric."
    )
    lines.append("=== end gaiamock solution-type-fraction validation ===")
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
        for name, values, xlabel in (
            ("ruwe", ruwe, "RUWE"),
            ("period_day", period_day, "period (day)"),
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
                    xlabel="step",
                    ylabel="count",
                    title=f"{stage_name} funnel",
                    dpi=dpi,
                    style=config.plotting,
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
                panel_xlabels=ELBADRY_PANEL_XLABELS,
                dpi=diag.figure_dpi,
                max_bins=int(diag.histogram_max_bins),
                title=title,
                style=config.plotting,
            ),
        )
        if grid is not None:
            result.figures.append(grid)
        if panel_order:
            first = panel_order[0]
            series = panels.get(first, {})
            if series:
                single = _maybe_plot(
                    True,
                    lambda: plot_overlay_histograms(
                        series,
                        dirs.figures / f"elbadry_panel_{first}.png",
                        xlabel=ELBADRY_PANEL_XLABELS.get(first, first),
                        title=first,
                        dpi=diag.figure_dpi,
                        max_bins=int(diag.histogram_max_bins),
                        style=config.plotting,
                    ),
                )
                if single is not None:
                    result.figures.append(single)
    return result


def emit_fit_tier_coverage(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    counts: Mapping[str, int] | None = None,
    candidates: Sequence[CandidateRecord] | None = None,
) -> HookEmissionResult:
    """Hook: fit-tier coverage map (bulk_estimate vs full_uberMS)."""
    diag = config.diagnostics
    if not diag.hooks.fit_tier_coverage:
        return HookEmissionResult(
            hook_name="fit_tier_coverage",
            skipped_reason="diagnostics.hooks.fit_tier_coverage=false",
        )
    resolved = dict(counts) if counts is not None else {}
    if not resolved and candidates is not None:
        resolved = count_fit_tiers(candidates)
    result = HookEmissionResult(
        hook_name="fit_tier_coverage",
        payload={"counts": {k: int(v) for k, v in resolved.items()}},
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "fit_tier_coverage.txt",
                format_fit_tier_coverage_report(resolved),
            )
        )
    if diag.write_figures and resolved:
        labels = list(resolved.keys())
        values = [float(resolved[k]) for k in labels]
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
                style=config.plotting,
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
                    xlabel="chi2 per dof",
                    title=f"{gate_name} chi2/dof",
                    dpi=diag.figure_dpi,
                    max_bins=int(diag.histogram_max_bins),
                    style=config.plotting,
                ),
            )
            if hist is not None:
                result.figures.append(hist)
    return result


def emit_age_stratified_wd(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    candidates: Sequence[CandidateRecord] | None = None,
    age_diagnostic: Any | None = None,
) -> HookEmissionResult:
    """Hook: age-stratified WD-debiasing / age-independence check."""
    diag = config.diagnostics
    if not diag.hooks.age_stratified_wd:
        return HookEmissionResult(
            hook_name="age_stratified_wd",
            skipped_reason="diagnostics.hooks.age_stratified_wd=false",
        )
    # Deferred import: see module-level circular-import note.
    from darkhunter_pop.companion_nature import age_bin_diagnostic

    diagnostic = age_diagnostic
    if diagnostic is None:
        diagnostic = age_bin_diagnostic(
            candidates or (),
            config.companion_nature,
        )
    result = HookEmissionResult(
        hook_name="age_stratified_wd",
        payload=diagnostic.as_dict(),
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "age_stratified_wd.txt",
                format_age_stratified_wd_report(diagnostic),
            )
        )
    if diag.write_figures and diagnostic.bin_counts:
        labels = [
            f"[{diagnostic.bin_edges_gyr[i]},{diagnostic.bin_edges_gyr[i + 1]})"
            for i in range(len(diagnostic.bin_counts))
        ]
        wd_means = [
            float(mean.get("WD", float("nan")))
            for mean in diagnostic.mean_weights_by_bin
        ]
        # Replace NaN with 0 for plotting empty bins.
        plot_vals = [0.0 if not math.isfinite(v) else v for v in wd_means]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                plot_vals,
                dirs.figures / "age_stratified_wd_mean.png",
                xlabel="primary age bin [Gyr]",
                ylabel="mean WD weight",
                title="age-stratified WD weights",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_triples_robustness(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    flagged_source_ids: Sequence[int] | None = None,
    metrics_with_flagged: Mapping[str, float] | None = None,
    metrics_without_flagged: Mapping[str, float] | None = None,
) -> HookEmissionResult:
    """Hook: with/without-flagged-triples robustness (safe when triples disabled)."""
    diag = config.diagnostics
    if not diag.hooks.triples_robustness:
        return HookEmissionResult(
            hook_name="triples_robustness",
            skipped_reason="diagnostics.hooks.triples_robustness=false",
        )
    enabled = bool(config.triples.enabled)
    flagged = list(flagged_source_ids or ())
    payload: dict[str, Any] = {
        "triples_enabled": enabled,
        "n_flagged": len(flagged),
        "flagged_source_ids": [int(x) for x in flagged],
    }
    if not enabled:
        result = HookEmissionResult(
            hook_name="triples_robustness",
            payload=payload,
        )
        if diag.write_reports:
            result.reports.append(
                write_report(
                    dirs.reports / "triples_robustness.txt",
                    format_triples_robustness_report(
                        triples_enabled=False,
                        n_flagged=len(flagged),
                        with_flags=None,
                        without_flags=None,
                        skip_reason=TRIPLES_DISABLED_SKIP_REASON,
                    ),
                )
            )
        return result

    with_flags = dict(metrics_with_flagged or {})
    without_flags = dict(metrics_without_flagged or {})
    payload["metrics_with_flagged"] = with_flags
    payload["metrics_without_flagged"] = without_flags
    result = HookEmissionResult(hook_name="triples_robustness", payload=payload)
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "triples_robustness.txt",
                format_triples_robustness_report(
                    triples_enabled=True,
                    n_flagged=len(flagged),
                    with_flags=with_flags,
                    without_flags=without_flags,
                ),
            )
        )
    if diag.write_figures and with_flags and without_flags:
        labels = sorted(set(with_flags) | set(without_flags))
        series = {
            "with_flagged": [float(with_flags.get(k, 0.0)) for k in labels],
            "without_flagged": [float(without_flags.get(k, 0.0)) for k in labels],
        }
        path = _maybe_plot(
            True,
            lambda: plot_grouped_bars(
                labels,
                series,
                dirs.figures / "triples_robustness.png",
                xlabel="metric",
                ylabel="value",
                title="triples with/without flagged",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_info_gain_followup(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    candidates: Sequence[CandidateRecord] | None = None,
    rows: Sequence[InfoGainSystemRow] | None = None,
) -> HookEmissionResult:
    """Hook: information-gain / follow-up-priority report (per-system + population)."""
    diag = config.diagnostics
    if not diag.hooks.info_gain_followup:
        return HookEmissionResult(
            hook_name="info_gain_followup",
            skipped_reason="diagnostics.hooks.info_gain_followup=false",
        )
    ranked = list(rows) if rows is not None else rank_information_gain(
        candidates or (), config
    )
    top_n = int(diag.info_gain_top_n)
    result = HookEmissionResult(
        hook_name="info_gain_followup",
        payload={
            "n_systems": len(ranked),
            "top_n": top_n,
            "top_rows": [r.as_dict() for r in ranked[:top_n]],
        },
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "info_gain_followup.txt",
                format_info_gain_report(ranked, top_n=top_n),
            )
        )
    if diag.write_figures and ranked:
        scores = [r.score for r in ranked]
        hist = _maybe_plot(
            True,
            lambda: plot_histogram(
                scores,
                dirs.figures / "info_gain_scores.png",
                xlabel="information-gain score",
                title="follow-up priority score distribution",
                dpi=diag.figure_dpi,
                max_bins=int(diag.histogram_max_bins),
                style=config.plotting,
            ),
        )
        if hist is not None:
            result.figures.append(hist)
        top = ranked[: min(top_n, len(ranked))]
        bars = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                [str(r.source_id) for r in top],
                [r.score for r in top],
                dirs.figures / "info_gain_top_n.png",
                xlabel="source_id",
                ylabel="score",
                title=f"top-{len(top)} follow-up priority",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if bars is not None:
            result.figures.append(bars)
    return result


def emit_sampler_consistency(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    sampler_runs: Sequence[Mapping[str, Any]] | None = None,
) -> HookEmissionResult:
    """Hook: sampler multi-run consistency (inference robustness protocol)."""
    diag = config.diagnostics
    if not diag.hooks.sampler_consistency:
        return HookEmissionResult(
            hook_name="sampler_consistency",
            skipped_reason="diagnostics.hooks.sampler_consistency=false",
        )
    assessment = assess_sampler_consistency(
        sampler_runs or (),
        logz_sigma_tol=float(diag.sampler_logz_sigma_tol),
    )
    result = HookEmissionResult(
        hook_name="sampler_consistency",
        payload=assessment.as_dict(),
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "sampler_consistency.txt",
                format_sampler_consistency_report(assessment),
            )
        )
    if diag.write_figures and assessment.n_runs:
        labels = [f"run{i}" for i in range(assessment.n_runs)]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                assessment.logz_values,
                dirs.figures / "sampler_logz.png",
                xlabel="robustness run",
                ylabel="logZ",
                title="sampler multi-run logZ",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_mc_noise_convergence(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    diagnostic: MCNoiseConvergenceDiagnostic | None = None,
) -> HookEmissionResult:
    """Hook: mock-injection Poisson-negligibility convergence plot + report."""
    diag = config.diagnostics
    if not diag.hooks.mc_noise_convergence:
        return HookEmissionResult(
            hook_name="mc_noise_convergence",
            skipped_reason="diagnostics.hooks.mc_noise_convergence=false",
        )
    if diagnostic is None:
        return HookEmissionResult(
            hook_name="mc_noise_convergence",
            skipped_reason="no MCNoiseConvergenceDiagnostic provided",
        )
    result = HookEmissionResult(
        hook_name="mc_noise_convergence",
        payload=diagnostic.as_dict(),
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "mc_noise_convergence.txt",
                format_mc_noise_convergence_report(diagnostic),
            )
        )
    if diag.write_figures and diagnostic.schedule_n_mock:
        path = _maybe_plot(
            True,
            lambda: plot_line_with_threshold(
                diagnostic.schedule_n_mock,
                diagnostic.schedule_max_ratio,
                dirs.figures / "mc_noise_convergence.png",
                xlabel="n_mock",
                ylabel="max sigma_MC / sigma_Poisson",
                title="mock-injection Poisson-negligibility convergence",
                dpi=diag.figure_dpi,
                threshold=float(diagnostic.threshold),
                threshold_label=f"threshold={diagnostic.threshold}",
                log_x=True,
                style=config.plotting,
            ),
        )
        if path is not None:
            result.figures.append(path)
    return result


def emit_solution_type_fractions(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    result: SolutionTypeFractionResult | None = None,
) -> HookEmissionResult:
    """Hook: gaiamock solution-type-fraction validation (wraps SF gate outputs)."""
    diag = config.diagnostics
    if not diag.hooks.solution_type_fractions:
        return HookEmissionResult(
            hook_name="solution_type_fractions",
            skipped_reason="diagnostics.hooks.solution_type_fractions=false",
        )
    if result is None:
        return HookEmissionResult(
            hook_name="solution_type_fractions",
            skipped_reason="no SolutionTypeFractionResult provided",
        )
    max_delta_cfg = float(
        config.selection_function_astrometric.validation_gate.solution_type_fraction_max_abs_delta
    )
    emission = HookEmissionResult(
        hook_name="solution_type_fractions",
        payload={
            "passed": result.passed,
            "max_abs_delta": result.max_abs_delta,
            "mock_fractions": dict(result.mock_fractions),
            "real_fractions": dict(result.real_fractions),
            "config_max_abs_delta": max_delta_cfg,
        },
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "solution_type_fractions.txt",
                format_solution_type_fraction_report(
                    result, max_abs_delta_config=max_delta_cfg
                ),
            )
        )
    if diag.write_figures:
        labels = list(SOLUTION_TYPE_LABELS)
        series = {
            "mock": [float(result.mock_fractions.get(k, 0.0)) for k in labels],
            "real": [float(result.real_fractions.get(k, 0.0)) for k in labels],
        }
        path = _maybe_plot(
            True,
            lambda: plot_grouped_bars(
                labels,
                series,
                dirs.figures / "solution_type_fractions.png",
                xlabel="solution type",
                ylabel="fraction",
                title="gaiamock solution-type fractions",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            emission.figures.append(path)
    return emission


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



def emit_sbc_recovery(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    sbc: SBCConfig | None = None,
    n_repeats: int | None = None,
) -> HookEmissionResult:
    """Hook: simulation-based calibration recovery + coverage report (issue #69)."""
    diag = config.diagnostics
    if not diag.hooks.sbc_recovery:
        return HookEmissionResult(
            hook_name="sbc_recovery",
            skipped_reason="diagnostics.hooks.sbc_recovery=false",
        )
    cfg = sbc if sbc is not None else diag.sbc
    if not cfg.enabled:
        return HookEmissionResult(
            hook_name="sbc_recovery",
            skipped_reason="diagnostics.sbc.enabled=false",
        )
    result = run_sbc_suite(config, sbc=cfg, n_repeats=n_repeats)
    emission = HookEmissionResult(
        hook_name="sbc_recovery",
        payload={
            "overall_empirical_coverage": result.overall_empirical_coverage,
            "overall_passed": result.overall_passed,
            "n_records": len(result.records),
            "recovery_backend": result.recovery_backend,
        },
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(dirs.reports / "sbc_recovery.txt", format_sbc_report(result))
        )
        h5_path = dirs.reports / "sbc_recovery.h5"
        write_sbc_artifact(h5_path, result)
        emission.reports.append(h5_path)
    return emission


def _ensure_builtin_helpers_registered() -> None:
    """Idempotently register the infrastructure hook helpers."""
    builtins: dict[str, DiagnosticHelper] = {
        "emit_funnel_sky": emit_funnel_sky,
        "emit_elbadry_six_panel": emit_elbadry_six_panel,
        "emit_fit_tier_coverage": emit_fit_tier_coverage,
        "emit_gate_pass_rate": emit_gate_pass_rate,
        "emit_age_stratified_wd": emit_age_stratified_wd,
        "emit_triples_robustness": emit_triples_robustness,
        "emit_info_gain_followup": emit_info_gain_followup,
        "emit_sampler_consistency": emit_sampler_consistency,
        "emit_mc_noise_convergence": emit_mc_noise_convergence,
        "emit_solution_type_fractions": emit_solution_type_fractions,
        "emit_known_truth_benchmarks": emit_known_truth_benchmarks,
        "emit_comparison_catalogs": emit_comparison_catalogs,
        "emit_sbc_recovery": emit_sbc_recovery,
        "format_funnel_report": format_funnel_report,
        "format_elbadry_panel_report": format_elbadry_panel_report,
        "format_fit_tier_coverage_report": format_fit_tier_coverage_report,
        "format_gate_pass_rate_report": format_gate_pass_rate_report,
        "format_age_stratified_wd_report": format_age_stratified_wd_report,
        "format_triples_robustness_report": format_triples_robustness_report,
        "format_info_gain_report": format_info_gain_report,
        "format_sampler_consistency_report": format_sampler_consistency_report,
        "format_mc_noise_convergence_report": format_mc_noise_convergence_report,
        "format_solution_type_fraction_report": format_solution_type_fraction_report,
        "format_known_truth_report": format_known_truth_report,
        "format_comparison_catalog_report": format_comparison_catalog_report,
        "format_sbc_report": format_sbc_report,
        "count_fit_tiers": count_fit_tiers,
        "assess_sampler_consistency": assess_sampler_consistency,
        "rank_information_gain": rank_information_gain,
    }
    for name, fn in builtins.items():
        if name not in _HELPER_REGISTRY:
            register_diagnostic_helper(name, fn)


def run_diagnostic_suite(
    config: PipelineConfig,
    *,
    run_id: str,
    candidates: Sequence[CandidateRecord] | None = None,
    funnel_counts: Mapping[str, int] | None = None,
    elbadry_panels: Mapping[
        str, Mapping[str, Sequence[float] | NDArray[np.floating]]
    ]
    | None = None,
    gate_counts: Mapping[str, int] | None = None,
    chi2_dof_values: Sequence[float] | None = None,
    age_diagnostic: Any | None = None,
    flagged_triple_ids: Sequence[int] | None = None,
    triples_metrics_with: Mapping[str, float] | None = None,
    triples_metrics_without: Mapping[str, float] | None = None,
    sampler_runs: Sequence[Mapping[str, Any]] | None = None,
    mc_noise: MCNoiseConvergenceDiagnostic | None = None,
    solution_types: SolutionTypeFractionResult | None = None,
    demo_missing: bool = False,
    run_sbc: bool | None = None,
) -> DiagnosticsStageResult:
    """Run the full required diagnostic suite against provided stage outputs.

    When ``demo_missing`` is True, empty/synthetic stand-ins fill hooks that lack
    upstream data so the on-disk layout is exercised without science artifacts.
    Known-truth and comparison-catalog hooks also run from fixtures when enabled.
    """
    _ensure_builtin_helpers_registered()
    dirs = resolve_diagnostic_dirs(config, run_id=run_id)
    cand = list(candidates or ())
    hooks: list[HookEmissionResult] = []

    if funnel_counts is not None or demo_missing:
        hooks.append(
            emit_funnel_sky(
                config,
                dirs,
                funnel_counts=funnel_counts
                or {"queried": 0, "after_quality_cut": 0, "candidates_written": 0},
            )
        )
    if elbadry_panels is not None or demo_missing:
        panels = elbadry_panels
        if panels is None:
            panels = {
                name: {"mock": np.linspace(0.0, 1.0, 8), "real": np.linspace(0.1, 1.1, 8)}
                for name in DEFAULT_ELBADRY_PANEL_ORDER
            }
        hooks.append(emit_elbadry_six_panel(config, dirs, panels=panels))

    hooks.append(
        emit_fit_tier_coverage(
            config,
            dirs,
            counts=None if cand else (
                {
                    FitTier.BULK_ESTIMATE.value: 0,
                    FitTier.FULL_UBERMS.value: 0,
                    "unset": 0,
                }
                if demo_missing
                else None
            ),
            candidates=cand or None,
        )
    )
    hooks.append(
        emit_gate_pass_rate(
            config,
            dirs,
            counts=gate_counts
            or ({"passed": 0, "failed": 0, "skipped": 0} if demo_missing else {}),
            chi2_dof_values=chi2_dof_values,
        )
    )
    hooks.append(
        emit_age_stratified_wd(
            config, dirs, candidates=cand, age_diagnostic=age_diagnostic
        )
    )
    hooks.append(
        emit_triples_robustness(
            config,
            dirs,
            flagged_source_ids=flagged_triple_ids,
            metrics_with_flagged=triples_metrics_with,
            metrics_without_flagged=triples_metrics_without,
        )
    )
    hooks.append(emit_info_gain_followup(config, dirs, candidates=cand))
    hooks.append(
        emit_sampler_consistency(
            config,
            dirs,
            sampler_runs=sampler_runs
            or (
                [{"logz": 0.0, "logz_err": 0.1, "seed": 0, "nlive": 1}]
                if demo_missing
                else None
            ),
        )
    )
    if mc_noise is not None or demo_missing:
        mc = mc_noise
        if mc is None:
            mc = run_mc_noise_convergence(
                [10.0, 5.0, 2.0],
                threshold=float(config.physics.mc_noise_threshold),
                n_mock_start=10,
                n_mock_max=200,
                growth_factor=2.0,
            )
        hooks.append(emit_mc_noise_convergence(config, dirs, diagnostic=mc))
    else:
        hooks.append(emit_mc_noise_convergence(config, dirs, diagnostic=None))

    if solution_types is not None or demo_missing:
        st = solution_types
        if st is None:
            frac = {label: 1.0 / len(SOLUTION_TYPE_LABELS) for label in SOLUTION_TYPE_LABELS}
            st = SolutionTypeFractionResult(
                mock_fractions=dict(frac),
                real_fractions=dict(frac),
                max_abs_delta=0.0,
                passed=True,
            )
        hooks.append(emit_solution_type_fractions(config, dirs, result=st))
    else:
        hooks.append(emit_solution_type_fractions(config, dirs, result=None))

    hooks.append(emit_known_truth_benchmarks(config, dirs))
    hooks.append(emit_comparison_catalogs(config, dirs))
    do_sbc = (
        bool(config.diagnostics.sbc.run_in_stage)
        if run_sbc is None
        else bool(run_sbc)
    )
    sbc_payload: dict[str, Any] | None = None
    if do_sbc and config.diagnostics.hooks.sbc_recovery and config.diagnostics.sbc.enabled:
        emission = emit_sbc_recovery(config, dirs)
        hooks.append(emission)
        if emission.skipped_reason is None:
            for path in emission.reports:
                if path.suffix == ".h5" and path.is_file():
                    sbc_payload = read_sbc_artifact(path)
                    break

    return DiagnosticsStageResult(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        dirs=dirs,
        hooks_run=hooks,
        helpers_registered=list_diagnostic_helpers(),
        matplotlib_available=matplotlib_available(),
        config_snapshot={
            "diagnostics": config.diagnostics.model_dump(mode="json"),
            "benchmarks": config.benchmarks.model_dump(mode="json"),
        },
        sbc_payload=sbc_payload,
    )


def run_diagnostics_scaffolding(
    config: PipelineConfig,
    *,
    run_id: str,
    demo_hooks: bool = True,
    run_sbc: bool | None = None,
) -> DiagnosticsStageResult:
    """Build directories, register helpers, optionally emit demo suite outputs.

    Prefer ``run_diagnostic_suite`` when upstream stage payloads are available.
    SBC runs when ``run_sbc`` is True, or when ``run_sbc is None`` and
    ``diagnostics.sbc.run_in_stage`` is True (default False keeps the stage fast).
    """
    return run_diagnostic_suite(
        config, run_id=run_id, demo_missing=demo_hooks, run_sbc=run_sbc
    )


def write_diagnostics_artifact(path: Path, result: DiagnosticsStageResult) -> None:
    """Persist diagnostic-suite metadata to the stage HDF5."""
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
            g.create_dataset(
                "payload_json",
                data=np.array(
                    json.dumps(emission.payload, sort_keys=True),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )


def read_diagnostics_artifact(path: Path) -> dict[str, Any]:
    """Load the diagnostic-suite payload from a diagnostics stage HDF5."""
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
        "=== diagnostics stage (full suite) ===",
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
    diag_snap = result.config_snapshot.get("diagnostics") or result.config_snapshot
    sbc_cfg = diag_snap.get("sbc") or {}
    lines.append("sbc_config:")
    lines.append(f"  enabled: {sbc_cfg.get('enabled')}")
    lines.append(f"  run_in_stage: {sbc_cfg.get('run_in_stage')}")
    lines.append(f"  recovery_backend: {sbc_cfg.get('recovery_backend')}")
    lines.append(
        f"  credible_interval_level: {sbc_cfg.get('credible_interval_level')}"
    )
    lines.append(
        f"  coverage_abs_tolerance: {sbc_cfg.get('coverage_abs_tolerance')}"
    )
    if result.sbc_payload is None:
        lines.append("sbc_payload: (not run in this stage invocation)")
    else:
        lines.append("sbc_payload:")
        lines.append(
            f"  overall_empirical_coverage: "
            f"{result.sbc_payload.get('overall_empirical_coverage')}"
        )
        lines.append(f"  overall_passed: {result.sbc_payload.get('overall_passed')}")
        lines.append(f"  n_records: {result.sbc_payload.get('n_records')}")
    lines.append(
        "scope_note: this stage owns the required diagnostic list (#71) plus "
        "known-truth / comparison-catalog hooks (#70). SBC recovery (#69) is "
        "wired via diagnostics.sbc / emit_sbc_recovery."
    )
    lines.append("=== end diagnostics stage ===")
    return "\n".join(lines)


def _optional_artifact_path(manifest: RunManifest, stage_name: str) -> Path | None:
    record = manifest.stages.get(stage_name)
    if record is None or not record.artifact_path:
        return None
    path = Path(record.artifact_path)
    return path if path.is_file() else None


def _read_mc_noise_from_sensitivity(path: Path) -> MCNoiseConvergenceDiagnostic | None:
    from darkhunter_pop.sensitivity_analysis import BinMCNoiseResult

    with h5py.File(path, "r") as handle:
        mc = handle.get("mc_noise_convergence")
        if mc is None:
            return None
        n_mock_final = int(handle.attrs.get("n_mock_final", 0))
        per_bin: tuple[BinMCNoiseResult, ...] = ()
        if "bin_expected_count" in mc:
            expected = np.asarray(mc["bin_expected_count"], dtype=np.float64)
            ratios = np.asarray(mc["bin_ratio"], dtype=np.float64)
            passed = np.asarray(mc["bin_passed"], dtype=bool)
            per_bin = tuple(
                BinMCNoiseResult(
                    bin_index=i,
                    expected_count=float(expected[i]),
                    n_mock=n_mock_final,
                    sigma_mc=float("nan"),
                    sigma_poisson=float("nan"),
                    ratio=float(ratios[i]),
                    passed=bool(passed[i]),
                )
                for i in range(len(expected))
            )
        return MCNoiseConvergenceDiagnostic(
            threshold=float(mc.attrs.get("threshold", 0.0)),
            n_mock_final=n_mock_final,
            all_bins_passed=bool(mc.attrs.get("all_bins_passed", False)),
            per_bin=per_bin,
            schedule_n_mock=tuple(int(x) for x in np.asarray(mc["schedule_n_mock"])),
            schedule_max_ratio=tuple(float(x) for x in np.asarray(mc["schedule_max_ratio"])),
            message=str(mc.attrs.get("message", "")),
        )


def _read_solution_types_from_sf(path: Path) -> SolutionTypeFractionResult | None:
    with h5py.File(path, "r") as handle:
        st = handle.get("validation_gate/solution_type_fractions")
        if st is None:
            return None
        mock_frac = {
            label: float(st.attrs.get(f"mock_{label}", 0.0)) for label in SOLUTION_TYPE_LABELS
        }
        real_frac = {
            label: float(st.attrs.get(f"real_{label}", 0.0)) for label in SOLUTION_TYPE_LABELS
        }
        return SolutionTypeFractionResult(
            mock_fractions=mock_frac,
            real_fractions=real_frac,
            max_abs_delta=float(st.attrs.get("max_abs_delta", 0.0)),
            passed=bool(st.attrs.get("passed", False)),
        )


def _read_elbadry_panels_from_manifest(
    manifest: RunManifest,
    config: PipelineConfig,
) -> dict[str, Mapping[str, NDArray[np.floating]]] | None:
    da_path = _optional_artifact_path(manifest, "data_acquisition")
    if da_path is None:
        return None
    try:
        from darkhunter_pop.forward_model import (
            SIX_PANEL_NAMES,
            load_real_panels_from_data_acquisition,
            load_reference_panels,
        )

        real_panels, _ = load_real_panels_from_data_acquisition(da_path)
        mock_panels, _ = load_reference_panels(config)
        return {
            name: {"mock": mock_panels[name], "real": real_panels[name]}
            for name in SIX_PANEL_NAMES
            if name in real_panels and name in mock_panels
        }
    except (KeyError, ValueError, FileNotFoundError, OSError):
        return None


def _hydrate_diagnostics_from_manifest(
    manifest: RunManifest,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Load upstream stage artifacts for the diagnostic suite when not passed explicitly."""
    hydrated: dict[str, Any] = {}

    cn_path = _optional_artifact_path(manifest, "companion_nature_likelihood")
    if cn_path is not None:
        from darkhunter_pop.companion_nature import AgeBinDiagnostic, read_stage_hdf5

        candidates, meta = read_stage_hdf5(cn_path)
        hydrated["candidates"] = candidates
        age_raw = meta.get("diagnostics.age_diagnostic")
        if isinstance(age_raw, str):
            hydrated["age_diagnostic"] = AgeBinDiagnostic(**json.loads(age_raw))
        elif isinstance(age_raw, Mapping):
            hydrated["age_diagnostic"] = AgeBinDiagnostic(**dict(age_raw))

    da_path = _optional_artifact_path(manifest, "data_acquisition")
    if da_path is not None:
        with h5py.File(da_path, "r") as handle:
            if "diagnostics" in handle:
                funnel = {
                    str(key): int(handle["diagnostics"].attrs[key])
                    for key in handle["diagnostics"].attrs
                }
                if funnel:
                    hydrated["funnel_counts"] = funnel

    gate_path = _optional_artifact_path(manifest, "rv_astrometry_gate")
    if gate_path is not None:
        from darkhunter_pop.rv_consistency import read_stage_hdf5 as read_rv_hdf5

        _, meta = read_rv_hdf5(gate_path)
        hydrated["gate_counts"] = {
            "passed": int(meta.get("diagnostics.n_passed", 0)),
            "failed": int(meta.get("diagnostics.n_failed", 0)),
            "skipped": int(meta.get("diagnostics.n_skipped_no_rv", 0))
            + int(meta.get("diagnostics.n_skipped_elements", 0)),
        }
        chi2_key = "diagnostics.chi2_dof_values"
        if chi2_key in meta:
            hydrated["chi2_dof_values"] = list(np.asarray(meta[chi2_key], dtype=np.float64))

    inf_path = _optional_artifact_path(manifest, "inference")
    if inf_path is not None:
        from darkhunter_pop.inference import read_inference_artifact

        payload = read_inference_artifact(inf_path)
        runs = payload.get("sampler_run_summaries")
        if runs:
            hydrated["sampler_runs"] = runs

    sa_path = _optional_artifact_path(manifest, "sensitivity_analysis")
    if sa_path is not None:
        mc_noise = _read_mc_noise_from_sensitivity(sa_path)
        if mc_noise is not None:
            hydrated["mc_noise"] = mc_noise

    sf_path = _optional_artifact_path(manifest, "selection_function_astrometric")
    if sf_path is not None:
        solution_types = _read_solution_types_from_sf(sf_path)
        if solution_types is not None:
            hydrated["solution_types"] = solution_types

    elbadry = _read_elbadry_panels_from_manifest(manifest, config)
    if elbadry is not None:
        hydrated["elbadry_panels"] = elbadry

    return hydrated


def run_diagnostics_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    demo_hooks: bool = True,
    candidates: Sequence[CandidateRecord] | None = None,
    sampler_runs: Sequence[Mapping[str, Any]] | None = None,
    mc_noise: MCNoiseConvergenceDiagnostic | None = None,
    solution_types: SolutionTypeFractionResult | None = None,
    gate_counts: Mapping[str, int] | None = None,
    chi2_dof_values: Sequence[float] | None = None,
) -> RunManifest:
    """Execute the ``diagnostics`` suite stage and update the run manifest."""
    spec = STAGE_REGISTRY["diagnostics"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    hydrated = _hydrate_diagnostics_from_manifest(manifest, config)
    resolved_candidates = candidates if candidates is not None else hydrated.get("candidates")
    resolved_sampler_runs = (
        sampler_runs if sampler_runs is not None else hydrated.get("sampler_runs")
    )
    resolved_gate_counts = gate_counts if gate_counts is not None else hydrated.get("gate_counts")
    resolved_chi2 = (
        chi2_dof_values if chi2_dof_values is not None else hydrated.get("chi2_dof_values")
    )
    resolved_mc_noise = mc_noise if mc_noise is not None else hydrated.get("mc_noise")
    resolved_solution_types = (
        solution_types if solution_types is not None else hydrated.get("solution_types")
    )

    result = run_diagnostic_suite(
        config,
        run_id=manifest.run_id,
        candidates=resolved_candidates,
        funnel_counts=hydrated.get("funnel_counts"),
        elbadry_panels=hydrated.get("elbadry_panels"),
        gate_counts=resolved_gate_counts,
        chi2_dof_values=resolved_chi2,
        age_diagnostic=hydrated.get("age_diagnostic"),
        sampler_runs=resolved_sampler_runs,
        mc_noise=resolved_mc_noise,
        solution_types=resolved_solution_types,
        demo_missing=demo_hooks
        and resolved_candidates is None
        and resolved_sampler_runs is None,
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
