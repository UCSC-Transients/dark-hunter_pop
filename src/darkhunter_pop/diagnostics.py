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
Phase 8 sample-reproduction diagnostics (#113) are wired via
``emit_sample_*`` hooks and ``sample_diagnostics.py``.
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
    observed_benchmarks_from_artifacts,
    validate_benchmarks_config,
)
from darkhunter_pop.config_schema import PipelineConfig, SBCConfig

# Shared hook primitives moved to ``diagnostic_hooks`` (#182); re-exported here so
# ``from darkhunter_pop.diagnostics import ...`` call sites keep working.
from darkhunter_pop.diagnostic_hooks import (
    DiagnosticDirs,
    HookEmissionResult,
    _maybe_plot,
    emit_funnel_sky,
    emit_gate_pass_rate,
    format_funnel_report,
    format_gate_pass_rate_report,
    resolve_artifact_root,
    resolve_diagnostic_dirs,
    write_report,
)
from darkhunter_pop.forward_model import (
    SOLUTION_TYPE_LABELS,
    SolutionTypeFractionResult,
)
from darkhunter_pop.nss_covariance import CovarianceHealth
from darkhunter_pop.plotting import (
    matplotlib_available,
    plot_categorical_bars,
    plot_grouped_bars,
    plot_histogram,
    plot_line_with_threshold,
    plot_m2_posterior_convergence,
    plot_overlay_histograms,
    plot_six_panel_grid,
)
from darkhunter_pop.mc_mass_function import (
    M2PosteriorConvergenceDiagnostic,
    format_m2_posterior_convergence_report,
    propagate_nss_solution,
    run_m2_posterior_convergence,
    synthetic_orbital_solution,
)
from darkhunter_pop.sample_diagnostics import (
    SampleDiagnosticsBundle,
    attrition_bar_series,
    build_reproduction_comparisons,
    compute_janssens_segment_occupancy,
    compute_mode_divergence,
    evaluate_selection_function_curves,
    format_attrition_waterfall_report,
    format_covariance_health_report,
    format_janssens_segment_occupancy_report,
    format_mode_divergence_report,
    format_sample_reproduction_report,
    format_sample_selection_function_report,
    forward_model_selection,
    load_specs_for_results,
    reproduction_cfg,
    run_simon2026_diagnostic,
)
from darkhunter_pop.sample_selection import (
    SampleSelectionError,
    SampleSelectionRegistry,
    load_evaluation_results_from_artifact,
    load_sample_selection_file,
    load_tilde_masses_from_artifact,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    TRIPLES_DISABLED_SKIP_REASON,
    mark_stage_finished,
    mark_stage_started,
    plan_and_guard,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.sbc import (
    sbc_validates_pipeline,
    format_sbc_report,
    read_sbc_artifact,
    run_sbc_suite,
    write_sbc_artifact,
)
from darkhunter_pop.run_validity import (
    format_science_validity_block,
    write_stand_ins_to_handle,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    FitTier,
    RunManifest,
    StageStatus,
    SyntheticStandIn,
)
from darkhunter_pop.sensitivity_analysis import (
    MC_NOISE_ANALYTIC_LABEL,
    MCNoiseConvergenceDiagnostic,
    run_mc_noise_convergence,
)

# Lazy-import note: ``companion_nature`` / ``mass_derivation`` (and the upstream-stage
# artifact readers) are imported only inside the helpers that need them. This was
# originally to break an import cycle through ``data_acquisition``; since #182 the
# early stages import ``diagnostic_hooks`` instead, so no cycle remains, but the
# imports stay lazy to keep this module's import-time reach small. They are still
# declared in the ``diagnostics`` stage's ``dependency_modules`` (#183).

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
    #: Point-of-use stand-ins the suite took (#354): demo placeholders, fixture
    #: comparisons, analytic SBC, fabricated known-truth observations, ...
    stand_ins: list[SyntheticStandIn] = field(default_factory=list)
    #: ``inference``'s science-validity verdict, surfaced in the stage report (#352).
    upstream_science_validity: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stand_ins": [si.model_dump(mode="json") for si in self.stand_ins],
            "upstream_science_validity": self.upstream_science_validity,
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
                "catalog reports (issue #70). Optional SBC recovery is issue #69. "
                "Phase 8 sample-reproduction diagnostics are issue #113."
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


def format_age_stratified_wd_report(diagnostic: Any) -> str:
    """Full-detail age-stratified WD-debiasing / age-independence report.

    ``diagnostic`` is a ``companion_nature.AgeBinDiagnostic`` (typed as Any to
    avoid a circular import with ``data_acquisition`` → diagnostics).
    """
    status = getattr(diagnostic, "status", "") or (
        "ok" if diagnostic.age_independence_ok else "flag"
    )
    lines = [
        "=== age-stratified WD-debiasing check ===",
        (
            "  status: insufficient_data — NOT TESTED (an empty comparison is not "
            "a pass, #334)"
            if status == "insufficient_data"
            else f"  status: {status}"
        ),
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
        MC_NOISE_ANALYTIC_LABEL,
        diagnostic.message,
        f"  threshold (physics.mc_noise_threshold): {diagnostic.threshold}",
        f"  n_mock_final: {diagnostic.n_mock_final}",
        (
            "  all_bins_passed (analytic identity, NOT a validation): "
            f"{diagnostic.all_bins_passed}"
        ),
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
                title=(
                    "MC-noise convergence: analytic placeholder, not measured"
                ),
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


def emit_m2_posterior_convergence(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    diagnostic: M2PosteriorConvergenceDiagnostic | None = None,
) -> HookEmissionResult:
    """Hook: MC noise on ``P(M2 > threshold)`` plus systems near the 95% cut."""
    diag = config.diagnostics
    if not diag.hooks.m2_posterior_convergence:
        return HookEmissionResult(
            hook_name="m2_posterior_convergence",
            skipped_reason="diagnostics.hooks.m2_posterior_convergence=false",
        )
    if diagnostic is None:
        return HookEmissionResult(
            hook_name="m2_posterior_convergence",
            skipped_reason="no M2PosteriorConvergenceDiagnostic provided",
        )
    result = HookEmissionResult(
        hook_name="m2_posterior_convergence",
        payload=diagnostic.as_dict(),
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "m2_posterior_convergence.txt",
                format_m2_posterior_convergence_report(diagnostic),
            )
        )
    if diag.write_figures and diagnostic.per_system:
        path = _maybe_plot(
            True,
            lambda: plot_m2_posterior_convergence(
                [row["p_m2_above"] for row in diagnostic.per_system],
                [row["p_sigma"] for row in diagnostic.per_system],
                dirs.figures / "m2_posterior_convergence.png",
                probability_cut=float(diagnostic.probability_cut),
                dpi=diag.figure_dpi,
                title="MC noise on P(M2 > threshold)",
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
    """Hook: Gaia BH known-truth check against this run's pipeline outputs (#348).

    ``observed`` is built by the stage from real artifacts
    (``benchmarks.observed_benchmarks_from_artifacts``). When it is ``None`` there
    is nothing to compare and every check is ``not_tested``; there is no synthetic
    fallback (``synthetic_observed_from_truth`` is reachable from tests only, by
    passing its output here explicitly).
    """
    diag = config.diagnostics
    if not diag.hooks.known_truth_benchmarks:
        return HookEmissionResult(
            hook_name="known_truth_benchmarks",
            skipped_reason="diagnostics.hooks.known_truth_benchmarks=false",
        )
    validate_benchmarks_config(config)
    table = load_known_truth_table_from_config(config)
    obs: Mapping[int, Any] | Sequence[Any] = observed if observed is not None else {}
    obs_list = list(obs.values()) if isinstance(obs, Mapping) else list(obs)
    sources = sorted({str(getattr(o, "source", "supplied")) for o in obs_list})
    observed_source = "none" if not obs_list else ",".join(sources)
    mass_n_sigma = float(config.benchmarks.mass_check_n_sigma)
    results = check_known_truth_expectations(
        table,
        obs,
        ruwe_match_tolerance=float(config.benchmarks.ruwe_match_tolerance),
        mass_n_sigma=mass_n_sigma,
    )
    result = HookEmissionResult(
        hook_name="known_truth_benchmarks",
        payload={
            "observed_source": observed_source,
            "status_by_system": {r.name: r.status for r in results},
            "n_failed": sum(1 for r in results if r.status == "failed"),
            "n_not_tested": sum(1 for r in results if r.status == "not_tested"),
        },
    )
    if diag.write_reports:
        result.reports.append(
            write_report(
                dirs.reports / "known_truth_gaia_bh.txt",
                format_known_truth_report(
                    table,
                    results,
                    ruwe_match_tolerance=float(config.benchmarks.ruwe_match_tolerance),
                    observed_source=observed_source,
                    mass_n_sigma=mass_n_sigma,
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


def emit_sample_attrition_waterfall(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    results: Mapping[str, Any] | None = None,
) -> HookEmissionResult:
    """Hook: per-sample cut attrition with failed vs not-applicable separated."""
    diag = config.diagnostics
    if not diag.hooks.sample_attrition_waterfall:
        return HookEmissionResult(
            hook_name="sample_attrition_waterfall",
            skipped_reason="diagnostics.hooks.sample_attrition_waterfall=false",
        )
    payloads = dict(results or {})
    if not payloads:
        return HookEmissionResult(
            hook_name="sample_attrition_waterfall",
            skipped_reason="no SampleEvaluationResult payloads provided",
        )
    emission = HookEmissionResult(
        hook_name="sample_attrition_waterfall",
        payload={
            name: {
                "n_parent": res.n_parent,
                "n_surviving": res.n_surviving,
                "n_cuts": len(res.attrition),
            }
            for name, res in payloads.items()
        },
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "sample_attrition_waterfall.txt",
                format_attrition_waterfall_report(payloads),
            )
        )
    if diag.write_figures:
        for name, res in payloads.items():
            if not res.attrition:
                continue
            labels, series = attrition_bar_series(res)
            path = _maybe_plot(
                True,
                lambda labels=labels, series=series, name=name: plot_grouped_bars(
                    labels,
                    series,
                    dirs.figures / f"sample_attrition_waterfall_{name}.png",
                    xlabel="cut_id",
                    ylabel="count",
                    title=f"{name} attrition (passed / failed / not_applicable)",
                    dpi=diag.figure_dpi,
                    style=config.plotting,
                ),
            )
            if path is not None:
                emission.figures.append(path)
    return emission


def emit_sample_reproduction_report(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    results: Mapping[str, Any] | None = None,
    specs: Mapping[str, Any] | None = None,
) -> HookEmissionResult:
    """Hook: recovered source-ID set vs published table / N."""
    diag = config.diagnostics
    if not diag.hooks.sample_reproduction_report:
        return HookEmissionResult(
            hook_name="sample_reproduction_report",
            skipped_reason="diagnostics.hooks.sample_reproduction_report=false",
        )
    payloads = dict(results or {})
    if not payloads:
        return HookEmissionResult(
            hook_name="sample_reproduction_report",
            skipped_reason="no SampleEvaluationResult payloads provided",
        )
    resolved_specs = dict(specs or {})
    if not resolved_specs:
        resolved_specs = load_specs_for_results(payloads, config)
    comparisons = build_reproduction_comparisons(payloads, resolved_specs, config)
    emission = HookEmissionResult(
        hook_name="sample_reproduction_report",
        payload={
            "comparisons": [c.as_dict() for c in comparisons],
            "all_n_match": all(c.n_match for c in comparisons),
        },
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "sample_reproduction_report.txt",
                format_sample_reproduction_report(comparisons),
            )
        )
    if diag.write_figures and comparisons:
        labels = [c.sample_name for c in comparisons]
        series = {
            "recovered_n": [float(c.recovered_n) for c in comparisons],
            "published_n": [
                float(c.published_n) if c.published_n is not None else float("nan")
                for c in comparisons
            ],
        }
        path = _maybe_plot(
            True,
            lambda: plot_grouped_bars(
                labels,
                series,
                dirs.figures / "sample_reproduction_counts.png",
                xlabel="sample",
                ylabel="count",
                title="recovered vs published N",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            emission.figures.append(path)
    return emission


def emit_simon2026_exclusion_breakdown(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    sample_ids: Sequence[int] | None = None,
    simon_rows: Sequence[Mapping[str, Any]] | None = None,
    spec: Any | None = None,
    elbadry_m2_over_m1_by_source: Mapping[int, float] | None = None,
) -> HookEmissionResult:
    """Hook: Simon et al. (2026) 5/2/1/1 exclusion breakdown (§8.9).

    ``elbadry_m2_over_m1_by_source`` is El-Badry 2026's own ``M̃2/M̃1`` per
    source, used for ``fails_m2_over_m1`` (#281); see
    :func:`~darkhunter_pop.sample_diagnostics.run_simon2026_diagnostic`.
    """
    diag = config.diagnostics
    if not diag.hooks.simon2026_exclusion_breakdown:
        return HookEmissionResult(
            hook_name="simon2026_exclusion_breakdown",
            skipped_reason="diagnostics.hooks.simon2026_exclusion_breakdown=false",
        )
    counts, report = run_simon2026_diagnostic(
        config,
        spec=spec,
        sample_ids=sample_ids,
        rows=simon_rows,
        elbadry_m2_over_m1_by_source=elbadry_m2_over_m1_by_source,
    )
    emission = HookEmissionResult(
        hook_name="simon2026_exclusion_breakdown",
        payload=dict(counts),
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(dirs.reports / "simon2026_exclusion_breakdown.txt", report)
        )
    if diag.write_figures:
        labels = [
            "in_sample",
            "sb1_fails_significance",
            "astrometric_f2_above_max",
            "fainter_than_g_limit",
            "fails_m2_over_m1",
            "unclassified",
        ]
        values = [float(counts.get(k, 0)) for k in labels]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                values,
                dirs.figures / "simon2026_exclusion_breakdown.png",
                xlabel="reason",
                ylabel="count",
                title="Simon et al. (2026) exclusion breakdown",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            emission.figures.append(path)
    return emission


def emit_covariance_health(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    health: CovarianceHealth | None = None,
) -> HookEmissionResult:
    """Hook: missing / non-PSD NSS covariance counts by solution type."""
    diag = config.diagnostics
    if not diag.hooks.covariance_health:
        return HookEmissionResult(
            hook_name="covariance_health",
            skipped_reason="diagnostics.hooks.covariance_health=false",
        )
    if health is None:
        return HookEmissionResult(
            hook_name="covariance_health",
            skipped_reason="no CovarianceHealth provided",
        )
    emission = HookEmissionResult(
        hook_name="covariance_health",
        payload=health.as_dict(),
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "covariance_health.txt",
                format_covariance_health_report(health),
            )
        )
    if diag.write_figures:
        labels = [
            "ok",
            "missing_corr",
            "unsupported_solution_type",
            "bit_index_mismatch",
            "unpack_failed",
            "missing_values",
            "missing_errors",
            "non_symmetric",
            "non_psd",
        ]
        values = [
            float(health.ok),
            float(health.missing_corr),
            float(health.unsupported_solution_type),
            float(health.bit_index_mismatch),
            float(health.unpack_failed),
            float(health.missing_values),
            float(health.missing_errors),
            float(health.non_symmetric),
            float(health.non_psd),
        ]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                values,
                dirs.figures / "covariance_health.png",
                xlabel="status",
                ylabel="count",
                title="NSS covariance health",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            emission.figures.append(path)
    return emission


def emit_sample_selection_function(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    specs: Mapping[str, Any] | None = None,
    sample_names: Sequence[str] | None = None,
) -> HookEmissionResult:
    """Hook: forward-model survival probability vs M2, P_orb, G."""
    diag = config.diagnostics
    if not diag.hooks.sample_selection_function:
        return HookEmissionResult(
            hook_name="sample_selection_function",
            skipped_reason="diagnostics.hooks.sample_selection_function=false",
        )
    resolved = dict(specs or {})
    if not resolved:
        try:
            registry = SampleSelectionRegistry(config)
            for name in registry.evaluation_order():
                resolved[name] = registry.resolved(name)
        except SampleSelectionError as exc:
            return HookEmissionResult(
                hook_name="sample_selection_function",
                skipped_reason=f"sample registry unavailable: {exc}",
            )
    if sample_names is not None:
        resolved = {k: v for k, v in resolved.items() if k in set(sample_names)}
    if not resolved:
        return HookEmissionResult(
            hook_name="sample_selection_function",
            skipped_reason="no sample specs available for survival sweep",
        )
    all_curves = []
    for _name, spec in sorted(resolved.items()):
        selection = forward_model_selection(spec)
        all_curves.extend(evaluate_selection_function_curves(selection, config))
    emission = HookEmissionResult(
        hook_name="sample_selection_function",
        payload={"curves": [c.as_dict() for c in all_curves]},
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "sample_selection_function.txt",
                format_sample_selection_function_report(all_curves),
            )
        )
    if diag.write_figures:
        for curve in all_curves:
            path = _maybe_plot(
                True,
                lambda curve=curve: plot_line_with_threshold(
                    curve.x,
                    curve.survival,
                    dirs.figures
                    / f"sample_selection_function_{curve.sample_name}_{curve.axis}.png",
                    xlabel=curve.axis,
                    ylabel="survival probability",
                    title=f"{curve.sample_name} selection function vs {curve.axis}",
                    dpi=diag.figure_dpi,
                    threshold=None,
                    style=config.plotting,
                ),
            )
            if path is not None:
                emission.figures.append(path)
    return emission


def emit_mode_divergence(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    results: Mapping[str, Any] | None = None,
) -> HookEmissionResult:
    """Hook: andrews2022 vs andrews2022_modified (and other configured pairs)."""
    diag = config.diagnostics
    if not diag.hooks.mode_divergence:
        return HookEmissionResult(
            hook_name="mode_divergence",
            skipped_reason="diagnostics.hooks.mode_divergence=false",
        )
    payloads = dict(results or {})
    if not payloads:
        return HookEmissionResult(
            hook_name="mode_divergence",
            skipped_reason="no SampleEvaluationResult payloads provided",
        )
    divergences = []
    for pair in reproduction_cfg(config).mode_divergence_pairs:
        left = payloads.get(pair.left)
        right = payloads.get(pair.right)
        if left is None or right is None:
            continue
        divergences.append(
            compute_mode_divergence(
                left,
                right,
                expected_only_left=pair.expected_only_in_left,
                expected_only_right=pair.expected_only_in_right,
            )
        )
    if not divergences:
        return HookEmissionResult(
            hook_name="mode_divergence",
            skipped_reason="configured mode_divergence pairs missing from results",
        )
    emission = HookEmissionResult(
        hook_name="mode_divergence",
        payload={
            "divergences": [d.as_dict() for d in divergences],
            "all_match": all(d.matches_expectation for d in divergences),
        },
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "mode_divergence.txt",
                format_mode_divergence_report(divergences),
            )
        )
    if diag.write_figures:
        for div in divergences:
            labels = [div.left_name, div.right_name, "only_left", "only_right"]
            values = [
                float(len(div.left_ids)),
                float(len(div.right_ids)),
                float(len(div.only_left)),
                float(len(div.only_right)),
            ]
            path = _maybe_plot(
                True,
                lambda labels=labels, values=values, div=div: plot_categorical_bars(
                    labels,
                    values,
                    dirs.figures / f"mode_divergence_{div.left_name}_vs_{div.right_name}.png",
                    xlabel="set",
                    ylabel="count",
                    title=f"mode divergence: {div.left_name} vs {div.right_name}",
                    dpi=diag.figure_dpi,
                    style=config.plotting,
                ),
            )
            if path is not None:
                emission.figures.append(path)
    return emission


def emit_janssens_segment_occupancy(
    config: PipelineConfig,
    dirs: DiagnosticDirs,
    *,
    mg_0_values: Sequence[float] | NDArray[np.floating] | None = None,
) -> HookEmissionResult:
    """Hook: Janssens mass-segment occupancy for El-Badry 2026 ``M̃1``."""
    diag = config.diagnostics
    if not diag.hooks.janssens_segment_occupancy:
        return HookEmissionResult(
            hook_name="janssens_segment_occupancy",
            skipped_reason="diagnostics.hooks.janssens_segment_occupancy=false",
        )
    if mg_0_values is None:
        return HookEmissionResult(
            hook_name="janssens_segment_occupancy",
            skipped_reason="no mg_0 values provided",
        )
    table = reproduction_cfg(config).janssens_table
    occupancy = compute_janssens_segment_occupancy(mg_0_values, table_path=table)
    emission = HookEmissionResult(
        hook_name="janssens_segment_occupancy",
        payload=occupancy.as_dict(),
    )
    if diag.write_reports:
        emission.reports.append(
            write_report(
                dirs.reports / "janssens_segment_occupancy.txt",
                format_janssens_segment_occupancy_report(occupancy),
            )
        )
    if diag.write_figures and occupancy.segment_counts:
        labels = [
            f"{row['m_low']:g}-{row['m_up']:g}" for row in occupancy.segment_counts
        ]
        values = [float(row["count"]) for row in occupancy.segment_counts]
        path = _maybe_plot(
            True,
            lambda: plot_categorical_bars(
                labels,
                values,
                dirs.figures / "janssens_segment_occupancy.png",
                xlabel="mass segment (Msun)",
                ylabel="count",
                title="Janssens segment occupancy",
                dpi=diag.figure_dpi,
                style=config.plotting,
            ),
        )
        if path is not None:
            emission.figures.append(path)
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
        "emit_m2_posterior_convergence": emit_m2_posterior_convergence,
        "emit_solution_type_fractions": emit_solution_type_fractions,
        "emit_known_truth_benchmarks": emit_known_truth_benchmarks,
        "emit_comparison_catalogs": emit_comparison_catalogs,
        "emit_sbc_recovery": emit_sbc_recovery,
        "emit_sample_attrition_waterfall": emit_sample_attrition_waterfall,
        "emit_sample_reproduction_report": emit_sample_reproduction_report,
        "emit_simon2026_exclusion_breakdown": emit_simon2026_exclusion_breakdown,
        "emit_covariance_health": emit_covariance_health,
        "emit_sample_selection_function": emit_sample_selection_function,
        "emit_mode_divergence": emit_mode_divergence,
        "emit_janssens_segment_occupancy": emit_janssens_segment_occupancy,
        "format_funnel_report": format_funnel_report,
        "format_elbadry_panel_report": format_elbadry_panel_report,
        "format_fit_tier_coverage_report": format_fit_tier_coverage_report,
        "format_gate_pass_rate_report": format_gate_pass_rate_report,
        "format_age_stratified_wd_report": format_age_stratified_wd_report,
        "format_triples_robustness_report": format_triples_robustness_report,
        "format_info_gain_report": format_info_gain_report,
        "format_sampler_consistency_report": format_sampler_consistency_report,
        "format_mc_noise_convergence_report": format_mc_noise_convergence_report,
        "format_m2_posterior_convergence_report": format_m2_posterior_convergence_report,
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


def diagnostics_suite_stand_ins(
    config: PipelineConfig,
    hooks: Sequence[HookEmissionResult],
    *,
    demo_fed_hooks: Sequence[str],
) -> list[SyntheticStandIn]:
    """Stand-ins the diagnostic suite actually took, read off the hooks that ran (#354)."""
    by_name = {h.hook_name: h for h in hooks}
    out: list[SyntheticStandIn] = []
    if demo_fed_hooks:
        out.append(
            SyntheticStandIn(
                name="diagnostics_demo_placeholders",
                stage="diagnostics",
                kind="synthetic_data",
                replaces="upstream stage outputs for these diagnostic hooks",
                description=(
                    "demo_missing was in effect, so these hooks were fed "
                    "placeholder inputs (zeros, linspace panels, uniform fractions, "
                    "a synthetic NSS solution) instead of this run's outputs. Their "
                    "reports describe the placeholders: "
                    + ", ".join(demo_fed_hooks)
                    + "."
                ),
                config_keys=[],
                values={"hooks": list(demo_fed_hooks)},
            )
        )
    kt = by_name.get("known_truth_benchmarks")
    if kt is not None and "synthetic_observed_from_truth" in str(
        kt.payload.get("observed_source", "")
    ):
        out.append(
            SyntheticStandIn(
                name="fabricated_known_truth_observations",
                stage="diagnostics",
                kind="synthetic_data",
                replaces="this run's pipeline outputs for Gaia BH1/BH2/BH3",
                description=(
                    "The known-truth check compared the benchmark expectations "
                    "against observations synthesized from those same expectations "
                    "(synthetic_observed_from_truth), so it cannot fail (#348)."
                ),
                config_keys=["benchmarks"],
            )
        )
    sbc = by_name.get("sbc_recovery")
    if (
        sbc is not None
        and sbc.skipped_reason is None
        and sbc.payload.get("recovery_backend") == "analytic_binned"
    ):
        out.append(
            SyntheticStandIn(
                name="sbc_analytic_binned_backend",
                stage="diagnostics",
                kind="analytic_identity",
                replaces="simulation-based calibration through run_inference",
                description=(
                    "SBC drew conjugate Gamma posteriors for independent Poisson "
                    "bins with SF = 1; coverage is correct by construction and "
                    "exercises no part of inference, the forward-model SF or the "
                    "population model (#356)."
                ),
                config_keys=["diagnostics.sbc.recovery_backend"],
                values={"recovery_backend": "analytic_binned"},
            )
        )
    ssf = by_name.get("sample_selection_function")
    if ssf is not None and ssf.skipped_reason is None:
        out.append(
            SyntheticStandIn(
                name="single_template_sample_sf_scan",
                stage="diagnostics",
                kind="config_placeholder",
                replaces="per-sample selection functions marginalized over the population",
                description=(
                    "The sample selection-function curves sweep one axis at a time "
                    "through a single configured template row and record 0/1 "
                    "survival; they are not selection probabilities over any "
                    "population."
                ),
                config_keys=["diagnostics.sample_reproduction.selection_function.template"],
            )
        )
    return out


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
    m2_posterior: M2PosteriorConvergenceDiagnostic | None = None,
    solution_types: SolutionTypeFractionResult | None = None,
    sample_bundle: SampleDiagnosticsBundle | None = None,
    known_truth_observed: Mapping[int, Any] | None = None,
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
    demo_fed: list[str] = []

    if funnel_counts is None and demo_missing:
        demo_fed.append("funnel_sky")
    if elbadry_panels is None and demo_missing:
        demo_fed.append("elbadry_six_panel")
    if not cand and demo_missing:
        demo_fed.append("fit_tier_coverage")
    if not gate_counts and demo_missing:
        demo_fed.append("gate_pass_rate")
    if not sampler_runs and demo_missing:
        demo_fed.append("sampler_consistency")
    if mc_noise is None and demo_missing:
        demo_fed.append("mc_noise_convergence")
    if m2_posterior is None and demo_missing:
        demo_fed.append("m2_posterior_convergence")
    if solution_types is None and demo_missing:
        demo_fed.append("solution_type_fractions")
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

    if m2_posterior is not None or demo_missing:
        m2c = m2_posterior
        if m2c is None:
            mc_cfg = config.mc_mass_function
            demo_draws = propagate_nss_solution(
                synthetic_orbital_solution(relative_error=0.03, seed=mc_cfg.random_seed),
                m1_msun=1.0,
                n_draws=min(256, int(mc_cfg.n_draws)),
                random_seed=int(mc_cfg.random_seed),
                eig_rel_floor=float(mc_cfg.eig_rel_floor),
                eig_abs_floor=float(mc_cfg.eig_abs_floor),
                source_id=0,
                covariance_mode=mc_cfg.covariance,
            )
            m2c = run_m2_posterior_convergence(
                [demo_draws],
                m2_threshold_msun=1.4,
                probability_cut=0.95,
                mc_noise_threshold=float(config.physics.mc_noise_threshold),
                boundary_n_sigma=float(mc_cfg.boundary_n_sigma),
            )
        hooks.append(emit_m2_posterior_convergence(config, dirs, diagnostic=m2c))
    else:
        hooks.append(emit_m2_posterior_convergence(config, dirs, diagnostic=None))

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

    hooks.append(
        emit_known_truth_benchmarks(config, dirs, observed=known_truth_observed)
    )
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

    bundle = sample_bundle if sample_bundle is not None else SampleDiagnosticsBundle()
    sample_results = dict(bundle.evaluation_results)
    sample_specs = dict(bundle.sample_specs)
    if demo_missing and not sample_results:
        # Demo layout only — zero-length stand-ins so hooks skip cleanly or emit empty.
        pass
    hooks.append(
        emit_sample_attrition_waterfall(config, dirs, results=sample_results or None)
    )
    hooks.append(
        emit_sample_reproduction_report(
            config,
            dirs,
            results=sample_results or None,
            specs=sample_specs or None,
        )
    )
    hooks.append(
        emit_simon2026_exclusion_breakdown(
            config,
            dirs,
            sample_ids=bundle.simon_in_sample_ids,
            simon_rows=bundle.simon_rows,
            spec=sample_specs.get("elbadry2026"),
            elbadry_m2_over_m1_by_source=bundle.simon_elbadry_m2_over_m1,
        )
    )
    cov_health = bundle.covariance_health
    if cov_health is None and demo_missing:
        cov_health = CovarianceHealth()
        demo_fed.append("covariance_health")
    hooks.append(emit_covariance_health(config, dirs, health=cov_health))
    hooks.append(
        emit_sample_selection_function(
            config,
            dirs,
            specs=sample_specs or None,
            sample_names=bundle.selection_function_samples,
        )
    )
    hooks.append(emit_mode_divergence(config, dirs, results=sample_results or None))
    mg_vals = bundle.mg_0_values
    if mg_vals is None and demo_missing:
        mg_vals = [4.73, 2.5, 8.0, -9.0]
        demo_fed.append("janssens_segment_occupancy")
    hooks.append(emit_janssens_segment_occupancy(config, dirs, mg_0_values=mg_vals))

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
        stand_ins=diagnostics_suite_stand_ins(config, hooks, demo_fed_hooks=demo_fed),
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
        write_stand_ins_to_handle(handle, result.stand_ins)
        if result.upstream_science_validity is not None:
            handle.attrs["upstream_science_valid"] = bool(
                result.upstream_science_validity.get("science_valid", False)
            )
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


def format_check_verdicts(result: DiagnosticsStageResult) -> list[str]:
    """One plain verdict line per check whose green has been mistaken for a test.

    ``not_tested`` / ``analytic`` are stated as such; nothing here reports a pass
    for a check that saw no data or that is an identity (#334, #348, #349, #356).
    """
    by_name = {h.hook_name: h for h in result.hooks_run}
    lines = ["check verdicts:"]
    kt = by_name.get("known_truth_benchmarks")
    if kt is not None and kt.skipped_reason is None:
        statuses = kt.payload.get("status_by_system") or {}
        if not statuses:
            verdict = "not_tested"
        elif any(v == "failed" for v in statuses.values()):
            verdict = "FAILED"
        elif any(v == "not_tested" for v in statuses.values()):
            verdict = "not_tested (partial)"
        else:
            verdict = "passed"
        lines.append(
            f"  known_truth_benchmarks: {verdict} {dict(statuses)} "
            f"(observed_source={kt.payload.get('observed_source')})"
        )
    age = by_name.get("age_stratified_wd")
    if age is not None and age.skipped_reason is None:
        lines.append(f"  age_stratified_wd: {age.payload.get('status', 'unknown')}")
    mc = by_name.get("mc_noise_convergence")
    if mc is not None and mc.skipped_reason is None:
        lines.append(
            "  mc_noise_convergence: analytic placeholder, not measured (#349)"
        )
    if result.sbc_payload is not None:
        backend = str(result.sbc_payload.get("recovery_backend", ""))
        lines.append(
            "  sbc_recovery: analytic sanity check only, does not validate inference (#356)"
            if not sbc_validates_pipeline(backend)
            else f"  sbc_recovery: overall_passed={result.sbc_payload.get('overall_passed')}"
        )
    if len(lines) == 1:
        lines.append("  (none of the tracked checks ran)")
    return lines


def format_diagnostics_stage_report(result: DiagnosticsStageResult) -> str:
    """Fully legible diagnostics-stage summary (exempt from caveman compression)."""
    lines = [
        "=== diagnostics stage (full suite) ===",
        (
            format_science_validity_block(result.upstream_science_validity)
            if result.upstream_science_validity
            else "upstream science validity: NOT ASSESSED (no inference verdict on "
            "this run) — nothing here is a science-valid result"
        ),
        *format_check_verdicts(result),
        f"stand-ins taken by this suite ({len(result.stand_ins)}):",
        *[f"  - {si.one_line()}" for si in result.stand_ins],
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
        backend = str(result.sbc_payload.get("recovery_backend", sbc_cfg.get("recovery_backend")))
        if sbc_validates_pipeline(backend):
            lines.append(f"  overall_passed: {result.sbc_payload.get('overall_passed')}")
        else:
            lines.append(
                "  overall_passed (analytic sanity check only — does NOT validate "
                f"inference, #356): {result.sbc_payload.get('overall_passed')}"
            )
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


def elbadry_m2_over_m1_from_tilde_masses(
    masses: Mapping[int, tuple[float, float]],
) -> dict[int, float] | None:
    """El-Badry 2026's own ``M̃2/M̃1`` per source from ``(M̃1, M̃2)`` pairs (#285).

    Pairs with a non-positive ``M̃1`` are skipped (the source falls back to the
    Simon catalog ratio). Returns ``None`` when nothing usable remains, so the
    Simon breakdown takes its documented fallback path.
    """
    ratios = {
        int(sid): float(m2) / float(m1)
        for sid, (m1, m2) in masses.items()
        if float(m1) > 0.0
    }
    return ratios or None


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
        validity = payload.get("science_validity")
        if validity:
            hydrated["science_validity"] = dict(validity)

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

    ss_path = _optional_artifact_path(manifest, "sample_selection")
    if ss_path is not None:
        try:
            eval_results = load_evaluation_results_from_artifact(ss_path)
        except (OSError, ValueError, FileNotFoundError):
            eval_results = {}
        if eval_results:
            specs = load_specs_for_results(eval_results, config)
            elbadry_ratios: dict[int, float] | None = None
            if "elbadry2026" in eval_results:
                try:
                    elbadry_ratios = elbadry_m2_over_m1_from_tilde_masses(
                        load_tilde_masses_from_artifact(ss_path, "elbadry2026")
                    )
                except (OSError, KeyError, ValueError):
                    elbadry_ratios = None
            hydrated["sample_bundle"] = SampleDiagnosticsBundle(
                evaluation_results=eval_results,
                sample_specs=specs,
                simon_in_sample_ids=(
                    list(eval_results["elbadry2026"].surviving_source_ids)
                    if "elbadry2026" in eval_results
                    else None
                ),
                simon_elbadry_m2_over_m1=elbadry_ratios,
            )

    if config.diagnostics.hooks.known_truth_benchmarks:
        try:
            table = load_known_truth_table_from_config(config)
            hydrated["known_truth_observed"] = observed_benchmarks_from_artifacts(
                table,
                data_acquisition_artifact=_optional_artifact_path(
                    manifest, "data_acquisition"
                ),
                mass_artifacts={
                    stage: path
                    for stage in config.benchmarks.mass_check_stages
                    if (path := _optional_artifact_path(manifest, stage)) is not None
                },
                orbital_solution_types=config.benchmarks.nss_orbital_solution_types,
            )
        except (OSError, KeyError, ValueError):
            hydrated["known_truth_observed"] = None

    da_cov_path = _optional_artifact_path(manifest, "data_acquisition")
    if da_cov_path is not None:
        try:
            with h5py.File(da_cov_path, "r") as handle:
                if (
                    "data_acquisition" in handle
                    and "nss_covariance" in handle["data_acquisition"]
                ):
                    grp = handle["data_acquisition"]["nss_covariance"]
                    health = CovarianceHealth(
                        ok=int(grp.attrs.get("covariance_ok", 0)),
                        missing_corr=int(grp.attrs.get("covariance_missing_corr", 0)),
                        unsupported_solution_type=int(
                            grp.attrs.get("covariance_unsupported_solution_type", 0)
                        ),
                        bit_index_mismatch=int(
                            grp.attrs.get("covariance_bit_index_mismatch", 0)
                        ),
                        unpack_failed=int(
                            grp.attrs.get("covariance_unpack_failed", 0)
                        ),
                        missing_values=int(
                            grp.attrs.get("covariance_missing_values", 0)
                        ),
                        missing_errors=int(
                            grp.attrs.get("covariance_missing_errors", 0)
                        ),
                        non_symmetric=int(
                            grp.attrs.get("covariance_non_symmetric", 0)
                        ),
                        non_psd=int(grp.attrs.get("covariance_non_psd", 0)),
                    )
                    hydrated.setdefault(
                        "sample_bundle", SampleDiagnosticsBundle()
                    )
                    bundle = hydrated["sample_bundle"]
                    assert isinstance(bundle, SampleDiagnosticsBundle)
                    bundle.covariance_health = health
        except OSError:
            pass

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
    guard = plan_and_guard(
        spec, manifest, config, run_path=run_path, force_rerun=force_rerun
    )
    if not guard.proceed:
        return guard.manifest
    manifest = guard.manifest
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

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
        sample_bundle=hydrated.get("sample_bundle"),
        known_truth_observed=hydrated.get("known_truth_observed"),
        demo_missing=demo_hooks
        and resolved_candidates is None
        and resolved_sampler_runs is None,
    )
    if hydrated.get("elbadry_panels") is not None:
        # _read_elbadry_panels_from_manifest takes the "mock" side from the bundled
        # reference file (#339 / #355 own the fix); declare it at the point of use.
        result.stand_ins.append(
            SyntheticStandIn(
                name="six_panel_mock_is_reference_fixture",
                stage="diagnostics",
                kind="reference_fixture",
                replaces="this run's gaiamock mock six-panel samples",
                description=(
                    "The diagnostics six-panel figure's 'mock' side is the bundled "
                    "El-Badry 2024 reference file "
                    "(tests/fixtures/elbadry2024_dr3_nss_reference.npz or "
                    "validation_gate.reference_path), not this run's mock injections."
                ),
                config_keys=[
                    "selection_function_astrometric.validation_gate.reference_path"
                ],
            )
        )
    if hydrated.get("science_validity") is not None:
        result.upstream_science_validity = hydrated["science_validity"]
    elif manifest.science_valid is not None:
        result.upstream_science_validity = {
            "science_valid": manifest.science_valid,
            "policy": config.inference.upstream_gate_policy,
            "gates": [],
            "reasons": list(manifest.science_validity_reasons),
        }
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
