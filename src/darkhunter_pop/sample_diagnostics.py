"""Phase 8 sample-reproduction diagnostics (CONTINUATION_PLAN §13 / issue #113).

Owned hooks: ``sample_attrition_waterfall``, ``sample_reproduction_report``,
``simon2026_exclusion_breakdown``, ``covariance_health``,
``sample_selection_function``, ``mode_divergence``,
``janssens_segment_occupancy``. Spuriousness diagnostics belong to #27.

Reports are full-detail (caveman exemption). Figures use ``plotting.py``
primitives only — no new rendering paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    PipelineConfig,
    SampleReproductionDiagnosticsConfig,
    SampleSelectionMode,
)
from darkhunter_pop.elbadry2026_selection import (
    load_simon2026_orbital,
    simon2026_exclusion_breakdown,
)
from darkhunter_pop.janssens_mass import (
    UNINFORMATIVE_SEGMENT_MLOW,
    UNINFORMATIVE_SEGMENT_MUP,
    invert_mg_to_mass,
    load_janssens_table,
    segments_from_table,
)
from darkhunter_pop.nss_covariance import CovarianceHealth
from darkhunter_pop.sample_selection import (
    CutAttrition,
    SampleEvaluationResult,
    SampleSelection,
    SampleSelectionFile,
    load_sample_selection_file,
    resolve_inherits,
)


@dataclass
class ReproductionCompareResult:
    """Recovered vs published membership for one named sample / branch."""

    sample_name: str
    recovered_n: int
    published_n: int | None
    recovered_ids: tuple[int, ...]
    published_ids: tuple[int, ...] | None
    only_recovered: tuple[int, ...]
    only_published: tuple[int, ...]
    n_match: bool
    id_match: bool | None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "recovered_n": self.recovered_n,
            "published_n": self.published_n,
            "recovered_ids": list(self.recovered_ids),
            "published_ids": (
                None if self.published_ids is None else list(self.published_ids)
            ),
            "only_recovered": list(self.only_recovered),
            "only_published": list(self.only_published),
            "n_match": self.n_match,
            "id_match": self.id_match,
            "notes": self.notes,
        }


@dataclass
class ModeDivergenceResult:
    """Symmetric difference between two named-sample recovered sets."""

    left_name: str
    right_name: str
    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    only_left: tuple[int, ...]
    only_right: tuple[int, ...]
    expected_only_left: tuple[int, ...]
    expected_only_right: tuple[int, ...]
    matches_expectation: bool
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "left_n": len(self.left_ids),
            "right_n": len(self.right_ids),
            "only_left": list(self.only_left),
            "only_right": list(self.only_right),
            "expected_only_left": list(self.expected_only_left),
            "expected_only_right": list(self.expected_only_right),
            "matches_expectation": self.matches_expectation,
            "explanation": self.explanation,
        }


@dataclass
class JanssensOccupancyResult:
    """Per-segment occupancy for El-Badry 2026 ``M̃1`` (§8.2 / §13)."""

    segment_counts: list[dict[str, Any]]
    n_boundary_resolved: int
    n_out_of_range: int
    n_missing_mg: int
    n_total: int
    uninformative_segment_flagged: bool
    uninformative_segment_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_counts": list(self.segment_counts),
            "n_boundary_resolved": self.n_boundary_resolved,
            "n_out_of_range": self.n_out_of_range,
            "n_missing_mg": self.n_missing_mg,
            "n_total": self.n_total,
            "uninformative_segment_flagged": self.uninformative_segment_flagged,
            "uninformative_segment_count": self.uninformative_segment_count,
            "uninformative_mass_range_msun": [
                UNINFORMATIVE_SEGMENT_MLOW,
                UNINFORMATIVE_SEGMENT_MUP,
            ],
        }


@dataclass
class SelectionFunctionCurve:
    """Survival probability along one property axis."""

    axis: str
    x: list[float]
    survival: list[float]
    sample_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "x": list(self.x),
            "survival": list(self.survival),
            "sample_name": self.sample_name,
        }


@dataclass
class SampleDiagnosticsBundle:
    """Optional payloads collected for the Phase 8 sample hooks."""

    evaluation_results: dict[str, SampleEvaluationResult] = field(default_factory=dict)
    sample_specs: dict[str, SampleSelectionFile] = field(default_factory=dict)
    covariance_health: CovarianceHealth | None = None
    mg_0_values: Sequence[float] | NDArray[np.floating] | None = None
    simon_rows: Sequence[Mapping[str, Any]] | None = None
    simon_in_sample_ids: Sequence[int] | None = None
    selection_function_samples: Sequence[str] | None = None


def _resolve_path(relative: str | Path, *, repo: Path | None = None) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    root = repo if repo is not None else repo_root()
    return root / path


def load_published_source_ids(
    table_path: str | Path,
    *,
    repo: Path | None = None,
) -> tuple[int, ...]:
    """Load ``source_id`` values from a frozen external YAML table."""
    path = _resolve_path(table_path, repo=repo)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "data" not in raw:
        raise ValueError(f"published table missing data list: {path}")
    return tuple(sorted(int(row["source_id"]) for row in raw["data"]))


def reproduction_cfg(
    config: PipelineConfig,
) -> SampleReproductionDiagnosticsConfig:
    """Return the Phase 8 sample-reproduction layout block."""
    return config.diagnostics.sample_reproduction


def format_attrition_waterfall_report(
    results: Mapping[str, SampleEvaluationResult],
) -> str:
    """Full-detail attrition table with failed vs not-applicable separated (§15 Q10).

    Branched samples (El-Badry 2026) emit cut ids prefixed by branch/subsample so
    the four-subsample UNION structure is visible rather than a single linear chain.
    """
    lines = [
        "=== sample_attrition_waterfall ===",
        "Columns: cut_id | n_in | n_passed | n_failed | n_not_applicable | n_out |",
        "         expected_n_after | match",
        "Note: n_not_applicable is distinct from n_failed (CONTINUATION_PLAN §15 Q10).",
        "Collapsing them makes El-Badry 2026 subsample counts irreproducible.",
    ]
    if not results:
        lines.append("  (no SampleEvaluationResult payloads provided)")
    for name, result in sorted(results.items()):
        lines.append(f"sample: {name}")
        lines.append(f"  mode: {result.mode.value}")
        lines.append(f"  mass_source: {result.mass_source}")
        lines.append(f"  n_parent: {result.n_parent}")
        lines.append(f"  n_surviving: {result.n_surviving}")
        if result.branch_surviving:
            for branch_id, ids in sorted(result.branch_surviving.items()):
                lines.append(f"  branch[{branch_id}].n: {len(ids)}")
        if result.subsample_surviving:
            for sub_id, ids in sorted(result.subsample_surviving.items()):
                lines.append(f"  subsample[{sub_id}].n: {len(ids)}")
            sub_ns = [len(ids) for ids in result.subsample_surviving.values()]
            if sub_ns:
                lines.append(
                    "  subsample_union_arithmetic: "
                    + " + ".join(str(n) for n in sub_ns)
                    + f" (unique survivors may be fewer than sum; unique={result.n_surviving})"
                )
        if result.route_counts:
            for route, count in sorted(result.route_counts.items()):
                lines.append(f"  route[{route}]: {count}")
        if not result.attrition:
            lines.append("  attrition: (empty)")
            continue
        lines.append("  attrition:")
        for row in result.attrition:
            match = _expected_match_token(row)
            lines.append(
                f"    {row.cut_id}: n_in={row.n_in} n_passed={row.n_passed} "
                f"n_failed={row.n_failed} n_not_applicable={row.n_not_applicable} "
                f"n_out={row.n_out} expected_n_after={row.expected_n_after} "
                f"match={match}"
            )
            if row.not_applicable_reasons:
                reason_bits = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(row.not_applicable_reasons.items())
                )
                lines.append(f"      not_applicable_reasons: {reason_bits}")
            if row.skipped_for_mode:
                lines.append("      skipped_for_mode: true")
    lines.append("=== end sample_attrition_waterfall ===")
    return "\n".join(lines)


def _expected_match_token(row: CutAttrition) -> str:
    if row.expected_n_after is None:
        return "n/a"
    return "yes" if row.n_out == row.expected_n_after else "NO"


def attrition_bar_series(
    result: SampleEvaluationResult,
) -> tuple[list[str], dict[str, list[float]]]:
    """Labels + series for ``plot_grouped_bars`` (passed / failed / not_applicable)."""
    labels = [row.cut_id for row in result.attrition]
    return labels, {
        "passed": [float(row.n_passed) for row in result.attrition],
        "failed": [float(row.n_failed) for row in result.attrition],
        "not_applicable": [float(row.n_not_applicable) for row in result.attrition],
    }


def compare_to_published(
    *,
    sample_name: str,
    recovered_ids: Sequence[int],
    published_n: int | None,
    published_ids: Sequence[int] | None,
) -> ReproductionCompareResult:
    """Symmetric-difference report; N-only when no published ID table is configured."""
    recovered = tuple(sorted(int(s) for s in recovered_ids))
    if published_ids is None:
        n_match = published_n is not None and len(recovered) == published_n
        return ReproductionCompareResult(
            sample_name=sample_name,
            recovered_n=len(recovered),
            published_n=published_n,
            recovered_ids=recovered,
            published_ids=None,
            only_recovered=(),
            only_published=(),
            n_match=n_match,
            id_match=None,
            notes=(
                "published source-ID table unavailable; N-only comparison "
                f"(recovered_n={len(recovered)}, published_n={published_n})"
            ),
        )
    published = tuple(sorted(int(s) for s in published_ids))
    rec_set = set(recovered)
    pub_set = set(published)
    only_rec = tuple(sorted(rec_set - pub_set))
    only_pub = tuple(sorted(pub_set - rec_set))
    expected_n = published_n if published_n is not None else len(published)
    return ReproductionCompareResult(
        sample_name=sample_name,
        recovered_n=len(recovered),
        published_n=expected_n,
        recovered_ids=recovered,
        published_ids=published,
        only_recovered=only_rec,
        only_published=only_pub,
        n_match=len(recovered) == expected_n,
        id_match=not only_rec and not only_pub,
        notes="",
    )


def build_reproduction_comparisons(
    results: Mapping[str, SampleEvaluationResult],
    specs: Mapping[str, SampleSelectionFile],
    config: PipelineConfig,
    *,
    repo: Path | None = None,
) -> list[ReproductionCompareResult]:
    """Build per-sample and per-branch reproduction comparisons from config tables."""
    cfg = reproduction_cfg(config)
    out: list[ReproductionCompareResult] = []
    for name, result in sorted(results.items()):
        spec = specs.get(name)
        published_n = None if spec is None else spec.provenance.published_n
        table = cfg.published_tables.get(name)
        published_ids = (
            None if table is None else load_published_source_ids(table, repo=repo)
        )
        out.append(
            compare_to_published(
                sample_name=name,
                recovered_ids=result.surviving_source_ids,
                published_n=published_n,
                published_ids=published_ids,
            )
        )
        if name == "elbadry2026" and result.branch_surviving:
            by_branch = (
                {} if spec is None or spec.provenance.published_n_by_branch is None
                else spec.provenance.published_n_by_branch
            )
            for branch_id, ids in sorted(result.branch_surviving.items()):
                key = f"elbadry2026_{branch_id}"
                table_b = cfg.published_tables.get(key)
                out.append(
                    compare_to_published(
                        sample_name=key,
                        recovered_ids=ids,
                        published_n=by_branch.get(branch_id),
                        published_ids=(
                            None
                            if table_b is None
                            else load_published_source_ids(table_b, repo=repo)
                        ),
                    )
                )
            if result.subsample_surviving:
                for sub_id, ids in sorted(result.subsample_surviving.items()):
                    expected = None
                    if spec is not None and spec.branches:
                        for branch in spec.branches:
                            for sub in branch.subsamples or []:
                                if sub.id == sub_id:
                                    expected = sub.expected_n
                    out.append(
                        compare_to_published(
                            sample_name=f"elbadry2026_subsample_{sub_id}",
                            recovered_ids=ids,
                            published_n=expected,
                            published_ids=None,
                        )
                    )
            if result.route_counts and spec is not None:
                for branch in spec.branches or []:
                    if branch.expected_n_by_route is None:
                        continue
                    expected_routes = {
                        "main_sequence_min_companion_mass": (
                            branch.expected_n_by_route.main_sequence_min_companion_mass
                        ),
                        "high_mass_function": branch.expected_n_by_route.high_mass_function,
                        "both": branch.expected_n_by_route.both,
                    }
                    for route, expected in expected_routes.items():
                        got = int(result.route_counts.get(route, -1))
                        out.append(
                            ReproductionCompareResult(
                                sample_name=f"elbadry2026_route_{route}",
                                recovered_n=got,
                                published_n=expected,
                                recovered_ids=(),
                                published_ids=None,
                                only_recovered=(),
                                only_published=(),
                                n_match=got == expected,
                                id_match=None,
                                notes="route-count comparison (no source-ID set)",
                            )
                        )
    return out


def format_sample_reproduction_report(
    comparisons: Sequence[ReproductionCompareResult],
) -> str:
    """Full-detail recovered vs published membership report."""
    lines = [
        "=== sample_reproduction_report ===",
        "A sample's reproduction path is not considered working until recovered N",
        "matches the published N exactly (CONTINUATION_PLAN §13).",
    ]
    if not comparisons:
        lines.append("  (no comparisons; provide SampleEvaluationResult payloads)")
    for cmp in comparisons:
        lines.append(f"sample: {cmp.sample_name}")
        lines.append(f"  recovered_n: {cmp.recovered_n}")
        lines.append(f"  published_n: {cmp.published_n}")
        lines.append(f"  n_match: {cmp.n_match}")
        lines.append(f"  id_match: {cmp.id_match}")
        if cmp.notes:
            lines.append(f"  notes: {cmp.notes}")
        if cmp.only_recovered:
            lines.append(
                "  only_in_recovered ("
                f"{len(cmp.only_recovered)}): {list(cmp.only_recovered)}"
            )
        if cmp.only_published:
            lines.append(
                "  only_in_published ("
                f"{len(cmp.only_published)}): {list(cmp.only_published)}"
            )
        if cmp.id_match is True:
            lines.append("  symmetric_difference: empty")
    lines.append("=== end sample_reproduction_report ===")
    return "\n".join(lines)


def format_simon2026_exclusion_report(
    counts: Mapping[str, int],
    *,
    expected: Mapping[str, int] | None = None,
) -> str:
    """Full-detail Simon et al. (2026) exclusion breakdown (§8.9)."""
    lines = [
        "=== simon2026_exclusion_breakdown ===",
        "Acceptance target: 5 / 2 / 1 / 1 for the four exclusion reasons,",
        "plus 11 overlapping sources in sample (CONTINUATION_PLAN §8.9).",
    ]
    for key in (
        "in_sample",
        "sb1_fails_significance",
        "astrometric_f2_above_max",
        "fainter_than_g_limit",
        "fails_m2_over_m1",
        "unclassified",
    ):
        got = int(counts.get(key, 0))
        exp = None if expected is None else expected.get(key)
        match = "n/a" if exp is None else ("yes" if got == exp else "NO")
        lines.append(f"  {key}: {got} expected={exp} match={match}")
    lines.append("=== end simon2026_exclusion_breakdown ===")
    return "\n".join(lines)


def run_simon2026_diagnostic(
    config: PipelineConfig,
    *,
    spec: SampleSelectionFile | None = None,
    sample_ids: Sequence[int] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    repo: Path | None = None,
) -> tuple[dict[str, int], str]:
    """Compute Simon §8.9 breakdown and format the report."""
    cfg = reproduction_cfg(config)
    simon_rows = (
        list(rows)
        if rows is not None
        else load_simon2026_orbital(cfg.simon2026_table)
    )
    if spec is None:
        path = _resolve_path("config/selections/elbadry2026.yaml", repo=repo)
        spec = load_sample_selection_file(path)
    ids = list(sample_ids) if sample_ids is not None else []
    counts = simon2026_exclusion_breakdown(simon_rows, ids, spec)
    expected_block = (
        None
        if spec.acceptance_tests is None
        or spec.acceptance_tests.simon2026_exclusion_breakdown is None
        else spec.acceptance_tests.simon2026_exclusion_breakdown
    )
    expected = None
    if expected_block is not None:
        expected = {
            "sb1_fails_significance": expected_block.sb1_fails_significance,
            "astrometric_f2_above_max": expected_block.astrometric_f2_above_max,
            "fainter_than_g_limit": expected_block.fainter_than_g_limit,
            "fails_m2_over_m1": expected_block.fails_m2_over_m1,
        }
    return counts, format_simon2026_exclusion_report(counts, expected=expected)


def format_covariance_health_report(health: CovarianceHealth) -> str:
    """Full-detail missing / non-PSD covariance funnel (§13)."""
    lines = [
        "=== covariance_health ===",
        "Counts of NSS solutions with missing / non-PSD covariance by failure mode.",
        "Silent diagonal fallback is forbidden; every failure is reported.",
        f"  covariance_ok: {health.ok}",
        f"  covariance_failed: {health.failed}",
        f"  missing_corr: {health.missing_corr}",
        f"  unsupported_solution_type: {health.unsupported_solution_type}",
        f"  bit_index_mismatch: {health.bit_index_mismatch}",
        f"  unpack_failed: {health.unpack_failed}",
        f"  missing_values: {health.missing_values}",
        f"  missing_errors: {health.missing_errors}",
        f"  non_symmetric: {health.non_symmetric}",
        f"  non_psd: {health.non_psd}",
        "  by_solution_type:",
    ]
    type_lines = health.by_type_lines()
    if type_lines:
        lines.extend(f"    {line}" for line in type_lines)
    else:
        lines.append("    (none)")
    lines.append("=== end covariance_health ===")
    return "\n".join(lines)


def evaluate_selection_function_curves(
    selection: SampleSelection,
    config: PipelineConfig,
    *,
    membership: Mapping[str, frozenset[int]] | None = None,
) -> list[SelectionFunctionCurve]:
    """Forward-model survival probability vs M2, P_orb, G for one sample."""
    grid = reproduction_cfg(config).selection_function
    template = dict(grid.template)
    curves: list[SelectionFunctionCurve] = []

    def _survival_for(axis_key: str, values: Sequence[float], row_keys: Sequence[str]) -> SelectionFunctionCurve:
        surv: list[float] = []
        for value in values:
            row = dict(template)
            row["source_id"] = 1
            for key in row_keys:
                row[key] = float(value) if key != "nss_solution_type" else value
            if axis_key == "m2_msun":
                # Keep probability / ratio columns consistent with the swept mass.
                row["m2_msun"] = float(value)
                row["m2_tilde_msun"] = float(value)
                row["m2_min_msun"] = float(value)
                row["p_m2_above"] = 1.0 if float(value) > 1.4 else 0.0
                row["p_m2_gt_threshold"] = row["p_m2_above"]
            result = selection.evaluate([row], membership=membership)
            surv.append(1.0 if result.n_surviving == 1 else 0.0)
        return SelectionFunctionCurve(
            axis=axis_key,
            x=[float(v) for v in values],
            survival=surv,
            sample_name=selection.spec.name,
        )

    curves.append(
        _survival_for(
            "m2_msun",
            grid.m2_msun,
            ("m2_msun", "m2_tilde_msun", "m2_min_msun"),
        )
    )
    curves.append(
        _survival_for("period_day", grid.period_day, ("period_day", "period"))
    )
    curves.append(
        _survival_for("g_mag", grid.g_mag, ("phot_g_mean_mag", "g_mag"))
    )
    return curves


def format_sample_selection_function_report(
    curves: Sequence[SelectionFunctionCurve],
) -> str:
    """Full-detail forward-model survival curves report."""
    lines = [
        "=== sample_selection_function ===",
        "Forward-model path: survival probability vs M2, P_orb, G.",
        "Expect smooth, monotonic behaviour where physically expected (§13).",
    ]
    if not curves:
        lines.append("  (no curves)")
    for curve in curves:
        lines.append(f"sample: {curve.sample_name}  axis: {curve.axis}")
        for x, s in zip(curve.x, curve.survival, strict=True):
            lines.append(f"  {curve.axis}={x:g} -> survival={s:g}")
        if len(curve.survival) >= 2:
            deltas = [
                curve.survival[i + 1] - curve.survival[i]
                for i in range(len(curve.survival) - 1)
            ]
            lines.append(f"  successive_deltas: {[float(d) for d in deltas]}")
    lines.append("=== end sample_selection_function ===")
    return "\n".join(lines)


def compute_mode_divergence(
    left: SampleEvaluationResult,
    right: SampleEvaluationResult,
    *,
    expected_only_left: Sequence[int] = (),
    expected_only_right: Sequence[int] = (),
) -> ModeDivergenceResult:
    """Quantify recovered-set divergence (Andrews §6.6: exactly Gaia BH1)."""
    left_ids = tuple(sorted(int(s) for s in left.surviving_source_ids))
    right_ids = tuple(sorted(int(s) for s in right.surviving_source_ids))
    only_left = tuple(sorted(set(left_ids) - set(right_ids)))
    only_right = tuple(sorted(set(right_ids) - set(left_ids)))
    exp_l = tuple(sorted(int(s) for s in expected_only_left))
    exp_r = tuple(sorted(int(s) for s in expected_only_right))
    matches = only_left == exp_l and only_right == exp_r
    if left.name == "andrews2022" and right.name == "andrews2022_modified":
        explanation = (
            "andrews2022 applies the published harmonic exclusion; "
            "andrews2022_modified drops it and restores Gaia BH1 "
            f"({exp_r[0] if exp_r else 'configured source'}). "
            "Divergence is the exclusion cut, not the primary-mass assumption."
        )
    else:
        explanation = (
            f"mass_source left={left.mass_source} ({left.mode.value}) vs "
            f"right={right.mass_source} ({right.mode.value}); "
            "divergence attributed to mode / mass-assumption difference."
        )
    return ModeDivergenceResult(
        left_name=left.name,
        right_name=right.name,
        left_ids=left_ids,
        right_ids=right_ids,
        only_left=only_left,
        only_right=only_right,
        expected_only_left=exp_l,
        expected_only_right=exp_r,
        matches_expectation=matches,
        explanation=explanation,
    )


def format_mode_divergence_report(
    divergences: Sequence[ModeDivergenceResult],
) -> str:
    """Full-detail mode / variant divergence report."""
    lines = [
        "=== mode_divergence ===",
        "Compares recovered sets across modes or named variants.",
        "andrews2022 vs andrews2022_modified must differ by exactly Gaia BH1 (§6.6).",
    ]
    if not divergences:
        lines.append("  (no pairs evaluated)")
    for div in divergences:
        lines.append(f"pair: {div.left_name} vs {div.right_name}")
        lines.append(f"  left_n: {len(div.left_ids)}")
        lines.append(f"  right_n: {len(div.right_ids)}")
        lines.append(f"  only_in_{div.left_name}: {list(div.only_left)}")
        lines.append(f"  only_in_{div.right_name}: {list(div.only_right)}")
        lines.append(f"  expected_only_left: {list(div.expected_only_left)}")
        lines.append(f"  expected_only_right: {list(div.expected_only_right)}")
        lines.append(f"  matches_expectation: {div.matches_expectation}")
        lines.append(f"  explanation: {div.explanation}")
    lines.append("=== end mode_divergence ===")
    return "\n".join(lines)


def compute_janssens_segment_occupancy(
    mg_0_values: Sequence[float] | NDArray[np.floating],
    *,
    table_path: str | Path | None = None,
) -> JanssensOccupancyResult:
    """Count El-Badry 2026 sources per Janssens mass segment (§8.2 / §13)."""
    table = load_janssens_table(table_path)
    segments = segments_from_table(table)
    segment_counts: list[dict[str, Any]] = [
        {
            "segment_index": i,
            "m_low": seg.m_low,
            "m_up": seg.m_up,
            "count": 0,
            "uninformative": (
                abs(seg.m_low - UNINFORMATIVE_SEGMENT_MLOW) < 1e-9
                and abs(seg.m_up - UNINFORMATIVE_SEGMENT_MUP) < 1e-9
            ),
        }
        for i, seg in enumerate(segments)
    ]
    n_boundary = 0
    n_oor = 0
    n_missing = 0
    n_uninformative = 0
    values = np.asarray(mg_0_values, dtype=np.float64).ravel()
    for mg in values:
        result = invert_mg_to_mass(float(mg), table=table, segments=segments)
        if result.reason == "missing_mg":
            n_missing += 1
            continue
        if result.mass_msun is None:
            n_oor += 1
            continue
        assert result.segment_index is not None
        segment_counts[result.segment_index]["count"] += 1
        if result.boundary_resolved:
            n_boundary += 1
        if result.uninformative_segment:
            n_uninformative += 1
    return JanssensOccupancyResult(
        segment_counts=segment_counts,
        n_boundary_resolved=n_boundary,
        n_out_of_range=n_oor,
        n_missing_mg=n_missing,
        n_total=int(values.size),
        uninformative_segment_flagged=n_uninformative > 0,
        uninformative_segment_count=n_uninformative,
    )


def format_janssens_segment_occupancy_report(
    occupancy: JanssensOccupancyResult,
) -> str:
    """Full-detail Janssens segment occupancy report."""
    lines = [
        "=== janssens_segment_occupancy ===",
        "Per-segment counts for El-Badry 2026 M̃1 (Janssens et al. 2022 Table 1).",
        "Boundary-resolved and out-of-range counts are never silent (§8.2 / §13).",
        f"  n_total: {occupancy.n_total}",
        f"  n_boundary_resolved: {occupancy.n_boundary_resolved}",
        f"  n_out_of_range: {occupancy.n_out_of_range}",
        f"  n_missing_mg: {occupancy.n_missing_mg}",
        f"  uninformative_segment_count (1.55–1.80 Msun): "
        f"{occupancy.uninformative_segment_count}",
        f"  uninformative_segment_flagged: {occupancy.uninformative_segment_flagged}",
        "  segments:",
    ]
    for row in occupancy.segment_counts:
        flag = "  [NEAR-UNINFORMATIVE]" if row["uninformative"] else ""
        lines.append(
            f"    [{row['segment_index']}] "
            f"{row['m_low']:g}–{row['m_up']:g} Msun: count={row['count']}{flag}"
        )
    lines.append("=== end janssens_segment_occupancy ===")
    return "\n".join(lines)


def load_specs_for_results(
    results: Mapping[str, SampleEvaluationResult],
    config: PipelineConfig,
    *,
    repo: Path | None = None,
) -> dict[str, SampleSelectionFile]:
    """Load inherit-resolved selection YAML for each named result.

    Loads every registry file first so ``inherits`` / ``depends_on`` targets
    resolve (e.g. ``andrews2022_modified`` → ``andrews2022``). Unresolved
    inherits must never reach ``SampleSelection()``.
    """
    root = repo if repo is not None else repo_root()
    files_by_name: dict[str, SampleSelectionFile] = {}
    for entry in config.sample_selection.samples:
        path = Path(entry.path)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            files_by_name[entry.name] = load_sample_selection_file(path)

    out: dict[str, SampleSelectionFile] = {}
    for name in results:
        if name not in files_by_name:
            continue
        out[name] = resolve_inherits(files_by_name[name], files_by_name=files_by_name)
    return out


def forward_model_selection(
    spec: SampleSelectionFile,
) -> SampleSelection:
    """Build a forward-model-mode evaluator for survival-function sweeps."""
    return SampleSelection(spec, mode=SampleSelectionMode.FORWARD_MODEL)
