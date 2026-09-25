"""Measure the empirical multi-solution ``source_id`` rate on the uncut NSS parent snapshot.

Issue #241 (measurement-only, umbrella #236, references #237/#231). Characterizes, on the
documented uncut NSS parent snapshot ``20260826T234425Z_3d3f740b080c`` (``CLAUDE.md`` Gotchas —
the quality-cut snapshot under-counts the parent and must not be substituted here):

1. Cross-``nss_solution_type`` multiplicity: which solution-type combinations co-occur for the
   same ``source_id`` (e.g. ``Orbital`` + ``AstroSpectroSB1``), with counts per combination.
2. Same-type multiplicity (period aliasing): sources with more than one ``Orbital``-family
   solution, counted separately from (1).
3. A descriptive breakdown of each sub-case conditioned on period, RUWE, goodness of fit, and
   G magnitude — the covariates actually present in this snapshot's schema (see "Limitations").
4. A source-ID-level report (counts + worked examples) for issue #242 (tag-and-keep) to validate
   against.

This script is read-only over the snapshot cache and writes nothing back into ``data_acquisition``,
``forward_model``, or any ``config/selections/*.yaml`` file. It does not build a parametric model —
descriptive counts and conditional summary statistics only, per #241's "Do not" list.

Distinct from the 6-group cross-match fan-out already handled by ``assert_unique_source_ids`` /
``DuplicateSourceIdError`` (#221, PR #240): those 6 groups share identical ``nss_solution_type`` AND
``period`` within the group (differing only in cross-matched external photometry) and are excluded
from the multi-solution counts here, then reported separately as a cross-check against the
documented figure.

Limitations
-----------
The uncut snapshot's ADQL (``data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml``)
does not select an RV-epoch-count column (e.g. ``rv_nb_transits``) — it was never queried, so it is
not "available but unused" here, it is simply absent from this snapshot's schema. The covariate
breakdown below uses period, RUWE, ``goodness_of_fit``, and G magnitude only; RV epoch count is
omitted for this reason rather than silently treated as zero or dropped without comment.

Usage
-----
    .venv/bin/python scripts/measure_multi_solution_rate.py
    .venv/bin/python scripts/measure_multi_solution_rate.py --no-figures
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from darkhunter_pop.config_loader import load_config, repo_root

UNCUT_SNAPSHOT_DIR = "20260826T234425Z_3d3f740b080c"
UNCUT_SNAPSHOT_ID = "20260826T234425Z_3d3f740b080c"
ORBITAL_FAMILY_PREFIX = "Orbital"


def _load_json(raw: bytes | str) -> dict[str, Any]:
    import json

    return json.loads(raw)


@dataclass
class SourceGroup:
    source_id: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def solution_types(self) -> tuple[str, ...]:
        return tuple(row["nss_solution_type"] for row in self.rows)

    @property
    def unique_types(self) -> set[str]:
        return set(self.solution_types)

    @property
    def periods(self) -> tuple[float | None, ...]:
        return tuple(row.get("period") for row in self.rows)

    @property
    def is_fanout(self) -> bool:
        """Same ``nss_solution_type`` AND same ``period`` across every row in the group.

        This is the #221/PR #240 cross-match fan-out signature (photometry-only difference),
        not a genuine multi-solution duplicate, and is excluded from the counts here.
        """
        if len(self.rows) < 2:
            return False
        types = self.unique_types
        if len(types) != 1:
            return False
        periods = [p for p in self.periods if p is not None]
        if len(periods) != len(self.rows):
            return False
        return len(set(periods)) == 1

    @property
    def is_cross_type(self) -> bool:
        return len(self.unique_types) > 1

    @property
    def is_orbital_family(self) -> bool:
        return all(t.startswith(ORBITAL_FAMILY_PREFIX) for t in self.unique_types)

    @property
    def is_same_type_period_aliased(self) -> bool:
        """More than one Orbital-family solution at (necessarily) different periods."""
        if self.is_fanout:
            return False
        orbital_rows = [r for r in self.rows if r["nss_solution_type"].startswith(ORBITAL_FAMILY_PREFIX)]
        if len(orbital_rows) < 2:
            return False
        periods = {r.get("period") for r in orbital_rows}
        return len(periods) > 1


def load_uncut_snapshot_rows() -> list[dict[str, Any]]:
    """Load every NSS row from the uncut parent snapshot's selection-parent cache.

    ``selection_parent_rows.h5`` sits beside ``meta.yaml`` in the same snapshot directory and is
    built by ``sample_selection.load_selection_rows_from_uncut_snapshot`` directly from this
    snapshot's ``query.ecsv`` with no DA quality cuts and no enrichment/MC merge (that variant
    lives under the separate ``+enrich`` / ``+enrich+mc10000`` directories) — row count matches
    ``meta.yaml``'s ``row_count`` (443211) exactly, confirmed at script-writing time, so this is a
    1:1, non-collapsing view of the raw uncut snapshot rather than a derived/filtered product.
    """
    snapshot_dir = repo_root() / "data" / "dr3" / "gaia_snapshots" / UNCUT_SNAPSHOT_DIR
    meta_path = snapshot_dir / "meta.yaml"
    cache_path = snapshot_dir / "selection_parent_rows.h5"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"uncut parent snapshot meta not found at {meta_path}; per issue #241, do not "
            "substitute a different (quality-cut) snapshot — stop and escalate instead."
        )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"uncut parent snapshot row cache not found at {cache_path}; per issue #241, do not "
            "substitute a different (quality-cut) snapshot — stop and escalate instead."
        )
    rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as handle:
        strings = handle["rows_json"].asstr()
        for raw in strings:
            rows.append(_load_json(raw))
    return rows


def group_by_source(rows: list[dict[str, Any]]) -> dict[int, SourceGroup]:
    groups: dict[int, SourceGroup] = {}
    for row in rows:
        sid = int(row["source_id"])
        groups.setdefault(sid, SourceGroup(source_id=sid)).rows.append(row)
    return groups


def _finite(values: list[float | None]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _quantile_summary(arr: np.ndarray) -> dict[str, float | int]:
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def covariate_summary_g_mag(rows: list[dict[str, Any]]) -> np.ndarray:
    return _finite([r.get("g_mag", r.get("phot_g_mean_mag")) for r in rows])


def covariate_summary_period(rows: list[dict[str, Any]]) -> np.ndarray:
    return _finite([r.get("period") for r in rows])


def covariate_summary_ruwe(rows: list[dict[str, Any]]) -> np.ndarray:
    return _finite([r.get("ruwe") for r in rows])


def covariate_summary_gof(rows: list[dict[str, Any]]) -> np.ndarray:
    return _finite([r.get("goodness_of_fit") for r in rows])


COVARIATES = {
    "period_day": covariate_summary_period,
    "ruwe": covariate_summary_ruwe,
    "goodness_of_fit": covariate_summary_gof,
    "g_mag": covariate_summary_g_mag,
}


def build_rate_table(
    all_rows: list[dict[str, Any]],
    groups: dict[int, SourceGroup],
) -> dict[str, Any]:
    """Build the machine-readable empirical rate table consumed by ``forward_model.py``.

    Issue #243 depends on #241's measurement being available as a config-referenced,
    machine-readable artifact rather than only the prose ``REPORT.md`` table. This is the
    *same* partitioning logic as :func:`build_report` (``SourceGroup.is_cross_type`` /
    ``.is_same_type_period_aliased``) restated as a dict that round-trips to YAML — no new
    numbers, no re-derivation with different logic.
    """
    n_rows = len(all_rows)
    n_sources = len(groups)
    multi_row_groups = {sid: g for sid, g in groups.items() if len(g.rows) > 1}
    fanout_groups = {sid: g for sid, g in multi_row_groups.items() if g.is_fanout}
    genuine_groups = {sid: g for sid, g in multi_row_groups.items() if not g.is_fanout}
    cross_type_groups = {sid: g for sid, g in genuine_groups.items() if g.is_cross_type}
    same_type_groups = {
        sid: g for sid, g in genuine_groups.items() if g.is_same_type_period_aliased
    }
    both_groups = {sid for sid in cross_type_groups if sid in same_type_groups}

    combo_counts: Counter[tuple[str, ...]] = Counter()
    for g in cross_type_groups.values():
        combo_counts[tuple(sorted(g.unique_types))] += 1

    mult_dist: Counter[int] = Counter(len(g.rows) for g in genuine_groups.values())

    return {
        "schema_version": 1,
        "source_snapshot_id": UNCUT_SNAPSHOT_ID,
        "provenance": (
            "scripts/measure_multi_solution_rate.py (issue #241); descriptive counts, "
            "no fitted model; see docs/multi_solution_characterization/REPORT.md"
        ),
        "n_rows": n_rows,
        "n_sources": n_sources,
        "n_multi_row_sources": len(multi_row_groups),
        "n_fanout_sources": len(fanout_groups),
        "n_cross_type_sources": len(cross_type_groups),
        "n_same_type_period_aliased_sources": len(same_type_groups),
        "n_both_sources": len(both_groups),
        "cross_type_combo_counts": {
            "+".join(combo): count for combo, count in combo_counts.most_common()
        },
        "row_multiplicity_distribution": {int(k): int(v) for k, v in sorted(mult_dist.items())},
    }


def build_report(
    all_rows: list[dict[str, Any]],
    groups: dict[int, SourceGroup],
) -> tuple[str, dict[str, np.ndarray]]:
    n_rows = len(all_rows)
    n_sources = len(groups)
    multi_row_groups = {sid: g for sid, g in groups.items() if len(g.rows) > 1}
    n_multi_row_sources = len(multi_row_groups)

    fanout_groups = {sid: g for sid, g in multi_row_groups.items() if g.is_fanout}
    genuine_groups = {sid: g for sid, g in multi_row_groups.items() if not g.is_fanout}

    cross_type_groups = {sid: g for sid, g in genuine_groups.items() if g.is_cross_type}
    same_type_groups = {
        sid: g for sid, g in genuine_groups.items() if g.is_same_type_period_aliased
    }
    # A group can be in both (e.g. 3-row group with two Orbital periods + one SB1 row).
    both_groups = {sid for sid in cross_type_groups if sid in same_type_groups}

    combo_counts: Counter[tuple[str, ...]] = Counter()
    for g in cross_type_groups.values():
        combo_counts[tuple(sorted(g.unique_types))] += 1

    # Multiplicity distribution (row count per source) over genuine multi-solution groups.
    mult_dist: Counter[int] = Counter(len(g.rows) for g in genuine_groups.values())

    # Covariate summaries: affected (any genuine multi-solution row) vs unaffected (single-row
    # sources), plus cross-type vs same-type sub-populations, each restricted to the rows that
    # belong to that sub-case's groups.
    single_row_ids = {sid for sid, g in groups.items() if len(g.rows) == 1}
    single_rows = [groups[sid].rows[0] for sid in single_row_ids]
    genuine_rows: list[dict[str, Any]] = [r for g in genuine_groups.values() for r in g.rows]
    cross_type_rows: list[dict[str, Any]] = [r for g in cross_type_groups.values() for r in g.rows]
    same_type_rows: list[dict[str, Any]] = [r for g in same_type_groups.values() for r in g.rows]

    covariate_arrays: dict[str, np.ndarray] = {}
    covariate_lines = []
    for name, fn in COVARIATES.items():
        single_arr = fn(single_rows)
        genuine_arr = fn(genuine_rows)
        cross_arr = fn(cross_type_rows)
        same_arr = fn(same_type_rows)
        covariate_arrays[f"{name}__single"] = single_arr
        covariate_arrays[f"{name}__genuine_multi"] = genuine_arr
        covariate_arrays[f"{name}__cross_type"] = cross_arr
        covariate_arrays[f"{name}__same_type"] = same_arr
        s_single = _quantile_summary(single_arr)
        s_genuine = _quantile_summary(genuine_arr)
        s_cross = _quantile_summary(cross_arr)
        s_same = _quantile_summary(same_arr)
        covariate_lines.append(
            f"| `{name}` "
            f"| n={s_single['n']}, median={s_single['median']:.4g} "
            f"(16/84%: {s_single['p16']:.4g}/{s_single['p84']:.4g}) "
            f"| n={s_genuine['n']}, median={s_genuine['median']:.4g} "
            f"(16/84%: {s_genuine['p16']:.4g}/{s_genuine['p84']:.4g}) "
            f"| n={s_cross['n']}, median={s_cross['median']:.4g} "
            f"(16/84%: {s_cross['p16']:.4g}/{s_cross['p84']:.4g}) "
            f"| n={s_same['n']}, median={s_same['median']:.4g} "
            f"(16/84%: {s_same['p16']:.4g}/{s_same['p84']:.4g}) |"
        )

    # Worked examples: smallest source_id per sub-case, deterministic across runs.
    def _example(group_map: dict[int, SourceGroup]) -> str:
        if not group_map:
            return "(none)"
        sid = min(group_map)
        g = group_map[sid]
        parts = [
            f"source_id={sid}, n_rows={len(g.rows)}, "
            f"types={list(g.solution_types)}, periods={list(g.periods)}"
        ]
        return parts[0]

    lines: list[str] = []
    lines.append("# Multi-solution NSS source_id characterization")
    lines.append("")
    lines.append(
        "Measured on the uncut NSS parent snapshot "
        f"`{UNCUT_SNAPSHOT_ID}` (issue #241). Descriptive counts only — no fitted model, "
        "no change to `data_acquisition.py`, `forward_model.py`, or any "
        "`config/selections/*.yaml` threshold. See `scripts/measure_multi_solution_rate.py` "
        "for the exact partitioning logic (`SourceGroup.is_fanout` / `.is_cross_type` / "
        "`.is_same_type_period_aliased`)."
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Total NSS rows in the uncut snapshot: **{n_rows}**")
    lines.append(f"- Total distinct `source_id`s: **{n_sources}**")
    lines.append(f"- `source_id`s carrying more than one row: **{n_multi_row_sources}**")
    lines.append(
        f"- Of those, cross-match fan-out (same type + same period, #221/PR #240; excluded "
        f"from the multi-solution counts below): **{len(fanout_groups)}**"
    )
    lines.append(
        f"- Genuine multi-solution `source_id`s (this ticket's subject): "
        f"**{len(genuine_groups)}**"
    )
    lines.append(
        f"  - Cross-type multiplicity (distinct `nss_solution_type` families co-occurring): "
        f"**{len(cross_type_groups)}**"
    )
    lines.append(
        f"  - Same-type multiplicity / period aliasing (>1 Orbital-family solution): "
        f"**{len(same_type_groups)}**"
    )
    lines.append(
        f"  - `source_id`s exhibiting both sub-cases at once: **{len(both_groups)}**"
    )
    lines.append("")
    lines.append(
        "Cross-check against `docs/ARCHITECTURE.md` §4: documented as \"5,926 of 5,932 "
        "duplicated source_ids are distinct NSS orbital solutions ... the remaining 6 groups "
        "(0.1%) are genuine cross-match fan-out\" — measured here: "
        f"{n_multi_row_sources} duplicated source_ids total, {len(fanout_groups)} fan-out, "
        f"{len(genuine_groups)} genuine."
    )
    lines.append("")
    lines.append("## Row-count multiplicity distribution (genuine multi-solution groups)")
    lines.append("")
    lines.append("| rows per source_id | count of source_ids |")
    lines.append("|---|---|")
    for k in sorted(mult_dist):
        lines.append(f"| {k} | {mult_dist[k]} |")
    lines.append("")
    lines.append("## Top co-occurring `nss_solution_type` combinations (cross-type only)")
    lines.append("")
    lines.append("| combination | source_id count |")
    lines.append("|---|---|")
    for combo, count in combo_counts.most_common():
        lines.append(f"| {' + '.join(combo)} | {count} |")
    lines.append("")
    lines.append("## Covariate breakdown (descriptive, not a fitted model)")
    lines.append("")
    lines.append(
        "RV epoch count is not in this breakdown: the uncut snapshot's ADQL "
        "(`data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml`) never selects an "
        "RV-epoch-count column, so it is absent from the schema rather than merely unused."
    )
    lines.append("")
    lines.append(
        "| covariate | single-row sources | all genuine multi-solution rows "
        "| cross-type rows | same-type (period-aliased) rows |"
    )
    lines.append("|---|---|---|---|---|")
    lines.extend(covariate_lines)
    lines.append("")
    lines.append("## Worked examples (smallest `source_id` per sub-case, deterministic)")
    lines.append("")
    lines.append(f"- Cross-type: {_example(cross_type_groups)}")
    lines.append(f"- Same-type / period aliasing: {_example(same_type_groups)}")
    lines.append(f"- Cross-match fan-out (excluded, for contrast): {_example(fanout_groups)}")
    if both_groups:
        both_map = {sid: genuine_groups[sid] for sid in both_groups}
        lines.append(f"- Both sub-cases at once: {_example(both_map)}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Descriptive only; #242 (tag-and-keep) and the eventual `forward_model.py` "
        "multiplicity-emission model (blocked-by this ticket, per the issue) still need to "
        "translate this into pipeline behavior — not done here."
    )
    lines.append(
        "- RUWE, `goodness_of_fit`, and G magnitude are as reported per-row in the uncut "
        "snapshot; for multi-row sources these differ row-to-row (e.g. an `Orbital` row and an "
        "`SB1` row for the same source can carry different `goodness_of_fit`), so the covariate "
        "table above summarizes rows, not deduplicated sources — stated explicitly so #242 does "
        "not misread it as a per-source statistic."
    )
    lines.append(
        "- RV epoch count is absent from this snapshot's schema (see above) and is not "
        "reconstructable from the columns queried; a future measurement would need a new ADQL "
        "column, out of scope here."
    )
    if not same_type_groups:
        lines.append(
            "- Measured **zero** same-type (period-aliasing) duplicates in this uncut snapshot "
            "under the strict definition (>1 row whose `nss_solution_type` *starts with* "
            "`Orbital` within one `source_id` group, at differing periods). Every genuine "
            "multi-solution `source_id` observed here co-occurs with at least one distinct "
            "solution-type family (cross-type). This is an empirical result for this snapshot, "
            "not evidence the phenomenon cannot occur — #242 should not assume it is structurally "
            "impossible, only that it was not observed in the ~443k rows measured."
        )
    return "\n".join(lines) + "\n", covariate_arrays


def emit_figures(covariate_arrays: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    from darkhunter_pop.plotting import matplotlib_available, plot_overlay_histograms

    if not matplotlib_available():
        print("matplotlib unavailable; skipping figures", file=sys.stderr)
        return []
    config = load_config()
    style = config.plotting
    dpi = config.diagnostics.figure_dpi
    max_bins = config.diagnostics.histogram_max_bins
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "period_day": ("period (day)", "Period: single-row vs multi-solution sources"),
        "ruwe": ("RUWE", "RUWE: single-row vs multi-solution sources"),
        "goodness_of_fit": (
            "goodness of fit",
            "Goodness of fit: single-row vs multi-solution sources",
        ),
        "g_mag": ("G magnitude", "G magnitude: single-row vs multi-solution sources"),
    }
    written: list[Path] = []
    for name, (xlabel, title) in labels.items():
        series = {
            "single-row": covariate_arrays[f"{name}__single"],
            "cross-type multi-solution": covariate_arrays[f"{name}__cross_type"],
            "same-type (period-aliased)": covariate_arrays[f"{name}__same_type"],
        }
        path = out_dir / f"{name}_by_multiplicity.png"
        result = plot_overlay_histograms(
            series,
            path,
            xlabel=xlabel,
            title=title,
            dpi=dpi,
            max_bins=max_bins,
            style=style,
        )
        if result is not None:
            written.append(result)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true", help="Skip matplotlib figure output")
    parser.add_argument(
        "--report-path",
        default=str(repo_root() / "docs" / "multi_solution_characterization" / "REPORT.md"),
    )
    parser.add_argument(
        "--figures-dir",
        default=str(repo_root() / "docs" / "multi_solution_characterization"),
    )
    parser.add_argument(
        "--rates-path",
        default=str(repo_root() / "config" / "multi_solution_rates.yaml"),
        help=(
            "Machine-readable rate table consumed by forward_model.py's multi-solution "
            "emission model (issue #243). Set to empty string to skip."
        ),
    )
    args = parser.parse_args(argv)

    rows = load_uncut_snapshot_rows()
    groups = group_by_source(rows)
    report_text, covariate_arrays = build_report(rows, groups)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"wrote report: {report_path}")

    if args.rates_path:
        import yaml

        rate_table = build_rate_table(rows, groups)
        rates_path = Path(args.rates_path)
        rates_path.parent.mkdir(parents=True, exist_ok=True)
        rates_path.write_text(
            "# Generated by scripts/measure_multi_solution_rate.py (issue #241).\n"
            "# Consumed by forward_model.py's multi-solution emission model (issue #243)\n"
            "# via PipelineConfig.multi_solution_rates.path. Do not hand-edit; re-run the\n"
            "# script against the uncut snapshot to regenerate.\n"
            + yaml.safe_dump(rate_table, sort_keys=False),
            encoding="utf-8",
        )
        print(f"wrote rate table: {rates_path}")

    if not args.no_figures:
        written = emit_figures(covariate_arrays, Path(args.figures_dir))
        for path in written:
            print(f"wrote figure: {path}")

    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
