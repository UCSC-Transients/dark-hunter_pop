#!/usr/bin/env python3
"""Calibrate and validate the GSP-Phot log g emulator on real DR3 rows (#680, RUNG5_SPEC §2.7.4).

Steps (definition: ``config/population/gspphot_logg_emulator_680.yaml``):

1. Load the uncut NSS snapshot; keep ``calibration.solution_types``; build ``(G, BP-RP, M_G)`` and log g.
2. Equivalence check: on every row of the VizieR I/355 snapshot the Andrews chain reads, ``logg_vizier_apsis``
   must equal ``logg_gspphot`` (same missing pattern; values within half the published 1e-4 dex, i.e. float32
   rounding of the archive column). Refuses to continue otherwise.
3. Calibration population: those rows, restricted to the VizieR snapshot's IDs when
   ``restrict_to_equivalence_ids``. Split train / holdout by a hash of ``(split_seed, source_id)``; calibrate the
   cells on the TRAIN half only and write the table (``--table-out``) with provenance and the definition SHA-256.
4. Validate on the holdout (gated): whole set, Orbital, per coarse CMD region, per G bin: real vs emulated undefined
   rate and giant-reject rate (defined and logg < the frozen ``logg_dwarf_min``), with pulls. The rows OUTSIDE the
   calibration population (never used for training) get the same tables, informational only (population shift).
   Write the markdown report (``--report-out``); exit 1 if the gate fails (the table is still written).

Peak RSS ~3.8 GB (full ECSV read): run under ``scripts/jobs/run_heavy.sh``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.andrews2022_atf import load_vizier_apsis
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.data_acquisition import load_gaia_snapshot
from darkhunter_pop.gspphot_logg_emulator import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TABLE_PATH,
    EmulatorDefinition,
    GspPhotLoggEmulator,
    abs_g_mag,
    build_table,
    calibrate_levels,
    cmd_region_labels,
    definition_sha256,
    g_bin_labels,
    holdout_mask,
    load_definition,
    markdown_table,
    selection_threshold,
    table_to_yaml,
    validation_rows,
)
from darkhunter_pop.run_management import short_git_commit


def _col(table, name: str) -> NDArray[np.float64]:
    return np.ma.filled(np.ma.asarray(table[name], dtype=np.float64), np.nan)


def _sections(
    d: EmulatorDefinition, emu: GspPhotLoggEmulator, sel: NDArray[np.bool_], train: NDArray[np.bool_],
    feats: dict[str, NDArray[np.float64]], logg: NDArray[np.float64], stype: NDArray[np.str_],
    sid: NDArray[np.int64], in_viz: NDArray[np.bool_], thr: float, title: str,
) -> tuple[list[str], bool]:
    """Markdown tables for rows ``sel`` (evaluated) against the train rows ``train``; returns (lines, gate ok)."""
    vs = d.validation
    fh = {k: x[sel] for k, x in feats.items()}
    emulated = emu.emulate(fh["g_mag"], fh["bp_rp"], fh["abs_g_mag"], generation=0, draw_index=sid[sel])
    reg_h, reg_t = (cmd_region_labels(f["bp_rp"], f["abs_g_mag"], vs) for f in (fh, {k: x[train] for k, x in feats.items()}))
    gb_h, gb_t = g_bin_labels(fh["g_mag"], vs), g_bin_labels(feats["g_mag"][train], vs)

    def by(lab_h: NDArray[np.str_], lab_t: NDArray[np.str_]) -> dict:
        return {n: (lab_h == n, int((lab_t == n).sum())) for n in sorted(set(lab_h.tolist()))}

    st_h = stype[sel]
    groups = [
        ("All rows / Orbital (Andrews parent) / Andrews ATF rows", {
            "all": (np.ones(int(sel.sum()), dtype=bool), int(train.sum())),
            "Orbital": (st_h == "Orbital", int((stype[train] == "Orbital").sum())),
            "Andrews ATF rows (VizieR I/355)": (in_viz[sel], int((in_viz & train).sum())),
        }),
        ("Per CMD region (observed BP-RP, M_G)", by(reg_h, reg_t)),
        ("Per G bin", by(gb_h, gb_t)),
    ]
    nlev = len(d.binning.fallback_levels)
    lines = [f"## {title}", "",
             f"Fallback level used ({', '.join(f'{k} = {lv.name}' for k, lv in enumerate(d.binning.fallback_levels))}): "
             f"missingness {np.bincount(emulated.defined_level, minlength=nlev).tolist()}, "
             f"value {np.bincount(emulated.value_level, minlength=nlev).tolist()}.", ""]
    ok = True
    for sub, grp in groups:
        rows = validation_rows(emu, grp, fh, logg[sel], emulated.logg, threshold=thr)
        ok &= all(r.gated(vs.n_min_report, vs.max_abs_pull) is not False for r in rows)
        lines += [f"### {sub}", "", markdown_table(rows, n_min=vs.n_min_report, max_abs_pull=vs.max_abs_pull), ""]
    return lines, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--definition", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--table-out", default=DEFAULT_TABLE_PATH)
    ap.add_argument("--report-out", required=True, help="markdown validation report")
    args = ap.parse_args(argv)

    root = repo_root()
    d = load_definition(args.definition)
    cal = d.calibration
    meta, t = load_gaia_snapshot(root / cal.snapshot_meta, verify_checksum=False)
    stype = np.asarray(t["nss_solution_type"]).astype(str)
    keep = np.isin(stype, cal.solution_types)
    sid = np.asarray(t["source_id"], dtype=np.int64)[keep]
    if np.unique(sid).size != sid.size:
        raise SystemExit("NSS rows: duplicated source_id within the kept solution types")
    stype = stype[keep]
    g = _col(t, cal.g_mag_column)[keep]
    bp_rp = (_col(t, cal.bp_mag_column) - _col(t, cal.rp_mag_column))[keep]
    mg = abs_g_mag(g, _col(t, cal.parallax_column)[keep])
    logg = np.round(_col(t, cal.logg_column)[keep], cal.logg_decimals)  # float32 archive -> published precision
    del t
    print(f"NSS rows: {sid.size} ({dict(zip(*np.unique(stype, return_counts=True)))}); "
          f"logg defined {np.isfinite(logg).mean():.4f}")

    # 2. equivalence of the column the Andrews chain reads
    _requested, viz = load_vizier_apsis(root / cal.equivalence_snapshot_meta)
    pos = {int(s): k for k, s in enumerate(sid)}
    missing = [s for s in viz if s not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} VizieR rows are not NSS rows of the kept types (e.g. {missing[:3]})")
    v = np.asarray([viz[s][cal.equivalence_column] for s in viz], dtype=np.float64)
    a = logg[[pos[s] for s in viz]]
    same_nan = bool(np.array_equal(np.isnan(v), np.isnan(a)))
    both = np.isfinite(v) & np.isfinite(a)
    max_diff = float(np.max(np.abs(v[both] - a[both]))) if both.any() else 0.0
    print(f"equivalence: {len(viz)} VizieR rows, missing-pattern equal {same_nan}, max |diff| {max_diff}")
    if not same_nan or max_diff > cal.equivalence_tolerance:
        raise SystemExit(f"{cal.equivalence_column} != {cal.logg_column} on the VizieR rows: refusing to calibrate")

    # 3. calibration population, split, calibrate on train
    in_viz = np.isin(sid, np.fromiter(viz.keys(), dtype=np.int64))
    pop = in_viz if cal.restrict_to_equivalence_ids else np.ones(sid.size, dtype=bool)
    hold = holdout_mask(sid, split_seed=cal.split_seed, holdout_fraction=cal.holdout_fraction)
    train, test = pop & ~hold, pop & hold
    feats = {"g_mag": g, "bp_rp": bp_rp, "abs_g_mag": mg}
    levels = calibrate_levels({k: x[train] for k, x in feats.items()}, logg[train], d.binning)
    pop_name = ("VizieR I/355 rows = Andrews ATF p_m2_above >= 0.1" if cal.restrict_to_equivalence_ids
                else "all NSS rows of the kept types")
    prov = {
        "issue": "#680",
        "script": "scripts/calibrate_gspphot_logg_emulator_680.py",
        "code_commit": short_git_commit(),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "snapshot_id": meta.snapshot_id,
        "snapshot_checksum": meta.checksum,
        "solution_types": list(cal.solution_types),
        "calibration_population": pop_name,
        "n_population": int(pop.sum()),
        "n_train": int(train.sum()),
        "n_holdout": int(test.sum()),
        "calibrated_on": "train half only",
        "equivalence": (f"{len(viz)} VizieR rows: {cal.equivalence_column} == {cal.logg_column}; missing pattern "
                        f"equal, max |diff| {max_diff:.1e}"),
        "cells_per_level": [len(c) for c in levels],
    }
    table = build_table(d, levels, definition_sha=definition_sha256(args.definition), provenance=prov)
    out = Path(args.table_out)
    out = out if out.is_absolute() else root / out
    tmp = out.with_suffix(".tmp")
    tmp.write_text(table_to_yaml(table))
    tmp.replace(out)
    print(f"wrote {out} cells/level {prov['cells_per_level']}")

    # 4. validation
    emu = GspPhotLoggEmulator(d, table)
    thr = selection_threshold(d)
    vs = d.validation
    head = [
        f"# #680 GSP-Phot log g emulator: holdout validation ({d.version})",
        "",
        f"Generated by `scripts/calibrate_gspphot_logg_emulator_680.py` at `{prov['code_commit']}`. Real DR3 snapshot "
        f"`{meta.snapshot_id}` ({', '.join(cal.solution_types)}: {sid.size} rows). Calibration population: "
        f"{pop_name} ({int(pop.sum())} rows), train {int(train.sum())} / holdout {int(test.sum())} (hash split, seed "
        f"{cal.split_seed}); the shipped table is calibrated on the train half only. Giant reject: logg < {thr:g} "
        f"(`{vs.cut_id}` in `{vs.selection}`). Equivalence: {prov['equivalence']}.",
        "",
        "Emulated rates are the emulator's exact expectation at the evaluated rows' own (G, BP-RP, M_G); the "
        "emulated median comes from one seeded realization. Pull: (real - emu) / sigma, sigma^2 = p(1-p)"
        "(1/N_eval + 1/N_train) at the pooled rate. Gate (calibration-population holdout only): |pull| <= "
        f"{vs.max_abs_pull:g} on both rates wherever N_eval >= {vs.n_min_report}.",
        "",
    ]
    lines_h, ok = _sections(d, emu, test, train, feats, logg, stype, sid, in_viz, thr,
                            f"Gated: holdout of the calibration population ({pop_name})")
    lines = head + lines_h
    other = ~pop
    if other.any():
        lines_o, _ = _sections(d, emu, other, train, feats, logg, stype, sid, in_viz, thr,
                               "Informational (not gated): NSS rows outside the calibration population")
        lines += lines_o
    lines.append(f"**Gate: {'PASS' if ok else 'FAIL'}**")
    rep = Path(args.report_out)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
