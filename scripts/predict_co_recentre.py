#!/usr/bin/env python3
"""Predict a new compact-object proposal's weight efficiency by reweighting existing draws (#552, spec §13.8).

For draws x_i ~ q (one artifact) and target p, with w = p/q and r = q/q' (q' the new proposal):
E_q'[f] = E_q[f q'/q], so the per-draw ESS fraction a sample from q' would have for a set S (all draws of a class, or
its detected draws) is (Σ_S w)² / (N Σ_S w² r), and its per-draw detection rate is mean(d / r).
Printed per class (WD, dark 1-2.2, 2.2-3, 3-40 Msun) next to the measured values for q.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/predict_co_recentre.py --parent-dir <snapshot> --artifact <h5> \\
        --fragment <old yaml> --new-fragment <new yaml> [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config

CLASSES = {"WD": (0, 0.0, np.inf), "dark 1-2.2": (1, 0.0, 2.2), "dark 2.2-3": (1, 2.2, 3.0), "dark 3-40": (1, 3.0, np.inf)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, required=True)
    ap.add_argument("--new-fragment", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    old, new = ps.load_proposal_set_fragment(args.fragment), ps.load_proposal_set_fragment(args.new_fragment)
    cob.check_co_generations([old.proposal, new.proposal])
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, old.proposal)
    t, o, _ = ps.read_proposal_artifact(args.artifact)
    w = cob.co_importance_weights(t, parent, [old.proposal], old.target_co, partition="all")  # type: ignore[arg-type]
    lq_old = cob.log_q_co_total_for(t, parent, old.proposal)
    lq_new = cob.log_q_co_total_for(t, parent, new.proposal)
    r = np.exp(lq_old - lq_new)
    det = np.asarray(o["accepted_orbital"], bool) | np.asarray(o["published_acceleration"], bool)
    n = t["draw_index"].size
    out = {}
    for name, (code, lo, hi) in CLASSES.items():
        c = (t["co_component"] == code) & (t["m2_msun"] >= lo) & (t["m2_msun"] < hi)
        row = {"n": int(c.sum()), "n_detected": int((c & det).sum())}
        for lab, s in (("all", c), ("detected", c & det)):
            sw, sw2 = np.sum(w[s]), np.sum(w[s] ** 2)
            row[f"ess_per_draw_{lab}_measured"] = float(sw**2 / sw2 / n) if sw2 > 0 else 0.0
            row[f"ess_per_draw_{lab}_predicted"] = float(sw**2 / np.sum(w[s] ** 2 * r[s]) / n) if sw2 > 0 else 0.0
        row["detected_per_draw_measured"] = float(np.sum(c & det) / n)
        row["detected_per_draw_predicted"] = float(np.sum((c & det) / r) / n)
        out[name] = row
        print(f"{name:11s} n={row['n']:4d} det={row['n_detected']:3d} | ESS/draw all {row['ess_per_draw_all_measured']:.4f} -> "
              f"{row['ess_per_draw_all_predicted']:.4f} | detected {row['ess_per_draw_detected_measured']:.5f} -> "
              f"{row['ess_per_draw_detected_predicted']:.5f} | det/draw {row['detected_per_draw_measured']:.4f} -> {row['detected_per_draw_predicted']:.4f}")
    # Per-cell (the §13.6 criterion: Kish ESS of all draws in each rung-4 table cell >= min_cell_ess).
    r4 = cob.load_rung4_config(args.rung4_config)
    vals = cob.selection_axis_values(t)
    target_ess = r4.selection_table.min_cell_ess
    for comp, code in cob.COMPONENT_CODES.items():
        sel = t["co_component"] == code
        tq = cob.build_selection_table(vals, w, det, r4.selection_table.axes, mask=sel)
        tr = cob.build_selection_table(vals, w * np.sqrt(r), det, r4.selection_table.axes, mask=sel)  # Σ w² r in sum_w2
        filled = tq.n_draws >= 2
        e_meas = tq.ess[filled] / n
        with np.errstate(divide="ignore", invalid="ignore"):
            e_pred = (tq.sum_w[filled] ** 2 / tr.sum_w2[filled]) / n
        need_m = np.sort(target_ess / e_meas)
        need_p = np.sort(target_ess / e_pred)
        k = max(int(np.ceil(0.9 * need_m.size)) - 1, 0)
        cell = {"cells_filled": int(filled.sum()), "median_ess_per_draw_measured": float(np.median(e_meas)),
                "median_ess_per_draw_predicted": float(np.median(e_pred)),
                "draws_for_90pct_filled_cells_measured": float(need_m[k]), "draws_for_90pct_filled_cells_predicted": float(need_p[k])}
        out[f"cells_{comp}"] = cell
        print(f"cells {comp}: {cell}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
