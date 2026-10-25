#!/usr/bin/env python3
"""Weight efficiency of compact-object generations under the default θ_CO (#552, spec §13.6/§13.8).

For each requested set of generations (deterministic mixture over the set, all draws): per class (WD, dark 1-2.2,
2.2-3, 3-40 Msun) the Kish ESS per draw of all and of detected draws, detections per draw and CPU s per draw; and per
component the per-cell Kish ESS of the rung-4 table (the §13.6 criterion), with the draws a set of this mix needs for
90 % of the filled cells to reach ``min_cell_ess``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/co_efficiency_report.py --parent-dir <snapshot> \\
        --set 1000=a.h5:a.yaml --set 1001=b.h5:b.yaml --set 1000+1001=a.h5:a.yaml,b.h5:b.yaml [--out out.json]
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
    ap.add_argument("--set", action="append", required=True, help="label=artifact.h5:fragment.yaml[,artifact:fragment...]")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    r4 = cob.load_rung4_config(args.rung4_config)
    parent = None
    report = {}
    for spec in args.set:
        label, items = spec.split("=", 1)
        pairs = [it.split(":") for it in items.split(",")]
        frags = [ps.load_proposal_set_fragment(f) for _, f in pairs]
        gens = [f.proposal for f in frags]
        cob.check_co_generations(gens)
        if parent is None:
            parent = ps.load_parent_snapshot(args.parent_dir, cfg, gens[0])
        ts, os_ = zip(*[ps.read_proposal_artifact(a)[:2] for a, _ in pairs])
        keys = set.intersection(*(set(t) for t in ts))
        t = {k: np.concatenate([x[k] for x in ts]) for k in keys}
        okeys = set.intersection(*(set(o) for o in os_))
        o = {k: np.concatenate([x[k] for x in os_]) for k in okeys}
        w = cob.co_importance_weights(t, parent, gens, frags[0].target_co, partition="all")  # type: ignore[arg-type]
        det = np.asarray(o["accepted_orbital"], bool) | np.asarray(o["published_acceleration"], bool)
        n = t["draw_index"].size
        cpu = np.asarray(o["cpu_seconds"], float)
        rep: dict = {"n_draws": int(n), "cpu_s_per_draw": float(cpu.mean())}
        for name, (code, lo, hi) in CLASSES.items():
            c = (t["co_component"] == code) & (t["m2_msun"] >= lo) & (t["m2_msun"] < hi)
            rep[name] = {"n": int(c.sum()), "detected": int((c & det).sum()), "ess_all_per_draw": ps.kish_ess(w[c]) / n,
                         "ess_detected_per_draw": ps.kish_ess(w[c & det]) / n, "cpu_s_per_draw": float(cpu[c].mean())}
        vals = cob.selection_axis_values(t)
        for comp, code in cob.COMPONENT_CODES.items():
            tab = cob.build_selection_table(vals, w, det, r4.selection_table.axes, mask=t["co_component"] == code)
            filled = tab.n_draws >= 2
            e = tab.ess[filled] / n
            need = np.sort(r4.selection_table.min_cell_ess / e)
            k = max(int(np.ceil(0.9 * need.size)) - 1, 0)
            rep[f"cells_{comp}"] = {"filled": int(filled.sum()), "total": int(filled.size), "median_cell_ess_per_draw": float(np.median(e)),
                                    "draws_for_90pct_filled_cells": float(need[k]), "draws_for_all_filled_cells": float(need[-1])}
        report[label] = rep
        print(f"== {label}: {n} draws, CPU s/draw {rep['cpu_s_per_draw']:.2f}")
        for name in CLASSES:
            r = rep[name]
            print(f"  {name:11s} n={r['n']:4d} det={r['detected']:3d} ESS/draw all {r['ess_all_per_draw']:.4f} detected {r['ess_detected_per_draw']:.5f} cpu {r['cpu_s_per_draw']:.2f}")
        for comp in cob.COMPONENT_CODES:
            print(f"  cells {comp}: {rep['cells_' + comp]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
