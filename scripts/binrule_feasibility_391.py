#!/usr/bin/env python3
"""§12.14 bin-rule feasibility (#391, Ryan 2026-10-10): do the gens 23-30 draws support a grid that resolves every
correction axis with ESS_b >= 30 in every big bin?

Candidate orbit grids satisfy the §12.14 axis requirements by construction (a G split; log P edges at the Beta range
edges 2.0 and 2.6; >= 3 e bins; >= 3 d bins). ESS_b of the mock weights is evaluated at the gens 23-30 control-fit best
θ (noW and cmdW) and at MdS17; real counts enter only through the ">= 1% of real orbits" big-bin definition. For each
grid: failing big bins and the draw cost to bring them to 30 (total-ESS scaling at the generation-30 rate, an
approximation, and the per-cell generation-30 rate). Analysis only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_refit4_391 as r4  # noqa: E402

CANDIDATES = {
    "g_mag": [[5.0, 13.0, 19.5], [5.0, 12.0, 14.0, 16.0, 19.5]],
    "distance_kpc": [[0.0, 0.5, 1.0, 6.0], [0.0, 0.3, 0.7, 1.5, 6.0]],
    "log10_period_days": [[0.0, 2.0, 2.6, 2.9193], [0.0, 2.0, 2.3, 2.6, 2.75, 2.9193]],
    "eccentricity": [[0.0, 0.3, 0.6, 1.0], [0.0, 0.2, 0.4, 0.6, 1.0]],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_30.npz"))
    ap.add_argument("--fit", type=Path, default=Path("docs/gate391/rung3_refit4/gens23_30_gen29_fine_grid/summary.json"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_refit4_fine.yaml"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--new-gen", type=int, default=30)
    ap.add_argument("--min-ess", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import yaml

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    ns = r4.setup(args, fc)
    fits = json.loads(args.fit.read_text())
    th_mds = np.array([0, 0.5, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1, 1, 1, 0], float)
    pts = {"control noW": ns.W["noW"] * np.exp(ns.log_mod(np.array(fits["noW"]["theta"]))),
           "control cmdW": ns.W["cmdW"] * np.exp(ns.log_mod(np.array(fits["cmdW"]["theta"]))),
           "MdS17": ns.W["noW"] * np.exp(ns.log_mod(th_mds))}
    w28 = ns.W["g28"] * np.exp(ns.log_mod(np.array(fits["noW"]["theta"])))  # generation-30 alone (key name kept)
    ae = [np.asarray(fc["outer"]["g_mag"]), np.asarray(fc["outer"]["distance_kpc"])]
    rep = ["#391 §12.14 bin-rule feasibility on gens 23-30 (axis requirements built in; ESS_b >= 30 in every big bin)", ""]
    rows = []
    for combo in itertools.product(*(CANDIDATES[a] for a in r4.AXES)):
        e_ = [np.asarray(c, float) for c in combo]
        ob, _, k, _, n, _ = r4.binned(ns, e_, ae)
        big = k >= 0.01 * k.sum()
        ess = {name: r4.ess_b(w, ob, n) for name, w in pts.items()}
        emin = np.minimum.reduce(list(ess.values()))
        fail = big & (emin < args.min_ess)
        e30 = r4.ess_b(w28, ob, n)
        rate = e30 / max(ns.cpu_28, 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            need_cell = np.where(fail, (args.min_ess - emin) / rate, 0.0)
        tot = {name: float(ns.ps.kish_ess(w[ob >= 0])) for name, w in pts.items()}
        fac = float(np.max(args.min_ess / np.maximum(emin[big], 1e-9)))
        e_all = min(tot.values())
        scale_cost = max(0.0, fac - 1.0) * e_all / max(ns.ps.kish_ess(w28[ob >= 0]) / ns.cpu_28, 1e-9)
        shp = [len(c) - 1 for c in combo]
        rows.append((int(fail.sum()), n, combo, int(big.sum()), float(k[fail].sum() / k.sum()), float(np.min(emin[big])),
                     scale_cost, float(np.nanmax(np.where(np.isfinite(need_cell), need_cell, np.nan))) if fail.any() else 0.0,
                     int(np.sum(~np.isfinite(need_cell) & fail)), shp, fail, emin, k))
    rows.sort(key=lambda r: (r[0], r[1]))
    for r in rows:
        rep.append(f"grid {r[9]} ({r[1]} bins, {r[3]} big): failing big bins {r[0]} holding {r[4]:.1%} of real; min big ESS_b {r[5]:.1f}; "
                   f"cost to 30: ~{r[6]:.0f} CPU-h (total-ESS scaling at the gen-30 rate); worst per-cell gen-30 estimate "
                   f"{r[7]:.0f} CPU-h ({r[8]} failing cells got no gen-30 ESS at all)")
    best = rows[0]
    rep += ["", f"least-failing grid {best[9]}: {[list(c) for c in best[2]]}"]
    shp = best[9]
    for b_ in np.flatnonzero(best[10]):
        i = np.unravel_index(b_, shp)
        rep.append("  failing big bin G {:g}-{:g} d {:g}-{:g} lP {:g}-{:.2f} e {:g}-{:g}: real {}, min ESS_b {:.1f}".format(
            best[2][0][i[0]], best[2][0][i[0] + 1], best[2][1][i[1]], best[2][1][i[1] + 1], best[2][2][i[2]], best[2][2][i[2] + 1],
            best[2][3][i[3]], best[2][3][i[3] + 1], int(best[12][b_]), best[11][b_]))
    feasible = [r for r in rows if r[0] == 0]
    rep.append("")
    rep.append(f"FEASIBLE grids (no failing big bin): {len(feasible)}" + (f"; most bins: {feasible[-1][9]}" if feasible else
               " -> CONFLICT: the axis requirements and ESS_b >= 30 cannot both hold with gens 23-30 (spec §12.14: report, do not relax)"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
