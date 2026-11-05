#!/usr/bin/env python3
"""Rung 3 v4 against v3 on the previously flagged mismatches (#603, spec §12.42).

Runs the #587 comparison (``noise_check_compare_587.py``: χ² before/after on common edges, noise metrics, side-by-side
2-D and 1-D figures) and adds the population checks the v4 terms target:

* A = (a0/ϖ)/P^(2/3) by distance: N(A > 0.5) and the 1-D A distribution per distance bin;
* eccentricity by G: weighted median e and N(e > 0.5) per G bin, and the 1-D e distribution per G bin;
* short-period bright systems: N(P < 100 d) at G < 13 by distance;
* twin-band occupancy per distance bin (the F4 term) from each run's ``summary.json``.

Writes ``population_metrics.json`` / ``population_metrics.md`` and the extra figures under ``--out-dir``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/rung3_v4_compare_603.py --edges <edges.json> \\
        --run v3=<dump>:<rank_dir>:<summary.json> --run v4=<dump>:<rank_dir>:<summary.json> --out-dir docs/gate391/rung3_v4/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_check_compare_587 as ncc  # noqa: E402
import rung3_plots_391 as rp  # noqa: E402

from darkhunter_pop.config_loader import load_config  # noqa: E402

G_BINS = ((10.0, 13.0), (13.0, 16.0), (16.0, 19.5))


def pop_metrics(kind):
    r = kind.real
    m, w = ncc.mock_arrays(kind)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for lo, hi in zip(rp.D_BINS[:-1], rp.D_BINS[1:]):
            sr, sm = (r["d"] >= lo) & (r["d"] < hi), (m["d"] >= lo) & (m["d"] < hi)
            out[f"N(A > 0.5), d {lo:g}-{hi:g}"] = ncc.count(sr & (r["A"] > 0.5), sm & (m["A"] > 0.5), w)
            out[f"N(P < 100 d, G < 13), d {lo:g}-{hi:g}"] = ncc.count(sr & (r["g"] < 13) & (r["P"] < 100), sm & (m["g"] < 13) & (m["P"] < 100), w)
        for lo, hi in G_BINS:
            sr, sm = (r["g"] >= lo) & (r["g"] < hi), (m["g"] >= lo) & (m["g"] < hi)
            out[f"e median, G {lo:g}-{hi:g}"] = {"real": float(np.nanmedian(r["e"][sr])), "mock": ncc.wquant(m["e"][sm], w[sm], 0.5)}
            out[f"N(e > 0.5), G {lo:g}-{hi:g}"] = ncc.count(sr & (r["e"] > 0.5), sm & (m["e"] > 0.5), w)
        lr, lm = np.log10(r["P"]), np.log10(m["P"])
        sr, sm = (r["g"] < 13) & (lr >= 2.5) & (lr < 3.0), (m["g"] < 13) & (lm >= 2.5) & (lm < 3.0)
        out["e median, G < 13 and log P 2.5-3"] = {"real": float(np.nanmedian(r["e"][sr])), "mock": ncc.wquant(m["e"][sm], w[sm], 0.5)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=DUMP:RANK_DIR:SUMMARY")
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    runs = []
    for s in a.run:
        n, rest = s.split("=", 1)
        d, rk, sm = rest.split(":")
        runs.append({"name": n, "dump": Path(d), "rank": Path(rk), "summary": Path(sm)})
    ncc.main(sum((["--run", f"{r['name']}={r['dump']}:{r['rank']}"] for r in runs), []) + ["--edges", str(a.edges), "--out-dir", str(a.out_dir)])
    cfg = load_config()
    style, dpi = cfg.plotting, int(cfg.diagnostics.figure_dpi) if hasattr(cfg.diagnostics, "figure_dpi") else 120
    edges = json.loads(a.edges.read_text())
    kinds = [(r["name"], rp.Kind(np.load(r["dump"]), "o", rp.ORBIT_OBS)) for r in runs]
    res = {"orbits": {n: pop_metrics(k) for n, k in kinds}, "twin_band": {}}
    for r in runs:
        tb = json.loads(r["summary"].read_text())["results"]["joint"]["twin_band"]
        res["twin_band"][r["name"]] = {"real_twin_over_single": [a_ / max(b_, 1) for a_, b_ in zip(tb["real_twin"], tb["real_single"])],
                                       "mock_twin_over_single": [a_ / max(b_, 1e-9) for a_, b_ in zip(tb["mock_twin"], tb["mock_single"])]}
    figs = []
    for lo, hi in zip(rp.D_BINS[:-1], rp.D_BINS[1:]):
        ncc.SUBSETS[f"d {lo:g}-{hi:g}"] = (lambda v, lo=lo, hi=hi: (v["d"] >= lo) & (v["d"] < hi))
        fn = f"orbits_1d_A_d{lo:g}-{hi:g}_compare.png"
        ncc.side_by_side_1d(kinds, "orbits", "A", edges["orbits"], a.out_dir / fn, style, dpi, subset=f"d {lo:g}-{hi:g}")
        figs.append(fn)
    for lo, hi in G_BINS:
        ncc.SUBSETS[f"G {lo:g}-{hi:g}"] = (lambda v, lo=lo, hi=hi: (v["g"] >= lo) & (v["g"] < hi))
        fn = f"orbits_1d_e_G{lo:g}-{hi:g}_compare.png"
        ncc.side_by_side_1d(kinds, "orbits", "e", edges["orbits"], a.out_dir / fn, style, dpi, subset=f"G {lo:g}-{hi:g}")
        figs.append(fn)
    ncc.SUBSETS["G<13, d<0.3"] = lambda v: (v["g"] < 13) & (v["d"] < 0.3)
    fn = "orbits_1d_P_Glt13_dlt0.3_compare.png"
    ncc.side_by_side_1d(kinds, "orbits", "P", edges["orbits"], a.out_dir / fn, style, dpi, subset="G<13, d<0.3")
    figs.append(fn)
    res["figures"] = figs
    (a.out_dir / "population_metrics.json").write_text(json.dumps(res, indent=1))
    names = [r["name"] for r in runs]
    md = ["| statistic | DR3 | " + " | ".join(f"{n} mock | {n} DR3/mock (pull)" for n in names) + " |", "|---|---:|" + "---:|---:|" * len(names)]
    for st in res["orbits"][names[0]]:
        v0 = res["orbits"][names[0]][st]
        if "pull" in v0:
            md.append(f"| {st} | {v0['real']:.0f} | " + " | ".join(f"{res['orbits'][n][st]['mock']:.0f} ± {res['orbits'][n][st]['mock_mc']:.0f} | "
                                                              f"{res['orbits'][n][st]['ratio']:.2f} ({res['orbits'][n][st]['pull']:+.1f})" for n in names) + " |")
        else:
            md.append(f"| {st} | {v0['real']:.3f} | " + " | ".join(f"{res['orbits'][n][st]['mock']:.3f} | –" for n in names) + " |")
    md += ["", "Twin / single band occupancy per distance bin (0–0.2 / 0.2–0.5 / 0.5–1.0 kpc):", ""]
    for n in names:
        tb = res["twin_band"][n]
        md.append(f"- {n}: DR3 " + " / ".join(f"{x:.2f}" for x in tb["real_twin_over_single"]) + "; mock " + " / ".join(f"{x:.2f}" for x in tb["mock_twin_over_single"]))
    md += ["", "Figures: " + ", ".join(f"[{f}]({f})" for f in figs)]
    (a.out_dir / "population_metrics.md").write_text("\n".join(md) + "\n")
    print(f"population metrics and {len(figs)} figures -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
