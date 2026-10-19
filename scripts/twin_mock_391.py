#!/usr/bin/env python
"""#391 §12.29 check: mock-detected twin/single by distance (MS C0 0.85–2.0, as ``cmd_twin_391.py``) for one or
more fit dumps, next to the real-host values in ``cmd_twin.json``; and the G-bin ratios. Writes JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--cmd-twin", type=Path, default=Path("docs/gate391/cmd_twin/cmd_twin.json"))
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from darkhunter_pop import giants
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(args.parent_dir, cfg, ps.load_proposal_set_fragment(args.fragment).proposal)
    rd = par.meta["ms_ridge"]
    ridge = giants.MSRidge(colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float),
                           sigma=np.asarray(rd["sigma"], float), n_rows=np.asarray(rd["n_rows"]))
    ct = json.loads(args.cmd_twin.read_text())
    c0, mg0 = np.asarray(par.cmd["colour0"], float), np.asarray(par.cmd["mg0"], float)
    r, _, inr = ridge.at(c0)
    dm = np.where(inr, mg0 - r, np.nan)
    d = 1.0 / par.truth_parallax_mas
    lo, hi = ct["ms_colour_range"]
    tw, si = ct["twin_band"], ct["single_band"]
    de = ct["d_edges"]
    out = {"d_edges": de, "real_orbits": [x["twin_over_single"] for x in ct["by_distance"]["real_orbits"]],
           "real_accels": [x["twin_over_single"] for x in ct["by_distance"]["real_accels"]]}
    ge = [5, 8, 9, 10, 11, 12, 13, 14, 16, 19.5]
    for item in args.dumps:
        lab, path = item.split("=", 1)
        z = np.load(path, allow_pickle=True)
        for kind, s in (("orbits", z["ob"] >= 0), ("accels", z["ab"] >= 0)):
            row = z["prow"][s]
            w = z["w"][s]
            ok = (c0[row] >= lo) & (c0[row] <= hi) & np.isfinite(dm[row])
            vals = []
            for a, b in zip(de[:-1], de[1:]):
                m = ok & (d[row] >= a) & (d[row] < b)
                t = w[m & (dm[row] >= tw[0]) & (dm[row] < tw[1])].sum()
                n = w[m & (dm[row] >= si[0]) & (dm[row] < si[1])].sum()
                vals.append(float(t / n) if n > 0 else None)
            out[f"{lab}_mock_{kind}"] = vals
        so, rin = z["ob"] >= 0, z["rob"] >= 0
        mm = np.histogram(z["m_g"][so], bins=ge, weights=z["w"][so])[0]
        sp = np.histogram(z["r_g"][rin], bins=ge, weights=z["r_spur_per_orbit"][rin])[0]
        rr = np.histogram(z["r_g"][rin], bins=ge)[0]
        out[f"{lab}_g_ratio"] = ((mm + sp) / rr).tolist()
    out["g_edges"] = ge
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
