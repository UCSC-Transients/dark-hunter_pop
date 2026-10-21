#!/usr/bin/env python
"""#391 gen 43: why the G < 9 excess and the G 16–19.5 deficit. At a fit dump: (mock + spurious) / real by distance
in each G range (with parent-row errors); mock composition by M1 and system colour; real solution-type shares."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import h5py

    z = np.load(a.dump, allow_pickle=True)
    tab = {}
    for f in a.artifacts:
        with h5py.File(f, "r") as h:
            c, sc = h["outcome/cascade"][()], h["truth/sys_colour0"][()]
        for i in range(c.shape[0]):
            if c[i, 0] != 0:
                tab[tuple(np.round(c[i, [0, 10, 17]], 9))] = sc[i]
    col = np.array([tab.get(tuple(np.round(c[[0, 10, 17]], 9)), np.nan) for c in z["casc"]])
    w, so, rin = z["w"], z["ob"] >= 0, z["rob"] >= 0
    spo = z["r_spur_per_orbit"]
    out = {}
    for lab, (g0, g1) in (("G<9", (5, 9)), ("G16-19.5", (16, 19.5))):
        m = so & (z["m_g"] >= g0) & (z["m_g"] < g1)
        r = rin & (z["r_g"] >= g0) & (z["r_g"] < g1)
        de = [0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 6.0]
        rows = []
        for lo, hi in zip(de[:-1], de[1:]):
            mm, rr = m & (z["m_d_o"] >= lo) & (z["m_d_o"] < hi), r & (z["r_d"] >= lo) & (z["r_d"] < hi)
            u, inv = np.unique(z["prow"][mm], return_inverse=True)
            rw = np.bincount(inv, weights=w[mm]) if mm.any() else np.zeros(1)
            k = int(rr.sum())
            ess = float(rw.sum() ** 2 / max(np.sum(rw**2), 1e-300))
            rows.append({"d": [lo, hi], "real": k, "mock": float(rw.sum()), "spur": float(spo[rr].sum()), "row_ess": round(ess, 1),
                         "ratio": float((rw.sum() + spo[rr].sum()) / max(k, 1)), "err": float(rw.sum() / max(k, 1) / np.sqrt(max(ess, 1)))})
        me = [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 10.0]
        ce = [-1, 0.3, 0.5, 0.7, 1.0, 1.5, 4]
        out[lab] = {"by_distance": rows,
                    "mock_M1_share": (np.histogram(z["m1_law"][m], bins=me, weights=w[m])[0] / max(w[m].sum(), 1e-300)).round(3).tolist(), "M1_edges": me,
                    "mock_sys_colour0_share": (np.histogram(col[m], bins=ce, weights=w[m])[0] / max(w[m].sum(), 1e-300)).round(3).tolist(), "colour_edges": ce,
                    "real_solution_type_share": {t: float(np.mean(z["r_soltype"][r] == t)) for t in np.unique(z["r_soltype"][r])},
                    "spurious_share_of_real": float(spo[r].sum() / max(r.sum(), 1))}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
