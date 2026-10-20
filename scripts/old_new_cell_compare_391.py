#!/usr/bin/env python
"""#391: old (deblended, row photometry) vs new (§12.33 system photometry) mock at IDENTICAL fit parameters in one
coarse cell, split by row G, M1, log f, G_sys − G_row and truth distance; plus per-draw M1/G shifts. Light; no draws."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(dump: Path, arts: list[Path]):
    import h5py

    z = np.load(dump, allow_pickle=True)
    tab = {}
    for a in arts:
        with h5py.File(a, "r") as h:
            c = h["outcome/cascade"][()]
            tr = {k: h["truth"][k][()] for k in ("log10_flux_ratio", "is_dark", "phot_g_mean_mag", "m1_msun", "parallax_mas")
                  if k in h["truth"]}
            rowg = h["truth"]["row_phot_g_mean_mag"][()] if "row_phot_g_mean_mag" in h["truth"] else tr["phot_g_mean_mag"]
            m1r = h["truth"]["m1_row_msun"][()] if "m1_row_msun" in h["truth"] else tr["m1_msun"]
        for i in range(c.shape[0]):
            if c[i, 0] != 0:
                tab[tuple(np.round(c[i, [0, 10, 17]], 9))] = (tr["log10_flux_ratio"][i], tr["is_dark"][i], tr["phot_g_mean_mag"][i],
                                                            rowg[i], tr["m1_msun"][i], m1r[i], 1.0 / tr["parallax_mas"][i])
    rows = np.array([tab.get(tuple(np.round(c[[0, 10, 17]], 9)), (np.nan,) * 7) for c in z["casc"]], float)
    return z, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-dump", type=Path, required=True)
    ap.add_argument("--old-artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--new-dump", type=Path, required=True)
    ap.add_argument("--new-artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out = {}
    for lab, dmp, arts in (("old", a.old_dump, a.old_artifacts), ("new", a.new_dump, a.new_artifacts)):
        z, r = load(dmp, arts)
        lf, dark, gsys, grow, m1, m1row, dtrue = r.T
        so = z["ob"] >= 0
        cell = so & (z["m_g"] >= 13) & (z["m_g"] < 19.5) & (z["m_d_o"] >= 1) & (z["m_d_o"] < 6) & (z["m_lp"] >= 2.6) & (z["m_lp"] < 2.9193) \
            & (z["m_e"] < 0.3) & (z["m_cls"] == 0)
        w = z["w"]
        rin = z["rob"] >= 0
        rcell = rin & (z["r_g"] >= 13) & (z["r_d"] >= 1) & (z["r_lp"] >= 2.6) & (z["r_e"] < 0.3) & (z["r_cls"] == 0)

        def split(x, e):
            return np.histogram(x[cell], bins=e, weights=w[cell])[0].round(1).tolist()

        out[lab] = {"cell_mock": float(w[cell].sum()), "cell_real": int(rcell.sum()), "matched": float(np.mean(np.isfinite(gsys[cell]))),
                    "kish": float(w[cell].sum() ** 2 / max(np.sum(w[cell] ** 2), 1e-300)),
                    "by_row_G": split(grow, [13, 15, 16, 17, 18, 19, 20]),
                    "by_system_G": split(gsys, [13, 15, 16, 17, 18, 19, 20]),
                    "by_M1": split(m1, [0, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]),
                    "by_log_f": split(np.where(dark > 0, -9, lf), [-10, -3, -2, -1, -0.5, 1]),
                    "by_Gsys_minus_Grow": split(gsys - grow, [-1.0, -0.5, -0.2, -0.05, 0.05, 1.0]),
                    "by_true_d_kpc": split(dtrue, [0, 1, 1.5, 2, 3, 6]),
                    "M1_over_M1row_cell_wmedian": float(np.average(m1[cell] / m1row[cell], weights=w[cell])) if cell.any() else np.nan,
                    "total_orbit_w": float(w[so].sum())}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
