#!/usr/bin/env python3
"""#589 / #590: population-level distribution widths at fixed (G, P, A), mock (gens 47–50) vs DR3, no simulation.

* log z, z = (a0/σ_a0)/√(RUWE² − 1), for accepted orbits with RUWE > 1.4 and P < 1000 d: per (G, log P, A) cell
  the weighted median, the half 16–84 width (σ68), p90 − p50 and p99 − p50, mock (default-target weights; also
  unweighted) vs DR3 (all published Orbital + AstroSpectroSB1). The quadrature excess √(σ68,DR3² − σ68,mock²).
* F2 at G 10–12 (and 12–13, 13–15 for reference) for C1 orbits, per (log P, A) cell and per class-free colour bin:
  the same width statistics.

A cell needs ≥ 50 DR3 rows and mock Kish ESS ≥ 20 (default weights). Writes ``width.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_common_v21tr1 as dc  # noqa: E402
import diag_edge_589 as de  # noqa: E402

G_CELLS = ((9, 11.5), (11.5, 13), (13, 15), (15, 17))
P_CELLS = ((1.5, 2.5), (2.5, 2.75), (2.75, 3.0))
A_CELLS = ((0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 5))


def wq(x, w, qs):
    ok = np.isfinite(x) & (w > 0)
    o = np.argsort(x[ok])
    c = np.cumsum(w[ok][o]) / w[ok].sum()
    return [float(x[ok][o][min(np.searchsorted(c, q), ok.sum() - 1)]) for q in qs]


def stats(x, w):
    p16, p50, p84, p90, p99 = wq(x, w, (0.16, 0.5, 0.84, 0.9, 0.99))
    return {"median": p50, "sigma68": 0.5 * (p84 - p16), "p90_minus_p50": p90 - p50, "p99_minus_p50": p99 - p50}


def cell_table(xr, wr, xm, wm, cells_r, cells_m, n_min=50, ess_min=20):
    out = {}
    for name in cells_r:
        br, bm = cells_r[name] & np.isfinite(xr), cells_m[name] & np.isfinite(xm) & (wm > 0)
        ess = dc.kish(wm[bm])
        if br.sum() < n_min or ess < ess_min:
            out[name] = {"n_dr3": int(br.sum()), "mock_ess": ess, "skipped": True}
            continue
        sr, sm, su = stats(xr[br], wr[br]), stats(xm[bm], wm[bm]), stats(xm[bm], np.ones(bm.sum()))
        out[name] = {"n_dr3": int(br.sum()), "mock_ess": ess, "mock_draws": int(bm.sum()), "dr3": sr, "mock": sm, "mock_unweighted": su,
                     "sigma68_ratio": sr["sigma68"] / sm["sigma68"] if sm["sigma68"] > 0 else float("nan"),
                     "sigma68_excess_quadrature": float(np.sqrt(max(sr["sigma68"] ** 2 - sm["sigma68"] ** 2, 0)))}
    return out


def main(argv=None) -> int:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    with h5py.File(de.D / "output/gate390/published_orbits.h5", "r") as f:
        pub = {k: f["published"][k][()] for k in f["published"] if k != "corr_vec"}
    st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in pub["nss_solution_type"]])
    a0d, _ = de.halbwachs_t(*(pub[x] for x in de.TI))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {"g": pub["g_mag"], "lp": np.log10(pub["period"]), "A": (a0d / pub["parallax"]) / (pub["period"] / 365.25) ** (2 / 3),
             "lz": np.log10(pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1))}
    rsel = np.isin(st, ["Orbital", "AstroSpectroSB1"]) & (pub["ruwe"] > 1.4) & (pub["period"] < 1000)
    M = dc.load_mock()
    cas = M["cascade"]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = {"g": M["phot_g_mean_mag"], "lp": np.log10(cas[:, 10]), "A": M["o_A"], "lz": np.log10(M["o_a0snr"] / np.sqrt(cas[:, 22] ** 2 - 1))}
    msel = M["accepted"] & (cas[:, 22] > 1.4) & (cas[:, 10] < 1000)

    def cells(s, sel):
        out = {}
        for gl, gh in G_CELLS:
            for pl_, ph in P_CELLS:
                for al, ah in A_CELLS:
                    out[f"G{gl:g}-{gh:g} logP{pl_:g}-{ph:g} A{al:g}-{ah:g}"] = sel & (s["g"] >= gl) & (s["g"] < gh) & (s["lp"] >= pl_) & (s["lp"] < ph) & (s["A"] >= al) & (s["A"] < ah)
        return out

    r["e"], m["e"] = pub["eccentricity"], cas[:, 14]

    def ecells(s, sel):
        out = {}
        for gl, gh in ((9, 13), (13, 17)):
            for pl_, ph in ((1.5, 2.5), (2.5, 2.75), (2.75, 3.0)):
                for el, eh in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)):
                    out[f"G{gl:g}-{gh:g} logP{pl_:g}-{ph:g} e{el:g}-{eh:g}"] = sel & (s["g"] >= gl) & (s["g"] < gh) & (s["lp"] >= pl_) & (s["lp"] < ph) & (s["e"] >= el) & (s["e"] < eh)
        return out

    res = {"log_z_G_P_e": cell_table(r["lz"], np.ones(r["g"].size), m["lz"], M["w"], ecells(r, rsel), ecells(m, msel))}
    # eccentricity distribution (default-target weights) vs DR3, per (G, log P)
    edist = {}
    for gl, gh in ((9, 13), (13, 17)):
        for pl_, ph in ((1.5, 2.5), (2.5, 2.75), (2.75, 3.0)):
            br = rsel & (r["g"] >= gl) & (r["g"] < gh) & (r["lp"] >= pl_) & (r["lp"] < ph)
            bm = msel & (m["g"] >= gl) & (m["g"] < gh) & (m["lp"] >= pl_) & (m["lp"] < ph)
            edist[f"G{gl:g}-{gh:g} logP{pl_:g}-{ph:g}"] = {"dr3_e_quartiles": np.percentile(r["e"][br], [25, 50, 75]).tolist(),
                                                           "mock_e_quartiles": wq(m["e"][bm], M["w"][bm], (0.25, 0.5, 0.75)), "mock_ess": dc.kish(M["w"][bm])}
    res["e_distribution"] = edist
    res["log_z"] = cell_table(r["lz"], np.ones(r["g"].size), m["lz"], M["w"], cells(r, rsel), cells(m, msel))
    # coarser: G only
    gc_r = {f"G{gl:g}-{gh:g}": rsel & (r["g"] >= gl) & (r["g"] < gh) for gl, gh in G_CELLS}
    gc_m = {f"G{gl:g}-{gh:g}": msel & (m["g"] >= gl) & (m["g"] < gh) for gl, gh in G_CELLS}
    res["log_z_by_G"] = cell_table(r["lz"], np.ones(r["g"].size), m["lz"], M["w"], gc_r, gc_m)
    # F2 for C1 orbits (rung-3 selection, real from the #587 dump)
    R = dc.load_real()["orbits"]
    ro = {"g": R["g"], "lp": np.log10(R["P"]), "A": R["A"], "f2": R["f2"], "c0": R["colour0"]}
    mo = {"g": M["phot_g_mean_mag"], "lp": np.log10(M["o_P"]), "A": M["o_A"], "f2": M["o_f2"], "c0": M["sys_colour0"]}
    f2res = {}
    for gl, gh in ((10, 11), (11, 12), (12, 13), (13, 15)):
        cr, cm = {}, {}
        cr["all"], cm["all"] = (ro["g"] >= gl) & (ro["g"] < gh), M["orbit"] & (mo["g"] >= gl) & (mo["g"] < gh)
        for pl_, ph in ((1.5, 2.5), (2.5, 2.75), (2.75, 3.0)):
            cr[f"logP{pl_:g}-{ph:g}"] = cr["all"] & (ro["lp"] >= pl_) & (ro["lp"] < ph)
            cm[f"logP{pl_:g}-{ph:g}"] = cm["all"] & (mo["lp"] >= pl_) & (mo["lp"] < ph)
        for al, ah in A_CELLS:
            cr[f"A{al:g}-{ah:g}"] = cr["all"] & (ro["A"] >= al) & (ro["A"] < ah)
            cm[f"A{al:g}-{ah:g}"] = cm["all"] & (mo["A"] >= al) & (mo["A"] < ah)
        for cl, ch in ((-1, 0.8), (0.8, 1.1), (1.1, 3)):
            cr[f"colour0 {cl:g}-{ch:g}"] = cr["all"] & (ro["c0"] >= cl) & (ro["c0"] < ch)
            cm[f"colour0 {cl:g}-{ch:g}"] = cm["all"] & (mo["c0"] >= cl) & (mo["c0"] < ch)
        f2res[f"G{gl:g}-{gh:g}"] = cell_table(ro["f2"], np.ones(ro["g"].size), mo["f2"], M["w"], cr, cm, n_min=50, ess_min=10)
    res["f2"] = f2res
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "width.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
