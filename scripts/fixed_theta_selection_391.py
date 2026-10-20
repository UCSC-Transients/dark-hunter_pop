#!/usr/bin/env python
"""#391 (Ryan 2026-10-20) diagnosis at FIXED parameters (a ``--dump-from`` dump): gen-41 draws under the old
(full-ridge) and new (below-turnoff) looks-single selection. The cut is relaxed, so no drawn row becomes ineligible;
newly eligible rows have no draws. Their contribution is estimated by stratified reweighting: draws on old-eligible
rows in the same (C0, G, distance) cell are up-weighted by (rows old + new) / rows old (row_weight-weighted), and the
caveat is that the new rows (ΔM < −0.4 near the turnoff) are not identical to the old ones. Reports the coarse-bin
expectations (mock + spurious vs real) before/after, the total, the G shape and the twin occupancy. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment-new", type=Path, default=Path("config/population/proposal_set_single42.yaml"))
    ap.add_argument("--old-colour-range", type=float, nargs=2, default=(0.35, 2.75))
    ap.add_argument("--cmd-twin", type=Path, default=Path("docs/gate391/cmd_twin/cmd_twin.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    prop = ps.load_proposal_set_fragment(a.fragment_new).proposal
    par = ps.load_parent_snapshot(a.parent_dir, cfg, prop)
    rd = par.meta["ms_ridge"]
    c0 = np.asarray(par.cmd["colour0"], float)
    dm = np.asarray(par.cmd["mg0"], float) - np.interp(c0, rd["colour"], rd["mag"], left=np.nan, right=np.nan)
    ms = ~np.asarray(par.is_giant, bool)
    lo, hi = a.old_colour_range
    old_cmd = ~(ms & (c0 >= lo) & (c0 <= hi)) | ~np.isfinite(dm) | (dm >= prop.single_catalog.dm_min)
    new_ok = par.usable
    old_ok = new_ok & old_cmd
    newly = new_ok & ~old_ok
    rw = np.ones(par.n_rows) if par.row_weight is None else par.row_weight
    g = np.asarray(par.columns["phot_g_mean_mag"], float)
    d = 1.0 / par.truth_parallax_mas
    ce = np.arange(0.35, 1.06, 0.1)
    ge, de = [5, 10, 13, 16, 19, 20.0], [0, 0.3, 0.7, 1.5, 6.0]
    cell = lambda x, e: np.clip(np.digitize(x, e) - 1, -1, len(e) - 1)  # noqa: E731
    ci, gi, di = cell(c0, ce), cell(g, ge), cell(d, de)
    valid = (ci >= 0) & (ci < ce.size - 1) & (gi >= 0) & (gi < len(ge) - 1) & (di >= 0) & (di < len(de) - 1)
    key = np.where(valid, (ci * 10 + gi) * 10 + di, -1)
    fac = np.ones(par.n_rows)
    for k in np.unique(key[valid & (old_ok | newly)]):
        sel = key == k
        n_old, n_new = rw[sel & old_ok].sum(), rw[sel & newly].sum()
        if n_old > 0:
            fac[sel & old_ok] = (n_old + n_new) / n_old
    lost = float(rw[newly & ~np.isin(key, np.unique(key[old_ok & valid]))].sum() / max(rw[newly].sum(), 1e-300))
    z = np.load(a.dump, allow_pickle=True)
    w0 = z["w"]
    w1 = w0 * fac[z["prow"]]
    rin = z["rob"] >= 0
    so, sa = z["ob"] >= 0, z["ab"] >= 0
    spur = z["r_spur_per_orbit"]
    out = {"rows": {"old_eligible": int(old_ok.sum()), "new_eligible": int(new_ok.sum()), "newly_eligible": int(newly.sum()),
                    "newly_eligible_in_cells_without_old_rows_frac": lost,
                    "draws_on_upweighted_rows_frac": float(np.mean(fac[z["prow"]] > 1)),
                    "median_factor_on_upweighted": float(np.median(fac[z["prow"]][fac[z["prow"]] > 1])) if np.any(fac[z["prow"]] > 1) else 1.0}}
    # coarse-grid cells (G 5-13/13-19.5, d 0-1/1-6, log P 0-2.6/2.6-2.92, e 0-.3/.3-1, class)
    bins = []
    for gl, gh in ((5, 13), (13, 19.5)):
        for dl, dh in ((0, 1), (1, 6)):
            for pl, ph in ((0, 2.6), (2.6, 2.9193)):
                for el, eh in ((0, 0.3), (0.3, 1.0)):
                    for cl in (0, 1):
                        m = so & (z["m_g"] >= gl) & (z["m_g"] < gh) & (z["m_d_o"] >= dl) & (z["m_d_o"] < dh) & (z["m_lp"] >= pl) & (z["m_lp"] < ph) \
                            & (z["m_e"] >= el) & (z["m_e"] < eh) & (z["m_cls"] == cl)
                        r_ = rin & (z["r_g"] >= gl) & (z["r_g"] < gh) & (z["r_d"] >= dl) & (z["r_d"] < dh) & (z["r_lp"] >= pl) & (z["r_lp"] < ph) \
                            & (z["r_e"] >= el) & (z["r_e"] < eh) & (z["r_cls"] == cl)
                        k = int(r_.sum())
                        bins.append({"G": [gl, gh], "d": [dl, dh], "logP": [pl, ph], "e": [el, eh], "cls": cl, "real": k,
                                     "before": float(w0[m].sum() + spur[r_].sum()), "after": float(w1[m].sum() + spur[r_].sum())})
    out["bins"] = bins
    out["total_orbits"] = {"before": float((w0[so].sum() + spur[rin].sum()) / rin.sum()), "after": float((w1[so].sum() + spur[rin].sum()) / rin.sum())}
    out["total_accels"] = {"before": float(w0[sa].sum() / z["ra_g"].size), "after": float(w1[sa].sum() / z["ra_g"].size)}
    gb = [5, 8, 9, 10, 11, 12, 13, 14, 16, 19.5]
    real_g = np.histogram(z["r_g"][rin], bins=gb)[0]
    spg = np.histogram(z["r_g"][rin], bins=gb, weights=spur[rin])[0]
    out["G_ratio"] = {"edges": gb, "before": ((np.histogram(z["m_g"][so], bins=gb, weights=w0[so])[0] + spg) / real_g).tolist(),
                      "after": ((np.histogram(z["m_g"][so], bins=gb, weights=w1[so])[0] + spg) / real_g).tolist()}
    t, _, _ = ps.read_proposal_artifact(a.artifact)
    pos = {int(x): i for i, x in enumerate(np.asarray(t["draw_index"], np.int64))}
    # map dump rows to artifact rows through the cascade vectors (refit keeps cache order; match on P, plx, a0)
    tc = None
    ct = json.loads(a.cmd_twin.read_text())
    try:
        import h5py

        with h5py.File(a.artifact, "r") as h:
            tc = h["outcome/cascade"][()]
        key_a = {tuple(np.round(tc[i, [0, 10, 17]], 9)): i for i in range(tc.shape[0]) if tc[i, 0] != 0}
        idx = np.array([key_a.get(tuple(np.round(c[[0, 10, 17]], 9)), -1) for c in z["casc"]])
    except Exception:  # pragma: no cover
        idx = np.full(z["w"].size, -1)
    smg, sc0 = np.asarray(t["sys_mg0"], float), np.asarray(t["sys_colour0"], float)
    ok = idx >= 0
    dms = np.full(z["w"].size, np.nan)
    cs = np.full(z["w"].size, np.nan)
    dms[ok] = smg[idx[ok]] - np.interp(sc0[idx[ok]], rd["colour"], rd["mag"], left=np.nan, right=np.nan)
    cs[ok] = sc0[idx[ok]]
    dd = d[z["prow"]]
    tw = {"d_edges": ct["d_edges"], "real_orbits": [x["twin_over_single"] for x in ct["by_distance"]["real_orbits"]]}
    for lab, w in (("before", w0), ("after", w1)):
        vals = []
        for dl, dh in zip(ct["d_edges"][:-1], ct["d_edges"][1:]):
            m = so & (cs >= 0.85) & (cs <= 2.0) & (dd >= dl) & (dd < dh)
            t_ = w[m & (dms >= -0.95) & (dms < -0.55)].sum()
            s_ = w[m & (dms >= -0.25) & (dms < 0.25)].sum()
            vals.append(float(t_ / s_) if s_ > 0 else None)
        tw[f"mock_{lab}"] = vals
    tw["matched_draw_frac"] = float(ok.mean())
    out["twin_over_single"] = tw
    a.out.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
