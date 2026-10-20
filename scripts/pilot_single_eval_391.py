#!/usr/bin/env python
"""#391 spec §12.33 pilot evaluation: mock-detected twin/single (SYSTEM CMD) vs real hosts by distance; G-bin
(mock + spurious)/real with parent-row errors at the pilot's best fit; accepted-orbit F2 at G < 13 mock vs real;
CPU per draw and detections per CPU-h. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single40.yaml"))
    ap.add_argument("--cmd-twin", type=Path, default=Path("docs/gate391/cmd_twin/cmd_twin.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    parts = [ps.read_proposal_artifact(x) for x in a.artifact]
    t = {k: np.concatenate([np.asarray(p_[0][k]) for p_ in parts]) for k in ("draw_index", "sys_mg0", "sys_colour0", "system_in_cut")}
    o = {k: np.concatenate([np.asarray(p_[1][k]) for p_ in parts]) for k in ("cpu_seconds", "accepted_orbital")}
    z = np.load(a.dump, allow_pickle=True)
    zc = np.load(a.cache, allow_pickle=True)
    rd = par.meta["ms_ridge"]
    ct = json.loads(a.cmd_twin.read_text())
    # rel: dump rows -> cache rows (same order as artifact truth): match by cascade rows
    di_cache = np.asarray(zc["draw_index"], np.int64)
    pos = {int(d): i for i, d in enumerate(np.asarray(t["draw_index"], np.int64))}
    # recover rel as in refit setup: draws in the orbit or acceleration sample
    casc = np.asarray(zc["cascade"], float)
    rel_mask = np.zeros(di_cache.size, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        cand = np.flatnonzero((np.asarray(zc["accepted"], bool) & (casc[:, 10] <= 830.4)) | np.asarray(zc["published_acceleration"], bool))
    rel_mask[cand] = True
    rel = np.flatnonzero(rel_mask)
    if rel.size != z["w"].size:  # outer-bin edges can drop a few; fall back to matching the cascade
        key = {tuple(np.round(casc[i, [0, 10, 17]], 9)): i for i in cand}
        rel = np.array([key[tuple(np.round(c[[0, 10, 17]], 9))] for c in z["casc"]])
    ti = np.array([pos[int(d)] for d in di_cache[rel]])
    sys_mg0, sys_c0 = np.asarray(t["sys_mg0"], float)[ti], np.asarray(t["sys_colour0"], float)[ti]
    dm = sys_mg0 - np.interp(sys_c0, rd["colour"], rd["mag"], left=np.nan, right=np.nan)
    d = 1.0 / par.truth_parallax_mas[z["prow"]]
    w = z["w"]
    out: dict = {"real_twin_over_single": {k: [x["twin_over_single"] for x in ct["by_distance"][k]] for k in ("real_orbits", "real_accels")},
                 "d_edges": ct["d_edges"]}
    for kind, s in (("orbits", z["ob"] >= 0), ("accels", z["ab"] >= 0)):
        vals = []
        for lo, hi in zip(ct["d_edges"][:-1], ct["d_edges"][1:]):
            m = s & (sys_c0 >= 0.85) & (sys_c0 <= 2.0) & (d >= lo) & (d < hi)
            tw = w[m & (dm >= -0.95) & (dm < -0.55)].sum()
            si = w[m & (dm >= -0.25) & (dm < 0.25)].sum()
            ww = w[m & (((dm >= -0.95) & (dm < -0.55)) | ((dm >= -0.25) & (dm < 0.25)))]
            vals.append([float(tw / si) if si > 0 else None, round(float(ww.sum() ** 2 / max(np.sum(ww**2), 1e-300)), 1)])
        out[f"mock_{kind}_twin_over_single_and_ess"] = vals
    rin = z["rob"] >= 0
    ge = [5, 8, 9, 10, 11, 12, 13, 14, 16, 19.5]
    for kind, gm, s, rg in (("orbit", z["m_g"], z["ob"] >= 0, z["r_g"][rin]), ("accel", z["m_g_a"], z["ab"] >= 0, z["ra_g"])):
        rr = []
        for lo, hi in zip(ge[:-1], ge[1:]):
            sel = s & (gm >= lo) & (gm < hi)
            rows, inv = np.unique(z["prow"][sel], return_inverse=True)
            rw = np.bincount(inv, weights=w[sel]) if sel.any() else np.zeros(1)
            sp = z["r_spur_per_orbit"][rin][(rg >= lo) & (rg < hi)].sum() if kind == "orbit" else 0.0
            real = int(((rg >= lo) & (rg < hi)).sum())
            r_ = (rw.sum() + sp) / max(real, 1)
            rr.append([round(float(r_), 3), round(float(rw.sum() / max(real, 1) / np.sqrt(max(rw.sum() ** 2 / max(np.sum(rw**2), 1e-300), 1))), 3)])
        out[f"{kind}_g_ratio_and_row_err"] = {"g_edges": ge, "values": rr}
    acc = (z["ob"] >= 0) & (z["m_g"] < 13)
    f2m = z["casc"][acc, 21]
    o_ = np.argsort(f2m)
    cw = np.cumsum(w[acc][o_]) / w[acc].sum()
    out["F2_G<13"] = {"mock_weighted_median": float(f2m[o_][np.searchsorted(cw, 0.5)]),
                      "real_median": float(np.nanmedian(z["r_cov_goodness_of_fit"][rin & (z["r_g"] < 13)]))}
    cpu = np.asarray(o["cpu_seconds"], float)
    acc_all = np.asarray(o["accepted_orbital"], bool)
    out["cpu"] = {"n_draws": int(cpu.size), "cpu_h": float(cpu.sum() / 3600), "orbits": int(acc_all.sum()),
                  "orbits_per_cpu_h": float(acc_all.sum() / (cpu.sum() / 3600)),
                  "frac_system_in_cut": float(np.mean(np.asarray(t["system_in_cut"], bool)))}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
