#!/usr/bin/env python
"""#391 (Ryan 2026-10-18) test 0: dereddened-CMD shape and equal-mass (twin) sequence across distance and
apparent-magnitude slices, for (i) the parent, (ii) real NSS orbit / acceleration hosts and (iii) mock-detected
systems (best-fit weights, at the system's observed CMD = its parent row's CMD, MP-Q6/Q36). Plus the isochrone
M̂1 and its failures vs the offset from the MS ridge, and the < 0.8 M⊙ rows at G < 10.

ΔM = M_G0 − ridge(C0) (``parent.meta['ms_ridge']``, the §10.2 ridge). Twin band ΔM ∈ [−0.95, −0.55], single band
ΔM ∈ [−0.25, 0.25], counted only on the MS below the subgiants: C0 ∈ [``--c-min``, ``--c-max``] (ridge M_G0 about
4.7–9.5), where a twin of two MS stars stays on the MS. Writes ``cmd_twin.json`` and arrays for the contact sheet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D_EDGES = [0.0, 0.1, 0.2, 0.5, 1.0, 3.0]
G_SLICES = [5.0, 10.0, 13.0, 16.0, 19.5]
TWIN = (-0.95, -0.55)
SINGLE = (-0.25, 0.25)
DM_EDGES = np.arange(-2.0, 1.01, 0.05)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    ap.add_argument("--n-real", type=int, default=30000, help="random real hosts per kind at G >= 11 (all G < 11 kept)")
    ap.add_argument("--c-min", type=float, default=0.85)
    ap.add_argument("--c-max", type=float, default=2.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import giants
    from darkhunter_pop import isochrone_mass as im
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    gcfg = giants.load_giants_config(Path("config/population/giants.yaml"))
    rd = parent.meta["ms_ridge"]
    ridge = giants.MSRidge(colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float),
                           sigma=np.asarray(rd["sigma"], float), n_rows=np.asarray(rd["n_rows"]))
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])

    def cmd_real(t, fit=True):
        def col(n):
            return np.ma.filled(np.ma.asarray(t[n], float), np.nan)
        g = col("g_mag")
        bprp = col("bp_mag") - col("rp_mag")
        plx = col("parallax") + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
        eplx = col("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(plx > 0, 1000 / plx, np.nan)
            rlo = np.where(plx > 0, 1000 / (plx + eplx), np.nan)
            rhi = np.where(plx - eplx > 0, 1000 / (plx - eplx), np.nan)
        gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
        c = giants.cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
        out = {"g": g, "d": rr / 1000, "mg0": c.mg0, "c0": c.colour0}
        if fit:
            d = model.fit(c.colour0, c.mg0, sigma_mu=c.sigma_mu, ebv=c.ebv, a_g=c.a_g, e_bp_rp=bprp - c.colour0).as_dict()
            out.update(m1=np.asarray(d["m1_mean"], float), ok=np.asarray(d["ok"], bool), feh=np.asarray(d["feh_mean"], float),
                       reason=np.asarray(d["reason"]).astype(str))
        return out

    z = np.load(args.dump, allow_pickle=True)
    rng = np.random.default_rng(391)

    # (ii) real hosts
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    pos = {int(s): i for i, s in enumerate(np.asarray(rt["source_id"], np.int64))}
    rin = np.flatnonzero(z["rob"] >= 0)
    keep = np.concatenate([rin[z["r_g"][rin] < 11], rng.choice(rin[z["r_g"][rin] >= 11], size=args.n_real, replace=False)])
    w_real_o = np.where(z["r_g"][keep] < 11, 1.0, (z["r_g"][rin] >= 11).sum() / args.n_real)
    ro = cmd_real(rt[np.array([pos[int(s)] for s in z["r_sid"][keep]])])
    del rt
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    apos = {int(s): i for i, s in enumerate(np.asarray(at["source_id"], np.int64))}
    ain = np.arange(z["ra_g"].size)
    akeep = np.concatenate([ain[z["ra_g"] < 11], rng.choice(ain[z["ra_g"] >= 11], size=args.n_real, replace=False)])
    w_real_a = np.where(z["ra_g"][akeep] < 11, 1.0, (z["ra_g"] >= 11).sum() / args.n_real)
    ra = cmd_real(at[np.array([apos[int(s)] for s in z["ra_sid"][akeep]])])
    del at

    # (i) parent (usable rows) and (iii) mock-detected (parent-row CMD, best-fit weight)
    usable = parent.usable
    p = {"g": pg, "d": 1.0 / parent.truth_parallax_mas, "mg0": np.asarray(parent.cmd["mg0"], float), "c0": np.asarray(parent.cmd["colour0"], float),
         "m1": np.asarray(parent.isochrone["m1_mean"], float), "ok": np.asarray(parent.isochrone["ok"], bool),
         "feh": np.asarray(parent.isochrone["feh_mean"], float), "reason": np.asarray(parent.isochrone["reason"]).astype(str)}
    prow = z["prow"]
    w = z["w"]
    so, sa = z["ob"] >= 0, z["ab"] >= 0

    def dm(s):
        r, _, inr = ridge.at(s["c0"])
        return np.where(inr, s["mg0"] - r, np.nan)

    for s in (p, ro, ra):
        s["dm"] = dm(s)
    sets = {
        "parent": (p, usable.astype(float)),
        "real_orbits": (ro, w_real_o),
        "real_accels": (ra, w_real_a),
        "mock_orbits": ({k: v[prow[so]] for k, v in p.items()}, w[so]),
        "mock_accels": ({k: v[prow[sa]] for k, v in p.items()}, w[sa]),
    }
    # mock d/G: the system's observed G (= row G) and the mock's measured distance are close to the row's; use row values

    def slice_stats(s, wt, sel):
        ms = sel & np.isfinite(s["dm"]) & (s["c0"] >= args.c_min) & (s["c0"] <= args.c_max) & (wt > 0)
        tw = ms & (s["dm"] >= TWIN[0]) & (s["dm"] < TWIN[1])
        si = ms & (s["dm"] >= SINGLE[0]) & (s["dm"] < SINGLE[1])
        n_tw, n_si = float(wt[tw].sum()), float(wt[si].sum())
        h = np.histogram(s["dm"][ms], bins=DM_EDGES, weights=wt[ms])[0]
        return {"n_ms_rows": int(ms.sum()), "twin_over_single": n_tw / n_si if n_si > 0 else np.nan,
                "n_twin_rows": int(tw.sum()), "n_single_rows": int(si.sum()),
                "dm_hist": (h / h.sum()).tolist() if h.sum() > 0 else []}

    res: dict = {"twin_band": TWIN, "single_band": SINGLE, "ms_colour_range": [args.c_min, args.c_max], "dm_edges": DM_EDGES.tolist(),
                 "d_edges": D_EDGES, "g_slices": G_SLICES, "by_distance": {}, "by_g": {}}
    for nm, (s, wt) in sets.items():
        res["by_distance"][nm] = [slice_stats(s, wt, (s["d"] >= a) & (s["d"] < b)) for a, b in zip(D_EDGES[:-1], D_EDGES[1:])]
        res["by_g"][nm] = [slice_stats(s, wt, (s["g"] >= a) & (s["g"] < b)) for a, b in zip(G_SLICES[:-1], G_SLICES[1:])]

    # isochrone M̂1 / failures vs ΔM (parent rows, MS colour range and all colours) by G
    dmb = [-3.0, -1.5, -0.95, -0.55, -0.25, 0.25, 1.0]
    iso = {"dm_edges": dmb, "by_g": {}}
    for a, b in ((5.0, 10.0), (10.0, 13.0), (13.0, 16.0)):
        g_sel = (p["g"] >= a) & (p["g"] < b) & np.isfinite(p["dm"])
        rows = []
        for lo, hi in zip(dmb[:-1], dmb[1:]):
            s_ = g_sel & (p["dm"] >= lo) & (p["dm"] < hi)
            ms_ = s_ & (p["c0"] >= args.c_min) & (p["c0"] <= args.c_max)
            okm = ms_ & p["ok"]
            rows.append({"n": int(s_.sum()), "ok_frac": float(p["ok"][s_].mean()) if s_.any() else np.nan,
                         "n_ms": int(ms_.sum()), "ok_frac_ms": float(p["ok"][ms_].mean()) if ms_.any() else np.nan,
                         "m1_median_ms": float(np.median(p["m1"][okm])) if okm.any() else np.nan,
                         "feh_median_ms": float(np.median(p["feh"][okm])) if okm.any() else np.nan})
        # failures: reason and where (ΔM, C0) for this G slice
        fail = (p["g"] >= a) & (p["g"] < b) & ~p["ok"]
        reasons, cnt = np.unique(p["reason"][fail], return_counts=True)
        iso["by_g"][f"{a:g}-{b:g}"] = {"rows": rows, "fail_reasons": dict(zip(reasons.tolist(), cnt.tolist())),
                                       "fail_dm_p10_50_90": np.nanpercentile(p["dm"][fail], [10, 50, 90]).tolist() if np.isfinite(p["dm"][fail]).any() else [],
                                       "fail_c0_p10_50_90": np.nanpercentile(p["c0"][fail], [10, 50, 90]).tolist() if fail.any() else [],
                                       "fail_mg0_p10_50_90": np.nanpercentile(p["mg0"][fail], [10, 50, 90]).tolist() if fail.any() else [],
                                       "fail_frac_outside_ridge_colour": float(np.mean(~np.isfinite(p["dm"][fail]))) if fail.any() else np.nan}
    # real hosts: M̂1 / failures vs ΔM at G < 10
    for nm, s in (("real_orbits", ro), ("real_accels", ra)):
        g_sel = (s["g"] < 10) & np.isfinite(s["dm"])
        iso[f"{nm}_G<10"] = [{"n": int((g_sel & (s["dm"] >= lo) & (s["dm"] < hi)).sum()),
                              "ok_frac": float(s["ok"][g_sel & (s["dm"] >= lo) & (s["dm"] < hi)].mean()) if (g_sel & (s["dm"] >= lo) & (s["dm"] < hi)).any() else np.nan}
                             for lo, hi in zip(dmb[:-1], dmb[1:])]
    res["isochrone_vs_dm"] = iso

    # < 0.8 M⊙ rows at G < 10: mock-detected (parent-row M̂1) and real hosts
    low = {}
    for nm, (s, wt) in (("mock_orbits", sets["mock_orbits"]), ("mock_accels", sets["mock_accels"]),
                        ("real_orbits", (ro, w_real_o)), ("real_accels", (ra, w_real_a))):
        sel = (s["g"] < 10) & (s["m1"] < 0.8) & (wt > 0)
        if "ok" in s:
            sel &= s["ok"]
        tot = float(wt[(s["g"] < 10) & (wt > 0)].sum())
        low[nm] = {"weight_share": float(wt[sel].sum()) / tot if tot > 0 else np.nan, "n_rows": int(sel.sum()),
                   "n_unique_rows": int(np.unique(np.flatnonzero(sel)).size),
                   "c0_p10_50_90": np.nanpercentile(s["c0"][sel], [10, 50, 90]).tolist() if sel.any() else [],
                   "mg0_p10_50_90": np.nanpercentile(s["mg0"][sel], [10, 50, 90]).tolist() if sel.any() else [],
                   "dm_p10_50_90": np.nanpercentile(s["dm"][sel], [10, 50, 90]).tolist() if np.isfinite(s["dm"][sel]).any() else [],
                   "d_kpc_p10_50_90": np.nanpercentile(s["d"][sel], [10, 50, 90]).tolist() if sel.any() else [],
                   "frac_dm_lt_minus1": float(np.sum(wt[sel & (s["dm"] < -1.0)]) / max(np.sum(wt[sel]), 1e-300))}
    if True:  # mock: unique parent rows behind the < 0.8 G < 10 weight
        sel = (sets["mock_orbits"][0]["g"] < 10) & (sets["mock_orbits"][0]["m1"] < 0.8)
        rows_ = prow[so][sel]
        wsum = np.bincount(np.unique(rows_, return_inverse=True)[1], weights=w[so][sel])
        low["mock_orbits"]["n_parent_rows"] = int(np.unique(rows_).size)
        low["mock_orbits"]["top_row_share"] = float(wsum.max() / wsum.sum()) if wsum.size else np.nan
    res["low_mass_G<10"] = low
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cmd_twin.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(args.out_dir / "cmd_points.npz",
                        p_c0=p["c0"][usable], p_mg0=p["mg0"][usable], p_d=p["d"][usable], p_g=p["g"][usable],
                        ro_c0=ro["c0"], ro_mg0=ro["mg0"], ro_d=ro["d"], ro_g=ro["g"], ro_w=w_real_o,
                        mo_c0=p["c0"][prow[so]], mo_mg0=p["mg0"][prow[so]], mo_d=p["d"][prow[so]], mo_g=p["g"][prow[so]], mo_w=w[so],
                        ridge_c=ridge.colour, ridge_m=ridge.mag)
    print(json.dumps({k: res[k] for k in ("by_distance", "by_g")}, default=float)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
