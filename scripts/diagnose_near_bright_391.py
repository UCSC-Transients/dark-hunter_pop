#!/usr/bin/env python3
"""Diagnose the bright (G < 10) excess and the near-distance residual (#391; Ryan 2026-10-13). Analysis only.

A. Combined19 E(B−V) (× native_to_ebv 0.884, as the CMD uses it) vs distance for parent rows (Bailer-Jones distance)
   and real C1 orbits (zero-point-corrected parallax) at d < 300 pc; mwdust's grid starts at µ = 4 (63 pc) and is
   extrapolated below it.
B. If E(B−V) -> 0 within ``--d0`` pc: shifts of M_G0 and (BP−RP)_0, evolved-class flips (parent rows, real orbits),
   and the 2-D CMD Malmquist log-weight change for mock draws on affected rows.
C. Bright / near cells at the batch-2 compliant-grid best fit (noW, cmdW): real vs (mock + spurious), by CMD class, ESS.
D. Bright-star detection self-consistency: near-sample re-injection acceptance, real vs mock-symmetric, by G.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_refit4_391 as r4  # noqa: E402
import rung3_refit5_391 as r5  # noqa: E402

ARTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5", "restart_gen26_topup.h5", "restart_gen27_evolved.h5",
        "restart_gen28_recentred.h5", "restart_gen29_topup.h5", "restart_gen30_topup.h5", "restart_gen31_batch1.h5", "restart_gen32_batch2.h5")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_32.npz"))
    ap.add_argument("--fit", type=Path, default=Path("docs/gate391/rung3_topup/batch2_fit/summary.json"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_compliant108.yaml"))
    ap.add_argument("--near-log", type=Path, default=Path("output/detection_vs_population_391/reinjection_near.jsonl"))
    ap.add_argument("--old-cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--d0", type=float, default=100.0, help="E(B-V) -> 0 inside this distance (pc) for test B")
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.new_gen, args.artifacts, args.artifact_dir = 32, list(ARTS), Path("output/proposal_set")
    import yaml
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.elbadry2026_selection import gaia_band_extinction_babusiaux2018
    from darkhunter_pop.giants import classify_evolved, classify_parent, combined19_ebv, load_giants_config, parent_row_cmd

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    ns = r5.extend(args, fc, r4.setup(args, fc))
    cfg = ns.cfg
    dust = cfg.sample_selection.dust_maps
    n2e = float(dust.combined19_native_to_ebv)
    coeffs = dust.gaia_band_extinction.babusiaux2018
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pcmd = parent_row_cmd(parent, cfg, gcfg)
    pcls, ridge = classify_parent(parent, cfg, gcfg, cmd=pcmd)
    pc = parent.columns
    dBJ = np.asarray(pc["r_med_geo"], float) if "r_med_geo" in pc else 1000.0 / parent.truth_parallax_mas
    rep = [f"#391 bright / near diagnosis (analysis only); E(B-V) = Combined19 x {n2e}", ""]

    # ---- A: E(B-V) vs distance ----
    near_p = parent.usable & np.isfinite(dBJ) & (dBJ < 300)
    e_p = combined19_ebv(np.asarray(pc["l"], float)[near_p], np.asarray(pc["b"], float)[near_p], dBJ[near_p], native_to_ebv=n2e)
    rt_ra, rt_dec = None, None
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")

    def col(n):
        return np.ma.filled(np.ma.asarray(rt[n], float), np.nan)

    d_r = ns.ro_cols[1] * 1000.0  # real C1 orbits in the outer edges (kpc -> pc)
    # positions of those orbits: re-derive from extend's masks via covariates (G, d) is not enough; recompute here
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    g_all = col("g_mag")
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    plx = col("parallax") + zp[np.clip(np.digitize(g_all, ge) - 1, 0, 4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        dr_all = np.where(plx > 0, 1000.0 / plx, np.nan)
    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    near_r = np.isfinite(dr_all) & (dr_all < 300) & (col("period") <= ns.PMAX)
    e_r = combined19_ebv(gal.l.deg[near_r], gal.b.deg[near_r], dr_all[near_r], native_to_ebv=n2e)
    edges = np.array([0, 40, 63, 100, 150, 200, 300.0])
    rep.append("A. E(B-V) vs distance (pc): bin: parent n, median, 90th pct, frac > 0.02, frac < 0 | real C1 orbits n, median, 90th pct")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sp = (dBJ[near_p] >= lo) & (dBJ[near_p] < hi)
        sr = (dr_all[near_r] >= lo) & (dr_all[near_r] < hi)
        fp = lambda x, q: float(np.nanpercentile(x, q)) if np.isfinite(x).any() else float("nan")
        rep.append(f"  {lo:.0f}-{hi:.0f}: {int(sp.sum())}, {fp(e_p[sp], 50):.4f}, {fp(e_p[sp], 90):.4f}, {np.mean(e_p[sp] > 0.02) if sp.any() else np.nan:.3f}, "
                   f"{np.mean(e_p[sp] < 0) if sp.any() else np.nan:.3f} | {int(sr.sum())}, {fp(e_r[sr], 50):.4f}, {fp(e_r[sr], 90):.4f}")

    # ---- B: E -> 0 inside d0 ----
    inside_p = near_p.copy()
    inside_p[near_p] = dBJ[near_p] < args.d0
    bprp_p = np.asarray(pc["bp_rp"], float)
    idx_in = np.flatnonzero(inside_p)
    e_in = combined19_ebv(np.asarray(pc["l"], float)[idx_in], np.asarray(pc["b"], float)[idx_in], dBJ[idx_in], native_to_ebv=n2e)
    a_g, e_br, ok = gaia_band_extinction_babusiaux2018(bprp_p[idx_in], e_in, r_v=float(dust.r_v), coeffs=coeffs)
    a_g = np.where(ok, a_g, np.nan)
    e_br = np.where(ok, e_br, np.nan)
    rep += ["", f"B. E(B-V) -> 0 inside {args.d0:.0f} pc: {idx_in.size} usable parent rows ({idx_in.size / parent.usable.sum():.4%} of usable); "
            f"A_G median {np.nanmedian(a_g):.4f} (90th {np.nanpercentile(a_g, 90):.4f}), E(BP-RP) median {np.nanmedian(e_br):.4f} (90th {np.nanpercentile(e_br, 90):.4f}) mag; "
            f"setting E = 0 makes M_G0 fainter by A_G and (BP-RP)_0 redder by E(BP-RP)"]
    mg0 = np.asarray(pcmd.mg0, float).copy()
    c0 = np.asarray(pcmd.colour0, float).copy()
    mg0n, c0n = mg0.copy(), c0.copy()
    mg0n[idx_in] = mg0[idx_in] + np.nan_to_num(a_g)
    c0n[idx_in] = c0[idx_in] + np.nan_to_num(e_br)
    cls_new = classify_evolved(mg0n, c0n, np.asarray(pcmd.sigma_mu, float), ridge, gcfg.provisional_n_sigma)
    ev_old = pcls.classified & pcls.evolved
    ev_new = cls_new.classified & cls_new.evolved
    rep.append(f"  parent evolved flags changed inside {args.d0:.0f} pc: {int(np.sum(ev_old[idx_in] != ev_new[idx_in]))} of {idx_in.size} "
               f"(evolved {int(ev_old[idx_in].sum())} -> {int(ev_new[idx_in].sum())})")
    # mock draws on affected rows and their weight share in the fit
    fits = json.loads(args.fit.read_text())["results"]
    z = np.load(args.cache, allow_pickle=True)
    prow = np.concatenate([np.asarray(ps.read_proposal_artifact(args.artifact_dir / a)[0]["parent_row"], np.int64) for a in ARTS])[ns.rel]
    aff = np.isin(prow, idx_in)
    for wkey in ("noW", "cmdW"):
        th = np.array(fits[wkey]["theta"])
        t = np.zeros(14)
        t[:13] = th[:13]
        w = ns.W[wkey] * np.exp(ns.log_mod(t) + th[13] * ns.m_cls)
        so = ns.in_o
        rep.append(f"  [{wkey}] mock C1-orbit weight on rows inside {args.d0:.0f} pc: {w[so & aff].sum() / w[so].sum():.4%}; "
                   f"real C1 orbits inside: {np.mean(ns.ro_cols[1] * 1000 < args.d0):.4%}")
    # CMD Malmquist log-weight change on affected draws (row-level CMD replaced)
    try:
        from darkhunter_pop import malmquist_cmd as mc

        tg = ps.load_proposal_set_fragment(args.fragment).target_mds17
        mcfg = mc.load_cmd_malmquist_config(Path("config/population/malmquist_cmd.yaml"))
        parts = [ps.read_proposal_artifact(args.artifact_dir / a)[0] for a in ARTS]
        keys_common = set.intersection(*(set(p) for p in parts))
        tt = {k: np.concatenate([np.asarray(p[k]) for p in parts])[ns.rel][aff] for k in parts[0] if k in keys_common}
        lw_old, _ = ps.malmquist_cmd_log_weight(tt, parent, tg, mcfg, cfg)
        par2 = dataclasses.replace(parent, cmd={**dict(parent.cmd), "mg0": mg0n, "colour0": c0n}) if parent.cmd is not None else parent
        lw_new, _ = ps.malmquist_cmd_log_weight(tt, par2, tg, mcfg, cfg)
        dl = lw_new - lw_old
        rep.append(f"  CMD Malmquist log-weight change on the {int(aff.sum())} affected draws: median {np.nanmedian(dl):+.3f}, "
                   f"5-95% {np.nanpercentile(dl, 5):+.3f} .. {np.nanpercentile(dl, 95):+.3f}")
    except Exception as exc:  # report, do not fail the diagnosis
        rep.append(f"  CMD Malmquist log-weight change: not evaluated ({exc!r})")
    # real orbits: class flips inside d0
    rin = ns.ro_cols[1] * 1000 < args.d0
    rep.append(f"  real C1 orbits inside {args.d0:.0f} pc: {int(rin.sum())}; evolved among them now {int(ns.r_cls[rin].sum())} "
               f"(the CMD shift of {np.nanmedian(a_g):.3f} mag in M_G0 is far below the classifier margin)")

    # ---- C: bright / near cells at the batch-2 fit ----
    rep += ["", "C. bright / near cells at the batch-2 compliant-grid fit: real n | (mock+spurious)/real | mock ESS (noW; cmdW)"]
    rg, rd = ns.ro_cols[0], ns.ro_cols[1]
    mg, md = ns.o_cols[0], ns.o_cols[1]
    so = ns.in_o
    for wkey in ("noW", "cmdW"):
        th = np.array(fits[wkey]["theta"])
        t = np.zeros(14)
        t[:13] = th[:13]
        w = ns.W[wkey] * np.exp(ns.log_mod(t) + th[13] * ns.m_cls)
        share = th[14] / rg.size
        for (glo, ghi) in ((5, 8), (8, 10), (10, 12)):
            for (dlo, dhi) in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 6)):
                for cls in (0, 1):
                    rs = (rg >= glo) & (rg < ghi) & (rd >= dlo) & (rd < dhi) & (ns.r_cls == cls)
                    ms = so & (mg >= glo) & (mg < ghi) & (md >= dlo) & (md < dhi) & (ns.m_cls == cls)
                    if rs.sum() < 50:
                        continue
                    wm = w[ms]
                    ess = wm.sum() ** 2 / max(np.sum(wm**2), 1e-300)
                    rep.append(f"  [{wkey}] G {glo}-{ghi} d {dlo}-{dhi} {'evolved' if cls else 'dwarf'}: {int(rs.sum())} | "
                               f"{(wm.sum() + share * rs.sum()) / rs.sum():.2f} | {ess:.0f}")

    # ---- D: bright-star detection self-consistency (near re-injection) ----
    OFF = 2 * 10**15
    recs = [json.loads(l_) for l_ in args.near_log.read_text().splitlines()]
    acc = lambda r: bool(r["accepted"]) and r["period"] is not None and np.isfinite(r["period"]) and r["period"] <= 830.4
    import h5py

    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        sid, gp = h["published/source_id"][()], h["published/g_mag"][()]
    pos = {int(s): i for i, s in enumerate(sid)}
    oc = np.load(args.old_cache, allow_pickle=True)
    gm_old = np.asarray(oc["phot_g_mean_mag"], float)
    real = {}
    mock = {}
    for r in recs:
        s = int(r["source_id"])
        if OFF <= s < OFF + 10**7:
            mock.setdefault(s, []).append(acc(r))
        elif not (10**15 <= s < OFF):
            real.setdefault(s, []).append(acc(r))
    rG = np.array([gp[pos[s]] for s in real])
    rA = np.array([np.mean(v) for v in real.values()])
    mG = np.array([gm_old[s - OFF] for s in mock])
    mA = np.array([np.mean(v) for v in mock.values()])
    rep += ["", "D. near (d < 0.5 kpc) re-injection self-consistency by G: real acceptance (n) | mock-symmetric acceptance (n)"]
    for lo, hi in ((5, 10), (10, 12), (12, 14), (14, 20)):
        a, b = (rG >= lo) & (rG < hi), (mG >= lo) & (mG < hi)
        rep.append(f"  G {lo}-{hi}: {rA[a].mean() if a.any() else np.nan:.2f} ({int(a.sum())}) | {mA[b].mean() if b.any() else np.nan:.2f} ({int(b.sum())})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
