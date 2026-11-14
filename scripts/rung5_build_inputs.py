#!/usr/bin/env python3
"""R5-T11 (#644): build the rung-5 fit inputs (cells, real counts, templates, responses) — HEAVY (M2_app MC on mock orbits).

Cells: frozen M2_app edges (``rung5_bins_dr3.yaml``, ``rung5_bins.assert_frozen``) × A = (a0/ϖ)/P_yr^(2/3) edges
(0, 0.5, ∞) × distance groups by 1/ϖ (0, 0.3, 0.7, 1.5, ∞ kpc). Domain: accepted Orbital / AstroSpectroSB1 orbits with
P_fit ≤ the C1 ceiling and ``--g-min`` ≤ G (G < 10 excluded, §12.36); real rows also pass ``real_comparison_keep`` (the
decided parent filters, as in rung 3).

* **Real counts**: ``real_m2app_dr3.h5`` (reason ``ok``; adopted ν #638) with A, 1/ϖ and G from the published orbits.
* **CO responses** (dark; WD per §12.39 branch): co_v1 gens of the adopted noise version, ``--partition`` train (fit)
  or holdout (gate-5b universe); deterministic-mixture weights under θ_CO; M2_app on the mock orbits with the SAME
  estimator as the real rows (``compact_objects.recover_masses``, Student-t, adopted ν). Per true-mass bin j:
  R_jb = N_prim Σ_{i∈j,b} w_i r_i / Σ_{i∈j} w_i r_i, with r_i = 1 (dark) or the §12.39 branch density over the θ_CO WD
  density (``joint_refit.wd_physics_log_density`` × the cooling-age law), and I_kj = Σ_{i∈j} w_i r_i / N_prim.
* **Luminous template**: luminous gens of the adopted version reweighted with ``--lum-weights`` (an npz of
  generation, draw_index, weight at the rung-3 v5 θ, exported by the rung-3 fit); M2_app by the f = 0 point inversion at
  the parent row's isochrone M1 (the luminous gens store no orbit covariance): a declared stand-in whose size is
  measured on the CO orbits (point inversion vs the Student-t median) and recorded in the inputs' meta.
* **Spurious template**: real C1 orbits weighted by π̂ (``--pihat`` npz: source_id, pihat; the §12.16 four-sample
  template of #634 / #636), normalised to unit sum; Σπ̂ in the domain is the N_s prior scale.

Never reads the old-ν tables (``sf_1006_1008/``); the M2_app config is ``config/population/rung4_co_v1.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

A_EDGES = np.array([0.0, 0.5, np.inf])
D_EDGES_KPC = np.array([0.0, 0.3, 0.7, 1.5, np.inf])


def m2_point_inversion(a0_mas, parallax_mas, period_day, m1_msun) -> np.ndarray:
    """Dark-companion M2 from the astrometric mass function M2³/(M1+M2)² = (a0/ϖ)³/P_yr² (bisection, vectorised)."""
    a0, plx, p, m1 = (np.asarray(x, float) for x in (a0_mas, parallax_mas, period_day, m1_msun))
    with np.errstate(divide="ignore", invalid="ignore"):
        mf = (a0 / plx) ** 3 / (p / 365.25) ** 2
    lo, hi = np.full(mf.shape, 1e-4), np.full(mf.shape, 1e4)
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        f = mid**3 / (m1 + mid) ** 2 - mf
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f >= 0, mid, hi)
    out = np.sqrt(lo * hi)
    return np.where(np.isfinite(mf) & (mf > 0) & np.isfinite(m1) & (m1 > 0), out, np.nan)


def cell_index(m2_app, a_size, d_kpc, m2_edges) -> np.ndarray:
    """Flat cell index (M2_app × A × d), −1 outside."""
    im = np.searchsorted(m2_edges, m2_app, side="right") - 1
    ia = np.searchsorted(A_EDGES, a_size, side="right") - 1
    idd = np.searchsorted(D_EDGES_KPC, d_kpc, side="right") - 1
    nm, na, nd = len(m2_edges) - 1, A_EDGES.size - 1, D_EDGES_KPC.size - 1
    ok = (im >= 0) & (im < nm) & (ia >= 0) & (ia < na) & (idd >= 0) & (idd < nd)
    ok &= np.isfinite(m2_app) & np.isfinite(a_size) & np.isfinite(d_kpc)
    return np.where(ok, (im * na + ia) * nd + idd, -1)


def n_cells(m2_edges) -> int:
    return (len(m2_edges) - 1) * (A_EDGES.size - 1) * (D_EDGES_KPC.size - 1)


def high_a_mask(m2_edges, m2_min_high: float) -> np.ndarray:
    """§2.6 s_A cells: A > 0.5, or M2_app ≥ m2_min_high."""
    nm, na, nd = len(m2_edges) - 1, A_EDGES.size - 1, D_EDGES_KPC.size - 1
    im, ia, _ = np.unravel_index(np.arange(nm * na * nd), (nm, na, nd))
    return (ia == 1) | (np.asarray(m2_edges)[im] >= m2_min_high)


def binned_response(cell, true_bin, n_true, w, r, n_cells_: int, n_prim: float):
    """(R[j,b], V[j,b], I[j]) — see the module docstring.

    V_jb = N_prim² Σ_{i∈j,b} (w_i r_i / Σ_{i∈j} w_i r_i)². The weights are normalised *before* squaring: a branch whose
    density is ~1e-300 in a bin (e.g. ``mt_he`` above the He-WD masses) has Σ w r ~ 1e-300, whose square underflows to
    0, and the former ``Σ (w r)² / tot²`` returned 0/0 = NaN there (#659). A bin with no weight (Σ w r ≤ 0, including
    no draws) has R = V = 0. Raises ``ValueError`` on a non-finite weight sum or result.
    """
    R = np.zeros((n_true, n_cells_))
    V = np.zeros((n_true, n_cells_))
    I = np.zeros(n_true)
    wr = w * r
    for j in range(n_true):
        sj = true_bin == j
        tot = float(np.sum(wr[sj]))
        if not math.isfinite(tot):
            raise ValueError(f"binned_response: non-finite weight sum in true bin {j}")
        I[j] = tot / n_prim if n_prim > 0 else 0.0
        if tot <= 0:
            continue
        sd = sj & (cell >= 0)
        u = wr[sd] / tot
        R[j] = n_prim * np.bincount(cell[sd], weights=u, minlength=n_cells_)
        V[j] = n_prim**2 * np.bincount(cell[sd], weights=u * u, minlength=n_cells_)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(V))):
        raise ValueError("binned_response: non-finite response or variance")
    return R, V, I


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bins-config", type=Path, default=Path("config/population/rung5_bins_dr3_st2.yaml"))
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--real-m2app", type=Path, default=Path("output/rung5/real_m2app_dr3.h5"))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--real-snapshot", type=Path, required=True, help="the uncut NSS query.ecsv (rung-3 real side)")
    ap.add_argument("--real-input-columns", type=Path, required=True, help="nss_orbit_input_columns snapshot dir")
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--co-artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--co-fragment", type=Path, nargs="+", required=True)
    ap.add_argument("--lum-artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--lum-weights", type=Path, required=True, help="npz: generation, draw_index, weight at the rung-3 v5 θ")
    ap.add_argument("--pihat", type=Path, required=True, help="npz: source_id, pihat (§12.16 four-sample template)")
    ap.add_argument("--wd-physics-config", type=Path, required=True, help="rung-3 fit config with joint.wd_physics")
    ap.add_argument("--ln-r-center", type=float, required=True, help="rung-3 v5 ln A_ni − ln R_mt (prior centre only)")
    ap.add_argument("--partition", choices=("train", "holdout"), default="train")
    ap.add_argument("--g-min", type=float, default=10.0)
    ap.add_argument("--m2-min-high", type=float, default=1.4)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import h5py

    from darkhunter_pop import compact_objects as cob
    from darkhunter_pop import joint_refit as jr
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop import rung5_bins as r5b
    from darkhunter_pop import rung5_fit as rf
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes

    bins = r5b.load_rung5_bins(a.bins_config)
    der = r5b.assert_frozen(bins)
    m2e = np.asarray(der.m2_app_edges_msun, float)
    dark_e, wd_e = np.asarray(der.true_m2_edges_msun["dark"], float), np.asarray(der.true_m2_edges_msun["wd"], float)
    pmax = bins.c1_period_max_days
    nc = n_cells(m2e)
    r4 = cob.load_rung4_config(a.rung4_config)
    cfg = load_config(a.config)

    # ---- real counts ------------------------------------------------------------------------------------------
    with h5py.File(a.real_m2app, "r") as h:
        g = h["m2app"]
        rsid = g["source_id"][()].astype(np.int64)
        rm2 = g["m2_med"][()]
        rok = g["reason"][()].astype(str) == "ok"
        nu_json = str(h.attrs.get("mass_recovery_json", ""))
    if json.loads(nu_json).get("student_t_nu") != list(r4.mass_recovery.student_t_nu):
        raise SystemExit(f"{a.real_m2app} was built with a different ν than {a.rung4_config} (old-ν input refused)")
    with h5py.File(a.published_orbits, "r") as h:
        p = h["published"]
        psid = p["source_id"][()].astype(np.int64)
        pa0 = photocenter_a0_from_thiele_innes(p["a_thiele_innes"][()], p["b_thiele_innes"][()], p["f_thiele_innes"][()], p["g_thiele_innes"][()])
        pplx, pper, pg = p["parallax"][()], p["period"][()], p["g_mag"][()]
    pos = {int(s): i for i, s in enumerate(psid)}
    ix = np.array([pos.get(int(s), -1) for s in rsid])
    okp = ix >= 0
    j = np.clip(ix, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ra = np.where(okp, (pa0[j] / pplx[j]) / (pper[j] / 365.25) ** (2 / 3), np.nan)
        rd = np.where(okp, 1.0 / pplx[j], np.nan)
    # mirror the decided parent filters on the real side, exactly as rung 3 does (ps.real_comparison_keep)
    from astropy.table import Table

    rt = Table.read(a.real_snapshot, format="ascii.ecsv")
    frag0 = ps.load_proposal_set_fragment(a.co_fragment[0])
    keep_rt, keep_counts = ps.real_comparison_keep(rt, frag0.proposal, ps.load_real_input_columns(a.real_input_columns))
    keep_ids = set(np.asarray(rt["source_id"], np.int64)[keep_rt].tolist())
    rkeep = np.array([int(x) in keep_ids for x in rsid])
    rdom = rok & okp & rkeep & (pper[j] <= pmax) & (pg[j] >= a.g_min)
    rcell = np.where(rdom, cell_index(rm2, ra, rd, m2e), -1)
    k_obs = np.bincount(rcell[rcell >= 0], minlength=nc).astype(float)

    # ---- spurious template ------------------------------------------------------------------------------------
    zp = np.load(a.pihat)
    pih = dict(zip(np.asarray(zp["source_id"], np.int64).tolist(), np.asarray(zp["pihat"], float).tolist()))
    pv = np.array([pih.get(int(s), np.nan) for s in rsid])
    sel = rcell >= 0
    if np.any(~np.isfinite(pv[sel])):
        raise SystemExit(f"{int(np.sum(~np.isfinite(pv[sel])))} domain rows have no π̂ in {a.pihat}")
    tsum = np.bincount(rcell[sel], weights=pv[sel], minlength=nc)
    pihat_sum = float(tsum.sum())
    spur = tsum / pihat_sum
    spur_var = np.bincount(rcell[sel], weights=pv[sel] ** 2, minlength=nc) / pihat_sum**2

    # ---- compact-object responses -------------------------------------------------------------------------------
    frags = [ps.load_proposal_set_fragment(f) for f in a.co_fragment]
    gens = [f.proposal for f in frags]
    co = cob.check_co_generations(gens)
    target = frags[0].target_co
    assert target is not None
    parent = ps.load_parent_snapshot(a.parent_dir, cfg, gens[0])
    parts = [ps.read_proposal_artifact(x)[:2] for x in a.co_artifact]
    tk = set.intersection(*(set(q[0]) for q in parts))
    ok_ = set.intersection(*(set(q[1]) for q in parts))
    t = {k: np.concatenate([q[0][k] for q in parts]) for k in tk}
    o = {k: np.concatenate([q[1][k] for q in parts]) for k in ok_}
    w = cob.co_importance_weights(t, parent, gens, target, partition=a.partition)
    casc = np.asarray(o["cascade"], float)
    acc = np.asarray(o["accepted_orbital"], bool) & (casc[:, 10] <= pmax) & (np.asarray(t["phot_g_mean_mag"], float) >= a.g_min)
    rec = cob.recover_masses(t, o, parent, r4.mass_recovery, select=acc & (w > 0))
    m2app = rec["m2_med"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ca = (casc[:, 17] / casc[:, 0]) / (casc[:, 10] / 365.25) ** (2 / 3)
        cd = 1.0 / casc[:, 0]
    ccell = np.where(acc, cell_index(m2app, ca, cd, m2e), -1)
    comp = np.asarray(t["co_component"])
    m2t = np.asarray(t["m2_msun"], float)
    rows_m1 = np.asarray(parent.isochrone["log_m1_mean"], float)[np.asarray(t["parent_row"], np.int64)]
    m2pt = m2_point_inversion(casc[:, 17], casc[:, 0], casc[:, 10], 10.0**rows_m1)
    fin = acc & np.isfinite(m2app) & np.isfinite(m2pt) & (w > 0)
    stand_in = {"luminous_m2app_point_inversion": {
        "median_ratio_point_over_studentt": float(np.median(m2pt[fin] / m2app[fin])) if fin.any() else None,
        "p16_p84": np.percentile(m2pt[fin] / m2app[fin], [16, 84]).tolist() if fin.any() else None,
        "n": int(fin.sum())}}

    dk = comp == cob.COMPONENT_CODES["dark"]
    n_prim_dark = float(np.sum(w[dk]))
    jd = np.where(dk, np.searchsorted(dark_e, m2t, side="right") - 1, -1)
    jd = np.where((jd >= 0) & (jd < dark_e.size - 1), jd, -1)
    dark, dark_var, _ = binned_response(np.where(dk, ccell, -1), jd, dark_e.size - 1, np.where(dk, w, 0.0), np.ones(w.size), nc, n_prim_dark)

    wdm = comp == cob.COMPONENT_CODES["wd"]
    n_prim_wd = float(np.sum(w[wdm]))
    jw = np.where(wdm, np.searchsorted(wd_e, m2t, side="right") - 1, -1)
    jw = np.where((jw >= 0) & (jw < wd_e.size - 1), jw, -1)
    wpc = jr.WDPhysicsConfig(**yaml.safe_load(a.wd_physics_config.read_text())["rung3"]["joint"]["wd_physics"])
    sub = {k: np.asarray(t[k])[wdm] for k in t}
    t_sys = 10.0 ** np.asarray(sub["iso_log_age"], float) / 1e9
    lcool = cob.wd_log_cooling_age_target({k: sub[k] for k in ("wd_cooling_age_gyr", "iso_log_age", "m2_msun")}, target.wd_cooling_age)
    lco = cob.co_log_target(sub, target, None)
    branch_kw = {"ni": dict(ln_a_ni=0.0, ln_r_mt=-1e9, f_he=0.5), "mt_he": dict(ln_a_ni=-1e9, ln_r_mt=0.0, f_he=1.0),
                 "mt_co": dict(ln_a_ni=-1e9, ln_r_mt=0.0, f_he=0.0)}
    wd = np.zeros((3, wd_e.size - 1, nc))
    wd_var = np.zeros_like(wd)
    wd_rate = np.zeros((3, wd_e.size - 1))
    for kk, br in enumerate(rf.WD_BRANCHES):
        lam = jr.wd_physics_log_density(sub["m2_msun"], sub["period_days"], sub["eccentricity"], sub["m1_msun"], t_sys,
                                        cfg=wpc, **branch_kw[br])
        rr = np.zeros(w.size)
        with np.errstate(over="ignore", invalid="ignore"):
            rr[wdm] = np.where(np.isfinite(lam + lcool - lco), np.exp(lam + lcool - lco), 0.0)
        wd[kk], wd_var[kk], wd_rate[kk] = binned_response(np.where(wdm, ccell, -1), jw, wd_e.size - 1, np.where(wdm, w, 0.0), rr, nc, n_prim_wd)

    # ---- luminous template --------------------------------------------------------------------------------------
    lw = np.load(a.lum_weights)
    wkey = {(int(gg), int(dd)): float(ww) for gg, dd, ww in zip(lw["generation"], lw["draw_index"], lw["weight"])}
    lum = np.zeros((D_EDGES_KPC.size - 1, nc))
    lum_var = np.zeros_like(lum)
    with h5py.File(a.co_artifact[0], "r") as h:
        parent_sha = str(h.attrs["parent_h5_sha256"])
    for art in a.lum_artifact:
        with h5py.File(art, "r") as h:
            if str(h.attrs["parent_h5_sha256"]) != parent_sha:
                raise SystemExit(f"{art}: parent snapshot differs from the CO artifacts' (parent_row indices would not match)")
            accl = np.asarray(h["outcome/accepted_orbital"][()], bool)
            idx = np.flatnonzero(accl)
            cl = np.asarray(h["outcome/cascade"][idx, :], float)
            gen = np.asarray(h["truth/generation"][()])[idx]
            di = np.asarray(h["truth/draw_index"][()])[idx]
            gm = np.asarray(h["truth/phot_g_mean_mag"][()])[idx]
            prow = np.asarray(h["truth/parent_row"][()])[idx].astype(np.int64)
        ww = np.array([wkey.get((int(x), int(y)), 0.0) for x, y in zip(gen, di)])
        keep = (cl[:, 10] <= pmax) & (gm >= a.g_min) & (ww > 0)
        iso_m1 = np.asarray(parent.isochrone["log_m1_mean"], float)
        if prow.size and prow.max() >= iso_m1.size:
            raise SystemExit(f"{art}: parent_row beyond the loaded parent ({iso_m1.size} rows); load the parent with the "
                             "luminous fragment's supplement settings")
        m1l = 10.0 ** iso_m1[prow]
        m2l = m2_point_inversion(cl[:, 17], cl[:, 0], cl[:, 10], m1l)
        with np.errstate(divide="ignore", invalid="ignore"):
            al = (cl[:, 17] / cl[:, 0]) / (cl[:, 10] / 365.25) ** (2 / 3)
            dl = 1.0 / cl[:, 0]
        cel = np.where(keep, cell_index(m2l, al, dl, m2e), -1)
        grp = np.searchsorted(D_EDGES_KPC, dl, side="right") - 1
        for gi in range(D_EDGES_KPC.size - 1):
            s_ = (cel >= 0) & (grp == gi)
            lum[gi] += np.bincount(cel[s_], weights=ww[s_], minlength=nc)
            lum_var[gi] += np.bincount(cel[s_], weights=ww[s_] ** 2, minlength=nc)

    meta = {"issue": 644, "partition": a.partition, "bins_digest": der.digest, "m2_app_edges": m2e.tolist(),
            "a_edges": A_EDGES.tolist(), "d_edges_kpc": D_EDGES_KPC.tolist(), "g_min": a.g_min, "c1_period_max_days": pmax,
            "student_t_nu": list(r4.mass_recovery.student_t_nu), "real_m2app": str(a.real_m2app),
            "co_artifacts": [str(x) for x in a.co_artifact], "lum_artifacts": [str(x) for x in a.lum_artifact],
            "lum_weights_sha256": hashlib.sha256(a.lum_weights.read_bytes()).hexdigest(),
            "pihat_sha256": hashlib.sha256(a.pihat.read_bytes()).hexdigest(), "n_prim_dark": n_prim_dark,
            "n_prim_wd": n_prim_wd, "stand_ins": stand_in, "limitations": list(rf.LIMITATIONS),
            "k_domain_total": float(k_obs.sum()), "real_comparison_keep": {k: int(v) for k, v in keep_counts.items()}}
    inp = rf.Rung5Inputs(k_obs=k_obs, high_a=high_a_mask(m2e, a.m2_min_high), lum=lum, lum_var=lum_var, spur=spur,
                         spur_var=spur_var, wd=wd, wd_var=wd_var, wd_rate=wd_rate, dark=dark, dark_var=dark_var,
                         wd_edges=wd_e, dark_edges=dark_e, pihat_sum=pihat_sum, ln_r_center=a.ln_r_center, meta=meta)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rf.write_inputs(a.out, inp)
    print(json.dumps({"k_total": meta["k_domain_total"], "n_cells": nc, "pihat_sum": pihat_sum,
                      "stand_in": stand_in}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
