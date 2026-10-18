#!/usr/bin/env python
"""#391 (Ryan 2026-10-18) follow-ups 3 and 4: real-side isochrone M1 at the bright end, and the bright
isochrone-M1 failure rate, real NSS rows vs the parent.

Real rows (C1 orbits and accelerations, as kept by the fit; ``--dump``) get the dereddened CMD and the
MIST isochrone posterior with the fit's real-side conventions (``rung3_refit5_391.extend.cls_of``:
distance 1/(ϖ + zp(G)), Combined19, ``isochrone_mass`` as configured), and the parent rows at the same G
are refit with that convention too (``parallax`` rows) next to their stored BJ-distance posterior.
Mock side: the detected mock systems' parent-row M̂1 (the same posterior mean) at the best-fit weights.
Writes ``real_m1.json`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

M1_EDGES = [0.0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 100.0]
G_EDGES = [3.0, 8.0, 9.0, 10.0, 11.0, 13.0, 15.0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    ap.add_argument("--g-max", type=float, default=11.0)
    ap.add_argument("--control-n", type=int, default=4000, help="random real rows at G 13-15 for the failure-rate reference")
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
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])

    def fit_rows(g, bprp, ra, dec, plx_obs, eplx, l=None, b=None):
        plx = plx_obs + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(plx > 0, 1000 / plx, np.nan)
            rlo = np.where(plx > 0, 1000 / (plx + eplx), np.nan)
            rhi = np.where(plx - eplx > 0, 1000 / (plx - eplx), np.nan)
        if l is None:
            gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
            l, b = gal.l.deg, gal.b.deg
        c = giants.cmd_for_rows(g, bprp, l, b, rr, rlo, rhi, cfg, gcfg)
        post = model.fit(c.colour0, c.mg0, sigma_mu=c.sigma_mu, ebv=c.ebv, a_g=c.a_g, e_bp_rp=np.asarray(bprp, float) - c.colour0)
        d = post.as_dict()
        return np.asarray(d["m1_mean"], float), np.asarray(d["ok"], bool), np.isfinite(c.mg0) & np.isfinite(c.colour0)

    def col(t, n):
        return np.ma.filled(np.ma.asarray(t[n], float), np.nan)

    z = np.load(args.dump, allow_pickle=True)
    out: dict = {"m1_edges": M1_EDGES, "g_edges": G_EDGES, "convention": "real side: 1/(plx + zp(G)) distances (fit's cls_of)"}

    # ---- real C1 orbits (all kept rows at G < g_max, plus a G 13-15 control) ----
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    rsid_all = np.asarray(rt["source_id"], np.int64)
    pos = {int(s): i for i, s in enumerate(rsid_all)}
    rin = z["rob"] >= 0
    rng = np.random.default_rng(391)
    sel_b = np.flatnonzero(rin & (z["r_g"] < args.g_max))
    ctrl = np.flatnonzero(rin & (z["r_g"] >= 13) & (z["r_g"] < 15))
    ctrl = rng.choice(ctrl, size=min(args.control_n, ctrl.size), replace=False)
    idx_o = np.concatenate([sel_b, ctrl])
    rows = np.array([pos[int(s)] for s in z["r_sid"][idx_o]])
    sub = rt[rows]
    g_o = col(sub, "g_mag")
    m1_o, ok_o, cmd_o = fit_rows(g_o, col(sub, "bp_mag") - col(sub, "rp_mag"), col(sub, "ra"), col(sub, "dec"),
                                 col(sub, "parallax"), col(sub, "parallax_error"))
    cls_o = z["r_cls"][idx_o]
    del rt

    # ---- real accelerations ----
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    apos = {int(s): i for i, s in enumerate(np.asarray(at["source_id"], np.int64))}
    sel_a = np.flatnonzero(z["ra_g"] < args.g_max)
    arows = np.array([apos[int(s)] for s in z["ra_sid"][sel_a]])
    asub = at[arows]
    g_a = col(asub, "g_mag")
    m1_a, ok_a, cmd_a = fit_rows(g_a, col(asub, "bp_mag") - col(asub, "rp_mag"), col(asub, "ra"), col(asub, "dec"),
                                 col(asub, "parallax"), col(asub, "parallax_error"))
    cls_a = z["ra_cls"][sel_a]
    del at

    # ---- parent rows at G < g_max (and a 13-15 control), refit with the real-side convention ----
    pb = np.flatnonzero(pg < args.g_max)
    pctl = rng.choice(np.flatnonzero((pg >= 13) & (pg < 15)), size=args.control_n, replace=False)
    pidx = np.concatenate([pb, pctl])
    m1_pp, ok_pp, cmd_pp = fit_rows(pg[pidx], np.asarray(pc["bp_rp"], float)[pidx], None, None,
                                    np.asarray(pc["parallax"], float)[pidx], np.asarray(pc["parallax_error"], float)[pidx],
                                    l=np.asarray(pc["l"], float)[pidx], b=np.asarray(pc["b"], float)[pidx])
    ok_bj = np.asarray(parent.isochrone["ok"], bool)[pidx] & np.isfinite(parent.m1_msun[pidx])
    cmd_bj = np.isfinite(np.asarray(parent.cmd["mg0"], float)[pidx])
    base = np.ones(pidx.size, bool)
    for kf in ("finite_astrometry_photometry", "parallax_floor"):
        base &= np.asarray(parent.flags[kf], bool)[pidx]

    def rate(ok, cmd, g, mask=None):
        mask = np.ones(g.size, bool) if mask is None else mask
        r = {"n": [], "cmd_ok": [], "m1_ok": []}
        for a, b in zip(G_EDGES[:-1], G_EDGES[1:]):
            s = mask & (g >= a) & (g < b)
            r["n"].append(int(s.sum()))
            r["cmd_ok"].append(float(np.mean(cmd[s])) if s.any() else np.nan)
            r["m1_ok"].append(float(np.mean(ok[s])) if s.any() else np.nan)
        return r

    pgx = pg[pidx]
    out["failure"] = {
        "parent_bj_stored": rate(ok_bj, cmd_bj, pgx, base),
        "parent_plx_convention": rate(ok_pp, cmd_pp, pgx, base),
        "real_orbits": rate(ok_o, cmd_o, g_o),
        "real_accelerations": rate(ok_a, cmd_a, g_a),
    }

    # ---- M1 distributions at G < 10: real (ok rows) vs mock (detected, parent-row M̂1, best-fit weights) ----
    w = z["w"]
    prow = z["prow"]
    m1_row = parent.m1_msun[prow]
    so, sa = z["ob"] >= 0, z["ab"] >= 0

    def hist(x, wt):
        h = np.histogram(x[np.isfinite(x)], bins=M1_EDGES, weights=wt[np.isfinite(x)])[0]
        return (h / h.sum()).tolist() if h.sum() > 0 else []

    dist = {}
    for lab, gmin, gmax in (("G<10", 3.0, 10.0), ("G10-11", 10.0, 11.0)):
        mo = so & (z["m_g"] >= gmin) & (z["m_g"] < gmax)
        ma = sa & (z["m_g_a"] >= gmin) & (z["m_g_a"] < gmax)
        ro = (g_o >= gmin) & (g_o < gmax) & ok_o
        ra = (g_a >= gmin) & (g_a < gmax) & ok_a
        dist[lab] = {
            "mock_orbit_rowM1": hist(m1_row[mo], w[mo]), "real_orbit_M1": hist(m1_o[ro], np.ones(ro.sum())),
            "mock_accel_rowM1": hist(m1_row[ma], w[ma]), "real_accel_M1": hist(m1_a[ra], np.ones(ra.sum())),
            "mock_orbit_rowM1_dwarf": hist(m1_row[mo & (z["m_cls"] == 0)], w[mo & (z["m_cls"] == 0)]),
            "real_orbit_M1_dwarf": hist(m1_o[ro & (cls_o == 0)], np.ones((ro & (cls_o == 0)).sum())),
            "mock_orbit_sum_w_by_M1": np.histogram(m1_row[mo], bins=M1_EDGES, weights=w[mo])[0].tolist(),
            "real_orbit_count_by_M1": np.histogram(m1_o[ro], bins=M1_EDGES)[0].tolist(),
            "real_orbit_n_no_M1": int(((g_o >= gmin) & (g_o < gmax) & ~ok_o).sum()),
            "mock_accel_sum_w_by_M1": np.histogram(m1_row[ma], bins=M1_EDGES, weights=w[ma])[0].tolist(),
            "real_accel_count_by_M1": np.histogram(m1_a[ra], bins=M1_EDGES)[0].tolist(),
            "real_accel_n_no_M1": int(((g_a >= gmin) & (g_a < gmax) & ~ok_a).sum()),
            "mock_orbit_ess_by_M1": [float(w[mo & (m1_row >= a) & (m1_row < b)].sum() ** 2 / max(np.sum(w[mo & (m1_row >= a) & (m1_row < b)] ** 2), 1e-300))
                                     for a, b in zip(M1_EDGES[:-1], M1_EDGES[1:])],
        }
    out["m1"] = dist
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "real_m1.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
