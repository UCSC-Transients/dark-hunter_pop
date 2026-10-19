#!/usr/bin/env python
"""#391 (Ryan 2026-10-18): G < 10 real/mock ratio in dereddened (BP−RP)_0 (Combined19 × the pipeline's native→E(B−V)
factor via giants.cmd_for_rows; real side at 1/(ϖ + zp), as the fit), orbits and accelerations, with Poisson (real) and
parent-row (mock) errors; and the vbroad split for (BP−RP)_0 < 0.5. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

C_EDGES = [-1.0, 0.1, 0.3, 0.5, 0.7, 1.0, 5.0]
V_EDGES = [0, 20, 50, 100, 500]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("--dump", "--parent-dir", "--real-snapshot", "--accel-snapshot", "--vbroad", "--out"):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    a = ap.parse_args(argv)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import giants
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    gcfg = giants.load_giants_config(Path("config/population/giants.yaml"))
    pc = par.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = par.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[par.usable & (pg >= x) & (pg < y) & (snr > 20)]) for x, y in zip(ge[:-1], ge[1:])])
    vt = Table.read(a.vbroad / "table.ecsv")
    vb = {(str(s), int(i)): float(v) for s, i, v in zip(vt["sample"], vt["source_id"], np.ma.filled(np.ma.asarray(vt["vbroad"], float), np.nan))}

    def c0_real(t):
        f = lambda c: np.ma.filled(np.ma.asarray(t[c], float), np.nan)  # noqa: E731
        g, bprp = f("g_mag"), f("bp_mag") - f("rp_mag")
        plx = f("parallax") + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
        e = f("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rr, rlo, rhi = np.where(plx > 0, 1e3 / plx, np.nan), np.where(plx > 0, 1e3 / (plx + e), np.nan), np.where(plx - e > 0, 1e3 / (plx - e), np.nan)
        gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
        return giants.cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg).colour0

    z = np.load(a.dump, allow_pickle=True)
    rt = Table.read(a.real_snapshot, format="ascii.ecsv")
    pos = {int(s): i for i, s in enumerate(np.asarray(rt["source_id"], np.int64))}
    rin = z["rob"] >= 0
    ro_sid = z["r_sid"][rin & (z["r_g"] < 10)]
    ro_c = c0_real(rt[np.array([pos[int(s)] for s in ro_sid])])
    del rt
    at = Table.read(a.accel_snapshot / "table.h5", path="data")
    apos = {int(s): i for i, s in enumerate(np.asarray(at["source_id"], np.int64))}
    ra_sid = z["ra_sid"][z["ra_g"] < 10]
    ra_c = c0_real(at[np.array([apos[int(s)] for s in ra_sid])])
    del at
    pcol = np.asarray(par.cmd["colour0"], float)
    psid = np.asarray(pc["source_id"], np.int64)
    pvb = np.asarray(pc["vbroad"], float) if "vbroad" in pc else np.full(psid.size, np.nan)
    out = {"colour_edges": C_EDGES, "vbroad_edges": V_EDGES}

    def mock_hist(sel, x, edges):
        rows, w = z["prow"][sel], z["w"][sel]
        h, err = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            u_, inv = np.unique(rows[m], return_inverse=True)
            rw = np.bincount(inv, weights=w[m]) if m.any() else np.zeros(0)
            h.append(float(rw.sum()))
            err.append(float(np.sqrt(np.sum(rw**2))))
        return np.array(h), np.array(err)

    for kind, sm, gm, rc, rsid in (("orbit", z["ob"] >= 0, z["m_g"], ro_c, ro_sid), ("accel", z["ab"] >= 0, z["m_g_a"], ra_c, ra_sid)):
        sel = sm & (gm < 10)
        mc = pcol[z["prow"][sel]]
        hm, em = mock_hist(sel, mc, C_EDGES)
        hr = np.histogram(rc[np.isfinite(rc)], bins=C_EDGES)[0].astype(float)
        ratio = hr / np.where(hm > 0, hm, np.nan)
        rel = np.sqrt(1 / np.maximum(hr, 1) + (em / np.where(hm > 0, hm, np.nan)) ** 2)
        out[f"{kind}_colour"] = {"real": hr.tolist(), "mock": hm.tolist(), "mock_row_err": em.tolist(),
                                 "real_over_mock": ratio.tolist(), "ratio_err": (ratio * rel).tolist()}
        # vbroad split, (BP-RP)_0 < 0.5
        mv = np.array([pvb[r] if np.isfinite(pvb[r]) else vb.get(("parent", int(psid[r])), np.nan) for r in z["prow"][sel]])
        rv = np.array([vb.get(("real_" + kind, int(s)), np.nan) for s in rsid])
        blue_m = sel.copy()
        blue_m[sel] = (mc < 0.5) & np.isfinite(mv)
        mv_full = np.full(sel.size, np.nan)
        mv_full[sel] = mv
        hmv, emv = mock_hist(blue_m, mv_full[blue_m], V_EDGES)
        hrv = np.histogram(rv[(rc < 0.5) & np.isfinite(rv)], bins=V_EDGES)[0].astype(float)
        out[f"{kind}_vbroad_blue"] = {"real": hrv.tolist(), "mock": hmv.tolist(), "mock_row_err": emv.tolist(),
                                      "real_over_mock": (hrv / np.where(hmv > 0, hmv, np.nan)).tolist(),
                                      "n_real_blue": int((rc < 0.5).sum()), "n_real_blue_with_vbroad": int(((rc < 0.5) & np.isfinite(rv)).sum())}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
