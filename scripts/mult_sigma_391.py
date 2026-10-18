#!/usr/bin/env python
"""#391 (Ryan 2026-10-18) tests 1 and 3.

1. The 1.5–3 M⊙ G < 10 excess: MdS17 companions per star (q > 0.1) in the astrometric window vs M1 (the
   Poisson-intensity target counts every companion as an independent detectable binary); the mock-detected
   weight split by flux ratio; finer M̂1 bins (mock vs real, isochrone M̂1 on both); GSP-Phot / MSC Teff of
   real vs parent 1.5–3 M⊙ G < 10 rows. (v sin i / vbroad is not in the local snapshots.)
3. σ_ϖ: recovered (gaiamock) vs published (DR3) for #390 re-injected orbits away from the ϖ/σ_ϖ cut, by log P;
   the share explained by the goodness-of-fit error inflation (gaiamock scales its covariance by
   cc = sqrt(χ²_red / (1 − 2/(9ν))³), i.e. by F2); and the published ϖ–orbit correlations from corr_vec.
Writes ``mult_sigma.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def chi2red_from_f2(f2, nu):
    """Inverse Wilson–Hilferty: F2 = sqrt(9ν/2)[(χ²/ν)^(1/3) + 2/(9ν) − 1]."""
    return (1.0 + np.asarray(f2, float) / np.sqrt(9 * nu / 2) - 2 / (9 * nu)) ** 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True, help="in cache order")
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_coarse32_v6a.yaml"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--inject390", type=Path, required=True, help="gate400 validation_u0/inject.jsonl (variant v2_u0)")
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    import h5py
    import yaml
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import giants
    from darkhunter_pop import isochrone_mass as im
    from darkhunter_pop import moe_distefano as mds
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.nss_covariance import reconstruct_nss_covariance

    cfg = load_config(Path("config/config.yaml"))
    frag = ps.load_proposal_set_fragment(args.fragment)
    tgt = frag.target_mds17
    table = mds.load_mds17_table(tgt.table_path)
    out: dict = {}

    # ---- 1a: MdS17 companions per star (q > 0.1) by window ----
    qg = np.linspace(0.1, 1.0, 901)
    m1g = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    lam = {}
    for lo, hi, nm in ((0.2, 8.0, "all P"), (1.0, 4.0, "log P 1-4"), (2.0, 3.0, "log P 2-3")):
        lpg = np.linspace(lo, hi, 241)
        row = []
        for m in m1g:
            fq = np.array([np.trapz(mds.q_density(qg, m, lp, table, m1_interpolation=tgt.provisional_m1_interpolation), qg) for lp in lpg])
            row.append(float(np.trapz(mds.f_logp_q03(np.full(lpg.size, m), lpg, table) * fq, lpg)))
        lam[nm] = row
    out["mds17_companions_per_star"] = {"m1": m1g, **lam,
                                        "P_other_in_logP1-4": [float(1 - np.exp(-x)) for x in lam["log P 1-4"]]}

    # ---- 1b: mock-detected weight at G < 10 by flux ratio and finer M̂1 ----
    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    zc = np.load(args.cache, allow_pickle=True)
    casc = np.asarray(zc["cascade"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_o, d_a, lp_o = 1 / casc[:, 0], 1 / casc[:, 2], np.log10(casc[:, 10])
    o, g = fc["outer"], np.asarray(zc["phot_g_mean_mag"], float)
    PM = float(fc["c1_period_max_days"])

    def inb(x, e):
        return (x >= e[0]) & (x < e[1])

    in_o = np.asarray(zc["accepted"], bool) & (casc[:, 10] <= PM) & inb(g, o["g_mag"]) & inb(d_o, o["distance_kpc"]) & \
        inb(lp_o, o["log10_period_days"]) & inb(casc[:, 14], o["eccentricity"])
    in_a = np.asarray(zc["published_acceleration"], bool) & inb(g, o["g_mag"]) & inb(d_a, o["distance_kpc"])
    rel = np.flatnonzero(in_o | in_a)
    lf_all = np.concatenate([h5py.File(a, "r")["truth/log10_flux_ratio"][()] for a in args.artifacts])
    dark_all = np.concatenate([h5py.File(a, "r")["truth/is_dark"][()] for a in args.artifacts]).astype(bool)
    m2_all = np.asarray(zc["m2_msun"], float)
    z = np.load(args.dump, allow_pickle=True)
    if rel.size != z["w"].size:
        raise SystemExit(f"rel size {rel.size} != dump {z['w'].size}")
    lf, dk, m2 = lf_all[rel], dark_all[rel], m2_all[rel]
    cfgp = cfg
    parent = ps.load_parent_snapshot(args.parent_dir, cfgp, frag.proposal)
    m1r = parent.m1_msun[z["prow"]]
    w, so, sa = z["w"], z["ob"] >= 0, z["ab"] >= 0
    fe = [-np.inf, -3.0, -2.0, -1.0, -0.5, 1.0]
    split = {}
    for lab, a_, b_ in (("1.0-1.2", 1.0, 1.2), ("1.5-2", 1.5, 2.0), ("2-3", 2.0, 3.0)):
        for kind, s in (("orbit", so & (z["m_g"] < 10)), ("accel", sa & (z["m_g_a"] < 10))):
            sel = s & (m1r >= a_) & (m1r < b_)
            tot = w[sel].sum()
            hf = np.histogram(np.where(dk[sel], -9.0, lf[sel]), bins=[-10.0] + fe[1:], weights=w[sel])[0] / tot
            q = m2[sel] / z["m1_law"][sel]
            hq = np.histogram(q, bins=[0, 0.2, 0.4, 0.6, 0.8, 0.95, 2.0], weights=w[sel])[0] / tot
            split[f"{kind} M1 {lab}"] = {"flux_ratio_share_dark_<-3_-3..-2_-2..-1_-1..-0.5_>-0.5": hf.tolist(),
                                          "q_share_0-.2_.2-.4_.4-.6_.6-.8_.8-.95_>.95": hq.tolist(), "sum_w": float(tot)}
    out["mock_G<10_split"] = split

    # finer M̂1 bins at G < 10: mock (parent-row M̂1) vs real (isochrone M̂1, fit's real-side convention)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    gcfg = giants.load_giants_config(Path("config/population/giants.yaml"))
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)

    def col(t, n):
        return np.ma.filled(np.ma.asarray(t[n], float), np.nan)

    def m1_real(t):
        gg = col(t, "g_mag")
        bprp = col(t, "bp_mag") - col(t, "rp_mag")
        plx = col(t, "parallax") + zp[np.clip(np.digitize(gg, ge) - 1, 0, 4)]
        e = col(t, "parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rr, rlo, rhi = (np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + e), np.nan),
                            np.where(plx - e > 0, 1000 / (plx - e), np.nan))
        gal = SkyCoord(ra=col(t, "ra") * u.deg, dec=col(t, "dec") * u.deg).galactic
        c = giants.cmd_for_rows(gg, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
        dd = model.fit(c.colour0, c.mg0, sigma_mu=c.sigma_mu, ebv=c.ebv, a_g=c.a_g, e_bp_rp=bprp - c.colour0).as_dict()
        return np.where(np.asarray(dd["ok"], bool), np.asarray(dd["m1_mean"], float), np.nan)

    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    pos = {int(s): i for i, s in enumerate(np.asarray(rt["source_id"], np.int64))}
    rin = (z["rob"] >= 0) & (z["r_g"] < 10)
    rsub = rt[np.array([pos[int(s)] for s in z["r_sid"][rin]])]
    m1_ro = m1_real(rsub)
    teff_ro = col(rsub, "teff_gspphot")
    teffm_ro = col(rsub, "teff_msc1")
    del rt
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    apos = {int(s): i for i, s in enumerate(np.asarray(at["source_id"], np.int64))}
    asel = z["ra_g"] < 10
    m1_ra = m1_real(at[np.array([apos[int(s)] for s in z["ra_sid"][asel]])])
    del at
    me = [0.0, 0.8, 1.0, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0]
    fine = {"edges": me}
    for kind, s_m, mr in (("orbit", so & (z["m_g"] < 10), m1_ro), ("accel", sa & (z["m_g_a"] < 10), m1_ra)):
        mk = np.histogram(m1r[s_m], bins=me, weights=w[s_m])[0]
        ess = [float(w[s_m & (m1r >= a_) & (m1r < b_)].sum() ** 2 / max(np.sum(w[s_m & (m1r >= a_) & (m1r < b_)] ** 2), 1e-300))
               for a_, b_ in zip(me[:-1], me[1:])]
        rk = np.histogram(mr[np.isfinite(mr)], bins=me)[0]
        fine[kind] = {"mock": mk.tolist(), "mock_ess": ess, "real": rk.tolist(),
                      "ratio": (mk / np.where(rk > 0, rk, np.nan)).tolist()}
    out["fine_m1_G<10"] = fine

    # Teff of 1.5-3 M⊙ G < 10 rows: real orbit hosts vs parent (GSP-Phot and MSC)
    pm = (pg < 10) & parent.usable & (parent.m1_msun >= 1.5) & (parent.m1_msun < 3.0)
    rm = np.isfinite(m1_ro) & (m1_ro >= 1.5) & (m1_ro < 3.0)

    def pct(x):
        x = x[np.isfinite(x)]
        return [float(v) for v in np.percentile(x, [10, 50, 90])] if x.size else []

    out["teff_1.5-3_G<10"] = {"real_orbit_gspphot": pct(teff_ro[rm]), "parent_gspphot": pct(np.asarray(pc["teff_gspphot"], float)[pm]),
                              "real_orbit_msc": pct(teffm_ro[rm]), "parent_msc": pct(np.asarray(pc["teff_msc1"], float)[pm]),
                              "real_frac_teff_gt_7500": float(np.mean(teff_ro[rm][np.isfinite(teff_ro[rm])] > 7500)),
                              "parent_frac_teff_gt_7500": float(np.mean(np.asarray(pc["teff_gspphot"], float)[pm][np.isfinite(np.asarray(pc["teff_gspphot"], float)[pm])] > 7500)),
                              "n_real": int(rm.sum()), "n_parent": int(pm.sum()), "vsini": "not in local snapshots (gaia_source vbroad needs a fetch)"}

    # ---- 3: σ_ϖ recovered vs published (#390 set, v2_u0), away from the cut ----
    with h5py.File(args.published, "r") as h:
        pub = {k: h["published"][k][()] for k in h["published"].keys()}
    pos_p = {int(s): i for i, s in enumerate(pub["source_id"])}
    recs = [json.loads(x) for x in args.inject390.read_text().splitlines()]
    recs = [r for r in recs if r.get("variant") == "v2_u0" and (r.get("parallax_error") or 0) > 0]
    rows = []
    for r in recs:
        j = pos_p.get(int(r["source_id"]))
        if j is None:
            continue
        P, pe, pp, f2p = pub["period"][j], pub["parallax_error"][j], pub["parallax"][j], pub["goodness_of_fit"][j]
        nu = max(float(r.get("n_obs") or np.nan) - 12, 20)
        rows.append((np.log10(P), pp / pe / (20000 / P), r["parallax_error"] / pe, r.get("goodness_of_fit", np.nan), f2p, nu, j,
                     pub["nss_solution_type"][j].decode()))
    arr = np.array([x[:6] for x in rows], float)
    lpb = [0.5, 2.0, 2.3, 2.5, 2.7, 3.0, 3.5]
    sig = {"log_p_edges": lpb, "rows": []}
    for a_, b_ in zip(lpb[:-1], lpb[1:]):
        for far in (True, False):
            s = (arr[:, 0] >= a_) & (arr[:, 0] < b_) & ((arr[:, 1] > 1.5) if far else (arr[:, 1] <= 1.5))
            if not s.any():
                continue
            cc = np.sqrt(chi2red_from_f2(arr[s, 3], arr[s, 5]) / chi2red_from_f2(arr[s, 4], arr[s, 5]))
            sig["rows"].append({"log_p": [a_, b_], "away_from_cut(>1.5x)": far, "n": int(s.sum()),
                                "sigma_ratio_median": float(np.median(arr[s, 2])),
                                "sigma_ratio_p16_84": np.percentile(arr[s, 2], [16, 84]).tolist(),
                                "F2_rec_median": float(np.nanmedian(arr[s, 3])), "F2_pub_median": float(np.nanmedian(arr[s, 4])),
                                "cc_ratio_from_F2_median": float(np.nanmedian(cc)),
                                "sigma_ratio_over_cc_ratio_median": float(np.nanmedian(arr[s, 2] / cc))})
    # published ϖ–orbit correlations (corr_vec) by log P, one per system
    corr_rows = {}
    seen = set()
    for x in rows:
        j = x[6]
        if j in seen:
            continue
        seen.add(j)
        row = {k: (v[j].decode() if isinstance(v[j], bytes) else v[j]) for k, v in pub.items()}
        cr = reconstruct_nss_covariance(row, nss_solution_type=x[7])
        if cr.parameter_set is None:
            continue
        nm = list(cr.parameter_set.names)
        cv = np.asarray(cr.parameter_set.covariance, float)
        sd = np.sqrt(np.diag(cv))
        R = cv / np.outer(sd, sd)
        ip = nm.index("parallax")
        others = [i for i, n in enumerate(nm) if n not in ("parallax", "ra", "dec")]
        b = int(np.clip(np.digitize(x[0], lpb) - 1, 0, len(lpb) - 2))
        corr_rows.setdefault(b, []).append(float(np.max(np.abs(R[ip, others]))))
    sig["published_max_abs_corr_parallax_vs_others_median"] = {f"{lpb[b]}-{lpb[b + 1]}": float(np.median(v)) for b, v in sorted(corr_rows.items())}
    out["sigma_parallax"] = sig
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mult_sigma.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float)[:8000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
