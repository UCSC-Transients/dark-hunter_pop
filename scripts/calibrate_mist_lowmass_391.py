#!/usr/bin/env python3
"""Calibrate the spec §12.24 MIST low-mass correction (Teff at fixed L + mass offset) and validate it.

Δlog Teff(M) drives the prior-weighted MIST MS colour onto the measured single-star ridge;
δ(M) drives isochrone M1 / Mann (2019) to 0 on the RUWE < 1.4 parent dwarfs with 2MASS Ks.
Validation: ridge residual, Mann, Hwang (singles near the ridge), posterior [Fe/H] of field
dwarfs, DEBCat ≥ 0.8 M⊙ unchanged. Writes the calibrated node values (YAML) and a report.
Analysis only: nothing here changes generation-time quantities.
"""
from __future__ import annotations

import argparse, csv, hashlib, sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--checks", type=Path, default=Path("config/population/mass_checks_391.yaml"))
    ap.add_argument("--out-yaml", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--hwang-dereddened", type=Path, default=None, help="optional npz with dereddened Hwang c0/mg0/mass/single mask")
    args = ap.parse_args(argv)
    from astropy.table import Table
    from darkhunter_pop import binary_cmd_evidence as bce, isochrone_mass as im, mist_correction as mcorr, proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.config_schema import LowmassCorrectionConfig
    from darkhunter_pop.giants import classify_evolved, classify_parent, load_giants_config, parent_row_cmd

    D = args.data_root
    mc = yaml.safe_load(args.checks.read_text())
    chk = lambda p, s: hashlib.sha256(Path(p).read_bytes()).hexdigest() == s or sys.exit(f"SHA256 mismatch {p}")  # noqa: E731
    cfg = load_config()
    icfg = cfg.isochrone_mass
    gc = load_giants_config(Path("config/population/giants.yaml"))
    prop = ps.load_proposal_set_fragment(Path("config/population/proposal_set_restart_gen26.yaml")).proposal
    parent = ps.load_parent_snapshot(D / "dr3/gaia_snapshots/20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj", cfg, prop)
    pc, ridge = classify_parent(parent, cfg, gc, cmd=parent_row_cmd(parent, cfg, gc))
    native = im.load_native_grid(icfg, cfg.paths.data_root)
    bcs = mcorr.load_bc_tables(icfg)
    bands = (icfg.grid.band_g, icfg.grid.band_bp, icfg.grid.band_rp)

    def sigmas(cm):
        with np.errstate(divide="ignore", invalid="ignore"):
            ka = np.where(cm["ebv"] > 0, cm["a_g"] / cm["ebv"], 0.0)
            ke = np.where(cm["ebv"] > 0, cm["e_bp_rp"] / cm["ebv"], 0.0)
        return im.likelihood_sigmas(cm["sigma_mu"], cm["ebv"], ka, ke, icfg.likelihood)

    # Mann sample
    chk(D / mc["mann2019"]["table"], mc["mann2019"]["sha256"]); chk(D / mc["tmass_parent"]["table"], mc["tmass_parent"]["sha256"])
    mt = yaml.safe_load((D / mc["mann2019"]["table"]).read_text()); a = np.array(mt["fits"][mc["mann2019"]["fit"]]["a"])
    rows = list(csv.DictReader(open(D / mc["tmass_parent"]["table"])))
    sid = np.array([int(r["source_id"]) for r in rows]); ks = np.array([float(r["ks_m"] or "nan") for r in rows])
    q = np.array([r["ph_qual"][2:3] for r in rows]); ad = np.array([float(r["angular_distance"]) for r in rows])
    psid = np.asarray(parent.columns["source_id"], np.int64); o = np.argsort(psid)
    pos = np.clip(np.searchsorted(psid, sid, sorter=o), 0, psid.size - 1); r = o[pos]
    mks = ks + 5 * np.log10(np.asarray(parent.columns["parallax"], float)[r] / 100)
    mmann = 10 ** np.polyval(a[::-1], mks - mt["zero_point_mag"])
    lo, hi = mc["tmass_parent"]["m_ks_range"]
    okm = (psid[r] == sid) & np.isin(q, mc["tmass_parent"]["ks_quality"]) & (ad < mc["tmass_parent"]["max_match_arcsec"]) & (mks > lo) & (mks < hi)
    okm &= (pc.classified & ~pc.evolved)[r] & (np.asarray(parent.columns["ruwe"], float)[r] < 1.4)
    rm = r[okm]; mmann = mmann[okm]
    cmM = {k: np.asarray(parent.cmd[k], float)[rm] for k in ("colour0", "mg0", "sigma_mu", "ebv", "a_g", "e_bp_rp")}
    # field dwarfs for posterior [Fe/H]
    rng = np.random.default_rng(391)
    fd = parent.usable & pc.classified & ~pc.evolved & (np.asarray(parent.columns["ruwe"], float) < 1.4) & (np.asarray(parent.isochrone["m1_mean"], float) < 0.8)
    rf = rng.choice(np.flatnonzero(fd), size=min(4000, fd.sum()), replace=False)
    cmF = {k: np.asarray(parent.cmd[k], float)[rf] for k in cmM}

    nodes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); join = 0.9
    dlt = np.zeros(nodes.size); dlm = np.zeros(nodes.size)
    mg_edges = np.arange(3.0, 16.01, 0.25)
    bct0 = bcs[min(bcs, key=abs)]

    def build(dlt, dlm):
        lc = LowmassCorrectionConfig(enabled=True, mass_nodes_msun=tuple(nodes), dlog_teff=tuple(dlt), dlog_mass=tuple(dlm), join_msun=join)
        g = mcorr.correct_grid(native, lc, bcs, bands)
        pts = im.prior_points(g, icfg)
        return lc, pts, im.IsochroneMassModel(cfg=icfg, cmap=im.build_cmd_map(pts, icfg.cmd_map), grid_key="corr")

    def ridge_resid(pts):
        return bce.ridge_colour_shift(pts, ridge.colour, ridge.mag, join_hi_msun=join, mg_edges=mg_edges)

    def node_mg(pts):
        m = np.asarray(pts.values["star_mass"], float); ms = np.asarray(pts.values["phase"], float) < 0.5
        out = []
        for mk in nodes:
            s = ms & (np.abs(m - mk) < 0.03)
            w, x = pts.weight[s], np.asarray(pts.mg, float)[s]; oo = np.argsort(x)
            out.append(float(np.interp(0.5, np.cumsum(w[oo]) / w[oo].sum(), x[oo])) if s.sum() > 5 else np.nan)
        return np.array(out)

    def node_teff(pts):
        m = np.asarray(pts.values["star_mass"], float); ms = np.asarray(pts.values["phase"], float) < 0.5
        return np.array([np.median(10 ** np.asarray(pts.values["log_teff"], float)[ms & (np.abs(m - mk) < 0.03)]) for mk in nodes])

    def colour_slope(teff):  # d(BP-RP)/d log Teff at log g 4.8 from the BC table
        lt = np.log10(teff); e = 0.005
        def col(t):
            b = bct0.interpolate(10 ** t, np.full(np.size(t), 4.8)); return b[bands[2]] - b[bands[1]]
        return (col(lt + e) - col(lt - e)) / (2 * e)

    def mann_ratio(model):
        sc, sm = sigmas(cmM)
        post = model.fit(cmM["colour0"], cmM["mg0"], sigma_mu=cmM["sigma_mu"], ebv=cmM["ebv"], a_g=cmM["a_g"], e_bp_rp=cmM["e_bp_rp"])
        return np.log10(post.m1_mean / mmann)

    rep = ["#391 spec §12.24 MIST low-mass correction: calibration and validation", ""]
    lc, pts, model = build(dlt, dlm)
    smg, dc0 = ridge_resid(pts)
    lr0 = mann_ratio(model)
    for it in range(args.iterations):  # Teff on the ridge
        smg, dc = ridge_resid(pts)
        dck = np.interp(node_mg(pts), smg, dc)
        dlt = dlt + np.nan_to_num(dck / colour_slope(node_teff(pts)))
        lc, pts, model = build(dlt, dlm)
        rep.append(f"Teff iteration {it + 1}: node Δc before update " + " ".join(f"{x:+.3f}" for x in dck))
    for it in range(args.iterations):  # mass offset on Mann
        lr = mann_ratio(model)
        upd = np.array([np.nanmedian(lr[np.abs(mmann - mk) < 0.05]) if np.sum(np.abs(mmann - mk) < 0.05) >= 10 else 0.0 for mk in nodes])
        dlm = dlm - upd
        lc, pts, model = build(dlt, dlm)
        rep.append(f"mass iteration {it + 1}: node median log(M_iso/M_Mann) before update " + " ".join(f"{x:+.3f}" for x in upd))
    smg, dc = ridge_resid(pts)
    lr = mann_ratio(model)
    bins = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.8))
    rep += ["", "calibrated nodes (M): " + " ".join(f"{x:.1f}" for x in nodes) + f", join {join}",
            "  dlog_teff: " + " ".join(f"{x:+.4f}" for x in dlt), "  dlog_mass: " + " ".join(f"{x:+.4f}" for x in dlm), "",
            "Validation 1, ridge colour residual c_ridge - c_model at M_G 5..11 (before -> after): " +
            " ".join(f"{a_:.1f}:{b_:+.3f}->{c_:+.3f}" for a_, b_, c_ in zip(smg, dc0, dc) if 5 <= a_ <= 11.2 and (b_ != 0 or c_ != 0)),
            "Validation 2, Mann median log(M_iso/M_Mann) by Mann mass (before -> after): " +
            "; ".join(f"[{a_},{b_}) {np.nanmedian(lr0[(mmann >= a_) & (mmann < b_)]):+.3f} -> {np.nanmedian(lr[(mmann >= a_) & (mmann < b_)]):+.3f}" for a_, b_ in bins)]
    base = im.build_model(icfg, cfg.paths.data_root)
    for lab, mod in (("MIST", base), ("corrected", model)):
        p = mod.fit(cmF["colour0"], cmF["mg0"], sigma_mu=cmF["sigma_mu"], ebv=cmF["ebv"], a_g=cmF["a_g"], e_bp_rp=cmF["e_bp_rp"])
        rep.append(f"Validation 4, field dwarfs M<0.8 (n {rf.size}), {lab}: posterior <[Fe/H]> median {np.nanmedian(p.feh_mean):+.3f} (prior mean {icfg.provisional_feh_prior.mean_dex:+.2f}); median M1 {np.nanmedian(p.m1_mean):.3f}")
    # DEBCat >= 0.8 unchanged (blended system fit, dynamical primary mass)
    import validate_isochrone_m1_benchmarks_418 as vb
    cat = vb.read_debcat(D / "external_catalogs/debcat_20261005T025234Z/debs.dat")
    t = Table.read(D / "external_catalogs/xmatch_gaiadr3_418_20261006T022820Z/debcat_gaiadr3.ecsv", format="ascii.ecsv")
    t = t[np.asarray(t["source_id"], np.int64) > 0]
    _, f1 = np.unique(np.asarray(t["cat_id"]).astype(str), return_index=True); t = t[np.sort(f1)]
    _, cmd, _, _ = vb.fit(t, cfg)
    ids = [str(x) for x in t["cat_id"]]
    mp = np.array([max(cat.get(i, {}).get("m1", np.nan), cat.get(i, {}).get("m2", np.nan)) for i in ids])
    for lab, mod in (("MIST", base), ("corrected", model)):
        p = mod.fit(cmd.colour0, cmd.mg0, sigma_mu=cmd.sigma_mu, ebv=cmd.ebv, a_g=cmd.a_g, e_bp_rp=np.zeros(cmd.ebv.size))
        rr = p.m1_mean / mp
        rep.append(f"Validation 5, DEBCat blended M1/M1_dyn, {lab}: " + "; ".join(
            f"M1_dyn [{a_},{b_}) n {np.sum(np.isfinite(rr) & (mp >= a_) & (mp < b_))} median {np.nanmedian(rr[(mp >= a_) & (mp < b_)]):.3f}" for a_, b_ in ((0.0, 0.8), (0.8, 1.3), (1.3, 3.0))))
    if args.hwang_dereddened is not None:
        z = np.load(args.hwang_dereddened)
        for lab, mod in (("MIST", base), ("corrected", model)):
            p = mod.fit(z["c0"], z["mg0"], sigma_mu=np.zeros(z["c0"].size)); rr = np.log10(p.m1_mean / z["mass"])
            rep.append(f"Validation 3, Hwang (dereddened singles), {lab}: " + "; ".join(
                f"M_Hwang [{a_},{b_}) {np.nanmedian(rr[(z['mass'] >= a_) & (z['mass'] < b_)]):+.3f}" for a_, b_ in ((0.5, 0.65), (0.65, 0.8), (0.8, 1.0), (1.0, 1.3))))
    args.out_yaml.write_text(yaml.safe_dump({"lowmass_correction": {"enabled": False, "mass_nodes_msun": [float(x) for x in nodes],
                                             "dlog_teff": [round(float(x), 5) for x in dlt], "dlog_mass": [round(float(x), 5) for x in dlm],
                                             "join_msun": join, "calibration_ref": "scripts/calibrate_mist_lowmass_391.py; spec §12.24"}}, sort_keys=False))
    args.out_report.write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
