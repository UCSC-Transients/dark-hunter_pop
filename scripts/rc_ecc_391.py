#!/usr/bin/env python3
"""Eccentricities of red-clump primaries vs a matched main-sequence control in real Gaia DR3 NSS orbits (#391, Ryan 2026-10-15).

Sample: Orbital + AstroSpectroSB1 orbits passing the rung-3 C1 cut (P <= c1_period_max_days, the El-Badry
et al. 2024 comparison filters), dereddened CMD (Combined19 x 0.884 at zero-point-corrected 1/plx, the same
convention as the rung-3 real side). Red clump = CMD-evolved and inside the clump box. Control = CMD dwarfs
matched (k-NN with replacement) on isochrone M_init, distance, [Fe/H] (GSP-Phot, else MSC; missing values
imputed and flagged) and log P. Quality: a0/sigma_a0 from the gate390 published orbits (cut configurable).
Outputs a PNG + PDF and a JSON/text summary. Analysis only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_compliant108_v6a.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/gate391/rc_ecc"))
    ap.add_argument("--sig-min", type=float, default=20.0, help="a0/sigma_a0 quality cut")
    ap.add_argument("--k", type=int, default=5, help="MS neighbours per RC star (with replacement)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=391)
    args = ap.parse_args(argv)
    import h5py
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from scipy.spatial import cKDTree
    from scipy.stats import anderson_ksamp, ks_2samp

    from darkhunter_pop import isochrone_mass as im, plotting, proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.giants import classify_evolved, classify_parent, cmd_for_rows, load_giants_config, parent_row_cmd

    D = args.data_root
    cfg = load_config()
    gc = load_giants_config(Path("config/population/giants.yaml"))
    prop = ps.load_proposal_set_fragment(Path("config/population/proposal_set_restart_gen26.yaml")).proposal
    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    parent = ps.load_parent_snapshot(D / "dr3/gaia_snapshots/20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj", cfg, prop)
    pc, ridge = classify_parent(parent, cfg, gc, cmd=parent_row_cmd(parent, cfg, gc))
    pcol = parent.columns
    pg = np.asarray(pcol["phot_g_mean_mag"], float)
    snr = np.asarray(pcol["parallax"], float) / np.asarray(pcol["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pcol["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    del parent

    rt = Table.read(D / "dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv", format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(D / "dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns"))
    rt = rt[keep]
    col = lambda n: np.ma.filled(np.ma.asarray(rt[n], float), np.nan)  # noqa: E731
    P, e = col("period"), col("eccentricity")
    rt = rt[P <= float(fc["c1_period_max_days"])]
    P, e = col("period"), col("eccentricity")
    g, bprp = col("g_mag"), col("bp_mag") - col("rp_mag")
    plx = col("parallax") + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
    ep = col("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(plx > 0, 1000 / plx, np.nan)
        rlo = np.where(plx > 0, 1000 / (plx + ep), np.nan)
        rhi = np.where(plx - ep > 0, 1000 / (plx - ep), np.nan)
    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    cmd = cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gc)
    ev = classify_evolved(cmd.mg0, cmd.colour0, cmd.sigma_mu, ridge, gc.provisional_n_sigma)
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    post = model.fit(cmd.colour0, cmd.mg0, sigma_mu=cmd.sigma_mu, ebv=cmd.ebv, a_g=cmd.a_g, e_bp_rp=bprp - cmd.colour0)
    m_init, m_now = np.asarray(post.m_init_mean, float), np.asarray(post.m1_mean, float)
    feh = col("mh_gspphot")
    feh_src = np.where(np.isfinite(feh), "gspphot", "none")
    msc = col("mh_msc")
    feh = np.where(np.isfinite(feh), feh, msc)
    feh_src = np.where((feh_src == "none") & np.isfinite(msc), "msc", feh_src)
    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        sp, sg = h["published/source_id"][()], h["published/significance"][()]
    sid = np.asarray(rt["source_id"], np.int64)
    o = np.argsort(sp)
    pos = np.clip(np.searchsorted(sp, sid, sorter=o), 0, sp.size - 1)
    sig = np.where(sp[o[pos]] == sid, sg[o[pos]], np.nan)

    box = (cmd.colour0 >= 0.95) & (cmd.colour0 < 1.35) & (cmd.mg0 >= -0.2) & (cmd.mg0 < 1.2)
    rc_all = ev.classified & ev.evolved & box & np.isfinite(m_init) & np.isfinite(e)
    ms_all = ev.classified & ~ev.evolved & np.isfinite(m_init) & np.isfinite(e) & np.isfinite(rr)
    lp = np.log10(P)

    # APOKASC-3 purity of the box (seismic labels where they overlap)
    apok = {}
    try:
        xm = Table.read(D / "external_catalogs/xmatch_gaiadr3_418_20261006T022820Z/apokasc3_gaiadr3.ecsv", format="ascii.ecsv")
        import gzip
        lab = {}
        with gzip.open(D / "external_catalogs/apokasc3_pinsonneault2025_20261005T025257Z/table4.dat.gz", "rt") as fh:
            for ln in fh:
                lab[ln[0:8].strip()] = ln[9:16].strip()
        m = {int(s): lab.get(str(c).replace("KIC", "").strip(), "") for s, c in zip(np.asarray(xm["source_id"], np.int64), xm["cat_id"])}
        ls = np.array([m.get(int(s), "") for s in sid])
        for name, sel in (("clump box (evolved)", rc_all), ("all evolved", ev.classified & ev.evolved)):
            have = sel & (ls != "")
            apok[name] = {"n_overlap": int(have.sum()), **{k: int(np.sum(have & (ls == k))) for k in ("RC", "RGB", "RC/RGB", "Dwarf", "Unknown")}}
    except Exception as exc:  # pragma: no cover
        apok = {"error": str(exc)}

    rng = np.random.default_rng(args.seed)

    def match(rc, ms):
        """k-NN with replacement in standardized (log M_init, log d, [Fe/H], log P) + a missing-[Fe/H] flag."""
        def feats(s):
            f = feh[s]
            miss = ~np.isfinite(f)
            fi = np.where(miss, np.nanmedian(feh[ms | rc]), f)
            return np.column_stack([np.log10(m_init[s]), np.log10(rr[s]), fi, lp[s], 2.0 * miss])
        A, B = feats(rc), feats(ms)
        sc = np.nanstd(np.vstack([A, B]), axis=0)
        sc[-1] = 1.0
        tree = cKDTree(B / sc)
        _, idx = tree.query(A / sc, k=args.k)
        return np.flatnonzero(ms)[np.ravel(idx)]

    def stats(a, b):
        f = lambda x: np.mean(x < 0.1)  # noqa: E731
        ba = [f(rng.choice(a, a.size)) for _ in range(args.n_boot)]
        bb = [f(rng.choice(b, b.size)) for _ in range(args.n_boot)]
        ks = ks_2samp(a, b).pvalue if min(a.size, b.size) > 3 else np.nan
        try:
            ad = anderson_ksamp([a, b]).significance_level if min(a.size, b.size) > 3 else np.nan
        except Exception:
            ad = np.nan
        return {"n_rc": int(a.size), "n_ms": int(b.size), "f_e_lt_0p1_rc": [float(f(a)), float(np.std(ba))],
                "f_e_lt_0p1_ms": [float(f(b)), float(np.std(bb))], "ks_p": float(ks), "ad_p_capped": float(ad),
                "median_e_rc": float(np.median(a)), "median_e_ms": float(np.median(b))}

    pbins = ((0.0, 2.3), (2.3, 2.6), (2.6, 2.92))
    out = {"sig_min": args.sig_min, "k": args.k, "apokasc_box_purity": apok, "feh_coverage": {}}
    results = {}
    for qlab, qsel in (("sig", np.isfinite(sig) & (sig >= args.sig_min)), ("all", np.ones(sig.size, bool))):
        rc, ms = rc_all & qsel, ms_all & qsel
        mi = match(rc, ms)
        res = {"overall": stats(e[rc], e[mi])}
        for a, b in pbins:
            r = rc & (lp >= a) & (lp < b)
            mb = match(r, ms & (lp >= a) & (lp < b)) if r.sum() > 0 else np.array([], int)
            res[f"logP_{a}_{b}"] = stats(e[r], e[mb]) if r.sum() > 3 and mb.size > 3 else {"n_rc": int(r.sum())}
        results[qlab] = (rc, mi, res)
        out[qlab] = res
        out["feh_coverage"][qlab] = {"rc": {s: int(np.sum(rc & (feh_src == s))) for s in ("gspphot", "msc", "none")},
                                     "ms_matched": {s: int(np.sum(feh_src[mi] == s)) for s in ("gspphot", "msc", "none")}}
    rc, mi, res = results["sig"]
    out["m_init_over_m_now_rc_median"] = float(np.nanmedian(m_init[rc] / m_now[rc]))
    out["matched_marginals_median"] = {k: [float(np.nanmedian(v[rc])), float(np.nanmedian(v[mi]))] for k, v in
                                       (("M_init", m_init), ("d_pc", rr), ("feh", feh), ("logP", lp))}

    # figure
    plt = plotting.require_pyplot()
    style = None  # schema defaults (docs/PLOTS.md: Okabe-Ito, serif, inward ticks)
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 0.8])
    c_rc, c_ms = plotting.series_style(1, style)["color"], plotting.series_style(2, style)["color"]
    def cdf(ax, x, lbl, c, ls="-"):
        s = np.sort(x)
        ax.step(s, np.arange(1, s.size + 1) / s.size, where="post", color=c, ls=ls, lw=2, label=lbl)
    axA = fig.add_subplot(gs[0, 0:2])
    st = res["overall"]
    cdf(axA, e[rc], f"red clump (n={st['n_rc']}; e<0.1: {st['f_e_lt_0p1_rc'][0]:.2f}±{st['f_e_lt_0p1_rc'][1]:.2f})", c_rc)
    cdf(axA, e[mi], f"matched MS (n={st['n_ms']}; e<0.1: {st['f_e_lt_0p1_ms'][0]:.2f}±{st['f_e_lt_0p1_ms'][1]:.2f})", c_ms)
    ra, ma, _ = results["all"]
    cdf(axA, e[ra], "red clump, no a0/σ cut", c_rc, ":")
    cdf(axA, e[ma], "matched MS, no a0/σ cut", c_ms, ":")
    plotting.apply_axes_style(axA, style, xlabel="eccentricity", ylabel="cumulative fraction", title="(A) all orbits")
    axA.text(0.98, 0.45, f"P ≤ {fc['c1_period_max_days']} d\nKS p = {st['ks_p']:.1e}", transform=axA.transAxes, ha="right", fontsize=9)
    axA.legend(frameon=False, fontsize=7, loc="lower right")
    axC = fig.add_subplot(gs[0, 2:4])
    axC.scatter(lp[mi], e[mi], s=4, color=c_ms, alpha=0.35, label="matched MS", rasterized=True)
    axC.scatter(lp[rc], e[rc], s=8, color=c_rc, alpha=0.8, label="red clump", rasterized=True)
    plotting.apply_axes_style(axC, style, xlabel=r"log$_{10}$ P [d]", ylabel="eccentricity", title="(C) e vs log P")
    axC.legend(frameon=False, fontsize=8, loc="upper left")
    for j, (a, b) in enumerate(pbins):
        ax = fig.add_subplot(gs[1, j] if j < 3 else gs[1, 3])
        r = rc & (lp >= a) & (lp < b)
        mb = match(r, ms_all & (np.isfinite(sig) & (sig >= args.sig_min)) & (lp >= a) & (lp < b)) if r.sum() > 0 else np.array([], int)
        s = res[f"logP_{a}_{b}"]
        if r.sum() > 0:
            cdf(ax, e[r], "RC", c_rc)
        if mb.size:
            cdf(ax, e[mb], "MS", c_ms)
        t = f"n {s.get('n_rc', 0)} / {s.get('n_ms', 0)}"
        if "ks_p" in s:
            t += f"\nKS p {s['ks_p']:.1e}\ne<0.1: {s['f_e_lt_0p1_rc'][0]:.2f}±{s['f_e_lt_0p1_rc'][1]:.2f} vs {s['f_e_lt_0p1_ms'][0]:.2f}±{s['f_e_lt_0p1_ms'][1]:.2f}"
        plotting.apply_axes_style(ax, style, xlabel="eccentricity", ylabel="cumulative fraction", title=f"(B) log P {a}–{b}")
        ax.text(0.97, 0.05, t, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    axn = fig.add_subplot(gs[1, 3])
    axn.axis("off")
    axn.text(0.0, 0.95, "\n".join([
        f"quality: a0/σ_a0 ≥ {args.sig_min:g} (dotted in A: no cut)",
        f"matching: {args.k}-NN with replacement on",
        "log M_init, log d, [Fe/H], log P",
        f"M_init/M_now (RC) median {out['m_init_over_m_now_rc_median']:.3f}",
        "[Fe/H] RC: " + ", ".join(f"{k} {v}" for k, v in out["feh_coverage"]["sig"]["rc"].items()),
        "APOKASC-3 in box: " + (", ".join(f"{k} {v}" for k, v in apok.get("clump box (evolved)", {}).items()) if isinstance(apok, dict) else "n/a"),
    ]), va="top", fontsize=8, family="serif")
    for j, (name, v, xl) in enumerate((("M_init", m_init, r"M$_{\rm init}$ [M$_\odot$]"), ("d", rr / 1000, "d [kpc]"),
                                        ("feh", feh, "[Fe/H] (GSP-Phot/MSC)"), ("logP", lp, r"log$_{10}$ P [d]"))):
        ax = fig.add_subplot(gs[2, j])
        a1, b1 = v[rc], v[mi]
        cdf(ax, a1[np.isfinite(a1)], "RC", c_rc)
        cdf(ax, b1[np.isfinite(b1)], "MS", c_ms)
        plotting.apply_axes_style(ax, style, xlabel=xl, ylabel="CDF", title="(D) matched marginals" if j == 0 else None)
    plotting.layout_figure(fig)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plotting.save_figure(fig, args.out_dir / "rc_vs_ms_eccentricity.png", dpi=150)
    fig.savefig(args.out_dir / "rc_vs_ms_eccentricity.pdf")
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
