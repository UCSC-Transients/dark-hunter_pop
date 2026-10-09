#!/usr/bin/env python3
"""Symmetric spurious-cut diagnostic for #391 rung 2 (analysis only, no new draws).

The same cuts are applied to the mock-accepted orbits (generations 23–27, cached weights from
``scripts/diagnose_distance_count_391.py``) and to the real DR3 Orbital + AstroSpectroSB1 orbits
(mirror filters, zero-point-corrected parallax as in docs/gate391/high_fm):

* C0: no cut (reference);
* C1: drop P > 0.8 × 1038 d (the DR3 baseline);
* C2: C1 + drop F2 (goodness_of_fit) above a threshold;
* C3: C1 + drop significance (a0/σ_a0) on the side enriched in high-f_m orbits.

Thresholds are fixed *before* any mock comparison, from the real data only: the value where the
real high-f_m (f_m > 0.1) and low-f_m empirical CDFs separate most (KS point) among real orbits
at d > 0.7 kpc. Each is also run at ±20% to show the sensitivity.

Per cut: mock/real count ratio vs distance and G, six-panel weighted KS on bins with ESS ≥ 30
(no W and 2-D CMD W), for all and for CMD dwarfs, and the high-f_m share vs distance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASELINE = 1038.0
DBIN = np.array([0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 3.0])
GBIN = np.array([5, 10, 12, 13, 14, 15, 16, 19.0])
PANELS = ("P_orb_days", "G_mag", "inv_parallax_mas_inv", "eccentricity", "f_m_msun", "cos_inclination")


def ks_split_point(hi: np.ndarray, lo: np.ndarray) -> tuple[float, str]:
    """Value maximizing |CDF_lo − CDF_hi| and which side (above/below) holds the high-f_m excess."""
    hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
    grid = np.unique(np.quantile(np.concatenate([hi, lo]), np.linspace(0.01, 0.99, 400)))
    ch = np.searchsorted(np.sort(hi), grid, side="right") / hi.size
    cl = np.searchsorted(np.sort(lo), grid, side="right") / lo.size
    j = int(np.argmax(np.abs(cl - ch)))
    side = "above" if cl[j] > ch[j] else "below"  # high-f_m more abundant above the split
    return float(grid[j]), side


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--enrichment", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--rung2", type=Path, default=Path("config/population/rung2_validation.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import yaml
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
    from darkhunter_pop.giants import classify_evolved, classify_parent, cmd_for_rows, load_giants_config, parent_row_cmd
    from darkhunter_pop.physics_utils import astrometric_mass_function
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style, six_panel_bin_edges

    cfg = load_config()
    gm = import_gaiamock_mod()
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    min_ess = float(yaml.safe_load(args.rung2.read_text())["rung2"]["min_ess_per_bin"])
    n_bins = int(yaml.safe_load(args.rung2.read_text())["rung2"]["n_bins"])
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- mock ----------------
    z = np.load(args.cache, allow_pickle=True)
    acc = np.asarray(z["accepted"], bool)
    c = np.asarray(z["cascade"], float)[acc]
    row = np.asarray(z["parent_row"], np.int64)[acc]
    W = {"noW": np.asarray(z["w_noW"], float)[acc], "cmdW": np.asarray(z["w_cmdW"], float)[acc]}
    m_plx, m_P, m_e = c[:, 0], c[:, 10], c[:, 14]
    m_a0, m_sa0, m_f2 = c[:, 17], c[:, 18], c[:, 21]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_sig = m_a0 / m_sa0
        m_fm = astrometric_mass_function(m_a0, m_plx, m_P)
    m_g = np.asarray(parent.columns["phot_g_mean_mag"], float)[row]
    m_cos = np.cos(np.radians(c[:, 16]))
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pcls, ridge = classify_parent(parent, cfg, gcfg, cmd=parent_row_cmd(parent, cfg, gcfg))
    m_dwarf = pcls.classified[row] & ~pcls.evolved[row]

    # ---------------- real ----------------
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr_p > 20)]) for a, b in zip(gedges[:-1], gedges[1:])])
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]

    def col(name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(rt[name], float), np.nan)

    r_g = col("g_mag")
    r_plx = col("parallax") + zp[np.clip(np.digitize(r_g, gedges) - 1, 0, zp.size - 1)]
    r_P, r_e, r_f2 = col("period"), col("eccentricity"), col("goodness_of_fit")
    a0, _, _, inc = gm.get_Campbell_elements(col("A"), col("B"), col("F"), col("G"))
    r_a0 = np.asarray(a0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_fm = astrometric_mass_function(r_a0, r_plx, r_P)
    r_cos = np.cos(np.asarray(inc, float))
    en = Table.read(args.enrichment, format="ascii.ecsv")
    if "SOURCE_ID" in en.colnames:
        en.rename_column("SOURCE_ID", "source_id")
    en = en[np.isin(np.asarray(en["nss_solution_type"]).astype(str), types)]
    ke = np.char.add(np.asarray(en["source_id"]).astype(str), np.asarray(en["nss_solution_type"]).astype(str))
    kr = np.char.add(np.asarray(rt["source_id"]).astype(str), np.asarray(rt["nss_solution_type"]).astype(str))
    o = np.argsort(ke)
    pos = np.clip(np.searchsorted(ke, kr, sorter=o), 0, o.size - 1)
    r_sig = np.where(ke[o[pos]] == kr, np.ma.filled(np.ma.asarray(en["significance"], float), np.nan)[o[pos]], np.nan)
    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    eplx = col("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(r_plx > 0, 1000 / r_plx, np.nan)
        rlo = np.where(r_plx > 0, 1000 / (r_plx + eplx), np.nan)
        rhi = np.where(r_plx - eplx > 0, 1000 / (r_plx - eplx), np.nan)
    rcmd = cmd_for_rows(r_g, col("bp_mag") - col("rp_mag"), gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
    rcls = classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, ridge, gcfg.provisional_n_sigma)
    r_dwarf = rcls.classified & ~rcls.evolved
    with np.errstate(divide="ignore", invalid="ignore"):
        r_d, m_d = 1.0 / r_plx, 1.0 / m_plx

    # ---------------- thresholds from the real data only ----------------
    far = (r_d > 0.7) & (r_P <= 0.8 * BASELINE)
    hi_r = r_fm > 0.1
    t_f2, side_f2 = ks_split_point(r_f2[far & hi_r], r_f2[far & ~hi_r])
    t_sig, side_sig = ks_split_point(r_sig[far & hi_r], r_sig[far & ~hi_r])
    rep = [
        "#391 symmetric spurious-cut diagnostic (analysis only; gens 23-27 mock, real mirror filters + ZP-corrected parallax)",
        f"thresholds from real orbits at d > 0.7 kpc with P <= 830 d (KS split of high- vs low-f_m CDFs): "
        f"F2 {t_f2:.2f} (high-f_m enriched {side_f2}); significance {t_sig:.2f} (high-f_m enriched {side_sig})",
        "mock F2 is known to sit ~1.2 below published F2 at fixed orbit (#390), so an F2 cut is NOT symmetric in effect; reported anyway.",
        "",
    ]

    def keep_mask(P, f2, sg, cut: str, scale: float) -> np.ndarray:
        k = np.ones(P.size, bool)
        if cut == "C0":
            return k
        k &= ~(P > 0.8 * BASELINE * (scale if cut == "C1" else 1.0))
        if cut == "C2":
            thr = t_f2 * scale
            k &= ~((f2 > thr) if side_f2 == "above" else (f2 < thr))
        if cut == "C3":
            thr = t_sig * scale
            k &= ~((sg > thr) if side_sig == "above" else (sg < thr))
        return k

    def ks_bins(rv, mv, w):
        out = {}
        for name in PANELS:
            sc, lo, hi = axes_cfg[name]
            edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=n_bins)
            bw = ps.binned_weights(mv[name], w, edges)
            good = bw.ess >= min_ess
            rb, mb = np.digitize(rv[name], edges) - 1, np.digitize(mv[name], edges) - 1
            rin = (rb >= 0) & (rb < n_bins) & good[np.clip(rb, 0, n_bins - 1)]
            min_ = (mb >= 0) & (mb < n_bins) & good[np.clip(mb, 0, n_bins - 1)]
            if good.any() and rin.any() and min_.any():
                d_, ne, p_ = ps.weighted_ks(rv[name][rin], mv[name][min_], w[min_])
                out[name] = (d_, p_, float(rin.mean()))
            else:
                out[name] = (np.nan, np.nan, 0.0)
        return out

    results = {}
    cuts = [("C0", 1.0), ("C1", 1.0), ("C2", 1.0), ("C3", 1.0), ("C1", 0.8), ("C1", 1.2), ("C2", 0.8), ("C2", 1.2), ("C3", 0.8), ("C3", 1.2)]
    for cut, scale in cuts:
        km = keep_mask(m_P, m_f2, m_sig, cut, scale)
        kr_ = keep_mask(r_P, r_f2, r_sig, cut, scale)
        lab = f"{cut}" + ("" if scale == 1.0 else f"x{scale:g}")
        w = W["noW"]
        rd = np.histogram(r_d[kr_], bins=DBIN)[0].astype(float)
        md = np.histogram(m_d[km], bins=DBIN, weights=w[km])[0]
        rgc = np.histogram(r_g[kr_], bins=GBIN)[0].astype(float)
        mgc = np.histogram(m_g[km], bins=GBIN, weights=w[km])[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_d, ratio_g = md / rd, mgc / rgc
        hs_r = [np.mean(r_fm[kr_ & (np.digitize(r_d, DBIN) - 1 == k)] > 0.1) for k in range(DBIN.size - 1)]
        hs_m = []
        for k in range(DBIN.size - 1):
            s = km & (np.digitize(m_d, DBIN) - 1 == k)
            hs_m.append(float(np.sum(w[s] * (m_fm[s] > 0.1)) / max(np.sum(w[s]), 1e-300)))
        e_r = float(np.nanmean(r_e[kr_]))
        e_m = float(np.sum(w[km] * m_e[km]) / np.sum(w[km]))
        tot = float(np.sum(w[km]) / kr_.sum())
        results[lab] = dict(ratio_d=ratio_d, ratio_g=ratio_g, hs_r=hs_r, hs_m=hs_m, e_r=e_r, e_m=e_m, tot=tot,
                            n_real=int(kr_.sum()), ess=ps.kish_ess(w[km]))
        rep.append(f"{lab}: real kept {kr_.sum()} ({kr_.mean():.3f}); mock kept weight {np.sum(w[km]) / np.sum(w):.3f}; "
                   f"total ratio {tot:.3f}; ESS {ps.kish_ess(w[km]):.0f}; mean e real {e_r:.3f} mock {e_m:.3f}")
        rep.append("   ratio vs d: " + " ".join(f"{DBIN[k]:g}-{DBIN[k+1]:g}:{ratio_d[k]:.2f}" for k in range(DBIN.size - 1)))
        rep.append("   ratio vs G: " + " ".join(f"{GBIN[k]:g}-{GBIN[k+1]:g}:{ratio_g[k]:.2f}" for k in range(GBIN.size - 1)))
        rep.append("   high-f_m share real/mock vs d: " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(hs_r, hs_m)))
        if scale == 1.0:
            rv_all = {"P_orb_days": r_P, "G_mag": r_g, "inv_parallax_mas_inv": r_d, "eccentricity": r_e, "f_m_msun": r_fm, "cos_inclination": r_cos}
            mv_all = {"P_orb_days": m_P, "G_mag": m_g, "inv_parallax_mas_inv": m_d, "eccentricity": m_e, "f_m_msun": m_fm, "cos_inclination": m_cos}
            for sub, rmask, mmask in (("all", kr_, km), ("dwarfs", kr_ & r_dwarf, km & m_dwarf)):
                for wk in ("noW", "cmdW"):
                    ks = ks_bins({k: v[rmask] for k, v in rv_all.items()}, {k: v[mmask] for k, v in mv_all.items()}, W[wk][mmask])
                    rep.append(f"   KS {sub} {wk} (D,p,real-frac in ESS>=30 bins): " + " ".join(
                        f"{k.split('_')[0]}:{v[0]:.3f},{v[1]:.2g},{v[2]:.2f}" for k, v in ks.items()))
                    results[lab][f"ks_{sub}_{wk}"] = ks

    # ---------------- figures ----------------
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    dc = 0.5 * (DBIN[1:] + DBIN[:-1])
    gc = 0.5 * (GBIN[1:] + GBIN[:-1])
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, lab in enumerate(("C0", "C1", "C2", "C3")):
        st = series_style(i, style)
        r = results[lab]
        axs[0].plot(dc, r["ratio_d"], "o-", color=st["color"], ls=st["linestyle"], lw=2, label=lab)
        axs[1].plot(gc, r["ratio_g"], "o-", color=st["color"], ls=st["linestyle"], lw=2, label=lab)
        axs[2].plot(dc, r["hs_r"], "o-", color=st["color"], lw=2, label=f"{lab} real")
        axs[2].plot(dc, r["hs_m"], "s:", color=st["color"], lw=2, label=f"{lab} mock")
    for ax, xl, yl in ((axs[0], "1/parallax (kpc)", "mock / real counts"), (axs[1], "G (mag)", "mock / real counts"),
                       (axs[2], "1/parallax (kpc)", "share with f_m > 0.1")):
        apply_axes_style(ax, style, xlabel=xl, ylabel=yl)
        ax.legend(fontsize=10, ncol=2)
    axs[0].axhline(1, color="0.4", ls=":"); axs[1].axhline(1, color="0.4", ls=":")
    fig.suptitle("Symmetric cuts: C1 P<=0.8x1038 d; C2 +F2; C3 +significance (no Malmquist weight)", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "cuts_ratio_and_highfm.png", dpi=int(cfg.diagnostics.figure_dpi))

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, cut in zip(axs, ("C1", "C2", "C3")):
        for i, scale in enumerate((0.8, 1.0, 1.2)):
            lab = cut if scale == 1.0 else f"{cut}x{scale:g}"
            st = series_style(i, style)
            ax.plot(dc, results[lab]["ratio_d"], "o-", color=st["color"], lw=2, label=f"threshold x{scale:g}")
        ax.axhline(1, color="0.4", ls=":")
        apply_axes_style(ax, style, xlabel="1/parallax (kpc)", ylabel="mock / real counts")
        ax.set_title(f"{cut}: sensitivity to the threshold (+-20%)")
        ax.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "cuts_threshold_sensitivity.png", dpi=int(cfg.diagnostics.figure_dpi))

    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
