#!/usr/bin/env python3
"""Diagnose the rung-3 G-shape mismatch (#391 step (a), Ryan 2026-10-10): evolved vs dwarf primaries.

At the gens 23-30 control-fit best θ (``docs/gate391/rung3_refit4/gens23_30_gen29_fine_grid``, noW), compare
real and mock C1 orbits (inside the rung-3 outer edges) by G, split by the §10.2 dereddened-CMD class
(evolved / dwarf / unclassified), plus the CMD itself and the distance mix at fixed G. Analysis only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_refit4_391 as r4  # noqa: E402

GEN_ARTIFACTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5", "restart_gen26_topup.h5",
                 "restart_gen27_evolved.h5", "restart_gen28_recentred.h5", "restart_gen29_topup.h5", "restart_gen30_topup.h5")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_30.npz"))
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--fit", type=Path, default=Path("docs/gate391/rung3_refit4/gens23_30_gen29_fine_grid/summary.json"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_refit4_fine.yaml"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--new-gen", type=int, default=30)
    ap.add_argument("--weights", choices=("noW", "cmdW"), default="noW")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    import yaml
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.giants import classify_evolved, classify_parent, cmd_for_rows, load_giants_config, parent_row_cmd
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    ns = r4.setup(args, fc)
    cfg = ns.cfg
    fit = json.loads(args.fit.read_text())[args.weights]
    th = np.array(fit["theta"], float)
    w = ns.W[args.weights] * np.exp(ns.log_mod(th))
    orb = ns.in_o
    # parent rows of the likelihood draws (cache order = artifact order)
    prow = np.concatenate([np.asarray(ps.read_proposal_artifact(args.artifact_dir / a)[0]["parent_row"], np.int64) for a in GEN_ARTIFACTS])
    row = prow[ns.rel]
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pcmd = parent_row_cmd(parent, cfg, gcfg)
    pcls, ridge = classify_parent(parent, cfg, gcfg, cmd=pcmd)
    m_cls = np.where(~pcls.classified[row], 2, np.where(pcls.evolved[row], 1, 0))  # 0 dwarf, 1 evolved, 2 unclassified
    m_mg0, m_c0 = np.asarray(pcmd.mg0, float)[row], np.asarray(pcmd.colour0, float)[row]
    m_g, m_d = ns.o_cols[0], ns.o_cols[1]

    # real C1 orbits in the outer edges, with CMD class
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]

    def col(n):
        return np.ma.filled(np.ma.asarray(rt[n], float), np.nan)

    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    r_g = col("g_mag")
    r_plx = col("parallax") + zp[np.clip(np.digitize(r_g, ge) - 1, 0, 4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_d = np.where(r_plx > 0, 1 / r_plx, np.nan)
        rr = np.where(r_plx > 0, 1000 / r_plx, np.nan)
        rlo = np.where(r_plx > 0, 1000 / (r_plx + col("parallax_error")), np.nan)
        rhi = np.where(r_plx - col("parallax_error") > 0, 1000 / (r_plx - col("parallax_error")), np.nan)
    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    rcmd = cmd_for_rows(r_g, col("bp_mag") - col("rp_mag"), gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
    rcls = classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, ridge, gcfg.provisional_n_sigma)
    out = {k: np.asarray(v, float) for k, v in fc["outer"].items()}
    r_in = (col("period") <= ns.PMAX) & (r4.bin_index([r_g, r_d, np.log10(col("period")), col("eccentricity")], [out[a] for a in r4.AXES]) >= 0)
    r_cls = np.where(~rcls.classified, 2, np.where(rcls.evolved, 1, 0))[r_in]
    r_mg0, r_c0 = np.asarray(rcmd.mg0, float)[r_in], np.asarray(rcmd.colour0, float)[r_in]
    r_gi, r_di = r_g[r_in], r_d[r_in]
    N_o = float(r_in.sum())
    s_share = th[13] / N_o  # spurious share of real counts in every (G, d) region (§12.12)

    rep = [f"#391 step (a): G-shape diagnosis, evolved vs dwarf primaries (gens 23-30 control fit, {args.weights})", "",
           f"real C1 orbits {int(N_o)}; spurious share N_s/N_o = {s_share:.3f} (added uniformly in G); mock weights = control best fit",
           "class fractions (real | mock): " + ", ".join(f"{n} {np.mean(r_cls == k):.3f} | {w[orb][m_cls[orb] == k].sum() / w[orb].sum():.3f}"
                                                   for k, n in enumerate(("dwarf", "evolved", "unclassified")))]
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    gb = np.arange(5.0, 19.01, 1.0)
    cen = 0.5 * (gb[1:] + gb[:-1])
    fig, axs = plt.subplots(2, 3, figsize=(19, 11))
    rep += ["", "per G bin: real n | (mock + spurious)/real all | mock/real dwarf | mock/real evolved | evolved share real / mock | median d real / mock (kpc)"]
    mtot = np.zeros(cen.size)
    for i, (lo, hi) in enumerate(zip(gb[:-1], gb[1:])):
        rs = (r_gi >= lo) & (r_gi < hi)
        ms = orb & (m_g >= lo) & (m_g < hi)
        if rs.sum() < 50:
            continue
        mo = w[ms].sum()
        rd_, re_ = np.sum(rs & (r_cls == 0)), np.sum(rs & (r_cls == 1))
        md_, me_ = w[ms & (m_cls == 0)].sum(), w[ms & (m_cls == 1)].sum()
        wd = w[ms]
        o = np.argsort(m_d[ms])
        med_m = m_d[ms][o][np.searchsorted(np.cumsum(wd[o]), 0.5 * wd.sum())] if wd.sum() > 0 else np.nan
        ess_g = float(wd.sum() ** 2 / max(np.sum(wd**2), 1e-300))
        ess_e = float(w[ms & (m_cls == 1)].sum() ** 2 / max(np.sum(w[ms & (m_cls == 1)] ** 2), 1e-300))
        top = float(np.max(wd) / wd.sum()) if wd.sum() > 0 else np.nan
        rep.append(f"  G {lo:.0f}-{hi:.0f}: ESS {ess_g:.1f} (evolved {ess_e:.1f}; heaviest draw {top:.0%} of the bin; {int(ms.sum())} draws)")
        rep.append(f"  G {lo:.0f}-{hi:.0f}: {int(rs.sum())} | {(mo + s_share * rs.sum()) / rs.sum():.2f} | "
                   f"{(md_ + s_share * rd_) / max(rd_, 1):.2f} | {(me_ + s_share * re_) / max(re_, 1):.2f} | "
                   f"{re_ / rs.sum():.2f} / {me_ / max(mo, 1e-9):.2f} | {np.nanmedian(r_di[rs]):.2f} / {med_m:.2f}")
    for j, (k, name) in enumerate(((None, "all"), (0, "dwarf"), (1, "evolved"))):
        ax = axs[0, j]
        rsel = np.ones(r_gi.size, bool) if k is None else r_cls == k
        msel = orb if k is None else orb & (m_cls == k)
        rn = np.histogram(r_gi[rsel], bins=gb)[0].astype(float)
        mn = np.histogram(m_g[msel], bins=gb, weights=w[msel])[0] + s_share * rn
        ax.step(cen, rn, where="mid", color="k", lw=2, label="real")
        ax.step(cen, mn, where="mid", color=series_style(1, style)["color"], lw=2, label="mock + spurious share")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel="G (mag)", ylabel="C1 orbits per mag")
        ax.set_title(name, fontsize=style.title_fontsize)
        ax.legend(fontsize=style.tick_label_fontsize)
    # CMDs
    cb, mb = np.linspace(-0.2, 2.6, 41), np.linspace(-3, 9, 49)
    Hr = np.histogram2d(r_c0, r_mg0, bins=[cb, mb])[0]
    Hm = np.histogram2d(m_c0[orb], m_mg0[orb], bins=[cb, mb], weights=w[orb])[0]
    for ax, H, t in ((axs[1, 0], Hr, "real C1 orbits"), (axs[1, 1], Hm, "mock (control fit)")):
        ax.imshow(np.log10(np.maximum(H.T, 0.5)), origin="lower", aspect="auto", extent=[cb[0], cb[-1], mb[0], mb[-1]], cmap="viridis")
        ax.invert_yaxis()
        apply_axes_style(ax, style, xlabel="dereddened BP-RP", ylabel="M_G0")
        ax.set_title(t + " (log counts)", fontsize=style.title_fontsize)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where((Hr > 20), (Hm + s_share * Hr) / Hr, np.nan)
    im = axs[1, 2].imshow(np.log2(R.T), origin="lower", aspect="auto", extent=[cb[0], cb[-1], mb[0], mb[-1]], cmap="RdBu_r", vmin=-2, vmax=2)
    axs[1, 2].invert_yaxis()
    fig.colorbar(im, ax=axs[1, 2], label="log2((mock + spurious)/real)")
    apply_axes_style(axs[1, 2], style, xlabel="dereddened BP-RP", ylabel="M_G0")
    axs[1, 2].set_title("CMD ratio (cells with > 20 real)", fontsize=style.title_fontsize)
    fig.suptitle("#391 (a): G shape by CMD class and the dereddened CMD, control fit gens 23-30 (noW)", fontsize=style.title_fontsize)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out_dir / "g_shape_evolved_dwarf_cmd.png", dpi=int(cfg.diagnostics.figure_dpi))
    # CMD ratio in coarse M_G0 bands
    rep += ["", "by dereddened M_G0 band: real n | (mock + spurious)/real | median G real / mock"]
    for lo, hi in ((-3, 0), (0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 7), (7, 9)):
        rs = (r_mg0 >= lo) & (r_mg0 < hi)
        ms = orb & (m_mg0 >= lo) & (m_mg0 < hi)
        if rs.sum() < 50:
            continue
        wm = w[ms]
        o = np.argsort(m_g[ms])
        mg_med = m_g[ms][o][np.searchsorted(np.cumsum(wm[o]), 0.5 * wm.sum())] if wm.sum() > 0 else np.nan
        rep.append(f"  M_G0 {lo}-{hi}: {int(rs.sum())} | {(wm.sum() + s_share * rs.sum()) / rs.sum():.2f} | {np.median(r_gi[rs]):.1f} / {mg_med:.1f} | "
                   f"ESS {wm.sum() ** 2 / max(np.sum(wm**2), 1e-300):.1f}")
    # evolved vs dwarf by period (MP-Q28b: is the evolved excess at short P?)
    m_lp, r_lp = ns.o_cols[2], np.log10(col("period"))[r_in]
    rep += ["", "by log10 P and class: real n | (mock + spurious)/real, dwarf | evolved"]
    for lo, hi in ((0, 2.3), (2.3, 2.5), (2.5, 2.65), (2.65, 2.8), (2.8, 2.92)):
        parts_ = []
        for k in (0, 1):
            rs = (r_lp >= lo) & (r_lp < hi) & (r_cls == k)
            ms = orb & (m_lp >= lo) & (m_lp < hi) & (m_cls == k)
            parts_.append(f"{int(rs.sum())} | {(w[ms].sum() + s_share * rs.sum()) / max(rs.sum(), 1):.2f}")
        rep.append(f"  log P {lo}-{hi}: dwarf {parts_[0]}; evolved {parts_[1]}")
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
