#!/usr/bin/env python3
"""Distance and count deficit diagnostics for #391 rung 2 (analysis only, no new draws).

Tests (Ryan, 2026-10-09):

* **A.** Absolute expected accepted-orbit counts, mock vs real, binned in 1/ϖ, G, BP−RP, l, b,
  ecliptic latitude β, a crowding proxy (parent-snapshot source density per HEALPix pixel, scaled
  to the full G < 19 parent) and a scan proxy (the star's own ``visibility_periods_used``).
* **B.** Truth-distance scale: simulated observed parallax of *all* draws vs the parent star's
  observed ``parallax``, against ϖ/σ, G and position.
* **C.** Orbit fraction N_orbit / N_parent vs distance, real and mock, split by G and by
  dereddened colour (an M1 proxy usable on both sides).
* **D.** Eccentricity vs distance and G: mock (weighted) vs real.

Weights are the decided target (MdS17 published, MIST coeval, evolved relation) with and
without the 2-D CMD Malmquist weight, over generations 23–27 by deterministic-mixture weights.
They are computed once and cached (``--cache``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style
from darkhunter_pop import proposal_set as ps

NSIDE = 8


def build_cache(args: argparse.Namespace, cfg) -> None:
    from darkhunter_pop import malmquist_cmd as mc

    parts = [ps.read_proposal_artifact(a) for a in args.artifact]
    common = set.intersection(*(set(p[0]) for p in parts))
    t = {k: np.concatenate([np.asarray(p[0][k]) for p in parts]) for k in parts[0][0] if k in common}
    o = {k: np.concatenate([np.asarray(p[1][k]) for p in parts]) for k in parts[0][1]}
    tg = ps.MdS17TargetConfig.model_validate(json.loads(parts[0][2]["provenance_json"])["target_mds17_json"])
    gcs = [ps.ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for p in parts]
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, gcs[0])
    if any(g.flux.evolved_rows_centre == "evolved_relation_deblended" for g in gcs):
        ps.ensure_m1_deblend_dark(t, parent, cfg, gcs)
    evo = ps.evolved_mg0_for_draws(t, parent)
    rel = ps.mist_relation_for_draws(t, parent, cfg)
    ll = ps.mds17_luminous_log_intensity(t, tg, evolved_mg0_system=evo, relation_log10_f=rel)
    lw, _ = ps.malmquist_cmd_log_weight(t, parent, tg, mc.load_cmd_malmquist_config(args.cmd_malmquist), cfg)
    lqs = [ps.log_q_total_for(t, parent, g, cfg) for g in gcs]
    ns = [g.n_draws for g in gcs]
    w0 = ps.importance_weights(ll, lqs, ns, scale_to_full=parent.scale_to_full)
    w1 = ps.importance_weights(ll + lw, lqs, ns, scale_to_full=parent.scale_to_full)
    keep = {k: t[k] for k in ("parent_row", "generation", "eccentricity", "period_days", "parallax_mas",
                              "measured_parallax_mas", "phot_g_mean_mag", "ra_deg", "dec_deg", "m1_msun")}
    np.savez(args.cache, w_noW=w0, w_cmdW=w1, accepted=o["accepted_orbital"], cascade=o["cascade"],
             solution_type=o["solution_type"], is_evolved=np.isfinite(evo), **keep)
    print(f"cached {w0.size} draws -> {args.cache}")


def sky(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    g = c.galactic
    e = c.transform_to(BarycentricTrueEcliptic())
    return g.l.deg, g.b.deg, e.lat.deg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, action="append", required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--cmd-malmquist", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config()
    if not args.cache.exists():
        build_cache(args, cfg)
    z = np.load(args.cache, allow_pickle=True)
    import healpy as hp
    from astropy.table import Table

    prop = ps.ProposalConfig.model_validate_json(ps.read_proposal_artifact(args.artifact[0])[2]["proposal_config_json"])
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    pc = parent.columns
    u = np.asarray(parent.usable, bool)
    scale = parent.scale_to_full
    acc = np.asarray(z["accepted"], bool)
    casc = np.asarray(z["cascade"], float)
    row = np.asarray(z["parent_row"], np.int64)
    W = {"noW": np.asarray(z["w_noW"], float), "cmdW": np.asarray(z["w_cmdW"], float)}

    # ---- parent-side properties (shared by mock draws through their row) ----
    p_ra, p_dec = np.asarray(pc["ra"], float), np.asarray(pc["dec"], float)
    p_l, p_b, p_beta = sky(p_ra, p_dec)
    p_pix = hp.ang2pix(NSIDE, p_ra, p_dec, lonlat=True)
    dens = np.bincount(p_pix[u], minlength=hp.nside2npix(NSIDE)) * scale / hp.nside2pixarea(NSIDE, degrees=True)
    p_bprp = np.asarray(pc["bp_rp"], float)
    p_vp = np.asarray(pc["visibility_periods_used"], float)
    p_plx = np.asarray(pc["parallax"], float)
    p_eplx = np.asarray(pc["parallax_error"], float)
    p_g = np.asarray(pc["phot_g_mean_mag"], float)
    colour0 = np.asarray(parent.cmd["colour0"], float) if parent.cmd is not None else p_bprp

    # ---- real side (167,911 after the mirror filters) ----
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    inp = ps.load_real_input_columns(args.real_input_columns)
    keep, _ = ps.real_comparison_keep(rt, prop, inp)
    rt = rt[keep]

    def rc(name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(rt[name], float), np.nan)

    r_ra, r_dec = rc("ra"), rc("dec")
    r_l, r_b, r_beta = sky(r_ra, r_dec)
    r_plx, r_g = rc("parallax"), rc("g_mag")
    r_bprp = rc("bp_mag") - rc("rp_mag")
    r_e = rc("eccentricity")
    r_dens = dens[hp.ang2pix(NSIDE, r_ra, r_dec, lonlat=True)]
    sid = np.asarray(rt["source_id"], np.int64)
    order = np.argsort(inp["source_id"])
    pos = np.clip(np.searchsorted(inp["source_id"], sid, sorter=order), 0, order.size - 1)
    r_vp = np.where(inp["source_id"][order[pos]] == sid, np.asarray(inp["visibility_periods_used"], float)[order[pos]], np.nan)
    n_real = len(rt)

    # ---- mock accepted-orbit quantities ----
    m_plx = np.where(acc, casc[:, 0], np.nan)
    m_e = np.where(acc, casc[:, 14], np.nan)
    m_g = p_g[row]
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep: list[str] = ["#391 distance / count deficit diagnostics (generations 23-27)", ""]
    for k, w in W.items():
        rep.append(f"{k}: expected accepted orbits {np.sum(w[acc]):.4g} vs real {n_real} (ratio {np.sum(w[acc]) / n_real:.3f}); ESS {ps.kish_ess(w[acc]):.0f}")

    # ================= A: absolute counts =================
    with np.errstate(divide="ignore", invalid="ignore"):
        variables = {
            "1/parallax (kpc)": (1.0 / m_plx, 1.0 / r_plx, np.linspace(0, 3, 16)),
            "G (mag)": (m_g, r_g, np.linspace(6, 18, 13)),
            "BP-RP (mag)": (p_bprp[row], r_bprp, np.linspace(0.0, 3.0, 13)),
            "Galactic l (deg)": (p_l[row], r_l, np.linspace(0, 360, 13)),
            "Galactic b (deg)": (p_b[row], r_b, np.linspace(-90, 90, 13)),
            "ecliptic latitude (deg)": (p_beta[row], r_beta, np.linspace(-90, 90, 13)),
            "log10 parent density (deg$^{-2}$)": (np.log10(dens[p_pix[row]]), np.log10(r_dens), np.linspace(2.5, 5.0, 11)),
            "visibility periods used": (p_vp[row], r_vp, np.arange(8.5, 30.6, 2.0)),
        }
    fig, axs = plt.subplots(4, 2, figsize=(14, 18))
    rep += ["", "A. mock/real ratio of absolute expected counts per bin (noW | cmdW); real count in brackets"]
    for ax, (name, (mv, rv, edges)) in zip(np.ravel(axs), variables.items()):
        rn = np.histogram(rv[np.isfinite(rv)], bins=edges)[0].astype(float)
        cen = 0.5 * (edges[1:] + edges[:-1])
        line = [name]
        for i, (k, w) in enumerate(W.items()):
            sel = acc & np.isfinite(mv)
            mn = np.histogram(mv[sel], bins=edges, weights=w[sel])[0]
            mn2 = np.histogram(mv[sel], bins=edges, weights=w[sel] ** 2)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = mn / rn
                err = np.sqrt(mn2) / rn
            st = series_style(i, style)
            ax.errorbar(cen, ratio, yerr=err, color=st["color"], ls=st["linestyle"], marker="o", lw=2, label=f"mock {k} / real")
            if i == 0:
                line += [f"{c:.3g}:{r:.2f}[{int(n)}]" for c, r, n in zip(cen, ratio, rn)]
        ax.axhline(1.0, color="0.4", ls=":")
        ax.set_ylim(0, 2)
        apply_axes_style(ax, style, xlabel=name, ylabel="mock / real counts")
        ax.legend(fontsize=style.tick_label_fontsize)
        rep.append("  " + " ".join(line))
    fig.suptitle("A: mock / real absolute accepted-orbit counts (gens 23-27)", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "A_count_ratio_profiles.png", dpi=int(cfg.diagnostics.figure_dpi))

    # sky ratio map (noW), HEALPix nside 4
    ns4 = 4
    rm = np.bincount(hp.ang2pix(ns4, r_ra, r_dec, lonlat=True), minlength=hp.nside2npix(ns4)).astype(float)
    mpix = hp.ang2pix(ns4, p_ra[row], p_dec[row], lonlat=True)
    mm = np.bincount(mpix[acc], weights=W["noW"][acc], minlength=hp.nside2npix(ns4))
    with np.errstate(divide="ignore", invalid="ignore"):
        rmap = np.where(rm > 20, mm / rm, hp.UNSEEN)
    plt.figure(figsize=(10, 6))
    hp.mollview(rmap, coord=["C", "G"], title="A: mock (noW) / real accepted orbits per HEALPix nside-4 pixel (Galactic)",
                min=0, max=2, cmap="RdBu_r", hold=True)
    hp.graticule()
    plt.savefig(args.out_dir / "A_count_ratio_sky.png", dpi=int(cfg.diagnostics.figure_dpi))
    plt.close("all")

    # ================= B: simulated vs observed parallax, all draws =================
    sim_plx = np.where(casc[:, 0] > 0, casc[:, 0], np.where(casc[:, 0] < 0, casc[:, 2], np.nan))
    sim_err = np.where(casc[:, 0] > 0, casc[:, 1], np.where(casc[:, 0] < 0, casc[:, 3], np.nan))
    obs, eobs = p_plx[row], p_eplx[row]
    truth = np.asarray(z["parallax_mas"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (sim_plx - obs) / np.sqrt(sim_err**2 + eobs**2)
        snr = obs / eobs
        tr_pull = (truth - obs) / eobs
    ok = np.isfinite(pull) & (sim_err > 0) & (sim_err < 10)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    rep += ["", "B. (simulated - observed parallax) / sqrt(sigma_sim^2 + sigma_obs^2), all draws with a 5/7/9/12-par solution;"
            " and (truth - observed)/sigma_obs (truth = 1000/r_med_geo)"]
    for ax, (name, x, edges) in zip(axs, [("parallax / error (observed)", snr, np.array([0, 2, 3, 5, 7, 10, 20, 50, 1e4])),
                                         ("G (mag)", p_g[row], np.linspace(6, 19, 14)),
                                         ("|ecliptic latitude| (deg)", np.abs(p_beta[row]), np.linspace(0, 90, 10))]):
        b = np.digitize(x, edges) - 1
        med, trm, cen, nn = [], [], [], []
        for i in range(edges.size - 1):
            s = ok & (b == i)
            if s.sum() < 50:
                continue
            med.append(np.median(pull[s])); trm.append(np.median(tr_pull[s])); nn.append(int(s.sum()))
            cen.append(np.sqrt(edges[i] * edges[i + 1]) if name.startswith("parallax") else 0.5 * (edges[i] + edges[i + 1]))
        ax.plot(cen, med, "o-", lw=2, color=series_style(0, style)["color"], label="median pull: simulated - observed")
        ax.plot(cen, trm, "s--", lw=2, color=series_style(1, style)["color"], label="median (truth - observed)/sigma_obs")
        ax.axhline(0, color="0.4", ls=":")
        if name.startswith("parallax"):
            ax.set_xscale("log")
        apply_axes_style(ax, style, xlabel=name, ylabel="median pull")
        ax.legend(fontsize=style.tick_label_fontsize)
        rep.append(f"  {name}: " + " ".join(f"{c:.3g}:{m:+.2f}/{t:+.2f}(n={n})" for c, m, t, n in zip(cen, med, trm, nn)))
    fig.suptitle("B: simulated vs observed parallax (all draws)", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "B_parallax_pulls.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ================= C: orbit fraction vs distance =================
    with np.errstate(divide="ignore", invalid="ignore"):
        p_d = 1.0 / p_plx
        r_d = 1.0 / r_plx
    m_drow = p_d[row]
    dedges = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    dc = 0.5 * (dedges[1:] + dedges[:-1])
    splits = {"G<12": (lambda g: g < 12), "12<=G<14": (lambda g: (g >= 12) & (g < 14)),
              "14<=G<16": (lambda g: (g >= 14) & (g < 16)), "G>=16": (lambda g: g >= 16)}
    csplits = {"colour0<0.6": (lambda c: c < 0.6), "0.6-0.9": (lambda c: (c >= 0.6) & (c < 0.9)),
               "0.9-1.3": (lambda c: (c >= 0.9) & (c < 1.3)), ">=1.3": (lambda c: c >= 1.3)}
    r_colour0 = None
    try:
        from darkhunter_pop.giants import cmd_for_rows, load_giants_config
        from astropy.coordinates import SkyCoord
        import astropy.units as au

        gcfg = load_giants_config(Path("config/population/giants.yaml"))
        eplx = rc("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(r_plx > 0, 1000 / r_plx, np.nan)
            rlo = np.where(r_plx > 0, 1000 / (r_plx + eplx), np.nan)
            rhi = np.where(r_plx - eplx > 0, 1000 / (r_plx - eplx), np.nan)
        r_colour0 = cmd_for_rows(r_g, r_bprp, r_l, r_b, rr, rlo, rhi, cfg, gcfg).colour0
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        rep.append(f"  (real dereddened colour unavailable: {exc}; using observed BP-RP)")
        r_colour0 = r_bprp
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    rep += ["", "C. N_orbit / N_parent vs distance (d = 1/observed parallax; parent usable rows x scale)"]
    for ax, (title, sp, pvar, rvar, mvar) in zip(axs, [("by G", splits, p_g, r_g, p_g[row]),
                                                       ("by dereddened colour (M1 proxy)", csplits, colour0, r_colour0, colour0[row])]):
        for i, (lab, f) in enumerate(sp.items()):
            pn = np.histogram(p_d[u & f(pvar)], bins=dedges)[0] * scale
            rn = np.histogram(r_d[f(rvar)], bins=dedges)[0]
            sel = acc & f(mvar)
            mn = np.histogram(m_drow[sel], bins=dedges, weights=W["noW"][sel])[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                fr, fm = rn / pn, mn / pn
            st = series_style(i, style)
            ax.plot(dc, fr, "o-", color=st["color"], lw=2, label=f"real {lab}")
            ax.plot(dc, fm, "s--", color=st["color"], lw=2, label=f"mock {lab}")
            rep.append(f"  {title} {lab}: " + " ".join(f"{c:.2f}:{a:.2e}/{b:.2e}" for c, a, b in zip(dc, fr, fm)) + "  (real/mock)")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel="1/parallax (kpc)", ylabel="orbits per parent star")
        ax.legend(fontsize=11, ncol=2)
        ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "C_orbit_fraction.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ================= D: eccentricity vs distance and G =================
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    rep += ["", "D. mean eccentricity, mock noW-weighted vs real (accepted orbits)"]
    for ax, (name, mv, rv, edges) in zip(axs, [("1/parallax (kpc)", 1.0 / m_plx, 1.0 / r_plx, np.array([0, .2, .4, .6, .8, 1.0, 1.5, 2.5])),
                                              ("G (mag)", m_g, r_g, np.array([5, 9, 11, 12, 13, 14, 15, 16, 19]))]):
        cen = 0.5 * (edges[1:] + edges[:-1])
        rb, mb = np.digitize(rv, edges) - 1, np.digitize(mv, edges) - 1
        re_, me_, rn_, ess_ = [], [], [], []
        for i in range(cen.size):
            rs = (rb == i) & np.isfinite(r_e)
            ms = acc & (mb == i) & np.isfinite(m_e)
            re_.append(np.mean(r_e[rs]) if rs.any() else np.nan)
            w = W["noW"][ms]
            me_.append(np.sum(w * m_e[ms]) / np.sum(w) if w.sum() > 0 else np.nan)
            rn_.append(int(rs.sum())); ess_.append(ps.kish_ess(w))
        ax.plot(cen, re_, "o-", lw=2, color=series_style(0, style)["color"], label="DR3")
        ax.plot(cen, me_, "s--", lw=2, color=series_style(1, style)["color"], label="mock (MdS17, noW)")
        apply_axes_style(ax, style, xlabel=name, ylabel="mean eccentricity")
        ax.legend(fontsize=style.tick_label_fontsize)
        rep.append(f"  {name}: " + " ".join(f"{c:.2f}:{a:.3f}/{b:.3f}(ESS {e:.0f})" for c, a, b, e in zip(cen, re_, me_, ess_)) + "  (real/mock)")
    fig.tight_layout()
    save_figure(fig, args.out_dir / "D_eccentricity_vs_distance_G.png", dpi=int(cfg.diagnostics.figure_dpi))

    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
