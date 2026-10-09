#!/usr/bin/env python3
"""Giant / evolved-primary diagnostics for the mock population (MP-Q28, #413).

docs/MOCK_POPULATION_SPEC.md §10. Classifies evolved primaries (subgiants + giants) from the
dereddened CMD (``darkhunter_pop.giants``) on three sides, identically:

* the ``gaia_source`` parent snapshot (usable rows, spec §0.1);
* the real DR3 Orbital + AstroSpectroSB1 orbits, with the parent's mirror filters
  (``proposal_set.real_comparison_keep``) and the Bailer-Jones distances of
  ``scripts/fetch_nss_bailer_jones.py``;
* the accepted mock orbits of existing proposal-set artifacts (no gaiamock is run), raw and
  reweighted to MdS17 at published parameters exactly as ``scripts/plot_proposal_pilot.py``.

Writes to ``--out-dir``: ``giants_cmd.png``, ``giants_six_panel.png`` (evolved, real vs mock),
``giants_period_radius.png`` (P vs CMD radius with the Eggleton Roche-lobe floor),
``giants_m1_flame.png`` (TAG10 M1 vs FLAME mass, real orbits) and ``giants_report.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.diagnostic_hooks import SIX_PANEL_NAMES
from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels
from darkhunter_pop.giants import (
    classify_evolved,
    classify_parent,
    cmd_for_rows,
    cmd_radius_rsun,
    evolved_log10_flux_ratio,
    load_giants_config,
    parent_row_cmd,
    roche_period_floor_days,
)
from darkhunter_pop.plotting import (
    apply_axes_style,
    plot_six_panel_grid,
    require_pyplot,
    resolve_plotting_style,
    save_figure,
    series_style,
)
from darkhunter_pop.proposal_set import (
    MdS17TargetConfig,
    ProposalConfig,
    importance_weights,
    janssens_absolute_g,
    kish_ess,
    load_parent_snapshot,
    log_q_total_for,
    mds17_luminous_log_intensity,
    read_proposal_artifact,
    real_comparison_keep,
    weighted_ks,
)

PANEL_LABELS = {
    "P_orb_days": "orbital period (day)",
    "G_mag": "G (mag)",
    "inv_parallax_mas_inv": r"1/$\varpi$ (kpc)",
    "eccentricity": "eccentricity",
    "f_m_msun": r"astrometric mass function (M$_\odot$)",
    "cos_inclination": r"cos $i$",
}


def _wfrac(flag: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted fraction and its 1σ from the Kish ESS of the whole set (binomial on ESS)."""
    tot = float(np.sum(w))
    if tot <= 0:
        return float("nan"), float("nan")
    p = float(np.sum(w[flag]) / tot)
    ess = kish_ess(w)
    return p, float(np.sqrt(max(p * (1 - p), 0.0) / max(ess, 1.0)))


def _bfrac(k: int, n: int) -> str:
    p = k / n if n else float("nan")
    return f"{k}/{n} = {p:.4f} ± {np.sqrt(p * (1 - p) / n) if n else float('nan'):.4f}"


def mock_panel_values(truth: dict[str, Any], outcome: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fitted six-panel values per draw (same definition as ``plot_proposal_pilot``)."""
    c = np.asarray(outcome["cascade"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_plx = np.where(c[:, 0] > 0, 1.0 / c[:, 0], np.nan)
    return {
        "P_orb_days": c[:, 10],
        "G_mag": np.asarray(truth["phot_g_mean_mag"], float),
        "inv_parallax_mas_inv": inv_plx,
        "eccentricity": c[:, 14],
        "f_m_msun": np.asarray(outcome["f_m_msun"], float),
        "cos_inclination": np.asarray(outcome["cos_inclination_fit"], float),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True, action="append")
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True, help="uncut NSS snapshot query.ecsv")
    ap.add_argument(
        "--real-bj-dir", type=Path, default=None,
        help="fetch_nss_bailer_jones.py output dir; without it the real side uses the inverse NSS parallax",
    )
    ap.add_argument("--flame-snapshot", type=Path, default=None, help="flame_enrichment query.ecsv")
    ap.add_argument("--real-types", nargs="+", default=None,
                    help="restrict the real side to these nss_solution_type values (e.g. Orbital); default: the comparison set")
    ap.add_argument("--real-input-columns", type=Path, default=None,
                    help="MP-Q24 snapshot (fetch_real_nss_input_columns.py): drop real rows failing IPD/C*")
    ap.add_argument("--cmd-malmquist", type=Path, default=None,
                    help="add the 2-D CMD Malmquist weight (#418) to the target")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--giants-config", type=Path, default=Path("config/population/giants.yaml"))
    ap.add_argument("--label", default="MP-Q28 DIAGNOSTIC (pre-noise-fix, pre-Malmquist mock)")
    args = ap.parse_args(argv)

    import h5py
    import yaml
    from astropy.table import Table

    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    gcfg = load_giants_config(args.giants_config)
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    rep: list[str] = [f"{args.label} — #413 / docs/MOCK_POPULATION_SPEC.md §10", ""]

    # ---------------- mock artifacts and parent ----------------
    parts = [read_proposal_artifact(a) for a in args.artifact]
    attrs = parts[0][2]
    prov = json.loads(attrs["provenance_json"])
    target = MdS17TargetConfig.model_validate(prov["target_mds17_json"])
    truth = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    outcome = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    gen_cfgs = [ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for p in parts]
    parent = load_parent_snapshot(args.parent_dir, cfg, gen_cfgs[0])
    # Current decided target (#391 restart): evolved-row relation (#416) and MIST coeval (MP-Q40).
    import darkhunter_pop.proposal_set as _ps

    evo = _ps.evolved_mg0_for_draws(truth, parent) if getattr(parent, "cmd", None) is not None else None
    rel = (_ps.mist_relation_for_draws(truth, parent, cfg)
           if getattr(target, "mass_luminosity", None) == "mist_coeval" else None)
    log_lam = mds17_luminous_log_intensity(truth, target, evolved_mg0_system=evo, relation_log10_f=rel)
    if args.cmd_malmquist is not None:
        from darkhunter_pop import malmquist_cmd as _mc

        lw_c, _ = _ps.malmquist_cmd_log_weight(truth, parent, target, _mc.load_cmd_malmquist_config(args.cmd_malmquist), cfg)
        log_lam = log_lam + lw_c
    if any(gc.flux.evolved_rows_centre == "evolved_relation_deblended" for gc in gen_cfgs):
        import darkhunter_pop.proposal_set as _pse

        _pse.ensure_m1_deblend_dark(truth, parent, cfg, gen_cfgs)  # #391 option (i)
    log_qs = [log_q_total_for(truth, parent, gc, cfg) for gc in gen_cfgs]
    w = importance_weights(log_lam, log_qs, [gc.n_draws for gc in gen_cfgs], scale_to_full=float(attrs["scale_to_full"]))
    acc = np.asarray(outcome["accepted_orbital"], bool)

    # ---------------- parent classification ----------------
    pcmd = parent_row_cmd(parent, cfg, gcfg)
    pcls, ridge = classify_parent(parent, cfg, gcfg, cmd=pcmd)
    u = np.asarray(parent.usable, bool)
    cols = parent.columns
    with np.errstate(divide="ignore", invalid="ignore"):
        psnr = np.asarray(cols["parallax"], float) / np.asarray(cols["parallax_error"], float)
    plogg = np.asarray(cols["logg_gspphot"], float)
    pc = u & pcls.classified
    pev = pcls.evolved & u
    rep += [
        "MS ridge (parent usable rows, parallax_over_error >= "
        f"{gcfg.ridge.min_parallax_over_error}): colour0, ridge M_G0, faint-side sigma, rows",
    ]
    rep += [f"  {c:.2f} {m:.3f} {s:.3f} {n}" for c, m, s, n in zip(ridge.colour, ridge.mag, ridge.sigma, ridge.n_rows)]
    rep += [
        "",
        f"provisional_n_sigma = {gcfg.provisional_n_sigma} (MP-Q28 option)",
        "",
        "=== GIANT (EVOLVED) FRACTIONS ===",
        f"parent usable rows: {int(u.sum())}; extinction law failed: {int(np.sum(u & ~pcmd.extinction_ok))}; "
        f"CMD-classified: {int(pc.sum())} (outside ridge colour range or no CMD: {int(np.sum(u & ~pcls.classified))})",
        f"parent evolved (CMD): {_bfrac(int(pev.sum()), int(pc.sum()))}",
        f"parent old flag (TAG10-atmosphere log g < 3.6): {_bfrac(int(np.sum(parent.is_giant & u)), int(u.sum()))}",
        f"parent GSP-Phot log g < 3.6: {_bfrac(int(np.sum(u & (plogg < 3.6))), int(np.sum(u & np.isfinite(plogg))))}",
    ]
    for n_sig in (2.0, 3.0, 4.0, 5.0):
        alt = classify_evolved(pcmd.mg0, pcmd.colour0, pcmd.sigma_mu, ridge, n_sig)
        rep.append(f"  sensitivity n_sigma={n_sig}: parent evolved {_bfrac(int(np.sum(alt.evolved & u)), int(pc.sum()))}")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, np.inf)):
        s = pc & (psnr >= lo) & (psnr < hi)
        rep.append(
            f"  parent parallax/error in [{lo},{hi}): evolved {_bfrac(int(np.sum(pev & s)), int(s.sum()))}; "
            f"GSP-Phot log g<3.6 {np.mean(plogg[s] < 3.6):.4f}"
        )

    # RUWE of the parent's own rows at fixed G: does an evolved primary need extra epoch noise?
    pg = np.asarray(cols["phot_g_mean_mag"], float)
    pruwe = np.asarray(cols["ruwe"], float)
    rep += ["", "=== PARENT RUWE AT FIXED G (real stars' own RUWE; median and fraction > 1.4) ==="]
    for lo in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
        sb = pc & (pg >= lo) & (pg < lo + 2.0) & np.isfinite(pruwe)
        e_, d_ = sb & pev, sb & ~pev
        if e_.sum() >= 20 and d_.sum() >= 20:
            rep.append(
                f"  G in [{lo:.0f},{lo + 2:.0f}): evolved N={int(e_.sum())} median {np.median(pruwe[e_]):.3f}, "
                f">1.4 {np.mean(pruwe[e_] > 1.4):.4f} | dwarfs N={int(d_.sum())} median {np.median(pruwe[d_]):.3f}, "
                f">1.4 {np.mean(pruwe[d_] > 1.4):.4f}"
            )

    # ---------------- real side ----------------
    real = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(args.real_types or cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    real = real[np.isin(np.asarray(real["nss_solution_type"]).astype(str), types)]
    _, first = np.unique(np.asarray(real["source_id"], np.int64), return_index=True)
    n_dup = len(real) - first.size
    real = real[np.sort(first)]
    in_cols = None
    if args.real_input_columns is not None:
        from darkhunter_pop.proposal_set import load_real_input_columns

        in_cols = load_real_input_columns(args.real_input_columns)
    keep, kcounts = real_comparison_keep(real, gen_cfgs[0], in_cols)
    real = real[keep]
    sid = np.asarray(real["source_id"], np.int64)
    if args.real_bj_dir is not None:
        meta = yaml.safe_load((args.real_bj_dir / "meta.yaml").read_text())
        with h5py.File(args.real_bj_dir / "rows.h5", "r") as h:
            bj = {k: h[k][()] for k in h.keys()}
        order = np.argsort(bj["source_id"])
        pos = np.clip(np.searchsorted(bj["source_id"], sid, sorter=order), 0, order.size - 1)
        matched = bj["source_id"][order[pos]] == sid
        idx = order[pos]

        def bjcol(name: str) -> np.ndarray:
            return np.where(matched, bj[name][idx], np.nan)

        rcmd = cmd_for_rows(
            bjcol("phot_g_mean_mag"), bjcol("bp_rp"), bjcol("l"), bjcol("b"),
            bjcol("r_med_geo"), bjcol("r_lo_geo"), bjcol("r_hi_geo"), cfg, gcfg,
        )
        dist_note = f"Bailer-Jones geometric, matched {int(matched.sum())} ({meta['rows_h5_sha256'][:12]}, {meta['query_date']})"
    else:
        # Fallback (archive join timed out): distance = 1000 / NSS parallax, 16/84 from parallax ± error.
        from astropy.coordinates import SkyCoord
        import astropy.units as au

        def fcol(name: str) -> np.ndarray:
            return np.ma.filled(np.ma.asarray(real[name], float), np.nan)

        plx, eplx = fcol("parallax"), fcol("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            r_med = np.where(plx > 0, 1000.0 / plx, np.nan)
            r_lo = np.where(plx > 0, 1000.0 / (plx + eplx), np.nan)
            r_hi = np.where(plx - eplx > 0, 1000.0 / (plx - eplx), np.nan)
        gal = SkyCoord(ra=fcol("ra") * au.deg, dec=fcol("dec") * au.deg).galactic
        rcmd = cmd_for_rows(
            fcol("g_mag"), fcol("bp_mag") - fcol("rp_mag"), gal.l.deg, gal.b.deg,
            r_med, r_lo, r_hi, cfg, gcfg,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rsnr = plx / eplx
        dist_note = (
            "inverse NSS parallax (Bailer-Jones join timed out on the Gaia archive); real parallax/error "
            f"p5/p50 = {np.nanpercentile(rsnr, 5):.1f}/{np.nanpercentile(rsnr, 50):.1f}"
        )
    rcls = classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, ridge, gcfg.provisional_n_sigma)
    rlogg = np.ma.filled(np.ma.asarray(real["logg_gspphot"], float), np.nan)
    rtype = np.asarray(real["nss_solution_type"]).astype(str)
    rep += [
        "",
        f"real {'+'.join(types)}: {len(real)} after mirror filters {kcounts} and {n_dup} duplicate source_id rows (#221) dropped; "
        f"distance: {dist_note}; "
        f"CMD-classified {int(rcls.classified.sum())}",
        f"real evolved (CMD): {_bfrac(int(rcls.evolved.sum()), int(rcls.classified.sum()))}",
        f"real GSP-Phot log g < 3.6: {_bfrac(int(np.sum(rlogg < 3.6)), int(np.sum(np.isfinite(rlogg))))}",
    ]
    for t in types:
        s = rcls.classified & (rtype == t)
        rep.append(f"  {t}: evolved {_bfrac(int(np.sum(rcls.evolved & s)), int(s.sum()))}")
    for n_sig in (2.0, 3.0, 4.0, 5.0):
        alt = classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, ridge, n_sig)
        rep.append(f"  sensitivity n_sigma={n_sig}: real evolved {_bfrac(int(alt.evolved.sum()), int(rcls.classified.sum()))}")

    # ---------------- mock side ----------------
    prow = np.asarray(truth["parent_row"], np.int64)
    mev = pcls.evolved[prow]
    mcl = pcls.classified[prow]
    am = acc & mcl
    pf, pf_err = _wfrac(mev[am], w[am])
    rep += [
        "",
        f"mock accepted orbits: {int(acc.sum())} (CMD-classified primaries {int(am.sum())}); artifacts "
        + ", ".join(str(a) for a in args.artifact),
        f"mock accepted evolved, raw draws: {_bfrac(int(np.sum(mev & am)), int(am.sum()))}",
        f"mock accepted evolved, MdS17-weighted: {pf:.4f} ± {pf_err:.4f} (ESS accepted {kish_ess(w[am]):.1f}, "
        f"ESS evolved accepted {kish_ess(w[am & mev]):.1f})",
        f"mock old flag (is_giant) among accepted: {_bfrac(int(np.sum(np.asarray(truth['is_giant'], bool) & acc)), int(acc.sum()))}",
        f"mock all draws on evolved primaries (proposal): {np.mean(mev[mcl]):.4f}",
    ]

    # ---------------- figure: CMDs ----------------
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 5.6), sharey=True)
    cc = np.linspace(ridge.colour[0], ridge.colour[-1], 200)
    r_c, s_c, _ = ridge.at(cc)
    from darkhunter_pop import constants as K

    thr_c = r_c - K.TWIN_BRIGHTENING_MAG - gcfg.provisional_n_sigma * s_c
    panels_cmd = [
        ("parent (usable)", pcmd.colour0[pc], pcmd.mg0[pc], pev[pc], None),
        ("real Orbital+AstroSpectroSB1", rcmd.colour0[rcls.classified], rcmd.mg0[rcls.classified], rcls.evolved[rcls.classified], None),
        ("mock accepted (MdS17 weight)", pcmd.colour0[prow[am]], pcmd.mg0[prow[am]], mev[am], w[am]),
    ]
    for ax, (ttl, x, y, ev, ww) in zip(axs, panels_cmd):
        h = ax.hist2d(x, y, bins=[np.linspace(-0.2, 3.2, 120), np.linspace(-4, 13, 140)], weights=ww, cmap="Greys", cmin=1e-30, norm="log")
        ax.plot(cc, r_c, color=series_style(0, style)["color"], lw=2, label="MS ridge (measured)")
        ax.plot(cc, r_c - K.TWIN_BRIGHTENING_MAG, color=series_style(1, style)["color"], lw=2, ls="--", label="twin line")
        ax.plot(cc, thr_c, color=series_style(2, style)["color"], lw=2, ls=":", label=f"evolved cut (n_sigma={gcfg.provisional_n_sigma:g}, sigma_mu=0)")
        frac = (np.sum(ww[ev]) / np.sum(ww)) if ww is not None else np.mean(ev)
        apply_axes_style(ax, style, xlabel=r"(BP$-$RP)$_0$ (mag)", ylabel=r"M$_{G,0}$ (mag)" if ax is axs[0] else None)
        ax.set_title(f"{ttl}\nN={x.size}, evolved fraction {frac:.3f}", fontsize=style.tick_label_fontsize + 1)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(13, -4)
        del h
    axs[0].legend(loc="lower left", fontsize=style.tick_label_fontsize - 1)
    fig.suptitle(f"{args.label}\nDereddened CMD (Combined19 + Babusiaux et al. 2018 law); parent/mock: Bailer-Jones geometric distance; real: {dist_note.split(';')[0]}", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "giants_cmd.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ---------------- six-panel: evolved real vs mock ----------------
    gm = import_gaiamock_mod()
    rp_ev, n_rev = build_elbadry2024_comparison_panels(real[rcls.evolved], nss_solution_types=types, gaiamock=gm)
    rp_dw, n_rdw = build_elbadry2024_comparison_panels(real[rcls.classified & ~rcls.evolved], nss_solution_types=types, gaiamock=gm)
    mv = mock_panel_values(truth, outcome)
    mev_acc = am & mev
    panels, weights = {}, {}
    ks_lines = []
    for name in SIX_PANEL_NAMES:
        panels[name] = {
            "DR3 evolved": rp_ev[name],
            "mock evolved, MdS17 wt": mv[name][mev_acc],
            "DR3 dwarfs": rp_dw[name],
        }
        weights[name] = {"mock evolved, MdS17 wt": w[mev_acc]}
        d, neff, p = weighted_ks(rp_ev[name], mv[name][mev_acc], w[mev_acc])
        ks_lines.append(f"  {name}: D={d:.3f}, n_eff={neff:.1f}, p={p:.3g}")
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    ess_ev = kish_ess(w[mev_acc])
    caption = (
        f"{args.label}. Evolved (subgiant + giant) primaries only, classified identically on both sides by the "
        f"dereddened CMD (docs/MOCK_POPULATION_SPEC.md §10; n_sigma={gcfg.provisional_n_sigma:g}, provisional). "
        f"DR3 evolved = real Orbital+AstroSpectroSB1 with the parent's mirror filters (N={n_rev}); DR3 dwarfs = the "
        f"non-evolved real orbits (N={n_rdw}), for contrast. Mock evolved = accepted gen10+gen11 proposal-set orbits "
        f"(#391, pre-noise-fix, pre-Malmquist) on CMD-evolved parent rows, reweighted to MdS17 at published parameters "
        f"with the current DWARF treatment of giants (TAG10 M1, Janssens M_G for the flux ratio, no Roche-lobe "
        f"truncation): N_acc={int(mev_acc.sum())}, Kish ESS={ess_ev:.1f}. Unit area per series. Weighted KS in giants_report.txt."
    )
    plot_six_panel_grid(
        panels, args.out_dir / "giants_six_panel.png", panel_order=SIX_PANEL_NAMES,
        dpi=int(cfg.diagnostics.figure_dpi),
        title=f"{args.label}\nevolved primaries: DR3 (N={n_rev}) vs MdS17-weighted mock (N_acc={int(mev_acc.sum())}, ESS={ess_ev:.1f})",
        panel_xlabels=PANEL_LABELS, panel_axes=axes_cfg, max_bins=20, density=True, caption=caption,
        style=cfg.plotting, series_weights=weights,
    )
    rep += ["", "=== SIX-PANEL, EVOLVED: weighted KS real vs mock (informational, ESS-limited) ==="] + ks_lines
    rep.append(f"  real evolved rows in panels {n_rev}; real dwarfs {n_rdw}; mock evolved accepted {int(mev_acc.sum())}, ESS {ess_ev:.1f}")
    for name in ("P_orb_days", "eccentricity", "f_m_msun"):
        q_r = np.nanpercentile(rp_ev[name], [10, 50, 90])
        x = mv[name][mev_acc]
        o = np.argsort(x)
        cw = np.cumsum(w[mev_acc][o]) / np.sum(w[mev_acc])
        q_m = np.interp([0.1, 0.5, 0.9], cw, x[o]) if x.size else [np.nan] * 3
        q_d = np.nanpercentile(rp_dw[name], [10, 50, 90])
        rep.append(f"  {name} p10/50/90: DR3 evolved {np.round(q_r, 4).tolist()}, mock evolved {np.round(q_m, 4).tolist()}, DR3 dwarfs {np.round(q_d, 4).tolist()}")

    # ---------------- P vs CMD radius, Roche floor ----------------
    rteff = np.ma.filled(np.ma.asarray(real["teff_gspphot"], float), np.nan)
    r_rad = cmd_radius_rsun(rcmd.mg0, rteff)
    rper = np.ma.filled(np.ma.asarray(real["period"], float), np.nan)
    pteff = np.asarray(cols["teff_gspphot"], float)
    p_rad = cmd_radius_rsun(pcmd.mg0, pteff)
    m_rad = p_rad[prow]
    m_per = np.asarray(truth["period_days"], float)
    m1 = np.asarray(truth["m1_msun"], float)
    m2 = np.asarray(truth["m2_msun"], float)
    m_floor = roche_period_floor_days(m_rad, m1, m2)
    s_m = mev_acc & np.isfinite(m_floor)
    below = s_m & (m_per < m_floor)
    wb = float(np.sum(w[below]) / np.sum(w[s_m])) if np.sum(w[s_m]) > 0 else float("nan")
    # real: floor for an illustrative M1 from TAG10 (data side) and the minimum M2 implied (q=0.1)
    s_r = rcls.evolved & np.isfinite(r_rad) & np.isfinite(rper)
    rep += [
        "",
        "=== PERIOD vs RADIUS (Eggleton 1983 Roche-lobe floor, circular orbit) ===",
        f"CMD radius (Andrae et al. 2018 BC_G, GSP-Phot Teff): real evolved median R1 = {np.nanmedian(r_rad[rcls.evolved]):.2f} Rsun, "
        f"real dwarfs {np.nanmedian(r_rad[rcls.classified & ~rcls.evolved]):.2f}; parent evolved {np.nanmedian(p_rad[pev]):.2f}",
        f"mock accepted evolved draws whose truth P is below their own Roche floor (truth M1, M2, CMD R1): raw "
        f"{int(below.sum())}/{int(s_m.sum())}, MdS17-weighted fraction {wb:.3f}",
    ]
    for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 30), (30, 1e4)):
        s = s_r & (r_rad >= lo) & (r_rad < hi)
        sm = s_m & (m_rad >= lo) & (m_rad < hi)
        mq = np.nan
        if sm.sum():
            o = np.argsort(m_per[sm])
            cw = np.cumsum(w[sm][o]) / np.sum(w[sm])
            mq = float(np.interp(0.05, cw, m_per[sm][o]))
        rep.append(
            f"  R1 in [{lo},{hi}) Rsun: real N={int(s.sum())}, P5={np.nanpercentile(rper[s], 5) if s.sum() else np.nan:.0f} d, "
            f"P_min={np.nanmin(rper[s]) if s.sum() else np.nan:.0f} d | mock acc N={int(sm.sum())}, weighted P5={mq:.0f} d"
        )
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    s_rd = rcls.classified & ~rcls.evolved & np.isfinite(r_rad) & np.isfinite(rper)
    ax.scatter(r_rad[s_rd], rper[s_rd], s=1, alpha=0.15, color="0.6", label=f"DR3 dwarfs (N={int(s_rd.sum())})", rasterized=True)
    ax.scatter(r_rad[s_r], rper[s_r], s=4, alpha=0.6, color=series_style(0, style)["color"], label=f"DR3 evolved (N={int(s_r.sum())})", rasterized=True)
    ax.scatter(m_rad[s_m], m_per[s_m], s=10, marker="x", color=series_style(1, style)["color"], label=f"mock accepted evolved, truth P (N={int(s_m.sum())})")
    rr = np.logspace(-0.5, 2.3, 100)
    for (a1, a2), ls in zip(((1.2, 0.12), (1.2, 1.2)), ("-", "--")):
        ax.plot(rr, roche_period_floor_days(rr, a1, a2), color="k", ls=ls, lw=2, label=f"Roche floor M1={a1}, M2={a2}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 200)
    ax.set_ylim(5, 5000)
    apply_axes_style(ax, style, xlabel=r"CMD radius R$_1$ (R$_\odot$)", ylabel="orbital period (day)")
    ax.legend(loc="lower right", fontsize=style.tick_label_fontsize - 1)
    ax.set_title(f"{args.label}\nperiod vs primary radius; Eggleton (1983) Roche-lobe floor (circular)", fontsize=style.tick_label_fontsize + 1)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "giants_period_radius.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ---------------- companion flux ratio and Malmquist check ----------------
    lf_dw = np.asarray(truth["log10_flux_ratio"], float)
    mg2 = janssens_absolute_g(m2)
    mg1_obs = pcmd.mg0[prow]
    lf_gi = evolved_log10_flux_ratio(m2, mg1_obs)
    ev_draw = mev & np.isfinite(lf_gi) & (w > 0)
    rel_dw = -0.4 * (mg2 - janssens_absolute_g(m1))
    with np.errstate(invalid="ignore"):
        delta_mag = 2.5 * np.log10(1.0 + 10.0**lf_gi)

    def wq(x: np.ndarray, ww: np.ndarray, qs: tuple[float, ...]) -> list[float]:
        o = np.argsort(x)
        cw = np.cumsum(ww[o]) / np.sum(ww[o])
        return [float(np.interp(q, cw, x[o])) for q in qs]

    rep += [
        "",
        "=== COMPANION FLUX RATIO FOR EVOLVED PRIMARIES (all draws with MdS17 weight > 0) ===",
        f"draws on evolved primaries with weight > 0: {int(ev_draw.sum())}",
        "log10 f, dwarf relation for both stars (current target), weighted p10/50/90: "
        + str(np.round(wq(rel_dw[ev_draw], w[ev_draw], (0.1, 0.5, 0.9)), 3).tolist()),
        "log10 f, evolved_log10_flux_ratio (companion share of the observed system light, spec §10.4), weighted p10/50/90: "
        + str(np.round(wq(lf_gi[ev_draw], w[ev_draw], (0.1, 0.5, 0.9)), 3).tolist()),
        "companion brightening delta = 2.5 log10(1 + f_obs) (mag), weighted p50/90/99: "
        + str(np.round(wq(delta_mag[ev_draw], w[ev_draw], (0.5, 0.9, 0.99)), 4).tolist()),
        "stored proposal log10 f on accepted evolved draws, p10/50/90 (raw): "
        + str(np.round(np.nanpercentile(lf_dw[mev_acc], [10, 50, 90]), 3).tolist() if mev_acc.any() else []),
        "dwarf-relation luminosity excess of evolved parent rows, M_G0 - M_G^J(TAG10 M1), p10/50/90: "
        + str(np.round(np.nanpercentile((pcmd.mg0 - janssens_absolute_g(parent.m1_msun))[pev], [10, 50, 90]), 2).tolist()),
    ]

    # ---------------- TAG10 vs FLAME ----------------
    if args.flame_snapshot is not None:
        fl = Table.read(args.flame_snapshot, format="ascii.ecsv")
        fsid = np.asarray(fl["source_id"], np.int64)
        fmass = np.ma.filled(np.ma.asarray(fl["mass_flame"], float), np.nan)
        _, fi = np.unique(fsid, return_index=True)
        fsid, fmass = fsid[fi], fmass[fi]
        rsid = np.asarray(real["source_id"], np.int64)
        p2 = np.clip(np.searchsorted(fsid, rsid), 0, fsid.size - 1)
        fm = np.where(fsid[p2] == rsid, fmass[p2], np.nan)
        from darkhunter_pop.proposal_set import tag10_m1_for_rows

        rc = {c: np.ma.filled(np.ma.asarray(real[c], float), np.nan) for c in real.colnames if c in (
            "teff_msc1", "teff_msc1_upper", "teff_msc1_lower", "logg_msc1", "logg_msc1_upper", "logg_msc1_lower",
            "mh_msc", "mh_msc_upper", "mh_msc_lower", "teff_gspphot", "teff_gspphot_upper", "teff_gspphot_lower",
            "logg_gspphot", "logg_gspphot_upper", "logg_gspphot_lower", "mh_gspphot", "mh_gspphot_upper", "mh_gspphot_lower")}
        rc["source_id"] = rsid
        tm1, tsrc, tlogg = tag10_m1_for_rows(rc, cfg)
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        for lab, s, k in (("DR3 dwarfs", rcls.classified & ~rcls.evolved, 0), ("DR3 evolved", rcls.evolved, 1)):
            ok = s & np.isfinite(fm) & np.isfinite(tm1)
            st = series_style(k, style)
            ax.scatter(fm[ok], tm1[ok], s=2 if k == 0 else 5, alpha=0.2 if k == 0 else 0.6, color=st["color"], label=f"{lab} (N={int(ok.sum())})", rasterized=True)
            ratio = tm1[ok] / fm[ok]
            rep.append(
                f"TAG10 M1 / FLAME mass, {lab}: N={int(ok.sum())} (FLAME available for {np.mean(np.isfinite(fm[s])):.3f} of the class), "
                f"median {np.median(ratio):.3f}, p16/p84 {np.percentile(ratio, 16):.3f}/{np.percentile(ratio, 84):.3f}; "
                f"TAG10 atmosphere log g median {np.nanmedian(tlogg[ok]):.2f}; MSC share {np.mean(tsrc[ok] == 'MSC'):.3f}"
            )
        ax.plot([0.3, 4], [0.3, 4], color="k", lw=2)
        ax.set_xlim(0.3, 4)
        ax.set_ylim(0.3, 4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel=r"FLAME mass (M$_\odot$)", ylabel=r"TAG10 M$_1$ (M$_\odot$)")
        ax.legend(loc="upper left", fontsize=style.tick_label_fontsize - 1)
        ax.set_title(f"{args.label}\nreal NSS orbits: data-side TAG10 M1 vs Gaia FLAME mass", fontsize=style.tick_label_fontsize + 1)
        fig.tight_layout()
        save_figure(fig, args.out_dir / "giants_m1_flame.png", dpi=int(cfg.diagnostics.figure_dpi))

    (args.out_dir / "giants_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
