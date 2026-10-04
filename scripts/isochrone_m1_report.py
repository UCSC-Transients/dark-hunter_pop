#!/usr/bin/env python3
"""MIST isochrone M1 and 2-D ridge calibration on real data (#418; spec §11.2, §11.6).

Measures, identically on the ``gaia_source`` parent and the real DR3 Orbital + AstroSpectroSB1
orbits:

* isochrone M1 (``darkhunter_pop.isochrone_mass``) against TAG10 (parent) and against Gaia FLAME
  masses, split into CMD dwarfs and CMD-evolved stars, beside the TAG10 / FLAME ratio of #417;
* the single-star ridge re-measured on RUWE < 1.4 rows (one ridge, shared by the giants
  classifier and the 2-D Malmquist weight) and the evolved fractions it gives;
* the MP-Q25 replacement: ΔM_2D = M_G0 − R(C0) on RUWE < 1.4 non-evolved parent rows, binned
  by distance and by M1 (the #414 tables), with the old 1-D ΔM beside it.

Writes ``isochrone_m1_report.txt``, ``isochrone_m1_report.json`` and figures to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import malmquist_cmd as mc
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style
from darkhunter_pop.proposal_set import (
    janssens_absolute_g,
    load_parent_snapshot,
    load_proposal_set_fragment,
    real_comparison_keep,
    tag10_m1_for_rows,
)

DIST_BINS_KPC = ((0.0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 50.0))
M1_BINS = ((0.0, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 100.0))
AP_COLS = (
    "teff_msc1", "teff_msc1_upper", "teff_msc1_lower", "logg_msc1", "logg_msc1_upper", "logg_msc1_lower",
    "mh_msc", "mh_msc_upper", "mh_msc_lower", "teff_gspphot", "teff_gspphot_upper", "teff_gspphot_lower",
    "logg_gspphot", "logg_gspphot_upper", "logg_gspphot_lower", "mh_gspphot", "mh_gspphot_upper", "mh_gspphot_lower",
)


def _plain_log_ticks(ax: Any, axes: str = "xy") -> None:
    """Plain-number tick labels on log axes (0.5, 1, 2 rather than 5x10^-1)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for name in axes:
        axis = ax.xaxis if name == "x" else ax.yaxis
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())


def _robust(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def _mode(x: np.ndarray, step: float = 0.02, smooth: int = 7) -> float:
    """Smoothed histogram mode (same estimator as the ridge)."""
    edges = np.arange(np.percentile(x, 0.5) - step, np.percentile(x, 99.5) + step, step)
    h, _ = np.histogram(x, bins=edges)
    k = np.convolve(h, np.ones(smooth) / smooth, mode="same")
    return float(edges[int(np.argmax(k))] + 0.5 * step)


def _ratio_stats(r: np.ndarray) -> dict[str, float]:
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return {"n": 0}
    lr = np.log10(r)
    return {
        "n": int(r.size),
        "median_ratio": float(np.median(r)),
        "p16": float(np.percentile(r, 16)),
        "p84": float(np.percentile(r, 84)),
        "bias_dex_median": float(np.median(lr)),
        "scatter_dex_robust": float(1.4826 * np.median(np.abs(lr - np.median(lr)))),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--proposal-fragment", type=Path, default=Path("config/population/proposal_set_decided_full.yaml"))
    ap.add_argument("--real-snapshot", type=Path, required=True, help="uncut NSS snapshot query.ecsv")
    ap.add_argument("--flame-snapshot", type=Path, required=True, help="flame_enrichment query.ecsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True, help="scratch cache for the real-side TAG10 loop")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--host-profile", default="laptop")
    ap.add_argument("--giants-config", type=Path, default=Path("config/population/giants.yaml"))
    args = ap.parse_args(argv)

    from astropy.coordinates import SkyCoord
    import astropy.units as au
    from astropy.table import Table

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, host_profile=args.host_profile)
    gcfg = giants.load_giants_config(args.giants_config)
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    out: dict[str, Any] = {"isochrone_config_key": im.config_key(cfg.isochrone_mass)}
    rep: list[str] = ["#418 isochrone M1 and 2-D ridge calibration (docs/MOCK_POPULATION_SPEC.md §11)", ""]

    # ---------------- parent: TAG10 and isochrone modes ----------------
    frag = load_proposal_set_fragment(args.proposal_fragment)
    prop_t = frag.proposal
    prop_i = prop_t.model_copy(update={"m1": "isochrone_mist_drop_unresolved"})
    par_t = load_parent_snapshot(args.parent_dir, cfg, prop_t)
    par_i = load_parent_snapshot(args.parent_dir, cfg, prop_i, giants_config_path=args.giants_config)
    cols = par_i.columns
    cmd = par_i.cmd
    iso = par_i.isochrone
    assert cmd is not None and iso is not None
    rd = par_i.meta["ms_ridge"]
    ridge = giants.MSRidge(colour=np.asarray(rd["colour"]), mag=np.asarray(rd["mag"]), sigma=np.asarray(rd["sigma"]),
                           n_rows=np.asarray(rd["n_rows"], np.int64))
    rt = mc.RidgeTables.from_ridge(ridge)
    u = np.asarray(par_i.usable, bool)
    ut = np.asarray(par_t.usable, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.asarray(cols["parallax"], float) / np.asarray(cols["parallax_error"], float)
    ruwe = np.asarray(cols["ruwe"], float)
    # old (#417) ridge: no RUWE cut, same rows otherwise
    ridge_old = giants.fit_ms_ridge(np.where(u, cmd["mg0"], np.nan), cmd["colour0"], snr,
                                    gcfg.ridge.model_copy(update={"ruwe_max": None}))
    reasons, counts = np.unique(np.asarray(iso["reason"]), return_counts=True)
    m1i, m1t = par_i.m1_msun, par_t.m1_msun
    both = u & ut
    rep += [
        "=== PARENT M1 ===",
        f"isochrone reasons (all {m1i.size} rows): " + ", ".join(f"{a}={b}" for a, b in zip(reasons, counts)),
        f"usable rows: isochrone mode {int(u.sum())}, TAG10 mode {int(ut.sum())}; both {int(both.sum())}",
        "M1 percentiles 1/5/25/50/75/95/99 (usable): isochrone "
        + " ".join(f"{x:.3f}" for x in np.percentile(m1i[u], [1, 5, 25, 50, 75, 95, 99]))
        + " | TAG10 " + " ".join(f"{x:.3f}" for x in np.percentile(m1t[ut], [1, 5, 25, 50, 75, 95, 99])),
        f"isochrone sigma_logM1 median {np.nanmedian(np.asarray(iso['log_m1_sigma'])[u]):.3f} dex, "
        f"sigma_M1/M1 > 0.25 fraction {np.mean((np.asarray(iso['m1_sigma']) / m1i)[u] > 0.25):.4f}",
        f"fraction of usable rows at the TAG10 floor (M1 < 0.6): TAG10 {np.mean(m1t[ut] < 0.6):.4f}, isochrone {np.mean(m1i[u] < 0.6):.4f}",
    ]
    out["parent"] = {
        "reasons": {str(a): int(b) for a, b in zip(reasons, counts)},
        "usable_iso": int(u.sum()), "usable_tag10": int(ut.sum()),
        "m1_pct_iso": np.percentile(m1i[u], [1, 5, 25, 50, 75, 95, 99]).tolist(),
        "m1_pct_tag10": np.percentile(m1t[ut], [1, 5, 25, 50, 75, 95, 99]).tolist(),
    }

    # ---------------- ridge and evolved fractions ----------------
    rep += ["", "=== SHARED MS RIDGE (usable, parallax/error >= "
            f"{gcfg.ridge.min_parallax_over_error}, RUWE < {gcfg.ridge.ruwe_max}) vs #417 ridge (no RUWE cut) ===",
            "  colour0  R_new   sigma_new  n_new | R_old  sigma_old  n_old"]
    for k, c in enumerate(ridge.colour):
        j = np.flatnonzero(np.isclose(ridge_old.colour, c))
        o = f"{ridge_old.mag[j[0]]:.3f} {ridge_old.sigma[j[0]]:.3f} {ridge_old.n_rows[j[0]]}" if j.size else "-"
        rep.append(f"  {c:.2f}  {ridge.mag[k]:.3f}  {ridge.sigma[k]:.3f}  {ridge.n_rows[k]} | {o}")
    cls_new = giants.classify_evolved(cmd["mg0"], cmd["colour0"], cmd["sigma_mu"], ridge, gcfg.provisional_n_sigma)
    cls_old = giants.classify_evolved(cmd["mg0"], cmd["colour0"], cmd["sigma_mu"], ridge_old, gcfg.provisional_n_sigma)
    pc = u & cls_new.classified
    pev = u & cls_new.evolved
    pev_old = u & cls_old.evolved
    p_evo_iso = np.asarray(iso["p_evolved"], float)
    rep += [
        f"parent evolved (CMD, n_sigma={gcfg.provisional_n_sigma}): new ridge {pev.sum()}/{pc.sum()} = {pev.sum() / pc.sum():.4f}; "
        f"#417 ridge {pev_old.sum()}/{int((u & cls_old.classified).sum())} = {pev_old.sum() / (u & cls_old.classified).sum():.4f}",
        f"isochrone P(evolved) > 0.5 among usable classified: {np.mean(p_evo_iso[pc] > 0.5):.4f}; agreement with CMD flag "
        f"{np.mean((p_evo_iso[pc] > 0.5) == cls_new.evolved[pc]):.4f}",
    ]
    out["ridge"] = {"colour": ridge.colour.tolist(), "mag": ridge.mag.tolist(), "sigma": ridge.sigma.tolist(),
                    "n_rows": ridge.n_rows.tolist(), "old_mag": ridge_old.mag.tolist(), "old_colour": ridge_old.colour.tolist(),
                    "old_sigma": ridge_old.sigma.tolist()}
    out["parent"]["evolved_fraction_new"] = float(pev.sum() / pc.sum())
    out["parent"]["evolved_fraction_417"] = float(pev_old.sum() / (u & cls_old.classified).sum())

    # ---------------- MP-Q25 replacement: ΔM_2D tables ----------------
    dm2 = mc.ridge_residual(cmd["colour0"], cmd["mg0"], rt)
    d_kpc = np.asarray(cols["r_med_geo"], float) / 1000.0
    sel = u & (ruwe < 1.4) & cls_new.classified & ~cls_new.evolved & np.isfinite(dm2)
    # old 1-D residuals: Janssens at TAG10 M1 (the #414 quantity, A_G from the same Combined19
    # + Babusiaux A_G here) and at isochrone M1
    dm1_t = cmd["mg0"] - janssens_absolute_g(m1t)
    dm1_i = cmd["mg0"] - janssens_absolute_g(m1i)
    med, rs = _robust(dm2[sel])
    mode = _mode(dm2[sel])
    # bootstrap error of the mode (statistical precision of the zero point)
    rng = np.random.default_rng(418)
    xs = dm2[sel]
    boots = [_mode(xs[rng.integers(0, xs.size, xs.size)]) for _ in range(50)]
    rep += ["", "=== MP-Q25 REPLACEMENT: dM_2D = M_G0 - R(C0) on usable RUWE<1.4 non-evolved classified parent rows ===",
            f"N = {int(sel.sum())}; mode {mode:+.3f} +/- {np.std(boots):.3f} (bootstrap) mag; median {med:+.3f}; robust sigma {rs:.3f}",
            "  (the ridge is the per-colour mode, so the pooled mode is ~0 by construction; the TEST is the trend)",
            "  by distance (kpc):     n      mode    median  robust_sigma | old 1-D TAG10 median | 1-D isochrone median"]
    tab_d = []
    for lo, hi in DIST_BINS_KPC:
        b = sel & (d_kpc >= lo) & (d_kpc < hi) & np.isfinite(dm1_t)
        if b.sum() < 50:
            continue
        mo = _mode(dm2[b])
        m_, s_ = _robust(dm2[b])
        row = {"lo": lo, "hi": hi, "n": int(b.sum()), "mode": mo, "median": m_, "robust_sigma": s_,
               "old_1d_tag10_median": float(np.nanmedian(dm1_t[b])), "iso_1d_median": float(np.nanmedian(dm1_i[b]))}
        tab_d.append(row)
        rep.append(f"  [{lo:.1f},{hi:.1f}) {row['n']:7d} {mo:+.3f} {m_:+.3f} {s_:.3f} | {row['old_1d_tag10_median']:+.3f} | {row['iso_1d_median']:+.3f}")
    rep.append("  by isochrone M1 (Msun):  n   mode   median  robust_sigma | by TAG10 M1: n  mode  median")
    tab_m = []
    for lo, hi in M1_BINS:
        b = sel & (m1i >= lo) & (m1i < hi)
        bt = sel & (m1t >= lo) & (m1t < hi)
        if b.sum() < 50:
            continue
        m_, s_ = _robust(dm2[b])
        row = {"lo": lo, "hi": hi, "n": int(b.sum()), "mode": _mode(dm2[b]), "median": m_, "robust_sigma": s_,
               "n_tag10_bin": int(bt.sum()), "mode_tag10_bin": _mode(dm2[bt]) if bt.sum() > 50 else float("nan"),
               "median_tag10_bin": float(np.median(dm2[bt])) if bt.sum() else float("nan")}
        tab_m.append(row)
        rep.append(f"  [{lo:.1f},{hi:.1f}) {row['n']:7d} {row['mode']:+.3f} {m_:+.3f} {s_:.3f} | {row['n_tag10_bin']} {row['mode_tag10_bin']:+.3f} {row['median_tag10_bin']:+.3f}")
    out["calibration"] = {"n": int(sel.sum()), "mode": mode, "mode_err": float(np.std(boots)), "median": med,
                          "robust_sigma": rs, "by_distance": tab_d, "by_m1": tab_m}

    # ---------------- real orbits: isochrone M1 vs FLAME and TAG10 vs FLAME ----------------
    real = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    real = real[np.isin(np.asarray(real["nss_solution_type"]).astype(str), types)]
    _, first = np.unique(np.asarray(real["source_id"], np.int64), return_index=True)
    real = real[np.sort(first)]
    keep, kcounts = real_comparison_keep(real, prop_t)
    real = real[keep]

    def fcol(name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(real[name], float), np.nan)

    plx, eplx = fcol("parallax"), fcol("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        r_med = np.where(plx > 0, 1000.0 / plx, np.nan)
        r_lo = np.where(plx > 0, 1000.0 / (plx + eplx), np.nan)
        r_hi = np.where(plx - eplx > 0, 1000.0 / (plx - eplx), np.nan)
    gal = SkyCoord(ra=fcol("ra") * au.deg, dec=fcol("dec") * au.deg).galactic
    rcmd = giants.cmd_for_rows(fcol("g_mag"), fcol("bp_mag") - fcol("rp_mag"), gal.l.deg, gal.b.deg,
                               r_med, r_lo, r_hi, cfg, gcfg)
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    rpost = model.fit(rcmd.colour0, rcmd.mg0, sigma_mu=rcmd.sigma_mu, ebv=rcmd.ebv, a_g=rcmd.a_g,
                      e_bp_rp=(fcol("bp_mag") - fcol("rp_mag")) - rcmd.colour0)
    rm1 = rpost.point(cfg.isochrone_mass.provisional_point_estimate)
    rcls = giants.classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, ridge, gcfg.provisional_n_sigma)
    rsid = np.asarray(real["source_id"], np.int64)
    tcache = args.cache_dir / "real_tag10_m1.npz"
    if tcache.exists() and np.array_equal(np.load(tcache)["source_id"], rsid):
        tm1 = np.load(tcache)["m1"]
    else:
        rc = {c: fcol(c) for c in AP_COLS if c in real.colnames}
        rc["source_id"] = rsid
        tm1, _, _ = tag10_m1_for_rows(rc, cfg)
        np.savez(tcache, source_id=rsid, m1=tm1)
    fl = Table.read(args.flame_snapshot, format="ascii.ecsv")
    fsid = np.asarray(fl["source_id"], np.int64)
    fmass = np.ma.filled(np.ma.asarray(fl["mass_flame"], float), np.nan)
    _, fi = np.unique(fsid, return_index=True)
    fsid, fmass = fsid[fi], fmass[fi]
    p2 = np.clip(np.searchsorted(fsid, rsid), 0, fsid.size - 1)
    fm = np.where(fsid[p2] == rsid, fmass[p2], np.nan)
    rreasons, rcounts = np.unique(rpost.reason, return_counts=True)
    rtype = np.asarray(real["nss_solution_type"]).astype(str)
    rep += ["", "=== REAL ORBITS (Orbital + AstroSpectroSB1, mirror filters; distance = inverse NSS parallax, MP-Q28g) ===",
            f"rows {len(real)} after filters {kcounts}; isochrone reasons: " + ", ".join(f"{a}={b}" for a, b in zip(rreasons, rcounts)),
            f"real evolved (CMD, shared ridge): {int(rcls.evolved.sum())}/{int(rcls.classified.sum())} = {rcls.evolved.sum() / rcls.classified.sum():.4f}; "
            f"Orbital only {np.sum(rcls.evolved & (rtype == 'Orbital')) / np.sum(rcls.classified & (rtype == 'Orbital')):.4f}"]
    flame_stats: dict[str, Any] = {}
    for lab, s in (("dwarfs", rcls.classified & ~rcls.evolved), ("evolved", rcls.evolved)):
        ok = s & np.isfinite(fm) & np.isfinite(rm1) & np.isfinite(tm1)
        st_i = _ratio_stats(rm1[ok] / fm[ok])
        st_t = _ratio_stats(tm1[ok] / fm[ok])
        flame_stats[lab] = {"isochrone_over_flame": st_i, "tag10_over_flame": st_t,
                            "flame_available_fraction": float(np.mean(np.isfinite(fm[s])))}
        rep.append(
            f"  {lab}: N={st_i['n']} (FLAME for {flame_stats[lab]['flame_available_fraction']:.3f}); isochrone/FLAME median {st_i['median_ratio']:.3f} "
            f"(p16/p84 {st_i['p16']:.3f}/{st_i['p84']:.3f}; bias {st_i['bias_dex_median']:+.3f} dex, robust scatter {st_i['scatter_dex_robust']:.3f} dex) | "
            f"TAG10/FLAME median {st_t['median_ratio']:.3f} (p16/p84 {st_t['p16']:.3f}/{st_t['p84']:.3f}; scatter {st_t['scatter_dex_robust']:.3f} dex)"
        )
        for lo, hi in ((0.0, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 10.0)):
            b = ok & (fm >= lo) & (fm < hi)
            if b.sum() >= 30:
                sb = _ratio_stats(rm1[b] / fm[b])
                rep.append(f"     FLAME M in [{lo},{hi}): N={sb['n']} iso/FLAME median {sb['median_ratio']:.3f}, scatter {sb['scatter_dex_robust']:.3f} dex")
    out["real"] = {"rows": int(len(real)), "reasons": {str(a): int(b) for a, b in zip(rreasons, rcounts)},
                   "evolved_fraction": float(rcls.evolved.sum() / rcls.classified.sum()), "flame": flame_stats}

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))
    for ax, (lab, s) in zip(axes, (("CMD dwarfs", rcls.classified & ~rcls.evolved), ("CMD evolved", rcls.evolved))):
        ok = s & np.isfinite(fm) & np.isfinite(rm1) & np.isfinite(tm1)
        ax.scatter(fm[ok], tm1[ok], s=2, alpha=0.15, color=series_style(1, style)["color"], label=f"TAG10 (N={int(ok.sum())})", rasterized=True)
        ax.scatter(fm[ok], rm1[ok], s=2, alpha=0.15, color=series_style(0, style)["color"], label="MIST isochrone", rasterized=True)
        lim = (0.3, 4.0)
        ax.plot(lim, lim, color="k", lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(0.2, 5.0)
        ax.set_xlabel(r"FLAME mass (M$_\odot$)")
        ax.set_ylabel(r"M1 (M$_\odot$)")
        ax.set_title(f"DR3 orbits, {lab}")
        ax.legend(loc="upper left", markerscale=6)
        apply_axes_style(ax, style)
        _plain_log_ticks(ax)
    save_figure(fig, args.out_dir / "m1_vs_flame.png", dpi=int(cfg.diagnostics.figure_dpi))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for k, (lab, x) in enumerate((("TAG10 M1 (#414)", dm1_t), ("2-D ridge residual (#418)", dm2))):
        meds = [np.nanmedian(x[sel & (d_kpc >= lo) & (d_kpc < hi)]) if np.sum(sel & (d_kpc >= lo) & (d_kpc < hi)) > 50 else np.nan for lo, hi in DIST_BINS_KPC]
        cen = [0.5 * (lo + min(hi, 10.0)) for lo, hi in DIST_BINS_KPC]
        axes[0].plot(cen, meds, marker="o", lw=2.5, color=series_style(k, style)["color"], label=lab)
    axes[0].axhline(0.0, color="k", lw=1)
    axes[0].set_xscale("log")
    _plain_log_ticks(axes[0], "x")
    axes[0].set_xlabel("Bailer-Jones distance (kpc)")
    axes[0].set_ylabel(r"median $\Delta M$ (mag)")
    axes[0].set_title("RUWE < 1.4 non-evolved parent rows")
    axes[0].legend()
    apply_axes_style(axes[0], style)
    h_edges = np.linspace(-4, 3, 141)
    axes[1].hist(dm2[sel], bins=h_edges, histtype="step", lw=2.5, color=series_style(1, style)["color"], label="2-D ridge residual")
    out_frac = float(np.mean((dm1_t[sel] < h_edges[0]) | (dm1_t[sel] > h_edges[-1])))
    axes[1].hist(dm1_t[sel], bins=h_edges, histtype="step", lw=2.5, color=series_style(0, style)["color"],
                 label=f"1-D, TAG10 M1 ({100 * out_frac:.0f}% off scale)")
    axes[1].set_xlabel(r"$\Delta M$ (mag; < 0 over-luminous)")
    axes[1].set_ylabel("rows")
    axes[1].legend()
    apply_axes_style(axes[1], style)
    save_figure(fig, args.out_dir / "ridge_calibration.png", dpi=int(cfg.diagnostics.figure_dpi))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    s = u & np.isfinite(cmd["mg0"]) & (ruwe < 1.4) & (snr >= gcfg.ridge.min_parallax_over_error)
    ax.scatter(cmd["colour0"][s], cmd["mg0"][s], s=1, alpha=0.1, color="0.5", rasterized=True, label="parent, RUWE<1.4, $\\varpi/\\sigma_\\varpi\\geq10$")
    ax.plot(ridge.colour, ridge.mag, lw=2.5, color=series_style(0, style)["color"], label="ridge (RUWE<1.4)")
    ax.plot(ridge_old.colour, ridge_old.mag, lw=2.0, ls="--", color=series_style(1, style)["color"], label="#417 ridge (all RUWE)")
    ax.plot(ridge.colour, ridge.mag - 0.753, lw=1.5, ls=":", color="k", label="twin line")
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(13, -3)
    ax.set_xlabel(r"(BP$-$RP)$_0$ (mag)")
    ax.set_ylabel(r"M$_{G,0}$ (mag)")
    ax.legend(loc="lower left", markerscale=6)
    apply_axes_style(ax, style)
    save_figure(fig, args.out_dir / "parent_cmd_ridge.png", dpi=int(cfg.diagnostics.figure_dpi))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    e = np.geomspace(0.08, 10, 80)
    ax.hist(m1t[ut], bins=e, histtype="step", lw=2.5, color=series_style(1, style)["color"], label="TAG10 + Santos")
    ax.hist(m1i[u], bins=e, histtype="step", lw=2.5, color=series_style(0, style)["color"], label="MIST isochrone")
    ax.set_xscale("log")
    _plain_log_ticks(ax, "x")
    ax.set_xlabel(r"parent M1 (M$_\odot$)")
    ax.set_ylabel("rows")
    ax.legend()
    apply_axes_style(ax, style)
    save_figure(fig, args.out_dir / "parent_m1_hist.png", dpi=int(cfg.diagnostics.figure_dpi))
    plt.close(fig)

    (args.out_dir / "isochrone_m1_report.txt").write_text("\n".join(rep) + "\n")
    (args.out_dir / "isochrone_m1_report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
