#!/usr/bin/env python3
"""Characterize the real high-f_m DR3 orbits vs distance (#391; analysis only, no draws).

Real sample: DR3 Orbital + AstroSpectroSB1 with the #391 mirror filters (167,911 rows). Parallax
is zero-point corrected by the median (Bailer-Jones r_med_geo truth − observed parallax) of the
parent's high-S/N stars per G bin (+22 to +37 µas, docs/gate391/distance_count). Per orbit:

* photocentre a0 (Thiele-Innes → Campbell, gaiamock), f_m = (a0/ϖ)³ / P_yr²;
* M1: the MIST isochrone posterior mean on the dereddened CMD (``parent_cmd_isochrone``, the
  same machinery as the mock parent; distances from the corrected parallax);
* Shahaf et al. (2019) AMRF A = (a0/ϖ) M1^(-1/3) P_yr^(-2/3) and class I / II / III, with the
  class boundaries computed for each M1 from the Janssens et al. (2022) main-sequence G-band
  mass–luminosity relation: class I below the largest AMRF of a single luminous MS companion
  (q ≤ 1), class II below the largest AMRF of an MS-pair (two equal MS stars, total q ≤ 2),
  class III above both, i.e. no luminous companion configuration can produce it;
* dark-companion M2 from the mass function (``get_companion_mass_from_mass_function``, f = 0);
  "beyond WD limit" when M2_dark > ``--wd-limit`` (1.5 Msun by default);
* indicators: parallax S/N, significance (a0/σ_a0), goodness_of_fit (F2), RUWE, P relative to
  the 1038 d DR3 baseline, eccentricity, ipd_frac_multi_peak, ipd_gof_harmonic_amplitude;
* P(spurious | x) from ``spuriousness_model`` (refit from its labelled fixtures) on the
  covariates it retains, and the labelled-table cross-match (good / spurious / unknown).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASELINE_DAYS = 1038.0
DBIN = np.array([0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 3.0])


def amrf_class_limits(m1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A_MS_max, A_triple_max) per M1 from the Janssens MS relation (Shahaf et al. 2019 Eq. 4)."""
    from darkhunter_pop.proposal_set import janssens_absolute_g

    q = np.linspace(0.02, 1.0, 400)
    out_ms = np.full(m1.size, np.nan)
    out_tr = np.full(m1.size, np.nan)
    grid = np.geomspace(0.2, 10.0, 120)
    a_ms, a_tr = [], []
    for m in grid:
        mg1 = janssens_absolute_g(m)
        s_single = 10 ** (-0.4 * (janssens_absolute_g(q * m) - mg1))
        a_single = (1 + q) ** (1 / 3) * (q / (1 + q) - s_single / (1 + s_single))
        q2 = 2 * q  # inner pair of two equal MS stars of q*M1 each
        s_pair = 2 * 10 ** (-0.4 * (janssens_absolute_g(q * m) - mg1))
        a_pair = (1 + q2) ** (1 / 3) * (q2 / (1 + q2) - s_pair / (1 + s_pair))
        a_ms.append(np.nanmax(a_single))
        a_tr.append(np.nanmax(np.concatenate([a_single, a_pair])))
    lm = np.log10(np.clip(m1, grid[0], grid[-1]))
    out_ms = np.interp(lm, np.log10(grid), a_ms)
    out_tr = np.interp(lm, np.log10(grid), a_tr)
    return out_ms, out_tr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--enrichment", type=Path, required=True, help="nss_enrichment/query.ecsv (significance)")
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--wd-limit", type=float, default=1.5)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop import spuriousness_model as sm
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
    from darkhunter_pop.giants import load_giants_config
    from darkhunter_pop.physics_utils import astrometric_mass_function
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config()
    gm = import_gaiamock_mod()
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- zero point per G from the parent (high S/N) ----
    pc = parent.columns
    pu = parent.usable
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = []
    for lo, hi in zip(gedges[:-1], gedges[1:]):
        s = pu & (pg >= lo) & (pg < hi) & (snr_p > 20)
        zp.append(float(np.median(dzp[s])))
    zp = np.array(zp)

    # ---- real sample ----
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    inp = ps.load_real_input_columns(args.real_input_columns)
    keep, _ = ps.real_comparison_keep(rt, prop, inp)
    rt = rt[keep]
    n = len(rt)

    def c(name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(rt[name], float), np.nan)

    sid = np.asarray(rt["source_id"], np.int64)
    stype = np.asarray(rt["nss_solution_type"]).astype(str)
    g = c("g_mag")
    plx_obs, eplx = c("parallax"), c("parallax_error")
    zpi = np.clip(np.digitize(g, gedges) - 1, 0, zp.size - 1)
    plx = plx_obs + zp[zpi]
    P = c("period")
    ecc = c("eccentricity")
    gof = c("goodness_of_fit")
    ruwe = c("ruwe")
    a0, _, _, _ = gm.get_Campbell_elements(c("A"), c("B"), c("F"), c("G"))
    a0 = np.asarray(a0, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fm = astrometric_mass_function(a0, plx, P)
        d = 1.0 / plx
        a_au = a0 / plx
    hi = fm > 0.1

    # joins: input columns (ipd), enrichment (significance)
    def join(ref_sid: np.ndarray, col: np.ndarray) -> np.ndarray:
        o = np.argsort(ref_sid)
        pos = np.clip(np.searchsorted(ref_sid, sid, sorter=o), 0, o.size - 1)
        hit = ref_sid[o[pos]] == sid
        return np.where(hit, np.asarray(col, float)[o[pos]], np.nan)

    ipd_mp = join(np.asarray(inp["source_id"], np.int64), inp["ipd_frac_multi_peak"])
    ipd_gof = join(np.asarray(inp["source_id"], np.int64), inp["ipd_gof_harmonic_amplitude"])
    en = Table.read(args.enrichment, format="ascii.ecsv")
    if "SOURCE_ID" in en.colnames and "source_id" not in en.colnames:
        en.rename_column("SOURCE_ID", "source_id")
    en = en[np.isin(np.asarray(en["nss_solution_type"]).astype(str), types)]
    key_en = np.char.add(np.asarray(en["source_id"]).astype(str), np.asarray(en["nss_solution_type"]).astype(str))
    key_rt = np.char.add(sid.astype(str), stype)
    o = np.argsort(key_en)
    pos = np.clip(np.searchsorted(key_en, key_rt, sorter=o), 0, o.size - 1)
    hit = key_en[o[pos]] == key_rt
    sig = np.where(hit, np.ma.filled(np.ma.asarray(en["significance"], float), np.nan)[o[pos]], np.nan)

    # ---- M1: MIST isochrone posterior on the dereddened CMD (as the parent) ----
    cpos = SkyCoord(ra=c("ra") * u.deg, dec=c("dec") * u.deg).galactic
    with np.errstate(divide="ignore", invalid="ignore"):
        r_med = np.where(plx > 0, 1000.0 / plx, np.nan)
        r_lo = np.where(plx > 0, 1000.0 / (plx + eplx), np.nan)
        r_hi = np.where(plx - eplx > 0, 1000.0 / (plx - eplx), r_med * 3)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    cols = {"phot_g_mean_mag": g, "bp_rp": c("bp_mag") - c("rp_mag"), "l": cpos.l.deg, "b": cpos.b.deg,
            "r_med_geo": r_med, "r_lo_geo": r_lo, "r_hi_geo": r_hi}
    cmd, iso = ps.parent_cmd_isochrone(cols, cfg, gcfg, args.cache_dir)
    m1 = np.asarray(iso["m1_mean"], float)
    p_evolved = np.asarray(iso["p_evolved"], float)

    # ---- AMRF classes and dark M2 ----
    p_yr = P / 365.25
    with np.errstate(divide="ignore", invalid="ignore"):
        amrf = a_au / (m1 ** (1 / 3) * p_yr ** (2 / 3))
    lim_ms, lim_tr = amrf_class_limits(np.where(np.isfinite(m1), m1, 1.0))
    cls = np.where(amrf <= lim_ms, 1, np.where(amrf <= lim_tr, 2, 3))
    cls = np.where(np.isfinite(amrf) & np.isfinite(m1), cls, 0)
    m2_dark = np.full(n, np.nan)
    okm = np.isfinite(m1) & np.isfinite(a0) & (a0 > 0) & (plx > 0) & np.isfinite(P)
    for i in np.flatnonzero(okm):
        try:
            m2_dark[i] = gm.get_companion_mass_from_mass_function(M1=m1[i], a0_mas=a0[i], period=P[i], parallax=plx[i], fluxratio=0.0)
        except Exception:
            pass
    beyond_wd = m2_dark > args.wd_limit

    # ---- spuriousness model ----
    fit = sm.fit_spuriousness_model()
    spec = sm.load_spuriousness_model_file()
    f2b = np.array([sm.f2_x_g_break(gg, ff, g_break=spec.f2_g_break if hasattr(spec, "f2_g_break") else 13.0,
                                    f2_bright=spec.f2_threshold_bright, f2_faint=spec.f2_threshold_faint) if np.isfinite(ff) else np.nan
                    for gg, ff in zip(g, gof)], dtype=float)
    covs = {"goodness_of_fit": gof, "a0_snr": sig, "log_implied_companion_mass": np.log10(m2_dark),
            "f2_x_g_break": f2b, "parallax_snr": plx_obs / eplx, "phot_g_mean_mag": g}
    p_spur = sm.predict_p_spurious(fit.model, covs)
    labels = sm.load_labeled_sources(spec)
    lab = {}
    for s in labels:
        lab.setdefault(int(s.source_id), set()).add(s.state)
    lab_state = np.array(["spurious" if "spurious" in lab.get(int(x), set()) else
                          ("good" if "good" in lab.get(int(x), set()) else
                           ("unknown" if int(x) in lab else "")) for x in sid])

    # ---- tables per distance bin ----
    db = np.digitize(d, DBIN) - 1
    rep = [
        "#391 high-f_m characterization (real DR3 Orbital+AstroSpectroSB1, mirror filters)",
        f"rows {n}; parallax zero point added per G bin {dict(zip([f'{a:g}-{b:g}' for a, b in zip(gedges[:-1], gedges[1:])], np.round(zp * 1000, 1)))} uas",
        f"M1: MIST isochrone posterior mean (finite {np.isfinite(m1).mean():.3f}); AMRF limits from Janssens MS relation",
        f"spuriousness model retained {fit.model.retained_covariates}; labelled sources matched {int((lab_state != '').sum())}",
        "",
        "per distance bin, f_m>0.1 (HI) vs <=0.1 (LO): N | frac AMRF class I/II/III | median plx/err, signif, F2, RUWE | frac P>0.8*1038d | median e | frac ipd_multi_peak>0 | median ipd_gof | frac M2dark>WD | mean P_spur | labels good/spur/unk",
    ]
    rows = []
    for k in range(DBIN.size - 1):
        for tag, m in (("HI", hi), ("LO", ~hi & np.isfinite(fm))):
            s = (db == k) & m
            if not s.any():
                continue
            nn = int(s.sum())
            cl = [np.mean(cls[s] == j) for j in (1, 2, 3)]
            row = dict(dbin=f"{DBIN[k]:g}-{DBIN[k+1]:g}", tag=tag, n=nn, cI=cl[0], cII=cl[1], cIII=cl[2],
                       snr=np.nanmedian(plx_obs[s] / eplx[s]), sig=np.nanmedian(sig[s]), f2=np.nanmedian(gof[s]),
                       ruwe=np.nanmedian(ruwe[s]), long=np.mean(P[s] > 0.8 * BASELINE_DAYS), e=np.nanmedian(ecc[s]),
                       ipdmp=np.nanmean(ipd_mp[s] > 0), ipdgof=np.nanmedian(ipd_gof[s]), bwd=np.nanmean(beyond_wd[s]),
                       pspur=np.nanmean(p_spur[s]), lg=int(np.sum(lab_state[s] == "good")), ls=int(np.sum(lab_state[s] == "spurious")),
                       lu=int(np.sum(lab_state[s] == "unknown")))
            rows.append(row)
            rep.append(f"  d {row['dbin']} {tag}: N={nn} | I/II/III {cl[0]:.2f}/{cl[1]:.2f}/{cl[2]:.2f} | snr {row['snr']:.1f} sig {row['sig']:.1f} F2 {row['f2']:.2f} RUWE {row['ruwe']:.2f}"
                       f" | P>830d {row['long']:.2f} | e {row['e']:.2f} | ipdMP {row['ipdmp']:.3f} | ipdGOF {row['ipdgof']:.3f} | >WD {row['bwd']:.2f}"
                       f" | Pspur {row['pspur']:.2f} | lab {row['lg']}/{row['ls']}/{row['lu']}")

    # ---- decomposition of high-f_m by distance ----
    spur_flag = (p_spur > 0.5) | (P > 0.8 * BASELINE_DAYS) | (lab_state == "spurious")
    cat = np.full(n, "other", dtype=object)
    cat[hi & spur_flag] = "plausibly spurious"
    cat[hi & ~spur_flag & (cls == 3) & ~beyond_wd] = "plausibly compact (class III, M2_dark <= WD limit)"
    cat[hi & ~spur_flag & (cls == 3) & beyond_wd] = "class III beyond WD limit (NS/BH or spurious)"
    cat[hi & ~spur_flag & (cls == 2)] = "class II: triple (luminous inner pair) or compact"
    cat[hi & ~spur_flag & (cls == 1)] = "class I: luminous companion allowed"
    cats = ["plausibly spurious", "plausibly compact (class III, M2_dark <= WD limit)",
            "class III beyond WD limit (NS/BH or spurious)", "class II: triple (luminous inner pair) or compact",
            "class I: luminous companion allowed"]
    rep += ["", "decomposition of f_m > 0.1 orbits by distance (fraction of all orbits in the bin; criteria in README):"]
    frac = np.zeros((len(cats), DBIN.size - 1))
    for k in range(DBIN.size - 1):
        s = db == k
        tot = max(int(s.sum()), 1)
        parts = []
        for j, cname in enumerate(cats):
            frac[j, k] = np.sum(s & (cat == cname)) / tot
            parts.append(f"{cname.split(' (')[0].split(':')[0]} {frac[j, k]:.3f}±{np.sqrt(np.sum(s & (cat == cname))) / tot:.3f}")
        rep.append(f"  d {DBIN[k]:g}-{DBIN[k+1]:g} (N={int(s.sum())}, high-f_m {np.mean(hi[s]):.3f}): " + "; ".join(parts))

    # ---- figures ----
    cfgp = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    dc = 0.5 * (DBIN[1:] + DBIN[:-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(dc.size)
    for j, cname in enumerate(cats):
        st = series_style(j, cfgp)
        ax.bar(dc, frac[j], width=np.diff(DBIN) * 0.9, bottom=bottom, color=st["color"], label=cname, edgecolor="k")
        bottom += frac[j]
    apply_axes_style(ax, cfgp, xlabel="1/parallax, zero-point corrected (kpc)", ylabel="fraction of orbits with f_m > 0.1")
    ax.set_ylim(0, float(bottom.max()) * 1.15)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("Real DR3 high-f_m orbits: decomposition by distance (criteria in README)")
    fig.tight_layout()
    save_figure(fig, args.out_dir / "decomposition_vs_distance.png", dpi=int(cfg.diagnostics.figure_dpi))

    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    keys = [("cIII", "frac AMRF class III"), ("bwd", "frac M2_dark > 1.5 Msun"), ("pspur", "mean P(spurious)"),
            ("long", "frac P > 0.8 x 1038 d"), ("sig", "median significance"), ("f2", "median F2"),
            ("ruwe", "median RUWE"), ("e", "median eccentricity")]
    for ax, (kk, lab_) in zip(np.ravel(axs), keys):
        for i, tag in enumerate(("HI", "LO")):
            xs = [0.5 * (DBIN[k] + DBIN[k + 1]) for k in range(DBIN.size - 1) for r in rows if r["tag"] == tag and r["dbin"] == f"{DBIN[k]:g}-{DBIN[k+1]:g}"]
            ys = [r[kk] for k in range(DBIN.size - 1) for r in rows if r["tag"] == tag and r["dbin"] == f"{DBIN[k]:g}-{DBIN[k+1]:g}"]
            st = series_style(i, cfgp)
            ax.plot(xs, ys, "o-", color=st["color"], lw=2, label="f_m > 0.1" if tag == "HI" else "f_m <= 0.1")
        apply_axes_style(ax, cfgp, xlabel="1/parallax (kpc)", ylabel=lab_)
        ax.legend(fontsize=10)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "indicators_vs_distance.png", dpi=int(cfg.diagnostics.figure_dpi))

    fig, ax = plt.subplots(figsize=(9, 7))
    mg = np.geomspace(0.3, 6, 100)
    l1, l2 = amrf_class_limits(mg)
    sel = np.isfinite(amrf) & np.isfinite(m1)
    rng = np.random.default_rng(0)
    pick = rng.choice(np.flatnonzero(sel), size=min(20000, sel.sum()), replace=False)
    sc = ax.scatter(m1[pick], amrf[pick], c=np.clip(d[pick], 0, 2.5), s=2, cmap="viridis", alpha=0.6)
    ax.plot(mg, l1, "k-", lw=2, label="class I/II (single MS companion max)")
    ax.plot(mg, l2, "k--", lw=2, label="class II/III (MS-pair max)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.01, 3)
    plt.colorbar(sc, ax=ax, label="1/parallax (kpc)")
    apply_axes_style(ax, cfgp, xlabel=r"M$_1$ (M$_\odot$), MIST isochrone", ylabel="AMRF")
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "amrf_vs_m1.png", dpi=int(cfg.diagnostics.figure_dpi))

    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
