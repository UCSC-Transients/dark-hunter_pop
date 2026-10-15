#!/usr/bin/env python3
"""Rung 3 refit 5 — INTERIM (#391; spec §12.15–12.16): f_evo + calibrated spurious template on gens 23-30.

NOT the rung-3 fit: the §12.14 bin rule is not met with the existing draws (docs/gate391/rung3_modelfix/), so this
evaluates the adopted model fixes on the best ESS-feasible grid, labelled "interim, not §12.14-compliant".

Modes:
* ``choose``: pick the interim grid from ESS alone (control-fit θ noW/cmdW and MdS17): orbit G × d × log P × e × CMD class
  and acceleration G × d × class, most bins with every big bin at ESS_b >= 30 (class axis always present, §12.15).
* ``fit --model old``: the gens 23-30 control model (no f_evo, the §12.12 template ∝ real (G, d, class) counts × T_PE,
  N_s ~ N(0.21, 0.04) × N_o) on the interim grid — the "before".
* ``fit --model new``: + ln f_evo (§12.15) and the §12.16 template T_b ∝ Σ_{real orbits in b} π̂(x_i), N_s = f_s Σ π̂,
  f_s ~ N(1, σ_cal); π̂ from ``scripts/calibrate_spurious_391.py``, covariates clamped to the calibrated range
  (MP-Q46 option (i), provisional).
* ``feasibility``: the §12.14 compliant grids (+ class axis) at this fit's θ and the draw cost to resolve them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_refit4_391 as r4  # noqa: E402
from rung3_fit_391 import PANELS, bin_index, say_loglike  # noqa: E402

GEN_ARTIFACTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5", "restart_gen26_topup.h5",
                 "restart_gen27_evolved.h5", "restart_gen28_recentred.h5", "restart_gen29_topup.h5", "restart_gen30_topup.h5")
POP = r4.NAMES[:13]
NAMES = POP + ("ln_f_evo", "N_s")
CLS_EDGES = np.array([-0.5, 0.5, 1.5])  # 0 dwarf + unclassified, 1 evolved
CAND = {"g_mag": [[5.0, 19.5], [5.0, 13.0, 19.5], [5.0, 12.0, 14.0, 16.0, 19.5]],
        "distance_kpc": [[0.0, 0.7, 6.0], [0.0, 0.5, 1.0, 6.0], [0.0, 0.3, 0.7, 1.5, 6.0]],
        "log10_period_days": [[0.0, 2.6, 2.9193], [0.0, 2.0, 2.6, 2.9193], [0.0, 2.3, 2.6, 2.75, 2.9193]],
        "eccentricity": [[0.0, 0.4, 1.0], [0.0, 0.3, 0.6, 1.0]]}
COMPLIANT = {"g_mag": [[5.0, 13.0, 19.5], [5.0, 12.0, 14.0, 16.0, 19.5]],
             "distance_kpc": [[0.0, 0.5, 1.0, 6.0], [0.0, 0.3, 0.7, 1.5, 6.0]],
             "log10_period_days": [[0.0, 2.0, 2.6, 2.9193]],
             "eccentricity": [[0.0, 0.3, 0.6, 1.0], [0.0, 0.2, 0.4, 0.6, 1.0]]}


def extend(args, fc, ns):
    """Mock CMD class per likelihood draw; real orbit / acceleration class and π covariates (same masks as r4.setup)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.giants import classify_evolved, classify_parent, cmd_for_rows, load_giants_config, parent_row_cmd
    from darkhunter_pop.spuriousness_model import harmonic_distance

    cfg = ns.cfg
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pcls, ridge = classify_parent(parent, cfg, gcfg, cmd=parent_row_cmd(parent, cfg, gcfg))
    arts = args.artifacts or GEN_ARTIFACTS
    prow = np.concatenate([np.asarray(ps.read_proposal_artifact(args.artifact_dir / a)[0]["parent_row"], np.int64) for a in arts])
    if prow.size != np.load(args.cache, allow_pickle=True)["generation"].size:
        raise SystemExit("artifact list does not match the weight cache")
    ns.m_cls = (pcls.classified & pcls.evolved)[prow[ns.rel]].astype(float)
    ns.prow = prow[ns.rel]
    # §12.19(a) (#483): MdS17 laws at M1,ZAMS by reweighting; §12.19(b) (#484): the slope-break hinge on that mass
    zc = np.load(args.cache, allow_pickle=True)
    m1c = np.asarray(zc["m1_msun"], float)[ns.rel]
    ns.m1_law_mass = fc.get("m1_law_mass", "current")
    ns.m1_law = m1c * ps.initial_to_current_ratio(parent.isochrone)[prow[ns.rel]] if ns.m1_law_mass == "initial" else m1c
    ns.m1_law_lr = np.zeros(m1c.size)
    if ns.m1_law_mass == "initial":
        tgt = ps.load_proposal_set_fragment(args.fragment).target_mds17
        ns.m1_law_lr = ps.m1_law_log_ratio(m1c, ns.m1_law, np.asarray(zc["m2_msun"], float)[ns.rel],
                                           np.asarray(zc["period_days"], float)[ns.rel], np.asarray(zc["eccentricity"], float)[ns.rel], tgt)
        ns.W = {k: v * np.exp(ns.m1_law_lr) for k, v in ns.W.items()}
    # §12.19(c) (#486): MdS17 frequency on Pecaut & Mamajek (2013)'s mass scale (their M1 source), by reweighting
    ns.mass_scale = (fc.get("mds17_mass_scale") or {}).get("method", "mist")
    if getattr(args, "mass_scale", None):
        ns.mass_scale = args.mass_scale
    ns.mass_scale_lr, ns.mass_scale_map = np.zeros(m1c.size), None
    if ns.mass_scale == "pm13_colour":
        from darkhunter_pop import moe_distefano as mds

        msc = fc["mds17_mass_scale"]
        tab13 = mds.load_pm13_dwarf_table(args.parent_dir.parents[2] / msc["pm13_table"])
        iso = parent.isochrone
        dwarf = parent.usable & np.asarray(iso["ok"], bool) & pcls.classified & ~pcls.evolved
        pm = mds.pm13_mass_from_colour(np.asarray(pcls.colour0, float)[dwarf], tab13)
        cen, mp = mds.mass_scale_mapping(np.asarray(iso["m1_mean"], float)[dwarf], pm, np.asarray(msc["mapping_edges_msun"], float),
                                         min_count=int(msc["mapping_min_count"]))
        ns.mass_scale_map = (cen, mp)
        tgt = ps.load_proposal_set_fragment(args.fragment).target_mds17
        table = mds.load_mds17_table(tgt.table_path)
        lp_t = np.log10(np.asarray(zc["period_days"], float)[ns.rel])

        def lf(m):
            ms_ = np.clip(m, table.m1_range[0], table.m1_range[1])
            f = mds.f_logp_q03(ms_, lp_t, table) * mds.low_mass_frequency_scale(
                m, m1_anchor_msun=tgt.provisional_low_mass_anchor_msun, m1_zero_msun=tgt.provisional_low_mass_zero_msun)
            with np.errstate(divide="ignore"):
                return np.log(f)

        a_, b_ = lf(mds.apply_mass_scale(ns.m1_law, cen, mp)), lf(ns.m1_law)
        ns.mass_scale_lr = np.where(np.isfinite(a_) & np.isfinite(b_), a_ - b_, 0.0)
        ns.W = {k: v * np.exp(ns.mass_scale_lr) for k, v in ns.W.items()}
    # §12.22 (#491): condition each row's companion on its observed CMD (single-star vs star + MdS17 companion evidence)
    ns.cmd_factor = np.ones(m1c.size)
    bcc = fc.get("cmd_binary_conditioning") or {}
    if getattr(args, "cmd_conditioning", None) is not None:
        bcc = dict(bcc, enabled=args.cmd_conditioning == "on")
    if bcc.get("enabled"):
        from darkhunter_pop import binary_cmd_evidence as bce
        from darkhunter_pop import isochrone_mass as im_

        rows_u, inv_u = np.unique(ns.prow, return_inverse=True)
        pcm = {k: np.asarray(v, float)[rows_u] for k, v in parent.cmd.items() if k in ("colour0", "mg0", "sigma_mu", "ebv", "a_g", "e_bp_rp")}
        with np.errstate(divide="ignore", invalid="ignore"):
            ka = np.where(pcm["ebv"] > 0, pcm["a_g"] / pcm["ebv"], 0.0)
            ke = np.where(pcm["ebv"] > 0, pcm["e_bp_rp"] / pcm["ebv"], 0.0)
        sc_, sm_ = im_.likelihood_sigmas(pcm["sigma_mu"], pcm["ebv"], ka, ke, cfg.isochrone_mass.likelihood)
        native = im_.load_native_grid(cfg.isochrone_mass, cfg.paths.data_root)
        pts = im_.prior_points(native, cfg.isochrone_mass)
        qe = np.asarray(bcc["q_edges"], float)
        qc = 0.5 * (qe[1:] + qe[:-1])
        cshift = None
        if bcc.get("ms_locus", "mist") == "ridge":  # §12.22 second amendment (b): empirical single-star MS below ~0.8 Msun
            jlo, jhi = float(bcc["join_lo_msun"]), float(bcc["join_hi_msun"])
            smg, sdc = bce.ridge_colour_shift(pts, ridge.colour, ridge.mag, join_hi_msun=jhi, mg_edges=np.asarray(bcc["ridge_mg_edges"], float))
            pts = bce.ridge_corrected_points(pts, smg, sdc, join_lo_msun=jlo, join_hi_msun=jhi)
            cshift = (smg, sdc, jlo, jhi)
            ns.ridge_shift = (smg, sdc)
        les, llr = bce.log10_lr_grid(pts, native, cfg.isochrone_mass, qc, pcm["colour0"], pcm["mg0"], sc_, sm_, companion_shift=cshift)
        del pts
        m1r_ = np.asarray(parent.isochrone["m1_mean"], float)[rows_u]
        if bcc.get("m1_min_msun") is not None:  # §12.22 second amendment (a): no conditioning below the mass floor
            llr = np.where((m1r_ < float(bcc["m1_min_msun"]))[:, None], 0.0, llr)
        tgt = ps.load_proposal_set_fragment(args.fragment).target_mds17
        m1r = np.asarray(parent.isochrone["m1_mean"], float)[rows_u]
        lam = bce.q_intensity_grid(np.where(np.isfinite(m1r), m1r, 1.0), qe, tgt)
        norm, F = bce.row_factor(lam, llr)
        qd = np.asarray(zc["m2_msun"], float)[ns.rel] / m1c
        ns.cmd_factor = bce.draw_factor(qd, inv_u, qc, llr, norm)
        if bcc.get("normalization", "theta") == "baseline":  # superseded fixed-MdS17 normalization (diagnostic only)
            ns.W = {k: v * ns.cmd_factor for k, v in ns.W.items()}
        else:  # §12.22 amendment: θ-dependent normalization inside the likelihood
            ns.cond = dict(grids=bce.precompute_theta_grids(np.where(np.isfinite(m1r), m1r, 1.0), qe, tgt),
                           llr=llr, inv=inv_u, lr_draw=bce.draw_factor(qd, inv_u, qc, llr, np.ones(rows_u.size)),
                           lm1=np.log(np.where(np.isfinite(m1r), m1r, 1.0) / float(fc["m1_pivot_msun"])),
                           lo=np.where(np.isfinite(m1r), m1r, 1.0) < float(fc["m1_pivot_msun"]),
                           evo=(pcls.classified & pcls.evolved)[rows_u].astype(float))
        ns.cmd_summary = dict(rows=int(rows_u.size), rows_undefined=int((~np.isfinite(les)).sum()),
                              factor_q=np.percentile(ns.cmd_factor, [5, 25, 50, 75, 95]).round(3).tolist(),
                              F_median=float(np.median(F)))
    # §12.23 (#494): MP-Q27 IPD-pass weight for companions near/below Gaia's resolution (El-Badry+2021 calibrated)
    ns.ipd_factor = np.ones(m1c.size)
    rcc = fc.get("resolved_companions") or {}
    if getattr(args, "below_envelope", None):
        rcc = dict(rcc, enabled=True, below_envelope=args.below_envelope)
    if rcc.get("enabled"):
        from darkhunter_pop import resolved_companions as rcm

        tr = {k: np.concatenate([np.asarray(ps.read_proposal_artifact(args.artifact_dir / a)[0][k]) for a in arts])[ns.rel]
              for k in ("parallax_mas", "log10_flux_ratio", "is_dark")}
        cut = prop.halbwachs_cuts
        fmp = float(cut.ipd_frac_multi_peak_max)
        gof = float(cut.ipd_gof_harmonic_amplitude_max)
        cat = rcm.load_elbadry2021(args.parent_dir.parents[2] / rcc["snapshot"], rcc["sha256"])
        tab = rcm.build_tables(cat, rcc["theta_edges_arcsec"], rcc["dg_edges"], r_chance_max=float(rcc["r_chance_max"]),
                               g2_max=float(rcc["g2_max"]), ref_theta_arcsec=tuple(rcc["ref_theta_arcsec"]), ipd_fmp_max=fmp,
                               ipd_gof_max=gof, min_pairs=int(rcc["min_pairs"]))
        del cat
        m2c = np.asarray(zc["m2_msun"], float)[ns.rel]
        pc_d = np.asarray(zc["period_days"], float)[ns.rel]
        a_au = np.cbrt((m1c + m2c) * (pc_d / 365.25) ** 2)
        theta = float(rcc["projection_factor"]) * a_au * np.asarray(tr["parallax_mas"], float) / 1000.0
        dg = np.where(np.asarray(tr["is_dark"], bool), np.inf, -2.5 * np.asarray(tr["log10_flux_ratio"], float))
        o6tab = None
        if rcc.get("below_envelope") == "orb6":  # §12.23 amendment: ORB6-calibrated partially resolved regime
            from astropy.table import Table as _T

            from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

            o6 = rcc["orb6"]
            root6 = args.parent_dir.parents[2] / o6["snapshot_dir"]
            for fn, sha in o6["sha256"].items():
                if rcm.hashlib.sha256((root6 / fn).read_bytes()).hexdigest() != sha:
                    raise ValueError(f"{root6 / fn}: SHA256 mismatch")
            pt, xm = _T.read(root6 / "orb6_parsed.ecsv"), _T.read(root6 / "orb6_gaiadr3_xmatch5as.ecsv")
            orb = rcm.parse_orb6([str(v) for v in pt["orb6_line"]])
            gmk = import_gaiamock_mod()
            cfun = gmk.read_in_C_functions()
            rho = rcm.orb6_separation(orb, float(o6["epoch_yr"]), lambda M, e: gmk.solve_kepler_eqn_on_array(M, e, cfun))
            pr = rcm.orb6_pairs(pt["orb6_row"], pt["ra"], pt["dec"], rho, orb["v2"] - orb["v1"],
                                {c: np.ma.filled(np.ma.asarray(xm[c], float), np.nan) if c != "orb6_row" else np.asarray(xm[c])
                                 for c in ("orb6_row", "RAJ2000", "DEJ2000", "RAdeg", "DEdeg", "Gmag", "IPDfmp", "IPDgofha")},
                                ipd_fmp_max=fmp, ipd_gof_max=gof, match_min_arcsec=float(o6["match_min_arcsec"]),
                                sep_tol_frac=float(o6["sep_tol_frac"]), sep_tol_min_arcsec=float(o6["sep_tol_min_arcsec"]))
            gw = {int(k): float(v) for k, v in o6["grade_weights"].items()}
            wgt = np.array([gw.get(int(gv), 0.0) if np.isfinite(gv) else 0.0 for gv in orb["grade"]])
            o6tab = rcm.orb6_pass_table(pr["theta"], pr["dg"], pr["passed"], wgt, o6["theta_edges_arcsec"], o6["dg_edges"],
                                        min_weight=float(o6["min_weight"]), ref_theta_max=float(o6["ref_theta_max_arcsec"]),
                                        fill_dg_max=float(o6["fill_dg_max"]))
        ns.ipd_factor = rcm.p_pass_ipd(theta, dg, tab, below_envelope=rcc.get("below_envelope", "none"),
                                       hold_min_arcsec=float(rcc.get("hold_min_arcsec", 0.1)), orb6=o6tab)
        ns.W = {k: v * ns.ipd_factor for k, v in ns.W.items()}
        dtr = 1.0 / np.asarray(tr["parallax_mas"], float)
        ns.ipd_summary = {f"d{lo}-{hi} {nm}": round(float(np.mean(ns.ipd_factor[sel & (dtr >= lo) & (dtr < hi)])), 4)
                          for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 6)) for nm, sel in (("orbit", ns.in_o), ("accel", ns.in_a))
                          if np.any(sel & (dtr >= lo) & (dtr < hi))}
        ns.ipd_summary["below_envelope"] = rcc.get("below_envelope", "none")
    m_br = getattr(args, "m_break", None) or fc.get("m1_break_msun")
    ns.m_break = float(m_br) if m_br else None
    ns.hinge = np.maximum(0.0, np.log(ns.m1_law / ns.m_break)) if ns.m_break else np.zeros(m1c.size)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])

    def col(t, n):
        return np.ma.filled(np.ma.asarray(t[n], float), np.nan)

    def cls_of(g, bprp, ra, dec, plx_obs, eplx):
        plx = plx_obs + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(plx > 0, 1000 / plx, np.nan)
            rlo = np.where(plx > 0, 1000 / (plx + eplx), np.nan)
            rhi = np.where(plx - eplx > 0, 1000 / (plx - eplx), np.nan)
            d = np.where(plx > 0, 1 / plx, np.nan)
        gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
        c = cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
        r = classify_evolved(c.mg0, c.colour0, c.sigma_mu, ridge, gcfg.provisional_n_sigma)
        return (r.classified & r.evolved).astype(float), d

    out = {k: np.asarray(v, float) for k, v in fc["outer"].items()}
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    g = col(rt, "g_mag")
    bprp = col(rt, "bp_mag") - col(rt, "rp_mag")
    rc, d = cls_of(g, bprp, col(rt, "ra"), col(rt, "dec"), col(rt, "parallax"), col(rt, "parallax_error"))
    P, e = col(rt, "period"), col(rt, "eccentricity")
    r_in = (P <= ns.PMAX) & (bin_index([g, d, np.log10(P), e], [out[a] for a in r4.AXES]) >= 0)
    assert np.allclose(np.nan_to_num(g[r_in]), np.nan_to_num(ns.ro_cols[0])), "real orbit mask differs from r4.setup"
    ns.r_cls = rc[r_in]
    # π covariates for real orbits (§12.16 definitions)
    sig = {}
    try:
        import h5py

        with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
            sid_p, sg = h["published/source_id"][()], h["published/significance"][()]
        sig = dict(zip(sid_p.tolist(), sg.tolist()))
    except Exception:  # pragma: no cover
        pass
    sid = np.asarray(rt["source_id"], np.int64)[r_in]
    import yaml

    scfg = yaml.safe_load(Path("config/spuriousness_model.yaml").read_text())
    ns.r_cov = {"g_mag": g[r_in], "bp_rp": bprp[r_in], "distance_kpc": d[r_in], "log10_period": np.log10(P[r_in]),
                "eccentricity": e[r_in], "log10_a0_snr": np.log10(np.array([sig.get(int(s), np.nan) for s in sid])),
                "log10_parallax_snr": np.log10(col(rt, "parallax") / col(rt, "parallax_error"))[r_in],
                "goodness_of_fit": col(rt, "goodness_of_fit")[r_in],
                "harmonic_distance": np.array([harmonic_distance(p, fundamental_period_days=float(scfg["scanning_law_fundamental_period_days"]),
                                                                 multiples_max=int(scfg["harmonic_multiples_max"])) for p in P[r_in]])}
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    inp = {"source_id": np.asarray(at["source_id"], np.int64)}
    inp.update({c: col(at, c) for c in ("phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ipd_frac_multi_peak",
                                        "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")})
    keep_a, _ = ps.real_comparison_keep(at, prop, inp)
    at = at[keep_a]
    ag = col(at, "g_mag")
    ac, ad = cls_of(ag, col(at, "bp_mag") - col(at, "rp_mag"), col(at, "ra"), col(at, "dec"), col(at, "parallax"), col(at, "parallax_error"))
    ra_in = bin_index([ag, ad], [out["g_mag"], out["distance_kpc"]]) >= 0
    ns.ra_cls = ac[ra_in]
    return ns


def binned5(ns, oe, ae):
    ob = np.where(ns.in_o, bin_index(ns.o_cols + [ns.m_cls], oe + [CLS_EDGES]), -1)
    ab = np.where(ns.in_a, bin_index(ns.a_cols + [ns.m_cls], ae + [CLS_EDGES]), -1)
    n_ob = int(np.prod([e.size - 1 for e in oe])) * 2
    n_ab = int(np.prod([e.size - 1 for e in ae])) * 2
    rob = bin_index(ns.ro_cols + [ns.r_cls], oe + [CLS_EDGES])
    rab = bin_index(ns.ra_cols + [ns.ra_cls], ae + [CLS_EDGES])
    k_o = np.bincount(rob[rob >= 0], minlength=n_ob).astype(float)
    k_a = np.bincount(rab[rab >= 0], minlength=n_ab).astype(float)
    return ob, ab, k_o, k_a, n_ob, n_ab, rob


def pi_hat(cal, cov, clamp=True, beta=None):
    """§12.16 π(x) via ``darkhunter_pop.rung3_model.spurious_probability`` (MP-Q46: clamp or logit)."""
    from darkhunter_pop.rung3_model import spurious_probability

    return spurious_probability(cal, cov, extrapolation="clamp" if clamp else "logit", beta=beta)


def theta_points(args, ns):
    fits = json.loads(args.control.read_text())
    th_mds = np.array([0, 0.5, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1, 1, 1, 0], float)
    return {"control noW": ns.W["noW"] * np.exp(ns.log_mod(np.array(fits["noW"]["theta"]))),
            "control cmdW": ns.W["cmdW"] * np.exp(ns.log_mod(np.array(fits["cmdW"]["theta"]))),
            "MdS17": ns.W["noW"] * np.exp(ns.log_mod(th_mds))}


def choose(args, fc, ns):
    import yaml

    pts = theta_points(args, ns)
    rows = []
    for combo in itertools.product(*(CAND[a] for a in r4.AXES)):
        oe = [np.asarray(c, float) for c in combo]
        for ae_c in itertools.product(CAND["g_mag"], CAND["distance_kpc"]):
            ae = [np.asarray(c, float) for c in ae_c]
            ob, ab, k_o, k_a, n_ob, n_ab, _ = binned5(ns, oe, ae)
            bo, ba = k_o >= 0.01 * k_o.sum(), k_a >= 0.01 * k_a.sum()
            eo = np.minimum.reduce([r4.ess_b(w, ob, n_ob) for w in pts.values()])
            ea = np.minimum.reduce([r4.ess_b(w, ab, n_ab) for w in pts.values()])
            ok = bool(np.all(eo[bo] >= args.min_ess) and np.all(ea[ba] >= args.min_ess))
            rows.append((ok, n_ob + n_ab, n_ob, combo, ae_c, float(k_o[(eo >= args.min_ess)].sum() / k_o.sum())))
    good = [r for r in rows if r[0]]
    best = max(good, key=lambda r: (r[2], r[1])) if good else max(rows, key=lambda r: (r[5], r[2]))
    newcfg = dict(fc)
    newcfg["orbit_bins"] = {a: list(map(float, e)) for a, e in zip(r4.AXES, best[3])}
    newcfg["acceleration_bins"] = {"g_mag": list(map(float, best[4][0])), "distance_kpc": list(map(float, best[4][1]))}
    newcfg["cmd_class_axis"] = True
    newcfg["label"] = "INTERIM, not §12.14-compliant (best ESS-feasible grid with the CMD-class axis)"
    newcfg["bin_choice"] = {"rule": "§12.15/§12.16 interim: CMD-class axis always; most orbit bins with every big bin (orbit and acceleration) "
                                    "at min ESS_b >= 30 over (control noW, control cmdW, MdS17); ESS alone",
                            "report": [f"{len(rows)} candidates, {len(good)} feasible; chosen orbit {[len(c) - 1 for c in best[3]]} x class 2 = {best[2]} bins, "
                                       f"acceleration {[len(c) - 1 for c in best[4]]} x class 2"]}
    args.fit_config.write_text("# Rung-3 INTERIM grid (#391 §12.15/§12.16): chosen by `rung3_refit5_391.py choose` from ESS alone.\n"
                               + yaml.safe_dump({"rung3": newcfg}, sort_keys=False))
    print(newcfg["bin_choice"]["report"][0], best[3], best[4])


def fit(args, fc, ns):
    from darkhunter_pop.rung3_model import evolved_log_modifier

    ps = ns.ps
    oe = [np.asarray(fc["orbit_bins"][a], float) for a in r4.AXES]
    ae = [np.asarray(fc["acceleration_bins"][a], float) for a in ("g_mag", "distance_kpc")]
    ob, ab, k_o, k_a, n_ob, n_ab, rob = binned5(ns, oe, ae)
    N_o = float(k_o.sum())
    big = k_o >= 0.01 * N_o
    shp = [e.size - 1 for e in oe] + [2]
    phi = float(fc["empty_bin_floor"])
    # §12.12 floor: a bin is "empty" when its mock expectation is zero for every θ, i.e. no draw with positive
    # θ-independent base weight (the modifiers are multiplicative), under either weighting.
    base_o = sum(np.bincount(ob[ob >= 0], weights=ns.W[k][ob >= 0], minlength=n_ob) for k in ("noW", "cmdW"))
    base_a = sum(np.bincount(ab[ab >= 0], weights=ns.W[k][ab >= 0], minlength=n_ab) for k in ("noW", "cmdW"))
    # "zero" = below 0.5 expected counts at the base (θ = 0) weights: a bin held by a single ~1e-284-weight draw is
    # empty in practice, but k·ln μ would still pull θ through it. Its mock term is dropped (θ-independent).
    empty_o, empty_a = base_o < 0.5, base_a < 0.5
    floor_o, floor_a = np.where(empty_o, phi, 0.0), np.where(empty_a, phi, 0.0)
    rep = [f"#391 rung 3 refit 5, model '{args.model}' — {fc.get('label', 'INTERIM')}", "",
           f"orbit bins {n_ob} ({' x '.join(map(str, shp))}) {[e.tolist() for e in oe]} x class; accelerations {n_ab}",
           f"real: {int(N_o)} C1 orbits ({ns.r_cls.mean():.3f} evolved), {int(k_a.sum())} accelerations ({ns.ra_cls.mean():.3f} evolved); "
           f"empty bins: orbit {int(empty_o.sum())} ({int(k_o[empty_o].sum())} real), acceleration {int(empty_a.sum())}"]
    if args.model == "old":
        # §12.12 template, made class-aware: T ∝ real (G, d, class) counts × T_PE(log P, e) from the control fit's run
        ctrl = json.loads(args.control.read_text())
        R = k_o.reshape(shp)
        T_GDc = R.sum(axis=(2, 3)) / N_o  # (G, d, class)
        cal_tpe = np.asarray(ctrl.get("template_T_PE", []), float)
        if cal_tpe.size != shp[2] * shp[3]:
            # rebuild T_PE on these (log P, e) edges from the real counts in the measured bin × the PR #455 marginal ratios
            cal_tpe = np.ones((shp[2], shp[3]))
            rg, rd, rlp, re_ = ns.ro_cols
            mb = (rd >= 0.7) & (rd <= 1.5) & (rg >= 13) & (rg <= 16)
            cal_tpe = np.histogram2d(rlp[mb], re_[mb], bins=[oe[2], oe[3]])[0]
        T = np.einsum("abk,cd->abcdk", T_GDc, cal_tpe / cal_tpe.sum()).ravel()
        prior = (0.21 * N_o, 0.04 * N_o)
        names = POP + ("N_s",)
    else:
        cal = json.loads(args.calibration.read_text())
        pi_i = pi_hat(cal, {k: v for k, v in ns.r_cov.items()}, clamp=(args.extrapolation == "clamp"))
        T = np.bincount(rob[rob >= 0], weights=pi_i[rob >= 0], minlength=n_ob)
        S0 = float(T.sum())
        T = T / S0
        # σ_cal: relative spread of Σ π̂ under the coefficient covariance
        rng = np.random.default_rng(args.seed)
        bs = rng.multivariate_normal(np.asarray(cal["pi_beta"]), np.asarray(cal["pi_cov"]), size=300)
        sums = np.array([pi_hat(cal, ns.r_cov, clamp=(args.extrapolation == "clamp"), beta=b).sum() for b in bs])
        sig_cal = float(np.std(sums) / S0)
        prior = (S0, sig_cal * S0)
        names = POP + ("ln_f_evo",) + (("alpha_br",) if ns.m_break else ()) + ("N_s",)
        rep.append(f"M1 law mass: {ns.m1_law_mass} (|Δ ln w| median {np.median(np.abs(ns.m1_law_lr)):.4f}); slope break M_br = {ns.m_break}")
        if getattr(ns, "ipd_summary", None):
            rep.append(f"MP-Q27 IPD-pass weight §12.23 (unweighted mean per truth-distance bin, detected draws): {ns.ipd_summary}")
        if getattr(ns, "cmd_summary", None):
            rep.append(f"CMD companion conditioning §12.22: {ns.cmd_summary}")
            if getattr(ns, "ridge_shift", None) is not None:
                rep.append("  ms_locus ridge: colour shift Δc(M_G) " + " ".join(f"{a:.1f}:{b:+.3f}" for a, b in zip(*ns.ridge_shift) if abs(b) > 0))
        if ns.mass_scale_map is not None:
            rep.append("MdS17 mass scale pm13_colour, M1_PM13(M1_MIST): " + " ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(*ns.mass_scale_map))
                       + f"; Δ ln w median {np.median(ns.mass_scale_lr):+.3f}")
        rep.append(f"spurious template §12.16: pi covariates {cal['pi_keys']}; Σ pi_hat over real C1 orbits = {S0:.0f} ({S0 / N_o:.3f} of them), "
                   f"σ_cal {sig_cal:.3f} (relative); extrapolation: {args.extrapolation}")
    T = np.where(empty_o, 0.0, T)
    rep.append(f"N_s prior N({prior[0]:.0f}, {prior[1]:.0f})")
    n_par = len(names)
    i_ns = names.index("N_s")
    i_fe = names.index("ln_f_evo") if "ln_f_evo" in names else None
    i_br = names.index("alpha_br") if "alpha_br" in names else None
    evo = ns.m_cls

    def full(th):
        t = np.zeros(14)
        t[:13] = th[:13]
        return t

    # §12.20 (#488): MC variance per bin. ``draw``: Σ w² (draws independent). ``parent_row``: Σ_rows (Σ_{draws of row in bin} w)²,
    # the variance of the bin expectation under resampling the parent subsample (draws sharing a parent row are correlated).
    mc_var = fc.get("mc_variance", "draw")
    if getattr(args, "mc_variance", None):
        mc_var = args.mc_variance
    rep.append(f"MC variance: {mc_var}")

    from darkhunter_pop.rung3_model import clustered_bin_variance, row_bin_groups

    grp_o, grp_a = row_bin_groups(ob, ns.prow), row_bin_groups(ab, ns.prow)

    def bin_var(w, bins, grp, n):
        if mc_var == "draw":
            s_ = bins >= 0
            return np.bincount(bins[s_], weights=w[s_] ** 2, minlength=n)
        return clustered_bin_variance(w, grp, n)

    def cond_factor(th):
        """§12.22 amendment: per-draw LR(q) × (1+F_r(θ))/(1+Σ Λ_r(q;θ) LR_r(q)) with the current modifiers."""
        from darkhunter_pop import binary_cmd_evidence as bce

        c = ns.cond
        lnA, alo, ahi, gP, lnL, dq, lnT = th[:7]
        log_s = lnA + np.where(c["lo"], alo, ahi) * c["lm1"] + (th[i_fe] * c["evo"] if i_fe is not None else 0.0)
        lpc = c["grids"]["lpc"]
        log_mp = gP * (lpc - fc["log_p_tilt_pivot"]) + lnL / (1.0 + np.exp(-(lpc - fc["long_p_centre"]) / fc["long_p_width"]))
        norm = bce.theta_row_norm(c["grids"], c["llr"], log_s, log_mp, dq, lnT, float(fc["twin_q_min"]))
        return c["lr_draw"] * norm[c["inv"]]

    def expected(th, wkey):
        lm = ns.log_mod(full(th)) + (evolved_log_modifier(evo, th[i_fe]) if i_fe is not None else 0.0)
        if i_br is not None:
            lm = lm + th[i_br] * ns.hinge
        w = ns.W[wkey] * np.exp(lm)
        if getattr(ns, "cond", None) is not None:
            w = w * cond_factor(th)
        so, sa = ob >= 0, ab >= 0
        keep_o, keep_a = ~empty_o, ~empty_a
        return (w, np.bincount(ob[so], weights=w[so], minlength=n_ob) * keep_o, bin_var(w, ob, grp_o, n_ob) * keep_o,
                np.bincount(ab[sa], weights=w[sa], minlength=n_ab) * keep_a, bin_var(w, ab, grp_a, n_ab) * keep_a)

    def nll(th, wkey):
        _, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        ll = say_loglike(k_o, mu_o + th[i_ns] * T + floor_o, v_o) + say_loglike(k_a, mu_a + floor_a, v_a)
        ll += -0.5 * ((th[i_ns] - prior[0]) / prior[1]) ** 2
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    br = par.get("alpha_br", [0.0, -5.0, 5.0])
    x0 = np.array([par[n][0] for n in POP] + ([0.0] if i_fe is not None else []) + ([br[0]] if i_br is not None else []) + [prior[0]], float)
    bounds = ([(par[n][1], par[n][2]) for n in POP] + ([(-3.0, 3.0)] if i_fe is not None else [])
              + ([(br[1], br[2])] if i_br is not None else []) + [(0.0, max(5 * prior[0], 1.0))])
    for item in (getattr(args, "fix", None) or []):  # profile: hold a parameter fixed (tiny box)
        nm_, val_ = item.split("=")
        j_ = names.index(nm_)
        bounds[j_] = (float(val_) - 1e-9, float(val_) + 1e-9)
        x0[j_] = float(val_)
        rep.append(f"profile: {nm_} fixed at {float(val_)}")
    rng = np.random.default_rng(args.seed)
    results = {}
    init = json.loads(args.init_from.read_text())["results"] if args.init_from else None
    for wkey in args.wkeys:
        sols, best = [], None
        x0w = x0.copy()
        if init is not None:  # start from a previous best fit (the noW one by default) and scatter around it
            src = init.get(args.init_key or wkey) or init["noW"]
            th_src = dict(zip(src.get("names", names), src["theta"]))
            x0w = np.clip(np.array([th_src.get(n, x0[j]) for j, n in enumerate(names)], float), [b[0] for b in bounds], [b[1] for b in bounds])
        for s_ in range(args.n_starts):
            s0 = x0w if s_ == 0 else np.array([rng.uniform(max(lo, v - 1), min(hi, v + 1)) if j != i_ns else rng.uniform(0.5 * v, 1.5 * v)
                                              for j, (v, (lo, hi)) in enumerate(zip(x0w, bounds))])
            r = minimize(nll, s0, args=(wkey,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 4000, "maxfun": 40000})
            r = minimize(nll, r.x, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
            # guard (#484): L-BFGS-B can end above its start (1e30 walls); also try Powell from the start and keep the best
            r2 = minimize(nll, s0, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
            if r2.fun < r.fun:
                r = r2
            f_s0 = nll(s0, wkey)
            if f_s0 < r.fun:
                r.x, r.fun = np.asarray(s0, float).copy(), f_s0
            sols.append((float(r.fun), r.x.copy()))
            if best is None or r.fun < best.fun:
                best = r
        th = best.x
        h = 1e-3 * np.maximum(1.0, np.abs(th))
        H = np.zeros((n_par, n_par))
        f0 = nll(th, wkey)
        for i in range(n_par):
            for j in range(i, n_par):
                def ff(di, dj):
                    xx = th.copy()
                    xx[i] += di
                    xx[j] += dj
                    return nll(xx, wkey)
                H[i, j] = H[j, i] = ((ff(h[i], 0) - 2 * f0 + ff(-h[i], 0)) / h[i] ** 2 if i == j else
                                     (ff(h[i], h[j]) - ff(h[i], -h[j]) - ff(-h[i], h[j]) + ff(-h[i], -h[j])) / (4 * h[i] * h[j]))
        try:
            dg = np.diag(np.linalg.inv(H))
            err = np.sqrt(np.where(dg > 0, dg, np.nan))
        except np.linalg.LinAlgError:
            err = np.full(n_par, np.nan)
        w, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        # §12.20: relative MC error per orbit bin, draw-independent vs parent-row clustered, at the best fit
        so_ = ob >= 0
        vd = np.bincount(ob[so_], weights=w[so_] ** 2, minlength=n_ob)
        vr = clustered_bin_variance(w, grp_o, n_ob)
        nrow = np.bincount(grp_o[2], minlength=n_ob)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_d, rel_r = np.sqrt(vd) / mu_o, np.sqrt(vr) / mu_o
        idx = np.unravel_index(np.arange(n_ob), shp)
        bins_tab = [dict(g=int(idx[0][b]), d=int(idx[1][b]), lp=int(idx[2][b]), e=int(idx[3][b]), cls=int(idx[4][b]), k=float(k_o[b]),
                         mu=float(mu_o[b]), rel_err_draw=float(rel_d[b]), rel_err_row=float(rel_r[b]), n_rows=int(nrow[b]),
                         z_pois=float(np.sqrt(k_o[b]) / max(k_o[b], 1)))
                    for b in range(n_ob) if mu_o[b] > 0]
        s = th[i_ns] * T + floor_o
        eb = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
        near = [f_ - best.fun for f_, _ in sols]
        at_b = [names[i] for i in range(n_par) if min(abs(th[i] - bounds[i][0]), abs(th[i] - bounds[i][1])) < 1e-3 * max(1, abs(th[i]))]
        so = ob >= 0
        evo_share = float(w[so & (evo > 0)].sum() / w[so].sum())
        evo_share_s = float((w[so & (evo > 0)].sum() + s.reshape(shp)[..., 1].sum()) / (w[so].sum() + s.sum()))
        rdn = np.histogram(ns.ro_cols[1], bins=[0, 0.3, 0.7, 1.5, 6])[0].astype(float)
        sd_ = np.bincount(rob[rob >= 0], minlength=n_ob)  # unused
        # spurious per real distance bin: sum over bins of s weighted by the real-orbit share in each d interval
        rd = ns.ro_cols[1]
        s_per_orbit = np.zeros(rd.size)
        okr = rob >= 0
        s_per_orbit[okr] = (s / np.maximum(k_o, 1))[rob[okr]]
        sdn = np.array([s_per_orbit[(rd >= a) & (rd < b)].sum() for a, b in zip([0, 0.3, 0.7, 1.5], [0.3, 0.7, 1.5, 6])])
        mdn = np.histogram(ns.o_cols[1][so], bins=[0, 0.3, 0.7, 1.5, 6], weights=w[so])[0]
        radn = np.histogram(ns.ra_cols[1], bins=[0, 0.3, 0.7, 1.5, 6])[0].astype(float)
        sa = ab >= 0
        madn = np.histogram(ns.a_cols[1][sa], bins=[0, 0.3, 0.7, 1.5, 6], weights=w[sa])[0]
        gb = np.array([5, 9, 10, 12, 14, 15, 16, 19.5])
        rgn = np.histogram(ns.ro_cols[0], bins=gb)[0].astype(float)
        mgn = np.histogram(ns.o_cols[0][so], bins=gb, weights=w[so])[0]
        sgn = np.array([s_per_orbit[(ns.ro_cols[0] >= a) & (ns.ro_cols[0] < b)].sum() for a, b in zip(gb[:-1], gb[1:])])
        results[wkey] = dict(theta=th.tolist(), err=err.tolist(), names=list(names), nll=float(best.fun), at_bound=at_b,
                             n_near=int(np.sum(np.array(near) < 5)), n_starts=len(sols), start_offsets=sorted(round(x, 2) for x in near),
                             N_s=float(th[i_ns]), N_s_pull=float((th[i_ns] - prior[0]) / prior[1]),
                             total=float((mu_o.sum() + s.sum()) / N_o), accel=float((mu_a.sum() + floor_a.sum()) / k_a.sum()),
                             evolved_share_mock=evo_share, evolved_share_with_spurious=evo_share_s, evolved_share_real=float(ns.r_cls.mean()),
                             d_ratio=((mdn + sdn) / rdn).tolist(), d_ratio_mock_only=(mdn / rdn).tolist(), d_orbit_accel=((mdn + sdn) / madn).tolist(),
                             d_orbit_accel_real=(rdn / radn).tolist(), g_ratio=((mgn + sgn) / rgn).tolist(), g_edges=gb.tolist(),
                             ess_orbits=float(ps.kish_ess(w[so])), n_big_low=int(np.sum(big & (eb < args.min_ess))), n_big=int(big.sum()),
                             ess_b_big_min=float(eb[big].min()), ess_b_big_median=float(np.median(eb[big])))
        results[wkey]["bins"] = bins_tab
        R5 = results[wkey]
        rep.append(f"[{wkey}] {R5['n_near']}/{R5['n_starts']} starts within delta(-lnL) < 5 (offsets {R5['start_offsets']}); -lnL {best.fun:.1f}; at bound: {at_b or 'none'}")
        for nm, v, e_ in zip(names, th, err):
            rep.append(f"    {nm:10s} {v:+10.3f} +- {e_:.3f}")
        rep.append(f"    N_s {th[i_ns]:.0f} vs prior {prior[0]:.0f} +- {prior[1]:.0f} (pull {R5['N_s_pull']:+.1f}); (mock + spurious)/real {R5['total']:.3f}; accelerations {R5['accel']:.3f}")
        rep.append(f"    evolved share of C1 orbits: real {R5['evolved_share_real']:.3f}, mock {evo_share:.3f} (with spurious {evo_share_s:.3f})")
        rep.append("    d bins <0.3 / 0.3-0.7 / 0.7-1.5 / >1.5: (mock+spur)/real " + " / ".join(f"{x:.2f}" for x in R5["d_ratio"])
                   + "; mock-only " + " / ".join(f"{x:.2f}" for x in R5["d_ratio_mock_only"])
                   + "; orbit:accel real " + " / ".join(f"{x:.3f}" for x in R5["d_orbit_accel_real"]) + " fit " + " / ".join(f"{x:.3f}" for x in R5["d_orbit_accel"]))
        rep.append("    G bins " + " ".join(f"{a:g}-{b:g}:{x:.2f}" for a, b, x in zip(gb[:-1], gb[1:], R5["g_ratio"])))
        rep.append(f"    ESS orbits {R5['ess_orbits']:.0f}; big bins below ESS_b {args.min_ess:.0f}: {R5['n_big_low']}/{R5['n_big']} (min {R5['ess_b_big_min']:.1f}, median {R5['ess_b_big_median']:.0f})")
        rep.append("")
        if wkey == "noW":
            # e < 0.1 count (six-panel fine bins): mock + spurious share of that e range
            emask = so & (ns.casc[:, 14] < 0.1)
            se = s_per_orbit[np.nan_to_num(ns.ro_cols[3], nan=9) < 0.1].sum()
            R5["e_below_0p1"] = {"real": float(np.sum(ns.ro_cols[3] < 0.1)), "fit": float(w[emask].sum() + se)}
            rep.append(f"    e < 0.1: real {R5['e_below_0p1']['real']:.0f}, fit {R5['e_below_0p1']['fit']:.0f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps({"model": args.model, "label": fc.get("label"), "results": results,
                                                            "bins": {"orbit": [e.tolist() for e in oe], "accel": [e.tolist() for e in ae]}}, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def feasibility(args, fc, ns):
    """§12.14 compliant grids + class axis at this fit's θ (noW, cmdW) and MdS17; draw cost."""
    res = json.loads((args.theta_summary or (args.out_dir / "summary.json")).read_text())["results"]
    lm = {}
    for k in ("noW", "cmdW"):
        th = np.array(res[k]["theta"])
        names = res[k]["names"]
        t = np.zeros(14)
        t[:13] = th[:13]
        lm[k] = ns.log_mod(t) + (th[names.index("ln_f_evo")] * ns.m_cls if "ln_f_evo" in names else 0.0)
    th_mds = np.array([0, 0.5, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1, 1, 1, 0], float)
    pts = {"fit noW": ns.W["noW"] * np.exp(lm["noW"]), "fit cmdW": ns.W["cmdW"] * np.exp(lm["cmdW"]), "MdS17": ns.W["noW"] * np.exp(ns.log_mod(th_mds))}
    w30 = ns.W["g28"] * np.exp(lm["noW"])
    rep = ["#391 §12.14 feasibility (+ CMD-class axis) at the refit-5 best fit", ""]
    ae = [np.asarray(fc["outer"]["g_mag"]), np.asarray(fc["outer"]["distance_kpc"])]
    for combo in itertools.product(*(COMPLIANT[a] for a in r4.AXES)):
        oe = [np.asarray(c, float) for c in combo]
        ob, _, k_o, _, n_ob, _, _ = binned5(ns, oe, ae)
        big = k_o >= 0.01 * k_o.sum()
        emin = np.minimum.reduce([r4.ess_b(w, ob, n_ob) for w in pts.values()])
        fail = big & (emin < args.min_ess)
        fac = float(np.max(args.min_ess / np.maximum(emin[big], 1e-9)))
        e_all = min(ns.ps.kish_ess(w[ob >= 0]) for w in pts.values())
        rate = ns.ps.kish_ess(w30[ob >= 0]) / ns.cpu_28
        e30 = r4.ess_b(w30, ob, n_ob) / ns.cpu_28
        with np.errstate(divide="ignore", invalid="ignore"):
            cell = np.where(fail, (args.min_ess - emin) / e30, 0.0)
        shp5 = [len(c) - 1 for c in combo] + [2]
        cells = []
        for b_ in np.flatnonzero(fail):
            i = np.unravel_index(b_, shp5)
            cells.append("G {:g}-{:g} d {:g}-{:g} lP {:g}-{:.2f} e {:g}-{:g} {}: real {}, ESS {:.1f}".format(
                combo[0][i[0]], combo[0][i[0] + 1], combo[1][i[1]], combo[1][i[1] + 1], combo[2][i[2]], combo[2][i[2] + 1],
                combo[3][i[3]], combo[3][i[3] + 1], "evolved" if i[4] else "dwarf", int(k_o[b_]), emin[b_]))
        rep.append(f"grid {[len(c) - 1 for c in combo]} x class ({n_ob} bins, {int(big.sum())} big): failing {int(fail.sum())} "
                   f"({k_o[fail].sum() / k_o.sum():.1%} of real), min big ESS_b {emin[big].min():.1f}; cost ~{max(0, fac - 1) * e_all / rate:.0f} CPU-h "
                   f"(total-ESS scaling), worst per-cell {np.nanmax(np.where(np.isfinite(cell), cell, np.nan)) if fail.any() else 0:.0f} CPU-h")
        rep += [f"    failing: {c_}" for c_ in cells]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feasibility.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("choose", "fit", "feasibility"))
    ap.add_argument("--model", choices=("old", "new"), default="new")
    ap.add_argument("--extrapolation", choices=("clamp", "logit"), default=None,
                    help="MP-Q46; default from the fit config key spurious_extrapolation (decided: clamp, Ryan 2026-10-11)")
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_30.npz"))
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--control", type=Path, default=Path("docs/gate391/rung3_refit4/gens23_30_gen29_fine_grid/summary.json"))
    ap.add_argument("--calibration", type=Path, default=Path("docs/gate391/rung3_modelfix2/spurious_calibration.json"))
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--base-config", type=Path, default=Path("config/population/rung3_refit3_base.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_compliant108_v6a.yaml"))  
    ap.add_argument("--new-gen", type=int, default=30)
    ap.add_argument("--theta-summary", type=Path, default=None, help="feasibility: θ from this summary.json")
    ap.add_argument("--artifacts", nargs="+", default=None, help="artifact file names in cache order (default: gens 23-30)")
    ap.add_argument("--min-ess", type=float, default=30.0)
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--wkeys", nargs="+", default=["noW", "cmdW"], choices=["noW", "cmdW"])
    ap.add_argument("--init-from", type=Path, default=None, help="summary.json whose best θ seeds the starts")
    ap.add_argument("--init-key", default=None, help="which weighting's θ in --init-from to start from (default: same, else noW)")
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--m-break", type=float, default=None, help="§12.19(b) M_br override (profile diagnostic only)")
    ap.add_argument("--lowmass-correction-off", action="store_true", help="§12.25: fit generations made on uncorrected MIST (gens 23-32)")
    ap.add_argument("--fix", nargs="*", default=None, help="NAME=VALUE parameters held fixed (profiles)")
    ap.add_argument("--below-envelope", choices=("none", "hold", "orb6"), default=None, help="§12.23: enable MP-Q27 with this below-envelope choice")
    ap.add_argument("--cmd-conditioning", choices=("on", "off"), default=None, help="§12.22 override of rung3.cmd_binary_conditioning.enabled")
    ap.add_argument("--mc-variance", choices=("draw", "parent_row"), default=None, help="§12.20 override of rung3.mc_variance")
    ap.add_argument("--mass-scale", choices=("mist", "pm13_colour"), default=None, help="§12.19(c) override of mds17_mass_scale.method")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/gate391/rung3_modelfix2/interim_new"))
    args = ap.parse_args(argv)
    import yaml

    fc = yaml.safe_load((args.base_config if args.mode == "choose" else args.fit_config).read_text())["rung3"]
    if args.extrapolation is None:
        args.extrapolation = fc.get("spurious_extrapolation", "clamp")
    ns = extend(args, fc, r4.setup(args, fc))
    {"choose": choose, "fit": fit, "feasibility": feasibility}[args.mode](args, fc, ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
