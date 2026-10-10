#!/usr/bin/env python3
"""Rung 3 refit 4 (#391; spec §12.12/§12.13): the refit-3 machinery on gens 23-29 (staged top-up), ESS before/after.

``choose``: evaluate the §12.12 candidate edge sets by ESS_b of the noW mock weights at the refit-2 best
fit and at MdS17 (no real-data residuals involved) and write the chosen edges to the refit-3 config.
``fit``: the §12.11 model (normalized q modifiers, Beta eccentricities per log P range, additive
spurious N_s · T with T_GD ∝ real (G, d) counts) on the chosen bins, bins without any mock draw at a
fixed spurious floor; noW and cmdW, multi-start; posterior-predictive checks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rung3_fit_391 import PANELS, bin_index, say_loglike  # noqa: E402
from rung3_refit_391 import MOCK_SYM_OFFSET, mixture_weights  # noqa: E402

NAMES = ("ln_A", "alpha_lo", "alpha_hi", "gamma_P", "ln_L_P", "dgamma_q", "ln_F_twin",
         "a1", "a2", "a3", "b1", "b2", "b3", "N_s")
AXES = ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")


def setup(args, fc):
    """Mock and real arrays independent of the binning (outer edges from ``fc['outer']``)."""
    import yaml  # noqa: F401
    from astropy.table import Table

    from darkhunter_pop import moe_distefano as mds
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    ns = SimpleNamespace()
    cfg = load_config()
    gm = import_gaiamock_mod()
    frag = ps.load_proposal_set_fragment(args.fragment)
    prop, tgt = frag.proposal, frag.target_mds17
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    table = mds.load_mds17_table(tgt.table_path)
    z = np.load(args.cache, allow_pickle=True)
    PMAX = float(fc["c1_period_max_days"])
    out = {k: np.asarray(v, float) for k, v in fc["outer"].items()}  # [lo, hi] per axis
    casc = np.asarray(z["cascade"], float)
    acc = np.asarray(z["accepted"], bool)
    pub_acc = np.asarray(z["published_acceleration"], bool)
    m_g = np.asarray(z["phot_g_mean_mag"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_d_o, m_d_a, m_lp_o = 1.0 / casc[:, 0], 1.0 / casc[:, 2], np.log10(casc[:, 10])
    m_orb = acc & (casc[:, 10] <= PMAX)
    o_cols = [m_g, m_d_o, m_lp_o, casc[:, 14]]
    a_cols = [m_g, m_d_a]
    in_o = m_orb & (bin_index(o_cols, [out[a] for a in AXES]) >= 0)
    in_a = pub_acc & (bin_index(a_cols, [out["g_mag"], out["distance_kpc"]]) >= 0)
    rel = np.flatnonzero(in_o | in_a)
    ns.rel, ns.in_o, ns.in_a = rel, in_o[rel], in_a[rel]
    ns.o_cols = [c[rel] for c in o_cols]
    ns.a_cols = [c[rel] for c in a_cols]
    m1 = np.asarray(z["m1_msun"], float)[rel]
    q = np.asarray(z["m2_msun"], float)[rel] / m1
    P_true = np.asarray(z["period_days"], float)[rel]
    lp_true = np.log10(P_true)
    e_true = np.asarray(z["eccentricity"], float)[rel]
    s_low = mds.low_mass_frequency_scale(m1, m1_anchor_msun=tgt.provisional_low_mass_anchor_msun,
                                         m1_zero_msun=tgt.provisional_low_mass_zero_msun)
    m1s = np.clip(m1, table.m1_range[0], table.m1_range[1])
    eta0 = np.maximum(mds.eta(m1s, lp_true, table, m1_interpolation=tgt.provisional_m1_interpolation, eta_floor=-0.999), float(fc["eta_floor"]))
    emax = mds.e_max(P_true, table)
    circ = mds.is_circular(P_true, table)
    inside = ~circ & (e_true > 0) & (e_true < emax)
    x = np.clip(np.where(inside, e_true / np.where(emax > 0, emax, 1.0), 0.5), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pe0 = np.where(inside, np.log(eta0 + 1.0) + eta0 * np.log(np.where(inside, e_true, 1.0)) - (eta0 + 1.0) * np.log(np.where(emax > 0, emax, 1.0)), 0.0)
    log_emax = np.log(np.where(emax > 0, emax, 1.0))
    rng_k = np.digitize(lp_true, np.asarray(fc["ecc_log_p_ranges"], float))
    lx, l1x = np.log(x), np.log1p(-x)
    lm1 = np.log(m1 / float(fc["m1_pivot_msun"]))
    lo_m = m1 < float(fc["m1_pivot_msun"])
    ng = int(fc["q_norm_grid_points"])
    qe = np.linspace(0.3, 1.0, ng + 1)
    qg = 0.5 * (qe[1:] + qe[:-1])
    Pq = mds.q_density(qg[None, :], m1s[:, None], lp_true[:, None], table, m1_interpolation=tgt.provisional_m1_interpolation) * np.diff(qe)[None, :]
    Pq = Pq / Pq.sum(axis=1, keepdims=True)
    lq_g, twin_g = np.log(qg), qg >= fc["twin_q_min"]

    def log_mod(th):
        lnA, alo, ahi, gP, lnL, dq, lnT = th[:7]
        a, b = np.asarray(th[7:10]), np.asarray(th[10:13])
        lm = lnA + np.where(lo_m, alo, ahi) * lm1
        lm += gP * (lp_true - fc["log_p_tilt_pivot"])
        lm += lnL / (1.0 + np.exp(-(lp_true - fc["long_p_centre"]) / fc["long_p_width"]))
        lm += dq * np.log(q) + lnT * (q >= fc["twin_q_min"]) - np.log(Pq @ np.exp(dq * lq_g + lnT * twin_g))
        ak, bk = a[rng_k], b[rng_k]
        lm += np.where(inside, (ak - 1.0) * lx + (bk - 1.0) * l1x - betaln(ak, bk) - log_emax - log_pe0, 0.0)
        return lm

    gens = [int(g) for g in z["gens"]]
    before = [g for g in gens if g != args.new_gen]
    ns.W = {"noW": mixture_weights(z, gens, False)[rel] / s_low, "cmdW": mixture_weights(z, gens, True)[rel] / s_low,
            "before_noW": mixture_weights(z, before, False)[rel] / s_low,
            "g28": mixture_weights(z, [args.new_gen], False)[rel] / s_low}
    ns.log_mod, ns.casc, ns.gen = log_mod, casc[rel], np.asarray(z["generation"])[rel]
    cpu, gen_all = np.asarray(z["cpu_seconds"], float), np.asarray(z["generation"])
    ns.cpu_all, ns.cpu_28 = float(cpu.sum() / 3600), float(cpu[gen_all == args.new_gen].sum() / 3600)
    ns.cpu_before = ns.cpu_all - ns.cpu_28

    # real
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr_p > 20)]) for a, b in zip(gedges[:-1], gedges[1:])])

    def col(tab, name):
        return np.ma.filled(np.ma.asarray(tab[name], float), np.nan)

    def dist(plx, g):
        p = plx + zp[np.clip(np.digitize(g, gedges) - 1, 0, zp.size - 1)]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, 1.0 / p, np.nan), p

    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    r_g, r_P, r_e = col(rt, "g_mag"), col(rt, "period"), col(rt, "eccentricity")
    r_d, r_plx = dist(col(rt, "parallax"), r_g)
    r_a0, _, _, r_inc = (np.asarray(v, float) for v in gm.get_Campbell_elements(col(rt, "A"), col(rt, "B"), col(rt, "F"), col(rt, "G")))
    rc1 = r_P <= PMAX
    ro_cols = [r_g, r_d, np.log10(r_P), r_e]
    r_in = rc1 & (bin_index(ro_cols, [out[a] for a in AXES]) >= 0)
    ns.ro_cols = [c[r_in] for c in ro_cols]
    ns.r_six = {"P_orb_days": r_P[r_in], "G_mag": r_g[r_in], "inv_parallax_mas_inv": r_d[r_in], "eccentricity": r_e[r_in],
                "cos_inclination": np.cos(r_inc[r_in])}
    from darkhunter_pop.physics_utils import astrometric_mass_function

    with np.errstate(divide="ignore", invalid="ignore"):
        ns.r_six["f_m_msun"] = astrometric_mass_function(r_a0, r_plx, r_P)[r_in]
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    inp = {"source_id": np.asarray(at["source_id"], np.int64)}
    inp.update({c: col(at, c) for c in ("phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ipd_frac_multi_peak",
                                        "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")})
    keep_a, _ = ps.real_comparison_keep(at, prop, inp)
    at = at[keep_a]
    ra_g = col(at, "g_mag")
    ra_d, _ = dist(col(at, "parallax"), ra_g)
    ra_in = bin_index([ra_g, ra_d], [out["g_mag"], out["distance_kpc"]]) >= 0
    ns.ra_cols = [ra_g[ra_in], ra_d[ra_in]]
    ns.N_o = float(r_in.sum())
    ns.cfg, ns.ps, ns.mds, ns.table, ns.PMAX = cfg, ps, mds, table, PMAX
    return ns


def binned(ns, oe, ae):
    ob = np.where(ns.in_o, bin_index(ns.o_cols, oe), -1)
    ab = np.where(ns.in_a, bin_index(ns.a_cols, ae), -1)
    n_ob, n_ab = int(np.prod([e.size - 1 for e in oe])), int(np.prod([e.size - 1 for e in ae]))
    k_o = np.bincount(bin_index(ns.ro_cols, oe), minlength=n_ob).astype(float)
    k_a = np.bincount(bin_index(ns.ra_cols, ae), minlength=n_ab).astype(float)
    return ob, ab, k_o, k_a, n_ob, n_ab


def ess_b(w, idx, n):
    s = idx >= 0
    mu = np.bincount(idx[s], weights=w[s], minlength=n)
    v = np.bincount(idx[s], weights=w[s] ** 2, minlength=n)
    return np.where(v > 0, mu**2 / np.maximum(v, 1e-300), 0.0)


def choose(args, fc, ns) -> None:
    cand = fc["candidates"]
    ff2 = json.loads(args.theta_from.read_text())["noW"]["theta"]
    th_mds = np.array([0, 0.5, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1, 1, 1, 0], float)
    wts = {"refit3": ns.W["noW"] * np.exp(ns.log_mod(np.array(ff2, float))), "MdS17": ns.W["noW"] * np.exp(ns.log_mod(th_mds))}
    lines = []

    def score(kind, edges_list):
        rows = []
        for combo in itertools.product(*edges_list):
            e_ = [np.asarray(c, float) for c in combo]
            if kind == "orbit" and (len(combo[2]) < 3 or len(combo[3]) < 3):
                continue  # §12.12: >= 2 bins in log P and in e (identifiability of the P tilt and the Beta parameters)
            if kind == "orbit":
                ob, _, k, _, n, _ = binned(ns, e_, [np.asarray(fc["outer"]["g_mag"]), np.asarray(fc["outer"]["distance_kpc"])])
                idx = ob
            else:
                _, ab, _, k, _, n = binned(ns, [np.asarray(fc["outer"][a]) for a in AXES], e_)
                idx = ab
            ess = np.minimum(*(ess_b(w, idx, n) for w in wts.values()))
            big = k >= 0.01 * k.sum()
            ok_big = bool(np.all(ess[big] >= fc["min_ess_per_bin"]))
            share = float(k[ess >= fc["min_ess_per_bin"]].sum() / k.sum())
            rows.append((ok_big, n, share, combo, int(big.sum()), float(np.median(ess[k > 0])), int(np.sum((ess >= fc["min_ess_per_bin"]) & (k > 0))), int(np.sum(k > 0))))
        good = [r for r in rows if r[0]]
        if kind == "orbit" and any(len(r[3][0]) > 2 for r in good):  # §12.13: prefer a G split (Ryan, 2026-10-10)
            good = [r for r in good if len(r[3][0]) > 2]
        best = max(good, key=lambda r: (r[1], r[2])) if good else max(rows, key=lambda r: (r[2], r[1]))
        lines.append(f"{kind}: {len(rows)} eligible candidates, {len(good)} with every big bin at ESS_b >= {fc['min_ess_per_bin']}; chosen {best[1]} bins "
                     f"(big {best[4]}; non-empty bins at ESS >= 30: {best[6]}/{best[7]}; real share in them {best[2]:.3f}; median ESS_b {best[5]:.0f}): "
                     f"{[list(c) for c in best[3]]}")
        return best

    bo = score("orbit", [cand[a] for a in AXES])
    ba = score("acceleration", [cand["g_mag"], cand["distance_kpc"]])
    newcfg = dict(fc)
    newcfg["orbit_bins"] = {a: list(map(float, e)) for a, e in zip(AXES, bo[3])}
    newcfg["acceleration_bins"] = {"g_mag": list(map(float, ba[3][0])), "distance_kpc": list(map(float, ba[3][1]))}
    newcfg["bin_choice"] = {"rule": "spec §12.12/§12.13: >= 2 bins in log P and e; most bins with every >=1%-of-real bin at "
                                    "min-over-(refit-3 best fit, MdS17) ESS_b >= 30, preferring a G split; else max real share in "
                                    "ESS_b >= 30 bins", "report": lines}
    import yaml

    hdr = ("# Rung-3 REFIT 4 fine grid (#391; spec §12.13, Ryan 2026-10-10). orbit_bins / acceleration_bins were CHOSEN BY\n"
           "# `scripts/rung3_refit4_391.py choose` from ESS alone on gens 23-29 (before any fit residual), see bin_choice.\n")
    args.fit_config.write_text(hdr + yaml.safe_dump({"rung3": newcfg}, sort_keys=False))
    print("\n".join(lines))


def fit(args, fc, ns) -> None:
    import h5py

    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style, six_panel_bin_edges

    ps, cfg = ns.ps, ns.cfg
    oe = [np.asarray(fc["orbit_bins"][a], float) for a in AXES]
    ae = [np.asarray(fc["acceleration_bins"][a], float) for a in ("g_mag", "distance_kpc")]
    shp = [e.size - 1 for e in oe]
    ob, ab, k_o, k_a, n_ob, n_ab = binned(ns, oe, ae)
    N_o = float(k_o.sum())
    big = k_o >= 0.01 * N_o
    nm_o = np.bincount(ob[ob >= 0], minlength=n_ob)
    nm_a = np.bincount(ab[ab >= 0], minlength=n_ab)
    empty_o, empty_a = nm_o == 0, nm_a == 0
    phi = float(fc["empty_bin_floor"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # spurious shape (§12.11 / §12.12)
    sp = fc["spurious"]
    recs = [json.loads(line) for line in args.reinjection_log.read_text().splitlines()]
    real_r = [r for r in recs if not (10**15 <= r["source_id"] < MOCK_SYM_OFFSET + 10**7)]
    sym_r = [r for r in recs if MOCK_SYM_OFFSET <= r["source_id"] < MOCK_SYM_OFFSET + 10**7]

    def acc_c1(r):
        return bool(r["accepted"]) and r["period"] is not None and np.isfinite(r["period"]) and r["period"] <= ns.PMAX

    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        p_sid, p_P, p_e = h["published/source_id"][()], h["published/period"][()], h["published/eccentricity"][()]
    pos = {int(s): i for i, s in enumerate(p_sid)}
    rs = sorted({r["source_id"] for r in real_r})
    ra_acc = np.array([np.mean([acc_c1(r) for r in real_r if r["source_id"] == s]) for s in rs])
    ra_lp, ra_e = np.log10(np.array([p_P[pos[s]] for s in rs])), np.array([p_e[pos[s]] for s in rs])
    oc = np.asarray(np.load(args.old_cache, allow_pickle=True)["cascade"], float)
    ms = sorted({r["source_id"] for r in sym_r})
    mi = np.array([s - MOCK_SYM_OFFSET for s in ms])
    ms_acc = np.array([np.mean([acc_c1(r) for r in sym_r if r["source_id"] == s]) for s in ms])
    ms_lp, ms_e = np.log10(oc[mi, 10]), oc[mi, 14]
    pibar = max(0.0, (ms_acc.mean() - ra_acc.mean()) / ms_acc.mean())

    def marginal(rv, mv, edges):
        out = []
        rb_, mb_ = np.digitize(rv, edges) - 1, np.digitize(mv, edges) - 1
        for i in range(edges.size - 1):
            nr, nm = int(np.sum(rb_ == i)), int(np.sum(mb_ == i))
            if nr < sp["min_systems_per_marginal_bin"] or nm < sp["min_systems_per_marginal_bin"]:
                out.append(1.0)
                continue
            ar, am = ra_acc[rb_ == i].mean(), ms_acc[mb_ == i].mean()
            raw = max(0.0, (am - ar) / am) / pibar if pibar > 0 else 1.0
            out.append(1.0 + (raw - 1.0) * nr / (nr + sp["shrink_n0"]))
        return np.array(out)

    rr_P, rr_e = marginal(ra_lp, ms_lp, oe[2]), marginal(ra_e, ms_e, oe[3])
    rg, rd, rlp, re_ = ns.ro_cols
    mb = (rd >= sp["prior_bin_distance_kpc"][0]) & (rd <= sp["prior_bin_distance_kpc"][1]) & (rg >= 13.0) & (rg <= 16.0)
    k_pe = np.histogram2d(rlp[mb], re_[mb], bins=[oe[2], oe[3]])[0]
    T_PE = pibar * np.outer(rr_P, rr_e) * k_pe
    T_PE /= T_PE.sum()
    R4 = k_o.reshape(shp)
    T_GD = R4.sum(axis=(2, 3)) / N_o
    T = np.einsum("ab,cd->abcd", T_GD, T_PE).ravel()
    T = np.where(empty_o, 0.0, T)  # empty bins carry the fixed floor instead (§12.12)
    floor_o = np.where(empty_o, phi, 0.0)
    floor_a = np.where(empty_a, phi, 0.0)
    ns_mu, ns_sd = sp["prior_mean"] * N_o, sp["prior_sigma"] * N_o
    rep = [f"#391 rung 3 refit 4 (spec §12.13): §12.12 machinery, §12.11 model; gens 23-{args.new_gen}", ""]
    rep += [f"bins: orbits {n_ob} ({' x '.join(str(s) for s in shp)}) {[e.tolist() for e in oe]}; accelerations {n_ab} {[e.tolist() for e in ae]}"]
    rep += [f"  bin choice: {line}" for line in fc.get("bin_choice", {}).get("report", [])]
    rep += [f"real: {int(N_o)} C1 orbits, {int(k_a.sum())} accelerations; big orbit bins (>= 1% of real): {int(big.sum())}",
            f"spurious: pi_bar {pibar:.3f}; log P ratios {np.round(rr_P, 2).tolist()}; e ratios {np.round(rr_e, 2).tolist()}; "
            f"T_PE {np.round(T_PE, 3).tolist()}; T_GD = real (G, d) counts (EXTRAPOLATION); N_s ~ N({ns_mu:.0f}, {ns_sd:.0f}) = (0.21, 0.04) x N_o"]
    rep.append(f"empty bins (no mock draw; fixed floor {phi:g} count, spurious, theta-independent): orbits {int(empty_o.sum())} holding "
               f"{int(k_o[empty_o].sum())} real orbits; accelerations {int(empty_a.sum())} holding {int(k_a[empty_a].sum())}")
    for b_ in np.flatnonzero(empty_o & (k_o > 0)):
        i = np.unravel_index(b_, shp)
        rep.append(f"    orbit bin G {oe[0][i[0]]:g}-{oe[0][i[0] + 1]:g} d {oe[1][i[1]]:g}-{oe[1][i[1] + 1]:g} lP {oe[2][i[2]]:g}-{oe[2][i[2] + 1]:.2f} "
                   f"e {oe[3][i[3]]:g}-{oe[3][i[3] + 1]:g}: {int(k_o[b_])} real")
    rep.append("")

    def expected(th, wkey):
        w = ns.W[wkey] * np.exp(ns.log_mod(th))
        so, sa = ob >= 0, ab >= 0
        return (w, np.bincount(ob[so], weights=w[so], minlength=n_ob), np.bincount(ob[so], weights=w[so] ** 2, minlength=n_ob),
                np.bincount(ab[sa], weights=w[sa], minlength=n_ab), np.bincount(ab[sa], weights=w[sa] ** 2, minlength=n_ab))

    def nll(th, wkey):
        _, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        ll = say_loglike(k_o, mu_o + th[13] * T + floor_o, v_o) + say_loglike(k_a, mu_a + floor_a, v_a)
        ll += -0.5 * ((th[13] - ns_mu) / ns_sd) ** 2
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    x0 = np.array([par[n][0] for n in NAMES], float)
    x0[13] = ns_mu
    bounds = [(par[n][1], par[n][2]) for n in NAMES]
    bounds[13] = (0.0, 5.0 * ns_mu)
    prev = {"refit3": json.loads(args.refit3.read_text())}
    rng = np.random.default_rng(args.seed)
    results = {}
    for wkey in ("noW", "cmdW"):
        sols, best = [], None
        for i in range(args.n_starts):
            s0 = x0 if i == 0 else np.array([rng.uniform(max(lo, v - 1), min(hi, v + 1)) if j < 13 else rng.uniform(0.5 * v, 1.5 * v)
                                             for j, (v, (lo, hi)) in enumerate(zip(x0, bounds))])
            r = minimize(nll, s0, args=(wkey,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 4000, "maxfun": 40000})
            r = minimize(nll, r.x, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
            sols.append((float(r.fun), r.x.copy()))
            if best is None or r.fun < best.fun:
                best = r
        th = best.x
        n = th.size
        h = 1e-3 * np.maximum(1.0, np.abs(th))
        H = np.zeros((n, n))
        f0 = nll(th, wkey)
        for i in range(n):
            for j in range(i, n):
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
            err = np.full(n, np.nan)
        w, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        s = th[13] * T + floor_o
        eb = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
        eba = np.where(v_a > 0, mu_a**2 / np.maximum(v_a, 1e-300), 0.0)
        near = np.array([v for f_, v in sols if f_ - best.fun < 5.0])
        at_b = [NAMES[i] for i in range(n) if min(abs(th[i] - bounds[i][0]), abs(th[i] - bounds[i][1])) < 1e-3 * max(1, abs(th[i]))]
        spread = {NAMES[i]: float(np.ptp(near[:, i])) for i in range(n)} if near.size else {}
        results[wkey] = dict(theta=th.tolist(), err=err.tolist(), nll=float(best.fun), n_near=len(near), n_starts=len(sols), at_bound=at_b,
                             start_nll=sorted(round(f_ - best.fun, 2) for f_, _ in sols), near_spread=spread,
                             N_s=float(th[13]), N_s_pull=float((th[13] - ns_mu) / ns_sd),
                             total_ratio=float((mu_o.sum() + s.sum()) / N_o), accel_ratio=float((mu_a.sum() + floor_a.sum()) / k_a.sum()),
                             ess_orbits=float(ps.kish_ess(w[ob >= 0])), ess_accel=float(ps.kish_ess(w[ab >= 0])),
                             ess_b_orbit=eb.tolist(), ess_b_accel=eba.tolist(),
                             n_low=int(np.sum((eb < fc["min_ess_per_bin"]) & (k_o > 0))), n_big_low=int(np.sum((eb < fc["min_ess_per_bin"]) & big)))
        results[wkey]["_e"] = (w, mu_o, mu_a, s)
        rep.append(f"[{wkey}] {len(near)}/{len(sols)} starts within delta(-lnL) < 5 of the best (offsets {results[wkey]['start_nll']}); "
                   f"-lnL {best.fun:.1f}; at bound: {at_b or 'none'}")
        rep.append("    name        refit 4 +- Laplace  [near-best spread]   refit 3 (PR #465) +- Laplace")
        for i_, nm in enumerate(NAMES):
            pj = prev["refit3"][wkey]
            rep.append(f"    {nm:10s} {th[i_]:+9.3f} +- {err[i_]:.3f}  [{spread.get(nm, float('nan')):.3f}]   "
                       f"{pj['theta'][i_]:+9.3f} +- {pj['err'][i_]:.3f}")
        rep.append(f"    N_s {th[13]:.0f} vs prior {ns_mu:.0f} +- {ns_sd:.0f}: pull {(th[13] - ns_mu) / ns_sd:+.1f} sigma; spurious {th[13] / N_o:.3f} of C1 orbits "
                   f"(+ floor {floor_o.sum():.0f})")
        rep.append(f"    totals: (mock + spurious)/real orbits {(mu_o.sum() + s.sum()) / N_o:.3f}; accelerations {(mu_a.sum() + floor_a.sum()) / k_a.sum():.3f}")
        wb_ = ns.W["before_noW"] * np.exp(ns.log_mod(th))  # gens 23..new_gen-1, noW, at this fit's θ
        eb_before = ess_b(wb_, ob, n_ob)
        results[wkey]["ess_b_orbit_before"] = eb_before.tolist()
        results[wkey]["ess_orbits_before"] = float(ps.kish_ess(wb_[ob >= 0]))
        rep.append(f"    ESS_b at this best fit, gens 23-{args.new_gen - 1} -> 23-{args.new_gen} (noW weights): "
                   + "; ".join(f"bin {b_}: {eb_before[b_]:.1f} -> {eb[b_]:.1f}{' (big)' if big[b_] else ''}" for b_ in np.flatnonzero(k_o > 0)))
        rep.append(f"    ESS at best fit: orbits {results[wkey]['ess_orbits']:.0f} (before {results[wkey]['ess_orbits_before']:.0f}), accelerations {results[wkey]['ess_accel']:.0f}; "
                   f"orbit ESS_b (non-empty bins): median {np.median(eb[k_o > 0]):.0f}, min {eb[k_o > 0].min():.1f}; "
                   f"below 30: {results[wkey]['n_low']}/{int((k_o > 0).sum())} (big {results[wkey]['n_big_low']}/{int(big.sum())}); "
                   f"acceleration ESS_b median {np.median(eba[k_a > 0]):.0f}, min {eba[k_a > 0].min():.1f}")
        rep.append("")

    # ---------------- PPC (noW) ----------------
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    w, mu_o, mu_a, s = results["noW"]["_e"]
    th = np.array(results["noW"]["theta"])
    so = ob >= 0
    casc = ns.casc
    mo = {"P_orb_days": casc[so, 10], "G_mag": ns.o_cols[0][so], "inv_parallax_mas_inv": ns.o_cols[1][so],
          "eccentricity": casc[so, 14], "cos_inclination": np.cos(np.radians(casc[so, 16]))}
    from darkhunter_pop.physics_utils import astrometric_mass_function

    with np.errstate(divide="ignore", invalid="ignore"):
        mo["f_m_msun"] = astrometric_mass_function(casc[so, 17], casc[so, 0], casc[so, 10])
    S4 = s.reshape(shp)
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    e_low = {}
    for ax, name in zip(np.ravel(axs), PANELS):
        sc, lo, hi = axes_cfg[name]
        edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=20)
        cen = np.sqrt(edges[1:] * edges[:-1]) if sc == "log" else 0.5 * (edges[1:] + edges[:-1])
        rn = np.histogram(ns.r_six[name], bins=edges)[0]
        mh = np.histogram(mo[name], bins=edges, weights=w[so])[0].astype(float)
        sfine = np.zeros(cen.size)
        if name in ("G_mag", "inv_parallax_mas_inv", "P_orb_days", "eccentricity"):
            ax_i = {"G_mag": 0, "inv_parallax_mas_inv": 1, "P_orb_days": 2, "eccentricity": 3}[name]
            se = 10 ** oe[2] if name == "P_orb_days" else oe[ax_i]
            sv = S4.sum(axis=tuple(a for a in range(4) if a != ax_i))
            for a_, b_, val in zip(se[:-1], se[1:], sv):
                if sc == "log" and a_ > 0:
                    ov = np.clip(np.log10(np.minimum(edges[1:], b_)) - np.log10(np.maximum(edges[:-1], a_)), 0, None)
                else:
                    ov = np.clip(np.minimum(edges[1:], b_) - np.maximum(edges[:-1], a_), 0, None)
                if ov.sum() > 0:
                    sfine += val * ov / ov.sum()
        ax.step(cen, rn, where="mid", color="k", lw=2, label="real DR3 (C1)")
        ax.step(cen, mh + sfine, where="mid", color=series_style(1, style)["color"], lw=2,
                label="refit 3: mock + spurious" if sfine.any() else "refit 3: mock (no spurious term)")
        ax.step(cen, sfine, where="mid", color=series_style(2, style)["color"], ls=":", lw=2, label="spurious component") if sfine.any() else None
        if sc == "log":
            ax.set_xscale("log")
        apply_axes_style(ax, style, xlabel=name, ylabel="count per bin")
        ax.legend(fontsize=style.tick_label_fontsize)
        if name == "eccentricity":
            e_low = {"real": float(rn[cen < 0.1].sum()), "refit3": float((mh + sfine)[cen < 0.1].sum()), "mock_only": float(mh[cen < 0.1].sum())}
    fig.suptitle(f"Rung 3 refit 4 PPC (noW, gens 23-{args.new_gen}): {fc.get('label', '')}", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_six_panel.png", dpi=int(cfg.diagnostics.figure_dpi))
    rep += [f"PPC e < 0.1 (two fine bins): real {e_low['real']:.0f} | refit 4 {e_low['refit3']:.0f} (mock {e_low['mock_only']:.0f}) | "
            f"refit 3 {prev['refit3']['e_below_0p1'].get('refit3', float('nan')):.0f}"]
    # G shape (six-panel G, fine bins): real vs fit, ratio per 1-mag bin
    gedg = np.arange(5.0, 19.6, 1.0)
    rg_ = np.histogram(ns.r_six["G_mag"], bins=gedg)[0].astype(float)
    mg_ = np.histogram(mo["G_mag"], bins=gedg, weights=w[so])[0] + th[13] / N_o * rg_
    rep.append("PPC G shape ((mock + spurious)/real per 1-mag bin): " + " ".join(f"{a:.0f}-{a + 1:.0f}:{m_ / r_:.2f}" for a, m_, r_ in zip(gedg[:-1], mg_, rg_) if r_ > 200))

    # counts vs distance (fine edges for comparability; spurious share = N_s / N_o in every (G, d) region)
    dfine = np.array([0, 0.3, 0.7, 1.5, 6.0])
    r_dn = np.histogram(ns.ro_cols[1], bins=dfine)[0].astype(float)
    m_dn = np.histogram(ns.o_cols[1][so], bins=dfine, weights=w[so])[0]
    s_dn = th[13] / N_o * r_dn  # T_GD ∝ real counts
    ra_dn = np.histogram(ns.ra_cols[1], bins=dfine)[0].astype(float)
    sa = ab >= 0
    ma_dn = np.histogram(ns.a_cols[1][sa], bins=dfine, weights=w[sa])[0]
    rep.append("PPC per distance bin (noW): (mock + spurious)/real orbits | mock-only/real | accelerations | orbit:accel real / refit 4")
    for j in range(dfine.size - 1):
        rep.append(f"  d {dfine[j]:.1f}-{dfine[j + 1]:.1f}: {(m_dn[j] + s_dn[j]) / r_dn[j]:.2f} | {m_dn[j] / r_dn[j]:.2f} | {ma_dn[j] / ra_dn[j]:.2f} | "
                   f"{r_dn[j] / ra_dn[j]:.3f} / {(m_dn[j] + s_dn[j]) / ma_dn[j]:.3f}  [{int(r_dn[j])}]")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    xx = np.arange(dfine.size - 1)
    lbl = [f"{a:g}-{b:g}" for a, b in zip(dfine[:-1], dfine[1:])]
    axs[0].plot(xx, (m_dn + s_dn) / r_dn, "o-", lw=2, color=series_style(1, style)["color"], label="orbits: mock + spurious")
    axs[0].plot(xx, m_dn / r_dn, "s--", lw=2, color=series_style(2, style)["color"], label="orbits: mock only")
    axs[0].plot(xx, ma_dn / ra_dn, "^:", lw=2, color=series_style(3, style)["color"], label="accelerations")
    axs[0].axhline(1, color="0.4", ls=":")
    axs[0].set_xticks(xx, lbl)
    axs[0].set_ylim(0, 2.5)
    apply_axes_style(axs[0], style, xlabel="distance (kpc)", ylabel="refit 3 / real")
    axs[0].legend(fontsize=style.tick_label_fontsize)
    axs[1].plot(xx, r_dn / ra_dn, "o-", color="k", lw=2, label="real")
    axs[1].plot(xx, (m_dn + s_dn) / ma_dn, "s--", lw=2, color=series_style(1, style)["color"], label="refit 3")
    axs[1].set_xticks(xx, lbl)
    apply_axes_style(axs[1], style, xlabel="distance (kpc)", ylabel="orbits (C1) / accelerations")
    axs[1].legend(fontsize=style.tick_label_fontsize)
    ebo = np.array(results["noW"]["ess_b_orbit"])
    axs[2].plot(np.arange(n_ob)[k_o > 0], ebo[k_o > 0], "o", color=series_style(1, style)["color"], label="orbit bins")
    axs[2].plot(np.arange(n_ob)[big], ebo[big], "s", mfc="none", color="k", label=">= 1% of real orbits")
    axs[2].axhline(fc["min_ess_per_bin"], color="0.4", ls=":")
    axs[2].set_yscale("log")
    apply_axes_style(axs[2], style, xlabel="orbit bin index", ylabel="ESS_b at best fit (noW)")
    axs[2].legend(fontsize=style.tick_label_fontsize)
    fig.suptitle(f"Rung 3 refit 4 (noW, gens 23-{args.new_gen}): counts and orbit:acceleration vs distance; per-bin ESS", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_distance_and_ess.png", dpi=int(cfg.diagnostics.figure_dpi))

    # top-up projection (§12.8) at this fit
    e_new = ps.kish_ess(w[so])
    w28 = ns.W["g28"] * np.exp(ns.log_mod(th))
    e_28 = ps.kish_ess(w28[so])
    fac = fc["min_ess_per_bin"] / np.maximum(ebo[big], 1e-9)
    proj = {}
    for lab, f_ in (("median big bin", np.median(fac)), ("90% of big bins", np.percentile(fac, 90)), ("all big bins", fac.max())):
        extra = max(0.0, f_ - 1.0) * e_new
        proj[lab] = (float(f_), float(extra / max(e_28 / ns.cpu_28, 1e-9)))
        rep.append(f"top-up projection at this fit, {lab}: x{f_:.2f} ESS -> {proj[lab][1]:.0f} CPU-h at generation {args.new_gen}'s rate here "
                   f"({e_28 / ns.cpu_28:.1f} ESS/CPU-h; all gens {e_new / ns.cpu_all:.1f})")
    weak = np.flatnonzero(big & (ebo < fc["min_ess_per_bin"]))
    rep.append("ESS-limited big orbit bins at this fit (focus of any further sampling): " + ("; ".join(
        "G {:g}-{:g} d {:g}-{:g} lP {:g}-{:.2f} e {:g}-{:g}: real {}, ESS_b {:.1f}".format(
            oe[0][a], oe[0][a + 1], oe[1][b], oe[1][b + 1], oe[2][c], oe[2][c + 1], oe[3][d], oe[3][d + 1], int(k_o[i]), ebo[i])
        for i in weak for a, b, c, d in [np.unravel_index(i, shp)]) or "none"))
    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    out.update(names=list(NAMES), N_s_prior=[ns_mu, ns_sd], e_below_0p1=e_low, projection=proj,
               bins={"orbit": [e.tolist() for e in oe], "accel": [e.tolist() for e in ae]},
               empty={"orbit_bins": int(empty_o.sum()), "orbit_real": int(k_o[empty_o].sum()), "accel_bins": int(empty_a.sum()),
                      "accel_real": int(k_a[empty_a].sum())})
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("choose", "fit"))
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_29.npz"))
    ap.add_argument("--old-cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--reinjection-log", type=Path, default=Path("output/detection_vs_population_391/reinjection.jsonl"))
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--refit3", type=Path, default=Path("docs/gate391/rung3_refit3/summary.json"))
    ap.add_argument("--theta-from", type=Path, default=Path("docs/gate391/rung3_refit3/summary.json"))
    ap.add_argument("--new-gen", type=int, default=29)
    ap.add_argument("--artifact", type=Path, action="append", default=None, help="all generations, to build --cache if missing")
    ap.add_argument("--cmd-malmquist", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--base-config", type=Path, default=Path("config/population/rung3_refit3_base.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_refit4_fine.yaml"))
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/gate391/rung3_refit4"))
    args = ap.parse_args(argv)
    import yaml

    if not args.cache.exists():
        from rung3_refit_391 import build_cache

        from darkhunter_pop import proposal_set as ps
        from darkhunter_pop.config_loader import load_config

        cfg = load_config()
        prop = ps.load_proposal_set_fragment(args.fragment).proposal
        build_cache(args, cfg, ps.load_parent_snapshot(args.parent_dir, cfg, prop))
    if args.mode == "choose":
        fc = yaml.safe_load(args.base_config.read_text())["rung3"]
        choose(args, fc, setup(args, fc))
    else:
        fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
        fit(args, fc, setup(args, fc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
