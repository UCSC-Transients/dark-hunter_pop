#!/usr/bin/env python3
"""Rung 3 refit 2 (#391; spec §12.11): additive spurious component and Beta eccentricities, gens 23-28.

Model: §12.3 modifiers with the MP-Q42b q normalization, the eccentricity density replaced by a
Beta(a_k, b_k) on e / e_max(P) in three log P ranges (Kipping 2013; MdS17 e_max and the P <= 2 d
circular class kept), and an additive spurious Poisson component μ_s = N_s · T with a fixed shape
T = T_GD(G, d) · T_PE(log P, e) (T_PE from the PR #455 re-injection; T_GD from the real orbits'
(G, d) marginal, an extrapolation). Fits: noW baseline (N_s free), N_s fixed at its prior mean,
no spurious component, and cmdW (N_s free). Also reports the e-proposal coverage, per-bin ESS and
the top-up projection at the new best fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rung3_fit_391 import PANELS, bin_index, say_loglike  # noqa: E402
from rung3_refit_391 import MOCK_SYM_OFFSET, mixture_weights  # noqa: E402

NAMES = ("ln_A", "alpha_lo", "alpha_hi", "gamma_P", "ln_L_P", "dgamma_q", "ln_F_twin",
         "a1", "a2", "a3", "b1", "b2", "b3", "N_s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, action="append", required=True, help="gens 23-28 in order (for the e-proposal configs)")
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_28.npz"))
    ap.add_argument("--old-cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--reinjection-log", type=Path, default=Path("output/detection_vs_population_391/reinjection.jsonl"))
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--previous", type=Path, default=Path("docs/gate391/rung3_refit/summary.json"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_refit2.yaml"))
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import h5py
    import yaml
    from astropy.table import Table

    from darkhunter_pop import moe_distefano as mds
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
    from darkhunter_pop.physics_utils import astrometric_mass_function
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style, six_panel_bin_edges

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    cfg = load_config()
    gm = import_gaiamock_mod()
    frag = ps.load_proposal_set_fragment(args.fragment)
    prop, tgt = frag.proposal, frag.target_mds17
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    table = mds.load_mds17_table(tgt.table_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.cache, allow_pickle=True)
    PMAX = float(fc["c1_period_max_days"])
    oe = [np.asarray(fc["orbit_bins"][k], float) for k in ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")]
    ae = [np.asarray(fc["acceleration_bins"][k], float) for k in ("g_mag", "distance_kpc")]
    shp = [e.size - 1 for e in oe]
    n_ob, n_ab = int(np.prod(shp)), (ae[0].size - 1) * (ae[1].size - 1)
    gens = [int(g) for g in z["gens"]]
    old_gens = [g for g in gens if g != 28]
    e_ranges = np.asarray(fc["ecc_log_p_ranges"], float)  # interior edges, e.g. [2.0, 2.6]

    # ---------------- mock ----------------
    casc = np.asarray(z["cascade"], float)
    acc = np.asarray(z["accepted"], bool)
    pub_acc = np.asarray(z["published_acceleration"], bool)
    m_g = np.asarray(z["phot_g_mean_mag"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_d_o, m_d_a, m_lp_o = 1.0 / casc[:, 0], 1.0 / casc[:, 2], np.log10(casc[:, 10])
    m_orb = acc & (casc[:, 10] <= PMAX)
    mob = np.where(m_orb, bin_index([m_g, m_d_o, m_lp_o, casc[:, 14]], oe), -1)
    mab = np.where(pub_acc, bin_index([m_g, m_d_a], ae), -1)
    rel = np.flatnonzero((mob >= 0) | (mab >= 0))
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
    rng_k = np.digitize(lp_true, e_ranges)
    lx, l1x = np.log(x), np.log1p(-x)
    piv = float(fc["m1_pivot_msun"])
    lm1 = np.log(m1 / piv)
    lo_m = m1 < piv
    ob_r, ab_r = mob[rel], mab[rel]
    ng = int(fc["q_norm_grid_points"])
    qe = np.linspace(0.3, 1.0, ng + 1)
    qg = 0.5 * (qe[1:] + qe[:-1])
    Pq = mds.q_density(qg[None, :], m1s[:, None], lp_true[:, None], table, m1_interpolation=tgt.provisional_m1_interpolation) * np.diff(qe)[None, :]
    Pq = Pq / Pq.sum(axis=1, keepdims=True)
    lq_g, twin_g = np.log(qg), qg >= fc["twin_q_min"]

    def log_mod(th: np.ndarray) -> np.ndarray:
        lnA, alo, ahi, gP, lnL, dq, lnT = th[:7]
        a, b = th[7:10], th[10:13]
        lm = lnA + np.where(lo_m, alo, ahi) * lm1
        lm += gP * (lp_true - fc["log_p_tilt_pivot"])
        lm += lnL / (1.0 + np.exp(-(lp_true - fc["long_p_centre"]) / fc["long_p_width"]))
        lm += dq * np.log(q) + lnT * (q >= fc["twin_q_min"]) - np.log(Pq @ np.exp(dq * lq_g + lnT * twin_g))
        ak, bk = a[rng_k], b[rng_k]
        log_beta = (ak - 1.0) * lx + (bk - 1.0) * l1x - betaln(ak, bk) - log_emax
        lm += np.where(inside, log_beta - log_pe0, 0.0)
        return lm

    W = {k: mixture_weights(z, gl, uw)[rel] / s_low for k, gl, uw in
         (("old_noW", old_gens, False), ("noW", gens, False), ("cmdW", gens, True), ("g28_noW", [28], False))}

    # ---------------- real ----------------
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr_p > 20)]) for a, b in zip(gedges[:-1], gedges[1:])])

    def col(tab, name):
        return np.ma.filled(np.ma.asarray(tab[name], float), np.nan)

    def zplx(plx, g):
        return plx + zp[np.clip(np.digitize(g, gedges) - 1, 0, zp.size - 1)]

    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    r_g, r_P, r_e = col(rt, "g_mag"), col(rt, "period"), col(rt, "eccentricity")
    r_plx = zplx(col(rt, "parallax"), r_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_d = np.where(r_plx > 0, 1.0 / r_plx, np.nan)
    r_a0, _, _, r_inc = (np.asarray(v, float) for v in gm.get_Campbell_elements(col(rt, "A"), col(rt, "B"), col(rt, "F"), col(rt, "G")))
    rob = np.where(r_P <= PMAX, bin_index([r_g, r_d, np.log10(r_P), r_e], oe), -1)
    k_o = np.bincount(rob[rob >= 0], minlength=n_ob).astype(float)
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    inp = {"source_id": np.asarray(at["source_id"], np.int64)}
    inp.update({c: col(at, c) for c in ("phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ipd_frac_multi_peak",
                                        "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")})
    keep_a, _ = ps.real_comparison_keep(at, prop, inp)
    at = at[keep_a]
    ra_g = col(at, "g_mag")
    with np.errstate(divide="ignore", invalid="ignore"):
        ra_d = np.where(zplx(col(at, "parallax"), ra_g) > 0, 1.0 / zplx(col(at, "parallax"), ra_g), np.nan)
    rab = bin_index([ra_g, ra_d], ae)
    k_a = np.bincount(rab[rab >= 0], minlength=n_ab).astype(float)
    N_o = float(k_o.sum())
    big = k_o >= 0.01 * N_o
    R4 = k_o.reshape(shp)

    # ---------------- spurious shape T (§12.11) ----------------
    sp = fc["spurious"]
    recs = [json.loads(line) for line in args.reinjection_log.read_text().splitlines()]
    real_r = [r for r in recs if not (10**15 <= r["source_id"] < MOCK_SYM_OFFSET + 10**7)]
    sym_r = [r for r in recs if MOCK_SYM_OFFSET <= r["source_id"] < MOCK_SYM_OFFSET + 10**7]

    def acc_c1(r):
        return bool(r["accepted"]) and r["period"] is not None and np.isfinite(r["period"]) and r["period"] <= PMAX

    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        p_sid, p_P, p_e = h["published/source_id"][()], h["published/period"][()], h["published/eccentricity"][()]
    pos = {int(s): i for i, s in enumerate(p_sid)}
    rs = sorted({r["source_id"] for r in real_r})
    ra_acc = np.array([np.mean([acc_c1(r) for r in real_r if r["source_id"] == s]) for s in rs])
    ra_lp = np.log10(np.array([p_P[pos[s]] for s in rs]))
    ra_e = np.array([p_e[pos[s]] for s in rs])
    oc = np.asarray(np.load(args.old_cache, allow_pickle=True)["cascade"], float)
    ms = sorted({r["source_id"] for r in sym_r})
    mi = np.array([s - MOCK_SYM_OFFSET for s in ms])
    ms_acc = np.array([np.mean([acc_c1(r) for r in sym_r if r["source_id"] == s]) for s in ms])
    ms_lp, ms_e = np.log10(oc[mi, 10]), oc[mi, 14]
    A_r, A_m = ra_acc.mean(), ms_acc.mean()
    pibar = max(0.0, (A_m - A_r) / A_m)

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
    pi_pe = pibar * np.outer(rr_P, rr_e)
    dsel = (oe[1][:-1] >= sp["prior_bin_distance_kpc"][0] - 1e-9) & (oe[1][1:] <= sp["prior_bin_distance_kpc"][1] + 1e-9)
    gsel = (oe[0][:-1] >= sp["prior_bin_g_mag"][0] - 1e-9) & (oe[0][1:] <= sp["prior_bin_g_mag"][1] + 1e-9)
    k_pe_bin = R4[gsel][:, dsel].sum(axis=(0, 1))  # real counts per (log P, e) cell inside the measured bin
    T_PE = pi_pe * k_pe_bin
    T_PE /= T_PE.sum()
    T_GD = R4.sum(axis=(2, 3))
    T_GD /= T_GD.sum()
    T = np.einsum("ab,cd->abcd", T_GD, T_PE).ravel()
    prior_mask = np.einsum("a,b,c,d->abcd", gsel, dsel, np.ones(shp[2]), np.ones(shp[3])).ravel() > 0
    k_prior, T_prior = float(k_o[prior_mask].sum()), float(T[prior_mask].sum())
    ns_mu, ns_sd = sp["prior_mean"] * k_prior / T_prior, sp["prior_sigma"] * k_prior / T_prior
    rep = ["#391 rung 3 refit 2 (spec §12.11): additive spurious component, Beta eccentricities; gens 23-28", "",
           f"real: {int(N_o)} C1 orbits / {int(k_a.sum())} accelerations; mock draws in the likelihood: {rel.size}",
           f"spurious shape: pi_bar {pibar:.3f} (A_real {A_r:.3f}, A_mock {A_m:.3f}); log P ratios {np.round(rr_P, 2).tolist()}; "
           f"e ratios {np.round(rr_e, 2).tolist()}; T_PE {np.round(T_PE, 3).tolist()}",
           f"  T_GD = real C1 orbits' (G, d) marginal: EXTRAPOLATION outside 0.7-1.5 kpc / G 12-16 (the 300 systems carry no G, d information)",
           f"  prior bin: k = {int(k_prior)}, T mass {T_prior:.4f} -> N_s ~ N({ns_mu:.0f}, {ns_sd:.0f}) (= 0.21 +- 0.04 of the bin's real orbits), fixed once", ""]

    def expected(th, wkey):
        w = W[wkey] * np.exp(log_mod(th))
        so_, sa_ = ob_r >= 0, ab_r >= 0
        return (w, np.bincount(ob_r[so_], weights=w[so_], minlength=n_ob), np.bincount(ob_r[so_], weights=w[so_] ** 2, minlength=n_ob),
                np.bincount(ab_r[sa_], weights=w[sa_], minlength=n_ab), np.bincount(ab_r[sa_], weights=w[sa_] ** 2, minlength=n_ab))

    def nll(th, wkey, use_prior=True):
        _, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        ll = say_loglike(k_o, mu_o + th[13] * T, v_o) + say_loglike(k_a, mu_a, v_a)
        if use_prior:
            ll += -0.5 * ((th[13] - ns_mu) / ns_sd) ** 2
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    x0 = np.array([par[n][0] for n in NAMES], float)
    x0[13] = ns_mu
    bounds = [(par[n][1], par[n][2]) for n in NAMES]
    bounds[13] = (0.0, 5.0 * ns_mu)
    prev = json.loads(args.previous.read_text())
    prev_names = prev["names"]
    cases = (("noW", "noW", "free"), ("noW_Ns_fixed", "noW", "fixed"), ("noW_no_spurious", "noW", "zero"), ("cmdW", "cmdW", "free"))
    results = {}
    rng = np.random.default_rng(args.seed)
    so = ob_r >= 0
    for case, wkey, mode in cases:
        free = np.ones(len(NAMES), bool)
        xfix = x0.copy()
        if mode != "free":
            free[13] = False
            xfix[13] = ns_mu if mode == "fixed" else 0.0
        b_free = [bounds[i] for i in np.flatnonzero(free)]

        def f_obj(xf):
            th = xfix.copy()
            th[free] = xf
            return nll(th, wkey, use_prior=(mode == "free"))

        sols, best = [], None
        x_start = xfix[free]
        n_st = args.n_starts if case == "noW" else max(4, args.n_starts // 3)
        for i in range(n_st):
            s0 = x_start if i == 0 else np.array([rng.uniform(max(lo, v - 1), min(hi, v + 1)) if hi - lo < 1e3 else
                                                  rng.uniform(lo + 0.5 * v, min(hi, 1.5 * v + 1)) for v, (lo, hi) in zip(x_start, b_free)])
            r = minimize(f_obj, s0, method="L-BFGS-B", bounds=b_free, options={"maxiter": 4000, "maxfun": 40000})
            r = minimize(f_obj, r.x, method="Powell", bounds=b_free, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
            sols.append((float(r.fun), r.x.copy()))
            if best is None or r.fun < best.fun:
                best = r
        th = xfix.copy()
        th[free] = best.x
        xf = best.x
        n = xf.size
        h = 1e-3 * np.maximum(1.0, np.abs(xf))
        H = np.zeros((n, n))
        f0 = f_obj(xf)
        for i in range(n):
            for j in range(i, n):
                def ff(di, dj):
                    xx = xf.copy()
                    xx[i] += di
                    xx[j] += dj
                    return f_obj(xx)
                H[i, j] = H[j, i] = ((ff(h[i], 0) - 2 * f0 + ff(-h[i], 0)) / h[i] ** 2 if i == j else
                                     (ff(h[i], h[j]) - ff(h[i], -h[j]) - ff(-h[i], h[j]) + ff(-h[i], -h[j])) / (4 * h[i] * h[j]))
        err = np.full(len(NAMES), np.nan)
        try:
            dg = np.diag(np.linalg.inv(H))
            err[free] = np.sqrt(np.where(dg > 0, dg, np.nan))
        except np.linalg.LinAlgError:
            pass
        w, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        s = th[13] * T
        frac = s[prior_mask].sum() / k_prior
        ess_b = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
        dev = 2 * (say_loglike(k_o, np.where(k_o > 0, k_o, 1e-300), np.zeros(n_ob)) - say_loglike(k_o, mu_o + s, v_o))
        near = np.array([v for f_, v in sols if f_ - best.fun < 5.0])
        at_b = [NAMES[i] for i in np.flatnonzero(free) if min(abs(th[i] - bounds[i][0]), abs(th[i] - bounds[i][1])) < 1e-3 * max(1, abs(th[i]))]
        results[case] = dict(theta=th.tolist(), err=err.tolist(), nll=float(best.fun), converged=bool(best.success), at_bound=at_b,
                             spurious_total=float(s.sum()), spurious_frac_prior_bin=float(frac), ess_orbits=float(ps.kish_ess(w[so])),
                             ess_accel=float(ps.kish_ess(w[ab_r >= 0])), n_low_ess=int(np.sum(ess_b < fc["min_ess_per_bin"])),
                             n_big_low_ess=int(np.sum((ess_b < fc["min_ess_per_bin"]) & big)), deviance=float(dev),
                             mock_orbits=float(mu_o.sum()), mock_accel=float(mu_a.sum()), n_near=len(near), n_starts=len(sols))
        results[case]["_e"] = (w, mu_o, v_o, mu_a, v_a, s, ess_b)
        rep.append(f"[{case}] converged {best.success}; -lnL {best.fun:.1f}; {len(near)}/{len(sols)} starts within 5 of the best; at bound: {at_b or 'none'}")
        if case in ("noW", "cmdW"):
            pk = "new_noW" if case == "noW" else "new_cmdW"
            rep.append("    name        refit 2 +- Laplace     previous refit (PR #461)")
            for i_, nm in enumerate(NAMES):
                pv = (f"{prev[pk]['theta'][prev_names.index(nm)]:+.3f} +- {prev[pk]['err'][prev_names.index(nm)]:.3f}" if nm in prev_names else "-")
                rep.append(f"    {nm:10s} {th[i_]:+9.3f} +- {err[i_]:.3f}     {pv}")
        rep.append(f"    N_s {th[13]:.0f} (prior {ns_mu:.0f} +- {ns_sd:.0f}; pull {(th[13] - ns_mu) / ns_sd:+.1f} sigma); spurious share in the prior bin {frac:.3f}; "
                   f"of all C1 orbits {s.sum() / N_o:.3f}")
        rep.append(f"    totals: mock orbits {mu_o.sum():.0f} + spurious {s.sum():.0f} = {(mu_o.sum() + s.sum()) / N_o:.3f} x real; "
                   f"accelerations {mu_a.sum() / k_a.sum():.3f} x real; deviance {dev:.0f}")
        rep.append(f"    ESS at best fit: orbits {results[case]['ess_orbits']:.0f}, accelerations {results[case]['ess_accel']:.0f}; "
                   f"bins with ESS_b < {fc['min_ess_per_bin']}: {results[case]['n_low_ess']}/{n_ob} (big bins {results[case]['n_big_low_ess']}/{int(big.sum())})")
        rep.append("")
    nmock_b = np.bincount(ob_r[so], minlength=n_ob)
    empty = (k_o > 0) & (nmock_b == 0)
    rep.append(f"orbit bins with real orbits but no mock draw at all: {int(empty.sum())} holding {int(k_o[empty].sum())} real orbits "
               f"({k_o[empty].sum() / N_o:.2%}); without a spurious floor these bins have mu = 0, which is why the no-spurious case runs away")
    rep.append("model comparison (-lnL; the free case includes its N_s prior term): " + ", ".join(f"{c} {results[c]['nll']:.1f}" for c, _, _ in cases))

    # ---------------- e-proposal coverage of the fitted Beta (§12.11) ----------------
    thb = np.array(results["noW"]["theta"])
    eg = np.linspace(1e-6, 1, 20001)
    rep += ["", "e-proposal coverage at the noW best fit: max p/q and the second moment E_q[(p/q)^2] per generation and log P range"]
    for path in args.artifact:
        with h5py.File(path, "r") as hh:
            pcfg = ps.ProposalConfig.model_validate_json(hh.attrs["proposal_config_json"])
        line = []
        for k_, lp_ in enumerate((1.5, 2.3, 2.75)):
            Pd = 10**lp_
            Em = float(mds.e_max(Pd, table))
            ein = eg[eg < Em]
            pdf = np.exp((thb[7 + k_] - 1) * np.log(ein / Em) + (thb[10 + k_] - 1) * np.log1p(-ein / Em) - betaln(thb[7 + k_], thb[10 + k_])) / Em
            qd = np.exp(ps.log_q_ecc(ein, np.full_like(ein, Pd), pcfg.eccentricity))
            r_ = pdf / qd
            line.append(f"range {k_ + 1} (log P {lp_}): max {r_.max():.2f}, E[w^2] {np.trapz(pdf * r_, ein):.2f}")
        rep.append(f"  generation {pcfg.generation}: " + "; ".join(line))

    # ---------------- PPC (noW) ----------------
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    idx_o = rel[so]
    mo = {"P_orb_days": casc[idx_o, 10], "G_mag": m_g[idx_o], "inv_parallax_mas_inv": m_d_o[idx_o],
          "eccentricity": casc[idx_o, 14], "cos_inclination": np.cos(np.radians(casc[idx_o, 16]))}
    with np.errstate(divide="ignore", invalid="ignore"):
        mo["f_m_msun"] = astrometric_mass_function(casc[idx_o, 17], casc[idx_o, 0], casc[idx_o, 10])
        rv = {"P_orb_days": r_P, "G_mag": r_g, "inv_parallax_mas_inv": r_d, "eccentricity": r_e,
              "f_m_msun": astrometric_mass_function(r_a0, r_plx, r_P), "cos_inclination": np.cos(r_inc)}
    rin = rob >= 0
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    e_low = {}
    for ax, name in zip(np.ravel(axs), PANELS):
        sc, lo, hi = axes_cfg[name]
        edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=20)
        cen = np.sqrt(edges[1:] * edges[:-1]) if sc == "log" else 0.5 * (edges[1:] + edges[:-1])
        rn = np.histogram(rv[name][rin], bins=edges)[0]
        ax.step(cen, rn, where="mid", color="k", lw=2, label="real DR3 (C1)")
        for j, case in enumerate(("noW", "noW_Ns_fixed")):
            w, _, _, _, _, s, _ = results[case]["_e"]
            mh = np.histogram(mo[name], bins=edges, weights=w[so])[0].astype(float)
            if name in ("G_mag", "inv_parallax_mas_inv", "P_orb_days", "eccentricity") and s.sum() > 0:
                s4 = s.reshape(shp)
                ax_i = {"G_mag": 0, "inv_parallax_mas_inv": 1, "P_orb_days": 2, "eccentricity": 3}[name]
                se = 10 ** oe[2] if name == "P_orb_days" else oe[ax_i]
                sv = s4.sum(axis=tuple(a for a in range(4) if a != ax_i))
                for a_, b_, val in zip(se[:-1], se[1:], sv):
                    if sc == "log" and a_ > 0:
                        ov = np.clip(np.log10(np.minimum(edges[1:], b_)) - np.log10(np.maximum(edges[:-1], a_)), 0, None)
                    else:
                        ov = np.clip(np.minimum(edges[1:], b_) - np.maximum(edges[:-1], a_), 0, None)
                    if ov.sum() > 0:
                        mh += val * ov / ov.sum()
            st_ = series_style(j + 1, style)
            ax.step(cen, mh, where="mid", color=st_["color"], ls=st_["linestyle"], lw=2,
                    label="refit 2 + spurious (N_s free)" if case == "noW" else "refit 2 + spurious (N_s fixed at prior mean)")
            if name == "eccentricity":
                e_low[case] = float(mh[cen < 0.1].sum())
        if name == "eccentricity":
            e_low["real"] = float(rn[cen < 0.1].sum())
        if sc == "log":
            ax.set_xscale("log")
        apply_axes_style(ax, style, xlabel=name, ylabel="count per bin")
        ax.legend(fontsize=style.tick_label_fontsize)
    fig.suptitle("Rung 3 refit 2 PPC (noW, gens 23-28): Beta eccentricities, additive spurious component", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_six_panel.png", dpi=int(cfg.diagnostics.figure_dpi))
    pe_prev = prev.get("e_below_0p1", {})
    rep += ["", f"PPC e < 0.1 (two fine bins): real {e_low['real']:.0f} | refit 2 {e_low['noW']:.0f} (N_s-fixed case {e_low['noW_Ns_fixed']:.0f}) | "
            f"previous refit {pe_prev.get('refit', float('nan')):.0f}"]
    _, mu_o, _, mu_a, _, s, ess_b = results["noW"]["_e"]
    M = (mu_o + s).reshape(shp)
    Ra = k_a.reshape(ae[0].size - 1, ae[1].size - 1)
    Ma = mu_a.reshape(Ra.shape)
    ecen = 0.5 * (oe[3][1:] + oe[3][:-1])
    rep.append("PPC per distance bin (noW): (mock + spurious)/real orbits | accelerations | <e> real / refit 2 | orbit:accel real / refit 2")
    for j in range(shp[1]):
        ro, m_ = R4[:, j].sum(), M[:, j].sum()
        rep.append(f"  d {oe[1][j]:.1f}-{oe[1][j + 1]:.1f}: orbits {m_ / ro:.2f} [{int(ro)}]; accels {Ma[:, j].sum() / Ra[:, j].sum():.2f}; "
                   f"<e> {(R4[:, j].sum(axis=(0, 1)) * ecen).sum() / ro:.3f} / {(M[:, j].sum(axis=(0, 1)) * ecen).sum() / m_:.3f}; "
                   f"o:a {ro / Ra[:, j].sum():.3f} / {m_ / Ma[:, j].sum():.3f}")
    for i in range(shp[0]):
        rep.append(f"  G {oe[0][i]:.1f}-{oe[0][i + 1]:.1f}: orbits {M[i].sum() / R4[i].sum():.2f} [{int(R4[i].sum())}]; accels {Ma[i].sum() / Ra[i].sum():.2f}")

    # ---------------- per-bin ESS and top-up projection at the refit-2 best fit ----------------
    lm_b = log_mod(thb)
    cpu = np.asarray(z["cpu_seconds"], float)
    gen_all = np.asarray(z["generation"])
    cpu_old, cpu_28 = float(cpu[np.isin(gen_all, old_gens)].sum() / 3600), float(cpu[gen_all == 28].sum() / 3600)

    def ess_bins(wv):
        mu = np.bincount(ob_r[so], weights=wv[so], minlength=n_ob)
        v = np.bincount(ob_r[so], weights=wv[so] ** 2, minlength=n_ob)
        return np.where(v > 0, mu**2 / np.maximum(v, 1e-300), 0.0), ps.kish_ess(wv[so])

    eb_old, e_old = ess_bins(W["old_noW"] * np.exp(lm_b))
    eb_new, e_new = ess_bins(W["noW"] * np.exp(lm_b))
    eb_28, e_28 = ess_bins(W["g28_noW"] * np.exp(lm_b))
    fac = fc["min_ess_per_bin"] / np.maximum(eb_new[big], 1e-9)
    rep += ["", f"ESS at the refit-2 best fit (noW): C1 orbits gens 23-27 {e_old:.0f} ({e_old / cpu_old:.2f}/CPU-h), generation 28 alone {e_28:.0f} "
            f"({e_28 / cpu_28:.1f}/CPU-h), gens 23-28 {e_new:.0f}",
            f"  {int(big.sum())} big bins: ESS_b median {np.median(eb_new[big]):.1f}, min {eb_new[big].min():.1f}, max {eb_new[big].max():.1f}; "
            f"at >= {fc['min_ess_per_bin']}: {int(np.sum(eb_new[big] >= fc['min_ess_per_bin']))}"]
    proj = {}
    for lab, f_ in (("median bin", np.median(fac)), ("90% of bins", np.percentile(fac, 90)), ("all bins (worst)", fac.max())):
        extra = max(0.0, f_ - 1.0) * e_new
        proj[lab] = (float(f_), float(extra / (e_28 / cpu_28)), float(extra / (e_old / cpu_old)))
        rep.append(f"  top-up projection, {lab}: x{f_:.1f} ESS -> {proj[lab][1]:.0f} CPU-h at the re-centred rate ({proj[lab][2]:.0f} at the old rate)")
    worst = np.argsort(eb_new + np.where(big, 0, 1e9))[:5]
    rep.append("  worst big bins (G, d, log P, e: real, ESS_b): " + "; ".join(
        "G {:g}-{:g} d {:g}-{:g} lP {:g}-{:.2f} e {:g}-{:g}: {}, {:.1f}".format(
            oe[0][a], oe[0][a + 1], oe[1][b], oe[1][b + 1], oe[2][c], oe[2][c + 1], oe[3][d], oe[3][d + 1], int(k_o[i]), eb_new[i])
        for i in worst for a, b, c, d in [np.unravel_index(i, shp)]))

    # distance / ESS figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    xx = np.arange(shp[1])
    lbl = [f"{a:g}-{b:g}" for a, b in zip(oe[1][:-1], oe[1][1:])]
    for j, case in enumerate(("noW", "noW_Ns_fixed", "noW_no_spurious")):
        _, mo_, _, _, _, s_, _ = results[case]["_e"]
        Mc = (mo_ + s_).reshape(shp)
        st_ = series_style(j, style)
        axs[0].plot(xx, Mc.sum(axis=(0, 2, 3)) / R4.sum(axis=(0, 2, 3)), marker="o", lw=2, color=st_["color"], ls=st_["linestyle"], label=case)
    axs[0].plot(xx, Ma.sum(axis=0) / Ra.sum(axis=0), "s:", lw=2, color="0.3", label="accelerations (noW)")
    axs[0].axhline(1, color="0.4", ls=":")
    axs[0].set_xticks(xx, lbl)
    axs[0].set_ylim(0, 2.5)
    apply_axes_style(axs[0], style, xlabel="distance bin (kpc)", ylabel="(mock + spurious) / real orbits")
    axs[0].legend(fontsize=style.tick_label_fontsize)
    axs[1].plot(xx, (R4.sum(axis=(0, 2)) * ecen).sum(axis=1) / R4.sum(axis=(0, 2, 3)), "o-", color="k", lw=2, label="real")
    axs[1].plot(xx, (M.sum(axis=(0, 2)) * ecen).sum(axis=1) / M.sum(axis=(0, 2, 3)), "s--", lw=2, color=series_style(1, style)["color"], label="refit 2 + spurious")
    axs[1].set_xticks(xx, lbl)
    apply_axes_style(axs[1], style, xlabel="distance bin (kpc)", ylabel="mean e (coarse-bin centres)")
    axs[1].legend(fontsize=style.tick_label_fontsize)
    axs[2].plot(np.arange(int(big.sum())), eb_new[big], "o", color=series_style(1, style)["color"], label="gens 23-28 at refit-2 best fit")
    axs[2].axhline(fc["min_ess_per_bin"], color="0.4", ls=":")
    axs[2].set_yscale("log")
    apply_axes_style(axs[2], style, xlabel="big likelihood bin (index)", ylabel="ESS_b")
    axs[2].legend(fontsize=style.tick_label_fontsize)
    fig.suptitle("Rung 3 refit 2: counts and mean e vs distance; per-bin ESS", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_distance_and_ess.png", dpi=int(cfg.diagnostics.figure_dpi))

    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    out.update(names=list(NAMES), N_s_prior=[ns_mu, ns_sd], e_below_0p1=e_low, projection=proj,
               template={"pi_bar": pibar, "r_P": rr_P.tolist(), "r_e": rr_e.tolist(), "T_PE": T_PE.tolist()},
               ess={"old": e_old, "new": e_new, "gen28": e_28, "big_bins": eb_new[big].tolist()})
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
