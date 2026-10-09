#!/usr/bin/env python3
"""Rung 3 first fit (#391; docs/MOCK_POPULATION_SPEC.md §12): reweight gens 23-27 onto the full DR3 sample.

Binned effective-likelihood fit (Arguelles, Schneider & Yuan 2019) of MdS17-anchored multiplicative
modifiers plus a spurious-orbit component to the real DR3 orbit (G x d x log P x e, C1) and acceleration
(G x d) counts. No simulation: weights are the cached gens 23-27 deterministic-mixture weights (noW
baseline, cmdW sensitivity) divided by the MP-Q7 low-mass scale and multiplied by m(x; theta).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

GEN_ARTIFACTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5",
                 "restart_gen26_topup.h5", "restart_gen27_evolved.h5")
NAMES = ("ln_A", "alpha_lo", "alpha_hi", "gamma_P", "ln_L_P", "dgamma_q", "ln_F_twin", "d_eta", "f_s", "k_P", "k_d")
PANELS = ("P_orb_days", "G_mag", "inv_parallax_mas_inv", "eccentricity", "f_m_msun", "cos_inclination")


def say_loglike(k: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Sum over bins of the SAY effective log-likelihood; Poisson where the MC variance vanishes."""
    mu = np.maximum(mu, 1e-300)
    pois = k * np.log(mu) - mu - gammaln(k + 1)
    use = var > 1e-12 * mu**2
    out = pois.copy()
    if np.any(use):
        m, v, kk = mu[use], var[use], k[use]
        a = m * m / v + 1.0
        b = m / v
        out[use] = a * np.log(b) + gammaln(kk + a) - gammaln(kk + 1) - (kk + a) * np.log1p(b) - gammaln(a)
    return float(np.sum(out))


def bin_index(cols: list[np.ndarray], edges: list[np.ndarray]) -> np.ndarray:
    """Flat bin index (C order) or -1 outside the edges / non-finite."""
    idx = np.zeros(cols[0].size, np.int64)
    ok = np.ones(cols[0].size, bool)
    for c, e in zip(cols, edges):
        j = np.searchsorted(e, c, side="right") - 1
        ok &= np.isfinite(c) & (j >= 0) & (j < e.size - 1)
        idx = idx * (e.size - 1) + np.clip(j, 0, e.size - 2)
    return np.where(ok, idx, -1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_fit.yaml"))
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

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
    PMAX = float(fc["c1_period_max_days"])
    oe = [np.asarray(fc["orbit_bins"][k], float) for k in ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")]
    ae = [np.asarray(fc["acceleration_bins"][k], float) for k in ("g_mag", "distance_kpc")]
    n_ob = int(np.prod([e.size - 1 for e in oe]))
    n_ab = int(np.prod([e.size - 1 for e in ae]))

    # ---------------- mock ----------------
    z = np.load(args.cache, allow_pickle=True)
    casc = np.asarray(z["cascade"], float)
    acc = np.asarray(z["accepted"], bool)
    tk = ("m1_msun", "m2_msun", "period_days", "eccentricity")
    parts = [ps.read_proposal_artifact(args.artifact_dir / a) for a in GEN_ARTIFACTS]
    t = {k: np.concatenate([np.asarray(p[0][k], float) for p in parts]) for k in tk}
    pub_acc = np.concatenate([np.asarray(p[1]["published_acceleration"], bool) for p in parts])
    cpu_h = float(sum(np.sum(np.asarray(p[1]["cpu_seconds"], float)) for p in parts) / 3600.0)
    if not np.array_equal(t["period_days"], np.asarray(z["period_days"], float)):
        raise SystemExit("artifact order differs from the weight cache")
    m_g = np.asarray(z["phot_g_mean_mag"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_d_o, m_d_a = 1.0 / casc[:, 0], 1.0 / casc[:, 2]
        m_lp_o = np.log10(casc[:, 10])
    m_orb = acc & (casc[:, 10] <= PMAX)
    mob = np.where(m_orb, bin_index([m_g, m_d_o, m_lp_o, casc[:, 14]], oe), -1)
    mab = np.where(pub_acc, bin_index([m_g, m_d_a], ae), -1)
    rel = np.flatnonzero((mob >= 0) | (mab >= 0))
    m1, q = t["m1_msun"][rel], t["m2_msun"][rel] / t["m1_msun"][rel]
    lp_true = np.log10(t["period_days"][rel])
    e_true = t["eccentricity"][rel]
    s_low = mds.low_mass_frequency_scale(m1, m1_anchor_msun=tgt.provisional_low_mass_anchor_msun,
                                         m1_zero_msun=tgt.provisional_low_mass_zero_msun)
    m1s = np.clip(m1, table.m1_range[0], table.m1_range[1])
    eta_raw = mds.eta(m1s, lp_true, table, m1_interpolation=tgt.provisional_m1_interpolation, eta_floor=-0.999)
    emax = mds.e_max(t["period_days"][rel], table)
    circ = mds.is_circular(t["period_days"][rel], table)
    floor = float(fc["eta_floor"])
    eta0 = np.maximum(eta_raw, floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_e_ratio = np.where(circ | (e_true <= 0), 0.0, np.log(np.clip(e_true / emax, 1e-12, None)))
    piv = float(fc["m1_pivot_msun"])
    lm1 = np.log(m1 / piv)
    lo_m = m1 < piv
    W0 = {k: np.asarray(z[f"w_{k}"], float)[rel] / s_low for k in ("noW", "cmdW")}
    ob_r, ab_r = mob[rel], mab[rel]

    def log_mod(th: np.ndarray) -> np.ndarray:
        lnA, alo, ahi, gP, lnL, dq, lnT, de = th[:8]
        lm = lnA + np.where(lo_m, alo, ahi) * lm1
        lm += gP * (lp_true - fc["log_p_tilt_pivot"])
        lm += lnL / (1.0 + np.exp(-(lp_true - fc["long_p_centre"]) / fc["long_p_width"]))
        lm += dq * np.log(q) + lnT * (q >= fc["twin_q_min"])
        eta1 = np.maximum(eta_raw + de, floor)
        lm += np.where(circ, 0.0, np.log((eta1 + 1.0) / (eta0 + 1.0)) + (eta1 - eta0) * log_e_ratio)
        return lm

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
    r_a0, _, _, r_inc = (np.asarray(x, float) for x in gm.get_Campbell_elements(col(rt, "A"), col(rt, "B"), col(rt, "F"), col(rt, "G")))
    r_c1 = r_P <= PMAX
    rob = np.where(r_c1, bin_index([r_g, r_d, np.log10(r_P), r_e], oe), -1)
    k_o = np.bincount(rob[rob >= 0], minlength=n_ob).astype(float)
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    inp = {"source_id": np.asarray(at["source_id"], np.int64)}
    inp.update({c: col(at, c) for c in ("phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ipd_frac_multi_peak",
                                        "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")})
    keep_a, _ = ps.real_comparison_keep(at, prop, inp)
    at = at[keep_a]
    ra_g = col(at, "g_mag")
    ra_plx = zplx(col(at, "parallax"), ra_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        ra_d = np.where(ra_plx > 0, 1.0 / ra_plx, np.nan)
    rab = bin_index([ra_g, ra_d], ae)
    k_a = np.bincount(rab[rab >= 0], minlength=n_ab).astype(float)
    N_o = float(k_o.sum())

    # ---------------- spurious template (§12.4, MP-Q43 a) ----------------
    pG = np.histogram(ra_g, bins=oe[0])[0].astype(float)
    pG /= pG.sum()
    pe = np.diff(oe[3]) / (oe[3][-1] - oe[3][0])

    def exp_bins(edges: np.ndarray, k: float, x0: float) -> np.ndarray:
        if abs(k) < 1e-8:
            v = np.diff(edges)
        else:
            v = (np.exp(k * (edges[1:] - x0)) - np.exp(k * (edges[:-1] - x0))) / k
        return v / v.sum()

    def spur(th: np.ndarray) -> np.ndarray:
        f_s, kP, kd = th[8:]
        pd = exp_bins(oe[1], kd, 0.0)
        pP = exp_bins(oe[2], kP, oe[2][-1])
        return f_s * N_o * np.einsum("a,b,c,d->abcd", pG, pd, pP, pe).ravel()

    dsel = (oe[1][:-1] >= fc["spurious"]["prior_bin_distance_kpc"][0] - 1e-9) & (oe[1][1:] <= fc["spurious"]["prior_bin_distance_kpc"][1] + 1e-9)
    gsel = (oe[0][:-1] >= fc["spurious"]["prior_bin_g_mag"][0] - 1e-9) & (oe[0][1:] <= fc["spurious"]["prior_bin_g_mag"][1] + 1e-9)
    prior_mask = np.einsum("a,b,c,d->abcd", gsel, dsel, np.ones(oe[2].size - 1), np.ones(oe[3].size - 1)).ravel() > 0
    k_prior_bin = float(k_o[prior_mask].sum())

    def expected(th: np.ndarray, wkey: str):
        w = W0[wkey] * np.exp(log_mod(th))
        so, sa = ob_r >= 0, ab_r >= 0
        mu_o = np.bincount(ob_r[so], weights=w[so], minlength=n_ob)
        v_o = np.bincount(ob_r[so], weights=w[so] ** 2, minlength=n_ob)
        mu_a = np.bincount(ab_r[sa], weights=w[sa], minlength=n_ab)
        v_a = np.bincount(ab_r[sa], weights=w[sa] ** 2, minlength=n_ab)
        return w, mu_o, v_o, mu_a, v_a

    sp = fc["spurious"]

    def nll(th: np.ndarray, wkey: str) -> float:
        _, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        s = spur(th)
        ll = say_loglike(k_o, mu_o + s, v_o) + say_loglike(k_a, mu_a, v_a)
        frac = s[prior_mask].sum() / max(k_prior_bin, 1.0)
        ll += -0.5 * ((frac - sp["prior_mean"]) / sp["prior_sigma"]) ** 2
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    x0 = np.array([par[n][0] for n in NAMES], float)
    bounds = [(par[n][1], par[n][2]) for n in NAMES]
    rng = np.random.default_rng(args.seed)
    rep = ["#391 rung 3 first fit (spec §12): gens 23-27 reweighted; no new simulation", "",
           f"real: {int(N_o)} C1 orbits in {n_ob} bins, {int(k_a.sum())} accelerations in {n_ab} bins; "
           f"prior bin (d 0.7-1.5 kpc, G 12-16) holds {int(k_prior_bin)} real orbits",
           f"mock draws entering the likelihood: {rel.size}", ""]
    results = {}
    for wkey in ("noW", "cmdW"):
        best = None
        starts = [x0] + [np.array([rng.uniform(max(lo, x - 1), min(hi, x + 1)) for x, (lo, hi) in zip(x0, bounds)])
                         for _ in range(args.n_starts - 1)]
        sols = []
        for s0 in starts:
            r = minimize(nll, s0, args=(wkey,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "maxfun": 30000})
            r = minimize(nll, r.x, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-7})
            sols.append((float(r.fun), r.x.copy()))
            if best is None or r.fun < best.fun:
                best = r
        # polish: derivative-free Powell from the best L-BFGS-B point (the SAY objective has kinks where a
        # bin's MC variance switches to the Poisson limit, which can stop L-BFGS-B's line search)
        pol = minimize(nll, best.x, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-7})
        if pol.fun <= best.fun:
            best = pol
        th = best.x
        # Laplace: central-difference Hessian
        h = 1e-3 * np.maximum(1.0, np.abs(th))
        n = th.size
        H = np.zeros((n, n))
        f0 = nll(th, wkey)
        for i in range(n):
            for j in range(i, n):
                def f(di, dj):
                    x = th.copy()
                    x[i] += di
                    x[j] += dj
                    return nll(x, wkey)
                if i == j:
                    H[i, i] = (f(h[i], 0) - 2 * f0 + f(-h[i], 0)) / h[i] ** 2
                else:
                    H[i, j] = H[j, i] = (f(h[i], h[j]) - f(h[i], -h[j]) - f(-h[i], h[j]) + f(-h[i], -h[j])) / (4 * h[i] * h[j])
        try:
            cov = np.linalg.inv(H)
            err = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            err = np.full(n, np.nan)
        at_bound = [NAMES[i] for i in range(n) if min(abs(th[i] - bounds[i][0]), abs(th[i] - bounds[i][1])) < 1e-3]
        w, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        x_mds = x0.copy()
        x_mds[8] = 0.0
        nll_mds = nll(x_mds, wkey)
        so, sa = ob_r >= 0, ab_r >= 0
        ess_o = ps.kish_ess(w[so])
        ess_a = ps.kish_ess(w[sa])
        ess_b = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
        low = ess_b < fc["min_ess_per_bin"]
        s = spur(th)
        frac = s[prior_mask].sum() / max(k_prior_bin, 1.0)
        dev = 2 * (say_loglike(k_o, np.where(k_o > 0, k_o, 1e-300), np.zeros(n_ob)) - say_loglike(k_o, mu_o + s, v_o))
        results[wkey] = dict(theta=th.tolist(), err=err.tolist(), nll=float(best.fun), nll_mds17=float(nll_mds), at_bound=at_bound,
                             ess_orbits=float(ess_o), ess_accel=float(ess_a), n_low_ess_bins=int(low.sum()),
                             real_share_low_ess=float(k_o[low].sum() / N_o), spurious_total=float(s.sum()),
                             spurious_frac_prior_bin=float(frac), mock_orbits=float(mu_o.sum()), mock_accel=float(mu_a.sum()),
                             converged=bool(best.success), poisson_deviance_orbits=float(dev))
        rep.append(f"[{wkey}] converged {best.success} ({best.message}); -lnL best {best.fun:.1f} vs MdS17 (+ f_s=0) {nll_mds:.1f}")
        near = np.array([x for f_, x in sols if f_ - best.fun < 5.0])
        rep.append(f"    {len(sols)} starts; {len(near)} end within delta(-lnL) < 5 of the best ({sorted(round(f_ - best.fun, 1) for f_, _ in sols)})")
        rep.append("    name        best +- Laplace   [min, max over the near-best starts]")
        for i_, (nm, v, e_) in enumerate(zip(NAMES, th, err)):
            rep.append(f"    {nm:10s} {v:+.3f} +- {e_:.3f}   [{near[:, i_].min():+.2f}, {near[:, i_].max():+.2f}]")
        results[wkey]["near_best_starts"] = near.tolist()
        results[wkey]["start_nll"] = [f_ for f_, _ in sols]
        rep.append(f"    at bound: {at_bound or 'none'}")
        rep.append(f"    expected: mock orbits {mu_o.sum():.0f} + spurious {s.sum():.0f} = {mu_o.sum() + s.sum():.0f} vs real {N_o:.0f}; "
                   f"mock accelerations {mu_a.sum():.0f} vs real {k_a.sum():.0f}")
        rep.append(f"    spurious: f_s {th[8]:.3f} of all C1 orbits; share in the prior bin {frac:.3f} (prior 0.21 +- 0.04)")
        rep.append(f"    ESS at best fit: orbits {ess_o:.0f}, accelerations {ess_a:.0f} (MdS17 weights: "
                   f"{ps.kish_ess(W0[wkey][so] * s_low[so]):.0f}, {ps.kish_ess(W0[wkey][sa] * s_low[sa]):.0f})"
                   + ("  ** ESS-COLLAPSED **" if ess_o < fc["ess_collapse_threshold"] else ""))
        rep.append(f"    orbit bins with ESS_b < {fc['min_ess_per_bin']}: {int(low.sum())}/{n_ob}, holding {k_o[low].sum() / N_o:.1%} of real orbits")
        rep.append(f"    Poisson-style deviance (orbits, incl. MC var): {dev:.0f} over {int((k_o > 0).sum())} non-empty bins")
        rep.append("")
        results[wkey]["_w"] = w
        results[wkey]["_mu"] = (mu_o, v_o, mu_a, v_a, s)

    # ---------------- posterior-predictive checks (noW best fit) ----------------
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    wb = results["noW"]["_w"]
    th_b = np.array(results["noW"]["theta"])
    w_mds = W0["noW"] * s_low
    so = ob_r >= 0
    idx_o = rel[so]
    mo = {"P_orb_days": casc[idx_o, 10], "G_mag": m_g[idx_o], "inv_parallax_mas_inv": m_d_o[idx_o],
          "eccentricity": casc[idx_o, 14], "cos_inclination": np.cos(np.radians(casc[idx_o, 16]))}
    with np.errstate(divide="ignore", invalid="ignore"):
        mo["f_m_msun"] = astrometric_mass_function(casc[idx_o, 17], casc[idx_o, 0], casc[idx_o, 10])
        rv = {"P_orb_days": r_P, "G_mag": r_g, "inv_parallax_mas_inv": r_d, "eccentricity": r_e,
              "f_m_msun": astrometric_mass_function(r_a0, r_plx, r_P), "cos_inclination": np.cos(r_inc)}
    rin = rob >= 0
    s_cells = spur(th_b).reshape([e.size - 1 for e in oe])
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    rep.append("PPC six-panel (C1 orbits in the likelihood edges): real | MdS17 | best fit (+ spurious where defined); per-panel totals")
    sp_marg = {"G_mag": (oe[0], s_cells.sum(axis=(1, 2, 3))), "inv_parallax_mas_inv": (oe[1], s_cells.sum(axis=(0, 2, 3))),
               "P_orb_days": (10 ** oe[2], s_cells.sum(axis=(0, 1, 3))), "eccentricity": (oe[3], s_cells.sum(axis=(0, 1, 2)))}
    for ax, name in zip(np.ravel(axs), PANELS):
        sc, lo, hi = axes_cfg[name]
        edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=20)
        cen = np.sqrt(edges[1:] * edges[:-1]) if sc == "log" else 0.5 * (edges[1:] + edges[:-1])
        rn = np.histogram(rv[name][rin], bins=edges)[0]
        m0 = np.histogram(mo[name], bins=edges, weights=w_mds[so])[0]
        m1_ = np.histogram(mo[name], bins=edges, weights=wb[so])[0]
        if name in sp_marg:
            se, sv = sp_marg[name]
            # spread each coarse spurious cell uniformly (in the panel's own coordinate) over the fine bins it covers
            sfine = np.zeros(cen.size)
            for a_, b_, val in zip(se[:-1], se[1:], sv):
                ov = np.clip(np.minimum(edges[1:], b_) - np.maximum(edges[:-1], a_), 0, None)
                if sc == "log":
                    ov = np.clip(np.log10(np.minimum(edges[1:], b_)) - np.log10(np.maximum(edges[:-1], a_)), 0, None) if a_ > 0 else ov
                tot = ov.sum()
                if tot > 0:
                    sfine += val * ov / tot
            m1_ = m1_ + sfine
        ax.step(cen, rn, where="mid", color="k", lw=2, label="real DR3 (C1)")
        ax.step(cen, m0, where="mid", color=series_style(1, style)["color"], ls="--", lw=2, label="MdS17 (rung 2)")
        ax.step(cen, m1_, where="mid", color=series_style(2, style)["color"], lw=2,
                label="best fit + spurious" if name in sp_marg else "best fit (no spurious term)")
        if sc == "log":
            ax.set_xscale("log")
        apply_axes_style(ax, style, xlabel=name, ylabel="count per bin")
        ax.legend(fontsize=style.tick_label_fontsize)
        rep.append(f"  {name}: real {rn.sum()} | MdS17 {m0.sum():.0f} | fit {m1_.sum():.0f}")
    fig.suptitle("Rung 3 PPC: six-panel, real vs MdS17 vs best fit (noW)", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_six_panel.png", dpi=int(cfg.diagnostics.figure_dpi))

    # counts vs distance and G, mean e vs distance, orbit:accel vs G and d
    mu_o, v_o, mu_a, v_a, s = results["noW"]["_mu"]
    shp = [e.size - 1 for e in oe]
    R = k_o.reshape(shp)
    M = (mu_o + s).reshape(shp)
    M0 = np.bincount(ob_r[so], weights=w_mds[so], minlength=n_ob).reshape(shp)
    Ra = k_a.reshape([e.size - 1 for e in ae])
    Ma = mu_a.reshape(Ra.shape)
    Ma0 = np.bincount(ab_r[ab_r >= 0], weights=w_mds[ab_r >= 0], minlength=n_ab).reshape(Ra.shape)
    ecen = 0.5 * (oe[3][1:] + oe[3][:-1])
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    rep += ["", "PPC per distance bin (d edges " + str(oe[1].tolist()) + "): mock/real orbits MdS17 -> fit | accelerations MdS17 -> fit | "
            "mean-e (bin centres) real / MdS17 / fit | orbit:accel real / fit"]
    for j in range(shp[1]):
        ro, mo0, mo1 = R[:, j].sum(), M0[:, j].sum(), M[:, j].sum()
        ra_, ma0, ma1 = Ra[:, j].sum(), Ma0[:, j].sum(), Ma[:, j].sum()
        e_r = (R[:, j].sum(axis=(0, 1)) * ecen).sum() / max(ro, 1)
        e_0 = (M0[:, j].sum(axis=(0, 1)) * ecen).sum() / max(mo0, 1e-9)
        e_1 = (M[:, j].sum(axis=(0, 1)) * ecen).sum() / max(mo1, 1e-9)
        rep.append(f"  d {oe[1][j]:.1f}-{oe[1][j + 1]:.1f}: orbits {mo0 / ro:.2f} -> {mo1 / ro:.2f} [{int(ro)}]; accels {ma0 / ra_:.2f} -> {ma1 / ra_:.2f} "
                   f"[{int(ra_)}]; <e> {e_r:.3f} / {e_0:.3f} / {e_1:.3f}; o:a {ro / ra_:.3f} / {mo1 / ma1:.3f}")
    for i in range(shp[0]):
        ro, mo0, mo1 = R[i].sum(), M0[i].sum(), M[i].sum()
        ra_, ma0, ma1 = Ra[i].sum(), Ma0[i].sum(), Ma[i].sum()
        rep.append(f"  G {oe[0][i]:.1f}-{oe[0][i + 1]:.1f}: orbits {mo0 / ro:.2f} -> {mo1 / ro:.2f} [{int(ro)}]; accels {ma0 / ra_:.2f} -> {ma1 / ra_:.2f} [{int(ra_)}]; "
                   f"o:a {ro / ra_:.3f} / {mo1 / ma1:.3f}")
    for row, (ax_i, lab, edges_) in enumerate(((1, "distance bin (kpc)", oe[1]), (0, "G bin (mag)", oe[0]))):
        other = tuple(a for a in range(4) if a != ax_i)
        other_a = tuple(a for a in range(2) if a != ax_i)
        x = np.arange(edges_.size - 1)
        lbl = [f"{a:g}-{b:g}" for a, b in zip(edges_[:-1], edges_[1:])]
        ro, r0, r1 = R.sum(axis=other), M0.sum(axis=other), M.sum(axis=other)
        ra_, a0_, a1_ = Ra.sum(axis=other_a), Ma0.sum(axis=other_a), Ma.sum(axis=other_a)
        ax = axs[row, 0]
        for jj, (lab2, y) in enumerate((("orbits MdS17", r0 / ro), ("orbits fit", r1 / ro), ("accels MdS17", a0_ / ra_), ("accels fit", a1_ / ra_))):
            st_ = series_style(jj, style)
            ax.plot(x, y, marker="o", lw=2, color=st_["color"], ls=st_["linestyle"], label=lab2)
        ax.axhline(1, color="0.4", ls=":")
        ax.set_xticks(x, lbl)
        ax.set_ylim(0, 2)
        apply_axes_style(ax, style, xlabel=lab, ylabel="mock / real")
        ax.legend(fontsize=style.tick_label_fontsize)
        ax = axs[row, 1]
        if ax_i == 1:
            er = (R.sum(axis=(0, 2)) * ecen).sum(axis=1) / R.sum(axis=(0, 2, 3))
            e0_ = (M0.sum(axis=(0, 2)) * ecen).sum(axis=1) / M0.sum(axis=(0, 2, 3))
            e1_ = (M.sum(axis=(0, 2)) * ecen).sum(axis=1) / M.sum(axis=(0, 2, 3))
        else:
            er = (R.sum(axis=(1, 2)) * ecen).sum(axis=1) / R.sum(axis=(1, 2, 3))
            e0_ = (M0.sum(axis=(1, 2)) * ecen).sum(axis=1) / M0.sum(axis=(1, 2, 3))
            e1_ = (M.sum(axis=(1, 2)) * ecen).sum(axis=1) / M.sum(axis=(1, 2, 3))
        for jj, (lab2, y) in enumerate((("real", er), ("MdS17", e0_), ("fit + spurious", e1_))):
            st_ = series_style(jj, style)
            ax.plot(x, y, marker="o", lw=2, color="k" if jj == 0 else st_["color"], ls=st_["linestyle"], label=lab2)
        ax.set_xticks(x, lbl)
        apply_axes_style(ax, style, xlabel=lab, ylabel="mean e (coarse-bin centres)")
        ax.legend(fontsize=style.tick_label_fontsize)
        ax = axs[row, 2]
        for jj, (lab2, y) in enumerate((("real", ro / ra_), ("MdS17", r0 / a0_), ("fit", r1 / a1_))):
            st_ = series_style(jj, style)
            ax.plot(x, y, marker="o", lw=2, color="k" if jj == 0 else st_["color"], ls=st_["linestyle"], label=lab2)
        ax.set_xticks(x, lbl)
        apply_axes_style(ax, style, xlabel=lab, ylabel="orbits (C1) / accelerations")
        ax.legend(fontsize=style.tick_label_fontsize)
    fig.suptitle("Rung 3 PPC (noW): counts, mean e and orbit:acceleration vs distance (top) and G (bottom)", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_counts_e_ratio.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ESS per bin map (noW best fit): fraction of real orbits in bins below threshold, by (d, log P)
    ess_b = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0).reshape(shp)
    rep += ["", "ESS_b at best fit (noW), summed over G and e per (d, log P) cell: min ESS_b in cell / real orbits in cell"]
    for j in range(shp[1]):
        rep.append(f"  d {oe[1][j]:.1f}-{oe[1][j + 1]:.1f}: " + " ".join(
            f"lP{oe[2][p]:.2f}-{oe[2][p + 1]:.2f}:{ess_b[:, j, p, :][R[:, j, p, :] > 0].min() if (R[:, j, p, :] > 0).any() else float('nan'):.0f}/{int(R[:, j, p, :].sum())}"
            for p in range(shp[2])))
    # ---------------- top-up sizing (§12.8) ----------------
    need = (k_o >= 0.01 * N_o) & (ess_b.ravel() < fc["min_ess_per_bin"])
    fac = fc["min_ess_per_bin"] / np.maximum(ess_b.ravel()[need], 0.5)
    rep += ["", f"top-up (§12.8): gens 23-27 used {cpu_h:.0f} CPU-h; ESS of C1 orbits at the best fit {results['noW']['ess_orbits']:.0f} "
            f"({results['noW']['ess_orbits'] / cpu_h:.1f} per CPU-h)",
            f"  bins holding >= 1% of real orbits: {int((k_o >= 0.01 * N_o).sum())}; of those below ESS_b {fc['min_ess_per_bin']}: {int(need.sum())} "
            f"({k_o[need].sum() / N_o:.1%} of real orbits)"]
    if need.any():
        rep.append(f"  ESS_b multiplier needed in those bins with the current proposal mix: median {np.median(fac):.0f}x, max {fac.max():.0f}x "
                   f"-> {np.median(fac) * cpu_h:.0f} / {fac.max() * cpu_h:.0f} CPU-h if the mix were unchanged (an upper bound; a proposal "
                   f"re-centred on the best fit is the intended route)")
        for b_ in np.flatnonzero(need):
            gi, di, pi, ei = np.unravel_index(b_, shp)
            rep.append(f"    G {oe[0][gi]:g}-{oe[0][gi + 1]:g}, d {oe[1][di]:g}-{oe[1][di + 1]:g}, log P {oe[2][pi]:g}-{oe[2][pi + 1]:.2f}, "
                       f"e {oe[3][ei]:g}-{oe[3][ei + 1]:g}: real {int(k_o[b_])}, ESS_b {ess_b.ravel()[b_]:.1f}")
    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    out["cpu_hours_gens23_27"] = cpu_h
    out["names"] = list(NAMES)
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
