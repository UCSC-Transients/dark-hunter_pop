#!/usr/bin/env python3
"""Rung 3 refit (#391; spec §12.10): gens 23-28 reweighted, normalized q modifiers, PR #455 spurious template.

Two parts:

* **A. Sampling gain of the re-centred generation 28.** ESS_b in the likelihood bins holding >= 1% of the
  real C1 orbits, evaluated at the *first-fit* best θ (PR #459, its own model) with the gens 23-27 mixture,
  the gens 23-28 mixture and generation 28 alone. Also ESS per CPU-h and the projected top-up CPU-h.
* **B. Refit** with MP-Q42b (q tilt and twin multiplier normalized over MdS17's p_q on 0.3 <= q <= 1) and
  MP-Q43b (spurious template π(log P, e) from the PR #455 re-injection, separable with shrinkage, times a
  free distance tilt, as a share of the observed counts), on the gens 23-28 mixture (noW baseline, cmdW
  sensitivity), with the posterior-predictive checks of §12.7.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rung3_fit_391 import PANELS, bin_index, say_loglike  # noqa: E402

NAMES = ("ln_A", "alpha_lo", "alpha_hi", "gamma_P", "ln_L_P", "dgamma_q", "ln_F_twin", "d_eta", "f_s", "k_d")
MOCK_SYM_OFFSET = 2 * 10**15  # PR #455 seed ids of the symmetric (mock fitted-orbit) re-injections


def build_cache(args, cfg, parent) -> None:
    from darkhunter_pop import malmquist_cmd as mc
    from darkhunter_pop import proposal_set as ps

    parts = [ps.read_proposal_artifact(a) for a in args.artifact]
    common = set.intersection(*(set(p[0]) for p in parts))
    t = {k: np.concatenate([np.asarray(p[0][k]) for p in parts]) for k in parts[0][0] if k in common}
    okeys = ("accepted_orbital", "published_acceleration", "cascade", "cpu_seconds")
    o = {k: np.concatenate([np.asarray(p[1][k]) for p in parts]) for k in okeys}
    tg = ps.MdS17TargetConfig.model_validate(json.loads(parts[0][2]["provenance_json"])["target_mds17_json"])
    gcs = [ps.ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for p in parts]
    if any(g.flux.evolved_rows_centre == "evolved_relation_deblended" for g in gcs):
        ps.ensure_m1_deblend_dark(t, parent, cfg, gcs)
    evo = ps.evolved_mg0_for_draws(t, parent)
    rel = ps.mist_relation_for_draws(t, parent, cfg)
    ll = ps.mds17_luminous_log_intensity(t, tg, evolved_mg0_system=evo, relation_log10_f=rel)
    lw, _ = ps.malmquist_cmd_log_weight(t, parent, tg, mc.load_cmd_malmquist_config(args.cmd_malmquist), cfg)
    lqs = np.vstack([ps.log_q_total_for(t, parent, g, cfg) for g in gcs])
    np.savez(args.cache, ll=ll, lw=lw, lqs=lqs, n_per_gen=np.array([g.n_draws for g in gcs]),
             gens=np.array([g.generation for g in gcs]), scale=parent.scale_to_full,
             generation=t["generation"], draw_index=t["draw_index"], m1_msun=t["m1_msun"], m2_msun=t["m2_msun"],
             period_days=t["period_days"], eccentricity=t["eccentricity"], phot_g_mean_mag=t["phot_g_mean_mag"],
             accepted=o["accepted_orbital"], published_acceleration=o["published_acceleration"],
             cascade=o["cascade"], cpu_seconds=o["cpu_seconds"])
    print(f"cached {ll.size} draws over generations {[g.generation for g in gcs]} -> {args.cache}", flush=True)


def mixture_weights(z, gens_used, use_w: bool):
    """Deterministic-mixture weights over the draws of ``gens_used`` only (others 0)."""
    from darkhunter_pop import proposal_set as ps

    gens = list(np.asarray(z["gens"]))
    sel = np.isin(np.asarray(z["generation"]), gens_used)
    j = [gens.index(g) for g in gens_used]
    lt = np.asarray(z["ll"], float)[sel] + (np.asarray(z["lw"], float)[sel] if use_w else 0.0)
    w = np.zeros(sel.size)
    w[sel] = ps.importance_weights(lt, [np.asarray(z["lqs"])[k][sel] for k in j], [int(z["n_per_gen"][k]) for k in j],
                                   scale_to_full=float(z["scale"]))
    return w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, action="append", required=True, help="gens 23-28 in order")
    ap.add_argument("--cache", type=Path, required=True, help="gens 23-28 cache (built if missing)")
    ap.add_argument("--old-cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--reinjection-log", type=Path, default=Path("output/detection_vs_population_391/reinjection.jsonl"))
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--first-fit", type=Path, default=Path("docs/gate391/rung3_first_fit/summary.json"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--cmd-malmquist", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    ap.add_argument("--first-fit-config", type=Path, default=Path("config/population/rung3_fit.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_refit.yaml"))
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
    fc1 = yaml.safe_load(args.first_fit_config.read_text())["rung3"]
    cfg = load_config()
    gm = import_gaiamock_mod()
    frag = ps.load_proposal_set_fragment(args.fragment)
    prop, tgt = frag.proposal, frag.target_mds17
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    table = mds.load_mds17_table(tgt.table_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.cache.exists():
        build_cache(args, cfg, parent)
    z = np.load(args.cache, allow_pickle=True)
    PMAX = float(fc["c1_period_max_days"])
    oe = [np.asarray(fc["orbit_bins"][k], float) for k in ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")]
    ae = [np.asarray(fc["acceleration_bins"][k], float) for k in ("g_mag", "distance_kpc")]
    shp = [e.size - 1 for e in oe]
    n_ob, n_ab = int(np.prod(shp)), (ae[0].size - 1) * (ae[1].size - 1)
    gens = [int(g) for g in z["gens"]]
    old_gens = [g for g in gens if g != 28]

    # ---------------- mock observed space ----------------
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
    gen_r = np.asarray(z["generation"])[rel]
    m1 = np.asarray(z["m1_msun"], float)[rel]
    q = np.asarray(z["m2_msun"], float)[rel] / m1
    P_true = np.asarray(z["period_days"], float)[rel]
    lp_true = np.log10(P_true)
    e_true = np.asarray(z["eccentricity"], float)[rel]
    s_low = mds.low_mass_frequency_scale(m1, m1_anchor_msun=tgt.provisional_low_mass_anchor_msun,
                                         m1_zero_msun=tgt.provisional_low_mass_zero_msun)
    m1s = np.clip(m1, table.m1_range[0], table.m1_range[1])
    eta_raw = mds.eta(m1s, lp_true, table, m1_interpolation=tgt.provisional_m1_interpolation, eta_floor=-0.999)
    emax = mds.e_max(P_true, table)
    circ = mds.is_circular(P_true, table)
    floor = float(fc["eta_floor"])
    eta0 = np.maximum(eta_raw, floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_e_ratio = np.where(circ | (e_true <= 0), 0.0, np.log(np.clip(e_true / emax, 1e-12, None)))
    piv = float(fc["m1_pivot_msun"])
    lm1 = np.log(m1 / piv)
    lo_m = m1 < piv
    ob_r, ab_r = mob[rel], mab[rel]
    # MP-Q42b normalization grid: MdS17 p_q(q | M1, P) on 0.3 <= q <= 1 (midpoint rule), per draw
    ng = int(fc["q_norm_grid_points"])
    qe = np.linspace(0.3, 1.0, ng + 1)
    qg = 0.5 * (qe[1:] + qe[:-1])
    Pq = mds.q_density(qg[None, :], m1s[:, None], lp_true[:, None], table, m1_interpolation=tgt.provisional_m1_interpolation) * np.diff(qe)[None, :]
    Pq = Pq / Pq.sum(axis=1, keepdims=True)
    lq_g = np.log(qg)
    twin_g = qg >= fc["twin_q_min"]

    def log_mod(th: np.ndarray, normalized: bool) -> np.ndarray:
        lnA, alo, ahi, gP, lnL, dq, lnT, de = th[:8]
        lm = lnA + np.where(lo_m, alo, ahi) * lm1
        lm += gP * (lp_true - fc["log_p_tilt_pivot"])
        lm += lnL / (1.0 + np.exp(-(lp_true - fc["long_p_centre"]) / fc["long_p_width"]))
        lm += dq * np.log(q) + lnT * (q >= fc["twin_q_min"])
        if normalized:
            lm -= np.log(Pq @ np.exp(dq * lq_g + lnT * twin_g))
        eta1 = np.maximum(eta_raw + de, floor)
        lm += np.where(circ, 0.0, np.log((eta1 + 1.0) / (eta0 + 1.0)) + (eta1 - eta0) * log_e_ratio)
        return lm

    W = {}
    for key, gl, uw in (("old_noW", old_gens, False), ("new_noW", gens, False), ("new_cmdW", gens, True), ("g28_noW", [28], False)):
        W[key] = mixture_weights(z, gl, uw)[rel] / s_low

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
    rob = np.where(r_P <= PMAX, bin_index([r_g, r_d, np.log10(r_P), r_e], oe), -1)
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
    big = k_o >= 0.01 * N_o
    rep = ["#391 rung 3 refit (spec §12.10): gens 23-28 (generation 28 re-centred), MP-Q42b + MP-Q43b", "",
           f"real: {int(N_o)} C1 orbits / {int(k_a.sum())} accelerations; mock draws in the likelihood: {rel.size} "
           f"({int(np.sum(gen_r == 28))} from generation 28)", ""]

    # ================= A. sampling gain at the first-fit best θ =================
    ff = json.loads(args.first_fit.read_text())["noW"]
    th1 = np.array(ff["theta"][:8], float)
    lm_ff = log_mod(th1, normalized=False)
    cpu = np.asarray(z["cpu_seconds"], float)
    gen_all = np.asarray(z["generation"])
    cpu_old = float(cpu[np.isin(gen_all, old_gens)].sum() / 3600)
    cpu_28 = float(cpu[gen_all == 28].sum() / 3600)
    so = ob_r >= 0

    def ess_bins(w):
        mu = np.bincount(ob_r[so], weights=w[so], minlength=n_ob)
        v = np.bincount(ob_r[so], weights=w[so] ** 2, minlength=n_ob)
        return np.where(v > 0, mu**2 / np.maximum(v, 1e-300), 0.0), ps.kish_ess(w[so])

    eb = {}
    for key in ("old_noW", "new_noW", "g28_noW"):
        eb[key] = ess_bins(W[key] * np.exp(lm_ff))
    ess_old_b, ess_old = eb["old_noW"]
    ess_new_b, ess_new = eb["new_noW"]
    ess_28_b, ess_28 = eb["g28_noW"]
    rep += ["A. Sampling gain of generation 28, at the first-fit best θ (PR #459 model; noW)",
            f"  CPU-h: gens 23-27 {cpu_old:.1f}, generation 28 {cpu_28:.2f}",
            f"  ESS of C1 orbits in the likelihood: gens 23-27 {ess_old:.0f} ({ess_old / cpu_old:.2f} per CPU-h); "
            f"generation 28 alone {ess_28:.0f} ({ess_28 / cpu_28:.1f} per CPU-h); gens 23-28 {ess_new:.0f} "
            f"(marginal {(ess_new - ess_old) / cpu_28:.1f} per CPU-h of generation 28)",
            f"  the {int(big.sum())} bins holding >= 1% of real orbits: ESS_b median {np.median(ess_old_b[big]):.1f} -> {np.median(ess_new_b[big]):.1f} "
            f"(generation 28 alone {np.median(ess_28_b[big]):.1f}); min {ess_old_b[big].min():.1f} -> {ess_new_b[big].min():.1f}; "
            f"bins at ESS_b >= {fc['min_ess_per_bin']}: {int(np.sum(ess_old_b[big] >= fc['min_ess_per_bin']))} -> {int(np.sum(ess_new_b[big] >= fc['min_ess_per_bin']))}"]
    rate = ess_28_b / max(cpu_28, 1e-9)
    deficit = np.maximum(0.0, fc["min_ess_per_bin"] - ess_new_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        need_h = np.where(deficit > 0, deficit / rate, 0.0)
    rep.append(f"  projection (ESS_b additive; generation-28 per-bin rates): CPU-h to bring every one of the {int(big.sum())} bins to ESS_b >= "
               f"{fc['min_ess_per_bin']}: {np.max(need_h[big]):.0f} (set by the worst bin); median bin {np.median(need_h[big]):.0f}; "
               f"to reach it in 90% of them {np.percentile(need_h[big], 90):.0f}")
    # Robust projection: generation 28 puts only 1-3 draws in each big bin, so its per-bin rates are Poisson-noisy.
    # Scale the whole mixture instead: ESS_b ~ (total ESS) x (bin share), extra total ESS at generation 28's rate.
    fac = fc["min_ess_per_bin"] / np.maximum(ess_new_b[big], 1e-9)
    proj = {}
    for lab, f_ in (("median bin", np.median(fac)), ("90% of bins", np.percentile(fac, 90)), ("all bins (worst)", fac.max())):
        extra = (f_ - 1.0) * ess_new
        proj[lab] = (float(f_), float(extra / (ess_28 / cpu_28)), float(extra / (ess_old / cpu_old)))
        rep.append(f"  projection by total-ESS scaling, {lab}: x{f_:.1f} ESS -> {proj[lab][1]:.0f} CPU-h at generation 28's "
                   f"{ess_28 / cpu_28:.1f} ESS/CPU-h ({proj[lab][2]:.0f} at the old {ess_old / cpu_old:.1f})")
    rep.append("  per bin (G, d, log P, e): real | ESS_b gens 23-27 -> 23-28 | gen 28 alone | CPU-h to 30 (per-bin rate; noisy)")
    for b_ in np.flatnonzero(big):
        gi, di, pi, ei = np.unravel_index(b_, shp)
        rep.append(f"    G {oe[0][gi]:g}-{oe[0][gi + 1]:g} d {oe[1][di]:g}-{oe[1][di + 1]:g} lP {oe[2][pi]:g}-{oe[2][pi + 1]:.2f} "
                   f"e {oe[3][ei]:g}-{oe[3][ei + 1]:g}: {int(k_o[b_])} | {ess_old_b[b_]:.1f} -> {ess_new_b[b_]:.1f} | {ess_28_b[b_]:.1f} | {need_h[b_]:.0f}")
    samp = {"cpu_h_old": cpu_old, "cpu_h_gen28": cpu_28, "ess_old": ess_old, "ess_new": ess_new, "ess_gen28": ess_28,
            "ess_per_cpu_h_old": ess_old / cpu_old, "ess_per_cpu_h_gen28": ess_28 / cpu_28,
            "big_bins_ess_old": ess_old_b[big].tolist(), "big_bins_ess_new": ess_new_b[big].tolist(),
            "big_bins_ess_gen28": ess_28_b[big].tolist(), "projected_cpu_h_all_bins": float(np.max(need_h[big])),
            "projected_cpu_h_median_bin": float(np.median(need_h[big])), "projected_cpu_h_90pct": float(np.percentile(need_h[big], 90)),
            "projection_total_ess_scaling": proj}

    # ================= B. spurious template from PR #455 =================
    sp = fc["spurious"]
    recs = [json.loads(line) for line in args.reinjection_log.read_text().splitlines()]
    # PR #455 log: DR3 source_ids (real), 1e15 + draw (engine cross-check), 2e15 + draw (symmetric mock)
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
    oz = np.load(args.old_cache, allow_pickle=True)
    oc = np.asarray(oz["cascade"], float)
    ms = sorted({r["source_id"] for r in sym_r})
    mi = np.array([s - MOCK_SYM_OFFSET for s in ms])
    ms_acc = np.array([np.mean([acc_c1(r) for r in sym_r if r["source_id"] == s]) for s in ms])
    ms_lp, ms_e = np.log10(oc[mi, 10]), oc[mi, 14]
    A_r, A_m = ra_acc.mean(), ms_acc.mean()
    pibar = max(0.0, (A_m - A_r) / A_m)

    def marginal(rv, mv, edges, label):
        ratios, lines = [], []
        rb_, mb_ = np.digitize(rv, edges) - 1, np.digitize(mv, edges) - 1
        for i in range(edges.size - 1):
            nr, nm = int(np.sum(rb_ == i)), int(np.sum(mb_ == i))
            if nr < sp["min_systems_per_marginal_bin"] or nm < sp["min_systems_per_marginal_bin"]:
                ratios.append(1.0)
                lines.append(f"{edges[i]:g}-{edges[i + 1]:.2f}: n_real {nr}, n_mock {nm} -> r = 1 (too sparse)")
                continue
            ar, am = ra_acc[rb_ == i].mean(), ms_acc[mb_ == i].mean()
            raw = max(0.0, (am - ar) / am) / pibar if pibar > 0 else 1.0
            r = 1.0 + (raw - 1.0) * nr / (nr + sp["shrink_n0"])
            ratios.append(r)
            lines.append(f"{edges[i]:g}-{edges[i + 1]:.2f}: n_real {nr}, n_mock {nm}, A_real {ar:.2f}, A_mock {am:.2f}, raw {raw:.2f} -> r {r:.2f}")
        rep.append(f"  {label}: " + "; ".join(lines))
        return np.array(ratios)

    rep += ["", "B. MP-Q43b spurious template from the PR #455 re-injection",
            f"  {len(rs)} real systems x 3 (A_real {A_r:.3f}); {len(ms)} mock accepted orbits x 3 (A_mock {A_m:.3f}); pi_bar {pibar:.3f}"]
    rr_P = marginal(ra_lp, ms_lp, oe[2], "log P marginal")
    rr_e = marginal(ra_e, ms_e, oe[3], "e marginal")
    dc = 0.5 * (oe[1][1:] + oe[1][:-1])
    tmpl = pibar * np.einsum("a,b,c,d->abcd", np.ones(shp[0]), np.ones(shp[1]), rr_P, rr_e)

    def spur(th):
        f_s, kd = th[8], th[9]
        return (f_s * tmpl * np.exp(kd * (dc - sp["distance_pivot_kpc"]))[None, :, None, None]).ravel() * k_o

    dsel = (oe[1][:-1] >= sp["prior_bin_distance_kpc"][0] - 1e-9) & (oe[1][1:] <= sp["prior_bin_distance_kpc"][1] + 1e-9)
    gsel = (oe[0][:-1] >= sp["prior_bin_g_mag"][0] - 1e-9) & (oe[0][1:] <= sp["prior_bin_g_mag"][1] + 1e-9)
    prior_mask = np.einsum("a,b,c,d->abcd", gsel, dsel, np.ones(shp[2]), np.ones(shp[3])).ravel() > 0
    k_prior = float(k_o[prior_mask].sum())

    def expected(th, wkey):
        w = W[wkey] * np.exp(log_mod(th, normalized=True))
        sa = ab_r >= 0
        return (w, np.bincount(ob_r[so], weights=w[so], minlength=n_ob), np.bincount(ob_r[so], weights=w[so] ** 2, minlength=n_ob),
                np.bincount(ab_r[sa], weights=w[sa], minlength=n_ab), np.bincount(ab_r[sa], weights=w[sa] ** 2, minlength=n_ab))

    def nll(th, wkey):
        _, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        s = spur(th)
        ll = say_loglike(k_o, mu_o + s, v_o) + say_loglike(k_a, mu_a, v_a)
        frac = s[prior_mask].sum() / max(k_prior, 1.0)
        ll += -0.5 * ((frac - sp["prior_mean"]) / sp["prior_sigma"]) ** 2
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    x0 = np.array([par[n][0] for n in NAMES], float)
    bounds = [(par[n][1], par[n][2]) for n in NAMES]
    rng = np.random.default_rng(args.seed)
    ff_names = json.loads(args.first_fit.read_text())["names"]
    results = {}
    rep.append("")
    for wkey in ("new_noW", "new_cmdW"):
        sols, best = [], None
        starts = [x0] + [np.array([rng.uniform(max(lo, x - 1), min(hi, x + 1)) for x, (lo, hi) in zip(x0, bounds)])
                         for _ in range(args.n_starts - 1)]
        for s0 in starts:
            r = minimize(nll, s0, args=(wkey,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "maxfun": 30000})
            r = minimize(nll, r.x, args=(wkey,), method="Powell", bounds=bounds, options={"maxiter": 20000, "xtol": 1e-4, "ftol": 1e-7})
            sols.append((float(r.fun), r.x.copy()))
            if best is None or r.fun < best.fun:
                best = r
        th = best.x
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
                H[i, j] = H[j, i] = ((f(h[i], 0) - 2 * f0 + f(-h[i], 0)) / h[i] ** 2 if i == j else
                                     (f(h[i], h[j]) - f(h[i], -h[j]) - f(-h[i], h[j]) + f(-h[i], -h[j])) / (4 * h[i] * h[j]))
        try:
            dg = np.diag(np.linalg.inv(H))
            err = np.sqrt(np.where(dg > 0, dg, np.nan))
        except np.linalg.LinAlgError:
            err = np.full(n, np.nan)
        w, mu_o, v_o, mu_a, v_a = expected(th, wkey)
        s = spur(th)
        frac = s[prior_mask].sum() / max(k_prior, 1.0)
        ess_b = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
        low = ess_b < fc["min_ess_per_bin"]
        ess_o, ess_a = ps.kish_ess(w[so]), ps.kish_ess(w[ab_r >= 0])
        near = np.array([x for f_, x in sols if f_ - best.fun < 5.0])
        dev = 2 * (say_loglike(k_o, np.where(k_o > 0, k_o, 1e-300), np.zeros(n_ob)) - say_loglike(k_o, mu_o + s, v_o))
        ffk = json.loads(args.first_fit.read_text())["noW" if wkey == "new_noW" else "cmdW"]
        rep.append(f"[{wkey}] converged {best.success}; -lnL {best.fun:.1f}; {len(near)}/{len(sols)} starts within 5 of the best")
        rep.append("    name        refit +- Laplace   [near-best range]   first fit (PR #459)")
        for i_, (nm, v, e_) in enumerate(zip(NAMES, th, err)):
            old = (f"{ffk['theta'][ff_names.index(nm)]:+.3f} +- {ffk['err'][ff_names.index(nm)]:.3f}" if nm in ff_names else "-")
            rep.append(f"    {nm:10s} {v:+.3f} +- {e_:.3f}   [{near[:, i_].min():+.2f}, {near[:, i_].max():+.2f}]   {old}")
        rep.append(f"    expected: mock orbits {mu_o.sum():.0f} + spurious {s.sum():.0f} = {mu_o.sum() + s.sum():.0f} vs real {N_o:.0f}; "
                   f"accelerations {mu_a.sum():.0f} vs {k_a.sum():.0f}")
        rep.append(f"    spurious: total {s.sum():.0f} ({s.sum() / N_o:.3f} of C1 orbits); share in the prior bin {frac:.3f} "
                   f"(prior {sp['prior_mean']} +- {sp['prior_sigma']}; pull {(frac - sp['prior_mean']) / sp['prior_sigma']:+.1f} sigma; "
                   f"first fit {ffk['spurious_frac_prior_bin']:.3f})")
        rep.append(f"    ESS at the best fit: orbits {ess_o:.0f}, accelerations {ess_a:.0f}" + ("  ** ESS-COLLAPSED **" if ess_o < fc["ess_collapse_threshold"] else ""))
        rep.append(f"    orbit bins with ESS_b < {fc['min_ess_per_bin']}: {int(low.sum())}/{n_ob} ({k_o[low].sum() / N_o:.1%} of real); "
                   f"of the {int(big.sum())} big bins: {int(np.sum(low & big))}")
        rep.append(f"    deviance (orbits, incl. MC var): {dev:.0f} over {int((k_o > 0).sum())} non-empty bins (first fit {ffk.get('poisson_deviance_orbits', float('nan')):.0f})")
        rep.append("")
        results[wkey] = dict(theta=th.tolist(), err=err.tolist(), nll=float(best.fun), converged=bool(best.success),
                             spurious_total=float(s.sum()), spurious_frac_prior_bin=float(frac), ess_orbits=float(ess_o),
                             ess_accel=float(ess_a), n_low_ess_bins=int(low.sum()), deviance=float(dev),
                             near_best=near.tolist())
        results[wkey]["_e"] = (w, mu_o, v_o, mu_a, v_a, s)

    # ================= PPC (noW) =================
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    w_b, mu_o, v_o, mu_a, v_a, s = results["new_noW"]["_e"]
    w_ff = W["old_noW"] * np.exp(lm_ff)
    idx_o = rel[so]
    mo = {"P_orb_days": casc[idx_o, 10], "G_mag": m_g[idx_o], "inv_parallax_mas_inv": m_d_o[idx_o],
          "eccentricity": casc[idx_o, 14], "cos_inclination": np.cos(np.radians(casc[idx_o, 16]))}
    with np.errstate(divide="ignore", invalid="ignore"):
        mo["f_m_msun"] = astrometric_mass_function(casc[idx_o, 17], casc[idx_o, 0], casc[idx_o, 10])
        rv = {"P_orb_days": r_P, "G_mag": r_g, "inv_parallax_mas_inv": r_d, "eccentricity": r_e,
              "f_m_msun": astrometric_mass_function(r_a0, r_plx, r_P), "cos_inclination": np.cos(r_inc)}
    rin = rob >= 0
    s_cells = s.reshape(shp)
    sp_marg = {"G_mag": (oe[0], s_cells.sum(axis=(1, 2, 3))), "inv_parallax_mas_inv": (oe[1], s_cells.sum(axis=(0, 2, 3))),
               "P_orb_days": (10 ** oe[2], s_cells.sum(axis=(0, 1, 3))), "eccentricity": (oe[3], s_cells.sum(axis=(0, 1, 2)))}
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    e_low = {}
    for ax, name in zip(np.ravel(axs), PANELS):
        sc, lo, hi = axes_cfg[name]
        edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=20)
        cen = np.sqrt(edges[1:] * edges[:-1]) if sc == "log" else 0.5 * (edges[1:] + edges[:-1])
        rn = np.histogram(rv[name][rin], bins=edges)[0]
        m_ff = np.histogram(mo[name], bins=edges, weights=w_ff[so])[0]
        m_new = np.histogram(mo[name], bins=edges, weights=w_b[so])[0]
        if name in sp_marg:
            se, sv = sp_marg[name]
            sfine = np.zeros(cen.size)
            for a_, b_, val in zip(se[:-1], se[1:], sv):
                if sc == "log" and a_ > 0:
                    ov = np.clip(np.log10(np.minimum(edges[1:], b_)) - np.log10(np.maximum(edges[:-1], a_)), 0, None)
                else:
                    ov = np.clip(np.minimum(edges[1:], b_) - np.maximum(edges[:-1], a_), 0, None)
                if ov.sum() > 0:
                    sfine += val * ov / ov.sum()
            m_new = m_new + sfine
        ax.step(cen, rn, where="mid", color="k", lw=2, label="real DR3 (C1)")
        ax.step(cen, m_ff, where="mid", color=series_style(1, style)["color"], ls="--", lw=2, label="first fit, mock part (gens 23-27)")
        ax.step(cen, m_new, where="mid", color=series_style(2, style)["color"], lw=2,
                label="refit + spurious" if name in sp_marg else "refit (no spurious term)")
        if sc == "log":
            ax.set_xscale("log")
        apply_axes_style(ax, style, xlabel=name, ylabel="count per bin")
        ax.legend(fontsize=style.tick_label_fontsize)
        if name == "eccentricity":
            e_low = {"real": float(rn[cen < 0.1].sum()), "refit": float(m_new[cen < 0.1].sum()), "edges": edges[:3].tolist()}
    fig.suptitle("Rung 3 refit PPC (noW, gens 23-28): six-panel", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_six_panel.png", dpi=int(cfg.diagnostics.figure_dpi))
    # first-fit e < 0.1 for comparison (its mock + its analytic flat-in-e spurious share)
    ffe = json.loads(args.first_fit.read_text())["noW"]
    e_edges = six_panel_bin_edges(*axes_cfg["eccentricity"][1:], scale=axes_cfg["eccentricity"][0], n_bins=20)
    e_ff_mock = np.histogram(mo["eccentricity"], bins=e_edges, weights=w_ff[so])[0][:2].sum()
    e_ff_spur = ffe["spurious_total"] * 0.1  # flat in e (MP-Q43 a)
    rep += ["PPC eccentricity, e < 0.1 (two fine six-panel bins): real {:.0f} | first fit {:.0f} (mock {:.0f} + flat spurious {:.0f}) | refit {:.0f}".format(
        e_low["real"], e_ff_mock + e_ff_spur, e_ff_mock, e_ff_spur, e_low["refit"])]

    R = k_o.reshape(shp)
    M = (mu_o + s).reshape(shp)
    Ra = k_a.reshape(ae[0].size - 1, ae[1].size - 1)
    Ma = mu_a.reshape(Ra.shape)
    ecen = 0.5 * (oe[3][1:] + oe[3][:-1])
    rep += ["", "PPC per distance bin: mock+spurious/real orbits | accelerations | <e> real / refit | orbit:accel real / refit"]
    for j in range(shp[1]):
        ro, m_ = R[:, j].sum(), M[:, j].sum()
        rep.append(f"  d {oe[1][j]:.1f}-{oe[1][j + 1]:.1f}: orbits {m_ / ro:.2f} [{int(ro)}]; accels {Ma[:, j].sum() / Ra[:, j].sum():.2f}; "
                   f"<e> {(R[:, j].sum(axis=(0, 1)) * ecen).sum() / ro:.3f} / {(M[:, j].sum(axis=(0, 1)) * ecen).sum() / m_:.3f}; "
                   f"o:a {ro / Ra[:, j].sum():.3f} / {m_ / Ma[:, j].sum():.3f}")
    for i in range(shp[0]):
        ro, m_ = R[i].sum(), M[i].sum()
        rep.append(f"  G {oe[0][i]:.1f}-{oe[0][i + 1]:.1f}: orbits {m_ / ro:.2f} [{int(ro)}]; accels {Ma[i].sum() / Ra[i].sum():.2f}")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(shp[1])
    lbl = [f"{a:g}-{b:g}" for a, b in zip(oe[1][:-1], oe[1][1:])]
    ax = axs[0]
    ax.plot(x, M.sum(axis=(0, 2, 3)) / R.sum(axis=(0, 2, 3)), "o-", lw=2, color=series_style(1, style)["color"], label="orbits (refit + spurious)")
    ax.plot(x, Ma.sum(axis=0) / Ra.sum(axis=0), "s--", lw=2, color=series_style(2, style)["color"], label="accelerations (refit)")
    ax.axhline(1, color="0.4", ls=":")
    ax.set_xticks(x, lbl)
    ax.set_ylim(0, 2)
    apply_axes_style(ax, style, xlabel="distance bin (kpc)", ylabel="mock / real")
    ax.legend(fontsize=style.tick_label_fontsize)
    ax = axs[1]
    ax.plot(x, (R.sum(axis=(0, 2)) * ecen).sum(axis=1) / R.sum(axis=(0, 2, 3)), "o-", color="k", lw=2, label="real")
    ax.plot(x, (M.sum(axis=(0, 2)) * ecen).sum(axis=1) / M.sum(axis=(0, 2, 3)), "s--", lw=2, color=series_style(1, style)["color"], label="refit + spurious")
    ax.set_xticks(x, lbl)
    apply_axes_style(ax, style, xlabel="distance bin (kpc)", ylabel="mean e (coarse-bin centres)")
    ax.legend(fontsize=style.tick_label_fontsize)
    ax = axs[2]
    big_idx = np.flatnonzero(big)
    ax.plot(np.arange(big_idx.size), ess_old_b[big], "o", color=series_style(1, style)["color"], label="gens 23-27")
    ax.plot(np.arange(big_idx.size), ess_new_b[big], "s", color=series_style(2, style)["color"], label="gens 23-28")
    ax.plot(np.arange(big_idx.size), ess_28_b[big], "^", color=series_style(3, style)["color"], label="generation 28 alone")
    ax.axhline(fc["min_ess_per_bin"], color="0.4", ls=":")
    ax.set_yscale("log")
    apply_axes_style(ax, style, xlabel="likelihood bin holding >= 1% of real orbits (index)", ylabel="ESS_b at first-fit θ")
    ax.legend(fontsize=style.tick_label_fontsize)
    fig.suptitle("Rung 3 refit (noW): counts and mean e vs distance; per-bin ESS gain of generation 28", fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "ppc_distance_and_ess.png", dpi=int(cfg.diagnostics.figure_dpi))

    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    out.update(names=list(NAMES), sampling=samp, template={"pi_bar": pibar, "r_P": rr_P.tolist(), "r_e": rr_e.tolist(),
                                                            "n_real_systems": len(rs), "n_mock_systems": len(ms)},
               e_below_0p1=e_low)
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
