#!/usr/bin/env python3
"""#597 iteration 2 (EPOCH_MODEL_SPEC §8.16, MOCK_POPULATION_SPEC §12.41): calibrate the remaining st1/e2 pieces on the
gens 47–50 population post hoc (no simulation), against DR3 C1 orbits by solution type.

Order (two passes of B–D, common random numbers for the routing draws):

A. **e2 at G < 10.** The same-orbit G < 10 ratio rests on 42 realizations. The knot at G 9.5 is re-solved so that the
   Orbital-routed population F2 median at G 9–10 equals DR3's Orbital median, and it is reported next to the same-orbit
   value. The G < 10 exclusion of the fit stays.
B. **Orbital σ_a0 offset jointly with the edge.** The Orbital offsets on σ_ϖ, σ_e and σ_P stay as measured on the same
   orbits. The σ_a0 offset is λ × the same-orbit value. λ ∈ [0, 1] minimises
   Σ_G [(z99_mock − z99_DR3)/0.02]² + Σ_G [((1 − λ) Δ_same)/0.02]², over the Orbital population edge (99th percentile of
   log z at G 10–11 / 11–12 / 12–13) and the same-orbit σ_a0 median residual.
C. **AstroSpectroSB1 F2 quantile map** per G bin (< 10, 10–11, 11–12, 12–13): mock quantiles of the e2-transformed F2 of
   AstroSpectroSB1-routed accepted draws mapped onto DR3 AstroSpectroSB1 quantiles (1, 5, 10, 25, 50, 75, 90, 95, 99%).
D. **Routing intercept δ(G)** at knots 9.5 / 10.5 / 11.5 / 12.5: bisection per bin so the post-cut AstroSpectroSB1
   fraction (unweighted mock) equals DR3's.

Writes ``iter2.json`` (parameters, checks of the type fraction against G, P, K1 and colour, both weightings) and a
config-ready block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_common_v21tr1 as dc  # noqa: E402
from calibrate_source_scale_595 import chi2red_from_f2, f2_from_chi2red  # noqa: E402
from validate_st1_population_597 import C_EDGES, K1_EDGES, LP_EDGES, frac_table, k1_of  # noqa: E402

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
GB = [(6.0, 10.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0)]
KN = [9.5, 10.5, 11.5, 12.5]


def main(argv=None) -> int:
    import h5py

    from darkhunter_pop.config_loader import load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--offsets", type=Path, default=Path("docs/gate_noise_colour/st1/offsets.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load_config()
    sec = cfg.dr3.epoch_model
    stc, e2c = sec.solution_type_model, sec.bright_stated_refit
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    off_same = json.loads(a.offsets.read_text())
    M = dc.load_mock()
    cas, g = M["cascade"], M["phot_g_mean_mag"]
    sol = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in M["solution_type"]])
    P = cas[:, 10]
    cand = np.isin(sol, ["twelve_parameter_orbital", "orbital_failed_cuts"]) & (P > 0) & (P <= dc.C1_PMAX) & (g >= 6) & (g < 19.5)
    bps = []
    for gg in (47, 48, 49, 50):
        with h5py.File(dc.V / f"single_gen{gg}.h5", "r") as f:
            bps.append(f["truth/sys_bp_rp"][()])
    bp = np.concatenate(bps)
    k1 = k1_of(cas[:, 17], cas[:, 0], P, cas[:, 14], cas[:, 16])
    nu = np.clip(cas[:, 20] - 12, 20, None)
    c = np.clip(np.where(np.isfinite(bp), bp, 1.0), *stc.colour_clip)
    grvs = g - np.polyval(stc.grvs_poly, c)
    feat = {"logK1": np.log10(np.clip(k1, 1e-3, None)), "bp_rp": c, "bp_rp2": c * c, "logP": np.log10(np.where(P > 0, P, 1.0)), "e": cas[:, 14]}
    X = np.column_stack([np.ones(g.size), grvs] + [np.maximum(0.0, grvs - k) for k in stc.grvs_knots] + [feat[t] for t in stc.terms])
    z0 = X @ np.asarray(stc.beta)
    rng = np.random.default_rng(5972)
    U = rng.random((a.n_rep, g.size))
    # DR3
    with h5py.File(dc.D / "output/gate390/published_orbits.h5", "r") as f:
        Pp = f["published"]
        pub = {k: Pp[k][()] for k in ("source_id", "g_mag", "period", "eccentricity", "parallax", "significance", "goodness_of_fit", "ruwe",
                                      "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        rtyp = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in Pp["nss_solution_type"][()]])
    with h5py.File(dc.D / "data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns/columns.h5", "r") as f:
        cmap = dict(zip(f["source_id"][()].tolist(), f["bp_rp"][()].tolist()))
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u_ = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v_ = A_ * G_ - B_ * F_
    a0d = np.sqrt(u_ + np.sqrt(np.clip(u_**2 - v_**2, 0, None)))
    rk1 = k1_of(a0d, pub["parallax"], pub["period"], pub["eccentricity"], np.degrees(np.arccos(np.clip(v_ / a0d**2, -1, 1))))
    rbp = np.array([cmap.get(int(s), np.nan) for s in pub["source_id"]], float)
    rsel = np.isin(rtyp, ["Orbital", "AstroSpectroSB1"]) & (pub["period"] <= dc.C1_PMAX) & (pub["g_mag"] >= 6) & (pub["g_mag"] < 19.5)
    rA = rtyp == "AstroSpectroSB1"
    with np.errstate(invalid="ignore", divide="ignore"):
        rz = np.log10(pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1))
    dr3_frac = {gb: float(rA[rsel & (pub["g_mag"] >= gb[0]) & (pub["g_mag"] < gb[1])].mean()) for gb in GB}
    dr3_orb_f2 = {gb: float(np.median(pub["goodness_of_fit"][rsel & ~rA & (pub["g_mag"] >= gb[0]) & (pub["g_mag"] < gb[1])])) for gb in GB}
    dr3_orb_z99 = {gb: float(np.nanpercentile(rz[rsel & ~rA & (pub["g_mag"] >= gb[0]) & (pub["g_mag"] < gb[1]) & np.isfinite(rz)], 99)) for gb in GB}
    dr3_asb1_q = {gb: np.quantile(pub["goodness_of_fit"][rsel & rA & (pub["g_mag"] >= gb[0]) & (pub["g_mag"] < gb[1])], QS).tolist() for gb in GB}
    gi = lambda knots, vals, outside=0.0: np.where(g < stc.break_g, np.interp(g, knots, vals), outside)  # noqa: E731
    st_off = {k: list(getattr(stc, k)) for k in ("orbital_dlog_sigma_a0", "orbital_dlog_sigma_plx", "orbital_dlog_sigma_e", "orbital_dlog_sigma_p",
                                                  "asb1_dlog_sigma_a0", "asb1_dlog_sigma_plx", "asb1_dlog_sigma_e", "asb1_dlog_sigma_p")}
    state = {"r": list(e2c.ratio), "lam": 1.0, "delta": [0.0] * 4, "qmap": None}

    def run(state):
        r = np.where(g < e2c.break_g, np.interp(g, e2c.knots_g, state["r"]), 1.0)
        f2e = f2_from_chi2red(np.clip(chi2red_from_f2(cas[:, 21], nu) / r**2, 1e-6, None), nu)
        pA = 1.0 / (1.0 + np.exp(-(z0 + np.interp(g, KN, state["delta"]))))
        outs = []
        for rep in range(a.n_rep):
            isA = cand & (U[rep] < pA)
            dd = {}
            for q in ("sigma_a0", "sigma_plx", "sigma_e", "sigma_p"):
                o_orb = np.array(st_off[f"orbital_dlog_{q}"]) * (state["lam"] if q == "sigma_a0" else 1.0)
                dd[q] = np.where(isA, gi(stc.knots_g, st_off[f"asb1_dlog_{q}"]), gi(stc.knots_g, o_orb))
            f2 = f2e.copy()
            if state["qmap"] is not None:
                for j, (lo, hi) in enumerate(GB):
                    b = isA & (g >= lo) & (g < hi)
                    m_, d_ = np.asarray(state["qmap"]["mock"][j]), np.asarray(state["qmap"]["dr3"][j])
                    x = f2e[b]
                    f2[b] = np.where(x < m_[0], d_[0] + x - m_[0], np.where(x > m_[-1], d_[-1] + x - m_[-1], np.interp(x, m_, d_)))
            sa0, splx, se = cas[:, 18] * 10 ** dd["sigma_a0"], cas[:, 1] * 10 ** dd["sigma_plx"], cas[:, 15] * 10 ** dd["sigma_e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                a0s, plxs = cas[:, 17] / sa0, cas[:, 0] / splx
                ok = cand & (a0s > cuts.a0_over_err_min) & (f2 < cuts.goodness_of_fit_f2_max) & (plxs > cuts.parallax_over_error_times_period_min_days / P) \
                    & (a0s > cuts.a0_over_err_times_sqrt_period_min / np.sqrt(P)) & (se < cuts.sigma_e_ln_period_slope * np.log(P) + cuts.sigma_e_intercept)
                z = np.log10(a0s / np.sqrt(cas[:, 22] ** 2 - 1))
            outs.append({"acc": ok, "isA": isA, "f2": f2, "f2e": f2e, "z": z})
        return outs

    def in_g(gb):
        return (g >= gb[0]) & (g < gb[1])

    log = []
    # A: r at the G < 10 knot from the population Orbital F2 median at G 9-10
    gb9 = (9.0, 10.0)
    dr3_9 = float(np.median(pub["goodness_of_fit"][rsel & ~rA & (pub["g_mag"] >= 9) & (pub["g_mag"] < 10)]))
    lo_, hi_ = 0.7, 1.3
    for _ in range(25):
        mid = 0.5 * (lo_ + hi_)
        state["r"][0] = mid
        o = run(state)
        med = float(np.median(np.concatenate([x["f2"][x["acc"] & ~x["isA"] & (g >= 9) & (g < 10)] for x in o])))
        if med > dr3_9:
            lo_ = mid
        else:
            hi_ = mid
    r_pop_lt10 = 0.5 * (lo_ + hi_)
    state["r"][0] = r_pop_lt10
    log.append({"step": "A", "r_G_lt_10_population": r_pop_lt10, "r_G_lt_10_same_orbit": float(e2c.ratio[0]), "dr3_orbital_f2_median_G9_10": dr3_9})
    for it in range(2):
        # B: lambda
        best = None
        d_same = np.array(st_off["orbital_dlog_sigma_a0"][1:])
        for lam in np.linspace(0.0, 1.0, 11):
            state["lam"] = float(lam)
            o = run(state)
            dev = []
            for gb in GB[1:]:
                zz = np.concatenate([x["z"][x["acc"] & ~x["isA"] & in_g(gb)] for x in o])
                dev.append((np.nanpercentile(zz[np.isfinite(zz)], 99) - dr3_orb_z99[gb]) / 0.02)
            obj = float(np.sum(np.square(dev)) + np.sum(np.square((1 - lam) * d_same / 0.02)))
            if best is None or obj < best[0]:
                best = (obj, float(lam), [float(x) for x in dev])
        state["lam"] = best[1]
        log.append({"step": f"B{it}", "lambda": best[1], "objective": best[0], "edge_dev_sigma": best[2],
                    "same_orbit_residual_dex": [float((1 - best[1]) * x) for x in d_same]})
        # C: quantile map
        o = run({**state, "qmap": None})
        qm = {"mock": [], "dr3": []}
        for gb in GB:
            x = np.concatenate([oo["f2e"][oo["acc"] & oo["isA"] & in_g(gb)] for oo in o])
            qm["mock"].append(np.maximum.accumulate(np.quantile(x, QS)).tolist())
            qm["dr3"].append(np.maximum.accumulate(np.asarray(dr3_asb1_q[gb])).tolist())
        state["qmap"] = qm
        # D: routing intercepts
        for j, gb in enumerate(GB):
            lo_, hi_ = -3.0, 3.0
            for _ in range(22):
                mid = 0.5 * (lo_ + hi_)
                state["delta"][j] = mid
                o = run(state)
                fr = np.mean([x["isA"][x["acc"] & in_g(gb)].mean() for x in o])
                if fr > dr3_frac[gb]:
                    hi_ = mid
                else:
                    lo_ = mid
            state["delta"][j] = 0.5 * (lo_ + hi_)
        log.append({"step": f"CD{it}", "delta": list(state["delta"])})
        print(log[-2:], flush=True)
    # checks
    o = run(state)
    W = {"unweighted": np.ones(g.size), "default": M["w"]}
    checks = {}
    for name, xm, xr, edges in (("G", g, pub["g_mag"], [6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]), ("logP", feat["logP"], np.log10(pub["period"]), LP_EDGES),
                                ("logK1", feat["logK1"], np.log10(np.clip(rk1, 1e-3, None)), K1_EDGES), ("bp_rp", bp, rbp, C_EDGES)):
        bright_r, bright_m = pub["g_mag"] < 13, g < 13
        checks[name] = {"edges": edges, "dr3_G_lt_13": frac_table(np.where(rsel & bright_r, xr, np.nan), rA, np.ones(xr.size) * (rsel & bright_r), edges)}
        for wn, w in W.items():
            checks[name][f"mock_{wn}_G_lt_13"] = list(np.nanmean([frac_table(np.where(bright_m, xm, np.nan), x["isA"], w * x["acc"], edges) for x in o], axis=0))
            # the same with the accepted mock re-weighted to DR3's G distribution (G bins 6-9 ... 12-13), so the marginal
            # fractions in P, K1 and colour compare at a matched G mix
            gm_vals = []
            for x in o:
                wg = w * x["acc"]
                for gb in [(6, 9), (9, 10), (10, 11), (11, 12), (12, 13)]:
                    bm = (g >= gb[0]) & (g < gb[1])
                    nr = float((rsel & (pub["g_mag"] >= gb[0]) & (pub["g_mag"] < gb[1])).sum())
                    sm = float(wg[bm].sum())
                    wg = np.where(bm, wg * (nr / sm if sm > 0 else 0.0), wg)
                gm_vals.append(frac_table(np.where(bright_m, xm, np.nan), x["isA"], wg, edges))
            checks[name][f"mock_{wn}_G_lt_13_G_matched"] = list(np.nanmean(gm_vals, axis=0))
    res = {"state": state, "log": log, "checks": checks, "dr3_targets": {"fraction": {f"{k[0]}-{k[1]}": v for k, v in dr3_frac.items()},
                                                                        "orbital_f2_median": {f"{k[0]}-{k[1]}": v for k, v in dr3_orb_f2.items()},
                                                                        "orbital_z99": {f"{k[0]}-{k[1]}": v for k, v in dr3_orb_z99.items()}},
           "config": {"bright_stated_refit.ratio": state["r"], "orbital_dlog_sigma_a0": [state["lam"] * x for x in st_off["orbital_dlog_sigma_a0"]],
                      "routing_delta_knots_g": KN, "routing_delta": state["delta"], "asb1_f2_qmap_g_edges": [10.0, 11.0, 12.0],
                      "asb1_f2_qmap_mock": state["qmap"]["mock"], "asb1_f2_qmap_dr3": state["qmap"]["dr3"]},
           "same_orbit_offsets_used": off_same["config"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    for k, v in checks.items():
        print(k, "dr3", [round(x, 3) for x in v["dr3_G_lt_13"]], "mock", [round(x, 3) for x in v["mock_unweighted_G_lt_13"]],
              "G-matched", [round(x, 3) for x in v["mock_unweighted_G_lt_13_G_matched"]], "default G-matched", [round(x, 3) for x in v["mock_default_G_lt_13_G_matched"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
