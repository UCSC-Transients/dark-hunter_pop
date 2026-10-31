#!/usr/bin/env python3
"""#599 iteration 3 of e2 + st1 (EPOCH_MODEL_SPEC §8.16 (v), MOCK_POPULATION_SPEC §12.41): λ_SB1 jointly with λ_Orb.

Calibration only, post hoc on gens 47–50 (colour_v2.1+tr1), against DR3 C1 orbits by type. Starts from the configured
iteration-2 parameters (r(G), δ(G), F2 quantile map). The AstroSpectroSB1 σ offsets become λ_SB1 × the same-orbit values
(scope "a0plx": σ_a0, σ_ϖ; scope "all": also σ_e, σ_P); the Orbital σ_a0 offset is λ_Orb × the same-orbit value. Joint
objective on a 0.1 grid:

  Σ_G [(pooled median log σ_a0/σ_ϖ − DR3)/0.01]²  (G 10–11/11–12/12–13, unweighted mock)
+ Σ_type [(short-P share at G < 13 − DR3)/σ]²     (default weights; σ from DR3 binomial + mock Kish ESS)
+ Σ_G [(Orbital edge z99 − DR3)/0.02]² + Σ_G [((1 − λ_Orb) Δ_same)/0.02]²   (iteration-2 Orbital terms)

then the F2 quantile map and δ(G) are re-solved (common random numbers); two passes per scope. Writes ``iter3.json``.
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
from validate_st1_population_597 import k1_of  # noqa: E402

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
GB = [(6.0, 10.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0)]
SCOPES = {"a0plx": ("sigma_a0", "sigma_plx"), "all": ("sigma_a0", "sigma_plx", "sigma_e", "sigma_p")}
GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


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
    same = json.loads(a.offsets.read_text())["config"]
    same_off = {"orbital_sigma_a0": same["orbital_lsa0"], "orbital_sigma_plx": same["orbital_lsplx"], "orbital_sigma_e": same["orbital_lse"],
                "orbital_sigma_p": same["orbital_lsP"], "asb1_sigma_a0": same["asb1_lsa0"], "asb1_sigma_plx": same["asb1_lsplx"],
                "asb1_sigma_e": same["asb1_lse"], "asb1_sigma_p": same["asb1_lsP"]}
    M = dc.load_mock()
    cas, g, W = M["cascade"], M["phot_g_mean_mag"], M["w"]
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
    rng = np.random.default_rng(5993)
    U = rng.random((a.n_rep, g.size))
    r = np.where(g < e2c.break_g, np.interp(g, e2c.knots_g, e2c.ratio), 1.0)
    f2e = f2_from_chi2red(np.clip(chi2red_from_f2(cas[:, 21], nu) / r**2, 1e-6, None), nu)
    # DR3
    with h5py.File(dc.D / "output/gate390/published_orbits.h5", "r") as f:
        Pp = f["published"]
        pub = {k: Pp[k][()] for k in ("g_mag", "period", "parallax_error", "significance", "goodness_of_fit", "ruwe",
                                      "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        rtyp = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in Pp["nss_solution_type"][()]])
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u_ = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v_ = A_ * G_ - B_ * F_
    a0d = np.sqrt(u_ + np.sqrt(np.clip(u_**2 - v_**2, 0, None)))
    rsel = np.isin(rtyp, ["Orbital", "AstroSpectroSB1"]) & (pub["period"] <= dc.C1_PMAX) & (pub["g_mag"] >= 6) & (pub["g_mag"] < 19.5)
    rA = rtyp == "AstroSpectroSB1"
    rg = pub["g_mag"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rz = np.log10(pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1))
        rratio = np.log10((a0d / pub["significance"]) / pub["parallax_error"])
    inr = lambda gb: (rg >= gb[0]) & (rg < gb[1])  # noqa: E731
    ing = lambda gb: (g >= gb[0]) & (g < gb[1])  # noqa: E731
    dr3 = {"frac": {gb: float(rA[rsel & inr(gb)].mean()) for gb in GB},
           "orb_z99": {gb: float(np.nanpercentile(rz[rsel & ~rA & inr(gb) & np.isfinite(rz)], 99)) for gb in GB[1:]},
           "asb1_q": {gb: np.maximum.accumulate(np.quantile(pub["goodness_of_fit"][rsel & rA & inr(gb)], QS)).tolist() for gb in GB},
           "ratio": {gb: float(np.median(rratio[rsel & inr(gb)])) for gb in GB[1:]}}
    dr3["sp"], dr3["sp_n"] = {}, {}
    for tn, tr in (("Orbital", ~rA), ("AstroSpectroSB1", rA), ("all", np.ones(rA.size, bool))):
        b = rsel & tr & (rg < 13)
        dr3["sp"][tn], dr3["sp_n"][tn] = float(np.mean(pub["period"][b] < 100)), int(b.sum())
    gi = lambda knots, vals: np.where(g < stc.break_g, np.interp(g, knots, vals), 0.0)  # noqa: E731

    def run(st):
        pA = 1.0 / (1.0 + np.exp(-(z0 + np.interp(g, stc.routing_delta_knots_g, st["delta"]))))
        offs = {}
        for q in ("sigma_a0", "sigma_plx", "sigma_e", "sigma_p"):
            lo = st["lam_orb"] if q == "sigma_a0" else 1.0
            ls = st["lam_sb1"] if q in SCOPES[st["scope"]] else 1.0
            offs[q] = (gi(stc.knots_g, lo * np.asarray(same_off[f"orbital_{q}"])), gi(stc.knots_g, ls * np.asarray(same_off[f"asb1_{q}"])))
        outs = []
        for rep in range(a.n_rep):
            isA = cand & (U[rep] < pA)
            dd = {q: np.where(isA, v[1], v[0]) for q, v in offs.items()}
            f2 = f2e.copy()
            if st["qmap"] is not None:
                for j, gb in enumerate(GB):
                    b = isA & ing(gb)
                    m_, d_ = np.asarray(st["qmap"]["mock"][j]), np.asarray(st["qmap"]["dr3"][j])
                    x = f2e[b]
                    f2[b] = np.where(x < m_[0], d_[0] + x - m_[0], np.where(x > m_[-1], d_[-1] + x - m_[-1], np.interp(x, m_, d_)))
            sa0, splx, se = cas[:, 18] * 10 ** dd["sigma_a0"], cas[:, 1] * 10 ** dd["sigma_plx"], cas[:, 15] * 10 ** dd["sigma_e"]
            with np.errstate(invalid="ignore", divide="ignore"):
                a0s, plxs = cas[:, 17] / sa0, cas[:, 0] / splx
                ok = cand & (a0s > cuts.a0_over_err_min) & (f2 < cuts.goodness_of_fit_f2_max) & (plxs > cuts.parallax_over_error_times_period_min_days / P) \
                    & (a0s > cuts.a0_over_err_times_sqrt_period_min / np.sqrt(P)) & (se < cuts.sigma_e_ln_period_slope * np.log(P) + cuts.sigma_e_intercept)
                z = np.log10(a0s / np.sqrt(cas[:, 22] ** 2 - 1))
                lr = np.log10(sa0 / splx)
            outs.append({"acc": ok, "isA": isA, "f2": f2, "z": z, "lr": lr})
        return outs

    def metrics(o, lam_orb):
        res = {"ratio": {}, "orb_z99": {}, "sp": {}, "sp_err": {}, "frac": {}}
        for gb in GB[1:]:
            res["ratio"][gb] = float(np.mean([np.median(x["lr"][x["acc"] & ing(gb)]) for x in o]))
            zz = [np.nanpercentile(x["z"][x["acc"] & ~x["isA"] & ing(gb) & np.isfinite(x["z"])], 99) for x in o]
            res["orb_z99"][gb] = float(np.mean(zz))
        for gb in GB:
            res["frac"][gb] = float(np.mean([x["isA"][x["acc"] & ing(gb)].mean() for x in o]))
        for tn in ("Orbital", "AstroSpectroSB1", "all"):
            vals, ess = [], []
            for x in o:
                tm = (~x["isA"]) if tn == "Orbital" else (x["isA"] if tn == "AstroSpectroSB1" else np.ones(g.size, bool))
                b = x["acc"] & tm & (g < 13)
                w = W[b]
                vals.append(float(w[P[b] < 100].sum() / w.sum()))
                ess.append(float(w.sum() ** 2 / (w**2).sum()))
            s, e = float(np.mean(vals)), float(np.mean(ess))
            sd = dr3["sp"][tn]
            res["sp"][tn] = s
            res["sp_err"][tn] = float(np.sqrt(sd * (1 - sd) / dr3["sp_n"][tn] + max(s * (1 - s), 1e-4) / e))
        d_same = np.asarray(same_off["orbital_sigma_a0"][1:])
        terms = {"ratio": float(sum(((res["ratio"][gb] - dr3["ratio"][gb]) / 0.01) ** 2 for gb in GB[1:])),
                 "short_p": float(sum(((res["sp"][t] - dr3["sp"][t]) / res["sp_err"][t]) ** 2 for t in ("Orbital", "AstroSpectroSB1"))),
                 "orb_edge": float(sum(((res["orb_z99"][gb] - dr3["orb_z99"][gb]) / 0.02) ** 2 for gb in GB[1:])),
                 "orb_same": float(np.sum(((1 - lam_orb) * d_same / 0.02) ** 2))}
        res["terms"], res["J"] = terms, float(sum(terms.values()))
        return res

    def fmt(m):
        return {"J": round(m["J"], 2), "terms": {k: round(v, 2) for k, v in m["terms"].items()},
                "ratio": [round(m["ratio"][gb], 4) for gb in GB[1:]], "sp": {k: round(v, 4) for k, v in m["sp"].items()},
                "sp_err": {k: round(v, 4) for k, v in m["sp_err"].items()}, "orb_z99": [round(m["orb_z99"][gb], 4) for gb in GB[1:]],
                "frac": [round(m["frac"][gb], 3) for gb in GB]}

    base_qmap = {"mock": [list(x) for x in stc.asb1_f2_qmap_mock], "dr3": [list(x) for x in stc.asb1_f2_qmap_dr3]}
    lam_orb0 = float(np.median(np.asarray(stc.orbital_dlog_sigma_a0) / np.asarray(same_off["orbital_sigma_a0"])))
    it2 = {"scope": "a0plx", "lam_orb": lam_orb0, "lam_sb1": 1.0, "delta": list(stc.routing_delta), "qmap": base_qmap}
    out = {"dr3": {"ratio": [dr3["ratio"][gb] for gb in GB[1:]], "short_p": dr3["sp"], "short_p_n": dr3["sp_n"],
                   "frac": [dr3["frac"][gb] for gb in GB], "orb_z99": [dr3["orb_z99"][gb] for gb in GB[1:]]},
           "iteration2": fmt(metrics(run(it2), lam_orb0)), "scopes": {}}
    print("iteration 2 (configured):", out["iteration2"], flush=True)
    for scope in SCOPES:
        st = {**it2, "scope": scope, "delta": list(it2["delta"]), "qmap": base_qmap}
        log = []
        for it in range(2):
            grid = []
            for lo in GRID[3:]:
                for ls in GRID:
                    st.update(lam_orb=float(lo), lam_sb1=float(ls))
                    m = metrics(run(st), float(lo))
                    grid.append((m["J"], float(lo), float(ls)))
            grid.sort()
            st.update(lam_orb=grid[0][1], lam_sb1=grid[0][2])
            # quantile map on the AstroSpectroSB1 F2 (e2-transformed), then routing intercepts
            o = run({**st, "qmap": None})
            qm = {"mock": [], "dr3": []}
            for gb in GB:
                x = np.concatenate([oo["f2"][oo["acc"] & oo["isA"] & ing(gb)] for oo in o])
                qm["mock"].append(np.maximum.accumulate(np.quantile(x, QS)).tolist())
                qm["dr3"].append(dr3["asb1_q"][gb])
            st["qmap"] = qm
            for j, gb in enumerate(GB):
                lo_, hi_ = -3.0, 3.0
                for _ in range(22):
                    mid = 0.5 * (lo_ + hi_)
                    st["delta"][j] = mid
                    fr = float(np.mean([x["isA"][x["acc"] & ing(gb)].mean() for x in run(st)]))
                    lo_, hi_ = (lo_, mid) if fr > dr3["frac"][gb] else (mid, hi_)
                st["delta"][j] = 0.5 * (lo_ + hi_)
            m = metrics(run(st), st["lam_orb"])
            log.append({"pass": it, "best5": grid[:5], "lam_orb": st["lam_orb"], "lam_sb1": st["lam_sb1"], "delta": list(st["delta"]), "after": fmt(m)})
            print(scope, log[-1], flush=True)
        out["scopes"][scope] = {"state": {k: v for k, v in st.items()}, "log": log, "final": fmt(m)}
    best = min(out["scopes"], key=lambda k: out["scopes"][k]["final"]["J"])
    sb = out["scopes"][best]["state"]
    cfgblk = {"orbital_dlog_sigma_a0": [sb["lam_orb"] * x for x in same_off["orbital_sigma_a0"]],
              "routing_delta": sb["delta"], "asb1_f2_qmap_mock": sb["qmap"]["mock"], "asb1_f2_qmap_dr3": sb["qmap"]["dr3"]}
    for q in ("sigma_a0", "sigma_plx", "sigma_e", "sigma_p"):
        cfgblk[f"asb1_dlog_{q}"] = [(sb["lam_sb1"] if q in SCOPES[best] else 1.0) * x for x in same_off[f"asb1_{q}"]]
    out["adopted_scope"], out["config"] = best, cfgblk
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("adopted scope", best, json.dumps(cfgblk)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
