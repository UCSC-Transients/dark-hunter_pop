#!/usr/bin/env python3
"""#609 (spec §12.16 re-check): is the π̂(ϖ/σ_ϖ) choice of the #601 calibration an extrapolation between two samples?

Light analysis on the existing re-injection logs (no simulation). A_mock is held at the #601 calibration
(``mock_keys``/``mock_beta``), and π alternatives are fitted with the same binomial likelihood, the same N(0, σ²) slope
prior and the same standardisation. For each variant it reports the coefficients, the data log-likelihood and BIC, and
Σπ̂ over the real C1 orbits of the rung-3 fit domain (clamp to the variant's calibrated range), with σ_cal from 300
coefficient draws. It also reports the far/near/population covariate ranges and the population share outside each
calibrated range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_spurious_391 import load_samples  # noqa: E402

from darkhunter_pop import proposal_set as ps  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

KEYS = ("g_mag", "distance_kpc", "log10_a0_snr", "log10_parallax_snr")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, required=True)
    ap.add_argument("--population-dump", type=Path, required=True, help="rung-3 plot_dump.npz (real C1 orbits of the fit domain)")
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cal = json.loads(a.calibration.read_text())
    logs = dict(s.split("=", 1) for s in a.log)
    real, _ = load_samples(logs)
    cfg = load_config()
    parent = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= x) & (pg < y) & (snr > 20)]) for x, y in zip(ge[:-1], ge[1:])])
    with h5py.File(a.gate390 / "published_orbits.h5", "r") as h:
        P = {k: h["published"][k][()] for k in ("source_id", "g_mag", "parallax", "parallax_error", "significance")}
    pos = {int(s): i for i, s in enumerate(P["source_id"])}
    rs = sorted(real)
    ii = np.array([pos[s] for s in rs])
    g = P["g_mag"][ii]
    plx = P["parallax"][ii] + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
    X = {"g_mag": g, "distance_kpc": 1.0 / plx, "log10_a0_snr": np.log10(P["significance"][ii]),
         "log10_parallax_snr": np.log10(P["parallax"][ii] / P["parallax_error"][ii])}
    k = np.array([real[s][0] for s in rs], float)
    n = np.array([real[s][1] for s in rs], float)
    tag = np.array([real[s][2] for s in rs])
    mu, sd = cal["standardize_mean"], cal["standardize_sd"]

    def zs(Xd, keys):
        return np.column_stack([(np.asarray(Xd[c], float) - mu[c]) / sd[c] for c in keys]) if keys else np.zeros((len(Xd["g_mag"]), 0))

    am = expit(cal["mock_beta"][0] + zs(X, cal["mock_keys"]) @ np.asarray(cal["mock_beta"][1:]))
    z = np.load(a.population_dump)
    pop = {"g_mag": z["o_real_g"], "distance_kpc": z["o_real_d"], "log10_a0_snr": np.log10(z["o_real_a0snr"]),
           "log10_parallax_snr": np.log10(z["o_real_plxsnr"])}

    def binom_ll(kk, nn, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return gammaln(nn + 1) - gammaln(kk + 1) - gammaln(nn - kk + 1) + kk * np.log(p) + (nn - kk) * np.log1p(-p)

    step_x = (X["distance_kpc"] < 0.5).astype(float)
    step_p = (pop["distance_kpc"] < 0.5).astype(float)

    def fit(keys, rows, extra=None):
        Zm = zs({c: np.asarray(v)[rows] for c, v in X.items()}, keys)
        if extra is not None:
            Zm = np.column_stack([Zm, step_x[rows]])
        A = am[rows]

        def nll(b):
            return -(np.sum(binom_ll(k[rows], n[rows], A * (1 - expit(b[0] + Zm @ b[1:])))) - 0.5 * np.sum(b[1:] ** 2) / a.prior_sigma**2)

        r = minimize(nll, np.zeros(Zm.shape[1] + 1), method="BFGS")
        ll = float(np.sum(binom_ll(k[rows], n[rows], A * (1 - expit(r.x[0] + Zm @ r.x[1:])))))
        return r.x, np.asarray(r.hess_inv), ll, float(n[rows].sum())

    def pop_sum(keys, beta, cov, rows, extra=None):
        rng_ = {c: (float(np.nanmin(X[c][rows])), float(np.nanmax(X[c][rows]))) for c in keys}
        Zp = zs(dict({"g_mag": pop["g_mag"]}, **{c: np.clip(pop[c], *rng_[c]) for c in keys}), keys)
        if extra is not None:
            Zp = np.column_stack([Zp, step_p])
        ok = np.all(np.isfinite(Zp), axis=1)

        def s(b):
            p = expit(b[0] + Zp @ b[1:])
            return float(np.where(ok, p, np.median(p[ok])).sum())

        bs = np.random.default_rng(391).multivariate_normal(beta, cov, 300)
        out_frac = {c: float(np.mean((pop[c] < rng_[c][0]) | (pop[c] > rng_[c][1]))) for c in keys}
        return s(beta), float(np.std([s(b) for b in bs])), rng_, out_frac

    allr = np.ones(k.size, bool)
    far = tag == "far"
    res = {"ranges": {}, "variants": {}}
    for c in KEYS:
        res["ranges"][c] = {nm_: [float(np.nanpercentile(v, q)) for q in (0, 5, 50, 95, 100)] for nm_, v in
                            (("far", X[c][far]), ("near", X[c][~far]), ("population", pop[c]))}
    # common support in log ϖ/σ_ϖ: the overlap of the far and near ranges
    lo_c = max(np.nanmin(X["log10_parallax_snr"][far]), np.nanmin(X["log10_parallax_snr"][~far]))
    hi_c = min(np.nanmax(X["log10_parallax_snr"][far]), np.nanmax(X["log10_parallax_snr"][~far]))
    common = (X["log10_parallax_snr"] >= lo_c) & (X["log10_parallax_snr"] <= hi_c)
    res["common_support_log10_plx_snr"] = [float(lo_c), float(hi_c), int(common.sum()), int((common & far).sum()), int((common & ~far).sum())]
    variants = {"V0 current: plx_snr": (["log10_parallax_snr"], allr, None),
                "V1 a0_snr (previous choice)": (["log10_a0_snr"], allr, None),
                "V2 plx_snr + a0_snr": (["log10_parallax_snr", "log10_a0_snr"], allr, None),
                "V3 distance": (["distance_kpc"], allr, None),
                "V4 a0_snr + distance": (["log10_a0_snr", "distance_kpc"], allr, None),
                "V5 constant": ([], allr, None),
                "V6 common support: plx_snr": (["log10_parallax_snr"], common, None),
                "V7 common support: constant": ([], common, None),
                "V8 far only: plx_snr": (["log10_parallax_snr"], far, None),
                "V9 plx_snr + near step 1[d < 0.5 kpc]": (["log10_parallax_snr"], allr, "near_step"),
                "V10 near step 1[d < 0.5 kpc] only": ([], allr, "near_step")}
    base_ll = None
    for name, (keys, rows, extra) in variants.items():
        b, cov, ll, N = fit(keys, rows, extra)
        bic = -2 * ll + np.log(N) * (len(keys) + 1)
        S, sS, rg, of = pop_sum(keys, b, cov, rows, extra)
        # per-sample mean π̂ of the re-injected systems under this variant
        Zx = zs(X, keys) if extra is None else np.column_stack([zs(X, keys), step_x])
        pr = expit(b[0] + Zx @ b[1:])
        res["variants"][name] = {"keys": keys, "beta": b.tolist(), "err": np.sqrt(np.diag(cov)).tolist(), "ll": ll, "bic": float(bic), "n_systems": int(rows.sum()),
                                 "sum_pi": S, "sum_pi_sd": sS, "calibrated_range": rg, "population_outside_range": of,
                                 "mean_pi_far": float(np.mean(pr[far])), "mean_pi_near": float(np.mean(pr[~far]))}
    # the sample split itself: separate constants for far and near (does π differ between the samples at all?)
    b_f, c_f, ll_f, N_f = fit([], far)
    b_n, c_n, ll_n, N_n = fit([], ~far)
    res["split_constants"] = {"far": [float(expit(b_f[0])), float(np.sqrt(c_f[0, 0]))], "near": [float(expit(b_n[0])), float(np.sqrt(c_n[0, 0]))],
                              "ll_sum": ll_f + ll_n, "bic_split": float(-2 * (ll_f + ll_n) + np.log(N_f + N_n) * 2)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["ranges"]["log10_parallax_snr"]), res["common_support_log10_plx_snr"])
    for nm_, v in res["variants"].items():
        print(f"{nm_:32s} BIC {v['bic']:8.1f} Σπ̂ {v['sum_pi']:7.0f} ± {v['sum_pi_sd']:5.0f}  π far/near {v['mean_pi_far']:.3f}/{v['mean_pi_near']:.3f} "
              f"β {np.round(v['beta'], 3).tolist()} ± {np.round(v['err'], 3).tolist()} outside {v['population_outside_range']}")
    print("split constants", res["split_constants"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
