#!/usr/bin/env python3
"""Calibrate the rung-3 spurious probability π(x) on the re-injection samples (#391; spec §12.16).

For each re-injected real C1 orbit i (published orbit, 3 noise realizations through the current forward model),
k_i ~ Binomial(n_i, A_mock(x_i) · (1 − π(x_i))):
* A_mock(x): the mock's own self-consistency re-detection probability, logistic, fitted on the matched mock accepted
  orbits re-injected at their fitted orbit (same samples), on covariates that exist on both sides;
* π(x): logistic, the probability that the real solution is spurious / non-Keplerian.
Covariates are standardized; slopes carry an N(0, 1) shrinkage prior; inclusion is forward selection by ΔBIC with the
threshold of ``config/spuriousness_model.yaml`` (``bic_delta_include_covariate``), whose covariate definitions
(harmonic distance, F2, a0 and parallax S/N, G) are reused. Statistical over the sample, never per star.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln

OFF = 2 * 10**15
CANDIDATES_BOTH = ("g_mag", "bp_rp", "distance_kpc", "log10_period", "eccentricity", "log10_a0_snr", "log10_parallax_snr")
CANDIDATES_REAL_ONLY = ("goodness_of_fit", "harmonic_distance")


def load_samples(logs):
    real, sym = {}, {}
    for tag, path in logs.items():
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            sid = int(r["source_id"])
            if 10**15 <= sid < OFF + 10**7 and not (OFF <= sid < OFF + 10**7):
                continue  # engine cross-check draws
            ok = bool(r["accepted"]) and r["period"] is not None and np.isfinite(r["period"]) and r["period"] <= 830.4
            d = sym if OFF <= sid < OFF + 10**7 else real
            k, n, t = d.get(sid, (0, 0, tag))
            d[sid] = (k + int(ok), n + 1, tag)
    return real, sym


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", action="append", required=True, help="tag=path of a re-injection JSONL (e.g. far=..., near=...)")
    ap.add_argument("--old-cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--spuriousness-config", type=Path, default=Path("config/spuriousness_model.yaml"))
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import h5py
    import yaml
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.spuriousness_model import harmonic_distance

    sc = yaml.safe_load(args.spuriousness_config.read_text())
    dbic = float(sc["bic_delta_include_covariate"])
    fund, mmax = float(sc["scanning_law_fundamental_period_days"]), int(sc["harmonic_multiples_max"])
    logs = dict(s.split("=", 1) for s in args.log)
    real, sym = load_samples(logs)
    cfg = load_config()
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])

    # real covariates
    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        P = {k: h["published"][k][()] for k in ("source_id", "g_mag", "parallax", "parallax_error", "period", "eccentricity",
                                                "significance", "goodness_of_fit")}
    pos = {int(s): i for i, s in enumerate(P["source_id"])}
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    bprp = dict(zip(np.asarray(rt["source_id"], np.int64).tolist(),
                    (np.ma.filled(np.ma.asarray(rt["bp_mag"], float), np.nan) - np.ma.filled(np.ma.asarray(rt["rp_mag"], float), np.nan)).tolist()))
    rs = sorted(real)
    ii = np.array([pos[s] for s in rs])
    g = P["g_mag"][ii]
    plx = P["parallax"][ii] + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
    Xr = {"g_mag": g, "bp_rp": np.array([bprp.get(s, np.nan) for s in rs]), "distance_kpc": 1.0 / plx,
          "log10_period": np.log10(P["period"][ii]), "eccentricity": P["eccentricity"][ii],
          "log10_a0_snr": np.log10(P["significance"][ii]), "log10_parallax_snr": np.log10(P["parallax"][ii] / P["parallax_error"][ii]),
          "goodness_of_fit": P["goodness_of_fit"][ii],
          "harmonic_distance": np.array([harmonic_distance(p, fundamental_period_days=fund, multiples_max=mmax) for p in P["period"][ii]])}
    kr = np.array([real[s][0] for s in rs], float)
    nr = np.array([real[s][1] for s in rs], float)
    tag_r = np.array([real[s][2] for s in rs])
    # mock covariates (fitted cascade values of the re-injected mock accepted orbits)
    z = np.load(args.old_cache, allow_pickle=True)
    c = np.asarray(z["cascade"], float)
    ms = sorted(sym)
    di = np.array([s - OFF for s in ms])
    prow = np.asarray(z["parent_row"], np.int64)[di]
    Xm = {"g_mag": np.asarray(z["phot_g_mean_mag"], float)[di], "bp_rp": np.asarray(pc["bp_rp"], float)[prow],
          "distance_kpc": 1.0 / c[di, 0], "log10_period": np.log10(c[di, 10]), "eccentricity": c[di, 14],
          "log10_a0_snr": np.log10(c[di, 17] / c[di, 18]), "log10_parallax_snr": np.log10(c[di, 0] / c[di, 1])}
    km = np.array([sym[s][0] for s in ms], float)
    nm = np.array([sym[s][1] for s in ms], float)

    # standardization on the pooled real sample (template is applied to real orbits)
    mu = {k: float(np.nanmean(v)) for k, v in Xr.items()}
    sd = {k: float(np.nanstd(v)) or 1.0 for k, v in Xr.items()}

    def Z(X, keys):
        return np.column_stack([(np.asarray(X[k], float) - mu[k]) / sd[k] for k in keys]) if keys else np.zeros((len(next(iter(X.values()))), 0))

    def binom_ll(k, n, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * np.log(p) + (n - k) * np.log1p(-p)

    def fit_logit(k, n, Zm, offset_fn=None):
        """Maximize the binomial log-likelihood with an N(0, prior_sigma^2) prior on the slopes.

        Returns (beta, covariance from the BFGS inverse Hessian, data log-likelihood at the optimum).
        """
        def p_of(b):
            return expit(b[0] + Zm @ b[1:]) if offset_fn is None else offset_fn(b)

        def nll(b):
            return -(np.sum(binom_ll(k, n, p_of(b))) - 0.5 * np.sum(b[1:] ** 2) / args.prior_sigma**2)

        r = minimize(nll, np.zeros(Zm.shape[1] + 1), method="BFGS")
        return r.x, r.hess_inv, float(np.sum(binom_ll(k, n, p_of(r.x))))

    def finite_rows(X, keys, n):
        ok = np.ones(n, bool)
        for k in keys:
            ok &= np.isfinite(np.asarray(X[k], float))
        return ok

    def forward(kind):
        """Forward selection by ΔBIC; kind 'mock' (A_mock) or 'pi' (needs A_mock fixed)."""
        cands = list(CANDIDATES_BOTH) + ([] if kind == "mock" else list(CANDIDATES_REAL_ONLY))
        chosen, trail = [], []
        while True:
            best = None
            for cv in cands:
                if cv in chosen:
                    continue
                keys = chosen + [cv]
                res = fit_model(kind, keys)
                base = fit_model(kind, chosen, rows=res["rows"])
                dB = (-2 * base["ll"] + np.log(res["N"]) * (len(chosen) + 1)) - (-2 * res["ll"] + np.log(res["N"]) * (len(keys) + 1))
                trail.append((tuple(keys), round(float(dB), 2)))
                if dB >= dbic and (best is None or dB > best[1]):
                    best = (cv, dB)
            if best is None:
                return chosen, trail
            chosen.append(best[0])

    state = {}

    def fit_model(kind, keys, rows=None):
        if kind == "mock":
            ok = finite_rows(Xm, keys, km.size) if rows is None else rows
            Zm = Z({k: np.asarray(v)[ok] for k, v in Xm.items()}, keys)
            b, cov, ll = fit_logit(km[ok], nm[ok], Zm)
            return {"beta": b, "cov": cov, "ll": ll, "N": float(nm[ok].sum()), "rows": ok, "keys": keys}
        Am = state["A_mock_fn"]
        ok = finite_rows(Xr, keys + state["mock_keys"], kr.size) if rows is None else rows
        Zr = Z({k: np.asarray(v)[ok] for k, v in Xr.items()}, keys)
        Amock = Am({k: np.asarray(v)[ok] for k, v in Xr.items()})

        def pfn(b):
            return Amock * (1.0 - expit(b[0] + Zr @ b[1:]))

        b, cov, ll = fit_logit(kr[ok], nr[ok], Zr, offset_fn=pfn)
        return {"beta": b, "cov": cov, "ll": ll, "N": float(nr[ok].sum()), "rows": ok, "keys": keys}

    mk, mtrail = forward("mock")
    mres = fit_model("mock", mk)

    def A_mock_fn(X):
        return expit(mres["beta"][0] + Z(X, mk) @ mres["beta"][1:])

    state["A_mock_fn"], state["mock_keys"] = A_mock_fn, mk
    pk, ptrail = forward("pi")
    pres = fit_model("pi", pk)
    pi_hat = expit(pres["beta"][0] + Z({k: np.asarray(v) for k, v in Xr.items()}, pk) @ pres["beta"][1:])
    rep = [f"#391 spurious-template calibration (§12.16): {len(rs)} real systems ({', '.join(f'{t} {np.sum(tag_r == t)}' for t in logs)}), "
           f"{len(ms)} mock accepted orbits; {int(nr.sum())} + {int(nm.sum())} realizations",
           f"ΔBIC inclusion threshold {dbic} (config/spuriousness_model.yaml); slope prior N(0, {args.prior_sigma}^2) on standardized covariates",
           f"A_mock covariates selected: {mk or 'intercept only'}; coefficients {np.round(mres['beta'], 3).tolist()} ± {np.round(np.sqrt(np.diag(mres['cov'])), 3).tolist()}",
           f"pi covariates selected: {pk or 'intercept only'}; coefficients {np.round(pres['beta'], 3).tolist()} ± {np.round(np.sqrt(np.diag(pres['cov'])), 3).tolist()}",
           f"mean pi over the re-injected real systems: all {np.nanmean(pi_hat):.3f}; " + "; ".join(f"{t} {np.nanmean(pi_hat[tag_r == t]):.3f}" for t in logs),
           "", "forward-selection trail (keys: ΔBIC vs the previous model):"]
    rep += [f"  A_mock {k}: {d}" for k, d in mtrail]
    rep += [f"  pi {k}: {d}" for k, d in ptrail]
    ranges = {k: [float(np.nanmin(np.asarray(Xr[k]))), float(np.nanmax(np.asarray(Xr[k])))] for k in pk + mk}
    out = {"mock_keys": mk, "mock_beta": mres["beta"].tolist(), "mock_cov": np.asarray(mres["cov"]).tolist(),
           "pi_keys": pk, "pi_beta": pres["beta"].tolist(), "pi_cov": np.asarray(pres["cov"]).tolist(),
           "standardize_mean": mu, "standardize_sd": sd, "calibrated_range": ranges, "dbic_threshold": dbic,
           "prior_sigma": args.prior_sigma, "n_real": len(rs), "n_mock": len(ms), "samples": {t: int(np.sum(tag_r == t)) for t in logs},
           "harmonic": {"fundamental_period_days": fund, "multiples_max": mmax}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    (args.out.with_suffix(".txt")).write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
