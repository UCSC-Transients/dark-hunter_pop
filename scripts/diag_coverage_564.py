#!/usr/bin/env python3
"""#564 diagnosis: residual M2 under-coverage of DR3-style Gaussian orbit solutions (spec §13.12).

On the gate sample (accepted orbits, P_fit ≤ C1, training partition, θ_CO weights) of compact-object generations:

1. pulls (fit − truth)/σ of ϖ, P, e, φ_p, A, B, F, G and a0 per a0/σ_a0 bin: σ_MAD and |pull| > 2, > 3 fractions;
2. M2 coverage (68/95 %) per a0/σ_a0 bin and per M2 bin with the production posterior (``rung4_co_v1.yaml``);
3. selection conditioning: posterior draws weighted by 1 / P_pass(θ), P_pass a mock-calibrated logistic in
   log10(a0/σ_a0) and log10 P (orbital-stage draws: P(accepted | orbital fit));
4. our own prior choices: parallax prior ∝ ϖ⁻⁴ (uniform space density) and period prior ∝ 1/P, against flat.
Per-draw arrays go to ``<out>_<component>.npz`` for the error-scale calibration.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/diag_coverage_564.py --parent-dir <snapshot> --artifact a.h5 ... \\
        --fragment a.yaml ... --out output/coverage564/diag.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes

SNR_EDGES = (0.0, 10.0, 20.0, 50.0, 100.0, 1e9)
NAMES = list(ps.ORBIT_PARAMETER_NAMES)


def truth_ti(t: dict, i: int, sign: float) -> np.ndarray:
    """Truth photocentre Thiele–Innes (gaiamock's predict_astrometry_luminous_binary convention × the photocentre factor)."""
    m1, m2, p, plx = (float(t[k][i]) for k in ("m1_msun", "m2_msun", "period_days", "parallax_mas"))
    f = float(t.get("flux_ratio_cascade", t["flux_ratio"])[i])
    a = ((m1 + m2) * (p / 365.25) ** 2) ** (1 / 3) * plx
    s = sign * (m2 / (m1 + m2) - f / (1 + f))
    w, om, inc = float(t["omega_rad"][i]), float(t["Omega_rad"][i]), math.radians(float(t["inc_deg"][i]))
    c = math.cos(inc)
    return s * a * np.array([math.cos(w) * math.cos(om) - math.sin(w) * math.sin(om) * c, math.cos(w) * math.sin(om) + math.sin(w) * math.cos(om) * c,
                             -(math.sin(w) * math.cos(om) + math.cos(w) * math.sin(om) * c), -(math.sin(w) * math.sin(om) - math.cos(w) * math.cos(om) * c)])


def pull_stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return {"n": int(x.size)}
    return {"n": int(x.size), "median": float(np.median(x)), "sigma_mad": float(1.4826 * np.median(np.abs(x - np.median(x)))),
            "frac_gt2": float(np.mean(np.abs(x) > 2)), "frac_gt3": float(np.mean(np.abs(x) > 3))}


def wpit_cov(pit: np.ndarray, w: np.ndarray) -> dict:
    ok = np.isfinite(pit) & (w > 0)
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    p, ww = pit[ok], w[ok]
    return {"n": int(ok.sum()), "ess": float(ww.sum() ** 2 / (ww**2).sum()),
            "cov68": float(np.sum(ww * (np.abs(p - 0.5) < 0.34)) / ww.sum()), "cov95": float(np.sum(ww * (np.abs(p - 0.5) < 0.475)) / ww.sum()),
            "pit_lt16": float(np.sum(ww * (p < 0.16)) / ww.sum()), "pit_gt84": float(np.sum(ww * (p > 0.84)) / ww.sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True)
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--n-mc", type=int, default=4000)
    ap.add_argument("--max-per-component", type=int, default=3000, help="random subset of the gate sample (MC cost)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config()
    r4 = cob.load_rung4_config(args.rung4_config)
    rc = r4.mass_recovery
    frags = [ps.load_proposal_set_fragment(f) for f in args.fragment]
    gens = [f.proposal for f in frags]
    cob.check_co_generations(gens)
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, gens[0])
    parts = [ps.read_proposal_artifact(a)[:2] for a in args.artifact]
    tk = set.intersection(*(set(p[0]) for p in parts))
    okk = set.intersection(*(set(p[1]) for p in parts))
    t = {k: np.concatenate([p[0][k] for p in parts]) for k in tk}
    o = {k: np.concatenate([p[1][k] for p in parts]) for k in okk}
    w = cob.co_importance_weights(t, parent, gens, frags[0].target_co, partition="train")
    casc = np.asarray(o["cascade"], float)
    p0s, covs = o["orbit_p0"], o["orbit_cov"]
    orbital_stage = np.isin(np.asarray(o["solution_type"]).astype(str), ["twelve_parameter_orbital", "orbital_failed_cuts"]) & np.all(np.isfinite(covs.reshape(len(w), -1)), axis=1)
    acc = np.asarray(o["accepted_orbital"], bool)
    # (3) P(accepted | orbital fit) as a logistic in (log10 a0/σ_a0, log10 P) of the TRUE a0 over the fit's σ_a0 (mock-calibrated)
    a0_true = np.full(len(w), np.nan)  # the Halbwachs a0 of the truth TI
    for i in np.flatnonzero(orbital_stage):
        ti = truth_ti(t, i, 1.0)
        a0_true[i] = float(photocenter_a0_from_thiele_innes(*ti))
    with np.errstate(divide="ignore", invalid="ignore"):
        x1 = np.log10(a0_true / casc[:, 18])
        x2 = np.log10(np.asarray(t["period_days"], float))
    cal = orbital_stage & np.isfinite(x1) & np.isfinite(x2)
    from scipy.optimize import minimize
    from scipy.special import expit

    X = np.column_stack([np.ones(cal.sum()), x1[cal], x2[cal], x1[cal] * x2[cal]])
    yv = acc[cal].astype(float)

    def nlog(b: np.ndarray) -> float:
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - yv * z))

    beta = minimize(nlog, np.zeros(4), method="BFGS").x
    out: dict = {"pass_logistic_beta": beta.tolist(), "pass_calibration_n": int(cal.sum())}
    names_idx = {k: NAMES.index(k) for k in NAMES}
    iA = [names_idx[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")]
    rng = np.random.default_rng(564)
    # TI sign convention: choose the sign that makes the median A pull smallest
    pmax = rc.gate_period_max_days if rc.gate_period_max_days is not None else np.inf
    gsel = np.flatnonzero(acc & (casc[:, 10] <= pmax) & (w > 0) & orbital_stage)
    sgn = 1.0
    if gsel.size:
        smp = gsel[:500]
        med = {s: np.nanmedian([(p0s[i, iA[0]] - truth_ti(t, i, s)[0]) / math.sqrt(covs[i][iA[0], iA[0]]) for i in smp]) for s in (1.0, -1.0)}
        sgn = min(med, key=lambda s: abs(med[s]))
    out["ti_sign"] = sgn
    for comp, code in cob.COMPONENT_CODES.items():
        gate = acc & (np.asarray(t["co_component"]) == code) & (casc[:, 10] <= pmax) & (w > 0) & orbital_stage
        sel = np.flatnonzero(gate)
        if sel.size > args.max_per_component:
            sel = np.sort(rng.choice(sel, args.max_per_component, replace=False))
        n = sel.size
        pulls = {k: np.full(n, np.nan) for k in ("parallax", "period", "eccentricity", "phi_p", "A", "B", "F", "G", "a0")}
        pits = {k: np.full(n, np.nan) for k in ("production", "selection_conditioned", "plx_prior_vol", "p_prior_logu", "truth_m1")}
        snr = casc[sel, 17] / casc[sel, 18]
        for j, i in enumerate(sel):
            c, p0 = covs[i], p0s[i]
            sd = np.sqrt(np.diag(c))
            pulls["parallax"][j] = (p0[2] - t["parallax_mas"][i]) / sd[2]
            pulls["period"][j] = (p0[5] - t["period_days"][i]) / sd[5]
            pulls["eccentricity"][j] = (p0[6] - t["eccentricity"][i]) / sd[6]
            phi_t = (2 * math.pi * float(t["Tp_days"][i]) / float(t["period_days"][i])) % (2 * math.pi)
            pulls["phi_p"][j] = (((p0[7] - phi_t) + math.pi) % (2 * math.pi) - math.pi) / sd[7]
            ti = truth_ti(t, i, sgn)
            for k, jj, tv in zip("ABFG", iA, ti):
                pulls[k][j] = (p0[jj] - tv) / sd[jj]
            a0f = float(photocenter_a0_from_thiele_innes(*p0[iA]))
            g = np.zeros(12)
            for jj in iA:
                d = np.zeros(12)
                d[jj] = 1e-6 * max(abs(p0[jj]), 1e-3)
                g[jj] = (float(photocenter_a0_from_thiele_innes(*(p0 + d)[iA])) - float(photocenter_a0_from_thiele_innes(*(p0 - d)[iA]))) / (2 * d[jj])
            pulls["a0"][j] = (a0f - float(photocenter_a0_from_thiele_innes(*ti))) / math.sqrt(g @ c @ g)
            r = int(t["parent_row"][i])
            m1d = cob.m1_posterior_draws(parent, r, args.n_mc, rng, rc.m1_posterior)
            if not np.all(np.isfinite(m1d)):
                continue
            seed = int(rng.integers(2**31))
            truth = float(t["m2_msun"][i])
            for name, m1x in (("production", m1d), ("truth_m1", np.full(args.n_mc, float(t["m1_msun"][i])))):
                _, m2, wq, _ = cob.m2_posterior_draws(p0, c, m1x, n_draws=args.n_mc, random_seed=seed, flux_ratio=rc.assumed_flux_ratio, prior=rc.m2_posterior_prior)
                okd = np.isfinite(m2) & (wq > 0)
                if not okd.any():
                    continue
                pits[name][j] = np.sum(wq[okd][m2[okd] < truth]) / wq[okd].sum()
                if name != "production":
                    continue
                # draws (reproduce the same Gaussian sample for the extra weights)
                from darkhunter_pop import mc_mass_function as mcm

                sol = cob.cascade_parameter_set(p0, c)
                fac = mcm.factorize_covariance(sol.covariance_array(), eig_rel_floor=1e-12, eig_abs_floor=1e-18)
                dd = mcm.sample_multivariate_normal(sol.values_array(), fac, args.n_mc, np.random.default_rng(seed))
                a0d = photocenter_a0_from_thiele_innes(*(dd[:, k] for k in iA))
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = beta[0] + beta[1] * np.log10(a0d / casc[i, 18]) + beta[2] * np.log10(dd[:, 5]) + beta[3] * np.log10(a0d / casc[i, 18]) * np.log10(dd[:, 5])
                    ppass = np.clip(expit(z), 1e-3, 1.0)
                    plx, per = dd[:, 2], dd[:, 5]
                for vname, extra in (("selection_conditioned", 1.0 / ppass), ("plx_prior_vol", np.where(plx > 0, plx ** -4.0, 0.0)),
                                     ("p_prior_logu", np.where(per > 0, 1.0 / per, 0.0))):
                    ww = np.where(okd, wq * extra, 0.0)
                    ww = np.where(np.isfinite(ww), ww, 0.0)
                    if ww.sum() > 0:
                        pits[vname][j] = np.sum(ww[m2 < truth]) / ww.sum()
        wsel = w[sel]
        res: dict = {"n": int(n), "pulls_by_snr": {}, "coverage_by_snr": {}, "coverage_by_m2": {}, "variants": {}}
        for lo, hi in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
            m = (snr >= lo) & (snr < hi)
            res["pulls_by_snr"][f"{lo:g}-{hi:g}"] = {k: pull_stats(v[m]) for k, v in pulls.items()}
            res["coverage_by_snr"][f"{lo:g}-{hi:g}"] = wpit_cov(pits["production"][m], wsel[m])
        lm2 = np.log10(np.asarray(t["m2_msun"], float)[sel])
        e = np.asarray(rc.log10_m2_edges, float)
        for a, b in zip(e[:-1], e[1:]):
            m = (lm2 >= a) & (lm2 < b)
            if m.sum() >= 5:
                res["coverage_by_m2"][f"{10**a:.2f}-{10**b:.2f}"] = wpit_cov(pits["production"][m], wsel[m])
        for k, v in pits.items():
            res["variants"][k] = wpit_cov(v, wsel)
        res["pulls_all"] = {k: pull_stats(v) for k, v in pulls.items()}
        out[comp] = res
        np.savez(args.out.with_name(args.out.stem + f"_{comp}.npz"), idx=sel, w=wsel, snr=snr, m2=np.asarray(t["m2_msun"], float)[sel],
                 period_fit=casc[sel, 10], **{f"pull_{k}": v for k, v in pulls.items()}, **{f"pit_{k}": v for k, v in pits.items()})
        print(comp, json.dumps({k: res[k] for k in ("n", "variants", "coverage_by_snr")}, indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
