#!/usr/bin/env python3
"""#557 diagnosis: why the rung-4 M2 posteriors under-cover (spec §13.9 gate 4b).

On accepted orbits of co_v1 generations (training partition, default θ_CO weights), reports
* pulls of the fitted parallax, period and photocentre a0 against truth, with the cascade covariance σ (σ_MAD and
  the fractions |pull| < 1, < 2);
* the log M1 pull of the truth M1 against the row's lognormal isochrone summary;
* M2 coverage (68/95 %) and median bias per component under variants that isolate each ingredient:
  ``baseline`` (as the selection-table build), ``truth_m1`` (M1 fixed at the truth), ``cov_x{k}`` (covariance × k²),
  ``truth_f`` (WD: inversion with the true flux ratio).

Usage::

    PYTHONPATH=src .venv/bin/python scripts/diag_coverage_557.py --parent-dir <snapshot> \\
        --artifact a.h5 b.h5 ... --fragment a.yaml b.yaml ... [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import mc_mass_function as mcm
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes


def _a0_true(t: dict, i: int) -> float:
    m1, m2, p = float(t["m1_msun"][i]), float(t["m2_msun"][i]), float(t["period_days"][i])
    f = float(t.get("flux_ratio_cascade", t["flux_ratio"])[i])
    a_au = ((m1 + m2) * (p / 365.25) ** 2) ** (1 / 3)
    return a_au * (m2 / (m1 + m2) - f / (1 + f)) * float(t["parallax_mas"][i])


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    mad = 1.4826 * np.median(np.abs(x - np.median(x)))
    return {"n": int(x.size), "median": float(np.median(x)), "sigma_mad": float(mad), "std": float(np.std(x)),
            "frac_lt1": float(np.mean(np.abs(x) < 1)), "frac_lt2": float(np.mean(np.abs(x) < 2))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--n-mc", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    r4 = cob.load_rung4_config(args.rung4_config)
    frags = [ps.load_proposal_set_fragment(f) for f in args.fragment]
    gens = [f.proposal for f in frags]
    co = cob.check_co_generations(gens)
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, gens[0])
    parts = [ps.read_proposal_artifact(a)[:2] for a in args.artifact]
    tk = set.intersection(*(set(p[0]) for p in parts))
    ok_ = set.intersection(*(set(p[1]) for p in parts))
    t = {k: np.concatenate([p[0][k] for p in parts]) for k in tk}
    o = {k: np.concatenate([p[1][k] for p in parts]) for k in ok_}
    w = cob.co_importance_weights(t, parent, gens, frags[0].target_co, partition="train")  # type: ignore[arg-type]
    orb = np.asarray(o["accepted_orbital"], bool) & (w > 0) & np.all(np.isfinite(o["orbit_cov"].reshape(len(w), -1)), axis=1)
    p0, cov = o["orbit_p0"], o["orbit_cov"]
    names = list(ps.ORBIT_PARAMETER_NAMES)
    ip, iP = names.index("parallax"), names.index("period")
    iA = [names.index(k) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")]
    rng = np.random.default_rng(557)
    out: dict = {}
    iso = parent.isochrone
    for comp, code in cob.COMPONENT_CODES.items():
        sel = np.flatnonzero(orb & (t["co_component"] == code))
        pulls = {"parallax": [], "period": [], "a0": [], "log_m1": []}
        variants = {"baseline": [], "truth_m1": [], "cov_x1.2": [], "truth_f": [], "iso_log_m2": [], "iso_log_m2_truth_f": []}
        for i in sel:
            c = cov[i]
            s = np.sqrt(np.diag(c))
            pulls["parallax"].append((p0[i, ip] - t["parallax_mas"][i]) / s[ip])
            pulls["period"].append((p0[i, iP] - t["period_days"][i]) / s[iP])
            # a0 and its σ by linear propagation of the Halbwachs a0 through the full covariance (numerical gradient)
            def a0f(v: np.ndarray) -> float:
                return float(photocenter_a0_from_thiele_innes(v[iA[0]], v[iA[1]], v[iA[2]], v[iA[3]]))
            a0 = a0f(p0[i])
            g = np.zeros(12)
            for j in iA:
                d = np.zeros(12); d[j] = 1e-6 * max(abs(p0[i, j]), 1e-3)
                g[j] = (a0f(p0[i] + d) - a0f(p0[i] - d)) / (2 * d[j])
            pulls["a0"].append((a0 - _a0_true(t, i)) / np.sqrt(g @ c @ g))
            r = int(t["parent_row"][i])
            pulls["log_m1"].append((np.log10(t["m1_msun"][i]) - iso["log_m1_mean"][r]) / iso["log_m1_sigma"][r])  # type: ignore[index]
            m1d = cob.m1_posterior_draws(parent, r, args.n_mc, rng, r4.mass_recovery.m1_posterior)
            if not np.all(np.isfinite(m1d)):
                for v in variants.values():
                    v.append((np.nan, np.nan))
                continue
            seed = int(rng.integers(2**31))
            truth = float(t["m2_msun"][i])
            fr = float(t.get("flux_ratio_cascade", t["flux_ratio"])[i])
            for name, (m1x, cx, fx, pr) in {"baseline": (m1d, c, 0.0, "flat_thiele_innes"),
                                             "truth_m1": (np.full(args.n_mc, float(t["m1_msun"][i])), c, 0.0, "flat_thiele_innes"),
                                             "cov_x1.2": (m1d, c * 1.44, 0.0, "flat_thiele_innes"),
                                             "truth_f": (m1d, c, fr, "flat_thiele_innes"),
                                             "iso_log_m2": (m1d, c, 0.0, "isotropic_log_m2"),
                                             "iso_log_m2_truth_f": (m1d, c, fr, "isotropic_log_m2")}.items():
                _, m2, wq, _ = cob.m2_posterior_draws(p0[i], cx, m1x, n_draws=args.n_mc, random_seed=seed, flux_ratio=fx, prior=pr)
                okd = np.isfinite(m2) & (wq > 0)
                if not okd.any():
                    variants[name].append((np.nan, np.nan))
                    continue
                m2o, wo = m2[okd], wq[okd]
                o_ = np.argsort(m2o)
                med = m2o[o_][min(np.searchsorted(np.cumsum(wo[o_]) / wo.sum(), 0.5), m2o.size - 1)]
                variants[name].append((float(np.sum(wo[m2o < truth]) / wo.sum()), float(med / truth - 1)))
        ww = w[sel]
        if args.out:  # per-draw arrays for the split analysis (#557)
            np.savez(args.out.with_name(args.out.stem + f"_{comp}.npz"), idx=sel, w=ww,
                     **{f"pull_{k}": np.asarray(v, float) for k, v in pulls.items()},
                     **{f"pit_{k}": np.asarray(v, float)[:, 0] for k, v in variants.items()},
                     **{f"bias_{k}": np.asarray(v, float)[:, 1] for k, v in variants.items()},
                     a0_snr=o["cascade"][sel, 17] / o["cascade"][sel, 18], period_fit=o["cascade"][sel, 10],
                     period_true=t["period_days"][sel], ecc_true=t["eccentricity"][sel], ecc_fit=o["cascade"][sel, 14],
                     plx_snr=o["cascade"][sel, 0] / o["cascade"][sel, 1], g=t["phot_g_mean_mag"][sel], m2=t["m2_msun"][sel],
                     f2=o["cascade"][sel, 21])
        res = {"n": int(sel.size), "ess": ps.kish_ess(ww), "pulls": {k: _stats(np.asarray(v)) for k, v in pulls.items()}}
        for name, v in variants.items():
            a = np.asarray(v)
            okm = np.isfinite(a[:, 0])
            wm = ww[okm]
            pit, bias = a[okm, 0], a[okm, 1]
            res[name] = {f"coverage_{lv:g}": float(np.sum(wm * (np.abs(pit - 0.5) < lv / 2)) / wm.sum()) for lv in r4.mass_recovery.coverage_levels}
            res[name].update({f"coverage_{lv:g}_unweighted": float(np.mean(np.abs(pit - 0.5) < lv / 2)) for lv in r4.mass_recovery.coverage_levels})
            res[name]["median_bias_unweighted"] = float(np.median(bias))
        out[comp] = res
        print(comp, json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
