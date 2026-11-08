#!/usr/bin/env python3
"""#564: calibrate the Student-t ν of the per-object likelihood by a0/σ_a0 bin on the HELD-OUT partition (spec §13.12).

For accepted orbits with P_fit ≤ the gate's C1 limit in the held-out SBC partition of compact-object generations
(both components), the a0 pull (Halbwachs a0 of the fit − truth) / σ_a0 (linear propagation through the full
covariance) is fitted with a unit-scale Student-t by maximum likelihood in each a0/σ_a0 bin. Prints the ν to put in
``mass_recovery.student_t_nu`` and writes a JSON record.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/calibrate_student_t_564.py --artifact a.h5 ... --fragment a.yaml ... \\
        --snr-edges 0 20 100 1e9 --out output/coverage564/student_t.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes


def _lean_load(paths: list[Path]) -> tuple[dict, dict]:
    """Only the datasets the pulls need, for accepted orbits only (keeps memory low on large artifacts)."""
    import h5py

    tk = ("draw_index", "m1_msun", "m2_msun", "period_days", "parallax_mas", "flux_ratio", "flux_ratio_cascade")
    t: dict[str, list] = {k: [] for k in tk}
    o: dict[str, list] = {"cascade": [], "orbit_p0": [], "orbit_cov": [], "accepted_orbital": []}
    for p in paths:
        with h5py.File(p, "r") as h:
            acc = np.asarray(h["outcome/accepted_orbital"][()], bool)
            idx = np.flatnonzero(acc)
            for k in tk:
                if f"truth/{k}" in h:
                    t[k].append(np.asarray(h[f"truth/{k}"][()])[idx])
                elif k == "flux_ratio_cascade":
                    t[k].append(np.asarray(h["truth/flux_ratio"][()])[idx])
            o["accepted_orbital"].append(acc[idx])
            o["cascade"].append(np.asarray(h["outcome/cascade"][idx, :]))
            o["orbit_p0"].append(np.asarray(h["outcome/orbit_p0"][idx, :]))
            o["orbit_cov"].append(np.asarray(h["outcome/orbit_cov"][idx, :, :]))
    return ({k: np.concatenate(v) for k, v in t.items() if v}, {k: np.concatenate(v) for k, v in o.items()})


def a0_pulls(t: dict, o: dict, sel: np.ndarray) -> np.ndarray:
    """(a0_fit − a0_true) / σ_a0 for the selected accepted orbits (Halbwachs photocentre a0, full covariance)."""
    names = list(ps.ORBIT_PARAMETER_NAMES)
    iA = [names.index(k) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")]
    pulls = np.full(sel.size, np.nan)
    for j, i in enumerate(sel):
        p0, c = np.asarray(o["orbit_p0"][i], float), np.asarray(o["orbit_cov"][i], float)
        m1, m2, p, plx = (float(t[k][i]) for k in ("m1_msun", "m2_msun", "period_days", "parallax_mas"))
        f = float(t.get("flux_ratio_cascade", t["flux_ratio"])[i])
        a0t = ((m1 + m2) * (p / 365.25) ** 2) ** (1 / 3) * plx * (m2 / (m1 + m2) - f / (1 + f))
        a0f = float(photocenter_a0_from_thiele_innes(*p0[iA]))
        g = np.zeros(12)
        for jj in iA:
            d = np.zeros(12)
            d[jj] = 1e-6 * max(abs(p0[jj]), 1e-3)
            g[jj] = (float(photocenter_a0_from_thiele_innes(*(p0 + d)[iA])) - float(photocenter_a0_from_thiele_innes(*(p0 - d)[iA]))) / (2 * d[jj])
        pulls[j] = (a0f - a0t) / math.sqrt(g @ c @ g)
    return pulls


def fit_nu(x: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float, float]:
    """ML ν of a unit-scale Student-t and its profile 68 % interval (Δ ln L = 0.5; a bound when it is not reached)."""
    lo_b, hi_b = (math.log(b) for b in bounds)

    def nll(ln: float) -> float:
        return -float(np.sum(stats.t.logpdf(x, df=math.exp(ln))))

    r = optimize.minimize_scalar(nll, bounds=(lo_b, hi_b), method="bounded")
    grid = np.linspace(lo_b, hi_b, 400)
    prof = np.array([nll(g) for g in grid]) - r.fun
    inside = grid[prof <= 0.5]
    return float(math.exp(r.x)), float(math.exp(inside.min())), float(math.exp(inside.max()))


SIGMA_OUT_MIN = 5.0  # the outlier component is broad by construction (prevents label switching with the core)
F_OUT_MAX = 0.5


def fit_outlier_mixture(x: np.ndarray, bounds: tuple[float, float]) -> dict:
    """(1 − f) Student-t(ν) + f N(0, σ_out²) with σ_out ≥ SIGMA_OUT_MIN and f ≤ F_OUT_MAX: ML ν_core, outlier
    fraction f and σ_out (a catastrophic-recovery check)."""
    lo_b, hi_b = (math.log(b) for b in bounds)

    def nll(th: np.ndarray) -> float:
        ln, lf, ls = th
        f = F_OUT_MAX / (1.0 + math.exp(-lf))
        so = SIGMA_OUT_MIN + math.exp(ls)
        nu_ = math.exp(min(max(ln, lo_b), hi_b))
        lp = np.logaddexp(math.log1p(-f) + stats.t.logpdf(x, df=nu_), math.log(f) + stats.norm.logpdf(x, scale=so))
        return -float(np.sum(lp))

    best = None
    for ln0 in (math.log(2.0), math.log(10.0), math.log(50.0)):
        r = optimize.minimize(nll, np.array([ln0, -3.0, math.log(20.0)]), method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    ln, lf, ls = best.x
    nu = float(np.clip(math.exp(ln), math.exp(lo_b), math.exp(hi_b)))
    return {"nu_core": round(nu, 2), "f_outlier": round(float(F_OUT_MAX / (1 + math.exp(-lf))), 4),
            "sigma_outlier": round(float(SIGMA_OUT_MIN + math.exp(ls)), 2), "neg_lnL": float(best.fun)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True)
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--snr-edges", type=float, nargs="+", required=True)
    ap.add_argument("--nu-bounds", type=float, nargs=2, default=(1.0, 300.0))
    ap.add_argument("--diagnostics", action="store_true",
                    help="#637: per-bin pull quantiles, |pull| > 5 counts, profile 68 %% ν intervals, outlier-mixture fit, pulls npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rc = cob.load_rung4_config(args.rung4_config).mass_recovery
    gens = [ps.load_proposal_set_fragment(f).proposal for f in args.fragment]
    co = cob.check_co_generations(gens)
    t, o = _lean_load(args.artifact)
    hold = cob.holdout_mask(t["draw_index"], co.sbc_holdout_fraction, co.sbc_holdout_salt)
    casc = np.asarray(o["cascade"], float)
    pmax = rc.gate_period_max_days if rc.gate_period_max_days is not None else np.inf
    sel = np.flatnonzero(hold & np.asarray(o["accepted_orbital"], bool) & (casc[:, 10] <= pmax)
                         & np.all(np.isfinite(o["orbit_cov"].reshape(casc.shape[0], -1)), axis=1))
    pulls = a0_pulls(t, o, sel)
    snr = casc[sel, 17] / casc[sel, 18]
    e = np.asarray(args.snr_edges, float)
    res: dict = {"snr_edges": e.tolist(), "nu": [], "n": [], "partition": "holdout", "n_total": int(sel.size)}
    if args.diagnostics:
        res.update({"nu_68": [], "pull_quantiles": [], "n_abs_pull_gt5": [], "n_abs_pull_gt10": [], "mixture": [],
                    "nu_without_abs_pull_gt5": []})
    for a, b in zip(e[:-1], e[1:]):
        x = pulls[(snr >= a) & (snr < b) & np.isfinite(pulls)]
        nu, lo, hi = fit_nu(x, tuple(args.nu_bounds))
        res["nu"].append(round(nu, 2))
        res["n"].append(int(x.size))
        if args.diagnostics:
            res["nu_68"].append([round(lo, 2), round(hi, 2)])
            res["pull_quantiles"].append({str(q): round(float(np.percentile(x, q)), 3) for q in (0.5, 2.5, 16, 50, 84, 97.5, 99.5)})
            res["n_abs_pull_gt5"].append(int(np.sum(np.abs(x) > 5)))
            res["n_abs_pull_gt10"].append(int(np.sum(np.abs(x) > 10)))
            res["mixture"].append(fit_outlier_mixture(x, tuple(args.nu_bounds)))
            xc = x[np.abs(x) <= 5]
            res["nu_without_abs_pull_gt5"].append([round(v, 2) for v in fit_nu(xc, tuple(args.nu_bounds))])
    print(json.dumps(res, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    if args.diagnostics:
        np.savez(args.out.with_suffix(".pulls.npz"), pull=pulls, snr=snr, draw_index=np.asarray(t["draw_index"])[sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
