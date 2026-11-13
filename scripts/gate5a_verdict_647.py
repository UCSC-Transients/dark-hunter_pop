#!/usr/bin/env python3
"""Gate 5a verdict (R5-T7 #624, #647) and the parallax-error diagnosis it raised. Light: reads stored outputs only.

**Part A, per-object verdict** (RUNG5_SPEC §5 gate 5a). For each injected object (``output/rung5/gate5a``):
* outcomes over the realizations: C1-recovered (accepted orbit, P_fit ≤ the C1 ceiling), accepted outside C1
  (accepted orbit, P_fit above it: a domain-definition outcome, never a failure), published acceleration, rejected;
* the **response prediction**: the local mock probability of a C1 recovery and of an accepted-outside-C1 orbit, from
  the dark compact-object draws of the adopted version (gens 1006–1008) within a kernel around the object's (log M2,
  log P, G, log ϖ) (unweighted draws in the proposal measure: at fixed parameters the detection probability does not
  depend on the population weights; ESS reported);
* z = (f_obs − p_mock) / √(p(1 − p)/n + σ²_MC) for the C1 recovery; an object PASSES if |z| < ``--z-max`` (2), or is
  ``outside_c1`` if most accepted realizations fall above the ceiling, or ``untested`` if the mock kernel has fewer than
  ``--min-ess`` draws. Overall: Σ z² against χ²(n_tested).

**Part B, σ_ϖ diagnosis.** The ratio of the mock to the DR3 parallax error of accepted C1 orbits in G bins, for the
rung-3 v5 weighted luminous mock (gens 53–59, ``lum_weights_v5.npz``) against the published DR3 orbits
(``published_orbits.h5``), as weighted medians of log10 σ_ϖ with bootstrap errors; also at fixed ϖ (ϖ/σ_ϖ medians).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

G_EDGES = np.array([10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 16.0, 17.0, 18.0, 19.0])


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, 0.5 * c[-1]), x.size - 1)]) if x.size else float("nan")


def boot_wmedian(x: np.ndarray, w: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if x.size < 5:
        return float("nan"), float("nan")
    m = wmedian(x, w)
    b = [wmedian(x[i], w[i]) for i in (rng.integers(0, x.size, x.size) for _ in range(n))]
    return m, float(np.std(b))


def m2_from_orbit(a0_mas: float, plx: float, p_day: float, m1: float) -> float:
    mf = (a0_mas / plx) ** 3 / (p_day / 365.25) ** 2
    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if mid**3 / (m1 + mid) ** 2 < mf:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def outcome_fractions(recs: list[dict], pmax: float, accel_ok) -> dict:
    ok = [r for r in recs if "error" not in r]
    n = len(ok)
    acc = [r for r in ok if int(r.get("outcome", -1)) == 12 and r.get("accepted")]
    c1 = sum(1 for r in acc if float(r.get("period", np.inf)) <= pmax)
    out = sum(1 for r in acc if float(r.get("period", np.inf)) > pmax)
    ac = sum(1 for r in ok if accel_ok(r))
    return {"n": n, "c1": c1, "outside_c1": out, "acceleration": ac, "rejected": n - c1 - out - ac}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate5a-dir", type=Path, default=Path("output/rung5/gate5a"))
    ap.add_argument("--co-artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--host-m1", type=Path, default=Path("output/rung5/nss_host_m1_dr3.h5"))
    ap.add_argument("--known-truth", type=Path, default=Path("config/benchmarks/known_truth_gaia_bh.yaml"))
    ap.add_argument("--lum-artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--lum-weights", type=Path, default=Path("output/rung5/lum_weights_v5.npz"))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--c1-period-max", type=float, default=830.4)
    ap.add_argument("--kernel", type=float, nargs=4, default=(0.1, 0.1, 0.5, 0.1), help="half-widths: dex M2, dex P, mag G, dex ϖ")
    ap.add_argument("--min-ess", type=float, default=20.0)
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--one-sided", action="store_true",
                    help="fail only when the object is recovered LESS often than the mock predicts (re-injection at published, "
                         "detection-conditioned parameters biases recovery upward)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from scipy import stats

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.forward_model import passes_acceleration_publication_cuts

    cuts = load_config(Path("config/config.yaml")).dr3.selection_function_astrometric.acceleration_publication_cuts

    def accel_ok(r: dict) -> bool:
        o = int(r.get("outcome", -1))
        return o in (7, 9) and passes_acceleration_publication_cuts(
            n_params=o, significance=float(r.get("acceleration_significance", np.nan)),
            goodness_of_fit_f2=float(r.get("goodness_of_fit", np.nan)), cuts=cuts)

    # ---- Part A ------------------------------------------------------------------------------------------------
    plan = json.loads((a.gate5a_dir / "plan.json").read_text())
    recs = [json.loads(x) for x in (a.gate5a_dir / "inject.jsonl").read_text().splitlines()]
    known = {int(s["source_id"]): float(s["published_m2_msun"]) for s in yaml.safe_load(a.known_truth.read_text())["systems"]}
    with h5py.File(a.host_m1, "r") as h:
        hm = dict(zip(h["source_id"][()].astype(np.int64).tolist(), h["m1_mean"][()].tolist()))
    cols = {k: [] for k in ("m2", "p", "g", "plx", "acc", "pfit", "comp")}
    for art in a.co_artifact:
        with h5py.File(art, "r") as h:
            cols["m2"].append(h["truth/m2_msun"][()])
            cols["p"].append(h["truth/period_days"][()])
            cols["g"].append(h["truth/phot_g_mean_mag"][()])
            cols["plx"].append(h["truth/parallax_mas"][()])
            cols["comp"].append(h["truth/co_component"][()])
            cols["acc"].append(h["outcome/accepted_orbital"][()])
            cols["pfit"].append(h["outcome/cascade"][:, 10])
    co = {k: np.concatenate(v) for k, v in cols.items()}
    lm2, lp, lpl = np.log10(co["m2"]), np.log10(co["p"]), np.log10(np.clip(co["plx"], 1e-6, None))
    c1 = co["acc"] & (co["pfit"] <= a.c1_period_max)
    outc = co["acc"] & (co["pfit"] > a.c1_period_max)
    hw = a.kernel
    rows = []
    for t in plan["targets"]:
        sid, tr = int(t["source_id"]), t["truth"]
        m1 = hm.get(sid, float("nan"))
        m2 = known.get(sid) or (m2_from_orbit(tr["a0_mas"], tr["parallax"], tr["period"], m1) if np.isfinite(m1) else float("nan"))
        rr = [r for r in recs if int(r["source_id"]) == sid]
        fr = outcome_fractions(rr, a.c1_period_max, accel_ok)
        e12 = [float(r["parallax_error"]) for r in rr if int(r.get("outcome", -1)) == 12 and float(r.get("parallax_error", 0)) > 0]
        sig_ratio = float(np.median(e12) / tr["parallax_error"]) if e12 and tr.get("parallax_error", 0) > 0 else float("nan")
        k = ((np.abs(lm2 - math.log10(m2)) < hw[0]) & (np.abs(lp - math.log10(tr["period"])) < hw[1])
             & (np.abs(co["g"] - tr["g_mag"]) < hw[2]) & (np.abs(lpl - math.log10(tr["parallax"])) < hw[3])) if np.isfinite(m2) else np.zeros(lm2.size, bool)
        nk = int(k.sum())
        p_c1 = float(c1[k].mean()) if nk else float("nan")
        p_out = float(outc[k].mean()) if nk else float("nan")
        n = fr["n"]
        f_c1 = fr["c1"] / n if n else float("nan")
        var = (p_c1 * (1 - p_c1) / n + p_c1 * (1 - p_c1) / nk) if nk else float("nan")
        z = (f_c1 - p_c1) / math.sqrt(var) if nk and var > 0 else (0.0 if nk and f_c1 == p_c1 else float("nan"))
        acc_n = fr["c1"] + fr["outside_c1"]
        if acc_n and fr["outside_c1"] > fr["c1"]:
            verdict = "outside_c1"
        elif nk < a.min_ess:
            verdict = "untested"
        else:
            verdict = "pass" if (z > -a.z_max if a.one_sided else abs(z) < a.z_max) else "fail"
        rows.append({"name": t["name"], "source_id": sid, "type": t["nss_solution_type"], "g": round(tr["g_mag"], 2),
                     "p_day": round(tr["period"], 1), "plx_mas": round(tr["parallax"], 3), "m2_msun": round(m2, 2),
                     "m2_source": "published" if sid in known else "f=0 inversion at host isochrone M1",
                     **fr, "f_c1": f_c1, "mock_kernel_n": nk, "p_c1_mock": p_c1, "p_outside_c1_mock": p_out,
                     "z": z, "verdict": verdict, "sigma_plx_injection_over_dr3": sig_ratio})
    tested = [r for r in rows if r["verdict"] in ("pass", "fail")]
    chi2 = float(sum(r["z"] ** 2 for r in tested))
    ratios = np.array([r["sigma_plx_injection_over_dr3"] for r in rows], float)
    ratios = ratios[np.isfinite(ratios)]
    overall = {"n_pass": sum(r["verdict"] == "pass" for r in rows), "n_fail": sum(r["verdict"] == "fail" for r in rows),
               "n_outside_c1": sum(r["verdict"] == "outside_c1" for r in rows), "n_untested": sum(r["verdict"] == "untested" for r in rows),
               "chi2": chi2, "dof": len(tested), "p_value": float(stats.chi2.sf(chi2, len(tested))) if tested else None,
               "one_sided": bool(a.one_sided),
               "sigma_plx_injection_over_dr3_median": float(np.median(ratios)) if ratios.size else None,
               "sigma_plx_injection_over_dr3_p16_p84": np.percentile(ratios, [16, 84]).tolist() if ratios.size else None}

    # ---- Part B ------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(647)
    lw = np.load(a.lum_weights, allow_pickle=True)
    wkey = {(int(g), int(d)): float(w) for g, d, w in zip(lw["generation"], lw["draw_index"], lw["weight"])}
    mg, mplx, merr, mw = [], [], [], []
    for art in a.lum_artifact:
        with h5py.File(art, "r") as h:
            acc = np.asarray(h["outcome/accepted_orbital"][()], bool)
            idx = np.flatnonzero(acc)
            cl = np.asarray(h["outcome/cascade"][idx, :], float)
            gen = h["truth/generation"][()][idx]
            di = h["truth/draw_index"][()][idx]
            g = h["truth/phot_g_mean_mag"][()][idx]
        w = np.array([wkey.get((int(x), int(y)), 0.0) for x, y in zip(gen, di)])
        s = (w > 0) & (cl[:, 10] <= a.c1_period_max) & (cl[:, 1] > 0)
        mg.append(g[s]); mplx.append(cl[s, 0]); merr.append(cl[s, 1]); mw.append(w[s])
    mg, mplx, merr, mw = (np.concatenate(v) for v in (mg, mplx, merr, mw))
    with h5py.File(a.published_orbits, "r") as h:
        p = h["published"]
        rg, rplx, rerr, rper = p["g_mag"][()], p["parallax"][()], p["parallax_error"][()], p["period"][()]
    rs = (rper <= a.c1_period_max) & (rerr > 0) & np.isfinite(rg)
    rg, rplx, rerr = rg[rs], rplx[rs], rerr[rs]
    sig_rows = []
    for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:]):
        sm, sr = (mg >= lo) & (mg < hi), (rg >= lo) & (rg < hi)
        mm, me = boot_wmedian(np.log10(merr[sm]), mw[sm], a.n_boot, rng)
        rm, re = boot_wmedian(np.log10(rerr[sr]), np.ones(int(sr.sum())), a.n_boot, rng)
        fx = {}
        for plo, phi in ((1.0, 3.0), (3.0, 10.0)):
            am, ar = sm & (mplx >= plo) & (mplx < phi), sr & (rplx >= plo) & (rplx < phi)
            fx[f"plx_{plo:g}_{phi:g}"] = {"mock_median_snr": wmedian(mplx[am] / merr[am], mw[am]) if am.any() else None,
                                         "dr3_median_snr": float(np.median(rplx[ar] / rerr[ar])) if ar.any() else None,
                                         "n_mock": int(am.sum()), "n_dr3": int(ar.sum())}
        sig_rows.append({"g_lo": float(lo), "g_hi": float(hi), "n_mock": int(sm.sum()), "ess_mock": float(mw[sm].sum() ** 2 / np.sum(mw[sm] ** 2)) if sm.any() else 0.0,
                         "n_dr3": int(sr.sum()), "log10_ratio_mock_over_dr3": mm - rm, "log10_ratio_err": math.hypot(me, re),
                         "ratio": 10 ** (mm - rm), "fixed_parallax": fx})
    out = {"issue": [624, 647], "part_a": {"rows": rows, "overall": overall, "kernel_half_widths": list(hw), "z_max": a.z_max,
                                           "min_ess": a.min_ess},
           "part_b_sigma_parallax": sig_rows}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate5a_verdict.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(overall))
    for r in rows:
        print(f"{r['name'][:20]:20s} G {r['g']:5.2f} P {r['p_day']:7.1f} M2 {r['m2_msun']:5.2f}  f_C1 {r['f_c1']:.2f} "
              f"out {r['outside_c1']:2d} | mock n {r['mock_kernel_n']:4d} p_C1 {r['p_c1_mock']:.2f} p_out {r['p_outside_c1_mock']:.2f} z {r['z']:6.2f} {r['verdict']:10s} sig_inj/DR3 {r['sigma_plx_injection_over_dr3']:.2f}")
    for r in sig_rows:
        print(f"G {r['g_lo']:4.1f}-{r['g_hi']:4.1f}  sigma_plx mock/DR3 {r['ratio']:.3f} (log10 {r['log10_ratio_mock_over_dr3']:+.3f} ± {r['log10_ratio_err']:.3f})  "
              f"ESS {r['ess_mock']:.0f}  nDR3 {r['n_dr3']}  fixed plx 1-3: {r['fixed_parallax']['plx_1_3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
