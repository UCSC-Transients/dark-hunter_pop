#!/usr/bin/env python3
"""R5-T11 (#644): first DR3 WD / NS / BH mass-function fit, and the gate-5b closed loop (RUNG5_SPEC §2.6, §3, §5, §6).

Modes:
* ``fit``: MAP (several starts) on the real cell counts of ``--inputs`` (``scripts/rung5_build_inputs.py``), then
  ``--n-runs`` independent dynesty runs (seeds, nlive) for the §5 gate-5f agreement, and the §2.6 outputs:
  posterior summaries, the correlation matrix of the dark heights with the nuisances, the dark heights with each
  nuisance at ±1σ, posterior-vs-prior overlap for small-N bins, and the declared limitations (incl. #643).
* ``closed-loop``: gate 5b. Counts are simulated from the HELD-OUT templates (``--inputs`` built with
  ``--partition holdout``) at injected θ (§2.6: N_s at 0.5 / 1 / 2 × Σπ̂, s_A at 0.3 / 3, ln r at ±1 e-fold around the
  centre, dark heights at the injected values) and fitted with the TRAINING templates (``--fit-inputs``); reports the
  dark-height pulls from MAP + Laplace σ per realization.

Writes ``<out>/rung5_fit.json`` (or ``rung5_closed_loop.json``) and posterior samples (``.npz``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from darkhunter_pop import rung5_fit as rf


def _summ(samples: np.ndarray, names: list[str]) -> dict:
    q = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)
    return {n: [round(float(x), 4) for x in q[:, i]] for i, n in enumerate(names)}


def cmd_fit(a: argparse.Namespace) -> int:
    inp = rf.read_inputs(a.inputs)
    pr = rf.Priors()
    names = rf.parameter_names(inp)
    t0 = time.time()
    th_map, lp_map = rf.map_fit(inp, pr, n_starts=a.map_starts, seed=a.seed)
    mu, var = rf.expected_counts(th_map, inp)
    runs = []
    for r in range(a.n_runs):
        nlive = a.nlive if r % 2 == 0 else int(a.nlive * 1.5)
        res = rf.run_dynesty(inp, pr, nlive=nlive, seed=a.seed + 101 * r, dlogz=a.dlogz, maxcall=a.maxcall)
        np.savez(a.out / f"rung5_fit_samples_run{r}.npz", samples=res["samples"], names=np.array(names))
        runs.append({"nlive": nlive, "seed": a.seed + 101 * r, "logz": res["logz"], "logz_err": res["logz_err"],
                     "ncall": res["ncall"], "quantiles": _summ(res["samples"], names), "samples": res["samples"]})
    allsamp = np.vstack([r["samples"] for r in runs])
    dark_names = [n for n in names if n.startswith("log10_h_dark")]
    agree = {}
    if len(runs) > 1:
        for n in names:
            i = names.index(n)
            meds = [float(np.median(r["samples"][:, i])) for r in runs]
            sds = [float(np.std(r["samples"][:, i])) for r in runs]
            agree[n] = {"median_spread_over_sigma": float((max(meds) - min(meds)) / max(np.mean(sds), 1e-12)),
                        "width_ratio": float(max(sds) / max(min(sds), 1e-12))}
    out = {
        "issue": 644, "inputs": str(a.inputs), "inputs_meta": inp.meta, "limitations": list(rf.LIMITATIONS),
        "map": {"theta": dict(zip(names, np.round(th_map, 5).tolist())), "log_posterior": lp_map,
                "chi2_like": float(np.sum((inp.k_obs - mu) ** 2 / np.maximum(mu + var, 1e-9))), "n_cells": int(inp.k_obs.size),
                "k_total": float(inp.k_obs.sum()), "mu_total": float(mu.sum())},
        "runs": [{k: v for k, v in r.items() if k != "samples"} for r in runs],
        "gate_5f_agreement": agree,
        "posterior": _summ(allsamp, names),
        "correlation": rf.correlation_matrix(allsamp, names, dark_names + list(rf.NUISANCES)),
        "dark_heights_at_nuisance_pm1sigma": rf.nuisance_shift_table(inp, pr, th_map, allsamp, names),
        "small_n_overlap": rf.small_n_overlap(inp, pr, allsamp, th_map, expected_min=a.small_n_expected,
                                              threshold=a.overlap_threshold, collapse_floor=a.collapse_floor),
        "dark_edges_msun": inp.dark_edges.tolist(), "wd_edges_msun": inp.wd_edges.tolist(),
        "wall_s": round(time.time() - t0, 1),
    }
    (a.out / "rung5_fit.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("map", "gate_5f_agreement")}, indent=1, default=float)[:4000])
    return 0


def closed_loop_grid(inp: rf.Rung5Inputs, theta_ref: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """§2.6 gate-5b injections around a reference θ (the training MAP or a supplied one)."""
    s = rf._slices(inp)
    grid = [("reference", theta_ref.copy())]
    for f in (0.5, 2.0):
        t = theta_ref.copy()
        t[s["ln_ns"]] = math.log(f * inp.pihat_sum)
        grid.append((f"ns_x{f:g}", t))
    for v in (0.3, 3.0):
        t = theta_ref.copy()
        t[s["ln_sa"]] = math.log(v)
        grid.append((f"sa_{v:g}", t))
    for d in (-1.0, 1.0):
        t = theta_ref.copy()
        t[s["ln_r"]] = inp.ln_r_center + d
        grid.append((f"ln_r_{d:+g}", t))
    t = theta_ref.copy()
    t[s["dark"]] = rf.bounds(inp, rf.Priors())[s["dark"], 0]  # zero-like dark rate
    grid.append(("dark_null", t))
    return grid


def cmd_closed_loop(a: argparse.Namespace) -> int:
    inj = rf.read_inputs(a.inputs)          # held-out templates: generate the universe
    fit_inp = rf.read_inputs(a.fit_inputs)  # training templates: fit
    pr = rf.Priors()
    names = rf.parameter_names(fit_inp)
    s = rf._slices(fit_inp)
    if a.theta_ref:
        ref = np.array([json.loads(Path(a.theta_ref).read_text())["map"]["theta"][n] for n in names])
    else:
        ref, _ = rf.map_fit(fit_inp, pr, n_starts=a.map_starts, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    rows = []
    for label, th in closed_loop_grid(fit_inp, ref):
        for r in range(a.n_real):
            k = rf.simulate_counts(th, inj, rng)
            sim = rf.Rung5Inputs(**{**fit_inp.__dict__, "k_obs": k})
            th_fit, _ = rf.map_fit(sim, pr, x0=th, n_starts=1, seed=a.seed + r)
            sig = rf.laplace_sigma(sim, pr, th_fit)
            pull = (th_fit[s["dark"]] - th[s["dark"]]) / sig[s["dark"]]
            rows.append({"injection": label, "realization": r, "dark_pull": np.round(pull, 3).tolist(),
                         "nuisance_fit": {n: float(th_fit[names.index(n)]) for n in rf.NUISANCES}})
    pulls = np.array([r["dark_pull"] for r in rows], float)
    with np.errstate(invalid="ignore"):
        cover68 = np.nanmean(np.abs(pulls) < 1.0, axis=0)
        cover95 = np.nanmean(np.abs(pulls) < 1.96, axis=0)
    out = {"issue": 644, "gate": "5b", "inputs_injection": str(a.inputs), "inputs_fit": str(a.fit_inputs),
           "n_realizations_per_injection": a.n_real, "rows": rows,
           "dark_coverage_68": np.round(cover68, 3).tolist(), "dark_coverage_95": np.round(cover95, 3).tolist(),
           "limitations": list(rf.LIMITATIONS) + ["closed-loop intervals are Laplace (MAP + Hessian), not dynesty"]}
    (a.out / "rung5_closed_loop.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("dark_coverage_68", "dark_coverage_95")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("fit", "closed-loop"))
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--fit-inputs", type=Path, default=None, help="closed-loop: training-partition inputs to fit with")
    ap.add_argument("--theta-ref", type=Path, default=None, help="closed-loop: a rung5_fit.json whose MAP θ is the reference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=644)
    ap.add_argument("--map-starts", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--maxcall", type=int, default=None)
    ap.add_argument("--n-real", type=int, default=20)
    ap.add_argument("--small-n-expected", type=float, default=10.0)
    ap.add_argument("--overlap-threshold", type=float, default=0.5)
    ap.add_argument("--collapse-floor", type=float, default=1e-6)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "fit":
        return cmd_fit(a)
    if a.fit_inputs is None:
        raise SystemExit("closed-loop needs --fit-inputs (training templates)")
    return cmd_closed_loop(a)


if __name__ == "__main__":
    sys.exit(main())
