#!/usr/bin/env python3
"""#595 headline check: bright P(F2 > 12) by dwarf / evolved class, gens 47–50 with ss1 applied post hoc, vs DR3.

Every gens 47–50 draw whose cascade reached the orbital solution (accepted or failing the cuts) gets one s = 10^(σ_s(G) z)
per replicate (σ_s from the configured ``dr3.epoch_model.source_noise_scale`` knots, 0 at G ≥ 13), and its fit is
transformed as in ``calibrate_source_scale_595.py``: χ²/ν → s² χ²/ν (F2 recomputed, ν = N_obs − 12), σ_ϖ, σ_a0, σ_e → s σ,
RUWE² → RUWE² + s² − 1. The orbital-solution cuts are then re-applied (a0/σ_a0, F2, ϖ/σ_ϖ × P, a0/σ_a0 × √P, σ_e(P)), so
draws can leave or join the accepted set. Routing (5/7/9-parameter vs orbit) is not recomputed (approximation; exact
only with re-simulation). Published accelerations: F2 only, ν = 390.

C1 orbits (P ≤ 830.4 d), G 10–11 / 11–12 / 12–13 and dwarf / evolved on the noise-free system CMD; unweighted draws
(the high-ESS view of a conditional noise property) and default-target weights. DR3: the rung-3 C1 orbits and
accelerations (#587 dump). Writes ``population_ss1.json``.
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

G_BINS = ((10.0, 11.0), (11.0, 12.0), (12.0, 13.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-rep", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    ssc = cfg.dr3.epoch_model.source_noise_scale
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    ridge, ns = dc.ridge_and_nsigma()
    M, R = dc.load_mock(), dc.load_real()
    cas = M["cascade"]
    g = M["phot_g_mean_mag"]
    st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in M["solution_type"]])
    orb = np.isin(st, ["twelve_parameter_orbital", "orbital_failed_cuts"]) & (cas[:, 10] > 0) & (cas[:, 10] <= dc.C1_PMAX) & (g >= 10) & (g < 13)
    acc0 = M["orbit"] & orb
    cls = dc.classes(M["sys_mg0"], M["sys_colour0"], ridge, ns)
    evolved = np.isin(cls, ["giant", "subgiant"])
    dwarf = cls == "dwarf"
    sig = np.where(g < ssc.break_g, np.interp(g, ssc.sigma_knots_g, ssc.sigma_log10), 0.0)
    rng = np.random.default_rng(595)
    P, f2, nu = cas[:, 10], cas[:, 21], np.clip(cas[:, 20] - 12, 20, None)
    rr = R["orbits"]
    rcls = dc.classes(rr["mg0"], rr["colour0"], ridge, ns)
    ra = R["accelerations"]
    racls = dc.classes(ra["mg0"], ra["colour0"], ridge, ns)
    res = {"sigma_knots": [list(ssc.sigma_knots_g), list(ssc.sigma_log10)], "orbits": {}, "accelerations": {}}
    reps_acc, reps_f2, reps_af2 = [], [], []
    for _ in range(a.n_rep):
        s = 10 ** (sig * rng.standard_normal(g.size))
        f2p = f2_from_chi2red(np.clip(s * s * chi2red_from_f2(f2, nu), 1e-6, None), nu)
        with np.errstate(invalid="ignore", divide="ignore"):
            a0s = cas[:, 17] / (cas[:, 18] * s)
            plxs = cas[:, 0] / (cas[:, 1] * s)
            ok = ((a0s > cuts.a0_over_err_min) & (f2p < cuts.goodness_of_fit_f2_max) & (plxs > cuts.parallax_over_error_times_period_min_days / P)
                  & (a0s > cuts.a0_over_err_times_sqrt_period_min / np.sqrt(P)) & (cas[:, 15] * s < cuts.sigma_e_ln_period_slope * np.log(P) + cuts.sigma_e_intercept))
        reps_acc.append(orb & ok)
        reps_f2.append(f2p)
        nine = np.isclose(cas[:, 0], -9.0)
        af2 = np.where(nine, cas[:, 13], cas[:, 9])
        reps_af2.append(f2_from_chi2red(np.clip(s * s * chi2red_from_f2(af2, 390.0), 1e-6, None), 390.0))
    af2_0 = np.where(np.isclose(cas[:, 0], -9.0), cas[:, 13], cas[:, 9])
    for lo, hi in G_BINS:
        for cname, cm, rc in (("dwarf", dwarf, rcls == "dwarf"), ("evolved", evolved, np.isin(rcls, ["giant", "subgiant"]))):
            key = f"G{lo:g}-{hi:g} {cname}"
            b = (g >= lo) & (g < hi) & cm
            br = (rr["g"] >= lo) & (rr["g"] < hi) & rc
            row = {"n_dr3": int(br.sum()), "p_dr3": float(np.mean(rr["f2"][br] > 12)) if br.sum() else float("nan"),
                   "n_mock_before": int((acc0 & b).sum())}
            for wname, w in (("unweighted", np.ones(g.size)), ("default", M["w"])):
                pb = float(w[acc0 & b & (f2 > 12)].sum() / w[acc0 & b].sum()) if w[acc0 & b].sum() > 0 else float("nan")
                pa = [float(w[ac & b & (fp > 12)].sum() / w[ac & b].sum()) if w[ac & b].sum() > 0 else float("nan") for ac, fp in zip(reps_acc, reps_f2)]
                na = [float(w[ac & b].sum() / w[acc0 & b].sum()) if w[acc0 & b].sum() > 0 else float("nan") for ac in reps_acc]
                row[wname] = {"p_before": pb, "p_after": float(np.mean(pa)), "accepted_after_over_before": float(np.mean(na)),
                              "ess_before": dc.kish(w[acc0 & b])}
            res["orbits"][key] = row
            # accelerations
            ba = M["accel"] & (g >= lo) & (g < hi) & cm
            bra = (ra["g"] >= lo) & (ra["g"] < hi) & (racls == "dwarf" if cname == "dwarf" else np.isin(racls, ["giant", "subgiant"]))
            res["accelerations"][key] = {"n_dr3": int(bra.sum()), "p_dr3": float(np.mean(ra["f2"][bra] > 12)) if bra.sum() else float("nan"),
                                         "n_mock": int(ba.sum()), "p_before_unweighted": float(np.mean(af2_0[ba] > 12)) if ba.sum() else float("nan"),
                                         "p_after_unweighted": float(np.mean([np.mean(x[ba] > 12) for x in reps_af2])) if ba.sum() else float("nan"),
                                         "median_f2_dr3": float(np.median(ra["f2"][bra])) if bra.sum() else float("nan"),
                                         "median_f2_before": float(np.median(af2_0[ba])) if ba.sum() else float("nan"),
                                         "median_f2_after": float(np.median(np.concatenate([x[ba] for x in reps_af2]))) if ba.sum() else float("nan")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for k, v in res["orbits"].items():
        u = v["unweighted"]
        print(f"orbits {k:18s} DR3 {v['p_dr3']:.3f} (n{v['n_dr3']}) | mock before {u['p_before']:.3f} after {u['p_after']:.3f} (n{v['n_mock_before']}, acc x{u['accepted_after_over_before']:.2f}) | default {v['default']['p_before']:.3f} -> {v['default']['p_after']:.3f}")
    for k, v in res["accelerations"].items():
        print(f"accel  {k:18s} DR3 {v['p_dr3']:.3f} | mock {v['p_before_unweighted']:.3f} -> {v['p_after_unweighted']:.3f} | median F2 DR3 {v['median_f2_dr3']:.1f} mock {v['median_f2_before']:.1f} -> {v['median_f2_after']:.1f} (n{v['n_mock']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
