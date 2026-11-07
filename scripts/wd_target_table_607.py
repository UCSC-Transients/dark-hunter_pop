#!/usr/bin/env python3
"""#607 (spec §12.43): diagnose the WD importance weights of co_v1 gen 1006 at the rung-3 v4 fit and tabulate the
§12.39 physical-target marginal in log10 M2 (and log10 P) of the detected WD draws, for a target-shaped proposal component.

The table is the weighted histogram of true log10 M2 (log10 P) over WD draws that enter the fit (C1 orbit or published
acceleration), with weights = co_v1 deterministic-mixture weight × λ_§12.39(θ_v4) / λ_default. Bins with no weight get a
floor (``--floor`` × the mean density) so the component has full support. Reports where the weight concentrates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import joint_refit as jr
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--co-artifact", type=Path, required=True)
    ap.add_argument("--co-fragment", type=Path, required=True)
    ap.add_argument("--fit-config", type=Path, required=True)
    ap.add_argument("--fit-summary", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--m2-edges", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.05, 1.2, 1.4])
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load_config()
    frag = ps.load_proposal_set_fragment(a.co_fragment)
    parent = ps.load_parent_snapshot(a.parent_dir, cfg, frag.proposal)
    t, o, _ = ps.read_proposal_artifact(a.co_artifact)
    fc = yaml.safe_load(a.fit_config.read_text())["rung3"]
    w = cob.co_importance_weights(t, parent, [frag.proposal], frag.target_co, partition="all")
    obs = jr.co_observables(t, o, float(fc["c1_period_max_days"]))
    wdp = jr.WDPhysicsConfig(**fc["joint"]["wd_physics"])
    r = json.loads(a.fit_summary.read_text())["results"]["joint"]
    th = dict(zip(r["names"], r["theta"]))
    comp = np.asarray(t["co_component"])
    use = (comp == cob.COMPONENT_CODES["wd"]) & (obs["in_orbit"] | obs["in_accel"])
    m2, p, e, m1 = (np.asarray(t[k], float)[use] for k in ("m2_msun", "period_days", "eccentricity", "m1_msun"))
    tsys = 10.0 ** np.asarray(t["iso_log_age"], float)[use] / 1e9
    lnew = jr.wd_physics_log_density(m2, p, e, m1, tsys, ln_a_ni=th["ln_R_wd"], ln_r_mt=th["ln_R_mt"], f_he=th["f_he"], cfg=wdp)
    lold = cob.co_shape_log_density(t, frag.target_co)[use]
    with np.errstate(invalid="ignore", over="ignore"):
        rr = np.exp(lnew - lold)
    ww = w[use] * np.where(np.isfinite(rr), rr, 0.0)
    orb = obs["in_orbit"][use]
    rep = {"n_wd_fit_draws": int(use.sum()), "ess_all": float(ps.kish_ess(ww)), "ess_orbits": float(ps.kish_ess(ww[orb])),
           "top10_weight_share": float(np.sort(ww)[::-1][:10].sum() / ww.sum())}
    # where does the weight concentrate? (share of weight vs share of draws)
    def split(x, edges, name):
        h_w = np.histogram(x, edges, weights=ww)[0] / ww.sum()
        h_n = np.histogram(x, edges)[0] / x.size
        rep[name] = {"edges": list(map(float, edges)), "weight_share": h_w.tolist(), "draw_share": h_n.tolist()}
    split(m2, np.asarray(a.m2_edges), "m2")
    split(np.log10(p), np.array([0.2, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8]), "log_p")
    split(m1, np.array([0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]), "m1")
    split(np.log10(tsys), np.array([-1.0, 0.0, 0.5, 0.8, 1.0, 1.2]), "log_tsys")
    # tabulated proposal density in log10 M2 (per dex), weight-proportional with a floor
    le = np.log10(np.asarray(a.m2_edges))
    h = np.histogram(np.log10(m2), le, weights=ww)[0]
    dens = h / np.diff(le)
    dens = dens / np.sum(dens * np.diff(le))
    dens = np.maximum(dens, a.floor * 1.0 / (le[-1] - le[0]))
    dens = dens / np.sum(dens * np.diff(le))
    rep["table"] = {"log10_m2_edges": le.round(6).tolist(), "density_per_dex": np.round(dens, 6).tolist()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("table",)}, indent=1))
    print("table", rep["table"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
