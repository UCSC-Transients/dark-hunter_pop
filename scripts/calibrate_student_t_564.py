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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True)
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--snr-edges", type=float, nargs="+", required=True)
    ap.add_argument("--nu-bounds", type=float, nargs=2, default=(1.0, 300.0))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rc = cob.load_rung4_config(args.rung4_config).mass_recovery
    gens = [ps.load_proposal_set_fragment(f).proposal for f in args.fragment]
    co = cob.check_co_generations(gens)
    parts = [ps.read_proposal_artifact(a)[:2] for a in args.artifact]
    tk = set.intersection(*(set(p[0]) for p in parts))
    ok_ = set.intersection(*(set(p[1]) for p in parts))
    t = {k: np.concatenate([p[0][k] for p in parts]) for k in tk}
    o = {k: np.concatenate([p[1][k] for p in parts]) for k in ok_}
    hold = cob.holdout_mask(t["draw_index"], co.sbc_holdout_fraction, co.sbc_holdout_salt)
    casc = np.asarray(o["cascade"], float)
    pmax = rc.gate_period_max_days if rc.gate_period_max_days is not None else np.inf
    sel = np.flatnonzero(hold & np.asarray(o["accepted_orbital"], bool) & (casc[:, 10] <= pmax)
                         & np.all(np.isfinite(o["orbit_cov"].reshape(casc.shape[0], -1)), axis=1))
    names = list(ps.ORBIT_PARAMETER_NAMES)
    iA = [names.index(k) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")]
    pulls = np.full(sel.size, np.nan)
    for j, i in enumerate(sel):
        p0, c = o["orbit_p0"][i], o["orbit_cov"][i]
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
    snr = casc[sel, 17] / casc[sel, 18]
    e = np.asarray(args.snr_edges, float)
    res = {"snr_edges": e.tolist(), "nu": [], "n": [], "partition": "holdout", "n_total": int(sel.size)}
    lo_b, hi_b = (math.log(b) for b in args.nu_bounds)
    for a, b in zip(e[:-1], e[1:]):
        x = pulls[(snr >= a) & (snr < b) & np.isfinite(pulls)]
        r = optimize.minimize_scalar(lambda ln: -np.sum(stats.t.logpdf(x, df=math.exp(ln))), bounds=(lo_b, hi_b), method="bounded")
        res["nu"].append(round(float(math.exp(r.x)), 2))
        res["n"].append(int(x.size))
    print(json.dumps(res))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
