#!/usr/bin/env python3
"""#590 (part 3): same-orbit F2 under colour_v2.1+tr1 (#390 orbits, reinject_v21tr1, 2 accepted realizations).

Per G bin: F2 medians, the mock realization scatter σ_mock = MAD(F2_0 − F2_1)/√2, the DR3 per-orbit scatter about the mock
mean σ_DR3 = √(MAD(F2_d − F̄2_m)² − σ_mock²/2), P(F2 > 12) and the 95th percentile, and the per-source noise-scale
dispersion that the extra F2 scatter implies, σ(log10 s) ≈ √(σ_DR3² − σ_mock²) / (√(9ν/2)·(2/3)·ln 10) with ν = N_obs − 12
(F2 ≈ √(9ν/2)·((χ²/ν)^{1/3} − 1), χ²/ν ∝ s²). Writes ``f2_sameorbit.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")


def main(argv=None) -> int:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reinject", type=Path, default=D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        sid, f2d, g = f["published/source_id"][()], f["published/goodness_of_fit"][()], f["published/g_mag"][()]
    ix = {int(s): i for i, s in enumerate(sid)}
    by: dict = {}
    for line in a.reinject.read_text().splitlines():
        r = json.loads(line)
        if r["accepted"]:
            by.setdefault(r["source_id"], {})[r["realization"]] = (r["goodness_of_fit"], r["n_obs"])
    pr = [(ix[s], v[0], v[1]) for s, v in by.items() if 0 in v and 1 in v]
    i = np.array([p[0] for p in pr])
    m0, m1 = np.array([p[1][0] for p in pr]), np.array([p[2][0] for p in pr])
    nu = 0.5 * (np.array([p[1][1] for p in pr]) + np.array([p[2][1] for p in pr])) - 12
    fd, gg = f2d[i], g[i]
    mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))  # noqa: E731
    res = {}
    for lo, hi in ((9, 11), (11, 12), (12, 13), (13, 15), (15, 19)):
        s = (gg >= lo) & (gg < hi)
        sm = mad((m0 - m1)[s]) / np.sqrt(2)
        sd = float(np.sqrt(max(mad((fd - (m0 + m1) / 2)[s]) ** 2 - sm**2 / 2, 0)))
        ex = float(np.sqrt(max(sd**2 - sm**2, 0)))
        k = float(np.sqrt(9 * np.median(nu[s]) / 2) * (2 / 3) * np.log(10))
        res[f"G{lo}-{hi}"] = {"n": int(s.sum()), "f2_median_dr3": float(np.median(fd[s])), "f2_median_mock": float(np.median(np.r_[m0[s], m1[s]])),
                              "sigma_mock": sm, "sigma_dr3": sd, "sigma_extra": ex, "sigma_log10_s_implied": ex / k,
                              "p_f2_gt12_dr3": float(np.mean(fd[s] > 12)), "p_f2_gt12_mock": float(np.mean(np.r_[m0[s], m1[s]] > 12)),
                              "p95_dr3": float(np.percentile(fd[s], 95)), "p95_mock": float(np.percentile(np.r_[m0[s], m1[s]], 95))}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "f2_sameorbit.json").write_text(json.dumps(res, indent=1))
    for k_, v in res.items():
        print(k_, {x: round(y, 3) for x, y in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
