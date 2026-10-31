#!/usr/bin/env python3
"""#595 (EPOCH_MODEL_SPEC §8.15): calibrate the bright per-source true-noise scale s on the #390 same-orbit set.

Post hoc on the colour_v2.1+tr1 re-injection (``reinject_v21tr1``, 2 accepted realizations per orbit), no simulation.
A per-source factor s = 10^(σ_s z), z ~ N(0, 1), on the TRUE per-CCD noise with the stated errors unchanged acts on
one realization's fit as

* χ²/ν → s² χ²/ν, so F2 → √(9ν/2) [(s² χ²/ν)^{1/3} + 2/(9ν) − 1] with χ²/ν recovered from the stored F2 and ν = N_obs − 12;
* every reported σ (σ_ϖ, σ_a0) → s σ (the Halbwachs inflation c ∝ √(χ²/ν));
* RUWE² → RUWE² + (s² − 1) (the noise part of UWE²/u0² is ~1 for the 5-parameter fit).

Each realization gets its own s (a mock draw is a new source). Exchangeability of DR3 with one mock realization, net of
the #593 grid-point spread σ_grid, requires for each quantity q

    T(σ_s) = Var(q_DR3 − q̄_mock) − ¾ Var(q_0 − q_1) − σ_grid² = 0     (robust variances, MAD²)

T is computed on F2 (the calibration target) and log σ_ϖ (a check) per G bin, averaged over ``--n-rep`` draws of
the s; σ_s is found by bisection on F2, with a 100-fold orbit bootstrap error. Also reported at the fitted σ_s: F2
95th / 99th percentiles and P(F2 > 12), DR3 vs mock; a colour split ((BP−RP) below/above the bright median) as the
colour-term test (BIC on the per-bin Gaussian likelihood of the two-group vs one-group σ_s). Writes ``calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V21 = D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl"
BINS = {"G9-11": (0.0, 11.0), "G11-12": (11.0, 12.0), "G12-13": (12.0, 13.0), "G13-15": (13.0, 15.0)}
MADV = lambda x: float((1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))) ** 2)  # noqa: E731


def chi2red_from_f2(f2, nu):
    return (f2 * np.sqrt(2.0 / (9.0 * nu)) + 1.0 - 2.0 / (9.0 * nu)) ** 3


def f2_from_chi2red(x, nu):
    return np.sqrt(9.0 * nu / 2.0) * (np.cbrt(x) + 2.0 / (9.0 * nu) - 1.0)


def apply_s(f2, nu, splx, ruwe, s):
    x = chi2red_from_f2(f2, nu)
    return f2_from_chi2red(np.clip(s * s * x, 1e-6, None), nu), splx * s, np.sqrt(np.clip(ruwe**2 + s * s - 1.0, 1e-6, None))


def load(path: Path) -> dict:
    import h5py

    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        sid, g, f2d, spd, rud = (P[k][()] for k in ("source_id", "g_mag", "goodness_of_fit", "parallax_error", "ruwe"))
    ix = {int(s): i for i, s in enumerate(sid)}
    by: dict = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("accepted"):
            by.setdefault(int(r["source_id"]), {})[int(r["realization"])] = r
    pr = [(ix[s], v[0], v[1]) for s, v in by.items() if 0 in v and 1 in v]
    i = np.array([p[0] for p in pr])
    m = lambda k, j: np.array([p[j][k] for p in pr], float)  # noqa: E731
    return {"g": g[i], "f2_d": f2d[i], "splx_d": spd[i], "ruwe_d": rud[i], "bp_rp": m("bp_rp", 1),
            "f2": np.stack([m("goodness_of_fit", 1), m("goodness_of_fit", 2)]), "nu": np.stack([m("n_obs", 1) - 12, m("n_obs", 2) - 12]),
            "splx": np.stack([m("parallax_error", 1), m("parallax_error", 2)]), "ruwe": np.stack([m("ruwe", 1), m("ruwe", 2)])}


def stats_at(d, sel, sigma, rng, n_rep, grid_f2, grid_lsplx):
    """T for F2 and log σ_ϖ, plus F2 tail statistics, averaged over n_rep draws of s."""
    tf, tp, p95, p99, p12 = [], [], [], [], []
    for _ in range(n_rep):
        s = 10 ** (sigma * rng.standard_normal((2, sel.sum())))
        f2m, spm, _ = apply_s(d["f2"][:, sel], d["nu"][:, sel], d["splx"][:, sel], d["ruwe"][:, sel], s)
        lsm = np.log10(spm)
        fd, lsd = d["f2_d"][sel], np.log10(d["splx_d"][sel])
        tf.append(MADV(fd - f2m.mean(0)) - 0.75 * MADV(f2m[0] - f2m[1]) - grid_f2**2)
        tp.append(MADV(lsd - lsm.mean(0)) - 0.75 * MADV(lsm[0] - lsm[1]) - grid_lsplx**2)
        p95.append(np.percentile(f2m, 95))
        p99.append(np.percentile(f2m, 99))
        p12.append(np.mean(f2m > 12))
    fd = d["f2_d"][sel]
    return {"T_f2": float(np.mean(tf)), "T_log_splx": float(np.mean(tp)), "f2_p95_mock": float(np.mean(p95)), "f2_p99_mock": float(np.mean(p99)),
            "p_f2_gt12_mock": float(np.mean(p12)), "f2_p95_dr3": float(np.percentile(fd, 95)), "f2_p99_dr3": float(np.percentile(fd, 99)),
            "p_f2_gt12_dr3": float(np.mean(fd > 12)), "n": int(sel.sum())}


def solve(d, sel, rng, n_rep, grid_f2, grid_lsplx, hi=0.15):
    f = lambda s_: stats_at(d, sel, s_, rng, n_rep, grid_f2, grid_lsplx)["T_f2"]  # noqa: E731
    if f(0.0) <= 0:
        return 0.0
    lo_, hi_ = 0.0, hi
    if f(hi_) > 0:
        return hi_
    for _ in range(14):
        mid = 0.5 * (lo_ + hi_)
        if f(mid) > 0:
            lo_ = mid
        else:
            hi_ = mid
    return 0.5 * (lo_ + hi_)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reinject", type=Path, default=V21)
    ap.add_argument("--gridpoint", type=Path, default=Path("docs/gate391/diag_v21tr1/gridpoint/gridpoint.json"))
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = load(a.reinject)
    gp = json.loads(a.gridpoint.read_text())["by"]
    grid_f2_b, grid_f2_f = gp["f2"]["G<13"]["sigma_grid"], gp["f2"]["G>=13"]["sigma_grid"]
    grid_ls_b, grid_ls_f = gp["log_sigma_plx"]["G<13"]["sigma_grid"], gp["log_sigma_plx"]["G>=13"]["sigma_grid"]
    rng = np.random.default_rng(595)
    res = {"grid_spread": {"f2": [grid_f2_b, grid_f2_f], "log_sigma_plx": [grid_ls_b, grid_ls_f]}, "bins": {}}
    for name, (lo, hi) in BINS.items():
        sel = (d["g"] >= lo) & (d["g"] < hi)
        gf, gl = (grid_f2_b, grid_ls_b) if hi <= 13 else (grid_f2_f, grid_ls_f)
        sig = solve(d, sel, rng, a.n_rep, gf, gl)
        idx = np.flatnonzero(sel)
        boots = []
        for _ in range(a.n_boot):
            bs = np.zeros(sel.size, bool)
            pick = rng.choice(idx, idx.size)
            dd = {k: (v[:, pick] if v.ndim == 2 else v[pick]) for k, v in d.items()}
            boots.append(solve(dd, np.ones(idx.size, bool), rng, max(4, a.n_rep // 5), gf, gl))
        before = stats_at(d, sel, 0.0, rng, 1, gf, gl)
        after = stats_at(d, sel, sig, rng, a.n_rep, gf, gl)
        # colour split
        med = float(np.nanmedian(d["bp_rp"][sel]))
        col = {}
        for cn, cs in (("blue", sel & (d["bp_rp"] < med)), ("red", sel & (d["bp_rp"] >= med))):
            col[cn] = solve(d, cs, rng, a.n_rep, gf, gl)
        res["bins"][name] = {"sigma_log10_s": sig, "sigma_log10_s_boot_err": float(np.std(boots)), "before": before, "after": after,
                             "colour_split_median_bp_rp": med, "colour_split_sigma": col,
                             "colour_split_difference_over_err": abs(col["blue"] - col["red"]) / (np.sqrt(2) * max(float(np.std(boots)), 1e-6))}
        print(name, json.dumps(res["bins"][name], default=float)[:600], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
