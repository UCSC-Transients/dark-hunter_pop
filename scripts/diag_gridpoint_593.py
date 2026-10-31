#!/usr/bin/env python3
"""#593: subtract the GOST grid-point (scan-pattern) spread from the #589/#590 same-orbit extra scatter.

Per orbit of the control (``reinject_gridpoints_593.py``): group 0 = its own grid point (the two reinject_v21tr1
realizations), groups 1–3 = the three nearest neighbouring healpix-16 grid points (two realizations each). For each
quantity q ∈ {log σ_a0, log σ_ϖ, ½ log(RUWE² − 1), log z, F2}:

* within-group (noise) scatter σ_w = MAD of the realization pair differences / √2;
* grid-point scatter σ_grid² = Var_robust(group mean − orbit mean) · K/(K − 1) − σ_w²/2 (K = groups with both
  realizations accepted), pooled over orbits per G range;
* the #589 / #590 same-orbit extra scatter σ_extra (DR3 about the mock mean beyond mock noise), recomputed on this
  subset and taken from the full-sample files, minus the grid spread in quadrature: σ_left = √(σ_extra² − f σ_grid²)
  with f = 1 (neighbour spacing, an upper bound on the true-vs-nearest-pattern mismatch) and f = ½.

Writes ``gridpoint.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V21 = D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl"
MAD = lambda x: float(1.4826 * np.nanmedian(np.abs(np.asarray(x) - np.nanmedian(x)))) if np.size(x) else float("nan")  # noqa: E731
Q = ("log_sigma_a0", "log_sigma_plx", "log_sqrt_ruwe2m1", "log_z", "f2")
G_RANGES = {"G<13": (0, 13), "G>=13": (13, 30), "G9-11": (0, 11), "G11-12": (11, 12), "G12-13": (12, 13)}


def qvec(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log10(r["sigma_a0_mas"])
        return {"log_sigma_a0": s, "log_sigma_plx": np.log10(r["parallax_error"]), "log_sqrt_ruwe2m1": 0.5 * np.log10(r["ruwe"] ** 2 - 1),
                "log_z": np.log10(r["a0_mas"]) - s - 0.5 * np.log10(r["ruwe"] ** 2 - 1), "f2": r["goodness_of_fit"]}


def main(argv=None) -> int:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--dispersion", type=Path, default=Path("docs/gate391/diag_v21tr1/edge/dispersion.json"))
    ap.add_argument("--f2-sameorbit", type=Path, default=Path("docs/gate391/diag_v21tr1/f2tail/f2_sameorbit.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    groups: dict = {}
    for path, gfun in ((V21, lambda r: 0), (a.control / "inject.jsonl", lambda r: int(r["realization"]) // 10)):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("accepted"):
                groups.setdefault(int(r["source_id"]), {}).setdefault(gfun(r), []).append(qvec(r))
    ctrl = {int(json.loads(x)["source_id"]) for x in (a.control / "inject.jsonl").read_text().splitlines()}
    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        sid = P["source_id"][()]
        pub = {k: P[k][()] for k in ("g_mag", "significance", "ruwe", "parallax_error", "goodness_of_fit", "a_thiele_innes", "b_thiele_innes",
                                     "f_thiele_innes", "g_thiele_innes")}
    ix = {int(s): i for i, s in enumerate(sid)}
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0d = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))
    with np.errstate(divide="ignore", invalid="ignore"):
        dr3 = {"log_sigma_a0": np.log10(a0d / pub["significance"]), "log_sigma_plx": np.log10(pub["parallax_error"]),
               "log_sqrt_ruwe2m1": 0.5 * np.log10(pub["ruwe"] ** 2 - 1), "f2": pub["goodness_of_fit"]}
        dr3["log_z"] = np.log10(a0d) - dr3["log_sigma_a0"] - dr3["log_sqrt_ruwe2m1"]
    rows = {q: {"g": [], "dev": [], "pair": [], "dr3_minus_own": [], "own_pair": []} for q in Q}
    n_groups = []
    for s in ctrl:
        gr = groups.get(s, {})
        full = {k: v_ for k, v_ in gr.items() if len(v_) >= 2}
        if 0 not in full or len(full) < 3:
            continue
        n_groups.append(len(full))
        g = pub["g_mag"][ix[s]]
        for q in Q:
            means = np.array([0.5 * (full[k][0][q] + full[k][1][q]) for k in sorted(full)])
            if not np.all(np.isfinite(means)):
                continue
            K = means.size
            rows[q]["g"] += [g] * K
            rows[q]["dev"] += list((means - means.mean()) * np.sqrt(K / (K - 1)))
            rows[q]["pair"] += [full[k][0][q] - full[k][1][q] for k in sorted(full)]
            rows[q]["dr3_minus_own"].append(dr3[q][ix[s]] - means[0])
            rows[q]["own_pair"].append((g, full[0][0][q] - full[0][1][q]))
    disp = json.loads(a.dispersion.read_text())["scatter"]
    f2so = json.loads(a.f2_sameorbit.read_text())
    keymap = {"log_sigma_a0": "log_sigma_a0", "log_sigma_plx": "log_sigma_plx", "log_sqrt_ruwe2m1": "log_sqrt_ruwe2m1", "log_z": "z"}
    res = {"n_orbits_used": len(n_groups), "groups_per_orbit_median": float(np.median(n_groups)) if n_groups else 0.0, "by": {}}
    for q in Q:
        g_ = np.array(rows[q]["g"])
        dev, pair = np.array(rows[q]["dev"]), np.array(rows[q]["pair"])
        og = np.array([x[0] for x in rows[q]["own_pair"]])
        dmo = np.array(rows[q]["dr3_minus_own"])
        opair = np.array([x[1] for x in rows[q]["own_pair"]])
        res["by"][q] = {}
        for gname, (lo, hi) in G_RANGES.items():
            if q != "f2" and gname not in ("G<13", "G>=13"):
                continue
            b, bo = (g_ >= lo) & (g_ < hi), (og >= lo) & (og < hi)
            if b.sum() < 20:
                continue
            sw = MAD(pair[b]) / np.sqrt(2)
            sgrid = float(np.sqrt(max(MAD(dev[b]) ** 2 - sw**2 / 2, 0)))
            # subset same-orbit extra: DR3 about the own-grid-point mean, beyond the mock noise
            sw0 = MAD(opair[bo]) / np.sqrt(2)
            sd = float(np.sqrt(max(MAD(dmo[bo]) ** 2 - sw0**2 / 2, 0)))
            ex_sub = float(np.sqrt(max(sd**2 - sw0**2, 0)))
            if q == "f2":
                fk = {"G<13": None, "G>=13": None, "G9-11": "G9-11", "G11-12": "G11-12", "G12-13": "G12-13"}[gname]
                ex_full = f2so[fk]["sigma_extra"] if fk else float("nan")
            else:
                ex_full = disp[gname][keymap[q]]["sigma_extra"]
            left = lambda ex, f: float(np.sqrt(max(ex**2 - f * sgrid**2, 0))) if np.isfinite(ex) else float("nan")  # noqa: E731
            res["by"][q][gname] = {"n_orbits": int(bo.sum()), "sigma_within": sw, "sigma_grid": sgrid, "sigma_extra_full_sample": ex_full,
                                   "sigma_extra_subset": ex_sub, "left_full_minus_grid": left(ex_full, 1.0), "left_full_minus_half_grid": left(ex_full, 0.5),
                                   "left_subset_minus_grid": left(ex_sub, 1.0)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "gridpoint.json").write_text(json.dumps(res, indent=1))
    print(f"orbits used {res['n_orbits_used']}, groups/orbit {res['groups_per_orbit_median']}")
    for q, t in res["by"].items():
        for gname, x in t.items():
            print(f"{q:18s} {gname:7s} n{x['n_orbits']:4d} within {x['sigma_within']:.3f} grid {x['sigma_grid']:.3f} | extra full {x['sigma_extra_full_sample']:.3f} "
                  f"sub {x['sigma_extra_subset']:.3f} | left full-grid {x['left_full_minus_grid']:.3f} (half {x['left_full_minus_half_grid']:.3f}) sub-grid {x['left_subset_minus_grid']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
