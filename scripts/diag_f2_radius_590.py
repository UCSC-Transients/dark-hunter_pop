#!/usr/bin/env python3
"""#590 (part 2): does the giant F2 excess scale with the angular radius θ* = R*/d (convective photocentre jitter)?

R* from M_G0 and T_eff (``giants.cmd_radius_rsun``; T_eff from (BP−RP)_0 by a quadratic fitted on the mock's own
primary T_eff vs system colour for single-like dwarfs + giants), θ* in µas. P(F2 > cut) vs log θ* for evolved and dwarf
orbits and accelerations at G < 13, DR3 vs mock (unweighted draws and default-target weights). Writes ``f2_radius.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_common_v21tr1 as dc  # noqa: E402
from darkhunter_pop import giants  # noqa: E402

TH_EDGES = np.array([-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0])  # log10 θ* (µas)


def theta_uas(mg0, teff, d_kpc):
    r = giants.cmd_radius_rsun(mg0, teff)
    return np.log10(r * 6.957e8 / 1.496e11 / d_kpc * 1e3)  # R in AU / d in kpc = mas; ×1e3 → µas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--f2-cut", type=float, default=12.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    ridge, ns = dc.ridge_and_nsigma()
    M = dc.load_mock()
    R = dc.load_real()
    ok = np.isfinite(M["primary_log_teff"]) & np.isfinite(M["sys_colour0"]) & (M["sys_colour0"] > 0.3) & (M["sys_colour0"] < 1.8) & (M["flux_ratio"] < 0.05)
    cf = np.polyfit(M["sys_colour0"][ok], M["primary_log_teff"][ok], 2)
    res = {"teff_poly_log10": cf.tolist(), "f2_cut": a.f2_cut}
    mcls = dc.classes(M["sys_mg0"], M["sys_colour0"], ridge, ns)
    for kind, sel, pre in (("orbits", M["orbit"], "o"), ("accelerations", M["accel"], "a")):
        r = R[kind]
        rcls = dc.classes(r["mg0"], r["colour0"], ridge, ns)
        th_r = theta_uas(r["mg0"], 10 ** np.polyval(cf, r["colour0"]), r["d"])
        th_m = theta_uas(M["sys_mg0"][sel], 10 ** M["primary_log_teff"][sel], M[f"{pre}_d"][sel])
        f2m, gm, cm, wm = M[f"{pre}_f2"][sel], M["phot_g_mean_mag"][sel], mcls[sel], M["w"][sel]
        res[kind] = {}
        for c in ("dwarf", "evolved"):
            isr = (rcls == "dwarf") if c == "dwarf" else np.isin(rcls, ["giant", "subgiant"])
            ism = (cm == "dwarf") if c == "dwarf" else np.isin(cm, ["giant", "subgiant"])
            rows = []
            for lo, hi in zip(TH_EDGES[:-1], TH_EDGES[1:]):
                br = isr & (r["g"] < 13) & (th_r >= lo) & (th_r < hi)
                bm = ism & (gm < 13) & (th_m >= lo) & (th_m < hi)
                rows.append({"log_theta_uas": [float(lo), float(hi)], "n_dr3": int(br.sum()), "n_mock": int(bm.sum()),
                             "p_dr3": float(np.mean(r["f2"][br] > a.f2_cut)) if br.sum() >= 20 else float("nan"),
                             "p_mock_unweighted": float(np.mean(f2m[bm] > a.f2_cut)) if bm.sum() >= 20 else float("nan"),
                             "p_mock_default": float(wm[bm & (f2m > a.f2_cut)].sum() / wm[bm].sum()) if bm.sum() >= 20 and wm[bm].sum() > 0 else float("nan"),
                             "f2_median_dr3": float(np.median(r["f2"][br])) if br.sum() >= 20 else float("nan"),
                             "f2_median_mock": float(np.median(f2m[bm])) if bm.sum() >= 20 else float("nan")})
            res[kind][c] = rows
            # the same at fixed apparent magnitude: G bins (separates a brightness trend from an angular-size one)
            gq = []
            for lo, hi in ((8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 16)):
                br, bm = isr & (r["g"] >= lo) & (r["g"] < hi), ism & (gm >= lo) & (gm < hi)
                row = {"g": [lo, hi], "n_dr3": int(br.sum()), "n_mock": int(bm.sum()),
                       "p_dr3": float(np.mean(r["f2"][br] > a.f2_cut)) if br.sum() >= 20 else float("nan"),
                       "p_mock_unweighted": float(np.mean(f2m[bm] > a.f2_cut)) if bm.sum() >= 20 else float("nan"),
                       "f2_median_dr3": float(np.median(r["f2"][br])) if br.sum() >= 20 else float("nan"),
                       "f2_median_mock": float(np.median(f2m[bm])) if bm.sum() >= 20 else float("nan"),
                       "f2_p99_dr3": float(np.percentile(r["f2"][br], 99)) if br.sum() >= 100 else float("nan"),
                       "f2_p99_mock": float(np.percentile(f2m[bm], 99)) if bm.sum() >= 100 else float("nan")}
                # within the G bin: does θ* still matter? (DR3 P above/below the bin's median θ*)
                if br.sum() >= 100:
                    med = np.nanmedian(th_r[br])
                    row["p_dr3_small_theta"] = float(np.mean(r["f2"][br & (th_r < med)] > a.f2_cut))
                    row["p_dr3_large_theta"] = float(np.mean(r["f2"][br & (th_r >= med)] > a.f2_cut))
                gq.append(row)
            res[kind][c + "_by_G"] = gq
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "f2_radius.json").write_text(json.dumps(res, indent=1))
    for kind in ("orbits", "accelerations"):
        for c in ("dwarf_by_G", "evolved_by_G"):
            print(kind, c, " | ".join(f"G{x['g'][0]}: p {x['p_dr3']:.3f}/{x['p_mock_unweighted']:.3f} med {x['f2_median_dr3']:.1f}/{x['f2_median_mock']:.1f} p99 {x['f2_p99_dr3']:.1f}/{x['f2_p99_mock']:.1f} θs/l {x.get('p_dr3_small_theta', float('nan')):.3f}/{x.get('p_dr3_large_theta', float('nan')):.3f}" for x in res[kind][c]))
        for c in ("dwarf", "evolved"):
            rows = res[kind][c]
            print(kind, c, " | ".join(f"{x['log_theta_uas'][0]:g}: n{x['n_dr3']}/{x['n_mock']} p {x['p_dr3']:.3f}/{x['p_mock_unweighted']:.3f} med {x['f2_median_dr3']:.1f}/{x['f2_median_mock']:.1f}" for x in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
