#!/usr/bin/env python3
"""#597: refit the bright stated-error scale e(G) on DR3 Orbital-only F2, then measure the AstroSpectroSB1 offsets.

Same orbits: the #390 published orbits re-injected under colour_v2.1+tr1 (``reinject_v21tr1``), accepted realizations,
split by their DR3 solution type.

1. **e(G), G < 13.** A factor r on the stated per-CCD error changes only χ² (χ²/ν → χ²/ν / r²): the Halbwachs
   inflation c ∝ 1/r cancels the formal-error scaling, so every reported σ is unchanged, and RUWE is kept unchanged by
   scaling u0 by the same r (UWE ∝ 1/r). Per G bin, r is solved so that the median mock F2 of the DR3-**Orbital**
   orbits equals their DR3 median (bisection, 100-fold orbit bootstrap error).
2. **AstroSpectroSB1 offsets** (after r): per G bin, median DR3 − mock of log σ_a0, log σ_ϖ, log σ_e, log σ_P (additive in
   log) and of F2 (additive), with the F2 quantiles; these are the calibrated type-dependent solution properties.

Writes ``offsets.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_source_scale_595 import chi2red_from_f2, f2_from_chi2red  # noqa: E402

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V21 = D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl"
BINS = {"G<10": (0.0, 10.0), "G10-11": (10.0, 11.0), "G11-12": (11.0, 12.0), "G12-13": (12.0, 13.0), "G13-15": (13.0, 15.0)}


def load(path: Path) -> dict:
    import h5py

    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        pub = {k: P[k][()] for k in ("source_id", "g_mag", "goodness_of_fit", "parallax_error", "significance", "period_error", "eccentricity_error",
                                     "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in P["nss_solution_type"][()]])
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0 = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))
    ix = {int(s): i for i, s in enumerate(pub["source_id"])}
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    acc = [r for r in rows if r.get("accepted") and int(r["source_id"]) in ix]
    i = np.array([ix[int(r["source_id"])] for r in acc])
    m = lambda k: np.array([r[k] for r in acc], float)  # noqa: E731
    return {"g": pub["g_mag"][i], "type": st[i], "f2_d": pub["goodness_of_fit"][i], "f2": m("goodness_of_fit"), "nu": m("n_obs") - 12,
            "lsa0": np.log10(m("sigma_a0_mas")), "lsa0_d": np.log10(a0[i] / pub["significance"][i]),
            "lsplx": np.log10(m("parallax_error")), "lsplx_d": np.log10(pub["parallax_error"][i]),
            "lse": np.log10(m("eccentricity_error")), "lse_d": np.log10(pub["eccentricity_error"][i]),
            "lsP": np.log10(m("period_error")), "lsP_d": np.log10(pub["period_error"][i]), "sid": m("source_id")}


def f2_r(f2, nu, r):
    return f2_from_chi2red(np.clip(chi2red_from_f2(f2, nu) / r**2, 1e-6, None), nu)


def solve_r(f2, nu, target):
    lo, hi = 0.6, 1.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.median(f2_r(f2, nu, mid)) > target:  # larger r -> smaller F2
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reinject", type=Path, default=V21)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = load(a.reinject)
    rng = np.random.default_rng(597)
    res = {"e_refit": {}, "astrospectrosb1_offsets": {}, "orbital_after": {}}
    for name, (lo, hi) in BINS.items():
        gb = (d["g"] >= lo) & (d["g"] < hi)
        orb = gb & (d["type"] == "Orbital")
        if hi <= 13 and orb.sum() >= 15:
            r = solve_r(d["f2"][orb], d["nu"][orb], float(np.median(d["f2_d"][orb])))
            sids = np.unique(d["sid"][orb])
            boots = []
            for _ in range(a.n_boot):
                pick = rng.choice(sids, sids.size)
                idx = np.concatenate([np.flatnonzero(orb & (d["sid"] == s)) for s in pick])
                boots.append(solve_r(d["f2"][idx], d["nu"][idx], float(np.median(d["f2_d"][idx]))))
            res["e_refit"][name] = {"r": r, "r_err": float(np.std(boots)), "n_orbital_realizations": int(orb.sum()),
                                    "f2_median_dr3_orbital": float(np.median(d["f2_d"][orb])), "f2_median_mock_before": float(np.median(d["f2"][orb]))}
        rr = res["e_refit"].get(name, {}).get("r", 1.0) if hi <= 13 else 1.0
        f2a = f2_r(d["f2"], d["nu"], rr)
        for typ, key in (("AstroSpectroSB1", "astrospectrosb1_offsets"), ("Orbital", "orbital_after")):
            b = gb & (d["type"] == typ)
            if b.sum() < 15:
                continue
            row = {"n_realizations": int(b.sum()), "r_applied": rr}
            for q in ("lsa0", "lsplx", "lse", "lsP"):
                row[f"d_{q}"] = float(np.nanmedian(d[q + "_d"][b] - d[q][b]))
            row["d_f2"] = float(np.median(d["f2_d"][b]) - np.median(f2a[b]))
            row["f2_q25_50_75_dr3"] = np.percentile(d["f2_d"][b], [25, 50, 75]).tolist()
            row["f2_q25_50_75_mock_after_r"] = np.percentile(f2a[b], [25, 50, 75]).tolist()
            row["p_f2_gt12_dr3"] = float(np.mean(d["f2_d"][b] > 12))
            row["p_f2_gt12_mock_after_r"] = float(np.mean(f2a[b] > 12))
            if typ == "AstroSpectroSB1":  # F2 affine map: F2' = alpha + kappa F2 (median and inter-quartile range matched)
                qd, qm = row["f2_q25_50_75_dr3"], row["f2_q25_50_75_mock_after_r"]
                kap = (qd[2] - qd[0]) / (qm[2] - qm[0])
                row["f2_kappa"], row["f2_alpha"] = float(kap), float(qd[1] - kap * qm[1])
                row["p_f2_gt12_mock_after_r_and_offset"] = float(np.mean(row["f2_alpha"] + kap * f2a[b] > 12))
            else:
                row["p_f2_gt12_mock_after_r_and_offset"] = row["p_f2_gt12_mock_after_r"]
            res[key][name] = row
    # config-ready arrays at knots 9.5 / 10.5 / 11.5 / 12.5 (bins G<10 / 10-11 / 11-12 / 12-13)
    kb = ("G<10", "G10-11", "G11-12", "G12-13")
    res["config"] = {"knots_g": [9.5, 10.5, 11.5, 12.5], "stated_ratio": [res["e_refit"][b]["r"] for b in kb],
                     **{f"orbital_{q}": [res["orbital_after"][b][f"d_{q}"] for b in kb] for q in ("lsa0", "lsplx", "lse", "lsP")},
                     **{f"asb1_{q}": [res["astrospectrosb1_offsets"][b][f"d_{q}"] for b in kb] for q in ("lsa0", "lsplx", "lse", "lsP")},
                     "asb1_f2_kappa": [res["astrospectrosb1_offsets"][b]["f2_kappa"] for b in kb],
                     "asb1_f2_alpha": [res["astrospectrosb1_offsets"][b]["f2_alpha"] for b in kb]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for k, v in res["e_refit"].items():
        print("e refit", k, {x: round(y, 3) for x, y in v.items()})
    for key in ("orbital_after", "astrospectrosb1_offsets"):
        for k, v in res[key].items():
            print(key, k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if not x.startswith("f2_q")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
