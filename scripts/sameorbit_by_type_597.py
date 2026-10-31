#!/usr/bin/env python3
"""#597: #390 same-orbit comparison by DR3 solution type for a re-injection run (median DR3 − mock of log σ_a0, log σ_ϖ,
F2; P(F2 > 12); RUWE ratio; agreement of the mock-routed type with the DR3 type), G < 13. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="label=inject.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        pub = {k: P[k][()] for k in ("source_id", "g_mag", "goodness_of_fit", "parallax_error", "significance", "ruwe",
                                     "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in P["nss_solution_type"][()]])
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    sa0 = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None))) / pub["significance"]
    ix = {int(s): i for i, s in enumerate(pub["source_id"])}
    res = {}
    for spec in a.run:
        lab, path = spec.split("=", 1)
        rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
        acc = [r for r in rows if r.get("accepted")]
        i = np.array([ix[int(r["source_id"])] for r in acc])
        m = lambda k: np.array([r[k] for r in acc], float)  # noqa: E731
        mt = np.array([r.get("nss_type") or "" for r in acc])
        res[lab] = {}
        for typ in ("Orbital", "AstroSpectroSB1"):
            for lo, hi in ((0, 10), (10, 11), (11, 12), (12, 13)):
                b = (pub["g_mag"][i] >= lo) & (pub["g_mag"][i] < hi) & (st[i] == typ)
                if b.sum() < 15:
                    continue
                res[lab][f"{typ} G{lo}-{hi}"] = {
                    "n": int(b.sum()), "d_log_sigma_a0": float(np.median(np.log10(sa0[i][b]) - np.log10(m("sigma_a0_mas")[b]))),
                    "d_log_sigma_plx": float(np.median(np.log10(pub["parallax_error"][i][b]) - np.log10(m("parallax_error")[b]))),
                    "d_f2": float(np.median(pub["goodness_of_fit"][i][b]) - np.median(m("goodness_of_fit")[b])),
                    "p_f2_gt12_dr3": float(np.mean(pub["goodness_of_fit"][i][b] > 12)), "p_f2_gt12_mock": float(np.mean(m("goodness_of_fit")[b] > 12)),
                    "ruwe_ratio_mock_over_dr3": float(np.median(m("ruwe")[b] / pub["ruwe"][i][b])),
                    "mock_type_agrees": float(np.mean(mt[b] == typ)) if (mt != "").any() else float("nan")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for lab, t in res.items():
        for k, v in t.items():
            print(lab, k, {x: round(y, 3) for x, y in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
