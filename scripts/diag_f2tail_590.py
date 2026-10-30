#!/usr/bin/env python3
"""#590: where is the bright high-F2 excess (F2 > 12, G 11–12.5) of DR3 over the colour_v2.1+tr1 mock?

Mock: every accepted C1 orbit / published acceleration of gens 47–50 (``diag_common_v21tr1.load_mock``), with the
default-target (MdS17) mixture weights and, as a check, unweighted (the conditional F2 distribution at fixed class and
G is a noise property, so draw counts are the high-ESS view). Mock CMD = the system's noise-free dereddened truth.
Real: the DR3 C1 orbits / accelerations of the rung-3 fit (#587 dump). Classes on the dereddened CMD with the parent's
MS ridge (``giants.classify_evolved``): dwarf, subgiant (evolved, M_G0 ≥ 2), giant (evolved, M_G0 < 2).

For orbits and accelerations at G 11–12.5: P(F2 > 12 | bin) in DR3 and mock, the DR3/mock ratio of the tail
count, and the tail composition, by class, M_G0, (BP−RP)_0, P, A and a0/σ_a0 (orbits). The F2 distribution of
dwarfs vs giants (quantiles), and the F2 > 12 fraction against M_G0 at all G (luminosity trend at fixed class).

Writes ``f2tail.json`` and ``f2tail.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag_common_v21tr1 as dc  # noqa: E402
from darkhunter_pop import plotting as pl
from darkhunter_pop.config_loader import load_config

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
NC = D.parent / "dark-hunter_pop-worktrees/noise-check-587/output/nc587"
BINS = {"mg0": [-3, 0, 1, 2, 3, 4, 5, 8], "colour0": [0, 0.6, 0.8, 1.0, 1.2, 1.5, 3], "logP": [0, 1.5, 2, 2.5, 2.75, 3.0],
        "A": [0, 0.1, 0.2, 0.35, 0.5, 5], "loga0snr": [0, 1, 1.25, 1.5, 1.75, 4]}


def tail_table(r, m, key, edges, f2cut, gsel):
    out = []
    xr = r[key] if key != "cls" else r["cls"]
    xm = m[key] if key != "cls" else m["cls"]
    cats = ["dwarf", "subgiant", "giant", "unclassified"] if key == "cls" else list(zip(edges[:-1], edges[1:]))
    for c in cats:
        br = gsel(r) & ((xr == c) if key == "cls" else ((xr >= c[0]) & (xr < c[1])))
        bm = gsel(m) & ((xm == c) if key == "cls" else ((xm >= c[0]) & (xm < c[1])))
        tr, tm = br & (r["f2"] > f2cut), bm & (m["f2"] > f2cut)
        nr, nm, kr, km = float(br.sum()), float(m["w"][bm].sum()), float(tr.sum()), float(m["w"][tm].sum())
        ess = float(m["w"][tm].sum() ** 2 / (m["w"][tm] ** 2).sum()) if km > 0 else 0.0
        out.append({"bin": c if key == "cls" else [float(c[0]), float(c[1])], "n_dr3": nr, "n_mock": nm, "tail_dr3": kr, "tail_mock": km,
                    "tail_mock_ess": ess, "p_tail_dr3": kr / nr if nr else float("nan"), "p_tail_mock": km / nm if nm else float("nan"),
                    "tail_ratio_dr3_over_mock": kr / km if km else float("nan")})
    tot_r = sum(x["tail_dr3"] for x in out)
    tot_m = sum(x["tail_mock"] for x in out)
    for x in out:
        x["share_dr3"] = x["tail_dr3"] / tot_r if tot_r else float("nan")
        x["share_mock"] = x["tail_mock"] / tot_m if tot_m else float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d-min", type=float, default=0.0)
    ap.add_argument("--f2-cut", type=float, default=12.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load_config()
    ridge, ns = dc.ridge_and_nsigma()
    M = dc.load_mock()
    R = dc.load_real()
    mcls = dc.classes(M["sys_mg0"], M["sys_colour0"], ridge, ns)
    sets = {}
    for kind, sel, pre in (("orbits", M["orbit"], "o"), ("accelerations", M["accel"], "a")):
        m = {"g": M["phot_g_mean_mag"][sel], "d": M[f"{pre}_d"][sel], "colour0": M["sys_colour0"][sel], "mg0": M["sys_mg0"][sel],
             "f2": M[f"{pre}_f2"][sel], "cls": mcls[sel], "w": M["w"][sel], "teff": 10 ** M["primary_log_teff"][sel], "m1": M["m1_msun"][sel]}
        if pre == "o":
            m.update(logP=np.log10(M["o_P"][sel]), A=M["o_A"][sel], loga0snr=np.log10(M["o_a0snr"][sel]))
        r = dict(R[kind])
        r["w"] = np.ones(r["g"].size)
        r["cls"] = dc.classes(r["mg0"], r["colour0"], ridge, ns)
        if "P" in r:
            r["logP"], r["loga0snr"] = np.log10(r["P"]), np.log10(r["a0snr"])
        km, kr = m["d"] >= a.d_min, r["d"] >= a.d_min
        sets[kind] = ({k: v[kr] for k, v in r.items()}, {k: v[km] for k, v in m.items()})
    res = {"f2_cut": a.f2_cut, "d_min": a.d_min}
    gsel = lambda s: (s["g"] >= 11.0) & (s["g"] < 12.5)  # noqa: E731
    for wname in ("default_target", "unweighted"):
        res[wname] = {}
        for kind, (r, m0) in sets.items():
            m = dict(m0)
            if wname == "unweighted":
                m["w"] = np.ones(m["w"].size)
            blk = {"cls": tail_table(r, m, "cls", None, a.f2_cut, gsel)}
            for key, e in BINS.items():
                if key in r and key in m:
                    blk[key] = tail_table(r, m, key, e, a.f2_cut, gsel)
            q = {}
            for c in ("dwarf", "subgiant", "giant"):
                br, bm = gsel(r) & (r["cls"] == c), gsel(m) & (m["cls"] == c)
                if br.sum() < 20 or bm.sum() < 20:
                    continue
                o = np.argsort(m["f2"][bm])
                cw = np.cumsum(m["w"][bm][o]) / m["w"][bm].sum()
                q[c] = {"dr3": np.percentile(r["f2"][br], [50, 90, 99]).tolist(), "n_dr3": int(br.sum()), "n_mock": int(bm.sum()), "mock_ess": dc.kish(m["w"][bm]),
                        "mock": [float(m["f2"][bm][o][min(np.searchsorted(cw, p), bm.sum() - 1)]) for p in (0.5, 0.9, 0.99)]}
            blk["f2_quantiles_by_class"] = q
            lt = {}
            for c in ("dwarf", "evolved"):
                rows = []
                for lo, hi in zip(BINS["mg0"][:-1], BINS["mg0"][1:]):
                    isc = (lambda s_: (s_["cls"] == "dwarf") if c == "dwarf" else np.isin(s_["cls"], ["giant", "subgiant"]))
                    mr = (r["g"] < 13) & (r["mg0"] >= lo) & (r["mg0"] < hi) & isc(r)
                    mm = (m["g"] < 13) & (m["mg0"] >= lo) & (m["mg0"] < hi) & isc(m)
                    rows.append({"mg0": [lo, hi], "n_dr3": int(mr.sum()), "n_mock": int(mm.sum()), "mock_ess": dc.kish(m["w"][mm]),
                                 "p_dr3": float(np.mean(r["f2"][mr] > a.f2_cut)) if mr.sum() >= 20 else float("nan"),
                                 "p_mock": float(m["w"][mm & (m["f2"] > a.f2_cut)].sum() / m["w"][mm].sum()) if mm.sum() >= 20 else float("nan")})
                lt[c] = rows
            blk["p_tail_vs_mg0_G_lt_13"] = lt
            res[wname][kind] = blk
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "f2tail.json").write_text(json.dumps(res, indent=1, default=float))
    # figure: P(F2>cut) vs M_G0 by class, DR3 vs mock (a), orbits and accelerations
    style = cfg.plotting
    plt = pl.require_pyplot()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    first = "unweighted"
    for ax, kind in zip(axs, ("orbits", "accelerations")):
        for j, c in enumerate(("dwarf", "evolved")):
            rows = res[first][kind]["p_tail_vs_mg0_G_lt_13"][c]
            x = [0.5 * (r_["mg0"][0] + r_["mg0"][1]) for r_ in rows]
            st = pl.series_style(j, style)
            ax.plot(x, [r_["p_dr3"] for r_ in rows], "o-", color=st["color"], label=f"DR3 {c}")
            ax.plot(x, [r_["p_mock"] for r_ in rows], "s--", color=st["color"], label=f"mock ({first}) {c}")
        ax.set_yscale("log")
        pl.apply_axes_style(ax, style, xlabel=r"$M_{G,0}$ (mag)", ylabel=f"P(F2 > {a.f2_cut:g}), G < 13", title=kind)
        ax.legend(fontsize=8, frameon=False)
    pl.save_figure(fig, a.out_dir / "f2tail.png", dpi=110)
    for name in ("default_target", "unweighted"):
        for kind in ("orbits", "accelerations"):
            print(name, kind, "class:", [(x["bin"], int(x["tail_dr3"]), round(x["tail_mock"]), round(x["tail_ratio_dr3_over_mock"], 2), round(x["share_dr3"], 2), round(x["share_mock"], 2)) for x in res[name][kind]["cls"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
