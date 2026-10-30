#!/usr/bin/env python3
"""#591: the short-period bright deficit (P < 100 d, G < 13) under colour_v2.1+tr1.

(a) Cascade acceptance on the #390 re-injection (``reinject_v21tr1``): accepted fraction, outcome (5/7/9-parameter,
    orbit rejected by a cut) and the failing cut, by published log P and G; recovered vs published period.
(b) Population, gens 47–50 accepted C1 orbits (default-target weights and unweighted) against DR3 C1 orbits:
    the short-P share at G < 13 by distance, class, (BP−RP)_0, M_G0, e; mock truth for short-P draws (q, flux ratio,
    M1, e) with the acceptance of all G < 13, P_true < 100 d draws by those truth quantities; the MdS17 short-P
    frequency against M1 in the mock (weights per M1 bin).
(c) Is DR3's short-P bright sample spurious-like? F2, a0/σ_a0, π̂ (v21tr1 §12.16 template), period values near
    scanning-law harmonics (62.97 d / n and its multiples) against the long-P bright sample.

Writes ``shortp.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_common_v21tr1 as dc  # noqa: E402

D = dc.D
LP = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 2.75, 2.92])
CUTS = ("cut_a0_over_err", "cut_f2", "cut_parallax_over_error", "cut_a0_over_err_sqrt_p", "cut_sigma_e")


def part_a(reinject: Path) -> dict:
    import h5py

    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        sid, per, g = P["source_id"][()], P["period"][()], P["g_mag"][()]
    ix = {int(s): i for i, s in enumerate(sid)}
    rows = [json.loads(x) for x in reinject.read_text().splitlines()]
    i = np.array([ix[r["source_id"]] for r in rows])
    lp, gg = np.log10(per[i]), g[i]
    acc = np.array([r["accepted"] for r in rows])
    out_ = np.array([r["outcome"] for r in rows])
    cut = {c: np.array([bool(r.get(c)) for r in rows]) for c in CUTS}
    prec = np.array([r["period"] if r["period"] is not None else np.nan for r in rows], float)
    res = {}
    for gname, gs in (("G<13", gg < 13), ("G>=13", gg >= 13)):
        lst = []
        for lo, hi in zip(LP[:-1], LP[1:]):
            b = gs & (lp >= lo) & (lp < hi)
            if not b.any():
                continue
            o12 = b & (out_ == 12)
            fail = {c: int((o12 & ~acc & ~cut[c]).sum()) for c in CUTS}
            with np.errstate(divide="ignore", invalid="ignore"):
                dp = np.abs(np.log10(prec[b & acc]) - lp[b & acc])
            lst.append({"logP": [float(lo), float(hi)], "n_realizations": int(b.sum()), "accepted": float(acc[b].mean()),
                        "outcome_fractions": {str(k): float(np.mean(out_[b] == k)) for k in (0, 5, 7, 9, 12)},
                        "orbit_rejected_by_cut": fail, "accepted_period_mismatch_gt_5pct": float(np.mean(dp > np.log10(1.05))) if dp.size else float("nan")})
        res[gname] = lst
    return res


def part_b(M, R, ridge, ns) -> dict:
    sel = M["orbit"]
    w = M["w"][sel]
    g = M["phot_g_mean_mag"][sel]
    P = M["o_P"][sel]
    d = M["o_d"][sel]
    cls = dc.classes(M["sys_mg0"][sel], M["sys_colour0"][sel], ridge, ns)
    r = R["orbits"]
    rcls = dc.classes(r["mg0"], r["colour0"], ridge, ns)
    res = {}
    for wname, ww in (("default_target", w), ("unweighted", np.ones(w.size))):
        blk = {}
        # short-P share among G < 13 orbits, by subsets
        subsets = {"all": (np.ones(r["g"].size, bool), np.ones(w.size, bool))}
        for lo, hi in ((0, 0.3), (0.3, 0.7), (0.7, 6)):
            subsets[f"d{lo:g}-{hi:g}"] = ((r["d"] >= lo) & (r["d"] < hi), (d >= lo) & (d < hi))
        for c in ("dwarf", "subgiant", "giant", "unclassified"):
            subsets[c] = (rcls == c, cls == c)
        for lo, hi in ((-1, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 3)):
            subsets[f"colour0 {lo:g}-{hi:g}"] = ((r["colour0"] >= lo) & (r["colour0"] < hi), (M["sys_colour0"][sel] >= lo) & (M["sys_colour0"][sel] < hi))
        for lo, hi in ((-3, 2), (2, 3.5), (3.5, 5), (5, 10)):
            subsets[f"M_G0 {lo:g}-{hi:g}"] = ((r["mg0"] >= lo) & (r["mg0"] < hi), (M["sys_mg0"][sel] >= lo) & (M["sys_mg0"][sel] < hi))
        for name, (sr, sm) in subsets.items():
            br, bm = sr & (r["g"] < 13), sm & (g < 13)
            kr, km = float((br & (r["P"] < 100)).sum()), float(ww[bm & (P < 100)].sum())
            nr, nm = float(br.sum()), float(ww[bm].sum())
            blk[name] = {"dr3_short": kr, "dr3_all": nr, "share_dr3": kr / nr if nr else float("nan"), "share_mock": km / nm if nm else float("nan"),
                         "mock_short_draws": int((bm & (P < 100)).sum()), "mock_short_ess": dc.kish(ww[bm & (P < 100)]),
                         "share_ratio_dr3_over_mock": (kr / nr) / (km / nm) if km > 0 and nr > 0 else float("nan")}
        # eccentricity of short-P bright: DR3 vs mock
        er, em = r["e"][(r["g"] < 13) & (r["P"] < 100)], M["o_e"][sel][(g < 13) & (P < 100)]
        wm_ = ww[(g < 13) & (P < 100)]
        blk["ecc_short_bright"] = {"dr3_quantiles": np.percentile(er, [25, 50, 75]).tolist() if er.size else [],
                                   "mock_quantiles": [float(np.interp(q, np.cumsum(wm_[np.argsort(em)]) / wm_.sum(), np.sort(em))) for q in (0.25, 0.5, 0.75)] if em.size else []}
        res[wname] = blk
    # truth: acceptance of all G < 13, P_true < 100 d draws by q, flux ratio, M1, e_true (unweighted)
    gb = M["phot_g_mean_mag"] < 13
    sp = gb & (M["period_days"] < 100)
    q = M["m2_msun"] / M["m1_msun"]
    fr = M["flux_ratio"]
    acc = M["orbit"]
    tr = {}
    for name, x, e in (("q", q, [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]), ("flux_ratio", fr, [0, 0.01, 0.05, 0.2, 0.5, 0.8, 1.5]),
                       ("m1", M["m1_msun"], [0, 0.8, 1.0, 1.2, 1.5, 2.0, 5]), ("e_true", M["eccentricity"], [0, 0.05, 0.2, 0.4, 0.6, 1.0]),
                       ("logP_true", np.log10(M["period_days"]), [0, 1, 1.5, 1.75, 2.0])):
        rows = []
        for lo, hi in zip(e[:-1], e[1:]):
            b = sp & (x >= lo) & (x < hi)
            rows.append({"bin": [lo, hi], "n_draws": int(b.sum()), "accepted_unweighted": float(acc[b].mean()) if b.sum() else float("nan"),
                         "accepted_default_weighted": float(M["w"][b & acc].sum() / M["w"][b].sum()) if M["w"][b].sum() > 0 else float("nan")})
        tr[name] = rows
    # where do the short-P bright draws go: outcomes of all G<13, P_true<100 d
    stp = M["solution_type"][sp]
    stp = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in stp])
    u, n = np.unique(stp, return_counts=True)
    tr["solution_type_fractions"] = {k: float(v / n.sum()) for k, v in zip(u, n)}
    res["truth_acceptance_short_bright"] = tr
    # MdS17: weight per draw per M1 bin, short vs long P (the target's own short-P fraction as function of M1)
    lpt = np.log10(M["period_days"])
    mf = []
    for lo, hi in ((0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 5)):
        b = gb & (M["m1_msun"] >= lo) & (M["m1_msun"] < hi) & (lpt < 2.92)
        mf.append({"m1": [lo, hi], "target_weight_share_P_lt_100": float(M["w"][b & (lpt < 2)].sum() / M["w"][b].sum()) if M["w"][b].sum() > 0 else float("nan"),
                   "accepted_weight_share_P_lt_100": float(M["w"][b & acc & (lpt < 2)].sum() / M["w"][b & acc].sum()) if M["w"][b & acc].sum() > 0 else float("nan")})
    res["mds17_short_share_vs_m1"] = mf
    # twins: share of the target weight vs of the accepted weight, short vs long P (G < 13)
    tw = {}
    for name, b in (("P<100", gb & (M["period_days"] < 100)), ("P100-830", gb & (M["period_days"] >= 100) & (M["period_days"] < 830.4))):
        tw[name] = {"twin_share_target": float(M["w"][b & (q >= 0.95)].sum() / M["w"][b].sum()),
                    "twin_share_accepted": float(M["w"][b & acc & (q >= 0.95)].sum() / M["w"][b & acc].sum()) if M["w"][b & acc].sum() > 0 else float("nan"),
                    "accepted_draws": int((b & acc).sum())}
    res["twins_g_lt_13"] = tw
    return res


def part_c(R) -> dict:
    from darkhunter_pop.rung3_model import spurious_probability
    from darkhunter_pop.spuriousness_model import harmonic_distance

    cal = json.loads((dc.V / "spur/spurious_calibration").read_text())
    r = R["orbits"]
    pi = spurious_probability(cal, {"log10_a0_snr": np.log10(r["a0snr"]), "log10_parallax_snr": np.log10(r["plxsnr"])})
    hd = np.array([harmonic_distance(p, fundamental_period_days=62.97, multiples_max=20) for p in r["P"]])
    sub = np.array([np.min(np.abs(p - 62.97 / np.arange(1, 8)) / p) for p in r["P"]])  # sub-harmonics 62.97 / n
    out = {}
    for name, b in (("short_bright", (r["g"] < 13) & (r["P"] < 100)), ("long_bright", (r["g"] < 13) & (r["P"] >= 100)),
                    ("short_faint", (r["g"] >= 13) & (r["P"] < 100))):
        out[name] = {"n": int(b.sum()), "f2_quantiles": np.percentile(r["f2"][b], [50, 90]).tolist(), "a0snr_median": float(np.median(r["a0snr"][b])),
                     "pi_mean": float(pi[b].mean()), "pi_median": float(np.median(pi[b])),
                     "frac_within_2pct_of_62.97d_multiple": float(np.mean(hd[b] < 0.02)), "frac_within_2pct_of_62.97d_over_n": float(np.mean(sub[b] < 0.02)),
                     "d_median_kpc": float(np.median(r["d"][b])), "frac_d_lt_0.3": float(np.mean(r["d"][b] < 0.3)),
                     "e_median": float(np.median(r["e"][b]))}
    h, e = np.histogram(np.log10(r["P"][(r["g"] < 13) & (r["P"] < 100)]), np.linspace(0.5, 2.0, 16))
    out["logP_hist_short_bright"] = {"edges": e.tolist(), "counts": h.tolist()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reinject", type=Path, default=D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    res = {"a_reinjection": part_a(a.reinject)}
    print("a done", flush=True)
    ridge, ns = dc.ridge_and_nsigma()
    M, R = dc.load_mock(), dc.load_real()
    res["b_population"] = part_b(M, R, ridge, ns)
    print("b done", flush=True)
    res["c_spurious"] = part_c(R)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "shortp.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
