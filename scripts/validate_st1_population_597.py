#!/usr/bin/env python3
"""#597 headline validation: gens 47–50 with e2 + st1 applied post hoc, by DR3 solution type, vs DR3 C1 orbits.

For every gens 47–50 draw whose cascade reached the orbital fit (accepted or failing the cuts), P ≤ 830.4 d, G 6–19.5,
per replicate:

1. e2: F2 → F2(χ²/ν / r(G)², ν = N_obs − 12) at G < 13 (σs and RUWE unchanged by construction);
2. st1: type AstroSpectroSB1 with P(ASB1 | G_RVS(G, BP−RP), K1, BP−RP, e, P) from the configured logistic (the system's
   observed BP−RP; K1 from the fitted cascade), else Orbital; log offsets on σ_ϖ, σ_P, σ_e, σ_a0 by type, and the
   AstroSpectroSB1 F2 affine map;
3. the orbital-solution cuts re-applied.

"Before" is the stored v21tr1 cascade (no e2/st1; the mock has no types). Routing among 5/7/9-parameter outcomes is not
recomputed. DR3: all Orbital + AstroSpectroSB1 rows with P ≤ 830.4 d; the CMD class at G 10–12 from the rung-3 dump.
Mock unweighted (high-ESS view of conditional quantities) and default-target weights. Writes ``population_st1.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_common_v21tr1 as dc  # noqa: E402
from calibrate_source_scale_595 import chi2red_from_f2, f2_from_chi2red  # noqa: E402

G_EDGES = [6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0]
LP_EDGES = [0.0, 1.5, 2.0, 2.5, 2.75, 2.92]
K1_EDGES = [-1.0, 0.0, 0.5, 1.0, 1.5, 3.0]
C_EDGES = [0.0, 0.6, 0.8, 1.0, 1.3, 4.0]


def interp_g(g, knots, vals, break_g, outside=0.0):
    return np.where(g < break_g, np.interp(g, knots, vals), outside)


def p_asb1(st, g, bp_rp, period, ecc, k1):
    c = np.clip(np.where(np.isfinite(bp_rp), bp_rp, 1.0), *st.colour_clip)
    grvs = g - np.polyval(st.grvs_poly, c)
    cols = [np.ones(g.size), grvs] + [np.maximum(0.0, grvs - k) for k in st.grvs_knots]
    feat = {"logK1": np.log10(np.clip(k1, 1e-3, None)), "bp_rp": c, "bp_rp2": c * c, "logP": np.log10(period), "e": ecc}
    cols += [feat[t] for t in st.terms]
    return 1.0 / (1.0 + np.exp(-(np.column_stack(cols) @ np.asarray(st.beta))))


def k1_of(a0, plx, period, ecc, inc_deg):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 29.78 * (a0 / plx) * np.abs(np.sin(np.radians(inc_deg))) / ((period / 365.25) * np.sqrt(np.clip(1 - ecc**2, 1e-6, 1)))


def frac_table(x, is_a, w, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (x >= lo) & (x < hi)
        out.append(float(w[b & is_a].sum() / w[b].sum()) if w[b].sum() > 0 else float("nan"))
    return out


def wq(x, w, q):
    ok = np.isfinite(x) & (w > 0)
    if ok.sum() < 10:
        return float("nan")
    o = np.argsort(x[ok])
    c = np.cumsum(w[ok][o]) / w[ok].sum()
    return float(x[ok][o][min(np.searchsorted(c, q), ok.sum() - 1)])


def main(argv=None) -> int:
    import h5py

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.epoch_model import _solution_type_from, _stated_refit_from

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load_config()
    sec = cfg.dr3.epoch_model
    e2 = _stated_refit_from({**sec.bright_stated_refit.model_dump(), "enabled": True})
    st = _solution_type_from({**sec.solution_type_model.model_dump(), "enabled": True})
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    ridge, ns = dc.ridge_and_nsigma()
    M = dc.load_mock()
    cas = M["cascade"]
    g = M["phot_g_mean_mag"]
    sol = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in M["solution_type"]])
    P = cas[:, 10]
    cand = np.isin(sol, ["twelve_parameter_orbital", "orbital_failed_cuts"]) & (P > 0) & (P <= dc.C1_PMAX) & (g >= 6) & (g < 19.5)
    acc0 = cand & M["accepted"]
    bps = []
    for gg in (47, 48, 49, 50):
        with h5py.File(dc.V / f"single_gen{gg}.h5", "r") as f:
            bps.append(f["truth/sys_bp_rp"][()])
    bp = np.concatenate(bps)
    cls = dc.classes(M["sys_mg0"], M["sys_colour0"], ridge, ns)
    ev = np.isin(cls, ["giant", "subgiant"])
    k1 = k1_of(cas[:, 17], cas[:, 0], P, cas[:, 14], cas[:, 16])
    nu = np.clip(cas[:, 20] - 12, 20, None)
    r = np.where(g < e2.break_g, np.interp(g, e2.knots_g, e2.ratio), 1.0)
    f2_e2 = f2_from_chi2red(np.clip(chi2red_from_f2(cas[:, 21], nu) / r**2, 1e-6, None), nu)
    pA = np.where(cand, p_asb1(st, g, bp, np.where(P > 0, P, 1.0), cas[:, 14], k1), 0.0)
    off = {k: interp_g(g, st.knots_g, v, st.break_g, 1.0 if k.endswith("kappa") else 0.0) for k, v in st.offsets.items()}
    rng = np.random.default_rng(597)
    reps = []
    for _ in range(a.n_rep):
        isA = cand & (rng.random(g.size) < pA)
        pre = np.where(isA, 1, 0)
        d = {q: np.where(pre == 1, off[f"asb1_{q}"], off[f"orbital_{q}"]) for q in ("dlog_sigma_a0", "dlog_sigma_plx", "dlog_sigma_e", "dlog_sigma_p")}
        sa0 = cas[:, 18] * 10 ** d["dlog_sigma_a0"]
        splx = cas[:, 1] * 10 ** d["dlog_sigma_plx"]
        se = cas[:, 15] * 10 ** d["dlog_sigma_e"]
        f2 = np.where(isA, off["asb1_f2_alpha"] + off["asb1_f2_kappa"] * f2_e2, f2_e2)
        with np.errstate(invalid="ignore", divide="ignore"):
            a0s, plxs = cas[:, 17] / sa0, cas[:, 0] / splx
            ok = cand & (a0s > cuts.a0_over_err_min) & (f2 < cuts.goodness_of_fit_f2_max) & (plxs > cuts.parallax_over_error_times_period_min_days / P) \
                & (a0s > cuts.a0_over_err_times_sqrt_period_min / np.sqrt(P)) & (se < cuts.sigma_e_ln_period_slope * np.log(P) + cuts.sigma_e_intercept)
            z = np.log10(a0s / np.sqrt(cas[:, 22] ** 2 - 1))
        reps.append({"acc": ok, "isA": isA, "f2": f2, "sa0": sa0, "splx": splx, "z": z})
    # DR3
    with h5py.File(dc.D / "output/gate390/published_orbits.h5", "r") as f:
        Pp = f["published"]
        pub = {k: Pp[k][()] for k in ("source_id", "g_mag", "period", "eccentricity", "parallax", "parallax_error", "significance", "goodness_of_fit", "ruwe",
                                      "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        rtyp = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in Pp["nss_solution_type"][()]])
    with h5py.File(dc.D / "data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns/columns.h5", "r") as f:
        cmap = dict(zip(f["source_id"][()].tolist(), f["bp_rp"][()].tolist()))
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0d = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))
    rk1 = k1_of(a0d, pub["parallax"], pub["period"], pub["eccentricity"], np.degrees(np.arccos(np.clip(v / a0d**2, -1, 1))))
    rbp = np.array([cmap.get(int(s), np.nan) for s in pub["source_id"]], float)
    rsel = np.isin(rtyp, ["Orbital", "AstroSpectroSB1"]) & (pub["period"] <= dc.C1_PMAX) & (pub["g_mag"] >= 6) & (pub["g_mag"] < 19.5)
    rA = rtyp == "AstroSpectroSB1"
    with np.errstate(invalid="ignore", divide="ignore"):
        rz = np.log10(pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1))
    rsa0 = a0d / pub["significance"]
    # DR3 class at G 10-19.5 from the rung-3 dump, joined on (G, P)
    Rd = dc.load_real()["orbits"]
    key = {(round(float(gg), 4), round(float(pp), 3)): i for i, (gg, pp) in enumerate(zip(Rd["g"], Rd["P"]))}
    rcls_d = dc.classes(Rd["mg0"], Rd["colour0"], ridge, ns)
    rcls = np.array([rcls_d[key[(round(float(gg), 4), round(float(pp), 3))]] if (round(float(gg), 4), round(float(pp), 3)) in key else "unknown"
                     for gg, pp in zip(pub["g_mag"], pub["period"])], dtype=object)
    rev = np.isin(rcls, ["giant", "subgiant"])
    res = {"n_rep": a.n_rep}
    W = {"unweighted": np.ones(g.size), "default": M["w"]}
    ones_r = np.ones(pub["g_mag"].size)
    # 1. type fractions
    tf = {}
    for name, xm, xr, edges in (("G", g, pub["g_mag"], G_EDGES), ("logP", np.log10(np.where(P > 0, P, np.nan)), np.log10(pub["period"]), LP_EDGES),
                                ("logK1", np.log10(np.clip(k1, 1e-3, None)), np.log10(np.clip(rk1, 1e-3, None)), K1_EDGES), ("bp_rp", bp, rbp, C_EDGES)):
        tf[name] = {"edges": edges, "dr3": frac_table(np.where(rsel, xr, np.nan), rA & rsel, ones_r * rsel, edges)}
        for wn, w in W.items():
            tf[name][f"mock_{wn}"] = list(np.nanmean([frac_table(xm, rp["isA"], w * rp["acc"], edges) for rp in reps], axis=0))
    res["type_fractions"] = tf
    # 2-4, 6: per type and G bin
    per = {}
    for lo, hi in ((6, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 15)):
        for tname, tm_fn, tr in (("Orbital", lambda rp: ~rp["isA"], ~rA), ("AstroSpectroSB1", lambda rp: rp["isA"], rA)):
            br = rsel & tr & (pub["g_mag"] >= lo) & (pub["g_mag"] < hi)
            row = {"n_dr3": int(br.sum())}
            if br.sum() >= 20:
                row["dr3"] = {"f2_median": float(np.median(pub["goodness_of_fit"][br])), "p_f2_gt12": float(np.mean(pub["goodness_of_fit"][br] > 12)),
                              "log_sigma_plx_median": float(np.median(np.log10(pub["parallax_error"][br]))),
                              "log_sigma_a0_over_sigma_plx_median": float(np.median(np.log10(rsa0[br] / pub["parallax_error"][br]))),
                              "log_z_p50": float(np.nanpercentile(rz[br], 50)), "log_z_p99": float(np.nanpercentile(rz[br][np.isfinite(rz[br])], 99))}
            for wn, w in W.items():
                vals = []
                for rp in reps:
                    bm = rp["acc"] & tm_fn(rp) & (g >= lo) & (g < hi)
                    if bm.sum() < 20:
                        continue
                    ww = w[bm]
                    vals.append({"n": int(bm.sum()), "f2_median": wq(rp["f2"][bm], ww, 0.5), "p_f2_gt12": float(ww[rp["f2"][bm] > 12].sum() / ww.sum()),
                                 "log_sigma_plx_median": wq(np.log10(rp["splx"][bm]), ww, 0.5),
                                 "log_sigma_a0_over_sigma_plx_median": wq(np.log10(rp["sa0"][bm] / rp["splx"][bm]), ww, 0.5),
                                 "log_z_p50": wq(rp["z"][bm], ww, 0.5), "log_z_p99": wq(rp["z"][bm], ww, 0.99)})
                if vals:
                    row[f"mock_{wn}"] = {k: float(np.nanmean([x[k] for x in vals])) for k in vals[0]}
            # before (no e2/st1, no type): pooled mock, for reference
            bm0 = acc0 & (g >= lo) & (g < hi)
            if bm0.sum() >= 20:
                row["mock_before_pooled_unweighted"] = {"f2_median": float(np.median(cas[bm0, 21])), "p_f2_gt12": float(np.mean(cas[bm0, 21] > 12)),
                                                        "log_z_p99": float(np.nanpercentile(np.log10((cas[bm0, 17] / cas[bm0, 18]) / np.sqrt(np.clip(cas[bm0, 22] ** 2 - 1, 1e-9, None))), 99))}
            per[f"G{lo}-{hi} {tname}"] = row
    res["by_type_and_G"] = per
    # 4b. high-F2 tail by type and class at G 10-12
    tail = {}
    for tname, tm_fn, tr in (("Orbital", lambda rp: ~rp["isA"], ~rA), ("AstroSpectroSB1", lambda rp: rp["isA"], rA)):
        for cname, cm, rc in (("dwarf", cls == "dwarf", rcls == "dwarf"), ("evolved", ev, rev)):
            br = rsel & tr & rc & (pub["g_mag"] >= 10) & (pub["g_mag"] < 12)
            pm = [float(np.mean(rp["f2"][rp["acc"] & tm_fn(rp) & cm & (g >= 10) & (g < 12)] > 12)) for rp in reps
                  if (rp["acc"] & tm_fn(rp) & cm & (g >= 10) & (g < 12)).sum() >= 10]
            p0 = acc0 & cm & (g >= 10) & (g < 12)
            tail[f"{tname} {cname}"] = {"n_dr3": int(br.sum()), "p_dr3": float(np.mean(pub["goodness_of_fit"][br] > 12)) if br.sum() else float("nan"),
                                        "p_mock_after_unweighted": float(np.mean(pm)) if pm else float("nan"),
                                        "p_mock_before_pooled_unweighted": float(np.mean(cas[p0, 21] > 12)) if p0.sum() else float("nan")}
    res["high_f2_tail_G10_12"] = tail
    # 5. short-P bright share among G < 13 orbits
    sp = {}
    for wn, w in W.items():
        b0 = acc0 & (g < 13)
        sp[f"mock_before_{wn}"] = float(w[b0 & (P < 100)].sum() / w[b0].sum())
        sp[f"mock_after_{wn}"] = float(np.mean([w[rp["acc"] & (g < 13) & (P < 100)].sum() / w[rp["acc"] & (g < 13)].sum() for rp in reps]))
        sp[f"mock_after_{wn}_count_ratio_after_over_before"] = float(np.mean([w[rp["acc"] & (g < 13) & (P < 100)].sum() for rp in reps]) / w[b0 & (P < 100)].sum())
    br = rsel & (pub["g_mag"] < 13)
    sp["dr3"] = float(np.mean(pub["period"][br] < 100))
    res["short_p_bright_share"] = sp
    # accepted totals per G (after / before), to see what the routing does to counts
    res["accepted_after_over_before_by_G"] = {f"G{lo}-{hi}": float(np.mean([(rp["acc"] & (g >= lo) & (g < hi)).sum() for rp in reps]) / max((acc0 & (g >= lo) & (g < hi)).sum(), 1))
                                              for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["type_fractions"]["G"]))
    for k, v in res["by_type_and_G"].items():
        print(k, json.dumps({kk: ({x: round(y, 3) for x, y in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()}))
    print(json.dumps(res["high_f2_tail_G10_12"]))
    print(json.dumps(res["short_p_bright_share"]))
    print(json.dumps(res["accepted_after_over_before_by_G"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
