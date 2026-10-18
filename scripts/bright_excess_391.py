#!/usr/bin/env python
"""#391 (Ryan 2026-10-18): diagnose the bright G < 10 orbit excess and characterise the real orbits in the
distant (d > 1 kpc) short-P (log P < 2.6) coarse cells. Analysis only; no draws.

(a) mock (best-fit weights, ``rung3_refit5_391.py fit --dump-from``) vs real, orbits and accelerations, split by
    G and, at G < 10, by class / distance / log P / M1;
(b) rung-1 (#390 set, #400 v2+N2-u0 re-injection) re-detection fraction vs G;
(c) mock single-star RUWE (v2+N2-u0) vs DR3 RUWE of the same sources, by G;
(d) parent subsample vs expectation, and Halbwachs IPD / C* pass rates by G: parent stars vs real NSS orbits;
(e) the distant short-P real orbits: significance, ϖ/σ, π̂ (spurious model), solution type, vs the bulk.
Writes ``summary.json`` and one contact sheet ``bright_excess.png`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

G_EDGES = np.array([5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 19.5])


def ratio_by_g(mock_g, mock_w, real_g, spur=None, edges=G_EDGES):
    m = np.histogram(mock_g, bins=edges, weights=mock_w)[0]
    r = np.histogram(real_g, bins=edges)[0].astype(float)
    s = np.histogram(real_g, bins=edges, weights=spur)[0] if spur is not None else np.zeros_like(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m, s, r, (m + s) / r


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w**2)) if w.size and np.sum(w**2) > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--gate390", type=Path, required=True)
    ap.add_argument("--gate400-validation", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import h5py
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config(Path("config/config.yaml"))
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    out: dict = {}

    # ---------------- (a) split ----------------
    dumps = {k: np.load(args.dump_dir / f"dump_{k}.npz", allow_pickle=True) for k in ("noW", "cmdW") if (args.dump_dir / f"dump_{k}.npz").exists()}
    out["a"] = {}
    for k, z in dumps.items():
        w = z["w"]
        so = (z["ob"] >= 0)
        sa = (z["ab"] >= 0)
        spo = z["r_spur_per_orbit"]
        rin = z["rob"] >= 0
        mo, so_, ro, xo = ratio_by_g(z["m_g"][so], w[so], z["r_g"][rin], spo[rin])
        ma, _, ra, xa = ratio_by_g(z["m_g_a"][sa], w[sa], z["ra_g"])
        ess_o = [kish(w[so & (z["m_g"] >= a) & (z["m_g"] < b)]) for a, b in zip(G_EDGES[:-1], G_EDGES[1:])]
        ess_a = [kish(w[sa & (z["m_g_a"] >= a) & (z["m_g_a"] < b)]) for a, b in zip(G_EDGES[:-1], G_EDGES[1:])]
        res = {"g_edges": G_EDGES.tolist(), "orbit_mock": mo.tolist(), "orbit_spur": so_.tolist(), "orbit_real": ro.tolist(),
               "orbit_ratio": xo.tolist(), "orbit_mock_ess": ess_o, "accel_mock": ma.tolist(), "accel_real": ra.tolist(),
               "accel_ratio": xa.tolist(), "accel_mock_ess": ess_a}
        # splits at G < 10: mock weight vs real (real minus spurious) per sub-bin
        bo, br = so & (z["m_g"] < 10), rin & (z["r_g"] < 10)
        ba, bra = sa & (z["m_g_a"] < 10), z["ra_g"] < 10
        m1 = z["m1_law"]
        splits = {}
        for nm, mv, rv, edges in (("class", z["m_cls"], z["r_cls"], [-0.5, 0.5, 1.5]),
                                  ("distance_kpc", z["m_d_o"], z["r_d"], [0, 0.1, 0.2, 0.3, 0.5, 1.0, 6.0]),
                                  ("log10_P", z["m_lp"], z["r_lp"], [0, 2.0, 2.4, 2.6, 2.75, 2.92]),
                                  ("eccentricity", z["m_e"], z["r_e"], [0, 0.1, 0.3, 0.5, 1.0])):
            m_ = np.histogram(mv[bo], bins=edges, weights=w[bo])[0]
            r_ = np.histogram(rv[br], bins=edges)[0].astype(float)
            s_ = np.histogram(rv[br], bins=edges, weights=spo[br])[0]
            e_ = [kish(w[bo & (mv >= a) & (mv < b)]) for a, b in zip(edges[:-1], edges[1:])]
            splits[nm] = {"edges": edges, "mock": m_.tolist(), "spur": s_.tolist(), "real": r_.tolist(), "mock_ess": e_,
                          "ratio": ((m_ + s_) / np.where(r_ > 0, r_, np.nan)).tolist()}
        # M1 (mock only; real M1 not in the dump): mock weight share by M1_law
        me = [0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 100.0]
        splits["mock_M1_law_share"] = {"edges": me, "orbit": (np.histogram(m1[bo], bins=me, weights=w[bo])[0] / w[bo].sum()).tolist(),
                                       "orbit_G10_14_ref": (np.histogram(m1[so & (z["m_g"] >= 10) & (z["m_g"] < 14)], bins=me,
                                                                         weights=w[so & (z["m_g"] >= 10) & (z["m_g"] < 14)])[0]
                                                            / w[so & (z["m_g"] >= 10) & (z["m_g"] < 14)].sum()).tolist()}
        for nm, mv, rv, edges in (("accel_class", z["m_cls"], z["ra_cls"], [-0.5, 0.5, 1.5]),
                                  ("accel_distance_kpc", z["m_d_a"], z["ra_d"], [0, 0.1, 0.2, 0.3, 0.5, 1.0, 6.0])):
            m_ = np.histogram(mv[ba], bins=edges, weights=w[ba])[0]
            r_ = np.histogram(rv[bra], bins=edges)[0].astype(float)
            splits[nm] = {"edges": edges, "mock": m_.tolist(), "real": r_.tolist(),
                          "ratio": (m_ / np.where(r_ > 0, r_, np.nan)).tolist()}
        res["G_lt_10_splits"] = splits
        out["a"][k] = res

    # ---------------- (a2) detected-orbit properties at the bright end: mock (weighted) vs real ----------------
    def wq(x, w, qs=(0.1, 0.5, 0.9)):
        ok = np.isfinite(x) & (w > 0)
        x, w = x[ok], w[ok]
        if not x.size:
            return []
        o = np.argsort(x)
        c = np.cumsum(w[o]) / w.sum()
        return [float(x[o][np.searchsorted(c, q_)]) for q_ in qs]

    if "noW" in dumps:
        z0 = dumps["noW"]
        so0, rin0 = z0["ob"] >= 0, z0["rob"] >= 0
        cas = z0["casc"]
        A_, B_, F_, G_ = (z0[f"r_x_{c}"] for c in "ABFG")
        u_ = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
        ra0 = np.sqrt(u_ + np.sqrt(np.maximum(u_**2 - (A_ * G_ - B_ * F_) ** 2, 0)))
        props = {}
        for a, b in ((5.0, 9.0), (9.0, 10.0), (11.0, 13.0)):
            m = so0 & (z0["m_g"] >= a) & (z0["m_g"] < b)
            r = rin0 & (z0["r_g"] >= a) & (z0["r_g"] < b)
            props[f"G{a:g}-{b:g}"] = {
                "mock_significance": wq(cas[:, 17] / cas[:, 18], z0["w"] * m), "real_significance": wq(10 ** z0["r_cov_log10_a0_snr"], r.astype(float)),
                "mock_a0_mas": wq(cas[:, 17], z0["w"] * m), "real_a0_mas": wq(ra0, r.astype(float)),
                "mock_F2": wq(cas[:, 21], z0["w"] * m), "real_F2": wq(z0["r_cov_goodness_of_fit"], r.astype(float)),
                "mock_ruwe": wq(cas[:, 22], z0["w"] * m), "real_ruwe": wq(z0["r_x_ruwe"], r.astype(float)),
                "mock_sigma_a0_mas": wq(cas[:, 18], z0["w"] * m),
                "mock_share_significance_lt_20": float(np.sum(z0["w"][m & (cas[:, 17] / cas[:, 18] < 20)]) / np.sum(z0["w"][m])),
                "real_share_significance_lt_20": float(np.mean(10 ** z0["r_cov_log10_a0_snr"][r] < 20)),
            }
        out["a2"] = props

    # ---------------- (b) rung-1 re-detection vs G ----------------
    with h5py.File(args.gate390 / "injection_test_full.h5", "r") as h:
        sys_sid = h["systems/source_id"][()]
        sys_g = h["systems/truth/g_mag"][()]
        sys_type = h["systems/nss_solution_type"][()].astype(str)
    gmap = dict(zip(sys_sid.tolist(), sys_g.tolist()))
    tmap = dict(zip(sys_sid.tolist(), sys_type.tolist()))
    acc, gg, ty = [], [], []
    for line in (args.gate400_validation / "inject.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("variant") != "v2_u0":
            continue
        acc.append(bool(r["accepted"]))
        gg.append(gmap.get(int(r["source_id"]), np.nan))
        ty.append(tmap.get(int(r["source_id"]), ""))
    acc, gg, ty = np.array(acc), np.array(gg), np.array(ty)
    be = [3.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 19.5]
    out["b"] = {"g_edges": be, "by_type": {}}
    for t in ("Orbital", "AstroSpectroSB1"):
        sel = ty == t
        n = np.histogram(gg[sel], bins=be)[0]
        k = np.histogram(gg[sel & acc], bins=be)[0]
        out["b"]["by_type"][t] = {"n": n.tolist(), "accepted": k.tolist(), "frac": (k / np.where(n > 0, n, np.nan)).tolist()}

    # ---------------- (c) single-star RUWE ----------------
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    pc = parent.columns
    psid = np.asarray(pc["source_id"], np.int64)
    pru = dict(zip(psid.tolist(), np.asarray(pc["ruwe"], float).tolist()))
    sg, sr, dr = [], [], []
    for line in (args.gate400_validation / "single.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("variant") != "v2_u0":
            continue
        sg.append(r["g"])
        sr.append(r["ruwe"])
        dr.append(pru.get(int(r["source_id"]), np.nan))
    sg, sr, dr = np.array(sg), np.array(sr), np.array(dr)
    ce = [3.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 19.0]
    c_ = {"g_edges": ce, "n": [], "n_dr3_matched": [], "mock_median": [], "dr3_median": [], "mock_frac_gt1p4": [], "dr3_frac_gt1p4": []}
    # DR3 side from the whole parent where the singles set has no match (bulk RUWE distribution of all stars at that G)
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    prw = np.asarray(pc["ruwe"], float)
    for a, b in zip(ce[:-1], ce[1:]):
        s = (sg >= a) & (sg < b)
        p = (pg >= a) & (pg < b) & np.isfinite(prw)
        c_["n"].append(int(s.sum()))
        c_["n_dr3_matched"].append(int(np.isfinite(dr[s]).sum()))
        c_["mock_median"].append(float(np.median(sr[s])) if s.any() else np.nan)
        c_["mock_frac_gt1p4"].append(float(np.mean(sr[s] > 1.4)) if s.any() else np.nan)
        c_["dr3_median"].append(float(np.median(prw[p])) if p.any() else np.nan)
        c_["dr3_frac_gt1p4"].append(float(np.mean(prw[p] > 1.4)) if p.any() else np.nan)
        c_.setdefault("n_parent", []).append(int(p.sum()))
    # u0 calibration singles (600 per 0.25 mag): RUWE = UWE / u0_mock(G) tail, mock singles only
    u0t = np.loadtxt(Path("config/epoch_model/dr3_ruwe_u0_mock.csv"), delimiter=",", comments="#", skiprows=0, dtype=str)
    u0t = np.array([[float(a), float(b)] for a, b in u0t if a != "g"])
    ug, uw = [], []
    for line in (args.gate400_validation.parent / "u0" / "u0_singles.jsonl").read_text().splitlines():
        r = json.loads(line)
        ug.append(r["g"])
        uw.append(r["uwe"])
    ug, uw = np.array(ug), np.array(uw)
    uru = uw / np.interp(ug, u0t[:, 0], u0t[:, 1])
    c_["u0_singles_frac_gt1p4"] = [float(np.mean(uru[(ug >= a) & (ug < b)] > 1.4)) if np.any((ug >= a) & (ug < b)) else np.nan
                                   for a, b in zip(ce[:-1], ce[1:])]
    c_["u0_singles_n"] = [int(np.sum((ug >= a) & (ug < b))) for a, b in zip(ce[:-1], ce[1:])]
    c_["u0_singles_p90"] = [float(np.percentile(uru[(ug >= a) & (ug < b)], 90)) if np.any((ug >= a) & (ug < b)) else np.nan
                            for a, b in zip(ce[:-1], ce[1:])]
    c_["dr3_p90"] = [float(np.percentile(prw[(pg >= a) & (pg < b) & np.isfinite(prw)], 90)) for a, b in zip(ce[:-1], ce[1:])]
    out["c"] = c_

    # ---------------- (d) parent counts and cut pass rates ----------------
    de = [3.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 19.0]
    fl = parent.flags
    base = np.ones(pg.size, bool)
    for kf in ("finite_astrometry_photometry", "parallax_floor", "isochrone_m1", "bailer_jones_distance"):
        if kf in fl:
            base &= np.asarray(fl[kf], bool)
    d_ = {"g_edges": de, "parent_rows": [], "parent_poisson_rel": [], "parent_pass_ipd": [], "parent_pass_cstar": [], "parent_isochrone_m1": [],
          "real_rows": [], "real_pass_ipd": [], "real_pass_cstar": []}
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    inp = ps.load_real_input_columns(args.real_input_columns)
    rsid = np.asarray(rt["source_id"], np.int64)
    order = np.argsort(inp["source_id"])
    pos = np.clip(np.searchsorted(inp["source_id"][order], rsid), 0, inp["source_id"].size - 1)
    found = inp["source_id"][order][pos] == rsid
    joined = {k_: np.where(found, np.asarray(v, float)[order][pos], np.nan) for k_, v in inp.items() if k_ != "source_id"}
    rflags = ps.halbwachs_input_flags(joined, prop.halbwachs_cuts)
    rg = np.ma.filled(np.ma.asarray(rt["g_mag"], float), np.nan)
    rplx = np.ma.filled(np.ma.asarray(rt["parallax"], float), np.nan)
    rbase = rplx > prop.parallax_floor_mas
    for a, b in zip(de[:-1], de[1:]):
        s = (pg >= a) & (pg < b)
        d_["parent_rows"].append(int(s.sum()))
        d_["parent_poisson_rel"].append(float(1 / np.sqrt(max(s.sum(), 1))))
        d_["parent_isochrone_m1"].append(float(np.mean(fl["isochrone_m1"][s])) if "isochrone_m1" in fl and s.any() else np.nan)
        sb = s & base
        d_["parent_pass_ipd"].append(float(np.mean(fl["halbwachs_ipd"][sb])) if sb.any() else np.nan)
        d_["parent_pass_cstar"].append(float(np.mean(fl["halbwachs_cstar"][sb & fl["halbwachs_ipd"]])) if (sb & fl["halbwachs_ipd"]).any() else np.nan)
        r = (rg >= a) & (rg < b) & rbase
        d_["real_rows"].append(int(r.sum()))
        d_["real_pass_ipd"].append(float(np.mean(rflags["halbwachs_ipd"][r])) if r.any() else np.nan)
        d_["real_pass_cstar"].append(float(np.mean(rflags["halbwachs_cstar"][r & rflags["halbwachs_ipd"]])) if (r & rflags["halbwachs_ipd"]).any() else np.nan)
    out["d"] = d_

    # ---------------- (e) distant short-P real orbits ----------------
    z = dumps.get("noW")
    if z is not None:
        rin = z["rob"] >= 0
        ds = rin & (z["r_d"] > 1.0) & (z["r_lp"] < 2.6)
        bulk = rin & ~ds
        sig = 10 ** z["r_cov_log10_a0_snr"]
        psnr = 10 ** z["r_cov_log10_parallax_snr"]
        gof = z["r_cov_goodness_of_fit"]

        def q(x, m):
            x = x[m & np.isfinite(x)]
            return [float(v) for v in np.percentile(x, [10, 50, 90])] if x.size else []

        st = z["r_soltype"]
        out["e"] = {"n": int(ds.sum()), "n_by_G": {"5-13": int((ds & (z["r_g"] < 13)).sum()), "13-19.5": int((ds & (z["r_g"] >= 13)).sum())},
                    "solution_type": {t: int((ds & (st == t)).sum()) for t in np.unique(st[ds])},
                    "bulk_solution_type_share": {t: float((bulk & (st == t)).sum() / bulk.sum()) for t in np.unique(st[bulk])},
                    "evolved_share": float(z["r_cls"][ds].mean()) if ds.any() else np.nan,
                    "significance_p10_50_90": q(sig, ds), "bulk_significance_p10_50_90": q(sig, bulk),
                    "parallax_snr_p10_50_90": q(psnr, ds), "bulk_parallax_snr_p10_50_90": q(psnr, bulk),
                    "goodness_of_fit_p10_50_90": q(gof, ds), "bulk_goodness_of_fit_p10_50_90": q(gof, bulk),
                    "pi_hat_mean": float(np.mean(z["r_pi"][ds])) if ds.any() else np.nan,
                    "pi_hat_sum": float(np.sum(z["r_pi"][ds])), "bulk_pi_hat_mean": float(np.mean(z["r_pi"][bulk])),
                    "model_spurious_expected_there": float(np.sum(z["r_spur_per_orbit"][ds])),
                    "mock_expected_there": float(z["w"][(z["ob"] >= 0) & (z["m_d_o"] > 1.0) & (z["m_lp"] < 2.6)].sum()),
                    "eccentricity_p10_50_90": q(z["r_e"], ds), "distance_kpc_p10_50_90": q(z["r_d"], ds),
                    "log10_P_p10_50_90": q(z["r_lp"], ds)}
        if "r_x_ruwe" in z.files:
            out["e"]["ruwe_p10_50_90"] = q(z["r_x_ruwe"], ds)
            out["e"]["bulk_ruwe_p10_50_90"] = q(z["r_x_ruwe"], bulk)
        if "r_x_A" in z.files:
            A, B, F, G = (z[f"r_x_{c}"] for c in "ABFG")
            u = 0.5 * (A**2 + B**2 + F**2 + G**2)
            v = A * G - B * F
            a0 = np.sqrt(u + np.sqrt(np.maximum(u**2 - v**2, 0)))
            out["e"]["a0_mas_p10_50_90"] = q(a0, ds)
            out["e"]["bulk_a0_mas_p10_50_90"] = q(a0, bulk)
            out["e"]["a0_over_parallax_au_p10_50_90"] = q(a0 / z["r_x_parallax"], ds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))

    # ---------------- contact sheet ----------------
    plt = require_pyplot()
    style = resolve_plotting_style(cfg.plotting)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    gc = 0.5 * (G_EDGES[1:] + G_EDGES[:-1])
    ax = axs[0, 0]
    for i, k in enumerate(dumps):
        ax.plot(gc, out["a"][k]["orbit_ratio"], marker="o", label=f"orbits {k}", **{kk: vv for kk, vv in series_style(i, style).items() if kk == "color"})
        ax.plot(gc, out["a"][k]["accel_ratio"], marker="s", ls="--", label=f"accelerations {k}",
                **{kk: vv for kk, vv in series_style(i, style).items() if kk == "color"})
    ax.axhline(1, color="0.5", lw=1)
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="(mock + spurious) / real")
    ax.set_title("(a) best-fit ratio by G")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    if "noW" in out["a"]:
        sp = out["a"]["noW"]["G_lt_10_splits"]
        labs, vals = [], []
        for nm in ("class", "distance_kpc", "log10_P", "eccentricity"):
            e = sp[nm]["edges"]
            for j, rr in enumerate(sp[nm]["ratio"]):
                short = {"class": "evolved" if j else "dwarf", "distance_kpc": f"d {e[j]:g}-{e[j + 1]:g} kpc",
                         "log10_P": f"log P {e[j]:g}-{e[j + 1]:g}", "eccentricity": f"e {e[j]:g}-{e[j + 1]:g}"}[nm]
                labs.append(short)
                vals.append(rr)
        ax.barh(np.arange(len(vals)), np.nan_to_num(vals), color=series_style(0, style)["color"])
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(labs, fontsize=7)
        ax.axvline(1, color="0.5", lw=1)
        apply_axes_style(ax, style, xlabel="(mock + spurious) / real, orbits G < 10 (noW)", ylabel="")
        ax.set_title("(a) G < 10 orbit split")
    ax = axs[0, 2]
    bc = 0.5 * (np.array(be[1:]) + np.array(be[:-1]))
    for i, t in enumerate(("Orbital", "AstroSpectroSB1")):
        ax.plot(bc, out["b"]["by_type"][t]["frac"], marker="o", label=t, color=series_style(i, style)["color"])
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="re-detected fraction")
    ax.set_title("(b) rung-1 re-detection (v2 + N2-u0)")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    cc = 0.5 * (np.array(ce[1:]) + np.array(ce[:-1]))
    ax.plot(cc, c_["mock_frac_gt1p4"], marker="o", label="mock single stars", color=series_style(0, style)["color"])
    ax.plot(cc, c_["dr3_frac_gt1p4"], marker="s", label="DR3 parent (all stars)", color=series_style(1, style)["color"])
    ax.plot(cc, c_["u0_singles_frac_gt1p4"], marker="^", ls=":", label="mock singles, u0 set (600 / 0.25 mag)", color=series_style(0, style)["color"])
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="fraction RUWE > 1.4")
    ax.set_yscale("log")
    ax.set_title("(c) RUWE tail")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    dc = 0.5 * (np.array(de[1:]) + np.array(de[:-1]))
    ax.plot(dc, d_["parent_pass_ipd"], marker="o", label="parent: IPD pass", color=series_style(0, style)["color"])
    ax.plot(dc, d_["real_pass_ipd"], marker="o", ls="--", label="real NSS orbits: IPD pass", color=series_style(0, style)["color"])
    ax.plot(dc, d_["parent_pass_cstar"], marker="s", label="parent: C* pass", color=series_style(1, style)["color"])
    ax.plot(dc, d_["real_pass_cstar"], marker="s", ls="--", label="real NSS orbits: C* pass", color=series_style(1, style)["color"])
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="pass fraction")
    ax.set_title("(d) Halbwachs cut pass rates")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    if z is not None:
        rin = z["rob"] >= 0
        ds = rin & (z["r_d"] > 1.0) & (z["r_lp"] < 2.6)
        sig = 10 ** z["r_cov_log10_a0_snr"]
        rng = np.random.default_rng(0)
        bk = np.flatnonzero(rin & ~ds)
        bk = rng.choice(bk, size=min(20000, bk.size), replace=False)
        ax.scatter(z["r_lp"][bk], sig[bk], s=2, alpha=0.2, color="0.6", label="all real C1 orbits (sample)")
        ax.scatter(z["r_lp"][ds], sig[ds], s=14, color=series_style(2, style)["color"], label=f"d > 1 kpc, log P < 2.6 (n={int(ds.sum())})")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel="log10 P (d)", ylabel="significance a0 / σ_a0")
        ax.set_title("(e) distant short-P real orbits")
        ax.legend(fontsize=8)
    fig.suptitle("#391 bright G < 10 excess and distant short-P cells (gens 33–38 coarse fit)")
    save_figure(fig, args.out_dir / "bright_excess.png", dpi=int(cfg.diagnostics.figure_dpi))
    print(json.dumps(out, indent=1, default=float)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
