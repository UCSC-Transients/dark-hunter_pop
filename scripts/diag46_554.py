#!/usr/bin/env python
"""#554 (#391 rung 3): diagnose the gens 42-46 v12 high-twin / high-spurious mode. Analysis only, at FIXED theta.

Reads refit-5 dumps (``rung3_refit5_391.py fit --dump-from``), the weight cache and the generation artifacts (truth +
outcome); runs no cascade and draws nothing. Sections (``--sections``):

* ``gens``    per-generation-subset estimators of the mock expectation per distance bin (deterministic-mixture weight
              re-expressed as the single-subset estimator w_S = w_mix * sum_all n_k q_k / sum_S n_k q_k), row-level errors.
* ``decomp``  mock vs real orbits per distance bin, split by G, M_G (observed, no extinction), BP-RP, CMD class, log P, e,
              log a0/sigma_a0, log plx/sigma, F2, RUWE, visibility periods, a0; mock truth splits (M1, q, log f, phase).
* ``stages``  cascade outcome per truth-distance bin for all draws with truth P <= 830 d (MdS17-prior mixture weights),
              orbital_failed_cuts broken down by the failing cut.
* ``cells``   per-fit-bin -lnL contributions (SAY, parent-row variance) of two dumps evaluated on the same data.
* ``twins``   mock twin-band (system dM in [-0.95, -0.55]) decomposition by q, flux ratio, primary dM, resolved flag.
* ``f2``      accepted-orbit F2 at G < 13, mock vs real, by class / M_G / d / P / a0-SNR.
* ``parent``  parent looks-single cut rates and BJ-vs-parallax distance-bin migration vs (G, d).

Every mock number carries Kish ESS (draw level) and the parent-row (clustered) relative MC error (spec §12.20).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

D_BINS = [0.0, 0.3, 0.7, 1.5, 6.0]
TWIN_D = [0.0, 0.1, 0.2, 0.5, 1.0, 3.0]
TWIN, SINGLE = (-0.95, -0.55), (-0.25, 0.25)
PMAX = 830.4


def bin_index(cols, edges):
    idx = np.zeros(cols[0].size, np.int64)
    ok = np.ones(cols[0].size, bool)
    for c, e in zip(cols, edges):
        j = np.searchsorted(e, c, side="right") - 1
        ok &= np.isfinite(c) & (j >= 0) & (j < e.size - 1)
        idx = idx * (e.size - 1) + np.clip(j, 0, e.size - 2)
    return np.where(ok, idx, -1)


def cell(w, rows, sel):
    """Weighted sum, parent-row clustered relative error, Kish ESS, row ESS and max row share for a draw subset."""
    ws, rs = w[sel], rows[sel]
    tot = float(ws.sum())
    if tot <= 0:
        return dict(sum=0.0, rel_err=np.nan, ess=0.0, ess_row=0.0, max_row_share=np.nan, n=int(sel.sum()))
    _, inv = np.unique(rs, return_inverse=True)
    rsum = np.bincount(inv, weights=ws)
    return dict(sum=tot, rel_err=float(np.sqrt(np.sum(rsum**2)) / tot), ess=float(tot**2 / np.sum(ws**2)),
                ess_row=float(tot**2 / np.sum(rsum**2)), max_row_share=float(rsum.max() / tot), n=int(sel.sum()))


def a0_from_ti(A, B, F, G):
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    return np.sqrt(u + np.sqrt(np.maximum(u**2 - v**2, 0.0)))


def load_mock(dump: Path, cache: Path, arts: list[Path], outer: dict, truth_keys=()):
    """Dump arrays + the matching cache/artifact rows (``rel`` reproduces r4.setup's selection)."""
    import h5py

    z = dict(np.load(dump, allow_pickle=True))
    zc = np.load(cache, allow_pickle=True)
    casc = np.asarray(zc["cascade"], float)
    g = np.asarray(zc["phot_g_mean_mag"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        o_cols = [g, 1.0 / casc[:, 0], np.log10(casc[:, 10]), casc[:, 14]]
        a_cols = [g, 1.0 / casc[:, 2]]
    ax = ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")
    in_o = np.asarray(zc["accepted"], bool) & (casc[:, 10] <= PMAX) & (bin_index(o_cols, [np.asarray(outer[a]) for a in ax]) >= 0)
    in_a = np.asarray(zc["published_acceleration"], bool) & (bin_index(a_cols, [np.asarray(outer["g_mag"]), np.asarray(outer["distance_kpc"])]) >= 0)
    rel = np.flatnonzero(in_o | in_a)
    if rel.size != z["w"].size or not np.array_equal(np.asarray(zc["generation"])[rel], z["gen"]) or not np.allclose(casc[rel], z["casc"]):
        raise SystemExit(f"{dump}: cannot reproduce the dump's draw selection from {cache}")
    z["rel"] = rel
    lqs = np.asarray(zc["lqs"], float)
    n = np.asarray(zc["n_per_gen"], float)
    z["L"] = lqs[:, rel] + np.log(n)[:, None]
    z["gens"] = np.asarray(zc["gens"])
    for k in ("m1_msun", "m2_msun", "period_days", "eccentricity"):
        z["c_" + k] = np.asarray(zc[k], float)[rel]
    if truth_keys:
        tr = {k: [] for k in truth_keys}
        di = []
        for a in arts:
            with h5py.File(a, "r") as h:
                di.append(h["truth/draw_index"][()])
                for k in truth_keys:
                    tr[k].append(h["truth"][k][()])
        di = np.concatenate(di)
        if not np.array_equal(di, np.asarray(zc["draw_index"])):
            raise SystemExit("artifact order does not match the cache")
        for k in truth_keys:
            z["t_" + k] = np.concatenate(tr[k])[rel]
    return z


def subset_factor(z, gens_sub):
    """w_S / w_mix for draws of generations in gens_sub (0 elsewhere)."""
    idx = [list(z["gens"]).index(g) for g in gens_sub]
    tot = np.logaddexp.reduce(z["L"], axis=0)
    sub = np.logaddexp.reduce(z["L"][idx], axis=0)
    return np.where(np.isin(z["gen"], gens_sub), np.exp(tot - sub), 0.0)


def fmt_cell(c):
    return f"{c['sum']:9.1f} ±{100 * c['rel_err']:5.1f}% ESS {c['ess']:6.1f} rowESS {c['ess_row']:6.1f} maxrow {c['max_row_share']:.2f}"


# ----------------------------------------------------------------------------------------------------------------------
def sec_gens(z, rep, out):
    """Q1(b): do the generation subsets agree on the mock expectation per distance bin?"""
    w, prow = z["w"], z["prow"]
    so, sa = z["ob"] >= 0, z["ab"] >= 0
    subsets = {"all": [int(g) for g in z["gens"]], "42-44": [42, 43, 44], "45": [45], "46": [46], "45-46": [45, 46]}
    subsets = {k: v for k, v in subsets.items() if all(g in list(z["gens"]) for g in v)}
    res = {}
    rep.append("== gens: single-subset estimators of the mock expectation (same theta), per measured-distance bin ==")
    for kind, s, d in (("orbits", so, z["m_d_o"]), ("accels", sa, z["m_d_a"])):
        rep.append(f"  {kind}: d bin | " + " | ".join(subsets))
        res[kind] = {}
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            row = []
            res[kind][f"{lo}-{hi}"] = {}
            for nm, gs in subsets.items():
                f = subset_factor(z, gs)
                c = cell(w * f, prow, s & (d >= lo) & (d < hi) & (f > 0))
                res[kind][f"{lo}-{hi}"][nm] = c
                row.append(f"{c['sum']:8.0f}±{100 * c['rel_err']:4.1f}% (ESS {c['ess']:5.0f})")
            rep.append(f"    {lo:>3}-{hi:<3} " + " | ".join(row))
        # chi2 of 42-44 vs 45 vs 46 against the all-gens estimate
        rep.append("")
    # real counts per d bin for reference
    rd = np.histogram(z["r_d"][z["rob"] >= 0], bins=D_BINS)[0]
    rad = np.histogram(z["ra_d"], bins=D_BINS)[0]
    res["real_orbits"], res["real_accels"] = rd.tolist(), rad.tolist()
    rep.append(f"  real orbits per d bin {rd.tolist()}, real accelerations {rad.tolist()}")
    # weight-tail diagnostics: share of the 0.7-1.5 kpc orbit expectation held by the top 1/5/20 draws, by generation
    m = so & (z["m_d_o"] >= 0.7) & (z["m_d_o"] < 1.5)
    ws = np.sort(w[m])[::-1]
    res["top_share_0p7_1p5_orbits"] = {k: float(ws[:k].sum() / ws.sum()) for k in (1, 5, 20)}
    gsh = {int(g): float(w[m & (z["gen"] == g)].sum() / w[m].sum()) for g in z["gens"]}
    ndr = {int(g): int((m & (z["gen"] == g)).sum()) for g in z["gens"]}
    res["gen_share_0p7_1p5_orbits"], res["gen_ndraws_0p7_1p5_orbits"] = gsh, ndr
    rep.append(f"  0.7-1.5 kpc orbits: top-1/5/20 draw share {res['top_share_0p7_1p5_orbits']}; weight share by gen {gsh}; draws by gen {ndr}")
    rep.append("")
    out["gens"] = res


def real_extra(z, real_input_columns: Path | None):
    """Observed real quantities aligned with the dump's real-orbit rows."""
    rin = z["rob"] >= 0
    r = {"in": rin}
    plx = 1.0 / z["r_d"]  # zero-point-corrected measured parallax (mas) as used for r_d
    r["M_G"] = z["r_g"] + 5 * np.log10(plx / 100.0)
    r["bp_rp"] = z["r_cov_bp_rp"]
    r["a0"] = a0_from_ti(z["r_x_A"], z["r_x_B"], z["r_x_F"], z["r_x_G"])
    r["log_a0_snr"] = z["r_cov_log10_a0_snr"]
    r["log_plx_snr"] = np.log10(z["r_x_parallax"] / z["r_x_parallax_error"])
    r["F2"] = z["r_x_goodness_of_fit"]
    r["ruwe"] = z["r_x_ruwe"]
    r["vis"] = np.full(rin.size, np.nan)
    if real_input_columns is not None:
        from darkhunter_pop import proposal_set as ps

        ic = ps.load_real_input_columns(real_input_columns)
        pos = {int(s): i for i, s in enumerate(np.asarray(ic["source_id"], np.int64))}
        ix = np.array([pos.get(int(s), -1) for s in z["r_sid"]])
        v = np.asarray(ic["visibility_periods_used"], float)
        r["vis"] = np.where(ix >= 0, v[np.maximum(ix, 0)], np.nan)
    return r


def mock_extra(z):
    c = z["casc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = {"M_G": z["m_g"] + 5 * np.log10(c[:, 0] / 100.0), "a0": c[:, 17], "log_a0_snr": np.log10(c[:, 17] / c[:, 18]),
             "log_plx_snr": np.log10(c[:, 0] / c[:, 1]), "F2": c[:, 21], "ruwe": c[:, 22], "vis": c[:, 19]}
    if "t_sys_bp_rp" in z:
        m["bp_rp"] = z["t_sys_bp_rp"]
    return m


SPLITS = {  # name: (mock key, real key, edges)
    "G": ("m_g", "r_g", [10, 12, 13, 14, 15, 16, 17, 19.5]),
    "M_G_obs": ("M_G", "M_G", [-3, 0, 1, 2, 3, 4, 5, 6, 8, 12]),
    "bp_rp_obs": ("bp_rp", "bp_rp", [-0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0]),
    "evolved": ("m_cls", "r_cls", [-0.5, 0.5, 1.5]),
    "log_P": ("m_lp", "r_lp", [0, 1.5, 2.0, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9193]),
    "e": ("m_e", "r_e", [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
    "log_a0_snr": ("log_a0_snr", "log_a0_snr", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.5]),
    "log_plx_snr": ("log_plx_snr", "log_plx_snr", [0, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 4]),
    "F2": ("F2", "F2", [-10, 0, 2, 4, 6, 10, 25]),
    "ruwe": ("ruwe", "ruwe", [0, 1.4, 2, 3, 5, 10, 100]),
    "vis": ("vis", "vis", [0, 14, 16, 18, 20, 22, 40]),
    "a0_mas": ("a0", "a0", [0, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 100]),
}


def sec_decomp(z, rep, out, real_input_columns):
    """Q1(a): mock vs real orbits per measured-distance bin, split by observables (and mock truth)."""
    w, prow = z["w"], z["prow"]
    so = z["ob"] >= 0
    me, re_ = mock_extra(z), real_extra(z, real_input_columns)
    rin = re_["in"]
    md, rd, spo = z["m_d_o"], z["r_d"], z["r_spur_per_orbit"]
    res = {}

    def get(src, key, mock):
        if mock:
            return me[key] if key in me else z[key]
        return re_[key] if key in re_ else z[key]

    rep.append("== decomp: orbits, mock (theta, no spurious) vs real, per measured-distance bin; cells: real | mock ±row-err ESS | mock/real | (mock+spur)/real ==")
    for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
        dm, dr = so & (md >= lo) & (md < hi), rin & (rd >= lo) & (rd < hi)
        key = f"{lo}-{hi}"
        res[key] = {}
        tot_m, tot_r = cell(w, prow, dm), int(dr.sum())
        rep.append(f"  d {key} kpc: real {tot_r}, mock {fmt_cell(tot_m)}, mock/real {tot_m['sum'] / max(tot_r, 1):.3f}, spurious {spo[dr].sum():.0f}")
        for nm, (mk, rk, ed) in SPLITS.items():
            xm, xr = get(z, mk, True), get(z, rk, False)
            if xm is None or xr is None:
                continue
            rows = []
            for a, b in zip(ed[:-1], ed[1:]):
                cm = cell(w, prow, dm & (xm >= a) & (xm < b))
                sr = dr & (xr >= a) & (xr < b)
                nr = int(sr.sum())
                sp = float(spo[sr].sum())
                rows.append(dict(lo=a, hi=b, real=nr, spur=sp, **{"mock_" + k: v for k, v in cm.items()},
                                 ratio=cm["sum"] / nr if nr else np.nan, ratio_spur=(cm["sum"] + sp) / nr if nr else np.nan))
            res[key][nm] = rows
            rep.append(f"    {nm}: " + "; ".join(f"[{r['lo']:g},{r['hi']:g}) {r['real']} | {r['mock_sum']:.0f}±{100 * r['mock_rel_err']:.0f}% E{r['mock_ess']:.0f} | "
                                                 f"{r['ratio']:.2f} | {r['ratio_spur']:.2f}" for r in rows))
        rep.append("")
    # mock-only truth splits in the 0.7-1.5 kpc bin vs the neighbouring bins (where are the mock orbits coming from?)
    if "t_parallax_mas" in z:
        q = z["c_m2_msun"] / z["c_m1_msun"]
        tsp = {"M1_true": (z["c_m1_msun"], [0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 10]), "q": (q, [0, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.01]),
               "log_f": (z["t_log10_flux_ratio"], [-30, -3, -2, -1, -0.5, -0.2, 0.5]),
               "dark": (z["t_is_dark"].astype(float), [-0.5, 0.5, 1.5]), "iso_phase": (z["t_iso_phase"], [-0.5, 0.5, 1.5, 2.5, 3.5, 9]),
               "d_true": (1.0 / z["t_parallax_mas"], [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 6.0])}
        res["mock_truth"] = {}
        rep.append("  mock-only truth splits of detected orbits (share of the d-bin expectation; ESS):")
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            dm = so & (md >= lo) & (md < hi)
            tot = w[dm].sum()
            res["mock_truth"][f"{lo}-{hi}"] = {}
            for nm, (x, ed) in tsp.items():
                cs = [cell(w, prow, dm & (x >= a) & (x < b)) for a, b in zip(ed[:-1], ed[1:])]
                res["mock_truth"][f"{lo}-{hi}"][nm] = [dict(lo=a, hi=b, share=c["sum"] / tot, **c) for (a, b), c in zip(zip(ed[:-1], ed[1:]), cs)]
                rep.append(f"    d {lo}-{hi} {nm}: " + " ".join(f"[{a:g},{b:g}) {c['sum'] / tot:.3f}(E{c['ess']:.0f})" for (a, b), c in zip(zip(ed[:-1], ed[1:]), cs)))
        rep.append("")
    out["decomp"] = res


def sec_stages(cache: Path, arts: list[Path], parent_scale_cut: float, rep, out, outer: dict):
    """Q1(c): cascade outcome vs truth distance for every draw with truth P <= 830 d (MdS17-prior mixture weights)."""
    import h5py

    from darkhunter_pop import proposal_set as ps

    zc = np.load(cache, allow_pickle=True)
    w0 = ps.importance_weights(np.asarray(zc["ll"], float), list(np.asarray(zc["lqs"], float)), [int(x) for x in zc["n_per_gen"]],
                               scale_to_full=float(zc["scale"]))
    st, plx_t, gsys, prow = [], [], [], []
    for a in arts:
        with h5py.File(a, "r") as h:
            st.append(h["outcome/solution_type"][()].astype(str))
            plx_t.append(h["truth/parallax_mas"][()])
            gsys.append(h["truth/phot_g_mean_mag"][()])
            prow.append(h["truth/parent_row"][()])
    st, plx_t, gsys, prow = np.concatenate(st), np.concatenate(plx_t), np.concatenate(gsys), np.concatenate(prow)
    casc = np.asarray(zc["cascade"], float)
    P = np.asarray(zc["period_days"], float)
    dt = 1.0 / plx_t
    # which orbital cut fails (El-Badry+2024 Eqs. 18, 20-22) for orbital_failed_cuts with a valid fit
    with np.errstate(divide="ignore", invalid="ignore"):
        a0s = casc[:, 17] / casc[:, 18]
        pls = casc[:, 0] / casc[:, 1]
        Pf = casc[:, 10]
        c_sig = a0s < np.maximum(5.0, 158.0 / np.sqrt(Pf))
        c_plx = pls < 20000.0 / Pf
        c_ecc = casc[:, 15] > 0.079 * np.log(Pf) - 0.244
        c_f2 = casc[:, 21] > 25.0
        c_pmax = Pf > PMAX
    ofc = st == "orbital_failed_cuts"
    cats = {"not_in_parent": st == "not_in_parent", "five_parameter": st == "five_parameter",
            "accel_7/9 (any)": np.isin(st, ["seven_parameter", "nine_parameter"]), "accel published": np.asarray(zc["published_acceleration"], bool),
            "orbit accepted": np.asarray(zc["accepted"], bool), "orbit accepted P<=830": np.asarray(zc["accepted"], bool) & (Pf <= PMAX),
            "failed: a0/sig": ofc & c_sig, "failed: plx/sig": ofc & c_plx, "failed: sigma_e": ofc & c_ecc, "failed: F2": ofc & c_f2,
            "failed: fitted P>830": ofc & c_pmax, "failed (any)": ofc, "insufficient_visibility": st == "insufficient_visibility"}
    res = {}
    rep.append("== stages: cascade outcome per TRUTH distance, truth P <= 830 d, system G 10-19 (MdS17-prior mixture weights; share of systems; ESS) ==")
    base = (P <= PMAX) & (gsys >= 10) & (gsys < 19)
    for gl, gh in ((10, 13), (13, 16), (16, 19)):
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            m = base & (gsys >= gl) & (gsys < gh) & (dt >= lo) & (dt < hi)
            tot = w0[m].sum()
            if tot <= 0:
                continue
            k = f"G{gl}-{gh} d{lo}-{hi}"
            res[k] = {c: dict(share=float(w0[m & s].sum() / tot), ess=cell(w0, prow, m & s)["ess"]) for c, s in cats.items()}
            rep.append(f"  {k}: " + "; ".join(f"{c} {v['share']:.4f}(E{v['ess']:.0f})" for c, v in res[k].items()))
    rep.append("")
    out["stages"] = res


def sec_cells(dumps: dict, rep, out):
    """Q2: per-bin -lnL at two theta on the same data; which cells drive the difference; their ESS."""
    from scipy.special import gammaln

    from darkhunter_pop.rung3_model import clustered_bin_variance, row_bin_groups

    def say_terms(k, mu, var):
        mu = np.maximum(mu, 1e-300)
        out_ = k * np.log(mu) - mu - gammaln(k + 1)
        use = var > 1e-12 * mu**2
        m, v, kk = mu[use], var[use], k[use]
        a = m * m / v + 1.0
        b = m / v
        out_[use] = a * np.log(b) + gammaln(kk + a) - gammaln(kk + 1) - (kk + a) * np.log1p(b) - gammaln(a)
        return out_

    res = {}
    terms = {}
    for nm, path in dumps.items():
        z = np.load(path, allow_pickle=True)
        w, ob, ab, prow = z["w"], z["ob"], z["ab"], z["prow"]
        n_ob, n_ab = z["k_o"].size, z["k_a"].size
        v_o = clustered_bin_variance(w, row_bin_groups(ob, prow), n_ob) * ~z["empty_o"]
        v_a = clustered_bin_variance(w, row_bin_groups(ab, prow), n_ab) * (z["mu_a"] > 0)
        th = z["theta"]
        # spurious per bin: recover s = N_s * T + floor from the dump's per-real-orbit spurious
        s = np.bincount(z["rob"][z["rob"] >= 0], weights=z["r_spur_per_orbit"][z["rob"] >= 0], minlength=n_ob)
        s = np.where(z["k_o"] > 0, s, np.where(z["empty_o"], 1.0, 0.0))
        mu_a = z["mu_a"] + np.where(z["mu_a"] > 0, 0.0, 1.0)  # empty-bin floor (acceleration bins with no mock: 1.0)
        lo = say_terms(z["k_o"], z["mu_o"] + s, v_o)
        la = say_terms(z["k_a"], mu_a, v_a)
        ess_o = np.where(v_o > 0, z["mu_o"] ** 2 / np.maximum(v_o, 1e-300), 0.0)
        ess_a = np.where(v_a > 0, z["mu_a"] ** 2 / np.maximum(v_a, 1e-300), 0.0)
        terms[nm] = dict(lo=lo, la=la, mu_o=z["mu_o"], s=s, k_o=z["k_o"], mu_a=z["mu_a"], k_a=z["k_a"], ess_o=ess_o, ess_a=ess_a, theta=th,
                         shp=z["shp"], rel_o=np.sqrt(v_o) / np.maximum(z["mu_o"], 1e-300), rel_a=np.sqrt(v_a) / np.maximum(z["mu_a"], 1e-300))
        prior_term = 0.5 * ((th[-1] - 21434.0) / 1833.0) ** 2  # §12.16 N_s prior (v12 report: N(21434, 1833))
        res[nm] = dict(sum_ll_orbit=float(lo.sum()), sum_ll_accel=float(la.sum()), N_s=float(th[-1]), theta=th.tolist(),
                       nll_total=float(-(lo.sum() + la.sum()) + prior_term), nll_prior=float(prior_term))
    names = list(dumps)
    rep.append("== cells: SAY log-likelihood per fit bin (parent-row variance), " + " vs ".join(names) + " ==")
    for nm in names:
        rep.append(f"  {nm}: sum ll orbits {res[nm]['sum_ll_orbit']:.1f}, accels {res[nm]['sum_ll_accel']:.1f}, N_s {res[nm]['N_s']:.0f} "
                   f"(prior term {res[nm]['nll_prior']:.1f}); total -lnL {res[nm]['nll_total']:.1f}")
    if len(names) >= 2:
        A, B = terms[names[0]], terms[names[1]]
        shp = tuple(int(x) for x in A["shp"])
        ge, de, pe, ee = ["10-13", "13-16", "16-19.5"], ["0-1", "1-6"], ["<2.6", "2.6-2.92"], ["<0.3", ">=0.3"]
        dl = B["lo"] - A["lo"]  # positive: second theta fits this bin better
        order = np.argsort(-np.abs(dl))
        rows = []
        for b in order[:20]:
            i = np.unravel_index(b, shp)
            lab = f"G {ge[i[0]]} d {de[i[1]]} lP {pe[i[2]]} e {ee[i[3]]} {'evo' if i[4] else 'dwarf'}"
            rows.append(dict(bin=int(b), label=lab, k=float(A["k_o"][b]), dll=float(dl[b]),
                             **{f"{nm}_{q}": float(terms[nm][q][b]) for nm in names[:2] for q in ("mu_o", "s", "ess_o", "rel_o")}))
            rep.append(f"  {lab:48s} k {A['k_o'][b]:7.0f} | Δll {dl[b]:+7.1f} | {names[0]}: mock {A['mu_o'][b]:7.0f} spur {A['s'][b]:6.0f} ESS {A['ess_o'][b]:5.1f} "
                       f"(±{100 * A['rel_o'][b]:.0f}%) | {names[1]}: mock {B['mu_o'][b]:7.0f} spur {B['s'][b]:6.0f} ESS {B['ess_o'][b]:5.1f} (±{100 * B['rel_o'][b]:.0f}%)")
        dla = B["la"] - A["la"]
        ga, da = ["10-13", "13-16", "16-19.5"], ["0-1", "1-6"]
        rep.append("  accelerations (G, d, class):")
        arows = []
        for b in np.argsort(-np.abs(dla)):
            i = np.unravel_index(b, (3, 2, 2))
            lab = f"G {ga[i[0]]} d {da[i[1]]} {'evo' if i[2] else 'dwarf'}"
            arows.append(dict(bin=int(b), label=lab, k=float(A["k_a"][b]), dll=float(dla[b]), mu_A=float(A["mu_a"][b]), mu_B=float(B["mu_a"][b]),
                              ess_A=float(A["ess_a"][b]), ess_B=float(B["ess_a"][b])))
            rep.append(f"    {lab:28s} k {A['k_a'][b]:7.0f} | Δll {dla[b]:+7.1f} | mock {A['mu_a'][b]:7.0f} (ESS {A['ess_a'][b]:.0f}) vs {B['mu_a'][b]:7.0f} (ESS {B['ess_a'][b]:.0f})")
        # aggregate Δll by low/high ESS
        low = (np.minimum(A["ess_o"], B["ess_o"]) < 30) & (A["k_o"] > 0)
        rep.append(f"  Δll orbits total {dl.sum():+.1f} (bins with min ESS < 30: {dl[low].sum():+.1f} over {int(low.sum())} bins; others {dl[~low].sum():+.1f}); "
                   f"accels {dla.sum():+.1f}")
        res["delta"] = dict(orbit_rows=rows, accel_rows=arows, dll_orbits=float(dl.sum()), dll_accels=float(dla.sum()),
                            dll_lowess=float(dl[low].sum()), dll_highess=float(dl[~low].sum()))
        res["_terms"] = {nm: {k: np.asarray(v).tolist() for k, v in t.items()} for nm, t in terms.items()}
    rep.append("")
    out["cells"] = res


def sec_twins(z, ridge, rep, out, label):
    """Q3: decompose the mock twin band (system CMD) per truth distance."""
    w, prow = z["w"], z["prow"]
    sc0, smg0 = z["t_sys_colour0"], z["t_sys_mg0"]
    pc0, pmg0 = z["t_primary_colour0"], z["t_primary_mg0"]
    rmag = lambda c: np.interp(c, ridge["colour"], ridge["mag"], left=np.nan, right=np.nan)  # noqa: E731
    dm_sys, dm_pri = smg0 - rmag(sc0), pmg0 - rmag(pc0)
    q = z["c_m2_msun"] / z["c_m1_msun"]
    lf = z["t_log10_flux_ratio"]
    dt = 1.0 / z["t_parallax_mas"]
    res = {}
    rep.append(f"== twins [{label}]: mock system-CMD twin band ΔM [-0.95,-0.55) vs single band [-0.25,0.25), C0 0.85-2.0, by TRUTH d ==")
    for kind, s in (("orbits", z["ob"] >= 0), ("accels", z["ab"] >= 0)):
        res[kind] = {}
        for lo, hi in zip(TWIN_D[:-1], TWIN_D[1:]):
            m = s & (sc0 >= 0.85) & (sc0 <= 2.0) & (dt >= lo) & (dt < hi)
            tw = m & (dm_sys >= TWIN[0]) & (dm_sys < TWIN[1])
            si = m & (dm_sys >= SINGLE[0]) & (dm_sys < SINGLE[1])
            ct, cs = cell(w, prow, tw), cell(w, prow, si)
            ratio = ct["sum"] / cs["sum"] if cs["sum"] > 0 else np.nan
            rel = np.sqrt(ct["rel_err"] ** 2 + cs["rel_err"] ** 2) if cs["sum"] > 0 and ct["sum"] > 0 else np.nan
            dec = {}
            if ct["sum"] > 0:
                tot = ct["sum"]
                dec["q"] = {f"[{a},{b})": float(w[tw & (q >= a) & (q < b)].sum() / tot) for a, b in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))}
                dec["primary_dM"] = {f"[{a},{b})": float(w[tw & (dm_pri >= a) & (dm_pri < b)].sum() / tot)
                                     for a, b in ((-9, -0.95), (-0.95, -0.55), (-0.55, -0.25), (-0.25, 0.25), (0.25, 9))}
                dec["primary_dM_nan"] = float(w[tw & ~np.isfinite(dm_pri)].sum() / tot)
                dec["log_f"] = {f"[{a},{b})": float(w[tw & (lf >= a) & (lf < b)].sum() / tot) for a, b in ((-30, -1), (-1, -0.3), (-0.3, -0.1), (-0.1, 0.5))}
                dec["resolved"] = float(w[tw & z["t_resolved_companion"]].sum() / tot)
                dec["primary_C0<1.05 (row dM cut not applied)"] = float(w[tw & (pc0 < 1.05)].sum() / tot)
                dec["top_row_share"] = ct["max_row_share"]
            res[kind][f"{lo}-{hi}"] = dict(twin=ct, single=cs, ratio=ratio, ratio_rel_err=rel, twin_decomp=dec)
            rep.append(f"  {kind} d {lo}-{hi}: twin/single {ratio:.2f} ±{100 * rel:.0f}% (twin ESS {ct['ess']:.1f}, rowESS {ct['ess_row']:.1f}, max row {ct['max_row_share']:.2f}; "
                       f"single ESS {cs['ess']:.1f})")
            if dec:
                rep.append(f"      twin-band composition: q {dec['q']}; primary ΔM {dec['primary_dM']} (nan {dec['primary_dM_nan']:.2f}); log f {dec['log_f']}; "
                           f"resolved {dec['resolved']:.3f}; primary C0<1.05 {dec['primary_C0<1.05 (row dM cut not applied)']:.2f}")
    rep.append("")
    out[f"twins_{label}"] = res


def sec_fscatter(cache: Path, arts: list[Path], rep, out):
    """Q3/Q2: does the 0.1 dex flux-ratio scatter (MP-Q13/Q40) give near-twins a photocentre wobble they would not have?"""
    import h5py

    from darkhunter_pop import proposal_set as ps

    zc = np.load(cache, allow_pickle=True)
    w0 = ps.importance_weights(np.asarray(zc["ll"], float), list(np.asarray(zc["lqs"], float)), [int(x) for x in zc["n_per_gen"]],
                               scale_to_full=float(zc["scale"]))
    lf, plx, dark, prow = [], [], [], []
    for a in arts:
        with h5py.File(a, "r") as h:
            lf.append(h["truth/log10_flux_ratio"][()])
            plx.append(h["truth/parallax_mas"][()])
            dark.append(h["truth/is_dark"][()])
            prow.append(h["truth/parent_row"][()])
    lf, plx, dark, prow = (np.concatenate(x) for x in (lf, plx, dark, prow))
    m1, m2 = np.asarray(zc["m1_msun"], float), np.asarray(zc["m2_msun"], float)
    q = m2 / m1
    P = np.asarray(zc["period_days"], float)
    g = np.asarray(zc["phot_g_mean_mag"], float)
    acc = np.asarray(zc["accepted"], bool) & (np.asarray(zc["cascade"], float)[:, 10] <= PMAX)
    dt = 1.0 / plx
    base = ~dark & (P <= PMAX) & (g >= 10) & (g < 16) & (dt < 0.7) & np.isfinite(lf)
    # relation centre per (q, M1) cell
    qe, me = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]), np.array([0.5, 0.7, 0.9, 1.1, 1.4])
    iq, im = np.digitize(q, qe) - 1, np.digitize(m1, me) - 1
    cen = np.full(q.size, np.nan)
    for i in range(qe.size - 1):
        for j in range(me.size - 1):
            m = base & (iq == i) & (im == j)
            if m.sum() > 50:  # TARGET-weighted median (the target is N(centre, 0.1 dex) in log f)
                o = np.argsort(lf[m])
                cw = np.cumsum(w0[m][o]) / w0[m].sum()
                cen[m] = lf[m][o][np.searchsorted(cw, 0.5)]
    dlf = lf - cen
    f, fc = 10**lf, 10**cen
    wob, wob_c = np.abs(q / (1 + q) - f / (1 + f)), np.abs(q / (1 + q) - fc / (1 + fc))
    res = {}
    rep.append("== fscatter: luminous companions, truth P <= 830 d, G 10-16, truth d < 0.7 kpc (MdS17-prior mixture weights) ==")
    for (ql, qh) in ((0.5, 0.8), (0.8, 0.95), (0.95, 1.0001)):
        sel = base & (q >= ql) & (q < qh) & np.isfinite(dlf)
        rows = []
        for a, b in ((-1, -0.2), (-0.2, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 1)):
            m = sel & (dlf >= a) & (dlf < b)
            ca, cd = w0[m].sum(), w0[m & acc].sum()
            rows.append(dict(lo=a, hi=b, share_all=float(ca / w0[sel].sum()), share_all_ess=cell(w0, prow, m)["ess"], det_rate=float(cd / ca) if ca > 0 else np.nan,
                             share_detected=float(cd / max(w0[sel & acc].sum(), 1e-300)), ess_det=cell(w0, prow, m & acc)["ess"],
                             wobble_med=float(np.median(wob[m])) if m.any() else np.nan, wobble_centre_med=float(np.median(wob_c[m])) if m.any() else np.nan))
        res[f"q{ql}-{qh}"] = rows
        rep.append(f"  q [{ql},{min(qh, 1):g}): detection rate and share of detected orbits by Δlog f = log f − centre (wobble |q/(1+q) − f/(1+f)| median, vs at the centre):")
        rep += [f"    Δlog f [{r['lo']:+.1f},{r['hi']:+.1f}): share of systems {r['share_all']:.3f} (ESS {r['share_all_ess']:.0f}), det rate {r['det_rate']:.4f}, share of detected {r['share_detected']:.3f} "
                f"(ESS {r['ess_det']:.0f}), wobble {r['wobble_med']:.3f} vs {r['wobble_centre_med']:.3f}" for r in rows]
    rep.append("")
    out["fscatter"] = res


def sec_amrf(dumps: dict, rep, out):
    """Q1/Q2: photocentre-orbit size A = (a0/plx) / P_yr^(2/3) [AU yr^-2/3] = M_tot^(1/3) (q/(1+q) - f/(1+f)), mock vs real by d."""
    ed = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 3]
    res = {}
    for nm, path in dumps.items():
        z = np.load(path, allow_pickle=True)
        so, w, c = z["ob"] >= 0, z["w"], z["casc"]
        with np.errstate(divide="ignore", invalid="ignore"):
            Am = (c[:, 17] / c[:, 0]) / (c[:, 10] / 365.25) ** (2 / 3)
        rin = z["rob"] >= 0
        rplx = 1 / z["r_d"]
        Ar = (a0_from_ti(z["r_x_A"], z["r_x_B"], z["r_x_F"], z["r_x_G"]) / rplx) / (z["r_x_period"] / 365.25) ** (2 / 3)
        res[nm] = {}
        rep.append(f"== amrf [{nm}]: orbits per A_obs bin {ed}; real | mock (ESS) | mock/real ==")
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            dm, dr = so & (z["m_d_o"] >= lo) & (z["m_d_o"] < hi), rin & (z["r_d"] >= lo) & (z["r_d"] < hi)
            rows = []
            for a, b in zip(ed[:-1], ed[1:]):
                cm = cell(w, z["prow"], dm & (Am >= a) & (Am < b))
                nr = int((dr & (Ar >= a) & (Ar < b)).sum())
                rows.append(dict(lo=a, hi=b, real=nr, mock=cm["sum"], ess=cm["ess"], rel_err=cm["rel_err"]))
            res[nm][f"{lo}-{hi}"] = rows
            rep.append(f"  d {lo}-{hi}: " + "; ".join(f"[{r['lo']:g},{r['hi']:g}) {r['real']} | {r['mock']:.0f} (E{r['ess']:.0f}) | {r['mock'] / max(r['real'], 1):.2f}" for r in rows))
        rep.append("")
    out["amrf"] = res


def sec_heavy(z, alt, rep, out, nmax=8):
    """Q2: the draws that dominate the cells driving the mode switch, and their weight at both theta."""
    w, wa = z["w"], alt["w"]
    q = z["c_m2_msun"] / z["c_m1_msun"]
    c = z["casc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        Am = (c[:, 17] / c[:, 0]) / (c[:, 10] / 365.25) ** (2 / 3)
    shp = tuple(int(x) for x in z["shp"])
    ge, de, pe, ee = ["10-13", "13-16", "16-19.5"], ["0-1", "1-6"], ["<2.6", "2.6-2.92"], ["<0.3", ">=0.3"]
    res = {}
    rep.append("== heavy: top draws per orbit fit bin (share of the bin at the primary theta), with the alt-theta weight ==")
    for b in range(int(np.prod(shp))):
        m = z["ob"] == b
        if not m.any() or w[m].sum() <= 0:
            continue
        tot, tota = w[m].sum(), wa[m].sum()
        idx = np.flatnonzero(m)[np.argsort(-w[m])][:nmax]
        if w[idx[0]] / tot < 0.2:  # only bins with a dominant draw
            continue
        i = np.unravel_index(b, shp)
        lab = f"G {ge[i[0]]} d {de[i[1]]} lP {pe[i[2]]} e {ee[i[3]]} {'evo' if i[4] else 'dwarf'}"
        rows = [dict(gen=int(z["gen"][j]), share=float(w[j] / tot), share_alt=float(wa[j] / tota) if tota > 0 else np.nan, w=float(w[j]), w_alt=float(wa[j]),
                     m1=float(z["c_m1_msun"][j]), q=float(q[j]), log_f=float(z["t_log10_flux_ratio"][j]), P=float(z["c_period_days"][j]), e=float(z["c_eccentricity"][j]),
                     G=float(z["m_g"][j]), d=float(z["m_d_o"][j]), A=float(Am[j]), evo=float(z["m_cls"][j]), phase=float(z["t_iso_phase"][j])) for j in idx[:3]]
        res[lab] = dict(k=float(z["k_o"][b]), mu=float(tot), mu_alt=float(tota), top=rows)
        rep.append(f"  {lab} (k {z['k_o'][b]:.0f}, mock {tot:.0f} / alt {tota:.0f}): " + "; ".join(
            f"gen{r['gen']} share {r['share']:.2f} (alt {r['share_alt']:.2f}) M1 {r['m1']:.2f} q {r['q']:.2f} logf {r['log_f']:+.2f} P {r['P']:.0f} e {r['e']:.2f} "
            f"G {r['G']:.1f} d {r['d']:.2f} A {r['A']:.2f} evo {r['evo']:.0f} phase {r['phase']:.0f}" for r in rows))
    # the A < 0.1 draws at 0.3-0.7 kpc
    m = (z["ob"] >= 0) & (Am < 0.1) & (z["m_d_o"] >= 0.3) & (z["m_d_o"] < 0.7)
    idx = np.flatnonzero(m)[np.argsort(-w[m])][:5]
    res["A<0.1, d 0.3-0.7"] = [dict(gen=int(z["gen"][j]), w=float(w[j]), w_alt=float(wa[j]), m1=float(z["c_m1_msun"][j]), q=float(q[j]),
                                    log_f=float(z["t_log10_flux_ratio"][j]), P=float(z["c_period_days"][j]), G=float(z["m_g"][j]), bin=int(z["ob"][j])) for j in idx]
    rep.append("  A < 0.1 at d 0.3-0.7 kpc (sum %.0f, alt %.0f): " % (w[m].sum(), wa[m].sum()) + "; ".join(
        f"gen{r['gen']} w {r['w']:.0f} (alt {r['w_alt']:.0f}) M1 {r['m1']:.2f} q {r['q']:.3f} logf {r['log_f']:+.2f} P {r['P']:.0f} G {r['G']:.1f} bin {r['bin']}"
        for r in res["A<0.1, d 0.3-0.7"]))
    # weight concentration: share of the total orbit expectation in q >= 0.95 draws, and its ESS, at both theta
    for nm, ww in (("primary", w), ("alt", wa)):
        so = z["ob"] >= 0
        ct = cell(ww, z["prow"], so & (q >= 0.95))
        res[f"twin_share_{nm}"] = dict(share=ct["sum"] / ww[so].sum(), **ct)
        rep.append(f"  q >= 0.95 share of all mock orbits [{nm}]: {ct['sum'] / ww[so].sum():.3f} (ESS {ct['ess']:.1f}, max row {ct['max_row_share']:.2f})")
    rep.append("")
    out["heavy"] = res


def sec_lsdeplete(par, cache: Path, arts: list[Path], rep, out):
    """Q1(d): model-predicted looks-single depletion of the parent's primaries vs the observed parent attrition.

    The parent keeps rows that look single (§12.33); the mock gives every kept row MdS17 companions with no
    1/P(looks single) weight. Under the model, the fraction of all systems that a companion makes look non-single is
    D = sum(w0 * nonls) / (N_rows * scale) per row cell (w0: MdS17-prior mixture weights; nonls: a published orbit or
    acceleration, or system RUWE >= 1.25, or system dM < -0.4 at C0 >= 1.05). Missing primaries then scale every mock
    binary count by about (1 - D)."""
    import h5py

    from darkhunter_pop import proposal_set as ps

    zc = np.load(cache, allow_pickle=True)
    w0 = ps.importance_weights(np.asarray(zc["ll"], float), list(np.asarray(zc["lqs"], float)), [int(x) for x in zc["n_per_gen"]],
                               scale_to_full=float(zc["scale"]))
    st, prow, smg, sc0, sev = [], [], [], [], []
    for a in arts:
        with h5py.File(a, "r") as h:
            sev.append(h["truth/sys_evolved"][()])
            st.append(h["outcome/solution_type"][()].astype(str))
            prow.append(h["truth/parent_row"][()])
            smg.append(h["truth/sys_mg0"][()])
            sc0.append(h["truth/sys_colour0"][()])
    st, prow, smg, sc0, sev = (np.concatenate(x) for x in (st, prow, smg, sc0, sev))
    casc = np.asarray(zc["cascade"], float)
    rd = par.meta["ms_ridge"]
    dm = smg - np.interp(sc0, rd["colour"], rd["mag"], left=np.nan, right=np.nan)
    later = np.isin(st, ["seven_parameter", "nine_parameter", "orbital_failed_cuts", "twelve_parameter_orbital"])
    # 5-parameter outcome: res[1] is the RUWE; later stages had RUWE >= ruwe_min (1.4) > 1.25; insufficient visibility: unknown (kept single)
    ruwe5 = np.where(st == "five_parameter", casc[:, 1], np.where(later, np.inf, 0.0))
    nonls = (np.asarray(zc["accepted"], bool) | np.asarray(zc["published_acceleration"], bool) | (ruwe5 >= 1.25)
             | (~sev & (sc0 >= 1.05) & (sc0 <= 2.75) & (dm < -0.4)))
    why = {"published": np.asarray(zc["accepted"], bool) | np.asarray(zc["published_acceleration"], bool), "5-par RUWE>=1.25": (st == "five_parameter") & (ruwe5 >= 1.25),
           "later cascade stage, unpublished": later & ~(np.asarray(zc["accepted"], bool) | np.asarray(zc["published_acceleration"], bool)),
           "dM<-0.4": ~sev & (sc0 >= 1.05) & (sc0 <= 2.75) & (dm < -0.4)}
    valid = st != "not_in_parent"
    g_row = np.asarray(par.columns["phot_g_mean_mag"], float)
    d_row = 1.0 / par.truth_parallax_mas
    usable = par.usable & (par.supplement is None or True)
    if par.supplement is not None:
        usable = usable & ~par.supplement
    fl = par.flags
    pre = np.logical_and.reduce([v for k, v in fl.items() if not k.startswith("single_")])
    if par.supplement is not None:
        pre = pre & ~par.supplement
    scale = float(zc["scale"])
    res = {}
    rep.append("== lsdeplete: model fraction D of systems made non-single by a companion (parent rows, G >= 11 original stratum) vs observed attrition ==")
    for gl, gh in ((11, 13), (13, 16), (16, 19.5)):
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            rows_c = usable & (g_row >= gl) & (g_row < gh) & (d_row >= lo) & (d_row < hi)
            n_rows = int(rows_c.sum())
            if n_rows < 20:
                continue
            m = valid & rows_c[prow]
            B = float(w0[m].sum() / (n_rows * scale))
            D = float(w0[m & nonls].sum() / (n_rows * scale))
            pc = pre & (g_row >= gl) & (g_row < gh) & (d_row >= lo) & (d_row < hi)
            att = float(1 - par.usable[pc].mean()) if pc.any() else np.nan
            c = cell(w0, prow, m & nonls)
            k = f"G{gl}-{gh} d{lo}-{hi}"
            res[k] = dict(n_rows=n_rows, companions_per_row=B, D=D, D_rel_err=c["rel_err"], D_ess=c["ess"], observed_attrition=att,
                          D_by_reason={r_: float(w0[m & v_].sum() / (n_rows * scale)) for r_, v_ in why.items()})
            rep.append(f"  {k}: rows {n_rows}, MdS17 companions/row {B:.3f}, model D {D:.3f} (±{100 * c['rel_err']:.0f}%, ESS {c['ess']:.0f}); "
                       f"observed looks-single attrition {att:.3f}; D by reason " + ", ".join(f"{r_} {v_:.3f}" for r_, v_ in res[k]["D_by_reason"].items()))
    rep.append("")
    out["lsdeplete"] = res


def sec_f2(z, rep, out):
    """Q4: accepted-orbit F2 at G < 13, mock vs real, by population."""
    w, prow = z["w"], z["prow"]
    so = (z["ob"] >= 0) & (z["m_g"] < 13)
    rin = (z["rob"] >= 0) & (z["r_g"] < 13)
    me = mock_extra(z)
    re_ = real_extra(z, None)

    def wmed(x, ww):
        o = np.argsort(x)
        c = np.cumsum(ww[o]) / ww.sum()
        return float(x[o][np.searchsorted(c, 0.5)])

    def wfrac(x, ww, t):
        return float(ww[x > t].sum() / ww.sum())

    groups = {"all": (np.ones(z["w"].size, bool), np.ones(z["r_g"].size, bool)),
              "dwarf": (z["m_cls"] == 0, z["r_cls"] == 0), "evolved": (z["m_cls"] == 1, z["r_cls"] == 1)}
    for a, b in ((-5, 1), (1, 3), (3, 4), (4, 5), (5, 12)):
        groups[f"M_G {a}..{b}"] = ((me["M_G"] >= a) & (me["M_G"] < b), (re_["M_G"] >= a) & (re_["M_G"] < b))
    for a, b in ((10, 11), (11, 12), (12, 13)):
        groups[f"G {a}-{b}"] = ((z["m_g"] >= a) & (z["m_g"] < b), (z["r_g"] >= a) & (z["r_g"] < b))
    for a, b in ((0.6, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 4)):
        groups[f"log a0/sig {a}-{b}"] = ((me["log_a0_snr"] >= a) & (me["log_a0_snr"] < b), (re_["log_a0_snr"] >= a) & (re_["log_a0_snr"] < b))
    for a, b in ((0, 0.3), (0.3, 1.0), (1.0, 6)):
        groups[f"d {a}-{b}"] = ((z["m_d_o"] >= a) & (z["m_d_o"] < b), (z["r_d"] >= a) & (z["r_d"] < b))
    for a, b in ((0, 2.3), (2.3, 2.6), (2.6, 2.92)):
        groups[f"logP {a}-{b}"] = ((z["m_lp"] >= a) & (z["m_lp"] < b), (z["r_lp"] >= a) & (z["r_lp"] < b))
    res = {}
    rep.append("== f2: accepted-orbit F2 at G < 13 (mock weighted at theta, no spurious; real C1 orbits) ==")
    tot_m = w[so].sum()
    for nm, (mm, rr) in groups.items():
        ms, rs = so & mm, rin & rr
        if not ms.any() or not rs.any():
            continue
        c = cell(w, prow, ms)
        res[nm] = dict(mock_median=wmed(me["F2"][ms], w[ms]), real_median=float(np.nanmedian(re_["F2"][rs])), mock_share=float(w[ms].sum() / tot_m),
                       real_share=float(rs.sum() / rin.sum()), mock_frac_F2gt10=wfrac(me["F2"][ms], w[ms], 10), real_frac_F2gt10=float(np.mean(re_["F2"][rs] > 10)),
                       mock_ess=c["ess"], mock_rel_err=c["rel_err"], n_real=int(rs.sum()))
        r = res[nm]
        rep.append(f"  {nm:18s} F2 median mock {r['mock_median']:5.2f} real {r['real_median']:5.2f} | share mock {r['mock_share']:.3f} real {r['real_share']:.3f} | "
                   f"F2>10 mock {r['mock_frac_F2gt10']:.3f} real {r['real_frac_F2gt10']:.3f} | ESS {r['mock_ess']:.0f} n_real {r['n_real']}")
    # counterfactual: real F2 per (class x M_G) cell reweighted to the mock's population mix
    rep.append("")
    out["f2"] = res


def sec_parent(parent_dir: Path, fragment: Path, rep, out):
    """Q1(d): looks-single attrition and BJ vs measured-parallax distance migration, by (G, d)."""
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    par = ps.load_parent_snapshot(parent_dir, cfg, ps.load_proposal_set_fragment(fragment).proposal)
    pc = par.columns
    g = np.asarray(pc["phot_g_mean_mag"], float)
    plx = np.asarray(pc["parallax"], float)
    snr = plx / np.asarray(pc["parallax_error"], float)
    dzp = par.truth_parallax_mas - plx
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[par.usable & (g >= a) & (g < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    d_bj = 1.0 / par.truth_parallax_mas
    with np.errstate(divide="ignore", invalid="ignore"):
        d_px = np.where(plx + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)] > 0, 1.0 / (plx + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]), np.nan)
    fl = par.flags
    pre = np.logical_and.reduce([v for k, v in fl.items() if not k.startswith("single_")])
    res = {"zp_mas": zp.tolist(), "cells": {}}
    rep.append("== parent: looks-single attrition and distance definitions (rows passing the non-single-catalog flags) ==")
    rw = par.row_weight if par.row_weight is not None else np.ones(g.size)
    for gl, gh in ((10, 13), (13, 16), (16, 19.5)):
        for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
            m = pre & (g >= gl) & (g < gh) & (d_bj >= lo) & (d_bj < hi)
            if par.supplement is not None:
                m &= ~par.supplement
            n = int(m.sum())
            if n == 0:
                continue
            fr = {k: float(1 - fl[k][m].mean()) for k in ("single_nss", "single_ruwe", "single_cmd") if k in fl}
            mig = float(np.mean(~((d_px[m] >= lo) & (d_px[m] < hi))))
            med = float(np.nanmedian(d_px[m] / d_bj[m]))
            k = f"G{gl}-{gh} dBJ{lo}-{hi}"
            res["cells"][k] = dict(n=n, fail=fr, fail_any=float(1 - par.usable[m].mean()), frac_other_bin_by_plx=mig, median_dplx_over_dbj=med,
                                   median_snr=float(np.nanmedian(snr[m])))
            rep.append(f"  {k}: n {n}, fail nss {fr.get('single_nss', 0):.3f} ruwe {fr.get('single_ruwe', 0):.3f} cmd {fr.get('single_cmd', 0):.3f} any {res['cells'][k]['fail_any']:.3f}; "
                       f"in another d bin by 1/(plx+zp) {mig:.3f}; median d_plx/d_BJ {med:.3f}; median plx SNR {res['cells'][k]['median_snr']:.0f}")
    rep.append("")
    out["parent"] = res
    return par


def figures(out, figdir: Path):
    from darkhunter_pop import plotting as pl

    plt = pl.require_pyplot()
    figdir.mkdir(parents=True, exist_ok=True)
    # (1) generation-subset estimators per d bin
    if "gens" in out:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, kind in zip(axs, ("orbits", "accels")):
            real = np.asarray(out["gens"]["real_" + kind], float)
            keys = list(out["gens"][kind])
            subs = list(out["gens"][kind][keys[0]])
            x = np.arange(len(keys))
            for j, s in enumerate(subs):
                v = np.array([out["gens"][kind][k][s]["sum"] for k in keys]) / real
                e = v * np.array([out["gens"][kind][k][s]["rel_err"] for k in keys])
                st = pl.series_style(j)
                ax.errorbar(x + 0.08 * (j - len(subs) / 2), v, yerr=e, color=st["color"], marker=st["marker"], linestyle="none", label=s)
            ax.axhline(1, color="k", lw=1)
            ax.set_xticks(x, [f"{k} kpc" for k in keys])
            pl.apply_axes_style(ax, xlabel="measured distance bin", ylabel="mock / real (no spurious)", title=f"{kind}: generation-subset estimators")
            ax.legend(fontsize=8)
        pl.save_figure(fig, figdir / "gens_subset_estimators.png", dpi=150)
    # (2) decomposition in the 0.7-1.5 kpc bin vs neighbours
    if "decomp" in out:
        keys = ["log_a0_snr", "G", "M_G_obs", "log_P", "F2", "ruwe", "a0_mas", "evolved", "bp_rp_obs"]
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        for ax, nm in zip(axs.ravel(), keys):
            for j, dk in enumerate(("0.3-0.7", "0.7-1.5", "1.5-6.0")):
                rows = out["decomp"].get(dk, {}).get(nm)
                if not rows:
                    continue
                xc = np.array([0.5 * (r["lo"] + r["hi"]) if np.isfinite(r["hi"]) else r["lo"] for r in rows])
                xi = np.arange(len(rows))
                v = np.array([r["ratio"] for r in rows])
                e = v * np.array([r["mock_rel_err"] for r in rows])
                st = pl.series_style(j)
                ax.errorbar(xi + 0.12 * (j - 1), v, yerr=e, color=st["color"], marker=st["marker"], linestyle="-", lw=1.5, label=f"d {dk} kpc")
                vs = np.array([r["ratio_spur"] for r in rows])
                ax.plot(xi + 0.12 * (j - 1), vs, color=st["color"], marker="x", linestyle=":", lw=1)
            ax.set_xticks(np.arange(len(rows)), [f"{r['lo']:g}" for r in rows], fontsize=7)
            ax.axhline(1, color="k", lw=1)
            ax.set_ylim(0, 2.5)
            pl.apply_axes_style(ax, xlabel=f"{nm} (bin lower edge)", ylabel="mock / real", title=nm)
        axs[0, 0].legend(fontsize=7)
        fig.suptitle("orbits, mock/real per measured-distance bin (solid: mock only, row-level MC error; x dotted: mock + spurious); primary theta")
        pl.save_figure(fig, figdir / "decomp_by_distance.png", dpi=150)
    # (2b) A_obs = (a0/plx)/P^(2/3) by distance: mock/real for both theta
    if "amrf" in out:
        labs = list(out["amrf"])
        dks = list(out["amrf"][labs[0]])
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        for ax, dk in zip(axs.ravel(), dks):
            rows = out["amrf"][labs[0]][dk]
            x = np.arange(len(rows))
            for j, lb in enumerate(labs):
                rr = out["amrf"][lb][dk]
                st = pl.series_style(j + 1)
                real = np.array([r["real"] for r in rr], float)
                v = np.where(real >= 20, np.array([r["mock"] for r in rr]) / np.maximum(real, 1), np.nan)
                ax.errorbar(x + 0.1 * j, v, yerr=v * np.nan_to_num(np.array([r["rel_err"] for r in rr])), color=st["color"], marker=st["marker"],
                            lw=1.5, label=f"mock {lb}")
            ax.axhline(1, color="k", lw=1)
            ax.set_ylim(0, 3)
            ax.set_xticks(x, [f"{r['lo']:g}" for r in rows], fontsize=8)
            pl.apply_axes_style(ax, xlabel="A bin lower edge [AU yr$^{-2/3}$]", ylabel="mock / real", title=f"d {dk} kpc")
        axs[0, 0].legend(fontsize=8)
        fig.suptitle("orbits by photocentre size A = (a0/plx) / P$^{2/3}$ (bins with >= 20 real orbits)")
        pl.save_figure(fig, figdir / "amrf_by_distance.png", dpi=150)
    # (2c) CMD twin/single by truth distance: real hosts vs both theta
    tw = [k for k in out if k.startswith("twins_")]
    if tw and "real_twin" in out:
        fig, ax = plt.subplots(figsize=(6, 4))
        xd = np.array(TWIN_D)
        xc = 0.5 * (xd[1:] + xd[:-1])
        ax.plot(xc, out["real_twin"], color="k", marker="s", lw=2, label="real orbit hosts")
        for j, k in enumerate(tw):
            rr = out[k]["orbits"]
            v = np.array([rr[f"{a}-{b}"]["ratio"] for a, b in zip(TWIN_D[:-1], TWIN_D[1:])], float)
            e = v * np.array([rr[f"{a}-{b}"]["ratio_rel_err"] for a, b in zip(TWIN_D[:-1], TWIN_D[1:])], float)
            st = pl.series_style(j + 1)
            ax.errorbar(xc * (1 + 0.04 * j), v, yerr=e, color=st["color"], marker=st["marker"], lw=1.5, label=f"mock {k[6:]}")
        ax.set_xscale("log")
        ax.set_ylim(0, 3)
        pl.apply_axes_style(ax, xlabel="distance [kpc]", ylabel="twin / single (orbit hosts, C0 0.85-2.0)", title="CMD twin-band occupancy")
        ax.legend(fontsize=8)
        pl.save_figure(fig, figdir / "twin_band_by_distance.png", dpi=150)
    # (3) per-cell Δll
    if "cells" in out and "delta" in out["cells"]:
        rows = out["cells"]["delta"]["orbit_rows"][:14]
        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(rows))
        ax.barh(y, [r["dll"] for r in rows], color=pl.series_style(0)["color"])
        ax.set_yticks(y, [f"{r['label']} (k {r['k']:.0f})" for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=1)
        pl.apply_axes_style(ax, xlabel="Δ ln L (theta 42-46 − theta 42-44), data 42-46", title="orbit cells driving the mode switch")
        pl.save_figure(fig, figdir / "cells_dll.png", dpi=150)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True, help="primary dump (theta on the data under study)")
    ap.add_argument("--alt-dump", type=Path, default=None, help="second dump on the same data (cells, twins)")
    ap.add_argument("--labels", nargs=2, default=["theta42_44", "theta42_46"], help="labels for (alt, primary) in cells")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_coarse48_v12.yaml"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single46.yaml"))
    ap.add_argument("--real-input-columns", type=Path, default=None)
    ap.add_argument("--cmd-twin", type=Path, default=Path("docs/gate391/cmd_twin/cmd_twin.json"), help="real twin/single by distance (cmd_twin_391.py)")
    ap.add_argument("--sections", nargs="+", default=["gens", "decomp", "stages", "cells", "twins", "f2", "parent", "fscatter", "amrf", "heavy", "lsdeplete"])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    import yaml

    fc = yaml.safe_load(a.fit_config.read_text())["rung3"]
    tk = ("parallax_mas", "log10_flux_ratio", "is_dark", "iso_phase", "sys_bp_rp", "sys_colour0", "sys_mg0", "primary_colour0", "primary_mg0",
          "resolved_companion", "row_phot_g_mean_mag")
    z = load_mock(a.dump, a.cache, a.artifacts, fc["outer"], truth_keys=tk)
    rep = [f"#554 diagnostics (#391), primary dump {a.dump}, alt dump {a.alt_dump}", f"theta (primary) {np.round(z['theta'], 3).tolist()}", ""]
    out: dict = {}
    if a.cmd_twin.exists():
        out["real_twin"] = [x["twin_over_single"] for x in json.loads(a.cmd_twin.read_text())["by_distance"]["real_orbits"]]
    if "gens" in a.sections:
        sec_gens(z, rep, out)
    if "decomp" in a.sections:
        sec_decomp(z, rep, out, a.real_input_columns)
    if "stages" in a.sections:
        sec_stages(a.cache, a.artifacts, 0.0, rep, out, fc["outer"])
    if "cells" in a.sections and a.alt_dump is not None:
        sec_cells({a.labels[0]: a.alt_dump, a.labels[1]: a.dump}, rep, out)
    par = None
    if "parent" in a.sections:
        par = sec_parent(a.parent_dir, a.fragment, rep, out)
    if "lsdeplete" in a.sections:
        if par is None:
            from darkhunter_pop import proposal_set as ps
            from darkhunter_pop.config_loader import load_config

            par = ps.load_parent_snapshot(a.parent_dir, load_config(), ps.load_proposal_set_fragment(a.fragment).proposal)
        sec_lsdeplete(par, a.cache, a.artifacts, rep, out)
    if "twins" in a.sections:
        if par is None:
            from darkhunter_pop import proposal_set as ps
            from darkhunter_pop.config_loader import load_config

            par = ps.load_parent_snapshot(a.parent_dir, load_config(), ps.load_proposal_set_fragment(a.fragment).proposal)
        ridge = par.meta["ms_ridge"]
        sec_twins(z, ridge, rep, out, a.labels[1])
        if a.alt_dump is not None:
            sec_twins(load_mock(a.alt_dump, a.cache, a.artifacts, fc["outer"], truth_keys=tk), ridge, rep, out, a.labels[0])
    if "f2" in a.sections:
        sec_f2(z, rep, out)
    if "amrf" in a.sections:
        sec_amrf({a.labels[1]: a.dump, **({a.labels[0]: a.alt_dump} if a.alt_dump else {})}, rep, out)
    if "heavy" in a.sections and a.alt_dump is not None:
        sec_heavy(z, np.load(a.alt_dump, allow_pickle=True), rep, out)
    if "fscatter" in a.sections:
        sec_fscatter(a.cache, a.artifacts, rep, out)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "diag.json").write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
    (a.out_dir / "diag.txt").write_text("\n".join(rep) + "\n")
    figures(out, a.out_dir)
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
