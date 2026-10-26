#!/usr/bin/env python3
"""Joint luminous + compact-object rung-3 refit with the #554 fixes (#561; spec §12.37).

Builds on ``rung3_refit5_391.py`` (setup, real samples, spurious template, SAY likelihood, parent-row MC variance):

* luminous gens 42–46 (one deterministic mixture) and co_v1 gens 1000–1003 (their own mixture), never pooled:
  μ_b = μ_lum,b(θ) + R_wd μ_wd,b + R_dark μ_dark,b + N_s T_b, variances added (R² for the compact objects);
* F2 variance inflation below ESS_min, F3 q-dependent flux scatter, F4 twin-band term, F5 f_evo on the primary
  phase, F6 looks-single depletion; F1 (A, a0/σ) and F2-goodness PPCs in the report.

Usage (one heavy job; see ``output/.../chain_joint.sh``)::

    PYTHONPATH=src .venv/bin/python scripts/rung3_joint_391.py --fit-config config/population/rung3_joint_v1.yaml \\
        --cache <weights_gen42_46.npz> --artifact-dir <dir> --artifacts single_gen42.h5 ... \\
        --co-artifacts co_gen1000_smoke.h5 ... --co-fragments <yaml> ... --fragment config/population/proposal_set_single46.yaml \\
        --gate390 ... --parent-dir ... --real-snapshot ... --real-input-columns ... --accel-snapshot ... --out-dir <dir> [--fix NAME=V ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_refit4_391 as r4  # noqa: E402
import rung3_refit5_391 as r5  # noqa: E402
from rung3_fit_391 import bin_index, say_loglike  # noqa: E402

from darkhunter_pop import compact_objects as cob  # noqa: E402
from darkhunter_pop import joint_refit as jr  # noqa: E402
from darkhunter_pop import proposal_set as ps  # noqa: E402
from darkhunter_pop.rung3_model import clustered_bin_variance, row_bin_groups  # noqa: E402


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(c, 0.5 * c[-1]), x.size - 1)]) if c.size and c[-1] > 0 else float("nan")


def luminous_extras(args, ns, parent, jcfg, rep):
    """Per-draw truth extras of the luminous generations: F3 factor, F5 primary phase, F6 non-single flags and the
    luminous part of the depletion numerator (base MdS17 weights over all draws), the system CMD of the fit draws."""
    zc = np.load(args.cache, allow_pickle=True)
    keys = ("parent_row", "log10_flux_ratio", "m1_msun", "m2_msun", "sys_evolved", "iso_phase", "sys_mg0", "sys_colour0", "iso_feh", "iso_log_age", "is_dark")
    parts = [ps.read_proposal_artifact(args.artifact_dir / a) for a in args.artifacts]
    t = {k: np.concatenate([np.asarray(p[0][k]) for p in parts]) for k in keys}
    st = np.concatenate([np.asarray(p[1]["solution_type"]) for p in parts])
    if st.size != np.asarray(zc["generation"]).size:
        raise SystemExit("luminous artifacts do not match the weight cache")
    rel = ns.rel
    ridge = parent.meta["ms_ridge"]
    out = {"ridge": ridge, "sys_mg0": t["sys_mg0"][rel], "sys_colour0": t["sys_colour0"][rel]}
    # F3
    fs = jr.FluxScatterConfig(**jcfg["flux_scatter"])
    sub = {k: t[k][rel] for k in ("parent_row", "m1_msun", "m2_msun", "iso_feh", "iso_log_age")}
    relf = ps.mist_relation_for_draws(sub, parent, ns.cfg)
    q = t["m2_msun"][rel] / t["m1_msun"][rel]
    lr3 = jr.flux_scatter_log_ratio(t["log10_flux_ratio"][rel], relf, q, t["sys_evolved"][rel] | t["is_dark"][rel], fs)
    out["f3_log_ratio"] = lr3
    rep.append(f"F3 flux scatter {fs.model_dump()}: |Δ ln w| median {np.median(np.abs(lr3)):.3f}, q >= 0.95 median Δ ln w {np.median(lr3[q >= 0.95]) if np.any(q >= 0.95) else float('nan'):+.3f}")
    # F5
    out["primary_evolved"] = (np.asarray(t["iso_phase"][rel], float) >= float(jcfg["primary_phase_evolved_min"])).astype(float)
    rep.append(f"F5 f_evo on primary phase >= {jcfg['primary_phase_evolved_min']}: {out['primary_evolved'].mean():.3f} of fit draws (system class {ns.m_cls.mean():.3f})")
    # F6: luminous base weights (MdS17, noW) over ALL draws
    w0 = ps.importance_weights(np.asarray(zc["ll"], float), list(np.asarray(zc["lqs"], float)), [int(x) for x in zc["n_per_gen"]],
                               scale_to_full=float(zc["scale"]))
    dcfg = jr.DepletionConfig(**jcfg["depletion"])
    nonls = jr.nonsingle_flags(st, np.asarray(zc["cascade"], float), np.asarray(zc["accepted"], bool), np.asarray(zc["published_acceleration"], bool),
                               t["sys_mg0"], t["sys_colour0"], t["sys_evolved"], ridge, dcfg)
    out.update(w0=w0, nonls=nonls, rows_all=t["parent_row"])
    return out


def co_block(args, parent, fc, jcfg, oe, ae, n_ob, n_ab, rep):
    """Compact-object draws (gens 1000–1003): base weights (rate 1 per primary per component), F6 inputs, bins."""
    frags = [ps.load_proposal_set_fragment(f) for f in args.co_fragments]
    gens = [f.proposal for f in frags]
    cob.check_co_generations(gens)
    parts = [ps.read_proposal_artifact(a) for a in args.co_artifacts]
    tk = set.intersection(*(set(p[0]) for p in parts))
    ok_ = set.intersection(*(set(p[1]) for p in parts))
    t = {k: np.concatenate([p[0][k] for p in parts]) for k in tk}
    o = {k: np.concatenate([p[1][k] for p in parts]) for k in ok_}
    w = cob.co_importance_weights(t, parent, gens, frags[0].target_co, partition="all")
    obs = jr.co_observables(t, o, float(fc["c1_period_max_days"]))
    out_ = {a: np.asarray(v, float) for a, v in fc["outer"].items()}
    o_cols = [obs["g"], obs["d_orbit"], obs["log_p"], obs["ecc"]]
    in_o = obs["in_orbit"] & (bin_index(o_cols, [out_[a] for a in r4.AXES]) >= 0)
    in_a = obs["in_accel"] & (bin_index([obs["g"], obs["d_accel"]], [out_["g_mag"], out_["distance_kpc"]]) >= 0)
    ob = np.where(in_o, bin_index(o_cols + [obs["cls"]], oe + [r5.CLS_EDGES]), -1)
    ab = np.where(in_a, bin_index([obs["g"], obs["d_accel"], obs["cls"]], ae + [r5.CLS_EDGES]), -1)
    dcfg = jr.DepletionConfig(**jcfg["depletion"])
    nonls = jr.nonsingle_flags(o["solution_type"], o["cascade"], o["accepted_orbital"], o["published_acceleration"],
                               t["sys_mg0"], t["sys_colour0"], t["sys_evolved"], parent.meta["ms_ridge"], dcfg)
    comp = np.asarray(t["co_component"])
    rep.append(f"compact objects: {comp.size} draws (gens {[g.generation for g in gens]}); C1 orbits WD {int((in_o & (comp == 0)).sum())}, dark {int((in_o & (comp == 1)).sum())}; "
               f"accelerations WD {int((in_a & (comp == 0)).sum())}, dark {int((in_a & (comp == 1)).sum())}")
    casc = np.asarray(o["cascade"], float)
    return dict(w=w, comp=comp, ob=ob, ab=ab, in_o=in_o, in_a=in_a, rows=np.asarray(t["parent_row"], np.int64), nonls=nonls,
                d_o=obs["d_orbit"], sys_mg0=np.asarray(t["sys_mg0"], float), sys_colour0=np.asarray(t["sys_colour0"], float),
                a0=casc[:, 17], sa0=casc[:, 18], plx=casc[:, 0], P=casc[:, 10], f2=casc[:, 21], g=obs["g"])


def real_cmd(args, ns, parent):
    """Real C1 orbits' dereddened (colour0, M_G0), as refit 5's class assignment (same zero point and dust)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    from darkhunter_pop.giants import cmd_for_rows, load_giants_config

    cfg = ns.cfg
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    ge = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr > 20)]) for a, b in zip(ge[:-1], ge[1:])])
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    col = lambda n: np.ma.filled(np.ma.asarray(rt[n], float), np.nan)  # noqa: E731
    sid = np.asarray(rt["source_id"], np.int64)
    pos = {s: i for i, s in enumerate(sid)}
    idx = np.array([pos[s] for s in ns.r_sid])
    g, bprp = col("g_mag")[idx], (col("bp_mag") - col("rp_mag"))[idx]
    plx = col("parallax")[idx] + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
    eplx = col("parallax_error")[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        rr, rlo, rhi = np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + eplx), np.nan), np.where(plx - eplx > 0, 1000 / (plx - eplx), np.nan)
    gal = SkyCoord(ra=col("ra")[idx] * u.deg, dec=col("dec")[idx] * u.deg).galactic
    c = cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
    return np.asarray(c.colour0, float), np.asarray(c.mg0, float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--gate390", "--parent-dir", "--real-snapshot", "--real-input-columns", "--accel-snapshot", "--cache", "--artifact-dir", "--fit-config", "--fragment", "--out-dir"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--co-artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--co-fragments", type=Path, nargs="+", required=True)
    ap.add_argument("--calibration", type=Path, default=Path("docs/gate391/rung3_modelfix2/spurious_calibration.json"))
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--min-ess", type=float, default=30.0)
    ap.add_argument("--fix", nargs="*", default=None, help="NAME=VALUE held fixed (profiles)")
    ap.add_argument("--fix-from", type=Path, default=None, help="summary.json whose theta supplies --fix-names")
    ap.add_argument("--fix-names", nargs="*", default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    args = ap.parse_args(argv)
    import yaml

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    jcfg = fc["joint"]
    # refit-5 plumbing: luminous setup + real samples (the "new" model with the §12.16 template)
    a5 = argparse.Namespace(**vars(args), model="new", extrapolation=fc.get("spurious_extrapolation", "clamp"), new_gen=46,
                            control=None, lowmass_correction_off=False, below_envelope=None, cmd_conditioning=None, mc_variance=None,
                            mass_scale=None, m_break=None, allow_mixed=False, dump_from=None, wkeys=["noW"], init_key=None, theta_summary=None,
                            base_config=args.fit_config)
    gcs = []
    import h5py

    for a_ in args.artifacts:
        with h5py.File(args.artifact_dir / a_, "r") as h_:
            gcs.append(ps.ProposalConfig.model_validate_json(h_.attrs["proposal_config_json"]))
    a5.primary_model = ps.check_single_primary_model(gcs)  # luminous family only (spec §13.2)
    ns = r5.extend(a5, fc, r4.setup(a5, fc))
    cfg = ns.cfg
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, ps.load_proposal_set_fragment(args.fragment).proposal)
    rep = [f"#561 joint luminous + co_v1 refit — {fc.get('label')}", f"luminous primary model: {a5.primary_model}", ""]
    lx = luminous_extras(args, ns, parent, jcfg, rep)
    # θ-independent luminous factors: F3, F6 (luminous base); the luminous cache order is the artifacts' order
    dcfg = jr.DepletionConfig(**jcfg["depletion"])
    row_g = np.asarray(parent.columns["phot_g_mean_mag"], float)
    row_d = 1.0 / parent.truth_parallax_mas
    oe = [np.asarray(fc["orbit_bins"][a], float) for a in r4.AXES]
    ae = [np.asarray(fc["acceleration_bins"][a], float) for a in ("g_mag", "distance_kpc")]
    ob, ab, k_o, k_a, n_ob, n_ab, rob = r5.binned5(ns, oe, ae)
    co = co_block(args, parent, fc, jcfg, oe, ae, n_ob, n_ab, rep)
    # F6 at base weights (luminous MdS17; compact objects at the configured starting rates)
    R0 = {"wd": np.exp(jcfg["co_rates"]["ln_R_wd"][0]), "dark": np.exp(jcfg["co_rates"]["ln_R_dark"][0])}
    w_dep = np.concatenate([lx["w0"], co["w"] * np.where(co["comp"] == 0, R0["wd"], R0["dark"])])
    lf6, dsum = jr.depletion_log_factor(w_dep, np.concatenate([lx["nonls"], co["nonls"]]), np.concatenate([lx["rows_all"], co["rows"]]),
                                        row_g, row_d, parent.usable, parent.row_weight, parent.scale_to_full, dcfg)
    co_share = jr.depletion_log_factor(co["w"] * np.where(co["comp"] == 0, R0["wd"], R0["dark"]), co["nonls"], co["rows"], row_g, row_d,
                                       parent.usable, parent.row_weight, parent.scale_to_full, dcfg)[0]
    rep.append("F6 looks-single depletion P(looks single) per (G, d) cell: " + "; ".join(f"{k} {v['p_looks_single']:.3f}" for k, v in dsum.items()))
    rep.append(f"   compact-object share of Λ_ns at the starting rates: max {np.max(co_share):.2e} (luminous max {np.max(lf6):.3f})")
    f_lum = np.exp(lx["f3_log_ratio"] + lf6[ns.prow])
    ns.W = {k: v * f_lum for k, v in ns.W.items()}
    w_co = co["w"] * np.exp(lf6[co["rows"]])
    # spurious template (§12.16), as refit 5
    cal = json.loads(args.calibration.read_text())
    pi_i = r5.pi_hat(cal, ns.r_cov, clamp=(a5.extrapolation == "clamp"))
    T = np.bincount(rob[rob >= 0], weights=pi_i[rob >= 0], minlength=n_ob)
    S0 = float(T.sum())
    T = T / S0
    rng = np.random.default_rng(args.seed)
    bs = rng.multivariate_normal(np.asarray(cal["pi_beta"]), np.asarray(cal["pi_cov"]), size=300)
    sig_cal = float(np.std([r5.pi_hat(cal, ns.r_cov, clamp=(a5.extrapolation == "clamp"), beta=b).sum() for b in bs]) / S0)
    prior = (S0, sig_cal * S0)
    N_o = float(k_o.sum())
    shp = [e.size - 1 for e in oe] + [2]
    phi = float(fc["empty_bin_floor"])
    base_o = np.bincount(ob[ob >= 0], weights=ns.W["noW"][ob >= 0], minlength=n_ob) + sum(
        np.bincount(co["ob"][(co["ob"] >= 0) & (co["comp"] == c)], weights=w_co[(co["ob"] >= 0) & (co["comp"] == c)], minlength=n_ob) for c in (0, 1))
    base_a = np.bincount(ab[ab >= 0], weights=ns.W["noW"][ab >= 0], minlength=n_ab)
    empty_o, empty_a = base_o < 0.5, base_a < 0.5
    floor_o, floor_a = np.where(empty_o, phi, 0.0), np.where(empty_a, phi, 0.0)
    T = np.where(empty_o, 0.0, T)
    # compact-object per-component bin sums (θ-independent apart from R)
    co_mu_o, co_v_o, co_mu_a, co_v_a = {}, {}, {}, {}
    for name, c in (("wd", 0), ("dark", 1)):
        so, sa = (co["ob"] >= 0) & (co["comp"] == c), (co["ab"] >= 0) & (co["comp"] == c)
        co_mu_o[name] = np.bincount(co["ob"][so], weights=w_co[so], minlength=n_ob)
        co_v_o[name] = np.bincount(co["ob"][so], weights=w_co[so] ** 2, minlength=n_ob)
        co_mu_a[name] = np.bincount(co["ab"][sa], weights=w_co[sa], minlength=n_ab)
        co_v_a[name] = np.bincount(co["ab"][sa], weights=w_co[sa] ** 2, minlength=n_ab)
    # F4 twin band: real counts per distance bin (C1 orbits), mock masks, spurious share from π̂
    tb = jr.TwinBandConfig(**jcfg["twin_band"])
    ridge = lx["ridge"]
    rc0, rmg = real_cmd(args, ns, parent)
    r_tw, r_si = jr.twin_band_masks(rc0, rmg, ridge, tb)
    de = np.asarray(tb.distance_edges_kpc, float)
    rdb = np.searchsorted(de, ns.ro_cols[1], side="right") - 1
    nd = de.size - 1
    kt = np.array([np.sum(r_tw & (rdb == j)) for j in range(nd)], float)
    ks = np.array([np.sum(r_si & (rdb == j)) for j in range(nd)], float)
    st_tw = np.array([pi_i[r_tw & (rdb == j)].sum() for j in range(nd)]) / S0
    st_si = np.array([pi_i[r_si & (rdb == j)].sum() for j in range(nd)]) / S0
    m_tw, m_si = jr.twin_band_masks(lx["sys_colour0"], lx["sys_mg0"], ridge, tb)
    mdb = np.searchsorted(de, ns.o_cols[1], side="right") - 1
    c_tw, c_si = jr.twin_band_masks(co["sys_colour0"], co["sys_mg0"], ridge, tb)
    cdb = np.searchsorted(de, co["d_o"], side="right") - 1
    rep.append(f"F4 twin band: real twin / single per d bin {de.tolist()}: " + ", ".join(f"{int(a)}/{int(b)}" for a, b in zip(kt, ks)))
    # F5 modifier on the primary phase; the class axis stays on the system class
    evo_mod = lx["primary_evolved"] if jcfg.get("f_evo_on") == "primary_phase" else ns.m_cls
    names = r5.POP + ("ln_f_evo", "ln_R_wd", "ln_R_dark", "N_s")
    i_fe, i_rw, i_rd, i_ns = (names.index(n) for n in ("ln_f_evo", "ln_R_wd", "ln_R_dark", "N_s"))
    grp_o, grp_a = row_bin_groups(ob, ns.prow), row_bin_groups(ab, ns.prow)
    ess_floor = float(jcfg["ess_min_inflate"])
    so_l = ob >= 0

    def full(th):
        t = np.zeros(14)
        t[:13] = th[:13]
        return t

    def lum_w(th):
        return ns.W["noW"] * np.exp(ns.log_mod(full(th)) + th[i_fe] * evo_mod)

    def expected(th):
        w = lum_w(th)
        Rw, Rd = np.exp(th[i_rw]), np.exp(th[i_rd])
        mu_o = np.bincount(ob[so_l], weights=w[so_l], minlength=n_ob) + Rw * co_mu_o["wd"] + Rd * co_mu_o["dark"]
        v_o = clustered_bin_variance(w, grp_o, n_ob) + Rw**2 * co_v_o["wd"] + Rd**2 * co_v_o["dark"]
        sa = ab >= 0
        mu_a = np.bincount(ab[sa], weights=w[sa], minlength=n_ab) + Rw * co_mu_a["wd"] + Rd * co_mu_a["dark"]
        v_a = clustered_bin_variance(w, grp_a, n_ab) + Rw**2 * co_v_a["wd"] + Rd**2 * co_v_a["dark"]
        return w, mu_o * ~empty_o, v_o * ~empty_o, mu_a * ~empty_a, v_a * ~empty_a

    def twin_terms(th, w):
        Rw, Rd = np.exp(th[i_rw]), np.exp(th[i_rd])
        wc = w_co * np.where(co["comp"] == 0, Rw, Rd)
        mt = np.array([w[so_l & m_tw & (mdb == j)].sum() + wc[co["in_o"] & c_tw & (cdb == j)].sum() for j in range(nd)]) + th[i_ns] * st_tw
        ms = np.array([w[so_l & m_si & (mdb == j)].sum() + wc[co["in_o"] & c_si & (cdb == j)].sum() for j in range(nd)]) + th[i_ns] * st_si
        return mt, ms

    def nll(th):
        w, mu_o, v_o, mu_a, v_a = expected(th)
        mo = mu_o + th[i_ns] * T + floor_o
        ll = say_loglike(k_o, mo, jr.inflate_low_ess_variance(mu_o, v_o, ess_floor))
        ll += say_loglike(k_a, mu_a + floor_a, jr.inflate_low_ess_variance(mu_a, v_a, ess_floor))
        ll += -0.5 * ((th[i_ns] - prior[0]) / prior[1]) ** 2
        if tb.enabled:
            mt, ms = twin_terms(th, w)
            ll += jr.twin_band_loglike(kt, ks, mt, ms)
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    cr = jcfg["co_rates"]
    x0 = np.array([par[n][0] for n in r5.POP] + [0.0, cr["ln_R_wd"][0], cr["ln_R_dark"][0], prior[0]], float)
    bounds = ([(par[n][1], par[n][2]) for n in r5.POP] + [(-3.0, 3.0), (cr["ln_R_wd"][1], cr["ln_R_wd"][2]), (cr["ln_R_dark"][1], cr["ln_R_dark"][2]),
                                                          (0.0, max(5 * prior[0], 1.0))])
    fixes = list(args.fix or [])
    if args.fix_from is not None:
        src = json.loads(args.fix_from.read_text())["results"]
        src = src.get("noW", src.get("joint"))
        th_src = dict(zip(src["names"], src["theta"]))
        fixes += [f"{n}={th_src[n]}" for n in (args.fix_names or [])]
    for item in fixes:
        nm_, val_ = item.split("=")
        j_ = names.index(nm_)
        val = prior[0] if val_ == "prior" else float(val_)
        bounds[j_] = (val - 1e-9, val + 1e-9)
        x0[j_] = val
        rep.append(f"profile: {nm_} fixed at {val:.4g}")
    if args.init_from is not None:
        src = json.loads(args.init_from.read_text())["results"]
        src = src.get("joint", src.get("noW"))
        th_src = dict(zip(src["names"], src["theta"]))
        x0 = np.clip(np.array([th_src.get(n, x0[j]) for j, n in enumerate(names)], float), [b[0] for b in bounds], [b[1] for b in bounds])
    rep.append(f"N_s prior N({prior[0]:.0f}, {prior[1]:.0f}); ESS floor for variance inflation {ess_floor:g}; real C1 orbits {int(N_o)}, accelerations {int(k_a.sum())}")
    sols, best = [], None
    for s_ in range(args.n_starts):
        s0 = x0 if s_ == 0 else np.array([rng.uniform(max(lo, v - 1), min(hi, v + 1)) if j != i_ns else rng.uniform(0.5 * v, 1.5 * v)
                                          for j, (v, (lo, hi)) in enumerate(zip(x0, bounds))])
        s0 = np.clip(s0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(nll, s0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 4000, "maxfun": 40000})
        r = minimize(nll, r.x, method="Powell", bounds=bounds, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
        r2 = minimize(nll, s0, method="Powell", bounds=bounds, options={"maxiter": 30000, "xtol": 1e-4, "ftol": 1e-8})
        if r2.fun < r.fun:
            r = r2
        sols.append((float(r.fun), r.x.copy()))
        if best is None or r.fun < best.fun:
            best = r
        print(f"start {s_}: -lnL {r.fun:.2f}", flush=True)
    th = best.x
    n_par = len(names)
    h = 1e-3 * np.maximum(1.0, np.abs(th))
    H = np.zeros((n_par, n_par))
    f0 = nll(th)
    for i in range(n_par):
        for j in range(i, n_par):
            def ff(di, dj):
                xx = th.copy()
                xx[i] += di
                xx[j] += dj
                return nll(xx)
            H[i, j] = H[j, i] = ((ff(h[i], 0) - 2 * f0 + ff(-h[i], 0)) / h[i] ** 2 if i == j else
                                 (ff(h[i], h[j]) - ff(h[i], -h[j]) - ff(-h[i], h[j]) + ff(-h[i], -h[j])) / (4 * h[i] * h[j]))
    try:
        dg = np.diag(np.linalg.inv(H))
        err = np.sqrt(np.where(dg > 0, dg, np.nan))
    except np.linalg.LinAlgError:
        err = np.full(n_par, np.nan)
    w, mu_o, v_o, mu_a, v_a = expected(th)
    s = th[i_ns] * T + floor_o
    Rw, Rd = np.exp(th[i_rw]), np.exp(th[i_rd])
    wc = w_co * np.where(co["comp"] == 0, Rw, Rd)
    near = sorted(round(f_ - best.fun, 2) for f_, _ in sols)
    fixed = {x.split("=")[0] for x in fixes}
    at_b = [names[i] for i in range(n_par) if names[i] not in fixed and min(abs(th[i] - bounds[i][0]), abs(th[i] - bounds[i][1])) < 1e-3 * max(1, abs(th[i]))]
    ess_b = np.where(v_o > 0, mu_o**2 / np.maximum(v_o, 1e-300), 0.0)
    big = k_o >= 0.01 * N_o
    # PPCs (F1, F2-goodness) and distance ratio, mock = luminous + compact objects (+ spurious as π̂-weighted real orbits)
    pc = jcfg["ppc"]
    dE = np.asarray(pc["distance_edges_kpc"], float)
    aE, sE = np.asarray(pc["a_edges"], float), np.asarray(pc["log_a0_snr_edges"], float)
    casc_l = ns.casc
    A_l = jr.photocentre_size(casc_l[:, 17], casc_l[:, 0], casc_l[:, 10])
    A_c = jr.photocentre_size(co["a0"], co["plx"], co["P"])
    from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes

    rx = ns.r_extra
    r_a0 = photocenter_a0_from_thiele_innes(rx["A"], rx["B"], rx["F"], rx["G"])
    A_r = jr.photocentre_size(r_a0, rx["parallax"], rx["period"])
    snr_l, snr_c = np.log10(casc_l[:, 17] / casc_l[:, 18]), np.log10(co["a0"] / co["sa0"])
    snr_r = ns.r_cov["log10_a0_snr"]
    spur_w = th[i_ns] * pi_i / S0
    ppc = {}
    for j, (lo, hi) in enumerate(zip(dE[:-1], dE[1:])):
        ml, mc_, mr = so_l & (ns.o_cols[1] >= lo) & (ns.o_cols[1] < hi), co["in_o"] & (co["d_o"] >= lo) & (co["d_o"] < hi), (ns.ro_cols[1] >= lo) & (ns.ro_cols[1] < hi)
        def hist(xl, xc, xr, edges):
            m = np.histogram(xl[ml], edges, weights=w[ml])[0] + np.histogram(xc[mc_], edges, weights=wc[mc_])[0]
            co_only = np.histogram(xc[mc_], edges, weights=wc[mc_])[0]
            sp = np.histogram(xr[mr], edges, weights=spur_w[mr])[0]
            rr = np.histogram(xr[mr], edges)[0].astype(float)
            return {"mock_lum_plus_co": m.tolist(), "co_only": co_only.tolist(), "spurious": sp.tolist(), "real": rr.tolist()}
        ppc[f"d{lo:g}-{hi:g}"] = {"A": hist(A_l, A_c, A_r, aE), "log_a0_snr": hist(snr_l, snr_c, snr_r, sE)}
    gE = np.asarray(pc["f2_g_edges"], float)
    f2 = {}
    for lo, hi in zip(gE[:-1], gE[1:]):
        ml, mc_, mr = so_l & (ns.o_cols[0] >= lo) & (ns.o_cols[0] < hi), co["in_o"] & (co["g"] >= lo) & (co["g"] < hi), (ns.ro_cols[0] >= lo) & (ns.ro_cols[0] < hi)
        f2[f"G{lo:g}-{hi:g}"] = {"mock_median": _wmedian(np.concatenate([casc_l[ml, 21], co["f2"][mc_]]), np.concatenate([w[ml], wc[mc_]])),
                                 "real_median": float(np.nanmedian(rx["goodness_of_fit"][mr])) if mr.any() else float("nan")}
    rdn = np.histogram(ns.ro_cols[1], dE)[0].astype(float)
    mdn = np.histogram(ns.o_cols[1][so_l], dE, weights=w[so_l])[0]
    cdn = np.histogram(co["d_o"][co["in_o"]], dE, weights=wc[co["in_o"]])[0]
    sdn = np.histogram(ns.ro_cols[1], dE, weights=spur_w)[0]
    mt, ms = twin_terms(th, w)
    res = dict(names=list(names), theta=th.tolist(), err=err.tolist(), nll=float(best.fun), start_offsets=near, n_near=int(sum(x < 5 for x in near)),
               n_starts=len(sols), at_bound=at_b, N_s=float(th[i_ns]), N_s_pull=float((th[i_ns] - prior[0]) / prior[1]),
               R_wd=float(Rw), R_dark=float(Rd), co_orbits={"wd": float(wc[co["in_o"] & (co["comp"] == 0)].sum()), "dark": float(wc[co["in_o"] & (co["comp"] == 1)].sum())},
               co_accels={"wd": float(wc[co["in_a"] & (co["comp"] == 0)].sum()), "dark": float(wc[co["in_a"] & (co["comp"] == 1)].sum())},
               total=float((mu_o.sum() + s.sum()) / N_o), accel=float((mu_a.sum() + floor_a.sum()) / k_a.sum()),
               d_ratio=((mdn + cdn + sdn) / rdn).tolist(), d_ratio_parts={"lum": (mdn / rdn).tolist(), "co": (cdn / rdn).tolist(), "spurious": (sdn / rdn).tolist()},
               twin_band={"real_twin": kt.tolist(), "real_single": ks.tolist(), "mock_twin": mt.tolist(), "mock_single": ms.tolist()},
               ppc=ppc, f2=f2, depletion=dsum, ess_orbits=float(ps.kish_ess(w[so_l])), n_big_low=int(np.sum(big & (ess_b < args.min_ess))), n_big=int(big.sum()))
    rep.append(f"{res['n_near']}/{res['n_starts']} starts within Δ(-lnL) < 5 (offsets {near}); -lnL {best.fun:.1f}; at bound: {at_b or 'none'}")
    for nm, v, e_ in zip(names, th, err):
        rep.append(f"    {nm:10s} {v:+10.3f} +- {e_:.3f}")
    rep.append(f"    N_s {th[i_ns]:.0f} vs prior {prior[0]:.0f} +- {prior[1]:.0f} (pull {res['N_s_pull']:+.1f}); (mock+CO+spurious)/real {res['total']:.3f}; accelerations {res['accel']:.3f}")
    rep.append(f"    compact objects: R_wd {Rw:.3g}, R_dark {Rd:.3g} per primary; expected C1 orbits WD {res['co_orbits']['wd']:.0f}, dark {res['co_orbits']['dark']:.0f}; "
               f"accelerations WD {res['co_accels']['wd']:.0f}, dark {res['co_accels']['dark']:.0f}")
    rep.append("    d bins " + " / ".join(f"{a:g}-{b:g}" for a, b in zip(dE[:-1], dE[1:])) + ": total/real " + " / ".join(f"{x:.2f}" for x in res["d_ratio"])
               + "; lum " + " / ".join(f"{x:.2f}" for x in res["d_ratio_parts"]["lum"]) + "; CO " + " / ".join(f"{x:.3f}" for x in res["d_ratio_parts"]["co"])
               + "; spurious " + " / ".join(f"{x:.2f}" for x in res["d_ratio_parts"]["spurious"]))
    rep.append("    twin/single real " + " ".join(f"{a / max(b, 1):.2f}" for a, b in zip(kt, ks)) + "; mock " + " ".join(f"{a / max(b, 1e-9):.2f}" for a, b in zip(mt, ms)))
    for k_, v_ in ppc.items():
        A_ = v_["A"]
        hi_m = sum(A_["mock_lum_plus_co"][4:]) + sum(A_["spurious"][4:])
        rep.append(f"    PPC {k_}: A > {aE[4]:g}: mock+spur {hi_m:.0f} (CO {sum(A_['co_only'][4:]):.0f}) vs real {sum(A_['real'][4:]):.0f}")
    rep.append("    F2 median mock/real: " + "; ".join(f"{k_} {v_['mock_median']:.2f}/{v_['real_median']:.2f}" for k_, v_ in f2.items()))
    rep.append(f"    ESS orbits {res['ess_orbits']:.0f}; big bins below ESS {args.min_ess:g}: {res['n_big_low']}/{res['n_big']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps({"label": fc.get("label"), "results": {"joint": res}}, indent=1, default=float))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
