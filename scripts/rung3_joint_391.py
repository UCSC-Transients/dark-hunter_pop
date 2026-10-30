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


def co_block(args, parent, fc, jcfg, oe, ae, n_ob, n_ab, rep, aE=None):
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
    casc0 = np.asarray(o["cascade"], float)
    if aE is None:
        ob = np.where(in_o, bin_index(o_cols + [obs["cls"]], oe + [r5.CLS_EDGES]), -1)
    else:
        a_c = jr.photocentre_size(casc0[:, 17], casc0[:, 0], casc0[:, 10])
        ob = np.where(in_o, bin_index(o_cols + [a_c, obs["cls"]], oe + [aE, r5.CLS_EDGES]), -1)
    ab = np.where(in_a, bin_index([obs["g"], obs["d_accel"], obs["cls"]], ae + [r5.CLS_EDGES]), -1)
    dcfg = jr.DepletionConfig(**jcfg["depletion"])
    nonls = jr.nonsingle_flags(o["solution_type"], o["cascade"], o["accepted_orbital"], o["published_acceleration"],
                               t["sys_mg0"], t["sys_colour0"], t["sys_evolved"], parent.meta["ms_ridge"], dcfg)
    comp = np.asarray(t["co_component"])
    rep.append(f"compact objects: {comp.size} draws (gens {[g.generation for g in gens]}); C1 orbits WD {int((in_o & (comp == 0)).sum())}, dark {int((in_o & (comp == 1)).sum())}; "
               f"accelerations WD {int((in_a & (comp == 0)).sum())}, dark {int((in_a & (comp == 1)).sum())}")
    casc = np.asarray(o["cascade"], float)
    shape_old = cob.co_shape_log_density(t, frags[0].target_co)  # #567: divided out when the WD target is replaced
    return dict(w=w, comp=comp, ob=ob, ab=ab, in_o=in_o, in_a=in_a, rows=np.asarray(t["parent_row"], np.int64), nonls=nonls,
                shape_old=shape_old, m2_true=np.asarray(t["m2_msun"], float), p_true=np.asarray(t["period_days"], float),
                e_true=np.asarray(t["eccentricity"], float), m1_true=np.asarray(t["m1_msun"], float),
                t_sys=10.0 ** np.asarray(t["iso_log_age"], float) / 1e9, ecc_fit=casc[:, 14], casc=casc,
                d_o=obs["d_orbit"], d_a=obs["d_accel"], sys_mg0=np.asarray(t["sys_mg0"], float), sys_colour0=np.asarray(t["sys_colour0"], float),
                a0=casc[:, 17], sa0=casc[:, 18], plx=casc[:, 0], P=casc[:, 10], f2=casc[:, 21], g=obs["g"])


def empty_co_block(n_casc: int) -> dict:
    """#587: no compact-object generations (``--no-co``): a zero-length block with the keys :func:`co_block` returns,
    so the luminous + spurious model runs without mixing in co_v1 draws simulated under another noise model."""
    e, ei = np.zeros(0), np.zeros(0, np.int64)
    eb = np.zeros(0, bool)
    return dict(w=e, comp=ei, ob=ei, ab=ei, in_o=eb, in_a=eb, rows=ei, nonls=eb, shape_old=e, m2_true=e, p_true=e, e_true=e, m1_true=e,
                t_sys=e, ecc_fit=e, casc=np.zeros((0, n_casc)), d_o=e, d_a=e, sys_mg0=e, sys_colour0=e, a0=e, sa0=e, plx=e, P=e, f2=e, g=e)


def real_accel_obs(args, ns, parent):
    """Real published accelerations in the fit domain (same order as ns.ra_sid): CMD, ϖ/σ_ϖ, significance, F2, RUWE."""
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
    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    sid = np.asarray(at["source_id"], np.int64)
    pos = {s_: i for i, s_ in enumerate(sid)}
    idx = np.array([pos[s_] for s_ in ns.ra_sid])
    col = lambda n: np.ma.filled(np.ma.asarray(at[n], float), np.nan)[idx]  # noqa: E731
    g, bprp = col("g_mag"), col("bp_mag") - col("rp_mag")
    plx = col("parallax") + zp[np.clip(np.digitize(g, ge) - 1, 0, 4)]
    eplx = col("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rr, rlo, rhi = np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + eplx), np.nan), np.where(plx - eplx > 0, 1000 / (plx - eplx), np.nan)
    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    c = cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
    return {"colour0": np.asarray(c.colour0, float), "mg0": np.asarray(c.mg0, float), "plx_snr": col("parallax") / eplx,
            "signif": col("significance"), "f2": col("goodness_of_fit"), "ruwe": col("ruwe")}


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
    ap.add_argument("--co-artifacts", type=Path, nargs="+", default=None)
    ap.add_argument("--co-fragments", type=Path, nargs="+", default=None)
    ap.add_argument("--no-co", action="store_true", help="#587: luminous + spurious only (no compact-object generations)")
    ap.add_argument("--eval-only", action="store_true", help="#587: no optimisation; evaluate at the start θ (--init-from + --fix)")
    ap.add_argument("--calibration", type=Path, default=Path("docs/gate391/rung3_modelfix2/spurious_calibration.json"))
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--min-ess", type=float, default=30.0)
    ap.add_argument("--fix", nargs="*", default=None, help="NAME=VALUE held fixed (profiles)")
    ap.add_argument("--fix-from", type=Path, default=None, help="summary.json whose theta supplies --fix-names")
    ap.add_argument("--fix-names", nargs="*", default=None)
    ap.add_argument("--init-from", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.no_co and not (args.co_artifacts and args.co_fragments):
        ap.error("--co-artifacts and --co-fragments are required unless --no-co")
    import yaml

    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    jcfg = fc["joint"]
    # refit-5 plumbing: luminous setup + real samples (the "new" model with the §12.16 template)
    a5 = argparse.Namespace(**vars(args), model="new", extrapolation=fc.get("spurious_extrapolation", "clamp"), new_gen=None,
                            control=None, lowmass_correction_off=False, below_envelope=None, cmd_conditioning=None, mc_variance=None,
                            mass_scale=None, m_break=None, allow_mixed=False, dump_from=None, wkeys=["noW"], init_key=None, theta_summary=None,
                            base_config=args.fit_config)
    gcs = []
    import h5py

    for a_ in args.artifacts:
        with h5py.File(args.artifact_dir / a_, "r") as h_:
            gcs.append(ps.ProposalConfig.model_validate_json(h_.attrs["proposal_config_json"]))
    a5.primary_model = ps.check_single_primary_model(gcs)  # luminous family only (spec §13.2)
    a5.new_gen = max(int(g.generation) for g in gcs)  # #587: last luminous generation (was hard-coded 46)
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
    # #570: optional A = (a0/ϖ)/P^(2/3) orbit axis (edges fixed in config before any fit), before the class axis
    from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes as _a0ti

    aE = np.asarray(fc["orbit_bins"]["a_photocentre"], float) if fc["orbit_bins"].get("a_photocentre") else None
    A_l_fit = jr.photocentre_size(ns.casc[:, 17], ns.casc[:, 0], ns.casc[:, 10])
    A_r_fit = jr.photocentre_size(_a0ti(ns.r_extra["A"], ns.r_extra["B"], ns.r_extra["F"], ns.r_extra["G"]), ns.r_extra["parallax"], ns.r_extra["period"])
    if aE is None:
        ob, ab, k_o, k_a, n_ob, n_ab, rob = r5.binned5(ns, oe, ae)
    else:
        ob = np.where(ns.in_o, bin_index(ns.o_cols + [A_l_fit, ns.m_cls], oe + [aE, r5.CLS_EDGES]), -1)
        ab = np.where(ns.in_a, bin_index(ns.a_cols + [ns.m_cls], ae + [r5.CLS_EDGES]), -1)
        n_ob = int(np.prod([e.size - 1 for e in oe])) * (aE.size - 1) * 2
        n_ab = int(np.prod([e.size - 1 for e in ae])) * 2
        rob = bin_index(ns.ro_cols + [A_r_fit, ns.r_cls], oe + [aE, r5.CLS_EDGES])
        rab = bin_index(ns.ra_cols + [ns.ra_cls], ae + [r5.CLS_EDGES])
        k_o = np.bincount(rob[rob >= 0], minlength=n_ob).astype(float)
        k_a = np.bincount(rab[rab >= 0], minlength=n_ab).astype(float)
        rep.append(f"A axis edges {aE.tolist()}: real C1 orbits per A bin {np.histogram(A_r_fit[rob >= 0], aE)[0].tolist()}")
    co = empty_co_block(ns.casc.shape[1]) if args.no_co else co_block(args, parent, fc, jcfg, oe, ae, n_ob, n_ab, rep, aE=aE)
    if args.no_co:
        rep.append("compact objects: none (--no-co, #587): luminous + spurious only")
    # F6 at base weights (luminous MdS17; compact objects at the configured starting rates)
    R0 = {"wd": np.exp(jcfg["co_rates"]["ln_R_wd"][0]), "dark": np.exp(jcfg["co_rates"]["ln_R_dark"][0])}
    w_dep = np.concatenate([lx["w0"], co["w"] * np.where(co["comp"] == 0, R0["wd"], R0["dark"])])
    lf6, dsum = jr.depletion_log_factor(w_dep, np.concatenate([lx["nonls"], co["nonls"]]), np.concatenate([lx["rows_all"], co["rows"]]),
                                        row_g, row_d, parent.usable, parent.row_weight, parent.scale_to_full, dcfg)
    co_share = jr.depletion_log_factor(co["w"] * np.where(co["comp"] == 0, R0["wd"], R0["dark"]), co["nonls"], co["rows"], row_g, row_d,
                                       parent.usable, parent.row_weight, parent.scale_to_full, dcfg)[0]
    rep.append("F6 looks-single depletion P(looks single) per (G, d) cell: " + "; ".join(f"{k} {v['p_looks_single']:.3f}" for k, v in dsum.items()))
    rep.append(f"   compact-object share of Λ_ns at the starting rates: max {np.max(co_share) if np.size(co_share) else 0.0:.2e} (luminous max {np.max(lf6):.3f})")
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
    shp = [e.size - 1 for e in oe] + ([aE.size - 1] if aE is not None else []) + [2]
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
    wdt = jr.WDTargetConfig(**jcfg["wd_target"]) if jcfg.get("wd_target") else None
    wd_on = wdt is not None and wdt.enabled
    wdp = jr.WDPhysicsConfig(**jcfg["wd_physics"]) if jcfg.get("wd_physics") else None
    wdp_on = wdp is not None and wdp.enabled
    if wdp_on:
        wd_on = False  # §12.39 replaces the §12.38 target
    wd_names = ("beta_wd", "f_mt", "f_he", "alpha_ni") if wd_on else (("ln_R_mt", "f_he") if wdp_on else ())
    names = r5.POP + ("ln_f_evo", "ln_R_wd", "ln_R_dark") + wd_names + ("N_s",)
    i_fe, i_rw, i_rd, i_ns = (names.index(n) for n in ("ln_f_evo", "ln_R_wd", "ln_R_dark", "N_s"))
    # #567: the WD draws that enter any bin or term; their target is re-evaluated at each θ
    wd_rel = np.flatnonzero((co["comp"] == 0) & (co["in_o"] | co["in_a"]))

    def wd_weights(th):
        """WD draw weights at θ: default-target weights × λ_§12.38(θ) / λ_default (or × R_wd without the new target)."""
        if wdp_on:  # §12.39: physics-derived non-interacting rate (ln A_ni = ln_R_wd) + post-MT branch
            lnew = jr.wd_physics_log_density(co["m2_true"][wd_rel], co["p_true"][wd_rel], co["e_true"][wd_rel], co["m1_true"][wd_rel], co["t_sys"][wd_rel],
                                             ln_a_ni=th[i_rw], ln_r_mt=th[names.index("ln_R_mt")], f_he=th[names.index("f_he")], cfg=wdp)
            with np.errstate(invalid="ignore", over="ignore"):
                r_ = np.exp(lnew - co["shape_old"][wd_rel])
            return w_co[wd_rel] * np.where(np.isfinite(r_), r_, 0.0)
        if not wd_on:
            return w_co[wd_rel] * np.exp(th[i_rw])
        j_ = {n: names.index(n) for n in wd_names}
        lnew = jr.wd_target_log_density(co["m2_true"][wd_rel], co["p_true"][wd_rel], co["e_true"][wd_rel], co["m1_true"][wd_rel], co["t_sys"][wd_rel],
                                        ln_r_wd=th[i_rw], beta=th[j_["beta_wd"]], f_mt=th[j_["f_mt"]], f_he=th[j_["f_he"]], alpha=th[j_["alpha_ni"]], cfg=wdt)
        with np.errstate(invalid="ignore", over="ignore"):
            r_ = np.exp(lnew - co["shape_old"][wd_rel])
        return w_co[wd_rel] * np.where(np.isfinite(r_), r_, 0.0)
    grp_o, grp_a = row_bin_groups(ob, ns.prow), row_bin_groups(ab, ns.prow)
    ess_floor = float(jcfg["ess_min_inflate"])
    so_l = ob >= 0

    def full(th):
        t = np.zeros(14)
        t[:13] = th[:13]
        return t

    def lum_w(th):
        return ns.W["noW"] * np.exp(ns.log_mod(full(th)) + th[i_fe] * evo_mod)

    wob, wab = co["ob"][wd_rel], co["ab"][wd_rel]

    def expected(th):
        w = lum_w(th)
        Rd = np.exp(th[i_rd])
        ww = wd_weights(th)
        so_w, sa_w = wob >= 0, wab >= 0
        mu_wd_o = np.bincount(wob[so_w], weights=ww[so_w], minlength=n_ob)
        v_wd_o = np.bincount(wob[so_w], weights=ww[so_w] ** 2, minlength=n_ob)
        mu_wd_a = np.bincount(wab[sa_w], weights=ww[sa_w], minlength=n_ab)
        v_wd_a = np.bincount(wab[sa_w], weights=ww[sa_w] ** 2, minlength=n_ab)
        mu_o = np.bincount(ob[so_l], weights=w[so_l], minlength=n_ob) + mu_wd_o + Rd * co_mu_o["dark"]
        v_o = clustered_bin_variance(w, grp_o, n_ob) + v_wd_o + Rd**2 * co_v_o["dark"]
        sa = ab >= 0
        mu_a = np.bincount(ab[sa], weights=w[sa], minlength=n_ab) + mu_wd_a + Rd * co_mu_a["dark"]
        v_a = clustered_bin_variance(w, grp_a, n_ab) + v_wd_a + Rd**2 * co_v_a["dark"]
        return w, mu_o * ~empty_o, v_o * ~empty_o, mu_a * ~empty_a, v_a * ~empty_a

    def co_draw_weights(th):
        """All compact-object draw weights at θ (dark × R_dark; WD as :func:`wd_weights`, 0 outside the fit draws)."""
        wc_ = w_co * np.where(co["comp"] == 1, np.exp(th[i_rd]), 0.0)
        wc_[wd_rel] = wd_weights(th)
        return wc_

    def twin_terms(th, w):
        wc = co_draw_weights(th)
        mt = np.array([w[so_l & m_tw & (mdb == j)].sum() + wc[co["in_o"] & c_tw & (cdb == j)].sum() for j in range(nd)]) + th[i_ns] * st_tw
        ms = np.array([w[so_l & m_si & (mdb == j)].sum() + wc[co["in_o"] & c_si & (cdb == j)].sum() for j in range(nd)]) + th[i_ns] * st_si
        return mt, ms

    def nll(th):
        w, mu_o, v_o, mu_a, v_a = expected(th)
        mo = mu_o + th[i_ns] * T + floor_o
        if relvar["o"] is not None:  # #570: relative MC variance fixed at the reference θ, v = (μ / μ_ref)² v_ref × F2
            ll = say_loglike(k_o, mo, mu_o**2 * relvar["o"]) + say_loglike(k_a, mu_a + floor_a, mu_a**2 * relvar["a"])
        elif infl["o"] is not None:  # #570: F2 factors fixed at the reference θ (not gameable by the optimiser)
            ll = say_loglike(k_o, mo, v_o * infl["o"]) + say_loglike(k_a, mu_a + floor_a, v_a * infl["a"])
        else:
            ll = say_loglike(k_o, mo, jr.inflate_low_ess_variance(mu_o, v_o, ess_floor))
            ll += say_loglike(k_a, mu_a + floor_a, jr.inflate_low_ess_variance(mu_a, v_a, ess_floor))
        ll += -0.5 * ((th[i_ns] - prior[0]) / prior[1]) ** 2
        if tb.enabled:
            mt, ms = twin_terms(th, w)
            ll += jr.twin_band_loglike(kt, ks, mt, ms)
        return -ll if np.isfinite(ll) else 1e30

    par = fc["parameters"]
    cr = jcfg["co_rates"]
    wp = jcfg.get("wd_params", {}) if (wd_on or wdp_on) else {}
    x0 = np.array([par[n][0] for n in r5.POP] + [0.0, cr["ln_R_wd"][0], cr["ln_R_dark"][0]] + [wp[n][0] for n in wd_names] + [prior[0]], float)
    bounds = ([(par[n][1], par[n][2]) for n in r5.POP] + [(-3.0, 3.0), (cr["ln_R_wd"][1], cr["ln_R_wd"][2]), (cr["ln_R_dark"][1], cr["ln_R_dark"][2])]
              + [(wp[n][1], wp[n][2]) for n in wd_names] + [(0.0, max(5 * prior[0], 1.0))])
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
    infl = {"o": None, "a": None}
    relvar = {"o": None, "a": None}
    if jcfg.get("mc_variance_at", "current") == "reference_relative":
        _, mo_r, vo_r, ma_r, va_r = expected(x0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ro = np.where(mo_r > 0, vo_r / np.where(mo_r > 0, mo_r, 1.0) ** 2, 0.0)
            ra = np.where(ma_r > 0, va_r / np.where(ma_r > 0, ma_r, 1.0) ** 2, 0.0)
            fo = np.where(vo_r > 0, np.maximum(1.0, ess_floor / np.maximum(mo_r**2 / np.where(vo_r > 0, vo_r, 1.0), 1e-300)), 1.0)
            fa = np.where(va_r > 0, np.maximum(1.0, ess_floor / np.maximum(ma_r**2 / np.where(va_r > 0, va_r, 1.0), 1e-300)), 1.0)
        relvar["o"], relvar["a"] = ro * fo, ra * fa
        rep.append(f"MC variance: relative error per bin fixed at the reference θ (F2 floor {ess_floor:g} applied there); "
                   f"median relative MC error of orbit bins {np.median(np.sqrt(ro[mo_r > 0])):.3f}")
    if jcfg.get("ess_inflation_at", "current") == "reference":
        _, mo_r, vo_r, ma_r, va_r = expected(x0)
        with np.errstate(divide="ignore", invalid="ignore"):
            infl["o"] = np.where(vo_r > 0, np.maximum(1.0, ess_floor / np.maximum(mo_r**2 / np.where(vo_r > 0, vo_r, 1.0), 1e-300)), 1.0)
            infl["a"] = np.where(va_r > 0, np.maximum(1.0, ess_floor / np.maximum(ma_r**2 / np.where(va_r > 0, va_r, 1.0), 1e-300)), 1.0)
        rep.append(f"F2 inflation at the reference θ (start point): {int(np.sum(infl['o'] > 1))} orbit and {int(np.sum(infl['a'] > 1))} acceleration bins inflated")
    rep.append(f"N_s prior N({prior[0]:.0f}, {prior[1]:.0f}); ESS floor for variance inflation {ess_floor:g}; real C1 orbits {int(N_o)}, accelerations {int(k_a.sum())}")
    sols, best = [], None
    if args.eval_only:  # #587: evaluate at x0 (no optimisation)
        from types import SimpleNamespace as _SN

        best = _SN(x=x0.copy(), fun=float(nll(x0)))
        sols.append((best.fun, x0.copy()))
        rep.append(f"eval-only at the start θ: -lnL {best.fun:.2f}")
    for s_ in range(0 if args.eval_only else args.n_starts):
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
    fixed_names = {x.split("=")[0] for x in fixes}
    free = [i for i in range(n_par) if names[i] not in fixed_names]  # #587: Hessian over the free parameters only
    for i in (free if not args.eval_only else []):
        for j in (k for k in free if k >= i):
            def ff(di, dj):
                xx = th.copy()
                xx[i] += di
                xx[j] += dj
                return nll(xx)
            H[i, j] = H[j, i] = ((ff(h[i], 0) - 2 * f0 + ff(-h[i], 0)) / h[i] ** 2 if i == j else
                                 (ff(h[i], h[j]) - ff(h[i], -h[j]) - ff(-h[i], h[j]) + ff(-h[i], -h[j])) / (4 * h[i] * h[j]))
    err = np.full(n_par, np.nan)
    if free and not args.eval_only:
        try:
            dg = np.diag(np.linalg.inv(H[np.ix_(free, free)]))
            err[free] = np.sqrt(np.where(dg > 0, dg, np.nan))
        except np.linalg.LinAlgError:
            pass
    w, mu_o, v_o, mu_a, v_a = expected(th)
    s = th[i_ns] * T + floor_o
    Rw, Rd = np.exp(th[i_rw]), np.exp(th[i_rd])
    wc = co_draw_weights(th)
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
    sa_l = ab >= 0
    radn = np.histogram(ns.ra_cols[1], dE)[0].astype(float)
    madn = np.histogram(ns.a_cols[1][sa_l], dE, weights=w[sa_l])[0]
    cadn = np.histogram(co["d_a"][co["in_a"]], dE, weights=wc[co["in_a"]])[0]
    accel_by_d = {"real": radn.tolist(), "mock_lum": madn.tolist(), "mock_co": cadn.tolist(),
                  "orbit_accel_real": (rdn / radn).tolist(), "orbit_accel_mock": ((mdn + cdn + sdn) / (madn + cadn)).tolist(),
                  "accel_mock_over_real": ((madn + cadn) / radn).tolist()}
    # #567 comparisons by distance: M2 inferred the same way on both sides (f = 0, Janssens M1 from M_G0), M1, m_f, P, e, colour
    from darkhunter_pop.janssens_mass import invert_mg_to_mass_array
    from darkhunter_pop.physics_utils import astrometric_mass_function, invert_astrometric_companion_mass

    def m2_inf(mg0, a0, plx, P):
        m1j = invert_mg_to_mass_array(mg0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mf = astrometric_mass_function(a0, plx, P)
            return m1j, mf, invert_astrometric_companion_mass(m1j, mf, 0.0)

    lo_ = so_l
    m1l, mfl, m2l = m2_inf(lx["sys_mg0"][lo_], casc_l[lo_, 17], casc_l[lo_, 0], casc_l[lo_, 10])
    co_o = co["in_o"]
    m1c, mfc, m2c = m2_inf(co["sys_mg0"][co_o], co["a0"][co_o], co["plx"][co_o], co["P"][co_o])
    m1r, mfr, m2r = m2_inf(rmg, r_a0, rx["parallax"], rx["period"])
    qs_ = (0.16, 0.5, 0.84)

    def wq(x, w_):
        ok_ = np.isfinite(x) & (w_ > 0)
        if ok_.sum() < 3:
            return [float("nan")] * 3
        o_ = np.argsort(x[ok_])
        c_ = np.cumsum(w_[ok_][o_]) / w_[ok_].sum()
        return [float(x[ok_][o_][min(np.searchsorted(c_, q), ok_.sum() - 1)]) for q in qs_]

    comp_tab = {}
    var_m = {"M2_inf": (m2l, m2c, m2r), "M1_janssens": (m1l, m1c, m1r), "m_f": (mfl, mfc, mfr), "P": (casc_l[lo_, 10], co["P"][co_o], rx["period"]),
             "e": (casc_l[lo_, 14], co["ecc_fit"][co_o], rx["eccentricity"]), "colour0": (lx["sys_colour0"][lo_], co["sys_colour0"][co_o], rc0)}
    dl, dc, dr = ns.o_cols[1][lo_], co["d_o"][co_o], ns.ro_cols[1]
    Al, Ac = A_l[lo_], A_c[co_o]
    for sub_name, (ml_, mc__, mr_) in {"all_C1": (np.ones(dl.size, bool), np.ones(dc.size, bool), np.ones(dr.size, bool)),
                                      "A>0.5": (Al > 0.5, Ac > 0.5, A_r > 0.5)}.items():
        for lo, hi in zip(dE[:-1], dE[1:]):
            bl, bc, br = ml_ & (dl >= lo) & (dl < hi), mc__ & (dc >= lo) & (dc < hi), mr_ & (dr >= lo) & (dr < hi)
            row = {}
            for vn, (xl, xc, xr) in var_m.items():
                xm = np.concatenate([xl[bl], xc[bc], xr[br]])
                wm = np.concatenate([w[lo_][bl], wc[co_o][bc], spur_w[br]])
                row[vn] = {"mock_q16_50_84": wq(xm, wm), "mock_wd_only": wq(xc[bc], wc[co_o][bc] * (co["comp"][co_o][bc] == 0)), "real_q16_50_84": wq(xr[br], np.ones(br.sum()))}
            comp_tab[f"{sub_name} d{lo:g}-{hi:g}"] = row
    # implied WD M2: the fitted mock's WD truth M2 (A > 0.5, by d) vs the real large-A excess over luminous + spurious (in inferred M2)
    m2e = np.array([0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 3.0])
    implied = {}
    for lo, hi in zip(dE[:-1], dE[1:]):
        bc = (Ac > 0.5) & (dc >= lo) & (dc < hi) & (co["comp"][co_o] == 0)
        bl = (Al > 0.5) & (dl >= lo) & (dl < hi)
        br = (A_r > 0.5) & (dr >= lo) & (dr < hi)
        implied[f"d{lo:g}-{hi:g}"] = {"m2_edges": m2e.tolist(),
                                      "mock_wd_truth_m2": np.histogram(co["m2_true"][co_o][bc], m2e, weights=wc[co_o][bc])[0].tolist(),
                                      "mock_wd_inferred_m2": np.histogram(m2c[bc], m2e, weights=wc[co_o][bc])[0].tolist(),
                                      "real_minus_lum_spur_inferred_m2": (np.histogram(m2r[br], m2e)[0] - np.histogram(m2l[bl], m2e, weights=w[lo_][bl])[0]
                                                                          - np.histogram(m2r[br], m2e, weights=spur_w[br])[0]).tolist()}
    res_extra = {"comparison_by_distance": comp_tab, "implied_wd_m2": implied,
                 "wd_ess_orbits": float(ps.kish_ess(wc[co["in_o"] & (co["comp"] == 0)])), "wd_ess_accels": float(ps.kish_ess(wc[co["in_a"] & (co["comp"] == 0)]))}
    res = dict(names=list(names), theta=th.tolist(), err=err.tolist(), nll=float(best.fun), start_offsets=near, n_near=int(sum(x < 5 for x in near)),
               n_starts=len(sols), at_bound=at_b, N_s=float(th[i_ns]), N_s_pull=float((th[i_ns] - prior[0]) / prior[1]),
               R_wd=float(Rw), R_dark=float(Rd), co_orbits={"wd": float(wc[co["in_o"] & (co["comp"] == 0)].sum()), "dark": float(wc[co["in_o"] & (co["comp"] == 1)].sum())},
               co_accels={"wd": float(wc[co["in_a"] & (co["comp"] == 0)].sum()), "dark": float(wc[co["in_a"] & (co["comp"] == 1)].sum())},
               total=float((mu_o.sum() + s.sum()) / N_o), accel=float((mu_a.sum() + floor_a.sum()) / k_a.sum()),
               d_ratio=((mdn + cdn + sdn) / rdn).tolist(), d_ratio_parts={"lum": (mdn / rdn).tolist(), "co": (cdn / rdn).tolist(), "spurious": (sdn / rdn).tolist()},
               twin_band={"real_twin": kt.tolist(), "real_single": ks.tolist(), "mock_twin": mt.tolist(), "mock_single": ms.tolist()},
               ppc=ppc, f2=f2, depletion=dsum, accel_by_distance=accel_by_d, ess_orbits=float(ps.kish_ess(w[so_l])), n_big_low=int(np.sum(big & (ess_b < args.min_ess))), n_big=int(big.sum()))
    res.update(res_extra)
    rep.append(f"    WD ESS at the fit: orbits {res['wd_ess_orbits']:.0f}, accelerations {res['wd_ess_accels']:.0f}")
    for k_ in ("A>0.5 d0-0.3", "A>0.5 d0.7-1.5", "all_C1 d0-0.3"):
        if k_ in comp_tab:
            rep.append(f"    {k_}: M2_inf mock {[round(x, 2) for x in comp_tab[k_]['M2_inf']['mock_q16_50_84']]} (WD {[round(x, 2) for x in comp_tab[k_]['M2_inf']['mock_wd_only']]}) "
                       f"real {[round(x, 2) for x in comp_tab[k_]['M2_inf']['real_q16_50_84']]}; M1_J mock {[round(x, 2) for x in comp_tab[k_]['M1_janssens']['mock_q16_50_84']]} "
                       f"real {[round(x, 2) for x in comp_tab[k_]['M1_janssens']['real_q16_50_84']]}")
    rep.append(f"{res['n_near']}/{res['n_starts']} starts within Δ(-lnL) < 5 (offsets {near}); -lnL {best.fun:.1f}; at bound: {at_b or 'none'}")
    for nm, v, e_ in zip(names, th, err):
        rep.append(f"    {nm:10s} {v:+10.3f} +- {e_:.3f}")
    rep.append(f"    N_s {th[i_ns]:.0f} vs prior {prior[0]:.0f} +- {prior[1]:.0f} (pull {res['N_s_pull']:+.1f}); (mock+CO+spurious)/real {res['total']:.3f}; accelerations {res['accel']:.3f}")
    rep.append(f"    compact objects: R_wd {Rw:.3g}, R_dark {Rd:.3g} per primary; expected C1 orbits WD {res['co_orbits']['wd']:.0f}, dark {res['co_orbits']['dark']:.0f}; "
               f"accelerations WD {res['co_accels']['wd']:.0f}, dark {res['co_accels']['dark']:.0f}")
    rep.append("    d bins " + " / ".join(f"{a:g}-{b:g}" for a, b in zip(dE[:-1], dE[1:])) + ": total/real " + " / ".join(f"{x:.2f}" for x in res["d_ratio"])
               + "; lum " + " / ".join(f"{x:.2f}" for x in res["d_ratio_parts"]["lum"]) + "; CO " + " / ".join(f"{x:.3f}" for x in res["d_ratio_parts"]["co"])
               + "; spurious " + " / ".join(f"{x:.2f}" for x in res["d_ratio_parts"]["spurious"]))
    rep.append("    orbit:accel by d real " + " / ".join(f"{x:.3f}" for x in accel_by_d["orbit_accel_real"]) + "; mock " + " / ".join(f"{x:.3f}" for x in accel_by_d["orbit_accel_mock"])
               + "; accel mock/real " + " / ".join(f"{x:.2f}" for x in accel_by_d["accel_mock_over_real"]))
    rep.append("    twin/single real " + " ".join(f"{a / max(b, 1):.2f}" for a, b in zip(kt, ks)) + "; mock " + " ".join(f"{a / max(b, 1e-9):.2f}" for a, b in zip(mt, ms)))
    for k_, v_ in ppc.items():
        A_ = v_["A"]
        hi_m = sum(A_["mock_lum_plus_co"][4:]) + sum(A_["spurious"][4:])
        rep.append(f"    PPC {k_}: A > {aE[4]:g}: mock+spur {hi_m:.0f} (CO {sum(A_['co_only'][4:]):.0f}) vs real {sum(A_['real'][4:]):.0f}")
    rep.append("    F2 median mock/real: " + "; ".join(f"{k_} {v_['mock_median']:.2f}/{v_['real_median']:.2f}" for k_, v_ in f2.items()))
    rep.append(f"    ESS orbits {res['ess_orbits']:.0f}; big bins below ESS {args.min_ess:g}: {res['n_big_low']}/{res['n_big']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # #570 plot dump: per-draw observables and weights at the fit (mock luminous, mock WD, real, spurious)
    ra_obs = real_accel_obs(args, ns, parent)
    cc = co["casc"]
    cl = ns.casc
    sa_l = ab >= 0
    wdo = co["in_o"] & (co["comp"] == 0)
    wda = co["in_a"] & (co["comp"] == 0)

    def acc_cols(c_):
        nine = np.isclose(c_[:, 0], -9.0)
        return {"plx_snr": c_[:, 2] / c_[:, 3], "signif": c_[:, 1], "f2": np.where(nine, c_[:, 13], c_[:, 9]), "ruwe": np.where(nine, c_[:, 12], c_[:, 8])}

    al, ac = acc_cols(cl[sa_l]), acc_cols(cc[wda])
    dump = {
        "o_lum_w": w[so_l], "o_lum_g": ns.o_cols[0][so_l], "o_lum_d": ns.o_cols[1][so_l], "o_lum_colour0": lx["sys_colour0"][so_l], "o_lum_mg0": lx["sys_mg0"][so_l],
        "o_lum_P": cl[so_l, 10], "o_lum_e": cl[so_l, 14], "o_lum_A": A_l_fit[so_l], "o_lum_a0snr": cl[so_l, 17] / cl[so_l, 18], "o_lum_plxsnr": cl[so_l, 0] / cl[so_l, 1],
        "o_lum_f2": cl[so_l, 21], "o_lum_ruwe": cl[so_l, 22],
        "o_wd_w": wc[wdo], "o_wd_g": co["g"][wdo], "o_wd_d": co["d_o"][wdo], "o_wd_colour0": co["sys_colour0"][wdo], "o_wd_mg0": co["sys_mg0"][wdo],
        "o_wd_P": cc[wdo, 10], "o_wd_e": cc[wdo, 14], "o_wd_A": jr.photocentre_size(cc[wdo, 17], cc[wdo, 0], cc[wdo, 10]), "o_wd_a0snr": cc[wdo, 17] / cc[wdo, 18],
        "o_wd_plxsnr": cc[wdo, 0] / cc[wdo, 1], "o_wd_f2": cc[wdo, 21], "o_wd_ruwe": cc[wdo, 22],
        "o_real_g": ns.ro_cols[0], "o_real_d": ns.ro_cols[1], "o_real_colour0": rc0, "o_real_mg0": rmg, "o_real_P": 10 ** ns.ro_cols[2], "o_real_e": ns.ro_cols[3],
        "o_real_A": A_r_fit, "o_real_a0snr": 10 ** ns.r_cov["log10_a0_snr"], "o_real_plxsnr": 10 ** ns.r_cov["log10_parallax_snr"], "o_real_f2": rx["goodness_of_fit"],
        "o_real_ruwe": rx["ruwe"] if "ruwe" in rx else np.full(rc0.size, np.nan), "o_spur_w": spur_w,
        "a_lum_w": w[sa_l], "a_lum_g": ns.a_cols[0][sa_l], "a_lum_d": ns.a_cols[1][sa_l], "a_lum_colour0": lx["sys_colour0"][sa_l], "a_lum_mg0": lx["sys_mg0"][sa_l],
        **{f"a_lum_{k}": v for k, v in al.items()},
        "a_wd_w": wc[wda], "a_wd_g": co["g"][wda], "a_wd_d": co["d_a"][wda], "a_wd_colour0": co["sys_colour0"][wda], "a_wd_mg0": co["sys_mg0"][wda],
        **{f"a_wd_{k}": v for k, v in ac.items()},
        "a_real_g": ns.ra_cols[0], "a_real_d": ns.ra_cols[1], **{f"a_real_{k}": v for k, v in ra_obs.items()},
    }
    np.savez_compressed(args.out_dir / "plot_dump.npz", **dump)
    (args.out_dir / "summary.json").write_text(json.dumps({"label": fc.get("label"), "results": {"joint": res}}, indent=1, default=float))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
