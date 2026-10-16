#!/usr/bin/env python3
"""Spec §12.27: build the interim combined weight cache (old gens restricted to R + new gens).

Step ``region``: per artifact, compute and store the versioned R column (sidecar
``reuseR_<stem>_<key>.npz``): old draws are recomputed under the current model, new draws under the
old one (same seeds). Step ``cache``: concatenate old and new draws; target log-intensity and CMD
weight per draw under the draw's own model; every generation's log q at every draw under that
generation's model (cross-evaluation); write a cache in the ``rung3_refit_391.build_cache`` format
plus ``in_R`` and ``restricted_gens`` so ``mixture_weights`` applies the restricted-support mixture.
Run the steps one at a time (memory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OLD = ["restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5", "restart_gen26_topup.h5",
       "restart_gen27_evolved.h5", "restart_gen28_recentred.h5", "restart_gen29_topup_retruth.h5", "restart_gen30_topup.h5",
       "restart_gen31_batch1_retruth.h5", "restart_gen32_batch2_retruth.h5"]


def models(parent_dir: Path):
    from darkhunter_pop import isochrone_mass as im, proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg_new = load_config()
    assert cfg_new.isochrone_mass.lowmass_correction.enabled, "current model must carry the §12.24 correction"
    lc = cfg_new.isochrone_mass.lowmass_correction.model_copy(update={"enabled": False})
    cfg_old = cfg_new.model_copy(update={"isochrone_mass": cfg_new.isochrone_mass.model_copy(update={"lowmass_correction": lc})})
    prop0 = ps.load_proposal_set_fragment(Path("config/population/proposal_set_restart_gen26.yaml")).proposal
    out = {}
    for k, c in (("old", cfg_old), ("new", cfg_new)):
        out[k] = (c, ps.load_parent_snapshot(parent_dir, c, prop0), im.load_native_grid(c.isochrone_mass, c.paths.data_root))
    key = hashlib.sha256(json.dumps([im.config_key(cfg_old.isochrone_mass), im.config_key(cfg_new.isochrone_mass)]).encode()).hexdigest()[:10]
    return out, key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("region", "cache"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--new", nargs="+", required=True, help="new-generation artifact basenames")
    ap.add_argument("--resim-fragment", type=Path, default=Path("config/population/proposal_set_resim35.yaml"))
    ap.add_argument("--tol-dlnm1", type=float, default=0.01)
    ap.add_argument("--tol-dlogf", type=float, default=0.02)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--cmd-malmquist", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    args = ap.parse_args(argv)
    from darkhunter_pop import proposal_set as ps, resolved_companions as rcm, reuse_region as rr

    M, key = models(args.parent_dir)
    tolkey = f"{key}_m{args.tol_dlnm1:g}_f{args.tol_dlogf:g}"
    side = lambda art: args.artifact_dir / f"reuseR_{Path(art).stem}_{tolkey}.npz"  # noqa: E731
    if args.step == "region":
        rc = ps.load_proposal_set_fragment(args.resim_fragment).proposal.resolved_companions
        cat = rcm.load_elbadry2021(args.parent_dir.parents[2] / rc.snapshot, rc.sha256)
        tab = rcm.build_tables(cat, rc.theta_edges_arcsec, rc.dg_edges, r_chance_max=rc.r_chance_max, g2_max=rc.g2_max,
                               ref_theta_arcsec=tuple(rc.ref_theta_arcsec), ipd_fmp_max=2.0, ipd_gof_max=0.1, min_pairs=rc.min_pairs)
        del cat
        for art, own, other in [(a, "old", "new") for a in OLD] + [(a, "new", "old") for a in args.new]:
            if side(art).exists():
                print(f"{art}: R exists ({side(art).name})", flush=True)
                continue
            t, _, a = ps.read_proposal_artifact(args.artifact_dir / art)
            prop = ps.ProposalConfig.model_validate_json(a["proposal_config_json"])
            co, po, no = M[own]
            ct, pt, nt = M[other]
            rec = rr.recompute_under(t, prop, pt, ct, nt)
            rel_own = rr.row_relation(t, po, co, no)
            m_new, m_old = (rec["m1"], np.asarray(t["m1_msun"], float)) if own == "old" else (np.asarray(t["m1_msun"], float), rec["m1"])
            dlnm1 = np.log(m_new / m_old)
            dlogf = (rec["relation_row"] - rel_own) * (1 if own == "old" else -1)
            if "resolved_companion" in t:
                res = np.asarray(t["resolved_companion"], bool)
            else:
                f = np.asarray(t["flux_ratio"], float)
                mt = np.asarray(t["m1_msun"], float) + np.asarray(t["m2_msun"], float)
                th = rc.projection_factor * np.cbrt(mt * (np.asarray(t["period_days"], float) / 365.25) ** 2) * np.asarray(t["parallax_mas"], float) / 1000
                with np.errstate(divide="ignore"):
                    dg = np.where(np.asarray(t["is_dark"], bool) | ~(f > 0), np.inf, -2.5 * np.log10(np.where(f > 0, f, 1.0)))
                res = ps.resolved_flags(th, dg, tab, prop.base_seed, ps.PROPOSAL_RNG_STREAM_BASE + prop.generation)
            inR = rr.region_mask(dlnm1, dlogf, res, t["is_dark"], tol_dlnm1=args.tol_dlnm1, tol_dlogf=args.tol_dlogf)
            np.savez(side(art), draw_index=t["draw_index"], in_R=inR, dlnm1=dlnm1, dlogf=dlogf, resolved=res,
                     meta=json.dumps({"key": tolkey, "own_model": own, "artifact": art, "spec": "§12.27"}))
            print(f"{art} ({own}): in R {inR.mean():.3f} of {inR.size}", flush=True)
        return 0
    # step cache
    from darkhunter_pop import malmquist_cmd as mc

    if args.cache is None:
        raise SystemExit("--cache is required for the cache step")
    okeys = ("accepted_orbital", "published_acceleration", "cascade", "cpu_seconds")
    blocks = []
    for own, arts in (("old", OLD), ("new", args.new)):
        parts = [ps.read_proposal_artifact(args.artifact_dir / a) for a in arts]
        blocks.append((own, arts, parts))
    common = set.intersection(*(set(p[0]) for _, _, parts in blocks for p in parts))
    tg = ps.MdS17TargetConfig.model_validate(json.loads(blocks[0][2][0][2]["provenance_json"])["target_mds17_json"])
    cm = mc.load_cmd_malmquist_config(args.cmd_malmquist)
    T, O, LL, LW, INR, OLDD, GCS = [], [], [], [], [], [], []
    all_gcs = [ps.ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for _, _, parts in blocks for p in parts]
    need_dark = rr.needs_dark_deblend(all_gcs)  # any generation's q needs m1_deblend_dark at EVERY draw (cross-evaluation)
    for own, arts, parts in blocks:
        cfg, parent, _ = M[own]
        t = {k: np.concatenate([np.asarray(p[0][k]) for p in parts]) for k in parts[0][0] if k in common}
        gcs = [ps.ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for p in parts]
        if need_dark:
            ps.ensure_m1_deblend_dark(t, parent, cfg, gcs)
        evo = ps.evolved_mg0_for_draws(t, parent)
        rel = ps.mist_relation_for_draws(t, parent, cfg)
        LL.append(ps.mds17_luminous_log_intensity(t, tg, evolved_mg0_system=evo, relation_log10_f=rel))
        LW.append(ps.malmquist_cmd_log_weight(t, parent, tg, cm, cfg)[0])
        INR.append(np.concatenate([np.load(side(a))["in_R"] for a in arts]))
        OLDD.append(np.full(t["draw_index"].size, own == "old"))
        T.append(t)
        O.append({k: np.concatenate([np.asarray(p[1][k]) for p in parts]) for k in okeys})
        GCS += [(own, g) for g in gcs]
        print(f"{own}: {t['draw_index'].size} draws, in R {INR[-1].mean():.3f}", flush=True)
    keys = set(T[0]) & set(T[1])
    t_all = {k: np.concatenate([T[0][k], T[1][k]]) for k in keys}
    lqs = np.vstack([ps.log_q_total_for(t_all, M[own][1], g, M[own][0]) for own, g in GCS])
    o_all = {k: np.concatenate([O[0][k], O[1][k]]) for k in okeys}
    scale = M["new"][1].scale_to_full
    np.savez(args.cache, ll=np.concatenate(LL), lw=np.concatenate(LW), lqs=lqs, n_per_gen=np.array([g.n_draws for _, g in GCS]),
             gens=np.array([g.generation for _, g in GCS]), scale=scale, generation=t_all["generation"], draw_index=t_all["draw_index"],
             m1_msun=t_all["m1_msun"], m2_msun=t_all["m2_msun"], period_days=t_all["period_days"], eccentricity=t_all["eccentricity"],
             phot_g_mean_mag=t_all["phot_g_mean_mag"], accepted=o_all["accepted_orbital"], published_acceleration=o_all["published_acceleration"],
             cascade=o_all["cascade"], cpu_seconds=o_all["cpu_seconds"], in_R=np.concatenate(INR), draw_is_old=np.concatenate(OLDD),
             restricted_gens=np.array([g.generation for own, g in GCS if own == "old"]), reuse_key=tolkey)
    print(f"cached {t_all['draw_index'].size} draws (old restricted to R) -> {args.cache}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
