#!/usr/bin/env python3
"""#391 old-draw reuse check: recompute gens 23-32 generation-time inputs under the current model.

For every stored draw (re-truthed artifacts for gens 29/31/32), keep its random content (parent row,
proposal q, P, e, orientation, flux-ratio noise) and recompute what the current model would have
fed gaiamock: the deblended M1 on the Teff-corrected MIST grid (same posterior-draw seeds), the
flux ratio (the proposal's MIST coeval relation re-centred on the corrected row M1 / isochrone), and
the §12.25 resolved-companion flag. Writes per-draw deltas (npz) and a per-cell report on the
108-bin grid. Analysis only: no new draws, nothing used in a fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ARTS = ["restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5", "restart_gen26_topup.h5",
        "restart_gen27_evolved.h5", "restart_gen28_recentred.h5", "restart_gen29_topup_retruth.h5", "restart_gen30_topup.h5",
        "restart_gen31_batch1_retruth.h5", "restart_gen32_batch2_retruth.h5"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--resim-fragment", type=Path, default=Path("config/population/proposal_set_resim34.yaml"))
    ap.add_argument("--fit-config", type=Path, default=Path("config/population/rung3_compliant108_v6a.yaml"))
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--tol-dlnm1", type=float, default=0.01)
    ap.add_argument("--tol-dlogf", type=float, default=0.02)
    args = ap.parse_args(argv)
    from darkhunter_pop import isochrone_mass as im, proposal_set as ps, resolved_companions as rcm
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.giants import classify_parent, load_giants_config, parent_row_cmd

    cfg_new = load_config()
    assert cfg_new.isochrone_mass.lowmass_correction.enabled
    lc = cfg_new.isochrone_mass.lowmass_correction.model_copy(update={"enabled": False})
    cfg_old = cfg_new.model_copy(update={"isochrone_mass": cfg_new.isochrone_mass.model_copy(update={"lowmass_correction": lc})})
    prop0 = ps.load_proposal_set_fragment(Path("config/population/proposal_set_restart_gen26.yaml")).proposal
    par_old = ps.load_parent_snapshot(args.parent_dir, cfg_old, prop0)
    par_new = ps.load_parent_snapshot(args.parent_dir, cfg_new, prop0)
    nat_old = im.load_native_grid(cfg_old.isochrone_mass, cfg_old.paths.data_root)
    nat_new = im.load_native_grid(cfg_new.isochrone_mass, cfg_new.paths.data_root)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    pcls, _ = classify_parent(par_new, cfg_new, gcfg, cmd=parent_row_cmd(par_new, cfg_new, gcfg))
    evo_row = np.asarray(pcls.classified & pcls.evolved)
    rc = ps.load_proposal_set_fragment(args.resim_fragment).proposal.resolved_companions
    cat = rcm.load_elbadry2021(args.parent_dir.parents[2] / rc.snapshot, rc.sha256)
    tab = rcm.build_tables(cat, rc.theta_edges_arcsec, rc.dg_edges, r_chance_max=rc.r_chance_max, g2_max=rc.g2_max,
                           ref_theta_arcsec=tuple(rc.ref_theta_arcsec), ipd_fmp_max=2.0, ipd_gof_max=0.1, min_pairs=rc.min_pairs)
    del cat
    m1row_new_all = np.asarray(par_new.isochrone["m1_mean"], float)
    out = {k: [] for k in ("gen", "draw_index", "dlnm1", "dlogf", "resolved", "accepted", "accel", "g", "d", "lp", "e", "cls", "m1_old")}
    for art in ARTS:
        t, o, a = ps.read_proposal_artifact(args.artifact_dir / art)
        prop = ps.ProposalConfig.model_validate_json(a["proposal_config_json"])
        r = np.asarray(t["parent_row"], np.int64)
        m1row_old = np.asarray(t["m1_row_msun"], float)
        m2prop_old = np.asarray(t["m2_proposal_msun"], float)
        q = m2prop_old / m1row_old
        m1row_new = m1row_new_all[r]
        base = {k: np.asarray(v) for k, v in t.items() if k not in ("m1_hat_msun", "iso_phase", "iso_feh", "iso_log_age", "deblend_chi2", "m1_row_msun", "m2_proposal_msun", "m1_deblend_dark_msun")}
        base["m1_msun"], base["m2_msun"] = m1row_new, q * m1row_new
        new = ps.apply_posterior_deblending(base, par_new, cfg_new, prop, native=nat_new)
        dlnm1 = np.log(np.asarray(new["m1_msun"], float) / np.asarray(t["m1_msun"], float))
        # flux-ratio relation at the row level (the proposal's mist_coeval_row centre), old vs new model
        ro = ps.mist_relation_for_draws({"parent_row": r, "m1_msun": m1row_old, "m2_msun": m2prop_old}, par_old, cfg_old, native=nat_old)
        rn = ps.mist_relation_for_draws({"parent_row": r, "m1_msun": m1row_new, "m2_msun": q * m1row_new}, par_new, cfg_new, native=nat_new)
        dark = np.asarray(t["is_dark"], bool)
        dlogf = np.where(dark, 0.0, rn - ro)
        f = np.asarray(t["flux_ratio"], float)
        mtot = np.asarray(t["m1_msun"], float) + np.asarray(t["m2_msun"], float)
        theta = rc.projection_factor * np.cbrt(mtot * (np.asarray(t["period_days"], float) / 365.25) ** 2) * np.asarray(t["parallax_mas"], float) / 1000
        with np.errstate(divide="ignore"):
            dg = np.where(dark | ~(f > 0), np.inf, -2.5 * np.log10(np.where(f > 0, f, 1.0)))
        res = ps.resolved_flags(theta, dg, tab, prop.base_seed, ps.PROPOSAL_RNG_STREAM_BASE + prop.generation)
        cas = np.asarray(o["cascade"], float)
        acc = np.asarray(o["accepted_orbital"], bool)
        for k, v in (("gen", np.full(r.size, prop.generation)), ("draw_index", t["draw_index"]), ("dlnm1", dlnm1), ("dlogf", dlogf),
                     ("resolved", res), ("accepted", acc), ("accel", np.asarray(o["published_acceleration"], bool)),
                     ("g", np.asarray(t["phot_g_mean_mag"], float)), ("d", np.where(acc, 1 / np.where(cas[:, 0] > 0, cas[:, 0], np.nan), 1 / np.asarray(t["parallax_mas"], float))),
                     ("lp", np.where(acc, np.log10(np.where(cas[:, 10] > 0, cas[:, 10], np.nan)), np.log10(np.asarray(t["period_days"], float)))),
                     ("e", np.where(acc, cas[:, 14], np.asarray(t["eccentricity"], float))), ("cls", evo_row[r]), ("m1_old", np.asarray(t["m1_msun"], float))):
            out[k].append(np.asarray(v))
        print(f"{art}: n {r.size}; |dlnM1| median {np.nanmedian(np.abs(dlnm1)):.4f}; |dlogf| median {np.nanmedian(np.abs(dlogf)):.4f}; resolved {res.sum()}", flush=True)
        del t, o, new, base
    Z = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(args.out_npz, **Z)
    ok = (np.abs(Z["dlnm1"]) < args.tol_dlnm1) & (np.abs(Z["dlogf"]) < args.tol_dlogf) & ~Z["resolved"]
    fc = yaml.safe_load(args.fit_config.read_text())["rung3"]
    E = [np.asarray(fc["orbit_bins"][k], float) for k in ("g_mag", "distance_kpc", "log10_period_days", "eccentricity")]
    idx = [np.clip(np.digitize(Z[k], ed) - 1, -1, ed.size - 2) for k, ed in zip(("g", "d", "lp", "e"), E)]
    inb = np.all([(i >= 0) & np.isfinite(Z[k]) & (Z[k] >= ed[0]) & (Z[k] <= ed[-1]) for i, k, ed in zip(idx, ("g", "d", "lp", "e"), E)], axis=0)
    rep = [f"#391 old-draw reuse check: tolerances |Δ ln M1| < {args.tol_dlnm1}, |Δ log10 f| < {args.tol_dlogf}, not resolved", "",
           f"all draws: {Z['dlnm1'].size}; qualify {ok.mean():.3f}; accepted orbits qualify {ok[Z['accepted']].mean():.3f} (n {Z['accepted'].sum()}); accelerations qualify {ok[Z['accel']].mean():.3f}",
           f"|Δ ln M1| quantiles 50/84/95/99: {np.nanpercentile(np.abs(Z['dlnm1']), [50, 84, 95, 99]).round(4).tolist()}",
           f"|Δ log10 f| (luminous) quantiles 50/84/95/99: {np.nanpercentile(np.abs(Z['dlogf'][Z['dlogf'] != 0]), [50, 84, 95, 99]).round(4).tolist()}",
           f"resolved (new flag): {int(Z['resolved'].sum())} draws, of which accepted orbits {int((Z['resolved'] & Z['accepted']).sum())}",
           "", "by old M1 (all draws, fraction qualifying): " + "; ".join(f"[{a},{b}) {ok[(Z['m1_old'] >= a) & (Z['m1_old'] < b)].mean():.3f}" for a, b in ((0.1, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.2), (1.2, 3), (3, 50))),
           "", "per 108-grid cell (accepted orbits): G d logP e class: n_acc, frac qualifying"]
    for gi in range(E[0].size - 1):
        for di in range(E[1].size - 1):
            for pi in range(E[2].size - 1):
                for ei in range(E[3].size - 1):
                    for ci in (0, 1):
                        s = inb & Z["accepted"] & (idx[0] == gi) & (idx[1] == di) & (idx[2] == pi) & (idx[3] == ei) & (Z["cls"] == bool(ci))
                        if s.sum():
                            rep.append(f"  G{E[0][gi]:g}-{E[0][gi+1]:g} d{E[1][di]:g}-{E[1][di+1]:g} lP{E[2][pi]:g}-{E[2][pi+1]:.2f} e{E[3][ei]:g}-{E[3][ei+1]:g} {'evo' if ci else 'dwarf'}: {s.sum()}, {ok[s].mean():.3f}")
    args.out_report.write_text("\n".join(rep) + "\n")
    print("\n".join(rep[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
