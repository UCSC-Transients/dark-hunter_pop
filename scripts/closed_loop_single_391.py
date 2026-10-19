#!/usr/bin/env python
"""#391 spec §12.33 closed loops (no cascade): (1) looks-single rows given a dark companion reproduce their own
dereddened CMD; (2) equal-mass twins land ~0.75 mag above the ridge; (3) the weighted model population (single
primaries + MdS17 companions) vs the full parent CMD: twin/single occupancy on the MS (C0 0.85–2.0) by distance.
Also P(looks single | M1) under the model (CMD criterion only; RUWE/NSS need the cascade). Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single40.yaml"))
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from darkhunter_pop import giants
    from darkhunter_pop import moe_distefano as mds
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    frag = ps.load_proposal_set_fragment(a.fragment)
    prop = frag.proposal.model_copy(update={"n_draws": a.n})
    par = ps.load_parent_snapshot(a.parent_dir, cfg, prop)
    rd = par.meta["ms_ridge"]
    ridge = giants.MSRidge(colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float),
                           sigma=np.asarray(rd["sigma"], float), n_rows=np.asarray(rd["n_rows"]))
    out: dict = {"usable_rows": int(par.usable.sum()), "rows": int(par.n_rows),
                 "single_flag_pass": {k: float(np.mean(v)) for k, v in par.flags.items() if k.startswith("single_")}}
    t = ps.sample_proposal(par, prop, config=cfg)
    # (1) singles: dark companion
    t1 = {k: v.copy() for k, v in t.items()}
    t1["is_dark"] = np.ones(a.n, bool)
    s1 = ps.apply_system_photometry(t1, par, cfg, prop)
    r = s1["parent_row"]
    g0r, c0r = np.asarray(par.cmd["mg0"], float)[r], np.asarray(par.cmd["colour0"], float)[r]
    d_mg, d_c = s1["sys_mg0"] - g0r, s1["sys_colour0"] - c0r
    out["singles_vs_row"] = {"dMG_p16_50_84": np.nanpercentile(d_mg, [16, 50, 84]).tolist(), "dC_p16_50_84": np.nanpercentile(d_c, [16, 50, 84]).tolist()}
    # (2) twins: q = 1, f = 1
    t2 = {k: v.copy() for k, v in t.items()}
    t2["is_dark"] = np.zeros(a.n, bool)
    t2["m2_msun"] = t2["m1_msun"].copy()
    t2["log10_flux_ratio"] = np.zeros(a.n)
    s2 = ps.apply_system_photometry(t2, par, cfg, prop)
    ms = ~np.asarray(par.is_giant, bool)[r] & (c0r > 0.85) & (c0r < 2.0)
    out["twins_dM_median_MS"] = float(np.nanmedian((s2["sys_mg0"] - s1["sys_mg0"])[ms]))
    # (3) model population vs full parent, twin/single by distance
    s3 = ps.apply_system_photometry(t, par, cfg, prop)
    tg = frag.target_mds17
    evo = ps.evolved_mg0_for_draws(s3, par)
    rel = ps.mist_relation_for_draws(s3, par, cfg)
    ll = ps.mds17_luminous_log_intensity(s3, tg, evolved_mg0_system=evo, relation_log10_f=rel) + par.log_row_weight(r)
    lq = ps.log_q_total_for(s3, par, prop, cfg)
    w = ps.importance_weights(ll, [lq], [a.n], scale_to_full=par.scale_to_full)
    table = mds.load_mds17_table(tg.table_path)
    qg, lpg = np.linspace(0.1, 1.0, 181), np.linspace(0.2, 8.0, 157)
    m1g = np.linspace(0.1, 6.0, 60)
    lam = np.array([np.trapz(mds.f_logp_q03(np.full(lpg.size, m), lpg, table) * np.array(
        [np.trapz(mds.q_density(qg, m, lp, table, m1_interpolation=tg.provisional_m1_interpolation), qg) for lp in lpg]), lpg) for m in m1g])
    lam *= np.array([mds.low_mass_frequency_scale(m, m1_anchor_msun=tg.provisional_low_mass_anchor_msun, m1_zero_msun=tg.provisional_low_mass_zero_msun) for m in m1g])
    p_single = np.exp(-np.interp(par.m1_msun, m1g, lam))
    rw = np.ones(par.n_rows) if par.row_weight is None else par.row_weight
    ridge_dm = lambda mg, c: mg - np.interp(c, ridge.colour, ridge.mag, left=np.nan, right=np.nan)  # noqa: E731
    dm_rows = ridge_dm(np.asarray(par.cmd["mg0"], float), np.asarray(par.cmd["colour0"], float))
    dm_sys = ridge_dm(s3["sys_mg0"], s3["sys_colour0"])
    c_rows, d_rows = np.asarray(par.cmd["colour0"], float), 1.0 / par.truth_parallax_mas
    c_sys, d_sys = s3["sys_colour0"], d_rows[r]
    full = np.ones(par.n_rows, bool)
    for k in ("finite_astrometry_photometry", "parallax_floor", "isochrone_m1", "bailer_jones_distance"):
        full &= np.asarray(par.flags[k], bool)
    full &= np.asarray(par.columns["phot_g_mean_mag"], float) < 19.0
    use = par.usable & full
    de = [0.0, 0.1, 0.2, 0.5, 1.0]
    occ = {"d_edges": de, "parent_full": [], "model": []}
    gs = s3["phot_g_mean_mag"] < 19.0
    for lo, hi in zip(de[:-1], de[1:]):
        def band(dm, c, d, wt, msk, b):
            m = msk & (c > 0.85) & (c < 2.0) & (d >= lo) & (d < hi) & (dm >= b[0]) & (dm < b[1])
            return float(np.sum(wt[m]))
        tw, si = (-0.95, -0.55), (-0.25, 0.25)
        occ["parent_full"].append(band(dm_rows, c_rows, d_rows, rw, full, tw) / max(band(dm_rows, c_rows, d_rows, rw, full, si), 1e-300))
        n_tw = band(dm_sys, c_sys, d_sys, w, gs, tw)
        n_si = band(dm_sys, c_sys, d_sys, w, gs, si) + par.scale_to_full * band(dm_rows, c_rows, d_rows, rw * p_single, use, si)
        n_tw += par.scale_to_full * band(dm_rows, c_rows, d_rows, rw * p_single, use, tw)
        occ["model"].append(n_tw / max(n_si, 1e-300))
    out["twin_over_single"] = occ
    # P(looks single | M1) by the CMD criterion (ΔM >= -0.4 on the MS), systems with companions
    mb = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    m1s = s3["m1_msun"]
    ok = gs & np.isfinite(dm_sys) & ~s3["sys_evolved"]
    out["P_cmd_single_given_companion_by_M1"] = {"edges": mb, "value": [
        float(np.sum(w[ok & (m1s >= x) & (m1s < y) & (dm_sys >= -0.4)]) / max(np.sum(w[ok & (m1s >= x) & (m1s < y)]), 1e-300)) for x, y in zip(mb[:-1], mb[1:])]}
    # unresolved-binary contamination where no ΔM cut applies (evolved rows, colours outside the cut range)
    sc_ = prop.single_catalog
    lo_c, hi_c = sc_.dm_colour_range
    nocut_rows = np.asarray(par.is_giant, bool) | (c_rows < lo_c) | (c_rows > hi_c)
    nocut_sys = nocut_rows[r] & gs
    f_ = np.where(s3["is_dark"], 0.0, 10.0 ** np.nan_to_num(s3["log10_flux_ratio"], nan=-9.0))
    bright = 2.5 * np.log10(1.0 + f_) >= -sc_.dm_min
    n_bin = float(np.sum(w[nocut_sys & bright]))
    n_all = float(np.sum(w[nocut_sys])) + par.scale_to_full * float(np.sum((rw * p_single)[use & nocut_rows]))
    out["nocut_domain"] = {"frac_usable_rows": float(np.mean(nocut_rows[par.usable])),
                           "contamination_frac_brightened_ge_0p4": n_bin / max(n_all, 1e-300)}
    cut_sys = ~nocut_rows[r] & gs
    out["cut_domain_model_frac_brightened_ge_0p4"] = float(np.sum(w[cut_sys & bright])) / max(
        float(np.sum(w[cut_sys])) + par.scale_to_full * float(np.sum((rw * p_single)[use & ~nocut_rows])), 1e-300)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
