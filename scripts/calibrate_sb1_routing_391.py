#!/usr/bin/env python
"""#391 spec §12.34: calibrate (S_min, p_route) of the bright spectroscopic routing on the bright parent's DR3
solution-type rates, and validate on G 9–12. Mock: gens 42–44 draws with deterministic-mixture weights at the published
MdS17 target (θ0), per star (row-weighted parent stars in the same system-G bin). Real: per-star rates by type from the
nss_two_body_orbit snapshots of parent rows. Targets (θ-insensitive ratios): astrometric / SB1 rate and the SB1 share
with P 30–830 d, at G < 9. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ASTRO = ("Orbital", "AstroSpectroSB1", "OrbitalTargetedSearch", "OrbitalTargetedSearchValidated")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single44.yaml"))
    ap.add_argument("--snapshots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import h5py
    from astropy.table import Table, vstack

    sys.path.insert(0, "scripts")
    from rung3_refit_391 import mixture_weights

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop import spectroscopic_routing as sr
    from darkhunter_pop import vbroad_model as vm
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    z = np.load(a.cache, allow_pickle=True)
    w = mixture_weights(z, [int(g) for g in z["gens"]], False)
    keys = ("draw_index", "m1_msun", "m2_msun", "period_days", "eccentricity", "inc_deg", "log10_flux_ratio", "is_dark",
            "phot_g_mean_mag", "sys_bp_rp", "primary_log_teff")
    tr = {k: [] for k in keys}
    acc = []
    for f in a.artifacts:
        with h5py.File(f, "r") as h:
            for k in keys:
                tr[k].append(h["truth"][k][()])
            acc.append(h["outcome/accepted_orbital"][()])
    t = {k: np.concatenate(v) for k, v in tr.items()}
    acc = np.concatenate(acc).astype(bool)
    pos = {int(d): i for i, d in enumerate(t["draw_index"])}
    idx = np.array([pos[int(d)] for d in z["draw_index"]])
    t = {k: v[idx] for k, v in t.items()}
    acc = acc[idx]
    k1, k2 = vm.k_semi_amplitudes(t["m1_msun"], t["m2_msun"], t["period_days"], t["eccentricity"], t["inc_deg"])
    f = np.where(t["is_dark"].astype(bool), 0.0, 10.0 ** np.nan_to_num(t["log10_flux_ratio"], nan=-9.0))
    kobs = sr.k_observed(k1, k2, f)
    grvs = sr.grvs_from_g(t["phot_g_mean_mag"], t["sys_bp_rp"])
    teff = 10.0 ** np.asarray(t["primary_log_teff"], float)
    G, P = t["phot_g_mean_mag"], t["period_days"]
    # per-star denominators: all parent rows (finite, parallax floor, BJ), row-weighted
    g_p = np.asarray(par.columns["phot_g_mean_mag"], float)
    base = np.ones(par.n_rows, bool)
    for k in ("finite_astrometry_photometry", "parallax_floor", "bailer_jones_distance"):
        base &= np.asarray(par.flags[k], bool)
    rw = (np.ones(par.n_rows) if par.row_weight is None else par.row_weight) * par.scale_to_full
    sid = np.asarray(par.columns["source_id"], np.int64)
    snap = vstack([Table.read(s / "table.ecsv") for s in a.snapshots])
    sty = {int(s): str(v) for s, v in zip(snap["source_id"], snap["nss_solution_type"])}
    sp = {int(s): float(v) for s, v in zip(snap["source_id"], np.ma.filled(np.ma.asarray(snap["period"], float), np.nan))}
    uploaded = set(int(s) for s in snap["source_id"])  # rows with any solution; rows without are "none"

    def real_rates(lo, hi):
        m = base & (g_p >= lo) & (g_p < hi)
        n = rw[m].sum()
        ty = np.array([sty.get(int(s), "none") for s in sid[m]])
        pp = np.array([sp.get(int(s), np.nan) for s in sid[m]])
        astro = rw[m][np.isin(ty, ASTRO)].sum() / n
        sb1 = rw[m][ty == "SB1"].sum() / n
        sb1c = rw[m][(ty == "SB1") & (pp >= 30) & (pp <= 830)].sum() / n
        return {"astro": astro, "SB1": sb1, "SB1_P30_830": sb1c, "n_stars": n}

    def mock_rates(lo, hi, cfg_):
        m = (G >= lo) & (G < hi)
        n = rw[base & (g_p >= lo) & (g_p < hi)].sum()
        det = sr.spectroscopic_detected(kobs, grvs, teff, cfg_)
        fac = sr.astrometric_publication_factor(det, cfg_)
        astro = np.sum(w[m & acc] * fac[m & acc]) / n
        sb1_mask = det & ~acc
        sb1 = (np.sum(w[m & sb1_mask]) + np.sum(w[m & det & acc] * (1 - cfg_.p_route))) / n
        cm = m & (P >= 30) & (P <= 830)
        sb1c = (np.sum(w[cm & sb1_mask]) + np.sum(w[cm & det & acc] * (1 - cfg_.p_route))) / n
        return {"astro": float(astro), "SB1": float(sb1), "SB1_P30_830": float(sb1c)}

    r9 = real_rates(5.0, 9.0)
    tgt_ratio, tgt_pfrac = r9["astro"] / r9["SB1"], r9["SB1_P30_830"] / r9["SB1"]
    best = None
    grid = []
    for s_min in (2, 3, 5, 8, 12, 20, 30, 50):
        for p in np.linspace(0.0, 1.0, 21):
            c_ = sr.RoutingConfig(s_min=float(s_min), p_route=float(p))
            mr = mock_rates(5.0, 9.0, c_)
            if mr["SB1"] <= 0:
                continue
            ratio, pfrac = mr["astro"] / mr["SB1"], mr["SB1_P30_830"] / mr["SB1"]
            loss = np.log(ratio / tgt_ratio) ** 2 + np.log(pfrac / tgt_pfrac) ** 2
            grid.append([s_min, float(p), ratio, pfrac, float(loss)])
            if best is None or loss < best[-1]:
                best = [s_min, float(p), ratio, pfrac, float(loss)]
    c_best = sr.RoutingConfig(s_min=float(best[0]), p_route=float(best[1]))
    out = {"target_G<9": {"astro_over_SB1": tgt_ratio, "SB1_P30_830_frac": tgt_pfrac, **r9}, "best": best,
           "mock_G<9_best": mock_rates(5.0, 9.0, c_best), "mock_G<9_no_routing": mock_rates(5.0, 9.0, sr.RoutingConfig(p_route=1.0)),
           "validation": {}}
    for lo, hi in ((9.0, 10.0), (10.0, 11.0), (11.0, 12.0)):
        rr = real_rates(lo, hi)
        mr = mock_rates(lo, hi, c_best)
        out["validation"][f"G{lo:g}-{hi:g}"] = {"real": rr, "mock": mr, "real_astro_over_SB1": rr["astro"] / max(rr["SB1"], 1e-12),
                                               "mock_astro_over_SB1": mr["astro"] / max(mr["SB1"], 1e-12)}
    out["grid"] = grid
    out["uploaded_rows_with_solution"] = len(uploaded)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
