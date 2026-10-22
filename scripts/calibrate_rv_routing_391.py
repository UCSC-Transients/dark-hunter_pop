#!/usr/bin/env python
"""#391 spec §12.34 v2: (1) DR3 per-epoch RV noise σ_RV(G_RVS, Teff) from RV-constant bright parent stars
(rv_method_used = 1, rv_chisq_pvalue > 0.01, rv_renormalised_gof < 4; σ_ep = (radial_velocity_error − 0.11)·√(2N/π),
Katz et al. 2023); (2) the bright routing re-calibrated with it: SB1 detection when N_RV ≥ 10 and K_obs²/(2σ²) > g_thr
(N_RV from the parent row), an SB1 period efficiency min(1, (P_b/P)^2), and p_route for systems detected both ways.
Fitted on G < 9 (astrometric/SB1 and the SB1 share at P 30–830 d), validated on G 9–12. Mock weights: gens 42–44 at θ0.
Writes JSON (σ table, grid, best, validation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ASTRO = ("Orbital", "AstroSpectroSB1", "OrbitalTargetedSearch", "OrbitalTargetedSearchValidated")
GR_E = np.arange(4.0, 12.51, 0.5)
TE_E = np.array([3000, 4500, 5500, 6500, 7500, 9000, 15000.0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rv-stats", type=Path, required=True)
    ap.add_argument("--nss", type=Path, nargs="+", required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single44.yaml"))
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

    rv = Table.read(a.rv_stats / "table.ecsv")
    col = lambda n: np.ma.filled(np.ma.asarray(rv[n], float), np.nan)  # noqa: E731
    par_rows = np.asarray(rv["sample"]).astype(str) == "parent"
    N = col("rv_nb_transits")
    const = par_rows & (col("rv_method_used") == 1) & (col("rv_chisq_pvalue") > 0.01) & (col("rv_renormalised_gof") < 4) & (N >= 6)
    eps = col("radial_velocity_error")
    sig_ep = np.sqrt(np.clip(eps**2 - 0.11**2, 0.0, None) * 2 * N / np.pi)  # Katz et al. 2023: eps² = (π/2N) σ² + 0.11²
    gr, te = col("grvs_mag"), col("rv_template_teff")
    tab = np.full((GR_E.size - 1, TE_E.size - 1), np.nan)
    cnt = np.zeros_like(tab, int)
    for i in range(GR_E.size - 1):
        for j in range(TE_E.size - 1):
            m = const & (gr >= GR_E[i]) & (gr < GR_E[i + 1]) & (te >= TE_E[j]) & (te < TE_E[j + 1]) & np.isfinite(sig_ep) & (sig_ep > 0)
            cnt[i, j] = int(m.sum())
            if m.sum() >= 15:
                tab[i, j] = float(np.median(sig_ep[m]))
    # fill gaps along G_RVS (log-linear) per Teff column, then across Teff
    for j in range(tab.shape[1]):
        ok = np.isfinite(tab[:, j])
        if ok.sum() >= 2:
            x = 0.5 * (GR_E[1:] + GR_E[:-1])
            tab[:, j] = 10 ** np.interp(x, x[ok], np.log10(tab[ok, j]))
    for i in range(tab.shape[0]):
        ok = np.isfinite(tab[i])
        if ok.any():
            tab[i] = np.where(ok, tab[i], np.interp(np.arange(tab.shape[1]), np.flatnonzero(ok), tab[i, ok]))
    nmed = {int(r): float(np.nanmedian(N[par_rows & (np.floor(gr) == r)])) for r in range(4, 13) if np.any(par_rows & (np.floor(gr) == r))}

    def sigma_lookup(g_rvs, teff):
        i = np.clip(np.digitize(g_rvs, GR_E) - 1, 0, GR_E.size - 2)
        j = np.clip(np.digitize(teff, TE_E) - 1, 0, TE_E.size - 2)
        return tab[i, j]

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    z = np.load(a.cache, allow_pickle=True)
    w = mixture_weights(z, [int(g) for g in z["gens"]], False)
    keys = ("draw_index", "parent_row", "m1_msun", "m2_msun", "period_days", "eccentricity", "inc_deg", "log10_flux_ratio",
            "is_dark", "phot_g_mean_mag", "sys_bp_rp", "primary_log_teff")
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
    fr = np.where(t["is_dark"].astype(bool), 0.0, 10.0 ** np.nan_to_num(t["log10_flux_ratio"], nan=-9.0))
    kobs = sr.k_observed(k1, k2, fr)
    grvs = sr.grvs_from_g(t["phot_g_mean_mag"], t["sys_bp_rp"])
    teff = 10.0 ** np.asarray(t["primary_log_teff"], float)
    sig = sigma_lookup(grvs, teff)
    psid = np.asarray(par.columns["source_id"], np.int64)
    nmap = {int(s): float(n) for s, n, p in zip(rv["source_id"], N, par_rows) if p}
    nrv = np.array([nmap.get(int(psid[r]), np.nan) for r in t["parent_row"]])
    nrv = np.where(np.isfinite(nrv), nrv, np.array([nmed.get(int(np.clip(np.floor(g), 4, 12)), 0.0) for g in grvs]))
    G, P = t["phot_g_mean_mag"], t["period_days"]
    g_p = np.asarray(par.columns["phot_g_mean_mag"], float)
    base = np.ones(par.n_rows, bool)
    for k in ("finite_astrometry_photometry", "parallax_floor", "bailer_jones_distance"):
        base &= np.asarray(par.flags[k], bool)
    rw = (np.ones(par.n_rows) if par.row_weight is None else par.row_weight) * par.scale_to_full
    nss = vstack([Table.read(s / "table.ecsv") for s in a.nss])
    sty = {int(s): str(v) for s, v in zip(nss["source_id"], nss["nss_solution_type"])}
    spd = {int(s): float(v) for s, v in zip(nss["source_id"], np.ma.filled(np.ma.asarray(nss["period"], float), np.nan))}

    def real_rates(lo, hi):
        m = base & (g_p >= lo) & (g_p < hi)
        n = rw[m].sum()
        ty = np.array([sty.get(int(s), "none") for s in psid[m]])
        pp = np.array([spd.get(int(s), np.nan) for s in psid[m]])
        return {"astro": rw[m][np.isin(ty, ASTRO)].sum() / n, "SB1": rw[m][ty == "SB1"].sum() / n,
                "SB1_P30_830": rw[m][(ty == "SB1") & (pp >= 30) & (pp <= 830)].sum() / n}

    def mock_rates(lo, hi, g_thr, p_b, p_route):
        m = (G >= lo) & (G < hi)
        n = rw[base & (g_p >= lo) & (g_p < hi)].sum()
        det = (grvs <= 12) & (nrv >= 10) & (kobs**2 / (2 * sig**2) > g_thr)
        eff = np.where(det, np.minimum(1.0, (p_b / P) ** 2), 0.0)  # SB1 orbit-solution efficiency
        sb1_w = w * eff * np.where(acc, 1 - p_route, 1.0)
        astro_w = w * np.where(acc, 1 - eff * (1 - p_route), 0.0)
        c = m & (P >= 30) & (P <= 830)
        return {"astro": float(astro_w[m].sum() / n), "SB1": float(sb1_w[m].sum() / n), "SB1_P30_830": float(sb1_w[c].sum() / n)}

    r9 = real_rates(5.0, 9.0)
    tr_, tp_ = r9["astro"] / r9["SB1"], r9["SB1_P30_830"] / r9["SB1"]
    grid, best = [], None
    for g_thr in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
        for p_b in (100, 200, 300, 600, 1000, 1500, 2500, 1e9):
            for p in np.linspace(0, 1, 11):
                mr = mock_rates(5.0, 9.0, g_thr, p_b, p)
                if mr["SB1"] <= 0 or mr["astro"] <= 0:
                    continue
                loss = np.log(mr["astro"] / mr["SB1"] / tr_) ** 2 + np.log(mr["SB1_P30_830"] / mr["SB1"] / tp_) ** 2
                row = [g_thr, p_b, float(p), mr["astro"] / mr["SB1"], mr["SB1_P30_830"] / mr["SB1"], float(loss)]
                grid.append(row)
                if best is None or loss < best[-1]:
                    best = row
    out = {"sigma_ep_table": {"grvs_edges": GR_E.tolist(), "teff_edges": TE_E.tolist(), "sigma_kms": tab.tolist(), "n": cnt.tolist()},
           "n_rv_median_by_grvs": nmed, "target_G<9": {**r9, "astro_over_SB1": tr_, "SB1_P30_830_frac": tp_}, "best": best,
           "mock_G<9_best": mock_rates(5.0, 9.0, *best[:3]), "mock_G<9_no_routing": mock_rates(5.0, 9.0, best[0], best[1], 1.0),
           "validation": {}}
    for lo, hi in ((9.0, 10.0), (10.0, 11.0), (11.0, 12.0)):
        rr, mr = real_rates(lo, hi), mock_rates(lo, hi, *best[:3])
        out["validation"][f"G{lo:g}-{hi:g}"] = {"real": rr, "mock": mr, "real_A_over_SB1": rr["astro"] / max(rr["SB1"], 1e-12),
                                               "mock_A_over_SB1": mr["astro"] / max(mr["SB1"], 1e-12)}
    out["grid"] = grid
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "sigma_ep_table")}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
