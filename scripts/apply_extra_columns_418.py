#!/usr/bin/env python3
"""MP-Q33 + MP-Q34 measurements with the new Gaia columns (#418; spec §11.9).

Uses the snapshots written by ``fetch_isochrone_extra_columns_418.py``:
* **MP-Q34**: per-star BP−RP errors σ_C = 1.0857 sqrt(1/foe_BP² + 1/foe_RP²) added in quadrature
  to the colour floor; reports the off-grid rate and the M1 change against the floor-only fit;
* **MP-Q33**: ``gdr3apcal``-calibrated GSP-Phot [Fe/H] where reliable (``isochrone_mass.feh_likelihood``
  cut), as a likelihood on the [Fe/H]-layered map; reports the reliable fraction, the M1 change,
  and, for the real orbits, isochrone / FLAME by CMD class with and without it.

Writes ``extra_columns_418.json`` / ``.txt`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop.config_loader import load_config


def load_cols(snap: Path) -> dict[str, np.ndarray]:
    with h5py.File(snap / "columns.h5", "r") as h:
        return {k: h[k][()] for k in h.keys()}


def join(sid: np.ndarray, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(cols["source_id"])
    pos = np.clip(np.searchsorted(cols["source_id"], sid, sorter=order), 0, order.size - 1)
    hit = cols["source_id"][order[pos]] == sid
    out = {}
    for k, v in cols.items():
        vv = v[order[pos]]
        if vv.dtype.kind in "fi":
            out[k] = np.where(hit, vv.astype(float), np.nan)
        else:
            out[k] = np.where(hit, np.char.decode(vv.astype("S16")) if vv.dtype.kind == "S" else vv.astype(str), "")
    out["_hit"] = hit
    return out


def run(label: str, sid: np.ndarray, cmd: giants.RowCMD, bprp: np.ndarray, b_deg: np.ndarray, snr: np.ndarray,
        extra: dict[str, np.ndarray], cfg: Any, model: im.IsochroneMassModel, lmap: im.LayeredCmdMap,
        flame: np.ndarray | None, evolved: np.ndarray | None) -> tuple[dict[str, Any], list[str]]:
    lk = cfg.isochrone_mass.likelihood
    e_br = bprp - cmd.colour0
    base = model.fit(cmd.colour0, cmd.mg0, sigma_mu=cmd.sigma_mu, ebv=cmd.ebv, a_g=cmd.a_g, e_bp_rp=e_br)
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(cmd.ebv > 0, cmd.a_g / cmd.ebv, 0.0)
        ke = np.where(cmd.ebv > 0, e_br / cmd.ebv, 0.0)
    sc0, sm = im.likelihood_sigmas(cmd.sigma_mu, cmd.ebv, ka, ke, lk)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_phot = 1.0857 * np.sqrt(1.0 / extra["phot_bp_mean_flux_over_error"] ** 2 + 1.0 / extra["phot_rp_mean_flux_over_error"] ** 2)
    s_phot = np.where(np.isfinite(s_phot), s_phot, 0.0)
    sc = np.sqrt(sc0**2 + s_phot**2)
    phot = im.posterior_moments(cmd.colour0, cmd.mg0, sc, sm, model.cmap, lk)
    rep = [f"=== {label}: {sid.size} rows; extra columns matched {int(extra['_hit'].sum())} ==="]
    res: dict[str, Any] = {"rows": int(sid.size), "matched": int(extra["_hit"].sum())}
    res["sigma_colour_phot_pct_16_50_84"] = np.percentile(s_phot[extra["_hit"]], [16, 50, 84]).tolist()
    res["off_grid_floor_only"] = float(np.mean(base.reason == "off_grid"))
    res["off_grid_with_flux_errors"] = float(np.mean(phot.reason == "off_grid"))
    both = base.ok & phot.ok
    res["m1_ratio_flux_errors_over_floor_pct_16_50_84"] = np.percentile(phot.m1_mean[both] / base.m1_mean[both], [16, 50, 84]).tolist()
    rep.append(f"MP-Q34 per-star BP-RP error (mag) 16/50/84%: {np.round(res['sigma_colour_phot_pct_16_50_84'], 4).tolist()}; "
               f"off-grid {res['off_grid_floor_only']:.4f} -> {res['off_grid_with_flux_errors']:.4f}; "
               f"M1 ratio 16/50/84% {np.round(res['m1_ratio_flux_errors_over_floor_pct_16_50_84'], 4).tolist()}")
    fcfg = cfg.isochrone_mass.feh_likelihood
    feh, rel = im.calibrated_gspphot_feh({k: extra[k] for k in im.GDR3APCAL_COLUMNS}, b_deg, snr, fcfg)
    res["feh_reliable_fraction"] = float(np.mean(rel))
    res["feh_cal_pct_5_50_95"] = np.percentile(feh[rel], [5, 50, 95]).tolist() if rel.any() else []
    fpost = im.posterior_moments_with_feh(cmd.colour0, cmd.mg0, sc, sm, np.where(rel, feh, np.nan),
                                          fcfg.provisional_sigma_dex, lmap, lk)
    ok3 = phot.ok & fpost.ok & rel
    res["m1_ratio_with_feh_over_without_pct_16_50_84"] = np.percentile(fpost.m1_mean[ok3] / phot.m1_mean[ok3], [16, 50, 84]).tolist() if ok3.any() else []
    res["posterior_feh_shift_median"] = float(np.median(fpost.feh_mean[ok3] - phot.feh_mean[ok3])) if ok3.any() else float("nan")
    rep.append(f"MP-Q33 gdr3apcal [Fe/H]: reliable {res['feh_reliable_fraction']:.4f}; calibrated [Fe/H] 5/50/95% "
               f"{np.round(res['feh_cal_pct_5_50_95'], 3).tolist()}; on reliable rows M1 with/without 16/50/84% "
               f"{np.round(res['m1_ratio_with_feh_over_without_pct_16_50_84'], 4).tolist()}; posterior <[Fe/H]> shift {res['posterior_feh_shift_median']:+.3f}")
    if flame is not None and evolved is not None:
        for lab, s in (("dwarfs", ~evolved), ("evolved", evolved)):
            for name, p in (("floor", base), ("flux errors", phot), ("flux errors + [Fe/H]", fpost)):
                ok = s & p.ok & np.isfinite(flame)
                r = p.m1_mean[ok] / flame[ok]
                lr = np.log10(r)
                key = f"flame_{lab}_{name.replace(' ', '_').replace('+', 'plus').replace('[', '').replace(']', '').replace('/', '')}"
                res[key] = {"n": int(ok.sum()), "median": float(np.median(r)), "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr))))}
                rep.append(f"  isochrone/FLAME {lab}, {name}: N={res[key]['n']} median {res[key]['median']:.3f} scatter {res[key]['scatter_dex']:.3f} dex")
            okf = s & fpost.ok & rel & np.isfinite(flame)
            if okf.sum() > 30:
                r = fpost.m1_mean[okf] / flame[okf]
                rep.append(f"  isochrone/FLAME {lab}, [Fe/H]-reliable rows only: N={int(okf.sum())} median {np.median(r):.3f}")
    return res, rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--parent-extra", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, default=None)
    ap.add_argument("--nss-extra", type=Path, default=None)
    ap.add_argument("--flame-snapshot", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config()
    gcfg = giants.load_giants_config("config/population/giants.yaml")
    native = im.load_native_grid(cfg.isochrone_mass, cfg.paths.data_root)
    pts = im.prior_points(native, cfg.isochrone_mass)
    model = im.IsochroneMassModel(cfg=cfg.isochrone_mass, cmap=im.build_cmd_map(pts, cfg.isochrone_mass.cmd_map), grid_key=im.native_grid_key(cfg.isochrone_mass))
    lmap = im.build_layered_cmd_map(pts, cfg.isochrone_mass.cmd_map, native.feh)
    del pts
    out: dict[str, Any] = {}
    rep = ["#418 MP-Q33 / MP-Q34 with the new Gaia columns (spec §11.9)", ""]
    with h5py.File(args.parent_dir / "parent.h5", "r") as h:
        pc = {k: h[k][()] for k in ("source_id", "phot_g_mean_mag", "bp_rp", "l", "b", "parallax", "parallax_error",
                                    "r_med_geo", "r_lo_geo", "r_hi_geo", "ruwe")}
    cmd = giants.cmd_for_rows(pc["phot_g_mean_mag"], pc["bp_rp"], pc["l"], pc["b"], pc["r_med_geo"], pc["r_lo_geo"], pc["r_hi_geo"], cfg, gcfg)
    extra = join(pc["source_id"], load_cols(args.parent_extra))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = pc["parallax"] / pc["parallax_error"]
    res, r = run("parent", pc["source_id"], cmd, pc["bp_rp"], pc["b"], snr, extra, cfg, model, lmap, None, None)
    out["parent"] = res
    rep += r + [""]
    if args.real_snapshot and args.nss_extra:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.table import Table

        t = Table.read(args.real_snapshot, format="ascii.ecsv")
        t = t[np.isin(np.asarray(t["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
        _, first = np.unique(np.asarray(t["source_id"], np.int64), return_index=True)
        t = t[np.sort(first)]
        f = lambda k: np.ma.filled(np.ma.asarray(t[k], float), np.nan)  # noqa: E731
        plx, ep = f("parallax"), f("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rm, rl, rh = (np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + ep), np.nan),
                          np.where(plx - ep > 0, 1000 / (plx - ep), np.nan))
        gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
        bprp = f("bp_mag") - f("rp_mag")
        rcmd = giants.cmd_for_rows(f("g_mag"), bprp, gal.l.deg, gal.b.deg, rm, rl, rh, cfg, gcfg)
        sid = np.asarray(t["source_id"], np.int64)
        flame = None
        if args.flame_snapshot is not None:
            fl = Table.read(args.flame_snapshot, format="ascii.ecsv")
            fs = np.asarray(fl["source_id"], np.int64)
            fm = np.ma.filled(np.ma.asarray(fl["mass_flame"], float), np.nan)
            _, fi = np.unique(fs, return_index=True)
            fs, fm = fs[fi], fm[fi]
            p2 = np.clip(np.searchsorted(fs, sid), 0, fs.size - 1)
            flame = np.where(fs[p2] == sid, fm[p2], np.nan)
        rid = giants.fit_ms_ridge(cmd.mg0, cmd.colour0, snr, gcfg.ridge, ruwe=pc["ruwe"])  # shared ridge (approx.: all parent rows)
        evo = giants.classify_evolved(rcmd.mg0, rcmd.colour0, rcmd.sigma_mu, rid, gcfg.provisional_n_sigma).evolved
        res, r = run("real Orbital + AstroSpectroSB1", sid, rcmd, bprp, gal.b.deg, plx / ep,
                     join(sid, load_cols(args.nss_extra)), cfg, model, lmap, flame, evo)
        out["real"] = res
        rep += r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "extra_columns_418.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / "extra_columns_418.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
