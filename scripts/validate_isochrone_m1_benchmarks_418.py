#!/usr/bin/env python3
"""Isochrone M1 against dynamical (DEBCat) and asteroseismic (APOKASC-3) masses (#418, item 8).

Inputs:
* the catalogues as snapshotted under ``data/external_catalogs/`` (DEBCat ``debs.dat``,
  APOKASC-3 ``table4.dat.gz``; byte layouts from their ReadMe files);
* a Gaia DR3 cross-match per catalogue (ECSV): ``cat_id`` (DEBCat system name / KIC),
  ``source_id``, ``phot_g_mean_mag``, ``bp_rp``, ``parallax``, ``parallax_error``, ``ra``,
  ``dec``. The cross-match is not made here (it needs archive / Simbad queries).

The isochrone M1 uses the pipeline CMD (Combined19 + Babusiaux law at 1/ϖ) and the configured
priors. DEBCat systems are blends of two stars, so the system fit is compared with the primary's
dynamical mass and the residual is binned by the dynamical mass ratio q (the blended-light bias
of spec §11.3 should grow toward q = 1). APOKASC-3 giants (RGB / RC) test the IMF + SFR prior
for evolved stars (the 0.78 isochrone / FLAME ratio). Writes ``m1_benchmarks.json`` / ``.txt``.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop.config_loader import load_config


def read_debcat(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        name = p[0]
        try:
            lm1, lm2 = float(p[6]), float(p[8])
        except (IndexError, ValueError):
            continue
        try:
            moh = float(p[26])
        except (IndexError, ValueError):
            moh = -9.99
        out[name] = {"m1": 10.0**lm1, "m2": 10.0**lm2, "moh": moh if moh > -9.0 else float("nan")}
    return out


def read_apokasc(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            kic = line[0:8].strip()
            try:
                mass = float(line[126:136])
                emass = float(line[137:147])
            except ValueError:
                continue
            if mass < -9000:
                continue
            out[kic] = {"mass": mass, "e_mass": emass, "evol": line[9:16].strip()}
    return out


def fit(t: Any, cfg: Any) -> tuple[im.IsochronePosterior, Any, np.ndarray, np.ndarray]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    f = lambda k: np.ma.filled(np.ma.asarray(t[k], float), np.nan)  # noqa: E731
    plx, ep = f("parallax"), f("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        r_med = np.where(plx > 0, 1000.0 / plx, np.nan)
        r_lo = np.where(plx > 0, 1000.0 / (plx + ep), np.nan)
        r_hi = np.where(plx - ep > 0, 1000.0 / (plx - ep), np.nan)
    gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
    cmd = giants.cmd_for_rows(f("phot_g_mean_mag"), f("bp_rp"), gal.l.deg, gal.b.deg, r_med, r_lo, r_hi, cfg, None)
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    e_br = f("bp_rp") - cmd.colour0
    post = model.fit(cmd.colour0, cmd.mg0, sigma_mu=cmd.sigma_mu, ebv=cmd.ebv, a_g=cmd.a_g, e_bp_rp=e_br)
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(cmd.ebv > 0, cmd.a_g / cmd.ebv, 0.0)
        ke = np.where(cmd.ebv > 0, e_br / cmd.ebv, 0.0)
    sc, sm = im.likelihood_sigmas(cmd.sigma_mu, cmd.ebv, ka, ke, cfg.isochrone_mass.likelihood)
    return post, cmd, sc, sm


def deblend_known_q(post: im.IsochronePosterior, cmd: Any, sc: np.ndarray, sm: np.ndarray, q: np.ndarray,
                    native: im.NativeGrid) -> np.ndarray:
    """MP-Q36 with the dynamical q: on the row's posterior-mean coeval isochrone, solve for the
    primary whose combined light with a coeval MS companion at q·M1 (MP-Q40 MIST flux ratio,
    iterated twice) matches the observed system."""
    from darkhunter_pop import malmquist_cmd as mc

    out = np.full(q.size, np.nan)
    for i in np.flatnonzero(post.ok & np.isfinite(q)):
        iso = im.isochrone_at(native, float(post.feh_mean[i]), float(post.log_age_mean[i]))
        if iso["star_mass"].size < 2:
            continue
        ms = mc.ms_colours(native, float(post.feh_mean[i]), float(post.log_age_mean[i]))
        m1 = float(post.m1_mean[i])
        for _ in range(2):
            lf = float(ms.log10_flux_ratio(m1, q[i] * m1))
            mm, _ = im.deblend_primary_mass(iso, cmd.colour0[i:i + 1], cmd.mg0[i:i + 1], sc[i:i + 1], sm[i:i + 1],
                                            q[i:i + 1], np.array([lf]), evolved=np.array([post.p_evolved[i] > 0.5]))
            m1 = float(mm[0])
        out[i] = m1
    return out


def stats(r: np.ndarray) -> dict[str, float]:
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return {"n": 0}
    lr = np.log10(r)
    return {"n": int(r.size), "median_ratio": float(np.median(r)), "bias_dex": float(np.median(lr)),
            "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr))))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--debcat", type=Path)
    ap.add_argument("--debcat-xmatch", type=Path)
    ap.add_argument("--apokasc", type=Path)
    ap.add_argument("--apokasc-xmatch", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feh-source", choices=("none", "catalog", "gspphot"), default="none")
    ap.add_argument("--catalog-feh-sigma", type=float, default=0.1)
    ap.add_argument("--gspphot-snapshot", type=Path, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    from astropy.table import Table

    cfg = load_config()
    out: dict[str, Any] = {}
    rep = ["#418 item 8: isochrone M1 vs dynamical (DEBCat) and asteroseismic (APOKASC-3) masses", ""]
    if args.debcat and args.debcat_xmatch:
        cat = read_debcat(args.debcat)
        t = Table.read(args.debcat_xmatch, format="ascii.ecsv")
        t = t[np.asarray(t["source_id"], np.int64) > 0]
        _, first = np.unique(np.asarray(t["cat_id"]).astype(str), return_index=True)
        t = t[np.sort(first)]
        _, first = np.unique(np.asarray(t["source_id"], np.int64), return_index=True)
        t = t[np.sort(first)]
        post, cmd, sc, sm = fit(t, cfg)
        ids = [str(x) for x in t["cat_id"]]
        m1 = np.array([cat.get(i, {}).get("m1", np.nan) for i in ids])
        m2 = np.array([cat.get(i, {}).get("m2", np.nan) for i in ids])
        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        mp = np.maximum(m1, m2)
        native = im.load_native_grid(cfg.isochrone_mass, cfg.paths.data_root)
        deb = deblend_known_q(post, cmd, sc, sm, q, native)
        # #418 item 2: the same with a [Fe/H] likelihood. ``--feh-source catalog`` uses DEBCat's own
        # spectroscopic [M/H] (an upper bound on what any [Fe/H] measurement can add);
        # ``--feh-source gspphot`` uses gdr3apcal-calibrated GSP-Phot [M/H] from ``--gspphot-snapshot``.
        out.setdefault("debcat", {})
        if args.feh_source != "none":
            moh = np.array([cat.get(i, {}).get("moh", np.nan) for i in ids])
            if args.feh_source == "gspphot":
                import h5py

                with h5py.File(args.gspphot_snapshot / "columns.h5", "r") as h:
                    cols = {k: h[k][()] for k in h.keys()}
                sidx = np.asarray(t["source_id"], np.int64)
                order = np.argsort(cols["source_id"])
                pos = np.clip(np.searchsorted(cols["source_id"], sidx, sorter=order), 0, order.size - 1)
                hit = cols["source_id"][order[pos]] == sidx
                sel = {k: (np.where(hit, v[order[pos]].astype(float), np.nan) if v.dtype.kind in "fi"
                           else np.char.decode(v[order[pos]].astype("S16"))) for k, v in cols.items()}
                from astropy.coordinates import SkyCoord
                import astropy.units as u

                bb = SkyCoord(ra=np.asarray(t["ra"], float) * u.deg, dec=np.asarray(t["dec"], float) * u.deg).galactic.b.deg
                plx_snr = np.asarray(t["parallax"], float) / np.asarray(t["parallax_error"], float)
                feh_meas, rel = im.calibrated_gspphot_feh({k: sel[k] for k in im.GDR3APCAL_COLUMNS}, bb, plx_snr,
                                                          cfg.isochrone_mass.feh_likelihood)
                feh_meas = np.where(rel, feh_meas, np.nan)
                ok_cmp = rel & np.isfinite(moh)
                if ok_cmp.sum() > 5:
                    dd = feh_meas[ok_cmp] - moh[ok_cmp]
                    out["debcat"]["gspphot_minus_spec"] = {"n": int(ok_cmp.sum()), "offset": float(np.median(dd)),
                                                           "sigma_robust": float(1.4826 * np.median(np.abs(dd - np.median(dd))))}
                    rep.append(f"  gdr3apcal [Fe/H] - DEBCat [M/H]: {out['debcat']['gspphot_minus_spec']}")
                feh_sig = cfg.isochrone_mass.feh_likelihood.provisional_sigma_dex
            else:
                feh_meas, feh_sig = moh, args.catalog_feh_sigma
            pts = im.prior_points(native, cfg.isochrone_mass)
            lmap = im.build_layered_cmd_map(pts, cfg.isochrone_mass.cmd_map, native.feh)
            del pts
            fpost = im.posterior_moments_with_feh(cmd.colour0, cmd.mg0, sc, sm, feh_meas, feh_sig, lmap, cfg.isochrone_mass.likelihood)
            debf = deblend_known_q(fpost, cmd, sc, sm, q, native)
            has = np.isfinite(feh_meas) & np.isfinite(deb) & np.isfinite(debf)
            out["debcat"][f"feh_{args.feh_source}"] = {"rows_with_feh": int(has.sum()), "without": stats(deb[has] / mp[has]),
                                                       "with": stats(debf[has] / mp[has])}
            rep.append(f"  [Fe/H] ({args.feh_source}) on {int(has.sum())} systems: deblended without {stats(deb[has] / mp[has])} | with {stats(debf[has] / mp[has])}")
        r = post.m1_mean / mp
        rd = deb / mp
        out["debcat"].update({"blended_all": stats(r), "deblended_all": stats(rd)})
        rep.append(f"DEBCat: {len(t)} systems with Gaia photometry (deduplicated); isochrone ok {int(post.ok.sum())}; "
                   f"catalogue match {int(np.isfinite(m1).sum())}")
        rep.append(f"  all: blended-CMD M1 / M1_dyn {stats(r)}; deblended with the dynamical q {stats(rd)}")
        for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            s = (q >= lo) & (q < hi)
            out["debcat"][f"q_{lo}_{hi}"] = {"blended": stats(r[s]), "deblended": stats(rd[s])}
            rep.append(f"  q in [{lo},{hi}): blended {stats(r[s])} | deblended {stats(rd[s])}")
        for lo, hi in ((0.0, 0.7), (0.7, 1.3), (1.3, 3.0), (3.0, 100.0)):
            s = (mp >= lo) & (mp < hi)
            rep.append(f"  M1_dyn in [{lo},{hi}): deblended {stats(rd[s])}")
    if args.apokasc and args.apokasc_xmatch:
        cat = read_apokasc(args.apokasc)
        t = Table.read(args.apokasc_xmatch, format="ascii.ecsv")
        post, _, _, _ = fit(t, cfg)
        ids = [str(x).replace("KIC", "").strip() for x in t["cat_id"]]
        ms = np.array([cat.get(i, {}).get("mass", np.nan) for i in ids])
        ev = np.array([cat.get(i, {}).get("evol", "") for i in ids])
        r = post.m1_mean / ms
        out["apokasc3"] = {"all": stats(r)}
        rep.append(f"APOKASC-3: {len(t)} cross-matched giants; isochrone ok {int(post.ok.sum())}")
        for st in np.unique(ev):
            out["apokasc3"][st or "unknown"] = stats(r[ev == st])
            rep.append(f"  {st or 'unknown'}: {out['apokasc3'][st or 'unknown']}")
        for lo, hi in ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10.0)):
            s = (ms >= lo) & (ms < hi)
            out["apokasc3"][f"mass_{lo}_{hi}"] = stats(r[s])
            rep.append(f"  seismic mass [{lo},{hi}): {out['apokasc3'][f'mass_{lo}_{hi}']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_{args.tag}" if args.tag else ""
    (args.out_dir / f"m1_benchmarks{suf}.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / f"m1_benchmarks{suf}.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
