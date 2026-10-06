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
        out[name] = {"m1": 10.0**lm1, "m2": 10.0**lm2}
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


def fit(t: Any, cfg: Any) -> im.IsochronePosterior:
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
    return model.fit(cmd.colour0, cmd.mg0, sigma_mu=cmd.sigma_mu, ebv=cmd.ebv, a_g=cmd.a_g, e_bp_rp=f("bp_rp") - cmd.colour0)


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
    args = ap.parse_args(argv)
    from astropy.table import Table

    cfg = load_config()
    out: dict[str, Any] = {}
    rep = ["#418 item 8: isochrone M1 vs dynamical (DEBCat) and asteroseismic (APOKASC-3) masses", ""]
    if args.debcat and args.debcat_xmatch:
        cat = read_debcat(args.debcat)
        t = Table.read(args.debcat_xmatch, format="ascii.ecsv")
        post = fit(t, cfg)
        ids = [str(x) for x in t["cat_id"]]
        m1 = np.array([cat.get(i, {}).get("m1", np.nan) for i in ids])
        m2 = np.array([cat.get(i, {}).get("m2", np.nan) for i in ids])
        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        mp = np.maximum(m1, m2)
        r = post.m1_mean / mp
        out["debcat"] = {"all": stats(r)}
        rep.append(f"DEBCat: {len(t)} cross-matched systems; isochrone ok {int(post.ok.sum())}")
        for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
            s = (q >= lo) & (q < hi)
            out["debcat"][f"q_{lo}_{hi}"] = stats(r[s])
            rep.append(f"  q in [{lo},{hi}): {out['debcat'][f'q_{lo}_{hi}']}")
    if args.apokasc and args.apokasc_xmatch:
        cat = read_apokasc(args.apokasc)
        t = Table.read(args.apokasc_xmatch, format="ascii.ecsv")
        post = fit(t, cfg)
        ids = [str(x) for x in t["cat_id"]]
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
    (args.out_dir / "m1_benchmarks.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / "m1_benchmarks.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
