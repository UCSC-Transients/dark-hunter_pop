#!/usr/bin/env python3
"""Calibrate the isochrone-M1 prior for evolved stars against APOKASC-3 (#418, Ryan 2026-10-07).

Two prior knobs (config ``isochrone_mass``): ``age.provisional_age_power`` γ, an extra
(age / 1 Gyr)^γ on the constant-SFR age weight, and ``provisional_cheb_weight`` ρ, a multiplier on
core-He-burning (red-clump) points. The prior points are built once; each (γ, ρ) reweights them,
rebuilds the CMD map and refits the APOKASC-3 giants (Gaia DR3 cross-match, pipeline CMD with
Combined19 × 0.884). Half the stars (even KIC) train: (γ, ρ) minimizes the squared median
log10(M̂ / M_seis) of RGB and RC; the odd-KIC half is the held-out test. Reports both halves by
evolutionary state and by seismic-mass and M̂ bins. Writes ``giant_prior_calibration.json`` / ``.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_isochrone_m1_benchmarks_418 import read_apokasc  # noqa: E402

from darkhunter_pop import giants  # noqa: E402
from darkhunter_pop import isochrone_mass as im  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

BINS = ((0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10.0))


def summarize(m_hat: np.ndarray, m_seis: np.ndarray, evol: np.ndarray, sel: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    lr = np.log10(m_hat / m_seis)
    for st in ("RGB", "RC"):
        s = sel & (evol == st) & np.isfinite(lr)
        out[st] = {"n": int(s.sum()), "median_ratio": float(10 ** np.median(lr[s])),
                   "scatter_dex": float(1.4826 * np.median(np.abs(lr[s] - np.median(lr[s]))))}
        for lo, hi in BINS:
            b = s & (m_seis >= lo) & (m_seis < hi)
            out[st][f"seis_{lo}_{hi}"] = float(10 ** np.median(lr[b])) if b.sum() > 20 else None
            c = s & (m_hat >= lo) & (m_hat < hi)
            out[st][f"mhat_{lo}_{hi}"] = float(10 ** np.median(-lr[c])) if c.sum() > 20 else None  # seis/M̂ by M̂
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apokasc", type=Path, required=True)
    ap.add_argument("--xmatch", type=Path, required=True)
    ap.add_argument("--powers", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--cheb", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.25, 1.6])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    cfg = load_config()
    icfg = cfg.isochrone_mass.model_copy(update={"provisional_cheb_weight": 1.0,
                                                 "age": cfg.isochrone_mass.age.model_copy(update={"provisional_age_power": 0.0})})
    cat = read_apokasc(args.apokasc)
    t = Table.read(args.xmatch, format="ascii.ecsv")
    ids = np.array([str(x).replace("KIC", "").strip() for x in t["cat_id"]])
    f = lambda k: np.ma.filled(np.ma.asarray(t[k], float), np.nan)  # noqa: E731
    plx, ep = f("parallax"), f("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rm, rl, rh = (np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + ep), np.nan),
                      np.where(plx - ep > 0, 1000 / (plx - ep), np.nan))
    gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
    bprp = f("bp_rp")
    cmd = giants.cmd_for_rows(f("phot_g_mean_mag"), bprp, gal.l.deg, gal.b.deg, rm, rl, rh, cfg, None)
    e_br = bprp - cmd.colour0
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(cmd.ebv > 0, cmd.a_g / cmd.ebv, 0.0)
        ke = np.where(cmd.ebv > 0, e_br / cmd.ebv, 0.0)
    sc, sm = im.likelihood_sigmas(cmd.sigma_mu, cmd.ebv, ka, ke, icfg.likelihood)
    m_seis = np.array([cat.get(i, {}).get("mass", np.nan) for i in ids])
    evol = np.array([cat.get(i, {}).get("evol", "") for i in ids])
    kic = np.array([int(i) if i.isdigit() else -1 for i in ids])
    train, test = (kic % 2 == 0), (kic % 2 == 1)
    native = im.load_native_grid(icfg, cfg.paths.data_root)
    pts = im.prior_points(native, icfg)
    base_w = pts.weight.copy()
    age_fac = 10.0 ** (pts.log_age - 9.0)
    cheb = (pts.values["phase"] >= 2.5) & (pts.values["phase"] < 3.5)
    grid: list[dict[str, Any]] = []
    best: tuple[float, float, float] | None = None
    for g in args.powers:
        for r in args.cheb:
            w = base_w * age_fac**g * np.where(cheb, r, 1.0)
            pw = im.PriorPoints(weight=w / w.sum(), colour=pts.colour, mg=pts.mg, feh=pts.feh, log_age=pts.log_age, values=pts.values)
            post = im.posterior_moments(cmd.colour0, cmd.mg0, sc, sm, im.build_cmd_map(pw, icfg.cmd_map), icfg.likelihood)
            s_tr = summarize(post.m1_mean, m_seis, evol, train & post.ok)
            obj = float(np.log10(s_tr["RGB"]["median_ratio"]) ** 2 + np.log10(s_tr["RC"]["median_ratio"]) ** 2)
            grid.append({"age_power": g, "cheb_weight": r, "objective": obj, "train": s_tr,
                         "test": summarize(post.m1_mean, m_seis, evol, test & post.ok)})
            print(f"gamma={g:+.2f} rho={r:.2f}: train RGB {s_tr['RGB']['median_ratio']:.3f} RC {s_tr['RC']['median_ratio']:.3f} obj {obj:.5f}", flush=True)
            if best is None or obj < best[0]:
                best = (obj, g, r)
    assert best is not None
    chosen = next(x for x in grid if x["age_power"] == best[1] and x["cheb_weight"] == best[2])
    base = next(x for x in grid if x["age_power"] == 0.0 and x["cheb_weight"] == 1.0)
    rep = ["#418 giant prior calibration against APOKASC-3 (train: even KIC, test: odd KIC)", "",
           f"chosen: age_power = {best[1]}, cheb_weight = {best[2]} (train objective {best[0]:.5f})", ""]
    for name, x in (("before (gamma 0, rho 1)", base), ("after", chosen)):
        for half in ("train", "test"):
            for st in ("RGB", "RC"):
                d = x[half][st]
                rep.append(f"{name} {half} {st}: N={d['n']} M_hat/M_seis median {d['median_ratio']:.3f} scatter {d['scatter_dex']:.3f} dex; "
                           f"by seismic mass {[None if d[f'seis_{lo}_{hi}'] is None else round(d[f'seis_{lo}_{hi}'], 3) for lo, hi in BINS]}; "
                           f"M_seis/M_hat by M_hat {[None if d[f'mhat_{lo}_{hi}'] is None else round(d[f'mhat_{lo}_{hi}'], 3) for lo, hi in BINS]}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "giant_prior_calibration.json").write_text(json.dumps({"grid": grid, "chosen": {"age_power": best[1], "cheb_weight": best[2]}}, indent=1, default=float))
    (args.out_dir / "giant_prior_calibration.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
