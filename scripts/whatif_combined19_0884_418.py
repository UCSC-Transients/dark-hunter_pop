#!/usr/bin/env python3
"""What-if: Combined19 E(B−V) × 0.884 (#418; the unit question is Ryan's, nothing is changed).

Bayestar19 / Combined19 has median exactly 0.884 in the north (``bayestar19_eval.txt``), i.e.
Combined19 returns Bayestar's native SFD-like unit there, while #295 converts Bayestar with 0.884.
This measures, for the parent and the real orbits, the off-grid rate, the C0 < 0.35 rows and the
isochrone M1 shift if Combined19 E(B−V) were multiplied by ``--scale`` (default 0.884).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop.config_loader import load_config


def compare(label: str, g, bprp, l, b, rm, rl, rh, cfg, model, scale):  # type: ignore[no-untyped-def]
    a = giants.cmd_for_rows(g, bprp, l, b, rm, rl, rh, cfg, None)
    s = giants.cmd_for_rows(g, bprp, l, b, rm, rl, rh, cfg, None, ebv=a.ebv * scale)
    pa = model.fit(a.colour0, a.mg0, sigma_mu=a.sigma_mu, ebv=a.ebv, a_g=a.a_g, e_bp_rp=bprp - a.colour0)
    ps = model.fit(s.colour0, s.mg0, sigma_mu=s.sigma_mu, ebv=s.ebv, a_g=s.a_g, e_bp_rp=bprp - s.colour0)
    ok = pa.ok & ps.ok
    r = ps.m1_mean[ok] / pa.m1_mean[ok]
    res = {"rows": int(g.size), "off_grid": [float(np.mean(pa.reason == "off_grid")), float(np.mean(ps.reason == "off_grid"))],
           "blue_C0_lt_0p35": [float(np.mean(a.colour0 < 0.35)), float(np.mean(s.colour0 < 0.35))],
           "m1_ratio_pct_16_50_84": np.percentile(r, [16, 50, 84]).tolist()}
    line = (f"{label}: off-grid {res['off_grid'][0]:.4f} -> {res['off_grid'][1]:.4f}; C0 < 0.35 "
            f"{res['blue_C0_lt_0p35'][0]:.4f} -> {res['blue_C0_lt_0p35'][1]:.4f}; M1(x{scale}) / M1 16/50/84% "
            f"{np.round(res['m1_ratio_pct_16_50_84'], 4).tolist()}")
    return res, line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=0.884)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    cfg = load_config()
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    out, rep = {}, [f"#418 what-if: Combined19 E(B-V) x {args.scale} (units unchanged in the pipeline)"]
    with h5py.File(args.parent_dir / "parent.h5", "r") as h:
        c = {k: h[k][()] for k in ("phot_g_mean_mag", "bp_rp", "l", "b", "r_med_geo", "r_lo_geo", "r_hi_geo")}
    out["parent"], line = compare("parent", c["phot_g_mean_mag"], c["bp_rp"], c["l"], c["b"], c["r_med_geo"], c["r_lo_geo"], c["r_hi_geo"], cfg, model, args.scale)
    rep.append(line)
    t = Table.read(args.real_snapshot, format="ascii.ecsv")
    t = t[np.isin(np.asarray(t["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
    f = lambda k: np.ma.filled(np.ma.asarray(t[k], float), np.nan)  # noqa: E731
    plx, ep = f("parallax"), f("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rm, rl, rh = np.where(plx > 0, 1000 / plx, np.nan), np.where(plx > 0, 1000 / (plx + ep), np.nan), np.where(plx - ep > 0, 1000 / (plx - ep), np.nan)
    gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
    out["real"], line = compare("real Orbital + AstroSpectroSB1", f("g_mag"), f("bp_mag") - f("rp_mag"), gal.l.deg, gal.b.deg, rm, rl, rh, cfg, model, args.scale)
    rep.append(line)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "whatif_c19_0884.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "whatif_c19_0884.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
