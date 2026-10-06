#!/usr/bin/env python3
"""Evaluate Bayestar19 against Combined19 for the dereddened CMD (#418, MP-Q34, MP-Q38).

Evaluation only: nothing here switches the pipeline's extinction map (spec §11.9). For the
``gaia_source`` parent (Bailer-Jones distances) and the real DR3 orbits (inverse NSS parallax):

* **coverage**: Bayestar19 (Green et al. 2019; ``dustmaps`` 1.0.14, local file, ``fetch()`` is
  never called) gives samples only for dec ≳ −30°; the ``converged`` / ``reliable_dist``
  flags are reported;
* **E(B−V) and its uncertainty**: the median and the sample standard deviation of the
  posterior samples at the row's distance, plus half the change between the r_lo and r_hi
  distances (distance term). Bayestar's SFD-like unit is converted with the configured
  ``dust_maps.maps.green2019.native_to_ebv`` (0.884, #295); the ratio to Combined19 on the same
  rows is measured;
* **the 11% off-grid rate**: the isochrone fit is rerun with Bayestar E(B−V) and its per-star
  σ in place of Combined19 with ε = 0.1;
* **the C0 < 0.35 "blue rows"**: how many stay bluer than 0.35 with Bayestar dereddening.

Writes ``bayestar19_eval.json`` / ``.txt`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop.config_loader import load_config


def bayestar_query(l: np.ndarray, b: np.ndarray, d_pc: dict[str, np.ndarray], data_dir: Path) -> dict[str, np.ndarray]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from dustmaps.config import config as dmcfg

    dmcfg["data_dir"] = str(data_dir)
    from dustmaps.bayestar import BayestarQuery

    q = BayestarQuery(version="bayestar2019", max_samples=None)
    out: dict[str, np.ndarray] = {}
    ok = np.isfinite(l) & np.isfinite(b) & np.isfinite(d_pc["med"]) & (d_pc["med"] > 0)
    n = l.size
    med = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    conv = np.zeros(n, bool)
    rel = np.zeros(n, bool)
    idx = np.flatnonzero(ok)
    for a in range(0, idx.size, 20000):
        ii = idx[a:a + 20000]
        c = SkyCoord(l=l[ii] * u.deg, b=b[ii] * u.deg, distance=d_pc["med"][ii] * u.pc, frame="galactic")
        s, flags = q(c, mode="samples", return_flags=True)
        med[ii] = np.nanmedian(s, axis=-1)
        sd[ii] = np.nanstd(s, axis=-1)
        conv[ii] = flags["converged"]
        rel[ii] = flags["reliable_dist"]
        for key, arr in (("lo", lo), ("hi", hi)):
            dd = np.where(np.isfinite(d_pc[key][ii]) & (d_pc[key][ii] > 0), d_pc[key][ii], d_pc["med"][ii])
            c2 = SkyCoord(l=l[ii] * u.deg, b=b[ii] * u.deg, distance=dd * u.pc, frame="galactic")
            arr[ii] = q(c2, mode="median")
    out.update(median=med, sample_sd=sd, at_r_lo=lo, at_r_hi=hi, converged=conv, reliable_dist=rel)
    return out


def evaluate(label: str, g: np.ndarray, bprp: np.ndarray, l: np.ndarray, b: np.ndarray, dec: np.ndarray,
             r: dict[str, np.ndarray], cfg: Any, model: im.IsochroneMassModel, data_dir: Path,
             to_ebv: float) -> tuple[dict[str, Any], list[str]]:
    rep = [f"=== {label}: {g.size} rows ==="]
    c19 = giants.cmd_for_rows(g, bprp, l, b, r["med"], r["lo"], r["hi"], cfg, None)
    bs = bayestar_query(l, b, r, data_dir)
    for k in ("median", "sample_sd", "at_r_lo", "at_r_hi"):  # native SFD-like unit -> E(B-V) (config, #295)
        bs[k] = bs[k] * to_ebv
    cov = np.isfinite(bs["median"])
    north = dec > -30.0
    res: dict[str, Any] = {"rows": int(g.size), "dec_gt_m30": int(north.sum()),
                           "bayestar_finite": int(cov.sum()),
                           "bayestar_converged_and_reliable": int(np.sum(cov & bs["converged"] & bs["reliable_dist"]))}
    rep.append(f"coverage: dec > -30 {north.mean():.3f}; Bayestar finite {cov.mean():.3f}; converged & reliable_dist {np.mean(cov & bs['converged'] & bs['reliable_dist']):.3f}")
    both = cov & np.isfinite(c19.ebv) & (c19.ebv > 0.01)
    ratio = bs["median"][both] / c19.ebv[both]
    sig_dist = 0.5 * np.abs(bs["at_r_hi"] - bs["at_r_lo"])
    sig_tot = np.sqrt(bs["sample_sd"] ** 2 + sig_dist**2)
    res["ratio_bayestar_over_c19_pct_16_50_84"] = np.nanpercentile(ratio, [16, 50, 84]).tolist()
    res["sigma_E_sample_median"] = float(np.nanmedian(bs["sample_sd"][cov]))
    res["sigma_E_total_median"] = float(np.nanmedian(sig_tot[cov]))
    res["frac_sigma_over_E_median"] = float(np.nanmedian((sig_tot / bs["median"])[cov & (bs["median"] > 0.05)]))
    rep.append(f"E_Bayestar / E_Combined19 (E_C19 > 0.01): 16/50/84% = {np.round(res['ratio_bayestar_over_c19_pct_16_50_84'], 3).tolist()}")
    rep.append(f"Bayestar sigma(E): sample sd median {res['sigma_E_sample_median']:.3f}; with distance term {res['sigma_E_total_median']:.3f}; sigma/E median (E > 0.05) {res['frac_sigma_over_E_median']:.3f}")
    bins = (0, 0.5, 1, 2, 5, 50)
    dk = r["med"] / 1000.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = both & (dk >= lo) & (dk < hi)
        if s.sum() > 50:
            rep.append(f"  d {lo}-{hi} kpc: N={int(s.sum())} ratio median {np.median(bs['median'][s] / c19.ebv[s]):.3f}; sigma/E median {np.nanmedian((sig_tot / bs['median'])[s & (bs['median'] > 0.05)]):.3f}")
    # isochrone fits on the Bayestar-covered rows: Combined19 (eps = config) vs Bayestar (per-star sigma)
    bsc = giants.cmd_for_rows(g, bprp, l, b, r["med"], r["lo"], r["hi"], cfg, None, ebv=np.where(cov, bs["median"], np.nan))
    lk = cfg.isochrone_mass.likelihood
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(bsc.ebv > 0, bsc.a_g / bsc.ebv, 0.0)
        ke = np.where(bsc.ebv > 0, (bprp - bsc.colour0) / bsc.ebv, 0.0)
    smu = np.where(np.isfinite(bsc.sigma_mu), bsc.sigma_mu, 0.0)
    sc = np.sqrt(lk.provisional_colour_floor_mag**2 + (sig_tot * ke) ** 2)
    sm = np.sqrt(smu**2 + lk.provisional_mag_floor_mag**2 + (sig_tot * ka) ** 2)
    p_b = im.posterior_moments(bsc.colour0, bsc.mg0, sc, sm, model.cmap, lk)
    p_c = model.fit(c19.colour0, c19.mg0, sigma_mu=c19.sigma_mu, ebv=c19.ebv, a_g=c19.a_g, e_bp_rp=bprp - c19.colour0)
    s = cov
    for name, p, cmd in (("Combined19 (eps=%.2f)" % lk.provisional_ebv_fractional_sigma, p_c, c19), ("Bayestar19 (per-star sigma)", p_b, bsc)):
        off = np.mean(p.reason[s] == "off_grid")
        blue = np.mean(cmd.colour0[s] < 0.35)
        res[f"off_grid_{name.split()[0]}"] = float(off)
        res[f"blue_C0_lt_0p35_{name.split()[0]}"] = float(blue)
        rep.append(f"  on Bayestar-covered rows, {name}: off-grid {off:.4f}; C0 < 0.35 {blue:.4f}; no_cmd {np.mean(p.reason[s] == 'no_cmd'):.4f}")
    oc = s & (p_c.reason == "off_grid")
    rep.append(f"  Combined19 off-grid rows rescued by Bayestar: {np.mean(p_b.reason[oc] == 'ok'):.3f} of {int(oc.sum())}")
    bl = s & (c19.colour0 < 0.35)
    rep.append(f"  Combined19 blue rows (C0 < 0.35) that Bayestar puts at C0 >= 0.35: {np.mean(bsc.colour0[bl] >= 0.35):.3f} of {int(bl.sum())}")
    both_ok = s & p_b.ok & p_c.ok
    rep.append(f"  M1 Bayestar / Combined19 (both ok): median {np.median(p_b.m1_mean[both_ok] / p_c.m1_mean[both_ok]):.3f}, 16/84 {np.round(np.percentile(p_b.m1_mean[both_ok] / p_c.m1_mean[both_ok], [16, 84]), 3).tolist()}")
    return res, rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    import h5py

    cfg = load_config()
    dm = cfg.sample_selection.dust_maps
    if dm.dustmaps_data_dir is None:
        raise ValueError("sample_selection.dust_maps.dustmaps_data_dir is not set (host-specific)")
    data_dir = Path(dm.dustmaps_data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(cfg.paths.data_root) / data_dir
    to_ebv = float(dm.maps["green2019"].native_to_ebv)  # 0.884 (#295, PI decision)
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    rep: list[str] = ["#418 MP-Q34: Bayestar19 vs Combined19 (evaluation only; the default map is unchanged)", ""]
    with h5py.File(args.parent_dir / "parent.h5", "r") as h:
        cols = {k: h[k][()] for k in ("phot_g_mean_mag", "bp_rp", "l", "b", "dec", "r_med_geo", "r_lo_geo", "r_hi_geo")}
    res, r = evaluate("parent (Bailer-Jones distances)", cols["phot_g_mean_mag"], cols["bp_rp"], cols["l"], cols["b"], cols["dec"],
                      {"med": cols["r_med_geo"], "lo": cols["r_lo_geo"], "hi": cols["r_hi_geo"]}, cfg, model, data_dir, to_ebv)
    out["parent"] = res
    rep += r + [""]
    if args.real_snapshot is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.table import Table

        t = Table.read(args.real_snapshot, format="ascii.ecsv")
        t = t[np.isin(np.asarray(t["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
        f = lambda k: np.ma.filled(np.ma.asarray(t[k], float), np.nan)  # noqa: E731
        plx, ep = f("parallax"), f("parallax_error")
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = {"med": np.where(plx > 0, 1000 / plx, np.nan), "lo": np.where(plx > 0, 1000 / (plx + ep), np.nan),
                  "hi": np.where(plx - ep > 0, 1000 / (plx - ep), np.nan)}
        gal = SkyCoord(ra=f("ra") * u.deg, dec=f("dec") * u.deg).galactic
        res, r = evaluate("real Orbital + AstroSpectroSB1 (inverse NSS parallax)", f("g_mag"), f("bp_mag") - f("rp_mag"),
                          gal.l.deg, gal.b.deg, f("dec"), rr, cfg, model, data_dir, to_ebv)
        out["real"] = res
        rep += r
    (args.out_dir / "bayestar19_eval.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / "bayestar19_eval.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
