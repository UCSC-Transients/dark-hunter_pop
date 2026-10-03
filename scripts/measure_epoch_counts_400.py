#!/usr/bin/env python3
"""Measure and decompose gaiamock's epoch-count excess over DR3 (#400, #398).

``gost``
    For every one of gaiamock's 3,072 HEALPix-16 GOST positions, read the DR3-window
    table with ``gaiamock_mod.get_gost_one_position`` (no random 10% drop), group rows
    into FoV transits and record: rows, transits, transit times, and which transits fall
    inside the ESA astrometric gap list and the 25-gap ``scanninglaw`` list.
    Output ``<out>/gost_pixels.npz``.

``compare``
    Join the epoch-count snapshot (``scripts/fetch_epoch_count_sample.py``) to the GOST
    summary of each star's nearest position (the same lookup gaiamock uses) and write
    per-star tables plus the decomposition numbers to ``<out>/epoch_compare.h5`` and
    ``<out>/decomposition.json``.

Usage::

    PY=.venv/bin/python
    $PY scripts/measure_epoch_counts_400.py gost --out output/gate400 --workers 6
    $PY scripts/measure_epoch_counts_400.py compare --out output/gate400 \
        --snapshot data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from darkhunter_pop.epoch_model import (
    GOST_CCD_ROW_COLUMN,
    GOST_TIME_COLUMN,
    EpochModelConfig,
    fov_transit_ids,
    gap_intervals_jd,
    in_gaps,
    n_visibility_periods_from_days,
)

N_PIX: int = 3072
NSIDE: int = 16


def _pixel_summary(args: tuple[int, str, float]) -> dict:
    pix, data_release, split_day = args
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
    import healpy as hp

    g = import_gaiamock_mod(verify=False)
    theta, phi = hp.pix2ang(NSIDE, pix, nest=False)
    ra, dec = float(np.degrees(phi)), float(90.0 - np.degrees(theta))
    tab = g.get_gost_one_position(ra, dec, data_release=data_release)
    jd = np.asarray(tab[GOST_TIME_COLUMN], dtype=np.float64)
    row = np.asarray(tab[GOST_CCD_ROW_COLUMN], dtype=np.int64)
    ids = fov_transit_ids(jd, split_day)
    n_tr = int(ids.max()) + 1 if ids.size else 0
    t_mid = np.array([jd[ids == k].mean() for k in range(n_tr)])
    rows_per = np.bincount(ids, minlength=n_tr)
    ccd_row = np.array([row[ids == k][0] for k in range(n_tr)])
    return {"pix": pix, "ra": ra, "dec": dec, "t_mid": t_mid, "rows": rows_per, "ccd_row": ccd_row}


def cmd_gost(args: argparse.Namespace) -> None:
    split = float(args.split_day)
    jobs = [(p, "dr3", split) for p in range(N_PIX)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_pixel_summary, jobs, chunksize=32))
    lens = np.array([len(r["t_mid"]) for r in res])
    off = np.r_[0, np.cumsum(lens)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "gost_pixels.npz",
        ra=np.array([r["ra"] for r in res]), dec=np.array([r["dec"] for r in res]),
        offsets=off, t_mid=np.concatenate([r["t_mid"] for r in res]),
        rows=np.concatenate([r["rows"] for r in res]),
        ccd_row=np.concatenate([r["ccd_row"] for r in res]),
        split_day=split,
    )
    print(f"wrote {out/'gost_pixels.npz'}: {off[-1]} transits over {N_PIX} positions")


# ---------------------------------------------------------------------------
# compare / calibrate
# ---------------------------------------------------------------------------

#: Gaia DR3 ``gaia_source`` row count, for the density proxy's scale factor.
GAIA_SOURCE_TOTAL_ROWS: int = 1_811_709_771


def _load_snapshot_table(snapshot: Path, name: str) -> dict[str, np.ndarray]:
    import h5py
    import healpy as hp

    with h5py.File(snapshot / f"{name}.h5", "r") as handle:
        d = {k: handle[k][:] for k in handle}
    d["pix"] = hp.ang2pix(NSIDE, np.radians(90.0 - d["dec"]), np.radians(d["ra"]), nest=False)
    return d


def _density_per_deg2(snapshot: Path, d: dict[str, np.ndarray], k_density: int, level: int) -> np.ndarray:
    import h5py
    import healpy as hp

    with h5py.File(snapshot / "density.h5", "r") as handle:
        hpx, n = handle["hpx"][:], handle["n"][:]
    nside = 2**level
    counts = np.zeros(hp.nside2npix(nside))
    counts[hpx] = n
    p = hp.ang2pix(nside, np.radians(90.0 - d["dec"]), np.radians(d["ra"]), nest=True)
    return counts[p] / hp.nside2pixarea(nside, degrees=True) * (GAIA_SOURCE_TOTAL_ROWS / k_density)


def _gap_config(gap_table: Path, window: tuple[float, float] | None) -> EpochModelConfig:
    return EpochModelConfig(
        enabled=True, gap_table=gap_table, apply_gaps=True,
        obmt_reference_rev=1717.6256, obmt_reference_jd_tcb=2457023.75, obmt_rev_per_day=4.0,
        transit_split_day=0.01, transit_loss_g_edges=(0.0, 30.0), transit_loss_prob=(0.0,),
        agis_window_obmt_rev=window,
    )


def _pixel_counts(g: np.lib.npyio.NpzFile, keep: np.ndarray, vis_gap_day: float) -> dict[str, np.ndarray]:
    off = g["offsets"]
    tm, rows = g["t_mid"], g["rows"]
    n = len(off) - 1
    ntr = np.add.reduceat(keep.astype(np.int64), off[:-1]) if n else np.zeros(0)
    nrow = np.add.reduceat(np.where(keep, rows, 0), off[:-1])
    nvis = np.array([
        n_visibility_periods_from_days(tm[off[i]:off[i + 1]][keep[off[i]:off[i + 1]]], vis_gap_day)
        for i in range(n)
    ])
    return {"ntr": ntr, "nrow": nrow, "nvis": nvis}


def _ratio(num: np.ndarray, den: np.ndarray) -> dict[str, float]:
    ok = den > 0
    return {
        "sum_ratio": float(num[ok].sum() / den[ok].sum()),
        "median_ratio": float(np.median(num[ok] / den[ok])),
        "p16": float(np.percentile(num[ok] / den[ok], 16)),
        "p84": float(np.percentile(num[ok] / den[ok], 84)),
        "n": int(ok.sum()),
    }


def _binned_keep(
    amt: np.ndarray, ngap: np.ndarray, x: np.ndarray, edges: np.ndarray, pix: np.ndarray,
    rng: np.random.Generator, n_boot: int,
) -> list[dict[str, float]]:
    """Sum-ratio keep fraction per bin with a bootstrap over GOST positions.

    Stars sharing a GOST position share its transit-count error, so positions (not
    stars) are resampled.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi) & (ngap > 0)
        if s.sum() == 0:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        upix, inv = np.unique(pix[s], return_inverse=True)
        a_p = np.bincount(inv, weights=amt[s])
        n_p = np.bincount(inv, weights=ngap[s])
        boot = []
        for _ in range(n_boot):
            j = rng.integers(0, upix.size, upix.size)
            boot.append(a_p[j].sum() / n_p[j].sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "n": int(s.sum()), "n_positions": int(upix.size),
            "keep": float(a_p.sum() / n_p.sum()),
            "keep_p16": float(np.percentile(boot, 16)), "keep_p84": float(np.percentile(boot, 84)),
        })
    return out


def _density_slope(
    amt: np.ndarray, ngap: np.ndarray, logrho: np.ndarray, gmag: np.ndarray,
    g_edges: np.ndarray, ref_log: float,
) -> list[dict[str, float]]:
    """Per G bin, weighted least-squares slope of the keep fraction against log10 density."""
    out = []
    for lo, hi in zip(g_edges[:-1], g_edges[1:]):
        s = (gmag >= lo) & (gmag < hi) & (ngap > 0) & np.isfinite(logrho)
        if s.sum() < 50:
            out.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum())})
            continue
        y = amt[s] / ngap[s]
        w = ngap[s]
        X = np.column_stack([np.ones(s.sum()), logrho[s] - ref_log])
        beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        out.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()),
                    "keep_at_ref": float(beta[0]), "slope_per_dex": float(beta[1])})
    return out


def cmd_compare(args: argparse.Namespace) -> None:
    import h5py
    import yaml

    out = Path(args.out)
    snapshot = Path(args.snapshot)
    meta = yaml.safe_load((snapshot / "meta.yaml").read_text())
    g = np.load(out / "gost_pixels.npz")
    tm = g["t_mid"]
    window = (float(args.agis_window[0]), float(args.agis_window[1]))
    cfg_gw = _gap_config(Path(args.gap_table), window)
    gaps_and_window = gap_intervals_jd(cfg_gw)
    # the two open-ended intervals gap_intervals_jd adds for the window
    window_only = gaps_and_window[~np.isfinite(gaps_and_window).all(axis=1)]
    keep_sets: dict[str, np.ndarray] = {
        "raw": np.ones(tm.size, dtype=bool),
        "window": ~in_gaps(tm, window_only),
        "esa_gaps": ~in_gaps(tm, gaps_and_window),
    }
    if args.alt_gap_table:
        keep_sets["alt_gaps"] = ~in_gaps(tm, gap_intervals_jd(_gap_config(Path(args.alt_gap_table), window)))
    pixc = {k: _pixel_counts(g, v, args.vis_gap_day) for k, v in keep_sets.items()}
    rows = g["rows"]
    res: dict = {
        "snapshot": str(snapshot), "snapshot_query_date": meta.get("query_date"),
        "gap_table": str(args.gap_table), "alt_gap_table": args.alt_gap_table,
        "agis_window_obmt_rev": list(window),
        "gost": {
            "n_positions": int(len(g["offsets"]) - 1),
            "rows_per_transit_mean": float(rows.mean()),
            "rows_per_transit_counts": {int(k): int(v) for k, v in enumerate(np.bincount(rows)) if v},
            "transit_fraction_removed": {k: float(1 - v.mean()) for k, v in keep_sets.items()},
        },
        "samples": {},
    }
    rng = np.random.default_rng(args.seed)
    g_edges = np.asarray(args.g_edges, dtype=np.float64)
    per_star = {}
    for name in ("random", "nss", "inj390"):
        d = _load_snapshot_table(snapshot, name)
        p = d["pix"]
        amt = d["astrometric_matched_transits"].astype(np.float64)
        vp = d["visibility_periods_used"].astype(np.float64)
        n_obs = d["astrometric_n_obs_al"].astype(np.float64)
        n_good = d["astrometric_n_good_obs_al"].astype(np.float64)
        mt = d["matched_transits"].astype(np.float64)
        rho = _density_per_deg2(snapshot, d, int(meta["k_density"]), int(meta["density_healpix_level_nested"]))
        subsets = {"all": np.ones(p.size, dtype=bool)}
        if name == "random":
            subsets["ruwe_lt_1p4"] = d["ruwe"] < 1.4
        for sub, m in subsets.items():
            key = f"{name}/{sub}"
            r = {}
            for ks, pc in pixc.items():
                r[f"transits_dr3_over_gost_{ks}"] = _ratio(amt[m], pc["ntr"][p][m])
                r[f"nvis_gost_{ks}_minus_dr3"] = {
                    "mean": float(np.mean(pc["nvis"][p][m] - vp[m])),
                    "p16_50_84": [float(x) for x in np.percentile(pc["nvis"][p][m] - vp[m], [16, 50, 84])],
                }
            r["matched_transits_over_gost_raw"] = _ratio(mt[m], pixc["raw"]["ntr"][p][m])
            r["astrometric_over_all_matched_transits"] = _ratio(amt[m], mt[m])
            r["ccd_per_transit"] = {
                "dr3_n_obs_al_per_transit": float(n_obs[m].sum() / amt[m].sum()),
                "dr3_n_good_obs_al_per_transit": float(n_good[m].sum() / amt[m].sum()),
                "dr3_bad_fraction": float(d["astrometric_n_bad_obs_al"][m].sum() / n_obs[m].sum()),
                "gost_rows_per_transit": float(pixc["raw"]["nrow"][p][m].sum() / pixc["raw"]["ntr"][p][m].sum()),
                "gost_af_rows_per_transit": float(
                    (pixc["raw"]["nrow"][p][m] - pixc["raw"]["ntr"][p][m]).sum() / pixc["raw"]["ntr"][p][m].sum()
                ),
            }
            r["ccd_dr3_n_good_over_gaiamock_0p9_rows"] = _ratio(n_good[m], 0.9 * pixc["raw"]["nrow"][p][m])
            ngap = pixc["esa_gaps"]["ntr"][p].astype(np.float64)
            r["keep_vs_g"] = _binned_keep(amt[m], ngap[m], d["phot_g_mean_mag"][m], g_edges, p[m], rng, args.n_boot)
            covs = {
                "log10_density_per_deg2": (np.log10(np.maximum(rho, 1.0)), np.array([2.5, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.5])),
                "abs_b_deg": (np.abs(d["b"]), np.array([0, 5, 10, 20, 30, 45, 60, 90.0])),
                "abs_ecl_lat_deg": (np.abs(d["ecl_lat"]), np.array([0, 10, 20, 30, 45, 60, 75, 90.0])),
                "scan_direction_strength_k1": (d["scan_direction_strength_k1"], np.array([0, 0.1, 0.2, 0.3, 0.4, 1.0])),
            }
            r["keep_vs"] = {
                c: _binned_keep(amt[m], ngap[m], x[m], e, p[m], rng, max(50, args.n_boot // 4))
                for c, (x, e) in covs.items()
            }
            r["density_slope_vs_g"] = _density_slope(
                amt[m], ngap[m], np.log10(np.maximum(rho[m], 1.0)), d["phot_g_mean_mag"][m], g_edges,
                float(args.density_ref_log10),
            )
            res["samples"][key] = r
        per_star[name] = {
            "source_id": d["source_id"], "phot_g_mean_mag": d["phot_g_mean_mag"], "ruwe": d["ruwe"],
            "density_per_deg2": rho, "pix": p, "dr3_astrometric_matched_transits": amt,
            "dr3_visibility_periods_used": vp, "dr3_n_good_obs_al": n_good,
            **{f"gost_ntr_{k}": pc["ntr"][p] for k, pc in pixc.items()},
            **{f"gost_nvis_{k}": pc["nvis"][p] for k, pc in pixc.items()},
        }
    with h5py.File(out / "epoch_compare.h5", "w") as handle:
        for name, cols in per_star.items():
            grp = handle.create_group(name)
            for k, v in cols.items():
                grp.create_dataset(k, data=np.asarray(v))
    (out / "decomposition.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith("transits") or kk.startswith("nvis")}
                      for k, v in res["samples"].items()}, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("gost")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split-day", type=float, default=0.01,
                   help="rows closer than this are one FoV transit (dr3.epoch_model.transit_split_day)")
    p.set_defaults(func=cmd_gost)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--gap-table", default="config/scanning_law/dr3_astrometry_gaps_edr3.csv")
    p.add_argument("--alt-gap-table", default=None, help="e.g. scanninglaw's 25-gap list (ESA CSV header)")
    p.add_argument("--agis-window", nargs=2, type=float, default=[1192.13, 5230.09],
                   help="OBMT revolutions (dr3.epoch_model.agis_window_obmt_rev)")
    p.add_argument("--vis-gap-day", type=float, default=4.0)
    p.add_argument("--g-edges", nargs="+", type=float,
                   default=[3, 11, 13, 15, 16, 17, 17.5, 18, 18.5, 19])
    p.add_argument("--density-ref-log10", type=float, default=4.5)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=400)
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
