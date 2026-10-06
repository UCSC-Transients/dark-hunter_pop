#!/usr/bin/env python3
"""Does gaiamock's 3,072-position GOST grid smooth away visibility-period structure? (#432)

For a seeded random subset of the #428 DR3 snapshot (all solution types, G < 19), compute
FoV transits and visibility periods (gaps > 4 d), after the AGIS window and the ESA DR3
astrometric gaps (``dr3.epoch_model``), from three scanning-law sources:

``exact``
    ``gaiaunlimited.scanninglaw.GaiaScanningLaw('dr3_nominal', gaplist=None).query`` at
    the star's own position. This is the commanded DR3 scanning law, queried per position.
``exact_at_grid``
    The same query at the centre of the star's nearest HEALPix-16 grid position. The
    difference from ``exact`` isolates the grid resolution.
``gaiamock_grid``
    gaiamock_mod's ``get_gost_one_position`` table, as the mock reads it.

Writes ``<out>/grid_resolution.h5`` and prints fractions below 12 by |beta|, |b| and G.
gaiaunlimited's times are TCB at Gaia − 2455197.5 d (its docstring), and are converted to
JD here.

Usage::

    nice -n 10 .venv/bin/python scripts/measure_grid_resolution_432.py --n 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import epoch_model as em  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

GU_TIME_ORIGIN_JD = 2455197.5  # gaiaunlimited query(): JD TCB with origin 2010-01-01T00:00


def counts(t_jd: np.ndarray, gaps: np.ndarray, split_day: float) -> tuple[int, int]:
    t = np.sort(np.asarray(t_jd, dtype=float))
    t = t[~em.in_gaps(t, gaps)]
    if t.size == 0:
        return 0, 0
    ids = em.fov_transit_ids(t, split_day)
    tm = np.bincount(ids, weights=t) / np.bincount(ids)
    return int(tm.size), em.n_visibility_periods_from_days(tm, 4.0)


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=str(P / "data/dr3/gaia_snapshots/20261004T164505Z_visibility_428"))
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=432)
    ap.add_argument("--out", default=str(P / "output/gate432"))
    args = ap.parse_args(argv)

    import healpy as hp
    import gaiaunlimited.scanninglaw as gsl
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    cfg = em.epoch_model_config_from_mapping(load_config().dr3.epoch_model)
    gaps = em.gap_intervals_jd(cfg)
    gm = import_gaiamock_mod()
    sl = gsl.GaiaScanningLaw(version="dr3_nominal", gaplist=None)
    with h5py.File(Path(args.snapshot) / "random_all_params.h5", "r") as f:
        d = {k: f[k][:] for k in f}
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(d["source_id"].size, args.n, replace=False))
    cols = {k: [] for k in ("ntr_exact", "nvis_exact", "ntr_exact_at_grid", "nvis_exact_at_grid",
                            "ntr_gaiamock", "nvis_gaiamock", "pix")}
    cache: dict[int, tuple[int, int, int, int]] = {}
    for n_done, i in enumerate(idx):
        ra, dec = float(d["ra"][i]), float(d["dec"][i])
        r = sl.query(ra, dec)
        te = np.concatenate([np.asarray(x, dtype=float) for x in r]) + GU_TIME_ORIGIN_JD if r else np.array([])
        ne, ve = counts(te, gaps, cfg.transit_split_day)
        pix = int(hp.ang2pix(16, np.radians(90.0 - dec), np.radians(ra)))
        if pix not in cache:
            th, ph = hp.pix2ang(16, pix)
            rag, decg = float(np.degrees(ph)), float(90.0 - np.degrees(th))
            rg = sl.query(rag, decg)
            tg = np.concatenate([np.asarray(x, dtype=float) for x in rg]) + GU_TIME_ORIGIN_JD if rg else np.array([])
            tab = gm.get_gost_one_position(rag, decg, data_release="dr3")
            tgm = np.asarray(tab[em.GOST_TIME_COLUMN], dtype=float)
            cache[pix] = (*counts(tg, gaps, cfg.transit_split_day), *counts(tgm, gaps, cfg.transit_split_day))
        c = cache[pix]
        for k, v in zip(cols, (ne, ve, c[0], c[1], c[2], c[3], pix)):
            cols[k].append(v)
        if (n_done + 1) % 1000 == 0:
            print(f"  {n_done + 1}/{idx.size}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "grid_resolution.h5", "w") as h:
        for k, v in cols.items():
            h.create_dataset(k, data=np.asarray(v))
        for k in ("source_id", "ra", "dec", "l", "b", "ecl_lat", "phot_g_mean_mag", "visibility_periods_used",
                  "astrometric_matched_transits", "astrometric_params_solved"):
            h.create_dataset(k, data=d[k][idx])
        h.attrs["snapshot"] = str(args.snapshot)
        h.attrs["seed"] = args.seed
    nv = {k: np.asarray(cols[f"nvis_{k}"]) for k in ("exact", "exact_at_grid", "gaiamock")}
    beta = np.abs(d["ecl_lat"][idx])
    res: dict = {"n": int(idx.size), "frac_lt12": {k: float(np.mean(v < 12)) for k, v in nv.items()},
                 "dr3_frac_lt12": float(np.mean(d["visibility_periods_used"][idx] < 12))}
    res["by_abs_beta"] = {}
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 90)):
        m = (beta >= lo) & (beta < hi)
        res["by_abs_beta"][f"{lo}-{hi}"] = {"n": int(m.sum()), **{k: float(np.mean(v[m] < 12)) for k, v in nv.items()},
                                            "dr3": float(np.mean(d["visibility_periods_used"][idx][m] < 12)),
                                            "mean_nvis": {k: float(np.mean(v[m])) for k, v in nv.items()}}
    res["mean_nvis"] = {k: float(v.mean()) for k, v in nv.items()}
    res["ntr_ratio_gaiamock_over_exact_at_grid"] = float(np.sum(cols["ntr_gaiamock"]) / np.sum(cols["ntr_exact_at_grid"]))
    res["ntr_ratio_exact_over_exact_at_grid"] = float(np.sum(cols["ntr_exact"]) / np.sum(cols["ntr_exact_at_grid"]))
    (out / "grid_resolution.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
