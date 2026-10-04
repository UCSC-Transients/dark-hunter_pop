#!/usr/bin/env python3
"""Per-star visibility-period structure at the exact positions, for the #432 calibration.

For every star of the #428 random snapshot (all solution types, G < 19) and of the #400
NSS snapshot, query the commanded DR3 scanning law at the star's own position
(``gaiamock``-independent: ``gaiaunlimited`` ``dr3_nominal``, no gaps), remove the AGIS
window and the ESA DR3 astrometric gaps (``dr3.epoch_model``), group into FoV transits
and visibility periods (gaps > 4 d), and store per star the number of transits in each
visibility period (ragged: ``vp_ntr`` with ``vp_offsets``), plus the DR3 columns needed
by the fit. Output ``<out>/vp_structure.h5``.

The grid-resolution test (``measure_grid_resolution_432.py``) showed the exact-position and
grid-position visibility-period distributions agree (mean N_vis 20.97 vs 21.00 on 8,000
stars), so the calibration conditions on the exact positions and the mock applies the
result to its grid positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import epoch_model as em  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

GU_TIME_ORIGIN_JD = 2455197.5


def vp_sizes(t_jd: np.ndarray, gaps: np.ndarray, split_day: float) -> np.ndarray:
    """Transits per visibility period after the gaps (time order)."""
    t = np.sort(np.asarray(t_jd, dtype=float))
    t = t[~em.in_gaps(t, gaps)]
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    ids = em.fov_transit_ids(t, split_day)
    tm = np.bincount(ids, weights=t) / np.bincount(ids)
    vp = np.r_[0, np.cumsum(np.diff(tm) > 4.0)]
    return np.bincount(vp).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--random", default=str(P / "data/dr3/gaia_snapshots/20261004T164505Z_visibility_428/random_all_params.h5"))
    ap.add_argument("--nss", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400/nss.h5"))
    ap.add_argument("--out", default=str(P / "output/gate432"))
    args = ap.parse_args(argv)

    import gaiaunlimited.scanninglaw as gsl

    cfg = em.epoch_model_config_from_mapping(load_config().dr3.epoch_model)
    gaps = em.gap_intervals_jd(cfg)
    sl = gsl.GaiaScanningLaw(version="dr3_nominal", gaplist=None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_cols = ("source_id", "ra", "dec", "l", "b", "ecl_lat", "phot_g_mean_mag", "visibility_periods_used",
                 "astrometric_matched_transits", "astrometric_params_solved", "ruwe")
    with h5py.File(out / "vp_structure.h5", "w") as h:
        for name, path in (("random", args.random), ("nss", args.nss)):
            with h5py.File(path, "r") as f:
                d = {k: f[k][:] for k in keep_cols if k in f}
            sizes, offs = [], [0]
            for i in range(d["source_id"].size):
                r = sl.query(float(d["ra"][i]), float(d["dec"][i]))
                t = np.concatenate([np.asarray(x, float) for x in r]) + GU_TIME_ORIGIN_JD if r else np.array([])
                s = vp_sizes(t, gaps, cfg.transit_split_day)
                sizes.append(s)
                offs.append(offs[-1] + s.size)
                if (i + 1) % 10000 == 0:
                    print(f"  {name}: {i + 1}/{d['source_id'].size}", flush=True)
            g = h.create_group(name)
            for k, v in d.items():
                g.create_dataset(k, data=v)
            g.create_dataset("vp_ntr", data=np.concatenate(sizes) if sizes else np.zeros(0, np.int64))
            g.create_dataset("vp_offsets", data=np.asarray(offs, dtype=np.int64))
            g.attrs["source"] = str(path)
    print(f"wrote {out / 'vp_structure.h5'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
