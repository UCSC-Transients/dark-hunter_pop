#!/usr/bin/env python3
"""Snapshot DR3 G-band epoch times for the time-clustered loss model (#400 E4).

docs/EPOCH_MODEL_SPEC.md §4.3. Two steps, written once to
``<data_root>/dr3/gaia_snapshots/<timestamp>_epoch_times_400/``:

1. One TAP query: ``gaia_source`` rows with ``has_epoch_photometry`` in a uniform
   ``random_index`` slice, ``--g-min`` < G < ``--g-max`` (the epoch-count columns plus
   position). Stored as ``sources.h5``.
2. DataLink ``EPOCH_PHOTOMETRY`` (RAW) for a seeded random subset of up to
   ``--n-faint`` sources with G > ``--g-split`` and ``--n-bright`` with G <= ``--g-split``,
   in chunks. Only the per-transit times (``g_transit_time``, BJD(TCB) − 2455197.5),
   ``transit_id`` and ``g_transit_n_obs`` are kept, as ragged arrays with offsets, in
   ``epochs.h5``.

Epoch photometry exists only for DR3 variability candidates, so these stars are not a
random sample of all stars (recorded in ``meta.yaml``). Their transit **times** are a
property of the scanning law and of the losses, which is what the E4 test uses.
``n_transits`` equals ``matched_transits``: these are matched transits, not the subset the
astrometric solution used.

Usage::

    .venv/bin/python scripts/fetch_epoch_photometry_400.py --data-root data
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

COLUMNS: tuple[str, ...] = (
    "source_id", "random_index", "ra", "dec", "l", "b", "ecl_lat", "phot_g_mean_mag", "bp_rp",
    "ruwe", "matched_transits", "astrometric_matched_transits", "astrometric_n_good_obs_al",
    "visibility_periods_used", "phot_g_n_obs", "astrometric_params_solved",
)


def sources_adql(k: int, g_min: float, g_max: float) -> str:
    cols = ",\n  ".join(COLUMNS)
    return (
        f"SELECT\n  {cols}\nFROM gaiadr3.gaia_source\n"
        f"WHERE random_index < {int(k)}\n  AND has_epoch_photometry = 'True'\n"
        f"  AND phot_g_mean_mag > {g_min}\n  AND phot_g_mean_mag < {g_max}\n"
        "  AND astrometric_params_solved > 3"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filled(x: object) -> np.ndarray:
    a = np.ma.asarray(x, dtype=np.float64)
    return np.ma.filled(a, np.nan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3_000_000)
    parser.add_argument("--g-min", type=float, default=11.0)
    parser.add_argument("--g-max", type=float, default=19.0)
    parser.add_argument("--g-split", type=float, default=17.0)
    parser.add_argument("--n-faint", type=int, default=1500)
    parser.add_argument("--n-bright", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    adql = sources_adql(args.k, args.g_min, args.g_max)
    print(adql)
    if args.dry_run:
        return 0

    import h5py
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    query_date = dt.datetime.now(dt.timezone.utc)
    out = args.data_root / "dr3" / "gaia_snapshots" / f"{query_date.strftime('%Y%m%dT%H%M%SZ')}_epoch_times_400"
    out.mkdir(parents=True, exist_ok=False)
    src = Gaia.launch_job_async(adql).get_results()
    with h5py.File(out / "sources.h5", "w") as h:
        for c in COLUMNS:
            h.create_dataset(c, data=np.asarray(src[c], dtype=np.int64) if c in ("source_id", "random_index")
                             else _filled(src[c]))
    g = np.asarray(src["phot_g_mean_mag"], dtype=float)
    sid = np.asarray(src["source_id"], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    faint = sid[g > args.g_split]
    bright = sid[g <= args.g_split]
    pick = np.concatenate([
        rng.choice(faint, min(args.n_faint, faint.size), replace=False),
        rng.choice(bright, min(args.n_bright, bright.size), replace=False),
    ])
    ids, times, tids, nobs = [], [], [], []
    for i in range(0, pick.size, args.chunk):
        chunk = [int(x) for x in pick[i:i + args.chunk]]
        res = Gaia.load_data(ids=chunk, data_release="Gaia DR3", retrieval_type="EPOCH_PHOTOMETRY",
                             data_structure="RAW", format="votable")
        tab = list(res.values())[0][0].to_table()
        for row in tab:
            ids.append(int(row["source_id"]))
            times.append(_filled(row["g_transit_time"]))
            tids.append(np.asarray(row["transit_id"], dtype=np.int64))
            nobs.append(_filled(row["g_transit_n_obs"]))
        print(f"  {len(ids)} sources", flush=True)
    off = np.r_[0, np.cumsum([t.size for t in times])]
    with h5py.File(out / "epochs.h5", "w") as h:
        h.create_dataset("source_id", data=np.asarray(ids, dtype=np.int64))
        h.create_dataset("offsets", data=off.astype(np.int64))
        h.create_dataset("g_transit_time", data=np.concatenate(times))
        h.create_dataset("transit_id", data=np.concatenate(tids))
        h.create_dataset("g_transit_n_obs", data=np.concatenate(nobs))
    meta = {
        "snapshot_kind": "epoch_times", "issue": 400, "spec": "docs/EPOCH_MODEL_SPEC.md §4.3",
        "query_date": query_date.isoformat(), "adql": adql, "n_sources_query": int(len(src)),
        "datalink": {"retrieval_type": "EPOCH_PHOTOMETRY", "data_structure": "RAW",
                     "data_release": "Gaia DR3", "n_sources": len(ids), "seed": args.seed,
                     "g_split": args.g_split, "n_faint_requested": args.n_faint,
                     "n_bright_requested": args.n_bright},
        "time_column": "g_transit_time = BJD(TCB) - 2455197.5 d",
        "caveat": "epoch photometry exists only for DR3 variability candidates; times are matched transits",
        "sources_sha256": _sha256(out / "sources.h5"), "epochs_sha256": _sha256(out / "epochs.h5"),
    }
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
