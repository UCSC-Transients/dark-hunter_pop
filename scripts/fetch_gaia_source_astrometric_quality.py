#!/usr/bin/env python3
"""Snapshot DR3 astrometric-quality columns for the looks-single parent rows (#572, #454; approved by Ryan 2026-10-26).

Same WHERE clause as the §12.33 parent snapshot (``random_index < K``, G < 20, parallax > 0.2), queried in
``random_index`` chunks (async; sync fallback per chunk). Writes
``<data_root>/dr3/gaia_snapshots/<timestamp>_gaia_source_astrometric_quality_K<K>/quality.h5`` + ``meta.yaml``
(ADQL, query date, row count, sha256). Joined to the parent by ``source_id``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml

COLUMNS = ("source_id", "astrometric_gof_al", "astrometric_chi2_al", "astrometric_excess_noise",
           "astrometric_excess_noise_sig", "astrometric_n_good_obs_al", "astrometric_n_obs_al",
           "astrometric_matched_transits", "astrometric_params_solved", "astrometric_sigma5d_max",
           "parallax_over_error", "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat")


def adql(lo: int, hi: int, g_max: float, plx_floor: float) -> str:
    return (f"SELECT {', '.join(COLUMNS)}\nFROM gaiadr3.gaia_source\nWHERE random_index >= {lo} AND random_index < {hi}\n"
            f"  AND phot_g_mean_mag < {g_max} AND parallax > {plx_floor}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--g-max", type=float, default=20.0)
    ap.add_argument("--parallax-floor", type=float, default=0.2)
    ap.add_argument("--data-root", type=Path, required=True)
    a = ap.parse_args(argv)
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = a.data_root / "dr3" / "gaia_snapshots" / f"{stamp}_gaia_source_astrometric_quality_K{a.k}"
    out.mkdir(parents=True, exist_ok=False)
    parts, queries = [], []
    for lo in range(0, a.k, a.chunk):
        q = adql(lo, min(lo + a.chunk, a.k), a.g_max, a.parallax_floor)
        queries.append(q)
        for attempt in range(4):
            try:
                t = (Gaia.launch_job_async(q) if attempt < 2 else Gaia.launch_job(q)).get_results()
                break
            except Exception as exc:  # noqa: BLE001 - archive flakiness, retried then raised
                print(f"chunk {lo}: attempt {attempt} failed: {exc!r}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(30)
        parts.append({c: (np.asarray(t[c], dtype=np.int64) if c == "source_id"
                          else np.ma.filled(np.ma.asarray(t[c]).astype(np.float64), np.nan)) for c in COLUMNS})
        print(f"chunk {lo}: {len(t)} rows", flush=True)
    import h5py

    h5 = out / "quality.h5"
    with h5py.File(h5, "w") as f:
        for c in COLUMNS:
            f.create_dataset(c, data=np.concatenate([p[c] for p in parts]))
    n = sum(len(p["source_id"]) for p in parts)
    sha = hashlib.sha256(h5.read_bytes()).hexdigest()
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "snapshot_kind": "gaia_source_astrometric_quality", "issue": [572, 454],
        "approved": "Ryan Foley, 2026-10-26 (via coordinator)", "query_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "adql_chunks": queries, "random_index_max_exclusive": a.k, "g_max": a.g_max, "parallax_floor_mas": a.parallax_floor,
        "n_rows": int(n), "columns": list(COLUMNS), "quality_h5_sha256": sha}, sort_keys=False))
    print(f"wrote {out} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
