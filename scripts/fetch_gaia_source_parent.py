#!/usr/bin/env python3
"""Snapshot a uniform random subsample of the ``gaia_source`` mock parent (#391).

docs/MOCK_POPULATION_SPEC.md §1.3. One async Gaia TAP query, a shape-neutral
subsample via ``random_index < K``, written once to
``<data_root>/dr3/gaia_snapshots/<timestamp>_gaia_source_parent_K<K>_plx<floor>/``
as ``parent.h5`` plus ``meta.yaml`` (ADQL, query date, row count, SHA256, ``K``,
full-table row count for scaling). A stage never re-queries the archive.

The selection is G < ``--g-max`` and parallax > ``--parallax-floor`` only. RUWE and
visibility periods are outcomes that gaiamock simulates and are never cut on here;
they and the Halbwachs et al. (2023) IPD / C* columns are stored for diagnostics
and for MP-Q3.

Usage::

    .venv/bin/python scripts/fetch_gaia_source_parent.py --k 1000000 \
        --parallax-floor 0.2 --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

from darkhunter_pop.proposal_set import (
    BAILER_JONES_COLUMNS,
    GAIA_SOURCE_PARENT_COLUMNS,
    GAIA_SOURCE_TOTAL_ROWS,
    build_gaia_source_parent_adql,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, required=True, help="random_index upper bound")
    parser.add_argument("--k-min", type=int, default=0, help="random_index lower bound (spec §12.30 supplement)")
    parser.add_argument("--extra-columns", nargs="*", default=[], help="extra gaia_source columns, e.g. vbroad")
    parser.add_argument("--parallax-floor", type=float, required=True, help="mas")
    parser.add_argument("--g-max", type=float, default=19.0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--bailer-jones", action="store_true",
        help="LEFT JOIN external.gaiaedr3_distance geometric distances (spec §0.1 MP-Q4)",
    )
    parser.add_argument("--chunk", type=int, default=0, help="query random_index in chunks of this size (archive row cap)")
    parser.add_argument("--row-cap", type=int, default=50000, help="a chunk returning this many rows is treated as truncated")
    parser.add_argument("--dry-run", action="store_true", help="print the ADQL only")
    args = parser.parse_args(argv)

    adql = build_gaia_source_parent_adql(
        k=args.k, parallax_floor_mas=args.parallax_floor, g_max=args.g_max,
        include_bailer_jones=args.bailer_jones, k_min=args.k_min, extra_columns=tuple(args.extra_columns),
    )
    print(adql)
    if args.dry_run:
        return 0

    import h5py
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    query_date = dt.datetime.now(dt.timezone.utc)
    if args.chunk:
        from astropy.table import vstack

        parts = []
        for lo in range(args.k_min, args.k, args.chunk):
            hi = min(lo + args.chunk, args.k)
            q = build_gaia_source_parent_adql(k=hi, parallax_floor_mas=args.parallax_floor, g_max=args.g_max,
                                              include_bailer_jones=args.bailer_jones, k_min=lo, extra_columns=tuple(args.extra_columns))
            t = Gaia.launch_job_async(q).get_results()
            if len(t) >= args.row_cap:
                raise SystemExit(f"chunk [{lo}, {hi}) returned {len(t)} rows (row cap); use a smaller --chunk")
            print(f"chunk [{lo}, {hi}): {len(t)} rows", flush=True)
            parts.append(t)
        table = vstack(parts)
    else:
        job = Gaia.launch_job_async(adql)
        table = job.get_results()
        if len(table) >= args.row_cap:
            raise SystemExit(f"{len(table)} rows hit the archive row cap; rerun with --chunk")
    stamp = query_date.strftime("%Y%m%dT%H%M%SZ")
    floor_tag = f"{args.parallax_floor:g}".replace(".", "p") + ("_bj" if args.bailer_jones else "")
    out_dir = (
        args.data_root
        / "dr3"
        / "gaia_snapshots"
        / (f"{stamp}_gaia_source_parent_K{args.k}_plx{floor_tag}" if not args.k_min
           else f"{stamp}_gaia_source_supplement_K{args.k_min}-{args.k}_G{args.g_max:g}_plx{floor_tag}")
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    h5_path = out_dir / "parent.h5"
    with h5py.File(h5_path, "w") as handle:
        names = list(GAIA_SOURCE_PARENT_COLUMNS) + list(args.extra_columns) + (list(BAILER_JONES_COLUMNS) if args.bailer_jones else [])
        for name in names:
            col = table[name]
            if name == "source_id" or name == "random_index":
                handle.create_dataset(name, data=np.asarray(col, dtype=np.int64))
            else:
                data = np.ma.filled(np.ma.asarray(col, dtype=np.float64), np.nan)
                handle.create_dataset(name, data=data)
    meta = {
        "snapshot_kind": "gaia_source_parent",
        "issue": 391,
        "spec": "docs/MOCK_POPULATION_SPEC.md §1.3",
        "query_date": query_date.isoformat(),
        "adql": adql,
        "chunk": int(args.chunk),
        "random_index_max_exclusive": int(args.k),
        "random_index_min_inclusive": int(args.k_min),
        "parallax_floor_mas": float(args.parallax_floor),
        "g_max": float(args.g_max),
        "gaia_source_total_rows": int(GAIA_SOURCE_TOTAL_ROWS),
        "subsample_fraction": float(args.k - args.k_min) / float(GAIA_SOURCE_TOTAL_ROWS),
        "n_rows": int(len(table)),
        "columns": names,
        "bailer_jones": bool(args.bailer_jones),
        "parent_h5_sha256": _sha256(h5_path),
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {len(table)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
