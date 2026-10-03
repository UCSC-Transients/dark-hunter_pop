#!/usr/bin/env python3
"""Snapshot Bailer-Jones distances and CMD photometry for real DR3 NSS orbits (MP-Q28, #413).

docs/MOCK_POPULATION_SPEC.md §10 (giants). The mock parent snapshot carries the
Bailer-Jones et al. (2021) geometric distances (spec §0.1, MP-Q4) and Galactic
coordinates, but the uncut real NSS snapshot does not. Classifying giants on the real side
the same way as in the parent (dereddened CMD with Combined19 at the geometric distance)
needs the same columns for the real orbits, so this one-off async TAP query fetches them
for the comparison solution types and writes them once to
``<data_root>/dr3/gaia_snapshots/<timestamp>_nss_bailer_jones/`` as ``rows.h5`` plus a
``meta.yaml`` (ADQL, query date, row count, SHA256). No stage queries the archive.

Usage::

    .venv/bin/python scripts/fetch_nss_bailer_jones.py --data-root /path/to/data
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
    "source_id", "nss_solution_type", "l", "b", "parallax", "parallax_error",
    "phot_g_mean_mag", "bp_rp", "logg_gspphot", "r_med_geo", "r_lo_geo", "r_hi_geo",
)


def build_adql(solution_types: tuple[str, ...]) -> str:
    """ADQL joining ``nss_two_body_orbit`` to ``gaia_source``, GSP-Phot and Bailer-Jones."""
    types = ", ".join(f"'{t}'" for t in solution_types)
    return (
        "SELECT nss.source_id, nss.nss_solution_type, gs.l, gs.b, gs.parallax, "
        "gs.parallax_error, gs.phot_g_mean_mag, gs.bp_rp, ap.logg_gspphot, "
        "bj.r_med_geo, bj.r_lo_geo, bj.r_hi_geo\n"
        "FROM gaiadr3.nss_two_body_orbit AS nss\n"
        "JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id\n"
        "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON nss.source_id = ap.source_id\n"
        "LEFT JOIN external.gaiaedr3_distance AS bj ON nss.source_id = bj.source_id\n"
        f"WHERE nss.nss_solution_type IN ({types})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--solution-types", nargs="+", default=["Orbital", "AstroSpectroSB1"],
        help="NSS solution types (El-Badry et al. 2024 §4 comparison union)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the ADQL only")
    args = parser.parse_args(argv)
    adql = build_adql(tuple(args.solution_types))
    print(adql)
    if args.dry_run:
        return 0

    import h5py
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    query_date = dt.datetime.now(dt.timezone.utc)
    table = Gaia.launch_job_async(adql).get_results()
    out_dir = args.data_root / "dr3" / "gaia_snapshots" / (
        query_date.strftime("%Y%m%dT%H%M%SZ") + "_nss_bailer_jones"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    h5_path = out_dir / "rows.h5"
    with h5py.File(h5_path, "w") as handle:
        for name in COLUMNS:
            col = table[name]
            if name == "source_id":
                handle.create_dataset(name, data=np.asarray(col, dtype=np.int64))
            elif name == "nss_solution_type":
                handle.create_dataset(name, data=np.asarray(col).astype("S32"))
            else:
                handle.create_dataset(name, data=np.ma.filled(np.ma.asarray(col, dtype=np.float64), np.nan))
    meta = {
        "snapshot_kind": "nss_bailer_jones",
        "issue": 413,
        "spec": "docs/MOCK_POPULATION_SPEC.md §10",
        "query_date": query_date.isoformat(),
        "adql": adql,
        "n_rows": int(len(table)),
        "columns": list(COLUMNS),
        "rows_h5_sha256": _sha256(h5_path),
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {len(table)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
