#!/usr/bin/env python3
"""Snapshot DR3 epoch-count columns for calibrating the pop-side epoch model (#400, #398).

docs/EPOCH_MODEL_SPEC.md §2. Four modest async Gaia TAP queries, written once to
``<data_root>/dr3/gaia_snapshots/<timestamp>_epoch_counts_400/`` with a ``meta.yaml``
(ADQL, query date, row count and SHA256 per table). A stage never re-queries the archive.

Tables
------
``random.h5``
    Uniform random ``gaia_source`` slice (``random_index < --k-random``), G < ``--g-max``,
    5- or 6-parameter solutions. Single-star-like and everything else; RUWE is stored,
    never cut on here.
``nss.h5``
    Orbital + AstroSpectroSB1 rows of ``nss_two_body_orbit`` whose ``gaia_source``
    ``random_index`` is below ``--k-nss`` (a uniform slice of the NSS sample).
``inj390.h5``
    The same columns for the 1,296 #390 injection systems (source IDs read from the #390
    artifact).
``density.h5``
    Source counts per HEALPix level-``--density-level`` pixel over ``random_index <
    --k-density`` (all G), the local source-density proxy.

Usage::

    .venv/bin/python scripts/fetch_epoch_count_sample.py --data-root <data> \
        --inj390 output/gate390/injection_test_full.h5 [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

#: ``gaia_source`` columns snapshotted for every star. Epoch counts (all the counting
#: definitions in DR3), position, brightness, colour, RUWE, scan-direction summary.
EPOCH_COUNT_COLUMNS: tuple[str, ...] = (
    "source_id", "random_index", "ra", "dec", "l", "b", "ecl_lon", "ecl_lat",
    "phot_g_mean_mag", "bp_rp", "ruwe", "astrometric_params_solved",
    "matched_transits", "matched_transits_removed", "astrometric_matched_transits",
    "astrometric_n_obs_al", "astrometric_n_good_obs_al", "astrometric_n_bad_obs_al",
    "visibility_periods_used", "phot_g_n_obs", "ipd_frac_multi_peak",
    "astrometric_excess_noise", "non_single_star",
    "scan_direction_strength_k1", "scan_direction_strength_k2",
    "scan_direction_strength_k3", "scan_direction_strength_k4",
    "scan_direction_mean_k1", "scan_direction_mean_k2",
    "scan_direction_mean_k3", "scan_direction_mean_k4",
)


def _select(prefix: str) -> str:
    return ",\n  ".join(f"{prefix}{c}" for c in EPOCH_COUNT_COLUMNS)


def random_adql(k: int, g_max: float) -> str:
    """Uniform random slice of ``gaia_source`` (5/6-parameter solutions, G < g_max)."""
    return (
        f"SELECT\n  {_select('gs.')}\nFROM gaiadr3.gaia_source AS gs\n"
        f"WHERE gs.random_index < {int(k)}\n  AND gs.phot_g_mean_mag < {g_max}\n"
        "  AND gs.astrometric_params_solved > 3"
    )


def nss_adql(k: int) -> str:
    """Uniform slice of the Orbital + AstroSpectroSB1 NSS sample."""
    return (
        f"SELECT\n  {_select('gs.')},\n  nss.nss_solution_type\n"
        "FROM gaiadr3.nss_two_body_orbit AS nss\n"
        "JOIN gaiadr3.gaia_source AS gs ON gs.source_id = nss.source_id\n"
        "WHERE nss.nss_solution_type IN ('Orbital', 'AstroSpectroSB1')\n"
        f"  AND gs.random_index < {int(k)}"
    )


def ids_adql(source_ids: list[int]) -> str:
    """The snapshot columns for an explicit source-ID list."""
    ids = ", ".join(str(int(s)) for s in source_ids)
    return (
        f"SELECT\n  {_select('gs.')}\nFROM gaiadr3.gaia_source AS gs\n"
        f"WHERE gs.source_id IN ({ids})"
    )


def density_adql(k: int, level: int) -> str:
    """Source counts per HEALPix pixel (nested) from a uniform ``random_index`` slice.

    The level-``level`` nested index is ``source_id // (2**35 * 4**(12 - level))``
    (Gaia DR3 source_id encodes the level-12 nested index in its top bits). Integer
    division is used rather than ``GAIA_HEALPIX_INDEX`` inside ``GROUP BY``, which the
    archive rejected with an HTTP 500 on 2026-10-03.
    """
    div = 2**35 * 4 ** (12 - int(level))
    return (
        f"SELECT source_id / {div} AS hpx, COUNT(*) AS n\n"
        "FROM gaiadr3.gaia_source\n"
        f"WHERE random_index < {int(k)}\n"
        "GROUP BY hpx"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(table, path: Path) -> list[str]:
    import h5py

    names = list(table.colnames)
    with h5py.File(path, "w") as handle:
        for name in names:
            col = table[name]
            if name in ("source_id", "random_index", "hpx"):
                handle.create_dataset(name, data=np.asarray(col, dtype=np.int64))
            elif col.dtype.kind in "SUO":
                handle.create_dataset(name, data=np.asarray(col).astype("S"))
            else:
                handle.create_dataset(
                    name, data=np.ma.filled(np.ma.asarray(col, dtype=np.float64), np.nan)
                )
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inj390", type=Path, required=True, help="#390 injection HDF5")
    parser.add_argument("--k-random", type=int, default=300_000)
    parser.add_argument("--k-nss", type=int, default=181_170_977, help="~10%% of gaia_source")
    parser.add_argument("--k-density", type=int, default=10_000_000)
    parser.add_argument("--density-level", type=int, default=5)
    parser.add_argument("--g-max", type=float, default=19.0)
    parser.add_argument(
        "--tables", nargs="+", default=["random", "nss", "inj390", "density"],
        help="subset of tables to fetch (resume after an archive error)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="existing snapshot directory to add tables to (meta.yaml is merged)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import h5py

    with h5py.File(args.inj390, "r") as handle:
        inj_ids = [int(s) for s in handle["systems/source_id"][:]]
    queries = {
        "random": random_adql(args.k_random, args.g_max),
        "nss": nss_adql(args.k_nss),
        "inj390": ids_adql(inj_ids),
        "density": density_adql(args.k_density, args.density_level),
    }
    queries = {k: v for k, v in queries.items() if k in set(args.tables)}
    for name, q in queries.items():
        print(f"-- {name}\n{q if name != 'inj390' else q[:400] + ' ...'}\n")
    if args.dry_run:
        return 0

    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    query_date = dt.datetime.now(dt.timezone.utc)
    if args.out_dir is not None:
        out_dir = args.out_dir
        old_meta = yaml.safe_load((out_dir / "meta.yaml").read_text()) if (
            out_dir / "meta.yaml"
        ).is_file() else None
    else:
        out_dir = (
            args.data_root / "dr3" / "gaia_snapshots"
            / f"{query_date.strftime('%Y%m%dT%H%M%SZ')}_epoch_counts_400"
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        old_meta = None
    meta: dict = {
        "snapshot_kind": "epoch_counts",
        "issue": [400, 398],
        "spec": "docs/EPOCH_MODEL_SPEC.md §2",
        "query_date": query_date.isoformat(),
        "gaia_source_total_rows": 1_811_709_771,
        "k_random": args.k_random, "k_nss": args.k_nss, "k_density": args.k_density,
        "density_healpix_level_nested": args.density_level, "g_max": args.g_max,
        "inj390_source": str(args.inj390), "inj390_sha256": _sha256(args.inj390),
        "tables": {},
    }
    if old_meta is not None:
        # keep the original snapshot-level fields; only add or replace tables
        meta = {**old_meta, "tables": dict(old_meta.get("tables", {}))}
    for name, q in queries.items():
        job = Gaia.launch_job_async(q)
        table = job.get_results()
        path = out_dir / f"{name}.h5"
        cols = _write(table, path)
        meta["tables"][name] = {
            "adql": q, "n_rows": int(len(table)), "columns": cols,
            "sha256": _sha256(path), "query_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        print(f"{name}: {len(table)} rows")
        # written after every table so a later archive error keeps what succeeded
        (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
