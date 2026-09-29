"""Snapshot VizieR I/355/paramp (Gaia DR3 Apsis) for the Andrews ATF giant cut (#306).

The ATF selection notebook (``data/reference/andrews2022_ATF_sample_selection.ipynb``,
cells 13-17) reads ``logg`` / ``Teff`` / ``Mass-Flame`` from a VizieR download
(``massive_apsis.vot``, catalog I/355/paramp), not from the Gaia archive
``astrophysical_parameters`` table. Ryan Foley (2026-09-28, #296 / #306): carry
the VizieR values as their **own** columns beside the Gaia-archive ones, and
have the Andrews giant cut read the VizieR column, as the notebook did.

This script fetches, for a fixed source-ID set, the paramp columns the notebook
touches and writes a timestamped snapshot the builder reads offline (a live
service is never read at evaluation time)::

    <paths.data_root>/dr3/vizier_apsis/<UTC>_<checksum12>/query.ecsv
    <paths.data_root>/dr3/vizier_apsis/<UTC>_<checksum12>/meta.yaml
    <paths.data_root>/dr3/vizier_apsis/<UTC>_<checksum12>/requested_source_ids.txt

The requested-ID list is kept so a source that VizieR has **no row** for
(recorded as absent) is distinguishable from a source that was **never asked
for** (the builder refuses those).

ID set: every source in an existing ATF sidecar whose notebook pass-1
``andrews_atf_p_m2_above`` is at least ``--min-p`` (pass 1 does not read
``logg``, so any sidecar built from the same NSS inputs gives the same set).
The default 0.1 keeps a wide margin below the 0.95 pass-1 threshold. ``--all``
fetches every source in the sidecar (≈ 1 h at ~5 s per 200-ID request).

Point ``dr3.vizier_apsis_snapshot_meta`` in ``config/config.yaml`` at the new
``meta.yaml`` afterwards.

Example::

    .venv/bin/python scripts/fetch_vizier_apsis.py \
        --sidecar data/reproduction_columns/dr3/atf_notebook_4181870cbce6ac8f.h5
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import astroquery
import h5py
import numpy as np
import yaml
from astropy.table import MaskedColumn, Table, vstack

from darkhunter_pop.andrews2022_atf import VIZIER_APSIS_COLUMN_MAP, VIZIER_APSIS_CATALOG
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.data_acquisition import save_gaia_snapshot

REQUESTED_IDS_FILENAME = "requested_source_ids.txt"


def _select_ids(sidecar: Path, *, min_p: float, fetch_all: bool) -> np.ndarray:
    with h5py.File(sidecar, "r") as handle:
        ids = np.asarray(handle["source_id"][()], dtype=np.int64)
        p = np.asarray(handle["andrews_atf_p_m2_above"][()], dtype=np.float64)
    if fetch_all:
        return np.sort(ids)
    keep = np.isfinite(p) & (p >= float(min_p))
    return np.sort(ids[keep])


def _query_chunk(ids: np.ndarray, *, timeout_s: float) -> Table | None:
    from astroquery.vizier import Vizier

    vizier = Vizier(
        columns=list(VIZIER_APSIS_COLUMN_MAP), row_limit=-1, timeout=timeout_s
    )
    result = vizier.query_constraints(
        catalog=VIZIER_APSIS_CATALOG,
        Source="=" + ",".join(str(int(s)) for s in ids),
    )
    if len(result) == 0:
        return None
    return result[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument("--all", action="store_true", dest="fetch_all")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    root = repo_root()
    cfg = load_config()
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    out_root = data_root / "dr3" / "vizier_apsis"

    ids = _select_ids(args.sidecar, min_p=args.min_p, fetch_all=args.fetch_all)
    print(f"requesting {len(ids)} source_ids from VizieR {VIZIER_APSIS_CATALOG}", flush=True)
    tables: list[Table] = []
    t0 = time.time()
    for start in range(0, len(ids), args.chunk):
        chunk = ids[start : start + args.chunk]
        table = _query_chunk(chunk, timeout_s=args.timeout)
        if table is not None:
            tables.append(table)
        print(
            f"  {min(start + args.chunk, len(ids))}/{len(ids)} "
            f"rows_so_far={sum(len(t) for t in tables)} t={time.time() - t0:.0f}s",
            flush=True,
        )
    if not tables:
        print("VizieR returned no rows", file=sys.stderr)
        return 1
    raw = vstack(tables, metadata_conflicts="silent")
    out = Table()
    for viz_name, our_name in VIZIER_APSIS_COLUMN_MAP.items():
        col = raw[viz_name]
        if our_name == "source_id":
            out[our_name] = np.asarray(col, dtype=np.int64)
            continue
        data = np.ma.asarray(col).astype(np.float64)
        out[our_name] = MaskedColumn(data=np.ma.getdata(data), mask=np.ma.getmaskarray(data))
    out.sort("source_id")
    requested = set(int(s) for s in ids)
    returned = [int(s) for s in out["source_id"]]
    stray = sorted(set(returned) - requested)
    if stray:
        print(f"VizieR returned unrequested source_ids: {stray[:5]}", file=sys.stderr)
        return 1

    description = (
        f"VizieR {VIZIER_APSIS_CATALOG} query_constraints(Source='=<ids>') "
        f"columns={list(VIZIER_APSIS_COLUMN_MAP)}; ids = {REQUESTED_IDS_FILENAME}"
    )
    meta = save_gaia_snapshot(out, description, snapshots_dir=out_root)
    ids_path = meta.meta_path.parent / REQUESTED_IDS_FILENAME
    ids_text = "".join(f"{int(s)}\n" for s in ids)
    ids_path.write_text(ids_text, encoding="utf-8")
    raw_meta = yaml.safe_load(meta.meta_path.read_text(encoding="utf-8"))
    raw_meta.update(
        {
            "service": "VizieR (astroquery.vizier)",
            "catalog": VIZIER_APSIS_CATALOG,
            "column_map": dict(VIZIER_APSIS_COLUMN_MAP),
            "requested_ids_path": REQUESTED_IDS_FILENAME,
            "requested_ids_sha256": hashlib.sha256(ids_text.encode("utf-8")).hexdigest(),
            "n_requested": len(ids),
            "n_returned": len(out),
            "n_duplicate_source_ids": len(returned) - len(set(returned)),
            "id_selection": (
                "all sidecar rows"
                if args.fetch_all
                else f"andrews_atf_p_m2_above >= {args.min_p}"
            ),
            "id_source_sidecar": str(args.sidecar),
            "astroquery": astroquery.__version__,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "issue": "#306",
        }
    )
    meta.meta_path.write_text(yaml.safe_dump(raw_meta, sort_keys=False), encoding="utf-8")
    print(
        f"wrote {meta.meta_path} (requested={len(ids)} returned={len(out)} "
        f"checksum={meta.checksum[:12]})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
