#!/usr/bin/env python3
"""Snapshot the Halbwachs (b)/(c) input columns for the real NSS comparison sample (#391 MP-Q24).

MP-Q24 (decided 2026-10-03, #391 issuecomment-5971280434): drop the real Orbital +
AstroSpectroSB1 rows that fail the IPD / C* cuts, for symmetry with the mock parent. The uncut
NSS snapshot lacks those columns, so this fetches them once from ``gaiadr3.gaia_source`` for
every comparison row and writes ``columns.h5`` + ``meta.yaml`` under
``<data_root>/dr3/gaia_snapshots/<timestamp>_nss_orbit_input_columns/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

COLUMNS = ("source_id", "phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor",
           "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")


def build_adql(solution_types: tuple[str, ...]) -> str:
    types = ", ".join(f"'{t}'" for t in solution_types)
    cols = ",\n  ".join(f"gs.{c}" for c in COLUMNS)
    return (
        f"SELECT\n  {cols}\nFROM gaiadr3.nss_two_body_orbit AS nss\n"
        "JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id\n"
        f"WHERE nss.nss_solution_type IN ({types})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--types", nargs="+", default=["Orbital", "AstroSpectroSB1"])
    args = ap.parse_args(argv)
    import h5py
    from astroquery.gaia import Gaia

    adql = build_adql(tuple(args.types))
    print(adql)
    when = dt.datetime.now(dt.timezone.utc)
    t = Gaia.launch_job_async(adql).get_results()
    _, first = np.unique(np.asarray(t["source_id"], dtype=np.int64), return_index=True)
    t = t[np.sort(first)]  # one row per source (a source may carry several solution types)
    out = args.data_root / "dr3" / "gaia_snapshots" / f"{when.strftime('%Y%m%dT%H%M%SZ')}_nss_orbit_input_columns"
    out.mkdir(parents=True, exist_ok=False)
    h5 = out / "columns.h5"
    with h5py.File(h5, "w") as h:
        for c in COLUMNS:
            data = np.asarray(t[c], dtype=np.int64) if c == "source_id" else np.ma.filled(np.ma.asarray(t[c], dtype=np.float64), np.nan)
            h.create_dataset(c, data=data)
    digest = hashlib.sha256(h5.read_bytes()).hexdigest()
    meta = {"snapshot_kind": "nss_orbit_input_columns", "issue": 391, "decision": "MP-Q24",
            "query_date": when.isoformat(), "adql": adql, "n_rows": int(len(t)), "columns": list(COLUMNS),
            "columns_h5_sha256": digest}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
