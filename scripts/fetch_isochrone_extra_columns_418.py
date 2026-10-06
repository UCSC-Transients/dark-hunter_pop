#!/usr/bin/env python3
"""Snapshot the extra Gaia DR3 columns #418 needs (MP-Q33, MP-Q34).

* BP/RP flux errors (``phot_bp_mean_flux_over_error``, ``phot_rp_mean_flux_over_error``,
  ``phot_g_mean_flux_over_error``): per-star colour errors for the isochrone likelihood.
* The GSP-Phot inputs of ``gdr3apcal`` (Andrae et al. 2023): ``teff_gspphot``, ``logg_gspphot``,
  ``mh_gspphot``, ``azero_gspphot``, ``ebpminrp_gspphot``, ``ag_gspphot``, ``mg_gspphot``,
  ``libname_gspphot``.

``--sample parent`` takes the mock parent slice (``random_index < K``, G < 19, ϖ > floor, the
same WHERE as ``fetch_gaia_source_parent.py``); ``--sample nss`` takes every Orbital +
AstroSpectroSB1 source of ``gaiadr3.nss_two_body_orbit``. Writes ``columns.h5`` + ``meta.yaml``
(ADQL, query date, row count, SHA256) under
``<data_root>/dr3/gaia_snapshots/<timestamp>_<sample>_isochrone_extra_columns/``.
``--print-adql`` prints the query and exits (nothing is fetched).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

GS_COLUMNS = ("phot_g_mean_flux_over_error", "phot_bp_mean_flux_over_error", "phot_rp_mean_flux_over_error")
AP_COLUMNS = ("teff_gspphot", "logg_gspphot", "mh_gspphot", "azero_gspphot", "ebpminrp_gspphot",
              "ag_gspphot", "mg_gspphot", "libname_gspphot")


def build_adql(sample: str, k: int, parallax_floor_mas: float, types: tuple[str, ...], ids: list[int] | None = None) -> str:
    cols = ",\n  ".join(["gs.source_id"] + [f"gs.{c}" for c in GS_COLUMNS] + [f"ap.{c}" for c in AP_COLUMNS])
    join_ap = "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id"
    if sample == "ids":
        idl = ", ".join(str(int(i)) for i in (ids or []))
        return f"SELECT\n  {cols}\nFROM gaiadr3.gaia_source AS gs\n{join_ap}\nWHERE gs.source_id IN ({idl})"
    if sample == "parent":
        return (f"SELECT\n  {cols}\nFROM gaiadr3.gaia_source AS gs\n{join_ap}\n"
                f"WHERE gs.random_index < {int(k)} AND gs.phot_g_mean_mag < 19 AND gs.parallax > {parallax_floor_mas}")
    tl = ", ".join(f"'{t}'" for t in types)
    return (f"SELECT\n  {cols}\nFROM gaiadr3.nss_two_body_orbit AS nss\n"
            f"JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id\n{join_ap}\n"
            f"WHERE nss.nss_solution_type IN ({tl})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--sample", choices=("parent", "nss", "ids"), required=True)
    ap.add_argument("--ids-from", type=Path, default=None,
                    help="--sample ids: an ECSV with a source_id column (e.g. the DEBCat cross-match)")
    ap.add_argument("--label", default=None, help="snapshot directory label (default: the sample name)")
    ap.add_argument("--k", type=int, default=1_000_000)
    ap.add_argument("--parallax-floor-mas", type=float, default=0.2)
    ap.add_argument("--types", nargs="+", default=["Orbital", "AstroSpectroSB1"])
    ap.add_argument("--print-adql", action="store_true")
    args = ap.parse_args(argv)
    ids = None
    if args.sample == "ids":
        from astropy.table import Table

        ids = sorted({int(x) for x in Table.read(args.ids_from, format="ascii.ecsv")["source_id"] if int(x) > 0})
    adql = build_adql(args.sample, args.k, args.parallax_floor_mas, tuple(args.types), ids)
    print(adql)
    if args.print_adql:
        return 0
    import h5py
    from astroquery.gaia import Gaia

    when = dt.datetime.now(dt.timezone.utc)
    t = Gaia.launch_job_async(adql).get_results()
    _, first = np.unique(np.asarray(t["source_id"], dtype=np.int64), return_index=True)
    t = t[np.sort(first)]
    out = args.data_root / "dr3" / "gaia_snapshots" / f"{when.strftime('%Y%m%dT%H%M%SZ')}_{args.label or args.sample}_isochrone_extra_columns"
    out.mkdir(parents=True, exist_ok=False)
    h5 = out / "columns.h5"
    with h5py.File(h5, "w") as h:
        h.create_dataset("source_id", data=np.asarray(t["source_id"], dtype=np.int64))
        for c in GS_COLUMNS + AP_COLUMNS:
            if c == "libname_gspphot":
                h.create_dataset(c, data=np.asarray(np.ma.filled(np.ma.asarray(t[c]).astype(str), ""), dtype="S16"))
            else:
                h.create_dataset(c, data=np.ma.filled(np.ma.asarray(t[c], dtype=np.float64), np.nan))
    meta = {"snapshot_kind": f"{args.sample}_isochrone_extra_columns", "issue": 418, "decisions": ["MP-Q33", "MP-Q34"],
            "query_date": when.isoformat(), "adql": adql, "n_rows": int(len(t)),
            "columns": ["source_id", *GS_COLUMNS, *AP_COLUMNS], "columns_h5_sha256": hashlib.sha256(h5.read_bytes()).hexdigest()}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
