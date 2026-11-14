#!/usr/bin/env python3
"""Snapshot the inputs of the Shahaf et al. (2023b) class-III reproduction (#682).

Writes two frozen snapshot directories under ``<data-root>/dr3/external_catalogs/`` (never overwrites one):

* ``binary_masses_<UTC>/`` — all of ``gaiadr3.binary_masses`` (195,315 rows; IsocLum M1 + 16/84 % bounds,
  ``m1_ref``, ``combination_method``), paged by ``source_id`` because the archive caps async results;
* ``shahaf2023b_table1_<UTC>/`` — VizieR ``J/MNRAS/518/2991/table1`` (101,380 rows: the authors' per-source M1,
  AMRF, AMRF_error, M2min, Pr II, Pr III for the clean sample).

Each holds ``table.ecsv`` and ``meta.yaml`` (query, date, row count, SHA-256). The Table 2 snapshot is
``scripts/fetch_shahaf2023b_class3.py``; the supplementary limiting curves are copied verbatim into
``config/selections/external/shahaf2023b_amrf_triage.yaml``. Network; run on the laptop, read-only thereafter.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

BINARY_MASSES_COLUMNS = (
    "source_id,m1,m1_lower,m1_upper,m2,m2_lower,m2_upper,fluxratio,fluxratio_lower,fluxratio_upper,"
    "combination_method,m1_ref,flag"
)
TABLE1 = "J/MNRAS/518/2991/table1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(table: object, out: Path, meta: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    path = out / "table.ecsv"
    table.write(path, format="ascii.ecsv")  # type: ignore[attr-defined]
    meta.update({"row_count": len(table), "sha256": _sha256(path), "result_path": "table.ecsv"})  # type: ignore[arg-type]
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"wrote {out} ({meta['row_count']} rows)")


def fetch_binary_masses(out_root: Path, stamp: str, page: int) -> None:
    from astropy.table import vstack
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    last, parts = -1, []
    while True:
        q = (f"SELECT TOP {page} {BINARY_MASSES_COLUMNS} FROM gaiadr3.binary_masses "
             f"WHERE source_id > {last} ORDER BY source_id")
        t = Gaia.launch_job_async(q).get_results()
        if len(t) == 0:
            break
        parts.append(t)
        last = int(t["source_id"][-1])
        if len(t) < page:
            break
    table = vstack(parts)
    _write(table, out_root / f"binary_masses_{stamp}", {
        "source": "Gaia archive", "table": "gaiadr3.binary_masses", "columns": BINARY_MASSES_COLUMNS.split(","),
        "query": f"paged SELECT TOP {page} ... WHERE source_id > <last> ORDER BY source_id",
        "query_date": stamp, "issue": 682,
    })


def fetch_table1(out_root: Path, stamp: str, server: str | None) -> None:
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=["**"], timeout=300)
    if server:
        v.VIZIER_SERVER = server
    table = v.get_catalogs(TABLE1)[0]
    _write(table, out_root / f"shahaf2023b_table1_{stamp}", {
        "source": "VizieR", "vizier_table": TABLE1, "vizier_server": server or "default", "query_date": stamp,
        "reference": "Shahaf et al. (2023), MNRAS 518, 2991, Table 1", "issue": 682,
        "note": "PIII / PII are fractions despite the VizieR '%' unit (min 1/(1e5+1)).",
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--what", choices=("all", "binary_masses", "table1"), default="all")
    ap.add_argument("--page", type=int, default=40000)
    ap.add_argument("--vizier-server", default=None, help="e.g. vizier.cfa.harvard.edu if CDS is unreachable")
    args = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.data_root / "dr3" / "external_catalogs"
    if args.what in ("all", "binary_masses"):
        fetch_binary_masses(out_root, stamp, args.page)
    if args.what in ("all", "table1"):
        fetch_table1(out_root, stamp, args.vizier_server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
