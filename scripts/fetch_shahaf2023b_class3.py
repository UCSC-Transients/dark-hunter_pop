"""Snapshot Shahaf et al. (2023b) Table 2 (177 class-III systems) from VizieR (#315).

Writes ``{data_root}/dr3/external_catalogs/shahaf2023b_class3_<UTC>_<sha8>/``
with ``table2.ecsv`` and ``meta.yaml`` (catalog id, query date, row count,
SHA-256). Point ``dr3.shahaf2023b_class3_snapshot`` in ``config/config.yaml`` at
the printed directory name. An existing snapshot is never overwritten.

Example::

    .venv/bin/python scripts/fetch_shahaf2023b_class3.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import yaml

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.shahaf2023b_catalog import (
    SHAHAF2023B_CLASS3_CATALOG_ID,
    SHAHAF2023B_DATA_FILE,
    SHAHAF2023B_META_FILE,
    SHAHAF2023B_TABLE2_N_ROWS,
    SHAHAF2023B_VIZIER_TABLE,
    file_sha256,
    load_shahaf2023b_class3,
)


def main() -> int:
    from astroquery.vizier import Vizier

    cfg = load_config()
    root = repo_root() / cfg.paths.data_root / "dr3" / "external_catalogs"
    queried = datetime.now(timezone.utc)
    table = Vizier(row_limit=-1, columns=["**"]).get_catalogs(SHAHAF2023B_VIZIER_TABLE)[0]
    if len(table) != SHAHAF2023B_TABLE2_N_ROWS:
        print(f"expected {SHAHAF2023B_TABLE2_N_ROWS} rows, got {len(table)}", file=sys.stderr)
        return 1
    table.meta = {}
    tmp = root / f".{SHAHAF2023B_CLASS3_CATALOG_ID}_incoming"
    tmp.mkdir(parents=True, exist_ok=True)
    data = tmp / SHAHAF2023B_DATA_FILE
    table.write(data, format="ascii.ecsv", overwrite=True)
    sha = file_sha256(data)
    name = f"{SHAHAF2023B_CLASS3_CATALOG_ID}_{queried:%Y%m%dT%H%M%SZ}_{sha[:8]}"
    meta = {
        "snapshot_id": name,
        "catalog_id": SHAHAF2023B_CLASS3_CATALOG_ID,
        "source": "VizieR",
        "vizier_table": SHAHAF2023B_VIZIER_TABLE,
        "reference": (
            "Shahaf S., Bashi D., Mazeh T., Faigler S., Arenou F., El-Badry K., "
            "Rix H.-W., 2023, MNRAS, 518, 2991 — Table 2"
        ),
        "description": "The highly probable class-III systems sample",
        "query_date": queried.isoformat(),
        "row_count": len(table),
        "columns": list(table.colnames),
        "sha256": sha,
        "result_path": SHAHAF2023B_DATA_FILE,
    }
    (tmp / SHAHAF2023B_META_FILE).write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    final = root / name
    if final.exists():
        print(f"{final} already exists; not overwriting", file=sys.stderr)
        return 1
    tmp.rename(final)
    load_shahaf2023b_class3(final)  # verifies checksum + row count
    print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
