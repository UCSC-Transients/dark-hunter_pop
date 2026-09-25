"""Merge Gaia Apsis FLAME-mass enrichment onto the ``+enrich`` selection-parent
cache (#257 — Andrews et al. 2022 M1 methodology).

Reads ``data/dr3/gaia_snapshots/flame_enrichment/query.ecsv`` (written by
``scripts/fetch_flame_enrichment.py``) and overlays ``mass_flame`` /
``mass_flame_upper`` / ``mass_flame_lower`` onto
``…+enrich/selection_parent_rows.h5`` by ``(source_id, nss_solution_type)``,
writing the result to a **new**, non-destructive
``…+enrich+flame/selection_parent_rows.h5`` cache — the original ``+enrich``
cache (built from the frozen ``nss_enrichment`` job) is never modified.
``scripts/attach_mc_to_selection_cache.py`` prefers ``+enrich+flame`` when it
exists.

Example::

    .venv/bin/python scripts/merge_flame_enrichment_into_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from astropy.table import Table

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.data_acquisition import (
    _enrichment_join_key,
    merge_nss_enrichment_into_row,
)
from darkhunter_pop.sample_selection import (
    _read_selection_parent_cache,
    _selection_parent_cache_path,
    _write_selection_parent_cache,
)

_PHOTO_ID = "20260826T234425Z_3d3f740b080c"
_FLAME_COLUMNS = ("mass_flame", "mass_flame_upper", "mass_flame_lower")


def main(argv: list[str] | None = None) -> int:
    del argv  # no options today; kept for parity with the fetch/attach scripts
    enrich_cache = _selection_parent_cache_path(f"{_PHOTO_ID}+enrich")
    flame_meta = repo_root() / "data/dr3/gaia_snapshots/flame_enrichment/meta.yaml"
    flame_ecsv = repo_root() / "data/dr3/gaia_snapshots/flame_enrichment/query.ecsv"
    if not enrich_cache.is_file():
        print(f"missing enrich cache {enrich_cache}", file=sys.stderr)
        return 1
    if not flame_meta.is_file() or not flame_ecsv.is_file():
        print(
            "missing flame_enrichment fetch — run "
            "scripts/fetch_flame_enrichment.py (launch, then --poll-job) first",
            file=sys.stderr,
        )
        return 1

    print(f"loading {enrich_cache} …", flush=True)
    rows = _read_selection_parent_cache(enrich_cache)

    print(f"loading {flame_ecsv} …", flush=True)
    flame_table = Table.read(flame_ecsv, format="ascii.ecsv")
    flame_cols = list(flame_table.colnames)
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for frow in flame_table:
        mapping = {name: frow[name] for name in flame_cols}
        by_key[_enrichment_join_key(mapping)] = mapping

    n_matched = 0
    n_with_flame = 0
    merged_rows: list[dict[str, object]] = []
    for row in rows:
        key = _enrichment_join_key(row)
        flame_row = by_key.get(key)
        if flame_row is not None:
            n_matched += 1
            row = merge_nss_enrichment_into_row(row, flame_row)
            if row.get("mass_flame") is not None:
                n_with_flame += 1
        merged_rows.append(row)

    out_cache = _selection_parent_cache_path(f"{_PHOTO_ID}+enrich+flame")
    print(f"writing {out_cache} …", flush=True)
    _write_selection_parent_cache(out_cache, merged_rows)
    print(
        f"done n_rows={len(merged_rows)} n_matched_by_key={n_matched} "
        f"n_with_mass_flame={n_with_flame}",
        flush=True,
    )
    if n_with_flame == 0:
        print(
            "WARNING: 0 rows carry a mass_flame value after merge — check "
            "flame_enrichment/meta.yaml's n_with_mass_flame and the join key "
            "match rate above before proceeding to attach_mc_to_selection_cache.py",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
