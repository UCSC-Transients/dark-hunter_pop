"""Shahaf et al. (2023b) class-III catalog: frozen snapshot I/O (#315).

Shahaf, Bashi, Mazeh et al. (2023, MNRAS 518, 2991), Table 2: the 177 "highly
probable class-III systems" of their probabilistic AMRF triage, as distributed
by VizieR (``J/MNRAS/518/2991/table2``). El-Badry et al. (2026) subsample 4
(``sub_chandrasekhar``) is this catalog with the paper's Eq. 5 cuts applied to
Shahaf's own ``M2min`` / ``e_M2min`` (``docs/SELECTION_REPRODUCTION_STATUS.md``
§3.3.7). The catalog is a **reproduction-mode membership table only**: it
cannot be applied to mock catalogs (§15 Q2), so ``forward_model`` evaluation
never reads it, and it is never a prior or an inference input.

A snapshot is a directory ``{data_root}/{dr}/external_catalogs/<name>/`` holding
``table2.ecsv`` and a ``meta.yaml`` (VizieR catalog id, query date, row count,
SHA-256 of ``table2.ecsv``), written by ``scripts/fetch_shahaf2023b_class3.py``.
``<name>`` is the DR-path key ``dr3.shahaf2023b_class3_snapshot`` (``dr4``: null).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

#: Registry id of this catalog, used by ``SampleSubsample.external_catalog`` and
#: as the ``in_sample('...')`` membership key.
SHAHAF2023B_CLASS3_CATALOG_ID = "shahaf2023b_class3"
#: VizieR table the snapshot is taken from.
SHAHAF2023B_VIZIER_TABLE = "J/MNRAS/518/2991/table2"
#: Snapshot data file and metadata file names.
SHAHAF2023B_DATA_FILE = "table2.ecsv"
SHAHAF2023B_META_FILE = "meta.yaml"
#: Published row count of Table 2.
SHAHAF2023B_TABLE2_N_ROWS = 177

#: VizieR column → row column merged onto selection rows (reproduction mode only).
#: Row names are prefixed so they can never collide with El-Badry-owned
#: ``m2_tilde_msun`` / ``sigma_m2_astrometric_msun`` or Andrews-owned columns.
SHAHAF2023B_ROW_COLUMNS: dict[str, str] = {
    "M2min": "shahaf2023b_m2min_msun",
    "e_M2min": "shahaf2023b_m2min_error_msun",
    "M1": "shahaf2023b_m1_msun",
    "P": "shahaf2023b_period_day",
    "Class": "shahaf2023b_class",
}


class Shahaf2023bCatalogError(RuntimeError):
    """The configured Shahaf 2023b snapshot is missing, altered or malformed."""


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_shahaf2023b_class3(snapshot_dir: Path) -> dict[int, dict[str, Any]]:
    """Read a Table 2 snapshot keyed by Gaia DR3 ``source_id``.

    The ``meta.yaml`` checksum and row count are verified; a mismatch raises
    rather than loading an altered table. Values are the row columns of
    :data:`SHAHAF2023B_ROW_COLUMNS`.
    """
    from astropy.table import Table

    meta_path = snapshot_dir / SHAHAF2023B_META_FILE
    data_path = snapshot_dir / SHAHAF2023B_DATA_FILE
    if not meta_path.is_file() or not data_path.is_file():
        raise Shahaf2023bCatalogError(
            f"Shahaf 2023b snapshot {snapshot_dir} is incomplete: need "
            f"{SHAHAF2023B_META_FILE} and {SHAHAF2023B_DATA_FILE} "
            "(scripts/fetch_shahaf2023b_class3.py)"
        )
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    expected = meta.get("sha256")
    got = file_sha256(data_path)
    if expected != got:
        raise Shahaf2023bCatalogError(
            f"{data_path} checksum {got} does not match meta.yaml sha256 {expected}"
        )
    table = Table.read(data_path, format="ascii.ecsv")
    if len(table) != meta.get("row_count"):
        raise Shahaf2023bCatalogError(
            f"{data_path} has {len(table)} rows; meta.yaml says {meta.get('row_count')}"
        )
    out: dict[int, dict[str, Any]] = {}
    for row in table:
        sid = int(row["GaiaDR3"])
        if sid in out:
            raise Shahaf2023bCatalogError(f"duplicate GaiaDR3 {sid} in {data_path}")
        out[sid] = {dst: _clean(row[src]) for src, dst in SHAHAF2023B_ROW_COLUMNS.items()}
    return out


def merge_shahaf2023b_columns(
    rows: list[Mapping[str, Any]], catalog: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Copy rows, adding the catalog's row columns to catalog members.

    Non-members get no Shahaf columns; the membership cut
    (``in_sample('shahaf2023b_class3')``) removes them before any cut reads one.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        extra = catalog.get(int(row["source_id"]))
        if extra:
            for key, value in extra.items():
                merged.setdefault(key, value)
        out.append(merged)
    return out
