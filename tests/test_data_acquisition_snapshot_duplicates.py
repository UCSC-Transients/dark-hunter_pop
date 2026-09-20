"""Duplicate-``source_id`` regression check against the documented uncut snapshot.

Deliberately its own module, and deliberately **not** ``unit``: it streams a ~296 MB
ECSV off the local ``data/`` tree, which no CI runner has. A module-level
``pytestmark`` is what actually keeps a test out of the required merge gate — a
per-test marker would be additive, not exclusive (see
``tests/test_data_acquisition_live.py``).

The pinned number is the measurement taken under issue #231 on
``20260826T234425Z_3d3f740b080c``, the uncut parent snapshot every literature-sample
parent query runs against (CLAUDE.md, Gotchas):

    total data rows        443,211
    unique source_ids      437,275
    duplicated source_ids    5,932
    redundant rows           5,936
    max multiplicity             3

Skipped when the snapshot is not present on this host.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.data_acquisition import gaia_snapshots_dir

pytestmark = pytest.mark.slow

UNCUT_SNAPSHOT_ID = "20260826T234425Z_3d3f740b080c"

# Measured under #231; see the module docstring.
EXPECTED_TOTAL_ROWS = 443_211
EXPECTED_UNIQUE_SOURCE_IDS = 437_275
EXPECTED_DUPLICATE_SOURCE_IDS = 5_932
EXPECTED_REDUNDANT_ROWS = 5_936
EXPECTED_MAX_MULTIPLICITY = 3


def _snapshot_query_path() -> Path:
    return gaia_snapshots_dir(load_config()) / UNCUT_SNAPSHOT_ID / "query.ecsv"


def _source_id_multiplicities(path: Path) -> Counter[str]:
    """Count rows per ``source_id`` by streaming the ECSV body's first column.

    Limitation: assumes the ECSV is space-delimited with ``source_id`` first, which
    is how ``save_gaia_snapshot`` writes this stage's table. Streaming keeps the peak
    memory to the id counter rather than a parsed 296 MB table.
    """
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        header_seen = False
        for line in handle:
            if line.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                assert line.split()[0] == "source_id", "unexpected ECSV column order"
                continue
            if not line.strip():
                continue
            counts[line.split(" ", 1)[0]] += 1
    return counts


def test_uncut_parent_snapshot_duplicate_source_id_count() -> None:
    """Pin the measured fan-out on the real uncut parent snapshot (#221, #231)."""
    path = _snapshot_query_path()
    if not path.is_file():
        pytest.skip(f"uncut parent snapshot not present at {path}")

    counts = _source_id_multiplicities(path)
    total_rows = sum(counts.values())
    duplicated = {sid: n for sid, n in counts.items() if n > 1}

    assert total_rows == EXPECTED_TOTAL_ROWS
    assert len(counts) == EXPECTED_UNIQUE_SOURCE_IDS
    assert len(duplicated) == EXPECTED_DUPLICATE_SOURCE_IDS
    assert total_rows - len(counts) == EXPECTED_REDUNDANT_ROWS
    assert max(duplicated.values()) == EXPECTED_MAX_MULTIPLICITY


def test_collapse_makes_the_uncut_parent_snapshot_unique() -> None:
    """The collapse turns that snapshot into exactly one row per ``source_id``."""
    from astropy.table import Table

    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    path = _snapshot_query_path()
    if not path.is_file():
        pytest.skip(f"uncut parent snapshot not present at {path}")

    table = Table.read(path, format="ascii.ecsv")
    collapsed, report = collapse_duplicate_source_ids(table)

    assert report.rows_in == EXPECTED_TOTAL_ROWS
    assert report.duplicate_source_ids == EXPECTED_DUPLICATE_SOURCE_IDS
    assert report.rows_removed == EXPECTED_REDUNDANT_ROWS
    assert report.max_multiplicity == EXPECTED_MAX_MULTIPLICITY
    assert report.rows_out == EXPECTED_UNIQUE_SOURCE_IDS
    assert len(set(int(v) for v in collapsed["source_id"])) == len(collapsed)
