"""What the duplicate ``source_id``s in the uncut parent snapshot actually are.

Written by the verification agent (roster #49) while verifying PR #238, which was
reverted. It pins facts about the data, not about any implementation, so it survives
the revert and constrains whatever re-lands under #231 / #221.

PR #238 diagnosed the duplication as **cross-match fan-out** — a Gaia source with more
than one accepted neighbour in a ``*_best_neighbour`` table producing several
otherwise-identical joined rows — and merged the rows cell-by-cell on that basis. The
measurement below shows that diagnosis covers 6 of the 5,932 duplicated ``source_id``s
and misses the other 5,926. The two populations partition the duplicates exactly:

* **6 groups** are genuine cross-match fan-out — same ``nss_solution_type``, same
  ``period``, differing external photometry (two accepted 2MASS neighbours).
* **5,926 groups** agree on their photometry and disagree on ``nss_solution_type``,
  ``period``, ``eccentricity`` and ``t_periastron``: they are **distinct NSS orbital
  solutions for one source** (typically an astrometric ``Orbital`` solution and a
  spectroscopic ``SB1`` one, with unrelated periods — e.g. 227 d vs 0.62 d).

The consequence that forced the revert: a cell-wise merge fills the ``Orbital`` row's
masked ``eccentricity_error`` from the ``SB1`` row, producing an orbit that exists in
no Gaia row, and — because "``Orbital*`` with null σ_e" is exactly the §7.2.5
pseudo-circular criterion — flips that flag for every source it touches.

Any future collapse must therefore choose a solution by a documented
``nss_solution_type`` priority and must **not** mix cells across solutions.

Marked ``slow``: streams a ~296 MB ECSV off the local ``data/`` tree, which no CI
runner has. Skipped when the snapshot is absent.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.data_acquisition import gaia_snapshots_dir

pytestmark = pytest.mark.slow

UNCUT_SNAPSHOT_ID = "20260826T234425Z_3d3f740b080c"

# Measured independently (raw first-column scan of the ECSV body) under #231/#238.
EXPECTED_TOTAL_ROWS = 443_211
EXPECTED_UNIQUE_SOURCE_IDS = 437_275
EXPECTED_DUPLICATE_SOURCE_IDS = 5_932
EXPECTED_REDUNDANT_ROWS = 5_936
EXPECTED_MAX_MULTIPLICITY = 3

# Of the duplicated source_ids, how many are genuine cross-match fan-out (same NSS
# solution, differing external photometry) versus several distinct NSS orbital
# solutions for one source. PR #238's merge rule assumed all 5,932 were the former.
EXPECTED_FANOUT_GROUPS = 6
EXPECTED_MULTI_SOLUTION_GROUPS = 5_926

# Columns a cross-match fan-out would differ on, and columns only a different
# orbital solution would differ on.
PHOTOMETRY_COLUMNS = ("g_mag", "bp_mag", "rp_mag", "J_mag", "H_mag", "Ks_mag")
ORBIT_COLUMNS = ("nss_solution_type", "period", "eccentricity", "t_periastron")


def _query_path() -> Path:
    return gaia_snapshots_dir(load_config()) / UNCUT_SNAPSHOT_ID / "query.ecsv"


def _require_snapshot() -> Path:
    path = _query_path()
    if not path.is_file():
        pytest.skip(f"uncut parent snapshot not present at {path}")
    return path


def test_uncut_parent_snapshot_duplicate_counts() -> None:
    """Pin the duplicate counts by a streaming scan independent of astropy."""
    path = _require_snapshot()
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

    total = sum(counts.values())
    duplicated = {sid: n for sid, n in counts.items() if n > 1}
    assert total == EXPECTED_TOTAL_ROWS
    assert len(counts) == EXPECTED_UNIQUE_SOURCE_IDS
    assert len(duplicated) == EXPECTED_DUPLICATE_SOURCE_IDS
    assert total - len(counts) == EXPECTED_REDUNDANT_ROWS
    assert max(duplicated.values()) == EXPECTED_MAX_MULTIPLICITY


def test_duplicates_are_distinct_orbital_solutions_not_crossmatch_fanout() -> None:
    """99.9% of the duplicates are several orbits, not fan-out (#231, #238).

    This is the fact that invalidates a fan-out-shaped merge rule.
    """
    from astropy.table import Table

    path = _require_snapshot()
    table = Table.read(path, format="ascii.ecsv")
    ids = np.asarray(np.ma.getdata(table["source_id"])).astype(np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    duplicated = unique[counts > 1]
    assert duplicated.size == EXPECTED_DUPLICATE_SOURCE_IDS

    groups: dict[int, list[int]] = {}
    for row in np.flatnonzero(np.isin(ids, duplicated)):
        groups.setdefault(int(ids[row]), []).append(int(row))

    fanout = set()
    multi_solution = set()
    for source_id, rows in groups.items():
        if any(
            len({str(table[name][r]) for r in rows}) > 1 for name in PHOTOMETRY_COLUMNS
        ):
            fanout.add(source_id)
        if any(
            len({str(table[name][r]) for r in rows}) > 1 for name in ORBIT_COLUMNS
        ):
            multi_solution.add(source_id)

    # The two populations partition the duplicates exactly, and the fan-out the
    # PR #238 merge rule was designed around is 0.1% of them.
    assert fanout & multi_solution == set()
    assert fanout | multi_solution == set(groups)
    assert len(fanout) == EXPECTED_FANOUT_GROUPS, (
        f"cross-match fan-out groups changed: {len(fanout)}"
    )
    assert len(multi_solution) == EXPECTED_MULTI_SOLUTION_GROUPS, (
        f"multi-orbital-solution groups changed: {len(multi_solution)}"
    )


def test_masked_eccentricity_error_is_the_pseudo_circular_criterion() -> None:
    """Filling σ_e across solutions would flip the §7.2.5 pseudo-circular flag.

    Pins the specific corruption PR #238 was reverted for: for every duplicated
    source whose surviving ``Orbital*`` row has a masked ``eccentricity_error``, some
    *other* row of the group carries a non-null one belonging to a different orbit.
    Any merge that fills that cell silently reclassifies the source.
    """
    from astropy.table import Table

    from darkhunter_pop.data_acquisition import _is_gaia_pseudo_circular

    path = _require_snapshot()
    config = load_config()
    table = Table.read(path, format="ascii.ecsv")
    colnames = list(table.colnames)
    ids = np.asarray(np.ma.getdata(table["source_id"])).astype(np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    duplicated = unique[counts > 1]

    groups: dict[int, list[int]] = {}
    for row in np.flatnonzero(np.isin(ids, duplicated)):
        groups.setdefault(int(ids[row]), []).append(int(row))

    err = np.ma.getmaskarray(table["eccentricity_error"])
    at_risk = 0
    for rows in groups.values():
        masked = [r for r in rows if err[r]]
        filled = [r for r in rows if not err[r]]
        if not masked or not filled:
            continue
        for r in masked:
            if _is_gaia_pseudo_circular(
                {name: table[name][r] for name in colnames}, config.dr3
            ):
                at_risk += 1

    # Non-zero is the point: these are exactly the rows a cross-solution fill would
    # silently reclassify. Measured at 102 on this snapshot.
    assert at_risk > 0
    assert at_risk == 102, f"pseudo-circular rows at risk changed: {at_risk}"
