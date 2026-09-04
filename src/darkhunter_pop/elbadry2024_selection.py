"""El-Badry et al. (2024) selection helpers (CONTINUATION_PLAN §7 / §4.8).

Catalog-level membership and the G<15 follow-up gate live in
``config/selections/elbadry2024.yaml``. Outcome-dependent criteria (b)–(d)
belong to the Poisson inclusion operator and are documented here only as
named constants sourced from that file — never applied as data filters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import SampleSelectionFile
from darkhunter_pop.sample_selection import load_sample_selection_file

PUBLISHED_NATURE: str = "compact_object_candidate"
TABLE3_RELATIVE: str = "config/selections/external/elbadry2024_table3.yaml"
SELECTION_RELATIVE: str = "config/selections/elbadry2024.yaml"


def load_elbadry2024_spec(
    path: str | Path | None = None,
) -> SampleSelectionFile:
    """Load the frozen El-Badry 2024 selection file."""
    selection_path = (
        Path(path)
        if path is not None
        else repo_root() / SELECTION_RELATIVE
    )
    if not selection_path.is_absolute():
        selection_path = repo_root() / selection_path
    return load_sample_selection_file(selection_path)


def load_elbadry2024_table3(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """All NS candidates (Andrews 2022 + Shahaf 2023b) from Table 3."""
    table_path = (
        Path(path)
        if path is not None
        else repo_root() / TABLE3_RELATIVE
    )
    if not table_path.is_absolute():
        table_path = repo_root() / table_path
    raw = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "data" not in raw:
        raise ValueError(f"elbadry2024_table3 missing data: {table_path}")
    return list(raw["data"])


def table3_g_lt_15_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    g_mag_faint_limit: float,
) -> list[dict[str, Any]]:
    """Table 3 rows brighter than the follow-up magnitude limit (criterion a)."""
    data = list(rows) if rows is not None else load_elbadry2024_table3()
    return [dict(row) for row in data if float(row["g_mag"]) < g_mag_faint_limit]


def published_sample_source_ids(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[int, ...]:
    """The 21 published systems: genuine + compact_object_candidate."""
    data = list(rows) if rows is not None else load_elbadry2024_table3()
    ids = [
        int(row["source_id"])
        for row in data
        if row.get("verdict") == "genuine"
        and row.get("nature") == PUBLISHED_NATURE
    ]
    return tuple(ids)


def spurious_fraction_g_lt_15(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    g_mag_faint_limit: float,
) -> tuple[float, int, int]:
    """Fixture rate the shared #27 model must reproduce — not a selection input.

    Returns ``(fraction, n_spurious, n_g_lt_15)``.
    """
    bright = table3_g_lt_15_rows(rows, g_mag_faint_limit=g_mag_faint_limit)
    n_bright = len(bright)
    n_spurious = sum(1 for row in bright if row.get("verdict") == "spurious")
    if n_bright == 0:
        raise ValueError("no Table 3 rows with G < g_mag_faint_limit")
    return float(n_spurious) / float(n_bright), n_spurious, n_bright


def g_mag_faint_limit_from_spec(spec: SampleSelectionFile) -> float:
    """Read criterion (a) threshold from the frozen cut parameters."""
    if not spec.branches:
        raise ValueError("elbadry2024 selection has no branches")
    for sub in spec.branches[0].subsamples or []:
        for cut in sub.cuts:
            if cut.id == "g_mag_limit":
                value = cut.parameters.get("g_mag_faint_limit")
                if isinstance(value, (int, float)):
                    return float(value)
    raise ValueError("elbadry2024 g_mag_limit cut missing g_mag_faint_limit")


def inclusion_criteria_ids(spec: SampleSelectionFile) -> tuple[str, ...]:
    """Named outcome-dependent criteria recorded on the inclusion operator."""
    op = spec.inclusion_operator
    if op is None or not op.notes:
        return ()
    notes = op.notes
    found: list[str] = []
    for label in ("(b)", "(c)", "(d)"):
        if label in notes:
            found.append(label)
    return tuple(found)
