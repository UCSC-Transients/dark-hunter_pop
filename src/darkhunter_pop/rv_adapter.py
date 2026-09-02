"""Load ``dark-hunter_rv`` JSON summaries into ``CandidateRecord.rv_summary``.

ARCHITECTURE.md §4 / Phase 1 #5: summaries are read from a configurable root directory
(``dr3.rv_summary_root``) so tests and pipeline runs can use fixture trees without
touching live ``dark-hunter_rv`` output directories.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import CandidateRecord


def resolve_rv_summary_path(config: PipelineConfig, source_id: int) -> Path | None:
    """Return the configured JSON path for one Gaia ``source_id``, or ``None`` if disabled."""
    dr = config.active_dr()
    root_spec = dr.rv_summary_root
    if root_spec is None:
        return None
    root = Path(root_spec)
    if not root.is_absolute():
        root = repo_root() / root
    return root / dr.rv_summary_filename_template.format(source_id=source_id)


def load_rv_summary_json(path: Path) -> dict[str, Any]:
    """Read one dark-hunter_rv summary JSON file."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"rv summary must be a JSON object: {path}")
    return data


def attach_rv_summaries(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Merge per-source RV JSON summaries into ``CandidateRecord.rv_summary`` when present."""
    dr = config.active_dr()
    if dr.rv_summary_root is None:
        return list(candidates), {"attached": 0, "missing": 0, "disabled": len(candidates)}

    attached = 0
    missing = 0
    updated: list[CandidateRecord] = []
    for candidate in candidates:
        path = resolve_rv_summary_path(config, candidate.source_id)
        if path is None or not path.is_file():
            missing += 1
            updated.append(candidate)
            continue
        summary = load_rv_summary_json(path)
        attached += 1
        updated.append(candidate.model_copy(update={"rv_summary": summary}))
    return updated, {
        "attached": attached,
        "missing": missing,
        "disabled": 0,
    }
