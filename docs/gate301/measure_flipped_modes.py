"""Gate 301 supplementary measurement: every sample in the mode the stage did not run.

The ``sample_selection`` stage evaluates each registry entry in exactly one mode
(``config/config.yaml``: andrews2022 reproduction, andrews2022_modified
forward_model, elbadry2024 reproduction, elbadry2026 reproduction). Issue #301's
reproduction table also needs the other mode of each. This script re-evaluates
the *same parent rows the stage used* (``load_selection_rows_from_manifest`` on
the gate-301 run manifest, read-only) with each sample's mode flipped, through
the same ``run_sample_selection`` entry point. No threshold, selection file or
cache is changed; the config differs from the run's only in the registry
``mode`` fields.

Usage (primary checkout, real ``data/``)::

    .venv/bin/python docs/gate301/measure_flipped_modes.py \\
        --run-file runs/<run_id>.yaml --config docs/gate301/config_gate301.yaml \\
        --out <scratch>/flipped_modes.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.run_management import load_run_manifest
from darkhunter_pop.sample_selection import (
    load_selection_rows_from_manifest,
    run_sample_selection,
)

FLIP = {"reproduction": "forward_model", "forward_model": "reproduction"}


def _summary(res: Any) -> dict[str, Any]:
    d = res.as_dict()
    keep = (
        "mode",
        "n_parent",
        "n_surviving",
        "n_distinct_source_ids",
        "n_stars_per_solution_type",
        "route_counts",
    )
    out = {k: d.get(k) for k in keep}
    out["surviving_source_ids"] = sorted(int(s) for s in d["surviving_source_ids"])
    out["branch_counts"] = {
        b: len(set(v)) for b, v in (d.get("branch_surviving") or {}).items()
    }
    out["subsample_counts"] = {
        b: len(set(v)) for b, v in (d.get("subsample_surviving") or {}).items()
    }
    out["branch_surviving_by_solution_type"] = d.get("branch_surviving_by_solution_type")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-file", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = load_config(args.config, host_profile="laptop")
    manifest = load_run_manifest(args.run_file)
    rows = load_selection_rows_from_manifest(manifest)

    samples = []
    for entry in config.sample_selection.samples:
        new = entry.model_copy()
        if entry.enabled:
            new = entry.model_copy(update={"mode": type(entry.mode)(FLIP[entry.mode.value])})
        samples.append(new)
    sel = config.sample_selection.model_copy(update={"samples": samples})
    flipped = config.model_copy(update={"sample_selection": sel})

    result = run_sample_selection(rows, flipped)
    payload = {
        "run_id": manifest.run_id,
        "n_rows": len(rows),
        "modes": {e.name: e.mode.value for e in samples if e.enabled},
        "results": {name: _summary(r) for name, r in result.results.items()},
    }
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for name, r in payload["results"].items():
        print(
            name,
            r["mode"],
            "n_surviving",
            r["n_surviving"],
            "per_type",
            r["n_stars_per_solution_type"],
            "branches",
            r["branch_counts"],
            "subsamples",
            r["subsample_counts"],
            "routes",
            r["route_counts"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
