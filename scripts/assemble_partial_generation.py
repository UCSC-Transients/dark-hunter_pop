#!/usr/bin/env python3
"""Assemble an interrupted proposal generation into a usable artifact (#391).

A generation stopped part-way (e.g. the paused gen-11 full run) leaves its outcomes in
``<out>.partial.jsonl``. The truth table is deterministic (``sample_proposal`` with the same
config and seed), so it is regenerated, cut to the **contiguous prefix** of completed
``draw_index`` values, and written with ``proposal.n_draws`` set to that prefix length. Each
truth row is an iid draw from ``q``, so the prefix is an iid sample of size ``N`` and the
deterministic-mixture weights use ``n_j = N`` (spec §3.7). The planned size is kept in the
provenance. Outcomes past the prefix (finished out of order) are left out and counted.

Usage::

    .venv/bin/python scripts/assemble_partial_generation.py --parent-dir <snapshot> \
        --fragment config/population/proposal_set_decided_full.yaml --draw-index-offset 2500 \
        --partial output/proposal_set/decided_gen11_full.partial.jsonl \
        --out output/proposal_set/decided_gen11_partial.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.proposal_set import (
    load_parent_snapshot,
    load_proposal_set_fragment,
    sample_proposal,
    write_proposal_artifact,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, required=True)
    ap.add_argument("--draw-index-offset", type=int, required=True)
    ap.add_argument("--partial", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    frag = load_proposal_set_fragment(args.fragment)
    prop = frag.proposal
    parent = load_parent_snapshot(args.parent_dir, cfg, prop)
    done: dict[int, dict] = {}
    n_bad = 0
    for line in args.partial.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        done[int(rec["draw_index"])] = rec
    off = args.draw_index_offset
    n_prefix = 0
    while (off + n_prefix) in done:
        n_prefix += 1
    if n_prefix == 0:
        raise SystemExit("no contiguous completed draws from the offset")
    truth_full = sample_proposal(parent, prop, draw_index_offset=off, config=cfg)
    truth = {k: np.asarray(v)[:n_prefix] for k, v in truth_full.items()}
    outcomes = [done[off + i] for i in range(n_prefix)]
    planned = prop.n_draws
    frag_cut = frag.model_copy(update={"proposal": prop.model_copy(update={"n_draws": n_prefix})})
    provenance = {
        "issue": 391,
        "decision_ref": prop.decision_ref,
        "parent_attrition": parent.attrition(),
        "target_mds17_json": frag.target_mds17.model_dump(mode="json"),
        "assembled_from_partial": str(args.partial),
        "planned_n_draws": planned,
        "contiguous_prefix_n_draws": n_prefix,
        "completed_outside_prefix": len(done) - n_prefix,
        "truncated_lines": n_bad,
        "note": "interrupted generation; prefix of iid draws, weights use n_j = prefix length",
    }
    path = write_proposal_artifact(args.out, truth, outcomes, fragment=frag_cut, parent=parent, provenance=provenance)
    print(f"wrote {path}: {n_prefix} of {planned} planned draws ({len(done) - n_prefix} completed past the prefix, {n_bad} truncated lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
