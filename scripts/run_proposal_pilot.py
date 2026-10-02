#!/usr/bin/env python3
"""Generate one proposal-set generation and run it through gaiamock_mod (#391 pilot).

docs/MOCK_POPULATION_SPEC.md §3, §7. Draws ``proposal.n_draws`` systems from the proposal
in ``config/population/proposal_set_pilot.yaml`` over a ``gaia_source`` parent snapshot, runs each
through ``gaiamock.run_full_astrometric_cascade`` (seeded per draw, #371) in a process
pool, and writes truth + outcome to one HDF5 artifact keyed by the fragment fingerprint.

Partial results stream to ``<out>.partial.jsonl`` so an interrupted run resumes without
redoing finished draws (each draw's outcome depends only on its own seeds).

Usage::

    .venv/bin/python scripts/run_proposal_pilot.py \
        --parent-dir data/dr3/gaia_snapshots/<id>_gaia_source_parent_K1000000_plx0p2 \
        --out output/proposal_set/pilot.h5 --workers 4 [--n-draws 50]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# One BLAS thread per worker (the cascade is single-threaded numerics).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.proposal_set import (
    load_parent_snapshot,
    load_proposal_set_fragment,
    sample_proposal,
    simulate_one,
    write_proposal_artifact,
)

_WORKER: dict[str, Any] = {}


def _init_worker(config_path: str, fragment_path: str) -> None:
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    cfg = load_config(Path(config_path))
    frag = load_proposal_set_fragment(fragment_path)
    gm = import_gaiamock_mod()
    _WORKER.update(
        gaiamock=gm,
        c_funcs=gm.read_in_C_functions(),
        proposal=frag.proposal,
        cuts=cfg.active_dr().selection_function_astrometric.orbital_solution_cuts,
    )


def _work(draw: dict[str, Any]) -> dict[str, Any]:
    return simulate_one(
        draw,
        gaiamock=_WORKER["gaiamock"],
        c_funcs=_WORKER["c_funcs"],
        cfg=_WORKER["proposal"],
        cuts=_WORKER["cuts"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_pilot.yaml"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-draws", type=int, default=None, help="override proposal.n_draws")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    frag = load_proposal_set_fragment(args.fragment)
    if args.n_draws is not None:
        frag = frag.model_copy(update={"proposal": frag.proposal.model_copy(update={"n_draws": args.n_draws})})
    prop = frag.proposal
    parent = load_parent_snapshot(args.parent_dir, cfg, parallax_floor_mas=prop.provisional_parallax_floor_mas)
    print(
        f"parent: {parent.n_rows} rows, {int(parent.usable.sum())} usable "
        f"(TAG10 M1 resolved, parallax > {prop.provisional_parallax_floor_mas} mas); "
        f"scale_to_full = {parent.scale_to_full:.1f}"
    )
    truth = sample_proposal(parent, prop)
    n = int(truth["draw_index"].size)

    partial = args.out.with_suffix(".partial.jsonl")
    done: dict[int, dict[str, Any]] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done[int(rec["draw_index"])] = rec
    todo = [i for i in range(n) if int(truth["draw_index"][i]) not in done]
    print(f"draws: {n}; already done: {len(done)}; to run: {len(todo)}; workers: {args.workers}")

    keys = [k for k, v in truth.items() if np.asarray(v).ndim == 1]
    payload = [{k: (truth[k][i].item() if hasattr(truth[k][i], "item") else truth[k][i]) for k in keys} for i in todo]
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a") as sink, ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(str(args.config), str(args.fragment))
    ) as pool:
        futures = [pool.submit(_work, d) for d in payload]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            done[rec["draw_index"]] = rec
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if k % 50 == 0 or k == len(futures):
                el = time.time() - t0
                print(f"  {k}/{len(futures)} done, {el / 60:.1f} min wall", flush=True)
    wall = time.time() - t0

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    from darkhunter_pop.gaiamock_vendor import read_versions

    provenance = {
        "issue": 391,
        "code_commit": commit,
        "gaiamock_versions": read_versions(),
        "wall_seconds_this_invocation": wall,
        "workers": args.workers,
        "target_mds17_json": frag.target_mds17.model_dump(mode="json"),
        "label": "PILOT (provisional settings, not decisions; spec §7)",
    }
    path = write_proposal_artifact(
        args.out, truth, list(done.values()), fragment=frag, parent=parent, provenance=provenance
    )
    cpu = np.array([r["cpu_seconds"] for r in done.values()])
    acc = sum(bool(r["accepted_orbital"]) for r in done.values())
    print(f"wrote {path}; accepted orbits {acc}/{n}; CPU s per draw mean {cpu.mean():.2f}, median {np.median(cpu):.2f}; wall {wall / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
