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
import itertools
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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


def blas_thread_report() -> list[dict[str, Any]]:
    """Per-library BLAS/OpenMP thread counts as threadpoolctl sees them (#408)."""
    from threadpoolctl import threadpool_info

    return [
        {"user_api": i.get("user_api"), "internal_api": i.get("internal_api"), "num_threads": i.get("num_threads")}
        for i in threadpool_info()
    ]


def build_epoch_setup(cfg: Any, prop: Any) -> Any:
    """The #400 epoch model for ``simulate_one`` (None when ``epoch_model: off``).

    Uses ``dr3.epoch_model`` from config with ``enabled`` forced on: #400 E1 was decided
    "on" (#391 issuecomment-5971280434). ``simulate_one`` routes every draw through
    ``epoch_model.run_cascade`` (epochs, per-CCD noise and the RUWE normalization the config
    switches on), with ``epoch_model_rng(base_seed, stream, draw_index)`` Generators.
    """
    if prop.epoch_model == "off":
        return None
    import dataclasses

    from darkhunter_pop import epoch_model as em
    from darkhunter_pop.proposal_set import EpochSetup

    section = getattr(cfg.active_dr(), "epoch_model", None)
    if section is None:
        raise ValueError("epoch_model: dr3_config but the active DR path has no epoch_model section")
    emc = dataclasses.replace(em.epoch_model_config_from_mapping(section), enabled=True)
    return EpochSetup(config=emc, gaps_jd=em.gap_intervals_jd(emc))


def _init_worker(config_path: str, fragment_path: str, niceness: int) -> None:
    from threadpoolctl import threadpool_limits

    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    if niceness:
        os.nice(niceness)
    # Pin every BLAS/OpenMP pool to one thread in this worker (#408); verified in _work.
    _WORKER["blas_limits"] = threadpool_limits(limits=1)
    cfg = load_config(Path(config_path))
    frag = load_proposal_set_fragment(fragment_path)
    gm = import_gaiamock_mod()
    _WORKER.update(
        gaiamock=gm,
        c_funcs=gm.read_in_C_functions(),
        proposal=frag.proposal,
        cuts=cfg.active_dr().selection_function_astrometric.orbital_solution_cuts,
        epoch=build_epoch_setup(cfg, frag.proposal),
    )


def _work(draw: dict[str, Any]) -> dict[str, Any]:
    rec = simulate_one(
        draw,
        gaiamock=_WORKER["gaiamock"],
        c_funcs=_WORKER["c_funcs"],
        cfg=_WORKER["proposal"],
        cuts=_WORKER["cuts"],
        epoch=_WORKER["epoch"],
    )
    if "blas_checked" not in _WORKER:
        _WORKER["blas_checked"] = True
        rec["blas_threads"] = blas_thread_report()
        rec["worker_niceness"] = os.nice(0)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_pilot.yaml"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-draws", type=int, default=None, help="override proposal.n_draws")
    ap.add_argument(
        "--draw-index-offset", type=int, default=0,
        help="first global draw_index of this generation (indices are never reused, spec §3.3)",
    )
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--nice", type=int, default=10, help="os.nice increment for workers (#408)")
    ap.add_argument("--min-free-gib", type=float, default=3.0, help="stop (resumable) below this")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    frag = load_proposal_set_fragment(args.fragment)
    if args.n_draws is not None:
        frag = frag.model_copy(update={"proposal": frag.proposal.model_copy(update={"n_draws": args.n_draws})})
    prop = frag.proposal
    parent = load_parent_snapshot(args.parent_dir, cfg, prop)
    print(f"parent attrition (cumulative, spec §0.1): {parent.attrition()}")
    print(
        f"parent: {parent.n_rows} rows, {int(parent.usable.sum())} usable, "
        f"{int((parent.is_giant & parent.usable).sum())} usable giants (flag only); "
        f"scale_to_full = {parent.scale_to_full:.1f}; decision_ref: {prop.decision_ref}"
    )
    truth = sample_proposal(parent, prop, draw_index_offset=args.draw_index_offset)
    n = int(truth["draw_index"].size)

    partial = args.out.with_suffix(".partial.jsonl")
    done: dict[int, dict[str, Any]] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line truncated by a crash; that draw simply reruns
            done[int(rec["draw_index"])] = rec
    todo = [i for i in range(n) if int(truth["draw_index"][i]) not in done]
    print(f"draws: {n}; already done: {len(done)}; to run: {len(todo)}; workers: {args.workers}")

    keys = [k for k, v in truth.items() if np.asarray(v).ndim == 1]

    def payload(i: int) -> dict[str, Any]:
        return {k: (truth[k][i].item() if hasattr(truth[k][i], "item") else truth[k][i]) for k in keys}

    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window = max(1, args.workers) * 8  # bounded in-flight futures: memory stays flat at any n
    queue = iter(todo)
    n_todo = len(todo)
    k = 0
    with partial.open("a") as sink, ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(str(args.config), str(args.fragment), int(args.nice)),
    ) as pool:
        pending = {pool.submit(_work, payload(i)) for i in itertools.islice(queue, window)}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                rec = fut.result()
                done[rec["draw_index"]] = rec
                sink.write(json.dumps(rec) + "\n")
                k += 1
            sink.flush()
            pending |= {pool.submit(_work, payload(i)) for i in itertools.islice(queue, len(finished))}
            if k % args.progress_every < len(finished) or not pending:
                free_gib = shutil.disk_usage(args.out.parent).free / 2**30
                el = time.time() - t0
                print(f"  {k}/{n_todo} done, {el / 60:.1f} min wall, disk free {free_gib:.1f} GiB", flush=True)
                if free_gib < args.min_free_gib:
                    print(f"STOP: disk free {free_gib:.2f} GiB < {args.min_free_gib}; partial results kept, rerun to resume", flush=True)
                    for f in pending:
                        f.cancel()
                    return 3
    wall = time.time() - t0

    blas = [r["blas_threads"] for r in done.values() if "blas_threads" in r]
    pinned = all(lib["num_threads"] == 1 for rep in blas for lib in rep)
    print(f"BLAS/OpenMP pools per worker (threadpoolctl, #408): {blas[:1]} ... all pinned to 1: {pinned}")
    if blas and not pinned:
        print("WARNING: a worker reported a thread pool with num_threads != 1", flush=True)
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
        "decision_ref": prop.decision_ref,
        "parent_attrition": parent.attrition(),
        "epoch_model": prop.epoch_model,
        "epoch_model_section": (
            None if prop.epoch_model == "off"
            else getattr(cfg.active_dr(), "epoch_model").model_dump(mode="json")
        ),
        "nice": args.nice,
        "blas_threads_per_worker": blas,
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
