"""Mark a pipeline stage completed when its HDF5 artifact exists but the run is stuck.

Use when science output was written but the process died before updating the run
manifest (e.g. GC hang after ``mass_derivation_bulk`` on a full catalog).

Example::

    python scripts/complete_stage_from_artifact.py \\
      --run-file runs/20260826-233413-a1a20a9.yaml \\
      --stage mass_derivation_bulk \\
      --artifact output/20260826-233413-a1a20a9/mass_derivation_bulk/dbcfab1b48af00a2.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    load_run_manifest,
    mark_stage_finished,
    save_run_manifest,
)
from darkhunter_pop.schemas import StageStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mark a stage completed on the run manifest when its HDF5 artifact "
            "already exists."
        )
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        required=True,
        help="Path to runs/<run_id>.yaml",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Stage name (e.g. mass_derivation_bulk)",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to the stage HDF5 artifact",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.yaml (default: config/config.yaml + fragments)",
    )
    args = parser.parse_args(argv)

    if args.stage not in STAGE_REGISTRY:
        print(f"complete_stage: unknown stage {args.stage!r}", file=sys.stderr)
        return 1
    if not args.run_file.is_file():
        print(f"complete_stage: run file not found: {args.run_file}", file=sys.stderr)
        return 1
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        print(f"complete_stage: artifact not found: {artifact}", file=sys.stderr)
        return 1

    with h5py.File(artifact, "r") as handle:
        stage_attr = handle["meta"].attrs.get("stage")
        n_candidates = int(handle["meta"].attrs.get("n_candidates", -1))

    spec = STAGE_REGISTRY[args.stage]
    if stage_attr != args.stage:
        print(
            f"complete_stage: artifact stage={stage_attr!r} != requested {args.stage!r}",
            file=sys.stderr,
        )
        return 1
    if n_candidates < 0:
        print("complete_stage: artifact missing meta n_candidates", file=sys.stderr)
        return 1

    config = load_config(args.config) if args.config is not None else load_config()
    manifest = load_run_manifest(args.run_file)
    record = manifest.stages.get(args.stage)
    if record is None:
        print(
            f"complete_stage: stage {args.stage!r} was never started on this run",
            file=sys.stderr,
        )
        return 1

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, args.run_file)
    print(
        f"complete_stage: {args.stage} -> completed "
        f"(n_candidates={n_candidates}, artifact={artifact})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
