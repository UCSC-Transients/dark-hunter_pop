"""Resume ``data_acquisition`` from a saved Gaia snapshot (no re-query).

Use after the archive query succeeded but a later step failed (e.g. quality cut).

Example::

    python scripts/resume_data_acquisition_from_snapshot.py \\
      --run-file runs/20260826-233413-a1a20a9.yaml \\
      --snapshot data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.data_acquisition import run_data_acquisition
from darkhunter_pop.run_management import load_run_manifest, save_run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finish data_acquisition from an existing Gaia snapshot without "
            "re-querying the archive."
        )
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        required=True,
        help="Path to the incomplete runs/<run_id>.yaml",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to data/.../gaia_snapshots/<id>/meta.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.yaml (default: config/config.yaml + fragments)",
    )
    args = parser.parse_args(argv)

    if not args.run_file.is_file():
        print(f"resume: run file not found: {args.run_file}", file=sys.stderr)
        return 1
    if not args.snapshot.is_file():
        print(f"resume: snapshot meta not found: {args.snapshot}", file=sys.stderr)
        return 1

    config = load_config(args.config) if args.config is not None else load_config()
    manifest = load_run_manifest(args.run_file)
    print(
        f"resume: run_id={manifest.run_id} snapshot={args.snapshot} "
        "(loading ECSV; may take a few minutes)",
        flush=True,
    )
    finished = run_data_acquisition(
        manifest,
        config,
        run_path=args.run_file,
        force_rerun=True,
        snapshot_meta_path=args.snapshot,
    )
    save_run_manifest(finished, args.run_file)
    record = finished.stages["data_acquisition"]
    print(
        f"resume: data_acquisition {record.status.value} "
        f"artifact={record.artifact_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
