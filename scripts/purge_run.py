"""CLI: purge a run file (and optionally its artifacts). ARCHITECTURE.md §5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_pop.run_management import purge_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Purge a dark-hunter_pop run YAML (optionally its HDF5 artifacts)."
    )
    parser.add_argument("run_file", type=Path, help="Path to runs/<run_id>.yaml")
    parser.add_argument(
        "--with-artifacts",
        action="store_true",
        help=(
            "Also delete HDF5 artifacts this run produced: only paths under "
            "<artifact_root>/<run_id>/ that no other run file references. "
            "Copied-forward / shared artifacts are kept and listed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exactly what would be deleted and kept; delete nothing",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory of run files scanned for shared artifacts (default: the run file's own directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow purging a completed run",
    )
    args = parser.parse_args(argv)
    try:
        purge_run(
            args.run_file,
            with_artifacts=args.with_artifacts,
            force=args.force,
            dry_run=args.dry_run,
            runs_dir=args.runs_dir,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"purge_run: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(f"purged {args.run_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
