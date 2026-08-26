"""CLI entry: print the run plan, then execute stages via ``darkhunter_pop.pipeline``.

ARCHITECTURE.md §5 / ORCHESTRATION_PLAN roster #16 / issue #79.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_pop.pipeline import STAGE_ORDER, run_pipeline, validate_stage_runners
from darkhunter_pop.run_management import STAGE_REGISTRY


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dark-hunter_pop pipeline: print the full run plan, then "
            "execute STAGE_ORDER via registered stage runners."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml + fragments)",
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        default=None,
        help="Path to an existing runs/<run_id>.yaml (required if incomplete runs exist)",
    )
    parser.add_argument(
        "--force-rerun",
        nargs="+",
        metavar="STAGE",
        default=[],
        help=(
            "Force re-run of named stage(s). Completed stages create a new run file "
            "with prior stage records copied (ARCHITECTURE.md §5)."
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        default=None,
        help="Optional subset of STAGE_ORDER to plan/execute (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan only; do not execute stages or write a new run file",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print STAGE_ORDER and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_stages:
        for name in STAGE_ORDER:
            print(name)
        return 0

    errors = validate_stage_runners()
    if errors:
        for err in errors:
            print(f"run_pipeline: {err}", file=sys.stderr)
        return 2

    for name in list(args.force_rerun) + list(args.stages or []):
        if name not in STAGE_REGISTRY:
            print(f"run_pipeline: unknown stage {name!r}", file=sys.stderr)
            print(
                "known stages: " + ", ".join(STAGE_ORDER),
                file=sys.stderr,
            )
            return 2

    try:
        run_pipeline(
            config_path=args.config,
            run_file=args.run_file,
            force_rerun_stages=args.force_rerun,
            stage_subset=args.stages,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"run_pipeline: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
