"""CLI entry: the Wave 0 end-to-end dry run (issue #201).

Runs all fourteen stages with every substitution declared, measures wall-clock
and peak RSS per stage, and emits the ``dN/dM``-by-class product figure plus a
labeled report.

**Nothing this produces is a science result.** The run is tagged ``dry_run: true``
in its ``runs/<run_id>.yaml`` and every output carries the banner.

Examples::

    # New dry run on the laptop host profile.
    python scripts/run_dry_run.py --host-profile laptop

    # Resume / prove reproducibility: same run file, second invocation.
    python scripts/run_dry_run.py --host-profile laptop \\
        --run-file runs/20260915-201530-abc1234.yaml

    # Print the plan (with the stand-in block) and execute nothing.
    python scripts/run_dry_run.py --host-profile laptop --plan-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_pop.config_loader import KNOWN_HOST_PROFILES, load_config
from darkhunter_pop.dry_run import (
    DRY_RUN_LABEL,
    build_dry_run_manifest,
    latest_gaia_snapshot_meta,
    run_dry_run,
)
from darkhunter_pop.pipeline import build_stage_plan, validate_stage_runners
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    STAGE_ORDER,
    format_run_plan,
    load_run_manifest,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all fourteen stages as a labeled dry run: documented synthetic "
            "stand-ins, per-stage cost measurement, and the dN/dM-by-class "
            "product figure. Not a science result."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml + fragments)",
    )
    parser.add_argument(
        "--host-profile",
        choices=KNOWN_HOST_PROFILES,
        default=None,
        help=(
            "Checked-in host profile (config/host_profiles/<name>.yaml), issue "
            "#196. Explicit selection only. Only 'laptop' is exercised."
        ),
    )
    parser.add_argument(
        "--run-file",
        type=Path,
        default=None,
        help=(
            "Resume an existing dry-run manifest. Refused for a run that is not "
            "already tagged dry_run."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directory for run files (default: the repository's runs/)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help=(
            "Gaia snapshot meta.yaml for data_acquisition to replay. Default: "
            "the newest pristine local snapshot."
        ),
    )
    parser.add_argument(
        "--live-archive",
        action="store_true",
        help=(
            "Let data_acquisition query the Gaia archive instead of replaying a "
            "local snapshot. Needs network and is not reproducible."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        nargs="+",
        metavar="STAGE",
        default=[],
        help=(
            "Force re-run of named stage(s). Force-re-running a completed stage "
            "always creates a new run file (ARCHITECTURE.md §5); the child "
            "inherits the dry-run label and stand-in list."
        ),
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="RSS sampling period in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the run plan, including the declared stand-ins, and execute "
            "nothing. Writes no run file unless --run-file was given."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    errors = validate_stage_runners()
    if errors:
        for err in errors:
            print(f"run_dry_run: {err}", file=sys.stderr)
        return 2
    for name in args.force_rerun:
        if name not in STAGE_REGISTRY:
            print(f"run_dry_run: unknown stage {name!r}", file=sys.stderr)
            print("known stages: " + ", ".join(STAGE_ORDER), file=sys.stderr)
            return 2

    config = load_config(args.config, host_profile=args.host_profile)
    snapshot = args.snapshot
    if snapshot is None and not args.live_archive:
        snapshot = latest_gaia_snapshot_meta(config)
        if snapshot is None:
            print(
                "run_dry_run: no pristine Gaia snapshot found under the "
                "configured snapshots directory; stage a snapshot or pass "
                "--live-archive",
                file=sys.stderr,
            )
            return 1
    if snapshot is not None and not snapshot.is_file():
        print(f"run_dry_run: snapshot meta not found: {snapshot}", file=sys.stderr)
        return 1

    if args.plan_only:
        if args.run_file is not None:
            manifest = load_run_manifest(args.run_file)
            run_path = args.run_file
            created = False
        else:
            manifest, run_path = build_dry_run_manifest(
                config, snapshot_meta=snapshot, runs=args.runs_dir
            )
            created = True
        plan = build_stage_plan(
            manifest, config, force_rerun_stages=args.force_rerun
        )
        print(
            format_run_plan(
                manifest,
                config,
                plan,
                run_path=run_path,
                created_new=created,
                dry_run=True,
            ),
            flush=True,
        )
        return 0

    try:
        result = run_dry_run(
            config,
            run_file=args.run_file,
            runs=args.runs_dir,
            snapshot_meta=snapshot,
            use_local_snapshot=not args.live_archive,
            force_rerun_stages=args.force_rerun,
            monitor_interval_seconds=args.monitor_interval,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"run_dry_run: {exc}", file=sys.stderr)
        return 1

    print("")
    print(result.report_text, flush=True)
    print("")
    print(f"run file:  {result.run_path}")
    print(f"report:    {result.report_path}")
    print(f"figure:    {result.figure_path or '(not written)'}")
    print(f"caption:   {result.caption_path}")

    unfinished = result.unfinished_stages()
    unreasoned = result.skips_without_reason()
    if unfinished:
        print(
            "run_dry_run: GATE FAIL — stages not in a terminal state: "
            + ", ".join(unfinished),
            file=sys.stderr,
        )
    if unreasoned:
        print(
            "run_dry_run: GATE FAIL — skipped stages with no stated reason: "
            + ", ".join(unreasoned),
            file=sys.stderr,
        )
    if unfinished or unreasoned:
        return 1
    print(f"\nall {len(STAGE_ORDER)} stages terminal. {DRY_RUN_LABEL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
