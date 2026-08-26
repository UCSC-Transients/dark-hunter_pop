"""Purge a run file (and optionally its artifacts).

CLI contract from ARCHITECTURE.md §5. Implementation is Foundation F4; this module is a stub.

Usage (intended)::

    python scripts/purge_run.py runs/<run_id>.yaml
    python scripts/purge_run.py runs/<run_id>.yaml --with-artifacts
    python scripts/purge_run.py runs/<run_id>.yaml --force

Default: delete the run YAML only. ``--with-artifacts`` also deletes recorded HDF5 paths.
Refuses to purge completed runs unless ``--force``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print(
        "purge_run: not implemented yet (Foundation F4). See ARCHITECTURE.md §5.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
