#!/usr/bin/env python
"""Report every deviation of proposal fragments (and the merged config / parent) or rung-3 fit configs
from Ryan's decided settings (#391 spec §12.28; config/population/decided_settings.yaml).

Exit 0 if none deviate, 1 otherwise. Fragments are given with --fragment, fit configs with --fit-config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.decided_settings import DEFAULT_PATH, check_fit_config, check_launch, load_decided_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fragment", type=Path, nargs="*", default=[])
    ap.add_argument("--fit-config", type=Path, nargs="*", default=[])
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--parent-dir", type=Path, default=None, help="parent snapshot dir used by the launch")
    ap.add_argument("--settings", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)
    st = load_decided_settings(args.settings)
    cfg = load_config(args.config)
    n_bad = 0
    for f in args.fragment:
        dev = check_launch(f, config=cfg, parent_dir=args.parent_dir, settings=st)
        print(f"{f}: {len(dev)} deviation(s)")
        for d in dev:
            print(f"  {d}")
        n_bad += bool(dev)
    for f in args.fit_config:
        dev = check_fit_config(f, st)
        print(f"{f}: {len(dev)} deviation(s)")
        for d in dev:
            print(f"  {d}")
        n_bad += bool(dev)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
