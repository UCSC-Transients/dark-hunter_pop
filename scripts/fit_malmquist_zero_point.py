#!/usr/bin/env python3
"""Fit σ_int and the Janssens M_G zero point on the parent's RUWE < 1.4 dwarfs (MP-Q25, #391).

Decided 2026-10-03 (#391 issuecomment-5971280434). Combined19 extinction (MP-Q29). Prints the
fit, its bootstrap uncertainties, a robust cross-check and a per-M1 breakdown, and writes them
as JSON. The two fitted numbers are then copied into ``config/population/malmquist_decided.yaml``
(the closed loop showed a 0.05 mag zero-point error undoes the correction, so the precision is
reported).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.malmquist import MalmquistConfig
from darkhunter_pop.proposal_set import (
    ParentExtinctionConfig,
    ZeroPointFitConfig,
    combined19_a_g,
    fit_mg_zero_point,
    load_parent_snapshot,
    load_proposal_set_fragment,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_decided_tune.yaml"))
    ap.add_argument("--malmquist", type=Path, default=Path("config/population/malmquist_decided.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config()
    prop = load_proposal_set_fragment(args.fragment).proposal
    raw = yaml.safe_load(args.malmquist.read_text())
    mcfg = MalmquistConfig.model_validate(raw["malmquist"])
    ext = ParentExtinctionConfig.model_validate(raw["extinction"])
    fit = ZeroPointFitConfig.model_validate(raw["zero_point_fit"])
    parent = load_parent_snapshot(args.parent_dir, cfg, prop)
    a_g = combined19_a_g(parent, ext)
    res = fit_mg_zero_point(parent, a_g, mcfg, fit, ext)
    out = dataclasses.asdict(res) | {"parent_dir": str(args.parent_dir), "malmquist_config": str(args.malmquist),
                                     "decision_ref": "#391 issuecomment-5971280434 (MP-Q25, MP-Q29)"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
