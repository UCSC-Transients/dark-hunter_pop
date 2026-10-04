#!/usr/bin/env python3
"""Weights for an isochrone-parent proposal-set artifact (#418 smoke; spec §11.4, §10.4).

Reads an artifact written by ``scripts/run_proposal_pilot.py`` with a fragment whose
``proposal.m1`` is ``isochrone_mist_drop_unresolved``, reloads its parent, and computes the
MdS17 target with the evolved flux relation (#416) plus the 2-D CMD Malmquist weight
(``proposal_set.malmquist_cmd_log_weight``). Reports outcome counts, unit-weight counts,
weight ESS with and without W, and the evolved share. Writes ``--out`` (JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darkhunter_pop import malmquist_cmd as mc
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--cmd-config", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    truth, outcome, attrs = ps.read_proposal_artifact(args.artifact)
    prov = json.loads(attrs["provenance_json"])
    target = ps.MdS17TargetConfig.model_validate(prov["target_mds17_json"])
    prop = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    cmcfg = mc.load_cmd_malmquist_config(args.cmd_config)
    evo = ps.evolved_mg0_for_draws(truth, parent)
    log_lam = ps.mds17_luminous_log_intensity(truth, target, evolved_mg0_system=evo)
    log_lam_dwarf = ps.mds17_luminous_log_intensity(truth, target)
    lw, counts = ps.malmquist_cmd_log_weight(truth, parent, target, cmcfg, cfg)
    lq = [ps.log_q_total_for(truth, parent, prop)]
    n = [prop.n_draws]
    w0 = ps.importance_weights(log_lam, lq, n, scale_to_full=parent.scale_to_full)
    w1 = ps.importance_weights(log_lam + lw, lq, n, scale_to_full=parent.scale_to_full)
    wd = ps.importance_weights(log_lam_dwarf, lq, n, scale_to_full=parent.scale_to_full)
    acc = np.asarray(outcome["accepted_orbital"], bool)
    sol, sol_n = np.unique(np.asarray(outcome["solution_type"]).astype(str), return_counts=True)
    giant = np.asarray(truth["is_giant"], bool)
    finite_lw = np.isfinite(lw)
    out = {
        "artifact": str(args.artifact),
        "draws": int(acc.size),
        "solution_types": {str(a): int(b) for a, b in zip(sol, sol_n)},
        "accepted_orbital": int(acc.sum()),
        "evolved_draws": int(giant.sum()),
        "malmquist_unit_weight_by_reason": counts["by_row_reason"],
        "log_w_percentiles_1_16_50_84_99": np.percentile(lw[finite_lw], [1, 16, 50, 84, 99]).tolist(),
        "draws_with_w_zero": int(np.sum(~finite_lw)),
        "ess_all": {"no_w": ps.kish_ess(w0), "two_d_w": ps.kish_ess(w1), "dwarf_relation_no_w": ps.kish_ess(wd)},
        "ess_accepted": {"no_w": ps.kish_ess(w0[acc]), "two_d_w": ps.kish_ess(w1[acc])},
        "sum_w_accepted": {"no_w": float(w0[acc].sum()), "two_d_w": float(w1[acc].sum())},
        "evolved_share_accepted_weighted": {
            "no_w": float(w0[acc & giant].sum() / max(w0[acc].sum(), 1e-300)),
            "two_d_w": float(w1[acc & giant].sum() / max(w1[acc].sum(), 1e-300)),
        },
        "parent_m1_source": {str(a): int(b) for a, b in zip(*np.unique(parent.m1_source, return_counts=True))},
        "parent_attrition": parent.attrition(),
        "parent_evolved_usable": int(np.sum(parent.is_giant & parent.usable)),
        "cpu_seconds_per_draw_median": float(np.median(np.asarray(outcome["cpu_seconds"], float))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
