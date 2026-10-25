#!/usr/bin/env python3
"""Rung-4 compact-object selection-function table, mass recovery and summary (#552, spec §13.3).

Reads compact-object proposal artifacts (one per generation, all of one ``co_vN`` version; the §12.33 /
§13.2 guard refuses anything else), computes the deterministic-mixture weights of the training partition
(the SBC held-out draws are excluded) under the fragment's default θ_CO, and writes

* ``<out-dir>/selection_table.h5``: per component (``wd``, ``dark``) the weighted detection counts on the
  ``rung4_co_v1.yaml`` axes, with Kish ESS and MC variance per cell; the contamination model is registered as a
  ``run_validity`` stand-in (phase 1 does not compute it);
* ``<out-dir>/summary.json``: draws, acceptance per component, CPU s per draw, cells passing the ESS target,
  covariance self-checks, and the gate-4b mass-recovery rows per M2 bin.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/build_co_selection_table.py --parent-dir <snapshot> \\
        --artifact output/proposal_set/co_v1/co_gen1000.h5 --fragment config/population/proposal_set_co_v1_smoke.yaml \\
        --out-dir output/proposal_set/co_v1/sf_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set(parts[0])
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in sorted(keys)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True, help="one per artifact, same order")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.artifact) != len(args.fragment):
        raise SystemExit("one --fragment per --artifact")

    cfg = load_config(args.config)
    r4 = cob.load_rung4_config(args.rung4_config)
    frags = [ps.load_proposal_set_fragment(f) for f in args.fragment]
    gens = [f.proposal for f in frags]
    co = cob.check_co_generations(gens)
    targets = {json.dumps(f.target_co.model_dump(mode="json"), sort_keys=True) for f in frags}  # type: ignore[union-attr]
    if len(targets) != 1 or r4.version != co.version:
        raise SystemExit(f"fragments disagree on target_co, or rung-4 config version {r4.version} != {co.version}")
    target = frags[0].target_co
    assert target is not None
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, gens[0])

    truths, outs, versions = [], [], set()
    for a in args.artifact:
        t, o, attrs = ps.read_proposal_artifact(a)
        versions.add(str(attrs["primary_model"]))
        truths.append(t)
        outs.append(o)
    if versions != {ps.primary_model_version(gens[0])}:
        raise SystemExit(f"artifact primary models {sorted(versions)} != fragments' {ps.primary_model_version(gens[0])}")
    truth, outcome = _concat(truths), _concat(outs)
    if sorted(set(np.asarray(truth["generation"]).tolist())) != sorted(g.generation for g in gens):
        raise SystemExit("artifacts and fragments cover different generations")

    w = cob.co_importance_weights(truth, parent, gens, target, partition="train")
    hold = cob.holdout_mask(truth["draw_index"], co.sbc_holdout_fraction, co.sbc_holdout_salt)
    det = cob.detected_flags(outcome, r4.selection_table.detection)
    vals = cob.selection_axis_values(truth)
    comp = np.asarray(truth["co_component"])
    st = np.asarray(outcome["solution_type"]).astype(str)
    in_cut = st != "not_in_parent"
    cpu = np.asarray(outcome["cpu_seconds"], float)
    cov_cpu = np.asarray(outcome.get("orbit_cov_cpu_seconds", np.zeros(cpu.size)), float)

    tables = {}
    summary: dict[str, Any] = {
        "issue": 552, "version": co.version, "primary_model": ps.primary_model_version(gens[0]),
        "artifacts": [str(a) for a in args.artifact], "generations": [g.generation for g in gens],
        "n_draws": int(comp.size), "n_holdout": int(hold.sum()), "detection": r4.selection_table.detection,
        "cpu_seconds_per_draw_mean": float(cpu.mean()), "cpu_seconds_per_draw_median": float(np.median(cpu)),
        "cpu_seconds_per_cascade_draw_mean": float(cpu[in_cut].mean()) if in_cut.any() else None,
        "covariance_cpu_seconds_total": float(cov_cpu.sum()),
        "components": {},
        "stand_ins": [cob.contamination_stand_in(co.version).one_line()],
    }
    for name, code in cob.COMPONENT_CODES.items():
        sel = comp == code
        t = cob.build_selection_table(vals, w, det, r4.selection_table.axes, mask=sel & ~hold,
                                      meta={"component": name, "partition": "train", "target_co": target.model_dump(mode="json")})
        tables[name] = t
        orb = np.asarray(outcome["accepted_orbital"], bool)
        acc = np.asarray(outcome["published_acceleration"], bool)
        filled = t.n_draws > 0
        rec_sel = sel & ~hold & orb & ("orbit_cov" in outcome)
        if r4.mass_recovery.gate_period_max_days is not None:  # #557: gate 4b in the C1 domain (fitted P)
            rec_sel &= np.asarray(outcome["cascade"])[:, 10] <= r4.mass_recovery.gate_period_max_days
        recovered = cob.recover_masses(truth, outcome, parent, r4.mass_recovery, select=rec_sel) if rec_sel.any() else None
        chk = np.asarray(outcome.get("orbit_cov_sigma_check", np.full(sel.size, np.nan)), float)[sel & orb]
        summary["components"][name] = {
            "n_draws": int(sel.sum()), "n_system_in_cut": int((sel & in_cut).sum()),
            "n_accepted_orbit": int((sel & orb).sum()), "n_published_acceleration": int((sel & acc).sum()),
            "n_detected": int((sel & det).sum()),
            "acceptance_raw": float((sel & det).sum() / max(int(sel.sum()), 1)),
            "acceptance_raw_in_cut": float((sel & det).sum() / max(int((sel & in_cut).sum()), 1)),
            "p_det_weighted_train": float(np.sum(w[sel] * det[sel]) / np.sum(w[sel])) if np.sum(w[sel]) > 0 else None,
            "ess_train": ps.kish_ess(w[sel]), "ess_train_detected": ps.kish_ess(w[sel & det]),
            "cpu_seconds_per_draw_mean": float(cpu[sel].mean()) if sel.any() else None,
            "cells_filled": int(filled.sum()), "cells_total": int(filled.size),
            "cells_ess_ok": int(np.sum(t.ess >= r4.selection_table.min_cell_ess)),
            "cells_mc_noise_ok": int(np.sum(t.mc_poisson_ratio < cfg.physics.mc_noise_threshold)),
            "covariance_sigma_check_median": float(np.nanmedian(chk)) if np.isfinite(chk).any() else None,
            "covariance_sigma_check_max": float(np.nanmax(chk)) if np.isfinite(chk).any() else None,
            "mass_recovery": None if recovered is None else cob.recovery_summary(truth["m2_msun"], recovered, w, r4.mass_recovery, mask=rec_sel),
            "m2_posterior_prior": r4.mass_recovery.m2_posterior_prior,
            "gate_period_max_days": r4.mass_recovery.gate_period_max_days,
            "factorization_counts": None if recovered is None else {
                str(int(k)): int(v) for k, v in zip(*np.unique(recovered["factorization"][rec_sel], return_counts=True))},
        }
    for c in summary["components"].values():  # spec §13.4 gate 4b (#557)
        if c["mass_recovery"]:
            c["gate_4b"] = cob.gate_4b(c["mass_recovery"], r4.mass_recovery, min_ess=r4.selection_table.min_cell_ess)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = cob.write_selection_tables(args.out_dir / "selection_table.h5", tables, attrs={
        "version": co.version, "primary_model": summary["primary_model"], "detection": r4.selection_table.detection,
        "partition": "train", "artifacts": summary["artifacts"], "scale_to_full": parent.scale_to_full,
    })
    from darkhunter_pop.run_validity import write_stand_ins

    write_stand_ins(path, [cob.contamination_stand_in(co.version)])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "components"}, indent=1, default=str))
    for name, c in summary["components"].items():
        print(f"{name}: {c['n_draws']} draws, {c['n_system_in_cut']} in cut, orbit {c['n_accepted_orbit']}, accel "
              f"{c['n_published_acceleration']}, detected {c['n_detected']} ({c['acceptance_raw']:.3f}); weighted P_det "
              f"{c['p_det_weighted_train']}; cells ESS>={r4.selection_table.min_cell_ess:g}: {c['cells_ess_ok']}/{c['cells_total']}")
    print(f"wrote {path} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
