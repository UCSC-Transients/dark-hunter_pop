#!/usr/bin/env python3
"""R5-T1 (#614): freeze the rung-5 mass-bin edges and merge map from the compact-object mock (RUNG5_SPEC §2.3).

Reads only compact-object proposal artifacts of one ``co_vN`` version carrying the required noise label (never real
rows). Under the fragments' default θ_CO, training partition only:
* true-mass bins per class: adjacent bins merged until each holds ``min_cell_ess`` Kish ESS of detected draws;
* M2_app bins: the f = 0 Student-t posterior median of ``compact_objects.recover_masses`` (the gate-4b estimator,
  which R5-T2 applies to real DR3 rows) on accepted C1 orbits, WD + dark pooled, merged the same way.
Writes the ``derived:`` block (edges, ESS, artifact SHA256s, digest, ``real_data_read: false``) back into the config,
and a JSON report next to it under ``--out-dir``.

Usage::
    PYTHONPATH=src .venv/bin/python scripts/rung5_freeze_bins.py --parent-dir <snapshot> \\
        --artifact output/proposal_set/v21tr1_e2st1/co_gen1006.h5 output/proposal_set/v21tr1_e2st1/co_gen1007.h5 \\
        --fragment config/population/proposal_set_co_v1_wd1006.yaml config/population/proposal_set_co_v1_wd1007.yaml \\
        --out-dir output/rung5/bins_dr3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop import rung5_bins as r5
from darkhunter_pop.config_loader import load_config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set(parts[0])
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in sorted(keys)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, nargs="+", required=True)
    ap.add_argument("--fragment", type=Path, nargs="+", required=True, help="one per artifact, same order")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--bins-config", type=Path, default=Path("config/population/rung5_bins_dr3_st2.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write the derived block")
    args = ap.parse_args(argv)
    if len(args.artifact) != len(args.fragment):
        raise SystemExit("one --fragment per --artifact")

    bins = r5.load_rung5_bins(args.bins_config)
    if bins.derived is not None:
        raise SystemExit(f"{args.bins_config} is already frozen (digest {bins.derived.digest[:12]}); edit the proposed block deliberately and clear `derived` to re-freeze")
    cfg = load_config(args.config)
    r4 = cob.load_rung4_config(bins.rung4_config)
    frags = [ps.load_proposal_set_fragment(f) for f in args.fragment]
    gens = [f.proposal for f in frags]
    co = cob.check_co_generations(gens)
    if co.version != bins.co_version or r4.version != bins.co_version:
        raise SystemExit(f"version mismatch: generations {co.version}, rung-4 {r4.version}, bins {bins.co_version}")
    targets = {json.dumps(f.target_co.model_dump(mode="json"), sort_keys=True) for f in frags}  # type: ignore[union-attr]
    if len(targets) != 1:
        raise SystemExit("fragments disagree on target_co")
    target = frags[0].target_co
    assert target is not None
    model = ps.primary_model_version(gens[0])
    if f"noise_{bins.required_noise_model}" not in model:
        raise SystemExit(f"primary model {model!r} does not carry noise label {bins.required_noise_model!r}")

    parent = ps.load_parent_snapshot(args.parent_dir, cfg, gens[0])
    truths, outs = [], []
    for a in args.artifact:
        t, o, attrs = ps.read_proposal_artifact(a)
        if str(attrs["primary_model"]) != model:
            raise SystemExit(f"{a}: primary model {attrs['primary_model']} != {model}")
        truths.append(t)
        outs.append(o)
    truth, outcome = _concat(truths), _concat(outs)

    w = cob.co_importance_weights(truth, parent, gens, target, partition="train")
    hold = cob.holdout_mask(truth["draw_index"], co.sbc_holdout_fraction, co.sbc_holdout_salt)
    train = ~hold & (w > 0)
    det = cob.detected_flags(outcome, bins.detection)
    comp = np.asarray(truth["co_component"])
    m2 = np.asarray(truth["m2_msun"], float)

    report: dict = {"issue": 614, "primary_model": model, "generations": [g.generation for g in gens],
                    "artifacts": [str(a) for a in args.artifact], "min_cell_ess": bins.min_cell_ess,
                    "real_data_read": False, "classes": {}}
    derived_true, ess_true = {}, {}
    for name in r5.CLASSES:
        sel = train & det & (comp == cob.COMPONENT_CODES[name])
        proposed = bins.true_m2_edges_msun[name]
        merged, ess = r5.greedy_merge_edges(m2, w, proposed, bins.min_cell_ess, mask=sel)
        derived_true[name], ess_true[name] = merged, ess
        report["classes"][name] = {
            "proposed_edges": list(proposed),
            "proposed_ess_detected": r5.binned_ess(m2, w, proposed, sel).tolist(),
            "merged_edges": list(merged), "merged_ess_detected": list(ess),
            "short_bins": int(np.sum(np.asarray(ess) < bins.min_cell_ess)),
        }

    orb = np.asarray(outcome["accepted_orbital"], bool)
    p_fit = np.asarray(outcome["cascade"], float)[:, 10]
    rec_sel = train & orb & (p_fit <= bins.c1_period_max_days)
    if "orbit_cov" not in outcome:
        raise SystemExit("artifacts carry no orbit_cov; M2_app cannot be computed")
    recovered = cob.recover_masses(truth, outcome, parent, r4.mass_recovery, select=rec_sel)
    m2_app = np.asarray(recovered["m2_med"], float)
    ok_app = rec_sel & np.isfinite(m2_app)
    app_merged, app_ess = r5.greedy_merge_edges(m2_app, w, bins.m2_app_edges_msun, bins.min_cell_ess, mask=ok_app)
    report["m2_app"] = {
        "n_orbits": int(ok_app.sum()), "proposed_edges": list(bins.m2_app_edges_msun),
        "proposed_ess": r5.binned_ess(m2_app, w, bins.m2_app_edges_msun, ok_app).tolist(),
        "merged_edges": list(app_merged), "merged_ess": list(app_ess),
        "per_class_ess": {n: r5.binned_ess(m2_app, w, app_merged, ok_app & (comp == cob.COMPONENT_CODES[n])).tolist()
                          for n in r5.CLASSES},
        "student_t_nu_note": "nu from rung4_co_v1.yaml (calibrated on old-noise co_v1 gens); re-calibration is a week-2 item (RUNG5_SPEC §1.3)",
    }

    derived = r5.derived_block(
        bins, true_m2_edges_msun=derived_true, m2_app_edges_msun=app_merged, true_bin_ess_detected=ess_true,
        m2_app_bin_ess_detected=app_ess, artifacts_sha256={str(a): _sha256(a) for a in args.artifact},
        primary_model=model, partition="train", real_data_read=False,
        frozen_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    report["digest"] = derived.digest
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rung5_bins_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    if args.dry_run:
        print("dry run: derived block not written")
        return 0
    raw = yaml.safe_load(args.bins_config.read_text())
    raw["derived"] = json.loads(derived.model_dump_json())
    text = args.bins_config.read_text().split("\nderived:")[0]
    args.bins_config.write_text(text + "\n" + yaml.safe_dump({"derived": raw["derived"]}, sort_keys=False, width=120))
    r5.assert_frozen(r5.load_rung5_bins(args.bins_config))
    print(f"froze {args.bins_config} (digest {derived.digest[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
