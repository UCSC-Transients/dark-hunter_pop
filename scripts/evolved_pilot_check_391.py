#!/usr/bin/env python
"""#391 spec §12.28 pilot check: evolved-detected weight quality of a pilot generation vs a reference set.

Pilot weights: the pilot alone, w = scale λ / (n q_pilot). Reference: the deterministic-mixture
weights of an existing cache (e.g. gens 33–36). "Detected" = accepted orbit or published acceleration;
"evolved" = CMD-evolved parent row. Reports, for noW (λ only) and cmdW (λ × 2-D CMD Malmquist W):
Kish ESS, ESS per CPU-h (all CPU and evolved-draw CPU), weight-tail shares, Σw, and the share of evolved
weight on draws whose own-generation flux density is at the defensive floor (relation part < 1%).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _floor_mask(t: dict, gcfg, parent) -> np.ndarray:
    """Draws whose own flux proposal density is within 1% of the defensive-uniform floor."""
    from darkhunter_pop import proposal_set as ps

    fc = gcfg.flux
    _, wr = ps.flux_width_and_weight(np.asarray(t["parent_row"], np.int64), parent, fc)
    floor = np.log((1.0 - fc.dark_fraction) * (1.0 - wr) / (fc.log_f_max - fc.log_f_min))
    lqf = np.asarray(t["log_q_flux"], float)
    return np.isfinite(lqf) & (lqf <= floor + math.log(1.01)) & ~np.asarray(t["is_dark"], bool)


def _stats(w: np.ndarray, sel: np.ndarray, floor: np.ndarray, cpu_all: float, cpu_evo: float) -> dict:
    from darkhunter_pop.proposal_set import kish_ess

    ws = w[sel]
    tot = ws.sum()
    srt = np.sort(ws)[::-1]
    k1 = max(1, int(math.ceil(0.01 * srt.size)))
    ess = kish_ess(ws)
    return {
        "n": int(sel.sum()), "kish_ess": ess, "sum_w": float(tot),
        "ess_per_cpu_h_all": ess / (cpu_all / 3600), "ess_per_cpu_h_evolved": ess / (cpu_evo / 3600),
        "top1_share": float(srt[:1].sum() / tot) if tot > 0 else math.nan,
        "top5_share": float(srt[:5].sum() / tot) if tot > 0 else math.nan,
        "top1pct_share": float(srt[:k1].sum() / tot) if tot > 0 else math.nan,
        "flux_floor_weight_share": float(w[sel & floor].sum() / tot) if tot > 0 else math.nan,
        "flux_floor_draw_share": float((sel & floor).sum() / max(1, sel.sum())),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--ref-cache", type=Path, required=True)
    ap.add_argument("--ref-artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--cmd-malmquist", type=Path, default=Path("config/population/malmquist_cmd.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from darkhunter_pop import malmquist_cmd as mc
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    t, o, a = ps.read_proposal_artifact(args.pilot)
    g = ps.ProposalConfig.model_validate_json(a["proposal_config_json"])
    tg = ps.MdS17TargetConfig.model_validate(json.loads(a["provenance_json"])["target_mds17_json"])
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, g)
    n_planned = g.n_draws
    n = int(t["draw_index"].size)
    if ps.uses_dark_deblend(g):
        ps.ensure_m1_deblend_dark(t, parent, cfg, [g])
    evo_mg = ps.evolved_mg0_for_draws(t, parent)
    rel = ps.mist_relation_for_draws(t, parent, cfg)
    ll = ps.mds17_luminous_log_intensity(t, tg, evolved_mg0_system=evo_mg, relation_log10_f=rel)
    lw, _ = ps.malmquist_cmd_log_weight(t, parent, tg, mc.load_cmd_malmquist_config(args.cmd_malmquist), cfg)
    lq = ps.log_q_total_for(t, parent, g, cfg)
    lq_gen = sum(np.asarray(t[k], float) for k in t if k.startswith("log_q_") and k != "log_q_total")
    max_dlq = float(np.nanmax(np.abs(np.where(np.isfinite(lq_gen), lq - lq_gen, 0.0))))
    # Prefix of a planned generation (budget stop): the density is the planned q; n = drawn count.
    evo = np.asarray(t["is_giant"], bool)
    det = np.asarray(o["accepted_orbital"], bool) | np.asarray(o["published_acceleration"], bool)
    cpu = np.asarray(o["cpu_seconds"], float)
    floor_p = _floor_mask(t, g, parent)
    out = {"pilot": {"generation": g.generation, "n_draws": n, "n_planned": n_planned,
                     "cpu_h": float(cpu.sum() / 3600), "cpu_h_evolved": float(cpu[evo].sum() / 3600),
                     "max_abs_logq_recompute_minus_generation": max_dlq,
                     "n_evolved": int(evo.sum()), "n_evolved_detected": int((evo & det).sum())}}
    for key, lt in (("noW", ll), ("cmdW", ll + lw)):
        w = ps.importance_weights(lt, [lq], [n], scale_to_full=parent.scale_to_full)
        out["pilot"][key] = _stats(w, evo & det, floor_p, cpu.sum(), cpu[evo].sum())

    # reference set
    z = np.load(args.ref_cache, allow_pickle=True)
    gens = [int(x) for x in z["gens"]]
    di = np.asarray(z["draw_index"], np.int64)
    r_evo = np.zeros(di.size, bool)
    r_floor = np.zeros(di.size, bool)
    pos = {int(d): i for i, d in enumerate(di)}
    for p in args.ref_artifacts:
        tr, _, ar = ps.read_proposal_artifact(p)
        gr = ps.ProposalConfig.model_validate_json(ar["proposal_config_json"])
        idx = np.array([pos[int(d)] for d in tr["draw_index"]])
        r_evo[idx] = np.asarray(tr["is_giant"], bool)
        r_floor[idx] = _floor_mask(tr, gr, parent)
        del tr
    r_det = np.asarray(z["accepted"], bool) | np.asarray(z["published_acceleration"], bool)
    r_cpu = np.asarray(z["cpu_seconds"], float)
    lqs = np.asarray(z["lqs"], float)
    out["reference"] = {"generations": gens, "n_draws": int(di.size), "cpu_h": float(r_cpu.sum() / 3600),
                        "cpu_h_evolved": float(r_cpu[r_evo].sum() / 3600), "n_evolved": int(r_evo.sum()),
                        "n_evolved_detected": int((r_evo & r_det).sum())}
    for key, lt in (("noW", np.asarray(z["ll"], float)), ("cmdW", np.asarray(z["ll"], float) + np.asarray(z["lw"], float))):
        w = ps.importance_weights(lt, list(lqs), [int(x) for x in z["n_per_gen"]], scale_to_full=float(z["scale"]))
        out["reference"][key] = _stats(w, r_evo & r_det, r_floor, r_cpu.sum(), r_cpu[r_evo].sum())
    for key in ("noW", "cmdW"):
        p, r = out["pilot"][key], out["reference"][key]
        out[f"ratio_{key}"] = {
            "ess_per_cpu_h_all": p["ess_per_cpu_h_all"] / r["ess_per_cpu_h_all"],
            "ess_per_cpu_h_evolved": p["ess_per_cpu_h_evolved"] / r["ess_per_cpu_h_evolved"],
            "sum_w_pilot_over_ref": p["sum_w"] / r["sum_w"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
