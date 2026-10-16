#!/usr/bin/env python3
"""Issue #509: rebuild the deblended truth of an assembled generation, deterministically, with no new draws.

Assemblies before 2026-10-15 stored ``sample_proposal`` truths without the MP-Q35/36 posterior
deblending the cascade ran on. This recomputes the full planned truth table exactly as
``run_proposal_pilot.py`` did (same seeds; MIST *uncorrected*, as those generations ran), deblends
it, takes the stored prefix, checks the raw draws match, and writes a NEW artifact with the
corrected truth and the stored outcomes (the old file is untouched).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    cfg = load_config()
    lc = cfg.isochrone_mass.lowmass_correction.model_copy(update={"enabled": False})  # these generations ran on uncorrected MIST
    cfg = cfg.model_copy(update={"isochrone_mass": cfg.isochrone_mass.model_copy(update={"lowmass_correction": lc})})
    t, o, a = ps.read_proposal_artifact(args.artifact)
    prov = json.loads(a["provenance_json"])
    prop_cut = ps.ProposalConfig.model_validate_json(a["proposal_config_json"])
    planned = int(prov.get("planned_n_draws", prop_cut.n_draws))
    prop = prop_cut.model_copy(update={"n_draws": planned})
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    n = int(t["draw_index"].size)
    off = int(t["draw_index"][0])
    full = ps.sample_proposal(parent, prop, draw_index_offset=off, config=cfg)
    for k in ("draw_index", "parent_row", "period_days", "eccentricity", "m1_msun", "m2_msun", "log10_flux_ratio"):
        if not np.allclose(np.asarray(full[k])[:n], np.asarray(t[k]), equal_nan=True):
            raise SystemExit(f"raw truth column {k} does not reproduce; stop")
    if prop.m1 == "isochrone_posterior_draw_deblended":
        full = ps.apply_posterior_deblending({k: np.asarray(v) for k, v in full.items()}, parent, cfg, prop)
    truth = {k: np.asarray(v)[:n] for k, v in full.items()}
    okeys = list(o.keys())
    outcomes = [{k: (o[k][i].tolist() if np.asarray(o[k]).ndim > 1 else np.asarray(o[k])[i].item()) for k in okeys} | {"draw_index": int(t["draw_index"][i])}
                for i in range(n)]
    frag = ps.ProposalSetFragment(proposal=prop_cut, target_mds17=ps.MdS17TargetConfig.model_validate(prov["target_mds17_json"]))
    prov2 = dict(prov, retruth_from=str(args.artifact), retruth_issue=509,
                 retruth_note="truth recomputed with MP-Q35/36 posterior deblending (uncorrected MIST), outcomes unchanged")
    ps.write_proposal_artifact(args.out, truth, outcomes, fragment=frag, parent=parent, provenance=prov2)
    r = truth["m1_msun"] / np.asarray(t["m1_msun"], float)
    print(f"wrote {args.out}: {n} draws; deblended/stored M1 median {np.nanmedian(r):.3f} (16-84 {np.nanpercentile(r, 16):.3f}-{np.nanpercentile(r, 84):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
