#!/usr/bin/env python3
"""#603 (spec §12.42): check the post-hoc routed solution type (joint_refit.routed_asb1) against run_cascade.

Re-simulates ``--n`` accepted orbital draws of one artifact with ``proposal_set.simulate_one`` (same seeds), records the
type ``epoch_model.apply_solution_type`` returns, and compares it with the reconstruction from stored quantities. Also
checks that the re-simulated cascade equals the stored one. Writes a JSON summary.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import epoch_model as em
from darkhunter_pop import joint_refit as jr
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=603)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t, o, attrs = ps.read_proposal_artifact(a.artifact)
    gcfg = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
    cfg = load_config()
    emc = dataclasses.replace(em.epoch_model_config_from_mapping(cfg.active_dr().epoch_model), enabled=True)
    ps.check_noise_model(gcfg, emc)
    st = emc.solution_type_model
    acc = np.flatnonzero(np.asarray(o["accepted_orbital"], bool))
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(acc, size=min(a.n, acc.size), replace=False))
    casc = np.asarray(o["cascade"], float)
    rec = jr.routed_asb1(t["phot_g_mean_mag"][pick], t["sys_bp_rp"][pick], casc[pick], t["generation"][pick], t["draw_index"][pick],
                         gcfg.base_seed, st)
    seen = []
    orig = em.apply_solution_type

    def spy(*args, **kw):
        c, typ = orig(*args, **kw)
        seen.append(typ)
        return c, typ

    em.apply_solution_type = spy
    gm = import_gaiamock_mod()
    cf = gm.read_in_C_functions()
    epoch = ps.EpochSetup(config=emc, gaps_jd=em.gap_intervals_jd(emc))
    cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
    rows, n_ok, n_casc = [], 0, 0
    for i, j in enumerate(pick):
        draw = {k: (v[j].decode() if isinstance(v[j], bytes) else v[j]) for k, v in t.items()}
        seen.clear()
        r = ps.simulate_one(draw, gaiamock=gm, c_funcs=cf, cfg=gcfg, cuts=cuts, epoch=epoch)
        typ = seen[-1] if seen else None
        sim = 1.0 if typ == "AstroSpectroSB1" else (0.0 if typ == "Orbital" else np.nan)
        same_c = bool(np.allclose(np.asarray(r["cascade"], float), casc[j], rtol=1e-9, atol=1e-12, equal_nan=True))
        n_ok += int(sim == rec[i])
        n_casc += int(same_c)
        rows.append({"draw_index": int(t["draw_index"][j]), "g": float(t["phot_g_mean_mag"][j]), "run_cascade": typ, "reconstructed": float(rec[i]), "cascade_identical": same_c})
        print(rows[-1], flush=True)
    res = {"artifact": str(a.artifact), "n": len(rows), "type_agree": n_ok, "cascade_identical": n_casc, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"type agreement {n_ok}/{len(rows)}; stored cascade reproduced {n_casc}/{len(rows)}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
