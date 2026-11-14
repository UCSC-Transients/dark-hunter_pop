#!/usr/bin/env python3
"""#690 (MOCK_POPULATION_SPEC §3.5): recover the routed DR3 solution type of an existing proposal-set generation.

Generations written before ``proposal_set`` stored ``outcome/nss_solution_type`` have no per-draw Orbital /
AstroSpectroSB1 label, although ``epoch_model.run_cascade`` computed it. The routing draw is the first child Generator
of ``epoch_model_rng(base_seed, stream, draw_index, tag=PER_CCD_NOISE_RNG_TAG)`` and P(ASB1 | x) reads a0, parallax, P,
e and inclination, which the type offsets do not change, so it is recomputed from the stored cascade without running
gaiamock (``joint_refit.routed_asb1``). Exact only when the per-source noise terms ss1 / sa1 (which spawn from the same
Generator first) are off; the artifact's ``primary_model`` label says (the script refuses otherwise).

Writes the sidecar ``<artifact stem>.nss_type.h5`` (labels per draw) and ``<artifact stem>.nss_type_check.json``.

Consistency check (``--verify-n N``, needs the gaiamock overlay): re-simulates N accepted orbital draws, half recovered
AstroSpectroSB1 and half Orbital (and half of each at G below the type break, where the type also rescales F2), with
``proposal_set.simulate_one`` (same seeds), and requires (a) the label returned by ``run_cascade`` equals the recovered
label and (b) the re-simulated cascade vector equals the stored one bit for bit (rtol 1e-9). The pre-routing sigma and F2
are not stored, so the type-dependent stored values cannot be checked on every draw without a replay; the replay is the
check, and the draws it selects are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import epoch_model as em
from darkhunter_pop import joint_refit as jr
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config


def recover_labels(truth: dict, outcome: dict, attrs: dict, st: object, base_seed: int) -> np.ndarray:
    """Per-draw label (``NSS_TYPE_*``) from the stored cascade; the empty label off the orbital branch."""
    routed = jr.routed_asb1(
        truth["phot_g_mean_mag"], truth["sys_bp_rp"], outcome["cascade"], truth["generation"], truth["draw_index"],
        base_seed, st,
    )
    lab = np.full(routed.size, ps.NSS_TYPE_NONE, dtype=object)
    lab[routed == 1.0] = ps.NSS_TYPE_ASB1
    lab[routed == 0.0] = ps.NSS_TYPE_ORBITAL
    return lab.astype(str)


def _pick(rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
    return np.sort(rng.choice(pool, size=min(n, pool.size), replace=False)) if pool.size else pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--verify-n", type=int, default=0, help="draws to re-simulate for the consistency check (0 = none)")
    ap.add_argument("--seed", type=int, default=690)
    a = ap.parse_args(argv)

    truth, outcome, attrs = ps.read_proposal_artifact(a.artifact)
    primary = str(attrs.get("primary_model", ""))
    if "+ss1" in primary or "+sa1" in primary:
        print(f"refusing: primary_model {primary!r} has a per-source noise term that spawns before the routing draw")
        return 2
    if "+st1" not in primary:
        print(f"refusing: primary_model {primary!r} has no st1 routing (no AstroSpectroSB1 type to recover)")
        return 2
    gcfg = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
    cfg = load_config()
    epoch = ps.build_epoch_setup(cfg, gcfg)
    st = epoch.config.solution_type_model
    if st is None or epoch.config.source_noise_scale is not None or epoch.config.source_sigma_a0_scale is not None:
        print("refusing: current epoch config has ss1 / sa1 on or no solution_type_model; recovery would not be exact")
        return 2

    labels = recover_labels(truth, outcome, attrs, st, gcfg.base_seed)
    acc = np.asarray(outcome["accepted_orbital"], bool)
    orbital_branch = labels != ps.NSS_TYPE_NONE
    cascade = np.asarray(outcome["cascade"], float)
    expect_branch = ~np.isin(cascade[:, 0], list(em.CASCADE_NON_ORBITAL_FLAGS)) & (cascade[:, em.CASCADE_SLOTS["period"]] > 0)
    assert np.array_equal(orbital_branch, expect_branch)
    n_acc = int(acc.sum())
    n_sb1 = int((acc & (labels == ps.NSS_TYPE_ASB1)).sum())
    report: dict = {
        "artifact": str(a.artifact), "primary_model": primary, "n_draws": int(labels.size), "n_accepted_orbital": n_acc,
        "n_accepted_asb1": n_sb1, "asb1_fraction_accepted": (n_sb1 / n_acc) if n_acc else None,
        "n_orbital_branch_asb1": int((labels == ps.NSS_TYPE_ASB1).sum()),
        "n_orbital_branch": int(orbital_branch.sum()),
    }

    if a.verify_n > 0:
        from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

        gm = import_gaiamock_mod()
        cf = gm.read_in_C_functions()
        cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
        rng = np.random.default_rng(a.seed)
        g = np.asarray(truth["phot_g_mean_mag"], float)
        bright = g < st.break_g
        h = a.verify_n // 4
        pick = np.unique(np.concatenate([
            _pick(rng, np.flatnonzero(acc & (labels == typ) & m), h)
            for typ in (ps.NSS_TYPE_ASB1, ps.NSS_TYPE_ORBITAL) for m in (bright, ~bright)
        ])).astype(int)
        rows, n_type, n_casc = [], 0, 0
        for j in pick:
            draw = {k: (v[j].decode() if isinstance(v[j], bytes) else v[j]) for k, v in truth.items()}
            r = ps.simulate_one(draw, gaiamock=gm, c_funcs=cf, cfg=gcfg, cuts=cuts, epoch=epoch)
            same_t = r["nss_solution_type"] == labels[j]
            same_c = bool(np.allclose(np.asarray(r["cascade"], float), cascade[j], rtol=1e-9, atol=1e-12, equal_nan=True))
            n_type += int(same_t)
            n_casc += int(same_c)
            rows.append({"draw_index": int(truth["draw_index"][j]), "g": float(g[j]), "recovered": str(labels[j]),
                         "run_cascade": r["nss_solution_type"], "type_agree": bool(same_t), "cascade_identical": same_c})
        report.update(verify_n=len(rows), verify_type_agree=n_type, verify_cascade_identical=n_casc,
                      verify_n_asb1=int(sum(r["recovered"] == ps.NSS_TYPE_ASB1 for r in rows)),
                      verify_n_bright=int(sum(r["g"] < st.break_g for r in rows)), verify_rows=rows)
        ok = n_type == len(rows) and n_casc == len(rows)
        report["consistent"] = bool(ok)
        print(f"verify: type agree {n_type}/{len(rows)}; stored cascade reproduced {n_casc}/{len(rows)}", flush=True)
        if not ok:  # do not write a sidecar for labels that failed the replay (fallback: re-simulate; needs Ryan)
            Path(str(a.artifact).replace(".h5", ".nss_type_check.json")).write_text(json.dumps(report, indent=1))
            print("INCONSISTENT: no sidecar written")
            return 1

    side = ps.write_nss_type_sidecar(a.artifact, truth["draw_index"], labels, provenance={
        "method": "joint_refit.routed_asb1 (spawn-0 child of the per-CCD noise Generator; P(ASB1|x) from stored a0, plx, P, e, i)",
        "script": "scripts/recover_nss_type_690.py", "primary_model": primary, "base_seed": int(gcfg.base_seed),
        "verified_by_replay": a.verify_n > 0,
    })
    Path(str(a.artifact).replace(".h5", ".nss_type_check.json")).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "verify_rows"}), flush=True)
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
