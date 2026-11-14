#!/usr/bin/env python3
"""#696 (EPOCH_MODEL_SPEC §8.16 st2, MOCK_POPULATION_SPEC §3.5): deterministic replay of a stored generation under a new
solution-type routing version, from the stored seeds, into a NEW artifact (the source is never modified).

Every draw is re-simulated with ``proposal_set.simulate_one`` from its stored truth row (same base seed, generation and
draw_index → same epochs, noise and routing uniform) under the current config, whose noise-model label must differ from the
source's only in the solution-type tag (e.g. ``…+e2+st1`` → ``…+e2+st2``).

Modes:

* ``targeted`` (default): re-simulates only the draws whose routed type changes. The routing draw ``u`` and every input
  of P(ASB1 | x) are untouched by the routing parameters, and the type is applied after the cascade fit, so a draw whose
  type does not change has a bit-identical outcome (the #690 replays reproduced stored cascades on 19–24 draws per
  generation at this code). Which draws change is computed exactly from the stored cascade (``calibrate_st2_routing_696``
  inverse), and ``--verify-unchanged`` further re-simulates that many unchanged orbital-branch draws (half at G < break)
  and requires them identical to the stored ones.
* ``full``: re-simulates every draw (≈ the original CPU cost).

Checks (refuses to write on any failure): the stored labels are reproduced under the source routing (from
``--source-iter2``); each replayed changed draw gets the predicted new type and the predicted cascade (rtol 1e-9); each
verified unchanged draw equals the stored outcome. Resumable through ``<out>.partial.jsonl``. Writes ``<out>`` (a copy of
the source with the replayed rows, ``outcome/nss_solution_type`` for every draw, and the new fragment's proposal config,
fingerprint and primary model) and ``<out stem>.replay.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import epoch_model as em
from darkhunter_pop import joint_refit as jr
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from run_proposal_pilot import _init_worker, _work, build_epoch_setup  # scripts/ is on sys.path when run as a script

OUTCOME_SCALARS = ("accepted_orbital", "published_acceleration", "f_m_msun", "cos_inclination_fit", "cpu_seconds")
COV_KEYS = ("orbit_p0", "orbit_cov", "orbit_cov_sigma_check", "orbit_cov_cpu_seconds")


def label_parts(label: str) -> tuple[str, str]:
    base, _, last = label.rpartition("+")
    return base, last


def stored_labels(path: Path, truth: dict[str, Any], outcome: dict[str, Any]) -> np.ndarray:
    if "nss_solution_type" in outcome:
        return np.char.decode(np.asarray(outcome["nss_solution_type"]).astype("S32"), "ascii")
    side = ps.read_nss_type_sidecar(path)
    if side is None or not np.array_equal(side[0], truth["draw_index"]):
        raise SystemExit(f"{path}: no stored solution type in draw order (run scripts/recover_nss_type_690.py)")
    return np.asarray(side[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True, help="source generation (read only)")
    ap.add_argument("--fragment", type=Path, required=True, help="the generation's fragment carrying the NEW noise_model label")
    ap.add_argument("--out", type=Path, required=True, help="new artifact (must not exist)")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--source-iter2", type=Path, default=Path("docs/gate_noise_colour/st1/iter2.json"),
                    help="routing intercept δ(G) the source was simulated with (st1 iteration 2)")
    ap.add_argument("--mode", choices=("targeted", "full"), default="targeted")
    ap.add_argument("--verify-unchanged", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--seed", type=int, default=696)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="checks and the replay list only; nothing simulated or written")
    ap.add_argument("--smoke", type=int, default=0,
                    help="replay only this many changed + this many unchanged draws, run the checks, write <out stem>.smoke.h5 (not a product)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; replays never overwrite")
    if a.out.resolve() == a.artifact.resolve():
        raise SystemExit("--out must differ from --artifact")

    truth, outcome, attrs = ps.read_proposal_artifact(a.artifact)
    cfg = load_config(a.config)
    frag = ps.load_proposal_set_fragment(a.fragment)
    old = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
    # n_draws is the stored count for an assembled (stopped) generation; the replay keeps the source's draws
    if old.model_dump(exclude={"noise_model", "n_draws"}) != frag.proposal.model_dump(exclude={"noise_model", "n_draws"}):
        raise SystemExit("fragment differs from the source's proposal config beyond noise_model / n_draws")
    new = old.model_copy(update={"noise_model": frag.proposal.noise_model})
    frag = frag.model_copy(update={"proposal": new})
    (ob_, ot), (nb_, nt) = label_parts(old.noise_model), label_parts(new.noise_model)
    if ob_ != nb_ or not (ot.startswith("st") and nt.startswith("st")) or ot == nt:
        raise SystemExit(f"noise_model {old.noise_model!r} -> {new.noise_model!r} is not a solution-type-only change")
    epoch = build_epoch_setup(cfg, new)  # refuses unless the configured epoch model carries the new label
    st_new = epoch.config.solution_type_model
    if st_new is None or epoch.config.source_noise_scale is not None or epoch.config.source_sigma_a0_scale is not None:
        raise SystemExit("needs solution_type_model on and ss1 / sa1 off (the routing uniform must be the first spawn)")
    it2 = json.loads(a.source_iter2.read_text())["config"]
    st_old = dataclasses.replace(st_new, version=ot, routing_delta_knots_g=tuple(it2["routing_delta_knots_g"]),
                                 routing_delta=tuple(it2["routing_delta"]))

    lab = stored_labels(a.artifact, truth, outcome)
    cas = np.asarray(outcome["cascade"], float)
    S = em.CASCADE_SLOTS
    ob = ~np.isin(cas[:, 0], list(em.CASCADE_NON_ORBITAL_FLAGS)) & (cas[:, S["period"]] > 0)
    if not np.array_equal(ob, lab != ps.NSS_TYPE_NONE):
        raise SystemExit("stored labels do not mark exactly the orbital-branch draws")
    g, bp = np.asarray(truth["phot_g_mean_mag"], float), np.asarray(truth["sys_bp_rp"], float)
    i_ob = np.flatnonzero(ob)
    c = cas[i_ob]
    u = jr.routing_uniforms(truth["generation"][i_ob], truth["draw_index"][i_ob], old.base_seed)
    k1 = em.predicted_k1_kms_array(c[:, S["a0"]], c[:, S["plx"]], c[:, S["period"]], c[:, S["ecc"]], c[:, S["inc_deg"]])
    a_old = lab[i_ob] == ps.NSS_TYPE_ASB1
    if not np.array_equal(u < st_old.p_asb1_array(g[i_ob], bp[i_ob], c[:, S["period"]], c[:, S["ecc"]], k1), a_old):
        raise SystemExit("the source routing (--source-iter2) does not reproduce the stored labels")
    a_new = u < st_new.p_asb1_array(g[i_ob], bp[i_ob], c[:, S["period"]], c[:, S["ecc"]], k1)
    c_pre = em.solution_type_properties_array(c, g[i_ob], a_old, st_old, inverse=True)
    c_pred = em.solution_type_properties_array(c_pre, g[i_ob], a_new, st_new)
    new_lab = lab.copy()
    new_lab[i_ob] = np.where(a_new, ps.NSS_TYPE_ASB1, ps.NSS_TYPE_ORBITAL)
    changed = i_ob[a_new != a_old]
    pred = {int(truth["draw_index"][i]): c_pred[j] for j, i in enumerate(i_ob) if a_new[j] != a_old[j]}
    rng = np.random.default_rng(a.seed)
    unchanged = i_ob[a_new == a_old]
    br = g[unchanged] < st_new.break_g
    h = a.verify_unchanged // 2
    verify = np.concatenate([rng.choice(unchanged[m], size=min(h, int(m.sum())), replace=False) for m in (br, ~br) if m.any()]).astype(int) \
        if a.verify_unchanged > 0 else np.array([], int)
    if a.smoke > 0:
        changed_run, verify = changed[: a.smoke], verify[: a.smoke]
        pred = {d: v for d, v in pred.items() if d in set(int(x) for x in truth["draw_index"][changed_run])}
        todo = np.unique(np.concatenate([changed_run, verify]))
    else:
        todo = np.arange(cas.shape[0]) if a.mode == "full" else np.unique(np.concatenate([changed, verify]))
    n_flip_bright = int(np.sum(g[changed] < st_new.break_g))
    print(f"{a.artifact.name}: {cas.shape[0]} draws, {i_ob.size} orbital branch, {changed.size} change type "
          f"({n_flip_bright} at G < {st_new.break_g}); replaying {todo.size} ({a.mode}, {verify.size} unchanged verification draws)", flush=True)
    est = float(np.sum(np.asarray(outcome["cpu_seconds"], float)[todo])) / 3600
    print(f"stored CPU of the replayed draws: {est:.2f} CPU-h", flush=True)
    if a.dry_run:
        return 0

    partial = Path(str(a.out) + (".smoke" if a.smoke > 0 else "") + ".partial.jsonl")
    done: dict[int, dict[str, Any]] = {}
    if partial.is_file():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["draw_index"])] = r
    di_all = np.asarray(truth["draw_index"], np.int64)
    queue_idx = [int(i) for i in todo if int(di_all[i]) not in done]
    keys = list(truth.keys())

    def payload(i: int) -> dict[str, Any]:
        return {k: (truth[k][i].decode() if isinstance(truth[k][i], bytes) else (truth[k][i].item() if hasattr(truth[k][i], "item") else truth[k][i]))
                for k in keys}

    t0 = time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    window = max(1, a.workers) * 8
    it = iter(queue_idx)
    k = 0
    with partial.open("a") as sink, ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                                                        initargs=(str(a.config), str(a.fragment), int(a.nice))) as pool:
        pending = {pool.submit(_work, payload(i)) for i in itertools.islice(it, window)}
        while pending:
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                r = f.result()
                r.pop("blas_threads", None)
                r.pop("worker_niceness", None)
                done[int(r["draw_index"])] = r
                sink.write(json.dumps(r) + "\n")
                k += 1
            sink.flush()
            pending |= {pool.submit(_work, payload(i)) for i in itertools.islice(it, len(fin))}
            if k % a.progress_every < len(fin) or not pending:
                print(f"  {k}/{len(queue_idx)} done, {(time.time() - t0) / 60:.1f} min wall", flush=True)

    # checks
    pos = {int(d): i for i, d in enumerate(di_all)}
    bad: list[dict[str, Any]] = []
    n_exact = 0
    for i in todo:
        d = int(di_all[i])
        r = done[d]
        rc = np.asarray(r["cascade"], float)
        if d in pred:
            ok = r["nss_solution_type"] == new_lab[i] and bool(np.allclose(rc, pred[d], rtol=1e-9, atol=1e-12, equal_nan=True))
        else:
            ok = r["nss_solution_type"] == new_lab[i] and bool(np.allclose(rc, cas[i], rtol=1e-9, atol=1e-12, equal_nan=True)) \
                and bool(r["accepted_orbital"]) == bool(outcome["accepted_orbital"][i])
            n_exact += int(np.array_equal(rc, cas[i]))
        if int(r["numpy_seed"]) != int(outcome["numpy_seed"][i]) or int(r["c_rand_seed"]) != int(outcome["c_rand_seed"][i]):
            ok = False
        if not ok:
            bad.append({"draw_index": d, "g": float(g[i]), "changed": d in pred, "type": r["nss_solution_type"], "expected": str(new_lab[i])})
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    acc_old = np.asarray(outcome["accepted_orbital"], bool)
    acc_new = acc_old.copy()
    for d, r in done.items():
        acc_new[pos[d]] = bool(r["accepted_orbital"])
    rep: dict[str, Any] = {
        "source": str(a.artifact), "out": str(a.out), "mode": a.mode, "source_noise_model": old.noise_model, "noise_model": new.noise_model,
        "n_draws": int(cas.shape[0]), "n_orbital_branch": int(i_ob.size), "n_type_changed": int(changed.size), "n_type_changed_bright": n_flip_bright,
        "n_replayed": int(todo.size), "n_verify_unchanged": int(verify.size), "n_verify_unchanged_bit_identical": n_exact,
        "n_failed_checks": len(bad), "failed": bad[:50], "n_accepted_source": int(acc_old.sum()), "n_accepted_replay": int(acc_new.sum()),
        "asb1_share_accepted_source": float(np.mean(lab[acc_old] == ps.NSS_TYPE_ASB1)),
        "asb1_share_accepted_replay": float(np.mean(new_lab[acc_new] == ps.NSS_TYPE_ASB1)),
        "wall_minutes": (time.time() - t0) / 60, "code_commit": commit, "config": str(a.config.resolve()), "fragment": str(a.fragment.resolve()),
    }
    rep_path = a.out.with_name(a.out.stem + (".smoke" if a.smoke > 0 else "") + ".replay.json")
    rep_path.write_text(json.dumps(rep, indent=1))
    if bad:
        print(f"FAILED: {len(bad)} replayed draws differ from the prediction / stored outcome; no artifact written ({rep_path})")
        return 1
    out = a.out.with_name(a.out.stem + ".smoke.h5") if a.smoke > 0 else a.out  # smoke: exercises the writer, never a product
    if out.exists():
        raise SystemExit(f"{out} exists")

    tmp = out.with_name(out.name + ".tmp")
    shutil.copyfile(a.artifact, tmp)
    import h5py

    with h5py.File(tmp, "r+") as hf:
        o = hf["outcome"]
        for d, r in done.items():
            i = pos[d]
            o["cascade"][i] = np.asarray(r["cascade"], float)
            o["solution_type"][i] = str(r["solution_type"]).encode()
            for key in OUTCOME_SCALARS:
                o[key][i] = r[key]
            if "orbit_cov" in o:
                for key in COV_KEYS:
                    if key in r:
                        v = np.asarray(r[key], float)
                        o[key][i] = v.reshape(o[key].shape[1:]) if o[key].ndim > 1 else v
        if "nss_solution_type" in o:
            del o["nss_solution_type"]
        o.create_dataset("nss_solution_type", data=new_lab.astype("S32"))
        hf.attrs["proposal_config_json"] = json.dumps(new.model_dump(mode="json"), sort_keys=True)
        hf.attrs["fragment_fingerprint"] = ps.fragment_fingerprint(frag)
        hf.attrs["primary_model"] = ps.primary_model_version(new)
        prov = json.loads(str(hf.attrs.get("provenance_json", "{}")))
        prov["replay_696"] = {k: rep[k] for k in ("source", "mode", "source_noise_model", "noise_model", "n_type_changed", "n_replayed",
                                                  "n_verify_unchanged", "code_commit")}
        prov["replay_696"]["script"] = "scripts/replay_routing_696.py"
        prov["replay_696"]["smoke"] = a.smoke > 0
        hf.attrs["provenance_json"] = json.dumps(prov, sort_keys=True, default=str)
    tmp.rename(out)
    print(json.dumps({k: v for k, v in rep.items() if k != "failed"}), flush=True)
    print(f"wrote {out}" + (" (SMOKE: partial replay, not a product)" if a.smoke > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
