#!/usr/bin/env python3
"""#704 (option B, Ryan 2026-11-14): orbit covariances of a luminous generation by deterministic replay.

Reads a luminous generation artifact simulated (or #696-replayed) under the CURRENT noise model (st2) and writes the
covariance sidecar ``<artifact stem>.orbit_cov.h5`` (``luminous_covariance``); the artifact is never modified.

1. Rows: every accepted orbital draw (Orbital + AstroSpectroSB1) -> ``mock_selection_rows`` row (pipeline M1 is NaN:
   the pre-filter does not read it).
2. Pre-filter (``config/population/luminous_covariance_704.yaml``; refuses unless strictly looser than the frozen
   cuts): Andrews point M2 at the pass-1 M1, El-Badry 2026 Janssens M~2 (the evaluator's own enrichment, dust maps
   from the config), period, G, solution type. No covariance is read.
3. Replay: each passing draw is re-simulated by ``proposal_set.simulate_one`` from its stored truth row (same base
   seed, generation, draw_index -> same epochs, noise and routing) under ``--fragment`` and ``--config``, with
   ``orbit_covariance_step`` = the config's ``covariance_jacobian_step`` (the CO fragments' step): the covariance is
   ``compact_objects.orbit_covariance_from_epochs``, exactly the compact-object path.
4. Checks (refuses to write on any failure): the fragment equals the artifact's proposal config (``n_draws`` aside),
   the configured epoch model carries the artifact's noise label (``build_epoch_setup``), no compact-object proposal;
   per replayed draw the stored seeds, ``solution_type``, ``accepted_orbital``, ``nss_solution_type`` and cascade
   (rtol 1e-9, atol 1e-12; bit-identical count reported) are reproduced and an orbit covariance record exists.
5. Writes ``<out>`` via ``<out>.tmp`` + rename (never overwrites) and ``<out stem>.replay.json``. Resumable through
   ``<out>.partial.jsonl``. ``--smoke N`` replays the first N passing draws and writes ``<out stem>.smoke.h5``
   (not a product: its replayed set is incomplete, so ``load_generation_draws`` cannot read it as the sidecar).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import luminous_covariance as lc
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.mock_selection_rows import mock_draws_to_selection_rows
from darkhunter_pop.sample_selection import SampleSelectionRegistry
from run_proposal_pilot import _WORKER, _init_worker, build_epoch_setup  # scripts/ is on sys.path when run as a script

RTOL, ATOL = 1e-9, 1e-12
_STEP: dict[str, float] = {}


def _init(config_path: str, fragment_path: str, niceness: int, step: float) -> None:
    _init_worker(config_path, fragment_path, niceness)
    _STEP["step"] = float(step)


def _work(draw: dict[str, Any]) -> dict[str, Any]:
    return ps.simulate_one(draw, gaiamock=_WORKER["gaiamock"], c_funcs=_WORKER["c_funcs"], cfg=_WORKER["proposal"],
                           cuts=_WORKER["cuts"], epoch=_WORKER["epoch"], orbit_covariance_step=_STEP["step"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_replayed(rec: dict[str, Any], i: int, outcome: dict[str, Any]) -> list[str]:
    """Mismatches of one replayed record against the stored outcome row ``i`` (empty = reproduced)."""
    bad = []
    if int(rec["numpy_seed"]) != int(outcome["numpy_seed"][i]) or int(rec["c_rand_seed"]) != int(outcome["c_rand_seed"][i]):
        bad.append("seeds")
    if str(rec["solution_type"]) != str(outcome["solution_type"][i]):
        bad.append("solution_type")
    if bool(rec["accepted_orbital"]) != bool(outcome["accepted_orbital"][i]):
        bad.append("accepted_orbital")
    if str(rec["nss_solution_type"]) != str(outcome["nss_solution_type"][i]):
        bad.append("nss_solution_type")
    if not np.allclose(np.asarray(rec["cascade"], float), np.asarray(outcome["cascade"][i], float), rtol=RTOL, atol=ATOL, equal_nan=True):
        bad.append("cascade")
    if "orbit_cov" not in rec or np.asarray(rec["orbit_cov"]).size != lc.N_ORBIT**2:
        bad.append("no_orbit_cov")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True, help="luminous generation under the current noise model (read only)")
    ap.add_argument("--fragment", type=Path, required=True, help="the generation's fragment (current noise_model label)")
    ap.add_argument("--out", type=Path, default=None, help="sidecar (default <artifact stem>.orbit_cov.h5; must not exist)")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--cov-config", type=Path, default=Path(lc.DEFAULT_CONFIG_PATH))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="checks, pre-filter counts and CPU estimate only; nothing simulated or written")
    ap.add_argument("--smoke", type=int, default=0, help="replay only the first N passing draws; writes <out stem>.smoke.h5 (not a product)")
    a = ap.parse_args(argv)
    out = a.out if a.out is not None else lc.orbit_cov_sidecar_path(a.artifact)
    target = out.with_name(out.stem + ".smoke.h5") if a.smoke > 0 else out
    if target.exists():
        raise SystemExit(f"{target} exists; covariance sidecars are never overwritten")

    truth, outcome, attrs = ps.read_proposal_artifact(a.artifact)
    cfg = load_config(a.config)
    ccfg = lc.load_luminous_covariance_config(a.cov_config)
    frag = ps.load_proposal_set_fragment(a.fragment)
    old = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
    if old.model_dump(exclude={"n_draws"}) != frag.proposal.model_dump(exclude={"n_draws"}):
        raise SystemExit("fragment differs from the artifact's proposal config beyond n_draws")
    if old.compact_object is not None:
        raise SystemExit("compact-object generation: it stores its own covariance (#704 is for luminous generations)")
    if str(attrs.get("primary_model")) != ps.primary_model_version(old):
        raise SystemExit(f"artifact primary_model {attrs.get('primary_model')!r} != {ps.primary_model_version(old)!r}")
    build_epoch_setup(cfg, old)  # refuses unless the configured epoch model carries the artifact's noise label
    if "nss_solution_type" not in outcome:
        raise SystemExit("artifact has no outcome/nss_solution_type (not a current-routing artifact)")

    acc = np.asarray(outcome["accepted_orbital"], bool)
    di_all = np.asarray(truth["draw_index"], np.int64)
    rows = mock_draws_to_selection_rows(truth, outcome, pipeline_m1=(np.full(di_all.size, np.nan), np.full(di_all.size, np.nan)))
    registry = SampleSelectionRegistry(cfg, repo=repo_root())
    pf = lc.evaluate_prefilter(rows, ccfg, registry=registry)
    acc_di = pf.source_id
    if not np.array_equal(acc_di, di_all[acc]):
        raise SystemExit("row order drifted from the accepted draws")
    pass_di = acc_di[pf.pass_any]
    pos = {int(d): k for k, d in enumerate(di_all)}
    todo = [pos[int(d)] for d in pass_di]
    if a.smoke > 0:
        todo = todo[: a.smoke]
    cpu = np.asarray(outcome["cpu_seconds"], float)
    summary = {
        "artifact": str(a.artifact), "noise_model": old.noise_model, "n_draws": int(di_all.size), "n_accepted": int(acc.sum()),
        "n_pass_andrews": int(pf.pass_andrews.sum()), "n_pass_elbadry2026": int(pf.pass_elbadry2026.sum()),
        "n_pass_any": int(pf.pass_any.sum()), "n_replay": len(todo), "prefilter_version": ccfg.prefilter.version,
        "stored_cpu_hours_of_replayed": float(cpu[todo].sum() / 3600.0),
    }
    print(json.dumps(summary), flush=True)
    if a.dry_run:
        return 0

    partial = Path(str(target) + ".partial.jsonl")
    done: dict[int, dict[str, Any]] = {}
    if partial.is_file():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["draw_index"])] = r
    queue_idx = [i for i in todo if int(di_all[i]) not in done]
    keys = list(truth.keys())

    def payload(i: int) -> dict[str, Any]:
        return {k: (truth[k][i].decode() if isinstance(truth[k][i], bytes) else (truth[k][i].item() if hasattr(truth[k][i], "item") else truth[k][i]))
                for k in keys}

    t0 = time.time()
    target.parent.mkdir(parents=True, exist_ok=True)
    window = max(1, a.workers) * 8
    it = iter(queue_idx)
    k = 0
    if queue_idx:
        with partial.open("a") as sink, ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                                                            initargs=(str(a.config), str(a.fragment), int(a.nice), ccfg.covariance_jacobian_step)) as pool:
            pending = {pool.submit(_work, payload(i)) for i in itertools.islice(it, window)}
            while pending:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    r = f.result()
                    done[int(r["draw_index"])] = r
                    sink.write(json.dumps(r) + "\n")
                    k += 1
                sink.flush()
                pending |= {pool.submit(_work, payload(i)) for i in itertools.islice(it, len(fin))}
                if k % a.progress_every < len(fin) or not pending:
                    print(f"  {k}/{len(queue_idx)} done, {(time.time() - t0) / 60:.1f} min wall", flush=True)

    bad: list[dict[str, Any]] = []
    n_exact = 0
    for i in todo:
        r = done[int(di_all[i])]
        mism = check_replayed(r, i, outcome)
        n_exact += int(np.array_equal(np.asarray(r["cascade"], float), np.asarray(outcome["cascade"][i], float)))
        if mism:
            bad.append({"draw_index": int(di_all[i]), "mismatch": mism})
    recs = {int(di_all[i]): done[int(di_all[i])] for i in todo}
    cov_ok = [bool(np.all(np.isfinite(np.asarray(r["orbit_cov"], float)))) for r in recs.values()]
    sig = np.array([float(r["orbit_cov_sigma_check"]) for r in recs.values()]) if recs else np.zeros(0)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    rep = {**summary, "smoke": a.smoke > 0, "n_failed_checks": len(bad), "failed": bad[:50], "n_bit_identical_cascade": n_exact,
           "n_cov_finite": int(sum(cov_ok)), "n_cov_nan": int(len(cov_ok) - sum(cov_ok)),
           "sigma_check_quantiles_50_90_99_max": [float(np.nanquantile(sig, q)) for q in (0.5, 0.9, 0.99, 1.0)] if np.any(np.isfinite(sig)) else [],
           "cov_cpu_seconds_total": float(sum(float(r["orbit_cov_cpu_seconds"]) for r in recs.values())),
           "replay_cpu_hours": float(sum(float(r["cpu_seconds"]) for r in recs.values()) / 3600.0),
           "wall_minutes": (time.time() - t0) / 60, "code_commit": commit, "config": str(a.config.resolve()),
           "fragment": str(a.fragment.resolve()), "cov_config": str(a.cov_config.resolve()), "covariance_jacobian_step": ccfg.covariance_jacobian_step}
    rep_path = target.with_name(target.stem + ".replay.json")
    rep_path.write_text(json.dumps(rep, indent=1))
    if bad:
        print(f"FAILED: {len(bad)} replayed draws do not reproduce the stored outcome; no sidecar written ({rep_path})")
        return 1
    side_attrs = {
        "artifact": a.artifact.name, "artifact_sha256": _sha256(a.artifact), "primary_model": str(attrs["primary_model"]),
        "fragment_fingerprint": str(attrs["fragment_fingerprint"]), "n_draws": int(di_all.size), "noise_model": old.noise_model,
        "prefilter_version": ccfg.prefilter.version, "cov_config_json": lc.config_json(ccfg),
        "covariance_jacobian_step": ccfg.covariance_jacobian_step, "code_commit": commit, "smoke": a.smoke > 0,
        "script": "scripts/replay_luminous_covariance_704.py",
    }
    pass_mask = pf.pass_any.copy()
    if a.smoke > 0:  # the smoke sidecar records only the replayed subset as passing (it is not a product)
        keep = {int(di_all[i]) for i in todo}
        pass_mask = np.array([int(d) in keep for d in acc_di], dtype=bool)
    sc = lc.sidecar_from_records(acc_di, pass_mask, recs, side_attrs)
    lc.write_orbit_cov_sidecar(target, sc)
    print(json.dumps({kk: v for kk, v in rep.items() if kk != "failed"}), flush=True)
    print(f"wrote {target}" + (" (SMOKE: partial replay, not a product)" if a.smoke > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
