#!/usr/bin/env python3
"""R5-T7 (#624): gate 5a known-object injection (RUNG5_SPEC §5): plan → inject → report.

* ``plan`` (light): resolve the target groups of ``config/benchmarks/rung5_known_objects_dr3_st2.yaml`` against the #390
  ``published_orbits.h5`` (every DR3 Orbital + AstroSpectroSB1 row), build each injection truth with
  ``injection_test.published_truth_from_row`` (Campbell elements from the published Thiele–Innes; the published a0 is
  the photocentre orbit, so the injection reproduces the observed signal whatever the companion's light), and write
  ``plan.json``. Targets not found are listed, never dropped silently.
* ``inject`` (heavy; run through ``scripts/jobs/run_heavy.sh``): ``n_realizations`` per target through
  ``epoch_model.run_cascade`` with the configured (adopted) DR3 epoch/noise model, reusing the #572 re-injection worker;
  refuses a configured noise label other than ``required_noise_model``. Resumable JSONL.
* ``report`` (light): per object, the recovered fraction (accepted orbit with fitted P ≤ ``c1_period_max_days``) and the
  published-acceleration fraction, with binomial 68 % intervals; the gate-5a comparison with the rung-5 response is made
  at the fit (R5-T11), where the response exists.

Usage::
    PYTHONPATH=src .venv/bin/python scripts/rung5_known_objects_5a.py plan --out output/rung5/gate5a
    scripts/jobs/run_heavy.sh gate5a scripts/rung5_known_objects_5a.py inject --out output/rung5/gate5a --workers 4
    PYTHONPATH=src .venv/bin/python scripts/rung5_known_objects_5a.py report --out output/rung5/gate5a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

CONFIG = Path("config/benchmarks/rung5_known_objects_dr3_st2.yaml")


def load_targets(cfg: Mapping[str, Any], root: Path = Path(".")) -> list[dict[str, Any]]:
    """Targets of every group: ``{"group", "name", "source_id"}``, in config order, de-duplicated by source_id."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group, spec in cfg["groups"].items():
        src = yaml.safe_load((root / spec["source"]).read_text())
        rows = src.get("systems", src.get("data", []))
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in spec.get("select", {}).items()):
                sid = int(r["source_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                out.append({"group": group, "name": str(r.get("name", sid)), "source_id": sid})
    return out


def binomial_interval(k: int, n: int, z: float = 1.0) -> tuple[float, float, float]:
    """(fraction, lo, hi): Wilson interval at ``z`` sigma."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def summarize(records: Sequence[Mapping[str, Any]], c1_period_max_days: float,
              accel_published: Any = None) -> dict[int, dict[str, Any]]:
    """Per source_id: realizations, errors, the C1-recovered fraction (accepted 12-parameter orbit with fitted
    P ≤ ``c1_period_max_days``) and the published-acceleration fraction (``accel_published(rec) -> bool``; by default
    any 7/9-parameter outcome)."""
    acc_fn = accel_published or (lambda r: int(r.get("outcome", -1)) in (7, 9))
    by: dict[int, list[Mapping[str, Any]]] = {}
    for r in records:
        by.setdefault(int(r["source_id"]), []).append(r)
    out = {}
    for sid, rs in by.items():
        ok = [r for r in rs if "error" not in r]
        rec = sum(1 for r in ok if int(r.get("outcome", -1)) == 12 and r.get("accepted")
                  and float(r.get("period", np.inf)) <= c1_period_max_days)
        acc = sum(1 for r in ok if acc_fn(r))
        out[sid] = {"n": len(ok), "n_error": len(rs) - len(ok),
                    "recovered_c1": binomial_interval(rec, len(ok)), "acceleration": binomial_interval(acc, len(ok))}
    return out


def _published_rows(path: Path, sids: Sequence[int]) -> dict[int, dict[str, Any]]:
    import h5py

    with h5py.File(path, "r") as h:
        p = h["published"]
        all_sid = p["source_id"][()].astype(np.int64)
        pos = {int(s): i for i, s in enumerate(all_sid)}
        rows = {}
        for s in sids:
            i = pos.get(int(s))
            if i is None:
                continue
            row = {}
            for k in p:
                v = p[k][i]
                row[k] = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
            rows[int(s)] = row
    return rows


def target_colours(columns_h5: Path, sids: Sequence[int]) -> dict[int, float]:
    """#648: each target's DR3 BP−RP from the ``nss_orbit_input_columns`` snapshot (NaN where DR3 has none; the epoch
    model then applies its documented no-colour rule, EPOCH_MODEL_SPEC §8.13)."""
    import h5py

    with h5py.File(columns_h5, "r") as h:
        cs = h["source_id"][()].astype(np.int64)
        br = h["bp_rp"][()].astype(float)
    pos = {int(x): i for i, x in enumerate(cs)}
    return {int(s): (float(br[pos[int(s)]]) if int(s) in pos else float("nan")) for s in sids}


def cmd_plan(cfg: Mapping[str, Any], out: Path) -> int:
    from darkhunter_pop import injection_test as it
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    targets = load_targets(cfg)
    rows = _published_rows(Path(cfg["published_orbits"]), [t["source_id"] for t in targets])
    gm = import_gaiamock_mod()
    colours = target_colours(Path(cfg["colour_columns"]), [t["source_id"] for t in targets])
    plan, missing = [], []
    for t in targets:
        row = rows.get(t["source_id"])
        if row is None:
            missing.append({**t, "reason": "not in published_orbits (no DR3 Orbital/AstroSpectroSB1 solution)"})
            continue
        truth = it.published_truth_from_row(row, gm)
        plan.append({**t, "nss_solution_type": truth.nss_solution_type, "bp_rp": colours[t["source_id"]],
                     "truth": {k: float(v) for k, v in truth.values.items()}})
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"config": str(CONFIG), "targets": plan, "missing": missing,
                                               "not_injected": cfg.get("not_injected", [])}, indent=1))
    groups = {g: sum(1 for p in plan if p["group"] == g) for g in cfg["groups"]}
    n_nan = sum(1 for p in plan if not np.isfinite(p["bp_rp"]))
    print(f"plan: {len(plan)} targets {groups}; missing {len(missing)}; not injected {len(cfg.get('not_injected', []))}; "
          f"targets without a DR3 BP−RP {n_nan}")
    return 0


def cmd_inject(cfg: Mapping[str, Any], out: Path, workers: int, limit: int) -> int:
    import multiprocessing as mp
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reinject_noise_colour_572 as rn  # the #572 worker: run_cascade with the configured epoch model

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.epoch_model import noise_model_version

    pc = load_config(Path("config/config.yaml"))
    if pc.active_dr_mode.value != cfg["data_release"]:
        raise SystemExit(f"config active_dr_mode {pc.active_dr_mode.value} != {cfg['data_release']}")
    em = pc.dr3.epoch_model
    label = noise_model_version(em)  # as scripts/reinject_noise_colour_572.py
    if label != cfg["required_noise_model"]:
        raise SystemExit(f"configured noise model {label!r} != required {cfg['required_noise_model']!r}")
    plan = json.loads((out / "plan.json").read_text())["targets"]
    if any("bp_rp" not in t for t in plan):
        raise SystemExit(f"{out / 'plan.json'} predates #648 (no bp_rp per target): re-run the plan step into a new --out")
    log = out / "inject.jsonl"
    meta_p = out / "inject_meta.json"
    if log.exists() and (not meta_p.exists() or json.loads(meta_p.read_text()).get("colour_source") != "dr3_bp_rp"):
        raise SystemExit(f"{log} was written without the #648 colours: use a new --out")
    done = set()
    if log.exists():
        done = {(int(r["source_id"]), int(r["realization"])) for r in map(json.loads, log.read_text().splitlines())}
    tasks = []
    for t in plan:
        todo = [r for r in range(int(cfg["n_realizations"])) if (t["source_id"], r) not in done]
        if todo:
            v = {k: t["truth"][k] for k in rn.TRUTH_KEYS}
            tasks.append((v, t["source_id"], float(t["bp_rp"]), todo))
    tasks = tasks[:limit] if limit else tasks
    pop = pc.selection_function_astrometric.mock_population
    initargs = (int(pop.random_seed), pc.dr3.selection_function_astrometric.orbital_solution_cuts.model_dump_json(),
                em.model_dump_json(), float(pop.ruwe_min))
    (out / "inject_meta.json").write_text(json.dumps({"issue": [624, 648], "colour_source": "dr3_bp_rp", "noise_model_version": label,
                                                      "n_realizations": cfg["n_realizations"]}, indent=1))
    print(f"{len(tasks)} targets ({label}), workers={workers}", flush=True)
    t0 = time.time()
    with log.open("a") as fh, mp.get_context("spawn").Pool(workers, initializer=rn._init, initargs=initargs, maxtasksperchild=50) as pool:
        for recs in pool.imap_unordered(rn._inject, tasks, chunksize=1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


def cmd_report(cfg: Mapping[str, Any], out: Path) -> int:
    plan = json.loads((out / "plan.json").read_text())
    recs = [json.loads(x) for x in (out / "inject.jsonl").read_text().splitlines()] if (out / "inject.jsonl").exists() else []
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.forward_model import passes_acceleration_publication_cuts

    cuts = load_config(Path("config/config.yaml")).dr3.selection_function_astrometric.acceleration_publication_cuts

    def published(r: Mapping[str, Any]) -> bool:
        o = int(r.get("outcome", -1))
        return o in (7, 9) and passes_acceleration_publication_cuts(
            n_params=o, significance=float(r.get("acceleration_significance", np.nan)),
            goodness_of_fit_f2=float(r.get("goodness_of_fit", np.nan)), cuts=cuts)

    s = summarize(recs, float(cfg["c1_period_max_days"]), published)
    lines = ["# gate 5a known-object injection (R5-T7, #624)", "",
             "| group | name | source_id | type | n | recovered C1 (68 %) | acceleration (68 %) |", "|---|---|---|---|---:|---|---|"]
    for t in plan["targets"]:
        r = s.get(t["source_id"])
        if r is None:
            lines.append(f"| {t['group']} | {t['name']} | {t['source_id']} | {t['nss_solution_type']} | 0 | not run | not run |")
            continue
        f = r["recovered_c1"]
        a = r["acceleration"]
        lines.append(f"| {t['group']} | {t['name']} | {t['source_id']} | {t['nss_solution_type']} | {r['n']} | "
                     f"{f[0]:.2f} [{f[1]:.2f}, {f[2]:.2f}] | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] |")
    for m in plan["missing"] + plan["not_injected"]:
        lines.append(f"| {m.get('group', '-')} | {m.get('name', '')} | {m['source_id']} | - | 0 | {m['reason']} | |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("plan", "inject", "report"))
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    if a.step == "plan":
        return cmd_plan(cfg, a.out)
    if a.step == "inject":
        return cmd_inject(cfg, a.out, a.workers, a.limit)
    return cmd_report(cfg, a.out)


if __name__ == "__main__":
    sys.exit(main())
