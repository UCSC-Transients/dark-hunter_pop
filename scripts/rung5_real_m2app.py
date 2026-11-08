#!/usr/bin/env python3
"""R5-T2 (#620): M2_app for real DR3 C1 rows with the mock's estimator (RUNG5_SPEC §3).

Input, one of:
* ``--stage-artifact``: ``CandidateRecord``s of a ``mass_derivation_bulk`` artifact (full NSS covariance in
  ``nss_solution``; MIST isochrone ``log_m1_mean`` / ``log_m1_sigma`` in ``extras``); ``--significance-h5`` supplies
  DR3 ``significance`` when the records lack it;
* ``--published-orbits`` (the #390 ``published_orbits.h5``: all 168,065 Orbital + AstroSpectroSB1 rows with
  ``corr_vec``, ``bit_index``, ``significance``) plus ``--m1-table`` (HDF5 datasets ``source_id``, ``log_m1_mean``,
  ``log_m1_sigma`` of the isochrone M1 posterior of the same hosts); the covariance is rebuilt with
  ``nss_covariance.reconstruct_nss_covariance`` (no diagonal fallback).
It applies ``rung5_m2app.m2_app_for_candidate`` with the
``mass_recovery`` block of the rung-4 config, and writes one HDF5 table: source_id, solution type, reason, m2_med,
m2_q16, m2_q84, m_f_med, mc_ess, student_t_nu, factorization. Rows outside the C1 domain or lacking inputs are kept
with their reason (never silently dropped). Heavy at full size (≈ 4,000 MC draws per C1 orbit): run it through
``scripts/jobs/run_heavy.sh``, one heavy job at a time; ``--limit`` gives a smoke run.

Usage::
    PYTHONPATH=src .venv/bin/python scripts/rung5_real_m2app.py --stage-artifact output/<run>/mass_derivation_bulk/<fp>.h5 \\
        --out output/rung5/real_m2app_dr3.h5 [--limit 100]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import rung5_m2app as rm
from darkhunter_pop.schemas import CandidateRecord

FIELDS = ("m2_med", "m2_q16", "m2_q84", "m_f_med", "mc_ess", "student_t_nu")


def iter_records(path: Path, limit: int | None):
    with h5py.File(path, "r") as h:
        ds = h["candidates"]["records_json"].asstr()
        n = ds.shape[0] if limit is None else min(limit, ds.shape[0])
        for start in range(0, n, 1024):
            for raw in ds[start:min(start + 1024, n)]:
                yield CandidateRecord.model_validate(json.loads(raw))


def _published_table(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        g = h["published"]
        out = {}
        for k in g:
            v = g[k][()]
            out[k] = v.astype(str) if v.dtype.kind in "SO" else v
    return out


def iter_published(path: Path, m1_table: Path | None, limit: int | None):
    from darkhunter_pop.nss_covariance import reconstruct_nss_covariance

    t = _published_table(path)
    m1: dict[int, tuple[float, float]] = {}
    if m1_table is not None:
        with h5py.File(m1_table, "r") as h:
            m1 = {int(s): (float(a), float(b)) for s, a, b in zip(h["source_id"][()], h["log_m1_mean"][()], h["log_m1_sigma"][()])}
    n = t["source_id"].size if limit is None else min(limit, t["source_id"].size)
    for i in range(n):
        row = {k: (v[i] if v.ndim == 1 else v[i].tolist()) for k, v in t.items()}
        sid = int(row["source_id"])
        res = reconstruct_nss_covariance(row, nss_solution_type=str(row["nss_solution_type"]))
        lm = m1.get(sid, (float("nan"), float("nan")))
        yield CandidateRecord(source_id=sid, nss_solution_type=str(row["nss_solution_type"]), nss_solution=res.parameter_set,
                              nss_orbital={"significance": float(row["significance"])},
                              extras={"log_m1_mean": lm[0], "log_m1_sigma": lm[1]})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stage-artifact", type=Path)
    src.add_argument("--published-orbits", type=Path)
    ap.add_argument("--m1-table", type=Path, default=None, help="with --published-orbits: isochrone log M1 moments")
    ap.add_argument("--significance-h5", type=Path, default=None, help="with --stage-artifact: published_orbits.h5 for DR3 significance")
    ap.add_argument("--rung4-config", type=Path, default=Path("config/population/rung4_co_v1.yaml"))
    ap.add_argument("--c1-period-max-days", type=float, default=None,
                    help="default: mass_recovery.gate_period_max_days of the rung-4 config")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    r4 = cob.load_rung4_config(args.rung4_config)
    mr = r4.mass_recovery
    pmax = args.c1_period_max_days if args.c1_period_max_days is not None else mr.gate_period_max_days
    if pmax is None:
        raise SystemExit("no C1 period ceiling (pass --c1-period-max-days or set gate_period_max_days)")
    rows: dict[str, list] = {k: [] for k in ("source_id", "nss_solution_type", "reason", "factorization", *FIELDS)}
    t0 = time.time()
    if args.published_orbits is not None:
        records = iter_published(args.published_orbits, args.m1_table, args.limit)
    else:
        records = iter_records(args.stage_artifact, args.limit)
        if args.significance_h5 is not None:
            pt = _published_table(args.significance_h5)
            sig = dict(zip(pt["source_id"].astype(np.int64).tolist(), pt["significance"].tolist()))
            records = (c.model_copy(update={"nss_orbital": {**c.nss_orbital, "significance": sig.get(int(c.source_id), float("nan"))}})
                       for c in records)
    for cand in records:
        summ, reason = rm.m2_app_for_candidate(cand, mr, c1_period_max_days=pmax)
        rows["source_id"].append(int(cand.source_id))
        rows["nss_solution_type"].append(str(cand.nss_solution_type))
        rows["reason"].append(reason)
        rows["factorization"].append("" if summ is None else summ["factorization"])
        for k in FIELDS:
            rows[k].append(float("nan") if summ is None else float(summ[k]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.rung4_config.read_bytes()).hexdigest()
    with h5py.File(args.out, "w") as h:
        g = h.create_group("m2app")
        g.create_dataset("source_id", data=np.asarray(rows["source_id"], np.int64))
        for k in ("nss_solution_type", "reason", "factorization"):
            g.create_dataset(k, data=np.asarray(rows[k], dtype=h5py.string_dtype("utf-8")))
        for k in FIELDS:
            g.create_dataset(k, data=np.asarray(rows[k], float))
        h.attrs.update({"issue": 620, "estimator": rm.ESTIMATOR, "input": str(args.published_orbits or args.stage_artifact), "m1_table": str(args.m1_table),
                        "rung4_config": str(args.rung4_config), "rung4_config_sha256": sha,
                        "c1_period_max_days": float(pmax), "data_release": r4.data_release if hasattr(r4, "data_release") else "dr3",
                        "mass_recovery_json": mr.model_dump_json()})
    reasons, counts = np.unique(np.asarray(rows["reason"]), return_counts=True)
    print(f"{len(rows['source_id'])} rows in {time.time() - t0:.1f} s; " + ", ".join(f"{r} {c}" for r, c in zip(reasons, counts)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
