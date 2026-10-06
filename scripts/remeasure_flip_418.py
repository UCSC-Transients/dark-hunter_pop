#!/usr/bin/env python3
"""End-to-end re-measurement for the MIST_isochrone flip (#418, #425; spec §11.9).

For each ``mass_calibration.method`` given, runs ``mass_derivation_bulk``
(:func:`mass_derivation.run_bulk_on_candidates`) on every candidate of an existing
``data_acquisition`` artifact, writes the bulk HDF5 to ``--work-dir``, then loads the
literature parent rows exactly as the ``sample_selection`` stage does
(:func:`sample_selection.load_selection_rows_from_manifest`, uncut snapshot plus the bulk
enrichment) and evaluates every enabled sample (:func:`sample_selection.run_sample_selection`).

Reports, per method: the bulk funnel, the known-truth BH M1/M2, and per sample and mode
``n_parent`` / ``n_surviving`` and a hash of the surviving source IDs. Reproduction-mode
results must be identical between methods (they own their M1). Writes ``flip_remeasure.json``
and ``flip_remeasure.txt`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import yaml

from darkhunter_pop import mass_derivation as md
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import MassCalibrationMethod
from darkhunter_pop.run_management import create_run_manifest
from darkhunter_pop.sample_selection import load_selection_rows_from_manifest, run_sample_selection
from darkhunter_pop.schemas import CandidateRecord, StageRecord, StageStatus


def _stream(path: Path, chunk: int = 2048):  # type: ignore[no-untyped-def]
    with h5py.File(path, "r") as h:
        s = h["candidates"]["records_json"].asstr()
        for a in range(0, s.shape[0], chunk):
            for raw in s[a:a + chunk]:
                yield CandidateRecord.model_validate(json.loads(raw))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-acquisition", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["TAG10", "MIST_isochrone"])
    ap.add_argument("--known-truth", type=Path, default=Path("config/benchmarks/known_truth_gaia_bh.yaml"))
    args = ap.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    api = import_gaiamock_mod()
    base = load_config()
    kt = yaml.safe_load(args.known_truth.read_text())
    bh = {int(s["source_id"]): s for s in kt["systems"]}
    out: dict[str, Any] = {"data_acquisition": str(args.data_acquisition), "methods": {}}
    rep = [f"#418/#425 flip re-measurement on {args.data_acquisition}", ""]
    for name in args.methods:
        method = MassCalibrationMethod(name)
        cfg = base.model_copy(update={"mass_calibration": base.mass_calibration.model_copy(update={"method": method})})
        cfg.paths.artifact_root = str(args.work_dir / "output")
        bulk_path = args.work_dir / f"bulk_{name}.h5"
        t0 = time.time()
        if bulk_path.exists():
            with h5py.File(bulk_path, "r") as h:
                funnel = {k: int(v) for k, v in h["diagnostics"].attrs.items() if not isinstance(v, str)} if "diagnostics" in h else {}
            kept = list(_stream(bulk_path))
        else:
            kept, diag = md.run_bulk_on_candidates(_stream(args.data_acquisition), cfg, gaiamock=api)
            funnel = diag.funnel.as_dict()
            md.write_stage_hdf5(bulk_path, kept, stage_name="mass_derivation_bulk", diagnostics=dict(funnel))
        t_bulk = time.time() - t0
        bh_rows = {}
        for c in kept:
            if c.source_id in bh:
                a, b = c.m1.marginal("M1"), c.m2.marginal("M2")
                bh_rows[bh[c.source_id]["name"]] = {
                    "m1": a.value, "m1_sigma": a.sigma, "m2": b.value, "m2_sigma": b.sigma,
                    "published_m1": bh[c.source_id].get("published_m1_msun"),
                    "published_m2": bh[c.source_id].get("published_m2_msun"),
                }
        manifest = create_run_manifest(cfg)
        stages = dict(manifest.stages)
        stages["data_acquisition"] = StageRecord(stage_name="data_acquisition", status=StageStatus.COMPLETED,
                                                 artifact_path=str(args.data_acquisition))
        stages["mass_derivation_bulk"] = StageRecord(stage_name="mass_derivation_bulk", status=StageStatus.COMPLETED,
                                                     artifact_path=str(bulk_path))
        manifest = manifest.model_copy(update={"stages": stages})
        t1 = time.time()
        rows = load_selection_rows_from_manifest(manifest)
        res = run_sample_selection(rows, cfg)
        samples = {}
        for key, r in res.results.items():
            ids = sorted(int(x) for x in r.surviving_source_ids)
            samples[key] = {
                "mode": r.mode.value, "mass_source": r.mass_source, "n_parent": r.n_parent,
                "n_surviving": r.n_surviving,
                "survivors_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12],
                "route_counts": dict(r.route_counts),
            }
        out["methods"][name] = {"bulk_funnel": funnel, "bulk_seconds": t_bulk, "known_truth": bh_rows,
                                "selection_seconds": time.time() - t1, "samples": samples, "n_rows": len(rows)}
        rep.append(f"=== {name}: bulk {t_bulk:.0f} s, kept {len(kept)}; selection rows {len(rows)} ===")
        rep += [f"  {k}: {v}" for k, v in funnel.items()]
        rep += [f"  {n}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})}" for n, r in bh_rows.items()]
        rep += [f"  {k}: mode={s['mode']} n_parent={s['n_parent']} n_surviving={s['n_surviving']} ids={s['survivors_sha']} routes={s['route_counts']}"
                for k, s in samples.items()]
        rep.append("")
    if len(args.methods) == 2:
        a, b = (out["methods"][m]["samples"] for m in args.methods)
        diff = {k: (a[k]["n_surviving"], b[k]["n_surviving"]) for k in a if k in b and
                (a[k]["survivors_sha"], a[k]["n_parent"]) != (b[k]["survivors_sha"], b[k]["n_parent"])}
        out["changed_samples"] = diff
        rep.append(f"samples whose parent or survivor set changed between {args.methods}: {diff or 'none'}")
    (args.out_dir / "flip_remeasure.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / "flip_remeasure.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
