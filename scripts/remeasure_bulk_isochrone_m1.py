#!/usr/bin/env python3
"""Re-measure ``mass_derivation_bulk`` under TAG10 and under MIST_isochrone M1 (#418).

Spec §11.7 / ARCHITECTURE.md ``mass_derivation_bulk``. Streams the candidates of an existing
``data_acquisition`` artifact and runs :func:`mass_derivation.run_bulk_on_candidates` twice,
once per ``mass_calibration.method``, on

* the known-truth Gaia BH systems (``config/benchmarks/known_truth_gaia_bh.yaml``), always;
* a deterministic subsample ``source_id % --subsample == 0`` (the full-covariance σ_M2 MC is
  the cost; ``--subsample 1`` is the whole artifact).

It also reports the M1 of every candidate under both methods (cheap: no MC). Writes
``bulk_remeasure.json`` and ``bulk_remeasure.txt`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from darkhunter_pop import mass_derivation as md
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import MassCalibrationMethod
from darkhunter_pop.schemas import CandidateRecord


def _stream(path: Path, keep: Any, chunk: int = 2048) -> list[CandidateRecord]:
    out: list[CandidateRecord] = []
    with h5py.File(path, "r") as h:
        s = h["candidates"]["records_json"].asstr()
        for a in range(0, s.shape[0], chunk):
            for raw in s[a:a + chunk]:
                c = CandidateRecord.model_validate(json.loads(raw))
                if keep(c):
                    out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-acquisition", type=Path, required=True, help="data_acquisition stage HDF5")
    ap.add_argument("--subsample", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--host-profile", default="laptop")
    ap.add_argument("--known-truth", type=Path, default=Path("config/benchmarks/known_truth_gaia_bh.yaml"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_config(host_profile=args.host_profile)
    kt = yaml.safe_load(args.known_truth.read_text())
    bh = {int(s["source_id"]): s["name"] for s in kt["systems"]}
    k = max(1, int(args.subsample))

    t0 = time.time()
    sub = _stream(args.data_acquisition, lambda c: c.source_id % k == 0 or c.source_id in bh)
    t_read = time.time() - t0
    out: dict[str, Any] = {"data_acquisition": str(args.data_acquisition), "subsample": k, "n_candidates": len(sub),
                           "read_seconds": t_read}
    rep = [f"#418 bulk re-measurement: {args.data_acquisition.name}, subsample source_id % {k} == 0 (+ known-truth BH), "
           f"{len(sub)} candidates", ""]
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    api = import_gaiamock_mod()
    results: dict[str, Any] = {}
    for method in (MassCalibrationMethod.TAG10, MassCalibrationMethod.MIST_ISOCHRONE):
        cfg = base.model_copy(update={"mass_calibration": base.mass_calibration.model_copy(update={"method": method})})
        t1 = time.time()
        kept, diag = md.run_bulk_on_candidates(sub, cfg, gaiamock=api)
        dt = time.time() - t1
        f = diag.funnel.as_dict()
        m1 = {c.source_id: c.m1.marginal("M1") for c in kept if c.m1 is not None}
        m2 = {c.source_id: c.m2.marginal("M2") for c in kept if c.m2 is not None}
        res = {"funnel": f, "wall_seconds": dt, "m2_pre_cut_percentiles": np.percentile(diag.m2_pre_cut_msun, [5, 16, 50, 84, 95]).tolist()
               if diag.m2_pre_cut_msun.size else []}
        bh_rows = {}
        for sid, name in bh.items():
            # the known-truth systems may be cut; recompute their M1/M2 directly
            cands = [c for c in sub if c.source_id == sid]
            if not cands:
                bh_rows[name] = {"in_artifact": False}
                continue
            upd, reason, m2_pre, _ = md.process_bulk_candidate(cands[0], cfg, api)
            row: dict[str, Any] = {"in_artifact": True, "skip_reason": reason, "m2_pre_cut": m2_pre}
            if upd is not None:
                a, b = upd.m1.marginal("M1"), upd.m2.marginal("M2")
                row.update({"m1": a.value, "m1_sigma": a.sigma, "m2": b.value, "m2_sigma": b.sigma})
                if "m1_isochrone" in upd.extras:
                    row["isochrone"] = upd.extras["m1_isochrone"]
            elif cfg.mass_calibration.method is MassCalibrationMethod.MIST_ISOCHRONE:
                row["isochrone"] = md.isochrone_m1_batch([cands[0]], cfg)[0][2]
            bh_rows[name] = row
        res["known_truth"] = bh_rows
        res["n_kept"] = len(kept)
        res["m1_kept_percentiles"] = np.percentile([v.value for v in m1.values()], [5, 16, 50, 84, 95]).tolist() if m1 else []
        results[method.value] = res
        rep.append(f"=== {method.value} ({dt:.0f} s) ===")
        rep += [f"  {key}: {val}" for key, val in f.items()]
        rep.append(f"  M2 pre-cut percentiles 5/16/50/84/95: {[round(x, 3) for x in res['m2_pre_cut_percentiles']]}")
        for name, row in bh_rows.items():
            rep.append(f"  {name}: {json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items() if kk != 'isochrone'})}")
        rep.append("")
    out["results"] = results
    (args.out_dir / "bulk_remeasure.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out_dir / "bulk_remeasure.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
