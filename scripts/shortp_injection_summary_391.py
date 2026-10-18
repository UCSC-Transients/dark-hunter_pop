#!/usr/bin/env python
"""#391: summarise the distant short-P re-injection (target vs matched controls): recovery and the cascade
step that rejects each realization (five-parameter = RUWE gate, seven/nine-parameter = acceleration-first,
orbital solution failing a published cut). Writes ``shortp_injection.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import h5py

    from darkhunter_pop import injection_test as it
    from darkhunter_pop.injection_test import binomial_fraction_interval

    with h5py.File(args.dir / "systems.h5", "r") as h:
        grp = dict(zip(h["systems/source_id"][()].tolist(), h["systems/group"][()].tolist()))
        a0 = dict(zip(h["systems/source_id"][()].tolist(), h["systems/truth/a0_mas"][()].tolist()))
    names = {0: "target d>1 kpc, log P<2.6", 1: "control d 0.8-1.0, log P 2.3-2.6", 2: "control d 1.0-1.2, log P 2.6-2.75"}
    codes = {getattr(it, k): k for k in dir(it) if k.startswith("OUTCOME_")}
    recs = [json.loads(x) for x in (args.dir / "inject.jsonl").read_text().splitlines()]
    out = {"n_records": len(recs), "groups": {}}
    cutk = ["cut_parallax_over_error", "cut_a0_over_err", "cut_a0_over_err_sqrt_p", "cut_f2", "cut_sigma_e"]
    for g, nm in names.items():
        rr = [r for r in recs if grp.get(int(r["source_id"])) == g]
        acc = np.array([bool(r.get("accepted")) for r in rr])
        k, n = int(acc.sum()), len(rr)
        lo_hi = binomial_fraction_interval(k, n)
        oc = {}
        for r in rr:
            if r.get("accepted"):
                continue
            c = codes.get(r.get("outcome"), "error" if "error" in r else str(r.get("outcome")))
            if c == "OUTCOME_ORBITAL":
                failed = [ck for ck in cutk if r.get(ck) is False]
                c = "orbital, fails " + ("+".join(failed) if failed else "?")
            oc[c] = oc.get(c, 0) + 1
        sig = np.array([r.get("significance", np.nan) for r in rr], float)
        ruwe = np.array([r.get("ruwe", np.nan) for r in rr], float)
        out["groups"][nm] = {"n_realizations": n, "n_systems": len({int(r["source_id"]) for r in rr}), "accepted": k,
                             "recovery": lo_hi, "rejections": dict(sorted(oc.items(), key=lambda x: -x[1])),
                             "recovered_significance_p10_50_90": np.nanpercentile(sig, [10, 50, 90]).tolist(),
                             "recovered_ruwe_p10_50_90": np.nanpercentile(ruwe, [10, 50, 90]).tolist(),
                             "truth_a0_mas_median": float(np.median([a0[int(r["source_id"])] for r in rr]))}
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
