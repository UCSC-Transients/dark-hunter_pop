#!/usr/bin/env python
"""#391: A-star / rotation detection-deficit test at G < 10. Real host counts vs mock-detected weight (parent rows,
best-fit weights) in bins of Teff (GSP-Phot, else ESP-HS) and vbroad, from the vbroad snapshot. Writes JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--parent-h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import h5py
    from astropy.table import Table

    t = Table.read(a.snapshot / "table.ecsv")
    f = lambda c: np.ma.filled(np.ma.asarray(t[c], float), np.nan)  # noqa: E731
    teff = np.where(np.isfinite(f("teff_gspphot")), f("teff_gspphot"), f("teff_esphs"))
    vb = {}
    for s, sid, te, v in zip(np.asarray(t["sample"]).astype(str), np.asarray(t["source_id"], np.int64), teff, f("vbroad")):
        vb[(s, int(sid))] = (te, v)
    z = np.load(a.dump, allow_pickle=True)
    with h5py.File(a.parent_h5, "r") as h:
        psid = h["source_id"][()]
    te_e, vb_e = [0, 6000, 7000, 8000, 20000], [0, 20, 50, 100, 500]
    out = {"teff_edges": te_e, "vbroad_edges": vb_e}
    for kind, sm, gm, rs, rg in (("orbit", z["ob"] >= 0, z["m_g"], z["r_sid"][z["rob"] >= 0], z["r_g"][z["rob"] >= 0]),
                                 ("accel", z["ab"] >= 0, z["m_g_a"], z["ra_sid"], z["ra_g"])):
        sel = sm & (gm < 10)
        mt = np.array([vb.get(("parent", int(psid[r])), (np.nan, np.nan)) for r in z["prow"][sel]])
        rr = np.array([vb.get(("real_" + kind, int(s)), (np.nan, np.nan)) for s in rs[rg < 10]])
        w = z["w"][sel]
        for nm, j, e in (("teff", 0, te_e), ("vbroad_teff_gt_6500", 1, vb_e)):
            mm = np.isfinite(mt[:, j]) & ((mt[:, 0] > 6500) if j else True)
            rm = np.isfinite(rr[:, j]) & ((rr[:, 0] > 6500) if j else True)
            hm = np.histogram(mt[mm, j], bins=e, weights=w[mm])[0]
            hr = np.histogram(rr[rm, j], bins=e)[0]
            out[f"{kind}_{nm}"] = {"mock": hm.tolist(), "real": hr.tolist(), "real_over_mock_norm":
                                   ((hr / hr.sum()) / np.where(hm > 0, hm / hm.sum(), np.nan)).tolist(),
                                   "mock_frac_with_value": float(w[mm].sum() / w.sum()), "real_frac_with_value": float(rm.mean())}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
