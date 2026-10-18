#!/usr/bin/env python
"""#391 (Ryan 2026-10-18): injection truth set for the distant short-P real orbits and matched controls.

Writes an HDF5 with ``systems/source_id``, ``systems/group`` and ``systems/truth/<key>`` (the layout
``scripts/validate_epoch_model_v2_400.py inject --inj390`` reads), built with
``injection_test.published_truth_from_row`` from the #390 published-orbit table.
Groups: 0 = target (real C1 orbits at d > 1 kpc, log P < 2.6); 1 = control "same P, nearer"
(d 0.8–1.0 kpc, log P 2.3–2.6); 2 = control "same d, longer P" (d 1.0–1.2 kpc, log P 2.6–2.75).
Controls are drawn at random (seeded) and matched to the target's G distribution by G-bin quota.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TRUTH_KEYS = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
              "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True, help="dump_noW.npz from rung3_refit5_391.py --dump-from")
    ap.add_argument("--published", type=Path, required=True, help="output/gate390/published_orbits.h5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=391)
    args = ap.parse_args(argv)

    import h5py

    from darkhunter_pop import injection_test as it
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    z = np.load(args.dump, allow_pickle=True)
    rin = z["rob"] >= 0
    g, d, lp, sid = z["r_g"], z["r_d"], z["r_lp"], z["r_sid"]
    tgt = rin & (d > 1.0) & (lp < 2.6)
    c1 = rin & (d >= 0.8) & (d < 1.0) & (lp >= 2.3) & (lp < 2.6)
    c2 = rin & (d >= 1.0) & (d < 1.2) & (lp >= 2.6) & (lp < 2.75)
    rng = np.random.default_rng(args.seed)
    ge = np.array([5.0, 13.0, 15.0, 16.0, 17.0, 19.5])
    quota = np.histogram(g[tgt], bins=ge)[0]

    def matched(mask):
        pick = []
        for j, q in enumerate(quota):
            pool = np.flatnonzero(mask & (g >= ge[j]) & (g < ge[j + 1]))
            pick.extend(rng.choice(pool, size=min(q, pool.size), replace=False).tolist())
        return np.array(sorted(pick), int)

    groups = [(0, np.flatnonzero(tgt)), (1, matched(c1)), (2, matched(c2))]
    with h5py.File(args.published, "r") as h:
        pub = {k: h["published"][k][()] for k in h["published"].keys()}
    psid = pub["source_id"]
    pos = {int(s): i for i, s in enumerate(psid)}
    gm = import_gaiamock_mod()
    sids, grp, tr = [], [], {k: [] for k in TRUTH_KEYS}
    for gid, idx in groups:
        for i in idx:
            j = pos.get(int(sid[i]))
            if j is None:
                continue
            row = {k: (v[j].decode() if isinstance(v[j], bytes) else v[j]) for k, v in pub.items()}
            t = it.published_truth_from_row(row, gm)
            sids.append(int(sid[i]))
            grp.append(gid)
            for k in TRUTH_KEYS:
                tr[k].append(float(t.values[k]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h["systems/source_id"] = np.array(sids, np.int64)
        h["systems/group"] = np.array(grp, np.int64)
        for k, v in tr.items():
            h[f"systems/truth/{k}"] = np.array(v, float)
    print(f"wrote {args.out}: " + ", ".join(f"group {k}: {grp.count(k)}" for k in (0, 1, 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
