#!/usr/bin/env python3
"""#593 (Ryan 2026-10-30, small control): re-inject a subset of the #390 published orbits at neighbouring GOST grid points.

gaiamock_mod takes the scan pattern from the nearest of 3,072 healpix-16 positions (~3.7°), so the "same orbit" in the
#589/#590 same-orbit comparison also differs from DR3 in its scan pattern. Here each selected orbit is injected with
its position moved to the centres of the 3 healpix-16 pixels nearest to its own pixel (2 realizations each, under the
current config = colour_v2.1+tr1, identical to ``reinject_v21tr1``), so the spread of the mock means across grid points
can be measured. Realization ids 10·k + r (k = 1..3 neighbour, r = 0, 1) keep the seeds independent of the original run.

Sample: orbits accepted in both reinject_v21tr1 realizations; 100 at G < 13 and 100 at G ≥ 13 (seed 593). Uses the
workers of ``scripts/reinject_noise_colour_572.py``. Heavy: run through ``scripts/jobs/run_heavy.sh``. Resumable JSONL.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reinject_noise_colour_572 as rj  # noqa: E402

P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V21 = P.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1"


def neighbour_positions(ra, dec, k=3, nside=16):
    import healpy as hp

    th, ph = np.radians(90.0 - dec), np.radians(ra)
    pix = hp.ang2pix(nside, th, ph)
    nb = [int(x) for x in hp.get_all_neighbours(nside, th, ph) if x >= 0 and x != pix]
    v0 = hp.pix2vec(nside, pix)
    nb = sorted(set(nb), key=lambda q: -float(np.dot(v0, hp.pix2vec(nside, q))))[:k]
    out = []
    for q in nb:
        t, p = hp.pix2ang(nside, q)
        out.append((float(np.degrees(p)), float(90.0 - np.degrees(t)), q))
    return int(pix), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bright", type=int, default=100)
    ap.add_argument("--n-faint", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    import h5py

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.epoch_model import noise_model_version

    cfg = load_config()
    em = cfg.dr3.epoch_model
    ref = json.loads((V21 / "inject_meta.json").read_text())
    if json.loads(em.model_dump_json()) != ref["epoch_model"]:
        raise SystemExit("config epoch model differs from reinject_v21tr1; refusing")
    by: dict = {}
    for line in (V21 / "inject.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("accepted"):
            by.setdefault(int(r["source_id"]), set()).add(int(r["realization"]))
    both = {s for s, v in by.items() if {0, 1} <= v}
    with h5py.File(P / "output/gate390/injection_test_full.h5", "r") as f:
        sids = f["systems/source_id"][:]
        truth = {k: f["systems/truth"][k][:] for k in rj.TRUTH_KEYS}
    with h5py.File(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400/inj390.h5", "r") as f:
        cmap = dict(zip(f["source_id"][:].tolist(), f["bp_rp"][:].tolist()))
    idx = np.array([i for i, s in enumerate(sids) if int(s) in both])
    g = truth["g_mag"][idx]
    rng = np.random.default_rng(593)
    pick = np.concatenate([rng.choice(idx[g < 13], min(a.n_bright, (g < 13).sum()), replace=False),
                           rng.choice(idx[g >= 13], min(a.n_faint, (g >= 13).sum()), replace=False)])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inject_meta.json").write_text(json.dumps({"issue": [593, 589, 590], "noise_model_version": noise_model_version(em),
                                                       "epoch_model": ref["epoch_model"], "n_orbits": int(pick.size), "neighbours": 3,
                                                       "realization_ids": "10*k + r, k = 1..3, r = 0, 1"}, indent=1))
    log = out / "inject.jsonl"
    done = set()
    if log.exists():
        done = {(int(r["source_id"]), int(r["realization"])) for r in map(json.loads, log.read_text().splitlines())}
    tasks, posmap = [], {}
    for i in pick:
        sid = int(sids[i])
        v = {k: float(truth[k][i]) for k in rj.TRUTH_KEYS}
        pix0, nbs = neighbour_positions(v["ra"], v["dec"])
        for k, (ra, dec, q) in enumerate(nbs, start=1):
            reals = [10 * k + r for r in (0, 1) if (sid, 10 * k + r) not in done]
            posmap[f"{sid}:{k}"] = {"own_pixel": pix0, "pixel": q, "ra": ra, "dec": dec}
            if reals:
                tasks.append((dict(v, ra=ra, dec=dec), sid, float(cmap.get(sid, np.nan)), reals))
    (out / "positions.json").write_text(json.dumps(posmap))
    pop = cfg.selection_function_astrometric.mock_population
    initargs = (int(pop.random_seed), cfg.dr3.selection_function_astrometric.orbital_solution_cuts.model_dump_json(),
                em.model_dump_json(), float(pop.ruwe_min))
    print(f"{len(tasks)} (orbit, grid point) tasks, {sum(len(t[3]) for t in tasks)} cascades, workers={a.workers}", flush=True)
    t0 = time.time()
    with log.open("a") as fh, mp.get_context("spawn").Pool(a.workers, initializer=rj._init, initargs=initargs, maxtasksperchild=100) as pool:
        for n, recs in enumerate(pool.imap_unordered(rj._inject, tasks, chunksize=1)):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (n + 1) % 50 == 0:
                print(f"  {n + 1}/{len(tasks)} tasks, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
