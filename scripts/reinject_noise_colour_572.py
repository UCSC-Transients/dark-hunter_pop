#!/usr/bin/env python3
"""#454 / #572: re-inject the 1,296 #390 published orbits through the adopted epoch model with a given per-CCD noise
model (production ``n2d_s1232`` or the §8.12 colour model), for the F2 calibration e(G) and its validation.

Same systems, seeds and parsing as ``scripts/validate_epoch_model_v2_400.py inject`` (``epoch_model.run_cascade``
around ``predict_astrometry_binary_in_terms_of_a0``); the source context carries the system's observed BP-RP (from
the ``20261003T063811Z_epoch_counts_400`` inj390 table) for the colour term. Heavy: run through
``scripts/jobs/run_heavy.sh``. Resumable JSONL; refuses to mix noise-model versions in one output.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_W: dict[str, Any] = {}
TRUTH_KEYS = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
              "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag")


def _init(base_seed: int, cuts_json: str, em_json: str, ruwe_min: float) -> None:
    from threadpoolctl import threadpool_limits

    _W["tp"] = threadpool_limits(limits=1)
    from darkhunter_pop.config_schema import EpochModelPathConfig, OrbitalSolutionCutsConfig
    from darkhunter_pop.epoch_model import epoch_model_config_from_mapping, gap_intervals_jd
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    em = dataclasses.replace(epoch_model_config_from_mapping(EpochModelPathConfig.model_validate_json(em_json)), enabled=True)
    _W.update(gm=gm, c_funcs=gm.read_in_C_functions(), base_seed=base_seed, em=em, gaps=gap_intervals_jd(em),
              cuts=OrbitalSolutionCutsConfig.model_validate_json(cuts_json), ruwe_min=ruwe_min)


def _inject(task: tuple[dict[str, float], int, float, list[int]]) -> list[dict[str, Any]]:
    from darkhunter_pop import injection_test as it
    from darkhunter_pop.epoch_model import PER_CCD_NOISE_RNG_TAG, epoch_model_rng, run_cascade, source_context
    from darkhunter_pop.forward_model import seeded_global_rng

    v, sid, bp_rp, reals = task
    gm, cf = _W["gm"], _W["c_funcs"]
    src = source_context(v["ra"], v["dec"], float(v["g_mag"]), bp_rp=bp_rp)

    def predict() -> Any:
        return gm.predict_astrometry_binary_in_terms_of_a0(
            ra=v["ra"], dec=v["dec"], parallax=v["parallax"], pmra=v["pmra"], pmdec=v["pmdec"],
            period=v["period"], Tp=v["t_periastron"], ecc=v["eccentricity"], omega=v["Omega_rad"],
            inc=v["inc_rad"], w=v["omega_rad"], a0_mas=v["a0_mas"], phot_g_mean_mag=v["g_mag"],
            data_release="dr3", c_funcs=cf)

    out = []
    for r in reals:
        seeds = it.injection_rng_seeds(_W["base_seed"], sid, r)
        c0 = time.process_time()
        try:
            with seeded_global_rng(seeds, cf):
                run = run_cascade(gm, cf, predict, _W["em"], src,
                                  epoch_rng=epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r),
                                  noise_rng=epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r, tag=PER_CCD_NOISE_RNG_TAG),
                                  ruwe_min=_W["ruwe_min"], skip_acceleration=False, gaps_jd=_W["gaps"])
            rec = it.parse_cascade_result(run.cascade, n_visibility_periods=run.n_visibility_periods, n_obs=run.n_obs, cuts=_W["cuts"])
            rec.update(sim_n_visibility_periods=run.n_visibility_periods, sim_n_obs=run.n_obs, ruwe_scale=run.ruwe_scale)
        except Exception as exc:  # noqa: BLE001 - recorded, never dropped
            rec = {"error": repr(exc)}
        rec.update(source_id=int(sid), realization=int(r), bp_rp=bp_rp, cpu_s=time.process_time() - c0)
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--inj390", default=str(P / "output/gate390/injection_test_full.h5"))
    ap.add_argument("--colours", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400/inj390.h5"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-colour", action="store_true", help="switch dr3.epoch_model.noise_colour on")
    ap.add_argument("--enable-recal", action="store_true", help="switch dr3.epoch_model.transit_recal on (#581)")
    ap.add_argument("--stated-scale", type=float, nargs="*", default=None, help="override noise_colour.stated_scale")
    ap.add_argument("--n-realizations", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    import h5py

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.epoch_model import noise_model_version

    cfg = load_config(Path(a.config))
    em = cfg.dr3.epoch_model
    if a.enable_colour:
        nc = em.noise_colour.model_copy(update={"enabled": True})
        if a.stated_scale:
            nc = nc.model_copy(update={"stated_scale": list(a.stated_scale)})
        em = em.model_copy(update={"noise_colour": nc})
    if a.enable_recal:  # #581 (spec §8.14)
        em = em.model_copy(update={"transit_recal": em.transit_recal.model_copy(update={"enabled": True})})
    nmv = noise_model_version(em)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "inject_meta.json"
    meta = {"issue": [454, 572], "noise_model_version": nmv, "epoch_model": json.loads(em.model_dump_json()),
            "n_realizations": a.n_realizations}
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        if old["epoch_model"] != meta["epoch_model"]:
            raise SystemExit(f"{meta_p}: refusing to mix noise models in one output")
    meta_p.write_text(json.dumps(meta, indent=1))
    with h5py.File(a.inj390, "r") as f:
        sids = f["systems/source_id"][:]
        truth = {k: f["systems/truth"][k][:] for k in TRUTH_KEYS}
    with h5py.File(a.colours, "r") as f:
        cmap = dict(zip(f["source_id"][:].tolist(), f["bp_rp"][:].tolist()))
    log = out / "inject.jsonl"
    done = set()
    if log.exists():
        done = {(int(r["source_id"]), int(r["realization"])) for r in map(json.loads, log.read_text().splitlines())}
    tasks = []
    for i, sid in enumerate(sids):
        todo = [r for r in range(a.n_realizations) if (int(sid), r) not in done]
        if todo:
            tasks.append(({k: float(truth[k][i]) for k in TRUTH_KEYS}, int(sid), float(cmap.get(int(sid), np.nan)), todo))
    tasks = tasks[: a.limit] if a.limit else tasks
    pop = cfg.selection_function_astrometric.mock_population
    initargs = (int(pop.random_seed), cfg.dr3.selection_function_astrometric.orbital_solution_cuts.model_dump_json(),
                em.model_dump_json(), float(pop.ruwe_min))
    print(f"{len(tasks)} systems ({nmv}), workers={a.workers}", flush=True)
    t0 = time.time()
    with log.open("a") as fh, mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=initargs, maxtasksperchild=100) as pool:
        for i, recs in enumerate(pool.imap_unordered(_inject, tasks, chunksize=1)):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)} systems, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
