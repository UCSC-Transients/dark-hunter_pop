#!/usr/bin/env python3
"""Re-run the #390 injection set with the #400 epoch model (and the §3.3.1 noise term).

docs/EPOCH_MODEL_SPEC.md §5. For each of the 1,296 #390 systems and realizations
``0 .. R-1``:

1. ``gaiamock_mod.predict_astrometry_binary_in_terms_of_a0`` with the #390 truth and the
   #390 global-RNG seeds (``injection_test.injection_rng_seeds``), inside
   ``epoch_model.gost_epoch_model`` so gaiamock reads the thinned GOST list (gaps + the
   calibrated per-transit loss). gaiamock's own 10% row rejection still applies.
2. ``fit_full_astrometric_cascade`` on those epochs → variant ``epoch``.
3. The same epochs plus El-Badry et al. (2024) §3.3.1 per-transit noise for G < 13
   (``epoch_model.bright_star_excess_noise``) → variant ``epoch_noise``.

The 0.5 mas term for marginally resolved epochs (xi > 0.5) needs the two stars'
separation and flux ratio; the #390 truth is a photocentre orbit (a0 only), so that
term is undefined here and is not applied (spec §5).

Both fits use the same seeds as #390, so the only differences from the stored #390
realization are the epoch model and the noise term. Results append to
``<out>/realizations.jsonl`` and resume after a kill.

Usage::

    $PY scripts/validate_epoch_model_400.py run --inj390 output/gate390/injection_test_full.h5 \
        --out output/gate400/validation --n-realizations 3 --workers 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_W: dict[str, Any] = {}

TRUTH_KEYS = (
    "ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
    "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag",
)


def _worker_init(base_seed: int, cuts_json: str, epoch_cfg_json: str, ruwe_min: float, skip_acc: bool) -> None:
    from darkhunter_pop.config_schema import EpochModelPathConfig, OrbitalSolutionCutsConfig
    from darkhunter_pop.epoch_model import epoch_model_config_from_mapping, gap_intervals_jd
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    gm = import_gaiamock_mod()
    path_cfg = EpochModelPathConfig.model_validate_json(epoch_cfg_json)
    em = epoch_model_config_from_mapping(path_cfg)
    em = dataclasses.replace(em, enabled=True)
    _W.update(
        gm=gm, c_funcs=gm.read_in_C_functions(), base_seed=base_seed,
        cuts=OrbitalSolutionCutsConfig.model_validate_json(cuts_json),
        em=em, gaps=gap_intervals_jd(em), noise=path_cfg.excess_noise,
        ruwe_min=ruwe_min, skip_acc=skip_acc,
    )


def _one(values: dict[str, float], source_id: int, r: int) -> list[dict[str, Any]]:
    from darkhunter_pop import injection_test as it
    from darkhunter_pop.epoch_model import (
        EXCESS_NOISE_RNG_TAG,
        SourceEpochContext,
        bright_star_excess_noise,
        epoch_model_rng,
        fov_transit_ids,
        gost_epoch_model,
    )
    from darkhunter_pop.forward_model import seeded_global_rng

    gm, cf, em = _W["gm"], _W["c_funcs"], _W["em"]
    seeds = it.injection_rng_seeds(_W["base_seed"], source_id, r)
    rng = epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, source_id, r)
    with seeded_global_rng(seeds, cf), gost_epoch_model(
        gm, em, SourceEpochContext(g_mag=float(values["g_mag"])), rng, gaps_jd=_W["gaps"]
    ):
        t, psi, pf, obs, err = gm.predict_astrometry_binary_in_terms_of_a0(
            ra=values["ra"], dec=values["dec"], parallax=values["parallax"],
            pmra=values["pmra"], pmdec=values["pmdec"], period=values["period"],
            Tp=values["t_periastron"], ecc=values["eccentricity"], omega=values["Omega_rad"],
            inc=values["inc_rad"], w=values["omega_rad"], a0_mas=values["a0_mas"],
            phot_g_mean_mag=values["g_mag"], data_release="dr3", c_funcs=cf,
        )
    n_vis = int(np.sum(np.diff(t * 365.25) > it.GAIAMOCK_VISIBILITY_GAP_DAY) + 1) if t.size else 0
    n_tr = int(fov_transit_ids(t * 365.25, em.transit_split_day).max() + 1) if t.size else 0
    nz = _W["noise"]
    nrng = epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, source_id, r, tag=EXCESS_NOISE_RNG_TAG)
    extra, sigma_x = bright_star_excess_noise(
        t, float(values["g_mag"]), nrng, g_max=nz.g_max, sigma_max_mas=nz.sigma_max_mas,
        split_day=em.transit_split_day,
    )
    out = []
    for variant, o in (("epoch", obs), ("epoch_noise", obs + extra)):
        c0 = time.process_time()
        with seeded_global_rng(seeds, cf):
            cascade = gm.fit_full_astrometric_cascade(
                t_ast_yr=t, psi=psi, plx_factor=pf, ast_obs=o, ast_err=err, c_funcs=cf,
                verbose=False, show_residuals=False, ruwe_min=_W["ruwe_min"],
                skip_acceleration=_W["skip_acc"],
            )
        rec = it.parse_cascade_result(cascade, n_visibility_periods=n_vis, n_obs=len(t), cuts=_W["cuts"])
        rec.update(
            variant=variant, source_id=int(source_id), realization=int(r),
            sim_n_visibility_periods=n_vis, sim_n_obs=int(len(t)), sim_n_transits=n_tr,
            excess_sigma_mas=sigma_x if variant == "epoch_noise" else 0.0,
            cpu_s=time.process_time() - c0,
        )
        out.append(rec)
    return out


def _task(task: tuple[dict[str, float], int, list[int]]) -> list[dict[str, Any]]:
    values, sid, reals = task
    recs = []
    for r in reals:
        try:
            recs.extend(_one(values, sid, r))
        except Exception as exc:  # recorded, never dropped
            recs.append({"source_id": sid, "realization": r, "variant": "error", "error": repr(exc)})
    return recs


def cmd_run(args: argparse.Namespace) -> None:
    import h5py

    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    em_cfg = cfg.dr3.epoch_model
    if em_cfg is None:
        raise SystemExit("dr3.epoch_model is not configured")
    pop = cfg.selection_function_astrometric.mock_population
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.inj390, "r") as f:
        sids = f["systems/source_id"][:]
        truth = {k: f["systems/truth"][k][:] for k in TRUTH_KEYS}
    log = out / "realizations.jsonl"
    done: set[tuple[int, int]] = set()
    if log.exists():
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("variant") == "epoch_noise":
                done.add((int(rec["source_id"]), int(rec["realization"])))
    tasks = []
    for i, sid in enumerate(sids):
        todo = [r for r in range(args.n_realizations) if (int(sid), r) not in done]
        if todo:
            tasks.append(({k: float(truth[k][i]) for k in TRUTH_KEYS}, int(sid), todo))
    if args.limit:
        tasks = tasks[: args.limit]
    meta = {
        "issue": [400, 398], "inj390": str(args.inj390), "base_seed": int(pop.random_seed),
        "n_realizations": args.n_realizations, "ruwe_min": pop.ruwe_min,
        "skip_acceleration": pop.skip_acceleration,
        "epoch_model": json.loads(em_cfg.model_dump_json()),
        "variants": {"epoch": "epoch model only", "epoch_noise": "epoch model + El-Badry 2024 §3.3.1 G<13 per-transit term"},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    n = sum(len(t[2]) for t in tasks)
    print(f"{len(tasks)} systems / {n} realizations to run; workers={args.workers}", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    k = 0
    with log.open("a") as fh, ctx.Pool(
        args.workers, initializer=_worker_init,
        initargs=(int(pop.random_seed), cuts.model_dump_json(), em_cfg.model_dump_json(), pop.ruwe_min, pop.skip_acceleration),
        maxtasksperchild=50,
    ) as pool:
        for recs in pool.imap_unordered(_task, tasks, chunksize=1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            k += 1
            if k % 50 == 0:
                free = shutil.disk_usage(out).free / 2**30
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} systems, {el/60:.1f} min, disk free {free:.1f} GiB", flush=True)
                if free < 3.0:
                    pool.terminate()
                    raise SystemExit("disk free < 3 GiB; stopping")
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("--inj390", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-realizations", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
