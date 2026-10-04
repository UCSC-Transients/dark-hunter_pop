#!/usr/bin/env python3
"""Validate the #400 v2 epoch model (Ryan's 2026-10-03 decisions) and the N2 noise.

docs/EPOCH_MODEL_SPEC.md §5. Two subcommands, both resumable (append to JSONL):

``inject``
    The 1,296 #390 systems × R realizations with the #390 seeds. Inside
    ``epoch_model.run_cascade`` (``predict_astrometry_binary_in_terms_of_a0``, then
    ``fit_full_astrometric_cascade``). Two variants on the same epochs:
    ``v2`` (config epoch model, no bright-star noise) and ``v2_n2`` (plus the configured
    per-CCD bright-star noise, RUWE renormalised).
``single``
    Single-star RUWE check. Non-binary stars at the real positions and G of a seeded
    random subset of the RUWE-unselected random ``gaia_source`` snapshot
    (``predict_astrometry_single_source`` then gaiamock ``check_ruwe``). Variants:
    ``gaiamock`` (bare), ``v2``, ``v2_n2``. The DR3 RUWE of the same stars is the comparison
    (it includes real binaries in its tail).

Every worker pins BLAS to 1 thread with ``threadpoolctl`` and records the thread counts
it sees (#408). Run under ``nice``.
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
SINGLE_STREAM = 5  # SeedSequence stream for the single-star check


def _init(base_seed: int, cuts_json: str, em_json: str, ruwe_min: float, skip_acc: bool,
          variants: tuple[str, ...] = ("v2", "v2_n2")) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    _W["tp"] = threadpool_limits(limits=1)  # keep a reference: the limit stays applied
    _W["threads"] = sorted({int(i["num_threads"]) for i in threadpool_info()})
    from darkhunter_pop.config_schema import EpochModelPathConfig, OrbitalSolutionCutsConfig
    from darkhunter_pop.epoch_model import epoch_model_config_from_mapping, gap_intervals_jd
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    em = epoch_model_config_from_mapping(EpochModelPathConfig.model_validate_json(em_json))
    _W.update(gm=gm, c_funcs=gm.read_in_C_functions(), base_seed=base_seed,
              cuts=OrbitalSolutionCutsConfig.model_validate_json(cuts_json),
              em_u0=em, em_n2=dataclasses.replace(em, ruwe_u0=None),
              em_v2=dataclasses.replace(em, excess_noise=None, ruwe_u0=None),
              gaps=gap_intervals_jd(em), ruwe_min=ruwe_min, skip_acc=skip_acc, variants=tuple(variants))


def _galactic(ra: float, dec: float) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
    return float(c.l.deg), float(c.b.deg)


def _inject(task: tuple[dict[str, float], int, list[int]]) -> list[dict[str, Any]]:
    from darkhunter_pop import injection_test as it
    from darkhunter_pop.epoch_model import (
        PER_CCD_NOISE_RNG_TAG, SourceEpochContext, epoch_model_rng, run_cascade,
    )
    from darkhunter_pop.forward_model import seeded_global_rng

    v, sid, reals = task
    gm, cf = _W["gm"], _W["c_funcs"]
    lg, bg = _galactic(v["ra"], v["dec"])
    src = SourceEpochContext(g_mag=float(v["g_mag"]), l_deg=lg, b_deg=bg)

    def predict() -> Any:
        return gm.predict_astrometry_binary_in_terms_of_a0(
            ra=v["ra"], dec=v["dec"], parallax=v["parallax"], pmra=v["pmra"], pmdec=v["pmdec"],
            period=v["period"], Tp=v["t_periastron"], ecc=v["eccentricity"], omega=v["Omega_rad"],
            inc=v["inc_rad"], w=v["omega_rad"], a0_mas=v["a0_mas"], phot_g_mean_mag=v["g_mag"],
            data_release="dr3", c_funcs=cf)

    out = []
    for r in reals:
        seeds = it.injection_rng_seeds(_W["base_seed"], sid, r)
        for variant in _W["variants"]:
            c0 = time.process_time()
            try:
                with seeded_global_rng(seeds, cf):
                    run = run_cascade(
                        gm, cf, predict, _W["em_" + variant.split("_")[-1]] if "_" in variant else _W["em_v2"], src,
                        epoch_rng=epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r),
                        noise_rng=epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r, tag=PER_CCD_NOISE_RNG_TAG),
                        ruwe_min=_W["ruwe_min"], skip_acceleration=_W["skip_acc"], gaps_jd=_W["gaps"])
                rec = it.parse_cascade_result(run.cascade, n_visibility_periods=run.n_visibility_periods,
                                              n_obs=run.n_obs, cuts=_W["cuts"])
                rec.update(sim_n_visibility_periods=run.n_visibility_periods, sim_n_obs=run.n_obs,
                           sim_n_transits=run.n_transits, ruwe_scale=run.ruwe_scale)
            except Exception as exc:  # recorded, never dropped
                rec = {"error": repr(exc)}
            rec.update(variant=variant, source_id=int(sid), realization=int(r),
                       cpu_s=time.process_time() - c0, blas_threads=_W["threads"])
            out.append(rec)
    return out


def _single(task: tuple[int, float, float, float]) -> list[dict[str, Any]]:
    from darkhunter_pop.epoch_model import (
        PER_CCD_NOISE_RNG_TAG, SourceEpochContext, epoch_model_rng, gost_epoch_model, per_ccd_excess_noise,
    )
    from darkhunter_pop.forward_model import mock_global_rng_seeds, seeded_global_rng

    sid, ra, dec, g = task
    gm, cf = _W["gm"], _W["c_funcs"]
    lg, bg = _galactic(ra, dec)
    src = SourceEpochContext(g_mag=float(g), l_deg=lg, b_deg=bg)
    seeds = mock_global_rng_seeds(_W["base_seed"], SINGLE_STREAM, sid)
    out = []
    for variant in ("gaiamock",) + _W["variants"]:
        em = _W["em_" + variant.split("_")[-1]] if "_" in variant else _W["em_v2"]
        em = dataclasses.replace(em, enabled=variant != "gaiamock")
        with seeded_global_rng(seeds, cf), gost_epoch_model(
            gm, em, src, epoch_model_rng(_W["base_seed"], SINGLE_STREAM, sid), gaps_jd=_W["gaps"]
        ):
            t, psi, pf, obs, err = gm.predict_astrometry_single_source(
                ra=ra, dec=dec, parallax=1.0, pmra=0.0, pmdec=0.0, phot_g_mean_mag=g, data_release="dr3")
        k = 1.0
        if variant != "gaiamock" and em.excess_noise is not None:
            extra, k = per_ccd_excess_noise(err, g, em.excess_noise,
                                            epoch_model_rng(_W["base_seed"], SINGLE_STREAM, sid, tag=PER_CCD_NOISE_RNG_TAG))
            obs = obs + extra
        if variant != "gaiamock" and em.ruwe_u0 is not None:
            k = em.ruwe_u0(g)
        nvis = int(np.sum(np.diff(np.sort(t) * 365.25) > 4.0) + 1) if len(t) else 0
        ruwe = float(gm.check_ruwe(t, psi, pf, obs, err)[0]) / k if len(t) > 6 else float("nan")
        out.append({"variant": variant, "source_id": int(sid), "g": g, "ruwe": ruwe, "n_obs": int(len(t)),
                    "n_vis": nvis, "ruwe_scale": k, "blas_threads": _W["threads"]})
    return out


U0_STREAM = 6  # SeedSequence stream for the u0 calibration singles


def _u0_star(task: tuple[int, int, float, float, float]) -> list[dict[str, Any]]:
    """One mock single star for the u0 calibration: v2 epochs + bright noise, gaiamock UWE."""
    from darkhunter_pop.epoch_model import (
        PER_CCD_NOISE_RNG_TAG, SourceEpochContext, epoch_model_rng, gost_epoch_model, per_ccd_excess_noise,
    )
    from darkhunter_pop.forward_model import mock_global_rng_seeds, seeded_global_rng

    ib, j, ra, dec, g = task
    gm, cf = _W["gm"], _W["c_funcs"]
    lg, bg = _galactic(ra, dec)
    src = SourceEpochContext(g_mag=float(g), l_deg=lg, b_deg=bg)
    em = dataclasses.replace(_W["em_n2"], enabled=True, ruwe_u0=None)
    idx = ib * 100_000 + j
    seeds = mock_global_rng_seeds(_W["base_seed"], U0_STREAM, idx)
    with seeded_global_rng(seeds, cf), gost_epoch_model(
        gm, em, src, epoch_model_rng(_W["base_seed"], U0_STREAM, idx), gaps_jd=_W["gaps"]
    ):
        t, psi, pf, obs, err = gm.predict_astrometry_single_source(
            ra=ra, dec=dec, parallax=1.0, pmra=0.0, pmdec=0.0, phot_g_mean_mag=g, data_release="dr3")
    if em.excess_noise is not None:
        extra, _ = per_ccd_excess_noise(err, g, em.excess_noise,
                                        epoch_model_rng(_W["base_seed"], U0_STREAM, idx, tag=PER_CCD_NOISE_RNG_TAG))
        obs = obs + extra
    uwe = float(gm.check_ruwe(t, psi, pf, obs, err)[0]) if len(t) > 6 else float("nan")
    return [{"bin": ib, "j": j, "g": g, "uwe": uwe, "n_obs": int(len(t)), "blas_threads": _W["threads"]}]


def _pool_run(fn: Any, tasks: list[Any], log: Path, workers: int, initargs: tuple[Any, ...]) -> None:
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with log.open("a") as fh, ctx.Pool(workers, initializer=_init, initargs=initargs, maxtasksperchild=200) as pool:
        for i, recs in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % max(100, len(tasks) // 50) == 0:
                free = shutil.disk_usage(log.parent).free / 2**30
                print(f"  {i + 1}/{len(tasks)} tasks, {(time.time() - t0) / 60:.1f} min, disk free {free:.1f} GiB", flush=True)
                if free < 3.0:
                    pool.terminate()
                    raise SystemExit("disk free < 3 GiB; stopping")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _common(args: argparse.Namespace) -> tuple[Any, tuple[Any, ...]]:
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    pop = cfg.selection_function_astrometric.mock_population
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    em = cfg.dr3.epoch_model
    if em is None or not em.enabled:
        raise SystemExit("dr3.epoch_model must be configured and enabled")
    upd: dict[str, Any] = {}
    if em.bright_excess_noise is not None:
        upd["bright_excess_noise"] = em.bright_excess_noise.model_copy(update={"enabled": True})
    if em.ruwe_u0 is not None:
        upd["ruwe_u0"] = em.ruwe_u0.model_copy(update={"enabled": True})
    em = em.model_copy(update=upd)  # validation variants switch the parts on/off themselves
    initargs = (int(pop.random_seed), cuts.model_dump_json(), em.model_dump_json(), pop.ruwe_min,
                pop.skip_acceleration, tuple(args.variants))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.cmd}_meta.json").write_text(json.dumps(
        {"issue": [400, 398], "epoch_model": json.loads(em.model_dump_json()), "base_seed": int(pop.random_seed),
         "args": {k: str(v) for k, v in vars(args).items() if k != "func"}}, indent=1))
    return cfg, initargs


def cmd_inject(args: argparse.Namespace) -> None:
    import h5py

    _, initargs = _common(args)
    with h5py.File(args.inj390, "r") as f:
        sids = f["systems/source_id"][:]
        truth = {k: f["systems/truth"][k][:] for k in TRUTH_KEYS}
    log = Path(args.out) / "inject.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("variant") == args.variants[-1]:
                done.add((int(r["source_id"]), int(r["realization"])))
    tasks = []
    for i, sid in enumerate(sids):
        todo = [r for r in range(args.n_realizations) if (int(sid), r) not in done]
        if todo:
            tasks.append(({k: float(truth[k][i]) for k in TRUTH_KEYS}, int(sid), todo))
    tasks = tasks[: args.limit] if args.limit else tasks
    print(f"{len(tasks)} systems to run, workers={args.workers}", flush=True)
    _pool_run(_inject, tasks, log, args.workers, initargs)


def cmd_single(args: argparse.Namespace) -> None:
    import h5py

    _, initargs = _common(args)
    with h5py.File(Path(args.snapshot) / "random.h5", "r") as f:
        d = {k: f[k][:] for k in ("source_id", "ra", "dec", "phot_g_mean_mag")}
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(d["source_id"].size, min(args.n, d["source_id"].size), replace=False))
    log = Path(args.out) / "single.jsonl"
    done = set()
    if log.exists():
        done = {int(json.loads(x)["source_id"]) for x in log.read_text().splitlines()}
    tasks = [(int(d["source_id"][i]), float(d["ra"][i]), float(d["dec"][i]), float(d["phot_g_mean_mag"][i]))
             for i in idx if int(d["source_id"][i]) not in done]
    print(f"{len(tasks)} stars to run, workers={args.workers}", flush=True)
    _pool_run(_single, tasks, log, args.workers, initargs)


def cmd_u0(args: argparse.Namespace) -> None:
    """Mock single stars on a G grid at random snapshot positions, for u0_mock(G)."""
    import h5py

    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    em = cfg.dr3.epoch_model
    if em is None or em.bright_excess_noise is None:
        raise SystemExit("dr3.epoch_model.bright_excess_noise must be configured (it may be disabled)")
    em_on = em.model_copy(update={"enabled": True,
                                  "bright_excess_noise": em.bright_excess_noise.model_copy(update={"enabled": True})})
    pop = cfg.selection_function_astrometric.mock_population
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    initargs = (int(pop.random_seed), cuts.model_dump_json(), em_on.model_dump_json(), pop.ruwe_min, pop.skip_acceleration)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(Path(args.snapshot) / "random.h5", "r") as f:
        ra, dec = f["ra"][:], f["dec"][:]
    rng = np.random.default_rng(args.seed)
    edges = np.round(np.arange(args.g_min, args.g_max + 1e-9, args.g_step), 4)
    tasks = []
    for ib, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        rows = rng.integers(0, ra.size, args.n_per_bin)
        gs = rng.uniform(lo, hi, args.n_per_bin)
        tasks += [(ib, j, float(ra[r]), float(dec[r]), float(g)) for j, (r, g) in enumerate(zip(rows, gs))]
    log = out / "u0_singles.jsonl"
    if log.exists():
        raise SystemExit(f"{log} exists; refusing to mix runs")
    (out / "u0_meta.json").write_text(json.dumps({"issue": 400, "edges": edges.tolist(), "n_per_bin": args.n_per_bin,
                                                   "seed": args.seed, "epoch_model": json.loads(em_on.model_dump_json())}, indent=1))
    print(f"{len(tasks)} mock singles, workers={args.workers}", flush=True)
    _pool_run(_u0_star, tasks, log, args.workers, initargs)


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True, dest="cmd")
    p = sub.add_parser("inject")
    p.add_argument("--inj390", default=str(P / "output/gate390/injection_test_full.h5"))
    p.add_argument("--out", default=str(P / "output/gate400/validation_v2"))
    p.add_argument("--n-realizations", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variants", nargs="+", default=["v2", "v2_n2"],
                   help="v2 (epochs only), v2_n2 (+ noise, sqrt(1+r2) RUWE), v2_u0 (+ noise, RUWE = UWE/u0)")
    p.set_defaults(func=cmd_inject)
    p = sub.add_parser("single")
    p.add_argument("--snapshot", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400"))
    p.add_argument("--out", default=str(P / "output/gate400/validation_v2"))
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=4001)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--variants", nargs="+", default=["v2", "v2_n2"],
                   help="v2 (epochs only), v2_n2 (+ noise, sqrt(1+r2) RUWE), v2_u0 (+ noise, RUWE = UWE/u0)")
    p.set_defaults(func=cmd_single)
    p = sub.add_parser("u0")
    p.add_argument("--snapshot", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400"))
    p.add_argument("--out", default=str(P / "output/gate400/u0"))
    p.add_argument("--g-min", type=float, default=4.0)
    p.add_argument("--g-max", type=float, default=19.0)
    p.add_argument("--g-step", type=float, default=0.2)
    p.add_argument("--n-per-bin", type=int, default=1500)
    p.add_argument("--seed", type=int, default=4006)
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_u0)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
