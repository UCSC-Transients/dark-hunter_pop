#!/usr/bin/env python3
"""Measure the insufficient-visibility channel: epoch-model mocks vs DR3 (#428).

gaiamock_mod's ``fit_full_astrometric_cascade`` returns flag 0 ("not enough visibility
periods") when a source has < 12 visibility periods or < 13 observations. The DR3 NSS
orbital pipeline likewise takes only sources with >= 12 visibility periods. Since #368
every forward-model mock goes through gaiamock, and the mock produced no flag-0 outcome.
This script asks whether the #400 v2 epoch model (``dr3.epoch_model``: the 138 DR3 gaps,
p(G, l, b) transit loss, clustered faint loss) restores that channel, and compares with
DR3's own fraction.

Subcommands, each resumable:

``fetch``
    Small uniform ``gaia_source`` slice (``random_index < --k``, G < 19, **every**
    ``astrometric_params_solved`` incl. 2-parameter solutions) with
    ``visibility_periods_used``. Written once to
    ``<data_root>/dr3/gaia_snapshots/<timestamp>_visibility_428/`` with ``meta.yaml``.
``simulate``
    Mock singles and binaries at real positions and G of a G-stratified, seeded subset of
    that slice, through ``epoch_model.run_cascade`` with the configured epoch model
    (``em``) and with it disabled (``bare`` = gaiamock alone). Records the cascade flag,
    the visibility periods and n_obs. Binary orbits are generic (they do not affect the
    epochs); the per-star seeds are shared by the two variants.
``report``
    Fractions with < 12 visibility periods / cascade flag 0 per G bin and per |ecliptic
    latitude| bin, mock vs DR3, and the DR3-G-weighted totals. Writes ``report.json``.

Every worker pins BLAS to 1 thread with ``threadpoolctl``. Run under ``nice``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

P = Path(__file__).resolve().parents[1]
#: SeedSequence stream for this measurement (disjoint from #390 / #400 streams).
STREAM = 428
COLUMNS: tuple[str, ...] = (
    "source_id", "random_index", "ra", "dec", "l", "b", "ecl_lat", "phot_g_mean_mag",
    "astrometric_params_solved", "visibility_periods_used", "astrometric_n_obs_al",
    "astrometric_matched_transits", "ruwe",
)
G_EDGES: tuple[float, ...] = (3.0, 13.0, 15.0, 16.0, 17.0, 18.0, 19.0)
ECL_EDGES: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
_W: dict[str, Any] = {}


def adql(k: int, g_max: float, top: int | None = None) -> str:
    """Uniform ``gaia_source`` slice, G < g_max, every solution type.

    ``top`` adds ``TOP n``: sync jobs are otherwise capped at 2,000 rows by the archive. The
    row count is checked against ``top`` after the query so a truncation cannot pass silently.
    """
    cols = ",\n  ".join(f"gs.{c}" for c in COLUMNS)
    head = f"SELECT TOP {int(top)}" if top else "SELECT"
    return (f"{head}\n  {cols}\nFROM gaiadr3.gaia_source AS gs\n"
            f"WHERE gs.random_index < {int(k)}\n  AND gs.phot_g_mean_mag < {g_max}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args: argparse.Namespace) -> None:
    import h5py
    import yaml
    from astroquery.gaia import Gaia

    top = args.sync_top if args.sync else None
    q = adql(args.k, args.g_max, top)
    print(q)
    Gaia.ROW_LIMIT = -1
    now = dt.datetime.now(dt.timezone.utc)
    out = Path(args.data_root) / "dr3" / "gaia_snapshots" / f"{now.strftime('%Y%m%dT%H%M%SZ')}_visibility_428"
    # The archive answered async jobs with HTTP 500 on 2026-10-04 while sync jobs worked.
    tab = (Gaia.launch_job(q) if args.sync else Gaia.launch_job_async(q)).get_results()
    if top is not None and len(tab) >= top:
        raise SystemExit(f"{len(tab)} rows = TOP {top}: result may be truncated; raise --sync-top")
    out.mkdir(parents=True, exist_ok=False)  # only after the archive answered
    path = out / "random_all_params.h5"
    with h5py.File(path, "w") as fh:
        for c in tab.colnames:
            dtype = np.int64 if c in ("source_id", "random_index") else np.float64
            data = np.asarray(tab[c], dtype=dtype) if dtype is np.int64 else np.ma.filled(
                np.ma.asarray(tab[c], dtype=np.float64), np.nan)
            fh.create_dataset(c, data=data)
    meta = {
        "snapshot_kind": "visibility_periods", "issue": [428], "query_date": now.isoformat(),
        "gaia_source_total_rows": 1_811_709_771, "k_random": args.k, "g_max": args.g_max,
        "tap_mode": "sync" if args.sync else "async",
        "tables": {"random_all_params": {"adql": q, "n_rows": int(len(tab)),
                                          "columns": list(tab.colnames), "sha256": _sha256(path)}},
    }
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"{len(tab)} rows -> {out}")


def _init(base_seed: int, em_json: str, ruwe_min: float, skip_acc: bool) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    _W["tp"] = threadpool_limits(limits=1)
    _W["threads"] = sorted({int(i["num_threads"]) for i in threadpool_info()})
    from darkhunter_pop.config_schema import EpochModelPathConfig
    from darkhunter_pop.epoch_model import epoch_model_config_from_mapping, gap_intervals_jd
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    em = epoch_model_config_from_mapping(EpochModelPathConfig.model_validate_json(em_json))
    _W.update(gm=gm, c_funcs=gm.read_in_C_functions(), base_seed=base_seed, em=em,
              bare=dataclasses.replace(em, enabled=False), gaps=gap_intervals_jd(em),
              ruwe_min=ruwe_min, skip_acc=skip_acc)


def _star(task: tuple[str, int, float, float, float, float, float]) -> list[dict[str, Any]]:
    from darkhunter_pop.epoch_model import PER_CCD_NOISE_RNG_TAG, SourceEpochContext, epoch_model_rng, run_cascade
    from darkhunter_pop.forward_model import mock_global_rng_seeds, seeded_global_rng

    kind, sid, ra, dec, g, lg, bg = task
    gm, cf, bs = _W["gm"], _W["c_funcs"], _W["base_seed"]
    idx = int(sid % 2**31)
    orb = np.random.default_rng(np.random.SeedSequence(bs, spawn_key=(STREAM, idx, 1)))
    period = float(10 ** orb.uniform(1.5, 3.5))
    kw = dict(ra=ra, dec=dec, parallax=2.0, pmra=0.0, pmdec=0.0, phot_g_mean_mag=g, data_release="dr3")

    def predict() -> Any:
        if kind == "single":
            return gm.predict_astrometry_single_source(**kw)
        return gm.predict_astrometry_binary_in_terms_of_a0(
            period=period, Tp=float(orb.uniform(0, period)), ecc=float(orb.uniform(0, 0.6)),
            omega=float(orb.uniform(0, 2 * np.pi)), inc=float(np.arccos(orb.uniform(-1, 1))),
            w=float(orb.uniform(0, 2 * np.pi)), a0_mas=float(10 ** orb.uniform(-1, 0.5)), c_funcs=cf, **kw)

    out = []
    for variant in ("bare", "em"):
        orb = np.random.default_rng(np.random.SeedSequence(bs, spawn_key=(STREAM, idx, 1)))
        period = float(10 ** orb.uniform(1.5, 3.5))
        c0 = time.process_time()
        try:
            with seeded_global_rng(mock_global_rng_seeds(bs, STREAM, idx), cf):
                run = run_cascade(
                    gm, cf, predict, _W[variant], SourceEpochContext(g_mag=g, l_deg=lg, b_deg=bg),
                    epoch_rng=epoch_model_rng(bs, STREAM, idx),
                    noise_rng=epoch_model_rng(bs, STREAM, idx, tag=PER_CCD_NOISE_RNG_TAG),
                    ruwe_min=_W["ruwe_min"], skip_acceleration=_W["skip_acc"], gaps_jd=_W["gaps"])
            rec = {"flag": float(run.cascade[0]), "n_vis": run.n_visibility_periods, "n_obs": run.n_obs,
                   "n_transits": run.n_transits}
        except Exception as exc:  # recorded, never dropped
            rec = {"error": repr(exc)}
        rec.update(kind=kind, variant=variant, source_id=int(sid), g=g, cpu_s=time.process_time() - c0,
                   blas_threads=_W["threads"])
        out.append(rec)
    return out


def _load(snapshot: Path) -> dict[str, np.ndarray]:
    import h5py

    # Own slice (every solution type) when fetched; else the #400 epoch-count slice
    # (``random.h5``: 5/6-parameter solutions only, so no 2-parameter fraction).
    path = snapshot / "random_all_params.h5"
    if not path.is_file():
        path = snapshot / "random.h5"
    with h5py.File(path, "r") as fh:
        return {k: fh[k][:] for k in fh if k in COLUMNS}


def cmd_simulate(args: argparse.Namespace) -> None:
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    pop = cfg.selection_function_astrometric.mock_population
    em = cfg.dr3.epoch_model
    if em is None or not em.enabled:
        raise SystemExit("dr3.epoch_model must be configured and enabled")
    d = _load(Path(args.snapshot))
    rng = np.random.default_rng(args.seed)
    tasks: list[Any] = []
    g = d["phot_g_mean_mag"]
    for kind, n in (("single", args.n_single), ("binary", args.n_binary)):
        for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:]):
            pool = np.flatnonzero((g >= lo) & (g < hi))
            for i in np.sort(rng.choice(pool, min(n, pool.size), replace=False)):
                tasks.append((kind, int(d["source_id"][i]), float(d["ra"][i]), float(d["dec"][i]),
                              float(g[i]), float(d["l"][i]), float(d["b"][i])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "sim.jsonl"
    done = set()
    if log.exists():
        done = {(r["kind"], int(r["source_id"])) for r in map(json.loads, log.read_text().splitlines())}
    tasks = [t for t in tasks if (t[0], t[1]) not in done]
    (out / "simulate_meta.json").write_text(json.dumps(
        {"issue": 428, "snapshot": str(args.snapshot), "seed": args.seed, "base_seed": int(pop.random_seed),
         "epoch_model": json.loads(em.model_dump_json()), "n_single_per_bin": args.n_single,
         "n_binary_per_bin": args.n_binary, "g_edges": G_EDGES}, indent=1))
    print(f"{len(tasks)} stars to run, workers={args.workers}", flush=True)
    initargs = (int(pop.random_seed), em.model_dump_json(), pop.ruwe_min, pop.skip_acceleration)
    t0 = time.time()
    with log.open("a") as fh, mp.get_context("spawn").Pool(
        args.workers, initializer=_init, initargs=initargs, maxtasksperchild=200
    ) as pool:
        for i, recs in enumerate(pool.imap_unordered(_star, tasks, chunksize=4)):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 200 == 0:
                free = shutil.disk_usage(out).free / 2**30
                print(f"  {i + 1}/{len(tasks)}, {(time.time() - t0) / 60:.1f} min, disk {free:.1f} GiB", flush=True)
                if free < 3.0:
                    pool.terminate()
                    raise SystemExit("disk free < 3 GiB; stopping")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def _frac(x: np.ndarray) -> dict[str, float]:
    n = int(x.size)
    k = int(np.sum(x))
    return {"n": n, "k": k, "frac": k / n if n else float("nan")}


def cmd_report(args: argparse.Namespace) -> None:
    d = _load(Path(args.snapshot))
    g, vpu, par = d["phot_g_mean_mag"], d["visibility_periods_used"], d["astrometric_params_solved"]
    ecl = np.abs(d["ecl_lat"])
    gb = np.digitize(g, G_EDGES) - 1
    rep: dict[str, Any] = {"dr3": {}, "mock": {}}
    ecl_of = dict(zip(d["source_id"].tolist(), ecl.tolist()))
    w_g = np.array([np.mean(gb == i) for i in range(len(G_EDGES) - 1)])
    rep["dr3"]["all"] = {"vpu_lt12": _frac(vpu < 12), "two_param": _frac(par == 3),
                         "vpu_lt12_among_5p6p": _frac(vpu[par > 3] < 12)}
    rep["dr3"]["by_g"] = {f"{G_EDGES[i]}-{G_EDGES[i + 1]}": {
        "vpu_lt12": _frac(vpu[gb == i] < 12), "two_param": _frac(par[gb == i] == 3)} for i in range(len(G_EDGES) - 1)}
    eb = np.digitize(ecl, ECL_EDGES) - 1
    rep["dr3"]["by_abs_ecl_lat"] = {f"{ECL_EDGES[i]}-{ECL_EDGES[i + 1]}": _frac(vpu[eb == i] < 12)
                                    for i in range(len(ECL_EDGES) - 1)}
    rep["g_weights_dr3"] = w_g.tolist()
    recs = [json.loads(x) for x in (Path(args.out) / "sim.jsonl").read_text().splitlines()]
    for kind in ("single", "binary"):
        for variant in ("bare", "em"):
            rr = [r for r in recs if r["kind"] == kind and r["variant"] == variant and "error" not in r]
            nerr = sum(1 for r in recs if r["kind"] == kind and r["variant"] == variant and "error" in r)
            if not rr:
                continue
            mg = np.array([r["g"] for r in rr])
            nv = np.array([r["n_vis"] for r in rr])
            no = np.array([r["n_obs"] for r in rr])
            fl = np.array([r["flag"] for r in rr])
            me = np.array([ecl_of[r["source_id"]] for r in rr])
            mgb = np.digitize(mg, G_EDGES) - 1
            meb = np.digitize(me, ECL_EDGES) - 1
            by_g = {f"{G_EDGES[i]}-{G_EDGES[i + 1]}": {
                "nvis_lt12": _frac(nv[mgb == i] < 12), "flag0": _frac(fl[mgb == i] == 0),
                "nobs_lt13": _frac(no[mgb == i] < 13), "mean_nvis": float(np.mean(nv[mgb == i]))}
                for i in range(len(G_EDGES) - 1)}
            wflag = float(sum(w_g[i] * by_g[k]["flag0"]["frac"] for i, k in enumerate(by_g)))
            wvis = float(sum(w_g[i] * by_g[k]["nvis_lt12"]["frac"] for i, k in enumerate(by_g)))
            rep["mock"][f"{kind}_{variant}"] = {
                "n": len(rr), "n_error": nerr, "by_g": by_g,
                "by_abs_ecl_lat": {f"{ECL_EDGES[i]}-{ECL_EDGES[i + 1]}": _frac(fl[meb == i] == 0)
                                   for i in range(len(ECL_EDGES) - 1)},
                "dr3_g_weighted_flag0": wflag, "dr3_g_weighted_nvis_lt12": wvis,
                "blas_threads": sorted({t for r in rr for t in r["blas_threads"]}),
            }
    (Path(args.out) / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--data-root", default=str(P / "data"))
    f.add_argument("--k", type=int, default=300_000)
    f.add_argument("--g-max", type=float, default=19.0)
    f.add_argument("--sync", action="store_true", help="synchronous TAP job (archive async outages)")
    f.add_argument("--sync-top", type=int, default=500_000, help="TOP n for sync jobs")
    f.set_defaults(func=cmd_fetch)
    for name, fn in (("simulate", cmd_simulate), ("report", cmd_report)):
        s = sub.add_parser(name)
        s.add_argument("--snapshot", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--seed", type=int, default=428)
        s.add_argument("--n-single", type=int, default=500, help="per G bin")
        s.add_argument("--n-binary", type=int, default=250, help="per G bin")
        s.add_argument("--workers", type=int, default=2)
        s.set_defaults(func=fn)
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
