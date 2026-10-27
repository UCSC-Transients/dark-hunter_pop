#!/usr/bin/env python3
"""Single-star control for the per-CCD noise model (#572 colour term, #454 F2; spec EPOCH_MODEL_SPEC §8.12).

``simulate`` (heavy; run through ``scripts/jobs/run_heavy.sh``)
    Rows of the looks-single parent (spec §12.33 flags from the decided launch fragment, every flag except
    ``single_ruwe`` so the DR3 RUWE tail stays visible; the RUWE < 1.25 cut is applied on both sides at
    comparison time), stratified in observed G. Each row becomes one mock single star at its own position,
    observed G and observed BP-RP, truth parallax 1000 / r_med_geo, observed proper motion. gaiamock's
    ``predict_astrometry_single_source`` runs inside ``epoch_model.gost_epoch_model`` (the adopted §8 epoch
    model), the configured per-CCD noise is added (``epoch_model.per_ccd_noise``: N2d, and the colour term when
    ``dr3.epoch_model.noise_colour`` is enabled), then gaiamock's ``fit_5par_solution_only``. Stored per star:
    UWE, RUWE = UWE / u0, nu, the stated per-CCD error, the formal and c-inflated parallax errors, the fitted
    parallax, N_obs / N_transits / N_vis. The mock excess noise follows from UWE because the stated error is
    the same for every CCD of a star: eps^2 = sigma^2 (UWE^2 - 1) for UWE > 1, else 0.

The DR3 side is the same rows (paired), from the parent snapshot plus, when present, the approved
``gaia_source`` astrometric-quality snapshot (gof_al, chi2_al, excess noise, n_good_obs_al).

Every worker pins BLAS to 1 thread (#408). Output JSONL is resumable; the meta file records the epoch-model
config, the noise-model version and the primary-model version of the fragment.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_W: dict[str, Any] = {}
STREAM = 572  # SeedSequence stream of the single-star control (issue number)
REAL_STRIDE = 64  # task index = parent row * REAL_STRIDE + realization


def _init(base_seed: int, em_json: str, cuts_json: str, acc_json: str, ruwe_min: float) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    _W["tp"] = threadpool_limits(limits=1)
    _W["threads"] = sorted({int(i["num_threads"]) for i in threadpool_info()})
    from darkhunter_pop.config_schema import EpochModelPathConfig
    from darkhunter_pop.epoch_model import epoch_model_config_from_mapping, gap_intervals_jd
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    em = epoch_model_config_from_mapping(EpochModelPathConfig.model_validate_json(em_json))
    from darkhunter_pop.config_schema import OrbitalSolutionCutsConfig
    from darkhunter_pop.proposal_set import AccelerationPublicationConfig

    _W.update(gm=gm, c_funcs=gm.read_in_C_functions(), base_seed=base_seed, em=em, gaps=gap_intervals_jd(em),
              cuts=OrbitalSolutionCutsConfig.model_validate_json(cuts_json),
              acc=AccelerationPublicationConfig.model_validate_json(acc_json), ruwe_min=ruwe_min)


def _star(task: tuple[int, int, dict[str, float]]) -> list[dict[str, Any]]:
    from darkhunter_pop import epoch_model as emod
    from darkhunter_pop.forward_model import mock_global_rng_seeds, seeded_global_rng

    idx, sid, v = task  # idx = row * REAL_STRIDE + realization
    gm, em = _W["gm"], _W["em"]
    c0 = time.process_time()
    src = emod.source_context(v["ra"], v["dec"], v["g"], bp_rp=v["bp_rp"])
    seeds = mock_global_rng_seeds(_W["base_seed"], STREAM, idx)
    with seeded_global_rng(seeds, _W["c_funcs"]), emod.gost_epoch_model(
        gm, em, src, emod.epoch_model_rng(_W["base_seed"], STREAM, idx), gaps_jd=_W["gaps"]
    ), emod.actual_noise_scale(gm, emod.colour_actual_scale(src, em)):
        t, psi, pf, obs, err = gm.predict_astrometry_single_source(
            ra=v["ra"], dec=v["dec"], parallax=v["plx_true"], pmra=v["pmra"], pmdec=v["pmdec"],
            phot_g_mean_mag=v["g"], data_release="dr3")
    obs, err, k = emod.per_ccd_noise(
        np.asarray(obs, float), np.asarray(err, float), v["g"], v["bp_rp"], em,
        emod.epoch_model_rng(_W["base_seed"], STREAM, idx, tag=emod.PER_CCD_NOISE_RNG_TAG))
    t = np.asarray(t, float)
    n_vis = emod.n_visibility_periods_from_days(t * 365.25, 4.0)
    n_tr = int(emod.fov_transit_ids(t * 365.25, em.transit_split_day).max() + 1) if t.size else 0
    rec: dict[str, Any] = {"idx": idx, "source_id": sid, "n_obs": int(t.size), "n_transits": n_tr, "n_vis": n_vis}
    if t.size > 6:
        res = gm.fit_5par_solution_only(t, psi, pf, obs, err)
        uwe, nu = float(res[10]), int(t.size) - 5
        cc = float(np.sqrt(uwe**2 / (1.0 - 2.0 / (9.0 * nu)) ** 3)) if nu > 0 else float("nan")
        rec.update(uwe=uwe, nu=nu, ruwe=uwe / k, ruwe_scale=k, plx_hat=float(res[4]),
                   sigma_plx_c=float(res[9]), sigma_plx_formal=float(res[9]) / cc if cc > 0 else float("nan"),
                   sigma5d_max=float(res[11]), stated_err=float(np.median(err)))
    # Full DR3 cascade (Ryan 2026-10-26): single stars misidentified as binaries. Same gate as proposal_set.simulate_one.
    sol, acc_ok, orb_ok = "insufficient_visibility", False, False
    if n_vis >= 12 and t.size >= 13:
        from darkhunter_pop.forward_model import classify_cascade_result
        from darkhunter_pop.proposal_set import published_acceleration

        with seeded_global_rng(seeds, _W["c_funcs"]):
            vec = gm.fit_full_astrometric_cascade(
                t_ast_yr=t, psi=psi, plx_factor=pf, ast_obs=obs, ast_err=err, c_funcs=_W["c_funcs"], verbose=False,
                show_residuals=False, ruwe_min=_W["ruwe_min"] * k, skip_acceleration=False)
        vec = emod.rescale_cascade_ruwe(list(vec), k)
        vec = [float(x) for x in vec] + [0.0] * (23 - len(vec))
        cr = classify_cascade_result(vec, m1_msun=1.0, m2_msun=0.0, flux_ratio=0.0, cuts=_W["cuts"], gaiamock=None)
        sol, orb_ok = cr.solution_type.value, bool(cr.accepted_orbital)
        acc_ok = published_acceleration(vec, _W["acc"])
        if vec[0] not in (-1.0, 0.0):
            rec["cascade"] = vec
    rec.update(solution_type=sol, accepted_orbital=orb_ok, published_acceleration=bool(acc_ok))
    rec.update(cpu_s=time.process_time() - c0, blas_threads=_W["threads"])
    return [rec]


def _pool_run(tasks: list[Any], log: Path, workers: int, initargs: tuple[Any, ...]) -> None:
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with log.open("a") as fh, ctx.Pool(workers, initializer=_init, initargs=initargs, maxtasksperchild=500) as pool:
        for i, recs in enumerate(pool.imap_unordered(_star, tasks, chunksize=8)):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            if (i + 1) % max(500, len(tasks) // 50) == 0:
                fh.flush()
                free = shutil.disk_usage(log.parent).free / 2**30
                print(f"  {i + 1}/{len(tasks)} stars, {(time.time() - t0) / 60:.1f} min, disk free {free:.1f} GiB", flush=True)
                if free < 3.0:
                    pool.terminate()
                    raise SystemExit("disk free < 3 GiB; stopping")
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def select_rows(parent_dir: Path, fragment: Path, cfg: Any) -> tuple[Any, np.ndarray, str]:
    """Looks-single parent rows (all §12.33 flags except ``single_ruwe``), G < system_g_max; original rows only."""
    from darkhunter_pop import proposal_set as ps

    frag = ps.load_proposal_set_fragment(fragment)
    prop = frag.proposal
    version = ps.primary_model_version(prop)
    prop = prop.model_copy(update={"parent_supplement": None,
                                   "parent": prop.parent.model_copy(update={"include_supplement": False,
                                                                            "extra_targets": ()})})
    par = ps.load_parent_snapshot(parent_dir, cfg, prop)
    ok = np.logical_and.reduce([v for k, v in par.flags.items() if k != "single_ruwe"])
    g = np.asarray(par.columns["phot_g_mean_mag"], float)
    ok &= np.isfinite(g) & (g < prop.single_catalog.system_g_max) & np.isfinite(np.asarray(par.columns["bp_rp"], float))
    ok &= np.isfinite(par.truth_parallax_mas) & (par.truth_parallax_mas > 0)
    return par, ok, version


def cmd_simulate(a: argparse.Namespace) -> None:
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path(a.config))
    em = cfg.dr3.epoch_model
    if em is None or not em.enabled:
        raise SystemExit("dr3.epoch_model must be configured and enabled")
    if a.enable_colour:  # §8.12 colour model (config section, switched on for this run)
        em = em.model_copy(update={"noise_colour": em.noise_colour.model_copy(update={"enabled": True})})
    par, ok, version = select_rows(Path(a.parent_dir), Path(a.fragment), cfg)
    g = np.asarray(par.columns["phot_g_mean_mag"], float)
    rng = np.random.default_rng(a.seed)
    edges = np.arange(a.g_min, a.g_max + 1e-9, a.g_step)
    pick = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero(ok & (g >= lo) & (g < hi))
        pick.append(rows if rows.size <= a.n_per_bin else np.sort(rng.choice(rows, a.n_per_bin, replace=False)))
    pick = np.concatenate(pick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "singles.jsonl"
    meta_p = out / "singles_meta.json"
    from darkhunter_pop.epoch_model import noise_model_version

    nmv = noise_model_version(em)
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        if old["noise_model_version"] != nmv or old["n_rows"] != int(pick.size) or old.get("realizations") != a.realizations:
            raise SystemExit(f"{meta_p}: refusing to mix runs ({old['noise_model_version']} vs {nmv})")
    else:
        meta_p.write_text(json.dumps({
            "issue": [572, 454], "noise_model_version": nmv, "primary_model_version": version,
            "parent": str(a.parent_dir), "fragment": str(a.fragment), "seed": a.seed, "stream": STREAM,
            "g_edges": edges.tolist(), "n_per_bin": a.n_per_bin, "n_rows": int(pick.size), "realizations": a.realizations,
            "real_stride": REAL_STRIDE,
            "base_seed": int(cfg.selection_function_astrometric.mock_population.random_seed),
            "epoch_model": json.loads(em.model_dump_json())}, indent=1))
    done = set()
    if log.exists():
        done = {int(json.loads(x)["idx"]) for x in log.read_text().splitlines() if x.strip()}
    c = par.columns
    tasks = []
    for i in pick:
        v = {"ra": float(c["ra"][i]), "dec": float(c["dec"][i]), "g": float(g[i]), "bp_rp": float(c["bp_rp"][i]),
             "plx_true": float(par.truth_parallax_mas[i]), "pmra": float(c["pmra"][i]), "pmdec": float(c["pmdec"][i])}
        for r in range(a.realizations):
            ti = int(i) * REAL_STRIDE + r
            if ti not in done:
                tasks.append((ti, int(c["source_id"][i]), v))
    tasks = tasks[: a.limit] if a.limit else tasks
    print(f"{len(tasks)} of {pick.size} stars to run ({nmv}; {version}), workers={a.workers}", flush=True)
    from darkhunter_pop import proposal_set as ps

    acc = ps.load_proposal_set_fragment(Path(a.fragment)).proposal.acceleration_publication
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    ruwe_min = ps.load_proposal_set_fragment(Path(a.fragment)).proposal.ruwe_min
    initargs = (int(cfg.selection_function_astrometric.mock_population.random_seed), em.model_dump_json(),
                cuts.model_dump_json(), acc.model_dump_json(), float(ruwe_min))
    _pool_run(tasks, log, a.workers, initargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("simulate")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--parent-dir", required=True)
    p.add_argument("--fragment", default="config/population/proposal_set_single46.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--g-min", type=float, default=6.0)
    p.add_argument("--g-max", type=float, default=19.0)
    p.add_argument("--g-step", type=float, default=0.5)
    p.add_argument("--n-per-bin", type=int, default=2500)
    p.add_argument("--seed", type=int, default=572)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--realizations", type=int, default=1, help="noise realizations per row (< REAL_STRIDE)")
    p.add_argument("--enable-colour", action="store_true", help="switch dr3.epoch_model.noise_colour on")
    p.set_defaults(func=cmd_simulate)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
