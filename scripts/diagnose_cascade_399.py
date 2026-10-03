#!/usr/bin/env python3
"""Take apart the #390 cascade outcomes: acceleration capture (#399) and noise (#398).

Reads the #390 artifact (``output/gate390/injection_test_full.h5``), replays each
realization's epochs bit for bit (``darkhunter_pop.cascade_replay``) and records what
``gaiamock_mod.fit_full_astrometric_cascade`` computed and discarded.

Subcommands
-----------
``linear``
    For every stored realization (and optionally extra seeded realizations of the
    long-period systems), the 5/9/7-parameter statistics under several
    ``noise_scale:err_scale`` variants, the branch they imply, the orbit coverage,
    and a check that the baseline variant reproduces every stored outcome. Cheap
    (~20 ms per realization per variant).
``orbit``
    The orbit fit the cascade skipped (``skip_acceleration=True``) on the identical
    data, for realizations that ended as accepted accelerations, plus a control set
    that reached the orbit in #390 (must reproduce the stored fit). Expensive
    (~14 CPU s each); resumable JSONL; use at most 2 workers on a shared laptop.

Figures and the report: ``scripts/plot_cascade_399.py``.

Example::

    PY=.venv/bin/python
    $PY scripts/diagnose_cascade_399.py linear --workers 1
    $PY scripts/diagnose_cascade_399.py orbit --workers 2 --max 400 --n-control 20
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import cascade_replay as cr  # noqa: E402
from darkhunter_pop import injection_test as it  # noqa: E402
from darkhunter_pop.forward_model import GlobalRNGSeeds  # noqa: E402

#: Primary checkout (gitignored ``output/`` lives there, not in worktrees).
PRIMARY = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
DEFAULT_IN = PRIMARY / "output" / "gate390" / "injection_test_full.h5"
DEFAULT_OUT = PRIMARY / "output" / "gate399"
#: Default variants as noise_scale:err_scale. 1.11 is the #398 sigma deficit;
#: 1.11:1.07 also matches the #398 F2 offset (recovered F2 1.2 low); <1 is higher S/N.
DEFAULT_VARIANTS = "1:1,1.11:1.11,1.11:1.07,1.25:1.25,0.71:0.71,0.5:0.5"
STAT_FIELDS = ("ruwe", "s9", "f2_9", "plx_snr9", "s7", "f2_7", "plx_snr7")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as f:
        truth = {k: f["systems/truth"][k][()] for k in f["systems/truth"]}
        truth["source_id"] = f["systems/source_id"][()]
        truth["nss_solution_type"] = f["systems/nss_solution_type"][()].astype(str)
        real = {
            k: f["realizations"][k][()]
            for k in ("system_index", "source_id", "realization", "outcome", "rng_seed_numpy", "rng_seed_c_rand")
        }
        real["recovered_period"] = f["realizations/recovered/period"][()]
        real["recovered_a0"] = f["realizations/recovered/a0_mas"][()]
        real["recovered_f2"] = f["realizations/recovered/goodness_of_fit"][()]
        attrs = dict(f.attrs)
    return truth, real, attrs


_W: dict[str, Any] = {}


def _init(variants: list[tuple[float, float]], ruwe_min: float) -> None:
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    _W.update(gm=gm, c=gm.read_in_C_functions(), variants=variants, ruwe_min=ruwe_min)


def _linear_task(task: tuple[dict[str, float], list[tuple[Any, ...]]]) -> list[dict[str, Any]]:
    """Items are ``(system_index, realization, numpy_seed, c_seed[, variants])``."""
    values, items = task
    out = []
    for item in items:
        sys_idx, r, nseed, cseed = item[:4]
        variants = item[4] if len(item) > 4 else _W["variants"]
        seeds = GlobalRNGSeeds(numpy_seed=nseed, c_rand_seed=cseed)
        ep = cr.replay_injection_epochs(_W["gm"], _W["c"], values, seeds, data_release="dr3")
        span, phase_span, peri = cr.orbit_coverage(ep.t_ast_yr, values["period"], values["t_periastron"])
        rec: dict[str, Any] = {
            "system_index": sys_idx,
            "realization": r,
            "span_day": span,
            "phase_span": phase_span,
            "periastron_in_window": peri,
            "n_obs": len(ep.t_ast_yr),
        }
        # noise-free misfit of the 9/7-parameter models (chi2 from F2 on the signal alone)
        st0 = cr.linear_cascade_statistics(_W["gm"], ep.t_ast_yr, ep.psi, ep.plx_factor, ep.signal, ep.ast_err)
        rec["chi2_misfit9"] = cr.chi2_from_f2(st0.f2_9, st0.n_obs - 9)
        rec["chi2_misfit7"] = cr.chi2_from_f2(st0.f2_7, st0.n_obs - 7)
        rec["noise_rms_over_err"] = float(np.sqrt(np.mean((ep.noise / ep.ast_err) ** 2)))
        for v, (kn, ke) in enumerate(variants):
            rec[f"noise_scale__{v}"], rec[f"err_scale__{v}"] = kn, ke
            obs, err = ep.observations(noise_scale=kn, err_scale=ke)
            st = cr.linear_cascade_statistics(_W["gm"], ep.t_ast_yr, ep.psi, ep.plx_factor, obs, err)
            for fld in STAT_FIELDS:
                rec[f"{fld}__{v}"] = getattr(st, fld)
            rec[f"branch__{v}"] = cr.predicted_branch(st, ruwe_min=_W["ruwe_min"])
            rec["n_vis"] = st.n_vis
        out.append(rec)
    return out


def cmd_linear(args: argparse.Namespace) -> None:
    truth, real, attrs = _load(Path(args.input))
    variants = [(float(a), float(b)) for a, b in (x.split(":") for x in args.variants.split(","))]
    if variants[0] != (1.0, 1.0):
        raise SystemExit("first variant must be 1:1 (the replay check uses it)")
    base_seed = int(attrs["base_seed"])
    n_sys = len(truth["source_id"])
    per_sys: dict[int, list[tuple[int, int, int, int]]] = {i: [] for i in range(n_sys)}
    for k in range(len(real["system_index"])):
        i = int(real["system_index"][k])
        r = int(real["realization"][k])
        s = it.injection_rng_seeds(base_seed, int(truth["source_id"][i]), r)
        if (s.numpy_seed, s.c_rand_seed) != (int(real["rng_seed_numpy"][k]), int(real["rng_seed_c_rand"][k])):
            raise SystemExit(f"stored seeds differ from the scheme for row {k}")
        per_sys[i].append((i, r, s.numpy_seed, s.c_rand_seed))
    n_stored = int(real["realization"].max()) + 1
    n_extra = 0
    if args.extra_realizations > 0:
        for i in range(n_sys):
            if truth["period"][i] >= args.extra_min_period:
                for r in range(n_stored, n_stored + args.extra_realizations):
                    s = it.injection_rng_seeds(base_seed, int(truth["source_id"][i]), r)
                    per_sys[i].append((i, r, s.numpy_seed, s.c_rand_seed))
                    n_extra += 1
    keys = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
            "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag")
    tasks = [({k: float(truth[k][i]) for k in keys}, items) for i, items in per_sys.items() if items]
    n_tot = sum(len(t[1]) for t in tasks)
    print(f"{n_tot} realizations ({n_extra} extra) x {len(variants)} variants, workers={args.workers}", flush=True)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(variants, float(attrs["ruwe_min"]))) as pool:
        for recs in pool.imap_unordered(_linear_task, tasks, chunksize=4):
            rows.extend(recs)
            if len(rows) % 2000 < len(recs):
                print(f"  {len(rows)}/{n_tot}  {(time.time()-t0)/60:.1f} min", flush=True)
    rows.sort(key=lambda d: (d["system_index"], d["realization"]))
    # replay check: baseline branch must equal the stored #390 outcome
    stored = {(int(a), int(b)): int(c) for a, b, c in zip(real["system_index"], real["realization"], real["outcome"])}
    mism = [(d["system_index"], d["realization"], stored[(d["system_index"], d["realization"])], d["branch__0"])
            for d in rows if (d["system_index"], d["realization"]) in stored
            and stored[(d["system_index"], d["realization"])] != d["branch__0"]]
    n_checked = sum(1 for d in rows if (d["system_index"], d["realization"]) in stored)
    print(f"replay check: {n_checked - len(mism)}/{n_checked} stored outcomes reproduced", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "linear_replay.h5"
    with h5py.File(path, "w") as h:
        for key in rows[0]:
            h.create_dataset(key, data=np.array([d[key] for d in rows]))
        h.attrs["variants_noise_err"] = json.dumps(variants)
        h.attrs["input"] = str(args.input)
        h.attrs["input_sha256"] = it_sha(Path(args.input))
        h.attrs["n_stored_realizations"] = n_stored
        h.attrs["replay_mismatches"] = json.dumps(mism)
        h.attrs["git_commit"] = _git_commit()
        h.attrs["issue"] = 399
    print(f"wrote {path} in {(time.time()-t0)/60:.1f} min; {len(mism)} mismatches", flush=True)


def _visibility(path: Path, source_ids: np.ndarray) -> np.ndarray:
    rows = json.loads(path.read_text())["rows"]
    return np.array(
        [rows.get(str(int(s)), {}).get("astrometric_n_good_obs_al", np.nan) for s in source_ids],
        dtype=np.float64,
    )


def dr3_matched_scales(
    truth: dict[str, np.ndarray], real: dict[str, np.ndarray], n_good: np.ndarray,
    recovered_f2: np.ndarray, recovered_nobs: np.ndarray, accepted: np.ndarray, g_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Per-system ``k_N`` and per-G-bin ``k_c`` that map gaiamock's noise onto DR3's (#398).

    ``k_N = sqrt(median n_obs(gaiamock) / astrometric_n_good_obs_al(DR3))`` (more epochs in
    the mock is equivalent to proportionally lower noise per epoch). ``k_c`` is the median
    ``c_DR3 / c_mock`` (Halbwachs et al. 2023 Eq. 2 from published vs recovered F2) over
    accepted ``Orbital`` realizations in each G bin: the residual scatter DR3 saw beyond
    its stated errors and gaiamock did not simulate (bright-star excess noise).
    """
    n_sys = len(truth["source_id"])
    si = real["system_index"]
    med_nobs = np.array([np.median(recovered_nobs[si == i]) for i in range(n_sys)])
    k_n = np.sqrt(med_nobs / n_good)
    orb = truth["nss_solution_type"][si] == "Orbital"
    m = accepted & orb & np.isfinite(n_good[si])
    c_mock = cr.inflation_factor_from_f2(recovered_f2[m], recovered_nobs[m] - 12)
    c_dr3 = cr.inflation_factor_from_f2(truth["goodness_of_fit"][si[m]], n_good[si[m]] - 12)
    ratio = c_dr3 / c_mock
    g = truth["g_mag"][si[m]]
    k_c_bins = []
    for lo, hi in zip(g_edges[:-1], g_edges[1:]):
        sel = (g >= lo) & (g < hi) & np.isfinite(ratio)
        k_c_bins.append(float(np.median(ratio[sel])) if sel.any() else 1.0)
    gb = np.clip(np.searchsorted(g_edges, truth["g_mag"], side="right") - 1, 0, len(k_c_bins) - 1)
    k_c = np.array(k_c_bins)[gb]
    return k_n, k_c, k_c_bins


def cmd_matched(args: argparse.Namespace) -> None:
    truth, real, attrs = _load(Path(args.input))
    with h5py.File(args.input, "r") as f:
        rec_f2 = f["realizations/recovered/goodness_of_fit"][()]
        rec_nobs = f["realizations/recovered/n_obs"][()]
        accepted = f["realizations/accepted"][()]
    n_good = _visibility(Path(args.visibility), truth["source_id"])
    g_edges = np.array([float(x) for x in args.g_edges.split(",")])
    k_n, k_c, k_c_bins = dr3_matched_scales(truth, real, n_good, rec_f2, rec_nobs, accepted, g_edges)
    print("k_c per G bin:", dict(zip(args.g_edges.split(",")[:-1], np.round(k_c_bins, 3))), flush=True)
    base_seed = int(attrs["base_seed"])
    n_sys = len(truth["source_id"])
    n_stored = int(real["realization"].max()) + 1
    per_sys: dict[int, list[tuple[Any, ...]]] = {i: [] for i in range(n_sys)}
    for i in range(n_sys):
        if not np.isfinite(k_n[i]):
            continue
        var = [(1.0, 1.0), (float(k_n[i] * k_c[i]), float(k_n[i])), (float(k_n[i]), float(k_n[i]))]
        n_r = n_stored + (args.extra_realizations if truth["period"][i] >= args.extra_min_period else 0)
        for r in range(n_r):
            s = it.injection_rng_seeds(base_seed, int(truth["source_id"][i]), r)
            per_sys[i].append((i, r, s.numpy_seed, s.c_rand_seed, var))
    keys = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
            "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag")
    tasks = [({k: float(truth[k][i]) for k in keys}, items) for i, items in per_sys.items() if items]
    n_tot = sum(len(t[1]) for t in tasks)
    print(f"{n_tot} realizations x 3 variants (baseline, DR3-matched, N-only); workers={args.workers}", flush=True)
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=([(1.0, 1.0)], float(attrs["ruwe_min"]))) as pool:
        for recs in pool.imap_unordered(_linear_task, tasks, chunksize=4):
            rows.extend(recs)
            if len(rows) % 2000 < len(recs):
                print(f"  {len(rows)}/{n_tot}  {(time.time()-t0)/60:.1f} min", flush=True)
    rows.sort(key=lambda d: (d["system_index"], d["realization"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "linear_matched.h5"
    with h5py.File(path, "w") as h:
        for key in rows[0]:
            h.create_dataset(key, data=np.array([d[key] for d in rows]))
        h.attrs["variants"] = "0: baseline; 1: noise x k_N k_c, err x k_N (DR3-matched); 2: noise, err x k_N"
        h.attrs["g_edges"] = json.dumps(g_edges.tolist())
        h.attrs["k_c_bins"] = json.dumps(k_c_bins)
        h.attrs["input"] = str(args.input)
        h.attrs["input_sha256"] = it_sha(Path(args.input))
        h.attrs["visibility"] = str(args.visibility)
        h.attrs["git_commit"] = _git_commit()
        h.attrs["issue"] = 399
    print(f"wrote {path} in {(time.time()-t0)/60:.1f} min", flush=True)


def it_sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _orbit_task(task: tuple[dict[str, float], int, int, int, int, str]) -> dict[str, Any]:
    values, sys_idx, r, nseed, cseed, role = task
    c0 = time.process_time()
    seeds = GlobalRNGSeeds(numpy_seed=nseed, c_rand_seed=cseed)
    ep = cr.replay_injection_epochs(_W["gm"], _W["c"], values, seeds, data_release="dr3")
    vec = cr.forced_orbit_fit(_W["gm"], _W["c"], ep, seeds, ruwe_min=_W["ruwe_min"])
    return {"system_index": sys_idx, "realization": r, "role": role, "cascade": vec,
            "cpu_s": time.process_time() - c0}


def cmd_orbit(args: argparse.Namespace) -> None:
    truth, real, attrs = _load(Path(args.input))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "forced_orbit.jsonl"
    done: set[tuple[int, int]] = set()
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((int(d["system_index"]), int(d["realization"])))
    keys = ("ra", "dec", "parallax", "pmra", "pmdec", "period", "t_periastron", "eccentricity",
            "Omega_rad", "inc_rad", "omega_rad", "a0_mas", "g_mag")
    accel = np.isin(real["outcome"], (7, 9))
    per = truth["period"][real["system_index"]]
    sel = np.flatnonzero(accel & (per >= args.min_period))
    rng = np.random.default_rng(399)
    sel = rng.permutation(sel)[: args.max]
    ctrl = np.flatnonzero((real["outcome"] == 12) & (per >= args.min_period))
    ctrl = rng.permutation(ctrl)[: args.n_control]
    tasks = []
    for role, idxs in (("control", ctrl), ("captured", sel)):
        for k in idxs:
            i, r = int(real["system_index"][k]), int(real["realization"][k])
            if (i, r) in done:
                continue
            tasks.append(({kk: float(truth[kk][i]) for kk in keys}, i, r,
                          int(real["rng_seed_numpy"][k]), int(real["rng_seed_c_rand"][k]), role))
    print(f"{len(tasks)} forced orbit fits to run ({len(done)} done); workers={args.workers}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with log.open("a") as fh, ctx.Pool(args.workers, initializer=_init,
                                       initargs=([(1.0, 1.0)], float(attrs["ruwe_min"])),
                                       maxtasksperchild=50) as pool:
        for n, rec in enumerate(pool.imap_unordered(_orbit_task, tasks, chunksize=1), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(tasks)}  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("linear")
    a.add_argument("--variants", default=DEFAULT_VARIANTS)
    a.add_argument("--extra-realizations", type=int, default=20)
    a.add_argument("--extra-min-period", type=float, default=600.0)
    a.add_argument("--workers", type=int, default=1)
    a.set_defaults(func=cmd_linear)
    c = sub.add_parser("matched")
    c.add_argument("--visibility", default=str(PRIMARY / "output" / "gate390" / "gaia_source_visibility.json"))
    c.add_argument("--g-edges", default="-inf,11,12,13,14,15,inf")
    c.add_argument("--extra-realizations", type=int, default=20)
    c.add_argument("--extra-min-period", type=float, default=600.0)
    c.add_argument("--workers", type=int, default=1)
    c.set_defaults(func=cmd_matched)
    b = sub.add_parser("orbit")
    b.add_argument("--min-period", type=float, default=600.0)
    b.add_argument("--max", type=int, default=400)
    b.add_argument("--n-control", type=int, default=20)
    b.add_argument("--workers", type=int, default=2)
    b.set_defaults(func=cmd_orbit)
    args = p.parse_args()
    if args.workers > 2:
        raise SystemExit("at most 2 workers on this laptop while #391 runs")
    args.func(args)


if __name__ == "__main__":
    main()
