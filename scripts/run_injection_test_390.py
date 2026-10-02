#!/usr/bin/env python3
"""Step 1a injection test (#390): re-inject published DR3 orbits through gaiamock_mod.

Subcommands, run in order (each is resumable / idempotent):

``build-published``
    Read the uncut DR3 NSS snapshot (G, RUWE) and the NSS enrichment snapshot
    (orbit, ``significance``, ``corr_vec`` / ``bit_index``), keep the
    ``elbadry2024_comparison_nss_solution_types`` (Orbital, AstroSpectroSB1), drop
    duplicate ``(source_id, nss_solution_type)`` rows from cross-match fan-out
    (#221), and write one compact HDF5 of the published solutions.
``select``
    Stratified sample (quantile bins of G, log P, log a0/sigma_a0) per solution
    type. With the same ``--sample-seed`` a smaller sample is a subset of a larger
    one, so pilot realizations are reused by the scale-up.
``fetch-vis``
    Gaia archive query of ``gaia_source.visibility_periods_used`` /
    ``astrometric_matched_transits`` for the selected source_ids (public DR3 data).
``run``
    Inject ``--n-realizations`` noise realizations per selected system with
    ``--workers`` processes (BLAS pinned to one thread each). Results are appended
    to a JSONL log keyed by (source_id, realization); a restart skips finished
    pairs.
``assemble``
    Join truth + recovered + cuts + acceptance into the final HDF5 table and
    record its SHA-256.

Plots and the report are made by ``scripts/plot_injection_test_390.py``.

Example::

    PY=.venv/bin/python
    $PY scripts/run_injection_test_390.py build-published
    $PY scripts/run_injection_test_390.py select --n-orbital 200 --n-astrospectro 50 --tag pilot
    $PY scripts/run_injection_test_390.py fetch-vis --tag pilot
    $PY scripts/run_injection_test_390.py run --tag pilot --n-realizations 5 --workers 8
    $PY scripts/run_injection_test_390.py assemble --tag pilot
"""

from __future__ import annotations

import os

# BLAS / OpenMP threads: one per worker. Must precede the numpy import (#390).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import injection_test as it  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

#: Default output root (gitignored). Large artifacts live here, never in git.
DEFAULT_OUT = REPO / "output" / "gate390"
DEFAULT_SNAPSHOT_META = (
    REPO / "data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml"
)

#: Published numeric columns carried into the cache (enrichment names).
_ENRICH_NUMERIC = (
    "ra",
    "dec",
    "ra_error",
    "dec_error",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "pmra_error",
    "pmdec_error",
    "period",
    "period_error",
    "t_periastron",
    "t_periastron_error",
    "eccentricity",
    "eccentricity_error",
    "a_thiele_innes",
    "a_thiele_innes_error",
    "b_thiele_innes",
    "b_thiele_innes_error",
    "f_thiele_innes",
    "f_thiele_innes_error",
    "g_thiele_innes",
    "g_thiele_innes_error",
    "c_thiele_innes",
    "c_thiele_innes_error",
    "h_thiele_innes",
    "h_thiele_innes_error",
    "center_of_mass_velocity",
    "center_of_mass_velocity_error",
    "goodness_of_fit",
    "significance",
)
#: Longest corr_vec in the cached types (AstroSpectroSB1: 15 params -> 105).
_CORR_VEC_MAX = 105


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip(x: Any) -> str:
    return str(x).strip().strip('"')


def _rss_gib() -> float:
    """Peak RSS of this process (macOS reports bytes, Linux KiB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


# ---------------------------------------------------------------------------
# build-published
# ---------------------------------------------------------------------------


def cmd_build_published(args: argparse.Namespace) -> None:
    from darkhunter_pop.data_acquisition import load_gaia_snapshot

    cfg = load_config()
    types = tuple(
        cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types
    )
    enrich_dir = cfg.active_dr().nss_enrichment_snapshot
    enrich_meta = REPO / "data/dr3/gaia_snapshots" / str(enrich_dir) / "meta.yaml"
    out = Path(args.out) / "published_orbits.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"loading uncut snapshot {args.snapshot_meta}", flush=True)
    smeta, snap = load_gaia_snapshot(Path(args.snapshot_meta), verify_checksum=False)
    sol = np.array([_strip(x) for x in snap["nss_solution_type"]])
    keep = np.isin(sol, types)
    sid = np.asarray(snap["source_id"], dtype=np.int64)[keep]
    sol_k = sol[keep]
    g_mag = np.asarray(np.ma.filled(snap["g_mag"][keep], np.nan), dtype=np.float64)
    ruwe = np.asarray(np.ma.filled(snap["ruwe"][keep], np.nan), dtype=np.float64)
    del snap
    snap_rows = len(sid)
    keyed: dict[tuple[int, str], tuple[float, float]] = {}
    n_dup = 0
    n_dup_conflict = 0
    for i in range(snap_rows):
        k = (int(sid[i]), str(sol_k[i]))
        if k in keyed:
            n_dup += 1
            if keyed[k] != (g_mag[i], ruwe[i]) and not (
                np.isnan(keyed[k][0]) and np.isnan(g_mag[i])
            ):
                n_dup_conflict += 1
            continue
        keyed[k] = (float(g_mag[i]), float(ruwe[i]))
    print(
        f"  {snap_rows} snapshot rows of {types}; {n_dup} duplicate keys dropped "
        f"({n_dup_conflict} with differing G/RUWE); {time.time()-t0:.0f}s",
        flush=True,
    )

    print(f"loading NSS enrichment {enrich_meta}", flush=True)
    emeta, enr = load_gaia_snapshot(enrich_meta, verify_checksum=False)
    esol = np.array([_strip(x) for x in enr["nss_solution_type"]])
    ekeep = np.flatnonzero(np.isin(esol, types))
    sid_col = "source_id" if "source_id" in enr.colnames else "SOURCE_ID"
    esid = np.asarray(enr[sid_col], dtype=np.int64)
    rows_out: dict[str, list[Any]] = {c: [] for c in _ENRICH_NUMERIC}
    out_sid: list[int] = []
    out_sol: list[str] = []
    out_g: list[float] = []
    out_ruwe: list[float] = []
    corr = np.full((len(ekeep), _CORR_VEC_MAX), np.nan, dtype=np.float32)
    bit = np.zeros(len(ekeep), dtype=np.int64)
    n_unmatched = 0
    seen: set[tuple[int, str]] = set()
    cols = {c: np.asarray(np.ma.filled(enr[c][ekeep].astype(float), np.nan)) for c in _ENRICH_NUMERIC}
    corr_col = enr["corr_vec"][ekeep]
    bit_col = np.asarray(np.ma.filled(enr["bit_index"][ekeep], 0), dtype=np.int64)
    j = 0
    for n, i in enumerate(ekeep):
        k = (int(esid[i]), str(esol[i]))
        if k not in keyed:
            n_unmatched += 1
            continue
        if k in seen:
            continue
        seen.add(k)
        out_sid.append(k[0])
        out_sol.append(k[1])
        out_g.append(keyed[k][0])
        out_ruwe.append(keyed[k][1])
        for c in _ENRICH_NUMERIC:
            rows_out[c].append(float(cols[c][n]))
        raw = corr_col[n]
        arr = np.asarray(np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else raw, dtype=np.float64).ravel()
        corr[j, : min(len(arr), _CORR_VEC_MAX)] = arr[:_CORR_VEC_MAX]
        bit[j] = bit_col[n]
        j += 1
    del enr
    corr = corr[:j]
    bit = bit[:j]
    print(
        f"  joined {j} solutions ({n_unmatched} enrichment rows without a snapshot row); "
        f"{time.time()-t0:.0f}s; peak RSS {_rss_gib():.2f} GiB",
        flush=True,
    )
    with h5py.File(out, "w") as h:
        g = h.create_group("published")
        g.create_dataset("source_id", data=np.asarray(out_sid, dtype=np.int64))
        g.create_dataset("nss_solution_type", data=np.asarray(out_sol, dtype="S32"))
        g.create_dataset("g_mag", data=np.asarray(out_g))
        g.create_dataset("ruwe", data=np.asarray(out_ruwe))
        for c in _ENRICH_NUMERIC:
            g.create_dataset(c, data=np.asarray(rows_out[c], dtype=np.float64))
        g.create_dataset("corr_vec", data=corr, compression="gzip")
        g.create_dataset("bit_index", data=bit)
        h.attrs["snapshot_id"] = smeta.snapshot_id
        h.attrs["snapshot_checksum"] = smeta.checksum
        h.attrs["enrichment_snapshot_id"] = emeta.snapshot_id
        h.attrs["enrichment_checksum"] = emeta.checksum
        h.attrs["nss_solution_types"] = json.dumps(list(types))
        h.attrs["n_snapshot_rows"] = snap_rows
        h.attrs["n_duplicate_keys_dropped"] = n_dup
        h.attrs["n_duplicate_keys_conflicting"] = n_dup_conflict
        h.attrs["n_enrichment_unmatched"] = n_unmatched
        h.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h.attrs["git_commit"] = _git_commit()
    print(f"wrote {out}  sha256={_sha256(out)}", flush=True)


def _load_published(out_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(out_dir / "published_orbits.h5", "r") as h:
        g = h["published"]
        data = {k: g[k][()] for k in g}
        attrs = dict(h.attrs)
    data["nss_solution_type"] = np.array([s.decode() for s in data["nss_solution_type"]])
    return data, attrs


def _row_mapping(pub: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        k: pub[k][i] for k in pub if k not in ("corr_vec", "nss_solution_type")
    }
    row["nss_solution_type"] = str(pub["nss_solution_type"][i])
    cv = np.asarray(pub["corr_vec"][i], dtype=np.float64)
    row["corr_vec"] = cv[np.isfinite(cv)]
    row["bit_index"] = int(pub["bit_index"][i])
    row["source_id"] = int(pub["source_id"][i])
    return row


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    pub, _ = _load_published(out_dir)
    sel_idx: list[np.ndarray] = []
    sel_w: list[np.ndarray] = []
    edges_all: dict[str, Any] = {}
    for sol_type, n_req in (("Orbital", args.n_orbital), ("AstroSpectroSB1", args.n_astrospectro)):
        if n_req <= 0:
            continue
        mask = pub["nss_solution_type"] == sol_type
        rows = np.flatnonzero(mask)
        strata = {
            "g_mag": pub["g_mag"][rows],
            "log10_period": np.log10(pub["period"][rows]),
            "log10_significance": np.log10(pub["significance"][rows]),
        }
        idx, w, edges = it.stratified_sample_indices(
            strata, n_bins=args.n_bins, n_total=n_req, seed=args.sample_seed
        )
        sel_idx.append(rows[idx])
        sel_w.append(w)
        edges_all[sol_type] = {k: v.tolist() for k, v in edges.items()}
        print(f"{sol_type}: {mask.sum()} published, {len(idx)} selected", flush=True)
    idx = np.concatenate(sel_idx)
    w = np.concatenate(sel_w)
    path = out_dir / f"sample_{args.tag}.json"
    path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "sample_seed": args.sample_seed,
                "n_bins": args.n_bins,
                "n_orbital_requested": args.n_orbital,
                "n_astrospectro_requested": args.n_astrospectro,
                "rows": idx.tolist(),
                "source_id": [int(pub["source_id"][i]) for i in idx],
                "nss_solution_type": [str(pub["nss_solution_type"][i]) for i in idx],
                "weight": w.tolist(),
                "edges": edges_all,
            },
            indent=1,
        )
    )
    print(f"wrote {path} ({len(idx)} systems)")


def _load_sample(out_dir: Path, tag: str) -> dict[str, Any]:
    return json.loads((out_dir / f"sample_{tag}.json").read_text())


# ---------------------------------------------------------------------------
# fetch-vis
# ---------------------------------------------------------------------------


def cmd_fetch_vis(args: argparse.Namespace) -> None:
    from astroquery.gaia import Gaia

    out_dir = Path(args.out)
    sample = _load_sample(out_dir, args.tag)
    path = out_dir / "gaia_source_visibility.json"
    have: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"rows": {}}
    todo = sorted({int(s) for s in sample["source_id"]} - {int(k) for k in have["rows"]})
    print(f"{len(todo)} source_ids to fetch", flush=True)
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        q = (
            "SELECT source_id, visibility_periods_used, astrometric_matched_transits, "
            "astrometric_n_good_obs_al, ecl_lat FROM gaiadr3.gaia_source WHERE source_id IN ("
            + ",".join(str(s) for s in chunk)
            + ")"
        )
        tab = Gaia.launch_job(q).get_results()
        for r in tab:
            have["rows"][str(int(r["source_id"]))] = {
                "visibility_periods_used": int(r["visibility_periods_used"]),
                "astrometric_matched_transits": int(r["astrometric_matched_transits"]),
                "astrometric_n_good_obs_al": int(r["astrometric_n_good_obs_al"]),
                "ecl_lat": float(r["ecl_lat"]),
            }
        have["query_utc"] = datetime.now(timezone.utc).isoformat()
        have["table"] = "gaiadr3.gaia_source"
        path.write_text(json.dumps(have))
        print(f"  fetched {min(start+args.chunk, len(todo))}/{len(todo)}", flush=True)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _worker_init(base_seed: int, cuts_json: str, data_release: str, ruwe_min: float, skip_acc: bool) -> None:
    from darkhunter_pop.config_schema import OrbitalSolutionCutsConfig
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    gm = import_gaiamock_mod()
    _W.update(
        gm=gm,
        c_funcs=gm.read_in_C_functions(),
        base_seed=base_seed,
        cuts=OrbitalSolutionCutsConfig.model_validate_json(cuts_json),
        data_release=data_release,
        ruwe_min=ruwe_min,
        skip_acc=skip_acc,
    )


def _worker_task(task: tuple[dict[str, Any], list[int]]) -> list[dict[str, Any]]:
    row, realizations = task
    gm = _W["gm"]
    truth = it.published_truth_from_row(row, gm)
    out = []
    for r in realizations:
        c0, w0 = time.process_time(), time.time()
        try:
            rec = it.inject_one_realization(
                gm,
                _W["c_funcs"],
                truth,
                realization=r,
                base_seed=_W["base_seed"],
                cuts=_W["cuts"],
                data_release=_W["data_release"],
                ruwe_min=_W["ruwe_min"],
                skip_acceleration=_W["skip_acc"],
            )
            rec["error"] = ""
        except Exception as exc:  # recorded, never silently dropped
            rec = {"source_id": truth.source_id, "realization": r, "error": repr(exc)}
        rec["cpu_s"] = time.process_time() - c0
        rec["wall_s"] = time.time() - w0
        rec["worker_rss_gib"] = _rss_gib()
        out.append(rec)
    return out


def _done_pairs(log: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if log.exists():
        with log.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a kill; rerun that pair
                done.add((int(d["source_id"]), int(d["realization"])))
    return done


def _disk_free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config()
    out_dir = Path(args.out)
    pub, _ = _load_published(out_dir)
    sample = _load_sample(out_dir, args.tag)
    log = out_dir / "realizations.jsonl"
    done = _done_pairs(log)
    pop = cfg.selection_function_astrometric.mock_population
    base_seed = int(args.base_seed if args.base_seed is not None else pop.random_seed)
    cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
    tasks = []
    for i in sample["rows"]:
        row = _row_mapping(pub, int(i))
        todo = [r for r in range(args.n_realizations) if (row["source_id"], r) not in done]
        if todo:
            tasks.append((row, todo))
    if args.limit:
        tasks = tasks[: args.limit]
    n_pairs = sum(len(t[1]) for t in tasks)
    print(
        f"{len(tasks)} systems / {n_pairs} realizations to run "
        f"({len(done)} pairs already in {log.name}); workers={args.workers}; base_seed={base_seed}",
        flush=True,
    )
    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("base_seed") != base_seed:
        raise SystemExit(f"base_seed {base_seed} != logged {meta.get('base_seed')}; refusing to mix")
    meta.update(
        base_seed=base_seed,
        seed_scheme=it.INJECTION_RNG_SEED_SCHEME,
        data_release=cfg.active_dr_mode.value,
        ruwe_min=pop.ruwe_min,
        skip_acceleration=pop.skip_acceleration,
        cuts=json.loads(cuts.model_dump_json()),
        git_commit=_git_commit(),
    )
    meta_path.write_text(json.dumps(meta, indent=1))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    n_done = 0
    with log.open("a") as fh, ctx.Pool(
        args.workers,
        initializer=_worker_init,
        initargs=(base_seed, cuts.model_dump_json(), cfg.active_dr_mode.value, pop.ruwe_min, pop.skip_acceleration),
        maxtasksperchild=args.max_tasks_per_child,
    ) as pool:
        for recs in pool.imap_unordered(_worker_task, tasks, chunksize=1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += len(recs)
            if n_done % args.progress_every < len(recs):
                el = time.time() - t0
                free = _disk_free_gib(out_dir)
                print(
                    f"  {n_done}/{n_pairs} realizations, {el/60:.1f} min, "
                    f"{el/max(n_done,1):.2f} wall-s/realization, disk free {free:.1f} GiB",
                    flush=True,
                )
                if free < args.min_free_gib:
                    pool.terminate()
                    raise SystemExit(f"disk free {free:.1f} GiB < {args.min_free_gib}; stopping")
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def cmd_assemble(args: argparse.Namespace) -> None:
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod, read_versions

    out_dir = Path(args.out)
    gm = import_gaiamock_mod()
    pub, pattrs = _load_published(out_dir)
    sample = _load_sample(out_dir, args.tag)
    meta = json.loads((out_dir / "run_meta.json").read_text())
    vis_path = out_dir / "gaia_source_visibility.json"
    vis = json.loads(vis_path.read_text())["rows"] if vis_path.exists() else {}
    wanted = {int(s) for s in sample["source_id"]}
    recs: dict[tuple[int, int], dict[str, Any]] = {}
    with (out_dir / "realizations.jsonl").open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(d["source_id"]) in wanted and int(d["realization"]) < args.n_realizations:
                recs[(int(d["source_id"]), int(d["realization"]))] = d
    # Truth table, one row per system.
    truth_cols: dict[str, list[float]] = {c: [] for c in it.TRUTH_COLUMNS}
    extra = {k: [] for k in ("visibility_periods_used", "astrometric_matched_transits", "ecl_lat")}
    roundtrip: list[float] = []
    for i, sid_expected in zip(sample["rows"], sample["source_id"]):
        row = _row_mapping(pub, int(i))
        if row["source_id"] != int(sid_expected):
            raise SystemExit(f"published cache row {i} is {row['source_id']}, sample says {sid_expected}")
        t = it.published_truth_from_row(row, gm)
        for c in it.TRUTH_COLUMNS:
            truth_cols[c].append(float(t.values[c]))
        roundtrip.append(it.thiele_innes_roundtrip_error(t.values))
        v = vis.get(str(row["source_id"]), {})
        for k in extra:
            extra[k].append(float(v.get(k, np.nan)))
    sids = np.asarray(sample["source_id"], dtype=np.int64)
    sid_to_sys = {int(s): k for k, s in enumerate(sids)}
    keys = sorted(recs, key=lambda k: (sid_to_sys[k[0]], k[1]))
    out = out_dir / f"injection_test_{args.tag}.h5"
    with h5py.File(out, "w") as h:
        g = h.create_group("systems")
        g.create_dataset("source_id", data=sids)
        g.create_dataset("nss_solution_type", data=np.asarray(sample["nss_solution_type"], dtype="S32"))
        g.create_dataset("stratum_weight", data=np.asarray(sample["weight"]))
        tg = g.create_group("truth")
        for c, vals in truth_cols.items():
            tg.create_dataset(c, data=np.asarray(vals))
        for k, vals in extra.items():
            tg.create_dataset(f"published_{k}", data=np.asarray(vals))
        tg.create_dataset("thiele_innes_roundtrip_max_abs_mas", data=np.asarray(roundtrip))
        tg.create_dataset(
            "outside_fit_bounds",
            data=np.asarray(
                [it.outside_fit_bounds(p, e) for p, e in zip(truth_cols["period"], truth_cols["eccentricity"])]
            ),
        )
        rg = h.create_group("realizations")
        rg.create_dataset("system_index", data=np.asarray([sid_to_sys[k[0]] for k in keys], dtype=np.int64))
        rg.create_dataset("source_id", data=np.asarray([k[0] for k in keys], dtype=np.int64))
        rg.create_dataset("realization", data=np.asarray([k[1] for k in keys], dtype=np.int64))
        for c in ("outcome", "rng_seed_numpy", "rng_seed_c_rand"):
            rg.create_dataset(c, data=np.asarray([int(recs[k].get(c, -1)) for k in keys], dtype=np.int64))
        rg.create_dataset("accepted", data=np.asarray([bool(recs[k].get("accepted", False)) for k in keys]))
        rg.create_dataset("error", data=np.asarray([recs[k].get("error", "") for k in keys], dtype="S200"))
        rec_g = rg.create_group("recovered")
        for c in it.RECOVERED_COLUMNS:
            rec_g.create_dataset(c, data=np.asarray([float(recs[k].get(c, np.nan)) for k in keys]))
        cut_g = rg.create_group("cuts")
        for c in it.CUT_FLAG_NAMES:
            cut_g.create_dataset(c, data=np.asarray([bool(recs[k].get(c, False)) for k in keys]))
        for c in ("cpu_s", "wall_s", "worker_rss_gib"):
            rg.create_dataset(c, data=np.asarray([float(recs[k].get(c, np.nan)) for k in keys]))
        v = read_versions()
        h.attrs.update(
            {
                "issue": 390,
                "tag": args.tag,
                "n_realizations_per_system": args.n_realizations,
                "seed_scheme": meta["seed_scheme"],
                "base_seed": meta["base_seed"],
                "data_release": meta["data_release"],
                "ruwe_min": meta["ruwe_min"],
                "skip_acceleration": meta["skip_acceleration"],
                "orbital_solution_cuts": json.dumps(meta["cuts"]),
                "gaiamock_mod_release": v.gaiamock_mod_release,
                "gaiamock_mod_sha256": v.gaiamock_mod_sha256,
                "gaiamock_git_commit": v.gaiamock_git_commit,
                "snapshot_id": pattrs["snapshot_id"],
                "enrichment_snapshot_id": pattrs["enrichment_snapshot_id"],
                "sample_seed": sample["sample_seed"],
                "strata_edges": json.dumps(sample["edges"]),
                "visibility_query_utc": json.loads(vis_path.read_text()).get("query_utc", "") if vis_path.exists() else "",
                "git_commit": _git_commit(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "injection_function": "gaiamock_mod.predict_astrometry_binary_in_terms_of_a0",
                "fit_function": "gaiamock_mod.fit_full_astrometric_cascade",
            }
        )
    n_missing = len(sids) * args.n_realizations - len(keys)
    print(f"wrote {out}: {len(sids)} systems, {len(keys)} realizations ({n_missing} missing)")
    print(f"sha256 {_sha256(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (gitignored)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-published")
    p.add_argument("--snapshot-meta", default=str(DEFAULT_SNAPSHOT_META))
    p.set_defaults(func=cmd_build_published)

    p = sub.add_parser("select")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-orbital", type=int, default=200)
    p.add_argument("--n-astrospectro", type=int, default=50)
    p.add_argument("--n-bins", type=int, default=3, help="quantile bins per stratification variable")
    p.add_argument("--sample-seed", type=int, default=390)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("fetch-vis")
    p.add_argument("--tag", required=True)
    p.add_argument("--chunk", type=int, default=500)
    p.set_defaults(func=cmd_fetch_vis)

    p = sub.add_parser("run")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-realizations", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-seed", type=int, default=None, help="default: mock_population.random_seed")
    p.add_argument("--limit", type=int, default=0, help="run only the first N systems (timing)")
    p.add_argument("--max-tasks-per-child", type=int, default=50)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--min-free-gib", type=float, default=3.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("assemble")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-realizations", type=int, default=5)
    p.set_defaults(func=cmd_assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
