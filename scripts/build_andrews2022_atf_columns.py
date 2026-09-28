"""Build the Andrews et al. (2022) ATF-notebook reproduction columns (#296).

Runs ``darkhunter_pop.andrews2022_atf`` — the PI-supplied selection notebook
(``data/reference/andrews2022_ATF_sample_selection.ipynb``) re-implemented —
over every parent-type NSS row (``Orbital`` for Andrews) and writes the
fingerprint-keyed sidecar that ``SampleSelectionRegistry`` merges into
``andrews2022`` / ``andrews2022_modified`` rows in **reproduction** mode only:

* pass 1 (``find_massive``) for every source, in parallel;
* pass 2 (``plot_system``) for the sources that survive the frozen chain
  through the pass-1 probability cut, decided by evaluating that chain itself.

Inputs are read, never modified: the raw ``nss_enrichment`` NSS table (errors,
``corr_vec``, GoF), the FLAME enrichment (``mass_flame``), and the uncut Gaia
snapshot (G / BP / RP, Apsis ``logg``). Output:
``<paths.data_root>/<sample_selection.reproduction_column_cache_dir>/<dr>/atf_notebook_<fingerprint>.h5``,
written atomically.

Example::

    .venv/bin/python scripts/build_andrews2022_atf_columns.py --workers 8
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the per-source work is tiny matrices.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import yaml  # noqa: E402

from darkhunter_pop.andrews2022_atf import (  # noqa: E402
    AtfSourceInputs,
    empty_pass2_columns,
    m2_threshold_msun,
    merge_reproduction_columns,
    notebook_float,
    procedure_fingerprint,
    reproduction_cache_dir,
    require_procedure,
    run_pass1,
    run_pass2,
    sidecar_path,
    static_columns,
    write_sidecar,
)
from darkhunter_pop.config_loader import load_config, repo_root  # noqa: E402
from darkhunter_pop.config_schema import (  # noqa: E402
    ReproductionProcedureSpec,
    SampleSelectionMode,
)
from darkhunter_pop.data_acquisition import load_gaia_snapshot  # noqa: E402
from darkhunter_pop.sample_selection import (  # noqa: E402
    CutOutcome,
    SampleSelection,
    SampleSelectionRegistry,
    parent_query_for_mode,
)

_UNCUT_SNAPSHOT_ID = "20260826T234425Z_3d3f740b080c"

_WORKER_PROC: ReproductionProcedureSpec | None = None
_WORKER_THRESHOLD: float | None = None


def _init_worker(proc_json: str, threshold: float) -> None:
    global _WORKER_PROC, _WORKER_THRESHOLD
    _WORKER_PROC = ReproductionProcedureSpec.model_validate_json(proc_json)
    _WORKER_THRESHOLD = float(threshold)


def _pass1_chunk(chunk: Sequence[AtfSourceInputs]) -> list[tuple[int, dict[str, Any]]]:
    assert _WORKER_PROC is not None and _WORKER_THRESHOLD is not None
    return [
        (item.source_id, run_pass1(item, _WORKER_PROC, m2_threshold=_WORKER_THRESHOLD))
        for item in chunk
    ]


def _pass2_chunk(chunk: Sequence[AtfSourceInputs]) -> list[tuple[int, dict[str, Any]]]:
    assert _WORKER_PROC is not None
    return [(item.source_id, run_pass2(item, _WORKER_PROC)) for item in chunk]


def _cell(value: Any) -> Any:
    """Table cell → Python scalar / array; masked → None."""
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.ndarray):
        return np.ma.filled(np.ma.asarray(value).astype(np.float64), np.nan)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _meta_checksum(meta_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    return {
        "meta_path": str(meta_path),
        "snapshot_id": raw.get("snapshot_id"),
        "checksum": raw.get("checksum"),
        "row_count": raw.get("row_count"),
    }


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _index_by_key(table: Any, *, id_col: str) -> dict[tuple[int, str], Any]:
    out: dict[tuple[int, str], Any] = {}
    for row in table:
        key = (int(row[id_col]), str(row["nss_solution_type"]).strip('"'))
        out.setdefault(key, row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample", default="andrews2022")
    parser.add_argument("--limit", type=int, default=None, help="smoke / debug cap")
    parser.add_argument("--chunk", type=int, default=250)
    args = parser.parse_args(argv)

    root = repo_root()
    cfg = load_config()
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    snaps = data_root / "dr3" / "gaia_snapshots"
    nss_meta = snaps / "nss_enrichment" / "meta.yaml"
    flame_meta = snaps / "flame_enrichment" / "meta.yaml"
    uncut_meta = snaps / _UNCUT_SNAPSHOT_ID / "meta.yaml"
    for path in (nss_meta, flame_meta, uncut_meta):
        if not path.is_file():
            print(f"missing input {path}", file=sys.stderr)
            return 1

    registry = SampleSelectionRegistry(cfg, repo=root)
    spec = registry.resolved(args.sample)
    proc = require_procedure(spec)
    threshold = m2_threshold_msun(spec)
    fingerprint = procedure_fingerprint(spec)
    out_path = sidecar_path(
        reproduction_cache_dir(cfg, repo=root), cfg.active_dr_mode.value, spec
    )
    if args.limit is not None:
        # A capped smoke run must never land where the registry would load it.
        out_path = out_path.with_name(f"{out_path.stem}.limit{args.limit}{out_path.suffix}")
    parent = parent_query_for_mode(spec, cfg.active_dr_mode)
    types = set(parent.solution_types)
    roundtrip = proc.covariance.float32_decimal_roundtrip
    p2 = proc.pass2
    print(
        f"sample={args.sample} fingerprint={fingerprint} types={sorted(types)} "
        f"pass1.n_draws={proc.pass1.n_draws} pass2.n_draws={p2.n_draws} "
        f"threshold={threshold} out={out_path}",
        flush=True,
    )

    t0 = time.time()
    _m, nss = load_gaia_snapshot(nss_meta, verify_checksum=False)
    id_col = "SOURCE_ID" if "SOURCE_ID" in nss.colnames else "source_id"
    _m, flame = load_gaia_snapshot(flame_meta, verify_checksum=False)
    flame_by_key = _index_by_key(flame, id_col="source_id")
    _m, uncut = load_gaia_snapshot(uncut_meta, verify_checksum=False)
    uncut_by_key = _index_by_key(uncut, id_col="source_id")
    print(f"inputs loaded in {time.time() - t0:.0f}s", flush=True)

    order = proc.covariance.parameter_order
    inputs: list[AtfSourceInputs] = []
    missing_aux = Counter[str]()
    for row in nss:
        sol = str(row["nss_solution_type"]).strip('"')
        if sol not in types:
            continue
        sid = int(row[id_col])
        means = np.array(
            [notebook_float(_cell(row[n]), float32_decimal_roundtrip=False) for n in order]
        )
        errors = np.array(
            [
                notebook_float(
                    _cell(row[f"{n}_error"]), float32_decimal_roundtrip=roundtrip
                )
                for n in order
            ]
        )
        raw_corr = _cell(row["corr_vec"])
        corr = np.asarray(raw_corr if raw_corr is not None else [], dtype=np.float64)
        corr = np.array(
            [notebook_float(v, float32_decimal_roundtrip=roundtrip) for v in corr.ravel()]
        )
        frow = flame_by_key.get((sid, sol))
        urow = uncut_by_key.get((sid, sol))
        if frow is None:
            missing_aux["flame_row"] += 1
        if urow is None:
            missing_aux["uncut_row"] += 1

        def aux(r: Any, col: str) -> float:
            if r is None or col not in r.colnames:
                return float("nan")
            return notebook_float(_cell(r[col]), float32_decimal_roundtrip=False)

        inputs.append(
            AtfSourceInputs(
                source_id=sid,
                means=means,
                errors=errors,
                corr_vec=corr,
                goodness_of_fit=notebook_float(
                    _cell(row["goodness_of_fit"]), float32_decimal_roundtrip=False
                ),
                mass_flame=aux(frow, p2.primary_mass.flame_column),
                logg=aux(urow, p2.giant_logg_column),
                g_mag=aux(urow, "g_mag"),
                bp_mag=aux(urow, "bp_mag"),
                rp_mag=aux(urow, "rp_mag"),
            )
        )
        if args.limit is not None and len(inputs) >= args.limit:
            break
    del nss, flame, uncut, flame_by_key, uncut_by_key
    print(
        f"parent rows={len(inputs)} missing_aux={dict(missing_aux)}",
        flush=True,
    )

    columns: dict[int, dict[str, Any]] = {
        item.source_id: {**static_columns(item, proc), **empty_pass2_columns()}
        for item in inputs
    }
    chunks = [inputs[i : i + args.chunk] for i in range(0, len(inputs), args.chunk)]
    proc_json = proc.model_dump_json()
    t1 = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(proc_json, threshold),
    ) as pool:
        for result in pool.map(_pass1_chunk, chunks):
            for sid, cols in result:
                columns[sid].update(cols)
            done += len(result)
            if done % (args.chunk * 40) < args.chunk or done == len(inputs):
                rate = done / max(time.time() - t1, 1e-6)
                print(
                    f"  pass1 {done}/{len(inputs)} {rate:.0f}/s "
                    f"eta_s={(len(inputs) - done) / max(rate, 1e-6):.0f}",
                    flush=True,
                )

        # Pass-1 survivors, decided by the frozen chain itself.
        selection = SampleSelection(
            spec, mode=SampleSelectionMode.REPRODUCTION, dr_mode=cfg.active_dr_mode
        )
        rows = merge_reproduction_columns(
            [
                {"source_id": item.source_id, "nss_solution_type": next(iter(types))}
                for item in inputs
            ],
            columns,
        )
        result = selection.evaluate(rows)
        cut_id = proc.pass1.probability_cut_id
        survivors = {
            sid
            for sid, outs in result.outcomes_by_source.items()
            if any(c == cut_id and o is CutOutcome.PASSED for c, o, _r in outs)
        }
        print(f"pass-1 survivors ({cut_id}) = {len(survivors)}", flush=True)
        todo = [item for item in inputs if item.source_id in survivors]
        chunks2 = [todo[i : i + 4] for i in range(0, len(todo), 4)]
        for result2 in pool.map(_pass2_chunk, chunks2):
            for sid, cols in result2:
                columns[sid].update(cols)
    print(f"MC done in {time.time() - t1:.0f}s", flush=True)

    fail = Counter(
        str(c.get("andrews_atf_covariance_failure"))
        for c in columns.values()
        if not c.get("andrews_atf_covariance_ok")
    )
    root_fail = sum(1 for c in columns.values() if c.get("andrews_atf_pass1_root_ok") is False)
    print(f"covariance failures: {dict(fail)}; pass-1 root failures: {root_fail}", flush=True)

    attrs = {
        "sample": args.sample,
        "procedure": proc.model_dump(mode="json"),
        "m2_threshold_msun": threshold,
        "inputs": {
            "nss_enrichment": _meta_checksum(nss_meta),
            "flame_enrichment": _meta_checksum(flame_meta),
            "uncut_snapshot": _meta_checksum(uncut_meta),
        },
        "n_rows": len(columns),
        "n_pass1_survivors": len(survivors),
        "limit": args.limit,
        "code_commit": _git_commit(root),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_sidecar(out_path, columns, fingerprint=fingerprint, attrs=attrs)
    print(f"wrote {out_path} ({len(columns)} rows) in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
