"""Attach Andrews MC columns onto enrich-only selection-parent cache (parallel).

Reads ``…+enrich/selection_parent_rows.h5``, runs full-covariance MC for
Orbital / AstroSpectroSB1 rows that have enrichment, writes
``…+enrich+mc{N}/selection_parent_rows.h5``.

Example::

    .venv/bin/python scripts/attach_mc_to_selection_cache.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from astropy.table import Table

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import (
    DRPathConfig,
    SpectroscopicMassFunctionConfig,
)
from darkhunter_pop.data_acquisition import (
    merge_nss_enrichment_into_row,
    table_row_to_candidate,
)
from darkhunter_pop.mc_mass_function import (
    ensemble_row_quantities,
    propagate_nss_solution,
)
from darkhunter_pop.sample_selection import (
    _read_selection_parent_cache,
    _selection_parent_cache_path,
    _write_selection_parent_cache,
    load_sample_selection_file,
)


def _mc_one(payload: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None]:
    (
        source_id,
        sol_type,
        enrich_cols,
        enrich_vals,
        m1,
        m2_thr,
        n_draws,
        seed_base,
        eig_rel,
        eig_abs,
        dr_dump,
        smf_dump,
    ) = payload
    dr = DRPathConfig.model_validate(dr_dump)
    smf = SpectroscopicMassFunctionConfig.model_validate(smf_dump)
    enrich = dict(zip(enrich_cols, enrich_vals, strict=True))
    mapping = merge_nss_enrichment_into_row(
        {"source_id": int(source_id), "nss_solution_type": str(sol_type)},
        enrich,
    )
    cand = table_row_to_candidate(mapping, dr, spectroscopic=smf)
    if cand.nss_solution is None:
        return int(source_id), None
    seed = int(seed_base) ^ (int(source_id) & 0x7FFFFFFF)
    try:
        draws = propagate_nss_solution(
            cand.nss_solution,
            m1_msun=float(m1),
            n_draws=int(n_draws),
            random_seed=seed,
            eig_rel_floor=float(eig_rel),
            eig_abs_floor=float(eig_abs),
            source_id=int(source_id),
        )
    except (ValueError, np.linalg.LinAlgError):
        return int(source_id), None
    q = ensemble_row_quantities(draws, m2_threshold_msun=float(m2_thr))
    if "sigma_m2_msun" in q:
        q["sigma_m2_astrometric_msun"] = q["sigma_m2_msun"]
        q["m2_msun_error"] = q["sigma_m2_msun"]
    return int(source_id), q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["Orbital"],
        help="nss_solution_type values to MC (default: Orbital for Andrews)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on MC jobs (smoke / debug)",
    )
    args = parser.parse_args(argv)

    photo_id = "20260826T234425Z_3d3f740b080c"
    enrich_meta = repo_root() / "data/dr3/gaia_snapshots/nss_enrichment/meta.yaml"
    enrich_cache = _selection_parent_cache_path(f"{photo_id}+enrich")
    if not enrich_meta.is_file():
        print("missing enrichment meta", file=sys.stderr)
        return 1
    if not enrich_cache.is_file():
        print(f"missing enrich cache {enrich_cache}", file=sys.stderr)
        return 1

    print(f"loading {enrich_cache} …", flush=True)
    rows = _read_selection_parent_cache(enrich_cache)
    cfg = load_config()
    mc = cfg.mc_mass_function
    andrews = load_sample_selection_file(
        repo_root() / "config/selections/andrews2022.yaml"
    )
    m1 = 1.0
    if (
        andrews.primary_mass is not None
        and andrews.primary_mass.value_msun is not None
    ):
        m1 = float(andrews.primary_mass.value_msun)
    m2_thr = 1.4
    for cut in andrews.cuts or []:
        if cut.id == "m2_probability":
            raw = cut.parameters.get("m2_threshold_msun")
            if isinstance(raw, (int, float)):
                m2_thr = float(raw)
            break

    print("loading enrichment ECSV for MC workers…", flush=True)
    enrich_table = Table.read(enrich_meta.parent / "query.ecsv", format="ascii.ecsv")
    enrich_cols = list(enrich_table.colnames)
    by_key: dict[tuple[int, str], tuple[Any, ...]] = {}
    for erow in enrich_table:
        sid = int(
            erow["SOURCE_ID"] if "SOURCE_ID" in enrich_cols else erow["source_id"]
        )
        sol = str(erow["nss_solution_type"])
        vals = tuple(
            v.tolist() if isinstance(v, np.ndarray) else v for v in (erow[c] for c in enrich_cols)
        )
        by_key[(sid, sol)] = vals

    types = set(args.types)
    jobs: list[tuple[Any, ...]] = []
    dr_dump = cfg.active_dr().model_dump(mode="json")
    smf_dump = cfg.spectroscopic_mass_function.model_dump(mode="json")
    for row in rows:
        sol = str(row.get("nss_solution_type", ""))
        if sol not in types:
            continue
        sid = int(row["source_id"])
        vals = by_key.get((sid, sol))
        if vals is None:
            continue
        jobs.append(
            (
                sid,
                sol,
                enrich_cols,
                vals,
                m1,
                m2_thr,
                int(mc.n_draws),
                int(mc.random_seed),
                float(mc.eig_rel_floor),
                float(mc.eig_abs_floor),
                dr_dump,
                smf_dump,
            )
        )
        if args.limit is not None and len(jobs) >= args.limit:
            break

    print(
        f"MC jobs={len(jobs)} workers={args.workers} n_draws={mc.n_draws}",
        flush=True,
    )
    mc_by_id: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    done = 0
    # Submit in waves so ~134k futures do not all sit in memory at once.
    wave = max(args.workers * 32, 256)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(jobs), wave):
            batch = jobs[start : start + wave]
            futures = [pool.submit(_mc_one, job) for job in batch]
            for fut in as_completed(futures):
                sid, q = fut.result()
                if q is not None:
                    mc_by_id[sid] = q
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(jobs) - done) / max(rate, 1e-6)
                    print(
                        f"  {done}/{len(jobs)} ok={len(mc_by_id)} "
                        f"{rate:.2f}/s eta_s={eta:.0f}",
                        flush=True,
                    )

    for row in rows:
        q = mc_by_id.get(int(row["source_id"]))
        if q:
            row.update(q)

    out_tag = f"{photo_id}+enrich+mc{int(mc.n_draws)}"
    out_path = _selection_parent_cache_path(out_tag)
    print(f"writing {out_path} …", flush=True)
    _write_selection_parent_cache(out_path, rows)
    n_p = sum(1 for r in rows if r.get("p_m2_above") is not None)
    print(
        f"done n_rows={len(rows)} with_p_m2={n_p} elapsed_s={time.time() - t0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
