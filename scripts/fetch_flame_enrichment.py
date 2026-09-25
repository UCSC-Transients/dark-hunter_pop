"""Fetch Gaia Apsis FLAME mass columns (supplement, #257 — Andrews et al. 2022 M1).

Neither the Aug 2026 photometry snapshot nor the existing ``nss_enrichment``
job (frozen/completed — see ``CLAUDE.md`` Gotchas; do not widen or re-run it)
carries ``astrophysical_parameters.mass_flame``. Andrews et al. (2022)'s
``flame_or_uniform_draw`` primary-mass method (#230/#257) needs it. This
script is a second, independent, lightweight async fetch — join key plus
``mass_flame``/``mass_flame_upper``/``mass_flame_lower`` only — writing
``query.ecsv`` + ``meta.yaml`` under
``data/dr3/gaia_snapshots/flame_enrichment/``. Merges the same way
``fetch_nss_enrichment.py``'s output does
(``data_acquisition.merge_nss_enrichment_into_row`` /
``_enrichment_join_key``), via ``scripts/merge_flame_enrichment_into_cache.py``.

Example::

    .venv/bin/python scripts/fetch_flame_enrichment.py
    .venv/bin/python scripts/fetch_flame_enrichment.py --poll-job JOBID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from astropy.table import Table

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.data_acquisition import build_flame_enrichment_adql


def _out_dir() -> Path:
    path = repo_root() / "data" / "dr3" / "gaia_snapshots" / "flame_enrichment"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_checksum(path: Path) -> str:
    """SHA-256 of on-disk bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def launch() -> int:
    from astroquery.gaia import Gaia

    adql = build_flame_enrichment_adql()
    out = _out_dir()
    print(f"fetch_flame_enrichment: launching async job → {out}", flush=True)
    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job_async(adql, dump_to_file=False, verbose=True)
    meta = {
        "jobid": job.jobid,
        "adql": adql,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "phase": job.get_phase(),
    }
    (out / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fetch_flame_enrichment: jobid={job.jobid} phase={meta['phase']}", flush=True)
    return 0


def poll_and_save(jobid: str) -> int:
    from astroquery.gaia import Gaia

    out = _out_dir()
    print(f"fetch_flame_enrichment: polling job {jobid}", flush=True)
    job = Gaia.load_async_job(jobid=jobid, verbose=True)
    phase = job.get_phase(update=True)
    print(f"fetch_flame_enrichment: phase={phase}", flush=True)
    if phase not in ("COMPLETED", "ERROR", "ABORTED"):
        print("fetch_flame_enrichment: still running; re-poll later", flush=True)
        return 2
    if phase != "COMPLETED":
        print(f"fetch_flame_enrichment: job failed phase={phase}", file=sys.stderr)
        return 1
    table = job.get_results()
    if not isinstance(table, Table):
        raise TypeError(f"expected Table, got {type(table)!r}")
    result_path = out / "query.ecsv"
    print(f"fetch_flame_enrichment: writing {result_path} ({len(table)} rows)...", flush=True)
    table.write(result_path, format="ascii.ecsv", overwrite=True)
    checksum = _file_checksum(result_path)
    n_with_flame = int(sum(1 for v in table["mass_flame"] if v is not None and v == v))
    meta = {
        "snapshot_id": f"flame_enrichment_{jobid}",
        "query_date": datetime.now(timezone.utc).isoformat(),
        "adql": build_flame_enrichment_adql(),
        "checksum": checksum,
        "row_count": len(table),
        "n_with_mass_flame": n_with_flame,
        "result_path": str(result_path.relative_to(repo_root())),
        "jobid": jobid,
    }
    (out / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
    )
    print(
        f"fetch_flame_enrichment: wrote {result_path} n={len(table)} "
        f"n_with_mass_flame={n_with_flame} checksum={checksum[:12]}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poll-job",
        type=str,
        default=None,
        help="Poll an existing async job id and write query.ecsv",
    )
    args = parser.parse_args(argv)
    if args.poll_job:
        return poll_and_save(args.poll_job)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
