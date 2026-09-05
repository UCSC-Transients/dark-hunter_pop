"""Fetch NSS corr_vec / K1 / significance columns (supplement to Aug 2026 snapshot).

The photometry-joined Gaia snapshot used by data_acquisition predates the
covariance + K1 ADQL columns. Literature reproduction needs those columns
without re-querying the full crossmatch. This script launches a lightweight
NSS-only async job and writes ``query.ecsv`` + ``meta.yaml`` under
``data/dr3/gaia_snapshots/nss_enrichment/``.

Example::

    .venv/bin/python scripts/fetch_nss_enrichment.py
    .venv/bin/python scripts/fetch_nss_enrichment.py --poll-job JOBID
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
from darkhunter_pop.data_acquisition import build_nss_enrichment_adql


def _out_dir() -> Path:
    path = repo_root() / "data" / "dr3" / "gaia_snapshots" / "nss_enrichment"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _table_checksum(table: Table) -> str:
    raw = table.to_pandas().to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def launch() -> int:
    from astroquery.gaia import Gaia

    adql = build_nss_enrichment_adql()
    out = _out_dir()
    print(f"fetch_nss_enrichment: launching async job → {out}", flush=True)
    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job_async(adql, dump_to_file=False, verbose=True)
    meta = {
        "jobid": job.jobid,
        "adql": adql,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "phase": job.get_phase(),
    }
    (out / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fetch_nss_enrichment: jobid={job.jobid} phase={meta['phase']}", flush=True)
    return 0


def poll_and_save(jobid: str) -> int:
    from astroquery.gaia import Gaia

    out = _out_dir()
    print(f"fetch_nss_enrichment: polling job {jobid}", flush=True)
    job = Gaia.load_async_job(jobid=jobid, verbose=True)
    phase = job.get_phase(update=True)
    print(f"fetch_nss_enrichment: phase={phase}", flush=True)
    if phase not in ("COMPLETED", "ERROR", "ABORTED"):
        print("fetch_nss_enrichment: still running; re-poll later", flush=True)
        return 2
    if phase != "COMPLETED":
        print(f"fetch_nss_enrichment: job failed phase={phase}", file=sys.stderr)
        return 1
    table = job.get_results()
    if not isinstance(table, Table):
        raise TypeError(f"expected Table, got {type(table)!r}")
    result_path = out / "query.ecsv"
    table.write(result_path, format="ascii.ecsv", overwrite=True)
    checksum = _table_checksum(table)
    meta = {
        "snapshot_id": f"nss_enrichment_{jobid}",
        "query_date": datetime.now(timezone.utc).isoformat(),
        "adql": build_nss_enrichment_adql(),
        "checksum": checksum,
        "row_count": len(table),
        "result_path": str(result_path.relative_to(repo_root())),
        "jobid": jobid,
    }
    (out / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
    )
    print(
        f"fetch_nss_enrichment: wrote {result_path} n={len(table)} "
        f"checksum={checksum[:12]}",
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
