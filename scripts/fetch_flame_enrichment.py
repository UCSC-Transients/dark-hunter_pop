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

**Network timeout (#257 follow-up).** The installed astroquery (0.4.11)'s TAP+
client builds its HTTP(S) connection with plain ``http.client.HTTPConnection``/
``HTTPSConnection`` and never passes a ``timeout`` — confirmed by reading
``astroquery.utils.tap.conn.tapconn.ConnectionHandler.get_connection[_secure]``
— and neither ``Gaia``/``TapPlus``/``Tap`` nor
``GaiaClass.launch_job_async``/``TapPlus.launch_job_async`` expose any
``timeout`` parameter or class attribute (checked via
``inspect.signature``/``dir`` against the installed version). Per Python's
``http.client`` / ``socket`` docs, a connection built with no explicit
``timeout`` falls back to ``socket.getdefaulttimeout()`` (``None`` = block
forever), which is exactly the hang observed against the live archive: no
exception, no job id, indefinite. ``_bounded_socket_timeout`` below sets that
process-wide default for the duration of one archive call and restores it
after — the only lever astroquery's own code actually reads — rather than
guessing at a nonexistent keyword argument.

Example::

    .venv/bin/python scripts/fetch_flame_enrichment.py
    .venv/bin/python scripts/fetch_flame_enrichment.py --poll-job JOBID
    .venv/bin/python scripts/fetch_flame_enrichment.py --timeout 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml
from astropy.table import Table

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.data_acquisition import build_flame_enrichment_adql

# Default per-call network timeout (seconds). A single TAP GET/POST against a
# healthy archive completes in well under a minute; this only needs to be
# long enough to not false-positive on ordinary latency, not to accommodate
# the *job's* own compute time (that is polled separately, in its own calls).
DEFAULT_NETWORK_TIMEOUT_S = 120.0


class GaiaArchiveTimeoutError(TimeoutError):
    """Raised when a single Gaia TAP+ network call exceeds the configured
    timeout — distinguishes a genuinely stalled connection (this) from a
    legitimately slow-to-compute async job (tracked via job phase polling,
    not this exception).
    """


@contextmanager
def _bounded_socket_timeout(seconds: float) -> Iterator[None]:
    """Bound every socket astroquery's TAP+ client opens for the duration of
    the ``with`` block to ``seconds``, restoring the prior process-wide
    default on exit. See the module docstring for why this is the mechanism
    astroquery's own connection code actually honors.

    Raises :class:`GaiaArchiveTimeoutError` (chained from the underlying
    ``socket.timeout``/``TimeoutError``/``OSError``) rather than letting a
    stalled connection hang the process indefinitely or bubble up an opaque
    low-level socket exception.
    """
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise GaiaArchiveTimeoutError(
            f"Gaia archive network call exceeded {seconds:.0f}s (or the "
            "connection dropped) — treat as a stalled/unavailable archive "
            "connection, not a legitimately slow async job (issue #184)."
        ) from exc
    finally:
        socket.setdefaulttimeout(previous)


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


def launch(*, timeout_s: float = DEFAULT_NETWORK_TIMEOUT_S) -> int:
    from astroquery.gaia import Gaia

    adql = build_flame_enrichment_adql()
    out = _out_dir()
    print(
        f"fetch_flame_enrichment: launching async job → {out} "
        f"(network timeout={timeout_s:.0f}s)",
        flush=True,
    )
    Gaia.ROW_LIMIT = -1
    try:
        with _bounded_socket_timeout(timeout_s):
            job = Gaia.launch_job_async(adql, dump_to_file=False, verbose=True)
            phase = job.get_phase()
    except GaiaArchiveTimeoutError as exc:
        print(f"fetch_flame_enrichment: {exc}", file=sys.stderr, flush=True)
        return 1
    meta = {
        "jobid": job.jobid,
        "adql": adql,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
    }
    (out / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fetch_flame_enrichment: jobid={job.jobid} phase={meta['phase']}", flush=True)
    return 0


def poll_and_save(jobid: str, *, timeout_s: float = DEFAULT_NETWORK_TIMEOUT_S) -> int:
    from astroquery.gaia import Gaia

    out = _out_dir()
    print(f"fetch_flame_enrichment: polling job {jobid}", flush=True)
    try:
        with _bounded_socket_timeout(timeout_s):
            job = Gaia.load_async_job(jobid=jobid, verbose=True)
            phase = job.get_phase(update=True)
    except GaiaArchiveTimeoutError as exc:
        print(f"fetch_flame_enrichment: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"fetch_flame_enrichment: phase={phase}", flush=True)
    if phase not in ("COMPLETED", "ERROR", "ABORTED"):
        print("fetch_flame_enrichment: still running; re-poll later", flush=True)
        return 2
    if phase != "COMPLETED":
        print(f"fetch_flame_enrichment: job failed phase={phase}", file=sys.stderr)
        return 1
    try:
        with _bounded_socket_timeout(timeout_s):
            table = job.get_results()
    except GaiaArchiveTimeoutError as exc:
        print(f"fetch_flame_enrichment: {exc}", file=sys.stderr, flush=True)
        return 1
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_NETWORK_TIMEOUT_S,
        help=(
            "Per-network-call timeout in seconds (default "
            f"{DEFAULT_NETWORK_TIMEOUT_S:.0f}); a stalled/dead connection "
            "fails fast with a clear error instead of hanging indefinitely."
        ),
    )
    args = parser.parse_args(argv)
    if args.poll_job:
        return poll_and_save(args.poll_job, timeout_s=args.timeout)
    return launch(timeout_s=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
