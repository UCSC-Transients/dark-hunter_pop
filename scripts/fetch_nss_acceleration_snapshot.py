#!/usr/bin/env python3
"""Snapshot ``gaiadr3.nss_acceleration_astro`` for the #391 / #402 orbit:acceleration comparison.

Ryan, 2026-10-09 (#402). Fetches the Acceleration7 / Acceleration9 solutions with the columns the
real orbit sample's mirror filters need: the NSS solution's parallax (as the orbit snapshot uses
the NSS parallax), significance and goodness_of_fit, the ``gaia_source`` photometry and Halbwachs
(b)/(c) input columns, and the ``astrophysical_parameters`` atmosphere columns (MP-Q5). The result is
written as ``table.h5`` + ``meta.yaml`` (ADQL, date, sha256, row counts) under
``<data_root>/dr3/gaia_snapshots/<timestamp>_nss_acceleration_astro/``.

The archive is flaky, and astroquery's job handling hung on the joined query (2026-10-09). The
query is therefore sent as direct TAP ``sync`` requests (CSV) over ``--chunks`` source_id ranges.
Each chunk is retried with backoff and checked against its own ``COUNT(*)``, and the total against
the global count: the snapshot is refused unless every count agrees.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml

NSS_COLUMNS = ("source_id", "nss_solution_type", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec",
               "significance", "goodness_of_fit")
GS_COLUMNS = (("phot_g_mean_mag", "g_mag"), ("phot_bp_mean_mag", "bp_mag"), ("phot_rp_mean_mag", "rp_mag"),
              ("phot_g_mean_mag", "phot_g_mean_mag"), ("bp_rp", "bp_rp"), ("phot_bp_rp_excess_factor", "phot_bp_rp_excess_factor"),
              ("ipd_frac_multi_peak", "ipd_frac_multi_peak"), ("ipd_gof_harmonic_amplitude", "ipd_gof_harmonic_amplitude"),
              ("ruwe", "ruwe"), ("visibility_periods_used", "visibility_periods_used"))
AP_COLUMNS = ("teff_msc1", "teff_msc1_upper", "teff_msc1_lower", "logg_msc1", "logg_msc1_upper", "logg_msc1_lower",
              "mh_msc", "mh_msc_upper", "mh_msc_lower", "teff_gspphot", "teff_gspphot_upper", "teff_gspphot_lower",
              "logg_gspphot", "logg_gspphot_upper", "logg_gspphot_lower", "mh_gspphot", "mh_gspphot_upper", "mh_gspphot_lower")
TYPES = ("Acceleration7", "Acceleration9")


SOURCE_ID_MAX = 6917528997577384320  # largest possible DR3 source_id (HEALPix level-12 index 50331647 << 35 | ...)


def build_adql(lo: int | None = None, hi: int | None = None) -> str:
    cols = [f"nss.{c}" for c in NSS_COLUMNS] + [f"gs.{a} AS {b}" for a, b in GS_COLUMNS] + [f"ap.{c}" for c in AP_COLUMNS]
    types = ", ".join(f"'{t}'" for t in TYPES)
    return ("SELECT\n  " + ",\n  ".join(cols) + "\nFROM gaiadr3.nss_acceleration_astro AS nss\n"
            "JOIN gaiadr3.gaia_source AS gs ON nss.source_id = gs.source_id\n"
            "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON nss.source_id = ap.source_id\n"
            f"WHERE nss.nss_solution_type IN ({types})" + _range("nss", lo, hi))


def _range(alias: str, lo: int | None, hi: int | None) -> str:
    return "" if lo is None else f" AND {alias}.source_id >= {lo} AND {alias}.source_id < {hi}"


def count_adql(lo: int | None = None, hi: int | None = None) -> str:
    types = ", ".join(f"'{t}'" for t in TYPES)
    return (f"SELECT COUNT(*) AS n FROM gaiadr3.nss_acceleration_astro AS nss WHERE nss.nss_solution_type IN ({types})"
            + _range("nss", lo, hi))


TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"


def tap_sync(query: str, timeout_s: float = 600.0):
    """One ADQL query via the Gaia TAP sync endpoint, returned as an astropy Table (CSV transport)."""
    from astropy.io import ascii

    import requests
    from astropy.table import Table

    r = requests.post(TAP_SYNC, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}, timeout=timeout_s)
    r.raise_for_status()
    body = r.text
    if not body.strip() or body.lstrip().startswith("<"):
        raise RuntimeError(f"unexpected TAP response: {body[:300]!r}")
    return Table(ascii.read(body.splitlines(), format="csv"), masked=True)


def with_retries(fn, attempts: int, wait_s: float):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # archive flakiness: HTTP / timeout / job errors
            if i == attempts - 1:
                raise
            print(f"attempt {i + 1} failed: {exc!r}; retrying in {wait_s * 2 ** i:.0f} s", flush=True)
            time.sleep(wait_s * 2 ** i)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--chunks", type=int, default=36, help="split by source_id range; each chunk is count-checked")
    args = ap.parse_args(argv)
    adql = build_adql()
    print(adql, flush=True)
    when = dt.datetime.now(dt.timezone.utc)
    n_expected = int(with_retries(lambda: tap_sync(count_adql())["n"][0], args.attempts, args.wait))
    print(f"COUNT(*) = {n_expected}", flush=True)
    from astropy.table import vstack

    edges = np.linspace(0, SOURCE_ID_MAX + 1, args.chunks + 1).astype(np.int64) if args.chunks > 1 else None
    if edges is None:
        t = with_retries(lambda: tap_sync(adql), args.attempts, args.wait)
    else:
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            lo, hi = int(lo), int(hi)
            n_c = int(with_retries(lambda: tap_sync(count_adql(lo, hi))["n"][0], args.attempts, args.wait))
            for _ in range(args.attempts):
                part = with_retries(lambda: tap_sync(build_adql(lo, hi)), args.attempts, args.wait) if n_c else None
                if part is None:
                    break
                if len(part) == n_c:
                    break
                print(f"chunk [{lo}, {hi}) truncated: {len(part)} vs {n_c}; retrying", flush=True)
            else:
                raise SystemExit(f"chunk [{lo}, {hi}) truncated after {args.attempts} attempts")
            if part is None:
                continue
            part.meta = {}
            parts.append(part)
            print(f"chunk [{lo}, {hi}): {len(part)} rows", flush=True)
        t = vstack(parts, metadata_conflicts="silent")
    t.meta = {}
    if len(t) != n_expected:
        raise SystemExit(f"truncated: got {len(t)} rows, COUNT(*) = {n_expected}")
    sid = np.asarray(t["source_id"], np.int64)
    out = args.data_root / "dr3" / "gaia_snapshots" / f"{when.strftime('%Y%m%dT%H%M%SZ')}_nss_acceleration_astro"
    out.mkdir(parents=True, exist_ok=False)
    path = out / "table.h5"
    t["nss_solution_type"] = np.asarray(t["nss_solution_type"]).astype(str)
    t.write(path, format="hdf5", path="data", compression=True, serialize_meta=True)
    meta = {"snapshot_kind": "nss_acceleration_astro", "issue": [391, 402], "query_date": when.isoformat(), "adql": adql,
            "count_adql": count_adql(), "chunks": args.chunks,
            "chunk_edges_source_id": (None if edges is None else [int(e) for e in edges]), "n_rows": int(len(t)), "n_count_query": n_expected,
            "n_unique_source_id": int(np.unique(sid).size),
            "n_by_type": {k: int(np.sum(np.asarray(t["nss_solution_type"]) == k)) for k in TYPES},
            "columns": list(t.colnames), "table_h5_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
