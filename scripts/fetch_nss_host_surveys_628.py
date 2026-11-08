#!/usr/bin/env python3
"""#628: spectroscopic-survey RVs for the 168,065 DR3 Orbital + AstroSpectroSB1 hosts (approved by Ryan 2026-11-06).

* ``apogee_dr17``: SDSS SkyServer DR17 SQL, matched by ``gaiaedr3_source_id`` (DR3 = EDR3 source_id), in chunks of
  ``--chunk`` ids per GET request: ``apogeeStar`` (one row per star × telescope; targeting bits, nvisits, vhelio_avg,
  vscatter, verr) and ``apogeeVisit`` (one row per visit epoch, vhelio, vrelerr, jd) for the matched stars, plus the
  whole ``apogeeField`` table (the field footprint used by the follow-up selection function). The allStar/allVisit
  files (2–4 GB) are not downloaded.
* ``galah_dr4``: the GALAH DR4 ``allstar`` catalogue (Buder et al. 2025; 0.76 GB, under the 1 GB report limit) is
  streamed to a temporary file, the rows whose ``gaiadr3_source_id`` is a host are kept, and the full file is deleted;
  ``meta.yaml`` records its URL and sha256.

One snapshot per survey: ``<data-root>/dr3/external_catalogs/<UTC stamp>_<survey>_nss_hosts/`` with ECSV tables,
``SHA256SUMS`` and ``meta.yaml``. Public catalogues only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml

SKYSERVER = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
GALAH_URL = "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/galah_dr4_allstar_240705.fits"
MAX_BYTES = 1 << 30
STAR_COLS = ("apstar_id, apogee_id, telescope, location_id, field, ra, dec, apogee_target1, apogee_target2, apogee2_target1, "
             "apogee2_target2, apogee2_target3, apogee2_target4, survey, nvisits, snr, starflag, vhelio_avg, vscatter, verr, "
             "rv_teff, rv_logg, rv_feh, rv_flag, n_components, gaiaedr3_source_id")
VISIT_COLS = ("v.visit_id, v.apogee_id, v.telescope, v.field, v.plate, v.mjd, v.jd, v.dateobs, v.snr, v.starflag, v.vhelio, "
              "v.vrel, v.vrelerr, v.xcorr_vhelio, v.xcorr_vrelerr, v.rv_chi2, v.ccfwhm")


TEXT_COLS = {"apstar_id", "apogee_id", "telescope", "field", "survey", "file", "dateobs", "visit_id", "alt_id",
             "target_id", "location_id"}


def host_ids(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as h:
        return np.unique(h["published"]["source_id"][()].astype(np.int64))


def skyserver(sql: str, retries: int = 8) -> str:
    url = SKYSERVER + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    for k in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=300).read().decode()
        except Exception:  # noqa: BLE001 - transient service errors are retried, then raised
            if k == retries - 1:
                raise
            time.sleep(min(120, 10 * 2**k))
    raise RuntimeError("unreachable")


def csv_table(text: str):
    from astropy.table import Table

    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) <= 1:
        return None
    t = Table.read("\n".join(lines) + "\n", format="ascii.csv", fast_reader=False)
    for c in TEXT_COLS & set(t.colnames):  # chunks can infer a text column as int (e.g. all-numeric field names)
        t[c] = np.asarray(t[c]).astype(str)
    return t


def _snapshot(data_root: Path, name: str, date: dt.datetime) -> Path:
    out = data_root / "dr3" / "external_catalogs" / f"{date.strftime('%Y%m%dT%H%M%SZ')}_{name}_nss_hosts"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _finish(out: Path, tables: dict, meta: dict) -> None:
    sums = {}
    for name, t in tables.items():
        p = out / f"{name}.ecsv"
        t.write(p, format="ascii.ecsv")
        sums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    meta.update({"files_sha256": sums, "n_rows": {k: int(len(t)) for k, t in tables.items()}})
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {out}: " + ", ".join(f"{k} {len(t)}" for k, t in tables.items()))


def _vstack(tables: list):
    """vstack chunk tables, casting any column whose dtype kind differs between chunks to str."""
    from astropy.table import vstack

    cols = set().union(*(t.colnames for t in tables))
    for c in cols:
        kinds = {t[c].dtype.kind for t in tables if c in t.colnames}
        if len(kinds) > 1:
            for t in tables:
                if c in t.colnames:
                    t[c] = np.asarray(t[c]).astype(str)
    return vstack(tables)


def _cached(cache: Path, key: str, sql: str):
    """SkyServer result for ``sql``, cached as ``cache/key.csv`` so an interrupted fetch resumes."""
    f = cache / f"{key}.csv"
    if not f.exists():
        tmp = f.with_suffix(".tmp")
        tmp.write_text(skyserver(sql))
        tmp.rename(f)
    return csv_table(f.read_text())


def fetch_apogee(ids: np.ndarray, chunk: int, data_root: Path) -> None:
    from astropy.table import vstack

    date = dt.datetime.now(dt.timezone.utc)
    cache = data_root / "dr3" / "external_catalogs" / f"_partial_apogee_dr17_chunk{chunk}_n{ids.size}"
    cache.mkdir(parents=True, exist_ok=True)
    stars = []
    for lo in range(0, ids.size, chunk):
        lst = ",".join(str(int(s)) for s in ids[lo:lo + chunk])
        t = _cached(cache, f"star_{lo:07d}", f"SELECT {STAR_COLS} FROM apogeeStar WHERE gaiaedr3_source_id IN ({lst})")
        if t is not None:
            stars.append(t)
        if (lo // chunk) % 50 == 0:
            print(f"  stars: {lo + chunk}/{ids.size} ids, {sum(len(x) for x in stars)} rows", flush=True)
    star = _vstack(stars)
    apids = sorted(set(str(x) for x in star["apogee_id"]))
    visits = []
    vchunk = max(1, chunk // 3)  # quoted apogee_ids make a longer URL than numeric source_ids (SkyServer 404s above ~8 kB)
    for lo in range(0, len(apids), vchunk):
        lst = ",".join(f"'{a}'" for a in apids[lo:lo + vchunk])
        t = _cached(cache, f"visit_{lo:07d}", f"SELECT {VISIT_COLS} FROM apogeeVisit AS v WHERE v.apogee_id IN ({lst})")
        if t is not None:
            visits.append(t)
    visit = _vstack(visits)
    field = csv_table(skyserver("SELECT * FROM apogeeField"))
    out = _snapshot(data_root, "apogee_dr17", date)
    meta_extra = {"partial_cache": str(cache)}
    _finish(out, {"apogee_star": star, "apogee_visit": visit, "apogee_field": field}, {**meta_extra,
        "issue": 628, "approved": "Ryan 2026-11-06", "source": "SDSS DR17 APOGEE (Abdurro'uf et al. 2022) via SkyServer SQL",
        "service": SKYSERVER, "query_date": date.isoformat(), "match": "apogeeStar.gaiaedr3_source_id IN (host source_ids)",
        "star_columns": STAR_COLS, "visit_columns": VISIT_COLS, "n_hosts_queried": int(ids.size),
        "n_hosts_matched": int(len(set(np.asarray(star["gaiaedr3_source_id"]).tolist()))),
    })


def fetch_galah(ids: np.ndarray, data_root: Path) -> None:
    from astropy.io import fits
    from astropy.table import Table

    head = urllib.request.urlopen(urllib.request.Request(GALAH_URL, method="HEAD"), timeout=60)
    size = int(head.headers.get("Content-Length", "0"))
    if size > MAX_BYTES:
        raise SystemExit(f"{GALAH_URL} is {size / 1e9:.2f} GB > 1 GB: ask before fetching")
    date = dt.datetime.now(dt.timezone.utc)
    out = _snapshot(data_root, "galah_dr4", date)
    tmp = out / "_allstar.fits"
    h = hashlib.sha256()
    with urllib.request.urlopen(GALAH_URL, timeout=600) as r, tmp.open("wb") as fh:
        for block in iter(lambda: r.read(1 << 22), b""):
            fh.write(block)
            h.update(block)
    with fits.open(tmp, memmap=True) as hd:
        data = hd[1].data
        keep = np.isin(np.asarray(data["gaiadr3_source_id"], np.int64), ids)
        sub = Table(data[keep])
    tmp.unlink()
    for c in sub.colnames:  # ECSV cannot hold multidimensional/byte columns cleanly; keep scalars only
        if sub[c].ndim > 1:
            sub.remove_column(c)
    _finish(out, {"galah_allstar": sub}, {
        "issue": 628, "approved": "Ryan 2026-11-06", "source": "GALAH DR4 allstar (Buder et al. 2025)", "url": GALAH_URL,
        "source_size_bytes": size, "source_sha256": h.hexdigest(), "query_date": date.isoformat(),
        "match": "gaiadr3_source_id in host source_ids", "n_hosts_queried": int(ids.size),
        "n_hosts_matched": int(len(set(np.asarray(sub["gaiadr3_source_id"]).tolist()))),
        "note": "the full allstar file was streamed to a temporary file, subset to the hosts, and deleted",
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("survey", choices=("apogee_dr17", "galah_dr4"))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--chunk", type=int, default=300, help="ids per SkyServer GET request (URL length)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    ids = host_ids(a.published_orbits)
    if a.limit:
        ids = ids[: a.limit]
    if a.survey == "apogee_dr17":
        fetch_apogee(ids, a.chunk, a.data_root)
    else:
        fetch_galah(ids, a.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
