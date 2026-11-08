#!/usr/bin/env python3
"""#628: CDS XMatch cross-matches of the 168,065 DR3 Orbital + AstroSpectroSB1 hosts against VizieR catalogues
(approved by Ryan 2026-11-06). Hosts only — never a full catalogue download.

Positions are the hosts' Gaia DR3 (J2016.0) positions from the #390 ``published_orbits.h5``, propagated with their
proper motions to the catalogue's mean epoch (``EPOCHS``; null proper motions keep J2016.0). Every catalogue source
within ``--radius`` arcsec is returned with its separation, so the nearest-match choice and ambiguity flags are made
downstream. One snapshot per catalogue:
``<data-root>/dr3/external_catalogs/<UTC stamp>_<name>_nss_hosts/`` with ``table.ecsv``, ``SHA256SUMS``, ``meta.yaml``
(service, VizieR table, radius, epoch, rows uploaded / returned, sha256).

Usage::
    PYTHONPATH=src .venv/bin/python scripts/fetch_nss_host_xmatch_628.py guvcat_ais [--limit 300]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

#: name -> (VizieR table, reference, mean epoch (yr) for the position propagation, default radius arcsec)
CATALOGS = {
    "guvcat_ais": ("vizier:II/335/gal_ais", "GUVcat_AIS, GALEX GR6+7 AIS unique sources (Bianchi et al. 2017, ApJS 230, 24)", 2007.0, 5.0),
    "galex_gr5_mis": ("vizier:II/312/mis", "GALEX GR5 Medium Imaging Survey (Bianchi et al. 2011, Ap&SS 335, 161)", 2007.0, 5.0),
}
CHUNK = 50000


def propagated_positions(path: Path, epoch: float, limit: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(source_id, ra, dec) at ``epoch`` from J2016.0 with the published proper motions (linear, small-angle)."""
    import h5py

    with h5py.File(path, "r") as h:
        p = h["published"]
        sid = p["source_id"][()].astype(np.int64)
        ra, dec = p["ra"][()].astype(float), p["dec"][()].astype(float)
        pmra, pmdec = p["pmra"][()].astype(float), p["pmdec"][()].astype(float)
    _, first = np.unique(sid, return_index=True)
    first = np.sort(first)
    sid, ra, dec, pmra, pmdec = sid[first], ra[first], dec[first], pmra[first], pmdec[first]
    if limit:
        sid, ra, dec, pmra, pmdec = sid[:limit], ra[:limit], dec[:limit], pmra[:limit], pmdec[:limit]
    dt_yr = epoch - 2016.0
    pmra, pmdec = np.nan_to_num(pmra), np.nan_to_num(pmdec)
    dec_e = dec + pmdec * dt_yr / 3.6e6
    ra_e = (ra + pmra * dt_yr / 3.6e6 / np.cos(np.radians(dec))) % 360.0
    return sid, ra_e, dec_e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=sorted(CATALOGS))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    cat, ref, epoch, rdef = CATALOGS[a.name]
    radius = a.radius or rdef
    sid, ra, dec = propagated_positions(a.published_orbits, epoch, a.limit)
    print(f"{a.name}: {cat}, {sid.size} hosts, radius {radius}\", positions at epoch {epoch}")

    from astropy import units as u
    from astropy.table import Table, vstack
    from astroquery.xmatch import XMatch

    date = dt.datetime.now(dt.timezone.utc)
    label = f"{date.strftime('%Y%m%dT%H%M%SZ')}_{a.name}_nss_hosts" + ("_smoke" if a.limit else "")
    out = a.data_root / "dr3" / "external_catalogs" / label
    parts = []
    for lo in range(0, sid.size, CHUNK):
        up = Table({"source_id": sid[lo:lo + CHUNK], "ra_epoch": ra[lo:lo + CHUNK], "dec_epoch": dec[lo:lo + CHUNK]})
        part = XMatch.query(cat1=up, cat2=cat, max_distance=radius * u.arcsec, colRA1="ra_epoch", colDec1="dec_epoch")
        parts.append(part)
        print(f"  chunk {lo // CHUNK + 1}: {len(part)} rows", flush=True)
    t = vstack(parts) if parts else Table()
    out.mkdir(parents=True, exist_ok=False)
    t.write(out / "table.ecsv", format="ascii.ecsv")
    sha = hashlib.sha256((out / "table.ecsv").read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{sha}  table.ecsv\n")
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "issue": 628, "name": a.name, "approved": "Ryan 2026-11-06", "source": ref, "vizier_table": cat,
        "service": "CDS XMatch (astroquery.xmatch)", "query_date": date.isoformat(), "radius_arcsec": radius,
        "position_epoch": epoch, "upload": str(a.published_orbits), "n_uploaded": int(sid.size),
        "n_rows": int(len(t)), "n_hosts_matched": int(len(set(np.asarray(t["source_id"]).tolist()))) if len(t) else 0,
        "table_ecsv_sha256": sha,
    }, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
