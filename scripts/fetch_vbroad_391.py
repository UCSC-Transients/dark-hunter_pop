#!/usr/bin/env python
"""#391 (Ryan 2026-10-18): snapshot Gaia DR3 vbroad (+ Teff) for bright real NSS hosts and parent rows.

One Gaia TAP query with an uploaded source_id table (real C1-orbit and acceleration hosts at G < ``--g-max``
from a fit dump, plus the parent rows at G < ``--g-max``). Written once to
``<data_root>/external_catalogs/<stamp>_dr3_vbroad_bright/`` as ``table.ecsv`` + ``meta.yaml`` (ADQL, date,
row count) + ``SHA256SUMS``. ``--dry-run`` prints the ADQL only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

ADQL = (
    "SELECT u.source_id, u.sample, g.phot_g_mean_mag, g.vbroad, g.vbroad_error, g.vbroad_nb_transits, "
    "g.rv_template_teff, ap.teff_gspphot, ap.teff_esphs\n"
    "FROM tap_upload.ids AS u\n"
    "JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id\n"
    "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = u.source_id"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--parent-h5", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--g-max", type=float, default=11.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    print(ADQL)
    if args.dry_run:
        return 0
    import h5py
    from astropy.table import Table, vstack
    from astroquery.gaia import Gaia

    z = np.load(args.dump, allow_pickle=True)
    rin = z["rob"] >= 0
    real_o = np.unique(z["r_sid"][rin & (z["r_g"] < args.g_max)])
    real_a = np.unique(z["ra_sid"][z["ra_g"] < args.g_max])
    with h5py.File(args.parent_h5, "r") as h:
        psid, pg = h["source_id"][()], h["phot_g_mean_mag"][()]
    par = np.unique(psid[pg < args.g_max])
    ids = vstack([Table({"source_id": s.astype(np.int64), "sample": np.full(s.size, k)})
                  for k, s in (("real_orbit", real_o), ("real_accel", real_a), ("parent", par))])
    tmp = args.data_root / "external_catalogs" / "_vbroad_upload.xml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ids.write(tmp, format="votable", overwrite=True)
    Gaia.ROW_LIMIT = -1
    date = dt.datetime.now(dt.timezone.utc)
    t = Gaia.launch_job_async(ADQL, upload_resource=str(tmp), upload_table_name="ids").get_results()
    tmp.unlink()
    out = args.data_root / "external_catalogs" / f"{date.strftime('%Y%m%dT%H%M%SZ')}_dr3_vbroad_bright"
    out.mkdir(parents=True, exist_ok=False)
    t.write(out / "table.ecsv", format="ascii.ecsv")
    sha = hashlib.sha256((out / "table.ecsv").read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{sha}  table.ecsv\n")
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "issue": 391, "approved": "Ryan 2026-10-18", "query_date": date.isoformat(), "adql": ADQL, "g_max": args.g_max,
        "n_uploaded": {"real_orbit": int(real_o.size), "real_accel": int(real_a.size), "parent": int(par.size)},
        "n_rows": int(len(t)), "dump": str(args.dump), "parent_h5": str(args.parent_h5)}, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
