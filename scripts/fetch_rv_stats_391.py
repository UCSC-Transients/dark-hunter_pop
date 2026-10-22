#!/usr/bin/env python
"""#391 (Ryan 2026-10-22): snapshot Gaia DR3 epoch-RV summary statistics for the bright parent rows (G < g_max) and the
real-sample bright hosts (C1 orbits and accelerations, from a fit dump). One TAP query with an uploaded source_id table;
written once to <data_root>/external_catalogs/<stamp>_dr3_rv_stats_bright/ (table.ecsv, meta.yaml, SHA256SUMS)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

COLS = ("radial_velocity", "radial_velocity_error", "rv_method_used", "rv_nb_transits", "rv_nb_deblended_transits",
        "rv_visibility_periods_used", "rv_expected_sig_to_noise", "rv_renormalised_gof", "rv_chisq_pvalue",
        "rv_time_duration", "rv_amplitude_robust", "rv_template_teff", "rv_template_logg", "rv_template_fe_h",
        "grvs_mag", "grvs_mag_nb_transits", "phot_g_mean_mag", "bp_rp", "non_single_star")
ADQL = ("SELECT u.source_id, u.sample, " + ", ".join(f"g.{c}" for c in COLS) +
        "\nFROM tap_upload.ids AS u JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-h5", type=Path, nargs="+", required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--g-max", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(ADQL)
    if a.dry_run:
        return 0
    import h5py
    from astropy.table import Table, vstack
    from astroquery.gaia import Gaia

    par = []
    for p in a.parent_h5:
        with h5py.File(p, "r") as h:
            s, g = h["source_id"][()], h["phot_g_mean_mag"][()]
        par.append(s[g < a.g_max])
    par = np.unique(np.concatenate(par))
    z = np.load(a.dump, allow_pickle=True)
    ro = np.unique(z["r_sid"][(z["rob"] >= 0) & (z["r_g"] < a.g_max)])
    ra = np.unique(z["ra_sid"][z["ra_g"] < a.g_max])
    ids = vstack([Table({"source_id": s.astype(np.int64), "sample": np.full(s.size, k)})
                  for k, s in (("parent", par), ("real_orbit", ro), ("real_accel", ra))])
    tmp = a.data_root / "external_catalogs" / "_rv_stats_upload.xml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    Gaia.ROW_LIMIT = -1
    date = dt.datetime.now(dt.timezone.utc)
    parts = []
    for lo in range(0, len(ids), 20000):  # the archive caps a result at 50,000 rows
        ids[lo:lo + 20000].write(tmp, format="votable", overwrite=True)
        part = Gaia.launch_job_async(ADQL, upload_resource=str(tmp), upload_table_name="ids").get_results()
        if len(part) >= 50000:
            raise SystemExit("chunk hit the row cap")
        parts.append(part)
    tmp.unlink()
    t = vstack(parts)
    out = a.data_root / "external_catalogs" / f"{date.strftime('%Y%m%dT%H%M%SZ')}_dr3_rv_stats_bright"
    out.mkdir(parents=True, exist_ok=False)
    t.write(out / "table.ecsv", format="ascii.ecsv")
    (out / "SHA256SUMS").write_text(hashlib.sha256((out / "table.ecsv").read_bytes()).hexdigest() + "  table.ecsv\n")
    (out / "meta.yaml").write_text(yaml.safe_dump({"issue": 391, "approved": "Ryan 2026-10-22", "query_date": date.isoformat(),
                                                   "adql": ADQL, "g_max": a.g_max, "n_uploaded": {"parent": int(par.size), "real_orbit": int(ro.size),
                                                   "real_accel": int(ra.size)}, "n_rows": int(len(t))}, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
