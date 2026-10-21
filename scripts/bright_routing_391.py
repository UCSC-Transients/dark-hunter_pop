#!/usr/bin/env python
"""#391 (Ryan 2026-10-21) bright-end routing: which DR3 NSS solution types do bright (G < 9) stars of the random parent
get, versus the mock's astrometric-orbit prediction at the same G. One Gaia TAP query (uploaded source_ids of the
parent's G < g_max rows, original and bright-supplement strata) against gaiadr3.nss_two_body_orbit, snapshotted under
<data_root>/external_catalogs/<stamp>_dr3_nss_bright_parent/ (table.ecsv, meta.yaml, SHA256SUMS). Then per-star rates
(row-weighted) by solution type vs the mock (fit dump) and the real C1 sample. Writes JSON."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ADQL = ("SELECT u.source_id, n.nss_solution_type, n.period, n.eccentricity, n.significance, n.goodness_of_fit\n"
        "FROM tap_upload.ids AS u JOIN gaiadr3.nss_two_body_orbit AS n ON n.source_id = u.source_id")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_single43.yaml"))
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=None, help="reuse an existing snapshot dir instead of querying")
    ap.add_argument("--g-max", type=float, default=9.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config(Path("config/config.yaml"))
    par = ps.load_parent_snapshot(a.parent_dir, cfg, ps.load_proposal_set_fragment(a.fragment).proposal)
    g = np.asarray(par.columns["phot_g_mean_mag"], float)
    base = np.ones(par.n_rows, bool)
    for k in ("finite_astrometry_photometry", "parallax_floor", "bailer_jones_distance"):
        base &= np.asarray(par.flags[k], bool)
    bright = base & (g < a.g_max)
    sid = np.asarray(par.columns["source_id"], np.int64)
    rw = np.ones(par.n_rows) if par.row_weight is None else par.row_weight
    if a.snapshot is None:
        from astroquery.gaia import Gaia

        ids = Table({"source_id": sid[bright]})
        tmp = a.data_root / "external_catalogs" / "_nss_bright_upload.xml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        ids.write(tmp, format="votable", overwrite=True)
        date = dt.datetime.now(dt.timezone.utc)
        t = Gaia.launch_job_async(ADQL, upload_resource=str(tmp), upload_table_name="ids").get_results()
        tmp.unlink()
        snap = a.data_root / "external_catalogs" / f"{date.strftime('%Y%m%dT%H%M%SZ')}_dr3_nss_bright_parent"
        snap.mkdir(parents=True, exist_ok=False)
        t.write(snap / "table.ecsv", format="ascii.ecsv")
        (snap / "SHA256SUMS").write_text(hashlib.sha256((snap / "table.ecsv").read_bytes()).hexdigest() + "  table.ecsv\n")
        (snap / "meta.yaml").write_text(yaml.safe_dump({"issue": 391, "approved": "Ryan 2026-10-21", "query_date": date.isoformat(),
                                                        "adql": ADQL, "n_uploaded": int(bright.sum()), "g_max": a.g_max,
                                                        "parent": str(a.parent_dir)}, sort_keys=False))
    else:
        snap = a.snapshot
        t = Table.read(snap / "table.ecsv")
    st = {int(s): str(v) for s, v in zip(t["source_id"], t["nss_solution_type"])}
    types = np.array([st.get(int(s), "none") for s in sid[bright]])
    w_b = rw[bright] * par.scale_to_full
    n_stars = float(w_b.sum())
    rates = {ty: float(w_b[types == ty].sum() / n_stars) for ty in np.unique(types)}
    nss = np.asarray(par.columns["non_single_star"], float)[bright] if "non_single_star" in par.columns else np.full(bright.sum(), np.nan)
    z = np.load(a.dump, allow_pickle=True)
    so, rin = z["ob"] >= 0, z["rob"] >= 0
    mock_orb = float(z["w"][so & (z["m_g"] < a.g_max)].sum())
    real_orb = int((rin & (z["r_g"] < a.g_max)).sum())
    real_st = {t_: int(np.sum(z["r_soltype"][rin & (z["r_g"] < a.g_max)] == t_)) for t_ in np.unique(z["r_soltype"][rin & (z["r_g"] < a.g_max)])}
    out = {"snapshot": str(snap), "n_bright_rows": int(bright.sum()), "n_stars_represented": n_stars,
           "per_star_rate_by_solution_type": rates,
           "non_single_star_flag_share": {str(int(v)): float(w_b[nss == v].sum() / n_stars) for v in np.unique(nss[np.isfinite(nss)])},
           "mock_C1_orbits_per_star": mock_orb / n_stars, "real_C1_orbits_per_star": real_orb / n_stars,
           "real_C1_by_type": real_st, "missing_per_star": (mock_orb - real_orb) / n_stars}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
