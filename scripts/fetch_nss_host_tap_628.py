#!/usr/bin/env python3
"""#628: Gaia-archive TAP cross-matches for the 168,065 DR3 Orbital + AstroSpectroSB1 hosts (approved by Ryan 2026-11-06).

Uploads the host source_ids of the #390 ``published_orbits.h5`` in chunks (the archive caps one result at 50,000 rows)
and writes one snapshot directory per kind:
``<data-root>/dr3/external_catalogs/<UTC stamp>_<kind>_nss_hosts/`` with ``table.ecsv``, ``SHA256SUMS`` and
``meta.yaml`` (ADQL, source, query date, rows uploaded / returned, sha256). Never overwrites an existing snapshot.

Kinds:
* ``bj_distances``: Bailer-Jones et al. (2021) EDR3 distances (``external.gaiaedr3_distance``; DR3 source_id = EDR3).
* ``rv_stats``: DR3 ``gaia_source`` RV statistics for every host (all magnitudes).

* ``galex_ais``: GUVcat_AIS (``external.galex_ais``); the archive has no Gaia–GALEX best-neighbour table and rejects
  uploads joined positionally to it, so this kind runs server-side on ``gaiadr3.nss_two_body_orbit`` in source_id-range
  chunks (no upload); every UV source within 10\" with the host's ra/dec/pm (one host can have several rows).

``--limit N`` uploads only the first N hosts (a smoke query; the snapshot is labelled ``_smoke``).
Public catalogs only. Network: the Gaia archive (astroquery); anonymous unless GAIA_ARCHIVE_USER is set.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

QUERIES = {
    "bj_distances": (
        "Bailer-Jones et al. (2021), AJ 161, 147; Gaia archive external.gaiaedr3_distance",
        "SELECT u.source_id, d.r_med_geo, d.r_lo_geo, d.r_hi_geo, d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo, d.flag\n"
        "FROM tap_upload.ids AS u LEFT JOIN external.gaiaedr3_distance AS d ON d.source_id = u.source_id",
    ),
    "galex_ais": (
        "GUVcat_AIS, GALEX GR6+7 AIS unique UV sources (Bianchi et al. 2017, ApJS 230, 24); Gaia archive external.galex_ais. "
        "Server-side positional join (the archive rejects an uploaded table joined positionally to external.galex_ais): "
        "every GUVcat source within 10 arcsec of the J2016.0 position of each DR3 Orbital/AstroSpectroSB1 host, chunked "
        "by source_id range; epoch propagation, the nearest-match choice and ambiguity flags are made downstream",
        "SELECT n.source_id, n.nss_solution_type, g.ra AS gaia_ra, g.dec AS gaia_dec, g.pmra AS gaia_pmra, g.pmdec AS gaia_pmdec,\n"
        "  x.objid, x.raj2000, x.dej2000, x.fuvmag, x.e_fuvmag, x.nuvmag, x.e_nuvmag, x.fuvexp, x.nuvexp,\n"
        "  x.fafl, x.nafl, x.fexf, x.nexf, x.e_b_v, x.r_fov, x.primid, x.groupid, x.grouptot, x.tile,\n"
        "  DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', x.raj2000, x.dej2000)) * 3600.0 AS sep_j2016_arcsec\n"
        "FROM gaiadr3.nss_two_body_orbit AS n\n"
        "JOIN gaiadr3.gaia_source AS g ON g.source_id = n.source_id\n"
        "JOIN external.galex_ais AS x ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', x.raj2000, x.dej2000, 10.0 / 3600.0))\n"
        "WHERE n.nss_solution_type IN ('Orbital', 'AstroSpectroSB1') AND n.source_id BETWEEN {lo} AND {hi}",
    ),
    "lamost_dr9_lrs": (
        "LAMOST DR9 low-resolution survey observations (Gaia archive external.lamost_dr9_lrs); server-side positional join, "
        "every LRS observation within 3 arcsec of the J2016.0 host position (one row per observation)",
        "SELECT n.source_id, g.ra AS gaia_ra, g.dec AS gaia_dec, x.obsid, x.uid, x.designation, x.obsdate, x.lmjd, x.mjd,\n"
        "  x.ra, x.dec, x.snrg, x.snrr, x.class, x.subclass, x.z, x.z_err, x.fibermask,\n"
        "  DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', x.ra, x.dec)) * 3600.0 AS sep_j2016_arcsec\n"
        "FROM gaiadr3.nss_two_body_orbit AS n\n"
        "JOIN gaiadr3.gaia_source AS g ON g.source_id = n.source_id\n"
        "JOIN external.lamost_dr9_lrs AS x ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', x.ra, x.dec, 3.0 / 3600.0))\n"
        "WHERE n.nss_solution_type IN ('Orbital', 'AstroSpectroSB1') AND n.source_id BETWEEN {lo} AND {hi}",
    ),
    "lamost_dr9_mrs": (
        "LAMOST DR9 medium-resolution survey epochs with RVs (Gaia archive external.lamost_dr9_mrs); server-side positional "
        "join, every MRS exposure within 3 arcsec of the J2016.0 host position (one row per exposure)",
        "SELECT n.source_id, g.ra AS gaia_ra, g.dec AS gaia_dec, x.mobsid, x.obsid, x.uid, x.designation, x.obsdate, x.lmjd,\n"
        "  x.lmjm, x.mjd, x.band, x.ra, x.dec, x.snr, x.coadd, x.rv_b1, x.rv_b1_err, x.rv_b_flag, x.rv_r1, x.rv_r1_err,\n"
        "  x.rv_r_flag, x.rv_br1, x.rv_br1_err, x.rv_br_flag, x.rv_lasp1, x.rv_lasp1_err, x.fibermask, x.bad_b, x.bad_r,\n"
        "  DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', x.ra, x.dec)) * 3600.0 AS sep_j2016_arcsec\n"
        "FROM gaiadr3.nss_two_body_orbit AS n\n"
        "JOIN gaiadr3.gaia_source AS g ON g.source_id = n.source_id\n"
        "JOIN external.lamost_dr9_mrs AS x ON 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', x.ra, x.dec, 3.0 / 3600.0))\n"
        "WHERE n.nss_solution_type IN ('Orbital', 'AstroSpectroSB1') AND n.source_id BETWEEN {lo} AND {hi}",
    ),
    "rv_stats": (
        "Gaia DR3 gaia_source RV statistics (Katz et al. 2023)",
        "SELECT u.source_id, g.radial_velocity, g.radial_velocity_error, g.rv_method_used, g.rv_nb_transits,\n"
        "  g.rv_nb_deblended_transits, g.rv_visibility_periods_used, g.rv_expected_sig_to_noise, g.rv_renormalised_gof,\n"
        "  g.rv_chisq_pvalue, g.rv_time_duration, g.rv_amplitude_robust, g.rv_template_teff, g.rv_template_logg,\n"
        "  g.rv_template_fe_h, g.grvs_mag, g.grvs_mag_nb_transits, g.phot_g_mean_mag, g.bp_rp, g.non_single_star\n"
        "FROM tap_upload.ids AS u JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id",
    ),
}
CHUNK = 20000
ROW_CAP = 50000


def host_ids(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as h:
        return np.unique(h["published"]["source_id"][()].astype(np.int64))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=sorted(QUERIES))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=CHUNK, help="hosts per query (lower if a result hits the row cap)")
    ap.add_argument("--dry-run", action="store_true", help="print the ADQL and the upload size; query nothing")
    a = ap.parse_args(argv)
    source, adql = QUERIES[a.kind]
    ids = host_ids(a.published_orbits)
    if a.limit:
        ids = ids[: a.limit]
    print(adql)
    print(f"{ids.size} host source_ids, {int(np.ceil(ids.size / a.chunk))} chunk(s)")
    if a.dry_run:
        return 0

    from astropy.table import Table, vstack
    from astroquery.gaia import Gaia

    date = dt.datetime.now(dt.timezone.utc)
    label = f"{date.strftime('%Y%m%dT%H%M%SZ')}_{a.kind}_nss_hosts" + ("_smoke" if a.limit else "")
    out = a.data_root / "dr3" / "external_catalogs" / label
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "_upload.xml"
    Gaia.ROW_LIMIT = -1
    parts = []
    if "{lo}" in adql:  # server-side kind: source_id-range chunks over the hosts, no upload
        edges = ids[::a.chunk].tolist() + [int(ids[-1]) + 1]
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            part = Gaia.launch_job_async(adql.format(lo=int(lo), hi=int(hi) - 1)).get_results()
            if len(part) >= ROW_CAP:
                raise SystemExit(f"range {lo}-{hi} hit the {ROW_CAP}-row cap; lower CHUNK")
            if a.limit:
                part = part[np.isin(np.asarray(part["source_id"], np.int64), ids)]
            parts.append(part)
            print(f"  range {k + 1}: {len(part)} rows", flush=True)
        ids_iter: range = range(0)
    else:
        ids_iter = range(0, ids.size, a.chunk)
    for lo in ids_iter:
        Table({"source_id": ids[lo:lo + a.chunk]}).write(tmp, format="votable", overwrite=True)
        part = Gaia.launch_job_async(adql, upload_resource=str(tmp), upload_table_name="ids").get_results()
        if len(part) >= ROW_CAP:
            raise SystemExit(f"chunk at {lo} hit the {ROW_CAP}-row cap; lower CHUNK")
        parts.append(part)
        print(f"  chunk {lo // a.chunk + 1}: {len(part)} rows", flush=True)
    if tmp.exists():
        tmp.unlink()
    t = vstack(parts)
    t.write(out / "table.ecsv", format="ascii.ecsv")
    sha = hashlib.sha256((out / "table.ecsv").read_bytes()).hexdigest()
    (out / "SHA256SUMS").write_text(f"{sha}  table.ecsv\n")
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "issue": 628, "kind": a.kind, "approved": "Ryan 2026-11-06", "source": source,
        "service": "Gaia archive TAP (astroquery.gaia)", "query_date": date.isoformat(), "adql": adql,
        "upload": str(a.published_orbits), "n_uploaded": int(ids.size), "n_rows": int(len(t)),
        "table_ecsv_sha256": sha,
    }, sort_keys=False))
    print(f"wrote {len(t)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
