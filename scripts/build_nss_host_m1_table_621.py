#!/usr/bin/env python3
"""#621 (R5-T2b): MIST isochrone M1 posterior table for every DR3 Orbital + AstroSpectroSB1 host (168,065 rows).

Same model and calibration as the mock's looks-single primaries (``proposal_set.parent_cmd_isochrone``):
``config.isochrone_mass`` (MIST v1.2, the §12.24 Teff-only low-mass correction calibrated on Mann et al. 2019, the
MP-Q33 MDF prior, the APOKASC-3 age weight), ``giants.cmd_for_rows`` for the dereddened CMD (Combined19 at the row
distance, Babusiaux et al. 2018 law) and ``isochrone_mass`` ``model.fit``. The table carries the posterior summary that
both M2_app paths consume (``log_m1_mean``, ``log_m1_sigma``) plus ``m1_mean``, ``m1_sigma``, ``p_evolved``,
``log_age_mean``, ``feh_mean`` and ``ok`` / ``reason``.

**Distance convention.** Default ``--distance bj``: the Bailer-Jones et al. (2021) geometric distance
(``r_med_geo``, ``r_lo_geo``, ``r_hi_geo``), the same convention as the mock's looks-single parent (MP-Q4), from the #628
host snapshot (``--bj-snapshot``; checksum verified against its meta.yaml). Hosts without a Bailer-Jones distance get
no M1 (reason ``no_cmd``), as parent rows without one are not usable. ``--distance zp_parallax`` keeps the rung-3
real-side convention (``scripts/rung3_refit5_391.py`` ``cls_of``: 1000 / (ϖ + zp(G)) pc with ±σ_ϖ bounds, zp(G) the
median truth − observed parallax of usable parent rows with ϖ/σ_ϖ > 20 in G bins [0, 11, 13, 15, 17, 25]) for comparison.

Heavy at full size (168k posteriors): run through ``scripts/jobs/run_heavy.sh``; ``--limit`` gives a smoke run.

Usage::
    scripts/jobs/run_heavy.sh nss_host_m1_621 scripts/build_nss_host_m1_table_621.py \\
        --published-orbits output/gate390/published_orbits.h5 \\
        --input-columns data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns \\
        --bj-snapshot data/dr3/external_catalogs/20261107T052615Z_bj_distances_nss_hosts --out output/rung5/nss_host_m1_dr3.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ZP_G_EDGES = np.array([0, 11, 13, 15, 17, 25.0])  # rung-3 real-side convention (rung3_refit5_391.py)
ZP_SNR_MIN = 20.0
SUMMARY = ("log_m1_mean", "log_m1_sigma", "m1_mean", "m1_sigma", "p_evolved", "log_age_mean", "feh_mean")


def parallax_zero_point(parent, edges: np.ndarray = ZP_G_EDGES, snr_min: float = ZP_SNR_MIN) -> np.ndarray:
    """Median (truth − observed) parallax of usable parent rows with ϖ/σ_ϖ > ``snr_min``, per G bin."""
    pc = parent.columns
    g = np.asarray(pc["phot_g_mean_mag"], float)
    snr = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    return np.array([np.median(dzp[parent.usable & (g >= a) & (g < b) & (snr > snr_min)]) for a, b in zip(edges[:-1], edges[1:])])


def row_distances(g: np.ndarray, plx_obs: np.ndarray, eplx: np.ndarray, zp: np.ndarray, edges: np.ndarray = ZP_G_EDGES):
    """(r_med, r_lo, r_hi) pc from the zero-point-corrected parallax; NaN where undefined."""
    plx = plx_obs + zp[np.clip(np.digitize(g, edges) - 1, 0, zp.size - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(plx > 0, 1000.0 / plx, np.nan)
        rlo = np.where(plx > 0, 1000.0 / (plx + eplx), np.nan)
        rhi = np.where(plx - eplx > 0, 1000.0 / (plx - eplx), np.nan)
    return rr, rlo, rhi


def bj_distances(snapshot: Path, sid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r_med_geo, r_lo_geo, r_hi_geo) pc for ``sid`` from a #628 snapshot (sha256 checked); NaN where absent."""
    import yaml
    from astropy.table import Table

    meta = yaml.safe_load((snapshot / "meta.yaml").read_text())
    got = hashlib.sha256((snapshot / "table.ecsv").read_bytes()).hexdigest()
    if got != meta["table_ecsv_sha256"]:
        raise SystemExit(f"{snapshot}: table.ecsv checksum {got} != meta {meta['table_ecsv_sha256']}")
    t = Table.read(snapshot / "table.ecsv", format="ascii.ecsv")
    pos = {int(s): i for i, s in enumerate(np.asarray(t["source_id"], np.int64))}
    idx = np.array([pos.get(int(s), -1) for s in sid])
    out = []
    for c in ("r_med_geo", "r_lo_geo", "r_hi_geo"):
        v = np.ma.filled(np.ma.asarray(t[c], float), np.nan)
        out.append(np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan))
    return out[0], out[1], out[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published-orbits", type=Path, required=True)
    ap.add_argument("--input-columns", type=Path, required=True, help="snapshot dir with columns.h5 (bp_rp, phot_g_mean_mag)")
    ap.add_argument("--parent-dir", type=Path, default=None, help="looks-single parent snapshot (zp_parallax only)")
    ap.add_argument("--fragment", type=Path, default=None, help="a proposal fragment of the adopted version (zp_parallax only)")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--giants-config", type=Path, default=Path("config/population/giants.yaml"))
    ap.add_argument("--distance", choices=("bj", "zp_parallax"), default="bj")
    ap.add_argument("--bj-snapshot", type=Path, default=None, help="#628 Bailer-Jones host snapshot dir (table.ecsv, meta.yaml)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from darkhunter_pop import isochrone_mass as im
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.giants import cmd_for_rows, load_giants_config

    t0 = time.time()
    cfg = load_config(args.config)
    gcfg = load_giants_config(args.giants_config)
    with h5py.File(args.published_orbits, "r") as h:
        p = h["published"]
        n = p["source_id"].shape[0] if args.limit is None else min(args.limit, p["source_id"].shape[0])
        sid = p["source_id"][:n].astype(np.int64)
        ra, dec = p["ra"][:n].astype(float), p["dec"][:n].astype(float)
        plx, eplx = p["parallax"][:n].astype(float), p["parallax_error"][:n].astype(float)
    with h5py.File(args.input_columns / "columns.h5", "r") as h:
        cs = h["source_id"][()].astype(np.int64)
        order = {int(s): i for i, s in enumerate(cs)}
        idx = np.array([order.get(int(s), -1) for s in sid])
        if np.any(idx < 0):
            raise SystemExit(f"{int(np.sum(idx < 0))} hosts missing from {args.input_columns}")
        g = h["phot_g_mean_mag"][()][idx].astype(float)
        bprp = h["bp_rp"][()][idx].astype(float)

    if args.distance == "bj":
        if args.bj_snapshot is None:
            raise SystemExit("--distance bj needs --bj-snapshot")
        rr, rlo, rhi = bj_distances(args.bj_snapshot, sid)
        zp = np.full(ZP_G_EDGES.size - 1, np.nan)
    else:
        if args.parent_dir is None or args.fragment is None:
            raise SystemExit("--distance zp_parallax needs --parent-dir and --fragment")
        parent = ps.load_parent_snapshot(args.parent_dir, cfg, ps.load_proposal_set_fragment(args.fragment).proposal)
        zp = parallax_zero_point(parent)
        del parent
        rr, rlo, rhi = row_distances(g, plx, eplx, zp)
    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
    rc = cmd_for_rows(g, bprp, gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg)
    model = im.build_model(cfg.isochrone_mass, cfg.paths.data_root)
    post = model.fit(rc.colour0, rc.mg0, sigma_mu=rc.sigma_mu, ebv=rc.ebv, a_g=rc.a_g, e_bp_rp=bprp - rc.colour0)
    d = post.as_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as h:
        h.create_dataset("source_id", data=sid)
        for k in SUMMARY:
            h.create_dataset(k, data=np.asarray(d[k], float))
        h.create_dataset("ok", data=np.asarray(d["ok"], bool))
        h.create_dataset("reason", data=np.asarray(d["reason"]).astype(str).astype(h5py.string_dtype("utf-8")))
        for k, v in (("mg0", rc.mg0), ("colour0", rc.colour0), ("sigma_mu", rc.sigma_mu), ("r_med_pc", rr)):
            h.create_dataset(k, data=np.asarray(v, float))
        h.attrs.update({
            "issue": 621, "data_release": "dr3", "distance": args.distance, "bj_snapshot": str(args.bj_snapshot), "zp_g_edges": ZP_G_EDGES, "zp_mas": zp,
            "isochrone_config_key": im.config_key(cfg.isochrone_mass),
            "isochrone_config_json": json.dumps(cfg.isochrone_mass.model_dump(mode="json"), sort_keys=True),
            "giants_extinction_json": json.dumps(gcfg.extinction.model_dump(mode="json"), sort_keys=True),
            "published_orbits_sha256": hashlib.sha256(args.published_orbits.read_bytes()).hexdigest(),
            "n_rows": int(n),
        })
    ok = np.asarray(d["ok"], bool)
    print(f"{n} hosts, {int(ok.sum())} with an isochrone M1 ({time.time() - t0:.0f} s); zp(G) = {np.round(zp, 4).tolist()} mas")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
