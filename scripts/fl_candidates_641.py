#!/usr/bin/env python3
"""FL-T1 (#641; RUNG5_SPEC §11.2, FL-Q2): the follow-up candidate set and its per-host coverage by the downloaded surveys.

Candidates: real DR3 C1 rows of ``real_m2app_dr3.h5`` (reason ``ok``) with M2_app ≥ ``--m2-min`` (1.0 M⊙), or A > ``--a-min``
(0.5) and M2_app ≥ ``--m2-min-large-a`` (0.6 M⊙). A = (a0/ϖ) / P_yr^(2/3) from the published orbit
(``physics_utils.photocenter_a0_from_thiele_innes``; ``output/gate390/published_orbits.h5``). M2_app is the R5-T2 estimator
with the adopted ν (#638).

Coverage flags per candidate, from the #628 snapshots (by source_id; public catalogues only):
* GALEX (GUVcat_AIS): the nearest UV source to the host position propagated from J2016.0 to ``--galex-epoch`` within
  ``--galex-radius`` arcsec; ``n_uv_in_radius`` > 1 is flagged ambiguous; NUV / FUV detected.
* Gaia DR3 RV statistics: ``rv_amplitude_robust`` present (G_RVS ≲ 12), ``radial_velocity`` present, ``rv_nb_transits``.
* LAMOST DR9: LRS observations; MRS exposures with a finite RV (``rv_br1`` or ``rv_lasp1``) and their distinct nights.
* APOGEE DR17: stars and ``nvisits`` (max over telescopes); GALAH DR4: present.
* ``multi_epoch_rv``: ≥ 2 epochs in LAMOST MRS (distinct nights) or APOGEE (nvisits ≥ 2).

Writes ``<out>/fl_candidates_dr3.ecsv`` (one row per candidate) and ``<out>/fl_candidates_summary.json`` (counts per
frozen M2_app bin and coverage fractions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

SNAP = "data/dr3/external_catalogs/"
DEFAULTS = {
    "galex": SNAP + "20261107T054614Z_galex_ais_nss_hosts",
    "rv_stats": SNAP + "20261107T054835Z_rv_stats_nss_hosts",
    "lamost_lrs": SNAP + "20261107T055245Z_lamost_dr9_lrs_nss_hosts",
    "lamost_mrs": SNAP + "20261107T055538Z_lamost_dr9_mrs_nss_hosts",
    "apogee": SNAP + "20261107T105347Z_apogee_dr17_nss_hosts",
    "galah": SNAP + "20261107T105411Z_galah_dr4_nss_hosts",
}


def _col(t, name: str) -> np.ndarray:
    return np.ma.filled(np.ma.asarray(t[name], float), np.nan)


def select_candidates(m2_app: np.ndarray, a_size: np.ndarray, ok: np.ndarray, m2_min: float, a_min: float,
                      m2_min_large_a: float) -> np.ndarray:
    """FL-Q2: M2_app ≥ m2_min, or A > a_min and M2_app ≥ m2_min_large_a (rows with an M2_app only)."""
    with np.errstate(invalid="ignore"):
        return ok & np.isfinite(m2_app) & ((m2_app >= m2_min) | ((a_size > a_min) & (m2_app >= m2_min_large_a)))


def propagated_sep_arcsec(ra, dec, pmra, pmdec, ra2, dec2, dt_yr: float) -> np.ndarray:
    """Separation (arcsec) between (ra2, dec2) and the J2016 position moved by dt_yr with the proper motion (small angle)."""
    pmra, pmdec = np.nan_to_num(np.asarray(pmra, float)), np.nan_to_num(np.asarray(pmdec, float))
    d = np.asarray(dec, float) + pmdec * dt_yr / 3.6e6
    r = np.asarray(ra, float) + pmra * dt_yr / 3.6e6 / np.cos(np.radians(dec))
    dra = (np.asarray(ra2, float) - r + 180.0) % 360.0 - 180.0
    return 3600.0 * np.hypot(dra * np.cos(np.radians(d)), np.asarray(dec2, float) - d)


def galex_flags(t, sids: np.ndarray, epoch: float, radius: float) -> dict[str, np.ndarray]:
    sep = propagated_sep_arcsec(_col(t, "gaia_ra"), _col(t, "gaia_dec"), _col(t, "gaia_pmra"), _col(t, "gaia_pmdec"),
                                _col(t, "raj2000"), _col(t, "dej2000"), epoch - 2016.0)
    sid = np.asarray(t["source_id"], np.int64)
    nuv, fuv = _col(t, "nuvmag"), _col(t, "fuvmag")
    out = {k: np.full(sids.size, np.nan) for k in ("uv_sep_arcsec", "nuv_mag", "fuv_mag")}
    out["n_uv_in_radius"] = np.zeros(sids.size, int)
    pos = {int(s): i for i, s in enumerate(sids)}
    for j in np.argsort(sep):
        i = pos.get(int(sid[j]))
        if i is None or not sep[j] <= radius:
            continue
        out["n_uv_in_radius"][i] += 1
        if out["n_uv_in_radius"][i] == 1:
            out["uv_sep_arcsec"][i], out["nuv_mag"][i], out["fuv_mag"][i] = sep[j], nuv[j], fuv[j]
    return out


def _by_sid(sids: np.ndarray, key: np.ndarray, values: np.ndarray, how: str) -> np.ndarray:
    """Aggregate ``values`` per source_id in ``sids`` (``count`` / ``max`` / ``nunique``)."""
    out = np.zeros(sids.size, float if how == "max" else int)
    if how == "max":
        out[:] = np.nan
    pos = {int(s): i for i, s in enumerate(sids)}
    seen: dict[int, set] = {}
    for k, v in zip(np.asarray(key, np.int64), values):
        i = pos.get(int(k))
        if i is None:
            continue
        if how == "count":
            out[i] += 1
        elif how == "max":
            out[i] = v if np.isnan(out[i]) else max(out[i], v)
        else:
            seen.setdefault(i, set()).add(v)
    if how == "nunique":
        for i, s in seen.items():
            out[i] = len(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m2app", type=Path, default=Path("output/rung5/real_m2app_dr3.h5"))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--bins-config", type=Path, default=Path("config/population/rung5_bins_dr3.yaml"))
    ap.add_argument("--m2-min", type=float, default=1.0)
    ap.add_argument("--a-min", type=float, default=0.5)
    ap.add_argument("--m2-min-large-a", type=float, default=0.6)
    ap.add_argument("--galex-epoch", type=float, default=2007.0)
    ap.add_argument("--galex-radius", type=float, default=3.0)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=Path(v))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import h5py
    from astropy.table import Table

    from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes

    with h5py.File(a.m2app, "r") as h:
        g = h["m2app"]
        sid = g["source_id"][()].astype(np.int64)
        m2 = g["m2_med"][()]
        m2_lo, m2_hi = g["m2_q16"][()], g["m2_q84"][()]
        ok = g["reason"][()].astype(str) == "ok"
        stype = g["nss_solution_type"][()].astype(str)
        nu_attr = h.attrs.get("mass_recovery_json", "")
    with h5py.File(a.published_orbits, "r") as h:
        p = h["published"]
        psid = p["source_id"][()].astype(np.int64)
        a0 = photocenter_a0_from_thiele_innes(p["a_thiele_innes"][()], p["b_thiele_innes"][()], p["f_thiele_innes"][()], p["g_thiele_innes"][()])
        with np.errstate(divide="ignore", invalid="ignore"):
            a_pub = (a0 / p["parallax"][()]) / (p["period"][()] / 365.25) ** (2.0 / 3.0)
        g_pub = p["g_mag"][()]
    pos = {int(s): i for i, s in enumerate(psid)}
    idx = np.array([pos.get(int(s), -1) for s in sid])
    a_size = np.where(idx >= 0, a_pub[np.clip(idx, 0, None)], np.nan)
    g_mag = np.where(idx >= 0, g_pub[np.clip(idx, 0, None)], np.nan)
    sel = select_candidates(m2, a_size, ok, a.m2_min, a.a_min, a.m2_min_large_a)
    c = Table({"source_id": sid[sel], "nss_solution_type": stype[sel], "g_mag": g_mag[sel], "a_size": a_size[sel],
               "m2_app": m2[sel], "m2_app_q16": m2_lo[sel], "m2_app_q84": m2_hi[sel]})
    cs = np.asarray(c["source_id"], np.int64)

    gx = galex_flags(Table.read(a.galex / "table.ecsv", format="ascii.ecsv"), cs, a.galex_epoch, a.galex_radius)
    for k, v in gx.items():
        c[k] = v
    c["galex_nuv"] = np.isfinite(c["nuv_mag"])
    c["galex_fuv"] = np.isfinite(c["fuv_mag"])
    c["galex_ambiguous"] = c["n_uv_in_radius"] > 1

    rv = Table.read(a.rv_stats / "table.ecsv", format="ascii.ecsv")
    rpos = {int(s): i for i, s in enumerate(np.asarray(rv["source_id"], np.int64))}
    ri = np.array([rpos.get(int(s), -1) for s in cs])
    for name in ("rv_amplitude_robust", "radial_velocity", "rv_nb_transits", "rv_chisq_pvalue", "grvs_mag"):
        v = _col(rv, name)
        c[name] = np.where(ri >= 0, v[np.clip(ri, 0, None)], np.nan)
    c["gaia_rv_amplitude"] = np.isfinite(c["rv_amplitude_robust"])

    lrs = Table.read(a.lamost_lrs / "table.ecsv", format="ascii.ecsv")
    c["lamost_lrs_n_obs"] = _by_sid(cs, lrs["source_id"], np.ones(len(lrs)), "count")
    mrs = Table.read(a.lamost_mrs / "table.ecsv", format="ascii.ecsv")
    rvok = np.isfinite(_col(mrs, "rv_br1")) | np.isfinite(_col(mrs, "rv_lasp1"))
    c["lamost_mrs_n_rv"] = _by_sid(cs, np.asarray(mrs["source_id"])[rvok], np.ones(int(rvok.sum())), "count")
    c["lamost_mrs_nights"] = _by_sid(cs, np.asarray(mrs["source_id"])[rvok], np.asarray(mrs["lmjd"])[rvok], "nunique")

    ap_star = Table.read(a.apogee / "apogee_star.ecsv", format="ascii.ecsv")
    c["apogee_nvisits"] = np.nan_to_num(_by_sid(cs, ap_star["gaiaedr3_source_id"], _col(ap_star, "nvisits"), "max")).astype(int)
    gal = Table.read(a.galah / "galah_allstar.ecsv", format="ascii.ecsv")
    c["galah"] = _by_sid(cs, gal["gaiadr3_source_id"], np.ones(len(gal)), "count") > 0
    c["multi_epoch_rv"] = (np.asarray(c["lamost_mrs_nights"]) >= 2) | (np.asarray(c["apogee_nvisits"]) >= 2)
    c["any_rv"] = (np.asarray(c["gaia_rv_amplitude"]) | (np.asarray(c["lamost_mrs_n_rv"]) > 0)
                   | (np.asarray(c["lamost_lrs_n_obs"]) > 0) | (np.asarray(c["apogee_nvisits"]) > 0) | np.asarray(c["galah"]))

    edges = np.asarray(yaml.safe_load(a.bins_config.read_text())["derived"]["m2_app_edges_msun"], float)
    flags = ("galex_nuv", "galex_fuv", "galex_ambiguous", "gaia_rv_amplitude", "multi_epoch_rv", "any_rv", "galah")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (np.asarray(c["m2_app"]) >= lo) & (np.asarray(c["m2_app"]) < hi)
        n = int(s.sum())
        rows.append({"m2_app_lo": float(lo), "m2_app_hi": float(hi), "n": n,
                     "n_large_a": int(np.sum(s & (np.asarray(c["a_size"]) > a.a_min))),
                     **{f"frac_{f}": (round(float(np.mean(np.asarray(c[f])[s])), 3) if n else None) for f in flags}})
    summary = {"issue": 641, "n_c1_ok": int(ok.sum()), "n_candidates": int(len(c)),
               "cuts": {"m2_min": a.m2_min, "a_min": a.a_min, "m2_min_large_a": a.m2_min_large_a},
               "galex": {"epoch": a.galex_epoch, "radius_arcsec": a.galex_radius}, "m2_app_edges": edges.tolist(),
               "by_m2_app_bin": rows, "overall": {f"frac_{f}": round(float(np.mean(np.asarray(c[f]))), 3) for f in flags},
               "inputs": {"m2app": str(a.m2app), "mass_recovery": nu_attr, **{k: str(getattr(a, k)) for k in DEFAULTS}}}
    a.out.mkdir(parents=True, exist_ok=True)
    c.write(a.out / "fl_candidates_dr3.ecsv", format="ascii.ecsv", overwrite=True)
    (a.out / "fl_candidates_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n_c1_ok", "n_candidates", "overall")}, indent=1))
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
