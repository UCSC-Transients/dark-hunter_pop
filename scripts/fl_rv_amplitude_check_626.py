#!/usr/bin/env python3
"""FL-T3 (#626; RUNG5_SPEC §11.3): Gaia DR3 RV amplitude vs the K1 the published astrometric orbit predicts.

A first look (report only; no thresholds are adopted) at the RV channel of the follow-up layer, using data already on
disk: the approved bright RV-stats snapshot (``scripts/fetch_rv_stats_391.py``) and the #390 ``published_orbits.h5``.

* **Noise floor.** ``rv_amplitude_robust`` of the looks-single parent rows (sample ``parent``) per ``grvs_mag`` bin:
  median and 90th percentile.
* **Prediction.** For each real orbit host (sample ``real_orbit``) with a published orbit:
  K1 = 29.78 (a0/ϖ) sin i / (P_yr √(1 − e²)) km/s, with (a0, i) from the Thiele–Innes elements through
  ``injection_test.campbell_from_thiele_innes`` (gaiamock). For a dark companion a0/ϖ is the primary's orbit, so K1 is
  exact; a luminous companion dilutes a0, so K1 is a lower bound.
* **Comparison.** r = rv_amplitude_robust / (2 K1) for hosts whose 2 K1 exceeds ``--detect-factor`` × the floor's 90th
  percentile at their G_RVS ("RV-detectable"); quantiles of r and the fraction with an amplitude below the floor's 90th
  percentile, by solution type and by photocentre size A = (a0/ϖ)/P_yr^(2/3) (edges 0, 0.5, 5; rung-3 axis).
  An amplitude far below 2 K1 is a candidate refutation (spurious or wrong a0); r ≈ 1 is consistent. The RVS epoch
  sampling is not modelled here, so r < 1 is expected for orbits the epochs under-sample (FL-Q6 is the forward model).

Usage::
    PYTHONPATH=src .venv/bin/python scripts/fl_rv_amplitude_check_626.py --out output/followup/rv_amplitude_626
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

K1_CONST_KMS = 29.78  # 2π AU / yr in km/s
GRVS_EDGES = np.array([4.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
A_EDGES = np.array([0.0, 0.5, 5.0])


def k1_pred_kms(a0_mas: np.ndarray, parallax_mas: np.ndarray, period_day: np.ndarray, ecc: np.ndarray, inc_rad: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        a_au = a0_mas / parallax_mas
        p_yr = period_day / 365.25
        return K1_CONST_KMS * a_au * np.abs(np.sin(inc_rad)) / (p_yr * np.sqrt(1.0 - ecc**2))


def floor_by_bin(grvs: np.ndarray, amp: np.ndarray, edges: np.ndarray = GRVS_EDGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(median, 90th percentile, n) of the single-star amplitude per G_RVS bin (NaN where empty)."""
    med, p90, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = np.isfinite(amp) & (grvs >= lo) & (grvs < hi)
        n.append(int(s.sum()))
        med.append(float(np.median(amp[s])) if s.any() else np.nan)
        p90.append(float(np.percentile(amp[s], 90)) if s.any() else np.nan)
    return np.array(med), np.array(p90), np.array(n)


def _col(t, name: str) -> np.ndarray:
    return np.ma.filled(np.ma.asarray(t[name], float), np.nan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rv-stats", type=Path, default=Path("data/external_catalogs/20261022T181049Z_dr3_rv_stats_bright/table.ecsv"))
    ap.add_argument("--published-orbits", type=Path, default=Path("output/gate390/published_orbits.h5"))
    ap.add_argument("--detect-factor", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import h5py
    from astropy.table import Table

    from darkhunter_pop import injection_test as it
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    t = Table.read(a.rv_stats, format="ascii.ecsv")
    sample = np.asarray(t["sample"]).astype(str)
    sid = np.asarray(t["source_id"], np.int64)
    grvs, amp = _col(t, "grvs_mag"), _col(t, "rv_amplitude_robust")
    par = sample == "parent"
    med, p90, nfl = floor_by_bin(grvs[par], amp[par])

    with h5py.File(a.published_orbits, "r") as h:
        p = h["published"]
        psid = p["source_id"][()].astype(np.int64)
        cols = {k: p[k][()] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes",
                                      "parallax", "period", "eccentricity")}
        ptype = p["nss_solution_type"][()].astype(str)
    pos = {int(s): i for i, s in enumerate(psid)}
    orb = np.flatnonzero(sample == "real_orbit")
    idx = np.array([pos.get(int(sid[i]), -1) for i in orb])
    keep = idx >= 0
    orb, idx = orb[keep], idx[keep]
    gm = import_gaiamock_mod()
    a0 = np.empty(idx.size)
    inc = np.empty(idx.size)
    for j, k in enumerate(idx):
        a0[j], _, _, inc[j] = it.campbell_from_thiele_innes(gm, cols["a_thiele_innes"][k], cols["b_thiele_innes"][k],
                                                            cols["f_thiele_innes"][k], cols["g_thiele_innes"][k])
    plx, per, ecc = (np.asarray(cols[c][idx], float) for c in ("parallax", "period", "eccentricity"))
    k1 = k1_pred_kms(a0, plx, per, ecc, inc)
    with np.errstate(divide="ignore", invalid="ignore"):
        big_a = (a0 / plx) / (per / 365.25) ** (2.0 / 3.0)
        ratio = amp[orb] / (2.0 * k1)
    b = np.clip(np.searchsorted(GRVS_EDGES, grvs[orb], side="right") - 1, 0, GRVS_EDGES.size - 2)
    in_range = (grvs[orb] >= GRVS_EDGES[0]) & (grvs[orb] < GRVS_EDGES[-1])
    fl90 = np.where(in_range, p90[b], np.nan)
    det = in_range & np.isfinite(amp[orb]) & np.isfinite(k1) & (2.0 * k1 > a.detect_factor * fl90)
    flat = det & (amp[orb] < fl90)

    def summ(sel: np.ndarray) -> dict:
        r = ratio[sel & det]
        return {"n_hosts": int(sel.sum()), "n_rv_detectable": int((sel & det).sum()),
                "ratio_q16_50_84": [float(x) for x in np.nanpercentile(r, [16, 50, 84])] if r.size else None,
                "frac_flat_among_detectable": float((sel & flat).sum() / max(int((sel & det).sum()), 1))}

    types = ptype[idx]
    report = {
        "issue": 626, "detect_factor": a.detect_factor, "grvs_edges": GRVS_EDGES.tolist(),
        "floor_median_kms": med.tolist(), "floor_p90_kms": p90.tolist(), "floor_n": nfl.tolist(),
        "n_real_orbit_with_published": int(orb.size),
        "all": summ(np.ones(orb.size, bool)),
        "by_type": {ty: summ(types == ty) for ty in sorted(set(types))},
        "by_A": {f"{lo:g}-{hi:g}": summ((big_a >= lo) & (big_a < hi)) for lo, hi in zip(A_EDGES[:-1], A_EDGES[1:])},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rv_amplitude_report.json").write_text(json.dumps(report, indent=1))
    np.savez(a.out / "rv_amplitude_rows.npz", source_id=sid[orb], k1_pred_kms=k1, amp_kms=amp[orb], ratio=ratio,
             a_size=big_a, grvs=grvs[orb], detectable=det, flat=flat, nss_solution_type=types)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
