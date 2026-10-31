#!/usr/bin/env python3
"""#597 (MOCK_POPULATION_SPEC §12.41, EPOCH_MODEL_SPEC §8.16): calibrate P(AstroSpectroSB1 | observables) on DR3 C1 orbits.

Statistical routing for the mock's astrometric orbits: which of them DR3 would have published as AstroSpectroSB1
(combined astrometric + SB1 solution) rather than Orbital. Only quantities available identically on both sides are used:

* predicted G_RVS = G − p(BP−RP), p a cubic fitted here on DR3 stars that have ``grvs_mag`` (the on-disk bright RV
  statistics snapshot ``data/external_catalogs/20261022T181049Z_dr3_rv_stats_bright``; no download);
* BP−RP (observed), log10 P, e;
* predicted K1 from the astrometric orbit, as if the photocentre orbit were the primary's:
  K1 = 29.78 km/s · (a0/ϖ) sin i / (P_yr √(1 − e²)) (a0 and i from the Thiele–Innes elements on the DR3 side, from the
  fitted cascade on the mock side; the same formula).

Model: logit P = β0 + hinge basis in G_RVS (knots in config) + optional terms, chosen by forward selection on BIC
(the spuriousness-model convention, ΔBIC > 6). Fit on DR3 Orbital + AstroSpectroSB1 rows with P ≤ 830.4 d and
G < 15; checked by type fraction per G, P, K1 and colour bin. Writes ``routing.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
RV = D / "data/external_catalogs/20261022T181049Z_dr3_rv_stats_bright/table.ecsv"
KNOTS = [8.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
P_MAX = 830.4


def grvs_poly(path: Path) -> list[float]:
    from astropy.table import Table

    t = Table.read(path, format="ascii.ecsv")
    g, gr, c = (np.ma.filled(np.ma.asarray(t[k], float), np.nan) for k in ("phot_g_mean_mag", "grvs_mag", "bp_rp"))
    ok = np.isfinite(g) & np.isfinite(gr) & np.isfinite(c) & (c > 0.2) & (c < 3.5)
    cf = np.polyfit(c[ok], (g - gr)[ok], 3)
    res = (g - gr)[ok] - np.polyval(cf, c[ok])
    return [float(x) for x in cf], float(np.std(res)), int(ok.sum())


def k1_pred(a0_mas, plx_mas, period_d, ecc, cos_i):
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_i = np.sqrt(np.clip(1 - cos_i**2, 0, 1))
        return 29.78 * (a0_mas / plx_mas) * sin_i / ((period_d / 365.25) * np.sqrt(np.clip(1 - ecc**2, 1e-6, 1)))


def design(x: dict, terms: list[str], knots: list[float]) -> np.ndarray:
    cols = [np.ones(x["grvs"].size), x["grvs"]] + [np.maximum(0.0, x["grvs"] - k) for k in knots]
    for t in terms:
        cols.append(x[t])
    return np.column_stack(cols)


def fit_logit(X, y):
    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)) + 0.5 * 1e-4 * float(b @ b)

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y) + 1e-4 * b

    r = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B", options={"maxiter": 5000})
    return r.x, r.fun


def real_covariates() -> dict:
    import h5py

    with h5py.File(D / "output/gate390/published_orbits.h5", "r") as f:
        P = f["published"]
        d = {k: P[k][()] for k in ("source_id", "g_mag", "period", "eccentricity", "parallax", "a_thiele_innes", "b_thiele_innes",
                                   "f_thiele_innes", "g_thiele_innes")}
        st = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in P["nss_solution_type"][()]])
    with h5py.File(D / "data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns/columns.h5", "r") as f:
        cmap = dict(zip(f["source_id"][()].tolist(), f["bp_rp"][()].tolist()))
    A_, B_, F_, G_ = (d[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0 = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))
    cos_i = v / a0**2
    bp = np.array([cmap.get(int(s), np.nan) for s in d["source_id"]], float)
    return {"g": d["g_mag"], "bp_rp": bp, "period": d["period"], "e": d["eccentricity"], "k1": k1_pred(a0, d["parallax"], d["period"], d["eccentricity"], cos_i),
            "type": st, "sid": d["source_id"]}


def covariates(g, bp_rp, period, e, k1, poly):
    c = np.clip(bp_rp, 0.2, 3.5)
    return {"grvs": g - np.polyval(poly, c), "bp_rp": c, "bp_rp2": c**2, "logP": np.log10(period), "e": e,
            "logK1": np.log10(np.clip(k1, 1e-3, None))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dbic", type=float, default=6.0)
    a = ap.parse_args(argv)
    poly, poly_rms, poly_n = grvs_poly(RV)
    r = real_covariates()
    sel = np.isin(r["type"], ["Orbital", "AstroSpectroSB1"]) & (r["period"] <= P_MAX) & (r["g"] < 15) & np.isfinite(r["bp_rp"]) & np.isfinite(r["k1"])
    y = (r["type"][sel] == "AstroSpectroSB1").astype(float)
    x = covariates(r["g"][sel], r["bp_rp"][sel], r["period"][sel], r["e"][sel], r["k1"][sel], poly)
    n = y.size
    chosen: list[str] = []
    b, nll = fit_logit(design(x, chosen, KNOTS), y)
    trail = [{"terms": [], "bic": 2 * nll + len(b) * np.log(n)}]
    pool = ["logK1", "bp_rp", "bp_rp2", "logP", "e"]
    while True:
        best = None
        for t in pool:
            if t in chosen:
                continue
            bb, nl = fit_logit(design(x, chosen + [t], KNOTS), y)
            bic = 2 * nl + len(bb) * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, t, bb, nl)
        if best is None or trail[-1]["bic"] - best[0] < a.dbic:
            break
        chosen.append(best[1])
        b, nll = best[2], best[3]
        trail.append({"terms": list(chosen), "bic": best[0], "dbic": trail[-1]["bic"] - best[0]})
    X = design(x, chosen, KNOTS)
    p = 1 / (1 + np.exp(-(X @ b)))
    checks = {}
    for key, xx, edges in (("G", r["g"][sel], [6, 9, 10, 11, 12, 13, 14, 15]), ("logP", x["logP"], [0, 1.5, 2, 2.5, 2.75, 3]),
                           ("logK1", x["logK1"], [-1, 0, 0.5, 1, 1.5, 3]), ("bp_rp", x["bp_rp"], [0, 0.6, 0.8, 1.0, 1.3, 4]),
                           ("grvs", x["grvs"], [5, 9, 10, 11, 12, 13, 16])):
        checks[key] = [{"bin": [lo, hi], "n": int(((xx >= lo) & (xx < hi)).sum()), "dr3_fraction": float(y[(xx >= lo) & (xx < hi)].mean()) if ((xx >= lo) & (xx < hi)).any() else float("nan"),
                        "model_fraction": float(p[(xx >= lo) & (xx < hi)].mean()) if ((xx >= lo) & (xx < hi)).any() else float("nan")}
                       for lo, hi in zip(edges[:-1], edges[1:])]
    res = {"grvs_poly_g_minus_grvs_vs_bp_rp": poly, "grvs_poly_rms": poly_rms, "grvs_poly_n": poly_n, "knots_grvs": KNOTS, "terms": chosen,
           "beta": [float(v) for v in b], "n_rows": n, "trail": trail, "checks": checks, "p_max_days": P_MAX}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print("terms", chosen, "poly", np.round(poly, 4), "rms", round(poly_rms, 3))
    for k, v in checks.items():
        print(k, " | ".join(f"{c['bin'][0]}-{c['bin'][1]}: {c['dr3_fraction']:.3f}/{c['model_fraction']:.3f} (n{c['n']})" for c in v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
