#!/usr/bin/env python3
"""Calibrate the #400 epoch model after Ryan's decisions (E3, E4, E6, E7, N2).

docs/EPOCH_MODEL_SPEC.md §4. Inputs: the epoch-count snapshot
(``scripts/fetch_epoch_count_sample.py``), ``output/gate400/gost_pixels.npz``
(``scripts/measure_epoch_counts_400.py gost``), the #390 ``published_orbits.h5`` (NSS
significance, period, F2), and, for ``cluster``, the epoch-time snapshot
(``scripts/fetch_epoch_photometry_400.py``).

``keep``  (E3, E6, E7)
    Quasi-Poisson GLM, log link, offset log(GOST transits after the gaps):
    ``log E[k] = log n + poly_d(G) + sum Y_lm(l, b) [+ c ln(RUWE/1.4) for NSS] [+ delta for random]``.
    Compares binned vs polynomial G (QAIC with a common dispersion), picks ``ell_max``
    by 5-fold cross-validation over sky blocks (HEALPix nside 4), checks aliasing of the
    GOST grid (refit with a bilinearly interpolated GOST denominator; project the grid
    error proxy log(n_nearest / n_interp) on the same harmonics), and tabulates the
    circularity checks (keep vs s / threshold and vs RUWE). Writes ``keep_fit.json``.
``noise``  (N2)
    Published c (Halbwachs Eq. 2, from F2 and n_good_obs_al - 12) for NSS Orbital stars
    vs G; knots of ``r2 = median(c^2) - 1`` for G < 13. Writes ``noise_fit.json``.
``cluster``  (E4)
    See ``cmd_cluster``. Writes ``cluster_fit.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import gammaln

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import epoch_model as em  # noqa: E402
from darkhunter_pop.cascade_replay import inflation_factor_from_f2  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

NSIDE = 16
RUWE_REF = 1.4  # NSS input selection threshold (Halbwachs et al. 2023 §1.2): own-loss reference


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _gost_counts(gost: Path, cfg: em.EpochModelConfig) -> np.ndarray:
    g = np.load(gost)
    keep = ~em.in_gaps(g["t_mid"], em.gap_intervals_jd(cfg))
    return np.add.reduceat(keep.astype(float), g["offsets"][:-1])


def _load(snapshot: Path, name: str, npix: np.ndarray) -> dict[str, np.ndarray]:
    import healpy as hp

    with h5py.File(snapshot / f"{name}.h5", "r") as f:
        d = {k: f[k][:] for k in f}
    th, ph = np.radians(90.0 - d["dec"]), np.radians(d["ra"])
    d["n"] = npix[hp.ang2pix(NSIDE, th, ph)]
    p, w = hp.get_interp_weights(NSIDE, th, ph)
    d["n_interp"] = (npix[p] * w).sum(0)
    d["k"] = d["astrometric_matched_transits"].astype(float)
    d["G"] = d["phot_g_mean_mag"].astype(float)
    return d


def _nss_extra(d: dict[str, np.ndarray], published: Path) -> dict[str, np.ndarray]:
    with h5py.File(published, "r") as f:
        sid = f["published/source_id"][:]
        typ = f["published/nss_solution_type"][:]
        sig, per, f2 = (f[f"published/{k}"][:] for k in ("significance", "period", "goodness_of_fit"))
    key = {(int(a), b): i for i, (a, b) in enumerate(zip(sid, typ))}
    i = np.array([key[(int(a), b)] for a, b in zip(d["source_id"], d["nss_solution_type"])])
    d["sig"], d["per"], d["f2"] = sig[i], per[i], f2[i]
    d["s_rel"] = sig[i] / np.maximum(5.0, 158.0 / np.sqrt(per[i]))  # El-Badry 2024 Eq. 18-19
    return d


# ---------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------


def glm_poisson(k: np.ndarray, n: np.ndarray, X: np.ndarray, maxit: int = 60) -> dict[str, Any]:
    """Poisson GLM, log link, offset ``log n``, intercept prepended; IRLS.

    Returns beta, the dispersion-scaled covariance, the Poisson log-likelihood, the
    Pearson dispersion phi, the number of parameters and the fitted means.
    """
    m = n > 0
    k, n, X = k[m], n[m], np.column_stack([np.ones(m.sum()), X[m]])
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(k.sum() / n.sum())
    for _ in range(maxit):
        mu = n * np.exp(X @ beta)
        z = X @ beta + (k - mu) / mu
        new = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (mu * z))
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break
    mu = n * np.exp(X @ beta)
    ll = float(np.sum(k * np.log(mu) - mu - gammaln(k + 1)))
    phi = float(np.sum((k - mu) ** 2 / mu) / (k.size - X.shape[1]))
    cov = np.linalg.inv(X.T @ (X * mu[:, None])) * phi
    return {"beta": beta, "cov": cov, "ll": ll, "phi": phi, "p": int(X.shape[1]), "N": int(k.size), "mu": mu, "mask": m}


def _deviance(k: np.ndarray, mu: np.ndarray) -> float:
    t = np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / mu), 0.0)
    return float(2.0 * np.sum(t - (k - mu)))


def _cont(deg: int, lmax: int, lo: float, hi: float) -> em.ContinuousLossConfig:
    return em.ContinuousLossConfig(g_clip=(lo, hi), g_ref=14.0, g_scale=4.0, coef_g=(0.0,) * (deg + 1),
                                   sky_lmax=lmax, coef_sky=(0.0,) * ((lmax + 1) ** 2 - 1))


def _bins(G: np.ndarray, edges: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(edges, G, side="right") - 1, 0, len(edges) - 2)
    X = np.zeros((G.size, len(edges) - 1))
    X[np.arange(G.size), i] = 1.0
    return X[:, 1:]


def cmd_keep(args: argparse.Namespace) -> None:
    cfg = em.epoch_model_config_from_mapping(load_config().dr3.epoch_model)
    snap = Path(args.snapshot)
    npix = _gost_counts(Path(args.gost), cfg)
    nss = _nss_extra(_load(snap, "nss", npix), Path(args.published))
    rnd = _load(snap, "random", npix)
    rsel = rnd["ipd_frac_multi_peak"] <= 2  # Halbwachs (b) part available in the snapshot (MP-Q3)
    lo, hi = args.g_clip
    out: dict[str, Any] = {"n_nss": int(nss["k"].size), "n_random_ipd_le_2": int(rsel.sum()), "ruwe_ref": RUWE_REF}

    # ---- circularity tables (E6)
    def keep_of(d: dict[str, np.ndarray], m: np.ndarray) -> float:
        return float(d["k"][m].sum() / d["n"][m].sum())

    circ = []
    for glo, ghi in ((3, 11), (11, 13), (13, 15), (15, 17)):
        mg = (nss["G"] >= glo) & (nss["G"] < ghi)
        mr = (rnd["G"] >= glo) & (rnd["G"] < ghi) & rsel
        row: dict[str, Any] = {"g": [glo, ghi], "random_ipd_le_2": keep_of(rnd, mr), "nss": keep_of(nss, mg)}
        row["vs_s_over_threshold"] = {f"{a}-{b}": [keep_of(nss, mg & (nss["s_rel"] >= a) & (nss["s_rel"] < b)),
                                                   int((mg & (nss["s_rel"] >= a) & (nss["s_rel"] < b)).sum())]
                                      for a, b in ((1, 1.5), (1.5, 2), (2, 4), (4, 1e9))}
        row["vs_ruwe"] = {f"{a}-{b}": [keep_of(nss, mg & (nss["ruwe"] >= a) & (nss["ruwe"] < b)),
                                       int((mg & (nss["ruwe"] >= a) & (nss["ruwe"] < b)).sum())]
                          for a, b in ((1.4, 2), (2, 3), (3, 5), (5, 1e9))}
        circ.append(row)
    out["circularity"] = circ

    # ---- NSS-only model selection (E7 and E3)
    kN, nN, GN, lN, bN = nss["k"], nss["n"], nss["G"], nss["l"], nss["b"]
    lrN = np.log(np.maximum(nss["ruwe"], RUWE_REF) / RUWE_REF)
    sel: dict[str, Any] = {}
    edges = np.array([3, 11, 12, 13, 14, 15, 16, 17, 30.0])
    cands = {"bins8": _bins(GN, edges)}
    for d in range(1, 6):
        cands[f"poly{d}"] = em.g_polynomial_basis(GN, _cont(d, 0, 6.0, 17.0))
    fits = {k_: glm_poisson(kN, nN, X) for k_, X in cands.items()}
    phi_ref = max(fits.values(), key=lambda r: r["p"])["phi"]
    sel["g_basis_nss"] = {k_: {"p": r["p"], "ll": r["ll"], "phi": r["phi"], "qaic": -2 * r["ll"] / phi_ref + 2 * r["p"]}
                          for k_, r in fits.items()}
    # sky ell_max by sky-blocked CV
    import healpy as hp
    blk = hp.ang2pix(4, np.radians(90 - nss["dec"]), np.radians(nss["ra"]))
    fold = np.random.default_rng(args.seed).permutation(hp.nside2npix(4)) % 5
    f = fold[blk]
    cv = {}
    for lm in range(0, args.lmax_cv + 1):
        X = np.column_stack([em.g_polynomial_basis(GN, _cont(args.deg, 0, 6.0, 17.0)),
                             em.real_sph_harm_galactic(lN, bN, lm), lrN])
        D = 0.0
        for i in range(5):
            tr = f != i
            r = glm_poisson(kN[tr], nN[tr], X[tr])
            mu = nN[~tr] * np.exp(np.column_stack([np.ones((~tr).sum()), X[~tr]]) @ r["beta"])
            D += _deviance(kN[~tr], mu)
        cv[lm] = D
    sel["sky_cv_deviance"] = cv
    lmax = int(min(cv, key=cv.get)) if args.lmax is None else int(args.lmax)
    sel["sky_lmax_chosen"] = lmax

    # ---- aliasing check (E3)
    alias: dict[str, Any] = {}
    sky_parts = {}
    for den in ("n", "n_interp"):
        X = np.column_stack([em.g_polynomial_basis(GN, _cont(args.deg, 0, 6.0, 17.0)),
                             em.real_sph_harm_galactic(lN, bN, max(lmax, 2)), lrN])
        r = glm_poisson(kN, nss[den], X)
        nsky = (max(lmax, 2) + 1) ** 2 - 1
        sky = em.real_sph_harm_galactic(lN, bN, max(lmax, 2)) @ r["beta"][1 + args.deg:1 + args.deg + nsky]
        sky_parts[den] = sky
        alias[f"denominator_{den}"] = {"phi": r["phi"], "rms_sky_log_keep": float(np.std(sky))}
    y = np.log(nss["n"] / nss["n_interp"])
    Y = em.real_sph_harm_galactic(lN, bN, max(lmax, 2))
    bgrid = np.linalg.lstsq(np.column_stack([np.ones(y.size), Y]), y, rcond=None)[0]
    grid_fit = Y @ bgrid[1:]
    alias["grid_proxy_rms_total"] = float(np.std(y))
    alias["grid_proxy_rms_on_harmonics"] = float(np.std(grid_fit))
    alias["corr_sky_nearest_vs_interp"] = float(np.corrcoef(sky_parts["n"], sky_parts["n_interp"])[0, 1])
    alias["corr_sky_vs_minus_grid_proxy"] = float(np.corrcoef(sky_parts["n"], -grid_fit)[0, 1])
    out["aliasing"] = alias

    # ---- adopted joint fit: NSS level, random stars (offset) for the faint shape
    cat = lambda key: np.r_[nss[key], rnd[key][rsel]]  # noqa: E731
    k, n, G, L, B = cat("k"), cat("n"), cat("G"), cat("l"), cat("b")
    isr = np.r_[np.zeros(nss["k"].size), np.ones(rsel.sum())]
    lr = np.r_[lrN, np.zeros(rsel.sum())]
    joint = {}
    for d in (3, 4, 5):
        X = np.column_stack([em.g_polynomial_basis(G, _cont(d, 0, lo, hi)), em.real_sph_harm_galactic(L, B, lmax), lr, isr])
        joint[d] = glm_poisson(k, n, X)
    Xb = np.column_stack([_bins(G, np.array([3, 11, 12, 13, 14, 15, 16, 17, 17.5, 18, 18.5, 30.0])),
                          em.real_sph_harm_galactic(L, B, lmax), lr, isr])
    jb = glm_poisson(k, n, Xb)
    phi_ref = max(list(joint.values()) + [jb], key=lambda r: r["p"])["phi"]
    sel["joint_g_basis"] = {**{f"poly{d}": {"p": r["p"], "ll": r["ll"], "qaic": -2 * r["ll"] / phi_ref + 2 * r["p"]} for d, r in joint.items()},
                            "bins11": {"p": jb["p"], "ll": jb["ll"], "qaic": -2 * jb["ll"] / phi_ref + 2 * jb["p"]}}
    deg = min(joint, key=lambda d: -2 * joint[d]["ll"] / phi_ref + 2 * joint[d]["p"]) if args.deg_joint is None else args.deg_joint
    r = joint[deg]
    nsky = (lmax + 1) ** 2 - 1
    beta, se = r["beta"], np.sqrt(np.diag(r["cov"]))
    out["selection"] = sel
    out["adopted"] = {
        "g_clip": [lo, hi], "g_ref": 14.0, "g_scale": 4.0, "degree": int(deg), "sky_lmax": lmax,
        "coef_g": [float(x) for x in beta[:deg + 1]], "coef_sky": [float(x) for x in beta[deg + 1:deg + 1 + nsky]],
        "ruwe_coef_ln_ruwe_over_1p4": float(beta[-2]), "ruwe_coef_se": float(se[-2]),
        "random_offset_log": float(beta[-1]), "random_offset_se": float(se[-1]), "phi": r["phi"],
        "coef_g_se": [float(x) for x in se[:deg + 1]], "coef_sky_se": [float(x) for x in se[deg + 1:deg + 1 + nsky]],
    }
    # observed / model per G bin and sample, at the adopted fit
    Xall = np.column_stack([np.ones(G.size), em.g_polynomial_basis(G, _cont(deg, 0, lo, hi)),
                            em.real_sph_harm_galactic(L, B, lmax), lr, isr])
    mu_all = n * np.exp(Xall @ beta)
    resid = []
    for glo, ghi in ((6, 9), (9, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 17.5), (17.5, 18), (18, 18.5), (18.5, 19)):
        for lab, ms in (("nss", isr == 0), ("random", isr == 1)):
            m = (G >= glo) & (G < ghi) & ms & (n > 0)
            if m.sum() >= 30:
                resid.append({"g": [glo, ghi], "sample": lab, "n": int(m.sum()),
                              "observed_over_model": float(k[m].sum() / mu_all[m].sum())})
    out["residuals_vs_g"] = resid
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "keep_fit.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"selection": sel, "aliasing": alias, "adopted": out["adopted"]}, indent=1, default=float))


def cmd_noise(args: argparse.Namespace) -> None:
    snap = Path(args.snapshot)
    with h5py.File(snap / "nss.h5", "r") as f:
        d = {k: f[k][:] for k in ("source_id", "nss_solution_type", "astrometric_n_good_obs_al", "phot_g_mean_mag")}
    with h5py.File(args.published, "r") as f:
        sid, typ, f2 = f["published/source_id"][:], f["published/nss_solution_type"][:], f["published/goodness_of_fit"][:]
    key = {(int(a), b): i for i, (a, b) in enumerate(zip(sid, typ))}
    i = np.array([key[(int(a), b)] for a, b in zip(d["source_id"], d["nss_solution_type"])])
    orb = d["nss_solution_type"] == b"Orbital"
    c = inflation_factor_from_f2(f2[i], d["astrometric_n_good_obs_al"] - 12.0)
    G = d["phot_g_mean_mag"]
    edges = np.asarray(args.g_edges, dtype=float)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = orb & (G >= a) & (G < b) & np.isfinite(c)
        rows.append({"g": [float(a), float(b)], "n": int(m.sum()), "median_f2": float(np.median(f2[i][m])),
                     "median_c": float(np.median(c[m])), "median_c2_minus_1": float(np.median(c[m] ** 2 - 1)),
                     "g_median": float(np.median(G[m]))})
    bright = [r_ for r_ in rows if r_["g"][1] <= args.g_max]
    out = {"bins": rows, "g_max": args.g_max,
           "knots_g": [r_["g_median"] for r_ in bright],
           "knots_r2": [max(0.0, r_["median_c2_minus_1"]) for r_ in bright],
           "note": "r2 = median(c^2) - 1 of published Orbital fits; gaiamock's own c is ~1.0 (#399), so r2 is the excess"}
    (Path(args.out) / "noise_fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_u0_table(args: argparse.Namespace) -> None:
    """u0_mock(G): the 41st percentile of mock single-star UWE per G bin (Lindegren 2018 TN LL-124).

    Input: ``validate_epoch_model_v2_400.py u0`` output. Writes the CSV calibration table
    (comment header with provenance) and ``u0_fit.json`` (percentiles, medians, counts).
    No smoothing beyond the bins: the mock has no colour axis and enough stars per bin
    (the binomial error of a 41st percentile with N = 600 is ~0.2% of u0).
    """
    import datetime as dt
    import hashlib

    src = Path(args.singles)
    meta = json.loads((src.parent / "u0_meta.json").read_text())
    edges = np.asarray(meta["edges"], dtype=float)
    rows = [json.loads(x) for x in src.read_text().splitlines()]
    b = np.array([r["bin"] for r in rows])
    u = np.array([r["uwe"] for r in rows], dtype=float)
    centres, p41, med, n = [], [], [], []
    for ib in range(edges.size - 1):
        m = (b == ib) & np.isfinite(u)
        if m.sum() < 50:
            continue
        centres.append(0.5 * (edges[ib] + edges[ib + 1]))
        p41.append(float(np.percentile(u[m], args.percentile)))
        med.append(float(np.median(u[m])))
        n.append(int(m.sum()))
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    out = Path(args.table)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# u0_mock(G) for DR3-like RUWE = UWE / u0 in the pop mock (#400 N2-u0; docs/EPOCH_MODEL_SPEC.md §8.8)",
        f"# statistic: {args.percentile:g}th percentile of mock single-star UWE per G bin (Lindegren 2018, GAIA-C3-TN-LU-LL-124 §4)",
        "# mock singles: gaiamock_mod predict_astrometry_single_source at random gaia_source positions,",
        "#   v2 epoch model + bright per-CCD excess noise (dr3.epoch_model), UWE from gaiamock check_ruwe",
        f"# source: {src.name} sha256 {digest}; {meta['n_per_bin']} stars per {edges[1] - edges[0]:g}-mag bin, seed {meta['seed']}",
        f"# created {dt.datetime.now(dt.timezone.utc).isoformat()} by scripts/calibrate_epoch_model_400.py u0-table",
        "# colour axis collapsed: gaiamock's per-CCD noise has no colour dependence",
    ]
    lines = header + ["g,u0"] + [f"{c:.3f},{v:.5f}" for c, v in zip(centres, p41)]
    out.write_text("\n".join(lines) + "\n")
    table_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    res = {"g": centres, "u0_p41": p41, "median": med, "n": n, "table": str(out), "table_sha256": table_sha,
           "singles_sha256": digest, "percentile": args.percentile}
    (Path(args.out) / "u0_fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("table", "table_sha256")}, indent=1))
    for c, a, m_, k_ in zip(centres, p41, med, n):
        print(f"{c:6.3f} p41={a:.4f} median={m_:.4f} n={k_}")


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--snapshot", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400"))
    common.add_argument("--gost", default=str(P / "output/gate400/gost_pixels.npz"))
    common.add_argument("--published", default=str(P / "output/gate390/published_orbits.h5"))
    common.add_argument("--out", default=str(P / "output/gate400/v2"))
    p = sub.add_parser("keep", parents=[common])
    p.add_argument("--g-clip", nargs=2, type=float, default=[6.0, 19.0])
    p.add_argument("--deg", type=int, default=3, help="G degree for the NSS-only sky/aliasing fits")
    p.add_argument("--deg-joint", type=int, default=None)
    p.add_argument("--lmax-cv", type=int, default=8)
    p.add_argument("--lmax", type=int, default=None, help="override the CV choice")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_keep)
    p = sub.add_parser("noise", parents=[common])
    p.add_argument("--g-edges", nargs="+", type=float, default=[5, 9, 10, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 17])
    p.add_argument("--g-max", type=float, default=13.0)
    p.set_defaults(func=cmd_noise)
    p = sub.add_parser("u0-table", parents=[common])
    p.add_argument("--singles", default=str(P / "output/gate400/u0/u0_singles.jsonl"))
    p.add_argument("--table", default=str(REPO / "config/epoch_model/dr3_ruwe_u0_mock.csv"))
    p.add_argument("--percentile", type=float, default=41.0)
    p.set_defaults(func=cmd_u0_table)
    args = parser.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
