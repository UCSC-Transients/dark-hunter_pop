#!/usr/bin/env python3
"""Calibrate the visibility-period loss model on the DR3 N_vis distribution (#432).

docs/EPOCH_MODEL_SPEC.md §8.10. Input: ``output/gate432/vp_structure.h5``
(``vp_structure_432.py``: per star, the transits in each visibility period at the exact
position after the window and the published gaps; DR3 ``visibility_periods_used``).

Model, per star (no per-star lookup; every parameter is shared):

* the star is **degraded** with probability ``pi = expit(c0 + c1 x + c2 x^2 + c3 |sin beta| + c4 |sin b|)``,
  ``x = (clip(G, 6, 19) - 14) / 4``;
* each visibility period is dropped whole with probability ``q_bad = expit(e0)`` if the
  star is degraded, else ``q0 = expit(d0 + d1 x)``;
* independently, each transit is lost with ``p_ind = 1 - p_keep / (1 - qbar)``, where
  ``p_keep`` is the configured transit-keep model (continuous G + Galactic sky; random
  stars carry the fitted random offset) and ``qbar = (1 - pi) q0 + pi q_bad``, so the
  expected kept-transit fraction is unchanged;
* a visibility period counts when it is not dropped and at least one transit survives.

Likelihood: Poisson-binomial over the star's visibility periods, a two-component mixture,
of DR3 ``visibility_periods_used``. Random stars (all solution types) are fitted as is.
NSS stars are fitted truncated at N >= 12, because DR3 selected the NSS input on it
(Halbwachs et al. 2023 §1.2). Visibility-period splitting by lost transits is ignored in
the likelihood and checked by simulation afterwards (``--check``).

Variants (``--variants``) are compared by AIC: ``none`` (q0 only), ``G`` (pi(G)),
``G_beta``, ``G_beta_b``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import epoch_model as em  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

#: Galactic-latitude feature of the degraded fraction: |sin b| or exp(-|b| / B_SCALE_DEG)
B_FEATURE = "abs_sin"
B_SCALE_DEG = 10.0

PARAM_NAMES = ("c0", "c1", "c2", "c_beta", "c_b", "e0", "d0", "d1")
VARIANT_FREE = {
    "none": ("d0", "d1"),
    "G": ("c0", "c1", "c2", "e0", "d0", "d1"),
    "G_beta": ("c0", "c1", "c2", "c_beta", "e0", "d0", "d1"),
    "G_beta_b": ("c0", "c1", "c2", "c_beta", "c_b", "e0", "d0", "d1"),
}


def load(path: Path, cfg: em.EpochModelConfig, random_offset: float, n_random: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        for name in ("random", "nss"):
            g = f[name]
            d = {k: g[k][:] for k in g}
            n = d["source_id"].size
            idx = np.arange(n) if name == "nss" or n_random <= 0 else np.sort(rng.choice(n, min(n_random, n), replace=False))
            off = d["vp_offsets"]
            sizes = [d["vp_ntr"][off[i]:off[i + 1]] for i in idx]
            vmax = max(len(s) for s in sizes)
            M = np.zeros((idx.size, vmax), dtype=float)
            for r, s in enumerate(sizes):
                M[r, : s.size] = s
            G = d["phot_g_mean_mag"][idx].astype(float)
            pk = np.array([em.keep_probability(float(G[r]), cfg, l_deg=float(d["l"][i]), b_deg=float(d["b"][i]))
                           for r, i in enumerate(idx)])
            if name == "random":
                pk = np.minimum(1.0, pk * np.exp(random_offset))
            out[name] = {
                "M": M, "G": G, "x": (np.clip(G, 6.0, 19.0) - 14.0) / 4.0,
                "sbeta": np.abs(np.sin(np.radians(d["ecl_lat"][idx]))),
                "sb": (np.abs(np.sin(np.radians(d["b"][idx]))) if B_FEATURE == "abs_sin"
                       else np.exp(-np.abs(d["b"][idx]) / B_SCALE_DEG)),
                "beta": d["ecl_lat"][idx], "b": d["b"][idx], "pkeep": pk,
                "nvis": d["visibility_periods_used"][idx].astype(int), "nvis_exact": (M > 0).sum(1),
                "params_solved": d["astrometric_params_solved"][idx],
            }
    return out


def pb_pmf(S: np.ndarray, nmax: int) -> np.ndarray:
    """Poisson-binomial PMF per row of survival probabilities ``S`` (N x V); returns N x (nmax+1)."""
    pmf = np.zeros((S.shape[0], nmax + 1))
    pmf[:, 0] = 1.0
    for j in range(S.shape[1]):
        s = S[:, j:j + 1]
        pmf[:, 1:] = pmf[:, 1:] * (1 - s) + pmf[:, :-1] * s
        pmf[:, :1] = pmf[:, :1] * (1 - s)
    return pmf


def components(theta: dict[str, float], d: dict[str, Any]) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x = d["x"]
    pi = expit(theta["c0"] + theta["c1"] * x + theta["c2"] * x**2 + theta["c_beta"] * d["sbeta"] + theta["c_b"] * d["sb"])
    q0 = expit(theta["d0"] + theta["d1"] * x)
    qb = np.full_like(x, expit(theta["e0"]))
    qbar = (1 - pi) * q0 + pi * qb
    pind = np.clip(1.0 - d["pkeep"] / np.maximum(1e-9, 1.0 - qbar), 0.0, 0.999)
    M = d["M"]
    surv_t = 1.0 - pind[:, None] ** M  # >= 1 transit survives; 0 for padded VPs
    surv_t[M == 0] = 0.0
    comps = []
    for w, q in ((1 - pi, q0), (pi, qb)):
        comps.append((w, (1 - q)[:, None] * surv_t))
    return pi, comps


def nvis_pmf(theta: dict[str, float], d: dict[str, Any]) -> np.ndarray:
    nmax = d["M"].shape[1]
    _, comps = components(theta, d)
    return sum(w[:, None] * pb_pmf(S, nmax) for w, S in comps)


CELL_EDGES = {"G": (3, 13, 15, 16, 17, 18, 19.5), "abs_beta": (0, 15, 30, 45, 90.1), "abs_b": (0, 5, 10, 30, 90.1)}
LIKELIHOOD = "star"


def cell_ids(d: dict[str, Any]) -> np.ndarray:
    """Cell index on G x |beta| x |b| (CELL_EDGES) for the cell-level likelihood."""
    ig = np.clip(np.searchsorted(CELL_EDGES["G"], d["G"], side="right") - 1, 0, len(CELL_EDGES["G"]) - 2)
    ib = np.clip(np.searchsorted(CELL_EDGES["abs_beta"], np.abs(d["beta"]), side="right") - 1, 0, 3)
    il = np.clip(np.searchsorted(CELL_EDGES["abs_b"], np.abs(d["b"]), side="right") - 1, 0, 3)
    return (ig * 4 + ib) * 4 + il


def loglike_cell(theta: dict[str, float], data: dict[str, Any]) -> float:
    """Multinomial log-likelihood of the DR3 N_vis histogram in G x |beta| x |b| cells.

    The predicted histogram of a cell is the mean of its stars' model PMFs. Used when the
    model is conditioned on gaiamock's grid GOST, which differs per star from the exact
    position (a star-level likelihood would then be ill-posed where DR3 N_vis exceeds the
    grid count). NSS cells are truncated at N >= 12.
    """
    ll = 0.0
    for name, d in data.items():
        pmf = nvis_pmf(theta, d)
        cid = cell_ids(d)
        n = np.clip(d["nvis"], 0, pmf.shape[1] - 1)
        for c in np.unique(cid):
            m = cid == c
            pm = pmf[m].mean(0)
            if name == "nss":
                pm = pm / max(1e-300, pm[12:].sum())
            counts = np.bincount(n[m], minlength=pm.size)
            ll += float(np.sum(counts * np.log(np.maximum(pm, 1e-300))))
    return ll


def loglike(theta: dict[str, float], data: dict[str, Any]) -> float:
    if LIKELIHOOD == "cell":
        return loglike_cell(theta, data)
    ll = 0.0
    for name, d in data.items():
        pmf = nvis_pmf(theta, d)
        n = np.clip(d["nvis"], 0, pmf.shape[1] - 1)
        p = pmf[np.arange(n.size), n]
        if name == "nss":
            p = p / np.maximum(1e-300, pmf[:, 12:].sum(1))
        ll += float(np.sum(np.log(np.maximum(p, 1e-300))))
    return ll


def fit(data: dict[str, Any], free: tuple[str, ...], start: dict[str, float]) -> dict[str, Any]:
    theta0 = dict(start)

    def nll(v: np.ndarray) -> float:
        th = dict(theta0)
        th.update(dict(zip(free, v)))
        return -loglike(th, data)

    r = minimize(nll, np.array([theta0[k] for k in free]), method="Nelder-Mead",
                 options={"maxiter": 3000, "xatol": 1e-3, "fatol": 0.05, "adaptive": True})
    th = dict(theta0)
    th.update(dict(zip(free, r.x)))
    return {"theta": th, "ll": -float(r.fun), "k": len(free), "aic": 2 * len(free) + 2 * float(r.fun),
            "success": bool(r.success), "nfev": int(r.nfev)}


def tail_table(theta: dict[str, float], d: dict[str, Any]) -> dict[str, Any]:
    pmf = nvis_pmf(theta, d)
    lt12 = pmf[:, :12].sum(1)
    mean = (pmf * np.arange(pmf.shape[1])).sum(1)
    out: dict[str, Any] = {"all": {"model_lt12": float(lt12.mean()), "dr3_lt12": float(np.mean(d["nvis"] < 12)),
                                   "model_mean": float(mean.mean()), "dr3_mean": float(d["nvis"].mean())}}
    for lab, v, edges in (("G", d["G"], (3, 13, 15, 16, 17, 18, 19)), ("abs_beta", np.abs(d["beta"]), (0, 15, 30, 45, 90)),
                          ("abs_b", np.abs(d["b"]), (0, 5, 10, 30, 90))):
        out[lab] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            out[lab][f"{lo}-{hi}"] = {"n": int(m.sum()), "model_lt12": float(lt12[m].mean()) if m.any() else None,
                                      "dr3_lt12": float(np.mean(d["nvis"][m] < 12)) if m.any() else None,
                                      "model_mean": float(mean[m].mean()) if m.any() else None,
                                      "dr3_mean": float(d["nvis"][m].mean()) if m.any() else None}
    hist_edges = np.arange(0, 46)
    out["hist"] = {"bins": hist_edges[:-1].tolist(), "model": pmf[:, :45].mean(0).tolist(),
                   "dr3": (np.bincount(np.clip(d["nvis"], 0, 44), minlength=45) / d["nvis"].size).tolist()}
    return out


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vp", default=str(P / "output/gate432/vp_structure.h5"))
    ap.add_argument("--keep-fit", default=str(P / "output/gate400/v2/keep_fit.json"))
    ap.add_argument("--n-random", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=432)
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_FREE))
    ap.add_argument("--out", default=str(P / "output/gate432"))
    ap.add_argument("--start-json", default=None, help="vp_tail_fit_*.json whose last variant theta starts the fit")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--b-feature", choices=("abs_sin", "exp"), default="abs_sin")
    ap.add_argument("--b-scale-deg", type=float, default=10.0)
    ap.add_argument("--likelihood", choices=("star", "cell"), default="star")
    args = ap.parse_args(argv)
    global B_FEATURE, B_SCALE_DEG, LIKELIHOOD
    B_FEATURE, B_SCALE_DEG, LIKELIHOOD = args.b_feature, args.b_scale_deg, args.likelihood
    cfg = em.epoch_model_config_from_mapping(load_config().dr3.epoch_model)
    delta = float(json.loads(Path(args.keep_fit).read_text())["adopted"]["random_offset_log"])
    data = load(Path(args.vp), cfg, delta, args.n_random, args.seed)
    start = {"c0": -4.0, "c1": 1.0, "c2": 0.0, "c_beta": 0.0, "c_b": 0.0, "e0": -0.5, "d0": -3.5, "d1": 0.5}
    results: dict[str, Any] = {"likelihood": LIKELIHOOD, "vp_file": str(args.vp), "b_feature": B_FEATURE, "b_scale_deg": B_SCALE_DEG,
                               "n_random": int(data["random"]["G"].size), "n_nss": int(data["nss"]["G"].size),
                               "random_offset_log": delta, "variants": {}}
    prev = start
    if args.start_json:
        sj = json.loads(Path(args.start_json).read_text())["variants"]
        prev = dict(list(sj.values())[-1]["theta"])
    for v in args.variants:
        st = dict(prev)
        if v == "none":
            st.update(c0=-30.0, c1=0.0, c2=0.0, c_beta=0.0, c_b=0.0)  # no degraded component
        elif "c_beta" not in VARIANT_FREE[v]:
            st.update(c_beta=0.0, c_b=0.0)
        elif "c_b" not in VARIANT_FREE[v]:
            st.update(c_b=0.0)
        if args.start_json:
            st = dict(prev)
        if v != "none" and st["c0"] < -20:
            st.update(c0=-4.0, c1=1.0)
        r = fit(data, VARIANT_FREE[v], st)
        prev = r["theta"]
        r["tails_random"] = tail_table(r["theta"], data["random"])
        r["tails_nss"] = tail_table(r["theta"], data["nss"])
        results["variants"][v] = r
        print(v, "ll", round(r["ll"], 1), "AIC", round(r["aic"], 1), "success", r["success"],
              {k: round(x, 3) for k, x in r["theta"].items()},
              "lt12 model/dr3", round(r["tails_random"]["all"]["model_lt12"], 4), round(r["tails_random"]["all"]["dr3_lt12"], 4),
              flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(args.variants)
    (Path(args.out) / f"vp_tail_fit_{tag}.json").write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
