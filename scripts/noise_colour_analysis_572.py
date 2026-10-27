#!/usr/bin/env python3
"""#572 / #454 / #574: single-star control analysis. DR3 vs mock singles (paired rows), colour-term fit, u0(G, C), and
the single-star false-orbit / false-acceleration rate. Light analysis (reads the simulate output; no gaiamock).

Subcommands
-----------
``fit``      fit ln s(G, C) (actual per-CCD noise scale) on the paired sigma_act ratio, choose terms by BIC; write
             ``colour_fit.json``.
``u0``       2-D u0(G, BP-RP) table = 41st percentile of mock single-star UWE x s/e in G x colour cells (exact: every
             noise term is a per-star scalar; spec §8.12); write the CSV with a provenance header.
``compare``  DR3 vs one or more mock runs: RUWE, sigma_act, parallax/sigma, excess noise, single-star F2, by G, colour
             (observed and dereddened) and N; the false-solution rates by G, colour, N and distance, scaled to the
             parent counts. Writes ``compare_<label>.json`` and figures.

DR3 conventions (spec §8.12): UWE = sqrt(chi2_AL / (n_good - 5)) uses AGIS weights without the excess noise eps;
sigma_act,DR3 = sigma_plx x UWE when eps = 0, else sigma_plx (eps absorbs the excess). Mock sigma_act = the c-inflated
5-parameter parallax error. Single-star F2: DR3 ``astrometric_gof_al`` includes eps (~0 by construction); the mock F2
uses the stated errors only, so the two are shown but not used to calibrate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import plotting as pl

G_EDGES = np.array([6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
#: finer G cells for the colour-term fit
G_FIT_EDGES = np.array([6.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0])
C_EDGES = np.array([-0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5, 6.0])
#: coarser colour cells where the parent has few rows (G < 11.5)
C_EDGES_BRIGHT = np.array([-0.5, 0.8, 1.2, 6.0])
NVIS_EDGES = np.array([0, 15, 18, 21, 24, 28, 60])
D_EDGES = np.array([0.0, 0.3, 0.7, 1.5, 6.0])


def f2_from_uwe(uwe: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Wilson-Hilferty F2 from chi2 / nu = UWE^2 (Lindegren et al. 2012 Eq. A.13)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(9.0 * nu / 2.0) * ((uwe**2) ** (1.0 / 3.0) + 2.0 / (9.0 * nu) - 1.0)


def load_dr3(parent_dir: Path, quality_dir: Path) -> dict[str, np.ndarray]:
    import h5py
    import yaml

    meta = yaml.safe_load((quality_dir / "meta.yaml").read_text())
    import hashlib

    got = hashlib.sha256((quality_dir / "quality.h5").read_bytes()).hexdigest()
    if got != meta["quality_h5_sha256"]:
        raise ValueError(f"quality snapshot checksum mismatch: {got}")
    with h5py.File(parent_dir / "parent.h5", "r") as f:
        par = {k: f[k][()] for k in ("source_id", "phot_g_mean_mag", "bp_rp", "ruwe", "parallax", "parallax_error",
                                     "visibility_periods_used", "r_med_geo", "ra", "dec")}
    with h5py.File(quality_dir / "quality.h5", "r") as f:
        q = {k: f[k][()] for k in f.keys()}
    o = np.argsort(q["source_id"])
    j = o[np.clip(np.searchsorted(q["source_id"][o], par["source_id"]), 0, o.size - 1)]
    hit = q["source_id"][j] == par["source_id"]
    for k, v in q.items():
        if k != "source_id":
            par["q_" + k] = np.where(hit, v[j], np.nan)
    nu = par["q_astrometric_n_good_obs_al"] - 5.0
    with np.errstate(invalid="ignore", divide="ignore"):
        par["uwe"] = np.sqrt(par["q_astrometric_chi2_al"] / nu)
    eps = par["q_astrometric_excess_noise"]
    par["sigma_act"] = par["parallax_error"] * np.where(eps > 0, 1.0, par["uwe"])
    par["f2"] = par["q_astrometric_gof_al"]
    par["plx_snr"] = par["parallax"] / par["parallax_error"]
    par["n_obs"] = par["q_astrometric_n_good_obs_al"]
    par["eps"] = eps
    return par


def load_mock(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    meta = json.loads((run_dir / "singles_meta.json").read_text())
    stride = int(meta.get("real_stride", 1))
    keys = ("idx", "n_obs", "n_vis", "uwe", "nu", "ruwe", "plx_hat", "sigma_plx_c", "sigma_plx_formal", "stated_err")
    cols: dict[str, list] = {k: [] for k in keys}
    sol, orb, acc = [], [], []
    with (run_dir / "singles.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            for k in keys:
                cols[k].append(r.get(k, np.nan))
            sol.append(r.get("solution_type", ""))
            orb.append(bool(r.get("accepted_orbital", False)))
            acc.append(bool(r.get("published_acceleration", False)))
    m = {k: np.asarray(v, float) for k, v in cols.items()}
    idx = m["idx"].astype(np.int64)
    m["row"], m["real"] = idx // stride, idx % stride
    m["solution_type"] = np.asarray(sol)
    m["accepted_orbital"], m["published_acceleration"] = np.asarray(orb), np.asarray(acc)
    m["f2"] = f2_from_uwe(m["uwe"], m["nu"])
    m["eps"] = m["stated_err"] * np.sqrt(np.clip(m["uwe"] ** 2 - 1.0, 0.0, None))
    return m, meta


def paired(m: dict[str, np.ndarray], d: dict[str, np.ndarray], ruwe_max: float) -> dict[str, np.ndarray]:
    """Per row: DR3 values and the mock median over realizations with RUWE < ruwe_max (both sides cut)."""
    rows = np.unique(m["row"])
    o = np.argsort(m["row"], kind="stable")
    rs = m["row"][o]
    starts = np.searchsorted(rs, rows)
    ends = np.searchsorted(rs, rows, side="right")
    lsig = np.log(m["sigma_plx_c"][o])
    ok = (m["ruwe"][o] < ruwe_max) & np.isfinite(lsig)
    med = np.full(rows.size, np.nan)
    for k, (a, b) in enumerate(zip(starts, ends)):
        s = ok[a:b]
        if s.any():
            med[k] = np.median(lsig[a:b][s])
    out = {"row": rows, "ln_sigma_act_mock": med}
    for k in ("phot_g_mean_mag", "bp_rp", "ruwe", "sigma_act", "visibility_periods_used", "r_med_geo", "eps", "uwe"):
        out[k] = d[k][rows]
    out["ln_ratio"] = np.log(out["sigma_act"]) - med
    out["ok"] = (out["ruwe"] < ruwe_max) & np.isfinite(out["ln_ratio"])
    return out


def cell_medians(p: dict[str, np.ndarray], n_boot: int, rng: np.random.Generator, min_n: int = 30,
                 g_edges: np.ndarray = G_FIT_EDGES) -> list[dict]:
    cells = []
    g, c, y, ok = p["phot_g_mean_mag"], p["bp_rp"], p["ln_ratio"], p["ok"]
    pairs = [((g0, g1), (c0, c1)) for g0, g1 in zip(g_edges[:-1], g_edges[1:])
             for c0, c1 in zip(*(lambda e: (e[:-1], e[1:]))(C_EDGES_BRIGHT if g1 <= 11.5 else C_EDGES))]
    for (g0, g1), (c0, c1) in pairs:
        b = np.flatnonzero(ok & (g >= g0) & (g < g1) & (c >= c0) & (c < c1))
        if b.size < min_n:
            continue
        boots = [np.median(y[rng.choice(b, b.size)]) for _ in range(n_boot)]
        cells.append({"g": float(np.median(g[b])), "c": float(np.median(c[b])), "g_bin": [g0, g1], "c_bin": [c0, c1],
                      "n": int(b.size), "ln_ratio": float(np.median(y[b])), "err": float(np.std(boots))})
    return cells


def hat(g: np.ndarray, knots: list[float]) -> np.ndarray:
    """Piecewise-linear (hat) basis on ``knots``, constant outside; columns sum to 1."""
    g = np.clip(np.asarray(g, float), knots[0], knots[-1])
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(g, knots, eye[k]) for k in range(len(knots))])


A_KNOTS = [8.5, 9.5, 10.5, 11.5, 12.5, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
B_KNOTS = [9.5, 12.0, 14.0, 16.0, 18.5]
FIT_FRAME = {"colour_clip": [0.0, 3.5], "colour_ref": 1.2, "a_knots_g": A_KNOTS, "b_knots_g": B_KNOTS}
#: nested models (spec §8.12: BIC picks the colour term)
MODELS = {"G_only": (True, False, False, False), "G+c": (True, True, False, False),
          "G+b(G)c": (True, False, True, False), "G+b(G)c+qc2": (True, False, True, True)}


def design(g: np.ndarray, c: np.ndarray, model: str, fr: dict) -> tuple[np.ndarray, list[str]]:
    use_a, use_c, use_b, use_q = MODELS[model]
    cc = np.clip(np.asarray(c, float), *fr["colour_clip"]) - fr["colour_ref"]
    cols, names = [], []
    if use_a:
        cols.append(hat(g, fr["a_knots_g"]))
        names += [f"a@{k:g}" for k in fr["a_knots_g"]]
    if use_c:
        cols.append(cc[:, None])
        names.append("b")
    if use_b:
        cols.append(hat(g, fr["b_knots_g"]) * cc[:, None])
        names += [f"b@{k:g}" for k in fr["b_knots_g"]]
    if use_q:
        cols.append((cc**2)[:, None])
        names.append("q")
    return np.column_stack(cols), names


def run_prior_ln_s(meta: dict, g: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ln of the actual-noise scale the run was simulated with, relative to bare gaiamock with N2d off: the colour
    model's s when enabled, else the N2d excess sqrt(1 + r2(G)) (stated ~ actual; iteration 0 only)."""
    em = meta["epoch_model"]
    nc = em.get("noise_colour")
    if nc and nc.get("enabled"):
        cc = np.clip(c, *nc["colour_clip"]) - nc["colour_ref"]
        return (np.interp(g, nc["a_knots_g"], nc["a_coef"]) + np.interp(g, nc["b_knots_g"], nc["b_coef"]) * cc
                + nc["q_coef"] * cc**2)
    bn = em.get("bright_excess_noise")
    if bn and bn.get("enabled"):
        r2 = np.where(g < bn["g_max"], np.interp(g, bn["knots_g"], bn["knots_r2"]), 0.0)
        return 0.5 * np.log1p(r2)
    return np.zeros_like(g)


def cmd_fit(a: argparse.Namespace) -> None:
    d = load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    m, meta = load_mock(Path(a.run))
    p = paired(m, d, a.ruwe_max)
    # target: ln s_needed = ln(sigma_act DR3 / mock) + ln s_run (s multiplies the actual noise exactly)
    p["ln_ratio"] = p["ln_ratio"] + run_prior_ln_s(meta, p["phot_g_mean_mag"], p["bp_rp"])
    rng = np.random.default_rng(572)
    cells = cell_medians(p, a.n_boot, rng, min_n=a.min_n)
    g = np.array([x["g"] for x in cells])
    c = np.array([x["c"] for x in cells])
    y = np.array([x["ln_ratio"] for x in cells])
    e = np.hypot(np.array([x["err"] for x in cells]), a.sys_floor)
    res: dict = {"run": str(a.run), "noise_model_version": meta["noise_model_version"], "ruwe_max": a.ruwe_max,
                 "sys_floor": a.sys_floor, "frame": FIT_FRAME, "cells": cells, "models": {},
                 "target": "ln s_needed = ln(sigma_act,DR3 / sigma_act,mock) + ln s_run (vs bare gaiamock, N2d off)"}
    best = None
    for name in MODELS:
        X, names = design(g, c, name, FIT_FRAME)
        W = 1.0 / e**2
        # weak N(0, 1) prior on every coefficient (ln s within ~e^±1) keeps knots without cells defined (they go to 0)
        cov = np.linalg.inv(X.T @ (X * W[:, None]) + np.eye(X.shape[1]) / a.prior_sigma**2)
        beta = cov @ (X.T @ (W * y))
        chi2 = float(np.sum(W * (y - X @ beta) ** 2))
        dof = y.size - X.shape[1]
        bic = chi2 + X.shape[1] * np.log(y.size)
        scale = max(1.0, chi2 / dof)
        res["models"][name] = {"names": names, "coef": beta.tolist(), "coef_err": np.sqrt(np.diag(cov) * scale).tolist(),
                               "chi2": chi2, "dof": dof, "bic": float(bic), "err_scale": float(np.sqrt(scale))}
    # lowest BIC, but the simplest model within dBIC < 2 of it (models are nested and listed simplest first)
    bmin = min(v["bic"] for v in res["models"].values())
    best = next(k for k, v in res["models"].items() if v["bic"] < bmin + 2.0)
    res["adopted"] = best
    res["n_cells"] = int(y.size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["models"].items():
        print(f"{k:14s} chi2 {v['chi2']:.1f} dof {v['dof']} BIC {v['bic']:.1f}")
    mb = res["models"][best]
    print("adopted", best)
    for nm, cf, er in zip(mb["names"], mb["coef"], mb["coef_err"]):
        print(f"  {nm:8s} {cf:+.4f} ± {er:.4f}")
    if a.plot_dir:
        plot_fit(res, Path(a.plot_dir))


def model_parts(fit: dict, name: str | None = None) -> dict:
    """a/b/q coefficient vectors of a fitted model, in the config layout (zeros where a part is absent)."""
    name = name or fit["adopted"]
    mdl = fit["models"][name]
    use_a, use_c, use_b, use_q = MODELS[name]
    fr = fit["frame"]
    coef, err = list(mdl["coef"]), list(mdl["coef_err"])
    na, nb = len(fr["a_knots_g"]), len(fr["b_knots_g"])
    out = {"a_coef": [0.0] * na, "a_coef_err": [0.0] * na, "b_coef": [0.0] * nb, "b_coef_err": [0.0] * nb, "q_coef": 0.0, "q_coef_err": 0.0}
    i = 0
    if use_a:
        out["a_coef"], out["a_coef_err"] = coef[i:i + na], err[i:i + na]
        i += na
    if use_c:
        out["b_coef"], out["b_coef_err"] = [coef[i]] * nb, [err[i]] * nb
        i += 1
    if use_b:
        out["b_coef"], out["b_coef_err"] = coef[i:i + nb], err[i:i + nb]
        i += nb
    if use_q:
        out["q_coef"], out["q_coef_err"] = coef[i], err[i]
    return out


def s_model(g: np.ndarray, c: np.ndarray, fit: dict, name: str | None = None) -> np.ndarray:
    pt = model_parts(fit, name)
    fr = fit["frame"]
    cc = np.clip(np.asarray(c, float), *fr["colour_clip"]) - fr["colour_ref"]
    g = np.asarray(g, float)
    return np.exp(np.interp(g, fr["a_knots_g"], pt["a_coef"]) + np.interp(g, fr["b_knots_g"], pt["b_coef"]) * cc + pt["q_coef"] * cc**2)


def plot_fit(res: dict, out: Path) -> None:
    plt = pl.require_pyplot()
    style = pl.default_plotting_style()
    out.mkdir(parents=True, exist_ok=True)
    cells = res["cells"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    gg = np.linspace(6, 19, 400)
    for k, (c0, c1) in enumerate(zip(C_EDGES[:-1], C_EDGES[1:])):
        sel = [x for x in cells if x["c_bin"] == [c0, c1]]
        if not sel:
            continue
        st = pl.series_style(k, style)
        ax.errorbar([x["g"] for x in sel], [np.exp(x["ln_ratio"]) for x in sel], yerr=[np.exp(x["ln_ratio"]) * x["err"] for x in sel],
                    color=st["color"], marker=st["marker"], ls="none", capsize=2, ms=5, label=f"BP−RP {c0:g}–{c1:g}")
        cm = float(np.median([x["c"] for x in sel]))
        ax.plot(gg, s_model(gg, np.full_like(gg, cm), res), color=st["color"], ls=st["linestyle"], lw=1.5)
    ax.axhline(1, color="0.5", lw=1)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel=r"required actual-noise scale $s$",
                        title=f"s(G, BP−RP) from {res['noise_model_version']} singles: {res['adopted']}")
    ax.legend(prop=pl.legend_prop(style), ncol=2, fontsize=8)
    pl.layout_figure(fig)
    pl.save_figure(fig, out / f"colour_fit_from_{res['noise_model_version']}.png", dpi=150)
    plt.close(fig)


def cmd_u0(a: argparse.Namespace) -> None:
    import hashlib

    m, meta = load_mock(Path(a.run))
    d = load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    fit = json.loads(Path(a.fit).read_text())
    g = d["phot_g_mean_mag"][m["row"]]
    c = d["bp_rp"][m["row"]]
    e = np.interp(g, a.stated_knots_g, a.stated_scale)
    uwe = m["uwe"] * s_model(g, c, fit) / e
    gn = np.arange(a.g_min, a.g_max + 1e-9, a.g_step)
    cn = np.array(a.colour_nodes)
    # u0 per G node from all colours (41st percentile of the baseline UWE in a +-g_step/2 window, as §8.8), then the exact
    # per-star scaling s/e is applied as the node's (G, C) factor: the baseline UWE distribution has no colour term.
    base = np.full(gn.size, np.nan)
    for i, g0 in enumerate(gn):
        b = np.isfinite(m["uwe"]) & (np.abs(g - g0) <= a.g_step / 2 + 1e-9)
        if b.sum() >= 100:
            base[i] = np.percentile(m["uwe"][b] / np.interp(g[b], a.stated_knots_g, a.stated_scale), 41)
    good = np.isfinite(base)
    base = np.interp(gn, gn[good], base[good])
    grid = base[:, None] * s_model(np.repeat(gn, cn.size), np.tile(cn, gn.size), fit).reshape(gn.size, cn.size)
    # check against the direct cell percentile where a cell has >= 200 stars
    chk = []
    for i, g0 in enumerate(gn):
        for j, c0 in enumerate(cn):
            half = 0.5 * (cn[min(j + 1, cn.size - 1)] - cn[max(j - 1, 0)]) / 2
            b = np.isfinite(uwe) & (np.abs(g - g0) <= a.g_step / 2 + 1e-9) & (np.abs(c - c0) <= max(half, 0.1))
            if b.sum() >= 200:
                chk.append(float(np.percentile(uwe[b], 41) / grid[i, j]))
    lines = [f"# u0(G, BP-RP) for noise model {a.version} (EPOCH_MODEL_SPEC §8.12, #572): 41st percentile of mock single-star",
             f"# UWE (looks-single parent rows, run {meta['noise_model_version']} {Path(a.run).name}) x s(G, C) / e(G);",
             f"# colour fit {Path(a.fit).name} ({fit['adopted']}); direct-cell check median {np.median(chk):.4f}, "
             f"range {min(chk):.4f}-{max(chk):.4f} over {len(chk)} cells", "g,bp_rp,u0"]
    for i, g0 in enumerate(gn):
        for j, c0 in enumerate(cn):
            lines.append(f"{g0:.2f},{c0:.2f},{grid[i, j]:.5f}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(Path(a.out), hashlib.sha256(Path(a.out).read_bytes()).hexdigest(), "check", np.median(chk), min(chk), max(chk))


def wq(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size >= 20 else float("nan")


def binned_stats(key_d: str, key_m: str, d: dict, m: dict, rows_d: np.ndarray, mm: np.ndarray, by: str, edges: np.ndarray,
                 qs: tuple[float, ...] = (16, 50, 84, 97.7)) -> dict:
    out = {"edges": edges.tolist(), "dr3": [], "mock": []}
    gd, gm = d[by][rows_d], (m[by][mm] if by in m else d[by][m["row"][mm]])
    for lo, hi in zip(edges[:-1], edges[1:]):
        bd = (gd >= lo) & (gd < hi)
        bm = (gm >= lo) & (gm < hi)
        out["dr3"].append([wq(d[key_d][rows_d][bd], q) for q in qs])
        out["mock"].append([wq(m[key_m][mm][bm], q) for q in qs])
    out["quantiles"] = list(qs)
    return out


def cmd_compare(a: argparse.Namespace) -> None:
    import h5py
    import yaml

    d = load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    d["dist_kpc"] = d["r_med_geo"] / 1000.0
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.runs:
        label, path = spec.split("=", 1)
        runs[label] = load_mock(Path(path))
    res: dict = {"runs": {k: v[1]["noise_model_version"] for k, v in runs.items()}, "ruwe_max": a.ruwe_max}
    # parent counts for the false-solution scaling: every parent row with G < 19 that passes the usable flags except the
    # single cuts (all stars the NSS input sees), and the looks-single subset; x scale_to_full
    pmeta = yaml.safe_load((Path(a.parent_dir) / "meta.yaml").read_text())
    scale = float(pmeta["gaia_source_total_rows"]) / float(pmeta["random_index_max_exclusive"])
    with h5py.File(Path(a.parent_dir) / "parent.h5", "r") as f:
        nss = f["non_single_star"][()]
    for label, (m, meta) in runs.items():
        rows_all = np.unique(m["row"])
        mm = (m["ruwe"] < a.ruwe_max) & np.isfinite(m["uwe"])
        rows_d = rows_all[d["ruwe"][rows_all] < a.ruwe_max]
        m["phot_g_mean_mag"] = d["phot_g_mean_mag"][m["row"]]
        m["bp_rp"] = d["bp_rp"][m["row"]]
        m["plx_snr"] = m["plx_hat"] / m["sigma_plx_c"]
        m["sigma_act"] = m["sigma_plx_c"]
        m["visibility_periods_used"] = m["n_vis"]
        m["dist_kpc"] = d["dist_kpc"][m["row"]]
        r: dict = {}
        for obs_d, obs_m in (("ruwe", "ruwe"), ("sigma_act", "sigma_act"), ("plx_snr", "plx_snr"), ("eps", "eps"),
                             ("f2", "f2"), ("uwe", "uwe"), ("n_obs", "n_obs")):
            r[obs_d] = {by: binned_stats(obs_d, obs_m, d, m, rows_d, mm, by, edges)
                        for by, edges in (("phot_g_mean_mag", G_EDGES), ("bp_rp", C_EDGES), ("visibility_periods_used", NVIS_EDGES))}
        # paired sigma_act ratio by G x colour (the calibration statistic)
        p = paired(m, d, a.ruwe_max)
        r["sigma_act_ratio_cells"] = cell_medians(p, 30, np.random.default_rng(1), g_edges=G_EDGES)
        r["sigma_act_ratio_by_nvis"] = [float(np.median(p["ln_ratio"][p["ok"] & (p["visibility_periods_used"] >= lo) & (p["visibility_periods_used"] < hi)]))
                                        for lo, hi in zip(NVIS_EDGES[:-1], NVIS_EDGES[1:])]
        # RUWE > 1.4 fraction (no RUWE cut on either side; DR3 includes real binaries and is an upper bound)
        r["ruwe_gt_1p4"] = {"g_edges": G_EDGES.tolist(),
                            "dr3": [float(np.mean(d["ruwe"][rows_all][(d["phot_g_mean_mag"][rows_all] >= lo) & (d["phot_g_mean_mag"][rows_all] < hi)] > 1.4))
                                    for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:])],
                            "mock": [float(np.mean(m["ruwe"][(m["phot_g_mean_mag"] >= lo) & (m["phot_g_mean_mag"] < hi)] > 1.4))
                                     for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:])]}
        # #574 false solutions
        r["false_solutions"] = false_rates(m, d, nss, scale, a)
        res[label] = r
    (out_dir / f"compare_{a.label}.json").write_text(json.dumps(res, indent=1, default=float))
    plot_compare(res, runs, d, out_dir, a)
    print_summary(res)


def false_rates(m: dict, d: dict, nss: np.ndarray, scale: float, a: argparse.Namespace) -> dict:
    """Accepted-orbit / published-acceleration rate per simulated single star, by G, colour, N_vis and distance;
    scaled to the parent: rate(G bin) x parent rows in that bin (G < 19, parallax > 0.2, same row-level flags) x
    scale_to_full. Two populations: all those rows, and those with non_single_star = 0 and RUWE < 1.25."""
    out: dict = {}
    for kind in ("accepted_orbital", "published_acceleration"):
        k = m[kind]
        res = {}
        for by, edges in (("phot_g_mean_mag", G_EDGES), ("bp_rp", C_EDGES), ("visibility_periods_used", NVIS_EDGES), ("dist_kpc", D_EDGES)):
            x = m[by]
            res[by] = {"edges": edges.tolist(),
                       "n_sim": [int(np.sum((x >= lo) & (x < hi))) for lo, hi in zip(edges[:-1], edges[1:])],
                       "n_false": [int(np.sum(k & (x >= lo) & (x < hi))) for lo, hi in zip(edges[:-1], edges[1:])]}
            res[by]["rate"] = [nf / ns if ns else float("nan") for nf, ns in zip(res[by]["n_false"], res[by]["n_sim"])]
        # implied counts in DR3 (G bins): parent rows simulated-eligible are the flagged rows; use all parent rows in G bin
        g = d["phot_g_mean_mag"]
        elig_all = np.isfinite(g) & (g < 19.0) & np.isfinite(d["bp_rp"])
        elig_single = elig_all & (nss == 0) & (d["ruwe"] < 1.25)
        rates = np.array(res["phot_g_mean_mag"]["rate"])
        nsim = np.array(res["phot_g_mean_mag"]["n_sim"])
        nf = np.array(res["phot_g_mean_mag"]["n_false"])
        for pop, el in (("all_parent_rows", elig_all), ("looks_single_rows", elig_single)):
            npar = np.array([np.sum(el & (g >= lo) & (g < hi)) for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:])]) * scale
            implied = np.where(nsim > 0, rates, 0.0) * npar
            # Poisson 68% upper limit for bins with zero events
            err = np.where(nsim > 0, np.sqrt(np.maximum(nf, 1.0)) / np.maximum(nsim, 1) * npar, 0.0)
            res[f"implied_{pop}"] = {"by_g": implied.tolist(), "total": float(implied.sum()), "total_err": float(np.sqrt(np.sum(err**2))),
                                     "total_g9_19": float(implied[G_EDGES[:-1] >= 9].sum()), "n_parent_scaled": npar.tolist()}
        res["n_total_sim"] = int(k.size)
        res["n_total_false"] = int(k.sum())
        out[kind] = res
    out["solution_types"] = {s: int(np.sum(m["solution_type"] == s)) for s in np.unique(m["solution_type"])}
    return out


def print_summary(res: dict) -> None:
    for label in res["runs"]:
        r = res[label]
        print(f"== {label} ({res['runs'][label]})")
        for obs in ("ruwe", "sigma_act", "plx_snr", "uwe"):
            t = r[obs]["phot_g_mean_mag"]
            print(f"  {obs:9s} median by G  DR3 " + " ".join(f"{x[1]:.3g}" for x in t["dr3"]))
            print(f"  {'':9s}              mock " + " ".join(f"{x[1]:.3g}" for x in t["mock"]))
        fs = r["false_solutions"]
        for kind in ("accepted_orbital", "published_acceleration"):
            f = fs[kind]
            print(f"  {kind}: {f['n_total_false']}/{f['n_total_sim']}; implied all-rows {f['implied_all_parent_rows']['total']:.0f}"
                  f" ± {f['implied_all_parent_rows']['total_err']:.0f} (G 9-19 {f['implied_all_parent_rows']['total_g9_19']:.0f});"
                  f" looks-single {f['implied_looks_single_rows']['total']:.0f}")


def plot_compare(res: dict, runs: dict, d: dict, out: Path, a: argparse.Namespace) -> None:
    plt = pl.require_pyplot()
    style = pl.default_plotting_style()
    labels = list(runs)
    obs_lab = {"ruwe": "RUWE", "sigma_act": r"$\sigma_{\rm act}$ (mas)", "plx_snr": r"$\varpi/\sigma_\varpi$", "eps": r"$\epsilon$ (mas)",
               "f2": r"single-star $F_2$", "uwe": "UWE", "n_obs": r"$N_{\rm obs}$"}
    by_lab = {"phot_g_mean_mag": "$G$ (mag)", "bp_rp": r"$G_{\rm BP}-G_{\rm RP}$ (mag)", "visibility_periods_used": r"$N_{\rm vis}$"}
    for obs in obs_lab:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
        for ax, by in zip(axs, by_lab):
            t0 = res[labels[0]][obs][by]
            e = np.array(t0["edges"], float)
            xc = 0.5 * (e[:-1] + e[1:])
            dr = np.array(t0["dr3"])
            ax.fill_between(xc, dr[:, 0], dr[:, 2], color="0.8", label="DR3 16–84%")
            ax.plot(xc, dr[:, 1], color="k", lw=2.5, marker="o", label="DR3 median")
            for i, lab in enumerate(labels):
                st = pl.series_style(i + 1, style)
                mk = np.array(res[lab][obs][by]["mock"])
                ax.plot(xc, mk[:, 1], color=st["color"], ls=st["linestyle"], marker=st["marker"], lw=st["linewidth"], label=f"mock {lab} median")
                ax.plot(xc, mk[:, 0], color=st["color"], ls=":", lw=1.2)
                ax.plot(xc, mk[:, 2], color=st["color"], ls=":", lw=1.2)
            if obs in ("sigma_act", "plx_snr"):
                ax.set_yscale("log")
            pl.apply_axes_style(ax, style, xlabel=by_lab[by], ylabel=obs_lab[obs])
        axs[0].legend(prop=pl.legend_prop(style), fontsize=8)
        fig.suptitle(f"single stars, RUWE < {a.ruwe_max:g} both sides: {obs_lab[obs]}")
        pl.layout_figure(fig)
        pl.save_figure(fig, out / f"singles_{obs}_{a.label}.png", dpi=150)
        plt.close(fig)
    # paired sigma_act ratio vs G per colour bin, one panel per run
    fig, axs = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 4.8), squeeze=False, sharey=True)
    for ax, lab in zip(axs[0], labels):
        cells = res[lab]["sigma_act_ratio_cells"]
        for k, (c0, c1) in enumerate(zip(C_EDGES[:-1], C_EDGES[1:])):
            sel = [x for x in cells if x["c_bin"] == [c0, c1]]
            if sel:
                st = pl.series_style(k, style)
                ax.errorbar([x["g"] for x in sel], [np.exp(x["ln_ratio"]) for x in sel], yerr=[np.exp(x["ln_ratio"]) * x["err"] for x in sel],
                            color=st["color"], marker=st["marker"], ls=st["linestyle"], lw=1.5, capsize=2, label=f"BP−RP {c0:g}–{c1:g}")
        ax.axhline(1, color="0.5", lw=1)
        pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel=r"$\sigma_{\rm act}$ DR3 / mock", title=f"{lab} ({res['runs'][lab]})")
    axs[0][0].legend(prop=pl.legend_prop(style), fontsize=7, ncol=2)
    pl.layout_figure(fig)
    pl.save_figure(fig, out / f"sigma_act_ratio_{a.label}.png", dpi=150)
    plt.close(fig)
    # false-solution rates
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, by in zip(axs, ("phot_g_mean_mag", "bp_rp", "visibility_periods_used", "dist_kpc")):
        for i, lab in enumerate(labels):
            for j, kind in enumerate(("accepted_orbital", "published_acceleration")):
                t = res[lab]["false_solutions"][kind][by]
                e = np.array(t["edges"], float)
                xc = 0.5 * (e[:-1] + e[1:])
                rate = np.array(t["rate"])
                nf = np.array(t["n_false"], float)
                ns = np.maximum(np.array(t["n_sim"], float), 1)
                up = np.where(nf > 0, np.sqrt(nf) / ns, 1.14 / ns)  # 68% Poisson upper limit for 0
                st = pl.series_style(2 * i + j, style)
                ax.errorbar(xc, np.where(nf > 0, rate, up), yerr=np.where(nf > 0, up, 0.5 * up), uplims=nf == 0, color=st["color"],
                            marker=st["marker"], ls=st["linestyle"], lw=1.5, capsize=2, label=f"{lab}: {'orbit' if j == 0 else 'acceleration'}")
        ax.set_yscale("log")
        pl.apply_axes_style(ax, style, xlabel={"dist_kpc": "distance (kpc)", **by_lab}[by], ylabel="false solutions per single star" if by == "phot_g_mean_mag" else None)
    axs[0].legend(prop=pl.legend_prop(style), fontsize=7)
    pl.layout_figure(fig)
    pl.save_figure(fig, out / f"false_solution_rates_{a.label}.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit", cmd_fit), ("u0", cmd_u0), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument("--parent-dir", required=True)
        p.add_argument("--quality-dir", required=True)
        p.add_argument("--ruwe-max", type=float, default=1.25)
        p.set_defaults(func=fn)
        if name == "fit":
            p.add_argument("--run", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--n-boot", type=int, default=100)
            p.add_argument("--sys-floor", type=float, default=0.01, help="per-cell systematic floor on ln ratio")
            p.add_argument("--plot-dir", default=None)
            p.add_argument("--prior-sigma", type=float, default=1.0)
            p.add_argument("--min-n", type=int, default=20)
        elif name == "u0":
            p.add_argument("--run", required=True)
            p.add_argument("--fit", required=True)
            p.add_argument("--out", required=True)
            p.add_argument("--version", default="colour_v1")
            p.add_argument("--stated-knots-g", type=float, nargs="+", default=[6.0, 19.0])
            p.add_argument("--stated-scale", type=float, nargs="+", default=[1.0, 1.0])
            p.add_argument("--g-min", type=float, default=6.0)
            p.add_argument("--g-max", type=float, default=19.0)
            p.add_argument("--g-step", type=float, default=0.5)
            p.add_argument("--colour-nodes", type=float, nargs="+", default=[0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
        else:
            p.add_argument("--runs", nargs="+", required=True, help="label=run_dir")
            p.add_argument("--label", required=True)
            p.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
