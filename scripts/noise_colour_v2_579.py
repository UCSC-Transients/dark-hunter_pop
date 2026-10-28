#!/usr/bin/env python3
"""#579 noise model ``colour_v2`` (EPOCH_MODEL_SPEC §8.13): colour term and F2 scale on top of N2d, with DR3's G = 13
bright/faint break kept. Light analysis; reads the single-star control and #390 re-injection outputs.

``fit-s``     s(G, C) = exp(a(G) + b(G) c + q c^2) per side of the break, relative to production ``n2d_s1232``
              (exact: s multiplies all actual noise, N2d included), on the paired single-star sigma_act ratio.
``fit-e``     e(G) per side so that the median accepted-orbit F2 of the #390 re-injection matches DR3. F2 is
              transformed analytically from the production re-injection with chi2_red x (s/e)^2.
``u0``        u0(G, BP-RP) per side: 41st percentile of the production single-star UWE x s / e in G windows that
              never cross the break.
``config``    write the ``dr3.epoch_model.noise_colour`` block (enabled: false) into config/config.yaml.
``validate``  sigma_plx lower envelope (2nd percentile, 0.1-mag bins) for singles and orbits vs DR3 with the jump at 13
              and the 13 -> 12.4 falloff; F2 vs G (El-Badry 2024 Fig. 10 style) and the bright F2 tail; medians and
              tails (RUWE > 1.4, F2 > 3) at 0.1 mag; the single-star before/after table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_colour_analysis_572 as n  # noqa: E402

from darkhunter_pop import plotting as pl  # noqa: E402

BREAK = 13.0
CLIP, CLIP_F, CREF = [0.2, 2.6], [0.0, 3.5], 1.2  # bright clip = colour range of the bright single-star cells
A_B = [8.0, 9.75, 10.25, 10.75, 11.25, 11.625, 11.875, 12.125, 12.375, 12.625, 12.875]
A_F = [13.125, 13.375, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75]
B_B = [10.75, 12.25, 12.875]
B_F = [13.25, 14.0, 16.0, 18.5]
G_CELLS = np.array([6.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 14.0, 14.5,
                    15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0])
E_BINS_B = [(6.0, 10.5), (10.5, 11.5), (11.5, 12.0), (12.0, 12.5), (12.5, 13.0)]
E_BINS_F = [(13.0, 13.5), (13.5, 14.0), (14.0, 15.0), (15.0, 16.0), (16.0, 17.5)]
P = "/Users/rfoley/darkhunter/pop/dark-hunter_pop/data/dr3/gaia_snapshots/"
PARENT = P + "20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj"
QUALITY = P + "20261027T054741Z_gaia_source_astrometric_quality_K1000000"


def design(g: np.ndarray, c: np.ndarray, use_colour: bool = True, use_q: bool = True) -> tuple[np.ndarray, list[str]]:
    cc = np.where(g < BREAK, np.clip(c, *CLIP), np.clip(c, *CLIP_F)) - CREF
    fb, ff = (g < BREAK).astype(float), (g >= BREAK).astype(float)
    cols, names = [n.hat(g, A_B) * fb[:, None], n.hat(g, A_F) * ff[:, None]], [f"aB@{k:g}" for k in A_B] + [f"aF@{k:g}" for k in A_F]
    if use_colour:
        cols += [n.hat(g, B_B) * (fb * cc)[:, None], n.hat(g, B_F) * (ff * cc)[:, None]]
        names += [f"bB@{k:g}" for k in B_B] + [f"bF@{k:g}" for k in B_F]
    if use_q:
        cols += [(fb * cc**2)[:, None], (ff * cc**2)[:, None]]
        names += ["qB", "qF"]
    return np.column_stack(cols), names


def ln_s(g: np.ndarray, c: np.ndarray, par: dict) -> np.ndarray:
    """ln s for a parameter dict in config layout (bright fields + *_faint), break at BREAK."""
    g = np.asarray(g, float)
    c = np.asarray(c, float)
    cc = np.where(g < BREAK, np.clip(c, *par.get("colour_clip", CLIP)), np.clip(c, *par.get("colour_clip_faint", CLIP_F))) - CREF
    bright = np.interp(g, par["a_knots_g"], par["a_coef"]) + np.interp(g, par["b_knots_g"], par["b_coef"]) * cc + par["q_coef"] * cc**2
    faint = (np.interp(g, par["a_knots_g_faint"], par["a_coef_faint"]) + np.interp(g, par["b_knots_g_faint"], par["b_coef_faint"]) * cc
             + par["q_coef_faint"] * cc**2)
    return np.where(g < BREAK, bright, faint)


def e_of(g: np.ndarray, par: dict) -> np.ndarray:
    g = np.asarray(g, float)
    return np.where(g < BREAK, np.interp(g, par["stated_scale_knots_g"], par["stated_scale"]),
                    np.interp(g, par["stated_scale_knots_g_faint"], par["stated_scale_faint"]))


def run_ln_s(meta: dict, g: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ln s of the run relative to production (0 for n2d_s1232; the configured s for a colour_v2 run)."""
    nc = meta["epoch_model"].get("noise_colour") or {}
    if nc.get("enabled") and nc.get("break_g") is not None:
        return ln_s(g, c, nc)
    if nc.get("enabled"):
        raise SystemExit("colour_v1 runs are not in the n2d frame")
    return np.zeros_like(np.asarray(g, float))


def cmd_fit_s(a: argparse.Namespace) -> None:
    d = n.load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    m, meta = n.load_mock(Path(a.run))
    p = n.paired(m, d, 1.25)
    p["ln_ratio"] = p["ln_ratio"] + run_ln_s(meta, p["phot_g_mean_mag"], p["bp_rp"])
    cells = n.cell_medians(p, a.n_boot, np.random.default_rng(579), min_n=20, g_edges=G_CELLS)
    g = np.array([x["g"] for x in cells])
    c = np.array([x["c"] for x in cells])
    y = np.array([x["ln_ratio"] for x in cells])
    e = np.hypot(np.array([x["err"] for x in cells]), 0.01)
    res: dict = {"run": str(a.run), "cells": cells, "models": {}}
    for name, (uc, uq) in {"G_only": (False, False), "G+b(G)c": (True, False), "G+b(G)c+qc2": (True, True)}.items():
        X, names = design(g, c, uc, uq)
        W = 1 / e**2
        cov = np.linalg.inv(X.T @ (X * W[:, None]) + np.eye(X.shape[1]))
        beta = cov @ (X.T @ (W * y))
        chi2 = float(np.sum(W * (y - X @ beta) ** 2))
        dof = y.size - X.shape[1]
        sc = max(1.0, chi2 / dof)
        res["models"][name] = {"names": names, "coef": beta.tolist(), "coef_err": np.sqrt(np.diag(cov) * sc).tolist(), "chi2": chi2,
                               "dof": dof, "bic": float(chi2 + X.shape[1] * np.log(y.size))}
    bmin = min(v["bic"] for v in res["models"].values())
    res["adopted"] = next(k for k, v in res["models"].items() if v["bic"] < bmin + 2.0)
    mb = res["models"][res["adopted"]]
    cf = dict(zip(mb["names"], zip(mb["coef"], mb["coef_err"])))
    get = lambda pre, kn: ([cf.get(f"{pre}@{k:g}", (0.0, 0.0))[0] for k in kn], [cf.get(f"{pre}@{k:g}", (0.0, 0.0))[1] for k in kn])
    par = {"a_knots_g": A_B, "b_knots_g": B_B, "a_knots_g_faint": A_F, "b_knots_g_faint": B_F, "colour_clip": CLIP, "colour_clip_faint": CLIP_F}
    par["a_coef"], par["a_coef_err"] = get("aB", A_B)
    par["a_coef_faint"], par["a_coef_faint_err"] = get("aF", A_F)
    par["b_coef"], par["b_coef_err"] = get("bB", B_B)
    par["b_coef_faint"], par["b_coef_faint_err"] = get("bF", B_F)
    par["q_coef"], par["q_coef_err"] = cf.get("qB", (0.0, 0.0))
    par["q_coef_faint"], par["q_coef_faint_err"] = cf.get("qF", (0.0, 0.0))
    res["params"] = par
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["models"].items():
        print(f"{k:12s} chi2 {v['chi2']:.1f}/{v['dof']} BIC {v['bic']:.1f}")
    print("adopted", res["adopted"])
    for nm, (v, er) in cf.items():
        if nm.startswith(("b", "q")):
            print(f"  {nm:9s} {v:+.4f} ± {er:.4f}")


def cmd_fit_e(a: argparse.Namespace) -> None:
    from scipy.optimize import brentq

    par = json.loads(Path(a.s_fit).read_text())["params"]
    r = n.load_reinjection(Path(a.reinjection), Path(a.gate390), Path(a.colours))
    c_ = np.where(np.isfinite(r["bp_rp"]), r["bp_rp"], CREF)
    # the run's own s and e are divided out (#581: e is refit on a colour_v2+tr1 run, exact in chi2_red x (s/e)^2)
    rmeta = json.loads((Path(a.reinjection) / "inject_meta.json").read_text())
    nc = rmeta["epoch_model"].get("noise_colour") or {}
    s_run = np.exp(run_ln_s(rmeta, r["g"], c_))
    e_run = e_of(r["g"], nc) if nc.get("enabled") and nc.get("break_g") is not None else np.ones(r["g"].size)
    r["e_run"] = e_run
    s = np.exp(ln_s(r["g"], c_, par)) / s_run * e_run  # relative actual/stated factor; e solved on top
    rng = np.random.default_rng(454)
    out: dict = {"reinjection": str(a.reinjection), "bins": []}
    for side, bins in (("bright", E_BINS_B), ("faint", E_BINS_F)):
        kn, val, err = [], [], []
        for lo, hi in bins:
            b = np.flatnonzero((r["g"] >= lo) & (r["g"] < hi) & np.isfinite(r["f2_mock"]) & np.isfinite(r["f2_dr3"]) & (r["nu"] > 0))
            if b.size < 15:
                continue

            def solve(ii: np.ndarray) -> float:
                t = np.median(r["f2_dr3"][ii])
                f = lambda le: np.median(n.f2_transform(r["f2_mock"][ii], r["nu"][ii], (s[ii] / np.exp(le)) ** 2)) - t
                return float(np.exp(brentq(f, -1.5, 1.5)))

            ev = solve(b)
            sysids = np.unique(r["sid"][b])
            boots = [solve(np.concatenate([b[r["sid"][b] == q] for q in rng.choice(sysids, sysids.size)])) for _ in range(a.n_boot)]
            kn.append(float(np.median(r["g"][b])))
            val.append(ev)
            err.append(float(np.std(boots)))
            out["bins"].append({"side": side, "g_bin": [lo, hi], "n": int(b.size), "f2_dr3": float(np.median(r["f2_dr3"][b])),
                                "f2_prod": float(np.median(r["f2_mock"][b])), "e": ev, "e_err": err[-1]})
        if side == "faint":  # extended to G = 19 at the last measured value (no #390 orbits fainter than 17.5)
            kn.append(19.0)
            val.append(val[-1])
            err.append(err[-1])
        out[side] = {"knots": kn, "e": val, "e_err": err}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for x in out["bins"]:
        print(f"{x['side']:6s} G {x['g_bin']}: n {x['n']} F2 DR3 {x['f2_dr3']:.2f} prod {x['f2_prod']:.2f} -> e {x['e']:.3f} ± {x['e_err']:.3f}")


def cmd_u0(a: argparse.Namespace) -> None:
    m, meta = n.load_mock(Path(a.run))
    d = n.load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    par = {**json.loads(Path(a.s_fit).read_text())["params"], **e_params(json.loads(Path(a.e_fit).read_text()))}
    g, c = d["phot_g_mean_mag"][m["row"]], d["bp_rp"][m["row"]]
    # divide out the run's own s and e (exact; production run: none), then UWE x s / e = (UWE / e) x s at the nodes
    nc = meta["epoch_model"].get("noise_colour") or {}
    e_run = e_of(g, nc) if nc.get("enabled") and nc.get("break_g") is not None else 1.0
    base_uwe = m["uwe"] / np.exp(run_ln_s(meta, g, c)) * e_run / e_of(g, par)
    cn = np.array([0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
    lines = [f"# u0(G, BP-RP) for noise model colour_v2 (EPOCH_MODEL_SPEC §8.13, #579): 41st percentile of production",
             f"# single-star UWE ({Path(a.run).name}) / e(G) in 0.5-mag windows that never cross G = {BREAK:g}, x s(G, C) at",
             f"# the node. Rows with g < {BREAK:g} and g >= {BREAK:g} are separate grids (no interpolation across the break).", "g,bp_rp,u0"]
    for nodes, side in ((np.r_[np.arange(6.0, 12.76, 0.25), 12.99], "b"), (np.arange(13.0, 19.01, 0.25), "f")):
        base = np.full(nodes.size, np.nan)
        for i, g0 in enumerate(nodes):
            w = (np.abs(g - g0) <= 0.25) & ((g < BREAK) if side == "b" else (g >= BREAK)) & np.isfinite(base_uwe)
            if w.sum() >= 100:
                base[i] = np.percentile(base_uwe[w], 41)
        ok = np.isfinite(base)
        base = np.interp(nodes, nodes[ok], base[ok])
        for i, g0 in enumerate(nodes):
            for c0 in cn:
                lines.append(f"{g0:.2f},{c0:.2f},{base[i] * float(np.exp(ln_s(np.array([g0]), np.array([c0]), par))[0]):.5f}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())


def e_params(ef: dict) -> dict:
    return {"stated_scale_knots_g": ef["bright"]["knots"], "stated_scale": ef["bright"]["e"],
            "stated_scale_knots_g_faint": ef["faint"]["knots"], "stated_scale_faint": ef["faint"]["e"]}


def cmd_config(a: argparse.Namespace) -> None:
    par = json.loads(Path(a.s_fit).read_text())["params"]
    ef = json.loads(Path(a.e_fit).read_text())
    u0 = Path(a.u0)
    f = lambda xs: "[" + ", ".join(f"{x:.4f}" for x in xs) + "]"
    k = lambda xs: "[" + ", ".join(f"{x:g}" for x in xs) + "]"
    block = f"""    # #579 noise model colour_v2 (EPOCH_MODEL_SPEC §8.13; Ryan 2026-10-27), supersedes colour_v1 (§8.12): N2d is KEPT and
    # the DR3 G = 13 bright/faint processing break is preserved (note after §8.5). Actual per-CCD noise (gaiamock + N2d)
    # x s(G, BP-RP), ln s = a(G) + b(G) c + q c^2, c = clip(BP-RP) - 1.2, separate parameters for G < 13 (fields
    # without suffix) and G >= 13 (*_faint), keyed on observed system G, never interpolated across 13. Stated error x
    # e(G) (#454 F2), same break, extended to G = 19. RUWE = UWE / u0(G, BP-RP), table split at 13. Fitted relative to
    # production on the paired single stars and the #390 re-injection (scripts/noise_colour_v2_579.py).
    # OFF until Ryan adopts it: switching it on invalidates every stored proposal-set draw (spec §3.5).
    noise_colour:
      enabled: false
      version: colour_v2
      break_g: {BREAK}
      keep_n2d: true
      colour_clip: {CLIP}
      colour_clip_faint: {CLIP_F}
      colour_ref: {CREF}
      a_knots_g: {k(par['a_knots_g'])}
      a_coef: {f(par['a_coef'])}
      a_coef_err: {f(par['a_coef_err'])}
      b_knots_g: {k(par['b_knots_g'])}
      b_coef: {f(par['b_coef'])}
      b_coef_err: {f(par['b_coef_err'])}
      q_coef: {par['q_coef']:.4f}
      q_coef_err: {par['q_coef_err']:.4f}
      a_knots_g_faint: {k(par['a_knots_g_faint'])}
      a_coef_faint: {f(par['a_coef_faint'])}
      a_coef_faint_err: {f(par['a_coef_faint_err'])}
      b_knots_g_faint: {k(par['b_knots_g_faint'])}
      b_coef_faint: {f(par['b_coef_faint'])}
      b_coef_faint_err: {f(par['b_coef_faint_err'])}
      q_coef_faint: {par['q_coef_faint']:.4f}
      q_coef_faint_err: {par['q_coef_faint_err']:.4f}
      stated_scale_knots_g: {k([round(x, 3) for x in ef['bright']['knots']])}
      stated_scale: {f(ef['bright']['e'])}
      stated_scale_knots_g_faint: {k([round(x, 3) for x in ef['faint']['knots']])}
      stated_scale_faint: {f(ef['faint']['e'])}
      u0_table: {u0}
      u0_table_sha256: {hashlib.sha256(u0.read_bytes()).hexdigest()}
      provenance: "{a.provenance}"
"""
    cfgp = Path("config/config.yaml")
    c = cfgp.read_text()
    c, nsub = re.subn(r"    # #(572 / #454 noise model v3|579 noise model colour_v2).*?      provenance: \"[^\n]*\"\n", "", c, flags=re.S)
    anchor = "    # El-Badry et al. (2024) Sect. 3.3.1 bright-star excess noise; validation only.\n    excess_noise:\n"
    assert c.count(anchor) == 1 and nsub == 1
    cfgp.write_text(c.replace(anchor, block + anchor))
    print(block)


# ------------------------------------------------------------------------------------------------ validation


def envelope(g: np.ndarray, y: np.ndarray, edges: np.ndarray, q: float = 2.0, min_n: int = 20) -> np.ndarray:
    out = np.full(edges.size - 1, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        b = (g >= lo) & (g < hi) & np.isfinite(y) & (y > 0)
        if b.sum() >= min_n:
            out[i] = np.percentile(y[b], q)
    return out


def jump(g_c: np.ndarray, env: np.ndarray) -> dict:
    """Envelope ratio across the break (12.95 vs 13.05 bin centres) and the 13 -> 12.4 falloff."""
    f = lambda x: float(env[np.argmin(np.abs(g_c - x))])
    return {"env_12p95": f(12.95), "env_13p05": f(13.05), "jump": f(12.95) / f(13.05) - 1.0, "env_12p45": f(12.45),
            "falloff_12p45_over_13p05": f(12.45) / f(13.05)}


def cmd_validate(a: argparse.Namespace) -> None:
    import h5py
    from astropy.table import Table

    plt = pl.require_pyplot()
    style = pl.default_plotting_style()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = n.load_dr3(Path(a.parent_dir), Path(a.quality_dir))
    e01 = np.round(np.arange(6.0, 19.01, 0.1), 2)
    gc = 0.5 * (e01[:-1] + e01[1:])
    res: dict = {"singles": {}, "orbits": {}}
    # singles: DR3 sigma_plx (published) of the simulated rows, RUWE < 1.25 both sides; mock c-inflated sigma_plx
    runs = {lab: n.load_mock(Path(p)) for lab, p in (s.split("=", 1) for s in a.singles)}
    rows = np.unique(next(iter(runs.values()))[0]["row"])
    keep = rows[d["ruwe"][rows] < 1.25]
    env_d = envelope(d["phot_g_mean_mag"][keep], d["parallax_error"][keep], e01)
    res["singles"]["dr3"] = {"env": env_d.tolist(), **jump(gc, env_d)}
    for lab, (m, meta) in runs.items():
        ok = m["ruwe"] < 1.25
        gm = d["phot_g_mean_mag"][m["row"]]
        env = envelope(gm[ok], m["sigma_plx_c"][ok], e01)
        res["singles"][lab] = {"env": env.tolist(), **jump(gc, env),
                               "faint_ratio_mock_over_dr3_g13_19": float(np.nanmedian(env[gc > 13] / env_d[gc > 13])),
                               "bright_ratio_mock_over_dr3_g11_13": float(np.nanmedian((env / env_d)[(gc > 11) & (gc < 13)]))}
    # orbits: #390 same orbits re-injected (accepted), DR3 published values of the same systems; plus all DR3 orbits
    reinj = {lab: n.load_reinjection(Path(p), Path(a.gate390), Path(a.colours)) for lab, p in (s.split("=", 1) for s in a.reinjections)}
    e02 = np.round(np.arange(6.0, 19.01, 0.2), 2)
    gc2 = 0.5 * (e02[:-1] + e02[1:])
    en = Table.read(a.enrichment, format="ascii.ecsv")
    en = en[np.isin(np.asarray(en["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
    with h5py.File(Path(a.input_columns) / "columns.h5", "r") as f:
        ic = {k: f[k][()] for k in ("source_id", "phot_g_mean_mag", "ruwe")}
    gmap = dict(zip(ic["source_id"].tolist(), ic["phot_g_mean_mag"].tolist()))
    rmap = dict(zip(ic["source_id"].tolist(), ic["ruwe"].tolist()))
    sid = np.asarray(en["SOURCE_ID"], np.int64)
    gall = np.array([gmap.get(int(s), np.nan) for s in sid])
    rall = np.array([rmap.get(int(s), np.nan) for s in sid])
    f2all, spall = np.asarray(en["goodness_of_fit"], float), np.asarray(en["parallax_error"], float)
    env_all = envelope(gall, spall, e01)
    res["orbits"]["dr3_all"] = {"env": env_all.tolist(), **jump(gc, env_all)}
    first = next(iter(reinj.values()))
    res["orbits"]["dr3_390"] = {"env": envelope(first["g"], first["plxerr_dr3"], e02, min_n=5).tolist()}
    for lab, r in reinj.items():
        res["orbits"][lab] = {"env": envelope(r["g"], r["plxerr_mock"], e02, min_n=5).tolist(),
                              "f2_median": [float(np.median(r["f2_mock"][(r["g"] >= lo) & (r["g"] < hi)])) if np.sum((r["g"] >= lo) & (r["g"] < hi)) >= 10 else float("nan")
                                            for lo, hi in zip(e02[:-1], e02[1:])]}
        for nm, (lo, hi) in (("bright", (6, 13)), ("faint", (13, 19))):
            b = (r["g"] >= lo) & (r["g"] < hi)
            res["orbits"][lab][f"f2_gt3_{nm}"] = float(np.mean(r["f2_mock"][b] > 3))
            res["orbits"][lab][f"f2_p90_{nm}"] = float(np.percentile(r["f2_mock"][b], 90))
            res["orbits"][lab][f"plxerr_ratio_{nm}"] = float(np.median(r["plxerr_mock"][b] / r["plxerr_dr3"][b]))
            res["orbits"][lab][f"ruwe_ratio_{nm}"] = float(np.median(r["ruwe_mock"][b] / r["ruwe_dr3"][b]))
            res["orbits"][lab][f"f2_median_{nm}"] = float(np.median(r["f2_mock"][b]))
            if lab == next(iter(reinj)):
                res["orbits"][f"dr3_390_f2_gt3_{nm}"] = float(np.mean(r["f2_dr3"][b] > 3))
                res["orbits"][f"dr3_390_f2_p90_{nm}"] = float(np.percentile(r["f2_dr3"][b], 90))
                res["orbits"][f"dr3_390_f2_median_{nm}"] = float(np.median(r["f2_dr3"][b]))
    res["orbits"]["dr3_390_f2_median"] = [float(np.median(first["f2_dr3"][(first["g"] >= lo) & (first["g"] < hi)])) if np.sum((first["g"] >= lo) & (first["g"] < hi)) >= 10 else float("nan")
                                          for lo, hi in zip(e02[:-1], e02[1:])]
    # all DR3 orbits: F2 median / tails vs G at 0.1 mag (Fig. 10 style), RUWE > 1.4 and F2 > 3 fractions
    res["orbits"]["dr3_all_f2"] = {"median": [], "p90": [], "frac_f2_gt3": []}
    for lo, hi in zip(e01[:-1], e01[1:]):
        b = (gall >= lo) & (gall < hi) & np.isfinite(f2all)
        res["orbits"]["dr3_all_f2"]["median"].append(float(np.median(f2all[b])) if b.sum() > 20 else float("nan"))
        res["orbits"]["dr3_all_f2"]["p90"].append(float(np.percentile(f2all[b], 90)) if b.sum() > 20 else float("nan"))
        res["orbits"]["dr3_all_f2"]["frac_f2_gt3"].append(float(np.mean(f2all[b] > 3)) if b.sum() > 20 else float("nan"))
    # singles: medians and RUWE > 1.4 tail at 0.1 mag (no RUWE cut)
    res["singles_tail"] = {"dr3_ruwe_gt1p4": [], **{f"{lab}_ruwe_gt1p4": [] for lab in runs}, "dr3_ruwe_med": [], **{f"{lab}_ruwe_med": [] for lab in runs}}
    gd = d["phot_g_mean_mag"][rows]
    for lo, hi in zip(e01[:-1], e01[1:]):
        b = (gd >= lo) & (gd < hi)
        res["singles_tail"]["dr3_ruwe_gt1p4"].append(float(np.mean(d["ruwe"][rows][b] > 1.4)) if b.sum() > 20 else float("nan"))
        res["singles_tail"]["dr3_ruwe_med"].append(float(np.median(d["ruwe"][rows][b & (d["ruwe"][rows] < 1.25)])) if b.sum() > 20 else float("nan"))
        for lab, (m, _) in runs.items():
            gm = d["phot_g_mean_mag"][m["row"]]
            bm = (gm >= lo) & (gm < hi)
            res["singles_tail"][f"{lab}_ruwe_gt1p4"].append(float(np.mean(m["ruwe"][bm] > 1.4)) if bm.sum() > 20 else float("nan"))
            res["singles_tail"][f"{lab}_ruwe_med"].append(float(np.median(m["ruwe"][bm & (m["ruwe"] < 1.25)])) if bm.sum() > 20 else float("nan"))
    (out / "validate.json").write_text(json.dumps(res, indent=1, default=float))
    # figures
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    ax = axs[0]
    ax.plot(gc, np.array(res["singles"]["dr3"]["env"]) * 1000, color="k", lw=2.5, label="DR3 singles (σϖ)")
    for i, lab in enumerate(runs):
        st = pl.series_style(i + 1, style)
        ax.plot(gc, np.array(res["singles"][lab]["env"]) * 1000, color=st["color"], ls=st["linestyle"], lw=2, label=f"mock singles {lab}")
    ax.axvline(BREAK, color="0.6", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(9, 19)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel=r"2nd pct of $\sigma_\varpi$ (µas), 0.1-mag bins", title="single stars, RUWE < 1.25")
    ax.legend(prop=pl.legend_prop(style))
    ax = axs[1]
    ax.plot(gc, np.array(res["orbits"]["dr3_all"]["env"]) * 1000, color="k", lw=2.5, label="DR3 all orbits")
    ax.plot(gc2, np.array(res["orbits"]["dr3_390"]["env"]) * 1000, color="0.5", lw=2, label="DR3, #390 orbits")
    for i, lab in enumerate(reinj):
        st = pl.series_style(i + 1, style)
        ax.plot(gc2, np.array(res["orbits"][lab]["env"]) * 1000, color=st["color"], ls=st["linestyle"], lw=2, marker=st["marker"], ms=3,
                label=f"mock #390 re-injected {lab}")
    ax.axvline(BREAK, color="0.6", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(9, 17.5)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel=None, title="orbits (0.1 mag DR3 all; 0.2 mag #390)")
    ax.legend(prop=pl.legend_prop(style), fontsize=8)
    for axx in axs:
        from matplotlib.ticker import NullFormatter, ScalarFormatter

        axx.yaxis.set_major_formatter(ScalarFormatter())
        axx.yaxis.set_minor_formatter(NullFormatter())
    pl.layout_figure(fig)
    pl.save_figure(fig, out / "sigma_plx_envelope_vs_G.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    ax = axs[0]
    ax.plot(gc, res["orbits"]["dr3_all_f2"]["median"], color="k", lw=2.5, label="DR3 all orbits, median")
    ax.plot(gc, res["orbits"]["dr3_all_f2"]["p90"], color="k", lw=1.5, ls=":", label="DR3 all orbits, 90th pct")
    ax.plot(gc2, res["orbits"]["dr3_390_f2_median"], color="0.5", lw=2, label="DR3 #390 orbits, median")
    for i, lab in enumerate(reinj):
        st = pl.series_style(i + 1, style)
        ax.plot(gc2, res["orbits"][lab]["f2_median"], color=st["color"], ls=st["linestyle"], marker=st["marker"], ms=3, lw=2, label=f"mock #390 {lab}, median")
    ax.axvline(BREAK, color="0.6", lw=1, ls=":")
    ax.set_xlim(8, 18)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel="$F_2$", title="orbit F2 vs G (El-Badry et al. 2024 Fig. 10 style)")
    ax.legend(prop=pl.legend_prop(style), fontsize=8)
    ax = axs[1]
    ax.plot(gc, res["singles_tail"]["dr3_ruwe_med"], color="k", lw=2.5, label="DR3 singles median RUWE (< 1.25)")
    for i, lab in enumerate(runs):
        st = pl.series_style(i + 1, style)
        ax.plot(gc, res["singles_tail"][f"{lab}_ruwe_med"], color=st["color"], ls=st["linestyle"], lw=2, label=f"mock {lab}")
    ax.axvline(BREAK, color="0.6", lw=1, ls=":")
    ax.set_xlim(8, 19)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel="median RUWE", title="single stars")
    ax.legend(prop=pl.legend_prop(style), fontsize=8)
    pl.layout_figure(fig)
    pl.save_figure(fig, out / "f2_and_ruwe_vs_G.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "env"} for k, v in res["singles"].items()}, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("env", "f2_median")} if isinstance(v, dict) else v)
                      for k, v in res["orbits"].items() if not k.endswith("f2_median") and k != "dr3_all_f2"}, indent=1, default=float))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = dict(parent_dir=PARENT, quality_dir=QUALITY, gate390="/Users/rfoley/darkhunter/pop/dark-hunter_pop/output/gate390/injection_test_full.h5",
                  colours=P + "20261003T063811Z_epoch_counts_400/inj390.h5")
    p = sub.add_parser("fit-s")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=100)
    p.set_defaults(func=cmd_fit_s, **common)
    p = sub.add_parser("fit-e")
    p.add_argument("--reinjection", required=True)
    p.add_argument("--s-fit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=50)
    p.set_defaults(func=cmd_fit_e, **common)
    p = sub.add_parser("u0")
    p.add_argument("--run", required=True)
    p.add_argument("--s-fit", required=True)
    p.add_argument("--e-fit", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_u0, **common)
    p = sub.add_parser("config")
    p.add_argument("--s-fit", required=True)
    p.add_argument("--e-fit", required=True)
    p.add_argument("--u0", required=True)
    p.add_argument("--provenance", required=True)
    p.set_defaults(func=cmd_config)
    p = sub.add_parser("validate")
    p.add_argument("--singles", nargs="+", required=True, help="label=run_dir")
    p.add_argument("--reinjections", nargs="+", required=True, help="label=run_dir")
    p.add_argument("--enrichment", default=P + "nss_enrichment/query.ecsv")
    p.add_argument("--input-columns", default=P + "20261003T182211Z_nss_orbit_input_columns")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_validate, **common)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
