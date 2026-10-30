#!/usr/bin/env python3
"""Noise-model go/no-go comparison (#587): rung-3 v3 against the colour_v2.1+tr1 generations, like for like.

Reads several ``plot_dump.npz`` files from ``scripts/rung3_joint_391.py`` (each optionally with mock components dropped,
e.g. v3 without its WD part) and their ``ranking.json`` files from ``scripts/rung3_plots_391.py`` (all made on the same
bin edges via ``--edges-in``), and writes under ``--out-dir``:

* ``before_after.json`` / ``before_after.md``: χ²/dof ± bootstrap MC error per distribution for every run;
* ``noise_metrics.json`` / ``noise_metrics.md``: the noise-sensitive summary statistics (bright high-F2 tail, F2 medians by
  G and M_G0, ϖ/σ_ϖ upper tail by colour and by P, the a0/σ_a0 upper edge per RUWE bin, short-period bright systems,
  RUWE tail), DR3 count ± √k against mock μ ± √Σw² (MC);
* side-by-side figures: per 2-D pair DR3 | each mock (same colour scale) with the normalised residual under each mock;
  per 1-D observable DR3 against every mock with DR3/mock ratio panels.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/noise_check_compare_587.py --edges <v3 edges.json> \\
        --run v3=<dump>:<rank_dir> --run "v3 lum+spur=<dump>:<rank_dir>:wd" ... --out-dir docs/gate391/noise_check_v21tr1/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung3_plots_391 as rp  # noqa: E402

from darkhunter_pop import plotting as pl  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

PAIRS = {"orbits": [("g", "f2"), ("mg0", "f2"), ("colour0", "plxsnr"), ("P", "plxsnr"), ("a0snr", "ruwe"), ("g", "P"), ("g", "ruwe")],
         "accelerations": [("g", "f2"), ("mg0", "f2"), ("colour0", "plxsnr"), ("g", "ruwe"), ("plxsnr", "ruwe")]}
ONED = {"orbits": ["f2", "ruwe", "plxsnr", "a0snr", "P"], "accelerations": ["f2", "ruwe", "plxsnr"]}
SUBSETS = {"G<13": lambda v: v["g"] < 13.0, "G 13-16": lambda v: (v["g"] >= 13.0) & (v["g"] < 16.0)}
NOISE_DISTS = {("orbits", d) for d in ("f2", "ruwe", "plxsnr", "a0snr", "g x f2", "mg0 x f2", "colour0 x plxsnr", "P x plxsnr",
                                       "a0snr x ruwe", "g x P", "g x ruwe", "plxsnr x ruwe", "f2 x ruwe", "P", "g")} | \
              {("accelerations", d) for d in ("f2", "ruwe", "plxsnr", "g x f2", "mg0 x f2", "colour0 x plxsnr", "g x ruwe", "plxsnr x ruwe",
                                              "f2 x ruwe", "g")}


def parse_run(s):
    name, rest = s.split("=", 1)
    parts = rest.split(":")
    return {"name": name, "dump": Path(parts[0]), "rank": Path(parts[1]), "drop": tuple(parts[2].split(",")) if len(parts) > 2 and parts[2] else ()}


def mock_arrays(kind):
    """Concatenated mock values and weights over the kept components (luminous, WD, spurious)."""
    keys = kind.obs
    vals = {k: np.concatenate([v[k] for v, _ in kind.comp.values()]) for k in keys}
    w = np.concatenate([w for _, w in kind.comp.values()])
    return vals, w


def wquant(x, w, q):
    ok = np.isfinite(x) & (w > 0)
    if ok.sum() < 3:
        return float("nan")
    o = np.argsort(x[ok])
    c = np.cumsum(w[ok][o]) / w[ok].sum()
    return float(x[ok][o][min(np.searchsorted(c, q), ok.sum() - 1)])


def count(mask_r, mask_m, w):
    """DR3 count k against mock μ ± √Σw² (MC); ratio k/μ, pull, and the shape ratio (k/K)/(μ/M) at the sample totals."""
    k = float(np.sum(mask_r))
    mu = float(np.sum(w[mask_m]))
    mc = float(np.sqrt(np.sum(w[mask_m] ** 2)))
    K, M = float(mask_r.size), float(np.sum(w))
    return {"real": k, "mock": mu, "mock_mc": mc, "ratio": (k / mu) if mu > 0 else float("nan"),
            "pull": (k - mu) / np.sqrt(max(mu, 0.5) + mc**2), "shape_ratio": (k / K) / (mu / M) if mu > 0 and M > 0 else float("nan")}


def metrics(kind, kname, thr):
    """Noise-sensitive statistics for one sample of one run; thresholds ``thr`` are fixed from DR3 (same for every run)."""
    r = kind.real
    m, w = mock_arrays(kind)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for gl, gh in ((5, 13), (13, 16), (16, 19.5)):
            sr, sm = (r["g"] >= gl) & (r["g"] < gh), (m["g"] >= gl) & (m["g"] < gh)
            out[f"F2 median, G {gl:g}-{gh:g}"] = {"real": float(np.nanmedian(r["f2"][sr])), "mock": wquant(m["f2"][sm], w[sm], 0.5)}
            out[f"F2 90th pct, G {gl:g}-{gh:g}"] = {"real": float(np.nanpercentile(r["f2"][sr][np.isfinite(r["f2"][sr])], 90)), "mock": wquant(m["f2"][sm], w[sm], 0.9)}
        for t in (3.0, 8.0):
            out[f"N(F2 > {t:g}), G < 13"] = count((r["g"] < 13) & (r["f2"] > t), (m["g"] < 13) & (m["f2"] > t), w)
        for ml, mh in ((-5, 2), (2, 4), (4, 6), (6, 15)):
            sr, sm = (r["mg0"] >= ml) & (r["mg0"] < mh), (m["mg0"] >= ml) & (m["mg0"] < mh)
            out[f"F2 median, M_G0 {ml:g}-{mh:g}"] = {"real": float(np.nanmedian(r["f2"][sr])), "mock": wquant(m["f2"][sm], w[sm], 0.5)}
            out[f"N(F2 > 3), M_G0 {ml:g}-{mh:g}"] = count(sr & (r["f2"] > 3), sm & (m["f2"] > 3), w)
        t = thr["plxsnr_hi"]
        for cl, ch in ((-1, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 5)):
            out[f"N(plx/err > {t:.0f}), colour0 {cl:g}-{ch:g}"] = count((r["colour0"] >= cl) & (r["colour0"] < ch) & (r["plxsnr"] > t),
                                                                        (m["colour0"] >= cl) & (m["colour0"] < ch) & (m["plxsnr"] > t), w)
        if "P" in r:
            for pl_, ph in ((0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)):
                lr, lm = np.log10(r["P"]), np.log10(m["P"])
                out[f"N(plx/err > {t:.0f}), log P {pl_:g}-{ph:g}"] = count((lr >= pl_) & (lr < ph) & (r["plxsnr"] > t), (lm >= pl_) & (lm < ph) & (m["plxsnr"] > t), w)
            for p_ in (50.0, 100.0):
                out[f"N(P < {p_:g} d), G < 13"] = count((r["g"] < 13) & (r["P"] < p_), (m["g"] < 13) & (m["P"] < p_), w)
                out[f"N(P < {p_:g} d), G 13-16"] = count((r["g"] >= 13) & (r["g"] < 16) & (r["P"] < p_), (m["g"] >= 13) & (m["g"] < 16) & (m["P"] < p_), w)
            # a0/σ_a0 upper edge per RUWE bin: mock 99th percentile, real 99th percentile, real fraction above the mock 99th
            re_ = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 2.0])  # log10 RUWE
            for a, b in zip(re_[:-1], re_[1:]):
                lr, lm = np.log10(r["ruwe"]), np.log10(m["ruwe"])
                sr, sm = (lr >= a) & (lr < b), (lm >= a) & (lm < b)
                q_m = wquant(m["a0snr"][sm], w[sm], 0.99)
                q_r = float(np.nanpercentile(r["a0snr"][sr], 99)) if sr.sum() > 10 else float("nan")
                frac = float(np.mean(r["a0snr"][sr] > q_m)) if sr.sum() and np.isfinite(q_m) else float("nan")
                out[f"a0/err edge, log RUWE {a:g}-{b:g}"] = {"real_p99": q_r, "mock_p99": q_m, "real_frac_above_mock_p99": frac, "n_real": int(sr.sum())}
        for t in (1.4, 3.0):
            out[f"N(RUWE > {t:g})"] = count(r["ruwe"] > t, m["ruwe"] > t, w)
        out["RUWE median"] = {"real": float(np.nanmedian(r["ruwe"])), "mock": wquant(m["ruwe"], w, 0.5)}
        out["N total"] = count(np.ones(r["g"].size, bool), np.ones(w.size, bool), w)
    return out


def side_by_side_2d(kinds, kname, ka, kb, edges, out, style, dpi):
    plt = pl.require_pyplot()
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    la, lb = kinds[0][1].obs[ka], kinds[0][1].obs[kb]
    ea = np.linspace(edges[ka][0], edges[ka][-1], rp.NB_2D + 1)
    eb = np.linspace(edges[kb][0], edges[kb][-1], rp.NB_2D + 1)
    k = kinds[0][1].real_hist([ka, kb], [ea, eb], [la[1], lb[1]])
    mus = [(n, *kd.mock_hist([ka, kb], [ea, eb], [la[1], lb[1]])[:2]) for n, kd in kinds]
    vmax = max([np.nanmax(k)] + [np.nanmax(m) for _, m, _ in mus] + [1.0])
    vmin = max(min([np.min(k[k > 0]) if np.any(k > 0) else 1.0] + [np.min(m[m > 0]) if np.any(m > 0) else 1.0 for _, m, _ in mus]), 0.3)
    n = len(mus) + 1
    fig, axs = plt.subplots(2, n + 1, figsize=(4.2 * n + 0.6, 7.6), squeeze=False, gridspec_kw={"width_ratios": [1.0] * n + [0.06]})
    xl = (r"$\log_{10}$ " if la[1] else "") + la[0]
    yl = (r"$\log_{10}$ " if lb[1] else "") + lb[0]
    norm = LogNorm(vmin=vmin, vmax=vmax)
    axs[0, 0].pcolormesh(ea, eb, np.where(k > 0, k, np.nan).T, norm=norm, cmap="viridis")
    pl.apply_axes_style(axs[0, 0], style, xlabel=xl, ylabel=yl, title="DR3")
    axs[1, 0].axis("off")
    res_all = []
    for j, (nm, mu, var) in enumerate(mus, start=1):
        mm = axs[0, j].pcolormesh(ea, eb, np.where(mu > 0, mu, np.nan).T, norm=norm, cmap="viridis")
        c2, nb = rp.chi2(k, mu, var)
        pl.apply_axes_style(axs[0, j], style, xlabel=xl, ylabel=yl, title=f"{nm}: χ²/dof {c2 / max(nb, 1):.1f}")
        with np.errstate(divide="ignore", invalid="ignore"):
            res_all.append(np.where((mu > 0) | (k > 0), (k - mu) / np.sqrt(np.maximum(mu, rp.MU_FLOOR) + var), np.nan))
    fig.colorbar(mm, cax=axs[0, -1], label="count")
    lim = max(3.0, float(np.nanpercentile(np.abs(np.concatenate([r_.ravel() for r_ in res_all])), 98)))
    for j, (res, (nm, _, _)) in enumerate(zip(res_all, mus), start=1):
        mr = axs[1, j].pcolormesh(ea, eb, res.T, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
        pl.apply_axes_style(axs[1, j], style, xlabel=xl, ylabel=yl, title=f"{nm}: residual")
    fig.colorbar(mr, cax=axs[1, -1], label=r"$(k-\mu)/\sqrt{\mu+\sigma^2_{\rm MC}}$")
    fig.suptitle(f"{kname}: {la[0]} vs {lb[0]} (DR3 | mocks; residuals below)")
    pl.save_figure(fig, out, dpi=dpi)


def side_by_side_1d(kinds, kname, key, edges, out, style, dpi, subset=None):
    plt = pl.require_pyplot()
    lab, lg = kinds[0][1].obs[key]
    e = np.asarray(edges[key], float)
    kd0 = kinds[0][1]
    rsel = None if subset is None else SUBSETS[subset](kd0.real)
    k = kd0.real_hist([key], [e], [lg], rsel)
    c = 0.5 * (e[1:] + e[:-1])
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax.errorbar(c, k, yerr=np.sqrt(k), fmt="o", color="k", ms=4, label="DR3", zorder=5)
    for i, (nm, kd) in enumerate(kinds):
        sel = None if subset is None else {cn: SUBSETS[subset](v) for cn, (v, _) in kd.comp.items()}
        mu, var, _ = kd.mock_hist([key], [e], [lg], sel)
        st = pl.series_style(i, style)
        c2, nb = rp.chi2(k, mu, var)
        ax.step(e, np.r_[mu, mu[-1]], where="post", color=st["color"], ls=st["linestyle"], lw=st["linewidth"], label=f"{nm} (χ²/dof {c2 / max(nb, 1):.1f})")
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = k / mu
            re = np.sqrt(k / mu**2 + k**2 * (var + mu) / mu**4)
        ar.errorbar(c + (i - len(kinds) / 2) * 0.12 * np.diff(e).mean(), rt, yerr=re, fmt="o", color=st["color"], ms=3)
    ax.set_yscale("log")
    pl.apply_axes_style(ax, style, ylabel="count", title=f"{kname}{'' if subset is None else ', ' + subset}: {lab}")
    ax.legend(fontsize=9, frameon=False)
    ar.axhline(1.0, color="0.4", ls="--")
    ar.set_ylim(0, 3)
    pl.apply_axes_style(ar, style, xlabel=(r"$\log_{10}$ " if lg else "") + lab, ylabel="DR3 / mock")
    pl.save_figure(fig, out, dpi=dpi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=DUMP:RANK_DIR[:drop,comps]")
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = load_config()
    style, dpi = cfg.plotting, int(cfg.diagnostics.figure_dpi) if hasattr(cfg.diagnostics, "figure_dpi") else 120
    runs = [parse_run(s) for s in args.run]
    edges = json.loads(args.edges.read_text())
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    # χ² before/after
    ranks = {r["name"]: {(x["kind"], x["distribution"], x["split"]): x for x in json.loads((r["rank"] / "ranking.json").read_text())} for r in runs}
    keys = list(ranks[runs[0]["name"]])
    rows = []
    for k_ in keys:
        row = {"kind": k_[0], "distribution": k_[1], "split": k_[2], "noise_sensitive": (k_[0], k_[1]) in NOISE_DISTS}
        for r in runs:
            x = ranks[r["name"]].get(k_)
            row[r["name"]] = None if x is None else {"chi2_dof": x["chi2_dof"], "mc_err_dof": (x["chi2_mc_err"] / max(x["dof"], 1)) if x["chi2_mc_err"] is not None else None, "dof": x["dof"]}
        rows.append(row)
    (out / "before_after.json").write_text(json.dumps(rows, indent=1))
    names = [r["name"] for r in runs]

    def cell(v):
        if v is None:
            return "–"
        return f"{v['chi2_dof']:.2f}" + (f" ± {v['mc_err_dof']:.2f}" if v["mc_err_dof"] is not None else "")

    md = ["| kind | distribution | split | " + " | ".join(names) + " |", "|---|---|---|" + "---:|" * len(names)]
    for row in sorted(rows, key=lambda r_: (not r_["noise_sensitive"], r_["kind"], r_["split"] != "all", r_["distribution"])):
        md.append(f"| {row['kind']} | {row['distribution']}{' (N)' if row['noise_sensitive'] else ''} | {row['split']} | " + " | ".join(cell(row[n]) for n in names) + " |")
    (out / "before_after.md").write_text("χ²/dof ± bootstrap MC error per dof (30-fold), same bin edges for every run; (N) = noise-sensitive.\n\n" + "\n".join(md) + "\n")
    # noise metrics
    allm = {}
    figs = []
    for kname, pfx, obs in (("orbits", "o", rp.ORBIT_OBS), ("accelerations", "a", rp.ACCEL_OBS)):
        kinds = []
        for r in runs:
            z = np.load(r["dump"])
            kinds.append((r["name"], rp.Kind(z, pfx, obs, drop=r["drop"])))
        real = kinds[0][1].real
        thr = {"plxsnr_hi": float(np.nanpercentile(real["plxsnr"], 90))}
        allm[kname] = {"thresholds": thr, "runs": {nm: metrics(kd, kname, thr) for nm, kd in kinds}}
        for ka, kb in PAIRS[kname]:
            fn = f"{kname}_2d_{ka}_{kb}_compare.png"
            side_by_side_2d(kinds, kname, ka, kb, edges[kname], out / fn, style, dpi)
            figs.append(fn)
        for key in ONED[kname]:
            for sub in (None, "G<13", "G 13-16"):
                fn = f"{kname}_1d_{key}{'' if sub is None else '_' + sub.replace('<', 'lt').replace(' ', '')}_compare.png"
                side_by_side_1d(kinds, kname, key, edges[kname], out / fn, style, dpi, subset=sub)
                figs.append(fn)
        print(f"{kname}: done", flush=True)
    (out / "noise_metrics.json").write_text(json.dumps(allm, indent=1))
    mdm = []
    for kname, blk in allm.items():
        mdm += [f"## {kname} (ϖ/σ_ϖ threshold = DR3 90th percentile {blk['thresholds']['plxsnr_hi']:.1f})", "",
                "| statistic | DR3 | " + " | ".join(f"{n} mock | {n} DR3/mock (pull) [shape]" for n in names) + " |", "|---|---:|" + "---:|---:|" * len(names)]
        for stat in blk["runs"][names[0]]:
            v0 = blk["runs"][names[0]][stat]
            if "pull" in v0:
                line = f"| {stat} | {v0['real']:.0f} | " + " | ".join(
                    f"{blk['runs'][n][stat]['mock']:.0f} ± {blk['runs'][n][stat]['mock_mc']:.0f} | {blk['runs'][n][stat]['ratio']:.2f} ({blk['runs'][n][stat]['pull']:+.1f}) [{blk['runs'][n][stat]['shape_ratio']:.2f}]" for n in names) + " |"
            elif "real_p99" in v0:
                line = f"| {stat} (n={v0['n_real']}): p99 / frac DR3 > mock p99 | {v0['real_p99']:.1f} | " + " | ".join(
                    f"{blk['runs'][n][stat]['mock_p99']:.1f} | {blk['runs'][n][stat]['real_frac_above_mock_p99']:.3f}" for n in names) + " |"
            else:
                line = f"| {stat} | {v0['real']:.2f} | " + " | ".join(f"{blk['runs'][n][stat]['mock']:.2f} | – " for n in names) + " |"
            mdm.append(line)
        mdm.append("")
    (out / "noise_metrics.md").write_text("\n".join(mdm) + "\n")
    (out / "figures.json").write_text(json.dumps(figs))
    print(f"wrote {len(figs)} comparison figures, before_after.*, noise_metrics.* -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
