#!/usr/bin/env python3
"""Rung-3 data comparison plots: DR3 against the weighted mock, every 1-D and pairwise 2-D observable (#570, spec §12.39).

Reads ``plot_dump.npz`` from ``scripts/rung3_joint_391.py`` (per-draw observables and weights at the fit: mock luminous,
mock WD, real; spurious = real orbits weighted by N_s π̂ / Σπ̂). Mock = luminous + WD + spurious, components shown.

* 1-D: real histogram with Poisson errors; mock with an MC + Poisson band (σ² = Σw² + μ); ratio panel. Also split by
  distance (< 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc).
* 2-D (every pair): real and mock on the same colour scale and the normalised residual (k − μ)/√(μ + Σw²).
* Ranking: χ² = Σ (k − μ)² / (μ + Σw²) per distribution (bins with μ > 0), χ²/dof, and its MC error from a bootstrap of
  the mock draws (luminous and WD resampled with replacement).

Writes PNGs, ``README.md``, ``index.html`` (relative links) and ``ranking.json`` under ``--out-dir``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/rung3_plots_391.py --dump output/v3/fit/plot_dump.npz --out-dir docs/gate391/rung3_v3/plots
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import plotting as pl
from darkhunter_pop.config_loader import load_config

# name -> (label, log10?)
ORBIT_OBS = {"g": ("$G$ (mag)", False), "d": ("distance (kpc)", True), "colour0": (r"$(G_{\rm BP}-G_{\rm RP})_0$ (mag)", False),
             "mg0": (r"$M_{G,0}$ (mag)", False), "P": ("$P$ (d)", True), "e": ("$e$", False), "A": (r"$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)", True),
             "a0snr": (r"$a_0/\sigma_{a_0}$", True), "plxsnr": (r"$\varpi/\sigma_\varpi$", True), "f2": ("$F_2$", False), "ruwe": ("RUWE", True)}
ACCEL_OBS = {"g": ORBIT_OBS["g"], "d": ORBIT_OBS["d"], "colour0": ORBIT_OBS["colour0"], "mg0": ORBIT_OBS["mg0"],
             "plxsnr": ORBIT_OBS["plxsnr"], "signif": ("significance $s$", True), "f2": ("$F_2$", False), "ruwe": ("RUWE", True)}
D_BINS = (0.0, 0.3, 0.7, 1.5, 6.0)
N_BOOT = 30
NB_1D, NB_2D = 25, 18


def _x(v, log):
    v = np.asarray(v, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(v) if log else v


def hdd(xs, edges, w=None):
    """histogramdd over finite rows only."""
    a = np.column_stack(xs)
    ok = np.all(np.isfinite(a), axis=1)
    return np.histogramdd(a[ok], bins=edges, weights=None if w is None else np.asarray(w)[ok])[0]


def edges_for(real, mock, w, log, nb):
    xr, xm = _x(real, log), _x(mock, log)
    pool = np.concatenate([xr[np.isfinite(xr)], xm[np.isfinite(xm) & (w > 0)]])
    lo, hi = np.percentile(pool, [0.5, 99.5]) if pool.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nb + 1)


class Kind:
    """One sample (orbits or accelerations): real arrays, mock components (luminous, WD, spurious) with weights."""

    def __init__(self, z, pfx, obs):
        self.obs = obs
        self.real = {k: np.asarray(z[f"{pfx}_real_{k}"], float) for k in obs}
        self.comp = {}
        for c in ("lum", "wd"):
            self.comp[c] = ({k: np.asarray(z[f"{pfx}_{c}_{k}"], float) for k in obs}, np.asarray(z[f"{pfx}_{c}_w"], float))
        if f"{pfx}_spur_w" in z.files:
            self.comp["spurious"] = (self.real, np.asarray(z[f"{pfx}_spur_w"], float))

    def mock_hist(self, cols, edges, logs, sel=None, boot=None):
        mu, var, parts = 0.0, 0.0, {}
        for c, (vals, w) in self.comp.items():
            m = np.ones(w.size, bool) if sel is None else sel[c]
            idx = np.flatnonzero(m)
            if boot is not None and c != "spurious":
                idx = idx[boot[c](idx.size)]
            xs = [_x(vals[k][idx], lg) for k, lg in zip(cols, logs)]
            h = hdd(xs, edges, w[idx])
            h2 = hdd(xs, edges, w[idx] ** 2)
            parts[c] = h
            mu = mu + h
            var = var + h2
        return mu, var, parts

    def real_hist(self, cols, edges, logs, sel=None):
        m = np.ones(self.real[cols[0]].size, bool) if sel is None else sel
        xs = [_x(self.real[k][m], lg) for k, lg in zip(cols, logs)]
        return hdd(xs, edges)

    def dsel(self, lo, hi):
        sel = {c: (vals["d"] >= lo) & (vals["d"] < hi) for c, (vals, _) in self.comp.items()}
        return sel, (self.real["d"] >= lo) & (self.real["d"] < hi)


def chi2(k, mu, var):
    ok = mu > 0
    return float(np.sum((k[ok] - mu[ok]) ** 2 / (mu[ok] + var[ok]))), int(ok.sum())


def boot_chi2(kind, cols, edges, logs, k, sel=None, rsel=None, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        b = {c: (lambda n, r=rng: r.integers(0, n, n)) for c in ("lum", "wd")}
        mu, var, _ = kind.mock_hist(cols, edges, logs, sel=sel, boot=b)
        vals.append(chi2(k, mu, var)[0])
    return float(np.std(vals))


def plot_1d(kind, key, label, log, edges, out, style, dpi, sel=None, rsel=None, title=""):
    plt = pl.require_pyplot()
    k = kind.real_hist([key], [edges], [log], rsel)
    mu, var, parts = kind.mock_hist([key], [edges], [log], sel)
    c = 0.5 * (edges[1:] + edges[:-1])
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    band = np.sqrt(var + mu)
    s0 = pl.series_style(0, style)
    ax.fill_between(edges, np.r_[mu - band, (mu - band)[-1]], np.r_[mu + band, (mu + band)[-1]], step="post", color=s0["color"], alpha=0.3, label="mock (MC+Poisson)")
    ax.step(edges, np.r_[mu, mu[-1]], where="post", color=s0["color"], lw=s0["linewidth"], label="mock total")
    for i, (cn, h) in enumerate(parts.items(), start=1):
        st = pl.series_style(i, style)
        ax.step(edges, np.r_[h, h[-1]], where="post", color=st["color"], ls=st["linestyle"], lw=0.7 * st["linewidth"], label=f"mock {cn}")
    ax.errorbar(c, k, yerr=np.sqrt(k), fmt="o", color="k", ms=4, label="DR3")
    ax.set_yscale("log" if np.nanmax(k) > 30 * max(np.nanmedian(k[k > 0]) if np.any(k > 0) else 1, 1) else "linear")
    pl.apply_axes_style(ax, style, ylabel="count", title=title)
    ax.legend(fontsize=9, frameon=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = k / mu
        re = np.sqrt(k / mu**2 + k**2 * (var + mu) / mu**4)
    ar.errorbar(c, rt, yerr=re, fmt="o", color="k", ms=4)
    ar.axhline(1.0, color="0.4", ls="--")
    ar.set_ylim(0, 2.5)
    pl.apply_axes_style(ar, style, xlabel=(r"$\log_{10}$ " if log else "") + label, ylabel="DR3 / mock")
    pl.save_figure(fig, out, dpi=dpi)
    return chi2(k, mu, var)


def plot_2d(kind, ka, kb, la, lb, ea, eb, out, style, dpi, title):
    plt = pl.require_pyplot()
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    k = kind.real_hist([ka, kb], [ea, eb], [la[1], lb[1]])
    mu, var, _ = kind.mock_hist([ka, kb], [ea, eb], [la[1], lb[1]])
    vmax = max(np.nanmax(k), np.nanmax(mu), 1.0)
    vmin = max(min(np.min(k[k > 0]) if np.any(k > 0) else 1.0, np.min(mu[mu > 0]) if np.any(mu > 0) else 1.0), 0.3)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(mu > 0, (k - mu) / np.sqrt(mu + var), np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, data, ttl in ((axs[0], k, "DR3"), (axs[1], mu, "mock (lum + WD + spurious)")):
        m = ax.pcolormesh(ea, eb, np.where(data > 0, data, np.nan).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        fig.colorbar(m, ax=ax, label="count")
        pl.apply_axes_style(ax, style, xlabel=(r"$\log_{10}$ " if la[1] else "") + la[0], ylabel=(r"$\log_{10}$ " if lb[1] else "") + lb[0], title=ttl)
    lim = max(3.0, float(np.nanpercentile(np.abs(res), 98)) if np.isfinite(res).any() else 3.0)
    m = axs[2].pcolormesh(ea, eb, res.T, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
    fig.colorbar(m, ax=axs[2], label=r"$(k-\mu)/\sqrt{\mu+\sigma^2_{\rm MC}}$")
    pl.apply_axes_style(axs[2], style, xlabel=(r"$\log_{10}$ " if la[1] else "") + la[0], ylabel=(r"$\log_{10}$ " if lb[1] else "") + lb[0], title="normalised residual")
    fig.suptitle(title)
    pl.save_figure(fig, out, dpi=dpi)
    return chi2(k, mu, var)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None, help="fit summary.json, for the README header")
    args = ap.parse_args(argv)
    cfg = load_config()
    style, dpi = cfg.plotting, int(cfg.diagnostics.figure_dpi) if hasattr(cfg.diagnostics, "figure_dpi") else 120
    z = np.load(args.dump)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rank, figs = [], []
    for kname, pfx, obs in (("orbits", "o", ORBIT_OBS), ("accelerations", "a", ACCEL_OBS)):
        kind = Kind(z, pfx, obs)
        mw = np.concatenate([w for _, (_, w) in kind.comp.items()])
        edges = {}
        for key, (lab, lg) in obs.items():
            mv = np.concatenate([vals[key] for _, (vals, _) in kind.comp.items()])
            edges[key] = edges_for(kind.real[key], mv, mw, lg, NB_1D)
            fn = f"{kname}_1d_{key}.png"
            c2, nb = plot_1d(kind, key, lab, lg, edges[key], out / fn, style, dpi, title=f"{kname}: {lab}")
            err = boot_chi2(kind, [key], [edges[key]], [lg], kind.real_hist([key], [edges[key]], [lg]))
            rank.append({"kind": kname, "distribution": key, "dims": 1, "split": "all", "chi2": c2, "dof": nb, "chi2_dof": c2 / max(nb, 1), "chi2_mc_err": err, "figure": fn})
            figs.append((kname, "1-D", fn, lab))
            # by distance
            for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
                sel, rsel = kind.dsel(lo, hi)
                fn = f"{kname}_1d_{key}_d{lo:g}-{hi:g}.png"
                c2, nb = plot_1d(kind, key, lab, lg, edges[key], out / fn, style, dpi, sel=sel, rsel=rsel, title=f"{kname}, d {lo:g}–{hi:g} kpc: {lab}")
                k_ = kind.real_hist([key], [edges[key]], [lg], rsel)
                err = boot_chi2(kind, [key], [edges[key]], [lg], k_, sel=sel)
                rank.append({"kind": kname, "distribution": key, "dims": 1, "split": f"d{lo:g}-{hi:g}", "chi2": c2, "dof": nb, "chi2_dof": c2 / max(nb, 1), "chi2_mc_err": err, "figure": fn})
                figs.append((kname, f"1-D, d {lo:g}–{hi:g} kpc", fn, lab))
        for (ka, la), (kb, lb) in itertools.combinations(obs.items(), 2):
            ea = np.linspace(edges[ka][0], edges[ka][-1], NB_2D + 1)
            eb = np.linspace(edges[kb][0], edges[kb][-1], NB_2D + 1)
            fn = f"{kname}_2d_{ka}_{kb}.png"
            c2, nb = plot_2d(kind, ka, kb, la, lb, ea, eb, out / fn, style, dpi, title=f"{kname}: {la[0]} vs {lb[0]}")
            rank.append({"kind": kname, "distribution": f"{ka} x {kb}", "dims": 2, "split": "all", "chi2": c2, "dof": nb, "chi2_dof": c2 / max(nb, 1), "chi2_mc_err": None, "figure": fn})
            figs.append((kname, "2-D", fn, f"{la[0]} vs {lb[0]}"))
        print(f"{kname}: done", flush=True)
    rank.sort(key=lambda r: -r["chi2_dof"])
    (out / "ranking.json").write_text(json.dumps(rank, indent=1))
    head = ""
    if args.summary and args.summary.exists():
        r = json.loads(args.summary.read_text())["results"]["joint"]
        head = f"Fit: -lnL {r['nll']:.1f}; {r['n_near']}/{r['n_starts']} starts within 5; N_s pull {r['N_s_pull']:+.1f}; at bound {r['at_bound']}.\n\n"
    md = ["# Rung 3 v3: DR3 against the weighted mock (luminous + WD + spurious)", "",
          "Spec §12.39 (#570). Mock = luminous (gens 42–46) + WD (co_v1 gens 1000–1003, physics-derived rate) + spurious (real orbits × N_s π̂/Σπ̂). "
          "1-D: DR3 with Poisson errors, mock total with an MC + Poisson band, components, ratio panel. 2-D: DR3 | mock | (k−μ)/√(μ+σ²_MC). "
          "χ² = Σ (k−μ)²/(μ+σ²_MC) over bins with μ > 0; MC error from a 30-fold bootstrap of the mock draws.", "", head,
          "## Ranking (worst first, χ²/dof)", "", "| kind | distribution | split | χ²/dof | χ² ± MC | figure |", "|---|---|---|---:|---:|---|"]
    for r_ in rank:
        e_ = f" ± {r_['chi2_mc_err']:.1f}" if r_["chi2_mc_err"] is not None else ""
        md.append(f"| {r_['kind']} | {r_['distribution']} | {r_['split']} | {r_['chi2_dof']:.2f} | {r_['chi2']:.1f}{e_} | [{r_['figure']}]({r_['figure']}) |")
    md += ["", "## All figures", ""] + [f"- {k_} {t_}: [{lab_}]({fn_})" for k_, t_, fn_, lab_ in figs]
    (out / "README.md").write_text("\n".join(md) + "\n")
    rows = "".join(f"<tr><td>{r_['kind']}</td><td>{html.escape(r_['distribution'])}</td><td>{r_['split']}</td><td>{r_['chi2_dof']:.2f}</td>"
                   f"<td>{r_['chi2']:.1f}{(' ± %.1f' % r_['chi2_mc_err']) if r_['chi2_mc_err'] is not None else ''}</td><td><a href=\"{r_['figure']}\">{r_['figure']}</a></td></tr>" for r_ in rank)
    sections = ""
    for kname in ("orbits", "accelerations"):
        for typ in sorted({t_ for k_, t_, _, _ in figs if k_ == kname}, key=lambda x: (x != "1-D", x != "2-D", x)):
            imgs = "".join(f"<figure><a href=\"{fn_}\"><img src=\"{fn_}\" loading=\"lazy\" alt=\"{html.escape(lab_)}\"></a><figcaption>{html.escape(lab_)}</figcaption></figure>"
                           for k_, t_, fn_, lab_ in figs if k_ == kname and t_ == typ)
            sections += f"<h2>{kname}: {html.escape(typ)}</h2><div class=\"grid\">{imgs}</div>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rung 3 v3 comparison</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#555;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16181c;--fg:#eee;--mut:#aaa;--line:#333}}}}
:root[data-theme="dark"]{{--bg:#16181c;--fg:#eee;--mut:#aaa;--line:#333}}
body{{background:var(--bg);color:var(--fg);font-family:Georgia,serif;margin:0 auto;max-width:1400px;padding:16px}}
table{{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}img{{width:100%;background:#fff}}figcaption{{color:var(--mut);font-size:13px}}
</style></head><body><h1>Rung 3 v3: DR3 vs weighted mock (luminous + WD + spurious)</h1>
<p>Spec §12.39 (#570). {html.escape(head.strip())} χ² = Σ (k−μ)²/(μ+σ²<sub>MC</sub>); MC error from a 30-fold bootstrap of the mock draws.</p>
<h2>Ranking (worst first)</h2><table><tr><th>kind</th><th>distribution</th><th>split</th><th>χ²/dof</th><th>χ² ± MC</th><th>figure</th></tr>{rows}</table>
{sections}</body></html>"""
    (out / "index.html").write_text(page)
    print(f"wrote {len(figs)} figures, README.md, index.html, ranking.json -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
