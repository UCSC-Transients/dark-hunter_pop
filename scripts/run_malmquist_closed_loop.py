#!/usr/bin/env python3
"""Closed-loop proof of the Malmquist / Öpik conditioning weight: figures + numbers (#405).

docs/MOCK_POPULATION_SPEC.md §9.6. Builds the synthetic universe of
``config/population/malmquist_closed_loop.yaml``, observes it with G_total < 19 and
ϖ_obs > 0.2 mas, runs the Gaia-star-primary mock on the synthetic parent with and without the
weight W, and writes, in ``--out-dir``:

- ``<prefix>_parent_binary_fraction.png``: parent binary fraction vs G and vs M̂1, with pulls;
- ``<prefix>_parent_companions.png``: parent companion log P, q, e, log f counts, with pulls;
- ``<prefix>_observed_alpha0.png``: photocentre semi-major axis α0 of the parent's systems;
- ``<prefix>_volume_limited.png``: 1/V_S-inverted binary fraction vs M1 and shapes vs the
  injected universe;
- ``<prefix>_results.json``: every number in the figures.

``--sigma-int-pipeline`` / ``--zero-point-pipeline`` run the misspecification experiments
(MP-Q25): the pipeline's σ_int or M_G zero point differs from the synthetic truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import malmquist_closed_loop as cl
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

LABELS = {
    "log_p": r"$\log_{10}(P$ / d$)$",
    "q": r"$q = M_2 / M_1$",
    "eccentricity": r"$e$",
    "log10_f": r"$\log_{10} f_G$ (flux ratio)",
}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[1:] + edges[:-1])


def _pull_panel(ax: Any, x: np.ndarray, c: cl.CountComparison, style: Any, xlabel: str) -> None:
    s1, s2 = series_style(1, style), series_style(2, style)
    ax.axhspan(-3, 3, color="0.9", zorder=0)
    ax.axhline(0, color="0.4", lw=1)
    ax.plot(x, c.pull_naive, marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="no weight W")
    ax.plot(x, c.pull_corrected, marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="with W")
    lim = max(4.0, float(np.nanmax(np.abs(np.r_[c.pull_naive, c.pull_corrected]))) * 1.15)
    ax.set_ylim(-lim, lim)
    apply_axes_style(ax, style, xlabel=xlabel, ylabel="pull")


def fig_binary_fraction(res: cl.ClosedLoopResult, style: Any, path: Path, dpi: int, tag: str) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    s0, s1, s2 = series_style(0, style), series_style(1, style), series_style(2, style)
    for j, (c, xl) in enumerate(
        ((res.parent_binary_by_g, r"$G$ (total system light, mag)"), (res.parent_binary_by_m1, r"$\hat M_1$ ($M_\odot$)"))
    ):
        x = _centres(c.edges)
        n = np.maximum(c.n_group, 1)
        ft = c.truth / n
        ax = axes[0, j]
        ax.errorbar(x, ft, yerr=np.sqrt(ft * (1 - ft) / n), fmt="o", color=s0["color"], ms=6, label="truth (synthetic parent)")
        ax.plot(x, c.naive / n, marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="mock, no weight W")
        ax.plot(x, c.corrected / n, marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="mock, with W")
        apply_axes_style(ax, style, ylabel="binary fraction in parent")
        if j == 0:
            ax.legend(loc="best", fontsize=10)
        _pull_panel(axes[1, j], x, c, style, xl)
    t = res.parent_binary_total
    fig.suptitle(
        f"{tag}: parent binary fraction (luminous MdS17 companions, q > 0.1)\n"
        f"N_parent = {int(t['n_rows'])}; total truth {int(t['truth'])}, no-W {t['naive']:.0f} "
        f"(pull {t['pull_naive']:.1f}), with-W {t['corrected']:.0f} (pull {t['pull_corrected']:.1f})",
        fontsize=12,
    )
    save_figure(fig, path, dpi=dpi)


def fig_companions(res: cl.ClosedLoopResult, style: Any, path: Path, dpi: int, tag: str) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    s0, s1, s2 = series_style(0, style), series_style(1, style), series_style(2, style)
    for j, (name, c) in enumerate(res.parent_shapes.items()):
        x = _centres(c.edges)
        ax = axes[0, j]
        ax.errorbar(x, c.truth, yerr=np.sqrt(np.maximum(c.truth, 1)), fmt="o", color=s0["color"], ms=6, label="truth")
        ax.plot(x, c.naive, marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="no W")
        ax.plot(x, c.corrected, marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="with W")
        apply_axes_style(ax, style, ylabel="companions in parent" if j == 0 else None)
        if j == 0:
            ax.legend(fontsize=10)
        _pull_panel(axes[1, j], x, c, style, LABELS[name])
    fig.suptitle(f"{tag}: companion statistics of the magnitude-limited parent (bin counts; pulls include MC noise)", fontsize=12)
    save_figure(fig, path, dpi=dpi)


def fig_alpha0(res: cl.ClosedLoopResult, style: Any, path: Path, dpi: int, tag: str) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    s0, s1, s2 = series_style(0, style), series_style(1, style), series_style(2, style)
    c = res.alpha0
    x = np.sqrt(c.edges[1:] * c.edges[:-1])
    axes[0].errorbar(x, c.truth, yerr=np.sqrt(np.maximum(c.truth, 1)), fmt="o", color=s0["color"], ms=6, label="truth (true distance)")
    axes[0].plot(x, c.naive, marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="mock, no W")
    axes[0].plot(x, c.corrected, marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="mock, with W (BJ-like distance)")
    axes[0].set_xscale("log")
    apply_axes_style(axes[0], style, ylabel="systems in parent")
    axes[0].legend(fontsize=10)
    _pull_panel(axes[1], x, c, style, r"photocentre semi-major axis $\alpha_0$ (mas)")
    w = res.nss_window
    fig.suptitle(
        f"{tag}: observed astrometric signal of the parent's binaries\n"
        f"NSS-like window (α0 > 0.3 mas, 30 < P < 1600 d): truth {w['truth']:.0f}, "
        f"no-W {w['naive']:.1f} (pull {w['pull_naive']:.1f}), with-W {w['corrected']:.1f} (pull {w['pull_corrected']:.1f})",
        fontsize=11,
    )
    save_figure(fig, path, dpi=dpi)


def fig_volume(res: cl.ClosedLoopResult, style: Any, path: Path, dpi: int, tag: str) -> None:
    plt = require_pyplot()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    s0, s1, s2, s3 = (series_style(i, style) for i in range(4))
    v = res.volume
    x = _centres(v.m1_edges)
    ax = axes[0]
    ax.errorbar(x, v.truth_fraction, yerr=v.truth_fraction_err, fmt="o", color=s0["color"], ms=7, label="injected (universe)")
    ax.plot(x, v.parent_truth_fraction, marker="^", color=s3["color"], ls=":", lw=2, label="parent truth / V_S")
    ax.errorbar(x, v.naive_fraction, yerr=v.naive_fraction_err, marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="mock no W / V_S")
    ax.errorbar(x, v.corrected_fraction, yerr=v.corrected_fraction_err, marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="mock with W / V_S")
    apply_axes_style(ax, style, xlabel=r"$M_1$ ($M_\odot$)", ylabel="volume-limited binary fraction")
    ax.legend(fontsize=9)
    for j, (name, s) in enumerate(v.shapes.items(), start=1):
        ax = axes[j]
        xc = _centres(s["edges"])
        ax.errorbar(xc, s["truth"], yerr=s["truth_err"], fmt="o", color=s0["color"], ms=6, label="injected")
        ax.errorbar(xc, s["naive"], yerr=s["naive_err"], marker="s", color=s1["color"], ls="--", lw=s1["linewidth"], label="no W")
        ax.errorbar(xc, s["corrected"], yerr=s["corrected_err"], marker="o", color=s2["color"], ls="-", lw=s2["linewidth"], label="with W")
        apply_axes_style(ax, style, xlabel=LABELS[name], ylabel="fraction of companions" if j == 1 else None)
    fig.suptitle(f"{tag}: parent inverted to a volume-limited population with 1/V_S (spec §9.5) vs the injected MdS17", fontsize=12)
    save_figure(fig, path, dpi=dpi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="large", help="'small', 'large' or an integer universe size")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/gate405/figures"))
    ap.add_argument("--prefix", default="closed_loop")
    ap.add_argument("--sigma-int-pipeline", type=float, default=None, help="MP-Q25 misspecification: pipeline σ_int")
    ap.add_argument("--zero-point-pipeline", type=float, default=None, help="MP-Q25 misspecification: pipeline M_G offset")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    size: Any = int(args.size) if args.size.isdigit() else args.size
    res, raw = cl.run_closed_loop(
        size, sigma_int_pipeline=args.sigma_int_pipeline, zero_point_pipeline=args.zero_point_pipeline
    )
    tag = f"synthetic closed loop ({res.counts['universe']:,} primaries"
    if args.sigma_int_pipeline is not None:
        tag += f", pipeline σ_int={args.sigma_int_pipeline}"
    if args.zero_point_pipeline is not None:
        tag += f", pipeline M_G offset={args.zero_point_pipeline}"
    tag += ")"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "tag": tag,
        "counts": res.counts,
        "ess": res.ess,
        "sigma_mu_quantiles_16_50_84_95": res.sigma_mu_quantiles,
        "parent_binary_total": res.parent_binary_total,
        "parent_binary_by_g": asdict(res.parent_binary_by_g),
        "parent_binary_by_m1": asdict(res.parent_binary_by_m1),
        "parent_shapes": {k: asdict(v) for k, v in res.parent_shapes.items()},
        "alpha0": asdict(res.alpha0),
        "nss_window": res.nss_window,
        "volume": asdict(res.volume),
    }
    (args.out_dir / f"{args.prefix}_results.json").write_text(json.dumps(_jsonable(out), indent=1))
    if not args.no_figures:
        cfg = load_config()
        style = resolve_plotting_style(cfg.plotting)
        dpi = int(cfg.diagnostics.figure_dpi)
        fig_binary_fraction(res, style, args.out_dir / f"{args.prefix}_parent_binary_fraction.png", dpi, tag)
        fig_companions(res, style, args.out_dir / f"{args.prefix}_parent_companions.png", dpi, tag)
        fig_alpha0(res, style, args.out_dir / f"{args.prefix}_observed_alpha0.png", dpi, tag)
        fig_volume(res, style, args.out_dir / f"{args.prefix}_volume_limited.png", dpi, tag)
    t = res.parent_binary_total
    print(tag, json.dumps(_jsonable({"counts": res.counts, "ess": res.ess, "total": t, "nss": res.nss_window})))


if __name__ == "__main__":
    main()
