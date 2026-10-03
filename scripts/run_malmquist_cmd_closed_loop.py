#!/usr/bin/env python3
"""Run the #418 closed loop (2-D CMD Malmquist weight with MIST photometry) and write figures.

Spec §11.5. Writes ``closed_loop_cmd_<size>.json`` and ``closed_loop_cmd_<size>_*.png`` to
``--out-dir``: binary fraction vs G / M̂1 / colour and companion shapes, truth vs no W vs the 1-D
§9.3 weight vs the 2-D §11.4 weight, with pulls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from darkhunter_pop import malmquist_cmd_closed_loop as cl
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

LABELS = {"none": "no W", "one_d": "1-D W (§9.3, isochrone M1)", "two_d": "2-D W (§11.4)"}


def _panel(ax_top, ax_bot, tab: dict, xlabel: str, style: dict, *, frac: bool, logx: bool = False) -> None:
    e = np.asarray(tab["edges"], float)
    x = 0.5 * (e[1:] + e[:-1]) if not logx else np.sqrt(np.clip(e[1:], 1e-9, None) * np.clip(e[:-1], 1e-9, None))
    norm = np.asarray(tab["n_group"], float) if frac else np.ones(x.size)
    t = np.asarray(tab["truth"], float) / np.where(norm > 0, norm, 1)
    ax_top.plot(x, t, color="k", lw=3, marker="o", label="truth (parent)")
    for k, key in enumerate(("none", "one_d", "two_d")):
        y = np.asarray(tab[key], float) / np.where(norm > 0, norm, 1)
        st = series_style(k, style)
        ax_top.plot(x, y, color=st["color"], lw=2.5, ls=st.get("linestyle", "-"), marker="s", label=LABELS[key])
        ax_bot.plot(x, tab[f"pull_{key}"], color=st["color"], lw=2.5, marker="s")
    ax_bot.axhspan(-3, 3, color="0.9")
    ax_bot.axhline(0, color="k", lw=1)
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("pull")
    ax_top.set_ylabel("binary fraction" if frac else "parent count")
    if logx:
        ax_top.set_xscale("log")
        ax_bot.set_xscale("log")
    apply_axes_style(ax_top, style)
    apply_axes_style(ax_bot, style)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="small")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/gate418"))
    ap.add_argument("--host-profile", default="laptop")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pc = load_config(host_profile=args.host_profile)
    t0 = time.time()
    size = args.size if args.size in ("small", "large") else int(args.size)
    res, raw = cl.run_cmd_closed_loop(size, pipeline_config=pc)
    wall = time.time() - t0
    out = {
        "size": res.size, "wall_seconds": wall, "counts": res.counts, "unit_weight_rows": res.unit_counts,
        "m1_hat_vs_true": res.m1_hat_vs_true, "total": res.total, "ess": res.ess, "tables": res.tables,
        "max_abs_pull": {k: cl.max_abs_pull(res, k) for k in ("none", "one_d", "two_d")},
    }
    tag = f"closed_loop_cmd_{res.size}"
    (args.out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    style = resolve_plotting_style(pc.plotting)
    plt = require_pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    for j, (name, xl, logx) in enumerate((("binary_by_g", "G (mag)", False), ("binary_by_m1_hat", r"isochrone $\hat M_1$ (M$_\odot$)", False),
                                          ("binary_by_colour0", r"(BP$-$RP)$_0$ (mag)", False))):
        _panel(axes[0, j], axes[1, j], res.tables[name], xl, style, frac=True, logx=logx)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"#418 closed loop ({res.size}: {int(res.total['n_rows'])} parent rows): parent binary fraction")
    save_figure(fig, args.out_dir / f"{tag}_binary_fraction.png", dpi=int(pc.diagnostics.figure_dpi))
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    for j, (name, xl, logx) in enumerate((("log10_f", r"log$_{10}$ f", False), ("q", r"q = M$_2$/M$_1$", False),
                                          ("m2", r"M$_2$ (M$_\odot$)", True), ("alpha0", r"$\alpha_0$ (mas)", True))):
        _panel(axes[0, j], axes[1, j], res.tables[name], xl, style, frac=False, logx=logx)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"#418 closed loop ({res.size}): parent companion counts")
    save_figure(fig, args.out_dir / f"{tag}_companions.png", dpi=int(pc.diagnostics.figure_dpi))
    plt.close(fig)
    print(json.dumps({k: out[k] for k in ("size", "wall_seconds", "counts", "unit_weight_rows", "m1_hat_vs_true", "total", "ess", "max_abs_pull")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
