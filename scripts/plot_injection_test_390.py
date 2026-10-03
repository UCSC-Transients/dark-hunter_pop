#!/usr/bin/env python3
"""Figures + summary statistics for the Step 1a injection test (#390).

Reads ``injection_test_<tag>.h5`` written by ``scripts/run_injection_test_390.py
assemble`` and writes, into ``--fig-dir``:

* ``pulls.png`` — (recovered - input) / sigma for P, e, a0, parallax, cos i with an
  N(0, 1) overlay (cos i uses the published full-covariance sigma, see below);
* ``sigma_recovered_vs_published.png`` — recovered vs published sigma, log-log, 1:1;
* ``significance_ruwe_f2.png`` — a0/sigma_a0, RUWE and F2 recovered vs published;
* ``acceptance.png`` — acceptance vs published a0/sigma_a0, G, P and |ecliptic latitude|;
* ``pulls_by_g.png`` / ``pulls_by_nvis.png`` — the pulls split by G and by the
  published number of visibility periods;
* ``nvis_and_outcomes.png`` — gaiamock vs published visibility periods, and the
  cascade outcome fractions;
* ``summary.json`` — every number quoted in the report.

Pull conventions
----------------
``pull_x = (x_recovered - x_published) / sigma_x,recovered`` for P, e, a0 and
parallax, over realizations that reached the orbital fit **and** passed all cuts
(``accepted``); ``gaiamock_mod`` returns no covariance for the 12-parameter fit, so
``cos i`` is normalised by the published sigma propagated from the full A/B/F/G
covariance. Robust width = 1.4826 x MAD; "outlier" = |pull| >= ``--clip``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import injection_test as it  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402
from darkhunter_pop.plotting import (  # noqa: E402
    apply_axes_style,
    legend_prop,
    reference_line_style,
    require_pyplot,
    save_figure,
    series_style,
)

TYPES = ("Orbital", "AstroSpectroSB1")
TYPE_LABEL = {"Orbital": "Orbital", "AstroSpectroSB1": "AstroSpectroSB1"}
PULL_PARAMS = (
    ("period", "period_error", r"$P$"),
    ("eccentricity", "eccentricity_error", r"$e$"),
    ("a0_mas", "sigma_a0_mas", r"$a_0$"),
    ("parallax", "parallax_error", r"$\varpi$"),
    ("cos_i", None, r"$\cos i$"),
)
SIGMA_PARAMS = (
    ("period_error", "period_error", r"$\sigma_P$ (day)"),
    ("eccentricity_error", "eccentricity_error", r"$\sigma_e$"),
    ("sigma_a0_mas", "sigma_a0_mas", r"$\sigma_{a_0}$ (mas)"),
    ("parallax_error", "parallax_error", r"$\sigma_\varpi$ (mas)"),
)


def load(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h:
        sysg = h["systems"]
        d: dict[str, Any] = {
            "attrs": dict(h.attrs),
            "sys_type": np.array([s.decode() for s in sysg["nss_solution_type"][()]]),
            "sys_weight": sysg["stratum_weight"][()],
            "truth": {k: sysg["truth"][k][()] for k in sysg["truth"]},
            "sys_source_id": sysg["source_id"][()],
        }
        rg = h["realizations"]
        d["si"] = rg["system_index"][()]
        d["outcome"] = rg["outcome"][()]
        d["accepted"] = rg["accepted"][()]
        d["error"] = np.array([e.decode() for e in rg["error"][()]])
        d["rec"] = {k: rg["recovered"][k][()] for k in rg["recovered"]}
        d["cuts"] = {k: rg["cuts"][k][()] for k in rg["cuts"]}
        d["cpu_s"] = rg["cpu_s"][()]
        d["wall_s"] = rg["wall_s"][()]
        d["worker_rss_gib"] = rg["worker_rss_gib"][()]
    return d


def pulls(d: dict[str, Any], name: str, sig_name: str | None) -> np.ndarray:
    truth = d["truth"][name][d["si"]]
    rec = d["rec"][name]
    if sig_name is None:
        sig = d["truth"]["sigma_cos_i"][d["si"]]
    else:
        sig = d["rec"][sig_name]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rec - truth) / sig


def pulls_vs_published_sigma(d: dict[str, Any], name: str, sig_name: str | None) -> np.ndarray:
    truth = d["truth"][name][d["si"]]
    rec = d["rec"][name]
    sig = d["truth"]["sigma_cos_i" if sig_name is None else sig_name][d["si"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rec - truth) / sig


def _summary_dict(s: it.PullSummary) -> dict[str, float]:
    return {
        "n": s.n,
        "median": s.median,
        "sigma_mad": s.sigma_mad,
        "mean_clipped": s.mean,
        "std_clipped": s.std,
        "frac_outlier": s.frac_outlier,
    }


def _norm_overlay(ax: Any, n: int, width: float, style: Any) -> None:
    lim = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (-6.0, 6.0)
    x = np.linspace(min(lim[0], -6.0), max(lim[1], 6.0), 400)
    ax.plot(
        x,
        n * width * np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi),
        label=r"$\mathcal{N}(0,1)$",
        **{k: v for k, v in reference_line_style(0, style).items()},
    )


def fig_pulls(d: dict[str, Any], sel: np.ndarray, path: Path, style: Any, dpi: int, clip: float) -> None:
    """One panel per parameter; one step histogram per solution type; N(0,1) scaled to each."""
    plt = require_pyplot()
    edges = np.linspace(-clip, clip, 31)
    width = edges[1] - edges[0]
    groups = [(TYPE_LABEL[t], sel & (d["sys_type"][d["si"]] == t)) for t in TYPES]
    fig, all_axes = plt.subplots(
        1, len(PULL_PARAMS) + 1, figsize=(4.2 * len(PULL_PARAMS) + 2.6, 5.0),
        gridspec_kw={"width_ratios": [1.0] * len(PULL_PARAMS) + [0.6]},
    )
    axes, legend_ax = all_axes[:-1], all_axes[-1]
    legend_ax.axis("off")
    for ax, (name, sig, lab) in zip(axes, PULL_PARAMS):
        notes = []
        for gi, (glab, m) in enumerate(groups):
            p = pulls(d, name, sig)[m]
            p = p[np.isfinite(p)]
            if len(p) == 0:
                continue
            st = series_style(gi + 1, style)
            inside = p[np.abs(p) < clip]
            ax.hist(inside, bins=edges, histtype="step", color=st["color"],
                    linestyle=st["linestyle"], linewidth=st["linewidth"], label=glab)
            x = np.linspace(-clip, clip, 300)
            ax.plot(x, len(p) * width * np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi),
                    color=st["color"], linestyle=":", linewidth=1.5,
                    label=r"$\mathcal{N}(0,1)$" if gi == 0 else None)
            notes.append(f"{glab}: N={len(p)}, out={100*np.mean(np.abs(p) >= clip):.0f}%")
        ax.set_xlim(-clip, clip)
        ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", ha="left",
                fontfamily=style.font_family, fontsize=style.tick_label_fontsize * 0.8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        apply_axes_style(ax, style, xlabel=f"pull in {lab}",
                         ylabel="realizations" if ax is axes[0] else None)
    for ax in axes:
        ax.set_xticks([-4, -2, 0, 2, 4])
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, prop=legend_prop(style), loc="center left", frameon=False)
    save_figure(fig, path, dpi=dpi)


def fig_pulls_split(d: dict[str, Any], sel: np.ndarray, path: Path, style: Any, dpi: int, clip: float,
                    key: np.ndarray, edges: np.ndarray, key_label: str, fmt: str) -> list[dict[str, Any]]:
    """Grid: rows = bins of ``key`` (per realization), columns = pull parameters."""
    plt = require_pyplot()
    nb = len(edges) - 1
    hedges = np.linspace(-clip, clip, 25)
    width = hedges[1] - hedges[0]
    fig, axes = plt.subplots(nb, len(PULL_PARAMS), figsize=(3.6 * len(PULL_PARAMS), 3.0 * nb), squeeze=False)
    rows_out = []
    for bi in range(nb):
        m = sel & (key >= edges[bi]) & (key < edges[bi + 1] if bi < nb - 1 else key <= edges[bi + 1])
        rng_lab = f"{key_label} {fmt.format(edges[bi])}–{fmt.format(edges[bi+1])}"
        for pj, (name, sig, lab) in enumerate(PULL_PARAMS):
            ax = axes[bi, pj]
            p = pulls(d, name, sig)[m]
            p = p[np.isfinite(p)]
            s = it.summarize_pulls(p, clip=clip)
            rows_out.append({"bin": rng_lab, "param": name, **_summary_dict(s)})
            st = series_style(5, style)
            if len(p):
                ax.hist(np.clip(p, -clip + 1e-9, clip - 1e-9), bins=hedges, histtype="stepfilled",
                        color=st["color"], alpha=0.35, edgecolor=st["color"], linewidth=st["linewidth"])
                _norm_overlay(ax, len(p), width, style)
            ax.set_xlim(-clip, clip)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.text(0.03, 0.95, f"N={s.n}\nmed={s.median:+.2f}\n$\\sigma_{{\\rm MAD}}$={s.sigma_mad:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontfamily=style.font_family,
                    fontsize=style.tick_label_fontsize * 0.85)
            apply_axes_style(
                ax, style,
                xlabel=f"pull in {lab}" if bi == nb - 1 else None,
                ylabel=rng_lab if pj == 0 else None,
            )
            ax.yaxis.label.set_fontsize(style.tick_label_fontsize)
    save_figure(fig, path, dpi=dpi)
    return rows_out


def fig_sigma(d: dict[str, Any], sel: np.ndarray, path: Path, style: Any, dpi: int) -> dict[str, Any]:
    plt = require_pyplot()
    fig, axes = plt.subplots(1, len(SIGMA_PARAMS), figsize=(4.4 * len(SIGMA_PARAMS), 4.6))
    out: dict[str, Any] = {}
    for ax, (rname, tname, lab) in zip(axes, SIGMA_PARAMS):
        lo, hi = np.inf, -np.inf
        for ti, t in enumerate(TYPES):
            m = sel & (d["sys_type"][d["si"]] == t)
            x = d["truth"][tname][d["si"]][m]
            y = d["rec"][rname][m]
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            st = series_style(ti + 1, style)
            ax.scatter(x[ok], y[ok], s=8, marker=st["marker"], color=st["color"], alpha=0.5,
                       label=f"{TYPE_LABEL[t]}", rasterized=True)
            if ok.any():
                lo, hi = min(lo, x[ok].min(), y[ok].min()), max(hi, x[ok].max(), y[ok].max())
                r = y[ok] / x[ok]
                out[f"{rname}:{t}"] = {
                    "n": int(ok.sum()),
                    "median_ratio": float(np.median(r)),
                    "p16_ratio": float(np.percentile(r, 16)),
                    "p84_ratio": float(np.percentile(r, 84)),
                }
        if np.isfinite(lo):
            ax.plot([lo, hi], [lo, hi], **reference_line_style(0, style), label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel=f"published {lab}", ylabel=f"recovered {lab}")
    axes[0].legend(prop=legend_prop(style), loc="upper left", frameon=False)
    save_figure(fig, path, dpi=dpi)
    return out


def fig_sig_ruwe_f2(d: dict[str, Any], sel_orb: np.ndarray, path: Path, style: Any, dpi: int) -> dict[str, Any]:
    plt = require_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    out: dict[str, Any] = {}
    specs = (
        ("significance", "significance", r"$a_0/\sigma_{a_0}$", "log", sel_orb),
        ("ruwe", "ruwe", "RUWE", "log", np.isfinite(d["rec"]["ruwe"])),
        ("goodness_of_fit", "goodness_of_fit", r"goodness of fit $F_2$", "linear", sel_orb),
    )
    for ax, (rn, tn, lab, scale, sel) in zip(axes, specs):
        lo, hi = np.inf, -np.inf
        for ti, t in enumerate(TYPES):
            m = sel & (d["sys_type"][d["si"]] == t)
            x = d["truth"][tn][d["si"]][m]
            y = d["rec"][rn][m]
            ok = np.isfinite(x) & np.isfinite(y)
            if scale == "log":
                ok &= (x > 0) & (y > 0)
            st = series_style(ti + 1, style)
            ax.scatter(x[ok], y[ok], s=8, marker=st["marker"], color=st["color"], alpha=0.5,
                       label=TYPE_LABEL[t], rasterized=True)
            if ok.any():
                lo, hi = min(lo, x[ok].min(), y[ok].min()), max(hi, x[ok].max(), y[ok].max())
                diff = (y[ok] / x[ok]) if scale == "log" else (y[ok] - x[ok])
                out[f"{rn}:{t}"] = {
                    "n": int(ok.sum()),
                    ("median_ratio" if scale == "log" else "median_difference"): float(np.median(diff)),
                    "p16": float(np.percentile(diff, 16)),
                    "p84": float(np.percentile(diff, 84)),
                }
        if rn == "goodness_of_fit":
            lo, hi = max(lo, -10.0), min(hi, 60.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        if np.isfinite(lo):
            ax.plot([lo, hi], [lo, hi], **reference_line_style(0, style), label="1:1")
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        if rn == "ruwe":
            from matplotlib.ticker import FuncFormatter

            fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
            minor_fmt = FuncFormatter(
                lambda v, _pos: f"{v:g}" if f"{v:g}"[0] in "25" else ""
            )
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(fmt)
                axis.set_minor_formatter(minor_fmt)
            ax.tick_params(axis="both", which="minor", labelsize=style.tick_label_fontsize * 0.8)
        apply_axes_style(ax, style, xlabel=f"published {lab}", ylabel=f"recovered {lab}")
    axes[0].legend(prop=legend_prop(style), loc="upper left", frameon=False)
    save_figure(fig, path, dpi=dpi)
    return out


def _acc_curve(x: np.ndarray, acc: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    c, f, lo, hi, n = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        k, nn = int(acc[m].sum()), int(m.sum())
        p, l_, h_ = it.binomial_fraction_interval(k, nn)
        c.append(0.5 * (a + b))
        f.append(p)
        lo.append(l_)
        hi.append(h_)
        n.append(nn)
    return tuple(np.asarray(v, dtype=float) for v in (c, f, lo, hi, n))


def fig_acceptance(d: dict[str, Any], path: Path, style: Any, dpi: int, cuts: Any) -> dict[str, Any]:
    plt = require_pyplot()
    tr = d["truth"]
    x_all = {
        "log10_significance": np.log10(tr["significance"])[d["si"]],
        "g_mag": tr["g_mag"][d["si"]],
        "log10_period": np.log10(tr["period"])[d["si"]],
        "abs_ecl_lat": np.abs(tr["published_ecl_lat"])[d["si"]],
    }
    labels = {
        "log10_significance": r"published $\log_{10}(a_0/\sigma_{a_0})$",
        "g_mag": r"$G$ (mag)",
        "log10_period": r"published $\log_{10}(P_{\rm orb}/{\rm day})$",
        "abs_ecl_lat": r"$|\beta|$ (deg)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8))
    out: dict[str, Any] = {}
    for ax, (k, x) in zip(axes, x_all.items()):
        finite = np.isfinite(x)
        edges = np.quantile(x[finite], np.linspace(0, 1, 9)) if finite.any() else np.array([0, 1])
        edges[-1] += 1e-9
        for ti, t in enumerate(TYPES):
            m = finite & (d["sys_type"][d["si"]] == t)
            c, f, lo, hi, n = _acc_curve(x[m], d["accepted"][m], edges)
            st = series_style(ti + 1, style)
            ax.errorbar(c, f, yerr=[np.clip(f - lo, 0, None), np.clip(hi - f, 0, None)], color=st["color"], marker=st["marker"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"], markersize=st["markersize"],
                        capsize=3, label=TYPE_LABEL[t])
            out[f"{k}:{t}"] = {"centers": c.tolist(), "fraction": f.tolist(), "n": n.tolist()}
        if k == "log10_significance":
            ax.axvline(math.log10(cuts.a0_over_err_min), **reference_line_style(1, style),
                       label=rf"$a_0/\sigma_{{a_0}}={cuts.a0_over_err_min:g}$")
        ax.set_ylim(-0.03, 1.05)
        apply_axes_style(ax, style, xlabel=labels[k], ylabel="acceptance fraction" if ax is axes[0] else None)
    axes[0].legend(prop=legend_prop(style), loc="lower right", frameon=False)
    save_figure(fig, path, dpi=dpi)
    return out


def fig_nvis_outcomes(d: dict[str, Any], path: Path, style: Any, dpi: int) -> dict[str, Any]:
    plt = require_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    x = d["truth"]["published_visibility_periods_used"][d["si"]]
    y = d["rec"]["n_visibility_periods"]
    ok = np.isfinite(x) & np.isfinite(y)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.25, 0.25, (2, ok.sum()))
    st = series_style(1, style)
    ax.scatter(x[ok] + jit[0], y[ok] + jit[1], s=6, color=st["color"], alpha=0.4, rasterized=True,
               label="realizations (jittered)")
    lo, hi = np.nanmin(np.r_[x[ok], y[ok]]), np.nanmax(np.r_[x[ok], y[ok]])
    ax.plot([lo, hi], [lo, hi], **reference_line_style(0, style), label="1:1")
    apply_axes_style(ax, style, xlabel="published visibility periods used",
                     ylabel="gaiamock visibility periods")
    ax.legend(prop=legend_prop(style), loc="upper left", frameon=False)
    out: dict[str, Any] = {
        "nvis_median_difference": float(np.median(y[ok] - x[ok])) if ok.any() else math.nan,
        "nvis_p16_difference": float(np.percentile(y[ok] - x[ok], 16)) if ok.any() else math.nan,
        "nvis_p84_difference": float(np.percentile(y[ok] - x[ok], 84)) if ok.any() else math.nan,
    }
    ax = axes[1]
    codes = [
        (it.OUTCOME_INSUFFICIENT_VISIBILITY, "< 12 vis. periods"),
        (it.OUTCOME_FIVE_PARAMETER, "5-par (RUWE < 1.4)"),
        (it.OUTCOME_SEVEN_PARAMETER, "7-par accel."),
        (it.OUTCOME_NINE_PARAMETER, "9-par accel."),
        (-1, "orbit, failed cuts"),
        (99, "orbit, accepted"),
    ]
    width = 0.38
    for ti, t in enumerate(TYPES):
        m = d["sys_type"][d["si"]] == t
        n = max(int(m.sum()), 1)
        fr = []
        for code, _ in codes:
            if code == -1:
                k = np.sum(m & (d["outcome"] == it.OUTCOME_ORBITAL) & ~d["accepted"])
            elif code == 99:
                k = np.sum(m & d["accepted"])
            else:
                k = np.sum(m & (d["outcome"] == code))
            fr.append(k / n)
        out[f"outcomes:{t}"] = dict(zip([c[1] for c in codes], [float(v) for v in fr]))
        st = series_style(ti + 1, style)
        ypos = np.arange(len(codes)) + (ti - 0.5) * width
        bars = ax.barh(ypos, fr, height=width, color=st["color"], hatch=["", "//"][ti],
                       edgecolor="black", label=TYPE_LABEL[t])
        for b_, v in zip(bars, fr):
            ax.text(v + 0.01, b_.get_y() + b_.get_height() / 2, f"{v:.3f}", va="center",
                    fontfamily=style.font_family, fontsize=style.tick_label_fontsize * 0.85)
    ax.set_yticks(np.arange(len(codes)))
    ax.set_yticklabels([c[1] for c in codes])
    ax.set_xlim(0, 1.15)
    apply_axes_style(ax, style, xlabel="fraction of realizations", enable_minor_ticks=False)
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.legend(prop=legend_prop(style), loc="lower right", frameon=False)
    save_figure(fig, path, dpi=dpi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", type=Path)
    ap.add_argument("--fig-dir", type=Path, default=REPO / "docs" / "gate390" / "figures")
    ap.add_argument("--clip", type=float, default=5.0, help="|pull| outlier threshold / histogram range")
    args = ap.parse_args()
    cfg = load_config()
    style = cfg.plotting
    dpi = int(cfg.diagnostics.figure_dpi)
    cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
    d = load(args.h5)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    acc = d["accepted"].astype(bool)
    orb = d["outcome"] == it.OUTCOME_ORBITAL
    ok_err = d["error"] == ""
    summary: dict[str, Any] = {
        "h5": str(args.h5),
        "n_systems": int(len(d["sys_type"])),
        "n_realizations": int(len(d["si"])),
        "n_errors": int((~ok_err).sum()),
        "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in d["attrs"].items()},
    }
    # Cost.
    summary["cost"] = {
        "cpu_s_mean_per_realization": float(np.nanmean(d["cpu_s"])),
        "cpu_s_median_per_realization": float(np.nanmedian(d["cpu_s"])),
        "cpu_s_mean_orbital_branch": float(np.nanmean(d["cpu_s"][orb])) if orb.any() else math.nan,
        "cpu_s_mean_non_orbital": float(np.nanmean(d["cpu_s"][~orb])) if (~orb).any() else math.nan,
        "wall_s_mean_per_realization": float(np.nanmean(d["wall_s"])),
        "worker_peak_rss_gib": float(np.nanmax(d["worker_rss_gib"])),
    }
    # Acceptance (raw and stratum-weighted).
    w = d["sys_weight"][d["si"]]
    acc_tab = {}
    for t in TYPES:
        m = d["sys_type"][d["si"]] == t
        k, n = int(acc[m].sum()), int(m.sum())
        p, lo, hi = it.binomial_fraction_interval(k, n)
        acc_tab[t] = {
            "n": n, "accepted": k, "fraction": p, "lo68": lo, "hi68": hi,
            "weighted_fraction": float(np.sum(w[m] * acc[m]) / np.sum(w[m])) if n else math.nan,
            "orbital_branch_fraction": float(np.mean(orb[m])) if n else math.nan,
            "cut_pass_given_orbit": {
                c: float(np.mean(d["cuts"][c][m & orb])) if (m & orb).any() else math.nan
                for c in it.CUT_FLAG_NAMES
            },
        }
    summary["acceptance"] = acc_tab
    # Pulls.
    ptab = {}
    for t in TYPES:
        m = acc & (d["sys_type"][d["si"]] == t)
        for name, sig, _ in PULL_PARAMS:
            ptab[f"{name}:{t}"] = _summary_dict(it.summarize_pulls(pulls(d, name, sig)[m], clip=args.clip))
            ptab[f"{name}:{t}:published_sigma"] = _summary_dict(
                it.summarize_pulls(pulls_vs_published_sigma(d, name, sig)[m], clip=args.clip)
            )
    summary["pulls_accepted"] = ptab

    fig_pulls(d, acc, args.fig_dir / "pulls.png", style, dpi, args.clip)
    summary["sigma_ratio"] = fig_sigma(d, acc, args.fig_dir / "sigma_recovered_vs_published.png", style, dpi)
    summary["significance_ruwe_f2"] = fig_sig_ruwe_f2(d, orb, args.fig_dir / "significance_ruwe_f2.png", style, dpi)
    summary["acceptance_curves"] = fig_acceptance(d, args.fig_dir / "acceptance.png", style, dpi, cuts)
    g_key = d["truth"]["g_mag"][d["si"]]
    g_edges = np.quantile(d["truth"]["g_mag"], [0, 1 / 3, 2 / 3, 1])
    summary["pulls_by_g"] = fig_pulls_split(
        d, acc, args.fig_dir / "pulls_by_g.png", style, dpi, args.clip, g_key, g_edges, "G", "{:.1f}"
    )
    nv = d["truth"]["published_visibility_periods_used"]
    if np.isfinite(nv).any():
        nv_key = nv[d["si"]]
        nv_edges = np.unique(np.quantile(nv[np.isfinite(nv)], [0, 1 / 3, 2 / 3, 1]))
        summary["pulls_by_nvis"] = fig_pulls_split(
            d, acc, args.fig_dir / "pulls_by_nvis.png", style, dpi, args.clip, nv_key, nv_edges,
            r"$N_{\rm vis}$", "{:.0f}"
        )
    summary["nvis_outcomes"] = fig_nvis_outcomes(d, args.fig_dir / "nvis_and_outcomes.png", style, dpi)
    summary["thiele_innes_roundtrip_max_abs_mas"] = float(np.nanmax(d["truth"]["thiele_innes_roundtrip_max_abs_mas"]))
    summary["n_outside_fit_bounds"] = int(np.sum(d["truth"]["outside_fit_bounds"]))
    summary["n_sigma_cos_i_nan"] = int(np.sum(~np.isfinite(d["truth"]["sigma_cos_i"])))
    sa = d["truth"]["sigma_a0_mas"]
    sc = d["truth"]["sigma_a0_cov_mas"]
    okc = np.isfinite(sa) & np.isfinite(sc) & (sa > 0)
    summary["published_sigma_a0_cov_over_a0_over_significance_median"] = (
        float(np.median(sc[okc] / sa[okc])) if okc.any() else math.nan
    )
    (args.fig_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("n_systems", "n_realizations", "n_errors", "cost", "acceptance")}, indent=1, default=float))


if __name__ == "__main__":
    main()
