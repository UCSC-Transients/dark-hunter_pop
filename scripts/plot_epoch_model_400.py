#!/usr/bin/env python3
"""Figures and summary numbers for the #400 epoch-model report (``docs/gate400``).

``measurement``
    From ``output/gate400/{gost_pixels.npz, epoch_compare.h5, decomposition.json}``
    (``scripts/measure_epoch_counts_400.py``):

    * ``transit_ratio.png``: DR3 ``astrometric_matched_transits`` / GOST transits per
      star, raw GOST vs GOST after the published gaps, random stars and NSS;
    * ``keep_vs_g.png``: the residual per-transit keep fraction after the gaps vs G,
      with the calibrated step function;
    * ``keep_vs_covariates.png``: the same against local source density, |b|, |beta| and
      the scan-direction strength k1;
    * ``model_counts_vs_dr3.png``: transit count and visibility periods per star from the
      statistical model (gaps + calibrated loss, drawn per star, no per-star lookup)
      against DR3, random stars and NSS;
    * ``gap_fraction_sky.png``: fraction of each GOST position's transits inside the gaps.

``validation``
    From ``output/gate400/validation/realizations.jsonl`` and the #390 artifact:
    ``validation_sigma_ratio.png``, ``validation_counts.png``,
    ``validation_capture_vs_period.png``; numbers into ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import epoch_model as em  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402
from darkhunter_pop.plotting import (  # noqa: E402
    apply_axes_style,
    legend_prop,
    require_pyplot,
    save_figure,
    series_style,
)


def _style() -> tuple[Any, int]:
    cfg = load_config()
    return cfg.plotting, int(cfg.diagnostics.figure_dpi)


def _hist_step(ax: Any, x: np.ndarray, bins: np.ndarray, i: int, style: Any, label: str) -> None:
    st = series_style(i, style)
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    ax.step(e[:-1], h / max(h.sum(), 1), where="post", color=st["color"],
            linestyle=st["linestyle"], linewidth=st["linewidth"], label=label)


def _model_draws(
    g: np.lib.npyio.NpzFile, pix: np.ndarray, gmag: np.ndarray, cfg: em.EpochModelConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per star: transits and visibility periods from gaps + calibrated loss (one draw)."""
    off, tm = g["offsets"], g["t_mid"]
    gaps = em.gap_intervals_jd(cfg)
    keep_gap = ~em.in_gaps(tm, gaps)
    rng = np.random.default_rng(seed)
    ntr = np.zeros(pix.size, dtype=np.int64)
    nvis = np.zeros(pix.size, dtype=np.int64)
    for i, (p, gm) in enumerate(zip(pix, gmag)):
        t = tm[off[p]:off[p + 1]][keep_gap[off[p]:off[p + 1]]]
        pl = em.transit_loss_probability(float(gm), cfg)
        t = t[rng.uniform(size=t.size) >= pl]
        ntr[i] = t.size
        nvis[i] = em.n_visibility_periods_from_days(t, 4.0)
    return ntr, nvis


def cmd_measurement(args: argparse.Namespace) -> None:
    plt = require_pyplot()
    style, dpi = _style()
    out = Path(args.out)
    fig_dir = Path(args.fig_dir)
    dec = json.loads((out / "decomposition.json").read_text())
    g = np.load(out / "gost_pixels.npz")
    cfgp = load_config().dr3.epoch_model
    cfg = em.epoch_model_config_from_mapping(cfgp)
    summary: dict[str, Any] = {"decomposition_file": str(out / "decomposition.json")}
    with h5py.File(out / "epoch_compare.h5", "r") as f:
        data = {n: {k: f[n][k][:] for k in f[n]} for n in ("random", "nss", "inj390")}

    # 1) ratio histograms
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    bins = np.linspace(0.4, 1.6, 61)
    for ax, (name, lab) in zip(axes, (("random", "random gaia_source, G < 19"), ("nss", "NSS Orbital + AstroSpectroSB1"))):
        d = data[name]
        amt = d["dr3_astrometric_matched_transits"]
        _hist_step(ax, amt / d["gost_ntr_raw"], bins, 0, style, "GOST as gaiamock reads it")
        _hist_step(ax, amt / d["gost_ntr_esa_gaps"], bins, 1, style, "GOST minus published gaps")
        ax.axvline(1.0, color="0.4", linewidth=1.5, linestyle=":")
        apply_axes_style(ax, style, xlabel="DR3 transits / GOST transits", ylabel="fraction of stars" if name == "random" else None,
                         title=f"{lab} (N = {amt.size:,})")
        ax.legend(prop=legend_prop(style), loc="upper left", frameon=False)
    save_figure(fig, fig_dir / "transit_ratio.png", dpi=dpi)

    # 2) keep vs G
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, key in enumerate(("random/all", "nss/all", "inj390/all")):
        bins_ = [b for b in dec["samples"][key]["keep_vs_g"] if b.get("n", 0) >= 30]
        x = np.array([(b["lo"] + b["hi"]) / 2 for b in bins_])
        y = np.array([b["keep"] for b in bins_])
        lo = y - np.array([b["keep_p16"] for b in bins_])
        hi = np.array([b["keep_p84"] for b in bins_]) - y
        st = series_style(i, style)
        ax.errorbar(x + 0.08 * (i - 1), y, yerr=[lo, hi], color=st["color"], marker=st["marker"],
                    linestyle="none", markersize=st["markersize"], linewidth=2, capsize=3,
                    label={"random/all": "random gaia_source (calibration)", "nss/all": "NSS (validation)",
                           "inj390/all": "#390 injection set"}[key])
    edges = np.asarray(cfg.transit_loss_g_edges)
    keep = 1 - np.asarray(cfg.transit_loss_prob)
    ax.stairs(keep, edges, baseline=None, color="k", linewidth=2.5, label="calibrated model (config)")
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="DR3 transits / GOST transits after gaps",
                     title="Per-transit keep fraction after the published gaps")
    ax.set_ylim(0.85, 1.07)
    ax.legend(prop=legend_prop(style), loc="lower left", frameon=False)
    save_figure(fig, fig_dir / "keep_vs_g.png", dpi=dpi)

    # 3) covariates
    labels = {
        "log10_density_per_deg2": r"log$_{10}$ source density (deg$^{-2}$)",
        "abs_b_deg": r"$|b|$ (deg)", "abs_ecl_lat_deg": r"$|\beta|$ (deg)",
        "scan_direction_strength_k1": r"scan_direction_strength_k1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    for ax, (cov, lab) in zip(axes.ravel(), labels.items()):
        for i, key in enumerate(("random/all", "nss/all")):
            bins_ = [b for b in dec["samples"][key]["keep_vs"][cov] if b.get("n", 0) >= 30]
            x = np.array([(b["lo"] + b["hi"]) / 2 for b in bins_])
            y = np.array([b["keep"] for b in bins_])
            st = series_style(i, style)
            ax.errorbar(x, y, yerr=[y - np.array([b["keep_p16"] for b in bins_]),
                                    np.array([b["keep_p84"] for b in bins_]) - y],
                        color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                        markersize=st["markersize"], linewidth=2, capsize=3,
                        label="random gaia_source" if key == "random/all" else "NSS")
        apply_axes_style(ax, style, xlabel=lab, ylabel="keep fraction after gaps")
        ax.set_ylim(0.85, 1.15)
    axes[0, 0].legend(prop=legend_prop(style), loc="upper right", frameon=False)
    save_figure(fig, fig_dir / "keep_vs_covariates.png", dpi=dpi)

    # 4) model draws vs DR3
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, name in enumerate(("random", "nss")):
        d = data[name]
        ntr_m, nvis_m = _model_draws(g, d["pix"], d["phot_g_mean_mag"], cfg, seed=400 + col)
        dr3_tr = d["dr3_astrometric_matched_transits"]
        dr3_vp = d["dr3_visibility_periods_used"]
        b_tr = np.arange(0, 121, 3)
        _hist_step(axes[0, col], dr3_tr, b_tr, 0, style, "DR3 astrometric_matched_transits")
        _hist_step(axes[0, col], d["gost_ntr_raw"], b_tr, 1, style, "GOST (gaiamock today)")
        _hist_step(axes[0, col], ntr_m.astype(float), b_tr, 2, style, "epoch model (one draw)")
        b_v = np.arange(5, 41, 1)
        _hist_step(axes[1, col], dr3_vp, b_v, 0, style, "DR3 visibility_periods_used")
        _hist_step(axes[1, col], d["gost_nvis_raw"].astype(float), b_v, 1, style, "GOST (gaiamock today)")
        _hist_step(axes[1, col], nvis_m.astype(float), b_v, 2, style, "epoch model (one draw)")
        t = "random gaia_source" if name == "random" else "NSS Orbital + AstroSpectroSB1"
        apply_axes_style(axes[0, col], style, xlabel="FoV transits", ylabel="fraction of stars", title=t)
        apply_axes_style(axes[1, col], style, xlabel="visibility periods", ylabel="fraction of stars")
        summary[f"model_vs_dr3_{name}"] = {
            "transits_mean": {"dr3": float(dr3_tr.mean()), "gost": float(d["gost_ntr_raw"].mean()), "model": float(ntr_m.mean())},
            "transits_model_over_dr3_sum": float(ntr_m.sum() / dr3_tr.sum()),
            "nvis_mean": {"dr3": float(dr3_vp.mean()), "gost": float(d["gost_nvis_raw"].mean()), "model": float(nvis_m.mean())},
            "nvis_model_minus_dr3_p16_50_84": [float(x) for x in np.percentile(nvis_m - dr3_vp, [16, 50, 84])],
            "nvis_gost_minus_dr3_p16_50_84": [float(x) for x in np.percentile(d["gost_nvis_raw"] - dr3_vp, [16, 50, 84])],
            "frac_below_12_vis": {"dr3": float(np.mean(dr3_vp < 12)), "gost": float(np.mean(d["gost_nvis_raw"] < 12)),
                                  "model": float(np.mean(nvis_m < 12))},
        }
    axes[0, 1].legend(prop=legend_prop(style), loc="upper right", frameon=False)
    axes[1, 1].legend(prop=legend_prop(style), loc="upper left", frameon=False)
    save_figure(fig, fig_dir / "model_counts_vs_dr3.png", dpi=dpi)

    # 5) sky map of gap fraction per GOST position
    off, tm = g["offsets"], g["t_mid"]
    ingap = em.in_gaps(tm, em.gap_intervals_jd(cfg))
    frac = np.add.reduceat(ingap.astype(float), off[:-1]) / np.diff(off)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ra = np.radians(np.where(g["ra"] > 180, g["ra"] - 360, g["ra"]))
    sc = ax.scatter(-ra, np.radians(g["dec"]), c=frac, s=14, cmap="viridis", vmin=0, vmax=0.25)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05)
    cb.set_label("fraction of GOST transits inside published DR3 astrometric gaps", fontfamily="serif", fontsize=14)
    ax.set_title("Gaps by sky position (equatorial, RA increasing left)", fontfamily="serif", fontsize=16)
    ax.grid(True, linewidth=0.5)
    ax.set_xticklabels([])
    save_figure(fig, fig_dir / "gap_fraction_sky.png", dpi=dpi)
    summary["gap_fraction_per_position_p5_50_95"] = [float(x) for x in np.percentile(frac, [5, 50, 95])]
    path = fig_dir / "summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["measurement"] = summary
    path.write_text(json.dumps(old, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("measurement")
    p.add_argument("--out", required=True)
    p.add_argument("--fig-dir", default=str(REPO / "docs" / "gate400" / "figures"))
    p.set_defaults(func=cmd_measurement)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
