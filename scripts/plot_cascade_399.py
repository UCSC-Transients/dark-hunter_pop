#!/usr/bin/env python3
"""Figures + summary numbers for #399 (acceleration capture) and #398 (sigma deficit).

Inputs (written by ``scripts/diagnose_cascade_399.py``, gitignored under
``output/gate399/``) plus the #390 artifact ``output/gate390/injection_test_full.h5``
and its ``gaia_source_visibility.json``. Writes into ``--fig-dir``:

* ``capture_vs_period.png`` - accepted-acceleration fraction vs published P under each
  test (noise level, DR3-matched noise, truth drawn from the published covariance,
  higher S/N, FOV-correlated noise);
* ``acceleration_statistics.png`` - the 9-parameter s, F2 and parallax criterion for
  P > 600 d Orbital realizations, by cascade outcome;
* ``capture_vs_coverage.png`` - capture vs span/P, e and published significance;
* ``capture_probability.png`` - per-system capture probability (25 realizations);
* ``forced_orbit.png`` - the orbit fit the cascade skipped, for captured realizations;
* ``sigma_ratio_decomposition.png`` - #398: recovered/published sigma vs G, P,
  a0/sigma_a0 and |ecliptic latitude|, raw and after the N-epoch and c-factor terms;
* ``c_factor_vs_g.png`` - #398: c_DR3, c_mock and the epoch-count ratio vs G;
* ``summary.json`` - every number quoted in ``docs/gate399/README.md``.
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

from darkhunter_pop import cascade_replay as cr  # noqa: E402
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

PRIMARY = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
P_EDGES = np.array([0.0, 300.0, 600.0, 1000.0, np.inf])
P_LABELS = ["< 300", "300-600", "600-1000", "> 1000"]
G_EDGES = np.array([-np.inf, 11.0, 12.0, 13.0, 14.0, 15.0, np.inf])
LONG_P = 600.0


def _read(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as h:
        return {k: h[k][()] for k in h}, dict(h.attrs)


def _load_390(path: Path, vis_path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        truth = {k: f["systems/truth"][k][()] for k in f["systems/truth"]}
        typ = f["systems/nss_solution_type"][()].astype(str)
        sid = f["systems/source_id"][()]
        rec = {k: f["realizations/recovered"][k][()] for k in f["realizations/recovered"]}
        si = f["realizations/system_index"][()]
        out = f["realizations/outcome"][()]
        acc = f["realizations/accepted"][()]
    rows = json.loads(vis_path.read_text())["rows"]
    n_good = np.array([rows.get(str(int(s)), {}).get("astrometric_n_good_obs_al", np.nan) for s in sid], float)
    return dict(truth=truth, typ=typ, rec=rec, si=si, out=out, acc=acc, n_good=n_good)


def _capture(branch: np.ndarray) -> np.ndarray:
    return np.isin(branch, (cr.BRANCH_SEVEN_PARAMETER, cr.BRANCH_NINE_PARAMETER))


def _frac_by_p(cap: np.ndarray, period: np.ndarray, sel: np.ndarray) -> list[dict[str, float]]:
    out = []
    for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
        s = sel & (period >= lo) & (period < hi)
        k, n = int(cap[s].sum()), int(s.sum())
        f, flo, fhi = it.binomial_fraction_interval(k, n)
        out.append({"n": n, "frac": f, "lo": flo, "hi": fhi})
    return out


def _binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    c, m, lo, hi, n = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (x >= a) & (x < b) & np.isfinite(y)
        if s.sum() < 5:
            continue
        xx = x[s]
        c.append(float(np.median(xx)))
        m.append(float(np.median(y[s])))
        lo.append(float(np.percentile(y[s], 16)))
        hi.append(float(np.percentile(y[s], 84)))
        n.append(int(s.sum()))
    return tuple(np.array(v) for v in (c, m, lo, hi, n))


def _grid_with_legend_row(
    nrows: int, ncols: int, figsize: tuple[float, float], legend_height: float, *, sharey: bool = False
) -> tuple[Any, np.ndarray, Any]:
    """Figure with an ``nrows x ncols`` panel grid plus a full-width legend row below it.

    ``tight_layout`` (``plotting.save_figure``) keeps the legend row clear of the tick
    labels, which a figure-level legend placed with ``bbox_to_anchor`` does not.
    """
    plt = require_pyplot()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.0] * nrows + [legend_height])
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            share = axes[i, 0] if (sharey and j > 0) else None
            axes[i, j] = fig.add_subplot(gs[i, j], sharey=share)
    leg_ax = fig.add_subplot(gs[nrows, :])
    leg_ax.axis("off")
    return fig, axes, leg_ax


# ---------------------------------------------------------------------------- #399


def capture_tests(d: dict[str, Any], gate: Path) -> dict[str, Any]:
    """Accepted-acceleration fraction vs published P for every test, Orbital + SB1."""
    T, typ = d["truth"], d["typ"]
    lin, la = _read(gate / "linear_replay.h5")
    mat, _ = _read(gate / "linear_matched.h5")
    dr, _ = _read(gate / "truth_draws.h5")
    fov, fa = _read(gate / "fov_correlated.h5")
    variants = json.loads(la["variants_noise_err"])
    fracs = json.loads(fa["fractions"])
    tests: dict[str, dict[str, Any]] = {}
    for t in ("Orbital", "AstroSpectroSB1"):
        res: dict[str, Any] = {}
        st = lin["realization"] < 5
        sel = (typ[lin["system_index"]] == t) & st
        per = T["period"][lin["system_index"]]
        for v, (kn, ke) in enumerate(variants):
            res[f"noise x{kn:g}, errors x{ke:g}"] = _frac_by_p(_capture(lin[f"branch__{v}"]), per, sel)
        selm = (typ[mat["system_index"]] == t) & (mat["realization"] < 5)
        perm = T["period"][mat["system_index"]]
        res["DR3-matched (N epochs + excess noise)"] = _frac_by_p(_capture(mat["branch__1"]), perm, selm)
        res["N epochs only"] = _frac_by_p(_capture(mat["branch__2"]), perm, selm)
        seld = typ[dr["system_index"]] == t
        res["truth drawn from published covariance"] = _frac_by_p(
            _capture(dr["branch"]), T["period"][dr["system_index"]], seld)
        self_ = (typ[fov["system_index"]] == t) & (fov["realization"] < 5)
        for v, fr in enumerate(fracs):
            if fr > 0:
                res[f"FOV-correlated noise f = {fr:g}"] = _frac_by_p(
                    _capture(fov[f"branch__{v}"]), T["period"][fov["system_index"]], self_)
        tests[t] = res
    return {"by_period": tests, "variants": variants, "fov_fractions": fracs}


def fig_capture_vs_period(caps: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    keys = [
        ("noise x1, errors x1", "baseline (#390)"),
        ("noise x1.11, errors x1.11", r"noise $\times$1.11 (#398)"),
        ("DR3-matched (N epochs + excess noise)", "DR3-matched noise"),
        ("truth drawn from published covariance", "truth from published covariance"),
        ("noise x0.5, errors x0.5", r"noise $\times$0.5 (higher S/N)"),
        ("FOV-correlated noise f = 0.5", "FOV-correlated noise, f = 0.5"),
    ]
    fig, ax2, leg_ax = _grid_with_legend_row(1, 2, (13.0, 6.2), 0.25, sharey=True)
    axes = ax2[0]
    x = np.arange(len(P_LABELS))
    for ax, t in zip(axes, ("Orbital", "AstroSpectroSB1")):
        for j, (k, lab) in enumerate(keys):
            rows = caps["by_period"][t][k]
            st = series_style(j, style)
            f = np.array([r["frac"] for r in rows])
            lo = np.array([r["lo"] for r in rows])
            hi = np.array([r["hi"] for r in rows])
            off = (j - (len(keys) - 1) / 2) * 0.07
            ax.errorbar(x + off, f, yerr=[f - lo, hi - f], color=st["color"], marker=st["marker"],
                        linestyle=st["linestyle"], linewidth=st["linewidth"], markersize=st["markersize"],
                        capsize=3, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(P_LABELS)
        apply_axes_style(ax, style, xlabel="published period (d)",
                         ylabel="accepted 7/9-parameter fraction" if t == "Orbital" else None,
                         title=t, enable_minor_ticks=False)
        ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator())
    h, lab = axes[0].get_legend_handles_labels()
    leg_ax.legend(h, lab, prop=legend_prop(style), loc="center", ncol=3, frameon=False)
    save_figure(fig, path, dpi=dpi)


def acceleration_breakdown(d: dict[str, Any], gate: Path) -> dict[str, Any]:
    """Which condition decided each long-period Orbital realization (baseline)."""
    T, typ = d["truth"], d["typ"]
    lin, _ = _read(gate / "linear_replay.h5")
    si = lin["system_index"]
    sel = (typ[si] == "Orbital") & (T["period"][si] >= LONG_P) & (lin["realization"] < 5)
    b = lin["branch__0"]
    s9, f9, p9 = lin["s9__0"], lin["f2_9__0"], lin["plx_snr9__0"]
    s7, f7, p7 = lin["s7__0"], lin["f2_7__0"], lin["plx_snr7__0"]
    r9 = p9 / (cr.GAIAMOCK_ACCEL9_PLX_COEFF * s9**cr.GAIAMOCK_ACCEL_PLX_EXPONENT)
    r7 = p7 / (cr.GAIAMOCK_ACCEL7_PLX_COEFF * s7**cr.GAIAMOCK_ACCEL_PLX_EXPONENT)
    orb = sel & (b == cr.BRANCH_ORBITAL)
    c9 = sel & (b == cr.BRANCH_NINE_PARAMETER)
    c7 = sel & (b == cr.BRANCH_SEVEN_PARAMETER)

    def pct(a: np.ndarray, m: np.ndarray) -> list[float]:
        return [float(v) for v in np.percentile(a[m], [16, 50, 84])]

    ok9s, ok9f, ok9p = s9 > cr.GAIAMOCK_ACCEL_SIGNIFICANCE_MIN, f9 < cr.GAIAMOCK_ACCEL_F2_MAX, r9 > 1
    ok7s, ok7f, ok7p = s7 > cr.GAIAMOCK_ACCEL_SIGNIFICANCE_MIN, f7 < cr.GAIAMOCK_ACCEL_F2_MAX, r7 > 1
    span = lin["phase_span"]
    ecc = T["eccentricity"][si]
    sig = T["significance"][si] / np.maximum(5.0, 158.0 / np.sqrt(T["period"][si]))
    cap = _capture(b)

    def frac_bins(x: np.ndarray, edges: list[float]) -> list[dict[str, float]]:
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (x >= lo) & (x < hi)
            f, flo, fhi = it.binomial_fraction_interval(int(cap[m].sum()), int(m.sum()))
            out.append({"lo_edge": lo, "hi_edge": hi, "n": int(m.sum()), "frac": f, "lo": flo, "hi": fhi})
        return out

    return {
        "n_long_period_orbital_realizations": int(sel.sum()),
        "captured_9par": int(c9.sum()),
        "captured_7par": int(c7.sum()),
        "captured_fraction": float(cap[sel].mean()),
        "nine_par_share_of_captures": float(c9.sum() / max(c9.sum() + c7.sum(), 1)),
        "captured9_s9_16_50_84": pct(s9, c9),
        "captured9_f2_9_16_50_84": pct(f9, c9),
        "captured9_plx_ratio9_16_50_84": pct(r9, c9),
        "captured7_s7_16_50_84": pct(s7, c7),
        "captured7_f2_7_16_50_84": pct(f7, c7),
        "captured7_plx_ratio7_16_50_84": pct(r7, c7),
        "orbit_s9_16_50_84": pct(s9, orb),
        "orbit_plx_ratio9_16_50_84": pct(r9, orb),
        "orbit_9par_failed_s": float(np.mean(~ok9s[orb])),
        "orbit_9par_failed_f2_given_s": float(np.mean((ok9s & ~ok9f)[orb])),
        "orbit_9par_failed_parallax_only": float(np.mean((ok9s & ok9f & ~ok9p)[orb])),
        "orbit_7par_failed_s": float(np.mean(~ok7s[orb])),
        "orbit_7par_failed_f2_given_s": float(np.mean((ok7s & ~ok7f)[orb])),
        "orbit_7par_failed_parallax_only": float(np.mean((ok7s & ok7f & ~ok7p)[orb])),
        "capture_vs_span_over_P": frac_bins(span, [0.0, 0.8, 1.0, 1.3, 1.8]),
        "capture_vs_eccentricity": frac_bins(ecc, [0.0, 0.1, 0.3, 0.5, 1.0]),
        "capture_vs_published_significance_over_threshold": frac_bins(sig, [1.0, 1.5, 2.0, 4.0, 1e9]),
        "periastron_in_window_fraction": float(np.mean(lin["periastron_in_window"][sel])),
        "span_day_median": float(np.median(lin["span_day"][sel])),
    }


def fig_acceleration_statistics(d: dict[str, Any], gate: Path, path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    T, typ = d["truth"], d["typ"]
    lin, _ = _read(gate / "linear_replay.h5")
    si = lin["system_index"]
    sel = (typ[si] == "Orbital") & (T["period"][si] >= LONG_P) & (lin["realization"] < 5)
    b = lin["branch__0"]
    s9, f9, p9 = lin["s9__0"], lin["f2_9__0"], lin["plx_snr9__0"]
    r9 = p9 / (cr.GAIAMOCK_ACCEL9_PLX_COEFF * s9**cr.GAIAMOCK_ACCEL_PLX_EXPONENT)
    groups = [(cr.BRANCH_ORBITAL, "orbit fitted"), (cr.BRANCH_SEVEN_PARAMETER, "accepted 7-par"),
              (cr.BRANCH_NINE_PARAMETER, "accepted 9-par")]
    fig, ax2, leg_ax = _grid_with_legend_row(1, 2, (13.0, 6.4), 0.25)
    axes = ax2[0]
    for j, (code, lab) in enumerate(groups):
        m = sel & (b == code)
        st = series_style(j, style)
        axes[0].scatter(s9[m], r9[m], s=10, color=st["color"], marker=st["marker"], alpha=0.5,
                        label=f"{lab} (N = {int(m.sum())})")
        axes[1].scatter(s9[m], f9[m], s=10, color=st["color"], marker=st["marker"], alpha=0.5, label=lab)
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(cr.GAIAMOCK_ACCEL_SIGNIFICANCE_MIN, **reference_line_style(0, style), label="s = 12")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, **reference_line_style(1, style),
                    label=r"$\varpi/\sigma_\varpi = 2.1\,s_9^{1.05}$")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].axhline(cr.GAIAMOCK_ACCEL_F2_MAX, **reference_line_style(1, style), label="F2 = 25")
    apply_axes_style(axes[0], style, xlabel=r"9-parameter significance $s_9$",
                     ylabel=r"$(\varpi/\sigma_\varpi)\,/\,(2.1\,s_9^{1.05})$")
    apply_axes_style(axes[1], style, xlabel=r"9-parameter significance $s_9$", ylabel="9-parameter F2")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    leg_ax.legend(h0 + h1[-1:], l0 + l1[-1:], prop=legend_prop(style), loc="center", ncol=3,
                  frameon=False, markerscale=2)
    fig.suptitle("Orbital, published P > 600 d, 5 realizations each: 9-parameter checks", fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


def fig_capture_vs_coverage(br: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    panels = [("capture_vs_span_over_P", "observed span / P"),
              ("capture_vs_eccentricity", "published eccentricity"),
              ("capture_vs_published_significance_over_threshold",
               r"published $(a_0/\sigma_{a_0})\,/\,\max(5, 158/\sqrt{P})$")]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    st = series_style(0, style)
    for ax, (k, lab) in zip(axes, panels):
        rows = br[k]
        x = np.arange(len(rows))
        f = np.array([r["frac"] for r in rows])
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        ax.errorbar(x, f, yerr=[f - lo, hi - f], color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], linewidth=st["linewidth"], markersize=st["markersize"], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['lo_edge']:g}-{r['hi_edge']:g}".replace("-1e+09", "+") for r in rows])
        apply_axes_style(ax, style, xlabel=lab,
                         ylabel="accepted 7/9-parameter fraction" if ax is axes[0] else None,
                         enable_minor_ticks=False)
    fig.suptitle("Orbital, published P > 600 d (baseline)", fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


def capture_probability(d: dict[str, Any], gate: Path) -> dict[str, Any]:
    T, typ = d["truth"], d["typ"]
    lin, _ = _read(gate / "linear_replay.h5")
    dr, _ = _read(gate / "truth_draws.h5")
    fov, fa = _read(gate / "fov_correlated.h5")
    fracs = json.loads(fa["fractions"])
    out: dict[str, Any] = {}
    sysl = np.array(sorted({int(i) for i in lin["system_index"]
                            if typ[i] == "Orbital" and T["period"][i] >= LONG_P}))

    def per_sys(si: np.ndarray, cap: np.ndarray) -> np.ndarray:
        return np.array([cap[si == i].mean() if np.any(si == i) else np.nan for i in sysl])

    sets = {"fixed published truth": per_sys(lin["system_index"], _capture(lin["branch__0"])),
            "truth drawn from published covariance": per_sys(dr["system_index"], _capture(dr["branch"]))}
    for v, fr in enumerate(fracs):
        if fr > 0:
            sets[f"FOV-correlated noise f = {fr:g}"] = per_sys(fov["system_index"], _capture(fov[f"branch__{v}"]))
    for k, p in sets.items():
        p = p[np.isfinite(p)]
        out[k] = {"n_systems": int(len(p)), "mean": float(p.mean()),
                  "n_p_ge_0p9": int((p >= 0.9).sum()), "n_p_eq_1": int((p >= 1.0).sum()),
                  "n_p_le_0p1": int((p <= 0.1).sum()), "values": [float(x) for x in p]}
    # share of captures contributed by near-certain systems (baseline)
    p0 = sets["fixed published truth"]
    out["share_of_captures_from_p_ge_0p9"] = float(p0[p0 >= 0.9].sum() / p0.sum())
    return out


def fig_capture_probability(cp: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    keys = ["fixed published truth", "truth drawn from published covariance", "FOV-correlated noise f = 0.5"]
    edges = np.linspace(0, 1, 11)
    edges[-1] = 1.0001
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for j, k in enumerate(keys):
        st = series_style(j, style)
        h, _ = np.histogram(cp[k]["values"], bins=edges)
        ax.step(np.r_[edges[:-1], 1.0], np.r_[h, h[-1]], where="post", color=st["color"],
                linestyle=st["linestyle"], linewidth=st["linewidth"], label=f"{k} (N = {cp[k]['n_systems']})")
    ax.set_yscale("log")
    ax.set_ylim(3, 3000)
    apply_axes_style(ax, style, xlabel="per-system capture probability (25 realizations)",
                     ylabel="number of systems")
    ax.legend(prop=legend_prop(style), loc="upper center", frameon=False)
    ax.set_title("Orbital, published P > 600 d", fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


def forced_orbit(d: dict[str, Any], gate: Path, cuts: Any) -> dict[str, Any]:
    T = d["truth"]
    recs = []
    for line in (gate / "forced_orbit.jsonl").read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    with h5py.File(PRIMARY / "output/gate390/injection_test_full.h5", "r") as f:
        si = f["realizations/system_index"][()]
        rr = f["realizations/realization"][()]
        rp = f["realizations/recovered/period"][()]
        ra = f["realizations/recovered/a0_mas"][()]
        acc_s = f["realizations/recovered/acceleration_significance"][()]
        acc_f2 = f["realizations/recovered/goodness_of_fit"][()]
        outc = f["realizations/outcome"][()]
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(si, rr))}
    ctrl = [r for r in recs if r["role"] == "control"]
    n_ctrl_ok = sum(1 for r in ctrl if r["cascade"][10] == rp[idx[(r["system_index"], r["realization"])]]
                    and r["cascade"][17] == ra[idx[(r["system_index"], r["realization"])]])
    rows = []
    for r in recs:
        if r["role"] != "captured":
            continue
        p = it.parse_cascade_result(r["cascade"], n_visibility_periods=0, n_obs=0, cuts=cuts)
        k = idx[(r["system_index"], r["realization"])]
        rows.append(dict(accepted=p["accepted"], s_orb=p["significance"], f2_orb=p["goodness_of_fit"],
                         dp=(p["period"] - T["period"][r["system_index"]]) / T["period"][r["system_index"]],
                         s_acc=acc_s[k], f2_acc=acc_f2[k], n_par=int(outc[k]),
                         **{c: p[c] for c in it.CUT_FLAG_NAMES}))
    acc = np.array([r["accepted"] for r in rows])
    k_, n_ = int(acc.sum()), len(acc)
    f, lo, hi = it.binomial_fraction_interval(k_, n_)
    return {
        "n_control": len(ctrl), "n_control_reproduced_bit_for_bit": n_ctrl_ok,
        "n_captured_refit": n_, "would_pass_all_orbital_cuts": f, "interval": [lo, hi],
        "orbit_s_median": float(np.median([r["s_orb"] for r in rows])),
        "orbit_f2_median": float(np.median([r["f2_orb"] for r in rows])),
        "acceleration_s_median": float(np.median([r["s_acc"] for r in rows])),
        "acceleration_f2_median": float(np.median([r["f2_acc"] for r in rows])),
        "abs_dP_over_P_median": float(np.median(np.abs([r["dp"] for r in rows]))),
        "cut_pass_fractions": {c: float(np.mean([r[c] for r in rows])) for c in it.CUT_FLAG_NAMES},
        "cpu_s_median": float(np.median([r["cpu_s"] for r in recs])),
        "_rows": rows,
    }


def fig_forced_orbit(fo: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    rows = fo["_rows"]
    fig, ax2, leg_ax = _grid_with_legend_row(1, 2, (13.0, 6.0), 0.18)
    axes = ax2[0]
    for j, (flag, lab) in enumerate(((True, "orbit passes all cuts"), (False, "orbit fails a cut"))):
        m = [r for r in rows if r["accepted"] == flag]
        st = series_style(j, style)
        axes[0].scatter([r["s_acc"] for r in m], [r["s_orb"] for r in m], s=14, color=st["color"],
                        marker=st["marker"], alpha=0.7, label=f"{lab} (N = {len(m)})")
        axes[1].scatter([r["f2_acc"] for r in m], [r["f2_orb"] for r in m], s=14, color=st["color"],
                        marker=st["marker"], alpha=0.7, label=lab)
    axes[0].set_yscale("log")
    axes[0].axhline(5.0, **reference_line_style(0, style), label=r"$a_0/\sigma_{a_0} = 5$")
    axes[1].axhline(25.0, **reference_line_style(0, style), label="F2 = 25")
    axes[1].axvline(25.0, **reference_line_style(1, style))
    apply_axes_style(axes[0], style, xlabel="accepted acceleration significance s",
                     ylabel=r"skipped orbit fit $a_0/\sigma_{a_0}$")
    apply_axes_style(axes[1], style, xlabel="accepted acceleration F2", ylabel="skipped orbit fit F2")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    leg_ax.legend(h0 + h1[-1:], l0 + l1[-1:], prop=legend_prop(style), loc="center", ncol=4,
                  frameon=False, markerscale=2)
    fig.suptitle("Captured realizations (P > 600 d) refit with the acceleration branch off, same data",
                 fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


# ---------------------------------------------------------------------------- #398


def sigma_decomposition(d: dict[str, Any]) -> dict[str, Any]:
    T, typ, R, si = d["truth"], d["typ"], d["rec"], d["si"]
    m = d["acc"] & (typ[si] == "Orbital") & np.isfinite(d["n_good"][si])
    n_mock = R["n_obs"][m]
    n_dr3 = d["n_good"][si[m]]
    c_mock = cr.inflation_factor_from_f2(R["goodness_of_fit"][m], n_mock - 12)
    c_dr3 = cr.inflation_factor_from_f2(T["goodness_of_fit"][si[m]], n_dr3 - 12)
    f_n = np.sqrt(n_mock / n_dr3)
    f_c = c_dr3 / c_mock
    ratios = {
        "parallax": R["parallax_error"][m] / T["parallax_error"][si[m]],
        "a0": R["sigma_a0_mas"][m] / T["sigma_a0_mas"][si[m]],
        "period": R["period_error"][m] / T["period_error"][si[m]],
        "eccentricity": R["eccentricity_error"][m] / T["eccentricity_error"][si[m]],
    }
    cov = {
        "G": T["g_mag"][si[m]],
        "log10_P": np.log10(T["period"][si[m]]),
        "log10_a0_over_sigma": np.log10(T["significance"][si[m]]),
        "abs_ecl_lat": np.abs(T["published_ecl_lat"][si[m]]),
    }
    edges = {
        "G": G_EDGES,
        "log10_P": np.quantile(cov["log10_P"], np.linspace(0, 1, 6)),
        "log10_a0_over_sigma": np.quantile(cov["log10_a0_over_sigma"], np.linspace(0, 1, 6)),
        "abs_ecl_lat": np.array([0, 15, 30, 45, 60, 90.01]),
    }
    edges["log10_P"][-1] += 1e-9
    edges["log10_a0_over_sigma"][-1] += 1e-9
    out: dict[str, Any] = {"n": int(m.sum())}
    for name, r in ratios.items():
        out[name] = {
            "raw_median": float(np.nanmedian(r)),
            "times_sqrt_N_median": float(np.nanmedian(r * f_n)),
            "times_sqrt_N_times_c_ratio_median": float(np.nanmedian(r * f_n * f_c)),
            "times_sqrt_N_times_c_ratio_16_84": [float(v) for v in np.nanpercentile(r * f_n * f_c, [16, 84])],
        }
    out["N_mock_over_N_dr3_median"] = float(np.median(n_mock / n_dr3))
    out["N_mock_over_N_dr3_16_84"] = [float(v) for v in np.percentile(n_mock / n_dr3, [16, 84])]
    out["c_dr3_over_c_mock_median"] = float(np.nanmedian(f_c))
    by_g = []
    G = cov["G"]
    for lo, hi in zip(G_EDGES[:-1], G_EDGES[1:]):
        s = (G >= lo) & (G < hi)
        by_g.append({"G_lo": float(lo), "G_hi": float(hi), "n": int(s.sum()),
                     "parallax_raw": float(np.median(ratios["parallax"][s])),
                     "parallax_times_sqrt_N": float(np.median((ratios["parallax"] * f_n)[s])),
                     "parallax_corrected": float(np.nanmedian((ratios["parallax"] * f_n * f_c)[s])),
                     "a0_raw": float(np.median(ratios["a0"][s])),
                     "a0_corrected": float(np.nanmedian((ratios["a0"] * f_n * f_c)[s])),
                     "N_ratio": float(np.median((n_mock / n_dr3)[s])),
                     "c_dr3": float(np.nanmedian(c_dr3[s])), "c_mock": float(np.nanmedian(c_mock[s])),
                     "c_ratio": float(np.nanmedian(f_c[s])),
                     "published_F2": float(np.median(T["goodness_of_fit"][si[m]][s])),
                     "recovered_F2": float(np.median(R["goodness_of_fit"][m][s]))})
    out["by_G"] = by_g
    out["_arrays"] = dict(ratios=ratios, cov=cov, edges=edges, f_n=f_n, f_c=f_c, c_mock=c_mock, c_dr3=c_dr3,
                          n_ratio=n_mock / n_dr3)
    return out


def fig_sigma_decomposition(sd: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    a = sd["_arrays"]
    labels = {"G": "G (mag)", "log10_P": r"$\log_{10}(P/\mathrm{d})$",
              "log10_a0_over_sigma": r"published $\log_{10}(a_0/\sigma_{a_0})$",
              "abs_ecl_lat": r"$|\beta|$ (deg)"}
    params = [("parallax", r"$\sigma_\varpi$"), ("a0", r"$\sigma_{a_0}$"), ("period", r"$\sigma_P$")]
    fig, axes, leg_ax = _grid_with_legend_row(len(params), 4, (17.0, 12.0), 0.12, sharey=True)
    for i, (pn, plab) in enumerate(params):
        r = a["ratios"][pn]
        for j, cn in enumerate(labels):
            ax = axes[i, j]
            x = a["cov"][cn]
            for k, (y, lab) in enumerate(((r, "raw"), (r * a["f_n"], r"$\times\sqrt{N_\mathrm{mock}/N_\mathrm{DR3}}$"),
                                          (r * a["f_n"] * a["f_c"], r"$\times\sqrt{N_\mathrm{mock}/N_\mathrm{DR3}}\;c_\mathrm{DR3}/c_\mathrm{mock}$"))):
                c, med, lo, hi, _ = _binned_median(x, y, a["edges"][cn])
                st = series_style(k, style)
                ax.plot(c, med, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                        linewidth=st["linewidth"], markersize=st["markersize"], label=lab)
                ax.fill_between(c, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
            ax.axhline(1.0, **reference_line_style(0, style))
            apply_axes_style(ax, style, xlabel=labels[cn] if i == len(params) - 1 else None,
                             ylabel=f"recovered / published {plab}" if j == 0 else None)
    h, lab = axes[0, 0].get_legend_handles_labels()
    leg_ax.legend(h, lab, prop=legend_prop(style), loc="center", ncol=3, frameon=False)
    for ax in axes.ravel():
        ax.set_ylim(0.4, 1.8)
    fig.suptitle("#398: accepted Orbital realizations, median and 16th-84th percentile per bin",
                 fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


def fig_c_factor(sd: dict[str, Any], path: Path, style: Any, dpi: int) -> None:
    plt = require_pyplot()
    a = sd["_arrays"]
    G = a["cov"]["G"]
    edges = np.arange(6.0, 18.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for k, (y, lab) in enumerate(((a["c_dr3"], r"$c_\mathrm{DR3}$ (published F2)"),
                                  (a["c_mock"], r"$c_\mathrm{mock}$ (recovered F2)"))):
        c, med, lo, hi, _ = _binned_median(G, y, edges)
        st = series_style(k, style)
        axes[0].plot(c, med, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                     linewidth=st["linewidth"], markersize=st["markersize"], label=lab)
        axes[0].fill_between(c, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
    axes[0].axvline(13.0, **reference_line_style(1, style), label="G = 13")
    apply_axes_style(axes[0], style, xlabel="G (mag)", ylabel="inflation factor c (Halbwachs Eq. 2)")
    axes[0].legend(prop=legend_prop(style), loc="upper right", frameon=False)
    c, med, lo, hi, _ = _binned_median(G, a["n_ratio"], edges)
    st = series_style(2, style)
    axes[1].plot(c, med, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                 linewidth=st["linewidth"], markersize=st["markersize"])
    axes[1].fill_between(c, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
    axes[1].axhline(1.0, **reference_line_style(0, style))
    apply_axes_style(axes[1], style, xlabel="G (mag)",
                     ylabel=r"$N_\mathrm{obs}$ (gaiamock) / $N_\mathrm{good}$ (DR3)")
    fig.suptitle("#398: accepted Orbital realizations", fontfamily="serif")
    save_figure(fig, path, dpi=dpi)


def _strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not k.startswith("_") and k != "values"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default=str(PRIMARY / "output" / "gate399"))
    p.add_argument("--input", default=str(PRIMARY / "output" / "gate390" / "injection_test_full.h5"))
    p.add_argument("--visibility", default=str(PRIMARY / "output" / "gate390" / "gaia_source_visibility.json"))
    p.add_argument("--fig-dir", default=str(REPO / "docs" / "gate399" / "figures"))
    args = p.parse_args()
    cfg = load_config()
    style = cfg.plotting
    dpi = int(cfg.diagnostics.figure_dpi)
    cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
    gate, figs = Path(args.gate), Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    d = _load_390(Path(args.input), Path(args.visibility))
    summary: dict[str, Any] = {}
    summary["capture_tests"] = capture_tests(d, gate)
    fig_capture_vs_period(summary["capture_tests"], figs / "capture_vs_period.png", style, dpi)
    summary["acceleration_breakdown"] = acceleration_breakdown(d, gate)
    fig_acceleration_statistics(d, gate, figs / "acceleration_statistics.png", style, dpi)
    fig_capture_vs_coverage(summary["acceleration_breakdown"], figs / "capture_vs_coverage.png", style, dpi)
    cp = capture_probability(d, gate)
    fig_capture_probability(cp, figs / "capture_probability.png", style, dpi)
    summary["capture_probability"] = cp
    fo = forced_orbit(d, gate, cuts)
    fig_forced_orbit(fo, figs / "forced_orbit.png", style, dpi)
    summary["forced_orbit"] = fo
    _, ma = _read(gate / "linear_matched.h5")
    summary["dr3_matched_k_c_by_G"] = {"g_edges": json.loads(ma["g_edges"]), "k_c": json.loads(ma["k_c_bins"])}
    sd = sigma_decomposition(d)
    fig_sigma_decomposition(sd, figs / "sigma_ratio_decomposition.png", style, dpi)
    fig_c_factor(sd, figs / "c_factor_vs_g.png", style, dpi)
    summary["sigma_decomposition_398"] = sd
    (figs / "summary.json").write_text(json.dumps(_strip(summary), indent=1))
    print(json.dumps(_strip({k: summary[k] for k in ("forced_orbit", "dr3_matched_k_c_by_G")}), indent=1))


if __name__ == "__main__":
    main()
