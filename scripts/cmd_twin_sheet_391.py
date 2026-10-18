#!/usr/bin/env python
"""#391 (Ryan 2026-10-18) contact sheet for the CMD / twin test, the 2–3 M⊙ split and the σ_ϖ test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("docs/gate391/cmd_twin"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--dump", type=Path, default=Path("output/proposal_set/dump38/dump_noW.npz"))
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_batch38.yaml"))
    args = ap.parse_args(argv)
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config(Path("config/config.yaml"))
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    ct = json.loads((args.dir / "cmd_twin.json").read_text())
    ms = json.loads((args.dir / "mult_sigma.json").read_text())
    pts = np.load(args.dir / "cmd_points.npz")
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, ps.load_proposal_set_fragment(args.fragment).proposal)
    z = np.load(args.dump, allow_pickle=True)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    c_lo, c_hi = ct["ms_colour_range"]

    def dm(c, m):
        r = np.interp(c, pts["ridge_c"], pts["ridge_m"], left=np.nan, right=np.nan)
        return m - r

    ax = axs[0, 0]
    e = np.arange(-2.0, 1.01, 0.05)
    for i, (lab, c, m, d, wt) in enumerate((("parent", pts["p_c0"], pts["p_mg0"], pts["p_d"], np.ones(pts["p_c0"].size)),
                                            ("real orbit hosts", pts["ro_c0"], pts["ro_mg0"], pts["ro_d"], pts["ro_w"]),
                                            ("mock-detected orbits", pts["mo_c0"], pts["mo_mg0"], pts["mo_d"], pts["mo_w"]))):
        s = (c >= c_lo) & (c <= c_hi) & (d >= 0.1) & (d < 0.5) & np.isfinite(m)
        h = np.histogram(dm(c[s], m[s]), bins=e, weights=wt[s])[0]
        ax.step(0.5 * (e[1:] + e[:-1]), h / h.sum(), where="mid", lw=2, color=series_style(i, style)["color"], label=lab)
    for x0, x1 in (ct["twin_band"], ct["single_band"]):
        ax.axvspan(x0, x1, color="0.9", zorder=0)
    ax.set_yscale("log")
    apply_axes_style(ax, style, xlabel="ΔM = M_G0 − ridge(C0) (mag), C0 0.85–2.0, d 0.1–0.5 kpc", ylabel="fraction")
    ax.set_title("(0) offset from the MS ridge (twin band −0.95…−0.55)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    de = np.array(ct["d_edges"])
    dc = 0.5 * (de[1:] + de[:-1])
    for i, nm in enumerate(("parent", "real_orbits", "real_accels", "mock_orbits", "mock_accels")):
        y = [r["twin_over_single"] for r in ct["by_distance"][nm]]
        ax.plot(dc, y, marker="o", ls="-" if "real" in nm or nm == "parent" else "--", color=series_style(i, style)["color"], label=nm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    apply_axes_style(ax, style, xlabel="distance (kpc)", ylabel="twin / single (MS, C0 0.85–2.0)")
    ax.set_title("(0) twin-sequence occupancy by distance")
    ax.legend(fontsize=7)

    ax = axs[0, 2]
    iso = ct["isochrone_vs_dm"]
    dmb = np.array(iso["dm_edges"])
    dmc = 0.5 * (dmb[1:] + dmb[:-1])
    axf = ax.twinx()
    for i, g in enumerate(("10-13", "13-16")):
        rows = iso["by_g"][g]["rows"]
        ax.plot(dmc, [r["m1_median_ms"] for r in rows], marker="o", color=series_style(i, style)["color"], label=f"M̂1, G {g}")
        axf.plot(dmc, [r["feh_median_ms"] for r in rows], marker="s", ls=":", color=series_style(i, style)["color"], label=f"[Fe/H], G {g}")
    axf.set_ylabel("posterior [Fe/H] median (dotted)")
    ax.axvspan(*ct["twin_band"], color="0.9", zorder=0)
    apply_axes_style(ax, style, xlabel="ΔM (mag)", ylabel="posterior M̂1 median (M⊙), solid")
    ax.set_title("(0) parent MS rows: isochrone fit vs ΔM")
    ax.legend(fontsize=7, loc="upper left")

    ax = axs[1, 0]
    pg = np.asarray(parent.columns["phot_g_mean_mag"], float)
    c0, m0 = np.asarray(parent.cmd["colour0"], float), np.asarray(parent.cmd["mg0"], float)
    ok = np.asarray(parent.isochrone["ok"], bool)
    sb = pg < 11
    ax.scatter(c0[sb & ok], m0[sb & ok], s=6, color="0.6", label="parent G < 11, M̂1 ok")
    ax.scatter(c0[sb & ~ok], m0[sb & ~ok], s=10, color=series_style(1, style)["color"], label="parent G < 11, off grid / no CMD")
    ax.plot(pts["ridge_c"], pts["ridge_m"], color="k", lw=2, label="MS ridge")
    ax.invert_yaxis()
    apply_axes_style(ax, style, xlabel="(BP−RP)_0", ylabel="M_G0")
    ax.set_title("(2) where the bright isochrone failures are")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    f = ms["fine_m1_G<10"]
    me = np.array(f["edges"])
    mc = np.arange(me.size - 1)
    for i, kind in enumerate(("orbit", "accel")):
        r = np.array(f[kind]["ratio"], float)
        ax.plot(mc + (i - 0.5) * 0.15, r, marker="o", ls="none", color=series_style(i, style)["color"], label=f"{kind}s: mock / real")
    ax.axhline(1, color="0.5", lw=1)
    ax.set_xticks(mc)
    ax.set_xticklabels([f"{a:g}-{b:g}" for a, b in zip(me[:-1], me[1:])], rotation=45, fontsize=7)
    ax.set_yscale("log")
    apply_axes_style(ax, style, xlabel="row M̂1 (M⊙), G < 10", ylabel="mock / real")
    ax.set_title("(1) G < 10 by M̂1 (181 parent rows behind it)")
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    rows = ms["sigma_parallax"]["rows"]
    for i, far in enumerate((True, False)):
        rr = [r for r in rows if r["away_from_cut(>1.5x)"] is far]
        x = [0.5 * sum(r["log_p"]) for r in rr]
        ax.plot(x, [r["sigma_ratio_median"] for r in rr], marker="o", color=series_style(i, style)["color"],
                label=f"σ_ϖ rec/pub, {'away from' if far else 'near'} cut")
        ax.plot(x, [r["cc_ratio_from_F2_median"] for r in rr], marker="s", ls=":", color=series_style(i, style)["color"],
                label=f"F2 inflation ratio, {'away' if far else 'near'}")
    ax.axhline(1, color="0.5", lw=1)
    apply_axes_style(ax, style, xlabel="log10 P (d)", ylabel="ratio (median)")
    ax.set_title("(3) σ_ϖ: gaiamock vs DR3 (#390 set, v2+N2-u0)")
    ax.legend(fontsize=7)
    fig.suptitle("#391 Ryan's CMD / twin test, bright-end M̂1 split and σ_ϖ (gens 33–38 noW best fit)")
    save_figure(fig, args.dir / "cmd_twin.png", dpi=int(cfg.diagnostics.figure_dpi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
