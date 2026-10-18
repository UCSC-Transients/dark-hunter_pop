#!/usr/bin/env python
"""#391 follow-ups (Ryan 2026-10-18) contact sheet: distant short-P re-injection, slope-break profile,
real vs mock M1 at G < 10, bright isochrone-M1 failure rates. Reads the JSON products and the fit dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def g_ratio(z, edges, scale=None):
    w = z["w"] * (1.0 if scale is None else scale)
    so, rin = z["ob"] >= 0, z["rob"] >= 0
    m = np.histogram(z["m_g"][so], bins=edges, weights=w[so])[0]
    s = np.histogram(z["r_g"][rin], bins=edges, weights=z["r_spur_per_orbit"][rin])[0]
    r = np.histogram(z["r_g"][rin], bins=edges)[0]
    return (m + s) / r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("docs/gate391/followups"))
    ap.add_argument("--inject-dir", type=Path, default=Path("output/proposal_set/shortp_inject"))
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--dump", type=Path, default=Path("output/proposal_set/dump38/dump_noW.npz"))
    ap.add_argument("--dump-break", type=Path, default=Path("output/proposal_set/dump38_mbreak/dump_noW.npz"))
    args = ap.parse_args(argv)
    import h5py

    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config(Path("config/config.yaml"))
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    inj = json.loads((args.dir / "shortp_injection.json").read_text())
    rm = json.loads((args.dir / "real_m1.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))

    ax = axs[0, 0]
    names = list(inj["groups"])
    rec = [inj["groups"][n]["recovery"] for n in names]
    x = np.arange(len(names))
    ax.errorbar(x, [r[0] for r in rec], yerr=[[r[0] - r[1] for r in rec], [r[2] - r[0] for r in rec]], fmt="o",
                color=series_style(0, style)["color"], ms=8)
    for i, n in enumerate(names):
        rj = inj["groups"][n]["rejections"]
        nrej = sum(rj.values())
        plx = sum(v for k, v in rj.items() if "parallax_over_error" in k)
        ax.annotate(f"rejected {nrej}: ϖ/σ cut {plx / nrej:.0%},\nRUWE gate {rj.get('OUTCOME_FIVE_PARAMETER', 0) / nrej:.0%}",
                    (i, rec[i][0]), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["target", "control: nearer", "control: longer P"], fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)
    apply_axes_style(ax, style, xlabel="", ylabel="re-detected fraction (3 realizations)")
    ax.set_title("(1) distant short-P re-injection (v2 + N2d + VP loss)")

    ax = axs[0, 1]
    with h5py.File(args.published, "r") as h:
        p = h["published"]
        sid = p["source_id"][()]
        pe = dict(zip(sid.tolist(), p["parallax_error"][()].tolist()))
        pp = dict(zip(sid.tolist(), p["parallax"][()].tolist()))
        per = dict(zip(sid.tolist(), p["period"][()].tolist()))
    with h5py.File(args.inject_dir / "systems.h5", "r") as h:
        grp = dict(zip(h["systems/source_id"][()].tolist(), h["systems/group"][()].tolist()))
    recs = [json.loads(s) for s in (args.inject_dir / "inject.jsonl").read_text().splitlines()]
    for gi in (0, 1, 2):
        rr = [r for r in recs if grp.get(int(r["source_id"])) == gi and (r.get("parallax_error") or 0) > 0]
        xs = np.array([pp[int(r["source_id"])] / pe[int(r["source_id"])] / (20000 / per[int(r["source_id"])]) for r in rr])
        ys = np.array([r["parallax_error"] / pe[int(r["source_id"])] for r in rr])
        ax.scatter(xs, ys, s=8, alpha=0.6, color=series_style(gi, style)["color"], label=["target", "control near", "control longer P"][gi])
    ax.axvline(1, color="0.5", lw=1)
    ax.axhline(1, color="0.5", lw=1)
    ax.set_xlim(0.95, 2.0)
    ax.set_ylim(0.6, 2.2)
    apply_axes_style(ax, style, xlabel="published (ϖ/σ_ϖ) / (20000 d / P)", ylabel="σ_ϖ recovered / published")
    ax.set_title("(1) the binding cut: ϖ/σ_ϖ > 20000 d / P")
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    e = [5, 8, 9, 10, 11, 12, 13, 14, 16, 19.5]
    gc = 0.5 * (np.array(e[1:]) + np.array(e[:-1]))
    za, zb = np.load(args.dump), np.load(args.dump_break)
    m = za["m1_law"]
    ahi = float(za["theta"][2])
    ax.plot(gc, g_ratio(za, e), marker="o", label="noW fit", color=series_style(0, style)["color"])
    ax.plot(gc, g_ratio(zb, e), marker="s", ls="--", label="noW + slope break 1.4", color=series_style(1, style)["color"])
    ax.plot(gc, g_ratio(za, e, np.where(m > 0.8, (m / 0.8) ** (-ahi), 1.0)), marker="^", ls=":",
            label="α_hi → 0 (reweighted, not refit)", color=series_style(2, style)["color"])
    ax.axhline(1, color="0.5", lw=1)
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="(mock + spurious) / real, orbits")
    ax.set_title("(2) slope break profile")
    ax.legend(fontsize=8)

    for j, kind in enumerate(("orbit", "accel")):
        ax = axs[1, j]
        d = rm["m1"]["G<10"]
        me = np.array(rm["m1_edges"], float)
        lab = [f"{a:g}-{b:g}" for a, b in zip(me[:-2], me[1:-1])]
        mk = np.array(d[f"mock_{kind}_sum_w_by_M1"])[:-1]
        rk = np.array(d[f"real_{kind}_count_by_M1"])[:-1]
        xx = np.arange(lab.__len__())
        ax.bar(xx - 0.2, mk, width=0.4, color=series_style(0, style)["color"], label="mock (best-fit weight)")
        ax.bar(xx + 0.2, rk, width=0.4, color=series_style(1, style)["color"], label="real (isochrone M̂1)")
        for i in range(len(xx)):
            if rk[i] > 0:
                ax.annotate(f"{mk[i] / rk[i]:.1f}", (xx[i], max(mk[i], rk[i])), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(xx)
        ax.set_xticklabels(lab, fontsize=7, rotation=45)
        apply_axes_style(ax, style, xlabel="row M̂1 (M⊙)", ylabel="count, G < 10")
        ax.set_title(f"(3) {'orbits' if kind == 'orbit' else 'accelerations'} at G < 10 by M̂1 (ratio mock/real)")
        ax.legend(fontsize=8)

    ax = axs[1, 2]
    ge = np.array(rm["g_edges"], float)
    gcc = 0.5 * (ge[1:] + ge[:-1])
    for i, (k, lab) in enumerate((("parent_bj_stored", "parent (BJ, stored)"), ("parent_plx_convention", "parent (1/ϖ, real convention)"),
                                  ("real_orbits", "real C1 orbits"), ("real_accelerations", "real accelerations"))):
        y = np.array(rm["failure"][k]["m1_ok"], float)
        ax.plot(gcc, y, marker="o", ls="-" if i < 2 else "--", label=lab, color=series_style(i, style)["color"])
    apply_axes_style(ax, style, xlabel="G (mag)", ylabel="isochrone-M1 success")
    ax.set_title("(4) bright isochrone-M1 failures")
    ax.legend(fontsize=8)
    fig.suptitle("#391 follow-ups (Ryan 2026-10-18): distant short-P re-injection, slope break, M1 at G < 10, M1 failures")
    save_figure(fig, args.dir / "followups.png", dpi=int(cfg.diagnostics.figure_dpi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
