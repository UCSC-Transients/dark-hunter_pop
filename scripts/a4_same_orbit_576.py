#!/usr/bin/env python3
"""#576 (A4): same-orbit decomposition of the mock vs DR3 a0/sigma_a0 offset, before/after the #572 noise model.

Same orbits on both sides: the 1,296 #390 published orbits re-injected through the epoch model
(``scripts/reinject_noise_colour_572.py``), production ``n2d_s1232`` and ``colour_v1``. For every accepted realization:
sigma_a0, sigma_plx, the noise-free geometry ratio sigma_a0/sigma_plx and RUWE, each mock/DR3, as a function of
A = (a0/plx)/P_yr^(2/3), G, BP-RP, P, e and N_vis (DR3). The log offset of z = (a0/sigma_a0)/sqrt(RUWE^2 - 1) is
split exactly into log(a0_m/a0_d) - log(sigma_a0 m/d) - 0.5 log((RUWE_m^2-1)/(RUWE_d^2-1)).

Secondary: fractional errors sigma_a0/a0 and sigma_plx/plx vs A and G for the full populations (rung-3 v3 luminous
gens 42-46 accepted orbits at the default-target weights vs all DR3 Orbital + AstroSpectroSB1).

Light analysis (no simulation). Writes ``a4.json`` and figures under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import plotting as pl

BINS = {
    "A": np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 5.0]),
    "g": np.array([6.0, 10.5, 11.5, 12.5, 13.0, 14.0, 15.0, 17.0]),
    "bp_rp": np.array([-0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.5]),
    "logp": np.array([0.0, 1.5, 2.0, 2.5, 2.75, 3.0, 4.0]),
    "e": np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
    "nvis": np.array([0, 15, 18, 21, 24, 28, 60]),
}
LABEL = {"A": r"$A=(a_0/\varpi)/P^{2/3}$", "g": "$G$ (mag)", "bp_rp": r"$G_{\rm BP}-G_{\rm RP}$ (mag)", "logp": r"$\log_{10}P$ (d)",
         "e": "$e$", "nvis": r"$N_{\rm vis}$ (DR3)"}


def load_pairs(run: Path, gate390: Path, colours: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(gate390, "r") as f:
        T = f["systems/truth"]
        tr = {k: T[k][()] for k in ("a0_mas", "sigma_a0_mas", "significance", "parallax", "parallax_error", "ruwe", "goodness_of_fit",
                                    "g_mag", "period", "eccentricity", "published_visibility_periods_used")}
        ix = {int(s_): i for i, s_ in enumerate(f["systems/source_id"][()])}
    with h5py.File(colours, "r") as f:
        cmap = dict(zip(f["source_id"][()].tolist(), f["bp_rp"][()].tolist()))
    rows = [json.loads(x) for x in (run / "inject.jsonl").read_text().splitlines()]
    acc = [r for r in rows if r.get("accepted")]
    i = np.array([ix[r["source_id"]] for r in acc])
    g = lambda k: np.array([r[k] for r in acc], float)
    d = {"sid": np.array([r["source_id"] for r in acc]), "real": np.array([r["realization"] for r in acc]),
         "a0_m": g("a0_mas"), "sa0_m": g("sigma_a0_mas"), "plx_m": g("parallax"), "splx_m": g("parallax_error"), "ruwe_m": g("ruwe"),
         "f2_m": g("goodness_of_fit"), "snr_m": g("significance"),
         "a0_d": tr["a0_mas"][i], "sa0_d": tr["a0_mas"][i] / tr["significance"][i], "plx_d": tr["parallax"][i],
         "splx_d": tr["parallax_error"][i], "ruwe_d": tr["ruwe"][i], "f2_d": tr["goodness_of_fit"][i], "snr_d": tr["significance"][i],
         "g": tr["g_mag"][i], "logp": np.log10(tr["period"][i]), "e": tr["eccentricity"][i], "nvis": tr["published_visibility_periods_used"][i],
         "bp_rp": np.array([cmap.get(int(s), np.nan) for s in (r["source_id"] for r in acc)], float)}
    d["A"] = (d["a0_d"] / d["plx_d"]) / (10 ** d["logp"] / 365.25) ** (2.0 / 3.0)
    d["mg"] = d["g"] + 5.0 * np.log10(np.clip(d["plx_d"], 1e-3, None) / 100.0)  # no extinction: an M_G proxy
    with np.errstate(invalid="ignore", divide="ignore"):
        d["l_sa0"] = np.log10(d["sa0_m"] / d["sa0_d"])
        d["l_splx"] = np.log10(d["splx_m"] / d["splx_d"])
        d["l_geom"] = d["l_sa0"] - d["l_splx"]  # (sigma_a0/sigma_plx) mock over DR3: noise-free geometry ratio
        d["l_ruwe"] = np.log10(d["ruwe_m"] / d["ruwe_d"])
        d["l_a0"] = np.log10(d["a0_m"] / d["a0_d"])
        d["l_zruwe"] = -0.5 * np.log10((d["ruwe_m"] ** 2 - 1.0) / (d["ruwe_d"] ** 2 - 1.0))
        d["l_z"] = d["l_a0"] - d["l_sa0"] + d["l_zruwe"]  # exact split of log z_m/z_d
        d["l_plxsnr"] = np.log10((d["plx_m"] / d["splx_m"]) / (d["plx_d"] / d["splx_d"]))
    return d


def binned(d: dict, key: str, val: str, edges: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    out = {"edges": edges.tolist(), "median": [], "err": [], "n": [], "x": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = np.flatnonzero((d[key] >= lo) & (d[key] < hi) & np.isfinite(d[val]))
        out["n"].append(int(b.size))
        if b.size < 15:
            out["median"].append(float("nan")), out["err"].append(float("nan")), out["x"].append(float("nan"))
            continue
        out["median"].append(float(np.median(d[val][b])))
        out["err"].append(float(np.std([np.median(d[val][rng.choice(b, b.size)]) for _ in range(n_boot)])))
        out["x"].append(float(np.median(d[key][b])))
    return out


QUANTS = ("l_z", "l_a0", "l_sa0", "l_zruwe", "l_splx", "l_geom", "l_ruwe", "l_plxsnr")


def population(a: argparse.Namespace) -> dict:
    """Fractional errors for the full populations (secondary)."""
    import h5py
    from astropy.table import Table

    c = np.load(a.cache)
    acc = c["accepted"]
    cas = c["cascade"][acc]
    n = c["n_per_gen"].astype(float)
    lq = np.logaddexp.reduce(c["lqs"][:, acc] + np.log(n / n.sum())[:, None], axis=0)
    lt = c["ll"][acc] + c["lw"][acc] - lq
    w = np.where(np.isfinite(lt), np.exp(lt - np.nanmax(lt)), 0.0)
    mock = {"g": c["phot_g_mean_mag"][acc], "fa0": cas[:, 18] / cas[:, 17], "fplx": cas[:, 1] / cas[:, 0], "w": w,
            "A": (cas[:, 17] / cas[:, 0]) / (cas[:, 10] / 365.25) ** (2.0 / 3.0)}
    en = Table.read(a.enrichment, format="ascii.ecsv")
    en = en[np.isin(np.asarray(en["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
    A_, B_, F_, G_ = (np.asarray(en[k], float) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0 = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))  # Halbwachs et al. (2023) Eq. A.?, as gaiamock
    plx = np.asarray(en["parallax"], float)
    sid = np.asarray(en["SOURCE_ID"], np.int64)
    with h5py.File(Path(a.input_columns) / "columns.h5", "r") as f:
        ic = {k: f[k][()] for k in ("source_id", "phot_g_mean_mag")}
    gm = dict(zip(ic["source_id"].tolist(), ic["phot_g_mean_mag"].tolist()))
    real = {"g": np.array([gm.get(int(s), np.nan) for s in sid]), "fa0": 1.0 / np.asarray(en["significance"], float),
            "fplx": np.asarray(en["parallax_error"], float) / plx, "w": np.ones(sid.size),
            "A": (a0 / plx) / (np.asarray(en["period"], float) / 365.25) ** (2.0 / 3.0)}
    out = {}
    for side, s in (("dr3", real), ("mock", mock)):
        ok = (s["g"] >= 9) & (s["g"] < 19) & np.isfinite(s["fa0"]) & np.isfinite(s["fplx"]) & (s["fplx"] > 0) & (s["w"] > 0)
        out[side] = {}
        for key in ("A", "g"):
            e = BINS[key]
            out[side][key] = {"edges": e.tolist(), "fa0": [], "fplx": []}
            for lo, hi in zip(e[:-1], e[1:]):
                b = ok & (s[key] >= lo) & (s[key] < hi)
                for q in ("fa0", "fplx"):
                    if b.sum() < 20:
                        out[side][key][q].append(float("nan"))
                        continue
                    o = np.argsort(s[q][b])
                    cw = np.cumsum(s["w"][b][o])
                    out[side][key][q].append(float(np.interp(0.5 * cw[-1], cw, s[q][b][o])))
    return out


def main(argv: list[str] | None = None) -> int:
    P = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="label=reinjection_dir")
    ap.add_argument("--gate390", default=str(P / "output/gate390/injection_test_full.h5"))
    ap.add_argument("--colours", default=str(P / "data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400/inj390.h5"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--enrichment", required=True)
    ap.add_argument("--input-columns", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=100)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(576)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    runs = {lab: load_pairs(Path(p), Path(a.gate390), Path(a.colours)) for lab, p in (s.split("=", 1) for s in a.runs)}
    res: dict = {"runs": {k: int(v["sid"].size) for k, v in runs.items()}, "overall": {}, "binned": {}}
    for lab, d in runs.items():
        ok = np.isfinite(d["l_z"])
        res["overall"][lab] = {q: float(np.median(d[q][np.isfinite(d[q])])) for q in QUANTS}
        res["overall"][lab]["n_z"] = int(ok.sum())
        res["binned"][lab] = {key: {q: binned(d, key, q, e, a.n_boot, rng) for q in QUANTS} for key, e in BINS.items()}
    res["population_fractional_errors"] = population(a)
    (a.out_dir / "a4.json").write_text(json.dumps(res, indent=1, default=float))
    for lab in runs:
        print(lab, {k: round(v, 4) for k, v in res["overall"][lab].items()})
    plots(runs, res, a)
    return 0


def plots(runs: dict, res: dict, a: argparse.Namespace) -> None:
    plt = pl.require_pyplot()
    style = pl.default_plotting_style()
    labs = list(runs)
    qlab = {"l_sa0": r"$\sigma_{a_0}$", "l_splx": r"$\sigma_\varpi$", "l_geom": r"$\sigma_{a_0}/\sigma_\varpi$", "l_ruwe": "RUWE",
            "l_a0": "$a_0$", "l_z": "$z$", "l_zruwe": r"$1/\sqrt{\rm RUWE^2-1}$", "l_plxsnr": r"$\varpi/\sigma_\varpi$"}
    # 1. ratios vs each covariate, one figure per run
    for lab in labs:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
        for ax, key in zip(axs.flat, BINS):
            for k, q in enumerate(("l_sa0", "l_splx", "l_geom", "l_ruwe")):
                t = res["binned"][lab][key][q]
                st = pl.series_style(k, style)
                ax.errorbar(t["x"], t["median"], yerr=t["err"], color=st["color"], marker=st["marker"], ls=st["linestyle"], lw=1.8, capsize=2,
                            label=f"{qlab[q]} mock/DR3")
            ax.axhline(0, color="0.5", lw=1)
            if key == "A":
                ax.set_xscale("log")
            pl.apply_axes_style(ax, style, xlabel=LABEL[key], ylabel="log10 mock/DR3 (same orbits)" if key in ("A", "logp") else None)
        axs[0][0].legend(prop=pl.legend_prop(style), fontsize=8)
        fig.suptitle(f"#390 same-orbit re-injection, {lab}: error and RUWE ratios")
        pl.layout_figure(fig)
        pl.save_figure(fig, a.out_dir / f"same_orbit_ratios_{lab}.png", dpi=150)
        plt.close(fig)
    # 2. decomposition of log z offset vs G and A, all runs
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key in zip(axs, ("g", "A")):
        for i, lab in enumerate(labs):
            for k, q in enumerate(("l_z", "l_a0", "l_sa0", "l_zruwe")):
                t = res["binned"][lab][key][q]
                st = pl.series_style(k, style)
                sign = -1.0 if q == "l_sa0" else 1.0
                ax.errorbar(t["x"], sign * np.array(t["median"]), yerr=t["err"], color=st["color"], marker=st["marker"],
                            ls="-" if i == 0 else "--", lw=1.8, capsize=2, mfc=None if i == 0 else "none",
                            label=f"{lab}: {'log z' if q == 'l_z' else ('−log σa0' if q == 'l_sa0' else ('log a0' if q == 'l_a0' else 'RUWE term'))}")
        ax.axhline(0, color="0.5", lw=1)
        if key == "A":
            ax.set_xscale("log")
        pl.apply_axes_style(ax, style, xlabel=LABEL[key], ylabel="log10 mock/DR3 contribution" if key == "g" else None)
    axs[0].legend(prop=pl.legend_prop(style), fontsize=7, ncol=2)
    pl.layout_figure(fig)
    pl.save_figure(fig, a.out_dir / "z_offset_decomposition.png", dpi=150)
    plt.close(fig)
    # 3. paired before/after plots on the same orbits
    d0 = runs[labs[0]]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.5))
    ax = axs[0][0]
    for i, lab in enumerate(labs):
        t = res["binned"][lab]["bp_rp"]["l_plxsnr"]
        st = pl.series_style(i + 1, style)
        ax.errorbar(t["x"], t["median"], yerr=t["err"], color=st["color"], marker=st["marker"], ls=st["linestyle"], lw=2, capsize=2, label=lab)
    ax.axhline(0, color="0.5", lw=1)
    pl.apply_axes_style(ax, style, xlabel=LABEL["bp_rp"], ylabel=r"log10 $(\varpi/\sigma_\varpi)$ mock/DR3")
    ax.legend(prop=pl.legend_prop(style))
    ax = axs[0][1]
    me = np.arange(-2.0, 8.01, 0.5)
    xc = 0.5 * (me[:-1] + me[1:])
    med = lambda x, y: [np.median(y[(x >= lo) & (x < hi) & np.isfinite(y)]) if np.sum((x >= lo) & (x < hi)) >= 15 else np.nan for lo, hi in zip(me[:-1], me[1:])]
    ax.plot(xc, med(d0["mg"], d0["f2_d"]), color="k", lw=2.5, marker="o", label="DR3")
    for i, lab in enumerate(labs):
        st = pl.series_style(i + 1, style)
        ax.plot(xc, med(runs[lab]["mg"], runs[lab]["f2_m"]), color=st["color"], ls=st["linestyle"], marker=st["marker"], lw=2, label=lab)
    pl.apply_axes_style(ax, style, xlabel=r"$M_G$ (mag; $G+5\log_{10}(\varpi/100)$, no extinction)", ylabel="median $F_2$")
    ax.legend(prop=pl.legend_prop(style))
    ax = axs[1][0]
    ax.scatter(d0["ruwe_d"], d0["snr_d"], s=3, color="k", alpha=0.4, label="DR3", rasterized=True)
    for i, lab in enumerate(labs):
        st = pl.series_style(i + 1, style)
        ax.scatter(runs[lab]["ruwe_m"], runs[lab]["snr_m"], s=3, color=st["color"], alpha=0.35, label=lab, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    pl.apply_axes_style(ax, style, xlabel="RUWE", ylabel=r"$a_0/\sigma_{a_0}$")
    ax.legend(prop=pl.legend_prop(style), markerscale=4)
    ax = axs[1][1]
    for i, lab in enumerate(labs):
        t = res["binned"][lab]["g"]["l_ruwe"]
        st = pl.series_style(i + 1, style)
        ax.errorbar(t["x"], t["median"], yerr=t["err"], color=st["color"], marker=st["marker"], ls=st["linestyle"], lw=2, capsize=2, label=lab)
    ax.axhline(0, color="0.5", lw=1)
    pl.apply_axes_style(ax, style, xlabel="$G$ (mag)", ylabel="log10 RUWE mock/DR3")
    ax.legend(prop=pl.legend_prop(style))
    fig.suptitle("same orbits (#390), before vs after the #572 noise model")
    pl.layout_figure(fig)
    pl.save_figure(fig, a.out_dir / "same_orbit_before_after.png", dpi=150)
    plt.close(fig)
    # 4. population fractional errors
    pop = res["population_fractional_errors"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key in zip(axs, ("A", "g")):
        e = np.array(pop["dr3"][key]["edges"])
        xc = np.sqrt(e[:-1] * np.maximum(e[1:], 1e-3)) if key == "A" else 0.5 * (e[:-1] + e[1:])
        for k, q in enumerate(("fa0", "fplx")):
            st = pl.series_style(k, style)
            ax.plot(xc, pop["dr3"][key][q], color=st["color"], marker="o", lw=2.5, label=f"DR3 {'σa0/a0' if q == 'fa0' else 'σϖ/ϖ'}")
            ax.plot(xc, pop["mock"][key][q], color=st["color"], marker="s", ls="--", lw=2, label=f"mock {'σa0/a0' if q == 'fa0' else 'σϖ/ϖ'}")
        ax.set_yscale("log")
        if key == "A":
            ax.set_xscale("log")
        pl.apply_axes_style(ax, style, xlabel=LABEL[key], ylabel="median fractional error" if key == "A" else None)
    axs[0].legend(prop=pl.legend_prop(style), fontsize=8)
    pl.layout_figure(fig)
    pl.save_figure(fig, a.out_dir / "population_fractional_errors.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
