#!/usr/bin/env python3
"""#573: the a0/sigma_a0 vs RUWE upper edge of accepted orbits, mock (rung-3 v3 luminous gens 42-46) vs DR3, by N and G.

For a single Keplerian photocentre orbit both statistics come from the same signal: with per-CCD noise sigma and
N_obs CCD observations, UWE^2 - 1 ~ kappa a0^2 / sigma^2 and a0 / sigma_a0 ~ sqrt(N_obs kappa') a0 / (c sigma), so at
fixed RUWE the largest a0/sigma_a0 rises as sqrt(N) and depends on u0 (RUWE = UWE / u0) and on the error scale c. The
script measures, in bins of N (visibility periods; N_obs where DR3 publishes it) and G:

* the edge E(RUWE) = q-th weighted percentile of a0/sigma_a0 in log-RUWE bins (mock: fit-free default-target weights
  exp(lw) of the gens 42-46 cache; DR3: all Orbital + AstroSpectroSB1 orbits);
* the normalised edge z = (a0/sigma_a0) / sqrt(RUWE^2 - 1), whose upper percentile removes the RUWE dependence;
* the DR3/mock offset of the normalised edge vs N, and its log-slope (sqrt(N) scaling of the offset = slope 0.5).

Writes ``edge_vs_n.json`` and figures under ``--out-dir``. Light analysis, no simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import plotting as pl


def wq(x: np.ndarray, w: np.ndarray, q: float) -> float:
    ok = np.isfinite(x) & (w > 0)
    if ok.sum() < 20:
        return float("nan")
    o = np.argsort(x[ok])
    cw = np.cumsum(w[ok][o])
    return float(np.interp(q * cw[-1], cw, x[ok][o]))


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / np.sum(w**2)) if w.size and w.sum() > 0 else 0.0


def boot_q(x: np.ndarray, w: np.ndarray, q: float, n: int, rng: np.random.Generator) -> float:
    ok = np.flatnonzero(np.isfinite(x) & (w > 0))
    if ok.size < 20:
        return float("nan")
    v = [wq(x[s], w[s], q) for s in (rng.choice(ok, ok.size) for _ in range(n))]
    return float(np.nanstd(v))


def load(a: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    import h5py
    from astropy.table import Table

    c = np.load(a.cache)
    acc = c["accepted"]
    cas = c["cascade"][acc]
    # deterministic-mixture importance weight at the default target: exp(ll + lw) / sum_j (n_j / N) q_j
    n = c["n_per_gen"].astype(float)
    lq = np.logaddexp.reduce(c["lqs"][:, acc] + np.log(n / n.sum())[:, None], axis=0)
    lt = c["ll"][acc] + c["lw"][acc] - lq
    w = np.where(np.isfinite(lt), np.exp(lt - np.nanmax(lt)), 0.0)
    mock = {"g": c["phot_g_mean_mag"][acc], "snr": cas[:, 17] / cas[:, 18], "ruwe": cas[:, 22], "nvis": cas[:, 19],
            "nobs": cas[:, 20], "w": w}
    en = Table.read(a.enrichment, format="ascii.ecsv")
    en = en[np.isin(np.asarray(en["nss_solution_type"]).astype(str), ["Orbital", "AstroSpectroSB1"])]
    sid = np.asarray(en["SOURCE_ID"], np.int64)
    with h5py.File(Path(a.input_columns) / "columns.h5", "r") as f:
        ic = {k: f[k][()] for k in ("source_id", "phot_g_mean_mag", "ruwe", "visibility_periods_used")}
    o = np.argsort(ic["source_id"])
    j = np.searchsorted(ic["source_id"][o], sid)
    j = np.clip(j, 0, o.size - 1)
    hit = ic["source_id"][o][j] == sid
    jj = o[j]
    real = {"sid": sid, "snr": np.asarray(en["significance"], float),
            "g": np.where(hit, ic["phot_g_mean_mag"][jj], np.nan), "ruwe": np.where(hit, ic["ruwe"][jj], np.nan),
            "nvis": np.where(hit, ic["visibility_periods_used"][jj], np.nan), "w": np.ones(sid.size)}
    with h5py.File(Path(a.epoch_counts) / "nss.h5", "r") as f:
        es, eo = f["source_id"][()], f["astrometric_n_good_obs_al"][()]
    m = dict(zip(es.tolist(), eo.tolist()))
    real["nobs"] = np.array([m.get(int(s), np.nan) for s in sid], float)
    for d in (mock, real):
        sel = (d["w"] > 0) & (d["g"] >= a.g_min) & (d["g"] < a.g_max) & (d["ruwe"] > 1.4) & np.isfinite(d["snr"]) & (d["snr"] > 0)
        for k in list(d):
            d[k] = d[k][sel]
        d["z"] = d["snr"] / np.sqrt(d["ruwe"] ** 2 - 1.0)
    return {"mock": mock, "real": real}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True, help="weights_gen42_46.npz (rung-3 v3 luminous gens)")
    ap.add_argument("--enrichment", required=True, help="nss_enrichment/query.ecsv (significance)")
    ap.add_argument("--input-columns", required=True, help="20261003T182211Z_nss_orbit_input_columns")
    ap.add_argument("--epoch-counts", required=True, help="20261003T063811Z_epoch_counts_400 (N_obs subset)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--q", type=float, default=0.99)
    ap.add_argument("--g-min", type=float, default=9.0)
    ap.add_argument("--g-max", type=float, default=19.0)
    ap.add_argument("--n-boot", type=int, default=50)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(573)
    d = load(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ruwe_e = np.array([1.4, 2.0, 3.0, 5.0, 10.0, 30.0])
    nvis_e = np.array([9, 15, 18, 21, 24, 28, 45])
    nobs_e = np.array([0, 150, 200, 250, 300, 400, 2000])
    g_e = np.array([9.0, 13.0, 15.0, 17.0, 19.0])
    res: dict = {"q": a.q, "n": {k: int(v["snr"].size) for k, v in d.items()},
                 "mock_ess": kish(d["mock"]["w"]), "ruwe_edges": ruwe_e.tolist(), "nvis_edges": nvis_e.tolist(),
                 "nobs_edges": nobs_e.tolist(), "g_edges": g_e.tolist()}

    def table(nkey: str, nedges: np.ndarray, gsel: tuple[float, float] | None) -> dict:
        out: dict = {"N_median": {}, "edge_z": {}, "edge_z_err": {}, "ess": {}, "edge_by_ruwe": {}}
        for side, s in d.items():
            ok = np.isfinite(s[nkey])
            if gsel is not None:
                ok &= (s["g"] >= gsel[0]) & (s["g"] < gsel[1])
            med, ez, ee, es, eb = [], [], [], [], []
            for lo, hi in zip(nedges[:-1], nedges[1:]):
                b = ok & (s[nkey] >= lo) & (s[nkey] < hi)
                med.append(float(np.median(s[nkey][b])) if b.any() else float("nan"))
                ez.append(wq(s["z"][b], s["w"][b], a.q))
                ee.append(boot_q(s["z"][b], s["w"][b], a.q, a.n_boot, rng))
                es.append(kish(s["w"][b]))
                eb.append([wq(s["snr"][b & (s["ruwe"] >= r0) & (s["ruwe"] < r1)], s["w"][b & (s["ruwe"] >= r0) & (s["ruwe"] < r1)], a.q)
                           for r0, r1 in zip(ruwe_e[:-1], ruwe_e[1:])])
            out["N_median"][side], out["edge_z"][side], out["edge_z_err"][side] = med, ez, ee
            out["ess"][side], out["edge_by_ruwe"][side] = es, eb
        rz = np.log10(np.array(out["edge_z"]["real"]) / np.array(out["edge_z"]["mock"]))
        rze = np.hypot(np.array(out["edge_z_err"]["real"]) / np.array(out["edge_z"]["real"]),
                       np.array(out["edge_z_err"]["mock"]) / np.array(out["edge_z"]["mock"])) / np.log(10)
        out["log10_offset_real_over_mock"], out["log10_offset_err"] = rz.tolist(), rze.tolist()
        # slopes vs log10 N: of each side's edge, and of the offset (sqrt(N) scaling -> 0.5)
        for side in ("mock", "real"):
            x, y = np.log10(out["N_median"][side]), np.log10(out["edge_z"][side])
            ey = np.array(out["edge_z_err"][side]) / np.array(out["edge_z"][side]) / np.log(10)
            ok = np.isfinite(x) & np.isfinite(y) & (ey > 0)
            if ok.sum() >= 3:
                p, cov = np.polyfit(x[ok], y[ok], 1, w=1 / ey[ok], cov="unscaled")
                out[f"slope_{side}"], out[f"slope_{side}_err"] = float(p[0]), float(np.sqrt(cov[0, 0]))
        x = np.log10(np.sqrt(np.array(out["N_median"]["real"]) * np.array(out["N_median"]["mock"])))
        ok = np.isfinite(x) & np.isfinite(rz) & (rze > 0)
        if ok.sum() >= 3:
            p, cov = np.polyfit(x[ok], rz[ok], 1, w=1 / rze[ok], cov="unscaled")
            out["slope_offset"], out["slope_offset_err"] = float(p[0]), float(np.sqrt(cov[0, 0]))
            out["offset_chi2_const"] = float(np.sum(((rz[ok] - np.average(rz[ok], weights=rze[ok] ** -2)) / rze[ok]) ** 2))
            out["offset_dof_const"] = int(ok.sum() - 1)
        return out

    res["by_nvis"] = table("nvis", nvis_e, None)
    res["by_nobs"] = table("nobs", nobs_e, None)
    res["by_nvis_g"] = {f"G{lo:g}-{hi:g}": table("nvis", nvis_e, (lo, hi)) for lo, hi in zip(g_e[:-1], g_e[1:])}
    res["by_nobs_g"] = {f"G{lo:g}-{hi:g}": table("nobs", nobs_e, (lo, hi)) for lo, hi in zip(g_e[:-1], g_e[1:])}
    # N distributions themselves (medians by G): does DR3 have more observations per orbit than the mock?
    res["N_by_g"] = {side: {k: [float(np.nanmedian(s[k][(s["g"] >= lo) & (s["g"] < hi)])) for lo, hi in zip(g_e[:-1], g_e[1:])]
                            for k in ("nvis", "nobs")} for side, s in d.items()}
    (a.out_dir / "edge_vs_n.json").write_text(json.dumps(res, indent=1, default=float))
    plots(d, res, a)
    return 0


def plots(d: dict, res: dict, a: argparse.Namespace) -> None:
    plt = pl.require_pyplot()
    style = pl.default_plotting_style()
    # 1. scatter of a0/sigma vs RUWE in N_vis bins, with the edge curves
    ne = res["nvis_edges"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    re = np.array(res["ruwe_edges"])
    rc = np.sqrt(re[:-1] * re[1:])
    for k, ax in enumerate(axs.flat):
        lo, hi = ne[k], ne[k + 1]
        for i, side in enumerate(("real", "mock")):
            s = d[side]
            b = (s["nvis"] >= lo) & (s["nvis"] < hi)
            st = pl.series_style(i, style)
            if side == "real":
                ax.scatter(s["ruwe"][b], s["snr"][b], s=1, color=st["color"], alpha=0.15, rasterized=True)
            else:
                ax.scatter(s["ruwe"][b], s["snr"][b], s=np.clip(40 * s["w"][b] / s["w"][b].max(), 0.5, 6), color=st["color"],
                           alpha=0.3, rasterized=True)
            ax.plot(rc, res["by_nvis"]["edge_by_ruwe"][side][k], color=st["color"], lw=st["linewidth"], ls=st["linestyle"],
                    label=f"{'DR3' if side == 'real' else 'mock'} {int(100 * a.q)}th pct")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.apply_axes_style(ax, style, xlabel="RUWE" if k >= 3 else None, ylabel=r"$a_0/\sigma_{a_0}$" if k % 3 == 0 else None,
                            title=f"$N_{{\\rm vis}}$ {lo}–{hi - 1}")
        if k == 0:
            ax.legend(prop=pl.legend_prop(style))
    pl.layout_figure(fig)
    pl.save_figure(fig, a.out_dir / "edge_scatter_by_nvis.png", dpi=150)
    plt.close(fig)
    # 2. normalised edge vs N (vis and obs), both sides, and the offset
    for key, lab in (("nvis", r"$N_{\rm vis}$"), ("nobs", r"$N_{\rm obs}$ (CCD)")):
        t = res[f"by_{key}"]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for i, side in enumerate(("real", "mock")):
            st = pl.series_style(i, style)
            a1.errorbar(t["N_median"][side], t["edge_z"][side], yerr=t["edge_z_err"][side], color=st["color"], marker=st["marker"],
                        lw=st["linewidth"], ls=st["linestyle"], capsize=3,
                        label=f"{'DR3' if side == 'real' else 'mock'}: slope {t.get(f'slope_{side}', np.nan):.2f} ± {t.get(f'slope_{side}_err', np.nan):.2f}")
        nn = np.array([x for x in t["N_median"]["mock"] if np.isfinite(x)])
        if nn.size:
            ref = np.nanmedian(t["edge_z"]["mock"]) * np.sqrt(nn / np.median(nn))
            a1.plot(nn, ref, color="0.5", ls=":", lw=1.5, label=r"$\propto\sqrt{N}$")
        a1.set_xscale("log")
        a1.set_yscale("log")
        pl.apply_axes_style(a1, style, ylabel=rf"{int(100 * a.q)}th pct of $(a_0/\sigma_{{a_0}})/\sqrt{{\rm RUWE^2-1}}$")
        a1.legend(prop=pl.legend_prop(style))
        xm = np.sqrt(np.array(t["N_median"]["real"]) * np.array(t["N_median"]["mock"]))
        a2.errorbar(xm, t["log10_offset_real_over_mock"], yerr=t["log10_offset_err"], color="k", marker="o", capsize=3, lw=2,
                    label=f"slope {t.get('slope_offset', np.nan):.2f} ± {t.get('slope_offset_err', np.nan):.2f}")
        a2.axhline(0, color="0.5", lw=1)
        pl.apply_axes_style(a2, style, xlabel=lab, ylabel="log10 DR3/mock")
        a2.legend(prop=pl.legend_prop(style))
        pl.layout_figure(fig)
        pl.save_figure(fig, a.out_dir / f"edge_norm_vs_{key}.png", dpi=150)
        plt.close(fig)
    # 3. offset vs N per G bin
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (gk, t) in enumerate(res["by_nvis_g"].items()):
        st = pl.series_style(i, style)
        xm = np.sqrt(np.array(t["N_median"]["real"]) * np.array(t["N_median"]["mock"]))
        ax.errorbar(xm, t["log10_offset_real_over_mock"], yerr=t["log10_offset_err"], color=st["color"], marker=st["marker"],
                    ls=st["linestyle"], lw=st["linewidth"], capsize=3, label=gk.replace("G", "$G$ ") + f" (slope {t.get('slope_offset', np.nan):.2f})")
    ax.axhline(0, color="0.5", lw=1)
    pl.apply_axes_style(ax, style, xlabel=r"$N_{\rm vis}$", ylabel="log10 edge DR3/mock")
    ax.legend(prop=pl.legend_prop(style))
    pl.layout_figure(fig)
    pl.save_figure(fig, a.out_dir / "edge_offset_vs_nvis_by_g.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
