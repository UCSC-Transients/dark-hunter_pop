#!/usr/bin/env python3
"""#589: the a0/σ_a0–RUWE upper edge under colour_v2.1+tr1, against DR3's per-row covariances.

Three blocks (analysis only, no simulation):

1. **DR3 covariance (all published Orbital + AstroSpectroSB1).** 12×12 (15×15) covariances from ``corr_vec`` /
   ``bit_index`` (``nss_covariance.reconstruct_nss_covariance``). Per row:
   * σ_a0 by first-order (Halbwachs et al. 2023) propagation of the Thiele–Innes block, exactly as gaiamock_mod
     does it, against DR3's own σ_a0 = a0 / significance. This tests the propagation formula on DR3's covariance.
   * The share of σ_a0² that comes through the coupling with the non-linear parameters (e, P, T_p):
     f_nl = 1 − (tᵀ C_TI|nl t) / (tᵀ C_TI t), with C_TI|nl the Schur complement.
2. **Same orbits** (#390 published orbits re-injected under colour_v2.1+tr1, ``reinject_v21tr1``). Mock/DR3 ratios of
   σ_a0, σ_ϖ, σ_P, σ_e, RUWE and the geometry ratio σ_a0/σ_ϖ, by A = (a0/ϖ)/P_yr^(2/3) and G. The coupling
   prediction for the geometry excess uses each orbit's DR3 f_nl and the mock's relative inflation of σ_P, σ_e:
   l_geom,pred = ½ log10(1 − f_nl + f_nl r²), r = √(σ_P,m σ_e,m / σ_P,d σ_e,d) / (σ_ϖ,m / σ_ϖ,d).
3. **Population** (gens 47–50, default-target mixture weights, accepted orbits) against DR3 at fixed G / log P / A:
   the edge z = (a0/σ_a0)/√(RUWE² − 1) (99th percentile), σ_a0/σ_ϖ, σ_TI/σ_ϖ (mean of the four Thiele–Innes
   errors) and σ_a0/σ_TI (propagation incl. cross terms).

Writes ``edge.json`` and figures to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from darkhunter_pop import nss_covariance as nc
from darkhunter_pop import plotting as pl
from darkhunter_pop.config_loader import load_config

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
TI = ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")
NL = ("eccentricity", "period", "t_periastron")
A_EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 5.0])
G_EDGES = np.array([6.0, 10.5, 11.5, 12.5, 13.0, 14.0, 15.0, 17.0, 19.5])
P_EDGES = np.array([0.0, 1.5, 2.0, 2.5, 2.75, 3.0, 4.0])


def halbwachs_t(A, B, F, G):
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    s = np.sqrt(np.clip(u**2 - v**2, 0, None))
    a0 = np.sqrt(u + s)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.stack([A + (A * u - G * v) / s, B + (B * u + F * v) / s, F + (F * u + B * v) / s, G + (G * u - A * v) / s], axis=-1)
    return a0, t


def dr3_covariance_block(pub: dict) -> dict:
    """Per published Orbital / AstroSpectroSB1 row: Halbwachs σ_a0 from the full covariance, f_nl, σ_ϖ."""
    n = pub["source_id"].size
    out = {k: np.full(n, np.nan) for k in ("sa0_lin", "sa0_cond", "f_nl", "corr_plx_ti")}
    status = np.zeros(n, int)
    for i in range(n):
        st = pub["nss_solution_type"][i]
        if st not in ("Orbital", "AstroSpectroSB1"):
            continue
        cv = pub["corr_vec"][i]
        cv = cv[np.isfinite(cv)]
        row = {k: pub[k][i] for k in pub if k not in ("corr_vec",)}
        row["corr_vec"] = cv
        row["nss_solution_type"] = st
        r = nc.reconstruct_nss_covariance(row, nss_solution_type=st)
        if r.parameter_set is None:
            status[i] = 1
            continue
        names = list(r.parameter_set.names)
        if not all(x in names for x in TI + NL + ("parallax",)):
            status[i] = 2
            continue
        C = np.asarray(r.parameter_set.covariance, float)
        it = [names.index(x) for x in TI]
        inl = [names.index(x) for x in NL]
        _, t = halbwachs_t(*(pub[x][i] for x in TI))
        a0, _ = halbwachs_t(*(pub[x][i] for x in TI))
        Ctt = C[np.ix_(it, it)]
        Ctn = C[np.ix_(it, inl)]
        Cnn = C[np.ix_(inl, inl)]
        try:
            Cc = Ctt - Ctn @ np.linalg.solve(Cnn, Ctn.T)
        except np.linalg.LinAlgError:
            status[i] = 3
            continue
        m, c = float(t @ Ctt @ t), float(t @ Cc @ t)
        out["sa0_lin"][i] = np.sqrt(m) / (2 * a0) if m > 0 else np.nan
        out["sa0_cond"][i] = np.sqrt(c) / (2 * a0) if c > 0 else np.nan
        out["f_nl"][i] = 1.0 - c / m if m > 0 else np.nan
        ip = names.index("parallax")
        sd = np.sqrt(np.diag(C))
        out["corr_plx_ti"][i] = float(np.sqrt(np.mean((C[ip, it] / (sd[ip] * sd[it])) ** 2)))
    out["status"] = status
    return out


def wquant(x, w, q):
    ok = np.isfinite(x) & (w > 0)
    if ok.sum() < 10:
        return float("nan")
    o = np.argsort(x[ok])
    c = np.cumsum(w[ok][o]) / w[ok].sum()
    return float(x[ok][o][min(np.searchsorted(c, q), ok.sum() - 1)])


def binstat(x, y, edges, w=None, q=0.5, nmin=15, ess_min=15):
    w = np.ones(x.size) if w is None else w
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (x >= lo) & (x < hi) & np.isfinite(y) & (w > 0)
        ess = float(w[b].sum() ** 2 / (w[b] ** 2).sum()) if b.any() else 0.0
        res.append({"lo": float(lo), "hi": float(hi), "n": int(b.sum()), "ess": ess,
                    "value": wquant(y[b], w[b], q) if (b.sum() >= nmin and ess >= ess_min) else float("nan")})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published", type=Path, default=D / "output/gate390/published_orbits.h5")
    ap.add_argument("--reinject", type=Path, default=D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl")
    ap.add_argument("--cache", type=Path, default=D / "output/proposal_set/v21tr1/weights_gen47_50.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    import h5py

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.published, "r") as f:
        P_ = f["published"]
        pub = {k: P_[k][()] for k in P_}
    pub["nss_solution_type"] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in pub["nss_solution_type"]])
    res = {}
    # 1. DR3 covariance
    cb = dr3_covariance_block(pub)
    sel = np.isin(pub["nss_solution_type"], ["Orbital", "AstroSpectroSB1"])
    a0_d, _ = halbwachs_t(*(pub[x] for x in TI))
    sa0_pub = a0_d / pub["significance"]
    A_d = (a0_d / pub["parallax"]) / (pub["period"] / 365.25) ** (2 / 3)
    lr = np.log10(cb["sa0_lin"] / sa0_pub)
    res["dr3_cov"] = {"n_rows": int(sel.sum()), "n_ok": int(np.isfinite(cb["sa0_lin"]).sum()), "status_counts": np.bincount(cb["status"][sel]).tolist(),
                      "log_sa0_lin_over_published_median": float(np.nanmedian(lr[sel])),
                      "log_sa0_lin_over_published_p16_p84": [float(np.nanpercentile(lr[sel], 16)), float(np.nanpercentile(lr[sel], 84))],
                      "by_A": {"lin_over_pub": binstat(A_d[sel], lr[sel], A_EDGES), "f_nl": binstat(A_d[sel], cb["f_nl"][sel], A_EDGES),
                               "sa0_over_splx": binstat(A_d[sel], np.log10(sa0_pub / pub["parallax_error"])[sel], A_EDGES)},
                      "by_G": {"lin_over_pub": binstat(pub["g_mag"][sel], lr[sel], G_EDGES), "f_nl": binstat(pub["g_mag"][sel], cb["f_nl"][sel], G_EDGES)}}
    print("DR3 covariance block done", flush=True)
    # 2. same orbits
    ix = {int(s): i for i, s in enumerate(pub["source_id"])}
    rows = [json.loads(x) for x in a.reinject.read_text().splitlines()]
    acc = [r for r in rows if r.get("accepted") and int(r["source_id"]) in ix]
    i = np.array([ix[int(r["source_id"])] for r in acc])
    gm = lambda k: np.array([r[k] for r in acc], float)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        so = {"A": A_d[i], "g": pub["g_mag"][i], "logp": np.log10(pub["period"][i]),
              "l_sa0": np.log10(gm("sigma_a0_mas") / sa0_pub[i]), "l_splx": np.log10(gm("parallax_error") / pub["parallax_error"][i]),
              "l_sP": np.log10(gm("period_error") / pub["period_error"][i]), "l_se": np.log10(gm("eccentricity_error") / pub["eccentricity_error"][i]),
              "l_ruwe": np.log10(gm("ruwe") / pub["ruwe"][i]), "l_a0": np.log10(gm("a0_mas") / a0_d[i]),
              "l_zruwe": -0.5 * np.log10((gm("ruwe") ** 2 - 1) / (pub["ruwe"][i] ** 2 - 1)), "f_nl": cb["f_nl"][i]}
        so["l_geom"] = so["l_sa0"] - so["l_splx"]
        so["l_z"] = so["l_a0"] - so["l_sa0"] + so["l_zruwe"]
        r_rel = 10 ** (0.5 * (so["l_sP"] + so["l_se"]) - so["l_splx"])
        so["l_geom_pred"] = 0.5 * np.log10(1 - so["f_nl"] + so["f_nl"] * r_rel**2)
        so["l_nl_rel"] = 0.5 * (so["l_sP"] + so["l_se"]) - so["l_splx"]
    res["same_orbit"] = {"n_accepted_realizations": len(acc), "by_A": {k: binstat(so["A"], so[k], A_EDGES) for k in
                                                                          ("l_z", "l_sa0", "l_splx", "l_geom", "l_geom_pred", "l_sP", "l_se", "l_nl_rel", "l_ruwe", "l_zruwe", "f_nl")},
                         "by_G": {k: binstat(so["g"], so[k], G_EDGES) for k in ("l_z", "l_sa0", "l_splx", "l_geom", "l_geom_pred", "l_sP", "l_se", "l_nl_rel", "l_ruwe", "f_nl")}}
    print("same-orbit block done", flush=True)
    # 3. population
    c = np.load(a.cache)
    accm = np.asarray(c["accepted"], bool)
    cas = np.asarray(c["cascade"], float)[accm]
    nn = c["n_per_gen"].astype(float)
    lq = np.logaddexp.reduce(c["lqs"][:, accm] + np.log(nn / nn.sum())[:, None], axis=0)
    lt = c["ll"][accm] + c["lw"][accm] - lq
    w = np.where(np.isfinite(lt), np.exp(lt - np.nanmax(lt)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = {"g": np.asarray(c["phot_g_mean_mag"], float)[accm], "logp": np.log10(cas[:, 10]), "A": (cas[:, 17] / cas[:, 0]) / (cas[:, 10] / 365.25) ** (2 / 3),
             "z": (cas[:, 17] / cas[:, 18]) / np.sqrt(cas[:, 22] ** 2 - 1), "geom": np.log10(cas[:, 18] / cas[:, 1]),
             "ti": np.log10(np.mean(cas[:, [3, 5, 7, 9]], axis=1) / cas[:, 1]), "prop": np.log10(cas[:, 18] / np.mean(cas[:, [3, 5, 7, 9]], axis=1)),
             "ruwe": cas[:, 22], "w": w}
        sti = np.mean(np.stack([pub[x + "_error"] for x in TI]), axis=0)
        r = {"g": pub["g_mag"], "logp": np.log10(pub["period"]), "A": A_d, "z": pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1),
             "geom": np.log10(sa0_pub / pub["parallax_error"]), "ti": np.log10(sti / pub["parallax_error"]), "prop": np.log10(sa0_pub / sti),
             "ruwe": pub["ruwe"], "w": sel.astype(float)}
    for s_ in (m, r):
        s_["w"] = s_["w"] * ((s_["ruwe"] > 1.4) & (s_["g"] >= 9) & (s_["g"] < 19) & (s_["logp"] <= np.log10(1000)))
    pop = {}
    for key, edges in (("A", A_EDGES), ("g", G_EDGES), ("logp", P_EDGES)):
        pop[key] = {}
        for q in ("z", "geom", "ti", "prop"):
            qq = 0.99 if q == "z" else 0.5
            mv, rv = binstat(m[key], np.log10(m[q]) if q == "z" else m[q], edges, m["w"], qq), binstat(r[key], np.log10(r[q]) if q == "z" else r[q], edges, r["w"], qq)
            pop[key][q] = {"mock": mv, "dr3": rv, "dr3_minus_mock": [x["value"] - y["value"] for x, y in zip(rv, mv)]}
    # fixed G x A cells for the edge (bright vs faint, small vs large A)
    cells = {}
    for gl, gh in ((9, 13), (13, 15), (15, 19)):
        for al, ah in ((0, 0.2), (0.2, 0.5), (0.5, 5)):
            bm = (m["g"] >= gl) & (m["g"] < gh) & (m["A"] >= al) & (m["A"] < ah)
            br = (r["g"] >= gl) & (r["g"] < gh) & (r["A"] >= al) & (r["A"] < ah)
            ess = float(m["w"][bm].sum() ** 2 / (m["w"][bm] ** 2).sum()) if (m["w"][bm] > 0).any() else 0.0
            cells[f"G{gl}-{gh} A{al}-{ah}"] = {
                "edge_dr3_minus_mock": wquant(np.log10(r["z"][br]), r["w"][br], 0.99) - wquant(np.log10(m["z"][bm]), m["w"][bm], 0.99),
                "geom_dr3_minus_mock": wquant(r["geom"][br], r["w"][br], 0.5) - wquant(m["geom"][bm], m["w"][bm], 0.5),
                "ti_dr3_minus_mock": wquant(r["ti"][br], r["w"][br], 0.5) - wquant(m["ti"][bm], m["w"][bm], 0.5),
                "prop_dr3_minus_mock": wquant(r["prop"][br], r["w"][br], 0.5) - wquant(m["prop"][bm], m["w"][bm], 0.5),
                "mock_ess": ess, "n_dr3": int((r["w"][br] > 0).sum())}
    res["population"] = {"by": pop, "cells_G_A": cells, "mock_ess_total": float(m["w"].sum() ** 2 / (m["w"] ** 2).sum())}
    (out / "edge.json").write_text(json.dumps(res, indent=1, default=float))
    # figure
    cfg = load_config()
    style = cfg.plotting
    plt = pl.require_pyplot()
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    xa = 0.5 * (A_EDGES[1:] + A_EDGES[:-1])
    for j, k in enumerate(("l_z", "l_sa0", "l_splx", "l_geom", "l_geom_pred", "l_nl_rel", "l_ruwe")):
        st = pl.series_style(j, style)
        axs[0].plot(np.arange(xa.size), [b["value"] for b in res["same_orbit"]["by_A"][k]], marker="o", color=st["color"], ls=st["linestyle"], label=k)
    axs[0].axhline(0, color="0.5", lw=0.8)
    axs[0].set_xticks(np.arange(xa.size), [f"{lo:g}–{hi:g}" for lo, hi in zip(A_EDGES[:-1], A_EDGES[1:])], rotation=30)
    pl.apply_axes_style(axs[0], style, xlabel="A bin", ylabel="log10 mock / DR3 (same orbit)", title="same orbits, colour_v2.1+tr1")
    axs[0].legend(fontsize=8, frameon=False)
    for j, k in enumerate(("lin_over_pub", "f_nl")):
        st = pl.series_style(j, style)
        axs[1].plot(np.arange(xa.size), [b["value"] for b in res["dr3_cov"]["by_A"][k]], marker="o", color=st["color"], label=k)
    axs[1].set_xticks(np.arange(xa.size), [f"{lo:g}–{hi:g}" for lo, hi in zip(A_EDGES[:-1], A_EDGES[1:])], rotation=30)
    pl.apply_axes_style(axs[1], style, xlabel="A bin", ylabel="value", title="DR3 covariance: log σ_a0,lin/σ_a0,pub; f_nl")
    axs[1].legend(fontsize=8, frameon=False)
    for j, q in enumerate(("z", "geom", "ti", "prop")):
        st = pl.series_style(j, style)
        axs[2].plot(np.arange(xa.size), pop["A"][q]["dr3_minus_mock"], marker="o", color=st["color"], ls=st["linestyle"], label=q)
    axs[2].axhline(0, color="0.5", lw=0.8)
    axs[2].set_xticks(np.arange(xa.size), [f"{lo:g}–{hi:g}" for lo, hi in zip(A_EDGES[:-1], A_EDGES[1:])], rotation=30)
    pl.apply_axes_style(axs[2], style, xlabel="A bin", ylabel="log10 DR3 − mock", title="population (gens 47–50 vs DR3, RUWE > 1.4)")
    axs[2].legend(fontsize=8, frameon=False)
    pl.save_figure(fig, out / "edge_diagnostics.png", dpi=110)
    print(json.dumps({k: res[k] if k != "population" else {"cells_G_A": res[k]["cells_G_A"]} for k in ("dr3_cov",)}, default=float)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
