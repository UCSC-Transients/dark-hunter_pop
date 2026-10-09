#!/usr/bin/env python3
"""Orbit : acceleration ratio, mock vs real (#391 / #402; Ryan, 2026-10-09). Analysis only.

The symmetric rung-1 test (``docs/gate391/detection_vs_population/``) found that real published
orbits, re-injected, fall to a 7/9-parameter (acceleration) solution 2-4x as often as the mock's
own orbits: a detection factor of about 0.8. This script tests whether that is gaiamock's
orbit-vs-acceleration decision rule, by comparing published-acceleration and accepted-orbit counts.

* **Real.**
  - Orbits: mirror-filtered DR3 Orbital + AstroSpectroSB1, as in rung 2.
  - Accelerations: a ``gaiadr3.nss_acceleration_astro`` snapshot (Acceleration7 / Acceleration9),
    with the *same* mirror filters (parallax floor, atmosphere, Halbwachs IPD / C* via
    ``real_comparison_keep``).
  - Distance is the zero-point-corrected 1/parallax from the NSS solution's parallax, for both.
* **Mock.** Gens 23-27 deterministic-mixture weights (noW and cmdW).
  - Orbits are ``accepted_orbital``.
  - Accelerations are ``published_acceleration``: a 7/9-parameter cascade outcome that passes the
    DR3 acceleration publication cuts.
  - Distance is 1/(fitted parallax) of the respective solution.
* **C1** (P <= 830 d) is applied to orbits on both sides (published P for real, fitted P for mock);
  accelerations have no period.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASELINE = 1038.0
P_MAX = 0.8 * BASELINE
GEN_ARTIFACTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5",
                 "restart_gen26_topup.h5", "restart_gen27_evolved.h5")
HALBWACHS_INPUT = ("phot_g_mean_mag", "bp_rp", "phot_bp_rp_excess_factor", "ipd_frac_multi_peak",
                   "ipd_gof_harmonic_amplitude", "ruwe", "visibility_periods_used")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--accel-snapshot", type=Path, required=True, help="<..>_nss_acceleration_astro directory")
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import yaml
    from astropy.table import Table

    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config()
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- mock ----------------
    z = np.load(args.cache, allow_pickle=True)
    casc = np.asarray(z["cascade"], float)
    acc = np.asarray(z["accepted"], bool)
    pub_acc = np.concatenate([np.asarray(ps.read_proposal_artifact(args.artifact_dir / a)[1]["published_acceleration"], bool)
                              for a in GEN_ARTIFACTS])
    if pub_acc.size != acc.size:
        raise SystemExit("artifact / cache length mismatch")
    st = np.asarray(z["solution_type"]).astype(str)
    W = {"noW": np.asarray(z["w_noW"], float), "cmdW": np.asarray(z["w_cmdW"], float)}
    m_g = np.asarray(z["phot_g_mean_mag"], float)
    m_Ptrue = np.asarray(z["period_days"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_d_orb = 1.0 / casc[:, 0]
        m_d_acc = 1.0 / casc[:, 2]
    m_orb = acc & (casc[:, 10] <= P_MAX)
    m_orb_all = acc
    m_acc = pub_acc
    m_d = np.where(m_acc, m_d_acc, m_d_orb)

    # ---------------- real ----------------
    pc = parent.columns
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr_p > 20)]) for a, b in zip(gedges[:-1], gedges[1:])])

    def zp_d(plx: np.ndarray, g: np.ndarray) -> np.ndarray:
        p = plx + zp[np.clip(np.digitize(g, gedges) - 1, 0, zp.size - 1)]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, 1.0 / p, np.nan)

    def col(t, name):
        return np.ma.filled(np.ma.asarray(t[name], float), np.nan)

    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, cnt_o = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]
    ro_g, ro_P = col(rt, "g_mag"), col(rt, "period")
    ro_d = zp_d(col(rt, "parallax"), ro_g)
    ro_c1 = ro_P <= P_MAX

    at = Table.read(args.accel_snapshot / "table.h5", path="data")
    ameta = yaml.safe_load((args.accel_snapshot / "meta.yaml").read_text())
    inp = {"source_id": np.asarray(at["source_id"], np.int64)}
    inp.update({c: col(at, c) for c in HALBWACHS_INPUT})
    keep_a, cnt_a = ps.real_comparison_keep(at, prop, inp)
    at = at[keep_a]
    ra_g = col(at, "g_mag")
    ra_d = zp_d(col(at, "parallax"), ra_g)
    ra_type = np.asarray(at["nss_solution_type"]).astype(str)

    rep = ["#391 / #402 orbit : acceleration ratio, mock (gens 23-27) vs real DR3", "",
           f"real accelerations: snapshot {args.accel_snapshot.name} ({ameta['n_rows']} rows; "
           f"{ameta['n_by_type']}); mirror filters {cnt_a} -> {len(at)}",
           f"real orbits: mirror filters -> {len(rt)} ({int(ro_c1.sum())} with P <= 830 d)", ""]

    def summarize(label: str, ro_sel, ra_sel, mo_sel, ma_sel, mo_all_sel) -> dict:
        n_ro, n_ra = int(ro_sel.sum()), int(ra_sel.sum())
        out = {"real_orbits_c1": n_ro, "real_accel": n_ra, "real_ratio": n_ro / max(n_ra, 1)}
        line = [f"{label}: real orbits (C1) {n_ro}, accelerations {n_ra} -> orbit/accel {n_ro / max(n_ra, 1):.3f}"]
        for k, w in W.items():
            mo, ma = float(w[mo_sel].sum()), float(w[ma_sel].sum())
            ess_o, ess_a = ps.kish_ess(w[mo_sel]), ps.kish_ess(w[ma_sel])
            out[k] = {"mock_orbits_c1": mo, "mock_accel": ma, "mock_ratio": mo / ma if ma > 0 else np.nan,
                      "orbit_mock_over_real": mo / max(n_ro, 1), "accel_mock_over_real": ma / max(n_ra, 1),
                      "ess_orbits": ess_o, "ess_accel": ess_a}
            line.append(f"    mock {k}: orbits {mo:.0f} (ESS {ess_o:.0f}), accelerations {ma:.0f} (ESS {ess_a:.0f}) -> orbit/accel "
                        f"{mo / ma:.3f}; mock/real orbits {mo / max(n_ro, 1):.3f}, accelerations {ma / max(n_ra, 1):.3f}; "
                        f"ratio of ratios {(mo / ma) / (n_ro / max(n_ra, 1)):.3f}")
        rep.extend(line)
        return out

    all_m = np.ones(acc.size, bool)
    summ = {"overall": summarize("overall", ro_c1, np.ones(len(at), bool), m_orb, m_acc, m_orb_all)}
    for ty in ("Acceleration7", "Acceleration9"):
        sol = "seven_parameter" if ty.endswith("7") else "nine_parameter"
        n_r = int(np.sum(ra_type == ty))
        rep.append(f"    {ty}: real {n_r}; mock noW {W['noW'][m_acc & (st == sol)].sum():.0f}, cmdW {W['cmdW'][m_acc & (st == sol)].sum():.0f}")
    rbin_o = ro_c1 & (ro_d >= 0.7) & (ro_d <= 1.5) & (ro_g >= 13) & (ro_g <= 16)
    rbin_a = (ra_d >= 0.7) & (ra_d <= 1.5) & (ra_g >= 13) & (ra_g <= 16)
    mbin = (m_g >= 13) & (m_g <= 16)
    mbin_o = m_orb & mbin & (m_d_orb >= 0.7) & (m_d_orb <= 1.5)
    mbin_a = m_acc & mbin & (m_d_acc >= 0.7) & (m_d_acc <= 1.5)
    rep.append("")
    summ["bin_0p7_1p5kpc_G13_16"] = summarize("bin 0.7-1.5 kpc, G 13-16", rbin_o, rbin_a, mbin_o, mbin_a, None)

    # ---------------- profiles ----------------
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    prof = {"G (mag)": (ro_g, ra_g, m_g, m_g, np.arange(6, 19.1, 1.0)),
            "distance (kpc; ZP-corrected real, fitted mock)": (ro_d, ra_d, m_d_orb, m_d_acc,
                                                                np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]))}
    fig, axs = plt.subplots(2, 3, figsize=(19, 11))
    rep += ["", "profiles: bin centre: real orbit/accel | mock noW orbit/accel | mock/real orbits | mock/real accels  [real orbits, real accels]"]
    for row, (name, (rov, rav, mov, mav, edges)) in enumerate(prof.items()):
        cen = 0.5 * (edges[1:] + edges[:-1])
        ro_n = np.histogram(rov[ro_c1], bins=edges)[0].astype(float)
        ra_n = np.histogram(rav, bins=edges)[0].astype(float)
        cols = {}
        for k, w in W.items():
            mo = np.histogram(mov[m_orb], bins=edges, weights=w[m_orb])[0]
            ma = np.histogram(mav[m_acc], bins=edges, weights=w[m_acc])[0]
            mo2 = np.histogram(mov[m_orb], bins=edges, weights=w[m_orb] ** 2)[0]
            ma2 = np.histogram(mav[m_acc], bins=edges, weights=w[m_acc] ** 2)[0]
            cols[k] = (mo, ma, mo2, ma2)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_real = ro_n / ra_n
            ax = axs[row, 0]
            ax.errorbar(cen, r_real, yerr=r_real * np.sqrt(1 / ro_n + 1 / ra_n), color="k", marker="s", lw=2, label="real")
            for j, (k, (mo, ma, mo2, ma2)) in enumerate(cols.items()):
                rm = mo / ma
                st_ = series_style(j + 1, style)
                ax.errorbar(cen, rm, yerr=rm * np.sqrt(mo2 / mo**2 + ma2 / ma**2), color=st_["color"], ls=st_["linestyle"],
                            marker="o", lw=2, label=f"mock {k}")
            ax.set_yscale("log")
            apply_axes_style(ax, style, xlabel=name, ylabel="orbits (C1) / accelerations")
            ax.legend(fontsize=style.tick_label_fontsize)
            for c_i, (lab, num, den, idx) in enumerate((("orbits", 0, ro_n, 1), ("accelerations", 1, ra_n, 2))):
                ax = axs[row, idx]
                for j, (k, v) in enumerate(cols.items()):
                    st_ = series_style(j + 1, style)
                    ax.errorbar(cen, v[num] / den, yerr=np.sqrt(v[num + 2]) / den, color=st_["color"], ls=st_["linestyle"],
                                marker="o", lw=2, label=f"mock {k} / real")
                ax.axhline(1, color="0.4", ls=":")
                ax.set_ylim(0, 2.5)
                apply_axes_style(ax, style, xlabel=name, ylabel=f"mock / real {lab}")
                ax.legend(fontsize=style.tick_label_fontsize)
            mo, ma = cols["noW"][0], cols["noW"][1]
            rep.append(f"  {name}: " + " ".join(f"{c:.2f}:{a:.2f}|{b:.2f}|{o:.2f}|{q:.2f}[{int(x)},{int(y)}]" for c, a, b, o, q, x, y in
                                                zip(cen, r_real, mo / ma, mo / ro_n, ma / ra_n, ro_n, ra_n)))
    fig.suptitle("Orbit : acceleration, mock (gens 23-27) vs real DR3 (mirror filters; orbits with P <= 830 d)",
                 fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "orbit_accel_profiles.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ---------------- mock by true P ----------------
    pedges = np.arange(0.0, 8.01, 0.25)
    lp = np.log10(m_Ptrue)
    cen = 0.5 * (pedges[1:] + pedges[:-1])
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    rep += ["", "mock (noW) by true log10 P: all G, all d | in the bin (true d 0.7-1.5 kpc, G 13-16): "
            "weighted orbits (C1), published accelerations, accel share of (orbits + accels)"]
    tbin = mbin & (1.0 / np.asarray(z["parallax_mas"], float) >= 0.7) & (1.0 / np.asarray(z["parallax_mas"], float) <= 1.5)
    for ax, (lab, sel) in zip(axs, (("all", np.ones(acc.size, bool)), ("0.7-1.5 kpc, G 13-16 (true d)", tbin))):
        w = W["noW"]
        o = np.histogram(lp[m_orb & sel], bins=pedges, weights=w[m_orb & sel])[0]
        a_ = np.histogram(lp[m_acc & sel], bins=pedges, weights=w[m_acc & sel])[0]
        ax.step(cen, o, where="mid", lw=2, color=series_style(0, style)["color"], label="orbits (C1)")
        ax.step(cen, a_, where="mid", lw=2, ls="--", color=series_style(1, style)["color"], label="published accelerations")
        ax.set_yscale("log")
        apply_axes_style(ax, style, xlabel="true log10 P (d)", ylabel="weighted mock count (noW)")
        ax.set_title(lab, fontsize=style.title_fontsize)
        ax.legend(fontsize=style.tick_label_fontsize)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = a_ / (o + a_)
        rep.append(f"  {lab}: " + " ".join(f"{c:.2f}:{x:.0f}/{y:.0f}/{s:.2f}" for c, x, y, s in zip(cen, o, a_, share) if x + y > 0))
        if sel is tbin:
            for lo, hi in ((450, P_MAX), (P_MAX, 2000), (2000, 1e9)):
                s = sel & (m_Ptrue > lo) & (m_Ptrue <= hi)
                oo, aa = w[s & m_orb].sum(), w[s & m_acc].sum()
                rep.append(f"    true P {lo:.0f}-{hi:.0f} d in bin: orbits {oo:.0f}, accelerations {aa:.0f}, accel share {aa / max(oo + aa, 1e-9):.2f}")
    fig.tight_layout()
    save_figure(fig, args.out_dir / "mock_orbit_accel_by_true_period.png", dpi=int(cfg.diagnostics.figure_dpi))

    (args.out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
