#!/usr/bin/env python3
"""Rung-2 pilot figures: MdS17-reweighted six-panel vs real, and ESS per bin (#391).

docs/MOCK_POPULATION_SPEC.md §3.6, §5 (rung 2), §7. Reads one proposal-set artifact from
``scripts/run_proposal_pilot.py``, reweights it to Moe & Di Stefano (2017) at published
parameters (luminous companions only; provisional settings from the artifact), and
compares accepted mock orbits with the real DR3 Orbital + AstroSpectroSB1 sample from the
uncut snapshot, filtered like the parent (spec §0.1). Every figure carries ``--label`` with N
accepted and ESS.

Several ``--artifact`` generations (top-ups, spec §3.7) are combined with
deterministic-mixture weights: every draw is evaluated under every generation's ``q_j``,
which needs the shared parent snapshot (``--parent-dir``).

Outputs (``--out-dir``): ``pilot_six_panel_mds17.png``, ``pilot_ess_per_bin.png``,
``pilot_report.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.diagnostic_hooks import SIX_PANEL_NAMES, SOLUTION_TYPE_LABELS
from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels
from darkhunter_pop.plotting import (
    apply_axes_style,
    plot_six_panel_grid,
    require_pyplot,
    resolve_plotting_style,
    save_figure,
    series_style,
    six_panel_bin_edges,
)
from darkhunter_pop.proposal_set import (
    MdS17TargetConfig,
    ParentExtinctionConfig,
    ProposalConfig,
    check_eccentricity_bounded,
    combined19_a_g,
    load_real_input_columns,
    malmquist_log_weight,
    load_parent_snapshot,
    log_q_total_for,
    real_comparison_keep,
    weighted_ks,
    binned_weights,
    importance_weights,
    kish_ess,
    mds17_luminous_log_intensity,
    read_proposal_artifact,
)

PANEL_LABELS = {
    "P_orb_days": "orbital period (day)",
    "G_mag": "G (mag)",
    "inv_parallax_mas_inv": r"1/$\varpi$ (kpc)",
    "eccentricity": "eccentricity",
    "f_m_msun": r"astrometric mass function (M$_\odot$)",
    "cos_inclination": r"cos $i$",
}


def mock_panel_values(truth: dict, outcome: dict) -> dict[str, np.ndarray]:
    """Fitted (catalog-like) six-panel values for every draw; NaN where not an orbit."""
    c = np.asarray(outcome["cascade"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_plx = np.where(c[:, 0] > 0, 1.0 / c[:, 0], np.nan)
    return {
        "P_orb_days": c[:, 10],
        "G_mag": np.asarray(truth["phot_g_mean_mag"], float),
        "inv_parallax_mas_inv": inv_plx,
        "eccentricity": c[:, 14],
        "f_m_msun": np.asarray(outcome["f_m_msun"], float),
        "cos_inclination": np.asarray(outcome["cos_inclination_fit"], float),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", type=Path, required=True, action="append")
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True, help="uncut snapshot query.ecsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--label", default="PILOT", help="figure/report label, e.g. PILOT or RUNG 2")
    ap.add_argument("--prefix", default="pilot", help="output filename prefix")
    ap.add_argument("--malmquist", type=Path, default=None,
                    help="decided Malmquist config (#405 weight added to the target); omit = no weight")
    ap.add_argument("--real-input-columns", type=Path, default=None,
                    help="snapshot from fetch_real_nss_input_columns.py (MP-Q24 real-side IPD/C* drop)")
    ap.add_argument("--rung2", type=Path, default=Path("config/population/rung2_validation.yaml"))
    ap.add_argument("--cmd-malmquist", type=Path, default=None,
                    help="2-D CMD Malmquist config (#418, malmquist_cmd.yaml); replaces --malmquist")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    parts = [read_proposal_artifact(a) for a in args.artifact]
    attrs = parts[0][2]
    prov = json.loads(attrs["provenance_json"])
    target = MdS17TargetConfig.model_validate(prov["target_mds17_json"])
    for _t, _o, at in parts[1:]:
        if json.loads(at["provenance_json"])["target_mds17_json"] != prov["target_mds17_json"]:
            raise ValueError("artifacts disagree on the target configuration")
        if at["parent_h5_sha256"] != attrs["parent_h5_sha256"]:
            raise ValueError("all generations must share one parent snapshot (spec §3.7)")
    truth = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    outcome = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    if np.unique(truth["draw_index"]).size != truth["draw_index"].size:
        raise ValueError("draw_index reused across generations")
    gen_cfgs = [ProposalConfig.model_validate_json(p[2]["proposal_config_json"]) for p in parts]
    prop_cfg = json.loads(attrs["proposal_config_json"])
    n_draw = int(truth["draw_index"].size)
    scale = float(attrs["scale_to_full"])
    gen_time_keys = (
        "parallax_floor_mas", "halbwachs_ipd_cstar_cuts", "halbwachs_cuts", "truth_distance",
        "m1", "light_split", "data_release", "ruwe_min", "skip_acceleration",
    )
    for gc in gen_cfgs[1:]:
        for key in gen_time_keys:
            if getattr(gc, key) != getattr(gen_cfgs[0], key):
                raise ValueError(f"generations differ in generation-time setting {key!r} (spec §3.5)")
    parent = load_parent_snapshot(args.parent_dir, cfg, gen_cfgs[0])
    acc = np.asarray(outcome["accepted_orbital"], bool)
    stype = np.asarray(outcome["solution_type"]).astype(str)

    # Target exactly as scripts/smoke_isochrone_weights.py: evolved-row flux relation (#416)
    # and, for mass_luminosity mist_coeval (MP-Q40), the per-draw MIST relation.
    import darkhunter_pop.proposal_set as _ps

    evo = _ps.evolved_mg0_for_draws(truth, parent) if getattr(parent, "cmd", None) is not None else None
    rel = (_ps.mist_relation_for_draws(truth, parent, cfg)
           if getattr(target, "mass_luminosity", None) == "mist_coeval" else None)
    log_lam = mds17_luminous_log_intensity(truth, target, evolved_mg0_system=evo, relation_log10_f=rel)
    ecc_shapes = sorted({gc.eccentricity.shape for gc in gen_cfgs})
    for gc in gen_cfgs:
        if gc.eccentricity.shape != "uniform":
            check_eccentricity_bounded(gc.eccentricity, target.provisional_eta_floor)
    malm_counts: dict[str, int] | None = None
    zp_note = "no Malmquist weight"
    if args.malmquist is not None:
        import yaml as _yaml

        from darkhunter_pop.malmquist import MalmquistConfig

        mraw = _yaml.safe_load(args.malmquist.read_text())
        mcfg = MalmquistConfig.model_validate(mraw["malmquist"])
        ext = ParentExtinctionConfig.model_validate(mraw["extinction"])
        a_g = combined19_a_g(parent, ext)
        lw_m, malm_counts = malmquist_log_weight(truth, parent, target, mcfg, a_g, ext)
        log_lam = log_lam + lw_m
        zp_note = (
            f"Malmquist weight (#405) with zp={mcfg.provisional_mg_zero_point_mag:+.3f} mag, "
            f"sigma_int={mcfg.provisional_sigma_int_mag:.3f} mag (MP-Q25 fit, see #414), Combined19 A_G"
        )
    if args.cmd_malmquist is not None:
        if args.malmquist is not None:
            raise SystemExit("use either --malmquist (1-D, #405) or --cmd-malmquist (2-D CMD, #418), not both")
        from darkhunter_pop import malmquist_cmd as _mc

        cmcfg = _mc.load_cmd_malmquist_config(args.cmd_malmquist)
        lw_c, cmd_counts = _ps.malmquist_cmd_log_weight(truth, parent, target, cmcfg, cfg)
        log_lam = log_lam + lw_c
        malm_counts = cmd_counts.get("by_row_reason", cmd_counts) if isinstance(cmd_counts, dict) else cmd_counts
        zp_note = f"2-D CMD Malmquist weight (#418, {args.cmd_malmquist}); MIST coeval flux ratios"
    rung2 = __import__("yaml").safe_load(args.rung2.read_text())["rung2"]
    min_ess = float(rung2["min_ess_per_bin"])
    if any(gc.flux.evolved_rows_centre == "evolved_relation_deblended" for gc in gen_cfgs):
        import darkhunter_pop.proposal_set as _pse

        _pse.ensure_m1_deblend_dark(truth, parent, cfg, gen_cfgs)  # #391 option (i)
    log_qs = [log_q_total_for(truth, parent, gc, cfg) for gc in gen_cfgs]
    w = importance_weights(
        log_lam, log_qs, [gc.n_draws for gc in gen_cfgs], scale_to_full=scale
    )
    per_gen = []
    for gc in gen_cfgs:
        sel = (truth["generation"] == gc.generation) & np.asarray(outcome["accepted_orbital"], bool)
        per_gen.append((gc.generation, gc.n_draws, int(sel.sum()), kish_ess(w[sel])))

    # --- real sample (El-Badry et al. 2024 §4: Orbital + AstroSpectroSB1, uncut) ---
    from astropy.table import Table

    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gm = import_gaiamock_mod()
    real_table = Table.read(args.real_snapshot, format="ascii.ecsv")
    real_types = cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types
    is_type = np.isin(np.asarray(real_table["nss_solution_type"]).astype(str), list(real_types))
    real_sel = real_table[is_type]
    # Mirror the decided parent filters on the real side (spec §0.1: MP-Q1, MP-Q5).
    in_cols = load_real_input_columns(args.real_input_columns) if args.real_input_columns else None
    keep, real_counts = real_comparison_keep(real_sel, gen_cfgs[0], in_cols)
    real_panels, n_real = build_elbadry2024_comparison_panels(
        real_sel[keep], nss_solution_types=real_types, gaiamock=gm
    )
    real_label = (
        f"DR3 Orbital+AstroSpectroSB1, parallax > {gen_cfgs[0].parallax_floor_mas} mas, "
        f"TAG10 atmosphere{', Halbwachs IPD/C* (MP-Q24)' if in_cols is not None else ''} "
        f"(N={n_real} of {real_counts['rows']})"
    )

    mock_vals = mock_panel_values(truth, outcome)
    axes_cfg = {k: (v.scale, v.xmin, v.xmax) for k, v in cfg.diagnostics.elbadry_six_panel_axes.items()}
    n_bins = int(rung2["n_bins"])

    lam_orbit = float(np.sum(w[acc]))
    ess_acc = kish_ess(w[acc])
    n_acc = int(acc.sum())
    n_acc_wpos = int(np.sum(acc & (w > 0)))
    thr = float(cfg.physics.mc_noise_threshold)

    # Per-panel bins, ESS and the MP-Q19 display/KS minimum (decided: ESS_b >= min_ess).
    panel_bins: dict[str, tuple[np.ndarray, Any]] = {}
    shade: dict[str, list[tuple[float, float]]] = {}
    for name in SIX_PANEL_NAMES:
        sc, lo, hi = axes_cfg[name]
        edges = six_panel_bin_edges(lo, hi, scale=sc, n_bins=n_bins)
        bw = binned_weights(mock_vals[name][acc], w[acc], edges)
        panel_bins[name] = (edges, bw)
        shade[name] = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins) if bw.ess[i] < min_ess]

    panels = {}
    weights = {}
    for name in SIX_PANEL_NAMES:
        # Short legend labels; N and ESS go in the title and caption (labels must not overflow).
        panels[name] = {
            "DR3": real_panels[name],  # filtered as in real_label
            "MdS17 wt": mock_vals[name][acc],
            "proposal": mock_vals[name][acc],
        }
        weights[name] = {"MdS17 wt": w[acc]}

    provisional = {k: v for k, v in prop_cfg.items() if k.startswith("provisional_")}
    provisional.update({k: v for k, v in target.model_dump(mode="json").items() if k.startswith("provisional_")})
    decided = {
        k: prop_cfg[k]
        for k in ("parallax_floor_mas", "halbwachs_ipd_cstar_cuts", "truth_distance", "m1", "light_split")
    }
    decided.update(mass_luminosity=target.mass_luminosity, flux_sigma_dex=target.flux_sigma_dex)
    lab = args.label
    caption = (
        f"{lab} (#391, docs/MOCK_POPULATION_SPEC.md), validation rung 2. Real: {real_label}. {n_draw} proposal draws "
        f"(generations {[g[0] for g in per_gen]}, deterministic-mixture weights, base seed "
        f"{prop_cfg['base_seed']}) through gaiamock_mod; "
        f"{n_acc} accepted orbits pass every orbital_solution_cut, {n_acc_wpos} with nonzero MdS17 weight; "
        f"Kish ESS of the accepted set = {ess_acc:.1f}. Series: DR3 = real sample; MdS17 wt = accepted mock orbits "
        f"reweighted to MdS17; proposal = the same accepted orbits unweighted. Weighted mock is Moe & Di Stefano (2017) at "
        f"published parameters, luminous companions only. Expected accepted orbits relative to the "
        f"G<19 parent: {lam_orbit:.3g} (real: {n_real}); shapes only are compared (unit area). "
        f"Generation-time settings ({prop_cfg['decision_ref']}): "
        + "; ".join(f"{k}={v}" for k, v in decided.items())
        + ". Reweightable provisional settings, not decisions: "
        + "; ".join(f"{k.removeprefix('provisional_')}={v}" for k, v in provisional.items())
        + f". {zp_note}. Eccentricity proposal: {ecc_shapes}. Grey bands: bins with MdS17 ESS < {min_ess:g} "
        f"(MP-Q19), excluded from the KS gate; details in {args.prefix}_report.txt."
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_six_panel_grid(
        panels,
        args.out_dir / f"{args.prefix}_six_panel_mds17.png",
        panel_order=SIX_PANEL_NAMES,
        dpi=int(cfg.diagnostics.figure_dpi),
        title=(
            f"{lab}\nrung 2: DR3 Orbital+AstroSpectroSB1 (N={n_real})\n"
            f"vs MdS17-reweighted mock "
            f"(N_acc={n_acc}, MdS17 ESS={ess_acc:.1f})"
        ),
        panel_xlabels=PANEL_LABELS,
        panel_axes=axes_cfg,
        max_bins=n_bins,
        density=True,
        caption=caption,
        style=cfg.plotting,
        series_weights=weights,
        shade_ranges=shade,
    )

    # --- ESS per six-panel bin ---
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    fig, axs = plt.subplots(2, 3, figsize=(style.figsize_landscape[0] / 7.0 * 12, 8.5))
    lines = []
    ks_info: dict[str, tuple[float, float, float]] = {}
    for ax, name in zip(np.ravel(axs), SIX_PANEL_NAMES):
        sc, lo, hi = axes_cfg[name]
        edges, bw = panel_bins[name]
        raw = np.histogram(mock_vals[name][acc], bins=edges)[0]
        s0, s1 = series_style(0, style), series_style(1, style)
        ax.stairs(np.maximum(bw.ess, 1e-1), edges, color=s0["color"], linestyle=s0["linestyle"],
                  linewidth=s0["linewidth"], label="ESS (MdS17 weights)")
        ax.stairs(np.maximum(raw, 1e-1), edges, color=s1["color"], linestyle=s1["linestyle"],
                  linewidth=s1["linewidth"], label="accepted draws")
        ax.axhline(min_ess, color="0.4", linestyle=":", linewidth=1.5, label=f"MP-Q19 minimum ESS {min_ess:g}")
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(2.0, raw.max() * 2.0))
        if sc == "log":
            ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        apply_axes_style(ax, style, xlabel=PANEL_LABELS[name], ylabel="draws per bin")
        ax.legend(loc="best", fontsize=style.tick_label_fontsize)
        trusted = int(np.sum(bw.trusted(thr)))
        good = bw.ess >= min_ess
        # MP-Q19 KS gate: compare real and mock only inside the bins with ESS_b >= min_ess.
        rv = np.asarray(real_panels[name], float)
        mv = np.asarray(mock_vals[name][acc], float)
        rb = np.digitize(rv, edges) - 1
        mb = np.digitize(mv, edges) - 1
        r_in = (rb >= 0) & (rb < n_bins) & good[np.clip(rb, 0, n_bins - 1)]
        m_in = (mb >= 0) & (mb < n_bins) & good[np.clip(mb, 0, n_bins - 1)]
        frac_real = float(r_in.mean()) if rv.size else float("nan")
        if good.any() and r_in.any() and m_in.any():
            d, n_eff, pval = weighted_ks(rv[r_in], mv[m_in], w[acc][m_in])
            ks_txt = (f"KS over the {int(good.sum())} bins with ESS >= {min_ess:g} "
                      f"(holding {frac_real:.1%} of the real sample): D={d:.3f}, n_eff={n_eff:.1f}, p={pval:.3g}")
        else:
            ks_txt = f"KS not computed: no bin reaches ESS >= {min_ess:g}"
        lines.append(
            f"{name}: bins={n_bins}, max ESS_b={bw.ess.max():.2f}, bins with ESS >= {min_ess:g}: {int(good.sum())}/{n_bins}; "
            f"(spec §3.6 sigma_MC/sigma_Poisson<{thr}: {trusted}/{n_bins}); {ks_txt}"
        )
        ks_info[name] = weighted_ks(real_panels[name], mock_vals[name][acc], w[acc])
    fig.suptitle(
        f"{lab}\neffective sample size per six-panel bin ({n_acc} accepted, MdS17 ESS {ess_acc:.1f})",
        fontfamily=style.font_family, fontsize=style.title_fontsize,
    )
    fig.tight_layout()
    save_figure(fig, args.out_dir / f"{args.prefix}_ess_per_bin.png", dpi=int(cfg.diagnostics.figure_dpi))

    # --- report ---
    cpu = np.asarray(outcome["cpu_seconds"], float)
    report = [
        "artifacts: " + ", ".join(str(a) for a in args.artifact),
        f"{lab} report (#391; docs/MOCK_POPULATION_SPEC.md). Provisional settings are not decisions.",
        f"decision_ref: {prop_cfg['decision_ref']}; decided: {json.dumps(decided, sort_keys=True)}",
        f"real comparison: {real_label}; filter counts {real_counts}",
        "parent attrition: " + json.dumps(json.loads(attrs["provenance_json"]).get("parent_attrition")),
        f"draws: {n_draw}; scale_to_full (N_full/N_snap): {scale:.2f}",
        "per generation (generation, n_draws, accepted, ESS of its accepted draws under mixture weights): "
        + "; ".join(f"({g}, {n}, {a}, {e:.2f})" for g, n, a, e in per_gen),
        f"accepted orbits: {n_acc}; with nonzero MdS17 weight: {n_acc_wpos}",
        f"MdS17 Kish ESS: accepted {ess_acc:.2f}; all draws {kish_ess(w):.2f}",
        f"max weight share among accepted: {(w[acc].max() / w[acc].sum()) if w[acc].sum() > 0 else float('nan'):.3f}",
        f"expected accepted orbits relative to parent (MdS17, luminous): {lam_orbit:.4g} vs real {n_real}",
        "",
        "solution-type mix (raw proposal fraction | MdS17-weighted expected count):",
    ]
    for lab in SOLUTION_TYPE_LABELS:
        sel = stype == lab
        report.append(f"  {lab:28s} {sel.mean():.3f} | {np.sum(w[sel]):.4g}")
    pub = np.asarray(outcome["published_acceleration"], bool)
    report.append(f"  published acceleration (s>20; F2<22 for 7-par): raw {pub.mean():.3f} | weighted {np.sum(w[pub]):.4g}")
    report += [f"Malmquist: {zp_note}; unit-weight counts {malm_counts}", f"eccentricity proposal shapes: {ecc_shapes}"]
    report += ["", f"per-panel ESS and KS (KS gated on MP-Q19, ESS_b >= {min_ess:g}):"] + ["  " + ln for ln in lines]
    report += ["", "informational weighted KS over the full panel range (NOT a gate):"]
    report += [f"  {k}: D={v[0]:.3f}, n_eff={v[1]:.1f}, p={v[2]:.3g}" for k, v in ks_info.items()]
    report += [
        "",
        "cost (CPU s per draw, gaiamock_mod, process_time in worker):",
        f"  mean {cpu.mean():.2f}, median {np.median(cpu):.2f}, p90 {np.percentile(cpu, 90):.2f}, total {cpu.sum() / 3600:.2f} CPU h",
    ]
    for lab in SOLUTION_TYPE_LABELS:
        sel = stype == lab
        if sel.any():
            report.append(f"  {lab:28s} n={int(sel.sum()):5d} mean {cpu[sel].mean():.2f} s")
    for _t, _o, at in parts:
        pv = json.loads(at["provenance_json"])
        report.append(f"  wall: {pv.get('wall_seconds_this_invocation', float('nan')) / 60:.1f} min at {pv.get('workers')} workers")
    report.append("provisional: " + json.dumps(provisional, sort_keys=True))
    (args.out_dir / f"{args.prefix}_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
