"""Closed loop for the 2-D CMD Malmquist weight with MIST photometry (#418; spec §11.5).

Extends the #405 universe (:mod:`darkhunter_pop.malmquist_closed_loop`: exponential disk,
MdS17 luminous companions from the same table and provisional settings as the target, the
same parent cuts, Bailer-Jones-like distances) so that every star has Gaia CMD photometry:

* primaries are MIST main-sequence points drawn with the isochrone prior weights (Kroupa IMF,
  constant SFR, the [Fe/H] prior) inside the closed-loop IMF mass range;
* each companion's G light follows the target's f model (decided Janssens relation with
  σ_f = 0.1 dex); its BP and RP come from the MIST main sequence at M2 **at the primary's own
  age and [Fe/H]** (the pipeline's weight uses a fiducial MS, MP-Q37, so its cost is measured);
* a synthetic extinction vector E(B−V) (dust layer, constant A_G / E(B−V) and E(BP−RP) /
  E(B−V)), which the pipeline knows to a configurable fractional error, and a colour error.

The pipeline under test then runs exactly as on real rows: dereddened CMD → isochrone M̂1
(:mod:`darkhunter_pop.isochrone_mass`) → the MS ridge measured from the synthetic parent
(:func:`darkhunter_pop.giants.fit_ms_ridge`) → :func:`proposal_set.sample_proposal` → the
MdS17 target → weights with no W, the 1-D W of spec §9.3 and the 2-D W of §11.4. No gaiamock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from darkhunter_pop import giants
from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import malmquist as mq
from darkhunter_pop import malmquist_closed_loop as mcl
from darkhunter_pop import malmquist_cmd as mc
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import repo_root

FloatArray = NDArray[np.float64]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticDustConfig(_Strict):
    """Synthetic dust layer: E(B−V) = rate × d_kpc × (1 − e^{−x}) / x, x = d |sin b| / h."""

    ebv_per_kpc: float = Field(..., ge=0)
    scale_height_pc: float = Field(..., gt=0)
    a_g_per_ebv: float = Field(..., gt=0)
    e_bp_rp_per_ebv: float = Field(..., gt=0)
    pipeline_fractional_error: float = Field(..., ge=0)  # the pipeline's E(B−V) = truth × (1 + ε N)


class CmdClosedLoopConfig(_Strict):
    base_config: str  # the #405 closed-loop YAML (disk, IMF range, observation, mock, analysis)
    malmquist_cmd_config: str
    giants_config: str
    seed: int
    sizes: dict[str, int]
    dust: SyntheticDustConfig
    colour_error_mag: float = Field(..., ge=0)
    one_d_sigma_int_mag: float = Field(..., gt=0)  # §9.3 comparison run (zero point 0)
    #: Numerical overrides of the giants ridge settings for the (much smaller) synthetic parent,
    #: e.g. ``min_rows_per_bin``; the estimator itself is unchanged.
    ridge_overrides: dict[str, float] = Field(default_factory=dict)
    #: Primaries are prior-weighted MIST points with (sub-stepped) phase <= this: 0 = MS only,
    #: 3 = MS + SGB/RGB + core-He burning (evolved rows then get W = 1, spec §10.6).
    universe_max_phase: float = Field(0.4, ge=0)
    #: MP-Q35 + MP-Q36 (decided 2026-10-04): per draw, a posterior draw of the primary and the
    #: coeval deblended truth M1 (``proposal_set.apply_posterior_deblending``).
    posterior_deblending: bool = True
    #: MP-Q40 (decided 2026-10-04): the universe's and the target's MS-companion G-flux ratio
    #: come from the coeval MIST isochrone (``mist_coeval``) instead of Janssens.
    mass_luminosity: Literal["janssens2022", "mist_coeval"] = "mist_coeval"


def load_cmd_closed_loop_config(path: str | Path = "config/population/malmquist_cmd_closed_loop.yaml") -> CmdClosedLoopConfig:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return CmdClosedLoopConfig.model_validate(yaml.safe_load(p.read_text())["closed_loop_cmd"])


# ---------------------------------------------------------------------------
# Universe with CMD photometry
# ---------------------------------------------------------------------------


@dataclass
class CmdUniverse:
    base: mcl.Universe  # m1 (true current mass), comp, pos, g_true (apparent, with A_G)
    colour_true: FloatArray  # apparent (BP−RP) without noise, with reddening
    colour_abs: FloatArray  # intrinsic system (BP−RP)
    mg_abs: FloatArray  # intrinsic system M_G
    mg1_abs: FloatArray  # primary alone
    colour1_abs: FloatArray
    ebv: FloatArray
    feh: FloatArray
    log_age: FloatArray
    evolved_true: NDArray[np.bool_]


def _sample_primaries(
    pts: im.PriorPoints, n: int, m_lo: float, m_hi: float, rng: np.random.Generator, max_phase: float = 0.4
) -> dict[str, FloatArray]:
    v = pts.values
    m = v["star_mass"]
    ok = (v["phase"] <= max_phase) & (m >= m_lo) & (m <= m_hi)
    idx = np.flatnonzero(ok)
    p = pts.weight[idx] / pts.weight[idx].sum()
    pick = idx[rng.choice(idx.size, size=n, p=p)]
    return {
        "m1": m[pick], "mg": v["mg"][pick], "bp": v["bp"][pick], "rp": v["rp"][pick],
        "feh": pts.feh[pick], "log_age": pts.log_age[pick], "phase": v["phase"][pick],
    }


def companion_colours_true(
    grid: im.NativeGrid, feh: FloatArray, log_age: FloatArray, m2: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """(BP−G, G−RP) of a MS star of mass m2 at the nearest native ([Fe/H], age) of each primary."""
    fi = np.abs(grid.feh[None, :] - feh[:, None]).argmin(axis=1)
    ai = np.abs(grid.log_age[None, :] - log_age[:, None]).argmin(axis=1)
    bpg = np.full(m2.size, np.nan)
    grp = np.full(m2.size, np.nan)
    key = fi * grid.log_age.size + ai
    for k in np.unique(key[np.isfinite(m2)]):
        sel = (key == k) & np.isfinite(m2)
        mass, rel = im.main_sequence_colour_relation(grid, float(grid.feh[k // grid.log_age.size]), float(grid.log_age[k % grid.log_age.size]))
        mm = np.clip(m2[sel], mass[0], mass[-1])
        bpg[sel] = np.interp(mm, mass, rel["bp_g"])
        grp[sel] = np.interp(mm, mass, rel["g_rp"])
    return bpg, grp


def companion_mg_true(grid: im.NativeGrid, feh: FloatArray, log_age: FloatArray, m: FloatArray) -> FloatArray:
    """M_G of a MS star of mass ``m`` at the nearest native ([Fe/H], age) of each primary."""
    fi = np.abs(grid.feh[None, :] - feh[:, None]).argmin(axis=1)
    ai = np.abs(grid.log_age[None, :] - log_age[:, None]).argmin(axis=1)
    out = np.full(m.size, np.nan)
    key = fi * grid.log_age.size + ai
    for k in np.unique(key[np.isfinite(m)]):
        sel = (key == k) & np.isfinite(m)
        mass, rel = im.main_sequence_colour_relation(grid, float(grid.feh[k // grid.log_age.size]), float(grid.log_age[k % grid.log_age.size]))
        out[sel] = np.interp(np.clip(m[sel], mass[0], mass[-1]), mass, rel["mg"])
    return out


def synthetic_ebv(pos: dict[str, FloatArray], dust: SyntheticDustConfig) -> FloatArray:
    d = pos["distance_pc"]
    x = d * np.abs(np.sin(np.radians(pos["b_deg"]))) / dust.scale_height_pc
    frac = np.where(x > 1e-6, (1.0 - np.exp(-x)) / np.maximum(x, 1e-12), 1.0)
    return dust.ebv_per_kpc * (d / 1000.0) * frac


def make_cmd_universe(
    n: int,
    base: mcl.ClosedLoopConfig,
    cfg: CmdClosedLoopConfig,
    target: ps.MdS17TargetConfig,
    grid1d: mq.FluxMarginalGrid,
    native: im.NativeGrid,
    pts: im.PriorPoints,
    rng: np.random.Generator,
) -> CmdUniverse:
    pos = mcl.draw_positions(n, base.disk, rng)
    pr = _sample_primaries(pts, n, base.imf.m_min_msun, base.imf.m_max_msun, rng, cfg.universe_max_phase)
    m1 = pr["m1"]
    mist = cfg.mass_luminosity == "mist_coeval"
    cgrid = mc.build_q_grid(target, mc.load_cmd_malmquist_config(cfg.malmquist_cmd_config).grid) if mist else grid1d
    parts = [mcl.draw_companions(m1[a:a + mcl.COMPANION_CHUNK], target, cgrid, rng, require_janssens=not mist)
             for a in range(0, n, mcl.COMPANION_CHUNK)]
    comp = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    has = comp["has_companion"]
    # Evolved primaries (MIST phase >= 1.5): the companion is a MS star whose G light follows the
    # dwarf relation for M2 itself, f = L2 / L1 = 10^{-0.4 (M_G^J(M2) - M_G,1)} (spec §10.4), with
    # the decided 0.1 dex scatter; the dwarf-relation f(M1, M2) would be ~2 dex too bright.
    evolved_true = pr["phase"] >= 1.5
    if cfg.mass_luminosity == "mist_coeval":  # MP-Q40: f from the primary's own MS, coeval
        mg2 = companion_mg_true(native, pr["feh"], pr["log_age"], np.where(has, comp["m2_msun"], np.nan))
        mg1_ms = companion_mg_true(native, pr["feh"], pr["log_age"], m1)
        lf_ms = -0.4 * (mg2 - mg1_ms) + target.flux_sigma_dex * rng.standard_normal(n)
        comp["log10_f"] = np.where(has & ~evolved_true, lf_ms, comp["log10_f"])
    with np.errstate(invalid="ignore"):
        lf_evo = -0.4 * (ps.janssens_absolute_g(comp["m2_msun"]) - pr["mg"]) + target.flux_sigma_dex * rng.standard_normal(n)
    comp["log10_f"] = np.where(has & evolved_true, np.where(np.isfinite(lf_evo), lf_evo, -np.inf), comp["log10_f"])
    f = np.where(has, 10.0 ** comp["log10_f"], 0.0)
    bpg2, grp2 = companion_colours_true(native, pr["feh"], pr["log_age"], np.where(has, comp["m2_msun"], np.nan))
    g2 = pr["mg"] - 2.5 * np.log10(np.where(has, f, 1.0))
    flux = lambda mag: 10.0 ** (-0.4 * mag)  # noqa: E731
    g_sys = -2.5 * np.log10(flux(pr["mg"]) + np.where(has, flux(g2), 0.0))
    bp_sys = -2.5 * np.log10(flux(pr["bp"]) + np.where(has, flux(g2 + np.nan_to_num(bpg2)), 0.0))
    rp_sys = -2.5 * np.log10(flux(pr["rp"]) + np.where(has, flux(g2 - np.nan_to_num(grp2)), 0.0))
    ebv = synthetic_ebv(pos, cfg.dust)
    mu = mq.distance_modulus(pos["distance_pc"])
    g_app = g_sys + mu + cfg.dust.a_g_per_ebv * ebv
    c_abs = bp_sys - rp_sys
    u = mcl.Universe(m1=m1, eps=np.zeros(n), comp=comp, pos=pos, g_true=g_app)
    return CmdUniverse(
        base=u, colour_true=c_abs + cfg.dust.e_bp_rp_per_ebv * ebv, colour_abs=c_abs, mg_abs=g_sys,
        mg1_abs=pr["mg"], colour1_abs=pr["bp"] - pr["rp"], ebv=ebv, feh=pr["feh"], log_age=pr["log_age"],
        evolved_true=evolved_true,
    )


# ---------------------------------------------------------------------------
# Pipeline on the synthetic parent
# ---------------------------------------------------------------------------


@dataclass
class CmdPipeline:
    parent: ps.ParentSnapshot
    par: mcl.SyntheticParent
    mg0: FloatArray
    colour0: FloatArray
    sigma_mu: FloatArray
    a_g_hat: FloatArray
    e_br_hat: FloatArray
    post: im.IsochronePosterior
    ridge: giants.MSRidge
    evolved: NDArray[np.bool_]


def run_pipeline_on_parent(
    uc: CmdUniverse,
    base: mcl.ClosedLoopConfig,
    cfg: CmdClosedLoopConfig,
    model: im.IsochroneMassModel,
    gcfg: giants.GiantsConfig,
    rng: np.random.Generator,
) -> CmdPipeline:
    u = uc.base
    par = mcl.observe(u, base, rng)
    i = par.index
    c_obs = uc.colour_true[i] + cfg.colour_error_mag * rng.standard_normal(i.size)
    e_hat = uc.ebv[i] * (1.0 + cfg.dust.pipeline_fractional_error * rng.standard_normal(i.size))
    a_hat = cfg.dust.a_g_per_ebv * e_hat
    ebr_hat = cfg.dust.e_bp_rp_per_ebv * e_hat
    mg0 = u.g_true[i] - mq.distance_modulus(par.r_med) - a_hat
    c0 = c_obs - ebr_hat
    smu = mq.sigma_mu_from_quantiles(par.r_lo, par.r_med, par.r_hi)
    post = model.fit(c0, mg0, sigma_mu=smu, ebv=e_hat, a_g=a_hat, e_bp_rp=ebr_hat)
    m1_hat = post.point(model.cfg.provisional_point_estimate)
    s_plx = mcl.parallax_error(u.g_true[i], base.observation.parallax_error, base.observation.g_limit)
    rcfg = gcfg.ridge.model_copy(update={"ruwe_max": None, **cfg.ridge_overrides})
    ridge = giants.fit_ms_ridge(mg0, c0, par.plx_obs / s_plx, rcfg)
    evolved = giants.classify_evolved(mg0, c0, smu, ridge, gcfg.provisional_n_sigma).evolved
    snap = mcl.parent_snapshot(u, par)
    usable = np.isfinite(m1_hat)
    parent = ps.ParentSnapshot(
        columns=snap.columns, m1_msun=np.where(usable, m1_hat, 1.0), m1_source=np.full(i.size, "MIST"),
        atmosphere_logg=np.asarray(post.log_g_mean), truth_parallax_mas=snap.truth_parallax_mas,
        is_giant=evolved, flags={"isochrone_m1": usable}, usable=usable, meta=snap.meta,
        scale_to_full=1.0, path=Path("synthetic"),
        cmd={"mg0": mg0, "colour0": c0, "sigma_mu": smu, "a_g": a_hat, "e_bp_rp": ebr_hat, "ebv": e_hat},
        isochrone=post.as_dict(),
    )
    return CmdPipeline(parent=parent, par=par, mg0=mg0, colour0=c0, sigma_mu=smu, a_g_hat=a_hat, e_br_hat=ebr_hat,
                       post=post, ridge=ridge, evolved=evolved)


@dataclass
class CmdMock:
    truth: dict[str, NDArray[Any]]
    w: dict[str, FloatArray]
    p_single: dict[str, FloatArray]
    unit_counts: dict[str, int]


def run_cmd_mock(
    pipe: CmdPipeline,
    base: mcl.ClosedLoopConfig,
    cfg: CmdClosedLoopConfig,
    frag: ps.ProposalSetFragment,
    mcfg: mq.MalmquistConfig,
    cmcfg: mc.CmdMalmquistConfig,
    grid1d: mq.FluxMarginalGrid,
    native: im.NativeGrid,
    cmap: im.CmdMap,
    *,
    pipeline_config: Any = None,
    sampler: Any = None,
) -> CmdMock:
    parent = pipe.parent
    prop = mcl.closed_loop_proposal(frag, base, parent.n_rows)
    truth = ps.sample_proposal(parent, prop)
    if cfg.posterior_deblending:
        truth = ps.apply_posterior_deblending(truth, parent, pipeline_config, prop, sampler=sampler, native=native)
    # #416 / spec §10.4: CMD-evolved rows use the evolved flux relation in the target.
    rel = ps.mist_relation_for_draws(truth, parent, pipeline_config, native=native) if frag.target_mds17.mass_luminosity == "mist_coeval" else None
    log_lam = ps.mds17_luminous_log_intensity(truth, frag.target_mds17, evolved_mg0_system=ps.evolved_mg0_for_draws(truth, parent),
                                              relation_log10_f=rel)
    n = [prop.n_draws]
    lq = [truth["log_q_total"]]
    w = {"none": ps.importance_weights(log_lam, lq, n, scale_to_full=1.0)}
    qf = mc.build_q_grid(frag.target_mds17, cmcfg.grid) if frag.target_mds17.mass_luminosity == "mist_coeval" else mc.build_qf_grid(frag.target_mds17, cmcfg.grid)
    _, f_lum = qf.interpolate(parent.m1_msun)
    p_single = {"none": np.clip(1.0 - f_lum, 0.0, 1.0)}
    # 1-D §9.3 weight fed the isochrone M̂1 (zero point 0, configured σ_int)
    m1d = mcfg.model_copy(update={"provisional_sigma_int_mag": cfg.one_d_sigma_int_mag, "provisional_mg_zero_point_mag": 0.0,
                                   "provisional_sigma_log_m1_dex": 0.0})
    rows1 = mq.row_conditioning(parent, m1d, a_g_mag=pipe.a_g_hat, sigma_a_mag=0.0)
    w["one_d"] = ps.importance_weights(log_lam + mq.log_weight_for_draws(truth, rows1, grid1d, m1d), lq, n, scale_to_full=1.0)
    lps1, _ = mq.log_no_companion_probability(rows1.m1_msun, rows1.delta_m, rows1.sigma, grid1d)
    p_single["one_d"] = np.where(rows1.is_giant | ~np.isfinite(lps1), p_single["none"], np.exp(lps1))
    # 2-D §11.4 weight
    rt = mc.RidgeTables.from_ridge(pipe.ridge)
    cc = cmcfg.companion_colour
    ms = mc.ms_colour_bank(native, pipe.post.feh_mean, pipe.post.log_age_mean, cc)  # MP-Q37 coeval
    rows = mc.cmd_rows(pipe.colour0, pipe.mg0, pipe.sigma_mu, parent.m1_msun, rt, evolved=pipe.evolved,
                       a_g=pipe.a_g_hat, e_bp_rp=pipe.e_br_hat)
    dens = None
    if cmcfg.single_star_density.provisional_model == "mist_density_ridge_anchored":
        dens = mc.build_single_star_density(cmap, cmcfg.single_star_density, pipe.ridge)
    norm = mc.row_normalization(rows, qf, rt, ms, cmcfg, dens=dens)
    lw2 = mc.log_weight_for_draws(truth, rows, norm, rt, ms, cmcfg, dens=dens)
    w["two_d"] = ps.importance_weights(log_lam + lw2, lq, n, scale_to_full=1.0)
    p_single["two_d"] = np.where(rows.unit_weight, p_single["none"], np.exp(norm.log_p_single))
    return CmdMock(truth=truth, w=w, p_single=p_single, unit_counts=rows.counts())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class CmdClosedLoopResult:
    size: str
    counts: dict[str, int]
    unit_counts: dict[str, int]
    m1_hat_vs_true: dict[str, float]
    tables: dict[str, dict[str, Any]]  # stat name -> {"edges", "truth", "<w>", "pull_<w>"}
    total: dict[str, float]
    ess: dict[str, float]


def _cc(
    truth_value: FloatArray, truth_has: NDArray[np.bool_], mock_value: FloatArray, row: NDArray[np.int64],
    w: dict[str, FloatArray], edges: FloatArray, n_rows: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"edges": np.asarray(edges, float).tolist()}
    for key in ("one_d", "two_d"):
        c = mcl.compare_counts(truth_value, truth_has, mock_value, row, w["none"], w[key], edges, n_rows)
        out["truth"] = c.truth.tolist()
        out["none"] = c.naive.tolist()
        out["pull_none"] = c.pull_naive.tolist()
        out[key] = c.corrected.tolist()
        out[f"pull_{key}"] = c.pull_corrected.tolist()
    return out


def _cf(truth_flag: NDArray[np.bool_], group: FloatArray, p: dict[str, FloatArray], edges: FloatArray) -> dict[str, Any]:
    out: dict[str, Any] = {"edges": np.asarray(edges, float).tolist()}
    for key in ("one_d", "two_d"):
        c = mcl.compare_row_fraction(truth_flag, group, p["none"], p[key], edges)
        out["truth"] = c.truth.tolist()
        out["n_group"] = (c.n_group.tolist() if c.n_group is not None else [])
        out["none"] = c.naive.tolist()
        out["pull_none"] = c.pull_naive.tolist()
        out[key] = c.corrected.tolist()
        out[f"pull_{key}"] = c.pull_corrected.tolist()
    return out


def run_cmd_closed_loop(
    size: Literal["small", "large"] | int,
    cfg: CmdClosedLoopConfig | None = None,
    *,
    data_root: str | Path | None = None,
    pipeline_config: Any = None,
    single_star_model: Literal["gaussian_ridge", "mist_density_ridge_anchored"] | None = None,
) -> tuple[CmdClosedLoopResult, dict[str, Any]]:
    """Build, observe, run the pipeline, mock and compare (spec §11.5)."""
    from darkhunter_pop.config_loader import load_config

    cfg = cfg or load_cmd_closed_loop_config()
    pc = pipeline_config or load_config(host_profile="laptop")
    droot = data_root or pc.paths.data_root
    base = mcl.load_closed_loop_config(cfg.base_config)
    frag = mcl.closed_loop_fragment(base)
    frag = frag.model_copy(update={"target_mds17": frag.target_mds17.model_copy(update={"mass_luminosity": cfg.mass_luminosity})})
    mcfg = mq.load_malmquist_config(base.malmquist_config)
    cmcfg = mc.load_cmd_malmquist_config(cfg.malmquist_cmd_config)
    gcfg = giants.load_giants_config(cfg.giants_config)
    grid1d = mq.build_flux_marginal(frag.target_mds17, mcfg.grid)
    native = im.load_native_grid(pc.isochrone_mass, droot)
    pts = im.prior_points(native, pc.isochrone_mass)
    model = im.IsochroneMassModel(cfg=pc.isochrone_mass, cmap=im.build_cmd_map(pts, pc.isochrone_mass.cmd_map),
                                  grid_key=im.native_grid_key(pc.isochrone_mass))
    n = cfg.sizes[size] if isinstance(size, str) else int(size)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=(418, n)))
    uc = make_cmd_universe(n, base, cfg, frag.target_mds17, grid1d, native, pts, rng)
    pipe = run_pipeline_on_parent(uc, base, cfg, model, gcfg, rng)
    if single_star_model is not None:
        cmcfg = cmcfg.model_copy(update={"single_star_density": cmcfg.single_star_density.model_copy(
            update={"provisional_model": single_star_model})})
    sampler = im.PosteriorSampler.build(pts, pc.isochrone_mass.cmd_map) if cfg.posterior_deblending else None
    mock = run_cmd_mock(pipe, base, cfg, frag, mcfg, cmcfg, grid1d, native, model.cmap, pipeline_config=pc, sampler=sampler)
    u, par = uc.base, pipe.par
    pi = par.index
    use = pipe.parent.usable
    has = u.comp["has_companion"][pi]
    n_rows = pi.size
    tr = mock.truth
    row = np.asarray(tr["parent_row"], np.int64)
    an = base.analysis
    prow = {k: mcl._row_sum(w, row, n_rows) for k, w in mock.w.items()}
    # unusable rows (no isochrone M1) are dropped from every comparison
    keep = use
    tables: dict[str, dict[str, Any]] = {}
    tables["binary_by_g"] = _cf(has[keep], u.g_true[pi][keep], {k: v[keep] for k, v in prow.items()}, np.asarray(an.g_bins, float))
    tables["binary_by_m1_hat"] = _cf(has[keep], pipe.parent.m1_msun[keep], {k: v[keep] for k, v in prow.items()}, np.asarray(an.m1_bins, float))
    colour_bins = np.array([0.3, 0.7, 0.9, 1.1, 1.4, 1.8, 2.5, 3.5])
    tables["binary_by_colour0"] = _cf(has[keep], pipe.colour0[keep], {k: v[keep] for k, v in prow.items()}, colour_bins)
    tot = _cf(has[keep], np.zeros(int(keep.sum())), {k: v[keep] for k, v in prow.items()}, np.array([-1.0, 1.0]))
    wk = {k: np.where(keep[row], v, 0.0) for k, v in mock.w.items()}
    hk = has & keep
    lf_mock = np.asarray(tr["log10_flux_ratio"], float)
    q_mock = np.asarray(tr["m2_msun"], float) / np.asarray(tr["m1_msun"], float)
    tables["log_p"] = _cc(u.comp["log_p"][pi], hk, np.log10(np.asarray(tr["period_days"], float)), row, wk, np.asarray(an.log_p_bins, float), n_rows)
    tables["q"] = _cc(u.comp["q"][pi], hk, q_mock, row, wk, np.asarray(an.q_bins, float), n_rows)
    tables["m2"] = _cc(u.comp["m2_msun"][pi], hk, np.asarray(tr["m2_msun"], float), row, wk, np.array([0.03, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.5]), n_rows)
    tables["eccentricity"] = _cc(u.comp["eccentricity"][pi], hk, np.asarray(tr["eccentricity"], float), row, wk, np.asarray(an.e_bins, float), n_rows)
    tables["log10_f"] = _cc(u.comp["log10_f"][pi], hk, lf_mock, row, wk, np.asarray(an.log_f_bins, float), n_rows)
    a_true = mcl.alpha0_mas(u.m1[pi], u.comp["m2_msun"][pi], 10.0 ** u.comp["log_p"][pi], u.comp["log10_f"][pi], 1000.0 / u.pos["distance_pc"][pi])
    a_mock = mcl.alpha0_mas(np.asarray(tr["m1_msun"], float), np.asarray(tr["m2_msun"], float), np.asarray(tr["period_days"], float), lf_mock, np.asarray(tr["parallax_mas"], float))
    tables["alpha0"] = _cc(a_true, hk, a_mock, row, wk, np.asarray(an.alpha0_bins_mas, float), n_rows)
    win = an.nss_window
    p_true = 10.0 ** u.comp["log_p"][pi]
    t_in = hk & (a_true > win.alpha0_min_mas) & (p_true > win.period_min_days) & (p_true < win.period_max_days)
    pm = np.asarray(tr["period_days"], float)
    m_in = (a_mock > win.alpha0_min_mas) & (pm > win.period_min_days) & (pm < win.period_max_days)
    tables["nss_window"] = _cc(np.where(t_in, 0.5, np.nan), t_in, np.where(m_in, 0.5, np.nan), row, wk, np.array([0.0, 1.0]), n_rows)
    # The same statistics on the rows W actually acts on (CMD non-evolved; evolved rows get W = 1).
    dk = use & ~pipe.evolved
    tables["dwarf_rows_binary_by_g"] = _cf(has[dk], u.g_true[pi][dk], {k: v[dk] for k, v in prow.items()}, np.asarray(an.g_bins, float))
    wd = {k: np.where(dk[row], v, 0.0) for k, v in mock.w.items()}
    tables["dwarf_rows_log10_f"] = _cc(u.comp["log10_f"][pi], has & dk, lf_mock, row, wd, np.asarray(an.log_f_bins, float), n_rows)
    tables["dwarf_rows_q"] = _cc(u.comp["q"][pi], has & dk, q_mock, row, wd, np.asarray(an.q_bins, float), n_rows)
    dtot = _cf(has[dk], np.zeros(int(dk.sum())), {k: v[dk] for k, v in prow.items()}, np.array([-1.0, 1.0]))
    tables["dwarf_rows_total"] = dtot
    total = {"truth": tot["truth"][0], "n_rows": float(keep.sum())}
    for k in ("none", "one_d", "two_d"):
        total[k] = tot[k][0]
        total[f"pull_{k}"] = tot[f"pull_{k}"][0]
    m1t = u.m1[pi][keep]
    m1h = pipe.parent.m1_msun[keep]
    lr = np.log10(m1h / m1t)
    single = ~has[keep]
    twin = has[keep] & (u.comp["q"][pi][keep] > 0.95)
    m1stats = {
        "usable_rows": float(keep.sum()), "rows": float(n_rows),
        "single_bias_dex": float(np.median(lr[single])), "single_scatter_dex": float(1.4826 * np.median(np.abs(lr[single] - np.median(lr[single])))),
        "binary_bias_dex": float(np.median(lr[has[keep]])), "twin_bias_dex": float(np.median(lr[twin])) if twin.any() else float("nan"),
        "evolved_flagged": float(np.sum(pipe.evolved & keep)),
    }
    ess = {k: ps.kish_ess(np.where(keep[row], v, 0.0)) for k, v in mock.w.items()}
    res = CmdClosedLoopResult(size=str(size), counts=par.counts, unit_counts=mock.unit_counts, m1_hat_vs_true=m1stats,
                              tables=tables, total=total, ess=ess)
    raw = {"universe": uc, "pipeline": pipe, "mock": mock}
    return res, raw


def max_abs_pull(res: CmdClosedLoopResult, key: str, stats: tuple[str, ...] = ("binary_by_g", "binary_by_m1_hat", "binary_by_colour0", "log10_f", "log_p")) -> float:
    """Largest |pull| over the parent statistics that the weight is meant to fix."""
    vals = [abs(x) for s in stats for x in res.tables[s][f"pull_{key}"] if np.isfinite(x)]
    vals.append(abs(res.total[f"pull_{key}"]))
    return float(max(vals))
