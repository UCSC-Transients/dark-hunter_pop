"""Closed-loop proof of the magnitude-limit conditioning weight (spec §9.6, #405).

A synthetic universe with known binary statistics is observed with the parent cuts
(G_total < 19, ϖ_obs > 0.2 mas, optional sky-dependent completeness). Its parent is then run
through the Gaia-star-primary pipeline logic exactly as real ``gaia_source`` rows would be:
:func:`darkhunter_pop.proposal_set.sample_proposal` draws companions, the target is
:func:`darkhunter_pop.proposal_set.mds17_luminous_log_intensity`, the weights are
:func:`darkhunter_pop.proposal_set.importance_weights`, with and without the
:mod:`darkhunter_pop.malmquist` factor. No gaiamock: this tests population bookkeeping only.

The universe: an exponential disk (no Galaxia), a Kroupa primary IMF, Janssens et al. (2022)
M_G(M) with intrinsic scatter, MdS17 luminous companions drawn from the **same** table and
provisional settings as the target, no extinction. Distances come from a geometric posterior
with the true density as prior, summarized as r_lo / r_med / r_hi like Bailer-Jones et al.
(2021). The parent-side M1 is an atmosphere-like estimate M1 × 10^N(0, σ).

Statistics compare (i) companion statistics of the parent and, after 1/V_S, of the volume-
limited universe, and (ii) observed system properties (photocentre semi-major axis α0 and the
count in an NSS-like (α0, P) window). Parent count statistics carry a pull
Σ_s d_s / sqrt(N var(d_s)), with d_s = (mock expectation − truth) for row s, which includes both
the per-row Bernoulli scatter of the truth and the Monte Carlo noise of the mock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import ndtr

from darkhunter_pop import constants
from darkhunter_pop import malmquist as mq
from darkhunter_pop import moe_distefano as mds
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import repo_root

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiskConfig(_Strict):
    r0_pc: float = Field(..., gt=0)
    scale_length_pc: float = Field(..., gt=0)
    scale_height_pc: float = Field(..., gt=0)
    r_min_pc: float = Field(..., gt=0)
    r_max_pc: float = Field(..., gt=0)


class ImfConfig(_Strict):
    m_min_msun: float = Field(..., gt=0)
    m_break_msun: float = Field(..., gt=0)
    m_max_msun: float = Field(..., gt=0)
    alpha_low: float
    alpha_high: float


class StarsConfig(_Strict):
    sigma_int_mag: float = Field(..., ge=0)
    sigma_log_m1_dex: float = Field(..., ge=0)


class ParallaxErrorConfig(_Strict):
    floor_mas: float = Field(..., gt=0)
    amp_mas: float = Field(..., ge=0)
    slope: float


class CompletenessConfig(_Strict):
    enabled: bool
    g50_high_latitude: float
    crowding_depth_mag: float = Field(..., ge=0)
    crowding_scale_deg: float = Field(..., gt=0)
    width_mag: float = Field(..., gt=0)


class DistancePosteriorConfig(_Strict):
    r_grid_min_pc: float = Field(..., gt=0)
    r_grid_max_pc: float = Field(..., gt=0)
    n_grid: int = Field(..., ge=50)


class ObservationConfig(_Strict):
    g_limit: float
    parallax_floor_mas: float
    parallax_error: ParallaxErrorConfig
    completeness: CompletenessConfig
    distance_posterior: DistancePosteriorConfig


class ProposalOverrides(_Strict):
    """Closed-loop proposal knobs (efficiency only; spec §3.2)."""

    parent_uniform_fraction: float = Field(..., ge=0, le=1)
    dark_fraction: float = Field(..., ge=0, le=1)
    relation_sigma_dex: float = Field(..., gt=0)
    relation_weight: float = Field(..., ge=0, le=1)
    e_cap: float = Field(..., gt=0, lt=1)
    period_core_weight: float = Field(..., ge=0, le=1)
    q_tied_weight: float = Field(..., ge=0, le=1)
    q_min: float = Field(..., gt=0)


class TargetOverrides(_Strict):
    provisional_eta_floor: float = Field(..., gt=-1.0)


class MockConfig(_Strict):
    draws_per_row: int = Field(..., ge=1)
    proposal_generation: int = Field(..., ge=0)
    proposal_overrides: ProposalOverrides


class NssWindowConfig(_Strict):
    alpha0_min_mas: float
    period_min_days: float
    period_max_days: float


class AnalysisConfig(_Strict):
    g_bins: list[float]
    m1_bins: list[float]
    log_p_bins: list[float]
    q_bins: list[float]
    e_bins: list[float]
    log_f_bins: list[float]
    alpha0_bins_mas: list[float]
    nss_window: NssWindowConfig
    m_abs_grid: tuple[float, float, int]
    n_volume_positions: int = Field(..., ge=1000)
    n_bootstrap: int = Field(..., ge=10)


class ClosedLoopConfig(_Strict):
    target_fragment: str
    malmquist_config: str
    seed: int
    sizes: dict[str, int]
    disk: DiskConfig
    imf: ImfConfig
    stars: StarsConfig
    observation: ObservationConfig
    target_overrides: TargetOverrides
    mock: MockConfig
    analysis: AnalysisConfig


def load_closed_loop_config(path: str | Path = "config/population/malmquist_closed_loop.yaml") -> ClosedLoopConfig:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return ClosedLoopConfig.model_validate(yaml.safe_load(p.read_text())["closed_loop"])


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def disk_density(x: FloatArray, y: FloatArray, z: FloatArray, disk: DiskConfig) -> FloatArray:
    """Relative density at heliocentric (x toward GC, y, z) for the exponential disk."""
    big_r = np.hypot(disk.r0_pc - x, y)
    return np.exp(-(big_r - disk.r0_pc) / disk.scale_length_pc - np.abs(z) / disk.scale_height_pc)


def draw_positions(n: int, disk: DiskConfig, rng: np.random.Generator) -> dict[str, FloatArray]:
    """n positions from the disk density inside r_min < r < r_max (rejection, chunked)."""
    out: list[FloatArray] = []
    have = 0
    rho_max = math.exp(disk.r_max_pc / disk.scale_length_pc)
    while have < n:
        m = max(4 * (n - have), 10_000)
        # Uniform in the plane disk of radius r_max, Laplace in z (the vertical profile).
        rad = disk.r_max_pc * np.sqrt(rng.uniform(size=m))
        phi = rng.uniform(0, 2 * np.pi, size=m)
        x, y = rad * np.cos(phi), rad * np.sin(phi)
        z = rng.laplace(0.0, disk.scale_height_pc, size=m)
        r = np.sqrt(x * x + y * y + z * z)
        big_r = np.hypot(disk.r0_pc - x, y)
        acc = rng.uniform(size=m) < np.exp(-(big_r - disk.r0_pc) / disk.scale_length_pc) / rho_max
        acc &= (r > disk.r_min_pc) & (r < disk.r_max_pc)
        out.append(np.stack([x[acc], y[acc], z[acc]]))
        have += int(acc.sum())
    xyz = np.concatenate(out, axis=1)[:, :n]
    x, y, z = xyz
    d = np.sqrt(x * x + y * y + z * z)
    return {
        "x": x,
        "y": y,
        "z": z,
        "distance_pc": d,
        "l_deg": np.degrees(np.arctan2(y, x)) % 360.0,
        "b_deg": np.degrees(np.arcsin(z / d)),
    }


def draw_imf(n: int, imf: ImfConfig, rng: np.random.Generator) -> FloatArray:
    """Broken power law dN/dM ∝ M^-α (continuous at the break) by inverse CDF."""

    def seg(lo: float, hi: float, a: float) -> float:
        g = 1.0 - a
        return (hi**g - lo**g) / g if abs(g) > 1e-12 else math.log(hi / lo)

    lo, br, hi = imf.m_min_msun, imf.m_break_msun, imf.m_max_msun
    c_high = br ** (imf.alpha_high - imf.alpha_low)  # continuity at the break
    n_low = seg(lo, br, imf.alpha_low)
    n_high = c_high * seg(br, hi, imf.alpha_high)
    is_low = rng.uniform(size=n) < n_low / (n_low + n_high)
    u = rng.uniform(size=n)

    def inv(lo_: float, hi_: float, a: float, uu: FloatArray) -> FloatArray:
        g = 1.0 - a
        return (lo_**g + uu * (hi_**g - lo_**g)) ** (1.0 / g)

    return np.where(is_low, inv(lo, br, imf.alpha_low, u), inv(br, hi, imf.alpha_high, u))


def _envelope(grid: mq.FluxMarginalGrid, target: ps.MdS17TargetConfig, n_q: int = 400, n_p: int = 400) -> FloatArray:
    """max over (log q, log P) of λ_{q,P}(M1) at every grid M1 (for rejection sampling)."""
    table = mds.load_mds17_table(target.table_path)
    lq = np.linspace(math.log10(table.q_range[0]), 0.0, n_q)
    lp = np.linspace(table.log_p_range[0], table.log_p_range[1], n_p)
    LQ, LP = np.meshgrid(lq, lp, indexing="ij")
    return np.array(
        [float(np.nanmax(mq.mds17_luminous_m2_p_intensity(10.0**m, LQ, LP, target))) for m in grid.log_m1]
    )


def draw_companions(
    m1: FloatArray,
    target: ps.MdS17TargetConfig,
    grid: mq.FluxMarginalGrid,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """At most one luminous MdS17 companion per primary (binary with probability F_lum(M1)).

    (log q, log P) by rejection from λ_{q,P}(M1) (the exact target factors); e by inverse CDF
    of e^η on [0, e_max); log10 f = log10 f_J(M1, M2) + σ_f N(0, 1). Draws whose M2 lies
    outside the Janssens range are rejected (the target is zero there, as in the grid).
    """
    table = mds.load_mds17_table(target.table_path)
    n = m1.size
    _, f_lum = grid.interpolate(m1)
    if np.any(f_lum > 1.0):
        raise ValueError("F_lum > 1 inside the synthetic IMF range (MP-Q31); narrow the IMF")
    has = rng.uniform(size=n) < f_lum
    env_grid = _envelope(grid, target) * 1.3
    j = np.clip(np.searchsorted(grid.log_m1, np.log10(m1), side="right") - 1, 0, grid.log_m1.size - 2)
    env = np.maximum(env_grid[j], env_grid[j + 1])
    lq_lo, lp_lo, lp_hi = math.log10(table.q_range[0]), table.log_p_range[0], table.log_p_range[1]
    log_q = np.full(n, np.nan)
    log_p = np.full(n, np.nan)
    todo = np.flatnonzero(has)
    n_over = 0
    while todo.size:
        cq = rng.uniform(lq_lo, 0.0, size=todo.size)
        cp = rng.uniform(lp_lo, lp_hi, size=todo.size)
        lam = mq.mds17_luminous_m2_p_intensity(m1[todo], cq, cp, target)
        rel = ps.relation_log10_flux_ratio(m1[todo], m1[todo] * 10.0**cq)
        n_over += int(np.sum(lam > env[todo]))
        ok = (rng.uniform(size=todo.size) * env[todo] < lam) & np.isfinite(rel)
        log_q[todo[ok]] = cq[ok]
        log_p[todo[ok]] = cp[ok]
        todo = todo[~ok]
    if n_over:
        raise RuntimeError(f"rejection envelope exceeded {n_over} times; raise its margin")
    q = 10.0**log_q
    period = 10.0**log_p
    m1_shape = np.clip(m1, table.m1_range[0], table.m1_range[1])
    with np.errstate(invalid="ignore"):
        et = mds.eta(
            m1_shape,
            log_p,
            table,
            m1_interpolation=target.provisional_m1_interpolation,
            eta_floor=target.provisional_eta_floor,
        )
        emax = mds.e_max(period, table)
        ecc = np.where(
            has & ~mds.is_circular(period, table),
            emax * rng.uniform(size=n) ** (1.0 / (et + 1.0)),
            0.0,
        )
    rel = ps.relation_log10_flux_ratio(m1, m1 * q)
    log_f = np.where(has, rel + target.flux_sigma_dex * rng.standard_normal(n), -np.inf)
    return {
        "has_companion": has,
        "q": np.where(has, q, np.nan),
        "m2_msun": np.where(has, m1 * q, np.nan),
        "log_p": np.where(has, log_p, np.nan),
        "eccentricity": np.where(has, ecc, np.nan),
        "log10_f": log_f,
    }


@dataclass
class Universe:
    """All synthetic primaries (truth), positions and true system magnitudes."""

    m1: FloatArray
    eps: FloatArray
    comp: dict[str, Any]
    pos: dict[str, FloatArray]
    g_true: FloatArray

    @property
    def n(self) -> int:
        return int(self.m1.size)


def make_universe(
    n: int,
    cfg: ClosedLoopConfig,
    target: ps.MdS17TargetConfig,
    grid: mq.FluxMarginalGrid,
    rng: np.random.Generator,
) -> Universe:
    pos = draw_positions(n, cfg.disk, rng)
    m1 = draw_imf(n, cfg.imf, rng)
    eps = cfg.stars.sigma_int_mag * rng.standard_normal(n)
    comp = draw_companions(m1, target, grid, rng)
    f = np.where(comp["has_companion"], 10.0 ** comp["log10_f"], 0.0)
    g = ps.janssens_absolute_g(m1) + eps - 2.5 * np.log10(1.0 + f) + mq.distance_modulus(pos["distance_pc"])
    return Universe(m1=m1, eps=eps, comp=comp, pos=pos, g_true=g)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def parallax_error(g: FloatArray, pe: ParallaxErrorConfig, g_ref: float) -> FloatArray:
    return pe.floor_mas + pe.amp_mas * 10.0 ** (pe.slope * (np.asarray(g, float) - g_ref))


def completeness(g: FloatArray, b_deg: FloatArray, c: CompletenessConfig) -> FloatArray:
    """Synthetic sky-dependent detection probability (crowding toward the plane)."""
    if not c.enabled:
        return np.ones(np.broadcast(g, b_deg).shape)
    g50 = c.g50_high_latitude - c.crowding_depth_mag * np.exp(-np.abs(b_deg) / c.crowding_scale_deg)
    return 1.0 / (1.0 + np.exp((g - g50) / c.width_mag))


def geometric_distance_quantiles(
    plx_obs: FloatArray,
    sigma_plx: FloatArray,
    l_deg: FloatArray,
    b_deg: FloatArray,
    disk: DiskConfig,
    dp: DistancePosteriorConfig,
    chunk: int = 5000,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """16/50/84th percentiles of p(r | ϖ_obs, l, b) ∝ r² ρ(r, l, b) N(ϖ_obs; 1000/r, σ_ϖ).

    The Bailer-Jones et al. (2021) *geometric* construction (no photometry) with the true
    synthetic density as the prior, so spec §9.3 (A2)'s prior assumption holds by construction.
    """
    r = np.geomspace(dp.r_grid_min_pc, dp.r_grid_max_pc, dp.n_grid)
    n = plx_obs.size
    out = np.empty((3, n))
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        lr, br = np.radians(l_deg[sl])[:, None], np.radians(b_deg[sl])[:, None]
        x = r[None, :] * np.cos(br) * np.cos(lr)
        y = r[None, :] * np.cos(br) * np.sin(lr)
        z = r[None, :] * np.sin(br)
        prior = r[None, :] ** 3 * disk_density(x, y, z, disk)  # r^2 dr = r^3 dln r on a log grid
        prior = np.where(r[None, :] < disk.r_max_pc, prior, 0.0)
        like = np.exp(-0.5 * ((plx_obs[sl, None] - 1000.0 / r[None, :]) / sigma_plx[sl, None]) ** 2)
        post = prior * like
        cdf = np.cumsum(post, axis=1)
        cdf /= cdf[:, -1:]
        lnr = np.log(r)
        for k, qv in enumerate((0.158655, 0.5, 0.841345)):
            j = np.clip((cdf < qv).sum(axis=1), 1, r.size - 1)
            c0 = np.take_along_axis(cdf, (j - 1)[:, None], 1)[:, 0]
            c1 = np.take_along_axis(cdf, j[:, None], 1)[:, 0]
            t = np.clip((qv - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0, 1.0)
            out[k, sl] = np.exp(lnr[j - 1] + t * (lnr[j] - lnr[j - 1]))
    return out[0], out[1], out[2]


@dataclass
class SyntheticParent:
    """The synthetic parent: universe indices plus the observed quantities."""

    index: NDArray[np.int64]
    plx_obs: FloatArray
    m1_hat: FloatArray
    r_lo: FloatArray
    r_med: FloatArray
    r_hi: FloatArray
    counts: dict[str, int] = field(default_factory=dict)


def observe(u: Universe, cfg: ClosedLoopConfig, rng: np.random.Generator) -> SyntheticParent:
    """Apply G < g_limit, ϖ_obs > floor and completeness; derive the parent observables."""
    ob = cfg.observation
    plx_true = 1000.0 / u.pos["distance_pc"]
    s_plx = parallax_error(u.g_true, ob.parallax_error, ob.g_limit)
    plx_obs = plx_true + s_plx * rng.standard_normal(u.n)
    det = rng.uniform(size=u.n) < completeness(u.g_true, u.pos["b_deg"], ob.completeness)
    g_cut = u.g_true < ob.g_limit
    p_cut = plx_obs > ob.parallax_floor_mas
    sel = g_cut & p_cut & det
    idx = np.flatnonzero(sel)
    m1_hat = u.m1[idx] * 10.0 ** (cfg.stars.sigma_log_m1_dex * rng.standard_normal(idx.size))
    r_lo, r_med, r_hi = geometric_distance_quantiles(
        plx_obs[idx], s_plx[idx], u.pos["l_deg"][idx], u.pos["b_deg"][idx], cfg.disk, ob.distance_posterior
    )
    return SyntheticParent(
        index=idx,
        plx_obs=plx_obs[idx],
        m1_hat=m1_hat,
        r_lo=r_lo,
        r_med=r_med,
        r_hi=r_hi,
        counts={
            "universe": u.n,
            "g_limit": int(g_cut.sum()),
            "g_limit_and_parallax_floor": int((g_cut & p_cut).sum()),
            "parent": int(sel.sum()),
        },
    )


def parent_snapshot(u: Universe, par: SyntheticParent) -> ps.ParentSnapshot:
    """The synthetic parent dressed as ``gaia_source`` rows (spec §1, §0.1)."""
    i = par.index
    n = i.size
    cols: dict[str, NDArray[Any]] = {
        "source_id": i.astype(np.int64),
        "ra": u.pos["l_deg"][i],
        "dec": u.pos["b_deg"][i],
        "l": u.pos["l_deg"][i],
        "b": u.pos["b_deg"][i],
        "parallax": par.plx_obs,
        "pmra": np.zeros(n),
        "pmdec": np.zeros(n),
        "phot_g_mean_mag": u.g_true[i],
        "r_lo_geo": par.r_lo,
        "r_med_geo": par.r_med,
        "r_hi_geo": par.r_hi,
    }
    usable = np.ones(n, dtype=bool)
    return ps.ParentSnapshot(
        columns=cols,
        m1_msun=par.m1_hat,
        m1_source=np.full(n, "SYNTH"),
        atmosphere_logg=np.full(n, np.nan),
        truth_parallax_mas=1000.0 / par.r_med,
        is_giant=np.zeros(n, dtype=bool),
        flags={"synthetic_parent": usable},
        usable=usable,
        meta={"parent_h5_sha256": "synthetic", "random_index_max_exclusive": n, "gaia_source_total_rows": n},
        scale_to_full=1.0,
        path=Path("synthetic"),
    )


# ---------------------------------------------------------------------------
# Mock (the pipeline under test)
# ---------------------------------------------------------------------------


def closed_loop_proposal(frag: ps.ProposalSetFragment, cfg: ClosedLoopConfig, n_rows: int) -> ps.ProposalConfig:
    """The fragment's proposal with the closed-loop efficiency overrides applied."""
    p = frag.proposal
    o = cfg.mock.proposal_overrides
    return p.model_copy(
        update={
            "n_draws": cfg.mock.draws_per_row * n_rows,
            "generation": cfg.mock.proposal_generation,
            "base_seed": cfg.seed,
            "parent": p.parent.model_copy(update={"uniform_fraction": o.parent_uniform_fraction}),
            "flux": p.flux.model_copy(
                update={
                    "dark_fraction": o.dark_fraction,
                    "relation_sigma_dex": o.relation_sigma_dex,
                    "relation_weight": o.relation_weight,
                }
            ),
            "eccentricity": p.eccentricity.model_copy(update={"e_cap": o.e_cap}),
            "period": p.period.model_copy(update={"core_weight": o.period_core_weight}),
            "m2": p.m2.model_copy(update={"q_tied_weight": o.q_tied_weight, "q_min": o.q_min}),
        }
    )


def closed_loop_fragment(cfg: ClosedLoopConfig) -> ps.ProposalSetFragment:
    """The target fragment with ``target_overrides`` applied (universe and mock alike)."""
    frag = ps.load_proposal_set_fragment(cfg.target_fragment)
    tgt = frag.target_mds17.model_copy(update=cfg.target_overrides.model_dump())
    return frag.model_copy(update={"target_mds17": tgt})


@dataclass
class MockResult:
    truth: dict[str, NDArray[Any]]
    w_naive: FloatArray
    w_corr: FloatArray
    p_single_naive: FloatArray
    p_single_corr: FloatArray
    rows: mq.RowConditioning


def run_mock(
    parent: ps.ParentSnapshot,
    cfg: ClosedLoopConfig,
    frag: ps.ProposalSetFragment,
    mcfg: mq.MalmquistConfig,
    grid: mq.FluxMarginalGrid,
    *,
    use_rows: mq.RowConditioning | None = None,
) -> MockResult:
    """Draw companions for every parent row and weight them with and without W (spec §9.3)."""
    prop = closed_loop_proposal(frag, cfg, parent.n_rows)
    truth = ps.sample_proposal(parent, prop)
    log_lam = ps.mds17_luminous_log_intensity(truth, frag.target_mds17)
    rows = use_rows or mq.row_conditioning(parent, mcfg, a_g_mag=np.zeros(parent.n_rows), sigma_a_mag=0.0)
    log_w = mq.log_weight_for_draws(truth, rows, grid, mcfg)
    n = [prop.n_draws]
    w_naive = ps.importance_weights(log_lam, [truth["log_q_total"]], n, scale_to_full=parent.scale_to_full)
    w_corr = ps.importance_weights(log_lam + log_w, [truth["log_q_total"]], n, scale_to_full=parent.scale_to_full)
    lp_single, _ = mq.log_no_companion_probability(rows.m1_msun, rows.delta_m, rows.sigma, grid)
    _, f_lum = grid.interpolate(rows.m1_msun)
    return MockResult(
        truth=truth,
        w_naive=w_naive,
        w_corr=w_corr,
        p_single_naive=np.clip(1.0 - f_lum, 0.0, 1.0),
        p_single_corr=np.exp(lp_single),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def alpha0_mas(m1: FloatArray, m2: FloatArray, period_days: FloatArray, log10_f: FloatArray, plx_mas: FloatArray) -> FloatArray:
    """Photocentre semi-major axis a ϖ |q/(1+q) − f/(1+f)| (Kepler's third law, AU, yr, Msun)."""
    a_au = ((m1 + m2) * (period_days / constants.JULIAN_YEAR_DAYS) ** 2) ** (1.0 / 3.0)
    q = m2 / m1
    f = np.where(np.isfinite(log10_f), 10.0**log10_f, 0.0)
    return a_au * plx_mas * np.abs(q / (1.0 + q) - f / (1.0 + f))


@dataclass(frozen=True)
class CountComparison:
    """Per-bin expected count from the mock vs the realized truth, with the pull."""

    edges: FloatArray
    truth: FloatArray
    naive: FloatArray
    corrected: FloatArray
    pull_naive: FloatArray
    pull_corrected: FloatArray


def _row_sum(values: FloatArray, row: NDArray[np.int64], n_rows: int) -> FloatArray:
    return np.bincount(row, weights=values, minlength=n_rows)


def compare_counts(
    truth_value: FloatArray,
    truth_has: NDArray[np.bool_],
    mock_value: FloatArray,
    mock_row: NDArray[np.int64],
    w_naive: FloatArray,
    w_corr: FloatArray,
    edges: FloatArray,
    n_rows: int,
) -> CountComparison:
    """Expected counts per bin of a companion property over parent rows, with row-level pulls.

    For row s and bin b: d_s = Σ_{draws of s in b} w − 1[s's true companion in b]; the pull is
    Σ_s d_s / sqrt(N var_s(d_s)) (rows are i.i.d. draws of the parent).
    """
    nb = edges.size - 1
    tb = np.where(truth_has, np.searchsorted(edges, truth_value, side="right") - 1, -1)
    mb = np.searchsorted(edges, mock_value, side="right") - 1
    out = {k: np.zeros(nb) for k in ("truth", "naive", "corrected", "pull_naive", "pull_corrected")}
    for b in range(nb):
        t_row = (tb == b).astype(float)  # one entry per row
        inb = (mb == b) & np.isfinite(mock_value)
        out["truth"][b] = t_row.sum()
        for key, w in (("naive", w_naive), ("corrected", w_corr)):
            m_row = _row_sum(np.where(inb, w, 0.0), mock_row, n_rows)
            d = m_row - t_row
            out[key][b] = m_row.sum()
            sd = math.sqrt(n_rows * float(np.var(d))) if n_rows > 1 else float("nan")
            out[f"pull_{key}"][b] = d.sum() / sd if sd > 0 else 0.0
    return CountComparison(edges=np.asarray(edges, float), **out)


def compare_row_fraction(
    truth_flag: NDArray[np.bool_],
    group_value: FloatArray,
    p_naive: FloatArray,
    p_corr: FloatArray,
    edges: FloatArray,
) -> CountComparison:
    """Per-group counts of a per-row probability (e.g. P(has companion)) vs truth flags."""
    nb = edges.size - 1
    gb = np.searchsorted(edges, group_value, side="right") - 1
    out = {k: np.zeros(nb) for k in ("truth", "naive", "corrected", "pull_naive", "pull_corrected")}
    for b in range(nb):
        sel = gb == b
        n = int(sel.sum())
        t = truth_flag[sel].astype(float)
        out["truth"][b] = t.sum()
        for key, p in (("naive", p_naive), ("corrected", p_corr)):
            d = p[sel] - t
            out[key][b] = p[sel].sum()
            sd = math.sqrt(n * float(np.var(d))) if n > 1 else float("nan")
            out[f"pull_{key}"][b] = d.sum() / sd if sd > 0 else 0.0
    return CountComparison(edges=np.asarray(edges, float), **out)


@dataclass
class VolumeComparison:
    """Volume-limited binary fraction per M1 bin and shape comparisons (spec §9.5)."""

    m1_edges: FloatArray
    truth_fraction: FloatArray
    truth_fraction_err: FloatArray
    parent_truth_fraction: FloatArray
    naive_fraction: FloatArray
    naive_fraction_err: FloatArray
    corrected_fraction: FloatArray
    corrected_fraction_err: FloatArray
    shapes: dict[str, dict[str, FloatArray]]


def selection_probability_factory(
    u: Universe, cfg: ClosedLoopConfig, pos_idx: NDArray[np.int64]
) -> Any:
    """P(S | G, position) for the synthetic observation, over a subset of universe positions."""
    ob = cfg.observation
    d = u.pos["distance_pc"][pos_idx]
    b = u.pos["b_deg"][pos_idx]

    def p_select(g: FloatArray, idx: NDArray[np.int64]) -> FloatArray:
        s_plx = parallax_error(g, ob.parallax_error, ob.g_limit)
        p_plx = ndtr((1000.0 / d[idx][None, :] - ob.parallax_floor_mas) / s_plx)
        return (g < ob.g_limit) * p_plx * completeness(g, b[idx][None, :], ob.completeness)

    return d, p_select


def volume_comparison(
    u: Universe,
    par: SyntheticParent,
    mock: MockResult,
    cfg: ClosedLoopConfig,
    mcfg: mq.MalmquistConfig,
    rng: np.random.Generator,
) -> VolumeComparison:
    """1/V_S inversion of the parent and the mock back to a volume-limited population."""
    an = cfg.analysis
    pos_idx = rng.choice(u.n, size=min(an.n_volume_positions, u.n), replace=False)
    d, p_sel = selection_probability_factory(u, cfg, pos_idx)
    m_grid = np.linspace(an.m_abs_grid[0], an.m_abs_grid[1], int(an.m_abs_grid[2]))
    v_tab = mq.tabulate_selection_volume(m_grid, d, p_sel)
    floor = 1e-12 * v_tab.max()

    def vol(m_abs: FloatArray, sigma: float | FloatArray) -> FloatArray:
        s = np.broadcast_to(np.asarray(sigma, float), np.shape(m_abs))
        nodes, hw = np.polynomial.hermite_e.hermegauss(9)
        hw = hw / hw.sum()
        vals = np.interp(np.asarray(m_abs)[..., None] + s[..., None] * nodes, m_grid, v_tab)
        return np.maximum(np.sum(vals * hw, axis=-1), floor)

    m_edges = np.asarray(an.m1_bins, float)
    # Truth (the injected universe), by true M1.
    ub = np.searchsorted(m_edges, u.m1, side="right") - 1
    has = u.comp["has_companion"]
    nb = m_edges.size - 1
    tf = np.array([has[ub == k].mean() if np.any(ub == k) else np.nan for k in range(nb)])
    tf_err = np.array([math.sqrt(tf[k] * (1 - tf[k]) / max(int(np.sum(ub == k)), 1)) for k in range(nb)])
    # Parent truth inverted with exact per-system V (validates V_S itself).
    pi = par.index
    f_true = np.where(has[pi], 10.0 ** u.comp["log10_f"][pi], 0.0)
    m_sys_true = ps.janssens_absolute_g(u.m1[pi]) + u.eps[pi] - 2.5 * np.log10(1.0 + f_true)
    inv_v_true = 1.0 / vol(m_sys_true, 0.0)
    pb = np.searchsorted(m_edges, u.m1[pi], side="right") - 1
    ptf = np.array([inv_v_true[(pb == k) & has[pi]].sum() / inv_v_true[pb == k].sum() for k in range(nb)])
    # Mock: per draw and per row, using M̂1 and the row σ (intrinsic + M1 scatter).
    tr = mock.truth
    row = np.asarray(tr["parent_row"], np.int64)
    rows = mock.rows
    sig_m = np.sqrt(np.maximum(rows.sigma**2 - sigma_mu_rows(par) ** 2, 0.0))
    m_single = ps.janssens_absolute_g(rows.m1_msun) + mcfg.provisional_mg_zero_point_mag
    lf = np.asarray(tr["log10_flux_ratio"], float)
    m_draw = m_single[row] - 2.5 * np.log10(1.0 + np.where(np.isfinite(lf), 10.0**lf, 0.0))
    inv_v_draw = 1.0 / vol(m_draw, sig_m[row])
    inv_v_single = 1.0 / vol(m_single, sig_m)
    rb = np.searchsorted(m_edges, rows.m1_msun, side="right") - 1
    n_rows = rows.m1_msun.size

    def frac(w: FloatArray, p_single: FloatArray, boot: FloatArray | None = None) -> FloatArray:
        comp_row = _row_sum(w * inv_v_draw, row, n_rows)
        sing_row = p_single * inv_v_single
        res = np.zeros(nb)
        for k in range(nb):
            sel = rb == k
            if boot is None:
                c, s = comp_row[sel].sum(), sing_row[sel].sum()
            else:
                c, s = (boot[sel] * comp_row[sel]).sum(), (boot[sel] * sing_row[sel]).sum()
            res[k] = c / (c + s) if (c + s) > 0 else np.nan
        return res

    nf = frac(mock.w_naive, mock.p_single_naive)
    cf = frac(mock.w_corr, mock.p_single_corr)
    boots_n, boots_c = [], []
    for _ in range(an.n_bootstrap):
        bw = rng.poisson(1.0, size=n_rows).astype(float)
        boots_n.append(frac(mock.w_naive, mock.p_single_naive, bw))
        boots_c.append(frac(mock.w_corr, mock.p_single_corr, bw))
    nf_err = np.nanstd(np.array(boots_n), axis=0)
    cf_err = np.nanstd(np.array(boots_c), axis=0)
    # Shapes of companion properties, volume-limited (normalized densities).
    shapes: dict[str, dict[str, FloatArray]] = {}
    u_has = np.flatnonzero(has)
    defs = {
        "log_p": (u.comp["log_p"], np.log10(np.asarray(tr["period_days"], float)), an.log_p_bins),
        "q": (u.comp["q"], np.asarray(tr["m2_msun"], float) / np.asarray(tr["m1_msun"], float), an.q_bins),
        "eccentricity": (u.comp["eccentricity"], np.asarray(tr["eccentricity"], float), an.e_bins),
        "log10_f": (u.comp["log10_f"], lf, an.log_f_bins),
    }
    boot_w = rng.poisson(1.0, size=(an.n_bootstrap, n_rows)).astype(float)
    for name, (tv, mv, edges) in defs.items():
        e = np.asarray(edges, float)
        th, _ = np.histogram(tv[u_has], bins=e)
        mb = np.searchsorted(e, mv, side="right") - 1
        ok = np.isfinite(mv) & (mb >= 0) & (mb < e.size - 1)
        entry: dict[str, FloatArray] = {
            "edges": e,
            "truth": th / max(th.sum(), 1),
            "truth_err": np.sqrt(th) / max(th.sum(), 1),
        }
        for key, w in (("naive", mock.w_naive), ("corrected", mock.w_corr)):
            # rows x bins matrix of Σ w / V_S, so a row bootstrap is one matrix product.
            mat = np.zeros((n_rows, e.size - 1))
            np.add.at(mat, (row[ok], mb[ok]), (w * inv_v_draw)[ok])
            h = mat.sum(axis=0)
            entry[key] = h / max(h.sum(), 1e-300)
            hb = boot_w @ mat
            entry[f"{key}_err"] = np.std(hb / hb.sum(axis=1, keepdims=True), axis=0)
        shapes[name] = entry
    return VolumeComparison(
        m1_edges=m_edges,
        truth_fraction=tf,
        truth_fraction_err=tf_err,
        parent_truth_fraction=ptf,
        naive_fraction=nf,
        naive_fraction_err=nf_err,
        corrected_fraction=cf,
        corrected_fraction_err=cf_err,
        shapes=shapes,
    )


def sigma_mu_rows(par: SyntheticParent) -> FloatArray:
    return mq.sigma_mu_from_quantiles(par.r_lo, par.r_med, par.r_hi)


@dataclass
class ClosedLoopResult:
    size: str
    counts: dict[str, int]
    parent_binary_by_g: CountComparison
    parent_binary_by_m1: CountComparison
    parent_binary_total: dict[str, float]
    parent_shapes: dict[str, CountComparison]
    alpha0: CountComparison
    nss_window: dict[str, float]
    volume: VolumeComparison
    ess: dict[str, float]
    sigma_mu_quantiles: FloatArray


def run_closed_loop(size: Literal["small", "large"] | int, cfg: ClosedLoopConfig | None = None) -> tuple[ClosedLoopResult, dict[str, Any]]:
    """Build, observe, mock and compare. Returns the result and the raw arrays (for figures)."""
    cfg = cfg or load_closed_loop_config()
    frag = closed_loop_fragment(cfg)
    mcfg = mq.load_malmquist_config(cfg.malmquist_config)
    # The pipeline's provisional σ_int / σ_logM1 are set to the synthetic truth here: the loop
    # tests bookkeeping, not the MP-Q25 choice (a misspecified σ is a separate experiment).
    mcfg = mcfg.model_copy(
        update={
            "provisional_sigma_int_mag": cfg.stars.sigma_int_mag,
            "provisional_sigma_log_m1_dex": cfg.stars.sigma_log_m1_dex,
            "provisional_mg_zero_point_mag": 0.0,
        }
    )
    grid = mq.build_flux_marginal(frag.target_mds17, mcfg.grid)
    n = cfg.sizes[size] if isinstance(size, str) else int(size)
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=(405, n)))
    u = make_universe(n, cfg, frag.target_mds17, grid, rng)
    par = observe(u, cfg, rng)
    parent = parent_snapshot(u, par)
    mock = run_mock(parent, cfg, frag, mcfg, grid)
    an = cfg.analysis
    pi = par.index
    has = u.comp["has_companion"][pi]
    n_rows = pi.size
    tr = mock.truth
    row = np.asarray(tr["parent_row"], np.int64)
    p_comp_naive = _row_sum(mock.w_naive, row, n_rows)
    p_comp_corr = _row_sum(mock.w_corr, row, n_rows)
    by_g = compare_row_fraction(has, u.g_true[pi], p_comp_naive, p_comp_corr, np.asarray(an.g_bins, float))
    by_m1 = compare_row_fraction(has, par.m1_hat, p_comp_naive, p_comp_corr, np.asarray(an.m1_bins, float))
    tot = compare_row_fraction(has, np.zeros(n_rows), p_comp_naive, p_comp_corr, np.array([-1.0, 1.0]))
    total = {
        "truth": float(tot.truth[0]),
        "naive": float(tot.naive[0]),
        "corrected": float(tot.corrected[0]),
        "pull_naive": float(tot.pull_naive[0]),
        "pull_corrected": float(tot.pull_corrected[0]),
        "closure_corrected": float(np.mean(p_comp_corr + mock.p_single_corr)),
        "n_rows": float(n_rows),
    }
    lf_mock = np.asarray(tr["log10_flux_ratio"], float)
    q_mock = np.asarray(tr["m2_msun"], float) / np.asarray(tr["m1_msun"], float)
    shapes = {
        "log_p": compare_counts(u.comp["log_p"][pi], has, np.log10(np.asarray(tr["period_days"], float)), row, mock.w_naive, mock.w_corr, np.asarray(an.log_p_bins, float), n_rows),
        "q": compare_counts(u.comp["q"][pi], has, q_mock, row, mock.w_naive, mock.w_corr, np.asarray(an.q_bins, float), n_rows),
        "eccentricity": compare_counts(u.comp["eccentricity"][pi], has, np.asarray(tr["eccentricity"], float), row, mock.w_naive, mock.w_corr, np.asarray(an.e_bins, float), n_rows),
        "log10_f": compare_counts(u.comp["log10_f"][pi], has, lf_mock, row, mock.w_naive, mock.w_corr, np.asarray(an.log_f_bins, float), n_rows),
    }
    # Observed system properties: α0 with the true distance (truth) / the BJ-like one (mock).
    a_true = alpha0_mas(u.m1[pi], u.comp["m2_msun"][pi], 10.0 ** u.comp["log_p"][pi], u.comp["log10_f"][pi], 1000.0 / u.pos["distance_pc"][pi])
    a_mock = alpha0_mas(np.asarray(tr["m1_msun"], float), np.asarray(tr["m2_msun"], float), np.asarray(tr["period_days"], float), lf_mock, np.asarray(tr["parallax_mas"], float))
    alpha = compare_counts(a_true, has, a_mock, row, mock.w_naive, mock.w_corr, np.asarray(an.alpha0_bins_mas, float), n_rows)
    win = an.nss_window
    p_true = 10.0 ** u.comp["log_p"][pi]
    t_in = has & (a_true > win.alpha0_min_mas) & (p_true > win.period_min_days) & (p_true < win.period_max_days)
    pm = np.asarray(tr["period_days"], float)
    m_in = (a_mock > win.alpha0_min_mas) & (pm > win.period_min_days) & (pm < win.period_max_days)
    wcmp = compare_counts(np.where(t_in, 0.5, np.nan), t_in, np.where(m_in, 0.5, np.nan), row, mock.w_naive, mock.w_corr, np.array([0.0, 1.0]), n_rows)
    nss = {k: float(getattr(wcmp, k)[0]) for k in ("truth", "naive", "corrected", "pull_naive", "pull_corrected")}
    vol = volume_comparison(u, par, mock, cfg, mcfg, rng)
    ess = {
        "draws": float(tr["draw_index"].size),
        "ess_corrected": ps.kish_ess(mock.w_corr),
        "ess_naive": ps.kish_ess(mock.w_naive),
    }
    s_mu = sigma_mu_rows(par)
    res = ClosedLoopResult(
        size=str(size),
        counts=par.counts,
        parent_binary_by_g=by_g,
        parent_binary_by_m1=by_m1,
        parent_binary_total=total,
        parent_shapes=shapes,
        alpha0=alpha,
        nss_window=nss,
        volume=vol,
        ess=ess,
        sigma_mu_quantiles=np.nanpercentile(s_mu, [16, 50, 84, 95]),
    )
    raw = {"universe": u, "parent": par, "mock": mock, "grid": grid, "a_true": a_true, "a_mock": a_mock}
    return res, raw
