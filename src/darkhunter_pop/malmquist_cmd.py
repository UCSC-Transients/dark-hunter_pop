"""2-D CMD magnitude-limit (Malmquist / Öpik) weight (#418; docs/MOCK_POPULATION_SPEC.md §11.4).

Ryan (2026-10-03): "do the normal stellar-locus binary check, but a binary will not just shift
M_G, but also BP−RP. Both have to be taken into account along with extinction." For parent row
``s`` with dereddened system point ``y_s = (C_s, M_s)`` and a drawn luminous companion of mass
M2 and G-band flux ratio f = F2/F1, the companion's light is subtracted in G, BP and RP:

    x_G  = f / (1 + f)
    x_BP = x_G 10^{−0.4 [(BP−G)_2 − (BP−G)_s]},   x_RP = x_G 10^{+0.4 [(G−RP)_2 − (G−RP)_s]}
    C_1  = C_s − 2.5 log10(1 − x_BP) + 2.5 log10(1 − x_RP),   M_1 = M_s + 2.5 log10(1 + f)

with the companion's intrinsic colours from the MIST main sequence at M2 and the system's
(BP−G), (G−RP) from the MIST single-star colour–colour relation at C_s. The remaining primary
is compared with the measured single-star ridge R(C) (:class:`darkhunter_pop.giants.MSRidge`):

    L_s(c) = N(M_1 − R(C_1); 0, σ_s(C_1)),   L_s(∅) = N(M_s − R(C_s); 0, σ_s(C_s))
    σ_s(C)² = σ_R(C)² + σ_μ,s² + σ_A² (1 − R′(C) / k_s)²
    Z_s = (1 − F_lum(M̂1)) L_s(∅) + Σ_{q,f} λ_qf(M̂1) L_s(q, f),   W_s(c) = L_s(c) / Z_s

No M1 → M_G relation enters L. ``λ_qf`` is the rung-2 MdS17 luminous target
(:func:`darkhunter_pop.proposal_set.mds17_luminous_log_intensity`) marginalized over P and e
onto (log q, log f) bins; f is spread about the decided Janssens relation (MP-Q13).

Rows that get W = 1 (counted by the caller): CMD-evolved rows (§10.6), rows outside the ridge
colour range, rows without a CMD. Every open choice is a ``provisional_*`` config field
(MP-Q33–Q38). Nothing here imports or reimplements gaiamock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import moe_distefano as mds
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.giants import MSRidge
from darkhunter_pop.malmquist import _log_q_edges, mds17_luminous_m2_p_intensity
from darkhunter_pop.proposal_set import MdS17TargetConfig, relation_log10_flux_ratio

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Config (config/population/malmquist_cmd.yaml; not under config/fragments/)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QFGridConfig(_Strict):
    """Quadrature grid for λ_qf(log q, log f | M1) (numerical only)."""

    log_m1_min: float
    log_m1_max: float
    n_m1: int = Field(..., ge=2)
    n_log_q_fine: int = Field(..., ge=8)  # quadrature cells in log q (on MdS17's breaks)
    n_log_q_bins: int = Field(..., ge=4)  # output bins in log q
    n_log_p: int = Field(..., ge=8)
    n_hermite: int = Field(..., ge=3)
    log_f_min: float  # companions fainter than this are put in the first bin (boost < 1e-5 mag)
    log_f_max: float
    n_log_f: int = Field(..., ge=8)

    @model_validator(mode="after")
    def _order(self) -> QFGridConfig:
        if not (self.log_m1_min < self.log_m1_max and self.log_f_min < self.log_f_max):
            raise ValueError("grid bounds must be increasing")
        return self


class CompanionColourConfig(_Strict):
    """Companion and system colours from the MIST main sequence (MP-Q37).

    ``coeval`` (decided 2026-10-04, spec §0.4): each row's colours come from the MS of its own
    isochrone, at the row's posterior ⟨[Fe/H]⟩ (rounded to ``feh_step_dex``, linear between
    MIST files) and the native age nearest its ⟨log age⟩. Rows without a posterior use the
    fiducial. ``fiducial_ms``: one MS for every row (the #418 first implementation).
    """

    mode: Literal["coeval", "fiducial_ms"] = "coeval"
    fiducial_feh_dex: float = -0.06
    fiducial_log_age: float = 9.6
    feh_step_dex: float = Field(0.05, gt=0)


class SingleStarDensityConfig(_Strict):
    """The single-star CMD density the primary is compared with (spec §11.4, MP-Q39).

    ``gaussian_ridge``: N(M − R(C); 0, σ_R(C)) with the measured ridge and its faint-side
    width (the §11.4 form as first specified). ``mist_density_ridge_anchored``: the MIST
    prior-predictive single-star density φ(C, M) of :mod:`darkhunter_pop.isochrone_mass`
    (all kept phases, so turnoff stars and subgiants give singles their bright-side tail),
    shifted in M per colour so that its mode matches the measured ridge, convolved with the
    row's σ_M, and divided by the colour Jacobian of the light subtraction.
    """

    provisional_model: Literal["gaussian_ridge", "mist_density_ridge_anchored"] = "mist_density_ridge_anchored"
    colour_smoothing_mag: float = Field(0.02, gt=0)  # colour error + model floor of the map
    mag_floor_mag: float = Field(0.05, gt=0)  # added in quadrature to σ_μ for the M smoothing
    sigma_levels_min_mag: float = Field(0.05, gt=0)  # numerical: tabulated M smoothings
    sigma_levels_max_mag: float = Field(2.5, gt=0)
    n_sigma_levels: int = Field(18, ge=2)
    anchor_to_ridge: bool = True
    #: Divide by the colour Jacobian of the light subtraction (diagnostic switch; see
    #: :func:`colour_jacobian`).
    colour_jacobian: bool = True


class CmdMalmquistConfig(_Strict):
    """Settings for the 2-D weight (spec §11.4)."""

    single_star_density: SingleStarDensityConfig = SingleStarDensityConfig()
    provisional_giant_policy: Literal["unit_weight"] = "unit_weight"  # §10.6 (determined)
    provisional_outside_ridge: Literal["unit_weight"] = "unit_weight"  # MP-Q38
    provisional_blending: Literal["all_unresolved"] = "all_unresolved"  # MP-Q27
    sigma_a_mag: float = Field(0.0, ge=0.0)  # MP-Q29 decided: 0
    companion_colour: CompanionColourConfig = CompanionColourConfig()
    grid: QFGridConfig


def load_cmd_malmquist_config(path: str | Path, key: str = "malmquist_cmd") -> CmdMalmquistConfig:
    """Validate the ``key`` section of a YAML file (relative paths from the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return CmdMalmquistConfig.model_validate(yaml.safe_load(p.read_text())[key])


# ---------------------------------------------------------------------------
# MIST colours: companion at M2 and the single-star colour–colour relation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MsColours:
    """MIST main sequence at one ([Fe/H], age): colours vs mass and vs (BP−RP)."""

    mass: FloatArray
    colour: FloatArray
    bp_g: FloatArray
    g_rp: FloatArray
    mg: FloatArray

    def companion(self, m2: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(BP−G, G−RP) of a main-sequence star of mass ``m2`` (clamped to the MIST range)."""
        m = np.clip(np.asarray(m2, float), self.mass[0], self.mass[-1])
        return np.interp(m, self.mass, self.bp_g), np.interp(m, self.mass, self.g_rp)

    def log10_flux_ratio(self, m1: ArrayLike, m2: ArrayLike) -> FloatArray:
        """MP-Q40: log10 f = −0.4 [M_G(M2) − M_G(M1)] on this MS (masses clamped to its range)."""
        a = np.clip(np.asarray(m1, float), self.mass[0], self.mass[-1])
        b = np.clip(np.asarray(m2, float), self.mass[0], self.mass[-1])
        return -0.4 * (np.interp(b, self.mass, self.mg) - np.interp(a, self.mass, self.mg))

    def from_colour(self, colour: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(BP−G, G−RP) of a single star of colour ``colour`` (BP−RP), clamped."""
        order = np.argsort(self.colour)
        c = np.asarray(colour, float)
        bpg = np.interp(c, self.colour[order], self.bp_g[order])
        return bpg, c - bpg


def ms_colours(grid: im.NativeGrid, feh: float, log_age: float) -> MsColours:
    """MIST MS relation at ([Fe/H], log age), linear in [Fe/H] between the bracketing files."""
    j = int(np.clip(np.searchsorted(grid.feh, feh) - 1, 0, grid.feh.size - 2))
    t = float(np.clip((feh - grid.feh[j]) / (grid.feh[j + 1] - grid.feh[j]), 0.0, 1.0))
    m_a, a = im.main_sequence_colour_relation(grid, float(grid.feh[j]), log_age)
    m_b, b = im.main_sequence_colour_relation(grid, float(grid.feh[j + 1]), log_age)
    lo, hi = max(m_a[0], m_b[0]), min(m_a[-1], m_b[-1])
    mass = np.geomspace(lo, hi, 400)
    out = {k: (1 - t) * np.interp(mass, m_a, a[k]) + t * np.interp(mass, m_b, b[k]) for k in ("mg", "bp_g", "g_rp", "colour")}
    return MsColours(mass=mass, colour=out["colour"], bp_g=out["bp_g"], g_rp=out["g_rp"], mg=out["mg"])


@dataclass(frozen=True)
class MsColourBank:
    """Per-row MS colour relations: ``groups[row_group[s]]`` is row s's :class:`MsColours`."""

    groups: list[MsColours]
    row_group: NDArray[np.int64]

    def for_rows(self, rows: NDArray[np.int64]) -> list[tuple[MsColours, NDArray[np.int64]]]:
        """(MsColours, positions in ``rows``) for every group present among ``rows``."""
        g = self.row_group[rows]
        return [(self.groups[k], np.flatnonzero(g == k)) for k in np.unique(g)]


def ms_colour_bank(
    grid: im.NativeGrid,
    feh: ArrayLike,
    log_age: ArrayLike,
    cfg: CompanionColourConfig,
) -> MsColourBank:
    """Group rows by (rounded ⟨[Fe/H]⟩, nearest native age) and build one MS relation per group."""
    fe = np.asarray(feh, float)
    la = np.asarray(log_age, float)
    if cfg.mode == "fiducial_ms":
        return MsColourBank(groups=[ms_colours(grid, cfg.fiducial_feh_dex, cfg.fiducial_log_age)],
                            row_group=np.zeros(fe.size, np.int64))
    ok = np.isfinite(fe) & np.isfinite(la)
    fe_r = np.where(ok, np.round(np.clip(fe, grid.feh[0], grid.feh[-1]) / cfg.feh_step_dex) * cfg.feh_step_dex,
                    cfg.fiducial_feh_dex)
    ai = np.where(ok, np.abs(grid.log_age[None, :] - np.where(ok, la, 0.0)[:, None]).argmin(axis=1),
                  int(np.abs(grid.log_age - cfg.fiducial_log_age).argmin()))
    keys = np.round(fe_r, 6) * 1000.0 + ai
    uniq, inv = np.unique(keys, return_inverse=True)
    groups = []
    for k in uniq:
        j = int(np.flatnonzero(keys == k)[0])
        groups.append(ms_colours(grid, float(fe_r[j]), float(grid.log_age[ai[j]])))
    return MsColourBank(groups=groups, row_group=inv.astype(np.int64))


# ---------------------------------------------------------------------------
# Subtracting the companion and the ridge likelihood
# ---------------------------------------------------------------------------


def subtract_companion(
    colour_sys: ArrayLike,
    mg_sys: ArrayLike,
    log10_f: ArrayLike,
    m2_msun: ArrayLike,
    ms: MsColours,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Primary (C_1, M_1) after removing the companion's G, BP, RP light; ``ok`` False where
    the companion would outshine the system in BP or RP. ``log10_f = -inf`` returns the system."""
    c1, m1, ok, _, _ = subtract_companion_full(colour_sys, mg_sys, log10_f, m2_msun, ms)
    return c1, m1, ok


def subtract_companion_full(
    colour_sys: ArrayLike,
    mg_sys: ArrayLike,
    log10_f: ArrayLike,
    m2_msun: ArrayLike,
    ms: MsColours,
) -> tuple[FloatArray, FloatArray, BoolArray, FloatArray, FloatArray]:
    """:func:`subtract_companion` plus the companion's BP and RP flux shares (x_BP, x_RP)."""
    cs = np.asarray(colour_sys, float)
    ms_ = np.asarray(mg_sys, float)
    lf = np.asarray(log10_f, float)
    with np.errstate(over="ignore", invalid="ignore"):
        f = np.where(np.isfinite(lf), 10.0**lf, 0.0)
    xg = f / (1.0 + f)
    bpg2, grp2 = ms.companion(m2_msun)
    bpgs, grps = ms.from_colour(cs)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xbp = xg * 10.0 ** (-0.4 * (bpg2 - bpgs))
        xrp = xg * 10.0 ** (0.4 * (grp2 - grps))
        ok = (xbp < 1.0) & (xrp < 1.0)
        c1 = cs - 2.5 * np.log10(np.where(ok, 1.0 - xbp, 1.0)) + 2.5 * np.log10(np.where(ok, 1.0 - xrp, 1.0))
        m1 = ms_ + 2.5 * np.log10(1.0 + f)
    return c1, m1, ok, xbp, xrp


def colour_jacobian(colour1: ArrayLike, x_bp: ArrayLike, x_rp: ArrayLike, ms: MsColours) -> FloatArray:
    """|∂C_s / ∂C_1| at fixed companion (M2, f): (1 − x_BP) h′ − (1 − x_RP)(h′ − 1).

    h(C) = (BP − G)(C) is the MIST single-star colour–colour relation. The companion's G light is
    a fixed fraction of the primary's and its colour is fixed by M2, so the system colour moves
    more slowly than the primary's: a population of primaries spread in colour is compressed by
    this factor in the system CMD (the density gains 1 / |J|). J = 1 without a companion.
    """
    order = np.argsort(ms.colour)
    cc, hh = ms.colour[order], ms.bp_g[order]
    slope = np.gradient(hh, cc)
    hp = np.interp(np.asarray(colour1, float), cc, slope)
    xb = np.asarray(x_bp, float)
    xr = np.asarray(x_rp, float)
    return np.abs((1.0 - xb) * hp - (1.0 - xr) * (hp - 1.0))


@dataclass(frozen=True)
class RidgeTables:
    """Ridge R(C), width σ_R(C) and slope R′(C), clamped at the measured ends."""

    colour: FloatArray
    mag: FloatArray
    sigma: FloatArray
    slope: FloatArray

    @classmethod
    def from_ridge(cls, ridge: MSRidge) -> RidgeTables:
        return cls(
            colour=np.asarray(ridge.colour, float),
            mag=np.asarray(ridge.mag, float),
            sigma=np.asarray(ridge.sigma, float),
            slope=np.gradient(np.asarray(ridge.mag, float), np.asarray(ridge.colour, float)),
        )

    def at(self, c: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
        x = np.asarray(c, float)
        inr = np.isfinite(x) & (x >= self.colour[0]) & (x <= self.colour[-1])
        return (
            np.interp(x, self.colour, self.mag),
            np.interp(x, self.colour, self.sigma),
            np.interp(x, self.colour, self.slope),
            inr,
        )


def log_ridge_likelihood(
    colour1: ArrayLike,
    mg1: ArrayLike,
    sigma_mu: ArrayLike,
    ridge: RidgeTables,
    *,
    sigma_a_mag: float = 0.0,
    k_ag_over_ebprp: ArrayLike | float = 2.0,
) -> FloatArray:
    """log N(M_1 − R(C_1); 0, σ_s(C_1)) with σ_s² = σ_R² + σ_μ² + σ_A² (1 − R′/k)²."""
    r, s_r, slope, _ = ridge.at(colour1)
    smu = np.asarray(sigma_mu, float)
    k = np.asarray(k_ag_over_ebprp, float)
    var = s_r**2 + smu**2 + (sigma_a_mag * (1.0 - slope / k)) ** 2
    resid = np.asarray(mg1, float) - r
    return -0.5 * resid * resid / var - 0.5 * np.log(var) - _LOG_SQRT_2PI


# ---------------------------------------------------------------------------
# MIST single-star density, anchored to the measured ridge (MP-Q39 option)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleStarDensity:
    """log φ(C, M | σ_M) on a grid: ``log_phi[k, i, j]`` at M smoothing ``sigma_levels[k]``,
    colour ``colour[i]`` and magnitude ``mag[j]`` (per mag²; MIST prior-predictive, single stars),
    and the per-colour magnitude shift ``shift(C)`` that puts its mode on the measured ridge."""

    colour: FloatArray
    mag: FloatArray
    sigma_levels: FloatArray
    log_phi: FloatArray
    shift_colour: FloatArray
    shift_mag: FloatArray

    def shift(self, c: ArrayLike) -> FloatArray:
        return np.interp(np.asarray(c, float), self.shift_colour, self.shift_mag)

    def log_density(self, c: ArrayLike, m: ArrayLike, sigma_m: ArrayLike) -> FloatArray:
        """Trilinear lookup (colour, magnitude, log σ_M), clamped to the grid."""
        cc = np.asarray(c, float)
        mm = np.asarray(m, float) - self.shift(cc)
        ss = np.log(np.clip(np.asarray(sigma_m, float), self.sigma_levels[0], self.sigma_levels[-1]))
        lv = np.log(self.sigma_levels)
        cc, mm, ss = np.broadcast_arrays(cc, mm, ss)

        def frac(x: FloatArray, grid: FloatArray) -> tuple[NDArray[np.int64], FloatArray]:
            t = (x - grid[0]) / (grid[1] - grid[0])
            i = np.clip(np.floor(t).astype(np.int64), 0, grid.size - 2)
            return i, np.clip(t - i, 0.0, 1.0)

        ic, tc = frac(cc, self.colour)
        im_, tm = frac(mm, self.mag)
        k = np.clip(np.searchsorted(lv, ss, side="right") - 1, 0, lv.size - 2)
        tk = np.clip((ss - lv[k]) / (lv[k + 1] - lv[k]), 0.0, 1.0)
        out = np.zeros(cc.shape)
        for dk, wk in ((0, 1.0 - tk), (1, tk)):
            for di, wi in ((0, 1.0 - tc), (1, tc)):
                for dj, wj in ((0, 1.0 - tm), (1, tm)):
                    out += wk * wi * wj * self.log_phi[k + dk, ic + di, im_ + dj]
        return out


def build_single_star_density(
    cmap: im.CmdMap, cfg: SingleStarDensityConfig, ridge: MSRidge | None, ridge_mag_window: tuple[float, float] = (1.5, 12.0)
) -> SingleStarDensity:
    """Smooth the isochrone prior map (channel ``one``) and anchor its mode to the ridge.

    The anchor: at each ridge colour bin the mode in M of the least-smoothed map (within
    ``ridge_mag_window``, averaged over the bin's colour width) is R_MIST(C); the shift is
    R_data(C) − R_MIST(C), interpolated in colour and clamped at the ends. ``ridge=None`` or
    ``cfg.anchor_to_ridge = False`` gives no shift.
    """
    from scipy.ndimage import gaussian_filter1d

    base = cmap.maps[..., 0] / (np.mean(np.diff(cmap.colour_edges)) * np.mean(np.diff(cmap.mag_edges)))
    dc = float(np.mean(np.diff(cmap.colour_edges)))
    dm = float(np.mean(np.diff(cmap.mag_edges)))
    base = gaussian_filter1d(base, cfg.colour_smoothing_mag / dc, axis=0, mode="constant")
    levels = np.geomspace(cfg.sigma_levels_min_mag, cfg.sigma_levels_max_mag, cfg.n_sigma_levels)
    stack = np.stack([gaussian_filter1d(base, s / dm, axis=1, mode="constant") for s in levels])
    floor = 1e-12 * float(stack.max())
    log_phi = np.log(np.maximum(stack, floor))
    colour = cmap.colour_centres
    mag = cmap.mag_centres
    sc = np.array([0.0, 1.0])
    sm = np.array([0.0, 0.0])
    if ridge is not None and cfg.anchor_to_ridge:
        half = 0.5 * float(np.median(np.diff(ridge.colour))) if ridge.colour.size > 1 else 0.05
        win = (mag > ridge_mag_window[0]) & (mag < ridge_mag_window[1])
        r_mist = []
        for c in ridge.colour:
            cols = (colour >= c - half) & (colour < c + half)
            prof = stack[0][cols][:, win].sum(axis=0)
            r_mist.append(float(mag[win][int(np.argmax(prof))]) if prof.max() > 0 else np.nan)
        r_mist_a = np.asarray(r_mist)
        ok = np.isfinite(r_mist_a)
        sc = np.asarray(ridge.colour, float)[ok]
        sm = (np.asarray(ridge.mag, float) - r_mist_a)[ok]
    return SingleStarDensity(colour=colour, mag=mag, sigma_levels=levels, log_phi=log_phi, shift_colour=sc, shift_mag=sm)


def log_primary_likelihood(
    colour1: ArrayLike,
    mg1: ArrayLike,
    sigma_mu: ArrayLike,
    ridge: RidgeTables,
    cfg: CmdMalmquistConfig,
    *,
    k_ag_over_ebprp: ArrayLike | float = 2.0,
    dens: SingleStarDensity | None = None,
    x_bp: ArrayLike | float = 0.0,
    x_rp: ArrayLike | float = 0.0,
    ms: MsColours | None = None,
) -> FloatArray:
    """log p(primary at (C_1, M_1)) under ``cfg.single_star_density`` (spec §11.4).

    ``gaussian_ridge``: :func:`log_ridge_likelihood`. ``mist_density_ridge_anchored``:
    log φ(C_1, M_1 | σ_M) − log |J| with σ_M² = σ_μ² + mag_floor² and J from
    :func:`colour_jacobian` (needs ``dens`` and ``ms``).
    """
    if cfg.single_star_density.provisional_model == "gaussian_ridge":
        return log_ridge_likelihood(colour1, mg1, sigma_mu, ridge, sigma_a_mag=cfg.sigma_a_mag, k_ag_over_ebprp=k_ag_over_ebprp)
    if dens is None or ms is None:
        raise ValueError("mist_density_ridge_anchored needs the SingleStarDensity and MS colours")
    sm = np.sqrt(np.asarray(sigma_mu, float) ** 2 + cfg.single_star_density.mag_floor_mag**2)
    jac = colour_jacobian(colour1, x_bp, x_rp, ms) if cfg.single_star_density.colour_jacobian else 1.0
    with np.errstate(divide="ignore"):
        return dens.log_density(colour1, mg1, sm) - np.log(np.maximum(jac, 1e-6))


# ---------------------------------------------------------------------------
# λ_qf(M1) grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QFGrid:
    """``lam[i, a, b]``: companions per primary of mass ``10**log_m1[i]`` with log q in bin
    ``a`` (centre ``log_q[a]``) and log f in bin ``b`` (centre ``log_f[b]``); ``f_lum[i]`` is
    the total (mass outside the f range is clipped into the end bins, none is lost)."""

    log_m1: FloatArray
    log_q: FloatArray
    log_f: FloatArray
    lam: FloatArray
    f_lum: FloatArray

    def interpolate(self, m1_msun: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """(λ_qf, F_lum) at arbitrary M1, linear in log10 M1, clamped to the grid ends."""
        x = np.clip(np.log10(np.asarray(m1_msun, float)), self.log_m1[0], self.log_m1[-1])
        j = np.clip(np.searchsorted(self.log_m1, x, side="right") - 1, 0, self.log_m1.size - 2)
        t = (x - self.log_m1[j]) / (self.log_m1[j + 1] - self.log_m1[j])
        lam = (1.0 - t)[..., None, None] * self.lam[j] + t[..., None, None] * self.lam[j + 1]
        return lam, (1.0 - t) * self.f_lum[j] + t * self.f_lum[j + 1]


def build_qf_grid(target: MdS17TargetConfig, grid: QFGridConfig) -> QFGrid:
    """Tabulate λ_qf by midpoint quadrature in (log q, log P) and Gauss–Hermite in log f."""
    table = mds.load_mds17_table(target.table_path)
    plo, phi = table.log_p_range
    lq_edges = _log_q_edges(table, grid.n_log_q_fine)
    lp_edges = np.linspace(plo, phi, grid.n_log_p + 1)
    lq = 0.5 * (lq_edges[1:] + lq_edges[:-1])
    lp = 0.5 * (lp_edges[1:] + lp_edges[:-1])
    LQ, LP = np.meshgrid(lq, lp, indexing="ij")
    cell = np.outer(np.diff(lq_edges), np.diff(lp_edges))
    nodes, hw = np.polynomial.hermite_e.hermegauss(grid.n_hermite)
    hw = hw / hw.sum()
    log_m1 = np.linspace(grid.log_m1_min, grid.log_m1_max, grid.n_m1)
    q_out = np.linspace(lq_edges[0], lq_edges[-1], grid.n_log_q_bins + 1)
    f_edges = np.linspace(grid.log_f_min, grid.log_f_max, grid.n_log_f + 1)
    lam = np.zeros((grid.n_m1, grid.n_log_q_bins, grid.n_log_f))
    f_lum = np.zeros(grid.n_m1)
    qb_fine = np.clip(np.searchsorted(q_out, lq, side="right") - 1, 0, grid.n_log_q_bins - 1)
    for i, lm1 in enumerate(log_m1):
        m1 = 10.0**lm1
        dens = (mds17_luminous_m2_p_intensity(m1, LQ, LP, target) * cell).sum(axis=1)  # per fine log q
        rel = relation_log10_flux_ratio(m1, m1 * 10.0**lq)
        ok = np.isfinite(dens) & np.isfinite(rel) & (dens > 0)
        d, r, qb = dens[ok], rel[ok], qb_fine[ok]
        f_lum[i] = float(d.sum())
        lf = r[:, None] + target.flux_sigma_dex * nodes[None, :]
        wt = d[:, None] * hw[None, :]
        fb = np.clip(np.searchsorted(f_edges, lf, side="right") - 1, 0, grid.n_log_f - 1)
        flat = (np.broadcast_to(qb[:, None], fb.shape) * grid.n_log_f + fb).ravel()
        lam[i] = np.bincount(flat, weights=wt.ravel(), minlength=grid.n_log_q_bins * grid.n_log_f).reshape(
            grid.n_log_q_bins, grid.n_log_f
        )
    return QFGrid(
        log_m1=log_m1,
        log_q=0.5 * (q_out[1:] + q_out[:-1]),
        log_f=0.5 * (f_edges[1:] + f_edges[:-1]),
        lam=lam,
        f_lum=f_lum,
    )


@dataclass(frozen=True)
class QGrid:
    """MP-Q40 form of the companion grid: ``lam[i, a]`` companions per primary of mass
    ``10**log_m1[i]`` with log q in bin ``a``, summed over P; the f distribution is applied per
    row as N(log10 f_MIST(M1, q M1; row isochrone), σ_f) with Gauss–Hermite ``nodes`` and
    normalized ``weights``."""

    log_m1: FloatArray
    log_q: FloatArray
    lam: FloatArray
    f_lum: FloatArray
    sigma_f_dex: float
    nodes: FloatArray
    weights: FloatArray

    def interpolate(self, m1_msun: ArrayLike) -> tuple[FloatArray, FloatArray]:
        x = np.clip(np.log10(np.asarray(m1_msun, float)), self.log_m1[0], self.log_m1[-1])
        j = np.clip(np.searchsorted(self.log_m1, x, side="right") - 1, 0, self.log_m1.size - 2)
        t = (x - self.log_m1[j]) / (self.log_m1[j + 1] - self.log_m1[j])
        lam = (1.0 - t)[..., None] * self.lam[j] + t[..., None] * self.lam[j + 1]
        return lam, (1.0 - t) * self.f_lum[j] + t * self.f_lum[j + 1]


def build_q_grid(target: MdS17TargetConfig, grid: QFGridConfig) -> QGrid:
    """λ_q(M1) by midpoint quadrature in (log q, log P) (P summed), for the MP-Q40 MIST relation.

    Every q cell of MdS17's 0.1–1 enters (the MIST relation is defined over the whole MS,
    clamped at its low-mass end), so ``F_lum`` can exceed :func:`build_qf_grid`'s, which drops
    companions outside the Janssens mass range."""
    table = mds.load_mds17_table(target.table_path)
    plo, phi = table.log_p_range
    lq_edges = _log_q_edges(table, grid.n_log_q_fine)
    lp_edges = np.linspace(plo, phi, grid.n_log_p + 1)
    lq = 0.5 * (lq_edges[1:] + lq_edges[:-1])
    lp = 0.5 * (lp_edges[1:] + lp_edges[:-1])
    LQ, LP = np.meshgrid(lq, lp, indexing="ij")
    cell = np.outer(np.diff(lq_edges), np.diff(lp_edges))
    log_m1 = np.linspace(grid.log_m1_min, grid.log_m1_max, grid.n_m1)
    q_out = np.linspace(lq_edges[0], lq_edges[-1], grid.n_log_q_bins + 1)
    qb = np.clip(np.searchsorted(q_out, lq, side="right") - 1, 0, grid.n_log_q_bins - 1)
    lam = np.zeros((grid.n_m1, grid.n_log_q_bins))
    for i, lm1 in enumerate(log_m1):
        dens = (mds17_luminous_m2_p_intensity(10.0**lm1, LQ, LP, target) * cell).sum(axis=1)
        ok = np.isfinite(dens) & (dens > 0)
        lam[i] = np.bincount(qb[ok], weights=dens[ok], minlength=grid.n_log_q_bins)
    nodes, hw = np.polynomial.hermite_e.hermegauss(grid.n_hermite)
    return QGrid(log_m1=log_m1, log_q=0.5 * (q_out[1:] + q_out[:-1]), lam=lam, f_lum=lam.sum(axis=1),
                 sigma_f_dex=float(target.flux_sigma_dex), nodes=nodes, weights=hw / hw.sum())


# ---------------------------------------------------------------------------
# Rows, Z_s, log W
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmdRows:
    """Per parent row: dereddened system point, σ_μ, k = A_G / E(BP−RP), M̂1, and unit-weight flags."""

    colour0: FloatArray
    mg0: FloatArray
    sigma_mu: FloatArray
    k_ag_over_ebprp: FloatArray
    m1_msun: FloatArray
    unit_weight: BoolArray
    unit_reason: NDArray[np.str_]

    def counts(self) -> dict[str, int]:
        r, k = np.unique(self.unit_reason, return_counts=True)
        return {str(a): int(b) for a, b in zip(r, k)}


def cmd_rows(
    colour0: ArrayLike,
    mg0: ArrayLike,
    sigma_mu: ArrayLike,
    m1_msun: ArrayLike,
    ridge: RidgeTables,
    *,
    evolved: ArrayLike | None = None,
    a_g: ArrayLike | None = None,
    e_bp_rp: ArrayLike | None = None,
) -> CmdRows:
    """Assemble the per-row inputs and mark the unit-weight rows (reason per row)."""
    c = np.asarray(colour0, float)
    m = np.asarray(mg0, float)
    smu = np.asarray(sigma_mu, float)
    smu = np.where(np.isfinite(smu), smu, 0.0)
    n = c.size
    reason = np.full(n, "weighted", dtype="<U16")
    _, _, _, inr = ridge.at(c)
    reason[~inr] = "outside_ridge"
    reason[~(np.isfinite(c) & np.isfinite(m))] = "no_cmd"
    if evolved is not None:
        reason[np.asarray(evolved, bool) & (reason == "weighted")] = "evolved"
    m1 = np.asarray(m1_msun, float)
    reason[~np.isfinite(m1) & (reason == "weighted")] = "no_m1"
    if a_g is not None and e_bp_rp is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.asarray(a_g, float) / np.asarray(e_bp_rp, float)
        k = np.where(np.isfinite(k) & (k > 0), k, 2.0)
    else:
        k = np.full(n, 2.0)
    return CmdRows(
        colour0=c, mg0=m, sigma_mu=smu, k_ag_over_ebprp=k, m1_msun=m1,
        unit_weight=reason != "weighted", unit_reason=reason,
    )


def _row_log_terms(
    rows: CmdRows,
    idx: NDArray[np.int64],
    qf: QFGrid | QGrid,
    ridge: RidgeTables,
    ms: MsColours,
    cfg: CmdMalmquistConfig,
    chunk: int,
    dens: SingleStarDensity | None = None,
) -> tuple[FloatArray, FloatArray]:
    """(log of (1 − F)L(∅), log Z) for rows ``idx`` (chunked)."""
    log_single = np.empty(idx.size)
    log_z = np.empty(idx.size)
    q = 10.0**qf.log_q
    for a in range(0, idx.size, chunk):
        ii = idx[a:a + chunk]
        c, m, smu, k, m1 = rows.colour0[ii], rows.mg0[ii], rows.sigma_mu[ii], rows.k_ag_over_ebprp[ii], rows.m1_msun[ii]
        lam, f_lum = qf.interpolate(m1)  # (n, nq, nf) or (n, nq) for QGrid
        l0 = log_primary_likelihood(c, m, smu, ridge, cfg, k_ag_over_ebprp=k, dens=dens, ms=ms)
        if isinstance(qf, QGrid):  # MP-Q40: f about the coeval MIST relation, per row
            nf = qf.nodes.size
            shape = (ii.size, q.size, nf)
            m2 = np.broadcast_to((m1[:, None] * q[None, :])[:, :, None], shape)
            rel = ms.log10_flux_ratio(m1[:, None], m1[:, None] * q[None, :])
            lf = rel[:, :, None] + qf.sigma_f_dex * qf.nodes[None, None, :]
            lam = lam[:, :, None] * qf.weights[None, None, :]
        else:
            shape = (ii.size, q.size, qf.log_f.size)
            m2 = np.broadcast_to((m1[:, None] * q[None, :])[:, :, None], shape)
            lf = np.broadcast_to(qf.log_f[None, None, :], shape)
        c1, mg1, ok, xbp, xrp = subtract_companion_full(
            np.broadcast_to(c[:, None, None], shape), np.broadcast_to(m[:, None, None], shape), lf, m2, ms
        )
        ll = log_primary_likelihood(
            c1, mg1, np.broadcast_to(smu[:, None, None], shape), ridge, cfg,
            k_ag_over_ebprp=np.broadcast_to(k[:, None, None], shape), dens=dens, x_bp=xbp, x_rp=xrp, ms=ms,
        )
        ll = np.where(ok, ll, -np.inf)
        mx = np.maximum(ll.reshape(ii.size, -1).max(axis=1), l0)
        lum = np.sum(lam * np.exp(ll - mx[:, None, None]), axis=(1, 2))
        p_single = np.clip(1.0 - f_lum, 0.0, None)  # MP-Q31 provisional clip
        with np.errstate(divide="ignore"):
            log_single[a:a + chunk] = np.log(p_single) + l0
            log_z[a:a + chunk] = mx + np.log(p_single * np.exp(l0 - mx) + lum)
    return log_single, log_z


@dataclass(frozen=True)
class RowNormalization:
    """log Z_s and log p(∅ | o_s) per row (NaN for unit-weight rows)."""

    log_z: FloatArray
    log_p_single: FloatArray


def row_normalization(
    rows: CmdRows,
    qf: QFGrid | QGrid,
    ridge: RidgeTables,
    ms: MsColours | MsColourBank,
    cfg: CmdMalmquistConfig,
    *,
    chunk: int = 400,
    dens: SingleStarDensity | None = None,
) -> RowNormalization:
    """Z_s for every weighted row (spec §11.4)."""
    n = rows.colour0.size
    log_z = np.full(n, np.nan)
    lps = np.full(n, np.nan)
    idx = np.flatnonzero(~rows.unit_weight)
    bank = ms if isinstance(ms, MsColourBank) else MsColourBank(groups=[ms], row_group=np.zeros(n, np.int64))
    for msg, pos in bank.for_rows(idx):
        sub = idx[pos]
        ls, lz = _row_log_terms(rows, sub, qf, ridge, msg, cfg, chunk, dens)
        log_z[sub] = lz
        lps[sub] = ls - lz
    return RowNormalization(log_z=log_z, log_p_single=lps)


def log_weight_for_draws(
    truth: Mapping[str, NDArray[Any]],
    rows: CmdRows,
    norm: RowNormalization,
    ridge: RidgeTables,
    ms: MsColours | MsColourBank,
    cfg: CmdMalmquistConfig,
    *,
    dens: SingleStarDensity | None = None,
) -> FloatArray:
    """log W per proposal-set draw; 0 on unit-weight rows. Dark draws carry L_s(∅).

    ``truth`` needs ``parent_row``, ``is_dark``, ``log10_flux_ratio`` and ``m2_msun`` (as
    written by :func:`darkhunter_pop.proposal_set.sample_proposal`).
    """
    r = np.asarray(truth["parent_row"], np.int64)
    dark = np.asarray(truth["is_dark"], bool)
    lf = np.where(dark, -np.inf, np.asarray(truth["log10_flux_ratio"], float))
    m2 = np.asarray(truth["m2_msun"], float)
    out = np.zeros(r.size)
    w = ~rows.unit_weight[r]
    if w.any():
        wi = np.flatnonzero(w)
        bank = ms if isinstance(ms, MsColourBank) else MsColourBank(groups=[ms], row_group=np.zeros(rows.colour0.size, np.int64))
        for msg, pos in bank.for_rows(r[wi]):
            di = wi[pos]
            rr = r[di]
            c1, mg1, ok, xbp, xrp = subtract_companion_full(rows.colour0[rr], rows.mg0[rr], lf[di], m2[di], msg)
            ll = log_primary_likelihood(
                c1, mg1, rows.sigma_mu[rr], ridge, cfg, k_ag_over_ebprp=rows.k_ag_over_ebprp[rr],
                dens=dens, x_bp=xbp, x_rp=xrp, ms=msg,
            )
            out[di] = np.where(ok, ll, -np.inf) - norm.log_z[rr]
    return out


def ridge_residual(colour0: ArrayLike, mg0: ArrayLike, ridge: RidgeTables) -> FloatArray:
    """ΔM_2D = M_G0 − R(C0) (NaN outside the ridge colour range); the MP-Q25 replacement."""
    r, _, _, inr = ridge.at(colour0)
    return np.where(inr, np.asarray(mg0, float) - r, np.nan)
