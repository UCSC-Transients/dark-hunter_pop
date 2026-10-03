"""Evolved (subgiant + giant) primaries in the mock population: CMD classifier and diagnostics.

docs/MOCK_POPULATION_SPEC.md §10 (MP-Q28, #413). The parent's current ``is_giant`` flag
(log g of the atmosphere TAG10 used < 3.6) catches almost no giants, because MSC fits every
source as a pair of dwarfs and its log g is dwarf-like by construction. This module identifies
evolved stars from the **dereddened colour-magnitude diagram** instead, identically for the
``gaia_source`` parent and the real NSS orbits:

    M_G0 = G − μ(d_BJ) − A_G,     C0 = (BP−RP) − E(BP−RP)

with ``d_BJ`` the Bailer-Jones et al. (2021) geometric distance and Combined19 E(B−V)
(MP-Q29) turned into A_G and E(BP−RP) with the Babusiaux et al. (2018) Gaia law. The
main-sequence ridge ``R(C0)`` and its faint-side 1σ width ``σ_R(C0)`` are **measured from the
parent itself** (:func:`fit_ms_ridge`). A row is evolved when

    ΔM = M_G0 − R(C0) < −2.5 log10 2 − n_σ σ_tot,   σ_tot² = σ_R(C0)² + σ_μ²

i.e. brighter than any main-sequence star plus an equal-light companion can be. ``n_σ`` is a
provisional setting (MP-Q28 option). Rows outside the ridge's colour range are not classified.

Also: the CMD radius (Andrae et al. 2018 bolometric correction) and the Eggleton (1983)
Roche-lobe period floor, used by the diagnostics only.

**Hook for** ``proposal_set`` / ``malmquist`` (another ticket owns both): the flag can replace
the TAG10-log g flag without editing either module,
``dataclasses.replace(parent, is_giant=classify_parent(parent, cfg, giants_cfg).evolved)``;
``malmquist.row_conditioning`` and ``log_weight_for_draws`` read ``parent.is_giant``.
Nothing here imports or reimplements gaiamock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import constants
from darkhunter_pop.config_loader import repo_root

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MU_PER_LN_D: float = 5.0 / math.log(10.0)


# ---------------------------------------------------------------------------
# Config (config/population/giants.yaml; not under config/fragments/)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GiantExtinctionConfig(_Strict):
    """Extinction for the CMD (MP-Q29: Combined19; Gaia band law from ``dust_maps``)."""

    map: Literal["combined19"]
    band_law: Literal["babusiaux2018"]


class RidgeConfig(_Strict):
    """Numerical settings of the main-sequence ridge measurement (:func:`fit_ms_ridge`)."""

    colour_min: float
    colour_max: float
    colour_step: float = Field(..., gt=0.0)
    mag_min: float
    mag_max: float
    hist_step_mag: float = Field(..., gt=0.0)
    smooth_bins: int = Field(..., ge=1)
    faint_window_mag: float = Field(..., gt=0.0)
    clip_sigma: float = Field(..., gt=0.0)
    clip_iterations: int = Field(..., ge=0)
    min_rows_per_bin: int = Field(..., ge=10)
    min_parallax_over_error: float = Field(..., ge=0.0)
    #: Measure the ridge on rows with RUWE below this only (#418: one ridge shared with the
    #: 2-D Malmquist weight, from single-star-like rows). ``None`` keeps every row (#413).
    ruwe_max: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def _order(self) -> RidgeConfig:
        if not (self.colour_min < self.colour_max and self.mag_min < self.mag_max):
            raise ValueError("ridge colour and magnitude ranges must be increasing")
        return self


class GiantsConfig(_Strict):
    """Settings for the CMD evolved-star classifier (spec §10)."""

    extinction: GiantExtinctionConfig
    ridge: RidgeConfig
    provisional_n_sigma: float = Field(..., ge=0.0)  # MP-Q28 option


def load_giants_config(path: str | Path, key: str = "giants") -> GiantsConfig:
    """Validate the ``key`` section of a YAML file (relative paths from the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return GiantsConfig.model_validate(yaml.safe_load(p.read_text())[key])


# ---------------------------------------------------------------------------
# Dereddened CMD
# ---------------------------------------------------------------------------


def combined19_ebv(l_deg: ArrayLike, b_deg: ArrayLike, distance_pc: ArrayLike) -> FloatArray:
    """Combined19 (mwdust) E(B−V) at each position; NaN where the distance is not finite/positive.

    ``mwdust.Combined19`` reads its local map (one-time download on first use).
    """
    import mwdust

    l = np.asarray(l_deg, dtype=np.float64)
    b = np.asarray(b_deg, dtype=np.float64)
    d = np.asarray(distance_pc, dtype=np.float64)
    out = np.full(d.shape, np.nan)
    ok = np.isfinite(l) & np.isfinite(b) & np.isfinite(d) & (d > 0)
    if ok.any():
        out[ok] = np.asarray(mwdust.Combined19()(l[ok], b[ok], d[ok] / 1000.0), dtype=np.float64)
    return out


def dereddened_cmd(
    g_mag: ArrayLike,
    bp_rp: ArrayLike,
    distance_pc: ArrayLike,
    a_g_mag: ArrayLike,
    e_bp_rp_mag: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """``(M_G0, (BP−RP)_0)``; NaN where any input is missing or the distance is not positive."""
    d = np.asarray(distance_pc, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(d > 0, 5.0 * np.log10(d / 10.0), np.nan)
    mg0 = np.asarray(g_mag, dtype=np.float64) - mu - np.asarray(a_g_mag, dtype=np.float64)
    c0 = np.asarray(bp_rp, dtype=np.float64) - np.asarray(e_bp_rp_mag, dtype=np.float64)
    return mg0, c0


def sigma_mu_from_quantiles(r_lo: ArrayLike, r_med: ArrayLike, r_hi: ArrayLike) -> FloatArray:
    """Distance-modulus 1σ from Bailer-Jones 16/50/84 percentiles (as in ``malmquist``)."""
    lo, med, hi = (np.asarray(x, dtype=np.float64) for x in (r_lo, r_med, r_hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _MU_PER_LN_D * (hi - lo) / (2.0 * med)


# ---------------------------------------------------------------------------
# Main-sequence ridge measured from the data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MSRidge:
    """Measured MS ridge: bin centres in (BP−RP)_0, ridge M_G0 and faint-side 1σ per bin."""

    colour: FloatArray
    mag: FloatArray
    sigma: FloatArray
    n_rows: NDArray[np.int64]

    def at(self, colour0: ArrayLike) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Linear interpolation ``(R, σ_R, in_range)``; NaN outside the measured colour range."""
        c = np.asarray(colour0, dtype=np.float64)
        inr = np.isfinite(c) & (c >= self.colour[0]) & (c <= self.colour[-1])
        r = np.where(inr, np.interp(c, self.colour, self.mag), np.nan)
        s = np.where(inr, np.interp(c, self.colour, self.sigma), np.nan)
        return r, s, inr


def fit_ms_ridge(
    mg0: ArrayLike,
    colour0: ArrayLike,
    parallax_over_error: ArrayLike,
    cfg: RidgeConfig,
    *,
    ruwe: ArrayLike | None = None,
) -> MSRidge:
    """Measure the MS ridge (smoothed mode of M_G0) and its faint-side width per colour bin.

    Only rows with finite CMD values and ``parallax_over_error ≥ cfg.min_parallax_over_error``
    and ``cfg.mag_min < M_G0 < cfg.mag_max`` are used; when ``cfg.ruwe_max`` is set, also only
    rows with ``ruwe < cfg.ruwe_max`` (``ruwe`` is then required). Binaries only brighten a star, so the
    faint side (``M_G0`` above the mode, within ``faint_window_mag``) holds singles plus noise;
    its width is the RMS about the mode, iteratively clipped at ``clip_sigma``. Bins with fewer
    than ``min_rows_per_bin`` rows are dropped. Raises if fewer than two bins remain.
    """
    m = np.asarray(mg0, dtype=np.float64)
    c = np.asarray(colour0, dtype=np.float64)
    snr = np.asarray(parallax_over_error, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        base = (
            np.isfinite(m) & np.isfinite(c) & (snr >= cfg.min_parallax_over_error)
            & (m > cfg.mag_min) & (m < cfg.mag_max)
        )
        if cfg.ruwe_max is not None:
            if ruwe is None:
                raise ValueError("ridge.ruwe_max is set but no ruwe values were given")
            base &= np.asarray(ruwe, dtype=np.float64) < cfg.ruwe_max
    n_bins = int(round((cfg.colour_max - cfg.colour_min) / cfg.colour_step))
    edges = cfg.colour_min + cfg.colour_step * np.arange(n_bins + 1)
    hist_edges = np.arange(cfg.mag_min, cfg.mag_max + cfg.hist_step_mag, cfg.hist_step_mag)
    kernel = np.ones(cfg.smooth_bins) / cfg.smooth_bins
    cen, mode, sig, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = base & (c >= lo) & (c < hi)
        n = int(sel.sum())
        if n < cfg.min_rows_per_bin:
            continue
        h, _ = np.histogram(m[sel], bins=hist_edges)
        k = np.convolve(h, kernel, mode="same")
        peak = float(hist_edges[int(np.argmax(k))] + 0.5 * cfg.hist_step_mag)
        x = m[sel] - peak
        faint = x[(x > 0) & (x < cfg.faint_window_mag)]
        if faint.size < 2:
            continue
        s = float(np.sqrt(np.mean(faint**2)))
        for _ in range(cfg.clip_iterations):
            kept = faint[faint < cfg.clip_sigma * s]
            if kept.size < 2:
                break
            s = float(np.sqrt(np.mean(kept**2)))
        cen.append(0.5 * (lo + hi))
        mode.append(peak)
        sig.append(s)
        cnt.append(n)
    if len(cen) < 2:
        raise ValueError("fewer than two colour bins have enough rows to measure the MS ridge")
    return MSRidge(
        colour=np.asarray(cen), mag=np.asarray(mode), sigma=np.asarray(sig), n_rows=np.asarray(cnt, dtype=np.int64)
    )


@dataclass(frozen=True)
class EvolvedClassification:
    """Per-row CMD classification. ``evolved`` is False wherever ``classified`` is False."""

    mg0: FloatArray
    colour0: FloatArray
    delta_m: FloatArray
    sigma_tot: FloatArray
    threshold: FloatArray
    classified: BoolArray
    evolved: BoolArray


def classify_evolved(
    mg0: ArrayLike,
    colour0: ArrayLike,
    sigma_mu: ArrayLike,
    ridge: MSRidge,
    n_sigma: float,
) -> EvolvedClassification:
    """Evolved iff ``ΔM < −TWIN_BRIGHTENING_MAG − n_sigma σ_tot`` (module docstring).

    ``σ_tot² = σ_R(C0)² + σ_μ²``; a non-finite ``σ_μ`` counts as 0 (then only the ridge width
    enters). Rows outside the ridge colour range or without a finite CMD are unclassified.
    """
    m = np.asarray(mg0, dtype=np.float64)
    c = np.asarray(colour0, dtype=np.float64)
    smu = np.asarray(sigma_mu, dtype=np.float64)
    r, s, inr = ridge.at(c)
    smu = np.where(np.isfinite(smu), smu, 0.0)
    stot = np.sqrt(s**2 + smu**2)
    dm = m - r
    thr = -constants.TWIN_BRIGHTENING_MAG - float(n_sigma) * stot
    classified = inr & np.isfinite(dm)
    with np.errstate(invalid="ignore"):
        evolved = classified & (dm < thr)
    return EvolvedClassification(
        mg0=m, colour0=c, delta_m=dm, sigma_tot=stot, threshold=thr, classified=classified, evolved=evolved
    )


# ---------------------------------------------------------------------------
# Radius and Roche-lobe period floor (diagnostics, spec §10.3)
# ---------------------------------------------------------------------------


def bolometric_correction_g(teff_k: ArrayLike) -> FloatArray:
    """Andrae et al. (2018) BC_G(Teff), Eq. 7 / Table 4; NaN outside 3300–8000 K."""
    t = np.asarray(teff_k, dtype=np.float64)
    lo, mid, hi = constants.ANDRAE2018_BCG_TEFF_RANGE_K
    x = t - constants.ANDRAE2018_TEFF_SUN_K
    warm = sum(a * x**i for i, a in enumerate(constants.ANDRAE2018_BCG_WARM))
    cool = sum(a * x**i for i, a in enumerate(constants.ANDRAE2018_BCG_COOL))
    out = np.where(t >= mid, warm, cool)
    return np.where(np.isfinite(t) & (t >= lo) & (t <= hi), out, np.nan)


def cmd_radius_rsun(mg0: ArrayLike, teff_k: ArrayLike) -> FloatArray:
    """Stefan–Boltzmann radius from M_G0 and Teff: R = 10^{0.2(M_bol,⊙ − M_bol)} (T_⊙/T)²."""
    t = np.asarray(teff_k, dtype=np.float64)
    mbol = np.asarray(mg0, dtype=np.float64) + bolometric_correction_g(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 ** (0.2 * (constants.ANDRAE2018_MBOL_SUN - mbol)) * (constants.ANDRAE2018_TEFF_SUN_K / t) ** 2


def eggleton_roche_lobe_fraction(q_donor: ArrayLike) -> FloatArray:
    """Eggleton (1983) r_L / a for ``q_donor = M_donor / M_accretor`` (> 0)."""
    q = np.asarray(q_donor, dtype=np.float64)
    a, b = constants.EGGLETON1983_RL
    q23 = q ** (2.0 / 3.0)
    return a * q23 / (b * q23 + np.log1p(q ** (1.0 / 3.0)))


def roche_period_floor_days(r1_rsun: ArrayLike, m1_msun: ArrayLike, m2_msun: ArrayLike) -> FloatArray:
    """Period (days) at which a star of radius R1 exactly fills its Roche lobe (circular orbit).

    a = R1 / (r_L/a)(M1/M2); P from Kepler's third law with G M⊙ and R⊙ from astropy.
    """
    r1 = np.asarray(r1_rsun, dtype=np.float64)
    m1 = np.asarray(m1_msun, dtype=np.float64)
    m2 = np.asarray(m2_msun, dtype=np.float64)
    a_rsun = r1 / eggleton_roche_lobe_fraction(m1 / m2)
    rsun_m = float(constants.const.R_sun.si.value)
    gm = float(constants.const.G.si.value * constants.const.M_sun.si.value)
    p_s = 2.0 * math.pi * np.sqrt((a_rsun * rsun_m) ** 3 / (gm * (m1 + m2)))
    return p_s / 86400.0


# ---------------------------------------------------------------------------
# Row-level wrappers (parent snapshot and real NSS rows)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowCMD:
    """Dereddened CMD inputs per row. ``extinction_ok`` is False where the band law failed."""

    mg0: FloatArray
    colour0: FloatArray
    sigma_mu: FloatArray
    ebv: FloatArray
    a_g: FloatArray
    extinction_ok: BoolArray


def cmd_for_rows(
    g_mag: ArrayLike,
    bp_rp: ArrayLike,
    l_deg: ArrayLike,
    b_deg: ArrayLike,
    r_med_pc: ArrayLike,
    r_lo_pc: ArrayLike,
    r_hi_pc: ArrayLike,
    pipeline_config: object,
    cfg: GiantsConfig,
    *,
    ebv: ArrayLike | None = None,
) -> RowCMD:
    """Combined19 + Babusiaux et al. (2018) dereddened CMD at the Bailer-Jones distance.

    ``pipeline_config`` is the :class:`~darkhunter_pop.config_schema.PipelineConfig`; its
    ``sample_selection.dust_maps`` supplies ``r_v`` and the band-law coefficients (#295).
    Pass a precomputed ``ebv`` to skip the map lookup. Rows where the law does not converge
    get NaN (reported by the caller, never passed through).
    """
    from darkhunter_pop.elbadry2026_selection import gaia_band_extinction_babusiaux2018

    dust = pipeline_config.sample_selection.dust_maps  # type: ignore[attr-defined]
    coeffs = dust.gaia_band_extinction.babusiaux2018
    if coeffs is None:
        raise ValueError("sample_selection.dust_maps.gaia_band_extinction.babusiaux2018 is required")
    e = combined19_ebv(l_deg, b_deg, r_med_pc) if ebv is None else np.asarray(ebv, dtype=np.float64)
    a_g, e_br, ok = gaia_band_extinction_babusiaux2018(
        np.asarray(bp_rp, dtype=np.float64), e, r_v=float(dust.r_v), coeffs=coeffs
    )
    mg0, c0 = dereddened_cmd(g_mag, bp_rp, r_med_pc, a_g, e_br)
    return RowCMD(
        mg0=mg0, colour0=c0, sigma_mu=sigma_mu_from_quantiles(r_lo_pc, r_med_pc, r_hi_pc),
        ebv=e, a_g=a_g, extinction_ok=ok,
    )


def parent_row_cmd(parent: object, pipeline_config: object, cfg: GiantsConfig) -> RowCMD:
    """:func:`cmd_for_rows` for a :class:`~darkhunter_pop.proposal_set.ParentSnapshot`."""
    cols = parent.columns  # type: ignore[attr-defined]
    return cmd_for_rows(
        cols["phot_g_mean_mag"], cols["bp_rp"], cols["l"], cols["b"],
        cols["r_med_geo"], cols["r_lo_geo"], cols["r_hi_geo"], pipeline_config, cfg,
    )


def classify_parent(
    parent: object, pipeline_config: object, cfg: GiantsConfig, *, cmd: RowCMD | None = None
) -> tuple[EvolvedClassification, MSRidge]:
    """Measure the MS ridge on the parent's usable rows and classify every parent row.

    The ridge uses only ``parent.usable`` rows (the rows the mock draws from), and only those
    with RUWE < ``cfg.ridge.ruwe_max`` when it is set (#418: the same ridge feeds the 2-D
    Malmquist weight, :mod:`darkhunter_pop.malmquist_cmd`). The returned
    ``evolved`` array is the drop-in replacement for ``parent.is_giant`` (module docstring).
    """
    cmd = parent_row_cmd(parent, pipeline_config, cfg) if cmd is None else cmd
    cols = parent.columns  # type: ignore[attr-defined]
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.asarray(cols["parallax"], float) / np.asarray(cols["parallax_error"], float)
    usable = np.asarray(parent.usable, bool)  # type: ignore[attr-defined]
    ruwe = np.asarray(cols["ruwe"], float) if cfg.ridge.ruwe_max is not None else None
    ridge = fit_ms_ridge(np.where(usable, cmd.mg0, np.nan), cmd.colour0, snr, cfg.ridge, ruwe=ruwe)
    return classify_evolved(cmd.mg0, cmd.colour0, cmd.sigma_mu, ridge, cfg.provisional_n_sigma), ridge


def evolved_log10_flux_ratio(m2_msun: ArrayLike, mg0_system: ArrayLike) -> FloatArray:
    """log10 f = log10(L2 / L1) in G for a main-sequence companion of an evolved primary.

    Spec §10.4. Under MP-Q6 (the row's observed G is the total system light) the companion's
    share of the dereddened system light is ``x = 10^{−0.4 (M_G^J(M2) − M_G0,sys)}`` with
    ``M_G^J`` the Janssens et al. (2022) dwarf relation (MP-Q13; the companion of an evolved
    primary is less massive and still on the main sequence), so ``f = x / (1 − x)``. Rows where
    the companion alone would outshine the system (``x ≥ 1``) or ``M_G^J`` is undefined return
    NaN; the target density is zero there. The 0.1 dex scatter of MP-Q13 is applied by the
    caller around this centre, exactly as for the dwarf relation.
    """
    from darkhunter_pop.proposal_set import janssens_absolute_g

    mg2 = janssens_absolute_g(m2_msun)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = 10.0 ** (-0.4 * (mg2 - np.asarray(mg0_system, dtype=np.float64)))
        lf = np.log10(x / (1.0 - x))
    return np.where(np.isfinite(x) & (x < 1.0), lf, np.nan)
