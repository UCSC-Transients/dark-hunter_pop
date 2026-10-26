"""Rung 4: compact-object injection-recovery machinery (#552; docs/MOCK_POPULATION_SPEC.md §13).

The rung-3 Step-1 forward model (looks-single primaries, §12.33 system photometry, epoch model, N2d noise,
gaiamock cascade, resolved companions) with the companion replaced by a compact object:

* **proposal** (:func:`sample_co_proposal`, :func:`log_q_co_total_for`): parent rows from the generation's
  ``parent`` mixture; component WD or dark; M2 log-uniform over the component's range (absolute, not tied to
  M1); WD cooling age log-uniform (a stored truth column, so the WD light is reweightable); log P from the
  ``period`` mixture (core 1.5-3.5 + defensive 0.2-8); e from the bounded ``eccentricity`` proposal; isotropic
  orientation. The WD's light comes from the Montreal / Bédard et al. (2020) DA sequences
  (:class:`WDCoolingGrid`); dark companions have f = 0.
* **target** (:func:`co_log_target`) and **weights** (:func:`co_importance_weights`): deterministic mixture
  over every compact-object generation of one version (§3.7), with the held-out SBC partition
  (:func:`holdout_mask`) removed from, or restricted to, the estimator.
* **selection-function table** (:func:`build_selection_table`): P(published orbit or acceleration | cell) per
  component over (M2, P, e, G, d, M1) with per-cell Kish ESS and the ratio-estimator MC variance.
* **mass recovery** (:func:`orbit_covariance_from_epochs`, :func:`recover_masses`, :func:`recovery_summary`):
  the full 12×12 covariance of each orbital cascade fit, from gaiamock's own residual and linear-solve
  functions (no diagonal fallback), then ``mc_mass_function.propagate_nss_solution`` (the code path real NSS
  rows use) with the primary's isochrone M1 posterior.
* **contamination**: not computed in phase 1; :func:`contamination_stand_in` declares it (``run_validity``).

No gaiamock function is reimplemented: the covariance is the inverse of JᵀJ of gaiamock's
``get_astrometric_residuals_12par`` at the fitted solution (gaiamock's ``get_astrometric_chi2`` supplies the
linear parameters), scaled by gaiamock's own error-inflation factor, the same algebra gaiamock's
``get_uncertainties_at_best_fit_binary_solution`` performs internally before it keeps only the diagonal.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import proposal_set as ps

FloatArray = NDArray[np.float64]

#: ``co_component`` truth codes.
COMPONENT_CODES: dict[str, int] = {"wd": 0, "dark": 1}
#: Cascade-vector indices (gaiamock ``fit_full_astrometric_cascade`` orbital layout).
_CASCADE_INDEX = {"parallax": 0, "sig_parallax": 1, "A": 2, "sig_A": 3, "B": 4, "sig_B": 5, "F": 6, "sig_F": 7,
                  "G": 8, "sig_G": 9, "period": 10, "sig_period": 11, "phi_p": 12, "sig_phi_p": 13, "ecc": 14,
                  "sig_ecc": 15}
#: ParameterSet provenance of a mock cascade covariance.
CASCADE_COVARIANCE_PROVENANCE: str = "gaiamock_cascade_jacobian_12x12"


# ---------------------------------------------------------------------------
# WD light (Montreal synthetic photometry on the Bédard et al. 2020 sequences)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WDCoolingGrid:
    """Per-mass DA cooling sequences: absolute G, BP, RP and Teff vs cooling age.

    :meth:`abs_mags` interpolates linearly in log10(age) along each sequence (ages clamped to the sequence's
    range, with ``min_age_yr`` as the floor) and then linearly in mass between the bracketing sequences.
    Masses outside the grid are clamped to its ends (the grid spans 0.2-1.3 M☉; the rung-4 WD range is
    0.15-1.40 M☉, so the ends use the nearest sequence: a documented approximation).
    """

    masses: FloatArray
    log_age_yr: tuple[FloatArray, ...]
    mags: tuple[FloatArray, ...]  # per mass: shape (n_age, 4) = (G, BP, RP, Teff)
    source: str

    def abs_mags(self, mass_msun: ArrayLike, age_gyr: ArrayLike, min_age_yr: float) -> FloatArray:
        """Shape (n, 4): (M_G, M_BP, M_RP, Teff) per draw; NaN where mass or age is not finite."""
        m = np.asarray(mass_msun, float)
        la = np.log10(np.maximum(np.asarray(age_gyr, float) * 1e9, min_age_yr))
        out = np.full((m.size, 4), np.nan)
        ok = np.isfinite(m) & np.isfinite(la)
        if not ok.any():
            return out
        mc = np.clip(m[ok], self.masses[0], self.masses[-1])
        per_mass = np.stack([np.stack([np.interp(la[ok], self.log_age_yr[k], self.mags[k][:, c]) for c in range(4)], axis=1)
                             for k in range(self.masses.size)])  # (n_mass, n_ok, 4)
        hi = np.clip(np.searchsorted(self.masses, mc, side="right"), 1, self.masses.size - 1)
        lo = hi - 1
        w = ((mc - self.masses[lo]) / (self.masses[hi] - self.masses[lo]))[:, None]
        j = np.arange(mc.size)
        out[ok] = (1.0 - w) * per_mass[lo, j] + w * per_mass[hi, j]
        return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wd_cooling_grid(snapshot_dir: str | Path, sha256: str) -> WDCoolingGrid:
    """Load ``tracks.csv`` from ``scripts/build_wd_cooling_snapshot.py`` (checksum verified)."""
    path = Path(snapshot_dir) / "tracks.csv"
    got = _sha256(path)
    if got != sha256:
        raise ValueError(f"WD cooling snapshot checksum mismatch: {got} != {sha256}")
    d = np.genfromtxt(path, delimiter=",", names=True)
    masses = np.unique(d["mass_msun"])
    ages, mags = [], []
    for m in masses:
        sel = d[d["mass_msun"] == m]
        la = np.log10(np.maximum(sel["age_gyr"] * 1e9, 1.0))
        order = np.argsort(la, kind="stable")
        la, block = la[order], np.stack([sel["Mg"], sel["Mbp"], sel["Mrp"], sel["teff_k"]], axis=1)[order]
        keep = np.concatenate([[True], np.diff(la) > 0])  # repeated ages (age 0 at the hot end): keep the first
        ages.append(la[keep])
        mags.append(block[keep])
    return WDCoolingGrid(masses=masses.astype(float), log_age_yr=tuple(ages), mags=tuple(mags), source=str(path))


def wd_grid_for(cfg: ps.ProposalConfig, data_root: str | Path) -> WDCoolingGrid:
    co = _co(cfg)
    return load_wd_cooling_grid(Path(data_root) / co.wd_light.snapshot, co.wd_light.sha256)


def wd_companion_abs_mags(truth: Mapping[str, NDArray[Any]], grid: WDCoolingGrid, cfg: ps.ProposalConfig) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """(M_G, M_BP, M_RP, Teff) of each draw's WD companion; NaN for dark companions."""
    co = _co(cfg)
    wd = np.asarray(truth["co_component"]) == COMPONENT_CODES["wd"]
    res = np.full((wd.size, 4), np.nan)
    if wd.any():
        res[wd] = grid.abs_mags(np.asarray(truth["m2_msun"], float)[wd], np.asarray(truth["wd_cooling_age_gyr"], float)[wd],
                                co.wd_light.min_age_yr)
    return res[:, 0], res[:, 1], res[:, 2], res[:, 3]


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def _co(cfg: ps.ProposalConfig) -> ps.CompactObjectProposalConfig:
    if cfg.compact_object is None:
        raise ValueError(f"generation {cfg.generation} is not a compact-object generation (proposal.compact_object)")
    return cfg.compact_object


def _log_uniform_logpdf(x: FloatArray, lo: float, hi: float) -> FloatArray:
    with np.errstate(invalid="ignore"):
        inside = (x >= lo) & (x <= hi)
    return np.where(inside, -math.log(hi - lo), -np.inf)


def log_q_co_m2(component: ArrayLike, m2_msun: ArrayLike, cfg: ps.ProposalConfig) -> FloatArray:
    """log q(component, log10 M2): component probability × log-uniform density per dex of M2."""
    co = _co(cfg)
    comp = np.asarray(component)
    lm2 = np.log10(np.asarray(m2_msun, float))
    out = np.full(lm2.size, -np.inf)
    for name, c in (("wd", co.wd), ("dark", co.dark)):
        sel = comp == COMPONENT_CODES[name]
        out[sel] = math.log(c.weight) + _log_uniform_logpdf(lm2[sel], math.log10(c.m2_min_msun), math.log10(c.m2_max_msun))
    return out


#: Spawn slot of the matched cooling-age stream (#552 re-centring; the main stream stays as in generation 1000).
CO_COOLING_MATCHED_RNG_SLOT: int = 21


def row_system_age_gyr(parent: ps.ParentSnapshot, rows: ArrayLike) -> FloatArray:
    """The row's posterior-mean isochrone age (Gyr); NaN without an isochrone fit."""
    r = np.asarray(rows, np.int64)
    if parent.isochrone is None or "log_age_mean" not in parent.isochrone:
        return np.full(r.size, np.nan)
    return 10.0 ** np.asarray(parent.isochrone["log_age_mean"], float)[r] / 1e9


def _matched_avail(cfg: ps.ProposalConfig, m2: ArrayLike, t_sys_row: ArrayLike) -> FloatArray:
    co = _co(cfg)
    assert co.wd_cooling_age_matched is not None
    av = wd_available_time_gyr(m2, t_sys_row, co.wd_cooling_age_matched.shape)
    return np.where(np.isfinite(av) & (av > 0), av, np.nan)


def log_q_co_cooling_age(
    component: ArrayLike, age_gyr: ArrayLike, cfg: ps.ProposalConfig, *, m2_msun: ArrayLike | None = None,
    t_sys_row_gyr: ArrayLike | None = None,
) -> FloatArray:
    """log q(log10 t_cool / Gyr) for WD draws; 0 (no such dimension) for dark draws.

    With ``wd_cooling_age_matched`` (#552): weight w uniform in t on [0, avail(M2, t_sys,row)] (per dex: t ln10 / avail)
    plus 1 − w log-uniform; where avail ≤ 0 or undefined, the log-uniform part takes the whole share."""
    co = _co(cfg)
    comp = np.asarray(component)
    lo, hi = co.wd_log10_cooling_age_gyr
    t = np.asarray(age_gyr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.log10(t)
    ld = _log_uniform_logpdf(lt, lo, hi)
    mc = co.wd_cooling_age_matched
    if mc is not None:
        if m2_msun is None or t_sys_row_gyr is None:
            raise ValueError("wd_cooling_age_matched needs m2_msun and t_sys_row_gyr")
        av = _matched_avail(cfg, m2_msun, t_sys_row_gyr)
        has = np.isfinite(av)
        w = np.where(has, mc.weight, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pm = np.where(has & (t > 0) & (t < np.where(has, av, 0.0)), t * math.log(10.0) / np.where(has, av, 1.0), 0.0)
            ld = np.log(w * pm + (1.0 - w) * np.exp(ld))
    return np.where(comp == COMPONENT_CODES["wd"], ld, 0.0)


def sample_co_proposal(parent: ps.ParentSnapshot, cfg: ps.ProposalConfig, *, draw_index_offset: int = 0) -> dict[str, NDArray[Any]]:
    """Draw ``cfg.n_draws`` compact-object systems; truth columns (luminous names where shared) + log q terms.

    Deterministic given ``cfg.base_seed`` and ``cfg.generation`` (the same ``SeedSequence`` stream as
    :func:`proposal_set.sample_proposal`). WD draws carry ``flux_ratio = NaN`` until
    :func:`proposal_set.apply_system_photometry` sets it from the WD's own light; dark draws have f = 0.
    """
    co = _co(cfg)
    stream = ps.PROPOSAL_RNG_STREAM_BASE + cfg.generation
    rng = np.random.default_rng(np.random.SeedSequence(entropy=cfg.base_seed, spawn_key=(stream, 0)))
    n = cfg.n_draws
    cols = parent.columns
    qs = ps.parent_proposal_probabilities(cols["parallax"], parent.usable_for(cfg.parent), cfg.parent, parent.is_giant,
                                          cols["phot_g_mean_mag"], parent.m1_msun, parent.row_weight)
    row = rng.choice(parent.n_rows, size=n, replace=True, p=qs)
    is_wd = rng.uniform(size=n) < co.wd.weight
    comp = np.where(is_wd, COMPONENT_CODES["wd"], COMPONENT_CODES["dark"]).astype(np.int8)
    lo = np.where(is_wd, math.log10(co.wd.m2_min_msun), math.log10(co.dark.m2_min_msun))
    hi = np.where(is_wd, math.log10(co.wd.m2_max_msun), math.log10(co.dark.m2_max_msun))
    m2 = 10.0 ** rng.uniform(lo, hi)
    a_lo, a_hi = co.wd_log10_cooling_age_gyr
    t_cool = np.where(is_wd, 10.0 ** rng.uniform(a_lo, a_hi, size=n), np.nan)
    t_sys_row = row_system_age_gyr(parent, row)
    if co.wd_cooling_age_matched is not None:  # #552 re-centring: own RNG slot, the main stream is unchanged
        rng_m = np.random.default_rng(np.random.SeedSequence(cfg.base_seed, spawn_key=(stream, CO_COOLING_MATCHED_RNG_SLOT)))
        av = _matched_avail(cfg, m2, t_sys_row)
        pick = (rng_m.uniform(size=n) < co.wd_cooling_age_matched.weight) & np.isfinite(av)
        u = rng_m.uniform(size=n)
        t_cool = np.where(is_wd & pick, u * np.nan_to_num(av), t_cool)

    pc = cfg.period
    core = rng.uniform(size=n) < pc.core_weight
    log_p = np.where(core, rng.uniform(pc.core_log_p_min, pc.core_log_p_max, size=n), rng.uniform(pc.log_p_min, pc.log_p_max, size=n))
    period = 10.0**log_p
    ecc = ps.sample_ecc(period, cfg.eccentricity, rng)
    cos_i = rng.uniform(-1.0, 1.0, size=n)
    big_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    small_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    tp = rng.uniform(0.0, 1.0, size=n) * period

    log_q = {
        "log_q_parent": np.log(qs[row]),
        "log_q_co_m2": log_q_co_m2(comp, m2, cfg),
        "log_q_co_cooling_age": log_q_co_cooling_age(comp, t_cool, cfg, m2_msun=m2, t_sys_row_gyr=t_sys_row),
        "log_q_log_p": ps.log_q_log_p(log_p, pc),
        "log_q_ecc": ps.log_q_ecc(ecc, period, cfg.eccentricity),
    }
    dark = ~is_wd
    out: dict[str, NDArray[Any]] = {
        "draw_index": np.arange(draw_index_offset, draw_index_offset + n, dtype=np.int64),
        "generation": np.full(n, cfg.generation, dtype=np.int64),
        "parent_row": row.astype(np.int64),
        "source_id": np.asarray(cols["source_id"], dtype=np.int64)[row],
        "ra_deg": np.asarray(cols["ra"], float)[row],
        "dec_deg": np.asarray(cols["dec"], float)[row],
        "l_deg": np.asarray(cols["l"], float)[row] if "l" in cols else np.full(n, np.nan),
        "b_deg": np.asarray(cols["b"], float)[row] if "b" in cols else np.full(n, np.nan),
        "parallax_mas": np.asarray(parent.truth_parallax_mas, float)[row],
        "measured_parallax_mas": np.asarray(cols["parallax"], float)[row],
        "is_giant": np.asarray(parent.is_giant, bool)[row],
        "atmosphere_logg": np.asarray(parent.atmosphere_logg, float)[row],
        "pmra_masyr": np.asarray(cols["pmra"], float)[row],
        "pmdec_masyr": np.asarray(cols["pmdec"], float)[row],
        "phot_g_mean_mag": np.asarray(cols["phot_g_mean_mag"], float)[row],
        "m1_msun": parent.m1_msun[row],
        "m2_msun": m2,
        "co_component": comp,
        "wd_cooling_age_gyr": t_cool,
        "is_dark": dark,
        "flux_ratio": np.where(dark, 0.0, np.nan),
        "log10_flux_ratio": np.where(dark, -np.inf, np.nan),
        "period_days": period,
        "eccentricity": ecc,
        "inc_deg": np.degrees(np.arccos(cos_i)),
        "Omega_rad": big_omega,
        "omega_rad": small_omega,
        "Tp_days": tp,
    }
    out.update(log_q)
    out["log_q_total"] = sum(log_q.values())  # type: ignore[assignment]
    return out


def log_q_co_total_for(truth: Mapping[str, NDArray[Any]], parent: ps.ParentSnapshot, cfg: ps.ProposalConfig) -> FloatArray:
    """One compact-object generation's log q at arbitrary stored draws (deterministic-mixture denominator)."""
    qs = ps.parent_proposal_probabilities(parent.columns["parallax"], parent.usable_for(cfg.parent), cfg.parent, parent.is_giant,
                                          parent.columns["phot_g_mean_mag"], parent.m1_msun, parent.row_weight)
    row = np.asarray(truth["parent_row"], np.int64)
    m2 = np.asarray(truth.get("m2_proposal_msun", truth["m2_msun"]), float)
    p = np.asarray(truth["period_days"], float)
    with np.errstate(divide="ignore"):
        lqs = np.log(qs[row])
    lqc = log_q_co_cooling_age(truth["co_component"], truth["wd_cooling_age_gyr"], cfg, m2_msun=m2,
                               t_sys_row_gyr=row_system_age_gyr(parent, row))
    return (lqs + log_q_co_m2(truth["co_component"], m2, cfg) + lqc
            + ps.log_q_log_p(np.log10(p), cfg.period) + ps.log_q_ecc(truth["eccentricity"], p, cfg.eccentricity))


def _splitmix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def holdout_mask(draw_index: ArrayLike, fraction: float, salt: int) -> NDArray[np.bool_]:
    """SBC held-out partition (spec §3.8, §13.2): a draw is held out iff a 64-bit hash of (draw_index, salt) maps
    below ``fraction``. Depends only on the draw itself, so it is the same whatever generations are combined, and
    independent of the proposal (each generation's draws are thinned by the same probability)."""
    idx = np.asarray(draw_index, np.int64).astype(np.uint64)
    u = (_splitmix64(idx ^ np.uint64(int(salt))) >> np.uint64(11)).astype(np.float64) / float(2**53)
    return u < float(fraction)


# ---------------------------------------------------------------------------
# Target θ_CO and weights
# ---------------------------------------------------------------------------


def _log_power_density(lm2: FloatArray, t: ps.CompactM2TargetConfig) -> FloatArray:
    """log of dN/dlog10 M2 ∝ M2^slope normalized on the target range (per dex)."""
    a, b, s = math.log10(t.m2_min_msun), math.log10(t.m2_max_msun), t.slope
    norm = (b - a) if s == 0 else (10.0 ** (s * b) - 10.0 ** (s * a)) / (s * math.log(10.0))
    with np.errstate(invalid="ignore"):
        inside = (lm2 >= a) & (lm2 <= b)
    return np.where(inside, s * math.log(10.0) * np.where(inside, lm2, 0.0) - math.log(norm), -np.inf)


def ifmr_initial_mass(m_final: ArrayLike, segments: Sequence[Sequence[float]]) -> FloatArray:
    """Invert a piecewise-linear IFMR (segments (M_i,lo, M_i,hi, slope, intercept), increasing M_f); M_f outside the
    relation's final-mass range is clamped to its ends."""
    mf = np.asarray(m_final, float)
    seg = np.asarray(segments, float)
    f_lo = seg[:, 2] * seg[:, 0] + seg[:, 3]
    f_hi = seg[:, 2] * seg[:, 1] + seg[:, 3]
    mfc = np.clip(mf, f_lo.min(), f_hi.max())
    out = np.full(mf.size, np.nan)
    for (mi_lo, mi_hi, a, b), lo, hi in zip(seg, f_lo, f_hi):
        sel = np.isnan(out) & (mfc >= lo) & (mfc <= hi)
        out[sel] = np.clip((mfc[sel] - b) / a, mi_lo, mi_hi)
    return out


def wd_available_time_gyr(m_wd: ArrayLike, t_sys_gyr: ArrayLike, tc: ps.WDCoolingAgeTargetConfig) -> FloatArray:
    """t_sys − t_prog(M_WD): the time a WD of this mass has had to cool in a coeval system (may be ≤ 0)."""
    m = np.asarray(m_wd, float)
    mi = ifmr_initial_mass(m, tc.ifmr_segments)
    t_prog = tc.ms_lifetime_gyr_at_1msun * mi**tc.ms_lifetime_mass_exponent
    if tc.below_ifmr == "no_progenitor_delay":
        seg = np.asarray(tc.ifmr_segments, float)
        t_prog = np.where(m < float(np.min(seg[:, 2] * seg[:, 0] + seg[:, 3])), 0.0, t_prog)
    return np.asarray(t_sys_gyr, float) - t_prog


def wd_log_cooling_age_target(truth: Mapping[str, NDArray[Any]], tc: ps.WDCoolingAgeTargetConfig) -> FloatArray:
    """log target density per dex of t_cool for WD draws (uniform in t on [0, t_sys − t_prog]); NaN-safe."""
    t = np.asarray(truth["wd_cooling_age_gyr"], float)
    avail = wd_available_time_gyr(truth["m2_msun"], 10.0 ** np.asarray(truth["iso_log_age"], float) / 1e9, tc)
    ok = np.isfinite(t) & (t > 0) & np.isfinite(avail) & (avail > 0) & (t < avail)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, np.log(np.where(ok, t * math.log(10.0) / avail, 1.0)), -np.inf)


def co_log_target(truth: Mapping[str, NDArray[Any]], target: ps.CompactObjectTargetConfig, parent: ps.ParentSnapshot | None = None) -> FloatArray:
    """log λ_CO(x | θ_CO) in the proposal's measure (per row, per dex M2, per dex t_cool for WDs, per dex P,
    per unit e or the circular point mass). Needs the truth after :func:`proposal_set.apply_system_photometry`
    (``iso_log_age``, the drawn ``m1_msun``)."""
    comp = np.asarray(truth["co_component"])
    lm2 = np.log10(np.asarray(truth["m2_msun"], float))
    out = np.full(comp.size, -np.inf)
    for name, t in (("wd", target.wd), ("dark", target.dark)):
        sel = comp == COMPONENT_CODES[name]
        out[sel] = math.log(t.rate_per_primary) + _log_power_density(lm2[sel], t)
    wd = comp == COMPONENT_CODES["wd"]
    if wd.any():
        sub = {k: np.asarray(truth[k])[wd] for k in ("wd_cooling_age_gyr", "iso_log_age", "m2_msun")}
        out[wd] = out[wd] + wd_log_cooling_age_target(sub, target.wd_cooling_age)
    o = target.orbit
    p = np.asarray(truth["period_days"], float)
    e = np.asarray(truth["eccentricity"], float)
    out = out + _log_uniform_logpdf(np.log10(p), o.log_p_min, o.log_p_max)
    circ = p <= o.circular_period_days
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        emax = np.clip(1.0 - (p / o.e_max_period_scale_days) ** o.e_max_exponent, 0.0, 1.0)
        if o.eccentricity == "uniform_to_emax":
            le = np.where((e >= 0) & (e < emax), -np.log(emax), -np.inf)
        else:
            le = np.where((e > 0) & (e < emax), np.log(2.0 * e / emax**2), -np.inf)
    out = out + np.where(circ, np.where(e == 0.0, 0.0, -np.inf), le)
    m1 = np.asarray(truth["m1_msun"], float)
    if target.primary_m1_power != 0.0:
        out = out + target.primary_m1_power * np.log(np.maximum(m1 / target.primary_m1_pivot_msun, 1.0))
    if parent is not None:
        out = out + parent.log_row_weight(truth["parent_row"])
    return np.where(np.isfinite(out), out, -np.inf)


def check_co_generations(gen_cfgs: Sequence[ps.ProposalConfig]) -> ps.CompactObjectProposalConfig:
    """All generations are compact-object generations of one primary-model version with one SBC partition."""
    ps.check_single_primary_model(gen_cfgs)
    cos = [_co(g) for g in gen_cfgs]
    keys = {(c.version, c.sbc_holdout_fraction, c.sbc_holdout_salt) for c in cos}
    if len(keys) != 1:
        raise ValueError(f"compact-object generations disagree on version / SBC partition: {sorted(keys)}")
    return cos[0]


def co_importance_weights(
    truth: Mapping[str, NDArray[Any]],
    parent: ps.ParentSnapshot,
    gen_cfgs: Sequence[ps.ProposalConfig],
    target: ps.CompactObjectTargetConfig,
    *,
    partition: Literal["train", "holdout", "all"] = "train",
) -> FloatArray:
    """Deterministic-mixture weights (spec §3.4, §3.7) of the draws in ``partition``; 0 elsewhere.

    The SBC partition thins every generation by the same probability h, so the training draws are
    Σ_j n_j (1 − h) draws from the mixture (h n_j for the held-out ones); the denominator uses those counts.
    ``n_j`` is the number of stored draws of generation j in ``truth``.
    """
    co = check_co_generations(gen_cfgs)
    if target.version != co.version:
        raise ValueError(f"target version {target.version} != generations' {co.version}")
    gen = np.asarray(truth["generation"], np.int64)
    hold = holdout_mask(truth["draw_index"], co.sbc_holdout_fraction, co.sbc_holdout_salt)
    frac = {"train": 1.0 - co.sbc_holdout_fraction, "holdout": co.sbc_holdout_fraction, "all": 1.0}[partition]
    keep = {"train": ~hold, "holdout": hold, "all": np.ones(gen.size, bool)}[partition]
    present = [g for g in gen_cfgs if np.any(gen == g.generation)]  # a generation with no stored draws adds nothing
    n_j = [int(np.sum(gen == g.generation)) * frac for g in present]
    logq = [log_q_co_total_for(truth, parent, g) for g in present]
    w = ps.importance_weights(co_log_target(truth, target, parent), logq, n_j, scale_to_full=parent.scale_to_full)
    return np.where(keep, w, 0.0)


# ---------------------------------------------------------------------------
# Selection-function table
# ---------------------------------------------------------------------------


class SelectionTableConfig(BaseModel):
    """Axes and detection definition of the rung-4 selection-function table (spec §13.3 item 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detection: Literal["orbit_or_acceleration", "orbit", "acceleration"]
    #: axis name (a key of :func:`selection_axis_values`) -> bin edges; values outside the edges are dropped.
    axes: dict[str, tuple[float, ...]]
    #: §13.6 per-cell ESS target; cells below are flagged, never silently trusted.
    min_cell_ess: float = Field(..., gt=0)


class MassRecoveryConfig(BaseModel):
    """Mass-recovery hooks (spec §13.3 item 2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_mc_draws: int = Field(..., ge=10)
    random_seed: int = Field(..., ge=0)
    #: The flux ratio the inversion assumes (the dark-companion hypothesis of the inference).
    assumed_flux_ratio: float = Field(..., ge=0.0)
    #: M1 posterior of the primary: lognormal from the row's isochrone posterior (log_m1_mean, log_m1_sigma).
    m1_posterior: Literal["isochrone_row_lognormal"]
    coverage_levels: tuple[float, ...]
    log10_m2_edges: tuple[float, ...]
    #: #557: prior of the per-object M2 posterior. ``flat_thiele_innes`` = the Gaussian likelihood alone (flat in
    #: A, B, F, G, ϖ, P; the NSS reproduction convention), whose implied prior ∝ a0³ sin³ i in (a0, ω, Ω, i) pushes
    #: M2 high. ``isotropic_log_m2``: isotropic orientation and log-uniform M2 (the injected R4-Q1 law), applied by
    #: importance-reweighting the likelihood draws with w = |d log M2 / d log a0| / (a0⁴ sin² i).
    m2_posterior_prior: Literal["flat_thiele_innes", "isotropic_log_m2"] = "flat_thiele_innes"
    #: #557: gate 4b is evaluated on fitted periods ≤ this (the C1 domain, = the decided rung3.c1_period_max_days);
    #: None = all accepted orbits.
    gate_period_max_days: float | None = Field(None, gt=0)
    #: Spec §13.4 gate 4b thresholds (reported; the gate is evaluated in a later phase).
    max_abs_bias: float = Field(..., gt=0)
    max_coverage_offset: float = Field(..., gt=0)


class Rung4AnalysisConfig(BaseModel):
    """``config/population/rung4_co_v1.yaml``: selection table, mass recovery, contamination mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., pattern=r"^co_v[0-9]+$")
    selection_table: SelectionTableConfig
    mass_recovery: MassRecoveryConfig
    contamination: Literal["declared_stub"]

    @model_validator(mode="after")
    def _levels(self) -> Rung4AnalysisConfig:
        if not all(0 < lv < 1 for lv in self.mass_recovery.coverage_levels):
            raise ValueError("coverage_levels must be in (0, 1)")
        return self


def load_rung4_config(path: str | Path) -> Rung4AnalysisConfig:
    p = Path(path)
    if not p.is_absolute():
        p = ps.repo_root() / p
    return Rung4AnalysisConfig.model_validate(yaml.safe_load(p.read_text()))


def selection_axis_values(truth: Mapping[str, NDArray[Any]]) -> dict[str, FloatArray]:
    """Per-draw truth values of the table axes: M2, P, e, system G, distance, M1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "log10_m2_msun": np.log10(np.asarray(truth["m2_msun"], float)),
            "log10_period_days": np.log10(np.asarray(truth["period_days"], float)),
            "eccentricity": np.asarray(truth["eccentricity"], float),
            "g_sys_mag": np.asarray(truth["phot_g_mean_mag"], float),
            "distance_kpc": 1.0 / np.asarray(truth["parallax_mas"], float),
            "m1_msun": np.asarray(truth["m1_msun"], float),
        }


def detected_flags(outcome: Mapping[str, NDArray[Any]], detection: str) -> NDArray[np.bool_]:
    """Published-solution indicator: accepted orbit, published acceleration, or either."""
    orb = np.asarray(outcome["accepted_orbital"], bool)
    acc = np.asarray(outcome["published_acceleration"], bool)
    return {"orbit": orb, "acceleration": acc, "orbit_or_acceleration": orb | acc}[detection]


@dataclass(frozen=True)
class SelectionTable:
    """Weighted detection fraction per cell (axes in ``axes`` order) with its Monte-Carlo noise.

    ``p_det = Σ w d / Σ w`` (ratio estimator), ``var_p_det = Σ w² (d − p)² / (Σ w)²``; ``ess`` and ``ess_det`` are
    the Kish ESS of all draws and of the detected draws in the cell; ``mc_poisson_ratio = sqrt(Σ w² d / Σ w d)``
    is σ_MC / σ_Poisson of the expected detected count (spec §3.6, gate 4d), relative to the Poisson noise of a
    sample whose total weight equals the weights' scale.
    """

    axes: tuple[str, ...]
    edges: tuple[FloatArray, ...]
    n_draws: NDArray[np.int64]
    n_detected: NDArray[np.int64]
    sum_w: FloatArray
    sum_w2: FloatArray
    sum_w_det: FloatArray
    sum_w2_det: FloatArray
    sum_w2_dev2: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def p_det(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sum_w > 0, self.sum_w_det / self.sum_w, np.nan)

    @property
    def var_p_det(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sum_w > 0, self.sum_w2_dev2 / self.sum_w**2, np.nan)

    @property
    def ess(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sum_w2 > 0, self.sum_w**2 / self.sum_w2, 0.0)

    @property
    def ess_det(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sum_w2_det > 0, self.sum_w_det**2 / self.sum_w2_det, 0.0)

    @property
    def mc_poisson_ratio(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sum_w_det > 0, np.sqrt(self.sum_w2_det / self.sum_w_det), np.nan)


def build_selection_table(
    values: Mapping[str, ArrayLike],
    weights: ArrayLike,
    detected: ArrayLike,
    axes: Mapping[str, Sequence[float]],
    *,
    mask: ArrayLike | None = None,
    meta: Mapping[str, Any] | None = None,
) -> SelectionTable:
    """Bin draws (weights ``w``, detection ``d``) on ``axes`` (name -> edges; draws outside any edge are dropped)."""
    w = np.asarray(weights, float)
    d = np.asarray(detected, bool).astype(float)
    keep = np.ones(w.size, bool) if mask is None else np.asarray(mask, bool).copy()
    names = tuple(axes)
    edges = tuple(np.asarray(axes[a], float) for a in names)
    idx = []
    for a, e in zip(names, edges):
        v = np.asarray(values[a], float)
        i = np.searchsorted(e, v, side="right") - 1
        keep &= np.isfinite(v) & (i >= 0) & (i < e.size - 1)
        idx.append(np.clip(i, 0, e.size - 2))
    shape = tuple(e.size - 1 for e in edges)
    flat = np.ravel_multi_index(tuple(i[keep] for i in idx), shape) if names else np.zeros(int(keep.sum()), np.int64)
    size = int(np.prod(shape)) if names else 1
    wk, dk = w[keep], d[keep]

    def acc(x: FloatArray) -> FloatArray:
        return np.bincount(flat, weights=x, minlength=size).reshape(shape)

    sw, swd = acc(wk), acc(wk * dk)
    p = np.where(sw > 0, swd / np.where(sw > 0, sw, 1.0), 0.0).ravel()
    dev2 = (dk - p[flat]) ** 2
    return SelectionTable(
        axes=names, edges=edges,
        n_draws=np.bincount(flat, minlength=size).reshape(shape).astype(np.int64),
        n_detected=np.bincount(flat, weights=dk, minlength=size).reshape(shape).astype(np.int64),
        sum_w=sw, sum_w2=acc(wk * wk), sum_w_det=swd, sum_w2_det=acc(wk * wk * dk), sum_w2_dev2=acc(wk * wk * dev2),
        meta=dict(meta or {}),
    )


def write_selection_tables(path: str | Path, tables: Mapping[str, SelectionTable], *, attrs: Mapping[str, Any]) -> Path:
    """One HDF5 group per component; root attrs JSON-encoded. Stand-ins are added by the caller (run_validity)."""
    import json

    import h5py

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as h:
        for comp, t in tables.items():
            g = h.create_group(comp)
            g.attrs["axes"] = json.dumps(list(t.axes))
            g.attrs["meta_json"] = json.dumps(t.meta, sort_keys=True, default=str)
            for a, e in zip(t.axes, t.edges):
                g.create_dataset(f"edges/{a}", data=e)
            for k in ("n_draws", "n_detected", "sum_w", "sum_w2", "sum_w_det", "sum_w2_det", "sum_w2_dev2"):
                g.create_dataset(k, data=getattr(t, k))
        for k, v in attrs.items():
            h.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v, sort_keys=True, default=str)
    return p


def read_selection_tables(path: str | Path) -> dict[str, SelectionTable]:
    import json

    import h5py

    out: dict[str, SelectionTable] = {}
    with h5py.File(Path(path), "r") as h:
        for comp in h.keys():
            g = h[comp]
            axes = tuple(json.loads(g.attrs["axes"]))
            out[comp] = SelectionTable(
                axes=axes, edges=tuple(g[f"edges/{a}"][()] for a in axes),
                **{k: g[k][()] for k in ("n_draws", "n_detected", "sum_w", "sum_w2", "sum_w_det", "sum_w2_det", "sum_w2_dev2")},
                meta=json.loads(g.attrs["meta_json"]),
            )
    return out


# ---------------------------------------------------------------------------
# Mass recovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitCovariance:
    """Fitted 12-parameter solution (gaiamock order, :data:`proposal_set.ORBIT_PARAMETER_NAMES`) and its covariance.

    ``max_sigma_mismatch``: max over the parameters the cascade vector reports (ϖ, A, B, F, G, P, φ_p, e) of
    |σ_cov / σ_cascade − 1|, a self-check that this is the covariance gaiamock's own uncertainties came from.
    """

    p0: FloatArray
    covariance: FloatArray
    chi2: float
    error_scale: float
    max_sigma_mismatch: float


def orbit_covariance_from_epochs(gaiamock: Any, c_funcs: Any, epochs: Sequence[ArrayLike], cascade: Sequence[float], *, step: float) -> OrbitCovariance:
    """Full covariance of one orbital cascade fit from its epochs (spec §13.3; no diagonal fallback).

    The nonlinear (P, φ_p, e) are the cascade's; gaiamock's ``get_astrometric_chi2`` gives the χ² and the nine
    linear parameters there (its own linear solve, so p0 is gaiamock's best fit). The covariance is
    (JᵀJ)⁻¹ c² with J the central-difference Jacobian (step ``step``) of gaiamock's
    ``get_astrometric_residuals_12par`` and c = sqrt(χ²_red / (1 − 2/(9ν))³), ν = N − 12, gaiamock's own
    error-inflation factor. A singular JᵀJ returns NaN (counted, never replaced by a diagonal).
    """
    t, psi, pf, obs, err = (np.asarray(a, dtype=np.float64) for a in epochs)
    v = [float(x) for x in cascade]
    period, phi_p, ecc = v[_CASCADE_INDEX["period"]], v[_CASCADE_INDEX["phi_p"]], v[_CASCADE_INDEX["ecc"]]
    chi2, mu = gaiamock.get_astrometric_chi2(t, psi, pf, obs, err, period, phi_p, ecc, c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, b_ti, g_ti, a_ti, f_ti = (float(x) for x in mu)
    p0 = np.array([ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, a_ti, b_ti, f_ti, g_ti], dtype=np.float64)

    def resid(theta: FloatArray) -> FloatArray:
        return np.asarray(gaiamock.get_astrometric_residuals_12par(t, psi, pf, obs, err, theta, c_funcs), dtype=np.float64)

    jac = np.empty((t.size, p0.size))
    for i in range(p0.size):
        up, dn = p0.copy(), p0.copy()
        up[i] += step
        dn[i] -= step
        jac[:, i] = (resid(up) - resid(dn)) / (2.0 * step)
    nu = t.size - p0.size
    cc = math.sqrt((float(chi2) / nu) / (1.0 - 2.0 / (9.0 * nu)) ** 3) if nu > 0 else float("nan")
    try:
        cov = np.linalg.inv(jac.T @ jac) * cc**2
    except np.linalg.LinAlgError:
        cov = np.full((p0.size, p0.size), np.nan)
    if not np.all(np.isfinite(cov)):
        cov = np.full((p0.size, p0.size), np.nan)
    cov = 0.5 * (cov + cov.T)
    pos = {"parallax": 2, "period": 5, "ecc": 6, "phi_p": 7, "A": 8, "B": 9, "F": 10, "G": 11}
    with np.errstate(invalid="ignore", divide="ignore"):
        mism = [abs(math.sqrt(cov[j, j]) / v[_CASCADE_INDEX["sig_" + k]] - 1.0) for k, j in pos.items()]
    return OrbitCovariance(p0=p0, covariance=cov, chi2=float(chi2), error_scale=cc, max_sigma_mismatch=float(np.nanmax(mism)) if np.all(np.isfinite(cov)) else float("nan"))


def cascade_parameter_set(p0: ArrayLike, covariance: ArrayLike) -> Any:
    """:class:`schemas.ParameterSet` of a mock cascade fit (canonical NSS names; full 12×12 covariance)."""
    from darkhunter_pop.schemas import ParameterSet

    cov = np.asarray(covariance, float)
    if cov.shape != (12, 12) or not np.all(np.isfinite(cov)):
        raise ValueError("cascade covariance must be a finite 12x12 matrix (no diagonal fallback)")
    return ParameterSet(names=list(ps.ORBIT_PARAMETER_NAMES), values=[float(x) for x in np.asarray(p0, float)],
                        covariance=(0.5 * (cov + cov.T)).tolist(), provenance=CASCADE_COVARIANCE_PROVENANCE)


def m1_posterior_draws(parent: ps.ParentSnapshot, row: int, n: int, rng: np.random.Generator, mode: str) -> FloatArray:
    """M1 posterior draws of a parent row: lognormal from its isochrone (log_m1_mean, log_m1_sigma)."""
    if mode != "isochrone_row_lognormal":
        raise ValueError(f"unknown m1_posterior {mode!r}")
    if parent.isochrone is None:
        raise ValueError("isochrone_row_lognormal needs an isochrone-mode parent")
    mu = float(np.asarray(parent.isochrone["log_m1_mean"], float)[row])
    sd = float(np.asarray(parent.isochrone["log_m1_sigma"], float)[row])
    if not (np.isfinite(mu) and np.isfinite(sd) and sd >= 0):
        return np.full(n, np.nan)
    return 10.0 ** (mu + sd * rng.standard_normal(n))


def m2_posterior_draws(
    p0: ArrayLike, covariance: ArrayLike, m1_msun: ArrayLike, *, n_draws: int, random_seed: int, flux_ratio: float,
    prior: str,
) -> tuple[FloatArray, FloatArray, FloatArray, Any]:
    """(m_f, M2, weight) per Monte-Carlo draw of one orbit, and the covariance factorization (#557).

    The likelihood draws are the full-covariance Gaussian of the cascade solution, propagated through
    ``mc_mass_function`` (``factorize_covariance`` → ``sample_multivariate_normal`` → ``propagate_thiele_innes_draws``),
    the real-row code path. ``prior`` sets the per-draw weight (see :class:`MassRecoveryConfig`):
    ``flat_thiele_innes`` → 1; ``isotropic_log_m2`` → |d log M2 / d log a0| / (a0⁴ sin² i), where |∂(A,B,F,G)/∂(a0, ω,
    Ω, i)| = a0³ sin³ i, the isotropic prior is ∝ sin i, log-uniform M2 maps to p(a0) ∝ |d log M2/d log a0| / a0, and
    d log a0 / d log M2 = 1 − (2/3) M2 / (M1 + M2) for a dark companion (f = 0) at fixed M1 and P.
    """
    from darkhunter_pop import mc_mass_function as mcm

    names = list(ps.ORBIT_PARAMETER_NAMES)
    idx = {k: names.index(k) for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes", "parallax", "period")}
    sol = cascade_parameter_set(p0, covariance)
    import inspect

    dflt = inspect.signature(mcm.propagate_nss_solution).parameters  # the real-row path's eigenvalue floors
    factor = mcm.factorize_covariance(sol.covariance_array(), eig_rel_floor=dflt["eig_rel_floor"].default,
                                      eig_abs_floor=dflt["eig_abs_floor"].default)
    d = mcm.sample_multivariate_normal(sol.values_array(), factor, n_draws, np.random.default_rng(int(random_seed)))
    a, b, f_, g = (d[:, idx[k]] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    a0, m_f, m2 = mcm.propagate_thiele_innes_draws(a, b, f_, g, d[:, idx["parallax"]], d[:, idx["period"]], m1_msun, flux_ratio=flux_ratio)
    if prior == "flat_thiele_innes":
        w = np.ones(n_draws)
    elif prior == "isotropic_log_m2":
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_i = (a * g - b * f_) / a0**2
            sin2 = np.clip(1.0 - cos_i**2, 0.0, 1.0)
            m1 = np.broadcast_to(np.asarray(m1_msun, float), m2.shape)
            dl = 1.0 - (2.0 / 3.0) * m2 / (m1 + m2)
            w = 1.0 / (np.abs(dl) * a0**4 * sin2)
        w = np.where(np.isfinite(w) & np.isfinite(m2) & (m2 > 0), w, 0.0)
    else:
        raise ValueError(f"unknown m2_posterior_prior {prior!r}")
    return np.asarray(m_f, float), np.asarray(m2, float), w, factor.method


def recover_masses(
    truth: Mapping[str, NDArray[Any]],
    outcome: Mapping[str, NDArray[Any]],
    parent: ps.ParentSnapshot,
    cfg: MassRecoveryConfig,
    *,
    select: ArrayLike,
) -> dict[str, FloatArray]:
    """Per selected draw (accepted orbits with a stored covariance): the m_f and M2 posterior from the cascade
    covariance and the primary's M1 posterior, through ``mc_mass_function.propagate_nss_solution`` (the real-row
    code path), assuming flux ratio ``cfg.assumed_flux_ratio``. Returns per-draw arrays (NaN where not selected or
    the covariance is singular): ``m_f_med``, ``m2_med``, ``m2_pit`` (fraction of posterior draws below the truth M2),
    ``factorization`` (0 Cholesky, 1 nugget, 2 eigen clip, −1 none)."""
    from darkhunter_pop import mc_mass_function as mcm

    sel = np.flatnonzero(np.asarray(select, bool))
    n = np.asarray(truth["draw_index"]).size
    out = {k: np.full(n, np.nan) for k in ("m_f_med", "m2_med", "m2_pit", "factorization", "mc_ess")}
    codes = {mcm.CovarianceFactorization.CHOLESKY: 0, mcm.CovarianceFactorization.CHOLESKY_NUGGET: 1, mcm.CovarianceFactorization.EIGEN_CLIP: 2}
    p0s, covs = np.asarray(outcome["orbit_p0"], float), np.asarray(outcome["orbit_cov"], float)
    m2_true = np.asarray(truth["m2_msun"], float)
    rows = np.asarray(truth["parent_row"], np.int64)
    for i in sel:
        if not np.all(np.isfinite(covs[i])):
            out["factorization"][i] = -1
            continue
        seed = int(cfg.random_seed) ^ int(truth["draw_index"][i])
        m1 = m1_posterior_draws(parent, int(rows[i]), cfg.n_mc_draws, np.random.default_rng([seed, 1]), cfg.m1_posterior)
        if not np.all(np.isfinite(m1)):
            continue
        m_f, m2, wq, method = m2_posterior_draws(p0s[i], covs[i], m1, n_draws=cfg.n_mc_draws, random_seed=seed,
                                                 flux_ratio=cfg.assumed_flux_ratio, prior=cfg.m2_posterior_prior)
        ok = np.isfinite(m2) & (wq > 0)
        if not ok.any():
            continue
        m2o, wo = m2[ok], wq[ok]
        order = np.argsort(m2o)
        cw = np.cumsum(wo[order]) / wo.sum()
        out["m_f_med"][i] = float(np.nanmedian(m_f))
        out["m2_med"][i] = float(m2o[order][min(np.searchsorted(cw, 0.5), m2o.size - 1)])
        out["m2_pit"][i] = float(np.sum(wo[m2o < m2_true[i]]) / wo.sum())
        out["mc_ess"][i] = ps.kish_ess(wo)
        out["factorization"][i] = codes[method]
    return out


def recovery_summary(
    m2_true: ArrayLike, recovered: Mapping[str, ArrayLike], weights: ArrayLike, cfg: MassRecoveryConfig, *, mask: ArrayLike | None = None,
) -> list[dict[str, Any]]:
    """Gate-4b quantities per M2 bin (spec §13.4): weighted median fractional bias of the posterior-median M2 and
    the weighted coverage of the central intervals (|PIT − 0.5| < level / 2), with N and Kish ESS."""
    t = np.asarray(m2_true, float)
    med = np.asarray(recovered["m2_med"], float)
    pit = np.asarray(recovered["m2_pit"], float)
    w = np.asarray(weights, float)
    ok = np.isfinite(med) & np.isfinite(pit) & (w > 0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    e = np.asarray(cfg.log10_m2_edges, float)
    lt = np.log10(t)
    rows = []
    for a, b in zip(e[:-1], e[1:]):
        s = ok & (lt >= a) & (lt < b)
        rec: dict[str, Any] = {"log10_m2_lo": float(a), "log10_m2_hi": float(b), "n": int(s.sum()), "ess": ps.kish_ess(w[s])}
        if s.any():
            frac = med[s] / t[s] - 1.0
            order = np.argsort(frac)
            cw = np.cumsum(w[s][order]) / w[s].sum()
            rec["median_bias"] = float(frac[order][min(np.searchsorted(cw, 0.5), frac.size - 1)])
            for lv in cfg.coverage_levels:
                rec[f"coverage_{lv:g}"] = float(np.sum(w[s] * (np.abs(pit[s] - 0.5) < lv / 2.0)) / w[s].sum())
                rec[f"coverage_{lv:g}_mc_sigma"] = float(math.sqrt(lv * (1 - lv) / rec["ess"])) if rec["ess"] > 0 else float("nan")
        rows.append(rec)
    return rows


def gate_4b(rows: Sequence[Mapping[str, Any]], cfg: MassRecoveryConfig, *, min_ess: float) -> dict[str, Any]:
    """Spec §13.4 gate 4b on :func:`recovery_summary` rows with ESS ≥ ``min_ess``: |bias| < ``max_abs_bias`` and
    |coverage − level| ≤ ``max_coverage_offset`` per bin (point-wise), plus the χ² of the coverage deviations against
    their MC σ per level (the noise-aware check reported beside it, #557). The point-wise result is the gate."""
    from scipy import stats

    used = [r for r in rows if r.get("ess", 0) >= min_ess and "median_bias" in r]
    out: dict[str, Any] = {"bins_used": len(used), "bias_ok": all(abs(r["median_bias"]) < cfg.max_abs_bias for r in used)}
    point = out["bias_ok"]
    for lv in cfg.coverage_levels:
        dev = [r[f"coverage_{lv:g}"] - lv for r in used]
        ok = [abs(d) <= cfg.max_coverage_offset for d in dev]
        chi = float(sum((d / r[f"coverage_{lv:g}_mc_sigma"]) ** 2 for d, r in zip(dev, used)))
        out[f"coverage_{lv:g}"] = {"bins_within": int(sum(ok)), "chi2": chi, "dof": len(used),
                                   "p_value": float(stats.chi2.sf(chi, len(used))) if used else float("nan")}
        point = point and all(ok)
    out["passed_pointwise"] = bool(point and used)
    return out


# ---------------------------------------------------------------------------
# Contamination (phase 1: declared stand-in)
# ---------------------------------------------------------------------------


def contamination_stand_in(version: str) -> Any:
    """The rung-4 contamination estimate (spec §13.3 item 3) is not computed in phase 1; this declares it."""
    from darkhunter_pop.schemas import SyntheticStandIn

    return SyntheticStandIn(
        name="rung4_contamination_not_computed",
        stage="selection_function_astrometric",
        kind="disabled_path",
        replaces="expected luminous-companion and spurious-orbit contamination per cell (spec §13.3 item 3, gate 4e)",
        description=(
            f"Rung-4 phase 1 ({version}) builds the compact-object selection-function table and the mass-recovery "
            "hooks only. The contamination model (rung-3 luminous fit + §12.16 spurious template, per cell) is not "
            "computed, so no table written here may be used for inference until it is."
        ),
        config_keys=["contamination"],
        values={"contamination": "declared_stub"},
    )
