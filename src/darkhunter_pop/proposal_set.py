"""Importance-reweighted mock proposal set (docs/MOCK_POPULATION_SPEC.md §1, §3; #391).

One broad proposal ``q(x)`` over ``x = (parent star s, M2, f, log P, e, orientation)`` is
drawn once, run through ``gaiamock_mod``'s cascade, and stored per draw with its truth,
seeds, outcome and fitted solution. Any population ``θ`` is then a reweighting

    w_i(θ) = (N_full / N_snap) λ(x_i | θ) / Σ_j n_j q_j(x_i)

(deterministic-mixture denominator over every proposal generation ``j``), with Kish ESS
per bin as the trust diagnostic. ``λ`` is a Poisson intensity (expected companions per
parent system), so ``Σ w_i 1[O_i]`` is an expected count relative to the parent sample.

What lives here: the ``gaia_source`` parent query and snapshot loader, the parent M1 via
the data-side TAG10 path (``mass_derivation``, called, never modified), the proposal
densities and sampler, the per-draw gaiamock call (seeded per #371), the Moe & Di Stefano
luminous-companion target intensity used for rung 2, the weights and the ESS.

What does not: anything gaiamock owns (docs/GAIAMOCK_API.md); the compact-object
mixture (``population_model``); the stage wiring (a follow-up, spec §3.8).

Every choice the spec leaves open (§8, MP-Q*) is an explicit config field whose name
starts with ``provisional_`` and whose value is recorded in the artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import moe_distefano as mds
from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import OrbitalSolutionCutsConfig, PipelineConfig
from darkhunter_pop.forward_model import (
    classify_cascade_result,
    mock_global_rng_seeds,
    seeded_global_rng,
)
from darkhunter_pop.janssens_mass import segments_from_table

FloatArray = NDArray[np.float64]

#: Rows in ``gaiadr3.gaia_source`` (= range of ``random_index``), for subsample scaling.
GAIA_SOURCE_TOTAL_ROWS: int = 1_811_709_771

#: ``SeedSequence`` spawn-key stream offset for proposal generations (forward_model uses 0, 1).
PROPOSAL_RNG_STREAM_BASE: int = 100

#: Number of entries in the gaiamock cascade return vector.
CASCADE_VECTOR_LENGTH: int = 23

GAIA_SOURCE_PARENT_COLUMNS: tuple[str, ...] = (
    "source_id",
    "random_index",
    "ra",
    "dec",
    "l",
    "b",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
    "ruwe",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "ipd_gof_harmonic_amplitude",
    "teff_msc1",
    "teff_msc1_upper",
    "teff_msc1_lower",
    "logg_msc1",
    "logg_msc1_upper",
    "logg_msc1_lower",
    "mh_msc",
    "mh_msc_upper",
    "mh_msc_lower",
    "teff_gspphot",
    "teff_gspphot_upper",
    "teff_gspphot_lower",
    "logg_gspphot",
    "logg_gspphot_upper",
    "logg_gspphot_lower",
    "mh_gspphot",
    "mh_gspphot_upper",
    "mh_gspphot_lower",
)

_AP_COLUMNS = frozenset(c for c in GAIA_SOURCE_PARENT_COLUMNS if "msc" in c or "gspphot" in c)


def build_gaia_source_parent_adql(*, k: int, parallax_floor_mas: float, g_max: float) -> str:
    """ADQL for the uniform ``random_index < k`` parent subsample (spec §1.3).

    Only G and parallax are selected on. RUWE, visibility periods and the Halbwachs et al.
    (2023) IPD / C* columns are fetched but never cut on (they are outcomes or MP-Q3).
    """
    if k <= 0 or k > GAIA_SOURCE_TOTAL_ROWS:
        raise ValueError(f"k must be in (0, {GAIA_SOURCE_TOTAL_ROWS}], got {k}")
    cols = ",\n  ".join(
        (f"ap.{c}" if c in _AP_COLUMNS else f"gs.{c}") for c in GAIA_SOURCE_PARENT_COLUMNS
    )
    return (
        f"SELECT\n  {cols}\n"
        "FROM gaiadr3.gaia_source AS gs\n"
        "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id\n"
        f"WHERE gs.random_index < {int(k)}\n"
        f"  AND gs.phot_g_mean_mag < {float(g_max)!r}\n"
        f"  AND gs.parallax > {float(parallax_floor_mas)!r}"
    )


# ---------------------------------------------------------------------------
# Config (fragment config/fragments/proposal_set.yaml; merged at stage integration)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParentProposalConfig(_Strict):
    """``q(s) = (1-λ) uniform + λ ∝ min(ϖ, ϖ_cap)^β`` over usable snapshot rows."""

    uniform_fraction: float = Field(..., ge=0.0, le=1.0)
    parallax_power: float = Field(..., ge=0.0)
    parallax_cap_mas: float = Field(..., gt=0.0)


class M2ProposalConfig(_Strict):
    """Mixture in log M2: log q ~ U[log q_min, log q_max] (tied to M1) + log-uniform M2."""

    q_tied_weight: float = Field(..., ge=0.0, le=1.0)
    q_min: float = Field(..., gt=0.0)
    q_max: float = Field(..., gt=0.0)
    m2_min_msun: float = Field(..., gt=0.0)
    m2_max_msun: float = Field(..., gt=0.0)


class FluxProposalConfig(_Strict):
    """Point mass at f = 0 (``dark_fraction``) else a mixture in log10 f.

    The luminous part is ``relation_weight × N(log10 f_rel(M1, M2), relation_sigma_dex)``
    plus a defensive ``U[log_f_min, log_f_max]``. ``f_rel`` is the mass–luminosity relation
    named by ``relation`` and only shapes efficiency; the target relation is separate.
    """

    dark_fraction: float = Field(..., ge=0.0, le=1.0)
    relation: Literal["janssens2022"]
    relation_weight: float = Field(..., ge=0.0, le=1.0)
    relation_sigma_dex: float = Field(..., gt=0.0)
    log_f_min: float
    log_f_max: float


class PeriodProposalConfig(_Strict):
    """Mixture in log P: a core log-uniform + a defensive log-uniform."""

    core_weight: float = Field(..., ge=0.0, le=1.0)
    core_log_p_min: float
    core_log_p_max: float
    log_p_min: float
    log_p_max: float


class EccentricityProposalConfig(_Strict):
    """e = 0 for P <= ``circular_period_days``, else U(0, ``e_cap``)."""

    e_cap: float = Field(..., gt=0.0, lt=1.0)
    circular_period_days: float = Field(..., gt=0.0)


class AccelerationPublicationConfig(_Strict):
    """Published acceleration solutions (El-Badry et al. 2024 §5.2.1)."""

    significance_min: float
    f2_max_seven_parameter: float


class ProposalConfig(_Strict):
    generation: int = Field(..., ge=0)
    n_draws: int = Field(..., ge=1)
    base_seed: int = Field(..., ge=0)
    data_release: Literal["dr3"]
    ruwe_min: float
    skip_acceleration: bool
    parent: ParentProposalConfig
    m2: M2ProposalConfig
    flux: FluxProposalConfig
    period: PeriodProposalConfig
    eccentricity: EccentricityProposalConfig
    acceleration_publication: AccelerationPublicationConfig
    # Generation-time open questions (spec §3.5, §7). Recorded in the artifact.
    provisional_parallax_floor_mas: float = Field(..., gt=0.0)  # MP-Q1
    provisional_halbwachs_ipd_cstar_cuts: Literal["not_applied"]  # MP-Q3
    provisional_truth_parallax: Literal["measured_parallax"]  # MP-Q4
    provisional_m1: Literal["tag10_point_drop_unresolved"]  # MP-Q5
    provisional_light_split: Literal["observed_g_is_system_total"]  # MP-Q6


class MdS17TargetConfig(_Strict):
    """Rung-2 target: MdS17 at published parameters, luminous companions only."""

    table_path: str
    provisional_m1_interpolation: mds.M1Interpolation  # MP-Q10
    provisional_eta_floor: float = Field(..., gt=-1.0)  # MP-Q11
    provisional_low_mass_policy: Literal["log_linear_to_zero"]  # MP-Q7
    provisional_low_mass_anchor_msun: float = Field(..., gt=0.0)
    provisional_low_mass_zero_msun: float = Field(..., gt=0.0)
    provisional_multiplicity: Literal["poisson_intensity"]  # MP-Q9
    provisional_mass_luminosity: Literal["janssens2022"]  # MP-Q13
    provisional_flux_sigma_dex: float = Field(..., gt=0.0)  # MP-Q13
    provisional_compact_mixture: Literal["none"]  # MP-Q17

    @model_validator(mode="after")
    def _low_mass_order(self) -> MdS17TargetConfig:
        if not self.provisional_low_mass_zero_msun < self.provisional_low_mass_anchor_msun:
            raise ValueError("provisional_low_mass_zero_msun must be < anchor")
        return self


class ProposalSetFragment(_Strict):
    """Whole ``config/fragments/proposal_set.yaml``."""

    proposal: ProposalConfig
    target_mds17: MdS17TargetConfig


def load_proposal_set_fragment(path: str | Path) -> ProposalSetFragment:
    """Validate the proposal-set fragment (relative paths resolve from the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return ProposalSetFragment.model_validate(yaml.safe_load(p.read_text()))


def fragment_fingerprint(fragment: ProposalSetFragment) -> str:
    """Short SHA256 of the proposal section (keys the artifact; target is reweight-only)."""
    blob = json.dumps(fragment.proposal.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Parent snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentSnapshot:
    """A ``gaia_source`` parent subsample with data-side TAG10 M1 per row.

    ``usable`` marks rows the proposal may draw (finite astrometry and photometry, an
    M1 from TAG10, parallax above the provisional floor). ``scale_to_full`` is
    ``N_full / N_snap = GAIA_SOURCE_TOTAL_ROWS / K``.
    """

    columns: Mapping[str, NDArray[Any]]
    m1_msun: FloatArray
    m1_source: NDArray[np.str_]
    usable: NDArray[np.bool_]
    meta: Mapping[str, Any]
    scale_to_full: float
    path: Path

    @property
    def n_rows(self) -> int:
        return int(self.m1_msun.size)


def tag10_m1_for_rows(
    columns: Mapping[str, NDArray[Any]], config: PipelineConfig
) -> tuple[FloatArray, NDArray[np.str_]]:
    """M1 per row via ``mass_derivation`` (MSC → GSP-Phot → TAG10), exactly as the data side.

    Rows with no resolvable atmosphere get NaN and source ``"none"``. ``mass_derivation``
    is imported and called, never modified.
    """
    from darkhunter_pop.mass_derivation import (
        derive_tag10_m1_r1,
        resolve_atmosphere_from_extras,
    )

    n = int(np.asarray(columns["source_id"]).size)
    names = [c for c in GAIA_SOURCE_PARENT_COLUMNS if c in _AP_COLUMNS]
    m1 = np.full(n, np.nan)
    src = np.full(n, "none", dtype="<U8")
    arrays = {c: np.asarray(columns[c], dtype=np.float64) for c in names}
    for i in range(n):
        extras = {c: float(arrays[c][i]) for c in names if np.isfinite(arrays[c][i])}
        atm = resolve_atmosphere_from_extras(extras)
        if atm is None:
            continue
        try:
            pset = derive_tag10_m1_r1(atm, config)
        except (ValueError, OverflowError):
            continue
        val = float(pset.values[0])
        if np.isfinite(val) and val > 0:
            m1[i] = val
            src[i] = atm.source
    return m1, src


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parent_snapshot(
    snapshot_dir: str | Path,
    config: PipelineConfig,
    *,
    parallax_floor_mas: float,
    m1_cache: bool = True,
) -> ParentSnapshot:
    """Load ``parent.h5`` + ``meta.yaml``, verify the checksum, attach TAG10 M1.

    ``m1_cache`` stores the per-row M1 beside the snapshot (``m1_tag10_<cfg>.npz``, keyed
    by the ``mass_calibration`` section) so repeated loads skip the per-row TAG10 loop.
    """
    import h5py

    d = Path(snapshot_dir)
    meta = yaml.safe_load((d / "meta.yaml").read_text())
    h5 = d / "parent.h5"
    got = _sha256(h5)
    if got != meta["parent_h5_sha256"]:
        raise ValueError(f"parent snapshot checksum mismatch: {got} != {meta['parent_h5_sha256']}")
    with h5py.File(h5, "r") as handle:
        cols = {name: handle[name][()] for name in handle.keys()}
    mc_blob = json.dumps(config.mass_calibration.model_dump(mode="json"), sort_keys=True)
    mc_key = hashlib.sha256(mc_blob.encode()).hexdigest()[:10]
    cache = d / f"m1_tag10_{mc_key}.npz"
    if m1_cache and cache.exists():
        z = np.load(cache)
        m1, src = z["m1_msun"], z["m1_source"]
    else:
        m1, src = tag10_m1_for_rows(cols, config)
        if m1_cache:
            np.savez(cache, m1_msun=m1, m1_source=src)
    finite = np.ones(m1.size, dtype=bool)
    for name in ("ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag"):
        finite &= np.isfinite(np.asarray(cols[name], dtype=np.float64))
    usable = finite & np.isfinite(m1) & (np.asarray(cols["parallax"]) > parallax_floor_mas)
    k = int(meta["random_index_max_exclusive"])
    return ParentSnapshot(
        columns=cols,
        m1_msun=np.asarray(m1, dtype=np.float64),
        m1_source=np.asarray(src),
        usable=usable,
        meta=meta,
        scale_to_full=float(meta["gaia_source_total_rows"]) / float(k),
        path=d,
    )


# ---------------------------------------------------------------------------
# Mass–luminosity (luminous-companion flux ratio)
# ---------------------------------------------------------------------------


def janssens_absolute_g(mass_msun: ArrayLike) -> FloatArray:
    """M_G(M) from the frozen Janssens et al. (2022) Table 1 segments; NaN out of range."""
    m = np.asarray(mass_msun, dtype=np.float64)
    out = np.full(m.shape, np.nan)
    for seg in segments_from_table():
        sel = (m >= seg.m_low) & (m <= seg.m_up)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sel & np.isnan(out), seg.a * np.log10(m) + seg.b, out)
    return out


def relation_log10_flux_ratio(m1_msun: ArrayLike, m2_msun: ArrayLike) -> FloatArray:
    """log10 of the G-band flux ratio f = F2/F1 from Janssens M_G(M) for both stars."""
    return -0.4 * (janssens_absolute_g(m2_msun) - janssens_absolute_g(m1_msun))


# ---------------------------------------------------------------------------
# Proposal densities and sampler
# ---------------------------------------------------------------------------


def parent_proposal_probabilities(parallax_mas: ArrayLike, usable: ArrayLike, cfg: ParentProposalConfig) -> FloatArray:
    """Normalized ``q(s)`` over snapshot rows (zero for unusable rows)."""
    plx = np.asarray(parallax_mas, dtype=np.float64)
    ok = np.asarray(usable, dtype=bool)
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise ValueError("no usable parent rows")
    uni = np.where(ok, 1.0 / n_ok, 0.0)
    tilt = np.where(ok, np.minimum(np.where(ok, plx, 0.0), cfg.parallax_cap_mas) ** cfg.parallax_power, 0.0)
    tilt = tilt / tilt.sum()
    return cfg.uniform_fraction * uni + (1.0 - cfg.uniform_fraction) * tilt


def log_q_log_m2(log_m2: ArrayLike, log_m1: ArrayLike, cfg: M2ProposalConfig) -> FloatArray:
    """log density of the M2 proposal per dex of M2."""
    lm2 = np.asarray(log_m2, float)
    lm1 = np.asarray(log_m1, float)
    lq = lm2 - lm1
    wq = cfg.q_tied_weight
    width_q = math.log10(cfg.q_max) - math.log10(cfg.q_min)
    width_m = math.log10(cfg.m2_max_msun) - math.log10(cfg.m2_min_msun)
    in_q = (lq >= math.log10(cfg.q_min)) & (lq <= math.log10(cfg.q_max))
    in_m = (lm2 >= math.log10(cfg.m2_min_msun)) & (lm2 <= math.log10(cfg.m2_max_msun))
    dens = wq * in_q / width_q + (1.0 - wq) * in_m / width_m
    with np.errstate(divide="ignore"):
        return np.log(dens)


def log_q_flux(
    log10_f: ArrayLike, is_dark: ArrayLike, m1_msun: ArrayLike, m2_msun: ArrayLike, cfg: FluxProposalConfig
) -> FloatArray:
    """log proposal for the flux ratio: log(ρ_dark) for dark draws, else log of the
    luminous mixture density per dex of f times (1 - ρ_dark)."""
    lf = np.asarray(log10_f, float)
    dark = np.asarray(is_dark, bool)
    rel = relation_log10_flux_ratio(m1_msun, m2_msun)
    sig = cfg.relation_sigma_dex
    with np.errstate(invalid="ignore"):
        gauss = np.exp(-0.5 * ((lf - rel) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    gauss = np.where(np.isfinite(gauss), gauss, 0.0)
    width = cfg.log_f_max - cfg.log_f_min
    uni = ((lf >= cfg.log_f_min) & (lf <= cfg.log_f_max)) / width
    # Without a finite relation value the relation component cannot be drawn, so its
    # weight moves to the uniform component (matches the sampler).
    has_rel = np.isfinite(rel)
    wr = np.where(has_rel, cfg.relation_weight, 0.0)
    lum = (1.0 - wr) * uni + wr * gauss
    with np.errstate(divide="ignore"):
        lum_log = np.log(1.0 - cfg.dark_fraction) + np.log(lum)
        dark_log = np.full(lf.shape, math.log(cfg.dark_fraction) if cfg.dark_fraction > 0 else -np.inf)
    return np.where(dark, dark_log, lum_log)


def log_q_log_p(log_p: ArrayLike, cfg: PeriodProposalConfig) -> FloatArray:
    lp = np.asarray(log_p, float)
    core = ((lp >= cfg.core_log_p_min) & (lp <= cfg.core_log_p_max)) / (cfg.core_log_p_max - cfg.core_log_p_min)
    wide = ((lp >= cfg.log_p_min) & (lp <= cfg.log_p_max)) / (cfg.log_p_max - cfg.log_p_min)
    with np.errstate(divide="ignore"):
        return np.log(cfg.core_weight * core + (1.0 - cfg.core_weight) * wide)


def log_q_ecc(ecc: ArrayLike, period_days: ArrayLike, cfg: EccentricityProposalConfig) -> FloatArray:
    """0 (probability 1) for circular-class draws; else log uniform density on [0, e_cap)."""
    e = np.asarray(ecc, float)
    p = np.asarray(period_days, float)
    circ = p <= cfg.circular_period_days
    inside = (e >= 0) & (e < cfg.e_cap)
    with np.errstate(divide="ignore"):
        cont = np.where(inside, -math.log(cfg.e_cap), -np.inf)
    return np.where(circ, np.where(e == 0.0, 0.0, -np.inf), cont)


def sample_proposal(
    parent: ParentSnapshot, cfg: ProposalConfig, *, draw_index_offset: int = 0
) -> dict[str, NDArray[Any]]:
    """Draw ``cfg.n_draws`` systems from ``q``; return truth columns and log-q components.

    Deterministic given ``cfg.base_seed`` and ``cfg.generation`` (one ``Generator`` from
    ``SeedSequence(base_seed, spawn_key=(PROPOSAL_RNG_STREAM_BASE + generation, 0))``).
    The per-draw gaiamock seeds are derived separately in :func:`simulate_draws`.
    """
    stream = PROPOSAL_RNG_STREAM_BASE + cfg.generation
    rng = np.random.default_rng(np.random.SeedSequence(entropy=cfg.base_seed, spawn_key=(stream, 0)))
    n = cfg.n_draws
    cols = parent.columns
    qs = parent_proposal_probabilities(cols["parallax"], parent.usable, cfg.parent)
    row = rng.choice(parent.n_rows, size=n, replace=True, p=qs)
    m1 = parent.m1_msun[row]
    log_m1 = np.log10(m1)

    # M2: mixture in log M2
    m2c = cfg.m2
    tied = rng.uniform(size=n) < m2c.q_tied_weight
    log_m2 = np.where(
        tied,
        log_m1 + rng.uniform(math.log10(m2c.q_min), math.log10(m2c.q_max), size=n),
        rng.uniform(math.log10(m2c.m2_min_msun), math.log10(m2c.m2_max_msun), size=n),
    )
    m2 = 10.0**log_m2

    # Flux ratio
    fc = cfg.flux
    dark = rng.uniform(size=n) < fc.dark_fraction
    rel = relation_log10_flux_ratio(m1, m2)
    use_rel = (rng.uniform(size=n) < fc.relation_weight) & np.isfinite(rel)
    log_f = np.where(
        use_rel,
        np.nan_to_num(rel) + fc.relation_sigma_dex * rng.standard_normal(n),
        rng.uniform(fc.log_f_min, fc.log_f_max, size=n),
    )
    log_f = np.where(dark, -np.inf, log_f)
    f = np.where(dark, 0.0, 10.0**log_f)

    # Period
    pc = cfg.period
    core = rng.uniform(size=n) < pc.core_weight
    log_p = np.where(
        core,
        rng.uniform(pc.core_log_p_min, pc.core_log_p_max, size=n),
        rng.uniform(pc.log_p_min, pc.log_p_max, size=n),
    )
    period = 10.0**log_p

    # Eccentricity
    ec = cfg.eccentricity
    ecc = np.where(period <= ec.circular_period_days, 0.0, rng.uniform(0.0, ec.e_cap, size=n))

    # Orientation and phase: isotropic, identical to the target (cancels in weights).
    cos_i = rng.uniform(-1.0, 1.0, size=n)
    inc_deg = np.degrees(np.arccos(cos_i))
    big_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    small_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    tp = rng.uniform(0.0, 1.0, size=n) * period

    log_q = {
        "log_q_parent": np.log(qs[row]),
        "log_q_log_m2": log_q_log_m2(log_m2, log_m1, m2c),
        "log_q_flux": log_q_flux(log_f, dark, m1, m2, fc),
        "log_q_log_p": log_q_log_p(log_p, pc),
        "log_q_ecc": log_q_ecc(ecc, period, ec),
    }
    out: dict[str, NDArray[Any]] = {
        "draw_index": np.arange(draw_index_offset, draw_index_offset + n, dtype=np.int64),
        "generation": np.full(n, cfg.generation, dtype=np.int64),
        "parent_row": row.astype(np.int64),
        "source_id": np.asarray(cols["source_id"], dtype=np.int64)[row],
        "ra_deg": np.asarray(cols["ra"], float)[row],
        "dec_deg": np.asarray(cols["dec"], float)[row],
        "parallax_mas": np.asarray(cols["parallax"], float)[row],
        "pmra_masyr": np.asarray(cols["pmra"], float)[row],
        "pmdec_masyr": np.asarray(cols["pmdec"], float)[row],
        "phot_g_mean_mag": np.asarray(cols["phot_g_mean_mag"], float)[row],
        "m1_msun": m1,
        "m2_msun": m2,
        "is_dark": dark,
        "flux_ratio": f,
        "log10_flux_ratio": log_f,
        "period_days": period,
        "eccentricity": ecc,
        "inc_deg": inc_deg,
        "Omega_rad": big_omega,
        "omega_rad": small_omega,
        "Tp_days": tp,
    }
    out.update(log_q)
    out["log_q_total"] = sum(log_q.values())  # type: ignore[assignment]
    return out


def log_q_total_for(truth: Mapping[str, NDArray[Any]], parent: ParentSnapshot, cfg: ProposalConfig) -> FloatArray:
    """Re-evaluate one proposal generation's log density at arbitrary stored draws.

    Needed for the deterministic-mixture denominator (spec §3.7), where every draw is
    evaluated under every generation's ``q_j``.
    """
    qs = parent_proposal_probabilities(parent.columns["parallax"], parent.usable, cfg.parent)
    row = np.asarray(truth["parent_row"], dtype=np.int64)
    m1 = np.asarray(truth["m1_msun"], float)
    m2 = np.asarray(truth["m2_msun"], float)
    with np.errstate(divide="ignore"):
        lqs = np.log(qs[row])
    return (
        lqs
        + log_q_log_m2(np.log10(m2), np.log10(m1), cfg.m2)
        + log_q_flux(truth["log10_flux_ratio"], truth["is_dark"], m1, m2, cfg.flux)
        + log_q_log_p(np.log10(np.asarray(truth["period_days"], float)), cfg.period)
        + log_q_ecc(truth["eccentricity"], truth["period_days"], cfg.eccentricity)
    )


# ---------------------------------------------------------------------------
# gaiamock per draw
# ---------------------------------------------------------------------------


def published_acceleration(cascade: Sequence[float], cfg: AccelerationPublicationConfig) -> bool:
    """El-Badry et al. (2024) §5.2.1 publication cuts on a 7- or 9-parameter cascade vector.

    gaiamock layout: 9-par → ``res[1] = s``, ``res[13] = F2``; 7-par → ``res[1] = s``,
    ``res[9] = F2``. F2 < ``f2_max_seven_parameter`` is required for 7-parameter only.
    """
    code = float(cascade[0])
    if np.isclose(code, -9.0):
        return bool(float(cascade[1]) > cfg.significance_min)
    if np.isclose(code, -7.0):
        return bool(float(cascade[1]) > cfg.significance_min and float(cascade[9]) < cfg.f2_max_seven_parameter)
    return False


def simulate_one(
    draw: Mapping[str, Any],
    *,
    gaiamock: ModuleType,
    c_funcs: Any,
    cfg: ProposalConfig,
    cuts: OrbitalSolutionCutsConfig,
) -> dict[str, Any]:
    """Run one stored draw through ``run_full_astrometric_cascade``, seeded per #371.

    Seeds: ``mock_global_rng_seeds(base_seed, PROPOSAL_RNG_STREAM_BASE + generation,
    draw_index)``, so a draw replays alone, independent of order or worker.
    """
    stream = PROPOSAL_RNG_STREAM_BASE + int(draw["generation"])
    seeds = mock_global_rng_seeds(cfg.base_seed, stream, int(draw["draw_index"]))
    t0 = time.process_time()
    with seeded_global_rng(seeds, c_funcs):
        cascade = gaiamock.run_full_astrometric_cascade(
            ra=float(draw["ra_deg"]),
            dec=float(draw["dec_deg"]),
            parallax=float(draw["parallax_mas"]),
            pmra=float(draw["pmra_masyr"]),
            pmdec=float(draw["pmdec_masyr"]),
            m1=float(draw["m1_msun"]),
            m2=float(draw["m2_msun"]),
            period=float(draw["period_days"]),
            Tp=float(draw["Tp_days"]),
            ecc=float(draw["eccentricity"]),
            omega=float(draw["Omega_rad"]),
            inc_deg=float(draw["inc_deg"]),
            w=float(draw["omega_rad"]),
            phot_g_mean_mag=float(draw["phot_g_mean_mag"]),
            f=float(draw["flux_ratio"]),
            data_release=cfg.data_release,
            c_funcs=c_funcs,
            verbose=False,
            show_residuals=False,
            ruwe_min=cfg.ruwe_min,
            skip_acceleration=cfg.skip_acceleration,
        )
    cpu = time.process_time() - t0
    vec = [float(v) for v in cascade] + [0.0] * (CASCADE_VECTOR_LENGTH - len(cascade))
    rec = classify_cascade_result(
        vec,
        m1_msun=float(draw["m1_msun"]),
        m2_msun=float(draw["m2_msun"]),
        flux_ratio=float(draw["flux_ratio"]),
        cuts=cuts,
        gaiamock=None,
    )
    return {
        "draw_index": int(draw["draw_index"]),
        "solution_type": rec.solution_type.value,
        "accepted_orbital": bool(rec.accepted_orbital),
        "published_acceleration": published_acceleration(vec, cfg.acceleration_publication),
        "cascade": vec[:CASCADE_VECTOR_LENGTH],
        "f_m_msun": rec.f_m_msun if rec.f_m_msun is not None else float("nan"),
        "cos_inclination_fit": rec.cos_inclination if rec.cos_inclination is not None else float("nan"),
        "numpy_seed": seeds.numpy_seed,
        "c_rand_seed": seeds.c_rand_seed,
        "cpu_seconds": cpu,
    }


# ---------------------------------------------------------------------------
# Target intensity (rung 2: MdS17 luminous companions) and weights
# ---------------------------------------------------------------------------


def mds17_luminous_log_intensity(
    truth: Mapping[str, NDArray[Any]], target: MdS17TargetConfig
) -> FloatArray:
    """log λ(x | θ_MdS17) in the proposal's measure (per dex M2, per dex P, per unit e or
    the circular point mass, per dex f), for luminous MS companions only.

    λ = f_logP;q>0.3(M1, P) × p_q(q) × q ln10 × p_e × N(log10 f; log10 f_rel, σ_f) ×
    s_low(M1). Dark draws (f = 0), q outside MdS17's 0.1-1 and log P outside 0.2-8 get
    −inf. M1 above the MdS17 domain is clamped to its edge for the shape (recorded as a
    limitation; TAG10 rarely exceeds it). Provisional choices come from ``target``.
    """
    table = mds.load_mds17_table(target.table_path)
    m1 = np.asarray(truth["m1_msun"], float)
    m2 = np.asarray(truth["m2_msun"], float)
    p = np.asarray(truth["period_days"], float)
    e = np.asarray(truth["eccentricity"], float)
    lf = np.asarray(truth["log10_flux_ratio"], float)
    dark = np.asarray(truth["is_dark"], bool)
    lp = np.log10(p)
    m1_shape = np.clip(m1, table.m1_range[0], table.m1_range[1])
    q = m2 / m1
    interp = target.provisional_m1_interpolation
    freq = mds.f_logp_q03(m1_shape, lp, table)
    freq = freq * mds.low_mass_frequency_scale(
        m1,
        m1_anchor_msun=target.provisional_low_mass_anchor_msun,
        m1_zero_msun=target.provisional_low_mass_zero_msun,
    )
    pq = mds.q_density(q, m1_shape, lp, table, m1_interpolation=interp)
    per_dex_m2 = pq * q * math.log(10.0)
    circ = mds.is_circular(p, table)
    pe = np.where(
        circ,
        (e == 0.0).astype(float),
        mds.e_density(e, m1_shape, p, table, m1_interpolation=interp, eta_floor=target.provisional_eta_floor),
    )
    rel = relation_log10_flux_ratio(m1, m2)
    sig = target.provisional_flux_sigma_dex
    with np.errstate(invalid="ignore"):
        pf = np.exp(-0.5 * ((lf - rel) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    pf = np.where(np.isfinite(pf) & ~dark, pf, 0.0)
    lam = freq * per_dex_m2 * pe * pf
    with np.errstate(divide="ignore"):
        return np.log(np.where(np.isfinite(lam) & (lam > 0), lam, 0.0))


def importance_weights(
    log_target: ArrayLike,
    log_q_components: Sequence[ArrayLike],
    n_per_generation: Sequence[int],
    *,
    scale_to_full: float,
) -> FloatArray:
    """Deterministic-mixture weights (spec §3.4, §3.7).

    ``w_i = scale_to_full × exp(log λ_i) / Σ_j n_j exp(log q_j(x_i))``. With one
    generation this is ``scale × λ / (N q)``. ``log_q_components[j]`` is generation
    ``j``'s log density evaluated at **every** draw.
    """
    lt = np.asarray(log_target, float)
    if len(log_q_components) != len(n_per_generation):
        raise ValueError("one log-q array per generation is required")
    stack = np.stack([np.asarray(c, float) + math.log(n) for c, n in zip(log_q_components, n_per_generation)])
    log_den = np.logaddexp.reduce(stack, axis=0)
    with np.errstate(invalid="ignore"):
        w = np.exp(lt - log_den) * scale_to_full
    return np.where(np.isfinite(w), w, 0.0)


def kish_ess(weights: ArrayLike) -> float:
    """Kish effective sample size ``(Σw)² / Σw²``; 0 for an empty or all-zero set."""
    w = np.asarray(weights, float)
    s2 = float(np.sum(w * w))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


@dataclass(frozen=True)
class BinnedWeights:
    """Per-bin weighted counts and ESS (spec §3.6)."""

    edges: FloatArray
    n_draws: NDArray[np.int64]
    sum_w: FloatArray
    sum_w2: FloatArray
    ess: FloatArray
    mc_to_poisson: FloatArray  # sqrt(Σw² / Σw) = σ_MC / σ_Poisson

    def trusted(self, threshold: float) -> NDArray[np.bool_]:
        """Bins meeting σ_MC/σ_Poisson < threshold (⇔ ESS ≥ N_b / threshold²)."""
        return (self.sum_w > 0) & (self.mc_to_poisson < threshold)


def binned_weights(values: ArrayLike, weights: ArrayLike, edges: ArrayLike) -> BinnedWeights:
    """Histogram ``weights`` by ``values`` with per-bin ESS; non-finite values dropped."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    e = np.asarray(edges, float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    n, _ = np.histogram(v, bins=e)
    s1, _ = np.histogram(v, bins=e, weights=w)
    s2, _ = np.histogram(v, bins=e, weights=w * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        ess = np.where(s2 > 0, s1 * s1 / s2, 0.0)
        ratio = np.where(s1 > 0, np.sqrt(s2 / s1), np.inf)
    return BinnedWeights(edges=e, n_draws=n.astype(np.int64), sum_w=s1, sum_w2=s2, ess=ess, mc_to_poisson=ratio)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def write_proposal_artifact(
    path: str | Path,
    truth: Mapping[str, NDArray[Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    fragment: ProposalSetFragment,
    parent: ParentSnapshot,
    provenance: Mapping[str, Any],
) -> Path:
    """Write one generation's draws (truth + outcome) to HDF5 (spec §3.3).

    ``outcomes`` must be in ``truth['draw_index']`` order. Attributes record the
    proposal config, the provisional settings, the parent snapshot and its checksum,
    the seeding scheme and any extra ``provenance`` (gaiamock versions, code commit).
    """
    import h5py

    order = {int(o["draw_index"]): o for o in outcomes}
    idx = np.asarray(truth["draw_index"], dtype=np.int64)
    if set(order) != set(idx.tolist()):
        raise ValueError("outcomes do not cover exactly the truth draws")
    rows = [order[int(i)] for i in idx]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as h:
        g = h.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=np.asarray(v))
        o = h.create_group("outcome")
        o.create_dataset("solution_type", data=np.array([r["solution_type"] for r in rows], dtype="S32"))
        for key in ("accepted_orbital", "published_acceleration"):
            o.create_dataset(key, data=np.array([r[key] for r in rows], dtype=bool))
        o.create_dataset("cascade", data=np.array([r["cascade"] for r in rows], dtype=np.float64))
        for key in ("f_m_msun", "cos_inclination_fit", "cpu_seconds"):
            o.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.float64))
        for key in ("numpy_seed", "c_rand_seed"):
            o.create_dataset(key, data=np.array([r[key] for r in rows], dtype=np.uint32))
        h.attrs["proposal_config_json"] = json.dumps(fragment.proposal.model_dump(mode="json"), sort_keys=True)
        h.attrs["fragment_fingerprint"] = fragment_fingerprint(fragment)
        h.attrs["parent_snapshot_dir"] = str(parent.path)
        h.attrs["parent_h5_sha256"] = str(parent.meta["parent_h5_sha256"])
        h.attrs["scale_to_full"] = float(parent.scale_to_full)
        h.attrs["seed_scheme"] = (
            "forward_model.mock_global_rng_seeds(base_seed, "
            f"{PROPOSAL_RNG_STREAM_BASE} + generation, draw_index) -> (np.random.seed, libc srand)"
        )
        h.attrs["cascade_layout"] = (
            "plx, sig_parallax, A, sig_A, B, sig_B, F, sig_F, G, sig_G, period, sig_period, "
            "phi_p, sig_phi_p, ecc, sig_ecc, inc_deg, a0_mas, sigma_a0_mas, "
            "N_visibility_periods, N_obs, F2, ruwe (gaiamock_mod fit_full_astrometric_cascade)"
        )
        h.attrs["provenance_json"] = json.dumps(dict(provenance), sort_keys=True, default=str)
    return p


def read_proposal_artifact(path: str | Path) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]], dict[str, Any]]:
    """Read ``(truth, outcome, attrs)`` written by :func:`write_proposal_artifact`."""
    import h5py

    with h5py.File(Path(path), "r") as h:
        truth = {k: h["truth"][k][()] for k in h["truth"].keys()}
        outcome = {k: h["outcome"][k][()] for k in h["outcome"].keys()}
        attrs = {k: h.attrs[k] for k in h.attrs.keys()}
    outcome["solution_type"] = np.char.decode(outcome["solution_type"].astype("S32"), "ascii")
    return truth, outcome, attrs
