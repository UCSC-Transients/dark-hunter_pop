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

Generation-time choices are decided (spec §0.1) and named without a prefix; every
reweightable choice the spec still leaves open (§8, MP-Q*) is a config field whose name
starts with ``provisional_``. All are recorded in the artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import constants
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

#: Bailer-Jones et al. (2021) geometric distances (``external.gaiaedr3_distance``, pc).
BAILER_JONES_COLUMNS: tuple[str, ...] = ("r_med_geo", "r_lo_geo", "r_hi_geo")

_AP_COLUMNS = frozenset(c for c in GAIA_SOURCE_PARENT_COLUMNS if "msc" in c or "gspphot" in c)


def build_gaia_source_parent_adql(
    *, k: int, parallax_floor_mas: float, g_max: float, include_bailer_jones: bool = False
) -> str:
    """ADQL for the uniform ``random_index < k`` parent subsample (spec §1.3).

    Only G and parallax are selected on. RUWE, visibility periods and the Halbwachs et al.
    (2023) IPD / C* columns are fetched but never cut on in the query (RUWE and visibility
    are outcomes; the IPD / C* cuts are applied as recorded flags, spec §0.1 MP-Q3).
    ``include_bailer_jones`` adds the geometric distances (spec §0.1 MP-Q4) by a LEFT JOIN,
    so rows without a distance survive the query and are counted when dropped.
    """
    if k <= 0 or k > GAIA_SOURCE_TOTAL_ROWS:
        raise ValueError(f"k must be in (0, {GAIA_SOURCE_TOTAL_ROWS}], got {k}")
    names = [(f"ap.{c}" if c in _AP_COLUMNS else f"gs.{c}") for c in GAIA_SOURCE_PARENT_COLUMNS]
    if include_bailer_jones:
        names += [f"bj.{c}" for c in BAILER_JONES_COLUMNS]
    cols = ",\n  ".join(names)
    bj_join = (
        "LEFT JOIN external.gaiaedr3_distance AS bj ON gs.source_id = bj.source_id\n"
        if include_bailer_jones
        else ""
    )
    return (
        f"SELECT\n  {cols}\n"
        "FROM gaiadr3.gaia_source AS gs\n"
        "LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id\n"
        + bj_join
        + 
        f"WHERE gs.random_index < {int(k)}\n"
        f"  AND gs.phot_g_mean_mag < {float(g_max)!r}\n"
        f"  AND gs.parallax > {float(parallax_floor_mas)!r}"
    )


# ---------------------------------------------------------------------------
# Config (fragment config/population/proposal_set_pilot.yaml; merged at stage integration)
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
    #: MP-Q28d (decided 2026-10-04): centre the relation component of CMD-evolved rows on the
    #: §10.4 evolved relation ``giants.evolved_log10_flux_ratio(M2, M_G0,sys)``. Generation-time
    #: (coverage); the default keeps older artifacts' densities unchanged.
    evolved_rows_centre: Literal["dwarf_relation", "evolved_relation"] = "dwarf_relation"


def proposal_relation_centre(
    m1_msun: ArrayLike,
    m2_msun: ArrayLike,
    rows: NDArray[np.int64],
    parent: Any,
    cfg: FluxProposalConfig,
) -> FloatArray:
    """log10 f centre of the flux proposal's relation component per draw (MP-Q28d).

    The dwarf relation, except on CMD-evolved rows of an isochrone-mode parent when
    ``cfg.evolved_rows_centre == "evolved_relation"``; NaN where neither is defined (the
    relation component then moves to the uniform one, in the sampler and the density alike).
    """
    rel = relation_log10_flux_ratio(m1_msun, m2_msun)
    if cfg.evolved_rows_centre == "evolved_relation" and getattr(parent, "cmd", None) is not None:
        from darkhunter_pop.giants import evolved_log10_flux_ratio

        evo = np.asarray(parent.is_giant, bool)[rows]
        mg0 = np.asarray(parent.cmd["mg0"], float)[rows]
        rel_e = evolved_log10_flux_ratio(m2_msun, np.where(evo, mg0, 0.0))
        rel = np.where(evo, rel_e, rel)
    return np.asarray(rel, float)


class PeriodProposalConfig(_Strict):
    """Mixture in log P: a core log-uniform + a defensive log-uniform."""

    core_weight: float = Field(..., ge=0.0, le=1.0)
    core_log_p_min: float
    core_log_p_max: float
    log_p_min: float
    log_p_max: float


class EccentricityProposalConfig(_Strict):
    """Eccentricity proposal ``q(e | P)``; e = 0 for P <= ``circular_period_days``.

    ``shape: uniform`` (the pilot and the paused run): U(0, ``e_cap``). It truncates the
    MdS17 target above ``e_cap`` (#409), and with eta < -0.5 its weights have infinite
    variance (#410). Kept only so those artifacts stay readable.

    ``shape: mds17_bounded`` (#409, #410): a three-part mixture for P > circular,
    with E = e_max(P) = 1 − (P / ``e_max_period_scale_days``)^``e_max_exponent``
    (MdS17 Eq. 3, the target's own support):

    * ``floor_weight`` × power law (η_q + 1) e^η_q / E^(η_q+1) on [0, E), η_q = ``floor_eta``;
    * ``support_weight`` × U(0, E);
    * the rest × U(0, ``defensive_e_max``), a defensive part that also covers e > e_max
      for any alternative eccentricity model (MP-Q12).

    The weight p/q is **bounded** whenever every target eta ≥ η_q: near e = 0 the power
    law dominates and (e/E)^(η−η_q) ≤ 1; elsewhere U(0, E) bounds it by (η+1)/``support_weight``.
    :func:`check_eccentricity_bounded` enforces η_q ≤ the target's eta floor. These are
    sampling-efficiency settings only.
    """

    shape: Literal["uniform", "mds17_bounded"] = "uniform"
    circular_period_days: float = Field(..., gt=0.0)
    e_cap: float | None = Field(None, gt=0.0, lt=1.0)
    floor_eta: float | None = Field(None, gt=-1.0)
    floor_weight: float | None = Field(None, ge=0.0, le=1.0)
    support_weight: float | None = Field(None, ge=0.0, le=1.0)
    defensive_e_max: float | None = Field(None, gt=0.0, lt=1.0)
    e_max_period_scale_days: float | None = Field(None, gt=0.0)
    e_max_exponent: float | None = Field(None, lt=0.0)

    @model_validator(mode="after")
    def _shape_fields(self) -> EccentricityProposalConfig:
        if self.shape == "uniform":
            if self.e_cap is None:
                raise ValueError("shape uniform needs e_cap")
            return self
        need = ("floor_eta", "floor_weight", "support_weight", "defensive_e_max",
                "e_max_period_scale_days", "e_max_exponent")
        missing = [k for k in need if getattr(self, k) is None]
        if missing:
            raise ValueError(f"shape mds17_bounded needs {missing}")
        if self.floor_weight + self.support_weight > 1.0 + 1e-12:  # type: ignore[operator]
            raise ValueError("floor_weight + support_weight must be <= 1")
        return self

    def e_max(self, period_days: ArrayLike) -> FloatArray:
        """MdS17 Eq. 3 support edge (0 for P <= circular_period_days)."""
        p = np.asarray(period_days, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            val = 1.0 - (p / float(self.e_max_period_scale_days)) ** float(self.e_max_exponent)  # type: ignore[arg-type]
        return np.where(p > self.circular_period_days, np.clip(val, 0.0, 1.0), 0.0)


class AccelerationPublicationConfig(_Strict):
    """Published acceleration solutions (El-Badry et al. 2024 §5.2.1)."""

    significance_min: float
    f2_max_seven_parameter: float


class HalbwachsCutsConfig(_Strict):
    """Halbwachs et al. (2023) §1.2 NSS input steps (b) and (c) (spec §0.1, MP-Q3)."""

    ipd_frac_multi_peak_max: float  # step (b): <=
    ipd_gof_harmonic_amplitude_max: float  # step (b): <
    cstar_nsigma: float  # step (c): |C*| < nsigma * sigma_C*


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
    # Decided 2026-10-02 (spec §0.1; #391 comment 5963152741), or the pilot's provisional
    # placeholders for reading pilot artifacts. ``decision_ref`` names the source.
    decision_ref: str
    parallax_floor_mas: float = Field(..., gt=0.0)  # MP-Q1
    halbwachs_ipd_cstar_cuts: Literal["applied_star_values", "not_applied"]  # MP-Q3
    halbwachs_cuts: HalbwachsCutsConfig
    truth_distance: Literal["bailer_jones2021_geometric", "measured_parallax"]  # MP-Q4
    # MP-Q5; #418 (Ryan 2026-10-03, spec §0.3/§11): ``isochrone_mist_drop_unresolved`` assigns
    # the MIST isochrone posterior point M1 (MP-Q35) from the dereddened CMD and replaces the
    # TAG10-log g giant flag with the §10.2 CMD evolved flag (shared ridge, MP-Q28a n_sigma).
    # ``isochrone_posterior_draw_deblended`` (MP-Q35 + MP-Q36, decided 2026-10-04, spec §11.9):
    # the parent as ``isochrone_mist_drop_unresolved``; then, per draw, one posterior draw of
    # (age, [Fe/H], M̂1) and the coeval deblended truth M1 (:func:`apply_posterior_deblending`).
    m1: Literal[
        "tag10_point_drop_unresolved", "isochrone_mist_drop_unresolved", "isochrone_posterior_draw_deblended"
    ]  # MP-Q5
    giant_flag_logg_max: float  # MP-Q5 flag only (Andrews 2022 ATF dwarf/giant log g); TAG10 mode only
    light_split: Literal["observed_g_is_system_total"]  # MP-Q6
    # #400 E1 (decided 2026-10-03): ``dr3_config`` wraps every cascade call in the epoch
    # model configured at ``dr3.epoch_model`` (the runner builds the wrapper); ``off`` keeps
    # gaiamock's GOST list. Generation-time: it changes the epochs (spec §3.5).
    epoch_model: Literal["off", "dr3_config"] = "off"


class MdS17TargetConfig(_Strict):
    """Rung-2 target: MdS17 at published parameters, luminous companions only."""

    table_path: str
    provisional_m1_interpolation: mds.M1Interpolation  # MP-Q10
    provisional_eta_floor: float = Field(..., gt=-1.0)  # MP-Q11
    provisional_low_mass_policy: Literal["log_linear_to_zero"]  # MP-Q7
    provisional_low_mass_anchor_msun: float = Field(..., gt=0.0)
    provisional_low_mass_zero_msun: float = Field(..., gt=0.0)
    provisional_multiplicity: Literal["poisson_intensity"]  # MP-Q9
    # MP-Q13 (decided 2026-10-02): janssens2022. MP-Q40 (decided 2026-10-04, spec §0.4, §11.9):
    # ``mist_coeval`` takes the main-sequence companion's G-flux ratio from the same coeval
    # MIST isochrone as the primary, f = 10^{-0.4 (M_G(M2) - M_G(M1))}, with the same
    # ``flux_sigma_dex`` scatter; callers pass it per draw (:func:`mist_relation_for_draws`).
    mass_luminosity: Literal["janssens2022", "mist_coeval"]
    flux_sigma_dex: float = Field(..., gt=0.0)  # MP-Q13, decided (spec §0.1)
    provisional_compact_mixture: Literal["none"]  # MP-Q17

    @model_validator(mode="after")
    def _low_mass_order(self) -> MdS17TargetConfig:
        if not self.provisional_low_mass_zero_msun < self.provisional_low_mass_anchor_msun:
            raise ValueError("provisional_low_mass_zero_msun must be < anchor")
        return self


class ProposalSetFragment(_Strict):
    """Whole ``config/population/proposal_set_pilot.yaml``."""

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
    """A ``gaia_source`` parent subsample with data-side TAG10 M1 and recorded filters.

    ``usable`` marks rows the proposal may draw: the AND of every entry of ``flags`` that
    the proposal's decided settings require (spec §0.1). ``flags`` always holds every
    per-row filter so attrition can be reported. ``truth_parallax_mas`` is the parallax
    handed to gaiamock (Bailer-Jones geometric, or measured, per ``truth_distance``).
    ``scale_to_full`` is ``N_full / N_snap = GAIA_SOURCE_TOTAL_ROWS / K``.
    """

    columns: Mapping[str, NDArray[Any]]
    m1_msun: FloatArray
    m1_source: NDArray[np.str_]
    atmosphere_logg: FloatArray
    truth_parallax_mas: FloatArray
    is_giant: NDArray[np.bool_]
    flags: Mapping[str, NDArray[np.bool_]]
    usable: NDArray[np.bool_]
    meta: Mapping[str, Any]
    scale_to_full: float
    path: Path
    #: #418 isochrone mode only: per-row dereddened CMD (``mg0``, ``colour0``, ``sigma_mu``,
    #: ``ebv``, ``a_g``, ``e_bp_rp``) and isochrone posterior summaries
    #: (:meth:`darkhunter_pop.isochrone_mass.IsochronePosterior.as_dict`). None in TAG10 mode.
    cmd: Mapping[str, NDArray[Any]] | None = None
    isochrone: Mapping[str, NDArray[Any]] | None = None

    @property
    def n_rows(self) -> int:
        return int(self.m1_msun.size)

    def attrition(self) -> dict[str, int]:
        """Cumulative rows surviving each required flag, in ``flags`` order."""
        keep = np.ones(self.n_rows, dtype=bool)
        out = {"snapshot_rows": self.n_rows}
        for name, flag in self.flags.items():
            keep &= flag
            out[name] = int(keep.sum())
        return out


def tag10_m1_for_rows(
    columns: Mapping[str, NDArray[Any]], config: PipelineConfig
) -> tuple[FloatArray, NDArray[np.str_], FloatArray]:
    """M1 per row via ``mass_derivation`` (MSC → GSP-Phot → TAG10), exactly as the data side.

    Returns ``(m1, source, logg)``: ``logg`` is the log g of the atmosphere TAG10 used
    (for the giant flag). Rows with no resolvable atmosphere get NaN and source ``"none"``.
    ``mass_derivation`` is imported and called, never modified.
    """
    from darkhunter_pop.mass_derivation import (
        derive_tag10_m1_r1,
        resolve_atmosphere_from_extras,
    )

    n = int(np.asarray(columns["source_id"]).size)
    names = [c for c in GAIA_SOURCE_PARENT_COLUMNS if c in _AP_COLUMNS]
    m1 = np.full(n, np.nan)
    logg = np.full(n, np.nan)
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
            logg[i] = float(atm.logg)
    return m1, src, logg


def corrected_flux_excess(bp_rp: ArrayLike, excess_factor: ArrayLike) -> FloatArray:
    """Riello et al. (2021) Eq. 6 corrected BP/RP flux excess C*; NaN without a colour."""
    x = np.asarray(bp_rp, dtype=np.float64)
    c = np.asarray(excess_factor, dtype=np.float64)
    lo, hi = constants.RIELLO2021_CSTAR_X_BREAKS

    def poly(co: tuple[float, float, float, float]) -> FloatArray:
        return co[0] + co[1] * x + co[2] * x**2 + co[3] * x**3

    corr = np.where(
        x < lo,
        poly(constants.RIELLO2021_CSTAR_BLUE),
        np.where(x < hi, poly(constants.RIELLO2021_CSTAR_GREEN), poly(constants.RIELLO2021_CSTAR_RED)),
    )
    return c - corr


def sigma_cstar(g_mag: ArrayLike) -> FloatArray:
    """Riello et al. (2021) Eq. 18: 1σ scatter of C* at magnitude G."""
    s0, s1, s2 = constants.RIELLO2021_SIGMA_CSTAR
    return s0 + s1 * np.asarray(g_mag, dtype=np.float64) ** s2


def halbwachs_input_flags(
    columns: Mapping[str, NDArray[Any]], cuts: HalbwachsCutsConfig
) -> dict[str, NDArray[np.bool_]]:
    """Halbwachs et al. (2023) §1.2 steps (b) and (c) per row, from the star's own values.

    Missing values fail (the condition cannot hold), e.g. no BP/RP means no C*.
    """
    def col(name: str) -> FloatArray:
        return np.asarray(columns[name], dtype=np.float64)

    with np.errstate(invalid="ignore"):
        ipd = (col("ipd_frac_multi_peak") <= cuts.ipd_frac_multi_peak_max) & (
            col("ipd_gof_harmonic_amplitude") < cuts.ipd_gof_harmonic_amplitude_max
        )
        cs = corrected_flux_excess(col("bp_rp"), col("phot_bp_rp_excess_factor"))
        cst = np.abs(cs) < cuts.cstar_nsigma * sigma_cstar(col("phot_g_mean_mag"))
    return {"halbwachs_ipd": ipd & np.isfinite(col("ipd_frac_multi_peak")), "halbwachs_cstar": cst & np.isfinite(cs)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_cmd_isochrone(
    cols: Mapping[str, NDArray[Any]],
    config: PipelineConfig,
    giants_cfg: Any,
    cache_dir: Path | None,
) -> tuple[dict[str, FloatArray], dict[str, NDArray[Any]]]:
    """Dereddened CMD and MIST isochrone posterior per parent row (#418, spec §11.2).

    CMD: :func:`darkhunter_pop.giants.cmd_for_rows` (Combined19 at the Bailer-Jones distance,
    Babusiaux et al. 2018 law). Posterior: :mod:`darkhunter_pop.isochrone_mass` with
    ``config.isochrone_mass``. Cached as ``cmd_isochrone_<key>.npz`` in ``cache_dir`` (key:
    isochrone config, giants extinction, dust-map section) when ``cache_dir`` is given.
    """
    from darkhunter_pop import giants
    from darkhunter_pop import isochrone_mass as im

    blob = json.dumps(
        {
            "iso": im.config_key(config.isochrone_mass),
            "ext": giants_cfg.extinction.model_dump(mode="json"),
            "dust": config.sample_selection.dust_maps.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    key = hashlib.sha256(blob.encode()).hexdigest()[:10]
    path = None if cache_dir is None else Path(cache_dir) / f"cmd_isochrone_{key}.npz"
    if path is not None and path.exists():
        z = np.load(path)
        cmd = {k[4:]: z[k] for k in z.files if k.startswith("cmd_")}
        iso = {k[4:]: z[k] for k in z.files if k.startswith("iso_")}
        return cmd, iso
    rc = giants.cmd_for_rows(
        cols["phot_g_mean_mag"], cols["bp_rp"], cols["l"], cols["b"],
        cols["r_med_geo"], cols["r_lo_geo"], cols["r_hi_geo"], config, giants_cfg,
    )
    cmd = {
        "mg0": rc.mg0, "colour0": rc.colour0, "sigma_mu": rc.sigma_mu, "ebv": rc.ebv,
        "a_g": rc.a_g, "e_bp_rp": np.asarray(cols["bp_rp"], float) - rc.colour0,
    }
    model = im.build_model(config.isochrone_mass, config.paths.data_root)
    post = model.fit(rc.colour0, rc.mg0, sigma_mu=rc.sigma_mu, ebv=rc.ebv, a_g=rc.a_g, e_bp_rp=cmd["e_bp_rp"])
    iso = post.as_dict()
    if path is not None:
        np.savez(path, **{f"cmd_{k}": v for k, v in cmd.items()}, **{f"iso_{k}": v for k, v in iso.items()})
    return cmd, iso


def load_parent_snapshot(
    snapshot_dir: str | Path,
    config: PipelineConfig,
    proposal: ProposalConfig,
    *,
    m1_cache: bool = True,
    giants_config_path: str | Path = "config/population/giants.yaml",
) -> ParentSnapshot:
    """Load ``parent.h5`` + ``meta.yaml``, verify the checksum, attach M1 and filters.

    Required filters, in order (each a ``flags`` entry; spec §0.1): finite astrometry and
    photometry; measured parallax > ``parallax_floor_mas`` (MP-Q1); an M1 from TAG10
    (MP-Q5); a Bailer-Jones geometric distance when ``truth_distance`` asks for it (MP-Q4);
    the Halbwachs (b) and (c) cuts when applied (MP-Q3). ``is_giant`` is only a flag.

    ``m1_cache`` stores per-row M1 beside the snapshot (``m1_tag10_<cfg>.npz``, keyed by
    the ``mass_calibration`` section) so repeated loads skip the per-row TAG10 loop.

    ``proposal.m1 == "isochrone_mist_drop_unresolved"`` (#418): M1 is the isochrone posterior
    point (``config.isochrone_mass.provisional_point_estimate``, MP-Q35) from
    :func:`parent_cmd_isochrone` (cached beside the snapshot), the required flag
    ``tag10_atmosphere`` is replaced by ``isochrone_m1``, and ``is_giant`` is the §10.2 CMD
    evolved flag from ``giants_config_path`` (ridge measured on the usable rows, RUWE cut
    when configured). ``atmosphere_logg`` then holds the posterior ⟨log g⟩.
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
    iso_mode = proposal.m1 in ("isochrone_mist_drop_unresolved", "isochrone_posterior_draw_deblended")
    cmd: dict[str, FloatArray] | None = None
    iso: dict[str, NDArray[Any]] | None = None
    if iso_mode:
        from darkhunter_pop import giants

        gcfg = giants.load_giants_config(giants_config_path)
        cmd, iso = parent_cmd_isochrone(cols, config, gcfg, d if m1_cache else None)
        pt = config.isochrone_mass.provisional_point_estimate
        m1 = np.asarray(iso["m1_mean"] if pt == "mean" else 10.0 ** np.asarray(iso["log_m1_mean"]), float)
        src = np.where(np.asarray(iso["ok"], bool), "MIST", "none")
        logg = np.asarray(iso["log_g_mean"], float)
    else:
        mc_dump = config.mass_calibration.model_dump(mode="json")
        mc_dump["method"] = "TAG10"  # TAG10 by name (#425): the cache does not move with the bulk switch
        mc_blob = json.dumps(mc_dump, sort_keys=True)
        mc_key = hashlib.sha256(mc_blob.encode()).hexdigest()[:10]
        cache = d / f"m1_tag10_{mc_key}.npz"
        if m1_cache and cache.exists() and "atmosphere_logg" in np.load(cache).files:
            z = np.load(cache)
            m1, src, logg = z["m1_msun"], z["m1_source"], z["atmosphere_logg"]
        else:
            m1, src, logg = tag10_m1_for_rows(cols, config)
            if m1_cache:
                np.savez(cache, m1_msun=m1, m1_source=src, atmosphere_logg=logg)
    finite = np.ones(m1.size, dtype=bool)
    for name in ("ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag"):
        finite &= np.isfinite(np.asarray(cols[name], dtype=np.float64))
    plx = np.asarray(cols["parallax"], dtype=np.float64)
    flags: dict[str, NDArray[np.bool_]] = {
        "finite_astrometry_photometry": finite,
        "parallax_floor": plx > proposal.parallax_floor_mas,
        ("isochrone_m1" if iso_mode else "tag10_atmosphere"): np.isfinite(m1),
    }
    if proposal.truth_distance == "bailer_jones2021_geometric":
        if "r_med_geo" not in cols:
            raise ValueError(f"{d} has no Bailer-Jones distances; refetch with --bailer-jones")
        r = np.asarray(cols["r_med_geo"], dtype=np.float64)
        flags["bailer_jones_distance"] = np.isfinite(r) & (r > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            truth_plx = np.where(flags["bailer_jones_distance"], 1000.0 / r, np.nan)
    else:
        truth_plx = plx
    if proposal.halbwachs_ipd_cstar_cuts == "applied_star_values":
        flags.update(halbwachs_input_flags(cols, proposal.halbwachs_cuts))
    usable = np.logical_and.reduce(list(flags.values()))
    if iso_mode:
        from darkhunter_pop import giants

        gcfg = giants.load_giants_config(giants_config_path)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.asarray(cols["parallax"], float) / np.asarray(cols["parallax_error"], float)
        ruwe = np.asarray(cols["ruwe"], float) if gcfg.ridge.ruwe_max is not None else None
        ridge = giants.fit_ms_ridge(np.where(usable, cmd["mg0"], np.nan), cmd["colour0"], snr, gcfg.ridge, ruwe=ruwe)  # type: ignore[index]
        giant = giants.classify_evolved(cmd["mg0"], cmd["colour0"], cmd["sigma_mu"], ridge, gcfg.provisional_n_sigma).evolved  # type: ignore[index]
        meta = {**meta, "ms_ridge": {"colour": ridge.colour.tolist(), "mag": ridge.mag.tolist(),
                                     "sigma": ridge.sigma.tolist(), "n_rows": ridge.n_rows.tolist()}}
    else:
        with np.errstate(invalid="ignore"):
            giant = np.isfinite(logg) & (logg < proposal.giant_flag_logg_max)
    k = int(meta["random_index_max_exclusive"])
    return ParentSnapshot(
        columns=cols,
        m1_msun=np.asarray(m1, dtype=np.float64),
        m1_source=np.asarray(src),
        atmosphere_logg=np.asarray(logg, dtype=np.float64),
        truth_parallax_mas=np.asarray(truth_plx, dtype=np.float64),
        is_giant=giant,
        flags=flags,
        usable=usable,
        meta=meta,
        scale_to_full=float(meta["gaia_source_total_rows"]) / float(k),
        path=d,
        cmd=cmd,
        isochrone=iso,
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
    log10_f: ArrayLike,
    is_dark: ArrayLike,
    m1_msun: ArrayLike,
    m2_msun: ArrayLike,
    cfg: FluxProposalConfig,
    *,
    centre: ArrayLike | None = None,
) -> FloatArray:
    """log proposal for the flux ratio: log(ρ_dark) for dark draws, else log of the
    luminous mixture density per dex of f times (1 - ρ_dark). ``centre`` overrides the
    relation centre (:func:`proposal_relation_centre`, MP-Q28d)."""
    lf = np.asarray(log10_f, float)
    dark = np.asarray(is_dark, bool)
    rel = relation_log10_flux_ratio(m1_msun, m2_msun) if centre is None else np.asarray(centre, float)
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
    """log q(e | P): 0 (probability 1) at e = 0 for circular-class draws; else the
    log density of ``cfg.shape`` (see :class:`EccentricityProposalConfig`)."""
    e = np.asarray(ecc, float)
    p = np.asarray(period_days, float)
    circ = p <= cfg.circular_period_days
    if cfg.shape == "uniform":
        inside = (e >= 0) & (e < cfg.e_cap)  # type: ignore[operator]
        with np.errstate(divide="ignore"):
            cont = np.where(inside, -math.log(cfg.e_cap), -np.inf)  # type: ignore[arg-type]
    else:
        big_e = cfg.e_max(p)
        eq = float(cfg.floor_eta)  # type: ignore[arg-type]
        wa, wb = float(cfg.floor_weight), float(cfg.support_weight)  # type: ignore[arg-type]
        wc = max(0.0, 1.0 - wa - wb)
        d = float(cfg.defensive_e_max)  # type: ignore[arg-type]
        in_e = (e >= 0) & (e < big_e) & (big_e > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pow_d = np.where(in_e, (eq + 1.0) * e**eq / big_e ** (eq + 1.0), 0.0)
            uni_e = np.where(in_e, 1.0 / big_e, 0.0)
        uni_d = ((e >= 0) & (e < d)) / d
        dens = wa * np.nan_to_num(pow_d, posinf=np.inf) + wb * uni_e + wc * uni_d
        with np.errstate(divide="ignore"):
            cont = np.log(dens)
    return np.where(circ, np.where(e == 0.0, 0.0, -np.inf), cont)


def sample_ecc(
    period_days: NDArray[np.float64], cfg: EccentricityProposalConfig, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draw e | P from ``cfg`` (vectorized; one uniform for the component, one for e)."""
    p = np.asarray(period_days, dtype=np.float64)
    n = p.size
    circ = p <= cfg.circular_period_days
    if cfg.shape == "uniform":
        return np.where(circ, 0.0, rng.uniform(0.0, cfg.e_cap, size=n))  # type: ignore[arg-type]
    big_e = cfg.e_max(p)
    eq = float(cfg.floor_eta)  # type: ignore[arg-type]
    wa, wb = float(cfg.floor_weight), float(cfg.support_weight)  # type: ignore[arg-type]
    comp = rng.uniform(size=n)
    u = rng.uniform(size=n)
    e_pow = big_e * u ** (1.0 / (eq + 1.0))
    e_uni = big_e * u
    e_def = float(cfg.defensive_e_max) * u  # type: ignore[arg-type]
    e = np.where(comp < wa, e_pow, np.where(comp < wa + wb, e_uni, e_def))
    return np.where(circ, 0.0, e)


def check_eccentricity_bounded(cfg: EccentricityProposalConfig, target_eta_floor: float) -> None:
    """Refuse an eccentricity proposal whose weights can be unbounded (#410).

    ``mds17_bounded`` needs η_q = ``floor_eta`` ≤ the target's eta floor and a nonzero
    ``floor_weight`` and ``support_weight``. ``uniform`` is refused whenever the target eta
    floor is below −0.5 (infinite-variance weights) and always truncates (#409).
    """
    if cfg.shape == "uniform":
        raise ValueError(
            "uniform eccentricity proposal truncates the MdS17 target at e_cap (#409) and has "
            "infinite-variance weights for eta < -0.5 (#410); use shape mds17_bounded"
        )
    if float(cfg.floor_eta) > target_eta_floor:  # type: ignore[arg-type]
        raise ValueError(
            f"proposal floor_eta {cfg.floor_eta} > target eta floor {target_eta_floor}: weights unbounded at e -> 0"
        )
    if not (float(cfg.floor_weight) > 0 and float(cfg.support_weight) > 0):  # type: ignore[arg-type]
        raise ValueError("floor_weight and support_weight must both be > 0 for bounded weights")


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
    rel = proposal_relation_centre(m1, m2, row, parent, fc)
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
    ecc = sample_ecc(period, ec, rng)

    # Orientation and phase: isotropic, identical to the target (cancels in weights).
    cos_i = rng.uniform(-1.0, 1.0, size=n)
    inc_deg = np.degrees(np.arccos(cos_i))
    big_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    small_omega = rng.uniform(0.0, 2 * np.pi, size=n)
    tp = rng.uniform(0.0, 1.0, size=n) * period

    log_q = {
        "log_q_parent": np.log(qs[row]),
        "log_q_log_m2": log_q_log_m2(log_m2, log_m1, m2c),
        "log_q_flux": log_q_flux(log_f, dark, m1, m2, fc, centre=rel),
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
        # #421: galactic (l, b) for the #400 v2 epoch model's sky term (NaN if the parent lacks them).
        "l_deg": np.asarray(cols["l"], float)[row] if "l" in cols else np.full(n, np.nan),
        "b_deg": np.asarray(cols["b"], float)[row] if "b" in cols else np.full(n, np.nan),
        # Truth parallax fed to gaiamock (spec §0.1 MP-Q4); the measured one is kept too.
        "parallax_mas": np.asarray(parent.truth_parallax_mas, float)[row],
        "measured_parallax_mas": np.asarray(cols["parallax"], float)[row],
        "is_giant": np.asarray(parent.is_giant, bool)[row],
        "atmosphere_logg": np.asarray(parent.atmosphere_logg, float)[row],
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
    # Deblended draws (spec §11.9) keep the proposal-space masses separately: q was proposed
    # relative to the row's M̂1, so the density is evaluated there, not at the truth M1.
    m1 = np.asarray(truth.get("m1_row_msun", truth["m1_msun"]), float)
    m2 = np.asarray(truth.get("m2_proposal_msun", truth["m2_msun"]), float)
    with np.errstate(divide="ignore"):
        lqs = np.log(qs[row])
    return (
        lqs
        + log_q_log_m2(np.log10(m2), np.log10(m1), cfg.m2)
        + log_q_flux(truth["log10_flux_ratio"], truth["is_dark"], m1, m2, cfg.flux,
                     centre=proposal_relation_centre(m1, m2, row, parent, cfg.flux))
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


@dataclass(frozen=True)
class EpochSetup:
    """The #400 epoch model for :func:`simulate_one` (built once per worker).

    ``config`` is an :class:`darkhunter_pop.epoch_model.EpochModelConfig` (``dr3.epoch_model``
    with ``enabled`` forced on, #400 E1); ``gaps_jd`` its gap table in JD.
    """

    config: Any
    gaps_jd: Any


def simulate_one(
    draw: Mapping[str, Any],
    *,
    gaiamock: ModuleType,
    c_funcs: Any,
    cfg: ProposalConfig,
    cuts: OrbitalSolutionCutsConfig,
    epoch: EpochSetup | None = None,
) -> dict[str, Any]:
    """Run one stored draw through the gaiamock cascade, seeded per #371.

    Seeds: ``mock_global_rng_seeds(base_seed, PROPOSAL_RNG_STREAM_BASE + generation,
    draw_index)``, so a draw replays alone, independent of order or worker.

    With ``cfg.epoch_model != "off"`` (``epoch`` then required) the draw goes through
    :func:`darkhunter_pop.epoch_model.run_cascade`: gaiamock's
    ``predict_astrometry_luminous_binary`` inside the epoch thinning, then the per-CCD excess
    noise and the RUWE normalization that the epoch config switches on, then
    ``fit_full_astrometric_cascade``, with gaiamock's own visibility gate (< 12 visibility
    periods or < 13 epochs → the all-zero vector, as ``run_full_astrometric_cascade``). The
    epoch and noise Generators are ``epoch_model_rng(base_seed, stream, draw_index[, tag])``;
    the source context carries G and the draw's galactic (l, b) (#421). Without the epoch
    model it is the bare ``run_full_astrometric_cascade`` call.
    """
    if (epoch is None) != (cfg.epoch_model == "off"):
        raise ValueError(f"epoch_model={cfg.epoch_model!r} but epoch is {'missing' if epoch is None else 'given'}")
    stream = PROPOSAL_RNG_STREAM_BASE + int(draw["generation"])
    seeds = mock_global_rng_seeds(cfg.base_seed, stream, int(draw["draw_index"]))
    kw = dict(
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
        w=float(draw["omega_rad"]),
        phot_g_mean_mag=float(draw["phot_g_mean_mag"]),
        f=float(draw["flux_ratio"]),
        data_release=cfg.data_release,
        c_funcs=c_funcs,
    )
    t0 = time.process_time()
    with seeded_global_rng(seeds, c_funcs):
        if epoch is None:
            cascade = gaiamock.run_full_astrometric_cascade(
                inc_deg=float(draw["inc_deg"]), verbose=False, show_residuals=False,
                ruwe_min=cfg.ruwe_min, skip_acceleration=cfg.skip_acceleration, **kw,
            )
        else:
            from darkhunter_pop import epoch_model as em
            from darkhunter_pop.cascade_replay import GAIAMOCK_MIN_OBSERVATIONS, GAIAMOCK_MIN_VISIBILITY_PERIODS

            def predict() -> Any:
                return gaiamock.predict_astrometry_luminous_binary(inc=math.radians(float(draw["inc_deg"])), **kw)

            def _opt(key: str) -> float | None:
                v = draw.get(key)
                return float(v) if v is not None and np.isfinite(float(v)) else None

            source = em.SourceEpochContext(g_mag=float(draw["phot_g_mean_mag"]), l_deg=_opt("l_deg"), b_deg=_opt("b_deg"))
            di = int(draw["draw_index"])
            run = em.run_cascade(
                gaiamock, c_funcs, predict, epoch.config, source,
                epoch_rng=em.epoch_model_rng(cfg.base_seed, stream, di),
                noise_rng=em.epoch_model_rng(cfg.base_seed, stream, di, tag=em.PER_CCD_NOISE_RNG_TAG),
                ruwe_min=cfg.ruwe_min, skip_acceleration=cfg.skip_acceleration, gaps_jd=epoch.gaps_jd,
            )
            if run.n_visibility_periods < GAIAMOCK_MIN_VISIBILITY_PERIODS or run.n_obs < GAIAMOCK_MIN_OBSERVATIONS:
                cascade = [0.0] * CASCADE_VECTOR_LENGTH
            else:
                cascade = run.cascade
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
    truth: Mapping[str, NDArray[Any]],
    target: MdS17TargetConfig,
    *,
    evolved_mg0_system: ArrayLike | None = None,
    relation_log10_f: ArrayLike | None = None,
) -> FloatArray:
    """log λ(x | θ_MdS17) in the proposal's measure (per dex M2, per dex P, per unit e or
    the circular point mass, per dex f), for luminous MS companions only.

    λ = f_logP;q>0.3(M1, P) × p_q(q) × q ln10 × p_e × N(log10 f; log10 f_rel, σ_f) ×
    s_low(M1). Dark draws (f = 0), q outside MdS17's 0.1-1 and log P outside 0.2-8 get
    −inf. M1 above the MdS17 domain is clamped to its edge for the shape (recorded as a
    limitation; TAG10 rarely exceeds it). Provisional choices come from ``target``.

    ``evolved_mg0_system`` (#416 / spec §10.4; per draw, NaN for non-evolved rows, e.g. from
    :func:`evolved_mg0_for_draws`): for evolved primaries the f density is centred on
    ``giants.evolved_log10_flux_ratio(M2, M_G0,sys)`` instead of the dwarf relation, and is
    zero where the companion alone would outshine the system. Dwarf rows are unchanged.
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
    if target.mass_luminosity == "mist_coeval":
        if relation_log10_f is None:
            raise ValueError("mass_luminosity mist_coeval: pass relation_log10_f (mist_relation_for_draws)")
        rel = np.asarray(relation_log10_f, float)
    else:
        rel = relation_log10_flux_ratio(m1, m2)
    if evolved_mg0_system is not None:
        from darkhunter_pop.giants import evolved_log10_flux_ratio

        mg0e = np.asarray(evolved_mg0_system, float)
        evo = np.isfinite(mg0e)
        rel = np.where(evo, evolved_log10_flux_ratio(m2, np.where(evo, mg0e, 0.0)), rel)
    sig = target.flux_sigma_dex
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


# ---------------------------------------------------------------------------
# Real comparison sample filters and the weighted KS statistic (rung 2)
# ---------------------------------------------------------------------------


def load_real_input_columns(snapshot_dir: str | Path) -> dict[str, NDArray[Any]]:
    """Load ``columns.h5`` from ``scripts/fetch_real_nss_input_columns.py`` (checksum verified)."""
    import h5py

    d = Path(snapshot_dir)
    meta = yaml.safe_load((d / "meta.yaml").read_text())
    got = _sha256(d / "columns.h5")
    if got != meta["columns_h5_sha256"]:
        raise ValueError(f"input-columns snapshot checksum mismatch: {got}")
    with h5py.File(d / "columns.h5", "r") as h:
        return {k: h[k][()] for k in h.keys()}


def real_comparison_keep(
    columns: Any,
    proposal: ProposalConfig,
    input_columns: Mapping[str, NDArray[Any]] | None = None,
) -> tuple[NDArray[np.bool_], dict[str, int]]:
    """Rows of the real comparison table kept to mirror the decided parent filters.

    Applies the same measured-parallax floor (MP-Q1) and the same no-atmosphere drop (MP-Q5:
    neither an MSC nor a GSP-Phot atmosphere resolves) as the parent. When ``input_columns``
    is given (MP-Q24, decided 2026-10-03: drop the real rows failing the Halbwachs (b)/(c)
    cuts, for symmetry), the same :func:`halbwachs_input_flags` are applied, joined by
    ``source_id``; a row missing from ``input_columns`` fails. Returns ``(mask, counts)``.
    """
    from darkhunter_pop.mass_derivation import resolve_atmosphere_from_extras

    names = set(getattr(columns, "colnames", None) or columns.keys())
    plx = np.ma.filled(np.ma.asarray(columns["parallax"], dtype=np.float64), np.nan)
    ap_names = [c for c in GAIA_SOURCE_PARENT_COLUMNS if c in _AP_COLUMNS and c in names]
    arrays = {c: np.ma.filled(np.ma.asarray(columns[c], dtype=np.float64), np.nan) for c in ap_names}
    n = plx.size
    has_atm = np.zeros(n, dtype=bool)
    for i in range(n):
        extras = {c: float(arrays[c][i]) for c in ap_names if np.isfinite(arrays[c][i])}
        has_atm[i] = resolve_atmosphere_from_extras(extras) is not None
    with np.errstate(invalid="ignore"):
        floor = plx > proposal.parallax_floor_mas
    keep = floor & has_atm
    counts = {
        "rows": int(n),
        "parallax_floor": int(floor.sum()),
        "parallax_floor_and_atmosphere": int(keep.sum()),
    }
    if input_columns is not None:
        sid = np.asarray(columns["source_id"], dtype=np.int64)
        ref = np.asarray(input_columns["source_id"], dtype=np.int64)
        order = np.argsort(ref)
        pos = np.searchsorted(ref[order], sid)
        pos_c = np.clip(pos, 0, max(ref.size - 1, 0))
        found = (ref.size > 0) & (ref[order][pos_c] == sid)
        idx = order[pos_c]
        joined = {k: np.where(found, np.asarray(v, dtype=np.float64)[idx], np.nan)
                  for k, v in input_columns.items() if k != "source_id"}
        hf = halbwachs_input_flags(joined, proposal.halbwachs_cuts)
        keep = keep & hf["halbwachs_ipd"] & hf["halbwachs_cstar"]
        counts["missing_input_columns"] = int((~found).sum())
        counts["and_halbwachs_ipd_cstar"] = int(keep.sum())
    return keep, counts


def weighted_ks(
    real: ArrayLike, mock: ArrayLike, mock_weights: ArrayLike
) -> tuple[float, float, float]:
    """Two-sample KS between an unweighted real sample and a weighted mock.

    Returns ``(D, n_eff, p)``: ``D`` is the sup distance between the empirical CDFs (mock CDF
    weighted), ``n_eff = n_real ESS / (n_real + ESS)`` with the mock's Kish ESS, and ``p`` the
    asymptotic Kolmogorov tail at ``sqrt(n_eff) D``. Non-finite values are dropped.
    """
    from scipy.special import kolmogorov

    r = np.asarray(real, float)
    m = np.asarray(mock, float)
    w = np.asarray(mock_weights, float)
    r = np.sort(r[np.isfinite(r)])
    ok = np.isfinite(m) & np.isfinite(w) & (w > 0)
    m, w = m[ok], w[ok]
    if r.size == 0 or m.size == 0:
        return float("nan"), 0.0, float("nan")
    order = np.argsort(m)
    m, w = m[order], w[order]
    grid = np.concatenate([r, m])
    cdf_r = np.searchsorted(r, grid, side="right") / r.size
    cw = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    cdf_m = cw[np.searchsorted(m, grid, side="right")]
    d = float(np.max(np.abs(cdf_r - cdf_m)))
    ess = kish_ess(w)
    n_eff = r.size * ess / (r.size + ess)
    return d, float(n_eff), float(kolmogorov(np.sqrt(n_eff) * d))


# ---------------------------------------------------------------------------
# Malmquist conditioning inputs (#405 wired in; spec §9, decisions 2026-10-03)
# ---------------------------------------------------------------------------


class ParentExtinctionConfig(_Strict):
    """MP-Q29 (decided): Combined19 extinction for ΔM.

    ``a_g_per_ebv`` converts Combined19 E(B−V) to A_G; 2.8 is the El-Badry et al. (2024) §3
    value the mock already uses. ``sigma_a_mag`` is 0 by construction: the extinction scatter
    is absorbed into the fitted σ_int (MP-Q25), which is fitted with the same A_G.
    """

    model: Literal["combined19"]
    a_g_per_ebv: float = Field(..., gt=0.0)
    sigma_a_mag: float = Field(..., ge=0.0)


class ZeroPointFitConfig(_Strict):
    """MP-Q25 (decided): fit σ_int and the M_G zero point on single-star-like parent stars."""

    ruwe_max: float = Field(..., gt=0.0)  # RUWE < ruwe_max ("single-star-like")
    exclude_giants: bool
    n_bootstrap: int = Field(..., ge=10)
    bootstrap_seed: int = Field(..., ge=0)


def combined19_a_g(parent: ParentSnapshot, cfg: ParentExtinctionConfig, *, cache: bool = True) -> FloatArray:
    """A_G per parent row from mwdust Combined19 at (l, b, Bailer-Jones distance); NaN where
    the row has no truth distance. Cached beside the snapshot (``a_g_combined19_<k>.npz``)."""
    key = hashlib.sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:10]
    path = parent.path / f"a_g_combined19_{key}.npz"
    if cache and path.exists():
        return np.load(path)["a_g_mag"]
    import mwdust

    with np.errstate(divide="ignore", invalid="ignore"):
        d_kpc = 1.0 / np.asarray(parent.truth_parallax_mas, float)
    ok = np.isfinite(d_kpc) & (d_kpc > 0)
    a_g = np.full(d_kpc.size, np.nan)
    if ok.any():
        # Raw (SFD-scale) mwdust value on purpose: the legacy 1-D weight's A_G = 2.8 x Combined19
        # follows El-Badry et al. (2024) §3 (#418, spec §0.6 keeps it unchanged).
        dust = mwdust.Combined19()
        l_deg = np.asarray(parent.columns["l"], float)[ok]
        b_deg = np.asarray(parent.columns["b"], float)[ok]
        ebv = np.asarray(dust(l_deg, b_deg, d_kpc[ok]), float)
        a_g[ok] = cfg.a_g_per_ebv * ebv
    if cache:
        np.savez(path, a_g_mag=a_g)
    return a_g


@dataclass(frozen=True)
class ZeroPointFit:
    """MLE of (δ_zp, σ_int) with ΔM_raw ~ N(δ_zp, σ_int² + σ_μ² + σ_A² + (∂M_G/∂log M1)² σ²_logM1)."""

    zero_point_mag: float
    sigma_int_mag: float
    zero_point_err_mag: float
    sigma_int_err_mag: float
    n_stars: int
    median_mag: float
    robust_sigma_mag: float
    by_m1_bin: list[dict[str, float]]


def fit_mg_zero_point(
    parent: ParentSnapshot,
    a_g_mag: ArrayLike,
    mcfg: Any,
    fit: ZeroPointFitConfig,
    ext: ParentExtinctionConfig,
) -> ZeroPointFit:
    """Fit the Janssens M_G zero point and σ_int on the parent's single-star-like rows.

    Rows: ``usable`` (spec §0.1 filters), RUWE < ``fit.ruwe_max``, finite ΔM and σ, and not
    giants when ``fit.exclude_giants`` (Janssens is a dwarf relation; MP-Q28 handles giants).
    Uncertainties are bootstrap standard deviations over rows. The median and 1.4826 × MAD are
    reported as a robustness check: unresolved binaries pull ΔM bright and the TAG10 floor
    (#393) pulls M dwarfs faint, so a Gaussian is only an approximation (MP-Q26 ignores the
    TAG10 blended-light bias by decision).
    """
    from scipy.optimize import minimize

    from darkhunter_pop import malmquist as mq

    cols = parent.columns
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1000.0 / np.asarray(parent.truth_parallax_mas, float)
    m1 = np.asarray(parent.m1_msun, float)
    raw = mq.luminosity_excess(cols["phot_g_mean_mag"], d, a_g_mag, m1, zero_point_mag=0.0)
    s_mu = mq.sigma_mu_from_quantiles(cols["r_lo_geo"], cols["r_med_geo"], cols["r_hi_geo"])
    slope = mq.mg_slope_per_dex(m1, mcfg.slope_step_dex)
    var0 = s_mu**2 + ext.sigma_a_mag**2 + (slope * mcfg.provisional_sigma_log_m1_dex) ** 2
    with np.errstate(invalid="ignore"):
        sel = parent.usable & (np.asarray(cols["ruwe"], float) < fit.ruwe_max)
    if fit.exclude_giants:
        sel &= ~np.asarray(parent.is_giant, bool)
    sel &= np.isfinite(raw) & np.isfinite(var0)
    x, v = raw[sel], var0[sel]

    def nll(theta: NDArray[np.float64], xx: FloatArray, vv: FloatArray) -> float:
        zp, ls = theta
        tot = vv + math.exp(2.0 * ls)
        return float(0.5 * np.sum((xx - zp) ** 2 / tot + np.log(tot)))

    def solve(xx: FloatArray, vv: FloatArray) -> tuple[float, float]:
        start = np.array([float(np.median(xx)), math.log(max(1e-3, float(np.std(xx))))])
        r = minimize(nll, start, args=(xx, vv), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
        return float(r.x[0]), float(math.exp(r.x[1]))

    zp, si = solve(x, v)
    rng = np.random.default_rng(fit.bootstrap_seed)
    boots = np.array([solve(x[i], v[i]) for i in (rng.integers(0, x.size, x.size) for _ in range(fit.n_bootstrap))])
    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    bins = []
    m1s = m1[sel]
    for lo, hi in ((0.0, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 100.0)):
        b = (m1s >= lo) & (m1s < hi)
        if b.any():
            bins.append({"m1_lo": lo, "m1_hi": hi, "n": int(b.sum()), "median": float(np.median(x[b])),
                         "robust_sigma": float(1.4826 * np.median(np.abs(x[b] - np.median(x[b]))))})
    return ZeroPointFit(
        zero_point_mag=zp, sigma_int_mag=si,
        zero_point_err_mag=float(np.std(boots[:, 0])), sigma_int_err_mag=float(np.std(boots[:, 1])),
        n_stars=int(x.size), median_mag=med, robust_sigma_mag=mad, by_m1_bin=bins,
    )


def malmquist_log_weight(
    truth: Mapping[str, NDArray[Any]],
    parent: ParentSnapshot,
    target: MdS17TargetConfig,
    mcfg: Any,
    a_g_mag: ArrayLike,
    ext: ParentExtinctionConfig,
) -> tuple[FloatArray, dict[str, int]]:
    """log W per draw (#405) for the decided settings, plus counts of unit-weight draws.

    Add to :func:`mds17_luminous_log_intensity` before :func:`importance_weights`.
    MP-Q30 (decided): ``gaussian_mu`` distance marginalization, enforced here. Giants get
    unit weight through ``mcfg.provisional_giant_policy`` (the MP-Q28 hook, owned elsewhere).
    """
    from darkhunter_pop import malmquist as mq

    if mcfg.provisional_distance_marginalization != "gaussian_mu":
        raise ValueError("MP-Q30 decided gaussian_mu distance marginalization")
    rows = mq.row_conditioning(parent, mcfg, a_g_mag=a_g_mag, sigma_a_mag=ext.sigma_a_mag)
    grid = mq.build_flux_marginal(target, mcfg.grid)
    lw = mq.log_weight_for_draws(truth, rows, grid, mcfg)
    r = np.asarray(truth["parent_row"], dtype=np.int64)
    unit = ~(np.isfinite(rows.delta_m[r]) & np.isfinite(rows.sigma[r]))
    return lw, {"draws": int(r.size), "unit_weight_no_delta_m": int(unit.sum()),
                "unit_weight_giant": int(np.asarray(rows.is_giant, bool)[r].sum())}


# ---------------------------------------------------------------------------
# #418: isochrone parent, 2-D CMD Malmquist weight, evolved flux ratio (#416)
# ---------------------------------------------------------------------------


def evolved_mg0_for_draws(truth: Mapping[str, NDArray[Any]], parent: ParentSnapshot) -> FloatArray:
    """Per draw: the system's dereddened M_G0 where the parent row is evolved, else NaN.

    Input for ``mds17_luminous_log_intensity(..., evolved_mg0_system=...)`` (#416). Needs a
    parent loaded in isochrone mode (``parent.cmd``); raises otherwise.
    """
    if parent.cmd is None:
        raise ValueError("parent has no CMD; load it with m1 = isochrone_mist_drop_unresolved")
    r = np.asarray(truth["parent_row"], np.int64)
    mg0 = np.asarray(parent.cmd["mg0"], float)[r]
    return np.where(np.asarray(parent.is_giant, bool)[r], mg0, np.nan)


def malmquist_cmd_log_weight(
    truth: Mapping[str, NDArray[Any]],
    parent: ParentSnapshot,
    target: MdS17TargetConfig,
    cmcfg: Any,
    config: PipelineConfig,
) -> tuple[FloatArray, dict[str, Any]]:
    """log W per draw from the 2-D CMD weight (spec §11.4), plus unit-weight counts.

    Uses the parent's dereddened CMD, its isochrone M1 and the shared MS ridge stored in
    ``parent.meta["ms_ridge"]`` by :func:`load_parent_snapshot` in isochrone mode. Add to the
    target log-intensity before :func:`importance_weights`. Evolved rows, rows outside the
    ridge and rows without a CMD get log W = 0 (counted).
    """
    from darkhunter_pop import giants
    from darkhunter_pop import isochrone_mass as im
    from darkhunter_pop import malmquist_cmd as mc

    if parent.cmd is None or "ms_ridge" not in parent.meta:
        raise ValueError("parent has no CMD / ridge; load it with m1 = isochrone_mist_drop_unresolved")
    rd = parent.meta["ms_ridge"]
    ridge = mc.RidgeTables.from_ridge(giants.MSRidge(
        colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float),
        sigma=np.asarray(rd["sigma"], float), n_rows=np.asarray(rd["n_rows"], np.int64),
    ))
    grid = im.load_native_grid(config.isochrone_mass, config.paths.data_root)
    cc = cmcfg.companion_colour
    iso = parent.isochrone or {}
    ms = mc.ms_colour_bank(  # MP-Q37 (decided): coeval, the row's own isochrone
        grid, np.asarray(iso.get("feh_mean", np.full(parent.n_rows, np.nan)), float),
        np.asarray(iso.get("log_age_mean", np.full(parent.n_rows, np.nan)), float), cc)
    rows = mc.cmd_rows(
        parent.cmd["colour0"], parent.cmd["mg0"], parent.cmd["sigma_mu"], parent.m1_msun, ridge,
        evolved=parent.is_giant, a_g=parent.cmd["a_g"], e_bp_rp=parent.cmd["e_bp_rp"],
    )
    r = np.asarray(truth["parent_row"], np.int64)
    used = np.zeros(rows.colour0.size, bool)
    used[np.unique(r)] = True
    rows_used = mc.CmdRows(
        colour0=rows.colour0, mg0=rows.mg0, sigma_mu=rows.sigma_mu, k_ag_over_ebprp=rows.k_ag_over_ebprp,
        m1_msun=rows.m1_msun, unit_weight=rows.unit_weight | ~used, unit_reason=rows.unit_reason,
    )
    qf = mc.build_q_grid(target, cmcfg.grid) if target.mass_luminosity == "mist_coeval" else mc.build_qf_grid(target, cmcfg.grid)
    dens = None
    if cmcfg.single_star_density.provisional_model == "mist_density_ridge_anchored":
        model = im.build_model(config.isochrone_mass, config.paths.data_root)
        dens = mc.build_single_star_density(model.cmap, cmcfg.single_star_density, giants.MSRidge(
            colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float),
            sigma=np.asarray(rd["sigma"], float), n_rows=np.asarray(rd["n_rows"], np.int64)))
    norm = mc.row_normalization(rows_used, qf, ridge, ms, cmcfg, dens=dens)
    lw = mc.log_weight_for_draws(truth, rows_used, norm, ridge, ms, cmcfg, dens=dens)
    reasons = rows.unit_reason[r]
    counts = {str(k): int(v) for k, v in zip(*np.unique(reasons, return_counts=True))}
    return lw, {"draws": int(r.size), "by_row_reason": counts}


#: ``SeedSequence`` spawn-key slot of the posterior-draw / deblending stream (after the
#: proposal's own slot 0); disjoint from the gaiamock seeds of the same generation.
POSTERIOR_DRAW_RNG_SLOT: int = 2


def apply_posterior_deblending(
    truth: Mapping[str, NDArray[Any]],
    parent: ParentSnapshot,
    config: PipelineConfig,
    cfg: ProposalConfig,
    *,
    sampler: Any = None,
    native: Any = None,
) -> dict[str, NDArray[Any]]:
    """MP-Q35 + MP-Q36 (decided 2026-10-04; spec §11.9): per draw, one posterior draw of the
    primary's (age, [Fe/H], M̂1), and the deblended truth M1 on that coeval isochrone.

    The proposal's q (= m2 / row M̂1) and f are kept. The truth M1 minimizes χ² of the combined
    (primary + coeval MS companion at q M1 with G-flux ratio f) G and BP−RP against the row's
    dereddened system point (:func:`darkhunter_pop.isochrone_mass.deblend_primary_mass`), on the
    posterior draw's own branch (main sequence or evolved); then
    M2 = q M1. Columns added: ``m1_hat_msun`` (the posterior draw), ``m1_row_msun`` and
    ``m2_proposal_msun`` (proposal space, for :func:`log_q_total_for`), ``iso_feh``,
    ``iso_log_age``, ``deblend_chi2``; ``m1_msun`` / ``m2_msun`` become the deblended truth.
    Draws whose row has no posterior keep their masses (``deblend_chi2`` NaN). Deterministic:
    ``SeedSequence(base_seed, spawn_key=(PROPOSAL_RNG_STREAM_BASE + generation,
    POSTERIOR_DRAW_RNG_SLOT))``. Interpolation only.
    """
    from darkhunter_pop import isochrone_mass as im

    if parent.cmd is None:
        raise ValueError("posterior deblending needs an isochrone-mode parent (parent.cmd)")
    icfg = config.isochrone_mass
    if native is None:
        native = im.load_native_grid(icfg, config.paths.data_root)
    if sampler is None:
        sampler = im.PosteriorSampler.build(im.prior_points(native, icfg), icfg.cmd_map)
    stream = PROPOSAL_RNG_STREAM_BASE + cfg.generation
    rng = np.random.default_rng(np.random.SeedSequence(cfg.base_seed, spawn_key=(stream, POSTERIOR_DRAW_RNG_SLOT)))
    out = {k: np.asarray(v).copy() for k, v in truth.items()}
    r = np.asarray(truth["parent_row"], np.int64)
    c0 = np.asarray(parent.cmd["colour0"], float)[r]
    g0 = np.asarray(parent.cmd["mg0"], float)[r]
    ebv = np.asarray(parent.cmd.get("ebv", np.zeros(parent.n_rows)), float)[r]
    a_g = np.asarray(parent.cmd.get("a_g", np.zeros(parent.n_rows)), float)[r]
    ebr = np.asarray(parent.cmd.get("e_bp_rp", np.zeros(parent.n_rows)), float)[r]
    with np.errstate(divide="ignore", invalid="ignore"):
        ka = np.where(ebv > 0, a_g / ebv, 0.0)
        ke = np.where(ebv > 0, ebr / ebv, 0.0)
    sc, sm = im.likelihood_sigmas(np.asarray(parent.cmd["sigma_mu"], float)[r], ebv, ka, ke, icfg.likelihood)
    draw = sampler.sample(c0, g0, sc, sm, rng, kernel_n_sigma=icfg.likelihood.kernel_n_sigma)
    m1_row = np.asarray(truth["m1_msun"], float)
    m2_prop = np.asarray(truth["m2_msun"], float)
    q = m2_prop / m1_row
    lf = np.where(np.asarray(truth["is_dark"], bool), -np.inf, np.asarray(truth["log10_flux_ratio"], float))
    m1_true = m1_row.copy()
    chi2 = np.full(r.size, np.nan)
    ok = np.isfinite(draw["m1"]) & np.isfinite(c0) & np.isfinite(g0)
    # Draws outside the target's support (q > 1, or a companion brighter than its primary,
    # f > 1) have zero target weight; they keep the row's M̂1 and are not deblended.
    ok &= (q <= 1.0) & ~(np.isfinite(lf) & (lf > 0.0))
    key = np.round(draw["feh"], 4) * 1e4 + np.round(draw["log_age"], 4)
    for k in np.unique(key[ok]):
        sel = np.flatnonzero(ok & (key == k))
        j = sel[0]
        iso = im.isochrone_at(native, float(draw["feh"][j]), float(draw["log_age"][j]))
        if iso["star_mass"].size < 2:
            continue
        for a in range(0, sel.size, 2000):
            ss = sel[a:a + 2000]
            mm, cc = im.deblend_primary_mass(iso, c0[ss], g0[ss], sc[ss], sm[ss], q[ss], lf[ss],
                                             evolved=draw["phase"][ss] >= 1.5)
            m1_true[ss] = mm
            chi2[ss] = cc
    out["m1_hat_msun"] = draw["m1"]
    out["iso_feh"] = draw["feh"]
    out["iso_log_age"] = draw["log_age"]
    out["deblend_chi2"] = chi2
    out["m1_row_msun"] = m1_row
    out["m2_proposal_msun"] = m2_prop
    out["m1_msun"] = m1_true
    out["m2_msun"] = q * m1_true
    return out


def mist_relation_for_draws(
    truth: Mapping[str, NDArray[Any]],
    parent: ParentSnapshot,
    config: PipelineConfig,
    *,
    native: Any = None,
    feh_step_dex: float = 0.05,
) -> FloatArray:
    """MP-Q40: log10 f = −0.4 [M_G(M2) − M_G(M1)] on the coeval MIST main sequence, per draw.

    The isochrone is the draw's own posterior draw (``iso_feh`` / ``iso_log_age``, after
    :func:`apply_posterior_deblending`) or else the row's posterior means; [Fe/H] is rounded
    to ``feh_step_dex`` and linear between MIST files, the age is the nearest native one
    (:func:`darkhunter_pop.malmquist_cmd.ms_colour_bank`). Masses are clamped to the MS mass
    range of that isochrone.
    """
    from darkhunter_pop import isochrone_mass as im
    from darkhunter_pop import malmquist_cmd as mc

    if native is None:
        native = im.load_native_grid(config.isochrone_mass, config.paths.data_root)
    r = np.asarray(truth["parent_row"], np.int64)
    iso = parent.isochrone or {}
    feh = np.asarray(truth.get("iso_feh", np.asarray(iso.get("feh_mean", np.full(parent.n_rows, np.nan)), float)[r]), float)
    age = np.asarray(truth.get("iso_log_age", np.asarray(iso.get("log_age_mean", np.full(parent.n_rows, np.nan)), float)[r]), float)
    bank = mc.ms_colour_bank(native, feh, age, mc.CompanionColourConfig(feh_step_dex=feh_step_dex))
    m1 = np.asarray(truth["m1_msun"], float)
    m2 = np.asarray(truth["m2_msun"], float)
    out = np.full(r.size, np.nan)
    for ms, pos in bank.for_rows(np.arange(r.size)):
        out[pos] = ms.log10_flux_ratio(m1[pos], m2[pos])
    return out

