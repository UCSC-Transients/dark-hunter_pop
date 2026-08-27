"""Stages: ``mass_derivation_bulk`` and ``mass_derivation_refined``.

Bulk path: MSC (else gspphot) atmospheric parameters → Torres, Andersen & Giménez
(2010) TAG10 mass/radius with analytic uncertainty propagation and optional Santos
et al. (2013) correction, then companion mass via ``gaiamock_mod`` mass-function
inversion and the config M2 cut.

Refined path: consume ``dark-hunter_sed`` / uberMS (async queue, cache, re-run on
new data) without forking photometry gathering. Watch-list when M1 approaches the
configured uberMS prior cap (ARCHITECTURE.md §4).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
from numpy.typing import NDArray

from darkhunter_pop import constants
from darkhunter_pop.config_loader import require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    MassCalibrationMethod,
    PipelineConfig,
)
from darkhunter_pop.data_acquisition import read_stage_hdf5 as read_data_acquisition_hdf5
from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.plotting import plot_histogram
from darkhunter_pop.schemas import (
    CandidateRecord,
    FitTier,
    ParameterSet,
    RunManifest,
    StageStatus,
    ThieleInnesElements,
)

# Optional dark-hunter_sed (uberMS) — refined stage wires to it when installed.
try:
    from darkhunter_sed import batch as _sed_batch
    from darkhunter_sed import config as _sed_config
    from darkhunter_sed.posterior import (
        read_sed_summary as _sed_read_summary,
        sed_summary_path as _sed_summary_path,
    )

    _SED_AVAILABLE = True
except ImportError:  # pragma: no cover - optional sibling package
    _sed_batch = None  # type: ignore[assignment]
    _sed_config = None  # type: ignore[assignment]
    _sed_read_summary = None  # type: ignore[assignment]
    _sed_summary_path = None  # type: ignore[assignment]
    _SED_AVAILABLE = False

# Gaia astrophysical_parameters keys expected on CandidateRecord.extras
# (populated by data_acquisition when the AP join is present).
_MSC_TEFF = "teff_msc1"
_MSC_LOGG = "logg_msc1"
_MSC_MH = "mh_msc"
_GSP_TEFF = "teff_gspphot"
_GSP_LOGG = "logg_gspphot"
_GSP_MH = "mh_gspphot"

LN10 = math.log(10.0)


class GaiamockMassAPI(Protocol):
    """Subset of ``gaiamock_mod`` used for photocenter a0 and M2 inversion."""

    def get_Campbell_elements(
        self, A: float, B: float, F: float, G: float
    ) -> tuple[float, float, float, float]:
        ...

    def get_companion_mass_from_mass_function(
        self,
        M1: float,
        a0_mas: float,
        period: float,
        parallax: float,
        fluxratio: float,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> float:
        ...


@dataclass(frozen=True)
class AtmosphereParams:
    """Teff / log g / [Fe/H] with optional 1σ uncertainties."""

    teff_k: float
    logg: float
    mh: float
    teff_err: float | None
    logg_err: float | None
    mh_err: float | None
    source: str  # "MSC" | "gspphot"


@dataclass(frozen=True)
class BulkFunnel:
    """Before/after counts for the bulk mass-derivation stage."""

    input_candidates: int
    atmosphere_ok: int
    m1_ok: int
    m2_ok: int
    after_m2_cut: int
    skipped_no_atmosphere: int
    skipped_no_orbit: int
    skipped_m2_failed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "input_candidates": self.input_candidates,
            "atmosphere_ok": self.atmosphere_ok,
            "m1_ok": self.m1_ok,
            "m2_ok": self.m2_ok,
            "after_m2_cut": self.after_m2_cut,
            "skipped_no_atmosphere": self.skipped_no_atmosphere,
            "skipped_no_orbit": self.skipped_no_orbit,
            "skipped_m2_failed": self.skipped_m2_failed,
        }


@dataclass(frozen=True)
class BulkDiagnostics:
    funnel: BulkFunnel
    m2_pre_cut_msun: NDArray[np.floating]
    m2_post_cut_msun: NDArray[np.floating]


@dataclass(frozen=True)
class RefinedDiagnostics:
    queued: int
    fit_attempted: int
    fit_cached: int
    fit_succeeded: int
    fit_failed: int
    watchlist_source_ids: tuple[int, ...]
    information_gain_order: tuple[int, ...]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _half_range_err(upper: Any, lower: Any) -> float | None:
    hi = _finite(upper)
    lo = _finite(lower)
    if hi is None or lo is None:
        return None
    return 0.5 * abs(hi - lo)


def resolve_atmosphere(candidate: CandidateRecord) -> AtmosphereParams | None:
    """Prefer MSC primary parameters; fall back to gspphot (ARCHITECTURE.md §4)."""
    extras = candidate.extras

    def _pack(
        teff_key: str,
        logg_key: str,
        mh_key: str,
        source: str,
    ) -> AtmosphereParams | None:
        teff = _finite(extras.get(teff_key))
        logg = _finite(extras.get(logg_key))
        mh = _finite(extras.get(mh_key))
        if teff is None or logg is None or mh is None:
            return None
        if teff <= 0:
            return None
        teff_err = _finite(extras.get(f"{teff_key}_error"))
        if teff_err is None:
            teff_err = _half_range_err(
                extras.get(f"{teff_key}_upper"), extras.get(f"{teff_key}_lower")
            )
        logg_err = _finite(extras.get(f"{logg_key}_error"))
        if logg_err is None:
            logg_err = _half_range_err(
                extras.get(f"{logg_key}_upper"), extras.get(f"{logg_key}_lower")
            )
        mh_err = _finite(extras.get(f"{mh_key}_error"))
        if mh_err is None:
            mh_err = _half_range_err(
                extras.get(f"{mh_key}_upper"), extras.get(f"{mh_key}_lower")
            )
        return AtmosphereParams(
            teff_k=teff,
            logg=logg,
            mh=mh,
            teff_err=teff_err,
            logg_err=logg_err,
            mh_err=mh_err,
            source=source,
        )

    msc = _pack(_MSC_TEFF, _MSC_LOGG, _MSC_MH, "MSC")
    if msc is not None:
        return msc
    return _pack(_GSP_TEFF, _GSP_LOGG, _GSP_MH, "gspphot")


def tag10_log_mass_radius(
    teff_k: float,
    logg: float,
    mh: float,
) -> tuple[float, float]:
    """Evaluate TAG10 Table 1 polynomials for ``log10 M`` and ``log10 R``.

    Coefficients come only from :mod:`darkhunter_pop.constants`.
    """
    if teff_k <= 0:
        raise ValueError("teff_k must be positive")
    x = math.log10(teff_k) - constants.TAG10_X_OFFSET
    a = constants.TAG10_A
    b = constants.TAG10_B
    log_g = float(logg)
    feh = float(mh)
    log_m = (
        a[0]
        + a[1] * x
        + a[2] * x**2
        + a[3] * x**3
        + a[4] * log_g**2
        + a[5] * log_g**3
        + a[6] * feh
    )
    log_r = (
        b[0]
        + b[1] * x
        + b[2] * x**2
        + b[3] * x**3
        + b[4] * log_g**2
        + b[5] * log_g**3
        + b[6] * feh
    )
    return float(log_m), float(log_r)


def tag10_analytic_sigma_log(
    teff_k: float,
    logg: float,
    mh: float,
    *,
    teff_err: float | None,
    logg_err: float | None,
    mh_err: float | None,
    sigma_log_intrinsic: float,
    coeffs: NDArray[np.floating],
) -> float:
    """Propagate atmospheric uncertainties through TAG10 via analytic partials.

    Combines input-parameter contributions in quadrature with ``sigma_log_intrinsic``
    (config ``sigma_logM`` / ``sigma_logR``). Coefficient-table errors are not mixed
    in (ARCHITECTURE.md §4: polynomial partials + config scatter).
    """
    x = math.log10(teff_k) - constants.TAG10_X_OFFSET
    # ∂logQ/∂X, ∂logQ/∂logg, ∂logQ/∂[Fe/H]
    d_dx = coeffs[1] + 2.0 * coeffs[2] * x + 3.0 * coeffs[3] * x**2
    d_dlogg = 2.0 * coeffs[4] * logg + 3.0 * coeffs[5] * logg**2
    d_dmh = coeffs[6]
    # X = log10(Teff) - offset → ∂X/∂Teff = 1/(Teff ln 10)
    d_dteff = d_dx / (teff_k * LN10)

    variance = float(sigma_log_intrinsic) ** 2
    if teff_err is not None and teff_err > 0:
        variance += (d_dteff * teff_err) ** 2
    if logg_err is not None and logg_err > 0:
        variance += (d_dlogg * logg_err) ** 2
    if mh_err is not None and mh_err > 0:
        variance += (d_dmh * mh_err) ** 2
    return float(math.sqrt(max(variance, 0.0)))


def apply_santos_correction(
    mass_msun: float,
    sigma_mass_msun: float,
) -> tuple[float, float]:
    """Santos et al. (2013) quadratic TAG10→isochrone mass correction.

    ``M_corr = s2 * M^2 + s1 * M + s0`` with coefficients from ``constants``.
    """
    s2 = constants.SANTOS2013_S2
    s1 = constants.SANTOS2013_S1
    s0 = constants.SANTOS2013_S0
    m_corr = s2 * mass_msun**2 + s1 * mass_msun + s0
    dmdm = 2.0 * s2 * mass_msun + s1
    sigma_corr = abs(dmdm) * sigma_mass_msun
    return float(m_corr), float(sigma_corr)


def log_mass_to_linear(log_m: float, sigma_log_m: float) -> tuple[float, float]:
    """Convert ``log10 M`` ± σ to linear mass (Msun) with first-order σ."""
    mass = 10.0**log_m
    sigma = abs(mass * LN10 * sigma_log_m)
    return float(mass), float(sigma)


def derive_tag10_m1_r1(
    atmosphere: AtmosphereParams,
    config: PipelineConfig,
) -> ParameterSet:
    """Build a ``ParameterSet`` for (M1, R1) at ``FitTier.bulk_estimate`` provenance."""
    method = config.mass_calibration.method
    if method is not MassCalibrationMethod.TAG10:
        raise NotImplementedError(
            f"mass_calibration.method={method.value!r} is not implemented "
            "(v1 supports TAG10 only)"
        )

    log_m, log_r = tag10_log_mass_radius(
        atmosphere.teff_k, atmosphere.logg, atmosphere.mh
    )
    sigma_log_m = tag10_analytic_sigma_log(
        atmosphere.teff_k,
        atmosphere.logg,
        atmosphere.mh,
        teff_err=atmosphere.teff_err,
        logg_err=atmosphere.logg_err,
        mh_err=atmosphere.mh_err,
        sigma_log_intrinsic=config.mass_calibration.sigma_logM,
        coeffs=constants.TAG10_A,
    )
    sigma_log_r = tag10_analytic_sigma_log(
        atmosphere.teff_k,
        atmosphere.logg,
        atmosphere.mh,
        teff_err=atmosphere.teff_err,
        logg_err=atmosphere.logg_err,
        mh_err=atmosphere.mh_err,
        sigma_log_intrinsic=config.mass_calibration.sigma_logR,
        coeffs=constants.TAG10_B,
    )
    m1, sigma_m1 = log_mass_to_linear(log_m, sigma_log_m)
    r1, sigma_r1 = log_mass_to_linear(log_r, sigma_log_r)

    provenance_parts = ["TAG10", atmosphere.source]
    if config.mass_calibration.santos_correction:
        m1, sigma_m1 = apply_santos_correction(m1, sigma_m1)
        provenance_parts.append("Santos2013")

    provenance = "+".join(provenance_parts)
    # Uncorrelated diagonals in v1 (partials treat Teff/logg/[Fe/H] jointly for each
    # marginal; M–R cross term deferred until samples-backed ParameterSet).
    return ParameterSet(
        names=["M1", "R1"],
        values=[m1, r1],
        covariance=[[sigma_m1**2, 0.0], [0.0, sigma_r1**2]],
        provenance=provenance,
        units=["Msun", "Rsun"],
    )


def photocenter_a0_mas(
    thiele_innes: ThieleInnesElements,
    gaiamock: GaiamockMassAPI,
) -> float:
    """Photocenter semi-major axis in mas from Thiele–Innes via ``gaiamock_mod``."""
    if (
        thiele_innes.A is None
        or thiele_innes.B is None
        or thiele_innes.F is None
        or thiele_innes.G is None
    ):
        raise ValueError("Thiele–Innes A/B/F/G required for a0")
    a0, _omega, _w, _inc = gaiamock.get_Campbell_elements(
        float(thiele_innes.A),
        float(thiele_innes.B),
        float(thiele_innes.F),
        float(thiele_innes.G),
    )
    return float(a0)


def companion_mass_m2(
    m1_msun: float,
    *,
    a0_mas: float,
    period_day: float,
    parallax_mas: float,
    flux_ratio: float,
    gaiamock: GaiamockMassAPI,
    sigma_m1_msun: float | None = None,
) -> ParameterSet:
    """Invert the astrometric mass function for M2 (dark-companion flux ratio from config)."""
    m2 = float(
        gaiamock.get_companion_mass_from_mass_function(
            M1=m1_msun,
            a0_mas=a0_mas,
            period=period_day,
            parallax=parallax_mas,
            fluxratio=flux_ratio,
        )
    )
    if sigma_m1_msun is None or sigma_m1_msun <= 0:
        sigma_m2 = 0.0
    else:
        # Finite-difference ∂M2/∂M1 at fixed orbit (gaiamock owns the transcendental root).
        delta = max(sigma_m1_msun * 0.1, 1e-4 * max(m1_msun, 1.0))
        m2_hi = float(
            gaiamock.get_companion_mass_from_mass_function(
                M1=m1_msun + delta,
                a0_mas=a0_mas,
                period=period_day,
                parallax=parallax_mas,
                fluxratio=flux_ratio,
            )
        )
        m2_lo = float(
            gaiamock.get_companion_mass_from_mass_function(
                M1=max(m1_msun - delta, 1e-6),
                a0_mas=a0_mas,
                period=period_day,
                parallax=parallax_mas,
                fluxratio=flux_ratio,
            )
        )
        dmdm = (m2_hi - m2_lo) / (2.0 * delta)
        sigma_m2 = abs(dmdm) * sigma_m1_msun

    return ParameterSet(
        names=["M2"],
        values=[m2],
        covariance=[[sigma_m2**2]],
        provenance="gaiamock_mass_function+TAG10_M1",
        units=["Msun"],
    )


def passes_m2_mass_cut(
    m2_msun: float,
    sigma_m2_msun: float | None,
    *,
    m_min_msun: float,
    n_sigma: float,
) -> bool:
    """Retain when ``M2 + n_sigma * sigma_M2 >= M_min`` (all values from config)."""
    sigma = 0.0 if sigma_m2_msun is None else float(sigma_m2_msun)
    return (m2_msun + n_sigma * sigma) >= m_min_msun


def information_gain_stub(
    candidate: CandidateRecord,
    config: PipelineConfig,
) -> float:
    """Higher = higher priority for the refined uberMS queue.

    Stub uses bulk M1 relative uncertainty until the diagnostics stage owns a full
    information-gain diagnostic (ARCHITECTURE.md §4).
    """
    if candidate.m1 is None:
        return 0.0
    try:
        marginal = candidate.m1.marginal("M1")
    except KeyError:
        return 0.0
    sigma = marginal.sigma if marginal.sigma is not None else 0.0
    floor = config.mass_derivation.information_gain_sigma_floor_msun
    scale = max(abs(marginal.value), floor)
    return float(sigma / scale)


def approaches_uberms_m1_prior_cap(
    m1_msun: float,
    config: PipelineConfig,
) -> bool:
    """True when M1 is within the configured fraction of the uberMS prior upper edge."""
    cap = config.mass_derivation.uberms_m1_prior_max_msun
    frac = config.mass_derivation.uberms_m1_watchlist_fraction
    return m1_msun >= frac * cap


def format_bulk_funnel_table(diagnostics: BulkDiagnostics) -> str:
    """Human-readable bulk funnel (exempt from caveman compression)."""
    funnel = diagnostics.funnel
    lines = [
        "mass_derivation_bulk funnel",
        f"  input_candidates:       {funnel.input_candidates}",
        f"  atmosphere_ok:          {funnel.atmosphere_ok}",
        f"  m1_ok:                  {funnel.m1_ok}",
        f"  m2_ok:                  {funnel.m2_ok}",
        f"  after_m2_cut:           {funnel.after_m2_cut}",
        f"  skipped_no_atmosphere:  {funnel.skipped_no_atmosphere}",
        f"  skipped_no_orbit:       {funnel.skipped_no_orbit}",
        f"  skipped_m2_failed:      {funnel.skipped_m2_failed}",
        f"  m2_pre_cut_n:           {len(diagnostics.m2_pre_cut_msun)}",
        f"  m2_post_cut_n:          {len(diagnostics.m2_post_cut_msun)}",
    ]
    return "\n".join(lines)


def format_refined_report(diagnostics: RefinedDiagnostics) -> str:
    """Human-readable refined-stage report (exempt from caveman compression)."""
    lines = [
        "mass_derivation_refined report",
        f"  queued:                 {diagnostics.queued}",
        f"  fit_attempted:          {diagnostics.fit_attempted}",
        f"  fit_cached:             {diagnostics.fit_cached}",
        f"  fit_succeeded:          {diagnostics.fit_succeeded}",
        f"  fit_failed:             {diagnostics.fit_failed}",
        f"  watchlist_n:            {len(diagnostics.watchlist_source_ids)}",
        f"  watchlist_source_ids:   {list(diagnostics.watchlist_source_ids)}",
        f"  information_gain_order: {list(diagnostics.information_gain_order)}",
    ]
    return "\n".join(lines)


def process_bulk_candidate(
    candidate: CandidateRecord,
    config: PipelineConfig,
    gaiamock: GaiamockMassAPI,
) -> tuple[CandidateRecord | None, str | None, float | None]:
    """Derive M1/M2 for one candidate.

    Returns ``(updated_candidate_or_None, skip_reason, m2_pre_cut)``.
    ``m2_pre_cut`` is set whenever M2 is computed (even if the cut rejects).
    """
    atmosphere = resolve_atmosphere(candidate)
    if atmosphere is None:
        return None, "no_atmosphere", None

    try:
        m1_set = derive_tag10_m1_r1(atmosphere, config)
    except (ValueError, NotImplementedError):
        return None, "m1_failed", None

    m1_marg = m1_set.marginal("M1")
    period = _finite(candidate.nss_orbital.get("period"))
    parallax = candidate.parallax_mas
    if parallax is None:
        parallax = _finite(candidate.nss_orbital.get("parallax"))
    if (
        candidate.thiele_innes is None
        or period is None
        or parallax is None
        or parallax <= 0
        or period <= 0
    ):
        return None, "no_orbit", None

    try:
        a0 = photocenter_a0_mas(candidate.thiele_innes, gaiamock)
        m2_set = companion_mass_m2(
            m1_marg.value,
            a0_mas=a0,
            period_day=period,
            parallax_mas=parallax,
            flux_ratio=config.mass_derivation.dark_companion_flux_ratio,
            gaiamock=gaiamock,
            sigma_m1_msun=m1_marg.sigma,
        )
    except (ValueError, ZeroDivisionError):
        return None, "m2_failed", None

    m2_marg = m2_set.marginal("M2")
    if not passes_m2_mass_cut(
        m2_marg.value,
        m2_marg.sigma,
        m_min_msun=config.classification.M_MIN_msun,
        n_sigma=config.classification.n_sigma_mass_cut,
    ):
        return None, "m2_cut", m2_marg.value

    updated = candidate.model_copy(
        update={
            "m1": m1_set,
            "m2": m2_set,
            "fit_tier": FitTier.BULK_ESTIMATE,
        }
    )
    return updated, None, m2_marg.value


def run_bulk_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
    *,
    gaiamock: GaiamockMassAPI | None = None,
) -> tuple[list[CandidateRecord], BulkDiagnostics]:
    """Apply bulk TAG10 + M2 cut to an in-memory candidate list."""
    api = gaiamock if gaiamock is not None else import_gaiamock_mod()
    kept: list[CandidateRecord] = []
    m2_pre: list[float] = []
    m2_post: list[float] = []
    atmosphere_ok = 0
    m1_ok = 0
    m2_ok = 0
    skipped_no_atmosphere = 0
    skipped_no_orbit = 0
    skipped_m2_failed = 0

    for candidate in candidates:
        updated, reason, m2_pre_val = process_bulk_candidate(candidate, config, api)
        if reason == "no_atmosphere":
            skipped_no_atmosphere += 1
            continue
        atmosphere_ok += 1
        if reason == "m1_failed":
            continue
        m1_ok += 1
        if reason == "no_orbit":
            skipped_no_orbit += 1
            continue
        if reason == "m2_failed":
            skipped_m2_failed += 1
            continue
        if m2_pre_val is not None:
            m2_pre.append(m2_pre_val)
            m2_ok += 1
        if reason == "m2_cut":
            continue
        if updated is None:
            continue
        kept.append(updated)
        if updated.m2 is not None:
            m2_post.append(updated.m2.marginal("M2").value)

    diagnostics = BulkDiagnostics(
        funnel=BulkFunnel(
            input_candidates=len(candidates),
            atmosphere_ok=atmosphere_ok,
            m1_ok=m1_ok,
            m2_ok=m2_ok,
            after_m2_cut=len(kept),
            skipped_no_atmosphere=skipped_no_atmosphere,
            skipped_no_orbit=skipped_no_orbit,
            skipped_m2_failed=skipped_m2_failed,
        ),
        m2_pre_cut_msun=np.asarray(m2_pre, dtype=np.float64),
        m2_post_cut_msun=np.asarray(m2_post, dtype=np.float64),
    )
    return kept, diagnostics


def write_bulk_diagnostic_artifacts(
    diagnostics: BulkDiagnostics,
    artifact_path: Path,
    config: PipelineConfig,
) -> list[Path]:
    """Write funnel text and optional M2 histograms beside the stage HDF5."""
    out_dir = artifact_path.parent / f"{artifact_path.stem}_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    funnel_path = out_dir / "funnel.txt"
    funnel_path.write_text(format_bulk_funnel_table(diagnostics), encoding="utf-8")
    written.append(funnel_path)

    diag = config.diagnostics
    if not diag.write_figures:
        return written

    dpi = int(diag.figure_dpi)
    max_bins = int(diag.histogram_max_bins)
    style = config.plotting
    for name, values, title in (
        ("m2_pre_cut", diagnostics.m2_pre_cut_msun, "M2 pre-cut"),
        ("m2_post_cut", diagnostics.m2_post_cut_msun, "M2 post-cut"),
    ):
        path = plot_histogram(
            values,
            out_dir / f"{name}.png",
            xlabel=r"companion mass (M$_\odot$)",
            ylabel="count",
            title=title,
            dpi=dpi,
            max_bins=max_bins,
            style=style,
        )
        if path is not None:
            written.append(path)
    return written


def write_stage_hdf5(
    path: Path,
    candidates: Sequence[CandidateRecord],
    *,
    stage_name: str,
    diagnostics: Mapping[str, Any],
) -> None:
    """Write one stage HDF5 (candidates JSON + diagnostics attrs/datasets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records_json = [
        json.dumps(candidate.model_dump(mode="json"), sort_keys=True)
        for candidate in candidates
    ]
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["stage"] = stage_name
        meta.attrs["n_candidates"] = len(candidates)

        diag = handle.create_group("diagnostics")
        for key, value in diagnostics.items():
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, np.integer)):
                diag.create_dataset(key, data=np.asarray(value, dtype=np.int64))
            elif isinstance(value, np.ndarray):
                diag.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)) and (
                not value or isinstance(value[0], (float, np.floating))
            ):
                diag.create_dataset(key, data=np.asarray(value, dtype=np.float64))
            elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                diag.attrs[key] = value
            else:
                diag.attrs[key] = json.dumps(value, sort_keys=True)

        cand = handle.create_group("candidates")
        cand.create_dataset(
            "records_json",
            data=np.array(records_json, dtype=h5py.string_dtype("utf-8")),
        )
        cand.create_dataset(
            "source_ids",
            data=np.array([c.source_id for c in candidates], dtype=np.int64),
        )


def read_stage_hdf5(path: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """Load candidates and meta/diagnostics from a mass_derivation stage HDF5."""
    candidates: list[CandidateRecord] = []
    meta: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        meta_group = handle["meta"]
        for key in meta_group.attrs:
            meta[key] = meta_group.attrs[key]
        if "diagnostics" in handle:
            diag = handle["diagnostics"]
            for key in diag.attrs:
                meta[f"diagnostics.{key}"] = diag.attrs[key]
            for key in diag.keys():
                meta[f"diagnostics.{key}"] = np.asarray(diag[key])
        for raw in handle["candidates"]["records_json"].asstr():
            candidates.append(CandidateRecord.model_validate(json.loads(raw)))
    return candidates, meta


def _upstream_artifact(manifest: RunManifest, stage_name: str) -> Path:
    record = manifest.stages.get(stage_name)
    if record is None or not record.artifact_path:
        raise FileNotFoundError(
            f"upstream stage {stage_name!r} has no artifact_path on the run manifest"
        )
    path = Path(record.artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"upstream artifact missing: {path}")
    return path


def _load_upstream_candidates(manifest: RunManifest, stage_name: str) -> list[CandidateRecord]:
    path = _upstream_artifact(manifest, stage_name)
    # data_acquisition and mass_derivation share the records_json layout.
    if stage_name == "data_acquisition":
        candidates, _meta = read_data_acquisition_hdf5(path)
        return candidates
    candidates, _meta = read_stage_hdf5(path)
    return candidates


def iter_upstream_candidates(
    manifest: RunManifest,
    stage_name: str,
    *,
    chunk_size: int = 1024,
) -> Iterator[CandidateRecord]:
    """Stream ``CandidateRecord``s from an upstream stage's HDF5 artifact.

    Reads the shared ``candidates/records_json`` dataset (the layout is
    identical whether the artifact was written by ``data_acquisition`` or by
    this module — only the ``meta``/``diagnostics`` groups differ) in
    ``chunk_size``-row slices rather than materializing the whole array, so a
    consumer that only needs one candidate at a time doesn't have to hold the
    full catalog in memory. At full NSS-catalog scale (tens of thousands of
    rows now, more with DR4) that matters; for anything that genuinely needs
    the full list at once, ``_load_upstream_candidates`` remains available.

    Parameters
    ----------
    manifest:
        The run manifest whose ``stages[stage_name]`` records the upstream
        artifact path.
    stage_name:
        Name of the upstream stage to read (e.g. ``"data_acquisition"``,
        ``"mass_derivation_bulk"``).
    chunk_size:
        Number of rows to read from the HDF5 dataset per slice. Must be >= 1.
        This only bounds peak memory use during iteration; it does not change
        what is yielded.

    Yields
    ------
    CandidateRecord
        One record at a time, in on-disk order.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    path = _upstream_artifact(manifest, stage_name)
    with h5py.File(path, "r") as handle:
        strings = handle["candidates"]["records_json"].asstr()
        n_rows = strings.shape[0]
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            for raw in strings[start:stop]:
                yield CandidateRecord.model_validate(json.loads(raw))


def _default_sed_summary_loader(source_id: int) -> dict[str, Any] | None:
    """Load ``darkhunter_sed`` sed_summary.json when the package is available."""
    if not _SED_AVAILABLE or _sed_summary_path is None or _sed_read_summary is None:
        return None
    path = _sed_summary_path(str(source_id))
    if not path.is_file():
        return None
    return _sed_read_summary(path)


def _default_sed_needs_update(source_id: int) -> tuple[bool, str]:
    if not _SED_AVAILABLE or _sed_batch is None or _sed_config is None:
        return False, "darkhunter_sed_unavailable"
    return _sed_batch.needs_update(
        str(source_id),
        spec_root_path=_sed_config.spec_root(),
        phot_dir=_sed_config.photometry_dir(),
        rv_out=_sed_config.rv_output_dir(),
    )


def _default_sed_fit(source_id: int) -> dict[str, Any] | None:
    """Run one-star uberMS via ``darkhunter_sed.batch.fit_one_star`` (no photometry fork)."""
    if not _SED_AVAILABLE or _sed_batch is None or _sed_summary_path is None:
        return None
    if _sed_read_summary is None:
        return None
    _sed_batch.fit_one_star(str(source_id))
    path = _sed_summary_path(str(source_id))
    if not path.is_file():
        return None
    return _sed_read_summary(path)


def run_mass_derivation_bulk(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    gaiamock: GaiamockMassAPI | None = None,
    candidates: Sequence[CandidateRecord] | None = None,
) -> RunManifest:
    """Execute ``mass_derivation_bulk``: TAG10 M1 + M2 cut → HDF5 + diagnostics."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["mass_derivation_bulk"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    if candidates is None:
        candidates = _load_upstream_candidates(manifest, "data_acquisition")

    kept, diagnostics = run_bulk_on_candidates(
        candidates, config, gaiamock=gaiamock
    )
    write_stage_hdf5(
        artifact,
        kept,
        stage_name="mass_derivation_bulk",
        diagnostics={
            **diagnostics.funnel.as_dict(),
            "m2_pre_cut_msun": diagnostics.m2_pre_cut_msun,
            "m2_post_cut_msun": diagnostics.m2_post_cut_msun,
        },
    )
    write_bulk_diagnostic_artifacts(diagnostics, artifact, config)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest


def parameterset_from_sed_summary(
    doc: Mapping[str, Any],
    *,
    provenance: str = "uberMS",
) -> ParameterSet | None:
    """Map sed_summary.json luminous M1 (+ optional radius) into a ``ParameterSet``."""
    block = doc.get("m1_msun")
    if not isinstance(block, dict):
        fits = doc.get("fits") or {}
        ums = fits.get("ums") or {}
        block = ums.get("m1_msun") or (ums.get("parameters") or {}).get("initial_Mass")
    if not isinstance(block, dict):
        return None
    median = _finite(block.get("median"))
    if median is None or median <= 0:
        return None
    p16 = _finite(block.get("p16"))
    p84 = _finite(block.get("p84"))
    if p16 is not None and p84 is not None:
        sigma = 0.5 * abs(p84 - p16)
    else:
        sigma = _finite(block.get("std")) or 0.0

    names = ["M1"]
    values = [median]
    variances = [sigma**2]
    units = ["Msun"]

    radius_block = None
    fits = doc.get("fits") or {}
    ums = fits.get("ums") or {}
    params = ums.get("parameters") or {}
    if isinstance(params.get("log(R)"), dict):
        radius_block = params["log(R)"]
    if isinstance(radius_block, dict):
        log_r = _finite(radius_block.get("median"))
        if log_r is not None:
            r_med = 10.0**log_r
            r_p16 = _finite(radius_block.get("p16"))
            r_p84 = _finite(radius_block.get("p84"))
            if r_p16 is not None and r_p84 is not None:
                r_sigma = 0.5 * abs(10.0**r_p84 - 10.0**r_p16)
            else:
                r_sigma = 0.0
            names.append("R1")
            values.append(r_med)
            variances.append(r_sigma**2)
            units.append("Rsun")

    n = len(names)
    cov = [[0.0] * n for _ in range(n)]
    for i, var in enumerate(variances):
        cov[i][i] = var
    return ParameterSet(
        names=names,
        values=values,
        covariance=cov,
        provenance=provenance,
        units=units,
    )


def run_refined_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
    *,
    summary_loader: Callable[[int], dict[str, Any] | None] | None = None,
    needs_update_fn: Callable[[int], tuple[bool, str]] | None = None,
    fit_fn: Callable[[int], dict[str, Any] | None] | None = None,
) -> tuple[list[CandidateRecord], RefinedDiagnostics]:
    """Queue and apply uberMS refined M1; prioritize by information-gain stub."""
    if config.mass_derivation.require_sed_package and not _SED_AVAILABLE:
        raise ImportError(
            "mass_derivation.require_sed_package=true but darkhunter_sed "
            "is not importable"
        )

    loader = summary_loader or _default_sed_summary_loader
    needs_update = needs_update_fn or _default_sed_needs_update
    fit = fit_fn or _default_sed_fit

    ordered = sorted(
        candidates,
        key=lambda c: information_gain_stub(c, config),
        reverse=True,
    )
    max_stars = config.mass_derivation.sed_queue_max_stars
    if max_stars is not None:
        ordered = ordered[:max_stars]

    order_ids = tuple(c.source_id for c in ordered)
    updated: list[CandidateRecord] = []
    watchlist: list[int] = []
    fit_attempted = 0
    fit_cached = 0
    fit_succeeded = 0
    fit_failed = 0

    for candidate in ordered:
        source_id = candidate.source_id
        should_run, reason = needs_update(source_id)
        doc: dict[str, Any] | None = None
        if should_run:
            fit_attempted += 1
            doc = fit(source_id)
            if doc is None:
                fit_failed += 1
        else:
            if reason == "up to date":
                fit_cached += 1
            doc = loader(source_id)

        if doc is None:
            # Keep bulk estimate; still emit watch-list from bulk M1 if near cap.
            m1_val = None
            if candidate.m1 is not None:
                try:
                    m1_val = candidate.m1.marginal("M1").value
                except KeyError:
                    m1_val = None
            if m1_val is not None and approaches_uberms_m1_prior_cap(m1_val, config):
                watchlist.append(source_id)
            updated.append(candidate)
            continue

        m1_set = parameterset_from_sed_summary(doc)
        if m1_set is None:
            fit_failed += 1
            updated.append(candidate)
            continue

        fit_succeeded += 1
        m1_val = m1_set.marginal("M1").value
        if approaches_uberms_m1_prior_cap(m1_val, config):
            watchlist.append(source_id)

        extras = dict(candidate.extras)
        extras["sed_summary"] = {
            "gaia_source_id": doc.get("gaia_source_id", str(source_id)),
            "m1_msun": doc.get("m1_msun"),
        }
        extras["uberms_m1_watchlist"] = approaches_uberms_m1_prior_cap(
            m1_val, config
        )
        updated.append(
            candidate.model_copy(
                update={
                    "m1": m1_set,
                    "fit_tier": FitTier.FULL_UBERMS,
                    "extras": extras,
                }
            )
        )

    diagnostics = RefinedDiagnostics(
        queued=len(ordered),
        fit_attempted=fit_attempted,
        fit_cached=fit_cached,
        fit_succeeded=fit_succeeded,
        fit_failed=fit_failed,
        watchlist_source_ids=tuple(watchlist),
        information_gain_order=order_ids,
    )
    return updated, diagnostics


def write_refined_diagnostic_artifacts(
    diagnostics: RefinedDiagnostics,
    artifact_path: Path,
) -> list[Path]:
    out_dir = artifact_path.parent / f"{artifact_path.stem}_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "refined_report.txt"
    report.write_text(format_refined_report(diagnostics), encoding="utf-8")
    watch = out_dir / "uberms_m1_watchlist.txt"
    watch.write_text(
        "\n".join(str(sid) for sid in diagnostics.watchlist_source_ids) + ("\n" if diagnostics.watchlist_source_ids else ""),
        encoding="utf-8",
    )
    return [report, watch]


def run_mass_derivation_refined(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
    summary_loader: Callable[[int], dict[str, Any] | None] | None = None,
    needs_update_fn: Callable[[int], tuple[bool, str]] | None = None,
    fit_fn: Callable[[int], dict[str, Any] | None] | None = None,
) -> RunManifest:
    """Execute ``mass_derivation_refined``: uberMS queue → HDF5 + watch-list."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["mass_derivation_refined"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    if candidates is None:
        candidates = _load_upstream_candidates(manifest, "mass_derivation_bulk")

    updated, diagnostics = run_refined_on_candidates(
        candidates,
        config,
        summary_loader=summary_loader,
        needs_update_fn=needs_update_fn,
        fit_fn=fit_fn,
    )
    write_stage_hdf5(
        artifact,
        updated,
        stage_name="mass_derivation_refined",
        diagnostics={
            "queued": diagnostics.queued,
            "fit_attempted": diagnostics.fit_attempted,
            "fit_cached": diagnostics.fit_cached,
            "fit_succeeded": diagnostics.fit_succeeded,
            "fit_failed": diagnostics.fit_failed,
            "watchlist_source_ids": list(diagnostics.watchlist_source_ids),
            "information_gain_order": list(diagnostics.information_gain_order),
        },
    )
    write_refined_diagnostic_artifacts(diagnostics, artifact)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
