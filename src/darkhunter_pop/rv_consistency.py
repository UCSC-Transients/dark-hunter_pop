"""Stages: ``rv_astrometry_gate`` and ``joint_orbit_fit``.

Consumes ``dark-hunter_rv`` JSON summaries into ``CandidateRecord.rv_summary``
(ARCHITECTURE.md §4; ``docs/RV_SUMMARY_JSON`` in dark-hunter_rv).

Gate: hold astrometric P, e, T_periastron, K, ω fixed; fit γ + jitter per instrument;
score whole-curve chi2/dof vs ``rv_consistency.chi2_dof_threshold``. SB2 orbits that
disagree with astrometry fail the gate (outlier path); consistent SB2 unlocks the
mass-ratio channel for ``companion_nature_likelihood``.

Joint fit: separate registered stage after the gate. Passers get free orbital elements
seeded by The Joker (when present) with soft NSS priors + RV likelihood; ``OrbitTier``
becomes ``joint_astrometry_rv``. Failures keep ``astrometry_only`` with skip reason
``rv_astrometry_gate_failed``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar

from darkhunter_pop import constants
from darkhunter_pop.config_loader import require_dr3_active_for_v1
from darkhunter_pop.config_schema import PipelineConfig, RvConsistencyConfig
from darkhunter_pop.diagnostics import (
    emit_gate_pass_rate,
    resolve_diagnostic_dirs,
)
from darkhunter_pop.mass_derivation import read_stage_hdf5 as read_mass_stage_hdf5
from darkhunter_pop.mass_derivation import write_stage_hdf5 as write_mass_stage_hdf5
from darkhunter_pop.plotting import plot_histogram
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    InstrumentNuisance,
    OrbitTier,
    OutlierTestResult,
    ParameterSet,
    RunManifest,
    StageStatus,
)

JOINT_ORBIT_SKIP_REASON = "rv_astrometry_gate_failed"
JOINT_PROVENANCE = "joint_astrometry_rv"
ORBIT_ELEMENT_NAMES = ("P", "e", "T_periastron", "K", "omega")


# ---------------------------------------------------------------------------
# Orbital / RV primitives
# ---------------------------------------------------------------------------


def t_periastron_to_mjd(t_periastron: float) -> float:
    """Gaia NSS ``t_periastron`` is days from J2016.0; already-MJD values pass through."""
    t = float(t_periastron)
    if t > 40000.0:
        return t
    return float(constants.GAIA_J2016_MJD + t)


def spectroscopic_mass_function_msun(
    period_day: float, k_kms: float, eccentricity: float
) -> float:
    """Binary mass function f(M) in solar masses (P days, K km/s)."""
    if period_day <= 0 or k_kms <= 0 or not np.isfinite(eccentricity):
        return float("nan")
    ecc = float(np.clip(eccentricity, 0.0, 0.999))
    one_minus_e2 = 1.0 - ecc * ecc
    if one_minus_e2 <= 0.0:
        return float("nan")
    val = (
        constants.SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS
        * (k_kms**3)
        * period_day
        * (one_minus_e2**1.5)
    )
    return float(val) if np.isfinite(val) else float("nan")


def solve_m2_with_inclination_msun(
    f_mass: float, m1: float, inclination_deg: float
) -> float | None:
    """Solve (M2^3 sin^3 i) / (M1 + M2)^2 = f for M2 > 0."""
    if not np.isfinite(f_mass) or not np.isfinite(m1) or not np.isfinite(inclination_deg):
        return None
    if f_mass <= 0 or m1 <= 0:
        return None
    sin_i = float(np.sin(np.deg2rad(float(inclination_deg))))
    if not np.isfinite(sin_i) or abs(sin_i) < 1e-6:
        return None
    s3 = abs(sin_i) ** 3
    lo, hi = 1e-8, 500.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = (mid**3) * s3 / ((m1 + mid) ** 2) - f_mass
        if g > 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def predicted_k_kms(
    m1_msun: float,
    m2_msun: float,
    period_day: float,
    eccentricity: float,
    inclination_deg: float,
) -> float | None:
    """RV semi-amplitude from component masses + inclination (km/s)."""
    if min(m1_msun, m2_msun, period_day) <= 0:
        return None
    if not np.isfinite(inclination_deg):
        return None
    sin_i = abs(float(np.sin(np.deg2rad(inclination_deg))))
    if sin_i < 1e-6:
        return None
    ecc = float(np.clip(eccentricity, 0.0, 0.999))
    total = m1_msun + m2_msun
    f_mass = (m2_msun * sin_i) ** 3 / (total**2)
    one_minus_e2 = 1.0 - ecc * ecc
    if one_minus_e2 <= 0:
        return None
    denom = (
        constants.SPECTROSCOPIC_MASS_FUNCTION_DAY_KMS
        * period_day
        * (one_minus_e2**1.5)
    )
    if denom <= 0:
        return None
    k3 = f_mass / denom
    if k3 <= 0:
        return None
    return float(k3 ** (1.0 / 3.0))


def _solve_kepler(
    mean_anomaly: NDArray[np.floating], eccentricity: float
) -> NDArray[np.floating]:
    ecc = float(np.clip(eccentricity, 0.0, 0.999))
    m = np.asarray(mean_anomaly, dtype=np.float64)
    e_anom = np.array(m, dtype=np.float64, copy=True)
    for _ in range(30):
        f = e_anom - ecc * np.sin(e_anom) - m
        fp = 1.0 - ecc * np.cos(e_anom)
        e_anom -= f / np.clip(fp, 1e-10, None)
    return e_anom


def rv_curve_kms(
    t_mjd: NDArray[np.floating],
    *,
    period_day: float,
    eccentricity: float,
    t_periastron_mjd: float,
    k_kms: float,
    omega_rad: float,
    gamma_kms: float = 0.0,
) -> NDArray[np.floating]:
    """Keplerian RV model (km/s) at observation times."""
    t = np.asarray(t_mjd, dtype=np.float64)
    p = float(period_day)
    e = float(np.clip(eccentricity, 1e-8, 0.999))
    n = 2.0 * np.pi / p
    mean_anom = n * (t - float(t_periastron_mjd))
    e_anom = _solve_kepler(mean_anom, e)
    cos_e = np.cos(e_anom)
    sin_e = np.sin(e_anom)
    cos_f = (cos_e - e) / (1.0 - e * cos_e)
    sin_f = (np.sqrt(1.0 - e * e) * sin_e) / (1.0 - e * cos_e)
    true_anom = np.arctan2(sin_f, cos_f)
    return gamma_kms + k_kms * (
        np.cos(true_anom + omega_rad) + e * np.cos(omega_rad)
    )


def _profile_jitter_kms(
    residual: NDArray[np.floating],
    sigma_formal: NDArray[np.floating],
    jitter_max_kms: float,
) -> float:
    """Homoscedastic jitter maximizing Gaussian likelihood at fixed residuals."""
    resid2 = np.asarray(residual, dtype=np.float64) ** 2
    e2 = np.clip(np.asarray(sigma_formal, dtype=np.float64), 1e-4, None) ** 2

    def nll(log_s: float) -> float:
        s2 = float(np.exp(2.0 * log_s))
        sig2 = e2 + s2
        return float(np.sum(resid2 / sig2 + np.log(sig2)))

    result = minimize_scalar(
        nll,
        bounds=(math.log(1e-4), math.log(max(1e-4, float(jitter_max_kms)))),
        method="bounded",
    )
    return float(np.exp(result.x))


def _weighted_mean(
    values: NDArray[np.floating], weights: NDArray[np.floating]
) -> float:
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if not np.any(w > 0):
        return float(np.mean(y))
    return float(np.sum(w * y) / np.sum(w))


# ---------------------------------------------------------------------------
# RV summary / orbital element extraction
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        xf = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf):
        return None
    return xf


def _first_finite(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in mapping:
            val = _finite(mapping[key])
            if val is not None:
                return val
    return None


@dataclass(frozen=True)
class AstrometricOrbit:
    """Fixed orbital elements for the RV/astrometry gate."""

    period_day: float
    eccentricity: float
    t_periastron_mjd: float
    k_kms: float
    omega_rad: float
    inclination_deg: float | None = None


@dataclass(frozen=True)
class RvEpoch:
    mjd: float
    rv_kms: float
    rv_err_kms: float
    instrument: str


def collect_rv_epochs(rv_summary: Mapping[str, Any]) -> list[RvEpoch]:
    """Flatten pipeline + external epochs from dark-hunter_rv summary JSON."""
    epochs: list[RvEpoch] = []
    for row in rv_summary.get("pipeline_epochs") or []:
        if not isinstance(row, Mapping):
            continue
        mjd = _finite(row.get("mjd"))
        rv = _finite(row.get("rv_kms"))
        err = _finite(row.get("rv_err_kms"))
        tel = str(row.get("telescope") or row.get("instrument") or "UNKNOWN")
        if mjd is None or rv is None or err is None or err <= 0:
            continue
        epochs.append(
            RvEpoch(mjd=mjd, rv_kms=rv, rv_err_kms=err, instrument=tel)
        )
    for row in rv_summary.get("external_rvs") or []:
        if not isinstance(row, Mapping):
            continue
        mjd = _finite(row.get("mjd"))
        rv = _finite(row.get("rv_kms", row.get("rv")))
        err = _finite(row.get("rv_err_kms", row.get("rv_err")))
        tel = str(row.get("telescope") or row.get("instrument") or "LITERATURE")
        if mjd is None or rv is None or err is None or err <= 0:
            continue
        epochs.append(
            RvEpoch(mjd=mjd, rv_kms=rv, rv_err_kms=err, instrument=tel)
        )
    return epochs


def _nss_blocks(candidate: CandidateRecord) -> dict[str, Any]:
    """Merge candidate.nss_orbital with rv_summary.nss_orbital (summary wins)."""
    merged: dict[str, Any] = dict(candidate.nss_orbital or {})
    summary = candidate.rv_summary or {}
    block = summary.get("nss_orbital")
    if isinstance(block, Mapping):
        merged.update(dict(block))
    return merged


def _inclination_deg(
    candidate: CandidateRecord, nss: Mapping[str, Any]
) -> float | None:
    inc = _first_finite(nss, ("inclination_deg", "inclination", "Inclination"))
    if inc is not None:
        return inc
    summary = candidate.rv_summary or {}
    joker = summary.get("joker_fit")
    if isinstance(joker, Mapping):
        inc = _finite(joker.get("inclination_deg"))
        if inc is not None:
            return inc
    return None


def _omega_from_joker_block(block: Mapping[str, Any]) -> float | None:
    om = _finite(block.get("omega_rad"))
    if om is not None:
        return om
    om_deg = _finite(block.get("omega_deg"))
    if om_deg is not None:
        return float(np.deg2rad(om_deg))
    return None


def _iter_joker_blocks(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    joker = summary.get("joker_fit")
    if not isinstance(joker, Mapping):
        return []
    blocks: list[Mapping[str, Any]] = []
    if isinstance(joker.get("variants"), Mapping):
        for name in ("full", "ecc", "period", "rv_only"):
            block = joker["variants"].get(name)
            if isinstance(block, Mapping):
                blocks.append(block)
    for name in ("full", "ecc", "period", "rv_only"):
        block = joker.get(name)
        if isinstance(block, Mapping):
            blocks.append(block)
    blocks.append(joker)
    return blocks


def _omega_rad(nss: Mapping[str, Any], summary: Mapping[str, Any]) -> float | None:
    if "arg_periastron_deg" in nss or "omega_deg" in nss or "Arg_Periastron" in nss:
        deg = _first_finite(
            nss, ("arg_periastron_deg", "omega_deg", "Arg_Periastron")
        )
        if deg is not None:
            return float(np.deg2rad(deg))
    deg = _first_finite(nss, ("omega",))
    if deg is not None:
        if abs(deg) > 2.0 * math.pi + 0.5:
            return float(np.deg2rad(deg))
        return float(deg)
    for block in _iter_joker_blocks(summary):
        om = _omega_from_joker_block(block)
        if om is not None:
            return om
    return None


def _k_from_joker(summary: Mapping[str, Any]) -> float | None:
    for block in _iter_joker_blocks(summary):
        if block.get("skip_reason") not in (None, ""):
            continue
        jk = _finite(block.get("K_kms"))
        if jk is not None and jk > 0:
            return abs(jk)
    return None


def _k_kms(
    candidate: CandidateRecord,
    nss: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    period_day: float,
    eccentricity: float,
    inclination_deg: float | None,
) -> float | None:
    k = _first_finite(
        nss,
        (
            "semi_amp_primary_kms",
            "K_kms",
            "k_kms",
            "Semi_Amp_Primary",
            "radial_velocity_amplitude",
        ),
    )
    if k is not None and k > 0:
        return abs(k)
    jk = _k_from_joker(summary)
    if jk is not None:
        return jk
    if (
        candidate.m1 is not None
        and candidate.m2 is not None
        and inclination_deg is not None
    ):
        try:
            m1 = candidate.m1.marginal("M1").value
            m2 = candidate.m2.marginal("M2").value
        except KeyError:
            return None
        return predicted_k_kms(
            m1, m2, period_day, eccentricity, inclination_deg
        )
    return None


def extract_astrometric_orbit(candidate: CandidateRecord) -> AstrometricOrbit | None:
    """Resolve fixed gate elements from NSS / rv_summary / predicted K."""
    nss = _nss_blocks(candidate)
    summary = candidate.rv_summary or {}
    period = _first_finite(nss, ("period_day", "period", "Period", "P_days"))
    ecc = _first_finite(nss, ("eccentricity", "Eccentricity", "e"))
    t_raw = _first_finite(
        nss,
        ("t_periastron_day", "t_periastron", "T_Periastron", "t_periastron_mjd"),
    )
    if period is None or period <= 0 or ecc is None or t_raw is None:
        return None
    t_mjd = t_periastron_to_mjd(t_raw)
    inc = _inclination_deg(candidate, nss)
    omega = _omega_rad(nss, summary)
    if omega is None:
        omega = 0.0
    k = _k_kms(
        candidate,
        nss,
        summary,
        period_day=period,
        eccentricity=ecc,
        inclination_deg=inc,
    )
    if k is None or k <= 0:
        return None
    return AstrometricOrbit(
        period_day=period,
        eccentricity=float(np.clip(ecc, 0.0, 0.999)),
        t_periastron_mjd=t_mjd,
        k_kms=float(k),
        omega_rad=float(omega),
        inclination_deg=inc,
    )


def is_sb2_candidate(candidate: CandidateRecord) -> bool:
    """Detect SB2 from NSS solution type / dual semi-amplitudes / summary flags."""
    sol = (candidate.nss_solution_type or "").upper()
    if "SB2" in sol:
        return True
    summary = candidate.rv_summary or {}
    if summary.get("sb2") or summary.get("is_sb2"):
        return True
    nss = _nss_blocks(candidate)
    k1 = _finite(nss.get("semi_amp_primary_kms"))
    k2 = _finite(nss.get("semi_amp_secondary_kms"))
    if k1 is not None and k2 is not None and k1 > 0 and k2 > 0:
        return True
    if isinstance(summary.get("sb2_orbit"), Mapping):
        return True
    return False


def sb2_orbit_consistent(
    candidate: CandidateRecord,
    orbit: AstrometricOrbit,
    config: RvConsistencyConfig,
) -> tuple[bool, str | None]:
    """Compare optional SB2 spectroscopic orbit to the astrometric solution."""
    summary = candidate.rv_summary or {}
    sb2 = summary.get("sb2_orbit")
    if not isinstance(sb2, Mapping):
        return True, None
    p_sb2 = _first_finite(sb2, ("period_day", "period", "P_days"))
    e_sb2 = _first_finite(sb2, ("eccentricity", "e"))
    notes: list[str] = []
    ok = True
    if p_sb2 is not None and p_sb2 > 0:
        frac = abs(p_sb2 - orbit.period_day) / orbit.period_day
        if frac > config.sb2_period_frac_tol:
            ok = False
            notes.append(f"sb2_period_frac={frac:.4f}")
    if e_sb2 is not None:
        de = abs(e_sb2 - orbit.eccentricity)
        if de > config.sb2_ecc_abs_tol:
            ok = False
            notes.append(f"sb2_ecc_abs={de:.4f}")
    return ok, ("; ".join(notes) if notes else None)


def sb2_mass_ratio(candidate: CandidateRecord) -> float | None:
    """Direct mass ratio from SB2 semi-amplitudes or NSS mass_ratio field."""
    nss = _nss_blocks(candidate)
    q = _finite(nss.get("mass_ratio"))
    if q is not None and q > 0:
        return q
    k1 = _finite(nss.get("semi_amp_primary_kms"))
    k2 = _finite(nss.get("semi_amp_secondary_kms"))
    if k1 is not None and k2 is not None and k1 > 0 and k2 > 0:
        return float(k1 / k2)
    summary = candidate.rv_summary or {}
    sb2 = summary.get("sb2_orbit")
    if isinstance(sb2, Mapping):
        q = _finite(sb2.get("mass_ratio"))
        if q is not None and q > 0:
            return q
        k1 = _finite(sb2.get("semi_amp_primary_kms", sb2.get("K1_kms")))
        k2 = _finite(sb2.get("semi_amp_secondary_kms", sb2.get("K2_kms")))
        if k1 is not None and k2 is not None and k1 > 0 and k2 > 0:
            return float(k1 / k2)
    return None


# ---------------------------------------------------------------------------
# Gate fit
# ---------------------------------------------------------------------------


@dataclass
class InstrumentGateFit:
    instrument: str
    gamma_kms: float
    jitter_kms: float
    chi2: float
    n_points: int


@dataclass
class GateDiagnostics:
    n_input: int = 0
    n_scored: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_skipped_no_rv: int = 0
    n_skipped_elements: int = 0
    n_sb2: int = 0
    n_sb2_mass_ratio_unlocked: int = 0
    chi2_dof_values: list[float] = field(default_factory=list)
    passed_source_ids: list[int] = field(default_factory=list)
    failed_source_ids: list[int] = field(default_factory=list)

    def counts_for_hook(self) -> dict[str, int]:
        return {
            "passed": self.n_passed,
            "failed": self.n_failed,
            "skipped": self.n_skipped_no_rv + self.n_skipped_elements,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_scored": self.n_scored,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_skipped_no_rv": self.n_skipped_no_rv,
            "n_skipped_elements": self.n_skipped_elements,
            "n_sb2": self.n_sb2,
            "n_sb2_mass_ratio_unlocked": self.n_sb2_mass_ratio_unlocked,
            "chi2_dof_values": self.chi2_dof_values,
            "passed_source_ids": self.passed_source_ids,
            "failed_source_ids": self.failed_source_ids,
        }


def fit_instrument_nuisance(
    epochs: Sequence[RvEpoch],
    orbit: AstrometricOrbit,
    *,
    jitter_max_kms: float,
) -> InstrumentGateFit:
    """Fit γ + jitter at fixed orbital elements for one instrument."""
    t = np.array([e.mjd for e in epochs], dtype=np.float64)
    y = np.array([e.rv_kms for e in epochs], dtype=np.float64)
    yerr = np.array([e.rv_err_kms for e in epochs], dtype=np.float64)
    model0 = rv_curve_kms(
        t,
        period_day=orbit.period_day,
        eccentricity=orbit.eccentricity,
        t_periastron_mjd=orbit.t_periastron_mjd,
        k_kms=orbit.k_kms,
        omega_rad=orbit.omega_rad,
        gamma_kms=0.0,
    )
    gamma = _weighted_mean(y - model0, 1.0 / np.clip(yerr, 1e-4, None) ** 2)
    resid = y - (model0 + gamma)
    jitter = _profile_jitter_kms(resid, yerr, jitter_max_kms)
    sigma = np.sqrt(yerr**2 + jitter**2)
    chi2 = float(np.sum((resid / sigma) ** 2))
    return InstrumentGateFit(
        instrument=epochs[0].instrument,
        gamma_kms=gamma,
        jitter_kms=jitter,
        chi2=chi2,
        n_points=len(epochs),
    )


def run_gate_on_candidate(
    candidate: CandidateRecord,
    config: RvConsistencyConfig,
) -> tuple[CandidateRecord, OutlierTestResult | None, str | None]:
    """Score one candidate; return updated record, result (or None), skip note."""
    epochs = collect_rv_epochs(candidate.rv_summary or {})
    if len(epochs) < config.min_epochs_total:
        extras = dict(candidate.extras)
        extras["rv_astrometry_gate"] = {
            "skipped": True,
            "reason": "insufficient_rv_epochs",
            "n_epochs": len(epochs),
        }
        updated = candidate.model_copy(
            update={
                "orbit_tier": candidate.orbit_tier or OrbitTier.ASTROMETRY_ONLY,
                "extras": extras,
            }
        )
        return updated, None, "insufficient_rv_epochs"

    orbit = extract_astrometric_orbit(candidate)
    if orbit is None:
        extras = dict(candidate.extras)
        extras["rv_astrometry_gate"] = {
            "skipped": True,
            "reason": "missing_astrometric_elements",
        }
        updated = candidate.model_copy(
            update={
                "orbit_tier": candidate.orbit_tier or OrbitTier.ASTROMETRY_ONLY,
                "extras": extras,
            }
        )
        return updated, None, "missing_astrometric_elements"

    by_inst: dict[str, list[RvEpoch]] = {}
    for ep in epochs:
        by_inst.setdefault(ep.instrument, []).append(ep)

    fits: list[InstrumentGateFit] = []
    for _inst, rows in sorted(by_inst.items()):
        if len(rows) < config.min_epochs_per_instrument:
            continue
        fits.append(
            fit_instrument_nuisance(
                rows, orbit, jitter_max_kms=config.jitter_max_kms
            )
        )
    if not fits:
        extras = dict(candidate.extras)
        extras["rv_astrometry_gate"] = {
            "skipped": True,
            "reason": "no_instrument_above_min_epochs",
        }
        updated = candidate.model_copy(
            update={
                "orbit_tier": candidate.orbit_tier or OrbitTier.ASTROMETRY_ONLY,
                "extras": extras,
            }
        )
        return updated, None, "no_instrument_above_min_epochs"

    n_points = sum(f.n_points for f in fits)
    n_params = 2 * len(fits)
    dof = max(n_points - n_params, 1)
    chi2 = sum(f.chi2 for f in fits)
    chi2_dof = float(chi2 / dof)

    sb2 = is_sb2_candidate(candidate)
    sb2_ok, sb2_note = (True, None)
    if sb2:
        sb2_ok, sb2_note = sb2_orbit_consistent(candidate, orbit, config)

    passed = chi2_dof <= config.chi2_dof_threshold and sb2_ok
    notes_parts: list[str] = []
    if chi2_dof > config.chi2_dof_threshold:
        notes_parts.append(
            f"chi2_dof={chi2_dof:.4f}>{config.chi2_dof_threshold}"
        )
    if sb2 and not sb2_ok:
        notes_parts.append(f"sb2_inconsistent({sb2_note})")
    elif sb2 and sb2_ok:
        notes_parts.append("sb2_consistent")

    result = OutlierTestResult(
        source_id=candidate.source_id,
        chi2_dof=chi2_dof,
        threshold=config.chi2_dof_threshold,
        passed=passed,
        instruments=[
            InstrumentNuisance(
                instrument=f.instrument,
                gamma_kms=f.gamma_kms,
                jitter_kms=f.jitter_kms,
            )
            for f in fits
        ],
        notes="; ".join(notes_parts) if notes_parts else None,
    )

    extras = dict(candidate.extras)
    extras["rv_astrometry_gate"] = result.model_dump(mode="json")
    extras["rv_astrometry_gate_orbit"] = {
        "period_day": orbit.period_day,
        "eccentricity": orbit.eccentricity,
        "t_periastron_mjd": orbit.t_periastron_mjd,
        "k_kms": orbit.k_kms,
        "omega_rad": orbit.omega_rad,
        "inclination_deg": orbit.inclination_deg,
    }
    if sb2 and passed:
        q = sb2_mass_ratio(candidate)
        extras["sb2_mass_ratio_channel"] = True
        if q is not None:
            extras["sb2_mass_ratio"] = q
    elif sb2:
        extras["sb2_mass_ratio_channel"] = False

    updated = candidate.model_copy(
        update={
            "orbit_tier": OrbitTier.ASTROMETRY_ONLY,
            "extras": extras,
        }
    )
    return updated, result, None


def run_gate_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> tuple[list[CandidateRecord], GateDiagnostics]:
    """Apply ``rv_astrometry_gate`` to an in-memory candidate list."""
    rc = config.rv_consistency
    diag = GateDiagnostics(n_input=len(candidates))
    out: list[CandidateRecord] = []
    for candidate in candidates:
        updated, result, skip = run_gate_on_candidate(candidate, rc)
        if is_sb2_candidate(candidate):
            diag.n_sb2 += 1
        if skip == "insufficient_rv_epochs":
            diag.n_skipped_no_rv += 1
        elif skip is not None:
            diag.n_skipped_elements += 1
        elif result is not None:
            diag.n_scored += 1
            diag.chi2_dof_values.append(result.chi2_dof)
            if result.passed:
                diag.n_passed += 1
                diag.passed_source_ids.append(candidate.source_id)
                if updated.extras.get("sb2_mass_ratio_channel"):
                    diag.n_sb2_mass_ratio_unlocked += 1
            else:
                diag.n_failed += 1
                diag.failed_source_ids.append(candidate.source_id)
        out.append(updated)
    return out, diag


# ---------------------------------------------------------------------------
# Joint orbit fit
# ---------------------------------------------------------------------------


@dataclass
class JointFitDiagnostics:
    n_input: int = 0
    n_fit: int = 0
    n_skipped_gate_failed: int = 0
    n_skipped_other: int = 0
    n_failed_optimize: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_fit": self.n_fit,
            "n_skipped_gate_failed": self.n_skipped_gate_failed,
            "n_skipped_other": self.n_skipped_other,
            "n_failed_optimize": self.n_failed_optimize,
        }


def _joker_seed_block(
    summary: Mapping[str, Any], variant: str
) -> Mapping[str, Any] | None:
    for block in _iter_joker_blocks(summary):
        if block.get("skip_reason") not in (None, ""):
            continue
        # Prefer configured variant when the block advertises fit_variant.
        fv = block.get("fit_variant")
        if fv is not None and fv != variant and _finite(block.get("P_days")) is None:
            continue
        if _finite(block.get("P_days")) is not None or _finite(block.get("K_kms")) is not None:
            if fv == variant or fv is None:
                return block
    for block in _iter_joker_blocks(summary):
        if block.get("skip_reason") not in (None, "") and _finite(block.get("P_days")) is None:
            continue
        if _finite(block.get("P_days")) is not None:
            return block
    return None


def _gate_passed(candidate: CandidateRecord) -> bool:
    gate = candidate.extras.get("rv_astrometry_gate")
    if not isinstance(gate, Mapping):
        return False
    if gate.get("skipped"):
        return False
    return bool(gate.get("passed"))


def _orbit_parameterset(
    *,
    period_day: float,
    eccentricity: float,
    t_periastron_mjd: float,
    k_kms: float,
    omega_rad: float,
    variances: Sequence[float],
    provenance: str,
) -> ParameterSet:
    cov = [[0.0] * 5 for _ in range(5)]
    for i, var in enumerate(variances):
        cov[i][i] = float(max(var, 0.0))
    return ParameterSet(
        names=list(ORBIT_ELEMENT_NAMES),
        values=[period_day, eccentricity, t_periastron_mjd, k_kms, omega_rad],
        covariance=cov,
        provenance=provenance,
        units=["day", "1", "MJD", "km/s", "rad"],
    )


def _joint_refine_m2(
    candidate: CandidateRecord,
    *,
    period_day: float,
    k_kms: float,
    eccentricity: float,
    inclination_deg: float | None,
) -> ParameterSet | None:
    if candidate.m1 is None or inclination_deg is None:
        return None
    try:
        m1 = candidate.m1.marginal("M1").value
        m1_sigma = candidate.m1.marginal("M1").sigma or 0.0
    except KeyError:
        return None
    f_mass = spectroscopic_mass_function_msun(period_day, k_kms, eccentricity)
    if not np.isfinite(f_mass) or f_mass <= 0:
        return None
    m2 = solve_m2_with_inclination_msun(f_mass, m1, inclination_deg)
    if m2 is None:
        return None
    step = max(m1_sigma, 1e-3)
    m2_hi = solve_m2_with_inclination_msun(f_mass, m1 + step, inclination_deg)
    m2_lo = solve_m2_with_inclination_msun(f_mass, m1 - step, inclination_deg)
    if m2_hi is not None and m2_lo is not None and m1_sigma > 0:
        dm2 = 0.5 * abs(m2_hi - m2_lo)
    else:
        dm2 = 0.1 * m2
    return ParameterSet(
        names=["M2"],
        values=[m2],
        covariance=[[float(dm2**2)]],
        provenance=JOINT_PROVENANCE,
        units=["Msun"],
    )


def fit_joint_orbit(
    candidate: CandidateRecord,
    config: RvConsistencyConfig,
) -> CandidateRecord:
    """Free-element RV fit with Joker/NSS seed + soft astrometric priors."""
    epochs = collect_rv_epochs(candidate.rv_summary or {})
    orbit0 = extract_astrometric_orbit(candidate)
    if orbit0 is None or len(epochs) < config.min_epochs_total:
        extras = dict(candidate.extras)
        extras["joint_orbit_fit_skip_reason"] = "insufficient_data"
        return candidate.model_copy(update={"extras": extras})

    by_inst: dict[str, list[RvEpoch]] = {}
    for ep in epochs:
        by_inst.setdefault(ep.instrument, []).append(ep)
    instruments = [
        inst
        for inst, rows in sorted(by_inst.items())
        if len(rows) >= config.min_epochs_per_instrument
    ]
    if not instruments:
        extras = dict(candidate.extras)
        extras["joint_orbit_fit_skip_reason"] = "no_instrument_above_min_epochs"
        return candidate.model_copy(update={"extras": extras})

    seed = _joker_seed_block(candidate.rv_summary or {}, config.joker_seed_variant)
    period = orbit0.period_day
    ecc = orbit0.eccentricity
    t_peri = orbit0.t_periastron_mjd
    k = orbit0.k_kms
    omega = orbit0.omega_rad
    if seed is not None:
        period = _finite(seed.get("P_days")) or period
        ecc = _finite(seed.get("e")) or ecc
        k = abs(_finite(seed.get("K_kms")) or k)
        om_seed = _omega_from_joker_block(seed)
        if om_seed is not None:
            omega = om_seed
        if _finite(seed.get("t_periastron_mjd")) is not None:
            t_peri = float(seed["t_periastron_mjd"])

    gate = candidate.extras.get("rv_astrometry_gate")
    gamma0: dict[str, float] = {}
    jitter0: dict[str, float] = {}
    if isinstance(gate, Mapping):
        for row in gate.get("instruments") or []:
            if isinstance(row, Mapping):
                gamma0[str(row.get("instrument"))] = float(row.get("gamma_kms", 0.0))
                jitter0[str(row.get("instrument"))] = float(
                    row.get("jitter_kms", 0.0)
                )

    e_clip = float(np.clip(ecc, 1e-4, 0.95))
    x0 = [
        math.log(max(period, 1e-3)),
        max(k, 1e-3),
        e_clip * math.cos(omega),
        e_clip * math.sin(omega),
        t_peri,
    ]
    for inst in instruments:
        rows = by_inst[inst]
        g = gamma0.get(inst)
        if g is None:
            g = float(np.median([r.rv_kms for r in rows]))
        x0.append(g)
        jit = max(jitter0.get(inst, 0.01), 1e-4)
        x0.append(math.log(jit))

    t_all = np.array(
        [e.mjd for e in epochs if e.instrument in instruments], dtype=np.float64
    )
    y_all = np.array(
        [e.rv_kms for e in epochs if e.instrument in instruments], dtype=np.float64
    )
    yerr_all = np.array(
        [e.rv_err_kms for e in epochs if e.instrument in instruments],
        dtype=np.float64,
    )
    inst_idx = np.array(
        [
            instruments.index(e.instrument)
            for e in epochs
            if e.instrument in instruments
        ],
        dtype=np.int64,
    )

    p0, e0, t0, om0 = period, e_clip, t_peri, omega
    sig_p = max(config.joint_prior_period_frac * p0, 1e-3)
    sig_e = config.joint_prior_ecc_abs
    sig_om = config.joint_prior_omega_rad
    sig_t = config.joint_prior_t_peri_day

    def pack_orbit(
        theta: NDArray[np.floating],
    ) -> tuple[float, float, float, float, float]:
        log_p, k_val, h, kk, t_val = theta[:5]
        p = float(np.exp(log_p))
        e = float(np.clip(np.hypot(h, kk), 1e-8, 0.95))
        om = float(math.atan2(kk, h))
        return p, e, float(t_val), float(abs(k_val)), om

    def residual(theta: NDArray[np.floating]) -> NDArray[np.floating]:
        p, e, t_val, k_val, om = pack_orbit(theta)
        model = np.empty_like(y_all)
        sigma = np.empty_like(y_all)
        for i, _inst in enumerate(instruments):
            mask = inst_idx == i
            gamma = float(theta[5 + 2 * i])
            log_jit = float(np.clip(theta[5 + 2 * i + 1], -10.0, math.log(config.jitter_max_kms)))
            jitter = float(np.exp(log_jit))
            model[mask] = rv_curve_kms(
                t_all[mask],
                period_day=p,
                eccentricity=e,
                t_periastron_mjd=t_val,
                k_kms=k_val,
                omega_rad=om,
                gamma_kms=gamma,
            )
            sigma[mask] = np.sqrt(yerr_all[mask] ** 2 + jitter**2)
        rv_resid = (y_all - model) / np.clip(sigma, 1e-6, None)
        prior = np.array(
            [
                (p - p0) / sig_p,
                (e - e0) / sig_e,
                (om - om0) / sig_om,
                (t_val - t0) / sig_t,
            ],
            dtype=np.float64,
        )
        return np.concatenate([rv_resid, prior])

    try:
        x0_arr = np.asarray(x0, dtype=np.float64)
        lower = np.full_like(x0_arr, -np.inf)
        upper = np.full_like(x0_arr, np.inf)
        lower[0] = math.log(1.0)  # P >= 1 day
        upper[0] = math.log(1.0e5)
        lower[1] = 1e-4  # K
        upper[1] = 1.0e3
        lower[2] = -0.95  # h, k for eccentricity
        upper[2] = 0.95
        lower[3] = -0.95
        upper[3] = 0.95
        log_j_max = math.log(config.jitter_max_kms)
        for i in range(len(instruments)):
            lower[5 + 2 * i + 1] = -10.0
            upper[5 + 2 * i + 1] = log_j_max
        sol = least_squares(
            residual,
            x0_arr,
            bounds=(lower, upper),
            max_nfev=config.joint_fit_max_nfev,
        )
        theta = sol.x
        r0 = residual(x0_arr)
        success = bool(sol.success) or float(sol.cost) <= float(0.5 * r0.dot(r0))
    except Exception as exc:  # noqa: BLE001 — keep candidate, record failure
        extras = dict(candidate.extras)
        extras["joint_orbit_fit_skip_reason"] = (
            f"optimize_failed:{type(exc).__name__}"
        )
        return candidate.model_copy(update={"extras": extras})

    if not success:
        extras = dict(candidate.extras)
        extras["joint_orbit_fit_skip_reason"] = "optimize_not_converged"
        return candidate.model_copy(update={"extras": extras})

    p, e, t_val, k_val, om = pack_orbit(theta)
    variances = [
        sig_p**2,
        sig_e**2,
        sig_t**2,
        max((0.05 * k_val) ** 2, 1e-6),
        sig_om**2,
    ]
    orbit_ps = _orbit_parameterset(
        period_day=p,
        eccentricity=e,
        t_periastron_mjd=t_val,
        k_kms=k_val,
        omega_rad=om,
        variances=variances,
        provenance=JOINT_PROVENANCE,
    )
    m2_ps = _joint_refine_m2(
        candidate,
        period_day=p,
        k_kms=k_val,
        eccentricity=e,
        inclination_deg=orbit0.inclination_deg,
    )

    instruments_out = [
        {
            "instrument": inst,
            "gamma_kms": float(theta[5 + 2 * i]),
            "jitter_kms": float(
                np.exp(np.clip(theta[5 + 2 * i + 1], -10.0, math.log(config.jitter_max_kms)))
            ),
        }
        for i, inst in enumerate(instruments)
    ]

    extras = dict(candidate.extras)
    extras.pop("joint_orbit_fit_skip_reason", None)
    extras["joint_orbit"] = orbit_ps.model_dump(mode="json")
    extras["joint_orbit_instruments"] = instruments_out
    extras["joint_orbit_seed"] = "joker" if seed is not None else "astrometric_gate"

    updates: dict[str, Any] = {
        "orbit_tier": OrbitTier.JOINT_ASTROMETRY_RV,
        "extras": extras,
    }
    if m2_ps is not None:
        updates["m2"] = m2_ps
    return candidate.model_copy(update=updates)


def run_joint_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> tuple[list[CandidateRecord], JointFitDiagnostics]:
    """Apply ``joint_orbit_fit``: passers refined; failures keep astrometry_only."""
    rc = config.rv_consistency
    diag = JointFitDiagnostics(n_input=len(candidates))
    out: list[CandidateRecord] = []
    for candidate in candidates:
        if not _gate_passed(candidate):
            extras = dict(candidate.extras)
            extras["joint_orbit_fit_skip_reason"] = JOINT_ORBIT_SKIP_REASON
            updated = candidate.model_copy(
                update={
                    "orbit_tier": OrbitTier.ASTROMETRY_ONLY,
                    "extras": extras,
                }
            )
            diag.n_skipped_gate_failed += 1
            out.append(updated)
            continue
        updated = fit_joint_orbit(candidate, rc)
        if updated.extras.get("joint_orbit_fit_skip_reason"):
            reason = str(updated.extras["joint_orbit_fit_skip_reason"])
            if reason.startswith("optimize_"):
                diag.n_failed_optimize += 1
            else:
                diag.n_skipped_other += 1
        else:
            diag.n_fit += 1
        out.append(updated)
    return out, diag


# ---------------------------------------------------------------------------
# Stage runners + HDF5
# ---------------------------------------------------------------------------


def write_stage_hdf5(
    path: Path,
    candidates: Sequence[CandidateRecord],
    *,
    stage_name: str,
    diagnostics: Mapping[str, Any],
) -> None:
    """Write one stage HDF5 via the mass_derivation layout helper."""
    write_mass_stage_hdf5(
        path, candidates, stage_name=stage_name, diagnostics=diagnostics
    )


def read_stage_hdf5(path: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    return read_mass_stage_hdf5(path)


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


def _load_upstream_candidates(
    manifest: RunManifest, stage_name: str
) -> list[CandidateRecord]:
    path = _upstream_artifact(manifest, stage_name)
    candidates, _meta = read_stage_hdf5(path)
    return candidates


def format_gate_funnel_table(diag: GateDiagnostics) -> str:
    """Full-detail gate funnel (exempt from caveman compression)."""
    lines = [
        "=== rv_astrometry_gate funnel ===",
        f"  input_candidates: {diag.n_input}",
        f"  scored: {diag.n_scored}",
        f"  passed: {diag.n_passed}",
        f"  failed: {diag.n_failed}",
        f"  skipped_no_rv: {diag.n_skipped_no_rv}",
        f"  skipped_elements: {diag.n_skipped_elements}",
        f"  sb2_flagged: {diag.n_sb2}",
        f"  sb2_mass_ratio_unlocked: {diag.n_sb2_mass_ratio_unlocked}",
    ]
    if diag.chi2_dof_values:
        arr = np.asarray(diag.chi2_dof_values, dtype=np.float64)
        lines.append(
            f"  chi2_dof: median={float(np.median(arr)):.4f} "
            f"p90={float(np.percentile(arr, 90)):.4f}"
        )
    lines.append("=== end rv_astrometry_gate funnel ===")
    return "\n".join(lines)


def write_gate_diagnostic_artifacts(
    diagnostics: GateDiagnostics,
    artifact_path: Path,
    config: PipelineConfig,
) -> list[Path]:
    """Write funnel report + gate pass-rate diagnostic hook outputs."""
    dirs = resolve_diagnostic_dirs(
        config, run_id="rv_astrometry_gate", beside_artifact=artifact_path
    )
    written: list[Path] = []
    report_path = dirs.reports / "rv_astrometry_gate_funnel.txt"
    report_path.write_text(
        format_gate_funnel_table(diagnostics) + "\n", encoding="utf-8"
    )
    written.append(report_path)
    hook = emit_gate_pass_rate(
        config,
        dirs,
        counts=diagnostics.counts_for_hook(),
        gate_name="rv_astrometry_gate",
    )
    written.extend(hook.reports)
    written.extend(hook.figures)

    if config.diagnostics.write_figures and diagnostics.chi2_dof_values:
        fig_path = plot_histogram(
            np.asarray(diagnostics.chi2_dof_values, dtype=np.float64),
            dirs.figures / "rv_gate_chi2_dof.png",
            xlabel=r"$\chi^2 / \mathrm{dof}$",
            ylabel="count",
            title="rv_astrometry_gate: χ²/dof for scored systems",
            dpi=int(config.diagnostics.figure_dpi),
            max_bins=int(config.diagnostics.histogram_max_bins),
            style=config.plotting,
        )
        if fig_path is not None:
            written.append(fig_path)
    return written


def run_rv_astrometry_gate(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
) -> RunManifest:
    """Execute ``rv_astrometry_gate`` → HDF5 + pass-rate diagnostics."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["rv_astrometry_gate"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    if candidates is None:
        candidates = _load_upstream_candidates(manifest, "mass_derivation_refined")

    updated, diagnostics = run_gate_on_candidates(candidates, config)
    write_stage_hdf5(
        artifact,
        updated,
        stage_name="rv_astrometry_gate",
        diagnostics=diagnostics.as_dict(),
    )
    write_gate_diagnostic_artifacts(diagnostics, artifact, config)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest


def run_joint_orbit_fit(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
) -> RunManifest:
    """Execute ``joint_orbit_fit`` for gate passers; failures keep astrometry_only."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["joint_orbit_fit"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    if candidates is None:
        candidates = _load_upstream_candidates(manifest, "rv_astrometry_gate")

    updated, diagnostics = run_joint_on_candidates(candidates, config)
    write_stage_hdf5(
        artifact,
        updated,
        stage_name="joint_orbit_fit",
        diagnostics=diagnostics.as_dict(),
    )

    status = StageStatus.COMPLETED
    reason = None
    if (
        diagnostics.n_input > 0
        and diagnostics.n_fit == 0
        and diagnostics.n_skipped_gate_failed == diagnostics.n_input
    ):
        status = StageStatus.SKIPPED
        reason = JOINT_ORBIT_SKIP_REASON

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=status,
        artifact_path=artifact,
        reason=reason,
    )
    save_run_manifest(manifest, run_path)
    return manifest
