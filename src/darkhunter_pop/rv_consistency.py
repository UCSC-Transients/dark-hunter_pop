"""Stages: ``rv_astrometry_gate`` and ``joint_orbit_fit``.

Consumes ``dark-hunter_rv`` JSON summaries into ``CandidateRecord.rv_summary``
(ARCHITECTURE.md §4; ``docs/RV_SUMMARY_JSON`` in dark-hunter_rv).

Gate: hold astrometric P, e, T_periastron, ω fixed; K is an **RV** quantity —
use measured ``semi_amp_primary`` / Joker K when present, otherwise estimate K from
M1, M2, and inclination (``predicted_k_kms``). Inclination comes from NSS/summary
when published, else from Thiele–Innes → Campbell. Fit γ + jitter per instrument;
score whole-curve chi2/dof vs ``rv_consistency.chi2_dof_threshold``. SB2 orbits that
disagree with astrometry fail the gate (outlier path); consistent SB2 unlocks the
mass-ratio channel for ``companion_nature_likelihood``.

Joint fit: separate registered stage after the gate. Passers get one Keplerian orbit
fit simultaneously to their RV epochs and the Gaia NSS astrometric solution vector
(full covariance), with K derived from a0/parallax rather than free (#347); seeds
from the NSS Campbell elements (both ω/Ω branches) and The Joker when present.
Converged fits with no bound hit become ``joint_astrometry_rv``. Gate failures keep
``astrometry_only`` with skip reason ``rv_astrometry_gate_failed``; skipped,
unconverged and bound-hit fits keep ``astrometry_only`` with their own reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from scipy.optimize import brentq, minimize, minimize_scalar

from darkhunter_pop import constants
from darkhunter_pop.config_loader import require_dr3_active_for_v1
from darkhunter_pop.config_schema import PipelineConfig, RvConsistencyConfig
from darkhunter_pop.diagnostic_hooks import (
    emit_gate_pass_rate,
    resolve_diagnostic_dirs,
)
from darkhunter_pop.mass_derivation import read_stage_hdf5 as read_mass_stage_hdf5
from darkhunter_pop.mass_derivation import write_stage_hdf5 as write_mass_stage_hdf5
from darkhunter_pop.physics_utils import (
    spectroscopic_mass_function,
    thiele_innes_to_campbell,
)
from darkhunter_pop.plotting import plot_histogram, plot_scatter_xy
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_and_guard,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.rv_adapter import attach_rv_summaries, rv_summary_mtime
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
    """Binary mass function f(M) in solar masses (P days, K km/s).

    Scalar wrapper around ``physics_utils.spectroscopic_mass_function`` so the
    RV-gate path cannot drift from the SB1 primitive.
    """
    value = spectroscopic_mass_function(period_day, k_kms, eccentricity)
    return float(np.asarray(value, dtype=np.float64))


def solve_m2_with_inclination_msun(
    f_mass: float,
    m1: float,
    inclination_deg: float,
    *,
    m2_min_msun: float,
    m2_max_msun: float,
) -> float | None:
    """Solve ``(M2^3 sin^3 i) / (M1 + M2)^2 = f`` for ``M2`` inside a bracket.

    The left side is monotonically increasing in ``M2``, so a root exists in
    ``[m2_min_msun, m2_max_msun]`` iff the residual changes sign across it.
    Returns ``None`` when there is no root inside the bracket (#347): bracket
    edges are never returned as if they were solutions.

    Args:
        f_mass: mass function (Msun); spectroscopic ``f`` with the true ``i``,
            or the astrometric ``(a1)^3 / P^2`` with ``inclination_deg=90``.
        m1: primary mass (Msun), > 0.
        inclination_deg: orbital inclination (deg); ``|sin i|`` < 1e-6 → None.
        m2_min_msun / m2_max_msun: search bracket (config-owned).
    """
    if not np.isfinite(f_mass) or not np.isfinite(m1) or not np.isfinite(inclination_deg):
        return None
    if f_mass <= 0 or m1 <= 0 or not (0.0 < m2_min_msun < m2_max_msun):
        return None
    sin_i = float(np.sin(np.deg2rad(float(inclination_deg))))
    if not np.isfinite(sin_i) or abs(sin_i) < 1e-6:
        return None
    s3 = abs(sin_i) ** 3

    def resid(m2: float) -> float:
        return (m2**3) * s3 / ((m1 + m2) ** 2) - f_mass

    lo_val = resid(m2_min_msun)
    hi_val = resid(m2_max_msun)
    if lo_val > 0.0 or hi_val < 0.0:
        return None
    if lo_val == 0.0:
        return float(m2_min_msun)
    if hi_val == 0.0:
        return float(m2_max_msun)
    return float(brentq(resid, m2_min_msun, m2_max_msun, xtol=1e-12, rtol=1e-12))


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


def _iter_epoch_rows(
    rv_summary: Mapping[str, Any],
) -> list[tuple[float | None, float | None, float | None, str]]:
    rows: list[tuple[float | None, float | None, float | None, str]] = []
    for row in rv_summary.get("pipeline_epochs") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            (
                _finite(row.get("mjd")),
                _finite(row.get("rv_kms")),
                _finite(row.get("rv_err_kms")),
                str(row.get("telescope") or row.get("instrument") or "UNKNOWN"),
            )
        )
    for row in rv_summary.get("external_rvs") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            (
                _finite(row.get("mjd")),
                _finite(row.get("rv_kms", row.get("rv"))),
                _finite(row.get("rv_err_kms", row.get("rv_err"))),
                str(row.get("telescope") or row.get("instrument") or "LITERATURE"),
            )
        )
    return rows


def collect_rv_epochs(
    rv_summary: Mapping[str, Any], *, min_mjd: float | None = None
) -> list[RvEpoch]:
    """Flatten pipeline + external epochs from dark-hunter_rv summary JSON.

    Rows with a missing/non-finite MJD, RV or error, or a non-positive error,
    are dropped. When ``min_mjd`` is set (``rv_consistency.rv_epoch_min_mjd``),
    rows with ``mjd < min_mjd`` are dropped too: upstream summaries carry
    placeholder ``mjd: 0.0`` rows (#347), and one such epoch at MJD 0 puts the
    point ~330 orbits away from the rest and wrecks any Keplerian fit.
    Count them with :func:`count_rejected_epochs_bad_mjd`.
    """
    epochs: list[RvEpoch] = []
    for mjd, rv, err, tel in _iter_epoch_rows(rv_summary):
        if mjd is None or rv is None or err is None or err <= 0:
            continue
        if min_mjd is not None and mjd < float(min_mjd):
            continue
        epochs.append(RvEpoch(mjd=mjd, rv_kms=rv, rv_err_kms=err, instrument=tel))
    return epochs


def count_rejected_epochs_bad_mjd(
    rv_summary: Mapping[str, Any], *, min_mjd: float
) -> int:
    """Number of otherwise-valid epochs rejected by the ``min_mjd`` cut."""
    n = 0
    for mjd, rv, err, _tel in _iter_epoch_rows(rv_summary):
        if mjd is None or rv is None or err is None or err <= 0:
            continue
        if mjd < float(min_mjd):
            n += 1
    return n


def _nss_blocks(candidate: CandidateRecord) -> dict[str, Any]:
    """Merge candidate.nss_orbital with rv_summary.nss_orbital (summary wins)."""
    merged: dict[str, Any] = dict(candidate.nss_orbital or {})
    summary = candidate.rv_summary or {}
    block = summary.get("nss_orbital")
    if isinstance(block, Mapping):
        merged.update(dict(block))
    return merged


def _ti_abfg(candidate: CandidateRecord) -> tuple[float, float, float, float] | None:
    """Return (A,B,F,G) mas from candidate or rv_summary thiele_innes, if complete."""
    ti = candidate.thiele_innes
    if ti is not None and None not in (ti.A, ti.B, ti.F, ti.G):
        return float(ti.A), float(ti.B), float(ti.F), float(ti.G)
    summary = candidate.rv_summary or {}
    block = summary.get("thiele_innes")
    if isinstance(block, Mapping):
        vals = [_finite(block.get(k)) for k in ("A", "B", "F", "G")]
        if all(v is not None for v in vals):
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
    return None


def _inclination_deg(
    candidate: CandidateRecord, nss: Mapping[str, Any]
) -> float | None:
    """Inclination in degrees: NSS/Joker first, else Thiele–Innes → Campbell."""
    inc = _first_finite(nss, ("inclination_deg", "inclination", "Inclination"))
    if inc is not None:
        return inc
    summary = candidate.rv_summary or {}
    joker = summary.get("joker_fit")
    if isinstance(joker, Mapping):
        inc = _finite(joker.get("inclination_deg"))
        if inc is not None:
            return inc
    # Published inclination sometimes lives under gaia_metadata after JSON refresh.
    meta = summary.get("gaia_metadata")
    if isinstance(meta, Mapping):
        inc = _first_finite(meta, ("Inclination", "inclination_deg", "inclination"))
        if inc is not None:
            return inc
    abfg = _ti_abfg(candidate)
    if abfg is None:
        return None
    _a0, _omega, inc_rad = thiele_innes_to_campbell(*abfg)
    inc_val = float(np.rad2deg(np.asarray(inc_rad, dtype=np.float64)))
    if not np.isfinite(inc_val):
        return None
    return inc_val


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


def _omega_rad_for_candidate(
    candidate: CandidateRecord,
    nss: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> float | None:
    """ω from NSS/Joker, else from Thiele–Innes Campbell conversion."""
    om = _omega_rad(nss, summary)
    if om is not None:
        return om
    abfg = _ti_abfg(candidate)
    if abfg is None:
        return None
    _a0, omega, _inc = thiele_innes_to_campbell(*abfg)
    om_val = float(np.asarray(omega, dtype=np.float64))
    if not np.isfinite(om_val):
        return None
    return om_val


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
    """Resolve fixed gate elements: astrometric P,e,T,ω + RV K.

    K is not an NSS Orbital column for pure astrometric solutions. Prefer measured
    RV/Joker K; otherwise estimate from M1, M2, and inclination
    (``predicted_k_kms``). Inclination / ω fall back to Thiele–Innes → Campbell
    when not published on the NSS row.
    """
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
    omega = _omega_rad_for_candidate(candidate, nss, summary)
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
    rv_summary_attached: int = 0
    rv_summary_missing: int = 0
    rv_summary_kept_existing: int = 0
    rv_summary_disabled: int = 0
    # #347: placeholder-MJD epochs dropped before scoring.
    n_epochs_rejected_bad_mjd: int = 0
    n_candidates_with_rejected_epochs: int = 0
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
            "rv_summary_attached": self.rv_summary_attached,
            "rv_summary_missing": self.rv_summary_missing,
            "rv_summary_kept_existing": self.rv_summary_kept_existing,
            "rv_summary_disabled": self.rv_summary_disabled,
            "n_epochs_rejected_bad_mjd": self.n_epochs_rejected_bad_mjd,
            "n_candidates_with_rejected_epochs": self.n_candidates_with_rejected_epochs,
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
    epochs = collect_rv_epochs(
        candidate.rv_summary or {}, min_mjd=config.rv_epoch_min_mjd
    )
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


def _gate_priority_key(
    candidate: CandidateRecord,
    config: PipelineConfig,
) -> tuple[int, int, float]:
    """Higher tuple sorts first: calibrators → public RVs → newer summary mtime."""
    rc = config.rv_consistency
    priority_ids = set(rc.priority_source_ids)
    is_priority = 1 if candidate.source_id in priority_ids else 0
    has_public = 0
    if rc.prefer_public_external_rvs:
        ext = (candidate.rv_summary or {}).get("external_rvs") or []
        if isinstance(ext, list) and len(ext) > 0:
            has_public = 1
    mtime = (
        rv_summary_mtime(config, candidate.source_id)
        if rc.prefer_recent_summary_mtime
        else 0.0
    )
    return (is_priority, has_public, mtime)


def run_gate_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> tuple[list[CandidateRecord], GateDiagnostics]:
    """Apply ``rv_astrometry_gate`` to an in-memory candidate list.

    Re-attaches ``dark-hunter_rv`` JSON summaries from the configured root so a
    root fill after ``data_acquisition`` still scores systems with epochs.
    Order: known calibrators, public/lit RVs, then newest on-disk summary mtime.
    """
    attached, rv_stats = attach_rv_summaries(candidates, config)
    ordered = sorted(
        attached,
        key=lambda c: _gate_priority_key(c, config),
        reverse=True,
    )
    rc = config.rv_consistency
    diag = GateDiagnostics(
        n_input=len(ordered),
        rv_summary_attached=int(rv_stats.get("attached", 0)),
        rv_summary_missing=int(rv_stats.get("missing", 0)),
        rv_summary_kept_existing=int(rv_stats.get("kept_existing", 0)),
        rv_summary_disabled=int(rv_stats.get("disabled", 0)),
    )
    out: list[CandidateRecord] = []
    for candidate in ordered:
        n_bad = count_rejected_epochs_bad_mjd(
            candidate.rv_summary or {}, min_mjd=rc.rv_epoch_min_mjd
        )
        if n_bad:
            diag.n_epochs_rejected_bad_mjd += n_bad
            diag.n_candidates_with_rejected_epochs += 1
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
#
# Model (#347). One Keplerian orbit is fit simultaneously to
#   (1) the RV epochs (per-instrument γ + jitter, Gaussian likelihood with the
#       jitter normalization term), and
#   (2) the Gaia NSS astrometric solution vector — parallax, Thiele–Innes
#       A/B/F/G (and C/H for AstroSpectroSB1), e, P, T_peri — through its FULL
#       covariance (sub-block of ``CandidateRecord.nss_solution``; marginalizing
#       over the unmodeled ra/dec/pm/γ_Gaia is exactly the sub-block), and
#   (3) a Gaussian prior on M1 from the upstream ``m1`` ParameterSet.
# Free physical parameters: P, √e cos ω, √e sin ω, T_peri, Ω, i, M1, M2, ϖ.
# K is *derived*, never free:  a_tot^3 = k (M1+M2) P^2,  a1 = a_tot M2/(M1+M2),
# a0 = a1 ϖ,  K = 2π a1 sin i / (P √(1-e²)).  The photocentre is taken to be
# the primary's orbit (dark-companion hypothesis, ARCHITECTURE.md §4
# "astrometry + RV joint fit for dark companions"); a luminous companion
# shrinks a0 and shows up as tension, not as a silently biased M2.
# The RV-only fit this replaces had K free and nothing tying it to a0, so
# sparse RVs drove K to its 1e-4 / 1e3 bounds (M2 = 0 / 500).
#
# Point estimate = maximum of the likelihood (flat priors in the fit
# parameters apart from the M1 Gaussian). Covariance = inverse Fisher
# information at the optimum (Gauss–Newton / Laplace approximation, built from
# model Jacobians), propagated to (P, e, T, K, ω, i, Ω, M1, M2, ϖ) with the
# numerical Jacobian of that map. No prior widths and no diagonal stand-ins.

JOINT_THETA_NAMES: tuple[str, ...] = (
    "P",
    "sqrt_e_cos_omega",
    "sqrt_e_sin_omega",
    "T_periastron",
    "Omega",
    "inclination",
    "M1",
    "M2",
    "parallax",
)
JOINT_OUTPUT_NAMES: tuple[str, ...] = (
    "P",
    "e",
    "T_periastron",
    "K",
    "omega",
    "inclination",
    "Omega",
    "M1",
    "M2",
    "parallax",
)
JOINT_OUTPUT_UNITS: tuple[str, ...] = (
    "day",
    "1",
    "MJD",
    "km/s",
    "rad",
    "rad",
    "rad",
    "Msun",
    "Msun",
    "mas",
)
# NSS solution-vector entries the orbit model predicts (others are marginalized).
_NSS_MODEL_NAMES: tuple[str, ...] = (
    "parallax",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "c_thiele_innes",
    "h_thiele_innes",
    "eccentricity",
    "period",
    "t_periastron",
)
_NSS_REQUIRED_NAMES: tuple[str, ...] = (
    "parallax",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "eccentricity",
    "period",
    "t_periastron",
)
# Optimizer conditioning scales (z = (θ − θ0)/scale). They change only the
# optimizer path and the units of the gradient / bound tolerances, never the
# optimum or the Fisher-based covariance (which is transformed back exactly).
_COND_SQRT_E = 0.02
_COND_ANGLE_RAD = 0.02
_COND_M2_FRAC = 0.05
_COND_LOG_JITTER = 0.1
# −ln L returned outside the e < joint_ecc_max region (L-BFGS-B backtracks).
_INFEASIBLE_NLL = 1.0e30

# Skip / failure reasons recorded in ``extras["joint_orbit_fit_skip_reason"]``.
JOINT_SKIP_MISSING_NSS_COVARIANCE = "missing_nss_covariance"
JOINT_SKIP_NO_ASTROMETRIC_ORBIT = "nss_solution_lacks_astrometric_orbit"
JOINT_SKIP_NSS_COV_NOT_PD = "nss_covariance_not_positive_definite"
JOINT_SKIP_MISSING_M1 = "missing_m1_or_uncertainty"
JOINT_FAIL_NOT_CONVERGED = "optimize_not_converged"
JOINT_FAIL_BOUND_HIT = "optimize_bound_hit"


@dataclass
class JointFitDiagnostics:
    """Fit-quality accounting for ``joint_orbit_fit`` (#347).

    ``n_fit`` counts only fits that passed the convergence test (identified,
    positive-definite Fisher matrix; scoring decrement below
    ``joint_fit_decrement_tol``) and hit no bound. Bound hits and non-converged fits are counted separately
    and keep their upstream ``astrometry_only`` masses.
    """

    n_input: int = 0
    n_fit: int = 0
    n_skipped_gate_failed: int = 0
    n_skipped_other: int = 0
    n_failed_optimize: int = 0
    n_attempted: int = 0
    n_converged: int = 0
    n_not_converged: int = 0
    n_bound_hit: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    bound_hit_parameters: dict[str, int] = field(default_factory=dict)
    fitted_source_ids: list[int] = field(default_factory=list)
    m2_joint_msun: list[float] = field(default_factory=list)
    m2_joint_sigma_msun: list[float] = field(default_factory=list)
    m2_upstream_msun: list[float] = field(default_factory=list)
    k_joint_kms: list[float] = field(default_factory=list)
    k_astrometry_seed_kms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_fit": self.n_fit,
            "n_skipped_gate_failed": self.n_skipped_gate_failed,
            "n_skipped_other": self.n_skipped_other,
            "n_failed_optimize": self.n_failed_optimize,
            "n_attempted": self.n_attempted,
            "n_converged": self.n_converged,
            "n_not_converged": self.n_not_converged,
            "n_bound_hit": self.n_bound_hit,
            "skip_reasons": dict(self.skip_reasons),
            "bound_hit_parameters": dict(self.bound_hit_parameters),
            "fitted_source_ids": list(self.fitted_source_ids),
            "m2_joint_msun": list(self.m2_joint_msun),
            "m2_joint_sigma_msun": list(self.m2_joint_sigma_msun),
            "m2_upstream_msun": list(self.m2_upstream_msun),
            "k_joint_kms": list(self.k_joint_kms),
            "k_astrometry_seed_kms": list(self.k_astrometry_seed_kms),
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


def campbell_from_thiele_innes(
    a_ti: float, b_ti: float, f_ti: float, g_ti: float
) -> tuple[float, float, float, float]:
    """Return ``(a0_mas, omega_rad, Omega_rad, inclination_rad)`` from A, B, F, G.

    Same convention as :func:`physics_utils.thiele_innes_to_campbell`
    (Halbwachs et al. 2023 / NSSTools), plus the node Ω it does not return.
    Inverse of :func:`thiele_innes_from_campbell`; (ω, Ω) and (ω+π, Ω+π) map to
    the same A, B, F, G — RVs break that degeneracy in the joint fit.
    """
    a0, omega, inc = thiele_innes_to_campbell(a_ti, b_ti, f_ti, g_ti)
    wp = math.atan2(b_ti - f_ti, a_ti + g_ti)
    wm = math.atan2(-b_ti - f_ti, a_ti - g_ti)
    node = 0.5 * (wp - wm)
    if node < 0:
        node += math.pi
    return (
        float(np.asarray(a0)),
        float(np.asarray(omega)),
        float(node),
        float(np.asarray(inc)),
    )


def thiele_innes_from_campbell(
    a0: float, omega: float, node: float, inc: float
) -> tuple[float, float, float, float]:
    """Thiele–Innes ``(A, B, F, G)`` (units of ``a0``) from Campbell elements (rad)."""
    cw, sw = math.cos(omega), math.sin(omega)
    co, so = math.cos(node), math.sin(node)
    ci = math.cos(inc)
    a_ti = a0 * (cw * co - sw * so * ci)
    b_ti = a0 * (cw * so + sw * co * ci)
    f_ti = -a0 * (sw * co + cw * so * ci)
    g_ti = -a0 * (sw * so - cw * co * ci)
    return a_ti, b_ti, f_ti, g_ti


def primary_orbit_au(m1_msun: float, m2_msun: float, period_day: float) -> float:
    """Primary's semi-major axis ``a1`` (AU) from Kepler's third law."""
    mtot = m1_msun + m2_msun
    a_tot = (constants.KEPLER_AU3_PER_MSUN_DAY2 * mtot * period_day**2) ** (1.0 / 3.0)
    return float(a_tot * m2_msun / mtot)


def k_from_primary_orbit_kms(
    a1_au: float, period_day: float, eccentricity: float, inclination_rad: float
) -> float:
    """RV semi-amplitude ``K = 2π a1 sin i / (P √(1−e²))`` in km/s."""
    return float(
        2.0
        * math.pi
        * a1_au
        * abs(math.sin(inclination_rad))
        / (period_day * math.sqrt(max(1.0 - eccentricity**2, 1e-300)))
        * constants.AU_PER_DAY_KMS
    )


@dataclass
class _JointProblem:
    """Data + bookkeeping for one candidate's joint fit (internal)."""

    t: NDArray[np.floating]
    y: NDArray[np.floating]
    yerr: NDArray[np.floating]
    inst_idx: NDArray[np.int64]
    instruments: list[str]
    nss_names: list[str]
    nss_values: NDArray[np.floating]
    nss_chol: NDArray[np.floating]
    m1_prior: float
    m1_sigma: float
    ecc_max: float
    scale: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]

    @property
    def n_theta(self) -> int:
        return len(JOINT_THETA_NAMES) + 2 * len(self.instruments)

    def theta_names(self) -> list[str]:
        names = list(JOINT_THETA_NAMES)
        for inst in self.instruments:
            names.extend([f"gamma[{inst}]", f"log_jitter[{inst}]"])
        return names

    @staticmethod
    def physical(theta: NDArray[np.floating]) -> dict[str, float]:
        p, h, k, t_p, node, inc, m1, m2, plx = (float(v) for v in theta[:9])
        return {
            "P": p,
            "e": h * h + k * k,
            "omega": math.atan2(k, h),
            "T_periastron": t_p,
            "Omega": node,
            "inclination": inc,
            "M1": m1,
            "M2": m2,
            "parallax": plx,
        }

    @staticmethod
    def observables(ph: Mapping[str, float]) -> dict[str, float]:
        """Model NSS observables + K for physical parameters ``ph``."""
        a1 = primary_orbit_au(ph["M1"], ph["M2"], ph["P"])
        a0 = a1 * ph["parallax"]
        a_ti, b_ti, f_ti, g_ti = thiele_innes_from_campbell(
            a0, ph["omega"], ph["Omega"], ph["inclination"]
        )
        sin_i = math.sin(ph["inclination"])
        return {
            "parallax": ph["parallax"],
            "a_thiele_innes": a_ti,
            "b_thiele_innes": b_ti,
            "f_thiele_innes": f_ti,
            "g_thiele_innes": g_ti,
            # Gaia AstroSpectroSB1: C = a1 sin ω sin i, H = a1 cos ω sin i (AU).
            "c_thiele_innes": a1 * math.sin(ph["omega"]) * sin_i,
            "h_thiele_innes": a1 * math.cos(ph["omega"]) * sin_i,
            "eccentricity": ph["e"],
            "period": ph["P"],
            "t_periastron": ph["T_periastron"] - constants.GAIA_J2016_MJD,
            "K": k_from_primary_orbit_kms(a1, ph["P"], ph["e"], ph["inclination"]),
            "a1_au": a1,
            "a0_mas": a0,
        }

    def blocks(
        self, theta: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float] | None:
        """Model pieces: (RV mean incl. γ, RV variance, whitened NSS residual,
        whitened M1 residual); ``None`` outside the feasible region."""
        ph = self.physical(theta)
        if (
            not (0.0 <= ph["e"] < self.ecc_max)
            or ph["P"] <= 0
            or ph["M1"] <= 0
            or ph["M2"] <= 0
            or ph["parallax"] <= 0
        ):
            return None
        obs = self.observables(ph)
        model = rv_curve_kms(
            self.t,
            period_day=ph["P"],
            eccentricity=ph["e"],
            t_periastron_mjd=ph["T_periastron"],
            k_kms=obs["K"],
            omega_rad=ph["omega"],
            gamma_kms=0.0,
        )
        n_inst = len(self.instruments)
        gammas = np.asarray(theta[9 : 9 + 2 * n_inst : 2], dtype=np.float64)
        log_jit = np.asarray(theta[10 : 10 + 2 * n_inst : 2], dtype=np.float64)
        mu = model + gammas[self.inst_idx]
        var = self.yerr**2 + np.exp(2.0 * log_jit)[self.inst_idx]
        pred = np.array([obs[name] for name in self.nss_names], dtype=np.float64)
        white = solve_triangular(self.nss_chol, self.nss_values - pred, lower=True)
        m1r = (ph["M1"] - self.m1_prior) / self.m1_sigma
        return mu, var, white, float(m1r)

    def nll_parts(self, theta: NDArray[np.floating]) -> tuple[float, float, float, float]:
        """Return (total −ln L, RV chi2, NSS chi2, M1 prior chi2)."""
        parts = self.blocks(theta)
        if parts is None:
            return _INFEASIBLE_NLL, math.inf, math.inf, math.inf
        mu, var, white, m1r = parts
        resid = self.y - mu
        rv_chi2 = float(np.sum(resid**2 / var))
        nll_rv = 0.5 * rv_chi2 + 0.5 * float(np.sum(np.log(var)))
        nss_chi2 = float(white @ white)
        m1_chi2 = m1r * m1r
        total = nll_rv + 0.5 * nss_chi2 + 0.5 * m1_chi2
        if not np.isfinite(total):
            return _INFEASIBLE_NLL, math.inf, math.inf, math.inf
        return float(total), rv_chi2, nss_chi2, float(m1_chi2)

    def gradient_and_fisher(
        self,
        theta: NDArray[np.floating],
        scale: NDArray[np.floating],
        free: NDArray[np.int64],
        step: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Gradient of −ln L and Fisher information w.r.t. z (θ = θ̂ + scale·z).

        Built from central-difference Jacobians of the *model* pieces (first
        derivatives only), so it stays accurate when −ln L is badly conditioned;
        a second-difference Hessian of the scalar −ln L does not (#347: on real
        BH1 data that Hessian gave σ(M2) anywhere from 1.3 to 1.9 Msun). For a
        Gaussian likelihood with parameter-dependent RV variance v:
        F = Jμᵀ V⁻¹ Jμ + ½ Jvᵀ V⁻² Jv + Jrᵀ Jr + Jm1ᵀ Jm1 (r, m1 whitened).
        Raises ValueError when a step leaves the feasible region.
        """
        base = self.blocks(theta)
        if base is None:
            raise ValueError("joint fit point is infeasible")
        mu, var, white, m1r = base
        n = free.size
        d_mu = np.zeros((mu.size, n))
        d_var = np.zeros((var.size, n))
        d_w = np.zeros((white.size, n))
        d_m1 = np.zeros(n)
        for b, i in enumerate(free):
            dz = np.zeros_like(theta)
            dz[i] = scale[i] * step
            hi = self.blocks(theta + dz)
            lo = self.blocks(theta - dz)
            if hi is None or lo is None:
                raise ValueError(f"derivative step leaves feasible region ({i})")
            d_mu[:, b] = (hi[0] - lo[0]) / (2.0 * step)
            d_var[:, b] = (hi[1] - lo[1]) / (2.0 * step)
            d_w[:, b] = (hi[2] - lo[2]) / (2.0 * step)
            d_m1[b] = (hi[3] - lo[3]) / (2.0 * step)
        resid = self.y - mu
        grad = (
            -d_mu.T @ (resid / var)
            + 0.5 * d_var.T @ (1.0 / var - resid**2 / var**2)
            + d_w.T @ white
            + d_m1 * m1r
        )
        fisher = (
            d_mu.T @ (d_mu / var[:, None])
            + 0.5 * d_var.T @ (d_var / var[:, None] ** 2)
            + d_w.T @ d_w
            + np.outer(d_m1, d_m1)
        )
        return grad, 0.5 * (fisher + fisher.T)

    def nll(self, theta: NDArray[np.floating]) -> float:
        return self.nll_parts(theta)[0]

    def outputs(self, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reported vector in ``JOINT_OUTPUT_NAMES`` order (angles unwrapped)."""
        ph = self.physical(theta)
        obs = self.observables(ph)
        return np.array(
            [
                ph["P"],
                ph["e"],
                ph["T_periastron"],
                obs["K"],
                ph["omega"],
                ph["inclination"],
                ph["Omega"],
                ph["M1"],
                ph["M2"],
                ph["parallax"],
            ],
            dtype=np.float64,
        )


def _nss_subblock(
    solution: ParameterSet,
) -> tuple[list[str], NDArray[np.floating], NDArray[np.floating]] | None:
    names = [n for n in _NSS_MODEL_NAMES if n in solution.names]
    if any(req not in names for req in _NSS_REQUIRED_NAMES):
        return None
    idx = [solution.names.index(n) for n in names]
    values = solution.values_array()[idx]
    cov = solution.covariance_array()[np.ix_(idx, idx)]
    return names, values, cov


def _wrap_angle(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.angle(np.exp(1j * np.asarray(x, dtype=np.float64)))


@dataclass
class JointFitResult:
    """Outcome of :func:`fit_joint_orbit_model` for one candidate."""

    converged: bool
    failure_reason: str | None
    bound_hits: list[str]
    theta: NDArray[np.floating]
    theta_names: list[str]
    output_values: NDArray[np.floating]
    output_covariance: NDArray[np.floating] | None
    nll: float
    rv_chi2: float
    nss_chi2: float
    m1_chi2: float
    n_rv: int
    n_nss: int
    n_free: int
    optimizer_success: bool
    optimizer_message: str
    nfev: int
    grad_max: float
    scoring_decrement: float
    polish_steps: int
    fisher_min_eig: float
    fisher_eig_ratio: float
    jitter_at_floor: list[str]
    start_label: str
    start_nlls: dict[str, float]
    k_astrometry_seed_kms: float
    instruments: list[dict[str, float | str]]

    def status_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "failure_reason": self.failure_reason,
            "bound_hits": list(self.bound_hits),
            "nll": self.nll,
            "rv_chi2": self.rv_chi2,
            "nss_chi2": self.nss_chi2,
            "m1_prior_chi2": self.m1_chi2,
            "n_rv": self.n_rv,
            "n_nss": self.n_nss,
            "n_free": self.n_free,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "nfev": self.nfev,
            "grad_max": self.grad_max,
            "scoring_decrement": self.scoring_decrement,
            "polish_steps": self.polish_steps,
            "fisher_min_eig": self.fisher_min_eig,
            "fisher_eig_ratio": self.fisher_eig_ratio,
            "jitter_at_floor": list(self.jitter_at_floor),
            "start": self.start_label,
            "start_nlls": dict(self.start_nlls),
            "k_astrometry_seed_kms": self.k_astrometry_seed_kms,
        }


def _optimize_from(
    problem: _JointProblem,
    theta0: NDArray[np.floating],
    config: RvConsistencyConfig,
) -> tuple[NDArray[np.floating], Any]:
    scale = problem.scale

    def to_theta(z: NDArray[np.floating]) -> NDArray[np.floating]:
        return theta0 + scale * z

    def fun(z: NDArray[np.floating]) -> float:
        return problem.nll(to_theta(z))

    zl = (problem.lower - theta0) / scale
    zu = (problem.upper - theta0) / scale
    bounds = [
        (None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi))
        for lo, hi in zip(zl, zu)
    ]
    res = minimize(
        fun,
        np.zeros_like(theta0),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxfun": int(config.joint_fit_max_nfev),
            "maxiter": int(config.joint_fit_max_nfev),
            "ftol": 1e-13,
            "gtol": 1e-8,
        },
    )
    return to_theta(np.asarray(res.x, dtype=np.float64)), res


def fit_joint_orbit_model(
    problem: _JointProblem,
    starts: Mapping[str, NDArray[np.floating]],
    config: RvConsistencyConfig,
    *,
    k_seed_kms: float,
) -> JointFitResult:
    """Maximize the joint likelihood from each start; test convergence; Laplace covariance.

    Convergence is tested on the final point, not on an optimizer flag (#347):
    after Fisher-scoring polish, the Fisher information over free parameters
    must be positive definite with smallest/largest eigenvalue ratio above
    ``joint_fisher_min_eig_ratio`` (all free parameters identified), and the
    scoring decrement ``gᵀ F⁻¹ g`` (≈ twice the −ln L a further step would gain;
    parametrization-invariant) must be at most ``joint_fit_decrement_tol``.
    Gradient and Fisher come from central-difference Jacobians of the model
    (steps of ``joint_derivative_step`` local-σ units). The covariance is F⁻¹. The
    L-BFGS-B termination status/message is recorded too.
    A parameter within ``joint_bound_tol`` of a box edge is a bound hit (except a
    per-instrument jitter at its floor, a legitimate zero-jitter boundary that is
    then held fixed). Bound hits are reported and never treated as solutions.
    """
    best: tuple[float, NDArray[np.floating], Any, str] | None = None
    start_nlls: dict[str, float] = {}
    for label, theta0 in starts.items():
        theta_opt, res = _optimize_from(problem, theta0, config)
        val = problem.nll(theta_opt)
        # L-BFGS-B's relative-reduction stop can fire early on a curved ridge;
        # restarting from its own result resets the quasi-Newton memory.
        for _ in range(int(config.joint_lbfgsb_restarts)):
            theta_new, res_new = _optimize_from(problem, theta_opt, config)
            val_new = problem.nll(theta_new)
            if not val_new < val - 1e-9:
                break
            theta_opt, res, val = theta_new, res_new, val_new
        start_nlls[label] = float(val)
        if best is None or val < best[0]:
            best = (val, theta_opt, res, label)
    assert best is not None
    _val, theta_hat, res, label = best

    step = float(config.joint_derivative_step)
    tol = float(config.joint_bound_tol)
    names = problem.theta_names()
    cond = problem.scale

    def bound_state(theta: NDArray[np.floating]) -> tuple[list[str], list[int]]:
        zl = (problem.lower - theta) / cond
        zu = (problem.upper - theta) / cond
        hits: list[str] = []
        floors: list[int] = []
        for i in range(theta.size):
            at_lo = bool(np.isfinite(zl[i]) and abs(zl[i]) <= tol)
            at_hi = bool(np.isfinite(zu[i]) and abs(zu[i]) <= tol)
            is_logjit = i >= 9 and (i - 9) % 2 == 1
            if is_logjit and at_lo:
                floors.append(i)
            elif at_lo or at_hi:
                hits.append(names[i])
        ph = problem.physical(theta)
        if ph["e"] >= problem.ecc_max * (1.0 - tol):
            hits.append("e")
        return hits, floors

    def local_scale(theta: NDArray[np.floating], free: NDArray[np.int64]) -> NDArray[np.floating]:
        """Per-parameter local σ (1/sqrt of the Fisher diagonal), so finite
        differences step a fixed fraction of the *posterior* width; the RV data
        can constrain P / T_peri orders of magnitude better than NSS does."""
        _g, fis = problem.gradient_and_fisher(theta, cond, free, step)
        diag = np.diag(fis)
        out = cond.copy()
        good = np.isfinite(diag) & (diag > 0)
        # Shrink only: never step wider than the conditioning scale, so a
        # weakly constrained direction (e.g. jitter near zero) cannot blow up.
        out[free[good]] = cond[free[good]] / np.maximum(np.sqrt(diag[good]), 1.0)
        return out

    hits, floors = bound_state(theta_hat)
    free = np.array([i for i in range(theta_hat.size) if i not in floors], dtype=np.int64)
    scale = local_scale(theta_hat, free)
    grad, fisher = problem.gradient_and_fisher(theta_hat, scale, free, step)
    polish_steps = 0
    # Fisher-scoring polish inside the box (only when no bound is active).
    if not hits:
        for _ in range(int(config.joint_polish_steps)):
            try:
                dz_free = -np.linalg.solve(fisher, grad)
            except np.linalg.LinAlgError:
                break
            if float(-grad @ dz_free) <= 1e-3 * float(config.joint_fit_decrement_tol):
                break
            f_old = problem.nll(theta_hat)
            accepted = False
            for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
                dz = np.zeros_like(theta_hat)
                dz[free] = shrink * dz_free
                cand = np.clip(theta_hat + scale * dz, problem.lower, problem.upper)
                if problem.nll(cand) < f_old:
                    theta_hat = cand
                    accepted = True
                    break
            if not accepted:
                break
            polish_steps += 1
            grad, fisher = problem.gradient_and_fisher(theta_hat, scale, free, step)
        hits, floors = bound_state(theta_hat)
        free = np.array(
            [i for i in range(theta_hat.size) if i not in floors], dtype=np.int64
        )
        scale = local_scale(theta_hat, free)
        grad, fisher = problem.gradient_and_fisher(theta_hat, scale, free, step)

    grad_max = float(np.max(np.abs(grad))) if grad.size else 0.0
    sym = fisher
    eig = np.linalg.eigvalsh(sym) if sym.size else np.array([np.inf])
    min_eig = float(np.min(eig))
    max_eig = float(np.max(eig))
    hess_pd = bool(
        np.all(np.isfinite(eig))
        and min_eig > max_eig * float(config.joint_fisher_min_eig_ratio)
    )
    decrement = float(grad @ np.linalg.solve(sym, grad)) if hess_pd else float("inf")

    failure: str | None = None
    if hits:
        failure = JOINT_FAIL_BOUND_HIT
    elif not hess_pd:
        failure = JOINT_FAIL_NOT_CONVERGED
    elif decrement > float(config.joint_fit_decrement_tol):
        failure = JOINT_FAIL_NOT_CONVERGED
    converged = failure is None

    out_vals = problem.outputs(theta_hat)
    out_cov: NDArray[np.floating] | None = None
    if hess_pd:
        cov_z = np.linalg.inv(sym)
        jac = np.zeros((len(JOINT_OUTPUT_NAMES), free.size), dtype=np.float64)
        for b, i in enumerate(free):
            dz = np.zeros_like(theta_hat)
            dz[i] = step
            hi_v = problem.outputs(theta_hat + scale * dz)
            lo_v = problem.outputs(theta_hat - scale * dz)
            diff = hi_v - lo_v
            for ang in (4, 6):
                diff[ang] = float(_wrap_angle(diff[ang]))
            jac[:, b] = diff / (2.0 * step)
        cov = jac @ cov_z @ jac.T
        out_cov = 0.5 * (cov + cov.T)
    out_vals = out_vals.copy()
    out_vals[4] = float(np.mod(out_vals[4], 2.0 * np.pi))
    out_vals[6] = float(np.mod(out_vals[6], 2.0 * np.pi))

    total, rv_chi2, nss_chi2, m1_chi2 = problem.nll_parts(theta_hat)
    instruments_out: list[dict[str, float | str]] = []
    for i, inst in enumerate(problem.instruments):
        instruments_out.append(
            {
                "instrument": inst,
                "gamma_kms": float(theta_hat[9 + 2 * i]),
                "jitter_kms": float(np.exp(theta_hat[10 + 2 * i])),
            }
        )
    return JointFitResult(
        converged=converged,
        failure_reason=failure,
        bound_hits=hits,
        theta=theta_hat,
        theta_names=names,
        output_values=out_vals,
        output_covariance=out_cov,
        nll=float(total),
        rv_chi2=float(rv_chi2),
        nss_chi2=float(nss_chi2),
        m1_chi2=float(m1_chi2),
        n_rv=int(problem.y.size),
        n_nss=len(problem.nss_names),
        n_free=int(free.size),
        optimizer_success=bool(res.success),
        optimizer_message=str(res.message),
        nfev=int(getattr(res, "nfev", 0)),
        grad_max=grad_max,
        scoring_decrement=decrement,
        polish_steps=polish_steps,
        fisher_min_eig=min_eig,
        fisher_eig_ratio=min_eig / max_eig if max_eig > 0 else float("nan"),
        jitter_at_floor=[problem.instruments[(i - 10) // 2] for i in floors],
        start_label=label,
        start_nlls=start_nlls,
        k_astrometry_seed_kms=float(k_seed_kms),
        instruments=instruments_out,
    )


def _skip(candidate: CandidateRecord, reason: str) -> CandidateRecord:
    extras = dict(candidate.extras)
    extras["joint_orbit_fit_skip_reason"] = reason
    return candidate.model_copy(
        update={
            "orbit_tier": candidate.orbit_tier or OrbitTier.ASTROMETRY_ONLY,
            "extras": extras,
        }
    )


def build_joint_problem(
    candidate: CandidateRecord,
    config: RvConsistencyConfig,
) -> tuple[_JointProblem, dict[str, NDArray[np.floating]], float] | str:
    """Assemble data, bounds, conditioning and starting points; or a skip reason."""
    epochs = collect_rv_epochs(
        candidate.rv_summary or {}, min_mjd=config.rv_epoch_min_mjd
    )
    if len(epochs) < config.min_epochs_total:
        return "insufficient_data"
    by_inst: dict[str, list[RvEpoch]] = {}
    for ep in epochs:
        by_inst.setdefault(ep.instrument, []).append(ep)
    instruments = [
        inst
        for inst, rows in sorted(by_inst.items())
        if len(rows) >= config.min_epochs_per_instrument
    ]
    if not instruments:
        return "no_instrument_above_min_epochs"
    if candidate.nss_solution is None:
        return JOINT_SKIP_MISSING_NSS_COVARIANCE
    block = _nss_subblock(candidate.nss_solution)
    if block is None:
        return JOINT_SKIP_NO_ASTROMETRIC_ORBIT
    nss_names, nss_values, nss_cov = block
    try:
        chol = cholesky(nss_cov, lower=True)
    except (LinAlgError, ValueError):
        return JOINT_SKIP_NSS_COV_NOT_PD
    if candidate.m1 is None:
        return JOINT_SKIP_MISSING_M1
    try:
        m1_marg = candidate.m1.marginal("M1")
    except KeyError:
        return JOINT_SKIP_MISSING_M1
    m1_0 = float(m1_marg.value)
    m1_sig = float(m1_marg.sigma) if m1_marg.sigma is not None else float("nan")
    if not (np.isfinite(m1_0) and m1_0 > 0 and np.isfinite(m1_sig) and m1_sig > 0):
        return JOINT_SKIP_MISSING_M1

    val = dict(zip(nss_names, (float(v) for v in nss_values)))
    sig = dict(zip(nss_names, (float(np.sqrt(nss_cov[i, i])) for i in range(len(nss_names)))))
    a0, omega0, node0, inc0 = campbell_from_thiele_innes(
        val["a_thiele_innes"], val["b_thiele_innes"], val["f_thiele_innes"], val["g_thiele_innes"]
    )
    p0 = val["period"]
    e0 = float(np.clip(val["eccentricity"], 0.0, config.joint_ecc_max * 0.99))
    t0 = val["t_periastron"] + constants.GAIA_J2016_MJD
    plx0 = val["parallax"]
    if not (p0 > 0 and plx0 > 0 and a0 > 0):
        return JOINT_SKIP_NO_ASTROMETRIC_ORBIT
    m1_lo, m1_hi = (float(x) for x in config.joint_m1_bounds_msun)
    m2_lo, m2_hi = (float(x) for x in config.joint_m2_bounds_msun)
    m1_seed = float(np.clip(m1_0, m1_lo * 1.01, m1_hi * 0.99))
    a1_0 = a0 / plx0
    f_astro = a1_0**3 / (constants.KEPLER_AU3_PER_MSUN_DAY2 * p0**2)
    m2_0 = solve_m2_with_inclination_msun(
        f_astro, m1_seed, 90.0, m2_min_msun=m2_lo, m2_max_msun=m2_hi
    )
    if m2_0 is None:
        # No astrometric root inside the M2 box: start just inside the violated
        # edge so the fit runs and, if the data agree, ends flagged as a bound hit.
        f_hi = m2_hi**3 / (m1_seed + m2_hi) ** 2
        m2_0 = m2_hi * 0.99 if f_astro > f_hi else m2_lo * 1.01
    k_seed = k_from_primary_orbit_kms(
        primary_orbit_au(m1_seed, m2_0, p0), p0, e0, inc0
    )

    t_all = np.array([e.mjd for e in epochs if e.instrument in instruments])
    y_all = np.array([e.rv_kms for e in epochs if e.instrument in instruments])
    yerr_all = np.array([e.rv_err_kms for e in epochs if e.instrument in instruments])
    inst_idx = np.array(
        [instruments.index(e.instrument) for e in epochs if e.instrument in instruments],
        dtype=np.int64,
    )
    log_j_lo = math.log(config.jitter_min_kms)
    log_j_hi = math.log(config.jitter_max_kms)

    n = len(JOINT_THETA_NAMES) + 2 * len(instruments)
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    lower[0] = 0.0
    lower[1:3], upper[1:3] = -1.0, 1.0
    lower[5], upper[5] = 0.0, math.pi
    lower[6], upper[6] = m1_lo, m1_hi
    lower[7], upper[7] = m2_lo, m2_hi
    lower[8] = 0.0
    scale = np.empty(n)
    scale[0] = sig["period"]
    scale[1:3] = _COND_SQRT_E
    scale[3] = sig["t_periastron"]
    scale[4:6] = _COND_ANGLE_RAD
    scale[6] = m1_sig
    scale[7] = max(_COND_M2_FRAC * m2_0, m2_lo)
    scale[8] = sig["parallax"]
    for i, inst in enumerate(instruments):
        mask = inst_idx == i
        lower[10 + 2 * i], upper[10 + 2 * i] = log_j_lo, log_j_hi
        scale[9 + 2 * i] = float(np.median(yerr_all[mask]))
        scale[10 + 2 * i] = _COND_LOG_JITTER

    problem = _JointProblem(
        t=t_all,
        y=y_all,
        yerr=yerr_all,
        inst_idx=inst_idx,
        instruments=instruments,
        nss_names=nss_names,
        nss_values=np.asarray(nss_values, dtype=np.float64),
        nss_chol=chol,
        m1_prior=m1_0,
        m1_sigma=m1_sig,
        ecc_max=float(config.joint_ecc_max),
        scale=scale,
        lower=lower,
        upper=upper,
    )

    def make_start(omega: float, node: float, p: float, e: float, t_p: float) -> NDArray[np.floating]:
        se = math.sqrt(e)
        theta = np.zeros(n)
        theta[:9] = [p, se * math.cos(omega), se * math.sin(omega), t_p, node, inc0, m1_seed, m2_0, plx0]
        model = rv_curve_kms(
            t_all,
            period_day=p,
            eccentricity=e,
            t_periastron_mjd=t_p,
            k_kms=k_seed,
            omega_rad=omega,
            gamma_kms=0.0,
        )
        margin = 0.01 * (log_j_hi - log_j_lo)
        for i, _inst in enumerate(instruments):
            mask = inst_idx == i
            w = 1.0 / yerr_all[mask] ** 2
            gamma = _weighted_mean(y_all[mask] - model[mask], w)
            jit = _profile_jitter_kms(
                y_all[mask] - model[mask] - gamma, yerr_all[mask], config.jitter_max_kms
            )
            theta[9 + 2 * i] = gamma
            theta[10 + 2 * i] = float(
                np.clip(math.log(max(jit, config.jitter_min_kms)), log_j_lo + margin, log_j_hi - margin)
            )
        return theta

    starts: dict[str, NDArray[np.floating]] = {
        "nss": make_start(omega0, node0, p0, e0, t0),
        "nss_flipped": make_start(omega0 + math.pi, node0 + math.pi, p0, e0, t0),
    }
    seed = _joker_seed_block(candidate.rv_summary or {}, config.joker_seed_variant)
    if seed is not None:
        om_seed = _omega_from_joker_block(seed)
        p_s = _finite(seed.get("P_days")) or p0
        e_s = float(np.clip(_finite(seed.get("e")) or e0, 0.0, config.joint_ecc_max * 0.99))
        t_s = _finite(seed.get("t_periastron_mjd")) or t0
        if om_seed is not None:
            # Keep the NSS node branch that matches the Joker ω (TI invariance).
            node_s = node0 if abs(float(_wrap_angle(om_seed - omega0))) < math.pi / 2 else node0 + math.pi
            starts["joker"] = make_start(om_seed, node_s, p_s, e_s, t_s)
    return problem, starts, k_seed


def fit_joint_orbit(
    candidate: CandidateRecord,
    config: RvConsistencyConfig,
) -> CandidateRecord:
    """Joint astrometry (NSS vector + covariance) + RV orbit fit for one gate passer.

    On success: ``orbit_tier = joint_astrometry_rv``; ``m2`` replaced by the fitted
    M2 marginal; ``extras["joint_orbit"]`` holds the full ``JOINT_OUTPUT_NAMES``
    ParameterSet with the Laplace covariance; ``extras["joint_orbit_fit_status"]``
    records the convergence test. On a skip, non-convergence or a bound hit: the
    upstream ``m2`` and ``astrometry_only`` tier are kept and
    ``extras["joint_orbit_fit_skip_reason"]`` names why (bound hits also list the
    parameters in the status block).
    """
    built = build_joint_problem(candidate, config)
    if isinstance(built, str):
        return _skip(candidate, built)
    problem, starts, k_seed = built
    try:
        result = fit_joint_orbit_model(problem, starts, config, k_seed_kms=k_seed)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _skip(candidate, f"optimize_failed:{type(exc).__name__}")

    extras = dict(candidate.extras)
    extras["joint_orbit_fit_status"] = result.status_dict()
    extras["joint_orbit_instruments"] = result.instruments
    extras["joint_orbit_seed"] = result.start_label
    if not result.converged or result.output_covariance is None:
        extras["joint_orbit_fit_skip_reason"] = result.failure_reason or JOINT_FAIL_NOT_CONVERGED
        extras["joint_orbit_unconverged_values"] = dict(
            zip(JOINT_OUTPUT_NAMES, (float(v) for v in result.output_values))
        )
        return candidate.model_copy(
            update={
                "orbit_tier": candidate.orbit_tier or OrbitTier.ASTROMETRY_ONLY,
                "extras": extras,
            }
        )

    cov = result.output_covariance
    orbit_ps = ParameterSet(
        names=list(JOINT_OUTPUT_NAMES),
        values=[float(v) for v in result.output_values],
        covariance=cov.tolist(),
        provenance=JOINT_PROVENANCE,
        units=list(JOINT_OUTPUT_UNITS),
    )
    i_m2 = JOINT_OUTPUT_NAMES.index("M2")
    m2_ps = ParameterSet(
        names=["M2"],
        values=[float(result.output_values[i_m2])],
        covariance=[[float(cov[i_m2, i_m2])]],
        provenance=JOINT_PROVENANCE,
        units=["Msun"],
    )
    extras.pop("joint_orbit_fit_skip_reason", None)
    extras["joint_orbit"] = orbit_ps.model_dump(mode="json")
    return candidate.model_copy(
        update={
            "orbit_tier": OrbitTier.JOINT_ASTROMETRY_RV,
            "extras": extras,
            "m2": m2_ps,
        }
    )


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
        status = updated.extras.get("joint_orbit_fit_status")
        reason = updated.extras.get("joint_orbit_fit_skip_reason")
        if isinstance(status, Mapping):
            diag.n_attempted += 1
        if reason:
            reason_s = str(reason)
            diag.skip_reasons[reason_s] = diag.skip_reasons.get(reason_s, 0) + 1
            if reason_s == JOINT_FAIL_BOUND_HIT:
                diag.n_bound_hit += 1
                diag.n_failed_optimize += 1
                for name in (status or {}).get("bound_hits", []):
                    diag.bound_hit_parameters[name] = diag.bound_hit_parameters.get(name, 0) + 1
            elif reason_s.startswith("optimize_"):
                diag.n_not_converged += 1
                diag.n_failed_optimize += 1
            else:
                diag.n_skipped_other += 1
        else:
            diag.n_fit += 1
            diag.n_converged += 1
            orbit = updated.extras["joint_orbit"]
            vals = dict(zip(orbit["names"], orbit["values"]))
            i_m2 = orbit["names"].index("M2")
            diag.fitted_source_ids.append(int(updated.source_id))
            diag.m2_joint_msun.append(float(vals["M2"]))
            diag.m2_joint_sigma_msun.append(float(math.sqrt(orbit["covariance"][i_m2][i_m2])))
            up = float("nan")
            if candidate.m2 is not None:
                try:
                    up = float(candidate.m2.marginal("M2").value)
                except KeyError:
                    pass
            diag.m2_upstream_msun.append(up)
            diag.k_joint_kms.append(float(vals["K"]))
            diag.k_astrometry_seed_kms.append(float(status["k_astrometry_seed_kms"]))  # type: ignore[index]
        out.append(updated)
    return out, diag


def format_joint_fit_report(
    diag: JointFitDiagnostics,
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> str:
    """Full-detail joint-fit quality report (exempt from caveman compression)."""
    lines = [
        "=== joint_orbit_fit quality report ===",
        "Model: RV epochs + Gaia NSS astrometric vector (full covariance) + M1 prior;",
        "K derived from a0/parallax (dark-companion photocentre). Covariance = inverse",
        "Fisher information at the optimum. Bound hits and unconverged fits keep the",
        "upstream astrometry_only mass and are NOT counted as fits.",
        f"  input_candidates: {diag.n_input}",
        f"  skipped_gate_failed: {diag.n_skipped_gate_failed}",
        f"  fits_attempted: {diag.n_attempted}",
        "  converged (Fisher identified + scoring decrement <= "
        f"{config.rv_consistency.joint_fit_decrement_tol:g} + no bound hit): "
        f"{diag.n_converged}",
        f"  not_converged: {diag.n_not_converged}",
        f"  bound_hits: {diag.n_bound_hit}",
        f"  bound_hit_parameters: {dict(sorted(diag.bound_hit_parameters.items()))}",
        f"  skipped_other: {diag.n_skipped_other}",
        f"  skip/failure reasons: {dict(sorted(diag.skip_reasons.items()))}",
    ]
    if diag.m2_joint_msun:
        arr = np.asarray(diag.m2_joint_msun, dtype=np.float64)
        lines.append(
            "  recovered M2 [Msun]: "
            f"min={float(np.min(arr)):.3f} median={float(np.median(arr)):.3f} "
            f"max={float(np.max(arr)):.3f}"
        )
    priority = set(config.rv_consistency.priority_source_ids)
    if priority:
        lines.append("  priority calibrators (rv_consistency.priority_source_ids):")
        by_id = {c.source_id: c for c in candidates}
        for sid in sorted(priority):
            cand = by_id.get(sid)
            if cand is None:
                lines.append(f"    {sid}: not in this stage's input")
                continue
            orbit = cand.extras.get("joint_orbit")
            if not isinstance(orbit, Mapping):
                lines.append(
                    f"    {sid}: no joint fit "
                    f"(reason={cand.extras.get('joint_orbit_fit_skip_reason')})"
                )
                continue
            vals = dict(zip(orbit["names"], orbit["values"]))
            sig = {
                n: math.sqrt(max(orbit["covariance"][i][i], 0.0))
                for i, n in enumerate(orbit["names"])
            }
            lines.append(
                f"    {sid}: M2={vals['M2']:.3f}±{sig['M2']:.3f} Msun "
                f"M1={vals['M1']:.3f}±{sig['M1']:.3f} K={vals['K']:.2f}±{sig['K']:.2f} km/s "
                f"P={vals['P']:.3f}±{sig['P']:.3f} d e={vals['e']:.4f}±{sig['e']:.4f} "
                f"i={math.degrees(vals['inclination']):.2f}±{math.degrees(sig['inclination']):.2f} deg"
            )
    lines.append("=== end joint_orbit_fit quality report ===")
    return "\n".join(lines)


def write_joint_diagnostic_artifacts(
    diagnostics: JointFitDiagnostics,
    candidates: Sequence[CandidateRecord],
    artifact_path: Path,
    config: PipelineConfig,
) -> list[Path]:
    """Report + figures: recovered-M2 histogram, joint-vs-upstream M2, K joint vs astrometric seed."""
    dirs = resolve_diagnostic_dirs(
        config, run_id="joint_orbit_fit", beside_artifact=artifact_path
    )
    written: list[Path] = []
    report_path = dirs.reports / "joint_orbit_fit_report.txt"
    report_path.write_text(
        format_joint_fit_report(diagnostics, candidates, config) + "\n", encoding="utf-8"
    )
    written.append(report_path)
    if not config.diagnostics.write_figures or not diagnostics.m2_joint_msun:
        return written
    dpi = int(config.diagnostics.figure_dpi)
    counts = (
        f"{diagnostics.n_converged} converged, {diagnostics.n_bound_hit} bound hit, "
        f"{diagnostics.n_not_converged} unconverged"
    )
    m2 = np.asarray(diagnostics.m2_joint_msun, dtype=np.float64)
    fig = plot_histogram(
        np.log10(m2[m2 > 0]),
        dirs.figures / "joint_orbit_fit_m2.png",
        xlabel=r"$\log_{10} M_2$ (M$_\odot$)",
        ylabel="count",
        title=f"Joint-fit M$_2$\n{counts}",
        dpi=dpi,
        max_bins=int(config.diagnostics.histogram_max_bins),
        style=config.plotting,
    )
    if fig is not None:
        written.append(fig)
    fig = plot_scatter_xy(
        np.asarray(diagnostics.m2_upstream_msun, dtype=np.float64),
        m2,
        dirs.figures / "joint_orbit_fit_m2_vs_upstream.png",
        xlabel=r"upstream $M_2$ (M$_\odot$)",
        ylabel=r"joint-fit $M_2$ (M$_\odot$)",
        title="Joint fit vs upstream M$_2$",
        dpi=dpi,
        style=config.plotting,
        yerr=np.asarray(diagnostics.m2_joint_sigma_msun, dtype=np.float64),
        log_axes=True,
        one_to_one=True,
    )
    if fig is not None:
        written.append(fig)
    fig = plot_scatter_xy(
        np.asarray(diagnostics.k_astrometry_seed_kms, dtype=np.float64),
        np.asarray(diagnostics.k_joint_kms, dtype=np.float64),
        dirs.figures / "joint_orbit_fit_k_vs_astrometric.png",
        xlabel=r"astrometry-only predicted $K$ (km s$^{-1}$)",
        ylabel=r"joint-fit $K$ (km s$^{-1}$)",
        title="Joint-fit K vs NSS-seed K",
        dpi=dpi,
        style=config.plotting,
        log_axes=True,
        one_to_one=True,
    )
    if fig is not None:
        written.append(fig)
    return written


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
        f"  rv_summary_attached: {diag.rv_summary_attached}",
        f"  rv_summary_missing: {diag.rv_summary_missing}",
        f"  rv_summary_kept_existing: {diag.rv_summary_kept_existing}",
        f"  rv_summary_disabled: {diag.rv_summary_disabled}",
        f"  epochs_rejected_mjd_below_min: {diag.n_epochs_rejected_bad_mjd} "
        f"(in {diag.n_candidates_with_rejected_epochs} candidates)",
    ]
    if diag.rv_summary_disabled == diag.n_input and diag.n_input > 0:
        lines.append(
            "  WARNING: rv_summary_root is null — attachment disabled; "
            "gate cannot score RV epochs (stage may look complete but empty)."
        )
    elif diag.n_scored == 0 and diag.n_input > 0:
        lines.append(
            "  WARNING: scored=0 with root enabled — check JSON layout "
            "(Gaia_DR3_{source_id}_summary.json) under rv_summary_root."
        )
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
    guard = plan_and_guard(
        spec, manifest, config, run_path=run_path, force_rerun=force_rerun
    )
    if not guard.proceed:
        return guard.manifest
    manifest = guard.manifest
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

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
    guard = plan_and_guard(
        spec, manifest, config, run_path=run_path, force_rerun=force_rerun
    )
    if not guard.proceed:
        return guard.manifest
    manifest = guard.manifest
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

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
    write_joint_diagnostic_artifacts(diagnostics, updated, artifact, config)

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
