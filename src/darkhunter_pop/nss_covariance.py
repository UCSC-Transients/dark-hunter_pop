"""Gaia DR3 NSS ``corr_vec`` / ``bit_index`` → covariance reconstruction.

Implements the published ``nss_two_body_orbit`` packing (Gaia DR3 datamodel
§20.6.1; Halbwachs et al. 2023 / COSMOS NSS Tools layout):

* ``bit_index`` has ``N+1`` bits; the MSB is always 1; the remaining ``N`` bits
  (MSB→LSB) flag which of the solution-type's ordered parameters were fitted.
* ``corr_vec`` is the upper triangle of the correlation matrix of the *fitted*
  parameters, column-major, length ``n(n-1)/2``, padded in the archive to 231.
* Covariance: ``C_ij = corr_ij * sigma_i * sigma_j`` (diagonal = ``sigma^2``).

No diagonal-only fallback: failures are recorded and leave ``nss_solution`` unset.
§15 Q4 (archive columns vs. staged local matrices for every solution type) is
escalated, not decided here — Eclipsing* parameter-bit layouts remain unverified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.schemas import ParameterSet

# Archive corr_vec width for nss_two_body_orbit (22 choose 2).
NSS_CORR_VEC_LENGTH: Final[int] = 231

# Numerical tolerance for symmetry / PSD checks on reconstructed matrices.
_COV_SYMM_RTOL: Final[float] = 1e-10
_COV_SYMM_ATOL: Final[float] = 1e-12
_COV_PSD_EIG_ATOL: Final[float] = 1e-10

NSS_SOLUTION_PROVENANCE_PREFIX: Final[str] = "gaiadr3.nss_two_body_orbit"

# ---------------------------------------------------------------------------
# Per-solution-type bit_index parameter order (Gaia DR3 datamodel §20.6.1).
# Length N drives bit_index width (N+1 bits). Never hardcode fitted size.
# ---------------------------------------------------------------------------

_ORBITAL: Final[tuple[str, ...]] = (
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "eccentricity",
    "period",
    "t_periastron",
)

# OrbitalAlternative* / OrbitalTargetedSearch*: period before eccentricity.
_ORBITAL_ALT: Final[tuple[str, ...]] = (
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "period",
    "eccentricity",
    "t_periastron",
)

_SB1: Final[tuple[str, ...]] = (
    "period",
    "center_of_mass_velocity",
    "semi_amplitude_primary",
    "eccentricity",
    "arg_periastron",
    "t_periastron",
)

_SB1C: Final[tuple[str, ...]] = (
    "period",
    "center_of_mass_velocity",
    "semi_amplitude_primary",
    "t_periastron",
)

_SB2: Final[tuple[str, ...]] = (
    "period",
    "center_of_mass_velocity",
    "semi_amplitude_primary",
    "semi_amplitude_secondary",
    "eccentricity",
    "arg_periastron",
    "t_periastron",
)

_SB2C: Final[tuple[str, ...]] = (
    "period",
    "center_of_mass_velocity",
    "semi_amplitude_primary",
    "semi_amplitude_secondary",
    "t_periastron",
)

_ASTRO_SPECTRO_SB1: Final[tuple[str, ...]] = (
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "a_thiele_innes",
    "b_thiele_innes",
    "f_thiele_innes",
    "g_thiele_innes",
    "c_thiele_innes",
    "h_thiele_innes",
    "center_of_mass_velocity",
    "eccentricity",
    "period",
    "t_periastron",
)

# Value units for ParameterSet metadata (Gaia archive). Note: ra/dec values are
# degrees while ra_error/dec_error are mas — covariance uses error units.
_PARAM_UNITS: Final[dict[str, str]] = {
    "ra": "deg",
    "dec": "deg",
    "parallax": "mas",
    "pmra": "mas/yr",
    "pmdec": "mas/yr",
    "a_thiele_innes": "mas",
    "b_thiele_innes": "mas",
    "f_thiele_innes": "mas",
    "g_thiele_innes": "mas",
    "c_thiele_innes": "AU",
    "h_thiele_innes": "AU",
    "period": "day",
    "t_periastron": "day",
    "eccentricity": "1",
    "center_of_mass_velocity": "km/s",
    "semi_amplitude_primary": "km/s",
    "semi_amplitude_secondary": "km/s",
    "arg_periastron": "deg",
    "mass_ratio": "1",
    "fill_factor_primary": "1",
    "fill_factor_secondary": "1",
    "inclination": "deg",
    "temperature_ratio": "1",
}

# ADQL aliases used by data_acquisition for Thiele–Innes columns.
_VALUE_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "a_thiele_innes": ("a_thiele_innes", "A"),
    "b_thiele_innes": ("b_thiele_innes", "B"),
    "f_thiele_innes": ("f_thiele_innes", "F"),
    "g_thiele_innes": ("g_thiele_innes", "G"),
}

_ERROR_COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "a_thiele_innes": ("a_thiele_innes_error", "A_error"),
    "b_thiele_innes": ("b_thiele_innes_error", "B_error"),
    "f_thiele_innes": ("f_thiele_innes_error", "F_error"),
    "g_thiele_innes": ("g_thiele_innes_error", "G_error"),
}


class CovarianceFailure(str, Enum):
    """Why an NSS covariance could not be used (no silent diagonal fallback)."""

    MISSING_CORR = "missing_corr"
    UNSUPPORTED_SOLUTION_TYPE = "unsupported_solution_type"
    BIT_INDEX_MISMATCH = "bit_index_mismatch"
    UNPACK_FAILED = "unpack_failed"
    MISSING_VALUES = "missing_values"
    MISSING_ERRORS = "missing_errors"
    NON_SYMMETRIC = "non_symmetric"
    NON_PSD = "non_psd"


@dataclass(frozen=True)
class CovarianceResult:
    """Outcome of reconstructing one NSS solution covariance."""

    parameter_set: ParameterSet | None
    status: CovarianceFailure | None
    n_param: int = 0
    bit_index: int | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.parameter_set is not None and self.status is None


@dataclass
class CovarianceHealth:
    """Funnel counters for NSS covariance reconstruction (by failure mode)."""

    ok: int = 0
    missing_corr: int = 0
    unsupported_solution_type: int = 0
    bit_index_mismatch: int = 0
    unpack_failed: int = 0
    missing_values: int = 0
    missing_errors: int = 0
    non_symmetric: int = 0
    non_psd: int = 0
    by_solution_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, solution_type: str | None, result: CovarianceResult) -> None:
        label = solution_type or "unknown"
        bucket = self.by_solution_type.setdefault(label, {})
        if result.ok:
            self.ok += 1
            bucket["ok"] = bucket.get("ok", 0) + 1
            return
        assert result.status is not None
        attr = result.status.value
        setattr(self, attr, getattr(self, attr) + 1)
        bucket[attr] = bucket.get(attr, 0) + 1

    @property
    def failed(self) -> int:
        return (
            self.missing_corr
            + self.unsupported_solution_type
            + self.bit_index_mismatch
            + self.unpack_failed
            + self.missing_values
            + self.missing_errors
            + self.non_symmetric
            + self.non_psd
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "covariance_ok": self.ok,
            "covariance_failed": self.failed,
            "covariance_missing_corr": self.missing_corr,
            "covariance_unsupported_solution_type": self.unsupported_solution_type,
            "covariance_bit_index_mismatch": self.bit_index_mismatch,
            "covariance_unpack_failed": self.unpack_failed,
            "covariance_missing_values": self.missing_values,
            "covariance_missing_errors": self.missing_errors,
            "covariance_non_symmetric": self.non_symmetric,
            "covariance_non_psd": self.non_psd,
        }

    def by_type_lines(self) -> list[str]:
        lines: list[str] = []
        for sol_type in sorted(self.by_solution_type):
            counts = self.by_solution_type[sol_type]
            parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"{sol_type}: {parts}")
        return lines


def model_param_names(nss_solution_type: str | None) -> tuple[str, ...] | None:
    """Return the ordered bit_index parameter list for ``nss_solution_type``.

    Returns ``None`` when the type's bit layout is not yet verified against the
    archive (escalated as CONTINUATION_PLAN §15 Q4) — callers must record a
    failure, not invent a diagonal covariance.
    """
    if not nss_solution_type:
        return None
    if nss_solution_type == "Orbital":
        return _ORBITAL
    if nss_solution_type.startswith("OrbitalAlternative") or nss_solution_type.startswith(
        "OrbitalTargetedSearch"
    ):
        return _ORBITAL_ALT
    if nss_solution_type == "SB1":
        return _SB1
    if nss_solution_type == "SB1C":
        return _SB1C
    if nss_solution_type == "SB2":
        return _SB2
    if nss_solution_type == "SB2C":
        return _SB2C
    if nss_solution_type == "AstroSpectroSB1":
        return _ASTRO_SPECTRO_SB1
    # EclipsingBinary / EclipsingSpectro: 20-parameter bit tables; Q4 open.
    return None


def fitted_params_from_bit_index(
    bit_index: int,
    model_params: Sequence[str],
) -> list[str]:
    """Select fitted parameter names from ``bit_index`` and the model order.

    ``bit_index`` has ``len(model_params)+1`` bits: MSB always 1; remaining bits
    MSB→LSB map onto ``model_params`` in documented order.
    """
    n = len(model_params)
    if n < 1:
        raise ValueError("model_params must be non-empty")
    bit_length = int(bit_index).bit_length()
    if bit_length != n + 1:
        raise ValueError(
            f"bit_index={bit_index} has bit_length={bit_length}, "
            f"expected {n + 1} for {n}-parameter model"
        )
    msb = 1 << n
    if not (int(bit_index) & msb):
        raise ValueError(f"bit_index={bit_index} leading bit must be set")
    fitted: list[str] = []
    for i, name in enumerate(model_params):
        bit = n - 1 - i
        if int(bit_index) & (1 << bit):
            fitted.append(name)
    return fitted


def pack_corr_vec_upper_triangle(corr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pack a correlation matrix to Gaia's column-major upper-triangle vector.

    Diagonal is omitted (implicitly 1). Trailing zeros pad to
    :data:`NSS_CORR_VEC_LENGTH`. Used for round-trip tests against the published
    layout.
    """
    matrix = np.asarray(corr, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"corr must be square, got shape {matrix.shape}")
    n = matrix.shape[0]
    out = np.zeros(NSS_CORR_VEC_LENGTH, dtype=np.float64)
    k = 0
    for j in range(n):
        for i in range(j):
            out[k] = matrix[i, j]
            k += 1
    return out


def unpack_corr_vec_upper_triangle(
    corr_vec: Sequence[float] | NDArray[np.floating],
    n: int,
) -> NDArray[np.floating]:
    """Unpack Gaia's column-major upper-triangle ``corr_vec`` into an ``n×n`` matrix.

    Uses the first ``n(n-1)/2`` elements (fitted-parameter packing). Does **not**
    drop exact zeros — that would misalign the packing.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    n_corr = n * (n - 1) // 2
    raw = np.asarray(corr_vec, dtype=np.float64).ravel()
    if raw.size < n_corr:
        raise ValueError(
            f"corr_vec length {raw.size} < required {n_corr} for n={n}"
        )
    vals = raw[:n_corr]
    if not np.all(np.isfinite(vals)):
        raise ValueError("corr_vec contains non-finite values in the fitted block")
    corr = np.eye(n, dtype=np.float64)
    k = 0
    for j in range(n):
        for i in range(j):
            corr[i, j] = vals[k]
            corr[j, i] = vals[k]
            k += 1
    return corr


def correlation_to_covariance(
    corr: NDArray[np.floating],
    errors: Sequence[float],
) -> NDArray[np.floating]:
    """``C_ij = corr_ij * sigma_i * sigma_j`` with diagonal ``sigma_i^2``."""
    sig = np.asarray(errors, dtype=np.float64)
    if sig.ndim != 1:
        raise ValueError("errors must be 1-D")
    matrix = np.asarray(corr, dtype=np.float64)
    if matrix.shape != (sig.size, sig.size):
        raise ValueError(
            f"corr shape {matrix.shape} incompatible with {sig.size} errors"
        )
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("errors must be finite and positive")
    cov = matrix * np.outer(sig, sig)
    np.fill_diagonal(cov, sig * sig)
    return cov


def assert_symmetric_psd(
    cov: NDArray[np.floating],
    *,
    rtol: float = _COV_SYMM_RTOL,
    atol: float = _COV_SYMM_ATOL,
    eig_atol: float = _COV_PSD_EIG_ATOL,
) -> CovarianceFailure | None:
    """Return a failure enum if ``cov`` is not symmetric or not PSD; else ``None``."""
    matrix = np.asarray(cov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return CovarianceFailure.UNPACK_FAILED
    if not np.allclose(matrix, matrix.T, rtol=rtol, atol=atol):
        return CovarianceFailure.NON_SYMMETRIC
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -eig_atol * scale:
        return CovarianceFailure.NON_PSD
    return None


def _optional_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if hasattr(value, "mask"):
            try:
                if bool(np.ma.is_masked(value)):
                    continue
            except (TypeError, ValueError):
                pass
        try:
            if np.ma.is_masked(value):
                continue
        except (TypeError, ValueError):
            pass
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(out):
            continue
        return out
    return None


def _lookup_value(row: Mapping[str, Any], param: str) -> float | None:
    aliases = _VALUE_COLUMN_ALIASES.get(param, (param,))
    return _optional_float(row, aliases)


def _lookup_error(row: Mapping[str, Any], param: str) -> float | None:
    aliases = _ERROR_COLUMN_ALIASES.get(param, (f"{param}_error",))
    return _optional_float(row, aliases)


def _parse_corr_vec(raw: Any) -> NDArray[np.floating] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text.strip():
            return None
        try:
            return np.fromstring(text, dtype=np.float64, sep=",")
        except ValueError:
            return None
    if isinstance(raw, np.ma.MaskedArray):
        data = np.asarray(raw.filled(np.nan), dtype=np.float64).ravel()
        return data
    try:
        arr = np.asarray(raw, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _parse_bit_index(raw: Any) -> int | None:
    if raw is None:
        return None
    if hasattr(raw, "mask"):
        try:
            if bool(np.ma.is_masked(raw)):
                return None
        except (TypeError, ValueError):
            pass
    try:
        if np.ma.is_masked(raw):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def reconstruct_nss_covariance(
    row: Mapping[str, Any],
    *,
    nss_solution_type: str | None = None,
) -> CovarianceResult:
    """Rebuild the NSS fitted-parameter covariance for one archive row.

    Driven by ``bit_index`` (never a hardcoded parameter count). On any failure,
    returns ``parameter_set=None`` with a :class:`CovarianceFailure` — never a
    diagonal-only substitute.
    """
    solution_type = nss_solution_type
    if solution_type is None:
        raw_type = row.get("nss_solution_type")
        if hasattr(raw_type, "item"):
            raw_type = raw_type.item()
        solution_type = str(raw_type) if raw_type is not None else None

    corr_raw = _parse_corr_vec(row.get("corr_vec"))
    bit_index = _parse_bit_index(row.get("bit_index"))
    if corr_raw is None or bit_index is None:
        return CovarianceResult(
            parameter_set=None,
            status=CovarianceFailure.MISSING_CORR,
            bit_index=bit_index,
            detail="corr_vec and/or bit_index absent",
        )

    model = model_param_names(solution_type)
    if model is None:
        return CovarianceResult(
            parameter_set=None,
            status=CovarianceFailure.UNSUPPORTED_SOLUTION_TYPE,
            bit_index=bit_index,
            detail=f"no verified bit_index layout for {solution_type!r} (§15 Q4)",
        )

    try:
        fitted = fitted_params_from_bit_index(bit_index, model)
    except ValueError as exc:
        return CovarianceResult(
            parameter_set=None,
            status=CovarianceFailure.BIT_INDEX_MISMATCH,
            bit_index=bit_index,
            detail=str(exc),
        )

    n = len(fitted)
    if n < 1:
        return CovarianceResult(
            parameter_set=None,
            status=CovarianceFailure.BIT_INDEX_MISMATCH,
            bit_index=bit_index,
            detail="bit_index selected zero fitted parameters",
        )

    values: list[float] = []
    errors: list[float] = []
    for name in fitted:
        value = _lookup_value(row, name)
        if value is None:
            return CovarianceResult(
                parameter_set=None,
                status=CovarianceFailure.MISSING_VALUES,
                n_param=n,
                bit_index=bit_index,
                detail=f"missing value for {name}",
            )
        err = _lookup_error(row, name)
        if err is None or err <= 0.0:
            return CovarianceResult(
                parameter_set=None,
                status=CovarianceFailure.MISSING_ERRORS,
                n_param=n,
                bit_index=bit_index,
                detail=f"missing/non-positive error for {name}",
            )
        values.append(value)
        errors.append(err)

    try:
        corr = unpack_corr_vec_upper_triangle(corr_raw, n)
        cov = correlation_to_covariance(corr, errors)
    except ValueError as exc:
        return CovarianceResult(
            parameter_set=None,
            status=CovarianceFailure.UNPACK_FAILED,
            n_param=n,
            bit_index=bit_index,
            detail=str(exc),
        )

    health = assert_symmetric_psd(cov)
    if health is not None:
        return CovarianceResult(
            parameter_set=None,
            status=health,
            n_param=n,
            bit_index=bit_index,
            detail=f"covariance failed {health.value}",
        )

    units = [_PARAM_UNITS.get(name, "1") for name in fitted]
    provenance = f"{NSS_SOLUTION_PROVENANCE_PREFIX}:{solution_type}"
    parameter_set = ParameterSet(
        names=list(fitted),
        values=values,
        covariance=cov.tolist(),
        provenance=provenance,
        units=units,
    )
    return CovarianceResult(
        parameter_set=parameter_set,
        status=None,
        n_param=n,
        bit_index=bit_index,
    )


def validate_loaded_nss_solution(
    parameter_set: ParameterSet,
) -> CovarianceFailure | None:
    """Re-check symmetry and PSD for a ``ParameterSet`` loaded from HDF5."""
    return assert_symmetric_psd(parameter_set.covariance_array())
