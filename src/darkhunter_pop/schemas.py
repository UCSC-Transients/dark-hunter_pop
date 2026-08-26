"""Pydantic interface contracts shared across every stage.

ARCHITECTURE.md §3. v1 ``ParameterSet`` stores a covariance matrix; joint posterior samples are
documented as a future HDF5 layout only (see ``SAMPLES_HDF5_FORMAT``).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Future ParameterSet samples layout (not implemented in v1)
# ---------------------------------------------------------------------------

SAMPLES_HDF5_FORMAT: Final[str] = """
Future on-disk layout when ParameterSet switches from covariance to joint samples
(one HDF5 file per stage, ARCHITECTURE.md §3):

  /candidates/{source_id}/parameters/{provenance}/
      names      (string dataset, length N)
      samples    (float64, shape [n_draw, N])
      units      (optional string dataset, length N)
      provenance (scalar string attribute)

v1 writes covariance under the same candidate/provenance groups instead of samples.
"""

# Per-system companion-nature → population_model weight contract (issues #56 / #57).
# Keys are fixed; values are non-negative responsibilities (normalized by population_model).
COMPANION_NATURE_WEIGHT_KEYS: Final[tuple[str, ...]] = (
    "BH",
    "NS",
    "WD",
    "other",
    "outlier",
)
COMPANION_NATURE_WEIGHT_SCHEMA_VERSION: Final[int] = 1


class OrbitTier(str, Enum):
    """Orbit solution tier recorded on candidates / ParameterSet provenance metadata."""

    ASTROMETRY_ONLY = "astrometry_only"
    JOINT_ASTROMETRY_RV = "joint_astrometry_rv"


class FitTier(str, Enum):
    """Mass/SED fit tier."""

    BULK_ESTIMATE = "bulk_estimate"
    FULL_UBERMS = "full_uberMS"


class StageStatus(str, Enum):
    """Per-stage status values in the live run file."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CACHED = "cached"


class ActiveDRMode(str, Enum):
    DR3 = "dr3"
    DR4 = "dr4"


class MarginalView(BaseModel):
    """Single-parameter marginal view of a ``ParameterSet`` (value ± sigma)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    sigma: float | None = None
    unit: str | None = None


class ScalarEstimate(BaseModel):
    """Bare scalar-with-uncertainty for standalone external inputs never jointly fit.

    Prefer ``ParameterSet`` for anything co-estimated with other tracked quantities.
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    sigma: float | None = None
    unit: str | None = None
    provenance: str | None = None


class ParameterSet(BaseModel):
    """Named parameter vector + covariance (v1) with mandatory provenance metadata.

    Provenance records which method/tier produced the fit. It must never be used to inflate
    reported uncertainties by itself.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    names: list[str] = Field(..., min_length=1)
    values: list[float]
    covariance: list[list[float]]
    provenance: str = Field(..., min_length=1)
    units: list[str] | None = None
    # Future: path to HDF5 samples group; unused in v1.
    samples_hdf5_path: str | None = None

    @model_validator(mode="after")
    def _check_shapes(self) -> ParameterSet:
        n = len(self.names)
        if len(self.values) != n:
            raise ValueError(f"values length {len(self.values)} != names length {n}")
        if self.units is not None and len(self.units) != n:
            raise ValueError(f"units length {len(self.units)} != names length {n}")
        if len(self.covariance) != n:
            raise ValueError(f"covariance rows {len(self.covariance)} != names length {n}")
        for i, row in enumerate(self.covariance):
            if len(row) != n:
                raise ValueError(f"covariance row {i} length {len(row)} != {n}")
        cov = np.asarray(self.covariance, dtype=np.float64)
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        return self

    def covariance_array(self) -> NDArray[np.floating]:
        return np.asarray(self.covariance, dtype=np.float64)

    def values_array(self) -> NDArray[np.floating]:
        return np.asarray(self.values, dtype=np.float64)

    def marginal(self, name: str) -> MarginalView:
        """Return the marginal mean and sqrt(diag) uncertainty for ``name``."""
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        var = self.covariance[index][index]
        sigma = float(np.sqrt(var)) if var >= 0 else float("nan")
        unit = self.units[index] if self.units is not None else None
        return MarginalView(
            name=name, value=self.values[index], sigma=sigma, unit=unit
        )

    def get_posterior_samples(self) -> NDArray[np.floating]:
        """Stub for the future samples-backed ParameterSet.

        Raises:
            NotImplementedError: always in v1 (covariance storage only).
        """
        raise NotImplementedError(
            "ParameterSet posterior samples are not implemented in v1; "
            "see SAMPLES_HDF5_FORMAT. Use covariance / marginal() instead."
        )


class ThieleInnesElements(BaseModel):
    """Thiele–Innes elements when available from the NSS solution."""

    model_config = ConfigDict(extra="forbid")

    A: float | None = None
    B: float | None = None
    F: float | None = None
    G: float | None = None
    A_err: float | None = None
    B_err: float | None = None
    F_err: float | None = None
    G_err: float | None = None


class PhotometryPoint(BaseModel):
    """Static (single/average) photometry in one band."""

    model_config = ConfigDict(extra="forbid")

    band: str
    mag: float
    mag_err: float | None = None
    system: str | None = None


class TessBlock(BaseModel):
    """Derived TESS products only; full light curve lives at ``light_curve_path``."""

    model_config = ConfigDict(extra="forbid")

    period_day: float | None = None
    amplitude: float | None = None
    variability_flag: bool | None = None
    implied_v_rot_kms: float | None = None
    light_curve_path: str | None = None


class CandidateRecord(BaseModel):
    """One Gaia source row flowing through the pipeline.

    ``rv_summary`` holds the full dark-hunter_rv JSON summary block (Phase 1 #5 conforms to
    this schema; ask before breaking field contracts). Extra keys under ``extras`` are allowed
    for forward-compatible ingestion only.

    ``companion_nature_weights`` (issue #56 → #57 contract): when set, keys MUST be exactly
    the five population classes in :data:`COMPANION_NATURE_WEIGHT_KEYS`
    (``BH``, ``NS``, ``WD``, ``other``, ``outlier``). Values are non-negative unnormalized
    responsibilities or probabilities; ``population_model`` normalizes. Never used as a
    pre-filter — every system remains in the sample with contamination included.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: int
    nss_solution_type: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    parallax_mas: float | None = None
    thiele_innes: ThieleInnesElements | None = None
    nss_orbital: dict[str, Any] = Field(default_factory=dict)
    rv_summary: dict[str, Any] = Field(default_factory=dict)
    photometry: list[PhotometryPoint] = Field(default_factory=list)
    tess: TessBlock | None = None
    m1: ParameterSet | None = None
    m2: ParameterSet | None = None
    orbit_tier: OrbitTier | None = None
    fit_tier: FitTier | None = None
    companion_nature_weights: dict[str, float] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class InstrumentNuisance(BaseModel):
    """Per-instrument systemic velocity and jitter from the RV/astrometry gate."""

    model_config = ConfigDict(extra="forbid")

    instrument: str
    gamma_kms: float
    jitter_kms: float


class OutlierTestResult(BaseModel):
    """Output of ``rv_astrometry_gate``."""

    model_config = ConfigDict(extra="forbid")

    source_id: int
    chi2_dof: float
    threshold: float
    passed: bool
    instruments: list[InstrumentNuisance] = Field(default_factory=list)
    notes: str | None = None


class FollowUpRecord(BaseModel):
    """Target-list membership and observational summary for follow-up selection."""

    model_config = ConfigDict(extra="forbid")

    source_id: int
    target_lists: list[str] = Field(default_factory=list)
    # e.g. {"APF": "2023-04-01", ...} — derived dates tracked in-repo
    adoption_dates: dict[str, str] = Field(default_factory=dict)
    n_observations: int = 0
    time_span_day: float | None = None
    brightness_g_mag: float | None = None
    declination_deg: float | None = None
    pm_ra_mas_yr: float | None = None
    pm_dec_mas_yr: float | None = None


class StageRecord(BaseModel):
    """One stage entry in the live run file (``RunManifest``)."""

    model_config = ConfigDict(extra="forbid")

    stage_name: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    source_hash: str | None = None
    config_subset: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    code_commit: str | None = None
    force_rerun: bool = False
    reason: str | None = None
    gaiamock_mod_release: str | None = None
    gaiamock_mod_sha256: str | None = None
    gaiamock_git_commit: str | None = None


class RunManifest(BaseModel):
    """Live YAML run file under ``runs/{run_id}.yaml`` (ARCHITECTURE.md §5)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    created_at: datetime
    parent_run_id: str | None = None
    config_checksum: str = Field(..., min_length=1)
    active_dr_mode: ActiveDRMode = ActiveDRMode.DR3
    artifact_root: str = "output"
    gaiamock_mod_release: str | None = None
    gaiamock_mod_sha256: str | None = None
    gaiamock_git_commit: str | None = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _run_id_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must be non-empty")
        return value

    def is_complete(self) -> bool:
        """True if every recorded stage is completed, skipped, or cached (none pending/running/failed)."""
        if not self.stages:
            return False
        terminal = {
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
            StageStatus.CACHED,
        }
        return all(stage.status in terminal for stage in self.stages.values())

    def is_incomplete(self) -> bool:
        return not self.is_complete()
