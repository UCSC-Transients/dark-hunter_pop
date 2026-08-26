"""Pydantic config schema for the merged ``config/config.yaml``.

ARCHITECTURE.md §6–§7. Path-specific Gaia/mission keys live under ``dr3`` / ``dr4``
independently (even when values match). Shared physics/population keys live at the top level
under ``physics``, ``mass_calibration``, ``classification``, etc.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop.schemas import ActiveDRMode


class MassCalibrationMethod(str, Enum):
    TAG10 = "TAG10"
    # Reserved for future methods — raise at use site until implemented.
    EKER = "Eker"
    MTGR = "mtgr"


class CoolingTracksModel(str, Enum):
    BEDARD = "bedard"


class CoolingAtmosphere(str, Enum):
    DA = "DA"
    HE = "He"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_root: str = "output"
    # Relative to repo or absolute; gitignored raw data roots.
    data_root: str = "data"


class GaiamockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mod_release: str = "gaiamock-mod-v1"
    # Optional pins; if set, run_management refuses on mismatch with installed overlay.
    mod_sha256: str | None = None
    git_commit: str | None = None


class MassCalibrationConfig(BaseModel):
    """Choosable mass-calibration options (coefficient tables live in ``constants``)."""

    model_config = ConfigDict(extra="forbid")

    method: MassCalibrationMethod = MassCalibrationMethod.TAG10
    sigma_logM: float = Field(0.027, gt=0)
    sigma_logR: float = Field(0.014, gt=0)
    santos_correction: bool = True
    delta_M_Ch_msun: float = 0.0


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    M_MIN_msun: float = Field(1.1, gt=0)
    n_sigma_mass_cut: float = Field(2.0, gt=0)
    # Soft NS truncation prior location (choosable); not a true constant.
    M_TOV_msun: float = Field(2.2, gt=0)


class PhysicsConfig(BaseModel):
    """Shared across DR3/DR4 — genuine population/physics choices."""

    model_config = ConfigDict(extra="forbid")

    cooling_tracks: CoolingTracksModel = CoolingTracksModel.BEDARD
    cooling_atmosphere: CoolingAtmosphere = CoolingAtmosphere.DA
    cooling_tracks_path: str | None = None  # local files when present
    imf: Literal["kroupa"] = "kroupa"
    mc_noise_threshold: float = Field(0.1, gt=0)


class ExtinctionModel(str, Enum):
    """Extinction map used by gaiamock mock photometry (ARCHITECTURE.md §4)."""

    COMBINED19 = "combined19"
    NONE = "none"


class ValidationGateConfig(BaseModel):
    """El-Badry et al. (2024) six-panel mock-vs-real gate + solution-type diagnostic."""

    model_config = ConfigDict(extra="forbid")

    ks_pvalue_min: float = Field(0.01, gt=0, le=1)
    solution_type_fraction_max_abs_delta: float = Field(0.05, gt=0, le=1)
    reference_path: str | None = None


class MockPopulationConfig(BaseModel):
    """Fiducial binary used for mock injection / validation (one realization set)."""

    model_config = ConfigDict(extra="forbid")

    period_days: float = Field(1000.0, gt=0)
    Mg_tot: float = 4.0
    flux_ratio: float = Field(0.01, gt=0)
    m1_msun: float = Field(1.0, gt=0)
    m2_msun: float = Field(0.5, gt=0)
    eccentricity: float = Field(0.3, ge=0, lt=1)
    N_realizations: int = Field(100, ge=1)
    ruwe_min: float = Field(1.4, gt=0)
    skip_acceleration: bool = False
    hz_pc: float = Field(300.0, gt=0)


class SelectionFunctionAstrometricConfig(BaseModel):
    """Shared selection-function astrometric settings (not DR-path-specific)."""

    model_config = ConfigDict(extra="forbid")

    extinction_model: ExtinctionModel = ExtinctionModel.COMBINED19
    validation_gate: ValidationGateConfig = Field(default_factory=ValidationGateConfig)
    mock_population: MockPopulationConfig = Field(default_factory=MockPopulationConfig)


class DRSelectionFunctionPathConfig(BaseModel):
    """Path-specific distance window for mock injection."""

    model_config = ConfigDict(extra="forbid")

    d_min_pc: float = Field(100.0, gt=0)
    d_max_pc: float = Field(500.0, gt=0)

    @model_validator(mode="after")
    def _distance_order(self) -> DRSelectionFunctionPathConfig:
        if self.d_min_pc >= self.d_max_pc:
            raise ValueError("d_min_pc must be < d_max_pc")
        return self


class QualityCutBin(BaseModel):
    """One (magnitude, goodness-of-fit threshold) bin.

    Stars with ``g_min < G <= g_max`` (open on the left if ``g_min`` is None; open on the
    right if ``g_max`` is None) must satisfy ``gof <= gof_max``.
    """

    model_config = ConfigDict(extra="forbid")

    g_min: float | None = None
    g_max: float | None = None
    gof_max: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _bounds(self) -> QualityCutBin:
        if self.g_min is None and self.g_max is None:
            raise ValueError("quality cut bin needs g_min and/or g_max")
        if (
            self.g_min is not None
            and self.g_max is not None
            and self.g_min >= self.g_max
        ):
            raise ValueError("g_min must be < g_max")
        return self


class DRPathConfig(BaseModel):
    """Gaia-mission / path-specific configuration for one data release."""

    model_config = ConfigDict(extra="forbid")

    mission_baseline_months: float = Field(..., gt=0)
    scanning_law_id: str
    zero_point_version: str
    sed_filters: list[str] = Field(default_factory=list)
    quality_cut_bins: list[QualityCutBin] = Field(..., min_length=1)
    # Reserved for DR4 epoch capabilities; ignored when inactive.
    allow_astrometric_epoch_outliers: bool = False
    selection_function_astrometric: DRSelectionFunctionPathConfig = Field(
        default_factory=DRSelectionFunctionPathConfig
    )


class PipelineConfig(BaseModel):
    """Root merged configuration document."""

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    active_dr_mode: ActiveDRMode = ActiveDRMode.DR3
    gaiamock: GaiamockConfig = Field(default_factory=GaiamockConfig)
    mass_calibration: MassCalibrationConfig = Field(
        default_factory=MassCalibrationConfig
    )
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    selection_function_astrometric: SelectionFunctionAstrometricConfig = Field(
        default_factory=SelectionFunctionAstrometricConfig
    )
    dr3: DRPathConfig
    dr4: DRPathConfig

    def active_dr(self) -> DRPathConfig:
        if self.active_dr_mode is ActiveDRMode.DR3:
            return self.dr3
        if self.active_dr_mode is ActiveDRMode.DR4:
            return self.dr4
        raise ValueError(f"unsupported active_dr_mode: {self.active_dr_mode}")


# Keys included in the resume/amend config checksum (ARCHITECTURE.md §5).
SHARED_CHECKSUM_SECTIONS: tuple[str, ...] = (
    "mass_calibration",
    "classification",
    "physics",
    "gaiamock",
    "paths",
    "selection_function_astrometric",
)


def checksum_payload(config: PipelineConfig) -> dict[str, Any]:
    """Active DR subtree + shared physics/population sections (not the inactive DR)."""
    data = config.model_dump(mode="json")
    active_key = config.active_dr_mode.value
    payload: dict[str, Any] = {
        "active_dr_mode": active_key,
        "active_dr": data[active_key],
    }
    for section in SHARED_CHECKSUM_SECTIONS:
        payload[section] = data[section]
    return payload
