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


class SensitivityAnalysisConfig(BaseModel):
    """Unified dimensionality + per-class covariate sensitivity (ARCHITECTURE.md §4).

    The MC/Poisson noise gate reads ``physics.mc_noise_threshold``; this section holds
    stage-specific design knobs only. Outputs are recommendation artifacts for
    ``population_model`` / ``inference`` — those modules' defaults are not rewritten here.
    """

    model_config = ConfigDict(extra="forbid")

    population_classes: list[str] = Field(
        default_factory=lambda: ["BH", "NS", "WD", "other", "outlier"],
        min_length=1,
    )
    candidate_covariates: list[str] = Field(
        default_factory=lambda: ["ruwe", "eccentricity", "period_day"],
        min_length=1,
    )
    joint_dimensions: list[str] = Field(
        default_factory=lambda: ["mass_msun", "period_day", "eccentricity"],
        min_length=1,
    )
    bic_delta_include_covariate: float = Field(6.0, gt=0)
    bic_delta_prefer_joint_nd: float = Field(10.0, gt=0)
    mean_count_per_bin_unbinned_preference: float = Field(5.0, gt=0)
    n_mass_bins: int = Field(8, ge=2)
    mass_min_msun: float = Field(0.2, gt=0)
    mass_max_msun: float = Field(20.0, gt=0)
    fiducial_expected_counts: list[float] = Field(
        default_factory=lambda: [5.0, 8.0, 12.0, 10.0, 6.0, 4.0, 2.0, 1.0],
        min_length=2,
    )
    n_mock_start: int = Field(50, ge=1)
    n_mock_max: int = Field(10000, ge=1)
    n_mock_growth_factor: float = Field(2.0, gt=1.0)
    n_synthetic_systems: int = Field(400, ge=10)
    random_seed: int = 38

    @model_validator(mode="after")
    def _sensitivity_bounds(self) -> SensitivityAnalysisConfig:
        if self.mass_min_msun >= self.mass_max_msun:
            raise ValueError("mass_min_msun must be < mass_max_msun")
        if len(self.fiducial_expected_counts) != self.n_mass_bins:
            raise ValueError(
                "fiducial_expected_counts length must equal n_mass_bins"
            )
        if any(c < 0 for c in self.fiducial_expected_counts):
            raise ValueError("fiducial_expected_counts must be non-negative")
        if self.n_mock_start > self.n_mock_max:
            raise ValueError("n_mock_start must be <= n_mock_max")
        if self.joint_dimensions[0] != "mass_msun":
            raise ValueError("joint_dimensions[0] must be 'mass_msun'")
        return self


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


class SurveyTier(str, Enum):
    """Data-source tier for follow-up RV provenance."""

    DOCUMENTED = "documented"
    AD_HOC = "ad_hoc"
    DEDICATED_CAMPAIGN = "dedicated_campaign"


class TargetListConfig(BaseModel):
    """One named follow-up target list with tracked adoption dates."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    adoption_dates_path: str
    cooler_star_preference: bool = True


class MajorSurveySFConfig(BaseModel):
    """Major spectroscopic survey with a documented selection function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    selection_function_path: str
    tier: SurveyTier = SurveyTier.DOCUMENTED


class AdHocLiteratureConfig(BaseModel):
    """Approximation for ad hoc literature RVs with unknown selection functions."""

    model_config = ConfigDict(extra="forbid")

    use_brightness: bool = True
    use_declination: bool = True
    use_proper_motion: bool = True
    pm_total_min_mas_yr: float = Field(0.0, ge=0)
    base_probability: float = Field(0.05, gt=0, le=1)


class FollowupCalibrationConfig(BaseModel):
    """Mock-vs-real histogram calibration for N_obs and follow-up time span."""

    model_config = ConfigDict(extra="forbid")

    n_obs_bin_edges: list[float] = Field(..., min_length=2)
    time_span_day_bin_edges: list[float] = Field(..., min_length=2)
    ks_pvalue_min: float = Field(0.01, gt=0, le=1)
    real_followup_catalog_path: str | None = None


class TargetListSheetConfig(BaseModel):
    """Google Sheet credentials *names* and snapshot/derived paths (no secrets)."""

    model_config = ConfigDict(extra="forbid")

    spreadsheet_id: str = ""
    sheet_range: str = "Sheet1"
    credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS"
    revision_history_incompleteness_caveat: bool = True
    weekly_snapshot_relative_dir: str = "target_lists/snapshots"
    derived_adoption_dates_relative_dir: str = "config/target_lists/derived"


class SelectionFunctionFollowupConfig(BaseModel):
    """Shared parametric follow-up selection function (ARCHITECTURE.md §4)."""

    model_config = ConfigDict(extra="forbid")

    target_lists: list[TargetListConfig] = Field(default_factory=list)
    declination_min_deg: float = -35.0
    declination_max_deg: float = 90.0
    g_mag_bright_limit: float = 5.0
    g_mag_faint_limit: float = 15.0
    cooler_star_teff_max_k: float = Field(6500.0, gt=0)
    cooler_star_weight: float = Field(1.5, gt=0)
    major_surveys: list[MajorSurveySFConfig] = Field(default_factory=list)
    ad_hoc_literature: AdHocLiteratureConfig = Field(
        default_factory=AdHocLiteratureConfig
    )
    calibration: FollowupCalibrationConfig = Field(
        default_factory=lambda: FollowupCalibrationConfig(
            n_obs_bin_edges=[0, 1, 2, 3, 5, 10, 20, 50],
            time_span_day_bin_edges=[0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1500.0],
        )
    )
    target_list_sheet: TargetListSheetConfig = Field(
        default_factory=TargetListSheetConfig
    )

    @model_validator(mode="after")
    def _limits_ordered(self) -> SelectionFunctionFollowupConfig:
        if self.declination_min_deg >= self.declination_max_deg:
            raise ValueError("declination_min_deg must be < declination_max_deg")
        if self.g_mag_bright_limit >= self.g_mag_faint_limit:
            raise ValueError("g_mag_bright_limit must be < g_mag_faint_limit")
        return self


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


class DRSelectionFunctionFollowupPathConfig(BaseModel):
    """Path-specific follow-up catalog pins (accel/jerk catalogs differ by DR)."""

    model_config = ConfigDict(extra="forbid")

    accel_jerk_catalog_id: str = "dr3_accel_jerk_pinned"


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


class ExternalPhotometryCrossmatch(BaseModel):
    """One external-band cross-match via a Gaia ``*_best_neighbour`` table."""

    model_config = ConfigDict(extra="forbid")

    band: str = Field(..., min_length=1)
    neighbour_table: str = Field(..., min_length=1)
    catalog_table: str = Field(..., min_length=1)
    mag_column: str = Field(..., min_length=1)
    mag_err_column: str | None = None
    enabled: bool = True


class DRPathConfig(BaseModel):
    """Gaia-mission / path-specific configuration for one data release."""

    model_config = ConfigDict(extra="forbid")

    mission_baseline_months: float = Field(..., gt=0)
    scanning_law_id: str
    zero_point_version: str
    sed_filters: list[str] = Field(default_factory=list)
    quality_cut_bins: list[QualityCutBin] = Field(..., min_length=1)
    gaia_source_photometry_bands: list[str] = Field(
        default_factory=lambda: ["G", "BP", "RP"]
    )
    external_photometry_crossmatches: list[ExternalPhotometryCrossmatch] = Field(
        default_factory=list
    )
    gaia_archive_user_env: str = "GAIA_ARCHIVE_USER"
    gaia_archive_password_env: str = "GAIA_ARCHIVE_PASSWORD"
    nss_table: str = "gaiadr3.nss_two_body_orbit"
    gaia_source_table: str = "gaiadr3.gaia_source"
    # Reserved for DR4 epoch capabilities; ignored when inactive.
    allow_astrometric_epoch_outliers: bool = False
    selection_function_astrometric: DRSelectionFunctionPathConfig = Field(
        default_factory=DRSelectionFunctionPathConfig
    )
    selection_function_followup: DRSelectionFunctionFollowupPathConfig = Field(
        default_factory=DRSelectionFunctionFollowupPathConfig
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
    selection_function_followup: SelectionFunctionFollowupConfig = Field(
        default_factory=SelectionFunctionFollowupConfig
    )
    sensitivity_analysis: SensitivityAnalysisConfig = Field(
        default_factory=SensitivityAnalysisConfig
    )
    dr3: DRPathConfig
    dr4: DRPathConfig

    def active_dr(self) -> DRPathConfig:
        if self.active_dr_mode is ActiveDRMode.DR3:
            return self.dr3
        if self.active_dr_mode is ActiveDRMode.DR4:
            return self.dr4
        raise ValueError(f"unsupported active_dr_mode: {self.active_dr_mode}")


# Gaia-mission / external-data-use leaf keys that MUST live under ``dr3`` / ``dr4``
# independently (ARCHITECTURE.md §6; dark-hunter-pop-workflow §6).
PATH_SPECIFIC_LEAF_KEYS: frozenset[str] = frozenset(
    {
        "mission_baseline_months",
        "scanning_law_id",
        "zero_point_version",
        "sed_filters",
        "quality_cut_bins",
        "gaia_source_photometry_bands",
        "external_photometry_crossmatches",
        "gaia_archive_user_env",
        "gaia_archive_password_env",
        "nss_table",
        "gaia_source_table",
        "allow_astrometric_epoch_outliers",
        "accel_jerk_catalog_id",
        "d_min_pc",
        "d_max_pc",
    }
)

# Genuine physics/population sections that MUST be single shared top-level keys.
SHARED_PHYSICS_SECTIONS: frozenset[str] = frozenset(
    {
        "mass_calibration",
        "classification",
        "physics",
    }
)

# Keys included in the resume/amend config checksum (ARCHITECTURE.md §5).
SHARED_CHECKSUM_SECTIONS: tuple[str, ...] = (
    "mass_calibration",
    "classification",
    "physics",
    "gaiamock",
    "paths",
    "selection_function_astrometric",
    "selection_function_followup",
    "sensitivity_analysis",
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
