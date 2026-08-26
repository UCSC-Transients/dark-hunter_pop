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


class DiagnosticsHooksConfig(BaseModel):
    """Enable flags for shared diagnostic emitters (layout hooks + Phase 6 SBC)."""

    model_config = ConfigDict(extra="forbid")

    funnel_sky: bool = True
    elbadry_six_panel: bool = True
    fit_tier_coverage: bool = True
    gate_pass_rate: bool = True
    sbc_recovery: bool = True


class InjectedMassFunctionProfile(BaseModel):
    """One distinct injected free-height mass-function profile for SBC."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    relative_heights: list[float] = Field(..., min_length=2)

    @model_validator(mode="after")
    def _positive_heights(self) -> InjectedMassFunctionProfile:
        if any(h <= 0.0 for h in self.relative_heights):
            raise ValueError(
                f"injected profile {self.name!r}: relative_heights must be > 0"
            )
        return self


class SBCConfig(BaseModel):
    """Simulation-based calibration recovery + credible-interval coverage (issue #69).

    Tolerances and injection knobs live here — never hardcoded in ``sbc.py``.
    ``analytic_binned`` is the fast unit/physics path; ``dynesty`` exercises the
    staged inference sampler (prefer ``@pytest.mark.slow`` for multi-repeat suites).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # When False, diagnostics stage skips the (potentially long) SBC suite;
    # tests and explicit callers still invoke ``run_sbc_suite`` directly.
    run_in_stage: bool = False
    recovery_backend: Literal["analytic_binned", "dynesty"] = "analytic_binned"
    credible_interval_level: float = Field(0.68, gt=0.0, lt=1.0)
    coverage_abs_tolerance: float = Field(0.20, gt=0.0, le=1.0)
    n_repeats: int = Field(24, ge=1)
    n_mass_bins: int = Field(4, ge=2)
    expected_total_rate: float = Field(40.0, gt=0.0)
    astrometric_sf: float = Field(1.0, gt=0.0)
    followup_sf: float = Field(1.0, gt=0.0)
    random_seed: int = 69
    # Analytic posterior Monte Carlo draws per bin (coverage estimator).
    n_posterior_samples: int = Field(2000, ge=64)
    # Dynesty overrides when recovery_backend == dynesty (CI/slow keep these small).
    inference_nlive: int = Field(12, ge=2)
    inference_maxcall: int = Field(200, ge=10)
    inference_dlogz: float = Field(1.0, gt=0.0)
    inference_n_mass_grid: int = Field(24, ge=8)
    injected_profiles: list[InjectedMassFunctionProfile] = Field(
        default_factory=lambda: [
            InjectedMassFunctionProfile(
                name="flat", relative_heights=[1.0, 1.0, 1.0, 1.0]
            ),
            InjectedMassFunctionProfile(
                name="rising", relative_heights=[0.5, 1.0, 2.0, 3.0]
            ),
            InjectedMassFunctionProfile(
                name="falling", relative_heights=[3.0, 2.0, 1.0, 0.5]
            ),
            InjectedMassFunctionProfile(
                name="peaked", relative_heights=[0.5, 2.5, 2.5, 0.5]
            ),
        ],
        min_length=1,
    )

    @model_validator(mode="after")
    def _profile_bin_lengths(self) -> SBCConfig:
        for profile in self.injected_profiles:
            if len(profile.relative_heights) != self.n_mass_bins:
                raise ValueError(
                    f"injected profile {profile.name!r}: "
                    f"len(relative_heights)={len(profile.relative_heights)} "
                    f"must equal n_mass_bins={self.n_mass_bins}"
                )
        return self


class DiagnosticsConfig(BaseModel):
    """Rendering / report layout for ``plotting`` + ``diagnostics`` (issue #39).

    Layout / DPI / hook flags stay non-checksum. Phase 6 SBC recovery tolerances
    live under ``sbc`` (issue #69) — still excluded from ``SHARED_CHECKSUM_SECTIONS``.
    Stage science thresholds (KS, chi2/dof, MC noise) remain in owning stage configs.
    """

    model_config = ConfigDict(extra="forbid")

    figure_dpi: int = Field(120, ge=36, le=600)
    figures_subdir: str = "figures"
    reports_subdir: str = "reports"
    write_figures: bool = True
    write_reports: bool = True
    hooks: DiagnosticsHooksConfig = Field(default_factory=DiagnosticsHooksConfig)
    sbc: SBCConfig = Field(default_factory=SBCConfig)


class TriplesConfig(BaseModel):
    """Unrelated outer-companion (genuine triple) stage (ARCHITECTURE.md §4).

    Off by default in v1; ``population_model`` forces P(triple)=0. Channel flags
    reserve TESS variability + rotation-consistency hooks so the stage can be
    enabled later without restructuring. No science thresholds are applied while
    the stage remains a stub.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    tess_variability_channel: bool = True
    rotation_consistency_channel: bool = True


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


class MassDerivationConfig(BaseModel):
    """Choosables for bulk TAG10→M2 and refined uberMS queue stages."""

    model_config = ConfigDict(extra="forbid")

    # Flux ratio F2/F1 for gaiamock mass-function inversion (0 = dark companion).
    dark_companion_flux_ratio: float = Field(0.0, ge=0)
    # uberMS initial_Mass prior upper edge (watch-list when M1 approaches this).
    uberms_m1_prior_max_msun: float = Field(3.0, gt=0)
    # Flag when M1 >= fraction * uberms_m1_prior_max_msun.
    uberms_m1_watchlist_fraction: float = Field(0.95, gt=0, le=1)
    # Floor for relative-uncertainty information-gain stub (avoid /0).
    information_gain_sigma_floor_msun: float = Field(0.01, gt=0)
    # Cap queue length; null = process all candidates passing the bulk cut.
    sed_queue_max_stars: int | None = Field(default=None, ge=1)
    # When true, refined stage raises if darkhunter_sed is not importable.
    require_sed_package: bool = False


class RvConsistencyConfig(BaseModel):
    """Choosables for ``rv_astrometry_gate`` and ``joint_orbit_fit`` (ARCHITECTURE.md §4)."""

    model_config = ConfigDict(extra="forbid")

    # Whole-curve chi2/dof threshold; above → gate fail / outlier path.
    chi2_dof_threshold: float = Field(5.0, gt=0)
    # Minimum usable RV epochs per instrument for a γ+jitter nuisance fit.
    min_epochs_per_instrument: int = Field(3, ge=2)
    # Minimum total epochs across instruments to score the gate.
    min_epochs_total: int = Field(3, ge=2)
    # Upper bound when profiling homoscedastic jitter (km/s).
    jitter_max_kms: float = Field(10.0, gt=0)
    # Relative |ΔP|/P tolerance for SB2 vs astrometric period consistency.
    sb2_period_frac_tol: float = Field(0.1, gt=0)
    # Absolute |Δe| tolerance for SB2 vs astrometric eccentricity.
    sb2_ecc_abs_tol: float = Field(0.15, gt=0, le=1)
    # Which dark-hunter_rv Joker variant block seeds the joint fit when present.
    joker_seed_variant: str = "full"
    # Soft NSS prior scales (fractional on P; absolute on e / ω_rad / T_day) for joint fit.
    joint_prior_period_frac: float = Field(0.05, gt=0)
    joint_prior_ecc_abs: float = Field(0.05, gt=0)
    joint_prior_omega_rad: float = Field(0.2, gt=0)
    joint_prior_t_peri_day: float = Field(5.0, gt=0)
    joint_fit_max_nfev: int = Field(200, ge=20)


class CompanionNatureConfig(BaseModel):
    """Choosables for ``companion_nature_likelihood`` (ARCHITECTURE.md §4).

    Emits continuous per-system weights over
    ``COMPANION_NATURE_WEIGHT_KEYS`` (``BH``/``NS``/``WD``/``other``/``outlier``)
    for ``population_model`` — never a discard filter. Photometric evidence is
    scored as WD / other / dark then mapped onto those five keys. Cooling-track
    files stay under ``physics.cooling_tracks_path`` (local only; not fetched).
    """

    model_config = ConfigDict(extra="forbid")

    # ΔBIC scale mapping continuous weights (ARCHITECTURE.md: config-driven threshold).
    delta_bic_threshold: float = Field(10.0, gt=0)
    # Softmax temperature on joint BIC; larger → flatter weights.
    evidence_scale: float = Field(1.0, gt=0)
    # Floor so a 5σ-style non-detection still carries small non-zero weight.
    weight_floor: float = Field(1.0e-6, gt=0, lt=0.5)
    # Fraction of photometric ``dark`` mass assigned to BH (rest → NS).
    dark_to_bh_fraction: float = Field(0.5, ge=0.0, le=1.0)
    # When RV/astrometry gate failed, blend this fraction into ``outlier``.
    outlier_gate_blend: float = Field(0.8, ge=0.0, le=1.0)
    # Channel enable flags (joint model still accounts for which exist per system).
    use_photometry: bool = True
    use_xp: bool = True
    use_sb2: bool = True
    # Photometry: default mag error when PhotometryPoint.mag_err is missing.
    default_mag_err: float = Field(0.05, gt=0)
    # Extra free parameters for luminous-companion SED models (BIC k term).
    n_params_dark: int = Field(0, ge=0)
    n_params_wd: int = Field(2, ge=0)
    n_params_other: int = Field(2, ge=0)
    # Absolute-magnitude offsets for analytic luminous / WD companions (band-agnostic).
    wd_mg_zero_point: float = Field(12.0)
    wd_mg_mass_slope: float = Field(-2.5)
    other_mg_zero_point: float = Field(5.0)
    other_mg_mass_slope: float = Field(-5.0)
    # XP residual keys under CandidateRecord.extras (absent → channel masked).
    xp_chi2_dark_key: str = "xp_chi2_dark"
    xp_chi2_wd_key: str = "xp_chi2_wd"
    xp_chi2_other_key: str = "xp_chi2_other"
    xp_n_data_key: str = "xp_n_data"
    # Optional precomputed photometry joint-SED chi2 keys (preferred when present).
    phot_chi2_dark_key: str = "phot_chi2_dark"
    phot_chi2_wd_key: str = "phot_chi2_wd"
    phot_chi2_other_key: str = "phot_chi2_other"
    phot_n_data_key: str = "phot_n_data"
    # Analytic primary Mg when building photometry residuals from band list.
    primary_mg_zero_point: float = Field(4.5)
    primary_mg_mass_slope: float = Field(-5.0)
    # SB2 spectral score in [0,1] under extras / rv_summary (higher → WD-like).
    sb2_wd_likeness_key: str = "sb2_wd_likeness"
    sb2_score_n_data: int = Field(4, ge=1)
    # SB2 chi2 scale: dark penalty (~σ^2 per datum) and WD/other contrast scale.
    sb2_dark_chi2_per_datum: float = Field(25.0, gt=0)
    sb2_type_chi2_scale: float = Field(9.0, gt=0)
    # Two-tier: fast bulk vs full queued for ambiguous / critical systems.
    default_tier: Literal["fast", "full"] = "fast"
    full_tier_ambiguity_delta_bic: float = Field(5.0, gt=0)
    full_tier_m2_msun_min: float | None = Field(default=None, gt=0)
    full_queue_max: int | None = Field(default=None, ge=1)
    full_tier_grid_factor: int = Field(3, ge=2, le=20)
    # Required age-independence diagnostic (primary age bins, Gyr).
    age_bin_edges_gyr: list[float] = Field(
        default_factory=lambda: [0.0, 1.0, 3.0, 10.0, 14.0],
        min_length=2,
    )
    age_extras_key: str = "age_gyr"

    @model_validator(mode="after")
    def _companion_nature_bounds(self) -> CompanionNatureConfig:
        edges = self.age_bin_edges_gyr
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("age_bin_edges_gyr must be strictly increasing")
        return self


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


class InferenceConfig(BaseModel):
    """Staged-but-connected Poisson + dynesty inference (ARCHITECTURE.md §4, issue #63).

    Per-system ``rv_astrometry_gate`` / ``companion_nature_likelihood`` results enter as
    fixed empirical-Bayes plug-in weights — not jointly re-sampled (v2 fully-joint is
    documented, not built here). Science knobs live here; no hardcoded sampler sizes.
    """

    model_config = ConfigDict(extra="forbid")

    # Opt-in: honor sensitivity_analysis dimensionality / likelihood-form advice.
    apply_sensitivity_dimensionality: bool = True
    # ``auto`` → SA preferred_likelihood when apply_sensitivity_dimensionality else unbinned.
    likelihood_form: Literal["auto", "unbinned", "binned"] = "auto"
    # Model-comparison switches (ARCHITECTURE.md §4 inference).
    eccentricity_hypothesis: Literal["thermal", "sn_kick"] = "thermal"
    circular_implies_wd: bool = False
    circular_e_threshold: float = Field(0.05, ge=0.0, le=1.0)
    # Free-height log-prior bounds (unit-cube → heights via exp).
    log_height_min: float = -5.0
    log_height_max: float = 5.0
    n_mass_grid: int = Field(64, ge=8)
    # Scalar SF multipliers when HDF5 artifacts absent / simplified path.
    default_astrometric_sf: float = Field(1.0, gt=0.0)
    default_followup_sf: float = Field(1.0, gt=0.0)
    # Dynesty nested sampling (CI smoke uses the small defaults; cluster recipe in docs).
    nlive: int = Field(20, ge=2)
    dlogz: float = Field(0.5, gt=0.0)
    maxcall: int = Field(400, ge=10)
    sample: Literal["auto", "rwalk", "slice", "rslice", "hslice", "unif"] = "rwalk"
    random_seed: int = 63
    # Multi-run robustness protocol (not bitwise seed identity). CI keeps n_robustness_runs=1.
    n_robustness_runs: int = Field(1, ge=1)
    robustness_nlive_scale: float = Field(1.5, gt=1.0)
    robustness_seed_stride: int = Field(17, ge=1)
    # Small-N generics.
    posterior_prior_overlap_threshold: float = Field(0.85, gt=0.0, le=2.0)
    zero_count_ul_confidence: float = Field(0.95, gt=0.0, lt=1.0)
    # When True, skip dynesty and evaluate fiducial logL only (unit tests / dry-run).
    skip_sampler: bool = False

    @model_validator(mode="after")
    def _inference_bounds(self) -> InferenceConfig:
        if self.log_height_min >= self.log_height_max:
            raise ValueError("log_height_min must be < log_height_max")
        return self


class PopulationModelConfig(BaseModel):
    """Hierarchical multiplicity → type mixture + non-parametric MF (ARCHITECTURE.md §4).

    Shared physics (``classification.M_TOV_msun``, ``mass_calibration.delta_M_Ch_msun``,
    ``physics.imf``) stay in their owning sections. External compact-object mass functions
    are never inference priors — ``allow_external_co_mf_priors`` is forced false.
    """

    model_config = ConfigDict(extra="forbid")

    # v1 multiplicity: NSS generative branch is binary-only (latent layer kept generic).
    p_single: float = Field(0.0, ge=0.0, le=1.0)
    p_binary: float = Field(1.0, ge=0.0, le=1.0)
    p_triple: float = Field(0.0, ge=0.0, le=1.0)
    population_classes: list[str] = Field(
        default_factory=lambda: ["BH", "NS", "WD", "other", "outlier"],
        min_length=1,
    )
    # Classes summed into tier-1 raw total compact-object dN/dM (no M_TOV).
    compact_object_classes: list[str] = Field(
        default_factory=lambda: ["BH", "NS", "WD"],
        min_length=1,
    )
    mass_function_model: Literal["free_height_bins", "gp_log_dndm"] = "free_height_bins"
    bin_edge_policy: Literal["equal_log_m", "equal_fiducial_count"] = "equal_log_m"
    n_mass_bins: int = Field(8, ge=2)
    mass_min_msun: float = Field(0.2, gt=0)
    mass_max_msun: float = Field(20.0, gt=0)
    # Fiducial expected detections per bin — fixes edges before real counts (workflow §7).
    fiducial_expected_counts: list[float] = Field(
        default_factory=lambda: [5.0, 8.0, 12.0, 10.0, 6.0, 4.0, 2.0, 1.0],
        min_length=2,
    )
    # Soft logistic width for NS M_TOV truncation; location = classification.M_TOV_msun.
    m_tov_soft_width_msun: float = Field(0.05, gt=0)
    m_tov_prior_sigma_msun: float = Field(0.2, gt=0)
    m_tov_prior_n_quad: int = Field(21, ge=5)
    # Auxiliary parametric families (swappable model-comparison hooks for inference).
    m1_family: Literal["kroupa"] = "kroupa"
    period_family: Literal["flat_log_p", "moe_di_stefano"] = "flat_log_p"
    eccentricity_family: Literal["thermal", "sn_kick"] = "thermal"
    # Opt-in: apply sensitivity_analysis class-covariate recommendations when present.
    apply_sensitivity_covariates: bool = True
    # GP-on-log(dN/dM) hyperparameters (used only when mass_function_model=gp_log_dndm).
    gp_length_scale_log_m: float = Field(0.5, gt=0)
    gp_variance: float = Field(1.0, gt=0)
    # Hard rule: pulsar / LIGO / literature CO MFs are comparison-only, never priors.
    allow_external_co_mf_priors: Literal[False] = False
    random_seed: int = 57

    @model_validator(mode="after")
    def _population_model_bounds(self) -> PopulationModelConfig:
        if self.mass_min_msun >= self.mass_max_msun:
            raise ValueError("mass_min_msun must be < mass_max_msun")
        if len(self.fiducial_expected_counts) != self.n_mass_bins:
            raise ValueError(
                "fiducial_expected_counts length must equal n_mass_bins"
            )
        if any(c < 0 for c in self.fiducial_expected_counts):
            raise ValueError("fiducial_expected_counts must be non-negative")
        mult_sum = self.p_single + self.p_binary + self.p_triple
        if abs(mult_sum - 1.0) > 1e-9:
            raise ValueError("p_single + p_binary + p_triple must equal 1")
        unknown = set(self.compact_object_classes) - set(self.population_classes)
        if unknown:
            raise ValueError(
                f"compact_object_classes not in population_classes: {sorted(unknown)}"
            )
        if self.allow_external_co_mf_priors is not False:
            raise ValueError(
                "allow_external_co_mf_priors must be false "
                "(external CO mass functions are comparison-only)"
            )
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
    mass_derivation: MassDerivationConfig = Field(default_factory=MassDerivationConfig)
    rv_consistency: RvConsistencyConfig = Field(default_factory=RvConsistencyConfig)
    companion_nature: CompanionNatureConfig = Field(
        default_factory=CompanionNatureConfig
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
    population_model: PopulationModelConfig = Field(
        default_factory=PopulationModelConfig
    )
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    triples: TriplesConfig = Field(default_factory=TriplesConfig)
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
    "mass_derivation",
    "rv_consistency",
    "companion_nature",
    "classification",
    "physics",
    "gaiamock",
    "paths",
    "selection_function_astrometric",
    "selection_function_followup",
    "sensitivity_analysis",
    "population_model",
    "triples",
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
