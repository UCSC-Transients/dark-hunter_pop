"""Pydantic config schema for the merged ``config/config.yaml``.

ARCHITECTURE.md §6–§7. Path-specific Gaia/mission keys live under ``dr3`` / ``dr4``
independently (even when values match). Shared physics/population keys live at the top level
under ``physics``, ``mass_calibration``, ``classification``, etc.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_serializer,
    model_validator,
)

from darkhunter_pop.schemas import ActiveDRMode

# JSON-scalar knobs on a cut / primary-mass block. Thresholds live in the
# per-sample YAML ``parameters`` map — never inline in Python.
CutParameterValue = float | int | bool | str | None


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
    # Name of the checked-in host profile (config/host_profiles/<name>.yaml) that was
    # merged on top of config.yaml/fragments to produce the path-shaped keys below, or
    # None when no profile was selected (config.yaml's own values apply unmodified).
    # Selection is always explicit (CLI flag / caller argument) — never inferred from
    # hostname (issue #196). Recorded here (not just in the run file) so it is part of
    # the ``paths`` section that already enters the resume/amend checksum
    # (SHARED_CHECKSUM_SECTIONS): a host switch must invalidate a resume even in the
    # edge case where two profiles happen to resolve to identical path strings.
    host_profile: str | None = None


class DiagnosticsHooksConfig(BaseModel):
    """Enable flags for shared diagnostic emitters (layout hooks + Phase 6/8)."""

    model_config = ConfigDict(extra="forbid")

    funnel_sky: bool = True
    elbadry_six_panel: bool = True
    fit_tier_coverage: bool = True
    gate_pass_rate: bool = True
    age_stratified_wd: bool = True
    triples_robustness: bool = True
    info_gain_followup: bool = True
    sampler_consistency: bool = True
    mc_noise_convergence: bool = True
    solution_type_fractions: bool = True
    known_truth_benchmarks: bool = True
    comparison_catalogs: bool = True
    sbc_recovery: bool = True
    m2_posterior_convergence: bool = True
    # Phase 8 sample-reproduction diagnostics (CONTINUATION_PLAN §13 / issue #113).
    sample_attrition_waterfall: bool = True
    sample_reproduction_report: bool = True
    simon2026_exclusion_breakdown: bool = True
    covariance_health: bool = True
    sample_selection_function: bool = True
    mode_divergence: bool = True
    janssens_segment_occupancy: bool = True


class ModeDivergencePairConfig(BaseModel):
    """Named-sample pair compared by ``mode_divergence`` (e.g. Andrews §6.6)."""

    model_config = ConfigDict(extra="forbid")

    left: str = Field(..., min_length=1)
    right: str = Field(..., min_length=1)
    expected_only_in_right: list[int] = Field(default_factory=list)
    expected_only_in_left: list[int] = Field(default_factory=list)


class SampleSelectionFunctionGridConfig(BaseModel):
    """Forward-model survival grids for ``sample_selection_function`` (§13)."""

    model_config = ConfigDict(extra="forbid")

    m2_msun: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.4, 2.0, 5.0, 10.0],
        min_length=2,
    )
    period_day: list[float] = Field(
        default_factory=lambda: [10.0, 100.0, 365.0, 900.0, 1200.0],
        min_length=2,
    )
    g_mag: list[float] = Field(
        default_factory=lambda: [10.0, 12.0, 14.0, 15.0, 16.0],
        min_length=2,
    )
    # Baseline mock row fields; axis sweeps overwrite one key at a time.
    template: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {
            "nss_solution_type": "Orbital",
            "main_sequence": True,
            "m1_msun": 1.0,
            "m1_tilde_msun": 1.0,
            "paper_m1_msun": 1.0,
            "pipeline_m1_msun": 1.0,
            "m2_msun": 2.0,
            "m2_tilde_msun": 2.0,
            "m2_msun_error": 0.1,
            "sigma_m2_astrometric_msun": 0.05,
            "p_m2_above": 1.0,
            "p_m2_gt_threshold": 1.0,
            "goodness_of_fit": 1.0,
            "period_day": 200.0,
            "phot_g_mean_mag": 12.0,
            "logg_apsis": 4.0,
            "abs_g_mag": 5.0,
            "bp_rp": 0.8,
            "k1_significance": 20.0,
            "fm_msun": 4.0,
            "m2_min_msun": 2.0,
            "andrews_member": False,
        }
    )


class SampleReproductionDiagnosticsConfig(BaseModel):
    """Layout knobs for Phase 8 sample-reproduction diagnostics (issue #113).

    Published membership tables and selection-function grids live here — never
    inline in diagnostic emitters. Science thresholds stay in frozen per-sample
    YAML under ``config/selections/``.
    """

    model_config = ConfigDict(extra="forbid")

    # sample_name -> YAML with a ``data[].source_id`` list (null = N-only check).
    published_tables: dict[str, str | None] = Field(
        default_factory=lambda: {
            "andrews2022": None,
            "elbadry2024": None,
            "elbadry2026": None,
            "elbadry2026_astrometric": (
                "config/selections/external/elbadry2026_table7.yaml"
            ),
            "elbadry2026_spectroscopic": (
                "config/selections/external/elbadry2026_table8.yaml"
            ),
        }
    )
    simon2026_table: str = "config/selections/external/simon2026_orbital.yaml"
    janssens_table: str = (
        "config/selections/external/janssens2022_mass_magnitude.yaml"
    )
    selection_function: SampleSelectionFunctionGridConfig = Field(
        default_factory=SampleSelectionFunctionGridConfig
    )
    mode_divergence_pairs: list[ModeDivergencePairConfig] = Field(
        default_factory=lambda: [
            ModeDivergencePairConfig(
                left="andrews2022",
                right="andrews2022_modified",
                expected_only_in_right=[4373465352415301632],
                expected_only_in_left=[],
            )
        ]
    )


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


class SixPanelAxisConfig(BaseModel):
    """Display axis for one El-Badry six-panel histogram (presentation only, #339).

    Values outside ``[xmin, xmax]`` are not drawn (the report counts them); KS
    statistics in ``selection_function_astrometric`` always use the full samples.
    """

    model_config = ConfigDict(extra="forbid")

    scale: Literal["linear", "log"] = "linear"
    xmin: float
    xmax: float

    @model_validator(mode="after")
    def _range(self) -> SixPanelAxisConfig:
        if self.xmin >= self.xmax:
            raise ValueError("xmin must be < xmax")
        if self.scale == "log" and self.xmin <= 0:
            raise ValueError("log-scale six-panel axis needs xmin > 0")
        return self


def _default_six_panel_axes() -> dict[str, SixPanelAxisConfig]:
    # Panel quantities and scales follow El-Badry et al. (2024) Fig. 5: log P_orb,
    # linear G, linear 1/parallax (kpc), linear e, log f_m,ast, linear cos i.
    return {
        "P_orb_days": SixPanelAxisConfig(scale="log", xmin=10.0, xmax=1.0e4),
        "G_mag": SixPanelAxisConfig(scale="linear", xmin=4.0, xmax=20.0),
        "inv_parallax_mas_inv": SixPanelAxisConfig(scale="linear", xmin=0.0, xmax=2.0),
        "eccentricity": SixPanelAxisConfig(scale="linear", xmin=0.0, xmax=1.0),
        "f_m_msun": SixPanelAxisConfig(scale="log", xmin=1.0e-4, xmax=3.0),
        "cos_inclination": SixPanelAxisConfig(scale="linear", xmin=-1.0, xmax=1.0),
    }


class DiagnosticsConfig(BaseModel):
    """Rendering / report layout for ``plotting`` + ``diagnostics`` (issues #39, #69–#71).

    Layout / DPI / hook flags stay non-checksum. Phase 6 SBC recovery tolerances
    live under ``sbc`` (issue #69). Known-truth / comparison fixture values live
    under ``benchmarks`` (issue #70). Required diagnostic-suite hooks are #71.
    Stage science thresholds (KS, chi2/dof, MC noise) remain in owning stage configs.
    Visual style defaults (fonts, ticks, Okabe–Ito colors) live under ``plotting``.
    """

    model_config = ConfigDict(extra="forbid")

    figure_dpi: int = Field(120, ge=36, le=600)
    figures_subdir: str = "figures"
    reports_subdir: str = "reports"
    write_figures: bool = True
    write_reports: bool = True
    # Cap for matplotlib bins="auto" so heavy-tailed large-N histos stay visible (#96).
    histogram_max_bins: int = Field(80, ge=8, le=2000)
    # Mollweide sky-map marker defaults sized for NSS-scale catalogs (#97).
    sky_map_point_size: float = Field(0.1, gt=0)
    sky_map_alpha: float = Field(0.25, gt=0, le=1)
    # Top-N systems listed in the information-gain / follow-up priority report.
    info_gain_top_n: int = Field(20, ge=1)
    # Max |ΔlogZ| / combined_err allowed across robustness runs (layout-side check).
    sampler_logz_sigma_tol: float = Field(3.0, gt=0)
    # El-Badry six-panel display axes, keyed by SIX_PANEL_NAMES (#339 / #333).
    elbadry_six_panel_axes: dict[str, SixPanelAxisConfig] = Field(
        default_factory=_default_six_panel_axes
    )
    hooks: DiagnosticsHooksConfig = Field(default_factory=DiagnosticsHooksConfig)
    sbc: SBCConfig = Field(default_factory=SBCConfig)
    # Phase 8 sample-reproduction diagnostic knobs (CONTINUATION_PLAN §13).
    sample_reproduction: SampleReproductionDiagnosticsConfig = Field(
        default_factory=SampleReproductionDiagnosticsConfig
    )


class PlottingStyleConfig(BaseModel):
    """Shared figure style for ``darkhunter_pop.plotting`` (see ``docs/PLOTS.md``).

    Typography, tick geometry, line weights, and colorblind-safe cycle live here —
    never hardcoded in call sites. Layout/DPI/hook enable flags remain under
    ``diagnostics``.
    """

    model_config = ConfigDict(extra="forbid")

    font_family: str = "serif"
    axes_label_fontsize: float = Field(18.0, gt=0)
    tick_label_fontsize: float = Field(14.0, gt=0)
    title_fontsize: float = Field(18.0, gt=0)
    legend_fontsize: float = Field(14.0, gt=0)
    # Tick geometry (inward ticks on all sides; minor ticks enabled).
    tick_width: float = Field(2.0, gt=0)
    tick_major_length: float = Field(8.0, gt=0)
    tick_minor_length: float = Field(4.0, gt=0)
    tick_direction: Literal["in", "out", "inout"] = "in"
    spines_width: float = Field(2.0, gt=0)
    # Line / marker defaults (thick enough to read in print).
    line_width: float = Field(2.0, gt=0)
    marker_size: float = Field(6.0, gt=0)
    # Default panel sizes (inches). Portrait preferred for light curves.
    figsize_landscape: tuple[float, float] = (7.0, 5.0)
    figsize_portrait: tuple[float, float] = (5.0, 7.0)
    figsize_wide: tuple[float, float] = (8.0, 4.0)
    # Okabe–Ito palette (https://jfly.uni-koeln.de/color/) — colorblind + B/W safe
    # when combined with linestyle / marker cycling.
    color_cycle: list[str] = Field(
        default_factory=lambda: [
            "#000000",  # black
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#009E73",  # bluish green
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
        ]
    )
    linestyle_cycle: list[str] = Field(
        default_factory=lambda: ["-", "--", "-.", ":"]
    )
    marker_cycle: list[str] = Field(
        default_factory=lambda: ["o", "s", "^", "D", "v", "P", "X", "*"]
    )
    hist_face_color: str = "#0072B2"
    hist_edge_color: str = "#000000"
    threshold_color: str = "#D55E00"
    threshold_linestyle: str = "--"
    #: Decades of dynamic range kept below the maximum on a log ``dN/dM`` panel.
    #: A soft ``M_TOV`` truncation drives the NS rate smoothly toward zero, so an
    #: unclipped log axis can span well over a hundred decades and flatten every
    #: curve against the top of the frame (``docs/PLOTS.md``, "Aspect ratio and
    #: dynamic range"). Display-only, like the rest of this section — `plotting`
    #: is not a stage checksum input and not in any stage's artifact fingerprint.
    dndm_y_decades: float = Field(12.0, gt=0)
    #: Line-spacing multiple applied when a product figure reserves a caption
    #: band beneath the axes, so wrapped caption text cannot overlap the x label.
    caption_line_spacing: float = Field(1.45, gt=0)
    #: Dynamic range, in decades, above which a primitive left on ``"auto"``
    #: switches an axis to log (strictly positive data: ``log10(max / min)``) or
    #: symlog (data with zeros or both signs: ``log10(max|x| / median|x|)``,
    #: over non-zero ``|x|``). Heavy-tailed NSS quantities (RUWE, period,
    #: chi2/dof, ΔBIC) otherwise leave the signal in the first few bins
    #: (``docs/PLOTS.md``, "Aspect ratio and dynamic range"; #333).
    auto_log_min_decades: float = Field(2.0, gt=0)
    #: Categorical bar charts with more categories than this are drawn as
    #: horizontal bars so category labels never overlap (#333). Fewer
    #: categories still go horizontal when any label is wider than its slot.
    categorical_max_vertical_labels: int = Field(8, ge=1)
    #: Height per bar (inches) on a horizontal bar chart; the figure grows with
    #: the number of bars so labels stay at ``tick_label_fontsize``.
    categorical_row_height_inches: float = Field(0.32, gt=0)
    #: Colours and linestyles for labelled reference lines (``M_Ch``, ``M_TOV``,
    #: cuts). Both cycle together so two reference lines never share a style
    #: (#333). Defaults avoid the first six ``color_cycle`` entries, which the
    #: data series use.
    reference_line_colors: list[str] = Field(
        default_factory=lambda: ["#D55E00", "#CC79A7", "#000000"]
    )
    reference_linestyle_cycle: list[str] = Field(
        default_factory=lambda: ["--", ":", "-."]
    )
    #: Hatch patterns cycled with colour on grouped bar charts, so series stay
    #: distinguishable in greyscale (``docs/PLOTS.md``, series discrimination).
    bar_hatch_cycle: list[str] = Field(default_factory=lambda: ["", "//", "..", "xx"])


class BenchmarkCatalogEntry(BaseModel):
    """One comparison-only catalog path entry (ARCHITECTURE.md §4)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    role: Literal["comparison_only"] = "comparison_only"
    never_as_prior: Literal[True] = True


class BenchmarksConfig(BaseModel):
    """Known-truth + comparison catalog paths (issue #70).

    System-level science values live in fixture YAML with provenance. This section
    holds paths and match tolerances only. Excluded from resume checksum (validation).
    External CO mass functions are comparison-only — never inference priors.
    """

    model_config = ConfigDict(extra="forbid")

    known_truth_path: str = "config/benchmarks/known_truth_gaia_bh.yaml"
    ruwe_match_tolerance: float = Field(0.25, gt=0)
    # Known-truth mass check (#348): pipeline M2 must agree with the published value
    # within this many combined sigmas (published ⊕ pipeline marginal sigma).
    mass_check_n_sigma: float = Field(3.0, gt=0)
    # Stages whose artifacts carry an M2 ParameterSet to check, in pipeline order.
    mass_check_stages: list[str] = Field(
        default_factory=lambda: [
            "mass_derivation_bulk",
            "mass_derivation_refined",
            "joint_orbit_fit",
        ]
    )
    # Gaia DR3 NSS solution types that carry an astrometric orbit (a "clean
    # detection" for the known-truth check). Catalog vocabulary, config-owned.
    nss_orbital_solution_types: list[str] = Field(
        default_factory=lambda: [
            "Orbital",
            "AstroSpectroSB1",
            "OrbitalTargetedSearch",
            "OrbitalTargetedSearchValidated",
            "OrbitalAlternative",
            "OrbitalAlternativeValidated",
        ]
    )
    catalogs: dict[str, BenchmarkCatalogEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _benchmarks_hard_rules(self) -> BenchmarksConfig:
        for catalog_id, entry in self.catalogs.items():
            if entry.role != "comparison_only":
                raise ValueError(
                    f"benchmarks.catalogs[{catalog_id}].role must be comparison_only"
                )
            if entry.never_as_prior is not True:
                raise ValueError(
                    f"benchmarks.catalogs[{catalog_id}].never_as_prior must be true"
                )
        return self


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


class SampleSelectionMode(str, Enum):
    """Per-sample execution path (CONTINUATION_PLAN §4.2).

    Dispatch over this enum must raise on an unhandled member rather than
    falling through to a default (Python analogue of exhaustive switch).
    """

    REPRODUCTION = "reproduction"
    FORWARD_MODEL = "forward_model"


class CutKind(str, Enum):
    """Ordered cut-chain entry kinds (CONTINUATION_PLAN §4.6)."""

    COLUMN = "column"
    DERIVED = "derived"
    PROBABILITY = "probability"
    EXCLUSION = "exclusion"


class SampleCut(BaseModel):
    """One ordered post-query cut. Thresholds live in ``parameters``, not Python."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    kind: CutKind
    expression: str = Field(..., min_length=1)
    applies_to: list[SampleSelectionMode] = Field(
        default_factory=lambda: [
            SampleSelectionMode.REPRODUCTION,
            SampleSelectionMode.FORWARD_MODEL,
        ],
        min_length=1,
    )
    expected_n_after: int | None = Field(default=None, ge=0)
    # If this predicate is true, the cut is NOT_APPLICABLE (CONTINUATION_PLAN §15 Q10).
    undefined_when: str | None = None
    # Columns that must be defined (not null / NaN / NotApplicable) else N/A.
    requires_defined: list[str] = Field(default_factory=list)
    # Cross-sample membership import (CONTINUATION_PLAN §8.7).
    from_sample: str | None = None
    parameters: dict[str, CutParameterValue] = Field(default_factory=dict)


class SampleExclusion(BaseModel):
    """Explicit source_id removal with a stated reason (CONTINUATION_PLAN §4.4)."""

    model_config = ConfigDict(extra="forbid")

    source_id: int
    reason: str = Field(..., min_length=1)
    expected_n_after: int | None = Field(default=None, ge=0)


FLAME_OR_UNIFORM_DRAW_METHOD: str = "flame_or_uniform_draw"


class PrimaryMassSpec(BaseModel):
    """Per-sample primary-mass assumption; ignored under ``forward_model`` (§4.3).

    ``method: "flame_or_uniform_draw"`` (#230/#257, Andrews et al. 2022 only):
    Gaia Apsis FLAME mass (``astrophysical_parameters.mass_flame``) with a
    **fixed** Gaussian error ``flame_fixed_error_msun`` when FLAME mass is
    available, else a uniform draw between ``uniform_low_msun`` and
    ``uniform_high_msun`` per Monte Carlo realization. The fixed error is a
    modeling choice, not Gaia's own reported uncertainty — DR3
    ``astrophysical_parameters`` publishes no symmetric ``mass_flame_error``
    column, only ``mass_flame_lower``/``mass_flame_upper`` (16%/84% CI).
    """

    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., min_length=1)
    value_msun: float | None = Field(default=None, gt=0)
    table: str | None = None
    propagate_fit_uncertainty: bool | None = None
    ab_correlation: float | None = None
    applies_only_when: str | None = None
    parameters: dict[str, CutParameterValue] = Field(default_factory=dict)
    # flame_or_uniform_draw fields (#257) — config-driven, no hardcoded
    # 0.1 / 0.63 / 1.0 in Python per dark-hunter-pop-workflow §1.
    flame_column: str | None = None
    flame_fixed_error_msun: float | None = Field(default=None, gt=0)
    uniform_low_msun: float | None = Field(default=None, gt=0)
    uniform_high_msun: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _flame_or_uniform_draw_requires_params(self) -> "PrimaryMassSpec":
        if self.method != FLAME_OR_UNIFORM_DRAW_METHOD:
            return self
        missing = [
            name
            for name, val in (
                ("flame_fixed_error_msun", self.flame_fixed_error_msun),
                ("uniform_low_msun", self.uniform_low_msun),
                ("uniform_high_msun", self.uniform_high_msun),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"primary_mass.method {FLAME_OR_UNIFORM_DRAW_METHOD!r} requires "
                f"{missing} to be set"
            )
        assert self.uniform_low_msun is not None and self.uniform_high_msun is not None
        if self.uniform_high_msun <= self.uniform_low_msun:
            raise ValueError(
                "primary_mass.uniform_high_msun must exceed uniform_low_msun"
            )
        return self


ATF_NOTEBOOK_PROCEDURE_METHOD: str = "atf_notebook"
ATF_COVARIANCE_GATE_SCIPY_STRICT: str = "scipy_multivariate_normal_strict"


class AtfNotebookCovarianceSpec(BaseModel):
    """How the ATF notebook builds and gates the 12×12 NSS covariance (#296).

    ``parameter_order`` is the notebook's ``means`` / ``var_err`` order;
    ``corr_vec`` fills the strict lower triangle row-major (the notebook's
    ``for i: for j < i`` loop). ``use_bit_index: false`` reproduces the
    notebook, which never reads ``bit_index`` (unfitted parameters carry NaN
    errors and fail the gate). ``gate`` names the sampler-construction check:
    ``scipy_multivariate_normal_strict`` rejects every source for which
    ``scipy.stats.multivariate_normal(mean, cov)`` (``allow_singular=False``)
    raises — NaN input, non-PSD, or numerically singular — instead of flooring
    eigenvalues. ``float32_decimal_roundtrip`` re-parses each float32 archive
    value from its shortest decimal string, as the notebook's CSV read did.
    """

    model_config = ConfigDict(extra="forbid")

    parameter_order: list[str] = Field(..., min_length=12, max_length=12)
    use_bit_index: Literal[False]
    gate: Literal["scipy_multivariate_normal_strict"]
    float32_decimal_roundtrip: bool


class AtfNotebookPass1Spec(BaseModel):
    """ATF ``find_massive`` (notebook cell 7): fixed-M1 probability pass (#296).

    The probability threshold itself lives on the ``probability_cut_id`` cut
    (``m2_threshold_msun`` / ``m2_probability_min``), so it is stated once.
    """

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(..., ge=1)
    random_seed: int
    m1_msun: float = Field(..., gt=0)
    root_bracket_msun: tuple[float, float]
    reject_source_on_any_draw_failure: bool
    probability_denominator: Literal["all_draws", "valid_draws"]
    probability_cut_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _bracket_ordered(self) -> AtfNotebookPass1Spec:
        lo, hi = self.root_bracket_msun
        if not (0.0 <= lo < hi):
            raise ValueError("pass1.root_bracket_msun must satisfy 0 <= lo < hi")
        return self


class AtfLickSpectroscopicMass(BaseModel):
    """One UCO Lick spectroscopic primary mass hardcoded in the ATF notebook."""

    model_config = ConfigDict(extra="forbid")

    source_id: int
    m1_msun: float = Field(..., gt=0)


class AtfNotebookPrimaryMassSpec(BaseModel):
    """ATF ``plot_system`` refined M1 (notebook cell 17), used only in pass 2 (#296).

    Priority: UCO Lick spectroscopic mass ``N(m1, lick_sigma_msun)`` for the
    listed sources; else Gaia Apsis FLAME ``N(flame, flame_sigma_msun)``;
    else ``Uniform(uniform_low_msun, uniform_high_msun)``. Non-positive Gaussian
    draws are not clipped (the notebook does not clip).
    """

    model_config = ConfigDict(extra="forbid")

    lick_sigma_msun: float = Field(..., gt=0)
    lick_provenance: str = Field(..., min_length=1)
    lick_spectroscopic_masses: list[AtfLickSpectroscopicMass] = Field(..., min_length=1)
    flame_column: str = Field(..., min_length=1)
    flame_sigma_msun: float = Field(..., gt=0)
    uniform_low_msun: float = Field(..., gt=0)
    uniform_high_msun: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _unique_and_ordered(self) -> AtfNotebookPrimaryMassSpec:
        ids = [row.source_id for row in self.lick_spectroscopic_masses]
        if len(ids) != len(set(ids)):
            raise ValueError("lick_spectroscopic_masses has duplicate source_ids")
        if self.uniform_high_msun <= self.uniform_low_msun:
            raise ValueError("uniform_high_msun must exceed uniform_low_msun")
        return self


class AtfForwardModelPrimaryMassSpec(BaseModel):
    """Pass-2 refined M1 in ``forward_model`` mode (#306).

    Ryan Foley (2026-09-28): the forward model follows the ATF notebook
    procedure, with the pipeline's own mass posterior in place of the paper's
    M1 assumption. Only the pass-2 M1 (the ``M2`` recomputation and the 3σ
    cut) changes; pass 1 stays at the notebook's fixed ``M1``.

    ``method: pipeline_tag10_bulk`` draws ``N(M1, σ_M1)`` from the pipeline's
    bulk-tier M1 (``mass_derivation.derive_tag10_m1_r1`` under the run's
    ``mass_calibration``, MSC then GSP-Phot atmosphere) — the M1 posterior
    ``sample_selection`` sees, since it runs before ``mass_derivation_refined``.
    A source with no TAG10 M1 (no usable atmosphere) takes the notebook's own
    no-mass fallback ``Uniform(uniform_low_msun, uniform_high_msun)``. There is
    no UCO Lick branch in this mode, so the giant ``logg`` cut applies to every
    source.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["pipeline_tag10_bulk"]
    fallback: Literal["uniform"]
    uniform_low_msun: float = Field(..., gt=0)
    uniform_high_msun: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> AtfForwardModelPrimaryMassSpec:
        if self.uniform_high_msun <= self.uniform_low_msun:
            raise ValueError("uniform_high_msun must exceed uniform_low_msun")
        return self


class AtfNotebookPass2Spec(BaseModel):
    """ATF ``plot_system`` (notebook cell 17): refined-M1 pass on pass-1 survivors.

    ``giant_logg_column`` and ``primary_mass.flame_column`` name the row
    column the notebook value comes from; ``*_vizier_apsis`` columns come from
    the VizieR I/355/paramp snapshot (``dr3.vizier_apsis_snapshot_meta``,
    #306), anything else from the Gaia-archive snapshot / FLAME enrichment.
    ``primary_mass`` is the reproduction-mode M1; ``forward_model_primary_mass``
    the forward-model one.
    """

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(..., ge=1)
    random_seed: int
    root_bracket_msun: tuple[float, float]
    reject_source_on_any_draw_failure: bool
    m2_std_ddof: int = Field(..., ge=0)
    giant_logg_column: str = Field(..., min_length=1)
    primary_mass: AtfNotebookPrimaryMassSpec
    forward_model_primary_mass: AtfForwardModelPrimaryMassSpec

    @model_validator(mode="after")
    def _bracket_ordered(self) -> AtfNotebookPass2Spec:
        lo, hi = self.root_bracket_msun
        if not (0.0 <= lo < hi):
            raise ValueError("pass2.root_bracket_msun must satisfy 0 <= lo < hi")
        return self


class ReproductionProcedureSpec(BaseModel):
    """Paper-author selection procedure that precomputes reproduction columns (#296).

    Only ``method: atf_notebook`` (the Andrews, Taggart & Foley 2022 selection
    notebook) exists. Its ``andrews_atf_*`` columns are built by
    ``scripts/build_andrews2022_atf_columns.py`` into one fingerprint-keyed
    sidecar per evaluation mode (#306): the ``reproduction`` sidecar uses the
    paper's pass-2 M1 (``pass2.primary_mass``), the ``forward_model`` sidecar
    the pipeline's (``pass2.forward_model_primary_mass``). Each mode merges
    only its own sidecar.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["atf_notebook"]
    source: str = Field(..., min_length=1)
    source_sha256: str = Field(..., min_length=64, max_length=64)
    covariance: AtfNotebookCovarianceSpec
    pass1: AtfNotebookPass1Spec
    pass2: AtfNotebookPass2Spec


class MonteCarloSpec(BaseModel):
    """Per-sample Monte Carlo settings (CONTINUATION_PLAN §11)."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(..., ge=1)
    covariance: str = Field(..., min_length=1)
    random_seed: int | None = None

    @model_validator(mode="after")
    def _full_covariance_only(self) -> MonteCarloSpec:
        if self.covariance != "full_12x12":
            raise ValueError(
                "monte_carlo.covariance must be 'full_12x12' "
                "(no diagonal-only fallback)"
            )
        return self


class SigmaM2TildeSpec(BaseModel):
    """How El-Badry 2026's ``σ_M̃2`` is propagated (#284 / #199 / Q12).

    ``method``:

    - ``analytic`` — first-order (Jacobian) propagation of ``M̃2`` through
      ``(A, B, F, G, ϖ, P)``, the nsstools-style estimate;
    - ``monte_carlo`` — the ``monte_carlo.n_draws`` full-12×12 NSS draw
      ensemble (the pre-#284 default).

    ``analytic_covariance`` (``analytic`` only):

    - ``full`` — one Jacobian through the 6×6 ``(A, B, F, G, ϖ, P)`` block of
      the full NSS covariance, cross terms included;
    - ``nsstools_blocks`` — ``σ_a0`` from the ``(A, B, F, G)`` block alone (as
      nsstools' ``campbell()`` does), then ``a0``, ``ϖ`` and ``P`` combined in
      quadrature as independent.

    Janssens ``M̃1`` fit uncertainty enters only when
    ``primary_mass.propagate_fit_uncertainty`` is true (Q12 default: false).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["analytic", "monte_carlo"]
    analytic_covariance: Literal["full", "nsstools_blocks"] = "full"


class AmrfCutSpec(BaseModel):
    """AMRF threshold criterion for El-Badry 2026 ``sub_chandrasekhar`` (#284).

    Fills the row column ``amrf_threshold_elbadry2026`` that the frozen cut
    compares ``amrf`` against:

    - ``flat`` — the constant ``flat_min``;
    - ``shahaf2019_class3`` — the Shahaf et al. (2019) class-II/III boundary
      ``max_q A_triple(q; M̃1)`` (companion an equal-mass close MS pair, fainter
      than the primary), evaluated with the ``mass_luminosity`` relation:
      ``shahaf2019_hp`` (their Table A1 Hipparcos broken power law, primaries
      0.6–1.8 M☉), ``janssens2022_g`` (the Gaia-G Janssens relation El-Badry
      2026 already uses for ``M̃1``) or ``power_law`` (``L ∝ M^β``, reference).
    """

    model_config = ConfigDict(extra="forbid")

    criterion: Literal["flat", "shahaf2019_class3"]
    flat_min: float | None = Field(default=None, gt=0)
    mass_luminosity: Literal["shahaf2019_hp", "janssens2022_g", "power_law"] = (
        "janssens2022_g"
    )
    power_law_beta: float | None = Field(default=None, gt=1)
    q_grid_points: int = Field(4001, ge=101)

    @model_validator(mode="after")
    def _criterion_parameters(self) -> AmrfCutSpec:
        if self.criterion == "flat" and self.flat_min is None:
            raise ValueError("amrf_cut.criterion 'flat' requires flat_min")
        if self.mass_luminosity == "power_law" and self.power_law_beta is None:
            raise ValueError(
                "amrf_cut.mass_luminosity 'power_law' requires power_law_beta"
            )
        return self


class ParentQueryVerification(BaseModel):
    """Archive-count verification block for a parent ADQL query (§4.5)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["unverified", "verified"] = "unverified"
    confirmed_count: int | None = Field(default=None, ge=0)
    confirming_query: str | None = None
    verified_on: str | None = None


class ParentQueryDRSpec(BaseModel):
    """DR-path-specific parent catalog query (workflow §6 / CONTINUATION_PLAN §12.4).

    ``adql``, ``nss_table``, and ``solution_types`` are independent under ``dr3`` /
    ``dr4`` even when the values happen to match.
    """

    model_config = ConfigDict(extra="forbid")

    adql: str = Field(..., min_length=1)
    expected_parent_n: int | None = Field(default=None, ge=0)
    nss_table: str | None = None
    solution_types: list[str] = Field(default_factory=list)
    verification: ParentQueryVerification | None = None


class ParentQuerySpec(BaseModel):
    """Parent catalog as literal ADQL, independently for each Gaia DR."""

    model_config = ConfigDict(extra="forbid")

    dr3: ParentQueryDRSpec
    dr4: ParentQueryDRSpec


class SampleProvenance(BaseModel):
    """Frozen literature pointer for a named sample file."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(..., min_length=1)
    section: str | None = None
    published_n: int | None = Field(default=None, ge=0)
    published_n_by_branch: dict[str, int] | None = None
    expected_n: int | None = Field(default=None, ge=0)
    data_release: Literal["dr3", "dr4"] = "dr3"
    frozen_on: str | None = None
    derived_from: str | None = None
    arxiv: str | None = None


class ExtinctionHemisphereSpec(BaseModel):
    """Declination-split dust map (per-sample; do not consolidate across papers)."""

    model_config = ConfigDict(extra="forbid")

    applies_when: str = Field(..., min_length=1)
    map: str = Field(..., min_length=1)


class ExtinctionSpec(BaseModel):
    """Per-sample extinction policy (CONTINUATION_PLAN §7.6 / §8.1)."""

    model_config = ConfigDict(extra="forbid")

    north: ExtinctionHemisphereSpec
    south: ExtinctionHemisphereSpec
    coefficients: dict[str, float] = Field(default_factory=dict)


class MainSequenceCutSpec(BaseModel):
    """Extinction-corrected CMD main-sequence gate (El-Badry 2026 §8.1)."""

    model_config = ConfigDict(extra="forbid")

    mg_floor: float
    cmd_intercept: float
    cmd_slope: float


class SpuriousFractionTargets(BaseModel):
    """Quoted literature rates recorded as outputs, never selection inputs (§4.8)."""

    model_config = ConfigDict(extra="forbid")

    astrometric: float | None = None
    spectroscopic: float | None = None
    source: str | None = None
    role: Literal["acceptance_test"] = "acceptance_test"


class ValidationTargetsSpec(BaseModel):
    """Acceptance-test rates. Selection/likelihood code must not read these."""

    model_config = ConfigDict(extra="forbid")

    spurious_fraction: SpuriousFractionTargets | None = None
    purity: dict[str, float] = Field(default_factory=dict)
    dominant_residual_contaminants: list[str] = Field(default_factory=list)


class Simon2026ExclusionBreakdown(BaseModel):
    """Required El-Badry 2026 acceptance test (CONTINUATION_PLAN §8.9)."""

    model_config = ConfigDict(extra="forbid")

    sb1_fails_significance: int = Field(..., ge=0)
    astrometric_f2_above_max: int = Field(..., ge=0)
    fainter_than_g_limit: int = Field(..., ge=0)
    fails_m2_over_m1: int = Field(..., ge=0)


class AcceptanceTestsSpec(BaseModel):
    """Named reproduction checks attached to a frozen sample file."""

    model_config = ConfigDict(extra="forbid")

    simon2026_exclusion_breakdown: Simon2026ExclusionBreakdown | None = None


class OpenItemRecord(BaseModel):
    """Escalated decision recorded in-file; not resolved by this slot."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    status: Literal["escalated", "resolved"]
    summary: str = Field(..., min_length=1)


class OutcomeDependentCriterionSpec(BaseModel):
    """One §7.3 / §8.5 inclusion term applied identically to mocks.

    Thresholds live here (selection YAML), never inline in Python. The
    ``not_spurious`` kind calls the shared ``P(spurious | ·)`` model — it must
    not read ``validation_targets.spurious_fraction``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    kind: Literal[
        "m2_joint_fit",
        "not_spurious",
        "orbit_coverage",
        "followup_characterized",
    ]
    m2_min_msun: float | None = Field(default=None, gt=0.0)
    orbit_coverage_min_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class InclusionOperatorSpec(BaseModel):
    """Outcome-dependent follow-up terms. Not a catalog cut chain (§7.3 / §8.5)."""

    model_config = ConfigDict(extra="forbid")

    catalog_level_only: bool = True
    notes: str | None = None
    outcome_dependent_criteria: list[OutcomeDependentCriterionSpec] = Field(
        default_factory=list
    )
    astrometric_followup_outcomes: dict[str, int] = Field(default_factory=dict)
    spectroscopic_followup_outcomes: dict[str, int] = Field(default_factory=dict)
    prioritization: list[str] = Field(default_factory=list)


class SampleSubsample(BaseModel):
    """One OR-group of AND cuts (El-Badry 2026 astrometric union)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    cuts: list[SampleCut]
    expected_n: int | None = Field(default=None, ge=0)
    expected_n_new: int | None = Field(default=None, ge=0)
    from_sample: str | None = None
    external_table: str | None = None
    # Frozen external catalog snapshot whose members (``in_sample('<id>')``) and
    # columns are merged onto this subsample's rows in ``reproduction`` mode only
    # (#315). Resolved through the DR-path key ``<id>_snapshot``. A catalog cannot
    # be applied to mocks (§15 Q2), so ``forward_model`` evaluation never reads it.
    external_catalog: Literal["shahaf2023b_class3"] | None = None
    require_main_sequence: bool | None = None
    apply_goodness_of_fit_cut: bool | None = None

    @model_validator(mode="after")
    def _unique_cut_ids(self) -> SampleSubsample:
        ids = [cut.id for cut in self.cuts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"subsample {self.id!r}: duplicate cut ids")
        return self


class SpectroscopicRouteCounts(BaseModel):
    """Published SB1 route breakdown (CONTINUATION_PLAN §8.4)."""

    model_config = ConfigDict(extra="forbid")

    main_sequence_min_companion_mass: int | None = Field(default=None, ge=0)
    high_mass_function: int | None = Field(default=None, ge=0)
    both: int | None = Field(default=None, ge=0)


class SampleBranch(BaseModel):
    """One parent catalog + cut machinery inside a multi-parent sample.

    Maps onto #17 by keeping each branch a ``parent_query`` plus either an
    ordered AND ``cuts`` list or a union of AND ``subsamples``. The named
    sample is the union of branch survivors. ``inference`` is the v1
    likelihood gate (§8.4.1): spectroscopic stays reproduction-only.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    parent_query: ParentQuerySpec
    inference: bool = True
    statistic: str | None = None
    a0_method: str | None = None
    cuts: list[SampleCut] | None = None
    subsamples: list[SampleSubsample] | None = None
    expected_union_n: int | None = Field(default=None, ge=0)
    expected_n_by_route: SpectroscopicRouteCounts | None = None

    @model_validator(mode="after")
    def _cuts_or_subsamples(self) -> SampleBranch:
        has_cuts = self.cuts is not None and len(self.cuts) > 0
        has_subs = self.subsamples is not None and len(self.subsamples) > 0
        if has_cuts == has_subs:
            raise ValueError(
                f"branch {self.id!r}: provide exactly one of cuts or subsamples"
            )
        if has_cuts:
            ids = [cut.id for cut in (self.cuts or [])]
            if len(ids) != len(set(ids)):
                raise ValueError(f"branch {self.id!r}: duplicate cut ids")
        if has_subs:
            ids = [sub.id for sub in (self.subsamples or [])]
            if len(ids) != len(set(ids)):
                raise ValueError(f"branch {self.id!r}: duplicate subsample ids")
        return self


class SampleCorrection(BaseModel):
    """Documented correction for a modified named-sample variant (§6.6)."""

    model_config = ConfigDict(extra="forbid")

    restores_source_id: int | None = None
    identification: str | None = None
    rationale: str | None = None


class SampleSelectionFile(BaseModel):
    """Body of one frozen file under ``config/selections/`` (CONTINUATION_PLAN §4.4).

    ``inherits`` lets a variant reuse another file's parent query and cuts
    (e.g. ``andrews2022_modified``). After inherit resolution, either
    top-level ``parent_query`` + ``cuts`` or ``branches`` is required.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1)
    name: str = Field(..., min_length=1)
    provenance: SampleProvenance
    mode: SampleSelectionMode
    inherits: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    inference_branches: list[str] = Field(default_factory=list)
    primary_mass: PrimaryMassSpec | None = None
    parent_query: ParentQuerySpec | None = None
    cuts: list[SampleCut] | None = None
    branches: list[SampleBranch] | None = None
    exclusions: list[SampleExclusion] | None = None
    monte_carlo: MonteCarloSpec | None = None
    sigma_m2_tilde: SigmaM2TildeSpec | None = None
    amrf_cut: AmrfCutSpec | None = None
    # #296: paper-author procedure that precomputes reproduction-only columns.
    reproduction_procedure: ReproductionProcedureSpec | None = None
    correction: SampleCorrection | None = None
    extinction: ExtinctionSpec | None = None
    main_sequence_cut: MainSequenceCutSpec | None = None
    validation_targets: ValidationTargetsSpec | None = None
    acceptance_tests: AcceptanceTestsSpec | None = None
    inclusion_operator: InclusionOperatorSpec | None = None
    open_items: list[OpenItemRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _inherits_or_complete(self) -> SampleSelectionFile:
        branched = self.branches is not None and len(self.branches) > 0
        if self.inherits is None and not branched:
            if self.parent_query is None:
                raise ValueError(
                    f"sample {self.name!r}: parent_query is required unless "
                    "inherits or branches is set"
                )
            if self.cuts is None:
                raise ValueError(
                    f"sample {self.name!r}: cuts is required unless "
                    "inherits or branches is set"
                )
        if branched and self.inherits is not None:
            raise ValueError(
                f"sample {self.name!r}: inherits is incompatible with branches"
            )
        if self.inherits is not None and self.inherits == self.name:
            raise ValueError(f"sample {self.name!r}: inherits cannot reference itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError(f"sample {self.name!r}: duplicate depends_on entries")
        if self.name in self.depends_on:
            raise ValueError(f"sample {self.name!r}: depends_on cannot include itself")
        cut_ids = [c.id for c in (self.cuts or [])]
        if len(cut_ids) != len(set(cut_ids)):
            raise ValueError(f"sample {self.name!r}: duplicate cut ids")
        if branched:
            branch_ids = [branch.id for branch in (self.branches or [])]
            if len(branch_ids) != len(set(branch_ids)):
                raise ValueError(f"sample {self.name!r}: duplicate branch ids")
            if self.inference_branches:
                unknown = [
                    name for name in self.inference_branches if name not in branch_ids
                ]
                if unknown:
                    raise ValueError(
                        f"sample {self.name!r}: inference_branches {unknown} "
                        "are not branch ids"
                    )
        return self


class SampleSelectionEntry(BaseModel):
    """On/off switch + path + mode for one named sample (CONTINUATION_PLAN §12.1)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    enabled: bool = False
    path: str = Field(..., min_length=1)
    mode: SampleSelectionMode


def default_sample_selection_entries() -> list[SampleSelectionEntry]:
    """Registry names from CONTINUATION_PLAN §12.1; disabled until cut files land."""

    return [
        SampleSelectionEntry(
            name="andrews2022",
            enabled=False,
            path="config/selections/andrews2022.yaml",
            mode=SampleSelectionMode.REPRODUCTION,
        ),
        SampleSelectionEntry(
            name="andrews2022_modified",
            enabled=False,
            path="config/selections/andrews2022_modified.yaml",
            mode=SampleSelectionMode.FORWARD_MODEL,
        ),
        SampleSelectionEntry(
            name="elbadry2024",
            enabled=False,
            path="config/selections/elbadry2024.yaml",
            mode=SampleSelectionMode.REPRODUCTION,
        ),
        SampleSelectionEntry(
            name="elbadry2026",
            enabled=False,
            path="config/selections/elbadry2026.yaml",
            mode=SampleSelectionMode.REPRODUCTION,
        ),
        SampleSelectionEntry(
            name="accel_jerk",
            enabled=False,
            path="config/selections/accel_jerk.yaml",
            mode=SampleSelectionMode.FORWARD_MODEL,
        ),
    ]


class DustMapFileSpec(BaseModel):
    """One on-disk 3D dust map that a frozen selection file names by ``map:`` key (#258).

    The *choice* of map (which hemisphere uses which map, and the split) is frozen
    in the per-sample selection file; this block only says where the file lives on
    this host and how the map's native line-of-sight integral converts to
    ``E(B-V)``. Values live in ``config.yaml`` (no schema defaults — no hardcoded
    physics).

    ``path`` is resolved relative to ``paths.data_root`` unless absolute.
    ``native_quantity`` says what the map integrates (#295):

    * ``reddening`` — a reddening-like unit (e.g. Bayestar19's SFD-like unit);
      ``native_to_ebv`` is required and multiplies the native integral to give
      ``E(B-V)``.
    * ``a0`` — monochromatic extinction ``A0`` (e.g. Lallement 2019 at 5500 Å);
      ``native_to_ebv`` is *derived* as ``1 / DustMapsConfig.r_v`` (the Gaia
      convention ``A0 = R_V E(B-V)``), so R_V is the single knob. Setting it
      explicitly is refused, and the derived value is left out of
      ``model_dump`` so a dumped config re-validates with a new ``r_v``.

    ``md5`` (optional) pins the exact published file; a mismatch refuses to load.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)
    native_quantity: Literal["reddening", "a0"] = "reddening"
    native_to_ebv: float | None = Field(default=None, gt=0.0)
    md5: str | None = None
    provenance: str | None = None
    _derived_factor: bool = PrivateAttr(default=False)

    @model_serializer(mode="wrap")
    def _omit_derived_factor(self, handler: Any) -> Any:
        data = handler(self)
        if self._derived_factor and isinstance(data, dict):
            data.pop("native_to_ebv", None)
        return data


class GaiaBandPolynomialCoefficients(BaseModel):
    """Colour/extinction-dependent Gaia extinction coefficients ``k_X = A_X / A0``.

    Gaia Collaboration, Babusiaux et al. (2018, A&A 616, A10), Eq. 1 / Table 1::

        k_X = c1 + c2 X + c3 X^2 + c4 X^3 + c5 A0 + c6 A0^2 + c7 X A0,
        X = (G_BP - G_RP)_0

    Each list is ``[c1, ..., c7]`` for band G, BP, RP. The paper's fit covers
    3500 K < Teff < 10000 K and 0.01 < A0 < 5 mag; outside that range the
    polynomial is extrapolated, not clipped. ``(G_BP-G_RP)_0`` enters
    ``k_BP - k_RP`` itself, so it is solved by fixed-point iteration from the
    observed colour: at most ``max_iterations`` steps, stopping once the colour
    moves by less than ``tolerance`` mag. A source that does not converge (the
    iteration diverges for very red, highly extincted sources outside the fitted
    range) is reported as ``NotApplicable("extinction_gaia_law_nonconvergent")``.
    """

    model_config = ConfigDict(extra="forbid")

    k_g: list[float] = Field(..., min_length=7, max_length=7)
    k_bp: list[float] = Field(..., min_length=7, max_length=7)
    k_rp: list[float] = Field(..., min_length=7, max_length=7)
    max_iterations: int = Field(..., ge=1)
    tolerance: float = Field(..., gt=0.0)
    provenance: str | None = None


class GaiaBandExtinctionConfig(BaseModel):
    """How ``E(B-V)`` becomes ``A_G`` and ``E(BP-RP)`` for El-Badry 2026 (#295).

    * ``paper_constant`` (default; the reproduction path) — the frozen selection
      file's ``extinction.coefficients`` (``A_G = 2.66 E(B-V)``,
      ``E(BP-RP) = 1.33 E(B-V)``, El-Badry et al. 2026 §2).
    * ``babusiaux2018`` — the Gaia collaboration's colour/A0-dependent law
      (:class:`GaiaBandPolynomialCoefficients`) with ``A0 = r_v · E(B-V)``. A
      sensitivity alternative: it overrides what the paper says it used.
    """

    model_config = ConfigDict(extra="forbid")

    law: Literal["paper_constant", "babusiaux2018"] = "paper_constant"
    babusiaux2018: GaiaBandPolynomialCoefficients | None = None


class DustMapsConfig(BaseModel):
    """Host-side location + unit conversion of 3D dust maps (#258, #295).

    ``ebv_cache_dir`` (relative to ``paths.data_root`` unless absolute) holds the
    per-source native-value cache, so ~440k map queries are paid once per
    map/split fingerprint rather than once per evaluation.

    ``r_v`` is the ratio ``A0 / E(B-V)`` used by every ``a0`` map and by the
    ``babusiaux2018`` Gaia band law. No schema default (config only); required
    whenever something consumes it.
    """

    model_config = ConfigDict(extra="forbid")

    ebv_cache_dir: str = "dust_maps/ebv_cache"
    r_v: float | None = Field(default=None, gt=0.0)
    maps: dict[str, DustMapFileSpec] = Field(default_factory=dict)
    gaia_band_extinction: GaiaBandExtinctionConfig = Field(
        default_factory=GaiaBandExtinctionConfig
    )

    @model_validator(mode="after")
    def _resolve_conversions(self) -> DustMapsConfig:
        for name, spec in self.maps.items():
            if spec.native_quantity == "reddening":
                if spec.native_to_ebv is None:
                    raise ValueError(
                        f"sample_selection.dust_maps.maps.{name}: native_quantity "
                        "'reddening' requires native_to_ebv"
                    )
                continue
            if self.r_v is None:
                raise ValueError(
                    f"sample_selection.dust_maps.maps.{name}: native_quantity 'a0' "
                    "requires sample_selection.dust_maps.r_v"
                )
            if spec.native_to_ebv is not None and not spec._derived_factor:
                raise ValueError(
                    f"sample_selection.dust_maps.maps.{name}: native_to_ebv must not "
                    "be set for an 'a0' map — it is derived as 1/r_v; set r_v instead"
                )
            # Copy, so a spec instance shared between configs never carries
            # another config's r_v.
            resolved = spec.model_copy(update={"native_to_ebv": 1.0 / self.r_v})
            resolved._derived_factor = True
            self.maps[name] = resolved
        band = self.gaia_band_extinction
        if band.law == "babusiaux2018":
            if band.babusiaux2018 is None:
                raise ValueError(
                    "sample_selection.dust_maps.gaia_band_extinction.law "
                    "'babusiaux2018' requires the babusiaux2018 coefficient block"
                )
            if self.r_v is None:
                raise ValueError(
                    "gaia_band_extinction.law 'babusiaux2018' requires "
                    "sample_selection.dust_maps.r_v (A0 = r_v * E(B-V))"
                )
        return self


class SampleSelectionConfig(BaseModel):
    """Pipeline registry for literature sample-selection functions (§12.1–§12.2).

    Thresholds do not live here — they live in the frozen per-sample files.
    Two variants of one paper are independent named entries (not a flag).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    samples: list[SampleSelectionEntry] = Field(
        default_factory=default_sample_selection_entries
    )
    dust_maps: DustMapsConfig = Field(default_factory=DustMapsConfig)
    # #296: fingerprint-keyed sidecars of reproduction-procedure columns
    # (``andrews_atf_*``), relative to ``paths.data_root`` unless absolute; one
    # subdirectory per ``active_dr_mode``, one sidecar per evaluation mode (#306).
    reproduction_column_cache_dir: str = "reproduction_columns"

    @model_validator(mode="after")
    def _unique_names_and_enabled_files(self) -> SampleSelectionConfig:
        names = [entry.name for entry in self.samples]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"sample_selection.samples has duplicate names: {dupes}")
        repo = Path(__file__).resolve().parents[2]
        missing: list[str] = []
        for entry in self.samples:
            if not entry.enabled:
                continue
            path = Path(entry.path)
            if not path.is_absolute():
                path = repo / path
            if not path.is_file():
                missing.append(f"{entry.name} ({entry.path})")
        if missing:
            raise ValueError(
                "enabled sample_selection entries missing selection files: "
                + ", ".join(missing)
            )
        return self


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
    # Snapshot root for dark-hunter_sed Gaia_DR3_*_sed_summary.json (null disables).
    sed_summary_root: str | None = None
    sed_summary_filename_template: str = "Gaia_DR3_{source_id}_sed_summary.json"
    # Snapshot root for dark-hunter_sed Path-2 photometric model comparison
    # (output/phot_sed/Gaia_DR3_<id>_<model>_summary.json); null disables the
    # phot_sed evidence channel and leaves companion_nature on its analytic
    # magnitude-mass fallback. Consumed by phot_sed_adapter.
    phot_sed_root: str | None = None
    phot_sed_filename_template: str = "Gaia_DR3_{source_id}_{model}_summary.json"
    # When true, refined stage raises if darkhunter_sed is not importable.
    require_sed_package: bool = False
    # Log bulk-stage progress every N input candidates; 0 disables heartbeat logs.
    bulk_progress_log_interval: int = Field(10_000, ge=0)
    # Companion-mass diagnostic histogram display range (Msun); outliers beyond
    # xmax are counted in the plot title, not binned.
    bulk_m2_histogram_xmin_msun: float = Field(0.0, ge=0)
    bulk_m2_histogram_xmax_msun: float = Field(30.0, gt=0)
    bulk_m2_histogram_log_y: bool = True


class SpectroscopicMassFunctionConfig(BaseModel):
    """SB1 / SB1C spectroscopic mass-function path (CONTINUATION_PLAN.md §8.4).

    Distinct from the astrometric mass function. v1 is reproduction and
    validation only: ``v1_inference_eligible`` is false and must not be read as
    permission to feed SB1 systems into the population likelihood.
    """

    model_config = ConfigDict(extra="forbid")

    nss_solution_types: list[str] = Field(default_factory=lambda: ["SB1", "SB1C"])
    circular_solution_types: list[str] = Field(default_factory=lambda: ["SB1C"])
    v1_inference_eligible: bool = False
    v1_role: str = "reproduction_and_validation_only"
    published_table_path: str = (
        "config/selections/external/elbadry2026_table8.yaml"
    )
    k1_significance_min: float = Field(10.0, gt=0)
    fm_msun_min: float = Field(3.0, gt=0)
    m2_min_msun_floor: float = Field(1.4, gt=0)
    expected_union_n: int = Field(151, ge=1)
    expected_n_ms_min_companion: int = Field(136, ge=0)
    expected_n_high_fm: int = Field(30, ge=0)
    expected_n_both_routes: int = Field(15, ge=0)
    inversion_max_m2_msun: float = Field(500.0, gt=0)
    inversion_n_bisection: int = Field(200, ge=1)

    @model_validator(mode="after")
    def _v1_inference_stays_off(self) -> SpectroscopicMassFunctionConfig:
        if self.v1_inference_eligible:
            raise ValueError(
                "spectroscopic_mass_function.v1_inference_eligible must be false "
                "in v1 (CONTINUATION_PLAN.md §8.4.1); flipping it on is a scoped "
                "v2 decision requiring human sign-off"
            )
        if not self.nss_solution_types:
            raise ValueError("nss_solution_types must be non-empty")
        return self


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
    # RV epochs with MJD below this are rejected as non-physical placeholders
    # (upstream summaries carry ``mjd: 0.0`` rows; #347). Counted, never fit.
    rv_epoch_min_mjd: float = 40000.0
    # Lower jitter bound (km/s). A fit sitting here is a legitimate zero-jitter
    # boundary (not a bound hit); the jitter is then held fixed for the Fisher matrix.
    jitter_min_kms: float = Field(1.0e-3, gt=0)
    # joint_orbit_fit (#347): RV epochs + the NSS astrometric solution vector
    # (parallax, Thiele–Innes A/B/F/G[/C/H], e, P, T_peri) with its full
    # covariance; K is derived from a1 = a0/parallax, never free. Search box for
    # the fitted masses — an optimum on either edge is flagged and counted as a
    # bound hit, never returned as a solution.
    joint_m1_bounds_msun: tuple[float, float] = (0.05, 100.0)
    joint_m2_bounds_msun: tuple[float, float] = (1.0e-3, 1.0e3)
    # Hard eccentricity ceiling inside the joint fit (an optimum here is a bound hit).
    joint_ecc_max: float = Field(0.99, gt=0, lt=1)
    # Max function evaluations for the L-BFGS-B stage of the joint fit.
    joint_fit_max_nfev: int = Field(20000, ge=20)
    # Convergence: scoring decrement gᵀF⁻¹g of −ln L at the optimum (≈ twice the
    # log-likelihood a further step would gain); parametrization-invariant.
    joint_fit_decrement_tol: float = Field(1.0e-4, gt=0)
    # Identifiability: smallest/largest Fisher eigenvalue (local-σ units) must
    # exceed this, else the fit is reported not converged (degenerate direction).
    joint_fisher_min_eig_ratio: float = Field(1.0e-12, gt=0, lt=1)
    # L-BFGS-B restarts from its own result (stop when −ln L gains < 1e-9).
    joint_lbfgsb_restarts: int = Field(5, ge=0)
    # Fisher-scoring polish steps after L-BFGS-B.
    joint_polish_steps: int = Field(20, ge=0)
    # Central-difference step for model Jacobians (gradient / Fisher / output
    # propagation), in units of the local posterior σ (1/sqrt Fisher diagonal).
    joint_derivative_step: float = Field(1.0e-3, gt=0)
    # A parameter within this many conditioning units (its NSS / M1 σ; 0.02 in
    # √e·cos/sin ω and angles; 5% of seed M2; 0.1 in ln jitter) of a box bound
    # counts as a bound hit.
    joint_bound_tol: float = Field(1.0e-3, gt=0)
    # Sort gate input by on-disk summary mtime (newest first). Prefer current
    # pipeline outputs over stale pre-pipeline summaries.
    prefer_recent_summary_mtime: bool = True
    # Optional known-truth / calibrator source_ids (e.g. Gaia BH1) scored first.
    priority_source_ids: list[int] = Field(default_factory=list)
    # Prefer candidates whose rv_summary has non-empty external_rvs (public/lit).
    prefer_public_external_rvs: bool = True


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


class McMassFunctionConfig(BaseModel):
    """Shared Monte Carlo mass-function knobs (CONTINUATION_PLAN §11).

    Per-sample files may override ``n_draws`` / ``random_seed`` via ``MonteCarloSpec``.
    """

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(10_000, ge=1)
    random_seed: int = 19
    covariance: str = "full_12x12"
    eig_rel_floor: float = Field(1.0e-12, ge=0.0)
    eig_abs_floor: float = Field(1.0e-18, ge=0.0)
    boundary_n_sigma: float = Field(1.0, gt=0.0)

    @model_validator(mode="after")
    def _full_covariance_only(self) -> McMassFunctionConfig:
        if self.covariance != "full_12x12":
            raise ValueError(
                "mc_mass_function.covariance must be 'full_12x12' "
                "(no diagonal-only fallback)"
            )
        return self


class SpuriousnessValidationTarget(BaseModel):
    """One published rate used as an acceptance test, never as a model input."""

    model_config = ConfigDict(extra="forbid")

    table: str
    g_max: float | None = None
    target_spurious_fraction: float | None = None
    target_reliable_fraction: float | None = None
    target_spurious_fraction_advisory: float | None = None
    fixture_numerator: int | None = None
    fixture_denominator: int | None = None
    fixture_good: int | None = None
    fixture_n: int | None = None
    fixture_spurious_solution_fraction: float | None = None
    role: Literal["acceptance_test", "advisory"] = "acceptance_test"


class SpuriousnessValidationConfig(BaseModel):
    """Fixture-recoverable rate targets for the shared spuriousness model (§4.8)."""

    model_config = ConfigDict(extra="forbid")

    elbadry2024: SpuriousnessValidationTarget
    elbadry2026_astrometric: SpuriousnessValidationTarget
    elbadry2026_sb1: SpuriousnessValidationTarget
    absolute_tolerance: float = Field(0.05, gt=0.0, le=0.5)


class SpuriousnessModelFile(BaseModel):
    """Body of ``config/spuriousness_model.yaml`` (sample-independent, §4.8)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    name: str = "spuriousness_model"
    link_function: Literal["probit", "logit"] = "probit"
    censoring: Literal["joint"] = "joint"
    regularization_l2: float = Field(0.25, ge=0.0)
    rho_abs_max: float = Field(0.95, gt=0.0, lt=1.0)
    scanning_law_fundamental_period_days: float = Field(62.0, gt=0.0)
    harmonic_multiples_max: int = Field(20, ge=1)
    g_window_class_break: float = Field(13.0)
    f2_threshold_bright: float = Field(6.0)
    f2_threshold_faint: float = Field(4.0)
    bic_delta_include_covariate: float = Field(6.0, gt=0.0)
    min_complete_rows_for_covariate: int = Field(20, ge=5)
    candidate_covariates: list[str] = Field(..., min_length=1)
    missing_at_selection_covariates: list[str] = Field(
        default_factory=lambda: ["rv_consistency"]
    )
    labeled_sets: list[str] = Field(..., min_length=1)
    validation: SpuriousnessValidationConfig
    gaia_bh1_source_id: int
    branch_key: str = "branch"
    branch_values: list[str] = Field(
        default_factory=lambda: ["astrometric", "spectroscopic"]
    )


class SpuriousnessModelConfig(BaseModel):
    """Registry pointer for the shared spuriousness model (outside selections/)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    path: str = "config/spuriousness_model.yaml"


class MultiSolutionRatesConfig(BaseModel):
    """Registry pointer for #241's shared empirical multi-solution rate table.

    Sample-independent (like ``SpuriousnessModelConfig``), consumed by both
    ``selection_function_astrometric`` and ``selection_function_followup`` (issue #243)
    via their respective ``multi_solution`` sub-configs. Generated by
    ``scripts/measure_multi_solution_rate.py --rates-path config/multi_solution_rates.yaml``
    against the uncut NSS parent snapshot — never hand-edit the target file's counts.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    path: str = "config/multi_solution_rates.yaml"


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


class InferenceMultiSampleConfig(BaseModel):
    """Multi-sample Poisson overlap policy (CONTINUATION_PLAN §4.7 / §15 Q1).

    Default formulation is ``unified_inclusion_indicator``. Separate Poisson
    processes are retained only as a diagnostic contrast and must not be the
    production joint-inference setting when samples overlap.
    """

    model_config = ConfigDict(extra="forbid")

    formulation: Literal[
        "unified_inclusion_indicator",
        "separate_poisson",
    ] = "unified_inclusion_indicator"
    # Inference-facing samples only. Frozen andrews2022 and SB1 are excluded;
    # accel_jerk stays off (§9).
    sample_names: list[str] = Field(
        default_factory=lambda: [
            "andrews2022_modified",
            "elbadry2024",
            "elbadry2026",
        ]
    )
    # Catalog-level survival scalars when mock Monte Carlo is unavailable (CI).
    default_catalog_sf: dict[str, float] = Field(
        default_factory=lambda: {
            "andrews2022_modified": 1.0,
            "elbadry2024": 1.0,
            "elbadry2026": 1.0,
        }
    )
    # Intercept-only P(spurious) used when no #111 fit artifact is supplied.
    default_p_spurious: float = Field(0.25, gt=0.0, lt=1.0)
    require_unified_for_joint: bool = True

    @model_validator(mode="after")
    def _refuse_frozen_andrews_and_accel(self) -> InferenceMultiSampleConfig:
        if "andrews2022" in self.sample_names and "andrews2022_modified" not in self.sample_names:
            raise ValueError(
                "inference.multi_sample must use andrews2022_modified, not frozen "
                "andrews2022 alone (§6.6)"
            )
        if "andrews2022" in self.sample_names:
            raise ValueError(
                "inference.multi_sample.sample_names must not include frozen "
                "andrews2022 (§6.6); use andrews2022_modified"
            )
        if "accel_jerk" in self.sample_names:
            raise ValueError(
                "accel_jerk is blocked (§9); omit it from inference.multi_sample"
            )
        if self.require_unified_for_joint and self.formulation != "unified_inclusion_indicator":
            raise ValueError(
                "joint multi-sample inference requires formulation="
                "unified_inclusion_indicator (§15 Q1); set "
                "require_unified_for_joint=false only for explicit diagnostics"
            )
        for name, value in self.default_catalog_sf.items():
            if value <= 0.0:
                raise ValueError(f"default_catalog_sf[{name!r}] must be > 0")
        return self


class InferenceConfig(BaseModel):
    """Staged-but-connected Poisson + dynesty inference (ARCHITECTURE.md §4, issue #63).

    Per-system ``rv_astrometry_gate`` / ``companion_nature_likelihood`` results enter as
    fixed empirical-Bayes plug-in weights — not jointly re-sampled (v2 fully-joint is
    documented, not built here). Science knobs live here; no hardcoded sampler sizes.
    Phase 8 extends the rate with ``sample_selection_function_s`` via the unified
    inclusion-indicator (§4.7 / §15 Q1).
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
    # Multi-sample selection integration (§4.7 / §15 Q1).
    multi_sample: InferenceMultiSampleConfig = Field(
        default_factory=InferenceMultiSampleConfig
    )
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
    # Collapse floor (#352): any free-height parameter with σ_post/σ_prior below this
    # is a degenerate (collapsed / identical-sample) posterior — a failure, not a pass.
    posterior_collapse_width_ratio_floor: float = Field(1e-6, gt=0.0, lt=1.0)
    zero_count_ul_confidence: float = Field(0.95, gt=0.0, lt=1.0)
    # Consumer-side gate policy (#352). ``refuse``: inference raises and records the
    # stage failed when an upstream validation gate (astrometric validation gate,
    # follow-up calibration) did not pass. ``mark_not_science_valid``: inference runs,
    # but the artifact, report and run file say ``science_valid: False`` with every
    # reason. Neither value ever yields a science-valid result on a failed gate.
    upstream_gate_policy: Literal["refuse", "mark_not_science_valid"] = (
        "mark_not_science_valid"
    )
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
    # Real-vs-mock multi-solution rate comparison (issue #243). Diagnostic-only in v1:
    # not folded into ValidationGateResult.passed because the empirical rate is ~1%, so a
    # mock_population.N_realizations-scale run is noisy relative to this tolerance — see
    # forward_model.MultiSolutionDiagnosticResult and the PR discussion for #243.
    multi_solution_rate_max_abs_delta: float = Field(0.05, gt=0, le=1)


class MultiSolutionEmissionConfig(BaseModel):
    """Per-stage knobs applying the shared empirical multi-solution rate table (#241).

    ``primary_type`` anchors which ``cross_type_combo_counts`` entries (from
    ``PipelineConfig.multi_solution_rates``) apply to this stage's mock realizations —
    ``"Orbital"`` for the astrometric cascade's accepted twelve-parameter solutions,
    ``"SB1"`` for follow-up-triggered spectroscopic solutions. The two stages draw
    independently (separate ``random_seed``) per issue #243's "do not assume the
    astrometric-side change already covers follow-up" instruction.

    ``same_type_period_alias_ratio_*`` has **no empirical calibration**: #241 measured
    zero same-type/period-aliased sources in the uncut snapshot, so there is no observed
    period-alias magnitude to match. This ratio range is a placeholder mechanism (the
    "how different" question is open) that only fires when
    ``PipelineConfig.multi_solution_rates`` reports a nonzero
    ``same_type_period_aliased_rate`` — a no-op at today's measured rate of exactly zero.
    Flagged for Ryan-level review if that rate is ever remeasured as nonzero.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    primary_type: str = Field(..., min_length=1)
    same_type_period_alias_ratio_min: float = Field(0.5, gt=0)
    same_type_period_alias_ratio_max: float = Field(2.0, gt=0)
    random_seed: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _ratio_range(self) -> MultiSolutionEmissionConfig:
        if self.same_type_period_alias_ratio_min >= self.same_type_period_alias_ratio_max:
            raise ValueError(
                "same_type_period_alias_ratio_min must be < "
                "same_type_period_alias_ratio_max"
            )
        if (
            self.same_type_period_alias_ratio_min == 1.0
            or self.same_type_period_alias_ratio_max == 1.0
        ):
            raise ValueError(
                "same_type_period_alias_ratio_min/max must not equal 1.0 exactly (an "
                "alias ratio of 1.0 would not be a distinct period)"
            )
        return self


class MockPopulationSampling(str, Enum):
    """How mock binary parameters are drawn before the gaiamock cascade."""

    FIXED = "fixed"
    ELBADRY_PRIOR = "elbadry_prior"


class MockPopulationConfig(BaseModel):
    """Fiducial binary used for mock injection / validation (one realization set)."""

    model_config = ConfigDict(extra="forbid")

    sampling: MockPopulationSampling = MockPopulationSampling.ELBADRY_PRIOR
    random_seed: int = 42
    # ``fixed`` mode uses the scalar fields below; ``elbadry_prior`` draws from the ranges.
    period_days: float = Field(1000.0, gt=0)
    Mg_tot: float = 4.0
    flux_ratio: float = Field(0.01, gt=0)
    m1_msun: float = Field(1.0, gt=0)
    m2_msun: float = Field(0.5, gt=0)
    eccentricity: float = Field(0.3, ge=0, lt=1)
    period_days_min: float = Field(60.0, gt=0)
    period_days_max: float = Field(8500.0, gt=0)
    eccentricity_max: float = Field(0.65, gt=0, lt=1)
    m1_msun_min: float = Field(0.7, gt=0)
    m1_msun_max: float = Field(2.8, gt=0)
    m2_msun_min: float = Field(0.08, gt=0)
    m2_msun_max: float = Field(3.5, gt=0)
    flux_ratio_min: float = Field(1e-4, gt=0)
    flux_ratio_max: float = Field(0.15, gt=0)
    Mg_tot_min: float = 3.0
    Mg_tot_max: float = 6.5
    # Fraction of draws tagged as NSS insufficient_visibility (not gaiamock plx==0).
    faint_draw_fraction: float = Field(0.45, ge=0.0, le=1.0)
    faint_Mg_tot_min: float = 8.0
    faint_Mg_tot_max: float = 11.5
    N_realizations: int = Field(500, ge=1)
    ruwe_min: float = Field(1.4, gt=0)
    skip_acceleration: bool = False
    hz_pc: float = Field(300.0, gt=0)

    @model_validator(mode="after")
    def _mock_population_bounds(self) -> MockPopulationConfig:
        if self.period_days_min >= self.period_days_max:
            raise ValueError("period_days_min must be < period_days_max")
        if self.m1_msun_min >= self.m1_msun_max:
            raise ValueError("m1_msun_min must be < m1_msun_max")
        if self.m2_msun_min >= self.m2_msun_max:
            raise ValueError("m2_msun_min must be < m2_msun_max")
        if self.flux_ratio_min >= self.flux_ratio_max:
            raise ValueError("flux_ratio_min must be < flux_ratio_max")
        if self.Mg_tot_min >= self.Mg_tot_max:
            raise ValueError("Mg_tot_min must be < Mg_tot_max")
        if self.faint_Mg_tot_min >= self.faint_Mg_tot_max:
            raise ValueError("faint_Mg_tot_min must be < faint_Mg_tot_max")
        return self


class SelectionFunctionAstrometricConfig(BaseModel):
    """Shared selection-function astrometric settings (not DR-path-specific)."""

    model_config = ConfigDict(extra="forbid")

    extinction_model: ExtinctionModel = ExtinctionModel.COMBINED19
    validation_gate: ValidationGateConfig = Field(default_factory=ValidationGateConfig)
    mock_population: MockPopulationConfig = Field(default_factory=MockPopulationConfig)
    multi_solution: MultiSolutionEmissionConfig = Field(
        default_factory=lambda: MultiSolutionEmissionConfig(
            primary_type="Orbital", random_seed=4241
        )
    )


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
    multi_solution: MultiSolutionEmissionConfig = Field(
        default_factory=lambda: MultiSolutionEmissionConfig(
            primary_type="SB1", random_seed=4242
        )
    )

    @model_validator(mode="after")
    def _limits_ordered(self) -> SelectionFunctionFollowupConfig:
        if self.declination_min_deg >= self.declination_max_deg:
            raise ValueError("declination_min_deg must be < declination_max_deg")
        if self.g_mag_bright_limit >= self.g_mag_faint_limit:
            raise ValueError("g_mag_bright_limit must be < g_mag_faint_limit")
        return self


class OrbitalSolutionCutsConfig(BaseModel):
    """Gaia orbital-solution acceptance cuts applied to mock cascade outputs.

    DR3 values are Halbwachs et al. (2023) as restated in El-Badry et al. (2024,
    OJAp 7, 100; arXiv:2411.00088) Eq. 18 (``a0/sigma_a0 > 5``, ``F2 < 25``) and
    Eqs. 20-22 (``parallax_over_error > 20000 d / P``,
    ``sigma_e < 0.079 ln(P/d) - 0.244``, ``a0/sigma_a0 > 158 / sqrt(P/d)``).
    Mission-specific, so configured independently per DR path (#339).
    """

    model_config = ConfigDict(extra="forbid")

    a0_over_err_min: float = Field(5.0, gt=0)
    goodness_of_fit_f2_max: float = Field(25.0, gt=0)
    parallax_over_error_times_period_min_days: float = Field(20000.0, gt=0)
    a0_over_err_times_sqrt_period_min: float = Field(158.0, gt=0)
    sigma_e_ln_period_slope: float = 0.079
    sigma_e_intercept: float = -0.244


class AccelerationPublicationCutsConfig(BaseModel):
    """Extra cuts a provisionally accepted acceleration solution needed to be *published*.

    El-Badry et al. (2024) §5.2.1 (from Halbwachs et al. 2023): significance s > 20 for
    both 7- and 9-parameter solutions (s > 12 only removed them from orbit fitting), and
    the F2 threshold for published 7-parameter solutions lowered from 25 to 22.
    Path-specific (DR3 post-processing).
    """

    model_config = ConfigDict(extra="forbid")

    acceleration_significance_min: float = Field(20.0, gt=0)
    acceleration7_f2_max: float = Field(22.0, gt=0)
    acceleration9_f2_max: float = Field(25.0, gt=0)


class DRSelectionFunctionPathConfig(BaseModel):
    """Path-specific mock-injection window, orbital cuts and real comparison sample."""

    model_config = ConfigDict(extra="forbid")

    d_min_pc: float = Field(100.0, gt=0)
    d_max_pc: float = Field(500.0, gt=0)
    orbital_solution_cuts: OrbitalSolutionCutsConfig = Field(
        default_factory=OrbitalSolutionCutsConfig
    )
    acceleration_publication_cuts: AccelerationPublicationCutsConfig = Field(
        default_factory=AccelerationPublicationCutsConfig
    )
    # Mock binaries with apparent G at or fainter than this are removed from the mock
    # sample before the cascade: El-Badry et al. (2024) §3.2 drops unresolved binaries
    # with G > 19, "which were not fit with binary solutions in DR3". None disables.
    mock_g_mag_max: float | None = 19.0
    # Real side of the El-Badry et al. (2024) six-panel comparison: the published
    # ``nss_two_body_orbit`` rows whose ``nss_solution_type`` is in this list, taken
    # from the uncut Gaia snapshot (no pipeline quality cut). DR3: the paper's §4
    # "nss_solution_type = Orbital or AstroSpectroSB1" (168,065 rows; footnote 6).
    elbadry2024_comparison_nss_solution_types: list[str] = Field(
        default_factory=lambda: ["Orbital", "AstroSpectroSB1"], min_length=1
    )

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
    """One external-band cross-match via Gaia archive precomputed tables.

    Simple path (AllWISE / PanSTARRS / SDSS): ``neighbour_table`` → ``catalog_table``
    with ``neighbour_to_catalog`` as ``neighbour_col=catalog_col``.

    2MASS path: ``neighbour_table`` → ``join_table`` → ``catalog_table`` using
    ``neighbour_to_join`` and ``join_to_catalog`` (ESA mandatory join pattern).
    """

    model_config = ConfigDict(extra="forbid")

    band: str = Field(..., min_length=1)
    neighbour_table: str = Field(..., min_length=1)
    catalog_table: str = Field(..., min_length=1)
    mag_column: str = Field(..., min_length=1)
    mag_err_column: str | None = None
    enabled: bool = True
    # "neighbour_col=catalog_col" when join_table is unset.
    neighbour_to_catalog: str | None = None
    join_table: str | None = None
    # "neighbour_col=join_col" when join_table is set.
    neighbour_to_join: str | None = None
    # "join_col=catalog_col" when join_table is set.
    join_to_catalog: str | None = None

    @model_validator(mode="after")
    def _validate_join_keys(self) -> ExternalPhotometryCrossmatch:
        if not self.enabled:
            return self
        if self.join_table:
            if not self.neighbour_to_join or not self.join_to_catalog:
                raise ValueError(
                    f"band {self.band!r}: join_table requires neighbour_to_join "
                    "and join_to_catalog"
                )
        elif not self.neighbour_to_catalog:
            raise ValueError(
                f"band {self.band!r}: neighbour_to_catalog is required when "
                "join_table is unset"
            )
        return self


class EpochContinuousLossConfig(BaseModel):
    """Continuous keep model: polynomial in clipped G + Galactic real harmonics (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    g_clip: list[float] = Field(..., min_length=2, max_length=2)
    g_ref: float
    g_scale: float = Field(..., gt=0)
    coef_g: list[float] = Field(..., min_length=1)
    sky_lmax: int = Field(..., ge=0)
    coef_sky: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> EpochContinuousLossConfig:
        if len(self.coef_sky) != (self.sky_lmax + 1) ** 2 - 1:
            raise ValueError("coef_sky needs (sky_lmax + 1)^2 - 1 entries")
        return self


class EpochClusteredLossConfig(BaseModel):
    """Time-clustered loss episodes for faint stars (#400 E4; §4.3)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    g_start: float
    g_full: float
    frac_max: float = Field(..., ge=0, le=1)
    tau_day: float = Field(..., gt=0)


class EpochBrightExcessNoiseConfig(BaseModel):
    """Bright-star per-CCD excess noise for the unbinned overlay (#398 / #400 N2; §4.5)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    g_max: float
    knots_g: list[float] = Field(..., min_length=1)
    knots_r2: list[float] = Field(..., min_length=1)
    renormalize_ruwe: bool = True


class EpochRuweU0Config(BaseModel):
    """Mock RUWE normalisation u0(G) emulating DR3's RUWE = UWE / u0 (#400 N2-u0; §8.8)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    table: str
    table_sha256: str
    provenance: str = ""


class EpochTransitLossConfig(BaseModel):
    """Per-FoV-transit loss probability after the gaps (docs/EPOCH_MODEL_SPEC.md §3)."""

    model_config = ConfigDict(extra="forbid")

    model: Literal["binned", "continuous"] = "binned"
    continuous: EpochContinuousLossConfig | None = None
    g_edges: list[float] = Field(..., min_length=2)
    prob: list[float] = Field(..., min_length=1)
    density_slope_per_dex: float = 0.0
    density_ref_per_deg2: float = Field(1.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> EpochTransitLossConfig:
        if len(self.prob) != len(self.g_edges) - 1:
            raise ValueError("transit_loss.prob needs len(g_edges) - 1 entries")
        if any(b <= a for a, b in zip(self.g_edges[:-1], self.g_edges[1:])):
            raise ValueError("transit_loss.g_edges must increase")
        if any(not (0.0 <= x < 1.0) for x in self.prob):
            raise ValueError("transit_loss.prob must be in [0, 1)")
        return self


class EpochExcessNoiseConfig(BaseModel):
    """El-Badry et al. (2024) Sect. 3.3.1 bright-star per-FoV-transit excess noise.

    Used by the #400 validation only (``scripts/validate_epoch_model_400.py``); no
    production path reads it while ``enabled`` is false (#398 option 3 is undecided).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    g_max: float = 13.0
    sigma_max_mas: float = Field(0.04, ge=0)


class EpochModelPathConfig(BaseModel):
    """Pop-side statistical epoch model around gaiamock's GOST list (#400, #398).

    docs/EPOCH_MODEL_SPEC.md. Path-specific (scanning law, data gaps, calibration on
    the release's own counts); DR4 has its own block (null until calibrated).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    gap_table: str
    gap_table_sha256: str
    gap_source: str = ""
    apply_gaps: bool = True
    agis_window_obmt_rev: list[float] | None = None
    obmt_reference_rev: float
    obmt_reference_jd_tcb: float
    obmt_rev_per_day: float = Field(..., gt=0)
    transit_split_day: float = Field(..., gt=0)
    transit_loss: EpochTransitLossConfig
    calibration_snapshot: str | None = None
    excess_noise: EpochExcessNoiseConfig = Field(default_factory=EpochExcessNoiseConfig)
    clustered_loss: EpochClusteredLossConfig | None = None
    bright_excess_noise: EpochBrightExcessNoiseConfig | None = None
    ruwe_u0: EpochRuweU0Config | None = None
    provenance: str = ""

    @model_validator(mode="after")
    def _window(self) -> EpochModelPathConfig:
        w = self.agis_window_obmt_rev
        if w is not None and (len(w) != 2 or not w[0] < w[1]):
            raise ValueError("agis_window_obmt_rev must be [start, end] with start < end")
        return self


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
    # astroquery Gaia.ROW_LIMIT; -1 = unlimited. Use 2000 for archive smoke tests.
    gaia_archive_row_limit: int = Field(-1, ge=-1)
    # Sync jobs hit ESA Error 408 on large NSS+crossmatch queries; async is default.
    gaia_archive_async: bool = True
    # Gaia NSS Orbital pseudo-circular flag: null eccentricity_error and e below this (§7.2.5).
    pseudo_circular_eccentricity_max: float = Field(0.0005, ge=0)
    # External crossmatch photometry: treat err<=0 as missing; optional floor at ingest.
    external_mag_err_floor: float = Field(0.05, gt=0)
    external_mag_err_zero_as_missing: bool = True
    impute_external_mag_err: bool = True
    # Cross-match fan-out resolution (#221, option B; ARCHITECTURE.md §4 "Multi-solution
    # sources"). Bands (``external_photometry_crossmatches[*].band``) whose value/error
    # cells are the ONLY cells allowed to differ between rows of one duplicated
    # ``source_id`` for data_acquisition to collapse that group to one row. Conflicting
    # bands are masked (value + error), never chosen between. Empty disables the
    # collapse: every fan-out group is then refused, as before #221.
    crossmatch_fanout_maskable_bands: list[str] = Field(default_factory=list)
    nss_table: str = "gaiadr3.nss_two_body_orbit"
    gaia_source_table: str = "gaiadr3.gaia_source"
    # NSS covariance / K1 / NSS-error enrichment merged into data_acquisition rows
    # per (source_id, nss_solution_type) (#308; ARCHITECTURE.md §4 data_acquisition).
    # Directory name under ``{data_root}/{dr}/gaia_snapshots/`` holding a ``meta.yaml``
    # written by scripts/fetch_nss_enrichment.py. Null disables the merge; configured
    # but missing is a hard error, never a silent skip.
    nss_enrichment_snapshot: str | None = None
    # Shahaf et al. (2023b) Table 2 class-III catalog snapshot (#315): directory name
    # under ``{data_root}/{dr}/external_catalogs/`` (or an absolute path) holding
    # ``table2.ecsv`` + ``meta.yaml`` from scripts/fetch_shahaf2023b_class3.py. Read
    # only by reproduction-mode subsamples with ``external_catalog:
    # shahaf2023b_class3`` (El-Badry 2026 ``sub_chandrasekhar``). Null: the catalog is
    # unavailable and such a subsample keeps no source; configured but missing is a
    # hard error.
    shahaf2023b_class3_snapshot: str | None = None
    # dark-hunter_rv Gaia_DR3_*_summary.json tree (null disables attachment).
    rv_summary_root: str | None = None
    rv_summary_filename_template: str = "Gaia_DR3_{source_id}_summary.json"
    # VizieR Apsis snapshot (``scripts/fetch_vizier_apsis.py``, #306): meta.yaml
    # of the I/355/paramp columns the Andrews ATF notebook read, relative to the
    # repo root unless absolute. Null = none.
    vizier_apsis_snapshot_meta: str | None = None
    # Reserved for DR4 epoch capabilities; ignored when inactive.
    allow_astrometric_epoch_outliers: bool = False
    selection_function_astrometric: DRSelectionFunctionPathConfig = Field(
        default_factory=DRSelectionFunctionPathConfig
    )
    selection_function_followup: DRSelectionFunctionFollowupPathConfig = Field(
        default_factory=DRSelectionFunctionFollowupPathConfig
    )
    # Statistical epoch-loss model wrapped around gaiamock's GOST transit list
    # (#400, docs/EPOCH_MODEL_SPEC.md). Null: gaiamock's epochs are used unchanged.
    epoch_model: EpochModelPathConfig | None = None

    @model_validator(mode="after")
    def _validate_crossmatch_fanout_maskable_bands(self) -> DRPathConfig:
        """Every maskable fan-out band must be a configured, enabled cross-match band."""
        known = {m.band for m in self.external_photometry_crossmatches if m.enabled}
        unknown = sorted(set(self.crossmatch_fanout_maskable_bands) - known)
        if unknown:
            raise ValueError(
                "crossmatch_fanout_maskable_bands names band(s) with no enabled "
                f"external_photometry_crossmatches entry: {unknown}"
            )
        return self


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
    spectroscopic_mass_function: SpectroscopicMassFunctionConfig = Field(
        default_factory=SpectroscopicMassFunctionConfig
    )
    rv_consistency: RvConsistencyConfig = Field(default_factory=RvConsistencyConfig)
    companion_nature: CompanionNatureConfig = Field(
        default_factory=CompanionNatureConfig
    )
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    mc_mass_function: McMassFunctionConfig = Field(default_factory=McMassFunctionConfig)
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
    plotting: PlottingStyleConfig = Field(default_factory=PlottingStyleConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)
    triples: TriplesConfig = Field(default_factory=TriplesConfig)
    sample_selection: SampleSelectionConfig = Field(
        default_factory=SampleSelectionConfig
    )
    spuriousness_model: SpuriousnessModelConfig = Field(
        default_factory=SpuriousnessModelConfig
    )
    multi_solution_rates: MultiSolutionRatesConfig = Field(
        default_factory=MultiSolutionRatesConfig
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
        "gaia_archive_row_limit",
        "gaia_archive_async",
        "pseudo_circular_eccentricity_max",
        "external_mag_err_floor",
        "external_mag_err_zero_as_missing",
        "impute_external_mag_err",
        "crossmatch_fanout_maskable_bands",
        "nss_table",
        "nss_enrichment_snapshot",
        "shahaf2023b_class3_snapshot",
        "gaia_source_table",
        "allow_astrometric_epoch_outliers",
        "accel_jerk_catalog_id",
        "d_min_pc",
        "d_max_pc",
        "a0_over_err_min",
        "goodness_of_fit_f2_max",
        "parallax_over_error_times_period_min_days",
        "a0_over_err_times_sqrt_period_min",
        "sigma_e_ln_period_slope",
        "sigma_e_intercept",
        "elbadry2024_comparison_nss_solution_types",
        "acceleration_significance_min",
        "acceleration7_f2_max",
        "acceleration9_f2_max",
        "mock_g_mag_max",
        "epoch_model",
    }
)

# Genuine physics/population sections that MUST be single shared top-level keys.
SHARED_PHYSICS_SECTIONS: frozenset[str] = frozenset(
    {
        "mass_calibration",
        "classification",
        "physics",
        "mc_mass_function",
    }
)

# Keys included in the resume/amend config checksum (ARCHITECTURE.md §5).
SHARED_CHECKSUM_SECTIONS: tuple[str, ...] = (
    "mass_calibration",
    "mass_derivation",
    "spectroscopic_mass_function",
    "rv_consistency",
    "companion_nature",
    "classification",
    "physics",
    "mc_mass_function",
    "gaiamock",
    "paths",
    "selection_function_astrometric",
    "selection_function_followup",
    "sample_selection",
    "spuriousness_model",
    "multi_solution_rates",
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
