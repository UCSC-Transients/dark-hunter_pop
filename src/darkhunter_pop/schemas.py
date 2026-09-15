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

    ``nss_solution`` (issue #105): fitted NSS parameter vector + covariance from archive
    ``corr_vec``/``bit_index``. Absent when reconstruction fails; never replaced by
    independent (diagonal-only) errors.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: int
    nss_solution_type: str | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    parallax_mas: float | None = None
    thiele_innes: ThieleInnesElements | None = None
    nss_orbital: dict[str, Any] = Field(default_factory=dict)
    # Full NSS fitted-parameter vector + covariance (corr_vec/bit_index unpack).
    # Absent when reconstruction fails — never replaced by a diagonal-only fallback.
    nss_solution: ParameterSet | None = None
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


#: Recognized :attr:`SyntheticStandIn.kind` values.
#:
#: ``analytic_surrogate``
#:     A closed-form relation standing in for a real fit / model comparison.
#: ``config_placeholder``
#:     A config value that is a deliberate placeholder, not a measured or
#:     forward-modeled quantity.
#: ``offline_replay``
#:     Real data replayed from a local snapshot instead of re-acquired live.
#: ``ci_scale_sampler``
#:     Sampler settings deliberately kept at CI smoke scale.
#: ``disabled_path``
#:     A pipeline path deliberately left off, so its contribution is absent.
STAND_IN_KINDS: Final[frozenset[str]] = frozenset(
    {
        "analytic_surrogate",
        "config_placeholder",
        "offline_replay",
        "ci_scale_sampler",
        "disabled_path",
    }
)


class SyntheticStandIn(BaseModel):
    """One documented substitution used in place of a real pipeline input.

    Recorded on :class:`RunManifest` and printed in the run plan, so a run built
    on placeholders can never be mistaken for a science result (issue #201). A
    stand-in is *declared*, never inferred: whichever harness injects it owns
    listing it here.

    Attributes
    ----------
    name:
        Short stable identifier, unique within one run (e.g.
        ``per_sample_selection_weights``).
    stage:
        Registered stage name the substitution affects, or ``"pipeline"`` when it
        is not scoped to a single stage.
    kind:
        Which *sort* of substitution this is — one of :data:`STAND_IN_KINDS`.
    replaces:
        The real input that does not yet exist, in one phrase.
    description:
        Full-detail prose (exempt from caveman compression): what the stand-in
        actually is, and why nothing downstream of it is a result.
    config_keys:
        Dotted config keys carrying the placeholder values, so a reader can go
        read the numbers rather than trust this prose.
    values:
        Optional snapshot of the placeholder values as resolved at run time.
        Free-form but JSON-serializable only (it is written to YAML).

    Limitations
    -----------
    This record is documentation, not enforcement. Nothing verifies that a
    declared stand-in is the *only* substitution in effect, and nothing stops a
    stage from using a placeholder without declaring one.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    stage: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    replaces: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    config_keys: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in STAND_IN_KINDS:
            raise ValueError(
                f"unknown stand-in kind {value!r}; known: {sorted(STAND_IN_KINDS)}"
            )
        return value

    def one_line(self) -> str:
        """One run-plan line: ``<name> [<kind>] @<stage> — replaces <replaces>``."""
        return f"{self.name} [{self.kind}] @{self.stage} — replaces {self.replaces}"


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
    # Measured cost of this stage's execution, when the caller monitored it.
    # All three stay None for cached / skipped stages and for any stage run
    # without a resource monitor attached (issue #201, EXECUTION_PLAN.md §5.6).
    wall_clock_seconds: float | None = None
    #: Peak resident set size sampled *during this stage only*, in bytes.
    peak_rss_bytes: int | None = None
    #: Process-lifetime RSS high-water mark read at this stage's end, in bytes.
    #: Exact (``getrusage``) but cumulative, so it never decreases across
    #: stages — the cross-check on the sampled per-stage figure.
    cumulative_peak_rss_bytes: int | None = None


class RunManifest(BaseModel):
    """Live YAML run file under ``runs/{run_id}.yaml`` (ARCHITECTURE.md §5)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    created_at: datetime
    parent_run_id: str | None = None
    config_checksum: str = Field(..., min_length=1)
    active_dr_mode: ActiveDRMode = ActiveDRMode.DR3
    artifact_root: str = "output"
    # Checked-in host profile that produced the path-shaped config keys for this run
    # (config/host_profiles/<name>.yaml), or None when no profile was selected
    # (issue #196). Explicit selection only — never inferred from hostname.
    host_profile: str | None = None
    gaiamock_mod_release: str | None = None
    gaiamock_mod_sha256: str | None = None
    gaiamock_git_commit: str | None = None
    # Sampler / MC seeds actually used (accounting, not bitwise replay).
    random_seeds: dict[str, Any] = Field(default_factory=dict)
    # True when this run was built on documented substitutions rather than the
    # real inputs, so nothing it produces is a result (issue #201). Set at run
    # *birth*, never after the fact, and printed in the run plan. Present in the
    # YAML for every run, so a dry run is distinguishable from a science run by
    # grepping the file rather than by reading it.
    dry_run: bool = False
    #: Human-readable banner carried by every report and figure caption built
    #: from this run. Required whenever ``dry_run`` is true.
    dry_run_label: str | None = None
    #: Every substitution in effect for this run (issue #201). Must be non-empty
    #: when ``dry_run`` is true — a dry run with nothing declared is a dry run
    #: whose stand-ins were not enumerated.
    synthetic_stand_ins: list[SyntheticStandIn] = Field(default_factory=list)
    stages: dict[str, StageRecord] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dry_run_must_be_labeled(self) -> RunManifest:
        """A dry run carries a label and at least one declared stand-in.

        Enforces the Wave 0 "label it at birth" requirement structurally rather
        than by convention (EXECUTION_PLAN.md §7, issue #201): a manifest with
        ``dry_run=True`` cannot be saved without the banner text and without
        naming what was substituted.
        """
        if self.dry_run:
            if not (self.dry_run_label or "").strip():
                raise ValueError(
                    "dry_run=True requires a non-empty dry_run_label "
                    "(label the run at birth — EXECUTION_PLAN.md §7)"
                )
            if not self.synthetic_stand_ins:
                raise ValueError(
                    "dry_run=True requires at least one entry in "
                    "synthetic_stand_ins (enumerate every substitution)"
                )
        return self

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
