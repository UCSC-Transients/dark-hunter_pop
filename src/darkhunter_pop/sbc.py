"""Simulation-based calibration (SBC) recovery + credible-interval coverage.

ARCHITECTURE.md §4 ``diagnostics`` / issue #69. Injects multiple distinct mass
functions, recovers through the staged inference path (population bin edges +
SF scalars + ``inference``), and reports empirical coverage of config-level
credible intervals across repeats.

``analytic_binned``: independent Gamma posteriors on free-height bins (fast
unit/physics path; same Poisson × SF mean model as ``inference.binned``).
``dynesty``: full ``run_inference`` nested-sampling recovery (prefer slow suites).

Diagnostic reports stay full-detail (caveman exemption).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.config_schema import PipelineConfig, SBCConfig
from darkhunter_pop.inference import ObservedEvent, run_inference
from darkhunter_pop.population_model import build_mass_bin_edges

SCHEMA_VERSION = 1

RecoveryBackend = Literal["analytic_binned", "dynesty"]


@dataclass(frozen=True)
class InjectedTruth:
    """One scaled free-height vector ready for injection."""

    profile_name: str
    heights: NDArray[np.floating]
    bin_edges_msun: NDArray[np.floating]


@dataclass(frozen=True)
class RepeatCoverageRecord:
    """Per-repeat, per-bin credible-interval check."""

    profile_name: str
    repeat_index: int
    bin_index: int
    truth: float
    ci_low: float
    ci_high: float
    covered: bool
    posterior_median: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "repeat_index": self.repeat_index,
            "bin_index": self.bin_index,
            "truth": self.truth,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "covered": self.covered,
            "posterior_median": self.posterior_median,
        }


@dataclass
class ProfileCoverageSummary:
    """Coverage aggregate for one injected mass-function profile."""

    profile_name: str
    n_checks: int
    n_covered: int
    empirical_coverage: float
    target_coverage: float
    abs_error: float
    passed: bool
    within_tolerance: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "n_checks": self.n_checks,
            "n_covered": self.n_covered,
            "empirical_coverage": self.empirical_coverage,
            "target_coverage": self.target_coverage,
            "abs_error": self.abs_error,
            "passed": self.passed,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class SBCResult:
    """Full SBC suite output (reports + HDF5)."""

    schema_version: int
    recovery_backend: RecoveryBackend
    credible_interval_level: float
    coverage_abs_tolerance: float
    n_repeats: int
    n_mass_bins: int
    astrometric_sf: float
    followup_sf: float
    profile_summaries: list[ProfileCoverageSummary]
    records: list[RepeatCoverageRecord]
    overall_empirical_coverage: float
    overall_passed: bool
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": "diagnostics_sbc",
            "recovery_backend": self.recovery_backend,
            "credible_interval_level": self.credible_interval_level,
            "coverage_abs_tolerance": self.coverage_abs_tolerance,
            "n_repeats": self.n_repeats,
            "n_mass_bins": self.n_mass_bins,
            "astrometric_sf": self.astrometric_sf,
            "followup_sf": self.followup_sf,
            "overall_empirical_coverage": self.overall_empirical_coverage,
            "overall_passed": self.overall_passed,
            "profile_summaries": [p.as_dict() for p in self.profile_summaries],
            "n_records": len(self.records),
            "records": [r.as_dict() for r in self.records],
            "config_snapshot": self.config_snapshot,
            "notes": self.notes
            or (
                "Simulation-based calibration: distinct injected free-height mass "
                "functions recovered through the staged Poisson×SF inference path; "
                "equal-tailed credible-interval coverage asserted vs config tolerance."
            ),
        }


def scale_injected_heights(
    relative_heights: Sequence[float],
    *,
    expected_total_rate: float,
) -> NDArray[np.floating]:
    """Scale a relative free-height profile so Σ h_i = ``expected_total_rate``."""
    rel = np.asarray(relative_heights, dtype=np.float64)
    if rel.size < 2:
        raise ValueError("relative_heights must have length >= 2")
    if np.any(rel <= 0.0):
        raise ValueError("relative_heights must be strictly positive")
    total = float(np.sum(rel))
    return rel * (float(expected_total_rate) / total)


def build_injected_truths(
    config: PipelineConfig,
    *,
    sbc: SBCConfig | None = None,
) -> list[InjectedTruth]:
    """Materialize distinct injected MF truths from config profiles + bin edges."""
    cfg = sbc if sbc is not None else config.diagnostics.sbc
    pop = config.population_model.model_copy(deep=True)
    pop.n_mass_bins = cfg.n_mass_bins
    # Fiducial counts only size the equal_fiducial_count policy; keep positive.
    pop.fiducial_expected_counts = [1.0] * cfg.n_mass_bins
    edges = build_mass_bin_edges(pop)
    truths: list[InjectedTruth] = []
    for profile in cfg.injected_profiles:
        heights = scale_injected_heights(
            profile.relative_heights, expected_total_rate=cfg.expected_total_rate
        )
        truths.append(
            InjectedTruth(
                profile_name=profile.name,
                heights=heights,
                bin_edges_msun=edges,
            )
        )
    return truths


def inject_binned_counts(
    true_heights: ArrayLike,
    *,
    astrometric_sf: float,
    followup_sf: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Draw Poisson counts with mean ``h_i × P_astro × P_followup``."""
    h = np.asarray(true_heights, dtype=np.float64)
    mu = h * float(astrometric_sf) * float(followup_sf)
    return rng.poisson(mu).astype(np.float64)


def inject_unbinned_events(
    true_heights: ArrayLike,
    *,
    bin_edges: NDArray[np.floating],
    astrometric_sf: float,
    followup_sf: float,
    rng: np.random.Generator,
    source_id_start: int = 1,
) -> list[ObservedEvent]:
    """Draw inhomogeneous Poisson events from free-height dN/dM × SF scalars.

    Within each bin the intensity is uniform in mass (piecewise-constant dN/dM).
    Per-bin counts are Poisson with mean ``h_i × SF_a × SF_f`` — matching the
    binned inference mean model — then masses are drawn uniformly in the bin.
    """
    h = np.asarray(true_heights, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if h.size != edges.size - 1:
        raise ValueError("true_heights length must equal n_bins")
    sf = float(astrometric_sf) * float(followup_sf)
    events: list[ObservedEvent] = []
    sid = int(source_id_start)
    for i, height in enumerate(h):
        n_i = int(rng.poisson(float(height) * sf))
        if n_i <= 0:
            continue
        masses = rng.uniform(edges[i], edges[i + 1], size=n_i)
        for mass in masses:
            events.append(
                ObservedEvent(
                    source_id=sid,
                    mass_msun=float(mass),
                    weight=1.0,
                )
            )
            sid += 1
    return events


def equal_tailed_credible_interval(
    samples: NDArray[np.floating],
    *,
    level: float,
) -> tuple[float, float]:
    """Equal-tailed credible interval from 1-D posterior samples."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    alpha = 1.0 - level
    lo = float(np.quantile(arr, 0.5 * alpha))
    hi = float(np.quantile(arr, 1.0 - 0.5 * alpha))
    return lo, hi


def truth_in_interval(truth: float, lo: float, hi: float) -> bool:
    """Inclusive containment check for coverage bookkeeping."""
    return bool(lo <= truth <= hi)


def analytic_binned_height_samples(
    counts: ArrayLike,
    *,
    astrometric_sf: float,
    followup_sf: float,
    n_samples: int,
    rng: np.random.Generator,
    prior_shape: float = 1.0,
    prior_rate: float = 1.0e-12,
) -> NDArray[np.floating]:
    """Independent Gamma posteriors on free heights (Poisson × SF mean model).

    Model: ``n_i ~ Poisson(μ_i)``, ``μ_i = h_i × S``, ``S = SF_a × SF_f``.
    Prior ``μ_i ~ Gamma(prior_shape, rate=prior_rate)`` yields
    ``μ_i | n_i ~ Gamma(prior_shape + n_i, rate=prior_rate + 1)``.
    Transform ``h_i = μ_i / S``. Near-improper ``prior_rate`` keeps the
    conjugacy path numerically stable while staying weakly informative.
    """
    n = np.asarray(counts, dtype=np.float64).ravel()
    s = float(astrometric_sf) * float(followup_sf)
    if s <= 0.0:
        raise ValueError("SF product must be > 0")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    out = np.empty((n_samples, n.size), dtype=np.float64)
    for i, count in enumerate(n):
        shape = float(prior_shape) + float(count)
        rate = float(prior_rate) + 1.0
        mu = rng.gamma(shape, 1.0 / rate, size=n_samples)
        out[:, i] = mu / s
    return out


def recover_heights_analytic(
    counts: ArrayLike,
    *,
    sbc: SBCConfig,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Analytic binned recovery → equal-weight height samples ``(n_draw, n_bin)``."""
    return analytic_binned_height_samples(
        counts,
        astrometric_sf=sbc.astrometric_sf,
        followup_sf=sbc.followup_sf,
        n_samples=sbc.n_posterior_samples,
        rng=rng,
    )


def recover_heights_dynesty(
    config: PipelineConfig,
    *,
    events: Sequence[ObservedEvent],
    bin_edges: NDArray[np.floating],
    sbc: SBCConfig,
) -> NDArray[np.floating]:
    """Staged inference recovery via ``run_inference`` (dynesty nested sampling)."""
    cfg = config.model_copy(deep=True)
    cfg.inference.skip_sampler = False
    cfg.inference.likelihood_form = "binned"
    cfg.inference.apply_sensitivity_dimensionality = False
    cfg.inference.nlive = sbc.inference_nlive
    cfg.inference.maxcall = sbc.inference_maxcall
    cfg.inference.dlogz = sbc.inference_dlogz
    cfg.inference.n_mass_grid = sbc.inference_n_mass_grid
    cfg.inference.n_robustness_runs = 1
    cfg.inference.default_astrometric_sf = sbc.astrometric_sf
    cfg.inference.default_followup_sf = sbc.followup_sf
    cfg.population_model.n_mass_bins = int(bin_edges.size - 1)
    cfg.population_model.fiducial_expected_counts = [1.0] * (bin_edges.size - 1)

    population_payload: dict[str, Any] = {
        "bin_edges_msun": np.asarray(bin_edges, dtype=np.float64).tolist(),
        "bin_heights": [1.0] * (bin_edges.size - 1),
        "n_systems": len(events),
        "system_weight_rows": [],
    }
    result = run_inference(
        cfg,
        population_payload=population_payload,
        events=list(events),
    )
    if not result.sampler_runs:
        raise RuntimeError("dynesty SBC recovery produced no sampler runs")
    preview = result.sampler_runs[0].get("samples_preview")
    if preview is None or len(preview) == 0:
        raise RuntimeError("dynesty SBC recovery missing posterior samples_preview")
    return np.asarray(preview, dtype=np.float64)


def _summarize_profile(
    profile_name: str,
    records: Sequence[RepeatCoverageRecord],
    *,
    target: float,
    abs_tol: float,
) -> ProfileCoverageSummary:
    subset = [r for r in records if r.profile_name == profile_name]
    n_checks = len(subset)
    n_covered = sum(1 for r in subset if r.covered)
    empirical = float(n_covered / n_checks) if n_checks else float("nan")
    abs_err = float(abs(empirical - target)) if n_checks else float("nan")
    within = bool(n_checks > 0 and abs_err <= abs_tol)
    return ProfileCoverageSummary(
        profile_name=profile_name,
        n_checks=n_checks,
        n_covered=n_covered,
        empirical_coverage=empirical,
        target_coverage=target,
        abs_error=abs_err,
        passed=within,
        within_tolerance=within,
    )


def run_sbc_suite(
    config: PipelineConfig,
    *,
    sbc: SBCConfig | None = None,
    recovery_backend: RecoveryBackend | None = None,
    n_repeats: int | None = None,
    random_seed: int | None = None,
) -> SBCResult:
    """Run multi-profile injection → recovery → coverage SBC.

    Parameters
    ----------
    config:
        Pipeline config (population bin-edge policy + diagnostics.sbc defaults).
    sbc:
        Optional override of ``config.diagnostics.sbc``.
    recovery_backend / n_repeats / random_seed:
        Optional call-site overrides (tests); config remains the source of truth
        for tolerances and injected profiles.
    """
    cfg = sbc if sbc is not None else config.diagnostics.sbc
    if not cfg.enabled:
        raise ValueError("diagnostics.sbc.enabled is False; refusing to run SBC")

    backend: RecoveryBackend = (
        recovery_backend if recovery_backend is not None else cfg.recovery_backend
    )
    repeats = int(n_repeats if n_repeats is not None else cfg.n_repeats)
    seed = int(random_seed if random_seed is not None else cfg.random_seed)
    if repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    truths = build_injected_truths(config, sbc=cfg)
    records: list[RepeatCoverageRecord] = []
    level = float(cfg.credible_interval_level)
    tol = float(cfg.coverage_abs_tolerance)

    for p_idx, truth in enumerate(truths):
        for rep in range(repeats):
            rng = np.random.default_rng(seed + 1009 * p_idx + 17 * rep)
            if backend == "analytic_binned":
                counts = inject_binned_counts(
                    truth.heights,
                    astrometric_sf=cfg.astrometric_sf,
                    followup_sf=cfg.followup_sf,
                    rng=rng,
                )
                samples = recover_heights_analytic(counts, sbc=cfg, rng=rng)
            elif backend == "dynesty":
                events = inject_unbinned_events(
                    truth.heights,
                    bin_edges=truth.bin_edges_msun,
                    astrometric_sf=cfg.astrometric_sf,
                    followup_sf=cfg.followup_sf,
                    rng=rng,
                    source_id_start=1 + 10_000 * p_idx + 100 * rep,
                )
                samples = recover_heights_dynesty(
                    config,
                    events=events,
                    bin_edges=truth.bin_edges_msun,
                    sbc=cfg,
                )
            else:
                raise ValueError(f"unknown recovery_backend: {backend!r}")

            for b in range(truth.heights.size):
                lo, hi = equal_tailed_credible_interval(samples[:, b], level=level)
                med = float(np.median(samples[:, b]))
                tval = float(truth.heights[b])
                records.append(
                    RepeatCoverageRecord(
                        profile_name=truth.profile_name,
                        repeat_index=rep,
                        bin_index=b,
                        truth=tval,
                        ci_low=lo,
                        ci_high=hi,
                        covered=truth_in_interval(tval, lo, hi),
                        posterior_median=med,
                    )
                )

    summaries = [
        _summarize_profile(t.profile_name, records, target=level, abs_tol=tol)
        for t in truths
    ]
    n_all = len(records)
    n_cov = sum(1 for r in records if r.covered)
    overall = float(n_cov / n_all) if n_all else float("nan")
    overall_passed = bool(n_all > 0 and abs(overall - level) <= tol)

    return SBCResult(
        schema_version=SCHEMA_VERSION,
        recovery_backend=backend,
        credible_interval_level=level,
        coverage_abs_tolerance=tol,
        n_repeats=repeats,
        n_mass_bins=cfg.n_mass_bins,
        astrometric_sf=cfg.astrometric_sf,
        followup_sf=cfg.followup_sf,
        profile_summaries=summaries,
        records=records,
        overall_empirical_coverage=overall,
        overall_passed=overall_passed,
        config_snapshot=cfg.model_dump(mode="json"),
        notes=(
            "SBC uses config-driven injected profiles, SF scalars, credible-interval "
            f"level={level}, and coverage_abs_tolerance={tol}. "
            f"Backend={backend}: analytic_binned matches the binned Poisson×SF mean; "
            "dynesty calls staged run_inference with fixed plug-in event weights."
        ),
    )


def assert_sbc_coverage(result: SBCResult) -> None:
    """Raise ``AssertionError`` with a full-detail message if coverage fails."""
    if result.overall_passed and all(p.passed for p in result.profile_summaries):
        return
    lines = [
        "SBC credible-interval coverage failed config-driven tolerance.",
        f"target_level={result.credible_interval_level}",
        f"coverage_abs_tolerance={result.coverage_abs_tolerance}",
        f"overall_empirical_coverage={result.overall_empirical_coverage}",
        f"overall_passed={result.overall_passed}",
        "per_profile:",
    ]
    for p in result.profile_summaries:
        lines.append(
            f"  {p.profile_name}: empirical={p.empirical_coverage:.4f} "
            f"abs_error={p.abs_error:.4f} passed={p.passed} "
            f"(n_covered={p.n_covered}/{p.n_checks})"
        )
    raise AssertionError("\n".join(lines))


def format_sbc_report(result: SBCResult) -> str:
    """Fully legible SBC report (exempt from caveman compression)."""
    lines = [
        "=== simulation-based calibration (SBC) ===",
        f"schema_version: {result.schema_version}",
        f"recovery_backend: {result.recovery_backend}",
        f"credible_interval_level: {result.credible_interval_level}",
        f"coverage_abs_tolerance: {result.coverage_abs_tolerance}",
        f"n_repeats: {result.n_repeats}",
        f"n_mass_bins: {result.n_mass_bins}",
        f"astrometric_sf: {result.astrometric_sf}",
        f"followup_sf: {result.followup_sf}",
        f"overall_empirical_coverage: {result.overall_empirical_coverage:.6f}",
        f"overall_passed: {result.overall_passed}",
        f"n_records: {len(result.records)}",
        "profile_summaries:",
    ]
    for p in result.profile_summaries:
        lines.append(
            f"  - {p.profile_name}: empirical_coverage={p.empirical_coverage:.6f} "
            f"target={p.target_coverage:.6f} abs_error={p.abs_error:.6f} "
            f"n_covered={p.n_covered}/{p.n_checks} passed={p.passed}"
        )
    lines.append("coverage_detail (first 32 records):")
    for rec in result.records[:32]:
        lines.append(
            f"  profile={rec.profile_name} repeat={rec.repeat_index} "
            f"bin={rec.bin_index} truth={rec.truth:.6g} "
            f"CI=[{rec.ci_low:.6g}, {rec.ci_high:.6g}] "
            f"median={rec.posterior_median:.6g} covered={rec.covered}"
        )
    if len(result.records) > 32:
        lines.append(f"  ... ({len(result.records) - 32} additional records omitted)")
    lines.append("notes:")
    lines.append(f"  {result.notes}")
    lines.append(
        "interpretation: empirical coverage should match the configured credible-"
        "interval level within coverage_abs_tolerance across distinct injected "
        "mass functions. Failures indicate miscalibrated recovery or too-small "
        "Monte Carlo / nested-sampling budgets for the chosen tolerance."
    )
    lines.append("=== end SBC report ===")
    return "\n".join(lines)


def write_sbc_artifact(path: Path, result: SBCResult) -> None:
    """Persist SBC payload to HDF5 under diagnostics artifact paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "diagnostics_sbc"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["recovery_backend"] = result.recovery_backend
        handle.attrs["overall_passed"] = result.overall_passed
        handle.attrs["overall_empirical_coverage"] = result.overall_empirical_coverage
        handle.attrs["credible_interval_level"] = result.credible_interval_level
        handle.create_dataset(
            "payload_json",
            data=np.array(
                json.dumps(payload, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        if result.records:
            handle.create_dataset(
                "covered",
                data=np.asarray([r.covered for r in result.records], dtype=bool),
            )
            handle.create_dataset(
                "truth",
                data=np.asarray([r.truth for r in result.records], dtype=np.float64),
            )
            handle.create_dataset(
                "ci_low",
                data=np.asarray([r.ci_low for r in result.records], dtype=np.float64),
            )
            handle.create_dataset(
                "ci_high",
                data=np.asarray([r.ci_high for r in result.records], dtype=np.float64),
            )


def read_sbc_artifact(path: Path) -> dict[str, Any]:
    """Load SBC payload JSON from an HDF5 artifact."""
    with h5py.File(path, "r") as handle:
        raw = handle["payload_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        return json.loads(text)


