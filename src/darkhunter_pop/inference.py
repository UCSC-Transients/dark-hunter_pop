"""Stage: ``inference``.

Staged-but-connected inhomogeneous Poisson point-process likelihood sampled with ``dynesty``
(ARCHITECTURE.md §4, issue #63).

**v1 strategy**: ``rv_astrometry_gate`` and ``companion_nature_likelihood`` results are
computed once upstream and enter here as **fixed** per-system empirical-Bayes plug-in
weights (via the ``population_model`` artifact). No joint re-sampling of those stages.

**v2 (documented, not built)**: fully joint population + selection + per-system evidence,
warm-started from the v1 posterior.

Likelihood rate:

``λ(θ) = population_model(θ) × astrometric_SF × followup_SF``

Poisson primitives come from ``physics_utils``. Dimensionality / binned-vs-unbinned advice
from ``sensitivity_analysis`` is applied only when ``inference.apply_sensitivity_dimensionality``
is true — SA never silently rewrites this module's defaults.

Reproducibility: multi-run robustness protocol (independent seeds / live-point counts),
**not** bitwise seed identity. Cluster recipe lives in ``config/fragments/inference.yaml``
comments; CI keeps tiny ``nlive`` / ``maxcall``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.config_loader import effective_M_Ch_msun
from darkhunter_pop.config_schema import InferenceConfig, PipelineConfig
from darkhunter_pop.physics_utils import (
    binned_poisson_log_likelihood,
    integrate_intensity_trapezoid,
    poisson_log_likelihood_inhomogeneous,
    poisson_upper_limit_zero_events,
)
from darkhunter_pop.population_model import (
    build_mass_bin_edges,
    default_bin_heights,
    eccentricity_pdf,
    evaluate_mass_function,
    read_population_model_artifact,
    run_population_model,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import RunManifest, StageStatus
from darkhunter_pop.sensitivity_analysis import read_sensitivity_analysis_artifact

try:
    import dynesty
    from dynesty import utils as dyfunc
except ImportError:  # optional: pip install -e ".[inference]"
    dynesty = None  # type: ignore[assignment]
    dyfunc = None  # type: ignore[assignment]

SCHEMA_VERSION = 1

LikelihoodForm = Literal["unbinned", "binned"]

# ---------------------------------------------------------------------------
# Multi-run robustness protocol (documented contract)
# ---------------------------------------------------------------------------

ROBUSTNESS_PROTOCOL = """
dynesty multi-run robustness protocol (ARCHITECTURE.md §4; workflow §7)
-----------------------------------------------------------------------
Do **not** treat identical random seeds as a reproducibility check. Nested
sampling explores a stochastic evidence integral; bitwise replay is neither
required nor informative.

Cluster / science runs:
  1. Set inference.n_robustness_runs >= 3.
  2. Run k independent nested-sampling jobs with seeds
       seed_k = random_seed + k * robustness_seed_stride
     and nlive_k = ceil(nlive * robustness_nlive_scale**k) for k = 0..N-1
     (or hold nlive fixed and only vary seeds — both are valid).
  3. Compare posterior medians and 68% credible intervals on each free-height
     bin (and logZ). Require science-chosen agreement (e.g. overlapping 68%
     intervals or median shifts << interval width).
  4. Record all run seeds, nlive, logZ, and the agreement summary in the
     inference HDF5 artifact / run file.

CI keeps n_robustness_runs=1 with tiny nlive/maxcall (smoke only).
"""


# ---------------------------------------------------------------------------
# Selection-function scalars (simplified / artifact plug-in path)
# ---------------------------------------------------------------------------


def read_astrometric_sf_scalar(
    path: Path | None,
    *,
    default: float,
) -> float:
    """Scalar astrometric selection probability from stage HDF5 (or default)."""
    if path is None or not path.is_file():
        return float(default)
    with h5py.File(path, "r") as handle:
        if "detection_fraction" in handle.attrs:
            return float(handle.attrs["detection_fraction"])
        if "mock_catalog" in handle and "accepted_orbital" in handle["mock_catalog"]:
            accepted = np.asarray(handle["mock_catalog"]["accepted_orbital"], dtype=bool)
            if accepted.size:
                return float(np.mean(accepted))
    return float(default)


def read_followup_sf_scalar(
    path: Path | None,
    *,
    default: float,
) -> float:
    """Mean follow-up selection probability from stage HDF5 (or default)."""
    if path is None or not path.is_file():
        return float(default)
    with h5py.File(path, "r") as handle:
        if "followup_catalog" in handle and "probability" in handle["followup_catalog"]:
            probs = np.asarray(handle["followup_catalog"]["probability"], dtype=np.float64)
            if probs.size:
                return float(np.mean(probs))
    return float(default)


# ---------------------------------------------------------------------------
# Dimensionality / likelihood form
# ---------------------------------------------------------------------------


def resolve_likelihood_form(
    cfg: InferenceConfig,
    sensitivity_payload: Mapping[str, Any] | None,
) -> LikelihoodForm:
    """Choose unbinned vs binned; prefer SA advice when opted in (small-N default: unbinned)."""
    if cfg.likelihood_form == "unbinned":
        return "unbinned"
    if cfg.likelihood_form == "binned":
        return "binned"
    # auto
    if cfg.apply_sensitivity_dimensionality and sensitivity_payload is not None:
        dim = sensitivity_payload.get("dimensionality") or {}
        preferred = dim.get("preferred_likelihood")
        if preferred in ("unbinned", "binned"):
            return preferred  # type: ignore[return-value]
    return "unbinned"


# ---------------------------------------------------------------------------
# Observed events + model-comparison plug-in weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedEvent:
    """One detected system entering the Poisson likelihood (fixed plug-in weights)."""

    source_id: int
    mass_msun: float
    weight: float
    eccentricity: float | None = None


def _circular_implies_wd_weight(
    responsibilities: Mapping[str, float],
    *,
    eccentricity: float | None,
    enabled: bool,
    e_threshold: float,
) -> dict[str, float]:
    """Boost WD responsibility for near-circular systems when the switch is on."""
    out = {k: float(v) for k, v in responsibilities.items()}
    if not enabled or eccentricity is None or eccentricity > e_threshold:
        return out
    # Soft boost: double WD mass then renormalize (scenario switch, not a hard cut).
    if "WD" in out:
        out["WD"] *= 2.0
    total = sum(out.values())
    if total <= 0:
        n = len(out)
        return {k: 1.0 / n for k in out}
    return {k: v / total for k, v in out.items()}


def collect_observed_events(
    population_payload: Mapping[str, Any],
    *,
    cfg: InferenceConfig,
    eccentricities: Mapping[int, float] | None = None,
) -> list[ObservedEvent]:
    """Build weighted events from ``population_model`` system_weights (+ optional e).

    The event weight is the compact-object responsibility sum (BH+NS+WD), optionally
    reweighted by the eccentricity hypothesis PDF and the circular⇒WD switch.
    Missing masses fall back to geometric-mean midpoints of the MF bins.
    """
    edges = np.asarray(population_payload["bin_edges_msun"], dtype=np.float64)
    mids = np.sqrt(edges[:-1] * edges[1:])
    n_sys = int(population_payload.get("n_systems", 0))
    events: list[ObservedEvent] = []

    rows = population_payload.get("system_weight_rows") or []
    if not rows and n_sys == 0:
        return events

    for i, row in enumerate(rows):
        source_id = int(row["source_id"])
        m2 = row.get("m2_msun")
        if m2 is None or (isinstance(m2, float) and not math.isfinite(m2)):
            mass = float(mids[i % mids.size])
        else:
            mass = float(m2)
        resp = dict(row["responsibilities"])
        e_val = None if eccentricities is None else eccentricities.get(source_id)
        if e_val is None and row.get("eccentricity") is not None:
            e_val = float(row["eccentricity"])
        resp = _circular_implies_wd_weight(
            resp,
            eccentricity=e_val,
            enabled=cfg.circular_implies_wd,
            e_threshold=cfg.circular_e_threshold,
        )
        co_w = float(resp.get("BH", 0.0) + resp.get("NS", 0.0) + resp.get("WD", 0.0))
        if e_val is not None:
            e_pdf = float(
                eccentricity_pdf(np.asarray([e_val]), cfg.eccentricity_hypothesis)[0]
            )
            co_w *= max(e_pdf, 1e-30)
        if co_w <= 0:
            continue
        events.append(
            ObservedEvent(
                source_id=source_id,
                mass_msun=mass,
                weight=co_w,
                eccentricity=e_val,
            )
        )
    return events


def events_from_population_result(
    result: Any,
    *,
    cfg: InferenceConfig,
    eccentricities: Mapping[int, float] | None = None,
) -> list[ObservedEvent]:
    """Adapter: ``PopulationModelResult.system_weights`` → ``ObservedEvent`` list."""
    rows = []
    for sw in result.system_weights:
        rows.append(
            {
                "source_id": sw.source_id,
                "m2_msun": sw.m2_msun,
                "responsibilities": dict(sw.responsibilities),
            }
        )
    payload = dict(result.recommendation_payload())
    payload["system_weight_rows"] = rows
    return collect_observed_events(payload, cfg=cfg, eccentricities=eccentricities)


# ---------------------------------------------------------------------------
# Intensity / likelihood
# ---------------------------------------------------------------------------


def mass_function_intensity(
    mass_msun: ArrayLike,
    heights: ArrayLike,
    *,
    bin_edges: NDArray[np.floating],
    population_cfg: Any,
    astrometric_sf: float,
    followup_sf: float,
) -> NDArray[np.floating]:
    """``λ(M) = MF(M; heights) × P_astro × P_followup``."""
    mf = evaluate_mass_function(
        mass_msun,
        model=population_cfg.mass_function_model,
        bin_edges=bin_edges,
        heights=np.asarray(heights, dtype=np.float64),
        cfg=population_cfg,
    )
    return mf * float(astrometric_sf) * float(followup_sf)


def unbinned_log_likelihood(
    heights: ArrayLike,
    *,
    event_masses: NDArray[np.floating],
    event_weights: NDArray[np.floating],
    mass_grid: NDArray[np.floating],
    bin_edges: NDArray[np.floating],
    population_cfg: Any,
    astrometric_sf: float,
    followup_sf: float,
) -> float:
    """Weighted inhomogeneous Poisson logL on mass (physics_utils primitives)."""
    lam_grid = mass_function_intensity(
        mass_grid,
        heights,
        bin_edges=bin_edges,
        population_cfg=population_cfg,
        astrometric_sf=astrometric_sf,
        followup_sf=followup_sf,
    )
    integrated = integrate_intensity_trapezoid(mass_grid, lam_grid)
    if event_masses.size == 0:
        return poisson_log_likelihood_inhomogeneous([], integrated)

    lam_evt = mass_function_intensity(
        event_masses,
        heights,
        bin_edges=bin_edges,
        population_cfg=population_cfg,
        astrometric_sf=astrometric_sf,
        followup_sf=followup_sf,
    )
    # Weighted events: Σ w_i log λ(x_i) − Λ  (w_i are fixed plug-in responsibilities).
    safe = np.maximum(lam_evt, 1e-300)
    log_term = float(np.sum(event_weights * np.log(safe)))
    return float(log_term - integrated)


def binned_log_likelihood(
    heights: ArrayLike,
    *,
    event_masses: NDArray[np.floating],
    event_weights: NDArray[np.floating],
    bin_edges: NDArray[np.floating],
    population_cfg: Any,
    astrometric_sf: float,
    followup_sf: float,
) -> float:
    """Binned Poisson logL using free-height expected counts × SF scalars."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    h = np.asarray(heights, dtype=np.float64)
    expected = h * float(astrometric_sf) * float(followup_sf)
    counts = np.zeros(h.size, dtype=np.float64)
    if event_masses.size:
        idx = np.searchsorted(edges, event_masses, side="right") - 1
        valid = (idx >= 0) & (idx < h.size)
        for i, w in zip(idx[valid], event_weights[valid], strict=True):
            counts[int(i)] += float(w)
    _ = population_cfg
    return binned_poisson_log_likelihood(counts, expected)


def zero_count_poisson_upper_limits(
    counts: ArrayLike,
    *,
    confidence: float,
) -> list[dict[str, float | int]]:
    """Explicit Poisson upper limits on mean count for every zero-count bin."""
    n = np.asarray(counts, dtype=np.float64)
    ul = poisson_upper_limit_zero_events(confidence)
    out: list[dict[str, float | int]] = []
    for i, c in enumerate(n.ravel()):
        if c == 0.0:
            out.append(
                {
                    "bin_index": i,
                    "count": 0.0,
                    "mu_upper": ul,
                    "confidence": confidence,
                }
            )
    return out


def posterior_vs_prior_overlap(
    samples: NDArray[np.floating],
    *,
    prior_low: NDArray[np.floating],
    prior_high: NDArray[np.floating],
    threshold: float,
) -> dict[str, Any]:
    """Small-N diagnostic: posterior width vs uniform-prior width.

    Uniform prior on ``[low, high]`` has ``σ_prior = width / √12``. The per-parameter
    ratio ``σ_post / σ_prior`` near 1 means the posterior did not beat the prior
    (prior-dominated / high overlap). Flag when the mean ratio exceeds ``threshold``.
    """
    if samples.size == 0:
        return {
            "per_param_width_ratio": [],
            "mean_width_ratio": float("nan"),
            "prior_dominated": True,
            "threshold": threshold,
        }
    width = np.asarray(prior_high, dtype=np.float64) - np.asarray(
        prior_low, dtype=np.float64
    )
    prior_std = width / math.sqrt(12.0)
    post_std = np.std(samples, axis=0)
    ratio = post_std / np.maximum(prior_std, 1e-30)
    mean_ratio = float(np.mean(ratio))
    return {
        "per_param_width_ratio": ratio.tolist(),
        "mean_width_ratio": mean_ratio,
        "prior_dominated": bool(mean_ratio > threshold),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# dynesty sampler
# ---------------------------------------------------------------------------


def _prior_transform(
    u: NDArray[np.floating],
    *,
    log_lo: float,
    log_hi: float,
) -> NDArray[np.floating]:
    """Unit cube → positive free-height vector via log-uniform prior."""
    log_h = log_lo + np.asarray(u, dtype=np.float64) * (log_hi - log_lo)
    return np.exp(log_h)


def run_dynesty_nested(
    loglike_fn,  # callable(heights) -> float
    *,
    ndim: int,
    cfg: InferenceConfig,
    seed: int,
    nlive: int | None = None,
) -> dict[str, Any]:
    """Run one ``dynesty.NestedSampler``; return summary + equal-weight samples.

    Cluster recipe (large nlive / small dlogz / n_robustness_runs≥3) is documented in
    ``config/fragments/inference.yaml`` — not the CI default.
    """
    if dynesty is None or dyfunc is None:
        raise ImportError(
            "dynesty is required for inference sampling; install with "
            'pip install -e ".[inference]"'
        )

    rng = np.random.default_rng(seed)
    live = int(nlive if nlive is not None else cfg.nlive)

    def _ptform(u: NDArray[np.floating]) -> NDArray[np.floating]:
        return _prior_transform(u, log_lo=cfg.log_height_min, log_hi=cfg.log_height_max)

    def _loglike(x: NDArray[np.floating]) -> float:
        return float(loglike_fn(x))

    sampler = dynesty.NestedSampler(
        _loglike,
        _ptform,
        ndim,
        nlive=live,
        sample=cfg.sample,
        rstate=rng,
    )
    sampler.run_nested(dlogz=cfg.dlogz, maxcall=cfg.maxcall, print_progress=False)
    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dyfunc.resample_equal(results.samples, weights)
    return {
        "logz": float(results.logz[-1]),
        "logz_err": float(results.logzerr[-1]),
        "niter": int(results.niter),
        "nlive": live,
        "seed": seed,
        "samples": np.asarray(samples, dtype=np.float64),
        "ncall": int(np.sum(np.asarray(getattr(results, "ncall", 0)))),
    }


def run_robustness_suite(
    loglike_fn,
    *,
    ndim: int,
    cfg: InferenceConfig,
) -> list[dict[str, Any]]:
    """Execute the multi-run robustness protocol (see ``ROBUSTNESS_PROTOCOL``)."""
    runs: list[dict[str, Any]] = []
    for k in range(cfg.n_robustness_runs):
        seed = cfg.random_seed + k * cfg.robustness_seed_stride
        nlive = int(math.ceil(cfg.nlive * (cfg.robustness_nlive_scale**k)))
        summary = run_dynesty_nested(
            loglike_fn, ndim=ndim, cfg=cfg, seed=seed, nlive=nlive
        )
        if k > 0:
            summary = {
                "logz": summary["logz"],
                "logz_err": summary["logz_err"],
                "niter": summary["niter"],
                "nlive": summary["nlive"],
                "seed": summary["seed"],
                "ncall": summary["ncall"],
                "sample_median": np.median(summary["samples"], axis=0).tolist(),
            }
        runs.append(summary)
    return runs


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Full stage output before HDF5 serialization."""

    schema_version: int
    likelihood_form: LikelihoodForm
    sensitivity_dimensionality_applied: bool
    eccentricity_hypothesis: str
    circular_implies_wd: bool
    astrometric_sf: float
    followup_sf: float
    bin_edges_msun: NDArray[np.floating]
    fiducial_heights: NDArray[np.floating]
    fiducial_log_likelihood: float
    n_events: int
    zero_count_upper_limits: list[dict[str, float | int]]
    posterior_prior_overlap: dict[str, Any]
    sampler_runs: list[dict[str, Any]] = field(default_factory=list)
    posterior_median_heights: list[float] = field(default_factory=list)
    logz: float | None = None
    logz_err: float | None = None
    robustness_protocol: str = ROBUSTNESS_PROTOCOL
    notes: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def recommendation_payload(self) -> dict[str, Any]:
        """Stable consumer contract for diagnostics / Review."""
        return {
            "schema_version": self.schema_version,
            "stage": "inference",
            "likelihood_form": self.likelihood_form,
            "sensitivity_dimensionality_applied": self.sensitivity_dimensionality_applied,
            "eccentricity_hypothesis": self.eccentricity_hypothesis,
            "circular_implies_wd": self.circular_implies_wd,
            "astrometric_sf": self.astrometric_sf,
            "followup_sf": self.followup_sf,
            "bin_edges_msun": self.bin_edges_msun.tolist(),
            "fiducial_heights": self.fiducial_heights.tolist(),
            "fiducial_log_likelihood": self.fiducial_log_likelihood,
            "n_events": self.n_events,
            "zero_count_upper_limits": self.zero_count_upper_limits,
            "posterior_prior_overlap": self.posterior_prior_overlap,
            "posterior_median_heights": self.posterior_median_heights,
            "logz": self.logz,
            "logz_err": self.logz_err,
            "n_sampler_runs": len(self.sampler_runs),
            "sampler_run_summaries": [
                {
                    "logz": r.get("logz"),
                    "logz_err": r.get("logz_err"),
                    "nlive": r.get("nlive"),
                    "seed": r.get("seed"),
                    "niter": r.get("niter"),
                    "ncall": r.get("ncall"),
                }
                for r in self.sampler_runs
            ],
            "v1_staged_but_connected": True,
            "v2_fully_joint": False,
            "notes": self.notes
            or (
                "v1 staged-but-connected Poisson×SF×dynesty. "
                "External CO MFs are never priors. "
                "Multi-run robustness protocol — not bitwise seeds."
            ),
        }


def run_inference(
    config: PipelineConfig,
    *,
    population_artifact_path: Path | None = None,
    sensitivity_artifact_path: Path | None = None,
    astrometric_sf_artifact_path: Path | None = None,
    followup_sf_artifact_path: Path | None = None,
    population_payload: Mapping[str, Any] | None = None,
    sensitivity_payload: Mapping[str, Any] | None = None,
    events: Sequence[ObservedEvent] | None = None,
    eccentricities: Mapping[int, float] | None = None,
) -> InferenceResult:
    """Assemble Poisson×SF likelihood and run dynesty (or fiducial-only when skipped).

    Simplified population path: when no population artifact / payload is supplied,
    build a fiducial ``run_population_model(config)`` in-memory (empty catalog OK).
    """
    icfg = config.inference
    pcfg = config.population_model

    sa_payload: Mapping[str, Any] | None = sensitivity_payload
    sa_applied = False
    if sa_payload is None and sensitivity_artifact_path is not None:
        sa_payload = read_sensitivity_analysis_artifact(sensitivity_artifact_path)
    if icfg.apply_sensitivity_dimensionality and sa_payload is not None:
        sa_applied = True

    like_form = resolve_likelihood_form(icfg, sa_payload if sa_applied else None)

    pop_payload: Mapping[str, Any] | None = population_payload
    pop_result = None
    if pop_payload is None and population_artifact_path is not None:
        pop_payload = read_population_model_artifact(population_artifact_path)
    if pop_payload is None:
        pop_result = run_population_model(
            config,
            sensitivity_payload=sa_payload if sa_applied else None,
        )
        pop_payload = pop_result.recommendation_payload()

    bin_edges = np.asarray(pop_payload["bin_edges_msun"], dtype=np.float64)
    if "bin_heights" in pop_payload:
        fiducial_heights = np.asarray(pop_payload["bin_heights"], dtype=np.float64)
    else:
        fiducial_heights = default_bin_heights(pcfg)
        if fiducial_heights.size != bin_edges.size - 1:
            bin_edges = build_mass_bin_edges(pcfg)
            fiducial_heights = default_bin_heights(pcfg)

    astro_sf = read_astrometric_sf_scalar(
        astrometric_sf_artifact_path, default=icfg.default_astrometric_sf
    )
    follow_sf = read_followup_sf_scalar(
        followup_sf_artifact_path, default=icfg.default_followup_sf
    )

    if events is not None:
        obs = list(events)
    elif pop_result is not None:
        obs = events_from_population_result(
            pop_result, cfg=icfg, eccentricities=eccentricities
        )
    else:
        obs = collect_observed_events(
            pop_payload, cfg=icfg, eccentricities=eccentricities
        )

    event_masses = np.asarray([e.mass_msun for e in obs], dtype=np.float64)
    event_weights = np.asarray([e.weight for e in obs], dtype=np.float64)
    mass_grid = np.geomspace(bin_edges[0], bin_edges[-1], icfg.n_mass_grid)

    def _loglike(heights: ArrayLike) -> float:
        h = np.asarray(heights, dtype=np.float64)
        if like_form == "unbinned":
            return unbinned_log_likelihood(
                h,
                event_masses=event_masses,
                event_weights=event_weights,
                mass_grid=mass_grid,
                bin_edges=bin_edges,
                population_cfg=pcfg,
                astrometric_sf=astro_sf,
                followup_sf=follow_sf,
            )
        if like_form == "binned":
            return binned_log_likelihood(
                h,
                event_masses=event_masses,
                event_weights=event_weights,
                bin_edges=bin_edges,
                population_cfg=pcfg,
                astrometric_sf=astro_sf,
                followup_sf=follow_sf,
            )
        raise ValueError(f"unknown likelihood_form: {like_form!r}")

    fiducial_ll = _loglike(fiducial_heights)

    edges = bin_edges
    counts = np.zeros(fiducial_heights.size, dtype=np.float64)
    if event_masses.size:
        idx = np.searchsorted(edges, event_masses, side="right") - 1
        valid = (idx >= 0) & (idx < counts.size)
        for i, w in zip(idx[valid], event_weights[valid], strict=True):
            counts[int(i)] += float(w)
    zero_uls = zero_count_poisson_upper_limits(
        counts, confidence=icfg.zero_count_ul_confidence
    )

    sampler_runs: list[dict[str, Any]] = []
    post_overlap: dict[str, Any] = {
        "per_param_width_ratio": [],
        "mean_width_ratio": float("nan"),
        "prior_dominated": False,
        "threshold": icfg.posterior_prior_overlap_threshold,
        "skipped": True,
    }
    post_median: list[float] = []
    logz: float | None = None
    logz_err: float | None = None

    ndim = int(fiducial_heights.size)
    _ = effective_M_Ch_msun(config)
    _ = float(config.classification.M_TOV_msun)

    if not icfg.skip_sampler:
        sampler_runs = run_robustness_suite(_loglike, ndim=ndim, cfg=icfg)
        primary = sampler_runs[0]
        samples = primary["samples"]
        post_median = np.median(samples, axis=0).tolist()
        logz = float(primary["logz"])
        logz_err = float(primary["logz_err"])
        prior_low = np.full(ndim, math.exp(icfg.log_height_min))
        prior_high = np.full(ndim, math.exp(icfg.log_height_max))
        post_overlap = posterior_vs_prior_overlap(
            samples,
            prior_low=prior_low,
            prior_high=prior_high,
            threshold=icfg.posterior_prior_overlap_threshold,
        )
        primary_compact = {k: v for k, v in primary.items() if k != "samples"}
        primary_compact["sample_median"] = post_median
        primary_compact["n_equal_weight_samples"] = int(samples.shape[0])
        max_store = min(200, samples.shape[0])
        primary_compact["samples_preview"] = samples[:max_store].tolist()
        sampler_runs[0] = primary_compact

    return InferenceResult(
        schema_version=SCHEMA_VERSION,
        likelihood_form=like_form,
        sensitivity_dimensionality_applied=sa_applied,
        eccentricity_hypothesis=icfg.eccentricity_hypothesis,
        circular_implies_wd=icfg.circular_implies_wd,
        astrometric_sf=astro_sf,
        followup_sf=follow_sf,
        bin_edges_msun=bin_edges,
        fiducial_heights=fiducial_heights,
        fiducial_log_likelihood=fiducial_ll,
        n_events=len(obs),
        zero_count_upper_limits=zero_uls,
        posterior_prior_overlap=post_overlap,
        sampler_runs=sampler_runs,
        posterior_median_heights=post_median,
        logz=logz,
        logz_err=logz_err,
        config_snapshot=icfg.model_dump(mode="json"),
        notes=(
            "v1 staged-but-connected: fixed companion_nature / gate plug-in weights. "
            "rate = population_model × astrometric_SF × followup_SF. "
            "External CO MFs never priors. See ROBUSTNESS_PROTOCOL."
        ),
    )


def write_inference_artifact(path: Path, result: InferenceResult) -> None:
    """Write stage HDF5 under the run-management artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.recommendation_payload()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "inference"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["likelihood_form"] = result.likelihood_form
        handle.attrs["astrometric_sf"] = result.astrometric_sf
        handle.attrs["followup_sf"] = result.followup_sf
        handle.attrs["n_events"] = result.n_events
        handle.attrs["v1_staged_but_connected"] = True
        if result.logz is not None:
            handle.attrs["logz"] = result.logz
        if result.logz_err is not None:
            handle.attrs["logz_err"] = result.logz_err

        rec = handle.create_group("results")
        rec.create_dataset(
            "payload_json",
            data=np.array(
                json.dumps(payload, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        rec.create_dataset("bin_edges_msun", data=result.bin_edges_msun)
        rec.create_dataset("fiducial_heights", data=result.fiducial_heights)
        if result.posterior_median_heights:
            rec.create_dataset(
                "posterior_median_heights",
                data=np.asarray(result.posterior_median_heights, dtype=np.float64),
            )

        proto = handle.create_group("robustness_protocol")
        proto.create_dataset(
            "text",
            data=np.array(ROBUSTNESS_PROTOCOL, dtype=h5py.string_dtype("utf-8")),
        )

        ul = handle.create_group("zero_count_upper_limits")
        if result.zero_count_upper_limits:
            ul.create_dataset(
                "bin_index",
                data=np.asarray(
                    [r["bin_index"] for r in result.zero_count_upper_limits],
                    dtype=np.int64,
                ),
            )
            ul.create_dataset(
                "mu_upper",
                data=np.asarray(
                    [r["mu_upper"] for r in result.zero_count_upper_limits],
                    dtype=np.float64,
                ),
            )


def read_inference_artifact(path: Path) -> dict[str, Any]:
    """Load the inference payload JSON from a stage HDF5 (consumer contract)."""
    with h5py.File(path, "r") as handle:
        raw = handle["results"]["payload_json"][()]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        return json.loads(text)


def format_inference_report(result: InferenceResult) -> str:
    """Fully legible diagnostic report (exempt from caveman compression)."""
    lines = [
        "=== inference report ===",
        f"likelihood_form: {result.likelihood_form}",
        f"sensitivity_dimensionality_applied: {result.sensitivity_dimensionality_applied}",
        f"eccentricity_hypothesis: {result.eccentricity_hypothesis}",
        f"circular_implies_wd: {result.circular_implies_wd}",
        f"astrometric_sf: {result.astrometric_sf}",
        f"followup_sf: {result.followup_sf}",
        f"n_events: {result.n_events}",
        f"fiducial_log_likelihood: {result.fiducial_log_likelihood:.6g}",
        f"logz: {result.logz}",
        f"logz_err: {result.logz_err}",
        f"n_sampler_runs: {len(result.sampler_runs)}",
        f"posterior_prior_overlap: {result.posterior_prior_overlap}",
        f"n_zero_count_bins: {len(result.zero_count_upper_limits)}",
        "v1 staged-but-connected (fixed plug-in weights); v2 fully-joint not built.",
        "Reproducibility: multi-run robustness protocol — not bitwise seeds.",
        "=== end inference report ===",
    ]
    return "\n".join(lines)


def run_inference_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    population_artifact_path: Path | None = None,
    sensitivity_artifact_path: Path | None = None,
    astrometric_sf_artifact_path: Path | None = None,
    followup_sf_artifact_path: Path | None = None,
) -> RunManifest:
    """Execute ``inference``: Poisson×SF×dynesty, write HDF5, update manifest."""
    spec = STAGE_REGISTRY["inference"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    def _resolved(stage_name: str, explicit: Path | None) -> Path | None:
        if explicit is not None:
            return explicit
        rec = manifest.stages.get(stage_name)
        if rec is not None and rec.artifact_path:
            return Path(rec.artifact_path)
        return None

    pop_path = _resolved("population_model", population_artifact_path)
    sa_path = _resolved("sensitivity_analysis", sensitivity_artifact_path)
    astro_path = _resolved("selection_function_astrometric", astrometric_sf_artifact_path)
    follow_path = _resolved("selection_function_followup", followup_sf_artifact_path)

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_inference(
        config,
        population_artifact_path=pop_path,
        sensitivity_artifact_path=sa_path,
        astrometric_sf_artifact_path=astro_path,
        followup_sf_artifact_path=follow_path,
    )
    write_inference_artifact(artifact, result)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
