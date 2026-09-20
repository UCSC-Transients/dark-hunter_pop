"""End-to-end dry-run harness: fourteen stages, labeled stand-ins, product figure.

Issue #201 / EXECUTION_PLAN.md §7 Wave 0. This module exists to answer one
question — does the plumbing carry a shape from one end of ``STAGE_ORDER`` to the
other — and to make it impossible to mistake the answer for a science result.

Three things it does that a plain ``scripts/run_pipeline.py`` invocation does not:

1. **Declares every substitution.** Wherever a real input does not exist yet, the
   substitution in force is enumerated as a :class:`~darkhunter_pop.schemas.SyntheticStandIn`
   on the run manifest, printed in the run plan before any stage executes, and
   repeated in the report header and the product figure's caption. The run itself
   is tagged ``dry_run: true`` in ``runs/<run_id>.yaml``, so a dry run is
   distinguishable from a science run by grepping the file.
2. **Measures cost.** Wall-clock and peak RSS per stage, feeding the §5.6
   concurrency budget.
3. **Emits the product figure** — total ``dN/dM`` plus BH, NS, WD, other and
   outlier overplotted, to ``docs/PLOTS.md`` standards.

**Nothing this module produces is a result.** It tunes nothing: every placeholder
is whatever ``config/config.yaml`` already says, deliberately including the flat
values, and no stand-in was chosen to make a curve look plausible.
"""

from __future__ import annotations

import math
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_loader import effective_M_Ch_msun, repo_root
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.data_acquisition import gaia_snapshots_dir, run_data_acquisition
from darkhunter_pop.inference import read_inference_artifact
from darkhunter_pop.pipeline import STAGE_RUNNERS, run_pipeline
from darkhunter_pop.plotting import matplotlib_available, plot_dndm_by_class
from darkhunter_pop.population_model import (
    build_mass_bin_edges,
    default_bin_heights,
    evaluate_two_tier_dndm,
    read_population_model_artifact,
)
from darkhunter_pop.run_management import (
    STAGE_ORDER,
    create_run_manifest,
    format_run_plan,
    record_stage_resources,
    run_file_path,
    runs_dir,
    save_run_manifest,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    RunManifest,
    StageStatus,
    SyntheticStandIn,
)

#: The mandatory banner. Carried by the run file, the run plan, the report header
#: and the product-figure caption (EXECUTION_PLAN.md §7 "Labeling requirement").
DRY_RUN_LABEL: Final[str] = (
    "synthetic inputs — plumbing check, not a science result"
)

#: Subdirectory under ``<artifact_root>/<run_id>/`` for dry-run deliverables.
#: Deliberately *not* a stage name: these are product outputs of the harness, not
#: a stage artifact, and no stage's cache key covers them.
DRY_RUN_SUBDIR: Final[str] = "dry_run"

#: Product figure basename (extension appended by the caller's config).
DNDM_FIGURE_STEM: Final[str] = "dndm_by_class"

#: Report basename written next to the figure.
DRY_RUN_REPORT_NAME: Final[str] = "dry_run_report.txt"

#: Caption sidecar, so the caption text is greppable without OCR'ing the PNG.
DNDM_CAPTION_NAME: Final[str] = "dndm_by_class_caption.txt"


# ---------------------------------------------------------------------------
# Per-stage resource measurement
# ---------------------------------------------------------------------------


def _rusage_rss_increase_bytes() -> int:
    """Process-lifetime RSS high-water mark in bytes, from ``getrusage``.

    ``ru_maxrss`` is bytes on Darwin and kibibytes on Linux; both are normalized
    here to bytes. Exact but monotonic — it never decreases, so it cannot give a
    per-stage figure on its own.
    """
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    import sys

    return raw if sys.platform == "darwin" else raw * 1024


@dataclass
class StageResourceMonitor:
    """Track the RSS high-water mark across one stage, without forking.

    Deliberately **fork-free**. An earlier version sampled ``ps`` from a
    background thread every half second; on macOS that is the classic
    fork-in-a-threaded-process hazard, and it deadlocked a real run mid-stage —
    the process sat at 0.2 CPU-seconds per minute with every thread idle. A
    measurement harness must not be able to hang the thing it is measuring, so
    the sampler now reads ``getrusage`` only, which is a syscall on the calling
    thread and cannot block.

    What the two numbers mean, given that ``ru_maxrss`` is a **process-lifetime**
    high-water mark that never decreases:

    * :meth:`stop` returns the *increase* in the high-water mark across the
      stage — what this stage added beyond whatever was already resident. Zero
      means the stage never pushed the process past an earlier peak, which is
      information, not a failure.
    * :meth:`high_water` is the absolute mark at that moment, which is the
      quantity a concurrency budget actually cares about: what the machine had
      to hold.

    Limitations
    -----------
    Because the mark never decreases, a stage that peaks *below* an earlier
    stage's peak reports an increase of zero, so this cannot rank the standalone
    cost of later stages. The absolute figure covers the whole process, not the
    stage's own allocation. Nothing here measures shared pages, swap, or
    children.
    """

    #: Kept for API compatibility with callers that pass a sampling period; the
    #: fork-free implementation has no sampling loop, so it is unused.
    interval_seconds: float = 0.5
    _baseline: int = 0
    _started: bool = False

    @staticmethod
    def high_water() -> int:
        """Process-lifetime RSS high-water mark, in bytes, right now."""
        return _rusage_rss_increase_bytes()

    def start(self) -> None:
        """Record the baseline high-water mark for a stage about to run."""
        self._baseline = self.high_water()
        self._started = True

    def stop(self) -> int | None:
        """Return the high-water *increase* over the stage, in bytes.

        ``None`` when :meth:`start` was never called. ``0`` is a real answer: the
        stage never pushed the process past an earlier peak.
        """
        if not self._started:
            return None
        self._started = False
        return max(0, self.high_water() - self._baseline)


# ---------------------------------------------------------------------------
# Stand-in declarations
# ---------------------------------------------------------------------------


#: A pristine snapshot directory is ``<UTC stamp>Z_<hash>``. Derived caches append
#: a ``+`` suffix (``+enrich``, ``+enrich+mc10000``) and the enrichment job writes
#: its own non-timestamped directory — neither is a snapshot of a parent query, so
#: neither may be replayed as one (CLAUDE.md "Gotchas").
_SNAPSHOT_DIR_PATTERN: Final[str] = r"^\d{8}T\d{6}Z_[0-9a-f]+$"


class AmbiguousSnapshotError(RuntimeError):
    """More than one pristine Gaia snapshot is staged, so replay cannot be inferred.

    Choosing by recency is exactly the mistake this exists to prevent. Two
    snapshots are two *different parent queries*; replaying the wrong one changes
    every downstream count with no visible error, and CLAUDE.md is explicit that
    literature-sample parent queries must run against the documented uncut
    snapshot rather than whichever one happens to be newest.
    """


def list_gaia_snapshots(config: PipelineConfig) -> list[Path]:
    """Every pristine local Gaia snapshot directory, oldest first.

    Only directories whose names match a raw snapshot id are returned, so a
    derived ``+enrich`` / ``+enrich+mc10000`` cache, the ``nss_enrichment``
    working directory, or anything an operator has renamed aside is never
    replayed as if it were the parent query — each would silently mis-count the
    parent (CLAUDE.md "Gotchas"). Ordering is by the UTC timestamp in the
    directory name, never filesystem mtime, matching ARCHITECTURE.md §5.
    """
    import re

    root = gaia_snapshots_dir(config)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and re.match(_SNAPSHOT_DIR_PATTERN, p.name)
        and (p / "meta.yaml").is_file()
    )


def latest_gaia_snapshot_meta(config: PipelineConfig) -> Path | None:
    """The single pristine local Gaia snapshot's ``meta.yaml``, or ``None``.

    Returns a path **only when the choice is unambiguous** — exactly one pristine
    snapshot is staged. With several it raises instead of guessing, and the
    caller passes ``--snapshot`` explicitly.

    Raises
    ------
    AmbiguousSnapshotError
        If more than one pristine snapshot is staged; the message lists them.

    Limitations
    -----------
    Does not check that the snapshot's stored ADQL matches what the current
    config would query. Two snapshots produced by the *same* ADQL months apart
    are still different data, which is why ambiguity is refused rather than
    resolved by any heuristic.
    """
    candidates = list_gaia_snapshots(config)
    if not candidates:
        return None
    if len(candidates) > 1:
        listed = "\n".join(f"  {p.name}" for p in candidates)
        raise AmbiguousSnapshotError(
            f"{len(candidates)} pristine Gaia snapshots are staged under "
            f"{gaia_snapshots_dir(config)}; refusing to guess which parent "
            "query to replay. Pass --snapshot <dir>/meta.yaml explicitly.\n"
            + listed
        )
    return candidates[0] / "meta.yaml"


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def declare_stand_ins(
    config: PipelineConfig,
    *,
    snapshot_meta: Path | None,
) -> list[SyntheticStandIn]:
    """Enumerate every substitution this harness puts in force.

    One entry per substitution, with the config keys that carry the placeholder
    values and a snapshot of those values as actually resolved — so a reader
    checks the numbers rather than trusting the prose.

    Parameters
    ----------
    config:
        The already-merged config the run will use (host profile applied).
    snapshot_meta:
        Gaia snapshot ``meta.yaml`` that ``data_acquisition`` will replay, or
        ``None`` when the stage will query the live archive instead.

    Returns
    -------
    list[SyntheticStandIn]
        Non-empty. ``RunManifest`` refuses a dry run with an empty list.

    Limitations
    -----------
    This is a hand-maintained enumeration, not a derivation: a placeholder added
    to ``config.yaml`` later will not appear here until someone adds it. It is
    the declaration the Wave 0 gate asks for, not a proof of exhaustiveness.
    """
    icfg = config.inference
    cncfg = config.companion_nature
    pcfg = config.population_model
    stand_ins: list[SyntheticStandIn] = []

    if snapshot_meta is not None:
        stand_ins.append(
            SyntheticStandIn(
                name="offline_gaia_snapshot_replay",
                stage="data_acquisition",
                kind="offline_replay",
                replaces="a live gaiadr3.nss_two_body_orbit archive query",
                description=(
                    "data_acquisition is replayed from a Gaia snapshot already on "
                    "this laptop instead of re-querying the archive. The rows are "
                    "real DR3 NSS rows, not synthetic — but they are frozen at the "
                    "snapshot's query date, so this run says nothing about the "
                    "current archive contents, and the funnel counts it reports are "
                    "the snapshot's. Replay also makes the run reproducible, which a "
                    "live query is not."
                ),
                config_keys=["dr3.nss_table", "paths.data_root"],
                values={"snapshot_meta": _relative_to_repo(snapshot_meta)},
            )
        )

    stand_ins.append(
        SyntheticStandIn(
            name="analytic_companion_nature_evidence",
            stage="companion_nature_likelihood",
            kind="analytic_surrogate",
            replaces=(
                "per-star SED / XP model comparison evidence (dark vs WD vs other)"
            ),
            description=(
                "The WD/other/dark decision is driven by closed-form "
                "magnitude-mass relations, not by a real model comparison. The "
                "phot_sed adapter that would supply genuine dynesty BIC and lnZ is "
                "wired (#197), but nothing reaches it yet: the upstream wd model "
                "emits no readable BIC at the pop-facing path (#206) and the channel "
                "needs all three hypotheses, so every candidate falls back to the "
                "analytic relations and is labelled analytic_fallback. Consequences: "
                "the five-class responsibilities are a smooth function of absolute "
                "magnitude rather than a fit, they carry no covariance of any kind "
                "because the surrogate has no uncertainty model at all, and WD "
                "contamination is entirely unconstrained by this run."
            ),
            config_keys=[
                "companion_nature.wd_mg_zero_point",
                "companion_nature.wd_mg_mass_slope",
                "companion_nature.other_mg_zero_point",
                "companion_nature.other_mg_mass_slope",
                "companion_nature.primary_mg_zero_point",
                "companion_nature.primary_mg_mass_slope",
                "companion_nature.dark_to_bh_fraction",
                "companion_nature.delta_bic_threshold",
            ],
            values={
                "wd_mg_zero_point": cncfg.wd_mg_zero_point,
                "wd_mg_mass_slope": cncfg.wd_mg_mass_slope,
                "other_mg_zero_point": cncfg.other_mg_zero_point,
                "other_mg_mass_slope": cncfg.other_mg_mass_slope,
                "dark_to_bh_fraction": cncfg.dark_to_bh_fraction,
                "delta_bic_threshold": cncfg.delta_bic_threshold,
            },
        )
    )

    stand_ins.append(
        SyntheticStandIn(
            name="per_sample_selection_weights",
            stage="inference",
            kind="config_placeholder",
            replaces=(
                "forward-modeled per-sample selection functions SF_sample_s(theta)"
            ),
            description=(
                "Each published sample's own cut chain is itself a selection "
                "function and must be forward-modeled (CLAUDE.md, Phase 8). None of "
                "them reproduces its published N yet, so none may be forward-modeled "
                "here. The multi-sample inclusion operator instead runs on flat "
                "catalog weights and one flat spurious rate shared by every sample. "
                "Flat is the crudest available stand-in and that is the point: a "
                "weight chosen to make the output look plausible would be worse than "
                "an obviously crude one. No inference conclusion survives this."
            ),
            config_keys=[
                "inference.multi_sample.default_catalog_sf",
                "inference.multi_sample.default_p_spurious",
                "inference.default_astrometric_sf",
                "inference.default_followup_sf",
            ],
            values={
                "default_catalog_sf": dict(icfg.multi_sample.default_catalog_sf),
                "default_p_spurious": icfg.multi_sample.default_p_spurious,
                "default_astrometric_sf": icfg.default_astrometric_sf,
                "default_followup_sf": icfg.default_followup_sf,
            },
        )
    )

    enabled_modes = {
        entry.name: entry.mode.value
        for entry in config.sample_selection.samples
        if entry.enabled
    }
    forward_model_names = sorted(
        name for name, mode in enabled_modes.items() if mode == "forward_model"
    )
    stand_ins.append(
        SyntheticStandIn(
            name="no_reproducing_literature_sample",
            stage="sample_selection",
            kind="disabled_path",
            replaces="a reproducing literature sample feeding inference",
            description=(
                "No literature sample reproduces its published N yet "
                "(CLAUDE.md Status; that is Wave A's work), so the sample-selection "
                "layer contributes nothing trustworthy to this run either way. "
                "Samples in reproduction mode are regression tests against a "
                "published number and are never inference inputs by construction. "
                + (
                    "This run's config additionally leaves "
                    f"{', '.join(forward_model_names)} enabled in forward_model "
                    "mode, inherited from config.yaml as authored — this harness "
                    "neither enabled it nor changed it. Its contribution is still "
                    "meaningless here, because the per-sample weight it would carry "
                    "is the flat placeholder described under "
                    "per_sample_selection_weights, not a forward-modeled selection "
                    "function."
                    if forward_model_names
                    else "Every enabled sample is in reproduction mode."
                )
            ),
            config_keys=["sample_selection.samples"],
            values={
                "modes": enabled_modes,
                "forward_model_mode_samples": forward_model_names,
            },
        )
    )

    stand_ins.append(
        SyntheticStandIn(
            name="ci_scale_dynesty",
            stage="inference",
            kind="ci_scale_sampler",
            replaces="a production nested-sampling run",
            description=(
                "dynesty stays at the CI smoke settings. The posterior is not "
                "converged, the evidence is not usable, and the multi-run robustness "
                "protocol that validates a real dynesty result "
                "(ARCHITECTURE.md §4 inference) has not been satisfied. The cluster "
                "recipe lives in config/fragments/inference.yaml and is deliberately "
                "not used: raising these numbers here would buy a slower run with no "
                "more meaning, since the likelihood it samples is built on the "
                "placeholders above."
            ),
            config_keys=[
                "inference.nlive",
                "inference.dlogz",
                "inference.maxcall",
                "inference.n_robustness_runs",
            ],
            values={
                "nlive": icfg.nlive,
                "dlogz": icfg.dlogz,
                "maxcall": icfg.maxcall,
                "n_robustness_runs": icfg.n_robustness_runs,
            },
        )
    )

    stand_ins.append(
        SyntheticStandIn(
            name="fiducial_mass_function_heights",
            stage="population_model",
            kind="config_placeholder",
            replaces="mass-function bin heights informed by anything",
            description=(
                "The non-parametric mass function starts from fiducial bin heights "
                "on bin edges fixed in config before any real count was looked at "
                "(which is the correct guardrail, not a stand-in). What makes it a "
                "stand-in is that the heights themselves are a configured fiducial: "
                "the shape of the product figure is set by that fiducial and by the "
                "class truncations, not by the data. Read the figure as a check that "
                "a shape propagates, and read nothing else from it."
            ),
            config_keys=[
                "population_model.mass_function_model",
                "population_model.mass_bin_edges_msun",
                "population_model.default_bin_height",
            ],
            values={
                "mass_function_model": pcfg.mass_function_model,
                "n_bins": len(build_mass_bin_edges(pcfg)) - 1,
            },
        )
    )
    return stand_ins


def dry_run_seeds(config: PipelineConfig) -> dict[str, Any]:
    """Seeds in force for this run, for the manifest's ``random_seeds`` block.

    Accounting only: ``dynesty`` is validated by multi-run posterior agreement,
    never by bitwise seed replay (ARCHITECTURE.md §4 ``inference``). Recorded so
    a reader can tell which seeds produced which artifact, not so the run can be
    replayed bit-for-bit.
    """
    return {
        "inference.random_seed": config.inference.random_seed,
        "inference.robustness_seed_stride": config.inference.robustness_seed_stride,
        "population_model.random_seed": config.population_model.random_seed,
        "sensitivity_analysis.random_seed": config.sensitivity_analysis.random_seed,
        "mc_mass_function.random_seed": config.mc_mass_function.random_seed,
        "diagnostics.sbc.random_seed": config.diagnostics.sbc.random_seed,
        "selection_function_astrometric.mock_population.random_seed": (
            config.selection_function_astrometric.mock_population.random_seed
        ),
    }


# ---------------------------------------------------------------------------
# Run construction and execution
# ---------------------------------------------------------------------------


def build_dry_run_manifest(
    config: PipelineConfig,
    *,
    snapshot_meta: Path | None,
    runs: Path | None = None,
) -> tuple[RunManifest, Path]:
    """Create and persist a fresh dry-run manifest, labeled at birth.

    Returns ``(manifest, run_path)``. The manifest is written before any stage
    executes, so the label and the stand-in list exist on disk from the moment
    the run does — there is no window in which an unlabeled run file exists.

    Parameters
    ----------
    config:
        Merged config for the run (host profile already applied).
    snapshot_meta:
        Snapshot ``meta.yaml`` that ``data_acquisition`` will replay, or ``None``
        for a live archive query.
    runs:
        Directory for run files; defaults to the repository's ``runs/``.
    """
    root = runs if runs is not None else runs_dir()
    manifest = create_run_manifest(
        config,
        random_seeds=dry_run_seeds(config),
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=declare_stand_ins(
            config, snapshot_meta=snapshot_meta
        ),
    )
    path = run_file_path(manifest.run_id, root)
    save_run_manifest(manifest, path)
    return manifest, path


@dataclass
class StageCost:
    """Measured cost of one executed stage."""

    stage: str
    wall_clock_seconds: float
    rss_increase_bytes: int | None
    rss_high_water_bytes: int | None


def instrumented_runners(
    *,
    snapshot_meta: Path | None,
    costs: list[StageCost],
    monitor_interval_seconds: float,
    base: Mapping[str, Callable[..., RunManifest]] | None = None,
) -> dict[str, Callable[..., RunManifest]]:
    """Wrap every stage runner with timing, RSS sampling, and manifest annotation.

    The wrapper measures around the runner call, then writes the measurements
    onto the returned manifest's stage record and persists the run file — so the
    numbers survive a later crash, and no stage runner needs to know it is being
    measured.

    Parameters
    ----------
    snapshot_meta:
        When given, ``data_acquisition`` is bound to replay this snapshot rather
        than query the archive.
    costs:
        Appended to, one entry per stage that actually executed. Cached and
        skipped stages never reach a runner, so they never appear.
    monitor_interval_seconds:
        RSS sampling period handed to :class:`StageResourceMonitor`.
    base:
        Runner mapping to wrap; defaults to ``pipeline.STAGE_RUNNERS``.

    Limitations
    -----------
    A runner that returns early because the stage was cached still gets timed;
    its wall-clock is then the cache check, not any science work. The harness
    filters those out by checking whether the stage's status actually advanced.
    """
    source = dict(base if base is not None else STAGE_RUNNERS)
    if snapshot_meta is not None and "data_acquisition" in source:
        source["data_acquisition"] = partial(
            run_data_acquisition, snapshot_meta_path=snapshot_meta
        )

    def _wrap(
        stage: str, runner: Callable[..., RunManifest]
    ) -> Callable[..., RunManifest]:
        def _instrumented(
            manifest: RunManifest,
            config: PipelineConfig,
            *,
            run_path: Path,
            force_rerun: bool = False,
            **kwargs: Any,
        ) -> RunManifest:
            monitor = StageResourceMonitor(
                interval_seconds=monitor_interval_seconds
            )
            monitor.start()
            start = time.monotonic()
            try:
                updated = runner(
                    manifest,
                    config,
                    run_path=run_path,
                    force_rerun=force_rerun,
                    **kwargs,
                )
            finally:
                elapsed = time.monotonic() - start
                peak = monitor.stop()
            record = updated.stages.get(stage)
            if record is None:
                return updated
            cumulative = _rusage_rss_increase_bytes()
            updated = record_stage_resources(
                updated,
                stage,
                wall_clock_seconds=elapsed,
                rss_increase_bytes=peak,
                rss_high_water_bytes=cumulative,
            )
            save_run_manifest(updated, run_path)
            costs.append(
                StageCost(
                    stage=stage,
                    wall_clock_seconds=elapsed,
                    rss_increase_bytes=peak,
                    rss_high_water_bytes=cumulative,
                )
            )
            print(
                f"[cost] {stage}: wall_clock={elapsed:.2f}s "
                f"rss_increase={_human_bytes(peak)} "
                f"rss_high_water={_human_bytes(cumulative)}",
                flush=True,
            )
            return updated

        return _instrumented

    return {stage: _wrap(stage, runner) for stage, runner in source.items()}


def _human_bytes(value: int | None) -> str:
    """Format bytes as GiB/MiB for report output, or ``n/a`` for ``None``."""
    if value is None:
        return "n/a"
    gib = value / float(1 << 30)
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{value / float(1 << 20):.1f} MiB"


# ---------------------------------------------------------------------------
# Product figure
# ---------------------------------------------------------------------------


def dry_run_dirs(config: PipelineConfig, run_id: str) -> tuple[Path, Path]:
    """``(figures_dir, reports_dir)`` for this run's dry-run deliverables."""
    root = Path(config.paths.artifact_root)
    if not root.is_absolute():
        root = repo_root() / root
    base = root / run_id / DRY_RUN_SUBDIR
    return (
        base / config.diagnostics.figures_subdir,
        base / config.diagnostics.reports_subdir,
    )


def caption_text(manifest: RunManifest) -> str:
    """Figure caption: the mandatory banner plus every stand-in by name.

    Full detail on purpose — captions and diagnostic reports are exempt from
    caveman compression (project skill §11). The caption names each stand-in so
    the figure carries its own provenance even when it escapes the run directory.
    """
    lines = [
        f"{DRY_RUN_LABEL.upper()}.",
        (
            f"Run {manifest.run_id} ({manifest.host_profile or 'no host profile'}), "
            f"config_checksum {manifest.config_checksum[:12]}. Total is the tier-1 "
            "raw compact-object rate with no M_TOV assumption; the class curves are "
            "tier-2 species-classified with M_TOV marginalized for NS. Axis values "
            "are NOT a measurement of the mass function and must not be quoted."
        ),
        "Stand-ins in force for this run:",
    ]
    for stand_in in manifest.synthetic_stand_ins:
        lines.append(
            f"  - {stand_in.name} ({stand_in.kind}, {stand_in.stage}): "
            f"replaces {stand_in.replaces}."
        )
    return "\n".join(lines)


@dataclass
class DnDmFigureInputs:
    """Everything the product figure needs, resolved from stage artifacts."""

    mass_grid_msun: NDArray[np.floating]
    total_dndm: NDArray[np.floating]
    class_dndm: dict[str, NDArray[np.floating]]
    bin_edges_msun: NDArray[np.floating]
    heights: NDArray[np.floating]
    heights_source: str
    class_fractions: dict[str, float]
    m_ch_msun: float
    m_tov_msun: float


def resolve_dndm_inputs(
    config: PipelineConfig,
    *,
    population_artifact: Path,
    inference_artifact: Path | None,
) -> DnDmFigureInputs:
    """Re-evaluate the two-tier ``dN/dM`` on a fine grid for the product figure.

    The curves are rebuilt rather than read straight out of the
    ``population_model`` artifact for one reason: rebuilding lets the figure use
    ``inference``'s posterior median bin heights, so the panel actually carries
    the whole chain instead of re-displaying the fiducial the chain started from.
    Everything else (bin edges, class fractions, truncation constants) comes from
    the stage artifacts.

    Parameters
    ----------
    config:
        Merged run config; supplies the grid size and truncation settings.
    population_artifact:
        ``population_model`` stage HDF5.
    inference_artifact:
        ``inference`` stage HDF5, or ``None`` to fall back to the fiducial
        heights. Falling back is recorded in ``heights_source``.

    Raises
    ------
    ValueError
        If the population artifact carries no usable bin edges.

    Limitations
    -----------
    The posterior medians are marginal per-bin medians, so the drawn curve is not
    a sample from the posterior and carries no credible band. With CI-scale
    dynesty settings the medians are not converged either — which is why nothing
    on this figure may be quoted.
    """
    pop = read_population_model_artifact(population_artifact)
    edges = np.asarray(pop.get("bin_edges_msun") or (), dtype=np.float64)
    if edges.size < 2:
        raise ValueError(
            f"population artifact {population_artifact} has no usable bin edges"
        )

    heights = np.asarray(pop.get("bin_heights") or (), dtype=np.float64)
    heights_source = "population_model fiducial bin heights"
    if inference_artifact is not None and inference_artifact.is_file():
        inf = read_inference_artifact(inference_artifact)
        medians = np.asarray(
            inf.get("posterior_median_heights") or (), dtype=np.float64
        )
        if medians.size == edges.size - 1 and np.all(np.isfinite(medians)):
            heights = medians
            heights_source = "inference posterior median bin heights (CI-scale)"
    if heights.size != edges.size - 1:
        heights = default_bin_heights(config.population_model)
        heights_source = "population_model schema default bin heights"

    rows = pop.get("system_weight_rows") or []
    fractions: dict[str, float] = {}
    for key in COMPANION_NATURE_WEIGHT_KEYS:
        values = [
            float(row["responsibilities"][key])
            for row in rows
            if isinstance(row.get("responsibilities"), Mapping)
            and key in row["responsibilities"]
        ]
        fractions[key] = (
            float(np.mean(values))
            if values
            else 1.0 / len(COMPANION_NATURE_WEIGHT_KEYS)
        )

    n_grid = int(config.inference.n_mass_grid)
    grid = np.geomspace(float(edges[0]), float(edges[-1]), max(n_grid, 8))

    m_ch = float(pop.get("m_ch_msun") or effective_M_Ch_msun(config))
    m_tov = float(pop.get("m_tov_msun") or config.classification.M_TOV_msun)

    two_tier = evaluate_two_tier_dndm(
        grid,
        bin_edges=edges,
        heights=heights,
        cfg=config.population_model,
        m_ch_msun=m_ch,
        m_tov_mean_msun=m_tov,
        class_fractions=fractions,
    )
    return DnDmFigureInputs(
        mass_grid_msun=two_tier.mass_grid_msun,
        total_dndm=two_tier.raw_total_co_dndm,
        class_dndm=dict(two_tier.classified_dndm),
        bin_edges_msun=edges,
        heights=heights,
        heights_source=heights_source,
        class_fractions=fractions,
        m_ch_msun=m_ch,
        m_tov_msun=m_tov,
    )


def emit_dndm_by_class(
    config: PipelineConfig,
    manifest: RunManifest,
    inputs: DnDmFigureInputs,
) -> tuple[Path | None, Path]:
    """Write the product figure and its caption sidecar.

    Returns ``(figure_path_or_None, caption_path)``. The figure is ``None`` only
    when matplotlib is unavailable or every curve was empty; the caption is
    always written, so the run's provenance text exists either way.
    """
    figures, reports = dry_run_dirs(config, manifest.run_id)
    caption = caption_text(manifest)
    reports.mkdir(parents=True, exist_ok=True)
    caption_path = reports / DNDM_CAPTION_NAME
    caption_path.write_text(caption + "\n", encoding="utf-8")

    if not matplotlib_available():
        return None, caption_path

    figure_path = figures / f"{DNDM_FIGURE_STEM}.png"
    written = plot_dndm_by_class(
        inputs.mass_grid_msun,
        inputs.total_dndm,
        inputs.class_dndm,
        figure_path,
        dpi=int(config.diagnostics.figure_dpi),
        class_order=list(config.population_model.population_classes),
        title=DRY_RUN_LABEL,
        caption=caption,
        vlines={
            r"$M_{\rm Ch}$": inputs.m_ch_msun,
            r"$M_{\rm TOV}$": inputs.m_tov_msun,
        },
        style=config.plotting,
    )
    return written, caption_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _installed_gaiamock_triple() -> str:
    """The overlay triple actually on disk, for the report's provenance block.

    Reported alongside the manifest's (config-sourced) triple because
    ``config/config.yaml`` currently leaves ``gaiamock.mod_sha256`` and
    ``gaiamock.git_commit`` null, so the manifest records only the release tag.
    Read-only: this never writes to the manifest and never overrides config.

    Returns
    -------
    str
        ``release=... sha256=... git_commit=...``, or a short note when the
        overlay is not installed / not readable.
    """
    try:
        from darkhunter_pop.gaiamock_vendor import read_versions

        versions = read_versions()
    except Exception as exc:  # noqa: BLE001 — provenance line, never fatal
        return f"(unavailable: {type(exc).__name__}: {exc})"
    return (
        f"release={versions.gaiamock_mod_release} "
        f"sha256={versions.gaiamock_mod_sha256} "
        f"git_commit={versions.gaiamock_git_commit}"
    )


def format_dry_run_report(
    manifest: RunManifest,
    *,
    plan_text: str,
    inputs: DnDmFigureInputs | None,
    figure_path: Path | None,
    total_wall_clock_seconds: float,
) -> str:
    """Full-detail dry-run report, banner first (exempt from caveman compression).

    The header carries the mandatory label and names every stand-in before any
    number appears, so a reader who stops after the header has still been told
    that nothing below it is a result.
    """
    bar = "=" * 78
    lines = [
        bar,
        f"  {DRY_RUN_LABEL.upper()}",
        bar,
        "",
        "This report is the Wave 0 end-to-end plumbing check (issue #201). Every",
        "number in it describes the machinery, not the sky. No value here is a",
        "measurement of the compact-object mass function, and none of it may be",
        "quoted in a figure, caption, talk or paper.",
        "",
        f"run_id:               {manifest.run_id}",
        f"created_at:           {manifest.created_at.isoformat()}",
        f"dry_run:              {manifest.dry_run}",
        f"config_checksum:      {manifest.config_checksum}",
        f"active_dr_mode:       {manifest.active_dr_mode.value}",
        f"host_profile:         {manifest.host_profile or 'none'}",
        f"artifact_root:        {manifest.artifact_root}",
        (
            "gaiamock triple (config): "
            f"release={manifest.gaiamock_mod_release} "
            f"sha256={manifest.gaiamock_mod_sha256} "
            f"git_commit={manifest.gaiamock_git_commit}"
        ),
        f"gaiamock triple (installed): {_installed_gaiamock_triple()}",
        f"random_seeds:         {manifest.random_seeds}",
        "",
        f"--- synthetic stand-ins in force ({len(manifest.synthetic_stand_ins)}) ---",
    ]
    for stand_in in manifest.synthetic_stand_ins:
        lines.append("")
        lines.append(f"* {stand_in.one_line()}")
        lines.append(f"  {stand_in.description}")
        if stand_in.config_keys:
            lines.append("  config keys:")
            for key in stand_in.config_keys:
                lines.append(f"    - {key}")
        if stand_in.values:
            lines.append(f"  resolved values: {stand_in.values}")

    lines.extend(["", "--- stage outcomes ---", ""])
    header = (
        f"{'stage':<34} {'status':<10} {'wall_clock':>11} "
        f"{'rss_added':>11} {'rss_high_water':>15}  reason"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name in STAGE_ORDER:
        record = manifest.stages.get(name)
        if record is None:
            lines.append(f"{name:<34} {'(no record)':<10}")
            continue
        wall = (
            f"{record.wall_clock_seconds:.2f}s"
            if record.wall_clock_seconds is not None
            else "-"
        )
        lines.append(
            f"{name:<34} {record.status.value:<10} {wall:>11} "
            f"{_human_bytes(record.rss_increase_bytes):>11} "
            f"{_human_bytes(record.rss_high_water_bytes):>15}  "
            f"{record.reason or ''}"
        )
    lines.append("")
    lines.append(f"total harness wall clock: {total_wall_clock_seconds:.1f}s")
    lines.append(
        "Both RSS figures come from getrusage's process-lifetime high-water "
        "mark, which never decreases. rss_high_water is that mark at the "
        "stage's end and is what a concurrency budget must hold; rss_added is "
        "the increase across the stage. A stage peaking below an earlier "
        "stage's peak therefore reports 0 added, which is information rather "
        "than a failure. Both feed EXECUTION_PLAN.md §5.6."
    )

    lines.extend(["", "--- product figure ---", ""])
    if inputs is None:
        lines.append(
            "dN/dM-by-class figure NOT built: the population_model artifact was "
            "unavailable."
        )
    else:
        lines.append(f"figure: {figure_path if figure_path else '(not written)'}")
        lines.append(f"bin heights source: {inputs.heights_source}")
        lines.append(
            f"n_bins: {inputs.bin_edges_msun.size - 1}  "
            f"mass range: {inputs.bin_edges_msun[0]:g}"
            f"-{inputs.bin_edges_msun[-1]:g} Msun  "
            f"grid points: {inputs.mass_grid_msun.size}"
        )
        lines.append(
            f"M_Ch: {inputs.m_ch_msun:g} Msun (WD hard truncation)  "
            f"M_TOV: {inputs.m_tov_msun:g} Msun (NS soft truncation)"
        )
        lines.append(
            "class fractions (mean companion-nature responsibilities, from the "
            "analytic surrogate — NOT measured class fractions):"
        )
        for key in COMPANION_NATURE_WEIGHT_KEYS:
            lines.append(f"    {key:<8} {inputs.class_fractions.get(key, 0.0):.6g}")
        drawn = [
            name
            for name, arr in inputs.class_dndm.items()
            if np.any(np.isfinite(arr) & (np.asarray(arr) > 0.0))
        ]
        missing = sorted(set(inputs.class_dndm) - set(drawn))
        lines.append(f"classes with a drawable curve: {sorted(drawn)}")
        if missing:
            lines.append(
                f"classes with no positive rate anywhere on the grid: {missing}"
            )

    lines.extend(["", "--- run plan as printed before execution ---", "", plan_text])
    lines.extend(["", bar, f"  {DRY_RUN_LABEL.upper()}", bar])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DryRunResult:
    """Everything the harness produced, for the CLI and for tests."""

    manifest: RunManifest
    run_path: Path
    plan_text: str
    report_path: Path
    report_text: str
    figure_path: Path | None
    caption_path: Path
    costs: list[StageCost]
    total_wall_clock_seconds: float

    def unfinished_stages(self) -> list[str]:
        """Stages not in a terminal state — must be empty for the gate to pass."""
        terminal = {
            StageStatus.COMPLETED,
            StageStatus.SKIPPED,
            StageStatus.CACHED,
        }
        return [
            name
            for name in STAGE_ORDER
            if (record := self.manifest.stages.get(name)) is None
            or record.status not in terminal
        ]

    def skips_without_reason(self) -> list[str]:
        """Skipped stages carrying no stated reason — must be empty for the gate."""
        return [
            name
            for name in STAGE_ORDER
            if (record := self.manifest.stages.get(name)) is not None
            and record.status is StageStatus.SKIPPED
            and not (record.reason or "").strip()
        ]


def run_dry_run(
    config: PipelineConfig,
    *,
    run_file: Path | None = None,
    runs: Path | None = None,
    snapshot_meta: Path | None = None,
    use_local_snapshot: bool = True,
    force_rerun_stages: Sequence[str] = (),
    monitor_interval_seconds: float = 0.5,
) -> DryRunResult:
    """Execute all fourteen stages as a labeled dry run, then emit the deliverables.

    Parameters
    ----------
    config:
        Merged config for the run. The caller applies the host profile (the CLI
        passes ``--host-profile laptop``).
    run_file:
        Existing dry-run manifest to **resume**. Omit to create a new one. A
        resumed run must already be tagged ``dry_run``; resuming a science run as
        a dry run would silently relabel it, so that is refused.
    runs:
        Directory for run files; defaults to the repository's ``runs/``.
    snapshot_meta:
        Explicit Gaia snapshot ``meta.yaml`` for ``data_acquisition`` to replay.
    use_local_snapshot:
        When true (default) and ``snapshot_meta`` is omitted, the newest local
        snapshot is discovered and replayed. Set false to let
        ``data_acquisition`` query the live archive instead.
    force_rerun_stages:
        Forwarded to ``run_pipeline``. Note that force-re-running a *completed*
        stage always creates a new run file (ARCHITECTURE.md §5), which for a dry
        run inherits the label and the stand-in list.
    monitor_interval_seconds:
        RSS sampling period.

    Raises
    ------
    ValueError
        If ``run_file`` names a manifest that is not tagged as a dry run.

    Limitations
    -----------
    The harness does not verify that the pipeline used *only* the declared
    stand-ins, and it does not fail when a stage ends up skipped — it reports
    what happened and leaves the gate decision to the caller
    (:meth:`DryRunResult.unfinished_stages`).
    """
    resolved_snapshot = snapshot_meta
    if resolved_snapshot is None and use_local_snapshot:
        resolved_snapshot = latest_gaia_snapshot_meta(config)

    if run_file is not None:
        from darkhunter_pop.run_management import load_run_manifest

        existing = load_run_manifest(run_file)
        if not existing.dry_run:
            raise ValueError(
                f"run file {run_file} is not tagged dry_run; refusing to relabel "
                "an existing run. Create a new dry run instead."
            )
        run_path = run_file
    else:
        _manifest, run_path = build_dry_run_manifest(
            config, snapshot_meta=resolved_snapshot, runs=runs
        )

    costs: list[StageCost] = []
    runners = instrumented_runners(
        snapshot_meta=resolved_snapshot,
        costs=costs,
        monitor_interval_seconds=monitor_interval_seconds,
    )

    start = time.monotonic()
    manifest, run_path, plan_text = run_pipeline(
        config=config,
        run_file=run_path,
        runs=runs,
        runners=runners,
        force_rerun_stages=force_rerun_stages,
    )
    total = time.monotonic() - start

    pop_record = manifest.stages.get("population_model")
    inputs: DnDmFigureInputs | None = None
    figure_path: Path | None = None
    caption_path = dry_run_dirs(config, manifest.run_id)[1] / DNDM_CAPTION_NAME
    if pop_record is not None and pop_record.artifact_path:
        pop_path = Path(pop_record.artifact_path)
        if pop_path.is_file():
            inf_record = manifest.stages.get("inference")
            inf_path = (
                Path(inf_record.artifact_path)
                if inf_record is not None and inf_record.artifact_path
                else None
            )
            inputs = resolve_dndm_inputs(
                config,
                population_artifact=pop_path,
                inference_artifact=inf_path,
            )
            figure_path, caption_path = emit_dndm_by_class(config, manifest, inputs)

    report_text = format_dry_run_report(
        manifest,
        plan_text=plan_text,
        inputs=inputs,
        figure_path=figure_path,
        total_wall_clock_seconds=total,
    )
    reports = dry_run_dirs(config, manifest.run_id)[1]
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / DRY_RUN_REPORT_NAME
    report_path.write_text(report_text + "\n", encoding="utf-8")

    return DryRunResult(
        manifest=manifest,
        run_path=run_path,
        plan_text=plan_text,
        report_path=report_path,
        report_text=report_text,
        figure_path=figure_path,
        caption_path=caption_path,
        costs=costs,
        total_wall_clock_seconds=total,
    )


__all__ = [
    "DNDM_CAPTION_NAME",
    "DNDM_FIGURE_STEM",
    "DRY_RUN_LABEL",
    "DRY_RUN_REPORT_NAME",
    "DRY_RUN_SUBDIR",
    "DnDmFigureInputs",
    "DryRunResult",
    "StageCost",
    "StageResourceMonitor",
    "build_dry_run_manifest",
    "caption_text",
    "declare_stand_ins",
    "dry_run_dirs",
    "dry_run_seeds",
    "emit_dndm_by_class",
    "format_dry_run_report",
    "instrumented_runners",
    "latest_gaia_snapshot_meta",
    "resolve_dndm_inputs",
    "run_dry_run",
]
