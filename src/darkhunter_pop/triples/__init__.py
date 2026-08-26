"""Stage: ``triples`` — unrelated outer-companion identification.

Off by default in v1 (``config.triples.enabled=false``). ``run_management`` plans
``SKIP_REASON`` with detail ``triples.enabled=false``; ``run_triples_stage`` records
skipped status without writing science artifacts. Evidence-channel hooks
(TESS variability + rotation-consistency vs uberMS v sin i) are stubbed so the
stage can be enabled later without restructuring. ``population_model`` forces
P(triple)=0 (ARCHITECTURE.md §4, §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import yaml

from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    StageAction,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import CandidateRecord, RunManifest, StageStatus
from darkhunter_pop.triples.rotation_check import (
    RotationConsistencyEvidence,
    evaluate_rotation_consistency,
)
from darkhunter_pop.triples.tess_variability import (
    TessVariabilityEvidence,
    evaluate_tess_variability,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TripleEvidenceStub:
    """Per-candidate composed stub from both evidence channels (no P(triple))."""

    source_id: int
    tess: TessVariabilityEvidence
    rotation: RotationConsistencyEvidence
    # Explicit: stub never assigns a triple probability.
    p_triple: float | None = None
    notes: str = "stub: P(triple) not evaluated; population_model forces 0 in v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tess": self.tess.as_dict(),
            "rotation": self.rotation.as_dict(),
            "p_triple": self.p_triple,
            "notes": self.notes,
        }


@dataclass
class TriplesStageResult:
    """Stage-level stub result (enable path only; disabled path writes nothing)."""

    schema_version: int
    enabled: bool
    n_candidates: int
    evidence: list[TripleEvidenceStub] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "n_candidates": self.n_candidates,
            "evidence": [e.as_dict() for e in self.evidence],
            "config_snapshot": self.config_snapshot,
        }


def evaluate_candidate_triple_stub(
    candidate: CandidateRecord,
    config: PipelineConfig,
    *,
    uberMS_v_sin_i_kms: float | None = None,
) -> TripleEvidenceStub:
    """Compose TESS + rotation stubs for one candidate; no triple probability."""
    tess = evaluate_tess_variability(candidate, config)
    rotation = evaluate_rotation_consistency(
        candidate, config, uberMS_v_sin_i_kms=uberMS_v_sin_i_kms
    )
    return TripleEvidenceStub(
        source_id=candidate.source_id,
        tess=tess,
        rotation=rotation,
    )


def run_triples_stub(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
) -> TriplesStageResult:
    """Run the enable-path stub over candidates (requires ``triples.enabled``).

    Raises
    ------
    ValueError
        If ``config.triples.enabled`` is false — disabled path must use
        ``run_triples_stage`` / ``plan_stage`` skip, not this function.
    """
    if not config.triples.enabled:
        raise ValueError(
            "run_triples_stub requires triples.enabled=true; "
            "use run_triples_stage for the default off path"
        )
    evidence = [evaluate_candidate_triple_stub(c, config) for c in candidates]
    return TriplesStageResult(
        schema_version=SCHEMA_VERSION,
        enabled=True,
        n_candidates=len(evidence),
        evidence=evidence,
        config_snapshot=config.triples.model_dump(mode="json"),
    )


def write_triples_artifact(path: Path, result: TriplesStageResult) -> None:
    """Write stub HDF5 + YAML sidecar for the enable path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "triples"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["enabled"] = result.enabled
        handle.attrs["n_candidates"] = result.n_candidates
        handle.attrs["stub"] = True
        handle.create_dataset(
            "source_id",
            data=[e.source_id for e in result.evidence],
        )
    sidecar = path.with_suffix(".yaml")
    sidecar.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_triples_artifact(path: Path) -> dict[str, Any]:
    """Load the YAML sidecar written beside the HDF5 stub artifact."""
    sidecar = path.with_suffix(".yaml")
    if not sidecar.is_file():
        raise FileNotFoundError(f"triples sidecar missing: {sidecar}")
    raw = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"triples sidecar must be a mapping: {sidecar}")
    return dict(raw)


def format_triples_stage_report(
    plan_detail: str,
    *,
    status: StageStatus,
    result: TriplesStageResult | None = None,
) -> str:
    """Full-detail stage report (caveman exemption for run-plan / diagnostics)."""
    lines = [
        "=== triples stage ===",
        f"status: {status.value}",
        f"plan_detail: {plan_detail}",
    ]
    if result is None:
        lines.append("science: no-op (stage disabled or skipped)")
    else:
        lines.append(f"enabled: {result.enabled}")
        lines.append(f"n_candidates: {result.n_candidates}")
        lines.append(
            "note: evidence channels are stubbed; P(triple) not evaluated; "
            "population_model forces P(triple)=0 in v1"
        )
    lines.append("=== end triples stage ===")
    return "\n".join(lines)


def run_triples_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
) -> RunManifest:
    """Execute or cleanly skip the ``triples`` stage and update the run manifest.

    Default path (``triples.enabled=false``): ``plan_stage`` returns
    ``SKIP_REASON`` / ``triples.enabled=false``; manifest records ``skipped``
    with no artifact. Enable path runs channel stubs only (no real science).
    """
    spec = STAGE_REGISTRY["triples"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action is StageAction.SKIP_CACHED:
        return manifest

    if plan.action is StageAction.SKIP_REASON:
        manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
        save_run_manifest(manifest, run_path)
        manifest = mark_stage_finished(
            manifest,
            spec,
            status=StageStatus.SKIPPED,
            reason=plan.detail,
            artifact_path=None,
        )
        save_run_manifest(manifest, run_path)
        return manifest

    # Enable path: stub science only.
    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_triples_stub(candidates or (), config)
    write_triples_artifact(artifact, result)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
