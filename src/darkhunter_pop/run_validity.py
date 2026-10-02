"""Stage honesty flags: point-of-use stand-ins, upstream gate status, science validity.

Issues #352 (consumer side of the validation gates) and #354 (stand-in inventory).

Two problems this module exists to close:

1. **Stand-ins were declared by hand.** ``dry_run.declare_stand_ins`` is a list someone
   maintains; a stage that takes a synthetic branch nobody listed goes undeclared, and a
   placeholder's value is reported as its config default rather than what the stage
   actually used. Here a stage that takes a synthetic or placeholder path **registers it
   in its own HDF5 artifact at the point of use** (:func:`write_stand_ins`). The run's
   inventory is then *collected* from the artifacts that actually ran
   (:func:`collect_stand_ins`), so it reflects resolved values and cannot omit a branch
   that registered itself.
2. **Validation gates failed silently.** ``selection_function_astrometric`` records
   ``validation_gate_passed`` and nothing read it; ``inference`` consumed
   ``detection_fraction`` regardless. :func:`astrometric_gate_status` and
   :func:`followup_gate_status` read each upstream gate, and
   :func:`assess_science_validity` combines them with the collected stand-ins under the
   configured ``inference.upstream_gate_policy``. **No outcome of this module ever claims a
   science-valid result when a gate failed, was not run, or cannot be read.**

Layering: infrastructure only. Imports ``schemas`` and ``config_schema``; never a stage
module, so every stage may depend on it without widening its ``source_hash`` closure.
Reports built from these records are exempt from caveman compression.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import h5py

from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import RunManifest, StageStatus, SyntheticStandIn

#: HDF5 root attribute carrying a stage's point-of-use stand-ins as a JSON list.
STAND_INS_ATTR: Final[str] = "stand_ins_json"

#: HDF5 root attribute on the follow-up artifact: ``passed`` / ``failed`` /
#: ``not_calibrated`` (no real follow-up catalog — synthetic vs synthetic, #351).
FOLLOWUP_CALIBRATION_STATUS_ATTR: Final[str] = "calibration_status"

#: Stages whose code registers its point-of-use stand-ins (always writing
#: :data:`STAND_INS_ATTR`, possibly as an empty list). An artifact of one of these
#: stages *without* the attribute predates registration and is reported as
#: unregistered — never treated as clean. Stages outside this set take no
#: synthetic branch that #354 identified, so they are not expected to register.
STAND_IN_REGISTERING_STAGES: Final[frozenset[str]] = frozenset(
    {
        "companion_nature_likelihood",
        "selection_function_astrometric",
        "selection_function_followup",
        "sensitivity_analysis",
        "inference",
        "diagnostics",
    }
)

GateStatusValue = Literal["passed", "failed", "not_calibrated", "not_run", "unreadable"]
UpstreamGatePolicy = Literal["refuse", "mark_not_science_valid"]


class UpstreamGateFailedError(RuntimeError):
    """Raised by a consumer stage under ``upstream_gate_policy: refuse``."""


# ---------------------------------------------------------------------------
# Point-of-use stand-in registration
# ---------------------------------------------------------------------------


def merge_stand_ins(*groups: Iterable[SyntheticStandIn]) -> list[SyntheticStandIn]:
    """Concatenate stand-in lists, de-duplicated by ``name`` (later entries win).

    Order is first appearance, so a pre-declared entry keeps its slot when a stage's
    point-of-use registration later replaces its values.
    """
    by_name: dict[str, SyntheticStandIn] = {}
    for group in groups:
        for item in group:
            by_name[item.name] = item
    return list(by_name.values())


def _stand_ins_to_json(stand_ins: Sequence[SyntheticStandIn]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in stand_ins], sort_keys=True)


def _stand_ins_from_json(raw: Any) -> list[SyntheticStandIn]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(str(raw))
    return [SyntheticStandIn(**item) for item in data]


def write_stand_ins_to_handle(
    handle: h5py.File | h5py.Group,
    stand_ins: Sequence[SyntheticStandIn],
) -> None:
    """Merge ``stand_ins`` into the handle's :data:`STAND_INS_ATTR` (by name).

    Always writes the attribute, even for an empty list: an empty list on a stage
    artifact means "this stage registered no stand-in", which is different from a
    missing attribute ("this artifact predates registration / never checked").
    """
    existing: list[SyntheticStandIn] = []
    if STAND_INS_ATTR in handle.attrs:
        existing = _stand_ins_from_json(handle.attrs[STAND_INS_ATTR])
    handle.attrs[STAND_INS_ATTR] = _stand_ins_to_json(
        merge_stand_ins(existing, stand_ins)
    )


def write_stand_ins(path: Path, stand_ins: Sequence[SyntheticStandIn]) -> None:
    """Open ``path`` in append mode and merge ``stand_ins`` into its root attrs."""
    with h5py.File(path, "a") as handle:
        write_stand_ins_to_handle(handle, stand_ins)


def read_stand_ins(path: Path) -> list[SyntheticStandIn] | None:
    """Stand-ins registered in one stage artifact.

    Returns ``None`` when the artifact carries no :data:`STAND_INS_ATTR` at all
    (unregistered: written before #354, or by a stage that never checks), and a
    possibly empty list otherwise.
    """
    with h5py.File(path, "r") as handle:
        if STAND_INS_ATTR not in handle.attrs:
            return None
        return _stand_ins_from_json(handle.attrs[STAND_INS_ATTR])


@dataclass(frozen=True)
class CollectedStandIns:
    """Stand-ins collected from a run's stage artifacts."""

    stand_ins: list[SyntheticStandIn]
    #: Registering stages (:data:`STAND_IN_REGISTERING_STAGES`) whose artifact
    #: exists but carries no registration attribute.
    unregistered_stages: list[str] = field(default_factory=list)
    #: Stages with a recorded (or explicitly passed) artifact path that is
    #: missing on disk or unreadable.
    missing_artifacts: list[str] = field(default_factory=list)
    #: Registering stages that were asked for but have no artifact at all — no
    #: stage record, a record without ``artifact_path``, and no explicit path
    #: (#373). Never treated as clean.
    not_run_stages: list[str] = field(default_factory=list)

    @property
    def unverified_stages(self) -> list[str]:
        """Registering stages whose stand-in inventory could not be read.

        Union (first-appearance order) of :attr:`unregistered_stages`,
        :attr:`not_run_stages`, and registering stages in
        :attr:`missing_artifacts`. A consumer that claims science validity must
        treat each of these as a reason it cannot.
        """
        out: list[str] = []
        for name in (
            *self.unregistered_stages,
            *self.not_run_stages,
            *(m for m in self.missing_artifacts if m in STAND_IN_REGISTERING_STAGES),
        ):
            if name not in out:
                out.append(name)
        return out


def collect_stand_ins(
    manifest: RunManifest,
    *,
    stages: Sequence[str] | None = None,
    artifact_paths: Mapping[str, Path | str | None] | None = None,
) -> CollectedStandIns:
    """Collect point-of-use stand-ins from every artifact a consumer actually used.

    Parameters
    ----------
    manifest:
        Run manifest; each stage record's ``artifact_path`` is read.
    stages:
        Restrict to these stage names (in this order). Default: every stage in
        the manifest, in manifest order, plus any stage in ``artifact_paths``.
    artifact_paths:
        Explicit per-stage artifact paths that **override** the manifest record
        (e.g. inference's ``*_artifact_path`` arguments, #373). The artifact a
        consumer read is the one scanned, whatever the manifest says. ``None``
        values fall back to the manifest record.

    Limitations
    -----------
    Only stages whose code registers its stand-ins can be collected. A stage in
    :data:`STAND_IN_REGISTERING_STAGES` whose artifact lacks :data:`STAND_INS_ATTR`
    is reported in ``unregistered_stages``, and one with no artifact at all in
    ``not_run_stages`` — never silently treated as clean.
    """
    overrides: dict[str, Path] = {
        name: Path(path)
        for name, path in (artifact_paths or {}).items()
        if path is not None
    }
    if stages is not None:
        names = list(stages)
    else:
        names = list(manifest.stages)
        names.extend(n for n in overrides if n not in names)
    found: list[SyntheticStandIn] = []
    unregistered: list[str] = []
    missing: list[str] = []
    not_run: list[str] = []
    for name in names:
        record = manifest.stages.get(name)
        if name in overrides:
            path = overrides[name]
        else:
            if record is not None and record.status is StageStatus.SKIPPED:
                continue
            if record is None or not record.artifact_path:
                if name in STAND_IN_REGISTERING_STAGES:
                    not_run.append(name)
                continue
            path = Path(record.artifact_path)
        if not path.is_file():
            missing.append(name)
            continue
        try:
            items = read_stand_ins(path)
        except OSError:
            missing.append(name)
            continue
        if items is None:
            if name in STAND_IN_REGISTERING_STAGES:
                unregistered.append(name)
            continue
        found.extend(items)
    return CollectedStandIns(
        stand_ins=merge_stand_ins(found),
        unregistered_stages=unregistered,
        missing_artifacts=missing,
        not_run_stages=not_run,
    )


# ---------------------------------------------------------------------------
# Forward-model stage stand-ins (registered by the pipeline wrappers so that
# forward_model.py itself is untouched; issue #354)
# ---------------------------------------------------------------------------


def astrometric_stage_stand_ins(
    config: PipelineConfig,
    *,
    n_realizations: int,
    n_accepted: int,
) -> list[SyntheticStandIn]:
    """Synthetic paths ``selection_function_astrometric`` takes on this config."""
    pop = config.selection_function_astrometric.mock_population
    return [
        SyntheticStandIn(
            name="box_prior_mock_population",
            stage="selection_function_astrometric",
            kind="synthetic_data",
            replaces="a mock population drawn from the population model being inferred",
            description=(
                "The mock binaries injected through gaiamock are drawn from fixed "
                "box priors in config (period, eccentricity, M1, M2, flux ratio, "
                "absolute magnitude), not from population_model's mass function. The "
                "detection_fraction this stage reports is the accepted fraction of "
                "that box-prior population, so it is a property of the box, not a "
                "selection function evaluated on the population being inferred."
            ),
            config_keys=[
                "selection_function_astrometric.mock_population.sampling",
                "selection_function_astrometric.mock_population.N_realizations",
            ],
            values={
                "sampling": str(getattr(pop.sampling, "value", pop.sampling)),
                "n_realizations": int(n_realizations),
                "n_accepted_orbital": int(n_accepted),
                "detection_fraction": (
                    float(n_accepted) / float(n_realizations) if n_realizations else None
                ),
            },
        ),
        SyntheticStandIn(
            name="mock_insufficient_visibility_fraction",
            stage="selection_function_astrometric",
            kind="config_placeholder",
            replaces="a forward-modeled insufficient-visibility (too faint) fraction",
            description=(
                "The fraction of mock draws tagged insufficient_visibility is the "
                "configured faint_draw_fraction, not a gaiamock outcome. Every "
                "solution-type fraction compared against the real catalog inherits "
                "this configured number."
            ),
            config_keys=[
                "selection_function_astrometric.mock_population.faint_draw_fraction"
            ],
            values={"faint_draw_fraction": float(pop.faint_draw_fraction)},
        ),
    ]


def followup_calibration_status(
    config: PipelineConfig,
    *,
    calibration_passed: bool,
) -> GateStatusValue:
    """``not_calibrated`` without a real follow-up catalog; else passed / failed.

    With ``calibration.real_followup_catalog_path`` null the stage compares one
    synthetic draw against a second draw from the same generator (#351), so a KS
    pass is guaranteed and tests nothing.
    """
    real = config.selection_function_followup.calibration.real_followup_catalog_path
    if not real:
        return "not_calibrated"
    return "passed" if calibration_passed else "failed"


def followup_stage_stand_ins(config: PipelineConfig) -> list[SyntheticStandIn]:
    """Synthetic paths ``selection_function_followup`` takes on this config."""
    fu = config.selection_function_followup
    out = [
        SyntheticStandIn(
            name="synthetic_followup_catalog",
            stage="selection_function_followup",
            kind="synthetic_data",
            replaces=(
                "follow-up selection probabilities evaluated on the astrometric "
                "stage's accepted mocks"
            ),
            description=(
                "The follow-up catalog is 200 synthetic systems from a hand-written "
                "generator (on-list and survey coin flips, uniform Teff, Gaussian "
                "proper motion, fixed adoption date, source_id 10000+i), not the "
                "astrometric stage's accepted mocks. Its mean probability is what "
                "inference uses as SF_followup (#351)."
            ),
            config_keys=["selection_function_followup"],
        )
    ]
    if not fu.calibration.real_followup_catalog_path:
        out.append(
            SyntheticStandIn(
                name="synthetic_followup_calibration_twin",
                stage="selection_function_followup",
                kind="synthetic_data",
                replaces="a real follow-up catalog to calibrate against",
                description=(
                    "calibration.real_followup_catalog_path is null, so the 'real' "
                    "side of the follow-up calibration is a second draw from the same "
                    "synthetic generator with a different seed. The KS comparison "
                    "is synthetic vs synthetic and cannot fail; the calibration is "
                    "reported as not_calibrated (#351)."
                ),
                config_keys=[
                    "selection_function_followup.calibration.real_followup_catalog_path"
                ],
                values={"real_followup_catalog_path": None},
            )
        )
    return out


def annotate_followup_artifact(
    path: Path,
    config: PipelineConfig,
) -> GateStatusValue:
    """Write calibration status + stand-ins onto a follow-up artifact; return status."""
    with h5py.File(path, "a") as handle:
        passed = bool(handle.attrs.get("calibration_passed", False))
        status = followup_calibration_status(config, calibration_passed=passed)
        handle.attrs[FOLLOWUP_CALIBRATION_STATUS_ATTR] = status
        write_stand_ins_to_handle(handle, followup_stage_stand_ins(config))
    return status


def followup_report_header(status: GateStatusValue) -> str:
    """Plain-language header prepended to the follow-up calibration report."""
    if status == "not_calibrated":
        return (
            "*** NOT CALIBRATED — synthetic vs synthetic (#351) ***\n"
            "No real follow-up catalog is configured "
            "(selection_function_followup.calibration.real_followup_catalog_path: "
            "null). Both sides of the KS comparison below are draws from the same "
            "synthetic generator, so 'overall_passed' tests nothing and is NOT a "
            "calibration. calibration_status: not_calibrated."
        )
    return f"calibration_status: {status}"


# ---------------------------------------------------------------------------
# Upstream gate status
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamGateStatus:
    """One upstream validation gate as seen by a consumer stage."""

    stage: str
    gate: str
    status: GateStatusValue
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "gate": self.gate,
            "status": self.status,
            "passed": self.passed,
            "detail": self.detail,
        }

    def one_line(self) -> str:
        return f"{self.stage}.{self.gate}: {self.status} — {self.detail}"


def astrometric_gate_status(path: Path | None) -> UpstreamGateStatus:
    """Read ``validation_gate_passed`` from a ``selection_function_astrometric`` artifact.

    A missing artifact or attribute is ``not_run`` / ``unreadable`` — never passed.
    """
    stage, gate = "selection_function_astrometric", "validation_gate"
    if path is None or not Path(path).is_file():
        return UpstreamGateStatus(stage, gate, "not_run", "no artifact for this run")
    try:
        with h5py.File(path, "r") as handle:
            if "validation_gate_passed" not in handle.attrs:
                return UpstreamGateStatus(
                    stage, gate, "unreadable", "artifact lacks validation_gate_passed"
                )
            passed = bool(handle.attrs["validation_gate_passed"])
            frac = handle.attrs.get("detection_fraction")
    except OSError as exc:
        return UpstreamGateStatus(stage, gate, "unreadable", f"cannot open: {exc}")
    detail = (
        f"validation_gate_passed={passed}, detection_fraction={float(frac):.4g}"
        if frac is not None
        else f"validation_gate_passed={passed}"
    )
    return UpstreamGateStatus(stage, gate, "passed" if passed else "failed", detail)


def followup_gate_status(path: Path | None) -> UpstreamGateStatus:
    """Read the follow-up calibration status (``calibration_status`` attr).

    Artifacts written before #351/#352 carry only ``calibration_passed``; that alone
    cannot distinguish a real calibration from synthetic-vs-synthetic, so it is
    reported ``unreadable`` rather than passed.
    """
    stage, gate = "selection_function_followup", "calibration"
    if path is None or not Path(path).is_file():
        return UpstreamGateStatus(stage, gate, "not_run", "no artifact for this run")
    try:
        with h5py.File(path, "r") as handle:
            raw = handle.attrs.get(FOLLOWUP_CALIBRATION_STATUS_ATTR)
            passed_attr = handle.attrs.get("calibration_passed")
    except OSError as exc:
        return UpstreamGateStatus(stage, gate, "unreadable", f"cannot open: {exc}")
    if raw is None:
        return UpstreamGateStatus(
            stage,
            gate,
            "unreadable",
            f"artifact lacks {FOLLOWUP_CALIBRATION_STATUS_ATTR} "
            f"(calibration_passed={passed_attr}); real vs synthetic unknown",
        )
    status = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    if status not in ("passed", "failed", "not_calibrated"):
        return UpstreamGateStatus(stage, gate, "unreadable", f"unknown status {status!r}")
    return UpstreamGateStatus(
        stage, gate, status, f"calibration_passed={passed_attr}"  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Science validity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScienceValidity:
    """Whether a consumer stage's output may be presented as a science result.

    ``science_valid`` is True only when every upstream gate passed, no stand-in fed
    the result, and every in-stage check passed. There is no configuration under
    which a failed or unreadable gate yields ``science_valid=True``.
    """

    science_valid: bool
    policy: UpstreamGatePolicy
    gates: list[UpstreamGateStatus]
    stand_in_names: list[str]
    failed_checks: list[str]
    unregistered_stages: list[str] = field(default_factory=list)

    @property
    def gate_failures(self) -> list[UpstreamGateStatus]:
        return [g for g in self.gates if not g.passed]

    def reasons(self) -> list[str]:
        """Every reason this result is not science-valid (empty when it is)."""
        out = [f"upstream gate not passed: {g.one_line()}" for g in self.gate_failures]
        out.extend(f"stand-in in force: {name}" for name in self.stand_in_names)
        out.extend(f"check failed: {c}" for c in self.failed_checks)
        out.extend(
            "stand-in registration not verified (artifact missing, not run, or "
            f"carries no stand-in registration): {s}"
            for s in self.unregistered_stages
        )
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "science_valid": self.science_valid,
            "policy": self.policy,
            "gates": [g.as_dict() for g in self.gates],
            "stand_in_names": list(self.stand_in_names),
            "failed_checks": list(self.failed_checks),
            "unregistered_stages": list(self.unregistered_stages),
            "reasons": self.reasons(),
        }


def assess_science_validity(
    *,
    policy: UpstreamGatePolicy,
    gates: Sequence[UpstreamGateStatus],
    stand_ins: Sequence[SyntheticStandIn] = (),
    failed_checks: Sequence[str] = (),
    unregistered_stages: Sequence[str] = (),
) -> ScienceValidity:
    """Combine gates, stand-ins and in-stage checks into one verdict (never raises)."""
    names = [s.name for s in stand_ins]
    valid = (
        all(g.passed for g in gates)
        and not names
        and not failed_checks
        and not unregistered_stages
    )
    return ScienceValidity(
        science_valid=valid,
        policy=policy,
        gates=list(gates),
        stand_in_names=names,
        failed_checks=list(failed_checks),
        unregistered_stages=list(unregistered_stages),
    )


def enforce_upstream_gates(
    gates: Sequence[UpstreamGateStatus],
    *,
    policy: UpstreamGatePolicy,
    consumer_stage: str,
) -> None:
    """Raise :class:`UpstreamGateFailedError` under ``refuse`` when any gate failed."""
    failures = [g for g in gates if not g.passed]
    if policy == "refuse" and failures:
        lines = "; ".join(g.one_line() for g in failures)
        raise UpstreamGateFailedError(
            f"{consumer_stage} refused (inference.upstream_gate_policy=refuse): {lines}"
        )


def format_science_validity_block(validity: ScienceValidity | Mapping[str, Any]) -> str:
    """Full-detail block for stage reports and captions (caveman-exempt)."""
    payload = validity.as_dict() if isinstance(validity, ScienceValidity) else dict(validity)
    valid = bool(payload.get("science_valid", False))
    lines = [
        "--- science validity (#352) ---",
        (
            "science_valid: True"
            if valid
            else "science_valid: False — this is NOT a science-valid result"
        ),
        f"upstream_gate_policy: {payload.get('policy')}",
        "upstream gates:",
    ]
    gates = payload.get("gates") or []
    if not gates:
        lines.append("  (none checked)")
    for g in gates:
        lines.append(f"  {g['stage']}.{g['gate']}: {g['status']} — {g['detail']}")
    reasons = payload.get("reasons") or []
    if reasons:
        lines.append(f"reasons ({len(reasons)}):")
        lines.extend(f"  - {r}" for r in reasons)
    lines.append("--- end science validity ---")
    return "\n".join(lines)


__all__ = [
    "FOLLOWUP_CALIBRATION_STATUS_ATTR",
    "STAND_INS_ATTR",
    "STAND_IN_REGISTERING_STAGES",
    "CollectedStandIns",
    "ScienceValidity",
    "UpstreamGateFailedError",
    "UpstreamGateStatus",
    "annotate_followup_artifact",
    "assess_science_validity",
    "astrometric_gate_status",
    "astrometric_stage_stand_ins",
    "collect_stand_ins",
    "enforce_upstream_gates",
    "followup_calibration_status",
    "followup_gate_status",
    "followup_report_header",
    "followup_stage_stand_ins",
    "format_science_validity_block",
    "merge_stand_ins",
    "read_stand_ins",
    "write_stand_ins",
    "write_stand_ins_to_handle",
]
