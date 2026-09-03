"""Stage: ``sample_selection`` — named literature cut chains.

CONTINUATION_PLAN §4, §8.7, §12, §15 Q10. Each published sample is a first-class
named selection with its own frozen YAML file, per-sample ``reproduction`` /
``forward_model`` mode, ordered cut chain, parent ADQL, and optional
``depends_on`` / ``inherits`` links.

This module owns the interface later Phase 8 slots build against. Per-sample
literature cut transcription is out of scope (#107 / #108 / #110).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
import yaml

from darkhunter_pop.config_loader import (
    enabled_selection_content_fingerprint,
    repo_root,
)
from darkhunter_pop.config_schema import (
    CutKind,
    ParentQueryDRSpec,
    PipelineConfig,
    SampleCut,
    SampleSelectionConfig,
    SampleSelectionEntry,
    SampleSelectionFile,
    SampleSelectionMode,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    StageAction,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import ActiveDRMode, RunManifest, StageStatus

SCHEMA_VERSION: Final[int] = 1
EXPLICIT_EXCLUSIONS_CUT_ID: Final[str] = "explicit_exclusions"
PAPER_MASS_SOURCE: Final[str] = "paper"
PIPELINE_MASS_SOURCE: Final[str] = "pipeline"
# Sentinel name in declarative predicates: ``column is not_applicable``.
_NOT_APPLICABLE_NAME: Final[str] = "not_applicable"
_ALLOWED_CALLS: Final[frozenset[str]] = frozenset(
    {"P", "in_sample", "defined", "not_applicable"}
)
_ALLOWED_BINOPS: Final[tuple[type, ...]] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)


class UnhandledSampleSelectionModeError(ValueError):
    """Raised when a ``SampleSelectionMode`` member is not dispatched."""


class CutExpressionError(ValueError):
    """Raised when a declarative cut expression is illegal or unbound."""


class SampleSelectionError(ValueError):
    """Raised for registry / inherit / depends_on resolution failures."""


class CutOutcome(str, Enum):
    """Three-state cut result (CONTINUATION_PLAN §15 Q10).

    ``NOT_APPLICABLE`` is distinct from ``FAILED``. Collapsing the two into
    boolean false makes subsample counts irreproducible (e.g. El-Badry 2026
    ``M̃1`` undefined for evolved sources and for magnitudes outside the
    Janssens range).
    """

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class NotApplicable:
    """Sentinel stored on a row when a quantity is undefined, not missing.

    Typical reasons later slots will set: ``evolved``, ``outside_janssens_range``.
    """

    reason: str = "undefined"


class _NotApplicableToken:
    """Bound to the name ``not_applicable`` inside predicate evaluation."""


_NOT_APPLICABLE_TOKEN = _NotApplicableToken()


def mass_source_for_mode(mode: SampleSelectionMode) -> str:
    """Return which primary-mass assumption a mode uses (CONTINUATION_PLAN §4.2–§4.3).

    Exhaustive: a new ``SampleSelectionMode`` member must be handled here or
    this raises rather than falling through.
    """
    if mode is SampleSelectionMode.REPRODUCTION:
        return PAPER_MASS_SOURCE
    if mode is SampleSelectionMode.FORWARD_MODEL:
        return PIPELINE_MASS_SOURCE
    raise UnhandledSampleSelectionModeError(
        f"unhandled SampleSelectionMode member: {mode!r}"
    )


def parent_query_for_mode(
    spec: SampleSelectionFile, dr_mode: ActiveDRMode
) -> ParentQueryDRSpec:
    """Return the DR-path-specific parent ADQL block (§4.5, §12.4)."""
    if spec.parent_query is None:
        raise SampleSelectionError(
            f"sample {spec.name!r}: parent_query unresolved (inherits not applied?)"
        )
    if dr_mode is ActiveDRMode.DR3:
        return spec.parent_query.dr3
    if dr_mode is ActiveDRMode.DR4:
        return spec.parent_query.dr4
    raise ValueError(f"unsupported active_dr_mode: {dr_mode!r}")


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_not_applicable(value: Any) -> bool:
    return isinstance(value, NotApplicable)


def _is_missing(value: Any) -> bool:
    return value is None or _is_nan(value)


@dataclass
class CutAttrition:
    """One waterfall row: N in/out plus distinct failed vs not-applicable counts."""

    cut_id: str
    kind: CutKind
    n_in: int
    n_passed: int
    n_failed: int
    n_not_applicable: int
    n_out: int
    expected_n_after: int | None = None
    not_applicable_reasons: dict[str, int] = field(default_factory=dict)
    skipped_for_mode: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cut_id": self.cut_id,
            "kind": self.kind.value,
            "n_in": self.n_in,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_not_applicable": self.n_not_applicable,
            "n_out": self.n_out,
            "expected_n_after": self.expected_n_after,
            "not_applicable_reasons": dict(self.not_applicable_reasons),
            "skipped_for_mode": self.skipped_for_mode,
        }


@dataclass
class SampleEvaluationResult:
    """Membership + attrition for one named sample under one mode."""

    name: str
    mode: SampleSelectionMode
    mass_source: str
    parent_adql: str
    surviving_source_ids: tuple[int, ...]
    attrition: list[CutAttrition]
    n_parent: int
    n_surviving: int
    outcomes_by_source: dict[int, list[tuple[str, CutOutcome, str | None]]] = field(
        default_factory=dict
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "mass_source": self.mass_source,
            "parent_adql": self.parent_adql,
            "surviving_source_ids": list(self.surviving_source_ids),
            "attrition": [row.as_dict() for row in self.attrition],
            "n_parent": self.n_parent,
            "n_surviving": self.n_surviving,
        }


class SampleSelection:
    """Evaluate one named sample's ordered cut chain against a row catalog.

    Parameters
    ----------
    spec
        Fully resolved ``SampleSelectionFile`` (inherits already applied).
    mode
        Registry mode for this run; overrides the file's default ``mode``.
    """

    def __init__(
        self,
        spec: SampleSelectionFile,
        *,
        mode: SampleSelectionMode,
        dr_mode: ActiveDRMode = ActiveDRMode.DR3,
    ) -> None:
        if spec.parent_query is None or spec.cuts is None:
            raise SampleSelectionError(
                f"sample {spec.name!r}: inherit resolution did not fill "
                "parent_query/cuts"
            )
        self.spec = spec
        self.mode = mode
        self.dr_mode = dr_mode
        self.mass_source = mass_source_for_mode(mode)

    def parent_adql(self) -> str:
        """Literal parent-catalog ADQL for the active data release (§4.5)."""
        return parent_query_for_mode(self.spec, self.dr_mode).adql

    def cuts_for_mode(self) -> list[SampleCut]:
        """Ordered cuts whose ``applies_to`` includes the current mode."""
        assert self.spec.cuts is not None
        return [cut for cut in self.spec.cuts if self.mode in cut.applies_to]

    def bind_row(
        self,
        row: Mapping[str, Any],
        *,
        membership: Mapping[str, frozenset[int]] | None = None,
    ) -> dict[str, Any]:
        """Copy a row and bind ``m1_msun`` according to the mode switch (§4.2)."""
        bound = dict(row)
        bound["_membership"] = membership or {}
        if self.mass_source == PAPER_MASS_SOURCE:
            paper = bound.get("paper_m1_msun", bound.get("m1_msun"))
            primary = self.spec.primary_mass
            if (
                primary is not None
                and primary.method == "fixed"
                and primary.value_msun is not None
            ):
                paper = primary.value_msun
            bound["m1_msun"] = paper
        elif self.mass_source == PIPELINE_MASS_SOURCE:
            bound["m1_msun"] = bound.get("pipeline_m1_msun", bound.get("m1_msun"))
        else:
            raise UnhandledSampleSelectionModeError(
                f"unhandled mass source {self.mass_source!r}"
            )
        return bound

    def evaluate(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        membership: Mapping[str, frozenset[int]] | None = None,
    ) -> SampleEvaluationResult:
        """Apply the ordered cut chain; survivors are AND of every cut.

        Not-applicable and failed are both non-survivors of a cut, but they
        are counted separately in the attrition waterfall.
        """
        dep_membership = membership or {}
        remaining = [self.bind_row(row, membership=dep_membership) for row in rows]
        n_parent = len(remaining)
        attrition: list[CutAttrition] = []
        outcomes: dict[int, list[tuple[str, CutOutcome, str | None]]] = {
            int(row["source_id"]): [] for row in remaining
        }

        for cut in self.cuts_for_mode():
            attrition.append(
                self._apply_cut(cut, remaining, outcomes, dep_membership)
            )
            remaining = [
                row
                for row in remaining
                if outcomes[int(row["source_id"])][-1][1] is CutOutcome.PASSED
            ]

        exclusions = list(self.spec.exclusions or [])
        if exclusions:
            exclusion_cut = SampleCut(
                id=EXPLICIT_EXCLUSIONS_CUT_ID,
                kind=CutKind.EXCLUSION,
                expression="source_id not in exclusion_ids",
                expected_n_after=exclusions[-1].expected_n_after,
                parameters={
                    "exclusion_ids": ",".join(str(e.source_id) for e in exclusions)
                },
            )
            attrition.append(
                self._apply_cut(exclusion_cut, remaining, outcomes, dep_membership)
            )
            remaining = [
                row
                for row in remaining
                if outcomes[int(row["source_id"])][-1][1] is CutOutcome.PASSED
            ]

        surviving = tuple(int(row["source_id"]) for row in remaining)
        return SampleEvaluationResult(
            name=self.spec.name,
            mode=self.mode,
            mass_source=self.mass_source,
            parent_adql=self.parent_adql(),
            surviving_source_ids=surviving,
            attrition=attrition,
            n_parent=n_parent,
            n_surviving=len(surviving),
            outcomes_by_source=outcomes,
        )

    def _apply_cut(
        self,
        cut: SampleCut,
        remaining: Sequence[Mapping[str, Any]],
        outcomes: dict[int, list[tuple[str, CutOutcome, str | None]]],
        membership: Mapping[str, frozenset[int]],
    ) -> CutAttrition:
        n_in = len(remaining)
        n_passed = 0
        n_failed = 0
        n_na = 0
        na_reasons: dict[str, int] = {}
        for row in remaining:
            outcome, reason = evaluate_cut(cut, row, membership=membership)
            source_id = int(row["source_id"])
            outcomes[source_id].append((cut.id, outcome, reason))
            if outcome is CutOutcome.PASSED:
                n_passed += 1
            elif outcome is CutOutcome.FAILED:
                n_failed += 1
            elif outcome is CutOutcome.NOT_APPLICABLE:
                n_na += 1
                key = reason or "undefined"
                na_reasons[key] = na_reasons.get(key, 0) + 1
            else:
                raise ValueError(f"unhandled CutOutcome member: {outcome!r}")
        return CutAttrition(
            cut_id=cut.id,
            kind=cut.kind,
            n_in=n_in,
            n_passed=n_passed,
            n_failed=n_failed,
            n_not_applicable=n_na,
            n_out=n_passed,
            expected_n_after=cut.expected_n_after,
            not_applicable_reasons=na_reasons,
        )


def evaluate_cut(
    cut: SampleCut,
    row: Mapping[str, Any],
    *,
    membership: Mapping[str, frozenset[int]] | None = None,
) -> tuple[CutOutcome, str | None]:
    """Evaluate one cut against one bound row.

    Returns ``(outcome, not_applicable_reason)``. The reason is set only for
    ``NOT_APPLICABLE``.
    """
    env = _evaluation_env(cut, row, membership=membership or {})

    for column in cut.requires_defined:
        value = env.get(column, None)
        if _is_not_applicable(value):
            return CutOutcome.NOT_APPLICABLE, value.reason
        if _is_missing(value) or column not in env:
            return CutOutcome.NOT_APPLICABLE, f"missing:{column}"

    if cut.undefined_when:
        flag, _reason = _eval_predicate(cut.undefined_when, env)
        if flag is True:
            return CutOutcome.NOT_APPLICABLE, "undefined_when"

    if cut.from_sample is not None:
        source_id = env.get("source_id")
        members = (membership or {}).get(cut.from_sample)
        if members is None:
            return CutOutcome.NOT_APPLICABLE, f"missing_sample:{cut.from_sample}"
        if int(source_id) not in members:
            return CutOutcome.FAILED, None

    if cut.kind is CutKind.EXCLUSION:
        excluded = _exclusion_ids(cut)
        source_id = int(env["source_id"])
        if source_id in excluded:
            return CutOutcome.FAILED, None
        return CutOutcome.PASSED, None

    try:
        result, na_reason = _eval_predicate(cut.expression, env)
    except CutExpressionError:
        raise
    if na_reason is not None:
        return CutOutcome.NOT_APPLICABLE, na_reason
    if result is True:
        return CutOutcome.PASSED, None
    if result is False:
        return CutOutcome.FAILED, None
    raise CutExpressionError(
        f"cut {cut.id!r}: expression did not evaluate to a boolean"
    )


def _exclusion_ids(cut: SampleCut) -> set[int]:
    raw = cut.parameters.get("exclusion_ids", "")
    if raw is None or raw == "":
        return set()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return {int(raw)}
    return {int(part) for part in str(raw).split(",") if part.strip()}


def _evaluation_env(
    cut: SampleCut,
    row: Mapping[str, Any],
    *,
    membership: Mapping[str, frozenset[int]],
) -> dict[str, Any]:
    env: dict[str, Any] = dict(row)
    env.update(cut.parameters)
    env[_NOT_APPLICABLE_NAME] = _NOT_APPLICABLE_TOKEN
    env["_membership"] = membership
    env["_cut"] = cut
    return env


def _eval_predicate(
    expression: str, env: Mapping[str, Any]
) -> tuple[bool | None, str | None]:
    """Return ``(bool_or_none, na_reason)``. ``None`` bool means N/A."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CutExpressionError(f"illegal cut expression {expression!r}: {exc}") from exc
    value = _eval_node(tree.body, env)
    if isinstance(value, NotApplicable):
        return None, value.reason
    if isinstance(value, bool):
        return value, None
    raise CutExpressionError(
        f"cut expression {expression!r} produced {type(value).__name__}, not bool"
    )


def _eval_node(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise CutExpressionError(f"unbound name {node.id!r} in cut expression")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        if isinstance(operand, NotApplicable):
            return operand
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise CutExpressionError(f"unsupported unary operator {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        return _eval_boolop(node, env)
    if isinstance(node, ast.BinOp):
        return _eval_binop(node, env)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, env)
    if isinstance(node, ast.Call):
        return _eval_call(node, env)
    if isinstance(node, ast.List):
        return [_eval_node(elt, env) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, env) for elt in node.elts)
    raise CutExpressionError(
        f"unsupported expression node {type(node).__name__} "
        "(declarative predicates only; no inline Python)"
    )


def _eval_boolop(node: ast.BoolOp, env: Mapping[str, Any]) -> Any:
    is_and = isinstance(node.op, ast.And)
    is_or = isinstance(node.op, ast.Or)
    if not is_and and not is_or:
        raise CutExpressionError(f"unsupported boolop {type(node.op).__name__}")
    na_hold: NotApplicable | None = None
    for value_node in node.values:
        value = _eval_node(value_node, env)
        if isinstance(value, NotApplicable):
            na_hold = value
            continue
        truth = bool(value)
        if is_and and not truth:
            return False
        if is_or and truth:
            return True
    if na_hold is not None:
        return na_hold
    return True if is_and else False


def _eval_binop(node: ast.BinOp, env: Mapping[str, Any]) -> Any:
    if not isinstance(node.op, _ALLOWED_BINOPS):
        raise CutExpressionError(f"unsupported operator {type(node.op).__name__}")
    left = _eval_node(node.left, env)
    right = _eval_node(node.right, env)
    if isinstance(left, NotApplicable):
        return left
    if isinstance(right, NotApplicable):
        return right
    if _is_missing(left) or _is_missing(right):
        return NotApplicable("missing_operand")
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.FloorDiv):
        return left // right
    if isinstance(node.op, ast.Mod):
        return left % right
    if isinstance(node.op, ast.Pow):
        return left**right
    raise CutExpressionError(f"unsupported operator {type(node.op).__name__}")


def _eval_compare(node: ast.Compare, env: Mapping[str, Any]) -> Any:
    left = _eval_node(node.left, env)
    comparators = [_eval_node(comp, env) for comp in node.comparators]
    current = left
    for op, right in zip(node.ops, comparators, strict=True):
        result = _compare_pair(op, current, right)
        if isinstance(result, NotApplicable):
            return result
        if result is False:
            return False
        current = right
    return True


def _compare_pair(op: ast.cmpop, left: Any, right: Any) -> bool | NotApplicable:
    if isinstance(right, _NotApplicableToken):
        if isinstance(op, ast.Is):
            return _is_not_applicable(left)
        if isinstance(op, ast.IsNot):
            return not _is_not_applicable(left)
        raise CutExpressionError("not_applicable is only valid with `is` / `is not`")
    if isinstance(left, _NotApplicableToken):
        raise CutExpressionError("not_applicable cannot appear on the left of a comparison")

    if isinstance(op, (ast.Is, ast.IsNot)):
        if right is None:
            is_null = left is None
            return is_null if isinstance(op, ast.Is) else not is_null
        if isinstance(op, ast.Is):
            return left is right
        return left is not right

    if _is_not_applicable(left):
        return left
    if _is_not_applicable(right):
        return right
    if _is_missing(left) or _is_missing(right):
        return NotApplicable("missing_operand")

    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    raise CutExpressionError(f"unsupported comparison {type(op).__name__}")


def _eval_call(node: ast.Call, env: Mapping[str, Any]) -> Any:
    if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
        raise CutExpressionError(
            f"unsupported call {ast.dump(node.func)}; allowed: {sorted(_ALLOWED_CALLS)}"
        )
    name = node.func.id
    if name == "defined":
        if len(node.args) != 1:
            raise CutExpressionError("defined() takes exactly one argument")
        value = _eval_node(node.args[0], env)
        return not _is_not_applicable(value) and not _is_missing(value)
    if name == "not_applicable":
        if len(node.args) != 1:
            raise CutExpressionError("not_applicable() takes exactly one argument")
        value = _eval_node(node.args[0], env)
        return _is_not_applicable(value)
    if name == "in_sample":
        if len(node.args) != 1:
            raise CutExpressionError("in_sample() takes exactly one sample name")
        sample_name = _eval_node(node.args[0], env)
        if not isinstance(sample_name, str):
            raise CutExpressionError("in_sample() argument must be a string")
        source_id = env.get("source_id")
        members = env.get("_membership", {}).get(sample_name)
        if members is None:
            return NotApplicable(f"missing_sample:{sample_name}")
        return int(source_id) in members
    if name == "P":
        return _eval_probability(node, env)
    raise CutExpressionError(f"unhandled allowed call {name!r}")


def _eval_probability(node: ast.Call, env: Mapping[str, Any]) -> Any:
    """``P(M2 > threshold)`` reads a precomputed probability column.

    Monte Carlo propagation (#19) fills the column; this framework only
    consumes it. Missing / NotApplicable probabilities are N/A, not failed.
    """
    if len(node.args) != 1:
        raise CutExpressionError("P() takes exactly one comparison argument")
    arg = node.args[0]
    if not isinstance(arg, ast.Compare) or len(arg.ops) != 1:
        raise CutExpressionError("P() argument must be a single comparison")
    cut: SampleCut | None = env.get("_cut")
    column = None if cut is None else cut.parameters.get("probability_column")
    if isinstance(column, str) and column in env:
        value = env[column]
    elif "p_m2_above" in env:
        value = env["p_m2_above"]
    elif "p_m2_gt_threshold" in env:
        value = env["p_m2_gt_threshold"]
    else:
        return NotApplicable("missing_probability")
    if _is_not_applicable(value):
        return value
    if _is_missing(value):
        return NotApplicable("missing_probability")
    return value


def load_sample_selection_file(path: Path) -> SampleSelectionFile:
    """Load and validate one per-sample YAML body (before inherit resolution)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SampleSelectionError(f"selection file root must be a mapping: {path}")
    return SampleSelectionFile.model_validate(raw)


def resolve_inherits(
    spec: SampleSelectionFile,
    *,
    files_by_name: Mapping[str, SampleSelectionFile],
    _stack: tuple[str, ...] = (),
) -> SampleSelectionFile:
    """Fill omitted parent_query / cuts from the inherited named sample."""
    if spec.inherits is None:
        return spec
    if spec.name in _stack:
        cycle = " -> ".join([*_stack, spec.name])
        raise SampleSelectionError(f"inherits cycle: {cycle}")
    parent = files_by_name.get(spec.inherits)
    if parent is None:
        raise SampleSelectionError(
            f"sample {spec.name!r} inherits {spec.inherits!r}, which is not loaded"
        )
    resolved_parent = resolve_inherits(
        parent, files_by_name=files_by_name, _stack=(*_stack, spec.name)
    )
    parent_query = (
        spec.parent_query
        if spec.parent_query is not None
        else resolved_parent.parent_query
    )
    cuts = spec.cuts if spec.cuts is not None else resolved_parent.cuts
    if spec.exclusions is None:
        exclusions = resolved_parent.exclusions
    else:
        exclusions = spec.exclusions
    primary_mass = (
        spec.primary_mass
        if spec.primary_mass is not None
        else resolved_parent.primary_mass
    )
    monte_carlo = (
        spec.monte_carlo
        if spec.monte_carlo is not None
        else resolved_parent.monte_carlo
    )
    depends_on = spec.depends_on or list(resolved_parent.depends_on)
    return spec.model_copy(
        update={
            "parent_query": parent_query,
            "cuts": cuts,
            "exclusions": exclusions,
            "primary_mass": primary_mass,
            "monte_carlo": monte_carlo,
            "depends_on": depends_on,
        }
    )


def topological_sample_order(
    specs: Mapping[str, SampleSelectionFile],
) -> list[str]:
    """Enabled-sample evaluation order honoring ``depends_on`` (§8.7)."""
    pending = set(specs)
    ordered: list[str] = []
    while pending:
        ready = [
            name
            for name in sorted(pending)
            if all(dep not in pending for dep in specs[name].depends_on)
        ]
        if not ready:
            raise SampleSelectionError(
                f"depends_on cycle or missing dependency among {sorted(pending)}"
            )
        for name in ready:
            missing = [dep for dep in specs[name].depends_on if dep not in specs]
            if missing:
                raise SampleSelectionError(
                    f"sample {name!r} depends_on {missing}, which are not enabled"
                )
            ordered.append(name)
            pending.remove(name)
    return ordered


class SampleSelectionRegistry:
    """Named-sample registry: load, inherit, depends_on, evaluate.

    Two variants of one paper (``andrews2022`` vs ``andrews2022_modified``) are
    independent entries — never a flag inside one file (§6.6).
    """

    def __init__(
        self,
        config: PipelineConfig | SampleSelectionConfig,
        *,
        repo: Path | None = None,
        files_by_name: Mapping[str, SampleSelectionFile] | None = None,
    ) -> None:
        if isinstance(config, PipelineConfig):
            self.pipeline = config
            self.cfg = config.sample_selection
            self.dr_mode = config.active_dr_mode
        else:
            self.pipeline = None
            self.cfg = config
            self.dr_mode = ActiveDRMode.DR3
        self.repo = repo if repo is not None else repo_root()
        self._raw_files = (
            dict(files_by_name) if files_by_name is not None else self._load_raw_files()
        )
        self._resolved: dict[str, SampleSelectionFile] = {
            name: resolve_inherits(spec, files_by_name=self._raw_files)
            for name, spec in self._raw_files.items()
        }
        for name, spec in self._resolved.items():
            if spec.name != name:
                raise SampleSelectionError(
                    f"selection file name {spec.name!r} does not match registry key {name!r}"
                )

    def _load_raw_files(self) -> dict[str, SampleSelectionFile]:
        by_name = {entry.name: entry for entry in self.cfg.samples}
        pending = {entry.name for entry in self.enabled_entries()}
        loaded: dict[str, SampleSelectionFile] = {}
        while pending:
            name = pending.pop()
            if name in loaded:
                continue
            entry = by_name.get(name)
            if entry is None:
                raise SampleSelectionError(
                    f"inherits/depends_on target {name!r} is not in the registry"
                )
            path = Path(entry.path)
            if not path.is_absolute():
                path = self.repo / path
            spec = load_sample_selection_file(path)
            loaded[name] = spec
            if spec.inherits is not None and spec.inherits not in loaded:
                pending.add(spec.inherits)
            for dep in spec.depends_on:
                if dep not in loaded:
                    pending.add(dep)
        return loaded

    def enabled_entries(self) -> list[SampleSelectionEntry]:
        return [entry for entry in self.cfg.samples if entry.enabled]

    def resolved(self, name: str) -> SampleSelectionFile:
        if name not in self._resolved:
            raise SampleSelectionError(f"sample {name!r} is not loaded")
        return self._resolved[name]

    def selection(self, name: str) -> SampleSelection:
        entry = next((e for e in self.cfg.samples if e.name == name), None)
        if entry is None:
            raise SampleSelectionError(f"sample {name!r} is not in the registry")
        return SampleSelection(
            self.resolved(name), mode=entry.mode, dr_mode=self.dr_mode
        )

    def evaluation_order(self) -> list[str]:
        enabled = {entry.name for entry in self.enabled_entries()}
        specs = {name: self._resolved[name] for name in enabled if name in self._resolved}
        missing_files = enabled - set(specs)
        if missing_files:
            raise SampleSelectionError(
                f"enabled samples have no loaded files: {sorted(missing_files)}"
            )
        return topological_sample_order(specs)

    def evaluate_all(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, SampleEvaluationResult]:
        """Evaluate enabled samples in ``depends_on`` order, threading membership."""
        membership: dict[str, frozenset[int]] = {}
        results: dict[str, SampleEvaluationResult] = {}
        for name in self.evaluation_order():
            result = self.selection(name).evaluate(rows, membership=membership)
            results[name] = result
            membership[name] = frozenset(result.surviving_source_ids)
        return results


@dataclass
class SampleSelectionStageResult:
    """Stage-level payload written beside the HDF5 artifact."""

    schema_version: int
    enabled: bool
    content_fingerprint: str
    results: dict[str, SampleEvaluationResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "content_fingerprint": self.content_fingerprint,
            "results": {name: res.as_dict() for name, res in self.results.items()},
        }


def write_sample_selection_artifact(
    path: Path, result: SampleSelectionStageResult
) -> None:
    """Write HDF5 membership + YAML sidecar waterfall (producer contract)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "sample_selection"
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["enabled"] = result.enabled
        handle.attrs["content_fingerprint"] = result.content_fingerprint
        samples_grp = handle.create_group("samples")
        for name, sample in result.results.items():
            grp = samples_grp.create_group(name)
            grp.attrs["mode"] = sample.mode.value
            grp.attrs["mass_source"] = sample.mass_source
            grp.attrs["n_parent"] = sample.n_parent
            grp.attrs["n_surviving"] = sample.n_surviving
            grp.attrs["parent_adql"] = sample.parent_adql
            ids = np.asarray(sample.surviving_source_ids, dtype=np.int64)
            grp.create_dataset("source_id", data=ids)
            attrition_grp = grp.create_group("attrition")
            for row in sample.attrition:
                cut_grp = attrition_grp.create_group(row.cut_id)
                cut_grp.attrs["kind"] = row.kind.value
                cut_grp.attrs["n_in"] = row.n_in
                cut_grp.attrs["n_passed"] = row.n_passed
                cut_grp.attrs["n_failed"] = row.n_failed
                cut_grp.attrs["n_not_applicable"] = row.n_not_applicable
                cut_grp.attrs["n_out"] = row.n_out
                if row.expected_n_after is not None:
                    cut_grp.attrs["expected_n_after"] = row.expected_n_after
    sidecar = path.with_suffix(".yaml")
    sidecar.write_text(
        yaml.safe_dump(result.as_dict(), sort_keys=False), encoding="utf-8"
    )


def read_sample_selection_artifact(path: Path) -> dict[str, Any]:
    """Load the YAML sidecar written beside the HDF5 artifact."""
    sidecar = path.with_suffix(".yaml")
    if not sidecar.is_file():
        raise FileNotFoundError(f"sample_selection sidecar missing: {sidecar}")
    raw = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"sample_selection sidecar must be a mapping: {sidecar}")
    return dict(raw)


def run_sample_selection(
    rows: Sequence[Mapping[str, Any]],
    config: PipelineConfig,
    *,
    repo: Path | None = None,
) -> SampleSelectionStageResult:
    """Evaluate all enabled samples (science entry point for tests / stage)."""
    registry = SampleSelectionRegistry(config, repo=repo)
    results = registry.evaluate_all(rows)
    return SampleSelectionStageResult(
        schema_version=SCHEMA_VERSION,
        enabled=config.sample_selection.enabled,
        content_fingerprint=enabled_selection_content_fingerprint(
            config, repo=repo
        ),
        results=results,
    )


def run_sample_selection_stage(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> RunManifest:
    """Execute or skip ``sample_selection`` and update the run manifest."""
    spec = STAGE_REGISTRY["sample_selection"]
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

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    result = run_sample_selection(rows or (), config)
    write_sample_selection_artifact(artifact, result)
    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
