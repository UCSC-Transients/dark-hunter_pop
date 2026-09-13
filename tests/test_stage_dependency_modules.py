"""Regression pins for ``STAGE_REGISTRY`` ``dependency_modules`` completeness.

A stage's ``source_hash`` is computed over its declared ``dependency_modules``
(``run_management`` line ~362). Any module whose source can change the stage's
behaviour but which is *not* declared is invisible to the cache: edit it, and
``plan_stage`` will happily reuse a stale artifact.

These tests were written by the verification agent while verifying PR #179
(issue #143). They pin defects found by a full 14-stage audit; they do NOT fix
them. The ``xfail(strict=True)`` markers keep the required suite green while the
defects stand, and will flip to failures the moment a forward fix lands --
at which point the marker (not the assertion) should be deleted.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from darkhunter_pop.run_management import STAGE_ORDER, STAGE_REGISTRY

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "darkhunter_pop"

# Cross-cutting infrastructure: imported nearly everywhere, and deliberately
# outside the per-stage dependency hashes.
_INFRA = {
    "run_management",
    "schemas",
    "config_schema",
    "config_loader",
    "constants",
    "plotting",
}


def _module_level_imports(module: str) -> set[str]:
    """First-party ``darkhunter_pop`` modules imported at module scope."""
    path = SRC / f"{module}.py"
    if not path.is_file():
        return set()
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.ImportFrom)
            and node.col_offset == 0
            and node.module
            and node.module.startswith("darkhunter_pop")
        ):
            found.add(node.module.split(".")[-1])
    return {m for m in found if (SRC / f"{m}.py").is_file()}


def _transitive(module: str) -> set[str]:
    seen: set[str] = set()
    stack = [module]
    while stack:
        for dep in _module_level_imports(stack.pop()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Defect found verifying PR #179: neither uses_gaiamock stage declares "
        "darkhunter_pop.gaiamock_vendor, so editing the vendor import / "
        "version-triple refusal does not invalidate their cached artifacts."
    ),
)
@pytest.mark.parametrize(
    "stage",
    ["mass_derivation_bulk", "selection_function_astrometric"],
)
def test_gaiamock_stages_declare_gaiamock_vendor(stage: str) -> None:
    """``uses_gaiamock`` stages must hash ``gaiamock_vendor``.

    ``gaiamock_vendor`` owns ``import_gaiamock_mod`` / ``read_versions`` and the
    version-triple refusal. Editing it must invalidate these stages' artifacts.
    """
    spec = STAGE_REGISTRY[stage]
    assert spec.uses_gaiamock, f"{stage} is expected to be a gaiamock stage"
    assert "darkhunter_pop.gaiamock_vendor" in spec.dependency_modules


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Defect found verifying PR #179: rv_astrometry_gate, joint_orbit_fit, "
        "companion_nature_likelihood, selection_function_astrometric and "
        "selection_function_followup are all left on the bare (module,) default."
    ),
)
def test_no_stage_is_registered_with_a_bare_module_default() -> None:
    """Every stage whose module pulls in first-party code declares more than itself.

    ``StageSpec`` defaults ``dependency_modules`` to ``(module,)``; a stage left
    on that default silently under-hashes everything its module imports.
    """
    offenders = []
    for name in STAGE_ORDER:
        spec = STAGE_REGISTRY[name]
        short = spec.module.split(".")[-1]
        if spec.dependency_modules != (spec.module,):
            continue
        if _transitive(short) - _INFRA:
            offenders.append(name)
    assert not offenders, (
        "stages left on the bare (module,) dependency default despite importing "
        f"first-party modules: {sorted(offenders)}"
    )


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap found verifying PR #179: 11 of 14 stages under-declare "
        "dependency_modules. Only sample_selection was fixed this wave "
        "(#151/#158/#166). Needs its own ticket; do not delete this test to "
        "make it pass."
    ),
)
def test_all_stages_declare_their_module_level_dependencies() -> None:
    gaps: dict[str, list[str]] = {}
    for name in STAGE_ORDER:
        spec = STAGE_REGISTRY[name]
        short = spec.module.split(".")[-1]
        declared = {d.split(".")[-1] for d in spec.dependency_modules}
        reachable = _transitive(short) | {short}
        missing = sorted((reachable - declared) - _INFRA)
        if missing:
            gaps[name] = missing
    assert not gaps, f"under-declared dependency_modules: {gaps}"


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Layering inversion found verifying PR #179: data_acquisition (stage 1) "
        "imports diagnostics (stage 14) at module scope, which transitively "
        "reaches ~25 modules and makes any honest dependency_modules list for "
        "the early stages degenerate to 'everything'."
    ),
)
def test_data_acquisition_does_not_import_diagnostics() -> None:
    assert "diagnostics" not in _module_level_imports("data_acquisition")
