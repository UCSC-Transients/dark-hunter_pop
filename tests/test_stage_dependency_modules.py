"""Regression pins for ``STAGE_REGISTRY`` ``dependency_modules`` completeness.

A stage's ``source_hash`` is computed over its declared ``dependency_modules``
(``run_management.compute_source_hash``) -- a *flat* digest that does not follow
imports. Any module whose source can change the stage's behaviour but which is
*not* declared is invisible to the cache: edit it, and ``plan_stage`` will
happily reuse a stale artifact.

History: the verification agent wrote the first version of these tests while
verifying PR #179 (issue #143), pinning the defects of a full 14-stage audit as
``xfail(strict=True)`` plus a ``_KNOWN_GAPS`` ratchet (PR #298). #182 removed
the root cause (``data_acquisition`` importing the ``diagnostics`` stage at
module scope) and #183 declared every remaining gap, so the xfails are now
plain assertions and the ledger is empty.

The rule the registry is held to (dependency lists are hand-declared; these
tests are what keeps them honest):

1. **Closure.** The declared set, plus the stage's own module, is closed under
   first-party *module-scope* imports (including imports nested in module-level
   ``try`` / ``if`` / class bodies, ``import darkhunter_pop.x`` and
   ``from darkhunter_pop import x``), minus the cross-cutting ``_INFRA``
   modules deliberately kept out of stage hashes.
2. **Lazy imports.** Every function-level first-party import made by a declared
   module is itself declared, or listed in ``_LAZY_IMPORT_ALLOWLIST`` with a
   reason. Allowlist entries must stay live (not declared, still imported).
"""

from __future__ import annotations

import ast
import pathlib
from functools import cache

import pytest

from darkhunter_pop.run_management import STAGE_ORDER, STAGE_REGISTRY

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
PKG = "darkhunter_pop"

# Cross-cutting infrastructure deliberately outside the per-stage dependency
# hashes. Each exemption must be justified; anything holding science numbers
# (``constants``: TAG10/Santos tables, M_Ch, ...) is NOT exempt and must be
# declared wherever reachable (#183, #49 review).
#
# - run_management: the registry / cache machinery itself; hashing it would
#   invalidate every stage on any bookkeeping edit.
# - schemas, config_schema: data-model definitions. A field/default change that
#   alters a stage's answer changes the config dump, which the config checksum
#   and per-stage config fingerprint already catch.
# - config_loader: YAML loading + checksum; its outputs are covered by the
#   config checksum refusal.
# - plotting: figure rendering only (CLAUDE.md "Run management": plotting-only
#   edits never invalidate a stage).
# - the package ``__init__`` (docstring only).
_INFRA = frozenset(
    f"{PKG}.{m}"
    for m in (
        "run_management",
        "schemas",
        "config_schema",
        "config_loader",
        "plotting",
    )
) | {PKG}

# (stage, lazily imported module) -> why it need not feed that stage's hash.
_LAZY_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    ("population_model", f"{PKG}.companion_nature"): (
        "population_model imports companion_nature.read_stage_hdf5 lazily only to "
        "read the upstream companion_nature_likelihood artifact. That stage hashes "
        "companion_nature, so a reader change makes it stale; the force-rerun that "
        "follows starts a new run file that re-runs every downstream stage."
    ),
    ("inference", f"{PKG}.companion_nature"): (
        "Reached only via population_model's lazy upstream-artifact reader "
        "(see the population_model entry)."
    ),
    **{
        ("inference", f"{PKG}.{m}"): (
            "Lazy import inside sample_selection's cut-evaluation / parent-cache "
            "paths. inference reaches sample_selection only through "
            "sample_inclusion's selection-file loaders (load_sample_selection_file, "
            "resolve_inherits), which never execute it; the sample_selection stage "
            "that does execute it hashes it."
        )
        for m in (
            # data_acquisition: declared for inference since #339 (forward_model
            # imports it at module scope), so no longer an allowlist entry.
            "andrews2022_atf",
            "elbadry2026_m2_sigma",
            "elbadry2026_selection",
            "janssens_mass",
            "mc_mass_function",
            "shahaf2023b_catalog",
        )
    },
}

# Stage-registry gaps still standing. The #183 fix emptied it; a future
# regression must be fixed, never added here.
_KNOWN_GAPS: dict[str, frozenset[str]] = {}

# Only orchestration entry points (never a stage module) may import the
# ``diagnostics`` *stage* module at module scope (#182).
_DIAGNOSTICS_IMPORTERS_ALLOWED = frozenset({f"{PKG}.pipeline", f"{PKG}.dry_run"})


def _source_path(module: str) -> pathlib.Path | None:
    base = SRC.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _dotted(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _first_party_targets(node: ast.stmt) -> set[str]:
    out: set[str] = set()
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module or node.module.split(".")[0] != PKG:
            return out
        out.add(node.module)
        for alias in node.names:
            # ``from darkhunter_pop import constants`` / ``from pkg.sub import mod``.
            sub = f"{node.module}.{alias.name}"
            if _source_path(sub) is not None:
                out.add(sub)
    elif isinstance(node, ast.Import):
        out.update(a.name for a in node.names if a.name.split(".")[0] == PKG)
    return {m for m in out if _source_path(m) is not None}


@cache
def _imports(module: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(module_scope, lazy)`` first-party imports of ``module``.

    Module scope includes imports nested in module-level ``if`` / ``try`` /
    ``with`` / class bodies (all execute at import time). Lazy means inside a
    function or method body.
    """
    path = _source_path(module)
    if path is None:
        return frozenset(), frozenset()
    top: set[str] = set()
    lazy: set[str] = set()

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        lazy.update(_first_party_targets(inner))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                top.update(_first_party_targets(node))
            elif isinstance(node, ast.ClassDef):
                walk(node.body)
            else:
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(node, attr, None)
                    if isinstance(sub, list):
                        walk(sub)

    walk(ast.parse(path.read_text(encoding="utf-8")).body)
    top.discard(module)
    lazy.discard(module)
    return frozenset(top), frozenset(lazy - top)


def _module_scope_closure(roots: set[str]) -> set[str]:
    seen = set(roots)
    stack = list(roots)
    while stack:
        for dep in _imports(stack.pop())[0]:
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def _declared(stage: str) -> set[str]:
    return set(STAGE_REGISTRY[stage].dependency_modules)


def _missing(stage: str) -> set[str]:
    spec = STAGE_REGISTRY[stage]
    required = _module_scope_closure(_declared(stage) | {spec.module})
    return (required - _declared(stage)) - _INFRA


def _lazy_of_declared(stage: str) -> set[str]:
    declared = _declared(stage) | {STAGE_REGISTRY[stage].module}
    return set().union(*(_imports(m)[1] for m in declared)) - declared - _INFRA


@pytest.mark.unit
def test_scanner_sees_known_imports() -> None:
    """Guard against a broken scanner passing every check vacuously."""
    md_top, _ = _imports(f"{PKG}.mass_derivation")
    assert f"{PKG}.gaiamock_vendor" in md_top
    assert f"{PKG}.data_acquisition" in md_top
    ss_top, ss_lazy = _imports(f"{PKG}.sample_selection")
    assert f"{PKG}.data_acquisition" in ss_lazy
    assert f"{PKG}.data_acquisition" not in ss_top
    assert f"{PKG}.constants" in _imports(f"{PKG}.physics_utils")[0]
    assert f"{PKG}.triples.rotation_check" in _imports(f"{PKG}.triples")[0]
    assert _dotted(SRC / PKG / "triples" / "__init__.py") == f"{PKG}.triples"


@pytest.mark.unit
@pytest.mark.parametrize(
    "stage", [name for name in STAGE_ORDER if STAGE_REGISTRY[name].uses_gaiamock]
)
def test_gaiamock_stages_declare_gaiamock_vendor(stage: str) -> None:
    """``uses_gaiamock`` stages must hash ``gaiamock_vendor``.

    ``gaiamock_vendor`` owns ``import_gaiamock_mod`` / ``read_versions`` and the
    version-triple refusal. Editing it must invalidate these stages' artifacts.
    """
    assert f"{PKG}.gaiamock_vendor" in STAGE_REGISTRY[stage].dependency_modules


@pytest.mark.unit
def test_uses_gaiamock_stages_are_the_expected_pair() -> None:
    """Keeps the parametrized gaiamock test above from silently shrinking."""
    assert {n for n in STAGE_ORDER if STAGE_REGISTRY[n].uses_gaiamock} == {
        "mass_derivation_bulk",
        "selection_function_astrometric",
    }


@pytest.mark.unit
def test_no_stage_is_registered_with_a_bare_module_default() -> None:
    """Every stage whose module pulls in first-party code declares more than itself.

    ``StageSpec`` defaults ``dependency_modules`` to ``(module,)``; a stage left
    on that default silently under-hashes everything its module imports.
    """
    offenders = []
    for name in STAGE_ORDER:
        spec = STAGE_REGISTRY[name]
        if spec.dependency_modules != (spec.module,):
            continue
        if _module_scope_closure({spec.module}) - {spec.module} - _INFRA:
            offenders.append(name)
    assert not offenders, (
        "stages left on the bare (module,) dependency default despite importing "
        f"first-party modules: {sorted(offenders)}"
    )


@pytest.mark.unit
def test_all_stages_declare_their_module_level_dependencies() -> None:
    gaps = {name: sorted(_missing(name)) for name in STAGE_ORDER if _missing(name)}
    assert not gaps, f"under-declared dependency_modules: {gaps}"


@pytest.mark.unit
def test_declared_dependency_modules_resolve_and_are_unique() -> None:
    for name in STAGE_ORDER:
        deps = STAGE_REGISTRY[name].dependency_modules
        assert len(deps) == len(set(deps)), f"{name}: duplicate dependency_modules"
        assert deps[0] == STAGE_REGISTRY[name].module, f"{name}: own module not first"
        for module in deps:
            assert _source_path(module) is not None, f"{name}: {module} has no source"


@pytest.mark.unit
def test_lazy_imports_are_declared_or_allowlisted() -> None:
    """Function-level first-party imports of declared modules feed the hash too."""
    undeclared = {
        name: sorted(m for m in _lazy_of_declared(name) if (name, m) not in _LAZY_IMPORT_ALLOWLIST)
        for name in STAGE_ORDER
    }
    undeclared = {k: v for k, v in undeclared.items() if v}
    assert not undeclared, (
        "lazy first-party imports neither declared in dependency_modules nor "
        f"allowlisted in _LAZY_IMPORT_ALLOWLIST: {undeclared}"
    )


@pytest.mark.unit
def test_lazy_import_allowlist_is_live() -> None:
    """Allowlist entries must name a stage, be undeclared, and still be imported."""
    stale = [
        key
        for key, reason in _LAZY_IMPORT_ALLOWLIST.items()
        if not reason.strip()
        or key[0] not in STAGE_REGISTRY
        or key[1] not in _lazy_of_declared(key[0])
    ]
    assert not stale, f"stale _LAZY_IMPORT_ALLOWLIST entries: {stale}"


@pytest.mark.unit
def test_data_acquisition_does_not_import_diagnostics() -> None:
    """#182: stage 1 must not reach the stage-14 ``diagnostics`` module (or gaiamock)."""
    reach = _module_scope_closure({f"{PKG}.data_acquisition"})
    assert f"{PKG}.diagnostics" not in reach
    assert f"{PKG}.forward_model" not in reach
    assert f"{PKG}.gaiamock_vendor" not in reach


@pytest.mark.unit
def test_only_orchestration_imports_diagnostics_stage_at_module_scope() -> None:
    """#182 layering audit: no stage module imports ``diagnostics`` at module scope."""
    importers = {
        _dotted(p)
        for p in (SRC / PKG).rglob("*.py")
        if f"{PKG}.diagnostics" in _imports(_dotted(p))[0]
    }
    assert importers <= _DIAGNOSTICS_IMPORTERS_ALLOWED, sorted(importers)


@pytest.mark.unit
def test_diagnostic_hooks_depends_only_on_infra() -> None:
    """``diagnostic_hooks`` sits below every stage; it must never import one.

    ``constants`` (a leaf, reached through ``config_loader``) is not a stage.
    """
    hooks = f"{PKG}.diagnostic_hooks"
    leaf = {hooks, f"{PKG}.constants"}
    assert _module_scope_closure({hooks}) - _INFRA <= leaf
    assert not (_imports(hooks)[1] - _INFRA - leaf)


@pytest.mark.unit
def test_dependency_module_gaps_match_known_ledger() -> None:
    """Ratchet: no stage may under-declare anything beyond ``_KNOWN_GAPS``.

    The ledger is empty since #183; keep it that way. A new gap is fixed by
    declaring the module in ``run_management.STAGE_REGISTRY``, never by growing
    the ledger.
    """
    new: dict[str, list[str]] = {}
    fixed: dict[str, list[str]] = {}
    for name in STAGE_ORDER:
        missing = _missing(name)
        known = _KNOWN_GAPS.get(name, frozenset())
        if missing - known:
            new[name] = sorted(missing - known)
        if known - missing:
            fixed[name] = sorted(known - missing)
    assert not new, f"new dependency_modules under-declarations: {new}"
    assert not fixed, f"gaps fixed -- remove them from _KNOWN_GAPS: {fixed}"
    assert set(_KNOWN_GAPS) <= set(STAGE_ORDER), "ledger names an unknown stage"
