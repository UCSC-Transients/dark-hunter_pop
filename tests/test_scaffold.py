"""Scaffold smoke test: the package and every declared stage module import cleanly.

Guards against a broken package layout or a missing module file. Replaced by real per-stage tests
as each stage is implemented.
"""

import importlib

import pytest

MODULES = [
    "darkhunter_pop",
    "darkhunter_pop.constants",
    "darkhunter_pop.config_schema",
    "darkhunter_pop.config_loader",
    "darkhunter_pop.gaiamock_vendor",
    "darkhunter_pop.schemas",
    "darkhunter_pop.run_management",
    "darkhunter_pop.physics_utils",
    "darkhunter_pop.data_acquisition",
    "darkhunter_pop.mass_derivation",
    "darkhunter_pop.rv_consistency",
    "darkhunter_pop.companion_nature",
    "darkhunter_pop.triples",
    "darkhunter_pop.triples.tess_variability",
    "darkhunter_pop.triples.rotation_check",
    "darkhunter_pop.forward_model",
    "darkhunter_pop.population_model",
    "darkhunter_pop.sensitivity_analysis",
    "darkhunter_pop.inference",
    "darkhunter_pop.plotting",
    "darkhunter_pop.diagnostics",
]


@pytest.mark.unit
@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
