"""API contracts for known-truth + comparison catalog loaders (issue #70)."""

from __future__ import annotations

import pytest

from darkhunter_pop.benchmarks import (
    REQUIRED_COMPARISON_CATALOG_IDS,
    check_known_truth_expectations,
    load_all_comparison_catalogs,
    load_known_truth_table_from_config,
    synthetic_observed_from_truth,
    validate_benchmarks_config,
)
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.diagnostics import (
    emit_comparison_catalogs,
    emit_known_truth_benchmarks,
    resolve_diagnostic_dirs,
)
from darkhunter_pop.run_management import STAGE_REGISTRY

pytestmark = pytest.mark.api


def test_benchmarks_config_and_loaders_api_contract(tmp_path) -> None:
    cfg = load_config()
    validate_benchmarks_config(cfg)
    table = load_known_truth_table_from_config(cfg)
    assert {s.name for s in table.systems} == {"Gaia-BH1", "Gaia-BH2", "Gaia-BH3"}
    results = check_known_truth_expectations(
        table,
        synthetic_observed_from_truth(table),
        ruwe_match_tolerance=float(cfg.benchmarks.ruwe_match_tolerance),
    )
    assert all(r.passed for r in results)

    catalogs = load_all_comparison_catalogs(cfg)
    assert set(REQUIRED_COMPARISON_CATALOG_IDS).issubset(catalogs)

    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="api_bench")
    known = emit_known_truth_benchmarks(cfg, dirs)
    comps = emit_comparison_catalogs(cfg, dirs)
    assert known.reports and comps.reports

    spec = STAGE_REGISTRY["diagnostics"]
    assert "benchmarks" in spec.config_fingerprint_keys
    assert "darkhunter_pop.benchmarks" in spec.dependency_modules
