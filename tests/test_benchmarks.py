"""Tests for known-truth benchmarks + comparison catalogs (issue #70)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from darkhunter_pop.benchmarks import (
    EXTERNAL_MF_CATALOG_IDS,
    REQUIRED_COMPARISON_CATALOG_IDS,
    ObservedBenchmark,
    assert_comparison_only,
    assert_required_catalogs_present,
    check_known_truth_expectations,
    format_comparison_catalog_report,
    format_known_truth_report,
    load_all_comparison_catalogs,
    load_comparison_catalog,
    load_known_truth_table,
    load_known_truth_table_from_config,
    synthetic_observed_from_truth,
    validate_benchmarks_config,
)
from darkhunter_pop.config_loader import config_checksum, load_config
from darkhunter_pop.config_schema import BenchmarkCatalogEntry, SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.diagnostics import (
    emit_comparison_catalogs,
    emit_known_truth_benchmarks,
    list_diagnostic_helpers,
    resolve_diagnostic_dirs,
    run_diagnostics_scaffolding,
)
from darkhunter_pop.run_management import STAGE_REGISTRY

pytestmark = pytest.mark.unit


def test_config_loads_benchmarks_fragment() -> None:
    cfg = load_config()
    validate_benchmarks_config(cfg)
    assert cfg.benchmarks.known_truth_path.endswith("known_truth_gaia_bh.yaml")
    assert cfg.benchmarks.ruwe_match_tolerance == pytest.approx(0.25)
    for catalog_id in REQUIRED_COMPARISON_CATALOG_IDS:
        assert catalog_id in cfg.benchmarks.catalogs
        entry = cfg.benchmarks.catalogs[catalog_id]
        assert entry.role == "comparison_only"
        assert entry.never_as_prior is True
    assert cfg.diagnostics.hooks.known_truth_benchmarks is True
    assert cfg.diagnostics.hooks.comparison_catalogs is True


def test_checksum_excludes_benchmarks() -> None:
    assert "benchmarks" not in SHARED_CHECKSUM_SECTIONS
    assert "diagnostics" not in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    altered.benchmarks.ruwe_match_tolerance = cfg.benchmarks.ruwe_match_tolerance + 0.1
    assert config_checksum(altered) == base


def test_registry_fingerprints_benchmarks() -> None:
    spec = STAGE_REGISTRY["diagnostics"]
    assert "benchmarks" in spec.config_fingerprint_keys
    assert "darkhunter_pop.benchmarks" in spec.dependency_modules


def test_known_truth_table_provenance_and_expectations() -> None:
    cfg = load_config()
    table = load_known_truth_table_from_config(cfg)
    assert table.table_id == "known_truth_gaia_bh"
    assert "ARCHITECTURE.md" in str(table.provenance.get("project_authority", ""))
    by_name = table.by_name()
    assert by_name["Gaia-BH1"].dr3_expectation == "clean_detection"
    assert by_name["Gaia-BH1"].source_id == 4373465352415301632
    assert by_name["Gaia-BH2"].dr3_expectation == "clean_detection"
    assert by_name["Gaia-BH2"].source_id == 5870569352746779008
    bh3 = by_name["Gaia-BH3"]
    assert bh3.dr3_expectation == "marginal_or_non_detection"
    assert bh3.source_id == 4318465066420528000
    assert bh3.ruwe_approx == pytest.approx(3.4)
    assert bh3.acceleration_catalog_parts_of_orbit is True
    assert bh3.nss_orbital_expected is False


def test_known_truth_checks_pass_on_synthetic_observed() -> None:
    cfg = load_config()
    table = load_known_truth_table_from_config(cfg)
    observed = synthetic_observed_from_truth(table)
    results = check_known_truth_expectations(
        table,
        observed,
        ruwe_match_tolerance=float(cfg.benchmarks.ruwe_match_tolerance),
    )
    assert len(results) == 3
    assert all(r.passed for r in results)
    report = format_known_truth_report(
        table,
        results,
        ruwe_match_tolerance=float(cfg.benchmarks.ruwe_match_tolerance),
    )
    assert "Gaia-BH3" in report
    assert "RUWE" in report or "ruwe" in report.lower()
    assert "acceleration catalog" in report


def test_known_truth_checks_fail_when_bh3_clean_detected() -> None:
    cfg = load_config()
    table = load_known_truth_table_from_config(cfg)
    observed = synthetic_observed_from_truth(table)
    bh3_id = table.by_name()["Gaia-BH3"].source_id
    observed[bh3_id] = ObservedBenchmark(
        source_id=bh3_id,
        in_nss_orbital=True,
        ruwe=3.4,
    )
    results = check_known_truth_expectations(
        table,
        observed,
        ruwe_match_tolerance=float(cfg.benchmarks.ruwe_match_tolerance),
    )
    bh3 = next(r for r in results if r.name == "Gaia-BH3")
    assert bh3.passed is False
    assert "unexpected NSS" in bh3.details


def test_known_truth_checks_fail_on_bad_bh3_ruwe() -> None:
    cfg = load_config()
    table = load_known_truth_table_from_config(cfg)
    observed = synthetic_observed_from_truth(table, bh3_ruwe=1.0)
    results = check_known_truth_expectations(
        table,
        observed,
        ruwe_match_tolerance=float(cfg.benchmarks.ruwe_match_tolerance),
    )
    bh3 = next(r for r in results if r.name == "Gaia-BH3")
    assert bh3.passed is False
    assert "RUWE" in bh3.details


def test_comparison_catalogs_load_and_are_never_priors() -> None:
    cfg = load_config()
    catalogs = load_all_comparison_catalogs(cfg)
    assert_required_catalogs_present(catalogs)
    for catalog_id, catalog in catalogs.items():
        assert_comparison_only(catalog)
        assert catalog.never_as_prior is True
        assert catalog.role == "comparison_only"
        assert catalog.provenance
    for mf_id in EXTERNAL_MF_CATALOG_IDS:
        assert catalogs[mf_id].is_mass_function
        assert len(catalogs[mf_id].mass_msun) >= 2
    report = format_comparison_catalog_report(catalogs)
    assert "never inference priors" in report
    assert "pulsar_mf" in report
    assert "ligo_bh_mf" in report
    assert "ns_candidate_21" in report
    assert "companions_156" in report
    assert "amrf_binary_masses" in report
    assert "andrews" in report
    assert "shahaf" in report


def test_schema_rejects_prior_capable_catalog_entry() -> None:
    with pytest.raises(ValidationError):
        BenchmarkCatalogEntry.model_validate(
            {
                "path": "config/benchmarks/catalogs/pulsar_mf.yaml",
                "role": "comparison_only",
                "never_as_prior": False,
            }
        )


def test_emit_hooks_write_benchmark_reports(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="bench")
    known = emit_known_truth_benchmarks(cfg, dirs)
    assert known.skipped_reason is None
    assert known.reports and known.reports[0].is_file()
    text = known.reports[0].read_text(encoding="utf-8")
    assert "Gaia-BH1" in text
    assert "marginal_or_non_detection" in text

    comps = emit_comparison_catalogs(cfg, dirs)
    assert comps.skipped_reason is None
    assert comps.reports and comps.reports[0].is_file()
    ctext = comps.reports[0].read_text(encoding="utf-8")
    assert "comparison-only" in ctext
    assert "forbidden as population prior" in ctext


def test_scaffolding_includes_benchmark_hooks(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    result = run_diagnostics_scaffolding(cfg, run_id="scaffold_bench", demo_hooks=True)
    hook_names = [h.hook_name for h in result.hooks_run]
    assert "known_truth_benchmarks" in hook_names
    assert "comparison_catalogs" in hook_names
    assert "emit_known_truth_benchmarks" in list_diagnostic_helpers()
    assert "emit_comparison_catalogs" in list_diagnostic_helpers()
    assert "benchmarks" in result.config_snapshot
    notes = result.as_dict()["notes"]
    assert "issue #70" in notes or "known-truth" in notes
    assert "issue #71" in notes or "diagnostic suite" in notes


def test_load_known_truth_from_explicit_path() -> None:
    cfg = load_config()
    table = load_known_truth_table(cfg.benchmarks.known_truth_path)
    assert len(table.systems) == 3


def test_load_single_comparison_catalog() -> None:
    cfg = load_config()
    path = cfg.benchmarks.catalogs["andrews"].path
    catalog = load_comparison_catalog(path)
    assert catalog.catalog_id == "andrews"
    assert len(catalog.systems) >= 1
