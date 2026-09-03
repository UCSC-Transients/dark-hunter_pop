"""Andrews et al. (2022) frozen selection files (CONTINUATION_PLAN §6 / §6.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionMode
from darkhunter_pop.sample_selection import (
    SampleSelectionRegistry,
    load_sample_selection_file,
    resolve_inherits,
)

pytestmark = pytest.mark.unit

_ANDREWS = repo_root() / "config" / "selections" / "andrews2022.yaml"
_ANDREWS_MODIFIED = repo_root() / "config" / "selections" / "andrews2022_modified.yaml"
_GAIA_BH1 = 4373465352415301632
_PARENT_N = 134598


def test_andrews2022_file_exists() -> None:
    assert _ANDREWS.is_file()
    assert _ANDREWS_MODIFIED.is_file()


def test_andrews2022_parent_is_orbital_only_and_verified() -> None:
    spec = load_sample_selection_file(_ANDREWS)
    assert spec.name == "andrews2022"
    assert spec.mode is SampleSelectionMode.REPRODUCTION
    assert spec.provenance.published_n == 24
    assert spec.primary_mass is not None
    assert spec.primary_mass.method == "fixed"
    assert spec.primary_mass.value_msun == 1.0
    assert spec.monte_carlo is not None
    assert spec.monte_carlo.n_draws == 10000
    assert spec.monte_carlo.covariance == "full_12x12"

    dr3 = spec.parent_query.dr3 if spec.parent_query is not None else None
    assert dr3 is not None
    assert dr3.expected_parent_n == _PARENT_N
    assert dr3.solution_types == ["Orbital"]
    assert dr3.nss_table == "gaiadr3.nss_two_body_orbit"
    assert "nss_solution_type = 'Orbital'" in dr3.adql
    assert "OrbitalAlternative" not in dr3.adql
    assert "AstroSpectroSB1" not in dr3.adql
    assert dr3.verification is not None
    assert dr3.verification.status == "verified"
    assert dr3.verification.confirmed_count == _PARENT_N
    confirming = dr3.verification.confirming_query
    assert confirming is not None
    assert "COUNT(*)" in confirming
    assert "nss_solution_type = 'Orbital'" in confirming

    dr4 = spec.parent_query.dr4
    assert dr4.nss_table == "gaiadr4.nss_two_body_orbit"
    assert dr4.solution_types == ["Orbital"]
    assert dr4.expected_parent_n is None
    assert "gaiadr4.nss_two_body_orbit" in dr4.adql
    assert dr3.adql != dr4.adql


def test_andrews2022_cut_chain_matches_section_6_4() -> None:
    spec = load_sample_selection_file(_ANDREWS)
    cuts = {cut.id: cut for cut in spec.cuts or []}
    assert list(cuts) == [
        "m2_probability",
        "goodness_of_fit",
        "m2_snr",
        "giant_reject_logg",
        "giant_reject_cmd",
    ]
    assert cuts["m2_probability"].kind.value == "probability"
    assert cuts["m2_probability"].expected_n_after == 106
    assert cuts["m2_probability"].parameters["m2_threshold_msun"] == 1.4
    assert cuts["m2_probability"].parameters["m2_probability_min"] == 0.95
    assert cuts["goodness_of_fit"].parameters["goodness_of_fit_max"] == 5.0
    assert cuts["m2_snr"].parameters["m2_snr_min"] == 3.0
    assert cuts["giant_reject_logg"].parameters["logg_dwarf_min"] == 3.6
    assert cuts["giant_reject_cmd"].parameters["cmd_slope"] == 3.14
    assert cuts["giant_reject_cmd"].parameters["cmd_intercept"] == -0.43
    assert cuts["giant_reject_cmd"].parameters["extinction_corrected"] is False
    assert spec.exclusions is not None
    assert len(spec.exclusions) == 1
    assert spec.exclusions[0].source_id == _GAIA_BH1
    assert spec.exclusions[0].expected_n_after == 24


def test_andrews2022_modified_inherits_and_restores_gaia_bh1() -> None:
    frozen = load_sample_selection_file(_ANDREWS)
    modified = load_sample_selection_file(_ANDREWS_MODIFIED)
    assert modified.name == "andrews2022_modified"
    assert modified.inherits == "andrews2022"
    assert modified.mode is SampleSelectionMode.FORWARD_MODEL
    assert modified.provenance.published_n is None
    assert modified.provenance.expected_n == 25
    assert modified.exclusions == []
    assert modified.correction is not None
    assert modified.correction.restores_source_id == _GAIA_BH1
    assert "Gaia BH1" in (modified.correction.identification or "")

    resolved = resolve_inherits(
        modified,
        files_by_name={"andrews2022": frozen, "andrews2022_modified": modified},
    )
    assert [c.id for c in resolved.cuts or []] == [c.id for c in (frozen.cuts or [])]
    assert resolved.exclusions == []
    assert frozen.exclusions
    assert frozen.exclusions[0].source_id == _GAIA_BH1
    assert resolved.parent_query is not None
    assert resolved.parent_query.dr3.expected_parent_n == _PARENT_N
    assert resolved.primary_mass is not None
    assert resolved.primary_mass.value_msun == 1.0
    assert resolved.monte_carlo is not None
    assert resolved.monte_carlo.n_draws == 10000


def test_andrews_variants_are_independent_registry_entries() -> None:
    cfg = load_config()
    registry = SampleSelectionRegistry(cfg, repo=repo_root())
    names = [entry.name for entry in cfg.sample_selection.samples]
    assert names.count("andrews2022") == 1
    assert names.count("andrews2022_modified") == 1
    frozen = registry.resolved("andrews2022")
    modified = registry.resolved("andrews2022_modified")
    assert frozen.mode is SampleSelectionMode.REPRODUCTION
    assert modified.mode is SampleSelectionMode.FORWARD_MODEL
    assert frozen.exclusions
    assert frozen.exclusions[0].source_id == _GAIA_BH1
    assert modified.exclusions == []
    assert registry.selection("andrews2022").parent_adql() == frozen.parent_query.dr3.adql
    assert all(cut.id != "harmonic_exclusion" for cut in (modified.cuts or []))
