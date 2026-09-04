"""El-Badry et al. (2024) frozen selection (CONTINUATION_PLAN §7 / §4.8)."""

from __future__ import annotations

import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionMode
from darkhunter_pop.elbadry2024_selection import (
    g_mag_faint_limit_from_spec,
    inclusion_criteria_ids,
    load_elbadry2024_spec,
    load_elbadry2024_table3,
    published_sample_source_ids,
    spurious_fraction_g_lt_15,
    table3_g_lt_15_rows,
)
from darkhunter_pop.sample_selection import (
    SampleSelection,
    SampleSelectionRegistry,
    parent_query_for_mode,
)
from darkhunter_pop.schemas import ActiveDRMode

pytestmark = pytest.mark.unit

_PARENT_N = 168065
_PUBLISHED_N = 21
_G_LT_15_N = 48
_SPURIOUS_G_LT_15 = 12


def test_file_loads_parent_mass_extinction_and_inclusion() -> None:
    spec = load_elbadry2024_spec()
    assert spec.name == "elbadry2024"
    assert spec.mode is SampleSelectionMode.REPRODUCTION
    assert spec.depends_on == ["andrews2022"]
    assert spec.provenance.published_n == _PUBLISHED_N
    assert spec.primary_mass is not None
    assert spec.primary_mass.method == "isoclum_binary_masses"
    assert spec.primary_mass.table == "gaiadr3.binary_masses"
    assert spec.extinction is not None
    assert spec.extinction.north.map == "green2019"
    assert spec.extinction.south.map == "lallement2022"
    assert "-30.0" in spec.extinction.north.applies_when
    assert "-30.0" in spec.extinction.south.applies_when
    assert "lallement2019" not in spec.extinction.south.map
    assert "-28" not in spec.extinction.north.applies_when
    assert spec.monte_carlo is not None
    assert spec.monte_carlo.n_draws == 10000
    assert spec.inclusion_operator is not None
    assert spec.inclusion_operator.catalog_level_only is True
    assert inclusion_criteria_ids(spec) == ("(b)", "(c)", "(d)")
    assert spec.inclusion_operator.astrometric_followup_outcomes[
        "published_sample"
    ] == _PUBLISHED_N
    q2 = next(item for item in spec.open_items if item.id == "Q2")
    assert q2.status == "resolved"


def test_parent_is_orbital_plus_astrospectrosb1_and_verified() -> None:
    spec = load_elbadry2024_spec()
    dr3 = parent_query_for_mode(spec, ActiveDRMode.DR3)
    assert dr3.expected_parent_n == _PARENT_N
    assert dr3.solution_types == ["Orbital", "AstroSpectroSB1"]
    assert "IN ('Orbital', 'AstroSpectroSB1')" in dr3.adql
    assert "OrbitalAlternative" not in dr3.adql
    assert dr3.verification is not None
    assert dr3.verification.status == "verified"
    assert dr3.verification.confirmed_count == _PARENT_N
    confirming = dr3.verification.confirming_query
    assert confirming is not None
    assert "COUNT(*)" in confirming
    assert "AstroSpectroSB1" in confirming

    dr4 = parent_query_for_mode(spec, ActiveDRMode.DR4)
    assert dr4.nss_table == "gaiadr4.nss_two_body_orbit"
    assert dr4.solution_types == ["Orbital", "AstroSpectroSB1"]
    assert dr4.expected_parent_n is None
    assert dr3.adql != dr4.adql


def test_catalog_cuts_are_pool_and_g_only() -> None:
    spec = load_elbadry2024_spec()
    assert spec.branches is not None
    branch = spec.branches[0]
    assert branch.id == "astrometric"
    assert branch.expected_union_n == _G_LT_15_N
    assert branch.subsamples is not None
    assert len(branch.subsamples) == 1
    sub = branch.subsamples[0]
    assert sub.id == "elbadry2024_table3"
    assert sub.external_table == (
        "config/selections/external/elbadry2024_table3.yaml"
    )
    cut_ids = [cut.id for cut in sub.cuts]
    assert cut_ids == ["ns_candidate_pool", "g_mag_limit"]
    assert g_mag_faint_limit_from_spec(spec) == 15.0
    # Outcome-dependent criteria must not appear as catalog cut ids.
    assert "m2_joint_fit" not in cut_ids
    assert "not_spurious" not in cut_ids
    assert "orbit_coverage" not in cut_ids


def test_validation_target_is_exact_25_percent_and_unread() -> None:
    spec = load_elbadry2024_spec()
    assert spec.validation_targets is not None
    frac = spec.validation_targets.spurious_fraction
    assert frac is not None
    assert frac.role == "acceptance_test"
    assert frac.astrometric == 0.25

    g_lim = g_mag_faint_limit_from_spec(spec)
    rate, n_spurious, n_bright = spurious_fraction_g_lt_15(
        g_mag_faint_limit=g_lim
    )
    assert n_spurious == _SPURIOUS_G_LT_15
    assert n_bright == _G_LT_15_N
    assert rate == pytest.approx(0.25)
    assert rate == frac.astrometric

    rows = [
        {
            "source_id": int(row["source_id"]),
            "nss_solution_type": "Orbital",
            "phot_g_mean_mag": float(row["g_mag"]),
            "period_day": float(row["period_days"]),
        }
        for row in load_elbadry2024_table3()
    ]
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    before = selection.evaluate(rows).n_surviving
    assert before == _G_LT_15_N
    # Mutating the validation target must not change catalog-level survivors.
    assert spec.validation_targets.spurious_fraction is not None
    spec.validation_targets.spurious_fraction.astrometric = 0.99
    after = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION).evaluate(
        rows
    )
    assert after.n_surviving == before == _G_LT_15_N


def test_published_sample_is_21_compact_object_candidates() -> None:
    ids = published_sample_source_ids()
    assert len(ids) == _PUBLISHED_N
    assert len(set(ids)) == _PUBLISHED_N
    g_lim = g_mag_faint_limit_from_spec(load_elbadry2024_spec())
    bright_ids = {
        int(row["source_id"])
        for row in table3_g_lt_15_rows(g_mag_faint_limit=g_lim)
    }
    assert set(ids).issubset(bright_ids)


def test_registry_enables_elbadry2024() -> None:
    cfg = load_config()
    entry = next(e for e in cfg.sample_selection.samples if e.name == "elbadry2024")
    assert entry.enabled is True
    assert entry.path == "config/selections/elbadry2024.yaml"
    registry = SampleSelectionRegistry(cfg, repo=repo_root())
    resolved = registry.resolved("elbadry2024")
    assert resolved.provenance.published_n == _PUBLISHED_N
    assert registry.selection("elbadry2024").parent_adql()
