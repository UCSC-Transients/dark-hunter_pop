"""El-Badry et al. (2026) two-branch selection (CONTINUATION_PLAN §8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionEntry, SampleSelectionMode
from darkhunter_pop.elbadry2026_selection import (
    classify_simon2026_row,
    is_main_sequence,
    load_simon2026_orbital,
    simon2026_exclusion_breakdown,
)
from darkhunter_pop.janssens_mass import invert_mg_to_mass, load_janssens_table
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    astrometric_mass_ratio_function,
    companion_mass_from_amrf,
    invert_astrometric_companion_mass,
)
from darkhunter_pop.sample_selection import (
    NotApplicable,
    SampleSelection,
    SampleSelectionRegistry,
    load_sample_selection_file,
)
from darkhunter_pop.schemas import ActiveDRMode

pytestmark = pytest.mark.unit

_E1_G_LT_15 = (
    3640889032890567040,
    4373465352415301632,
    6281177228434199296,
    5870569352746779008,
    3664684869697065984,
)
_SIMON_IN_SAMPLE = (
    5593444799901901696,
    3509370326763016704,
    3640889032890567040,
    6281177228434199296,
    6588211521163024640,
    6593763230249162112,
    1864406790238257536,
    2086448353089047808,
    4060365702574410752,
    6102598776102841344,
    5352109964757046528,
)


def _elbadry_spec():
    return load_sample_selection_file(
        repo_root() / "config/selections/elbadry2026.yaml"
    )


def test_file_loads_and_owns_two_branches() -> None:
    spec = _elbadry_spec()
    assert spec.name == "elbadry2026"
    assert spec.depends_on == ["andrews2022"]
    assert spec.inference_branches == ["astrometric"]
    assert spec.extinction is not None
    assert spec.extinction.south.map == "lallement2019"
    assert "-28.0" in spec.extinction.south.applies_when
    assert spec.extinction.north.map == "green2019"
    ids = [b.id for b in spec.branches or []]
    assert ids == ["astrometric", "spectroscopic"]
    astro = spec.branches[0]
    specb = spec.branches[1]
    assert astro.inference is True
    assert specb.inference is False
    assert astro.expected_union_n == 76
    assert specb.expected_union_n == 151
    assert spec.provenance.published_n == 227
    assert spec.validation_targets is not None
    assert spec.validation_targets.spurious_fraction.role == "acceptance_test"
    assert spec.inclusion_operator is not None
    assert spec.inclusion_operator.catalog_level_only is True
    q7 = next(item for item in spec.open_items if item.id == "Q7")
    assert q7.status == "escalated"
    assert astro.parent_query.dr3.nss_table != astro.parent_query.dr4.nss_table
    assert specb.parent_query.dr3.solution_types == ["SB1", "SB1C"]


def test_table_e1_is_not_trimmed_to_ids() -> None:
    raw = yaml.safe_load(
        (
            repo_root() / "config/selections/external/elbadry2023_table_e1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert len(raw["data"]) == 6
    for row in raw["data"]:
        assert "verdict" in row
        assert "g_mag" in row
        assert "period_days" in row


def test_janssens_solar_anchor_and_no_extrapolation() -> None:
    table = load_janssens_table()
    solar = invert_mg_to_mass(4.73, table=table)
    assert solar.mass_msun == pytest.approx(1.0, rel=1e-10)
    forbidden = invert_mg_to_mass(-9.0, table=table)
    assert forbidden.mass_msun is None
    assert forbidden.reason == "outside_janssens_range"


def test_amrf_is_cube_root_of_mf_over_m1() -> None:
    a0, plx, p, m1 = 2.0, 1.0, 365.25, 1.5
    mf = float(astrometric_mass_function(a0, plx, p))
    amrf = float(astrometric_mass_ratio_function(a0, plx, p, m1))
    assert amrf ** 3 == pytest.approx(mf / m1)
    m2 = float(companion_mass_from_amrf(amrf, m1))
    m2_from_mf = float(invert_astrometric_companion_mass(m1, mf, 0.0))
    assert m2 == pytest.approx(m2_from_mf)


def test_main_sequence_cut_from_yaml() -> None:
    spec = _elbadry_spec()
    cut = spec.main_sequence_cut
    assert cut is not None
    assert is_main_sequence(5.0, 0.8, cut)
    assert not is_main_sequence(0.0, 1.5, cut)


def _astro_row(source_id: int, **kwargs: object) -> dict:
    row = {
        "source_id": source_id,
        "nss_solution_type": "Orbital",
        "main_sequence": True,
        "m1_tilde_msun": 1.0,
        "m2_tilde_msun": 1.6,
        "goodness_of_fit": 1.0,
        "period_day": 100.0,
        "phot_g_mean_mag": 12.0,
        "sigma_m2_astrometric_msun": 0.05,
        "andrews_member": False,
    }
    row.update(kwargs)
    return row


def _sb1_row(source_id: int, **kwargs: object) -> dict:
    row = {
        "source_id": source_id,
        "nss_solution_type": "SB1",
        "k1_significance": 12.0,
        "fm_msun": 0.5,
        "main_sequence": True,
        "m2_min_msun": 1.5,
        "m1_tilde_msun": 1.0,
        "phot_g_mean_mag": 12.0,
        "andrews_member": False,
    }
    row.update(kwargs)
    return row


def _catalog_rows() -> list[dict]:
    rows: list[dict] = []
    sub1_ids = list(_E1_G_LT_15[:2]) + list(range(3, 48))
    andrews_in_sub1 = set(range(3, 15))
    for sid in sub1_ids:
        rows.append(
            _astro_row(
                sid,
                m2_tilde_msun=1.6,
                andrews_member=sid in andrews_in_sub1,
            )
        )
    for sid in _E1_G_LT_15[2:]:
        rows.append(
            _astro_row(
                sid,
                m2_tilde_msun=0.5,
                goodness_of_fit=20.0,
                phot_g_mean_mag=12.0,
                main_sequence=False,
            )
        )
    rows.append(
        _astro_row(
            4467000291193143808,
            phot_g_mean_mag=15.5,
            m2_tilde_msun=0.5,
            goodness_of_fit=20.0,
            main_sequence=False,
        )
    )
    for sid in (300, 301, 302, 303):
        rows.append(_astro_row(sid, m2_tilde_msun=0.5, goodness_of_fit=20.0, andrews_member=True))
    for sid in range(400, 408):
        rows.append(
            _astro_row(
                sid,
                phot_g_mean_mag=16.0,
                m2_tilde_msun=0.5,
                goodness_of_fit=20.0,
                andrews_member=True,
            )
        )
    for sid in range(200, 222):
        rows.append(
            _astro_row(
                sid,
                m2_tilde_msun=1.20,
                period_day=400.0,
                sigma_m2_astrometric_msun=0.05,
            )
        )
    for sid in range(1000, 1121):
        rows.append(_sb1_row(sid, fm_msun=0.5, m2_min_msun=1.5))
    for sid in range(2000, 2015):
        rows.append(_sb1_row(sid, fm_msun=4.0, m2_min_msun=1.5))
    for sid in range(3000, 3015):
        rows.append(
            _sb1_row(
                sid,
                fm_msun=4.0,
                main_sequence=False,
                m2_min_msun=NotApplicable("evolved"),
                m1_tilde_msun=NotApplicable("evolved"),
            )
        )
    return rows


def _registry() -> SampleSelectionRegistry:
    cfg = load_config().model_copy(deep=True)
    fixtures = Path(__file__).resolve().parent / "fixtures" / "selections"
    cfg.sample_selection.samples = [
        SampleSelectionEntry(
            name="andrews2022",
            enabled=True,
            path=str(fixtures / "andrews2022_stub.yaml"),
            mode=SampleSelectionMode.REPRODUCTION,
        ),
        SampleSelectionEntry(
            name="elbadry2026",
            enabled=True,
            path="config/selections/elbadry2026.yaml",
            mode=SampleSelectionMode.REPRODUCTION,
        ),
    ]
    return SampleSelectionRegistry(cfg)


def test_two_branch_union_and_inference_gate() -> None:
    results = _registry().evaluate_all(_catalog_rows())
    eb = results["elbadry2026"]
    astro = set(eb.branch_surviving["astrometric"])
    spec = set(eb.branch_surviving["spectroscopic"])
    assert len(eb.subsample_surviving["primary_ns_bh"]) == 47
    assert len(eb.subsample_surviving["elbadry2023_table_e1"]) == 5
    assert len(eb.subsample_surviving["andrews2022_import"]) == 16
    assert len(eb.subsample_surviving["sub_chandrasekhar"]) == 22
    assert len(astro) == 76
    assert len(spec) == 151
    assert eb.n_surviving == 227
    assert set(eb.surviving_source_ids) == astro | spec
    assert set(eb.inference_source_ids) == astro
    assert spec.isdisjoint(set(eb.inference_source_ids))
    assert eb.route_counts["main_sequence_min_companion_mass"] == 136
    assert eb.route_counts["high_mass_function"] == 30
    assert eb.route_counts["both"] == 15


def test_q9_andrews_g_lt_15_is_exactly_16() -> None:
    results = _registry().evaluate_all(_catalog_rows())
    andrews = results["andrews2022"].surviving_source_ids
    assert len(andrews) == 24
    imported = results["elbadry2026"].subsample_surviving["andrews2022_import"]
    assert len(imported) == 16
    faint = [sid for sid in andrews if sid in range(400, 408)]
    assert len(faint) == 8
    assert all(sid not in imported for sid in faint)


def test_validation_targets_are_not_cut_inputs() -> None:
    spec = _elbadry_spec()
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    assert spec.validation_targets is not None
    rate = spec.validation_targets.spurious_fraction.astrometric
    rows = [_astro_row(1)]
    before = selection.evaluate(rows).n_surviving
    spec.validation_targets.spurious_fraction.astrometric = 0.99
    after = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION).evaluate(rows)
    assert after.n_surviving == before
    assert rate == 0.40


def test_m1_dependent_cut_is_not_applicable_for_evolved() -> None:
    spec = _elbadry_spec()
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    row = _astro_row(
        99,
        main_sequence=True,
        m1_tilde_msun=NotApplicable("outside_janssens_range"),
        m2_tilde_msun=NotApplicable("outside_janssens_range"),
        nss_solution_type="Orbital",
    )
    result = selection.evaluate([row])
    outcomes = result.outcomes_by_source[99]
    m2_floor = [o for o in outcomes if o[0].endswith("m2_floor")]
    assert m2_floor
    assert m2_floor[0][1].value == "not_applicable"


def test_parent_adql_default_is_inference_branch() -> None:
    spec = _elbadry_spec()
    from darkhunter_pop.sample_selection import parent_query_for_mode

    dr3 = parent_query_for_mode(spec, ActiveDRMode.DR3)
    assert "IN ('Orbital', 'AstroSpectroSB1')" in dr3.adql
    assert "IN ('SB1', 'SB1C')" not in dr3.adql
    sb1 = parent_query_for_mode(spec, ActiveDRMode.DR3, branch_id="spectroscopic")
    assert "SB1" in sb1.adql
    assert sb1.nss_table.startswith("gaiadr3")
    sb1_dr4 = parent_query_for_mode(spec, ActiveDRMode.DR4, branch_id="spectroscopic")
    assert sb1_dr4.nss_table.startswith("gaiadr4")


def test_simon2026_exclusion_breakdown_5_2_1_1() -> None:
    spec = _elbadry_spec()
    rows = load_simon2026_orbital()
    assert len(rows) == 20
    counts = simon2026_exclusion_breakdown(rows, _SIMON_IN_SAMPLE, spec)
    expected = spec.acceptance_tests.simon2026_exclusion_breakdown
    assert expected is not None
    assert counts["sb1_fails_significance"] == expected.sb1_fails_significance
    assert counts["astrometric_f2_above_max"] == expected.astrometric_f2_above_max
    assert counts["fainter_than_g_limit"] == expected.fainter_than_g_limit
    assert counts["fails_m2_over_m1"] == expected.fails_m2_over_m1
    assert counts["in_sample"] == 11
    assert counts["unclassified"] == 0
    astro = next(b for b in spec.branches or [] if b.id == "astrometric")
    specb = next(b for b in spec.branches or [] if b.id == "spectroscopic")
    reasons = [
        classify_simon2026_row(
            row,
            in_sample=int(row["source_id"]) in _SIMON_IN_SAMPLE,
            g_mag_faint_limit=15.0,
            goodness_of_fit_max=10.0,
            k1_significance_min=10.0,
            m2_over_m1_min=1.2,
            astrometric_types=astro.parent_query.dr3.solution_types,
            spectroscopic_types=specb.parent_query.dr3.solution_types,
        )
        for row in rows
    ]
    assert reasons.count("sb1_fails_significance") == 5
