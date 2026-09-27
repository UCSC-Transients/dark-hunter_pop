"""El-Badry et al. (2026) two-branch selection (CONTINUATION_PLAN §8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionEntry, SampleSelectionMode
from darkhunter_pop.elbadry2026_selection import (
    EBV_COLUMN,
    classify_simon2026_row,
    deredden_elbadry2026_rows,
    enrich_elbadry2026_row,
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
_TABLE7_INPUTS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "selections"
    / "elbadry2026_table7_dr3_inputs.yaml"
)
_SIMON_M2_RATIO_SOURCE = 1864406790238257536


def _external_ids(name: str) -> set[int]:
    path = repo_root() / "config/selections/external" / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {int(row["source_id"]) for row in raw["data"]}


def _simon_in_sample() -> tuple[int, ...]:
    """Simon sources El-Badry 2026 published as selected (Table 7 ∪ Table 8, #281)."""
    published = _external_ids("elbadry2026_table7.yaml") | _external_ids(
        "elbadry2026_table8.yaml"
    )
    simon = [int(row["source_id"]) for row in load_simon2026_orbital()]
    return tuple(sid for sid in simon if sid in published)


def _table7_inputs() -> dict[int, dict]:
    raw = yaml.safe_load(_TABLE7_INPUTS.read_text(encoding="utf-8"))
    return {int(row["source_id"]): dict(row) for row in raw["sources"]}


def _enrich_with_ebv(inputs: dict, ebv: float) -> dict:
    """Run the El-Badry 2026 point-estimate chain on real DR3 inputs at fixed E(B-V)."""
    spec = _elbadry_spec()
    row = {**inputs, EBV_COLUMN: float(ebv)}
    return enrich_elbadry2026_row(deredden_elbadry2026_rows([row], spec, None)[0], spec)


def _elbadry_ratio_for_1864() -> dict[int, float]:
    inputs = _table7_inputs()[_SIMON_M2_RATIO_SOURCE]
    enriched = _enrich_with_ebv(inputs, inputs["ebv_map_elbadry2026"])
    return {
        _SIMON_M2_RATIO_SOURCE: float(enriched["m2_tilde_msun"])
        / float(enriched["m1_tilde_msun"])
    }


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
        assert "state" in row
        assert "g_mag" in row
        assert "period_days" in row


def test_janssens_solar_anchor_and_no_extrapolation() -> None:
    table = load_janssens_table()
    solar = invert_mg_to_mass(4.73, table=table)
    assert solar.mass_msun == pytest.approx(1.0, rel=1e-10)
    forbidden = invert_mg_to_mass(-9.0, table=table)
    assert forbidden.mass_msun is None
    assert forbidden.reason == "outside_janssens_range"


def test_janssens_table_is_immutable_shared_cache() -> None:
    """Regression #152: load_janssens_table is @lru_cache'd; every caller shares

    the same object. A mutable dict/list return would let one caller's
    mutation poison global state for the rest of the process. Prove the
    returned structure is read-only at every level a caller could reach:
    the top-level mapping, a nested segment mapping, and a nested list
    (which must come back as a tuple, since MappingProxyType only wraps
    dicts).
    """
    table = load_janssens_table()

    with pytest.raises(TypeError):
        table["segments"] = ()  # type: ignore[index]

    segments = table["segments"]
    assert isinstance(segments, tuple)
    first_segment = segments[0]
    with pytest.raises(TypeError):
        first_segment["a"] = 999.0  # type: ignore[index]

    mass_range = table["mass_range_msun"]
    assert isinstance(mass_range, tuple)

    # Repeated calls (cache hit) return the identical shared object.
    assert load_janssens_table() is table


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


def test_simon2026_in_sample_matches_published_tables() -> None:
    """Table 7 lists 3263804373319076480 and omits 1864406790238257536 (#281)."""
    in_sample = _simon_in_sample()
    assert len(in_sample) == 11
    assert 3263804373319076480 in in_sample
    assert _SIMON_M2_RATIO_SOURCE not in in_sample


def test_simon2026_exclusion_breakdown_5_2_1_1() -> None:
    spec = _elbadry_spec()
    rows = load_simon2026_orbital()
    assert len(rows) == 20
    in_sample = _simon_in_sample()
    ratios = _elbadry_ratio_for_1864()
    counts = simon2026_exclusion_breakdown(
        rows, in_sample, spec, elbadry_m2_over_m1_by_source=ratios
    )
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
    reasons = {
        int(row["source_id"]): classify_simon2026_row(
            row,
            in_sample=int(row["source_id"]) in in_sample,
            g_mag_faint_limit=15.0,
            goodness_of_fit_max=10.0,
            k1_significance_min=10.0,
            m2_over_m1_min=1.2,
            astrometric_types=astro.parent_query.dr3.solution_types,
            spectroscopic_types=specb.parent_query.dr3.solution_types,
            elbadry_m2_over_m1=ratios.get(int(row["source_id"])),
        )
        for row in rows
    }
    assert list(reasons.values()).count("sb1_fails_significance") == 5
    assert reasons[_SIMON_M2_RATIO_SOURCE] == "fails_m2_over_m1"


def test_simon2026_breakdown_falls_back_to_catalog_ratio() -> None:
    """Without El-Badry's M̃2/M̃1, Simon's catalog ratio (5.08) cannot bucket 1864… (#281)."""
    spec = _elbadry_spec()
    counts = simon2026_exclusion_breakdown(
        load_simon2026_orbital(), _simon_in_sample(), spec
    )
    assert counts["fails_m2_over_m1"] == 0
    assert counts["unclassified"] == 1


@pytest.mark.physics
def test_point_chain_reproduces_elbadry2026_table7() -> None:
    """Real DR3 inputs + the paper's E(B-V) reproduce Table 7's M̃1 / M̃2 / AMRF (#275).

    Covers every Table 7 row that is main-sequence on our CMD cut (73 of 76). Measured
    residuals at 2470fe5: median |Δ| 0.06 % (M̃1), 0.03 % (M̃2), 0.03 % (AMRF); max 1.7 %,
    0.8 %, 0.5 %, all on 2032579979951732736, consistent with E(B-V) rounding in the table.
    """
    inputs = _table7_inputs()
    table = yaml.safe_load(
        (repo_root() / "config/selections/external/elbadry2026_table7.yaml").read_text(
            encoding="utf-8"
        )
    )["data"]
    n_checked = 0
    for pub in table:
        sid = int(pub["source_id"])
        enriched = _enrich_with_ebv(inputs[sid], pub["ebv_mag"])
        m1 = enriched["m1_tilde_msun"]
        if isinstance(m1, NotApplicable):
            # Subsample 2/3 members need no main-sequence cut; Table 7 still quotes a
            # photometric M̃1 for them. They never enter the M̃2 subsample chains.
            assert enriched["main_sequence"] is False
            continue
        n_checked += 1
        assert m1 == pytest.approx(pub["m1_msun"], rel=0.02), sid
        assert enriched["m2_tilde_msun"] == pytest.approx(pub["m2_msun"], rel=0.01), sid
        assert enriched["amrf"] == pytest.approx(pub["amrf"], rel=0.01), sid
    assert n_checked == 73


@pytest.mark.physics
def test_simon_3263804373319076480_is_a_table7_member_on_our_chain() -> None:
    """The source #200 attributed to the a0/AMRF chain matches El-Badry's own values (#281)."""
    inputs = _table7_inputs()[3263804373319076480]
    enriched = _enrich_with_ebv(inputs, 0.08)
    assert enriched["amrf"] == pytest.approx(1.09, abs=0.005)
    assert enriched["m1_tilde_msun"] == pytest.approx(1.16, abs=0.01)
    assert enriched["m2_tilde_msun"] == pytest.approx(2.925, abs=0.01)
    assert enriched["m2_tilde_msun"] / enriched["m1_tilde_msun"] > 1.2
    ratio = _elbadry_ratio_for_1864()[_SIMON_M2_RATIO_SOURCE]
    assert ratio < 1.2
