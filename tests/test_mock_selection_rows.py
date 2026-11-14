"""Contract: mock draws -> real-column rows -> frozen cut chains run unchanged (#678, RUNG5_SPEC §2.7.4).

Synthetic draw arrays (no data/, no HDF5). The dust maps are replaced by a constant-E(B-V) stub that
implements the ``ExtinctionLookup.ebv_for`` contract; the dereddening code path is the real one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.mock_selection_rows import (
    CASCADE_COLUMNS,
    CUT_EVALUABLE,
    CUT_EXTERNAL_MEMBERSHIP,
    CUT_NOT_APPLICABLE,
    CUT_PARTIAL,
    CUT_UNBOUND_COLUMN,
    MOCK_SOURCE_ID_MAX,
    MissingSolutionTypeError,
    REAL_COLUMNS,
    classify_cuts,
    forward_model_selection,
    mock_draws_to_selection_rows,
    rows_columns,
)
from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes
from darkhunter_pop.sample_selection import SampleSelectionRegistry

pytestmark = pytest.mark.unit

_EBV = 0.05
_N = 7
_IDS = np.array([101_000_000 + k for k in range(_N)], dtype=np.int64)


class _ConstantEbv:
    """``ExtinctionLookup`` stand-in: same ``ebv_for`` / ``dust_cfg`` contract, constant E(B-V), status OK."""

    def __init__(self, dust_cfg: Any) -> None:
        self.dust_cfg = dust_cfg

    def ebv_for(self, ids: Any, ra: Any, dec: Any, plx: Any) -> tuple[Any, Any, Any]:
        n = len(ids)
        return np.full(n, _EBV), np.zeros(n, dtype=int), np.zeros(n, dtype=int)


def _draws() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Seven draws. 0 passes primary_ns_bh; 1 has no orbit; 2 bad GoF; 3 long P; 4 faint; 5 giant; 6 passes, shares a parent with 0."""
    plx = np.full(_N, 2.0)
    period = np.array([500.0, 500.0, 500.0, 2000.0, 500.0, 500.0, 500.0])
    # Kepler: a0 ~ P^(2/3) keeps M2 ~ 3 Msun at 3 mas for P = 500 d; A=a0, G=a0, B=F=0 gives photocenter a0.
    a0 = 3.0 * (period / 500.0) ** (2.0 / 3.0)
    f2 = np.array([3.0, 3.0, 50.0, 3.0, 3.0, 3.0, 3.0])
    g = np.array([13.0, 13.0, 13.0, 13.0, 16.0, 8.0, 13.0])
    cascade = np.zeros((_N, len(CASCADE_COLUMNS)))
    for name, val in {
        "plx": plx, "sig_parallax": 0.02, "A": a0, "sig_A": 0.01, "B": 0.0, "sig_B": 0.01,
        "F": 0.0, "sig_F": 0.01, "G": a0, "sig_G": 0.01, "period": period, "sig_period": 5.0,
        "ecc": 0.1, "sig_ecc": 0.02, "a0_mas": a0, "sigma_a0_mas": 0.05,
        "N_visibility_periods": 20.0, "N_obs": 300.0, "F2": f2, "ruwe": 1.5,
    }.items():
        cascade[:, CASCADE_COLUMNS.index(name)] = val
    accepted = np.array([True, False, True, True, True, True, True])
    truth = {
        "draw_index": _IDS,
        "source_id": np.array([4_000_000_000_000_000_000 + (k % 6) for k in range(_N)], dtype=np.int64),
        "ra_deg": np.full(_N, 150.0), "dec_deg": np.full(_N, 20.0),
        "phot_g_mean_mag": g, "sys_bp_rp": np.full(_N, 0.85),
        "m1_row_msun": np.full(_N, 0.9), "m1_msun": np.full(_N, 0.9), "m2_msun": np.full(_N, 3.0),
        "is_dark": np.ones(_N, dtype=bool), "co_component": np.ones(_N, dtype=np.int8),
        "period_days": period, "sys_evolved": np.array([False] * 5 + [True, False]),
    }
    nss = np.array(["Orbital", "", "Orbital", "Orbital", "Orbital", "Orbital", "AstroSpectroSB1"])  # draw 6 is SB1-routed
    outcome = {
        "cascade": cascade, "accepted_orbital": accepted, "nss_solution_type": nss,
        "orbit_p0": np.zeros((_N, 12)), "orbit_cov": np.tile(np.eye(12), (_N, 1, 1)),
    }
    truth["parent_row"] = np.zeros(_N, dtype=np.int64)
    return truth, outcome


def _kw() -> dict[str, Any]:
    return {"pipeline_m1": (np.full(_N, 0.9), np.full(_N, 0.05))}


@pytest.fixture(scope="module")
def registry() -> SampleSelectionRegistry:
    return SampleSelectionRegistry(load_config(), repo=repo_root())


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return mock_draws_to_selection_rows(*_draws(), **_kw())


def _selection(registry: SampleSelectionRegistry, name: str) -> Any:
    spec = registry.resolved(name)
    lookup = _ConstantEbv(registry.cfg.dust_maps) if spec.extinction is not None else None
    return forward_model_selection(registry, name, extinction_lookup=lookup)


def test_rows_use_real_column_names_and_synthetic_ids(rows: list[dict[str, Any]]) -> None:
    assert len(rows) == 6  # draw 1 has no published orbit
    assert set(REAL_COLUMNS) <= rows_columns(rows)
    ids = [r["source_id"] for r in rows]
    assert len(set(ids)) == len(ids) and max(ids) < MOCK_SOURCE_ID_MAX
    assert [r["nss_solution_type"] for r in rows] == ["Orbital"] * 4 + ["Orbital", "AstroSpectroSB1"]
    assert len({r["mock_parent_source_id"] for r in rows}) < len(rows)  # parent ids repeat, source_id does not
    r0 = rows[0]
    assert r0["period_day"] == r0["period"] == 500.0 and r0["goodness_of_fit"] == 3.0
    assert r0["abs_g_mag"] == pytest.approx(13.0 + 5.0 * np.log10(2.0) - 10.0)
    assert r0["pipeline_m1_msun"] == 0.9 and r0["pipeline_m1_sigma_msun"] == 0.05 and r0["mock_m1_msun"] == 0.9
    assert photocenter_a0_from_thiele_innes(
        r0["a_thiele_innes"], r0["b_thiele_innes"], r0["f_thiele_innes"], r0["g_thiele_innes"]
    ) == pytest.approx(3.0)


def test_orbit_arrays_only_on_request() -> None:
    plain = mock_draws_to_selection_rows(*_draws(), **_kw())
    full = mock_draws_to_selection_rows(*_draws(), include_orbit=True, **_kw())
    assert "mock_orbit_cov" not in rows_columns(plain)
    assert full[0]["mock_orbit_cov"].shape == (12, 12) and full[0]["mock_orbit_p0"].shape == (12,)


def test_bad_inputs_raise() -> None:
    truth, outcome = _draws()
    with pytest.raises(ValueError, match="cascade"):
        mock_draws_to_selection_rows(truth, {**outcome, "cascade": outcome["cascade"][:, :5]}, **_kw())
    with pytest.raises(ValueError, match="MOCK_SOURCE_ID_MAX"):
        mock_draws_to_selection_rows({**truth, "draw_index": truth["draw_index"] + 5_000_000_000}, outcome, **_kw())
    with pytest.raises(ValueError, match="not unique"):
        mock_draws_to_selection_rows({**truth, "draw_index": np.zeros(_N, dtype=np.int64)}, outcome, **_kw())


def test_frozen_chains_run_unchanged_on_mock_rows(
    registry: SampleSelectionRegistry, rows: list[dict[str, Any]]
) -> None:
    """The unmodified evaluator runs every forward-model chain; primary_ns_bh keeps exactly the designed rows."""
    for name in ("andrews2022_modified", "elbadry2024", "elbadry2026"):
        result = _selection(registry, name).evaluate(rows)
        assert result.n_parent == (len(rows) - 1 if name.startswith("andrews") else len(rows))  # Andrews: no SB1
        if name == "elbadry2026":
            assert result.surviving_source_ids == (int(_IDS[0]), int(_IDS[6]))
            by_cut = {a.cut_id: a for a in result.attrition}
            assert by_cut["astrometric:primary_ns_bh:main_sequence"].n_not_applicable == 0
            # giant draw 5 is off the main sequence: a failed MS cut, not N/A
            assert by_cut["astrometric:primary_ns_bh:main_sequence"].n_failed == 1
            assert by_cut["astrometric:primary_ns_bh:goodness_of_fit"].n_failed == 1  # draw 2
            assert by_cut["astrometric:primary_ns_bh:period"].n_failed == 1  # draw 3
            assert by_cut["astrometric:primary_ns_bh:g_mag"].n_failed == 1  # draw 4
        else:
            assert result.n_surviving == 0  # ATF sidecar (#679) / external Table 3 membership: no mock path yet


def test_cuts_evaluable_today(registry: SampleSelectionRegistry, rows: list[dict[str, Any]]) -> None:
    """Pin which cuts run on mock rows now. A ticket that unblocks one (#679-#682) updates this table."""
    expected: dict[str, dict[str, str]] = {
        "andrews2022_modified": {
            "atf_covariance_valid": CUT_NOT_APPLICABLE,  # #679
            "atf_m2_probability": CUT_NOT_APPLICABLE,  # #679
            "atf_goodness_of_fit": CUT_UNBOUND_COLUMN,  # #679
            "atf_giant_reject_logg": CUT_UNBOUND_COLUMN,  # #679, #680
            "atf_giant_reject_cmd": CUT_UNBOUND_COLUMN,  # #679
            "atf_m2_3sigma": CUT_NOT_APPLICABLE,  # #679
        },
        "elbadry2024": {
            "astrometric:elbadry2024_table3:ns_candidate_pool": CUT_EXTERNAL_MEMBERSHIP,  # #682
            "astrometric:elbadry2024_table3:g_mag_limit": CUT_EVALUABLE,
        },
        "elbadry2026": {
            "astrometric:primary_ns_bh:main_sequence": CUT_EVALUABLE,
            "astrometric:primary_ns_bh:m2_floor": CUT_PARTIAL,  # evolved draw: M1-tilde undefined (Q10)
            "astrometric:primary_ns_bh:m2_over_m1": CUT_PARTIAL,
            "astrometric:primary_ns_bh:goodness_of_fit": CUT_EVALUABLE,
            "astrometric:primary_ns_bh:period": CUT_EVALUABLE,
            "astrometric:primary_ns_bh:g_mag": CUT_EVALUABLE,
            "astrometric:elbadry2023_table_e1:table_e1_membership": CUT_EXTERNAL_MEMBERSHIP,  # no SF (N)
            "astrometric:andrews2022_import:andrews2022_membership": CUT_NOT_APPLICABLE,  # #679
            "astrometric:sub_chandrasekhar:m2_error": CUT_NOT_APPLICABLE,  # #681
            "spectroscopic:k1_significance": CUT_NOT_APPLICABLE,  # SB1 not in mock (Q6)
        },
    }
    for name, cuts in expected.items():
        report = classify_cuts(_selection(registry, name), rows)
        got = {k: report[k]["category"] for k in cuts}
        assert got == cuts, name


def test_label_is_required_and_never_assumed() -> None:
    truth, outcome = _draws()
    no_label = {k: v for k, v in outcome.items() if k != "nss_solution_type"}
    with pytest.raises(MissingSolutionTypeError, match="recover_nss_type_690"):
        mock_draws_to_selection_rows(truth, no_label, **_kw())
    unlabelled = {**outcome, "nss_solution_type": np.array([""] * _N)}
    with pytest.raises(ValueError, match="no Orbital / AstroSpectroSB1 label"):
        mock_draws_to_selection_rows(truth, unlabelled, **_kw())


def test_pipeline_m1_must_be_given_and_is_tag10_from_the_parent_row() -> None:
    from darkhunter_pop.mass_derivation import pipeline_tag10_m1

    truth, outcome = _draws()
    with pytest.raises(ValueError, match="TAG10"):
        mock_draws_to_selection_rows(truth, outcome)
    cfg = load_config()
    atm = {"teff_gspphot": np.array([5800.0]), "logg_gspphot": np.array([4.4]), "mh_gspphot": np.array([0.0])}
    rows = mock_draws_to_selection_rows(truth, outcome, parent_columns=atm, config=cfg)
    m1, sig = pipeline_tag10_m1({k: float(v[0]) for k, v in atm.items()}, cfg)
    assert np.isfinite(m1) and rows[0]["pipeline_m1_msun"] == m1 and rows[0]["pipeline_m1_sigma_msun"] == sig
    assert rows[0]["pipeline_m1_msun"] != rows[0]["mock_m1_row_msun"]  # not the proposal's MIST M1


def test_parent_filters_split_orbital_and_sb1(registry: SampleSelectionRegistry, rows: list[dict[str, Any]]) -> None:
    """El-Badry parents include AstroSpectroSB1 mock rows; Andrews (Orbital only) excludes them."""
    sb1_id = int(_IDS[6])
    assert any(r["source_id"] == sb1_id and r["nss_solution_type"] == "AstroSpectroSB1" for r in rows)
    andrews = _selection(registry, "andrews2022_modified").evaluate(rows)
    assert andrews.n_parent == len(rows) - 1  # the SB1-routed row is not in Andrews' parent
    assert sb1_id not in andrews.outcomes_by_source
    for name in ("elbadry2024", "elbadry2026"):
        res = _selection(registry, name).evaluate(rows)
        assert res.n_parent == len(rows)  # branched samples count every row; the astrometric parent takes both types
        assert sb1_id in res.outcomes_by_source
    elb = _selection(registry, "elbadry2026").evaluate(rows)
    assert sb1_id in elb.surviving_source_ids  # the SB1-routed draw passes primary_ns_bh like draw 0
