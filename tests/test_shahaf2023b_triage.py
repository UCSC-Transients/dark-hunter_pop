"""Shahaf et al. (2023b) AMRF class-III triage reimplementation (#682)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from darkhunter_pop import shahaf2023b_triage as tri
from darkhunter_pop.mock_atf_columns import MOCK_ORBIT_NAMES
from darkhunter_pop.mock_selection_rows import mock_draws_to_selection_rows
from darkhunter_pop.physics_utils import photocenter_a0_from_thiele_innes
from darkhunter_pop.sample_selection import NotApplicable
from darkhunter_pop.schemas import ParameterSet

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def definition() -> tri.TriageDefinition:
    return tri.load_definition()


@pytest.fixture(scope="module")
def small(definition: tri.TriageDefinition) -> tri.TriageDefinition:
    """Same definition with 4,000 draws (unit-test speed; the decision cap n_below ≤ 16 is unchanged)."""
    mc = definition.monte_carlo.model_copy(update={"n_draws": 4000, "chunk_draws": 1000})
    return definition.model_copy(update={"monte_carlo": mc})


def test_definition_is_the_paper(definition: tri.TriageDefinition) -> None:
    par = definition.parent
    assert par.solution_types == ["Orbital", "AstroSpectroSB1"]
    assert par.bit_index == [8191, 65535]
    assert par.binary_masses_m1_ref == "IsocLum"
    assert (par.eccentricity_error_ln_period_slope, par.eccentricity_error_offset) == (0.079, -0.244)
    assert (par.parallax_snr_period_coeff, par.significance_period_coeff) == (20000.0, 158.0)
    assert par.ti_relative_error_sq_sum_max == 36.0 and par.period_max_day == 1000.0
    assert par.expected_counts["clean"] == 101380
    assert definition.monte_carlo.n_draws == 100000
    assert definition.decision.alpha == 0.10 and definition.decision.fixed_n_below_max == 16
    m1 = definition.curve_m1
    assert m1.size == 91 and m1[0] == pytest.approx(0.2) and m1[-1] == pytest.approx(2.0)
    assert np.allclose(np.diff(m1), 0.02, atol=1e-6)
    assert np.all(definition.curve_a_tr > definition.curve_a_ms)
    assert len(tri.definition_sha256()) == 64


def _cut_columns(**over: Any) -> dict[str, np.ndarray]:
    base: dict[str, Any] = {
        "nss_solution_type": "Orbital", "bit_index": 8191, "period": 500.0, "eccentricity_error": 0.05,
        "parallax": 2.0, "parallax_error": 0.02, "significance": 50.0,
        "binary_masses_m1_ref": "IsocLum", "binary_masses_combination_method": "AstroSpectroSB1+M1",
    }
    for x in "abfg":
        base[f"{x}_thiele_innes"] = 1.0
        base[f"{x}_thiele_innes_error"] = 0.1
    base.update(over)
    return {k: np.asarray([v]) for k, v in base.items()}


@pytest.mark.parametrize(
    ("over", "failed"),
    [
        ({}, None),
        ({"nss_solution_type": "SB1"}, "solution_type"),
        ({"bit_index": 8189}, "bit_index"),
        ({"binary_masses_m1_ref": ""}, "binary_masses"),
        ({"binary_masses_combination_method": "SB2+M1"}, "binary_masses"),
        ({"eccentricity_error": 0.079 * np.log(500.0) - 0.244}, "eccentricity_error"),  # strict <
        ({"parallax_error": 2.0 / 40.0}, "parallax_snr"),  # ϖ/σ = 20000/P exactly fails (strict >)
        ({"significance": 158.0 / np.sqrt(500.0)}, "significance"),
        ({"a_thiele_innes_error": 5.92}, None),  # Eq. 4: Σ = 35.05 + 0.03 ≤ 36 passes
        ({"a_thiele_innes_error": 6.0}, "thiele_innes_relative_error"),  # Σ = 36.03 > 36
        ({"period": 1000.0}, "period"),  # strict <
    ],
)
def test_parent_cuts_follow_section_3_1(definition: tri.TriageDefinition, over: dict[str, Any], failed: str | None) -> None:
    masks = tri.parent_cut_masks(_cut_columns(**over), definition)
    assert tuple(masks) == tri.PARENT_CUT_IDS
    failing = [k for k, m in masks.items() if not m[0]]
    if failed is None:
        assert failing == []
    else:
        assert failed in failing


def test_parent_cuts_without_binary_masses_or_bit_index(definition: tri.TriageDefinition) -> None:
    cols = _cut_columns()
    del cols["bit_index"], cols["binary_masses_m1_ref"], cols["binary_masses_combination_method"]
    masks = tri.parent_cut_masks(cols, definition, binary_masses=False)
    assert "binary_masses" not in masks and "bit_index" not in masks
    assert bool(tri.parent_mask(masks)[0])


def test_amrf_is_equation_1(definition: tri.TriageDefinition) -> None:
    a, b, f, g, plx, p, m1 = 1.9, 0.3, -0.3, 1.9, 2.0, 500.0, 0.9
    a0 = float(photocenter_a0_from_thiele_innes(a, b, f, g))
    expected = a0 / plx * m1 ** (-1.0 / 3.0) * (p / 365.25) ** (-2.0 / 3.0)
    assert float(tri.amrf(a, b, f, g, plx, p, m1, definition)) == pytest.approx(expected, rel=1e-12)


def test_limiting_values_and_off_grid(definition: tri.TriageDefinition) -> None:
    ms, tr = tri.limiting_values(np.array([0.21, 0.1, 3.0]), definition)
    rows = np.asarray(definition.limiting_curves.rows)
    assert tr[0] == pytest.approx(0.5 * (rows[0, 2] + rows[1, 2]), rel=1e-6)
    # clip_m1 (adopted): the end values off the 0.2–2.0 grid
    assert tr[1] == rows[0, 2] and tr[2] == rows[-1, 2]
    nc = definition.model_copy(update={"monte_carlo": definition.monte_carlo.model_copy(update={"off_grid": "not_classified"})})
    ms2, tr2 = tri.limiting_values(np.array([0.1, 3.0]), nc)
    assert np.all(np.isnan(tr2)) and np.all(np.isnan(ms2))


def _orbit(a0_scale: float, sigma: float = 1e-3, ecc: float = 0.2, ecc_sigma: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([2.0, 1.9 * a0_scale, 0.3 * a0_scale, -0.3 * a0_scale, 1.9 * a0_scale, 500.0, ecc])
    cov = np.diag([sigma, sigma, sigma, sigma, sigma, 0.1, ecc_sigma]) ** 2
    return mean, cov


def test_class_probabilities_extremes_and_estimator(small: tri.TriageDefinition) -> None:
    hi = tri.class_probabilities(*_orbit(1.0), 0.9, 0.85, 0.95, source_id=1, definition=small)  # A ≈ 0.81 > A_TR ≈ 0.59
    lo = tri.class_probabilities(*_orbit(0.2), 0.9, 0.85, 0.95, source_id=1, definition=small)  # A ≈ 0.16 < A_MS
    assert hi.complete and hi.n_draws == 4000 and hi.n_class3 == 4000 and hi.n_below == 0
    assert hi.p_iii == pytest.approx(4001 / 4001)
    assert lo.n_class3 == 0 and lo.n_class2 == 0 and lo.p_iii == pytest.approx(1 / 4001)
    assert hi.p_value("n_below_plus_one") == pytest.approx(1 / 4001)
    assert hi.p_value("one_minus_pr_iii") == pytest.approx(0.0)


def test_class_probabilities_deterministic_and_chunk_invariant(definition: tri.TriageDefinition) -> None:
    mean, cov = _orbit(0.55, sigma=0.08, ecc=0.05, ecc_sigma=0.1)  # straddles A_TR; ~30 % of e draws rejected
    runs = []
    for chunk in (700, 5000):
        d = definition.model_copy(update={"monte_carlo": definition.monte_carlo.model_copy(update={"n_draws": 5000, "chunk_draws": chunk})})
        runs.append(tri.class_probabilities(mean, cov, 0.9, 0.8, 0.95, source_id=7, definition=d))
    assert runs[0] == runs[1]
    assert runs[0].n_draws == 5000 and 0 < runs[0].n_class3 < 5000
    other = tri.class_probabilities(mean, cov, 0.9, 0.8, 0.95, source_id=8, definition=d)
    assert other != runs[1]


def test_early_stop_never_passes_fixed_threshold(small: tri.TriageDefinition) -> None:
    mean, cov = _orbit(0.55, sigma=0.08)
    c = tri.class_probabilities(mean, cov, 0.9, 0.8, 0.95, source_id=3, definition=small, n_below_stop=16)
    assert not c.complete and c.n_below > 16 and c.n_draws < small.monte_carlo.n_draws
    assert tri.fixed_threshold_class3(c, small) is False


def test_benjamini_hochberg_and_p_value_reading(definition: tri.TriageDefinition) -> None:
    assert tri.benjamini_hochberg([0.01, 0.04, 0.03, 0.5], 0.1).tolist() == [True, True, True, False]
    assert tri.benjamini_hochberg([np.nan, 0.001], 0.1).tolist() == [False, True]
    # The documented reading (#682): with m = 101,380 clean rows, 177 rows at n_below ≤ 16 and two at 17,
    # the n_below_plus_one p-value keeps exactly the 177 (Table 2); 1 − Pr III would keep 179.
    n = definition.monte_carlo.n_draws
    counts = ([tri.ClassCounts(n, 0, n - 16, True)] * 177 + [tri.ClassCounts(n, 0, n - 17, True)] * 2
              + [tri.ClassCounts(n, 0, 0, True)] * (101380 - 179))
    for convention, expected in (("n_below_plus_one", 177), ("one_minus_pr_iii", 179)):
        p = np.array([c.p_value(convention) for c in counts])
        assert int(tri.benjamini_hochberg(p, definition.decision.alpha).sum()) == expected


def _mock_inputs(a_scale: float = 1.0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from darkhunter_pop.mock_selection_rows import CASCADE_COLUMNS

    n = 3
    a, b, f, g = 1.9 * a_scale, 0.3 * a_scale, -0.3 * a_scale, 1.9 * a_scale
    a0 = float(photocenter_a0_from_thiele_innes(a, b, f, g))
    cascade = np.zeros((n, len(CASCADE_COLUMNS)))
    for name, val in {
        "plx": 2.0, "sig_parallax": 0.02, "A": a, "sig_A": 0.01, "B": b, "sig_B": 0.01, "F": f, "sig_F": 0.01,
        "G": g, "sig_G": 0.01, "period": 500.0, "sig_period": 1.0, "ecc": 0.2, "sig_ecc": 0.02,
        "a0_mas": a0, "sigma_a0_mas": 0.01, "N_visibility_periods": 20.0, "N_obs": 300.0, "F2": 1.0, "ruwe": 1.0,
    }.items():
        cascade[:, CASCADE_COLUMNS.index(name)] = val
    values = dict(zip(MOCK_ORBIT_NAMES, [0.0] * len(MOCK_ORBIT_NAMES)))
    values.update({"parallax": 2.0, "period": 500.0, "eccentricity": 0.2, "a_thiele_innes": a,
                   "b_thiele_innes": b, "f_thiele_innes": f, "g_thiele_innes": g})
    sig = {"parallax": 0.02, "period": 1.0, "eccentricity": 0.02}
    cov = np.diag([sig.get(k, 0.01) ** 2 for k in MOCK_ORBIT_NAMES])
    truth = {
        "draw_index": np.array([11, 12, 13]), "source_id": np.array([5, 6, 7]) + 4_000_000_000_000_000_000,
        "ra_deg": np.full(n, 10.0), "dec_deg": np.full(n, -5.0), "phot_g_mean_mag": np.full(n, 14.0),
        "sys_bp_rp": np.full(n, 1.5), "m1_row_msun": np.full(n, 0.6), "m1_msun": np.full(n, 0.6),
        "m2_msun": np.full(n, 0.6), "is_dark": np.ones(n, bool), "period_days": np.full(n, 500.0),
        "sys_evolved": np.zeros(n, bool), "parent_row": np.zeros(n, np.int64),
    }
    outcome = {
        "cascade": cascade, "accepted_orbital": np.ones(n, bool),
        "nss_solution_type": np.array(["Orbital", "AstroSpectroSB1", "Orbital"]),
        "orbit_p0": np.tile([values[k] for k in MOCK_ORBIT_NAMES], (n, 1)), "orbit_cov": np.tile(cov, (n, 1, 1)),
        "orbit_cov_status": np.array(["replayed", "replayed", "prefilter_excluded:lc1"]),
    }
    return truth, outcome


def test_mock_adapter_emits_class3_columns(small: tri.TriageDefinition) -> None:
    definition = small
    truth, outcome = _mock_inputs(1.0)
    m1 = (np.full(3, 0.9), np.full(3, 0.05))
    with pytest.raises(ValueError, match="include_orbit"):
        mock_draws_to_selection_rows(truth, outcome, pipeline_m1=m1, shahaf_triage=definition)
    rows = mock_draws_to_selection_rows(truth, outcome, pipeline_m1=m1, include_orbit=True, shahaf_triage=definition)
    a0 = float(photocenter_a0_from_thiele_innes(1.9, 0.3, -0.3, 1.9))
    assert rows[0]["mock_significance_a0"] == pytest.approx(a0 / 0.01)
    assert "significance" not in rows[0]  # the evaluator would alias it to the SB1 k1_significance
    for row in rows[:2]:  # Orbital and AstroSpectroSB1 both enter the S23 parent
        assert row[tri.COLUMN_CLASS3] is True
        assert row[tri.COLUMN_N_BELOW] == 0 and row[tri.COLUMN_N_DRAWS] == definition.monte_carlo.n_draws
        assert row[tri.COLUMN_M1_SOURCE] == tri.MOCK_M1_SOURCE
        assert row[tri.COLUMN_PARENT_STATUS] == tri.MOCK_PARENT_STATUS
        assert row[tri.COLUMN_AMRF] > 0.75  # ≈ 0.81, above A_TR(0.9) ≈ 0.59
    excluded = rows[2][tri.COLUMN_CLASS3]
    assert isinstance(excluded, NotApplicable) and excluded.reason == "mock_prefilter_excluded:lc1"


def test_mock_adapter_luminous_like_orbit_is_not_class3(small: tri.TriageDefinition) -> None:
    definition = small
    truth, outcome = _mock_inputs(0.2)  # A ≈ 0.22: a single MS companion fits
    rows = mock_draws_to_selection_rows(
        truth, outcome, pipeline_m1=(np.full(3, 0.9), np.full(3, 0.05)), include_orbit=True, shahaf_triage=definition
    )
    assert rows[0][tri.COLUMN_CLASS3] is False
    assert rows[0][tri.COLUMN_N_DRAWS] < definition.monte_carlo.n_draws  # early stop: certainly above 16 below


def test_attach_marks_missing_inputs_not_applicable(definition: tri.TriageDefinition) -> None:
    ps = ParameterSet(names=list(tri.ORBIT_PARAMETERS), values=[2.0, 1.9, 0.3, -0.3, 1.9, 500.0, 0.2],
                      covariance=(np.eye(7) * 1e-6).tolist(), provenance="test")
    base = {k: v[0].item() for k, v in _cut_columns().items()}
    base.update({"source_id": 1, "pipeline_m1_msun": float("nan"), "pipeline_m1_sigma_msun": 0.05, "orb": ps})
    out = tri.attach_class3_columns([base, {"source_id": 2}], definition, orbit_key="orb", m1_key="pipeline_m1_msun",
                                    m1_sigma_key="pipeline_m1_sigma_msun", m1_source="x", parent_status="y")
    assert out[0][tri.COLUMN_CLASS3].reason == "shahaf2023b_no_m1"
    assert out[1][tri.COLUMN_CLASS3].reason == "shahaf2023b_missing_parent_columns"
