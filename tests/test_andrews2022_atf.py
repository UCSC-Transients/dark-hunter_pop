"""Andrews et al. (2022) ATF-notebook reproduction procedure (#296).

The notebook (``data/reference/andrews2022_ATF_sample_selection.ipynb``,
gitignored) is the spec; these tests pin its semantics — covariance built
without ``bit_index`` and gated by SciPy (no flooring), any-draw root failure
rejects, strict ``> 0.95`` over all draws, Lick > FLAME > uniform refined M1,
logg cut skipped for Lick sources, exact CMD line — and that forward_model
evaluation never sees the reproduction columns.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import brentq

from darkhunter_pop.andrews2022_atf import (
    COV_FAIL_CORR_VEC_SHORT,
    COV_FAIL_NONFINITE,
    COV_FAIL_SINGULAR,
    M1_SOURCE_FLAME,
    M1_SOURCE_LICK,
    M1_SOURCE_UNIFORM,
    AtfProcedureError,
    AtfSourceInputs,
    build_notebook_covariance,
    gate_covariance,
    m2_threshold_msun,
    merge_reproduction_columns,
    notebook_cmd_quantities,
    notebook_float,
    procedure_fingerprint,
    read_sidecar,
    resolve_refined_m1,
    run_pass1,
    run_pass2,
    solve_m2_brentq,
    solve_m2_fixed_m1,
    write_sidecar,
)
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionMode
from darkhunter_pop.sample_selection import (
    SampleSelection,
    SampleSelectionRegistry,
    load_sample_selection_file,
)

pytestmark = pytest.mark.unit

_GAIA_BH1 = 4373465352415301632
_LICK = {
    3649963989549165440: 0.47,
    1581117310088807552: 0.70,
    1947292821452944896: 0.73,
    1350295047363872512: 1.44,
    1525829295599805184: 0.64,
    4373465352415301632: 0.98,
    1854241667792418304: 0.70,
    1749013354127453696: 1.00,
}


@pytest.fixture(scope="module")
def registry() -> SampleSelectionRegistry:
    return SampleSelectionRegistry(load_config(), repo=repo_root())


@pytest.fixture(scope="module")
def spec(registry: SampleSelectionRegistry) -> Any:
    return registry.resolved("andrews2022")


def _notebook_loop_covariance(errs: np.ndarray, corr_vec: np.ndarray) -> np.ndarray:
    """Literal transcription of notebook ``get_random_samples`` (cell 7)."""
    size = errs.size
    corr = np.ones((size, size))
    n = 0
    for i in range(size):
        for j in range(size):
            if j >= i:
                continue
            corr[i, j] = corr_vec[n] * errs[i] * errs[j]
            corr[j, i] = corr_vec[n] * errs[i] * errs[j]
            n += 1
    for i in range(size):
        corr[i, i] = errs[i] ** 2
    return corr


def _inputs(
    *,
    source_id: int = 1,
    parallax: float = 5.0,
    parallax_error: float = 0.05,
    a_mas: float = 3.0,
    ti_error: float = 0.05,
    period: float = 500.0,
    corr_vec: np.ndarray | None = None,
    mass_flame: float = float("nan"),
    logg: float = float("nan"),
) -> AtfSourceInputs:
    # Order: ra, dec, parallax, pmra, pmdec, A, B, F, G, e, P, t_peri.
    means = np.array(
        [10.0, 20.0, parallax, 1.0, 2.0, a_mas, 0.3, -0.2, a_mas, 0.3, period, 50.0]
    )
    errors = np.array(
        [0.1, 0.1, parallax_error, 0.1, 0.1, ti_error, ti_error, ti_error, ti_error,
         0.02, 5.0, 10.0]
    )
    return AtfSourceInputs(
        source_id=source_id,
        means=means,
        errors=errors,
        corr_vec=np.zeros(66) if corr_vec is None else corr_vec,
        goodness_of_fit=1.0,
        mass_flame=mass_flame,
        logg=logg,
        g_mag=14.0,
        bp_mag=14.5,
        rp_mag=13.7,
    )


def test_covariance_matches_notebook_loop_order() -> None:
    rng = np.random.default_rng(3)
    errs = rng.uniform(0.01, 2.0, 12)
    corr_vec = rng.uniform(-0.3, 0.3, 66)
    np.testing.assert_array_equal(
        build_notebook_covariance(errs, corr_vec), _notebook_loop_covariance(errs, corr_vec)
    )


def test_short_corr_vec_is_a_covariance_failure(spec: Any) -> None:
    with pytest.raises(AtfProcedureError):
        build_notebook_covariance(np.ones(12), np.zeros(10))
    out = run_pass1(
        _inputs(corr_vec=np.zeros(10)),
        spec.reproduction_procedure,
        m2_threshold=m2_threshold_msun(spec),
    )
    assert out["andrews_atf_covariance_ok"] is False
    assert out["andrews_atf_covariance_failure"] == COV_FAIL_CORR_VEC_SHORT


def test_gate_rejects_nan_and_singular_without_flooring() -> None:
    means = np.zeros(3)
    ok, failure = gate_covariance(means, np.eye(3))
    assert ok is not None and failure is None
    nan_cov = np.eye(3)
    nan_cov[1, 1] = np.nan
    assert gate_covariance(means, nan_cov) == (None, COV_FAIL_NONFINITE)
    # Perfectly correlated pair: Cholesky with a nugget would sample it; the
    # notebook's scipy constructor refuses it.
    singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dist, failure = gate_covariance(means, singular)
    assert dist is None
    assert failure == COV_FAIL_SINGULAR


def test_vectorized_root_matches_brentq_success_set_and_values() -> None:
    rng = np.random.default_rng(11)
    mf = np.concatenate(
        [rng.uniform(-2.0, 1200.0, 400), [0.0, -1e-9, 998.002996004994, 998.1, np.nan, np.inf]]
    )
    m2, ok = solve_m2_fixed_m1(mf, m1_msun=1.0, bracket_msun=(0.0, 1000.0))

    def func(x: float, target: float) -> float:
        return x**3 / (1.0 + x) ** 2 - target

    for value, got, good in zip(mf, m2, ok, strict=True):
        try:
            expected = brentq(func, 0.0, 1000.0, args=(float(value),))
        except ValueError:
            assert not good, value
            continue
        assert good, value
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_brentq_pass2_solver_flags_out_of_bracket_draws() -> None:
    m2, ok = solve_m2_brentq(
        np.array([0.5, -0.1, 2.0e4]), np.array([0.8, 0.8, 0.8]), bracket_msun=(0.0, 1.0e4)
    )
    assert ok.tolist() == [True, False, False]
    assert m2[0] ** 3 / (0.8 + m2[0]) ** 2 == pytest.approx(0.5)


def test_pass1_rejects_source_with_any_negative_parallax_draw(spec: Any) -> None:
    proc = spec.reproduction_procedure
    thr = m2_threshold_msun(spec)
    # Parallax S/N 2.5: ~0.6% of 10^4 draws are negative -> brentq fails.
    weak = run_pass1(
        _inputs(parallax=0.5, parallax_error=0.2, a_mas=3.0), proc, m2_threshold=thr
    )
    assert weak["andrews_atf_covariance_ok"] is True
    assert weak["andrews_atf_pass1_n_negative_parallax"] > 0
    assert weak["andrews_atf_pass1_n_mf_below_bracket"] > 0
    assert weak["andrews_atf_pass1_root_ok"] is False
    # All-draws denominator is never above the valid-draws one.
    assert weak["andrews_atf_p_m2_above"] <= weak["andrews_atf_p_m2_above_valid_draws"]
    strong = run_pass1(_inputs(), proc, m2_threshold=thr)
    assert strong["andrews_atf_pass1_root_ok"] is True
    assert strong["andrews_atf_pass1_n_root_failed"] == 0
    assert strong["andrews_atf_p_m2_above"] == strong["andrews_atf_p_m2_above_valid_draws"]


def test_pass1_is_seeded_per_source(spec: Any) -> None:
    proc = spec.reproduction_procedure
    thr = m2_threshold_msun(spec)
    item = _inputs(parallax=2.0, parallax_error=0.3, a_mas=2.2)
    assert run_pass1(item, proc, m2_threshold=thr) == run_pass1(item, proc, m2_threshold=thr)


def test_refined_m1_priority_lick_then_flame_then_uniform(spec: Any) -> None:
    pm = spec.reproduction_procedure.pass2.primary_mass
    rng = np.random.default_rng(0)
    label, draws = resolve_refined_m1(1350295047363872512, 0.5, pm, n_draws=20000, rng=rng)
    assert label == M1_SOURCE_LICK
    assert float(np.mean(draws)) == pytest.approx(1.44, abs=0.005)
    assert float(np.std(draws)) == pytest.approx(0.1, abs=0.005)
    label, draws = resolve_refined_m1(7, 1.8, pm, n_draws=20000, rng=rng)
    assert label == M1_SOURCE_FLAME
    assert float(np.mean(draws)) == pytest.approx(1.8, abs=0.005)
    label, draws = resolve_refined_m1(7, float("nan"), pm, n_draws=20000, rng=rng)
    assert label == M1_SOURCE_UNIFORM
    assert draws.min() >= 0.63 and draws.max() <= 1.0


def test_pass2_columns(spec: Any) -> None:
    out = run_pass2(_inputs(mass_flame=0.9), spec.reproduction_procedure)
    assert out["andrews_atf_m1_source"] == M1_SOURCE_FLAME
    assert out["andrews_atf_pass2_root_ok"] is True
    assert out["andrews_atf_m2_mean_msun"] > 3.0 * out["andrews_atf_m2_std_msun"]


def test_notebook_cmd_quantities_and_float32_roundtrip() -> None:
    g_abs, color = notebook_cmd_quantities(14.0, 14.5, 13.7, 10.0)
    assert g_abs == pytest.approx(14.0 - 5.0 * math.log10(10.0))
    assert color == pytest.approx(0.8)
    g_neg, c_neg = notebook_cmd_quantities(14.0, 14.5, 13.7, -1.0)
    assert math.isnan(g_neg) and math.isnan(c_neg)
    raw = float(np.float32(0.1925101))
    assert notebook_float(raw, float32_decimal_roundtrip=True) == 0.1925101
    assert notebook_float(raw, float32_decimal_roundtrip=False) == raw
    assert math.isnan(notebook_float(None, float32_decimal_roundtrip=True))


def test_andrews2022_v3_frozen_file_carries_the_notebook(spec: Any) -> None:
    raw = load_sample_selection_file(repo_root() / "config/selections/andrews2022.yaml")
    assert raw.schema_version == 3
    proc = raw.reproduction_procedure
    assert proc is not None and proc.method == "atf_notebook"
    assert "data/reference/andrews2022_ATF_sample_selection.ipynb" in proc.source
    assert proc.covariance.use_bit_index is False
    assert proc.covariance.parameter_order[2] == "parallax"
    assert proc.pass1.m1_msun == 1.0
    assert proc.pass1.root_bracket_msun == (0.0, 1000.0)
    assert proc.pass1.probability_denominator == "all_draws"
    assert proc.pass1.reject_source_on_any_draw_failure is True
    assert proc.pass2.root_bracket_msun == (0.0, 10000.0)
    assert proc.pass2.giant_logg_column == "logg_gspphot"
    pm = proc.pass2.primary_mass
    assert {r.source_id: r.m1_msun for r in pm.lick_spectroscopic_masses} == _LICK
    assert (pm.lick_sigma_msun, pm.flame_sigma_msun) == (0.1, 0.1)
    assert (pm.uniform_low_msun, pm.uniform_high_msun) == (0.63, 1.0)
    cuts = {c.id: c for c in raw.cuts or []}
    repro = [c.id for c in raw.cuts or [] if c.applies_to == [SampleSelectionMode.REPRODUCTION]]
    assert repro == [
        "atf_covariance_valid",
        "atf_pass1_root_found",
        "atf_m2_probability",
        "atf_goodness_of_fit",
        "atf_giant_reject_logg",
        "atf_giant_reject_cmd",
        "atf_pass2_root_found",
        "atf_m2_3sigma",
    ]
    prob = cuts["atf_m2_probability"]
    assert prob.expression == "P(M2 > m2_threshold_msun) > m2_probability_min"
    assert prob.expected_n_after == 106
    cmd = cuts["atf_giant_reject_cmd"].parameters
    slope = (float(cmd["cmd_mag_2"]) - float(cmd["cmd_mag_1"])) / (
        float(cmd["cmd_color_2"]) - float(cmd["cmd_color_1"])
    )
    assert slope == (9 + 2) / (3 + 0.5)
    assert float(cmd["cmd_mag_2"]) - float(cmd["cmd_color_2"]) * slope == 9 - 3 * slope
    # schema_version-2 chain kept verbatim, forward_model only.
    fm = [c.id for c in raw.cuts or [] if c.applies_to == [SampleSelectionMode.FORWARD_MODEL]]
    assert fm == [
        "m2_probability",
        "goodness_of_fit",
        "m2_snr",
        "giant_reject_logg",
        "giant_reject_cmd",
    ]
    assert cuts["giant_reject_cmd"].parameters["cmd_slope"] == 3.14
    assert cuts["m2_probability"].expression == "P(M2 > m2_threshold_msun) >= m2_probability_min"


def test_modified_inherits_procedure(registry: SampleSelectionRegistry, spec: Any) -> None:
    modified = registry.resolved("andrews2022_modified")
    assert modified.schema_version == 2
    assert modified.reproduction_procedure == spec.reproduction_procedure
    assert procedure_fingerprint(modified) == procedure_fingerprint(spec)
    assert modified.exclusions == []


def _atf_row(source_id: int, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_id": source_id,
        "nss_solution_type": "Orbital",
        "andrews_atf_covariance_ok": True,
        "andrews_atf_pass1_root_ok": True,
        "andrews_atf_p_m2_above": 0.99,
        "andrews_atf_goodness_of_fit": 1.0,
        "andrews_atf_m1_source": M1_SOURCE_FLAME,
        "andrews_atf_logg": 4.5,
        "andrews_atf_bp_rp": 1.0,
        "andrews_atf_abs_g_mag": 6.0,
        "andrews_atf_pass2_root_ok": True,
        "andrews_atf_m2_mean_msun": 1.6,
        "andrews_atf_m2_std_msun": 0.2,
    }
    row.update(overrides)
    return row


def test_reproduction_chain_semantics_on_synthetic_rows(spec: Any) -> None:
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    rows = [
        _atf_row(1),
        _atf_row(2, andrews_atf_p_m2_above=0.95),  # strict > 0.95 fails
        _atf_row(3, andrews_atf_pass1_root_ok=False),
        _atf_row(4, andrews_atf_covariance_ok=False),
        _atf_row(5, andrews_atf_goodness_of_fit=5.0),  # > 5 rejects; 5.0 passes
        _atf_row(6, andrews_atf_goodness_of_fit=5.01),
        _atf_row(7, andrews_atf_goodness_of_fit=None),  # NaN passes
        _atf_row(8, andrews_atf_logg=3.5),  # Apsis giant
        _atf_row(9, andrews_atf_logg=3.5, andrews_atf_m1_source=M1_SOURCE_LICK),
        _atf_row(10, andrews_atf_logg=None),
        # On the line G = slope*1 + intercept = 2.714...: below it (brighter) rejects.
        _atf_row(11, andrews_atf_abs_g_mag=2.7),
        _atf_row(12, andrews_atf_abs_g_mag=2.72),
        _atf_row(13, andrews_atf_bp_rp=None),
        _atf_row(14, andrews_atf_pass2_root_ok=False),
        # mean < 3 std rejects; equality (3 * 0.25 == 0.75 exactly) passes.
        _atf_row(15, andrews_atf_m2_mean_msun=0.74, andrews_atf_m2_std_msun=0.25),
        _atf_row(16, andrews_atf_m2_mean_msun=0.75, andrews_atf_m2_std_msun=0.25),
        _atf_row(_GAIA_BH1),
        _atf_row(17, nss_solution_type="SB1"),
    ]
    result = selection.evaluate(rows)
    assert result.n_parent == 17
    assert sorted(result.surviving_source_ids) == [1, 5, 7, 9, 10, 12, 13, 16]


def test_reproduction_chain_without_columns_is_not_applicable(spec: Any) -> None:
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    result = selection.evaluate([{"source_id": 1, "nss_solution_type": "Orbital"}])
    assert result.surviving_source_ids == ()
    first = result.attrition[0]
    assert first.cut_id == "atf_covariance_valid"
    assert first.n_not_applicable == 1


def test_forward_model_never_merges_reproduction_columns(
    registry: SampleSelectionRegistry,
) -> None:
    modified = registry.resolved("andrews2022_modified")
    calls: list[int] = []

    def loader() -> dict[int, dict[str, Any]]:
        calls.append(1)
        return {1: _atf_row(1)}

    fm = SampleSelection(
        modified, mode=SampleSelectionMode.FORWARD_MODEL, reproduction_columns_loader=loader
    )
    row = {
        "source_id": 1,
        "nss_solution_type": "Orbital",
        "p_m2_above": 0.99,
        "goodness_of_fit": 1.0,
        "m2_msun": 2.0,
        "m2_msun_error": 0.1,
        "logg_apsis": 4.5,
        "abs_g_mag": 5.0,
        "bp_rp": 1.0,
    }
    assert fm.evaluate([row]).surviving_source_ids == (1,)
    assert calls == []
    repro = SampleSelection(
        modified, mode=SampleSelectionMode.REPRODUCTION, reproduction_columns_loader=loader
    )
    assert repro.evaluate([{"source_id": 1, "nss_solution_type": "Orbital"}]).surviving_source_ids == (1,)
    assert calls == [1]


def test_merge_does_not_overwrite_row_keys() -> None:
    rows = [{"source_id": 1, "andrews_atf_logg": 4.0}, {"source_id": 2}]
    merged = merge_reproduction_columns(rows, {1: {"andrews_atf_logg": 3.0, "x": 1}})
    assert merged[0] == {"source_id": 1, "andrews_atf_logg": 4.0, "x": 1}
    assert merged[1] is rows[1]


def test_sidecar_roundtrip_and_fingerprint_guard(tmp_path: Path, spec: Any) -> None:
    fp = procedure_fingerprint(spec)
    cols = {
        5: {**_atf_row(5), "andrews_atf_covariance_failure": None,
            "andrews_atf_pass1_n_root_failed": 0},
        6: {"andrews_atf_covariance_ok": False,
            "andrews_atf_covariance_failure": COV_FAIL_SINGULAR,
            "andrews_atf_pass1_root_ok": None,
            "andrews_atf_p_m2_above": float("nan")},
    }
    path = tmp_path / "dr3" / f"atf_notebook_{fp}.h5"
    write_sidecar(path, cols, fingerprint=fp, attrs={"note": "test"})
    back = read_sidecar(path, fingerprint=fp)
    assert back[5]["andrews_atf_m1_source"] == M1_SOURCE_FLAME
    assert back[5]["andrews_atf_p_m2_above"] == pytest.approx(0.99)
    assert back[5]["andrews_atf_pass1_root_ok"] is True
    assert back[6]["andrews_atf_covariance_ok"] is False
    assert back[6]["andrews_atf_pass1_root_ok"] is None
    assert back[6]["andrews_atf_p_m2_above"] is None
    assert back[6]["andrews_atf_covariance_failure"] == COV_FAIL_SINGULAR
    assert back[6]["andrews_atf_pass1_n_root_failed"] == -1
    with pytest.raises(AtfProcedureError):
        read_sidecar(path, fingerprint="0" * 16)
