"""Tests for selection_function_astrometric (forward_model)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config, require_dr3_active_for_v1
from darkhunter_pop.config_schema import ExtinctionModel
from darkhunter_pop.forward_model import (
    SIX_PANEL_NAMES,
    SOLUTION_TYPE_LABELS,
    MockRealizationRecord,
    SolutionType,
    classify_cascade_result,
    format_validation_gate_report,
    load_reference_panels,
    run_mock_injections,
    run_selection_function_astrometric,
    run_six_panel_validation,
    run_solution_type_validation,
    solution_type_fractions,
    verify_gaiamock_versions,
    write_selection_function_artifact,
    SelectionFunctionAstrometricResult,
    ValidationGateResult,
    SixPanelValidationResult,
    SolutionTypeFractionResult,
)
from darkhunter_pop.gaiamock_vendor import GaiamockModVersions, is_overlay_ready
from darkhunter_pop.schemas import ActiveDRMode

pytestmark = pytest.mark.unit


def _orbital_cascade(
    *,
    period: float = 1000.0,
    plx: float = 5.0,
    ecc: float = 0.2,
    inc_deg: float = 60.0,
) -> list[float]:
    """Synthetic successful 12-parameter cascade return vector."""
    sig_plx = 0.05
    a0 = 1.0
    sig_a0 = 0.05
    sig_ecc = 0.01
    return [
        plx,
        sig_plx,
        0.1,
        0.01,
        0.1,
        0.01,
        0.1,
        0.01,
        0.1,
        0.01,
        period,
        1.0,
        0.0,
        0.01,
        ecc,
        sig_ecc,
        inc_deg,
        a0,
        sig_a0,
        20.0,
        40.0,
        10.0,
        1.2,
    ]


def test_config_loads_selection_function_fragment() -> None:
    cfg = load_config()
    assert cfg.selection_function_astrometric.extinction_model is ExtinctionModel.COMBINED19
    assert cfg.dr3.selection_function_astrometric.d_min_pc == pytest.approx(100.0)
    assert cfg.selection_function_astrometric.mock_population.period_days == pytest.approx(
        1000.0
    )
    require_dr3_active_for_v1(cfg)


def test_classify_cascade_sentinels() -> None:
    rec = classify_cascade_result([0.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01)
    assert rec.solution_type is SolutionType.INSUFFICIENT_VISIBILITY

    rec5 = classify_cascade_result([-1.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01)
    assert rec5.solution_type is SolutionType.FIVE_PARAMETER

    rec7 = classify_cascade_result([-7.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01)
    assert rec7.solution_type is SolutionType.SEVEN_PARAMETER

    rec9 = classify_cascade_result([-9.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01)
    assert rec9.solution_type is SolutionType.NINE_PARAMETER


def test_classify_orbital_passes_dr3_cuts() -> None:
    rec = classify_cascade_result(
        _orbital_cascade(),
        m1_msun=1.0,
        m2_msun=0.5,
        flux_ratio=0.01,
    )
    assert rec.solution_type is SolutionType.TWELVE_PARAMETER_ORBITAL
    assert rec.accepted_orbital
    assert rec.P_orb_days == pytest.approx(1000.0)
    assert rec.inv_parallax_mas_inv == pytest.approx(0.2)


def test_classify_orbital_fails_cuts() -> None:
    bad = _orbital_cascade()
    bad[17] = 0.01
    bad[18] = 1.0
    rec_bad = classify_cascade_result(
        bad, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01
    )
    assert rec_bad.solution_type is SolutionType.ORBITAL_FAILED_CUTS
    assert not rec_bad.accepted_orbital


def test_solution_type_fractions_sum_to_one() -> None:
    records = [
        MockRealizationRecord(SolutionType.FIVE_PARAMETER, False),
        MockRealizationRecord(SolutionType.TWELVE_PARAMETER_ORBITAL, True),
        MockRealizationRecord(SolutionType.TWELVE_PARAMETER_ORBITAL, True),
        MockRealizationRecord(SolutionType.INSUFFICIENT_VISIBILITY, False),
    ]
    frac = solution_type_fractions(records)
    assert sum(frac.values()) == pytest.approx(1.0)
    assert frac["twelve_parameter_orbital"] == pytest.approx(0.5)


def test_six_panel_validation_identical_passes() -> None:
    rng = np.random.default_rng(0)
    n = 200
    panels = {
        "P_orb_days": rng.uniform(100.0, 5000.0, size=n),
        "G_mag": rng.uniform(10.0, 15.0, size=n),
        "inv_parallax_mas_inv": rng.uniform(0.002, 0.02, size=n),
        "eccentricity": rng.uniform(0.0, 0.8, size=n),
        "f_m_msun": rng.uniform(0.1, 2.0, size=n),
        "cos_inclination": rng.uniform(-1.0, 1.0, size=n),
    }
    records = [
        MockRealizationRecord(
            SolutionType.TWELVE_PARAMETER_ORBITAL,
            True,
            P_orb_days=float(panels["P_orb_days"][i]),
            G_mag=float(panels["G_mag"][i]),
            inv_parallax_mas_inv=float(panels["inv_parallax_mas_inv"][i]),
            eccentricity=float(panels["eccentricity"][i]),
            f_m_msun=float(panels["f_m_msun"][i]),
            cos_inclination=float(panels["cos_inclination"][i]),
        )
        for i in range(n)
    ]
    result = run_six_panel_validation(records, panels, ks_pvalue_min=0.01)
    assert result.all_passed


def test_solution_type_validation_pass_fail() -> None:
    records = [
        MockRealizationRecord(SolutionType.FIVE_PARAMETER, False),
        MockRealizationRecord(SolutionType.TWELVE_PARAMETER_ORBITAL, True),
    ]
    real = {label: 0.5 if label == "five_parameter" else 0.0 for label in SOLUTION_TYPE_LABELS}
    fail = run_solution_type_validation(records, real, max_abs_delta=0.01)
    assert fail.passed is False
    real_ok = solution_type_fractions(records)
    ok = run_solution_type_validation(records, real_ok, max_abs_delta=0.01)
    assert ok.passed is True


def test_load_reference_panels_fixture() -> None:
    cfg = load_config()
    panels, st = load_reference_panels(cfg)
    assert set(panels) == set(SIX_PANEL_NAMES)
    assert len(panels["P_orb_days"]) > 0
    assert set(st) == set(SOLUTION_TYPE_LABELS)


def test_write_artifact_round_trip(tmp_path: Path) -> None:
    versions = GaiamockModVersions(
        gaiamock_mod_release="gaiamock-mod-v1",
        gaiamock_mod_sha256="a" * 64,
        gaiamock_git_commit="b" * 40,
    )
    validation = ValidationGateResult(
        six_panel=SixPanelValidationResult(
            panel_names=SIX_PANEL_NAMES,
            ks_pvalues={n: 0.5 for n in SIX_PANEL_NAMES},
            ks_statistics={n: 0.1 for n in SIX_PANEL_NAMES},
            passed={n: True for n in SIX_PANEL_NAMES},
        ),
        solution_type=SolutionTypeFractionResult(
            mock_fractions={l: 1.0 / len(SOLUTION_TYPE_LABELS) for l in SOLUTION_TYPE_LABELS},
            real_fractions={l: 1.0 / len(SOLUTION_TYPE_LABELS) for l in SOLUTION_TYPE_LABELS},
            max_abs_delta=0.0,
            passed=True,
        ),
        detection_fraction=0.1,
        n_mock=10,
        n_real=10,
    )
    result = SelectionFunctionAstrometricResult(
        gaiamock_versions=versions,
        records=[],
        validation=validation,
        data_release="dr3",
    )
    path = tmp_path / "out.h5"
    write_selection_function_artifact(path, result)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["stage"] == "selection_function_astrometric"
        assert bool(handle.attrs["validation_gate_passed"])


def test_format_validation_gate_report() -> None:
    cfg = load_config()
    panels, st = load_reference_panels(cfg)
    validation = ValidationGateResult(
        six_panel=SixPanelValidationResult(
            panel_names=SIX_PANEL_NAMES,
            ks_pvalues={n: 0.5 for n in SIX_PANEL_NAMES},
            passed={n: True for n in SIX_PANEL_NAMES},
        ),
        solution_type=SolutionTypeFractionResult(
            mock_fractions=st,
            real_fractions=st,
            max_abs_delta=0.0,
            passed=True,
        ),
        detection_fraction=0.0,
        n_mock=0,
        n_real=0,
    )
    result = SelectionFunctionAstrometricResult(
        gaiamock_versions=GaiamockModVersions(
            gaiamock_mod_release="gaiamock-mod-v1",
            gaiamock_mod_sha256="c" * 64,
            gaiamock_git_commit="d" * 40,
        ),
        records=[],
        validation=validation,
        data_release="dr3",
    )
    text = format_validation_gate_report(result)
    assert "six_panel_ks" in text
    assert "overall_passed" in text


def test_dr4_refused() -> None:
    cfg = load_config()
    bad = cfg.model_copy(update={"active_dr_mode": ActiveDRMode.DR4})
    with pytest.raises(ValueError, match="not runnable"):
        require_dr3_active_for_v1(bad)


@pytest.mark.gaiamock
def test_run_selection_function_astrometric_smoke(tmp_path: Path) -> None:
    if not is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.mock_population.N_realizations = 2
    tweaked.selection_function_astrometric.validation_gate.ks_pvalue_min = 0.0
    tweaked.selection_function_astrometric.validation_gate.solution_type_fraction_max_abs_delta = (
        1.0
    )
    verify_gaiamock_versions(tweaked)
    artifact = tmp_path / "sel.h5"
    result = run_selection_function_astrometric(tweaked, artifact)
    assert artifact.is_file()
    assert result.gaiamock_versions.gaiamock_mod_release == "gaiamock-mod-v1"
    assert len(result.records) == 2
    report = format_validation_gate_report(result)
    assert "validation gate" in report


@pytest.mark.gaiamock
def test_run_mock_injections_returns_records() -> None:
    if not is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.mock_population.N_realizations = 1
    tweaked.selection_function_astrometric.extinction_model = ExtinctionModel.NONE
    gaiamock = import_gaiamock_mod()
    records, g_mag = run_mock_injections(tweaked, gaiamock)
    assert len(records) == 1
    assert len(g_mag) == 1
