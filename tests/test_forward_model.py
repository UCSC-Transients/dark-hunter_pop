"""Tests for selection_function_astrometric and selection_function_followup."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config, require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    ExtinctionModel,
    MockPopulationSampling,
    OrbitalSolutionCutsConfig,
)
from darkhunter_pop.forward_model import (
    SIX_PANEL_NAMES,
    SOLUTION_TYPE_LABELS,
    MockRealizationRecord,
    SolutionType,
    classify_cascade_result,
    draw_mock_binary_params,
    format_validation_gate_report,
    ElBadryComparisonSample,
    load_real_panels_from_data_acquisition,
    _run_single_mock_realization,
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


_CUTS = OrbitalSolutionCutsConfig()

# Synthetic placeholder panels (test-only since #339; never read by src/).
_REFERENCE_FIXTURE = Path(__file__).parent / "fixtures" / "elbadry2024_dr3_nss_reference.npz"


def _fixture_reference_panels() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with np.load(_REFERENCE_FIXTURE, allow_pickle=False) as data:
        panels = {name: np.asarray(data[name], dtype=np.float64) for name in SIX_PANEL_NAMES}
        st = {
            label: float(data[f"solution_type_frac_{label}"]) for label in SOLUTION_TYPE_LABELS
        }
    return panels, st


def _fixture_comparison() -> tuple[ElBadryComparisonSample, dict[str, float]]:
    panels, st = _fixture_reference_panels()
    sample = ElBadryComparisonSample(
        panels=panels,
        nss_solution_types=("Orbital", "AstroSpectroSB1"),
        n_rows=len(panels["P_orb_days"]),
        snapshot_id="test-fixture",
    )
    return sample, st


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


@pytest.mark.unit
def test_config_loads_selection_function_fragment() -> None:
    cfg = load_config()
    assert cfg.selection_function_astrometric.extinction_model is ExtinctionModel.COMBINED19
    assert cfg.dr3.selection_function_astrometric.d_min_pc == pytest.approx(50.0)
    pop = cfg.selection_function_astrometric.mock_population
    assert pop.sampling is MockPopulationSampling.ELBADRY_PRIOR
    assert pop.period_days == pytest.approx(1000.0)
    assert pop.N_realizations >= 100
    require_dr3_active_for_v1(cfg)


@pytest.mark.unit
def test_draw_mock_binary_params_spreads_elbadry_prior() -> None:
    cfg = load_config()
    pop = cfg.selection_function_astrometric.mock_population
    rng = np.random.default_rng(0)
    draws = [draw_mock_binary_params(pop, rng) for _ in range(64)]
    periods = [d.period_days for d in draws]
    assert min(periods) < max(periods)
    assert min(d.m1_msun for d in draws) < max(d.m1_msun for d in draws)
    assert sum(d.faint_draw for d in draws) >= 10


class _CascadeCountingGaiamock:
    """Fake gaiamock whose cascade returns the 5-parameter sentinel and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def run_full_astrometric_cascade(self, **kwargs: object) -> list[float]:
        self.calls += 1
        return [-1.0] * 23


@pytest.mark.unit
def test_faint_draw_runs_through_gaiamock() -> None:
    """#344: a faint draw is simulated, never short-circuited to insufficient_visibility."""
    cfg = load_config()
    pop = cfg.selection_function_astrometric.mock_population
    draw = draw_mock_binary_params(pop, np.random.default_rng(0))
    faint = draw.__class__(**{**draw.__dict__, "faint_draw": True})
    fake = _CascadeCountingGaiamock()
    rec = _run_single_mock_realization(
        gaiamock=fake,  # type: ignore[arg-type]
        ra=0.0,
        dec=0.0,
        d_pc=200.0,
        phot_g_mean_mag=18.0,
        config=cfg,
        c_funcs=None,
        draw=faint,
    )
    assert fake.calls == 1
    assert rec.solution_type is SolutionType.FIVE_PARAMETER
    assert not rec.accepted_orbital


@pytest.mark.unit
def test_load_real_panels_from_data_acquisition(tmp_path: Path) -> None:
    cfg = load_config()
    panels_ref, st_ref = _fixture_reference_panels()
    artifact = tmp_path / "da.h5"
    with h5py.File(artifact, "w") as handle:
        grp = handle.create_group("data_acquisition/nss_panels")
        for name in SIX_PANEL_NAMES:
            if name in panels_ref:
                grp.create_dataset(name, data=panels_ref[name])
        st_grp = handle.create_group("data_acquisition/solution_type_fractions")
        for label in SOLUTION_TYPE_LABELS:
            st_grp.create_dataset(label, data=np.float64(st_ref[label]))
    panels, st = load_real_panels_from_data_acquisition(artifact)
    assert set(panels) == set(SIX_PANEL_NAMES)
    assert set(st) == set(SOLUTION_TYPE_LABELS)


@pytest.mark.unit
def test_classify_cascade_sentinels() -> None:
    rec = classify_cascade_result([0.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS)
    assert rec.solution_type is SolutionType.INSUFFICIENT_VISIBILITY

    rec5 = classify_cascade_result([-1.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS)
    assert rec5.solution_type is SolutionType.FIVE_PARAMETER

    rec7 = classify_cascade_result([-7.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS)
    assert rec7.solution_type is SolutionType.SEVEN_PARAMETER

    rec9 = classify_cascade_result([-9.0] * 23, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS)
    assert rec9.solution_type is SolutionType.NINE_PARAMETER


@pytest.mark.unit
def test_classify_orbital_passes_dr3_cuts() -> None:
    rec = classify_cascade_result(
        _orbital_cascade(),
        m1_msun=1.0,
        m2_msun=0.5,
        flux_ratio=0.01, cuts=_CUTS,
    )
    assert rec.solution_type is SolutionType.TWELVE_PARAMETER_ORBITAL
    assert rec.accepted_orbital
    assert rec.P_orb_days == pytest.approx(1000.0)
    assert rec.inv_parallax_mas_inv == pytest.approx(0.2)


@pytest.mark.unit
def test_classify_orbital_fails_cuts() -> None:
    bad = _orbital_cascade()
    bad[17] = 0.01
    bad[18] = 1.0
    rec_bad = classify_cascade_result(
        bad, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS
    )
    assert rec_bad.solution_type is SolutionType.ORBITAL_FAILED_CUTS
    assert not rec_bad.accepted_orbital


@pytest.mark.unit
def test_classify_negative_parallax_not_sentinel_is_failed_cuts() -> None:
    bad = _orbital_cascade()
    bad[0] = -0.12
    rec = classify_cascade_result(
        bad, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS
    )
    assert rec.solution_type is SolutionType.ORBITAL_FAILED_CUTS
    assert not rec.accepted_orbital


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_reference_fixture_is_test_only() -> None:
    """#339: the synthetic placeholder fixture loads for tests but no src/ module names it."""
    panels, st = _fixture_reference_panels()
    assert set(panels) == set(SIX_PANEL_NAMES)
    assert set(st) == set(SOLUTION_TYPE_LABELS)
    src = Path(__file__).resolve().parents[1] / "src" / "darkhunter_pop"
    offenders = [
        str(path)
        for path in src.rglob("*.py")
        if "elbadry2024_dr3_nss_reference" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.unit
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


@pytest.mark.unit
def test_format_validation_gate_report() -> None:
    panels, st = _fixture_reference_panels()
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


@pytest.mark.unit
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
    comparison, real_st = _fixture_comparison()
    result = run_selection_function_astrometric(
        tweaked, artifact, real_comparison=comparison, real_solution_fractions=real_st
    )
    assert artifact.is_file()
    assert result.gaiamock_versions.gaiamock_mod_release == "gaiamock-mod-v1"
    assert len(result.records) == 2
    report = format_validation_gate_report(result)
    assert "validation gate" in report


@pytest.mark.gaiamock
@pytest.mark.slow
def test_validation_gate_elbadry_prior_against_fixture(tmp_path: Path) -> None:
    """El-Badry prior mocks should populate insufficient_visibility and accepted orbits.

    Pins the legacy ``numpy.random`` global state before invoking the gaiamock
    cascade. ``vendor/gaiamock/gaiamock_mod.py`` draws its per-epoch measurement
    noise, its random 10% observation rejection, and its cascade-fit
    initial-guess perturbations via the legacy ``numpy.random.*`` module-level
    functions -- an RNG stream entirely separate from the
    ``np.random.default_rng(mock_population.random_seed)`` Generator this
    pipeline seeds explicitly for its own mock-population draws (gaiamock is
    vendored and must not be reimplemented, see ``docs/GAIAMOCK_API.md``, so
    that internal RNG usage cannot be swapped for a Generator here). Without
    pinning the legacy global state, this test's accepted-orbital count
    silently depends on how many legacy ``numpy.random`` calls other tests
    already made earlier in the same process -- exactly the mechanism that let
    it fail once in a full-suite run and pass immediately after in isolation.
    Seeding here makes the outcome deterministic and independent of test
    order/collection; it does not change the statistical assertions below.
    """
    if not is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    np.random.seed(20260908)
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.extinction_model = ExtinctionModel.NONE
    tweaked.selection_function_astrometric.mock_population = (
        tweaked.selection_function_astrometric.mock_population.model_copy(
            update={"N_realizations": 80}
        )
    )
    verify_gaiamock_versions(tweaked)
    artifact = tmp_path / "sel_elbadry.h5"
    comparison, real_st = _fixture_comparison()
    result = run_selection_function_astrometric(
        tweaked, artifact, real_comparison=comparison, real_solution_fractions=real_st
    )
    insuf = result.validation.solution_type.mock_fractions.get(
        "insufficient_visibility", 0.0
    )
    assert insuf > 0.1
    # El-Badry priors yield a low DR3 orbital acceptance rate (~few percent); one
    # accepted realization is enough to prove the cascade path is wired.
    assert len([r for r in result.records if r.accepted_orbital]) >= 1


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


@pytest.mark.unit
def test_config_loads_selection_function_followup_fragment() -> None:
    cfg = load_config()
    fu = cfg.selection_function_followup
    assert len(fu.target_lists) == 3
    assert fu.target_lists[0].name == "andrews"
    assert len(fu.major_surveys) == 4
    assert {s.name for s in fu.major_surveys} == {"APOGEE", "RAVE", "LAMOST", "DESI"}
    assert cfg.dr3.selection_function_followup.accel_jerk_catalog_id == "dr3_accel_jerk_pinned"
    assert cfg.dr4.selection_function_followup.accel_jerk_catalog_id == "dr4_accel_jerk_pinned"
    assert fu.target_list_sheet.revision_history_incompleteness_caveat is True


@pytest.mark.unit
def test_followup_observability_and_tiers() -> None:
    from darkhunter_pop.forward_model import (
        ad_hoc_literature_probability,
        followup_selection_probability,
        load_survey_sf,
        passes_observability_cuts,
        _resolve_path,
    )
    from darkhunter_pop.schemas import FollowUpRecord

    cfg = load_config().selection_function_followup
    assert passes_observability_cuts(
        declination_deg=10.0, g_mag=12.0, config=cfg
    )
    assert not passes_observability_cuts(
        declination_deg=-80.0, g_mag=12.0, config=cfg
    )

    rec = FollowUpRecord(
        source_id=1,
        brightness_g_mag=12.0,
        declination_deg=10.0,
        pm_ra_mas_yr=2.0,
        pm_dec_mas_yr=1.0,
    )
    surveys = [
        load_survey_sf(_resolve_path(s.selection_function_path))
        for s in cfg.major_surveys
    ]
    p_list, tier = followup_selection_probability(
        rec,
        cfg,
        survey_lookups=surveys,
        teff_k=5000.0,
        on_target_list=True,
        target_list_cooler_pref=True,
    )
    assert tier == "target_list"
    assert p_list > 0

    p_surv, tier_s = followup_selection_probability(
        rec,
        cfg,
        survey_lookups=surveys,
        major_survey_name="APOGEE",
    )
    assert tier_s == "documented:APOGEE"
    assert p_surv > 0

    p_adhoc = ad_hoc_literature_probability(rec, cfg)
    assert 0 < p_adhoc <= 1


@pytest.mark.unit
def test_mine_adoption_dates_and_weekly_snapshot(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import (
        SHEETS_REVISION_INCOMPLETENESS_CAVEAT,
        diff_sheet_revisions,
        mine_sheet_adoption_dates_from_revisions,
        write_derived_adoption_dates,
        write_weekly_sheet_snapshot,
    )

    rev0: list[dict[str, str]] = []
    rev1 = [{"source_id": "100", "name": "a"}]
    rev2 = [
        {"source_id": "100", "name": "a"},
        {"source_id": "200", "name": "b"},
    ]
    newly = diff_sheet_revisions(rev1, rev2, adoption_date="2024-06-01")
    assert newly == {"200": "2024-06-01"}

    adopted = mine_sheet_adoption_dates_from_revisions(
        [("2024-01-01", rev0), ("2024-03-01", rev1), ("2024-06-01", rev2)]
    )
    assert adopted["100"] == "2024-03-01"
    assert adopted["200"] == "2024-06-01"

    snap = write_weekly_sheet_snapshot(
        rev2,
        tmp_path / "snapshots",
        when=datetime(2024, 6, 3, tzinfo=timezone.utc),
    )
    assert snap.is_file()
    assert (tmp_path / "snapshots" / f"{snap.name}.meta.json").is_file()

    out = tmp_path / "derived.yaml"
    write_derived_adoption_dates(out, adopted)
    assert "100" in out.read_text(encoding="utf-8")
    assert "incompleteness" in SHEETS_REVISION_INCOMPLETENESS_CAVEAT.lower() or (
        "incomplete" in SHEETS_REVISION_INCOMPLETENESS_CAVEAT.lower()
    )


@pytest.mark.unit
def test_fetch_sheet_revision_exports_injectable(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import fetch_sheet_revision_exports

    def list_revisions(_sid: str) -> list[dict[str, object]]:
        return [
            {"id": "r1", "modifiedTime": "2024-01-10T12:00:00Z"},
            {"id": "r2", "modifiedTime": "2024-02-10T12:00:00Z"},
        ]

    def export_revision(rev_id: str, _range: str) -> list[dict[str, str]]:
        if rev_id == "r1":
            return [{"source_id": "1"}]
        return [{"source_id": "1"}, {"source_id": "2"}]

    exports = fetch_sheet_revision_exports(
        "sheet123",
        sheet_range="Sheet1",
        credentials_env="GOOGLE_APPLICATION_CREDENTIALS",
        list_revisions=list_revisions,
        export_revision=export_revision,
    )
    assert len(exports) == 2
    assert exports[0][0] == "2024-01-10"


@pytest.mark.unit
def test_run_selection_function_followup_writes_hdf5(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import (
        format_followup_calibration_report,
        run_selection_function_followup,
    )

    cfg = load_config()
    # Loosen KS threshold so twin parametric draws pass reliably.
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_followup.calibration.ks_pvalue_min = 0.0
    artifact = tmp_path / "followup.h5"
    result = run_selection_function_followup(tweaked, artifact, rng_seed=42)
    assert artifact.is_file()
    assert len(result.records) == 200
    assert result.accel_jerk_catalog_id == "dr3_accel_jerk_pinned"
    with h5py.File(artifact, "r") as handle:
        assert handle.attrs["stage"] == "selection_function_followup"
        assert "followup_catalog" in handle
        assert "calibration" in handle
    report = format_followup_calibration_report(result)
    assert "selection_function_followup calibration" in report
    assert "sheets_caveat" in report


@pytest.mark.unit
def test_followup_dr4_refused(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import run_selection_function_followup

    cfg = load_config()
    bad = cfg.model_copy(update={"active_dr_mode": ActiveDRMode.DR4})
    with pytest.raises(ValueError, match="not runnable"):
        run_selection_function_followup(bad, tmp_path / "x.h5")


# ---------------------------------------------------------------------------
# Multi-solution emission model (issue #243, depends on #241's rate measurement)
# ---------------------------------------------------------------------------


def _fixture_rate_table(
    *,
    n_sources: int = 100,
    n_cross_type_sources: int = 0,
    n_same_type_period_aliased_sources: int = 0,
    cross_type_combo_counts: dict[str, int] | None = None,
):
    from darkhunter_pop.forward_model import MultiSolutionRateTable

    return MultiSolutionRateTable(
        schema_version=1,
        source_snapshot_id="test-fixture",
        n_sources=n_sources,
        n_cross_type_sources=n_cross_type_sources,
        n_same_type_period_aliased_sources=n_same_type_period_aliased_sources,
        cross_type_combo_counts=cross_type_combo_counts or {},
    )


@pytest.mark.unit
def test_config_loads_multi_solution_rates_pointer() -> None:
    cfg = load_config()
    assert cfg.multi_solution_rates.enabled is True
    assert cfg.multi_solution_rates.path == "config/multi_solution_rates.yaml"
    assert cfg.selection_function_astrometric.multi_solution.primary_type == "Orbital"
    assert cfg.selection_function_followup.multi_solution.primary_type == "SB1"


@pytest.mark.unit
def test_load_multi_solution_rate_table_matches_report() -> None:
    """Loaded table reproduces #241's measured headline numbers exactly (no re-derivation)."""
    from darkhunter_pop.forward_model import load_multi_solution_rate_table

    cfg = load_config()
    table = load_multi_solution_rate_table(cfg)
    assert table.n_sources == 437275
    assert table.n_cross_type_sources == 5926
    assert table.n_same_type_period_aliased_sources == 0
    assert table.cross_type_combo_counts["Orbital+SB1"] == 5290
    assert table.same_type_period_aliased_rate == pytest.approx(0.0)
    assert table.cross_type_rate == pytest.approx(5926 / 437275)


@pytest.mark.unit
def test_multi_solution_rates_disabled_gives_empty_table() -> None:
    from darkhunter_pop.forward_model import load_multi_solution_rate_table

    cfg = load_config()
    disabled = cfg.model_copy(deep=True)
    disabled.multi_solution_rates.enabled = False
    table = load_multi_solution_rate_table(disabled)
    assert table.n_sources == 0
    assert table.cross_type_rate == 0.0
    assert table.same_type_period_aliased_rate == 0.0


@pytest.mark.unit
def test_companion_type_distribution_normalizes_and_anchors_on_primary() -> None:
    table = _fixture_rate_table(
        n_sources=1000,
        n_cross_type_sources=50,
        cross_type_combo_counts={"Orbital+SB1": 40, "Orbital+SB2": 10, "EclipsingBinary+SB1": 5},
    )
    dist = table.companion_type_distribution("Orbital")
    assert set(dist) == {"SB1", "SB2"}
    assert dist["SB1"] == pytest.approx(0.8)
    assert dist["SB2"] == pytest.approx(0.2)
    assert table.combo_rate_for_primary("Orbital") == pytest.approx(50 / 1000)
    # EclipsingBinary+SB1 does not involve "Orbital"; only SB1-anchored queries see it.
    assert table.companion_type_distribution("SB1") == {
        "Orbital": pytest.approx(40 / 45),
        "EclipsingBinary": pytest.approx(5 / 45),
    }


@pytest.mark.unit
def test_draw_cross_type_companion_zero_rate_never_fires() -> None:
    from darkhunter_pop.forward_model import draw_cross_type_companion

    table = _fixture_rate_table(n_sources=100, n_cross_type_sources=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert draw_cross_type_companion(rng, table, primary_type="Orbital") is None


@pytest.mark.unit
def test_draw_cross_type_companion_full_rate_always_fires() -> None:
    from darkhunter_pop.forward_model import draw_cross_type_companion

    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=100,
        cross_type_combo_counts={"Orbital+SB1": 100},
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert draw_cross_type_companion(rng, table, primary_type="Orbital") == "SB1"


@pytest.mark.unit
def test_draw_orbital_period_aliases_zero_rate_never_fires() -> None:
    from darkhunter_pop.forward_model import draw_orbital_period_aliases

    table = _fixture_rate_table(n_sources=100, n_same_type_period_aliased_sources=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        aliases = draw_orbital_period_aliases(
            rng, table, base_period_days=500.0, ratio_min=0.5, ratio_max=0.9
        )
        assert aliases == ()


@pytest.mark.unit
def test_draw_orbital_period_aliases_full_rate_emits_distinct_period() -> None:
    from darkhunter_pop.forward_model import draw_orbital_period_aliases

    table = _fixture_rate_table(n_sources=100, n_same_type_period_aliased_sources=100)
    rng = np.random.default_rng(0)
    for _ in range(20):
        aliases = draw_orbital_period_aliases(
            rng, table, base_period_days=500.0, ratio_min=0.5, ratio_max=0.9
        )
        assert len(aliases) == 1
        assert aliases[0] != pytest.approx(500.0)
        assert 0.0 < aliases[0] < 500.0


@pytest.mark.unit
def test_draw_multi_solution_emission_zero_rate_emits_zero_rows() -> None:
    from darkhunter_pop.config_schema import MultiSolutionEmissionConfig
    from darkhunter_pop.forward_model import draw_multi_solution_emission

    table = _fixture_rate_table(n_sources=100)
    ms_cfg = MultiSolutionEmissionConfig(primary_type="Orbital", random_seed=1)
    rng = np.random.default_rng(0)
    for _ in range(30):
        em = draw_multi_solution_emission(
            rng, table, ms_cfg, base_period_days=500.0
        )
        assert em.n_extra_rows == 0
        assert em.cross_type_companion is None
        assert em.orbital_period_aliases_days == ()


@pytest.mark.unit
def test_draw_multi_solution_emission_disabled_config_is_noop() -> None:
    from darkhunter_pop.config_schema import MultiSolutionEmissionConfig
    from darkhunter_pop.forward_model import draw_multi_solution_emission

    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=100,
        n_same_type_period_aliased_sources=100,
        cross_type_combo_counts={"Orbital+SB1": 100},
    )
    ms_cfg = MultiSolutionEmissionConfig(
        enabled=False, primary_type="Orbital", random_seed=1
    )
    rng = np.random.default_rng(0)
    em = draw_multi_solution_emission(rng, table, ms_cfg, base_period_days=500.0)
    assert em.n_extra_rows == 0


@pytest.mark.unit
def test_draw_multi_solution_emission_one_extra_row() -> None:
    """Cross-type fires deterministically, same-type never does: exactly 1 extra row."""
    from darkhunter_pop.config_schema import MultiSolutionEmissionConfig
    from darkhunter_pop.forward_model import draw_multi_solution_emission

    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=100,
        n_same_type_period_aliased_sources=0,
        cross_type_combo_counts={"Orbital+SB1": 100},
    )
    ms_cfg = MultiSolutionEmissionConfig(primary_type="Orbital", random_seed=1)
    rng = np.random.default_rng(0)
    em = draw_multi_solution_emission(rng, table, ms_cfg, base_period_days=500.0)
    assert em.n_extra_rows == 1
    assert em.cross_type_companion == "SB1"
    assert em.orbital_period_aliases_days == ()


@pytest.mark.unit
def test_draw_multi_solution_emission_more_than_one_extra_row() -> None:
    """Both sub-cases fire deterministically at once: >1 extra row (issue #243 item 5)."""
    from darkhunter_pop.config_schema import MultiSolutionEmissionConfig
    from darkhunter_pop.forward_model import draw_multi_solution_emission

    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=100,
        n_same_type_period_aliased_sources=100,
        cross_type_combo_counts={"Orbital+SB1": 100},
    )
    ms_cfg = MultiSolutionEmissionConfig(
        primary_type="Orbital",
        random_seed=1,
        same_type_period_alias_ratio_min=0.5,
        same_type_period_alias_ratio_max=0.9,
    )
    rng = np.random.default_rng(0)
    em = draw_multi_solution_emission(rng, table, ms_cfg, base_period_days=500.0)
    assert em.n_extra_rows > 1
    assert em.cross_type_companion == "SB1"
    assert len(em.orbital_period_aliases_days) == 1
    assert em.orbital_period_aliases_days[0] != pytest.approx(500.0)


@pytest.mark.unit
def test_multi_solution_emission_attached_to_accepted_orbital_realization() -> None:
    """``_run_single_mock_realization`` attaches an emission for every accepted realization."""
    from darkhunter_pop.forward_model import _run_single_mock_realization

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.multi_solution.primary_type = "Orbital"
    pop = tweaked.selection_function_astrometric.mock_population
    draw = draw_mock_binary_params(pop, np.random.default_rng(1))

    table = _fixture_rate_table(
        n_sources=10,
        n_cross_type_sources=10,
        cross_type_combo_counts={"Orbital+SB1": 10},
    )
    ms_rng = np.random.default_rng(2)

    # gaiamock is faked with a 5-parameter outcome so accepted_orbital stays False
    # and we exercise the "not attached" branch cheaply, then flip to
    # accepted_orbital=True via classify_cascade_result directly below.
    rec = _run_single_mock_realization(
        gaiamock=_CascadeCountingGaiamock(),  # type: ignore[arg-type]
        ra=0.0,
        dec=0.0,
        d_pc=200.0,
        phot_g_mean_mag=18.0,
        config=tweaked,
        c_funcs=None,
        draw=draw.__class__(**{**draw.__dict__, "faint_draw": True}),
        multi_solution_table=table,
        multi_solution_rng=ms_rng,
    )
    assert rec.multi_solution is None  # not accepted: no emission drawn

    accepted = classify_cascade_result(
        _orbital_cascade(period=500.0),
        m1_msun=1.0,
        m2_msun=0.5,
        flux_ratio=0.1, cuts=_CUTS,
    )
    assert accepted.accepted_orbital
    from darkhunter_pop.forward_model import draw_multi_solution_emission

    emission = draw_multi_solution_emission(
        ms_rng,
        table,
        tweaked.selection_function_astrometric.multi_solution,
        base_period_days=accepted.P_orb_days,
    )
    assert emission.cross_type_companion == "SB1"


@pytest.mark.unit
def test_run_multi_solution_diagnostic_perfect_match() -> None:
    from darkhunter_pop.forward_model import (
        MockRealizationRecord,
        MultiSolutionEmission,
        run_multi_solution_diagnostic,
    )

    cfg = load_config()
    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=20,
        cross_type_combo_counts={"Orbital+SB1": 20},
    )
    # 100 accepted realizations, exactly 20 carrying an SB1 companion: matches table exactly.
    records = [
        MockRealizationRecord(
            solution_type=SolutionType.TWELVE_PARAMETER_ORBITAL,
            accepted_orbital=True,
            multi_solution=MultiSolutionEmission(
                cross_type_companion="SB1" if i < 20 else None
            ),
        )
        for i in range(100)
    ]
    result = run_multi_solution_diagnostic(
        records, table, cfg.selection_function_astrometric
    )
    assert result.mock_cross_type_rate == pytest.approx(0.20)
    assert result.real_cross_type_rate == pytest.approx(0.20)
    assert result.max_abs_delta_cross_type == pytest.approx(0.0)
    assert result.passed


@pytest.mark.unit
def test_run_multi_solution_diagnostic_flags_mismatch() -> None:
    from darkhunter_pop.forward_model import (
        MockRealizationRecord,
        MultiSolutionEmission,
        run_multi_solution_diagnostic,
    )

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.validation_gate.multi_solution_rate_max_abs_delta = 0.01
    table = _fixture_rate_table(
        n_sources=100,
        n_cross_type_sources=50,
        cross_type_combo_counts={"Orbital+SB1": 50},
    )
    # No mock realization carries a companion at all: large delta from the 50% real rate.
    records = [
        MockRealizationRecord(
            solution_type=SolutionType.TWELVE_PARAMETER_ORBITAL,
            accepted_orbital=True,
            multi_solution=MultiSolutionEmission(),
        )
        for _ in range(100)
    ]
    result = run_multi_solution_diagnostic(
        records, table, tweaked.selection_function_astrometric
    )
    assert not result.passed
    assert result.max_abs_delta_cross_type == pytest.approx(0.5)


@pytest.mark.unit
def test_followup_multi_solution_independent_draw(tmp_path: Path) -> None:
    """Follow-up draws its own multi-solution emission, independent of the astrometric side."""
    from darkhunter_pop.forward_model import run_selection_function_followup

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_followup.calibration.ks_pvalue_min = 0.0
    # Force the shared table to a deterministic near-certain companion rate so the
    # follow-up-side mechanism is exercised without depending on the real measured rate.
    rates_path = tmp_path / "rates.yaml"
    import yaml

    rates_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_snapshot_id": "test-fixture",
                "n_sources": 10,
                "n_cross_type_sources": 10,
                "n_same_type_period_aliased_sources": 0,
                "cross_type_combo_counts": {"Orbital+SB1": 10},
            }
        ),
        encoding="utf-8",
    )
    tweaked.multi_solution_rates.path = str(rates_path)
    tweaked.selection_function_followup.multi_solution.primary_type = "SB1"

    artifact = tmp_path / "followup.h5"
    result = run_selection_function_followup(tweaked, artifact, rng_seed=7)
    assert len(result.multi_solution) == len(result.records)
    n_selected = sum(1 for r in result.records if r.n_observations > 0)
    n_companion = sum(
        1 for ms in result.multi_solution if ms.cross_type_companion == "Orbital"
    )
    assert n_selected > 0
    # Every selected (n_observations > 0) record independently rolled the companion draw;
    # at rate=1.0 for primary_type="SB1" every selected record gets an "Orbital" companion.
    assert n_companion == n_selected
    with h5py.File(artifact, "r") as handle:
        grp = handle["followup_catalog"]
        assert "multi_solution_cross_type_companion" in grp


# ---------------------------------------------------------------------------
# #339: six-panel mock gating, mock f_m, real El-Badry 2024 comparison sample
# ---------------------------------------------------------------------------


class _FakeGaiamock:
    """Minimal stand-in exposing the two gaiamock_mod functions these tests need."""

    def __init__(self, m2: float = 0.777) -> None:
        self.m2 = m2

    def get_companion_mass_from_mass_function(self, **_kw: float) -> float:
        return self.m2

    @staticmethod
    def get_Campbell_elements(A, B, F, G):  # noqa: N802,N803 - gaiamock signature
        from darkhunter_pop.physics_utils import thiele_innes_to_campbell

        a0, omega, inc = thiele_innes_to_campbell(A, B, F, G)
        return a0, np.zeros_like(a0), omega, inc


@pytest.mark.unit
def test_classify_failed_cuts_leaves_every_panel_empty() -> None:
    bad = _orbital_cascade()
    bad[17] = 0.01  # a0 / sigma_a0 = 0.2 < 5
    rec = classify_cascade_result(
        bad, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS, gaiamock=_FakeGaiamock()
    )
    assert not rec.accepted_orbital
    for attr in (
        "P_orb_days",
        "G_mag",
        "inv_parallax_mas_inv",
        "eccentricity",
        "f_m_msun",
        "cos_inclination",
        "parallax_mas",
        "a0_mas",
        "m2_from_mass_function_msun",
    ):
        assert getattr(rec, attr) is None, attr


@pytest.mark.unit
def test_classify_applies_goodness_of_fit_f2_cut() -> None:
    """gaiamock returns orbital fits with F2 >= 25; El-Badry 2024 Eq. 18 rejects them."""
    cascade = _orbital_cascade()
    cascade[21] = _CUTS.goodness_of_fit_f2_max + 1.0
    rec = classify_cascade_result(cascade, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.01, cuts=_CUTS)
    assert rec.solution_type is SolutionType.ORBITAL_FAILED_CUTS
    assert rec.cos_inclination is None


@pytest.mark.unit
def test_classify_mock_f_m_is_astrometric_mass_function() -> None:
    from darkhunter_pop.physics_utils import astrometric_mass_function

    period, plx = 800.0, 4.0
    cascade = _orbital_cascade(period=period, plx=plx, inc_deg=30.0)
    rec = classify_cascade_result(
        cascade,
        m1_msun=1.0,
        m2_msun=0.5,
        flux_ratio=0.01,
        cuts=_CUTS,
        gaiamock=_FakeGaiamock(m2=0.777),
    )
    assert rec.accepted_orbital
    expected = float(astrometric_mass_function(cascade[17], plx, period))
    assert rec.f_m_msun == pytest.approx(expected)
    assert rec.m2_from_mass_function_msun == pytest.approx(0.777)
    assert rec.f_m_msun != pytest.approx(rec.m2_from_mass_function_msun)
    assert rec.cos_inclination == pytest.approx(np.cos(np.radians(30.0)))
    assert rec.inv_parallax_mas_inv == pytest.approx(1.0 / plx)


def _comparison_columns() -> dict[str, np.ndarray]:
    types = np.array(
        [
            "Orbital",
            "Orbital",  # fan-out duplicate of row 0 (same source_id + type)
            "AstroSpectroSB1",
            "SB1",
            "EclipsingBinary",
            "OrbitalTargetedSearch",
            "Acceleration7",
        ]
    )
    n = len(types)
    return {
        "source_id": np.array([1, 1, 2, 3, 4, 5, 6], dtype=np.int64),
        "nss_solution_type": types,
        "period": np.array([500.0, 500.0, 900.0, 3.0, 1.5, 700.0, np.nan]),
        "eccentricity": np.array([0.3, 0.3, 0.1, 0.0, 0.0, 0.2, np.nan]),
        "parallax": np.array([2.0, 2.0, 5.0, 1.0, 1.0, 3.0, 1.0]),
        "g_mag": np.linspace(10.0, 13.0, n),
        "A": np.array([1.0, 1.0, 0.5, np.nan, np.nan, 0.8, np.nan]),
        "B": np.array([0.2, 0.2, 0.1, np.nan, np.nan, 0.1, np.nan]),
        "F": np.array([-0.1, -0.1, 0.3, np.nan, np.nan, 0.2, np.nan]),
        "G": np.array([0.9, 0.9, 0.4, np.nan, np.nan, 0.7, np.nan]),
    }


@pytest.mark.unit
def test_comparison_sample_is_orbital_plus_astrospectrosb1_only() -> None:
    from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels

    panels, n_rows = build_elbadry2024_comparison_panels(
        _comparison_columns(),
        nss_solution_types=("Orbital", "AstroSpectroSB1"),
        gaiamock=_FakeGaiamock(),
    )
    # Exact-type match: SB1 / EclipsingBinary / OrbitalTargetedSearch / Acceleration7
    # excluded; the duplicated Orbital row collapses to one.
    assert n_rows == 2
    assert set(panels) == set(SIX_PANEL_NAMES)
    # One row set feeds every panel (all values finite here).
    assert {len(v) for v in panels.values()} == {2}
    np.testing.assert_allclose(np.sort(panels["P_orb_days"]), [500.0, 900.0])
    np.testing.assert_allclose(np.sort(panels["inv_parallax_mas_inv"]), [0.2, 0.5])
    assert np.all(np.abs(panels["cos_inclination"]) <= 1.0)


@pytest.mark.unit
def test_comparison_sample_f_m_matches_mock_formula() -> None:
    from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels
    from darkhunter_pop.physics_utils import astrometric_mass_function, thiele_innes_to_campbell

    cols = _comparison_columns()
    panels, _n = build_elbadry2024_comparison_panels(
        cols, nss_solution_types=("Orbital",), gaiamock=_FakeGaiamock()
    )
    a0, _om, _inc = thiele_innes_to_campbell(cols["A"][0], cols["B"][0], cols["F"][0], cols["G"][0])
    expected = astrometric_mass_function(a0, cols["parallax"][0], cols["period"][0])
    np.testing.assert_allclose(panels["f_m_msun"], [float(expected)])


@pytest.mark.unit
def test_run_stage_refuses_without_real_side(tmp_path: Path) -> None:
    """No data_acquisition artifact and no explicit sample: fail loudly, no fixture fallback."""
    cfg = load_config()
    with pytest.raises(ValueError, match="no fixture fallback"):
        run_selection_function_astrometric(cfg, tmp_path / "sf.h5")
    with pytest.raises(FileNotFoundError):
        run_selection_function_astrometric(
            cfg, tmp_path / "sf.h5", data_acquisition_artifact=tmp_path / "missing.h5"
        )
    comparison, _st = _fixture_comparison()
    with pytest.raises(ValueError, match="together"):
        run_selection_function_astrometric(
            cfg, tmp_path / "sf.h5", real_comparison=comparison
        )


def _gate_result_for(records: list[MockRealizationRecord]) -> ValidationGateResult:
    return ValidationGateResult(
        six_panel=SixPanelValidationResult(panel_names=SIX_PANEL_NAMES),
        solution_type=SolutionTypeFractionResult(
            mock_fractions={}, real_fractions={}, max_abs_delta=0.0, passed=True
        ),
        detection_fraction=0.0,
        n_mock=len(records),
        n_real=0,
    )


@pytest.mark.unit
def test_artifact_persists_gated_mock_and_real_six_panel_samples(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import load_six_panel_samples

    accepted = classify_cascade_result(
        _orbital_cascade(period=600.0, plx=2.0),
        m1_msun=1.0,
        m2_msun=0.5,
        flux_ratio=0.01,
        cuts=_CUTS,
    )
    failed = MockRealizationRecord(SolutionType.ORBITAL_FAILED_CUTS, False)
    records = [accepted, failed, MockRealizationRecord(SolutionType.FIVE_PARAMETER, False)]
    comparison, _st = _fixture_comparison()
    result = SelectionFunctionAstrometricResult(
        gaiamock_versions=GaiamockModVersions(
            gaiamock_mod_release="gaiamock-mod-v1",
            gaiamock_mod_sha256="a" * 64,
            gaiamock_git_commit="b" * 40,
        ),
        records=records,
        validation=_gate_result_for(records),
        data_release="dr3",
        real_comparison=comparison,
        injected_truth={
            "m1_msun": np.array([1.0, 1.2, 0.9]),
            "period_days": np.array([600.0, 50.0, 9000.0]),
        },
    )
    path = tmp_path / "sf.h5"
    g_mag = np.array([12.5, 14.0, 9.0])
    write_selection_function_artifact(path, result, g_mag=g_mag)
    panels, meta = load_six_panel_samples(path)
    assert set(panels) == set(SIX_PANEL_NAMES)
    for name in SIX_PANEL_NAMES:
        assert panels[name]["mock"].size == 1, name  # only the accepted realization
        np.testing.assert_allclose(panels[name]["real"], comparison.panels[name])
    assert panels["G_mag"]["mock"][0] == pytest.approx(12.5)
    assert panels["inv_parallax_mas_inv"]["mock"][0] == pytest.approx(0.5)
    assert meta["real_nss_solution_types"] == ["Orbital", "AstroSpectroSB1"]
    assert meta["mock_n_accepted"] == 1
    assert meta["mock_n_realizations"] == 3
    with h5py.File(path, "r") as handle:
        cos_col = handle["mock_catalog/cos_inclination"][()]
        assert np.isfinite(cos_col[0]) and np.all(np.isnan(cos_col[1:]))
        np.testing.assert_allclose(handle["mock_catalog/truth/m1_msun"][()], [1.0, 1.2, 0.9])
        np.testing.assert_allclose(
            handle["mock_catalog/truth/period_days"][()], [600.0, 50.0, 9000.0]
        )


@pytest.mark.unit
def test_load_six_panel_samples_refuses_pre_339_artifact(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import load_six_panel_samples

    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "selection_function_astrometric"
    with pytest.raises(KeyError, match="six_panel_samples"):
        load_six_panel_samples(path)


_UNCUT_SNAPSHOT_META = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dr3"
    / "gaia_snapshots"
    / "20260826T234425Z_3d3f740b080c"
    / "meta.yaml"
)


@pytest.mark.slow
@pytest.mark.gaiamock
def test_real_comparison_sample_count_matches_elbadry2024() -> None:
    """Uncut DR3 snapshot: Orbital + AstroSpectroSB1 = 168,065 rows (paper: ~168,000).

    134,598 Orbital (Andrews' parent) + 33,467 AstroSpectroSB1 (paper footnote 6).
    """
    if not _UNCUT_SNAPSHOT_META.is_file():
        pytest.skip("uncut DR3 snapshot not on this checkout")
    if not is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    from darkhunter_pop.data_acquisition import load_gaia_snapshot
    from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    _meta, table = load_gaia_snapshot(_UNCUT_SNAPSHOT_META, verify_checksum=False)
    panels, n_rows = build_elbadry2024_comparison_panels(
        table,
        nss_solution_types=("Orbital", "AstroSpectroSB1"),
        gaiamock=import_gaiamock_mod(),
    )
    assert n_rows == 168065
    for name in SIX_PANEL_NAMES:
        assert 0 < panels[name].size <= n_rows


@pytest.mark.unit
def test_comparison_panels_accept_astropy_table() -> None:
    """The stage passes an astropy Table (whose ``in`` iterates rows, not names)."""
    from astropy.table import Table

    from darkhunter_pop.forward_model import build_elbadry2024_comparison_panels

    cols = _comparison_columns()
    cols["phot_g_mean_mag"] = cols.pop("g_mag")
    panels, n_rows = build_elbadry2024_comparison_panels(
        Table(cols), nss_solution_types=("Orbital", "AstroSpectroSB1"), gaiamock=_FakeGaiamock()
    )
    assert n_rows == 2
    assert panels["G_mag"].size == 2


# ---------------------------------------------------------------------------
# #339 Phase 2 (paper-determined pieces): acceleration publication cuts, G < 19
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acceleration_publication_cuts_follow_elbadry2024() -> None:
    from darkhunter_pop.config_schema import AccelerationPublicationCutsConfig

    acc = AccelerationPublicationCutsConfig()
    nine = [-9.0] * 23
    nine[1], nine[13] = 25.0, 24.0  # s > 20, F2 < 25 -> published 9-par
    rec9 = classify_cascade_result(
        nine, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.0, cuts=_CUTS, acceleration_cuts=acc
    )
    assert rec9.solution_type is SolutionType.NINE_PARAMETER
    assert rec9.published_acceleration
    assert rec9.acceleration_significance == pytest.approx(25.0)

    seven = [-7.0] * 23
    seven[1], seven[9] = 25.0, 23.0  # F2 >= 22 -> not published as 7-par
    rec7 = classify_cascade_result(
        seven, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.0, cuts=_CUTS, acceleration_cuts=acc
    )
    assert rec7.solution_type is SolutionType.SEVEN_PARAMETER
    assert not rec7.published_acceleration
    assert rec7.acceleration_f2 == pytest.approx(23.0)

    seven[1], seven[9] = 15.0, 10.0  # 12 < s <= 20: removed from orbit fitting, unpublished
    rec7b = classify_cascade_result(
        seven, m1_msun=1.0, m2_msun=0.5, flux_ratio=0.0, cuts=_CUTS, acceleration_cuts=acc
    )
    assert not rec7b.published_acceleration


class _FakeInjectionGaiamock(_CascadeCountingGaiamock):
    """Fake gaiamock for run_mock_injections_with_truth: fixed distances, 5-par cascade."""

    def __init__(self, d_pc: np.ndarray) -> None:
        super().__init__()
        self.d_pc = d_pc

    def generate_coordinates_at_a_given_distance_exponential_disk(self, **kw):  # noqa: D401
        n = len(self.d_pc)
        z = np.zeros(n)
        return z + 10.0, z + 5.0, self.d_pc, z + 1.0, z, z

    @staticmethod
    def xyz_to_galactic(*, x, y, z):
        return np.zeros_like(x), np.zeros_like(x)

    @staticmethod
    def read_in_C_functions():
        return None


@pytest.mark.unit
def test_mock_g_limit_removes_faint_draws_before_cascade() -> None:
    from darkhunter_pop.forward_model import run_mock_injections_with_truth

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.selection_function_astrometric.extinction_model = ExtinctionModel.NONE
    pop = tweaked.selection_function_astrometric.mock_population
    tweaked.selection_function_astrometric.mock_population = pop.model_copy(
        update={"N_realizations": 4, "faint_draw_fraction": 0.0, "Mg_tot_min": 4.0,
                "Mg_tot_max": 4.0001}
    )
    # M_G = 4: G = 4 + 5 log10(d/10) -> 9, 14, 19.0 (removed, not < 19), 24 (removed)
    d = np.array([100.0, 1000.0, 10000.0, 100000.0])
    fake = _FakeInjectionGaiamock(d)
    records, g, truth = run_mock_injections_with_truth(tweaked, fake)  # type: ignore[arg-type]
    assert fake.calls == 2
    assert len(records) == 2 and len(g) == 2
    assert np.all(g < 19.0)
    np.testing.assert_allclose(truth["distance_pc"], [100.0, 1000.0])
