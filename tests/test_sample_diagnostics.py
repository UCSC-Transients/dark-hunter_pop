"""Phase 8 sample-reproduction diagnostics (CONTINUATION_PLAN §13 / issue #113)."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionEntry, SampleSelectionMode
from darkhunter_pop.diagnostics import (
    clear_diagnostic_helpers,
    emit_covariance_health,
    emit_janssens_segment_occupancy,
    emit_mode_divergence,
    emit_sample_attrition_waterfall,
    emit_sample_reproduction_report,
    emit_sample_selection_function,
    emit_simon2026_exclusion_breakdown,
    get_diagnostic_helper,
    list_diagnostic_helpers,
    resolve_diagnostic_dirs,
    run_diagnostic_suite,
    _ensure_builtin_helpers_registered,
)
from darkhunter_pop.elbadry2026_selection import load_simon2026_orbital
from darkhunter_pop.nss_covariance import CovarianceFailure, CovarianceHealth, CovarianceResult
from darkhunter_pop.sample_diagnostics import (
    SampleDiagnosticsBundle,
    compare_to_published,
    compute_janssens_segment_occupancy,
    compute_mode_divergence,
    format_attrition_waterfall_report,
    load_published_source_ids,
)
from darkhunter_pop.sample_selection import (
    CutAttrition,
    CutKind,
    NotApplicable,
    SampleEvaluationResult,
    SampleSelectionRegistry,
    load_sample_selection_file,
    sample_evaluation_result_from_dict,
)

pytestmark = pytest.mark.unit

_GAIA_BH1 = 4373465352415301632
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
_E1_G_LT_15 = (
    3640889032890567040,
    4373465352415301632,
    6281177228434199296,
    5870569352746779008,
    3664684869697065984,
)


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
        rows.append(
            _astro_row(sid, m2_tilde_msun=0.5, goodness_of_fit=20.0, andrews_member=True)
        )
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


def _elbadry_registry() -> SampleSelectionRegistry:
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


def test_config_loads_sample_reproduction_hooks() -> None:
    cfg = load_config()
    hooks = cfg.diagnostics.hooks
    assert hooks.sample_attrition_waterfall is True
    assert hooks.sample_reproduction_report is True
    assert hooks.simon2026_exclusion_breakdown is True
    assert hooks.covariance_health is True
    assert hooks.sample_selection_function is True
    assert hooks.mode_divergence is True
    assert hooks.janssens_segment_occupancy is True
    assert cfg.diagnostics.sample_reproduction.mode_divergence_pairs[0].left == (
        "andrews2022"
    )
    assert cfg.diagnostics.sample_reproduction.mode_divergence_pairs[
        0
    ].expected_only_in_right == [_GAIA_BH1]


def test_helpers_register_sample_hooks() -> None:
    clear_diagnostic_helpers()
    _ensure_builtin_helpers_registered()
    names = list_diagnostic_helpers()
    for name in (
        "emit_sample_attrition_waterfall",
        "emit_sample_reproduction_report",
        "emit_simon2026_exclusion_breakdown",
        "emit_covariance_health",
        "emit_sample_selection_function",
        "emit_mode_divergence",
        "emit_janssens_segment_occupancy",
    ):
        assert name in names
        assert callable(get_diagnostic_helper(name))


def test_attrition_separates_failed_and_not_applicable(tmp_path: Path) -> None:
    results = _elbadry_registry().evaluate_all(_catalog_rows())
    report = format_attrition_waterfall_report({"elbadry2026": results["elbadry2026"]})
    assert "n_not_applicable" in report
    assert "n_failed" in report
    assert "§15 Q10" in report or "Q10" in report
    assert "subsample[primary_ns_bh].n: 47" in report
    assert "subsample[sub_chandrasekhar].n: 22" in report

    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="attrition")
    emission = emit_sample_attrition_waterfall(
        cfg, dirs, results={"elbadry2026": results["elbadry2026"]}
    )
    assert emission.skipped_reason is None
    assert emission.reports
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "n_not_applicable" in text
    assert "astrometric:primary_ns_bh" in text or "primary_ns_bh" in text


def test_reproduction_report_elbadry2026_tables(tmp_path: Path) -> None:
    results = _elbadry_registry().evaluate_all(_catalog_rows())
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="repro")
    specs = {
        "elbadry2026": load_sample_selection_file(
            repo_root() / "config/selections/elbadry2026.yaml"
        )
    }
    emission = emit_sample_reproduction_report(
        cfg, dirs, results={"elbadry2026": results["elbadry2026"]}, specs=specs
    )
    assert emission.payload["all_n_match"] is True
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "recovered_n: 227" in text
    assert "elbadry2026_astrometric" in text
    assert "elbadry2026_spectroscopic" in text
    table7 = load_published_source_ids(
        "config/selections/external/elbadry2026_table7.yaml"
    )
    assert len(table7) == 76
    cmp = compare_to_published(
        sample_name="elbadry2026_astrometric",
        recovered_ids=results["elbadry2026"].branch_surviving["astrometric"],
        published_n=76,
        published_ids=table7,
    )
    # Synthetic catalog IDs will not match published table IDs; N must match.
    assert cmp.n_match is True
    assert cmp.id_match is False


def test_simon2026_hook_reproduces_5_2_1_1(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="simon")
    emission = emit_simon2026_exclusion_breakdown(
        cfg,
        dirs,
        sample_ids=_SIMON_IN_SAMPLE,
        simon_rows=load_simon2026_orbital(),
        spec=load_sample_selection_file(
            repo_root() / "config/selections/elbadry2026.yaml"
        ),
    )
    assert emission.payload["sb1_fails_significance"] == 5
    assert emission.payload["astrometric_f2_above_max"] == 2
    assert emission.payload["fainter_than_g_limit"] == 1
    assert emission.payload["fails_m2_over_m1"] == 1
    assert emission.payload["in_sample"] == 11


def test_covariance_health_hook(tmp_path: Path) -> None:
    health = CovarianceHealth()
    health.record("Orbital", CovarianceResult(None, CovarianceFailure.MISSING_CORR))
    health.record("Orbital", CovarianceResult(None, CovarianceFailure.NON_PSD))
    health.ok = 1
    health.by_solution_type.setdefault("SB1", {})["ok"] = 1

    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="cov")
    emission = emit_covariance_health(cfg, dirs, health=health)
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "missing_corr: 1" in text
    assert "non_psd: 1" in text
    assert "covariance_ok: 1" in text


def test_mode_divergence_andrews_gaia_bh1(tmp_path: Path) -> None:
    left = SampleEvaluationResult(
        name="andrews2022",
        mode=SampleSelectionMode.REPRODUCTION,
        mass_source="paper",
        parent_adql="SELECT 1",
        surviving_source_ids=(1, 2, 3),
        attrition=[],
        n_parent=4,
        n_surviving=3,
    )
    right = SampleEvaluationResult(
        name="andrews2022_modified",
        mode=SampleSelectionMode.FORWARD_MODEL,
        mass_source="pipeline",
        parent_adql="SELECT 1",
        surviving_source_ids=(1, 2, 3, _GAIA_BH1),
        attrition=[],
        n_parent=4,
        n_surviving=4,
    )
    div = compute_mode_divergence(
        left,
        right,
        expected_only_right=[_GAIA_BH1],
    )
    assert div.matches_expectation is True
    assert div.only_right == (_GAIA_BH1,)

    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="mode")
    emission = emit_mode_divergence(
        cfg,
        dirs,
        results={"andrews2022": left, "andrews2022_modified": right},
    )
    assert emission.payload["all_match"] is True
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "Gaia BH1" in text
    assert str(_GAIA_BH1) in text


def test_janssens_segment_occupancy_flags_uninformative(tmp_path: Path) -> None:
    # 4.73 → 1 Msun (0.87–1.55); ~2.6 → near 1.55–1.80; -9 → out of range.
    occupancy = compute_janssens_segment_occupancy([4.73, 2.6, -9.0, float("nan")])
    assert occupancy.n_out_of_range >= 1
    assert occupancy.n_missing_mg >= 1
    assert any(row["count"] > 0 for row in occupancy.segment_counts)

    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="janssens")
    emission = emit_janssens_segment_occupancy(
        cfg, dirs, mg_0_values=[4.73, 2.6, -9.0]
    )
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "n_out_of_range" in text
    assert "1.55" in text and "1.80" in text
    assert "NEAR-UNINFORMATIVE" in text or "uninformative_segment" in text


def test_sample_selection_function_curves(tmp_path: Path) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    dirs = resolve_diagnostic_dirs(cfg, run_id="sf")
    spec = load_sample_selection_file(
        Path(__file__).resolve().parent / "fixtures" / "selections" / "andrews2022_stub.yaml"
    )
    emission = emit_sample_selection_function(
        cfg, dirs, specs={"andrews2022": spec}, sample_names=["andrews2022"]
    )
    assert emission.skipped_reason is None
    assert emission.payload["curves"]
    text = emission.reports[0].read_text(encoding="utf-8")
    assert "axis: m2_msun" in text
    assert "axis: period_day" in text
    assert "axis: g_mag" in text


def test_evaluation_result_roundtrip() -> None:
    original = SampleEvaluationResult(
        name="demo",
        mode=SampleSelectionMode.REPRODUCTION,
        mass_source="paper",
        parent_adql="SELECT 1",
        surviving_source_ids=(10, 20),
        attrition=[
            CutAttrition(
                cut_id="cut_a",
                kind=CutKind.COLUMN,
                n_in=5,
                n_passed=3,
                n_failed=1,
                n_not_applicable=1,
                n_out=3,
                expected_n_after=3,
                not_applicable_reasons={"evolved": 1},
            )
        ],
        n_parent=5,
        n_surviving=2,
        branch_surviving={"astro": (10,)},
        subsample_surviving={"sub1": (10, 20)},
        route_counts={"both": 1},
    )
    restored = sample_evaluation_result_from_dict(original.as_dict())
    assert restored.surviving_source_ids == (10, 20)
    assert restored.attrition[0].n_not_applicable == 1
    assert restored.attrition[0].not_applicable_reasons == {"evolved": 1}
    assert restored.route_counts["both"] == 1


def test_suite_runs_sample_hooks_with_bundle(tmp_path: Path) -> None:
    results = _elbadry_registry().evaluate_all(_catalog_rows())
    cfg = load_config()
    cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)})}
    )
    health = CovarianceHealth()
    health.missing_corr = 2
    bundle = SampleDiagnosticsBundle(
        evaluation_results={
            "elbadry2026": results["elbadry2026"],
            "andrews2022": results["andrews2022"],
        },
        sample_specs={
            "elbadry2026": load_sample_selection_file(
                repo_root() / "config/selections/elbadry2026.yaml"
            ),
            "andrews2022": load_sample_selection_file(
                Path(__file__).resolve().parent
                / "fixtures"
                / "selections"
                / "andrews2022_stub.yaml"
            ),
        },
        covariance_health=health,
        mg_0_values=[4.73, 2.6, -9.0],
        simon_in_sample_ids=_SIMON_IN_SAMPLE,
        simon_rows=load_simon2026_orbital(),
        selection_function_samples=["andrews2022"],
    )
    # Add fake modified set for mode_divergence.
    modified = SampleEvaluationResult(
        name="andrews2022_modified",
        mode=SampleSelectionMode.FORWARD_MODEL,
        mass_source="pipeline",
        parent_adql="SELECT 1",
        surviving_source_ids=tuple(
            sorted(set(results["andrews2022"].surviving_source_ids) | {_GAIA_BH1})
        ),
        attrition=[],
        n_parent=results["andrews2022"].n_parent,
        n_surviving=results["andrews2022"].n_surviving + 1,
    )
    bundle.evaluation_results["andrews2022_modified"] = modified

    suite = run_diagnostic_suite(
        cfg, run_id="sample_suite", sample_bundle=bundle, demo_missing=False, run_sbc=False
    )
    by_name = {h.hook_name: h for h in suite.hooks_run}
    for name in (
        "sample_attrition_waterfall",
        "sample_reproduction_report",
        "simon2026_exclusion_breakdown",
        "covariance_health",
        "sample_selection_function",
        "mode_divergence",
        "janssens_segment_occupancy",
    ):
        assert name in by_name
        assert by_name[name].skipped_reason is None
        assert by_name[name].reports
