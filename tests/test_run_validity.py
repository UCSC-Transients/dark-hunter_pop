"""Stage reports (#331), consumer-side gate policy (#352) and point-of-use stand-ins (#354).

These drive the real stage runners against ``tmp_path`` artifact roots. The
astrometric selection-function science needs gaiamock, so that one stage's science
call is replaced by a stub that writes a genuine artifact through
``forward_model.write_selection_function_artifact``; everything the wrapper does
after the science call (stand-ins, report) runs for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import pytest
import yaml

from darkhunter_pop import pipeline
from darkhunter_pop.companion_nature import (
    CompanionNatureDiagnostics,
    companion_nature_stand_ins,
)
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.forward_model import (
    SIX_PANEL_NAMES,
    SOLUTION_TYPE_LABELS,
    SelectionFunctionAstrometricResult,
    SixPanelValidationResult,
    SolutionTypeFractionResult,
    ValidationGateResult,
    write_selection_function_artifact,
)
from darkhunter_pop.gaiamock_vendor import GaiamockModVersions
from darkhunter_pop.inference import load_sample_membership, run_inference_stage
from darkhunter_pop.population_model import run_population_model_stage
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    load_run_manifest,
    mark_stage_finished,
    mark_stage_started,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.run_validity import (
    FOLLOWUP_CALIBRATION_STATUS_ATTR,
    STAND_INS_ATTR,
    UpstreamGateFailedError,
    UpstreamGateStatus,
    assess_science_validity,
    astrometric_gate_status,
    collect_stand_ins,
    followup_gate_status,
    merge_stand_ins,
    read_stand_ins,
    write_stand_ins,
)
from darkhunter_pop.sample_selection import SampleEvaluationResult, SampleSelectionMode
from darkhunter_pop.schemas import RunManifest, StageStatus, SyntheticStandIn
from darkhunter_pop.sensitivity_analysis import run_sensitivity_analysis_stage

pytestmark = pytest.mark.unit


def _cfg(tmp_path: Path, **inference_updates: Any) -> PipelineConfig:
    cfg = load_config()
    icfg = cfg.inference.model_copy(update={"skip_sampler": True, **inference_updates})
    return cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"artifact_root": str(tmp_path)}),
            "inference": icfg,
        }
    )


def _fresh_run(cfg: PipelineConfig, tmp_path: Path) -> tuple[RunManifest, Path]:
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)
    return manifest, run_path


def _reports_dir(artifact: Path) -> Path:
    return artifact.parent / f"{artifact.stem}_diagnostics" / "reports"


def _astrometric_result(*, passed: bool) -> SelectionFunctionAstrometricResult:
    frac = {label: 1.0 / len(SOLUTION_TYPE_LABELS) for label in SOLUTION_TYPE_LABELS}
    validation = ValidationGateResult(
        six_panel=SixPanelValidationResult(
            panel_names=SIX_PANEL_NAMES,
            ks_pvalues={n: (0.5 if passed else 1e-9) for n in SIX_PANEL_NAMES},
            ks_statistics={n: 0.1 for n in SIX_PANEL_NAMES},
            passed={n: passed for n in SIX_PANEL_NAMES},
        ),
        solution_type=SolutionTypeFractionResult(
            mock_fractions=frac,
            real_fractions=frac,
            max_abs_delta=0.0,
            passed=True,
        ),
        detection_fraction=0.026,
        n_mock=0,
        n_real=0,
    )
    return SelectionFunctionAstrometricResult(
        gaiamock_versions=GaiamockModVersions(
            gaiamock_mod_release="gaiamock-mod-v1",
            gaiamock_mod_sha256="a" * 64,
            gaiamock_git_commit="b" * 40,
        ),
        records=[],
        validation=validation,
        data_release="dr3",
    )


def _stub_astrometric(passed: bool):
    def _run(config: PipelineConfig, artifact_path: Path, **_: Any):
        result = _astrometric_result(passed=passed)
        write_selection_function_artifact(artifact_path, result)
        return result

    return _run


# ---------------------------------------------------------------------------
# #331 + #354: forward-model stage wrappers
# ---------------------------------------------------------------------------


def test_astrometric_stage_writes_report_and_registers_stand_ins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        pipeline, "run_selection_function_astrometric", _stub_astrometric(False)
    )
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = pipeline.STAGE_RUNNERS["selection_function_astrometric"](
        manifest, cfg, run_path=run_path
    )
    artifact = Path(manifest.stages["selection_function_astrometric"].artifact_path)
    report = _reports_dir(artifact) / "validation_gate_report.txt"
    assert report.is_file()
    text = report.read_text()
    assert "VALIDATION GATE FAILED" in text
    assert "overall_passed: False" in text
    names = {s.name for s in read_stand_ins(artifact) or []}
    assert {"box_prior_mock_population", "mock_insufficient_visibility_fraction"} <= names


def test_followup_stage_reports_not_calibrated_without_real_catalog(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.selection_function_followup.calibration.real_followup_catalog_path is None
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = pipeline.STAGE_RUNNERS["selection_function_followup"](
        manifest, cfg, run_path=run_path
    )
    artifact = Path(manifest.stages["selection_function_followup"].artifact_path)
    with h5py.File(artifact, "r") as handle:
        assert handle.attrs[FOLLOWUP_CALIBRATION_STATUS_ATTR] == "not_calibrated"
    report = (_reports_dir(artifact) / "followup_calibration_report.txt").read_text()
    assert "NOT CALIBRATED" in report
    names = {s.name for s in read_stand_ins(artifact) or []}
    assert {"synthetic_followup_catalog", "synthetic_followup_calibration_twin"} <= names
    status = followup_gate_status(artifact)
    assert status.status == "not_calibrated"
    assert not status.passed


# ---------------------------------------------------------------------------
# #331 + #354: population / sensitivity
# ---------------------------------------------------------------------------


def test_population_model_stage_writes_report(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = run_population_model_stage(
        manifest, cfg, run_path=run_path, candidates=[]
    )
    artifact = Path(manifest.stages["population_model"].artifact_path)
    report = _reports_dir(artifact) / "population_model_report.txt"
    assert report.is_file()
    assert "population_model report" in report.read_text()


def test_sensitivity_stage_writes_report_and_registers_synthetic_paths(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = run_sensitivity_analysis_stage(manifest, cfg, run_path=run_path)
    artifact = Path(manifest.stages["sensitivity_analysis"].artifact_path)
    assert (_reports_dir(artifact) / "sensitivity_analysis_report.txt").is_file()
    names = {s.name for s in read_stand_ins(artifact) or []}
    assert "analytic_mc_noise_identity" in names
    assert "synthetic_fiducial_sensitivity_catalog" in names
    with h5py.File(artifact, "r") as handle:
        assert handle.attrs["catalog_source"] == "synthetic_fiducial"


def test_reports_respect_write_reports_switch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = cfg.model_copy(
        update={"diagnostics": cfg.diagnostics.model_copy(update={"write_reports": False})}
    )
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = run_sensitivity_analysis_stage(manifest, cfg, run_path=run_path)
    artifact = Path(manifest.stages["sensitivity_analysis"].artifact_path)
    assert not (_reports_dir(artifact) / "sensitivity_analysis_report.txt").exists()


def test_companion_nature_registers_analytic_fallbacks() -> None:
    diag = CompanionNatureDiagnostics(
        n_input=3, n_by_evidence_provenance={"analytic_fallback": 3}, track_source=None
    )
    names = {s.name for s in companion_nature_stand_ins(diag)}
    assert names == {"analytic_companion_nature_evidence", "analytic_wd_cooling_fallback"}
    clean = CompanionNatureDiagnostics(
        n_input=3, n_by_evidence_provenance={"phot_sed": 3}, track_source="tracks.csv"
    )
    assert companion_nature_stand_ins(clean) == []


# ---------------------------------------------------------------------------
# #352: inference consumer side
# ---------------------------------------------------------------------------


def _manifest_with_astrometric(
    cfg: PipelineConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool,
) -> tuple[RunManifest, Path]:
    monkeypatch.setattr(
        pipeline, "run_selection_function_astrometric", _stub_astrometric(passed)
    )
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = pipeline.STAGE_RUNNERS["selection_function_astrometric"](
        manifest, cfg, run_path=run_path
    )
    manifest = pipeline.STAGE_RUNNERS["selection_function_followup"](
        manifest, cfg, run_path=run_path
    )
    return manifest, run_path


def test_inference_marks_run_not_science_valid_on_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, upstream_gate_policy="mark_not_science_valid")
    manifest, run_path = _manifest_with_astrometric(
        cfg, tmp_path, monkeypatch, passed=False
    )
    manifest = run_inference_stage(manifest, cfg, run_path=run_path)
    assert manifest.stages["inference"].status is StageStatus.COMPLETED
    assert manifest.science_valid is False
    reasons = "\n".join(manifest.science_validity_reasons)
    assert "selection_function_astrometric.validation_gate: failed" in reasons
    assert "selection_function_followup.calibration: not_calibrated" in reasons
    reloaded = load_run_manifest(run_path)
    assert reloaded.science_valid is False

    artifact = Path(manifest.stages["inference"].artifact_path)
    with h5py.File(artifact, "r") as handle:
        assert bool(handle.attrs["science_valid"]) is False
        assert handle.attrs["astrometric_sf"] == pytest.approx(0.026)
        assert STAND_INS_ATTR in handle.attrs
    report = (_reports_dir(artifact) / "inference_report.txt").read_text()
    assert "science_valid: False" in report
    # SF actually used is reported, with its source — not the config default.
    assert "astrometric_sf: 0.026" in report


def test_inference_refuses_on_failed_gate_under_refuse_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, upstream_gate_policy="refuse")
    manifest, run_path = _manifest_with_astrometric(
        cfg, tmp_path, monkeypatch, passed=False
    )
    with pytest.raises(UpstreamGateFailedError, match="validation_gate: failed"):
        run_inference_stage(manifest, cfg, run_path=run_path)
    reloaded = load_run_manifest(run_path)
    assert reloaded.stages["inference"].status is StageStatus.FAILED
    assert reloaded.science_valid is False
    assert not stage_artifact_path(
        cfg, STAGE_REGISTRY["inference"], run_id=reloaded.run_id
    ).exists()


def test_inference_without_upstream_artifacts_is_not_science_valid(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = run_inference_stage(manifest, cfg, run_path=run_path)
    assert manifest.science_valid is False
    assert any("not_run" in r for r in manifest.science_validity_reasons)


def test_no_policy_claims_science_valid_on_a_failed_gate() -> None:
    failed = UpstreamGateStatus("selection_function_astrometric", "validation_gate", "failed", "x")
    for policy in ("refuse", "mark_not_science_valid"):
        verdict = assess_science_validity(policy=policy, gates=[failed])  # type: ignore[arg-type]
        assert verdict.science_valid is False
    unread = UpstreamGateStatus("selection_function_followup", "calibration", "unreadable", "x")
    assert assess_science_validity(policy="mark_not_science_valid", gates=[unread]).science_valid is False
    ok = UpstreamGateStatus("selection_function_astrometric", "validation_gate", "passed", "x")
    assert assess_science_validity(policy="mark_not_science_valid", gates=[ok]).science_valid is True


def test_astrometric_gate_status_reads_attr(tmp_path: Path) -> None:
    path = tmp_path / "sf.h5"
    write_selection_function_artifact(path, _astrometric_result(passed=True))
    assert astrometric_gate_status(path).passed
    write_selection_function_artifact(path, _astrometric_result(passed=False))
    assert astrometric_gate_status(path).status == "failed"
    assert astrometric_gate_status(None).status == "not_run"


def test_load_sample_membership_reads_inference_source_ids(tmp_path: Path) -> None:
    def _res(name: str, ids: tuple[int, ...]) -> dict[str, Any]:
        return SampleEvaluationResult(
            name=name,
            mode=SampleSelectionMode.FORWARD_MODEL,
            mass_source="test",
            parent_adql="",
            surviving_source_ids=ids,
            attrition=[],
            n_parent=10,
            n_surviving=len(ids),
            inference_source_ids=ids,
        ).as_dict()

    h5 = tmp_path / "ss.h5"
    h5.write_bytes(b"")
    sidecar = {
        "results": {
            "andrews2022_modified": _res("andrews2022_modified", (5, 6, 7)),
            "elbadry2024": _res("elbadry2024", (6, 7)),
            "andrews2022": _res("andrews2022", (99,)),
        }
    }
    h5.with_suffix(".yaml").write_text(yaml.safe_dump(sidecar))
    membership = load_sample_membership(
        h5, sample_names=["andrews2022_modified", "elbadry2024", "elbadry2026"]
    )
    assert membership == {"andrews2022_modified": [5, 6, 7], "elbadry2024": [6, 7]}
    assert load_sample_membership(None, sample_names=["elbadry2024"]) is None


# ---------------------------------------------------------------------------
# #354: collection
# ---------------------------------------------------------------------------


def _si(name: str, value: int = 0) -> SyntheticStandIn:
    return SyntheticStandIn(
        name=name,
        stage="inference",
        kind="config_placeholder",
        replaces="x",
        description="y",
        values={"v": value},
    )


def test_merge_stand_ins_later_wins_keeps_order() -> None:
    merged = merge_stand_ins([_si("a", 1), _si("b", 1)], [_si("a", 2)])
    assert [s.name for s in merged] == ["a", "b"]
    assert merged[0].values == {"v": 2}


def test_collect_flags_registering_stage_without_attribute(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    manifest = create_run_manifest(cfg)
    reg = tmp_path / "sens.h5"
    with h5py.File(reg, "w") as handle:
        handle.attrs["stage"] = "sensitivity_analysis"
    clean = tmp_path / "pop.h5"
    with h5py.File(clean, "w") as handle:
        handle.attrs["stage"] = "population_model"
    registered = tmp_path / "inf.h5"
    with h5py.File(registered, "w") as handle:
        handle.attrs["stage"] = "inference"
    write_stand_ins(registered, [_si("x")])
    for name, path in (
        ("sensitivity_analysis", reg),
        ("population_model", clean),
        ("inference", registered),
    ):
        manifest = mark_stage_started(manifest, STAGE_REGISTRY[name], cfg)
        manifest = mark_stage_finished(
            manifest, STAGE_REGISTRY[name], status=StageStatus.COMPLETED, artifact_path=path
        )
    collected = collect_stand_ins(manifest)
    assert [s.name for s in collected.stand_ins] == ["x"]
    # sensitivity_analysis registers; an artifact without the attr is flagged.
    assert collected.unregistered_stages == ["sensitivity_analysis"]


def test_followup_gate_without_status_attr_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "fu.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["calibration_passed"] = True
    status = followup_gate_status(path)
    assert status.status == "unreadable"
    assert not status.passed


# ---------------------------------------------------------------------------
# #349 / #350 / #356: honesty labels
# ---------------------------------------------------------------------------


def test_sensitivity_report_labels_synthetic_catalog_and_analytic_mc(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    manifest, run_path = _fresh_run(cfg, tmp_path)
    manifest = run_sensitivity_analysis_stage(manifest, cfg, run_path=run_path)
    artifact = Path(manifest.stages["sensitivity_analysis"].artifact_path)
    text = (_reports_dir(artifact) / "sensitivity_analysis_report.txt").read_text()
    assert "SYNTHETIC CATALOG — NOT A PIPELINE TEST" in text
    assert "ANALYTIC PLACEHOLDER — NOT MEASURED" in text


def test_sbc_analytic_backend_is_not_a_pipeline_validation() -> None:
    from darkhunter_pop.sbc import SBC_ANALYTIC_BANNER, sbc_validates_pipeline

    assert sbc_validates_pipeline("analytic_binned") is False
    assert sbc_validates_pipeline("dynesty") is True
    assert "does NOT validate inference" in SBC_ANALYTIC_BANNER
