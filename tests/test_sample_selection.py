"""Sample-selection framework: registry, mode switch, three-state cuts, depends_on."""

from __future__ import annotations

from datetime import datetime, timezone
from math import log10
from pathlib import Path

import h5py
import pytest
import yaml
from pydantic import ValidationError

from darkhunter_pop.config_loader import (
    config_checksum,
    enabled_selection_content_fingerprint,
    load_config,
    repo_root,
)
from darkhunter_pop.config_schema import (
    SHARED_CHECKSUM_SECTIONS,
    CutKind,
    SampleCut,
    SampleSelectionConfig,
    SampleSelectionEntry,
    SampleSelectionMode,
)
from darkhunter_pop.data_acquisition import (
    FunnelCounts,
    SnapshotMeta,
    compute_stage_diagnostics,
    write_stage_hdf5 as write_da_hdf5,
)
from darkhunter_pop.mass_derivation import write_stage_hdf5 as write_mdb_hdf5
from darkhunter_pop.run_management import (
    SAMPLE_SELECTION_DISABLED_SKIP_REASON,
    SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON,
    STAGE_ORDER,
    STAGE_REGISTRY,
    StageAction,
    create_run_manifest,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
    stage_default_skip_reason,
)
from darkhunter_pop.sample_selection import (
    CutExpressionError,
    CutOutcome,
    NotApplicable,
    SampleEvaluationResult,
    SampleSelection,
    SampleSelectionError,
    SampleSelectionRegistry,
    UnhandledSampleSelectionModeError,
    assert_nonzero_parent_when_da_nonempty,
    candidate_to_selection_row,
    evaluate_cut,
    load_sample_selection_file,
    load_selection_rows_from_manifest,
    mass_source_for_mode,
    parent_query_for_mode,
    read_sample_selection_artifact,
    resolve_inherits,
    run_sample_selection,
    run_sample_selection_stage,
    write_sample_selection_artifact,
)
from darkhunter_pop.schemas import (
    ActiveDRMode,
    CandidateRecord,
    ParameterSet,
    PhotometryPoint,
    StageRecord,
    StageStatus,
)

pytestmark = pytest.mark.unit

_FIXTURES = (
    Path(__file__).resolve().parent / "fixtures" / "selections"
)


def _entry(
    name: str,
    filename: str,
    *,
    enabled: bool = True,
    mode: SampleSelectionMode = SampleSelectionMode.REPRODUCTION,
) -> SampleSelectionEntry:
    return SampleSelectionEntry(
        name=name,
        enabled=enabled,
        path=str(_FIXTURES / filename),
        mode=mode,
    )


def _registry_config(*entries: SampleSelectionEntry) -> SampleSelectionConfig:
    return SampleSelectionConfig(enabled=True, samples=list(entries))


def _pipeline_with_samples(*entries: SampleSelectionEntry):
    cfg = load_config().model_copy(deep=True)
    cfg.sample_selection = _registry_config(*entries)
    return cfg


def _paper_a_rows() -> list[dict]:
    return [
        {
            "source_id": 10,
            "goodness_of_fit": 1.0,
            "m1_tilde_msun": 1.0,
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
        {
            "source_id": 20,
            "goodness_of_fit": 1.0,
            "m1_tilde_msun": NotApplicable("evolved"),
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
        {
            "source_id": 30,
            "goodness_of_fit": 1.0,
            "m1_tilde_msun": NotApplicable("outside_janssens_range"),
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
        {
            "source_id": 40,
            "goodness_of_fit": 1.0,
            "m1_tilde_msun": 0.2,
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
        {
            "source_id": 50,
            "goodness_of_fit": 9.0,
            "m1_tilde_msun": 1.0,
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
        {
            "source_id": 99,
            "goodness_of_fit": 1.0,
            "m1_tilde_msun": 1.0,
            "p_m2_above": 0.99,
            "phot_g_mean_mag": 12.0,
            "pipeline_m1_msun": 2.0,
        },
    ]


def test_mode_dispatch_covers_every_enum_member() -> None:
    assert set(SampleSelectionMode) == {
        SampleSelectionMode.REPRODUCTION,
        SampleSelectionMode.FORWARD_MODEL,
    }
    for mode in SampleSelectionMode:
        mass_source_for_mode(mode)
    assert mass_source_for_mode(SampleSelectionMode.REPRODUCTION) == "paper"
    assert mass_source_for_mode(SampleSelectionMode.FORWARD_MODEL) == "pipeline"


def test_unhandled_mode_raises() -> None:
    with pytest.raises(UnhandledSampleSelectionModeError, match="unhandled"):
        mass_source_for_mode("not_a_mode")  # type: ignore[arg-type]


def test_cut_models_forbid_extra_keys() -> None:
    with pytest.raises(ValidationError):
        SampleCut.model_validate(
            {
                "id": "x",
                "kind": "column",
                "expression": "a > 0",
                "unknown_threshold": 1.4,
            }
        )


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="duplicate names"):
        SampleSelectionConfig(
            enabled=True,
            samples=[
                _entry("paper_a", "paper_a.yaml"),
                _entry("paper_a", "paper_a.yaml"),
            ],
        )


def test_registry_rejects_missing_file_for_enabled_entry() -> None:
    with pytest.raises(ValidationError, match="missing selection files"):
        SampleSelectionConfig(
            enabled=True,
            samples=[
                SampleSelectionEntry(
                    name="ghost",
                    enabled=True,
                    path="config/selections/does_not_exist.yaml",
                    mode=SampleSelectionMode.REPRODUCTION,
                )
            ],
        )


def test_disabled_entry_may_point_at_missing_file() -> None:
    cfg = SampleSelectionConfig(
        enabled=True,
        samples=[
            SampleSelectionEntry(
                name="andrews2022",
                enabled=False,
                path="config/selections/andrews2022.yaml",
                mode=SampleSelectionMode.REPRODUCTION,
            )
        ],
    )
    assert cfg.samples[0].enabled is False


def test_canonical_load_lists_two_andrews_variants() -> None:
    cfg = load_config()
    names = [entry.name for entry in cfg.sample_selection.samples]
    assert names.count("andrews2022") == 1
    assert names.count("andrews2022_modified") == 1
    assert "andrews2022" in names and "andrews2022_modified" in names
    by_name = {entry.name: entry for entry in cfg.sample_selection.samples}
    assert by_name["andrews2022"].mode is SampleSelectionMode.REPRODUCTION
    assert by_name["andrews2022_modified"].mode is SampleSelectionMode.FORWARD_MODEL
    assert by_name["andrews2022"].enabled is True
    assert by_name["andrews2022_modified"].enabled is True
    assert by_name["andrews2022"].path == "config/selections/andrews2022.yaml"
    assert (
        by_name["andrews2022_modified"].path
        == "config/selections/andrews2022_modified.yaml"
    )


def test_checksum_includes_sample_selection() -> None:
    assert "sample_selection" in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    base = config_checksum(cfg)
    flipped = cfg.model_copy(deep=True)
    flipped.sample_selection.enabled = False
    assert config_checksum(flipped) != base


def test_stage_sits_between_bulk_and_followup() -> None:
    order = list(STAGE_ORDER)
    assert order.index("mass_derivation_bulk") < order.index("sample_selection")
    assert order.index("sample_selection") < order.index("selection_function_followup")
    spec = STAGE_REGISTRY["sample_selection"]
    assert spec.module.endswith("sample_selection")
    assert spec.inputs_from == ("mass_derivation_bulk",)
    assert "sample_selection" in spec.config_fingerprint_keys


def test_plan_skips_when_no_enabled_samples() -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.sample_selection.samples = [
        entry.model_copy(update={"enabled": False})
        for entry in cfg.sample_selection.samples
    ]
    manifest = create_run_manifest(cfg)
    entry = plan_stage(STAGE_REGISTRY["sample_selection"], manifest, cfg)
    assert entry.action is StageAction.SKIP_REASON
    assert entry.detail == SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON
    assert (
        stage_default_skip_reason(STAGE_REGISTRY["sample_selection"], cfg)
        == SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON
    )
    disabled = cfg.model_copy(deep=True)
    disabled.sample_selection.enabled = False
    assert (
        stage_default_skip_reason(STAGE_REGISTRY["sample_selection"], disabled)
        == SAMPLE_SELECTION_DISABLED_SKIP_REASON
    )
    entry = plan_stage(STAGE_REGISTRY["sample_selection"], manifest, cfg)
    assert entry.action is StageAction.SKIP_REASON
    assert entry.detail == SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON
    assert (
        stage_default_skip_reason(STAGE_REGISTRY["sample_selection"], cfg)
        == SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON
    )
    disabled = cfg.model_copy(deep=True)
    disabled.sample_selection.enabled = False
    assert (
        stage_default_skip_reason(STAGE_REGISTRY["sample_selection"], disabled)
        == SAMPLE_SELECTION_DISABLED_SKIP_REASON
    )


def test_not_applicable_is_distinct_from_failed() -> None:
    """CONTINUATION_PLAN §15 Q10: evolved / out-of-range ≠ cut failed."""
    spec = load_sample_selection_file(_FIXTURES / "paper_a.yaml")
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    result = selection.evaluate(_paper_a_rows())
    by_id = {row.cut_id: row for row in result.attrition}

    gof = by_id["goodness_of_fit"]
    assert gof.n_in == 6
    assert gof.n_failed == 1
    assert gof.n_not_applicable == 0
    assert gof.n_passed == 5

    floor = by_id["m1_tilde_floor"]
    assert floor.n_in == 5
    assert floor.n_passed == 2
    assert floor.n_failed == 1
    assert floor.n_not_applicable == 2
    assert floor.not_applicable_reasons == {
        "evolved": 1,
        "outside_janssens_range": 1,
    }
    assert floor.n_out == floor.n_passed
    assert floor.n_passed + floor.n_failed + floor.n_not_applicable == floor.n_in

    outcomes_20 = result.outcomes_by_source[20]
    floor_outcome = next(o for o in outcomes_20 if o[0] == "m1_tilde_floor")
    assert floor_outcome[1] is CutOutcome.NOT_APPLICABLE
    outcomes_40 = result.outcomes_by_source[40]
    failed = next(o for o in outcomes_40 if o[0] == "m1_tilde_floor")
    assert failed[1] is CutOutcome.FAILED
    assert result.surviving_source_ids == (10,)


def test_undefined_when_and_requires_defined_are_not_applicable() -> None:
    cut = SampleCut(
        id="m1",
        kind=CutKind.DERIVED,
        expression="m1_tilde_msun > 0.5",
        undefined_when="m1_tilde_msun is not_applicable",
        requires_defined=["m1_tilde_msun"],
        parameters={"unused": 0},
    )
    na, reason = evaluate_cut(
        cut, {"source_id": 1, "m1_tilde_msun": NotApplicable("evolved")}
    )
    assert na is CutOutcome.NOT_APPLICABLE
    assert reason is not None
    failed, _ = evaluate_cut(cut, {"source_id": 1, "m1_tilde_msun": 0.1})
    assert failed is CutOutcome.FAILED
    passed, _ = evaluate_cut(cut, {"source_id": 1, "m1_tilde_msun": 1.0})
    assert passed is CutOutcome.PASSED
    missing, reason_m = evaluate_cut(cut, {"source_id": 1, "m1_tilde_msun": None})
    assert missing is CutOutcome.NOT_APPLICABLE
    assert reason_m is not None


def test_reproduction_uses_paper_mass_forward_model_uses_pipeline() -> None:
    spec = load_sample_selection_file(_FIXTURES / "paper_a.yaml")
    cut = SampleCut(
        id="m1_bound",
        kind=CutKind.DERIVED,
        expression="m1_msun > m1_cut",
        parameters={"m1_cut": 1.5},
    )
    spec = spec.model_copy(update={"cuts": [cut], "exclusions": []})
    row = {
        "source_id": 1,
        "pipeline_m1_msun": 2.0,
        "paper_m1_msun": 1.0,
    }
    repro = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION).evaluate(
        [row]
    )
    fwd = SampleSelection(spec, mode=SampleSelectionMode.FORWARD_MODEL).evaluate(
        [row]
    )
    assert repro.mass_source == "paper"
    assert fwd.mass_source == "pipeline"
    assert repro.surviving_source_ids == ()
    assert fwd.surviving_source_ids == (1,)


def test_inherits_restores_exclusion_as_independent_named_sample() -> None:
    paper_a = load_sample_selection_file(_FIXTURES / "paper_a.yaml")
    modified = load_sample_selection_file(_FIXTURES / "paper_a_modified.yaml")
    resolved = resolve_inherits(
        modified, files_by_name={"paper_a": paper_a, "paper_a_modified": modified}
    )
    assert resolved.cuts is not None
    assert [c.id for c in resolved.cuts] == [c.id for c in (paper_a.cuts or [])]
    assert resolved.exclusions == []
    assert paper_a.exclusions
    assert paper_a.exclusions[0].source_id == 99

    registry = SampleSelectionRegistry(
        _registry_config(
            _entry("paper_a", "paper_a.yaml"),
            _entry(
                "paper_a_modified",
                "paper_a_modified.yaml",
                mode=SampleSelectionMode.FORWARD_MODEL,
            ),
        )
    )
    results = registry.evaluate_all(_paper_a_rows())
    assert results["paper_a"].surviving_source_ids == (10,)
    assert results["paper_a_modified"].surviving_source_ids == (10, 99)
    assert results["paper_a"].mode is SampleSelectionMode.REPRODUCTION
    assert results["paper_a_modified"].mode is SampleSelectionMode.FORWARD_MODEL


def test_depends_on_threads_published_membership_not_modified() -> None:
    registry = SampleSelectionRegistry(
        _registry_config(
            _entry("paper_a", "paper_a.yaml"),
            _entry(
                "paper_a_modified",
                "paper_a_modified.yaml",
                mode=SampleSelectionMode.FORWARD_MODEL,
            ),
            _entry("paper_b", "paper_b.yaml"),
        )
    )
    assert registry.evaluation_order() == [
        "paper_a",
        "paper_a_modified",
        "paper_b",
    ]
    results = registry.evaluate_all(_paper_a_rows())
    # paper_b imports paper_a (N=1), never the modified variant (which keeps 99).
    assert results["paper_b"].surviving_source_ids == (10,)
    assert 99 not in results["paper_b"].surviving_source_ids
    assert 99 in results["paper_a_modified"].surviving_source_ids


def test_depends_on_missing_enabled_dependency_raises() -> None:
    registry = SampleSelectionRegistry(
        _registry_config(
            _entry("paper_a", "paper_a.yaml", enabled=False),
            _entry("paper_b", "paper_b.yaml"),
        )
    )
    with pytest.raises(SampleSelectionError, match="depends_on"):
        registry.evaluation_order()


def test_parent_adql_is_dr_specific() -> None:
    spec = load_sample_selection_file(_FIXTURES / "paper_a.yaml")
    dr3 = parent_query_for_mode(spec, ActiveDRMode.DR3)
    dr4 = parent_query_for_mode(spec, ActiveDRMode.DR4)
    assert "gaiadr3.nss_two_body_orbit" in dr3.adql
    assert "gaiadr4.nss_two_body_orbit" in dr4.adql
    assert dr3.nss_table != dr4.nss_table
    selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    assert selection.parent_adql() == dr3.adql


def test_reproduction_only_cut_skipped_in_forward_model() -> None:
    spec = load_sample_selection_file(_FIXTURES / "paper_a.yaml")
    cuts = list(spec.cuts or [])
    cuts.append(
        SampleCut(
            id="repro_only",
            kind=CutKind.COLUMN,
            expression="source_id != 10",
            applies_to=[SampleSelectionMode.REPRODUCTION],
        )
    )
    spec = spec.model_copy(update={"cuts": cuts, "exclusions": []})
    rows = _paper_a_rows()
    repro = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION).evaluate(rows)
    fwd = SampleSelection(spec, mode=SampleSelectionMode.FORWARD_MODEL).evaluate(rows)
    assert "repro_only" in {row.cut_id for row in repro.attrition}
    assert "repro_only" not in {row.cut_id for row in fwd.attrition}
    assert 10 not in repro.surviving_source_ids
    assert 10 in fwd.surviving_source_ids


def test_artifact_path_keyed_on_enabled_file_hash_and_mode(tmp_path: Path) -> None:
    src = (_FIXTURES / "paper_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "paper_a.yaml"
    path.write_text(src, encoding="utf-8")
    cfg = _pipeline_with_samples(
        SampleSelectionEntry(
            name="paper_a",
            enabled=True,
            path=str(path),
            mode=SampleSelectionMode.REPRODUCTION,
        )
    )
    cfg.paths.artifact_root = str(tmp_path / "output")
    spec = STAGE_REGISTRY["sample_selection"]
    p1 = stage_artifact_path(cfg, spec, run_id="runA")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["cuts"][0]["parameters"]["goodness_of_fit_max"] = 4.0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    p2 = stage_artifact_path(cfg, spec, run_id="runA")
    assert p1 != p2
    cfg_mode = cfg.model_copy(deep=True)
    cfg_mode.sample_selection.samples[0] = cfg_mode.sample_selection.samples[0].model_copy(
        update={"mode": SampleSelectionMode.FORWARD_MODEL}
    )
    path.write_text(src, encoding="utf-8")
    p3 = stage_artifact_path(cfg_mode, spec, run_id="runA")
    assert p3 != p1
    h1 = enabled_selection_content_fingerprint(cfg, repo=tmp_path)
    assert len(h1) == 64


@pytest.mark.api
def test_hdf5_producer_consumer_round_trip(tmp_path: Path) -> None:
    cfg = _pipeline_with_samples(
        _entry("paper_a", "paper_a.yaml"),
        _entry(
            "paper_a_modified",
            "paper_a_modified.yaml",
            mode=SampleSelectionMode.FORWARD_MODEL,
        ),
        _entry("paper_b", "paper_b.yaml"),
    )
    result = run_sample_selection(_paper_a_rows(), cfg, repo=repo_root())
    path = tmp_path / "sample_selection.h5"
    write_sample_selection_artifact(path, result)
    payload = read_sample_selection_artifact(path)
    assert payload["schema_version"] == 1
    assert payload["results"]["paper_a"]["surviving_source_ids"] == [10]
    assert payload["results"]["paper_a_modified"]["surviving_source_ids"] == [10, 99]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["stage"] == "sample_selection"
        assert "source_id" in handle["samples/paper_a"]
        floor = handle["samples/paper_a/attrition/m1_tilde_floor"]
        assert int(floor.attrs["n_not_applicable"]) == 2
        assert int(floor.attrs["n_failed"]) == 1


def test_stage_runner_skip_and_run(tmp_path: Path) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    cfg.sample_selection.samples = [
        entry.model_copy(update={"enabled": False})
        for entry in cfg.sample_selection.samples
    ]
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)
    skipped = run_sample_selection_stage(manifest, cfg, run_path=run_path)
    record = skipped.stages["sample_selection"]
    assert record.status is StageStatus.SKIPPED
    assert record.reason == SAMPLE_SELECTION_NO_SAMPLES_SKIP_REASON

    enabled = _pipeline_with_samples(_entry("paper_a", "paper_a.yaml"))
    enabled.paths.artifact_root = str(tmp_path / "output")
    manifest2 = create_run_manifest(enabled)
    run_path2 = tmp_path / f"{manifest2.run_id}.yaml"
    save_run_manifest(manifest2, run_path2)
    finished = run_sample_selection_stage(
        manifest2, enabled, run_path=run_path2, rows=_paper_a_rows()
    )
    rec = finished.stages["sample_selection"]
    assert rec.status is StageStatus.COMPLETED
    assert rec.artifact_path is not None
    assert Path(rec.artifact_path).is_file()


def test_illegal_expression_rejected() -> None:
    cut = SampleCut(
        id="bad",
        kind=CutKind.COLUMN,
        expression="__import__('os').system('pwd')",
    )
    with pytest.raises(CutExpressionError):
        evaluate_cut(cut, {"source_id": 1})


def test_candidate_to_selection_row_flattens_orbital_photometry_masses() -> None:
    cand = CandidateRecord(
        source_id=42,
        nss_solution_type="Orbital",
        parallax_mas=10.0,
        nss_orbital={"goodness_of_fit": 1.5, "period": 200.0, "parallax": 10.0},
        extras={"logg_gspphot": 4.1},
        photometry=[
            PhotometryPoint(band="G", mag=12.0),
            PhotometryPoint(band="BP", mag=12.5),
            PhotometryPoint(band="RP", mag=11.5),
        ],
        m1=ParameterSet(
            names=["M1", "R1"],
            values=[1.2, 1.1],
            covariance=[[0.01, 0.0], [0.0, 0.02]],
            provenance="test",
            units=["Msun", "Rsun"],
        ),
        m2=ParameterSet(
            names=["M2"],
            values=[2.0],
            covariance=[[0.25]],
            provenance="test",
            units=["Msun"],
        ),
    )
    row = candidate_to_selection_row(cand)
    assert row["source_id"] == 42
    assert row["nss_solution_type"] == "Orbital"
    assert row["goodness_of_fit"] == pytest.approx(1.5)
    assert row["period_day"] == pytest.approx(200.0)
    assert row["logg_apsis"] == pytest.approx(4.1)
    assert row["phot_g_mean_mag"] == pytest.approx(12.0)
    assert row["bp_rp"] == pytest.approx(1.0)
    assert row["abs_g_mag"] == pytest.approx(12.0 + 5.0 * log10(10.0) - 10.0)
    assert row["pipeline_m1_msun"] == pytest.approx(1.2)
    assert row["m2_msun"] == pytest.approx(2.0)
    assert row["m2_msun_error"] == pytest.approx(0.5)
    assert row["m2_snr"] == pytest.approx(4.0)


def test_stage_loads_da_rows_when_rows_none(tmp_path: Path) -> None:
    """Regression #129: empty ``rows or ()`` must not silently yield n_parent=0."""
    da_cand = CandidateRecord(
        source_id=10,
        nss_solution_type="Orbital",
        parallax_mas=5.0,
        nss_orbital={"goodness_of_fit": 1.0, "period": 100.0, "parallax": 5.0},
        extras={"logg_gspphot": 4.0},
        photometry=[
            PhotometryPoint(band="G", mag=13.0),
            PhotometryPoint(band="BP", mag=13.4),
            PhotometryPoint(band="RP", mag=12.6),
        ],
    )
    mdb_cand = da_cand.model_copy(
        update={
            "m1": ParameterSet(
                names=["M1", "R1"],
                values=[1.1, 1.0],
                covariance=[[0.01, 0.0], [0.0, 0.01]],
                provenance="test",
                units=["Msun", "Rsun"],
            ),
            "m2": ParameterSet(
                names=["M2"],
                values=[1.8],
                covariance=[[0.04]],
                provenance="test",
                units=["Msun"],
            ),
        }
    )

    enabled = _pipeline_with_samples(_entry("paper_a", "paper_a.yaml"))
    enabled.paths.artifact_root = str(tmp_path / "output")
    manifest = create_run_manifest(enabled)
    run_path = tmp_path / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    da_path = tmp_path / "da.h5"
    snapshot = SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=1,
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )
    funnel = FunnelCounts(queried=1, after_quality_cut=1, candidates_written=1)
    diagnostics = compute_stage_diagnostics(
        [da_cand], funnel=funnel, quality_cut_bin_counts={"bin0": 1}
    )
    write_da_hdf5(da_path, [da_cand], snapshot=snapshot, diagnostics=diagnostics)
    mdb_path = tmp_path / "mdb.h5"
    write_mdb_hdf5(
        mdb_path,
        [mdb_cand],
        stage_name="mass_derivation_bulk",
        diagnostics={"n_input": 1, "n_kept": 1},
    )

    stages = dict(manifest.stages)
    stages["data_acquisition"] = StageRecord(
        stage_name="data_acquisition",
        status=StageStatus.COMPLETED,
        artifact_path=str(da_path),
    )
    stages["mass_derivation_bulk"] = StageRecord(
        stage_name="mass_derivation_bulk",
        status=StageStatus.COMPLETED,
        artifact_path=str(mdb_path),
    )
    manifest = manifest.model_copy(update={"stages": stages})
    save_run_manifest(manifest, run_path)

    rows = load_selection_rows_from_manifest(manifest)
    assert len(rows) == 1
    assert rows[0]["source_id"] == 10
    assert rows[0]["pipeline_m1_msun"] == pytest.approx(1.1)
    assert rows[0]["m2_msun"] == pytest.approx(1.8)

    finished = run_sample_selection_stage(manifest, enabled, run_path=run_path)
    rec = finished.stages["sample_selection"]
    assert rec.status is StageStatus.COMPLETED
    assert rec.artifact_path is not None
    payload = read_sample_selection_artifact(Path(rec.artifact_path))
    assert payload["results"]["paper_a"]["n_parent"] == 1
    assert payload["results"]["paper_a"]["n_parent"] > 0


def test_assert_nonzero_parent_fails_loud_when_da_nonempty() -> None:
    from darkhunter_pop.sample_selection import SampleSelectionStageResult

    empty = SampleSelectionStageResult(
        schema_version=1,
        enabled=True,
        content_fingerprint="x",
        results={
            "paper_a": SampleEvaluationResult(
                name="paper_a",
                mode=SampleSelectionMode.REPRODUCTION,
                mass_source="paper",
                parent_adql="SELECT 1",
                surviving_source_ids=(),
                attrition=[],
                n_parent=0,
                n_surviving=0,
            )
        },
    )
    with pytest.raises(SampleSelectionError, match="parent N==0"):
        assert_nonzero_parent_when_da_nonempty(empty, n_da_rows=100)
