"""Tests for the Wave 0 dry-run harness (issue #201).

Covers the labeling contract (a dry run cannot exist unlabeled or with its
stand-ins unenumerated), the run-plan printout, per-stage cost recording, the
snapshot-discovery guard, the ``dN/dM``-by-class product primitive, and the
gate predicates the CLI exits on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.dry_run import (
    DNDM_CAPTION_NAME,
    DRY_RUN_LABEL,
    DRY_RUN_REPORT_NAME,
    DRY_RUN_SUBDIR,
    DnDmFigureInputs,
    DryRunResult,
    StageCost,
    build_dry_run_manifest,
    caption_text,
    declare_stand_ins,
    dry_run_dirs,
    dry_run_seeds,
    format_dry_run_report,
    instrumented_runners,
    latest_gaia_snapshot_meta,
)
from darkhunter_pop.plotting import matplotlib_available, plot_dndm_by_class
from darkhunter_pop.run_management import (
    STAGE_ORDER,
    STAGE_REGISTRY,
    create_run_manifest,
    format_run_plan,
    load_run_manifest,
    mark_stage_finished,
    mark_stage_started,
    new_run_for_force_rerun,
    plan_stage,
    record_stage_resources,
    save_run_manifest,
)
from darkhunter_pop.schemas import (
    STAND_IN_KINDS,
    RunManifest,
    StageStatus,
    SyntheticStandIn,
)

pytestmark = pytest.mark.unit


def _stand_in(name: str = "demo") -> SyntheticStandIn:
    return SyntheticStandIn(
        name=name,
        stage="inference",
        kind="config_placeholder",
        replaces="a real thing",
        description="Placeholder used only in tests.",
        config_keys=["inference.nlive"],
        values={"nlive": 20},
    )


# ---------------------------------------------------------------------------
# Labeling contract
# ---------------------------------------------------------------------------


def test_stand_in_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown stand-in kind"):
        SyntheticStandIn(
            name="x",
            stage="inference",
            kind="vibes",
            replaces="y",
            description="z",
        )


def test_stand_in_one_line_names_kind_and_stage() -> None:
    line = _stand_in("per_sample_selection_weights").one_line()
    assert "per_sample_selection_weights" in line
    assert "config_placeholder" in line
    assert "inference" in line


def test_dry_run_manifest_requires_label() -> None:
    with pytest.raises(ValueError, match="dry_run_label"):
        RunManifest(
            run_id="r",
            created_at=datetime.now(tz=timezone.utc),
            config_checksum="c",
            dry_run=True,
            synthetic_stand_ins=[_stand_in()],
        )


def test_dry_run_manifest_requires_at_least_one_stand_in() -> None:
    with pytest.raises(ValueError, match="synthetic_stand_ins"):
        RunManifest(
            run_id="r",
            created_at=datetime.now(tz=timezone.utc),
            config_checksum="c",
            dry_run=True,
            dry_run_label=DRY_RUN_LABEL,
        )


def test_non_dry_run_needs_neither() -> None:
    manifest = RunManifest(
        run_id="r",
        created_at=datetime.now(tz=timezone.utc),
        config_checksum="c",
    )
    assert manifest.dry_run is False
    assert manifest.synthetic_stand_ins == []


def test_dry_run_flag_survives_yaml_round_trip(tmp_path: Path) -> None:
    config = load_config()
    manifest = create_run_manifest(
        config,
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=[_stand_in()],
        random_seeds={"inference.random_seed": 63},
    )
    path = save_run_manifest(manifest, tmp_path / f"{manifest.run_id}.yaml")
    # Distinguishable without parsing: the tag is greppable in the YAML text.
    assert "dry_run: true" in path.read_text(encoding="utf-8")
    reloaded = load_run_manifest(path)
    assert reloaded.dry_run is True
    assert reloaded.dry_run_label == DRY_RUN_LABEL
    assert [s.name for s in reloaded.synthetic_stand_ins] == ["demo"]
    assert reloaded.random_seeds == {"inference.random_seed": 63}


def test_force_rerun_child_stays_a_dry_run() -> None:
    config = load_config()
    parent = create_run_manifest(
        config,
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=[_stand_in()],
        random_seeds={"inference.random_seed": 63},
    )
    child = new_run_for_force_rerun(parent, config, "inference")
    assert child.dry_run is True
    assert child.dry_run_label == DRY_RUN_LABEL
    assert [s.name for s in child.synthetic_stand_ins] == ["demo"]
    assert child.random_seeds == parent.random_seeds
    assert child.parent_run_id == parent.run_id


# ---------------------------------------------------------------------------
# Run plan printout
# ---------------------------------------------------------------------------


def test_run_plan_prints_banner_and_every_stand_in(tmp_path: Path) -> None:
    config = load_config()
    stand_ins = [_stand_in("alpha"), _stand_in("beta")]
    manifest = create_run_manifest(
        config,
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=stand_ins,
        random_seeds={"inference.random_seed": 63},
    )
    plan = [plan_stage(STAGE_REGISTRY["inference"], manifest, config)]
    text = format_run_plan(
        manifest,
        config,
        plan,
        run_path=tmp_path / "r.yaml",
        created_new=True,
    )
    assert DRY_RUN_LABEL in text
    assert "synthetic_stand_ins: 2 declared" in text
    for stand_in in stand_ins:
        assert stand_in.name in text
        assert stand_in.description in text
    assert "inference.nlive" in text
    assert "dry_run: True" in text
    assert "gaiamock version triple:" in text
    assert "code_commit:" in text


def test_run_plan_for_a_science_run_declares_zero_stand_ins(tmp_path: Path) -> None:
    config = load_config()
    manifest = create_run_manifest(config)
    plan = [plan_stage(STAGE_REGISTRY["inference"], manifest, config)]
    text = format_run_plan(
        manifest, config, plan, run_path=tmp_path / "r.yaml", created_new=True
    )
    assert "synthetic_stand_ins: 0 declared" in text
    assert DRY_RUN_LABEL not in text


# ---------------------------------------------------------------------------
# Per-stage cost recording
# ---------------------------------------------------------------------------


def test_record_stage_resources_annotates_the_record() -> None:
    config = load_config()
    spec = STAGE_REGISTRY["inference"]
    manifest = create_run_manifest(config)
    manifest = mark_stage_started(manifest, spec, config)
    manifest = mark_stage_finished(manifest, spec, status=StageStatus.COMPLETED)
    updated = record_stage_resources(
        manifest,
        "inference",
        wall_clock_seconds=1.25,
        peak_rss_bytes=3 << 30,
        cumulative_peak_rss_bytes=4 << 30,
    )
    record = updated.stages["inference"]
    assert record.wall_clock_seconds == pytest.approx(1.25)
    assert record.peak_rss_bytes == 3 << 30
    assert record.cumulative_peak_rss_bytes == 4 << 30
    # Original untouched (manifests are copied, never mutated).
    assert manifest.stages["inference"].wall_clock_seconds is None


def test_record_stage_resources_rejects_unknown_stage() -> None:
    config = load_config()
    manifest = create_run_manifest(config)
    with pytest.raises(KeyError):
        record_stage_resources(manifest, "inference", wall_clock_seconds=1.0)


def test_unmeasured_stage_keeps_null_cost_fields() -> None:
    config = load_config()
    spec = STAGE_REGISTRY["triples"]
    manifest = create_run_manifest(config)
    manifest = mark_stage_started(manifest, spec, config)
    manifest = mark_stage_finished(
        manifest, spec, status=StageStatus.SKIPPED, reason="triples.enabled=false"
    )
    record = manifest.stages["triples"]
    assert record.wall_clock_seconds is None
    assert record.peak_rss_bytes is None
    assert record.cumulative_peak_rss_bytes is None


def test_instrumented_runner_measures_and_persists(tmp_path: Path) -> None:
    config = load_config()
    spec = STAGE_REGISTRY["inference"]
    manifest = create_run_manifest(
        config,
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=[_stand_in()],
    )
    run_path = save_run_manifest(manifest, tmp_path / f"{manifest.run_id}.yaml")

    def fake_runner(
        current: RunManifest,
        cfg: Any,
        *,
        run_path: Path,
        force_rerun: bool = False,
    ) -> RunManifest:
        current = mark_stage_started(current, spec, cfg)
        return mark_stage_finished(current, spec, status=StageStatus.COMPLETED)

    costs: list[StageCost] = []
    runners = instrumented_runners(
        snapshot_meta=None,
        costs=costs,
        monitor_interval_seconds=0.01,
        base={"inference": fake_runner},
    )
    updated = runners["inference"](manifest, config, run_path=run_path)

    assert [c.stage for c in costs] == ["inference"]
    assert costs[0].wall_clock_seconds >= 0.0
    record = updated.stages["inference"]
    assert record.wall_clock_seconds is not None
    # Persisted, so the numbers survive a later crash.
    assert load_run_manifest(run_path).stages["inference"].wall_clock_seconds is not None


def test_instrumented_runners_cover_every_stage() -> None:
    costs: list[StageCost] = []
    runners = instrumented_runners(
        snapshot_meta=None, costs=costs, monitor_interval_seconds=0.5
    )
    assert set(runners) == set(STAGE_ORDER)


def test_instrumented_runner_binds_the_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "meta.yaml"
    snapshot.write_text("{}\n", encoding="utf-8")
    costs: list[StageCost] = []
    runners = instrumented_runners(
        snapshot_meta=snapshot, costs=costs, monitor_interval_seconds=0.5
    )
    # The data_acquisition entry is the wrapper; the binding lives in the closure,
    # so assert it is not the bare registry runner.
    from darkhunter_pop.pipeline import STAGE_RUNNERS

    assert runners["data_acquisition"] is not STAGE_RUNNERS["data_acquisition"]


# ---------------------------------------------------------------------------
# Stand-in declaration and snapshot discovery
# ---------------------------------------------------------------------------


def test_declare_stand_ins_is_non_empty_and_well_formed(tmp_path: Path) -> None:
    config = load_config()
    snapshot = tmp_path / "20260101T000000Z_deadbeef" / "meta.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}\n", encoding="utf-8")
    stand_ins = declare_stand_ins(config, snapshot_meta=snapshot)
    assert stand_ins
    names = [s.name for s in stand_ins]
    assert len(names) == len(set(names)), "stand-in names must be unique"
    for stand_in in stand_ins:
        assert stand_in.kind in STAND_IN_KINDS
        assert stand_in.description.strip()
        assert stand_in.replaces.strip()
    assert "offline_gaia_snapshot_replay" in names
    assert "analytic_companion_nature_evidence" in names
    assert "per_sample_selection_weights" in names
    assert "ci_scale_dynesty" in names


def test_declare_stand_ins_omits_replay_when_querying_live() -> None:
    config = load_config()
    names = [s.name for s in declare_stand_ins(config, snapshot_meta=None)]
    assert "offline_gaia_snapshot_replay" not in names
    assert names, "the other stand-ins still apply to a live-archive dry run"


def test_declare_stand_ins_reports_forward_model_samples_honestly() -> None:
    config = load_config()
    stand_ins = {
        s.name: s for s in declare_stand_ins(config, snapshot_meta=None)
    }
    entry = stand_ins["no_reproducing_literature_sample"]
    enabled = {
        s.name: s.mode.value
        for s in config.sample_selection.samples
        if s.enabled
    }
    assert entry.values["modes"] == enabled
    expected = sorted(k for k, v in enabled.items() if v == "forward_model")
    assert entry.values["forward_model_mode_samples"] == expected
    # The prose must not claim everything is in reproduction mode when it isn't.
    if expected:
        assert "forward_model" in entry.description
    else:
        assert "Every enabled sample is in reproduction mode." in entry.description


def test_dry_run_seeds_resolve_against_the_real_schema() -> None:
    config = load_config()
    seeds = dry_run_seeds(config)
    assert seeds
    assert all(isinstance(v, int) for v in seeds.values())
    assert seeds["inference.random_seed"] == config.inference.random_seed


def test_snapshot_discovery_ignores_derived_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "gaia_snapshots"
    for name in (
        "20260101T000000Z_aaaa",
        "20260202T000000Z_bbbb",
        "20260202T000000Z_bbbb+enrich",
        "20260202T000000Z_bbbb+enrich+mc10000",
        "nss_enrichment",
    ):
        (root / name).mkdir(parents=True)
        (root / name / "meta.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "darkhunter_pop.dry_run.gaia_snapshots_dir", lambda _cfg: root
    )
    found = latest_gaia_snapshot_meta(load_config())
    assert found is not None
    assert found.parent.name == "20260202T000000Z_bbbb"


def test_snapshot_discovery_returns_none_when_nothing_pristine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "gaia_snapshots"
    (root / "nss_enrichment").mkdir(parents=True)
    (root / "nss_enrichment" / "meta.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "darkhunter_pop.dry_run.gaia_snapshots_dir", lambda _cfg: root
    )
    assert latest_gaia_snapshot_meta(load_config()) is None


def test_build_dry_run_manifest_writes_the_label_before_any_stage(
    tmp_path: Path,
) -> None:
    config = load_config()
    manifest, path = build_dry_run_manifest(
        config, snapshot_meta=None, runs=tmp_path
    )
    assert path.is_file()
    assert manifest.dry_run is True
    assert manifest.stages == {}, "no stage may have run yet"
    assert "dry_run: true" in path.read_text(encoding="utf-8")
    assert manifest.host_profile == config.paths.host_profile


# ---------------------------------------------------------------------------
# Caption and report
# ---------------------------------------------------------------------------


def _dry_manifest(config: Any, stand_ins: list[SyntheticStandIn]) -> RunManifest:
    return create_run_manifest(
        config,
        dry_run=True,
        dry_run_label=DRY_RUN_LABEL,
        synthetic_stand_ins=stand_ins,
        random_seeds=dry_run_seeds(config),
    )


def test_caption_carries_the_banner_and_names_each_stand_in() -> None:
    config = load_config()
    stand_ins = declare_stand_ins(config, snapshot_meta=None)
    caption = caption_text(_dry_manifest(config, stand_ins))
    assert DRY_RUN_LABEL.upper() in caption
    for stand_in in stand_ins:
        assert stand_in.name in caption
        assert stand_in.replaces in caption


def test_report_header_carries_the_banner_before_any_number() -> None:
    config = load_config()
    stand_ins = declare_stand_ins(config, snapshot_meta=None)
    manifest = _dry_manifest(config, stand_ins)
    text = format_dry_run_report(
        manifest,
        plan_text="(plan)",
        inputs=None,
        figure_path=None,
        total_wall_clock_seconds=12.5,
    )
    head = text.splitlines()[:4]
    assert any(DRY_RUN_LABEL.upper() in line for line in head)
    for stand_in in stand_ins:
        assert stand_in.name in text
    # Every stage appears in the outcome table, even with no record.
    for name in STAGE_ORDER:
        assert name in text
    assert "not a science result" in text.lower()
    assert "(plan)" in text


def test_report_includes_cost_columns_and_the_measurement_caveat() -> None:
    config = load_config()
    manifest = _dry_manifest(config, [_stand_in()])
    spec = STAGE_REGISTRY["inference"]
    manifest = mark_stage_started(manifest, spec, config)
    manifest = mark_stage_finished(manifest, spec, status=StageStatus.COMPLETED)
    manifest = record_stage_resources(
        manifest,
        "inference",
        wall_clock_seconds=2.5,
        peak_rss_bytes=2 << 30,
        cumulative_peak_rss_bytes=3 << 30,
    )
    text = format_dry_run_report(
        manifest,
        plan_text="",
        inputs=None,
        figure_path=None,
        total_wall_clock_seconds=2.5,
    )
    assert "2.50s" in text
    assert "2.00 GiB" in text
    assert "3.00 GiB" in text
    assert "getrusage" in text


def test_dry_run_dirs_are_not_a_stage_directory() -> None:
    config = load_config()
    figures, reports = dry_run_dirs(config, "20260101-000000-abc1234")
    assert figures.parent.name == DRY_RUN_SUBDIR
    assert reports.parent.name == DRY_RUN_SUBDIR
    assert DRY_RUN_SUBDIR not in set(STAGE_ORDER)
    assert figures.name == config.diagnostics.figures_subdir
    assert reports.name == config.diagnostics.reports_subdir


# ---------------------------------------------------------------------------
# Gate predicates
# ---------------------------------------------------------------------------


def _result(manifest: RunManifest) -> DryRunResult:
    return DryRunResult(
        manifest=manifest,
        run_path=Path("r.yaml"),
        plan_text="",
        report_path=Path("r.txt"),
        report_text="",
        figure_path=None,
        caption_path=Path("c.txt"),
        costs=[],
        total_wall_clock_seconds=0.0,
    )


def test_unfinished_stages_flags_running_and_missing() -> None:
    config = load_config()
    manifest = _dry_manifest(config, [_stand_in()])
    manifest = mark_stage_started(manifest, STAGE_REGISTRY["diagnostics"], config)
    unfinished = _result(manifest).unfinished_stages()
    assert "diagnostics" in unfinished, "a stage left running must fail the gate"
    assert "data_acquisition" in unfinished, "a stage with no record at all fails"


def test_skips_without_reason_flags_a_bare_skip() -> None:
    config = load_config()
    manifest = _dry_manifest(config, [_stand_in()])
    spec = STAGE_REGISTRY["triples"]
    manifest = mark_stage_started(manifest, spec, config)
    manifest = mark_stage_finished(manifest, spec, status=StageStatus.SKIPPED)
    assert _result(manifest).skips_without_reason() == ["triples"]
    manifest = mark_stage_finished(
        manifest, spec, status=StageStatus.SKIPPED, reason="triples.enabled=false"
    )
    assert _result(manifest).skips_without_reason() == []


# ---------------------------------------------------------------------------
# Product figure primitive
# ---------------------------------------------------------------------------


def test_plot_dndm_by_class_rejects_a_mismatched_curve(tmp_path: Path) -> None:
    grid = np.geomspace(0.1, 20.0, 16)
    with pytest.raises(ValueError, match="does not match mass grid"):
        plot_dndm_by_class(
            grid,
            np.ones(8),
            {},
            tmp_path / "f.png",
            dpi=60,
            class_order=["BH"],
        )


def test_plot_dndm_by_class_returns_none_when_nothing_is_drawable(
    tmp_path: Path,
) -> None:
    grid = np.geomspace(0.1, 20.0, 16)
    out = plot_dndm_by_class(
        grid,
        np.zeros_like(grid),
        {"BH": np.zeros_like(grid)},
        tmp_path / "f.png",
        dpi=60,
        class_order=["BH"],
    )
    assert out is None
    assert not (tmp_path / "f.png").exists()


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_plot_dndm_by_class_writes_all_five_classes(tmp_path: Path) -> None:
    config = load_config()
    classes = list(config.population_model.population_classes)
    grid = np.geomspace(0.5, 20.0, 24)
    curves = {name: np.full_like(grid, float(i + 1)) for i, name in enumerate(classes)}
    out = plot_dndm_by_class(
        grid,
        sum(curves.values()),
        curves,
        tmp_path / "dndm.png",
        dpi=60,
        class_order=classes,
        title=DRY_RUN_LABEL,
        caption=DRY_RUN_LABEL + "\nnaming every stand-in here.",
        vlines={"M_Ch": 1.4, "M_TOV": 2.2},
        style=config.plotting,
    )
    assert out is not None and out.is_file()
    assert out.stat().st_size > 0
    assert set(classes) == {"BH", "NS", "WD", "other", "outlier"}


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_plot_dndm_by_class_appends_classes_missing_from_the_order(
    tmp_path: Path,
) -> None:
    grid = np.geomspace(0.5, 20.0, 12)
    curves = {"BH": np.ones_like(grid), "NS": np.full_like(grid, 2.0)}
    # A stale class_order must not silently drop a curve.
    out = plot_dndm_by_class(
        grid,
        np.full_like(grid, 3.0),
        curves,
        tmp_path / "dndm.png",
        dpi=60,
        class_order=["BH"],
    )
    assert out is not None and out.is_file()


def test_dndm_figure_inputs_is_a_plain_container() -> None:
    grid = np.geomspace(0.5, 5.0, 4)
    inputs = DnDmFigureInputs(
        mass_grid_msun=grid,
        total_dndm=np.ones_like(grid),
        class_dndm={"BH": np.ones_like(grid)},
        bin_edges_msun=np.asarray([0.5, 5.0]),
        heights=np.asarray([1.0]),
        heights_source="test",
        class_fractions={"BH": 1.0},
        m_ch_msun=1.4,
        m_tov_msun=2.2,
    )
    assert inputs.heights_source == "test"
    assert inputs.class_dndm["BH"].shape == grid.shape


def test_report_names_classes_with_and_without_a_drawable_curve() -> None:
    config = load_config()
    manifest = _dry_manifest(config, [_stand_in()])
    grid = np.geomspace(0.5, 5.0, 4)
    inputs = DnDmFigureInputs(
        mass_grid_msun=grid,
        total_dndm=np.ones_like(grid),
        class_dndm={"BH": np.ones_like(grid), "WD": np.zeros_like(grid)},
        bin_edges_msun=np.asarray([0.5, 5.0]),
        heights=np.asarray([1.0]),
        heights_source="inference posterior median bin heights (CI-scale)",
        class_fractions={"BH": 0.5, "WD": 0.5},
        m_ch_msun=1.4,
        m_tov_msun=2.2,
    )
    text = format_dry_run_report(
        manifest,
        plan_text="",
        inputs=inputs,
        figure_path=Path("dndm.png"),
        total_wall_clock_seconds=1.0,
    )
    assert "classes with a drawable curve: ['BH']" in text
    assert "no positive rate anywhere on the grid: ['WD']" in text
    assert "inference posterior median bin heights (CI-scale)" in text
    assert "NOT measured class fractions" in text


def test_report_and_caption_filenames_are_stable() -> None:
    assert DRY_RUN_REPORT_NAME.endswith(".txt")
    assert DNDM_CAPTION_NAME.endswith(".txt")


# ---------------------------------------------------------------------------
# Measured caption / title layout, and log-y clipping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_measured_text_width_grows_with_length_and_size() -> None:
    from darkhunter_pop.plotting import measure_text_width_inches

    short = measure_text_width_inches("abc", font_family="serif", fontsize=14.0)
    longer = measure_text_width_inches(
        "abcabcabc", font_family="serif", fontsize=14.0
    )
    bigger = measure_text_width_inches("abc", font_family="serif", fontsize=28.0)
    assert 0.0 < short < longer
    assert bigger > short
    assert measure_text_width_inches("", font_family="serif", fontsize=14.0) == 0.0


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_wrapped_caption_lines_all_fit_the_budget() -> None:
    from darkhunter_pop.plotting import (
        measure_text_width_inches,
        wrap_text_to_inches,
    )

    config = load_config()
    text = caption_text(
        _dry_manifest(config, declare_stand_ins(config, snapshot_meta=None))
    )
    budget = 6.5
    lines = wrap_text_to_inches(
        text,
        max_width_inches=budget,
        font_family=config.plotting.font_family,
        fontsize=config.plotting.tick_label_fontsize,
    )
    assert len(lines) > len(text.splitlines()), "a long caption must wrap"
    for line in lines:
        # Only a single unbreakable word may exceed the budget.
        if len(line.split()) > 1:
            assert (
                measure_text_width_inches(
                    line,
                    font_family=config.plotting.font_family,
                    fontsize=config.plotting.tick_label_fontsize,
                )
                <= budget
            ), line


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_wrap_preserves_blank_lines() -> None:
    from darkhunter_pop.plotting import wrap_text_to_inches

    lines = wrap_text_to_inches(
        "first\n\nsecond",
        max_width_inches=6.0,
        font_family="serif",
        fontsize=14.0,
    )
    assert lines == ["first", "", "second"]


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_captioned_figure_reserves_a_band_above_the_caption(tmp_path: Path) -> None:
    from PIL import Image

    config = load_config()
    grid = np.geomspace(0.5, 20.0, 12)
    curves = {"BH": np.ones_like(grid)}
    kwargs: dict[str, Any] = dict(dpi=60, class_order=["BH"], style=config.plotting)
    bare = plot_dndm_by_class(
        grid, np.ones_like(grid), curves, tmp_path / "bare.png", **kwargs
    )
    captioned = plot_dndm_by_class(
        grid,
        np.ones_like(grid),
        curves,
        tmp_path / "captioned.png",
        caption="line one\nline two\nline three\nline four",
        **kwargs,
    )
    assert bare is not None and captioned is not None
    # The reserved band makes the image strictly taller at the same width, which
    # is how caption text is kept off the x-axis label.
    with Image.open(bare) as b, Image.open(captioned) as c:
        assert c.height > b.height
        assert c.width == b.width


def test_dndm_log_ylim_keeps_the_configured_window() -> None:
    from darkhunter_pop.plotting import dndm_log_ylim

    floor, ceiling = dndm_log_ylim(100.0, 12.0)
    assert np.log10(ceiling) - np.log10(floor) == pytest.approx(12.5, abs=1e-9)
    assert ceiling > 100.0, "needs headroom so the top markers are not clipped"
    assert floor == pytest.approx(100.0 * 1e-12)


def test_dndm_log_ylim_rejects_nonpositive_inputs() -> None:
    from darkhunter_pop.plotting import dndm_log_ylim

    with pytest.raises(ValueError, match="largest must be positive"):
        dndm_log_ylim(0.0, 12.0)
    with pytest.raises(ValueError, match="decades must be positive"):
        dndm_log_ylim(1.0, 0.0)


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")
def test_log_y_clipping_drops_a_plunging_truncation_tail(tmp_path: Path) -> None:
    from darkhunter_pop.plotting import dndm_log_ylim

    config = load_config()
    decades = float(config.plotting.dndm_y_decades)
    grid = np.geomspace(0.5, 20.0, 16)
    # A curve decaying far below the configured window, as a soft M_TOV
    # truncation does: unclipped, the axis would span ~40 decades.
    plunging = 10.0 ** np.linspace(0.0, -40.0, grid.size)
    out = plot_dndm_by_class(
        grid,
        np.ones_like(grid),
        {"NS": plunging},
        tmp_path / "clip.png",
        dpi=60,
        class_order=["NS"],
        style=config.plotting,
    )
    assert out is not None and out.is_file()
    floor, _ceiling = dndm_log_ylim(1.0, decades)
    assert floor > float(np.min(plunging)), "the plunging tail must be clipped off"


def test_plotting_style_exposes_the_new_layout_knobs() -> None:
    from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS

    config = load_config()
    assert config.plotting.dndm_y_decades > 0
    assert config.plotting.caption_line_spacing > 0
    # Display-only: `plotting` must stay out of the resume checksum, so a style
    # edit never invalidates a resume.
    assert "plotting" not in SHARED_CHECKSUM_SECTIONS
