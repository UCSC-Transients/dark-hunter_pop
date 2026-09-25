"""Tests for the data_acquisition stage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.table import Table

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import QualityCutBin
from darkhunter_pop.data_acquisition import (
    SnapshotMeta,
    apply_quality_cuts,
    build_nss_adql,
    build_nss_type_smoke_adql,
    format_funnel_table,
    gaia_snapshots_dir,
    load_gaia_snapshot,
    passes_quality_cut,
    quality_bin_for_star,
    read_stage_hdf5,
    run_data_acquisition,
    save_gaia_snapshot,
    table_row_to_candidate,
    table_to_candidates,
    write_stage_hdf5,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import CandidateRecord, StageStatus

pytestmark = pytest.mark.unit

FIXTURE_TABLE = Path(__file__).parent / "fixtures" / "nss_sample.ecsv"


def _sample_table() -> Table:
    return Table(
        {
            "source_id": [1001, 1002, 1003],
            "nss_solution_type": ["Orbital", "Orbital", "Orbital"],
            "ra": [10.0, 15.0, 20.0],
            "dec": [-20.0, 5.0, 10.0],
            "parallax": [5.0, 3.0, 2.0],
            "parallax_error": [0.1, 0.1, 0.1],
            "pmra": [1.0, 0.5, -0.5],
            "pmdec": [-2.0, 0.5, 1.0],
            "period": [100.0, 200.0, 50.0],
            "period_error": [1.0, 2.0, 0.5],
            "eccentricity": [0.1, 0.5, 0.05],
            "eccentricity_error": [0.01, 0.02, 0.01],
            "goodness_of_fit": [4.0, 8.0, 11.0],
            "ruwe": [1.2, 1.5, 1.8],
            "A": [0.1, 0.2, 0.3],
            "A_error": [0.01, 0.01, 0.01],
            "B": [0.2, 0.3, 0.4],
            "B_error": [0.02, 0.02, 0.02],
            "F": [0.3, 0.4, 0.5],
            "F_error": [0.03, 0.03, 0.03],
            "G": [0.4, 0.5, 0.6],
            "G_error": [0.04, 0.04, 0.04],
            "phot_g_mean_mag": [12.5, 14.0, 12.0],
            "g_mag": [12.5, 14.0, 12.0],
            "g_mag_err": [0.01, 0.01, 0.01],
            "bp_mag": [13.0, 14.5, 12.5],
            "rp_mag": [11.8, 13.5, 11.5],
            "J_mag": [11.0, 12.0, 10.5],
            "J_mag_err": [0.05, 0.05, 0.05],
        }
    )


def _dr_config():
    return load_config().dr3


@pytest.mark.parametrize(
    ("g_mag", "gof", "expected"),
    [
        (12.5, 4.0, True),
        (12.5, 6.0, True),
        (14.0, 4.0, True),
        (14.0, 8.0, False),
        (None, 4.0, False),
    ],
)
def test_passes_quality_cut_default_bins(
    g_mag: float | None,
    gof: float | None,
    expected: bool,
) -> None:
    bins = load_config().dr3.quality_cut_bins
    assert passes_quality_cut(g_mag, gof, bins) is expected


def test_quality_cut_supports_arbitrary_bin_count() -> None:
    bins = [
        QualityCutBin(g_max=12.0, gof_max=8.0),
        QualityCutBin(g_min=12.0, g_max=15.0, gof_max=6.0),
        QualityCutBin(g_min=15.0, gof_max=4.0),
    ]
    assert quality_bin_for_star(11.0, 7.0, bins) == 0
    assert quality_bin_for_star(13.0, 5.0, bins) == 1
    assert quality_bin_for_star(16.0, 3.0, bins) == 2
    assert passes_quality_cut(13.0, 7.0, bins) is False
    assert passes_quality_cut(13.0, 5.0, bins) is True


def test_apply_quality_cuts_on_fixture_table() -> None:
    table = _sample_table()
    bins = load_config().dr3.quality_cut_bins
    filtered, counts = apply_quality_cuts(table, bins)
    assert len(filtered) == 1
    assert counts["unclassified"] == 0
    assert filtered["source_id"][0] == 1001


def test_apply_quality_cuts_accepts_g_mag_alias() -> None:
    """Production ADQL aliases Gaia G as g_mag, not phot_g_mean_mag."""
    table = Table(
        {
            "source_id": [1, 2],
            "g_mag": [12.0, 14.0],
            "goodness_of_fit": [4.0, 8.0],
        }
    )
    bins = load_config().dr3.quality_cut_bins
    filtered, counts = apply_quality_cuts(table, bins)
    assert len(filtered) == 1
    assert filtered["source_id"][0] == 1
    assert counts["unclassified"] == 0


def test_build_nss_adql_contains_configured_tables() -> None:
    dr = _dr_config()
    adql = build_nss_adql(dr)
    assert dr.nss_table in adql
    assert dr.gaia_source_table in adql
    assert "tmass_psc_xsc_best_neighbour" in adql
    assert "tmass_psc_xsc_join" in adql
    assert "gaiadr1.tmass_original_valid" in adql
    assert "panstarrs1_best_neighbour" in adql
    assert "g_mean_psf_mag AS g_ps1_mag" in adql
    assert "y_mean_psf_mag AS y_ps1_mag" in adql
    assert "g AS g_sdss_mag" in adql
    assert "z AS z_sdss_mag" in adql
    assert "a_thiele_innes AS A" in adql
    assert "COALESCE(nss.ra, gs.ra) AS ra" in adql
    assert "COALESCE(nss.period_error, nss.input_period_error) AS period_error" in adql
    assert "nss.t_periastron" in adql
    assert "nss.corr_vec" in adql
    assert "nss.bit_index" in adql
    assert "nss.semi_amplitude_primary" in adql
    assert "nss.semi_amplitude_primary_error" in adql
    assert "nss.ra_error" in adql
    assert "nss.A," not in adql.replace("\n", "")
    assert "phot_g_mean_mag_error" not in adql
    assert "LEFT JOIN" in adql
    assert "galex_ais_best_neighbour" not in adql  # disabled


def test_build_nss_type_smoke_adql_one_solution_type() -> None:
    dr = _dr_config()
    adql = build_nss_type_smoke_adql(dr, nss_solution_type="EclipsingBinary", top_n=20)
    assert "TOP 20 source_id, nss_solution_type" in adql
    assert "WHERE nss_solution_type = 'EclipsingBinary'" in adql
    assert "pick.nss_solution_type = nss.nss_solution_type" in adql
    assert "COALESCE(nss.period_error, nss.input_period_error) AS period_error" in adql
    assert "UNION ALL" not in adql


def test_gaia_archive_async_defaults_true() -> None:
    cfg = load_config()
    assert cfg.dr3.gaia_archive_async is True
    assert cfg.dr3.gaia_archive_row_limit == -1


def test_snapshot_round_trip(tmp_path: Path) -> None:
    table = _sample_table()
    adql = "SELECT * FROM test"
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    meta = save_gaia_snapshot(
        table,
        adql,
        snapshots_dir=tmp_path,
        query_date=when,
    )
    loaded_meta, loaded_table = load_gaia_snapshot(meta.meta_path)
    assert loaded_meta.checksum == meta.checksum
    assert loaded_meta.adql == adql
    assert len(loaded_table) == len(table)
    assert meta.meta_path.is_file()
    assert meta.result_path.is_file()


def test_table_row_to_candidate_maps_owned_fields() -> None:
    row = _sample_table()[0]
    candidate = table_row_to_candidate(row, _dr_config())
    assert candidate.source_id == 1001
    assert candidate.nss_solution_type == "Orbital"
    assert candidate.ra_deg == pytest.approx(10.0)
    assert candidate.thiele_innes is not None
    assert candidate.thiele_innes.A == pytest.approx(0.1)
    assert candidate.rv_summary == {}
    assert candidate.m1 is None
    assert any(point.band == "G" for point in candidate.photometry)
    assert "period" in candidate.nss_orbital


def test_optional_float_treats_masked_as_missing() -> None:
    from darkhunter_pop.data_acquisition import _optional_float

    row = {"eccentricity_error": np.ma.masked}
    assert _optional_float(row, "eccentricity_error") is None
    row = {"eccentricity_error": np.ma.array([0.01], mask=[False])[0]}
    assert _optional_float(row, "eccentricity_error") == pytest.approx(0.01)


def test_eclipsing_binary_period_error_from_input_period_error() -> None:
    dr = _dr_config()
    row = {
        "source_id": 2001,
        "nss_solution_type": "EclipsingBinary",
        "ra": 10.0,
        "dec": 5.0,
        "period": 1.5,
        "period_error": None,
        "eccentricity": 0.0,
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    candidate = table_row_to_candidate(row, dr)
    assert "period_error" not in candidate.nss_orbital

    row["period_error"] = 0.001
    candidate = table_row_to_candidate(row, dr)
    assert candidate.nss_orbital["period_error"] == pytest.approx(0.001)


def test_pseudo_circular_orbital_flagged_in_extras() -> None:
    dr = _dr_config()
    row = {
        "source_id": 3001,
        "nss_solution_type": "Orbital",
        "ra": 10.0,
        "dec": 5.0,
        "period": 100.0,
        "eccentricity": 1.5e-7,
        "eccentricity_error": None,
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    candidate = table_row_to_candidate(row, dr)
    assert candidate.extras.get("gaia_pseudo_circular") is True

    row["eccentricity_error"] = 0.01
    candidate = table_row_to_candidate(row, dr)
    assert "gaia_pseudo_circular" not in candidate.extras


def test_external_mag_err_zero_imputed_with_floor() -> None:
    dr = _dr_config()
    row = {
        "source_id": 4001,
        "nss_solution_type": "SB1",
        "ra": 10.0,
        "dec": 5.0,
        "period": 10.0,
        "eccentricity": 0.1,
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
        "g_ps1_mag": 12.1,
        "g_ps1_mag_err": 0.0,
    }
    candidate = table_row_to_candidate(row, dr)
    ps1 = next(p for p in candidate.photometry if p.band == "g_ps1")
    assert ps1.mag_err == pytest.approx(dr.external_mag_err_floor)
    assert candidate.extras.get("mag_err_imputed_bands") == ["g_ps1"]
    spec = candidate.extras.get("spectroscopic_mass_function")
    assert spec is not None
    assert spec["inference_eligible"] is False
    assert spec["feeds_population_likelihood"] is False
    assert spec["sin3i_marginalization"] is False
    assert spec["f_m_msun"] is None


def test_sb1_ingests_k1_and_computes_spectroscopic_f_m() -> None:
    from darkhunter_pop.physics_utils import spectroscopic_mass_function

    dr = _dr_config()
    row = {
        "source_id": 5001,
        "nss_solution_type": "SB1",
        "ra": 10.0,
        "dec": 5.0,
        "period": 100.0,
        "eccentricity": 0.2,
        "semi_amplitude_primary": 50.0,
        "semi_amplitude_primary_error": 2.0,
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    candidate = table_row_to_candidate(row, dr)
    assert candidate.nss_orbital["k1_kms"] == pytest.approx(50.0)
    assert candidate.nss_orbital["k1_error_kms"] == pytest.approx(2.0)
    spec = candidate.extras["spectroscopic_mass_function"]
    expected = float(spectroscopic_mass_function(100.0, 50.0, 0.2))
    assert spec["f_m_msun"] == pytest.approx(expected)
    assert spec["k1_significance"] == pytest.approx(25.0)
    assert spec["inference_eligible"] is False
    assert candidate.m2 is None


def test_sb1c_missing_eccentricity_is_circular() -> None:
    dr = _dr_config()
    row = {
        "source_id": 5002,
        "nss_solution_type": "SB1C",
        "ra": 10.0,
        "dec": 5.0,
        "period": 20.0,
        "semi_amplitude_primary": 30.0,
        "semi_amplitude_primary_error": 1.0,
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    candidate = table_row_to_candidate(row, dr)
    assert candidate.nss_orbital["eccentricity"] == pytest.approx(0.0)
    assert candidate.extras.get("sb1c_circular_eccentricity") is True
    spec = candidate.extras["spectroscopic_mass_function"]
    assert spec["f_m_msun"] is not None
    assert spec["feeds_population_likelihood"] is False


def test_write_and_read_stage_hdf5(tmp_path: Path) -> None:
    table = _sample_table()
    dr = _dr_config()
    candidates = table_to_candidates(table[:2], dr)
    snapshot = SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=len(table),
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )
    from darkhunter_pop.data_acquisition import FunnelCounts, compute_stage_diagnostics

    funnel = FunnelCounts(queried=3, after_quality_cut=2, candidates_written=2)
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=funnel,
        quality_cut_bin_counts={"bin0": 2},
    )
    artifact = tmp_path / "stage.h5"
    write_stage_hdf5(artifact, candidates, snapshot=snapshot, diagnostics=diagnostics)
    loaded, meta = read_stage_hdf5(artifact)
    assert len(loaded) == 2
    assert loaded[0].source_id == candidates[0].source_id
    assert meta["stage"] == "data_acquisition"
    assert CandidateRecord.model_validate(loaded[0].model_dump(mode="json"))
    import h5py

    with h5py.File(artifact, "r") as handle:
        spec = handle["data_acquisition"]["spectroscopic_mass_function"]
        assert spec.attrs["inference_eligible"] in (False, 0)
        assert spec.attrs["feeds_population_likelihood"] in (False, 0)
        assert spec.attrs["sin3i_marginalization"] in (False, 0)
        assert spec.attrs["n_sb1"] == 0
        assert handle["meta"].attrs["spectroscopic_mass_function_inference_eligible"] in (
            False,
            0,
        )


def test_nss_panels_round_trip(tmp_path: Path) -> None:
    from darkhunter_pop.data_acquisition import FunnelCounts, compute_stage_diagnostics
    from darkhunter_pop.forward_model import SIX_PANEL_NAMES, load_real_panels_from_data_acquisition

    table = _sample_table()
    dr = _dr_config()
    candidates = table_to_candidates(table[:2], dr)
    snapshot = SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=len(table),
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )
    funnel = FunnelCounts(queried=3, after_quality_cut=2, candidates_written=2)
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=funnel,
        quality_cut_bin_counts={"bin0": 2},
    )
    assert diagnostics.nss_panels
    assert "f_m_msun" in diagnostics.nss_panels
    assert "cos_inclination" in diagnostics.nss_panels
    assert len(diagnostics.nss_panels["f_m_msun"]) == 2
    assert diagnostics.solution_type_fractions["twelve_parameter_orbital"] == pytest.approx(1.0)

    artifact = tmp_path / "stage.h5"
    write_stage_hdf5(artifact, candidates, snapshot=snapshot, diagnostics=diagnostics)
    panels, st_frac = load_real_panels_from_data_acquisition(artifact)
    for name in SIX_PANEL_NAMES:
        if name in diagnostics.nss_panels:
            assert name in panels
            assert len(panels[name]) == len(diagnostics.nss_panels[name])
    assert st_frac["twelve_parameter_orbital"] == pytest.approx(1.0)


def test_nss_solution_type_mapping() -> None:
    from darkhunter_pop.data_acquisition import nss_solution_type_to_cascade_label

    assert nss_solution_type_to_cascade_label("Orbital") == "twelve_parameter_orbital"
    assert nss_solution_type_to_cascade_label("Orbital9") == "nine_parameter"
    assert nss_solution_type_to_cascade_label("EclipsingBinary") == "insufficient_visibility"


def test_run_data_acquisition_writes_manifest_and_artifact(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    def fake_query(adql: str, dr) -> Table:
        assert "nss_two_body_orbit" in adql
        return _sample_table()

    data_root = tmp_path / "data"
    artifact_root = tmp_path / "output"
    tweaked = cfg.model_copy(deep=True)
    tweaked.paths = cfg.paths.model_copy(
        update={"data_root": str(data_root), "artifact_root": str(artifact_root)}
    )

    finished = run_data_acquisition(
        manifest,
        tweaked,
        run_path=run_path,
        query_fn=fake_query,
    )
    spec = STAGE_REGISTRY["data_acquisition"]
    record = finished.stages["data_acquisition"]
    assert record.status is StageStatus.COMPLETED
    artifact = stage_artifact_path(tweaked, spec, run_id=finished.run_id)
    assert artifact.is_file()
    candidates, _ = read_stage_hdf5(artifact)
    assert len(candidates) == 1
    snap_dir = gaia_snapshots_dir(tweaked)
    assert snap_dir.is_dir()
    meta_files = list(snap_dir.glob("*/meta.yaml"))
    assert len(meta_files) == 1
    meta = yaml.safe_load(meta_files[0].read_text(encoding="utf-8"))
    assert "adql" in meta
    assert "checksum" in meta


def test_run_data_acquisition_from_snapshot_skips_query(tmp_path: Path) -> None:
    cfg = load_config()
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    table = _sample_table()
    snap_dir = tmp_path / "snaps"
    meta = save_gaia_snapshot(table, "SELECT 1", snapshots_dir=snap_dir)

    data_root = tmp_path / "data"
    artifact_root = tmp_path / "output"
    tweaked = cfg.model_copy(deep=True)
    tweaked.paths = cfg.paths.model_copy(
        update={"data_root": str(data_root), "artifact_root": str(artifact_root)}
    )

    def boom(adql: str, dr) -> Table:  # noqa: ARG001
        raise AssertionError("archive query must not run when snapshot is provided")

    finished = run_data_acquisition(
        manifest,
        tweaked,
        run_path=run_path,
        query_fn=boom,
        snapshot_meta_path=meta.meta_path,
    )
    assert finished.stages["data_acquisition"].status is StageStatus.COMPLETED
    artifact = stage_artifact_path(
        tweaked, STAGE_REGISTRY["data_acquisition"], run_id=finished.run_id
    )
    candidates, _ = read_stage_hdf5(artifact)
    assert len(candidates) == 1


def test_format_funnel_table_is_legible() -> None:
    from darkhunter_pop.data_acquisition import FunnelCounts
    from darkhunter_pop.nss_covariance import CovarianceHealth

    health = CovarianceHealth()
    health.ok = 2
    health.missing_corr = 1
    health.by_solution_type = {"Orbital": {"ok": 2}, "SB1": {"missing_corr": 1}}
    text = format_funnel_table(
        FunnelCounts(
            queried=10,
            after_quality_cut=4,
            candidates_written=4,
            covariance_ok=2,
            covariance_failed=1,
        ),
        {"bin0": 4},
        covariance_health=health,
    )
    assert "data_acquisition funnel" in text
    assert "queried" in text
    assert "covariance_ok" in text
    assert "covariance_health (by solution type)" in text
    assert "Orbital" in text


# --- duplicate source_id: tag-and-keep genuine multi-solution, refuse the rest ---
# (issues #221, #237, #241, #242) -------------------------------------------------
#
# Domain decision (CONTINUATION_PLAN.md §15 Q17, resolved 2026-09-24): genuine
# multi-solution source_id duplicates are real and must be kept, each row tagged by
# its own nss_solution_type, never merged. PR #238 instead collapsed/merged rows
# cell-by-cell on a cross-match-fan-out premise; on the real snapshot 5,926 of the
# 5,932 duplicated source_ids are instead *distinct NSS orbital solutions for one
# source*, so that merge fabricated orbits and was reverted (PR #239). This file's
# tests below cover: cross-type multi-solution (kept), same-type/period-aliased
# multi-solution (kept), and genuine cross-match fan-out (still refused, since no
# fan-out resolution logic exists — #242 does not design one here).


def _cross_type_multi_solution_table() -> Table:
    """source_id 1002 carries two distinct NSS orbital solutions (cross-type, #241).

    Mirrors the dominant real-snapshot combination (Orbital + SB1, 5,290 of 5,926
    measured genuine multi-solution source_ids): identical photometry/astrometry,
    different ``nss_solution_type`` and ``period``. Both rows are genuine and must be
    kept and tagged, never merged.
    """
    table = Table(_sample_table()[[0, 1, 1, 2]])
    table["nss_solution_type"] = ["Orbital", "Orbital", "SB1", "Orbital"]
    table["period"] = [100.0, 200.0, 0.62, 50.0]
    return table


def _same_type_period_aliased_table() -> Table:
    """source_id 1002 carries two period-aliased ``Orbital`` solutions (#241).

    Empirically zero on the real uncut snapshot
    (``docs/multi_solution_characterization/REPORT.md``), but the classifier must not
    assume the phenomenon is structurally impossible — this is the synthetic
    regression case for that sub-path.
    """
    table = Table(_sample_table()[[0, 1, 1, 2]])
    table["nss_solution_type"] = ["Orbital", "Orbital", "Orbital", "Orbital"]
    table["period"] = [100.0, 200.0, 400.0, 50.0]
    return table


def _cross_match_fanout_table() -> Table:
    """source_id 1002 carries the genuine cross-match fan-out signature (#221/PR #240).

    Identical ``nss_solution_type`` AND identical ``period`` for both rows — the
    ~0.1% real-snapshot case that differs only in cross-matched external photometry
    (2MASS ``J_mag`` here). This shape has no resolution logic yet and must keep
    raising ``DuplicateSourceIdError``.
    """
    table = Table(_sample_table()[[0, 1, 1, 2]])
    table["nss_solution_type"] = ["Orbital", "Orbital", "Orbital", "Orbital"]
    table["period"] = [100.0, 200.0, 200.0, 50.0]
    table["J_mag"] = [11.0, 12.0, 12.5, 10.5]
    return table


def _snapshot_meta(tmp_path: Path, *, row_count: int) -> SnapshotMeta:
    return SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=row_count,
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )


def test_classify_duplicate_source_id_group_distinguishes_the_three_shapes() -> None:
    from darkhunter_pop.data_acquisition import classify_duplicate_source_id_group

    dr = _dr_config()

    cross_type = [
        c
        for c in table_to_candidates(_cross_type_multi_solution_table(), dr)
        if c.source_id == 1002
    ]
    assert classify_duplicate_source_id_group(cross_type) == "cross_type"

    same_type = [
        c
        for c in table_to_candidates(_same_type_period_aliased_table(), dr)
        if c.source_id == 1002
    ]
    assert classify_duplicate_source_id_group(same_type) == "same_type_period_aliased"

    fanout = [
        c
        for c in table_to_candidates(_cross_match_fanout_table(), dr)
        if c.source_id == 1002
    ]
    assert classify_duplicate_source_id_group(fanout) == "fanout"


def test_assert_unique_source_ids_keeps_and_tags_cross_type_multi_solution() -> None:
    """Cross-type multi-solution (#241): kept, tagged, not raised, not merged."""
    from darkhunter_pop.data_acquisition import assert_unique_source_ids

    dr = _dr_config()
    candidates = table_to_candidates(_cross_type_multi_solution_table(), dr)
    counts = assert_unique_source_ids(candidates)
    assert counts.cross_type == 1
    assert counts.same_type_period_aliased == 0
    assert counts.both == 0
    assert counts.total_kept == 1

    # No row was merged, dropped, or reordered: source_id 1002 still has exactly two
    # records, one per Gaia row, each carrying only that row's own fields.
    dup = [c for c in candidates if c.source_id == 1002]
    assert len(dup) == 2
    types = {c.nss_solution_type for c in dup}
    assert types == {"Orbital", "SB1"}
    orbital_row = next(c for c in dup if c.nss_solution_type == "Orbital")
    sb1_row = next(c for c in dup if c.nss_solution_type == "SB1")
    assert orbital_row.nss_orbital["period"] == pytest.approx(200.0)
    assert sb1_row.nss_orbital["period"] == pytest.approx(0.62)
    # Every other candidate is untouched.
    assert {c.source_id for c in candidates} == {1001, 1002, 1003}


def test_assert_unique_source_ids_keeps_and_tags_same_type_period_aliasing() -> None:
    """Same-type/period-aliased multi-solution (#241): kept, tagged, not raised."""
    from darkhunter_pop.data_acquisition import assert_unique_source_ids

    dr = _dr_config()
    candidates = table_to_candidates(_same_type_period_aliased_table(), dr)
    counts = assert_unique_source_ids(candidates)
    assert counts.cross_type == 0
    assert counts.same_type_period_aliased == 1
    assert counts.both == 0
    assert counts.total_kept == 1

    dup = [c for c in candidates if c.source_id == 1002]
    assert len(dup) == 2
    assert all(c.nss_solution_type == "Orbital" for c in dup)
    periods = {c.nss_orbital["period"] for c in dup}
    assert periods == {200.0, 400.0}


def test_assert_unique_source_ids_still_refuses_genuine_fanout() -> None:
    """Cross-match fan-out (#221/PR #240) has no resolution logic yet — still a hard stop."""
    from darkhunter_pop.data_acquisition import (
        DuplicateSourceIdError,
        assert_unique_source_ids,
    )

    dr = _dr_config()
    candidates = table_to_candidates(_cross_match_fanout_table(), dr)
    with pytest.raises(DuplicateSourceIdError) as excinfo:
        assert_unique_source_ids(candidates)
    error = excinfo.value
    assert error.stage == "data_acquisition"
    assert error.duplicate_source_ids == 1
    assert error.duplicate_records == 1
    assert error.example_source_id == 1002
    assert error.example_multiplicity == 2
    message = str(error)
    assert "data_acquisition" in message
    assert "1002" in message

    # A unique table passes silently, and so does an empty one.
    empty_counts = assert_unique_source_ids(table_to_candidates(_sample_table(), dr))
    assert empty_counts.total_kept == 0
    assert assert_unique_source_ids([]).total_kept == 0


def test_write_stage_hdf5_refuses_genuine_fanout_before_writing_bytes(
    tmp_path: Path,
) -> None:
    """#221: the refusal must land before a single HDF5 byte reaches disk."""
    from darkhunter_pop.data_acquisition import (
        DuplicateSourceIdError,
        FunnelCounts,
        compute_stage_diagnostics,
    )

    dr = _dr_config()
    candidates = table_to_candidates(_cross_match_fanout_table(), dr)
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=4, after_quality_cut=4, candidates_written=4),
        quality_cut_bin_counts={"bin0": 4},
    )
    artifact = tmp_path / "stage.h5"
    with pytest.raises(DuplicateSourceIdError):
        write_stage_hdf5(
            artifact,
            candidates,
            snapshot=_snapshot_meta(tmp_path, row_count=4),
            diagnostics=diagnostics,
        )
    # Zero bytes on disk: neither the artifact nor a .partial sibling.
    assert not artifact.exists()
    assert list(tmp_path.glob("stage.h5*")) == []


def test_write_stage_hdf5_keeps_genuine_multi_solution_rows(tmp_path: Path) -> None:
    """A genuine (cross-type) multi-solution duplicate must write successfully, with
    both rows surviving the round-trip untouched — the positive counterpart to the
    fan-out refusal test above."""
    from darkhunter_pop.data_acquisition import (
        FunnelCounts,
        assert_unique_source_ids,
        compute_stage_diagnostics,
    )

    dr = _dr_config()
    candidates = table_to_candidates(_cross_type_multi_solution_table(), dr)
    multi_solution = assert_unique_source_ids(candidates)
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(
            queried=4,
            after_quality_cut=4,
            candidates_written=4,
            multi_solution=multi_solution,
        ),
        quality_cut_bin_counts={"bin0": 4},
    )
    artifact = tmp_path / "stage.h5"
    write_stage_hdf5(
        artifact,
        candidates,
        snapshot=_snapshot_meta(tmp_path, row_count=4),
        diagnostics=diagnostics,
    )
    assert artifact.is_file()
    loaded, meta_attrs = read_stage_hdf5(artifact)
    assert sorted(c.source_id for c in loaded) == [1001, 1002, 1002, 1003]
    dup = [c for c in loaded if c.source_id == 1002]
    assert {c.nss_solution_type for c in dup} == {"Orbital", "SB1"}

    import h5py

    with h5py.File(artifact, "r") as handle:
        funnel_attrs = handle["diagnostics"].attrs
        assert funnel_attrs["multi_solution_cross_type"] == 1
        assert funnel_attrs["multi_solution_same_type_period_aliased"] == 0
        assert funnel_attrs["multi_solution_kept_total"] == 1


def test_write_stage_hdf5_leaves_no_partial_on_failure(tmp_path: Path) -> None:
    """A mid-write crash must not leave a file where plan_stage could cache-hit it."""
    import darkhunter_pop.data_acquisition as da

    from darkhunter_pop.data_acquisition import FunnelCounts, compute_stage_diagnostics

    dr = _dr_config()
    candidates = table_to_candidates(_sample_table()[:2], dr)
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=2, after_quality_cut=2, candidates_written=2),
        quality_cut_bin_counts={"bin0": 2},
    )
    artifact = tmp_path / "stage.h5"
    original = da._write_stage_hdf5_body

    def _crash_midway(path: Path, *args: object, **kwargs: object) -> None:
        # Write real bytes first, so the test exercises cleanup of an actual partial
        # file rather than a failure that never touched the filesystem.
        Path(path).write_bytes(b"\x89HDF\r\n\x1a\n truncated")
        raise RuntimeError("simulated mid-write failure")

    da._write_stage_hdf5_body = _crash_midway  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="simulated mid-write failure"):
            write_stage_hdf5(
                artifact,
                candidates,
                snapshot=_snapshot_meta(tmp_path, row_count=2),
                diagnostics=diagnostics,
            )
    finally:
        da._write_stage_hdf5_body = original  # type: ignore[assignment]
    assert list(tmp_path.glob("stage.h5*")) == []

    # And the happy path still produces a readable artifact at the final name.
    write_stage_hdf5(
        artifact,
        candidates,
        snapshot=_snapshot_meta(tmp_path, row_count=2),
        diagnostics=diagnostics,
    )
    assert artifact.is_file()
    assert not artifact.with_name(artifact.name + ".partial").exists()
    loaded, _meta = read_stage_hdf5(artifact)
    assert [c.source_id for c in loaded] == [1001, 1002]
