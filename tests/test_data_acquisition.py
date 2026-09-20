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


# --- cross-match fan-out: duplicate source_id collapse (issues #221, #231) -------


def _fanned_out_table() -> Table:
    """Sample table where source_id 1002 fans out over two cross-match rows.

    Row 1 carries PanSTARRS1-style photometry and no 2MASS; row 2 is the second
    accepted neighbour, carrying 2MASS and a *different* PanSTARRS1 magnitude.
    """
    base = _sample_table()
    table = Table(base[[0, 1, 1, 2]])
    table["g_ps1_mag"] = np.ma.masked_array(
        [np.nan, 13.9, 14.4, np.nan], mask=[True, False, False, True]
    )
    table["J_mag"] = np.ma.masked_array(
        [11.0, np.nan, 12.25, 10.5], mask=[False, True, False, False]
    )
    return table


def test_collapse_duplicate_source_ids_merges_photometry() -> None:
    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    collapsed, report = collapse_duplicate_source_ids(_fanned_out_table())

    assert report.rows_in == 4
    assert report.rows_out == 3
    assert report.duplicate_source_ids == 1
    assert report.rows_removed == 1
    assert report.max_multiplicity == 2
    assert report.example_source_id == 1002
    assert list(collapsed["source_id"]) == [1001, 1002, 1003]

    row = collapsed[list(collapsed["source_id"]).index(1002)]
    # The row with more non-null cells (the second, which has both PS1 and 2MASS)
    # is the base row, so its PS1 magnitude survives the conflict.
    assert float(row["g_ps1_mag"]) == pytest.approx(14.4)
    assert float(row["J_mag"]) == pytest.approx(12.25)


def test_collapse_fills_nulls_from_secondary_row() -> None:
    """The base row's missing cells are filled from the other rows of the group."""
    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    table = _fanned_out_table()
    # Make row 1 (index 1) the richer base by giving it J_mag too, and blank the
    # secondary row's PS1 magnitude so the fill direction is unambiguous.
    table["J_mag"] = np.ma.masked_array(
        [11.0, 12.0, np.nan, 10.5], mask=[False, False, True, False]
    )
    table["g_ps1_mag"] = np.ma.masked_array(
        [np.nan, np.nan, 14.4, np.nan], mask=[True, True, False, True]
    )
    collapsed, report = collapse_duplicate_source_ids(table)

    row = collapsed[list(collapsed["source_id"]).index(1002)]
    assert float(row["J_mag"]) == pytest.approx(12.0)
    assert float(row["g_ps1_mag"]) == pytest.approx(14.4)
    assert report.cells_filled >= 1


def test_collapse_is_a_no_op_on_a_unique_table() -> None:
    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    table = _sample_table()
    collapsed, report = collapse_duplicate_source_ids(table)
    assert collapsed is table
    assert report.rows_in == report.rows_out == 3
    assert report.duplicate_source_ids == 0
    assert report.rows_removed == 0
    assert report.max_multiplicity == 1
    assert report.example_source_id is None


def test_collapse_counts_conflicting_cells() -> None:
    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    _collapsed, report = collapse_duplicate_source_ids(_fanned_out_table())
    # g_ps1_mag differs between the two rows of source 1002; the base row wins and
    # the discarded value is counted rather than silently dropped.
    assert report.conflicting_cells >= 1


def test_collapsed_table_yields_one_candidate_per_source_id() -> None:
    """The whole point: collapse, not crash and not silent passthrough."""
    from darkhunter_pop.data_acquisition import collapse_duplicate_source_ids

    dr = _dr_config()
    duplicated = _fanned_out_table()
    assert len(table_to_candidates(duplicated, dr)) == 4  # pre-fix behaviour

    collapsed, _report = collapse_duplicate_source_ids(duplicated)
    candidates = table_to_candidates(collapsed, dr)
    source_ids = [c.source_id for c in candidates]
    assert source_ids == [1001, 1002, 1003]
    assert len(set(source_ids)) == len(source_ids)


def test_assert_unique_source_ids_names_stage_count_and_example() -> None:
    from darkhunter_pop.data_acquisition import (
        DuplicateSourceIdError,
        assert_unique_source_ids,
        collapse_duplicate_source_ids,
    )

    dr = _dr_config()
    candidates = table_to_candidates(_fanned_out_table(), dr)
    with pytest.raises(DuplicateSourceIdError) as excinfo:
        assert_unique_source_ids(candidates)
    error = excinfo.value
    assert error.stage == "data_acquisition"
    assert error.duplicate_source_ids == 1
    assert error.example_source_id == 1002
    message = str(error)
    assert "data_acquisition" in message
    assert "1002" in message

    collapsed, _report = collapse_duplicate_source_ids(_fanned_out_table())
    assert_unique_source_ids(table_to_candidates(collapsed, dr))


def test_write_stage_hdf5_refuses_duplicates_before_writing_bytes(
    tmp_path: Path,
) -> None:
    from darkhunter_pop.data_acquisition import (
        DuplicateSourceIdError,
        FunnelCounts,
        compute_stage_diagnostics,
    )

    dr = _dr_config()
    candidates = table_to_candidates(_fanned_out_table(), dr)
    snapshot = SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=4,
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=4, after_quality_cut=4, candidates_written=4),
        quality_cut_bin_counts={"bin0": 4},
    )
    artifact = tmp_path / "stage.h5"
    with pytest.raises(DuplicateSourceIdError):
        write_stage_hdf5(
            artifact, candidates, snapshot=snapshot, diagnostics=diagnostics
        )
    # Nothing on disk: neither the artifact nor a partial sibling.
    assert not artifact.exists()
    assert list(tmp_path.glob("stage.h5*")) == []


def test_write_stage_hdf5_leaves_no_partial_on_failure(tmp_path: Path) -> None:
    """A mid-write failure must not leave a file where plan_stage could cache-hit it."""
    import darkhunter_pop.data_acquisition as da

    from darkhunter_pop.data_acquisition import FunnelCounts, compute_stage_diagnostics

    dr = _dr_config()
    candidates = table_to_candidates(_sample_table()[:2], dr)
    snapshot = SnapshotMeta(
        snapshot_id="test_snap",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=2,
        result_path=tmp_path / "query.ecsv",
        meta_path=tmp_path / "meta.yaml",
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=2, after_quality_cut=2, candidates_written=2),
        quality_cut_bin_counts={"bin0": 2},
    )
    artifact = tmp_path / "stage.h5"

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated mid-write failure")

    original = da._write_stage_hdf5_body
    da._write_stage_hdf5_body = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            write_stage_hdf5(
                artifact, candidates, snapshot=snapshot, diagnostics=diagnostics
            )
    finally:
        da._write_stage_hdf5_body = original  # type: ignore[assignment]
    assert list(tmp_path.glob("stage.h5*")) == []


def test_funnel_reports_duplicate_collapse() -> None:
    from darkhunter_pop.data_acquisition import (
        DuplicateCollapseReport,
        FunnelCounts,
        collapse_duplicate_source_ids,
    )

    _collapsed, report = collapse_duplicate_source_ids(_fanned_out_table())
    funnel = FunnelCounts(
        queried=report.rows_in,
        after_quality_cut=3,
        candidates_written=3,
        duplicate_source_ids=report.duplicate_source_ids,
        duplicate_rows_removed=report.rows_removed,
        after_duplicate_collapse=report.rows_out,
    )
    counts = funnel.as_dict()
    assert counts["queried"] == 4
    assert counts["duplicate_source_ids"] == 1
    assert counts["duplicate_rows_removed"] == 1
    assert counts["after_duplicate_collapse"] == 3

    text = format_funnel_table(funnel, {"bin0": 3}, duplicate_collapse=report)
    assert "duplicate_source_id_collapse" in text
    assert "duplicated_source_ids: 1" in text
    assert "example_source_id: 1002" in text
    assert "max_multiplicity: 2" in text

    clean = DuplicateCollapseReport(rows_in=3, rows_out=3)
    clean_text = format_funnel_table(funnel, {"bin0": 3}, duplicate_collapse=clean)
    assert "duplicated_source_ids: 0" in clean_text
