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
    assert "nss.A," not in adql.replace("\n", "")
    assert "phot_g_mean_mag_error" not in adql
    assert "LEFT JOIN" in adql
    assert "galex_ais_best_neighbour" not in adql  # disabled


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


def test_format_funnel_table_is_legible() -> None:
    from darkhunter_pop.data_acquisition import FunnelCounts

    text = format_funnel_table(
        FunnelCounts(queried=10, after_quality_cut=4, candidates_written=4),
        {"bin0": 4},
    )
    assert "data_acquisition funnel" in text
    assert "queried" in text


@pytest.mark.network
def test_live_gaia_nss_query_smoke() -> None:
    """Optional live archive query; skipped in required CI marker set."""
    from darkhunter_pop.data_acquisition import default_gaia_query

    dr = load_config().dr3
    adql = (
        f"SELECT TOP 3 nss.source_id, nss.goodness_of_fit, gs.phot_g_mean_mag "
        f"FROM {dr.nss_table} AS nss "
        f"JOIN {dr.gaia_source_table} AS gs ON nss.source_id = gs.source_id"
    )
    table = default_gaia_query(adql, dr)
    assert len(table) <= 3
    assert "source_id" in table.colnames
