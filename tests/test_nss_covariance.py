"""Round-trip tests for Gaia NSS corr_vec / bit_index covariance reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.data_acquisition import (
    FunnelCounts,
    SnapshotMeta,
    build_nss_adql,
    compute_stage_diagnostics,
    format_funnel_table,
    read_stage_hdf5,
    table_row_to_candidate,
    write_stage_hdf5,
)
from darkhunter_pop.nss_covariance import (
    CovarianceFailure,
    assert_symmetric_psd,
    correlation_to_covariance,
    fitted_params_from_bit_index,
    model_param_names,
    pack_corr_vec_upper_triangle,
    reconstruct_nss_covariance,
    unpack_corr_vec_upper_triangle,
    validate_loaded_nss_solution,
)

pytestmark = pytest.mark.unit

_ORBITAL_PARAMS = model_param_names("Orbital")
assert _ORBITAL_PARAMS is not None


def _spd_corr(n: int, seed: int = 0) -> np.ndarray:
    """Build a random SPD correlation matrix (unit diagonal)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T + n * np.eye(n)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def _orbital_row(
    *,
    bit_index: int = 8191,
    corr: np.ndarray | None = None,
    seed: int = 1,
) -> dict:
    assert _ORBITAL_PARAMS is not None
    fitted = fitted_params_from_bit_index(bit_index, _ORBITAL_PARAMS)
    n = len(fitted)
    if corr is None:
        corr = _spd_corr(n, seed=seed)
    errors = {name: 0.01 * (i + 1) for i, name in enumerate(fitted)}
    values = {name: 1.0 + 0.1 * i for i, name in enumerate(fitted)}
    row: dict = {
        "source_id": 42,
        "nss_solution_type": "Orbital",
        "bit_index": bit_index,
        "corr_vec": pack_corr_vec_upper_triangle(corr),
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    shorts = {
        "a_thiele_innes": "A",
        "b_thiele_innes": "B",
        "f_thiele_innes": "F",
        "g_thiele_innes": "G",
    }
    for name in fitted:
        if name in shorts:
            short = shorts[name]
            row[short] = values[name]
            row[f"{short}_error"] = errors[name]
            continue
        row[name] = values[name]
        row[f"{name}_error"] = errors[name]
    return row


def test_bit_index_orbital_8191_and_8179() -> None:
    assert _ORBITAL_PARAMS is not None
    assert fitted_params_from_bit_index(8191, _ORBITAL_PARAMS) == list(_ORBITAL_PARAMS)
    fitted_10 = fitted_params_from_bit_index(8179, _ORBITAL_PARAMS)
    assert "g_thiele_innes" not in fitted_10
    assert "eccentricity" not in fitted_10
    assert len(fitted_10) == 10


def test_pack_unpack_round_trip_gaia_layout() -> None:
    """Published column-major upper-triangle packing must round-trip exactly."""
    n = 12
    corr = _spd_corr(n, seed=7)
    packed = pack_corr_vec_upper_triangle(corr)
    assert packed.shape == (231,)
    # First off-diagonal in column-major upper triangle is corr[0,1].
    assert packed[0] == pytest.approx(corr[0, 1])
    assert packed[1] == pytest.approx(corr[0, 2])
    assert packed[2] == pytest.approx(corr[1, 2])
    restored = unpack_corr_vec_upper_triangle(packed, n)
    assert np.allclose(restored, corr, rtol=1e-12, atol=1e-14)


def test_reconstruct_orbital_12_param() -> None:
    assert _ORBITAL_PARAMS is not None
    corr = _spd_corr(12, seed=3)
    row = _orbital_row(bit_index=8191, corr=corr)
    result = reconstruct_nss_covariance(row)
    assert result.ok
    assert result.parameter_set is not None
    assert result.parameter_set.names == list(_ORBITAL_PARAMS)
    assert "gaiadr3.nss_two_body_orbit:Orbital" in result.parameter_set.provenance
    cov = result.parameter_set.covariance_array()
    assert assert_symmetric_psd(cov) is None
    # Diagonal must match sigma^2.
    for i, name in enumerate(_ORBITAL_PARAMS):
        sigma = row[f"{name}_error"] if f"{name}_error" in row else row[
            {"a_thiele_innes": "A_error", "b_thiele_innes": "B_error", "f_thiele_innes": "F_error", "g_thiele_innes": "G_error"}[name]
        ]
        assert cov[i, i] == pytest.approx(sigma ** 2)


def test_reconstruct_orbital_10_param_pseudo_circular() -> None:
    corr = _spd_corr(10, seed=5)
    row = _orbital_row(bit_index=8179, corr=corr, seed=5)
    result = reconstruct_nss_covariance(row)
    assert result.ok
    assert result.parameter_set is not None
    assert result.n_param == 10
    assert "g_thiele_innes" not in result.parameter_set.names
    assert "eccentricity" not in result.parameter_set.names


def test_no_diagonal_fallback_on_missing_corr() -> None:
    row = _orbital_row()
    del row["corr_vec"]
    del row["bit_index"]
    result = reconstruct_nss_covariance(row)
    assert not result.ok
    assert result.parameter_set is None
    assert result.status is CovarianceFailure.MISSING_CORR


def test_no_diagonal_fallback_on_non_psd() -> None:
    # Classic 3×3 indefinite correlation, embedded in a 12-d identity.
    bad3 = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    assert assert_symmetric_psd(correlation_to_covariance(bad3, [1.0, 1.0, 1.0])) is (
        CovarianceFailure.NON_PSD
    )
    full = np.eye(12)
    full[:3, :3] = bad3
    row = _orbital_row(bit_index=8191, corr=full)
    result = reconstruct_nss_covariance(row)
    assert not result.ok
    assert result.parameter_set is None
    assert result.status is CovarianceFailure.NON_PSD


def test_eclipsing_escalates_as_unsupported_q4() -> None:
    row = {
        "source_id": 1,
        "nss_solution_type": "EclipsingBinary",
        "bit_index": 1329216,
        "corr_vec": np.zeros(231),
    }
    result = reconstruct_nss_covariance(row)
    assert result.status is CovarianceFailure.UNSUPPORTED_SOLUTION_TYPE
    assert result.parameter_set is None


def test_build_nss_adql_selects_corr_vec_and_bit_index() -> None:
    adql = build_nss_adql(load_config().dr3)
    assert "nss.corr_vec" in adql
    assert "nss.bit_index" in adql
    assert "nss.ra_error" in adql
    assert "nss.pmra_error" in adql


def test_candidate_and_hdf5_persist_nss_solution(tmp_path: Path) -> None:
    row = _orbital_row()
    candidate = table_row_to_candidate(row, load_config().dr3)
    assert candidate.nss_solution is not None
    assert candidate.nss_solution.names[0] == "ra"

    table = Table(
        {
            "source_id": [candidate.source_id],
            "nss_solution_type": ["Orbital"],
            "ra": [row["ra"]],
            "dec": [row["dec"]],
            "parallax": [row["parallax"]],
            "goodness_of_fit": [3.0],
            "g_mag": [12.0],
            "period": [row["period"]],
            "eccentricity": [row["eccentricity"]],
            "bit_index": [row["bit_index"]],
            "corr_vec": [row["corr_vec"]],
            "ra_error": [row["ra_error"]],
            "dec_error": [row["dec_error"]],
            "parallax_error": [row["parallax_error"]],
            "pmra": [row["pmra"]],
            "pmra_error": [row["pmra_error"]],
            "pmdec": [row["pmdec"]],
            "pmdec_error": [row["pmdec_error"]],
            "period_error": [row["period_error"]],
            "t_periastron": [row["t_periastron"]],
            "t_periastron_error": [row["t_periastron_error"]],
            "eccentricity_error": [row["eccentricity_error"]],
            "A": [row["A"]],
            "A_error": [row["A_error"]],
            "B": [row["B"]],
            "B_error": [row["B_error"]],
            "F": [row["F"]],
            "F_error": [row["F_error"]],
            "G": [row["G"]],
            "G_error": [row["G_error"]],
        }
    )
    from darkhunter_pop.data_acquisition import table_to_candidates

    candidates = table_to_candidates(table, load_config().dr3)
    assert candidates[0].nss_solution is not None
    funnel = FunnelCounts(
        queried=1,
        after_quality_cut=1,
        candidates_written=1,
        covariance_ok=1,
        covariance_failed=0,
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=funnel,
        quality_cut_bin_counts={},
    )
    assert diagnostics.covariance_health is not None
    assert diagnostics.covariance_health.ok == 1
    artifact = tmp_path / "da.h5"
    snapshot = SnapshotMeta(
        snapshot_id="cov",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="x",
        row_count=1,
        result_path=tmp_path / "q.ecsv",
        meta_path=tmp_path / "m.yaml",
    )
    write_stage_hdf5(artifact, candidates, snapshot=snapshot, diagnostics=diagnostics)
    loaded, _meta = read_stage_hdf5(artifact)
    assert loaded[0].nss_solution is not None
    assert validate_loaded_nss_solution(loaded[0].nss_solution) is None
    with h5py.File(artifact, "r") as handle:
        assert "data_acquisition/nss_covariance" in handle
        assert handle["data_acquisition/nss_covariance"].attrs["covariance_ok"] == 1
        assert str(candidate.source_id) in handle["data_acquisition/nss_covariance/matrices"]

    text = format_funnel_table(
        funnel, {}, covariance_health=diagnostics.covariance_health
    )
    assert "covariance_ok" in text
    assert "covariance_health (by solution type)" in text
    assert "Orbital" in text
