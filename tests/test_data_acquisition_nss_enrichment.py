"""Regression tests for #308: NSS enrichment merged into the data_acquisition path.

The documented uncut Gaia snapshot (``20260826T234425Z_3d3f740b080c``) was queried
with an ADQL that predates ``corr_vec`` / ``bit_index`` / the NSS-native astrometric
errors, so replaying it reconstructed **0** covariances. The frozen ``nss_enrichment``
snapshot carries those columns; ``data_acquisition`` now overlays it per
``(source_id, nss_solution_type)``. No diagonal-only fallback is ever introduced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from darkhunter_pop.config_loader import audit_dr_independence, load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.data_acquisition import (
    NssEnrichmentError,
    NssEnrichmentIndex,
    SnapshotMeta,
    gaia_snapshots_dir,
    load_nss_enrichment_index,
    merge_nss_enrichment_into_rows,
    read_stage_hdf5,
    resolve_nss_enrichment_meta,
    run_data_acquisition,
    save_gaia_snapshot,
)
from darkhunter_pop.nss_covariance import (
    CovarianceFailure,
    fitted_params_from_bit_index,
    model_param_names,
    pack_corr_vec_upper_triangle,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import StageStatus

pytestmark = pytest.mark.unit

_ORBITAL = model_param_names("Orbital")
assert _ORBITAL is not None
_SHORT = {
    "a_thiele_innes": "A",
    "b_thiele_innes": "B",
    "f_thiele_innes": "F",
    "g_thiele_innes": "G",
}
# Columns the Aug 2026 snapshot's ADQL never selected (enrichment-only).
_ENRICHMENT_ONLY = ("ra_error", "dec_error", "pmra_error", "pmdec_error")


def _spd_corr(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T + n * np.eye(n)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def _split_rows(
    source_ids: list[int], *, bit_index: int = 8191
) -> tuple[Table, Table]:
    """(snapshot-shaped base table without covariance columns, enrichment table)."""
    fitted = fitted_params_from_bit_index(bit_index, _ORBITAL)
    base: dict[str, list] = {
        "source_id": [],
        "nss_solution_type": [],
        "goodness_of_fit": [],
        "g_mag": [],
    }
    enr: dict[str, list] = {
        "SOURCE_ID": [],
        "nss_solution_type": [],
        "corr_vec": [],
        "bit_index": [],
    }
    for name in _ENRICHMENT_ONLY:
        enr[name] = []
    for k, sid in enumerate(source_ids):
        base["source_id"].append(sid)
        base["nss_solution_type"].append("Orbital")
        base["goodness_of_fit"].append(3.0)
        base["g_mag"].append(12.0)
        enr["SOURCE_ID"].append(sid)
        # Gaia TAP's JSON-subtype strings arrive quoted; the join key strips them.
        enr["nss_solution_type"].append('"Orbital"')
        corr = pack_corr_vec_upper_triangle(_spd_corr(len(fitted), seed=k + 1))
        enr["corr_vec"].append("[" + ",".join(repr(float(v)) for v in corr) + "]")
        enr["bit_index"].append(bit_index)
        for i, name in enumerate(fitted):
            value = 1.0 + 0.1 * i
            error = 0.01 * (i + 1)
            if f"{name}_error" in _ENRICHMENT_ONLY:
                enr[f"{name}_error"].append(error)
                base.setdefault(name, []).append(value)
                continue
            col = _SHORT.get(name, name)
            base.setdefault(col, []).append(value)
            base.setdefault(f"{col}_error", []).append(error)
    return Table(base), Table(enr)


def _config(tmp_path: Path, *, enrichment: str | None) -> PipelineConfig:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.paths = cfg.paths.model_copy(
        update={
            "data_root": str(tmp_path / "data"),
            "artifact_root": str(tmp_path / "output"),
        }
    )
    tweaked.dr3 = cfg.dr3.model_copy(update={"nss_enrichment_snapshot": enrichment})
    return tweaked


def _run(tmp_path: Path, cfg: PipelineConfig, base: Table):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)
    snap = save_gaia_snapshot(base, "SELECT old_adql", snapshots_dir=tmp_path / "snaps")
    finished = run_data_acquisition(
        manifest, cfg, run_path=run_path, snapshot_meta_path=snap.meta_path
    )
    assert finished.stages["data_acquisition"].status is StageStatus.COMPLETED
    return stage_artifact_path(
        cfg, STAGE_REGISTRY["data_acquisition"], run_id=finished.run_id
    )


def test_replay_without_enrichment_reconstructs_nothing(tmp_path: Path) -> None:
    """The #308 failure shape: every row fails with MISSING_CORR, none downgraded."""
    base, _enr = _split_rows([11, 12])
    artifact = _run(tmp_path, _config(tmp_path, enrichment=None), base)
    loaded, _ = read_stage_hdf5(artifact)
    assert all(c.nss_solution is None for c in loaded)
    assert {c.extras["nss_covariance_status"] for c in loaded} == {
        CovarianceFailure.MISSING_CORR.value
    }


def test_replay_with_enrichment_reconstructs_full_covariance(tmp_path: Path) -> None:
    base, enr = _split_rows([11, 12])
    cfg0 = _config(tmp_path, enrichment=None)
    enr_meta = save_gaia_snapshot(
        enr, "SELECT corr_vec", snapshots_dir=gaia_snapshots_dir(cfg0)
    )
    cfg = _config(tmp_path, enrichment=enr_meta.snapshot_id)
    artifact = _run(tmp_path, cfg, base)
    loaded, meta = read_stage_hdf5(artifact)
    assert [c.source_id for c in loaded] == [11, 12]
    for candidate in loaded:
        assert candidate.nss_solution is not None
        assert candidate.nss_solution.names == list(_ORBITAL)
        cov = candidate.nss_solution.covariance_array()
        # Full correlated matrix, never a diagonal-only substitute.
        assert np.count_nonzero(cov - np.diag(np.diag(cov))) > 0
    assert meta["nss_enrichment_snapshot_id"] == enr_meta.snapshot_id
    assert meta["nss_enrichment_checksum"] == enr_meta.checksum
    with h5py.File(artifact, "r") as handle:
        attrs = handle["diagnostics"].attrs
        assert attrs["covariance_ok"] == 2
        assert attrs["covariance_failed"] == 0
        assert attrs["nss_enrichment_enabled"] == 1
        assert attrs["nss_enrichment_rows_matched"] == 2
        assert attrs["nss_enrichment_rows_unmatched"] == 0


def test_multi_solution_rows_each_get_their_own_solution() -> None:
    """Join on (source_id, nss_solution_type): never another row's covariance."""
    base = Table(
        {
            "source_id": [7, 7, 8],
            "nss_solution_type": ["Orbital", "SB1", "Orbital"],
            "goodness_of_fit": [1.0, 1.0, 1.0],
        }
    )
    enr = Table(
        {
            "SOURCE_ID": [7, 7, 7],
            "nss_solution_type": ['"SB1"', '"Orbital"', '"EclipsingBinary"'],
            "bit_index": [127, 8191, 99],
        }
    )
    index = _index_from_table(enr)
    rows, counts = merge_nss_enrichment_into_rows(base, index)
    assert [r["bit_index"] if "bit_index" in r else None for r in rows] == [
        8191,
        127,
        None,
    ]
    # Base identity preserved even though the enrichment uses SOURCE_ID / quotes.
    assert [(r["source_id"], r["nss_solution_type"]) for r in rows] == [
        (7, "Orbital"),
        (7, "SB1"),
        (8, "Orbital"),
    ]
    assert counts.as_dict() == {
        "nss_enrichment_enabled": 1,
        "nss_enrichment_rows_matched": 2,
        "nss_enrichment_rows_unmatched": 1,
    }


def test_duplicate_enrichment_key_is_refused(tmp_path: Path) -> None:
    enr = Table(
        {
            "SOURCE_ID": [7, 7],
            "nss_solution_type": ["Orbital", '"Orbital"'],
            "bit_index": [1, 2],
        }
    )
    meta = save_gaia_snapshot(enr, "SELECT 1", snapshots_dir=tmp_path)
    with pytest.raises(NssEnrichmentError, match="duplicate join key"):
        load_nss_enrichment_index(meta.meta_path)


def test_configured_but_missing_enrichment_fails_fast(tmp_path: Path) -> None:
    cfg = _config(tmp_path, enrichment="nss_enrichment")
    with pytest.raises(NssEnrichmentError, match="does not exist"):
        resolve_nss_enrichment_meta(cfg)
    assert resolve_nss_enrichment_meta(_config(tmp_path, enrichment=None)) is None


def test_enrichment_source_is_config_driven_and_fingerprinted() -> None:
    cfg = load_config()
    assert cfg.dr3.nss_enrichment_snapshot == "nss_enrichment"
    assert cfg.dr4.nss_enrichment_snapshot is None
    keys = STAGE_REGISTRY["data_acquisition"].config_fingerprint_keys
    assert "dr3.nss_enrichment_snapshot" in keys
    assert "dr4.nss_enrichment_snapshot" in keys
    assert not [
        m
        for m in audit_dr_independence(cfg).messages()
        if "violation" in m and "nss_enrichment_snapshot" in m
    ]


def _index_from_table(table: Table) -> NssEnrichmentIndex:
    meta = SnapshotMeta(
        snapshot_id="t",
        query_date=datetime.now(tz=timezone.utc),
        adql="",
        checksum="",
        row_count=len(table),
        result_path=Path("x"),
        meta_path=Path("x"),
    )
    row_by_key = {
        (int(r["SOURCE_ID"]), str(r["nss_solution_type"]).strip('"')): i
        for i, r in enumerate(table)
    }
    return NssEnrichmentIndex(meta=meta, table=table, row_by_key=row_by_key)
