"""Shahaf et al. (2023b) class-III catalog snapshot + El-Badry 2026 reproduction cross-match (#315)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from astropy.table import Table

from darkhunter_pop.config_loader import audit_dr_independence, load_config, repo_root
from darkhunter_pop.config_schema import (
    PATH_SPECIFIC_LEAF_KEYS,
    SampleSelectionEntry,
    SampleSelectionMode,
)
from darkhunter_pop.sample_selection import (
    SampleSelection,
    SampleSelectionRegistry,
    load_sample_selection_file,
)
from darkhunter_pop.shahaf2023b_catalog import (
    SHAHAF2023B_CLASS3_CATALOG_ID,
    Shahaf2023bCatalogError,
    file_sha256,
    load_shahaf2023b_class3,
    merge_shahaf2023b_columns,
)

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "selections" / "shahaf2023b_class3_fixture"


def _write_snapshot(directory: Path, rows: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    cols = ("GaiaDR3", "M2min", "e_M2min", "M1", "P", "Class")
    Table({c: [r[c] for r in rows] for c in cols}).write(
        directory / "table2.ecsv", format="ascii.ecsv", overwrite=True
    )
    meta = {
        "snapshot_id": directory.name,
        "row_count": len(rows),
        "sha256": file_sha256(directory / "table2.ecsv"),
    }
    (directory / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return directory


def _cat_row(sid: int, m2: float = 1.2, e_m2: float = 0.05) -> dict[str, Any]:
    return {"GaiaDR3": sid, "M2min": m2, "e_M2min": e_m2, "M1": 1.0, "P": 400.0, "Class": "NS"}


def _row(sid: int, **kw: Any) -> dict[str, Any]:
    row = {
        "source_id": sid,
        "nss_solution_type": "Orbital",
        "main_sequence": True,
        "m1_tilde_msun": 1.0,
        # Janssens M̃2 far outside the window: reproduction must not read it.
        "m2_tilde_msun": 2.0,
        "amrf": 0.5,
        "goodness_of_fit": 1.0,
        "period_day": 400.0,
        "phot_g_mean_mag": 12.0,
        "sigma_m2_astrometric_msun": 0.5,
    }
    row.update(kw)
    return row


def _registry(snapshot: str | None) -> SampleSelectionRegistry:
    cfg = load_config().model_copy(deep=True)
    cfg.dr3 = cfg.dr3.model_copy(update={"shahaf2023b_class3_snapshot": snapshot})
    cfg.sample_selection.samples = [
        SampleSelectionEntry(
            name="andrews2022",
            enabled=True,
            path=str(Path(__file__).resolve().parent / "fixtures" / "selections" / "andrews2022_stub.yaml"),
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


def _sub_chandra(sel: SampleSelection, rows: list[dict[str, Any]]) -> set[int]:
    # Rows carry their own sigma_M2~; skip the (laptop-only, ~0.5 GB) NSS
    # enrichment read the forward_model sigma cut would otherwise trigger.
    sel._nss_enrichment_index = lambda: {}  # type: ignore[method-assign]
    result = sel.evaluate(rows, membership={"andrews2022": frozenset()})
    return set(result.subsample_surviving["sub_chandrasekhar"])


def test_fixture_snapshot_loads_and_verifies_checksum(tmp_path: Path) -> None:
    catalog = load_shahaf2023b_class3(FIXTURE)
    assert sorted(catalog) == list(range(200, 222))
    assert catalog[200]["shahaf2023b_m2min_msun"] == pytest.approx(1.2)
    assert catalog[200]["shahaf2023b_m2min_error_msun"] == pytest.approx(0.05)
    tampered = shutil.copytree(FIXTURE, tmp_path / "tampered")
    text = (tampered / "table2.ecsv").read_text(encoding="utf-8")
    (tampered / "table2.ecsv").write_text(text.replace("1.2", "1.3", 1), encoding="utf-8")
    with pytest.raises(Shahaf2023bCatalogError, match="checksum"):
        load_shahaf2023b_class3(tampered)


def test_merge_adds_columns_to_members_only() -> None:
    merged = merge_shahaf2023b_columns(
        [{"source_id": 200}, {"source_id": 1}], load_shahaf2023b_class3(FIXTURE)
    )
    assert merged[0]["shahaf2023b_m2min_msun"] == pytest.approx(1.2)
    assert "shahaf2023b_m2min_msun" not in merged[1]


def test_reproduction_is_catalog_membership_plus_eq5_on_shahaf_columns(tmp_path: Path) -> None:
    snap = _write_snapshot(
        tmp_path / "shahaf",
        [
            _cat_row(1),  # passes
            _cat_row(2, m2=1.42),  # above the frozen 1.40
            _cat_row(3, m2=1.04),  # below the frozen 1.05
            _cat_row(4, e_m2=0.11),  # sigma above 0.105
            _cat_row(5),  # fails P (row below)
            _cat_row(6),  # fails G (row below)
            _cat_row(7),  # not main sequence (row below)
        ],
    )
    rows = [
        _row(1),
        _row(2),
        _row(3),
        _row(4),
        _row(5, period_day=901.0),
        _row(6, phot_g_mean_mag=15.0),
        _row(7, main_sequence=False),
        # Not in the catalog, but passes every Janssens (forward_model) cut.
        _row(8, m2_tilde_msun=1.2, amrf=0.709, sigma_m2_astrometric_msun=0.05),
    ]
    sel = _registry(str(snap)).selection("elbadry2026")
    assert _sub_chandra(sel, [dict(r) for r in rows]) == {1}
    # Forward model is the unchanged schema-v2 Janssens chain: only row 8.
    fm = SampleSelection(sel.spec, mode=SampleSelectionMode.FORWARD_MODEL)
    assert _sub_chandra(fm, [dict(r) for r in rows]) == {8}


def test_forward_model_never_reads_the_catalog() -> None:
    def _boom(_catalog_id: str):
        raise AssertionError("forward_model must not load the Shahaf catalog")

    spec = load_sample_selection_file(repo_root() / "config/selections/elbadry2026.yaml")
    fm = SampleSelection(spec, mode=SampleSelectionMode.FORWARD_MODEL, external_catalog_loader=_boom)
    assert _sub_chandra(fm, [_row(8, m2_tilde_msun=1.2, amrf=0.709, sigma_m2_astrometric_msun=0.05)]) == {8}


def test_null_snapshot_keeps_no_source_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    sel = _registry(None).selection("elbadry2026")
    with caplog.at_level(logging.WARNING):
        assert _sub_chandra(sel, [_row(200, m2_tilde_msun=1.2)]) == set()
    assert "shahaf2023b_class3" in caplog.text


def test_configured_but_missing_snapshot_raises() -> None:
    sel = _registry("does_not_exist_snapshot").selection("elbadry2026")
    with pytest.raises(Shahaf2023bCatalogError, match="does not exist"):
        sel.evaluate([_row(200)], membership={"andrews2022": frozenset()})


def test_frozen_file_v3_reproduction_cuts_keep_published_thresholds() -> None:
    spec = load_sample_selection_file(repo_root() / "config/selections/elbadry2026.yaml")
    assert spec.schema_version == 3
    astro = next(b for b in spec.branches if b.id == "astrometric")
    sub = next(s for s in astro.subsamples if s.id == "sub_chandrasekhar")
    assert sub.external_catalog == SHAHAF2023B_CLASS3_CATALOG_ID
    repro = [c for c in sub.cuts if SampleSelectionMode.REPRODUCTION in c.applies_to]
    assert [c.id for c in repro] == [
        "shahaf2023b_class3_membership",
        "main_sequence",
        "m2_range_shahaf2023b",
        "m2_error_shahaf2023b",
        "period",
        "g_mag",
    ]
    cuts = {c.id: c for c in repro}
    assert cuts["m2_range_shahaf2023b"].parameters == {"m2_msun_min": 1.05, "m2_msun_max": 1.40}
    assert cuts["m2_error_shahaf2023b"].parameters == {"m2_msun_error_max": 0.105}
    assert cuts["period"].parameters == {"period_days_max": 900.0}
    assert cuts["g_mag"].parameters == {"g_mag_faint_limit": 15.0}
    assert next(i for i in spec.open_items if i.id == "315").status == "resolved"


def test_snapshot_key_is_dr_path_specific() -> None:
    cfg = load_config()
    assert "shahaf2023b_class3_snapshot" in PATH_SPECIFIC_LEAF_KEYS
    assert cfg.dr3.shahaf2023b_class3_snapshot is not None
    assert cfg.dr4.shahaf2023b_class3_snapshot is None
    assert not [f for f in audit_dr_independence(cfg).findings if f.severity == "violation"]
