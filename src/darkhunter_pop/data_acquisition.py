"""Stage: ``data_acquisition``.

Gaia NSS query, cross-matched photometry, N-bin goodness-of-fit quality cut, query
snapshotting, and stage HDF5 output (ARCHITECTURE.md §4).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from astropy.table import Table
from numpy.typing import NDArray

from darkhunter_pop.config_loader import load_config, repo_root, require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    DRPathConfig,
    ExternalPhotometryCrossmatch,
    PipelineConfig,
    QualityCutBin,
    SpectroscopicMassFunctionConfig,
)
from darkhunter_pop.diagnostics import (
    emit_funnel_sky,
    format_funnel_report,
    resolve_diagnostic_dirs,
)
from darkhunter_pop.forward_model import SIX_PANEL_NAMES, SOLUTION_TYPE_LABELS
from darkhunter_pop.nss_covariance import (
    CovarianceFailure,
    CovarianceHealth,
    CovarianceResult,
    reconstruct_nss_covariance,
    validate_loaded_nss_solution,
)
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    spectroscopic_mass_function,
    thiele_innes_to_campbell,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.rv_adapter import attach_rv_summaries
from darkhunter_pop.schemas import (
    CandidateRecord,
    PhotometryPoint,
    RunManifest,
    StageStatus,
    ThieleInnesElements,
)

_GAIA_SOURCE_MAG_FIELDS: dict[str, tuple[str, str | None]] = {
    # DR3 gaia_source has phot_*_mean_mag but not phot_*_mean_mag_error columns.
    "G": ("phot_g_mean_mag", None),
    "BP": ("phot_bp_mean_mag", None),
    "RP": ("phot_rp_mean_mag", None),
}

# Internal short names (schemas.ThieleInnesElements) → Gaia ADQL column stems.
_THIELE_INNES_ADQL: dict[str, str] = {
    "A": "a_thiele_innes",
    "B": "b_thiele_innes",
    "F": "f_thiele_innes",
    "G": "g_thiele_innes",
}
_THIELE_INNES_FIELDS: tuple[str, ...] = tuple(_THIELE_INNES_ADQL.keys())
_THIELE_INNES_ERR_SUFFIX = "_error"

# Astrophysical-parameters columns for mass_derivation (MSC preferred, gspphot fallback).
_AP_PARAM_STEMS: tuple[str, ...] = (
    "teff_msc1",
    "logg_msc1",
    "mh_msc",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
)


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadata for one Gaia archive query snapshot."""

    snapshot_id: str
    query_date: datetime
    adql: str
    checksum: str
    row_count: int
    result_path: Path
    meta_path: Path


@dataclass(frozen=True)
class FunnelCounts:
    """Row counts at each data_acquisition filtering step."""

    queried: int
    after_quality_cut: int
    candidates_written: int
    covariance_ok: int = 0
    covariance_failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "queried": self.queried,
            "after_quality_cut": self.after_quality_cut,
            "candidates_written": self.candidates_written,
            "covariance_ok": self.covariance_ok,
            "covariance_failed": self.covariance_failed,
        }


@dataclass(frozen=True)
class SB1ReproductionRoute:
    """El-Badry 2026 §8.4 route flags for reproduction checks only.

    Not an inference membership bit. ``in_union`` is the published 151-source
    disjunction after the K1 significance gate; it does not authorize a
    population-likelihood term.
    """

    k1_significant: bool
    high_mass_function: bool
    main_sequence_min_companion_mass: bool
    in_union: bool


@dataclass(frozen=True)
class StageDiagnostics:
    """Structured diagnostics emitted by ``data_acquisition``."""

    funnel: FunnelCounts
    quality_cut_bin_counts: dict[str, int]
    ruwe: NDArray[np.floating] | None
    period_day: NDArray[np.floating] | None
    eccentricity: NDArray[np.floating] | None
    ra_deg: NDArray[np.floating] | None
    dec_deg: NDArray[np.floating] | None
    nss_panels: dict[str, NDArray[np.floating]]
    solution_type_fractions: dict[str, float]
    covariance_health: CovarianceHealth | None = None


def gaia_snapshots_dir(config: PipelineConfig) -> Path:
    """Return ``{data_root}/{dr}/gaia_snapshots/`` for the active DR mode."""
    root = Path(config.paths.data_root)
    if not root.is_absolute():
        root = repo_root() / root
    return root / config.active_dr_mode.value / "gaia_snapshots"


def _bin_label(index: int, bin_cfg: QualityCutBin) -> str:
    g_min = bin_cfg.g_min
    g_max = bin_cfg.g_max
    if g_min is None and g_max is not None:
        return f"bin{index}_G<={g_max}_gof<={bin_cfg.gof_max}"
    if g_max is None and g_min is not None:
        return f"bin{index}_G>{g_min}_gof<={bin_cfg.gof_max}"
    return (
        f"bin{index}_{g_min}<G<={g_max}_gof<={bin_cfg.gof_max}"
    )


def quality_bin_for_star(
    g_mag: float | None,
    goodness_of_fit: float | None,
    bins: Sequence[QualityCutBin],
) -> int | None:
    """Return the index of the matching quality-cut bin, or ``None`` if unclassified."""
    if g_mag is None or goodness_of_fit is None or np.isnan(g_mag) or np.isnan(goodness_of_fit):
        return None
    for index, bin_cfg in enumerate(bins):
        above_min = bin_cfg.g_min is None or g_mag > bin_cfg.g_min
        at_or_below_max = bin_cfg.g_max is None or g_mag <= bin_cfg.g_max
        if above_min and at_or_below_max:
            return index
    return None


def passes_quality_cut(
    g_mag: float | None,
    goodness_of_fit: float | None,
    bins: Sequence[QualityCutBin],
) -> bool:
    """True when the star falls in a configured bin and meets that bin's gof threshold."""
    index = quality_bin_for_star(g_mag, goodness_of_fit, bins)
    if index is None:
        return False
    if goodness_of_fit is None or np.isnan(goodness_of_fit):
        return False
    return float(goodness_of_fit) <= bins[index].gof_max


def resolve_quality_g_column(
    table: Table,
    *,
    preferred: str | None = None,
) -> str:
    """Pick the G magnitude column used by quality cuts.

    Production ADQL aliases ``phot_g_mean_mag`` as ``g_mag``; fixtures / legacy
    snapshots may still use the Gaia catalog name.
    """
    candidates: list[str] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(("g_mag", "phot_g_mean_mag"))
    for name in candidates:
        if name in table.colnames:
            return name
    raise KeyError(
        "table missing G magnitude column "
        f"(tried {candidates!r}; have {list(table.colnames)!r})"
    )


def apply_quality_cuts(
    table: Table,
    bins: Sequence[QualityCutBin],
    *,
    g_column: str | None = None,
    gof_column: str = "goodness_of_fit",
) -> tuple[Table, dict[str, int]]:
    """Filter ``table`` with the N-bin quality-cut scheme; return per-bin survivor counts."""
    g_col = resolve_quality_g_column(table, preferred=g_column)
    if gof_column not in table.colnames:
        raise KeyError(f"table missing {gof_column!r}")

    keep = np.zeros(len(table), dtype=bool)
    bin_counts = {f"{_bin_label(i, b)}_pass": 0 for i, b in enumerate(bins)}
    bin_counts["unclassified"] = 0
    bin_counts["failed_gof"] = 0

    g_values = np.asarray(table[g_col], dtype=np.float64)
    gof_values = np.asarray(table[gof_column], dtype=np.float64)

    for row_index in range(len(table)):
        g_mag = float(g_values[row_index]) if np.isfinite(g_values[row_index]) else None
        gof = float(gof_values[row_index]) if np.isfinite(gof_values[row_index]) else None
        bin_index = quality_bin_for_star(g_mag, gof, bins)
        if bin_index is None:
            bin_counts["unclassified"] += 1
            continue
        label = _bin_label(bin_index, bins[bin_index])
        if gof is not None and gof <= bins[bin_index].gof_max:
            keep[row_index] = True
            bin_counts[f"{label}_pass"] += 1
        else:
            bin_counts["failed_gof"] += 1

    return table[keep], bin_counts


def _enabled_crossmatches(
    crossmatches: Sequence[ExternalPhotometryCrossmatch],
) -> list[ExternalPhotometryCrossmatch]:
    return [match for match in crossmatches if match.enabled]


def _split_join_eq(spec: str) -> tuple[str, str]:
    if spec.count("=") != 1:
        raise ValueError(f"join key must be 'left=right', got {spec!r}")
    left, right = spec.split("=")
    left, right = left.strip(), right.strip()
    if not left or not right:
        raise ValueError(f"join key must be 'left=right', got {spec!r}")
    return left, right


def build_nss_enrichment_adql(nss_table: str = "gaiadr3.nss_two_body_orbit") -> str:
    """NSS-only ADQL for columns missing from the Aug 2026 photometry snapshot.

    Returns ``corr_vec`` / ``bit_index`` (covariance), K1 + errors, ``significance``,
    and NSS-native astrometric errors needed by ``reconstruct_nss_covariance``.
    No photometry joins — much lighter than ``build_nss_adql``.
    """
    cols = [
        "nss.source_id",
        "nss.nss_solution_type",
        "nss.ra",
        "nss.dec",
        "nss.ra_error",
        "nss.dec_error",
        "nss.parallax",
        "nss.parallax_error",
        "nss.pmra",
        "nss.pmdec",
        "nss.pmra_error",
        "nss.pmdec_error",
        "nss.period",
        "nss.period_error",
        "nss.input_period_error",
        "nss.t_periastron",
        "nss.t_periastron_error",
        "nss.eccentricity",
        "nss.eccentricity_error",
        "nss.a_thiele_innes",
        "nss.a_thiele_innes_error",
        "nss.b_thiele_innes",
        "nss.b_thiele_innes_error",
        "nss.f_thiele_innes",
        "nss.f_thiele_innes_error",
        "nss.g_thiele_innes",
        "nss.g_thiele_innes_error",
        "nss.c_thiele_innes",
        "nss.c_thiele_innes_error",
        "nss.h_thiele_innes",
        "nss.h_thiele_innes_error",
        "nss.center_of_mass_velocity",
        "nss.center_of_mass_velocity_error",
        "nss.semi_amplitude_primary",
        "nss.semi_amplitude_primary_error",
        "nss.semi_amplitude_secondary",
        "nss.semi_amplitude_secondary_error",
        "nss.arg_periastron",
        "nss.arg_periastron_error",
        "nss.corr_vec",
        "nss.bit_index",
        "nss.goodness_of_fit",
        "nss.significance",
    ]
    return (
        "SELECT\n  "
        + ",\n  ".join(cols)
        + f"\nFROM {nss_table} AS nss"
    )


def merge_nss_enrichment_into_row(
    base: Mapping[str, Any],
    enrichment: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay enrichment NSS columns onto a photometry-snapshot row.

    Enrichment wins for K1 / corr_vec / bit_index / significance / NSS errors.
    Photometry and atmosphere columns on ``base`` are preserved.

    Gaia TAP may emit ``SOURCE_ID`` (uppercase); normalize to ``source_id``.
    """
    out = dict(base)
    for key, value in enrichment.items():
        name = str(key)
        if name.upper() == "SOURCE_ID":
            name = "source_id"
        if name in ("source_id", "nss_solution_type"):
            # Keep base identity; solution type must already match the join key.
            continue
        if value is None:
            continue
        try:
            if np.ma.is_masked(value):
                continue
        except (TypeError, ValueError):
            pass
        out[name] = value
    # Thiele-Innes aliases used by reconstruct / selection enrich.
    for short, long_name in (
        ("A", "a_thiele_innes"),
        ("B", "b_thiele_innes"),
        ("F", "f_thiele_innes"),
        ("G", "g_thiele_innes"),
    ):
        if long_name in out and short not in out:
            out[short] = out[long_name]
        err_long = f"{long_name}_error"
        err_short = f"{short}_error"
        if err_long in out and err_short not in out:
            out[err_short] = out[err_long]
    return out


def _enrichment_join_key(mapping: Mapping[str, Any]) -> tuple[int, str]:
    """``(source_id, nss_solution_type)`` with Gaia ``SOURCE_ID`` normalization."""
    sid_raw = mapping.get("source_id", mapping.get("SOURCE_ID"))
    sol_raw = mapping.get("nss_solution_type", "")
    return int(sid_raw), str(sol_raw).strip('"')


def build_nss_adql(dr: DRPathConfig) -> str:
    """Build the literal ADQL for the NSS catalog + photometry cross-matches.

    Astrometry (ra/dec/parallax/pm) is ``COALESCE(nss.*, gs.*)``: Orbital* rows
    keep NSS solution values when filled; SB1 / EclipsingBinary / etc. fall back to
    ``gaia_source`` because those solution types leave NSS astrometry null by design.
    Thiele-Innes (A/B/F/G) stay NSS-only and are null for non-astrometric solutions.
    """
    select_parts = [
        "nss.source_id",
        "nss.nss_solution_type",
        # NSS columns are sparsely filled by solution type; fall back to gaia_source.
        "COALESCE(nss.ra, gs.ra) AS ra",
        "COALESCE(nss.dec, gs.dec) AS dec",
        "COALESCE(nss.parallax, gs.parallax) AS parallax",
        "COALESCE(nss.parallax_error, gs.parallax_error) AS parallax_error",
        "COALESCE(nss.pmra, gs.pmra) AS pmra",
        "COALESCE(nss.pmdec, gs.pmdec) AS pmdec",
        # NSS-solution errors for corr_vec reconstruction (never gaia_source fallback).
        "nss.ra_error",
        "nss.dec_error",
        "nss.pmra_error",
        "nss.pmdec_error",
        "nss.period",
        # EclipsingBinary: period σ is in input_period_error, not period_error (Gaia datamodel).
        "COALESCE(nss.period_error, nss.input_period_error) AS period_error",
        "nss.t_periastron",
        "nss.t_periastron_error",
        "nss.eccentricity",
        "nss.eccentricity_error",
        "nss.c_thiele_innes",
        "nss.c_thiele_innes_error",
        "nss.h_thiele_innes",
        "nss.h_thiele_innes_error",
        "nss.center_of_mass_velocity",
        "nss.center_of_mass_velocity_error",
        "nss.semi_amplitude_primary",
        "nss.semi_amplitude_primary_error",
        "nss.semi_amplitude_secondary",
        "nss.semi_amplitude_secondary_error",
        "nss.arg_periastron",
        "nss.arg_periastron_error",
        "nss.corr_vec",
        "nss.bit_index",
        "nss.goodness_of_fit",
        "gs.ruwe",
    ]
    for short, adql_stem in _THIELE_INNES_ADQL.items():
        select_parts.append(f"nss.{adql_stem} AS {short}")
        select_parts.append(
            f"nss.{adql_stem}{_THIELE_INNES_ERR_SUFFIX} AS {short}{_THIELE_INNES_ERR_SUFFIX}"
        )

    # MSC / gspphot atmospheric parameters for mass_derivation_bulk (ARCHITECTURE.md §4).
    for stem in _AP_PARAM_STEMS:
        select_parts.append(f"ap.{stem}")
        select_parts.append(f"ap.{stem}_upper")
        select_parts.append(f"ap.{stem}_lower")

    for band in dr.gaia_source_photometry_bands:
        if band not in _GAIA_SOURCE_MAG_FIELDS:
            raise ValueError(f"unsupported Gaia source photometry band: {band!r}")
        mag_col, err_col = _GAIA_SOURCE_MAG_FIELDS[band]
        select_parts.append(f"gs.{mag_col} AS {band.lower()}_mag")
        if err_col is not None:
            select_parts.append(f"gs.{err_col} AS {band.lower()}_mag_err")

    enabled = _enabled_crossmatches(dr.external_photometry_crossmatches)
    # One JOIN chain per unique neighbour(+join)+catalog graph; bands share aliases.
    neighbour_alias: dict[str, str] = {}
    join_alias: dict[str, str] = {}
    catalog_alias: dict[tuple[str, str | None, str], str] = {}

    for match in enabled:
        if match.neighbour_table not in neighbour_alias:
            neighbour_alias[match.neighbour_table] = f"nb_{len(neighbour_alias)}"
        if match.join_table and match.join_table not in join_alias:
            join_alias[match.join_table] = f"xj_{len(join_alias)}"
        cat_key = (match.neighbour_table, match.join_table, match.catalog_table)
        if cat_key not in catalog_alias:
            catalog_alias[cat_key] = f"cat_{len(catalog_alias)}"
        cat_ref = catalog_alias[cat_key]
        select_parts.append(f"{cat_ref}.{match.mag_column} AS {match.band}_mag")
        if match.mag_err_column is not None:
            select_parts.append(
                f"{cat_ref}.{match.mag_err_column} AS {match.band}_mag_err"
            )

    lines = ["SELECT", "  " + ",\n  ".join(select_parts)]
    lines.append(f"FROM {dr.nss_table} AS nss")
    lines.append(
        "JOIN {0} AS gs ON nss.source_id = gs.source_id".format(dr.gaia_source_table)
    )
    # Astrophysical parameters share the same DR catalog family as gaia_source.
    ap_table = dr.gaia_source_table.replace("gaia_source", "astrophysical_parameters")
    lines.append(
        f"LEFT JOIN {ap_table} AS ap ON nss.source_id = ap.source_id"
    )

    for neighbour_table, alias in neighbour_alias.items():
        lines.append(
            f"LEFT JOIN {neighbour_table} AS {alias} "
            f"ON nss.source_id = {alias}.source_id"
        )

    emitted_joins: set[str] = set()
    emitted_catalogs: set[tuple[str, str | None, str]] = set()
    for match in enabled:
        nb_ref = neighbour_alias[match.neighbour_table]
        cat_key = (match.neighbour_table, match.join_table, match.catalog_table)
        cat_ref = catalog_alias[cat_key]

        if match.join_table:
            xj_ref = join_alias[match.join_table]
            if match.join_table not in emitted_joins:
                left, right = _split_join_eq(match.neighbour_to_join or "")
                lines.append(
                    f"LEFT JOIN {match.join_table} AS {xj_ref} "
                    f"ON {nb_ref}.{left} = {xj_ref}.{right}"
                )
                emitted_joins.add(match.join_table)
            if cat_key not in emitted_catalogs:
                left, right = _split_join_eq(match.join_to_catalog or "")
                lines.append(
                    f"LEFT JOIN {match.catalog_table} AS {cat_ref} "
                    f"ON {xj_ref}.{left} = {cat_ref}.{right}"
                )
                emitted_catalogs.add(cat_key)
        else:
            if cat_key not in emitted_catalogs:
                left, right = _split_join_eq(match.neighbour_to_catalog or "")
                lines.append(
                    f"LEFT JOIN {match.catalog_table} AS {cat_ref} "
                    f"ON {nb_ref}.{left} = {cat_ref}.{right}"
                )
                emitted_catalogs.add(cat_key)

    return "\n".join(lines)


def build_nss_type_smoke_adql(
    dr: DRPathConfig,
    *,
    nss_solution_type: str,
    top_n: int = 20,
) -> str:
    """Bounded ADQL for Archive smoke tests (one ``nss_solution_type``, async recommended).

    Joins on ``(source_id, nss_solution_type)`` so a source with multiple NSS rows does
    not leak across solution types.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if not nss_solution_type:
        raise ValueError("nss_solution_type must be non-empty")
    if "'" in nss_solution_type:
        raise ValueError("nss_solution_type must not contain single quotes")

    full = build_nss_adql(dr)
    lines = full.split("\n")
    from_idx = next(i for i, line in enumerate(lines) if line.startswith("FROM "))
    select_clause = "\n".join(lines[:from_idx])
    tail = lines[from_idx + 1 :]
    if not tail or not tail[0].startswith("JOIN "):
        raise ValueError("unexpected ADQL shape from build_nss_adql")
    # Drop the implicit ``FROM nss`` — replaced by pick + keyed join below.
    join_tail = "\n".join(tail)

    pick_from = "\n".join(
        [
            "FROM (",
            f"  SELECT TOP {top_n} source_id, nss_solution_type",
            f"  FROM {dr.nss_table}",
            f"  WHERE nss_solution_type = '{nss_solution_type}'",
            "  ORDER BY source_id",
            ") AS pick",
            f"JOIN {dr.nss_table} AS nss",
            "  ON pick.source_id = nss.source_id",
            " AND pick.nss_solution_type = nss.nss_solution_type",
        ]
    )
    return (
        f"{select_clause}\n"
        f"{pick_from}\n"
        f"{join_tail}\n"
        "ORDER BY nss.source_id"
    )


def _table_checksum(table: Table) -> str:
    from io import StringIO

    buffer = StringIO()
    table.write(buffer, format="ascii.ecsv")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def _path_for_snapshot_meta(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def save_gaia_snapshot(
    table: Table,
    adql: str,
    *,
    snapshots_dir: Path,
    query_date: datetime | None = None,
) -> SnapshotMeta:
    """Persist raw query result + checksum + literal ADQL + query date."""
    when = query_date or datetime.now(tz=timezone.utc)
    checksum = _table_checksum(table)
    snapshot_id = f"{when.strftime('%Y%m%dT%H%M%SZ')}_{checksum[:12]}"
    out_dir = snapshots_dir / snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "query.ecsv"
    meta_path = out_dir / "meta.yaml"
    table.write(result_path, format="ascii.ecsv", overwrite=True)
    meta = SnapshotMeta(
        snapshot_id=snapshot_id,
        query_date=when,
        adql=adql,
        checksum=checksum,
        row_count=len(table),
        result_path=result_path,
        meta_path=meta_path,
    )
    meta_path.write_text(
        yaml.safe_dump(
            {
                "snapshot_id": meta.snapshot_id,
                "query_date": meta.query_date.isoformat(),
                "adql": meta.adql,
                "checksum": meta.checksum,
                "row_count": meta.row_count,
                "result_path": _path_for_snapshot_meta(meta.result_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return meta


def load_gaia_snapshot(
    meta_path: Path,
    *,
    verify_checksum: bool = True,
) -> tuple[SnapshotMeta, Table]:
    """Load a previously saved snapshot from its ``meta.yaml``.

    Set ``verify_checksum=False`` when resuming a large archive download so the
    table is not re-serialized solely to recompute the SHA-256 (slow for ~10^5 rows).
    """
    raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    result_path = Path(raw["result_path"])
    if not result_path.is_absolute():
        result_path = repo_root() / result_path
    table = Table.read(result_path, format="ascii.ecsv")
    meta = SnapshotMeta(
        snapshot_id=str(raw["snapshot_id"]),
        query_date=datetime.fromisoformat(str(raw["query_date"])),
        adql=str(raw["adql"]),
        checksum=str(raw["checksum"]),
        row_count=int(raw["row_count"]),
        result_path=result_path,
        meta_path=meta_path,
    )
    if verify_checksum and meta.checksum != _table_checksum(table):
        raise ValueError(f"snapshot checksum mismatch for {meta_path}")
    if meta.row_count != len(table):
        raise ValueError(
            f"snapshot row_count mismatch for {meta_path}: "
            f"meta={meta.row_count} table={len(table)}"
        )
    return meta, table


def _row_as_mapping(row: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    colnames = getattr(row, "colnames", None)
    if colnames is not None:
        return {name: row[name] for name in colnames}
    raise TypeError(f"unsupported row type: {type(row)!r}")


def _row_value(row: Mapping[str, Any] | Any, key: str) -> Any:
    mapping = _row_as_mapping(row)
    value = mapping[key]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _optional_float(row: Mapping[str, Any] | Any, key: str) -> float | None:
    mapping = _row_as_mapping(row)
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    # Astropy / numpy masked nulls (common in Gaia VOTable → Table).
    if hasattr(value, "mask"):
        try:
            if bool(np.ma.is_masked(value)):
                return None
        except (TypeError, ValueError):
            pass
    try:
        if np.ma.is_masked(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _build_thiele_innes(row: Mapping[str, Any]) -> ThieleInnesElements | None:
    values = {name: _optional_float(row, name) for name in _THIELE_INNES_FIELDS}
    if all(v is None for v in values.values()):
        return None
    return ThieleInnesElements(
        A=values["A"],
        B=values["B"],
        F=values["F"],
        G=values["G"],
        A_err=_optional_float(row, f"A{_THIELE_INNES_ERR_SUFFIX}"),
        B_err=_optional_float(row, f"B{_THIELE_INNES_ERR_SUFFIX}"),
        F_err=_optional_float(row, f"F{_THIELE_INNES_ERR_SUFFIX}"),
        G_err=_optional_float(row, f"G{_THIELE_INNES_ERR_SUFFIX}"),
    )


def _resolve_external_mag_err(
    raw_err: float | None,
    dr: DRPathConfig,
) -> tuple[float | None, bool]:
    """Return (mag_err, imputed). Zero/negative errors are treated as missing when configured."""
    err = raw_err
    if err is not None and dr.external_mag_err_zero_as_missing and err <= 0.0:
        err = None
    if err is not None and err > 0.0:
        return err, False
    if dr.impute_external_mag_err:
        return dr.external_mag_err_floor, True
    return None, False


def _is_gaia_pseudo_circular(row: Mapping[str, Any], dr: DRPathConfig) -> bool:
    """True for Gaia Orbital* rows with null σ_e and e below the pseudo-circular threshold."""
    solution_type = row.get("nss_solution_type")
    if hasattr(solution_type, "item"):
        solution_type = str(solution_type.item())
    elif solution_type is not None:
        solution_type = str(solution_type)
    else:
        return False
    if not solution_type.startswith("Orbital"):
        return False
    eccentricity = _optional_float(row, "eccentricity")
    if eccentricity is None:
        return False
    if _optional_float(row, "eccentricity_error") is not None:
        return False
    return eccentricity < dr.pseudo_circular_eccentricity_max


def _build_photometry(
    row: Mapping[str, Any],
    dr: DRPathConfig,
) -> tuple[list[PhotometryPoint], list[str]]:
    points: list[PhotometryPoint] = []
    imputed_bands: list[str] = []
    for band in dr.gaia_source_photometry_bands:
        mag = _optional_float(row, f"{band.lower()}_mag")
        if mag is None:
            continue
        err = _optional_float(row, f"{band.lower()}_mag_err")
        points.append(PhotometryPoint(band=band, mag=mag, mag_err=err, system="Gaia"))

    for match in _enabled_crossmatches(dr.external_photometry_crossmatches):
        mag = _optional_float(row, f"{match.band}_mag")
        if mag is None:
            continue
        err_key = f"{match.band}_mag_err"
        raw_err = _optional_float(row, err_key) if err_key in row else None
        err, imputed = _resolve_external_mag_err(raw_err, dr)
        if imputed:
            imputed_bands.append(match.band)
        points.append(PhotometryPoint(band=match.band, mag=mag, mag_err=err))
    return points, imputed_bands


def _build_nss_orbital(
    row: Mapping[str, Any],
    dr: DRPathConfig,
    *,
    spectroscopic: SpectroscopicMassFunctionConfig,
    nss_solution_type: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    orbital: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key in (
        "period",
        "period_error",
        "t_periastron",
        "t_periastron_error",
        "eccentricity",
        "eccentricity_error",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "goodness_of_fit",
        "ruwe",
        "semi_amplitude_primary",
        "semi_amplitude_primary_error",
    ):
        value = _optional_float(row, key)
        if value is not None:
            orbital[key] = value
    if "semi_amplitude_primary" in orbital:
        orbital["k1_kms"] = orbital["semi_amplitude_primary"]
    if "semi_amplitude_primary_error" in orbital:
        orbital["k1_error_kms"] = orbital["semi_amplitude_primary_error"]

    sol = nss_solution_type or ""
    if sol in spectroscopic.circular_solution_types and "eccentricity" not in orbital:
        orbital["eccentricity"] = 0.0
        extras["sb1c_circular_eccentricity"] = True

    if _is_gaia_pseudo_circular(row, dr):
        extras["gaia_pseudo_circular"] = True
    return orbital, extras


def classify_sb1_reproduction_route(
    *,
    k1_significance: float | None,
    f_m_msun: float | None,
    m1_msun: float | None,
    m2_min_msun: float | None,
    spectroscopic: SpectroscopicMassFunctionConfig,
) -> SB1ReproductionRoute:
    """Score the §8.4 disjunction for reproduction/validation.

    Does not implement ``sin³i`` marginalization and does not mark a source as
    inference-eligible. The second (``M2,min``) route is undefined when ``M1``
    is missing (evolved / not main sequence) — that is "not applicable", not
    a failed cut.
    """
    k1_ok = (
        k1_significance is not None
        and math.isfinite(k1_significance)
        # Published Table 8 reports significance to 1 decimal; the paper threshold
        # ``> 10`` is recovered as ``>= k1_significance_min`` against that fixture
        # (source 6443896148956045568 is transcribed as exactly 10.0).
        and k1_significance >= spectroscopic.k1_significance_min
    )
    high_fm = (
        f_m_msun is not None
        and math.isfinite(f_m_msun)
        and f_m_msun > spectroscopic.fm_msun_min
    )
    ms_route = False
    if (
        m1_msun is not None
        and m2_min_msun is not None
        and math.isfinite(m1_msun)
        and math.isfinite(m2_min_msun)
    ):
        ms_route = (
            m2_min_msun > spectroscopic.m2_min_msun_floor and m2_min_msun > m1_msun
        )
    return SB1ReproductionRoute(
        k1_significant=k1_ok,
        high_mass_function=bool(k1_ok and high_fm),
        main_sequence_min_companion_mass=bool(k1_ok and ms_route),
        in_union=bool(k1_ok and (high_fm or ms_route)),
    )


def _attach_spectroscopic_mass_function(
    candidate: CandidateRecord,
    spectroscopic: SpectroscopicMassFunctionConfig,
) -> CandidateRecord:
    """Stamp SB1/SB1C ``f_m`` extras. Never writes ``CandidateRecord.m2``."""
    sol = candidate.nss_solution_type
    if sol not in spectroscopic.nss_solution_types:
        return candidate
    orb = candidate.nss_orbital
    period = orb.get("period")
    ecc = orb.get("eccentricity")
    k1 = orb.get("k1_kms", orb.get("semi_amplitude_primary"))
    k1_err = orb.get("k1_error_kms", orb.get("semi_amplitude_primary_error"))
    f_m = spectroscopic_mass_function(
        period if period is not None else float("nan"),
        k1 if k1 is not None else float("nan"),
        ecc if ecc is not None else float("nan"),
    )
    f_m_val = float(np.asarray(f_m, dtype=np.float64))
    significance: float | None = None
    if k1 is not None and k1_err is not None and k1_err > 0.0:
        significance = float(k1) / float(k1_err)

    extras = dict(candidate.extras)
    extras["spectroscopic_mass_function"] = {
        "f_m_msun": f_m_val if math.isfinite(f_m_val) else None,
        "k1_kms": float(k1) if k1 is not None else None,
        "k1_error_kms": float(k1_err) if k1_err is not None else None,
        "k1_significance": significance,
        "v1_role": spectroscopic.v1_role,
        "inference_eligible": False,
        "inclination_treatment": "edge_on_m2_min_only",
        "sin3i_marginalization": False,
        "feeds_population_likelihood": False,
    }
    return candidate.model_copy(update={"extras": extras})


def _build_atmosphere_extras(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy MSC/gspphot columns into ``CandidateRecord.extras`` for mass_derivation."""
    extras: dict[str, Any] = {}
    for stem in _AP_PARAM_STEMS:
        for suffix in ("", "_upper", "_lower", "_error"):
            key = f"{stem}{suffix}"
            value = _optional_float(row, key)
            if value is not None:
                extras[key] = value
    return extras


def table_row_to_candidate(
    row: Mapping[str, Any] | Any,
    dr: DRPathConfig,
    *,
    spectroscopic: SpectroscopicMassFunctionConfig | None = None,
) -> CandidateRecord:
    """Map one joined archive row to a ``CandidateRecord`` (data_acquisition-owned fields)."""
    spec_cfg = spectroscopic if spectroscopic is not None else SpectroscopicMassFunctionConfig()
    mapping = _row_as_mapping(row)
    source_id = int(_row_value(mapping, "source_id"))
    solution_type = mapping.get("nss_solution_type")
    if hasattr(solution_type, "item"):
        solution_type = str(solution_type.item())
    elif solution_type is not None:
        solution_type = str(solution_type)

    photometry, imputed_bands = _build_photometry(mapping, dr)
    nss_orbital, orbital_extras = _build_nss_orbital(
        mapping,
        dr,
        spectroscopic=spec_cfg,
        nss_solution_type=solution_type,
    )
    extras = _build_atmosphere_extras(mapping)
    extras.update(orbital_extras)
    if imputed_bands:
        extras["mag_err_imputed_bands"] = imputed_bands

    cov_result = reconstruct_nss_covariance(mapping, nss_solution_type=solution_type)
    if cov_result.ok:
        nss_solution = cov_result.parameter_set
    else:
        nss_solution = None
        assert cov_result.status is not None
        extras["nss_covariance_status"] = cov_result.status.value
        extras["nss_covariance_usable"] = False
        if cov_result.detail:
            extras["nss_covariance_detail"] = cov_result.detail
        if cov_result.bit_index is not None:
            extras["nss_bit_index"] = cov_result.bit_index

    candidate = CandidateRecord(
        source_id=source_id,
        nss_solution_type=solution_type,
        ra_deg=_optional_float(mapping, "ra"),
        dec_deg=_optional_float(mapping, "dec"),
        parallax_mas=_optional_float(mapping, "parallax"),
        thiele_innes=_build_thiele_innes(mapping),
        nss_orbital=nss_orbital,
        nss_solution=nss_solution,
        photometry=photometry,
        extras=extras,
    )
    return _attach_spectroscopic_mass_function(candidate, spec_cfg)


def _health_from_candidates(candidates: Sequence[CandidateRecord]) -> CovarianceHealth:
    health = CovarianceHealth()
    for candidate in candidates:
        if candidate.nss_solution is not None:
            health.record(
                candidate.nss_solution_type,
                CovarianceResult(parameter_set=candidate.nss_solution, status=None),
            )
            continue
        status_raw = candidate.extras.get("nss_covariance_status")
        try:
            status = (
                CovarianceFailure(status_raw)
                if status_raw is not None
                else CovarianceFailure.MISSING_CORR
            )
        except ValueError:
            status = CovarianceFailure.UNPACK_FAILED
        health.record(
            candidate.nss_solution_type,
            CovarianceResult(parameter_set=None, status=status),
        )
    return health


def table_to_candidates(
    table: Table,
    dr: DRPathConfig,
    *,
    spectroscopic: SpectroscopicMassFunctionConfig | None = None,
) -> list[CandidateRecord]:
    spec_cfg = spectroscopic if spectroscopic is not None else SpectroscopicMassFunctionConfig()
    return [
        table_row_to_candidate(row, dr, spectroscopic=spec_cfg) for row in table
    ]


def nss_solution_type_to_cascade_label(nss_solution_type: str | None) -> str:
    """Map Gaia DR3 ``nss_solution_type`` strings to gaiamock cascade labels.

    Used for ``solution_type_fractions`` in the DA artifact (SF validation gate).
    """
    if not nss_solution_type:
        return "insufficient_visibility"
    upper = nss_solution_type.upper()
    if "ORBITAL12" in upper:
        return "twelve_parameter_orbital"
    if "ORBITAL9" in upper:
        return "nine_parameter"
    if "ORBITAL7" in upper:
        return "seven_parameter"
    if "ORBITAL5" in upper:
        return "five_parameter"
    if upper.startswith("ORBITAL"):
        return "twelve_parameter_orbital"
    return "insufficient_visibility"


def _phot_g_mag(candidate: CandidateRecord) -> float | None:
    for point in candidate.photometry:
        if point.band.upper() == "G":
            return float(point.mag)
    orb = candidate.nss_orbital
    g = orb.get("phot_g_mean_mag")
    if g is not None:
        return float(g)
    return None


def _compute_solution_type_fractions(
    candidates: Sequence[CandidateRecord],
) -> dict[str, float]:
    counts = {label: 0 for label in SOLUTION_TYPE_LABELS}
    for candidate in candidates:
        label = nss_solution_type_to_cascade_label(candidate.nss_solution_type)
        counts[label] += 1
    n = float(len(candidates))
    if n == 0:
        return {label: 0.0 for label in SOLUTION_TYPE_LABELS}
    return {label: counts[label] / n for label in SOLUTION_TYPE_LABELS}


def compute_stage_diagnostics(
    candidates: Sequence[CandidateRecord],
    *,
    funnel: FunnelCounts,
    quality_cut_bin_counts: Mapping[str, int],
    covariance_health: CovarianceHealth | None = None,
) -> StageDiagnostics:
    """Build histogram inputs for RUWE / period / eccentricity, sky, and NSS panels."""
    ruwe = []
    period = []
    ecc = []
    ra = []
    dec = []
    panel_lists: dict[str, list[float]] = {name: [] for name in SIX_PANEL_NAMES}
    health = (
        covariance_health
        if covariance_health is not None
        else _health_from_candidates(candidates)
    )

    for candidate in candidates:
        ra_val = candidate.ra_deg
        dec_val = candidate.dec_deg
        if ra_val is not None and dec_val is not None:
            ra.append(ra_val)
            dec.append(dec_val)
        orb = candidate.nss_orbital
        if "ruwe" in orb and orb["ruwe"] is not None:
            ruwe.append(float(orb["ruwe"]))
        if "period" in orb and orb["period"] is not None:
            period.append(float(orb["period"]))
            panel_lists["P_orb_days"].append(float(orb["period"]))
        if "eccentricity" in orb and orb["eccentricity"] is not None:
            ecc.append(float(orb["eccentricity"]))
            panel_lists["eccentricity"].append(float(orb["eccentricity"]))

        g_mag = _phot_g_mag(candidate)
        if g_mag is not None:
            panel_lists["G_mag"].append(g_mag)

        plx = candidate.parallax_mas
        if plx is None:
            plx_val = orb.get("parallax")
            if plx_val is not None:
                plx = float(plx_val)
        if plx is not None and plx > 0.0:
            panel_lists["inv_parallax_mas_inv"].append(1.0 / plx)

        nss_type = candidate.nss_solution_type or ""
        ti = candidate.thiele_innes
        if nss_type.startswith("Orbital") and ti is not None:
            if (
                ti.A is not None
                and ti.B is not None
                and ti.F is not None
                and ti.G is not None
            ):
                a0, _omega, inc = thiele_innes_to_campbell(ti.A, ti.B, ti.F, ti.G)
                a0_f = float(a0)
                inc_f = float(inc)
                if np.isfinite(a0_f) and np.isfinite(inc_f):
                    panel_lists["cos_inclination"].append(float(np.cos(inc_f)))
                    period_day = orb.get("period")
                    plx_use = plx if plx is not None else candidate.parallax_mas
                    if period_day is not None and plx_use is not None and plx_use > 0:
                        fm = float(
                            astrometric_mass_function(a0_f, plx_use, float(period_day))
                        )
                        if np.isfinite(fm):
                            panel_lists["f_m_msun"].append(fm)

    def _arr(values: list[float]) -> NDArray[np.floating] | None:
        return np.asarray(values, dtype=np.float64) if values else None

    nss_panels = {
        name: np.asarray(panel_lists[name], dtype=np.float64)
        for name in SIX_PANEL_NAMES
        if panel_lists[name]
    }

    return StageDiagnostics(
        funnel=funnel,
        quality_cut_bin_counts=dict(quality_cut_bin_counts),
        ruwe=_arr(ruwe),
        period_day=_arr(period),
        eccentricity=_arr(ecc),
        ra_deg=_arr(ra),
        dec_deg=_arr(dec),
        nss_panels=nss_panels,
        solution_type_fractions=_compute_solution_type_fractions(candidates),
        covariance_health=health,
    )


def format_funnel_table(
    funnel: FunnelCounts,
    quality_cut_bin_counts: Mapping[str, int],
    *,
    covariance_health: CovarianceHealth | None = None,
) -> str:
    """Human-readable funnel table (exempt from caveman compression)."""
    text = format_funnel_report(
        funnel.as_dict(),
        quality_cut_bin_counts=quality_cut_bin_counts,
        stage_name="data_acquisition",
    )
    if covariance_health is None:
        return text
    end_marker = "=== end data_acquisition funnel ==="
    type_lines = covariance_health.by_type_lines()
    health_block = "  covariance_health (by solution type):\n" + (
        "\n".join(f"    {line}" for line in type_lines) if type_lines else "    (none)"
    )
    if text.endswith(end_marker):
        body = text[: -len(end_marker)].rstrip()
        return f"{body}\n{health_block}\n{end_marker}"
    return f"{text}\n{health_block}"


def write_diagnostic_artifacts(
    diagnostics: StageDiagnostics,
    artifact_path: Path,
    *,
    config: PipelineConfig | None = None,
) -> list[Path]:
    """Write funnel/sky diagnostics beside the stage HDF5 via shared plotting hooks."""
    cfg = config if config is not None else load_config()
    dirs = resolve_diagnostic_dirs(
        cfg,
        run_id=artifact_path.parent.parent.name,
        beside_artifact=artifact_path,
    )
    emission = emit_funnel_sky(
        cfg,
        dirs,
        funnel_counts=diagnostics.funnel.as_dict(),
        quality_cut_bin_counts=diagnostics.quality_cut_bin_counts,
        ruwe=diagnostics.ruwe,
        period_day=diagnostics.period_day,
        eccentricity=diagnostics.eccentricity,
        ra_deg=diagnostics.ra_deg,
        dec_deg=diagnostics.dec_deg,
        stage_name="data_acquisition",
    )
    # Preserve legacy funnel.txt at the diagnostics root for existing callers.
    written: list[Path] = list(emission.figures) + list(emission.reports)
    funnel_text = format_funnel_table(
        diagnostics.funnel,
        diagnostics.quality_cut_bin_counts,
        covariance_health=diagnostics.covariance_health,
    )
    legacy = dirs.root / "funnel.txt"
    legacy.write_text(funnel_text + "\n", encoding="utf-8")
    if legacy not in written:
        written.append(legacy)
    report_funnel = dirs.reports / "data_acquisition_funnel.txt"
    report_funnel.parent.mkdir(parents=True, exist_ok=True)
    report_funnel.write_text(funnel_text + "\n", encoding="utf-8")
    if report_funnel not in written:
        written.append(report_funnel)
    return written


def write_stage_hdf5(
    path: Path,
    candidates: Sequence[CandidateRecord],
    *,
    snapshot: SnapshotMeta,
    diagnostics: StageDiagnostics,
    spectroscopic: SpectroscopicMassFunctionConfig | None = None,
) -> None:
    """Write one stage HDF5 under ``paths.artifact_root``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records_json = [
        json.dumps(candidate.model_dump(mode="json"), sort_keys=True)
        for candidate in candidates
    ]
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["stage"] = "data_acquisition"
        meta.attrs["query_date"] = snapshot.query_date.isoformat()
        meta.attrs["snapshot_id"] = snapshot.snapshot_id
        meta.attrs["snapshot_checksum"] = snapshot.checksum
        meta.attrs["adql"] = snapshot.adql
        meta.attrs["n_candidates"] = len(candidates)

        funnel = handle.create_group("diagnostics")
        for key, value in diagnostics.funnel.as_dict().items():
            funnel.attrs[key] = value
        funnel.create_dataset(
            "quality_cut_bin_counts",
            data=np.array(
                [f"{k}={v}" for k, v in diagnostics.quality_cut_bin_counts.items()],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        for name in ("ruwe", "period_day", "eccentricity", "ra_deg", "dec_deg"):
            values = getattr(diagnostics, name)
            if values is not None and len(values) > 0:
                funnel.create_dataset(name, data=values)

        cand = handle.create_group("candidates")
        cand.create_dataset(
            "records_json",
            data=np.array(records_json, dtype=h5py.string_dtype("utf-8")),
        )
        cand.create_dataset(
            "source_ids",
            data=np.array([c.source_id for c in candidates], dtype=np.int64),
        )

        da_grp = handle.create_group("data_acquisition")
        panels_grp = da_grp.create_group("nss_panels")
        for name in SIX_PANEL_NAMES:
            values = diagnostics.nss_panels.get(name)
            if values is not None and len(values) > 0:
                panels_grp.create_dataset(name, data=values)
        st_grp = da_grp.create_group("solution_type_fractions")
        for label in SOLUTION_TYPE_LABELS:
            st_grp.create_dataset(
                label,
                data=np.float64(diagnostics.solution_type_fractions.get(label, 0.0)),
            )

        cov_grp = da_grp.create_group("nss_covariance")
        health = diagnostics.covariance_health or CovarianceHealth()
        for key, value in health.as_dict().items():
            cov_grp.attrs[key] = int(value)
        type_lines = health.by_type_lines()
        cov_grp.create_dataset(
            "by_solution_type",
            data=np.array(type_lines, dtype=h5py.string_dtype("utf-8")),
        )
        usable = [c for c in candidates if c.nss_solution is not None]
        cov_grp.create_dataset(
            "source_ids",
            data=np.array([c.source_id for c in usable], dtype=np.int64),
        )
        matrices = cov_grp.create_group("matrices")
        for candidate in usable:
            assert candidate.nss_solution is not None
            ds = matrices.create_dataset(
                str(candidate.source_id),
                data=candidate.nss_solution.covariance_array(),
            )
            ds.attrs["names"] = np.array(
                candidate.nss_solution.names, dtype=h5py.string_dtype("utf-8")
            )
            ds.attrs["provenance"] = candidate.nss_solution.provenance

        spec_cfg = spectroscopic if spectroscopic is not None else SpectroscopicMassFunctionConfig()
        sb1 = [
            c
            for c in candidates
            if c.nss_solution_type in spec_cfg.nss_solution_types
        ]
        spec_grp = da_grp.create_group("spectroscopic_mass_function")
        spec_grp.attrs["v1_role"] = spec_cfg.v1_role
        spec_grp.attrs["inference_eligible"] = False
        spec_grp.attrs["feeds_population_likelihood"] = False
        spec_grp.attrs["sin3i_marginalization"] = False
        spec_grp.attrs["inclination_treatment"] = "edge_on_m2_min_only"
        spec_grp.attrs["scope"] = "CONTINUATION_PLAN.md §8.4.1 reproduction_and_validation_only"
        spec_grp.attrs["n_sb1"] = len(sb1)
        spec_grp.create_dataset(
            "source_ids",
            data=np.array([c.source_id for c in sb1], dtype=np.int64),
        )
        f_m_vals = []
        k1_vals = []
        sig_vals = []
        for candidate in sb1:
            block = candidate.extras.get("spectroscopic_mass_function") or {}
            f_raw = block.get("f_m_msun")
            k_raw = block.get("k1_kms")
            s_raw = block.get("k1_significance")
            f_m_vals.append(float(f_raw) if f_raw is not None else float("nan"))
            k1_vals.append(float(k_raw) if k_raw is not None else float("nan"))
            sig_vals.append(float(s_raw) if s_raw is not None else float("nan"))
        spec_grp.create_dataset("f_m_msun", data=np.asarray(f_m_vals, dtype=np.float64))
        spec_grp.create_dataset("k1_kms", data=np.asarray(k1_vals, dtype=np.float64))
        spec_grp.create_dataset(
            "k1_significance", data=np.asarray(sig_vals, dtype=np.float64)
        )

        meta.attrs["spectroscopic_mass_function_v1_role"] = spec_cfg.v1_role
        meta.attrs["spectroscopic_mass_function_inference_eligible"] = False


def read_stage_hdf5(path: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """Load candidates and meta from a ``data_acquisition`` stage HDF5.

    Re-asserts symmetry and positive-semidefiniteness on every loaded
    ``nss_solution``. Failures clear the field and stamp
    ``extras['nss_covariance_status']`` — never a diagonal-only substitute.
    """
    candidates: list[CandidateRecord] = []
    meta: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        meta_group = handle["meta"]
        for key in meta_group.attrs:
            meta[key] = meta_group.attrs[key]
        for raw in handle["candidates"]["records_json"].asstr():
            candidate = CandidateRecord.model_validate(json.loads(raw))
            if candidate.nss_solution is not None:
                failure = validate_loaded_nss_solution(candidate.nss_solution)
                if failure is not None:
                    extras = dict(candidate.extras)
                    extras["nss_covariance_status"] = failure.value
                    extras["nss_covariance_usable"] = False
                    extras["nss_covariance_detail"] = f"failed {failure.value} on load"
                    candidate = candidate.model_copy(
                        update={"nss_solution": None, "extras": extras}
                    )
            candidates.append(candidate)
    return candidates, meta


def default_gaia_query(adql: str, dr: DRPathConfig) -> Table:
    """Query the Gaia archive via astroquery (network required)."""
    from astroquery.gaia import Gaia

    user = os.environ.get(dr.gaia_archive_user_env)
    password = os.environ.get(dr.gaia_archive_password_env)
    if user and password:
        Gaia.login(user=user, password=password)
    # astroquery injects TOP when ROW_LIMIT > 0; -1 disables the cap.
    Gaia.ROW_LIMIT = int(dr.gaia_archive_row_limit)
    if dr.gaia_archive_async:
        # Sync jobs abort with ESA Error 408 on the full NSS+crossmatch ADQL.
        job = Gaia.launch_job_async(adql, dump_to_file=False)
    else:
        job = Gaia.launch_job(adql, dump_to_file=False)
    result = job.get_results()
    if not isinstance(result, Table):
        raise TypeError(f"expected astropy Table from Gaia job, got {type(result)!r}")
    return result


def run_data_acquisition(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    query_fn: Callable[[str, DRPathConfig], Table] | None = None,
    snapshot_meta_path: Path | None = None,
) -> RunManifest:
    """Execute ``data_acquisition``: query, snapshot, quality cut, HDF5, manifest update.

    Pass ``snapshot_meta_path`` to reuse an existing Gaia snapshot (skip archive query
    and avoid writing a duplicate snapshot). Useful after a mid-stage failure.
    """
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["data_acquisition"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    dr = config.active_dr()
    if snapshot_meta_path is not None:
        snapshot, raw_table = load_gaia_snapshot(
            snapshot_meta_path,
            verify_checksum=False,
        )
    else:
        adql = build_nss_adql(dr)
        query = query_fn or default_gaia_query
        raw_table = query(adql, dr)
        snapshot = save_gaia_snapshot(
            raw_table,
            adql,
            snapshots_dir=gaia_snapshots_dir(config),
        )

    filtered, bin_counts = apply_quality_cuts(raw_table, dr.quality_cut_bins)
    candidates = table_to_candidates(
        filtered, dr, spectroscopic=config.spectroscopic_mass_function
    )
    candidates, _rv_stats = attach_rv_summaries(candidates, config)
    cov_health = _health_from_candidates(candidates)
    funnel = FunnelCounts(
        queried=len(raw_table),
        after_quality_cut=len(filtered),
        candidates_written=len(candidates),
        covariance_ok=cov_health.ok,
        covariance_failed=cov_health.failed,
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=funnel,
        quality_cut_bin_counts=bin_counts,
        covariance_health=cov_health,
    )
    write_stage_hdf5(
        artifact,
        candidates,
        snapshot=snapshot,
        diagnostics=diagnostics,
        spectroscopic=config.spectroscopic_mass_function,
    )
    write_diagnostic_artifacts(diagnostics, artifact, config=config)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
