"""Stage: ``data_acquisition``.

Gaia NSS query, cross-matched photometry, N-bin goodness-of-fit quality cut, query
snapshotting, and stage HDF5 output (ARCHITECTURE.md §4).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from numpy.typing import NDArray

from darkhunter_pop.config_loader import repo_root, require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    DRPathConfig,
    ExternalPhotometryCrossmatch,
    PipelineConfig,
    QualityCutBin,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    PhotometryPoint,
    RunManifest,
    StageStatus,
    ThieleInnesElements,
)

_GAIA_SOURCE_MAG_FIELDS: dict[str, tuple[str, str | None]] = {
    "G": ("phot_g_mean_mag", "phot_g_mean_mag_error"),
    "BP": ("phot_bp_mean_mag", "phot_bp_mean_mag_error"),
    "RP": ("phot_rp_mean_mag", "phot_rp_mean_mag_error"),
}

_THIELE_INNES_FIELDS: tuple[str, ...] = ("A", "B", "F", "G")
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

    def as_dict(self) -> dict[str, int]:
        return {
            "queried": self.queried,
            "after_quality_cut": self.after_quality_cut,
            "candidates_written": self.candidates_written,
        }


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


def apply_quality_cuts(
    table: Table,
    bins: Sequence[QualityCutBin],
    *,
    g_column: str = "phot_g_mean_mag",
    gof_column: str = "goodness_of_fit",
) -> tuple[Table, dict[str, int]]:
    """Filter ``table`` with the N-bin quality-cut scheme; return per-bin survivor counts."""
    if g_column not in table.colnames or gof_column not in table.colnames:
        raise KeyError(f"table missing {g_column!r} or {gof_column!r}")

    keep = np.zeros(len(table), dtype=bool)
    bin_counts = {f"{_bin_label(i, b)}_pass": 0 for i, b in enumerate(bins)}
    bin_counts["unclassified"] = 0
    bin_counts["failed_gof"] = 0

    g_values = np.asarray(table[g_column], dtype=np.float64)
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


def build_nss_adql(dr: DRPathConfig) -> str:
    """Build the literal ADQL for the NSS catalog + photometry cross-matches."""
    select_parts = [
        "nss.source_id",
        "nss.nss_solution_type",
        "nss.ra",
        "nss.dec",
        "nss.parallax",
        "nss.parallax_error",
        "nss.pmra",
        "nss.pmdec",
        "nss.period",
        "nss.period_error",
        "nss.eccentricity",
        "nss.eccentricity_error",
        "nss.goodness_of_fit",
        "gs.ruwe",
    ]
    for thiele in _THIELE_INNES_FIELDS:
        select_parts.append(f"nss.{thiele}")
        select_parts.append(f"nss.{thiele}{_THIELE_INNES_ERR_SUFFIX}")

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
    neighbour_alias: dict[str, str] = {}
    for match in enabled:
        if match.neighbour_table not in neighbour_alias:
            neighbour_alias[match.neighbour_table] = f"nb_{len(neighbour_alias)}"

    catalog_alias: dict[tuple[str, str], str] = {}
    for match in enabled:
        key = (match.neighbour_table, match.catalog_table)
        if key not in catalog_alias:
            catalog_alias[key] = f"cat_{len(catalog_alias)}"
        cat_ref = catalog_alias[key]
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

    for (neighbour_table, catalog_table), cat_ref in catalog_alias.items():
        nb_ref = neighbour_alias[neighbour_table]
        lines.append(
            f"LEFT JOIN {catalog_table} AS {cat_ref} "
            f"ON {nb_ref}.original_ext_source_id = {cat_ref}.original_ext_source_id"
        )

    return "\n".join(lines)


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


def load_gaia_snapshot(meta_path: Path) -> tuple[SnapshotMeta, Table]:
    """Load a previously saved snapshot from its ``meta.yaml``."""
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
    if meta.checksum != _table_checksum(table):
        raise ValueError(f"snapshot checksum mismatch for {meta_path}")
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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


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


def _build_photometry(
    row: Mapping[str, Any],
    dr: DRPathConfig,
) -> list[PhotometryPoint]:
    points: list[PhotometryPoint] = []
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
        err = _optional_float(row, err_key) if err_key in row else None
        points.append(PhotometryPoint(band=match.band, mag=mag, mag_err=err))
    return points


def _build_nss_orbital(row: Mapping[str, Any]) -> dict[str, Any]:
    orbital: dict[str, Any] = {}
    for key in (
        "period",
        "period_error",
        "eccentricity",
        "eccentricity_error",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "goodness_of_fit",
        "ruwe",
    ):
        value = _optional_float(row, key)
        if value is not None:
            orbital[key] = value
    return orbital


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


def table_row_to_candidate(row: Mapping[str, Any] | Any, dr: DRPathConfig) -> CandidateRecord:
    """Map one joined archive row to a ``CandidateRecord`` (data_acquisition-owned fields)."""
    mapping = _row_as_mapping(row)
    source_id = int(_row_value(mapping, "source_id"))
    solution_type = mapping.get("nss_solution_type")
    if hasattr(solution_type, "item"):
        solution_type = str(solution_type.item())
    elif solution_type is not None:
        solution_type = str(solution_type)

    return CandidateRecord(
        source_id=source_id,
        nss_solution_type=solution_type,
        ra_deg=_optional_float(mapping, "ra"),
        dec_deg=_optional_float(mapping, "dec"),
        parallax_mas=_optional_float(mapping, "parallax"),
        thiele_innes=_build_thiele_innes(mapping),
        nss_orbital=_build_nss_orbital(mapping),
        photometry=_build_photometry(mapping, dr),
        extras=_build_atmosphere_extras(mapping),
    )


def table_to_candidates(table: Table, dr: DRPathConfig) -> list[CandidateRecord]:
    return [table_row_to_candidate(row, dr) for row in table]


def compute_stage_diagnostics(
    candidates: Sequence[CandidateRecord],
    *,
    funnel: FunnelCounts,
    quality_cut_bin_counts: Mapping[str, int],
) -> StageDiagnostics:
    """Build histogram inputs for RUWE / period / eccentricity and sky coverage."""
    ruwe = []
    period = []
    ecc = []
    ra = []
    dec = []
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
        if "eccentricity" in orb and orb["eccentricity"] is not None:
            ecc.append(float(orb["eccentricity"]))

    def _arr(values: list[float]) -> NDArray[np.floating] | None:
        return np.asarray(values, dtype=np.float64) if values else None

    return StageDiagnostics(
        funnel=funnel,
        quality_cut_bin_counts=dict(quality_cut_bin_counts),
        ruwe=_arr(ruwe),
        period_day=_arr(period),
        eccentricity=_arr(ecc),
        ra_deg=_arr(ra),
        dec_deg=_arr(dec),
    )


def format_funnel_table(
    funnel: FunnelCounts,
    quality_cut_bin_counts: Mapping[str, int],
) -> str:
    """Human-readable funnel table (exempt from caveman compression)."""
    lines = [
        "data_acquisition funnel",
        f"  queried:              {funnel.queried}",
        f"  after_quality_cut:    {funnel.after_quality_cut}",
        f"  candidates_written:   {funnel.candidates_written}",
        "  quality_cut_bins:",
    ]
    for key, count in sorted(quality_cut_bin_counts.items()):
        lines.append(f"    {key}: {count}")
    return "\n".join(lines)


def write_diagnostic_artifacts(
    diagnostics: StageDiagnostics,
    artifact_path: Path,
) -> list[Path]:
    """Write optional matplotlib diagnostics beside the stage HDF5."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir = artifact_path.parent / f"{artifact_path.stem}_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _hist(
        values: NDArray[np.floating] | None,
        name: str,
        xlabel: str,
    ) -> None:
        if values is None or len(values) == 0:
            return
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins="auto", color="steelblue", edgecolor="white")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("count")
        axis.set_title(name)
        path = out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    _hist(diagnostics.ruwe, "ruwe", "RUWE")
    _hist(diagnostics.period_day, "period_day", "period [day]")
    _hist(diagnostics.eccentricity, "eccentricity", "eccentricity")

    if diagnostics.ra_deg is not None and diagnostics.dec_deg is not None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="mollweide")
        coord = SkyCoord(
            diagnostics.ra_deg * u.deg,
            diagnostics.dec_deg * u.deg,
            frame="icrs",
        )
        ax.scatter(
            coord.ra.wrap_at(180 * u.deg).radian,
            coord.dec.radian,
            s=4,
            alpha=0.6,
        )
        ax.set_title("sky coverage")
        path = out_dir / "sky_map.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    funnel_path = out_dir / "funnel.txt"
    funnel_path.write_text(
        format_funnel_table(diagnostics.funnel, diagnostics.quality_cut_bin_counts),
        encoding="utf-8",
    )
    written.append(funnel_path)
    return written


def write_stage_hdf5(
    path: Path,
    candidates: Sequence[CandidateRecord],
    *,
    snapshot: SnapshotMeta,
    diagnostics: StageDiagnostics,
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


def read_stage_hdf5(path: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """Load candidates and meta from a ``data_acquisition`` stage HDF5."""
    candidates: list[CandidateRecord] = []
    meta: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        meta_group = handle["meta"]
        for key in meta_group.attrs:
            meta[key] = meta_group.attrs[key]
        for raw in handle["candidates"]["records_json"].asstr():
            candidates.append(CandidateRecord.model_validate(json.loads(raw)))
    return candidates, meta


def default_gaia_query(adql: str, dr: DRPathConfig) -> Table:
    """Query the Gaia archive via astroquery (network required)."""
    from astroquery.gaia import Gaia

    user = os.environ.get(dr.gaia_archive_user_env)
    password = os.environ.get(dr.gaia_archive_password_env)
    if user and password:
        Gaia.login(user=user, password=password)
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
) -> RunManifest:
    """Execute ``data_acquisition``: query, snapshot, quality cut, HDF5, manifest update."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY["data_acquisition"]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    dr = config.active_dr()
    adql = build_nss_adql(dr)
    query = query_fn or default_gaia_query
    raw_table = query(adql, dr)
    snapshot = save_gaia_snapshot(
        raw_table,
        adql,
        snapshots_dir=gaia_snapshots_dir(config),
    )

    filtered, bin_counts = apply_quality_cuts(raw_table, dr.quality_cut_bins)
    candidates = table_to_candidates(filtered, dr)
    funnel = FunnelCounts(
        queried=len(raw_table),
        after_quality_cut=len(filtered),
        candidates_written=len(candidates),
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=funnel,
        quality_cut_bin_counts=bin_counts,
    )
    write_stage_hdf5(artifact, candidates, snapshot=snapshot, diagnostics=diagnostics)
    write_diagnostic_artifacts(diagnostics, artifact)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
