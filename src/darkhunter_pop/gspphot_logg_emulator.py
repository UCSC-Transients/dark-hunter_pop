"""GSP-Phot log g emulator for mock rows: Andrews' giant reject on the mock (#680, RUNG5_SPEC §2.7.4).

Andrews' ``atf_giant_reject_logg`` reads the VizieR I/355/paramp GSP-Phot log g (``logg_vizier_apsis``, the archive
``logg_gspphot``; the calibration refuses to run unless the two agree). A mock luminous star has a truth
``atmosphere_logg`` but no GSP-Phot measurement, so without this module the cut sees an undefined log g on every mock
row and passes them all, while it removes giants from the real data.

The emulator is statistical, never per-star (Ryan, 2026-11-13). It is conditioned on the observed CMD position
``x = (G, BP-RP, M_G)``, computed identically on real DR3 and mock rows, and reproduces both failure modes of the real
column:

(a) missingness: ``P(defined | x)`` = the defined fraction of the real rows in the finest fallback cell of ``x``
    holding at least ``n_min_rows`` rows;
(b) value: ``logg_obs | x`` drawn from the histogram (``logg_bins``) of the defined real log g in the finest fallback
    cell holding at least ``n_min_defined`` defined rows, uniform within a bin. GSP-Phot biases, scatter and
    catastrophic giant / dwarf confusions are carried in that histogram; ``P(logg < t | defined)`` is exact for any
    ``t`` on the bin grid.

The mock's truth ``atmosphere_logg`` is not used: real rows have no truth log g, so a truth -> observed link cannot be
calibrated on them. Within a cell the emulated value is therefore independent of the mock star's truth evolutionary
state; the reject rate follows the mock's CMD occupancy.

All numbers (edges, minimum counts, log g bins, seed, calibration population, validation gate) live in
``config/population/gspphot_logg_emulator_680.yaml`` (definition, hand-written; its header gives the justification)
and the calibrated cells in ``config/population/gspphot_logg_emulator_680_table.yaml`` (written by
``scripts/calibrate_gspphot_logg_emulator_680.py``; it records the SHA-256 of the definition it was built from, and
:func:`load_emulator` refuses a mismatch).

Randomness is per draw and deterministic in ``(seed, generation, draw_index)`` (counter-based splitmix64), so a draw
gets the same log g whatever other rows it is evaluated with.

Limitations
-----------
* Cells are coarse where real rows are sparse: the fallback pools over G, then over finer CMD cells. The level used
  per row is returned (:class:`EmulatedLogg`).
* ``x`` is the system photometry. At fixed ``x`` GSP-Phot behaves differently in different real sub-populations
  (the Andrews rows vs the whole NSS orbit sample, docs/gate680); the emulator carries only the calibration
  population's behaviour.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Default definition / table (repo-relative).
DEFAULT_CONFIG_PATH: Final[str] = "config/population/gspphot_logg_emulator_680.yaml"
DEFAULT_TABLE_PATH: Final[str] = "config/population/gspphot_logg_emulator_680_table.yaml"

#: Conditioning axes.
AXES: Final[tuple[str, ...]] = ("g_mag", "bp_rp", "abs_g_mag")
AxisName = Literal["g_mag", "bp_rp", "abs_g_mag"]

#: Random streams of the per-draw hash.
STREAM_DEFINED: Final[int] = 0
STREAM_VALUE: Final[int] = 1

# splitmix64 (Steele, Lea & Flood 2014) algorithm constants; not physics.
_SM_GAMMA: Final[np.uint64] = np.uint64(0x9E3779B97F4A7C15)
_SM_M1: Final[np.uint64] = np.uint64(0xBF58476D1CE4E5B9)
_SM_M2: Final[np.uint64] = np.uint64(0x94D049BB133111EB)
_TWO_M53: Final[float] = 2.0 ** -53


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _strictly_increasing(v: tuple[float, ...]) -> tuple[float, ...]:
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0.0):
        raise ValueError("edges must be finite and strictly increasing")
    return v


class FallbackLevel(_Strict):
    """One fallback level: the axes it bins and their edges (``{}`` = all rows in one cell)."""

    name: str = Field(..., min_length=1)
    edges: dict[AxisName, tuple[float, ...]]

    @field_validator("edges")
    @classmethod
    def edges_ok(cls, v: dict[str, tuple[float, ...]]) -> dict[str, tuple[float, ...]]:
        for e in v.values():
            _strictly_increasing(e)
        return v

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(a for a in AXES if a in self.edges)

    def axis_edges(self, axis: str) -> NDArray[np.float64]:
        return np.asarray(self.edges[axis], dtype=np.float64)  # type: ignore[index]


class LoggBins(_Strict):
    """Regular log g histogram grid ``[min, max]`` in steps of ``width``."""

    min: float
    max: float
    width: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _grid_ok(self) -> LoggBins:
        n = (self.max - self.min) / self.width
        if self.max <= self.min or abs(n - round(n)) > 1e-9:
            raise ValueError("logg_bins: (max - min) must be a positive whole number of widths")
        return self

    def edges(self) -> NDArray[np.float64]:
        n = int(round((self.max - self.min) / self.width))
        return np.round(self.min + self.width * np.arange(n + 1), 10)


class EmulatorBinning(_Strict):
    """Fallback levels (finest first, last = all rows), minimum counts and the log g histogram grid."""

    fallback_levels: tuple[FallbackLevel, ...] = Field(..., min_length=1)
    n_min_rows: int = Field(..., ge=1)
    n_min_defined: int = Field(..., ge=1)
    logg_bins: LoggBins

    @field_validator("fallback_levels")
    @classmethod
    def _fallback_ok(cls, v: tuple[FallbackLevel, ...]) -> tuple[FallbackLevel, ...]:
        if v[-1].edges:
            raise ValueError("the last fallback level must have edges {} (all rows), so every row has a cell")
        if len({lv.name for lv in v}) != len(v):
            raise ValueError("fallback level names must be unique")
        return v


class CalibrationSpec(_Strict):
    """Real DR3 calibration set and the column-equivalence check."""

    snapshot_meta: str
    solution_types: tuple[str, ...] = Field(..., min_length=1)
    restrict_to_equivalence_ids: bool
    logg_column: str
    g_mag_column: str
    bp_mag_column: str
    rp_mag_column: str
    parallax_column: str
    equivalence_snapshot_meta: str
    equivalence_column: str
    logg_decimals: int = Field(..., ge=0)
    equivalence_tolerance: float = Field(..., gt=0.0)
    holdout_fraction: float = Field(..., gt=0.0, lt=1.0)
    split_seed: int = Field(..., ge=0)


class ValidationSpec(_Strict):
    """Holdout gate: per coarse CMD region and G bin, |pull| of the undefined and giant-reject rates."""

    selection: str
    cut_id: str
    threshold_parameter: str
    bp_rp_edges: tuple[float, ...]
    abs_g_mag_edges: tuple[float, ...]
    g_mag_edges: tuple[float, ...]
    n_min_report: int = Field(..., ge=1)
    max_abs_pull: float = Field(..., gt=0.0)

    @field_validator("bp_rp_edges", "abs_g_mag_edges", "g_mag_edges")
    @classmethod
    def edges_ok(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        return _strictly_increasing(v)


class EmulatorDefinition(_Strict):
    """``config/population/gspphot_logg_emulator_680.yaml`` (header: definitions and justification)."""

    schema_version: int = Field(..., ge=1)
    version: str = Field(..., pattern=r"^gle[0-9]+$")
    provenance: str = Field(..., min_length=1)
    output_column: str = Field(..., min_length=1)
    seed: int = Field(..., ge=0)
    binning: EmulatorBinning
    calibration: CalibrationSpec
    validation: ValidationSpec


class CellEntry(_Strict):
    """One kept cell: bin index per level axis, real-row counts and, where enough rows are defined, the log g
    histogram (``hist`` counts starting at grid bin ``hist_offset``; zeros outside are trimmed)."""

    bins: tuple[int, ...]
    n_rows: int = Field(..., ge=1)
    n_defined: int = Field(..., ge=0)
    hist_offset: int | None = None
    hist: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _counts_ok(self) -> CellEntry:
        if self.n_defined > self.n_rows:
            raise ValueError("n_defined > n_rows")
        if (self.hist is None) != (self.hist_offset is None):
            raise ValueError("hist and hist_offset go together")
        if self.hist is not None and sum(self.hist) != self.n_defined:
            raise ValueError("histogram counts must sum to n_defined")
        return self


class EmulatorTable(_Strict):
    """Calibrated cells per fallback level (``gspphot_logg_emulator_680_table.yaml``)."""

    schema_version: int = Field(..., ge=1)
    version: str
    definition_sha256: str = Field(..., min_length=64, max_length=64)
    provenance: dict[str, Any]
    levels: tuple[tuple[CellEntry, ...], ...]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.is_file():
        from darkhunter_pop.config_loader import repo_root

        p = repo_root() / p
    return p


def definition_sha256(path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """SHA-256 of the definition file bytes (the table records it)."""
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def load_definition(path: str | Path = DEFAULT_CONFIG_PATH) -> EmulatorDefinition:
    return EmulatorDefinition.model_validate(yaml.safe_load(_resolve(path).read_text()))


# ---------------------------------------------------------------------------
# Features, binning, hashing
# ---------------------------------------------------------------------------


def abs_g_mag(g_mag: ArrayLike, parallax_mas: ArrayLike) -> NDArray[np.float64]:
    """``M_G = G + 5 log10(parallax_mas) - 10``; NaN where the parallax is not finite and positive.

    The real-row recipe (``candidate_to_selection_row``) and ``mock_selection_rows`` use the same expression.
    """
    g = np.asarray(g_mag, dtype=np.float64)
    plx = np.asarray(parallax_mas, dtype=np.float64)
    ok = np.isfinite(plx) & (plx > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, g + 5.0 * np.log10(np.where(ok, plx, 1.0)) - 10.0, np.nan)


def bin_index(x: ArrayLike, edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Bin of each value: ``0 .. len(edges)`` (open outer bins), ``len(edges) + 1`` for a non-finite value."""
    a = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(edges, np.where(np.isfinite(a), a, 0.0), side="right").astype(np.int64)
    return np.where(np.isfinite(a), idx, edges.size + 1)


def _level_codes(
    level: FallbackLevel, feats: Mapping[str, NDArray[np.float64]]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """``(code, bins)`` of each row at ``level``: flat cell code and the ``(n, n_axes)`` bin indices."""
    n = next(iter(feats.values())).shape[0]
    axes = level.axes
    if not axes:
        return np.zeros(n, dtype=np.int64), np.zeros((n, 0), dtype=np.int64)
    idx = [bin_index(feats[a], level.axis_edges(a)) for a in axes]
    dims = tuple(level.axis_edges(a).size + 2 for a in axes)
    return np.ravel_multi_index(tuple(idx), dims).astype(np.int64), np.stack(idx, axis=1)


def _splitmix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        z = x + _SM_GAMMA
        z = (z ^ (z >> np.uint64(30))) * _SM_M1
        z = (z ^ (z >> np.uint64(27))) * _SM_M2
        return z ^ (z >> np.uint64(31))


def draw_uniforms(seed: int, generation: ArrayLike, draw_index: ArrayLike, stream: int) -> NDArray[np.float64]:
    """Per-draw uniforms in [0, 1), deterministic in ``(seed, generation, draw_index, stream)``; vectorized."""
    gen = np.asarray(generation, dtype=np.int64).astype(np.uint64)
    di = np.asarray(draw_index, dtype=np.int64).astype(np.uint64)
    gen, di = np.broadcast_arrays(gen, di)
    h = _splitmix64(np.full(di.shape, np.uint64(seed), dtype=np.uint64))
    h = _splitmix64(h ^ gen)
    h = _splitmix64(h ^ di)
    h = _splitmix64(h ^ np.uint64(stream))
    return (h >> np.uint64(11)).astype(np.float64) * _TWO_M53


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_levels(
    feats: Mapping[str, ArrayLike], logg: ArrayLike, binning: EmulatorBinning
) -> tuple[tuple[CellEntry, ...], ...]:
    """Cells of every fallback level from real rows (``feats``: ``g_mag``, ``bp_rp``, ``abs_g_mag``; log g NaN =
    undefined). A cell is kept when it holds ``>= n_min_rows`` rows (the all-rows level always); its histogram is
    stored when it holds ``>= n_min_defined`` defined rows (the all-rows level: whenever any row is defined).

    Raises ``ValueError`` for a defined log g outside ``logg_bins`` or a calibration set with no defined log g.
    """
    f = {a: np.asarray(feats[a], dtype=np.float64) for a in AXES}
    lg = np.asarray(logg, dtype=np.float64)
    defined = np.isfinite(lg)
    edges = binning.logg_bins.edges()
    if np.any((lg[defined] < edges[0]) | (lg[defined] > edges[-1])):
        raise ValueError(f"defined log g outside logg_bins [{edges[0]}, {edges[-1]}]: widen the grid")
    hbin = np.clip(np.searchsorted(edges, np.where(defined, lg, edges[0]), side="right") - 1, 0, edges.size - 2)
    out: list[tuple[CellEntry, ...]] = []
    for level in binning.fallback_levels:
        code, bins = _level_codes(level, f)
        uniq, first, inv, counts = np.unique(code, return_index=True, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).ravel()
        n_def = np.bincount(inv, weights=defined.astype(np.float64), minlength=uniq.size).astype(np.int64)
        hists = np.zeros((uniq.size, edges.size - 1), dtype=np.int64)
        np.add.at(hists, (inv[defined], hbin[defined]), 1)
        cells: list[CellEntry] = []
        for k in range(uniq.size):
            if level.axes and counts[k] < binning.n_min_rows:
                continue
            off = hist = None
            if n_def[k] >= binning.n_min_defined or (not level.axes and n_def[k] > 0):
                nz = np.flatnonzero(hists[k])
                off = int(nz[0])
                hist = tuple(int(c) for c in hists[k, nz[0]: nz[-1] + 1])
            cells.append(CellEntry(bins=tuple(int(b) for b in bins[first[k]]), n_rows=int(counts[k]),
                                   n_defined=int(n_def[k]), hist_offset=off, hist=hist))
        if not level.axes and (not cells or cells[0].hist is None):
            raise ValueError("calibration set has no defined log g: the all-rows level needs a histogram")
        out.append(tuple(cells))
    return tuple(out)


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LevelLookup:
    codes: NDArray[np.int64]        # sorted flat codes of kept cells
    p_defined: NDArray[np.float64]
    has_hist: NDArray[np.bool_]
    cdf: NDArray[np.float64]        # (n_cells, n_logg_edges): CDF at the grid edges (0 ... 1); NaN rows w/o histogram


@dataclass(frozen=True)
class EmulatedLogg:
    """Per-row result: emulated log g (NaN = undefined, as the real column) and the fallback level index used for
    each model part (0 = finest)."""

    logg: NDArray[np.float64]
    defined_level: NDArray[np.int64]
    value_level: NDArray[np.int64]


class GspPhotLoggEmulator:
    """Calibrated emulator (definition + table). Build with :func:`load_emulator`."""

    def __init__(self, definition: EmulatorDefinition, table: EmulatorTable) -> None:
        b = definition.binning
        if table.version != definition.version or len(table.levels) != len(b.fallback_levels):
            raise ValueError("emulator table does not match the definition (version / fallback levels)")
        self.definition = definition
        self.table = table
        self._edges = b.logg_bins.edges()
        nb = self._edges.size - 1
        lookups: list[_LevelLookup] = []
        for level, cells in zip(b.fallback_levels, table.levels):
            if level.axes:
                dims = tuple(level.axis_edges(a).size + 2 for a in level.axes)
                codes = np.asarray([np.ravel_multi_index(c.bins, dims) for c in cells], dtype=np.int64)
            else:
                if len(cells) != 1 or cells[0].hist is None:
                    raise ValueError("the all-rows level must hold exactly one cell with a histogram")
                codes = np.zeros(1, dtype=np.int64)
            cdf = np.full((len(cells), nb + 1), np.nan)
            has = np.zeros(len(cells), dtype=bool)
            for k, c in enumerate(cells):
                if c.hist is not None and c.hist_offset is not None:
                    h = np.zeros(nb)
                    if c.hist_offset + len(c.hist) > nb:
                        raise ValueError("cell histogram runs past logg_bins")
                    h[c.hist_offset: c.hist_offset + len(c.hist)] = c.hist
                    cdf[k] = np.concatenate(([0.0], np.cumsum(h) / h.sum()))
                    has[k] = True
            p_def = np.asarray([c.n_defined / c.n_rows for c in cells], dtype=np.float64)
            order = np.argsort(codes)
            lookups.append(_LevelLookup(codes[order], p_def[order], has[order], cdf[order]))
        self._lookups = tuple(lookups)

    @property
    def version(self) -> str:
        return self.definition.version

    @property
    def output_column(self) -> str:
        return self.definition.output_column

    def _cells(
        self, g_mag: ArrayLike, bp_rp: ArrayLike, abs_g: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """``(p_defined, cdf rows, defined_level, value_level)`` per row (finest populated level each)."""
        feats = {
            "g_mag": np.atleast_1d(np.asarray(g_mag, dtype=np.float64)),
            "bp_rp": np.atleast_1d(np.asarray(bp_rp, dtype=np.float64)),
            "abs_g_mag": np.atleast_1d(np.asarray(abs_g, dtype=np.float64)),
        }
        n = feats["g_mag"].shape[0]
        if any(v.shape != (n,) for v in feats.values()):
            raise ValueError("g_mag, bp_rp and abs_g must be 1-D of equal length")
        levels = self.definition.binning.fallback_levels
        p_def = np.full(n, np.nan)
        cdf = np.full((n, self._edges.size), np.nan)
        d_lvl = np.full(n, -1, dtype=np.int64)
        v_lvl = np.full(n, -1, dtype=np.int64)
        for li in range(len(self._lookups) - 1, -1, -1):  # coarse -> fine: finer levels overwrite
            lk = self._lookups[li]
            if lk.codes.size == 0:
                continue
            code, _ = _level_codes(levels[li], feats)
            pos = np.clip(np.searchsorted(lk.codes, code), 0, lk.codes.size - 1)
            found = lk.codes[pos] == code
            p_def = np.where(found, lk.p_defined[pos], p_def)
            d_lvl = np.where(found, li, d_lvl)
            vh = found & lk.has_hist[pos]
            cdf[vh] = lk.cdf[pos[vh]]
            v_lvl = np.where(vh, li, v_lvl)
        return p_def, cdf, d_lvl, v_lvl

    def _inverse_cdf(self, cdf: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per row: the value at CDF level ``u`` (uniform within the histogram bin), vectorized over rows."""
        e = self._edges
        k = np.clip((cdf <= u[:, None]).sum(axis=1) - 1, 0, e.size - 2)  # last edge with CDF <= u
        rows = np.arange(u.size)
        lo, hi = cdf[rows, k], cdf[rows, k + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(hi > lo, (u - lo) / (hi - lo), 0.0)
        return e[k] + np.clip(frac, 0.0, 1.0) * (e[k + 1] - e[k])

    def emulate(
        self, g_mag: ArrayLike, bp_rp: ArrayLike, abs_g: ArrayLike, *, generation: ArrayLike, draw_index: ArrayLike
    ) -> EmulatedLogg:
        """Emulated GSP-Phot log g per row (NaN = undefined), seeded per ``(generation, draw_index)``."""
        p_def, cdf, d_lvl, v_lvl = self._cells(g_mag, bp_rp, abs_g)
        n = p_def.size
        gen = np.broadcast_to(np.asarray(generation, dtype=np.int64), (n,))
        di = np.broadcast_to(np.asarray(draw_index, dtype=np.int64), (n,))
        seed = self.definition.seed
        defined = draw_uniforms(seed, gen, di, STREAM_DEFINED) < p_def
        val = self._inverse_cdf(cdf, draw_uniforms(seed, gen, di, STREAM_VALUE))
        return EmulatedLogg(np.where(defined, val, np.nan), d_lvl, v_lvl)

    def expected_rates(
        self, g_mag: ArrayLike, bp_rp: ArrayLike, abs_g: ArrayLike, *, threshold: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(P(undefined), P(defined and logg < threshold))`` per row under the emulator, exactly (no MC)."""
        p_def, cdf, _, _ = self._cells(g_mag, bp_rp, abs_g)
        e = self._edges
        t = float(np.clip(threshold, e[0], e[-1]))
        k = int(np.clip(np.searchsorted(e, t, side="right") - 1, 0, e.size - 2))
        frac = (t - e[k]) / (e[k + 1] - e[k])
        below = cdf[:, k] + frac * (cdf[:, k + 1] - cdf[:, k])
        return 1.0 - p_def, p_def * below


def build_table(
    definition: EmulatorDefinition, levels: tuple[tuple[CellEntry, ...], ...], *,
    definition_sha: str, provenance: Mapping[str, Any],
) -> EmulatorTable:
    return EmulatorTable(schema_version=1, version=definition.version, definition_sha256=definition_sha,
                         provenance=dict(provenance), levels=levels)


def table_to_yaml(table: EmulatorTable) -> str:
    """Compact YAML: one flow-style line per cell."""
    head = {k: v for k, v in table.model_dump(mode="json").items() if k != "levels"}
    lines = [
        "# GENERATED by scripts/calibrate_gspphot_logg_emulator_680.py (#680); do not edit by hand.",
        "# Definition: config/population/gspphot_logg_emulator_680.yaml (SHA-256 below; a mismatch is refused).",
        "# One list per fallback level (definition order). Cell: bins = bin index per level axis (g_mag, bp_rp,",
        "# abs_g_mag order; len(edges) + 1 = undefined), real-row counts, log g histogram from grid bin hist_offset.",
        yaml.safe_dump(head, sort_keys=False, width=1000).rstrip(),
        "levels:",
    ]
    for cells in table.levels:
        if not cells:
            lines.append("  - []")
            continue
        lines.append("  -")
        for c in cells:
            h = "" if c.hist is None else f", hist_offset: {c.hist_offset}, hist: [{', '.join(str(v) for v in c.hist)}]"
            lines.append(f"    - {{bins: [{', '.join(str(b) for b in c.bins)}], n_rows: {c.n_rows}, "
                         f"n_defined: {c.n_defined}{h}}}")
    return "\n".join(lines) + "\n"


def load_emulator(
    definition_path: str | Path = DEFAULT_CONFIG_PATH, table_path: str | Path = DEFAULT_TABLE_PATH
) -> GspPhotLoggEmulator:
    """Definition + calibrated table; refuses a table built from a different definition file."""
    definition = load_definition(definition_path)
    table = EmulatorTable.model_validate(yaml.safe_load(_resolve(table_path).read_text()))
    sha = definition_sha256(definition_path)
    if table.definition_sha256 != sha:
        raise ValueError(
            f"{table_path}: built from definition sha256 {table.definition_sha256[:12]}, current {sha[:12]}; "
            "re-run scripts/calibrate_gspphot_logg_emulator_680.py"
        )
    return GspPhotLoggEmulator(definition, table)


def selection_threshold(definition: EmulatorDefinition) -> float:
    """The giant-reject log g threshold, read from the frozen selection file (never copied into the emulator)."""
    raw = yaml.safe_load(_resolve(definition.validation.selection).read_text())
    for cut in raw.get("cuts", []):
        if cut.get("id") == definition.validation.cut_id:
            return float(cut["parameters"][definition.validation.threshold_parameter])
    raise KeyError(f"{definition.validation.selection}: no cut {definition.validation.cut_id!r}")


# ---------------------------------------------------------------------------
# Holdout split and validation report
# ---------------------------------------------------------------------------


def holdout_mask(source_id: ArrayLike, *, split_seed: int, holdout_fraction: float) -> NDArray[np.bool_]:
    """Deterministic train / holdout split by a hash of ``(split_seed, source_id)``."""
    sid = np.asarray(source_id, dtype=np.int64)
    return draw_uniforms(split_seed, np.zeros_like(sid), sid, STREAM_DEFINED) < holdout_fraction


@dataclass(frozen=True)
class ValidationRow:
    label: str
    n_holdout: int
    n_train: int
    undef_real: float
    undef_emu: float
    undef_pull: float
    reject_real: float
    reject_emu: float
    reject_pull: float
    median_logg_real: float
    median_logg_emu: float

    def gated(self, n_min: int, max_abs_pull: float) -> bool | None:
        """True / False when the row is gated (``n_holdout >= n_min``), else None."""
        if self.n_holdout < n_min:
            return None
        return bool(abs(self.undef_pull) <= max_abs_pull and abs(self.reject_pull) <= max_abs_pull)


def rate_pull(real: float, emu: float, n_h: int, n_t: int) -> float:
    """``(real - emu) / sigma`` with ``sigma^2 = p(1-p)(1/n_holdout + 1/n_train)`` at the pooled rate
    ``p = (n_holdout real + n_train emu) / (n_holdout + n_train)``: holdout binomial noise plus the train-half
    calibration noise of the same region (approximate where the fallback pools train rows from outside it)."""
    n_h, n_t = max(n_h, 1), max(n_t, 1)
    p = (n_h * real + n_t * emu) / (n_h + n_t)
    var = p * (1.0 - p) * (1.0 / n_h + 1.0 / n_t)
    if var <= 0.0:
        return 0.0
    return float((real - emu) / np.sqrt(var))


def validation_rows(
    emu: GspPhotLoggEmulator,
    groups: Mapping[str, tuple[NDArray[np.bool_], int]],
    feats_h: Mapping[str, NDArray[np.float64]],
    logg_h: NDArray[np.float64],
    emulated_h: NDArray[np.float64],
    *,
    threshold: float,
) -> list[ValidationRow]:
    """One :class:`ValidationRow` per group ``label -> (holdout-row mask, n_train in the same group)``."""
    p_undef, p_rej = emu.expected_rates(feats_h["g_mag"], feats_h["bp_rp"], feats_h["abs_g_mag"], threshold=threshold)
    out: list[ValidationRow] = []
    for label, (mh, n_t) in groups.items():
        n_h = int(mh.sum())
        if n_h == 0:
            out.append(ValidationRow(label, 0, int(n_t), *([float("nan")] * 8)))
            continue
        lg = logg_h[mh]
        dr = np.isfinite(lg)
        u_real, u_emu = float(np.mean(~dr)), float(np.mean(p_undef[mh]))
        r_real = float(np.mean(dr & (np.where(dr, lg, np.inf) < threshold)))
        r_emu = float(np.mean(p_rej[mh]))
        e = emulated_h[mh]
        out.append(ValidationRow(
            label, n_h, int(n_t), u_real, u_emu, rate_pull(u_real, u_emu, n_h, n_t),
            r_real, r_emu, rate_pull(r_real, r_emu, n_h, n_t),
            float(np.median(lg[dr])) if dr.any() else float("nan"),
            float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else float("nan"),
        ))
    return out


def _bin_names(edges: Sequence[float], prefix: str) -> list[str]:
    e = list(edges)
    return ([f"{prefix} <{e[0]:g}"] + [f"{prefix} {a:g}-{b:g}" for a, b in zip(e[:-1], e[1:])]
            + [f"{prefix} >={e[-1]:g}", f"{prefix} undef"])


def cmd_region_labels(bp_rp: ArrayLike, abs_g: ArrayLike, spec: ValidationSpec) -> NDArray[np.str_]:
    """Coarse validation CMD region label per row (``ValidationSpec`` edges)."""
    cn, mn = _bin_names(spec.bp_rp_edges, "BP-RP"), _bin_names(spec.abs_g_mag_edges, "M_G")
    ci = bin_index(bp_rp, np.asarray(spec.bp_rp_edges))
    mi = bin_index(abs_g, np.asarray(spec.abs_g_mag_edges))
    return np.asarray([f"{cn[c]} \\| {mn[m]}" for c, m in zip(ci, mi)])


def g_bin_labels(g_mag: ArrayLike, spec: ValidationSpec) -> NDArray[np.str_]:
    names = _bin_names(spec.g_mag_edges, "G")
    return np.asarray([names[i] for i in bin_index(g_mag, np.asarray(spec.g_mag_edges))])


def markdown_table(rows: Sequence[ValidationRow], *, n_min: int, max_abs_pull: float) -> str:
    head = ("| region | N holdout | N train | undef real | undef emu | pull | reject real | reject emu | pull "
            "| med logg real | med logg emu | gate |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    body = []
    for r in rows:
        g = r.gated(n_min, max_abs_pull)
        gs = "n/a" if g is None else ("ok" if g else "FAIL")
        body.append(
            f"| {r.label} | {r.n_holdout} | {r.n_train} | {r.undef_real:.3f} | {r.undef_emu:.3f} | {r.undef_pull:+.2f} "
            f"| {r.reject_real:.3f} | {r.reject_emu:.3f} | {r.reject_pull:+.2f} | {r.median_logg_real:.2f} "
            f"| {r.median_logg_emu:.2f} | {gs} |"
        )
    return "\n".join([head, *body])
