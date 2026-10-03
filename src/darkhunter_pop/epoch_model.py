"""Pop-side statistical epoch model wrapped around gaiamock_mod's GOST transit list (#400, #398).

Spec: ``docs/EPOCH_MODEL_SPEC.md``. gaiamock_mod predicts epochs from the GOST
(Gaia Observation Forecast Tool) scanning law for the nearest of 3,072 HEALPix-16 sky
positions and then drops a random 10% of the rows. GOST knows nothing about the
spacecraft gaps and the transits that DR3's astrometry did not use, so gaiamock has more
FoV transits and visibility periods than DR3 (#400, #398 / ``docs/gate399``).

This module **does not edit or reimplement gaiamock_mod**. It changes only the transit
list gaiamock reads: inside :func:`gost_epoch_model` the module attribute
``gaiamock.get_gost_one_position`` is replaced by a wrapper that calls the original
function and then thins its table with :func:`thin_gost_table`. Everything downstream
(``predict_astrometry_*``, its own 10% row rejection, noise, the cascade) is gaiamock's
code, unchanged.

The model is **statistical**: no per-star lookup of real DR3 counts. It has two parts:

1. deterministic removal of every GOST row inside a published DR3 astrometric data gap
   (ESA, "Gaps in Gaia (E)DR3 data", ``Astrometry_EDR3.csv``, 138 gaps; Lindegren et al.
   2021 Sect. 2.2 and Table 1);
2. a calibrated probability that a whole FoV transit is lost, a function of the source's
   G magnitude (and optionally the local source density), fit to real DR3
   ``astrometric_matched_transits`` (``scripts/measure_epoch_counts_400.py``).

Hook point for the mock population (#391): wrap any block that calls
``run_full_astrometric_cascade`` / ``predict_astrometry_*`` (``proposal_set`` /
``forward_model`` draws, ``cascade_replay``) in::

    with seeded_global_rng(seeds, c_funcs), gost_epoch_model(gaiamock, model, star, rng):
        res = gaiamock.run_full_astrometric_cascade(...)

where ``rng = epoch_model_rng(base_seed, stream, index)`` is seeded from the draw's own
seed key, so the thinning replays bit for bit. ``proposal_set`` / ``moe_distefano``
(#391) are not touched by this module; the call above is where they would hook in.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

#: GOST column holding the barycentric observation time (JD, TCB).
GOST_TIME_COLUMN: Final[str] = "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"
#: GOST column holding the CCD row (1-7) of the transit.
GOST_CCD_ROW_COLUMN: Final[str] = "CcdRow[1-7]"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochModelConfig:
    """Parameters of the epoch model (``dr3.epoch_model`` in config).

    Attributes
    ----------
    enabled
        Master switch. ``False`` leaves gaiamock's GOST list untouched.
    gap_table
        Path to the published astrometric gap list (OBMT revolutions; CSV columns
        ``start,end,duration [rev],description``).
    apply_gaps
        Remove GOST rows inside the gaps.
    obmt_reference_rev, obmt_reference_jd_tcb, obmt_rev_per_day
        Lindegren et al. (2021) Eq. 2: ``JD_TCB = ref_jd + (OBMT - ref_rev) / rev_per_day``.
    transit_split_day
        Consecutive GOST rows closer than this belong to the same FoV transit (the ten
        rows of one transit span < 1 min; two transits are >= 1.76 h apart).
    transit_loss_g_edges
        Bin edges in G (mag) of the per-transit loss probability; ``len(edges) - 1``
        probabilities in ``transit_loss_prob``. Outside the edges the nearest bin applies.
    transit_loss_prob
        Probability that a whole FoV transit (all its rows) is dropped, per G bin, after
        the gaps. Calibrated on DR3 (spec §3).
    transit_loss_density_slope
        Optional linear change of the loss probability per dex of local source density
        relative to ``transit_loss_density_ref`` (sources per deg^2). 0 disables it.
    transit_loss_density_ref
        Reference density (sources per deg^2) at which ``transit_loss_prob`` applies.
    agis_window_obmt_rev
        ``(start, end)`` OBMT of the data the DR3 astrometric solution used (Lindegren et
        al. 2021 Sect. 2.2: 1192.13-5230.09). Rows outside are removed with the gaps.
        ``None`` keeps gaiamock's own JD window.
    """

    enabled: bool
    gap_table: Path
    apply_gaps: bool
    obmt_reference_rev: float
    obmt_reference_jd_tcb: float
    obmt_rev_per_day: float
    transit_split_day: float
    transit_loss_g_edges: tuple[float, ...]
    transit_loss_prob: tuple[float, ...]
    transit_loss_density_slope: float = 0.0
    transit_loss_density_ref: float = 1.0
    agis_window_obmt_rev: tuple[float, float] | None = None
    provenance: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.transit_loss_prob) != len(self.transit_loss_g_edges) - 1:
            raise ValueError(
                "transit_loss_prob needs len(transit_loss_g_edges) - 1 entries, got "
                f"{len(self.transit_loss_prob)} and {len(self.transit_loss_g_edges)}"
            )
        if np.any(np.diff(self.transit_loss_g_edges) <= 0):
            raise ValueError("transit_loss_g_edges must increase")
        p = np.asarray(self.transit_loss_prob, dtype=np.float64)
        if np.any((p < 0) | (p >= 1)):
            raise ValueError("transit_loss_prob must be in [0, 1)")
        if self.obmt_rev_per_day <= 0 or self.transit_split_day <= 0:
            raise ValueError("obmt_rev_per_day and transit_split_day must be > 0")
        if self.transit_loss_density_ref <= 0:
            raise ValueError("transit_loss_density_ref must be > 0")
        if self.agis_window_obmt_rev is not None and not (
            self.agis_window_obmt_rev[0] < self.agis_window_obmt_rev[1]
        ):
            raise ValueError("agis_window_obmt_rev must be (start, end) with start < end")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_model_config_from_mapping(
    section: Mapping[str, Any] | Any, *, repo_root: Path | None = None
) -> EpochModelConfig:
    """Build :class:`EpochModelConfig` from ``dr3.epoch_model`` (mapping or schema model).

    A relative ``gap_table`` is resolved against ``repo_root`` (default: the repository
    that contains this package). When ``gap_table_sha256`` is given, the file's checksum
    must match it (raises ``ValueError``), so a silently edited gap list cannot be used.
    """
    if hasattr(section, "model_dump"):
        section = section.model_dump(mode="python")
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    gap = Path(str(section["gap_table"]))
    if not gap.is_absolute():
        gap = root / gap
    expected = section.get("gap_table_sha256")
    if expected:
        got = _sha256(gap)
        if got != expected:
            raise ValueError(f"gap table {gap} sha256 {got} != configured {expected}")
    loss = section["transit_loss"]
    window = section.get("agis_window_obmt_rev")
    return EpochModelConfig(
        enabled=bool(section["enabled"]),
        gap_table=gap,
        apply_gaps=bool(section["apply_gaps"]),
        obmt_reference_rev=float(section["obmt_reference_rev"]),
        obmt_reference_jd_tcb=float(section["obmt_reference_jd_tcb"]),
        obmt_rev_per_day=float(section["obmt_rev_per_day"]),
        transit_split_day=float(section["transit_split_day"]),
        transit_loss_g_edges=tuple(float(x) for x in loss["g_edges"]),
        transit_loss_prob=tuple(float(x) for x in loss["prob"]),
        transit_loss_density_slope=float(loss.get("density_slope_per_dex", 0.0)),
        transit_loss_density_ref=float(loss.get("density_ref_per_deg2", 1.0)),
        agis_window_obmt_rev=(
            None if window is None else (float(window[0]), float(window[1]))
        ),
        provenance=str(section.get("provenance", "")),
    )


# --------------------------------------------------------------------------------------
# Time and gaps
# --------------------------------------------------------------------------------------


def obmt_to_jd_tcb(
    obmt_rev: NDArray[np.float64] | float,
    *,
    reference_rev: float,
    reference_jd_tcb: float,
    rev_per_day: float,
) -> NDArray[np.float64]:
    """OBMT revolutions to JD (TCB), Lindegren et al. (2021) Eq. 2.

    The relation is approximate (the spin period is not exactly 6 h); its error over the
    DR3 window is far below the duration of any gap that matters here. GOST times are
    barycentric, which differs from the spacecraft TCB by the Roemer delay (< 8.5 min);
    that is also negligible against the gaps, and is recorded as a limitation.
    """
    return reference_jd_tcb + (np.asarray(obmt_rev, dtype=np.float64) - reference_rev) / rev_per_day


def load_gap_table(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Read the ESA astrometric gap CSV: ``(start_rev, end_rev, description)``.

    Handles the UTF-8 byte-order mark the published file carries.
    """
    starts: list[float] = []
    ends: list[float] = []
    desc: list[str] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0].strip().lower() != "start":
            raise ValueError(f"unexpected gap-table header {header!r} in {path}")
        for row in reader:
            if not row:
                continue
            starts.append(float(row[0]))
            ends.append(float(row[1]))
            desc.append(row[3].strip() if len(row) > 3 else "")
    s = np.asarray(starts, dtype=np.float64)
    e = np.asarray(ends, dtype=np.float64)
    if np.any(e < s):
        raise ValueError("gap with end < start")
    return s, e, desc


def gap_intervals_jd(config: EpochModelConfig) -> NDArray[np.float64]:
    """The gaps as an ``(n, 2)`` array of JD (TCB) intervals, sorted by start.

    With ``config.agis_window_obmt_rev`` set, the time before and after the AGIS window
    is added as two open-ended gaps (as the ``scanninglaw`` package does).
    """
    s, e, _ = load_gap_table(config.gap_table)
    if config.agis_window_obmt_rev is not None:
        lo, hi = config.agis_window_obmt_rev
        s = np.r_[-np.inf, s, hi]
        e = np.r_[lo, e, np.inf]
    kw = dict(
        reference_rev=config.obmt_reference_rev,
        reference_jd_tcb=config.obmt_reference_jd_tcb,
        rev_per_day=config.obmt_rev_per_day,
    )
    out = np.column_stack([obmt_to_jd_tcb(s, **kw), obmt_to_jd_tcb(e, **kw)])
    return out[np.argsort(out[:, 0])]


def in_gaps(jd: NDArray[np.float64], gaps_jd: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True where ``jd`` falls inside any closed interval of ``gaps_jd`` (vectorized).

    Overlapping gaps are allowed.
    """
    t = np.asarray(jd, dtype=np.float64)
    if gaps_jd.size == 0:
        return np.zeros(t.shape, dtype=bool)
    starts = gaps_jd[:, 0]
    # running max of ends handles overlaps: a time is inside if, among gaps starting
    # at or before it, the largest end is >= it.
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_runmax = np.maximum.accumulate(gaps_jd[order, 1])
    idx = np.searchsorted(s_sorted, t, side="right") - 1
    inside = np.zeros(t.shape, dtype=bool)
    ok = idx >= 0
    inside[ok] = t[ok] <= e_runmax[idx[ok]]
    return inside


# --------------------------------------------------------------------------------------
# Transits
# --------------------------------------------------------------------------------------


def fov_transit_ids(jd: NDArray[np.float64], split_day: float) -> NDArray[np.int64]:
    """Label each GOST row with its FoV transit (0, 1, ...), in time order of ``jd``.

    Rows need not be sorted; labels follow sorted time. Rows separated by less than
    ``split_day`` from the previous row (in time order) share a transit.
    """
    t = np.asarray(jd, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    new = np.r_[True, np.diff(t[order]) >= split_day]
    labels_sorted = np.cumsum(new) - 1
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels.astype(np.int64)


def n_visibility_periods_from_days(t_day: NDArray[np.float64], gap_day: float) -> int:
    """Visibility periods as gaiamock and DR3 define them (groups split by > ``gap_day``)."""
    t = np.sort(np.asarray(t_day, dtype=np.float64))
    if t.size == 0:
        return 0
    return int(np.sum(np.diff(t) > gap_day) + 1)


def transit_loss_probability(
    g_mag: float, config: EpochModelConfig, *, density_per_deg2: float | None = None
) -> float:
    """Per-FoV-transit loss probability for a source of magnitude ``g_mag``.

    Piecewise constant in G (nearest bin outside the edges). With a non-zero
    ``transit_loss_density_slope`` and a ``density_per_deg2``, adds
    ``slope * log10(density / ref)``. Clipped to [0, 0.99].
    """
    edges = np.asarray(config.transit_loss_g_edges, dtype=np.float64)
    probs = np.asarray(config.transit_loss_prob, dtype=np.float64)
    i = int(np.clip(np.searchsorted(edges, g_mag, side="right") - 1, 0, probs.size - 1))
    p = float(probs[i])
    if config.transit_loss_density_slope != 0.0 and density_per_deg2 is not None:
        p += config.transit_loss_density_slope * float(
            np.log10(max(density_per_deg2, 1e-12) / config.transit_loss_density_ref)
        )
    return float(np.clip(p, 0.0, 0.99))


def thin_gost_mask(
    jd: NDArray[np.float64],
    config: EpochModelConfig,
    gaps_jd: NDArray[np.float64],
    *,
    g_mag: float,
    rng: np.random.Generator,
    density_per_deg2: float | None = None,
) -> NDArray[np.bool_]:
    """Boolean keep-mask over GOST rows: gaps removed, then whole transits dropped.

    One uniform draw per FoV transit (in time order), so the loss is fully correlated
    within a transit and independent between transits. The number of draws depends only
    on the number of transits, which makes the stream reproducible for a seeded ``rng``.
    """
    t = np.asarray(jd, dtype=np.float64)
    keep = np.ones(t.shape, dtype=bool)
    if config.apply_gaps:
        keep &= ~in_gaps(t, gaps_jd)
    ids = fov_transit_ids(t, config.transit_split_day)
    n_tr = int(ids.max()) + 1 if ids.size else 0
    p = transit_loss_probability(g_mag, config, density_per_deg2=density_per_deg2)
    u = rng.uniform(0.0, 1.0, n_tr)
    keep &= (u >= p)[ids]
    return keep


# --------------------------------------------------------------------------------------
# The gaiamock wrapper
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEpochContext:
    """What the epoch model needs to know about the simulated source."""

    g_mag: float
    density_per_deg2: float | None = None


@contextlib.contextmanager
def gost_epoch_model(
    gaiamock: ModuleType,
    config: EpochModelConfig,
    source: SourceEpochContext,
    rng: np.random.Generator,
    *,
    gaps_jd: NDArray[np.float64] | None = None,
) -> Iterator[None]:
    """Within the block, gaiamock's GOST lookup returns the epoch-model-thinned table.

    Replaces the module attribute ``gaiamock.get_gost_one_position`` with a wrapper
    around the original and restores it on exit (also on error). gaiamock's
    ``predict_astrometry_*`` functions look the name up at call time, so they read the
    thinned table; their own 10% row rejection and everything else is unchanged. With
    ``config.enabled`` false the block is a no-op.

    Not thread-safe (it patches a module attribute); use one process per worker, as
    every pop gaiamock driver already does.
    """
    if not config.enabled:
        yield
        return
    original = gaiamock.get_gost_one_position
    gaps = gap_intervals_jd(config) if gaps_jd is None else gaps_jd

    def _thinned(ra: float, dec: float, data_release: str) -> Any:
        tab = original(ra, dec, data_release=data_release)
        jd = np.asarray(tab[GOST_TIME_COLUMN], dtype=np.float64)
        mask = thin_gost_mask(
            jd, config, gaps, g_mag=source.g_mag, rng=rng,
            density_per_deg2=source.density_per_deg2,
        )
        return tab[mask]

    gaiamock.get_gost_one_position = _thinned
    try:
        yield
    finally:
        gaiamock.get_gost_one_position = original


#: Last spawn-key element of every epoch-model Generator; keeps its stream disjoint
#: from the gaiamock global-RNG seeds that share the other key elements.
EPOCH_MODEL_RNG_TAG: Final[int] = 400
#: Tag of the validation-only excess-noise Generator.
EXCESS_NOISE_RNG_TAG: Final[int] = 3981


def epoch_model_rng(base_seed: int, *spawn_key: int, tag: int = EPOCH_MODEL_RNG_TAG) -> np.random.Generator:
    """Independent, reproducible Generator for one draw's epoch thinning.

    ``SeedSequence(base_seed, spawn_key=(*spawn_key, tag))``. Pass the same key the draw
    uses for its gaiamock seeds (e.g. ``(stream, index)`` or ``(stream, source_id,
    realization)``); the trailing ``tag`` keeps the two streams disjoint.
    """
    key = tuple(int(k) for k in spawn_key) + (int(tag),)
    if int(base_seed) < 0 or any(k < 0 for k in key):
        raise ValueError("seed components must be non-negative")
    return np.random.default_rng(np.random.SeedSequence(int(base_seed), spawn_key=key))


# --------------------------------------------------------------------------------------
# Validation-only: El-Badry et al. (2024) Sect. 3.3.1 bright-star excess noise
# --------------------------------------------------------------------------------------


def bright_star_excess_noise(
    t_ast_yr: NDArray[np.float64],
    g_mag: float,
    rng: np.random.Generator,
    *,
    g_max: float,
    sigma_max_mas: float,
    split_day: float,
) -> tuple[NDArray[np.float64], float]:
    """Per-FoV-transit unmodelled noise for ``G < g_max`` (El-Badry et al. 2024 §3.3.1).

    One ``sigma ~ U(0, sigma_max_mas)`` is drawn for the source (this realization), then
    one ``N(0, sigma)`` offset per FoV transit, shared by every CCD row of that transit
    (the paper bins CCDs per transit, so its per-transit noise is common to them). The
    stated errors are not changed: the noise is unmodelled, as in the paper. Whether
    sigma is drawn per source or per transit is not stated unambiguously in the paper
    and is an open option (spec §6).

    Returns ``(offsets_mas, sigma_mas)``; zeros and 0.0 for ``G >= g_max`` (no draw is
    consumed then, so the stream is reproducible whatever G is).
    """
    t = np.asarray(t_ast_yr, dtype=np.float64)
    if not g_mag < g_max or sigma_max_mas <= 0:
        return np.zeros(t.shape), 0.0
    sigma = float(rng.uniform(0.0, sigma_max_mas))
    ids = fov_transit_ids(t * 365.25, split_day)
    n_tr = int(ids.max()) + 1 if ids.size else 0
    per_transit = rng.normal(0.0, sigma, n_tr)
    return per_transit[ids], sigma
