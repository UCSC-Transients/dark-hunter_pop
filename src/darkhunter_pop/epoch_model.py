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
class ContinuousLossConfig:
    """Continuous per-transit keep probability (#400 E3, E6, E7; spec §4.2).

    ``log p_keep = coef_g[0] + sum_k coef_g[k] x^k + sum_j coef_sky[j] Y_j(l, b)``, with
    ``x = (clip(G, *g_clip) - g_ref) / g_scale`` and ``Y_j`` the real spherical harmonics
    in Galactic coordinates for ``1 <= ell <= sky_lmax`` (order: ell ascending, m from
    -ell to ell). ``p_keep`` is capped at 1. Calibrated on NSS stars at the reference
    RUWE (spec §4.2).
    """

    g_clip: tuple[float, float]
    g_ref: float
    g_scale: float
    coef_g: tuple[float, ...]
    sky_lmax: int
    coef_sky: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.g_clip[0] < self.g_clip[1] or self.g_scale <= 0:
            raise ValueError("g_clip must increase and g_scale must be > 0")
        if len(self.coef_g) < 1:
            raise ValueError("coef_g needs at least the intercept")
        if self.sky_lmax < 0 or len(self.coef_sky) != (self.sky_lmax + 1) ** 2 - 1:
            raise ValueError("coef_sky needs (sky_lmax + 1)^2 - 1 entries")


@dataclass(frozen=True)
class ClusteredLossConfig:
    """Time-clustered part of the transit loss for faint stars (#400 E4; spec §4.3).

    A fraction ``f(G)`` of the total loss comes from loss *episodes*: a Poisson process
    in time with durations ``~ Exponential(tau_day)``. Every transit inside an episode is
    lost. ``f`` rises linearly from 0 at ``g_start`` to ``frac_max`` at ``g_full`` and
    stays there. The rest of the loss is independent per transit.
    """

    g_start: float
    g_full: float
    frac_max: float
    tau_day: float

    def __post_init__(self) -> None:
        if not self.g_start < self.g_full or not 0.0 <= self.frac_max <= 1.0 or self.tau_day <= 0:
            raise ValueError("need g_start < g_full, 0 <= frac_max <= 1 and tau_day > 0")


@dataclass(frozen=True)
class VisibilityPeriodLossConfig:
    """Whole-visibility-period loss with a degraded-star mixture (#432; spec §8.10).

    The star is *degraded* with probability
    ``pi = expit(c0 + c1 x + c2 x^2 + c_beta |sin beta| + c_b f(b))``, ``f(b) = |sin b|`` or
    ``exp(-|b| / b_scale_deg)`` (``b_feature``),
    ``x = (clip(G, *g_clip) - g_ref) / g_scale``, beta ecliptic and b Galactic latitude.
    Each visibility period (after the gaps) is then dropped whole with
    ``q_bad = expit(e0)`` (degraded) or ``q0 = expit(d0 + d1 x)``. The remaining loss is
    independent per transit with ``p_ind = 1 - p_keep / (1 - qbar)``,
    ``qbar = (1 - pi) q0 + pi q_bad``, so the expected kept-transit fraction stays at the
    calibrated ``p_keep``. When set, it replaces the episode model (``clustered``).
    """

    g_clip: tuple[float, float]
    g_ref: float
    g_scale: float
    c0: float
    c1: float
    c2: float
    c_beta: float
    c_b: float
    e0: float
    d0: float
    d1: float
    visibility_gap_day: float = 4.0
    b_feature: str = "abs_sin"
    b_scale_deg: float = 10.0

    def b_term(self, b_deg: float) -> float:
        """Galactic-latitude feature: ``|sin b|`` or ``exp(-|b| / b_scale_deg)``."""
        if self.b_feature == "exp":
            return float(np.exp(-abs(b_deg) / self.b_scale_deg))
        return float(abs(np.sin(np.radians(b_deg))))

    def x(self, g_mag: float) -> float:
        return (float(np.clip(g_mag, *self.g_clip)) - self.g_ref) / self.g_scale

    def degraded_probability(self, g_mag: float, beta_deg: float, b_deg: float) -> float:
        from scipy.special import expit

        x = self.x(g_mag)
        return float(expit(self.c0 + self.c1 * x + self.c2 * x * x
                           + self.c_beta * abs(np.sin(np.radians(beta_deg)))
                           + self.c_b * self.b_term(b_deg)))

    def q_normal(self, g_mag: float) -> float:
        from scipy.special import expit

        return float(expit(self.d0 + self.d1 * self.x(g_mag)))

    def q_degraded(self) -> float:
        from scipy.special import expit

        return float(expit(self.e0))


@dataclass(frozen=True)
class PerCcdExcessNoiseConfig:
    """Bright-star unmodelled per-CCD noise for the unbinned overlay (#398 / #400 N2).

    Extra white noise with variance ``r2(G) * sigma_stated^2`` is added to every CCD
    observation, where ``r2`` is linearly interpolated in ``(knots_g, knots_r2)`` for
    ``G < g_max`` and 0 above. The stated errors are not changed (DR3's NSS fits saw
    this excess as F2 > 0). With ``renormalize_ruwe`` the RUWE gaiamock reports, and the
    RUWE threshold of its cascade, are divided by ``k = sqrt(1 + r2)``, the analogue of
    DR3's RUWE normalisation (Lindegren et al. 2021).
    """

    g_max: float
    knots_g: tuple[float, ...]
    knots_r2: tuple[float, ...]
    renormalize_ruwe: bool = True

    def __post_init__(self) -> None:
        if len(self.knots_g) != len(self.knots_r2) or len(self.knots_g) < 1:
            raise ValueError("knots_g and knots_r2 need the same, non-zero length")
        if np.any(np.diff(self.knots_g) <= 0) or np.any(np.asarray(self.knots_r2) < 0):
            raise ValueError("knots_g must increase and knots_r2 must be >= 0")


@dataclass(frozen=True)
class RuweU0Table:
    """Mock RUWE normalisation ``u0(G)`` emulating DR3's RUWE = UWE / u0(G, C) (#400 N2-u0).

    Lindegren (2018, GAIA-C3-TN-LU-LL-124) defines u0 as the 41st percentile of UWE in
    magnitude-colour bins, smoothed and interpolated. The mock version is the same
    statistic of mock single-star UWE in G bins. gaiamock's noise has no colour term, so the
    colour axis collapses (spec §8.8). Linear interpolation in G, held constant outside
    the table.
    """

    g: tuple[float, ...]
    u0: tuple[float, ...]
    path: Path | None = None

    def __post_init__(self) -> None:
        if len(self.g) != len(self.u0) or len(self.g) < 2 or np.any(np.diff(self.g) <= 0):
            raise ValueError("u0 table needs >= 2 increasing G values and matching u0")
        if np.any(np.asarray(self.u0) <= 0):
            raise ValueError("u0 must be > 0")

    def __call__(self, g_mag: float) -> float:
        return float(np.interp(g_mag, self.g, self.u0))


def load_u0_table(path: Path, expected_sha256: str | None = None) -> RuweU0Table:
    """Read the u0 CSV (``#`` comment lines, then ``g,u0`` header and rows); verify sha256."""
    if expected_sha256:
        got = _sha256(path)
        if got != expected_sha256:
            raise ValueError(f"u0 table {path} sha256 {got} != configured {expected_sha256}")
    rows = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if rows[0].replace(" ", "").lower().split(",")[:2] != ["g", "u0"]:
        raise ValueError(f"unexpected u0 table header {rows[0]!r}")
    vals = np.array([[float(x) for x in r.split(",")[:2]] for r in rows[1:]])
    return RuweU0Table(g=tuple(vals[:, 0]), u0=tuple(vals[:, 1]), path=Path(path))


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
    continuous
        Continuous loss model; when set it replaces the binned ``transit_loss_*``.
    clustered
        Time-clustered loss for faint stars; ``None`` = all loss independent per transit.
    excess_noise
        Bright-star per-CCD excess noise (used by :func:`run_cascade`); ``None`` = none.
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
    continuous: ContinuousLossConfig | None = None
    clustered: ClusteredLossConfig | None = None
    excess_noise: PerCcdExcessNoiseConfig | None = None
    ruwe_u0: RuweU0Table | None = None
    vp_loss: VisibilityPeriodLossConfig | None = None
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
        continuous=_continuous_from(loss.get("continuous")) if loss.get("model", "binned") == "continuous" else None,
        clustered=_clustered_from(section.get("clustered_loss")),
        excess_noise=_noise_from(section.get("bright_excess_noise")),
        ruwe_u0=_u0_from(section.get("ruwe_u0"), root),
        vp_loss=_vp_from(section.get("visibility_period_loss")),
        provenance=str(section.get("provenance", "")),
    )


def _vp_from(m: Mapping[str, Any] | None) -> VisibilityPeriodLossConfig | None:
    if m is None or not m.get("enabled", False):
        return None
    return VisibilityPeriodLossConfig(
        g_clip=(float(m["g_clip"][0]), float(m["g_clip"][1])), g_ref=float(m["g_ref"]),
        g_scale=float(m["g_scale"]), c0=float(m["c0"]), c1=float(m["c1"]), c2=float(m["c2"]),
        c_beta=float(m["c_beta"]), c_b=float(m["c_b"]), e0=float(m["e0"]), d0=float(m["d0"]),
        d1=float(m["d1"]), visibility_gap_day=float(m.get("visibility_gap_day", 4.0)),
        b_feature=str(m.get("b_feature", "abs_sin")), b_scale_deg=float(m.get("b_scale_deg", 10.0)),
    )


def _u0_from(m: Mapping[str, Any] | None, root: Path) -> RuweU0Table | None:
    if m is None or not m.get("enabled", False):
        return None
    path = Path(str(m["table"]))
    return load_u0_table(path if path.is_absolute() else root / path, m.get("table_sha256"))


def _continuous_from(m: Mapping[str, Any] | None) -> ContinuousLossConfig | None:
    if m is None:
        raise ValueError("transit_loss.model is 'continuous' but transit_loss.continuous is missing")
    return ContinuousLossConfig(
        g_clip=(float(m["g_clip"][0]), float(m["g_clip"][1])), g_ref=float(m["g_ref"]),
        g_scale=float(m["g_scale"]), coef_g=tuple(float(x) for x in m["coef_g"]),
        sky_lmax=int(m["sky_lmax"]), coef_sky=tuple(float(x) for x in m["coef_sky"]),
    )


def _clustered_from(m: Mapping[str, Any] | None) -> ClusteredLossConfig | None:
    if m is None or not m.get("enabled", False):
        return None
    return ClusteredLossConfig(g_start=float(m["g_start"]), g_full=float(m["g_full"]),
                               frac_max=float(m["frac_max"]), tau_day=float(m["tau_day"]))


def _noise_from(m: Mapping[str, Any] | None) -> PerCcdExcessNoiseConfig | None:
    if m is None or not m.get("enabled", False):
        return None
    return PerCcdExcessNoiseConfig(
        g_max=float(m["g_max"]), knots_g=tuple(float(x) for x in m["knots_g"]),
        knots_r2=tuple(float(x) for x in m["knots_r2"]),
        renormalize_ruwe=bool(m.get("renormalize_ruwe", True)),
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


def real_sph_harm_galactic(
    l_deg: NDArray[np.float64] | float, b_deg: NDArray[np.float64] | float, lmax: int
) -> NDArray[np.float64]:
    """Real spherical harmonics ``Y_{ell m}(l, b)`` for ``1 <= ell <= lmax``, shape ``(N, (lmax+1)^2 - 1)``.

    Columns ordered by ell ascending and m from -ell to ell: ``sqrt(2) Im Y_ell^|m|`` for
    m < 0, ``Re Y_ell^0`` for m = 0, ``sqrt(2) Re Y_ell^m`` for m > 0 (orthonormal on the
    sphere). The monopole is the model intercept and is not included.
    """
    from scipy import special

    phi = np.radians(np.atleast_1d(np.asarray(l_deg, dtype=np.float64)))
    theta = np.radians(90.0 - np.atleast_1d(np.asarray(b_deg, dtype=np.float64)))
    cols = []
    for ell in range(1, int(lmax) + 1):
        for m in range(-ell, ell + 1):
            y = special.sph_harm_y(ell, abs(m), theta, phi)
            cols.append(np.sqrt(2.0) * y.imag if m < 0 else (y.real if m == 0 else np.sqrt(2.0) * y.real))
    return np.column_stack(cols) if cols else np.zeros((phi.size, 0))


def g_polynomial_basis(g_mag: NDArray[np.float64] | float, cont: ContinuousLossConfig) -> NDArray[np.float64]:
    """``[x, x^2, ..., x^d]`` with ``x = (clip(G) - g_ref) / g_scale``; shape ``(N, d)``."""
    g = np.clip(np.atleast_1d(np.asarray(g_mag, dtype=np.float64)), *cont.g_clip)
    x = (g - cont.g_ref) / cont.g_scale
    d = len(cont.coef_g) - 1
    return np.column_stack([x**k for k in range(1, d + 1)]) if d > 0 else np.zeros((g.size, 0))


def keep_probability(
    g_mag: float,
    config: EpochModelConfig,
    *,
    l_deg: float | None = None,
    b_deg: float | None = None,
    density_per_deg2: float | None = None,
) -> float:
    """Total probability that a FoV transit outside the gaps is kept.

    Continuous model (``config.continuous``): ``min(1, exp(eta(G, l, b)))``; the sky term
    needs ``l_deg`` and ``b_deg`` (raises if they are missing and ``sky_lmax > 0``).
    Otherwise the binned model, ``1 - transit_loss_probability``.
    """
    cont = config.continuous
    if cont is None:
        return 1.0 - transit_loss_probability(g_mag, config, density_per_deg2=density_per_deg2)
    eta = cont.coef_g[0] + float(g_polynomial_basis(g_mag, cont)[0] @ np.asarray(cont.coef_g[1:]))
    if cont.sky_lmax > 0:
        if l_deg is None or b_deg is None:
            raise ValueError("the continuous loss model has a sky term; pass l_deg and b_deg")
        eta += float(real_sph_harm_galactic(l_deg, b_deg, cont.sky_lmax)[0] @ np.asarray(cont.coef_sky))
    return float(min(1.0, np.exp(eta)))


def clustered_fraction(g_mag: float, config: EpochModelConfig) -> float:
    """Fraction of the loss that comes from time-clustered episodes (0 without E4)."""
    cl = config.clustered
    if cl is None:
        return 0.0
    ramp = (g_mag - cl.g_start) / (cl.g_full - cl.g_start)
    return float(cl.frac_max * np.clip(ramp, 0.0, 1.0))


def loss_episode_mask(
    t_day: NDArray[np.float64], covered_fraction: float, tau_day: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """True where ``t_day`` falls inside a loss episode.

    Episodes start as a Poisson process with rate ``lambda = -ln(1 - e) / tau`` and last
    ``~ Exponential(tau)``, which covers a long-run fraction ``e = covered_fraction`` of
    the time. Simulated from ``min(t) - 10 tau`` so episodes may start before the first
    transit. Draw order: number of episodes, starts, durations.
    """
    t = np.asarray(t_day, dtype=np.float64)
    if t.size == 0 or covered_fraction <= 0.0:
        return np.zeros(t.shape, dtype=bool)
    e = min(float(covered_fraction), 0.999)
    lam = -np.log1p(-e) / tau_day
    t0, t1 = float(t.min()) - 10.0 * tau_day, float(t.max())
    n = int(rng.poisson(lam * (t1 - t0)))
    starts = np.sort(rng.uniform(t0, t1, n))
    ends = starts + rng.exponential(tau_day, n)
    if n == 0:
        return np.zeros(t.shape, dtype=bool)
    return in_gaps(t, np.column_stack([starts, ends]))


def thin_gost_mask(
    jd: NDArray[np.float64],
    config: EpochModelConfig,
    gaps_jd: NDArray[np.float64],
    *,
    g_mag: float,
    rng: np.random.Generator,
    density_per_deg2: float | None = None,
    l_deg: float | None = None,
    b_deg: float | None = None,
    beta_deg: float | None = None,
) -> NDArray[np.bool_]:
    """Boolean keep-mask over GOST rows: gaps removed, then whole transits dropped.

    Loss after the gaps is ``q = 1 - keep_probability``. Three modes:

    * ``config.vp_loss`` set (#432): visibility periods (groups of post-gap transits
      separated by > 4 d) are dropped whole with the degraded-star mixture of
      :class:`VisibilityPeriodLossConfig` (needs ``beta_deg`` and ``b_deg``); the rest is
      independent per transit. Draw order: one uniform per transit, one for the degraded
      state, one per visibility period.
    * else ``config.clustered`` (E4): a fraction ``e = clustered_fraction(G) * q`` is lost in
      time-clustered episodes (:func:`loss_episode_mask`), the rest independently per
      transit with ``p_ind = 1 - (1 - q) / (1 - e)``. Draw order: transits, then episodes.
    * else independent per transit.

    In every mode the expected total keep is ``1 - q`` and every row of a transit shares
    its fate.
    """
    t = np.asarray(jd, dtype=np.float64)
    keep = np.ones(t.shape, dtype=bool)
    if config.apply_gaps:
        keep &= ~in_gaps(t, gaps_jd)
    ids = fov_transit_ids(t, config.transit_split_day)
    n_tr = int(ids.max()) + 1 if ids.size else 0
    q = 1.0 - keep_probability(g_mag, config, l_deg=l_deg, b_deg=b_deg, density_per_deg2=density_per_deg2)
    vp = config.vp_loss
    if vp is not None:
        if beta_deg is None or b_deg is None:
            raise ValueError("the visibility-period loss needs beta_deg and b_deg")
        pi = vp.degraded_probability(g_mag, beta_deg, b_deg)
        q0, qb = vp.q_normal(g_mag), vp.q_degraded()
        qbar = (1.0 - pi) * q0 + pi * qb
        p_ind = float(np.clip(1.0 - (1.0 - q) / max(1e-9, 1.0 - qbar), 0.0, 0.999))
        u = rng.uniform(0.0, 1.0, n_tr)
        keep &= (u >= p_ind)[ids]
        degraded = rng.uniform() < pi
        qv = qb if degraded else q0
        if n_tr:
            # visibility periods of the transits that survive the gaps (as the scan law sees them)
            row_ok = ~in_gaps(t, gaps_jd) if config.apply_gaps else np.ones(t.shape, dtype=bool)
            tr_ok = np.bincount(ids, weights=row_ok.astype(float), minlength=n_tr) > 0
            tr_idx = np.flatnonzero(tr_ok)
            if tr_idx.size:
                t_mid = np.bincount(ids, weights=t, minlength=n_tr) / np.bincount(ids, minlength=n_tr)
                vp_id = np.r_[0, np.cumsum(np.diff(t_mid[tr_idx]) > vp.visibility_gap_day)]
                drop_vp = rng.uniform(0.0, 1.0, int(vp_id.max()) + 1) < qv
                lost_tr = np.zeros(n_tr, dtype=bool)
                lost_tr[tr_idx[drop_vp[vp_id]]] = True
                keep &= ~lost_tr[ids]
        return keep
    e = clustered_fraction(g_mag, config) * q
    p_ind = 1.0 - (1.0 - q) / (1.0 - e) if e < 1.0 else 1.0
    u = rng.uniform(0.0, 1.0, n_tr)
    keep &= (u >= p_ind)[ids]
    if e > 0.0 and config.clustered is not None and n_tr:
        t_mid = np.bincount(ids, weights=t, minlength=n_tr) / np.bincount(ids, minlength=n_tr)
        lost = loss_episode_mask(t_mid, e, config.clustered.tau_day, rng)
        keep &= ~lost[ids]
    return keep


# --------------------------------------------------------------------------------------
# The gaiamock wrapper
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEpochContext:
    """What the epoch model needs to know about the simulated source."""

    g_mag: float
    density_per_deg2: float | None = None
    l_deg: float | None = None
    b_deg: float | None = None
    beta_deg: float | None = None


def source_context(ra_deg: float, dec_deg: float, g_mag: float,
                   density_per_deg2: float | None = None) -> SourceEpochContext:
    """:class:`SourceEpochContext` with Galactic (l, b) and ecliptic latitude from (ra, dec)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    gal = c.galactic
    ecl = c.barycentrictrueecliptic
    return SourceEpochContext(g_mag=float(g_mag), density_per_deg2=density_per_deg2,
                              l_deg=float(gal.l.deg), b_deg=float(gal.b.deg), beta_deg=float(ecl.lat.deg))


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
            density_per_deg2=source.density_per_deg2, l_deg=source.l_deg, b_deg=source.b_deg,
            beta_deg=source.beta_deg,
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


# --------------------------------------------------------------------------------------
# Bright-star per-CCD excess noise (#398 / #400 N2) and the cascade wrapper
# --------------------------------------------------------------------------------------

#: Tag of the per-CCD excess-noise Generator.
PER_CCD_NOISE_RNG_TAG: Final[int] = 3982

#: Index of the RUWE entry in ``fit_full_astrometric_cascade``'s return vector, by the
#: branch flag in element 0 (gaiamock_mod ``gaiamock-mod-v1``; see
#: ``injection_test.parse_cascade_result``). The orbital branch (any other flag) has it
#: at 22. Flag 0 (too few visibility periods) has none.
CASCADE_RUWE_INDEX: Final[dict[float, int]] = {-1.0: 1, -7.0: 8, -9.0: 12}
CASCADE_RUWE_INDEX_ORBITAL: Final[int] = 22


def excess_noise_r2(g_mag: float, noise: PerCcdExcessNoiseConfig) -> float:
    """``r2(G)``: excess variance in units of the stated per-CCD variance (0 at G >= g_max)."""
    if not g_mag < noise.g_max:
        return 0.0
    return float(max(0.0, np.interp(g_mag, noise.knots_g, noise.knots_r2)))


def per_ccd_excess_noise(
    ast_err: NDArray[np.float64], g_mag: float, noise: PerCcdExcessNoiseConfig, rng: np.random.Generator
) -> tuple[NDArray[np.float64], float]:
    """White per-CCD noise ``N(0, r2(G) ast_err^2)`` and the RUWE scale ``k = sqrt(1 + r2)``.

    Returns zeros and ``k = 1`` without drawing when ``r2 = 0``.
    """
    err = np.asarray(ast_err, dtype=np.float64)
    r2 = excess_noise_r2(g_mag, noise)
    if r2 <= 0.0:
        return np.zeros(err.shape), 1.0
    k = float(np.sqrt(1.0 + r2)) if noise.renormalize_ruwe else 1.0
    return rng.normal(0.0, 1.0, err.size) * np.sqrt(r2) * err, k


def ruwe_scale_u0(g_mag: float, config: EpochModelConfig) -> float:
    """``u0_mock(G)`` when the u0 table is configured, else 1 (RUWE = UWE / u0)."""
    return 1.0 if config.ruwe_u0 is None else config.ruwe_u0(g_mag)


def rescale_cascade_ruwe(cascade: list[float], k: float) -> list[float]:
    """Divide the RUWE entry of a cascade vector by ``k`` (no-op for k = 1 or flag 0)."""
    out = [float(x) for x in cascade]
    if k == 1.0 or not out:
        return out
    flag = out[0]
    if flag == 0.0:
        return out
    idx = CASCADE_RUWE_INDEX.get(flag, CASCADE_RUWE_INDEX_ORBITAL)
    if idx < len(out):
        out[idx] = out[idx] / k
    return out


@dataclass(frozen=True)
class CascadeRun:
    """One wrapped gaiamock prediction + cascade."""

    cascade: list[float]
    n_obs: int
    n_transits: int
    n_visibility_periods: int
    ruwe_scale: float


def run_cascade(
    gaiamock: ModuleType,
    c_funcs: Any,
    predict: Any,
    config: EpochModelConfig,
    source: SourceEpochContext,
    *,
    epoch_rng: np.random.Generator,
    noise_rng: np.random.Generator,
    ruwe_min: float,
    skip_acceleration: bool,
    gaps_jd: NDArray[np.float64] | None = None,
    visibility_gap_day: float = 4.0,
) -> CascadeRun:
    """gaiamock prediction and cascade with the epoch model and the excess noise.

    ``predict()`` is a zero-argument callable that calls one of gaiamock's
    ``predict_astrometry_*`` functions and returns ``(t_ast_yr, psi, plx_factor, obs,
    err)``. It runs inside :func:`gost_epoch_model`. Then the per-CCD excess noise is
    added to ``obs`` and ``gaiamock.fit_full_astrometric_cascade`` runs with
    ``ruwe_min * k``. The RUWE in the returned vector is divided by ``k``, which is the
    same as computing RUWE with errors inflated by ``k`` (RUWE scales as 1/error).
    ``k`` is ``u0_mock(G)`` when ``config.ruwe_u0`` is set (DR3's RUWE = UWE / u0, so the
    cascade's RUWE > ruwe_min gate becomes UWE > ruwe_min * u0), else ``sqrt(1 + r2)`` from
    the excess noise when it renormalises, else 1. This is
    the composition ``run_full_astrometric_cascade`` itself performs (predict, then fit),
    with the noise inserted between the two calls. No gaiamock function is reimplemented.

    The caller enters ``forward_model.seeded_global_rng`` around this call, exactly as for
    a bare gaiamock call.
    """
    with gost_epoch_model(gaiamock, config, source, epoch_rng, gaps_jd=gaps_jd):
        t, psi, pf, obs, err = predict()
    t = np.asarray(t, dtype=np.float64)
    k = 1.0
    if config.enabled and config.excess_noise is not None:
        extra, k = per_ccd_excess_noise(err, source.g_mag, config.excess_noise, noise_rng)
        obs = np.asarray(obs, dtype=np.float64) + extra
    if config.enabled and config.ruwe_u0 is not None:
        k = ruwe_scale_u0(source.g_mag, config)
    n_vis = n_visibility_periods_from_days(t * 365.25, visibility_gap_day)
    n_tr = int(fov_transit_ids(t * 365.25, config.transit_split_day).max() + 1) if t.size else 0
    cascade = gaiamock.fit_full_astrometric_cascade(
        t_ast_yr=t, psi=psi, plx_factor=pf, ast_obs=obs, ast_err=err, c_funcs=c_funcs,
        verbose=False, show_residuals=False, ruwe_min=ruwe_min * k,
        skip_acceleration=skip_acceleration,
    )
    return CascadeRun(
        cascade=rescale_cascade_ruwe(list(cascade), k), n_obs=int(t.size), n_transits=n_tr,
        n_visibility_periods=n_vis, ruwe_scale=k,
    )
