"""Janssens et al. (2022) dwarf mass–magnitude inversion (CONTINUATION_PLAN §8.2).

Frozen Table 1 parameters only. Do not re-fit, re-digitize, or extrapolate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.config_loader import repo_root

UNINFORMATIVE_SEGMENT_MLOW: float = 1.55
UNINFORMATIVE_SEGMENT_MUP: float = 1.80
DEFAULT_TABLE: str = "config/selections/external/janssens2022_mass_magnitude.yaml"


@dataclass(frozen=True)
class JanssensSegment:
    """One piecewise ``M_G = a log10(M) + b`` regime."""

    m_low: float
    m_up: float
    a: float
    b: float
    a_err: float
    b_err: float
    mg_at_m_low: float
    mg_at_m_up: float

    @property
    def mg_min(self) -> float:
        return min(self.mg_at_m_low, self.mg_at_m_up)

    @property
    def mg_max(self) -> float:
        return max(self.mg_at_m_low, self.mg_at_m_up)


@dataclass
class JanssensInversionResult:
    """Inverted mass plus occupancy diagnostics. ``mass_msun`` is None if N/A."""

    mass_msun: float | None
    segment_index: int | None
    boundary_resolved: bool
    uninformative_segment: bool
    reason: str | None = None


def _mg_of_mass(mass_msun: float, a: float, b: float) -> float:
    return float(a * np.log10(mass_msun) + b)


def load_janssens_table(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen Janssens Table 1 YAML (verbatim segments)."""
    table_path = Path(path) if path is not None else repo_root() / DEFAULT_TABLE
    if not table_path.is_absolute():
        table_path = repo_root() / table_path
    raw = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Janssens table root must be a mapping: {table_path}")
    return raw


def segments_from_table(raw: dict[str, Any] | None = None) -> tuple[JanssensSegment, ...]:
    """Build segments with precomputed ``M_G`` intervals from mass bounds."""
    table = raw if raw is not None else load_janssens_table()
    out: list[JanssensSegment] = []
    for row in table["segments"]:
        a = float(row["a"])
        b = float(row["b"])
        m_low = float(row["m_low"])
        m_up = float(row["m_up"])
        out.append(
            JanssensSegment(
                m_low=m_low,
                m_up=m_up,
                a=a,
                b=b,
                a_err=float(row["a_err"]),
                b_err=float(row["b_err"]),
                mg_at_m_low=_mg_of_mass(m_low, a, b),
                mg_at_m_up=_mg_of_mass(m_up, a, b),
            )
        )
    return tuple(out)


def invert_mg_to_mass(
    mg: float,
    *,
    table: dict[str, Any] | None = None,
    segments: tuple[JanssensSegment, ...] | None = None,
    ab_correlation: float = 0.0,
) -> JanssensInversionResult:
    """Invert ``M = 10**((M_G - b) / a)`` with ``M_G``-interval segment selection.

    Extrapolation outside 0.02–57.95 Msun is forbidden: ``mass_msun`` is None
    with reason ``outside_janssens_range``.
    """
    del ab_correlation  # unpublished; default 0 is recorded in the table
    segs = segments if segments is not None else segments_from_table(table)
    table_raw = table if table is not None else load_janssens_table()
    inversion = table_raw["inversion"]
    tolerance = float(inversion["boundary_tolerance_mag"])
    tie_break: Literal["lower_mass_segment"] = inversion["boundary_tie_break"]
    if tie_break != "lower_mass_segment":
        raise ValueError(f"unsupported boundary_tie_break: {tie_break!r}")
    if table_raw.get("extrapolation") != "forbid":
        raise ValueError("Janssens table must keep extrapolation: forbid")

    if not np.isfinite(mg):
        return JanssensInversionResult(
            mass_msun=None,
            segment_index=None,
            boundary_resolved=False,
            uninformative_segment=False,
            reason="missing_mg",
        )

    hits: list[int] = []
    near: list[int] = []
    for i, seg in enumerate(segs):
        lo = seg.mg_min
        hi = seg.mg_max
        if lo <= mg <= hi:
            hits.append(i)
        elif lo - tolerance <= mg <= hi + tolerance:
            near.append(i)

    chosen: int | None = None
    boundary = False
    if hits:
        chosen = min(hits, key=lambda i: segs[i].m_low)
        boundary = len(hits) > 1
    elif near:
        chosen = min(near, key=lambda i: segs[i].m_low)
        boundary = True
    else:
        return JanssensInversionResult(
            mass_msun=None,
            segment_index=None,
            boundary_resolved=False,
            uninformative_segment=False,
            reason="outside_janssens_range",
        )

    seg = segs[chosen]
    mass = float(10.0 ** ((mg - seg.b) / seg.a))
    mass_lo, mass_hi = table_raw["mass_range_msun"]
    if mass < float(mass_lo) or mass > float(mass_hi):
        return JanssensInversionResult(
            mass_msun=None,
            segment_index=chosen,
            boundary_resolved=boundary,
            uninformative_segment=False,
            reason="outside_janssens_range",
        )
    uninformative = (
        abs(seg.m_low - UNINFORMATIVE_SEGMENT_MLOW) < 1e-9
        and abs(seg.m_up - UNINFORMATIVE_SEGMENT_MUP) < 1e-9
    )
    return JanssensInversionResult(
        mass_msun=mass,
        segment_index=chosen,
        boundary_resolved=boundary,
        uninformative_segment=uninformative,
    )


def invert_mg_to_mass_array(
    mg: ArrayLike,
    **kwargs: Any,
) -> NDArray[np.floating]:
    """Vector wrapper; out-of-range / missing → NaN."""
    values = np.asarray(mg, dtype=np.float64)
    table = kwargs.pop("table", None)
    segs = segments_from_table(table)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for idx, mag in np.ndenumerate(values):
        result = invert_mg_to_mass(float(mag), table=table, segments=segs, **kwargs)
        if result.mass_msun is not None:
            out[idx] = result.mass_msun
    return out
