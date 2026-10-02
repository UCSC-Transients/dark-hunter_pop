"""Moe & Di Stefano (2017) companion densities (docs/MOCK_POPULATION_SPEC.md §2, #391).

Vectorized evaluations of the MdS17 analytic fits for luminous (zero-age MS) companions:
companion frequency per decade of period (Eqs. 20-23), the broken power-law mass-ratio
distribution with its twin excess (Eqs. 2, 5-7, 9-11, 13-15) and the eccentricity
distribution with e_max and circularization (Eqs. 3, 17, 18).

Coefficients are never hardcoded here: they come from a frozen table
(``config/population/moe_distefano2017.yaml``) loaded by :func:`load_mds17_table`, so a
rung-3 fit can swap in a different table. ``population_model`` owns how these densities are
combined with the compact-object mixture; this module only evaluates them.

Choices MdS17 does not settle (spec §8) are explicit arguments, never silent defaults:

- ``m1_interpolation`` (MP-Q10): how anchor equations are interpolated in M1.
- ``eta_floor`` (MP-Q11): the lower clip applied to eta so ``e^eta`` stays normalizable.
- Primaries below the MdS17 domain (MP-Q7) are the caller's responsibility
  (:func:`low_mass_frequency_scale` implements one candidate policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.config_loader import repo_root

DEFAULT_MDS17_TABLE: str = "config/population/moe_distefano2017.yaml"

M1Interpolation = Literal["linear_m1", "linear_log_m1"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiecewiseSegment:
    """``value + slope * (log P - pivot)`` on ``[logp_lo, logp_hi)``."""

    logp_lo: float
    logp_hi: float
    value: float
    slope: float
    pivot: float


@dataclass(frozen=True)
class MassAnchor:
    """One anchor equation (piecewise in log P) at an anchor primary mass."""

    m1_msun: float
    segments: tuple[PiecewiseSegment, ...]


@dataclass(frozen=True)
class EtaAnchor:
    """``eta = a - b / (log P - c)`` at an anchor primary mass (Eqs. 17, 18)."""

    m1_msun: float
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class MdS17Table:
    """Parsed MdS17 coefficient table (see ``config/population/moe_distefano2017.yaml``)."""

    m1_range: tuple[float, float]
    q_range: tuple[float, float]
    log_p_range: tuple[float, float]
    q_break: float
    twin_q_min: float
    circular_period_days: float
    f_logp_lt1: tuple[float, float, float]
    f_logp_2p7: tuple[float, float, float]
    f_logp_5p5: tuple[float, float, float]
    alpha: float
    delta_logp: float
    wide_decay: float
    gamma_largeq: tuple[MassAnchor, ...]
    gamma_smallq: tuple[MassAnchor, ...]
    twin_short: tuple[float, float]
    log_p_twin_intercept: float
    log_p_twin_m1_break: float
    log_p_twin_massive: float
    log_p_twin_short_edge: float
    e_max_period_scale_days: float
    e_max_exponent: float
    eta_anchors: tuple[EtaAnchor, ...]
    source_path: str


def _anchors(raw: list[dict[str, Any]]) -> tuple[MassAnchor, ...]:
    out = []
    for entry in raw:
        segs = tuple(PiecewiseSegment(**{k: float(v) for k, v in s.items()}) for s in entry["segments"])
        out.append(MassAnchor(m1_msun=float(entry["m1_msun"]), segments=segs))
    out.sort(key=lambda a: a.m1_msun)
    return tuple(out)


@lru_cache(maxsize=8)
def _load_cached(path: str) -> MdS17Table:
    raw = yaml.safe_load(Path(path).read_text())
    dom = raw["domain"]
    cf = raw["companion_frequency"]
    tw = raw["twin"]
    ec = raw["eccentricity"]
    eta = tuple(
        sorted(
            (EtaAnchor(**{k: float(v) for k, v in e.items()}) for e in ec["eta_anchors"]),
            key=lambda a: a.m1_msun,
        )
    )
    return MdS17Table(
        m1_range=(float(dom["m1_msun"][0]), float(dom["m1_msun"][1])),
        q_range=(float(dom["q"][0]), float(dom["q"][1])),
        log_p_range=(float(dom["log_p"][0]), float(dom["log_p"][1])),
        q_break=float(dom["q_break"]),
        twin_q_min=float(dom["twin_q_min"]),
        circular_period_days=float(dom["circular_period_days"]),
        f_logp_lt1=tuple(float(c) for c in cf["f_logp_lt1"]),  # type: ignore[arg-type]
        f_logp_2p7=tuple(float(c) for c in cf["f_logp_2p7"]),  # type: ignore[arg-type]
        f_logp_5p5=tuple(float(c) for c in cf["f_logp_5p5"]),  # type: ignore[arg-type]
        alpha=float(cf["alpha"]),
        delta_logp=float(cf["delta_logp"]),
        wide_decay=float(cf["wide_decay"]),
        gamma_largeq=_anchors(raw["gamma_largeq"]),
        gamma_smallq=_anchors(raw["gamma_smallq"]),
        twin_short=(float(tw["short_period"][0]), float(tw["short_period"][1])),
        log_p_twin_intercept=float(tw["log_p_twin_intercept"]),
        log_p_twin_m1_break=float(tw["log_p_twin_m1_break"]),
        log_p_twin_massive=float(tw["log_p_twin_massive"]),
        log_p_twin_short_edge=float(tw["log_p_short_edge"]),
        e_max_period_scale_days=float(ec["e_max_period_scale_days"]),
        e_max_exponent=float(ec["e_max_exponent"]),
        eta_anchors=eta,
        source_path=path,
    )


def load_mds17_table(path: str | Path | None = None) -> MdS17Table:
    """Load (and cache) the MdS17 coefficient table; relative paths resolve from the repo root."""
    p = Path(path) if path is not None else Path(DEFAULT_MDS17_TABLE)
    if not p.is_absolute():
        p = repo_root() / p
    return _load_cached(str(p.resolve()))


def _poly(coeffs: tuple[float, float, float], x: FloatArray) -> FloatArray:
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x


def f_logp_q03(m1_msun: ArrayLike, log_p: ArrayLike, table: MdS17Table) -> FloatArray:
    """Companions with q > 0.3 per decade of P, ``f_logP;q>0.3(M1, P)`` (Eq. 23).

    Zero outside ``table.log_p_range``. M1 is used as given (no domain clamp); the
    caller decides what happens outside 0.8-40 Msun (MP-Q7).
    """
    m1 = np.asarray(m1_msun, dtype=np.float64)
    lp = np.asarray(log_p, dtype=np.float64)
    x = np.log10(m1)
    f1 = _poly(table.f_logp_lt1, x)
    f27 = _poly(table.f_logp_2p7, x)
    f55 = _poly(table.f_logp_5p5, x)
    a, d = table.alpha, table.delta_logp
    lo, hi = table.log_p_range
    conds = [
        (lp >= lo) & (lp < 1.0),
        (lp >= 1.0) & (lp < 2.7 - d),
        (lp >= 2.7 - d) & (lp < 2.7 + d),
        (lp >= 2.7 + d) & (lp < 5.5),
        (lp >= 5.5) & (lp < hi),
    ]
    vals = [
        f1,
        f1 + (lp - 1.0) / (1.7 - d) * (f27 - f1 - a * d),
        f27 + a * (lp - 2.7),
        f27 + a * d + (lp - 2.7 - d) / (2.8 - d) * (f55 - f27 - a * d),
        f55 * np.exp(-table.wide_decay * (lp - 5.5)),
    ]
    return np.asarray(np.select(conds, vals, default=0.0), dtype=np.float64)


def _eval_anchor(anchor: MassAnchor, lp: FloatArray) -> FloatArray:
    out = np.full(lp.shape, np.nan)
    for seg in anchor.segments:
        sel = (lp >= seg.logp_lo) & (lp < seg.logp_hi)
        out = np.where(sel, seg.value + seg.slope * (lp - seg.pivot), out)
    # Close the top edge (log P == upper bound of the last segment).
    last = anchor.segments[-1]
    out = np.where(lp == last.logp_hi, last.value + last.slope * (lp - last.pivot), out)
    return out


def _interp_anchors(
    anchors: tuple[MassAnchor, ...],
    m1: FloatArray,
    lp: FloatArray,
    m1_interpolation: M1Interpolation,
) -> FloatArray:
    """Interpolate anchor equations in M1, clamped to the end anchors."""
    vals = np.stack([_eval_anchor(a, lp) for a in anchors], axis=0)
    masses = np.array([a.m1_msun for a in anchors])
    if m1_interpolation == "linear_m1":
        xm, xa = m1, masses
    elif m1_interpolation == "linear_log_m1":
        xm, xa = np.log10(m1), np.log10(masses)
    else:
        raise ValueError(f"unknown m1_interpolation {m1_interpolation!r}")
    xm = np.clip(xm, xa[0], xa[-1])
    j = np.clip(np.searchsorted(xa, xm, side="right") - 1, 0, len(xa) - 2)
    t = (xm - xa[j]) / (xa[j + 1] - xa[j])
    lo = np.take_along_axis(vals, j[np.newaxis, ...], axis=0)[0]
    hi = np.take_along_axis(vals, (j + 1)[np.newaxis, ...], axis=0)[0]
    return (1.0 - t) * lo + t * hi


def gamma_largeq(
    m1_msun: ArrayLike, log_p: ArrayLike, table: MdS17Table, *, m1_interpolation: M1Interpolation
) -> FloatArray:
    """Mass-ratio slope across 0.3 < q < 1 (Eqs. 9-11). NaN outside the tabulated log P."""
    m1, lp = np.broadcast_arrays(np.asarray(m1_msun, float), np.asarray(log_p, float))
    return _interp_anchors(table.gamma_largeq, m1, lp, m1_interpolation)


def gamma_smallq(
    m1_msun: ArrayLike, log_p: ArrayLike, table: MdS17Table, *, m1_interpolation: M1Interpolation
) -> FloatArray:
    """Mass-ratio slope across 0.1 < q < 0.3 (Eqs. 13-15). NaN outside the tabulated log P."""
    m1, lp = np.broadcast_arrays(np.asarray(m1_msun, float), np.asarray(log_p, float))
    return _interp_anchors(table.gamma_smallq, m1, lp, m1_interpolation)


def f_twin(m1_msun: ArrayLike, log_p: ArrayLike, table: MdS17Table) -> FloatArray:
    """Excess twin fraction F_twin (Eqs. 5-7), relative to companions with q > 0.3."""
    m1, lp = np.broadcast_arrays(np.asarray(m1_msun, float), np.asarray(log_p, float))
    short = table.twin_short[0] + table.twin_short[1] * np.log10(m1)
    lp_twin = np.where(
        m1 <= table.log_p_twin_m1_break,
        table.log_p_twin_intercept - m1,
        table.log_p_twin_massive,
    )
    edge = table.log_p_twin_short_edge
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = short * (1.0 - (lp - edge) / (lp_twin - edge))
    out = np.where(lp < edge, short, np.where(lp < lp_twin, mid, 0.0))
    return np.clip(out, 0.0, 1.0)


def _power_integral(gamma: FloatArray, lo: float, hi: float) -> FloatArray:
    """∫_lo^hi q^gamma dq, with the gamma = -1 limit handled."""
    g1 = gamma + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gen = (hi**g1 - lo**g1) / g1
    return np.where(np.abs(g1) < 1e-12, np.log(hi / lo), gen)


def q_density(
    q: ArrayLike,
    m1_msun: ArrayLike,
    log_p: ArrayLike,
    table: MdS17Table,
    *,
    m1_interpolation: M1Interpolation,
) -> FloatArray:
    """Mass-ratio density per unit q, normalized so ∫_{0.3}^{1} = 1 (Eq. 2, Fig. 2).

    Multiply by :func:`f_logp_q03` to get companions per decade of P per unit q. On
    ``0.1 <= q < 0.3`` the ``gamma_smallq`` power law joins the ``gamma_largeq`` power-law
    component continuously at ``q = 0.3``; the twin excess ``F_twin`` is uniform on
    ``0.95 <= q <= 1``. Zero outside ``table.q_range``.
    """
    q_arr, m1, lp = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(m1_msun, float), np.asarray(log_p, float)
    )
    gl = gamma_largeq(m1, lp, table, m1_interpolation=m1_interpolation)
    gs = gamma_smallq(m1, lp, table, m1_interpolation=m1_interpolation)
    ft = f_twin(m1, lp, table)
    qb, qt = table.q_break, table.twin_q_min
    qlo, qhi = table.q_range
    norm = _power_integral(gl, qb, qhi)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        large = (1.0 - ft) * q_arr**gl / norm + ft * (q_arr >= qt) / (qhi - qt)
        small = (1.0 - ft) * qb ** (gl - gs) * q_arr**gs / norm
    out = np.where((q_arr >= qb) & (q_arr <= qhi), large, 0.0)
    out = np.where((q_arr >= qlo) & (q_arr < qb), small, out)
    return np.where(np.isfinite(out), out, 0.0)


def e_max(period_days: ArrayLike, table: MdS17Table) -> FloatArray:
    """Maximum eccentricity without Roche-lobe filling (Eq. 3); 0 for P <= 2 d."""
    p = np.asarray(period_days, dtype=np.float64)
    scale = table.e_max_period_scale_days
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 1.0 - (p / scale) ** table.e_max_exponent
    return np.where(p > table.circular_period_days, np.clip(val, 0.0, 1.0), 0.0)


def eta(
    m1_msun: ArrayLike,
    log_p: ArrayLike,
    table: MdS17Table,
    *,
    m1_interpolation: M1Interpolation,
    eta_floor: float,
) -> FloatArray:
    """Eccentricity power-law index (Eqs. 17, 18), interpolated in M1 between anchors.

    ``eta_floor`` (MP-Q11) clips eta from below; it must be > -1 so ``e^eta`` is
    normalizable on ``[0, e_max)``. The formulas are evaluated as written outside their
    fitted log P ranges (also MP-Q11).
    """
    if not eta_floor > -1.0:
        raise ValueError(f"eta_floor must be > -1, got {eta_floor}")
    m1, lp = np.broadcast_arrays(np.asarray(m1_msun, float), np.asarray(log_p, float))
    anchors = table.eta_anchors
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.stack([a.a - a.b / (lp - a.c) for a in anchors], axis=0)
    masses = np.array([a.m1_msun for a in anchors])
    if m1_interpolation == "linear_m1":
        xm, xa = m1, masses
    else:
        xm, xa = np.log10(m1), np.log10(masses)
    t = np.clip((xm - xa[0]) / (xa[-1] - xa[0]), 0.0, 1.0)
    out = (1.0 - t) * vals[0] + t * vals[-1]
    out = np.where(lp > anchors[0].c, out, eta_floor)
    return np.maximum(np.where(np.isfinite(out), out, eta_floor), eta_floor)


def e_density(
    ecc: ArrayLike,
    m1_msun: ArrayLike,
    period_days: ArrayLike,
    table: MdS17Table,
    *,
    m1_interpolation: M1Interpolation,
    eta_floor: float,
) -> FloatArray:
    """Eccentricity density per unit e on ``[0, e_max(P))`` for P > 2 d; zero elsewhere.

    For P <= 2 d MdS17 binaries are circular (a point mass at e = 0); this function
    returns 0 there and :func:`is_circular` flags those draws.
    """
    e_arr, m1, p = np.broadcast_arrays(
        np.asarray(ecc, float), np.asarray(m1_msun, float), np.asarray(period_days, float)
    )
    emax = e_max(p, table)
    et = eta(m1, np.log10(p), table, m1_interpolation=m1_interpolation, eta_floor=eta_floor)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dens = (et + 1.0) * e_arr**et / emax ** (et + 1.0)
    ok = (p > table.circular_period_days) & (e_arr >= 0.0) & (e_arr < emax) & (emax > 0)
    return np.where(ok & np.isfinite(dens), dens, 0.0)


def is_circular(period_days: ArrayLike, table: MdS17Table) -> NDArray[np.bool_]:
    """True where MdS17 sets e = 0 exactly (P <= 2 d)."""
    return np.asarray(period_days, float) <= table.circular_period_days


def low_mass_frequency_scale(
    m1_msun: ArrayLike, *, m1_anchor_msun: float, m1_zero_msun: float
) -> FloatArray:
    """One candidate MP-Q7 policy: frequency falls linearly in log M1 below the anchor.

    1 at and above ``m1_anchor_msun``, 0 at and below ``m1_zero_msun``. This is
    El-Badry et al. (2024) §3's binary-fraction prescription (0.8 -> 0.08 Msun) and is
    used only as a pilot provisional setting, not a decision.
    """
    m1 = np.asarray(m1_msun, float)
    t = (np.log10(m1) - np.log10(m1_zero_msun)) / (
        np.log10(m1_anchor_msun) - np.log10(m1_zero_msun)
    )
    return np.clip(t, 0.0, 1.0)
