"""Forward model for the observed Gaia DR3 ``vbroad`` of a mock system (#391 spec §12.31; DRAFT, uncalibrated).

DR3 (Frémat et al. 2023, A&A 674, A8, §2.1–2.2): vbroad is fitted per transit at the transit's own RV (no SB1
orbital smearing), the published value is the median over ≥ 6 transits, SB2-flagged sources get none, and
the published range is 5 < vbroad < 500 km/s for G_RVS ≤ 12 and template Teff ≥ 3500 K.

Per system (statistical, never per star; reweight-level, from stored truth plus a seeded draw):

1. intrinsic v sin i of each component from :class:`RotationPrior` (Zorec & Royer 2012 for M ≥ ``a_star_min_msun``
   main-sequence stars, by mass and fractional MS age; an empirical sample, e.g. DR3 single-star vbroad, for cooler
   stars; a slow-rotator distribution for evolved stars);
2. per transit (``n_transits`` random orbital phases), the line profile is the RVS-flux-weighted pair of
   rotational profiles at the components' RVs; its width is matched by second moment (rotational profile with no
   limb darkening: variance v²/4), giving a per-transit vbroad; a transit whose components are separated by more
   than ``sb2_separation_factor`` × (half-widths + instrumental) with RVS flux ratio ≥ ``sb2_min_flux_ratio`` is
   SB2-like; a source with more than half its transits SB2-like is flagged (vbroad = NaN);
3. DR3 response: multiplicative bias for hot stars (Frémat §4: medians 8–28 % high from F0 to A0) and the §2.2
   filters.

Every number is a parameter (:class:`VbroadModelConfig`), to be calibrated statistically (§12.31).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VbroadModelConfig:
    n_transits: int = 12
    instrumental_kms: float = 13.0  # ~ c / (2 R), R ≈ 11 500 (Frémat §2.2)
    sb2_separation_factor: float = 1.0
    sb2_min_flux_ratio: float = 0.2
    hot_bias_teff_k: tuple[float, float] = (7200.0, 9600.0)  # F0 -> A0
    hot_bias: tuple[float, float] = (1.08, 1.28)
    rel_scatter: float = 0.10
    vmin_kms: float = 5.0
    vmax_kms: float = 500.0
    grvs_max: float = 12.0
    teff_min_k: float = 3500.0


class RotationPrior:
    """Empirical v sin i prior: Zorec & Royer (2012) for A/late-B MS stars (mass, fractional age), a supplied
    empirical sample for cooler stars, and a slow-rotator law for evolved stars."""

    def __init__(self, zr_mass: ArrayLike, zr_age: ArrayLike, zr_vsini: ArrayLike, zr_class: ArrayLike | None = None, *,
                 cool_vsini: ArrayLike | None = None, a_star_min_msun: float = 1.6, evolved_range_kms: tuple[float, float] = (1.0, 8.0),
                 mass_edges: ArrayLike = (1.6, 2.0, 2.5, 3.0, 4.0, 10.0), age_edges: ArrayLike = (0.0, 0.5, 1.01),
                 exclude_classes: tuple[str, ...] = ()) -> None:
        m, a, v = (np.asarray(x, float) for x in (zr_mass, zr_age, zr_vsini))
        keep = np.isfinite(m) & np.isfinite(a) & np.isfinite(v)
        if zr_class is not None and exclude_classes:
            c = np.char.strip(np.asarray(zr_class).astype(str))
            keep &= ~np.isin(c, exclude_classes)
        self.m, self.a, self.v = m[keep], a[keep], v[keep]
        self.mass_edges, self.age_edges = np.asarray(mass_edges, float), np.asarray(age_edges, float)
        self.cool = None if cool_vsini is None else np.asarray(cool_vsini, float)[np.isfinite(np.asarray(cool_vsini, float))]
        self.a_star_min = float(a_star_min_msun)
        self.evolved_range = evolved_range_kms

    def sample(self, mass: ArrayLike, frac_age: ArrayLike, evolved: ArrayLike, rng: np.random.Generator) -> FloatArray:
        m, fa, ev = np.asarray(mass, float), np.asarray(frac_age, float), np.asarray(evolved, bool)
        out = np.full(m.size, np.nan)
        hot = ~ev & (m >= self.a_star_min)
        ib = np.clip(np.digitize(m, self.mass_edges) - 1, 0, self.mass_edges.size - 2)
        ia = np.clip(np.digitize(np.nan_to_num(fa, nan=0.5), self.age_edges) - 1, 0, self.age_edges.size - 2)
        zb = np.clip(np.digitize(self.m, self.mass_edges) - 1, 0, self.mass_edges.size - 2)
        za = np.clip(np.digitize(self.a, self.age_edges) - 1, 0, self.age_edges.size - 2)
        for b in np.unique(ib[hot]):
            for g in np.unique(ia[hot & (ib == b)]):
                sel = np.flatnonzero(hot & (ib == b) & (ia == g))
                pool = self.v[(zb == b) & (za == g)]
                if pool.size < 10:
                    pool = self.v[zb == b]
                out[sel] = rng.choice(pool, size=sel.size)
        cool = ~ev & ~hot
        if cool.any():
            pool = self.cool if self.cool is not None and self.cool.size else np.array([5.0])
            out[cool] = rng.choice(pool, size=int(cool.sum()))
        lo, hi = self.evolved_range
        out[ev] = rng.uniform(lo, hi, size=int(ev.sum()))
        return out

    @classmethod
    def from_zorec_royer(cls, table1: str | Path, **kw: Any) -> RotationPrior:
        """Read VizieR J/A+A/537/A120 ``table1.dat`` (fixed width, ReadMe byte columns)."""
        mass, age, vs, klass = [], [], [], []
        for line in Path(table1).read_text().splitlines():
            if len(line) < 75:
                continue
            try:
                mass.append(float(line[50:54]))
                age.append(float(line[60:65]))
                vs.append(float(line[72:75]))
            except ValueError:
                continue
            klass.append(line[79:84].strip() if len(line) >= 80 else "")
        return cls(mass, age, vs, klass, **kw)


def k_semi_amplitudes(m1: ArrayLike, m2: ArrayLike, period_days: ArrayLike, ecc: ArrayLike, inc_deg: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """K1, K2 (km/s) from Kepler's third law."""
    from astropy import constants as const

    G, msun, day = const.G.cgs.value, const.M_sun.cgs.value, 86400.0
    m1, m2, p, e, i = (np.asarray(x, float) for x in (m1, m2, period_days, ecc, inc_deg))
    a = np.cbrt(G * (m1 + m2) * msun * (p * day) ** 2 / (4 * np.pi**2))  # cm
    vtot = 2 * np.pi * a / (p * day) * np.sin(np.radians(i)) / np.sqrt(1 - e**2) / 1e5
    return vtot * m2 / (m1 + m2), vtot * m1 / (m1 + m2)


def observed_vbroad(v1: ArrayLike, v2: ArrayLike, f_rvs: ArrayLike, k1: ArrayLike, k2: ArrayLike, ecc: ArrayLike,
                    omega_rad: ArrayLike, teff1: ArrayLike, grvs: ArrayLike, rng: np.random.Generator,
                    cfg: VbroadModelConfig = VbroadModelConfig()) -> FloatArray:
    """Per-system observed vbroad (NaN when DR3 would publish none). ``f_rvs`` = companion/primary RVS flux
    (0 for dark); ``v1``, ``v2`` intrinsic v sin i (km/s)."""
    v1, v2, f, k1, k2, e, w, t1, gr = (np.asarray(x, float) for x in (v1, v2, f_rvs, k1, k2, ecc, omega_rad, teff1, grvs))
    n, nt = v1.size, cfg.n_transits
    f = np.nan_to_num(f, nan=0.0)
    v2 = np.nan_to_num(v2, nan=0.0)
    # random orbital phases: uniform mean anomaly -> true anomaly (small-e accurate enough for a second-moment width)
    M = rng.uniform(0, 2 * np.pi, size=(n, nt))
    E = M.copy()
    for _ in range(8):
        E = E - (E - e[:, None] * np.sin(E) - M) / (1 - e[:, None] * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e[:, None]) * np.sin(E / 2), np.sqrt(1 - e[:, None]) * np.cos(E / 2))
    s = np.cos(nu + w[:, None]) + e[:, None] * np.cos(w[:, None])
    rv1, rv2 = k1[:, None] * s, -k2[:, None] * s
    w1, w2 = 1.0 / (1.0 + f), f / (1.0 + f)
    mean = w1[:, None] * rv1 + w2[:, None] * rv2
    var = (w1[:, None] * (v1[:, None] ** 2 / 4 + (rv1 - mean) ** 2) + w2[:, None] * (v2[:, None] ** 2 / 4 + (rv2 - mean) ** 2))
    v_t = 2.0 * np.sqrt(var)
    sep = np.abs(rv1 - rv2)
    sb2 = (f[:, None] >= cfg.sb2_min_flux_ratio) & (sep > cfg.sb2_separation_factor * (0.5 * (v1[:, None] + v2[:, None]) + 2 * cfg.instrumental_kms))
    v_t = v_t * (1 + cfg.rel_scatter * rng.standard_normal(v_t.shape))
    vb = np.median(v_t, axis=1)
    flagged = sb2.mean(axis=1) > 0.5
    lo, hi = cfg.hot_bias_teff_k
    b = np.interp(t1, [lo, hi], list(cfg.hot_bias), left=1.0, right=cfg.hot_bias[1])
    vb = vb * b
    ok = ~flagged & (vb > cfg.vmin_kms) & (vb < cfg.vmax_kms) & (gr <= cfg.grvs_max) & (t1 >= cfg.teff_min_k)
    return np.where(ok, vb, np.nan)
