"""MP-Q27: resolved and partially resolved companions (spec §12.23).

Statistical, calibrated on El-Badry, Rix & Heintz (2021) Gaia EDR3 wide binaries (pairs with
θ < 10″; the snapshot is read through its config key and SHA256-verified), never per star.

* Resolution envelope ``C(θ, ΔG)``: per ΔG bin, observed pairs per dlog θ divided by a power
  law fitted where the catalogue is complete (``ref_theta_arcsec``) and extrapolated inward.
  Diagnostic only: a resolved companion becomes a separate source, which changes the mock's
  photocentre (f → 0) at generation time; that part is not a reweight (spec §12.23).
* IPD failure ``P_fail(θ, ΔG)``: fraction of pair primaries failing the Halbwachs IPD cuts
  (``ipd_frac_multi_peak > max`` or ``ipd_gof_harmonic_amplitude ≥ max``) per (θ, ΔG) cell,
  relative to the isolated-star baseline at ``ref_theta_arcsec``:
  ``P_pass = (1 − P_fail) / (1 − P_fail,ref)``, clipped to [0, 1]. This is the reweight.
* Below the smallest calibrated θ the catalogue says nothing (the pair is unresolved).
  ``below_envelope`` is an explicit choice: ``none`` (P_pass = 1) or ``hold`` (the innermost
  calibrated value, down to ``hold_min_arcsec``; P_pass = 1 below that).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
COLUMNS: tuple[str, ...] = ("theta", "R", "Gmag1", "Gmag2", "IPDfmp1", "IPDgofha1")


def load_elbadry2021(path: str | Path, sha256: str) -> dict[str, FloatArray]:
    """Column-load the snapshot CSV after verifying its SHA256. θ is returned in arcsec."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise ValueError(f"{p}: SHA256 {h.hexdigest()} != configured {sha256}")
    with p.open() as fh:
        head = fh.readline().strip().split(",")
    idx = [head.index(c) for c in COLUMNS]
    a = np.loadtxt(p, delimiter=",", skiprows=1, usecols=idx, ndmin=2)
    out = {c: a[:, k].astype(float) for k, c in enumerate(COLUMNS)}
    out["theta"] = out["theta"] * 3600.0
    return out


@dataclass(frozen=True)
class ResolutionTables:
    theta_edges: FloatArray
    dg_edges: FloatArray
    p_pass: FloatArray  # (n_theta, n_dg), NaN where uncalibrated
    completeness: FloatArray  # (n_theta, n_dg)
    n_pairs: NDArray[np.int64]


def build_tables(cat: dict[str, FloatArray], theta_edges: ArrayLike, dg_edges: ArrayLike, *, r_chance_max: float,
                 g2_max: float, ref_theta_arcsec: tuple[float, float], ipd_fmp_max: float, ipd_gof_max: float,
                 min_pairs: int) -> ResolutionTables:
    te = np.asarray(theta_edges, float)
    de = np.asarray(dg_edges, float)
    th, dg = cat["theta"], cat["Gmag2"] - cat["Gmag1"]
    ok = (cat["R"] < r_chance_max) & (cat["Gmag2"] < g2_max) & np.isfinite(dg)
    fail = ~((cat["IPDfmp1"] <= ipd_fmp_max) & (cat["IPDgofha1"] < ipd_gof_max))
    nt, nd = te.size - 1, de.size - 1
    pp, cc = np.full((nt, nd), np.nan), np.full((nt, nd), np.nan)
    nn = np.zeros((nt, nd), np.int64)
    lc = np.log10(np.sqrt(te[1:] * te[:-1]))
    dl = np.diff(np.log10(te))
    ref = (np.sqrt(te[1:] * te[:-1]) >= ref_theta_arcsec[0]) & (np.sqrt(te[1:] * te[:-1]) <= ref_theta_arcsec[1])
    for j in range(nd):
        s = ok & (dg >= de[j]) & (dg < de[j + 1])
        n = np.histogram(th[s], te)[0]
        nf = np.histogram(th[s & fail], te)[0]
        nn[:, j] = n
        dens = n / dl
        good = ref & (n >= min_pairs)
        if good.sum() >= 2:
            b, a = np.polyfit(lc[good], np.log10(dens[good]), 1)
            cc[:, j] = np.clip(dens / 10.0 ** (a + b * lc), 0.0, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(n >= min_pairs, nf / n, np.nan)
        f_ref = np.nansum(nf[ref]) / max(np.sum(n[ref]), 1)
        pp[:, j] = np.clip((1.0 - f) / (1.0 - f_ref), 0.0, 1.0)
    return ResolutionTables(theta_edges=te, dg_edges=de, p_pass=pp, completeness=cc, n_pairs=nn)


def p_pass_ipd(theta_arcsec: ArrayLike, delta_g: ArrayLike, tab: ResolutionTables, *,
               below_envelope: Literal["none", "hold", "orb6"], hold_min_arcsec: float,
               orb6: "Orb6PassTable | None" = None) -> FloatArray:
    """Per draw P(pass Halbwachs IPD cuts | companion at θ, ΔG). Non-finite ΔG (dark) → 1;
    θ beyond the table → 1; below the innermost calibrated cell per ``below_envelope``."""
    th = np.asarray(theta_arcsec, float)
    dg = np.asarray(delta_g, float)
    out = np.ones(th.size)
    te, de = tab.theta_edges, tab.dg_edges
    j = np.clip(np.searchsorted(de, dg, side="right") - 1, 0, de.size - 2)
    lit = np.isfinite(dg)
    i = np.searchsorted(te, th, side="right") - 1
    inside = lit & (i >= 0) & (i < te.size - 1)
    v = np.where(inside, tab.p_pass[np.clip(i, 0, te.size - 2), j], np.nan)
    # innermost calibrated cell per ΔG column
    first = np.array([np.flatnonzero(np.isfinite(tab.p_pass[:, k]))[0] if np.isfinite(tab.p_pass[:, k]).any() else -1
                      for k in range(de.size - 1)])
    fk = first[j]
    inner_edge = np.where(fk >= 0, te[np.clip(fk, 0, te.size - 1)], np.inf)
    below = lit & (th < inner_edge)
    out = np.where(inside & np.isfinite(v) & ~below, v, out)
    if below_envelope == "orb6":
        if orb6 is None:
            raise ValueError("below_envelope orb6 needs the ORB6 pass table")
        io = np.clip(np.searchsorted(orb6.theta_edges, th, side="right") - 1, 0, orb6.theta_edges.size - 2)
        jo = np.clip(np.searchsorted(orb6.dg_edges, dg, side="right") - 1, 0, orb6.dg_edges.size - 2)
        in_o = below & (th >= orb6.theta_edges[0])
        out = np.where(in_o, orb6.p_pass[io, jo], out)
    if below_envelope == "hold":
        hv = np.where(fk >= 0, tab.p_pass[np.clip(fk, 0, te.size - 2), j], 1.0)
        out = np.where(below & (th >= hold_min_arcsec), hv, out)
    return out


# ---------------------------------------------------------------------------
# ORB6 calibration of the partially resolved regime (spec §12.23 amendment)
# ---------------------------------------------------------------------------
#: ORB6 fixed-width slices (0-based, end-exclusive) from the file's own format line.
ORB6_SLICES: dict[str, tuple[int, int]] = {
    "v1": (66, 72), "v2": (73, 79), "p": (81, 92), "p_unit": (92, 93), "a": (105, 114), "a_unit": (114, 115),
    "i": (125, 133), "node": (143, 151), "t": (162, 174), "t_unit": (174, 175), "e": (187, 195), "omega": (205, 213),
    "grade": (233, 234),
}
_P_UNIT_YR = {"y": 1.0, "d": 1.0 / 365.25, "c": 100.0, "m": 1.0 / (365.25 * 1440.0), "h": 1.0 / (365.25 * 24.0)}
_A_UNIT_AS = {"a": 1.0, "m": 1e-3, "u": 1e-6}


def parse_orb6(lines: list[str]) -> dict[str, NDArray[np.float64]]:
    """ORB6 orbit rows → P [yr], a [″], i, Ω, ω [deg], T [yr, Julian], e, V1, V2, grade (NaN when
    missing or in an unknown unit). T in ``d`` is JD − 2400000 (or a full JD), ``m`` MJD."""
    out: dict[str, list[float]] = {k: [] for k in ("p_yr", "a_as", "i", "node", "omega", "t_yr", "e", "v1", "v2", "grade")}

    def num(s: str) -> float:
        try:
            return float(s.strip().rstrip("."))
        except ValueError:
            return float("nan")

    for ln in lines:
        ln = ln.ljust(240)
        f = {k: ln[a:b] for k, (a, b) in ORB6_SLICES.items()}
        p = num(f["p"]) * _P_UNIT_YR.get(f["p_unit"].strip(), np.nan)
        a = num(f["a"]) * _A_UNIT_AS.get(f["a_unit"].strip(), np.nan)
        t = num(f["t"])
        tu = f["t_unit"].strip()
        if tu == "y":
            ty = t
        elif tu in ("d", "m"):
            jd = t + (2400000.5 if tu == "m" else (2400000.0 if t < 1e6 else 0.0))
            ty = 2000.0 + (jd - 2451545.0) / 365.25
        else:
            ty = float("nan")
        for k, v in (("p_yr", p), ("a_as", a), ("i", num(f["i"])), ("node", num(f["node"])), ("omega", num(f["omega"])),
                     ("t_yr", ty), ("e", num(f["e"])), ("v1", num(f["v1"])), ("v2", num(f["v2"])), ("grade", num(f["grade"]))):
            out[k].append(v)
    return {k: np.asarray(v, float) for k, v in out.items()}


def orb6_separation(orb: dict[str, NDArray[np.float64]], epoch_yr: float, solve_kepler: Any) -> FloatArray:
    """Apparent separation ρ [″] at ``epoch_yr`` from Campbell elements; ``solve_kepler(M, e)`` is
    gaiamock's Kepler solver (never reimplemented). NaN where elements are missing."""
    out = np.full(orb["p_yr"].size, np.nan)
    ok = np.all([np.isfinite(orb[k]) for k in ("p_yr", "a_as", "i", "omega", "t_yr", "e")], axis=0) & (orb["e"] < 1)
    ok &= (orb["p_yr"] > 0) & (orb["a_as"] > 0)  # e.g. ORB6 rows with P = 0: excluded (NaN, never enters the tables)
    for j in np.flatnonzero(ok):
        M = np.array([2 * np.pi * ((epoch_yr - orb["t_yr"][j]) / orb["p_yr"][j] % 1.0)])
        E = float(np.asarray(solve_kepler(M, float(orb["e"][j])))[0])
        e = orb["e"][j]
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = orb["a_as"][j] * (1 - e * np.cos(E))
        u = nu + np.radians(orb["omega"][j])
        out[j] = r * np.sqrt(np.cos(u) ** 2 + (np.sin(u) * np.cos(np.radians(orb["i"][j]))) ** 2)
    return out


def binned_fraction(theta: ArrayLike, dg: ArrayLike, flag: ArrayLike, weight: ArrayLike, theta_edges: ArrayLike,
                    dg_edges: ArrayLike, *, min_weight: float) -> tuple[FloatArray, FloatArray]:
    """Weighted fraction of ``flag`` per (θ, ΔG) cell (NaN below ``min_weight``) and the cell weights."""
    th, d, f, w = (np.asarray(x, float) for x in (theta, dg, flag, weight))
    te, de = np.asarray(theta_edges, float), np.asarray(dg_edges, float)
    ok = np.isfinite(th) & np.isfinite(d) & (w > 0)
    W = np.histogram2d(th[ok], d[ok], [te, de], weights=w[ok])[0]
    Wf = np.histogram2d(th[ok], d[ok], [te, de], weights=(w * f)[ok])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(W >= min_weight, Wf / W, np.nan), W


@dataclass(frozen=True)
class Orb6PassTable:
    theta_edges: FloatArray
    dg_edges: FloatArray
    p_pass: FloatArray  # (n_theta, n_dg), normalized and filled
    weight: FloatArray


def orb6_pass_table(theta: ArrayLike, dg: ArrayLike, passed: ArrayLike, weight: ArrayLike, theta_edges: ArrayLike,
                    dg_edges: ArrayLike, *, min_weight: float, ref_theta_max: float, fill_dg_max: float) -> Orb6PassTable:
    """IPD pass vs (θ, ΔG) for the partially resolved regime from ORB6 pairs, normalized by the
    pass rate of the same ΔG column at θ < ``ref_theta_max`` (pairs too close to affect the IPD:
    removes ORB6's bright-star baseline), clipped to [0, 1]. Empty cells: ΔG columns below
    ``fill_dg_max`` take the nearest populated lower-ΔG column at the same θ (explicit choice,
    spec §12.23); columns above it, and anything still empty, get 1."""
    fr, W = binned_fraction(theta, dg, passed, weight, theta_edges, dg_edges, min_weight=min_weight)
    te, de = np.asarray(theta_edges, float), np.asarray(dg_edges, float)
    ref_rows = 0.5 * (te[1:] + te[:-1]) < ref_theta_max
    pp = np.full_like(fr, np.nan)
    for j in range(de.size - 1):
        ref = fr[ref_rows, j]
        ref = ref[np.isfinite(ref)]
        if ref.size:
            pp[:, j] = np.clip(fr[:, j] / max(float(np.mean(ref)), 1e-6), 0.0, 1.0)
    for j in range(de.size - 1):
        if de[j] >= fill_dg_max:
            pp[:, j] = 1.0
            continue
        for i in range(te.size - 1):
            if not np.isfinite(pp[i, j]):
                lower = [k for k in range(j - 1, -1, -1) if np.isfinite(pp[i, k])]
                pp[i, j] = pp[i, lower[0]] if lower else 1.0
    return Orb6PassTable(theta_edges=te, dg_edges=de, p_pass=pp, weight=W)


def orb6_pairs(orb6_row: ArrayLike, ra: ArrayLike, dec: ArrayLike, rho: ArrayLike, dv: ArrayLike, xm: dict[str, ArrayLike],
               *, ipd_fmp_max: float, ipd_gof_max: float, match_min_arcsec: float, sep_tol_frac: float,
               sep_tol_min_arcsec: float) -> dict[str, FloatArray]:
    """Per ORB6 system: the DR3 primary (brightest source whose J2000-propagated position lies within
    max(``match_min_arcsec``, ρ + 1″) of the ORB6 position), whether the companion is a separate
    DR3 source (another source at |sep(2016) − ρ| < max(``sep_tol_min_arcsec``, ``sep_tol_frac`` ρ)),
    ΔG (DR3 pair if resolved, else V2 − V1) and whether the primary passes the Halbwachs IPD cuts.
    ``xm``: XMatch columns orb6_row, RAJ2000, DEJ2000, RAdeg, DEdeg, Gmag, IPDfmp, IPDgofha."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rows = np.asarray(orb6_row, np.int64)
    n = rows.size
    out = {k: np.full(n, np.nan) for k in ("theta", "dg", "passed", "resolved")}
    xr = np.asarray(xm["orb6_row"], np.int64)
    G = np.asarray(xm["Gmag"], float)
    fmp, gof = np.asarray(xm["IPDfmp"], float), np.asarray(xm["IPDgofha"], float)
    c2000 = SkyCoord(np.asarray(xm["RAJ2000"], float) * u.deg, np.asarray(xm["DEJ2000"], float) * u.deg)
    c2016 = SkyCoord(np.asarray(xm["RAdeg"], float) * u.deg, np.asarray(xm["DEdeg"], float) * u.deg)
    pos = {int(r): i for i, r in enumerate(rows)}
    orb_c = SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg)
    rho_, dv_ = np.asarray(rho, float), np.asarray(dv, float)
    for r in np.unique(xr):
        i = pos.get(int(r))
        if i is None or not np.isfinite(rho_[i]):
            continue
        s = np.flatnonzero((xr == r) & np.isfinite(G))
        if s.size == 0:
            continue
        d0 = orb_c[i].separation(c2000[s]).arcsec
        cand = s[d0 < max(match_min_arcsec, rho_[i] + 1.0)]
        if cand.size == 0:
            continue
        k = cand[np.argmin(G[cand])]
        oth = s[s != k]
        sep = c2016[k].separation(c2016[oth]).arcsec if oth.size else np.array([])
        m = np.abs(sep - rho_[i]) < max(sep_tol_min_arcsec, sep_tol_frac * rho_[i])
        out["resolved"][i] = float(m.any())
        out["dg"][i] = float(G[oth[m]].min() - G[k]) if m.any() else dv_[i]
        out["theta"][i] = rho_[i]
        out["passed"][i] = float((fmp[k] <= ipd_fmp_max) & (gof[k] < ipd_gof_max))
    return out
