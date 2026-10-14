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
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
COLUMNS: tuple[str, ...] = ("theta", "R", "Gmag1", "Gmag2", "IPDfmp1", "IPDgofha1")


def load_elbadry2021(path: str | Path, sha256: str) -> dict[str, FloatArray]:
    """Column-load the snapshot CSV after verifying its SHA256. θ is returned in arcsec."""
    import pandas as pd

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise ValueError(f"{p}: SHA256 {h.hexdigest()} != configured {sha256}")
    d = pd.read_csv(p, usecols=list(COLUMNS))
    out = {c: d[c].to_numpy(float) for c in COLUMNS}
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
               below_envelope: Literal["none", "hold"], hold_min_arcsec: float) -> FloatArray:
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
    if below_envelope == "hold":
        hv = np.where(fk >= 0, tab.p_pass[np.clip(fk, 0, te.size - 2), j], 1.0)
        out = np.where(below & (th >= hold_min_arcsec), hv, out)
    return out
