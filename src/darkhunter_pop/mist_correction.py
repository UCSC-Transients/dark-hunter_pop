"""MIST low-mass main-sequence correction (spec §12.24), applied to the native grid.

For MS points (MIST phase 0) with current mass M below ``join_msun``:

* Teff at fixed bolometric luminosity: log Teff += Δ(M); log R −= 2Δ; log g += 4Δ; the Gaia
  G/BP/RP magnitudes are recomputed from MIST's own UBVRIplus BC tables at the new (Teff,
  log g, [Fe/H]), M_bol unchanged (any other band follows the same way);
* mass at fixed luminosity: M_star, M_init → M · 10^δ(M); log g += δ.

Δ and δ are piecewise linear in the *uncorrected* M on ``mass_nodes_msun`` and fall linearly to
zero at ``join_msun``; points at or above it are untouched (DEBCat ≥ 0.8 M⊙ validation).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop import isochrone_mass as im
from darkhunter_pop.config_schema import IsochroneMassConfig, LowmassCorrectionConfig

FloatArray = NDArray[np.float64]


def node_profile(m: ArrayLike, nodes: ArrayLike, values: ArrayLike, join: float) -> FloatArray:
    """Piecewise-linear profile on ``nodes`` (held flat below the first node), → 0 at ``join``."""
    x = np.concatenate([np.asarray(nodes, float), [join]])
    y = np.concatenate([np.asarray(values, float), [0.0]])
    mm = np.asarray(m, float)
    return np.where(mm >= join, 0.0, np.interp(mm, x, y))


def load_bc_tables(cfg: IsochroneMassConfig) -> dict[float, im.BcTable]:
    """The A_V = 0 Gaia-band BC tables per configured [Fe/H] (read in place under mist_root)."""
    if cfg.mist_root is None:
        raise ValueError("isochrone_mass.mist_root is not set")
    root = Path(cfg.mist_root).expanduser()
    g = cfg.grid
    bands = (g.band_g, g.band_bp, g.band_rp)
    return {float(f): im.read_bc_table(root / g.bc_subdir / g.bc_filename_template.format(feh=im._feh_token(f, "bc")), bands)
            for f in g.feh_values}


def correct_grid(grid: im.NativeGrid, lc: LowmassCorrectionConfig, bc_tables: Mapping[float, im.BcTable],
                 bands: tuple[str, str, str] | None = None) -> im.NativeGrid:
    """Return a corrected copy of ``grid`` (see module docstring)."""
    k = {n: i for i, n in enumerate(im.NATIVE_QUANTITIES)}
    v = grid.values.astype(np.float64).copy()
    for f, feh in enumerate(grid.feh):
        bct = bc_tables[min(bc_tables, key=lambda x: abs(x - float(feh)))]
        bnames = bands or tuple(bct.bc.keys())
        cube = v[f]
        m = cube[..., k["star_mass"]]
        ms = np.isfinite(m) & (cube[..., k["phase"]] < 0.5) & (m < lc.join_msun)
        if not ms.any():
            continue
        mm = m[ms]
        dt = node_profile(mm, lc.mass_nodes_msun, lc.dlog_teff, lc.join_msun)
        dmass = node_profile(mm, lc.mass_nodes_msun, lc.dlog_mass, lc.join_msun)
        lt, lg = cube[..., k["log_teff"]][ms], cube[..., k["log_g"]][ms]
        old = bct.interpolate(10.0 ** lt, lg)
        mbol = cube[..., k["mg"]][ms] + old[bnames[0]]
        lt2, lg2 = lt + dt, lg + 4.0 * dt + dmass
        new = bct.interpolate(10.0 ** lt2, lg2)
        upd = {
            "log_teff": lt2, "log_g": lg2, "log_r": cube[..., k["log_r"]][ms] - 2.0 * dt,
            "mg": mbol - new[bnames[0]], "bp": mbol - new[bnames[1]], "rp": mbol - new[bnames[2]],
            "star_mass": mm * 10.0 ** dmass, "initial_mass": cube[..., k["initial_mass"]][ms] * 10.0 ** dmass,
        }
        for name, val in upd.items():
            sl = cube[..., k[name]]
            sl[ms] = val
    return im.NativeGrid(feh=grid.feh, log_age=grid.log_age, eep=grid.eep, values=v.astype(np.float32))
