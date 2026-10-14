"""Spec §12.24: MIST low-mass correction on the native grid (synthetic grid, no MIST files)."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import isochrone_mass as im
from darkhunter_pop import mist_correction as mcorr
from darkhunter_pop.config_schema import IsochroneMassConfig, LowmassCorrectionConfig


def _bc() -> im.BcTable:
    teff = np.linspace(2500.0, 8000.0, 12)
    logg = np.linspace(3.0, 5.5, 6)
    T = teff[:, None] + 0 * logg[None, :]
    # BCs linear in Teff: G -> 0, BP - RP colour = (BC_RP - BC_BP) = -(T - 6000)/1000 + 0.8
    return im.BcTable(teff=teff, logg=logg, bc={"G": 0.0 * T, "BP": -0.5 * (0.8 - (T - 6000) / 1000), "RP": 0.5 * (0.8 - (T - 6000) / 1000)})


def _grid() -> im.NativeGrid:
    k = {n: i for i, n in enumerate(im.NATIVE_QUANTITIES)}
    v = np.full((1, 1, 3, len(im.NATIVE_QUANTITIES)), np.nan, np.float32)
    for e, (m, t, mg) in enumerate(((0.3, 3400.0, 10.0), (0.6, 3900.0, 7.5), (1.0, 5800.0, 4.6))):
        v[0, 0, e, [k["initial_mass"], k["star_mass"], k["log_teff"], k["log_g"], k["log_r"], k["phase"], k["mg"]]] = [
            m, m, np.log10(t), 4.8, -0.5, 0.0, mg]
        col = 0.8 - (t - 6000) / 1000
        v[0, 0, e, k["bp"]] = mg + 0.5 * col
        v[0, 0, e, k["rp"]] = mg - 0.5 * col
    return im.NativeGrid(feh=np.array([0.0]), log_age=np.array([9.5]), eep=np.arange(3), values=v)


@pytest.mark.unit
def test_correct_grid_teff_and_mass() -> None:
    k = {n: i for i, n in enumerate(im.NATIVE_QUANTITIES)}
    lc = LowmassCorrectionConfig(enabled=True, mass_nodes_msun=(0.3, 0.6), dlog_teff=(-0.02, -0.01),
                                 dlog_mass=(-0.04, -0.02), join_msun=0.9)
    g = _grid()
    out = mcorr.correct_grid(g, lc, {0.0: _bc()}, bands=("G", "BP", "RP"))
    a, b = g.values[0, 0].astype(float), out.values[0, 0].astype(float)
    np.testing.assert_allclose(b[2], a[2], equal_nan=True)  # >= join untouched
    np.testing.assert_allclose(b[:2, k["log_teff"]] - a[:2, k["log_teff"]], [-0.02, -0.01], atol=1e-6)
    np.testing.assert_allclose(b[:2, k["mg"]], a[:2, k["mg"]], atol=1e-5)  # BC_G = 0: M_G = M_bol unchanged
    assert np.all((b[:2, k["bp"]] - b[:2, k["rp"]]) > (a[:2, k["bp"]] - a[:2, k["rp"]]))  # cooler -> redder
    np.testing.assert_allclose(b[:2, k["star_mass"]] / a[:2, k["star_mass"]], 10 ** np.array([-0.04, -0.02]), rtol=1e-5)
    np.testing.assert_allclose(b[:2, k["log_r"]] - a[:2, k["log_r"]], [0.04, 0.02], atol=1e-6)


@pytest.mark.unit
def test_node_profile_and_config_key() -> None:
    np.testing.assert_allclose(mcorr.node_profile([0.1, 0.3, 0.75, 0.9, 1.2], [0.3, 0.6], [-0.02, -0.01], 0.9),
                               [-0.02, -0.02, -0.005, 0.0, 0.0])
    base = IsochroneMassConfig()
    off = IsochroneMassConfig(lowmass_correction=LowmassCorrectionConfig(dlog_teff=(0.01,) * 8))
    on = IsochroneMassConfig(lowmass_correction=LowmassCorrectionConfig(enabled=True))
    assert im.config_key(base) == im.config_key(off) != im.config_key(on)
    with pytest.raises(ValueError):
        LowmassCorrectionConfig(mass_nodes_msun=(0.5, 1.0), dlog_teff=(0, 0), dlog_mass=(0, 0), join_msun=0.9)
