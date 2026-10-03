"""CMD evolved-primary classifier and giant diagnostics (MP-Q28, #413).

docs/MOCK_POPULATION_SPEC.md §10. Synthetic CMDs only; no mwdust map lookups, no gaiamock.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import constants
from darkhunter_pop import giants as gi
from darkhunter_pop.config_loader import load_config

GCFG = "config/population/giants.yaml"


@pytest.fixture(scope="module")
def gcfg() -> gi.GiantsConfig:
    return gi.load_giants_config(GCFG)


def _synthetic_cmd(rng: np.random.Generator, n: int, sigma: float, twin_frac: float, n_giant: int):
    """MS ridge R(C) = 2 + 3.5 C, Gaussian singles, a twin fraction 0.753 mag brighter, and giants."""
    c = rng.uniform(0.4, 2.4, n)
    m = 2.0 + 3.5 * c + rng.normal(0.0, sigma, n)
    twin = rng.random(n) < twin_frac
    m[twin] -= constants.TWIN_BRIGHTENING_MAG
    cg = rng.uniform(1.0, 1.6, n_giant)
    mg = rng.normal(0.5, 0.4, n_giant)
    return np.r_[c, cg], np.r_[m, mg], np.r_[np.zeros(n, bool), np.ones(n_giant, bool)], np.r_[twin, np.zeros(n_giant, bool)]


@pytest.mark.unit
def test_config_loads(gcfg: gi.GiantsConfig) -> None:
    assert gcfg.extinction.map == "combined19"
    assert gcfg.provisional_n_sigma > 0
    assert gcfg.ridge.colour_min < gcfg.ridge.colour_max


@pytest.mark.physics
def test_ridge_recovers_injected_mode_and_faint_width(gcfg: gi.GiantsConfig) -> None:
    rng = np.random.default_rng(1)
    sigma = 0.25
    c, m, _, _ = _synthetic_cmd(rng, 400_000, sigma, twin_frac=0.2, n_giant=5_000)
    ridge = gi.fit_ms_ridge(m, c, np.full(c.size, 50.0), gcfg.ridge)
    inner = (ridge.colour > 0.6) & (ridge.colour < 2.2)
    truth = 2.0 + 3.5 * ridge.colour[inner]
    # the mode sits within one histogram-smoothing scale of the truth (ridge slope × bin half-width)
    assert np.max(np.abs(ridge.mag[inner] - truth)) < 0.1 + 3.5 * gcfg.ridge.colour_step / 2
    # the faint side carries singles only: width ≈ injected σ despite 20% twins (bin-slope broadening adds a little)
    expected = math.sqrt(sigma**2 + (3.5 * gcfg.ridge.colour_step) ** 2 / 12.0)
    assert np.median(ridge.sigma[inner]) == pytest.approx(expected, rel=0.15)


@pytest.mark.physics
def test_classifier_keeps_twins_and_flags_giants(gcfg: gi.GiantsConfig) -> None:
    rng = np.random.default_rng(2)
    c, m, is_g, is_twin = _synthetic_cmd(rng, 300_000, 0.25, twin_frac=0.3, n_giant=3_000)
    ridge = gi.fit_ms_ridge(m, c, np.full(c.size, 50.0), gcfg.ridge)
    cls = gi.classify_evolved(m, c, np.zeros(c.size), ridge, n_sigma=3.0)
    assert np.mean(cls.evolved[is_g]) > 0.99
    assert np.mean(cls.evolved[is_twin & cls.classified]) < 0.01
    assert np.mean(cls.evolved[~is_g & ~is_twin & cls.classified]) < 1e-3


@pytest.mark.unit
def test_classifier_distance_error_widens_threshold(gcfg: gi.GiantsConfig) -> None:
    ridge = gi.MSRidge(
        colour=np.array([0.5, 2.0]), mag=np.array([4.0, 9.0]), sigma=np.array([0.3, 0.3]), n_rows=np.array([1000, 1000])
    )
    m = np.array([4.0 - 0.753 - 1.0, 4.0 - 0.753 - 1.0, 0.0, np.nan])
    c = np.array([0.5, 0.5, 3.0, 1.0])
    cls = gi.classify_evolved(m, c, np.array([0.0, 1.0, 0.0, 0.0]), ridge, n_sigma=3.0)
    # 1.0 mag beyond the twin line: evolved at σ_tot=0.3 (cut 0.9) but not at σ_tot=√1.09 (cut 3.13)
    assert cls.evolved.tolist() == [True, False, False, False]
    assert cls.classified.tolist() == [True, True, False, False]  # out of colour range / no CMD


@pytest.mark.unit
def test_dereddened_cmd_and_sigma_mu() -> None:
    mg0, c0 = gi.dereddened_cmd([10.0], [1.0], [100.0], [0.2], [0.1])
    assert mg0[0] == pytest.approx(10.0 - 5.0 - 0.2)
    assert c0[0] == pytest.approx(0.9)
    s = gi.sigma_mu_from_quantiles([90.0], [100.0], [110.0])
    assert s[0] == pytest.approx(5 / math.log(10) * 0.1)
    bad, _ = gi.dereddened_cmd([10.0], [1.0], [-1.0], [0.0], [0.0])
    assert np.isnan(bad[0])


@pytest.mark.unit
def test_cmd_for_rows_zero_extinction(gcfg: gi.GiantsConfig) -> None:
    cfg = load_config(Path("config/config.yaml"))
    row = gi.cmd_for_rows([12.0], [1.2], [10.0], [5.0], [1000.0], [900.0], [1100.0], cfg, gcfg, ebv=[0.0])
    assert bool(row.extinction_ok[0])
    assert row.mg0[0] == pytest.approx(12.0 - 10.0)
    assert row.colour0[0] == pytest.approx(1.2)


@pytest.mark.unit
def test_bolometric_correction_and_solar_radius() -> None:
    assert gi.bolometric_correction_g([5772.0])[0] == pytest.approx(0.06)
    # the cool fit's a0 makes BC_G continuous at 4000 K (Andrae et al. 2018 Table 4 caption)
    lo, hi = gi.bolometric_correction_g([3999.999, 4000.0])
    assert lo == pytest.approx(hi, abs=0.02)
    assert np.isnan(gi.bolometric_correction_g([9000.0])[0])
    r = gi.cmd_radius_rsun([constants.ANDRAE2018_MBOL_SUN - 0.06], [5772.0])
    assert r[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_eggleton_and_roche_period_floor() -> None:
    assert gi.eggleton_roche_lobe_fraction([1.0])[0] == pytest.approx(0.3789, abs=1e-3)
    # a 10 Rsun, 1.2 Msun giant with a 0.6 Msun companion: a = R1/(r_L/a), Kepler's law
    p = gi.roche_period_floor_days([10.0], [1.2], [0.6])[0]
    rl = gi.eggleton_roche_lobe_fraction([2.0])[0]
    a_au = 10.0 / rl * 0.00465047
    assert p == pytest.approx(365.25 * math.sqrt(a_au**3 / 1.8), rel=2e-3)
    # floor rises steeply with radius (P ∝ R^1.5)
    assert gi.roche_period_floor_days([40.0], [1.2], [0.6])[0] == pytest.approx(8.0 * p, rel=1e-6)


@pytest.mark.unit
def test_hook_replaces_parent_flag() -> None:
    """The documented hook: dataclasses.replace swaps ParentSnapshot.is_giant (frozen dataclass)."""
    from darkhunter_pop.proposal_set import ParentSnapshot

    assert dataclasses.is_dataclass(ParentSnapshot)
    assert "is_giant" in {f.name for f in dataclasses.fields(ParentSnapshot)}
