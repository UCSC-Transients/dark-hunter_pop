"""#595 (docs/EPOCH_MODEL_SPEC.md §8.15): bright per-source noise scale ss1 and the empirical sigma_a0 scale sa1."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from darkhunter_pop import epoch_model as em
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import EpochSourceScaleConfig

pytestmark = pytest.mark.unit


def _ss(version: str = "ss_test") -> em.SourceScaleConfig:
    return em.SourceScaleConfig(version=version, break_g=13.0, knots_g=(10.0, 11.5, 12.5), sigma_log10=(0.05, 0.06, 0.02))


def _cfg(**kw) -> em.EpochModelConfig:
    sec = load_config().dr3.epoch_model
    return dataclasses.replace(em.epoch_model_config_from_mapping(sec), enabled=True, noise_colour=None, transit_recal=None, bright_stated_refit=None, solution_type_model=None, **kw)


def test_sigma_bright_only_no_interpolation_across_break() -> None:
    ss = _ss()
    assert ss.sigma(9.0) == pytest.approx(0.05)  # constant outside the knots
    assert ss.sigma(10.75) == pytest.approx(0.055)
    assert ss.sigma(12.99) == pytest.approx(0.02)
    assert ss.sigma(13.0) == 0.0 and ss.sigma(17.0) == 0.0
    assert ss.draw(15.0, np.random.default_rng(1)) == 1.0


def test_draw_is_lognormal_and_reproducible() -> None:
    ss = _ss()
    z = np.array([np.log10(ss.draw(11.5, np.random.default_rng(i))) for i in range(4000)])
    assert np.std(z) == pytest.approx(0.06, rel=0.06) and abs(np.mean(z)) < 0.005
    assert ss.draw(11.5, np.random.default_rng(7)) == ss.draw(11.5, np.random.default_rng(7))


def test_factor_helpers_do_not_move_the_noise_stream() -> None:
    src = em.source_context(10.0, 20.0, 11.0, bp_rp=1.0)
    on = _cfg(source_noise_scale=_ss(), source_sigma_a0_scale=_ss("sa_test"))
    r = em.epoch_model_rng(1, 2, 3, tag=em.PER_CCD_NOISE_RNG_TAG)
    f1 = em.source_noise_factor(src, on, r)
    f2 = em.source_sigma_a0_factor(src, on, r)
    ref = em.epoch_model_rng(1, 2, 3, tag=em.PER_CCD_NOISE_RNG_TAG)
    assert r.random() == ref.random()  # spawned children only
    assert f1 != 1.0 and f2 != 1.0 and f1 != f2
    off = _cfg()
    assert em.source_noise_factor(src, off, r) == 1.0 and em.source_sigma_a0_factor(src, off, r) == 1.0


def test_per_ccd_noise_scales_the_n2d_excess_only() -> None:
    cfg = _cfg()
    obs, err = np.zeros(200), np.full(200, 0.1)
    o1, e1, k1 = em.per_ccd_noise(obs, err, 11.0, 1.0, cfg, np.random.default_rng(3))
    o2, e2, k2 = em.per_ccd_noise(obs, err, 11.0, 1.0, cfg, np.random.default_rng(3), source_scale=1.7)
    assert np.allclose(o2, 1.7 * o1) and np.array_equal(e1, e2) and k1 == k2  # stated errors unchanged


def test_version_labels_and_config_off_by_default() -> None:
    sec = load_config().dr3.epoch_model
    sec = sec.model_copy(update={"bright_stated_refit": None, "solution_type_model": None})  # #599: e2 + st1 adopted; labels checked without them
    assert sec.source_noise_scale is not None and sec.source_noise_scale.enabled is False
    assert sec.source_sigma_a0_scale is not None and sec.source_sigma_a0_scale.enabled is False
    assert em.noise_model_version(sec) == "colour_v2.1+tr1"
    on = sec.model_copy(update={"source_noise_scale": sec.source_noise_scale.model_copy(update={"enabled": True})})
    assert em.noise_model_version(on) == "colour_v2.1+tr1+ss1"
    both = on.model_copy(update={"source_sigma_a0_scale": sec.source_sigma_a0_scale.model_copy(update={"enabled": True})})
    assert em.noise_model_version(both) == "colour_v2.1+tr1+ss1+sa1"
    cfg = em.epoch_model_config_from_mapping(on)
    assert cfg.source_noise_scale is not None and cfg.source_noise_scale.sigma(13.5) == 0.0
    assert em.epoch_model_config_from_mapping(sec).source_noise_scale is None


def test_schema_rejects_knots_across_the_break() -> None:
    with pytest.raises(ValueError):
        EpochSourceScaleConfig(version="x", break_g=13.0, sigma_knots_g=[10.0, 13.5], sigma_log10=[0.05, 0.05])
    with pytest.raises(ValueError):
        EpochSourceScaleConfig(version="x", sigma_knots_g=[10.0, 11.0], sigma_log10=[0.05])
