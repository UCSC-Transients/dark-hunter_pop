"""Noise model v3: colour term s(G, BP-RP), stated-error scale e(G), u0(G, BP-RP) and the noise-model version guard
(#572, #454; docs/EPOCH_MODEL_SPEC.md §8.12)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from darkhunter_pop import epoch_model as em
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import EpochNoiseColourConfig

REPO = Path(__file__).resolve().parents[1]


def _u0_2d() -> em.RuweU0Table2D:
    return em.RuweU0Table2D(g=(10.0, 14.0), colour=(0.5, 2.5), u0=((1.2, 1.4), (1.0, 1.1)))


def _colour(**kw: Any) -> em.NoiseColourConfig:
    base: dict[str, Any] = dict(version="colour_test", colour_clip=(0.0, 3.5), colour_ref=1.2, a_knots_g=(10.0, 14.0),
                                a_coef=(np.log(0.8), 0.0), b_knots_g=(10.0, 14.0), b_coef=(0.0, 0.05), q_coef=0.0,
                                stated_knots_g=(10.0, 14.0), stated_scale=(0.9, 1.0), u0=_u0_2d())
    base.update(kw)
    return em.NoiseColourConfig(**base)


def _cfg(colour: em.NoiseColourConfig | None = None) -> em.EpochModelConfig:
    sec = load_config().dr3.epoch_model
    assert sec is not None
    return dataclasses.replace(em.epoch_model_config_from_mapping(sec), enabled=True, noise_colour=colour)


@pytest.mark.unit
def test_production_label_and_colour_off_is_default() -> None:
    sec = load_config().dr3.epoch_model
    assert sec is not None
    assert em.noise_model_version(sec) == ps.DEFAULT_NOISE_MODEL == "n2d_s1232"
    assert em.noise_model_version(_cfg()) == "n2d_s1232"
    assert em.noise_model_version(_cfg(_colour())) == "colour_test"


@pytest.mark.unit
def test_actual_scale_stated_factor_and_u0_2d() -> None:
    nc = _colour()
    assert nc.actual_scale(10.0, 1.2) == pytest.approx(0.8)
    assert nc.actual_scale(14.0, 2.2) == pytest.approx(np.exp(0.05))
    assert nc.actual_scale(20.0, 9.0) == pytest.approx(np.exp(0.05 * (3.5 - 1.2)))  # clamped G and colour
    assert nc.stated_factor(12.0) == pytest.approx(0.95)
    with pytest.raises(ValueError, match="BP-RP"):
        nc.actual_scale(12.0, float("nan"))
    u0 = _u0_2d()
    assert u0(12.0, 1.5) == pytest.approx(0.25 * (1.2 + 1.4 + 1.0 + 1.1))
    assert u0(5.0, -1.0) == pytest.approx(1.2) and u0(30.0, 9.0) == pytest.approx(1.1)


@pytest.mark.unit
def test_u0_2d_table_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "u0.csv"
    p.write_text("# test\ng,bp_rp,u0\n10,0.5,1.2\n10,2.5,1.4\n14,0.5,1.0\n14,2.5,1.1\n")
    t = em.load_u0_table_2d(p)
    assert t(12.0, 1.5) == pytest.approx(_u0_2d()(12.0, 1.5))
    with pytest.raises(ValueError, match="sha256"):
        em.load_u0_table_2d(p, "0" * 64)
    p.write_text("g,u0\n10,1\n")
    with pytest.raises(ValueError, match="header"):
        em.load_u0_table_2d(p)


@pytest.mark.unit
def test_per_ccd_noise_colour_off_is_the_n2d_model() -> None:
    cfg = _cfg()
    obs, err = np.zeros(500), np.full(500, 0.13)
    o1, e1, k1 = em.per_ccd_noise(obs, err, 11.0, 1.0, cfg, np.random.default_rng(3))
    extra, _ = em.per_ccd_excess_noise(err, 11.0, cfg.excess_noise, np.random.default_rng(3))
    np.testing.assert_array_equal(o1, obs + extra)
    np.testing.assert_array_equal(e1, err)
    assert k1 == em.ruwe_scale_u0(11.0, cfg)


@pytest.mark.unit
def test_per_ccd_noise_colour_on_scales_stated_and_uses_u0_2d() -> None:
    cfg = _cfg(_colour())
    obs, err = np.ones(50), np.full(50, 0.13)
    rng = np.random.default_rng(3)
    o, e, k = em.per_ccd_noise(obs, err, 12.0, 1.5, cfg, rng)
    np.testing.assert_array_equal(o, obs)  # N2d excess replaced by s (applied inside the predict call)
    np.testing.assert_allclose(e, 0.13 * 0.95)
    assert k == pytest.approx(_u0_2d()(12.0, 1.5))
    assert rng.random() == np.random.default_rng(3).random()  # no draw consumed


@pytest.mark.unit
def test_actual_noise_scale_patches_and_restores() -> None:
    def realistic(ra: float, dec: float, g: float) -> float:
        return 0.2

    fake = SimpleNamespace(get_realistic_epoch_astrometry_errors=realistic)
    with em.actual_noise_scale(fake, 0.5):
        assert fake.get_realistic_epoch_astrometry_errors(0.0, 0.0, 12.0) == pytest.approx(0.1)
    assert fake.get_realistic_epoch_astrometry_errors is realistic
    with em.actual_noise_scale(fake, 1.0):
        assert fake.get_realistic_epoch_astrometry_errors is realistic
    src = em.SourceEpochContext(12.0, bp_rp=1.2)
    assert em.colour_actual_scale(src, _cfg()) == 1.0
    assert em.colour_actual_scale(src, _cfg(_colour())) == pytest.approx(np.exp(np.log(0.8) / 2))


@pytest.mark.unit
def test_schema_validation() -> None:
    ok = dict(u0_table="x.csv", u0_table_sha256="0" * 64)
    EpochNoiseColourConfig(**ok)
    with pytest.raises(ValueError):
        EpochNoiseColourConfig(**ok, version="n2d_s1232")
    with pytest.raises(ValueError):
        EpochNoiseColourConfig(**ok, a_knots_g=[10.0, 9.0], a_coef=[0.0, 0.0])
    with pytest.raises(ValueError):
        EpochNoiseColourConfig(**ok, stated_scale_knots_g=[10.0], stated_scale=[0.0])


@pytest.mark.unit
def test_config_noise_colour_section_loads_when_present() -> None:
    sec = load_config().dr3.epoch_model
    assert sec is not None
    if sec.noise_colour is None:
        pytest.skip("no noise_colour section configured")
    on = sec.model_copy(update={"noise_colour": sec.noise_colour.model_copy(update={"enabled": True})})
    cfg = em.epoch_model_config_from_mapping(on)
    assert cfg.noise_colour is not None and em.noise_model_version(cfg) == sec.noise_colour.version
    assert 0.5 < cfg.noise_colour.actual_scale(15.0, 1.0) < 1.5
    assert 0.8 < cfg.noise_colour.u0(15.0, 1.0) < 1.3


@pytest.mark.unit
def test_primary_model_version_guard() -> None:
    frag = ps.load_proposal_set_fragment(REPO / "config/population/proposal_set_single46.yaml")
    old = ps.primary_model_version(frag.proposal)
    assert old == "looks_single_v2:ruwe1.25:dm-0.4:c1.05-2.75:g19"  # older generations keep their version
    new = frag.proposal.model_copy(update={"noise_model": "colour_v1"})
    assert ps.primary_model_version(new).endswith(":noise_colour_v1")
    with pytest.raises(ValueError, match="different primary models"):
        ps.check_single_primary_model([frag.proposal, new])
    co = ps.load_proposal_set_fragment(REPO / "config/population/proposal_set_co_v1_wd1003.yaml").proposal
    co_new = co.model_copy(update={"noise_model": "colour_v1"})
    assert ps.primary_model_version(co_new).endswith(":noise_colour_v1:co_v1")
    assert ps.is_compact_object_version(ps.primary_model_version(co_new))


@pytest.mark.unit
def test_fingerprint_unchanged_for_default_noise_model() -> None:
    frag = ps.load_proposal_set_fragment(REPO / "config/population/proposal_set_single46.yaml")
    assert ps.fragment_fingerprint(frag) == "f5e2d14a13ba"
    other = frag.model_copy(update={"proposal": frag.proposal.model_copy(update={"noise_model": "colour_v1"})})
    assert ps.fragment_fingerprint(other) != "f5e2d14a13ba"


@pytest.mark.unit
def test_launch_refuses_noise_model_mismatch() -> None:
    frag = ps.load_proposal_set_fragment(REPO / "config/population/proposal_set_single46.yaml")
    ps.check_noise_model(frag.proposal, _cfg())
    with pytest.raises(ValueError, match="noise"):
        ps.check_noise_model(frag.proposal, _cfg(_colour()))
    ps.check_noise_model(frag.proposal.model_copy(update={"noise_model": "colour_test"}), _cfg(_colour()))


@pytest.mark.unit
def test_epoch_source_context_carries_system_colour() -> None:
    src = ps.epoch_source_context({"ra_deg": 10.0, "dec_deg": 20.0, "phot_g_mean_mag": 12.0, "sys_bp_rp": 1.3})
    assert src.bp_rp == pytest.approx(1.3)
    assert ps.epoch_source_context({"ra_deg": 10.0, "dec_deg": 20.0, "phot_g_mean_mag": 12.0}).bp_rp is None
