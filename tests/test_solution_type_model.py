"""#597 (docs/EPOCH_MODEL_SPEC.md §8.16, MOCK_POPULATION_SPEC §12.41): bright stated refit e2 and solution-type model st1."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import epoch_model as em
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import EpochBrightStatedRefitConfig

pytestmark = pytest.mark.unit


def _sec():
    return load_config().dr3.epoch_model


def _on():
    sec = _sec()
    return sec.model_copy(update={"bright_stated_refit": sec.bright_stated_refit.model_copy(update={"enabled": True}),
                                  "solution_type_model": sec.solution_type_model.model_copy(update={"enabled": True})})


def _orbital_vec(g_like: float = 0.0) -> list[float]:
    v = [0.0] * 23
    v[0], v[1], v[10], v[11], v[14], v[15], v[16], v[17], v[18], v[21], v[22] = 2.0, 0.02, 300.0, 3.0, 0.3, 0.05, 60.0, 1.0, 0.05, 10.0, 3.0
    return v


def test_off_by_default_and_labels() -> None:
    sec = _sec()
    assert sec.bright_stated_refit.enabled is False and sec.solution_type_model.enabled is False
    assert em.noise_model_version(sec) == "colour_v2.1+tr1"
    assert em.noise_model_version(_on()) == "colour_v2.1+tr1+e2+st1"
    cfg = em.epoch_model_config_from_mapping(sec)
    assert cfg.bright_stated_refit is None and cfg.solution_type_model is None


def test_e2_scales_stated_error_and_u0_together() -> None:
    off = em.epoch_model_config_from_mapping(_sec())
    on = em.epoch_model_config_from_mapping(_on())
    obs, err = np.zeros(50), np.full(50, 0.1)
    o1, e1, k1 = em.per_ccd_noise(obs, err, 10.5, 1.0, off, np.random.default_rng(1))
    o2, e2, k2 = em.per_ccd_noise(obs, err, 10.5, 1.0, on, np.random.default_rng(1))
    r = on.bright_stated_refit.r(10.5)
    assert r < 1.0 and np.allclose(o1, o2) and np.allclose(e2, r * e1) and k2 == pytest.approx(r * k1)
    _, e3, k3 = em.per_ccd_noise(obs, err, 14.0, 1.0, on, np.random.default_rng(1))
    _, e4, k4 = em.per_ccd_noise(obs, err, 14.0, 1.0, off, np.random.default_rng(1))
    assert np.allclose(e3, e4) and k3 == k4  # no change at G >= 13 (break)


def test_routing_probability_follows_grvs_limit() -> None:
    st = em.epoch_model_config_from_mapping(_on()).solution_type_model
    assert st.p_asb1(10.5, 1.0, 300.0, 0.3, 10.0) > 0.4
    assert st.p_asb1(14.0, 1.0, 300.0, 0.3, 10.0) < 1e-3
    assert st.p_asb1(10.5, float("nan"), 300.0, 0.3, 10.0) == pytest.approx(st.p_asb1(10.5, 1.0, 300.0, 0.3, 10.0))


def test_apply_solution_type_offsets_and_stream() -> None:
    cfg = em.epoch_model_config_from_mapping(_on())
    src = em.source_context(10.0, 20.0, 10.5, bp_rp=1.0)
    rng = em.epoch_model_rng(1, 2, tag=em.PER_CCD_NOISE_RNG_TAG)
    types = set()
    for _ in range(200):
        out, typ = em.apply_solution_type(_orbital_vec(), src, cfg, rng)
        types.add(typ)
        st = cfg.solution_type_model
        pre = "asb1" if typ == "AstroSpectroSB1" else "orbital"
        assert out[18] == pytest.approx(0.05 * 10 ** st.offset(f"{pre}_dlog_sigma_a0", 10.5))
        assert out[1] == pytest.approx(0.02 * 10 ** st.offset(f"{pre}_dlog_sigma_plx", 10.5))
        if typ == "AstroSpectroSB1":
            assert out[21] == pytest.approx(st.offset("asb1_f2_alpha", 10.5) + st.offset("asb1_f2_kappa", 10.5) * 10.0)
        else:
            assert out[21] == 10.0
    assert types == {"Orbital", "AstroSpectroSB1"}
    assert rng.random() == em.epoch_model_rng(1, 2, tag=em.PER_CCD_NOISE_RNG_TAG).random()  # spawned children only
    nonorb = [-7.0] + [1.0] * 12
    assert em.apply_solution_type(nonorb, src, cfg, rng) == (nonorb, None)
    off = em.epoch_model_config_from_mapping(_sec())
    assert em.apply_solution_type(_orbital_vec(), src, off, rng) == (_orbital_vec(), None)


def test_predicted_k1() -> None:
    # a0/plx = 1 AU, P = 1 yr, i = 90, e = 0 -> 29.78 km/s
    assert em.predicted_k1_kms(2.0, 2.0, 365.25, 0.0, 90.0) == pytest.approx(29.78)


def test_schema_guards() -> None:
    with pytest.raises(ValueError):
        EpochBrightStatedRefitConfig(knots_g=[10.0, 13.5], ratio=[0.9, 0.9])
    sec = _sec()
    bad = sec.solution_type_model.model_dump()
    bad["beta"] = bad["beta"][:-1]
    with pytest.raises(ValueError):
        type(sec.solution_type_model).model_validate(bad)
