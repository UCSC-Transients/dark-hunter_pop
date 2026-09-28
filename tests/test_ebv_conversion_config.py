"""PI decision on E(B-V) conversions: every coefficient in config (#295)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from darkhunter_pop import dust_maps
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import (
    DustMapFileSpec,
    DustMapsConfig,
    GaiaBandPolynomialCoefficients,
    PipelineConfig,
)
from darkhunter_pop.elbadry2026_selection import (
    EBV_COLUMN,
    EXTINCTION_STATUS_COLUMN,
    deredden_elbadry2026_rows,
    gaia_band_extinction_babusiaux2018,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    config_subset_fingerprint,
    config_subset_for_stage,
)
from darkhunter_pop.sample_selection import NotApplicable, load_sample_selection_file

ELBADRY2026 = repo_root() / "config" / "selections" / "elbadry2026.yaml"


def _coeffs(**overrides) -> GaiaBandPolynomialCoefficients:
    cfg = load_config().sample_selection.dust_maps.gaia_band_extinction.babusiaux2018
    assert cfg is not None
    return cfg.model_copy(update=overrides)


def _with_dust(cfg: PipelineConfig, **dust_updates) -> PipelineConfig:
    dump = cfg.model_dump(mode="json")
    dust = dump["sample_selection"]["dust_maps"]
    for dotted, value in dust_updates.items():
        node = dust
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return PipelineConfig.model_validate(dump)


def _fingerprint(cfg: PipelineConfig) -> str:
    return config_subset_fingerprint(
        config_subset_for_stage(cfg, STAGE_REGISTRY["sample_selection"])
    )


@pytest.mark.unit
def test_config_carries_pi_decision_values() -> None:
    dust = load_config().sample_selection.dust_maps
    assert dust.r_v == 3.1
    assert dust.maps["green2019"].native_quantity == "reddening"
    assert dust.maps["green2019"].native_to_ebv == 0.884
    assert dust.maps["lallement2019"].native_quantity == "a0"
    assert dust.maps["lallement2019"].native_to_ebv == pytest.approx(1.0 / 3.1, rel=1e-15)
    assert dust.gaia_band_extinction.law == "paper_constant"
    band = dust.gaia_band_extinction.babusiaux2018
    assert band is not None
    assert band.k_g[0] == 0.9761 and band.k_bp[0] == 1.1517 and band.k_rp[0] == 0.6104


@pytest.mark.unit
def test_config_yaml_has_no_pending_marker() -> None:
    text = (repo_root() / "config" / "config.yaml").read_text()
    assert "PENDING PI CONFIRMATION" not in text


@pytest.mark.unit
def test_a0_map_derives_factor_from_r_v() -> None:
    cfg = DustMapsConfig(r_v=2.5, maps={"l": DustMapFileSpec(path="l.h5", native_quantity="a0")})
    assert cfg.maps["l"].native_to_ebv == pytest.approx(0.4)
    # The derived factor is not dumped: a dump re-validates, and follows a new r_v.
    dump = cfg.model_dump(mode="json")
    assert "native_to_ebv" not in dump["maps"]["l"]
    again = DustMapsConfig.model_validate(dump)
    assert again.maps["l"].native_to_ebv == cfg.maps["l"].native_to_ebv
    dump["r_v"] = 4.0
    assert DustMapsConfig.model_validate(dump).maps["l"].native_to_ebv == pytest.approx(0.25)
    # A shared spec instance is not mutated by another config's r_v.
    shared = DustMapFileSpec(path="l.h5", native_quantity="a0")
    one = DustMapsConfig(r_v=2.0, maps={"l": shared})
    two = DustMapsConfig(r_v=4.0, maps={"l": shared})
    assert one.maps["l"].native_to_ebv == 0.5 and two.maps["l"].native_to_ebv == 0.25
    assert shared.native_to_ebv is None


@pytest.mark.unit
def test_a0_map_refusals() -> None:
    with pytest.raises(ValidationError, match="requires sample_selection.dust_maps.r_v"):
        DustMapsConfig(maps={"l": DustMapFileSpec(path="l.h5", native_quantity="a0")})
    with pytest.raises(ValidationError, match="must not be set for an 'a0' map"):
        DustMapsConfig(
            r_v=3.1,
            maps={"l": DustMapFileSpec(path="l.h5", native_quantity="a0", native_to_ebv=0.5)},
        )
    with pytest.raises(ValidationError, match="requires native_to_ebv"):
        DustMapsConfig(maps={"g": DustMapFileSpec(path="g.h5")})


@pytest.mark.unit
def test_babusiaux_law_requires_coefficients_and_r_v() -> None:
    with pytest.raises(ValidationError, match="coefficient block"):
        DustMapsConfig(r_v=3.1, gaia_band_extinction={"law": "babusiaux2018"})
    with pytest.raises(ValidationError, match="requires sample_selection.dust_maps.r_v"):
        DustMapsConfig(
            gaia_band_extinction={"law": "babusiaux2018", "babusiaux2018": _coeffs().model_dump()}
        )


@pytest.mark.unit
def test_every_conversion_knob_changes_stage_fingerprint() -> None:
    base = load_config()
    fp0 = _fingerprint(base)
    variants = {
        "r_v": _with_dust(base, r_v=3.3),
        "green_factor": _with_dust(base, maps__green2019__native_to_ebv=0.996),
        "law": _with_dust(base, gaia_band_extinction__law="babusiaux2018"),
        "k_g": _with_dust(
            base, gaia_band_extinction__babusiaux2018__k_g=[0.97, -0.17, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
        ),
    }
    for name, cfg in variants.items():
        assert _fingerprint(cfg) != fp0, name
    # r_v moves the derived Lallement factor too, not only the r_v key.
    assert variants["r_v"].sample_selection.dust_maps.maps[
        "lallement2019"
    ].native_to_ebv == pytest.approx(1.0 / 3.3)


@pytest.mark.unit
def test_native_cache_fingerprint_ignores_conversion_knobs() -> None:
    base = load_config()
    spec = load_sample_selection_file(ELBADRY2026).extinction
    assert spec is not None
    fp0 = dust_maps.extinction_fingerprint(spec, base.sample_selection.dust_maps)
    for cfg in (
        _with_dust(base, r_v=3.3),
        _with_dust(base, maps__green2019__native_to_ebv=0.996),
        _with_dust(base, gaia_band_extinction__law="babusiaux2018"),
    ):
        assert dust_maps.extinction_fingerprint(spec, cfg.sample_selection.dust_maps) == fp0


@pytest.mark.physics
def test_babusiaux_small_a0_limit_matches_table1_polynomial() -> None:
    coeffs = _coeffs()
    ebv = np.array([1e-6])
    a_g, e_bp_rp, _ = gaia_band_extinction_babusiaux2018(
        np.array([0.72]), ebv, r_v=3.1, coeffs=coeffs
    )
    x = 0.72
    k_g = 0.9761 - 0.1704 * x + 0.0086 * x**2 + 0.0011 * x**3
    k_bp = 1.1517 - 0.0871 * x - 0.0333 * x**2 + 0.0173 * x**3
    k_rp = 0.6104 - 0.0170 * x - 0.0026 * x**2 - 0.0017 * x**3
    assert a_g[0] / ebv[0] == pytest.approx(3.1 * k_g, rel=1e-4)
    assert e_bp_rp[0] / ebv[0] == pytest.approx(3.1 * (k_bp - k_rp), rel=1e-4)


@pytest.mark.physics
def test_babusiaux_fixed_point_is_self_consistent_and_handles_nan() -> None:
    coeffs = _coeffs()
    obs = np.array([1.2, 0.9, np.nan, 1.0])
    ebv = np.array([0.5, 0.1, 0.2, np.nan])
    a_g, e_bp_rp, _ = gaia_band_extinction_babusiaux2018(obs, ebv, r_v=3.1, coeffs=coeffs)
    assert np.isnan(a_g[2:]).all() and np.isnan(e_bp_rp[2:]).all()
    for i in (0, 1):
        colour0 = obs[i] - e_bp_rp[i]
        a0 = 3.1 * ebv[i]
        k = lambda c: c[0] + c[1] * colour0 + c[2] * colour0**2 + c[3] * colour0**3 + c[4] * a0 + c[5] * a0**2 + c[6] * colour0 * a0  # noqa: E731
        assert e_bp_rp[i] == pytest.approx((k(coeffs.k_bp) - k(coeffs.k_rp)) * a0, abs=1e-5)
        assert a_g[i] == pytest.approx(k(coeffs.k_g) * a0, abs=1e-5)


@pytest.mark.unit
def test_babusiaux_non_convergence_is_flagged_not_raised() -> None:
    a_g, e_bp_rp, converged = gaia_band_extinction_babusiaux2018(
        np.array([1.0, 4.0]), np.array([0.1, 3.0]), r_v=3.1, coeffs=_coeffs()
    )
    # Blue/low-A0 converges; very red with A0 ~ 9 diverges (outside the fit range).
    assert converged.tolist() == [True, False]
    assert np.isfinite(a_g[0]) and np.isnan(a_g[1]) and np.isnan(e_bp_rp[1])
    _, _, few = gaia_band_extinction_babusiaux2018(
        np.array([1.0]), np.array([1.0]), r_v=3.1,
        coeffs=_coeffs(max_iterations=1, tolerance=1e-15),
    )
    assert not few[0]


@pytest.mark.physics
def test_nonconvergent_rows_are_not_applicable_with_reason() -> None:
    spec = load_sample_selection_file(ELBADRY2026)
    base = load_config().sample_selection.dust_maps
    gaia_cfg = base.model_copy(
        update={"gaia_band_extinction": base.gaia_band_extinction.model_copy(update={"law": "babusiaux2018"})}
    )
    rows = [{"source_id": 1, "abs_g_mag": 5.0, "bp_rp": 4.0, EBV_COLUMN: 3.0}]
    out = deredden_elbadry2026_rows(rows, spec, None, dust_cfg=gaia_cfg)[0]
    assert isinstance(out["mg_0"], NotApplicable)
    assert out["mg_0"].reason == "extinction_gaia_law_nonconvergent"
    assert out[EXTINCTION_STATUS_COLUMN] == "gaia_law_nonconvergent"


@pytest.mark.physics
def test_dereddening_law_switch() -> None:
    spec = load_sample_selection_file(ELBADRY2026)
    rows = [{"source_id": 1, "abs_g_mag": 5.0, "bp_rp": 1.0, EBV_COLUMN: 0.2}]
    base = load_config().sample_selection.dust_maps
    paper = deredden_elbadry2026_rows(rows, spec, None, dust_cfg=base)[0]
    assert paper["mg_0"] == 5.0 - 2.66 * 0.2
    assert paper["bp_rp_0"] == 1.0 - 1.33 * 0.2
    gaia_cfg = base.model_copy(
        update={"gaia_band_extinction": base.gaia_band_extinction.model_copy(update={"law": "babusiaux2018"})}
    )
    gaia = deredden_elbadry2026_rows(rows, spec, None, dust_cfg=gaia_cfg)[0]
    a_g, e_bp_rp, _ = gaia_band_extinction_babusiaux2018(
        np.array([1.0]), np.array([0.2]), r_v=3.1, coeffs=_coeffs()
    )
    assert gaia["mg_0"] == pytest.approx(5.0 - a_g[0])
    assert gaia["bp_rp_0"] == pytest.approx(1.0 - e_bp_rp[0])
    assert gaia["mg_0"] != paper["mg_0"]
