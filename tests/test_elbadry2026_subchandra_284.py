"""El-Badry 2026 ``sub_chandrasekhar`` #284: ``M̃2 > M̃1``, AMRF cut, analytic ``σ_M̃2``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    AmrfCutSpec,
    SampleSelectionMode,
    SigmaM2TildeSpec,
)
from darkhunter_pop.elbadry2026_m2_sigma import (
    LEGACY_MC_FIXED_PROVENANCE,
    clear_non_elbadry_m2_astrometric_sigma,
    is_elbadry_sigma_provenance,
    sigma_m2_tilde_analytic_msun,
    sigma_m2_tilde_astrometric_msun,
    sigma_provenance_tag,
)
from darkhunter_pop.elbadry2026_selection import (
    AMRF_THRESHOLD_COLUMN,
    amrf_threshold_for_row,
    amrf_with_luminous_companion,
    attach_amrf_threshold,
    mass_magnitude_relation,
    shahaf2019_class_boundary,
)
from darkhunter_pop.janssens_mass import (
    invert_mg_to_mass,
    segments_from_table,
    sigma_log10_mass_from_fit,
)
from darkhunter_pop.mc_mass_function import synthetic_orbital_solution
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    invert_astrometric_companion_mass,
    photocenter_a0_from_thiele_innes,
)
from darkhunter_pop.sample_selection import (
    NotApplicable,
    SampleSelection,
    load_sample_selection_file,
)
from darkhunter_pop.schemas import ParameterSet

pytestmark = pytest.mark.physics


def _spec():
    return load_sample_selection_file(repo_root() / "config/selections/elbadry2026.yaml")


def _sub_chandrasekhar(spec):
    astro = next(b for b in spec.branches if b.id == "astrometric")
    return next(s for s in astro.subsamples if s.id == "sub_chandrasekhar")


# --- frozen file -----------------------------------------------------------


def test_frozen_file_carries_284_cuts_and_switches_for_forward_model() -> None:
    spec = _spec()
    assert spec.schema_version == 3
    sub = _sub_chandrasekhar(spec)
    forward = [
        c.id for c in sub.cuts if SampleSelectionMode.FORWARD_MODEL in c.applies_to
    ]
    # The schema-v2 (#284) chain, unchanged, is the forward_model chain (#315).
    assert forward == [
        "main_sequence",
        "m2_range",
        "m2_over_m1",
        "amrf",
        "m2_error",
        "period",
        "g_mag",
    ]
    cuts = {c.id: c for c in sub.cuts}
    # Frozen thresholds unchanged by #284.
    assert cuts["m2_range"].parameters == {"m2_msun_min": 1.05, "m2_msun_max": 1.40}
    assert cuts["m2_error"].parameters == {"m2_msun_error_max": 0.105}
    assert cuts["period"].parameters == {"period_days_max": 900.0}
    assert cuts["m2_over_m1"].parameters == {"m2_over_m1_min": 1.0}
    assert AMRF_THRESHOLD_COLUMN in cuts["amrf"].expression
    assert spec.sigma_m2_tilde is not None
    assert spec.sigma_m2_tilde.method == "analytic"
    assert spec.amrf_cut is not None
    assert spec.primary_mass is not None
    assert spec.primary_mass.propagate_fit_uncertainty is False
    item = next(i for i in spec.open_items if i.id == "284")
    assert item.status == "resolved"


def test_amrf_cut_spec_validation() -> None:
    with pytest.raises(ValueError):
        AmrfCutSpec(criterion="flat")
    with pytest.raises(ValueError):
        AmrfCutSpec(criterion="shahaf2019_class3", mass_luminosity="power_law")
    with pytest.raises(ValueError):
        SigmaM2TildeSpec(method="bootstrap")  # type: ignore[arg-type]


# --- Shahaf et al. (2019) triage boundary ---------------------------------


def test_shahaf2019_power_law_beta5_matches_published_maxima() -> None:
    """Shahaf+2019 §3, Fig. 1 (β = 5): max A_MS ≈ 0.36, max A_triple(q2=1) ≈ 0.56."""
    mag = mass_magnitude_relation("power_law", power_law_beta=5.0)
    assert shahaf2019_class_boundary(1.0, mag, companion="ms") == pytest.approx(
        0.36, abs=0.005
    )
    assert shahaf2019_class_boundary(1.0, mag, companion="triple") == pytest.approx(
        0.56, abs=0.005
    )
    # Pure power law: the boundary does not depend on M1.
    assert shahaf2019_class_boundary(0.7, mag) == pytest.approx(
        shahaf2019_class_boundary(1.6, mag), abs=1e-6
    )


def test_shahaf2019_triple_fainter_than_primary_cap() -> None:
    """Eq. (13): S_triple = 1 at q = 2^(1 − 1/β) for q2 = 1; A there is still > 0."""
    beta = 5.0
    q_cap = 2.0 ** (1.0 - 1.0 / beta)
    s_cap = 2.0 * (q_cap / 2.0) ** beta
    assert s_cap == pytest.approx(1.0, rel=1e-12)
    assert float(amrf_with_luminous_companion(q_cap, s_cap)) > 0.0


# Digitized from Shahaf et al. (2019) Fig. 3 (Hipparcos band, Table A1 relation).
_SHAHAF2019_FIG3 = (
    # (M1, max A_triple, max A_MS)
    (0.85, 0.675, 0.435),
    (1.0, 0.655, 0.418),
    (1.2, 0.625, 0.395),
    (1.4, 0.600, 0.378),
    (1.8, 0.550, 0.340),
)


@pytest.mark.parametrize(("m1", "a_triple", "a_ms"), _SHAHAF2019_FIG3)
def test_shahaf2019_hp_boundary_reproduces_fig3(
    m1: float, a_triple: float, a_ms: float
) -> None:
    mag = mass_magnitude_relation("shahaf2019_hp")
    assert shahaf2019_class_boundary(m1, mag) == pytest.approx(a_triple, abs=0.01)
    assert shahaf2019_class_boundary(m1, mag, companion="ms") == pytest.approx(
        a_ms, abs=0.01
    )


def test_shahaf2019_hp_flux_ratio_matches_eq_a4() -> None:
    """Table A1 → Eq. (A4): ``S = 1.40 (M1)^3.2 q^8.2`` for F/G primaries, qM1 < 0.9."""
    mag = mass_magnitude_relation("shahaf2019_hp")
    m1, q = 1.2, 0.5
    s = 10.0 ** (-0.4 * (mag(np.array([q * m1]))[0] - mag(np.array([m1]))[0]))
    # Exact Table A1 algebra: 10^{0.4(4.955-4.57)} M1^{(20.39-12.39)/2.5} q^{20.39/2.5}.
    exact = 10.0 ** (0.4 * (4.955 - 4.57)) * m1 ** (8.0 / 2.5) * q ** (20.39 / 2.5)
    assert s == pytest.approx(exact, rel=1e-9)
    # The paper prints the coefficients rounded (1.40, 3.2, 8.2): agree to ~5 %.
    assert s == pytest.approx(1.40 * m1**3.2 * q**8.2, rel=0.06)
    # Same segment for both stars → S = q^(12.39/2.5) ≈ q^5 (Eq. A4 first line).
    q_same = 0.9
    s_same = 10.0 ** (
        -0.4 * (mag(np.array([q_same * 1.4]))[0] - mag(np.array([1.4]))[0])
    )
    assert s_same == pytest.approx(q_same ** (12.39 / 2.5), rel=1e-9)


def test_amrf_threshold_flat_class3_and_not_applicable() -> None:
    spec = _spec()
    flat = spec.model_copy(
        update={"amrf_cut": AmrfCutSpec(criterion="flat", flat_min=0.65)}
    )
    assert amrf_threshold_for_row({"m1_tilde_msun": 1.1}, flat) == (0.65, "flat")
    hp = spec.model_copy(
        update={
            "amrf_cut": AmrfCutSpec(
                criterion="shahaf2019_class3", mass_luminosity="shahaf2019_hp"
            )
        }
    )
    value, source = amrf_threshold_for_row({"m1_tilde_msun": 1.0}, hp)
    assert source == "shahaf2019_class3:shahaf2019_hp"
    assert value == pytest.approx(0.655, abs=0.01)
    outside, _ = amrf_threshold_for_row({"m1_tilde_msun": 2.5}, hp)
    assert isinstance(outside, NotApplicable)
    evolved, _ = amrf_threshold_for_row({"m1_tilde_msun": NotApplicable("evolved")}, hp)
    assert isinstance(evolved, NotApplicable) and evolved.reason == "evolved"
    none_spec = spec.model_copy(update={"amrf_cut": None})
    assert amrf_threshold_for_row({"m1_tilde_msun": 1.0}, none_spec) is None
    # Idempotent attach.
    pre = {"m1_tilde_msun": 1.0, AMRF_THRESHOLD_COLUMN: 0.1}
    assert attach_amrf_threshold(pre, hp)[AMRF_THRESHOLD_COLUMN] == 0.1


# --- the new cuts through the real evaluator ----------------------------------


def _window_row(sid: int, **kw: Any) -> dict[str, Any]:
    row = {
        "source_id": sid,
        "nss_solution_type": "Orbital",
        "main_sequence": True,
        "m1_tilde_msun": 1.0,
        "m2_tilde_msun": 1.2,
        "amrf": 0.709,
        "goodness_of_fit": 1.0,
        "period_day": 400.0,
        "phot_g_mean_mag": 12.0,
        "sigma_m2_astrometric_msun": 0.05,
    }
    row.update(kw)
    return row


def _sub_chandra_survivors(rows, spec=None) -> set[int]:
    # The #284 Janssens chain applies to forward_model only since schema v3
    # (#315); reproduction mode is the Shahaf 2023b catalog cross-match.
    spec = spec or _spec()
    sel = SampleSelection(spec, mode=SampleSelectionMode.FORWARD_MODEL)
    result = sel.evaluate(rows, membership={"andrews2022": frozenset()})
    return set(result.subsample_surviving["sub_chandrasekhar"])


def test_m2_over_m1_and_amrf_cuts_regression() -> None:
    rows = [
        _window_row(1),  # q = 1.2, A = 0.709: passes
        _window_row(2, m1_tilde_msun=1.3, m2_tilde_msun=1.2, amrf=0.58),  # q < 1
        _window_row(3, m1_tilde_msun=1.2, m2_tilde_msun=1.2, amrf=0.63),  # q = 1
        _window_row(4, m1_tilde_msun=1.1, m2_tilde_msun=1.13, amrf=0.64),  # A ≤ 0.65
        _window_row(5, sigma_m2_astrometric_msun=0.2),  # σ fails (unchanged cut)
    ]
    assert _sub_chandra_survivors(rows) == {1}


def test_amrf_cut_class3_variant_is_switchable() -> None:
    spec = _spec().model_copy(
        update={
            "amrf_cut": AmrfCutSpec(
                criterion="shahaf2019_class3", mass_luminosity="shahaf2019_hp"
            )
        }
    )
    # q = 1.03 at M1 = 1.1: A ≈ 0.644 > Hp class-III boundary (~0.64 at 1.1)?
    m1 = 1.1
    boundary = shahaf2019_class_boundary(m1, mass_magnitude_relation("shahaf2019_hp"))
    above = _window_row(10, m1_tilde_msun=m1, m2_tilde_msun=1.2, amrf=boundary + 0.005)
    below = _window_row(11, m1_tilde_msun=m1, m2_tilde_msun=1.2, amrf=boundary - 0.005)
    assert _sub_chandra_survivors([above, below], spec) == {10}
    # The flat default rejects both (both < 0.65).
    assert boundary + 0.005 < 0.65
    assert _sub_chandra_survivors([above, below]) == set()


# --- analytic σ_M̃2 -----------------------------------------------------------


def _numeric_m2(values: np.ndarray, idx: dict[str, int], m1: float) -> float:
    a0 = photocenter_a0_from_thiele_innes(
        values[idx["a_thiele_innes"]],
        values[idx["b_thiele_innes"]],
        values[idx["f_thiele_innes"]],
        values[idx["g_thiele_innes"]],
    )
    mf = astrometric_mass_function(a0, values[idx["parallax"]], values[idx["period"]])
    return float(invert_astrometric_companion_mass(m1, mf, 0.0))


def test_analytic_sigma_matches_finite_difference_jacobian() -> None:
    sol = synthetic_orbital_solution(relative_error=0.02, seed=3)
    m1 = 1.1
    names = list(sol.names)
    idx = {n: i for i, n in enumerate(names)}
    mean = sol.values_array()
    cov = sol.covariance_array()
    grad = np.zeros(len(names))
    for i in range(len(names)):
        h = 1e-6 * max(abs(mean[i]), 1e-3)
        up, dn = mean.copy(), mean.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (_numeric_m2(up, idx, m1) - _numeric_m2(dn, idx, m1)) / (2 * h)
    expected = float(np.sqrt(grad @ cov @ grad))
    got = sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=m1)
    assert got == pytest.approx(expected, rel=1e-5)


def test_analytic_sigma_agrees_with_mc_when_well_constrained() -> None:
    sol = synthetic_orbital_solution(relative_error=0.01, seed=7)
    analytic = sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=1.0)
    mc = sigma_m2_tilde_astrometric_msun(
        sol,
        m1_tilde_msun=1.0,
        n_draws=40000,
        random_seed=1,
        eig_rel_floor=1e-12,
        eig_abs_floor=1e-18,
    )
    assert analytic is not None and mc is not None
    assert analytic == pytest.approx(mc, rel=0.05)


def test_nsstools_blocks_equals_full_when_blocks_uncorrelated() -> None:
    sol = synthetic_orbital_solution(relative_error=0.02, seed=5)
    names = list(sol.names)
    cov = sol.covariance_array().copy()
    abfg = [names.index(n) for n in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")]
    others = [names.index("parallax"), names.index("period")]
    mask = np.zeros_like(cov, dtype=bool)
    mask[np.ix_(abfg, abfg)] = True
    for i in others:
        mask[i, i] = True
    cov[~mask] = 0.0
    block = ParameterSet(
        names=names,
        values=list(sol.values),
        covariance=cov.tolist(),
        provenance="synthetic_block",
    )
    full = sigma_m2_tilde_analytic_msun(block, m1_tilde_msun=1.2, analytic_covariance="full")
    split = sigma_m2_tilde_analytic_msun(
        block, m1_tilde_msun=1.2, analytic_covariance="nsstools_blocks"
    )
    assert full == pytest.approx(split, rel=1e-12)
    # With the real cross terms they generally differ.
    full_c = sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=1.2)
    split_c = sigma_m2_tilde_analytic_msun(
        sol, m1_tilde_msun=1.2, analytic_covariance="nsstools_blocks"
    )
    assert full_c != pytest.approx(split_c, rel=1e-6)


def test_janssens_m1_uncertainty_inflates_sigma_both_methods() -> None:
    sol = synthetic_orbital_solution(relative_error=0.01, seed=9)
    kw = dict(m1_tilde_msun=1.1)
    base = sigma_m2_tilde_analytic_msun(sol, **kw)
    with_m1 = sigma_m2_tilde_analytic_msun(sol, sigma_log10_m1=0.04, **kw)
    assert with_m1 > base
    mc_kw = dict(n_draws=20000, random_seed=2, eig_rel_floor=1e-12, eig_abs_floor=1e-18)
    mc_base = sigma_m2_tilde_astrometric_msun(sol, **kw, **mc_kw)
    mc_m1 = sigma_m2_tilde_astrometric_msun(sol, sigma_log10_m1=0.04, **kw, **mc_kw)
    assert mc_m1 > mc_base
    assert mc_m1 == pytest.approx(with_m1, rel=0.1)


def test_analytic_sigma_undefined_cases_return_none() -> None:
    sol = synthetic_orbital_solution(seed=1)
    assert sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=0.0) is None
    values = list(sol.values)
    values[2] = -1.0  # parallax
    bad = ParameterSet(
        names=list(sol.names),
        values=values,
        covariance=sol.covariance,
        provenance="bad",
    )
    assert sigma_m2_tilde_analytic_msun(bad, m1_tilde_msun=1.0) is None


def test_janssens_sigma_log10_mass_formula() -> None:
    mg = 3.5
    result = invert_mg_to_mass(mg)
    seg = segments_from_table()[result.segment_index]
    log_m = np.log10(result.mass_msun)
    expected = np.hypot(seg.b_err / seg.a, log_m * seg.a_err / seg.a)
    assert sigma_log10_mass_from_fit(mg) == pytest.approx(expected, rel=1e-12)
    rho = sigma_log10_mass_from_fit(mg, ab_correlation=0.5)
    assert rho != pytest.approx(expected, rel=1e-9)
    assert sigma_log10_mass_from_fit(-20.0) is None


# --- provenance + stage wiring -------------------------------------------


def test_sigma_provenance_tags_name_method_and_m1() -> None:
    assert (
        sigma_provenance_tag("analytic", m1_uncertainty=False, analytic_covariance="full")
        == "elbadry2026_analytic_full_m1_tilde_fixed"
    )
    assert (
        sigma_provenance_tag("monte_carlo", m1_uncertainty=True)
        == "elbadry2026_mc_m1_tilde_janssens"
    )
    assert is_elbadry_sigma_provenance(LEGACY_MC_FIXED_PROVENANCE)
    assert not is_elbadry_sigma_provenance("andrews2022")
    kept = clear_non_elbadry_m2_astrometric_sigma(
        {
            "sigma_m2_astrometric_msun": 0.03,
            "sigma_m2_msun": 0.03,
            "_sigma_m2_astrometric_provenance": "elbadry2026_analytic_full_m1_tilde_fixed",
        }
    )
    assert kept["sigma_m2_astrometric_msun"] == 0.03


@pytest.mark.parametrize(
    ("method", "propagate", "tag"),
    [
        ("analytic", False, "elbadry2026_analytic_full_m1_tilde_fixed"),
        ("monte_carlo", False, "elbadry2026_mc_m1_tilde_fixed"),
        ("analytic", True, "elbadry2026_analytic_full_m1_tilde_janssens"),
    ],
)
def test_attach_sigma_dispatches_on_config(
    monkeypatch: pytest.MonkeyPatch, method: str, propagate: bool, tag: str
) -> None:
    import darkhunter_pop.data_acquisition as da

    sol = synthetic_orbital_solution(relative_error=0.01, seed=4)
    monkeypatch.setattr(
        da,
        "reconstruct_nss_covariance",
        lambda mapping, nss_solution_type=None: SimpleNamespace(parameter_set=sol),
    )
    monkeypatch.setattr(da, "merge_nss_enrichment_into_row", lambda row, extra: dict(row))
    base = _spec()
    spec = base.model_copy(
        update={
            "sigma_m2_tilde": SigmaM2TildeSpec(method=method),
            "primary_mass": base.primary_mass.model_copy(
                update={"propagate_fit_uncertainty": propagate}
            ),
            "monte_carlo": base.monte_carlo.model_copy(update={"n_draws": 2000}),
        }
    )
    sel = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION)
    row = {"source_id": 7, "nss_solution_type": "Orbital", "m1_tilde_msun": 1.1, "mg_0": 3.5}
    monkeypatch.setattr(
        sel, "_nss_enrichment_index", lambda: {da._enrichment_join_key(row): {}}
    )
    rows = [dict(row)]
    sel._attach_elbadry_m2_sigma_inplace(rows)
    assert rows[0]["_sigma_m2_astrometric_provenance"] == tag
    sigma = rows[0]["sigma_m2_astrometric_msun"]
    if method == "analytic" and not propagate:
        assert sigma == pytest.approx(
            sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=1.1), rel=1e-12
        )
    if propagate:
        assert sigma > sigma_m2_tilde_analytic_msun(sol, m1_tilde_msun=1.1)
