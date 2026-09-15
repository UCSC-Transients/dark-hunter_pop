"""Tests for the ``phot_sed`` adapter (issue #197).

Fixture matrix lives in ``tests/fixtures/phot_sed/`` (see its README).
"""

from __future__ import annotations

import math

import pytest

from darkhunter_pop.companion_nature import (
    run_companion_nature_on_candidates,
)
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.phot_sed_adapter import (
    MODEL_TO_HYPOTHESIS,
    PROVENANCE_ANALYTIC_FALLBACK,
    PROVENANCE_PHOT_SED,
    PROVENANCE_PRECOMPUTED_EXTRAS,
    attach_phot_sed_evidence,
    chi2_from_bic,
    evidence_provenance,
    load_phot_sed_evidence,
    resolve_phot_sed_summary_path,
)
from darkhunter_pop.schemas import CandidateRecord, ParameterSet

pytestmark = pytest.mark.unit

FIXTURE_ROOT = "tests/fixtures/phot_sed"


def _config(root: str | None = FIXTURE_ROOT) -> PipelineConfig:
    cfg = load_config().model_copy(deep=True)
    cfg.mass_derivation.phot_sed_root = root
    return cfg


def _candidate(source_id: int, **extras: object) -> CandidateRecord:
    m1 = ParameterSet(
        names=["M1"],
        values=[1.2],
        covariance=[[0.01]],
        provenance="TAG10",
        units=["Msun"],
    )
    m2 = ParameterSet(
        names=["M2"],
        values=[0.6],
        covariance=[[0.04]],
        provenance="gaiamock_mass_function+TAG10_M1",
        units=["Msun"],
    )
    return CandidateRecord(
        source_id=source_id,
        m1=m1,
        m2=m2,
        parallax_mas=2.0,
        extras=dict(extras),
    )


# ---------------------------------------------------------------------------
# Config + path resolution
# ---------------------------------------------------------------------------


def test_default_phot_sed_paths_match_upstream_layout() -> None:
    cfg = load_config()
    assert cfg.mass_derivation.phot_sed_root == "data/phot_sed"
    assert (
        cfg.mass_derivation.phot_sed_filename_template
        == "Gaia_DR3_{source_id}_{model}_summary.json"
    )


def test_resolve_path_uses_template_and_is_none_when_disabled() -> None:
    cfg = _config()
    path = resolve_phot_sed_summary_path(cfg, 900001, "wd")
    assert path is not None
    assert path.name == "Gaia_DR3_900001_wd_summary.json"
    assert resolve_phot_sed_summary_path(_config(None), 900001, "wd") is None


def test_resolve_path_rejects_unknown_model() -> None:
    with pytest.raises(KeyError):
        resolve_phot_sed_summary_path(_config(), 900001, "3star")


def test_model_to_hypothesis_mapping_is_the_confirmed_one() -> None:
    assert MODEL_TO_HYPOTHESIS == {"1star": "dark", "wd": "WD", "2star": "other"}


# ---------------------------------------------------------------------------
# BIC → chi2
# ---------------------------------------------------------------------------


def test_chi2_from_bic_inverts_upstream_definition() -> None:
    # dark-hunter_sed: BIC = k ln n - 2 ln L_max
    ln_l_max, n_free, n_data = -32.6, 6, 23
    bic = n_free * math.log(n_data) - 2.0 * ln_l_max
    assert chi2_from_bic(bic=bic, n_free=n_free, n_data=n_data) == pytest.approx(
        -2.0 * ln_l_max
    )


@pytest.mark.parametrize(("n_free", "n_data"), [(0, 20), (6, 0)])
def test_chi2_from_bic_guards_match_upstream(n_free: int, n_data: int) -> None:
    with pytest.raises(ValueError):
        chi2_from_bic(bic=10.0, n_free=n_free, n_data=n_data)


# ---------------------------------------------------------------------------
# Fixture matrix
# ---------------------------------------------------------------------------


def test_all_three_models_present_populate_the_chi2_triple() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900001), cfg)
    assert ev.usable is True
    assert set(ev.summaries) == {"dark", "WD", "other"}
    assert ev.n_data == 20
    assert ev.chi2_offset == 0.0

    cn = cfg.companion_nature
    assert cand.extras[cn.phot_n_data_key] == 20
    assert cand.extras[cn.phot_chi2_dark_key] == pytest.approx(
        chi2_from_bic(bic=100.0, n_free=6, n_data=20)
    )
    assert cand.extras[cn.phot_chi2_wd_key] == pytest.approx(
        chi2_from_bic(bic=60.0, n_free=7, n_data=20)
    )
    assert cand.extras[cn.phot_chi2_other_key] == pytest.approx(
        chi2_from_bic(bic=110.0, n_free=9, n_data=20)
    )
    assert evidence_provenance(cand) == PROVENANCE_PHOT_SED


def test_wd_only_is_not_enough_and_nothing_is_imputed() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900002), cfg)
    assert set(ev.summaries) == {"WD"}
    assert ev.usable is False
    assert "missing:1star" in ev.notes
    assert "missing:2star" in ev.notes
    cn = cfg.companion_nature
    for key in (cn.phot_chi2_dark_key, cn.phot_chi2_wd_key, cn.phot_chi2_other_key):
        assert key not in cand.extras
    assert evidence_provenance(cand) == PROVENANCE_ANALYTIC_FALLBACK


def test_no_summaries_at_all_is_labelled_fallback() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900003), cfg)
    assert ev.summaries == {}
    assert ev.usable is False
    assert evidence_provenance(cand) == PROVENANCE_ANALYTIC_FALLBACK


def test_malformed_json_is_refused_not_guessed() -> None:
    cfg = _config()
    _cand, ev = attach_phot_sed_evidence(_candidate(900004), cfg)
    assert "malformed:1star" in ev.notes
    assert "dark" not in ev.summaries
    assert ev.usable is False


def test_source_id_mismatch_is_refused() -> None:
    cfg = _config()
    _cand, ev = attach_phot_sed_evidence(_candidate(900005), cfg)
    assert "unusable:1star" in ev.notes
    assert "dark" not in ev.summaries
    assert ev.usable is False


def test_n_data_mismatch_blocks_the_triple() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900006), cfg)
    assert set(ev.summaries) == {"dark", "WD", "other"}
    assert ev.usable is False
    assert "n_data_mismatch" in ev.notes
    assert cfg.companion_nature.phot_chi2_dark_key not in cand.extras


def test_negative_recovered_chi2_gets_a_common_offset() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900007), cfg)
    assert ev.usable is True
    assert ev.chi2_offset > 0.0
    cn = cfg.companion_nature
    chi2 = [
        cand.extras[cn.phot_chi2_dark_key],
        cand.extras[cn.phot_chi2_wd_key],
        cand.extras[cn.phot_chi2_other_key],
    ]
    assert min(chi2) == pytest.approx(0.0)
    # A common offset preserves every pairwise difference.
    assert chi2[1] - chi2[0] == pytest.approx(
        chi2_from_bic(bic=20.0, n_free=6, n_data=20)
        - chi2_from_bic(bic=5.0, n_free=6, n_data=20)
    )


def test_cleaning_settings_absence_is_reported_per_model() -> None:
    cfg = _config()
    ev = load_phot_sed_evidence(cfg, 900001)
    assert all(not s.cleaning for s in ev.summaries.values())
    for model in ("1star", "wd", "2star"):
        assert f"cleaning_settings_absent:{model}" in ev.notes
    # sigma_int (the fitted error inflation) is recorded where present.
    assert ev.summaries["dark"].sigma_int == pytest.approx(0.05)
    assert ev.as_dict()["models"]["dark"]["cleaning_settings_absent"] is True


def test_preexisting_extras_are_left_alone_and_labelled() -> None:
    cfg = _config()
    cn = cfg.companion_nature
    cand = _candidate(
        900003,
        **{
            cn.phot_chi2_dark_key: 1.0,
            cn.phot_chi2_wd_key: 2.0,
            cn.phot_chi2_other_key: 3.0,
            cn.phot_n_data_key: 5,
        },
    )
    updated, ev = attach_phot_sed_evidence(cand, cfg)
    assert ev.usable is False
    assert updated.extras[cn.phot_chi2_dark_key] == 1.0
    assert evidence_provenance(updated) == PROVENANCE_PRECOMPUTED_EXTRAS


def test_disabled_root_does_no_io_and_labels_fallback() -> None:
    cfg = _config(None)
    cand, ev = attach_phot_sed_evidence(_candidate(900001), cfg)
    assert ev.summaries == {}
    assert ev.notes == ("phot_sed_root_unset",)
    assert evidence_provenance(cand) == PROVENANCE_ANALYTIC_FALLBACK


# ---------------------------------------------------------------------------
# companion_nature integration: coverage split + no-regression
# ---------------------------------------------------------------------------


def test_real_evidence_changes_weights_and_fallback_does_not() -> None:
    cfg = _config()
    with_evidence, _ = run_companion_nature_on_candidates([_candidate(900001)], cfg)
    baseline_cfg = _config(None)
    without, _ = run_companion_nature_on_candidates([_candidate(900001)], baseline_cfg)
    w_evidence = with_evidence[0].companion_nature_weights or {}
    w_plain = without[0].companion_nature_weights or {}
    # Fixture 900001 strongly prefers the WD hypothesis.
    assert w_evidence["WD"] > w_plain["WD"]


def test_bit_identical_weights_for_candidates_without_phot_sed_files() -> None:
    """Regression guard: the adapter must not perturb fallback candidates."""
    cand = _candidate(900003)
    with_adapter, diag = run_companion_nature_on_candidates([cand], _config())
    disabled, _ = run_companion_nature_on_candidates([cand], _config(None))
    a, b = with_adapter[0], disabled[0]
    assert a.companion_nature_weights == b.companion_nature_weights
    assert (
        a.extras["companion_nature_photometric_bic"]
        == b.extras["companion_nature_photometric_bic"]
    )
    assert (
        a.extras["companion_nature_delta_bic_wd_vs_dark"]
        == b.extras["companion_nature_delta_bic_wd_vs_dark"]
    )
    assert a.extras["companion_nature_tier"] == b.extras["companion_nature_tier"]
    assert diag.n_by_evidence_provenance == {PROVENANCE_ANALYTIC_FALLBACK: 1}


def test_diagnostics_report_the_coverage_split_both_ways() -> None:
    cfg = _config()
    candidates = [_candidate(900001), _candidate(900002), _candidate(900003)]
    _out, diag = run_companion_nature_on_candidates(candidates, cfg)
    assert diag.n_by_evidence_provenance[PROVENANCE_PHOT_SED] == 1
    assert diag.n_by_evidence_provenance[PROVENANCE_ANALYTIC_FALLBACK] == 2
    assert diag.phot_sed_model_coverage["WD"] == 2
    assert diag.phot_sed_model_coverage["dark"] == 1
    means = diag.mean_weights_by_evidence_provenance()
    assert set(means) == {PROVENANCE_PHOT_SED, PROVENANCE_ANALYTIC_FALLBACK}
    for tag in means:
        assert 0.0 <= means[tag]["mean_weight_WD"] <= 1.0
        assert 0.0 <= means[tag]["mean_weight_dark"] <= 1.0
    payload = diag.as_dict()
    assert payload["phot_sed_root"].endswith(FIXTURE_ROOT)
    assert "mean_weights_by_evidence_provenance" in payload


def test_report_names_the_split() -> None:
    from darkhunter_pop.companion_nature import format_companion_nature_report

    cfg = _config()
    _out, diag = run_companion_nature_on_candidates(
        [_candidate(900001), _candidate(900003)], cfg
    )
    text = format_companion_nature_report(diag)
    assert "real phot_sed evidence (phot_sed): 1" in text
    assert "analytic fallback (analytic_fallback): 1" in text
    assert "mean weights [phot_sed]" in text


# ---------------------------------------------------------------------------
# Real upstream ``wd`` schema (issue #215, closing #206)
#
# ``dark-hunter_sed#67`` (``feat/wd-pop-summary``) landed
# ``darkhunter_sed.wd_model.write_wd_pop_summary``, which writes the
# pop-facing ``Gaia_DR3_<id>_wd_summary.json`` this adapter was already built
# against (#197). The 900001-900007 fixtures above predate that fix and use a
# placeholder field set. 900008 copies the real writer's field set
# field-for-field (real ``WD_STAR_PARAM_NAMES``, ``n_free: 8``, plus the extra
# ``atm_type``/``ifmr`` keys the real writer adds) to prove the adapter reads
# the actual upstream contract, not just the assumed one.
# ---------------------------------------------------------------------------


def test_real_upstream_wd_schema_parses_and_is_usable() -> None:
    cfg = _config()
    cand, ev = attach_phot_sed_evidence(_candidate(900008), cfg)
    assert ev.usable is True
    assert set(ev.summaries) == {"dark", "WD", "other"}
    assert ev.n_data == 20

    wd_summary = ev.summaries["WD"]
    # Real upstream WD_STAR_PARAM_NAMES has 8 entries -> n_free: 8, distinct
    # from the placeholder fixtures' n_free: 7.
    assert wd_summary.n_free == 8
    assert wd_summary.sigma_int == pytest.approx(0.04)

    cn = cfg.companion_nature
    assert cand.extras[cn.phot_n_data_key] == 20
    assert cand.extras[cn.phot_chi2_wd_key] == pytest.approx(
        chi2_from_bic(bic=27.965858188431927, n_free=8, n_data=20)
    )
    assert evidence_provenance(cand) == PROVENANCE_PHOT_SED


def test_real_upstream_wd_schema_extra_keys_are_ignored_not_asserted() -> None:
    """``atm_type``/``ifmr`` are real upstream keys the adapter never reads."""
    cfg = _config()
    ev = load_phot_sed_evidence(cfg, 900008)
    wd_dict = ev.summaries["WD"].as_dict()
    # The adapter's own record of the summary carries no atm_type/ifmr field —
    # those keys are upstream-only and never round-tripped by this adapter.
    assert "atm_type" not in wd_dict
    assert "ifmr" not in wd_dict
