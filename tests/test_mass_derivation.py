"""Tests for mass_derivation_bulk and mass_derivation_refined."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import darkhunter_pop.mass_derivation as mass_derivation
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import MassCalibrationMethod
from darkhunter_pop.mass_derivation import (
    SED_UNAVAILABLE_SKIP_REASON,
    BulkDiagnostics,
    BulkFunnel,
    apply_santos_correction,
    approaches_uberms_m1_prior_cap,
    companion_mass_m2,
    derive_tag10_m1_r1,
    format_bulk_funnel_table,
    format_refined_report,
    information_gain_stub,
    parameterset_from_sed_summary,
    passes_m2_mass_cut,
    read_stage_hdf5,
    refined_completion_reason,
    resolve_atmosphere,
    run_bulk_on_candidates,
    run_mass_derivation_bulk,
    run_mass_derivation_refined,
    run_refined_on_candidates,
    sed_unavailable_plan_note,
    tag10_log_mass_radius,
    write_bulk_diagnostic_artifacts,
    write_stage_hdf5,
)
from darkhunter_pop.plotting import matplotlib_available
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    FitTier,
    ParameterSet,
    StageStatus,
    ThieleInnesElements,
)

pytestmark = pytest.mark.unit


class FakeGaiamock:
    """Deterministic stand-in for gaiamock_mod mass helpers (no submodule required)."""

    def get_Campbell_elements(
        self, A: float, B: float, F: float, G: float
    ) -> tuple[float, float, float, float]:
        a0 = float(np.sqrt(A * A + B * B + F * F + G * G) / 2.0)
        return a0, 0.0, 0.0, 0.0

    def get_companion_mass_from_mass_function(
        self,
        M1: float,
        a0_mas: float,
        period: float,
        parallax: float,
        fluxratio: float,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> float:
        # Monotonic in a0 so bulk cut tests can separate high/low photocenter stars.
        _ = (period, parallax, fluxratio, tol, max_iter)
        return float(0.2 + 0.4 * M1 + 2.0 * a0_mas)


def _sunlike_extras(**overrides: float) -> dict[str, float]:
    base = {
        "teff_msc1": 5772.0,
        "teff_msc1_error": 50.0,
        "logg_msc1": 4.44,
        "logg_msc1_error": 0.05,
        "mh_msc": 0.0,
        "mh_msc_error": 0.05,
    }
    base.update(overrides)
    return base


def _candidate(
    source_id: int = 1001,
    *,
    extras: dict | None = None,
    m2_boost_a0: bool = True,
) -> CandidateRecord:
    # Large a0 / short period → high M2 so the default M_min cut keeps the star.
    scale = 5.0 if m2_boost_a0 else 0.05
    return CandidateRecord(
        source_id=source_id,
        parallax_mas=10.0,
        thiele_innes=ThieleInnesElements(
            A=scale, B=scale, F=scale * 0.5, G=scale * 0.5
        ),
        nss_orbital={"period": 200.0, "parallax": 10.0},
        extras=extras if extras is not None else _sunlike_extras(),
    )


def test_tag10_solar_like_near_one_msun() -> None:
    log_m, log_r = tag10_log_mass_radius(5772.0, 4.44, 0.0)
    m = 10**log_m
    r = 10**log_r
    assert 0.7 < m < 1.4
    assert 0.7 < r < 1.5


def test_santos_correction_matches_constants_quadratic() -> None:
    from darkhunter_pop import constants as C

    m = 1.2
    m_corr, _ = apply_santos_correction(m, 0.1)
    expected = C.SANTOS2013_S2 * m**2 + C.SANTOS2013_S1 * m + C.SANTOS2013_S0
    assert m_corr == pytest.approx(expected)


def test_resolve_atmosphere_prefers_msc_over_gspphot() -> None:
    cand = _candidate(
        extras={
            "teff_msc1": 6000.0,
            "logg_msc1": 4.3,
            "mh_msc": -0.1,
            "teff_gspphot": 5000.0,
            "logg_gspphot": 4.0,
            "mh_gspphot": 0.2,
        }
    )
    atm = resolve_atmosphere(cand)
    assert atm is not None
    assert atm.source == "MSC"
    assert atm.teff_k == pytest.approx(6000.0)


def test_resolve_atmosphere_falls_back_to_gspphot() -> None:
    cand = _candidate(
        extras={
            "teff_gspphot": 5500.0,
            "logg_gspphot": 4.2,
            "mh_gspphot": 0.0,
        }
    )
    atm = resolve_atmosphere(cand)
    assert atm is not None
    assert atm.source == "gspphot"


def test_derive_tag10_uses_config_scatter_not_hardcoded() -> None:
    cfg = load_config()
    atm = resolve_atmosphere(_candidate())
    assert atm is not None
    ps = derive_tag10_m1_r1(atm, cfg)
    assert "TAG10" in ps.provenance
    assert "MSC" in ps.provenance
    if cfg.mass_calibration.santos_correction:
        assert "Santos2013" in ps.provenance
    m1 = ps.marginal("M1")
    assert m1.sigma is not None and m1.sigma > 0


def test_unimplemented_method_raises() -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "mass_calibration": cfg.mass_calibration.model_copy(
                update={"method": MassCalibrationMethod.EKER}
            )
        }
    )
    atm = resolve_atmosphere(_candidate())
    assert atm is not None
    with pytest.raises(NotImplementedError, match="Eker"):
        derive_tag10_m1_r1(atm, cfg)


def test_m2_cut_uses_config_n_sigma_and_m_min() -> None:
    assert passes_m2_mass_cut(1.0, 0.1, m_min_msun=1.1, n_sigma=2.0) is True
    assert passes_m2_mass_cut(1.0, 0.01, m_min_msun=1.1, n_sigma=2.0) is False


def test_bulk_pipeline_keeps_high_m2_rejects_low() -> None:
    cfg = load_config()
    gaiamock = FakeGaiamock()
    keepers, diag = run_bulk_on_candidates(
        [_candidate(1, m2_boost_a0=True), _candidate(2, m2_boost_a0=False)],
        cfg,
        gaiamock=gaiamock,
    )
    assert diag.funnel.input_candidates == 2
    assert diag.funnel.after_m2_cut == 1
    assert len(keepers) == 1
    assert keepers[0].source_id == 1
    assert keepers[0].fit_tier is FitTier.BULK_ESTIMATE
    assert keepers[0].m1 is not None and keepers[0].m2 is not None
    text = format_bulk_funnel_table(diag)
    assert "after_m2_cut" in text


def test_bulk_stream_matches_list_on_fixture() -> None:
    cfg = load_config()
    gaiamock = FakeGaiamock()
    candidates = [
        _candidate(1, m2_boost_a0=True),
        _candidate(2, m2_boost_a0=False),
        _candidate(
            3,
            extras={
                "teff_gspphot": 5500.0,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.0,
            },
            m2_boost_a0=True,
        ),
    ]
    list_keepers, list_diag = run_bulk_on_candidates(
        candidates, cfg, gaiamock=gaiamock
    )
    stream_keepers, stream_diag = run_bulk_on_candidates(
        iter(candidates), cfg, gaiamock=gaiamock
    )
    assert list_diag.funnel == stream_diag.funnel
    assert np.allclose(list_diag.m2_pre_cut_msun, stream_diag.m2_pre_cut_msun)
    assert np.allclose(list_diag.m2_post_cut_msun, stream_diag.m2_post_cut_msun)
    assert [c.source_id for c in list_keepers] == [
        c.source_id for c in stream_keepers
    ]


def test_companion_mass_parameterset_provenance() -> None:
    ps = companion_mass_m2(
        1.0,
        a0_mas=1.0,
        period_day=300.0,
        parallax_mas=5.0,
        flux_ratio=0.0,
        gaiamock=FakeGaiamock(),
        sigma_m1_msun=0.05,
    )
    assert ps.names == ["M2"]
    assert "gaiamock" in ps.provenance
    assert ps.marginal("M2").sigma is not None


def test_watchlist_uses_config_fraction_not_hardcoded_3() -> None:
    cfg = load_config()
    cap = cfg.mass_derivation.uberms_m1_prior_max_msun
    frac = cfg.mass_derivation.uberms_m1_watchlist_fraction
    assert approaches_uberms_m1_prior_cap(frac * cap, cfg) is True
    assert approaches_uberms_m1_prior_cap(frac * cap * 0.5, cfg) is False


def test_information_gain_orders_by_relative_sigma() -> None:
    cfg = load_config()
    low = CandidateRecord(
        source_id=1,
        m1=ParameterSet(
            names=["M1"], values=[1.0], covariance=[[0.01]], provenance="TAG10"
        ),
    )
    high = CandidateRecord(
        source_id=2,
        m1=ParameterSet(
            names=["M1"], values=[1.0], covariance=[[0.25]], provenance="TAG10"
        ),
    )
    assert information_gain_stub(high, cfg) > information_gain_stub(low, cfg)


def test_parameterset_from_sed_summary() -> None:
    doc = {
        "m1_msun": {"median": 1.25, "p16": 1.1, "p84": 1.4},
        "fits": {
            "ums": {
                "parameters": {
                    "log(R)": {"median": 0.0, "p16": -0.05, "p84": 0.05},
                }
            }
        },
    }
    ps = parameterset_from_sed_summary(doc)
    assert ps is not None
    assert ps.provenance == "uberMS"
    assert ps.marginal("M1").value == pytest.approx(1.25)
    assert "R1" in ps.names


def test_refined_queue_cache_and_watchlist() -> None:
    cfg = load_config()
    bulk_m1 = ParameterSet(
        names=["M1", "R1"],
        values=[2.95, 2.0],
        covariance=[[0.01, 0.0], [0.0, 0.01]],
        provenance="TAG10+MSC",
        units=["Msun", "Rsun"],
    )
    cand = CandidateRecord(
        source_id=42,
        m1=bulk_m1,
        fit_tier=FitTier.BULK_ESTIMATE,
    )

    summaries = {
        42: {
            "gaia_source_id": "42",
            "m1_msun": {"median": 2.96, "p16": 2.9, "p84": 3.0},
        }
    }

    def loader(sid: int):
        return summaries.get(sid)

    def needs_update(sid: int):
        return False, "up to date"

    def fit(_sid: int):
        raise AssertionError("should not fit when cached")

    out, diag = run_refined_on_candidates(
        [cand],
        cfg,
        summary_loader=loader,
        needs_update_fn=needs_update,
        fit_fn=fit,
    )
    assert diag.fit_cached == 1
    assert diag.fit_succeeded == 1
    assert out[0].fit_tier is FitTier.FULL_UBERMS
    assert 42 in diag.watchlist_source_ids
    assert format_refined_report(diag).startswith("mass_derivation_refined")


# --- Issue #181: loud SED-unavailable degraded path (never silent) ---------


def _unrefined_candidate(source_id: int) -> CandidateRecord:
    return CandidateRecord(
        source_id=source_id,
        m1=ParameterSet(
            names=["M1"], values=[1.0], covariance=[[0.01]], provenance="TAG10"
        ),
        fit_tier=FitTier.BULK_ESTIMATE,
    )


def test_refined_diagnostics_loud_when_sed_unavailable() -> None:
    """No pre-staged snapshot + SED unavailable: skip is recorded, never silent."""
    cfg = load_config()
    cand = _unrefined_candidate(1)

    out, diag = run_refined_on_candidates(
        [cand],
        cfg,
        summary_loader=lambda _sid: None,
        needs_update_fn=lambda _sid: (True, "no snapshot"),
        fit_fn=lambda _sid: None,
        sed_package_available=False,
    )

    assert diag.sed_package_available is False
    assert diag.fit_succeeded == 0
    assert out[0].fit_tier is FitTier.BULK_ESTIMATE  # unrefined, kept bulk M1

    report = format_refined_report(diag)
    assert "SED PACKAGE UNAVAILABLE" in report
    assert "1/1 queued candidate(s) kept" in report

    reason = refined_completion_reason(diag)
    assert reason is not None
    assert reason.startswith(SED_UNAVAILABLE_SKIP_REASON)
    assert "1/1" in reason


def test_refined_diagnostics_not_misleading_when_snapshot_covers_gap() -> None:
    """SED package unavailable but every candidate had a staged snapshot: no
    candidate was actually left unrefined, so the completion reason (which
    drives the run-file record) must not falsely claim a skip happened."""
    cfg = load_config()
    cand = _unrefined_candidate(2)
    doc = {"m1_msun": {"median": 1.3, "p16": 1.2, "p84": 1.4}}

    out, diag = run_refined_on_candidates(
        [cand],
        cfg,
        summary_loader=lambda _sid: doc,
        needs_update_fn=lambda _sid: (False, "up to date"),
        fit_fn=lambda _sid: (_ for _ in ()).throw(AssertionError("should not fit")),
        sed_package_available=False,
    )

    assert diag.sed_package_available is False
    assert diag.fit_succeeded == 1
    assert out[0].fit_tier is FitTier.FULL_UBERMS

    report = format_refined_report(diag)
    assert "SED PACKAGE UNAVAILABLE" in report  # the fact is still visible
    assert "All 1 queued candidate(s) were refined" in report

    # No candidate was actually left unrefined, so nothing loud belongs on the
    # run-file record for this run.
    assert refined_completion_reason(diag) is None


def test_refined_diagnostics_quiet_when_sed_available() -> None:
    """Baseline: SED package available, normal behavior is unaffected."""
    cfg = load_config()
    cand = _unrefined_candidate(3)
    doc = {"m1_msun": {"median": 1.3, "p16": 1.2, "p84": 1.4}}

    _out, diag = run_refined_on_candidates(
        [cand],
        cfg,
        summary_loader=lambda _sid: doc,
        needs_update_fn=lambda _sid: (False, "up to date"),
        fit_fn=lambda _sid: (_ for _ in ()).throw(AssertionError("should not fit")),
        sed_package_available=True,
    )

    assert diag.sed_package_available is True
    report = format_refined_report(diag)
    assert "SED PACKAGE UNAVAILABLE" not in report
    assert refined_completion_reason(diag) is None


def test_sed_unavailable_plan_note_none_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan-time note is silent when darkhunter_sed is importable."""
    monkeypatch.setattr(mass_derivation, "_SED_AVAILABLE", True)
    cfg = load_config()
    assert sed_unavailable_plan_note(cfg) is None


def test_sed_unavailable_plan_note_degraded_vs_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan-time note distinguishes degrade (default) vs refuse (require_sed_package=true)."""
    monkeypatch.setattr(mass_derivation, "_SED_AVAILABLE", False)
    cfg = load_config()

    degraded_note = sed_unavailable_plan_note(cfg)
    assert degraded_note is not None
    assert "DEGRADED" in degraded_note

    strict_cfg = cfg.model_copy(
        update={
            "mass_derivation": cfg.mass_derivation.model_copy(
                update={"require_sed_package": True}
            )
        }
    )
    refuse_note = sed_unavailable_plan_note(strict_cfg)
    assert refuse_note is not None
    assert "REFUSE" in refuse_note


def test_build_stage_plan_surfaces_sed_unavailable_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The printed run plan (--dry-run and the real run) carries the note, not
    just the artifact after the fact (issue #181)."""
    monkeypatch.setattr(mass_derivation, "_SED_AVAILABLE", False)
    from darkhunter_pop.pipeline import build_stage_plan
    from darkhunter_pop.run_management import create_run_manifest

    cfg = load_config()
    manifest = create_run_manifest(cfg)
    plan = build_stage_plan(
        manifest, cfg, stage_subset=["data_acquisition", "mass_derivation_refined"]
    )
    refined_entry = next(e for e in plan if e.stage == "mass_derivation_refined")
    assert "SED UNAVAILABLE" in refined_entry.detail
    daq_entry = next(e for e in plan if e.stage == "data_acquisition")
    assert "SED UNAVAILABLE" not in daq_entry.detail


def test_run_mass_derivation_refined_records_loud_reason_on_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end stage runner: when SED is unavailable and a candidate has no
    pre-staged snapshot, the run-file StageRecord.reason and the HDF5
    diagnostics attr both carry the loud flag — never a silent COMPLETED."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mass_derivation, "_SED_AVAILABLE", False)

    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"artifact_root": str(tmp_path / "output")}
            )
        }
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    manifest = run_mass_derivation_refined(
        manifest,
        cfg,
        run_path=run_path,
        candidates=[_unrefined_candidate(99)],
        summary_loader=lambda _sid: None,
        needs_update_fn=lambda _sid: (True, "no snapshot"),
        fit_fn=lambda _sid: None,
    )

    ref = manifest.stages["mass_derivation_refined"]
    assert ref.status is StageStatus.COMPLETED
    assert ref.reason is not None
    assert ref.reason.startswith(SED_UNAVAILABLE_SKIP_REASON)

    refined, _ = read_stage_hdf5(Path(ref.artifact_path))
    assert refined[0].fit_tier is FitTier.BULK_ESTIMATE  # unrefined, not silently kept as if refined

    import h5py

    with h5py.File(ref.artifact_path, "r") as handle:
        assert handle["diagnostics"].attrs["sed_package_available"] == False  # noqa: E712


def test_stage_runners_write_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Point artifact_root / runs into tmp via config override.
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"artifact_root": str(tmp_path / "output")}
            )
        }
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = create_run_manifest(cfg)
    run_path = runs / f"{manifest.run_id}.yaml"
    save_run_manifest(manifest, run_path)

    # Seed a fake completed data_acquisition artifact on the manifest.
    daq_spec = STAGE_REGISTRY["data_acquisition"]
    daq_path = stage_artifact_path(cfg, daq_spec, run_id=manifest.run_id)
    from darkhunter_pop.data_acquisition import (
        FunnelCounts,
        SnapshotMeta,
        compute_stage_diagnostics,
        write_stage_hdf5 as write_daq,
    )
    from datetime import datetime, timezone

    candidates = [_candidate(7)]
    snapshot = SnapshotMeta(
        snapshot_id="t",
        query_date=datetime.now(tz=timezone.utc),
        adql="SELECT 1",
        checksum="abc",
        row_count=1,
        result_path=tmp_path / "q.ecsv",
        meta_path=tmp_path / "m.yaml",
    )
    diagnostics = compute_stage_diagnostics(
        candidates,
        funnel=FunnelCounts(queried=1, after_quality_cut=1, candidates_written=1),
        quality_cut_bin_counts={},
    )
    write_daq(daq_path, candidates, snapshot=snapshot, diagnostics=diagnostics)
    from darkhunter_pop.schemas import StageRecord

    manifest = manifest.model_copy(
        update={
            "stages": {
                "data_acquisition": StageRecord(
                    stage_name="data_acquisition",
                    status=StageStatus.COMPLETED,
                    artifact_path=str(daq_path),
                )
            }
        }
    )
    save_run_manifest(manifest, run_path)

    manifest = run_mass_derivation_bulk(
        manifest,
        cfg,
        run_path=run_path,
        gaiamock=FakeGaiamock(),
    )
    bulk_rec = manifest.stages["mass_derivation_bulk"]
    assert bulk_rec.status is StageStatus.COMPLETED
    assert bulk_rec.artifact_path is not None
    loaded, meta = read_stage_hdf5(Path(bulk_rec.artifact_path))
    assert meta["stage"] == "mass_derivation_bulk"
    assert len(loaded) == 1

    def loader(sid: int):
        return {"m1_msun": {"median": 1.1, "p16": 1.0, "p84": 1.2}}

    manifest = run_mass_derivation_refined(
        manifest,
        cfg,
        run_path=run_path,
        summary_loader=loader,
        needs_update_fn=lambda _sid: (False, "up to date"),
        fit_fn=lambda _sid: None,
    )
    ref = manifest.stages["mass_derivation_refined"]
    assert ref.status is StageStatus.COMPLETED
    refined, _ = read_stage_hdf5(Path(ref.artifact_path))
    assert refined[0].fit_tier is FitTier.FULL_UBERMS


def test_sed_summary_root_fixture_loader() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_derivation.sed_summary_root = "tests/fixtures/sed_summaries"
    from darkhunter_pop.mass_derivation import (
        _sed_summary_loader_for_config,
        parameterset_from_sed_summary,
    )

    doc = _sed_summary_loader_for_config(tweaked)(515151)
    assert doc is not None
    ps = parameterset_from_sed_summary(doc)
    assert ps is not None
    assert ps.marginal("M1").value == pytest.approx(1.2)


def test_default_sed_summary_paths_match_upstream() -> None:
    cfg = load_config()
    assert cfg.mass_derivation.sed_summary_root == "data/sed_summaries"
    assert (
        cfg.mass_derivation.sed_summary_filename_template
        == "Gaia_DR3_{source_id}_sed_summary.json"
    )


def test_refined_consumes_fixture_summary_sets_full_uberms() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_derivation.sed_summary_root = "tests/fixtures/sed_summaries"
    cand = CandidateRecord(
        source_id=515151,
        m1=ParameterSet(
            names=["M1"], values=[1.0], covariance=[[0.01]], provenance="TAG10"
        ),
        fit_tier=FitTier.BULK_ESTIMATE,
    )
    updated, diag = run_refined_on_candidates([cand], tweaked)
    assert diag.fit_succeeded == 1
    assert updated[0].fit_tier is FitTier.FULL_UBERMS
    assert updated[0].m1 is not None
    assert updated[0].m1.marginal("M1").value == pytest.approx(1.2)


def test_refined_missing_summary_queues_fit() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_derivation.sed_summary_root = "tests/fixtures/sed_summaries"
    calls: list[int] = []

    def needs_update(sid: int) -> tuple[bool, str]:
        # Mirror production: snapshot miss → package says needs update.
        return True, "no prior sed_summary"

    def fit(sid: int) -> dict:
        calls.append(sid)
        # Keep M1 near the uberMS prior cap so watch-list still fires post-fit.
        return {"m1_msun": {"median": 2.92, "p16": 2.9, "p84": 2.94}}

    cand = CandidateRecord(
        source_id=999001,
        m1=ParameterSet(
            names=["M1"], values=[2.9], covariance=[[0.01]], provenance="TAG10"
        ),
        fit_tier=FitTier.BULK_ESTIMATE,
    )
    updated, diag = run_refined_on_candidates(
        [cand],
        tweaked,
        needs_update_fn=needs_update,
        fit_fn=fit,
    )
    assert calls == [999001]
    assert diag.fit_attempted == 1
    assert diag.fit_succeeded == 1
    assert updated[0].fit_tier is FitTier.FULL_UBERMS
    assert 999001 in diag.watchlist_source_ids


def test_refined_watchlist_queued_before_others() -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_derivation.sed_summary_root = None
    order: list[int] = []

    def needs_update(sid: int) -> tuple[bool, str]:
        return True, "force"

    def fit(sid: int) -> dict:
        order.append(sid)
        return {"m1_msun": {"median": 1.0, "p16": 0.9, "p84": 1.1}}

    low = CandidateRecord(
        source_id=1,
        m1=ParameterSet(
            names=["M1"], values=[1.0], covariance=[[0.25]], provenance="TAG10"
        ),
        fit_tier=FitTier.BULK_ESTIMATE,
    )
    watch = CandidateRecord(
        source_id=2,
        m1=ParameterSet(
            names=["M1"], values=[2.95], covariance=[[0.01]], provenance="TAG10"
        ),
        fit_tier=FitTier.BULK_ESTIMATE,
    )
    _, diag = run_refined_on_candidates(
        [low, watch],
        tweaked,
        needs_update_fn=needs_update,
        fit_fn=fit,
    )
    assert order[0] == 2
    assert list(diag.information_gain_order)[0] == 2


def test_sed_needs_update_uses_snapshot_then_falls_through() -> None:
    from darkhunter_pop.mass_derivation import _sed_needs_update_for_config

    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_derivation.sed_summary_root = "tests/fixtures/sed_summaries"
    needs = _sed_needs_update_for_config(tweaked)
    assert needs(515151) == (False, "up to date")
    should_run, reason = needs(999001)
    # Package may be absent in CI; either way we must not hard-stop as sed_summary_missing.
    assert reason != "sed_summary_missing"
    assert should_run is False or reason in {
        "no prior sed_summary",
        "darkhunter_sed_unavailable",
    }


def test_hdf5_round_trip(tmp_path: Path) -> None:
    cand = _candidate()
    cand = cand.model_copy(
        update={
            "m1": ParameterSet(
                names=["M1"], values=[1.0], covariance=[[0.01]], provenance="TAG10"
            ),
            "fit_tier": FitTier.BULK_ESTIMATE,
        }
    )
    path = tmp_path / "out.h5"
    write_stage_hdf5(
        path,
        [cand],
        stage_name="mass_derivation_bulk",
        diagnostics={"after_m2_cut": 1, "m2_pre_cut_msun": np.array([1.5])},
    )
    loaded, meta = read_stage_hdf5(path)
    assert loaded[0].source_id == cand.source_id
    assert meta["n_candidates"] == 1


def _sample_bulk_diagnostics() -> BulkDiagnostics:
    return BulkDiagnostics(
        funnel=BulkFunnel(
            input_candidates=3,
            atmosphere_ok=3,
            m1_ok=3,
            m2_ok=2,
            after_m2_cut=2,
            skipped_no_atmosphere=0,
            skipped_no_orbit=0,
            skipped_m2_failed=1,
        ),
        m2_pre_cut_msun=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        m2_post_cut_msun=np.array([0.5, 1.0], dtype=np.float64),
    )


def test_write_bulk_diagnostic_artifacts_respects_write_figures_flag(
    tmp_path: Path,
) -> None:
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "diagnostics": cfg.diagnostics.model_copy(update={"write_figures": False}),
        }
    )
    artifact = tmp_path / "mass_derivation_bulk.h5"
    artifact.write_bytes(b"")
    written = write_bulk_diagnostic_artifacts(_sample_bulk_diagnostics(), artifact, cfg)
    assert any(p.name == "funnel.txt" for p in written)
    assert not any(p.suffix == ".png" for p in written)


@pytest.mark.skipif(not matplotlib_available(), reason="matplotlib optional")
def test_write_bulk_diagnostic_artifacts_uses_shared_plotting(tmp_path: Path) -> None:
    cfg = load_config()
    artifact = tmp_path / "mass_derivation_bulk.h5"
    artifact.write_bytes(b"")
    written = write_bulk_diagnostic_artifacts(_sample_bulk_diagnostics(), artifact, cfg)
    pngs = [p for p in written if p.suffix == ".png"]
    assert {p.name for p in pngs} == {"m2_pre_cut.png", "m2_post_cut.png"}
    diag_dir = tmp_path / "mass_derivation_bulk_diagnostics"
    assert diag_dir.is_dir()
    for path in pngs:
        assert path.is_file() and path.stat().st_size > 0


def test_write_bulk_diagnostic_artifacts_m2_histogram_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    def _capture_hist(
        values: object, path: Path, /, **kwargs: object
    ) -> Path:
        captured.append({"path": path, **kwargs})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr("darkhunter_pop.mass_derivation.plot_histogram", _capture_hist)
    cfg = load_config()
    artifact = tmp_path / "mass_derivation_bulk.h5"
    artifact.write_bytes(b"")
    write_bulk_diagnostic_artifacts(_sample_bulk_diagnostics(), artifact, cfg)
    m2_calls = [c for c in captured if "m2_" in c["path"].name]
    assert len(m2_calls) == 2
    for call in m2_calls:
        assert call.get("xlim") == (0.0, 30.0)
        assert call.get("log_y") is True


def test_write_bulk_diagnostic_artifacts_m2_histogram_axes_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    def _capture_hist(
        values: object, path: Path, /, **kwargs: object
    ) -> Path:
        captured.append({"path": path, **kwargs})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr("darkhunter_pop.mass_derivation.plot_histogram", _capture_hist)
    cfg = load_config()
    cfg = cfg.model_copy(
        update={
            "mass_derivation": cfg.mass_derivation.model_copy(
                update={
                    "bulk_m2_histogram_xmin_msun": 0.5,
                    "bulk_m2_histogram_xmax_msun": 10.0,
                    "bulk_m2_histogram_log_y": False,
                }
            ),
        }
    )
    artifact = tmp_path / "mass_derivation_bulk.h5"
    artifact.write_bytes(b"")
    write_bulk_diagnostic_artifacts(_sample_bulk_diagnostics(), artifact, cfg)
    m2_calls = [c for c in captured if "m2_" in c["path"].name]
    assert len(m2_calls) == 2
    for call in m2_calls:
        assert call.get("xlim") == (0.5, 10.0)
        assert call.get("log_y") is False
