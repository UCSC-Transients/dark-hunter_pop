"""Tests for mass_derivation_bulk and mass_derivation_refined."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import MassCalibrationMethod
from darkhunter_pop.mass_derivation import (
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
    resolve_atmosphere,
    run_bulk_on_candidates,
    run_mass_derivation_bulk,
    run_mass_derivation_refined,
    run_refined_on_candidates,
    tag10_log_mass_radius,
    write_stage_hdf5,
)
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
