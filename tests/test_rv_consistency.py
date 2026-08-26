"""Tests for rv_astrometry_gate and joint_orbit_fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.rv_consistency import (
    JOINT_ORBIT_SKIP_REASON,
    collect_rv_epochs,
    extract_astrometric_orbit,
    predicted_k_kms,
    read_stage_hdf5,
    run_gate_on_candidates,
    run_joint_on_candidates,
    run_joint_orbit_fit,
    run_rv_astrometry_gate,
    rv_curve_kms,
    spectroscopic_mass_function_msun,
    t_periastron_to_mjd,
)
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    CandidateRecord,
    OrbitTier,
    ParameterSet,
    StageStatus,
)

pytestmark = pytest.mark.unit


def _diag(cov: float = 0.01) -> list[list[float]]:
    return [[cov]]


def _m1_m2() -> tuple[ParameterSet, ParameterSet]:
    m1 = ParameterSet(
        names=["M1"],
        values=[1.2],
        covariance=_diag(0.01),
        provenance="TAG10",
        units=["Msun"],
    )
    m2 = ParameterSet(
        names=["M2"],
        values=[1.5],
        covariance=_diag(0.04),
        provenance="gaiamock_mass_function+TAG10_M1",
        units=["Msun"],
    )
    return m1, m2


def _synthetic_epochs(
    *,
    period_day: float,
    eccentricity: float,
    t_peri_mjd: float,
    k_kms: float,
    omega_rad: float,
    gamma_kms: float,
    n: int = 12,
    instrument: str = "APF",
    noise: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = t_peri_mjd + np.linspace(0.0, 2.5 * period_day, n)
    y = rv_curve_kms(
        t,
        period_day=period_day,
        eccentricity=eccentricity,
        t_periastron_mjd=t_peri_mjd,
        k_kms=k_kms,
        omega_rad=omega_rad,
        gamma_kms=gamma_kms,
    )
    y = y + rng.normal(0.0, noise, size=n)
    return [
        {
            "file": f"epoch_{i}.txt",
            "basename": f"epoch_{i}",
            "mjd": float(t[i]),
            "rv_kms": float(y[i]),
            "rv_err_kms": noise,
            "wrms_kms": noise,
            "fallback": False,
            "telescope": instrument,
        }
        for i in range(n)
    ]


def _candidate_consistent(*, source_id: int = 1001, sb2: bool = False) -> CandidateRecord:
    period = 200.0
    ecc = 0.2
    t_gaia = 100.0  # days from J2016
    t_mjd = t_periastron_to_mjd(t_gaia)
    omega = 0.7
    k = 15.0
    gamma = 30.0
    m1, m2 = _m1_m2()
    nss = {
        "period_day": period,
        "eccentricity": ecc,
        "t_periastron_day": t_gaia,
        "semi_amp_primary_kms": k,
        "arg_periastron_deg": float(np.rad2deg(omega)),
        "inclination_deg": 60.0,
    }
    if sb2:
        nss["semi_amp_secondary_kms"] = 20.0
        nss["mass_ratio"] = k / 20.0
    epochs = _synthetic_epochs(
        period_day=period,
        eccentricity=ecc,
        t_peri_mjd=t_mjd,
        k_kms=k,
        omega_rad=omega,
        gamma_kms=gamma,
        seed=source_id,
    )
    rv_summary: dict = {
        "schema_version": 1,
        "source_id": source_id,
        "instruments": ["APF"],
        "n_epochs": len(epochs),
        "nss_orbital": nss,
        "pipeline_epochs": epochs,
        "external_rvs": [],
        "joker_fit": {
            "variants": {
                "full": {
                    "fit_variant": "full",
                    "P_days": period,
                    "K_kms": k,
                    "e": ecc,
                    "omega_rad": omega,
                    "t_periastron_mjd": t_mjd,
                    "skip_reason": None,
                }
            }
        },
    }
    if sb2:
        rv_summary["sb2_orbit"] = {
            "period_day": period,
            "eccentricity": ecc,
            "semi_amp_primary_kms": k,
            "semi_amp_secondary_kms": 20.0,
        }
    return CandidateRecord(
        source_id=source_id,
        nss_solution_type="SB2" if sb2 else "Orbital",
        nss_orbital={"period": period, "eccentricity": ecc, "parallax": 5.0},
        rv_summary=rv_summary,
        m1=m1,
        m2=m2,
        orbit_tier=OrbitTier.ASTROMETRY_ONLY,
    )


def _candidate_outlier(*, source_id: int = 2002) -> CandidateRecord:
    """Same elements metadata, but RV curve uses a very different K → gate fail."""
    base = _candidate_consistent(source_id=source_id)
    period = 200.0
    ecc = 0.2
    t_mjd = t_periastron_to_mjd(100.0)
    # Inject epochs from a wildly different orbit.
    bad_epochs = _synthetic_epochs(
        period_day=period,
        eccentricity=ecc,
        t_peri_mjd=t_mjd,
        k_kms=80.0,
        omega_rad=2.5,
        gamma_kms=-10.0,
        noise=0.05,
        seed=99,
    )
    summary = dict(base.rv_summary)
    summary["pipeline_epochs"] = bad_epochs
    summary["n_epochs"] = len(bad_epochs)
    return base.model_copy(update={"rv_summary": summary})


def test_t_periastron_gaia_and_mjd() -> None:
    assert t_periastron_to_mjd(0.0) == pytest.approx(57388.5)
    assert t_periastron_to_mjd(60000.0) == 60000.0


def test_mass_function_roundtrip_k() -> None:
    m1, m2 = 1.2, 1.5
    p, e, inc = 200.0, 0.1, 60.0
    k = predicted_k_kms(m1, m2, p, e, inc)
    assert k is not None and k > 0
    f = spectroscopic_mass_function_msun(p, k, e)
    assert f == pytest.approx((m2 * np.sin(np.deg2rad(inc))) ** 3 / (m1 + m2) ** 2, rel=1e-6)


def test_collect_epochs_and_orbit_extract() -> None:
    cand = _candidate_consistent()
    epochs = collect_rv_epochs(cand.rv_summary)
    assert len(epochs) >= 3
    orbit = extract_astrometric_orbit(cand)
    assert orbit is not None
    assert orbit.period_day == pytest.approx(200.0)
    assert orbit.k_kms == pytest.approx(15.0)


def test_gate_passes_consistent_and_fails_outlier() -> None:
    cfg = load_config()
    good, diag_g = run_gate_on_candidates([_candidate_consistent()], cfg)
    assert diag_g.n_passed == 1
    assert good[0].extras["rv_astrometry_gate"]["passed"] is True
    assert good[0].orbit_tier is OrbitTier.ASTROMETRY_ONLY

    bad, diag_b = run_gate_on_candidates([_candidate_outlier()], cfg)
    assert diag_b.n_failed == 1
    assert bad[0].extras["rv_astrometry_gate"]["passed"] is False


def test_sb2_consistent_unlocks_mass_ratio_channel() -> None:
    cfg = load_config()
    out, diag = run_gate_on_candidates([_candidate_consistent(sb2=True)], cfg)
    assert diag.n_passed == 1
    assert diag.n_sb2_mass_ratio_unlocked == 1
    assert out[0].extras.get("sb2_mass_ratio_channel") is True
    assert out[0].extras.get("sb2_mass_ratio") == pytest.approx(15.0 / 20.0)


def test_sb2_inconsistent_fails_gate() -> None:
    cfg = load_config()
    cand = _candidate_consistent(sb2=True)
    summary = dict(cand.rv_summary)
    summary["sb2_orbit"] = {
        "period_day": 500.0,  # far from 200
        "eccentricity": 0.8,
    }
    cand = cand.model_copy(update={"rv_summary": summary})
    out, diag = run_gate_on_candidates([cand], cfg)
    assert diag.n_failed == 1
    assert out[0].extras.get("sb2_mass_ratio_channel") is False


def test_joint_fit_passers_and_gate_failure_skip() -> None:
    cfg = load_config()
    gated, _ = run_gate_on_candidates(
        [_candidate_consistent(source_id=1), _candidate_outlier(source_id=2)],
        cfg,
    )
    joint, diag = run_joint_on_candidates(gated, cfg)
    assert diag.n_fit == 1
    assert diag.n_skipped_gate_failed == 1
    passer = next(c for c in joint if c.source_id == 1)
    failer = next(c for c in joint if c.source_id == 2)
    assert passer.orbit_tier is OrbitTier.JOINT_ASTROMETRY_RV
    assert "joint_orbit" in passer.extras
    assert passer.m2 is not None
    assert passer.m2.provenance == "joint_astrometry_rv"
    assert failer.orbit_tier is OrbitTier.ASTROMETRY_ONLY
    assert failer.extras.get("joint_orbit_fit_skip_reason") == JOINT_ORBIT_SKIP_REASON


def test_stage_runners_write_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    monkeypatch.chdir(tmp_path)
    # Point artifacts into tmp via config copy.
    cfg = cfg.model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")

    # Seed a fake upstream mass_derivation_refined artifact.
    from darkhunter_pop.mass_derivation import write_stage_hdf5 as write_mass_h5

    manifest = create_run_manifest(cfg)
    run_path = tmp_path / "runs" / f"{manifest.run_id}.yaml"
    run_path.parent.mkdir(parents=True, exist_ok=True)

    upstream_spec = STAGE_REGISTRY["mass_derivation_refined"]
    upstream = stage_artifact_path(cfg, upstream_spec, run_id=manifest.run_id)
    cands = [_candidate_consistent(), _candidate_outlier(source_id=9)]
    write_mass_h5(
        upstream,
        cands,
        stage_name="mass_derivation_refined",
        diagnostics={"n": len(cands)},
    )
    from darkhunter_pop.schemas import StageRecord
    from datetime import datetime, timezone

    stages = dict(manifest.stages)
    stages["mass_derivation_refined"] = StageRecord(
        stage_name="mass_derivation_refined",
        status=StageStatus.COMPLETED,
        artifact_path=str(upstream),
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
    )
    manifest = manifest.model_copy(update={"stages": stages})
    save_run_manifest(manifest, run_path)

    manifest = run_rv_astrometry_gate(manifest, cfg, run_path=run_path)
    gate_rec = manifest.stages["rv_astrometry_gate"]
    assert gate_rec.status is StageStatus.COMPLETED
    assert gate_rec.artifact_path
    gate_cands, meta = read_stage_hdf5(Path(gate_rec.artifact_path))
    assert len(gate_cands) == 2
    assert meta["stage"] == "rv_astrometry_gate"
    diag_dir = Path(gate_rec.artifact_path).parent / (
        f"{Path(gate_rec.artifact_path).stem}_diagnostics"
    )
    assert (diag_dir / "reports" / "rv_astrometry_gate_funnel.txt").is_file()
    assert (diag_dir / "reports" / "rv_astrometry_gate_pass_rate.txt").is_file()

    manifest = run_joint_orbit_fit(manifest, cfg, run_path=run_path)
    joint_rec = manifest.stages["joint_orbit_fit"]
    assert joint_rec.status is StageStatus.COMPLETED
    joint_cands, _ = read_stage_hdf5(Path(joint_rec.artifact_path))
    tiers = {c.source_id: c.orbit_tier for c in joint_cands}
    assert tiers[1001] is OrbitTier.JOINT_ASTROMETRY_RV
    assert tiers[9] is OrbitTier.ASTROMETRY_ONLY


def test_joint_stage_skipped_when_all_gate_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config().model_copy(deep=True)
    cfg.paths.artifact_root = str(tmp_path / "output")
    monkeypatch.chdir(tmp_path)
    manifest = create_run_manifest(cfg)
    run_path = tmp_path / "runs" / f"{manifest.run_id}.yaml"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    save_run_manifest(manifest, run_path)

    # Only outliers → joint stage SKIPPED with canonical reason.
    gated, _ = run_gate_on_candidates([_candidate_outlier()], cfg)
    manifest = run_joint_orbit_fit(
        manifest, cfg, run_path=run_path, candidates=gated
    )
    rec = manifest.stages["joint_orbit_fit"]
    assert rec.status is StageStatus.SKIPPED
    assert rec.reason == JOINT_ORBIT_SKIP_REASON


def test_registry_fingerprint_keys_and_checksum_section() -> None:
    assert STAGE_REGISTRY["rv_astrometry_gate"].config_fingerprint_keys == (
        "rv_consistency",
    )
    assert STAGE_REGISTRY["joint_orbit_fit"].inputs_from == ("rv_astrometry_gate",)
    assert "rv_consistency" in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    assert cfg.rv_consistency.chi2_dof_threshold == pytest.approx(5.0)
