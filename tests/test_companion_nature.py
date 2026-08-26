"""Tests for companion_nature_likelihood."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.companion_nature import (
    NATURE_CLASSES,
    STAGE_NAME,
    age_bin_diagnostic,
    evaluate_companion_nature,
    load_cooling_tracks,
    read_stage_hdf5,
    run_companion_nature_likelihood,
    run_companion_nature_on_candidates,
    weights_from_bic,
    write_stage_hdf5,
)
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import SHARED_CHECKSUM_SECTIONS
from darkhunter_pop.population_model import validate_companion_nature_weights
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    create_run_manifest,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    CandidateRecord,
    ParameterSet,
    PhotometryPoint,
    StageRecord,
    StageStatus,
)

pytestmark = pytest.mark.unit

FIXTURE_TRACKS = Path(__file__).parent / "fixtures" / "bedard_tracks_tiny.csv"


def _diag(cov: float = 0.01) -> list[list[float]]:
    return [[cov]]


def _m1_m2(m1: float = 1.2, m2: float = 0.6) -> tuple[ParameterSet, ParameterSet]:
    return (
        ParameterSet(
            names=["M1"],
            values=[m1],
            covariance=_diag(0.01),
            provenance="TAG10",
            units=["Msun"],
        ),
        ParameterSet(
            names=["M2"],
            values=[m2],
            covariance=_diag(0.04),
            provenance="gaiamock_mass_function+TAG10_M1",
            units=["Msun"],
        ),
    )


def _candidate(
    *,
    source_id: int = 1001,
    phot_chi2: dict[str, float] | None = None,
    xp_chi2: dict[str, float] | None = None,
    sb2: bool = False,
    sb2_wd_likeness: float | None = None,
    age_gyr: float | None = 2.0,
    with_bands: bool = False,
) -> CandidateRecord:
    m1, m2 = _m1_m2()
    extras: dict = {}
    if age_gyr is not None:
        extras["age_gyr"] = age_gyr
    if phot_chi2 is not None:
        extras["phot_chi2_dark"] = phot_chi2["dark"]
        extras["phot_chi2_wd"] = phot_chi2["WD"]
        extras["phot_chi2_other"] = phot_chi2["other"]
        extras["phot_n_data"] = 5
    if xp_chi2 is not None:
        extras["xp_chi2_dark"] = xp_chi2["dark"]
        extras["xp_chi2_wd"] = xp_chi2["WD"]
        extras["xp_chi2_other"] = xp_chi2["other"]
        extras["xp_n_data"] = 10
    if sb2_wd_likeness is not None:
        extras["sb2_wd_likeness"] = sb2_wd_likeness
        extras["sb2_mass_ratio_unlocked"] = True
    photometry: list[PhotometryPoint] = []
    if with_bands:
        photometry = [
            PhotometryPoint(band="G", mag=12.0, mag_err=0.02),
            PhotometryPoint(band="BP", mag=12.3, mag_err=0.03),
            PhotometryPoint(band="RP", mag=11.6, mag_err=0.03),
        ]
    rv_summary: dict = {}
    if sb2:
        rv_summary["sb2_orbit"] = {"period_day": 100.0, "eccentricity": 0.1}
        extras["sb2_mass_ratio_unlocked"] = True
    return CandidateRecord(
        source_id=source_id,
        nss_solution_type="SB2" if sb2 else "Orbital",
        parallax_mas=5.0,
        photometry=photometry,
        m1=m1,
        m2=m2,
        rv_summary=rv_summary,
        extras=extras,
    )


def test_registry_inputs_and_fingerprint() -> None:
    spec = STAGE_REGISTRY[STAGE_NAME]
    assert spec.inputs_from == ("joint_orbit_fit", "mass_derivation_refined")
    assert "companion_nature" in spec.config_fingerprint_keys
    assert "physics.cooling_tracks_path" in spec.config_fingerprint_keys
    assert "companion_nature" in SHARED_CHECKSUM_SECTIONS


def test_weights_continuous_never_hard_zero() -> None:
    cfg = load_config().companion_nature
    # Huge ΔBIC favoring dark — WD/other still above floor after renorm.
    bic = {"dark": 0.0, "WD": 1.0e6, "other": 1.0e6}
    weights = weights_from_bic(bic, cfg)
    assert set(weights) == set(COMPANION_NATURE_WEIGHT_KEYS)
    assert set(weights) == set(NATURE_CLASSES)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["BH"] + weights["NS"] > 0.9
    assert weights["WD"] > 0.0 and weights["other"] > 0.0
    assert weights["outlier"] > 0.0
    validate_companion_nature_weights(weights)


def test_joint_not_independent_product() -> None:
    """Photometry+XP share one BIC; adding XP favoring WD shifts weights jointly."""
    cfg = load_config()
    phot_only = _candidate(
        phot_chi2={"dark": 0.0, "WD": 40.0, "other": 40.0},
    )
    both = _candidate(
        phot_chi2={"dark": 0.0, "WD": 40.0, "other": 40.0},
        xp_chi2={"dark": 50.0, "WD": 0.0, "other": 50.0},
    )
    e_phot = evaluate_companion_nature(phot_only, cfg)
    e_both = evaluate_companion_nature(both, cfg)
    assert e_phot.weights["BH"] + e_phot.weights["NS"] > (
        e_both.weights["BH"] + e_both.weights["NS"]
    )
    assert e_both.weights["WD"] > e_phot.weights["WD"]
    assert e_both.channels[0].available and e_both.channels[1].available


def test_sb2_downweights_dark() -> None:
    cfg = load_config()
    no_sb2 = _candidate(phot_chi2={"dark": 5.0, "WD": 5.0, "other": 5.0})
    sb2 = _candidate(
        phot_chi2={"dark": 5.0, "WD": 5.0, "other": 5.0},
        sb2=True,
        sb2_wd_likeness=0.9,
    )
    e0 = evaluate_companion_nature(no_sb2, cfg)
    e1 = evaluate_companion_nature(sb2, cfg)
    dark0 = e0.weights["BH"] + e0.weights["NS"]
    dark1 = e1.weights["BH"] + e1.weights["NS"]
    assert dark1 < dark0
    assert e1.weights["WD"] > e1.weights["other"]


def test_never_discards_candidates() -> None:
    cfg = load_config()
    cands = [
        _candidate(source_id=1, phot_chi2={"dark": 0.0, "WD": 30.0, "other": 30.0}),
        _candidate(source_id=2),  # no channels
        _candidate(
            source_id=3,
            phot_chi2={"dark": 40.0, "WD": 0.0, "other": 40.0},
            age_gyr=None,
        ),
    ]
    out, diag = run_companion_nature_on_candidates(cands, cfg)
    assert len(out) == 3
    assert diag.n_weighted == 3
    assert diag.n_input == 3
    assert all(c.companion_nature_weights is not None for c in out)
    for c in out:
        assert c.companion_nature_weights is not None
        validate_companion_nature_weights(c.companion_nature_weights)
        assert abs(sum(c.companion_nature_weights.values()) - 1.0) < 1e-9



def test_cooling_tracks_local_only(tmp_path: Path) -> None:
    cfg = load_config()
    # Missing path raises — never fetches.
    bad = cfg.model_copy(deep=True)
    bad.physics.cooling_tracks_path = str(tmp_path / "missing_tracks.csv")
    with pytest.raises(FileNotFoundError, match="no fetch"):
        load_cooling_tracks(bad, repo_root=tmp_path)

    ok = cfg.model_copy(deep=True)
    ok.physics.cooling_tracks_path = str(FIXTURE_TRACKS)
    table = load_cooling_tracks(ok, repo_root=tmp_path)
    assert table.source_path is not None
    assert table.mass_msun.size > 0
    mg = table.abs_mag_g(0.6, 1.0)
    assert np.isfinite(mg)


def test_two_tier_full_on_ambiguity() -> None:
    cfg = load_config()
    # Nearly equal BIC → full tier.
    amb = _candidate(phot_chi2={"dark": 10.0, "WD": 10.5, "other": 30.0})
    clear = _candidate(phot_chi2={"dark": 0.0, "WD": 80.0, "other": 80.0})
    e_amb = evaluate_companion_nature(amb, cfg)
    e_clear = evaluate_companion_nature(clear, cfg)
    assert e_amb.tier == "full"
    assert e_clear.tier == "fast"


def test_age_bin_diagnostic_required() -> None:
    cfg = load_config()
    cands = [
        _candidate(
            source_id=i,
            age_gyr=age,
            phot_chi2={"dark": 0.0, "WD": 20.0, "other": 20.0},
        )
        for i, age in enumerate([0.5, 2.0, 5.0, 12.0], start=1)
    ]
    out, diag = run_companion_nature_on_candidates(cands, cfg)
    assert diag.age_diagnostic is not None
    assert sum(diag.age_diagnostic.bin_counts) == 4
    age_diag = age_bin_diagnostic(out, cfg.companion_nature)
    assert "Age-bin diagnostic" in age_diag.message
    assert len(age_diag.mean_weights_by_bin) == len(cfg.companion_nature.age_bin_edges_gyr) - 1


def test_stage_runner_hdf5_and_report(tmp_path: Path) -> None:
    cfg = load_config()
    tweaked = cfg.model_copy(deep=True)
    tweaked.paths.artifact_root = str(tmp_path / "output")
    tweaked.physics.cooling_tracks_path = str(FIXTURE_TRACKS)

    manifest = create_run_manifest(tweaked)
    run_path = tmp_path / "runs" / f"{manifest.run_id}.yaml"
    run_path.parent.mkdir(parents=True)
    save_run_manifest(manifest, run_path)

    # Seed upstream joint_orbit_fit artifact.
    upstream_spec = STAGE_REGISTRY["joint_orbit_fit"]
    upstream = stage_artifact_path(tweaked, upstream_spec, run_id=manifest.run_id)
    seeds = [
        _candidate(
            source_id=11,
            phot_chi2={"dark": 0.0, "WD": 25.0, "other": 25.0},
            age_gyr=1.0,
        ),
        _candidate(
            source_id=12,
            phot_chi2={"dark": 30.0, "WD": 0.0, "other": 30.0},
            xp_chi2={"dark": 20.0, "WD": 1.0, "other": 20.0},
            age_gyr=8.0,
        ),
    ]
    write_stage_hdf5(upstream, seeds, diagnostics={"n_input": 2})

    manifest = manifest.model_copy(
        update={
            "stages": {
                **manifest.stages,
                "joint_orbit_fit": StageRecord(
                    stage_name="joint_orbit_fit",
                    status=StageStatus.COMPLETED,
                    artifact_path=str(upstream),
                ),
            }
        }
    )
    save_run_manifest(manifest, run_path)

    manifest = run_companion_nature_likelihood(
        manifest, tweaked, run_path=run_path, force_rerun=True
    )
    record = manifest.stages[STAGE_NAME]
    assert record.status is StageStatus.COMPLETED
    assert record.artifact_path is not None
    path = Path(record.artifact_path)
    assert path.is_file()
    loaded, meta = read_stage_hdf5(path)
    assert len(loaded) == 2
    assert all(c.companion_nature_weights is not None for c in loaded)
    assert meta.get("stage") == STAGE_NAME or meta.get("stage") == STAGE_NAME
    report = path.parent / "reports" / "companion_nature_funnel.txt"
    age_json = path.parent / "reports" / "companion_nature_age_bins.json"
    assert report.is_file()
    assert age_json.is_file()
    text = report.read_text(encoding="utf-8")
    assert "none discarded" in text
    assert "age-bin diagnostic" in text.lower()


def test_config_fingerprint_changes_artifact_path() -> None:
    cfg = load_config()
    spec = STAGE_REGISTRY[STAGE_NAME]
    p1 = stage_artifact_path(cfg, spec, run_id="runA")
    tweaked = cfg.model_copy(deep=True)
    tweaked.companion_nature.delta_bic_threshold = (
        cfg.companion_nature.delta_bic_threshold + 1.0
    )
    p2 = stage_artifact_path(tweaked, spec, run_id="runA")
    assert p1.name != p2.name
