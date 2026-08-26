"""Tests for config schema and loader (F3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from darkhunter_pop.config_loader import (
    assert_config_checksum,
    audit_dr_independence,
    config_checksum,
    deep_merge,
    effective_M_Ch_msun,
    load_config,
    require_dr3_active_for_v1,
)
from darkhunter_pop.config_schema import (
    PATH_SPECIFIC_LEAF_KEYS,
    SHARED_CHECKSUM_SECTIONS,
    PipelineConfig,
    QualityCutBin,
)
from darkhunter_pop.schemas import ActiveDRMode

pytestmark = pytest.mark.unit


def test_load_canonical_config() -> None:
    cfg = load_config()
    assert cfg.active_dr_mode is ActiveDRMode.DR3
    assert cfg.mass_calibration.method.value == "TAG10"
    assert len(cfg.dr3.quality_cut_bins) == 2
    assert len(cfg.dr4.quality_cut_bins) == 2
    require_dr3_active_for_v1(cfg)


def test_quality_cuts_accept_arbitrary_bin_count(tmp_path: Path) -> None:
    raw = load_config().model_dump(mode="json")
    raw["dr3"]["quality_cut_bins"] = [
        {"g_max": 12.0, "gof_max": 8.0},
        {"g_min": 12.0, "g_max": 15.0, "gof_max": 6.0},
        {"g_min": 15.0, "gof_max": 4.0},
    ]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path, merge_fragments=False)
    assert len(cfg.dr3.quality_cut_bins) == 3


def test_path_specific_keys_are_not_shareable() -> None:
    """DR path fields only exist under dr3/dr4 — cannot be a single shared top-level key."""
    fields = set(PipelineConfig.model_fields)
    assert "mission_baseline_months" not in fields
    assert "quality_cut_bins" not in fields
    assert "dr3" in fields and "dr4" in fields


def test_checksum_ignores_inactive_dr(tmp_path: Path) -> None:
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    # Mutate inactive DR4 only.
    altered.dr4.mission_baseline_months = 999.0
    assert config_checksum(altered) == base
    # Mutate active DR3 → checksum changes.
    altered2 = cfg.model_copy(deep=True)
    altered2.dr3.mission_baseline_months = 40.0
    assert config_checksum(altered2) != base
    # Mutate shared physics → checksum changes.
    altered3 = cfg.model_copy(deep=True)
    altered3.physics.mc_noise_threshold = 0.05
    assert config_checksum(altered3) != base


def test_assert_config_checksum_refuses_mismatch() -> None:
    cfg = load_config()
    with pytest.raises(ValueError, match="config checksum mismatch"):
        assert_config_checksum(cfg, "0" * 64)


def test_effective_m_ch_applies_delta() -> None:
    cfg = load_config()
    assert effective_M_Ch_msun(cfg) == pytest.approx(1.4)
    tweaked = cfg.model_copy(deep=True)
    tweaked.mass_calibration.delta_M_Ch_msun = 0.05
    assert effective_M_Ch_msun(tweaked) == pytest.approx(1.45)


def test_fragment_merge_then_canonical_wins(tmp_path: Path) -> None:
    frag = tmp_path / "fragments"
    frag.mkdir()
    (frag / "a.yaml").write_text(
        yaml.safe_dump({"physics": {"mc_noise_threshold": 0.2}}),
        encoding="utf-8",
    )
    canonical = tmp_path / "config.yaml"
    # Minimal valid config: start from real dump then override.
    data = load_config().model_dump(mode="json")
    data["physics"]["mc_noise_threshold"] = 0.1
    canonical.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(canonical, fragments_dir=frag, merge_fragments=True)
    assert cfg.physics.mc_noise_threshold == pytest.approx(0.1)


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {
        "a": {"b": 1, "c": 3}
    }


def test_quality_cut_bin_validation() -> None:
    with pytest.raises(ValidationError):
        QualityCutBin(gof_max=5.0)
    with pytest.raises(ValidationError):
        QualityCutBin(g_min=14.0, g_max=13.0, gof_max=5.0)


def test_dr4_active_refused_in_v1() -> None:
    cfg = load_config()
    bad = cfg.model_copy(update={"active_dr_mode": ActiveDRMode.DR4})
    with pytest.raises(ValueError, match="not runnable"):
        require_dr3_active_for_v1(bad)


def test_audit_notes_identical_path_values() -> None:
    cfg = load_config()
    result = audit_dr_independence(cfg)
    notes = result.messages()
    # sed_filters currently match across DR in the scaffold defaults.
    assert any("sed_filters" in n for n in notes)
    assert result.ok
    assert result.violations == []


def test_audit_flags_shared_path_specific_key_in_raw_dict() -> None:
    raw = load_config().model_dump(mode="json")
    raw["mission_baseline_months"] = 34.0  # illegally shared
    result = audit_dr_independence(raw)
    assert not result.ok
    assert any(
        f.severity == "violation" and f.key == "mission_baseline_months"
        for f in result.findings
    )


def test_audit_flags_path_specific_leaf_outside_dr_subtree() -> None:
    raw = load_config().model_dump(mode="json")
    raw["selection_function_followup"]["accel_jerk_catalog_id"] = "shared_bad"
    result = audit_dr_independence(raw)
    assert not result.ok
    assert any("accel_jerk_catalog_id" in (f.key or "") for f in result.violations)


def test_audit_flags_shared_physics_under_dr_paths() -> None:
    raw = load_config().model_dump(mode="json")
    raw["dr3"]["physics"] = {"mc_noise_threshold": 0.1, "imf": "kroupa"}
    raw["dr4"]["physics"] = {"mc_noise_threshold": 0.2, "imf": "kroupa"}
    result = audit_dr_independence(raw)
    assert not result.ok
    assert any(f.key == "physics" for f in result.violations)


def test_path_specific_leaf_registry_covers_mission_keys() -> None:
    assert "scanning_law_id" in PATH_SPECIFIC_LEAF_KEYS
    assert "quality_cut_bins" in PATH_SPECIFIC_LEAF_KEYS
    assert "sed_filters" in PATH_SPECIFIC_LEAF_KEYS
    assert "accel_jerk_catalog_id" in PATH_SPECIFIC_LEAF_KEYS
    fields = set(PipelineConfig.model_fields)
    for leaf in PATH_SPECIFIC_LEAF_KEYS:
        assert leaf not in fields


def test_phase2_canonical_matches_fragment_merge() -> None:
    """Integration checkpoint: config.yaml materializes fragments (issue #40)."""
    with_frags = load_config(merge_fragments=True)
    canon_only = load_config(merge_fragments=False)
    assert with_frags.model_dump(mode="json") == canon_only.model_dump(mode="json")
    assert canon_only.dr3.nss_table.startswith("gaiadr3.")
    assert canon_only.dr4.nss_table.startswith("gaiadr4.")
    assert canon_only.dr4.selection_function_followup.accel_jerk_catalog_id.startswith(
        "dr4_"
    )
    assert len(canon_only.dr3.external_photometry_crossmatches) >= 1
    assert len(canon_only.selection_function_followup.target_lists) >= 1
    assert len(canon_only.selection_function_followup.major_surveys) >= 1


def test_checksum_includes_phase2_shared_sections() -> None:
    assert "mass_derivation" in SHARED_CHECKSUM_SECTIONS
    assert "rv_consistency" in SHARED_CHECKSUM_SECTIONS
    assert "companion_nature" in SHARED_CHECKSUM_SECTIONS
    assert "selection_function_followup" in SHARED_CHECKSUM_SECTIONS
    assert "sensitivity_analysis" in SHARED_CHECKSUM_SECTIONS
    assert "triples" in SHARED_CHECKSUM_SECTIONS
    assert "diagnostics" not in SHARED_CHECKSUM_SECTIONS
    cfg = load_config()
    base = config_checksum(cfg)
    altered = cfg.model_copy(deep=True)
    altered.sensitivity_analysis.n_mass_bins = cfg.sensitivity_analysis.n_mass_bins + 1
    assert config_checksum(altered) != base
    layout_only = cfg.model_copy(deep=True)
    layout_only.diagnostics.figure_dpi = cfg.diagnostics.figure_dpi + 10
    assert config_checksum(layout_only) == base
    rv_altered = cfg.model_copy(deep=True)
    rv_altered.rv_consistency.chi2_dof_threshold = (
        cfg.rv_consistency.chi2_dof_threshold + 1.0
    )
    assert config_checksum(rv_altered) != base
    cn_altered = cfg.model_copy(deep=True)
    cn_altered.companion_nature.delta_bic_threshold = (
        cfg.companion_nature.delta_bic_threshold + 1.0
    )
    assert config_checksum(cn_altered) != base
    triples_flip = cfg.model_copy(deep=True)
    triples_flip.triples.enabled = True
    assert config_checksum(triples_flip) != base
