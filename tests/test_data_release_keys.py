"""R5-T13 (#618): data-release keys for config/population/* and the release in the primary-model version."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_pop import compact_objects as cob
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import audit_population_data_release

ROOT = Path(__file__).resolve().parents[1]
CO_FRAGMENT = "config/population/proposal_set_co_v1_wd1006.yaml"
LUMINOUS = "config/population/proposal_set_single53.yaml"
ACTIVE = ["rung4_co_v1.yaml", "rung3_joint_v4.yaml", "proposal_set_single53.yaml", "proposal_set_single57.yaml",
          "proposal_set_single58.yaml", "proposal_set_single59.yaml", "proposal_set_co_v1_wd1006.yaml",
          "proposal_set_co_v1_wd1007.yaml"]


@pytest.mark.unit
@pytest.mark.parametrize("path", [CO_FRAGMENT, LUMINOUS])
def test_dr3_labels_unchanged(path: str) -> None:
    cfg = ps.load_proposal_set_fragment(path).proposal
    v = ps.primary_model_version(cfg)
    assert ":dr3" not in v and ":dr4" not in v
    assert ps.data_release_of_version(v) == "dr3"


@pytest.mark.unit
@pytest.mark.parametrize("path", [CO_FRAGMENT, LUMINOUS])
def test_dr4_label_never_mixes_with_dr3(path: str) -> None:
    cfg = ps.load_proposal_set_fragment(path).proposal
    dr4 = cfg.model_copy(update={"data_release": "dr4"})
    v4 = ps.primary_model_version(dr4)
    assert ps.data_release_of_version(v4) == "dr4"
    assert v4 != ps.primary_model_version(cfg)
    if cfg.compact_object is not None:
        assert v4.endswith(f":dr4:{cfg.compact_object.version}") and ps.is_compact_object_version(v4)
    with pytest.raises(ValueError, match="different primary models"):
        ps.check_single_primary_model([cfg, dr4])


@pytest.mark.unit
def test_launch_guard() -> None:
    cfg = ps.load_proposal_set_fragment(LUMINOUS).proposal
    ps.check_data_release(cfg, "dr3")
    with pytest.raises(ValueError, match="active_dr_mode"):
        ps.check_data_release(cfg, "dr4")


@pytest.mark.unit
def test_release_match() -> None:
    ps.check_release_match("dr3", ["looks_single_v2:a:co_v1"], "x")
    with pytest.raises(ValueError, match="another release"):
        ps.check_release_match("dr3", ["looks_single_v2:a:dr4:co_v1"], "x")
    with pytest.raises(ValueError, match="more than one"):
        ps.data_release_of_version("a:dr3:dr4")


@pytest.mark.unit
def test_rung4_config_requires_release() -> None:
    r4 = cob.load_rung4_config("config/population/rung4_co_v1.yaml")
    assert r4.data_release == "dr3"
    raw = r4.model_dump()
    raw.pop("data_release")
    with pytest.raises(ValueError):
        cob.Rung4AnalysisConfig.model_validate(raw)


@pytest.mark.unit
def test_active_population_configs_state_dr3() -> None:
    audit = audit_population_data_release(ROOT / "config" / "population")
    assert not audit["conflict"]
    assert not audit["dr4"]
    missing = set(ACTIVE) & set(audit["missing"])
    assert not missing, f"active configs without data_release: {sorted(missing)}"
