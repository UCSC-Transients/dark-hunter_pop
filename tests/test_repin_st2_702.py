"""#702: st2 re-pinned configs load and carry the st2 label; st1 records stay as history; schema default is st2."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from darkhunter_pop import config_schema as cs
from darkhunter_pop import rung5_bins as r5

ROOT = Path(__file__).resolve().parents[1]
ST2 = "colour_v2.1+tr1+e2+st2"


@pytest.mark.unit
def test_bins_st2_loads_unfrozen() -> None:
    cfg = r5.load_rung5_bins(str(ROOT / "config/population/rung5_bins_dr3_st2.yaml"))
    assert cfg.required_noise_model == ST2
    assert cfg.derived is None
    with pytest.raises(ValueError, match="not frozen"):
        r5.assert_frozen(cfg)


@pytest.mark.unit
def test_bins_st1_kept_as_history() -> None:
    cfg = r5.load_rung5_bins(str(ROOT / "config/population/rung5_bins_dr3.yaml"))
    assert cfg.required_noise_model.endswith("st1")
    r5.assert_frozen(cfg)


@pytest.mark.unit
def test_known_objects_st2() -> None:
    d = yaml.safe_load((ROOT / "config/benchmarks/rung5_known_objects_dr3_st2.yaml").read_text())
    assert d["required_noise_model"] == ST2
    old = yaml.safe_load((ROOT / "config/benchmarks/rung5_known_objects_dr3.yaml").read_text())
    assert old["required_noise_model"].endswith("st1")
    assert {k: v for k, v in d.items() if k != "required_noise_model"} == {
        k: v for k, v in old.items() if k != "required_noise_model"}


@pytest.mark.unit
def test_rung3_v6_pinned_st2() -> None:
    d = yaml.safe_load((ROOT / "config/population/rung3_joint_v6.yaml").read_text())
    txt = (ROOT / "config/population/rung3_joint_v6.yaml").read_text()
    assert "st1" not in txt
    assert ST2 in txt
    assert "joint" in str(d)


@pytest.mark.unit
def test_schema_default_is_st2() -> None:
    assert cs.EpochSolutionTypeConfig.model_fields["version"].default == "st2"
