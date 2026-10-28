"""#391 spec §12.28: decided-settings guard (config/population/decided_settings.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from darkhunter_pop import decided_settings as ds
from darkhunter_pop.config_loader import load_config

POP = Path("config/population")
FIT_CONFIGS = [POP / "rung3_coarse48_v11.yaml"]
BJ_PARENT = Path("data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj")
OLD_PARENT = Path("data/dr3/gaia_snapshots/20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj")


def _guarded_launch_fragments() -> list[Path]:
    st = ds.load_decided_settings()
    out = []
    for p in sorted(POP.glob("proposal_set_*.yaml")):
        gen = (yaml.safe_load(p.read_text()).get("proposal") or {}).get("generation")
        if gen is not None and int(gen) >= st.min_guarded_generation and int(gen) not in st.exempt_generations:
            out.append(p)
    return out


@pytest.mark.unit
def test_settings_file_loads_and_every_entry_has_a_ref() -> None:
    st = ds.load_decided_settings()
    assert st.settings and all(s.ref.strip() for s in st.settings)
    keys = [(s.scope, s.key) for s in st.settings]
    assert len(keys) == len(set(keys))
    must = {
        ("launch_fragment", "proposal.flux.evolved_rows_centre"),
        ("launch_fragment", "proposal.resolved_companions"),
        ("launch_fragment", "proposal.truth_distance"),
        ("launch_fragment", "proposal.parallax_floor_mas"),
        ("pipeline_config", "isochrone_mass.lowmass_correction.enabled"),
        ("launch_args", "parent_snapshot"),
        ("fit_config", "rung3.m1_law_mass"),
        ("fit_config", "rung3.c1_period_max_days"),
    }
    assert must <= set(keys)


@pytest.mark.unit
def test_guarded_launch_fragments_carry_the_decided_settings() -> None:
    frags = _guarded_launch_fragments()
    assert POP / "proposal_set_single47.yaml" in frags and POP / "proposal_set_co_v1_wd1005.yaml" in frags
    cfg = load_config(Path("config/config.yaml"))
    for f in frags:
        dev = ds.check_launch(f, config=cfg, parent_dir=BJ_PARENT)
        assert not dev, f"{f}: " + "; ".join(map(str, dev))


@pytest.mark.unit
@pytest.mark.parametrize("path", FIT_CONFIGS, ids=lambda p: p.name)
def test_default_fit_configs_carry_the_decided_settings(path: Path) -> None:
    assert not ds.check_fit_config(path)


@pytest.mark.unit
def test_gens_33_36_deviate_only_in_the_evolved_centre() -> None:
    # The recorded drift that motivated §12.28; any other deviation here is a new finding.
    cfg = load_config(Path("config/config.yaml"))
    for g in (33, 34, 35, 36):
        dev = ds.check_launch(POP / f"proposal_set_resim{g}.yaml", config=cfg, parent_dir=OLD_PARENT)
        assert "proposal.flux.evolved_rows_centre" in [d.key for d in dev]


@pytest.mark.unit
def test_check_flags_value_missing_and_absent(tmp_path: Path) -> None:
    st = ds.DecidedSettings.model_validate({
        "schema_version": 1, "min_guarded_generation": 0,
        "settings": [
            {"scope": "launch_fragment", "key": "proposal.a", "equals": 0.2, "ref": "r"},
            {"scope": "launch_fragment", "key": "proposal.b", "present": True, "ref": "r"},
            {"scope": "launch_fragment", "key": "proposal.c", "equals": True, "ref": "r"},
            {"scope": "launch_args", "key": "parent_snapshot", "equals": "bj", "ref": "r"},
        ],
    })
    good = {"proposal": {"a": 0.2, "b": {"x": 1}, "c": True}}
    assert ds.check_mapping(good, st, "launch_fragment") == []
    bad = {"proposal": {"a": 0.3, "b": None, "c": 1}}
    assert [d.key for d in ds.check_mapping(bad, st, "launch_fragment")] == ["proposal.a", "proposal.b", "proposal.c"]
    assert [d.key for d in ds.check_mapping({}, st, "launch_fragment")] == ["proposal.a", "proposal.b", "proposal.c"]
    f = tmp_path / "frag.yaml"
    f.write_text(yaml.safe_dump(good))
    assert [d.key for d in ds.check_launch(f, parent_dir=tmp_path / "other", settings=st)] == ["parent_snapshot"]
    with pytest.raises(ValueError):
        ds.DecidedSetting.model_validate({"scope": "fit_config", "key": "k", "ref": "r"})


@pytest.mark.unit
def test_v6a_fit_configs_deviate_only_in_the_low_mass_conditioning_floor() -> None:
    # §12.29: v6a keeps the superseded 0.8 Msun floor; any other deviation is a new finding.
    for name in ("rung3_compliant108_v6a.yaml", "rung3_coarse32_v6a.yaml"):
        assert "rung3.cmd_binary_conditioning.m1_min_msun" in [d.key for d in ds.check_fit_config(POP / name)]
