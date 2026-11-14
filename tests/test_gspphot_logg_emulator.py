"""GSP-Phot log g emulator (#680, RUNG5_SPEC §2.7.4): config contract, calibration, sampling, seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.gspphot_logg_emulator import (
    DEFAULT_CONFIG_PATH,
    EmulatorDefinition,
    GspPhotLoggEmulator,
    abs_g_mag,
    bin_index,
    build_table,
    calibrate_levels,
    definition_sha256,
    draw_uniforms,
    holdout_mask,
    load_definition,
    load_emulator,
    rate_pull,
    selection_threshold,
    table_to_yaml,
)

pytestmark = pytest.mark.unit


def _definition(**binning: object) -> EmulatorDefinition:
    raw = yaml.safe_load((repo_root() / DEFAULT_CONFIG_PATH).read_text())
    raw["binning"] = {
        "fallback_levels": [
            {"name": "g_cmd", "edges": {"g_mag": [12.0], "bp_rp": [1.0], "abs_g_mag": [3.0]}},
            {"name": "cmd", "edges": {"bp_rp": [1.0], "abs_g_mag": [3.0]}},
            {"name": "all", "edges": {}},
        ],
        "n_min_rows": 50,
        "n_min_defined": 40,
        "logg_bins": {"min": 0.0, "max": 5.5, "width": 0.1},
        **binning,
    }
    return EmulatorDefinition.model_validate(raw)


def _synthetic(n: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Giants (M_G < 3): 50% undefined, log g ~ U(2, 3). Dwarfs: 10% undefined, log g ~ U(4, 5). G uniform 10-14."""
    g = rng.uniform(10.0, 14.0, n)
    giant = rng.random(n) < 0.3
    mg = np.where(giant, rng.uniform(-1.0, 2.9, n), rng.uniform(3.1, 7.0, n))
    bp_rp = np.where(giant, 1.3, 0.8)
    lg = np.where(giant, rng.uniform(2.0, 3.0, n), rng.uniform(4.0, 5.0, n))
    undef = rng.random(n) < np.where(giant, 0.5, 0.1)
    return {"g_mag": g, "bp_rp": bp_rp, "abs_g_mag": mg}, np.where(undef, np.nan, lg)


def _emulator(d: EmulatorDefinition, feats: dict[str, np.ndarray], logg: np.ndarray) -> GspPhotLoggEmulator:
    levels = calibrate_levels(feats, logg, d.binning)
    return GspPhotLoggEmulator(d, build_table(d, levels, definition_sha="0" * 64, provenance={"test": True}))


def test_shipped_definition_and_table_load_and_match_the_andrews_column() -> None:
    emu = load_emulator()
    spec = yaml.safe_load((repo_root() / "config/selections/andrews2022.yaml").read_text())
    assert emu.output_column == spec["reproduction_procedure"]["pass2"]["giant_logg_column"]
    cut = next(c for c in spec["cuts"] if c["id"] == "atf_giant_reject_logg")
    assert selection_threshold(emu.definition) == cut["parameters"]["logg_dwarf_min"]
    assert emu.table.definition_sha256 == definition_sha256()


def test_table_from_another_definition_is_refused(tmp_path: Path) -> None:
    src = (repo_root() / DEFAULT_CONFIG_PATH).read_text()
    alt = tmp_path / "def.yaml"
    alt.write_text(src + "\n# edited\n")
    with pytest.raises(ValueError, match="re-run"):
        load_emulator(alt)


def test_uniforms_deterministic_per_key_and_uniform() -> None:
    di = np.arange(200_000)
    u = draw_uniforms(680, 3, di, 0)
    assert np.all((u >= 0.0) & (u < 1.0))
    assert abs(u.mean() - 0.5) < 3e-3 and abs(np.mean(u < 0.1) - 0.1) < 3e-3
    assert np.array_equal(u, draw_uniforms(680, 3, di, 0))
    assert np.array_equal(u[[5, 17]], draw_uniforms(680, 3, np.array([5, 17]), 0))  # row-order independent
    for other in (draw_uniforms(680, 4, di, 0), draw_uniforms(680, 3, di, 1), draw_uniforms(681, 3, di, 0)):
        assert abs(np.corrcoef(u, other)[0, 1]) < 0.01


def test_bin_index_and_abs_g() -> None:
    e = np.array([1.0, 2.0])
    assert bin_index([0.5, 1.0, 1.5, 2.5, np.nan, np.inf], e).tolist() == [0, 1, 1, 2, 3, 3]
    m = abs_g_mag([10.0, 10.0, 10.0], [10.0, 0.0, np.nan])
    assert m[0] == pytest.approx(5.0) and np.isnan(m[1]) and np.isnan(m[2])


def test_calibration_recovers_known_missingness_and_reject_rate() -> None:
    rng = np.random.default_rng(1)
    d = _definition()
    feats, logg = _synthetic(40_000, rng)
    emu = _emulator(d, feats, logg)
    tf, tl = _synthetic(20_000, rng)
    p_undef, p_rej = emu.expected_rates(tf["g_mag"], tf["bp_rp"], tf["abs_g_mag"], threshold=3.6)
    giant = tf["abs_g_mag"] < 3.0
    assert p_undef[giant].mean() == pytest.approx(0.5, abs=0.02)
    assert p_undef[~giant].mean() == pytest.approx(0.1, abs=0.01)
    assert p_rej[giant].mean() == pytest.approx(0.5, abs=0.02)   # every defined giant has log g < 3.6
    assert p_rej[~giant].max() == 0.0                              # no defined dwarf below 3.6
    # stochastic draws agree with the exact expectation; values stay inside the calibrated support
    out = emu.emulate(tf["g_mag"], tf["bp_rp"], tf["abs_g_mag"], generation=7, draw_index=np.arange(tl.size))
    assert np.isnan(out.logg).mean() == pytest.approx(p_undef.mean(), abs=0.01)
    assert np.mean(out.logg < 3.6) == pytest.approx(p_rej.mean(), abs=0.01)
    lg = out.logg[np.isfinite(out.logg)]
    assert lg.min() >= 2.0 and lg.max() <= 5.0
    assert set(np.unique(out.defined_level)) == {0}


def test_sparse_cells_fall_back_to_coarser_levels() -> None:
    rng = np.random.default_rng(2)
    d = _definition()
    feats, logg = _synthetic(20_000, rng)
    bright_giant = (feats["g_mag"] < 12.0) & (feats["abs_g_mag"] < 3.0)
    keep = ~bright_giant | (rng.random(logg.size) < 0.004)  # ~10 bright giants: below n_min_rows
    emu = _emulator(d, {k: v[keep] for k, v in feats.items()}, logg[keep])
    out = emu.emulate([11.0, 13.0, 13.0], [1.3, 1.3, 0.8], [1.0, 1.0, 5.0], generation=0, draw_index=[0, 1, 2])
    assert out.defined_level.tolist() == [1, 0, 0] and out.value_level.tolist() == [1, 0, 0]
    out = emu.emulate([13.0], [np.nan], [np.nan], generation=0, draw_index=[3])  # no such real row: all-rows level
    assert out.defined_level.tolist() == [2]


def test_calibration_refuses_log_g_outside_grid() -> None:
    d = _definition()
    feats = {"g_mag": np.full(3, 13.0), "bp_rp": np.full(3, 1.0), "abs_g_mag": np.full(3, 4.0)}
    with pytest.raises(ValueError, match="outside logg_bins"):
        calibrate_levels(feats, np.array([4.0, 6.0, np.nan]), d.binning)


def test_table_yaml_round_trip() -> None:
    rng = np.random.default_rng(3)
    d = _definition()
    feats, logg = _synthetic(5_000, rng)
    emu = _emulator(d, feats, logg)
    from darkhunter_pop.gspphot_logg_emulator import EmulatorTable

    back = EmulatorTable.model_validate(yaml.safe_load(table_to_yaml(emu.table)))
    assert back == emu.table


def test_holdout_split_and_pull() -> None:
    sid = np.arange(10_000, dtype=np.int64) * 7919
    h = holdout_mask(sid, split_seed=6801, holdout_fraction=0.5)
    assert abs(h.mean() - 0.5) < 0.02 and np.array_equal(h, holdout_mask(sid, split_seed=6801, holdout_fraction=0.5))
    assert rate_pull(0.1, 0.1, 100, 100) == 0.0
    assert rate_pull(0.2, 0.1, 1000, 1000) > 3.0
    assert rate_pull(0.0, 0.0, 10, 10) == 0.0


def test_shipped_definition_fields() -> None:
    d = load_definition()
    assert d.calibration.restrict_to_equivalence_ids is True
    assert d.binning.fallback_levels[-1].edges == {}
