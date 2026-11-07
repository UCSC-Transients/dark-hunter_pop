"""R5-T1 (#614): rung-5 bin config, greedy ESS merge, freeze digest and guard. No data/ access."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import rung5_bins as r5

CFG = "config/population/rung5_bins_dr3.yaml"


@pytest.fixture(scope="module")
def cfg() -> r5.Rung5BinsConfig:
    return r5.load_rung5_bins(CFG)


@pytest.mark.unit
def test_shipped_config_is_dr3_and_unfrozen(cfg: r5.Rung5BinsConfig) -> None:
    assert cfg.data_release == "dr3"
    assert cfg.true_m2_edges_msun["wd"][-1] == pytest.approx(1.4)
    assert cfg.true_m2_edges_msun["dark"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not frozen"):
        r5.assert_frozen(cfg)


@pytest.mark.unit
def test_wd_top_edge_must_be_mch(cfg: r5.Rung5BinsConfig) -> None:
    raw = cfg.model_dump()
    raw["true_m2_edges_msun"]["wd"] = [0.15, 0.6, 1.3]
    with pytest.raises(ValueError, match="M_Ch"):
        r5.Rung5BinsConfig.model_validate(raw)


@pytest.mark.unit
def test_edges_must_increase(cfg: r5.Rung5BinsConfig) -> None:
    raw = cfg.model_dump()
    raw["m2_app_edges_msun"] = [0.0, 1.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        r5.Rung5BinsConfig.model_validate(raw)


@pytest.mark.unit
def test_binned_ess_equal_weights_counts() -> None:
    v = np.array([0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 9.0])
    ess = r5.binned_ess(v, np.ones_like(v), [0, 1, 2, 3])
    np.testing.assert_allclose(ess, [2, 1, 3])


@pytest.mark.unit
def test_greedy_merge_reaches_min_ess_and_keeps_ends() -> None:
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.uniform(0, 1, 50), rng.uniform(1, 2, 5), rng.uniform(2, 3, 5), rng.uniform(3, 4, 40), rng.uniform(4, 5, 3)])
    edges, ess = r5.greedy_merge_edges(v, np.ones_like(v), [0, 1, 2, 3, 4, 5], 8)
    assert edges[0] == 0 and edges[-1] == 5
    assert set(edges) <= {0, 1, 2, 3, 4, 5}
    assert all(e >= 8 for e in ess)
    assert edges == (0.0, 1.0, 3.0, 5.0)


@pytest.mark.unit
def test_greedy_merge_all_short_gives_one_bin() -> None:
    v = np.array([0.5, 1.5])
    edges, ess = r5.greedy_merge_edges(v, np.ones(2), [0, 1, 2, 3], 30)
    assert edges == (0.0, 3.0) and ess[0] == pytest.approx(2.0)


@pytest.mark.unit
def test_greedy_merge_respects_mask() -> None:
    v = np.array([0.5] * 10 + [1.5] * 10)
    m = np.array([True] * 10 + [False] * 10)
    edges, _ = r5.greedy_merge_edges(v, np.ones(20), [0, 1, 2], 5, mask=m)
    assert edges == (0.0, 2.0)


def _frozen(cfg: r5.Rung5BinsConfig) -> r5.Rung5BinsConfig:
    d = r5.derived_block(
        cfg, true_m2_edges_msun={"wd": (0.15, 0.6, 1.4), "dark": (1.0, 3.0, 40.0)}, m2_app_edges_msun=(0.0, 1.0, 1000.0),
        true_bin_ess_detected={"wd": (40.0, 31.0), "dark": (100.0, 50.0)}, m2_app_bin_ess_detected=(200.0, 60.0),
        artifacts_sha256={"a.h5": "0" * 64}, primary_model="x", partition="train", real_data_read=False, frozen_at="t",
    )
    return cfg.model_copy(update={"derived": d})


@pytest.mark.unit
def test_freeze_digest_roundtrip_and_tamper(cfg: r5.Rung5BinsConfig) -> None:
    frozen = _frozen(cfg)
    assert r5.assert_frozen(frozen).digest == r5.freeze_digest(frozen)
    tampered = frozen.model_copy(update={"min_cell_ess": 10.0})
    with pytest.raises(ValueError, match="digest mismatch"):
        r5.assert_frozen(tampered)
    revalidated = r5.Rung5BinsConfig.model_validate(frozen.model_dump())
    r5.assert_frozen(revalidated)


@pytest.mark.unit
def test_derived_edges_must_be_subset(cfg: r5.Rung5BinsConfig) -> None:
    raw = _frozen(cfg).model_dump()
    raw["derived"]["true_m2_edges_msun"]["dark"] = [1.0, 2.5, 40.0]
    with pytest.raises(ValueError, match="sub-set"):
        r5.Rung5BinsConfig.model_validate(raw)


@pytest.mark.unit
def test_real_data_flag_is_pinned_false(cfg: r5.Rung5BinsConfig) -> None:
    raw = _frozen(cfg).model_dump()
    raw["derived"]["real_data_read"] = True
    with pytest.raises(ValueError):
        r5.Rung5BinsConfig.model_validate(raw)
