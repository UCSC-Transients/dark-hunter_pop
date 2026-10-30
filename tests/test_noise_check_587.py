"""#587: like-for-like rung-3 comparison helpers (plot ranking with fixed edges / dropped components; noise metrics)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytestmark = pytest.mark.unit


def _dump(path: Path, n_lum: int = 400, n_wd: int = 0, n_real: int = 300, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    d = {}
    for pfx, obs in (("o", ("g", "d", "colour0", "mg0", "P", "e", "A", "a0snr", "plxsnr", "f2", "ruwe")),
                     ("a", ("g", "d", "colour0", "mg0", "plxsnr", "signif", "f2", "ruwe"))):
        for comp, n in (("lum", n_lum), ("wd", n_wd), ("real", n_real)):
            for k in obs:
                d[f"{pfx}_{comp}_{k}"] = rng.uniform(1.0, 20.0, n)
            if comp != "real":
                d[f"{pfx}_{comp}_w"] = np.full(n, n_real / max(n_lum + n_wd, 1))
        if pfx == "o":
            d["o_spur_w"] = np.full(n_real, 0.01)
    np.savez(path, **d)


def test_rank_only_reuses_edges_and_drops_components(tmp_path):
    import rung3_plots_391 as rp

    _dump(tmp_path / "a.npz", n_wd=50)
    _dump(tmp_path / "b.npz", n_wd=0, seed=2)  # no WD draws at all (--no-co)
    assert rp.main(["--dump", str(tmp_path / "a.npz"), "--out-dir", str(tmp_path / "ra"), "--rank-only", "--boot-2d"]) == 0
    e = json.loads((tmp_path / "ra" / "edges.json").read_text())
    assert rp.main(["--dump", str(tmp_path / "b.npz"), "--out-dir", str(tmp_path / "rb"), "--rank-only", "--edges-in", str(tmp_path / "ra" / "edges.json")]) == 0
    assert json.loads((tmp_path / "rb" / "edges.json").read_text()) == e
    assert rp.main(["--dump", str(tmp_path / "a.npz"), "--out-dir", str(tmp_path / "rc"), "--rank-only", "--drop", "wd"]) == 0
    ra = json.loads((tmp_path / "ra" / "ranking.json").read_text())
    assert all(r["chi2_mc_err"] is not None for r in ra)  # 2-D bootstrap on
    assert not list((tmp_path / "ra").glob("*.png"))
    z = np.load(tmp_path / "a.npz")
    k = rp.Kind(z, "o", rp.ORBIT_OBS, drop=("wd",))
    assert set(k.comp) == {"lum", "spurious"}


def test_count_shape_ratio():
    import noise_check_compare_587 as nc

    r = np.array([True, False, False, False])
    m = np.array([True, True, False, False])
    w = np.ones(4)
    c = nc.count(r, m, w)
    assert c["real"] == 1 and c["mock"] == 2 and c["ratio"] == pytest.approx(0.5)
    assert c["shape_ratio"] == pytest.approx((1 / 4) / (2 / 4))


def test_empty_co_block_shapes():
    import rung3_joint_391 as rj

    co = rj.empty_co_block(23)
    assert co["casc"].shape == (0, 23)
    assert co["w"].size == 0 and co["in_o"].dtype == bool
