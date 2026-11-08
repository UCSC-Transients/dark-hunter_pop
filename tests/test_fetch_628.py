"""#628 fetch helpers (no network, no data/)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    s = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)  # type: ignore[union-attr]
    return m


@pytest.mark.unit
def test_tap_queries_are_host_scoped() -> None:
    m = _load("fetch_nss_host_tap_628")
    for kind, (_, adql) in m.QUERIES.items():
        assert ("tap_upload.ids" in adql) or ("{lo}" in adql and "nss_two_body_orbit" in adql), kind
    assert "external.gaiaedr3_distance" in m.QUERIES["bj_distances"][1]


@pytest.mark.unit
def test_csv_table_parses_skyserver_output() -> None:
    m = _load("fetch_nss_host_surveys_628")
    t = m.csv_table("#Table1\na,b\n1,x\n2,y\n")
    assert list(t["a"]) == [1, 2]
    assert m.csv_table("#Table1\na,b\n") is None


@pytest.mark.unit
def test_xmatch_position_propagation(tmp_path: Path) -> None:
    import h5py

    m = _load("fetch_nss_host_xmatch_628")
    p = tmp_path / "po.h5"
    with h5py.File(p, "w") as h:
        g = h.create_group("published")
        g["source_id"] = np.array([1, 2], np.int64)
        g["ra"], g["dec"] = np.array([10.0, 20.0]), np.array([0.0, 60.0])
        g["pmra"], g["pmdec"] = np.array([3600.0, np.nan]), np.array([-3600.0, np.nan])
    sid, ra, dec = m.propagated_positions(p, 2006.0)
    assert sid.tolist() == [1, 2]
    assert ra[0] == pytest.approx(10.0 - 0.01) and dec[0] == pytest.approx(0.01)
    assert ra[1] == 20.0 and dec[1] == 60.0
