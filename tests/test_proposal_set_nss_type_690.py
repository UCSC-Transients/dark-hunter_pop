"""#690: the routed solution type is stored per draw, read back, and recovered into a sidecar (MOCK_POPULATION_SPEC §3.5)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop import proposal_set as ps

pytestmark = pytest.mark.unit


def _artifact(path: Path, with_label: bool) -> None:
    n = 4
    with h5py.File(path, "w") as h:
        t = h.create_group("truth")
        t.create_dataset("draw_index", data=np.arange(n, dtype=np.int64) + 10)
        o = h.create_group("outcome")
        o.create_dataset("solution_type", data=np.array([b"twelve_parameter_orbital"] * n, dtype="S32"))
        if with_label:
            o.create_dataset("nss_solution_type", data=np.array([b"Orbital", b"AstroSpectroSB1", b"", b"Orbital"], dtype="S32"))
        h.attrs["primary_model"] = "m"


def test_stored_label_is_decoded(tmp_path: Path) -> None:
    a = tmp_path / "a.h5"
    _artifact(a, True)
    _t, o, _attrs = ps.read_proposal_artifact(a)
    assert list(o["nss_solution_type"]) == ["Orbital", "AstroSpectroSB1", "", "Orbital"]


def test_sidecar_round_trip_and_loader_attaches_it(tmp_path: Path) -> None:
    a = tmp_path / "gen.h5"
    _artifact(a, False)
    assert ps.read_nss_type_sidecar(a) is None
    ps.write_nss_type_sidecar(a, [13, 12, 11, 10], ["Orbital", "", "AstroSpectroSB1", "Orbital"], provenance={"m": 1})
    assert ps.nss_type_sidecar_path(a).name == "gen.nss_type.h5"
    _t, outcome = ps.load_generation_draws([a], require_primary_model="m")
    assert dict(zip(_t["draw_index"].tolist(), outcome["nss_solution_type"].tolist())) == {
        13: "Orbital", 12: "", 11: "AstroSpectroSB1", 10: "Orbital"}


def test_unknown_label_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown nss_solution_type"):
        ps.write_nss_type_sidecar(tmp_path / "x.h5", [1], ["Acceleration"], provenance={})
