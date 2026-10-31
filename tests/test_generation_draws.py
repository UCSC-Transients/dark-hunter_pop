"""#601 (spec §12.16): current-version generation artifacts read directly, with the version guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import proposal_set as ps

pytestmark = pytest.mark.unit

V_NEW = "looks_single_v2:ruwe1.25:dm-0.4:c1.05-2.75:g19:noise_colour_v2.1+tr1+e2+st1"
V_OLD = "looks_single_v2:ruwe1.25:dm-0.4:c1.05-2.75:g19:noise_colour_v2.1+tr1"


def _artifact(path: Path, version: str, draw_index: np.ndarray) -> Path:
    import h5py

    n = draw_index.size
    with h5py.File(path, "w") as h:
        h.attrs["primary_model"] = version
        h.create_dataset("truth/draw_index", data=draw_index.astype(np.int64))
        h.create_dataset("truth/period_days", data=np.linspace(10.0, 500.0, n))
        h.create_dataset("outcome/cascade", data=np.zeros((n, 23)))
        h.create_dataset("outcome/accepted_orbital", data=np.arange(n) % 2 == 0)
        h.create_dataset("outcome/solution_type", data=np.array([b"twelve_parameter_orbital"] * n))
    return path


def test_concatenates_one_version(tmp_path: Path) -> None:
    a = _artifact(tmp_path / "g53.h5", V_NEW, np.arange(10_000_000, 10_000_005))
    b = _artifact(tmp_path / "g54.h5", V_NEW, np.arange(10_500_000, 10_500_003))
    truth, outcome = ps.load_generation_draws([a, b], require_primary_model=V_NEW)
    assert truth["draw_index"].size == 8 and outcome["cascade"].shape == (8, 23)
    assert outcome["accepted_orbital"].dtype == bool


def test_refuses_other_versions_and_duplicates(tmp_path: Path) -> None:
    a = _artifact(tmp_path / "g53.h5", V_NEW, np.arange(10_000_000, 10_000_005))
    old = _artifact(tmp_path / "g50.h5", V_OLD, np.arange(8_500_000, 8_500_005))
    with pytest.raises(ValueError, match="never mixed"):
        ps.load_generation_draws([a, old], require_primary_model=V_NEW)
    dup = _artifact(tmp_path / "dup.h5", V_NEW, np.arange(10_000_000, 10_000_002))
    with pytest.raises(ValueError, match="unique"):
        ps.load_generation_draws([a, dup], require_primary_model=V_NEW)
    with pytest.raises(ValueError):
        ps.load_generation_draws([], require_primary_model=V_NEW)


def test_gen53_fragment_carries_the_current_version() -> None:
    frag = ps.load_proposal_set_fragment(Path(__file__).resolve().parents[1] / "config/population/proposal_set_single53.yaml")
    assert ps.primary_model_version(frag.proposal) == V_NEW
