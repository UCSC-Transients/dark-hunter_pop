"""Live archive check for the frozen El-Badry 2024 parent COUNT."""

from __future__ import annotations

import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.sample_selection import load_sample_selection_file

pytestmark = pytest.mark.network

_ELBADRY = repo_root() / "config" / "selections" / "elbadry2024.yaml"


def test_live_elbadry2024_parent_count_matches_frozen_query() -> None:
    """Optional TAP replay of the frozen confirming ADQL; not in required CI."""
    from darkhunter_pop.data_acquisition import default_gaia_query

    spec = load_sample_selection_file(_ELBADRY)
    assert spec.branches is not None
    dr3 = spec.branches[0].parent_query.dr3
    verifying = dr3.verification
    assert verifying is not None
    assert verifying.confirming_query is not None
    table = default_gaia_query(verifying.confirming_query, load_config().dr3)
    assert int(table["n"][0]) == 168065
    assert int(table["n"][0]) == verifying.confirmed_count
