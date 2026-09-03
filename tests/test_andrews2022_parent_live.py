"""Live archive check for the frozen Andrews parent COUNT.

Kept out of ``test_andrews2022_selection.py`` so ``@pytest.mark.network`` is
not combined with a module-level ``unit`` mark (see test_data_acquisition_live).
"""

from __future__ import annotations

import pytest

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.sample_selection import load_sample_selection_file

pytestmark = pytest.mark.network

_ANDREWS = repo_root() / "config" / "selections" / "andrews2022.yaml"


def test_live_andrews_parent_count_matches_frozen_query() -> None:
    """Optional TAP replay of the frozen confirming ADQL; not in required CI."""
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.data_acquisition import default_gaia_query

    spec = load_sample_selection_file(_ANDREWS)
    dr3 = spec.parent_query.dr3  # type: ignore[union-attr]
    verifying = dr3.verification
    assert verifying is not None
    assert verifying.confirming_query is not None
    table = default_gaia_query(verifying.confirming_query, load_config().dr3)
    assert int(table["n"][0]) == 134598
    assert int(table["n"][0]) == verifying.confirmed_count
