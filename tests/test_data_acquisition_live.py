"""Live-network smoke test for data_acquisition, deliberately kept out of
tests/test_data_acquisition.py.

That module sets ``pytestmark = pytest.mark.unit`` for its whole file; a
per-test ``@pytest.mark.network`` decorator does not override a module-level
``pytestmark`` (marks are additive, not exclusive), so a test placed there
would carry *both* marks and would still be selected by the required CI job's
``-m "unit or physics or api"`` filter despite hitting the real ESA Gaia
archive. Keeping this test in its own, non-``unit``-marked module is what
actually keeps it out of the required merge gate.
"""

from __future__ import annotations

import pytest

from darkhunter_pop.config_loader import load_config

pytestmark = pytest.mark.network


def test_live_gaia_nss_query_smoke() -> None:
    """Optional live archive query; excluded from the required CI marker set."""
    from darkhunter_pop.data_acquisition import default_gaia_query

    dr = load_config().dr3
    adql = (
        f"SELECT TOP 3 nss.source_id, nss.goodness_of_fit, gs.phot_g_mean_mag "
        f"FROM {dr.nss_table} AS nss "
        f"JOIN {dr.gaia_source_table} AS gs ON nss.source_id = gs.source_id"
    )
    table = default_gaia_query(adql, dr)
    assert len(table) <= 3
    assert "source_id" in table.colnames
