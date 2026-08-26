"""Optional network/slow live-pull surface for comparison catalogs (issue #70).

Not part of the required merge gate (``unit|physics|api``). Fixture loaders are
the default path; this documents the opt-in live-pull API.
"""

from __future__ import annotations

import pytest

from darkhunter_pop.benchmarks import fetch_live_comparison_catalog

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_live_pull_optional_surface() -> None:
    """Live pulls are optional; fixture path is the merge-gate default."""
    with pytest.raises(NotImplementedError, match="never a prior"):
        fetch_live_comparison_catalog("ns_candidate_21", url="https://example.invalid")
