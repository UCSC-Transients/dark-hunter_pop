"""#391 spec §12.34: bright spectroscopic routing."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import spectroscopic_routing as sr


@pytest.mark.unit
def test_grvs_relation_and_kobs() -> None:
    g = sr.grvs_from_g([10.0, 10.0], [0.8, 1.5])
    assert g[0] < 10.0 and g[1] < g[0]  # redder -> brighter in the RVS band
    assert sr.k_observed([10.0], [20.0], [0.0])[0] == pytest.approx(10.0)
    assert sr.k_observed([10.0], [20.0], [0.5])[0] == pytest.approx(0.0)  # flux-weighted line centre cancels


@pytest.mark.unit
def test_detection_and_factor() -> None:
    cfg = sr.RoutingConfig(s_min=5.0, p_route=0.3)
    det = sr.spectroscopic_detected([5.0, 0.1, 5.0], [8.0, 8.0, 12.5], [6000.0, 6000.0, 6000.0], cfg)
    assert det.tolist() == [True, False, False]
    assert sr.astrometric_publication_factor(det, cfg).tolist() == [0.3, 1.0, 1.0]
    assert sr.sigma_rv_epoch([8.0], [8000.0], cfg)[0] == pytest.approx(1.5)
