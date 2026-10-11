"""Tests for the rung-3 fit-model pieces (#391; spec §12.15, §12.16)."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import rung3_model as r3

CAL = {"pi_keys": ["log10_a0_snr"], "pi_beta": [-1.5, -0.6], "standardize_mean": {"log10_a0_snr": 1.2},
       "standardize_sd": {"log10_a0_snr": 0.3}, "calibrated_range": {"log10_a0_snr": [0.7, 2.0]}}


@pytest.mark.unit
def test_evolved_modifier_only_on_evolved_rows() -> None:
    evo = np.array([0, 1, 1, 0], bool)
    np.testing.assert_array_equal(r3.evolved_log_modifier(evo, 0.3), [0.0, 0.3, 0.3, 0.0])
    np.testing.assert_array_equal(r3.evolved_log_modifier(evo, 0.0), np.zeros(4))


@pytest.mark.unit
def test_spurious_probability_monotone_clamped_and_finite() -> None:
    x = np.array([0.1, 0.7, 1.2, 2.0, 3.5, np.nan])
    p = r3.spurious_probability(CAL, {"log10_a0_snr": x})
    assert np.all(np.isfinite(p)) and np.all((p > 0) & (p < 1))
    assert p[1] > p[2] > p[3]  # lower significance -> more spurious (negative slope)
    assert p[0] == pytest.approx(p[1]) and p[4] == pytest.approx(p[3])  # clamped outside the calibrated range
    pl = r3.spurious_probability(CAL, {"log10_a0_snr": x}, extrapolation="logit")
    assert pl[0] > p[0] and pl[4] < p[4]  # logit extrapolation continues the trend
    with pytest.raises(ValueError):
        r3.spurious_probability(CAL, {"log10_a0_snr": x}, extrapolation="zero")  # type: ignore[arg-type]


@pytest.mark.unit
def test_spurious_template_normalized_sum() -> None:
    idx = np.array([0, 0, 2, -1, 2, 1])
    pi = np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.5])
    t, tot = r3.spurious_template(idx, pi, 4)
    assert tot == pytest.approx(1.2)
    np.testing.assert_allclose(t, np.array([0.3, 0.5, 0.4, 0.0]) / 1.2)
