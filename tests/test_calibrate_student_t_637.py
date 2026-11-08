"""#637: ν fit, profile interval and outlier mixture of scripts/calibrate_student_t_564.py (synthetic pulls only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_P = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_student_t_564.py"
_s = importlib.util.spec_from_file_location("calibrate_student_t_564", _P)
mod = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mod)  # type: ignore[union-attr]


@pytest.mark.unit
def test_fit_nu_recovers_and_brackets() -> None:
    x = stats.t.rvs(df=4.0, size=4000, random_state=np.random.default_rng(1))
    nu, lo, hi = mod.fit_nu(x, (1.0, 300.0))
    assert lo <= nu <= hi and 3.0 < nu < 5.5


@pytest.mark.unit
def test_mixture_separates_catastrophic_outliers() -> None:
    rng = np.random.default_rng(2)
    core = rng.standard_normal(1000)
    x = np.concatenate([core, rng.normal(0.0, 2000.0, 30)])
    nu_plain, _, _ = mod.fit_nu(x, (1.0, 300.0))
    mix = mod.fit_outlier_mixture(x, (1.0, 300.0))
    assert nu_plain < 2.0
    assert mix["nu_core"] > 10.0 and 0.01 < mix["f_outlier"] < 0.06 and mix["sigma_outlier"] > 100.0
