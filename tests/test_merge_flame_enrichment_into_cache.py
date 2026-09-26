"""Regression coverage for scripts/merge_flame_enrichment_into_cache.py's
numpy-float-to-JSON-string bug (#257 follow-up).

Caught before it reached the production MC rebuild: astropy ``Table``
columns hold ``numpy.float32``/``float64`` scalars, which are **not**
JSON-native. ``sample_selection._write_selection_parent_cache`` serializes
each row with ``json.dumps(dict(row), default=str)`` — for a raw numpy
scalar this silently falls through to ``str(value)`` (e.g. ``"2.802136"``),
which then round-trips back as a Python ``str``, not a ``float``. Every
downstream ``isinstance(x, (int, float))`` check then silently sees "no
FLAME mass" for every row, even though real values were merged in — this
was caught live (0/443211 rows detected as having FLAME mass despite the
merge step itself reporting 241196) before the multi-hour MC rebuild
consumed the corrupted cache.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_flame_enrichment_into_cache.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "merge_flame_enrichment_into_cache", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merge_mod() -> ModuleType:
    return _load_module()


def test_native_float_or_none_casts_numpy_float32_to_plain_python_float(
    merge_mod: ModuleType,
) -> None:
    value = np.float32(2.8021364)
    out = merge_mod._native_float_or_none(value)
    assert type(out) is float
    assert out == pytest.approx(2.8021364, rel=1e-6)


def test_native_float_or_none_returns_none_for_masked(merge_mod: ModuleType) -> None:
    assert merge_mod._native_float_or_none(np.ma.masked) is None


def test_native_float_or_none_returns_none_for_non_finite(merge_mod: ModuleType) -> None:
    assert merge_mod._native_float_or_none(np.float32("nan")) is None
    assert merge_mod._native_float_or_none(np.float32("inf")) is None


def test_native_float_or_none_returns_none_for_unconvertible(merge_mod: ModuleType) -> None:
    assert merge_mod._native_float_or_none(object()) is None


def test_raw_numpy_float32_would_have_silently_become_a_string(
    merge_mod: ModuleType,
) -> None:
    """Documents the exact failure mode this fix guards against: without the
    cast, json.dumps(..., default=str) -- what
    sample_selection._write_selection_parent_cache actually uses -- turns a
    real numpy.float32 measurement into a numeric-looking *string* rather
    than raising, so the corruption is silent.
    """
    raw = np.float32(2.8021364)
    with pytest.raises(TypeError):
        json.dumps(raw)
    stringified = json.dumps(raw, default=str)
    assert isinstance(json.loads(stringified), str)


def test_cast_value_round_trips_as_float_through_the_real_cache_writer(
    merge_mod: ModuleType, tmp_path: Path
) -> None:
    """End-to-end: a _native_float_or_none-cast value survives the actual
    write_selection_parent_cache / read_selection_parent_cache round trip as
    a float, not a string -- the concrete symptom (isinstance(...,
    (int, float)) silently failing for every row) is what this proves fixed.
    """
    from darkhunter_pop.sample_selection import (
        _read_selection_parent_cache,
        _write_selection_parent_cache,
    )

    raw = np.float32(2.8021364)
    cast = merge_mod._native_float_or_none(raw)
    row = {"source_id": 1, "nss_solution_type": "Orbital", "mass_flame": cast}

    path = tmp_path / "selection_parent_rows.h5"
    _write_selection_parent_cache(path, [row])
    (read_row,) = _read_selection_parent_cache(path)

    assert isinstance(read_row["mass_flame"], (int, float))
    assert read_row["mass_flame"] == pytest.approx(2.8021364, rel=1e-6)
