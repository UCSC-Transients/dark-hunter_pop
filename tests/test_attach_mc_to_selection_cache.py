"""Regression coverage for issue #267:
``scripts/attach_mc_to_selection_cache.py`` loaded the FLAME-merged
``+enrich+flame`` cache only for job enumeration and a coverage print, then
built the actual per-job payload sent to ``_mc_one`` from a *separate*,
FLAME-less table (``nss_enrichment/query.ecsv``) -- so every source's M1
Monte Carlo draw silently fell back to ``Uniform(uniform_low_msun,
uniform_high_msun)`` regardless of whether it had a real Gaia Apsis FLAME
mass. 632 tests passed throughout while this was live in production (the
2026-09-26 ~6h22m rebuild underlying #257/PR #266, reverted by PR #268).

These tests exercise the real ``_mc_one`` / ``_flame_value_lookup`` /
``_augment_job_payload_with_flame`` functions from the actual script (no
reimplementation), with a synthetic but fully valid FLAME-bearing Orbital
row (corr_vec / bit_index built the same way as
``tests/test_nss_covariance.py``'s ``_orbital_row`` helper), and assert the
resulting ``m1_msun_mc_sigma`` lands near the fixed FLAME-branch error
(``flame_fixed_error_msun``), not near the uniform-fallback standard
deviation ``(uniform_high_msun - uniform_low_msun) / sqrt(12)``.

Verified per the issue's instructions before finalizing this fix: with
``_augment_job_payload_with_flame`` reverted to drop ``flame_value`` (i.e.
``return list(enrich_cols), tuple(vals)`` -- exactly the pre-fix behavior,
since the buggy ``main()`` never folded the FLAME-merged cache's value into
the payload at all), ``test_augment_job_payload_with_flame_appends_column_and_value``
and ``test_main_job_construction_uses_flame_merged_rows_not_only_raw_enrich_table``
both fail -- the latter's ``m1_msun_mc_mean`` lands at ``0.8143`` (the
uniform-fallback mean), not the asserted ``1.3`` FLAME value. Restoring the
real implementation makes both pass again (5/5). See the PR description for
the literal before/after pytest output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import PrimaryMassSpec
from darkhunter_pop.nss_covariance import (
    fitted_params_from_bit_index,
    model_param_names,
    pack_corr_vec_upper_triangle,
)

pytestmark = pytest.mark.unit

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "attach_mc_to_selection_cache.py"
)

_ORBITAL_PARAMS = model_param_names("Orbital")
assert _ORBITAL_PARAMS is not None

_FLAME_FIXED_ERROR_MSUN = 0.1
_UNIFORM_LOW_MSUN = 0.63
_UNIFORM_HIGH_MSUN = 1.0
_UNIFORM_FALLBACK_SIGMA = (_UNIFORM_HIGH_MSUN - _UNIFORM_LOW_MSUN) / np.sqrt(12)
_FLAME_MASS_MSUN = 1.3  # well outside [0.63, 1.0] -- unambiguous vs. the fallback


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "attach_mc_to_selection_cache", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def attach_mod() -> ModuleType:
    return _load_module()


def _spd_corr(n: int, seed: int = 0) -> np.ndarray:
    """Build a random SPD correlation matrix (unit diagonal); mirrors
    ``tests/test_nss_covariance.py``'s helper of the same name."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T + n * np.eye(n)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def _synthetic_orbital_row(*, source_id: int, with_flame: bool) -> dict[str, Any]:
    """A fully reconstructible Orbital row (real ``corr_vec``/``bit_index``,
    so ``table_row_to_candidate`` actually yields a usable ``nss_solution``)
    -- optionally carrying a real ``mass_flame`` value, as the on-disk
    ``+enrich+flame`` cache does for a genuine FLAME-bearing source.
    """
    bit_index = 8191
    fitted = fitted_params_from_bit_index(bit_index, _ORBITAL_PARAMS)
    n = len(fitted)
    corr = _spd_corr(n, seed=source_id)
    errors = {name: 0.01 * (i + 1) for i, name in enumerate(fitted)}
    values = {name: 1.0 + 0.1 * i for i, name in enumerate(fitted)}
    row: dict[str, Any] = {
        "source_id": source_id,
        "nss_solution_type": "Orbital",
        "bit_index": bit_index,
        "corr_vec": pack_corr_vec_upper_triangle(corr),
        "goodness_of_fit": 3.0,
        "g_mag": 12.0,
    }
    shorts = {
        "a_thiele_innes": "A",
        "b_thiele_innes": "B",
        "f_thiele_innes": "F",
        "g_thiele_innes": "G",
    }
    for name in fitted:
        if name in shorts:
            short = shorts[name]
            row[short] = values[name]
            row[f"{short}_error"] = errors[name]
            continue
        row[name] = values[name]
        row[f"{name}_error"] = errors[name]
    if with_flame:
        row["mass_flame"] = _FLAME_MASS_MSUN
    return row


def _job_payload_for_row(
    row: dict[str, Any], *, n_draws: int, seed_base: int
) -> tuple[Any, ...]:
    """Build the same payload shape ``main()`` submits to ``_mc_one``,
    treating ``row`` as if it were both the raw ``nss_enrichment`` fetch
    (``enrich_table``) *and* the FLAME-merged cache row (``rows``) -- since
    the synthetic row already carries everything both real tables supply."""
    primary_mass = PrimaryMassSpec(
        method="flame_or_uniform_draw",
        flame_column="mass_flame",
        flame_fixed_error_msun=_FLAME_FIXED_ERROR_MSUN,
        uniform_low_msun=_UNIFORM_LOW_MSUN,
        uniform_high_msun=_UNIFORM_HIGH_MSUN,
    )
    cfg = load_config()
    dr_dump = cfg.active_dr().model_dump(mode="json")
    smf_dump = cfg.spectroscopic_mass_function.model_dump(mode="json")
    enrich_cols = list(row.keys())
    vals = tuple(row[c] for c in enrich_cols)
    return (
        row["source_id"],
        row["nss_solution_type"],
        enrich_cols,
        vals,
        primary_mass.model_dump(mode="json"),
        1.4,
        n_draws,
        seed_base,
        1e-12,
        1e-18,
        dr_dump,
        smf_dump,
    )


def test_flame_value_lookup_reads_flame_column_from_enrich_cache_rows(
    attach_mod: ModuleType,
) -> None:
    rows = [
        {"source_id": 1, "nss_solution_type": "Orbital", "mass_flame": 1.3},
        {"source_id": 2, "nss_solution_type": "Orbital"},
    ]
    lookup = attach_mod._flame_value_lookup(rows, "mass_flame")
    assert lookup[(1, "Orbital")] == pytest.approx(1.3)
    assert lookup[(2, "Orbital")] is None


def test_augment_job_payload_with_flame_appends_column_and_value(
    attach_mod: ModuleType,
) -> None:
    cols, vals = attach_mod._augment_job_payload_with_flame(
        ["source_id", "corr_vec"], (1, [0.1, 0.2]), "mass_flame", 1.3
    )
    assert cols == ["source_id", "corr_vec", "mass_flame"]
    assert vals == (1, [0.1, 0.2], 1.3)


def test_mc_one_uses_flame_mass_when_payload_carries_it(
    attach_mod: ModuleType,
) -> None:
    """With the FLAME value actually present in the per-job payload,
    ``_mc_one`` must draw M1 from the fixed-error FLAME branch, not the
    uniform fallback. See
    ``test_main_job_construction_uses_flame_merged_rows_not_only_raw_enrich_table``
    below for the test that fails against the pre-#267-fix payload
    construction (this one builds the payload directly, so it does not by
    itself exercise ``main()``'s two-table merge).
    """
    row = _synthetic_orbital_row(source_id=42, with_flame=True)
    payload = _job_payload_for_row(row, n_draws=20_000, seed_base=0)

    sid, q = attach_mod._mc_one(payload)

    assert sid == 42
    assert q is not None
    # Fixed FLAME-branch error (flame_fixed_error_msun=0.1): well below the
    # uniform fallback's ~0.1068, with wide margin for 20k-draw MC noise.
    assert q["m1_msun_mc_sigma"] < 0.5 * (_UNIFORM_FALLBACK_SIGMA + _FLAME_FIXED_ERROR_MSUN)
    assert q["m1_msun_mc_sigma"] == pytest.approx(_FLAME_FIXED_ERROR_MSUN, abs=0.01)
    assert q["m1_msun_mc_mean"] == pytest.approx(_FLAME_MASS_MSUN, abs=0.02)


def test_mc_one_falls_back_to_uniform_when_flame_absent_from_payload(
    attach_mod: ModuleType,
) -> None:
    """Documents the (correct) fallback: a source that genuinely has no
    FLAME mass still draws Uniform(low, high) -- this is not the bug, and
    must keep working. The #267 bug was that *every* source hit this path
    even when the FLAME-merged cache held a real value for it."""
    row = _synthetic_orbital_row(source_id=43, with_flame=False)
    payload = _job_payload_for_row(row, n_draws=20_000, seed_base=0)

    sid, q = attach_mod._mc_one(payload)

    assert sid == 43
    assert q is not None
    assert q["m1_msun_mc_mean"] == pytest.approx(
        (_UNIFORM_LOW_MSUN + _UNIFORM_HIGH_MSUN) / 2, abs=0.02
    )
    assert q["m1_msun_mc_sigma"] == pytest.approx(_UNIFORM_FALLBACK_SIGMA, abs=0.01)


def test_main_job_construction_uses_flame_merged_rows_not_only_raw_enrich_table(
    attach_mod: ModuleType,
) -> None:
    """End-to-end through the actual helpers ``main()`` calls: given a
    FLAME-merged-cache ``rows`` entry with a real ``mass_flame`` and a
    *separate* raw-enrichment ``by_key`` entry (modeling
    ``nss_enrichment/query.ecsv``, which never carries ``mass_flame``), the
    job payload assembled the way ``main()`` assembles it must still carry
    the FLAME value through to ``_mc_one``.
    """
    full_row = _synthetic_orbital_row(source_id=44, with_flame=True)
    # The raw nss_enrichment table has everything except mass_flame/source
    # identity duplicated as a plain dict -- model it as the same row minus
    # the FLAME column, exactly like the real `enrich_table` would be.
    raw_enrich_row = {k: v for k, v in full_row.items() if k != "mass_flame"}
    enrich_cols = list(raw_enrich_row.keys())
    vals = tuple(raw_enrich_row[c] for c in enrich_cols)

    flame_by_key = attach_mod._flame_value_lookup([full_row], "mass_flame")
    enrich_cols_full, full_vals = attach_mod._augment_job_payload_with_flame(
        enrich_cols, vals, "mass_flame", flame_by_key[(44, "Orbital")]
    )

    primary_mass = PrimaryMassSpec(
        method="flame_or_uniform_draw",
        flame_column="mass_flame",
        flame_fixed_error_msun=_FLAME_FIXED_ERROR_MSUN,
        uniform_low_msun=_UNIFORM_LOW_MSUN,
        uniform_high_msun=_UNIFORM_HIGH_MSUN,
    )
    cfg = load_config()
    payload = (
        44,
        "Orbital",
        enrich_cols_full,
        full_vals,
        primary_mass.model_dump(mode="json"),
        1.4,
        20_000,
        0,
        1e-12,
        1e-18,
        cfg.active_dr().model_dump(mode="json"),
        cfg.spectroscopic_mass_function.model_dump(mode="json"),
    )

    sid, q = attach_mod._mc_one(payload)

    assert sid == 44
    assert q is not None
    assert q["m1_msun_mc_mean"] == pytest.approx(_FLAME_MASS_MSUN, abs=0.02)
    assert q["m1_msun_mc_sigma"] == pytest.approx(_FLAME_FIXED_ERROR_MSUN, abs=0.01)
