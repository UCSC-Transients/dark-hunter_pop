"""#681: El-Badry 2026 columns on mock rows come from the real-row code, and match it on the same observables.

Real-style rows (NSS ``corr_vec`` / ``bit_index`` / errors, reconstructed by ``reconstruct_nss_covariance``) and
mock rows (cascade observables + ``mock_orbit_p0`` / ``mock_orbit_cov``) are built from the *same* orbit numbers and
pushed through the evaluator's own ``_enrich_rows_for_spec`` / sigma / AMRF-threshold steps. Every column must agree.
Synthetic (no data/); the dust maps are the constant-E(B-V) stub of ``test_mock_selection_rows``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import darkhunter_pop.data_acquisition as da
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.elbadry2026_m2_sigma import is_elbadry_sigma_provenance
from darkhunter_pop.elbadry2026_selection import (
    AMRF_THRESHOLD_COLUMN,
    AMRF_THRESHOLD_SOURCE_COLUMN,
    attach_amrf_threshold,
)
from darkhunter_pop.mock_selection_rows import (
    CASCADE_COLUMNS,
    forward_model_selection,
    mock_draws_to_selection_rows,
    mock_orbit_parameter_set,
)
from darkhunter_pop.mock_atf_columns import MOCK_ORBIT_NAMES
from darkhunter_pop.nss_covariance import (
    correlation_to_covariance,
    model_param_names,
    pack_corr_vec_upper_triangle,
)
from darkhunter_pop.sample_selection import NotApplicable, SampleSelectionRegistry

pytestmark = pytest.mark.unit

_N = 6
_EBV = 0.05
_REAL_TO_MOCK = {"ra": "ra_offset", "dec": "dec_offset", "t_periastron": "phi_p"}
_SHORT = {"a_thiele_innes": "A", "b_thiele_innes": "B", "f_thiele_innes": "F", "g_thiele_innes": "G"}


class _ConstantEbv:
    def __init__(self, dust_cfg: Any) -> None:
        self.dust_cfg = dust_cfg

    def ebv_for(self, ids: Any, ra: Any, dec: Any, plx: Any) -> tuple[Any, Any, Any]:
        n = len(ids)
        return np.full(n, _EBV), np.zeros(n, dtype=int), np.zeros(n, dtype=int)


def _orbits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-draw 12-vector (MOCK_ORBIT_NAMES order), its covariance, and a correlation matrix source."""
    rng = np.random.default_rng(681)
    n = len(MOCK_ORBIT_NAMES)
    p0 = np.zeros((_N, n))
    cov = np.zeros((_N, n, n))
    for k in range(_N):
        a0 = rng.uniform(1.5, 4.0)
        om, node, inc = rng.uniform(0.2, 1.2, 3)
        a = a0 * (np.cos(om) * np.cos(node) - np.sin(om) * np.sin(node) * np.cos(inc))
        b = a0 * (np.cos(om) * np.sin(node) + np.sin(om) * np.cos(node) * np.cos(inc))
        f = a0 * (-np.sin(om) * np.cos(node) - np.cos(om) * np.sin(node) * np.cos(inc))
        g = a0 * (-np.sin(om) * np.sin(node) + np.cos(om) * np.cos(node) * np.cos(inc))
        vals = dict(ra_offset=0.0, dec_offset=0.0, parallax=rng.uniform(1.5, 4.0), pmra=1.0, pmdec=-2.0,
                    period=rng.uniform(200.0, 800.0), eccentricity=rng.uniform(0.0, 0.3), phi_p=0.3,
                    a_thiele_innes=a, b_thiele_innes=b, f_thiele_innes=f, g_thiele_innes=g)
        p0[k] = [vals[name] for name in MOCK_ORBIT_NAMES]
        sig = np.maximum(np.abs(p0[k]) * 0.02, 1e-3)
        m = rng.normal(size=(n, n))
        corr = m @ m.T + n * np.eye(n)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        cov[k] = corr * np.outer(sig, sig)
    return p0, cov, np.zeros(1)


def _mock_rows() -> list[dict[str, Any]]:
    p0, cov, _ = _orbits()
    names = list(MOCK_ORBIT_NAMES)
    sig = np.sqrt(np.einsum("kii->ki", cov))
    cascade = np.zeros((_N, len(CASCADE_COLUMNS)))
    for k in range(_N):
        v = dict(zip(names, p0[k], strict=True))
        s = dict(zip(names, sig[k], strict=True))
        for name, val in {
            "plx": v["parallax"], "sig_parallax": s["parallax"], "period": v["period"], "sig_period": s["period"],
            "ecc": v["eccentricity"], "sig_ecc": s["eccentricity"], "F2": 3.0, "ruwe": 1.5,
            "N_visibility_periods": 20.0, "N_obs": 300.0,
            **{short: v[long] for long, short in _SHORT.items()},
            **{f"sig_{short}": s[long] for long, short in _SHORT.items()},
        }.items():
            cascade[k, CASCADE_COLUMNS.index(name)] = val
    ids = np.array([101_000_000 + k for k in range(_N)], dtype=np.int64)
    truth = {
        "draw_index": ids, "source_id": np.full(_N, 4_000_000_000_000_000_000, dtype=np.int64),
        "ra_deg": np.full(_N, 150.0), "dec_deg": np.full(_N, 20.0),
        "phot_g_mean_mag": np.array([13.0, 13.5, 12.0, 14.0, 8.0, 14.5]),  # draw 4: giant (off the main sequence)
        "sys_bp_rp": np.array([0.85, 0.9, 0.8, 1.0, 1.4, 0.95]),
        "m1_row_msun": np.full(_N, 0.9), "m1_msun": np.full(_N, 0.9), "m2_msun": np.full(_N, 3.0),
        "is_dark": np.ones(_N, dtype=bool), "co_component": np.ones(_N, dtype=np.int8),
        "period_days": p0[:, names.index("period")], "sys_evolved": np.zeros(_N, dtype=bool),
        "parent_row": np.zeros(_N, dtype=np.int64),
    }
    outcome = {
        "cascade": cascade, "accepted_orbital": np.ones(_N, dtype=bool),
        "nss_solution_type": np.array(["Orbital"] * _N),  # the NSS bit_index layout used here exists for Orbital
        "orbit_p0": p0, "orbit_cov": cov,
    }
    return mock_draws_to_selection_rows(
        truth, outcome, pipeline_m1=(np.full(_N, 0.9), np.full(_N, 0.05)), include_orbit=True
    )


def _real_rows(mock: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    """Real-style rows: the mock row's observables, no mock_* keys, covariance as NSS ``corr_vec`` + errors."""
    real: list[dict[str, Any]] = []
    index: dict[Any, dict[str, Any]] = {}
    fitted = model_param_names("Orbital")
    assert fitted is not None
    for row in mock:
        cov = np.asarray(row["mock_orbit_cov"])
        p0 = dict(zip(MOCK_ORBIT_NAMES, row["mock_orbit_p0"], strict=True))
        sig = dict(zip(MOCK_ORBIT_NAMES, np.sqrt(np.diag(cov)), strict=True))
        # NSS Orbital order is the ATF/Gaia order (t_periastron for phi_p); only the six sigma inputs matter.
        order = list(fitted)
        src = [_REAL_TO_MOCK.get(n, n) for n in fitted]
        mi = [MOCK_ORBIT_NAMES.index(n) for n in src]
        corr = cov[np.ix_(mi, mi)] / np.outer(np.sqrt(np.diag(cov))[mi], np.sqrt(np.diag(cov))[mi])
        out = {k: v for k, v in row.items() if not k.startswith("mock_")}
        enrich = {
            "source_id": row["source_id"], "nss_solution_type": row["nss_solution_type"],
            "bit_index": 8191, "corr_vec": pack_corr_vec_upper_triangle(corr),
        }
        for n, s in zip(order, src, strict=True):
            val, err = float(p0[s]), float(sig[s])
            enrich[n] = val
            enrich[f"{n}_error"] = err
        for long, short in _SHORT.items():
            enrich[short] = enrich[long]
            enrich[f"{short}_error"] = enrich[f"{long}_error"]
        enrich["period"], enrich["period_error"] = row["period"], row["period_error"]
        index[da._enrichment_join_key(row)] = enrich
        real.append(out)
    assert correlation_to_covariance is not None
    return real, index


@pytest.fixture(scope="module")
def registry() -> SampleSelectionRegistry:
    return SampleSelectionRegistry(load_config(), repo=repo_root())


def _run(sel: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The evaluator's own steps, in its order: enrichment, then sigma and AMRF threshold on the survivors."""
    enriched = [dict(r) for r in sel._enrich_rows_for_spec(rows)]
    sel._attach_elbadry_m2_sigma_inplace(enriched)
    return [attach_amrf_threshold(r, sel.spec) for r in enriched]


def test_mock_columns_equal_real_row_columns(registry: SampleSelectionRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _mock_rows()
    real, index = _real_rows(mock)
    sel_mock = forward_model_selection(registry, "elbadry2026", extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    sel_real = forward_model_selection(registry, "elbadry2026", extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    monkeypatch.setattr(sel_mock, "_nss_enrichment_index", lambda: {})  # the mock path never touches the real index
    monkeypatch.setattr(sel_real, "_nss_enrichment_index", lambda: index)
    got, want = _run(sel_mock, mock), _run(sel_real, real)
    n_sigma = 0
    for g, w in zip(got, want, strict=True):
        for col in ("main_sequence", "mg_0", "bp_rp_0", "m1_tilde_msun", "m2_tilde_msun", "amrf", "a0_mas",
                    "goodness_of_fit", AMRF_THRESHOLD_COLUMN, AMRF_THRESHOLD_SOURCE_COLUMN,
                    "_sigma_m2_astrometric_provenance"):
            gv, wv = g.get(col), w.get(col)
            if isinstance(wv, NotApplicable):
                assert isinstance(gv, NotApplicable) and gv.reason == wv.reason, col
            else:
                assert gv == wv, col
        gs, ws = g.get("sigma_m2_astrometric_msun"), w.get("sigma_m2_astrometric_msun")
        assert (gs is None) == (ws is None)
        if ws is not None:
            n_sigma += 1
            # only differs by the rounding of cov -> (corr, sigma) -> cov
            assert gs == pytest.approx(ws, rel=1e-9)
            assert is_elbadry_sigma_provenance(g["_sigma_m2_astrometric_provenance"])
    assert n_sigma >= 4  # the giant has no M1-tilde (not applicable), so it carries no sigma
    assert any(r["main_sequence"] is False for r in got)


def test_no_orbit_no_sigma_and_no_diagonal(registry: SampleSelectionRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    mock = [{k: v for k, v in r.items() if k not in ("mock_orbit_p0", "mock_orbit_cov")} for r in _mock_rows()]
    sel = forward_model_selection(registry, "elbadry2026", extinction_lookup=_ConstantEbv(registry.cfg.dust_maps))
    monkeypatch.setattr(sel, "_nss_enrichment_index", lambda: {})
    assert all("sigma_m2_astrometric_msun" not in r for r in _run(sel, mock))


def test_orbit_parameter_set_rejects_bad_input() -> None:
    row = _mock_rows()[0]
    ps = mock_orbit_parameter_set(row)
    assert ps is not None and list(ps.names) == list(MOCK_ORBIT_NAMES)
    bad = dict(row, mock_orbit_cov=np.full((12, 12), np.nan))
    assert mock_orbit_parameter_set(bad) is None
    skew = np.array(row["mock_orbit_cov"], copy=True)
    skew[0, 1] += 10.0 * abs(skew[0, 1]) + 1.0
    assert mock_orbit_parameter_set(dict(row, mock_orbit_cov=skew)) is None
    assert mock_orbit_parameter_set({k: v for k, v in row.items() if k != "mock_orbit_p0"}) is None
