"""Mock ``andrews_atf_*`` columns go through the real-row ATF code path (#679).

Synthetic tests pin the covariance slot ordering (a mismatch is a silent wrong answer). The data-backed test feeds
real DR3 NSS-derived orbits through the *mock* path (converted to the mock's gaiamock parameter order) and requires
the forward-model sidecar columns of ``scripts/build_andrews2022_atf_columns.py`` back; it skips without ``data/``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkhunter_pop.andrews2022_atf import (
    ALL_COLUMNS,
    build_notebook_covariance,
    load_reproduction_columns,
    notebook_float,
    require_procedure,
)
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import SampleSelectionMode
from darkhunter_pop.mock_atf_columns import (
    ATF_FROM_MOCK,
    MOCK_ORBIT_NAMES,
    compute_atf_columns,
    mock_orbit_to_atf_frame,
    mock_orbit_to_atf_inputs,
)
from darkhunter_pop.mass_derivation import pipeline_tag10_m1
from darkhunter_pop.proposal_set import ORBIT_PARAMETER_NAMES
from darkhunter_pop.sample_selection import SampleSelectionRegistry

pytestmark = pytest.mark.unit

ATF_ORDER = (
    "ra", "dec", "parallax", "pmra", "pmdec", "a_thiele_innes", "b_thiele_innes", "f_thiele_innes",
    "g_thiele_innes", "eccentricity", "period", "t_periastron",
)
_TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def cfg() -> Any:
    return load_config()


@pytest.fixture(scope="module")
def spec(cfg: Any) -> Any:
    return SampleSelectionRegistry(cfg, repo=repo_root()).resolved("andrews2022")


def _mock_from_atf(means: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``mock_orbit_to_atf_frame`` (test helper): ATF-slot ``(means, cov)`` -> gaiamock order."""
    n = len(ATF_ORDER)
    it, ip = ATF_ORDER.index("t_periastron"), ATF_ORDER.index("period")
    period, t_p = means[ip], means[it]
    phi = _TWO_PI * t_p / period
    jac = np.eye(n)
    jac[it, it] = period / _TWO_PI
    jac[it, ip] = phi / _TWO_PI
    jinv = np.linalg.inv(jac)
    cov_phi = jinv @ cov @ jinv.T
    m = means.copy()
    m[it] = phi
    perm = [MOCK_ORBIT_NAMES.index(ATF_FROM_MOCK[name]) for name in ATF_ORDER]  # ATF slot -> mock index
    p0 = np.empty(n)
    c0 = np.empty((n, n))
    p0[perm] = m
    c0[np.ix_(perm, perm)] = cov_phi
    return p0, c0


def test_orders_are_pinned(spec: Any) -> None:
    assert tuple(ORBIT_PARAMETER_NAMES) == MOCK_ORBIT_NAMES
    assert tuple(require_procedure(spec).covariance.parameter_order) == ATF_ORDER
    assert sorted(ATF_FROM_MOCK) == sorted(ATF_ORDER)
    assert sorted(ATF_FROM_MOCK.values()) == sorted(MOCK_ORBIT_NAMES)


def test_slot_values_land_in_the_named_atf_slot() -> None:
    """Each mock parameter value arrives in the ATF slot of the same physical quantity."""
    p0 = np.arange(1.0, 13.0) * 10.0  # value 10*(mock index+1)
    cov = np.diag(np.arange(1.0, 13.0))
    means, cc = mock_orbit_to_atf_frame(p0, cov, ATF_ORDER)
    for slot, name in enumerate(ATF_ORDER):
        if name == "t_periastron":
            continue
        k = MOCK_ORBIT_NAMES.index(ATF_FROM_MOCK[name])
        assert means[slot] == p0[k]
        assert cc[slot, slot] == cov[k, k]
    # an off-diagonal element follows its parameters, not its position
    cov2 = np.eye(12)
    ia, ig = MOCK_ORBIT_NAMES.index("a_thiele_innes"), MOCK_ORBIT_NAMES.index("g_thiele_innes")
    cov2[ia, ig] = cov2[ig, ia] = 0.3
    _m, c2 = mock_orbit_to_atf_frame(p0, cov2, ATF_ORDER)
    assert c2[ATF_ORDER.index("a_thiele_innes"), ATF_ORDER.index("g_thiele_innes")] == 0.3
    # t_periastron = phi_p P / 2 pi
    assert means[ATF_ORDER.index("t_periastron")] == pytest.approx(
        p0[MOCK_ORBIT_NAMES.index("phi_p")] * p0[MOCK_ORBIT_NAMES.index("period")] / _TWO_PI
    )


def test_wrong_parameter_set_is_refused() -> None:
    with pytest.raises(ValueError):
        mock_orbit_to_atf_frame(np.zeros(12), np.eye(12), ATF_ORDER[:-1] + ("bogus",))


def test_notebook_covariance_reproduces_mock_covariance(spec: Any) -> None:
    """errors + corr_vec of the mock inputs rebuild exactly the (slot-ordered) mock covariance."""
    rng = np.random.default_rng(1)
    proc = require_procedure(spec)
    a = rng.normal(size=(12, 12))
    cov = a @ a.T + 12.0 * np.eye(12)
    p0 = np.abs(rng.normal(size=12)) + 1.0
    item = mock_orbit_to_atf_inputs(5, p0, cov, proc=proc, goodness_of_fit=1.0, g_mag=12.0, bp_rp=0.8)
    _m, expected = mock_orbit_to_atf_frame(p0, cov, ATF_ORDER)
    np.testing.assert_allclose(build_notebook_covariance(item.errors, item.corr_vec), expected, rtol=1e-12)


def test_nonfinite_covariance_is_a_covariance_failure_not_a_diagonal(spec: Any) -> None:
    proc = require_procedure(spec)
    item = mock_orbit_to_atf_inputs(
        7, np.ones(12), np.full((12, 12), np.nan), proc=proc, goodness_of_fit=1.0, g_mag=12.0, bp_rp=0.8
    )
    out = compute_atf_columns([item], spec)[7]
    assert out["andrews_atf_covariance_ok"] is False
    assert out["andrews_atf_covariance_failure"] is not None
    assert out["andrews_atf_m1_source"] is None and math.isnan(out["andrews_atf_p_m2_above"])


# ---------------------------------------------------------------------------
# Real rows through the mock path reproduce the real-row sidecar (needs data/).
# ---------------------------------------------------------------------------

_UNCUT = "20260826T234425Z_3d3f740b080c"
_N_REAL = 8


@pytest.fixture(scope="module")
def real_inputs(cfg: Any, spec: Any) -> dict[str, Any]:
    root = repo_root()
    data = Path(cfg.paths.data_root)
    data = data if data.is_absolute() else root / data
    snaps = data / "dr3" / "gaia_snapshots"
    meta_nss, meta_uncut = snaps / "nss_enrichment" / "meta.yaml", snaps / _UNCUT / "meta.yaml"
    side = load_reproduction_columns(spec, cfg, repo=root, mode=SampleSelectionMode.FORWARD_MODEL)
    if not (meta_nss.is_file() and meta_uncut.is_file() and side):
        pytest.skip("needs data/ (NSS enrichment, uncut snapshot, forward-model ATF sidecar)")
    from darkhunter_pop.data_acquisition import load_gaia_snapshot

    # Pass-1 survivors (pipeline-M1 and uniform-fallback) plus rows pass 1 rejected or the covariance gate failed.
    surv = [s for s, c in side.items() if c["andrews_atf_m1_source"] is not None]
    pipe = [s for s in surv if side[s]["andrews_atf_m1_source"] == "pipeline_tag10_bulk"][: _N_REAL - 3]
    uni = [s for s in surv if side[s]["andrews_atf_m1_source"] == "uniform"][:1]
    rej = [s for s, c in side.items() if c["andrews_atf_m1_source"] is None and c["andrews_atf_covariance_ok"]][:1]
    bad = [s for s, c in side.items() if not c["andrews_atf_covariance_ok"]][:1]
    want = set(pipe + uni + rej + bad)
    _m, nss = load_gaia_snapshot(meta_nss, verify_checksum=False)
    _m, uncut = load_gaia_snapshot(meta_uncut, verify_checksum=False)
    id_col = "SOURCE_ID" if "SOURCE_ID" in nss.colnames else "source_id"
    nss_rows = {}
    for row in nss:
        sid = int(row[id_col])
        if sid in want and str(row["nss_solution_type"]).strip('"') == "Orbital":
            nss_rows.setdefault(sid, row)
    uncut_rows = {}
    for row in uncut:
        sid = int(row["source_id"])
        if sid in want and str(row["nss_solution_type"]).strip('"') == "Orbital":
            uncut_rows.setdefault(sid, row)
    return {"side": side, "nss": nss_rows, "uncut": uncut_rows, "ids": sorted(want & set(nss_rows) & set(uncut_rows))}


def _cell(v: Any) -> Any:
    if np.ma.is_masked(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def test_real_rows_through_mock_path_reproduce_real_sidecar(cfg: Any, spec: Any, real_inputs: dict[str, Any]) -> None:
    from darkhunter_pop.andrews2022_atf import AtfSourceInputs  # noqa: F401  (documented real-row input type)
    from darkhunter_pop.mock_atf_columns import _ATMOSPHERE_BASES

    proc = require_procedure(spec)
    order = list(proc.covariance.parameter_order)
    rt = proc.covariance.float32_decimal_roundtrip
    ids = real_inputs["ids"]
    assert len(ids) >= 5, "need a handful of real rows with NSS + uncut rows"
    atm_cols = [c for c in real_inputs["uncut"][ids[0]].colnames if c.split("_upper")[0].split("_lower")[0] in _ATMOSPHERE_BASES]
    items = []
    for sid in ids:
        r, u = real_inputs["nss"][sid], real_inputs["uncut"][sid]
        means = np.array([notebook_float(_cell(r[n]), float32_decimal_roundtrip=False) for n in order])
        errs = np.array([notebook_float(_cell(r[f"{n}_error"]), float32_decimal_roundtrip=rt) for n in order])
        corr = np.array([notebook_float(v, float32_decimal_roundtrip=rt) for v in np.asarray(_cell(r["corr_vec"]), float).ravel()])
        try:
            cov = build_notebook_covariance(errs, corr)
        except ValueError:
            cov = np.full((12, 12), np.nan)
        cov_ok = bool(np.all(np.isfinite(cov)) and np.all(np.isfinite(means)))
        p0, c0 = _mock_from_atf(means, cov if cov_ok else np.eye(12))
        if not cov_ok:  # the real covariance failed its gate: keep the point estimate, NaN covariance
            c0 = np.full((12, 12), np.nan)
        extras = {c: _cell(u[c]) for c in atm_cols}
        mu, sg = pipeline_tag10_m1(extras, cfg)
        g, bp, rp = (float(_cell(u[c])) for c in ("g_mag", "bp_mag", "rp_mag"))
        item = mock_orbit_to_atf_inputs(
            sid, p0, c0, proc=proc, goodness_of_fit=notebook_float(_cell(r["goodness_of_fit"]), float32_decimal_roundtrip=False),
            g_mag=g, bp_rp=bp - rp, pipeline_m1_msun=mu, pipeline_m1_sigma_msun=sg,
        )
        items.append(item)
    got = compute_atf_columns(items, spec, mode=SampleSelectionMode.FORWARD_MODEL, workers=1)
    side = real_inputs["side"]
    # logg / FLAME are #680 / forward_model-unused: the mock path leaves them NaN by design.
    skip = {"andrews_atf_logg", "andrews_atf_mass_flame"}
    n_pass2 = 0
    for sid in ids:
        for col in ALL_COLUMNS:
            if col in skip:
                continue
            want, have = side[sid][col], got[sid][col]
            if isinstance(want, float) or isinstance(have, float) or want is None or have is None:
                if want is None or (isinstance(want, float) and math.isnan(want)):
                    assert have is None or (isinstance(have, float) and (math.isnan(have) or have == -1.0)), (sid, col, have)
                else:
                    assert have == pytest.approx(want, rel=1e-7, abs=1e-9), (sid, col, want, have)
            else:
                assert have == want, (sid, col, want, have)
        n_pass2 += side[sid]["andrews_atf_m1_source"] is not None
    assert n_pass2 >= 3, "test must exercise pass 2 (pipeline M1) on real rows"
