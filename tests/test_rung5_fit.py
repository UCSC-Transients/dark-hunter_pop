"""R5-T11 (#644): rung-5 fit core at fixture scale (synthetic templates; no data/, no gaiamock)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop import rung5_fit as rf

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("rung5_build_inputs", ROOT / "scripts" / "rung5_build_inputs.py")
bld = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bld)  # type: ignore[union-attr]


def _inputs(scale: float = 1.0) -> rf.Rung5Inputs:
    """20 cells with distinct (seeded) template shapes: 1 luminous group, 2 dark bins, 2 WD bins, 3 WD branches."""
    B = 20
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 1.0, B)
    lum = scale * np.array([400.0 * np.exp(-6 * x) + 1.0])
    spur = rng.uniform(0.5, 1.5, B)
    spur /= spur.sum()
    wd = np.zeros((3, 2, B))
    wd[0, 0] = 300 * np.exp(-((x - 0.2) / 0.1) ** 2)
    wd[1, 0] = 300 * np.exp(-((x - 0.1) / 0.08) ** 2)
    wd[2, 1] = 300 * np.exp(-((x - 0.4) / 0.1) ** 2)
    wd[0, 1] = 50 * np.exp(-((x - 0.5) / 0.1) ** 2)
    dark = np.array([300 * np.exp(-((x - 0.65) / 0.1) ** 2), 300 * np.exp(-((x - 0.9) / 0.1) ** 2)]) * 1e3
    return rf.Rung5Inputs(
        k_obs=np.zeros(B), high_a=x > 0.55,
        lum=lum, lum_var=lum * 1e-4, spur=spur, spur_var=spur**2 * 1e-4, wd=wd * 1e3, wd_var=(wd * 1e3) ** 2 * 1e-6,
        wd_rate=np.array([[1.0, 0.2], [0.5, 0.1], [0.1, 1.0]]), dark=dark, dark_var=dark**2 * 1e-6,
        wd_edges=np.array([0.15, 0.6, 1.4]), dark_edges=np.array([1.0, 3.0, 40.0]), pihat_sum=100.0, ln_r_center=0.0,
        meta={"fixture": True})


def _theta(inp: rf.Rung5Inputs) -> np.ndarray:
    s = rf._slices(inp)
    t = rf.start_point(inp, rf.Priors())
    t[s["dark"]] = [-4.0, -4.5]
    t[s["wd"]] = [-2.0, -2.5]
    t[s["lum"]] = [0.1]
    t[s["ln_ns"]] = math.log(120.0)
    t[s["ln_sa"]] = math.log(2.0)
    t[s["ln_r"]] = 0.3
    t[s["f_he"]] = 0.4
    return t


@pytest.mark.unit
def test_branch_fractions_normalised() -> None:
    pi = rf.wd_branch_fractions(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), 0.0, 0.5)
    np.testing.assert_allclose(pi.sum(axis=0), 1.0)
    np.testing.assert_allclose(pi[:, 1], 1 / 3)
    np.testing.assert_allclose(pi[:, 0], [0.5, 0.25, 0.25])


@pytest.mark.unit
def test_expected_counts_components_add() -> None:
    inp = _inputs()
    t = _theta(inp)
    s = rf._slices(inp)
    mu, var = rf.expected_counts(t, inp)
    ns = 120.0 * np.where(inp.high_a, 2.0, 1.0)
    pi = rf.wd_branch_fractions(inp.wd_rate, 0.3, 0.4)
    ref = (math.exp(0.1) * inp.lum[0] + ns * inp.spur + np.einsum("j,kj,kjb->b", 10.0 ** t[s["wd"]], pi, inp.wd)
           + 10.0 ** t[s["dark"]] @ inp.dark)
    np.testing.assert_allclose(mu, ref)
    assert np.all(var > 0)


@pytest.mark.unit
def test_say_reduces_to_poisson() -> None:
    k, mu = np.array([3.0, 0.0, 10.0]), np.array([2.5, 0.4, 12.0])
    from scipy.special import gammaln

    assert rf.say_loglike(k, mu, np.zeros(3)) == pytest.approx(float(np.sum(k * np.log(mu) - mu - gammaln(k + 1))))
    assert rf.say_loglike(k, mu, mu * 0.5) < rf.say_loglike(k, mu, np.zeros(3)) + 5


@pytest.mark.unit
def test_map_recovers_dark_heights_noiseless() -> None:
    inp = _inputs(scale=50.0)
    t0 = _theta(inp)
    mu, _ = rf.expected_counts(t0, inp)
    inp.k_obs = mu.copy()
    th, lp = rf.map_fit(inp, rf.Priors(), x0=t0 + 0.05, n_starts=1)
    s = rf._slices(inp)
    assert lp >= rf.log_posterior(t0, inp, rf.Priors()) - 1e-3
    sig = rf.laplace_sigma(inp, rf.Priors(), th)
    assert np.all(np.abs(th[s["dark"]] - t0[s["dark"]]) < 2 * sig[s["dark"]])


@pytest.mark.unit
def test_prior_bounds_and_names() -> None:
    inp = _inputs()
    names = rf.parameter_names(inp)
    bb = rf.bounds(inp, rf.Priors())
    assert len(names) == bb.shape[0] == 2 + 2 + 1 + 4
    i = names.index("ln_ns")
    assert bb[i, 0] == pytest.approx(math.log(25.0)) and bb[i, 1] == pytest.approx(math.log(400.0))
    t = _theta(inp)
    t[i] = bb[i, 1] + 1
    assert rf.log_prior(t, inp, rf.Priors()) == -math.inf


@pytest.mark.unit
def test_simulate_and_closed_loop_pulls_reasonable() -> None:
    inp = _inputs(scale=50.0)
    t0 = _theta(inp)
    rng = np.random.default_rng(3)
    inp.k_obs = rf.simulate_counts(t0, inp, rng)
    th, _ = rf.map_fit(inp, rf.Priors(), x0=t0, n_starts=1)
    sig = rf.laplace_sigma(inp, rf.Priors(), th)
    s = rf._slices(inp)
    pull = (th[s["dark"]] - t0[s["dark"]]) / sig[s["dark"]]
    assert np.all(np.isfinite(pull)) and np.all(np.abs(pull) < 5)


@pytest.mark.unit
def test_diagnostic_outputs_shapes() -> None:
    inp = _inputs(scale=50.0)
    t0 = _theta(inp)
    inp.k_obs, _ = rf.expected_counts(t0, inp)
    names = rf.parameter_names(inp)
    rng = np.random.default_rng(1)
    samples = t0 + rng.normal(0, 0.02, (400, t0.size))
    cm = rf.correlation_matrix(samples, names, [n for n in names if "dark" in n] + list(rf.NUISANCES))
    assert np.asarray(cm["matrix"]).shape == (6, 6)
    tab = rf.nuisance_shift_table(inp, rf.Priors(), t0, samples, names)
    assert set(tab) == set(rf.NUISANCES) and len(tab["ln_ns"]["plus"]["log10_h_dark"]) == 2
    ov = rf.small_n_overlap(inp, rf.Priors(), samples, t0, expected_min=1e12, threshold=0.5, collapse_floor=1e-9)
    assert {r["component"] for r in ov} == {"dark", "wd"} and all("overlap" in r for r in ov)


@pytest.mark.unit
def test_inputs_roundtrip(tmp_path: Path) -> None:
    inp = _inputs()
    rf.write_inputs(tmp_path / "in.h5", inp)
    back = rf.read_inputs(tmp_path / "in.h5")
    np.testing.assert_array_equal(back.wd, inp.wd)
    assert back.meta == {"fixture": True} and back.pihat_sum == 100.0


@pytest.mark.unit
def test_point_inversion_bh1_like() -> None:
    m1, m2, p, plx = 0.93, 9.6, 185.6, 2.08
    a_au = ((p / 365.25) ** 2 * (m1 + m2)) ** (1 / 3) * m2 / (m1 + m2)
    out = bld.m2_point_inversion(np.array([a_au * plx]), np.array([plx]), np.array([p]), np.array([m1]))
    assert out[0] == pytest.approx(m2, rel=1e-6)
    assert np.isnan(bld.m2_point_inversion(np.array([1.0]), np.array([-1.0]), np.array([100.0]), np.array([1.0]))[0])


@pytest.mark.unit
def test_cell_index_and_high_a() -> None:
    m2e = np.array([0.0, 1.0, 1000.0])
    c = bld.cell_index(np.array([0.5, 2.0, np.nan]), np.array([0.1, 0.9, 0.1]), np.array([0.2, 2.0, 0.2]), m2e)
    assert c.tolist() == [0, (1 * 2 + 1) * 4 + 3, -1]
    assert bld.n_cells(m2e) == 16
    ha = bld.high_a_mask(m2e, 1.0)
    assert ha.sum() == 4 + 8  # A > 0.5 cells of the first M2 bin, plus every cell of the second


@pytest.mark.unit
def test_binned_response_normalisation() -> None:
    cell = np.array([0, 0, 1, -1])
    jb = np.array([0, 0, 0, 0])
    w = np.ones(4)
    R, V, I = bld.binned_response(cell, jb, 1, w, np.ones(4), 3, n_prim=100.0)
    # 4 draws in bin j (total weight 4 = rate fraction 0.04 per primary); 2 + 1 detected in cells 0, 1
    np.testing.assert_allclose(R[0], [50.0, 25.0, 0.0])
    assert I[0] == pytest.approx(0.04)
    np.testing.assert_allclose(V[0], [100.0**2 * 2 / 16, 100.0**2 / 16, 0.0])


@pytest.mark.slow
def test_dynesty_tiny_run() -> None:
    inp = _inputs(scale=50.0)
    t0 = _theta(inp)
    inp.k_obs, _ = rf.expected_counts(t0, inp)
    res = rf.run_dynesty(inp, rf.Priors(), nlive=30, seed=1, dlogz=5.0, maxcall=3000)
    assert res["samples"].shape[1] == t0.size and np.isfinite(res["logz"])
