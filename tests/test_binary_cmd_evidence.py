"""Spec §12.22 (#491): CMD conditioning of the companion (composite prior points, LR weights)."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_pop import binary_cmd_evidence as bce
from darkhunter_pop import isochrone_mass as im


def _grid() -> im.NativeGrid:
    # one ([Fe/H], age), three MS points: mass 0.5, 1.0, 2.0 with M_G 8, 4.5, 1.5 and BP-RP 2, 0.8, 0.0
    v = np.zeros((1, 1, 3, len(im.NATIVE_QUANTITIES)), np.float32)
    k = {n: i for i, n in enumerate(im.NATIVE_QUANTITIES)}
    for e, (m, mg, c) in enumerate(((0.5, 8.0, 2.0), (1.0, 4.5, 0.8), (2.0, 1.5, 0.0))):
        v[0, 0, e, k["initial_mass"]] = m
        v[0, 0, e, k["star_mass"]] = m
        v[0, 0, e, k["mg"]] = mg
        v[0, 0, e, k["bp"]] = mg + 0.5 * c
        v[0, 0, e, k["rp"]] = mg - 0.5 * c
    return im.NativeGrid(feh=np.array([0.0]), log_age=np.array([9.0]), eep=np.arange(3), values=v)


@pytest.mark.unit
def test_composite_points_adds_flux() -> None:
    g = _grid()
    vals = {n: g.values[0, 0, :, i].astype(float) for i, n in enumerate(im.NATIVE_QUANTITIES)}
    pts = im.PriorPoints(weight=np.ones(3) / 3, colour=vals["bp"] - vals["rp"], mg=vals["mg"], feh=np.zeros(3),
                         log_age=np.full(3, 9.0), values=vals)
    c = bce.composite_points(pts, g, 1.0)  # twins: +0.753 mag brighter, same colour
    np.testing.assert_allclose(c.mg, vals["mg"] - 2.5 * np.log10(2.0), atol=1e-6)
    np.testing.assert_allclose(c.colour, pts.colour, atol=1e-6)
    lo = bce.composite_points(pts, g, 0.2)  # companion below the MS range for M <= 2.0 except 0.4 < 0.5: dark
    np.testing.assert_allclose(lo.mg[:2], vals["mg"][:2])
    np.testing.assert_array_equal(c.weight, pts.weight)


@pytest.mark.unit
def test_row_and_draw_factor_identity_and_direction() -> None:
    lam = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    llr0 = np.zeros((2, 3))
    norm, F = bce.row_factor(lam, llr0)
    np.testing.assert_allclose(norm, 1.0)
    np.testing.assert_allclose(F, 0.6)
    qg = [0.2, 0.5, 0.9]
    np.testing.assert_allclose(bce.draw_factor([0.3, 0.9], [0, 1], qg, llr0, norm), 1.0)
    llr = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 0.0]])  # row 0 favours twins; row 1 undefined
    norm, _ = bce.row_factor(lam, llr)
    f = bce.draw_factor([0.9, 0.2, 0.5], [0, 0, 1], qg, llr, norm)
    assert f[0] > 1.0 > f[1] and f[2] == 1.0
    # posterior companion probability sums to Σ Λ LR / (1 + Σ Λ LR)
    np.testing.assert_allclose(norm[0], 1.6 / (1 + 0.1 + 0.2 + 3.0))


@pytest.mark.unit
def test_theta_row_norm_identity_and_scaling() -> None:
    """§12.22 amendment: LR ≡ 1 gives 1 for any θ; larger intensity strengthens the pull."""
    rng = np.random.default_rng(0)
    R, P, K, J = 3, 4, 5, 6
    grids = {"lam0": rng.uniform(0.01, 0.05, (R, P, K)), "pqn": rng.uniform(0.1, 0.2, (R, P, J)),
             "qc": np.linspace(0.15, 0.975, K), "qn": np.linspace(0.35, 0.98, J)}
    one = np.zeros((R, K))
    np.testing.assert_allclose(bce.theta_row_norm(grids, one, np.zeros(R), np.zeros(P), 0.5, 0.3, 0.95), 1.0)
    llr = np.tile(np.array([0, 0, 0, 0.5, 1.0]), (R, 1))
    a = bce.theta_row_norm(grids, llr, np.zeros(R), np.zeros(P), 0.0, 0.0, 0.95)
    b = bce.theta_row_norm(grids, llr, np.full(R, 1.0), np.zeros(P), 0.0, 0.0, 0.95)
    assert np.all(a < 1.0) and np.all(b < a)
    llr[1, 2] = np.nan
    assert bce.theta_row_norm(grids, llr, np.zeros(R), np.zeros(P), 0.0, 0.0, 0.95)[1] == 1.0
