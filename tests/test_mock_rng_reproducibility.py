"""Mock realizations replay bit-for-bit from ``mock_population.random_seed`` (#371).

gaiamock_mod draws from numpy's global RNG (sky positions, transit loss, epoch
noise) and from libc ``rand()`` inside its compiled fitter. ``forward_model`` seeds
both per realization at the call boundary. The unit tests use a fake gaiamock that
draws from the same two global RNGs; the ``gaiamock`` test runs the real cascade.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config
from darkhunter_pop.config_schema import ExtinctionModel, PipelineConfig
from darkhunter_pop.forward_model import (
    MOCK_RNG_SEED_SCHEME,
    MOCK_RNG_STREAM_REALIZATION,
    MOCK_RNG_STREAM_SKY,
    MockRealizationRecord,
    _run_single_mock_realization,
    mock_global_rng_seeds,
    mock_rng_manifest_record,
    run_mock_injections_with_truth,
    seeded_global_rng,
)
from darkhunter_pop.gaiamock_vendor import is_overlay_ready

_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))


def _perturb_global_rngs(salt: int) -> None:
    """Advance numpy's global RNG and libc rand() by a salt-dependent amount."""
    np.random.seed(salt)
    np.random.uniform(size=17 + salt)
    _LIBC.srand(ctypes.c_uint(salt))
    for _ in range(31 + salt):
        _LIBC.rand()


class _GlobalRNGFakeGaiamock:
    """Fake gaiamock whose every draw comes from the two global RNGs gaiamock uses.

    The cascade outcome depends on numpy's global state *and* libc ``rand()``, so
    leaving either unseeded makes runs differ.
    """

    def read_in_C_functions(self) -> None:
        return None

    def generate_coordinates_at_a_given_distance_exponential_disk(
        self, *, d_min: float, d_max: float, N_stars: int, hz_pc: float
    ) -> tuple[np.ndarray, ...]:
        ra = np.random.uniform(0.0, 360.0, N_stars)
        dec = np.degrees(np.arcsin(np.random.uniform(-1.0, 1.0, N_stars)))
        d_pc = np.random.uniform(d_min, d_max, N_stars)
        xyz = np.random.normal(size=(3, N_stars))
        return ra, dec, d_pc, xyz[0], xyz[1], xyz[2]

    def xyz_to_galactic(
        self, *, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(x), np.zeros_like(x)

    def run_full_astrometric_cascade(self, **kwargs: Any) -> list[float]:
        noise = float(np.random.randn())
        c_draw = int(_LIBC.rand())
        if (c_draw % 3) == 0:
            return [0.0] * 23  # insufficient visibility
        res = [-1.0] * 23  # five-parameter outcome; ruwe/plx slots carry the draws
        res[1] = 1.0 + abs(noise)
        res[2] = float(c_draw % 1000)
        return res


def _small_config(n: int, seed: int) -> PipelineConfig:
    cfg = load_config().model_copy(deep=True)
    pop = cfg.selection_function_astrometric.mock_population
    pop.N_realizations = n
    pop.random_seed = seed
    cfg.selection_function_astrometric.extinction_model = ExtinctionModel.NONE
    return cfg


def _signature(records: list[MockRealizationRecord]) -> list[tuple[Any, ...]]:
    return [
        (
            r.solution_type.value,
            r.accepted_orbital,
            r.P_orb_days,
            r.eccentricity,
            r.f_m_msun,
            r.parallax_mas,
            r.a0_mas,
            r.cos_inclination,
            None if r.multi_solution is None else repr(r.multi_solution),
        )
        for r in records
    ]


@pytest.mark.unit
def test_mock_global_rng_seeds_deterministic_and_distinct() -> None:
    a = mock_global_rng_seeds(42, MOCK_RNG_STREAM_REALIZATION, 3)
    assert a == mock_global_rng_seeds(42, MOCK_RNG_STREAM_REALIZATION, 3)
    others = {
        mock_global_rng_seeds(43, MOCK_RNG_STREAM_REALIZATION, 3),
        mock_global_rng_seeds(42, MOCK_RNG_STREAM_REALIZATION, 4),
        mock_global_rng_seeds(42, MOCK_RNG_STREAM_SKY, 3),
    }
    assert a not in others and len(others) == 3
    for s in (a, *others):
        assert 0 <= s.numpy_seed < 2**32 and 0 <= s.c_rand_seed < 2**32
    with pytest.raises(ValueError):
        mock_global_rng_seeds(-1, MOCK_RNG_STREAM_SKY, 0)


@pytest.mark.unit
def test_seeded_global_rng_restores_numpy_and_seeds_libc() -> None:
    seeds = mock_global_rng_seeds(7, MOCK_RNG_STREAM_REALIZATION, 0)
    np.random.seed(2024)
    expected_after = np.random.get_state()[1].copy()
    with seeded_global_rng(seeds):
        first = (float(np.random.uniform()), int(_LIBC.rand()))
    # numpy's global state outside the block is untouched.
    assert np.array_equal(np.random.get_state()[1], expected_after)
    _perturb_global_rngs(5)
    with seeded_global_rng(seeds):
        second = (float(np.random.uniform()), int(_LIBC.rand()))
    assert first == second


@pytest.mark.unit
def test_mock_injections_replay_identically_with_same_seed() -> None:
    """Same seed → identical records and truth, whatever the global RNGs held before."""
    cfg = _small_config(12, seed=42)
    fake = _GlobalRNGFakeGaiamock()

    _perturb_global_rngs(1)
    rec_a, g_a, truth_a = run_mock_injections_with_truth(cfg, fake)  # type: ignore[arg-type]
    _perturb_global_rngs(2)
    rec_b, g_b, truth_b = run_mock_injections_with_truth(cfg, fake)  # type: ignore[arg-type]

    assert _signature(rec_a) == _signature(rec_b)
    assert np.array_equal(g_a, g_b)
    assert truth_a.keys() == truth_b.keys()
    for key in truth_a:
        assert np.array_equal(truth_a[key], truth_b[key]), key
    # realization_index is the original draw index (G-limit removal may drop draws).
    idx = truth_a["realization_index"].tolist()
    assert idx == sorted(set(idx)) and set(idx) <= set(range(12))
    expected = [
        mock_global_rng_seeds(42, MOCK_RNG_STREAM_REALIZATION, i) for i in idx
    ]
    assert truth_a["rng_seed_numpy"].tolist() == [s.numpy_seed for s in expected]
    assert truth_a["rng_seed_c_rand"].tolist() == [s.c_rand_seed for s in expected]

    rec_c, _g_c, truth_c = run_mock_injections_with_truth(  # type: ignore[arg-type]
        _small_config(12, seed=43), fake
    )
    assert _signature(rec_c) != _signature(rec_a)
    assert not np.array_equal(truth_c["ra_deg"], truth_a["ra_deg"])


@pytest.mark.unit
def test_mock_realization_independent_of_order() -> None:
    """A realization depends only on (seed, index): reversed order gives the same records."""
    cfg = _small_config(8, seed=11)
    fake = _GlobalRNGFakeGaiamock()
    records, g_mag, truth = run_mock_injections_with_truth(cfg, fake)  # type: ignore[arg-type]

    n = len(records)
    reversed_records: dict[int, MockRealizationRecord] = {}
    for i in reversed(range(n)):
        index = int(truth["realization_index"][i])
        with seeded_global_rng(
            mock_global_rng_seeds(11, MOCK_RNG_STREAM_REALIZATION, index)
        ):
            reversed_records[i] = _run_single_mock_realization(
                fake,  # type: ignore[arg-type]
                ra=float(truth["ra_deg"][i]),
                dec=float(truth["dec_deg"][i]),
                d_pc=float(truth["distance_pc"][i]),
                phot_g_mean_mag=float(g_mag[i]),
                config=cfg,
                c_funcs=None,
                draw=_draw_from_truth(truth, i),
            )
    assert _signature([reversed_records[i] for i in range(n)]) == _signature(records)


def _draw_from_truth(truth: dict[str, np.ndarray], i: int) -> Any:
    from darkhunter_pop.forward_model import MockBinaryDraw

    return MockBinaryDraw(
        period_days=float(truth["period_days"][i]),
        eccentricity=float(truth["eccentricity"][i]),
        m1_msun=float(truth["m1_msun"][i]),
        m2_msun=float(truth["m2_msun"][i]),
        flux_ratio=float(truth["flux_ratio"][i]),
        Mg_tot=float(truth["Mg_tot"][i]),
        Tp=float(truth["Tp_days"][i]),
        omega_rad=float(truth["Omega_rad"][i]),
        w_rad=float(truth["omega_rad"][i]),
        inc_deg=float(truth["inc_deg"][i]),
        faint_draw=bool(truth["faint_draw"][i]),
    )


@pytest.mark.api
def test_mock_rng_provenance_in_artifact_and_manifest_record(tmp_path: Path) -> None:
    from darkhunter_pop.forward_model import (
        SelectionFunctionAstrometricResult,
        run_validation_gate,
        write_selection_function_artifact,
    )
    from darkhunter_pop.gaiamock_vendor import GaiamockModVersions

    cfg = _small_config(6, seed=42)
    records, g_mag, truth = run_mock_injections_with_truth(  # type: ignore[arg-type]
        cfg, _GlobalRNGFakeGaiamock()
    )
    validation = run_validation_gate(
        cfg,
        records,
        real_panels={},
        real_solution_fractions={"five_parameter": 1.0},
        g_mag=g_mag,
    )
    result = SelectionFunctionAstrometricResult(
        gaiamock_versions=GaiamockModVersions(
            gaiamock_mod_release="test",
            gaiamock_mod_sha256="0" * 64,
            gaiamock_git_commit="0" * 40,
        ),
        records=records,
        validation=validation,
        data_release="dr3",
        injected_truth=truth,
        mock_random_seed=42,
    )
    path = tmp_path / "sfa.h5"
    write_selection_function_artifact(path, result, g_mag=g_mag)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["mock_random_seed"]) == 42
        assert handle.attrs["mock_rng_seed_scheme"] == MOCK_RNG_SEED_SCHEME
        tg = handle["mock_catalog/truth"]
        assert tg.attrs["rng_seed_scheme"] == MOCK_RNG_SEED_SCHEME
        assert np.issubdtype(tg["rng_seed_numpy"].dtype, np.integer)
        assert np.array_equal(tg["rng_seed_numpy"][()], truth["rng_seed_numpy"])
        assert np.array_equal(tg["rng_seed_c_rand"][()], truth["rng_seed_c_rand"])

    record = mock_rng_manifest_record(cfg)
    assert record["random_seed"] == 42
    assert record["n_realizations"] == 6
    assert record["scheme"] == MOCK_RNG_SEED_SCHEME


@pytest.mark.gaiamock
def test_real_gaiamock_cascade_replays_bit_for_bit() -> None:
    """Real gaiamock_mod: same seed → identical cascade outputs, even after the
    global RNGs are perturbed; reversed order matches; a new seed differs."""
    if not is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    gaiamock = import_gaiamock_mod()
    n = 6
    cfg = _small_config(n, seed=42)

    _perturb_global_rngs(3)
    rec_a, g_a, truth_a = run_mock_injections_with_truth(cfg, gaiamock)
    _perturb_global_rngs(4)
    rec_b, _g_b, truth_b = run_mock_injections_with_truth(cfg, gaiamock)
    assert _signature(rec_a) == _signature(rec_b)
    assert sum(r.accepted_orbital for r in rec_a) == sum(r.accepted_orbital for r in rec_b)
    for key in truth_a:
        assert np.array_equal(truth_a[key], truth_b[key]), key

    # Raw cascade vectors, in reversed order, under the per-realization seeds.
    c_funcs = gaiamock.read_in_C_functions()
    pop = cfg.selection_function_astrometric.mock_population

    def raw(i: int) -> list[float]:
        index = int(truth_a["realization_index"][i])
        with seeded_global_rng(
            mock_global_rng_seeds(42, MOCK_RNG_STREAM_REALIZATION, index), c_funcs
        ):
            return list(
                gaiamock.run_full_astrometric_cascade(
                    ra=float(truth_a["ra_deg"][i]),
                    dec=float(truth_a["dec_deg"][i]),
                    parallax=1000.0 / float(truth_a["distance_pc"][i]),
                    pmra=0.0,
                    pmdec=0.0,
                    m1=float(truth_a["m1_msun"][i]),
                    m2=float(truth_a["m2_msun"][i]),
                    period=float(truth_a["period_days"][i]),
                    Tp=float(truth_a["Tp_days"][i]),
                    ecc=float(truth_a["eccentricity"][i]),
                    omega=float(truth_a["Omega_rad"][i]),
                    inc_deg=float(truth_a["inc_deg"][i]),
                    w=float(truth_a["omega_rad"][i]),
                    phot_g_mean_mag=float(g_a[i]),
                    f=float(truth_a["flux_ratio"][i]),
                    data_release=cfg.active_dr_mode.value,
                    c_funcs=c_funcs,
                    ruwe_min=pop.ruwe_min,
                    skip_acceleration=pop.skip_acceleration,
                )
            )

    n_sim = len(rec_a)
    forward = [raw(i) for i in range(n_sim)]
    backward = {i: raw(i) for i in reversed(range(n_sim))}
    assert forward == [backward[i] for i in range(n_sim)]

    rec_c, _g_c, _truth_c = run_mock_injections_with_truth(_small_config(n, seed=43), gaiamock)
    assert _signature(rec_c) != _signature(rec_a)
