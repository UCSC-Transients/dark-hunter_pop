"""El-Badry 2026 frozen extinction policy: readers, split, cache, dereddening (#258)."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.config_schema import (
    DustMapFileSpec,
    DustMapsConfig,
    ExtinctionHemisphereSpec,
    ExtinctionSpec,
)
from darkhunter_pop.dust_maps import (
    HEMISPHERE_NONE,
    HEMISPHERE_NORTH,
    HEMISPHERE_SOUTH,
    Bayestar2019Map,
    DustMapError,
    ExtinctionLookup,
    ExtinctionStatus,
    Lallement2019Map,
    declination_mask,
    distance_kpc_from_parallax,
    hemisphere_codes,
    read_extinction_cache,
)
from darkhunter_pop.elbadry2026_selection import (
    EBV_COLUMN,
    EXTINCTION_MAP_COLUMN,
    EXTINCTION_STATUS_COLUMN,
    deredden_elbadry2026_rows,
    extinction_coefficients,
)
from darkhunter_pop.sample_selection import (
    NotApplicable,
    SampleSelectionRegistry,
    load_sample_selection_file,
)

ELBADRY2026 = repo_root() / "config" / "selections" / "elbadry2026.yaml"


def _spec26():
    return load_sample_selection_file(ELBADRY2026)


def _split(value: float = -28.0) -> ExtinctionSpec:
    return ExtinctionSpec(
        north=ExtinctionHemisphereSpec(applies_when=f"dec_deg > {value}", map="green2019"),
        south=ExtinctionHemisphereSpec(applies_when=f"dec_deg <= {value}", map="lallement2019"),
        coefficients={"a_g_over_e_bv": 2.66, "e_bp_rp_over_e_bv": 1.33},
    )


def _dust_cfg(north_factor: float = 1.0, south_factor: float = 1.0) -> DustMapsConfig:
    return DustMapsConfig(
        ebv_cache_dir="cache",
        maps={
            "green2019": DustMapFileSpec(path="g.h5", native_to_ebv=north_factor),
            "lallement2019": DustMapFileSpec(path="l.h5", native_to_ebv=south_factor),
        },
    )


class _ConstReader:
    """Fake map: native value = ``value × d_kpc`` everywhere; counts calls."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0
        self.n_queried = 0

    def query_native(self, l_deg, b_deg, d_kpc):
        self.calls += 1
        self.n_queried += len(d_kpc)
        d = np.asarray(d_kpc, dtype=np.float64)
        return self.value * d, np.zeros(d.shape, dtype=np.int8)


def _lookup(tmp_path: Path, *, north=0.1, south=0.3, use_cache=True, factors=(1.0, 1.0)):
    readers = {"green2019": _ConstReader(north), "lallement2019": _ConstReader(south)}
    lookup = ExtinctionLookup(
        _split(),
        _dust_cfg(*factors),
        data_root=tmp_path,
        cache_tag="elbadry2026",
        readers=readers,
        use_cache=use_cache,
    )
    return lookup, readers


# --------------------------------------------------------------------------
# Split, distance
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_frozen_split_is_dec_minus_28_green_north_lallement_south() -> None:
    spec = _spec26()
    assert spec.extinction is not None
    assert spec.extinction.north.map == "green2019"
    assert spec.extinction.south.map == "lallement2019"
    assert extinction_coefficients(spec) == (2.66, 1.33)
    codes = hemisphere_codes(spec.extinction, np.array([-27.999, -28.0, -28.001, 45.0, -80.0]))
    assert codes.tolist() == [
        HEMISPHERE_NORTH,
        HEMISPHERE_SOUTH,
        HEMISPHERE_SOUTH,
        HEMISPHERE_NORTH,
        HEMISPHERE_SOUTH,
    ]


@pytest.mark.unit
def test_hemisphere_nan_dec_is_none_and_bad_splits_refuse() -> None:
    assert hemisphere_codes(_split(), np.array([np.nan]))[0] == HEMISPHERE_NONE
    overlap = ExtinctionSpec(
        north=ExtinctionHemisphereSpec(applies_when="dec_deg >= -28.0", map="green2019"),
        south=ExtinctionHemisphereSpec(applies_when="dec_deg <= -28.0", map="lallement2019"),
    )
    with pytest.raises(ValueError, match="overlap"):
        hemisphere_codes(overlap, np.array([-28.0]))
    gap = ExtinctionSpec(
        north=ExtinctionHemisphereSpec(applies_when="dec_deg > -28.0", map="green2019"),
        south=ExtinctionHemisphereSpec(applies_when="dec_deg < -28.0", map="lallement2019"),
    )
    with pytest.raises(ValueError, match="gap"):
        hemisphere_codes(gap, np.array([-28.0]))
    with pytest.raises(ValueError, match="unsupported"):
        declination_mask("ra_deg > 3", np.array([0.0]))


@pytest.mark.unit
def test_distance_from_parallax_flags_invalid() -> None:
    d, ok = distance_kpc_from_parallax(np.array([2.0, 0.0, -1.0, np.nan]))
    assert ok.tolist() == [True, False, False, False]
    assert d[0] == pytest.approx(0.5)
    assert np.all(np.isnan(d[1:]))


# --------------------------------------------------------------------------
# Dereddening arithmetic and lookup routing
# --------------------------------------------------------------------------


@pytest.mark.physics
def test_dereddening_arithmetic_uses_frozen_coefficients() -> None:
    spec = _spec26()
    rows = [{"source_id": 1, "abs_g_mag": 5.0, "bp_rp": 1.0, EBV_COLUMN: 0.2}]
    out = deredden_elbadry2026_rows(rows, spec, lookup=None)[0]
    assert out["mg_0"] == pytest.approx(5.0 - 2.66 * 0.2)
    assert out["bp_rp_0"] == pytest.approx(1.0 - 1.33 * 0.2)
    assert out[EXTINCTION_STATUS_COLUMN] == "precomputed"
    # Raw photometry is preserved alongside.
    assert out["abs_g_mag"] == 5.0 and out["bp_rp"] == 1.0


@pytest.mark.unit
def test_missing_lookup_refuses_undereddened_evaluation() -> None:
    with pytest.raises(DustMapError, match="undereddened"):
        deredden_elbadry2026_rows(
            [{"source_id": 1, "abs_g_mag": 5.0, "bp_rp": 1.0, "ra_deg": 1.0, "dec_deg": 0.0,
              "parallax_mas": 1.0}],
            _spec26(),
            lookup=None,
        )


@pytest.mark.physics
def test_lookup_routes_by_hemisphere_and_converts_units(tmp_path: Path) -> None:
    lookup, readers = _lookup(tmp_path, north=0.1, south=0.3, factors=(0.5, 2.0))
    rows = [
        {"source_id": 1, "abs_g_mag": 5.0, "bp_rp": 1.0, "ra_deg": 10.0, "dec_deg": -27.9,
         "parallax_mas": 1.0},
        {"source_id": 2, "abs_g_mag": 5.0, "bp_rp": 1.0, "ra_deg": 10.0, "dec_deg": -28.0,
         "parallax_mas": 2.0},
    ]
    north, south = deredden_elbadry2026_rows(rows, _spec26(), lookup)
    # d = 1 kpc, native 0.1, factor 0.5 → E = 0.05; d = 0.5 kpc, native 0.15, ×2 → 0.3.
    assert north[EBV_COLUMN] == pytest.approx(0.05)
    assert north[EXTINCTION_MAP_COLUMN] == "green2019"
    assert south[EBV_COLUMN] == pytest.approx(0.3)
    assert south[EXTINCTION_MAP_COLUMN] == "lallement2019"
    assert south["mg_0"] == pytest.approx(5.0 - 2.66 * 0.3)
    assert readers["green2019"].n_queried == 1
    assert readers["lallement2019"].n_queried == 1


@pytest.mark.unit
def test_invalid_parallax_is_not_applicable_with_reason(tmp_path: Path) -> None:
    lookup, readers = _lookup(tmp_path)
    rows = [
        {"source_id": 5, "bp_rp": 1.0, "ra_deg": 10.0, "dec_deg": 5.0, "parallax_mas": -0.3},
        {"source_id": 6, "bp_rp": 1.0, "ra_deg": 10.0, "dec_deg": 5.0, "parallax_mas": None},
    ]
    out = deredden_elbadry2026_rows(rows, _spec26(), lookup)
    for row in out:
        assert isinstance(row["mg_0"], NotApplicable)
        assert row["mg_0"].reason == "extinction_invalid_parallax"
        assert row[EXTINCTION_STATUS_COLUMN] == "invalid_parallax"
    assert readers["green2019"].n_queried == 0


@pytest.mark.unit
def test_cache_round_trip_avoids_requery(tmp_path: Path) -> None:
    rows = [
        {"source_id": i, "abs_g_mag": 5.0, "bp_rp": 1.0, "ra_deg": 10.0 + i,
         "dec_deg": 10.0, "parallax_mas": 1.0}
        for i in range(3)
    ]
    lookup, readers = _lookup(tmp_path)
    deredden_elbadry2026_rows(rows, _spec26(), lookup)
    assert readers["green2019"].n_queried == 3
    assert lookup.cache_path.is_file()
    assert len(read_extinction_cache(lookup.cache_path)) == 3
    fresh, fresh_readers = _lookup(tmp_path)
    assert fresh.cache_path == lookup.cache_path
    deredden_elbadry2026_rows(rows, _spec26(), fresh)
    assert fresh_readers["green2019"].calls == 0
    # A changed parallax is a different key → re-queried, appended.
    moved = [dict(rows[0], parallax_mas=0.5)]
    deredden_elbadry2026_rows(moved, _spec26(), fresh)
    assert fresh_readers["green2019"].n_queried == 1
    assert len(read_extinction_cache(fresh.cache_path)) == 4


@pytest.mark.unit
def test_cache_is_independent_of_unit_conversion(tmp_path: Path) -> None:
    a, _ = _lookup(tmp_path, factors=(1.0, 1.0))
    b, _ = _lookup(tmp_path, factors=(0.884, 1.0 / 3.1))
    assert a.cache_path == b.cache_path


@pytest.mark.unit
def test_missing_map_file_is_hard_error(tmp_path: Path) -> None:
    lookup = ExtinctionLookup(
        _split(), _dust_cfg(), data_root=tmp_path, cache_tag="x", use_cache=False
    )
    with pytest.raises(DustMapError, match="not found"):
        lookup.ebv_for([1], [10.0], [10.0], [1.0])


# --------------------------------------------------------------------------
# Registry wiring: El-Badry 2026 only
# --------------------------------------------------------------------------


@pytest.mark.api
def test_only_elbadry2026_gets_an_extinction_lookup() -> None:
    registry = SampleSelectionRegistry(load_config())
    assert registry.selection("elbadry2026").extinction_lookup is not None
    for name in ("andrews2022", "andrews2022_modified", "elbadry2024"):
        assert registry.selection(name).extinction_lookup is None


@pytest.mark.api
def test_elbadry2024_and_andrews_rows_are_not_dereddened() -> None:
    registry = SampleSelectionRegistry(load_config())
    raw = {
        "source_id": 11,
        "nss_solution_type": "Orbital",
        "abs_g_mag": 3.0,
        "bp_rp": 0.95,
        "mg_0": 3.0,
        "bp_rp_0": 0.95,
        "ra_deg": 10.0,
        "dec_deg": 10.0,
        "parallax_mas": 1.0,
    }
    for name in ("andrews2022", "andrews2022_modified", "elbadry2024"):
        out = registry.selection(name)._enrich_rows_for_spec([dict(raw)])
        assert out[0]["mg_0"] == 3.0
        assert out[0]["bp_rp_0"] == 0.95
        assert EBV_COLUMN not in out[0]


@pytest.mark.api
def test_elbadry2026_main_sequence_uses_dereddened_cmd(tmp_path: Path) -> None:
    """Raw (3.0, 0.95) is off the MS; E(B-V)=0.1 moves it onto the CMD branch."""
    registry = SampleSelectionRegistry(load_config())
    selection = registry.selection("elbadry2026")
    lookup, _ = _lookup(tmp_path, north=0.1)
    selection.extinction_lookup = lookup
    raw = {
        "source_id": 12,
        "nss_solution_type": "Orbital",
        "abs_g_mag": 3.0,
        "bp_rp": 0.95,
        "mg_0": 3.0,  # the legacy undereddened alias from candidate_to_selection_row
        "bp_rp_0": 0.95,
        "ra_deg": 10.0,
        "dec_deg": 10.0,
        "parallax_mas": 1.0,
    }
    (out,) = selection._enrich_rows_for_spec([raw])
    assert out["mg_0"] == pytest.approx(3.0 - 0.266)
    assert out["bp_rp_0"] == pytest.approx(0.95 - 0.133)
    assert out["main_sequence"] is True
    assert selection.extinction_status_counts == {"ok": 1}


@pytest.mark.api
def test_unusable_extinction_propagates_not_applicable_to_mass(tmp_path: Path) -> None:
    registry = SampleSelectionRegistry(load_config())
    selection = registry.selection("elbadry2026")
    selection.extinction_lookup, _ = _lookup(tmp_path)
    raw = {
        "source_id": 13,
        "nss_solution_type": "SB1",
        "bp_rp": 0.95,
        "ra_deg": 10.0,
        "dec_deg": 10.0,
        "parallax_mas": 0.0,
        "period_day": 100.0,
        "k1_kms": 30.0,
        "k1_error_kms": 1.0,
    }
    (out,) = selection._enrich_rows_for_spec([raw])
    for key in ("main_sequence", "m1_tilde_msun", "m2_min_msun"):
        assert isinstance(out[key], NotApplicable)
        assert out[key].reason == "extinction_invalid_parallax"


# --------------------------------------------------------------------------
# Readers on synthetic files
# --------------------------------------------------------------------------


def _write_bayestar(path: Path, profile: np.ndarray) -> None:
    """All 12 nside=1 pixels share ``profile`` (120 nodes)."""
    info = np.zeros(
        12,
        dtype=[("nside", "<u4"), ("healpix_index", "<u8"), ("converged", "u1"),
               ("DM_reliable_min", "<f4"), ("DM_reliable_max", "<f4"),
               ("n_stars", "<u4"), ("n_good", "<u4"), ("n_dwarfs", "<u4")],
    )
    info["nside"] = 1
    info["healpix_index"] = np.arange(12)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixel_info", data=info)
        handle.create_dataset("best_fit", data=np.tile(profile, (12, 1)).astype(np.float32))


@pytest.mark.physics
def test_bayestar_reader_linear_in_distance_modulus(tmp_path: Path) -> None:
    pytest.importorskip("mwdust")
    grid = np.linspace(4.0, 18.875, 120)
    profile = 0.01 * (grid - 4.0) + 0.02  # E = 0.02 at DM=4
    path = tmp_path / "bayestar.h5"
    _write_bayestar(path, profile)
    reader = Bayestar2019Map(path)
    d = np.array([10.0 ** (10.0 / 5.0 - 2.0), 10.0 ** (4.0 / 5.0 - 2.0) / 2.0, 1.0e3])
    vals, status = reader.query_native(np.array([30.0, 30.0, 30.0]), np.array([10.0, 10.0, 10.0]), d)
    assert vals[0] == pytest.approx(0.01 * 6.0 + 0.02, rel=1e-6)  # DM = 10 inside grid
    assert vals[1] == pytest.approx(0.02 / 2.0, rel=1e-6)  # half the first-node distance
    assert status[1] == ExtinctionStatus.OK
    assert vals[2] == pytest.approx(profile[-1], rel=1e-6)  # held beyond the grid
    assert status[2] == ExtinctionStatus.BEYOND_MAP_LIMIT


def _write_lallement(path: Path, density: float, half_width_pc: float, step_pc: float) -> None:
    """Synthetic cube in the published ``stilism/cube_datas`` layout."""
    n = int(round(2 * half_width_pc / step_pc)) + 1
    cube = np.full((n, n, n), density, dtype=np.float32)
    centre = (n - 1) / 2.0 + 0.5  # voxel-index units, voxel k spans [k, k+1)
    with h5py.File(path, "w") as handle:
        dset = handle.create_dataset("stilism/cube_datas", data=cube)
        dset.attrs["gridstep_values"] = np.array([step_pc] * 3)
        dset.attrs["gridstep_unit"] = "parsec"
        dset.attrs["sun_position"] = np.array([centre] * 3)
        dset.attrs["values_unit"] = "magnitude/parsec"


@pytest.mark.physics
def test_lallement_reader_integrates_constant_density(tmp_path: Path) -> None:
    path = tmp_path / "lallement.h5"
    _write_lallement(path, density=1.0e-3, half_width_pc=200.0, step_pc=10.0)
    reader = Lallement2019Map(path)
    vals, status = reader.query_native(
        np.array([0.0, 90.0, 0.0]), np.array([0.0, 0.0, 90.0]), np.array([0.1, 0.15, 0.5])
    )
    assert vals[0] == pytest.approx(0.1, rel=1e-3)  # 1e-3 mag/pc × 100 pc
    assert vals[1] == pytest.approx(0.15, rel=1e-3)
    assert status[:2].tolist() == [ExtinctionStatus.OK, ExtinctionStatus.OK]
    # 500 pc toward the NGP exits the cube at 200 pc → integral to the boundary.
    assert vals[2] == pytest.approx(0.2, rel=1e-3)
    assert status[2] == ExtinctionStatus.BEYOND_MAP_LIMIT


# --------------------------------------------------------------------------
# Real map files (optional: slow, skipped when the data tree lacks them)
# --------------------------------------------------------------------------


def _real_map(name: str) -> Path:
    cfg = load_config()
    spec = cfg.sample_selection.dust_maps.maps[name]
    path = repo_root() / cfg.paths.data_root / spec.path
    if not path.is_file():
        pytest.skip(f"{name} map not present at {path}")
    return path


@pytest.mark.slow
def test_real_bayestar_matches_mwdust_green19_inside_grid() -> None:
    mwdust = pytest.importorskip("mwdust")
    path = _real_map("green2019")
    mw_file = Path(mwdust.util.download.dust_dir) / "green19" / "bayestar2019.h5"
    if not mw_file.is_file():
        pytest.skip("mwdust green19 file not linked")
    ours = Bayestar2019Map(path)
    theirs = mwdust.Green19()
    l = np.array([30.0, 120.0, 200.0, 75.0])
    b = np.array([2.0, -5.0, 20.0, 0.5])
    d = np.array([0.5, 1.0, 2.0, 3.0])
    vals, status = ours.query_native(l, b, d)
    ref = np.array([float(theirs(li, bi, di)[0]) for li, bi, di in zip(l, b, d)])
    assert np.all(status == ExtinctionStatus.OK)
    np.testing.assert_allclose(vals, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_real_lallement_is_finite_and_monotone() -> None:
    reader = Lallement2019Map(_real_map("lallement2019"))
    d = np.array([0.1, 0.5, 1.0, 2.0])
    vals, _ = reader.query_native(np.full(4, 300.0), np.full(4, -2.0), d)
    assert np.all(np.isfinite(vals))
    assert np.all(np.diff(vals) >= -1e-9)
    assert math.isfinite(float(vals[-1])) and vals[-1] > 0.0


@pytest.mark.api
def test_extinction_counts_reach_the_evaluation_result(tmp_path: Path) -> None:
    from darkhunter_pop.sample_selection import sample_evaluation_result_from_dict

    registry = SampleSelectionRegistry(load_config())
    selection = registry.selection("elbadry2026")
    selection.extinction_lookup, _ = _lookup(tmp_path)
    rows = [
        {"source_id": 21, "nss_solution_type": "Orbital", "abs_g_mag": 4.0, "bp_rp": 0.8,
         "ra_deg": 10.0, "dec_deg": 10.0, "parallax_mas": 1.0, "period_day": 300.0},
        {"source_id": 22, "nss_solution_type": "SB1", "bp_rp": 0.8, "ra_deg": 10.0,
         "dec_deg": 10.0, "parallax_mas": -1.0, "period_day": 300.0},
    ]
    result = selection.evaluate(rows)
    assert result.extinction_status_counts == {"ok": 1, "invalid_parallax": 1}
    again = sample_evaluation_result_from_dict(result.as_dict())
    assert again.extinction_status_counts == result.extinction_status_counts
