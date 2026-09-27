"""3D dust-map readers and the per-source ``E(B-V)`` cache (#258).

Two readers, one per map a frozen selection file can name by ``map:`` key:

* ``green2019`` — Bayestar19 (Green et al. 2019), the published
  ``bayestar2019.h5`` (Harvard Dataverse ``doi:10.7910/DVN/2EJ9TX``). A
  hierarchical nested-HEALPix map of cumulative reddening on a 120-point grid
  in distance modulus. Pixel lookup follows ``mwdust.HierarchicalHealpixMap``
  exactly (finest matching ``nside`` wins); interpolation is linear in distance
  modulus, which inside the grid is identical to ``mwdust.Green19``'s default
  ``interpk=1`` spline.
* ``lallement2019`` — Lallement et al. (2019, A&A 625, A135) Gaia-2MASS cube of
  differential extinction ``dA0/ds`` (mag pc⁻¹) on a Sun-centred Cartesian
  Galactic grid (VizieR ``J/A+A/625/A135``, ``map3D_GAIAdr2_feb2019.h5``). The
  native value returned is ``A0`` integrated from the Sun to the source.

Neither reader applies a unit conversion: they return the map's **native**
line-of-sight integral. Conversion to ``E(B-V)`` is a config value
(``sample_selection.dust_maps.maps.<name>.native_to_ebv``), applied by
:func:`native_to_ebv`, so the expensive per-source cache stays valid if a
conversion factor is revised.

Which map applies to which source is **not** decided here: it is the frozen
per-sample ``extinction:`` block (``ExtinctionSpec``), evaluated per sample by
:func:`hemisphere_codes`. This module is only consumed through
``elbadry2026_selection`` — El-Badry 2024 and Andrews do not deredden.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_schema import (
    DustMapFileSpec,
    DustMapsConfig,
    ExtinctionSpec,
)

logger = logging.getLogger(__name__)

#: Manual salt for the cache key. No longer the invalidation mechanism (#278):
#: :func:`extinction_fingerprint` also hashes this module's own source, so any
#: edit to a reader invalidates cached native values without a bump. Kept so a
#: deliberate invalidation that does not touch this file remains possible.
READER_VERSION = "1"


def module_source_sha256() -> str:
    """SHA-256 of this module's source bytes (#278).

    Same convention as ``run_management.compute_source_hash`` (raw file bytes),
    so the per-source cache is invalidated by exactly the edits that change the
    ``sample_selection`` stage ``source_hash`` through this module. Any edit —
    including a docstring — therefore forces a cold rebuild; that is the
    deliberate, conservative trade (a rebuild costs ~45 s).
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

FloatArray = NDArray[np.float64]


class DustMapError(RuntimeError):
    """A configured dust map is missing, unreadable, or fails its checksum."""


class ExtinctionStatus(IntEnum):
    """Per-source outcome of an ``E(B-V)`` lookup (stored in the cache)."""

    OK = 0
    #: Parallax missing, non-finite, or ≤ 0 — no distance, no lookup.
    INVALID_PARALLAX = 1
    #: Sky position has no pixel in the map (e.g. Bayestar south of δ ≈ −30°).
    OUTSIDE_MAP = 2
    #: RA/Dec missing or non-finite.
    INVALID_POSITION = 3
    #: Source lies beyond the map's distance/volume limit; the value is the
    #: integral to that limit (a lower bound). Counted and reported, still used.
    BEYOND_MAP_LIMIT = 4


#: Statuses whose native value is usable for dereddening.
USABLE_STATUSES: frozenset[int] = frozenset(
    {int(ExtinctionStatus.OK), int(ExtinctionStatus.BEYOND_MAP_LIMIT)}
)


class DustMapReader(Protocol):
    """Native line-of-sight integral at Galactic ``(l, b)`` and distance."""

    def query_native(
        self, l_deg: FloatArray, b_deg: FloatArray, d_kpc: FloatArray
    ) -> tuple[FloatArray, NDArray[np.int8]]:
        """Return ``(native_value, status)``; status uses :class:`ExtinctionStatus`."""
        ...


# ---------------------------------------------------------------------------
# Paths and checksums
# ---------------------------------------------------------------------------


def resolve_data_path(path: str, data_root: Path) -> Path:
    """Resolve ``path`` against ``data_root`` unless it is absolute (``~`` expanded)."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else data_root / candidate


def file_md5(path: Path, *, chunk_bytes: int = 1 << 24) -> str:
    """Streaming MD5 of a (possibly multi-hundred-MB) file."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_map_path(name: str, spec: DustMapFileSpec, data_root: Path) -> Path:
    path = resolve_data_path(spec.path, data_root)
    if not path.is_file():
        raise DustMapError(
            f"dust map {name!r} not found at {path} "
            f"(sample_selection.dust_maps.maps.{name}.path). The El-Badry 2026 "
            "extinction correction cannot run without it — see issue #258."
        )
    if spec.md5 is not None:
        got = file_md5(path)
        if got != spec.md5:
            raise DustMapError(
                f"dust map {name!r} at {path} has md5 {got}, config pins {spec.md5}"
            )
    return path


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def radec_to_galactic(
    ra_deg: FloatArray, dec_deg: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """ICRS ``(ra, dec)`` → Galactic ``(l, b)`` in degrees (astropy, vectorized)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(
        ra=np.asarray(ra_deg, dtype=np.float64) * u.deg,
        dec=np.asarray(dec_deg, dtype=np.float64) * u.deg,
        frame="icrs",
    ).galactic
    return (
        np.asarray(coords.l.deg, dtype=np.float64),
        np.asarray(coords.b.deg, dtype=np.float64),
    )


_DEC_CONDITION = re.compile(
    r"^\s*dec_deg\s*(?P<op>>=|<=|>|<)\s*(?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$"
)

_OPS: dict[str, Callable[[FloatArray, float], NDArray[np.bool_]]] = {
    ">": lambda x, v: x > v,
    ">=": lambda x, v: x >= v,
    "<": lambda x, v: x < v,
    "<=": lambda x, v: x <= v,
}


def declination_mask(applies_when: str, dec_deg: FloatArray) -> NDArray[np.bool_]:
    """Evaluate a frozen ``applies_when: "dec_deg <op> <value>"`` condition.

    Only single declination comparisons are supported — that is the only form
    the frozen selection files use. Anything else raises rather than guessing.
    """
    match = _DEC_CONDITION.match(applies_when)
    if match is None:
        raise ValueError(
            f"unsupported extinction applies_when {applies_when!r}; expected "
            "'dec_deg <op> <number>'"
        )
    return _OPS[match.group("op")](
        np.asarray(dec_deg, dtype=np.float64), float(match.group("value"))
    )


#: Hemisphere codes stored per source in the cache.
HEMISPHERE_NORTH = 0
HEMISPHERE_SOUTH = 1
HEMISPHERE_NONE = -1


def hemisphere_codes(spec: ExtinctionSpec, dec_deg: FloatArray) -> NDArray[np.int8]:
    """Which frozen hemisphere leg applies to each source.

    Refuses a split whose north/south conditions overlap for any finite input
    declination. Non-finite declinations get :data:`HEMISPHERE_NONE`.
    """
    dec = np.asarray(dec_deg, dtype=np.float64)
    finite = np.isfinite(dec)
    north = declination_mask(spec.north.applies_when, dec) & finite
    south = declination_mask(spec.south.applies_when, dec) & finite
    if np.any(north & south):
        raise ValueError("extinction north/south conditions overlap")
    codes = np.full(dec.shape, HEMISPHERE_NONE, dtype=np.int8)
    codes[north] = HEMISPHERE_NORTH
    codes[south] = HEMISPHERE_SOUTH
    if np.any(finite & (codes == HEMISPHERE_NONE)):
        raise ValueError("extinction north/south conditions leave a declination gap")
    return codes


def distance_kpc_from_parallax(parallax_mas: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    """``d = 1 / ϖ`` (kpc for ϖ in mas); returns ``(d, valid)`` with NaN where invalid."""
    plx = np.asarray(parallax_mas, dtype=np.float64)
    valid = np.isfinite(plx) & (plx > 0.0)
    dist = np.full(plx.shape, np.nan, dtype=np.float64)
    dist[valid] = 1.0 / plx[valid]
    return dist, valid


# ---------------------------------------------------------------------------
# Bayestar19 (green2019)
# ---------------------------------------------------------------------------


class Bayestar2019Map:
    """Bayestar19 cumulative reddening in native Bayestar19 units.

    Inside the published distance-modulus grid the value is linear in distance
    modulus (``mwdust.Green19`` with ``interpk=1`` agrees there). Outside it:

    * nearer than the first grid node: linear in *distance* from zero at the Sun
      to the first node (``mwdust`` extrapolates the spline instead, which can
      go negative) — status ``OK``;
    * farther than the last node: held at the last node — status
      ``BEYOND_MAP_LIMIT``.

    Pixels absent from the map (the survey footprint ends near δ ≈ −30°) are
    ``OUTSIDE_MAP``.
    """

    def __init__(self, path: Path) -> None:
        with h5py.File(path, "r") as handle:
            self._pix_info = handle["pixel_info"][:]
            self._best_fit = np.asarray(handle["best_fit"][:], dtype=np.float32)
            n_dm = int(self._best_fit.shape[1])
            dm_min = handle.attrs.get("DM_min")
            dm_max = handle.attrs.get("DM_max")
        # Published Bayestar19 grid (Green et al. 2019; same as mwdust.Green19):
        # 120 nodes spanning distance modulus 4 … 18.875. The file carries no
        # attributes for it, so fall back to the published values.
        lo = 4.0 if dm_min is None else float(dm_min)
        hi = 18.875 if dm_max is None else float(dm_max)
        self._distmods = np.linspace(lo, hi, n_dm)
        nsides = np.unique(self._pix_info["nside"])
        # Finest resolution first — the first match wins (mwdust convention).
        self._by_nside: list[tuple[int, NDArray[np.uint64], NDArray[np.int64]]] = []
        for nside in sorted((int(n) for n in nsides), reverse=True):
            rows = np.flatnonzero(self._pix_info["nside"] == nside)
            hp = np.asarray(self._pix_info["healpix_index"][rows], dtype=np.int64)
            order = np.argsort(hp)
            self._by_nside.append((nside, hp[order], rows[order]))

    def _pixel_rows(self, l_deg: FloatArray, b_deg: FloatArray) -> NDArray[np.int64]:
        from mwdust.util.healpix import ang2pix

        theta = np.deg2rad(90.0 - b_deg)
        phi = np.deg2rad(l_deg)
        out = np.full(l_deg.shape, -1, dtype=np.int64)
        pending = np.ones(l_deg.shape, dtype=bool)
        for nside, sorted_hp, rows in self._by_nside:
            if not pending.any():
                break
            idx = np.flatnonzero(pending)
            tpix = np.asarray(ang2pix(nside, theta[idx], phi[idx], nest=True), dtype=np.int64)
            pos = np.searchsorted(sorted_hp, tpix)
            inside = pos < sorted_hp.size
            hit = np.zeros(idx.size, dtype=bool)
            hit[inside] = sorted_hp[pos[inside]] == tpix[inside]
            out[idx[hit]] = rows[pos[hit]]
            pending[idx[hit]] = False
        return out

    def query_native(
        self, l_deg: FloatArray, b_deg: FloatArray, d_kpc: FloatArray
    ) -> tuple[FloatArray, NDArray[np.int8]]:
        l_deg = np.asarray(l_deg, dtype=np.float64)
        b_deg = np.asarray(b_deg, dtype=np.float64)
        d_kpc = np.asarray(d_kpc, dtype=np.float64)
        values = np.full(d_kpc.shape, np.nan, dtype=np.float64)
        status = np.full(d_kpc.shape, int(ExtinctionStatus.OK), dtype=np.int8)
        rows = self._pixel_rows(l_deg, b_deg)
        status[rows < 0] = int(ExtinctionStatus.OUTSIDE_MAP)
        good = rows >= 0
        if not good.any():
            return values, status
        profiles = self._best_fit[rows[good]].astype(np.float64)  # (n, n_dm)
        dm = 5.0 * np.log10(d_kpc[good]) + 10.0
        grid = self._distmods
        step = grid[1] - grid[0]
        # Fractional node position, clipped into the grid; then linear blend.
        pos = np.clip((dm - grid[0]) / step, 0.0, grid.size - 1.0)
        lo = np.minimum(np.floor(pos).astype(np.int64), grid.size - 2)
        frac = pos - lo
        n = np.arange(profiles.shape[0])
        interp = profiles[n, lo] * (1.0 - frac) + profiles[n, lo + 1] * frac
        near = dm < grid[0]
        if near.any():
            d_first = 10.0 ** (grid[0] / 5.0 - 2.0)  # kpc at the first node
            interp[near] = profiles[near, 0] * (d_kpc[good][near] / d_first)
        far = dm > grid[-1]
        sub_status = status[good]
        sub_status[far] = int(ExtinctionStatus.BEYOND_MAP_LIMIT)
        status[good] = sub_status
        values[good] = interp
        return values, status


# ---------------------------------------------------------------------------
# Lallement et al. (2019) cube (lallement2019)
# ---------------------------------------------------------------------------


class Lallement2019Map:
    """Lallement et al. (2019) ``A0`` integrated from the Sun along the sightline.

    The cube stores differential extinction ``dA0/ds`` (mag pc⁻¹) on a regular
    Sun-centred Cartesian Galactic grid (X toward the Galactic centre, Y toward
    rotation, Z toward the NGP). The integral uses trilinear interpolation of
    the density at ``step_pc`` spacing along the sightline (trapezoid rule).
    Sources beyond the cube boundary get the integral to the boundary and status
    ``BEYOND_MAP_LIMIT`` (a lower bound; dust beyond ±400 pc of the plane and
    3 kpc in-plane is negligible for most of the sample, but the count is
    reported, not hidden).
    """

    def __init__(self, path: Path, *, step_pc: float | None = None) -> None:
        with h5py.File(path, "r") as handle:
            density, origin_pc, spacing_pc = _read_lallement_cube(handle)
        self._density = density  # (nx, ny, nz), mag / pc, axis order X, Y, Z
        self._origin = origin_pc  # coordinate of voxel (0,0,0) centre, pc
        self._spacing = spacing_pc  # pc per voxel along X, Y, Z
        self._step = float(step_pc) if step_pc is not None else float(np.min(spacing_pc)) / 2.0
        shape = np.asarray(density.shape, dtype=np.float64)
        self._lo = origin_pc
        self._hi = origin_pc + (shape - 1.0) * spacing_pc

    def _trilinear(self, xyz: FloatArray) -> FloatArray:
        """Trilinear density at points ``xyz`` (n, 3) in pc; 0 outside the grid."""
        f = (xyz - self._origin) / self._spacing
        shape = np.asarray(self._density.shape)
        inside = np.all((f >= 0.0) & (f <= shape - 1.0), axis=1)
        out = np.zeros(xyz.shape[0], dtype=np.float64)
        if not inside.any():
            return out
        fi = f[inside]
        i0 = np.minimum(np.floor(fi).astype(np.int64), shape - 2)
        t = fi - i0
        acc = np.zeros(fi.shape[0], dtype=np.float64)
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1.0 - t[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1.0 - t[:, 2]
                    acc += (
                        wx
                        * wy
                        * wz
                        * self._density[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                    )
        out[inside] = acc
        return out

    def _exit_distance_pc(self, unit: FloatArray) -> FloatArray:
        """Distance from the Sun to the cube boundary along each unit vector."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hi = np.where(unit > 0, self._hi / unit, np.inf)
            t_lo = np.where(unit < 0, self._lo / unit, np.inf)
        return np.min(np.minimum(t_hi, t_lo), axis=1)

    def query_native(
        self, l_deg: FloatArray, b_deg: FloatArray, d_kpc: FloatArray
    ) -> tuple[FloatArray, NDArray[np.int8]]:
        l_rad = np.deg2rad(np.asarray(l_deg, dtype=np.float64))
        b_rad = np.deg2rad(np.asarray(b_deg, dtype=np.float64))
        d_pc = np.asarray(d_kpc, dtype=np.float64) * 1.0e3
        unit = np.stack(
            [
                np.cos(b_rad) * np.cos(l_rad),
                np.cos(b_rad) * np.sin(l_rad),
                np.sin(b_rad),
            ],
            axis=1,
        )
        exit_pc = self._exit_distance_pc(unit)
        status = np.full(d_pc.shape, int(ExtinctionStatus.OK), dtype=np.int8)
        beyond = d_pc > exit_pc
        status[beyond] = int(ExtinctionStatus.BEYOND_MAP_LIMIT)
        s_max = np.minimum(d_pc, exit_pc)
        values = np.zeros(d_pc.shape, dtype=np.float64)
        # March sightlines in chunks sorted by length; within a chunk every
        # sightline uses the same number of nodes, so its step is ≤ self._step.
        if not s_max.size or float(np.max(s_max)) <= 0.0:
            return values, status
        # Chunk sources to bound memory; each chunk walks its own step count.
        order = np.argsort(s_max)
        chunk = 4096
        for start in range(0, order.size, chunk):
            idx = order[start : start + chunk]
            smax = s_max[idx]
            k = max(1, int(np.ceil(float(smax.max()) / self._step)))
            frac = np.linspace(0.0, 1.0, k + 1)  # (k+1,)
            s = smax[:, None] * frac[None, :]  # (m, k+1)
            pts = unit[idx][:, None, :] * s[:, :, None]  # (m, k+1, 3)
            dens = self._trilinear(pts.reshape(-1, 3)).reshape(s.shape)
            values[idx] = np.trapz(dens, s, axis=1)
        return values, status


def _read_lallement_cube(
    handle: h5py.File,
) -> tuple[NDArray[np.float32], FloatArray, FloatArray]:
    """Parse the VizieR ``map3D_GAIAdr2_feb2019.h5`` layout → (density, origin, spacing).

    Published layout (inspected, VizieR ``J/A+A/625/A135``): dataset
    ``stilism/cube_datas`` of shape ``(1201, 1201, 161)`` float32 in axis order
    X, Y, Z with attributes ``gridstep_values = [5, 5, 5]`` (``gridstep_unit =
    parsec``), ``sun_position = [600.5, 600.5, 80.5]`` and ``values_unit =
    magnitude/parsec``. ``sun_position`` is in voxel-index units where voxel
    ``k`` (0-based) spans ``[k, k + 1)``, i.e. the Sun sits at the centre of
    voxel 600/600/80 — the grid's midpoint, ±3000 pc in X/Y and ±400 pc in Z.
    Returned ``origin`` is the Sun-centred coordinate (pc) of voxel (0,0,0)'s
    centre.
    """
    dset = handle[_LALLEMENT_DATASET]
    unit = dset.attrs.get("gridstep_unit")
    if unit is not None and _as_str(unit).strip().lower() not in {"parsec", "pc"}:
        raise DustMapError(f"unexpected Lallement grid unit {unit!r}")
    values_unit = dset.attrs.get("values_unit")
    if values_unit is not None and _as_str(values_unit).strip().lower() != "magnitude/parsec":
        raise DustMapError(f"unexpected Lallement values unit {values_unit!r}")
    spacing = np.asarray(dset.attrs["gridstep_values"], dtype=np.float64)
    sun = np.asarray(dset.attrs["sun_position"], dtype=np.float64)
    if spacing.shape != (3,) or sun.shape != (3,):
        raise DustMapError("Lallement cube attributes must be 3-vectors")
    density = np.asarray(dset[...], dtype=np.float32)
    origin = -(sun - 0.5) * spacing
    return density, origin, spacing


_LALLEMENT_DATASET = "stilism/cube_datas"


def _as_str(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


# ---------------------------------------------------------------------------
# Registry, conversion, cache
# ---------------------------------------------------------------------------

_READERS: dict[str, Callable[[Path], DustMapReader]] = {
    "green2019": Bayestar2019Map,
    "lallement2019": Lallement2019Map,
}


def supported_maps() -> tuple[str, ...]:
    """Map keys this module can read (other names in a selection file raise)."""
    return tuple(sorted(_READERS))


def native_to_ebv(
    native: FloatArray,
    map_codes: NDArray[np.int8],
    spec: ExtinctionSpec,
    dust_cfg: DustMapsConfig,
) -> FloatArray:
    """Convert cached native integrals to ``E(B-V)`` with the configured factors."""
    out = np.full(np.shape(native), np.nan, dtype=np.float64)
    for code, leg in ((HEMISPHERE_NORTH, spec.north), (HEMISPHERE_SOUTH, spec.south)):
        sel = map_codes == code
        if not sel.any():
            continue
        file_spec = dust_cfg.maps.get(leg.map)
        if file_spec is None:
            raise DustMapError(
                f"extinction map {leg.map!r} has no sample_selection.dust_maps.maps entry"
            )
        out[sel] = np.asarray(native, dtype=np.float64)[sel] * float(file_spec.native_to_ebv)
    return out


def extinction_fingerprint(spec: ExtinctionSpec, dust_cfg: DustMapsConfig) -> str:
    """Stable key for a native-value cache: split + map identities + reader code.

    "Reader code" is :func:`module_source_sha256` (plus the :data:`READER_VERSION`
    salt), so a change to any reader's numerics can never reuse stale cached
    integrals (#278). Deliberately excludes ``native_to_ebv`` and the photometric coefficients —
    those are applied after the cache, so revising them never invalidates it.
    """
    digest = hashlib.sha256()
    digest.update(READER_VERSION.encode())
    digest.update(b"\0" + module_source_sha256().encode())
    for leg in (spec.north, spec.south):
        file_spec = dust_cfg.maps.get(leg.map)
        digest.update(b"\0" + leg.applies_when.encode() + b"\0" + leg.map.encode())
        if file_spec is not None:
            digest.update(b"\0" + (file_spec.md5 or Path(file_spec.path).name).encode())
    return digest.hexdigest()[:16]


@dataclass
class ExtinctionTable:
    """Per-source native line-of-sight integrals for one extinction policy."""

    source_id: NDArray[np.int64]
    ra_deg: FloatArray
    dec_deg: FloatArray
    parallax_mas: FloatArray
    hemisphere: NDArray[np.int8]
    native: FloatArray
    status: NDArray[np.int8]
    _index: dict[tuple[int, float, float, float], int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {
            _row_key(int(s), float(r), float(d), float(p)): i
            for i, (s, r, d, p) in enumerate(
                zip(self.source_id, self.ra_deg, self.dec_deg, self.parallax_mas)
            )
        }

    def __len__(self) -> int:
        return int(self.source_id.size)

    def lookup(self, source_id: int, ra: float, dec: float, parallax: float) -> int | None:
        return self._index.get(_row_key(source_id, ra, dec, parallax))

    @classmethod
    def empty(cls) -> ExtinctionTable:
        return cls(
            source_id=np.zeros(0, dtype=np.int64),
            ra_deg=np.zeros(0),
            dec_deg=np.zeros(0),
            parallax_mas=np.zeros(0),
            hemisphere=np.zeros(0, dtype=np.int8),
            native=np.zeros(0),
            status=np.zeros(0, dtype=np.int8),
        )

    def extend(self, other: ExtinctionTable) -> ExtinctionTable:
        return ExtinctionTable(
            source_id=np.concatenate([self.source_id, other.source_id]),
            ra_deg=np.concatenate([self.ra_deg, other.ra_deg]),
            dec_deg=np.concatenate([self.dec_deg, other.dec_deg]),
            parallax_mas=np.concatenate([self.parallax_mas, other.parallax_mas]),
            hemisphere=np.concatenate([self.hemisphere, other.hemisphere]),
            native=np.concatenate([self.native, other.native]),
            status=np.concatenate([self.status, other.status]),
        )


def _row_key(source_id: int, ra: float, dec: float, parallax: float) -> tuple[int, float, float, float]:
    # NaN != NaN would make NaN-keyed rows unfindable; map them to a sentinel.
    def k(v: float) -> float:
        return -1.0e300 if v != v else float(v)

    return (int(source_id), k(ra), k(dec), k(parallax))


_CACHE_COLUMNS = ("source_id", "ra_deg", "dec_deg", "parallax_mas", "hemisphere", "native", "status")


def read_extinction_cache(path: Path) -> ExtinctionTable:
    """Load a native-value cache written by :func:`write_extinction_cache`."""
    with h5py.File(path, "r") as handle:
        cols = {name: handle[name][:] for name in _CACHE_COLUMNS}
    return ExtinctionTable(**cols)


def write_extinction_cache(path: Path, table: ExtinctionTable, *, fingerprint: str) -> None:
    """Atomically write the per-source cache (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as handle:
        for name in _CACHE_COLUMNS:
            handle.create_dataset(name, data=getattr(table, name))
        handle.attrs["fingerprint"] = fingerprint
        handle.attrs["reader_version"] = READER_VERSION
        handle.attrs["reader_source_sha256"] = module_source_sha256()
        handle.attrs["n_rows"] = len(table)
    tmp.replace(path)


def compute_native_extinction(
    ra_deg: FloatArray,
    dec_deg: FloatArray,
    parallax_mas: FloatArray,
    spec: ExtinctionSpec,
    readers: Mapping[str, DustMapReader] | Callable[[str], DustMapReader],
) -> tuple[NDArray[np.int8], FloatArray, NDArray[np.int8]]:
    """Query the frozen hemisphere map for each source.

    Returns ``(hemisphere_code, native_value, status)``. ``readers`` maps a map
    key to a reader, or is a factory called lazily only for hemispheres that
    actually have sources.
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    hemi = hemisphere_codes(spec, dec)
    dist, plx_ok = distance_kpc_from_parallax(parallax_mas)
    pos_ok = np.isfinite(ra) & np.isfinite(dec)
    native = np.full(ra.shape, np.nan, dtype=np.float64)
    status = np.full(ra.shape, int(ExtinctionStatus.OK), dtype=np.int8)
    status[~plx_ok] = int(ExtinctionStatus.INVALID_PARALLAX)
    status[~pos_ok] = int(ExtinctionStatus.INVALID_POSITION)
    queryable = plx_ok & pos_ok
    if not queryable.any():
        return hemi, native, status
    l_deg = np.full(ra.shape, np.nan)
    b_deg = np.full(ra.shape, np.nan)
    l_deg[queryable], b_deg[queryable] = radec_to_galactic(ra[queryable], dec[queryable])
    for code, leg in ((HEMISPHERE_NORTH, spec.north), (HEMISPHERE_SOUTH, spec.south)):
        sel = queryable & (hemi == code)
        if not sel.any():
            continue
        if leg.map not in _READERS:
            raise DustMapError(
                f"extinction map {leg.map!r} is not implemented (supported: {supported_maps()})"
            )
        reader = readers(leg.map) if callable(readers) else readers[leg.map]
        vals, stat = reader.query_native(l_deg[sel], b_deg[sel], dist[sel])
        native[sel] = vals
        status[sel] = stat
    return hemi, native, status


class ExtinctionLookup:
    """Lazy, cached ``E(B-V)`` provider for one frozen extinction policy.

    Map files are opened only when a source missing from the cache must be
    queried; a missing map is then a hard :class:`DustMapError`, never a silent
    fall-back to undereddened photometry.
    """

    def __init__(
        self,
        spec: ExtinctionSpec,
        dust_cfg: DustMapsConfig,
        *,
        data_root: Path,
        cache_tag: str,
        readers: Mapping[str, DustMapReader] | None = None,
        use_cache: bool = True,
    ) -> None:
        self.spec = spec
        self.dust_cfg = dust_cfg
        self.data_root = data_root
        self.fingerprint = extinction_fingerprint(spec, dust_cfg)
        self.cache_path = (
            resolve_data_path(dust_cfg.ebv_cache_dir, data_root)
            / f"{cache_tag}_{self.fingerprint}.h5"
        )
        self._readers: dict[str, DustMapReader] = dict(readers or {})
        self._use_cache = use_cache
        self._table: ExtinctionTable | None = None

    def _reader(self, name: str) -> DustMapReader:
        if name not in self._readers:
            file_spec = self.dust_cfg.maps.get(name)
            if file_spec is None:
                raise DustMapError(
                    f"extinction map {name!r} has no sample_selection.dust_maps.maps entry"
                )
            path = _checked_map_path(name, file_spec, self.data_root)
            logger.info("loading dust map %s from %s", name, path)
            self._readers[name] = _READERS[name](path)
        return self._readers[name]

    def _load_table(self) -> ExtinctionTable:
        if self._table is None:
            if self._use_cache and self.cache_path.is_file():
                self._table = read_extinction_cache(self.cache_path)
            else:
                self._table = ExtinctionTable.empty()
        return self._table

    def ebv_for(
        self,
        source_id: Sequence[int],
        ra_deg: Sequence[float],
        dec_deg: Sequence[float],
        parallax_mas: Sequence[float],
    ) -> tuple[FloatArray, NDArray[np.int8], NDArray[np.int8]]:
        """``(E(B-V), status, hemisphere)`` per source; queries maps for cache misses."""
        sid = np.asarray(source_id, dtype=np.int64)
        ra = np.asarray(ra_deg, dtype=np.float64)
        dec = np.asarray(dec_deg, dtype=np.float64)
        plx = np.asarray(parallax_mas, dtype=np.float64)
        table = self._load_table()
        pos = np.array(
            [
                -1 if (i := table.lookup(int(s), float(r), float(d), float(p))) is None else i
                for s, r, d, p in zip(sid, ra, dec, plx)
            ],
            dtype=np.int64,
        )
        miss = pos < 0
        if miss.any():
            # Deduplicate misses on the full key before querying.
            keys = {}
            for j in np.flatnonzero(miss):
                keys.setdefault(_row_key(int(sid[j]), float(ra[j]), float(dec[j]), float(plx[j])), j)
            first = np.fromiter(keys.values(), dtype=np.int64)
            logger.info(
                "querying dust maps for %d sources not in %s", first.size, self.cache_path
            )
            hemi, native, status = compute_native_extinction(
                ra[first], dec[first], plx[first], self.spec, self._reader
            )
            new = ExtinctionTable(
                source_id=sid[first],
                ra_deg=ra[first],
                dec_deg=dec[first],
                parallax_mas=plx[first],
                hemisphere=hemi,
                native=native,
                status=status,
            )
            table = table.extend(new)
            self._table = table
            if self._use_cache:
                write_extinction_cache(self.cache_path, table, fingerprint=self.fingerprint)
            pos = np.array(
                [table.lookup(int(s), float(r), float(d), float(p)) for s, r, d, p in zip(sid, ra, dec, plx)],
                dtype=np.int64,
            )
        hemi_all = table.hemisphere[pos]
        status_all = table.status[pos]
        ebv = native_to_ebv(table.native[pos], hemi_all, self.spec, self.dust_cfg)
        unusable = ~np.isin(status_all, list(USABLE_STATUSES))
        ebv[unusable] = np.nan
        return ebv, status_all, hemi_all
