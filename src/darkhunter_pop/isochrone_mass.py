"""Primary mass from MIST isochrones fitted to the dereddened Gaia CMD (#418).

docs/MOCK_POPULATION_SPEC.md §11 and docs/ARCHITECTURE.md ``mass_derivation_bulk``. A star's
dereddened absolute magnitude and colour

    y = (C0, M_G0),   C0 = (BP−RP) − E(BP−RP),   M_G0 = G − μ(d) − A_G

are compared with the MIST v1.2 (vvcrit 0.4) isochrones (Choi et al. 2016; Dotter 2016) put into
Gaia DR3 bands with the MIST UBVRIplus bolometric-correction tables (``Gaia_*_EDR3`` columns;
DR3 and EDR3 photometry are the same). The posterior over isochrone points ``j`` (initial
mass, age, [Fe/H], EEP) is

    p(j | y) ∝ w_j N(y; y_j, Σ_y),   w_j = ξ(M_init,j) ΔM_init,j × p(τ_j) Δτ_j × p([Fe/H]_j) Δ[Fe/H]_j

with ξ the IMF, p(τ) the age prior and p([Fe/H]) the metallicity prior (all config;
``provisional_*`` where Ryan has not chosen). The current mass M1 posterior is summarized by
moments (mean, σ, and the mean and σ of log10 M1), with the same moments for the initial
(progenitor) mass, the current and maximum past radius, log g, age, [Fe/H] and the probability
of being past the main sequence (MIST phase ≥ 2).

**Computation.** The prior-weighted isochrone points are deposited once into a fine (C0, M_G0)
map with one channel per moment (:func:`build_cmd_map`). With a Gaussian likelihood that is
diagonal in (C0, M_G0) the posterior moments are then two cell-integrated 1-D kernels around the
map, ``k_C^T Map_q k_M`` per star, evaluated as one matrix product per chunk of colour-sorted
stars (:func:`posterior_moments`). About 10⁵ stars per ~10 s on one thread.

**What this module does not do.** It never decides whether a star is a binary. A blended
system's CMD position biases M1 (spec §11.3); the mock and the data side apply the **same**
estimator to the same kind of photometry, and the 2-D Malmquist weight
(:mod:`darkhunter_pop.malmquist_cmd`) carries the companion's displacement. Nothing here
imports or reimplements gaiamock.

Data: read in place from ``isochrone_mass.mist_root`` (host-specific path, config). The parsed
native grid is cached under ``<data_root>/<cache_subdir>/`` with a SHA256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from darkhunter_pop import constants

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Columns kept from a MIST ``*.iso`` file, by 0-based index (MIST v1.2 full isochrone layout:
#: EEP, log10_isochrone_age_yr, initial_mass, star_mass, log_L, log_Teff, log_R, log_g, phase).
MIST_ISO_COLUMNS: dict[str, int] = {
    "eep": 0,
    "log_age": 1,
    "initial_mass": 2,
    "star_mass": 3,
    "log_l": 8,
    "log_teff": 13,
    "log_r": 15,
    "log_g": 16,
    "phase": 78,
}
#: Number of columns in a MIST v1.2 full isochrone row (checked on parse).
MIST_ISO_NCOLS: int = 79
#: Leading columns of a UBVRIplus BC table: Teff, logg, [Fe/H], Av, Rv.
BC_INDEX_COLUMNS: tuple[str, ...] = ("teff", "logg", "feh", "av", "rv")

#: Moment channels of the CMD map, in order (``value`` is the per-point quantity).
MOMENT_CHANNELS: tuple[str, ...] = (
    "one",
    "mass",
    "mass2",
    "log_mass",
    "log_mass2",
    "initial_mass",
    "log_r",
    "log_r2",
    "r_max_past",
    "log_g",
    "evolved",
    "log_age",
    "feh",
)
#: Bumped whenever the per-star outputs change, so result caches keyed by
#: :func:`config_key` are rebuilt.
RESULT_SCHEMA_VERSION: int = 2


# ---------------------------------------------------------------------------
# Config (``isochrone_mass`` section of PipelineConfig)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MistGridConfig(_Strict):
    """Which MIST files and which part of the grid to read (numerical/data choices)."""

    iso_subdir: str = "mist/MIST_v1.2_vvcrit0.4_full_isos"
    iso_filename_template: str = "MIST_v1.2_feh_{feh}_afe_p0.0_vvcrit0.4_full.iso"
    bc_subdir: str = "BC/mist"
    bc_filename_template: str = "feh{feh}.UBVRIplus"
    #: Gaia DR3 = EDR3 passbands (Riello et al. 2021).
    band_g: str = "Gaia_G_EDR3"
    band_bp: str = "Gaia_BP_EDR3"
    band_rp: str = "Gaia_RP_EDR3"
    feh_values: tuple[float, ...] = (-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5)
    log_age_min: float = 8.0
    log_age_max: float = 10.15
    #: MIST phases kept: 0 MS, 2 SGB+RGB, 3 CHeB, 4 EAGB, 5 TPAGB (PMS −1 and post-AGB/WD 6, 9 dropped).
    phases: tuple[int, ...] = (0, 2, 3, 4, 5)
    #: EEP-matched linear sub-steps between adjacent isochrones (age) and [Fe/H] files.
    n_age_substeps: int = Field(4, ge=1)
    n_feh_substeps: int = Field(5, ge=1)

    @model_validator(mode="after")
    def _order(self) -> MistGridConfig:
        if self.log_age_min >= self.log_age_max:
            raise ValueError("log_age_min must be < log_age_max")
        if list(self.feh_values) != sorted(self.feh_values) or len(self.feh_values) < 2:
            raise ValueError("feh_values must be increasing with at least two entries")
        return self


class CmdMapConfig(_Strict):
    """Fine (C0, M_G0) map the prior-weighted isochrone points are deposited into."""

    colour_min: float = -0.6
    colour_max: float = 4.6
    colour_step: float = Field(0.01, gt=0)
    mag_min: float = -4.0
    mag_max: float = 16.0
    mag_step: float = Field(0.02, gt=0)


class ImfPriorConfig(_Strict):
    """Kroupa (2001) broken power law ξ(M) ∝ M^−α in initial mass (α breaks in M⊙)."""

    kind: Literal["kroupa2001"] = "kroupa2001"
    breaks_msun: tuple[float, ...] = (0.08, 0.5)
    alphas: tuple[float, ...] = (0.3, 1.3, 2.3)

    @model_validator(mode="after")
    def _len(self) -> ImfPriorConfig:
        if len(self.alphas) != len(self.breaks_msun) + 1:
            raise ValueError("need one more slope than breaks")
        return self


class AgePriorConfig(_Strict):
    """Age prior. ``uniform_linear``: constant star-formation rate between the grid ends."""

    kind: Literal["uniform_linear", "uniform_log"] = "uniform_linear"


class FehPriorConfig(_Strict):
    """[Fe/H] prior: Gaussian (solar-neighbourhood-like). Values are provisional (MP-Q33)."""

    kind: Literal["gaussian"] = "gaussian"
    mean_dex: float = -0.1
    sigma_dex: float = Field(0.25, gt=0)


class IsochroneLikelihoodConfig(_Strict):
    """Per-star Gaussian likelihood widths, diagonal in (C0, M_G0).

    σ_C² = colour_floor² + (σ_E(B−V) k_E)², σ_M² = σ_μ² + mag_floor² + (σ_E(B−V) k_A)²
    with the E(B−V) error ``ebv_fractional_sigma × E(B−V)``. The floors stand for photometric
    calibration and model (isochrone + BC) systematics; they are provisional (MP-Q34).
    """

    provisional_colour_floor_mag: float = Field(0.02, gt=0)
    provisional_mag_floor_mag: float = Field(0.05, gt=0)
    provisional_ebv_fractional_sigma: float = Field(0.0, ge=0)
    #: Stars whose prior-predictive density at y is below this (per mag²) get no M1
    #: (off the isochrone grid: white dwarfs, hot subdwarfs, bad photometry).
    min_log10_evidence: float = -4.0
    #: Kernel half-width in σ (numerical only).
    kernel_n_sigma: float = Field(6.0, gt=2)
    chunk_rows: int = Field(2000, ge=10)


class IsochroneMassConfig(_Strict):
    """``isochrone_mass`` section (spec §11). ``mist_root`` is host-specific (host profiles)."""

    mist_root: str | None = None
    cache_subdir: str = "isochrone_mist"
    grid: MistGridConfig = MistGridConfig()
    cmd_map: CmdMapConfig = CmdMapConfig()
    imf: ImfPriorConfig = ImfPriorConfig()
    age: AgePriorConfig = AgePriorConfig()
    provisional_feh_prior: FehPriorConfig = FehPriorConfig()
    likelihood: IsochroneLikelihoodConfig = IsochroneLikelihoodConfig()
    #: Which M1 summary feeds point uses (mock truth M1 and bulk M1): the posterior mean of
    #: M1 or exp of the mean of ln M1 (MP-Q35, provisional).
    provisional_point_estimate: Literal["mean", "log_mean"] = "mean"


def load_isochrone_config(path: str | Path, key: str = "isochrone_mass") -> IsochroneMassConfig:
    """Validate the ``key`` section of a YAML file (used by scripts and tests)."""
    return IsochroneMassConfig.model_validate(yaml.safe_load(Path(path).read_text())[key])


# ---------------------------------------------------------------------------
# Parsing MIST (read in place) and building the native grid
# ---------------------------------------------------------------------------


def _feh_token(feh: float, style: Literal["iso", "bc"]) -> str:
    """MIST file token: iso ``m0.25`` / ``p0.00``; BC ``m025`` / ``p000``."""
    sign = "m" if feh < 0 else "p"
    if style == "iso":
        return f"{sign}{abs(feh):.2f}"
    return f"{sign}{int(round(abs(feh) * 100)):03d}"


def read_mist_iso(path: Path) -> FloatArray:
    """Parse one MIST full-isochrone file into an (n, 9) array (``MIST_ISO_COLUMNS`` order)."""
    import pandas as pd

    first = None
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                first = line
                break
    if first is None or len(first.split()) != MIST_ISO_NCOLS:
        raise ValueError(f"{path}: expected {MIST_ISO_NCOLS} columns per row")
    idx = list(MIST_ISO_COLUMNS.values())
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=idx, engine="c")
    return df[idx].to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class BcTable:
    """UBVRIplus BCs at A_V = 0 on the regular (Teff, log g) grid of one [Fe/H] file."""

    teff: FloatArray
    logg: FloatArray
    bc: dict[str, FloatArray]  # band -> (n_teff, n_logg)

    def interpolate(self, teff: ArrayLike, logg: ArrayLike) -> dict[str, FloatArray]:
        """Bilinear BC at (Teff, log g), clamped to the table edges."""
        from scipy.interpolate import RegularGridInterpolator

        pts = np.column_stack([
            np.clip(np.asarray(teff, float), self.teff[0], self.teff[-1]),
            np.clip(np.asarray(logg, float), self.logg[0], self.logg[-1]),
        ])
        return {
            b: RegularGridInterpolator((self.teff, self.logg), v, method="linear")(pts) for b, v in self.bc.items()
        }


def read_bc_table(path: Path, bands: tuple[str, ...]) -> BcTable:
    """Parse one ``feh*.UBVRIplus`` file; keep A_V = 0 rows and the named bands."""
    import pandas as pd

    header: list[str] | None = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") and "Teff" in line and "logg" in line:
                header = line.lstrip("#").split()
                break
    if header is None:
        raise ValueError(f"{path}: no column header")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=header, engine="c")
    df = df[np.isclose(df["Av"].to_numpy(float), 0.0)]
    teff = np.unique(df["Teff"].to_numpy(float))
    logg = np.unique(df["logg"].to_numpy(float))
    if len(df) != teff.size * logg.size:
        raise ValueError(f"{path}: A_V = 0 rows are not a full (Teff, logg) grid")
    df = df.sort_values(["Teff", "logg"])
    bc = {b: df[b].to_numpy(float).reshape(teff.size, logg.size) for b in bands}
    return BcTable(teff=teff, logg=logg, bc=bc)


#: Quantities stored per native grid point (``NativeGrid.values`` last axis).
NATIVE_QUANTITIES: tuple[str, ...] = (
    "initial_mass",
    "star_mass",
    "log_r",
    "log_g",
    "log_teff",
    "phase",
    "mg",
    "bp",
    "rp",
    "r_max_past",
)


@dataclass(frozen=True)
class NativeGrid:
    """MIST on a dense (feh, age, EEP) cube; NaN where an EEP does not exist.

    ``values[f, a, e, k]`` is quantity ``NATIVE_QUANTITIES[k]`` at [Fe/H] ``feh[f]``,
    log age ``log_age[a]`` and EEP ``eep[e]``.
    """

    feh: FloatArray
    log_age: FloatArray
    eep: NDArray[np.int64]
    values: NDArray[np.float32]

    def q(self, name: str) -> NDArray[np.float32]:
        return self.values[..., NATIVE_QUANTITIES.index(name)]


def _r_max_past(log_age: FloatArray, m_init: FloatArray, log_r: FloatArray) -> FloatArray:
    """Max radius (R⊙) a star of initial mass M_init has had up to each isochrone age.

    Two estimates, combined with max: (i) along the same isochrone, the running maximum of R
    over all lower EEPs. Past the main sequence, M_init changes by < 1% along an isochrone,
    so the lower-EEP points are this star's own earlier stages; this is what catches the
    RGB tip for a core-helium-burning star, which the 0.05 dex age grid steps over.
    (ii) Across ages: max over earlier isochrones ``a' < a`` of R(M_init) interpolated along
    ``a'`` (``initial_mass`` is monotonic in EEP). Shapes ``(n_age, n_eep)``.
    """
    n_age, n_eep = m_init.shape
    r = np.where(np.isfinite(log_r), 10.0 ** log_r, np.nan)
    run = np.fmax.accumulate(np.where(np.isfinite(r), r, 0.0), axis=1)
    out = np.where(np.isfinite(r), run, np.nan)
    for a2 in range(n_age):
        ok = np.isfinite(m_init[a2]) & np.isfinite(r[a2])
        if ok.sum() < 2:
            continue
        mi, rr = m_init[a2][ok], r[a2][ok]
        order = np.argsort(mi)
        mi, rr = mi[order], rr[order]
        later = slice(a2 + 1, n_age)
        q = m_init[later]
        val = np.interp(q, mi, rr, left=np.nan, right=np.nan)
        val = np.where((q >= mi[0]) & (q <= mi[-1]), val, np.nan)
        out[later] = np.fmax(out[later], val)
    return out


def build_native_grid(cfg: IsochroneMassConfig) -> NativeGrid:
    """Parse the MIST isochrone and BC files under ``cfg.mist_root`` (read in place)."""
    if cfg.mist_root is None:
        raise ValueError("isochrone_mass.mist_root is not set (host-specific; see config/host_profiles)")
    root = Path(cfg.mist_root).expanduser()
    g = cfg.grid
    bands = (g.band_g, g.band_bp, g.band_rp)
    cubes = []
    ages_ref: FloatArray | None = None
    for feh in g.feh_values:
        iso = read_mist_iso(root / g.iso_subdir / g.iso_filename_template.format(feh=_feh_token(feh, "iso")))
        bct = read_bc_table(root / g.bc_subdir / g.bc_filename_template.format(feh=_feh_token(feh, "bc")), bands)
        la = iso[:, 1]
        keep = (la >= g.log_age_min - 1e-9) & (la <= g.log_age_max + 1e-9)
        keep &= np.isin(iso[:, 8].astype(int), np.asarray(g.phases))
        iso = iso[keep]
        ages = np.unique(np.round(iso[:, 1], 4))
        if ages_ref is None:
            ages_ref = ages
        elif not np.array_equal(ages, ages_ref):
            raise ValueError(f"[Fe/H] = {feh}: isochrone ages differ from the first file")
        eep = iso[:, 0].astype(np.int64)
        ai = np.searchsorted(ages, np.round(iso[:, 1], 4))
        cube = np.full((ages.size, int(eep.max()) + 1, len(NATIVE_QUANTITIES)), np.nan)
        teff = 10.0 ** iso[:, 5]
        bc = bct.interpolate(teff, iso[:, 7])
        mbol = constants.MIST_MBOL_SUN - 2.5 * iso[:, 4]
        vals = {
            "initial_mass": iso[:, 2],
            "star_mass": iso[:, 3],
            "log_r": iso[:, 6],
            "log_g": iso[:, 7],
            "log_teff": iso[:, 5],
            "phase": iso[:, 8],
            "mg": mbol - bc[g.band_g],
            "bp": mbol - bc[g.band_bp],
            "rp": mbol - bc[g.band_rp],
        }
        for k, name in enumerate(NATIVE_QUANTITIES[:-1]):
            cube[ai, eep, k] = vals[name]
        cube[..., -1] = _r_max_past(ages, cube[..., 0], cube[..., 2])
        cubes.append(cube)
    assert ages_ref is not None
    n_eep = max(c.shape[1] for c in cubes)
    full = np.full((len(cubes), ages_ref.size, n_eep, len(NATIVE_QUANTITIES)), np.nan, dtype=np.float32)
    for f, c in enumerate(cubes):
        full[f, :, : c.shape[1]] = c
    return NativeGrid(
        feh=np.asarray(g.feh_values, float), log_age=ages_ref, eep=np.arange(n_eep, dtype=np.int64), values=full
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def native_grid_key(cfg: IsochroneMassConfig) -> str:
    """Cache key: the grid section (files, ages, phases) and the bolometric zero point."""
    blob = json.dumps({"grid": cfg.grid.model_dump(mode="json"), "mbol_sun": constants.MIST_MBOL_SUN}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_native_grid(cfg: IsochroneMassConfig, data_root: str | Path, *, rebuild: bool = False) -> NativeGrid:
    """Native grid from the cache ``<data_root>/<cache_subdir>/native_<key>.npz`` (built once).

    ``meta.yaml`` beside it records the source files' sizes and mtimes, the key, and the npz
    SHA256, which is verified on every load (a mismatch raises).
    """
    d = Path(data_root) / cfg.cache_subdir
    key = native_grid_key(cfg)
    npz = d / f"native_{key}.npz"
    meta_path = d / f"native_{key}.meta.yaml"
    if npz.exists() and meta_path.exists() and not rebuild:
        meta = yaml.safe_load(meta_path.read_text())
        got = _sha256(npz)
        if got != meta["npz_sha256"]:
            raise ValueError(f"{npz}: checksum mismatch ({got} != {meta['npz_sha256']})")
        z = np.load(npz)
        return NativeGrid(feh=z["feh"], log_age=z["log_age"], eep=z["eep"], values=z["values"])
    grid = build_native_grid(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(npz, feh=grid.feh, log_age=grid.log_age, eep=grid.eep, values=grid.values)
    root = Path(str(cfg.mist_root)).expanduser()
    g = cfg.grid
    sources = []
    for feh in g.feh_values:
        for p in (
            root / g.iso_subdir / g.iso_filename_template.format(feh=_feh_token(feh, "iso")),
            root / g.bc_subdir / g.bc_filename_template.format(feh=_feh_token(feh, "bc")),
        ):
            st = p.stat()
            sources.append({"path": str(p), "bytes": int(st.st_size), "mtime": float(st.st_mtime)})
    meta = {
        "key": key,
        "grid": cfg.grid.model_dump(mode="json"),
        "mist_mbol_sun": constants.MIST_MBOL_SUN,
        "sources": sources,
        "npz_sha256": _sha256(npz),
        "shape": list(grid.values.shape),
        "quantities": list(NATIVE_QUANTITIES),
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return grid


# ---------------------------------------------------------------------------
# Prior-weighted points (age and [Fe/H] sub-steps) and the CMD map
# ---------------------------------------------------------------------------


def imf_density(m: ArrayLike, cfg: ImfPriorConfig) -> FloatArray:
    """Continuous broken power law ξ(M) (unnormalized), zero for M ≤ 0."""
    x = np.asarray(m, float)
    edges = (0.0, *cfg.breaks_msun, np.inf)
    out = np.zeros_like(x)
    scale = 1.0
    for k, a in enumerate(cfg.alphas):
        lo, hi = edges[k], edges[k + 1]
        if k > 0:
            scale *= lo ** (-cfg.alphas[k - 1]) / lo ** (-a)
        sel = (x > lo) & (x <= hi)
        out = np.where(sel, scale * np.power(np.where(x > 0, x, 1.0), -a), out)
    return out


def _segment_widths(m: FloatArray) -> FloatArray:
    """ΔM_init per point along each isochrone row (half the distance to both neighbours).

    ``m`` has shape (..., n_eep) with NaN gaps; widths are computed over each row's finite
    points in EEP order and are 0 where M_init is not finite.
    """
    out = np.zeros_like(m)
    flat = m.reshape(-1, m.shape[-1])
    of = out.reshape(-1, m.shape[-1])
    for i in range(flat.shape[0]):
        ok = np.flatnonzero(np.isfinite(flat[i]))
        if ok.size < 2:
            continue
        v = flat[i, ok]
        w = np.empty_like(v)
        w[1:-1] = 0.5 * (v[2:] - v[:-2])
        w[0] = 0.5 * (v[1] - v[0])
        w[-1] = 0.5 * (v[-1] - v[-2])
        of[i, ok] = np.clip(w, 0.0, None)
    return out


@dataclass(frozen=True)
class PriorPoints:
    """Prior-weighted isochrone points after sub-stepping (flat arrays)."""

    weight: FloatArray
    colour: FloatArray
    mg: FloatArray
    feh: FloatArray
    log_age: FloatArray
    values: dict[str, FloatArray]


def _substep(a: NDArray[np.float32], n: int, axis: int) -> NDArray[np.float32]:
    """Linear sub-steps along ``axis``: n−1 points between neighbours, NaN if either is NaN."""
    if n <= 1:
        return a
    a = np.moveaxis(a, axis, 0)
    t = (np.arange(n) / n).astype(np.float32)
    lo, hi = a[:-1], a[1:]
    mid = lo[:, None] + t.reshape((1, n) + (1,) * (a.ndim - 1)) * (hi - lo)[:, None]
    mid = mid.reshape((-1,) + a.shape[1:])
    return np.moveaxis(np.concatenate([mid, a[-1:]], axis=0), 0, axis)


def _substep_coords(x: FloatArray, n: int) -> FloatArray:
    if n <= 1:
        return np.asarray(x, float)
    t = np.arange(n) / n
    mid = (x[:-1, None] + t[None, :] * np.diff(x)[:, None]).ravel()
    return np.concatenate([mid, x[-1:]])


def prior_points(grid: NativeGrid, cfg: IsochroneMassConfig) -> PriorPoints:
    """Sub-step the native grid in [Fe/H] and age (EEP-matched) and attach prior weights.

    Weight per point = ξ(M_init) ΔM_init × p(τ) Δτ × p([Fe/H]) Δ[Fe/H]; normalized to sum 1.
    """
    g = cfg.grid
    feh = _substep_coords(grid.feh, g.n_feh_substeps)
    lage = _substep_coords(grid.log_age, g.n_age_substeps)
    # Age weights: constant SFR (linear) or uniform in log age, times the cell width.
    lage_edges = np.concatenate([[lage[0]], 0.5 * (lage[1:] + lage[:-1]), [lage[-1]]])
    if cfg.age.kind == "uniform_linear":
        w_age = np.diff(10.0 ** lage_edges)
    else:
        w_age = np.diff(lage_edges)
    from scipy.special import ndtr

    fe_edges = np.concatenate([[feh[0]], 0.5 * (feh[1:] + feh[:-1]), [feh[-1]]])
    fp = cfg.provisional_feh_prior
    w_feh = ndtr((fe_edges[1:] - fp.mean_dex) / fp.sigma_dex) - ndtr((fe_edges[:-1] - fp.mean_dex) / fp.sigma_dex)
    out: dict[str, list[FloatArray]] = {k: [] for k in ("weight", "colour", "mg", "feh", "log_age")}
    vals: dict[str, list[FloatArray]] = {k: [] for k in NATIVE_QUANTITIES}
    # [Fe/H] sub-steps one native interval at a time (bounded memory).
    nf = grid.feh.size
    for f0 in range(nf):
        if f0 == nf - 1:
            block = grid.values[f0:f0 + 1]
            fsub = grid.feh[-1:]
        else:
            block = _substep(grid.values[f0:f0 + 2], g.n_feh_substeps, 0)[:-1]
            fsub = _substep_coords(grid.feh[f0:f0 + 2], g.n_feh_substeps)[:-1]
        cube = _substep(block, g.n_age_substeps, 1)  # (nfs, n_age_fine, n_eep, k)
        for j, fv in enumerate(fsub):
            fi = int(np.argmin(np.abs(feh - fv)))
            c = cube[j].astype(np.float64)
            mi = c[..., NATIVE_QUANTITIES.index("initial_mass")]
            ph = c[..., NATIVE_QUANTITIES.index("phase")]
            # A sub-stepped point between two phases gets the later phase's label rounding.
            dm = _segment_widths(mi)
            w = imf_density(mi, cfg.imf) * dm * w_age[:, None] * w_feh[fi]
            ok = np.isfinite(w) & (w > 0) & np.isfinite(c[..., NATIVE_QUANTITIES.index("mg")])
            ok &= np.isfinite(ph)
            a_idx = np.nonzero(ok)[0]
            out["weight"].append(w[ok])
            bp = c[..., NATIVE_QUANTITIES.index("bp")][ok]
            rp = c[..., NATIVE_QUANTITIES.index("rp")][ok]
            out["colour"].append(bp - rp)
            out["mg"].append(c[..., NATIVE_QUANTITIES.index("mg")][ok])
            out["feh"].append(np.full(int(ok.sum()), fv))
            out["log_age"].append(lage[a_idx])
            for k, name in enumerate(NATIVE_QUANTITIES):
                vals[name].append(c[..., k][ok])
    w = np.concatenate(out["weight"])
    w = w / w.sum()
    return PriorPoints(
        weight=w,
        colour=np.concatenate(out["colour"]),
        mg=np.concatenate(out["mg"]),
        feh=np.concatenate(out["feh"]),
        log_age=np.concatenate(out["log_age"]),
        values={k: np.concatenate(v) for k, v in vals.items()},
    )


@dataclass(frozen=True)
class CmdMap:
    """Prior-weighted moment maps on cell edges ``colour_edges`` × ``mag_edges``.

    ``maps[c, m, k]`` = Σ_points w × value_k for points in cell (c, m) (cloud-in-cell),
    channel order :data:`MOMENT_CHANNELS`.
    """

    colour_edges: FloatArray
    mag_edges: FloatArray
    maps: FloatArray

    @property
    def colour_centres(self) -> FloatArray:
        return 0.5 * (self.colour_edges[1:] + self.colour_edges[:-1])

    @property
    def mag_centres(self) -> FloatArray:
        return 0.5 * (self.mag_edges[1:] + self.mag_edges[:-1])


def point_channels(pts: PriorPoints) -> FloatArray:
    """(n_points, n_channels) per-point values in :data:`MOMENT_CHANNELS` order."""
    v = pts.values
    m = v["star_mass"]
    lm = np.log10(m)
    return np.column_stack([
        np.ones_like(m),
        m,
        m * m,
        lm,
        lm * lm,
        v["initial_mass"],
        v["log_r"],
        v["log_r"] ** 2,
        v["r_max_past"],
        v["log_g"],
        (v["phase"] >= 1.5).astype(float),
        pts.log_age,
        pts.feh,
    ])


def build_cmd_map(pts: PriorPoints, cfg: CmdMapConfig) -> CmdMap:
    """Deposit the weighted points into the (C0, M_G0) map with cloud-in-cell weights."""
    ce = np.arange(cfg.colour_min, cfg.colour_max + 0.5 * cfg.colour_step, cfg.colour_step)
    me = np.arange(cfg.mag_min, cfg.mag_max + 0.5 * cfg.mag_step, cfg.mag_step)
    nc, nm = ce.size - 1, me.size - 1
    ch = point_channels(pts) * pts.weight[:, None]
    # cell-centred coordinates for CIC
    xc = (pts.colour - ce[0]) / cfg.colour_step - 0.5
    xm = (pts.mg - me[0]) / cfg.mag_step - 0.5
    i0 = np.floor(xc).astype(np.int64)
    j0 = np.floor(xm).astype(np.int64)
    tc = xc - i0
    tm = xm - j0
    maps = np.zeros((nc * nm, ch.shape[1]))
    for di, wi in ((0, 1.0 - tc), (1, tc)):
        for dj, wj in ((0, 1.0 - tm), (1, tm)):
            ii = np.clip(i0 + di, 0, nc - 1)
            jj = np.clip(j0 + dj, 0, nm - 1)
            inside = (i0 + di >= 0) & (i0 + di < nc) & (j0 + dj >= 0) & (j0 + dj < nm)
            flat = ii * nm + jj
            ww = wi * wj * inside
            for k in range(ch.shape[1]):
                maps[:, k] += np.bincount(flat, weights=ch[:, k] * ww, minlength=nc * nm)
    return CmdMap(colour_edges=ce, mag_edges=me, maps=maps.reshape(nc, nm, ch.shape[1]))


# ---------------------------------------------------------------------------
# Posterior moments per star
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsochronePosterior:
    """Per-star posterior summaries (NaN where ``ok`` is False)."""

    m1_mean: FloatArray
    m1_sigma: FloatArray
    log_m1_mean: FloatArray
    log_m1_sigma: FloatArray
    m_init_mean: FloatArray
    log_r_mean: FloatArray
    log_r_sigma: FloatArray
    r_max_past_mean: FloatArray
    log_g_mean: FloatArray
    p_evolved: FloatArray
    log_age_mean: FloatArray
    feh_mean: FloatArray
    log10_evidence: FloatArray
    ok: BoolArray
    reason: NDArray[np.str_]

    def point(self, kind: Literal["mean", "log_mean"]) -> FloatArray:
        """M1 point value: posterior mean, or 10^⟨log10 M1⟩."""
        return self.m1_mean if kind == "mean" else 10.0 ** self.log_m1_mean

    def as_dict(self) -> dict[str, NDArray[Any]]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _cell_kernel(x: FloatArray, sigma: FloatArray, edges: FloatArray) -> FloatArray:
    """Probability of N(x, σ) in each cell of ``edges``: shape (n, len(edges) − 1)."""
    from scipy.special import ndtr

    z = (edges[None, :] - x[:, None]) / sigma[:, None]
    c = ndtr(z)
    return np.diff(c, axis=1)


def likelihood_sigmas(
    sigma_mu: ArrayLike,
    ebv: ArrayLike,
    a_g_per_ebv: ArrayLike,
    e_bp_rp_per_ebv: ArrayLike,
    lk: IsochroneLikelihoodConfig,
) -> tuple[FloatArray, FloatArray]:
    """(σ_C, σ_M) per star (see :class:`IsochroneLikelihoodConfig`); NaN σ_μ counts as 0."""
    smu = np.asarray(sigma_mu, float)
    smu = np.where(np.isfinite(smu), smu, 0.0)
    se = lk.provisional_ebv_fractional_sigma * np.nan_to_num(np.asarray(ebv, float))
    sa = se * np.nan_to_num(np.asarray(a_g_per_ebv, float))
    sc_e = se * np.nan_to_num(np.asarray(e_bp_rp_per_ebv, float))
    sig_c = np.sqrt(lk.provisional_colour_floor_mag**2 + sc_e**2)
    sig_m = np.sqrt(smu**2 + lk.provisional_mag_floor_mag**2 + sa**2)
    return sig_c, sig_m


def posterior_moments(
    colour0: ArrayLike,
    mg0: ArrayLike,
    sigma_colour: ArrayLike,
    sigma_mag: ArrayLike,
    cmap: CmdMap,
    lk: IsochroneLikelihoodConfig,
) -> IsochronePosterior:
    """Posterior moments for every star (vectorized over chunks of colour-sorted stars).

    Rows with non-finite inputs get reason ``no_cmd``; rows whose prior-predictive density
    ``Σ w K / (Δc Δm)`` is below ``10^min_log10_evidence`` get ``off_grid``. Both are NaN.
    """
    c = np.asarray(colour0, float)
    m = np.asarray(mg0, float)
    sc = np.broadcast_to(np.asarray(sigma_colour, float), c.shape).astype(float)
    sm = np.broadcast_to(np.asarray(sigma_mag, float), c.shape).astype(float)
    n = c.size
    nk = len(MOMENT_CHANNELS)
    s = np.full((n, nk), np.nan)
    valid = np.isfinite(c) & np.isfinite(m) & np.isfinite(sc) & np.isfinite(sm) & (sc > 0) & (sm > 0)
    idx = np.flatnonzero(valid)
    idx = idx[np.argsort(c[idx], kind="stable")]
    ce, me = cmap.colour_edges, cmap.mag_edges
    nm = me.size - 1
    kn = lk.kernel_n_sigma
    for a in range(0, idx.size, lk.chunk_rows):
        ii = idx[a:a + lk.chunk_rows]
        clo = float(np.min(c[ii] - kn * sc[ii]))
        chi = float(np.max(c[ii] + kn * sc[ii]))
        i0 = int(np.clip(np.searchsorted(ce, clo) - 1, 0, ce.size - 2))
        i1 = int(np.clip(np.searchsorted(ce, chi) + 1, i0 + 1, ce.size - 1))
        mlo = float(np.min(m[ii] - kn * sm[ii]))
        mhi = float(np.max(m[ii] + kn * sm[ii]))
        j0 = int(np.clip(np.searchsorted(me, mlo) - 1, 0, me.size - 2))
        j1 = int(np.clip(np.searchsorted(me, mhi) + 1, j0 + 1, me.size - 1))
        kc = _cell_kernel(c[ii], sc[ii], ce[i0:i1 + 1])  # (n, nc_w)
        km = _cell_kernel(m[ii], sm[ii], me[j0:j1 + 1])  # (n, nm_w)
        sub = cmap.maps[i0:i1, j0:j1, :]  # (nc_w, nm_w, k)
        t = kc @ sub.reshape(sub.shape[0], -1)  # (n, nm_w * k)
        t = t.reshape(ii.size, j1 - j0, nk)
        s[ii] = np.einsum("nm,nmk->nk", km, t)
    one = s[:, 0]
    area = float(np.mean(np.diff(ce)) * np.mean(np.diff(me)))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ev = np.log10(one / area)
        mom = s / one[:, None]
    reason = np.full(n, "ok", dtype="<U8")
    reason[~valid] = "no_cmd"
    off = valid & ~(log_ev >= lk.min_log10_evidence)
    reason[off] = "off_grid"
    ok = reason == "ok"

    def col(name: str) -> FloatArray:
        return np.where(ok, mom[:, MOMENT_CHANNELS.index(name)], np.nan)

    m1 = col("mass")
    lm = col("log_mass")
    return IsochronePosterior(
        m1_mean=m1,
        m1_sigma=np.sqrt(np.clip(col("mass2") - m1**2, 0.0, None)),
        log_m1_mean=lm,
        log_m1_sigma=np.sqrt(np.clip(col("log_mass2") - lm**2, 0.0, None)),
        m_init_mean=col("initial_mass"),
        log_r_mean=col("log_r"),
        log_r_sigma=np.sqrt(np.clip(col("log_r2") - col("log_r") ** 2, 0.0, None)),
        r_max_past_mean=col("r_max_past"),
        log_g_mean=col("log_g"),
        p_evolved=col("evolved"),
        log_age_mean=col("log_age"),
        feh_mean=col("feh"),
        log10_evidence=np.where(valid, log_ev, np.nan),
        ok=ok,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Convenience: one object holding the map (built once per config)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IsochroneMassModel:
    """The prior-weighted CMD map plus the config that built it."""

    cfg: IsochroneMassConfig
    cmap: CmdMap
    grid_key: str

    def fit(
        self,
        colour0: ArrayLike,
        mg0: ArrayLike,
        *,
        sigma_mu: ArrayLike,
        ebv: ArrayLike | None = None,
        a_g: ArrayLike | None = None,
        e_bp_rp: ArrayLike | None = None,
    ) -> IsochronePosterior:
        """Posterior for dereddened (C0, M_G0) with distance-modulus σ and optional E(B−V).

        ``a_g`` / ``e_bp_rp`` (with ``ebv``) give the per-star extinction vector used to project
        the fractional E(B−V) error; without them the extinction term is zero.
        """
        c0 = np.asarray(colour0, float)
        if ebv is None or a_g is None or e_bp_rp is None:
            zeros = np.zeros_like(c0)
            sc, sm = likelihood_sigmas(sigma_mu, zeros, zeros, zeros, self.cfg.likelihood)
        else:
            e = np.asarray(ebv, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ka = np.where(e > 0, np.asarray(a_g, float) / e, 0.0)
                ke = np.where(e > 0, np.asarray(e_bp_rp, float) / e, 0.0)
            sc, sm = likelihood_sigmas(sigma_mu, e, ka, ke, self.cfg.likelihood)
        return posterior_moments(c0, mg0, sc, sm, self.cmap, self.cfg.likelihood)


def build_model(cfg: IsochroneMassConfig, data_root: str | Path) -> IsochroneMassModel:
    """Load (or build and cache) the native grid, sub-step it, weight it, deposit the map."""
    grid = load_native_grid(cfg, data_root)
    pts = prior_points(grid, cfg)
    return IsochroneMassModel(cfg=cfg, cmap=build_cmd_map(pts, cfg.cmd_map), grid_key=native_grid_key(cfg))


def config_key(cfg: IsochroneMassConfig) -> str:
    """Short hash of the whole section except ``mist_root`` (host path), for result caches."""
    d = cfg.model_dump(mode="json")
    d.pop("mist_root", None)
    blob = json.dumps({"cfg": d, "mbol_sun": constants.MIST_MBOL_SUN, "schema": RESULT_SCHEMA_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def main_sequence_colour_relation(
    grid: NativeGrid, feh: float, log_age: float
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """MIST main-sequence (phase 0) photometry vs current mass at the nearest grid (feh, age).

    Returns ``(mass, {"mg", "bp_g", "g_rp", "colour"})`` sorted by mass; used for companion
    colours by :mod:`darkhunter_pop.malmquist_cmd`.
    """
    f = int(np.argmin(np.abs(grid.feh - feh)))
    a = int(np.argmin(np.abs(grid.log_age - log_age)))
    v = grid.values[f, a].astype(float)
    ph = v[:, NATIVE_QUANTITIES.index("phase")]
    ok = np.isfinite(ph) & (ph < 0.5)
    mass = v[ok, NATIVE_QUANTITIES.index("star_mass")]
    mg = v[ok, NATIVE_QUANTITIES.index("mg")]
    bp = v[ok, NATIVE_QUANTITIES.index("bp")]
    rp = v[ok, NATIVE_QUANTITIES.index("rp")]
    order = np.argsort(mass)
    return mass[order], {"mg": mg[order], "bp_g": (bp - mg)[order], "g_rp": (mg - rp)[order], "colour": (bp - rp)[order]}


def summary_for_report(post: IsochronePosterior) -> Mapping[str, Any]:
    """Counts by reason and M1 percentiles (for reports)."""
    reasons, counts = np.unique(post.reason, return_counts=True)
    m = post.m1_mean[post.ok]
    return {
        "rows": int(post.ok.size),
        "reasons": {str(r): int(k) for r, k in zip(reasons, counts)},
        "m1_percentiles_1_5_25_50_75_95_99": [float(x) for x in np.percentile(m, [1, 5, 25, 50, 75, 95, 99])] if m.size else [],
        "median_sigma_log_m1": float(np.nanmedian(post.log_m1_sigma)) if m.size else math.nan,
    }
