"""Luminous-generation orbit covariances by deterministic replay (#704, option B; Ryan 2026-11-14).

The luminous proposal-set generations (``single_gen53``-``59``) store no ``outcome/orbit_p0`` / ``orbit_cov``, so the
per-sample cuts that need the 12x12 cascade covariance (Andrews ATF, El-Badry 2026 sigma_M~2) cannot run on their
mock rows. Option B: replay, from the stored seeds and under the current st2 config, only the accepted luminous
orbital draws that can reach a published sample, and store the covariance exactly as the compact-object path does
(``proposal_set.simulate_one`` -> ``compact_objects.orbit_covariance_from_epochs``, same Jacobian step).

Terminology: luminous star (M1) / dark companion (M2); "real DR3" is the archive, "mock" a draw.

This module holds

* the config (``config/population/luminous_covariance_704.yaml``) and the check that every pre-filter threshold is
  strictly looser than the frozen selection-file cut it stands for (:func:`check_prefilter_looser`);
* the mock-only pre-filter (:func:`evaluate_prefilter`), defined on quantities computable without the covariance:
  the point M2 at the Andrews pass-1 M1 (``andrews2022_atf`` root code), the El-Badry 2026 evaluator's own Janssens
  M~2 (dereddened as for a real row), period, G and the routed solution type;
* the covariance sidecar ``<artifact stem>.orbit_cov.h5`` (write / read / merge into ``outcome``), keyed by
  ``draw_index`` and tied to its generation artifact by ``primary_model``, ``fragment_fingerprint``, the draw count
  and the accepted-draw set (any mismatch refuses).

A merged ``outcome`` carries ``orbit_p0`` (n, 12), ``orbit_cov`` (n, 12, 12) (NaN where not replayed) and
``orbit_cov_status``: ``""`` (not an accepted orbit), :data:`STATUS_REPLAYED`, or
``prefilter_excluded:<version>`` (:func:`excluded_status`). ``mock_selection_rows`` turns the latter into a
not-applicable mark with reason ``mock_prefilter_excluded:<version>`` (:func:`excluded_reason`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

#: Default config (repo-relative).
DEFAULT_CONFIG_PATH: Final[str] = "config/population/luminous_covariance_704.yaml"
#: ``orbit_cov_status`` of a replayed draw (its covariance may still be NaN: singular J^T J, counted, never diagonal).
STATUS_REPLAYED: Final[str] = "replayed"
_EXCLUDED_PREFIX: Final[str] = "prefilter_excluded:"
#: Number of orbit parameters (``proposal_set.ORBIT_PARAMETER_NAMES``).
N_ORBIT: Final[int] = 12
#: El-Badry 2026 subsamples whose chains read the Janssens M~2 (the elbadry2026 pre-filter branch covers them).
ELBADRY2026_M2_TILDE_SUBSAMPLES: Final[tuple[str, ...]] = ("primary_ns_bh", "sub_chandrasekhar")


def excluded_status(version: str) -> str:
    """``orbit_cov_status`` of an accepted draw outside pre-filter ``version``."""
    return _EXCLUDED_PREFIX + str(version)


def is_excluded_status(status: Any) -> bool:
    return isinstance(status, str) and status.startswith(_EXCLUDED_PREFIX)


def excluded_reason(status: str) -> str:
    """Not-applicable reason the evaluator records for an excluded mock row (``mock_prefilter_excluded:<version>``)."""
    if not is_excluded_status(status):
        raise ValueError(f"{status!r} is not a prefilter-excluded status")
    return "mock_" + status


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AndrewsPrefilterConfig(_Strict):
    """Andrews branch: point M2 at the frozen pass-1 M1 >= ``m2_point_min_msun`` (looser than the 1.4 threshold)."""

    selection: str
    m2_point_min_msun: float = Field(..., gt=0.0)


class ElBadry2026PrefilterConfig(_Strict):
    """El-Badry 2026 branch: Janssens M~2 >= ``m2_tilde_min_msun``, P < ``period_days_max``, G < ``g_mag_faint_limit``."""

    selection: str
    solution_types: tuple[str, ...] = Field(..., min_length=1)
    m2_tilde_min_msun: float = Field(..., gt=0.0)
    period_days_max: float = Field(..., gt=0.0)
    g_mag_faint_limit: float


class PrefilterConfig(_Strict):
    version: str = Field(..., pattern=r"^lcf[0-9]+$")
    andrews: AndrewsPrefilterConfig
    elbadry2026: ElBadry2026PrefilterConfig


class PrefilterValidationConfig(_Strict):
    generations: tuple[int, ...] = Field(..., min_length=1)
    max_outside_filter_passing: int = Field(..., ge=0)


class LuminousCovarianceConfig(_Strict):
    """``config/population/luminous_covariance_704.yaml`` (header: the definitions and their justification)."""

    schema_version: int = Field(..., ge=1)
    provenance: str = Field(..., min_length=1)
    covariance_jacobian_step: float = Field(..., gt=0.0)
    prefilter: PrefilterConfig
    validation: PrefilterValidationConfig


def load_luminous_covariance_config(path: str | Path = DEFAULT_CONFIG_PATH) -> LuminousCovarianceConfig:
    p = Path(path)
    if not p.is_absolute() and not p.is_file():
        from darkhunter_pop.config_loader import repo_root

        p = repo_root() / p
    return LuminousCovarianceConfig.model_validate(yaml.safe_load(p.read_text()))


def config_json(cfg: LuminousCovarianceConfig) -> str:
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True)


# ---------------------------------------------------------------------------
# Looser-than-frozen check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenBounds:
    """The frozen-file cut values each pre-filter threshold loosens (read, never hardcoded)."""

    andrews_m2_threshold_msun: float
    andrews_pass1_m1_msun: float
    andrews_solution_types: tuple[str, ...]
    elbadry2026_m2_tilde_min_msun: float
    elbadry2026_period_days_max: float
    elbadry2026_g_mag_faint_limit: float
    elbadry2026_solution_types: tuple[str, ...]


def _cut_param(cut: Any, key: str) -> float | None:
    v = (cut.parameters or {}).get(key)
    return None if v is None else float(v)


def frozen_bounds(registry: Any, cfg: LuminousCovarianceConfig) -> FrozenBounds:
    """Read the frozen bounds from the selection files ``cfg`` names (forward_model cut lists)."""
    from darkhunter_pop.andrews2022_atf import m2_threshold_msun, require_procedure
    from darkhunter_pop.config_schema import SampleSelectionMode
    from darkhunter_pop.sample_selection import parent_query_for_mode

    fm = SampleSelectionMode.FORWARD_MODEL
    a_spec = registry.resolved(cfg.prefilter.andrews.selection)
    proc = require_procedure(a_spec)
    a_types = tuple(parent_query_for_mode(a_spec, registry.dr_mode).solution_types)
    e_spec = registry.resolved(cfg.prefilter.elbadry2026.selection)
    m2_min: list[float] = []
    p_max: list[float] = []
    g_max: list[float] = []
    e_types: set[str] = set()
    found: set[str] = set()
    for branch in e_spec.branches or []:
        for sub in branch.subsamples or []:
            if sub.id not in ELBADRY2026_M2_TILDE_SUBSAMPLES:
                continue
            found.add(sub.id)
            e_types.update(parent_query_for_mode(e_spec, registry.dr_mode, branch_id=branch.id).solution_types)
            for cut in sub.cuts:
                if fm not in cut.applies_to:
                    continue
                if "m2_tilde_msun" in cut.expression and _cut_param(cut, "m2_msun_min") is not None:
                    m2_min.append(float(_cut_param(cut, "m2_msun_min")))  # type: ignore[arg-type]
                if _cut_param(cut, "period_days_max") is not None:
                    p_max.append(float(_cut_param(cut, "period_days_max")))  # type: ignore[arg-type]
                if _cut_param(cut, "g_mag_faint_limit") is not None:
                    g_max.append(float(_cut_param(cut, "g_mag_faint_limit")))  # type: ignore[arg-type]
    if found != set(ELBADRY2026_M2_TILDE_SUBSAMPLES) or not (m2_min and p_max and g_max):
        raise ValueError(f"{e_spec.name}: expected M~2 / period / G cuts in subsamples {ELBADRY2026_M2_TILDE_SUBSAMPLES}")
    # every M~2 subsample must carry each bound, else the branch is not a necessary condition
    for sub_id in ELBADRY2026_M2_TILDE_SUBSAMPLES:
        sub = next(s for b in e_spec.branches or [] for s in b.subsamples or [] if s.id == sub_id)
        cuts = [c for c in sub.cuts if fm in c.applies_to]
        for key in ("m2_msun_min", "period_days_max", "g_mag_faint_limit"):
            if not any(_cut_param(c, key) is not None for c in cuts):
                raise ValueError(f"{e_spec.name}:{sub_id} has no forward_model {key} cut; the pre-filter would not be necessary")
    return FrozenBounds(
        andrews_m2_threshold_msun=float(m2_threshold_msun(a_spec)),
        andrews_pass1_m1_msun=float(proc.pass1.m1_msun),
        andrews_solution_types=a_types,
        elbadry2026_m2_tilde_min_msun=min(m2_min),
        elbadry2026_period_days_max=max(p_max),
        elbadry2026_g_mag_faint_limit=max(g_max),
        elbadry2026_solution_types=tuple(sorted(e_types)),
    )


def check_prefilter_looser(registry: Any, cfg: LuminousCovarianceConfig) -> list[str]:
    """Violations (empty = every threshold strictly looser than the frozen cut, every frozen type covered)."""
    b = frozen_bounds(registry, cfg)
    a, e = cfg.prefilter.andrews, cfg.prefilter.elbadry2026
    bad: list[str] = []
    if not a.m2_point_min_msun < b.andrews_m2_threshold_msun:
        bad.append(f"andrews m2_point_min_msun {a.m2_point_min_msun} !< frozen {b.andrews_m2_threshold_msun}")
    if not e.m2_tilde_min_msun < b.elbadry2026_m2_tilde_min_msun:
        bad.append(f"elbadry2026 m2_tilde_min_msun {e.m2_tilde_min_msun} !< frozen {b.elbadry2026_m2_tilde_min_msun}")
    if not e.period_days_max > b.elbadry2026_period_days_max:
        bad.append(f"elbadry2026 period_days_max {e.period_days_max} !> frozen {b.elbadry2026_period_days_max}")
    if not e.g_mag_faint_limit > b.elbadry2026_g_mag_faint_limit:
        bad.append(f"elbadry2026 g_mag_faint_limit {e.g_mag_faint_limit} !> frozen {b.elbadry2026_g_mag_faint_limit}")
    missing = set(b.elbadry2026_solution_types) - set(e.solution_types)
    if missing:
        bad.append(f"elbadry2026 solution_types miss frozen parent types {sorted(missing)}")
    return bad


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefilterResult:
    """Per-row pre-filter quantities and decisions (row order of the input)."""

    source_id: NDArray[np.int64]
    andrews_m2_point_msun: NDArray[np.float64]
    m2_tilde_msun: NDArray[np.float64]
    pass_andrews: NDArray[np.bool_]
    pass_elbadry2026: NDArray[np.bool_]

    @property
    def pass_any(self) -> NDArray[np.bool_]:
        return self.pass_andrews | self.pass_elbadry2026


def _col(rows: Sequence[Mapping[str, Any]], key: str) -> NDArray[np.float64]:
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(key)
        try:
            out[i] = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return out


def andrews_point_m2(rows: Sequence[Mapping[str, Any]], proc: Any) -> NDArray[np.float64]:
    """Point M2 (f = 0) of each row's fitted orbit at the ATF pass-1 M1, through the ATF code itself.

    One "draw" per row at the point values (``notebook_mass_function``, ``solve_m2_fixed_m1`` with the frozen pass-1
    M1 and root bracket). NaN where the root does not exist (non-finite / non-positive m_f, e.g. parallax <= 0): pass 1
    then rejects the real draws too (any-draw failure), so such a row cannot pass Andrews.
    """
    from darkhunter_pop.andrews2022_atf import notebook_mass_function, solve_m2_fixed_m1

    order = list(proc.covariance.parameter_order)
    samples = np.full((len(rows), len(order)), np.nan)
    for name in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes", "parallax", "period"):
        samples[:, order.index(name)] = _col(rows, name)
    m_f, _ = notebook_mass_function(samples, order)
    m2, ok = solve_m2_fixed_m1(m_f, m1_msun=float(proc.pass1.m1_msun), bracket_msun=proc.pass1.root_bracket_msun)
    return np.where(ok, m2, np.nan)


def elbadry2026_m2_tilde(rows: Sequence[Mapping[str, Any]], selection: Any) -> NDArray[np.float64]:
    """The El-Badry 2026 evaluator's own Janssens M~2 per row (its enrichment, dereddening included); NaN if undefined."""
    if not rows:
        return np.zeros(0)
    enriched = selection._enrich_rows_for_spec([dict(r) for r in rows])
    by_id = {int(r["source_id"]): r.get("m2_tilde_msun") for r in enriched}
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = by_id.get(int(r["source_id"]))
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[i] = float(v)
    return out


def evaluate_prefilter(
    rows: Sequence[Mapping[str, Any]],
    cfg: LuminousCovarianceConfig,
    *,
    registry: Any,
    extinction_lookup: Any | None = None,
) -> PrefilterResult:
    """Pre-filter decisions for mock rows (``mock_selection_rows.mock_draws_to_selection_rows``; no covariance read).

    Refuses (``ValueError``) unless every threshold is strictly looser than its frozen cut. ``extinction_lookup``
    defaults to the El-Badry 2026 builder (needs the dust maps); tests pass a stub.
    """
    from darkhunter_pop.andrews2022_atf import require_procedure
    from darkhunter_pop.mock_selection_rows import forward_model_selection

    bad = check_prefilter_looser(registry, cfg)
    if bad:
        raise ValueError("pre-filter not looser than the frozen cuts: " + "; ".join(bad))
    b = frozen_bounds(registry, cfg)
    pf = cfg.prefilter
    sid = np.array([int(r["source_id"]) for r in rows], dtype=np.int64)
    sol = np.array([str(r.get("nss_solution_type", "")) for r in rows])
    proc = require_procedure(registry.resolved(pf.andrews.selection))
    m2_pt = andrews_point_m2(rows, proc)
    with np.errstate(invalid="ignore"):
        pass_a = np.isin(sol, list(b.andrews_solution_types)) & (m2_pt >= pf.andrews.m2_point_min_msun)
    e = pf.elbadry2026
    g = _col(rows, "phot_g_mean_mag")
    per = _col(rows, "period_day")
    with np.errstate(invalid="ignore"):
        cheap = np.isin(sol, list(e.solution_types)) & (g < e.g_mag_faint_limit) & (per < e.period_days_max)
    m2_t = np.full(len(rows), np.nan)
    idx = np.flatnonzero(cheap)
    if idx.size:
        sel = forward_model_selection(registry, e.selection, extinction_lookup=extinction_lookup)
        m2_t[idx] = elbadry2026_m2_tilde([rows[i] for i in idx], sel)
    with np.errstate(invalid="ignore"):
        pass_e = cheap & (m2_t >= e.m2_tilde_min_msun)
    return PrefilterResult(source_id=sid, andrews_m2_point_msun=m2_pt, m2_tilde_msun=m2_t,
                           pass_andrews=pass_a, pass_elbadry2026=pass_e)


# ---------------------------------------------------------------------------
# Covariance sidecar
# ---------------------------------------------------------------------------


def orbit_cov_sidecar_path(artifact: str | Path) -> Path:
    """``<artifact stem>.orbit_cov.h5`` beside the generation artifact."""
    p = Path(artifact)
    return p.with_name(p.stem + ".orbit_cov.h5")


@dataclass(frozen=True)
class OrbitCovSidecar:
    """Replayed covariances of one generation (``draw_index`` order of the replayed set) + the pre-filter record."""

    accepted_draw_index: NDArray[np.int64]
    prefilter_pass: NDArray[np.bool_]
    draw_index: NDArray[np.int64]
    orbit_p0: NDArray[np.float64]
    orbit_cov: NDArray[np.float64]
    orbit_cov_sigma_check: NDArray[np.float64]
    orbit_cov_cpu_seconds: NDArray[np.float64]
    attrs: Mapping[str, Any]

    @property
    def prefilter_version(self) -> str:
        return str(self.attrs["prefilter_version"])


#: Artifact attributes a sidecar records and must match on merge.
SIDECAR_IDENTITY_ATTRS: Final[tuple[str, ...]] = ("primary_model", "fragment_fingerprint")


def write_orbit_cov_sidecar(path: str | Path, sidecar: OrbitCovSidecar) -> Path:
    """Write ``sidecar`` to ``path`` via ``<path>.tmp`` then rename; refuses an existing ``path`` (never overwrites)."""
    import h5py

    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} exists; covariance sidecars are never overwritten")
    _check_sidecar_shapes(sidecar)
    for key in (*SIDECAR_IDENTITY_ATTRS, "n_draws", "prefilter_version", "artifact"):
        if key not in sidecar.attrs:
            raise ValueError(f"sidecar attrs need {key!r}")
    tmp = p.with_name(p.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h:
        h.create_dataset("accepted_draw_index", data=np.asarray(sidecar.accepted_draw_index, np.int64))
        h.create_dataset("prefilter_pass", data=np.asarray(sidecar.prefilter_pass, bool))
        h.create_dataset("draw_index", data=np.asarray(sidecar.draw_index, np.int64))
        h.create_dataset("orbit_p0", data=np.asarray(sidecar.orbit_p0, np.float64))
        h.create_dataset("orbit_cov", data=np.asarray(sidecar.orbit_cov, np.float64))
        h.create_dataset("orbit_cov_sigma_check", data=np.asarray(sidecar.orbit_cov_sigma_check, np.float64))
        h.create_dataset("orbit_cov_cpu_seconds", data=np.asarray(sidecar.orbit_cov_cpu_seconds, np.float64))
        for k, v in sidecar.attrs.items():
            h.attrs[k] = v if isinstance(v, (str, int, float, bool, np.integer, np.floating)) else json.dumps(v, sort_keys=True, default=str)
    tmp.rename(p)
    return p


def read_orbit_cov_sidecar(path: str | Path) -> OrbitCovSidecar:
    import h5py

    with h5py.File(Path(path), "r") as h:
        sc = OrbitCovSidecar(
            accepted_draw_index=h["accepted_draw_index"][()], prefilter_pass=h["prefilter_pass"][()].astype(bool),
            draw_index=h["draw_index"][()], orbit_p0=h["orbit_p0"][()], orbit_cov=h["orbit_cov"][()],
            orbit_cov_sigma_check=h["orbit_cov_sigma_check"][()], orbit_cov_cpu_seconds=h["orbit_cov_cpu_seconds"][()],
            attrs={k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()},
        )
    _check_sidecar_shapes(sc)
    return sc


def _check_sidecar_shapes(sc: OrbitCovSidecar) -> None:
    na, nr = np.asarray(sc.accepted_draw_index).size, np.asarray(sc.draw_index).size
    if np.asarray(sc.prefilter_pass).shape != (na,):
        raise ValueError("prefilter_pass must align with accepted_draw_index")
    if np.asarray(sc.orbit_p0).shape != (nr, N_ORBIT) or np.asarray(sc.orbit_cov).shape != (nr, N_ORBIT, N_ORBIT):
        raise ValueError(f"orbit_p0 / orbit_cov must be ({nr}, {N_ORBIT}) / ({nr}, {N_ORBIT}, {N_ORBIT})")
    if np.asarray(sc.orbit_cov_sigma_check).shape != (nr,) or np.asarray(sc.orbit_cov_cpu_seconds).shape != (nr,):
        raise ValueError("per-draw scalars must align with draw_index")
    acc = np.asarray(sc.accepted_draw_index, np.int64)
    want = np.sort(acc[np.asarray(sc.prefilter_pass, bool)])
    if not np.array_equal(np.sort(np.asarray(sc.draw_index, np.int64)), want):
        raise ValueError("replayed draw_index must be exactly the accepted draws that pass the pre-filter")


def merge_orbit_cov_sidecar(
    truth: Mapping[str, NDArray[Any]], outcome: dict[str, NDArray[Any]], attrs: Mapping[str, Any], sidecar: OrbitCovSidecar,
) -> None:
    """Fill ``outcome`` orbit_p0 / orbit_cov / orbit_cov_status from ``sidecar`` in place (module docstring).

    Refuses (``ValueError``) when the sidecar was not built from this artifact: identity attributes, draw count, or the
    accepted-orbital draw set differ.
    """
    if "orbit_cov" in outcome:
        raise ValueError("outcome already carries orbit_cov (compact-object generation); no sidecar merge")
    if bool(sidecar.attrs.get("smoke", False)):
        raise ValueError("smoke sidecar (partial replay) is not a product; never merged")
    for key in SIDECAR_IDENTITY_ATTRS:
        if str(sidecar.attrs.get(key)) != str(attrs.get(key)):
            raise ValueError(f"sidecar {key} {sidecar.attrs.get(key)!r} != artifact {attrs.get(key)!r}")
    di = np.asarray(truth["draw_index"], np.int64)
    if int(sidecar.attrs["n_draws"]) != di.size:
        raise ValueError(f"sidecar n_draws {sidecar.attrs['n_draws']} != artifact {di.size}")
    acc = np.asarray(outcome["accepted_orbital"], bool)
    if not np.array_equal(np.sort(di[acc]), np.sort(np.asarray(sidecar.accepted_draw_index, np.int64))):
        raise ValueError("sidecar accepted-draw set differs from the artifact's accepted_orbital draws")
    pos = {int(d): k for k, d in enumerate(di)}
    n = di.size
    p0 = np.full((n, N_ORBIT), np.nan)
    cov = np.full((n, N_ORBIT, N_ORBIT), np.nan)
    status = np.full(n, "", dtype=object)
    ex = excluded_status(sidecar.prefilter_version)
    for d, ok in zip(np.asarray(sidecar.accepted_draw_index, np.int64), np.asarray(sidecar.prefilter_pass, bool), strict=True):
        status[pos[int(d)]] = STATUS_REPLAYED if ok else ex
    rows = np.array([pos[int(d)] for d in np.asarray(sidecar.draw_index, np.int64)], dtype=np.int64)
    if rows.size:
        p0[rows] = np.asarray(sidecar.orbit_p0, np.float64)
        cov[rows] = np.asarray(sidecar.orbit_cov, np.float64)
    outcome["orbit_p0"] = p0
    outcome["orbit_cov"] = cov
    outcome["orbit_cov_status"] = status.astype(str)


def sidecar_from_records(
    accepted_draw_index: ArrayLike, prefilter_pass: ArrayLike, records: Mapping[int, Mapping[str, Any]], attrs: Mapping[str, Any],
) -> OrbitCovSidecar:
    """Assemble a sidecar from ``simulate_one`` records keyed by draw_index (replayed draws, any order)."""
    d = np.array(sorted(int(k) for k in records), dtype=np.int64)
    rec = [records[int(k)] for k in d]
    return OrbitCovSidecar(
        accepted_draw_index=np.asarray(accepted_draw_index, np.int64), prefilter_pass=np.asarray(prefilter_pass, bool),
        draw_index=d,
        orbit_p0=np.array([np.asarray(r["orbit_p0"], float) for r in rec]).reshape(d.size, N_ORBIT),
        orbit_cov=np.array([np.asarray(r["orbit_cov"], float).reshape(N_ORBIT, N_ORBIT) for r in rec]).reshape(d.size, N_ORBIT, N_ORBIT),
        orbit_cov_sigma_check=np.array([float(r["orbit_cov_sigma_check"]) for r in rec]),
        orbit_cov_cpu_seconds=np.array([float(r["orbit_cov_cpu_seconds"]) for r in rec]),
        attrs=dict(attrs),
    )


__all__ = [
    "DEFAULT_CONFIG_PATH", "STATUS_REPLAYED", "LuminousCovarianceConfig", "OrbitCovSidecar", "PrefilterResult",
    "andrews_point_m2", "check_prefilter_looser", "elbadry2026_m2_tilde", "evaluate_prefilter", "excluded_reason",
    "excluded_status", "frozen_bounds", "is_excluded_status", "load_luminous_covariance_config",
    "merge_orbit_cov_sidecar", "orbit_cov_sidecar_path", "read_orbit_cov_sidecar", "sidecar_from_records",
    "write_orbit_cov_sidecar",
]
