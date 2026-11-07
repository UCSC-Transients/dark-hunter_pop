"""Rung-5 mass-bin edges and merge map, fixed from the mock before any real count is looked at (R5-T1, #614).

``docs/RUNG5_SPEC.md`` §2.3: per-class true-mass bins (WD, dark) carry the non-parametric dN/dlog M2 heights; the
apparent-mass axis M2_app (the f = 0 Student-t posterior median of :func:`compact_objects.recover_masses`, applied
identically to mock draws and real rows) is the observable axis of the response matrix. Both are proposed in config and
then merged greedily, left to right, until every merged cell holds at least ``min_cell_ess`` Kish ESS of detected mock
draws (training partition only). The result is written back to the config with a digest; the rung-5 fit refuses a
bins config whose digest does not match (:func:`assert_frozen`).

Nothing here reads real data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop import constants
from darkhunter_pop import proposal_set as ps

CLASSES: tuple[str, ...] = ("wd", "dark")


def _increasing(edges: Sequence[float], name: str) -> None:
    e = np.asarray(edges, float)
    if e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError(f"{name}: edges must be finite and strictly increasing, got {list(edges)}")


class Rung5Derived(BaseModel):
    """What the freeze script derived from the mock (empty until frozen)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_m2_edges_msun: dict[str, tuple[float, ...]]
    m2_app_edges_msun: tuple[float, ...]
    true_bin_ess_detected: dict[str, tuple[float, ...]]
    m2_app_bin_ess_detected: tuple[float, ...]
    artifacts_sha256: dict[str, str]
    primary_model: str
    partition: Literal["train"]
    real_data_read: Literal[False]
    frozen_at: str
    #: sha256 of the canonical JSON of (proposed block, derived edges) — see :func:`freeze_digest`.
    digest: str


class Rung5BinsConfig(BaseModel):
    """``config/population/rung5_bins_<dr>.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_release: Literal["dr3", "dr4"]
    co_version: str = Field(..., pattern=r"^co_v[0-9]+$")
    #: noise/epoch label every mock artifact's primary-model version must contain.
    required_noise_model: str
    rung4_config: str
    #: proposed true-mass edges per class (Msun); the WD top edge must equal M_Ch.
    true_m2_edges_msun: dict[str, tuple[float, ...]]
    #: proposed M2_app edges (Msun); the last edge closes an overflow bin.
    m2_app_edges_msun: tuple[float, ...]
    #: detection used for the true-mass merge (the M2_app merge uses accepted C1 orbits).
    detection: Literal["orbit_or_acceleration", "orbit", "acceleration"]
    c1_period_max_days: float = Field(..., gt=0)
    min_cell_ess: float = Field(..., gt=0)
    derived: Rung5Derived | None = None

    @model_validator(mode="after")
    def _edges(self) -> Rung5BinsConfig:
        if set(self.true_m2_edges_msun) != set(CLASSES):
            raise ValueError(f"true_m2_edges_msun needs exactly the classes {CLASSES}")
        for k, e in self.true_m2_edges_msun.items():
            _increasing(e, f"true_m2_edges_msun.{k}")
            if e[0] <= 0:
                raise ValueError(f"true_m2_edges_msun.{k}: masses must be positive")
        m_ch = float(constants.M_CH.to_value("Msun"))
        if not np.isclose(self.true_m2_edges_msun["wd"][-1], m_ch):
            raise ValueError(f"the WD top edge must be M_Ch = {m_ch} Msun (hard truncation)")
        _increasing(self.m2_app_edges_msun, "m2_app_edges_msun")
        if self.m2_app_edges_msun[0] < 0:
            raise ValueError("m2_app_edges_msun must start at >= 0")
        if self.derived is not None:
            for k, e in self.derived.true_m2_edges_msun.items():
                if not set(e) <= set(self.true_m2_edges_msun[k]) or e[0] != self.true_m2_edges_msun[k][0] or e[-1] != self.true_m2_edges_msun[k][-1]:
                    raise ValueError(f"derived edges of {k} must be a sub-set of the proposed edges with the same ends")
            d = self.derived.m2_app_edges_msun
            if not set(d) <= set(self.m2_app_edges_msun) or d[0] != self.m2_app_edges_msun[0] or d[-1] != self.m2_app_edges_msun[-1]:
                raise ValueError("derived M2_app edges must be a sub-set of the proposed edges with the same ends")
        return self


def load_rung5_bins(path: str | Path) -> Rung5BinsConfig:
    p = Path(path)
    if not p.is_absolute():
        p = ps.repo_root() / p
    return Rung5BinsConfig.model_validate(yaml.safe_load(p.read_text()))


def binned_ess(values: ArrayLike, weights: ArrayLike, edges: Sequence[float], mask: ArrayLike | None = None) -> NDArray[np.float64]:
    """Kish ESS of the weights falling in each bin of ``edges`` (values outside are dropped)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    e = np.asarray(edges, float)
    idx = np.searchsorted(e, v[ok], side="right") - 1
    inb = (idx >= 0) & (idx < e.size - 1)
    s1 = np.bincount(idx[inb], weights=w[ok][inb], minlength=e.size - 1)
    s2 = np.bincount(idx[inb], weights=w[ok][inb] ** 2, minlength=e.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s2 > 0, s1**2 / s2, 0.0)


def greedy_merge_edges(values: ArrayLike, weights: ArrayLike, edges: Sequence[float], min_ess: float,
                       mask: ArrayLike | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Merge adjacent bins left to right until each merged bin reaches ``min_ess``; a short last bin joins its left
    neighbour. Returns (merged edges, ESS per merged bin). The outer edges never move. If the whole range is short,
    one bin remains (its ESS is reported, and the caller flags it)."""
    e = [float(x) for x in edges]
    out = [e[0]]
    for hi in e[1:-1]:
        if binned_ess(values, weights, [out[-1], hi], mask)[0] >= min_ess:
            out.append(hi)
    out.append(e[-1])
    if len(out) > 2 and binned_ess(values, weights, out[-2:], mask)[0] < min_ess:
        del out[-2]
    return tuple(out), tuple(float(x) for x in binned_ess(values, weights, out, mask))


def freeze_digest(cfg: Rung5BinsConfig) -> str:
    """sha256 over the proposed block and the derived edges (not the bookkeeping fields)."""
    if cfg.derived is None:
        raise ValueError("not frozen: no derived block")
    payload = {
        "data_release": cfg.data_release, "co_version": cfg.co_version, "required_noise_model": cfg.required_noise_model,
        "true_m2_edges_msun": {k: list(v) for k, v in sorted(cfg.true_m2_edges_msun.items())},
        "m2_app_edges_msun": list(cfg.m2_app_edges_msun), "detection": cfg.detection,
        "c1_period_max_days": cfg.c1_period_max_days, "min_cell_ess": cfg.min_cell_ess,
        "derived_true": {k: list(v) for k, v in sorted(cfg.derived.true_m2_edges_msun.items())},
        "derived_app": list(cfg.derived.m2_app_edges_msun),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_frozen(cfg: Rung5BinsConfig) -> Rung5Derived:
    """The rung-5 fit calls this before touching real rows: the bins must be frozen and untampered."""
    if cfg.derived is None:
        raise ValueError("rung-5 bins are not frozen; run scripts/rung5_freeze_bins.py on the mock first")
    if freeze_digest(cfg) != cfg.derived.digest:
        raise ValueError("rung-5 bins config changed after freezing (digest mismatch); re-freeze from the mock")
    return cfg.derived


def derived_block(cfg: Rung5BinsConfig, **fields: Any) -> Rung5Derived:
    """Build the derived block and its digest for ``cfg``."""
    tmp = Rung5Derived(**fields, digest="")
    probe = cfg.model_copy(update={"derived": tmp})
    return tmp.model_copy(update={"digest": freeze_digest(probe)})
