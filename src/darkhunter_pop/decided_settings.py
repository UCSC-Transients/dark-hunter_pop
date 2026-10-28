"""Decided-settings guard (#391 spec §12.28, Ryan 2026-10-17).

One list of Ryan's decided settings (``config/population/decided_settings.yaml``) and a diff of a
launch (proposal fragment, merged pipeline config, parent snapshot) or a rung-3 fit config against
it. Pure bookkeeping: no physics, no thresholds of its own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

DEFAULT_PATH = Path("config/population/decided_settings.yaml")
Scope = Literal["launch_fragment", "pipeline_config", "launch_args", "fit_config"]
_MISSING = object()


class DecidedSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Scope
    key: str
    equals: Any = None
    present: bool | None = None
    ref: str

    @model_validator(mode="after")
    def _one_check(self) -> DecidedSetting:
        if (self.present is None) == ("equals" not in self.model_fields_set):
            raise ValueError(f"{self.key}: give exactly one of equals / present")
        return self


class DecidedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    min_guarded_generation: int
    #: #583: generations at or above min_guarded_generation that are superseded and never relaunched under the guard
    exempt_generations: list[int] = []
    settings: list[DecidedSetting]

    def for_scope(self, scope: Scope) -> list[DecidedSetting]:
        return [s for s in self.settings if s.scope == scope]


@dataclass(frozen=True)
class Deviation:
    scope: str
    key: str
    expected: Any
    actual: Any
    ref: str

    def __str__(self) -> str:
        act = "<missing>" if self.actual is _MISSING else repr(self.actual)
        return f"[{self.scope}] {self.key}: expected {self.expected!r}, got {act}  ({self.ref})"


def load_decided_settings(path: str | Path = DEFAULT_PATH) -> DecidedSettings:
    return DecidedSettings.model_validate(yaml.safe_load(Path(path).read_text()))


def _get(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, str) and not isinstance(b, str):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(a == b)


def check_mapping(data: Mapping[str, Any], settings: DecidedSettings, scope: Scope) -> list[Deviation]:
    """Every deviation of ``data`` from the decided settings of ``scope``."""
    out: list[Deviation] = []
    for s in settings.for_scope(scope):
        val = _get(data, s.key)
        if s.present is not None:
            ok = (val is not _MISSING and val is not None) == s.present
            if not ok:
                out.append(Deviation(scope, s.key, "present" if s.present else "absent", val, s.ref))
        elif val is _MISSING or not _same(val, s.equals):
            out.append(Deviation(scope, s.key, s.equals, val, s.ref))
    return out


def check_launch(
    fragment_path: str | Path,
    *,
    config: Any = None,
    parent_dir: str | Path | None = None,
    settings: DecidedSettings | None = None,
) -> list[Deviation]:
    """Deviations of one launch: the raw fragment YAML (``launch_fragment``), the merged pipeline
    config (``pipeline_config``; a ``PipelineConfig`` or a mapping, skipped if None) and the parent
    snapshot directory (``launch_args``; skipped if None)."""
    st = settings if settings is not None else load_decided_settings()
    frag = yaml.safe_load(Path(fragment_path).read_text())
    dev = check_mapping(frag, st, "launch_fragment")
    if config is not None:
        cfg = config.model_dump(mode="json") if hasattr(config, "model_dump") else config
        dev += check_mapping(cfg, st, "pipeline_config")
    if parent_dir is not None:
        dev += check_mapping({"parent_snapshot": Path(parent_dir).name}, st, "launch_args")
    return dev


def check_fit_config(path: str | Path, settings: DecidedSettings | None = None) -> list[Deviation]:
    st = settings if settings is not None else load_decided_settings()
    return check_mapping(yaml.safe_load(Path(path).read_text()), st, "fit_config")
