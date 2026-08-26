"""Load, merge, validate, and checksum pipeline configuration.

ARCHITECTURE.md §5, §7. Fragments under ``config/fragments/*.yaml`` are deep-merged (later
files override earlier on key conflicts) then overlaid by ``config/config.yaml``. Constants from
``darkhunter_pop.constants`` are never replaced by config; choosable offsets (e.g.
``delta_M_Ch_msun``) combine with constants at accessors below.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from astropy import units as u

from darkhunter_pop import constants
from darkhunter_pop.config_schema import (
    PipelineConfig,
    checksum_payload,
)
from darkhunter_pop.schemas import ActiveDRMode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"
_FRAGMENTS_DIR = _REPO_ROOT / "config" / "fragments"


def repo_root() -> Path:
    return _REPO_ROOT


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts only)."""
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def load_fragment_dicts(fragments_dir: Path = _FRAGMENTS_DIR) -> dict[str, Any]:
    """Merge all ``*.yaml`` fragments in sorted filename order."""
    if not fragments_dir.is_dir():
        return {}
    merged: dict[str, Any] = {}
    for path in sorted(fragments_dir.glob("*.yaml")):
        merged = deep_merge(merged, _load_yaml(path))
    return merged


def load_config(
    config_path: Path | None = None,
    *,
    fragments_dir: Path | None = None,
    merge_fragments: bool = True,
) -> PipelineConfig:
    """Load and validate the pipeline config.

    Order: fragments (optional) ← ``config.yaml`` (canonical file wins on conflicts).
    """
    path = config_path or _DEFAULT_CONFIG
    frag_dir = _FRAGMENTS_DIR if fragments_dir is None else fragments_dir
    merged: dict[str, Any] = {}
    if merge_fragments:
        merged = load_fragment_dicts(frag_dir)
    merged = deep_merge(merged, _load_yaml(path))
    return PipelineConfig.model_validate(merged)


def config_checksum(config: PipelineConfig) -> str:
    """SHA256 over canonical JSON of the active-DR + shared-physics payload."""
    payload = checksum_payload(config)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_config_checksum(config: PipelineConfig, expected: str) -> None:
    """Refuse resume/amend when the checksum does not match (ARCHITECTURE.md §5)."""
    got = config_checksum(config)
    if got != expected:
        raise ValueError(
            "config checksum mismatch (active DR + shared physics changed):\n"
            f"  got={got}\n"
            f"  expected={expected}\n"
            "Start a new run file rather than amending."
        )


def effective_M_Ch_msun(config: PipelineConfig) -> float:
    """``constants.M_CH`` plus optional config ``delta_M_Ch_msun``."""
    base = constants.M_CH.to_value(u.Msun)
    return float(base + config.mass_calibration.delta_M_Ch_msun)


def audit_dr_independence(config: PipelineConfig) -> list[str]:
    """Informational audit: flag identical path-specific values across DR3/DR4.

    Shared physics is intentionally single-keyed; this does not compare those. Unexpected
    divergence in shared physics cannot occur by construction (one object). Returns a list of
    human-readable notes (empty if none).
    """
    notes: list[str] = []
    d3 = config.dr3.model_dump(mode="json")
    d4 = config.dr4.model_dump(mode="json")
    for key in sorted(set(d3) & set(d4)):
        if d3[key] == d4[key]:
            notes.append(
                f"informational: dr3.{key} == dr4.{key} "
                f"(path-specific keys are independent even when values match)"
            )
    return notes


def require_dr3_active_for_v1(config: PipelineConfig) -> None:
    """DR4 execution is not enabled yet (ARCHITECTURE.md §5)."""
    if config.active_dr_mode is ActiveDRMode.DR4:
        raise ValueError(
            "active_dr_mode=dr4 is reserved but not runnable in v1; use dr3"
        )
