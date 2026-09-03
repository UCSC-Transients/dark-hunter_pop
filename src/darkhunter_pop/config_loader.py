"""Load, merge, validate, and checksum pipeline configuration.

ARCHITECTURE.md §5, §7. Fragments under ``config/fragments/*.yaml`` are deep-merged (later
files override earlier on key conflicts) then overlaid by ``config/config.yaml``. Constants from
``darkhunter_pop.constants`` are never replaced by config; choosable offsets (e.g.
``delta_M_Ch_msun``) combine with constants at accessors below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from astropy import units as u

from darkhunter_pop import constants
from darkhunter_pop.config_schema import (
    PATH_SPECIFIC_LEAF_KEYS,
    SHARED_PHYSICS_SECTIONS,
    PipelineConfig,
    checksum_payload,
)
from darkhunter_pop.schemas import ActiveDRMode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "config.yaml"
_FRAGMENTS_DIR = _REPO_ROOT / "config" / "fragments"

AuditSeverity = Literal["violation", "informational"]


@dataclass(frozen=True)
class DRAuditFinding:
    """One DR3/DR4 parameter-independence audit finding."""

    severity: AuditSeverity
    message: str
    key: str | None = None


@dataclass
class DRAuditResult:
    """Full walk of the parameter set for DR3/DR4 independence (ARCHITECTURE.md §6)."""

    findings: list[DRAuditFinding] = field(default_factory=list)

    @property
    def violations(self) -> list[DRAuditFinding]:
        return [f for f in self.findings if f.severity == "violation"]

    @property
    def informational(self) -> list[DRAuditFinding]:
        return [f for f in self.findings if f.severity == "informational"]

    @property
    def ok(self) -> bool:
        return not self.violations

    def messages(self) -> list[str]:
        """Flat human-readable notes (backward-compatible with list[str] callers)."""
        return [f.message for f in self.findings]


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


def enabled_selection_content_fingerprint(
    config: PipelineConfig, *, repo: Path | None = None
) -> str:
    """SHA256 over every enabled selection file's bytes plus its registry mode.

    CONTINUATION_PLAN §12.3: changing a threshold in an enabled per-sample file
    must produce a different ``sample_selection`` artifact path.
    """
    root = repo if repo is not None else repo_root()
    digest = hashlib.sha256()
    for entry in config.sample_selection.samples:
        if not entry.enabled:
            continue
        path = Path(entry.path)
        if not path.is_absolute():
            path = root / path
        digest.update(entry.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.mode.value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def _walk_keys(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested mappings to dotted paths (lists treated as leaves)."""
    out: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                out.extend(_walk_keys(value, path))
            else:
                out.append((path, value))
    else:
        out.append((prefix, node))
    return out


def audit_dr_independence(
    config: PipelineConfig | Mapping[str, Any],
) -> DRAuditResult:
    """Walk the full parameter set for DR3/DR4 independence (ARCHITECTURE.md §6).

    Flags:
    - **violation**: Gaia-mission-specific or external-data-use keys placed as shared
      top-level keys (must be independent under ``dr3`` / ``dr4``).
    - **violation**: shared physics/population sections unexpectedly duplicated under
      both DR paths with divergent values.
    - **informational**: identical values in path-specific keys (allowed, but values
      remain independently configured even when they match).

    Returns a :class:`DRAuditResult`. Call ``.messages()`` for a flat ``list[str]``.
    """
    result = DRAuditResult()
    if isinstance(config, PipelineConfig):
        data = config.model_dump(mode="json")
        top_fields = set(PipelineConfig.model_fields)
    else:
        data = dict(config)
        top_fields = set(data)

    # 1) Path-specific leaf keys must not appear as shared top-level PipelineConfig fields.
    for leaf in sorted(PATH_SPECIFIC_LEAF_KEYS):
        if leaf in top_fields and leaf not in {"dr3", "dr4"}:
            result.findings.append(
                DRAuditFinding(
                    severity="violation",
                    key=leaf,
                    message=(
                        f"violation: path-specific key '{leaf}' must not be a shared "
                        "top-level config field; place independently under dr3/ and dr4/"
                    ),
                )
            )

    # 2) Walk full tree: any top-level (non-dr) path containing a path-specific leaf
    # that is not nested under selection_function_* shared blocks' path children.
    for path, _value in _walk_keys(data):
        parts = path.split(".")
        leaf = parts[-1]
        under_dr = parts[0] in {"dr3", "dr4"}
        if leaf in PATH_SPECIFIC_LEAF_KEYS and not under_dr:
            # Shared selection_function_followup / astrometric blocks must not own
            # path-specific leaves like d_min_pc / accel_jerk_catalog_id.
            result.findings.append(
                DRAuditFinding(
                    severity="violation",
                    key=path,
                    message=(
                        f"violation: Gaia-mission/external-data key '{path}' is shared "
                        "across DR; must be independent under dr3.*/dr4.*"
                    ),
                )
            )

    # 3) Shared physics must exist once at top level; flag divergence if duplicated
    # under both DR paths (should not happen with the frozen schema, but catch raw YAML).
    d3 = data.get("dr3") if isinstance(data.get("dr3"), Mapping) else {}
    d4 = data.get("dr4") if isinstance(data.get("dr4"), Mapping) else {}
    assert isinstance(d3, Mapping) and isinstance(d4, Mapping)
    for section in sorted(SHARED_PHYSICS_SECTIONS):
        if section in d3 or section in d4:
            v3 = d3.get(section)
            v4 = d4.get(section)
            if v3 is not None and v4 is not None and v3 != v4:
                result.findings.append(
                    DRAuditFinding(
                        severity="violation",
                        key=section,
                        message=(
                            f"violation: shared physics section '{section}' diverges "
                            "between dr3 and dr4; keep a single top-level key"
                        ),
                    )
                )
            else:
                result.findings.append(
                    DRAuditFinding(
                        severity="violation",
                        key=section,
                        message=(
                            f"violation: shared physics section '{section}' must not live "
                            "under dr3/dr4; use a single top-level key"
                        ),
                    )
                )
        if section not in data:
            result.findings.append(
                DRAuditFinding(
                    severity="violation",
                    key=section,
                    message=(
                        f"violation: shared physics section '{section}' missing at "
                        "top level"
                    ),
                )
            )

    # 4) Both DR path blocks must be present so the audit can fire.
    if "dr3" not in data or "dr4" not in data:
        result.findings.append(
            DRAuditFinding(
                severity="violation",
                key="dr3/dr4",
                message="violation: both dr3 and dr4 path configs must be present",
            )
        )
        return result

    # 5) Informational: identical values in path-specific keys across DR3/DR4.
    for key in sorted(set(d3) & set(d4)):
        if d3[key] == d4[key]:
            result.findings.append(
                DRAuditFinding(
                    severity="informational",
                    key=key,
                    message=(
                        f"informational: dr3.{key} == dr4.{key} "
                        "(path-specific keys are independent even when values match)"
                    ),
                )
            )

    return result


def require_dr3_active_for_v1(config: PipelineConfig) -> None:
    """DR4 execution is not enabled yet (ARCHITECTURE.md §5)."""
    if config.active_dr_mode is ActiveDRMode.DR4:
        raise ValueError(
            "active_dr_mode=dr4 is reserved but not runnable in v1; use dr3"
        )
