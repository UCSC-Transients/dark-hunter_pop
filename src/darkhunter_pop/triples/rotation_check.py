"""Rotation-consistency check (implied v_rot vs. uberMS v sin i) for ``triples``.

Interface hook only in v1 (ARCHITECTURE.md §4). Compares TESS-implied rotation
speed against uberMS ``v sin i`` when both are available; does not decide triples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import CandidateRecord, ParameterSet


@dataclass(frozen=True)
class RotationConsistencyEvidence:
    """Stub evidence payload for the rotation-consistency channel."""

    source_id: int
    channel_enabled: bool
    available: bool
    implied_v_rot_kms: float | None
    uberMS_v_sin_i_kms: float | None
    # None while stubbed — never a real consistency verdict in v1.
    consistent: bool | None
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "channel_enabled": self.channel_enabled,
            "available": self.available,
            "implied_v_rot_kms": self.implied_v_rot_kms,
            "uberMS_v_sin_i_kms": self.uberMS_v_sin_i_kms,
            "consistent": self.consistent,
            "notes": self.notes,
        }


def _v_sin_i_from_parameterset(ps: ParameterSet | None) -> float | None:
    if ps is None:
        return None
    for name in ("v_sin_i", "vsini", "vsin_i", "v_sin_i_kms"):
        if name in ps.names:
            return float(ps.values[ps.names.index(name)])
    return None


def resolve_uberMS_v_sin_i_kms(
    candidate: CandidateRecord,
    *,
    uberMS_v_sin_i_kms: float | None = None,
) -> float | None:
    """Resolve uberMS ``v sin i`` from an explicit value, extras, or M1 ParameterSet."""
    if uberMS_v_sin_i_kms is not None:
        return float(uberMS_v_sin_i_kms)
    extras = candidate.extras
    for key in ("v_sin_i_kms", "vsini_kms", "uberMS_v_sin_i_kms"):
        raw = extras.get(key)
        if raw is not None:
            return float(raw)
    return _v_sin_i_from_parameterset(candidate.m1)


def evaluate_rotation_consistency(
    candidate: CandidateRecord,
    config: PipelineConfig,
    *,
    uberMS_v_sin_i_kms: float | None = None,
) -> RotationConsistencyEvidence:
    """Compare implied v_rot vs uberMS v sin i when both exist; no triple science.

    Parameters
    ----------
    candidate:
        Pipeline candidate; may carry ``tess.implied_v_rot_kms`` and/or uberMS fields.
    config:
        Merged pipeline config; ``triples.rotation_consistency_channel`` gates the hook.
    uberMS_v_sin_i_kms:
        Optional explicit uberMS value; otherwise read from ``extras`` / ``m1``.

    Returns
    -------
    RotationConsistencyEvidence
        Stub: ``consistent`` is always ``None`` (no threshold applied).
    """
    channel_on = config.triples.rotation_consistency_channel
    if not channel_on:
        return RotationConsistencyEvidence(
            source_id=candidate.source_id,
            channel_enabled=False,
            available=False,
            implied_v_rot_kms=None,
            uberMS_v_sin_i_kms=None,
            consistent=None,
            notes="rotation_consistency_channel=false; hook skipped",
        )

    implied = None if candidate.tess is None else candidate.tess.implied_v_rot_kms
    vsini = resolve_uberMS_v_sin_i_kms(
        candidate, uberMS_v_sin_i_kms=uberMS_v_sin_i_kms
    )
    available = implied is not None and vsini is not None
    if not available:
        return RotationConsistencyEvidence(
            source_id=candidate.source_id,
            channel_enabled=True,
            available=False,
            implied_v_rot_kms=implied,
            uberMS_v_sin_i_kms=vsini,
            consistent=None,
            notes=(
                "need both tess.implied_v_rot_kms and uberMS v sin i; "
                "stub does not apply a consistency threshold"
            ),
        )
    return RotationConsistencyEvidence(
        source_id=candidate.source_id,
        channel_enabled=True,
        available=True,
        implied_v_rot_kms=implied,
        uberMS_v_sin_i_kms=vsini,
        consistent=None,
        notes=(
            "both rotation observables present; consistency threshold not "
            "implemented (stub); population_model forces P(triple)=0 in v1"
        ),
    )
