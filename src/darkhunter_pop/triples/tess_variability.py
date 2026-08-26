"""TESS photometric-variability evidence for the ``triples`` stage.

Interface hook only in v1 (ARCHITECTURE.md §4). Reads ``CandidateRecord.tess`` when
present; does not infer a triple probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import CandidateRecord


@dataclass(frozen=True)
class TessVariabilityEvidence:
    """Stub evidence payload for the TESS variability channel."""

    source_id: int
    channel_enabled: bool
    available: bool
    period_day: float | None
    amplitude: float | None
    variability_flag: bool | None
    implied_v_rot_kms: float | None
    light_curve_path: str | None
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "channel_enabled": self.channel_enabled,
            "available": self.available,
            "period_day": self.period_day,
            "amplitude": self.amplitude,
            "variability_flag": self.variability_flag,
            "implied_v_rot_kms": self.implied_v_rot_kms,
            "light_curve_path": self.light_curve_path,
            "notes": self.notes,
        }


def evaluate_tess_variability(
    candidate: CandidateRecord,
    config: PipelineConfig,
) -> TessVariabilityEvidence:
    """Return TESS-derived fields when present; no triple science.

    Parameters
    ----------
    candidate:
        Pipeline candidate; ``tess`` may be unset.
    config:
        Merged pipeline config; ``triples.tess_variability_channel`` gates the hook.

    Returns
    -------
    TessVariabilityEvidence
        Always a stub: ``available`` reflects whether a ``TessBlock`` was attached,
        never a triple classification.
    """
    channel_on = config.triples.tess_variability_channel
    if not channel_on:
        return TessVariabilityEvidence(
            source_id=candidate.source_id,
            channel_enabled=False,
            available=False,
            period_day=None,
            amplitude=None,
            variability_flag=None,
            implied_v_rot_kms=None,
            light_curve_path=None,
            notes="tess_variability_channel=false; hook skipped",
        )
    block = candidate.tess
    if block is None:
        return TessVariabilityEvidence(
            source_id=candidate.source_id,
            channel_enabled=True,
            available=False,
            period_day=None,
            amplitude=None,
            variability_flag=None,
            implied_v_rot_kms=None,
            light_curve_path=None,
            notes="no TessBlock on candidate; stub does not fetch light curves",
        )
    return TessVariabilityEvidence(
        source_id=candidate.source_id,
        channel_enabled=True,
        available=True,
        period_day=block.period_day,
        amplitude=block.amplitude,
        variability_flag=block.variability_flag,
        implied_v_rot_kms=block.implied_v_rot_kms,
        light_curve_path=block.light_curve_path,
        notes="TessBlock present; triple inference not implemented (stub)",
    )
