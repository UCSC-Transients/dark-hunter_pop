"""Stage: ``companion_nature_likelihood``.

Continuous joint likelihood over the true nature of M2 from broadband photometry
residual (ΔBIC), Gaia XP residual, and SB2 spectral characteristics when present
(ARCHITECTURE.md §4). Feeds ``population_model`` as per-system weights over
``COMPANION_NATURE_WEIGHT_KEYS`` (``BH``/``NS``/``WD``/``other``/``outlier``) —
never a pre-filter / discard. Photometric evidence is scored as WD / other / dark
then mapped onto those five keys. Bédard cooling tracks load from
``physics.cooling_tracks_path`` only (local files; no runtime fetch).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_loader import repo_root as default_repo_root
from darkhunter_pop.config_loader import require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    CompanionNatureConfig,
    CoolingAtmosphere,
    PipelineConfig,
)
from darkhunter_pop.mass_derivation import read_stage_hdf5 as read_mass_stage_hdf5
from darkhunter_pop.mass_derivation import write_stage_hdf5 as write_mass_stage_hdf5
from darkhunter_pop.plotting import plot_histogram
from darkhunter_pop.run_management import (
    STAGE_REGISTRY,
    mark_stage_finished,
    mark_stage_started,
    plan_stage,
    save_run_manifest,
    stage_artifact_path,
)
from darkhunter_pop.schemas import (
    COMPANION_NATURE_WEIGHT_KEYS,
    CandidateRecord,
    PhotometryPoint,
    RunManifest,
    StageStatus,
)

SCHEMA_VERSION = 1
STAGE_NAME = "companion_nature_likelihood"
# Internal photometric hypotheses (joint BIC); mapped → five population keys.
PHOTOMETRIC_CLASSES: tuple[str, ...] = ("WD", "other", "dark")
NATURE_CLASSES = COMPANION_NATURE_WEIGHT_KEYS
NatureClass = Literal["BH", "NS", "WD", "other", "outlier"]
NatureTier = Literal["fast", "full"]

# Provenance tags for weight metadata (extras); not a freeze schema enum.
TIER_PROVENANCE = {
    "fast": "companion_nature_fast",
    "full": "companion_nature_full",
}


class ChannelName(str, Enum):
    PHOTOMETRY = "photometry"
    XP = "xp"
    SB2 = "sb2"


@dataclass(frozen=True)
class CoolingTrackTable:
    """Local Bédard-style cooling track grid (mass, age → absolute mag).

    Files are CSV with columns ``mass_msun,age_gyr,Mg`` (or ``M_G``). Atmosphere
    and model name are recorded for fingerprinting; no network I/O.
    """

    mass_msun: NDArray[np.floating]
    age_gyr: NDArray[np.floating]
    mg: NDArray[np.floating]
    model: str
    atmosphere: str
    source_path: str | None

    def abs_mag_g(self, mass_msun: float, age_gyr: float | None) -> float:
        """Nearest-neighbour lookup; falls back to mass-only median when age absent."""
        if self.mass_msun.size == 0:
            raise ValueError("empty cooling track table")
        mass = float(mass_msun)
        if age_gyr is None or not np.isfinite(age_gyr):
            # Mass-only: median Mg among nearest mass slice.
            dm = np.abs(self.mass_msun - mass)
            nearest_mass = float(self.mass_msun[int(np.argmin(dm))])
            mask = np.isclose(self.mass_msun, nearest_mass, rtol=0.0, atol=1e-6)
            if not np.any(mask):
                mask = dm <= np.min(dm) + 1e-12
            return float(np.median(self.mg[mask]))
        age = float(age_gyr)
        dist2 = (self.mass_msun - mass) ** 2 + (self.age_gyr - age) ** 2
        return float(self.mg[int(np.argmin(dist2))])


@dataclass(frozen=True)
class ChannelContribution:
    """Per-hypothesis chi2 and data count for one evidence channel."""

    channel: ChannelName
    chi2_by_class: dict[str, float]
    n_data: int
    available: bool


@dataclass(frozen=True)
class NatureEvidence:
    """Joint multi-band evidence for one candidate (available channels only)."""

    source_id: int
    channels: tuple[ChannelContribution, ...]
    bic_by_class: dict[str, float]
    delta_bic_wd_vs_dark: float
    delta_bic_other_vs_dark: float
    weights: dict[str, float]
    tier: NatureTier
    track_source: str | None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "channels": [
                {
                    "channel": c.channel.value,
                    "available": c.available,
                    "n_data": c.n_data,
                    "chi2_by_class": dict(c.chi2_by_class),
                }
                for c in self.channels
            ],
            "bic_by_class": dict(self.bic_by_class),
            "delta_bic_wd_vs_dark": self.delta_bic_wd_vs_dark,
            "delta_bic_other_vs_dark": self.delta_bic_other_vs_dark,
            "weights": dict(self.weights),
            "tier": self.tier,
            "track_source": self.track_source,
            "notes": self.notes,
        }


@dataclass
class AgeBinDiagnostic:
    """Required age-independence diagnostic (ARCHITECTURE.md §4)."""

    bin_edges_gyr: list[float]
    bin_counts: list[int]
    mean_weights_by_bin: list[dict[str, float]]
    global_mean_weights: dict[str, float]
    max_abs_mean_weight_delta: float
    age_independence_ok: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bin_edges_gyr": list(self.bin_edges_gyr),
            "bin_counts": list(self.bin_counts),
            "mean_weights_by_bin": [dict(w) for w in self.mean_weights_by_bin],
            "global_mean_weights": dict(self.global_mean_weights),
            "max_abs_mean_weight_delta": self.max_abs_mean_weight_delta,
            "age_independence_ok": self.age_independence_ok,
            "message": self.message,
        }


@dataclass
class CompanionNatureDiagnostics:
    """Stage-level funnel + age diagnostic (full detail; not caveman-compressed)."""

    n_input: int = 0
    n_weighted: int = 0
    n_fast: int = 0
    n_full: int = 0
    n_photometry: int = 0
    n_xp: int = 0
    n_sb2: int = 0
    n_no_channels: int = 0
    delta_bic_wd_vs_dark: list[float] = field(default_factory=list)
    age_diagnostic: AgeBinDiagnostic | None = None
    track_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_input": self.n_input,
            "n_weighted": self.n_weighted,
            "n_fast": self.n_fast,
            "n_full": self.n_full,
            "n_photometry": self.n_photometry,
            "n_xp": self.n_xp,
            "n_sb2": self.n_sb2,
            "n_no_channels": self.n_no_channels,
            "delta_bic_wd_vs_dark": list(self.delta_bic_wd_vs_dark),
            "track_source": self.track_source,
        }
        if self.age_diagnostic is not None:
            payload["age_diagnostic"] = self.age_diagnostic.as_dict()
        return payload


# ---------------------------------------------------------------------------
# Cooling tracks (local only)
# ---------------------------------------------------------------------------


def _analytic_mg_wd(mass_msun: float, cfg: CompanionNatureConfig) -> float:
    return float(cfg.wd_mg_zero_point + cfg.wd_mg_mass_slope * math.log10(max(mass_msun, 1e-3)))


def _analytic_mg_other(mass_msun: float, cfg: CompanionNatureConfig) -> float:
    return float(
        cfg.other_mg_zero_point + cfg.other_mg_mass_slope * math.log10(max(mass_msun, 1e-3))
    )


def load_cooling_tracks(
    config: PipelineConfig,
    *,
    repo_root: Path | None = None,
) -> CoolingTrackTable:
    """Load local cooling-track CSV; never fetch remotely.

    When ``physics.cooling_tracks_path`` is null or missing, return an empty table
    so callers fall back to the analytic Mg model parameterized in
    ``companion_nature`` (unit tests / scaffold). Production runs should point
    ``cooling_tracks_path`` at staged Bédard files under ``data/``.
    """
    physics = config.physics
    model = physics.cooling_tracks.value
    atmosphere = physics.cooling_atmosphere.value
    raw_path = physics.cooling_tracks_path
    if raw_path is None or str(raw_path).strip() == "":
        return CoolingTrackTable(
            mass_msun=np.asarray([], dtype=np.float64),
            age_gyr=np.asarray([], dtype=np.float64),
            mg=np.asarray([], dtype=np.float64),
            model=model,
            atmosphere=atmosphere,
            source_path=None,
        )

    path = Path(raw_path)
    if not path.is_absolute():
        root = repo_root if repo_root is not None else default_repo_root()
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(
            f"physics.cooling_tracks_path not found (local only, no fetch): {path}"
        )

    # CSV: mass_msun, age_gyr, Mg  (header required)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"cooling track file empty: {path}")
    names = {n.lower(): n for n in data.dtype.names or ()}
    mass_key = names.get("mass_msun") or names.get("mass")
    age_key = names.get("age_gyr") or names.get("age")
    mg_key = names.get("mg") or names.get("m_g") or names.get("abs_mag_g")
    if mass_key is None or age_key is None or mg_key is None:
        raise ValueError(
            f"cooling track CSV needs mass_msun, age_gyr, Mg columns: {path}"
        )
    return CoolingTrackTable(
        mass_msun=np.asarray(data[mass_key], dtype=np.float64).reshape(-1),
        age_gyr=np.asarray(data[age_key], dtype=np.float64).reshape(-1),
        mg=np.asarray(data[mg_key], dtype=np.float64).reshape(-1),
        model=model,
        atmosphere=atmosphere,
        source_path=str(path),
    )


def wd_abs_mag_g(
    mass_msun: float,
    *,
    age_gyr: float | None,
    tracks: CoolingTrackTable,
    cfg: CompanionNatureConfig,
    tier: NatureTier,
) -> float:
    """WD absolute G mag from local tracks, else analytic companion_nature knobs."""
    if tracks.mass_msun.size > 0:
        mg = tracks.abs_mag_g(mass_msun, age_gyr)
        if tier == "full":
            # Full tier: average nearest grid_factor neighbours for a mild smooth.
            k = min(cfg.full_tier_grid_factor, tracks.mass_msun.size)
            if age_gyr is None or not np.isfinite(age_gyr):
                dist = np.abs(tracks.mass_msun - mass_msun)
            else:
                dist = np.sqrt(
                    (tracks.mass_msun - mass_msun) ** 2
                    + (tracks.age_gyr - float(age_gyr)) ** 2
                )
            idx = np.argpartition(dist, kth=k - 1)[:k]
            return float(np.mean(tracks.mg[idx]))
        return mg
    return _analytic_mg_wd(mass_msun, cfg)


# ---------------------------------------------------------------------------
# Evidence channels + joint BIC
# ---------------------------------------------------------------------------


def _m2_msun(candidate: CandidateRecord) -> float | None:
    if candidate.m2 is None:
        return None
    try:
        val = candidate.m2.marginal("M2").value
    except KeyError:
        if not candidate.m2.values:
            return None
        val = candidate.m2.values[0]
    if val is None or not np.isfinite(val) or val <= 0:
        return None
    return float(val)


def _primary_age_gyr(candidate: CandidateRecord, cfg: CompanionNatureConfig) -> float | None:
    key = cfg.age_extras_key
    raw = candidate.extras.get(key)
    if raw is None and candidate.m1 is not None and "age_gyr" in candidate.m1.names:
        try:
            raw = candidate.m1.marginal("age_gyr").value
        except KeyError:
            raw = None
    if raw is None:
        return None
    try:
        age = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(age) or age < 0:
        return None
    return age


def _combine_mags(m_primary: float, m_companion: float | None) -> float:
    """Flux-sum two magnitudes; ``None`` companion → primary only."""
    if m_companion is None:
        return float(m_primary)
    f1 = 10.0 ** (-0.4 * float(m_primary))
    f2 = 10.0 ** (-0.4 * float(m_companion))
    return float(-2.5 * math.log10(f1 + f2))


def _photometry_points(candidate: CandidateRecord) -> list[PhotometryPoint]:
    return list(candidate.photometry or [])


def _m1_msun(candidate: CandidateRecord) -> float | None:
    if candidate.m1 is None:
        return None
    try:
        val = candidate.m1.marginal("M1").value
    except KeyError:
        if not candidate.m1.values:
            return None
        val = candidate.m1.values[0]
    if val is None or not np.isfinite(val) or val <= 0:
        return None
    return float(val)


def _analytic_mg_primary(mass_msun: float, cfg: CompanionNatureConfig) -> float:
    return float(
        cfg.primary_mg_zero_point
        + cfg.primary_mg_mass_slope * math.log10(max(mass_msun, 1e-3))
    )


def _distance_modulus(parallax_mas: float | None) -> float | None:
    if parallax_mas is None or not np.isfinite(parallax_mas) or parallax_mas <= 0:
        return None
    return float(10.0 - 5.0 * math.log10(float(parallax_mas)))


def _precomputed_photometry_channel(
    candidate: CandidateRecord, cfg: CompanionNatureConfig
) -> ChannelContribution | None:
    extras = candidate.extras
    dark = extras.get(cfg.phot_chi2_dark_key)
    wd = extras.get(cfg.phot_chi2_wd_key)
    other = extras.get(cfg.phot_chi2_other_key)
    n_raw = extras.get(cfg.phot_n_data_key)
    if dark is None or wd is None or other is None:
        return None
    try:
        chi2 = {"dark": float(dark), "WD": float(wd), "other": float(other)}
        n_data = int(n_raw) if n_raw is not None else max(1, len(candidate.photometry))
    except (TypeError, ValueError):
        return None
    if n_data < 1 or any(not np.isfinite(v) or v < 0 for v in chi2.values()):
        return None
    return ChannelContribution(
        channel=ChannelName.PHOTOMETRY,
        chi2_by_class=chi2,
        n_data=n_data,
        available=True,
    )


def _photometric_empty() -> dict[str, float]:
    return {c: 0.0 for c in PHOTOMETRIC_CLASSES}


def photometry_channel(
    candidate: CandidateRecord,
    *,
    cfg: CompanionNatureConfig,
    tracks: CoolingTrackTable,
    tier: NatureTier,
    age_gyr: float | None,
    m2: float | None,
) -> ChannelContribution:
    """Broadband photometry residual under dark / WD / other hypotheses.

    Prefer precomputed joint-SED chi2 in ``extras`` (single vs single+companion ΔBIC
    inputs). Otherwise build an absolute-mag flux-sum residual from M1/M2 + parallax
    + local cooling tracks. Missing ingredients → channel masked (not a discard).
    """
    empty = _photometric_empty()
    if not cfg.use_photometry:
        return ChannelContribution(
            channel=ChannelName.PHOTOMETRY,
            chi2_by_class=empty,
            n_data=0,
            available=False,
        )

    precomputed = _precomputed_photometry_channel(candidate, cfg)
    if precomputed is not None:
        return precomputed

    points = _photometry_points(candidate)
    m1 = _m1_msun(candidate)
    mu = _distance_modulus(candidate.parallax_mas)
    if not points or m1 is None or m2 is None or mu is None:
        return ChannelContribution(
            channel=ChannelName.PHOTOMETRY,
            chi2_by_class=empty,
            n_data=0,
            available=False,
        )

    mg_primary = _analytic_mg_primary(m1, cfg)
    m_primary = mg_primary + mu
    mg_wd = wd_abs_mag_g(m2, age_gyr=age_gyr, tracks=tracks, cfg=cfg, tier=tier)
    mg_other = _analytic_mg_other(m2, cfg)
    m_c_wd = mg_wd + mu
    m_c_other = mg_other + mu

    chi2 = _photometric_empty()
    n_data = 0
    for point in points:
        err = point.mag_err if point.mag_err is not None else cfg.default_mag_err
        if err is None or not np.isfinite(err) or err <= 0:
            continue
        m_obs = float(point.mag)
        preds = {
            "dark": m_primary,
            "WD": _combine_mags(m_primary, m_c_wd),
            "other": _combine_mags(m_primary, m_c_other),
        }
        for name, m_pred in preds.items():
            resid = (m_obs - m_pred) / float(err)
            chi2[name] += float(resid * resid)
        n_data += 1

    if n_data == 0:
        return ChannelContribution(
            channel=ChannelName.PHOTOMETRY,
            chi2_by_class=empty,
            n_data=0,
            available=False,
        )
    return ChannelContribution(
        channel=ChannelName.PHOTOMETRY,
        chi2_by_class=chi2,
        n_data=n_data,
        available=True,
    )


def xp_channel(candidate: CandidateRecord, cfg: CompanionNatureConfig) -> ChannelContribution:
    """Gaia XP residual channel from extras chi2 keys when present."""
    empty = _photometric_empty()
    if not cfg.use_xp:
        return ChannelContribution(
            channel=ChannelName.XP, chi2_by_class=empty, n_data=0, available=False
        )
    extras = candidate.extras
    dark = extras.get(cfg.xp_chi2_dark_key)
    wd = extras.get(cfg.xp_chi2_wd_key)
    other = extras.get(cfg.xp_chi2_other_key)
    n_raw = extras.get(cfg.xp_n_data_key)
    if dark is None or wd is None or other is None:
        return ChannelContribution(
            channel=ChannelName.XP, chi2_by_class=empty, n_data=0, available=False
        )
    try:
        chi2 = {
            "dark": float(dark),
            "WD": float(wd),
            "other": float(other),
        }
        n_data = int(n_raw) if n_raw is not None else 1
    except (TypeError, ValueError):
        return ChannelContribution(
            channel=ChannelName.XP, chi2_by_class=empty, n_data=0, available=False
        )
    if n_data < 1 or any(not np.isfinite(v) or v < 0 for v in chi2.values()):
        return ChannelContribution(
            channel=ChannelName.XP, chi2_by_class=empty, n_data=0, available=False
        )
    return ChannelContribution(
        channel=ChannelName.XP,
        chi2_by_class=chi2,
        n_data=n_data,
        available=True,
    )


def _sb2_unlocked(candidate: CandidateRecord) -> bool:
    extras = candidate.extras
    if extras.get("sb2_mass_ratio_unlocked") is True:
        return True
    if extras.get("sb2_consistent") is True:
        return True
    rv = candidate.rv_summary or {}
    if isinstance(rv.get("sb2_orbit"), Mapping):
        return True
    nss = candidate.nss_solution_type or ""
    return nss.upper() == "SB2"


def _sb2_wd_likeness(candidate: CandidateRecord, cfg: CompanionNatureConfig) -> float | None:
    key = cfg.sb2_wd_likeness_key
    raw = candidate.extras.get(key)
    if raw is None:
        rv = candidate.rv_summary or {}
        raw = rv.get(key)
        if raw is None and isinstance(rv.get("sb2_orbit"), Mapping):
            raw = rv["sb2_orbit"].get(key)
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return float(np.clip(score, 0.0, 1.0))


def sb2_channel(candidate: CandidateRecord, cfg: CompanionNatureConfig) -> ChannelContribution:
    """SB2 spectral characteristics → chi2-like scores when SB2 present.

    A luminous secondary strongly disfavors ``dark``. WD-likeness score maps to
    relative chi2 between WD and other; absent score → equal WD/other, dark high.
    """
    empty = _photometric_empty()
    if not cfg.use_sb2 or not _sb2_unlocked(candidate):
        return ChannelContribution(
            channel=ChannelName.SB2, chi2_by_class=empty, n_data=0, available=False
        )
    likeness = _sb2_wd_likeness(candidate, cfg)
    n_data = int(cfg.sb2_score_n_data)
    # Dark is incompatible with an SB2 detection of a luminous secondary.
    chi2_dark = float(n_data * cfg.sb2_dark_chi2_per_datum)
    scale = float(cfg.sb2_type_chi2_scale)
    if likeness is None:
        # Informative that secondary is luminous, but type unconstrained.
        chi2 = {
            "dark": chi2_dark,
            "WD": float(n_data),
            "other": float(n_data),
        }
    else:
        # likeness=1 → WD preferred; 0 → other preferred.
        chi2 = {
            "dark": chi2_dark,
            "WD": float(n_data * (1.0 - likeness) ** 2 * scale),
            "other": float(n_data * likeness**2 * scale),
        }
    return ChannelContribution(
        channel=ChannelName.SB2,
        chi2_by_class=chi2,
        n_data=n_data,
        available=True,
    )


def joint_bic_by_class(
    channels: Sequence[ChannelContribution],
    cfg: CompanionNatureConfig,
) -> tuple[dict[str, float], int]:
    """Joint multi-band BIC over photometric hypotheses: sum chi2 + k ln n.

    Not an independent per-band probability product — one shared residual budget.
    """
    n_params = {
        "dark": cfg.n_params_dark,
        "WD": cfg.n_params_wd,
        "other": cfg.n_params_other,
    }
    chi2_tot = _photometric_empty()
    n_data = 0
    for ch in channels:
        if not ch.available or ch.n_data < 1:
            continue
        n_data += ch.n_data
        for name in PHOTOMETRIC_CLASSES:
            chi2_tot[name] += float(ch.chi2_by_class[name])
    if n_data < 1:
        # No evidence → uniform BIC (weights → uniform after floor).
        return _photometric_empty(), 0
    log_n = math.log(float(n_data))
    bic = {
        name: chi2_tot[name] + float(n_params[name]) * log_n
        for name in PHOTOMETRIC_CLASSES
    }
    return bic, n_data


def photometric_weights_from_bic(
    bic_by_class: Mapping[str, float],
    cfg: CompanionNatureConfig,
) -> dict[str, float]:
    """Continuous softmax over photometric WD/other/dark BIC; apply floor."""
    scale = float(cfg.evidence_scale) * float(cfg.delta_bic_threshold)
    if scale <= 0:
        raise ValueError("evidence_scale * delta_bic_threshold must be > 0")
    shifted = {k: -0.5 * float(v) / scale for k, v in bic_by_class.items()}
    m = max(shifted.values()) if shifted else 0.0
    exps = {k: math.exp(v - m) for k, v in shifted.items()}
    total = sum(exps.values()) or 1.0
    raw = {k: exps[k] / total for k in PHOTOMETRIC_CLASSES}
    floored = {k: max(float(cfg.weight_floor), raw[k]) for k in PHOTOMETRIC_CLASSES}
    norm = sum(floored.values()) or 1.0
    return {k: floored[k] / norm for k in PHOTOMETRIC_CLASSES}


def _gate_failed_outlier_route(candidate: CandidateRecord) -> bool:
    extras = candidate.extras
    if extras.get("rv_astrometry_gate_passed") is False:
        return True
    skip = extras.get("joint_orbit_fit_skip_reason")
    if skip == "rv_astrometry_gate_failed":
        return True
    gate = extras.get("rv_astrometry_gate")
    if isinstance(gate, Mapping) and gate.get("passed") is False:
        return True
    return False


def map_photometric_to_population_weights(
    photometric: Mapping[str, float],
    candidate: CandidateRecord,
    cfg: CompanionNatureConfig,
) -> dict[str, float]:
    """Map WD/other/dark → ``COMPANION_NATURE_WEIGHT_KEYS`` (#56 ↔ #57).

    Photometric ``dark`` splits into BH/NS by ``dark_to_bh_fraction``. Gate failures
    blend mass into ``outlier`` via ``outlier_gate_blend``. Floor + renorm so no
    class is hard-zero.
    """
    dark = float(photometric.get("dark", 0.0))
    bh_frac = float(cfg.dark_to_bh_fraction)
    mapped = {
        "BH": dark * bh_frac,
        "NS": dark * (1.0 - bh_frac),
        "WD": float(photometric.get("WD", 0.0)),
        "other": float(photometric.get("other", 0.0)),
        "outlier": 0.0,
    }
    if _gate_failed_outlier_route(candidate):
        alpha = float(cfg.outlier_gate_blend)
        mapped = {k: (1.0 - alpha) * v for k, v in mapped.items()}
        mapped["outlier"] = alpha
    floored = {k: max(float(cfg.weight_floor), mapped[k]) for k in NATURE_CLASSES}
    norm = sum(floored.values()) or 1.0
    return {k: floored[k] / norm for k in NATURE_CLASSES}


def weights_from_bic(
    bic_by_class: Mapping[str, float],
    cfg: CompanionNatureConfig,
    *,
    candidate: CandidateRecord | None = None,
) -> dict[str, float]:
    """Photometric BIC → five-key population weights (compat helper)."""
    photo = photometric_weights_from_bic(bic_by_class, cfg)
    if candidate is None:
        # Unit tests without a candidate: no gate blend.
        stub = CandidateRecord(source_id=0)
        return map_photometric_to_population_weights(photo, stub, cfg)
    return map_photometric_to_population_weights(photo, candidate, cfg)


def select_tier(
    candidate: CandidateRecord,
    *,
    cfg: CompanionNatureConfig,
    delta_bic_best_second: float,
    m2: float | None,
) -> NatureTier:
    """Two-tier policy: full when ambiguous ΔBIC or critical M2."""
    if cfg.default_tier == "full":
        return "full"
    if abs(delta_bic_best_second) < float(cfg.full_tier_ambiguity_delta_bic):
        return "full"
    if (
        cfg.full_tier_m2_msun_min is not None
        and m2 is not None
        and m2 >= float(cfg.full_tier_m2_msun_min)
    ):
        return "full"
    if candidate.extras.get("companion_nature_force_full") is True:
        return "full"
    return "fast"


def evaluate_companion_nature(
    candidate: CandidateRecord,
    config: PipelineConfig,
    *,
    tracks: CoolingTrackTable | None = None,
    tier: NatureTier | None = None,
) -> NatureEvidence:
    """Joint continuous nature likelihood for one candidate (never discards)."""
    cfg = config.companion_nature
    track_table = tracks if tracks is not None else load_cooling_tracks(config)

    m2 = _m2_msun(candidate)
    age = _primary_age_gyr(candidate, cfg)

    # Fast pass first to decide tier when not forced.
    channels_fast = (
        photometry_channel(
            candidate, cfg=cfg, tracks=track_table, tier="fast", age_gyr=age, m2=m2
        ),
        xp_channel(candidate, cfg),
        sb2_channel(candidate, cfg),
    )
    bic_fast, _ = joint_bic_by_class(channels_fast, cfg)
    ordered = sorted(bic_fast, key=bic_fast.get)
    best, second = ordered[0], ordered[1]
    delta_best_second = float(bic_fast[second] - bic_fast[best])
    chosen_tier = tier or select_tier(
        candidate, cfg=cfg, delta_bic_best_second=delta_best_second, m2=m2
    )

    if chosen_tier == "full":
        channels = (
            photometry_channel(
                candidate,
                cfg=cfg,
                tracks=track_table,
                tier="full",
                age_gyr=age,
                m2=m2,
            ),
            xp_channel(candidate, cfg),
            sb2_channel(candidate, cfg),
        )
        bic, _ = joint_bic_by_class(channels, cfg)
    else:
        channels = channels_fast
        bic = bic_fast

    weights = weights_from_bic(bic, cfg, candidate=candidate)
    notes_parts: list[str] = []
    if not any(c.available for c in channels):
        notes_parts.append("no_evidence_channels:uniform_weights")
    if m2 is None:
        notes_parts.append("missing_m2")
    if track_table.source_path is None:
        notes_parts.append("analytic_wd_mg_fallback")
    elif config.physics.cooling_atmosphere is CoolingAtmosphere.HE:
        notes_parts.append("He_atmosphere")
    else:
        notes_parts.append("DA_atmosphere")
    if _gate_failed_outlier_route(candidate):
        notes_parts.append("outlier_gate_blend")

    return NatureEvidence(
        source_id=candidate.source_id,
        channels=channels,
        bic_by_class=bic,
        delta_bic_wd_vs_dark=float(bic["WD"] - bic["dark"]),
        delta_bic_other_vs_dark=float(bic["other"] - bic["dark"]),
        weights=weights,
        tier=chosen_tier,
        track_source=track_table.source_path,
        notes=";".join(notes_parts),
    )


def apply_nature_to_candidate(
    candidate: CandidateRecord,
    evidence: NatureEvidence,
) -> CandidateRecord:
    """Attach weights; never drop the candidate."""
    extras = dict(candidate.extras)
    extras["companion_nature_tier"] = evidence.tier
    extras["companion_nature_provenance"] = TIER_PROVENANCE[evidence.tier]
    extras["companion_nature_delta_bic_wd_vs_dark"] = evidence.delta_bic_wd_vs_dark
    extras["companion_nature_delta_bic_other_vs_dark"] = evidence.delta_bic_other_vs_dark
    extras["companion_nature_channels"] = [
        c.channel.value for c in evidence.channels if c.available
    ]
    extras["companion_nature_photometric_bic"] = dict(evidence.bic_by_class)
    if evidence.notes:
        extras["companion_nature_notes"] = evidence.notes
    return candidate.model_copy(
        update={
            "companion_nature_weights": {
                k: float(evidence.weights[k]) for k in NATURE_CLASSES
            },
            "extras": extras,
        }
    )


# ---------------------------------------------------------------------------
# Age-bin diagnostic (required)
# ---------------------------------------------------------------------------


def age_bin_diagnostic(
    candidates: Sequence[CandidateRecord],
    cfg: CompanionNatureConfig,
    *,
    independence_tol: float | None = None,
) -> AgeBinDiagnostic:
    """Stratify mean nature weights by primary age bin (age-independence check).

    ``independence_tol`` defaults to ``1 / delta_bic_threshold`` so the tolerance
    stays config-owned rather than a hardcoded magic number.
    """
    edges = list(cfg.age_bin_edges_gyr)
    n_bins = len(edges) - 1
    tol = (
        float(independence_tol)
        if independence_tol is not None
        else 1.0 / float(cfg.delta_bic_threshold)
    )
    sums = [{c: 0.0 for c in NATURE_CLASSES} for _ in range(n_bins)]
    counts = [0 for _ in range(n_bins)]
    global_sum = {c: 0.0 for c in NATURE_CLASSES}
    n_global = 0

    for cand in candidates:
        weights = cand.companion_nature_weights
        if not weights:
            continue
        age = _primary_age_gyr(cand, cfg)
        for k in NATURE_CLASSES:
            global_sum[k] += float(weights.get(k, 0.0))
        n_global += 1
        if age is None:
            continue
        # Rightmost edge inclusive.
        idx = None
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if (age >= lo and age < hi) or (i == n_bins - 1 and age <= hi):
                idx = i
                break
        if idx is None:
            continue
        for k in NATURE_CLASSES:
            sums[idx][k] += float(weights.get(k, 0.0))
        counts[idx] += 1

    global_mean = (
        {k: global_sum[k] / n_global for k in NATURE_CLASSES}
        if n_global
        else {k: 1.0 / len(NATURE_CLASSES) for k in NATURE_CLASSES}
    )
    mean_by_bin: list[dict[str, float]] = []
    max_delta = 0.0
    for i in range(n_bins):
        if counts[i] == 0:
            mean_by_bin.append({k: float("nan") for k in NATURE_CLASSES})
            continue
        mean = {k: sums[i][k] / counts[i] for k in NATURE_CLASSES}
        mean_by_bin.append(mean)
        for k in NATURE_CLASSES:
            max_delta = max(max_delta, abs(mean[k] - global_mean[k]))

    ok = max_delta <= tol or n_global == 0
    msg = (
        f"Age-bin diagnostic: max |Δmean weight|={max_delta:.4g} "
        f"(tol={tol:.4g}); bins={counts}; "
        f"{'age-independence OK' if ok else 'age-dependence FLAG'}."
    )
    return AgeBinDiagnostic(
        bin_edges_gyr=edges,
        bin_counts=counts,
        mean_weights_by_bin=mean_by_bin,
        global_mean_weights=global_mean,
        max_abs_mean_weight_delta=max_delta,
        age_independence_ok=ok,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Batch + stage runner
# ---------------------------------------------------------------------------


def run_companion_nature_on_candidates(
    candidates: Sequence[CandidateRecord],
    config: PipelineConfig,
    *,
    tracks: CoolingTrackTable | None = None,
) -> tuple[list[CandidateRecord], CompanionNatureDiagnostics]:
    """Score every candidate; return all with weights (no discard)."""
    track_table = tracks if tracks is not None else load_cooling_tracks(config)
    cfg = config.companion_nature
    diag = CompanionNatureDiagnostics(
        n_input=len(candidates),
        track_source=track_table.source_path,
    )
    out: list[CandidateRecord] = []
    full_queue = 0
    max_full = cfg.full_queue_max

    for cand in candidates:
        # Provisional tier from fast pass; may promote to full.
        evidence = evaluate_companion_nature(cand, config, tracks=track_table)
        if evidence.tier == "full":
            if max_full is not None and full_queue >= max_full:
                # Cap full queue: keep fast result.
                evidence = evaluate_companion_nature(
                    cand, config, tracks=track_table, tier="fast"
                )
            else:
                full_queue += 1
                if evidence.tier != "full":
                    evidence = evaluate_companion_nature(
                        cand, config, tracks=track_table, tier="full"
                    )

        updated = apply_nature_to_candidate(cand, evidence)
        out.append(updated)
        diag.n_weighted += 1
        if evidence.tier == "full":
            diag.n_full += 1
        else:
            diag.n_fast += 1
        available = {c.channel for c in evidence.channels if c.available}
        if ChannelName.PHOTOMETRY in available:
            diag.n_photometry += 1
        if ChannelName.XP in available:
            diag.n_xp += 1
        if ChannelName.SB2 in available:
            diag.n_sb2 += 1
        if not available:
            diag.n_no_channels += 1
        diag.delta_bic_wd_vs_dark.append(evidence.delta_bic_wd_vs_dark)

    diag.age_diagnostic = age_bin_diagnostic(out, cfg)
    return out, diag


def write_stage_hdf5(
    path: Path,
    candidates: Sequence[CandidateRecord],
    *,
    diagnostics: Mapping[str, Any],
) -> None:
    """Write stage HDF5 via the shared mass_derivation layout."""
    write_mass_stage_hdf5(
        path,
        candidates,
        stage_name=STAGE_NAME,
        diagnostics=diagnostics,
    )


def read_stage_hdf5(path: Path) -> tuple[list[CandidateRecord], dict[str, Any]]:
    return read_mass_stage_hdf5(path)


def _upstream_artifact(manifest: RunManifest, stage_name: str) -> Path:
    record = manifest.stages.get(stage_name)
    if record is None or not record.artifact_path:
        raise FileNotFoundError(
            f"upstream stage {stage_name!r} has no artifact_path on the run manifest"
        )
    path = Path(record.artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"upstream artifact missing: {path}")
    return path


def _load_upstream_candidates(manifest: RunManifest) -> list[CandidateRecord]:
    """Prefer ``joint_orbit_fit``; fall back to ``mass_derivation_refined``."""
    for stage_name in ("joint_orbit_fit", "mass_derivation_refined"):
        record = manifest.stages.get(stage_name)
        if record is not None and record.artifact_path:
            path = Path(record.artifact_path)
            if path.is_file():
                candidates, _meta = read_stage_hdf5(path)
                return candidates
    raise FileNotFoundError(
        "need joint_orbit_fit or mass_derivation_refined artifact on the run manifest"
    )


def format_companion_nature_report(
    diagnostics: CompanionNatureDiagnostics,
) -> str:
    """Full-detail stage report (caveman exemption for diagnostics)."""
    lines = [
        "=== companion_nature_likelihood funnel ===",
        f"  input_candidates: {diagnostics.n_input}",
        f"  weighted (none discarded): {diagnostics.n_weighted}",
        f"  tier_fast: {diagnostics.n_fast}",
        f"  tier_full: {diagnostics.n_full}",
        f"  channel_photometry: {diagnostics.n_photometry}",
        f"  channel_xp: {diagnostics.n_xp}",
        f"  channel_sb2: {diagnostics.n_sb2}",
        f"  no_channels: {diagnostics.n_no_channels}",
        f"  track_source: {diagnostics.track_source or 'analytic_fallback'}",
    ]
    if diagnostics.delta_bic_wd_vs_dark:
        arr = np.asarray(diagnostics.delta_bic_wd_vs_dark, dtype=np.float64)
        lines.append(
            f"  ΔBIC(WD−dark): median={float(np.median(arr)):.4f} "
            f"p90={float(np.percentile(arr, 90)):.4f}"
        )
    if diagnostics.age_diagnostic is not None:
        lines.append("  --- age-bin diagnostic ---")
        lines.append(f"  {diagnostics.age_diagnostic.message}")
        lines.append(f"  bin_edges_gyr: {diagnostics.age_diagnostic.bin_edges_gyr}")
        lines.append(f"  bin_counts: {diagnostics.age_diagnostic.bin_counts}")
        lines.append(
            f"  global_mean_weights: {diagnostics.age_diagnostic.global_mean_weights}"
        )
    lines.append("=== end companion_nature_likelihood funnel ===")
    return "\n".join(lines)


def write_diagnostic_artifacts(
    diagnostics: CompanionNatureDiagnostics,
    artifact_path: Path,
    config: PipelineConfig,
) -> list[Path]:
    """Write funnel + age-bin report and optional ΔBIC histogram beside the HDF5."""
    out_dir = artifact_path.parent / f"{artifact_path.stem}_diagnostics"
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "companion_nature_funnel.txt"
    path.write_text(format_companion_nature_report(diagnostics) + "\n", encoding="utf-8")
    written: list[Path] = [path]
    if diagnostics.age_diagnostic is not None:
        age_path = report_dir / "companion_nature_age_bins.json"
        age_path.write_text(
            json.dumps(diagnostics.age_diagnostic.as_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        written.append(age_path)

    diag_cfg = config.diagnostics
    if (
        diag_cfg.write_figures
        and diagnostics.delta_bic_wd_vs_dark
    ):
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_histogram(
            np.asarray(diagnostics.delta_bic_wd_vs_dark, dtype=np.float64),
            figures_dir / "delta_bic_wd_vs_dark.png",
            xlabel="ΔBIC (WD − dark); negative ⇒ data prefer WD",
            ylabel="count",
            title="Companion nature: ΔBIC(WD − dark)",
            dpi=int(diag_cfg.figure_dpi),
            max_bins=int(diag_cfg.histogram_max_bins),
            style=config.plotting,
        )
        if fig_path is not None:
            written.append(fig_path)
    return written


def run_companion_nature_likelihood(
    manifest: RunManifest,
    config: PipelineConfig,
    *,
    run_path: Path,
    force_rerun: bool = False,
    candidates: Sequence[CandidateRecord] | None = None,
    tracks: CoolingTrackTable | None = None,
) -> RunManifest:
    """Execute ``companion_nature_likelihood`` → HDF5 + age-bin diagnostic."""
    require_dr3_active_for_v1(config)
    spec = STAGE_REGISTRY[STAGE_NAME]
    plan = plan_stage(spec, manifest, config, force_rerun=force_rerun)
    artifact = stage_artifact_path(config, spec, run_id=manifest.run_id)

    if plan.action.name == "SKIP_CACHED":
        return manifest

    manifest = mark_stage_started(manifest, spec, config, force_rerun=force_rerun)
    save_run_manifest(manifest, run_path)

    if candidates is None:
        candidates = _load_upstream_candidates(manifest)

    updated, diagnostics = run_companion_nature_on_candidates(
        candidates, config, tracks=tracks
    )
    write_stage_hdf5(artifact, updated, diagnostics=diagnostics.as_dict())
    write_diagnostic_artifacts(diagnostics, artifact, config)

    manifest = mark_stage_finished(
        manifest,
        spec,
        status=StageStatus.COMPLETED,
        artifact_path=artifact,
    )
    save_run_manifest(manifest, run_path)
    return manifest
