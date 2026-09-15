"""Adapter: ``dark-hunter_sed`` Path-2 photometric SED model comparison → pop extras.

Reads the per-model JSON summaries ``dark-hunter_sed`` writes under
``output/phot_sed/`` (snapshot target ``mass_derivation.phot_sed_root``, default
``data/phot_sed/``) and fills the photometric-evidence hooks
``companion_nature.phot_chi2_dark_key`` / ``phot_chi2_wd_key`` /
``phot_chi2_other_key`` / ``phot_n_data_key`` on ``CandidateRecord.extras``, so
``companion_nature.photometry_channel`` scores real dynesty evidence instead of
its analytic magnitude–mass relations.

**Pop only ever reads the JSON files.** Nothing here imports or runs
``darkhunter_sed``; no SED fit is launched.

Model → hypothesis mapping (confirmed upstream; not to be re-derived)
--------------------------------------------------------------------
======================  ================  ==========================================
``dark-hunter_sed``     pop hypothesis    meaning
======================  ================  ==========================================
``1star``               ``dark``          single star; no luminous secondary
``wd``                  ``WD``            luminous normal star + white dwarf
``2star``               ``other``         **coeval** binary
======================  ================  ==========================================

**Documented limitation.** Pop's ``other`` class is formally broader than
``2star``: it covers a luminous non-degenerate secondary of *any* age, while
``2star`` is specifically coeval. A non-coeval luminous secondary therefore has
no dedicated evidence model in this channel. That is a known gap in the
evidence, not a mapping error, and it is recorded rather than corrected here.

BIC → chi2 conversion
---------------------
``dark-hunter_sed.phot_sed_fit.bic_from_max_likelihood`` defines

    ``BIC = k ln n - 2 ln L_max``

with ``k = n_free`` and ``n = n_data`` both reported in the summary. Pop's
``companion_nature`` hooks are chi2-shaped and pop re-applies its own penalty
(``companion_nature.n_params_*`` × ``ln n``), so the quantity recovered here is

    ``chi2_equivalent = BIC - n_free ln(n_data) = -2 ln L_max``

which is exactly the form the ticket (issue #197) prescribes. Two consequences
are deliberate and documented rather than silently absorbed:

* ``-2 ln L_max`` carries the Gaussian normalisation ``Σ ln(2π σ_i²)`` of the
  upstream likelihood, and the upstream ``σ`` includes a *fitted* ``sigma_int``.
  The normalisation is common to every model fitted to the same photometry rows,
  so it cancels in every ΔBIC and in ``photometric_weights_from_bic`` (a softmax
  over differences). It is not subtracted, because doing so would require
  re-deriving the upstream likelihood.
* Pop's penalty ``n_params_*`` replaces upstream's ``n_free``. The parameter
  counts are pop's own config choice; this adapter does not touch them.

``-2 ln L_max`` may be negative, while ``companion_nature`` requires
non-negative chi2. A single common offset is therefore added to all three
hypotheses when needed (recorded as ``chi2_offset``). A common offset shifts
every class's BIC equally, so every ΔBIC and every weight is unchanged.

What is recorded, and why
-------------------------
Bad photometry points are not fully resolved upstream and ΔBIC depends on both
chi2 and ``n_data``, so every candidate carries, per model: ``n_data``,
``n_free``, ``bic``, ``ln_z``, the recovered chi2, the fitted ``sigma_int`` when
present, and whichever photometry-cleaning settings the summary records
(:data:`CLEANING_SUMMARY_KEYS`). Today's upstream summaries record **no**
cleaning settings at all; that absence is reported per model as
``cleaning_settings_absent`` rather than assumed benign.

Only hypotheses with a real summary are ever written. A missing ``wd`` fit is
missing — never "no WD". The chi2 triple is written only when all three
hypotheses are present *and* agree on ``n_data``; otherwise the photometry
channel keeps its analytic fallback and the candidate is labelled
``analytic_fallback``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import PipelineConfig
from darkhunter_pop.schemas import CandidateRecord

SCHEMA_VERSION: Final = 1

#: ``dark-hunter_sed`` ``--model`` value → pop photometric hypothesis.
MODEL_TO_HYPOTHESIS: Final[dict[str, str]] = {
    "1star": "dark",
    "wd": "WD",
    "2star": "other",
}
#: Hypotheses the photometry channel needs before precomputed chi2 can be used.
REQUIRED_HYPOTHESES: Final[tuple[str, ...]] = ("dark", "WD", "other")

#: Evidence-provenance tags written to ``extras``.
PROVENANCE_PHOT_SED: Final = "phot_sed"
PROVENANCE_ANALYTIC_FALLBACK: Final = "analytic_fallback"
#: Candidate already carried a chi2 triple from somewhere other than this
#: adapter (fixtures, or a future channel). Neither real ``phot_sed`` evidence
#: nor the analytic fallback, so it is labelled and counted separately.
PROVENANCE_PRECOMPUTED_EXTRAS: Final = "precomputed_extras"

#: Extras keys owned by this adapter (chi2/n_data keys are config-owned).
PROVENANCE_EXTRAS_KEY: Final = "phot_sed_evidence_provenance"
EVIDENCE_EXTRAS_KEY: Final = "phot_sed_evidence"

#: Photometry-cleaning / error-treatment settings copied from a summary when the
#: summary happens to record them. ``dark-hunter_sed`` does not write these
#: today; their absence is reported, never imputed.
CLEANING_SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "phot_err_floor",
    "gaia_phot_err_floor",
    "phot_outlier_sigma",
    "phot_bands_excluded",
    "phot_source",
    "stride",
    "ir_bb",
)
#: Fitted error-inflation parameter, read out of ``best_theta`` when present.
SIGMA_INT_PARAM: Final = "sigma_int"


@dataclass(frozen=True)
class PhotSedModelSummary:
    """One ``dark-hunter_sed`` Path-2 model summary, converted for pop.

    Parameters
    ----------
    model:
        Upstream ``--model`` value (``1star`` / ``2star`` / ``wd``).
    hypothesis:
        Pop photometric hypothesis per :data:`MODEL_TO_HYPOTHESIS`.
    source_id:
        Gaia DR3 source id the summary claims (``gaia_id`` in the JSON).
    chi2:
        ``bic - n_free ln(n_data)`` = ``-2 ln L_max`` (see module docstring).
    bic, ln_z, n_data, n_free:
        As reported upstream. ``ln_z`` is ``logz`` (``None`` when absent).
    sigma_int:
        Fitted intrinsic-scatter parameter from ``best_theta``, when present.
    cleaning:
        Photometry-cleaning settings the summary records (usually empty).
    path:
        Absolute path the summary was read from.

    Limits
    ------
    Construction is via :func:`parse_phot_sed_summary`; a summary missing
    ``bic``, ``n_free`` or ``n_data`` cannot be converted and yields ``None``
    there rather than a partially filled record.
    """

    model: str
    hypothesis: str
    source_id: int
    chi2: float
    bic: float
    ln_z: float | None
    n_data: int
    n_free: int
    sigma_int: float | None
    cleaning: dict[str, Any]
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hypothesis": self.hypothesis,
            "source_id": self.source_id,
            "chi2": self.chi2,
            "bic": self.bic,
            "ln_z": self.ln_z,
            "n_data": self.n_data,
            "n_free": self.n_free,
            "sigma_int": self.sigma_int,
            "cleaning": dict(self.cleaning),
            "cleaning_settings_absent": not self.cleaning,
            "path": self.path,
        }


@dataclass(frozen=True)
class PhotSedEvidence:
    """Per-candidate outcome of the adapter (never a discard).

    Parameters
    ----------
    source_id:
        Candidate source id the lookup was performed for.
    summaries:
        Converted summaries keyed by pop hypothesis; only models with a real,
        readable, source-id-matching summary appear.
    chi2_offset:
        Common offset added to every chi2 so the minimum is non-negative.
    n_data:
        Shared ``n_data`` when the present models agree, else ``None``.
    usable:
        True when all of :data:`REQUIRED_HYPOTHESES` are present and agree on
        ``n_data`` — i.e. when the chi2 triple is written to ``extras``.
    notes:
        Machine-readable reasons (missing / malformed / mismatched / disagreeing
        ``n_data``), one per affected model where applicable.
    """

    source_id: int
    summaries: dict[str, PhotSedModelSummary] = field(default_factory=dict)
    chi2_offset: float = 0.0
    n_data: int | None = None
    usable: bool = False
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_id": self.source_id,
            "usable": self.usable,
            "n_data": self.n_data,
            "chi2_offset": self.chi2_offset,
            "models": {
                hypothesis: summary.as_dict()
                for hypothesis, summary in sorted(self.summaries.items())
            },
            "notes": list(self.notes),
            "other_class_limitation": (
                "2star is a coeval binary; pop's 'other' also covers non-coeval "
                "luminous secondaries, which have no dedicated evidence model"
            ),
        }


def chi2_from_bic(*, bic: float, n_free: int, n_data: int) -> float:
    """Recover ``-2 ln L_max`` from ``dark-hunter_sed``'s BIC.

    Parameters
    ----------
    bic:
        ``BIC = n_free ln(n_data) - 2 ln L_max`` as written upstream by
        ``darkhunter_sed.phot_sed_fit.bic_from_max_likelihood``.
    n_free:
        Upstream free-parameter count ``k`` (must be >= 1).
    n_data:
        Upstream photometry-row count ``n`` (must be >= 1).

    Returns
    -------
    float
        ``bic - n_free * ln(n_data)``.

    Limits
    ------
    Raises ``ValueError`` on ``n_free < 1`` or ``n_data < 1``, matching the
    upstream guard. The result is a chi2 *equivalent*: it retains the Gaussian
    normalisation of the upstream likelihood, which cancels in every ΔBIC. It is
    not guaranteed non-negative; see :func:`_offset_chi2`.
    """
    if n_data < 1:
        raise ValueError("n_data must be >= 1 for the BIC → chi2 conversion")
    if n_free < 1:
        raise ValueError("n_free must be >= 1 for the BIC → chi2 conversion")
    return float(bic) - float(n_free) * math.log(float(n_data))


def phot_sed_root(config: PipelineConfig) -> Path | None:
    """Absolute snapshot root for Path-2 summaries, or ``None`` when disabled."""
    root_spec = config.mass_derivation.phot_sed_root
    if root_spec is None or str(root_spec).strip() == "":
        return None
    root = Path(root_spec).expanduser()
    if not root.is_absolute():
        root = repo_root() / root
    return root


def resolve_phot_sed_summary_path(
    config: PipelineConfig, source_id: int, model: str
) -> Path | None:
    """Templated path for one ``(source_id, model)`` summary.

    Returns ``None`` when ``mass_derivation.phot_sed_root`` is unset. Does not
    check existence. ``model`` must be a key of :data:`MODEL_TO_HYPOTHESIS`.
    """
    if model not in MODEL_TO_HYPOTHESIS:
        raise KeyError(f"unknown dark-hunter_sed phot_sed model: {model!r}")
    root = phot_sed_root(config)
    if root is None:
        return None
    name = config.mass_derivation.phot_sed_filename_template.format(
        source_id=source_id, model=model
    )
    return root / name


def _as_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _as_int(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value


def parse_phot_sed_summary(
    payload: Mapping[str, Any],
    *,
    model: str,
    source_id: int,
    path: Path,
) -> PhotSedModelSummary | None:
    """Convert one loaded summary mapping, or ``None`` when unusable.

    ``None`` is returned (never a partial record, never an imputed value) when
    the payload's ``gaia_id`` disagrees with ``source_id``, or when ``bic`` /
    ``n_free`` / ``n_data`` are absent, non-numeric or out of range.
    """
    claimed = payload.get("gaia_id", payload.get("source_id"))
    if claimed is not None:
        claimed_id = _as_int(claimed)
        if claimed_id is None or claimed_id != int(source_id):
            return None
    bic = _as_float(payload.get("bic"))
    n_free = _as_int(payload.get("n_free"))
    n_data = _as_int(payload.get("n_data"))
    if bic is None or n_free is None or n_data is None:
        return None
    if n_free < 1 or n_data < 1:
        return None
    chi2 = chi2_from_bic(bic=bic, n_free=n_free, n_data=n_data)
    if not math.isfinite(chi2):
        return None
    best_theta = payload.get("best_theta")
    sigma_int = (
        _as_float(best_theta.get(SIGMA_INT_PARAM))
        if isinstance(best_theta, Mapping)
        else None
    )
    cleaning = {
        key: payload[key] for key in CLEANING_SUMMARY_KEYS if key in payload
    }
    return PhotSedModelSummary(
        model=model,
        hypothesis=MODEL_TO_HYPOTHESIS[model],
        source_id=int(source_id),
        chi2=chi2,
        bic=bic,
        ln_z=_as_float(payload.get("logz")),
        n_data=n_data,
        n_free=n_free,
        sigma_int=sigma_int,
        cleaning=cleaning,
        path=str(path),
    )


def load_phot_sed_summary(
    path: Path, *, model: str, source_id: int
) -> tuple[PhotSedModelSummary | None, str | None]:
    """Read one summary file. Returns ``(summary, note)``; note names any refusal.

    Notes are machine-readable: ``missing:<model>``, ``malformed:<model>``
    (unreadable JSON, or JSON that is not an object) and ``unusable:<model>``
    (source-id mismatch, or missing/invalid ``bic``/``n_free``/``n_data``). A
    refusal never falls back to another format or another file.
    """
    if not path.is_file():
        return None, f"missing:{model}"
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, f"malformed:{model}"
    if not isinstance(payload, Mapping):
        return None, f"malformed:{model}"
    summary = parse_phot_sed_summary(
        payload, model=model, source_id=source_id, path=path
    )
    if summary is None:
        return None, f"unusable:{model}"
    return summary, None


def _offset_chi2(summaries: Mapping[str, PhotSedModelSummary]) -> float:
    """Common offset making every recovered chi2 non-negative (0 when already)."""
    if not summaries:
        return 0.0
    lowest = min(s.chi2 for s in summaries.values())
    return float(-lowest) if lowest < 0.0 else 0.0


def load_phot_sed_evidence(config: PipelineConfig, source_id: int) -> PhotSedEvidence:
    """Load every available Path-2 summary for one source.

    Tolerates any subset present — none, one, two or all three. The returned
    evidence is ``usable`` only when all three hypotheses are present and report
    the same ``n_data``; a disagreement is reported (``n_data_mismatch``) rather
    than resolved by picking one, because pop applies a single shared ``ln n``
    penalty across the three hypotheses.
    """
    summaries: dict[str, PhotSedModelSummary] = {}
    notes: list[str] = []
    if phot_sed_root(config) is None:
        return PhotSedEvidence(source_id=int(source_id), notes=("phot_sed_root_unset",))
    for model in MODEL_TO_HYPOTHESIS:
        path = resolve_phot_sed_summary_path(config, source_id, model)
        if path is None:  # pragma: no cover - root checked above
            continue
        summary, note = load_phot_sed_summary(path, model=model, source_id=source_id)
        if note is not None:
            notes.append(note)
        if summary is not None:
            summaries[summary.hypothesis] = summary
            if not summary.cleaning:
                notes.append(f"cleaning_settings_absent:{model}")

    n_data_values = {s.n_data for s in summaries.values()}
    complete = all(h in summaries for h in REQUIRED_HYPOTHESES)
    consistent = len(n_data_values) == 1
    if complete and not consistent:
        notes.append("n_data_mismatch")
    offset = _offset_chi2(summaries) if (complete and consistent) else 0.0
    if offset > 0.0:
        notes.append("chi2_offset_applied")
    return PhotSedEvidence(
        source_id=int(source_id),
        summaries=summaries,
        chi2_offset=offset,
        n_data=next(iter(n_data_values)) if consistent and n_data_values else None,
        usable=complete and consistent,
        notes=tuple(notes),
    )


def _has_precomputed_triple(candidate: CandidateRecord, config: PipelineConfig) -> bool:
    cfg = config.companion_nature
    extras = candidate.extras
    return all(
        extras.get(key) is not None
        for key in (
            cfg.phot_chi2_dark_key,
            cfg.phot_chi2_wd_key,
            cfg.phot_chi2_other_key,
        )
    )


def attach_phot_sed_evidence(
    candidate: CandidateRecord,
    config: PipelineConfig,
    *,
    evidence: PhotSedEvidence | None = None,
) -> tuple[CandidateRecord, PhotSedEvidence]:
    """Attach Path-2 evidence + provenance to one candidate.

    Parameters
    ----------
    candidate:
        Input record; never dropped or otherwise filtered.
    config:
        Supplies ``mass_derivation.phot_sed_root`` /
        ``phot_sed_filename_template`` and the ``companion_nature.phot_*_key``
        extras names.
    evidence:
        Pre-loaded evidence (tests, or a batch that already did the I/O).
        Defaults to :func:`load_phot_sed_evidence`.

    Returns
    -------
    tuple
        The updated record and the evidence used.

    Limits
    ------
    The chi2/n_data extras are written **only** when the evidence is ``usable``;
    a partial set of models leaves them absent so ``photometry_channel`` keeps
    its analytic fallback. A candidate that already carries a chi2 triple from
    elsewhere is left untouched and tagged
    :data:`PROVENANCE_PRECOMPUTED_EXTRAS`. Weights, ΔBIC and tiers are unchanged
    for any candidate without usable ``phot_sed`` evidence; the only additions
    are the two labelling extras.
    """
    ev = evidence if evidence is not None else load_phot_sed_evidence(config, candidate.source_id)
    cfg = config.companion_nature
    extras = dict(candidate.extras)

    if ev.usable:
        offset = float(ev.chi2_offset)
        extras[cfg.phot_chi2_dark_key] = float(ev.summaries["dark"].chi2) + offset
        extras[cfg.phot_chi2_wd_key] = float(ev.summaries["WD"].chi2) + offset
        extras[cfg.phot_chi2_other_key] = float(ev.summaries["other"].chi2) + offset
        extras[cfg.phot_n_data_key] = int(ev.n_data or 0)
        provenance = PROVENANCE_PHOT_SED
    elif _has_precomputed_triple(candidate, config):
        provenance = PROVENANCE_PRECOMPUTED_EXTRAS
    else:
        provenance = PROVENANCE_ANALYTIC_FALLBACK

    extras[PROVENANCE_EXTRAS_KEY] = provenance
    extras[EVIDENCE_EXTRAS_KEY] = ev.as_dict()
    return candidate.model_copy(update={"extras": extras}), ev


def evidence_provenance(candidate: CandidateRecord) -> str:
    """Provenance tag attached by :func:`attach_phot_sed_evidence`.

    Falls back to :data:`PROVENANCE_ANALYTIC_FALLBACK` for records the adapter
    never saw (older artifacts), so callers never have to special-case ``None``.
    """
    raw = candidate.extras.get(PROVENANCE_EXTRAS_KEY)
    return str(raw) if isinstance(raw, str) and raw else PROVENANCE_ANALYTIC_FALLBACK
