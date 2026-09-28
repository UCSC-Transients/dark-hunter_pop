"""El-Badry et al. (2026) derived quantities and Simon et al. (2026) acceptance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    DustMapsConfig,
    GaiaBandPolynomialCoefficients,
    MainSequenceCutSpec,
    PipelineConfig,
    SampleSelectionFile,
    Simon2026ExclusionBreakdown,
)
from darkhunter_pop.dust_maps import (
    HEMISPHERE_NORTH,
    HEMISPHERE_SOUTH,
    USABLE_STATUSES,
    DustMapError,
    ExtinctionLookup,
    ExtinctionStatus,
)
from darkhunter_pop.janssens_mass import invert_mg_to_mass, load_janssens_table
from darkhunter_pop.physics_utils import (
    astrometric_mass_function,
    astrometric_mass_ratio_function,
    companion_mass_from_amrf,
    photocenter_a0_from_thiele_innes,
    spectroscopic_mass_function,
    invert_spectroscopic_minimum_companion_mass,
)
from darkhunter_pop.sample_selection import NotApplicable

SimonReason = Literal[
    "in_sample",
    "sb1_fails_significance",
    "astrometric_f2_above_max",
    "fainter_than_g_limit",
    "fails_m2_over_m1",
    "unclassified",
]


def is_main_sequence(
    mg_0: float,
    bp_rp_0: float,
    cut: MainSequenceCutSpec,
) -> bool:
    """``MG,0 > mg_floor`` OR ``MG,0 > intercept + slope * (BP-RP)_0``."""
    if not (np_isfinite(mg_0) and np_isfinite(bp_rp_0)):
        return False
    return mg_0 > cut.mg_floor or mg_0 > cut.cmd_intercept + cut.cmd_slope * bp_rp_0


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def paper_m1_from_mg(
    mg_0: float,
    *,
    main_sequence: bool,
    table_path: str | None,
) -> float | NotApplicable:
    """Janssens ``M̃1`` for main-sequence sources only (§8.2 / §15 Q10)."""
    if not main_sequence:
        return NotApplicable("evolved")
    table = load_janssens_table(table_path) if table_path else None
    result = invert_mg_to_mass(mg_0, table=table)
    if result.mass_msun is None:
        return NotApplicable(result.reason or "outside_janssens_range")
    return result.mass_msun


#: Row column carrying the per-source ``E(B-V)`` used for El-Badry 2026's CMD.
EBV_COLUMN = "ebv_elbadry2026"
#: Row column naming the ``ExtinctionStatus`` of that lookup (``ok``, ...).
EXTINCTION_STATUS_COLUMN = "extinction_status_elbadry2026"
#: Row column naming the frozen map that supplied ``E(B-V)``.
EXTINCTION_MAP_COLUMN = "extinction_map_elbadry2026"

_A_G_KEY = "a_g_over_e_bv"
_E_BP_RP_KEY = "e_bp_rp_over_e_bv"


def extinction_coefficients(spec: SampleSelectionFile) -> tuple[float, float]:
    """``(A_G/E(B-V), E(BP-RP)/E(B-V))`` from the frozen ``extinction.coefficients``."""
    if spec.extinction is None:
        raise ValueError(f"sample {spec.name!r} has no extinction block")
    coeffs = spec.extinction.coefficients
    missing = [k for k in (_A_G_KEY, _E_BP_RP_KEY) if k not in coeffs]
    if missing:
        raise ValueError(
            f"sample {spec.name!r} extinction.coefficients missing {missing}"
        )
    return float(coeffs[_A_G_KEY]), float(coeffs[_E_BP_RP_KEY])


def _babusiaux_k(
    coeffs: Sequence[float], colour0: np.ndarray, a0: np.ndarray
) -> np.ndarray:
    """``k_X`` of Gaia Collaboration, Babusiaux et al. (2018) Eq. 1 (vectorized)."""
    c1, c2, c3, c4, c5, c6, c7 = (float(c) for c in coeffs)
    return (
        c1
        + c2 * colour0
        + c3 * colour0**2
        + c4 * colour0**3
        + c5 * a0
        + c6 * a0**2
        + c7 * colour0 * a0
    )


def gaia_band_extinction_babusiaux2018(
    bp_rp_observed: np.ndarray,
    ebv: np.ndarray,
    *,
    r_v: float,
    coeffs: GaiaBandPolynomialCoefficients,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A_G, E(BP-RP), converged)`` per source from the Gaia law (#295).

    ``A0 = r_v · E(B-V)``; ``(BP-RP)_0`` solves
    ``(BP-RP)_0 = (BP-RP)_obs − (k_BP − k_RP)((BP-RP)_0, A0) · A0`` by
    fixed-point iteration from ``(BP-RP)_obs``. A source converges once its
    colour moves by less than ``coeffs.tolerance`` mag within
    ``coeffs.max_iterations`` steps. The iteration only contracts where
    ``|∂[(k_BP−k_RP)A0]/∂X| < 1``; for very red, highly extincted sources
    (roughly ``(BP-RP)_0 ≳ 2.5`` with ``A0 ≳ 3``, i.e. outside the law's
    fitted 3500–10000 K range) it diverges. Those sources get NaN and
    ``converged = False`` — the caller must report them, never pass them
    through. Non-finite inputs give NaN with ``converged = False``.
    Coefficients and limits come from config (see
    :class:`~darkhunter_pop.config_schema.GaiaBandPolynomialCoefficients`).
    """
    obs = np.asarray(bp_rp_observed, dtype=np.float64)
    a0 = float(r_v) * np.asarray(ebv, dtype=np.float64)
    a_g = np.full(obs.shape, np.nan)
    e_bp_rp = np.full(obs.shape, np.nan)
    converged = np.zeros(obs.shape, dtype=bool)
    active = np.isfinite(obs) & np.isfinite(a0)
    colour0 = np.where(active, obs, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(coeffs.max_iterations):
            idx = np.flatnonzero(active & ~converged)
            if idx.size == 0:
                break
            x, a = colour0[idx], a0[idx]
            k_diff = _babusiaux_k(coeffs.k_bp, x, a) - _babusiaux_k(coeffs.k_rp, x, a)
            updated = obs[idx] - k_diff * a
            colour0[idx] = updated
            done = np.abs(updated - x) < coeffs.tolerance
            converged[idx[done]] = True
            diverged = ~np.isfinite(updated)
            active[idx[diverged]] = False
        ok = np.flatnonzero(converged)
        x, a = colour0[ok], a0[ok]
        a_g[ok] = _babusiaux_k(coeffs.k_g, x, a) * a
        e_bp_rp[ok] = (_babusiaux_k(coeffs.k_bp, x, a) - _babusiaux_k(coeffs.k_rp, x, a)) * a
    return a_g, e_bp_rp, converged


def _finite_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("nan")


def _row_parallax(row: Mapping[str, Any]) -> float:
    """Parallax behind ``abs_g_mag``: ``parallax_mas``, else NSS ``parallax``.

    Mirrors ``sample_selection.candidate_to_selection_row``'s choice.
    """
    for key in ("parallax_mas", "parallax"):
        value = _finite_or_nan(row.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def deredden_elbadry2026_rows(
    rows: Sequence[Mapping[str, Any]],
    spec: SampleSelectionFile,
    lookup: ExtinctionLookup | None,
    *,
    dust_cfg: DustMapsConfig | None = None,
) -> list[dict[str, Any]]:
    """Apply El-Badry 2026's frozen extinction policy to raw ``abs_g_mag``/``bp_rp``.

    ``mg_0 = abs_g_mag − (A_G/E(B-V))·E(B-V)`` and
    ``bp_rp_0 = bp_rp − (E(BP-RP)/E(B-V))·E(B-V)``, with both coefficients from
    the frozen ``extinction.coefficients`` block and ``E(B-V)`` from the frozen
    hemisphere map via ``lookup``. Owned by El-Badry 2026 only: El-Badry 2024
    and Andrews never call this.

    A row already carrying :data:`EBV_COLUMN` (pre-computed, e.g. a fixture) is
    dereddened with that value and not looked up. Every other row is looked
    up; a row whose lookup is unusable (invalid parallax, no map pixel, bad
    position) gets ``mg_0``/``bp_rp_0`` = ``NotApplicable("extinction_<status>")``
    so the attrition waterfall reports it as not-applicable with that reason —
    never silently passed through undereddened.

    The E(B-V) → Gaia-band step follows
    ``dust_cfg.gaia_band_extinction.law`` (#295): ``paper_constant`` (the
    default and the reproduction path) uses the frozen coefficients above;
    ``babusiaux2018`` uses :func:`gaia_band_extinction_babusiaux2018` with
    ``A0 = dust_cfg.r_v · E(B-V)``. ``dust_cfg`` defaults to ``lookup.dust_cfg``;
    with neither (fixture rows carrying a pre-computed E(B-V)), the frozen
    constants apply.

    Raises ``DustMapError`` if any row needs a lookup and ``lookup`` is ``None``.
    """
    if spec.extinction is None:
        raise ValueError(f"sample {spec.name!r} has no extinction block")
    a_g, e_bp_rp = extinction_coefficients(spec)
    if dust_cfg is None and lookup is not None:
        dust_cfg = lookup.dust_cfg
    out = [dict(row) for row in rows]
    need = [i for i, row in enumerate(out) if EBV_COLUMN not in row]
    if need:
        if lookup is None:
            raise DustMapError(
                f"sample {spec.name!r}: {len(need)} rows need an E(B-V) lookup for "
                "the frozen extinction policy but no ExtinctionLookup was supplied "
                "(evaluate through SampleSelectionRegistry with a PipelineConfig, "
                f"or pre-compute {EBV_COLUMN!r}) — refusing to evaluate "
                "undereddened photometry"
            )
        ebv, status, hemi = lookup.ebv_for(
            [int(out[i]["source_id"]) for i in need],
            [_finite_or_nan(out[i].get("ra_deg")) for i in need],
            [_finite_or_nan(out[i].get("dec_deg")) for i in need],
            [_row_parallax(out[i]) for i in need],
        )
        legs = {
            HEMISPHERE_NORTH: spec.extinction.north.map,
            HEMISPHERE_SOUTH: spec.extinction.south.map,
        }
        for j, i in enumerate(need):
            code = ExtinctionStatus(int(status[j]))
            out[i][EXTINCTION_STATUS_COLUMN] = code.name.lower()
            out[i][EXTINCTION_MAP_COLUMN] = legs.get(int(hemi[j]))
            if int(code) in USABLE_STATUSES:
                out[i][EBV_COLUMN] = float(ebv[j])
            else:
                out[i][EBV_COLUMN] = NotApplicable(f"extinction_{code.name.lower()}")
    usable: list[int] = []
    for i, row in enumerate(out):
        ebv_value = row[EBV_COLUMN]
        row.setdefault(EXTINCTION_STATUS_COLUMN, "precomputed")
        if isinstance(ebv_value, NotApplicable):
            row["mg_0"] = ebv_value
            row["bp_rp_0"] = ebv_value
            continue
        usable.append(i)
    if not usable:
        return out
    ebv_arr = np.array([float(out[i][EBV_COLUMN]) for i in usable], dtype=np.float64)
    band = dust_cfg.gaia_band_extinction if dust_cfg is not None else None
    if band is not None and band.law == "babusiaux2018":
        assert band.babusiaux2018 is not None and dust_cfg is not None  # schema-validated
        assert dust_cfg.r_v is not None
        bp_rp_arr = np.array([_finite_or_nan(out[i].get("bp_rp")) for i in usable])
        a_g_arr, e_bp_rp_arr, converged = gaia_band_extinction_babusiaux2018(
            bp_rp_arr, ebv_arr, r_v=dust_cfg.r_v, coeffs=band.babusiaux2018
        )
        # Finite inputs the law could not solve are reported, never passed through.
        unsolved = ~converged & np.isfinite(bp_rp_arr) & np.isfinite(ebv_arr)
    else:
        a_g_arr = a_g * ebv_arr
        e_bp_rp_arr = e_bp_rp * ebv_arr
        unsolved = np.zeros(len(usable), dtype=bool)
    for j, i in enumerate(usable):
        row = out[i]
        if unsolved[j]:
            reason = NotApplicable("extinction_gaia_law_nonconvergent")
            row[EXTINCTION_STATUS_COLUMN] = "gaia_law_nonconvergent"
            row["mg_0"] = reason
            row["bp_rp_0"] = reason
            continue
        row["mg_0"] = _finite_or_nan(row.get("abs_g_mag")) - float(a_g_arr[j])
        row["bp_rp_0"] = _finite_or_nan(row.get("bp_rp")) - float(e_bp_rp_arr[j])
    return out


def build_elbadry2026_extinction_lookup(
    spec: SampleSelectionFile,
    config: PipelineConfig,
    *,
    repo: Path | None = None,
) -> ExtinctionLookup | None:
    """Lazy ``E(B-V)`` provider for El-Badry 2026's frozen extinction block.

    Returns ``None`` when the spec has no extinction block. No map file is
    opened until a cache miss forces a query. ``paths.data_root`` anchors the
    configured map paths and the per-source cache.
    """
    if spec.extinction is None:
        return None
    root = repo if repo is not None else repo_root()
    data_root = Path(config.paths.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    return ExtinctionLookup(
        spec.extinction,
        config.sample_selection.dust_maps,
        data_root=data_root,
        cache_tag=spec.name,
    )


def extinction_status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Funnel summary: how many rows got each extinction outcome."""
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get(EXTINCTION_STATUS_COLUMN)
        if status is None:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def enrich_elbadry2026_row(
    row: Mapping[str, Any],
    spec: SampleSelectionFile,
) -> dict[str, Any]:
    """Catalog-level derived columns used by the frozen cut chain.

    Expects ``mg_0``/``bp_rp_0`` already dereddened by
    :func:`deredden_elbadry2026_rows` (the registry path does this). A
    ``NotApplicable`` ``mg_0`` (unusable extinction lookup) propagates to
    ``main_sequence`` and the mass columns with its reason.
    """
    out = dict(row)
    ms_cut = spec.main_sequence_cut
    mg_na = out.get("mg_0")
    if ms_cut is not None and isinstance(mg_na, NotApplicable):
        out["main_sequence"] = mg_na
        out["m1_tilde_msun"] = mg_na
        out["m2_tilde_msun"] = mg_na
        out["amrf"] = mg_na
        ms_cut = None
    if ms_cut is not None:
        mg_0 = float(out.get("mg_0", float("nan")))
        color = float(out.get("bp_rp_0", float("nan")))
        out["main_sequence"] = is_main_sequence(mg_0, color, ms_cut)
        table = None if spec.primary_mass is None else spec.primary_mass.table
        m1 = paper_m1_from_mg(
            mg_0, main_sequence=bool(out["main_sequence"]), table_path=table
        )
        out["m1_tilde_msun"] = m1
        if isinstance(m1, NotApplicable):
            out["m2_tilde_msun"] = NotApplicable(m1.reason)
            out["amrf"] = NotApplicable(m1.reason)
        else:
            a0 = out.get("a0_mas")
            if a0 is None and all(
                key in out
                for key in (
                    "a_thiele_innes",
                    "b_thiele_innes",
                    "f_thiele_innes",
                    "g_thiele_innes",
                )
            ):
                a0 = float(
                    photocenter_a0_from_thiele_innes(
                        out["a_thiele_innes"],
                        out["b_thiele_innes"],
                        out["f_thiele_innes"],
                        out["g_thiele_innes"],
                    )
                )
                out["a0_mas"] = a0
                out["a0_method_used"] = "photocenter_a0_from_thiele_innes"
            if a0 is not None:
                plx = float(out["parallax"])
                period = float(out.get("period_day", out.get("period")))
                mf = float(astrometric_mass_function(a0, plx, period))
                out["m_f_msun"] = mf
                amrf = float(astrometric_mass_ratio_function(a0, plx, period, m1))
                out["amrf"] = amrf
                out["m2_tilde_msun"] = float(companion_mass_from_amrf(amrf, m1))
    period = out.get("period_day", out.get("period"))
    k1 = out.get("k1_kms", out.get("semi_amplitude_primary"))
    ecc = out.get("eccentricity", 0.0)
    if period is not None and k1 is not None:
        fm = float(spectroscopic_mass_function(period, k1, ecc))
        out["fm_msun"] = fm
        m1_sb = out.get("m1_tilde_msun")
        if isinstance(m1_sb, float):
            out["m2_min_msun"] = float(
                invert_spectroscopic_minimum_companion_mass(fm, m1_sb)
            )
        elif isinstance(m1_sb, NotApplicable):
            out["m2_min_msun"] = m1_sb
    k1_err = out.get("k1_error", out.get("semi_amplitude_primary_error"))
    if k1 is not None and k1_err not in (None, 0):
        out["k1_significance"] = float(k1) / float(k1_err)
    return out


def load_simon2026_orbital(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Twenty Simon et al. (2026) sources with DR3 orbital solutions (§8.9)."""
    table_path = (
        Path(path)
        if path is not None
        else repo_root() / "config/selections/external/simon2026_orbital.yaml"
    )
    if not table_path.is_absolute():
        table_path = repo_root() / table_path
    raw = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    return list(raw["data"])


def classify_simon2026_row(
    row: Mapping[str, Any],
    *,
    in_sample: bool,
    g_mag_faint_limit: float,
    goodness_of_fit_max: float,
    k1_significance_min: float,
    m2_over_m1_min: float,
    astrometric_types: Sequence[str],
    spectroscopic_types: Sequence[str],
    elbadry_m2_over_m1: float | None = None,
) -> SimonReason:
    """First matching exclusion reason; ``in_sample`` wins if already selected.

    ``elbadry_m2_over_m1`` is El-Badry 2026's own ``M̃2/M̃1`` for this source
    (Janssens ``M̃1``, AMRF ``M̃2``). The paper's ``fails_m2_over_m1`` exclusion
    refers to that quantity, so it is used when supplied. When it is ``None``
    the Simon catalog ``m2_over_m1`` column is used as a fallback. That column
    is a different mass estimate (Simon et al. 2026 Table 1), and it passes
    ``1864406790238257536``, which the paper excludes on its own M̃ (#281).
    """
    if in_sample:
        return "in_sample"
    sol = str(row.get("nss_solution_type", ""))
    g_mag = float(row["g_mag"])
    if g_mag >= g_mag_faint_limit:
        return "fainter_than_g_limit"
    if sol in spectroscopic_types:
        sig = float(row["significance"])
        if sig <= k1_significance_min:
            return "sb1_fails_significance"
        return "unclassified"
    if sol in astrometric_types:
        f2 = float(row["goodness_of_fit"])
        if f2 > goodness_of_fit_max:
            return "astrometric_f2_above_max"
        ratio = (
            float(elbadry_m2_over_m1)
            if elbadry_m2_over_m1 is not None
            else float(row["m2_over_m1"])
        )
        if ratio <= m2_over_m1_min:
            return "fails_m2_over_m1"
        return "unclassified"
    return "unclassified"


def simon2026_exclusion_breakdown(
    rows: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[int],
    spec: SampleSelectionFile,
    *,
    elbadry_m2_over_m1_by_source: Mapping[int, float] | None = None,
) -> dict[str, int]:
    """Reproduce the published 5 / 2 / 1 / 1 split (§8.9).

    ``elbadry_m2_over_m1_by_source`` maps ``source_id`` to El-Badry 2026's own
    ``M̃2/M̃1`` (e.g. ``m2_tilde_msun / m1_tilde_msun`` from
    :func:`enrich_elbadry2026_row`). Sources missing from it fall back to the
    Simon catalog ratio; see :func:`classify_simon2026_row`.
    """
    tests = spec.acceptance_tests
    if tests is None or tests.simon2026_exclusion_breakdown is None:
        raise ValueError("elbadry2026.yaml missing simon2026_exclusion_breakdown")
    expected: Simon2026ExclusionBreakdown = tests.simon2026_exclusion_breakdown
    astro = next(b for b in (spec.branches or []) if b.id == "astrometric")
    specb = next(b for b in (spec.branches or []) if b.id == "spectroscopic")
    astro_types = astro.parent_query.dr3.solution_types
    spec_types = specb.parent_query.dr3.solution_types
    sub1 = next(s for s in (astro.subsamples or []) if s.id == "primary_ns_bh")
    g_lim = float(sub1.cuts[-1].parameters["g_mag_faint_limit"])
    f2_max = float(
        next(c for c in sub1.cuts if c.id == "goodness_of_fit").parameters[
            "goodness_of_fit_max"
        ]
    )
    m2m1 = float(
        next(c for c in sub1.cuts if c.id == "m2_over_m1").parameters["m2_over_m1_min"]
    )
    sig_cut = next(c for c in (specb.cuts or []) if c.id == "k1_significance")
    sig_min = float(sig_cut.parameters["k1_significance_min"])
    sample = set(int(s) for s in sample_ids)
    ratios = {
        int(k): float(v) for k, v in (elbadry_m2_over_m1_by_source or {}).items()
    }
    counts = {
        "in_sample": 0,
        "sb1_fails_significance": 0,
        "astrometric_f2_above_max": 0,
        "fainter_than_g_limit": 0,
        "fails_m2_over_m1": 0,
        "unclassified": 0,
    }
    for row in rows:
        reason = classify_simon2026_row(
            row,
            in_sample=int(row["source_id"]) in sample,
            g_mag_faint_limit=g_lim,
            goodness_of_fit_max=f2_max,
            k1_significance_min=sig_min,
            m2_over_m1_min=m2m1,
            astrometric_types=astro_types,
            spectroscopic_types=spec_types,
            elbadry_m2_over_m1=ratios.get(int(row["source_id"])),
        )
        counts[reason] += 1
    del expected
    return counts
