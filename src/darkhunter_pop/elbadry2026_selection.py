"""El-Badry et al. (2026) derived quantities and Simon et al. (2026) acceptance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    MainSequenceCutSpec,
    SampleSelectionFile,
    Simon2026ExclusionBreakdown,
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


def enrich_elbadry2026_row(
    row: Mapping[str, Any],
    spec: SampleSelectionFile,
) -> dict[str, Any]:
    """Catalog-level derived columns used by the frozen cut chain."""
    out = dict(row)
    ms_cut = spec.main_sequence_cut
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
) -> SimonReason:
    """First matching exclusion reason; ``in_sample`` wins if already selected."""
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
        ratio = float(row["m2_over_m1"])
        if ratio <= m2_over_m1_min:
            return "fails_m2_over_m1"
        return "unclassified"
    return "unclassified"


def simon2026_exclusion_breakdown(
    rows: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[int],
    spec: SampleSelectionFile,
) -> dict[str, int]:
    """Reproduce the published 5 / 2 / 1 / 1 split (§8.9)."""
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
        )
        counts[reason] += 1
    del expected
    return counts
