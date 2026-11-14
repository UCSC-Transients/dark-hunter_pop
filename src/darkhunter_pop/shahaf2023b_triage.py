"""Shahaf et al. (2023b) AMRF class-III triage, reimplemented (#682, RUNG5_SPEC §2.7.4, CONTINUATION_PLAN §15 Q2).

Shahaf, Bashi, Mazeh et al. (2023, MNRAS 518, 2991) classify each Gaia DR3 astrometric orbit by its astrometric
mass-ratio function (AMRF, their Eq. 1) against two limiting curves of the luminous-star mass M1: ``A_MS`` (a single
main-sequence companion) and ``A_TR`` (a close, equal-mass main-sequence pair). An orbit above ``A_TR`` cannot be
explained by luminous companions: class III, a probable WD / NS / BH dark companion. The published "highly probable
class-III" catalogue (their Table 2, 177 rows) is what El-Badry et al. (2024) used as its S23 NS-candidate pool. A
catalogue cannot be applied to a mock; this module is the method, so the same triage runs on real DR3 rows and on
mock rows.

The definition (every threshold, the Monte-Carlo settings, the decision rule and the authors' supplementary
limiting-curve table) is the frozen file :data:`DEFAULT_DEFINITION_PATH`. Steps, as in the paper:

1. **Parent** (§3.1): solution type, ``bit_index`` (all orbital parameters fitted), ``gaiadr3.binary_masses``
   ``m1_ref = IsocLum`` with combination method ``Orbital+M1`` / ``AstroSpectroSB1+M1``, the Halbwachs et al. (2022)
   cuts on Δe, ϖ/σϖ and a0/σa0, the Thiele–Innes relative-error cut (Eq. 4) and P < 1000 d
   (:func:`parent_cut_masks`).
2. **AMRF** (Eq. 1) from the NSS ``a0`` (Thiele–Innes), ϖ, P and the IsocLum M1 (:func:`amrf`).
3. **Class probabilities** (§3.3, Eqs. 5–6): ``N`` Monte-Carlo draws of (ϖ, A, B, F, G, P, e) from the NSS
   covariance and of M1 from the ``binary_masses`` 16/84 % interval; ``Pr III = (r + 1)/(N + 1)`` with ``r`` the
   draws above ``A_TR(M1)`` (:func:`class_probabilities`).
4. **Decision** (§4): Benjamini & Hochberg (1995) false-discovery rate α = 10 % over the clean sample
   (:func:`benjamini_hochberg`), or, where that sample is not the real one (a mock), the paper's stated equivalent
   cut ``n_below ≤ 16`` of ``10^5`` (:func:`fixed_threshold_class3`).

The steps the paper does not spell out (M1 draw shape, eccentricity domain, off-grid M1, the p-value convention)
are set to the reading that reproduces the authors' own per-source Table 1; the alternatives are listed in the
definition file under ``readings`` and measured in ``docs/gate682/README.md``.

Terminology: luminous star (M1), dark companion (M2); "real DR3" is the archive, "mock" a forward-model draw.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from darkhunter_pop.physics_utils import astrometric_mass_ratio_function, photocenter_a0_from_thiele_innes

#: Frozen definition (paper thresholds, MC settings, decision rule, supplementary limiting curves).
DEFAULT_DEFINITION_PATH: Final[str] = "config/selections/external/shahaf2023b_amrf_triage.yaml"

#: Orbit parameters the Monte Carlo draws, in this order (real NSS and mock ParameterSet names alike).
ORBIT_PARAMETERS: Final[tuple[str, ...]] = (
    "parallax", "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes", "period", "eccentricity",
)
_I_PLX, _I_A, _I_B, _I_F, _I_G, _I_P, _I_E = range(len(ORBIT_PARAMETERS))

#: Ordered parent-cut ids of :func:`parent_cut_masks` (§3.1).
PARENT_CUT_IDS: Final[tuple[str, ...]] = (
    "solution_type", "bit_index", "binary_masses", "eccentricity_error", "parallax_snr", "significance",
    "thiele_innes_relative_error", "period",
)

#: Row columns written by :func:`attach_class3_columns` (prefixed: never collide with catalogue columns).
COLUMN_AMRF: Final[str] = "shahaf2023b_amrf_reimpl"
COLUMN_N_DRAWS: Final[str] = "shahaf2023b_n_draws"
COLUMN_N_BELOW: Final[str] = "shahaf2023b_n_below_atr"
COLUMN_P_III: Final[str] = "shahaf2023b_p_iii"
COLUMN_CLASS3: Final[str] = "shahaf2023b_class3_reimpl"
COLUMN_M1_SOURCE: Final[str] = "shahaf2023b_m1_source"
COLUMN_PARENT_STATUS: Final[str] = "shahaf2023b_parent_status"
#: Mock-row provenance values (``mock_selection_rows``): the M1 the mock carries and the unemulated parent cut.
MOCK_M1_SOURCE: Final[str] = "tag10_pipeline_m1"
MOCK_PARENT_STATUS: Final[str] = "binary_masses_not_emulated"


# ---------------------------------------------------------------------------------------------------------------
# Definition file
# ---------------------------------------------------------------------------------------------------------------


class TriageParent(BaseModel):
    """§3.1 parent cuts (paper values)."""

    model_config = ConfigDict(extra="forbid")

    solution_types: list[str] = Field(..., min_length=1)
    bit_index: list[int] = Field(..., min_length=1)
    binary_masses_m1_ref: str
    binary_masses_combination_methods: list[str] = Field(..., min_length=1)
    eccentricity_error_ln_period_slope: float
    eccentricity_error_offset: float
    parallax_snr_period_coeff: float = Field(..., gt=0)
    significance_period_coeff: float = Field(..., gt=0)
    ti_relative_error_sq_sum_max: float = Field(..., gt=0)
    period_max_day: float = Field(..., gt=0)
    expected_counts: dict[str, int] = Field(default_factory=dict)


class TriageAmrf(BaseModel):
    """Eq. (1) unit convention."""

    model_config = ConfigDict(extra="forbid")

    year_day: float = Field(..., gt=0)


class TriageMonteCarlo(BaseModel):
    """§3.3 Monte-Carlo settings."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(..., ge=1000)
    chunk_draws: int = Field(..., ge=100)
    seed: int = Field(..., ge=0)
    eccentricity_domain: tuple[float, float]
    m1_draw: Literal["two_half_normals", "gaussian_max_side"]
    limiting_curve_interpolation: Literal["linear"]
    off_grid: Literal["clip_m1", "clamp", "not_classified"]
    probability_estimator: Literal["plus_one"]

    @model_validator(mode="after")
    def _domain(self) -> TriageMonteCarlo:
        lo, hi = self.eccentricity_domain
        if not lo < hi:
            raise ValueError("eccentricity_domain must be (low, high) with low < high")
        return self


class TriageDecision(BaseModel):
    """§4 decision rule."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["fdr_bh"]
    alpha: float = Field(..., gt=0, lt=1)
    p_value: Literal["n_below_plus_one", "one_minus_pr_iii"]
    fixed_n_below_max: int = Field(..., ge=0)


class LimitingCurves(BaseModel):
    """§2.3 supplementary lookup table ``[m1_msun, a_ms, a_tr]``."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[list[float]] = Field(..., min_length=2)

    @model_validator(mode="after")
    def _table(self) -> LimitingCurves:
        if self.columns != ["m1_msun", "a_ms", "a_tr"]:
            raise ValueError("limiting_curves.columns must be [m1_msun, a_ms, a_tr]")
        arr = np.asarray(self.rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
            raise ValueError("limiting_curves.rows must be finite [m1, a_ms, a_tr] triples")
        if not np.all(np.diff(arr[:, 0]) > 0):
            raise ValueError("limiting_curves m1 must be strictly increasing")
        return self


class TriageDefinition(BaseModel):
    """The frozen ``shahaf2023b_amrf_triage.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    name: Literal["shahaf2023b_amrf_triage"]
    provenance: dict[str, Any]
    parent: TriageParent
    amrf: TriageAmrf
    monte_carlo: TriageMonteCarlo
    decision: TriageDecision
    readings: dict[str, Any] = Field(default_factory=dict)
    limiting_curves: LimitingCurves

    @property
    def curve_m1(self) -> NDArray[np.float64]:
        return np.asarray(self.limiting_curves.rows, dtype=np.float64)[:, 0]

    @property
    def curve_a_ms(self) -> NDArray[np.float64]:
        return np.asarray(self.limiting_curves.rows, dtype=np.float64)[:, 1]

    @property
    def curve_a_tr(self) -> NDArray[np.float64]:
        return np.asarray(self.limiting_curves.rows, dtype=np.float64)[:, 2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path | None) -> Path:
    p = Path(path if path is not None else DEFAULT_DEFINITION_PATH)
    return p if p.is_absolute() else _repo_root() / p


def load_definition(path: str | Path | None = None) -> TriageDefinition:
    """Load and validate the frozen triage definition (default :data:`DEFAULT_DEFINITION_PATH`)."""
    raw = yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))
    return TriageDefinition.model_validate(raw)


def definition_sha256(path: str | Path | None = None) -> str:
    """SHA-256 of the definition file bytes (provenance for reports and mock columns)."""
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Step 1: parent cuts (§3.1)
# ---------------------------------------------------------------------------------------------------------------


def _col(columns: Mapping[str, ArrayLike], name: str) -> NDArray[np.float64]:
    return np.asarray(columns[name], dtype=np.float64)


def parent_cut_masks(
    columns: Mapping[str, ArrayLike], definition: TriageDefinition, *, binary_masses: bool = True
) -> dict[str, NDArray[np.bool_]]:
    """Per-cut pass masks of the §3.1 parent, in :data:`PARENT_CUT_IDS` order (each cut alone, not cumulative).

    ``columns`` holds NSS-named arrays: ``nss_solution_type``, ``bit_index``, ``period``, ``eccentricity_error``,
    ``parallax``, ``parallax_error``, ``significance`` (a0/σa0) and ``{a,b,f,g}_thiele_innes`` (+ ``_error``); with
    ``binary_masses=True`` also ``binary_masses_m1_ref`` and ``binary_masses_combination_method`` (empty / None for a
    source absent from ``gaiadr3.binary_masses``). ``binary_masses=False`` omits that cut (the caller then owns the
    substitute and must say so; a mock has no ``binary_masses``). A missing ``bit_index`` key is likewise omitted
    (mock fits always fit every orbital parameter). NaN inputs fail their cut.
    """
    par = definition.parent
    sol = np.asarray(columns["nss_solution_type"]).astype(str)
    period = _col(columns, "period")
    out: dict[str, NDArray[np.bool_]] = {"solution_type": np.isin(sol, par.solution_types)}
    if "bit_index" in columns:
        bits = np.asarray(columns["bit_index"])
        out["bit_index"] = np.isin(bits, np.asarray(par.bit_index))
    if binary_masses:
        ref = np.asarray(columns["binary_masses_m1_ref"]).astype(str)
        method = np.asarray(columns["binary_masses_combination_method"]).astype(str)
        out["binary_masses"] = (ref == par.binary_masses_m1_ref) & np.isin(
            method, par.binary_masses_combination_methods
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        ln_p = np.log(period)
        out["eccentricity_error"] = _col(columns, "eccentricity_error") < (
            par.eccentricity_error_ln_period_slope * ln_p + par.eccentricity_error_offset
        )
        out["parallax_snr"] = _col(columns, "parallax") / _col(columns, "parallax_error") > (
            par.parallax_snr_period_coeff / period
        )
        out["significance"] = _col(columns, "significance") > par.significance_period_coeff / np.sqrt(period)
        rel_sq = np.zeros(period.shape, dtype=np.float64)
        for x in "abfg":
            rel_sq = rel_sq + (_col(columns, f"{x}_thiele_innes_error") / _col(columns, f"{x}_thiele_innes")) ** 2
        out["thiele_innes_relative_error"] = rel_sq <= par.ti_relative_error_sq_sum_max
        out["period"] = period < par.period_max_day
    return {k: np.asarray(out[k], dtype=bool) for k in PARENT_CUT_IDS if k in out}


def parent_mask(masks: Mapping[str, NDArray[np.bool_]]) -> NDArray[np.bool_]:
    """AND of :func:`parent_cut_masks`."""
    arrays = list(masks.values())
    acc = np.ones(arrays[0].shape, dtype=bool)
    for m in arrays:
        acc &= m
    return acc


# ---------------------------------------------------------------------------------------------------------------
# Step 2: AMRF (Eq. 1)
# ---------------------------------------------------------------------------------------------------------------


def amrf(
    a_ti: ArrayLike, b_ti: ArrayLike, f_ti: ArrayLike, g_ti: ArrayLike,
    parallax_mas: ArrayLike, period_day: ArrayLike, m1_msun: ArrayLike, definition: TriageDefinition,
) -> NDArray[np.float64]:
    """Eq. (1) ``A = (a0/ϖ) M1^{-1/3} (P/yr)^{-2/3}`` with ``a0`` from the Thiele–Innes constants (NaN if undefined)."""
    if definition.amrf.year_day != 365.25:
        # physics_utils.astrometric_mass_function fixes the Julian year; a different convention needs its own path.
        raise ValueError(f"amrf.year_day {definition.amrf.year_day} is not the physics_utils Julian year 365.25")
    a0 = photocenter_a0_from_thiele_innes(a_ti, b_ti, f_ti, g_ti)
    return astrometric_mass_ratio_function(a0, parallax_mas, period_day, m1_msun)


def limiting_values(
    m1_msun: ArrayLike, definition: TriageDefinition
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(A_MS, A_TR)`` at ``m1_msun`` from the §2.3 table (linear); off-grid per ``monte_carlo.off_grid``.

    ``clip_m1`` / ``clamp``: the end values (``clip_m1`` also clips the M1 draw itself before the AMRF, in
    :func:`class_probabilities`). ``not_classified``: NaN outside the tabulated M1 range, so the draw counts in no
    class (it is below ``A_TR``).
    """
    m1 = np.asarray(m1_msun, dtype=np.float64)
    grid = definition.curve_m1
    if definition.monte_carlo.off_grid in ("clamp", "clip_m1"):
        return np.interp(m1, grid, definition.curve_a_ms), np.interp(m1, grid, definition.curve_a_tr)
    return (
        np.interp(m1, grid, definition.curve_a_ms, left=np.nan, right=np.nan),
        np.interp(m1, grid, definition.curve_a_tr, left=np.nan, right=np.nan),
    )


# ---------------------------------------------------------------------------------------------------------------
# Step 3: class probabilities (§3.3)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassCounts:
    """Monte-Carlo outcome for one source.

    ``n_draws`` valid draws were classified; ``n_class3`` lie above ``A_TR``, ``n_class2`` between ``A_MS`` and
    ``A_TR``; ``n_below = n_draws − n_class3``. ``complete`` is False when the run stopped early because
    ``n_below`` exceeded the caller's cap: the source is then certainly not class III under that cap, and
    :attr:`p_iii` is only an upper bound.
    """

    n_draws: int
    n_class2: int
    n_class3: int
    complete: bool

    @property
    def n_below(self) -> int:
        return self.n_draws - self.n_class3

    @property
    def p_iii(self) -> float:
        """Eq. (6): ``(r + 1)/(N + 1)``."""
        return (self.n_class3 + 1) / (self.n_draws + 1)

    @property
    def p_ii(self) -> float:
        return (self.n_class2 + 1) / (self.n_draws + 1)

    def p_value(self, convention: str) -> float:
        """Monte-Carlo p-value of the class-III null (``decision.p_value``)."""
        if convention == "n_below_plus_one":
            return (self.n_below + 1) / (self.n_draws + 1)
        if convention == "one_minus_pr_iii":
            return 1.0 - self.p_iii
        raise ValueError(f"unknown p_value convention {convention!r}")


def orbit_mean_covariance(parameter_set: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(mean, cov)`` of :data:`ORBIT_PARAMETERS` from a real NSS or mock ``ParameterSet`` (by name)."""
    names = list(parameter_set.names)
    missing = [n for n in ORBIT_PARAMETERS if n not in names]
    if missing:
        raise KeyError(f"orbit ParameterSet lacks {missing}")
    idx = [names.index(n) for n in ORBIT_PARAMETERS]
    mean = np.asarray(parameter_set.values, dtype=np.float64)[idx]
    cov = np.asarray(parameter_set.covariance, dtype=np.float64)[np.ix_(idx, idx)]
    return mean, cov


def _sqrt_cov(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # PSD to rounding (validated upstream): symmetric square root, negative rounding eigenvalues clipped.
        w, v = np.linalg.eigh(0.5 * (cov + cov.T))
        return v * np.sqrt(np.clip(w, 0.0, None))


def _m1_draws(z: NDArray[np.float64], m1: float, m1_lower: float, m1_upper: float, kind: str) -> NDArray[np.float64]:
    lo = m1 - m1_lower
    hi = m1_upper - m1
    if kind == "two_half_normals":
        return m1 + np.where(z < 0.0, lo, hi) * z
    if kind == "gaussian_max_side":
        return m1 + max(lo, hi) * z
    raise ValueError(f"unknown m1_draw {kind!r}")


def class_probabilities(
    mean: ArrayLike,
    cov: ArrayLike,
    m1_msun: float,
    m1_lower_msun: float,
    m1_upper_msun: float,
    *,
    source_id: int,
    definition: TriageDefinition,
    n_below_stop: int | None = None,
) -> ClassCounts:
    """§3.3 Monte Carlo for one source: class-II / class-III draw counts.

    ``mean`` / ``cov`` are over :data:`ORBIT_PARAMETERS` (:func:`orbit_mean_covariance`); M1 is the IsocLum
    ``m1`` with its 16 / 84 % ``m1_lower`` / ``m1_upper`` (a symmetric ±σ for a mock TAG10 M1). Draws with
    eccentricity outside ``monte_carlo.eccentricity_domain`` are rejected and replaced, so ``n_draws`` valid draws
    are always classified. The orbit and M1 streams are independent children of ``(seed, source_id)`` and are
    consumed sequentially, so the result does not depend on ``chunk_draws``.

    ``n_below_stop``: stop as soon as more than this many draws lie at or below ``A_TR`` (the source can then no
    longer pass a cut ``n_below ≤ n_below_stop``); the returned counts are partial (``complete=False``).
    """
    mc = definition.monte_carlo
    mu = np.asarray(mean, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    if mu.shape != (len(ORBIT_PARAMETERS),) or c.shape != (len(ORBIT_PARAMETERS),) * 2:
        raise ValueError("mean / cov must cover ORBIT_PARAMETERS")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(c))):
        raise ValueError("non-finite orbit mean or covariance")
    for v in (m1_msun, m1_lower_msun, m1_upper_msun):
        if not math.isfinite(v):
            raise ValueError("non-finite M1 interval")
    root = _sqrt_cov(c)
    orbit_ss, m1_ss = np.random.SeedSequence([mc.seed, int(source_id)]).spawn(2)
    orbit_rng = np.random.default_rng(orbit_ss)
    m1_rng = np.random.default_rng(m1_ss)
    e_lo, e_hi = mc.eccentricity_domain
    grid_lo, grid_hi = float(definition.curve_m1[0]), float(definition.curve_m1[-1])
    n_target = mc.n_draws
    n_done = n_c2 = n_c3 = 0
    while n_done < n_target:
        x = mu + orbit_rng.standard_normal((mc.chunk_draws, len(ORBIT_PARAMETERS))) @ root.T
        ecc = x[:, _I_E]
        x = x[(ecc >= e_lo) & (ecc < e_hi)][: n_target - n_done]
        if x.shape[0] == 0:
            continue
        m1 = _m1_draws(m1_rng.standard_normal(x.shape[0]), m1_msun, m1_lower_msun, m1_upper_msun, mc.m1_draw)
        if mc.off_grid == "clip_m1":
            m1 = np.clip(m1, grid_lo, grid_hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = amrf(x[:, _I_A], x[:, _I_B], x[:, _I_F], x[:, _I_G], x[:, _I_PLX], x[:, _I_P], m1, definition)
            a_ms, a_tr = limiting_values(m1, definition)
            above_tr = a > a_tr
            n_c3 += int(np.count_nonzero(above_tr))
            n_c2 += int(np.count_nonzero((a > a_ms) & ~above_tr))
        n_done += x.shape[0]
        if n_below_stop is not None and n_done - n_c3 > n_below_stop and n_done < n_target:
            return ClassCounts(n_draws=n_done, n_class2=n_c2, n_class3=n_c3, complete=False)
    return ClassCounts(n_draws=n_done, n_class2=n_c2, n_class3=n_c3, complete=True)


# ---------------------------------------------------------------------------------------------------------------
# Step 4: decision (§4)
# ---------------------------------------------------------------------------------------------------------------


def benjamini_hochberg(p_values: ArrayLike, alpha: float) -> NDArray[np.bool_]:
    """Benjamini & Hochberg (1995) step-up selection at FDR ``alpha`` (ties keep input order; NaN never selected)."""
    p = np.asarray(p_values, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(p.size)
    selected = np.zeros(m, dtype=bool)
    if m == 0:
        return selected
    order = np.argsort(np.where(finite, p, np.inf), kind="stable")
    ranked = np.where(finite, p, np.inf)[order]
    passing = np.nonzero(ranked <= np.arange(1, m + 1) * alpha / m)[0]
    if passing.size:
        selected[order[: passing[-1] + 1]] = True
    return selected


def fixed_threshold_class3(counts: ClassCounts, definition: TriageDefinition) -> bool:
    """The paper's stated equivalent of its FDR cut: ``n_below ≤ decision.fixed_n_below_max`` of a complete run."""
    if not counts.complete:
        return False
    return counts.n_below <= definition.decision.fixed_n_below_max


# ---------------------------------------------------------------------------------------------------------------
# Rows: the class-III column on selection rows (real DR3 or mock)
# ---------------------------------------------------------------------------------------------------------------


def attach_class3_columns(
    rows: Sequence[Mapping[str, Any]],
    definition: TriageDefinition,
    *,
    orbit_key: str,
    m1_key: str,
    m1_sigma_key: str,
    m1_source: str,
    parent_status: str,
) -> list[dict[str, Any]]:
    """Copy ``rows`` with the reimplemented class-III columns (fixed-threshold decision, §4 equivalent).

    For mock rows (and any sample that is not the real DR3 clean sample, where the BH step cannot be rerun): each
    row is classified with :func:`class_probabilities` (early stop at ``decision.fixed_n_below_max``) and
    :func:`fixed_threshold_class3`. ``orbit_key`` names the row's orbit ``ParameterSet`` (mock:
    ``mock_orbit_solution``); M1 is ``row[m1_key] ± row[m1_sigma_key]`` (symmetric interval). Rows failing the
    orbit parent cuts (all §3.1 cuts except ``binary_masses``, which the caller reports in ``parent_status``) get
    ``COLUMN_CLASS3 = False``; rows without an orbit covariance or finite M1 get a ``NotApplicable`` reason, never a
    guess. ``source_id`` seeds the per-row stream.
    """
    from darkhunter_pop.luminous_covariance import excluded_reason, is_excluded_status
    from darkhunter_pop.sample_selection import NotApplicable

    cut_columns = (
        "nss_solution_type", "period", "eccentricity_error", "parallax", "parallax_error", "significance",
        *(f"{x}_thiele_innes{s}" for x in "abfg" for s in ("", "_error")),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        new = dict(row)
        new[COLUMN_M1_SOURCE] = m1_source
        new[COLUMN_PARENT_STATUS] = parent_status
        if any(k not in row for k in cut_columns):
            new[COLUMN_CLASS3] = NotApplicable("shahaf2023b_missing_parent_columns")
            out.append(new)
            continue
        cols = {k: np.asarray([row[k]]) for k in cut_columns}
        in_parent = bool(parent_mask(parent_cut_masks(cols, definition, binary_masses=False))[0])
        m1 = float(row.get(m1_key, float("nan")))
        sig = float(row.get(m1_sigma_key, float("nan")))
        if math.isfinite(m1) and m1 > 0.0:
            new[COLUMN_AMRF] = float(
                amrf(row["a_thiele_innes"], row["b_thiele_innes"], row["f_thiele_innes"], row["g_thiele_innes"],
                     row["parallax"], row["period"], m1, definition)
            )
        if not in_parent:
            new[COLUMN_CLASS3] = False
            out.append(new)
            continue
        orbit = row.get(orbit_key)
        if orbit is None:
            status = row.get("mock_orbit_cov_status")
            if is_excluded_status(status):  # #704 luminous pre-filter: no covariance by design, never a stand-in
                new[COLUMN_CLASS3] = NotApplicable(excluded_reason(str(status)))
            else:
                new[COLUMN_CLASS3] = NotApplicable("shahaf2023b_no_orbit_covariance")
            out.append(new)
            continue
        if not (math.isfinite(m1) and m1 > 0.0 and math.isfinite(sig) and sig > 0.0):
            new[COLUMN_CLASS3] = NotApplicable("shahaf2023b_no_m1")
            out.append(new)
            continue
        mean, cov = orbit_mean_covariance(orbit)
        counts = class_probabilities(
            mean, cov, m1, m1 - sig, m1 + sig, source_id=int(row["source_id"]), definition=definition,
            n_below_stop=definition.decision.fixed_n_below_max,
        )
        new[COLUMN_N_DRAWS] = counts.n_draws
        new[COLUMN_N_BELOW] = counts.n_below
        new[COLUMN_P_III] = counts.p_iii
        new[COLUMN_CLASS3] = fixed_threshold_class3(counts, definition)
        out.append(new)
    return out
