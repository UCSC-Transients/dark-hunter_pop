"""Andrews et al. (2022) reproduction: the ATF selection notebook, exactly (#296).

The PI (Ryan Foley, co-author of Andrews, Taggart & Foley 2022) supplied the
notebook that actually selected the published sample. Its code is the spec for
``andrews2022`` **reproduction** mode; this module re-implements it, including
its implicit rejections, and precomputes the ``andrews_atf_*`` columns the
frozen cut chain (``config/selections/andrews2022.yaml``, schema_version 3)
tests. Reference copy (gitignored, not committed):
``data/reference/andrews2022_ATF_sample_selection.ipynb``.

Notebook → function map
-----------------------
* ``get_random_samples`` (cell 7) → :func:`build_notebook_covariance`,
  :func:`gate_covariance`, :func:`draw_notebook_samples`. The 12×12
  covariance is built from the ``*_error`` columns and ``corr_vec`` (strict
  lower triangle, row-major) **without** ``bit_index``; the source is rejected
  whenever ``scipy.stats.multivariate_normal`` refuses the matrix (NaN input,
  non-PSD, numerically singular). No eigenvalue flooring.
* ``find_massive`` (cell 7) → :func:`run_pass1`: ``a0`` from Thiele–Innes,
  ``m_f = a0³ yr² / (P² ϖ³)``, ``M2`` at fixed ``M1`` via a root in a fixed
  bracket. The source is rejected if the root is missing for **any** draw
  (negative-parallax draw, ``m_f`` beyond the bracket, non-finite ``m_f``).
  ``P(M2 > threshold)`` uses **all** draws as denominator.
* ``plot_system`` (cell 17) / ``calc_M2`` (cell 34) → :func:`run_pass2`:
  refined ``M1`` (UCO Lick spectroscopic mass, else Apsis FLAME, else uniform),
  a fresh draw set, ``brentq`` per draw in the pass-2 bracket (any failure
  rejects), and ``mean(M2)`` / ``std(M2)`` for the 3σ cut.
* CMD inputs (cell 17) → :func:`notebook_cmd_quantities`: absolute
  magnitudes from the NSS parallax point value, no extinction.

Thresholds and switches live in the frozen selection file; nothing here is a
choosable number. Column ownership: ``andrews_atf_*`` are Andrews-owned and
are never aliased into El-Badry columns.

Limitations
-----------
* The notebook is unseeded; this module is seeded per source
  (``random_seed ^ (source_id & 0x7FFFFFFF)``), so sources whose
  ``P(M2 > 1.4)`` sits within MC noise of the threshold can legitimately
  differ from the published run.
* Pass 1 solves the cubic in closed-form-seeded Newton
  (:func:`~darkhunter_pop.physics_utils.invert_astrometric_companion_mass`)
  rather than per-draw ``brentq``; the failure set is identical by
  construction (monotone ``M2³/(M1+M2)²`` on a bracket), the root agrees to
  ``brentq``'s ``xtol``. Pass 2 (``M1`` per draw, possibly ``<= 0``) uses
  ``scipy.optimize.brentq`` per draw, as the notebook does.
* ``brentq`` on a non-finite ``m_f`` raises in the installed SciPy; the
  notebook's 2022 SciPy is not pinned. Such draws count as failures here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

from darkhunter_pop.config_schema import (
    AtfNotebookPrimaryMassSpec,
    PipelineConfig,
    ReproductionProcedureSpec,
    SampleCut,
    SampleSelectionFile,
)
from darkhunter_pop.constants import JULIAN_YEAR_DAYS, PARALLAX_MAS_AT_10PC
from darkhunter_pop.physics_utils import invert_astrometric_companion_mass

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION: Final[int] = 1
COLUMN_PREFIX: Final[str] = "andrews_atf_"
M1_SOURCE_LICK: Final[str] = "lick"
M1_SOURCE_FLAME: Final[str] = "flame"
M1_SOURCE_UNIFORM: Final[str] = "uniform"
_SEED_MASK: Final[int] = 0x7FFFFFFF

# Covariance-gate failure classes (scipy messages mapped to stable labels).
COV_FAIL_CORR_VEC_SHORT: Final[str] = "corr_vec_too_short"
COV_FAIL_NONFINITE: Final[str] = "nonfinite_input"
COV_FAIL_SINGULAR: Final[str] = "singular"
COV_FAIL_NOT_PSD: Final[str] = "not_positive_semidefinite"
COV_FAIL_OTHER: Final[str] = "other"

FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "andrews_atf_p_m2_above",
    "andrews_atf_p_m2_above_valid_draws",
    "andrews_atf_cov_min_eig_ratio",
    "andrews_atf_goodness_of_fit",
    "andrews_atf_logg",
    "andrews_atf_mass_flame",
    "andrews_atf_abs_g_mag",
    "andrews_atf_bp_rp",
    "andrews_atf_m1_mean_msun",
    "andrews_atf_m2_mean_msun",
    "andrews_atf_m2_std_msun",
)
INT_COLUMNS: Final[tuple[str, ...]] = (
    "andrews_atf_pass1_n_root_failed",
    "andrews_atf_pass1_n_mf_below_bracket",
    "andrews_atf_pass1_n_mf_above_bracket",
    "andrews_atf_pass1_n_nonfinite_mf",
    "andrews_atf_pass1_n_negative_parallax",
    "andrews_atf_pass2_n_root_failed",
)
# Tri-state booleans: stored as int8 (1 true, 0 false, -1 not computed).
BOOL_COLUMNS: Final[tuple[str, ...]] = (
    "andrews_atf_covariance_ok",
    "andrews_atf_pass1_root_ok",
    "andrews_atf_pass2_root_ok",
)
STR_COLUMNS: Final[tuple[str, ...]] = (
    "andrews_atf_covariance_failure",
    "andrews_atf_m1_source",
)
ALL_COLUMNS: Final[tuple[str, ...]] = (
    FLOAT_COLUMNS + INT_COLUMNS + BOOL_COLUMNS + STR_COLUMNS
)


class AtfProcedureError(ValueError):
    """Raised when a selection file's reproduction procedure is unusable."""


@dataclass(frozen=True)
class AtfSourceInputs:
    """Per-source inputs for the ATF notebook procedure.

    ``means`` / ``errors`` follow ``covariance.parameter_order``; missing values
    are NaN (they fail the covariance gate, as in the notebook). ``corr_vec``
    is the archive vector as stored (any length; shorter than the 66 entries
    the notebook indexes is a covariance failure). Photometry / ``logg`` /
    FLAME are NaN when absent.
    """

    source_id: int
    means: NDArray[np.floating]
    errors: NDArray[np.floating]
    corr_vec: NDArray[np.floating]
    goodness_of_fit: float
    mass_flame: float
    logg: float
    g_mag: float
    bp_mag: float
    rp_mag: float


def notebook_float(value: Any, *, float32_decimal_roundtrip: bool) -> float:
    """Return ``value`` as float64, optionally via its float32 shortest decimal.

    The notebook read the archive CSV, whose float32 fields are printed as the
    shortest decimal that round-trips the float32 (``0.1925101``); parsing that
    string gives a float64 that differs from the exact float32 value by up to
    ~1e-8 relative. ``None`` / masked / non-numeric become NaN.
    """
    if value is None or value is np.ma.masked:
        return float("nan")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(as_float) or not float32_decimal_roundtrip:
        return as_float
    return float(str(np.float32(as_float)))


def build_notebook_covariance(
    errors: NDArray[np.floating], corr_vec: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Notebook ``get_random_samples`` covariance: ``C_ij = ρ_N σ_i σ_j``.

    ``corr_vec`` fills the strict lower triangle in the notebook's loop order
    (``for i in range(n): for j in range(i)``), i.e. ``np.tril_indices(n, -1)``;
    the diagonal is ``σ_i²``. ``bit_index`` is deliberately not consulted.

    Raises
    ------
    AtfProcedureError
        When ``corr_vec`` has fewer than ``n (n-1) / 2`` entries (the notebook
        raised ``IndexError`` there, which its bare ``except`` turned into a
        rejection).
    """
    sig = np.asarray(errors, dtype=np.float64)
    n = sig.size
    rho = np.asarray(corr_vec, dtype=np.float64).ravel()
    n_off = n * (n - 1) // 2
    if rho.size < n_off:
        raise AtfProcedureError(COV_FAIL_CORR_VEC_SHORT)
    cov = np.ones((n, n), dtype=np.float64)
    rows, cols = np.tril_indices(n, k=-1)
    vals = rho[:n_off] * sig[rows] * sig[cols]
    cov[rows, cols] = vals
    cov[cols, rows] = vals
    cov[np.arange(n), np.arange(n)] = sig * sig
    return cov


def classify_gate_failure(message: str) -> str:
    """Map a ``scipy.stats.multivariate_normal`` error message to a stable label."""
    text = message.lower()
    if "inf" in text and "nan" in text:
        return COV_FAIL_NONFINITE
    if "semidefinite" in text and "positive definite" not in text:
        return COV_FAIL_NOT_PSD
    if "singular" in text or "positive definite" in text:
        return COV_FAIL_SINGULAR
    return COV_FAIL_OTHER


def gate_covariance(
    means: NDArray[np.floating], cov: NDArray[np.floating]
) -> tuple[Any | None, str | None]:
    """Construct the notebook's ``multivariate_normal(means, cov)`` or report why not.

    Returns ``(frozen_distribution, None)`` on success and
    ``(None, failure_label)`` when SciPy raises (``allow_singular=False``:
    NaN input, a negative eigenvalue beyond SciPy's tolerance, or an eigenvalue
    below ``1e6 · eps · max|λ|``). Nothing is floored or repaired.
    """
    try:
        dist = multivariate_normal(mean=means, cov=cov)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, classify_gate_failure(str(exc))
    return dist, None


def covariance_min_eig_ratio(cov: NDArray[np.floating]) -> float:
    """``min λ / max |λ|`` of a finite covariance (diagnostic only); NaN otherwise."""
    if not np.all(np.isfinite(cov)):
        return float("nan")
    evals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    scale = float(np.max(np.abs(evals)))
    if scale <= 0.0:
        return float("nan")
    return float(np.min(evals) / scale)


def draw_notebook_samples(
    dist: Any, n_draws: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """``dist.rvs(n_draws)`` with an explicit generator; shape ``(n_draws, 12)``."""
    samples = np.asarray(dist.rvs(size=int(n_draws), random_state=rng), dtype=np.float64)
    return samples.reshape(int(n_draws), -1)


def notebook_mass_function(
    samples: NDArray[np.floating], parameter_order: Sequence[str]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Notebook ``a0`` and ``m_f`` per draw; returns ``(m_f, parallax_draws)``.

    ``u = (A²+B²+F²+G²)/2``, ``v = AG − BF``, ``a0 = √(u + √(u² − v²))`` with no
    clamping (a rounding-negative radicand gives NaN, as in the notebook), and
    ``m_f = a0³ · yr² / (P² ϖ³)`` in the samples' own sign (a negative-parallax
    draw gives a negative ``m_f``).
    """
    index = {name: i for i, name in enumerate(parameter_order)}
    a = samples[:, index["a_thiele_innes"]]
    b = samples[:, index["b_thiele_innes"]]
    f = samples[:, index["f_thiele_innes"]]
    g = samples[:, index["g_thiele_innes"]]
    plx = samples[:, index["parallax"]]
    period = samples[:, index["period"]]
    u = (a * a + b * b + f * f + g * g) / 2.0
    v = a * g - b * f
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a0 = np.sqrt(u + np.sqrt(u * u - v * v))
        m_f = a0**3 * JULIAN_YEAR_DAYS**2 / (period**2 * plx**3)
    return m_f, plx


def _g_of_m2(m2: float, m1: float) -> float:
    """``M2³ / (M1 + M2)²`` — the function whose root the notebook finds."""
    return m2**3 / (m1 + m2) ** 2


def solve_m2_fixed_m1(
    m_f: NDArray[np.floating],
    *,
    m1_msun: float,
    bracket_msun: tuple[float, float],
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Vectorized equivalent of ``brentq(M2³/(M1+M2)² − m_f, lo, hi)`` per draw.

    For ``M1 > 0`` the function is monotone on ``M2 >= 0``, so ``brentq``
    succeeds exactly when ``m_f`` is finite and ``g(lo) <= m_f <= g(hi)``;
    the root is then the unique positive root of the cubic. Returns
    ``(m2, ok)`` with NaN where ``ok`` is false.
    """
    if m1_msun <= 0.0:
        raise AtfProcedureError("pass-1 M1 must be > 0 for the vectorized root")
    lo, hi = float(bracket_msun[0]), float(bracket_msun[1])
    mf = np.asarray(m_f, dtype=np.float64)
    g_lo = _g_of_m2(lo, m1_msun)
    g_hi = _g_of_m2(hi, m1_msun)
    ok = np.isfinite(mf) & (mf >= g_lo) & (mf <= g_hi)
    m2 = np.full(mf.shape, np.nan, dtype=np.float64)
    at_lo = ok & (mf == g_lo)
    m2[at_lo] = lo
    interior = ok & ~at_lo & (mf > 0.0)
    if np.any(interior):
        m2[interior] = invert_astrometric_companion_mass(m1_msun, mf[interior], 0.0)
    return m2, ok


def solve_m2_brentq(
    m_f: NDArray[np.floating],
    m1_msun: NDArray[np.floating],
    *,
    bracket_msun: tuple[float, float],
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Per-draw ``scipy.optimize.brentq`` exactly as notebook cells 17/34 call it.

    Returns ``(m2, ok)``; a draw whose ``brentq`` raises is NaN / ``False``.
    Used for pass 2, where ``M1`` varies per draw and may be ``<= 0``.
    """
    lo, hi = float(bracket_msun[0]), float(bracket_msun[1])
    mf = np.asarray(m_f, dtype=np.float64)
    m1 = np.asarray(m1_msun, dtype=np.float64)
    m2 = np.full(mf.shape, np.nan, dtype=np.float64)
    ok = np.zeros(mf.shape, dtype=bool)

    def func(x: float, target: float, primary: float) -> float:
        total = primary + x
        return x**3 / total**2 - target

    for i in range(mf.size):
        try:
            with np.errstate(all="ignore"):
                root = brentq(func, lo, hi, args=(float(mf[i]), float(m1[i])))
        except (ValueError, RuntimeError, ZeroDivisionError):
            continue
        m2[i] = float(root)
        ok[i] = True
    return m2, ok


def source_seed(base_seed: int, source_id: int) -> int:
    """Per-source seed ``base_seed ^ (source_id & 0x7FFFFFFF)`` (repo convention)."""
    return int(base_seed) ^ (int(source_id) & _SEED_MASK)


def probability_cut(spec: SampleSelectionFile) -> SampleCut:
    """The cut named by ``reproduction_procedure.pass1.probability_cut_id``."""
    proc = require_procedure(spec)
    for cut in spec.cuts or []:
        if cut.id == proc.pass1.probability_cut_id:
            return cut
    raise AtfProcedureError(
        f"sample {spec.name!r}: probability cut "
        f"{proc.pass1.probability_cut_id!r} not in cut chain"
    )


def m2_threshold_msun(spec: SampleSelectionFile) -> float:
    """``m2_threshold_msun`` parameter of the pass-1 probability cut."""
    raw = probability_cut(spec).parameters.get("m2_threshold_msun")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise AtfProcedureError("probability cut lacks numeric m2_threshold_msun")
    return float(raw)


def require_procedure(spec: SampleSelectionFile) -> ReproductionProcedureSpec:
    """Return ``spec.reproduction_procedure`` or raise."""
    proc = spec.reproduction_procedure
    if proc is None:
        raise AtfProcedureError(f"sample {spec.name!r} has no reproduction_procedure")
    return proc


def run_pass1(
    inputs: AtfSourceInputs,
    proc: ReproductionProcedureSpec,
    *,
    m2_threshold: float,
) -> dict[str, Any]:
    """Notebook ``find_massive`` for one source → ``andrews_atf_*`` pass-1 columns.

    Always returns the covariance / root diagnostics, even for a rejected
    source, so the waterfall can attribute each implicit rejection.
    """
    out: dict[str, Any] = {
        "andrews_atf_covariance_ok": False,
        "andrews_atf_covariance_failure": None,
        "andrews_atf_cov_min_eig_ratio": float("nan"),
        "andrews_atf_pass1_root_ok": None,
        "andrews_atf_p_m2_above": float("nan"),
        "andrews_atf_p_m2_above_valid_draws": float("nan"),
        "andrews_atf_pass1_n_root_failed": -1,
        "andrews_atf_pass1_n_mf_below_bracket": -1,
        "andrews_atf_pass1_n_mf_above_bracket": -1,
        "andrews_atf_pass1_n_nonfinite_mf": -1,
        "andrews_atf_pass1_n_negative_parallax": -1,
    }
    try:
        cov = build_notebook_covariance(inputs.errors, inputs.corr_vec)
    except AtfProcedureError as exc:
        out["andrews_atf_covariance_failure"] = str(exc)
        return out
    out["andrews_atf_cov_min_eig_ratio"] = covariance_min_eig_ratio(cov)
    dist, failure = gate_covariance(inputs.means, cov)
    if dist is None:
        out["andrews_atf_covariance_failure"] = failure
        return out
    out["andrews_atf_covariance_ok"] = True
    p1 = proc.pass1
    rng = np.random.default_rng(source_seed(p1.random_seed, inputs.source_id))
    samples = draw_notebook_samples(dist, p1.n_draws, rng)
    m_f, plx = notebook_mass_function(samples, proc.covariance.parameter_order)
    m2, ok = solve_m2_fixed_m1(
        m_f, m1_msun=float(p1.m1_msun), bracket_msun=p1.root_bracket_msun
    )
    g_lo = _g_of_m2(float(p1.root_bracket_msun[0]), float(p1.m1_msun))
    g_hi = _g_of_m2(float(p1.root_bracket_msun[1]), float(p1.m1_msun))
    finite = np.isfinite(m_f)
    n_fail = int(np.count_nonzero(~ok))
    out["andrews_atf_pass1_n_root_failed"] = n_fail
    out["andrews_atf_pass1_n_nonfinite_mf"] = int(np.count_nonzero(~finite))
    out["andrews_atf_pass1_n_mf_below_bracket"] = int(
        np.count_nonzero(finite & (m_f < g_lo))
    )
    out["andrews_atf_pass1_n_mf_above_bracket"] = int(np.count_nonzero(finite & (m_f > g_hi)))
    out["andrews_atf_pass1_n_negative_parallax"] = int(np.count_nonzero(plx <= 0.0))
    out["andrews_atf_pass1_root_ok"] = (
        n_fail == 0 if p1.reject_source_on_any_draw_failure else True
    )
    n_above = int(np.count_nonzero(ok & (m2 > m2_threshold)))
    n_valid = int(np.count_nonzero(ok))
    p_all = n_above / float(p1.n_draws)
    p_valid = n_above / float(n_valid) if n_valid > 0 else float("nan")
    out["andrews_atf_p_m2_above_valid_draws"] = p_valid
    out["andrews_atf_p_m2_above"] = (
        p_all if p1.probability_denominator == "all_draws" else p_valid
    )
    return out


def resolve_refined_m1(
    source_id: int,
    mass_flame: float,
    spec: AtfNotebookPrimaryMassSpec,
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[str, NDArray[np.floating]]:
    """Notebook ``plot_system`` M1: Lick, else FLAME, else uniform (no clipping)."""
    lick = {row.source_id: row.m1_msun for row in spec.lick_spectroscopic_masses}
    if int(source_id) in lick:
        return M1_SOURCE_LICK, rng.normal(
            loc=float(lick[int(source_id)]), scale=float(spec.lick_sigma_msun), size=n_draws
        )
    if not math.isfinite(float(mass_flame)):
        return M1_SOURCE_UNIFORM, rng.uniform(
            low=float(spec.uniform_low_msun), high=float(spec.uniform_high_msun), size=n_draws
        )
    return M1_SOURCE_FLAME, rng.normal(
        loc=float(mass_flame), scale=float(spec.flame_sigma_msun), size=n_draws
    )


def run_pass2(inputs: AtfSourceInputs, proc: ReproductionProcedureSpec) -> dict[str, Any]:
    """Notebook ``plot_system`` M2 recomputation for one pass-1 survivor.

    Draws a fresh covariance sample (the notebook calls ``get_random_samples``
    again), resolves the refined M1, solves every draw with ``brentq`` in the
    pass-2 bracket, and returns ``mean`` / ``std`` (``m2_std_ddof``) of ``M2``.
    """
    p2 = proc.pass2
    rng = np.random.default_rng(source_seed(p2.random_seed, inputs.source_id))
    cov = build_notebook_covariance(inputs.errors, inputs.corr_vec)
    dist, failure = gate_covariance(inputs.means, cov)
    if dist is None:
        raise AtfProcedureError(
            f"source {inputs.source_id}: pass-2 covariance gate failed ({failure}) "
            "after pass 1 accepted it"
        )
    samples = draw_notebook_samples(dist, p2.n_draws, rng)
    m_f, _plx = notebook_mass_function(samples, proc.covariance.parameter_order)
    m1_source, m1 = resolve_refined_m1(
        inputs.source_id, inputs.mass_flame, p2.primary_mass, n_draws=p2.n_draws, rng=rng
    )
    m2, ok = solve_m2_brentq(m_f, m1, bracket_msun=p2.root_bracket_msun)
    n_fail = int(np.count_nonzero(~ok))
    valid = m2[ok]
    return {
        "andrews_atf_m1_source": m1_source,
        "andrews_atf_m1_mean_msun": float(np.mean(m1)),
        "andrews_atf_pass2_n_root_failed": n_fail,
        "andrews_atf_pass2_root_ok": (
            n_fail == 0 if p2.reject_source_on_any_draw_failure else True
        ),
        "andrews_atf_m2_mean_msun": float(np.mean(valid)) if valid.size else float("nan"),
        "andrews_atf_m2_std_msun": (
            float(np.std(valid, ddof=int(p2.m2_std_ddof)))
            if valid.size > int(p2.m2_std_ddof)
            else float("nan")
        ),
    }


def notebook_cmd_quantities(
    g_mag: float, bp_mag: float, rp_mag: float, parallax_mas: float
) -> tuple[float, float]:
    """Notebook CMD inputs ``(G_abs, BP_abs − RP_abs)``; no extinction.

    ``M = m − 5 log10(100 / ϖ)`` with the NSS parallax point value. A
    non-positive or missing parallax gives NaN (the notebook's comparison is
    then false, so the source is **not** rejected).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = 5.0 * np.log10(PARALLAX_MAS_AT_10PC / np.float64(parallax_mas))
        g_abs = float(np.float64(g_mag) - dm)
        bp_abs = np.float64(bp_mag) - dm
        rp_abs = np.float64(rp_mag) - dm
    return g_abs, float(bp_abs - rp_abs)


def static_columns(inputs: AtfSourceInputs, proc: ReproductionProcedureSpec) -> dict[str, Any]:
    """Columns that need no MC: GoF, logg, FLAME, CMD inputs."""
    parallax = float(inputs.means[proc.covariance.parameter_order.index("parallax")])
    g_abs, color = notebook_cmd_quantities(
        inputs.g_mag, inputs.bp_mag, inputs.rp_mag, parallax
    )
    return {
        "andrews_atf_goodness_of_fit": float(inputs.goodness_of_fit),
        "andrews_atf_logg": float(inputs.logg),
        "andrews_atf_mass_flame": float(inputs.mass_flame),
        "andrews_atf_abs_g_mag": g_abs,
        "andrews_atf_bp_rp": color,
    }


def empty_pass2_columns() -> dict[str, Any]:
    """Pass-2 columns for sources pass 1 rejected (not computed)."""
    return {
        "andrews_atf_m1_source": None,
        "andrews_atf_m1_mean_msun": float("nan"),
        "andrews_atf_pass2_n_root_failed": -1,
        "andrews_atf_pass2_root_ok": None,
        "andrews_atf_m2_mean_msun": float("nan"),
        "andrews_atf_m2_std_msun": float("nan"),
    }


# ---------------------------------------------------------------------------
# Sidecar: fingerprint-keyed HDF5 of per-source columns
# ---------------------------------------------------------------------------


def procedure_fingerprint(spec: SampleSelectionFile) -> str:
    """16-hex digest of everything that changes an ``andrews_atf_*`` value.

    Covers the procedure block and the pass-1 probability cut (its threshold
    drives ``p_m2_above`` and the pass-2 subset). The rest of the cut chain is
    applied at evaluation time and is deliberately excluded.
    """
    proc = require_procedure(spec)
    payload = {
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "procedure": proc.model_dump(mode="json"),
        "probability_cut": probability_cut(spec).model_dump(mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def sidecar_path(cache_dir: Path, dr_mode: str, spec: SampleSelectionFile) -> Path:
    """``<cache_dir>/<dr_mode>/<method>_<fingerprint>.h5``."""
    proc = require_procedure(spec)
    return cache_dir / dr_mode / f"{proc.method}_{procedure_fingerprint(spec)}.h5"


def write_sidecar(
    path: Path,
    rows_by_source: Mapping[int, Mapping[str, Any]],
    *,
    fingerprint: str,
    attrs: Mapping[str, Any],
) -> None:
    """Atomically write the columnar sidecar (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(sorted(int(s) for s in rows_by_source), dtype=np.int64)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".h5.tmp")
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as handle:
            handle.attrs["schema_version"] = SIDECAR_SCHEMA_VERSION
            handle.attrs["fingerprint"] = fingerprint
            for key, value in attrs.items():
                handle.attrs[key] = json.dumps(value, default=str)
            handle.create_dataset("source_id", data=ids)
            for col in FLOAT_COLUMNS:
                data = [_as_float(rows_by_source[int(s)].get(col)) for s in ids]
                handle.create_dataset(col, data=np.asarray(data, dtype=np.float64))
            for col in INT_COLUMNS:
                data = [int(rows_by_source[int(s)].get(col, -1)) for s in ids]
                handle.create_dataset(col, data=np.asarray(data, dtype=np.int64))
            for col in BOOL_COLUMNS:
                data = [_tri_state(rows_by_source[int(s)].get(col)) for s in ids]
                handle.create_dataset(col, data=np.asarray(data, dtype=np.int8))
            for col in STR_COLUMNS:
                data = [str(rows_by_source[int(s)].get(col) or "") for s in ids]
                handle.create_dataset(
                    col, data=np.asarray(data, dtype=h5py.string_dtype("utf-8"))
                )
        # mkstemp creates 0600; the sidecar is a shared, read-only data cache.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_sidecar(path: Path, *, fingerprint: str) -> dict[int, dict[str, Any]]:
    """Read a sidecar into ``source_id → {column: value}``.

    NaN floats and ``-1`` tri-states become ``None``; empty strings become
    ``None``. Refuses a file whose stored fingerprint differs from
    ``fingerprint``.
    """
    with h5py.File(path, "r") as handle:
        stored = str(handle.attrs.get("fingerprint", ""))
        if stored != fingerprint:
            raise AtfProcedureError(
                f"sidecar {path} fingerprint {stored!r} != expected {fingerprint!r}"
            )
        ids = np.asarray(handle["source_id"][()], dtype=np.int64)
        columns: dict[str, list[Any]] = {}
        for col in FLOAT_COLUMNS:
            columns[col] = [
                None if not math.isfinite(v) else float(v) for v in handle[col][()]
            ]
        for col in INT_COLUMNS:
            columns[col] = [int(v) for v in handle[col][()]]
        for col in BOOL_COLUMNS:
            columns[col] = [None if v < 0 else bool(v) for v in handle[col][()]]
        for col in STR_COLUMNS:
            columns[col] = [s or None for s in handle[col].asstr()[()]]
    out: dict[int, dict[str, Any]] = {}
    for i, sid in enumerate(ids):
        out[int(sid)] = {col: columns[col][i] for col in ALL_COLUMNS}
    return out


def reproduction_cache_dir(config: PipelineConfig, *, repo: Path) -> Path:
    """``sample_selection.reproduction_column_cache_dir`` resolved against ``paths.data_root``."""
    raw = Path(config.sample_selection.reproduction_column_cache_dir)
    if raw.is_absolute():
        return raw
    data_root = Path(config.paths.data_root)
    if not data_root.is_absolute():
        data_root = repo / data_root
    return data_root / raw


def load_reproduction_columns(
    spec: SampleSelectionFile, config: PipelineConfig, *, repo: Path
) -> dict[int, dict[str, Any]]:
    """Sidecar columns for ``spec``'s procedure on this host, or ``{}`` if not built.

    A missing sidecar is logged, not raised: the ``atf_*`` cuts then report
    ``NOT_APPLICABLE`` (``missing:andrews_atf_*``) for every row, which is
    visible in the attrition waterfall.
    """
    path = sidecar_path(
        reproduction_cache_dir(config, repo=repo), config.active_dr_mode.value, spec
    )
    if not path.is_file():
        logger.warning(
            "sample %s: reproduction-procedure sidecar %s not built; run "
            "scripts/build_andrews2022_atf_columns.py (#296)",
            spec.name,
            path,
        )
        return {}
    return read_sidecar(path, fingerprint=procedure_fingerprint(spec))


def merge_reproduction_columns(
    rows: Iterable[Mapping[str, Any]],
    columns_by_source: Mapping[int, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Add each row's sidecar columns without overwriting existing row keys.

    Rows with no sidecar entry are passed through unchanged (not copied).
    """
    merged: list[Mapping[str, Any]] = []
    for row in rows:
        extra = columns_by_source.get(int(row["source_id"]))
        if not extra:
            merged.append(row)
            continue
        out = dict(row)
        for key, value in extra.items():
            out.setdefault(key, value)
        merged.append(out)
    return merged


def _as_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _tri_state(value: Any) -> int:
    if value is None:
        return -1
    return 1 if bool(value) else 0
