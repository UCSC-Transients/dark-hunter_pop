"""Andrews ``andrews_atf_*`` columns for mock rows, through the real-row ATF code path (#679, RUNG5_SPEC §2.7.4).

Ryan (absolute): the mock passes through exactly the same cuts as the real DR3 rows, per sample. The Andrews
chain reads ``andrews_atf_*`` columns that, for real DR3 rows, ``scripts/build_andrews2022_atf_columns.py``
computes from the NSS orbit (``*_error`` + ``corr_vec``) with the ATF notebook procedure
(:mod:`darkhunter_pop.andrews2022_atf`). This module feeds the **same** functions (``static_columns``,
``run_pass1``, ``run_pass2``, ``pipeline_tag10_m1``) with the mock's fitted orbit. No ATF logic is copied.

Mock orbit -> ATF inputs
------------------------
The proposal-set artifact stores ``orbit_p0`` (12) and ``orbit_cov`` (12x12) in gaiamock order
(:data:`darkhunter_pop.proposal_set.ORBIT_PARAMETER_NAMES`)::

    ra_offset, dec_offset, parallax, pmra, pmdec, period, eccentricity, phi_p, A, B, F, G

The ATF procedure expects ``covariance.parameter_order`` of the frozen selection file::

    ra, dec, parallax, pmra, pmdec, A, B, F, G, eccentricity, period, t_periastron

A mismatch is a silent wrong answer, so :func:`mock_orbit_to_atf_inputs` permutes by **name** and asserts both
orders are exactly the expected 12-name sets (tested). Units agree (mas, mas yr^-1, d, Thiele-Innes in mas).
Means are the fitted ``p0``; errors are ``sqrt(diag(cov))`` and ``corr_vec`` is the strict lower triangle of the
correlation matrix in ``np.tril_indices(12, -1)`` order, which is what ``build_notebook_covariance`` reads back
(the NSS ``corr_vec`` convention). No float32 round trip (that reproduces the archive CSV text, not a mock).
A non-finite covariance gives NaN errors, so the notebook covariance gate rejects it (counted, **never** a
diagonal fallback).

The one non-identity slot is ``t_periastron``: gaiamock's nonlinear parameter is the phase angle ``phi_p`` (rad),
DR3's is a time in days. It is mapped with the linearised ``T_p = phi_p P / (2 pi)`` (covariance propagated with
the Jacobian, sign and epoch origin immaterial). ``t_periastron`` does not enter ``m_f``, ``M2`` or any Andrews
cut; it only enters the 12-D covariance gate and the draw stream, and the 11 other parameters keep their exact
marginal covariance. This is an assumption of the mock path, not of the real-row path.

Pass-2 M1
---------
``forward_model`` pass 2 uses the pipeline TAG10 posterior ``N(M1, sigma_M1)`` of the *data-side* luminous star
(``andrews2022.yaml`` ``forward_model_primary_mass.method: pipeline_tag10_bulk``). For mock rows that is the parent
Gaia row's TAG10 posterior: :func:`parent_tag10_m1` runs ``pipeline_tag10_m1`` (the function the real sidecar builder
uses) on the parent row's atmosphere columns. The caller passes ``(M1, sigma_M1)`` per row; a row without one
falls back to the notebook's Uniform(0.63, 1.0), exactly as a real row without an atmosphere.

Known limitation (#680)
-----------------------
``andrews_atf_logg`` (GSP-Phot log g, ``logg_vizier_apsis`` in the real chain) is NaN on mock rows and
``andrews_atf_mass_flame`` is NaN (unused in ``forward_model``). ``atf_giant_reject_logg`` treats an undefined
log g as a pass, so until #680 it removes no mock row while it removes giants from the real data. The CMD giant cut
(``atf_giant_reject_cmd``) is evaluated: G_abs and BP-RP come from the system G and ``bp_rp`` of the mock row.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from darkhunter_pop.andrews2022_atf import (
    AtfSourceInputs,
    empty_pass2_columns,
    m2_threshold_msun,
    merge_reproduction_columns,
    pipeline_tag10_m1,
    require_procedure,
    run_pass1,
    run_pass2,
    static_columns,
)
from darkhunter_pop.config_schema import (
    ActiveDRMode,
    PipelineConfig,
    ReproductionProcedureSpec,
    SampleSelectionFile,
    SampleSelectionMode,
)

logger = logging.getLogger(__name__)

#: gaiamock order of ``orbit_p0`` / ``orbit_cov`` (``proposal_set.ORBIT_PARAMETER_NAMES``; pinned by a test).
MOCK_ORBIT_NAMES: Final[tuple[str, ...]] = (
    "ra_offset", "dec_offset", "parallax", "pmra", "pmdec", "period", "eccentricity", "phi_p",
    "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes",
)

#: ATF parameter name -> mock parameter name that fills its slot.
ATF_FROM_MOCK: Final[dict[str, str]] = {
    "ra": "ra_offset", "dec": "dec_offset", "parallax": "parallax", "pmra": "pmra", "pmdec": "pmdec",
    "a_thiele_innes": "a_thiele_innes", "b_thiele_innes": "b_thiele_innes",
    "f_thiele_innes": "f_thiele_innes", "g_thiele_innes": "g_thiele_innes",
    "eccentricity": "eccentricity", "period": "period", "t_periastron": "phi_p",
}

_TWO_PI: Final[float] = 2.0 * np.pi


def _slot_permutation(parameter_order: Sequence[str]) -> list[int]:
    """Mock index feeding each ATF slot; asserts the ATF order is the expected 12-name set."""
    order = list(parameter_order)
    if sorted(order) != sorted(ATF_FROM_MOCK):
        raise ValueError(f"ATF covariance.parameter_order {order} is not the expected parameter set")
    if list(MOCK_ORBIT_NAMES) != list(_proposal_orbit_names()):
        raise ValueError("MOCK_ORBIT_NAMES drifted from proposal_set.ORBIT_PARAMETER_NAMES")
    return [MOCK_ORBIT_NAMES.index(ATF_FROM_MOCK[name]) for name in order]


def _proposal_orbit_names() -> tuple[str, ...]:
    from darkhunter_pop.proposal_set import ORBIT_PARAMETER_NAMES

    return tuple(ORBIT_PARAMETER_NAMES)


def mock_orbit_to_atf_frame(
    p0: ArrayLike, cov: ArrayLike, parameter_order: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mock ``(p0, cov)`` -> ``(means, covariance)`` in the ATF slot order, ``phi_p`` -> ``t_periastron``."""
    perm = _slot_permutation(parameter_order)
    p = np.asarray(p0, dtype=np.float64).ravel()
    c = np.asarray(cov, dtype=np.float64)
    n = len(perm)
    if p.shape != (n,) or c.shape != (n, n):
        raise ValueError(f"orbit_p0 / orbit_cov must be ({n},) / ({n}, {n}); got {p.shape} / {c.shape}")
    means = p[perm].copy()
    cc = c[np.ix_(perm, perm)].copy()
    it, ip = list(parameter_order).index("t_periastron"), list(parameter_order).index("period")
    phi, period = means[it], means[ip]
    jac = np.eye(n)
    jac[it, it] = period / _TWO_PI
    jac[it, ip] = phi / _TWO_PI
    means[it] = phi * period / _TWO_PI
    return means, jac @ cc @ jac.T


def mock_orbit_to_atf_inputs(
    source_id: int,
    p0: ArrayLike,
    cov: ArrayLike,
    *,
    proc: ReproductionProcedureSpec,
    goodness_of_fit: float,
    g_mag: float,
    bp_rp: float,
    pipeline_m1_msun: float = float("nan"),
    pipeline_m1_sigma_msun: float = float("nan"),
) -> AtfSourceInputs:
    """:class:`AtfSourceInputs` of one mock row (module docstring); the ATF functions take it unchanged.

    ``BP`` / ``RP`` are set to ``G + bp_rp`` / ``G`` so the notebook's ``BP_abs - RP_abs`` is the mock's observed
    ``bp_rp`` (the notebook CMD cut uses no extinction correction). ``logg`` and FLAME are NaN (#680).
    """
    order = list(proc.covariance.parameter_order)
    means, cc = mock_orbit_to_atf_frame(p0, cov, order)
    n = len(order)
    diag = np.diag(cc)
    ok = bool(np.all(np.isfinite(cc)) and np.all(diag > 0.0))
    if ok:
        sigma = np.sqrt(diag)
        rows, cols = np.tril_indices(n, k=-1)
        corr_vec = cc[rows, cols] / (sigma[rows] * sigma[cols])
    else:
        sigma = np.full(n, np.nan)
        corr_vec = np.full(n * (n - 1) // 2, np.nan)
    return AtfSourceInputs(
        source_id=int(source_id), means=means, errors=sigma, corr_vec=corr_vec,
        goodness_of_fit=float(goodness_of_fit), mass_flame=float("nan"), logg=float("nan"),
        g_mag=float(g_mag), bp_mag=float(g_mag) + float(bp_rp), rp_mag=float(g_mag),
        pipeline_m1_msun=float(pipeline_m1_msun), pipeline_m1_sigma_msun=float(pipeline_m1_sigma_msun),
    )


def parent_tag10_m1(
    parent_columns: Mapping[str, ArrayLike], parent_row: ArrayLike, cfg: PipelineConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Data-side TAG10 ``(M1, sigma_M1)`` of mock rows' parent Gaia rows, via ``andrews2022_atf.pipeline_tag10_m1``.

    ``parent_columns`` holds the parent snapshot's atmosphere columns (``teff_msc1`` ... ``mh_gspphot`` with
    ``_upper`` / ``_lower``); ``parent_row`` indexes it per mock row (``truth/parent_row``). NaN where TAG10 has no
    usable atmosphere (pass 2 then draws Uniform(0.63, 1.0), as a real row would).
    """
    rows = np.asarray(parent_row, dtype=np.int64)
    uniq, inv = np.unique(rows, return_inverse=True)
    names = [c for c in parent_columns if c.split("_upper")[0].split("_lower")[0] in _ATMOSPHERE_BASES]
    arrays = {c: np.asarray(parent_columns[c], dtype=np.float64) for c in names}
    mu = np.full(uniq.size, np.nan)
    sg = np.full(uniq.size, np.nan)
    for k, r in enumerate(uniq):
        extras = {c: float(a[r]) for c, a in arrays.items() if np.isfinite(a[r])}
        mu[k], sg[k] = pipeline_tag10_m1(extras, cfg)
    return mu[inv], sg[inv]


_ATMOSPHERE_BASES: Final[frozenset[str]] = frozenset(
    {"teff_msc1", "logg_msc1", "mh_msc", "teff_gspphot", "logg_gspphot", "mh_gspphot"}
)

_WORKER_PROC: ReproductionProcedureSpec | None = None
_WORKER_THRESHOLD: float | None = None


def _init_worker(proc_json: str, threshold: float) -> None:
    global _WORKER_PROC, _WORKER_THRESHOLD
    _WORKER_PROC = ReproductionProcedureSpec.model_validate_json(proc_json)
    _WORKER_THRESHOLD = float(threshold)


def _pass1_chunk(chunk: Sequence[AtfSourceInputs]) -> list[tuple[int, dict[str, Any]]]:
    assert _WORKER_PROC is not None and _WORKER_THRESHOLD is not None
    return [(i.source_id, run_pass1(i, _WORKER_PROC, m2_threshold=_WORKER_THRESHOLD)) for i in chunk]


def _pass2_chunk(job: tuple[str, Sequence[AtfSourceInputs]]) -> list[tuple[int, dict[str, Any]]]:
    assert _WORKER_PROC is not None
    mode = SampleSelectionMode(job[0])
    return [(i.source_id, run_pass2(i, _WORKER_PROC, mode=mode)) for i in job[1]]


def compute_atf_columns(
    inputs: Sequence[AtfSourceInputs],
    spec: SampleSelectionFile,
    *,
    mode: SampleSelectionMode = SampleSelectionMode.FORWARD_MODEL,
    workers: int = 1,
    chunk: int = 50,
    dr_mode: ActiveDRMode = ActiveDRMode.DR3,
) -> dict[int, dict[str, Any]]:
    """``andrews_atf_*`` columns per ``source_id`` through the real-row ATF orchestration.

    Mirrors ``scripts/build_andrews2022_atf_columns.py`` ``main`` step for step: static columns, pass 1 for every
    source, pass-1 survivors decided by the frozen chain itself (the probability cut evaluated in reproduction
    mode, as the script does), pass 2 for survivors only with ``mode``'s M1. ``inputs`` carry the pass-2 M1
    (``pipeline_m1_*``). ``workers <= 1`` runs in process (tests); otherwise a process pool.
    """
    from darkhunter_pop.sample_selection import CutOutcome, SampleSelection

    proc = require_procedure(spec)
    threshold = m2_threshold_msun(spec)
    ids = [i.source_id for i in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("source_id must be unique (columns are keyed by it)")
    columns: dict[int, dict[str, Any]] = {
        i.source_id: {**static_columns(i, proc), **empty_pass2_columns()} for i in inputs
    }
    chunks = [list(inputs[k : k + chunk]) for k in range(0, len(inputs), chunk)]
    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(proc.model_dump_json(), threshold)
        )
        pmap = pool.map
    else:
        _init_worker(proc.model_dump_json(), threshold)
        pmap = map  # type: ignore[assignment]
    try:
        for res in pmap(_pass1_chunk, chunks):
            for sid, cols in res:
                columns[sid].update(cols)
        selection = SampleSelection(spec, mode=SampleSelectionMode.REPRODUCTION, dr_mode=dr_mode)
        rows = merge_reproduction_columns(
            [{"source_id": i.source_id, "nss_solution_type": "Orbital"} for i in inputs], columns
        )
        result = selection.evaluate(rows)
        cut_id = proc.pass1.probability_cut_id
        survivors = {
            sid for sid, outs in result.outcomes_by_source.items()
            if any(c == cut_id and o is CutOutcome.PASSED for c, o, _r in outs)
        }
        todo = [i for i in inputs if i.source_id in survivors]
        jobs = [(mode.value, todo[k : k + 4]) for k in range(0, len(todo), 4)]
        for res in pmap(_pass2_chunk, jobs):
            for sid, cols in res:
                columns[sid].update(cols)
    finally:
        if pool is not None:
            pool.shutdown()
    logger.info("mock ATF: %d rows, %d pass-1 survivors", len(inputs), len(survivors))
    return columns


def mock_rows_atf_columns(
    rows: Sequence[Mapping[str, Any]],
    spec: SampleSelectionFile,
    *,
    m1_msun: ArrayLike,
    m1_sigma_msun: ArrayLike,
    workers: int = 1,
    chunk: int = 50,
    dr_mode: ActiveDRMode = ActiveDRMode.DR3,
) -> dict[int, dict[str, Any]]:
    """``andrews_atf_*`` columns for rows of ``mock_selection_rows.mock_draws_to_selection_rows(include_orbit=True)``.

    ``m1_msun`` / ``m1_sigma_msun`` are the per-row data-side TAG10 posterior (:func:`parent_tag10_m1`). Returns
    ``{source_id: columns}`` ready for ``SampleSelection``'s ``reproduction_columns_loader`` (mock ``source_id`` is
    the draw index; pass-2 randomness is seeded by it, as the real ATF seeds by the Gaia id). The logg column is
    NaN (#680).
    """
    proc = require_procedure(spec)
    mu = np.asarray(m1_msun, dtype=np.float64)
    sg = np.asarray(m1_sigma_msun, dtype=np.float64)
    if mu.shape != (len(rows),) or sg.shape != (len(rows),):
        raise ValueError("m1_msun / m1_sigma_msun must have one entry per row")
    inputs = []
    for k, r in enumerate(rows):
        if "mock_orbit_p0" not in r or "mock_orbit_cov" not in r:
            raise ValueError("rows need mock_orbit_p0 / mock_orbit_cov (include_orbit=True)")
        inputs.append(
            mock_orbit_to_atf_inputs(
                int(r["source_id"]), r["mock_orbit_p0"], r["mock_orbit_cov"], proc=proc,
                goodness_of_fit=r["goodness_of_fit"], g_mag=r["g_mag"], bp_rp=r["bp_rp"],
                pipeline_m1_msun=mu[k], pipeline_m1_sigma_msun=sg[k],
            )
        )
    cols = compute_atf_columns(
        inputs, spec, mode=SampleSelectionMode.FORWARD_MODEL, workers=workers, chunk=chunk, dr_mode=dr_mode
    )
    fail = Counter(str(c["andrews_atf_covariance_failure"]) for c in cols.values() if not c["andrews_atf_covariance_ok"])
    if fail:
        logger.info("mock ATF covariance failures (no diagonal fallback): %s", dict(fail))
    return cols


__all__ = [
    "ATF_FROM_MOCK", "MOCK_ORBIT_NAMES", "compute_atf_columns", "mock_orbit_to_atf_frame",
    "mock_orbit_to_atf_inputs", "mock_rows_atf_columns", "parent_tag10_m1",
]
