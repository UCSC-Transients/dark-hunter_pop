"""Mock draws -> ``sample_selection`` rows with the real DR3 column names (#678, RUNG5_SPEC §2.7.4).

Rung 5 forward-models each published sample by running that sample's frozen cut chain on the mock the same
way it ran on the real DR3 rows (Andrews cuts vs Andrews, El-Badry vs El-Badry; Ryan, absolute). This module
is the generic adapter only: it turns the ``(truth, outcome)`` arrays of a proposal-set artifact
(:func:`darkhunter_pop.proposal_set.load_generation_draws`) into the flat row dicts that
:meth:`darkhunter_pop.sample_selection.SampleSelection.evaluate` consumes, keyed by the **real** snapshot
column names, so the evaluator runs **unchanged** in ``forward_model`` mode. No cut logic is duplicated here.

Terminology: luminous star (M1) and dark companion (M2); "real DR3" is the archive, "mock" is a draw.

Row identity
------------
``source_id`` is the mock ``draw_index`` (unique across generations; the evaluator keys its outcome tables by
``source_id`` and the parent Gaia ``source_id`` repeats across draws). :data:`MOCK_SOURCE_ID_MAX` keeps it below
the smallest real Gaia ``source_id`` so no join to real-row data (enrichment index, ATF sidecar, external
catalogues) can ever silently match. The parent Gaia id is kept in ``mock_parent_source_id``.

Which draws become rows
-----------------------
Only draws whose cascade published an astrometric **Orbital** solution (``outcome/accepted_orbital``) become
rows, with ``nss_solution_type = "Orbital"``. The mock has no ``AstroSpectroSB1`` branch (SB1 is v1
out-of-scope, Q6), so no row carries that type.

Column provenance (real column <- mock quantity)
------------------------------------------------
Cascade vector layout (``cascade_layout`` attribute of the artifact, gaiamock_mod
``fit_full_astrometric_cascade``): see :data:`CASCADE_COLUMNS`.

=================================  ==========================================================================
real DR3 / evaluator column        mock quantity
=================================  ==========================================================================
``source_id``                      ``truth/draw_index`` (synthetic, see above)
``nss_solution_type``              ``"Orbital"`` where ``outcome/accepted_orbital``
``ra_deg``, ``dec_deg``            ``truth/ra_deg``, ``truth/dec_deg`` (parent Gaia row position)
``parallax``, ``parallax_mas``     cascade ``plx`` (NSS-fit parallax, as the real orbital solution)
``parallax_error``                 cascade ``sig_parallax``
``period``, ``period_day``         cascade ``period`` (fit, d); ``period_error`` <- ``sig_period``
``eccentricity``                   cascade ``ecc``; ``eccentricity_error`` <- ``sig_ecc``
``a_thiele_innes`` (+ ``_error``)  cascade ``A`` (``sig_A``), likewise ``b``/``B``, ``f``/``F``, ``g``/``G`` (mas)
``goodness_of_fit``                cascade ``F2`` (the DR3 orbital ``goodness_of_fit`` statistic; its
                                   calibration is a known limitation, EPOCH_MODEL_SPEC §8.16)
``ruwe``                           cascade ``ruwe`` (modified RUWE = UWE / u0)
``visibility_periods_used``        cascade ``N_visibility_periods``
``phot_g_mean_mag``, ``g_mag``     ``truth/phot_g_mean_mag`` (system G: luminous star + companion light)
``bp_rp``                          ``truth/sys_bp_rp`` (observed, **reddened**, see below)
``abs_g_mag``                      ``G + 5 log10(plx_fit) - 10`` (real recipe of ``candidate_to_selection_row``)
``mg_0``, ``bp_rp_0``              aliases of ``abs_g_mag`` / ``bp_rp`` (the real-row defaults; the evaluator
                                   overwrites them with the sample's own dereddening)
``pipeline_m1_msun``               ``truth/m1_row_msun``: the data-side M1 of the parent Gaia row (TAG10, or MIST
                                   isochrone when the proposal used ``MIST_isochrone``), never the deblended truth
                                   M1; the evaluator binds ``m1_msun`` from it in ``forward_model`` mode
=================================  ==========================================================================

Reddening. The mock photometry (``sys_bp_rp``, system G) is built in ``proposal_set.apply_system_photometry`` from
the parent row's dereddened CMD plus that row's own extinction vector (Combined19 at the Bailer-Jones distance,
Babusiaux et al. 2018 law, ``giants.cmd_for_rows``) at the parent (l, b, d): the mock's *nature*. Each sample's
frozen dereddening *procedure* (El-Badry 2026: Green19 / Lallement19 + paper coefficients) is then applied by
the evaluator itself (``deredden_elbadry2026_rows``) to ``ra_deg`` / ``dec_deg`` / ``parallax`` / ``abs_g_mag`` /
``bp_rp`` of the mock row, exactly as for a real row. This module therefore does no dereddening and adds no
new dust path. The cascade parallax replaces the Bailer-Jones distance in that lookup, as for real rows.

Columns that need extra machinery are **not** emitted (separate tickets), so the evaluator sees them as missing:

* Andrews ``andrews_atf_*`` sidecar columns (#679): no ``reproduction_columns_loader`` is supplied. The first
  ``requires_defined`` ATF cut makes every row not applicable (so the chain stops there); the later ATF cuts
  that name an ``andrews_atf_*`` column directly raise ``CutExpressionError`` (unbound name) if run on their own.
  :func:`classify_cuts` reports both, never silently.
* GSP-Phot ``logg_apsis`` / ``logg_gspphot`` (#680).
* El-Badry 2026 ``sigma_m2_astrometric_msun`` and ``amrf_threshold_elbadry2026`` (#681); the evaluator reads the
  real NSS enrichment index by ``source_id``, which a synthetic id never matches, so those cuts are not applicable.
  ``m1_tilde_msun`` / ``m2_tilde_msun`` / ``main_sequence`` come from the evaluator's own enrichment, unchanged.
* ``shahaf2023b_*`` / AMRF class III (#682), Table E1 / Table 3 membership (external tables; no mock analogue).
* Follow-up labels (``not_spurious`` etc.): catalog-level, SF_followup.
* ``t_periastron``: the cascade stores a phase angle, not the DR3 epoch-based quantity; no cut reads it.

Mock-only bookkeeping columns carry the ``mock_`` prefix and are never read by a frozen cut:
``mock_draw_index``, ``mock_parent_source_id``, ``mock_m1_msun`` / ``mock_m2_msun`` (truth masses),
``mock_is_dark``, ``mock_co_component``, ``mock_period_days`` (truth), ``mock_sys_evolved``, and with
``include_orbit=True`` ``mock_orbit_p0`` / ``mock_orbit_cov`` (12 and 12x12, for #679 / #681).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

#: Names of the 23-vector ``outcome/cascade`` (artifact attribute ``cascade_layout``).
CASCADE_COLUMNS: Final[tuple[str, ...]] = (
    "plx", "sig_parallax", "A", "sig_A", "B", "sig_B", "F", "sig_F", "G", "sig_G",
    "period", "sig_period", "phi_p", "sig_phi_p", "ecc", "sig_ecc", "inc_deg",
    "a0_mas", "sigma_a0_mas", "N_visibility_periods", "N_obs", "F2", "ruwe",
)

#: Smallest real Gaia DR3 ``source_id`` (2^32 + 2^15 style lower bound is above 4.29e9): synthetic ids stay below.
MOCK_SOURCE_ID_MAX: Final[int] = 4_295_806_720

#: ``nss_solution_type`` label of a published astrometric orbit (the only type the mock cascade produces).
ORBITAL_SOLUTION_TYPE: Final[str] = "Orbital"

#: ``(row column, cascade column)`` pairs copied verbatim.
_CASCADE_TO_ROW: Final[tuple[tuple[str, str], ...]] = (
    ("parallax", "plx"),
    ("parallax_error", "sig_parallax"),
    ("period", "period"),
    ("period_error", "sig_period"),
    ("eccentricity", "ecc"),
    ("eccentricity_error", "sig_ecc"),
    ("a_thiele_innes", "A"),
    ("a_thiele_innes_error", "sig_A"),
    ("b_thiele_innes", "B"),
    ("b_thiele_innes_error", "sig_B"),
    ("f_thiele_innes", "F"),
    ("f_thiele_innes_error", "sig_F"),
    ("g_thiele_innes", "G"),
    ("g_thiele_innes_error", "sig_G"),
    ("goodness_of_fit", "F2"),
    ("ruwe", "ruwe"),
    ("visibility_periods_used", "N_visibility_periods"),
)

#: Real-row columns the adapter emits (contract test checks the key set).
REAL_COLUMNS: Final[tuple[str, ...]] = (
    "source_id", "nss_solution_type", "ra_deg", "dec_deg", "parallax_mas", "period_day",
    "phot_g_mean_mag", "g_mag", "bp_rp", "abs_g_mag", "mg_0", "bp_rp_0", "pipeline_m1_msun",
    *(r for r, _ in _CASCADE_TO_ROW),
)


def _finite_positive(x: ArrayLike) -> NDArray[np.bool_]:
    a = np.asarray(x, dtype=np.float64)
    return np.isfinite(a) & (a > 0.0)


def mock_draws_to_selection_rows(
    truth: Mapping[str, NDArray[Any]],
    outcome: Mapping[str, NDArray[Any]],
    *,
    include_orbit: bool = False,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Map mock draws to ``sample_selection`` rows (module docstring: provenance table).

    Parameters
    ----------
    truth, outcome
        Dicts of arrays as returned by ``proposal_set.load_generation_draws`` (needs ``draw_index``,
        ``source_id``, ``ra_deg``, ``dec_deg``, ``phot_g_mean_mag``, ``sys_bp_rp``, ``m1_row_msun`` in ``truth``;
        ``cascade``, ``accepted_orbital`` in ``outcome``).
    include_orbit
        Also attach ``mock_orbit_p0`` / ``mock_orbit_cov`` (heavy: 12x12 per row) for the ATF / covariance
        consumers (#679, #681).
    max_rows
        Keep at most this many accepted-Orbital rows (first in draw order); smoke use only.

    Returns
    -------
    One dict per accepted-Orbital draw, in draw order. Draws without a published orbit are not rows (they are
    not in the real NSS Orbital parent either). ``abs_g_mag`` is NaN-free only where the fit parallax is
    positive; other rows get no ``abs_g_mag`` (as ``candidate_to_selection_row``).
    """
    cascade = np.asarray(outcome["cascade"], dtype=np.float64)
    if cascade.ndim != 2 or cascade.shape[1] != len(CASCADE_COLUMNS):
        raise ValueError(
            f"outcome/cascade must be (n, {len(CASCADE_COLUMNS)}) per the artifact cascade_layout; "
            f"got {cascade.shape}"
        )
    accepted = np.asarray(outcome["accepted_orbital"], dtype=bool)
    if accepted.shape[0] != cascade.shape[0]:
        raise ValueError("outcome arrays have inconsistent lengths")
    idx = np.flatnonzero(accepted)
    if max_rows is not None:
        idx = idx[: int(max_rows)]
    draw_index = np.asarray(truth["draw_index"], dtype=np.int64)[idx]
    if draw_index.size and (draw_index.min() < 0 or draw_index.max() >= MOCK_SOURCE_ID_MAX):
        raise ValueError(
            "draw_index must lie in [0, MOCK_SOURCE_ID_MAX) so synthetic ids cannot collide with real Gaia ids"
        )
    if np.unique(draw_index).size != draw_index.size:
        raise ValueError("draw_index is not unique: source_id would collide in the evaluator outcome tables")

    col = {name: cascade[idx, k] for k, name in enumerate(CASCADE_COLUMNS)}
    g = np.asarray(truth["phot_g_mean_mag"], dtype=np.float64)[idx]
    bp_rp = np.asarray(truth["sys_bp_rp"], dtype=np.float64)[idx]
    plx = col["plx"]
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_g = np.where(_finite_positive(plx), g + 5.0 * np.log10(plx) - 10.0, np.nan)
    m1_row = np.asarray(truth["m1_row_msun"], dtype=np.float64)[idx]
    ra = np.asarray(truth["ra_deg"], dtype=np.float64)[idx]
    dec = np.asarray(truth["dec_deg"], dtype=np.float64)[idx]

    parent_id = np.asarray(truth["source_id"], dtype=np.int64)[idx]
    m1_truth = np.asarray(truth["m1_msun"], dtype=np.float64)[idx]
    m2_truth = np.asarray(truth["m2_msun"], dtype=np.float64)[idx]
    is_dark = np.asarray(truth["is_dark"], dtype=bool)[idx]
    co_comp = np.asarray(truth["co_component"])[idx]
    p_truth = np.asarray(truth["period_days"], dtype=np.float64)[idx]
    evolved = np.asarray(truth["sys_evolved"], dtype=bool)[idx]
    p0 = cov = None
    if include_orbit:
        p0 = np.asarray(outcome["orbit_p0"], dtype=np.float64)[idx]
        cov = np.asarray(outcome["orbit_cov"], dtype=np.float64)[idx]

    rows: list[dict[str, Any]] = []
    for i in range(idx.size):
        row: dict[str, Any] = {
            "source_id": int(draw_index[i]),
            "nss_solution_type": ORBITAL_SOLUTION_TYPE,
            "ra_deg": float(ra[i]),
            "dec_deg": float(dec[i]),
        }
        for real, casc in _CASCADE_TO_ROW:
            row[real] = float(col[casc][i])
        row["parallax_mas"] = row["parallax"]
        row["period_day"] = row["period"]
        row["phot_g_mean_mag"] = float(g[i])
        row["g_mag"] = row["phot_g_mean_mag"]
        row["bp_rp"] = float(bp_rp[i])
        if np.isfinite(abs_g[i]):
            row["abs_g_mag"] = float(abs_g[i])
            row["mg_0"] = row["abs_g_mag"]
        row["bp_rp_0"] = row["bp_rp"]
        row["pipeline_m1_msun"] = float(m1_row[i])
        row["mock_draw_index"] = int(draw_index[i])
        row["mock_parent_source_id"] = int(parent_id[i])
        row["mock_m1_msun"] = float(m1_truth[i])
        row["mock_m2_msun"] = float(m2_truth[i])
        row["mock_is_dark"] = bool(is_dark[i])
        row["mock_co_component"] = int(co_comp[i])
        row["mock_period_days"] = float(p_truth[i])
        row["mock_sys_evolved"] = bool(evolved[i])
        if p0 is not None and cov is not None:
            row["mock_orbit_p0"] = p0[i]
            row["mock_orbit_cov"] = cov[i]
        rows.append(row)
    logger.info("mock draws -> rows: %d draws, %d accepted-Orbital rows", accepted.size, len(rows))
    return rows


def forward_model_selection(
    registry: Any,
    name: str,
    *,
    extinction_lookup: Any | None = None,
) -> Any:
    """A ``SampleSelection`` for sample ``name`` in ``forward_model`` mode, whatever the registry mode.

    The registry's configured mode is the real-data mode (``reproduction`` for the published samples); the mock
    must run the sample's ``forward_model`` cut list. This builds the evaluator exactly as
    ``SampleSelectionRegistry.selection`` does, with the mode forced and **no** reproduction-columns loader
    (the ATF sidecar is keyed by real ``source_id``; #679). ``extinction_lookup`` defaults to the El-Badry
    2026 builder (the real-row path); pass ``None`` rows-with-``ebv_elbadry2026`` fixtures through a test stub.
    """
    from darkhunter_pop.config_schema import SampleSelectionMode
    from darkhunter_pop.elbadry2026_selection import build_elbadry2026_extinction_lookup
    from darkhunter_pop.sample_selection import SampleSelection

    spec = registry.resolved(name)
    lookup = extinction_lookup
    if lookup is None and registry.pipeline is not None and spec.extinction is not None:
        lookup = build_elbadry2026_extinction_lookup(spec, registry.pipeline, repo=registry.repo)
    return SampleSelection(
        spec,
        mode=SampleSelectionMode.FORWARD_MODEL,
        dr_mode=registry.dr_mode,
        extinction_lookup=lookup,
        reproduction_columns_loader=None,
        external_catalog_loader=None,
    )


#: Categories returned by :func:`classify_cuts`.
CUT_EVALUABLE: Final[str] = "evaluable"
CUT_PARTIAL: Final[str] = "partial"
CUT_NOT_APPLICABLE: Final[str] = "not_applicable"
CUT_UNBOUND_COLUMN: Final[str] = "unbound_column"
CUT_EXTERNAL_MEMBERSHIP: Final[str] = "external_membership"

def _all_mode_cuts(selection: Any) -> list[tuple[str, Any]]:
    """``(key, cut)`` for every cut of the mode; key is ``branch:subsample:cut`` (cut ids repeat across subsamples)."""
    spec = selection.spec
    cuts: list[tuple[str, Any]] = []
    if spec.branches:
        for branch in spec.branches:
            for sub in branch.subsamples or []:
                cuts += [(f"{branch.id}:{sub.id}:{c.id}", c) for c in sub.cuts if selection.mode in c.applies_to]
            cuts += [(f"{branch.id}:{c.id}", c) for c in (branch.cuts or []) if selection.mode in c.applies_to]
    else:
        cuts += [(c.id, c) for c in selection.cuts_for_mode()]
    return cuts


def classify_cuts(selection: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Which cuts of ``selection``'s frozen chain are evaluable on ``rows``, each cut run **independently**.

    A chain stops feeding a cut once an earlier cut removed every row, so chain attrition alone cannot say
    whether a downstream cut is evaluable. This applies every cut of the mode's cut list (subsample cuts
    included) to all enriched rows through the evaluator's own ``_apply_cut`` and reports, per cut key (``branch:subsample:cut`` for branched samples, the cut id otherwise):
    ``category`` (:data:`CUT_EVALUABLE`: every row passed or failed on real columns; :data:`CUT_PARTIAL`: some
    rows not applicable; :data:`CUT_NOT_APPLICABLE`: every row not applicable; :data:`CUT_UNBOUND_COLUMN`: the
    expression names a column no mock row carries, which the evaluator raises on (``CutExpressionError``; a
    chain never reaches such a cut only because an earlier cut already made every row not applicable);
    :data:`CUT_EXTERNAL_MEMBERSHIP`: ``in_sample`` of an external table, which a mock row
    can never be a member of), the three counts, and the not-applicable reasons.
    """
    from darkhunter_pop.sample_selection import CutExpressionError

    enriched = selection._enrich_rows_for_spec(selection._merge_reproduction_columns(list(rows)))
    membership = dict(selection._external_membership())
    bound = [selection.bind_row(r, membership=membership) for r in enriched]
    report: dict[str, dict[str, Any]] = {}
    for key, cut in _all_mode_cuts(selection):
        outcomes: dict[int, list[Any]] = {int(r["source_id"]): [] for r in bound}
        try:
            att = selection._apply_cut(cut, [dict(r) for r in bound], outcomes, membership)
        except CutExpressionError as exc:
            report[key] = {"category": CUT_UNBOUND_COLUMN, "passed": 0, "failed": 0,
                              "not_applicable": 0, "reasons": {str(exc): len(bound)}}
            continue
        n = att.n_in
        if att.n_not_applicable == n and n > 0:
            cat = CUT_NOT_APPLICABLE
        elif att.n_not_applicable > 0:
            cat = CUT_PARTIAL
        elif "in_sample(" in cut.expression:
            cat = CUT_EXTERNAL_MEMBERSHIP
        else:
            cat = CUT_EVALUABLE
        report[key] = {
            "category": cat,
            "passed": att.n_passed,
            "failed": att.n_failed,
            "not_applicable": att.n_not_applicable,
            "reasons": dict(att.not_applicable_reasons),
        }
    return report


def rows_columns(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Union of row keys (contract-test helper)."""
    keys: set[str] = set()
    for r in rows:
        keys.update(r)
    return keys
