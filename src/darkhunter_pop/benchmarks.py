"""Known-truth benchmarks and comparison-only catalogs (ARCHITECTURE.md §4).

Gaia-BH1/BH2: clean DR3 detection expectations.
Gaia-BH3: DR3 marginal/non-detection (RUWE≈3.4) with acceleration-catalog note.

All literature / external mass-function catalogs are comparison-only and must never
be wired as population inference priors (see ``population_model.allow_external_co_mf_priors``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import h5py
import numpy as np
import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    BenchmarkCatalogEntry,
    BenchmarksConfig,
    PipelineConfig,
)

Dr3Expectation = Literal["clean_detection", "marginal_or_non_detection"]
CatalogRole = Literal["comparison_only"]
#: Per-check outcome (#348). ``not_tested`` means the input needed to evaluate the
#: check was absent — never a pass.
CheckStatus = Literal["passed", "failed", "not_tested"]

REQUIRED_COMPARISON_CATALOG_IDS: tuple[str, ...] = (
    "ns_candidate_21",
    "companions_156",
    "amrf_binary_masses",
    "andrews",
    "shahaf",
    "pulsar_mf",
    "ligo_bh_mf",
)

EXTERNAL_MF_CATALOG_IDS: frozenset[str] = frozenset({"pulsar_mf", "ligo_bh_mf"})


@dataclass(frozen=True)
class KnownTruthSystem:
    """One known-truth benchmark system from the fixture table."""

    name: str
    source_id: int
    dr3_expectation: Dr3Expectation
    nss_orbital_expected: bool
    ruwe_approx: float | None
    acceleration_catalog_parts_of_orbit: bool
    literature: Mapping[str, Any]
    #: Published dark-companion mass and 1σ (fixture schema v2, #348).
    published_m2_msun: float | None = None
    published_m2_sigma_msun: float | None = None


@dataclass(frozen=True)
class KnownTruthTable:
    """Fixture-backed known-truth table with provenance."""

    schema_version: int
    table_id: str
    active_dr_mode: str
    provenance: Mapping[str, Any]
    systems: tuple[KnownTruthSystem, ...]
    path: Path

    def by_name(self) -> dict[str, KnownTruthSystem]:
        return {s.name: s for s in self.systems}

    def by_source_id(self) -> dict[int, KnownTruthSystem]:
        return {s.source_id: s for s in self.systems}


@dataclass(frozen=True)
class ObservedMass:
    """One pipeline M2 estimate for a known-truth source (one stage, one NSS row).

    ``not_computed_reason`` is set when the stage did not compute this M2 itself (it
    carried the upstream value through, e.g. ``joint_orbit_fit`` with no RVs). The
    mass check for that stage is then ``not_tested`` with this reason, never a pass
    credited to a stage that did no work (#382).
    """

    stage: str
    nss_solution_type: str | None
    m2_msun: float
    m2_sigma_msun: float | None
    not_computed_reason: str | None = None


@dataclass(frozen=True)
class ObservedBenchmark:
    """Observed pipeline state for one known-truth source.

    ``source`` records where the observation came from: ``pipeline_artifacts`` for a
    stage run (#348), ``synthetic_observed_from_truth`` for test scaffolding only.
    """

    source_id: int
    in_nss_orbital: bool
    ruwe: float | None = None
    in_acceleration_catalog: bool | None = None
    nss_solution_types: tuple[str, ...] = ()
    masses: tuple[ObservedMass, ...] = ()
    source: str = "supplied"


@dataclass(frozen=True)
class MassCheck:
    """Pipeline M2 vs published M2 for one stage / row (#348)."""

    stage: str
    nss_solution_type: str | None
    observed_m2_msun: float
    observed_sigma_msun: float | None
    published_m2_msun: float
    published_sigma_msun: float
    n_sigma_deviation: float
    status: CheckStatus
    #: Why the check is ``not_tested`` when the stage did not compute M2 (#382).
    not_tested_reason: str | None = None

    def one_line(self) -> str:
        sig = (
            f"{self.observed_sigma_msun:.3g}"
            if self.observed_sigma_msun is not None
            else "n/a"
        )
        if self.not_tested_reason is not None:
            return (
                f"{self.stage} [{self.nss_solution_type}]: not_tested — "
                f"{self.not_tested_reason} (carried M2={self.observed_m2_msun:.4g}"
                f"±{sig}, not credited to this stage)"
            )
        return (
            f"{self.stage} [{self.nss_solution_type}]: M2={self.observed_m2_msun:.4g}"
            f"±{sig} vs published {self.published_m2_msun:.4g}"
            f"±{self.published_sigma_msun:.3g} → {self.n_sigma_deviation:.2f}σ "
            f"({self.status})"
        )


@dataclass(frozen=True)
class KnownTruthCheckResult:
    """Outcome of one system's known-truth checks.

    ``status`` is ``failed`` if any component check failed, else ``not_tested`` if
    any component lacked its input, else ``passed``. ``passed`` is True only for
    ``passed``.
    """

    name: str
    source_id: int
    dr3_expectation: Dr3Expectation
    passed: bool
    details: str
    status: CheckStatus = "passed"
    mass_checks: tuple[MassCheck, ...] = ()


@dataclass(frozen=True)
class ComparisonCatalog:
    """Loaded comparison-only catalog (systems and/or mass-function samples)."""

    schema_version: int
    catalog_id: str
    role: CatalogRole
    never_as_prior: bool
    provenance: Mapping[str, Any]
    path: Path
    n_systems_expected: int | None = None
    systems: tuple[Mapping[str, Any], ...] = ()
    kind: str | None = None
    mass_msun: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_mass_function(self) -> bool:
        return self.kind == "mass_function" or bool(self.mass_msun)


def resolve_benchmark_path(path: str | Path) -> Path:
    """Resolve a benchmarks path relative to the repo root when not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark fixture root must be a mapping: {path}")
    return raw


def load_known_truth_table(path: str | Path) -> KnownTruthTable:
    """Load the Gaia BH known-truth fixture table."""
    resolved = resolve_benchmark_path(path)
    raw = _load_yaml_mapping(resolved)
    systems_raw = raw.get("systems")
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError(f"known-truth fixture missing systems list: {resolved}")
    systems: list[KnownTruthSystem] = []
    for entry in systems_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"known-truth system entry must be a mapping: {resolved}")
        expectation = entry.get("dr3_expectation")
        if expectation not in ("clean_detection", "marginal_or_non_detection"):
            raise ValueError(
                f"invalid dr3_expectation {expectation!r} in {resolved} "
                "(expected clean_detection|marginal_or_non_detection)"
            )
        systems.append(
            KnownTruthSystem(
                name=str(entry["name"]),
                source_id=int(entry["source_id"]),
                dr3_expectation=expectation,
                nss_orbital_expected=bool(entry.get("nss_orbital_expected", False)),
                ruwe_approx=(
                    float(entry["ruwe_approx"])
                    if entry.get("ruwe_approx") is not None
                    else None
                ),
                acceleration_catalog_parts_of_orbit=bool(
                    entry.get("acceleration_catalog_parts_of_orbit", False)
                ),
                literature=dict(entry.get("literature") or {}),
                published_m2_msun=(
                    float(entry["published_m2_msun"])
                    if entry.get("published_m2_msun") is not None
                    else None
                ),
                published_m2_sigma_msun=(
                    float(entry["published_m2_sigma_msun"])
                    if entry.get("published_m2_sigma_msun") is not None
                    else None
                ),
            )
        )
    return KnownTruthTable(
        schema_version=int(raw.get("schema_version", 1)),
        table_id=str(raw.get("table_id", resolved.stem)),
        active_dr_mode=str(raw.get("active_dr_mode", "dr3")),
        provenance=dict(raw.get("provenance") or {}),
        systems=tuple(systems),
        path=resolved,
    )


def load_known_truth_table_from_config(config: PipelineConfig) -> KnownTruthTable:
    """Load known-truth table using ``config.benchmarks.known_truth_path``."""
    return load_known_truth_table(config.benchmarks.known_truth_path)


def _combine_status(parts: Sequence[CheckStatus]) -> CheckStatus:
    if any(p == "failed" for p in parts):
        return "failed"
    if not parts or any(p == "not_tested" for p in parts):
        return "not_tested"
    return "passed"


def check_masses(
    system: KnownTruthSystem,
    masses: Sequence[ObservedMass],
    *,
    n_sigma: float,
) -> list[MassCheck]:
    """Compare each pipeline M2 against the published value (#348).

    Deviation is ``|M2_pipe − M2_pub| / sqrt(σ_pub² + σ_pipe²)`` (``σ_pipe`` omitted
    when absent or non-finite); ``failed`` above ``n_sigma``. An ``ObservedMass``
    with ``not_computed_reason`` set (the stage passed the upstream M2 through) is
    ``not_tested`` with that reason, whatever its deviation (#382).
    """
    if system.published_m2_msun is None or system.published_m2_sigma_msun is None:
        return []
    out: list[MassCheck] = []
    for m in masses:
        sig_pipe = (
            float(m.m2_sigma_msun)
            if m.m2_sigma_msun is not None and math.isfinite(m.m2_sigma_msun)
            else None
        )
        comb = math.hypot(float(system.published_m2_sigma_msun), sig_pipe or 0.0)
        dev = abs(float(m.m2_msun) - float(system.published_m2_msun)) / comb
        if m.not_computed_reason is not None or not math.isfinite(float(m.m2_msun)):
            status: CheckStatus = "not_tested"
        else:
            status = "passed" if dev <= float(n_sigma) else "failed"
        out.append(
            MassCheck(
                stage=m.stage,
                nss_solution_type=m.nss_solution_type,
                observed_m2_msun=float(m.m2_msun),
                observed_sigma_msun=sig_pipe,
                published_m2_msun=float(system.published_m2_msun),
                published_sigma_msun=float(system.published_m2_sigma_msun),
                n_sigma_deviation=float(dev),
                status=status,
                not_tested_reason=m.not_computed_reason,
            )
        )
    return out


def check_known_truth_expectations(
    table: KnownTruthTable,
    observed: Mapping[int, ObservedBenchmark] | Sequence[ObservedBenchmark],
    *,
    ruwe_match_tolerance: float,
    mass_n_sigma: float | None = None,
) -> list[KnownTruthCheckResult]:
    """Evaluate DR3 known-truth expectations against observed pipeline states.

    Parameters
    ----------
    table:
        Fixture-backed truth table.
    observed:
        Observed state keyed by ``source_id`` (or a sequence thereof). A system with
        no entry was not observed at all: every one of its checks is
        ``not_tested`` (no data), never passed.
    ruwe_match_tolerance:
        Absolute tolerance for ``ruwe_approx`` matches (from config).
    mass_n_sigma:
        Combined-sigma tolerance for the M2 check (``benchmarks.mass_check_n_sigma``).
        ``None`` disables the mass check (it is then reported ``not_tested`` for a
        clean detection with a published mass).

    Limitations
    -----------
    A clean detection whose observed state carries no pipeline M2 has its mass
    check ``not_tested``; the system can then be at best ``not_tested``.
    """
    if isinstance(observed, Mapping):
        by_id = dict(observed)
    else:
        by_id = {o.source_id: o for o in observed}

    results: list[KnownTruthCheckResult] = []
    for system in table.systems:
        obs = by_id.get(system.source_id)
        if obs is None:
            results.append(
                KnownTruthCheckResult(
                    name=system.name,
                    source_id=system.source_id,
                    dr3_expectation=system.dr3_expectation,
                    passed=False,
                    status="not_tested",
                    details="no observed state for source_id (no pipeline data) — not tested",
                )
            )
            continue

        parts: list[str] = []
        statuses: list[CheckStatus] = []
        if system.dr3_expectation == "clean_detection":
            det: CheckStatus = (
                "passed"
                if obs.in_nss_orbital is True and system.nss_orbital_expected is True
                else "failed"
            )
            statuses.append(det)
            parts.append(
                f"detection: in_nss_orbital={obs.in_nss_orbital} "
                f"(solution types {list(obs.nss_solution_types) or 'none'}, expected "
                f"an orbital solution) → {det}"
            )
        else:
            # marginal_or_non_detection (Gaia-BH3)
            if obs.in_nss_orbital:
                statuses.append("failed")
                parts.append("detection: unexpected NSS orbital solution → failed")
            else:
                statuses.append("passed")
                parts.append("detection: no NSS orbital solution (expected) → passed")
            if system.ruwe_approx is not None:
                if obs.ruwe is None:
                    statuses.append("not_tested")
                    parts.append(
                        f"RUWE: not available in pipeline data (expected ≈"
                        f"{system.ruwe_approx}) → not_tested"
                    )
                else:
                    delta = abs(float(obs.ruwe) - float(system.ruwe_approx))
                    ok = delta <= float(ruwe_match_tolerance)
                    statuses.append("passed" if ok else "failed")
                    parts.append(
                        f"RUWE={obs.ruwe:.4g} vs ≈{system.ruwe_approx} "
                        f"(tol={ruwe_match_tolerance}) → {'passed' if ok else 'failed'}"
                    )
            if system.acceleration_catalog_parts_of_orbit:
                parts.append(
                    "documented (not checked): would appear in acceleration catalog "
                    "for parts of orbit"
                )

        mass_checks: list[MassCheck] = []
        if system.published_m2_msun is not None:
            if mass_n_sigma is None:
                if system.dr3_expectation == "clean_detection":
                    statuses.append("not_tested")
                    parts.append("mass: check disabled → not_tested")
            else:
                mass_checks = check_masses(system, obs.masses, n_sigma=mass_n_sigma)
                if mass_checks:
                    statuses.extend(m.status for m in mass_checks)
                    parts.extend(f"mass: {m.one_line()}" for m in mass_checks)
                elif system.dr3_expectation == "clean_detection":
                    statuses.append("not_tested")
                    parts.append("mass: no pipeline M2 for this source → not_tested")

        status = _combine_status(statuses)
        results.append(
            KnownTruthCheckResult(
                name=system.name,
                source_id=system.source_id,
                dr3_expectation=system.dr3_expectation,
                passed=status == "passed",
                status=status,
                details="; ".join(parts),
                mass_checks=tuple(mass_checks),
            )
        )
    return results


def _read_candidate_rows(path: Path, source_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Targeted read of ``candidates/records_json`` rows for a few ``source_id``s.

    Every candidate-carrying stage artifact (``data_acquisition``, both
    ``mass_derivation`` stages, ``joint_orbit_fit``) shares this layout. Only the
    matching rows are decoded, so a 350k-row parent stays cheap.
    """
    wanted = np.asarray(list(source_ids), dtype=np.int64)
    with h5py.File(path, "r") as handle:
        if "candidates" not in handle or "source_ids" not in handle["candidates"]:
            return []
        sids = np.asarray(handle["candidates"]["source_ids"][:], dtype=np.int64)
        idx = np.nonzero(np.isin(sids, wanted))[0]
        records = handle["candidates"]["records_json"].asstr()
        return [json.loads(records[int(i)]) for i in idx]


def _m2_from_record(rec: Mapping[str, Any]) -> tuple[float, float | None] | None:
    m2 = rec.get("m2")
    if not isinstance(m2, Mapping):
        return None
    names = list(m2.get("names") or [])
    if "M2" not in names:
        return None
    i = names.index("M2")
    value = float(m2["values"][i])
    cov = m2.get("covariance")
    sigma: float | None = None
    if cov is not None:
        try:
            var = float(np.asarray(cov, dtype=np.float64)[i, i])
            sigma = math.sqrt(var) if var >= 0 else None
        except (IndexError, ValueError, TypeError):
            sigma = None
    return value, sigma


def _m2_fingerprint(rec: Mapping[str, Any]) -> str | None:
    """Canonical JSON of a record's full ``m2`` ParameterSet, for pass-through detection."""
    m2 = rec.get("m2")
    if not isinstance(m2, Mapping):
        return None
    return json.dumps(m2, sort_keys=True, default=str)


def _not_computed_reason(
    stage: str,
    rec: Mapping[str, Any],
    previous: tuple[str, str] | None,
) -> str | None:
    """Why ``stage`` did not compute this row's M2, or ``None`` if it did (#382).

    Two signals, checked in order:

    1. An explicit ``extras["<stage>_skip_reason"]`` the stage wrote itself (e.g.
       ``joint_orbit_fit_skip_reason = rv_astrometry_gate_failed`` for a system with
       no RVs, or an optimizer failure) — the stage kept the upstream M2.
    2. The row's full ``m2`` ParameterSet (values, covariance, provenance, units) is
       identical to the one the previous checked stage emitted for the same
       ``(source_id, nss_solution_type)`` row — a silent pass-through (e.g.
       ``mass_derivation_refined`` updates M1 only and never rewrites M2).

    Limitations
    -----------
    Signal 2 needs the previous stage's artifact; when it is absent, a silent
    pass-through with no skip marker cannot be detected and is checked as computed.
    """
    extras = rec.get("extras")
    if isinstance(extras, Mapping):
        skip = extras.get(f"{stage}_skip_reason")
        if skip not in (None, ""):
            return f"{stage} did not compute M2 (skip_reason={skip}), upstream M2 passed through"
    if previous is not None:
        prev_stage, prev_fp = previous
        if _m2_fingerprint(rec) == prev_fp:
            return (
                f"{stage} did not recompute M2 (M2 ParameterSet identical to "
                f"{prev_stage}'s), upstream M2 passed through"
            )
    return None


def observed_benchmarks_from_artifacts(
    table: KnownTruthTable,
    *,
    data_acquisition_artifact: Path | None,
    mass_artifacts: Mapping[str, Path],
    orbital_solution_types: Sequence[str],
) -> dict[int, ObservedBenchmark]:
    """Build observed states for the known-truth systems from real stage artifacts (#348).

    NSS membership, solution types and RUWE come from ``data_acquisition``; M2 comes
    from every artifact in ``mass_artifacts`` (stage name → path, **in pipeline
    order** — ``benchmarks.mass_check_stages``). A stage that carried the upstream M2
    through rather than computing it is marked via ``ObservedMass.not_computed_reason``
    (see :func:`_not_computed_reason`), so its check is ``not_tested`` (#382). With no
    ``data_acquisition`` artifact nothing was observed and an empty mapping is
    returned, so every check is ``not_tested``.
    """
    if data_acquisition_artifact is None or not Path(data_acquisition_artifact).is_file():
        return {}
    ids = [s.source_id for s in table.systems]
    orbital = set(orbital_solution_types)
    da_rows: dict[int, list[dict[str, Any]]] = {sid: [] for sid in ids}
    for rec in _read_candidate_rows(Path(data_acquisition_artifact), ids):
        da_rows[int(rec["source_id"])].append(rec)
    masses: dict[int, list[ObservedMass]] = {sid: [] for sid in ids}
    # (source_id, nss_solution_type) → (stage, m2 fingerprint) of the last checked
    # stage that emitted an M2 for that row; detects silent pass-through (#382).
    previous: dict[tuple[int, str | None], tuple[str, str]] = {}
    for stage, path in mass_artifacts.items():
        if path is None or not Path(path).is_file():
            continue
        emitted: dict[tuple[int, str | None], tuple[str, str]] = {}
        for rec in _read_candidate_rows(Path(path), ids):
            got = _m2_from_record(rec)
            if got is None:
                continue
            sid = int(rec["source_id"])
            row_key = (sid, rec.get("nss_solution_type"))
            masses[sid].append(
                ObservedMass(
                    stage=stage,
                    nss_solution_type=rec.get("nss_solution_type"),
                    m2_msun=got[0],
                    m2_sigma_msun=got[1],
                    not_computed_reason=_not_computed_reason(
                        stage, rec, previous.get(row_key)
                    ),
                )
            )
            fp = _m2_fingerprint(rec)
            if fp is not None:
                emitted[row_key] = (stage, fp)
        previous.update(emitted)
    out: dict[int, ObservedBenchmark] = {}
    for sid in ids:
        rows = da_rows[sid]
        types = tuple(str(r.get("nss_solution_type")) for r in rows)
        ruwes = [
            float(r["nss_orbital"]["ruwe"])
            for r in rows
            if isinstance(r.get("nss_orbital"), Mapping)
            and r["nss_orbital"].get("ruwe") is not None
        ]
        out[sid] = ObservedBenchmark(
            source_id=sid,
            in_nss_orbital=any(t in orbital for t in types),
            ruwe=ruwes[0] if ruwes else None,
            in_acceleration_catalog=None,
            nss_solution_types=types,
            masses=tuple(masses[sid]),
            source="pipeline_artifacts",
        )
    return out


def load_comparison_catalog(path: str | Path) -> ComparisonCatalog:
    """Load one comparison-only catalog fixture."""
    resolved = resolve_benchmark_path(path)
    raw = _load_yaml_mapping(resolved)
    role = raw.get("role", "comparison_only")
    if role != "comparison_only":
        raise ValueError(
            f"catalog {resolved} role must be 'comparison_only', got {role!r}"
        )
    never = bool(raw.get("never_as_prior", True))
    systems_raw = raw.get("systems") or []
    if systems_raw and not isinstance(systems_raw, list):
        raise ValueError(f"catalog systems must be a list: {resolved}")
    mass = tuple(float(x) for x in (raw.get("mass_msun") or []))
    weights = tuple(float(x) for x in (raw.get("weights") or []))
    if mass and weights and len(mass) != len(weights):
        raise ValueError(f"mass_msun/weights length mismatch in {resolved}")
    n_expected = raw.get("n_systems_expected")
    return ComparisonCatalog(
        schema_version=int(raw.get("schema_version", 1)),
        catalog_id=str(raw.get("catalog_id", resolved.stem)),
        role="comparison_only",
        never_as_prior=never,
        provenance=dict(raw.get("provenance") or {}),
        path=resolved,
        n_systems_expected=int(n_expected) if n_expected is not None else None,
        systems=tuple(dict(s) for s in systems_raw if isinstance(s, dict)),
        kind=str(raw["kind"]) if raw.get("kind") is not None else None,
        mass_msun=mass,
        weights=weights,
        raw=raw,
    )


def assert_comparison_only(catalog: ComparisonCatalog) -> None:
    """Hard rule: catalogs must be comparison-only and never priors."""
    if catalog.role != "comparison_only":
        raise ValueError(
            f"catalog {catalog.catalog_id!r} role={catalog.role!r}; "
            "external catalogs are comparison-only"
        )
    if not catalog.never_as_prior:
        raise ValueError(
            f"catalog {catalog.catalog_id!r} must set never_as_prior=true "
            "(ARCHITECTURE.md §4)"
        )


def assert_config_catalog_comparison_only(entry: BenchmarkCatalogEntry) -> None:
    """Validate a config catalog entry's comparison-only contract."""
    if entry.role != "comparison_only":
        raise ValueError(
            f"benchmarks catalog role must be comparison_only, got {entry.role!r}"
        )
    if entry.never_as_prior is not True:
        raise ValueError(
            "benchmarks catalog never_as_prior must be true "
            "(external catalogs are never inference priors)"
        )


def load_all_comparison_catalogs(
    config: PipelineConfig | BenchmarksConfig,
) -> dict[str, ComparisonCatalog]:
    """Load every configured comparison catalog and assert the hard rule."""
    bench = config.benchmarks if isinstance(config, PipelineConfig) else config
    out: dict[str, ComparisonCatalog] = {}
    for catalog_id, entry in bench.catalogs.items():
        assert_config_catalog_comparison_only(entry)
        catalog = load_comparison_catalog(entry.path)
        if catalog.catalog_id != catalog_id:
            # Allow stem mismatches only when fixture catalog_id matches config key.
            raise ValueError(
                f"config key {catalog_id!r} != fixture catalog_id "
                f"{catalog.catalog_id!r}"
            )
        assert_comparison_only(catalog)
        if catalog_id in EXTERNAL_MF_CATALOG_IDS and not catalog.is_mass_function:
            raise ValueError(
                f"{catalog_id} must be a mass_function comparison fixture"
            )
        out[catalog_id] = catalog
    return out


def assert_required_catalogs_present(catalogs: Mapping[str, ComparisonCatalog]) -> None:
    """Ensure the ARCHITECTURE.md §4 comparison catalog set is configured."""
    missing = [c for c in REQUIRED_COMPARISON_CATALOG_IDS if c not in catalogs]
    if missing:
        raise ValueError(f"missing required comparison catalogs: {missing}")


def format_known_truth_report(
    table: KnownTruthTable,
    results: Sequence[KnownTruthCheckResult],
    *,
    ruwe_match_tolerance: float,
    observed_source: str = "unspecified",
    mass_n_sigma: float | None = None,
) -> str:
    """Full-detail known-truth diagnostic report (caveman exemption)."""
    overall = _combine_status([r.status for r in results]) if results else "not_tested"
    lines = [
        "=== known-truth benchmarks (Gaia BH) ===",
        f"overall: {overall.upper()}",
        f"observed_source: {observed_source}",
        (
            "WARNING: observations are synthetic (built from the expectations "
            "themselves); this report tests nothing (#348)."
            if observed_source == "synthetic_observed_from_truth"
            else "observations: read from this run's stage artifacts"
            if observed_source == "pipeline_artifacts"
            else "observations: none — every check is not_tested"
            if observed_source == "none"
            else f"observations: {observed_source}"
        ),
        f"mass_check_n_sigma: {mass_n_sigma}",
        f"table_id: {table.table_id}",
        f"schema_version: {table.schema_version}",
        f"active_dr_mode: {table.active_dr_mode}",
        f"fixture_path: {table.path}",
        f"ruwe_match_tolerance: {ruwe_match_tolerance}",
        "provenance:",
    ]
    for key, value in table.provenance.items():
        lines.append(f"  {key}: {value}")
    lines.append("systems:")
    by_name = {r.name: r for r in results}
    for system in table.systems:
        lit = system.literature
        lines.append(f"  - {system.name} (source_id={system.source_id})")
        lines.append(f"    dr3_expectation: {system.dr3_expectation}")
        lines.append(f"    nss_orbital_expected: {system.nss_orbital_expected}")
        lines.append(f"    ruwe_approx: {system.ruwe_approx}")
        lines.append(
            f"    published_m2_msun: {system.published_m2_msun} "
            f"± {system.published_m2_sigma_msun}"
        )
        lines.append(
            "    acceleration_catalog_parts_of_orbit: "
            f"{system.acceleration_catalog_parts_of_orbit}"
        )
        if lit:
            lines.append(f"    literature.citation: {lit.get('citation')}")
            if lit.get("notes"):
                lines.append(f"    literature.notes: {lit.get('notes')}")
        check = by_name.get(system.name)
        if check is None:
            lines.append("    check: (not evaluated)")
        else:
            lines.append(f"    check.status: {check.status}")
            for part in check.details.split("; "):
                lines.append(f"    - {part}")
    n_pass = sum(1 for r in results if r.status == "passed")
    n_fail = sum(1 for r in results if r.status == "failed")
    n_nt = sum(1 for r in results if r.status == "not_tested")
    lines.append(
        f"summary: passed={n_pass} failed={n_fail} not_tested={n_nt} "
        f"of {len(results)}"
    )
    lines.append("=== end known-truth benchmarks ===")
    return "\n".join(lines)


def format_comparison_catalog_report(
    catalogs: Mapping[str, ComparisonCatalog],
) -> str:
    """Full-detail comparison-catalog diagnostic report (caveman exemption)."""
    lines = [
        "=== comparison-only catalogs ===",
        "status: FIXTURE LISTING ONLY — no comparison against this run is computed "
        "(#348). Nothing below is a validation of the pipeline.",
        "hard_rule: external compact-object populations are never inference priors",
        f"n_catalogs: {len(catalogs)}",
    ]
    for catalog_id in sorted(catalogs):
        cat = catalogs[catalog_id]
        lines.append(f"  - {catalog_id}")
        lines.append(f"    completeness: {catalog_completeness(cat)}")
        lines.append(f"    role: {cat.role}")
        lines.append(f"    never_as_prior: {cat.never_as_prior}")
        lines.append(f"    path: {cat.path}")
        lines.append(f"    schema_version: {cat.schema_version}")
        if cat.n_systems_expected is not None:
            lines.append(f"    n_systems_expected: {cat.n_systems_expected}")
        lines.append(f"    n_systems_in_fixture: {len(cat.systems)}")
        if cat.is_mass_function:
            lines.append(f"    kind: mass_function")
            lines.append(f"    n_mass_samples: {len(cat.mass_msun)}")
            if catalog_id in EXTERNAL_MF_CATALOG_IDS:
                lines.append(
                    "    caveat: pulsar/LIGO MF comparison-only — "
                    "forbidden as population prior"
                )
        prov = cat.provenance
        if prov.get("citation"):
            lines.append(f"    provenance.citation: {prov.get('citation')}")
        if prov.get("caveats"):
            lines.append(f"    provenance.caveats: {prov.get('caveats')}")
        if prov.get("notes"):
            lines.append(f"    provenance.notes: {prov.get('notes')}")
    lines.append("=== end comparison-only catalogs ===")
    return "\n".join(lines)


def catalog_completeness(cat: ComparisonCatalog) -> str:
    """``complete`` / ``incomplete (n of N)`` / ``empty`` / ``unknown`` (#348)."""
    if cat.is_mass_function:
        return (
            f"mass-function fixture ({len(cat.mass_msun)} samples; completeness "
            "against the published mass function not checked)"
            if cat.mass_msun
            else "empty (mass-function fixture with 0 samples)"
        )
    n = len(cat.systems)
    if cat.n_systems_expected is None:
        return f"unknown ({n} systems, no n_systems_expected)" if n else "empty"
    if n == 0:
        return f"empty (0 of {cat.n_systems_expected} expected systems)"
    if n < cat.n_systems_expected:
        return f"incomplete ({n} of {cat.n_systems_expected} expected systems)"
    return f"complete ({n} of {cat.n_systems_expected})"


def synthetic_observed_from_truth(
    table: KnownTruthTable,
    *,
    bh3_ruwe: float | None = None,
) -> dict[int, ObservedBenchmark]:
    """Build fixture-consistent observed states that satisfy known-truth expectations.

    **Tests only** (#348): it fabricates observations that satisfy the expectations,
    so any check run on its output passes by construction. No stage path calls it;
    the diagnostics stage builds observations with
    :func:`observed_benchmarks_from_artifacts`.
    """
    out: dict[int, ObservedBenchmark] = {}
    for system in table.systems:
        if system.dr3_expectation == "clean_detection":
            out[system.source_id] = ObservedBenchmark(
                source_id=system.source_id,
                in_nss_orbital=True,
                ruwe=1.1,
                in_acceleration_catalog=False,
                source="synthetic_observed_from_truth",
            )
        else:
            ruwe = (
                float(bh3_ruwe)
                if bh3_ruwe is not None
                else float(system.ruwe_approx if system.ruwe_approx is not None else 3.4)
            )
            out[system.source_id] = ObservedBenchmark(
                source_id=system.source_id,
                in_nss_orbital=False,
                ruwe=ruwe,
                in_acceleration_catalog=None,
                source="synthetic_observed_from_truth",
            )
    return out


def fetch_live_comparison_catalog(
    catalog_id: str,
    *,
    url: str | None = None,
) -> dict[str, Any]:
    """Optional live catalog pull (network).

    Fixture loaders are the default path. This helper documents the live-pull
    surface for ``@pytest.mark.network`` / ``slow`` tests and refuses to treat
    any result as an inference prior.

    Raises
    ------
    NotImplementedError
        Live pulls are opt-in and not required for the merge gate; call sites
        must use fixtures unless a concrete URL/protocol is supplied by a
        follow-up issue.
    """
    _ = (catalog_id, url)
    raise NotImplementedError(
        f"live pull for {catalog_id!r} is optional (network/slow); "
        "use fixture loaders for required CI. Comparison-only — never a prior."
    )


def validate_benchmarks_config(config: BenchmarksConfig | PipelineConfig) -> None:
    """Validate benchmarks fragment paths and comparison-only hard rule."""
    bench = config.benchmarks if isinstance(config, PipelineConfig) else config
    if not bench.known_truth_path:
        raise ValueError("benchmarks.known_truth_path must be set")
    for catalog_id, entry in bench.catalogs.items():
        assert_config_catalog_comparison_only(entry)
        if catalog_id in EXTERNAL_MF_CATALOG_IDS and not entry.never_as_prior:
            raise ValueError(f"{catalog_id} must set never_as_prior=true")
    missing = [c for c in REQUIRED_COMPARISON_CATALOG_IDS if c not in bench.catalogs]
    if missing:
        raise ValueError(f"benchmarks.catalogs missing required ids: {missing}")
