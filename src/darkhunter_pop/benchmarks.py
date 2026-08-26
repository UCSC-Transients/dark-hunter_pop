"""Known-truth benchmarks and comparison-only catalogs (ARCHITECTURE.md §4).

Gaia-BH1/BH2: clean DR3 detection expectations.
Gaia-BH3: DR3 marginal/non-detection (RUWE≈3.4) with acceleration-catalog note.

All literature / external mass-function catalogs are comparison-only and must never
be wired as population inference priors (see ``population_model.allow_external_co_mf_priors``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.config_schema import (
    BenchmarkCatalogEntry,
    BenchmarksConfig,
    PipelineConfig,
)

Dr3Expectation = Literal["clean_detection", "marginal_or_non_detection"]
CatalogRole = Literal["comparison_only"]

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
class ObservedBenchmark:
    """Observed pipeline state for one known-truth source (fixture or live)."""

    source_id: int
    in_nss_orbital: bool
    ruwe: float | None = None
    in_acceleration_catalog: bool | None = None


@dataclass(frozen=True)
class KnownTruthCheckResult:
    """Pass/fail for one known-truth expectation check."""

    name: str
    source_id: int
    dr3_expectation: Dr3Expectation
    passed: bool
    details: str


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


def check_known_truth_expectations(
    table: KnownTruthTable,
    observed: Mapping[int, ObservedBenchmark] | Sequence[ObservedBenchmark],
    *,
    ruwe_match_tolerance: float,
) -> list[KnownTruthCheckResult]:
    """Evaluate DR3 known-truth expectations against observed states.

    Parameters
    ----------
    table:
        Fixture-backed truth table.
    observed:
        Observed NSS/RUWE state keyed by ``source_id`` (or a sequence thereof).
    ruwe_match_tolerance:
        Absolute tolerance for ``ruwe_approx`` matches (from config).
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
                    details="missing observed state for source_id",
                )
            )
            continue

        if system.dr3_expectation == "clean_detection":
            ok = obs.in_nss_orbital is True and system.nss_orbital_expected is True
            detail = (
                f"clean_detection: in_nss_orbital={obs.in_nss_orbital} "
                f"(expected True)"
            )
            results.append(
                KnownTruthCheckResult(
                    name=system.name,
                    source_id=system.source_id,
                    dr3_expectation=system.dr3_expectation,
                    passed=ok,
                    details=detail,
                )
            )
            continue

        # marginal_or_non_detection (Gaia-BH3)
        parts: list[str] = []
        ok = True
        if obs.in_nss_orbital:
            ok = False
            parts.append("unexpected NSS orbital detection")
        else:
            parts.append("no NSS orbital detection (expected)")

        if system.ruwe_approx is not None:
            if obs.ruwe is None:
                ok = False
                parts.append(
                    f"missing RUWE (expected ≈{system.ruwe_approx})"
                )
            else:
                delta = abs(float(obs.ruwe) - float(system.ruwe_approx))
                if delta > float(ruwe_match_tolerance):
                    ok = False
                    parts.append(
                        f"RUWE={obs.ruwe} outside ±{ruwe_match_tolerance} of "
                        f"{system.ruwe_approx}"
                    )
                else:
                    parts.append(
                        f"RUWE={obs.ruwe} matches ≈{system.ruwe_approx} "
                        f"(tol={ruwe_match_tolerance})"
                    )

        if system.acceleration_catalog_parts_of_orbit:
            parts.append(
                "documented: would appear in acceleration catalog for parts of orbit"
            )
            if obs.in_acceleration_catalog is False:
                # Soft documentation check: false is noted but does not fail when
                # acceleration membership is unknown/untested in the fixture path.
                parts.append(
                    "observed acceleration-catalog flag=False "
                    "(documentation expectation remains)"
                )

        results.append(
            KnownTruthCheckResult(
                name=system.name,
                source_id=system.source_id,
                dr3_expectation=system.dr3_expectation,
                passed=ok,
                details="; ".join(parts),
            )
        )
    return results


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
) -> str:
    """Full-detail known-truth diagnostic report (caveman exemption)."""
    lines = [
        "=== known-truth benchmarks (Gaia BH) ===",
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
            lines.append(f"    check.passed: {check.passed}")
            lines.append(f"    check.details: {check.details}")
    n_pass = sum(1 for r in results if r.passed)
    lines.append(f"summary: passed={n_pass}/{len(results)}")
    lines.append("=== end known-truth benchmarks ===")
    return "\n".join(lines)


def format_comparison_catalog_report(
    catalogs: Mapping[str, ComparisonCatalog],
) -> str:
    """Full-detail comparison-catalog diagnostic report (caveman exemption)."""
    lines = [
        "=== comparison-only catalogs ===",
        "hard_rule: external compact-object populations are never inference priors",
        f"n_catalogs: {len(catalogs)}",
    ]
    for catalog_id in sorted(catalogs):
        cat = catalogs[catalog_id]
        lines.append(f"  - {catalog_id}")
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


def synthetic_observed_from_truth(
    table: KnownTruthTable,
    *,
    bh3_ruwe: float | None = None,
) -> dict[int, ObservedBenchmark]:
    """Build fixture-consistent observed states that satisfy known-truth expectations.

    Used by unit tests and scaffolding demos — not a live Gaia query.
    """
    out: dict[int, ObservedBenchmark] = {}
    for system in table.systems:
        if system.dr3_expectation == "clean_detection":
            out[system.source_id] = ObservedBenchmark(
                source_id=system.source_id,
                in_nss_orbital=True,
                ruwe=1.1,
                in_acceleration_catalog=False,
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
