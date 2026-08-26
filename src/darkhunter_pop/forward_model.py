"""Stages: ``selection_function_astrometric`` and ``selection_function_followup``.

``selection_function_astrometric`` wraps vendored ``gaiamock_mod`` (modified RUWE) only via
``import_gaiamock_mod()``. Records the gaiamock version triple and refuses on mismatch.
Includes the El-Badry et al. (2024) six-panel mock-vs-real validation gate and a solution-type
fraction diagnostic. DR3 execution only in v1 (ARCHITECTURE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from darkhunter_pop.config_loader import repo_root, require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    ExtinctionModel,
    PipelineConfig,
    SelectionFunctionAstrometricConfig,
)
from darkhunter_pop.gaiamock_vendor import (
    GaiamockModVersions,
    assert_versions_match,
    read_versions,
    import_gaiamock_mod,
)
from darkhunter_pop.schemas import ActiveDRMode

# Six El-Badry et al. (2024) comparison panels (ARCHITECTURE.md §4).
SIX_PANEL_NAMES: tuple[str, ...] = (
    "P_orb_days",
    "G_mag",
    "inv_parallax_mas_inv",
    "eccentricity",
    "f_m_msun",
    "cos_inclination",
)

SOLUTION_TYPE_LABELS: tuple[str, ...] = (
    "insufficient_visibility",
    "five_parameter",
    "seven_parameter",
    "nine_parameter",
    "twelve_parameter_orbital",
    "orbital_failed_cuts",
)


class SolutionType(str, Enum):
    """Gaia astrometric cascade outcome for one mock realization."""

    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    FIVE_PARAMETER = "five_parameter"
    SEVEN_PARAMETER = "seven_parameter"
    NINE_PARAMETER = "nine_parameter"
    TWELVE_PARAMETER_ORBITAL = "twelve_parameter_orbital"
    ORBITAL_FAILED_CUTS = "orbital_failed_cuts"


@dataclass(frozen=True)
class MockRealizationRecord:
    """One gaiamock mock injection outcome."""

    solution_type: SolutionType
    accepted_orbital: bool
    P_orb_days: float | None = None
    G_mag: float | None = None
    inv_parallax_mas_inv: float | None = None
    eccentricity: float | None = None
    f_m_msun: float | None = None
    cos_inclination: float | None = None


@dataclass
class SixPanelValidationResult:
    """KS-test results for each El-Badry comparison panel."""

    panel_names: tuple[str, ...]
    ks_statistics: dict[str, float] = field(default_factory=dict)
    ks_pvalues: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return bool(self.passed) and all(self.passed.values())


@dataclass
class SolutionTypeFractionResult:
    """Mock vs real solution-type fraction comparison."""

    mock_fractions: dict[str, float]
    real_fractions: dict[str, float]
    max_abs_delta: float
    passed: bool


@dataclass
class ValidationGateResult:
    """Combined blocking validation gate for science trust."""

    six_panel: SixPanelValidationResult
    solution_type: SolutionTypeFractionResult
    detection_fraction: float
    n_mock: int
    n_real: int

    @property
    def passed(self) -> bool:
        return self.six_panel.all_passed and self.solution_type.passed


@dataclass
class SelectionFunctionAstrometricResult:
    """Stage output summary before HDF5 serialization."""

    gaiamock_versions: GaiamockModVersions
    records: list[MockRealizationRecord]
    validation: ValidationGateResult
    data_release: str


def expected_gaiamock_versions(config: PipelineConfig) -> GaiamockModVersions:
    """Build the version triple expected from config pins (fills unset pins from install)."""
    installed = read_versions(release=config.gaiamock.mod_release)
    return GaiamockModVersions(
        gaiamock_mod_release=config.gaiamock.mod_release,
        gaiamock_mod_sha256=config.gaiamock.mod_sha256 or installed.gaiamock_mod_sha256,
        gaiamock_git_commit=config.gaiamock.git_commit or installed.gaiamock_git_commit,
    )


def verify_gaiamock_versions(config: PipelineConfig) -> GaiamockModVersions:
    """Refuse if installed gaiamock differs from config pins / recorded triple."""
    expected = expected_gaiamock_versions(config)
    assert_versions_match(expected)
    return read_versions(release=config.gaiamock.mod_release)


def _do_dust_from_config(config: SelectionFunctionAstrometricConfig) -> bool:
    if config.extinction_model is ExtinctionModel.COMBINED19:
        return True
    if config.extinction_model is ExtinctionModel.NONE:
        return False
    raise ValueError(f"unsupported extinction_model: {config.extinction_model}")


def classify_cascade_result(
    cascade: Sequence[float],
    *,
    m1_msun: float,
    m2_msun: float,
    flux_ratio: float,
    gaiamock: ModuleType | None = None,
) -> MockRealizationRecord:
    """Map ``fit_full_astrometric_cascade`` return vector to solution type + six panels.

    Return layout (gaiamock_mod): plx, sig_parallax, A, sig_A, B, sig_B, F, sig_F, G, sig_G,
    period, sig_period, phi_p, sig_phi_p, ecc, sig_ecc, inc_deg, a0_mas, sigma_a0_mas,
    N_visibility_periods, N_obs, F2, ruwe.
    """
    res = list(cascade)
    if len(res) < 23:
        res = res + [0.0] * (23 - len(res))
    plx = float(res[0])
    sig_parallax = float(res[1])
    period = float(res[10])
    sig_period = float(res[11])
    ecc = float(res[14])
    sig_ecc = float(res[15])
    inc_deg = float(res[16])
    a0_mas = float(res[17])
    sigma_a0_mas = float(res[18])

    if plx == 0.0:
        return MockRealizationRecord(
            solution_type=SolutionType.INSUFFICIENT_VISIBILITY,
            accepted_orbital=False,
        )
    if plx < 0:
        mapping = {
            -1.0: SolutionType.FIVE_PARAMETER,
            -7.0: SolutionType.SEVEN_PARAMETER,
            -9.0: SolutionType.NINE_PARAMETER,
        }
        stype = mapping.get(plx)
        if stype is None:
            raise ValueError(f"unexpected cascade sentinel plx={plx}")
        return MockRealizationRecord(solution_type=stype, accepted_orbital=False)

    if period <= 0 or sig_parallax <= 0 or sigma_a0_mas <= 0:
        return MockRealizationRecord(
            solution_type=SolutionType.ORBITAL_FAILED_CUTS,
            accepted_orbital=False,
        )

    a0_over_err = a0_mas / sigma_a0_mas
    plx_over_err = plx / sig_parallax
    passed_cuts = (
        (a0_over_err > 5.0)
        and (plx_over_err > 20000.0 / period)
        and (a0_over_err > 158.0 / np.sqrt(period))
        and (sig_ecc < 0.079 * np.log(period) - 0.244)
    )
    stype = (
        SolutionType.TWELVE_PARAMETER_ORBITAL
        if passed_cuts
        else SolutionType.ORBITAL_FAILED_CUTS
    )

    f_m: float | None = None
    if passed_cuts and a0_mas > 0 and gaiamock is not None:
        try:
            f_m = float(
                gaiamock.get_companion_mass_from_mass_function(
                    M1=m1_msun,
                    a0_mas=a0_mas,
                    period=period,
                    parallax=plx,
                    fluxratio=flux_ratio,
                )
            )
        except Exception:
            f_m = None

    inv_parallax = 1.0 / plx if plx > 0 else None
    cos_inc = float(np.cos(np.radians(inc_deg))) if inc_deg == inc_deg else None

    return MockRealizationRecord(
        solution_type=stype,
        accepted_orbital=passed_cuts,
        P_orb_days=period if passed_cuts else None,
        inv_parallax_mas_inv=inv_parallax,
        eccentricity=ecc if passed_cuts else None,
        f_m_msun=f_m,
        cos_inclination=cos_inc,
    )


def solution_type_fractions(
    records: Sequence[MockRealizationRecord],
) -> dict[str, float]:
    """Fraction of realizations in each cascade outcome bin."""
    counts = {label: 0 for label in SOLUTION_TYPE_LABELS}
    if not records:
        return counts
    for rec in records:
        counts[rec.solution_type.value] += 1
    n = float(len(records))
    return {key: val / n for key, val in counts.items()}


def _panel_values(
    records: Sequence[MockRealizationRecord],
    panel: str,
    *,
    g_mag: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Extract one six-panel axis from accepted orbital records."""
    values: list[float] = []
    for i, rec in enumerate(records):
        if not rec.accepted_orbital:
            continue
        if panel == "P_orb_days" and rec.P_orb_days is not None:
            values.append(rec.P_orb_days)
        elif panel == "G_mag":
            if g_mag is not None and i < len(g_mag):
                values.append(float(g_mag[i]))
            elif rec.G_mag is not None:
                values.append(rec.G_mag)
        elif panel == "inv_parallax_mas_inv" and rec.inv_parallax_mas_inv is not None:
            values.append(rec.inv_parallax_mas_inv)
        elif panel == "eccentricity" and rec.eccentricity is not None:
            values.append(rec.eccentricity)
        elif panel == "f_m_msun" and rec.f_m_msun is not None:
            values.append(rec.f_m_msun)
        elif panel == "cos_inclination" and rec.cos_inclination is not None:
            values.append(rec.cos_inclination)
    return np.asarray(values, dtype=np.float64)


def run_six_panel_validation(
    mock_records: Sequence[MockRealizationRecord],
    real_panels: Mapping[str, NDArray[np.floating]],
    *,
    ks_pvalue_min: float,
    g_mag: NDArray[np.floating] | None = None,
) -> SixPanelValidationResult:
    """Two-sample KS tests for El-Badry six panels (mock vs real DR3 NSS)."""
    result = SixPanelValidationResult(panel_names=SIX_PANEL_NAMES)
    for panel in SIX_PANEL_NAMES:
        mock_vals = _panel_values(mock_records, panel, g_mag=g_mag)
        real_vals = np.asarray(real_panels.get(panel, []), dtype=np.float64)
        if len(mock_vals) < 2 or len(real_vals) < 2:
            result.ks_statistics[panel] = float("nan")
            result.ks_pvalues[panel] = 0.0
            result.passed[panel] = False
            continue
        stat, pvalue = stats.ks_2samp(mock_vals, real_vals)
        result.ks_statistics[panel] = float(stat)
        result.ks_pvalues[panel] = float(pvalue)
        result.passed[panel] = pvalue >= ks_pvalue_min
    return result


def run_solution_type_validation(
    mock_records: Sequence[MockRealizationRecord],
    real_fractions: Mapping[str, float],
    *,
    max_abs_delta: float,
) -> SolutionTypeFractionResult:
    """Compare mock vs real Gaia solution-type fractions."""
    mock_frac = solution_type_fractions(mock_records)
    deltas = [
        abs(mock_frac.get(label, 0.0) - real_fractions.get(label, 0.0))
        for label in SOLUTION_TYPE_LABELS
    ]
    max_delta = max(deltas) if deltas else 0.0
    return SolutionTypeFractionResult(
        mock_fractions=mock_frac,
        real_fractions=dict(real_fractions),
        max_abs_delta=max_delta,
        passed=max_delta <= max_abs_delta,
    )


def _resolve_reference_path(config: PipelineConfig) -> Path | None:
    ref = config.selection_function_astrometric.validation_gate.reference_path
    if ref is None:
        default = (
            repo_root()
            / "tests"
            / "fixtures"
            / "elbadry2024_dr3_nss_reference.npz"
        )
        return default if default.is_file() else None
    path = Path(ref)
    if not path.is_absolute():
        path = repo_root() / path
    return path if path.is_file() else None


def load_reference_panels(config: PipelineConfig) -> tuple[dict[str, NDArray[np.float64]], dict[str, float]]:
    """Load bundled or configured El-Badry DR3 NSS reference histograms."""
    path = _resolve_reference_path(config)
    if path is None:
        raise FileNotFoundError(
            "no El-Badry reference panels: set validation_gate.reference_path or "
            "install tests/fixtures/elbadry2024_dr3_nss_reference.npz"
        )
    with np.load(path, allow_pickle=False) as data:
        panels = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in SIX_PANEL_NAMES
            if name in data
        }
        st_frac = {
            label: float(data[f"solution_type_frac_{label}"])
            for label in SOLUTION_TYPE_LABELS
            if f"solution_type_frac_{label}" in data
        }
    missing = [p for p in SIX_PANEL_NAMES if p not in panels]
    if missing:
        raise ValueError(f"reference file missing panels: {missing}")
    if not st_frac:
        raise ValueError("reference file missing solution_type_frac_* keys")
    return panels, st_frac


def load_real_panels_from_data_acquisition(
    artifact_path: Path,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, float]]:
    """Read real DR3 NSS six-panel samples + solution-type fractions from data_acquisition HDF5."""
    with h5py.File(artifact_path, "r") as handle:
        if "data_acquisition/nss_panels" not in handle:
            raise KeyError(
                f"{artifact_path} lacks data_acquisition/nss_panels group"
            )
        grp = handle["data_acquisition/nss_panels"]
        panels = {
            name: np.asarray(grp[name], dtype=np.float64) for name in SIX_PANEL_NAMES
        }
        st_grp = handle.get("data_acquisition/solution_type_fractions")
        if st_grp is None:
            raise KeyError(
                f"{artifact_path} lacks data_acquisition/solution_type_fractions"
            )
        st_frac = {label: float(st_grp[label][()]) for label in SOLUTION_TYPE_LABELS}
    return panels, st_frac


def _run_single_mock_realization(
    gaiamock: ModuleType,
    *,
    ra: float,
    dec: float,
    d_pc: float,
    phot_g_mean_mag: float,
    config: PipelineConfig,
    c_funcs: Any,
) -> MockRealizationRecord:
    pop = config.selection_function_astrometric.mock_population
    parallax = 1000.0 / d_pc
    Tp = float(np.random.uniform(0.0, pop.period_days))
    omega = float(np.random.uniform(0.0, 2.0 * np.pi))
    w = float(np.random.uniform(0.0, 2.0 * np.pi))
    inc_deg = float(np.degrees(np.arccos(np.random.uniform(-1.0, 1.0))))

    data_release = config.active_dr_mode.value
    cascade = gaiamock.run_full_astrometric_cascade(
        ra=ra,
        dec=dec,
        parallax=parallax,
        pmra=0.0,
        pmdec=0.0,
        m1=pop.m1_msun,
        m2=pop.m2_msun,
        period=pop.period_days,
        Tp=Tp,
        ecc=pop.eccentricity,
        omega=omega,
        inc_deg=inc_deg,
        w=w,
        phot_g_mean_mag=phot_g_mean_mag,
        f=pop.flux_ratio,
        data_release=data_release,
        c_funcs=c_funcs,
        verbose=False,
        show_residuals=False,
        ruwe_min=pop.ruwe_min,
        skip_acceleration=pop.skip_acceleration,
    )
    rec = classify_cascade_result(
        cascade,
        m1_msun=pop.m1_msun,
        m2_msun=pop.m2_msun,
        flux_ratio=pop.flux_ratio,
        gaiamock=gaiamock,
    )
    if rec.accepted_orbital:
        return MockRealizationRecord(
            solution_type=rec.solution_type,
            accepted_orbital=True,
            P_orb_days=rec.P_orb_days,
            G_mag=phot_g_mean_mag,
            inv_parallax_mas_inv=rec.inv_parallax_mas_inv,
            eccentricity=rec.eccentricity,
            f_m_msun=rec.f_m_msun,
            cos_inclination=rec.cos_inclination,
        )
    return rec


def run_mock_injections(
    config: PipelineConfig,
    gaiamock: ModuleType,
) -> tuple[list[MockRealizationRecord], NDArray[np.float64]]:
    """Monte Carlo mock binaries through the gaiamock DR3 cascade."""
    require_dr3_active_for_v1(config)
    if config.active_dr_mode is not ActiveDRMode.DR3:
        raise ValueError("selection_function_astrometric: DR4 execution not enabled in v1")

    pop = config.selection_function_astrometric.mock_population
    path_cfg = config.active_dr().selection_function_astrometric
    do_dust = _do_dust_from_config(config.selection_function_astrometric)

    ra, dec, d_pc, _x, _y, _z = (
        gaiamock.generate_coordinates_at_a_given_distance_exponential_disk(
            d_min=path_cfg.d_min_pc,
            d_max=path_cfg.d_max_pc,
            N_stars=pop.N_realizations,
            hz_pc=pop.hz_pc,
        )
    )
    l_deg, b_deg = gaiamock.xyz_to_galactic(x=_x, y=_y, z=_z)

    if do_dust:
        try:
            import mwdust
        except ImportError as exc:
            raise ImportError(
                "extinction_model=combined19 requires the mwdust package; "
                "pip install -e '.[gaiamock]' or set extinction_model=none"
            ) from exc
        combined19_ebv = mwdust.Combined19()
        ebv = combined19_ebv(l_deg, b_deg, d_pc / 1000.0)
        a_g = 2.80 * ebv
    else:
        a_g = np.zeros(pop.N_realizations)

    phot_g = pop.Mg_tot + 5.0 * np.log10(d_pc / 10.0) + a_g
    c_funcs = gaiamock.read_in_C_functions()

    records: list[MockRealizationRecord] = []
    for i in range(pop.N_realizations):
        records.append(
            _run_single_mock_realization(
                gaiamock,
                ra=float(ra[i]),
                dec=float(dec[i]),
                d_pc=float(d_pc[i]),
                phot_g_mean_mag=float(phot_g[i]),
                config=config,
                c_funcs=c_funcs,
            )
        )
    return records, np.asarray(phot_g, dtype=np.float64)


def run_validation_gate(
    config: PipelineConfig,
    mock_records: Sequence[MockRealizationRecord],
    *,
    real_panels: Mapping[str, NDArray[np.floating]],
    real_solution_fractions: Mapping[str, float],
    g_mag: NDArray[np.floating] | None = None,
) -> ValidationGateResult:
    """El-Badry six-panel + solution-type blocking gate."""
    gate = config.selection_function_astrometric.validation_gate
    six = run_six_panel_validation(
        mock_records,
        real_panels,
        ks_pvalue_min=gate.ks_pvalue_min,
        g_mag=g_mag,
    )
    st = run_solution_type_validation(
        mock_records,
        real_solution_fractions,
        max_abs_delta=gate.solution_type_fraction_max_abs_delta,
    )
    n_acc = sum(1 for r in mock_records if r.accepted_orbital)
    n_mock = len(mock_records)
    real_n = sum(
        int(round(frac * max(n_mock, 1)))
        for frac in real_solution_fractions.values()
    )
    return ValidationGateResult(
        six_panel=six,
        solution_type=st,
        detection_fraction=n_acc / n_mock if n_mock else 0.0,
        n_mock=n_mock,
        n_real=real_n,
    )


def write_selection_function_artifact(
    path: Path,
    result: SelectionFunctionAstrometricResult,
) -> None:
    """Persist one HDF5 artifact for ``selection_function_astrometric``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "selection_function_astrometric"
        handle.attrs["data_release"] = result.data_release
        handle.attrs["gaiamock_mod_release"] = result.gaiamock_versions.gaiamock_mod_release
        handle.attrs["gaiamock_mod_sha256"] = result.gaiamock_versions.gaiamock_mod_sha256
        handle.attrs["gaiamock_git_commit"] = result.gaiamock_versions.gaiamock_git_commit
        handle.attrs["validation_gate_passed"] = result.validation.passed
        handle.attrs["detection_fraction"] = result.validation.detection_fraction

        vg = handle.create_group("validation_gate")
        sp = vg.create_group("six_panel")
        for name in SIX_PANEL_NAMES:
            sp.attrs[f"ks_statistic_{name}"] = result.validation.six_panel.ks_statistics.get(
                name, float("nan")
            )
            sp.attrs[f"ks_pvalue_{name}"] = result.validation.six_panel.ks_pvalues.get(
                name, float("nan")
            )
            sp.attrs[f"passed_{name}"] = result.validation.six_panel.passed.get(name, False)

        st = vg.create_group("solution_type_fractions")
        for label in SOLUTION_TYPE_LABELS:
            st.attrs[f"mock_{label}"] = result.validation.solution_type.mock_fractions.get(
                label, 0.0
            )
            st.attrs[f"real_{label}"] = result.validation.solution_type.real_fractions.get(
                label, 0.0
            )
        st.attrs["max_abs_delta"] = result.validation.solution_type.max_abs_delta
        st.attrs["passed"] = result.validation.solution_type.passed

        mock_grp = handle.create_group("mock_catalog")
        stypes = [r.solution_type.value for r in result.records]
        mock_grp.create_dataset("solution_type", data=np.array(stypes, dtype="S32"))
        accepted = np.array([r.accepted_orbital for r in result.records], dtype=bool)
        mock_grp.create_dataset("accepted_orbital", data=accepted)


def run_selection_function_astrometric(
    config: PipelineConfig,
    artifact_path: Path,
    *,
    data_acquisition_artifact: Path | None = None,
) -> SelectionFunctionAstrometricResult:
    """Execute the astrometric selection-function stage (DR3 only in v1).

    Parameters
    ----------
    config
        Validated pipeline configuration.
    artifact_path
        Destination HDF5 path from ``run_management.stage_artifact_path``.
    data_acquisition_artifact
        Optional ``data_acquisition`` HDF5 with real DR3 NSS panels. When omitted, the
        bundled El-Badry reference under ``tests/fixtures/`` or ``validation_gate.reference_path``
        is used.

    Returns
    -------
    SelectionFunctionAstrometricResult
        In-memory summary including validation gate outcome.

    Raises
    ------
    ValueError
        DR4 active mode, gaiamock version mismatch, or validation gate failure when
        ``validation_gate.strict`` is added in a future revision (currently records pass/fail
        in artifact attrs only).
    FileNotFoundError
        Missing reference panels when no data_acquisition artifact is supplied.
    """
    require_dr3_active_for_v1(config)
    versions = verify_gaiamock_versions(config)
    gaiamock = import_gaiamock_mod(verify=True)

    if data_acquisition_artifact is not None and data_acquisition_artifact.is_file():
        real_panels, real_st = load_real_panels_from_data_acquisition(
            data_acquisition_artifact
        )
    else:
        real_panels, real_st = load_reference_panels(config)

    mock_records, g_mag = run_mock_injections(config, gaiamock)
    validation = run_validation_gate(
        config,
        mock_records,
        real_panels=real_panels,
        real_solution_fractions=real_st,
        g_mag=g_mag,
    )

    result = SelectionFunctionAstrometricResult(
        gaiamock_versions=versions,
        records=mock_records,
        validation=validation,
        data_release=config.active_dr_mode.value,
    )
    write_selection_function_artifact(artifact_path, result)
    return result


def format_validation_gate_report(result: SelectionFunctionAstrometricResult) -> str:
    """Fully legible validation summary (exempt from caveman compression)."""
    lines = [
        "=== selection_function_astrometric validation gate ===",
        f"data_release: {result.data_release}",
        f"gaiamock: {result.gaiamock_versions.gaiamock_mod_release} "
        f"sha256={result.gaiamock_versions.gaiamock_mod_sha256[:12]}… "
        f"commit={result.gaiamock_versions.gaiamock_git_commit[:7]}",
        f"detection_fraction: {result.validation.detection_fraction:.4f} "
        f"({sum(r.accepted_orbital for r in result.records)}/{len(result.records)})",
        "six_panel_ks:",
    ]
    for name in SIX_PANEL_NAMES:
        pval = result.validation.six_panel.ks_pvalues.get(name, float("nan"))
        ok = result.validation.six_panel.passed.get(name, False)
        lines.append(f"  {name}: p={pval:.4g} passed={ok}")
    lines.append("solution_type_fractions (mock vs real):")
    for label in SOLUTION_TYPE_LABELS:
        m = result.validation.solution_type.mock_fractions.get(label, 0.0)
        r = result.validation.solution_type.real_fractions.get(label, 0.0)
        lines.append(f"  {label}: mock={m:.4f} real={r:.4f}")
    lines.append(
        f"solution_type max_abs_delta={result.validation.solution_type.max_abs_delta:.4f} "
        f"passed={result.validation.solution_type.passed}"
    )
    lines.append(f"overall_passed: {result.validation.passed}")
    lines.append("=== end validation gate ===")
    return "\n".join(lines)
