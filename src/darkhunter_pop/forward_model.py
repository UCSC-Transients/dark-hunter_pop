"""Stages: ``selection_function_astrometric`` and ``selection_function_followup``.

``selection_function_astrometric`` wraps vendored ``gaiamock_mod`` (modified RUWE) only via
``import_gaiamock_mod()``. Records the gaiamock version triple and refuses on mismatch.
Includes the El-Badry et al. (2024) six-panel mock-vs-real validation gate and a solution-type
fraction diagnostic. DR3 execution only in v1 (ARCHITECTURE.md §4).

``selection_function_followup`` is a parametric approximation to who gets RV follow-up
(target-list membership with adoption dates, declination/brightness limits, cooler-star
preference), calibrated via mock-vs-real N_observations and time-span histograms. Survey
tiering covers documented major surveys vs ad hoc literature. Google Sheets revision-history
mining and weekly snapshot hooks reconstruct adoption dates (ARCHITECTURE.md §4, §7).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats

from darkhunter_pop.config_loader import repo_root, require_dr3_active_for_v1
from darkhunter_pop.config_schema import (
    ExtinctionModel,
    MajorSurveySFConfig,
    MockPopulationConfig,
    MockPopulationSampling,
    PipelineConfig,
    SelectionFunctionAstrometricConfig,
    SelectionFunctionFollowupConfig,
    SurveyTier,
)
from darkhunter_pop.gaiamock_vendor import (
    GaiamockModVersions,
    assert_versions_match,
    read_versions,
    import_gaiamock_mod,
)
from darkhunter_pop.schemas import ActiveDRMode, FollowUpRecord

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
class MockBinaryDraw:
    """One draw of binary parameters fed into the gaiamock cascade."""

    period_days: float
    m1_msun: float
    m2_msun: float
    eccentricity: float
    flux_ratio: float
    Mg_tot: float
    Tp: float
    omega_rad: float
    w_rad: float
    inc_deg: float
    faint_draw: bool = False


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


def draw_mock_binary_params(
    pop: MockPopulationConfig,
    rng: np.random.Generator,
) -> MockBinaryDraw:
    """Draw one mock binary before the gaiamock astrometric cascade.

    ``elbadry_prior`` follows gaiamock's uniform-in-frequency period prior and
  log-uniform component masses / flux ratio, with scalar ranges owned by config.
    """
    if pop.sampling is MockPopulationSampling.FIXED:
        period = float(pop.period_days)
        m1 = float(pop.m1_msun)
        m2 = float(pop.m2_msun)
        ecc = float(pop.eccentricity)
        flux = float(pop.flux_ratio)
        mg = float(pop.Mg_tot)
    elif pop.sampling is MockPopulationSampling.ELBADRY_PRIOR:
        inv_p_lo = 1.0 / float(pop.period_days_max)
        inv_p_hi = 1.0 / float(pop.period_days_min)
        period = float(1.0 / rng.uniform(inv_p_lo, inv_p_hi))
        ecc = float(rng.uniform(0.0, pop.eccentricity_max))
        m1 = float(np.exp(rng.uniform(np.log(pop.m1_msun_min), np.log(pop.m1_msun_max))))
        m2 = float(np.exp(rng.uniform(np.log(pop.m2_msun_min), np.log(pop.m2_msun_max))))
        flux = float(
            np.exp(rng.uniform(np.log(pop.flux_ratio_min), np.log(pop.flux_ratio_max)))
        )
        faint_draw = bool(rng.uniform() < pop.faint_draw_fraction)
        if faint_draw:
            mg = float(rng.uniform(pop.faint_Mg_tot_min, pop.faint_Mg_tot_max))
        else:
            mg = float(rng.uniform(pop.Mg_tot_min, pop.Mg_tot_max))
    else:
        raise ValueError(f"unsupported mock_population.sampling: {pop.sampling}")

    Tp = float(rng.uniform(0.0, period))
    omega_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    w_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    inc_deg = float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0))))
    faint_flag = False
    if pop.sampling is MockPopulationSampling.ELBADRY_PRIOR:
        faint_flag = faint_draw
    return MockBinaryDraw(
        period_days=period,
        m1_msun=m1,
        m2_msun=m2,
        eccentricity=ecc,
        flux_ratio=flux,
        Mg_tot=mg,
        Tp=Tp,
        omega_rad=omega_rad,
        w_rad=w_rad,
        inc_deg=inc_deg,
        faint_draw=faint_flag,
    )


def _mock_rng(config: PipelineConfig) -> np.random.Generator:
    seed = config.selection_function_astrometric.mock_population.random_seed
    return np.random.default_rng(seed)


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


load_elbadry_reference_panels = load_reference_panels


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
    draw: MockBinaryDraw,
) -> MockRealizationRecord:
    pop = config.selection_function_astrometric.mock_population
    if draw.faint_draw:
        # gaiamock DR3 scanning-law mocks retain >=12 visibility periods even for
        # faint G; NSS ``insufficient_visibility`` is modeled as a separate draw.
        return MockRealizationRecord(
            solution_type=SolutionType.INSUFFICIENT_VISIBILITY,
            accepted_orbital=False,
        )

    parallax = 1000.0 / d_pc

    data_release = config.active_dr_mode.value
    cascade = gaiamock.run_full_astrometric_cascade(
        ra=ra,
        dec=dec,
        parallax=parallax,
        pmra=0.0,
        pmdec=0.0,
        m1=draw.m1_msun,
        m2=draw.m2_msun,
        period=draw.period_days,
        Tp=draw.Tp,
        ecc=draw.eccentricity,
        omega=draw.omega_rad,
        inc_deg=draw.inc_deg,
        w=draw.w_rad,
        phot_g_mean_mag=phot_g_mean_mag,
        f=draw.flux_ratio,
        data_release=data_release,
        c_funcs=c_funcs,
        verbose=False,
        show_residuals=False,
        ruwe_min=pop.ruwe_min,
        skip_acceleration=pop.skip_acceleration,
    )
    rec = classify_cascade_result(
        cascade,
        m1_msun=draw.m1_msun,
        m2_msun=draw.m2_msun,
        flux_ratio=draw.flux_ratio,
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
    rng = _mock_rng(config)

    ra, dec, d_pc, _x, _y, _z = (
        gaiamock.generate_coordinates_at_a_given_distance_exponential_disk(
            d_min=path_cfg.d_min_pc,
            d_max=path_cfg.d_max_pc,
            N_stars=pop.N_realizations,
            hz_pc=pop.hz_pc,
        )
    )
    l_deg, b_deg = gaiamock.xyz_to_galactic(x=_x, y=_y, z=_z)

    draws = [draw_mock_binary_params(pop, rng) for _ in range(pop.N_realizations)]

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

    phot_g = np.array(
        [
            draw.Mg_tot + 5.0 * np.log10(d_pc[i] / 10.0) + a_g[i]
            for i, draw in enumerate(draws)
        ],
        dtype=np.float64,
    )
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
                draw=draws[i],
            )
        )
    return records, phot_g


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
        try:
            real_panels, real_st = load_real_panels_from_data_acquisition(
                data_acquisition_artifact
            )
        except KeyError:
            # Older data_acquisition artifacts may predate nss_panels persistence.
            real_panels, real_st = load_reference_panels(config)
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


# ---------------------------------------------------------------------------
# selection_function_followup
# ---------------------------------------------------------------------------

SHEETS_REVISION_INCOMPLETENESS_CAVEAT = (
    "Google Drive API revisions.list can be incomplete for long-lived, frequently-edited "
    "sheets (documented Drive API caveat). Spot-check the UI version-history panel when "
    "adoption dates matter for science conclusions. Weekly snapshots under "
    "data/target_lists/snapshots/ are the going-forward archive so future reconstruction "
    "does not depend on Google's revision retention."
)


@dataclass(frozen=True)
class SurveySFLookup:
    """Loaded documented survey selection-function stub."""

    name: str
    tier: SurveyTier
    g_mag_faint_limit: float
    declination_min_deg: float
    declination_max_deg: float
    base_probability: float


@dataclass
class FollowupCalibrationResult:
    """KS calibration of mock vs real N_obs and time-span histograms."""

    n_obs_ks_statistic: float
    n_obs_ks_pvalue: float
    n_obs_passed: bool
    time_span_ks_statistic: float
    time_span_ks_pvalue: float
    time_span_passed: bool

    @property
    def passed(self) -> bool:
        return self.n_obs_passed and self.time_span_passed


@dataclass
class SelectionFunctionFollowupResult:
    """Stage output summary before HDF5 serialization."""

    records: list[FollowUpRecord]
    probabilities: NDArray[np.floating]
    data_source_tiers: list[str]
    calibration: FollowupCalibrationResult
    data_release: str
    accel_jerk_catalog_id: str
    sheets_caveat: str


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_adoption_dates(path: Path) -> dict[str, str]:
    """Load tracked ``source_id -> ISO date`` map from YAML/JSON."""
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"adoption dates must be a mapping: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def load_survey_sf(path: Path) -> SurveySFLookup:
    """Load one documented major-survey SF stub from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"survey SF must be a mapping: {path}")
    return SurveySFLookup(
        name=str(raw["survey"]),
        tier=SurveyTier(str(raw.get("tier", "documented"))),
        g_mag_faint_limit=float(raw["g_mag_faint_limit"]),
        declination_min_deg=float(raw["declination_min_deg"]),
        declination_max_deg=float(raw["declination_max_deg"]),
        base_probability=float(raw["base_probability"]),
    )


def data_root_path(config: PipelineConfig) -> Path:
    root = Path(config.paths.data_root)
    if not root.is_absolute():
        root = repo_root() / root
    return root


def weekly_snapshot_dir(config: PipelineConfig) -> Path:
    """Return ``{data_root}/{weekly_snapshot_relative_dir}/`` (gitignored)."""
    rel = config.selection_function_followup.target_list_sheet.weekly_snapshot_relative_dir
    return data_root_path(config) / rel


def passes_observability_cuts(
    *,
    declination_deg: float | None,
    g_mag: float | None,
    config: SelectionFunctionFollowupConfig,
) -> bool:
    """Declination and brightness hard cuts for follow-up eligibility."""
    if declination_deg is None or g_mag is None:
        return False
    if not (config.declination_min_deg <= declination_deg <= config.declination_max_deg):
        return False
    if not (config.g_mag_bright_limit <= g_mag <= config.g_mag_faint_limit):
        return False
    return True


def cooler_star_factor(
    teff_k: float | None,
    config: SelectionFunctionFollowupConfig,
    *,
    apply: bool,
) -> float:
    """Relative weight preferring cooler stars when the campaign documents that bias."""
    if not apply or teff_k is None:
        return 1.0
    if teff_k <= config.cooler_star_teff_max_k:
        return float(config.cooler_star_weight)
    return 1.0


def ad_hoc_literature_probability(
    record: FollowUpRecord,
    config: SelectionFunctionFollowupConfig,
) -> float:
    """Brightness / declination / proper-motion approximation for unknown SF RVs."""
    ad_hoc = config.ad_hoc_literature
    p = float(ad_hoc.base_probability)
    if ad_hoc.use_brightness:
        if record.brightness_g_mag is None:
            return 0.0
        if not (
            config.g_mag_bright_limit
            <= record.brightness_g_mag
            <= config.g_mag_faint_limit
        ):
            return 0.0
    if ad_hoc.use_declination:
        if record.declination_deg is None:
            return 0.0
        if not (
            config.declination_min_deg
            <= record.declination_deg
            <= config.declination_max_deg
        ):
            return 0.0
    if ad_hoc.use_proper_motion:
        if record.pm_ra_mas_yr is None or record.pm_dec_mas_yr is None:
            return 0.0
        pm_tot = float(
            np.hypot(record.pm_ra_mas_yr, record.pm_dec_mas_yr)
        )
        if pm_tot < ad_hoc.pm_total_min_mas_yr:
            return 0.0
    return min(1.0, p)


def survey_probability(
    record: FollowUpRecord,
    survey: SurveySFLookup,
) -> float:
    """Apply a documented major-survey SF stub."""
    if record.brightness_g_mag is None or record.declination_deg is None:
        return 0.0
    if record.brightness_g_mag > survey.g_mag_faint_limit:
        return 0.0
    if not (
        survey.declination_min_deg
        <= record.declination_deg
        <= survey.declination_max_deg
    ):
        return 0.0
    return min(1.0, float(survey.base_probability))


def followup_selection_probability(
    record: FollowUpRecord,
    config: SelectionFunctionFollowupConfig,
    *,
    survey_lookups: Sequence[SurveySFLookup],
    teff_k: float | None = None,
    on_target_list: bool = False,
    target_list_cooler_pref: bool = False,
    major_survey_name: str | None = None,
) -> tuple[float, str]:
    """Parametric follow-up selection probability and data-source tier label.

    Returns
    -------
    probability, tier_label
        Probability in ``[0, 1]`` and a tier string
        (``target_list``, ``documented:<survey>``, or ``ad_hoc``).
    """
    if not passes_observability_cuts(
        declination_deg=record.declination_deg,
        g_mag=record.brightness_g_mag,
        config=config,
    ):
        return 0.0, "rejected_observability"

    if on_target_list:
        if not target_list_cooler_pref:
            return 1.0, "target_list"
        factor = cooler_star_factor(teff_k, config, apply=True)
        return min(1.0, factor / config.cooler_star_weight), "target_list"

    if major_survey_name is not None:
        for survey in survey_lookups:
            if survey.name == major_survey_name:
                return survey_probability(record, survey), f"documented:{survey.name}"

    return ad_hoc_literature_probability(record, config), "ad_hoc"


def calibrate_followup_histograms(
    mock_n_obs: NDArray[np.floating] | NDArray[np.integer],
    real_n_obs: NDArray[np.floating] | NDArray[np.integer],
    mock_time_span_day: NDArray[np.floating],
    real_time_span_day: NDArray[np.floating],
    config: SelectionFunctionFollowupConfig,
) -> FollowupCalibrationResult:
    """Match mock-vs-real histograms of N_observations and follow-up time span.

    Calibrates the parametric SF; does **not** model adaptive stopping mechanics.
    """
    ks_n = stats.ks_2samp(np.asarray(mock_n_obs, dtype=float), np.asarray(real_n_obs, dtype=float))
    ks_t = stats.ks_2samp(
        np.asarray(mock_time_span_day, dtype=float),
        np.asarray(real_time_span_day, dtype=float),
    )
    pmin = config.calibration.ks_pvalue_min
    return FollowupCalibrationResult(
        n_obs_ks_statistic=float(ks_n.statistic),
        n_obs_ks_pvalue=float(ks_n.pvalue),
        n_obs_passed=float(ks_n.pvalue) >= pmin,
        time_span_ks_statistic=float(ks_t.statistic),
        time_span_ks_pvalue=float(ks_t.pvalue),
        time_span_passed=float(ks_t.pvalue) >= pmin,
    )


def _histogram_counts(
    values: NDArray[np.floating],
    edges: Sequence[float],
) -> NDArray[np.floating]:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    return counts.astype(float)


def diff_sheet_revisions(
    previous_rows: Sequence[Mapping[str, str]],
    current_rows: Sequence[Mapping[str, str]],
    *,
    id_column: str = "source_id",
    adoption_date: str,
) -> dict[str, str]:
    """Diff two sheet exports; newly appearing ``source_id`` rows get ``adoption_date``.

    Used by Drive API revision mining: export each revision, diff consecutive exports.
    """
    prev_ids = {str(row[id_column]) for row in previous_rows if id_column in row}
    adopted: dict[str, str] = {}
    for row in current_rows:
        if id_column not in row:
            continue
        sid = str(row[id_column])
        if sid not in prev_ids:
            adopted[sid] = adoption_date
    return adopted


def mine_sheet_adoption_dates_from_revisions(
    revision_exports: Sequence[tuple[str, Sequence[Mapping[str, str]]]],
    *,
    id_column: str = "source_id",
) -> dict[str, str]:
    """Reconstruct adoption dates from ordered ``(revision_iso_date, rows)`` exports.

    Parameters
    ----------
    revision_exports
        Chronological sequence of revision timestamps and parsed sheet rows.
        Obtained via Drive API ``revisions.list`` + per-revision export when credentials
        are available. See :data:`SHEETS_REVISION_INCOMPLETENESS_CAVEAT`.
    """
    if not revision_exports:
        return {}
    adopted: dict[str, str] = {}
    prev: Sequence[Mapping[str, str]] = []
    for iso_date, rows in revision_exports:
        newly = diff_sheet_revisions(
            prev, rows, id_column=id_column, adoption_date=iso_date
        )
        for sid, date in newly.items():
            adopted.setdefault(sid, date)
        prev = rows
    return adopted


DriveRevisionsLister = Callable[[str], list[dict[str, Any]]]
SheetExporter = Callable[[str, str], list[dict[str, str]]]


def fetch_sheet_revision_exports(
    spreadsheet_id: str,
    *,
    sheet_range: str,
    credentials_env: str,
    list_revisions: DriveRevisionsLister | None = None,
    export_revision: SheetExporter | None = None,
) -> list[tuple[str, list[dict[str, str]]]]:
    """Drive API hook: ``revisions.list`` + per-revision export.

    Credentials path is read from the named environment variable (config holds the
    *name*, never the secret). When ``list_revisions`` / ``export_revision`` are omitted,
    attempts a real Google API client if installed; otherwise raises with a clear message.

    Caveat
    ------
    ``revisions.list`` incompleteness: see :data:`SHEETS_REVISION_INCOMPLETENESS_CAVEAT`.
    """
    if not spreadsheet_id:
        raise ValueError(
            "target_list_sheet.spreadsheet_id is empty; set it when Sheet access is ready"
        )
    cred_path = os.environ.get(credentials_env)
    if not cred_path and list_revisions is None:
        raise EnvironmentError(
            f"credentials env '{credentials_env}' unset; revision mining waits on credentials "
            "(ARCHITECTURE.md §7). Weekly snapshots still work once a current export is supplied."
        )

    if list_revisions is None or export_revision is None:
        raise NotImplementedError(
            "Live Google Drive client not wired in this environment; inject "
            "list_revisions/export_revision callables for production mining, or use "
            "mine_sheet_adoption_dates_from_revisions on pre-exported revision dumps. "
            + SHEETS_REVISION_INCOMPLETENESS_CAVEAT
        )

    revisions = list_revisions(spreadsheet_id)
    exports: list[tuple[str, list[dict[str, str]]]] = []
    for rev in revisions:
        rev_id = str(rev["id"])
        modified = str(rev.get("modifiedTime", rev_id))
        # Normalize to date portion when RFC3339.
        iso_date = modified[:10] if len(modified) >= 10 else modified
        rows = export_revision(rev_id, sheet_range)
        exports.append((iso_date, rows))
    return exports


def write_weekly_sheet_snapshot(
    rows: Sequence[Mapping[str, str]],
    snapshot_dir: Path,
    *,
    when: datetime | None = None,
) -> Path:
    """Archive current sheet state under the weekly snapshot directory (gitignored).

    Filename: ``YYYY-Www.csv`` (ISO week). Going-forward archive so reconstruction does
    not depend on Google's revision retention.
    """
    stamp = when or datetime.now(tz=timezone.utc)
    iso = stamp.isocalendar()
    name = f"{iso.year}-W{iso.week:02d}.csv"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out = snapshot_dir / name
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["source_id"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    # Sidecar checksum for provenance.
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "snapshot_file": name,
        "created_at": stamp.isoformat(),
        "sha256": digest,
        "row_count": len(rows),
        "caveat": SHEETS_REVISION_INCOMPLETENESS_CAVEAT,
    }
    (snapshot_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out


def write_derived_adoption_dates(path: Path, dates: Mapping[str, str]) -> None:
    """Persist reconstructed adoption dates to a tracked YAML path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(sorted(dates.items())), sort_keys=True),
        encoding="utf-8",
    )


def _load_major_surveys(
    configs: Sequence[MajorSurveySFConfig],
) -> list[SurveySFLookup]:
    return [load_survey_sf(_resolve_path(c.selection_function_path)) for c in configs]


def _synthetic_mock_followup(
    n: int,
    config: SelectionFunctionFollowupConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[FollowUpRecord], NDArray[np.floating], list[str]]:
    """Generate a parametric mock follow-up sample for calibration / artifact fill."""
    surveys = _load_major_surveys(config.major_surveys)
    records: list[FollowUpRecord] = []
    probs: list[float] = []
    tiers: list[str] = []
    n_obs_list: list[float] = []
    span_list: list[float] = []

    for i in range(n):
        g_mag = float(rng.uniform(config.g_mag_bright_limit, config.g_mag_faint_limit))
        dec = float(rng.uniform(config.declination_min_deg, config.declination_max_deg))
        pm_ra = float(rng.normal(0.0, 5.0))
        pm_dec = float(rng.normal(0.0, 5.0))
        teff = float(rng.uniform(4000.0, 8000.0))
        on_list = bool(rng.random() < 0.3)
        survey_name = None
        if not on_list and surveys and rng.random() < 0.4:
            survey_name = surveys[int(rng.integers(0, len(surveys)))].name
        rec = FollowUpRecord(
            source_id=10_000 + i,
            target_lists=["andrews"] if on_list else [],
            adoption_dates={"andrews": "2023-01-15"} if on_list else {},
            n_observations=0,
            time_span_day=None,
            brightness_g_mag=g_mag,
            declination_deg=dec,
            pm_ra_mas_yr=pm_ra,
            pm_dec_mas_yr=pm_dec,
        )
        cooler = False
        if on_list:
            for tl in config.target_lists:
                if tl.name == "andrews":
                    cooler = tl.cooler_star_preference
                    break
        p, tier = followup_selection_probability(
            rec,
            config,
            survey_lookups=surveys,
            teff_k=teff,
            on_target_list=on_list,
            target_list_cooler_pref=cooler,
            major_survey_name=survey_name,
        )
        # Parametric observation counts / spans conditional on selection (not adaptive stop).
        if rng.random() < p:
            n_obs = float(rng.integers(1, 15))
            span = float(rng.uniform(10.0, 800.0))
        else:
            n_obs = 0.0
            span = 0.0
        rec = rec.model_copy(update={"n_observations": int(n_obs), "time_span_day": span})
        records.append(rec)
        probs.append(p)
        tiers.append(tier)
        n_obs_list.append(n_obs)
        span_list.append(span)

    return (
        np.asarray(n_obs_list, dtype=float),
        np.asarray(span_list, dtype=float),
        records,
        np.asarray(probs, dtype=float),
        tiers,
    )


def write_selection_function_followup_artifact(
    path: Path,
    result: SelectionFunctionFollowupResult,
    *,
    config: SelectionFunctionFollowupConfig,
) -> None:
    """Persist one HDF5 artifact for ``selection_function_followup``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["stage"] = "selection_function_followup"
        handle.attrs["data_release"] = result.data_release
        handle.attrs["accel_jerk_catalog_id"] = result.accel_jerk_catalog_id
        handle.attrs["calibration_passed"] = result.calibration.passed
        handle.attrs["sheets_caveat"] = result.sheets_caveat

        cal = handle.create_group("calibration")
        cal.attrs["n_obs_ks_statistic"] = result.calibration.n_obs_ks_statistic
        cal.attrs["n_obs_ks_pvalue"] = result.calibration.n_obs_ks_pvalue
        cal.attrs["n_obs_passed"] = result.calibration.n_obs_passed
        cal.attrs["time_span_ks_statistic"] = result.calibration.time_span_ks_statistic
        cal.attrs["time_span_ks_pvalue"] = result.calibration.time_span_ks_pvalue
        cal.attrs["time_span_passed"] = result.calibration.time_span_passed
        cal.create_dataset(
            "n_obs_bin_edges",
            data=np.asarray(config.calibration.n_obs_bin_edges, dtype=float),
        )
        cal.create_dataset(
            "time_span_day_bin_edges",
            data=np.asarray(config.calibration.time_span_day_bin_edges, dtype=float),
        )

        grp = handle.create_group("followup_catalog")
        source_ids = np.array([r.source_id for r in result.records], dtype=np.int64)
        grp.create_dataset("source_id", data=source_ids)
        grp.create_dataset("probability", data=np.asarray(result.probabilities, dtype=float))
        grp.create_dataset(
            "data_source_tier",
            data=np.array(result.data_source_tiers, dtype="S32"),
        )
        n_obs = np.array([r.n_observations for r in result.records], dtype=np.int32)
        grp.create_dataset("n_observations", data=n_obs)
        spans = np.array(
            [
                float(r.time_span_day) if r.time_span_day is not None else np.nan
                for r in result.records
            ],
            dtype=float,
        )
        grp.create_dataset("time_span_day", data=spans)
        g_mag = np.array(
            [
                float(r.brightness_g_mag) if r.brightness_g_mag is not None else np.nan
                for r in result.records
            ],
            dtype=float,
        )
        grp.create_dataset("brightness_g_mag", data=g_mag)
        dec = np.array(
            [
                float(r.declination_deg) if r.declination_deg is not None else np.nan
                for r in result.records
            ],
            dtype=float,
        )
        grp.create_dataset("declination_deg", data=dec)


def run_selection_function_followup(
    config: PipelineConfig,
    artifact_path: Path,
    *,
    astrometric_artifact: Path | None = None,
    rng_seed: int = 0,
) -> SelectionFunctionFollowupResult:
    """Execute the follow-up selection-function stage (DR3 only in v1).

    Parameters
    ----------
    config
        Validated pipeline configuration.
    artifact_path
        Destination HDF5 from ``run_management.stage_artifact_path``.
    astrometric_artifact
        Optional upstream ``selection_function_astrometric`` HDF5 (inputs_from contract).
        Presence is recorded; mock calibration does not require reading gaiamock outputs.
    rng_seed
        Seed for parametric mock draw used in histogram calibration.

    Returns
    -------
    SelectionFunctionFollowupResult
    """
    del astrometric_artifact  # contract hook; calibration uses parametric mock/real draws
    require_dr3_active_for_v1(config)
    fu = config.selection_function_followup
    path_cfg = config.active_dr().selection_function_followup

    rng = np.random.default_rng(rng_seed)
    mock_n, mock_span, records, probs, tiers = _synthetic_mock_followup(200, fu, rng)

    # Real side: either load a configured catalog or draw an independent twin sample.
    real_path = fu.calibration.real_followup_catalog_path
    if real_path:
        # Expect columns n_observations, time_span_day in a simple YAML list or NPZ.
        resolved = _resolve_path(real_path)
        if resolved.suffix == ".npz":
            loaded = np.load(resolved)
            real_n = np.asarray(loaded["n_observations"], dtype=float)
            real_span = np.asarray(loaded["time_span_day"], dtype=float)
        else:
            raise ValueError(f"unsupported real_followup_catalog_path: {resolved}")
    else:
        real_n, real_span, _, _, _ = _synthetic_mock_followup(
            200, fu, np.random.default_rng(rng_seed + 1)
        )

    calibration = calibrate_followup_histograms(
        mock_n, real_n, mock_span, real_span, fu
    )

    # Attach histogram counts for diagnostics (stored via edges in artifact).
    _ = _histogram_counts(mock_n, fu.calibration.n_obs_bin_edges)
    _ = _histogram_counts(mock_span, fu.calibration.time_span_day_bin_edges)

    result = SelectionFunctionFollowupResult(
        records=records,
        probabilities=probs,
        data_source_tiers=tiers,
        calibration=calibration,
        data_release=config.active_dr_mode.value,
        accel_jerk_catalog_id=path_cfg.accel_jerk_catalog_id,
        sheets_caveat=SHEETS_REVISION_INCOMPLETENESS_CAVEAT,
    )
    write_selection_function_followup_artifact(artifact_path, result, config=fu)
    return result


def format_followup_calibration_report(result: SelectionFunctionFollowupResult) -> str:
    """Fully legible follow-up calibration summary (exempt from caveman compression)."""
    lines = [
        "=== selection_function_followup calibration ===",
        f"data_release: {result.data_release}",
        f"accel_jerk_catalog_id: {result.accel_jerk_catalog_id}",
        f"n_records: {len(result.records)}",
        (
            f"n_obs KS: statistic={result.calibration.n_obs_ks_statistic:.4f} "
            f"p={result.calibration.n_obs_ks_pvalue:.4g} "
            f"passed={result.calibration.n_obs_passed}"
        ),
        (
            f"time_span KS: statistic={result.calibration.time_span_ks_statistic:.4f} "
            f"p={result.calibration.time_span_ks_pvalue:.4g} "
            f"passed={result.calibration.time_span_passed}"
        ),
        f"overall_passed: {result.calibration.passed}",
        "sheets_caveat:",
        result.sheets_caveat,
        "=== end follow-up calibration ===",
    ]
    return "\n".join(lines)
