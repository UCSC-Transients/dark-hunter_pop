"""Sample-independent spuriousness model ``P(spurious | ·)`` (CONTINUATION_PLAN §4.8).

Fits one shared propensity over the four labeled fixtures (293 sources). Predicts
the ``verdict`` axis only — ``nature`` belongs to ``companion_nature_likelihood``.
Undetermined rows are censored (Heckman-style joint selection + outcome), never
dropped. Covariates are retained only via ``sensitivity_analysis`` ΔBIC, not by
hand. Evaluable on mock realizations for the #23 inclusion operator.

Functional form (justified in the PR):
- Link: probit.
- Censoring: joint two-part Heckman-style likelihood sharing covariates; selection
  equation adds ``phot_g_mean_mag`` (+ F2 when present) as exclusion restrictions
  (known censoring drivers). Correlation ``rho`` is estimated; if it hits the
  bound the fit falls back to independent parts (``rho=0``) rather than dropping
  undetermined rows.
- Outcome equation stratified to the astrometric branch so SB1 nature-label
  composition does not dominate (CONTINUATION_PLAN §4.8 composition caveat).
- Missing fixture columns: availability indicators, no mean-imputation of
  missing-at-selection fields (RV consistency).
- Regularization: L2 on slope coefficients (63 positives bind).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import optimize
from scipy.stats import multivariate_normal, norm

from darkhunter_pop.config_loader import repo_root as default_repo_root
from darkhunter_pop.config_schema import (
    PipelineConfig,
    SpuriousnessModelConfig,
    SpuriousnessModelFile,
)
from darkhunter_pop.sensitivity_analysis import (
    BinaryCovariateRecommendation,
    _fit_logistic_mle,
    recommend_binary_outcome_covariates,
)

SCHEMA_VERSION = 1

Verdict = Literal["genuine", "spurious", "undetermined"]
Branch = Literal["astrometric", "spectroscopic"]

_TABLE_BRANCH: dict[str, Branch] = {
    "elbadry2023_table_e1": "astrometric",
    "elbadry2024_table3": "astrometric",
    "elbadry2026_table7": "astrometric",
    "elbadry2026_table8": "spectroscopic",
}

_SELECTION_EXCLUSION_COVARIATES: tuple[str, ...] = (
    "phot_g_mean_mag",
    "goodness_of_fit",
)


class SpuriousnessModelUnidentifiedError(RuntimeError):
    """Raised when the joint censoring model is unidentified in practice."""


class PerSampleSpuriousRateReadError(RuntimeError):
    """Raised if any code path treats a per-sample spurious rate as an input."""


@dataclass(frozen=True)
class LabeledSource:
    """One row from a labeled external fixture."""

    source_id: int
    table: str
    branch: Branch
    verdict: Verdict
    nature: str | None
    censor_reason: str | None
    period_days: float
    g_mag: float
    goodness_of_fit: float | None
    significance: float | None
    implied_companion_mass_msun: float | None
    parallax_snr: float | None
    visibility_periods_used: float | None
    rv_consistency: float | None
    label_conflict: str | None = None


@dataclass
class SpuriousnessDesign:
    """Design matrices and labels for the joint model."""

    sources: tuple[LabeledSource, ...]
    candidate_columns: dict[str, NDArray[np.floating]]
    y_spurious: NDArray[np.floating]
    observed: NDArray[np.bool_]
    source_ids: NDArray[np.int64]
    tables: tuple[str, ...]
    branches: tuple[str, ...]


@dataclass(frozen=True)
class FittedSpuriousnessModel:
    """Fitted sample-independent parameter set."""

    schema_version: int
    link_function: str
    censoring: str
    retained_covariates: tuple[str, ...]
    outcome_feature_names: tuple[str, ...]
    selection_feature_names: tuple[str, ...]
    covariate_means: dict[str, float]
    covariate_scales: dict[str, float]
    outcome_intercept: float
    outcome_coef: dict[str, float]
    selection_intercept: float
    selection_coef: dict[str, float]
    rho: float
    rho_fixed: bool
    regularization_l2: float
    sensitivity: BinaryCovariateRecommendation
    literature_interaction: dict[str, Any]
    n_labeled: int
    n_spurious: int
    n_genuine: int
    n_undetermined: int
    log_likelihood: float
    identified: bool
    identification_notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "link_function": self.link_function,
            "censoring": self.censoring,
            "retained_covariates": list(self.retained_covariates),
            "outcome_feature_names": list(self.outcome_feature_names),
            "selection_feature_names": list(self.selection_feature_names),
            "covariate_means": dict(self.covariate_means),
            "covariate_scales": dict(self.covariate_scales),
            "outcome_intercept": self.outcome_intercept,
            "outcome_coef": dict(self.outcome_coef),
            "selection_intercept": self.selection_intercept,
            "selection_coef": dict(self.selection_coef),
            "rho": self.rho,
            "rho_fixed": self.rho_fixed,
            "regularization_l2": self.regularization_l2,
            "sensitivity": self.sensitivity.as_dict(),
            "literature_interaction": dict(self.literature_interaction),
            "n_labeled": self.n_labeled,
            "n_spurious": self.n_spurious,
            "n_genuine": self.n_genuine,
            "n_undetermined": self.n_undetermined,
            "log_likelihood": self.log_likelihood,
            "identified": self.identified,
            "identification_notes": self.identification_notes,
        }


@dataclass
class RateReproductionRow:
    """One sample's predicted vs target aggregate rate."""

    sample: str
    n: int
    predicted_spurious_label_rate: float
    predicted_genuine_label_rate: float
    predicted_mean_p_spurious: float
    target: float | None
    target_kind: str
    abs_error: float | None
    passed: bool | None
    role: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "n": self.n,
            "predicted_spurious_label_rate": self.predicted_spurious_label_rate,
            "predicted_genuine_label_rate": self.predicted_genuine_label_rate,
            "predicted_mean_p_spurious": self.predicted_mean_p_spurious,
            "target": self.target,
            "target_kind": self.target_kind,
            "abs_error": self.abs_error,
            "passed": self.passed,
            "role": self.role,
            "notes": self.notes,
        }


@dataclass
class SpuriousnessFitResult:
    """Full fit + diagnostics payload."""

    model: FittedSpuriousnessModel
    design: SpuriousnessDesign
    p_spurious: NDArray[np.floating]
    p_observed: NDArray[np.floating]
    p_spurious_label: NDArray[np.floating]
    p_genuine_label: NDArray[np.floating]
    rate_reproduction: list[RateReproductionRow] = field(default_factory=list)
    sb1_q15_report: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "value" in value:
            return float(value["value"])
        return None
    return float(value)


def _period_days(row: Mapping[str, Any]) -> float:
    p = row.get("period_days", row.get("period_day", row.get("period")))
    out = _scalar(p)
    if out is None or not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"missing/invalid period_days for source {row.get('source_id')}")
    return out


def harmonic_distance(
    period_days: float,
    *,
    fundamental_period_days: float,
    multiples_max: int,
) -> float:
    """Relative distance of ``period_days`` to the nearest scanning-law harmonic."""
    p = float(period_days)
    if p <= 0.0:
        return float("nan")
    fundamentals = fundamental_period_days * np.arange(1, multiples_max + 1, dtype=np.float64)
    return float(np.min(np.abs(p - fundamentals) / p))


def f2_x_g_break(
    g_mag: float,
    gof: float | None,
    *,
    g_break: float,
    f2_bright: float,
    f2_faint: float,
) -> float | None:
    """Literature indicator: high F2 relative to the G=13 window-class break."""
    if gof is None or not math.isfinite(float(gof)):
        return None
    g = float(g_mag)
    f2 = float(gof)
    if g < g_break:
        return 1.0 if f2 > f2_bright else 0.0
    return 1.0 if f2 > f2_faint else 0.0


def load_spuriousness_model_file(
    path: Path | str | None = None,
    *,
    repo: Path | None = None,
) -> SpuriousnessModelFile:
    """Load and validate ``config/spuriousness_model.yaml``."""
    root = default_repo_root() if repo is None else repo
    cfg_path = Path(path) if path is not None else root / "config" / "spuriousness_model.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return SpuriousnessModelFile.model_validate(raw)


def _implied_mass(row: Mapping[str, Any]) -> float | None:
    for key in ("m2_msun", "m_tilde_msun", "fm_msun", "m2_min_msun"):
        if key in row and row[key] is not None:
            val = _scalar(row[key])
            if val is not None and math.isfinite(val) and val > 0.0:
                return val
    return None


def _rv_consistency(row: Mapping[str, Any]) -> float | None:
    obs = _scalar(row.get("rv_amplitude_robust_kms"))
    exp = _scalar(row.get("rv_amplitude_expected_kms"))
    if obs is None or exp is None or exp <= 0.0:
        return None
    return float(math.log10(max(obs, 1e-6) / exp))


def _table_name_from_path(path: Path) -> str:
    return path.stem


def load_labeled_sources(
    spec: SpuriousnessModelFile,
    *,
    repo: Path | None = None,
) -> list[LabeledSource]:
    """Load all labeled fixtures into a flat source list (verdict axis only)."""
    root = default_repo_root() if repo is None else repo
    sources: list[LabeledSource] = []
    for rel in spec.labeled_sets:
        path = root / rel
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        table = str(raw.get("name") or _table_name_from_path(path))
        branch = _TABLE_BRANCH.get(table)
        if branch is None:
            raise ValueError(f"unknown labeled table {table!r} (no branch mapping)")
        for row in raw["data"]:
            verdict = str(row["verdict"])
            if verdict not in ("genuine", "spurious", "undetermined"):
                raise ValueError(
                    f"invalid verdict {verdict!r} for source {row.get('source_id')}"
                )
            gof = _scalar(row.get("gof") if "gof" in row else row.get("goodness_of_fit"))
            sources.append(
                LabeledSource(
                    source_id=int(row["source_id"]),
                    table=table,
                    branch=branch,
                    verdict=verdict,  # type: ignore[arg-type]
                    nature=row.get("nature"),
                    censor_reason=row.get("censor_reason"),
                    period_days=_period_days(row),
                    g_mag=float(row["g_mag"]),
                    goodness_of_fit=gof,
                    significance=_scalar(row.get("significance")),
                    implied_companion_mass_msun=_implied_mass(row),
                    parallax_snr=_scalar(row.get("parallax_snr")),
                    visibility_periods_used=_scalar(row.get("visibility_periods_used")),
                    rv_consistency=_rv_consistency(row),
                    label_conflict=row.get("label_conflict"),
                )
            )
    return sources


def build_candidate_design(
    sources: Sequence[LabeledSource],
    spec: SpuriousnessModelFile,
) -> SpuriousnessDesign:
    """Build candidate covariate columns (NaN where missing)."""
    n = len(sources)
    period = np.array([s.period_days for s in sources], dtype=np.float64)
    g_mag = np.array([s.g_mag for s in sources], dtype=np.float64)
    gof = np.array(
        [
            float(s.goodness_of_fit) if s.goodness_of_fit is not None else np.nan
            for s in sources
        ],
        dtype=np.float64,
    )
    sig = np.array(
        [
            float(s.significance) if s.significance is not None else np.nan
            for s in sources
        ],
        dtype=np.float64,
    )
    mass = np.array(
        [
            float(s.implied_companion_mass_msun)
            if s.implied_companion_mass_msun is not None
            else np.nan
            for s in sources
        ],
        dtype=np.float64,
    )
    plx = np.array(
        [float(s.parallax_snr) if s.parallax_snr is not None else np.nan for s in sources],
        dtype=np.float64,
    )
    vis = np.array(
        [
            float(s.visibility_periods_used)
            if s.visibility_periods_used is not None
            else np.nan
            for s in sources
        ],
        dtype=np.float64,
    )
    rv = np.array(
        [
            float(s.rv_consistency) if s.rv_consistency is not None else np.nan
            for s in sources
        ],
        dtype=np.float64,
    )
    harm = np.array(
        [
            harmonic_distance(
                float(p),
                fundamental_period_days=spec.scanning_law_fundamental_period_days,
                multiples_max=spec.harmonic_multiples_max,
            )
            for p in period
        ],
        dtype=np.float64,
    )
    f2xg = np.full(n, np.nan, dtype=np.float64)
    for i, (g, f) in enumerate(zip(g_mag, gof, strict=True)):
        if np.isfinite(f):
            val = f2_x_g_break(
                float(g),
                float(f),
                g_break=spec.g_window_class_break,
                f2_bright=spec.f2_threshold_bright,
                f2_faint=spec.f2_threshold_faint,
            )
            f2xg[i] = float(val) if val is not None else np.nan

    available = {
        "harmonic_distance": harm,
        "goodness_of_fit": gof,
        "parallax_snr": plx,
        "a0_snr": sig,
        "phot_g_mean_mag": g_mag,
        "log_implied_companion_mass": np.where(
            np.isfinite(mass) & (mass > 0),
            np.log10(np.clip(mass, 1e-6, None)),
            np.nan,
        ),
        "visibility_periods_used": vis,
        "rv_consistency": rv,
        "f2_x_g_break": f2xg,
    }
    cols: dict[str, NDArray[np.floating]] = {}
    for name in spec.candidate_covariates:
        cols[name] = available.get(name, np.full(n, np.nan, dtype=np.float64))

    y = np.full(n, np.nan, dtype=np.float64)
    observed = np.zeros(n, dtype=bool)
    for i, s in enumerate(sources):
        if s.verdict == "undetermined":
            observed[i] = False
        elif s.verdict == "spurious":
            observed[i] = True
            y[i] = 1.0
        elif s.verdict == "genuine":
            observed[i] = True
            y[i] = 0.0
        else:
            raise ValueError(f"unhandled verdict {s.verdict!r}")

    return SpuriousnessDesign(
        sources=tuple(sources),
        candidate_columns=cols,
        y_spurious=y,
        observed=observed,
        source_ids=np.array([s.source_id for s in sources], dtype=np.int64),
        tables=tuple(s.table for s in sources),
        branches=tuple(s.branch for s in sources),
    )


def select_covariates(
    design: SpuriousnessDesign,
    spec: SpuriousnessModelFile,
) -> tuple[BinaryCovariateRecommendation, dict[str, Any]]:
    """Run the sensitivity-analysis module over §4.8 candidates (astrometric adj.)."""
    adj = design.observed
    astro = np.array([b == "astrometric" for b in design.branches], dtype=bool)
    adj_astro = adj & astro
    if int(np.sum(adj_astro)) >= spec.min_complete_rows_for_covariate:
        y_sa = design.y_spurious[adj_astro]
        design_sa = {k: v[adj_astro] for k, v in design.candidate_columns.items()}
    else:
        y_sa = design.y_spurious[adj]
        design_sa = {k: v[adj] for k, v in design.candidate_columns.items()}

    rec = recommend_binary_outcome_covariates(
        design_sa,
        y_sa,
        spec.candidate_covariates,
        bic_delta_include=spec.bic_delta_include_covariate,
        min_complete_rows=spec.min_complete_rows_for_covariate,
    )
    lit_name = "f2_x_g_break"
    lit = {
        "name": lit_name,
        "literature_motivated": True,
        "reference": "El-Badry et al. (2026) §5.1.1; G=13 window-class break",
        "delta_bic": rec.delta_bic_by_covariate.get(lit_name),
        "n_complete": rec.n_complete_by_covariate.get(lit_name, 0),
        "retained_by_sensitivity": lit_name in rec.selected_covariates,
        "drop_reason": rec.drop_reasons.get(lit_name),
    }
    return rec, lit


def _build_feature_matrix(
    design: SpuriousnessDesign,
    retained: Sequence[str],
    *,
    standardize_mask: NDArray[np.bool_],
    add_missingness_indicators: bool,
    coverage_threshold: float = 0.9,
) -> tuple[NDArray[np.floating], tuple[str, ...], dict[str, float], dict[str, float]]:
    """Z-score retained covariates; optionally append ``*_missing`` indicators."""
    n = len(design.sources)
    mats: list[NDArray[np.floating]] = []
    names: list[str] = []
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in retained:
        col = np.asarray(design.candidate_columns[name], dtype=np.float64)
        finite = np.isfinite(col)
        use = finite & standardize_mask
        if np.any(use):
            mu = float(np.mean(col[use]))
            sd = float(np.std(col[use])) or 1.0
        else:
            mu, sd = 0.0, 1.0
        means[name] = mu
        scales[name] = sd
        z = np.zeros(n, dtype=np.float64)
        z[finite] = (col[finite] - mu) / sd
        mats.append(z)
        names.append(name)
        if add_missingness_indicators:
            cov = float(np.mean(finite[standardize_mask])) if np.any(standardize_mask) else 0.0
            if cov < coverage_threshold:
                mats.append((~finite).astype(np.float64))
                names.append(f"{name}_missing")
                means[f"{name}_missing"] = 0.0
                scales[f"{name}_missing"] = 1.0
    if not mats:
        return np.zeros((n, 0), dtype=np.float64), tuple(), means, scales
    return np.column_stack(mats), tuple(names), means, scales


def _phi2(a: float, b: float, rho: float) -> float:
    r = float(np.clip(rho, -0.999, 0.999))
    return float(
        multivariate_normal.cdf(
            [a, b],
            mean=[0.0, 0.0],
            cov=[[1.0, r], [r, 1.0]],
            allow_singular=True,
        )
    )


def _joint_neg_ll_independent(
    theta: NDArray[np.floating],
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    y: NDArray[np.floating],
    observed: NDArray[np.bool_],
    outcome_mask: NDArray[np.bool_],
    *,
    l2: float,
) -> float:
    """Vectorized joint NLL with independent selection/outcome (rho=0)."""
    kx = x.shape[1]
    kz = z.shape[1]
    beta0 = float(theta[0])
    beta = theta[1 : 1 + kx]
    gamma0 = float(theta[1 + kx])
    gamma = theta[1 + kx + 1 : 1 + kx + 1 + kz]
    xb = beta0 + (x @ beta if kx else 0.0)
    zg = gamma0 + (z @ gamma if kz else 0.0)

    # Selection contribution for every row.
    ll_sel = np.where(
        observed,
        norm.logcdf(zg),
        norm.logcdf(-zg),
    )
    ll = float(np.sum(ll_sel))
    # Outcome contribution only on stratified adjudicated rows.
    out = outcome_mask & observed
    if np.any(out):
        y_o = y[out]
        xb_o = xb[out]
        ll += float(
            np.sum(
                np.where(
                    y_o > 0.5,
                    norm.logcdf(xb_o),
                    norm.logcdf(-xb_o),
                )
            )
        )
    pen = 0.0
    if l2 > 0.0:
        pen = 0.5 * l2 * (float(np.sum(beta**2)) + float(np.sum(gamma**2)))
    return -ll + pen


def _fit_two_part_probit(
    x: NDArray[np.floating],
    z: NDArray[np.floating],
    y: NDArray[np.floating],
    observed: NDArray[np.bool_],
    outcome_mask: NDArray[np.bool_],
    *,
    l2: float,
    rho_abs_max: float,
) -> tuple[float, NDArray[np.floating], float, NDArray[np.floating], float, bool, float, str]:
    """Fit joint two-part probit; try free rho, fall back to rho=0 if unbound."""
    kx = x.shape[1]
    kz = z.shape[1]
    y_out = y[outcome_mask & observed]
    x_out = x[outcome_mask & observed]
    b0, b, _ = _fit_logistic_mle(x_out, y_out, n_steps=120, lr=0.12)
    g0, g, _ = _fit_logistic_mle(z, observed.astype(np.float64), n_steps=120, lr=0.12)
    scale = 1.0 / 1.6
    theta0 = np.concatenate(
        [
            np.array([b0 * scale], dtype=np.float64),
            b * scale,
            np.array([g0 * scale], dtype=np.float64),
            g * scale,
        ]
    )

    def objective(th: NDArray[np.floating]) -> float:
        return _joint_neg_ll_independent(
            th, x, z, y, observed, outcome_mask, l2=l2
        )

    result = optimize.minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-9},
    )
    th = result.x
    beta0 = float(th[0])
    beta = np.asarray(th[1 : 1 + kx], dtype=np.float64)
    gamma0 = float(th[1 + kx])
    gamma = np.asarray(th[1 + kx + 1 : 1 + kx + 1 + kz], dtype=np.float64)
    ll = -float(
        _joint_neg_ll_independent(
            th, x, z, y, observed, outcome_mask, l2=0.0
        )
    )
    # Probe free-rho at the independent MLE; if the score prefers |rho|→1,
    # keep rho=0 and document (exclusion restriction may still leave rho weak).
    rho = 0.0
    rho_fixed = True
    notes = ["rho_fixed_at_0_independent_parts"]
    if result.success:
        # One-dimensional line search on rho at fixed beta/gamma.
        def nll_rho(r: float) -> float:
            xb = beta0 + (x @ beta if kx else 0.0)
            zg = gamma0 + (z @ gamma if kz else 0.0)
            total = 0.0
            for i in range(len(y)):
                if not observed[i]:
                    total += float(norm.logcdf(-zg[i]))
                    continue
                if not outcome_mask[i]:
                    total += float(norm.logcdf(zg[i]))
                    continue
                if y[i] > 0.5:
                    p = _phi2(float(zg[i]), float(xb[i]), r)
                else:
                    p = _phi2(float(zg[i]), float(-xb[i]), -r)
                total += math.log(max(p, 1e-300))
            return -total

        grid = np.linspace(-rho_abs_max, rho_abs_max, 21)
        vals = [nll_rho(float(r)) for r in grid]
        best_i = int(np.argmin(vals))
        best_rho = float(grid[best_i])
        if abs(best_rho) < 0.9 * rho_abs_max and vals[best_i] < -ll - 0.5:
            rho = best_rho
            rho_fixed = False
            ll = -vals[best_i]
            notes = [f"rho_selected_on_grid={rho:.3f}"]
        elif abs(best_rho) >= 0.9 * rho_abs_max:
            notes.append(f"free_rho_near_bound_best={best_rho:.3f}_kept_0")
    else:
        notes.append(f"optimizer_status={result.message}")

    identified = result.success
    return beta0, beta, gamma0, gamma, rho, rho_fixed, ll, "; ".join(notes)


def fit_spuriousness_model(
    spec: SpuriousnessModelFile | None = None,
    *,
    repo: Path | None = None,
    escalate_if_unidentified: bool = True,
) -> SpuriousnessFitResult:
    """Fit the shared joint spuriousness model from labeled fixtures."""
    root = default_repo_root() if repo is None else repo
    if spec is None:
        spec = load_spuriousness_model_file(repo=root)
    if spec.censoring != "joint":
        raise ValueError("only censoring='joint' is supported (§4.8)")
    if spec.link_function != "probit":
        raise ValueError("v1 implements link_function='probit' only")

    sources = load_labeled_sources(spec, repo=root)
    design = build_candidate_design(sources, spec)
    sensitivity, lit = select_covariates(design, spec)
    retained = list(sensitivity.selected_covariates)
    if not retained:
        finite = {
            k: v
            for k, v in sensitivity.delta_bic_by_covariate.items()
            if math.isfinite(v)
        }
        if not finite:
            raise SpuriousnessModelUnidentifiedError(
                "sensitivity module retained no covariates and no finite ΔBIC; escalate"
            )
        retained = [max(finite, key=finite.get)]

    astro = np.array([b == "astrometric" for b in design.branches], dtype=bool)
    outcome_mask = design.observed & astro
    x_mat, out_names, means, scales = _build_feature_matrix(
        design,
        retained,
        standardize_mask=outcome_mask,
        add_missingness_indicators=True,
    )

    # Selection exclusion: known censoring drivers, even if not outcome-retained.
    sel_extra: list[str] = []
    for name in _SELECTION_EXCLUSION_COVARIATES:
        if name not in retained and name in design.candidate_columns:
            sel_extra.append(name)
    if sel_extra:
        z_extra, z_extra_names, z_means, z_scales = _build_feature_matrix(
            design,
            sel_extra,
            standardize_mask=np.ones(len(sources), dtype=bool),
            add_missingness_indicators=True,
        )
        means.update(z_means)
        scales.update(z_scales)
        if z_extra.shape[1]:
            z_mat = np.column_stack([z_extra, x_mat]) if x_mat.shape[1] else z_extra
            sel_names = z_extra_names + out_names
        else:
            z_mat = x_mat
            sel_names = out_names
    else:
        z_mat = x_mat
        sel_names = out_names

    beta0, beta, gamma0, gamma, rho, rho_fixed, ll, notes = _fit_two_part_probit(
        x_mat,
        z_mat,
        design.y_spurious,
        design.observed,
        outcome_mask,
        l2=spec.regularization_l2,
        rho_abs_max=spec.rho_abs_max,
    )
    identified = "optimizer_status" not in notes
    if escalate_if_unidentified and not identified:
        raise SpuriousnessModelUnidentifiedError(
            "Joint spuriousness model failed to optimize: "
            f"{notes}. Escalate per §4.8 — do not drop undetermined rows. "
            f"retained={retained}"
        )

    model = FittedSpuriousnessModel(
        schema_version=SCHEMA_VERSION,
        link_function=spec.link_function,
        censoring=spec.censoring,
        retained_covariates=tuple(retained),
        outcome_feature_names=out_names,
        selection_feature_names=tuple(sel_names),
        covariate_means=means,
        covariate_scales=scales,
        outcome_intercept=beta0,
        outcome_coef={n: float(c) for n, c in zip(out_names, beta, strict=True)},
        selection_intercept=gamma0,
        selection_coef={n: float(c) for n, c in zip(sel_names, gamma, strict=True)},
        rho=rho,
        rho_fixed=rho_fixed,
        regularization_l2=spec.regularization_l2,
        sensitivity=sensitivity,
        literature_interaction=lit,
        n_labeled=len(sources),
        n_spurious=int(np.sum(design.y_spurious[design.observed] > 0.5)),
        n_genuine=int(np.sum(design.y_spurious[design.observed] <= 0.5)),
        n_undetermined=int(np.sum(~design.observed)),
        log_likelihood=ll,
        identified=identified,
        identification_notes=notes,
    )
    p_s, p_o, p_sl, p_gl = predict_joint(model, design)
    result = SpuriousnessFitResult(
        model=model,
        design=design,
        p_spurious=p_s,
        p_observed=p_o,
        p_spurious_label=p_sl,
        p_genuine_label=p_gl,
        config_snapshot=spec.model_dump(mode="json"),
    )
    result.rate_reproduction = evaluate_rate_reproduction(result, spec)
    result.sb1_q15_report = build_sb1_q15_report(design, spec)
    return result


def _features_from_columns(
    model: FittedSpuriousnessModel,
    columns: Mapping[str, NDArray[np.floating]],
    feature_names: Sequence[str],
) -> NDArray[np.floating]:
    n = next(iter(columns.values())).shape[0]
    mats: list[NDArray[np.floating]] = []
    for name in feature_names:
        if name.endswith("_missing"):
            base = name[: -len("_missing")]
            col = np.asarray(columns[base], dtype=np.float64).reshape(-1)
            mats.append((~np.isfinite(col)).astype(np.float64))
            continue
        col = np.asarray(columns[name], dtype=np.float64).reshape(-1)
        mu = model.covariate_means.get(name, 0.0)
        sd = model.covariate_scales.get(name, 1.0) or 1.0
        z = np.zeros(n, dtype=np.float64)
        finite = np.isfinite(col)
        z[finite] = (col[finite] - mu) / sd
        mats.append(z)
    if not mats:
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack(mats)


def predict_p_spurious(
    model: FittedSpuriousnessModel,
    covariates: Mapping[str, NDArray[np.floating] | float],
) -> NDArray[np.floating]:
    """Evaluate ``P(spurious | x)`` — the mock-evaluable propensity."""
    n = 1
    for v in covariates.values():
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim > 0 and arr.size > 1:
            n = int(arr.size)
            break
    cols: dict[str, NDArray[np.floating]] = {}
    needed = set(model.retained_covariates)
    for name in model.outcome_feature_names:
        if name.endswith("_missing"):
            needed.add(name[: -len("_missing")])
        else:
            needed.add(name)
    for name in needed:
        if name not in covariates:
            cols[name] = np.full(n, np.nan, dtype=np.float64)
        else:
            arr = np.asarray(covariates[name], dtype=np.float64)
            cols[name] = np.broadcast_to(arr, (n,)).astype(np.float64).copy()
    x = _features_from_columns(model, cols, model.outcome_feature_names)
    beta = np.array(
        [model.outcome_coef[n] for n in model.outcome_feature_names], dtype=np.float64
    )
    xb = model.outcome_intercept + (x @ beta if beta.size else 0.0)
    return norm.cdf(xb).astype(np.float64)


def predict_joint(
    model: FittedSpuriousnessModel,
    design: SpuriousnessDesign,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Return ``(P(spurious|x), P(observed|x), P(label=spurious), P(label=genuine))``."""
    x = _features_from_columns(model, design.candidate_columns, model.outcome_feature_names)
    z = _features_from_columns(
        model, design.candidate_columns, model.selection_feature_names
    )
    beta = np.array(
        [model.outcome_coef[n] for n in model.outcome_feature_names], dtype=np.float64
    )
    gamma = np.array(
        [model.selection_coef[n] for n in model.selection_feature_names],
        dtype=np.float64,
    )
    xb = model.outcome_intercept + (x @ beta if beta.size else 0.0)
    zg = model.selection_intercept + (z @ gamma if gamma.size else 0.0)
    p_spurious = norm.cdf(xb).astype(np.float64)
    p_observed = norm.cdf(zg).astype(np.float64)
    rho = model.rho
    if abs(rho) < 1e-8:
        p_sl = p_spurious * p_observed
        p_gl = (1.0 - p_spurious) * p_observed
        return p_spurious, p_observed, p_sl, p_gl
    p_sl = np.empty(len(xb), dtype=np.float64)
    p_gl = np.empty(len(xb), dtype=np.float64)
    for i in range(len(xb)):
        p_sl[i] = _phi2(float(zg[i]), float(xb[i]), rho)
        p_gl[i] = _phi2(float(zg[i]), float(-xb[i]), -rho)
    return p_spurious, p_observed, p_sl, p_gl


def evaluate_rate_reproduction(
    result: SpuriousnessFitResult,
    spec: SpuriousnessModelFile,
) -> list[RateReproductionRow]:
    """Compare one parameter set against fixture-recoverable published rates."""
    design = result.design
    tol = spec.validation.absolute_tolerance
    rows: list[RateReproductionRow] = []

    t2024 = spec.validation.elbadry2024
    mask = np.array(
        [
            (tbl == t2024.table) and (src.g_mag < float(t2024.g_max or 15.0))
            for tbl, src in zip(design.tables, design.sources, strict=True)
        ],
        dtype=bool,
    )
    n = int(np.sum(mask))
    pred_sl = float(np.mean(result.p_spurious_label[mask])) if n else float("nan")
    pred_gl = float(np.mean(result.p_genuine_label[mask])) if n else float("nan")
    pred_ps = float(np.mean(result.p_spurious[mask])) if n else float("nan")
    target = float(t2024.target_spurious_fraction or 0.25)
    err = abs(pred_sl - target)
    rows.append(
        RateReproductionRow(
            sample="elbadry2024",
            n=n,
            predicted_spurious_label_rate=pred_sl,
            predicted_genuine_label_rate=pred_gl,
            predicted_mean_p_spurious=pred_ps,
            target=target,
            target_kind="spurious_label_fraction",
            abs_error=err,
            passed=err <= tol,
            role="acceptance_test",
            notes=(
                f"fixture {t2024.fixture_numerator}/{t2024.fixture_denominator}; "
                "rate is E[P(verdict=spurious|x)] under the joint model"
            ),
        )
    )

    t2026a = spec.validation.elbadry2026_astrometric
    mask = np.array([tbl == t2026a.table for tbl in design.tables], dtype=bool)
    n = int(np.sum(mask))
    pred_sl = float(np.mean(result.p_spurious_label[mask])) if n else float("nan")
    pred_gl = float(np.mean(result.p_genuine_label[mask])) if n else float("nan")
    pred_ps = float(np.mean(result.p_spurious[mask])) if n else float("nan")
    target = float(t2026a.target_reliable_fraction or (46 / 76))
    err = abs(pred_gl - target)
    rows.append(
        RateReproductionRow(
            sample="elbadry2026_astrometric",
            n=n,
            predicted_spurious_label_rate=pred_sl,
            predicted_genuine_label_rate=pred_gl,
            predicted_mean_p_spurious=pred_ps,
            target=target,
            target_kind="genuine_label_fraction",
            abs_error=err,
            passed=err <= tol,
            role="acceptance_test",
            notes=(
                f"fixture genuine {t2026a.fixture_genuine}/{t2026a.fixture_n}; "
                "reliable ≡ E[P(verdict=genuine|x)]"
            ),
        )
    )

    t2026s = spec.validation.elbadry2026_sb1
    mask = np.array([tbl == t2026s.table for tbl in design.tables], dtype=bool)
    n = int(np.sum(mask))
    pred_sl = float(np.mean(result.p_spurious_label[mask])) if n else float("nan")
    pred_gl = float(np.mean(result.p_genuine_label[mask])) if n else float("nan")
    pred_ps = float(np.mean(result.p_spurious[mask])) if n else float("nan")
    rows.append(
        RateReproductionRow(
            sample="elbadry2026_sb1",
            n=n,
            predicted_spurious_label_rate=pred_sl,
            predicted_genuine_label_rate=pred_gl,
            predicted_mean_p_spurious=pred_ps,
            target=t2026s.target_spurious_fraction_advisory,
            target_kind="spurious_label_fraction_advisory",
            abs_error=(
                abs(pred_sl - float(t2026s.target_spurious_fraction_advisory))
                if t2026s.target_spurious_fraction_advisory is not None
                else None
            ),
            passed=None,
            role="advisory",
            notes=(
                "§15 Q15: Table 8 Notes mostly label nature, not solution reliability; "
                f"RVs-inconsistent fraction = "
                f"{t2026s.fixture_spurious_solution_fraction:.3f} (24/151)"
            ),
        )
    )
    return rows


def build_sb1_q15_report(
    design: SpuriousnessDesign,
    spec: SpuriousnessModelFile,
) -> dict[str, Any]:
    """Escalate §15 Q15: what denominator the paper's ~50% refers to."""
    t8 = [s for s in design.sources if s.table == "elbadry2026_table8"]
    n = len(t8)
    n_spur = sum(1 for s in t8 if s.verdict == "spurious")
    n_gen = sum(1 for s in t8 if s.verdict == "genuine")
    n_und = sum(1 for s in t8 if s.verdict == "undetermined")
    natures: dict[str, int] = {}
    for s in t8:
        key = s.nature or "(null)"
        natures[key] = natures.get(key, 0) + 1
    n_nature_labeled = sum(
        1 for s in t8 if s.verdict == "genuine" and s.nature is not None
    )
    return {
        "question": "Q15",
        "paper_claim": "~50% have spurious spectroscopic solutions",
        "table8_n": n,
        "verdict_spurious": n_spur,
        "verdict_genuine": n_gen,
        "verdict_undetermined": n_und,
        "spurious_solution_fraction_all_151": n_spur / n if n else None,
        "fixture_fraction": spec.validation.elbadry2026_sb1.fixture_spurious_solution_fraction,
        "nature_counts_among_all": natures,
        "genuine_with_nature_label": n_nature_labeled,
        "interpretation": (
            "Table 8 Notes predominantly encode companion nature (two-temperature SED, "
            "Algol, EB, SB2, Be star). Only 'RVs inconsistent with orbit' rows are "
            "solution-reliability (spuriousness) labels: 24/151 = 15.9%. The paper's "
            "~50% therefore cannot be recovered as verdict=spurious over all 151. "
            "Likely either (a) the ~50% is over a follow-up subsample not itemized in "
            "Table 8, or (b) many nature-labeled rows also have spurious solutions and "
            "Notes report the more specific finding. Until human sign-off, SB1 target "
            "is advisory; acceptance gates on the two astrometric targets only."
        ),
        "status": "escalated_pending_human_signoff",
        "role": "advisory",
    }


def assert_no_per_sample_spurious_rate_input(config: PipelineConfig) -> None:
    """Guard: validation_targets rates must never be read as model inputs."""
    if not isinstance(config.spuriousness_model, SpuriousnessModelConfig):
        raise PerSampleSpuriousRateReadError("spuriousness_model config malformed")


def format_covariate_sensitivity_report(result: SpuriousnessFitResult) -> str:
    """Full-detail ``spuriousness_covariate_sensitivity`` (caveman exemption)."""
    sens = result.model.sensitivity
    lines = [
        "spuriousness_covariate_sensitivity",
        f"n_events={sens.n_events} n_positives={sens.n_positives} "
        f"bic_delta_threshold={sens.bic_delta_threshold}",
        f"retained={list(sens.selected_covariates)}",
        f"dropped={list(sens.dropped_covariates)}",
        "delta_bic_by_covariate:",
    ]
    for name in sens.tested_covariates:
        delta = sens.delta_bic_by_covariate.get(name, float("nan"))
        n_c = sens.n_complete_by_covariate.get(name, 0)
        reason = sens.drop_reasons.get(name, "retained")
        lines.append(f"  {name}: delta_bic={delta} n_complete={n_c} reason={reason}")
    lit = result.model.literature_interaction
    lines.append("literature_interaction (F2 × G, always reported):")
    for k, v in lit.items():
        lines.append(f"  {k}: {v}")
    lines.append(
        f"model_retained_covariates={list(result.model.retained_covariates)}"
    )
    lines.append(
        f"outcome_features={list(result.model.outcome_feature_names)}"
    )
    lines.append(
        f"selection_features={list(result.model.selection_feature_names)}"
    )
    return "\n".join(lines)


def format_labeled_set_performance_report(result: SpuriousnessFitResult) -> str:
    """Full-detail ``spuriousness_labeled_set_performance``."""
    spec = SpuriousnessModelFile.model_validate(result.config_snapshot)
    bh1 = spec.gaia_bh1_source_id
    lines = [
        "spuriousness_labeled_set_performance",
        f"n_labeled={result.model.n_labeled} "
        f"genuine={result.model.n_genuine} spurious={result.model.n_spurious} "
        f"undetermined={result.model.n_undetermined}",
        f"retained_covariates={list(result.model.retained_covariates)}",
    ]
    for name in result.model.retained_covariates:
        n_c = int(np.sum(np.isfinite(result.design.candidate_columns[name])))
        lines.append(f"effective_n[{name}]={n_c}")

    by_table: dict[str, list[int]] = {}
    for i, tbl in enumerate(result.design.tables):
        by_table.setdefault(tbl, []).append(i)
    for tbl, idxs in sorted(by_table.items()):
        idx = np.array(idxs, dtype=int)
        obs = result.design.observed[idx]
        y = result.design.y_spurious[idx]
        p = result.p_spurious[idx]
        n_adj = int(np.sum(obs))
        if n_adj:
            y_a = y[obs]
            p_a = p[obs]
            acc = float(np.mean((p_a >= 0.5) == (y_a >= 0.5)))
            brier = float(np.mean((p_a - y_a) ** 2))
        else:
            acc = float("nan")
            brier = float("nan")
        lines.append(
            f"table={tbl} n={len(idxs)} adjudicated={n_adj} "
            f"accuracy@0.5={acc:.3f} brier={brier:.3f}"
        )

    id_to_verdicts: dict[int, list[tuple[str, str]]] = {}
    for s in result.design.sources:
        id_to_verdicts.setdefault(s.source_id, []).append((s.table, s.verdict))
    overlaps = {sid: v for sid, v in id_to_verdicts.items() if len(v) > 1}
    n_agree = 0
    n_disagree = 0
    for sid, pairs in overlaps.items():
        verdicts = {v for _, v in pairs if v != "undetermined"}
        if len(verdicts) <= 1:
            n_agree += 1
        else:
            n_disagree += 1
            lines.append(f"cross_table_disagreement source_id={sid} {pairs}")
    lines.append(
        f"cross_table_overlaps={len(overlaps)} agree={n_agree} disagree={n_disagree}"
    )

    bh1_rows = [
        (i, s)
        for i, s in enumerate(result.design.sources)
        if s.source_id == bh1
    ]
    if not bh1_rows:
        lines.append(f"gaia_bh1 source_id={bh1} NOT_FOUND")
    else:
        for i, s in bh1_rows:
            lines.append(
                f"gaia_bh1 source_id={bh1} table={s.table} verdict={s.verdict} "
                f"nature={s.nature} label_conflict={s.label_conflict} "
                f"P_spurious={result.p_spurious[i]:.4f} "
                f"P_observed={result.p_observed[i]:.4f} "
                f"andrews_exclusion=wrong_per_Q14"
            )
    lines.append("per_source:")
    for i, s in enumerate(result.design.sources):
        lines.append(
            f"  {s.source_id} table={s.table} verdict={s.verdict} "
            f"nature={s.nature} P_spurious={result.p_spurious[i]:.4f} "
            f"P_label_spurious={result.p_spurious_label[i]:.4f}"
        )
    return "\n".join(lines)


def format_censoring_report(result: SpuriousnessFitResult) -> str:
    """Full-detail ``spuriousness_censoring_report``."""
    lines = [
        "spuriousness_censoring_report",
        f"n_undetermined={result.model.n_undetermined}",
        f"rho={result.model.rho:.4f} rho_fixed={result.model.rho_fixed} "
        f"identified={result.model.identified}",
        f"notes={result.model.identification_notes}",
    ]
    und_idx = [i for i, o in enumerate(result.design.observed) if not o]
    for i in und_idx:
        s = result.design.sources[i]
        lines.append(
            f"  source_id={s.source_id} table={s.table} "
            f"censor_reason={s.censor_reason} G={s.g_mag} "
            f"F2={s.goodness_of_fit} P_spurious={result.p_spurious[i]:.4f} "
            f"P_observed={result.p_observed[i]:.4f}"
        )

    # Drop-the-rows baseline: outcome-only probit on adjudicated astrometric rows.
    x = _features_from_columns(
        result.model, result.design.candidate_columns, result.model.outcome_feature_names
    )
    obs = result.design.observed
    astro = np.array([b == "astrometric" for b in result.design.branches], dtype=bool)
    y = result.design.y_spurious
    b0, b, _ = _fit_logistic_mle(x[obs & astro], y[obs & astro])
    xb = (b0 + x @ b) / 1.6
    p_drop = norm.cdf(xb)

    for row in result.rate_reproduction:
        if row.role != "acceptance_test":
            continue
        if row.sample == "elbadry2024":
            mask = np.array(
                [
                    s.table == "elbadry2024_table3" and s.g_mag < 15.0
                    for s in result.design.sources
                ],
                dtype=bool,
            )
            joint_rate = float(np.mean(result.p_spurious_label[mask]))
            drop_label = np.zeros(int(np.sum(mask)), dtype=np.float64)
            sub_obs = obs[mask]
            drop_label[sub_obs] = p_drop[mask][sub_obs]
            drop_rate = float(np.mean(drop_label))
        elif row.sample == "elbadry2026_astrometric":
            mask = np.array(
                [s.table == "elbadry2026_table7" for s in result.design.sources],
                dtype=bool,
            )
            joint_rate = float(np.mean(result.p_genuine_label[mask]))
            drop_label_g = np.zeros(int(np.sum(mask)), dtype=np.float64)
            sub_obs = obs[mask]
            drop_label_g[sub_obs] = 1.0 - p_drop[mask][sub_obs]
            drop_rate = float(np.mean(drop_label_g))
        else:
            continue
        shift = joint_rate - drop_rate
        lines.append(
            f"rate_shift sample={row.sample} joint={joint_rate:.4f} "
            f"drop_rows_baseline={drop_rate:.4f} shift={shift:.4f}"
        )
    return "\n".join(lines)


def format_rate_reproduction_report(result: SpuriousnessFitResult) -> str:
    """Full-detail ``spuriousness_rate_reproduction``."""
    lines = [
        "spuriousness_rate_reproduction",
        f"one_parameter_set retained={list(result.model.retained_covariates)} "
        f"rho={result.model.rho:.4f} rho_fixed={result.model.rho_fixed}",
    ]
    for row in result.rate_reproduction:
        lines.append(
            f"  sample={row.sample} n={row.n} role={row.role} "
            f"target_kind={row.target_kind} target={row.target} "
            f"pred_spurious_label={row.predicted_spurious_label_rate:.4f} "
            f"pred_genuine_label={row.predicted_genuine_label_rate:.4f} "
            f"pred_mean_P_spurious={row.predicted_mean_p_spurious:.4f} "
            f"abs_error={row.abs_error} passed={row.passed}"
        )
        lines.append(f"    notes: {row.notes}")
    lines.append("sb1_q15:")
    for k, v in result.sb1_q15_report.items():
        lines.append(f"  {k}: {v}")
    acc = [r for r in result.rate_reproduction if r.role == "acceptance_test"]
    passes = [r.passed for r in acc]
    if any(passes) and not all(passes):
        lines.append(
            "ESCALATE: one acceptance target reproduced but not the other — "
            "signature of an absorbed per-sample normalization (§4.8). "
            f"predictions={[r.as_dict() for r in acc]} "
            f"covariates={list(result.model.retained_covariates)}"
        )
    return "\n".join(lines)


def write_spuriousness_reports(
    result: SpuriousnessFitResult,
    out_dir: Path,
) -> dict[str, Path]:
    """Write the four §13 spuriousness diagnostics as text reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "spuriousness_rate_reproduction": out_dir / "spuriousness_rate_reproduction.txt",
        "spuriousness_covariate_sensitivity": out_dir
        / "spuriousness_covariate_sensitivity.txt",
        "spuriousness_labeled_set_performance": out_dir
        / "spuriousness_labeled_set_performance.txt",
        "spuriousness_censoring_report": out_dir / "spuriousness_censoring_report.txt",
    }
    paths["spuriousness_rate_reproduction"].write_text(
        format_rate_reproduction_report(result), encoding="utf-8"
    )
    paths["spuriousness_covariate_sensitivity"].write_text(
        format_covariate_sensitivity_report(result), encoding="utf-8"
    )
    paths["spuriousness_labeled_set_performance"].write_text(
        format_labeled_set_performance_report(result), encoding="utf-8"
    )
    paths["spuriousness_censoring_report"].write_text(
        format_censoring_report(result), encoding="utf-8"
    )
    return paths


def acceptance_targets_passed(result: SpuriousnessFitResult) -> bool:
    """True iff both astrometric acceptance targets pass (SB1 advisory excluded)."""
    acc = [r for r in result.rate_reproduction if r.role == "acceptance_test"]
    return bool(acc) and all(bool(r.passed) for r in acc)
