"""R5-T11 (#644): the rung-5 binned observable-space Poisson fit for the WD / NS / BH mass functions (RUNG5_SPEC §2–§3, §2.6).

Cells b are (M2_app bin × photocentre-size A bin × distance group). The expected count per cell is

    μ_b = Σ_g a_g L_gb  +  N_s s_A^[b ∈ high-A] T_b  +  Σ_j h^WD_j Σ_k π_kj(r, f_He) R^WD_kjb  +  Σ_j h^dark_j R^dark_jb

* L_gb: luminous expected counts per distance group g at the rung-3 v5 θ; a_g free amplitudes (the v0 stand-in for a
  free luminous θ, declared in every output).
* T_b: the §12.16 π̂ spurious template (shape, Σ_b T_b = 1); N_s the free spurious amplitude; s_A the free multiplier on
  the high-A cells (§2.6). The rung-3 N_s is never imported.
* R^WD_kjb: expected detections in cell b per unit WD rate (per looks-single primary) in true-mass bin j for §12.39
  branch k ∈ (non-interacting, post-MT He, post-MT CO); π_kj the branch fractions in bin j from the unit-amplitude
  branch rates I_kj and the nuisances r = ln A_ni − ln R_mt and f_He. With free WD heights only the ratio A_ni / R_mt is
  identifiable, so it is the fitted parameter (§2.6's ln A_ni, ln R_mt enter through it).
* R^dark_jb: the same for dark companions (one population; NS / BH split by M_TOV is tier 2, downstream).

The likelihood is the SAY effective Poisson likelihood with the Monte-Carlo variance of every template (§3).
Heights are log10 rates per looks-single primary with uniform priors; the nuisance priors are §2.6's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.special import gammaln

FloatArray = NDArray[np.float64]
WD_BRANCHES = ("ni", "mt_he", "mt_co")
NUISANCES = ("ln_ns", "ln_sa", "ln_r_ni_mt", "f_he")
LIMITATIONS = (
    "luminous_template_amplitudes: the luminous component is the rung-3 v5 θ template with free per-distance-group "
    "amplitudes, not a free luminous θ (stand-in until the joint_refit adapter, RUNG5_SPEC §2.2)",
    "wd_per_object_likelihood_coverage: WD gate 4b fails at 95 % with the adopted ν (#643); WD numbers are binned "
    "observable-space results and are not quoted as a WD mass function until #643 is resolved",
    "mc_variance_at_current_theta: the template MC variance is evaluated at the current θ, not fixed relative to a "
    "reference θ (§12.39 F2 amendment); check the ESS of the fit cells",
)


@dataclass
class Rung5Inputs:
    """Everything the fit needs, per cell b (see the module docstring)."""

    k_obs: FloatArray                      # (B,)
    high_a: NDArray[np.bool_]              # (B,)
    lum: FloatArray                        # (G, B)
    lum_var: FloatArray                    # (G, B)
    spur: FloatArray                       # (B,), Σ = 1
    spur_var: FloatArray                   # (B,) relative to the unit-sum template
    wd: FloatArray                         # (K, J_wd, B)
    wd_var: FloatArray                     # (K, J_wd, B)
    wd_rate: FloatArray                    # (K, J_wd) unit-amplitude branch rates per primary
    dark: FloatArray                       # (J_dark, B)
    dark_var: FloatArray                   # (J_dark, B)
    wd_edges: FloatArray
    dark_edges: FloatArray
    pihat_sum: float                       # Σ π̂ (the spurious prior scale)
    ln_r_center: float                     # rung-3 v5 ln A_ni − ln R_mt (prior centre only)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_dark(self) -> int:
        return int(self.dark.shape[0])

    @property
    def n_wd(self) -> int:
        return int(self.wd.shape[1])

    @property
    def n_lum(self) -> int:
        return int(self.lum.shape[0])


@dataclass(frozen=True)
class Priors:
    """§2.6 priors (config `rung5_fit` block; defaults are the spec's)."""

    log10_h_dark: tuple[float, float] = (-9.0, -1.0)
    log10_h_wd: tuple[float, float] = (-7.0, 0.0)
    ln_a_lum: tuple[float, float] = (-1.0, 1.0)
    ns_over_pihat: tuple[float, float] = (0.25, 4.0)
    s_a: tuple[float, float] = (0.1, 10.0)
    ln_r_sigma: float = 1.5 * math.sqrt(2.0)       # difference of two σ = 1.5 normals
    f_he: tuple[float, float] = (0.0, 1.0)


def parameter_names(inp: Rung5Inputs) -> list[str]:
    return ([f"log10_h_dark_{j}" for j in range(inp.n_dark)] + [f"log10_h_wd_{j}" for j in range(inp.n_wd)]
            + [f"ln_a_lum_{g}" for g in range(inp.n_lum)] + list(NUISANCES))


def _slices(inp: Rung5Inputs) -> dict[str, slice]:
    d, w, g = inp.n_dark, inp.n_wd, inp.n_lum
    return {"dark": slice(0, d), "wd": slice(d, d + w), "lum": slice(d + w, d + w + g),
            "ln_ns": slice(d + w + g, d + w + g + 1), "ln_sa": slice(d + w + g + 1, d + w + g + 2),
            "ln_r": slice(d + w + g + 2, d + w + g + 3), "f_he": slice(d + w + g + 3, d + w + g + 4)}


def wd_branch_fractions(rate: FloatArray, ln_r: float, f_he: float) -> FloatArray:
    """π_kj ∝ (e^r I_ni, f_He I_He, (1 − f_He) I_CO) per bin j, normalised over k (equal split where all vanish)."""
    amp = np.array([math.exp(ln_r), f_he, 1.0 - f_he])[:, None]
    x = amp * np.asarray(rate, float)
    s = x.sum(axis=0, keepdims=True)
    return np.where(s > 0, x / np.where(s > 0, s, 1.0), 1.0 / x.shape[0])


def expected_counts(theta: ArrayLike, inp: Rung5Inputs) -> tuple[FloatArray, FloatArray]:
    """(μ_b, MC variance_b) at θ."""
    t = np.asarray(theta, float)
    s = _slices(inp)
    h_dark, h_wd = 10.0 ** t[s["dark"]], 10.0 ** t[s["wd"]]
    a = np.exp(t[s["lum"]])
    ns, sa = math.exp(float(t[s["ln_ns"]][0])), math.exp(float(t[s["ln_sa"]][0]))
    pi = wd_branch_fractions(inp.wd_rate, float(t[s["ln_r"]][0]), float(np.clip(t[s["f_he"]][0], 0.0, 1.0)))
    spur_scale = ns * np.where(inp.high_a, sa, 1.0)
    mu = (a @ inp.lum + spur_scale * inp.spur + np.einsum("j,kj,kjb->b", h_wd, pi, inp.wd) + h_dark @ inp.dark)
    var = ((a**2) @ inp.lum_var + spur_scale**2 * inp.spur_var + np.einsum("j,kj,kjb->b", h_wd**2, pi**2, inp.wd_var)
           + (h_dark**2) @ inp.dark_var)
    return mu, var


def say_loglike(k: FloatArray, mu: FloatArray, var: FloatArray) -> float:
    """SAY effective log-likelihood summed over cells (Poisson where the MC variance vanishes)."""
    mu = np.maximum(mu, 1e-300)
    out = k * np.log(mu) - mu - gammaln(k + 1)
    use = var > 1e-12 * mu**2
    if np.any(use):
        m, v, kk = mu[use], var[use], k[use]
        a_ = m * m / v + 1.0
        b_ = m / v
        out[use] = a_ * np.log(b_) + gammaln(kk + a_) - gammaln(kk + 1) - (kk + a_) * np.log1p(b_) - gammaln(a_)
    return float(np.sum(out))


def bounds(inp: Rung5Inputs, pr: Priors) -> FloatArray:
    """(P, 2) parameter bounds (the ln_r bound is ±4σ around its centre)."""
    b = ([pr.log10_h_dark] * inp.n_dark + [pr.log10_h_wd] * inp.n_wd + [pr.ln_a_lum] * inp.n_lum
         + [(math.log(pr.ns_over_pihat[0] * inp.pihat_sum), math.log(pr.ns_over_pihat[1] * inp.pihat_sum)),
            (math.log(pr.s_a[0]), math.log(pr.s_a[1])),
            (inp.ln_r_center - 4 * pr.ln_r_sigma, inp.ln_r_center + 4 * pr.ln_r_sigma), pr.f_he])
    return np.asarray(b, float)


def log_prior(theta: ArrayLike, inp: Rung5Inputs, pr: Priors) -> float:
    """Uniform within the bounds (log-uniform for N_s and s_A, uniform in log10 h), normal on ln_r."""
    t = np.asarray(theta, float)
    bb = bounds(inp, pr)
    if np.any(t < bb[:, 0]) or np.any(t > bb[:, 1]):
        return -math.inf
    r = float(t[_slices(inp)["ln_r"]][0])
    return -0.5 * ((r - inp.ln_r_center) / pr.ln_r_sigma) ** 2


def log_posterior(theta: ArrayLike, inp: Rung5Inputs, pr: Priors) -> float:
    lp = log_prior(theta, inp, pr)
    if not math.isfinite(lp):
        return -math.inf
    mu, var = expected_counts(theta, inp)
    return lp + say_loglike(inp.k_obs, mu, var)


def start_point(inp: Rung5Inputs, pr: Priors) -> FloatArray:
    bb = bounds(inp, pr)
    t = bb.mean(axis=1)
    s = _slices(inp)
    t[s["lum"]] = 0.0
    t[s["ln_ns"]] = math.log(inp.pihat_sum)
    t[s["ln_sa"]] = 0.0
    t[s["ln_r"]] = inp.ln_r_center
    t[s["f_he"]] = 0.5
    t[s["dark"]] = bb[s["dark"], 0] + 0.25 * (bb[s["dark"], 1] - bb[s["dark"], 0])
    return t


def map_fit(inp: Rung5Inputs, pr: Priors, *, x0: ArrayLike | None = None, fixed: Mapping[int, float] | None = None,
            n_starts: int = 4, seed: int = 0, polish_maxiter: int = 4000) -> tuple[FloatArray, float]:
    """MAP by bounded L-BFGS from ``n_starts`` jittered starts; ``fixed`` holds parameters at given values."""
    bb = bounds(inp, pr)
    fixed = dict(fixed or {})
    free = np.array([i for i in range(bb.shape[0]) if i not in fixed])
    base = np.asarray(start_point(inp, pr) if x0 is None else x0, float).copy()
    for i, v in fixed.items():
        base[i] = v
    rng = np.random.default_rng(seed)

    def full(x: FloatArray) -> FloatArray:
        t = base.copy()
        t[free] = x
        return t

    def nlp(x: FloatArray) -> float:
        v = log_posterior(full(x), inp, pr)
        return 1e300 if not math.isfinite(v) else -v

    best_x, best_f = None, math.inf
    for k in range(max(1, n_starts)):
        x = base[free].copy()
        if k:
            span = bb[free, 1] - bb[free, 0]
            x = np.clip(x + rng.normal(0.0, 0.05, x.size) * span, bb[free, 0], bb[free, 1])
        r = optimize.minimize(nlp, x, method="L-BFGS-B", bounds=[tuple(b) for b in bb[free]])
        r = optimize.minimize(nlp, r.x, method="Nelder-Mead",  # polish: L-BFGS stalls on the flat nuisance directions
                              options={"maxiter": polish_maxiter, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True})
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
    assert best_x is not None
    return full(best_x), -best_f


def run_dynesty(inp: Rung5Inputs, pr: Priors, *, nlive: int, seed: int, dlogz: float = 0.5,
                maxcall: int | None = None) -> dict[str, Any]:
    """Static nested sampling over the full parameter vector; returns equal-weight samples and log-evidence."""
    import dynesty
    from scipy import stats

    bb = bounds(inp, pr)
    ir = _slices(inp)["ln_r"].start
    width = bb[:, 1] - bb[:, 0]

    def prior_transform(u: FloatArray) -> FloatArray:
        x = bb[:, 0] + u * width
        lo_c, hi_c = stats.norm.cdf(bb[ir], loc=inp.ln_r_center, scale=pr.ln_r_sigma)
        x[ir] = stats.norm.ppf(lo_c + u[ir] * (hi_c - lo_c), loc=inp.ln_r_center, scale=pr.ln_r_sigma)
        return x

    def loglike(t: FloatArray) -> float:
        mu, var = expected_counts(t, inp)
        return say_loglike(inp.k_obs, mu, var)

    sampler = dynesty.NestedSampler(loglike, prior_transform, bb.shape[0], nlive=nlive, rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=False)
    res = sampler.results
    w = np.exp(res.logwt - res.logz[-1])
    eq = dynesty.utils.resample_equal(res.samples, w / w.sum(), rstate=np.random.default_rng(seed + 1))
    return {"samples": eq, "logz": float(res.logz[-1]), "logz_err": float(res.logzerr[-1]), "ncall": int(np.sum(res.ncall))}


def correlation_matrix(samples: FloatArray, names: Sequence[str], keep: Sequence[str]) -> dict[str, Any]:
    idx = [list(names).index(k) for k in keep]
    c = np.corrcoef(np.asarray(samples)[:, idx], rowvar=False)
    return {"names": list(keep), "matrix": np.round(c, 4).tolist()}


def nuisance_shift_table(inp: Rung5Inputs, pr: Priors, theta_map: FloatArray, samples: FloatArray,
                         names: Sequence[str]) -> dict[str, Any]:
    """Dark log10 heights at the MAP re-optimised with each nuisance held at its posterior mean ± 1σ (§2.6)."""
    out: dict[str, Any] = {}
    s = _slices(inp)
    for nm in NUISANCES:
        i = list(names).index(nm)
        m, sd = float(np.mean(samples[:, i])), float(np.std(samples[:, i]))
        row = {}
        for tag, v in (("minus", m - sd), ("plus", m + sd)):
            th, _ = map_fit(inp, pr, x0=theta_map, fixed={i: v}, n_starts=1)
            row[tag] = {"value": v, "log10_h_dark": np.round(th[s["dark"]], 4).tolist()}
        out[nm] = row
    return out


def small_n_overlap(inp: Rung5Inputs, pr: Priors, samples: FloatArray, theta_map: FloatArray, *,
                    expected_min: float, threshold: float, collapse_floor: float) -> list[dict[str, Any]]:
    """Posterior-vs-prior overlap (``inference.posterior_vs_prior_overlap``) for every dark / WD height whose expected
    detected count at the MAP is below ``expected_min``."""
    from darkhunter_pop.inference import posterior_vs_prior_overlap

    s = _slices(inp)
    bb = bounds(inp, pr)
    t = np.asarray(theta_map, float)
    rows = []
    for comp, resp in (("dark", inp.dark), ("wd", None)):
        sl = s[comp]
        for j, i in enumerate(range(sl.start, sl.stop)):
            if comp == "dark":
                n_exp = float(10.0 ** t[i] * resp[j].sum())
            else:
                pi = wd_branch_fractions(inp.wd_rate, float(t[s["ln_r"]][0]), float(t[s["f_he"]][0]))
                n_exp = float(10.0 ** t[i] * np.einsum("k,kb->", pi[:, j], inp.wd[:, j, :]))
            if n_exp >= expected_min:
                continue
            ov = posterior_vs_prior_overlap(samples[:, [i]], prior_low=bb[[i], 0], prior_high=bb[[i], 1],
                                            threshold=threshold, collapse_floor=collapse_floor)
            rows.append({"component": comp, "bin": j, "expected_detected": n_exp,
                         "log10_h_q95": float(np.percentile(samples[:, i], 95)), "overlap": ov})
    return rows


def simulate_counts(theta: ArrayLike, inp: Rung5Inputs, rng: np.random.Generator) -> FloatArray:
    """Gate-5b closed loop: Poisson counts per cell at θ (the held-out templates carry the injected universe)."""
    mu, _ = expected_counts(theta, inp)
    return rng.poisson(np.maximum(mu, 0.0)).astype(float)


def laplace_sigma(inp: Rung5Inputs, pr: Priors, theta: FloatArray, step: float = 1e-3) -> FloatArray:
    """σ per parameter from the numerical Hessian (absolute step) of −log posterior at θ, which must be a converged
    optimum (closed-loop pulls; NaN where the inverse Hessian is not positive)."""
    n = theta.size
    bb = bounds(inp, pr)
    h = np.full(n, step)
    f0 = -log_posterior(theta, inp, pr)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            def f(di: float, dj: float) -> float:
                t = theta.copy()
                t[i] += di
                t[j] += dj
                return -log_posterior(np.clip(t, bb[:, 0], bb[:, 1]), inp, pr)
            if i == j:
                H[i, i] = (f(h[i], 0) - 2 * f0 + f(-h[i], 0)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(h[i], h[j]) - f(h[i], -h[j]) - f(-h[i], h[j]) + f(-h[i], -h[j])) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_ARRAYS = ("k_obs", "high_a", "lum", "lum_var", "spur", "spur_var", "wd", "wd_var", "wd_rate", "dark", "dark_var",
           "wd_edges", "dark_edges")


def write_inputs(path: str | Path, inp: Rung5Inputs) -> None:
    import json

    import h5py

    with h5py.File(path, "w") as h:
        for k in _ARRAYS:
            h.create_dataset(k, data=np.asarray(getattr(inp, k)))
        h.attrs["pihat_sum"] = inp.pihat_sum
        h.attrs["ln_r_center"] = inp.ln_r_center
        h.attrs["meta_json"] = json.dumps(inp.meta, default=str)


def read_inputs(path: str | Path) -> Rung5Inputs:
    import json

    import h5py

    with h5py.File(path, "r") as h:
        arr = {k: h[k][()] for k in _ARRAYS}
        return Rung5Inputs(**arr, pihat_sum=float(h.attrs["pihat_sum"]), ln_r_center=float(h.attrs["ln_r_center"]),
                           meta=json.loads(h.attrs["meta_json"]))

