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
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(var))):
        raise ValueError(f"expected_counts: non-finite μ ({int(np.sum(~np.isfinite(mu)))} cells) or MC variance "
                         f"({int(np.sum(~np.isfinite(var)))} cells) at θ — check the templates (validate_inputs, #659)")
    return mu, var


# Below this gamma shape the SAY terms are evaluated with gammaln directly (no large cancellation); at or above it the
# Stirling form is used, whose 4-term remainder series is truncated at < 1 / (1188 x⁹) ≈ 1e-12 for x ≥ 10.
_STIRLING_MIN_SHAPE = 10.0


def _stirling_remainder(x: FloatArray) -> FloatArray:
    """ω(x) = lnΓ(x) − [(x − ½) ln x − x + ½ ln 2π], asymptotic series (valid for x ≥ ``_STIRLING_MIN_SHAPE``)."""
    r = 1.0 / (x * x)
    return (1.0 / 12.0 - r * (1.0 / 360.0 - r * (1.0 / 1260.0 - r / 1680.0))) / x


def say_loglike(k: FloatArray, mu: FloatArray, var: FloatArray) -> float:
    """SAY effective log-likelihood summed over cells (Poisson where the MC variance vanishes).

    Per cell, with gamma shape a = μ²/V + 1 and rate b = μ/V,
    ``a ln b + lnΓ(k+a) − lnΓ(k+1) − (k+a) ln(1+b) − lnΓ(a)``. For well-sampled templates a reaches 10⁸–10¹², where the
    literal form cancels terms of size ~a ln a ≈ 10⁹ and leaves roundoff ~10⁻⁶ per evaluation — enough to stall the
    optimiser and to make the finite-difference Laplace Hessian non-positive-definite at random, i.e. platform-
    dependent (#658). For a ≥ ``_STIRLING_MIN_SHAPE`` the same expression is evaluated in the algebraically identical
    cancellation-free form

        −a ln(1 + V/μ) + k ln[(μ² + V)/(μ + V)] + (k + a − ½) ln(1 + k/a) − k + ω(k+a) − ω(a) − lnΓ(k+1),

    using lnΓ(k+a) − lnΓ(a) = k ln a + (k+a−½) ln(1+k/a) − k + ω(k+a) − ω(a) (Stirling with remainder ω), so every
    term is O(k ln μ) or O(μ).
    """
    if not np.all(np.isfinite(var)):  # NaN fails every `var > …` test and would silently mean pure Poisson (#659)
        raise ValueError(f"say_loglike: non-finite MC variance in {int(np.sum(~np.isfinite(var)))} cells")
    mu = np.maximum(mu, 1e-300)
    out = k * np.log(mu) - mu - gammaln(k + 1)
    use = var > 1e-12 * mu**2
    if np.any(use):
        m, v, kk = mu[use], var[use], k[use]
        a_ = m * m / v + 1.0
        b_ = m / v
        lit = a_ < _STIRLING_MIN_SHAPE
        res = np.empty_like(m)
        if np.any(lit):
            al, bl, kl = a_[lit], b_[lit], kk[lit]
            res[lit] = al * np.log(bl) + gammaln(kl + al) - gammaln(kl + 1) - (kl + al) * np.log1p(bl) - gammaln(al)
        st = ~lit
        if np.any(st):
            a, ms, vs, ks = a_[st], m[st], v[st], kk[st]
            res[st] = (-a * np.log1p(vs / ms) + ks * np.log((ms * ms + vs) / (ms + vs))
                       + (ks + a - 0.5) * np.log1p(ks / a) - ks
                       + _stirling_remainder(ks + a) - _stirling_remainder(a) - gammaln(ks + 1))
        out[use] = res
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


@dataclass(frozen=True)
class MapFitResult:
    """MAP fit (#658, #668).

    ``converged`` tests the returned θ (see :func:`map_fit_result`): the Laplace Hessian over the free interior
    parameters is positive definite, the Newton decrement there predicts a remaining gain ``newton_gain`` = ½ gᵀH⁻¹g
    below ``newton_tol`` (ln p units — scale-free, so flat nuisance directions with σ ≫ 1 are not held to an absolute
    gradient), and every free parameter within one Laplace step of a bound has projected gradient below ``grad_tol``.
    ``max_projected_gradient`` is max |g| over the free parameters; ``message`` is scipy's message from the final
    L-BFGS-B polish of the best start (``ABNORMAL`` there only means its line search hit the roundoff floor).
    """

    theta: FloatArray
    log_post: float
    converged: bool
    message: str
    max_projected_gradient: float = math.nan
    newton_gain: float = math.nan
    hessian_positive_definite: bool = False


def projected_gradient(inp: Rung5Inputs, pr: Priors, theta: ArrayLike, *, step: float = 1e-5,
                       active_tol: float = 1e-3, free: ArrayLike | None = None) -> FloatArray:
    """Central-difference gradient (``step``) of −ln p at θ (one-sided, inside the support, near a bound), projected
    onto the feasible set: a component whose descent direction leaves the support at a bound closer than
    ``active_tol`` (default = the Laplace step, so the active set matches :func:`laplace_fit`'s pegged set) is set to 0.
    Only the indices in ``free`` (default all) are differenced; the rest are returned as 0."""
    t0 = np.asarray(theta, float)
    bb = bounds(inp, pr)
    idx = np.arange(t0.size) if free is None else np.asarray(free, int)
    g = np.zeros(t0.size)
    for i in idx:
        hi, lo = min(t0[i] + step, bb[i, 1]), max(t0[i] - step, bb[i, 0])
        tp, tm = t0.copy(), t0.copy()
        tp[i], tm[i] = hi, lo
        g[i] = (log_posterior(tm, inp, pr) - log_posterior(tp, inp, pr)) / (hi - lo)
    at_lo, at_hi = t0 - bb[:, 0] < active_tol, bb[:, 1] - t0 < active_tol
    g[(at_lo & (g > 0)) | (at_hi & (g < 0))] = 0.0
    return g


def map_fit(inp: Rung5Inputs, pr: Priors, *, x0: ArrayLike | None = None, fixed: Mapping[int, float] | None = None,
            n_starts: int = 4, seed: int = 0, polish_maxiter: int = 4000) -> tuple[FloatArray, float]:
    """MAP θ and log posterior; see :func:`map_fit_result` (which also reports convergence)."""
    r = map_fit_result(inp, pr, x0=x0, fixed=fixed, n_starts=n_starts, seed=seed, polish_maxiter=polish_maxiter)
    return r.theta, r.log_post


def map_fit_result(inp: Rung5Inputs, pr: Priors, *, x0: ArrayLike | None = None,
                   fixed: Mapping[int, float] | None = None, n_starts: int = 4, seed: int = 0,
                   polish_maxiter: int = 4000, grad_tol: float = 1e-3, newton_tol: float = 1e-4,
                   lbfgs_maxfun: int = 100_000) -> MapFitResult:
    """MAP from ``n_starts`` jittered starts; ``fixed`` holds parameters at given values.

    Each start runs bounded L-BFGS-B → Nelder–Mead (``polish_maxiter`` iterations, xatol 1e-7, fatol 1e-9; escapes the
    shallow N_s / s_A ridge) → bounded L-BFGS-B with a 3-point (central) finite-difference gradient (``ftol`` 1e-15,
    ``gtol`` = ``grad_tol`` / 10, at most ``lbfgs_maxfun`` evaluations). The final step is what converges at the
    23-parameter DR3 scale: Nelder–Mead alone stalls at ``polish_maxiter`` with |∇ln p| ~ 1e-2 and a non-PD Hessian,
    and restarting it does not help (#668). The best start's θ is then tested as described in :class:`MapFitResult`
    (``newton_tol`` in ln p; ``grad_tol`` in ln p per unit parameter for pegged parameters).
    """
    bb = bounds(inp, pr)
    fixed = dict(fixed or {})
    free = np.array([i for i in range(bb.shape[0]) if i not in fixed])
    base = np.asarray(start_point(inp, pr) if x0 is None else x0, float).copy()
    for i, v in fixed.items():
        base[i] = v
    rng = np.random.default_rng(seed)
    bfree = [tuple(b) for b in bb[free]]

    def full(x: FloatArray) -> FloatArray:
        t = base.copy()
        t[free] = x
        return t

    def nlp(x: FloatArray) -> float:
        v = log_posterior(full(x), inp, pr)
        return 1e300 if not math.isfinite(v) else -v

    best_x, best_f, best_msg = None, math.inf, ""
    for k in range(max(1, n_starts)):
        x = np.clip(base[free].copy(), bb[free, 0], bb[free, 1])
        if k:
            span = bb[free, 1] - bb[free, 0]
            x = np.clip(x + rng.normal(0.0, 0.05, x.size) * span, bb[free, 0], bb[free, 1])
        r = optimize.minimize(nlp, x, method="L-BFGS-B", bounds=bfree)
        r = optimize.minimize(nlp, r.x, method="Nelder-Mead",
                              options={"maxiter": polish_maxiter, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True})
        r = optimize.minimize(nlp, np.clip(r.x, bb[free, 0], bb[free, 1]), method="L-BFGS-B", jac="3-point",
                              bounds=bfree, options={"ftol": 1e-15, "gtol": grad_tol / 10.0, "maxfun": lbfgs_maxfun,
                                                     "maxiter": lbfgs_maxfun})
        if r.fun < best_f:
            best_x, best_f, best_msg = r.x, float(r.fun), str(r.message)
    assert best_x is not None
    theta = full(best_x)
    if free.size == 0:
        return MapFitResult(theta, -best_f, True, best_msg, 0.0, 0.0, True)
    g = projected_gradient(inp, pr, theta, free=free)
    lap = laplace_fit(inp, pr, theta)
    is_free = np.zeros(theta.size, bool)
    is_free[free] = True
    interior = np.flatnonzero(~lap.at_bound)
    keep = is_free[interior]  # free interior parameters, as positions in the interior Hessian
    pd, gain = False, math.inf
    if lap.hessian is not None and np.all(np.isfinite(lap.hessian)):
        Hs = lap.hessian[np.ix_(keep, keep)]
        gi = g[interior[keep]]
        try:
            L = np.linalg.cholesky(Hs) if Hs.size else np.zeros((0, 0))
            pd = True
            z = np.linalg.solve(L, gi) if gi.size else gi
            gain = 0.5 * float(z @ z)
        except np.linalg.LinAlgError:
            pass
    pegged_free = lap.at_bound & is_free
    g_peg = float(np.max(np.abs(g[pegged_free]))) if np.any(pegged_free) else 0.0
    ok = pd and gain < newton_tol and g_peg < grad_tol
    return MapFitResult(theta, -best_f, ok, best_msg, float(np.max(np.abs(g[free]))), gain, pd)


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


@dataclass(frozen=True)
class LaplaceResult:
    """Laplace approximation at a MAP θ (#658).

    ``sigma`` is the σ of each interior parameter conditional on the pegged ones (held at their bound), NaN for a
    pegged parameter, and NaN everywhere when the interior Hessian is not positive definite. ``at_bound`` marks the
    parameters closer than one finite-difference step to a prior bound, where the central stencil would leave the
    prior support and no Gaussian approximation applies. ``positive_definite`` is the Cholesky test of the interior
    Hessian; ``min_eigenvalue`` is its smallest eigenvalue (NaN when it is undefined).
    """

    sigma: FloatArray
    at_bound: NDArray[np.bool_]
    positive_definite: bool
    min_eigenvalue: float
    hessian: FloatArray | None = None  # interior Hessian of −ln p, rows/cols = np.flatnonzero(~at_bound) (#668)

    @property
    def ok(self) -> bool:
        """True when the interior Hessian is positive definite (pegged parameters are reported, not failures)."""
        return self.positive_definite


def laplace_fit(inp: Rung5Inputs, pr: Priors, theta: ArrayLike, step: float = 1e-3) -> LaplaceResult:
    """Numerical Hessian (absolute central-difference step ``step`` in every parameter) of −log posterior at θ, which
    must be a converged optimum.

    A parameter closer than ``step`` to a prior bound is *pegged*: the MAP sits on the bound (a flat or one-sided
    direction, e.g. N_s against s_A, or f_He), and the central stencil would leave the support. The former clipped
    stencil then returned a one-sided difference dominated by the boundary gradient (≈ |∂f/∂θ| / step), so the
    inverse Hessian depended on how far below ``step`` the optimiser happened to stop — platform-dependent NaN σ, or a
    wrong σ for every correlated parameter (#658). Pegged parameters are therefore held fixed: the Hessian is built over
    the interior parameters only (every stencil point lies inside the bounds, no clipping) and ``sigma`` is their σ
    conditional on the pegged values. Positive definiteness is tested by Cholesky and reported in the result.
    """
    t0 = np.asarray(theta, float)
    n = t0.size
    bb = bounds(inp, pr)
    at_bound = np.minimum(t0 - bb[:, 0], bb[:, 1] - t0) < step
    free = np.flatnonzero(~at_bound)
    m = free.size
    sigma = np.full(n, np.nan)
    if m == 0:
        return LaplaceResult(sigma, at_bound, False, math.nan)

    def f(d: FloatArray) -> float:
        t = t0.copy()
        t[free] += d
        return -log_posterior(t, inp, pr)

    f0 = f(np.zeros(m))
    E = np.eye(m) * step
    fp = np.array([f(E[a]) for a in range(m)])
    fm = np.array([f(-E[a]) for a in range(m)])
    H = np.diag((fp - 2.0 * f0 + fm) / step**2)
    for a in range(m):
        for b in range(a + 1, m):
            H[a, b] = H[b, a] = (f(E[a] + E[b]) - f(E[a] - E[b]) - f(-E[a] + E[b]) + f(-E[a] - E[b])) / (4 * step**2)
    if not np.all(np.isfinite(H)):
        return LaplaceResult(sigma, at_bound, False, math.nan, H)
    min_eig = float(np.linalg.eigvalsh(H)[0])
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return LaplaceResult(sigma, at_bound, False, min_eig, H)
    Linv = np.linalg.solve(L, np.eye(m))
    sigma[free] = np.sqrt(np.sum(Linv**2, axis=0))  # diag(H⁻¹) = Σ_k (L⁻¹)_ki², H = L Lᵀ
    return LaplaceResult(sigma, at_bound, True, min_eig, H)


def laplace_sigma(inp: Rung5Inputs, pr: Priors, theta: ArrayLike, step: float = 1e-3) -> FloatArray:
    """σ per parameter from :func:`laplace_fit` (NaN for pegged parameters, and everywhere when the interior Hessian is
    not positive definite; use :func:`laplace_fit` to tell the two apart)."""
    return laplace_fit(inp, pr, theta, step).sigma


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_ARRAYS = ("k_obs", "high_a", "lum", "lum_var", "spur", "spur_var", "wd", "wd_var", "wd_rate", "dark", "dark_var",
           "wd_edges", "dark_edges")


def validate_inputs(inp: Rung5Inputs, source: str = "Rung5Inputs") -> None:
    """Raise ``ValueError`` naming every array that holds a non-finite entry (#659).

    A NaN in any template variance makes the MC variance NaN in every cell, which used to switch ``say_loglike`` to
    pure Poisson without notice; a non-finite template or count makes μ undefined. Neither is ever repaired here.
    """
    bad = {}
    for k in _ARRAYS:
        a = np.asarray(getattr(inp, k))
        if a.dtype.kind in "fc":
            n = int(np.count_nonzero(~np.isfinite(a)))
            if n:
                bad[k] = n
    for k in ("pihat_sum", "ln_r_center"):
        if not math.isfinite(float(getattr(inp, k))):
            bad[k] = 1
    if bad:
        raise ValueError(f"{source}: non-finite entries (array: count) {bad}")


def write_inputs(path: str | Path, inp: Rung5Inputs) -> None:
    """Write the fit inputs to HDF5; refuses inputs with non-finite templates, variances or counts (#659)."""
    import json

    import h5py

    validate_inputs(inp, f"write_inputs({path})")
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
        inp = Rung5Inputs(**arr, pihat_sum=float(h.attrs["pihat_sum"]), ln_r_center=float(h.attrs["ln_r_center"]),
                          meta=json.loads(h.attrs["meta_json"]))
    validate_inputs(inp, f"read_inputs({path})")
    return inp

