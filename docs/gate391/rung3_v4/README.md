# Rung 3 v4: joint refit on the `colour_v2.1+tr1+e2+st1` draws (#603, spec §12.42)

**Result: the fit is Monte-Carlo limited and not a usable population measurement.** It improves several v3 mismatches: the e
distribution by G, the bright F2 levels, the a0/σ_a0–RUWE edge, the high-ϖ/σ_ϖ blue/mid-colour deficit and the RUWE total.
But N_s runs to 3.6× its new prior (pull +16.7). The spurious term absorbs the distant orbits (0.66 / 1.07 of DR3 at
0.7–1.5 / > 1.5 kpc). The ESS in the cells the new terms target is 1–16.

## Inputs
- Luminous gens 53–57: 1,259,768 draws, `looks_single_v2:…:noise_colour_v2.1+tr1+e2+st1`, cache built by [build_cache_53_57.py](build_cache_53_57.py).
- WD from co_v1 gen 1006 (400,000 draws), R_dark ≡ 0.
- Version guard in `rung3_joint_391.py` (`joint.required_noise_model`).
- §12.16 spurious template recalibrated on gens 53–56 (#601, #604):
  - π̂ uses log10(ϖ/σ_ϖ), coefficients [−2.343, −1.494] ± [0.294, 0.367] (previously log10(a0/σ_a0), [−1.846, −0.591]);
  - Σπ̂ = 12,729 ± 2,006 (previously 17,133 ± 1,790).
- Config `config/population/rung3_joint_v4.yaml`. Chains: [chain.sh](chain.sh), [chain_post.sh](chain_post.sh).
- Routed st1 type reconstructed post hoc: 30/30 exact against `run_cascade`, cascades bit-identical ([gen 57](routed_type_check_gen57.json), [gen 1006](routed_type_check_gen1006.json)).

## Fit ([fit/report.txt](fit/report.txt), [fit/summary.json](fit/summary.json))
- **Convergence.** 4/12 starts within Δ(−lnL) < 5; offsets 0.0, 1.1, 3.1, 3.4, 5.9, 7.6, 7.7, 8.5, 14.1, 44.6, 69.8, 82.3; −lnL 17,218.3.
- The profiles found points up to 2.4 lower (17,215.9), so the global optimum was not reached.
- **At bounds:** f_He, Δln F_twin,sP (+2). κ_cb is at +4.98 (bound 5).
- Hessian errors are undefined for a3, b3, Δln F_twin,sP and κ_cb.

| parameter | value ± error | bounds |
|---|---|---|
| ln A | −0.32 ± 0.14 | [−3, 3] |
| α_lo / α_hi | +0.77 ± 0.35 / −0.77 ± 0.44 | [−3, 5] / [−5, 5] |
| γ_P / ln L_P | −0.55 ± 0.58 / +0.56 ± 1.02 | [−2, 1] / [−1.5, 2.5] |
| dγ_q / ln F_twin | +0.30 ± 0.07 / −0.23 ± 0.35 | [−1.5, 1] / [−4, 1.5] |
| a1 a2 a3 | 15.7 ± 35, 1.39 ± 0.50, 0.92 (nan) | [0.1, 20] |
| b1 b2 b3 | 20.5 ± 33, 2.26 ± 1.14, 1.05 (nan) | [1, 30] |
| ln f_evo | −0.23 ± 0.16 | [−3, 3] |
| ln A_ni (ln_R_wd) | +0.09 ± 0.44 (A_ni = 1.10) | [−5, 5] |
| ln R_mt / f_He | −6.87 ± 0.13 / 0.011 ± 0.019 (bound) | [−14, 0] / [0.01, 0.99] |
| **β_e (M1 e-shape)** | +0.03 ± 0.48 | [−3, 3] |
| **Δln F_twin,sP** | +2.0 (bound) | [−4, 2] |
| **κ_cb** | +4.98 (bound) | [−5, 5] |
| N_s | 46,136 ± 584, prior 12,729 ± 2,006, **pull +16.7** | |

- **Totals.** (mock + WD + spurious)/DR3 is 1.07 for orbits and 0.97 for accelerations.
- **By distance** (< 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc), total/DR3 0.96 / 0.95 / 1.28 / 1.42:
  - luminous 0.81 / 0.63 / 0.49 / 0.35;
  - WD 0.08 / 0.07 / 0.14 / 0.00;
  - spurious 0.08 / 0.26 / 0.66 / 1.07.
- **WD.** 11.1k C1 orbits and 152k accelerations.
- **ESS:**
  - luminous orbits 668, accelerations 606; WD orbits 16, accelerations 116;
  - cells (luminous / WD): d < 0.3: 273 / 5; G < 13: 311 / 6; G < 13 & log P < 2: **2 / 1**;
  - A > 0.5 by distance: 4/4, 12/3, 16/2, 13/0;
  - 7 of 13 big bins are below ESS 30; the median relative MC error of an orbit bin at the reference θ is 0.67.

## Profiles ([fit/profile.json](fit/profile.json); 1 start per point from the fit, same reference θ)
| parameter | grid → −lnL |
|---|---|
| dγ_q | 0: 17,218.9; 0.15: 17,216.2; 0.45: 17,216.5; 0.6: 17,217.7 |
| ln F_twin | −1.5: 17,218.8; −0.9: 17,216.3; 0.4: 17,216.5; 1.0: 17,222.0 |
| Δln F_twin,sP | −4: 17,216.7; −2: 17,216.9; 0: 17,215.9; 1: 17,215.9 |
| N_s | 12,729 (prior): 18,482.4; 20k: 17,850.6; 30k: 17,420.2; 38k: 17,264.7 |

- The twin parameters are flat within Δ ≈ 3 over the grids, so they are unconstrained.
- Holding N_s at its prior costs Δ ≈ 1,265.

## Before/after against v3 (orbits; common v3 edges; [compare/](compare/))
DR3/mock ratio with pull, or mock value against DR3; d ≥ 0.3 kpc where marked.

| flagged mismatch | DR3 | v3 | v4 |
|---|---|---|---|
| N(F2 > 3), G < 13 | 35,964 | 0.62 (−3.3) | 0.88 (−1.9) |
| N(F2 > 8), G < 13 | 9,959 | 0.48 (−1.9) | 0.71 (−2.9) |
| F2 median G < 13 / 90th pct | 4.55 / 11.1 | 4.89 / 10.3 | 5.40 / 11.8 |
| F2 median, M_G0 2–4 | 2.08 | 3.11 | 2.65 |
| 2-D G × F2, χ²/dof (d ≥ 0.3) | | 12.0 ± 2.2 | 9.5 ± 2.9 |
| 2-D M_G0 × F2, χ²/dof (d ≥ 0.3) | | 10.6 ± 2.8 | 6.9 ± 1.4 |
| N(ϖ/σ_ϖ > 238), (BP−RP)₀ < 0.8 / 0.8–1.2 / 1.2–1.8 | 1,647 / 4,718 / 3,432 | 1.60 / 1.94 / 1.58 | 1.39 / 1.22 / 0.89 |
| N(ϖ/σ_ϖ > 238), log P 2–2.5 / 2.5–3 | 6,612 / 5,031 | 1.58 / 1.58 | 1.02 / 1.39 |
| 2-D colour × ϖ/σ_ϖ / P × ϖ/σ_ϖ, χ²/dof (d ≥ 0.3) | | 14.4 / 20.6 | 42.7 ± 19.6 / 132.7 ± 36.7 (worse; low-ESS cells) |
| a0/σ_a0 edge: DR3 fraction above mock p99, log RUWE 0.1–0.2 / 0.4–0.6 / 0.6–1 (expect 0.01) | | 0.034 / 0.042 / 0.040 | 0.008 / 0.012 / 0.024 |
| 2-D a0/σ_a0 × RUWE, χ²/dof (d ≥ 0.3) | | 46.4 ± 7.8 | 34.7 ± 5.9 |
| N(P < 100 d), G < 13: all / d ≥ 0.3 | 677 / 16 | 1.30 / 2.82 | 0.88 / 24.6 (mock 1 ± 0; ESS 2) |
| 2-D G × P, χ²/dof (d ≥ 0.3) | | 18.7 | 108.3 ± 9.8 (worse) |
| N(RUWE > 1.4) (= orbit total) | 125,226 | 0.79 (−4.0) | 0.93 (−2.2) |
| 2-D G × RUWE, χ²/dof (d ≥ 0.3) | | 44.3 ± 5.0 | 22.1 ± 3.6 |
| N(A > 0.5) d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 | 600 / 2,516 / 3,288 / 849 | 0.35 / 1.62 / 1.08 / 0.96 | 0.33 / 1.02 / 0.61 / 0.65 |
| twin/single, 0–0.2 / 0.2–0.5 / 0.5–1 kpc | 0.32 / 0.33 / 0.46 | 0.32 / 0.32 / 0.48 | 0.32 / 0.33 / 0.46 |
| e median G 10–13 / 13–16 / 16–19.5 | 0.32 / 0.31 / 0.31 | 0.39 / 0.32 / 0.23 | 0.30 / 0.29 / 0.24 |
| e median, G < 13 and log P 2.5–3 | 0.315 | 0.396 | 0.318 |
| N(e > 0.5), G 10–13 | 10,931 | 0.58 (−3.1) | 0.94 (−0.5) |

- Acceleration 2-D χ²/dof values are 10³–10⁴ in both v3 and v4, with bootstrap errors of the same size. Mock-empty cells under large DR3 counts dominate them; they measure MC coverage, not model shape.
- The bright e excess is gone at G < 13. The fit reaches this through the free b2 and a1/b1 rather than β_e, which stays at 0 ± 0.5.

## Plots
- [plots/index.html](plots/index.html) and [plots/README.md](plots/README.md): 200 figures.
  - Every 1-D (all, by distance, and orbits by solution type: Orbital vs AstroSpectroSB1) and every 2-D pair, for orbits and accelerations.
  - χ²/dof ranking with 30-fold bootstrap MC errors (2-D included).
- The worst 1-D orbit splits are AstroSpectroSB1 (e 22, RUWE 18, F2 16). The Orbital splits are 2–8.
- Side-by-side with v3: [compare/](compare/). It holds [before_after.md](compare/before_after.md), [noise_metrics.md](compare/noise_metrics.md), [population_metrics.md](compare/population_metrics.md) and [chi2_2d_d_ge_0p3.md](compare/chi2_2d_d_ge_0p3.md), plus 44 figures, among them A by distance, e by G, and P at G < 13, d < 0.3.

## Reading
1. **Spurious takes the distant orbits.** The new π̂ depends on ϖ/σ_ϖ, so the template peaks at large distance. With too few luminous distant/large-A draws, the optimiser sends N_s to 46k. Until the luminous model reaches the distant orbits, N_s is not a measurement.
2. **The new terms are not constrained by these draws.**
   - The log P < 2, G < 13 cell has luminous ESS 2 and WD ESS 1, so Δln F_twin,sP and κ_cb run to their bounds.
   - β_e is consistent with 0: the per-range Beta shapes already fix the e distribution.
3. **WD ESS is 16 (orbits).** Gen 1006 alone does not cover the §12.39 target efficiently, and A > 0.5 at d < 0.3 is still ×3 over-predicted, carried by WD.

## Recommendation on the paused queue steps
- **Gen-52 luminous top-up: run it,** extended. Its A > 0.5 and evolved/distant cells are exactly the starved ones (ESS 2–16), and the distant shortfall is what lets N_s run away. Add a nearby low-mass short-P component (M1 < 1, log P < 2, d < 0.3), since the short-P terms cannot be fitted at ESS 2.
- **Dark CO gen-1004: not now.** R_dark ≡ 0 at rung 3, and the bottleneck is WD ESS. A second WD-focused generation (or a proposal tabulated from the §12.39 target, §12.40 c) would help rung 3; gen 1004 belongs to rung 4/5.
- Before the next refit, re-check the π̂ covariate choice. Near-sample mean π̂ is 0.036 against 0.190 far, so the template extrapolates in ϖ/σ_ϖ. The clamp applies, but the shape is set by two re-injection samples.
