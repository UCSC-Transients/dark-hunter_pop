# #391 rung 3, refit 3: bins chosen by ESS, floor on empty bins (spec §12.12)

Ryan approved steps 1, 3 and 4 on 2026-10-09. He declined step 2, so there are no new priors and the P < 100 d Beta range stays free. The spec is PR #464. No long top-up was run.

- **Script:** `scripts/rung3_refit3_391.py`. `choose` picks the bins from ESS alone; `fit` runs the refit.
- **Config:** `config/population/rung3_refit3_base.yaml` (candidate edges) and `config/population/rung3_refit3.yaml` (the chosen edges plus the choice log, written by `choose` before fitting).
- **Outputs:** `report.txt`, `summary.json`, `ppc_six_panel.png`, `ppc_distance_and_ess.png`.

## 1. Bins (from ESS alone, before any residual)

- **Candidates.** 81 orbit products of the §12.12 edge lists, evaluated with the noW gens 23–28 weights at the refit-2 best fit and at MdS17 (total C1-orbit ESS 267 and 686).
- **No orbit candidate gets every big bin (≥ 1% of real) to ESS_b ≥ 30 at both points.** Not even 4- or 6-bin grids manage it: their smallest big-bin ESS is 10–27.
- **Rule amended in PR #464.** The literal fallback (maximize the real share in bins with ESS ≥ 30) picked a 4-bin grid with no e split, leaving the Beta parameters unconstrained. The amendment restricts eligible candidates to **≥ 2 bins in log P and in e**, an identifiability requirement that involves no residuals.
- **Chosen:**

  | likelihood | edges | bins | ESS_b |
  |---|---|---|---|
  | orbits | G 5–19.5 (one bin) × d [0, 0.7, 6] × log P [0, 2.6, 2.919] × e [0, 0.4, 1] | 8 | 5 of 8 at ≥ 30; 88% of real orbits in bins with ESS ≥ 30; median 50 |
  | accelerations | G [5, 13, 19.5] × d [0, 0.7, 6] | 4 | all ≥ 30 (median 81) |

- **Data bins vs parameters: 12 against 14 free parameters.** With the existing draws, the resolution that reaches ESS ≥ 30 leaves fewer data bins than parameters.

## 3. Empty-coverage bins

- At this binning, no orbit or acceleration bin is empty, so the floor (φ = 1 count, spurious, θ-independent) is never used.
- The 16 empty bins of the 192-bin grid (246 real orbits) are absorbed into the coarse cells.

## 4. Refit (noW baseline; cmdW in `report.txt`)

**Convergence.**
- noW: 6 of 12 starts lie within Δ(−lnL) < 5 of the best, at offsets 0–4.2; the other six lie 23–202 above it. b3 is at its bound.
- cmdW: only 2 of 12 agree; b1 and b3 are at their bounds.

**Parameters.**

| θ | refit 3 noW (spread over near-best starts) | refit 2 | refit 1 |
|---|---|---|---|
| ln A | −0.41 ± 0.37 (0.25) | +0.83 | +0.14 |
| α_lo | −0.09 ± 1.06 (0.93) | +0.38 | +0.71 |
| α_hi | +0.64 ± 0.43 (0.40) | +1.33 | +0.17 |
| γ_P | −1.11 ± 0.81 (1.21) | −2.22 | −0.22 |
| ln L_P | +1.66 ± 0.63 (0.66) | +1.84 | +0.84 |
| Δγ_q | +0.79 ± 0.84 (1.01) | +3.00 (bound) | +2.48 |
| ln F_twin | −1.14 ± 1.98 (2.86) | +0.61 | −1.20 |
| Beta a1, b1 | 4.4, 23.4 (spread 18, 25) | 20, 1 (bounds) | |
| Beta a2, b2 | 1.00, 1.02 (spread 6.3, 10.5) | 1.79, 1.41 | |
| Beta a3, b3 | 0.68, **1.00 (bound)** | 1.11, 2.20 | |
| **N_s** | **26,612 ± 4,886** | 52,207 | |

- **Spurious amplitude:** N_s = 26.6k against the prior 27.5k ± 5.2k, a pull of **−0.2σ**; cmdW is 30.3k (+0.5σ). **The tension is resolved** at this binning: the spurious share is 0.20 of C1 orbits.
- **Totals:** (mock + spurious) / real orbits is **0.96**, and accelerations are **1.01**. The ×1.3–1.8 overshoot of refits 1 and 2 is gone.
- **Posterior-predictive counts vs distance** (fine edges, noW):

  | d (kpc) | (mock + spurious) / real | mock / real | accelerations | orbit:accel, real / fit |
  |---|---|---|---|---|
  | 0–0.3 | **1.13** | 0.92 | 1.03 | 0.475 / 0.521 |
  | 0.3–0.7 | 0.94 | 0.74 | 0.97 | 0.375 / 0.362 |
  | 0.7–1.5 | 0.85 | 0.65 | 1.01 | 0.357 / 0.303 |
  | 1.5–6 | **1.35** | 1.15 | 1.67 | 0.425 / 0.344 |

  The near and far overshoots fall from 2.06 / 4.11 (refit 2) to 1.13 / 1.35, but residual structure in distance remains inside the coarse d bins.
- **e < 0.1:** refit 3 gives **23,256** (mock 18,396) against the real **14,230**. That is better than refit 2 (33,853) and refit 1 (25,433), but **not closed**. The six-panel e shape still peaks at e ≈ 0.05–0.1 where the real distribution peaks at 0.2–0.3. With two e bins, the Beta shapes are not constrained below e = 0.4.
- **Six-panel, other panels:**
  - G: the fit is too high at G < 10 and too low at G 12.5–14.5. G is a single orbit bin, so the fit does not see G. The flat spurious line in the G panel is a plotting simplification; the template actually follows the real G distribution.
  - P and 1/ϖ: reasonable. cos i still shows draw-noise spikes.
- **ESS at the best fit:**
  - Totals: orbits **709** and accelerations **1,239** (refit 2: 267 and 335).
  - Orbit ESS_b: median 119, minimum 3.1. Three of 8 bins are below 30, two of them big.
  - Acceleration ESS_b: median 327, minimum 70.

## Verdict

**Not stable, and the checks are not closed. The counts are close, but the shape checks still fail.**
- **What works:** the spurious amplitude now matches its prior, the totals match (0.96 and 1.01), and the distance overshoots shrink to 1.13 and 1.35.
- **What drives the instability:** **12 data bins against 14 free parameters.**
  - The Beta parameters (all six) and the low-mass and twin parameters are not identified: their spreads across the near-best starts are as large as the values, and their Laplace errors are NaN or larger than the values.
  - Only 6 of 12 starts agree (2 of 12 for cmdW).
  - The e < 0.1 excess (1.6× real) and the G shape cannot be fitted, because the bins needed to see them have ESS_b < 30 with the current draws.
- **Why coarsening can't fix it:** resolution and ESS trade off. Every grid with a G split, or a third log P or e bin, has a big bin at ESS_b ≈ 2–6 at the refit-2 point (see the bin-choice log).
- **What closes it: more draws, not coarser bins.** With step 2 declined, the parameter count stays at 14.
  - The projection at this fit is about **115 CPU-h** for the coarse 8-bin grid, a ×2.3 ESS in its two weak bins.
  - Restoring enough resolution to constrain e and G (≥ 30 bins with ESS_b ≥ 30) is the earlier estimate of **≈ 290–1,000 CPU-h**. It should be targeted at d 0.7–1.5 kpc, log P > 2.6, e 0.3–0.6 and G 12–16, the persistently weakest cells.
  - Re-centring on this fit is not recommended, because its Beta shapes are unconstrained. Re-centre on the robust part only: ln A ≈ −0.4, ln L_P ≈ +1.7, γ_P ≈ −1.1, α_hi ≈ +0.6, N_s at the prior. Keep the e proposal broad.
