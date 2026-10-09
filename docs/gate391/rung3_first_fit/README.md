# #391 rung 3, first fit (spec §12): gens 23–27 reweighted onto the full DR3 sample

Ryan approved this on 2026-10-09. There is no new simulation. Spec: `docs/MOCK_POPULATION_SPEC.md` §12 (PR #458).

- Script: `scripts/rung3_fit_391.py`.
- Bins and bounds: `config/population/rung3_fit.yaml`. These are provisional defaults MP-Q41–Q45 (a), none of them a decision.
- Outputs: `report.txt`, `summary.json`, `ppc_six_panel.png`, `ppc_counts_e_ratio.png`.

## Setup

- **Data.** 131,168 real C1 orbits in 192 bins (G × d × log P × e) and 337,949 real accelerations in 16 bins (G × d).
- **Likelihood.** The SAY effective likelihood (MC noise included), with 11 free parameters.
- **Starts.** 12 starts (L-BFGS-B, then a Powell polish). 11 of 12 reach the same optimum (Δ(−lnL) = 0.0); one stalls at +103.
- **Uncertainties.** These are Laplace errors. They are **understated**; see the ESS section.
- **Weights.** The noW baseline, with cmdW as the sensitivity case.

## Best fit vs MdS17 (published = 0, except α_lo, which replaces MP-Q7's log-linear ≈ 0.5)

| θ | noW (baseline) | cmdW | reading |
|---|---|---|---|
| ln A | +1.54 ± 0.13 | +1.87 ± 0.16 | Large, because it absorbs the unnormalized q tilt (MP-Q42 a). Net counts: mock orbits ×1.51, accelerations ×1.81 vs MdS17 |
| α_lo | +0.89 ± 0.14 | +1.82 ± 0.15 | Frequency falls faster below 0.8 M⊙ than MP-Q7 |
| α_hi | +0.06 ± 0.16 | −1.19 ± 0.18 | MdS17-consistent without W; depends on the Malmquist weight |
| γ_P | −0.23 ± 0.17 | −0.16 ± 0.14 | Consistent with MdS17 |
| ln L_P | +0.58 ± 0.17 | +0.52 ± 0.15 | Long-P (log P > 3) weight ×1.7–1.8, set by the accelerations |
| Δγ_q | +2.40 ± 0.28 | +1.81 ± 0.24 | Mass moves from low q to intermediate and high q… |
| ln F_twin | −1.53 ± 0.23 | −1.46 ± 0.25 | …but twins (q ≥ 0.95) are suppressed ×0.22 |
| **Δη** | **−0.45 ± 0.03** | **−0.46 ± 0.03** | **Less eccentric than MdS17; robust to W** |
| **f_s** | **0.100 ± 0.004** | 0.116 ± 0.005 | Spurious share of all C1 orbits: ≈ 13,100 orbits |
| k_P | +6.4 ± 0.1 | +6.6 ± 0.1 | Spurious orbits sit at the longest P (P ≈ 560–830 d) |
| k_d | −1.60 ± 0.04 | −1.70 ± 0.04 | Spurious orbits *fall* with distance; the expectation was the opposite |

Additional notes:
- **Spurious share in the prior bin** (d 0.7–1.5 kpc, G 12–16): 0.068 (noW) and 0.075 (cmdW), against a prior of 0.21 ± 0.04. That is a 3.5σ tension: the data prefer less spurious contamination there than the symmetric re-detection deficit implied.
- **Fit quality.** −lnL is 2,065 at the best fit, against 505,076 at MdS17 with f_s = 0. The Poisson-style deviance, including the MC variance, is 2,795 over 143 non-empty orbit bins, so the model is not yet an adequate fit.

## Posterior-predictive checks (noW)

- **The distance deficit closes in the middle.**
  - 0.7–1.5 kpc: orbits 0.48 → **0.93**, accelerations 0.49 → **0.98**.
  - 0.3–0.7 kpc: orbits 0.66 → 1.02, accelerations 0.57 → 1.00.
  - It overshoots at the ends: d < 0.3 kpc orbits ×1.33, and d > 1.5 kpc orbits ×2.19 (only 3,059 real; the ESS there is ≈ 0).
  - Against G it is flat at 0.84–1.22.
- **The e excess partly closes.** Mean e (coarse-bin centres), real / MdS17 / fit:
  - 0–0.3 kpc: 0.374 / 0.460 / 0.406;
  - 0.3–0.7 kpc: 0.339 / 0.433 / 0.376;
  - 0.7–1.5 kpc: 0.322 / 0.368 / 0.311.

  So about 60% of the excess goes at d < 0.7 kpc, and it is gone at 0.7–1.5 kpc. The six-panel e shape is still wrong: the fit has too many e < 0.1 and too few at 0.2–0.45, and it is spiky from MC noise.
- **Six-panel.** P, G and 1/ϖ now match the real totals and shapes closely. f_m and cos i carry visible MC spikes (cos i near ±1 and 0).
- **Orbit : acceleration** matches at 0.3–1.5 kpc and G 12–16 (0.385 / 0.337 vs real 0.375 / 0.357). It is too high at d < 0.3 kpc (0.66 vs 0.48) and d > 1.5 kpc.

## ESS: the fit is MC-noise limited

- At the best fit the ESS of the C1 orbits is **426** (704 at MdS17) and that of the accelerations is 877 (1,336). That is above the < 100 "ESS-collapsed" line overall.
- But **191 of 192 orbit bins have ESS_b < 30, and they hold 99% of the real orbits.** The §3.6 / rung-3 acceptance (ESS_b ≥ threshold in every likelihood bin) **fails**.
- Consequences:
  - The Laplace errors understate the true uncertainty.
  - The spikes in the PPC are draw noise.
  - The fit can lean on noisy bins: the d > 1.5 kpc overshoot and the k_d sign are suspects.
- Parameters to treat as **indicative only**: the slopes and q / twin numbers.
- Parameters that are **robust** across W and starts: Δη ≈ −0.45, the long-P weight ≈ ×1.7, f_s ≈ 0.10, and the closing of the 0.3–1.5 kpc deficit.

## Top-up: needed

- Gens 23–27 cost 183 CPU-h. At the best fit that is 2.3 ESS per CPU-h for C1 orbits.
- 33 bins each hold ≥ 1% of the real orbits (69% of them in total), and all 33 are below ESS_b 30.
- The required ESS_b multiplier is median ×3 and max ×10. With the current proposal mix that is **≈ 480 CPU-h (median bin) to ≈ 1,800 CPU-h (worst bin)**, an upper bound.
- **Where:** G 12–16 and G < 12, d 0.3–1.5 kpc, log P 2.6–2.92 (P ≈ 400–830 d), e < 0.6. The worst bins are G < 12 or 12–14, d 0.7–1.5 kpc, log P 2.75–2.92 (ESS_b 3–5).
- **How:**
  1. Re-centre the proposal on the best-fit θ (q tilt, low-mass slope, Δη, long-P weight) as a new deterministic-mixture generation (§12.8), so no stored draw is invalidated.
  2. Run a ~1 h tuning generation first to measure ESS per CPU-h under the re-centred proposal.
  3. Size the run from that measurement. Expect well under the unchanged-mix bound, though the gain is not yet measured.
- This is the long top-up that is on hold. It is Ryan's call.

## Options this raises (no choice made)

- **The f_s prior is in tension** (0.07 fitted vs 0.21 ± 0.04). Either the symmetric re-detection deficit is not all contamination, or the spurious template (MP-Q43 a: p_G from accelerations, flat e, exp in d) is wrong. MP-Q43 b (the template from the non-re-detected real orbits) is the natural test.
- **Normalization.** MP-Q42 b would make A interpretable as MdS17's f_logP;q>0.3 and reduce the A–Δγ_q–F_twin degeneracy.
- **Fitter.** MP-Q44 b (dynesty) is worth it only after a top-up; with ESS_b < 30 the posterior width is MC-dominated.
