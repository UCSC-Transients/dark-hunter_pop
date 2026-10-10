# #391 rung 3: re-centred tuning generation 28 and refit (spec §12.10)

Ryan approved items 1 and 2 on 2026-10-09. The long top-up was **not** started.

| | |
|---|---|
| Spec | §12.10, PR #460 |
| Generation-28 config | `config/population/proposal_set_restart_recentred28.yaml` |
| Code | re-centring modifiers on `M2PeriodShapeConfig` (`m2_p_shape_modifier`, with tests) |
| Script | `scripts/rung3_refit_391.py` |
| Refit config | `config/population/rung3_refit.yaml` |
| Outputs | `report.txt` (full numbers), `summary.json`, `ppc_six_panel.png`, `ppc_distance_and_ess.png` |

## 1. Re-centred tuning generation 28

- **Design.** The (log q, log P) shape component is moved to the first-fit best fit: q^2.40, twins ×e^−1.53, the P tilt −0.23, and the long-P weight ×e^0.58. The eccentricity floor component goes from 0.30 to 0.45. The defensive parts are unchanged, so coverage stays complete. The weights stay bounded: the tests check the ratio against both the MdS17 target and the refit target.
- **Parent unchanged.** The best-fit M1 modifier is within 25% of the current target at every M1, which is too small to justify a new component.
- **Run.**
  - 30,000 draws (indices 837,000–866,999), 8 workers, `nice`, BLAS pinned, detached with `nohup`. The resume command is in `output/proposal_set/gen28_recentred.log`.
  - Wall time about 40 min; 4.1 CPU-h; peak about 3.3 GB.
- **Efficiency.** Measured at the first-fit θ, ESS of C1 orbits per CPU-h:
  - generation 28 alone: **7.4**;
  - gens 23–27: 2.3;
  - marginal gain from adding generation 28: 8.6.

  So the re-centred proposal is about 3.2× more efficient per CPU-h. Most of the gain is lower weight variance; the fraction of draws landing in the likelihood is unchanged (554 of 30,000).
- **Per-bin gain in the 33 bins that each hold ≥ 1% of real orbits.**
  - Median ESS_b goes 11.5 → 12.4; the minimum goes 3.0 → 3.1.
  - Bins at ESS_b ≥ 30: 0 → 1.
  - A 4 CPU-h generation adds only 1–3 effective draws per bin. Several of the worst bins received none: G < 12, d 0.7–1.5 kpc, long P; and G 14–16, d 0.7–1.5 kpc, e 0.3–0.6.
- **Projected top-up to ESS_b ≥ 30.** Per-bin rates from generation 28 are Poisson-noisy, so the projection scales the whole mixture's ESS at generation 28's rate:

  | coverage | ESS factor | CPU-h at the new rate | CPU-h at the old rate |
  |---|---|---|---|
  | median bin | ×2.4 | **≈ 90** | 280 |
  | 90% of the bins | ×5.0 | **≈ 250** | 790 |
  | all 33 bins | ×9.7 | **≈ 540** | 1,720 |

## 2. Refit (gens 23–28; MP-Q42b normalized q, MP-Q43b PR #455 template)

All 12 starts converge to one optimum. −lnL is 1,481; the orbit deviance is 1,600, down from 2,795 in the first fit.

| θ | refit (noW) | first fit (noW) | cmdW refit |
|---|---|---|---|
| ln A | **+0.14 ± 0.06** | +1.54 ± 0.13 | +0.73 ± 0.08 |
| α_lo | +0.71 ± 0.13 | +0.89 ± 0.14 | +1.75 ± 0.16 |
| α_hi | +0.17 ± 0.16 | +0.06 ± 0.16 | −1.22 ± 0.18 |
| γ_P | −0.22 ± 0.17 | −0.23 ± 0.17 | +0.01 ± 0.15 |
| ln L_P | +0.84 ± 0.17 | +0.58 ± 0.17 | +0.63 ± 0.15 |
| Δγ_q | +2.48 ± 0.26 | +2.40 ± 0.28 | +1.76 ± 0.21 |
| ln F_twin | −1.20 ± 0.29 | −1.53 ± 0.23 | −1.03 ± 0.33 |
| Δη | **−0.45 ± 0.03** | −0.45 ± 0.03 | −0.46 ± 0.03 |
| f_s (template multiplier) | **2.43 ± 0.08** | (0.100 share, old template) | 2.55 ± 0.08 |
| k_d | +0.25 ± 0.02 | −1.60 (old template) | +0.23 ± 0.02 |

- **MP-Q42b works.** Normalizing the q modifiers removes the degeneracy. A returns to near MdS17 (e^0.14 = 1.15× f_logP;q>0.3). Δγ_q and the twin suppression barely move, so they are a real shape preference: fewer q < 0.3 companions and fewer twins. Δη ≈ −0.45 and a long-P weight of about ×2 are stable.
- **Template measurement** (300 real + 218 mock orbits):
  - The overall spurious probability is π̄ = 0.184.
  - The log P marginal is concentrated at log P > 2.75 (r = 0.17 at 2.6–2.75, where real and mock acceptance are equal).
  - The e marginal is roughly flat (r ≈ 1), with r = 1.35 at e > 0.6.
  - Below log P 2.3 there is 1 system, so that bin is r = 1 by rule.
  - G and d are **not measured**: every system sits in 0.7–1.5 kpc, G 13–16. Outside that bin the template is extrapolated with no G dependence plus the free distance tilt. It is too sparse to say more.
- **The 3.5σ spurious tension does not resolve: it gets worse.**
  - The prior-bin share is 0.45 (cmdW 0.47) against the 0.21 ± 0.04 prior, a pull of +6.0σ (first fit: −3.5σ, at 0.07).
  - The spurious component totals 51k (39% of C1 orbits), and the mock part falls to 106k.
  - Cause: as specified, MP-Q43b makes the spurious count a share of the *observed* count in each bin. That makes it a free filler that can absorb any population deficit in the long-P bins, so it is degenerate with the population parameters. The prior cannot hold it.
  - **Options for Ryan** (none chosen):
    - (a) make the template a share of the *mock* expectation, μ_b, instead of the observed count;
    - (b) fix f_s at the measured amplitude (1) and fit the population only;
    - (c) MP-Q43a shape × the PR #455 log P and e ratios, with an analytic G and d.
- **Distance counts, (mock + spurious) / real:** 0–0.3 kpc **1.31** (first fit 1.33), 0.3–0.7 kpc 1.10, 0.7–1.5 kpc 1.14, > 1.5 kpc **2.72** (first fit 2.19). Accelerations are 0.96–0.97 out to 1.5 kpc and 1.25 beyond.
  - Both overshoots remain, and the > 1.5 kpc one grows through the spurious distance tilt (k_d > 0).
  - Total orbits are 1.19× real in the refit (1.09× in the first fit): the effective likelihood with ESS_b < 30 tolerates μ > k.
- **Eccentricity at e < 0.1:** real 14,230, first fit 22,824, refit 25,433. **Not fixed; slightly worse.** Δη < 0 lowers the mean e, which matches at 0.7–1.5 kpc (0.312 vs 0.322), but it piles systems up at e → 0. The real distribution turns over below e ≈ 0.15. One power law per (M1, P) cannot fit both the mean-e excess and the e < 0.1 deficit. That is a parameterization question (MP-Q22), and possibly also low-e fit bias in the real solutions that the mock reproduces differently.
- **ESS at the best fit:** 397 for orbits and 915 for accelerations. 190 of 192 bins have ESS_b < 30 (32 of the 33 big bins), so the Laplace errors are still understated.

## Bottom line

- The re-centred proposal is about 3× more efficient per CPU-h.
- Bringing all 33 heavy bins to ESS_b ≥ 30 needs about **540 CPU-h** (about 90 CPU-h for the median bin, 250 CPU-h for 90% of them) of re-centred draws. The worst bins should be targeted explicitly (G < 12 and G 14–16 at d 0.7–1.5 kpc), which generation 28 barely reached.
- Not started; waiting for Ryan.
- Before a top-up is worth spending, the spurious-template form (options above) and the e < 0.1 shape need a decision. Both are model issues, not sampling issues.
