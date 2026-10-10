# #391 rung 3, refit 2: additive spurious component and Beta eccentricities (spec §12.11)

Ryan approved both model changes on 2026-10-09. The spec is PR #462. No long top-up was run.

- **Script:** `scripts/rung3_refit2_391.py`
- **Config:** `config/population/rung3_refit2.yaml` (same bins as before)
- **Outputs:** `report.txt`, `summary.json`, `ppc_six_panel.png`, `ppc_distance_and_ess.png`
- **Mixture:** gens 23–28, deterministic.
- **Cases:**
  - noW, the baseline (N_s free, with its prior);
  - noW with N_s fixed at the prior mean;
  - noW with no spurious component;
  - cmdW (N_s free).

## Setup

- **Spurious template** T = T_GD × T_PE.
  - T_PE comes from the PR #455 re-injection: π̄ = 0.184 with log P ratios [1, 0.82, 0.17, 1.49] and e ratios [0.97, 0.99, 1.35], times the real counts in the measured bin.
  - T_GD is the real orbits' (G, d) marginal. **This is an extrapolation.**
  - The template mass in the prior bin is 0.278, so the prior is **N_s ~ N(27,545, 5,247)**. It was fixed once.
- **Eccentricity:** Beta(a_k, b_k) on e / e_max(P) for log P < 2.0, 2.0–2.6 and ≥ 2.6.

## Results (noW baseline; cmdW in `report.txt`)

| θ | refit 2 | previous refit (PR #461) |
|---|---|---|
| ln A | +0.83 ± 0.21 | +0.14 ± 0.06 |
| α_lo / α_hi | +0.38 ± 0.15 / +1.33 ± 0.22 | +0.71 / +0.17 |
| γ_P | −2.22 ± 0.28 | −0.22 ± 0.17 |
| ln L_P | +1.84 ± 0.28 | +0.84 ± 0.17 |
| Δγ_q | **+3.00 (at bound)** | +2.48 ± 0.26 |
| ln F_twin | +0.61 ± 0.33 | −1.20 ± 0.29 |
| Beta range 1 (log P < 2) | a = **20 (bound)**, b = **1 (bound)** | (Δη = −0.45) |
| Beta range 2 (2.0–2.6) | a = 1.79 ± 0.19, b = 1.41 ± 0.29 | |
| Beta range 3 (≥ 2.6) | a = 1.11 ± 0.08, b = 2.20 ± 0.19 | |
| N_s | 52,207 ± 1,824 | (share-of-observed f_s = 2.43) |

All 12 starts converge, but only 4 land within Δ(−lnL) < 5 of the best. −lnL is 15,390, against 1,481 for the previous refit; the previous template filled every bin in proportion to the data, so the two values are not comparable.

- **The spurious tension is not resolved.** N_s = 52.2k is +4.7σ above the 27.5k ± 5.2k prior (cmdW +4.6σ). That is a 0.40 share in the prior bin and 40% of all C1 orbits.
  - With N_s fixed at the prior mean, −lnL rises by 181. The data push N_s up.
  - With no spurious component the fit **runs away**: mock counts reach 186× real. The reason is that **16 orbit bins holding 246 real orbits (0.19%) contain no mock draw at all**. Without a spurious floor their expected count is 0, and the optimizer inflates everything to reach them. Those bins are mostly:
    - d > 1.5 kpc at log P 2.6–2.75 (G 12–14, e 0.3–0.6: 120 orbits);
    - log P < 2.3 at 0.7–1.5 kpc;
    - e > 0.6 at short P.

    Part of the high N_s is the template filling these coverage holes.
- **Distance counts, (mock + spurious) / real:**

  | distance | ratio | first fit | previous refit |
  |---|---|---|---|
  | 0–0.3 kpc | **2.06** | 1.33 | 1.31 |
  | 0.3–0.7 kpc | 1.63 | | |
  | 0.7–1.5 kpc | 1.59 | | |
  | > 1.5 kpc | **4.11** | | 2.72 |

  The total is **1.77× real** (accelerations 1.18×); with N_s fixed it is 1.29×. Both overshoots, and the total, get worse.
- **Why the totals overshoot.** With ESS_b < 30 in **all 192** orbit bins (median 7.9 in the 33 heavy bins), the effective likelihood barely constrains a bin's normalization when one or two heavy draws dominate it. For α ≈ 1–2, a factor-2 overshoot costs < 1 in −lnL per bin. The fit uses that freedom to match shapes elsewhere, which pushes parameters to their bounds (Δγ_q, a1, b1).
- **Eccentricity at e < 0.1:**

  | | count |
  |---|---|
  | real | 14,230 |
  | refit 2 | **33,853** |
  | refit 2, N_s fixed | 24,573 |
  | previous refit | 25,433 |

  **Not fixed.** Range 1 runs to the bounds (a = 20, b = 1: all mass near e_max, which is unphysical) on very few short-P orbits. Ranges 2 and 3 get a ≈ 1.1–1.8, i.e. near-flat or rising densities at low e. The six-panel e shape still over-predicts e < 0.15.
- **e-proposal coverage at the fit:**
  - Ranges 2 and 3: max p/q ≈ 1.9–2.3, with E[w²] ≈ 1.5–1.8. Fine in every generation.
  - **Range 1: max p/q 29–35, E[w²] 15–18.** Bounded but poorly covered. The cause is a1 at its bound (a Beta piled at e_max), not the proposal. Range 1 needs a prior or a sensible upper bound on a, not new draws.
- **ESS at the best fit:**
  - Orbits 267 and accelerations 335; it was 397 and 915 in the previous refit, so it falls as θ moves away from the proposal centre.
  - In the 33 heavy bins ESS_b has median 7.9 and min 2.8, and none reaches 30.
  - Generation 28 alone runs at 2.6 ESS per CPU-h at this θ (7.4 at the first-fit θ it was centred on).
- **Updated top-up projection** (to ESS_b ≥ 30 in the 33 heavy bins, at this θ):

  | target | ESS factor | CPU-h, re-centred rate | CPU-h, old rate |
  |---|---|---|---|
  | median bin | ×3.8 | **≈ 290** | 560 |
  | 90% of bins | ×6.9 | **≈ 610** | 1,160 |
  | all 33 bins | ×10.7 | **≈ 1,000** | 1,910 |

  This is up from 90 / 250 / 540 at the previous refit's θ.

## Reading

Both model changes behave as specified. With the share-of-observed filler gone, the fit now shows two things the previous refit hid: **the likelihood is not informative enough at the current ESS**, and **the mock has coverage holes**. The parameters run to their bounds, the totals overshoot, the spurious amplitude sits 4.7σ high and e < 0.1 stays over-predicted. None of these numbers is a measurement yet.

## Options for Ryan (none chosen)

1. **Constrain the fit before sampling more.**
   - (a) Coarsen the bins (MP-Q45, e.g. merge the e and log P cells, or drop the G axis for orbits) so that most ESS_b ≥ 30 now.
   - (b) MP-Q41c: drop the bins with ESS_b < 30, or put a floor on the SAY α.
   - (c) Put priors on the Beta parameters (MdS17-like, a ∈ [1, 3]) and fix range 1 (log P < 2.0, few orbits) to MdS17.
2. **Coverage holes** (16 bins, 0.19% of the orbits; mostly d > 1.5 kpc and short-P / high-e). Accept them as spurious-only, or target them in the top-up.
3. **Top-up:** about 290–1,000 CPU-h at this θ, but only after 1 is settled. The proposal centre should follow whatever fit survives 1.
