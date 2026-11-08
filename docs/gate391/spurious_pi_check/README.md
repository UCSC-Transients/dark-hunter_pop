# §12.16 π̂ covariate re-check (#609): proposal, not adopted

This is an analysis of the existing #601 re-injection logs (far: 300 systems at 0.7–1.5 kpc and G 13–16; near: 130 systems at d < 0.5 kpc), with A_mock held at the #601 calibration. No simulation was run. Script: `scripts/spurious_pi_check_609.py`. Numbers: [pi_check.json](pi_check.json). Σπ̂ is taken over the 125,226 real C1 orbits of the rung-3 fit domain, with the covariate clamped to each variant's calibrated range; ± comes from 300 coefficient draws.

## Covariate support (log10 ϖ/σ_ϖ; min / 5 % / median / 95 % / max)

| sample | min | 5 % | median | 95 % | max |
|---|---:|---:|---:|---:|---:|
| far | 1.40 | 1.49 | 1.70 | 1.96 | 2.10 |
| near | 1.61 | 1.74 | 2.20 | 2.77 | 2.87 |
| real C1 population | 0.78 | 1.54 | 1.93 | 2.51 | 3.46 |

- **Common support.** The two samples overlap only on 1.61–2.10, which holds 263 systems: 221 far and 42 near.
- **Population outside the calibrated range** of V0: 0.55 %, which is clamped. 45 % of the population lies at d < 0.5 kpc, where only the 130 near systems constrain π.

## Variants

| variant | BIC | mean π̂ far / near | Σπ̂ (N_s prior) |
|---|---:|---|---:|
| V0 ϖ/σ_ϖ (the #601 choice) | 1163.2 | 0.190 / 0.036 | 12,729 ± 2,006 |
| V1 a0/σ_a0 (the previous choice) | 1171.1 | 0.200 / 0.116 | 19,455 ± 2,571 |
| V2 ϖ/σ_ϖ + a0/σ_a0 | 1168.7 | 0.208 / 0.046 | 14,453 ± 2,251 |
| V3 distance | 1198.1 | 0.156 / 0.097 | 15,444 ± 1,975 |
| V4 a0/σ_a0 + distance | 1176.8 | 0.193 / 0.134 | 20,507 ± 1,992 |
| V5 constant | 1196.4 | 0.132 / 0.132 | 16,586 ± 1,950 |
| **V9 ϖ/σ_ϖ + near step 1[d < 0.5 kpc]** | **1160.9** | 0.185 / 0.058 | **13,691 ± 1,483** |
| V10 near step only | 1190.5 | 0.159 / 0.090 | 16,015 ± 1,766 |
| V6 common support: ϖ/σ_ϖ | 701.3 (263 systems) | 0.136 / 0.033 | 9,711 ± 2,177 |
| V7 common support: constant | 701.5 | 0.093 | 11,663 ± 2,407 |
| V8 far only: ϖ/σ_ϖ | 854.8 | 0.180 / 0.023 | 11,270 ± 1,607 |

Fitted separately, the two samples give π_far = 0.161 and π_near = 0.086.

## Reading
1. **The ϖ/σ_ϖ slope is real inside the far sample.** Fitted on the far sample alone, it is −2.05 ± 0.35. On the common support the evidence is weak: ΔBIC 0.2 against a constant.
2. **V0 extrapolates that far-sample slope into the near regime.** It predicts π = 0.036 for the near systems, while their own rate is 0.086 (V10 0.090; V9 0.058).
3. **No alternative clears the configured inclusion threshold (ΔBIC ≥ 6).** V9 is the best by BIC, 2.3 better than V0. So under the current rule, V0 stands.
4. **Σπ̂ spread.** Across the defensible variants (V0, V2, V9, V5) Σπ̂ spans 12.7k–16.6k, which is wider than σ_cal of a single variant.
5. **None of this explains the v4 N_s = 46k.** That is the luminous model's distant shortfall (#603).

## Proposal for Ryan (none adopted)
- **(a) Template shape V9: ϖ/σ_ϖ plus a step at the sample boundary d = 0.5 kpc.** The step lets the near regime keep its own rate. Σπ̂ = **13,691 ± 1,483**. It needs a rule exception: ΔBIC 2.3 is below 6. Alternatively, treat the step as a design term of the two-sample calibration rather than a selected covariate.
- **(b) Keep V0 and widen the N_s prior for the covariate choice.** BIC-weighted over V0, V2 and V9 (weights 0.24 / 0.015 / 0.75) gives Σπ̂ = **13,460 ± 1,700**.
- **(c) Close the support gap (recommended before the next refit).** Add a third re-injection sample in the gap and the faint tail: d 0.5–0.7 kpc at all G, plus d > 1.5 kpc at G 16–19 (log10 ϖ/σ_ϖ < 1.4, 3 % of the population). About 130 + 130 systems × 3 realizations, roughly 1 h of CPU. Then rerun the forward selection.
