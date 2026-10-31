# Solution-type-aware bright model: e2 + st1 (#597; EPOCH_MODEL_SPEC §8.16, MOCK_POPULATION_SPEC §12.41)

**Verdict (2026-10-31): STOP — not adopted, and the queue is not restarted.** Iteration 2 fixes the headline problems: type fractions, the pooled a0/σ_a0 edge, the bright high-F2 tail and σ_ϖ. Several checks are still mixed or slightly worse, so under Ryan's rule ("if anything fails or is ambiguous, stop") the model stays OFF. See "Remaining issues".

## What was built (all OFF by default; `+e2` / `+st1` version tags)
- **e2: refit of the bright stated-error scale.**
  - The stated per-CCD error is multiplied by r(G) and u0 is divided by r(G) at G < 13. RUWE and every reported σ are unchanged; only χ² / F2 move.
  - r = 0.956 / 0.868 / 0.915 / 0.974 at knots G 9.5 / 10.5 / 11.5 / 12.5.
- **st1: routing.** P(AstroSpectroSB1 | predicted G_RVS, K1 from the astrometric orbit, BP−RP, e, P): a logistic on 125,251 DR3 C1 orbits, plus a G-dependent intercept so the post-cut fractions match.
- **st1: type-dependent solution properties.**
  - log offsets on σ_ϖ, σ_e, σ_P and σ_a0 by type, from the #390 same orbits; the Orbital σ_a0 offset is λ = 0.6 × the same-orbit value, set jointly with the edge;
  - for AstroSpectroSB1, an F2 quantile map.
- **Calibration files:** [routing.json](routing.json), [offsets.json](offsets.json), [iter2.json](iter2.json).
- **Bug found and fixed during validation:** the first e2 multiplied u0 by r, which inflated RUWE (#390 bright RUWE ratio 1.216).

## Validation, iteration 2 (iteration 1 kept in `*_iter1.json`, `validate_iter1/`)

**Single stars** ([validate.json](validate/validate.json)), G < 13.5: identical to v21tr1, so no regression. σ_ϖ envelope jump +23.0% (DR3 +24.6%), falloff 1.055, bright envelope / DR3 0.971.

**Gens 47–50 post hoc against DR3 C1 orbits** ([population_st1.json](population_st1.json)); mock unweighted, cuts re-applied.

| G 10–13 (pooled) | DR3 | before (v21tr1) | e2+st1 |
|---|---:|---:|---:|
| log z 99th pct, G 10–11 / 11–12 / 12–13 | 1.447 / 1.432 / 1.385 | 1.342 / 1.352 / 1.344 | **1.439 / 1.441 / 1.388** |
| log z median | 1.125 / 1.128 / 1.077 | 1.057 / 1.073 / 1.061 | 1.091 / 1.093 / 1.056 |
| P(F2 > 12) | 0.120 / 0.143 / 0.038 | 0.036 / 0.068 / 0.053 | 0.087 / 0.114 / 0.059 |
| log σ_a0/σ_ϖ median | 0.058 / 0.067 / 0.088 | 0.097 / 0.086 / 0.096 | 0.035 / 0.038 / 0.057 (worse at 11–13) |
| short-P (P < 100 d) share | 0.025 / 0.015 / 0.010 | 0.027 / 0.006 / 0.012 | 0.030 / 0.013 / 0.012 |

| AstroSpectroSB1 fraction after cuts, G 9–10 / 10–11 / 11–12 / 12–13 | DR3 0.622 / 0.629 / 0.574 / 0.342 | iteration 1 0.708 / 0.713 / 0.649 / 0.454 | **0.676 / 0.626 / 0.571 / 0.337** |
|---|---|---|---|

At matched G, the type fraction against log P, K1 and colour is within about 0.06. The exception is BP−RP < 0.6 (0.33 vs 0.23, the RV template T_eff ceiling).

By type (G 10–11 / 11–12 / 12–13), mock vs DR3:
- **F2 median, Orbital:** 7.27 / 8.79 / 5.86 vs 8.80 / 8.65 / 5.33 (before, pooled: 3.1 / 5.5 / 4.5).
- **F2 median, AstroSpectroSB1:** 3.45 / 3.60 / 2.92 vs 3.44 / 3.47 / 2.95.
- **P(F2 > 12), AstroSpectroSB1:** 0.041 / 0.028 / 0.015 vs 0.041 / 0.030 / 0.009.
- **log σ_ϖ:** within 0.03.
- **log σ_a0/σ_ϖ:** low by about 0.03 in both types.
- **Edge, log z 99th percentile:** Orbital 1.289 / 1.306 / 1.302 vs 1.326 / 1.330 / 1.345; AstroSpectroSB1 1.453 / 1.457 / 1.412 vs 1.463 / 1.450 / 1.411.

**High-F2 tail at G 10–12 by type and class**, P(F2 > 12), mock vs DR3:

| | mock before (pooled) | mock e2+st1 | DR3 |
|---|---:|---:|---:|
| Orbital dwarfs | 0.045 | 0.179 | 0.283 |
| Orbital evolved | 0.016 | 0.090 | 0.249 |
| AstroSpectroSB1 dwarfs | 0.045 | 0.050 | 0.042 |
| AstroSpectroSB1 evolved | 0.016 | 0.022 | 0.012 |

**Fresh #390 re-injection with e2+st1** ([validate.json](validate/validate.json), [sameorbit_by_type.json](sameorbit_by_type.json)):
- pooled, G < 13: σ_ϖ ratio 0.948 → 1.004; RUWE ratio 1.057 → 1.057; F2 median 5.07 → 5.29 (DR3 4.90); F2 > 3 0.757 → 0.789 (0.742); F2 90th percentile 11.0 → 12.2 (11.5);
- G ≥ 13: identical;
- the mock-routed type agrees with the DR3 type for only 34–78% of these same orbits.

## Remaining issues (why STOP)
1. **Same-orbit F2 tail of DR3-AstroSpectroSB1 orbits got worse.** Mock P(F2 > 12) is 0.13–0.20 against DR3 0.014–0.041; it was 0.08–0.15 before. About half of these orbits are routed Orbital in the mock and then carry e2's higher F2. Statistical routing cannot fix this per orbit; the population by type is fine.
2. **Same-orbit pooled F2 moved slightly away from DR3:** median dev 0.17 → 0.39, F2 > 3 dev 0.015 → 0.047, 90th percentile dev 0.5 → 0.7.
3. **log σ_a0/σ_ϖ (pooled, G 11–13) moved from +0.01–0.02 to −0.03 dex** against DR3. Both types are low by about 0.03, which suggests the AstroSpectroSB1 σ_a0 offset (−0.05 to −0.13) is too strong for the population, for the same selection reason as λ for Orbital.
4. **Short-P AstroSpectroSB1 overproduced:** share 0.046 (default weights) vs DR3 0.025. The smaller AstroSpectroSB1 σ_ϖ lets more short-P orbits pass ϖ/σ_ϖ > 20000/P. At G < 13 overall the default-weight short-P share goes 0.017 → 0.024 (DR3 0.017), driven mostly by G < 10; at G 10–13 it is about matched.
5. **Orbital-evolved high-F2 tail** is still ×2.8 short (0.090 vs 0.249). The Orbital F2 median at G 10–11 is still low (7.3 vs 8.8).
6. **G < 10:** type fraction 0.68 vs 0.62 at G 9–10 and F2 still high at G 6–9 (Orbital P(F2 > 12) 0.22 vs 0.06). Not ready to lift the exclusion.

## Suggested next iteration (for Ryan)
- Apply a λ-type shrinkage to the AstroSpectroSB1 σ offsets as well (σ_a0 and σ_ϖ), calibrated jointly on the population σ_a0/σ_ϖ and the short-P share by type. This addresses items 3 and 4.
- Item 1 is inherent to statistical routing. Judge it on population-by-type metrics, not same-orbit-by-DR3-type, or add covariates that predict DR3's routing better per orbit (e.g. RV template T_eff, rv_renormalised_gof / rv_amplitude_robust analogues for mock primaries).
- Items 5 and 6 need a G-dependent e2 knot at G < 10 (or keep the exclusion) and a colour/evolved term on e2.
