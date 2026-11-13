# Gate 5a: known-object injection (R5-T7 #624; verdict #647)

**Inputs.** 26 targets × 50 cascade realizations under `colour_v2.1+tr1+e2+st1` (`output/rung5/gate5a`; per-object outcomes in `injection_report.md`). Verdict and σ_ϖ diagnosis come from `scripts/gate5a_verdict_647.py`; numbers are in `gate5a_verdict.json`.

**Method.**
- **Response prediction.** For each object, the local mock probability of a C1 recovery and of an accepted orbit outside C1. It comes from the adopted-version dark compact-object draws (gens 1006–1008) in a kernel of ±0.2 dex M2, ±0.2 dex P, ±0.75 mag G and ±0.2 dex ϖ around the object.
  - The draws are unweighted: at fixed parameters, detection does not depend on the population weights.
  - M2 is the published mass for BH1 and BH2. For the NS candidates it is the f = 0 inversion at the host isochrone M1.
- **Test.** z = (f − p)/√(p(1 − p)/n + p(1 − p)/n_kernel). The test is one-sided (RUNG5_SPEC §5, clarified here): an object fails only if z < −2.

## Verdict: 16 pass, 1 fail (Gaia BH1), 8 accepted outside C1, 1 untested

| outcome | objects |
|---|---|
| pass (z > −2) | 16 NS candidates with P ≤ 830 d. All have z between +2.9 and +15: they are recovered *more* often than the kernel mock, as expected for detection-conditioned published parameters |
| **fail** | **Gaia BH1**: C1 recovery 1/50 against a predicted 0.41 (z = −3.8, 41 kernel draws) |
| accepted outside C1 | Gaia BH2 (P = 1352 d) and 7 NS candidates with P = 836–1046 d. Their accepted orbits have P_fit > 830.4 d; the mock predicts p_outside ≈ 0.4–0.8 for the same cells |
| untested | 1522897482203494784 (G = 11.05, P = 45.5 d; 5 kernel draws) |

- A two-sided test is not meaningful here: χ² = 906 for 17 tested objects, all from positive z apart from BH1.
- The published parameters of known detections are biased toward detectable configurations: their inclination, eccentricity and noise realization all favoured detection. This is why the criterion is one-sided.

## Gaia BH1: a per-object, not a population, σ_ϖ excess

**BH1 itself.** For 96 % of BH1's realizations, the cut that fails is ϖ/σ_ϖ > 20000/P = 108.
- The mock median ϖ/σ_ϖ is 96, against DR3's 120.
- Every BH1 error is about 1.3× its DR3 value: σ_ϖ 1.24, σ_a0 1.36, σ_P 1.27, σ_e 1.28.
- The mock F2 median is 1.4, against DR3's 0.3.
- The visibility periods agree (20 against 21).

A uniform scale across all errors points to a per-source noise level, not a scan-geometry effect. BH1's DR3 solution is quieter than the G-dependent noise model predicts for a typical source.

**The same orbits, all objects.** The ratio of injected to DR3 σ_ϖ across the 26 published orbits has a median of 0.965, with a 16–84 % range of 0.91–1.24. BH1 sits at the upper 1σ edge.

**The population.** Mock (rung-3 v5 weighted, gens 53–59) against DR3 σ_ϖ for accepted C1 orbits, as weighted medians with bootstrap errors:

| G | σ_ϖ mock/DR3 − 1 | mock ESS |
|---|---|---:|
| 10–11 | +2.4 ± 2.4 % | 435 |
| 11–12 | −1.7 ± 7.4 % | 74 |
| 12–13 | +6.6 ± 4.3 % | 141 |
| 13–13.5 | −3.1 ± 5.4 % | 86 |
| 13.5–14 | −3.7 ± 5.3 % | 97 |
| 14–14.5 | −8.0 ± 4.3 % | 81 |
| 14.5–15 | −2.9 ± 6.0 % | 38 |
| 15–16 | −4.9 ± 4.5 % | 35 |

At G 13–14.5 the mock σ_ϖ is, if anything, slightly *smaller* than DR3's (−3 to −8 %, each within 2σ). At fixed ϖ = 1–3 mas, the mock median ϖ/σ_ϖ is at or above DR3's (G 13.5–14: 90 against 84). There is **no population-level σ_ϖ excess** that would bias the selection function near the ϖ/σ_ϖ cut.

**Relation to the noise-model limitations** (EPOCH_MODEL_SPEC §8.16, MOCK_POPULATION_SPEC §12.41):
- The adopted noise model is statistical in G and colour, with no per-source scale. The bright per-source scale `ss1` (§8.15) exists but is off.
- Known-object recovery is therefore sensitive to where a particular source sits in DR3's per-source noise distribution. BH1 sits about 1σ quiet.
- This does not bias the population response.
- It is consistent with the known limitation that the pooled σ_a0/σ_ϖ ratio is about 0.03 dex low at G 10–13; the σ_ϖ level itself is within ±0.03 dex.

**Injection input defect.** The gate-5a injections passed BP−RP = NaN, so the colour term sat at its pivot and st1 routing used the pivot colour. The size is about 1–2 % on σ at G ≈ 14, so it does not explain BH1. It is tracked separately and the targets should be re-run with DR3 colours.

## Bearing on the first DR3 fit
Gate 5a does not block the fit. The dark response near the ϖ/σ_ϖ cut has no measurable population-level σ_ϖ bias (≤ 5 % at G 13–15). For BH1 specifically, the mock under-predicts its own recoverability; this is a statement about one source, not the selection function.
