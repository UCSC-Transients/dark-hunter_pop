# ss1 per-source noise scale and sa1 empirical σ_a0 scale (#595; EPOCH_MODEL_SPEC §8.15)

**Recommendation: do not adopt ss1, and sa1 is not needed. Both stay OFF.**

The bright "per-source dispersion" that motivated these terms (#589, #590, #593) turns out to be mostly the
**Orbital vs AstroSpectroSB1 split** of DR3. At G < 13 the two solution types sit on opposite sides of the mock:

| #390 same orbits, G < 13 (DR3 − mock, median) | Orbital (n = 230) | AstroSpectroSB1 (n = 210) |
|---|---:|---:|
| Δ log σ_a0 | +0.062 | −0.095 |
| Δ log σ_ϖ | +0.053 | −0.012 |
| Δ F2 | +1.48 | −1.44 |
| P(F2 > 12): DR3 / mock | 0.139 / 0.078 | 0.024 / 0.093 |

- In DR3 at G 10–12, P(F2 > 12) is 0.25–0.30 for Orbital solutions (median F2 8.7) and only 0.03–0.04 for AstroSpectroSB1 (median 3.4).
- The mock fits astrometry only, so it gives both types the same F2. Pooled, the two offsets look like an extra symmetric scatter: about 0.1 dex in σ_a0 and about 3 in F2.
- Split by type, the extra Orbital scatter (net of noise) is σ_a0 0.046 dex, which lies inside the #593 grid spread of 0.071; σ_ϖ 0.056 (grid 0.047); and F2 3.3 (grid about 1.6).
- **Step 2: no residual random σ_a0 term remains for Orbital solutions, so sa1 is not needed.** The AstroSpectroSB1 σ_a0 is systematically 0.10 dex lower, consistent with the RV data constraining P, e and T_p (#589 found that coupling carries 37–66% of σ_a0²). That is a type effect, not a random per-source scale.

## Implementation (kept, OFF by default)
- **Code:** `dr3.epoch_model.source_noise_scale` (ss1) and `source_sigma_a0_scale` (sa1) in config, the schema and `epoch_model`. They add "+ss1" / "+sa1" to the noise version.
- **RNG:** draws come from child Generators spawned off the noise Generator. With both terms off the outputs are bit-identical; the faint side, where σ = 0, also stays identical with ss1 on.
- **Run-script flags:** `--enable-source-scale` and `--enable-sigma-a0-scale` in `reinject_noise_colour_572.py` and `single_star_noise_control.py`.
- **Tests:** `tests/test_source_noise_scale.py`.
- **Calibration:** post hoc on the pooled #390 same-orbit set net of the #593 grid spread ([calibration.json](calibration.json)). σ(log10 s) = 0.052 / 0.0525 / 0.025 at G 9–11 / 11–12 / 12–13; no colour dependence by BIC. This calibration is now known to be driven by the type split.

## Validation of ss1 (fresh #390 re-injection, single-star control, gens 47–50 post hoc)

Runs:
- fresh #390 re-injection: `output/ss1/reinject_ss1`, 2,592 cascades, 75 min;
- single stars: `output/ss1/singles_ss1_r16`, G < 13.5, 71,696 realizations, 34 min;
- validation output: [validate.json](validate/validate.json), [f2_and_ruwe_vs_G.png](validate/f2_and_ruwe_vs_G.png), [sigma_plx_envelope_vs_G.png](validate/sigma_plx_envelope_vs_G.png).

**Headline: gens 47–50 population with ss1 applied post hoc ([population_ss1.json](population_ss1.json))**, P(F2 > 12) for C1 orbits, unweighted mock draws, cuts re-applied:

| G, class | DR3 (all C1) | mock before | mock with ss1 |
|---|---:|---:|---:|
| 10–11 dwarf | 0.128 | 0.040 | 0.075 |
| 10–11 evolved | 0.058 | 0.016 | 0.033 |
| 11–12 dwarf | 0.142 | 0.085 | 0.121 |
| 11–12 evolved | 0.094 | 0.013 | 0.066 |
| 12–13 dwarf | 0.040 | 0.060 | 0.081 (overshoots) |
| 12–13 evolved | 0.021 | 0.037 | 0.044 |

Published accelerations at G 10–12 barely move: P(F2 > 12) goes 0.12–0.27 → 0.16–0.31 against DR3's 0.38–0.47. The median F2 stays low: evolved 4.4 → 4.8 against DR3 10.8; dwarfs 7.1 → 7.7 against 11.6. So the bright acceleration F2 is a level offset, not a dispersion.

**Why the population and same-orbit results differ.** At G 10–12 about 60% of DR3's C1 orbits are AstroSpectroSB1, whose F2 is low.
- The #390 set mixes the two types. The mock overshoots the AstroSpectroSB1 tail (0.093 vs 0.024) and undershoots the Orbital tail (0.078 vs 0.139), so the pooled numbers looked close.
- In the population, the high-F2 tail is DR3's *Orbital* tail (0.25–0.30 at G 10–12). Mock astrometric fits at the same (A, a0/σ_a0, P, e, RUWE, d), all of which match DR3, reach only 0.03–0.04 before ss1.
- The deficit is a level/median offset for Orbital solutions (Δ F2 +1.5 on the same orbits), not a missing dispersion.

**Re-injection and single stars, before → with ss1 (DR3):**

| | v21tr1 | ss1 | DR3 |
|---|---:|---:|---:|
| #390 F2 median, G < 13 | 5.07 | 5.39 | 4.90 |
| #390 F2 > 3, G < 13 | 0.757 | 0.682 | 0.742 |
| #390 F2 90th pct, G < 13 | 11.0 | 12.6 | 11.5 |
| #390 σ_ϖ ratio, G < 13 | 0.948 | 0.951 | – |
| #390 RUWE ratio, G < 13 | 1.057 | 1.060 | – |
| #390, G ≥ 13 | identical | identical | – |
| singles: σ_ϖ envelope jump at 13 | +23.0% | +20.7% (worse) | +24.6% |
| singles: falloff 12.45 / 13.05 | 1.055 | 1.045 | 0.974 |
| singles: bright envelope / DR3, G 11–13 | 0.971 | 0.924 (worse) | – |

**Verdict.** ss1 closes only part of the population deficit and overshoots at G 12–13. It also degrades the single-star σ_ϖ envelope and its G = 13 jump, and fixes a symptom of the type split rather than a real per-source spread.

## Proposed next step (for Ryan): a solution-type-aware bright model
1. Recalibrate the stated-error scale e(G) (and the colour term) at G < 13 against DR3 **Orbital-only** F2 medians and tails. DR3 Orbital F2 median at G 10–12 is about 8.7, against the pooled 4.9 that e(G) was fitted to.
2. Route bright mock orbits to AstroSpectroSB1 with a calibrated probability (bright, RV-detectable; an SB1-routing calibration script exists, `scripts/calibrate_sb1_routing_391.py`). Give those orbits the measured type offsets: σ_a0 about −0.10 dex and F2 about −1.4 against an astrometry-only fit. Alternatively compare rung-3 F2 / σ separately by solution type.
3. Then re-check the a0/σ_a0 edge and the bright F2 tail by type.

Steps 2–3 change the population model: re-weighting cannot do them exactly, and they need spec work first.
