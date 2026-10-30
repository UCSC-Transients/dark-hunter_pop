# #587 go/no-go: rung-3 check of the noise model `colour_v2.1+tr1` (luminous gens 47–50)

**Recommendation: NO-GO** for spending ~40 h on gens 51, 52 and CO 1004/1005 in their current form. The new noise
model improves several 1-D noise distributions at d ≥ 0.3 kpc. The 2-D mismatches Ryan flagged are not reduced: the
sharp a0/σ_a0–RUWE upper edge, the bright high-F2 excess in G–F2, and the short-period bright systems. See
"Go/no-go" below.

## What was run (no new simulation)

* Gen 50 (gen-45 recipe) was assembled exactly as the #583 queue would: `run_heavy.sh single_gen50_assemble`,
  `assemble_partial_generation.py`, offset 8,500,000, run from the `colour-nan` worktree (git 71ee6ad). Result: 419,955 of
  500,000 draws (wall-clock stop). `queue.log` has a `STEP single_gen50 OK` line, and the queue stays paused at gen 51.
* Weight cache `output/proposal_set/v21tr1/weights_gen47_50.npz` (362 MB) covers gens 47 (481,588 draws), 48 (148,399),
  49 (161,385) and 50 (419,955). All four have version `looks_single_v2:ruwe1.25:dm-0.4:c1.05-2.75:g19:noise_colour_v2.1+tr1`
  ([build_cache_47_50.py](build_cache_47_50.py) refuses any other version). Gens 42–46 and co_v1 gens 1000–1003 are never mixed in.
* The model is the v3 configuration with WD **off** ([config](../../../config/population/rung3_noise_check_587.yaml), `--no-co`),
  R_dark = R_wd = 0. It uses the v21tr1 §12.16 spurious template (`output/proposal_set/v21tr1/spur/spurious_calibration`). New N_s prior: N(17133, 1790); v3's was N(21434, 1833).
  * **(a)** evaluates gens 47–50 at the v3 best-fit θ, with N_s at v3's 41,694 (`--eval-only`).
  * **(b)** is a luminous + spurious refit with 8 starts.
* Plots use the same script and style as v3, on v3's bin edges (`--edges-in`). χ² carries 30-fold bootstrap MC errors, 2-D included.
  * Plot sets: [(a)](a_v3theta/index.html), [(b)](b_refit/index.html).
  * Side-by-side comparison: [side-by-side figures](index.html).
  * Tables: [before/after table](compare/before_after.md), [noise metrics](compare/noise_metrics.md), [2-D χ² at d ≥ 0.3](compare/chi2_2d_d_ge_0p3.md).
* "v3 L+S" is v3's own dump with its WD component dropped. This is the like-for-like reference for (a) and (b).
* Chain: [chain.sh](chain.sh). Heavy steps ran one at a time through `run_heavy.sh`.

## Caveats that limit the test

1. **WD off is not a small change.** In v3 the WD component supplied 181k of the 325k accelerations and about 11k C1 orbits.
   * Without it, (b) has to produce the accelerations from luminous systems. That overshoots orbits by 2.3× (orbit:accel 0.79 vs DR3 0.38).
   * (b) is badly converged: 1/8 starts within Δ(−lnL) < 5, −lnL 35,610 vs v3's 16,911. `a1` and `a2` sit at their bound.
   * N_s is 23,559 ± 1,827. Its pull against the new prior is +3.6; v3's was +11.1 against the old prior.
   * **(b) is therefore not a usable noise test.** Its 2-D χ² is worse everywhere for this reason.
   * Every acceleration comparison without WD is dominated by the missing WD accelerations.
2. **Two draws dominate d < 0.3 kpc in (a).** Draws with weights 85,540 and 18,310 (G 11.0, e 0.74; G 13.2, e 0.99) give
   orbit ESS 2 in that bin. v3 has ESS 260 there.
   * Gens 47–50 re-simulate the recipes of gens 42–45 only. The nearby gen-46 recipe is gen 51, which was not run.
   * At d ≥ 0.3 kpc the ESS of (a) is comparable to v3's (129 / 71 / 15 vs 120 / 40 / 21).
   * The statements below therefore use **d ≥ 0.3 kpc**.

## Before / after (orbits, d ≥ 0.3 kpc unless noted; DR3/mock ratio, shape ratio in brackets)

| observable | v3 | v3 L+S | (a) new noise, v3 θ | verdict |
|---|---:|---:|---:|---|
| N(F2 > 3), G < 13 | 0.53 [0.72] | 0.55 [0.70] | 0.82 [0.87] | better |
| N(F2 > 8), G < 13 | 0.38 [0.52] | 0.39 [0.50] | 0.80 [0.85] | better |
| F2 median G < 13 (DR3 4.57) | 4.84 | 4.81 | 4.22 | similar |
| 2-D G × F2, χ²/dof | 12.0 ± 2.2 | 13.8 ± 3.1 | 18.9 ± 2.5 | **worse**; DR3 excess at F2 > 12, G 11–12.5 remains ([fig](compare/orbits_2d_g_f2_compare.png)) |
| 2-D M_G0 × F2, χ²/dof | 10.6 ± 2.8 | 12.5 ± 3.0 | 12.1 ± 3.5 | no change |
| ϖ/σ_ϖ > 238, (BP−RP)_0 < 0.8 | 3.18 | 3.26 | 1.73 | better |
| ϖ/σ_ϖ > 238, (BP−RP)_0 0.8–1.2 | 2.58 | 2.58 | 1.27 | better |
| ϖ/σ_ϖ > 238, (BP−RP)_0 1.2–1.8 | 3.54 | 3.54 | 3.58 | no change |
| ϖ/σ_ϖ > 238, log P 2.5–3 | 3.99 | 3.99 | 0.77 | better |
| 2-D colour × ϖ/σ_ϖ, χ²/dof | 14.4 ± 3.6 | 16.4 ± 3.1 | 17.1 ± 3.5 | no change |
| 2-D P × ϖ/σ_ϖ, χ²/dof | 20.6 ± 4.6 | 25.7 ± 5.3 | 15.1 ± 4.4 | better |
| a0/σ_a0 edge: DR3 fraction above mock p99 (expect 0.01), log RUWE 0.2–0.3 / 0.4–0.6 / 0.6–1 | 0.019 / 0.038 / 0.028 | 0.016 / 0.038 / 0.029 | 0.029 / 0.019 / 0.044 | **no change** |
| 2-D a0/σ_a0 × RUWE, χ²/dof | 46.4 ± 7.8 | 52.3 ± 7.4 | 57.7 ± 9.6 | **worse**; sharp mock edge unchanged ([fig](compare/orbits_2d_a0snr_ruwe_compare.png)) |
| N(P < 100 d), G < 13: all d / d ≥ 0.3 | 1.30 / 2.82 | 1.61 / 2.82 | 1.45 / 2.83 | **no change** |
| 2-D G × P, χ²/dof | 18.7 ± 3.1 | 19.6 ± 3.7 | 22.1 ± 3.7 | no change |
| 2-D G × RUWE, χ²/dof | 44.3 ± 5.0 | 50.0 ± 5.8 | 59.8 ± 5.7 | worse |
| 1-D F2, d 0.3–0.7 / 0.7–1.5 / 1.5–6, χ²/dof | 29.2 / 5.2 / 2.1 | 34.7 / 5.4 / 2.1 | 11.8 / 2.6 / 4.7 | better (nearer bins) |
| 1-D ϖ/σ_ϖ, d 0.3–0.7 / 1.5–6 | 25.1 / 11.1 | 25.2 / 11.1 | 4.8 / 1.9 | better |
| 1-D RUWE, d 0.3–0.7 / 0.7–1.5 / 1.5–6 | 41.0 / 29.3 / 13.9 | 67.1 / 29.6 / 13.8 | 13.8 / 17.4 / 2.7 | better |
| 1-D a0/σ_a0, d 0.7–1.5 / 1.5–6 | 95.3 / 23.5 | 117.4 / 23.5 | 13.0 / 5.1 | better |
| orbits total (mock / DR3), d ≥ 0.3 | 1.35 | 1.28 | 1.06 | better |

Bootstrap MC errors on the 1-D distance-split χ² are large (see [before_after.md](compare/before_after.md)), so single
entries are 1–2σ. The pattern across the d ≥ 0.3 bins is consistent, though.

Accelerations, (a) vs v3 L+S (shapes only, since WD is missing): the bright F2 median moves from 7.61 to 9.11 against DR3's 9.50 for G < 13 at d ≥ 0.3. That is a clear
improvement; v3 including WD gave 5.66. The G < 13 F2 90th percentile moves from 15.2 to 17.1 against DR3's 17.6.

## Go / no-go

**NO-GO** on resuming the queue unchanged.

* **What the new model fixed.** The marginal noise distributions improve substantially at d ≥ 0.3 kpc:
  * the 1-D F2, ϖ/σ_ϖ, RUWE and a0/σ_a0 shapes;
  * the overall high-F2 level at bright G;
  * the blue (BP−RP)_0 < 1.2 high-ϖ/σ_ϖ deficit;
  * the long-P ϖ/σ_ϖ tail;
  * the bright-acceleration F2 level.
* **What it did not fix.** The three structural 2-D mismatches in Ryan's list are unchanged:
  1. **The a0/σ_a0–RUWE upper edge is still sharp.** DR3 lies above it in the same diagonal band (2–4% above the mock 99th percentile, against the expected 1%). The colour/F2 noise terms do not move it. A likely cause is the RUWE ↔ a0/σ_a0 coupling in the cascade (σ_a0 scaling with UWE / the NSS uncertainty inflation, #576, #573), not the per-CCD noise level.
  2. **The bright high-F2 excess at G ≈ 11–12.5, F2 > 12 remains.** The level moved but the location in G–F2 did not. G × F2 χ² is not better. The likely cause is that the bright, high-F2 tail needs a non-Gaussian / outlier component at G < 13 (window-class / gate changes), rather than a scale e(G).
  3. **Short-period bright systems are still missing.** P < 100 d at G < 13 has DR3/mock 2.8 at d ≥ 0.3, unchanged. This is a population or acceptance issue (period-dependent cascade acceptance or the P shape at θ), not noise.
  4. The red (BP−RP)_0 1.2–1.8 high-ϖ/σ_ϖ deficit (×3.6) is untouched.
* **What a GO would need.** It needs a like-for-like test with a CO component under the new noise. The luminous + spurious refit cannot carry the accelerations: (b) overshoots orbits by 2.3× and is unconverged. If Ryan wants the test completed rather than the model revised, the cheapest order is:
  * CO gen 1005 (WD) first;
  * then gen 51 (nearby, fixes the d < 0.3 ESS);
  * then a joint refit, before gen 52 or 1004.

  Revising the noise model for items 1–2 should come first.

## Sizes

Scratch is `output/nc587/` (65 MB, gitignored: dumps and logs) plus the weight cache (362 MB). The docs here total
about 35 MB: (a) 14 MB, (b) 15 MB, compare 6 MB. Gen 50 is 258 MB, and its partial `.jsonl` (193 MB) is kept as the queue keeps it.
Disk free afterwards: 18 GB.

Also noted: idle orphan workers from the stopped gens 49 and 50 remain (`spawn_main`, PIDs 41630–41638, 99867–99875, 36758).
They were not touched.
