# #554: why the gens 42–46 v12 refit goes to the high-twin / high-spurious mode (#391 rung 3)

Analysis only. No new draws were made and no generation was run. Every number below is computed from the stored gens 42–46
draws.
- Fixed-θ re-evaluations: three refit-5 dumps (`chain_xdump.sh`), covering θ(42–44) on data 42–46, θ(42–46) on data 42–44, and θ(42–44) on data 42–44.
- Four profile fits on the existing draws (`chain_profiles.sh`).
- `scripts/diag46_554.py`, which produced `d46/diag.txt`, `d46_theta42_44/diag.txt` and `d44/diag.txt`, plus the figures in `d46/`.

Every mock number carries its Kish ESS and the parent-row (§12.20) relative MC error. The scratch outputs (dumps, JSON) are in `output/proposal_set/v2_single/diag46/`.

The two modes:
- **θ44**: the v12 gens 42–44 fit, with dγ_q +0.40, ln F_twin −0.77 and N_s 22.6k.
- **θ46**: the v12 gens 42–46 fit, with dγ_q +1.00 (at its bound), ln F_twin +1.46 and N_s 30.3k (+4.8σ).

## Key numbers

**Likelihood (SAY, parent-row variance).** My recomputation reproduces the fit's −lnL of 796.0 exactly.

| data | θ44 | θ46 |
|---|---:|---:|
| gens 42–46 | 845.2 | **796.0** |
| gens 42–44 | **848.9** | 930.8 |

**Profiles on data 42–46.** Each holds a parameter group fixed and refits the rest (4 starts each, all converged).

| held fixed | −lnL | Δ | N_s (pull) | mock-only / real by d (<0.3 / 0.3–0.7 / 0.7–1.5 / >1.5 kpc) | ESS orbits |
|---|---:|---:|---|---|---:|
| nothing (θ46) | 796.0 | 0 | 30.3k (+4.8) | 0.99 / 0.83 / 0.61 / 1.04 | 326 |
| twins at θ44 | 804.6 | 8.6 | 30.1k (**+4.7**) | 0.73 / 0.81 / 0.75 / 0.94 | 627 |
| N_s at prior | 812.6 | 16.6 | 21.4k | 1.08 / 0.90 / 0.67 / 1.12 (dγ_q stays at its bound, ln F_twin +1.32) | 373 |
| e-shape at θ44 | 827.6 | 31.6 | 26.2k (+2.6) | 1.02 / 0.92 / 0.79 / 1.09 (ln F_twin → −4, its lower bound) | 579 |
| twins at θ44 and N_s at prior | 819.8 | 23.8 | 21.4k | 0.87 / 0.88 / 0.78 / 1.05; with spurious 1.04 / 1.04 / 0.96 / 1.26 | 761 |

## Answers

**Q1. The 0.7–1.5 kpc deficit.** It is not a weighting bug. Most of it is produced by the θ46 mode itself. A smaller part is a real model shortfall.

*Weighting and MC.* Each generation subset gives an unbiased estimate on its own, and the subsets agree. At θ46 the 0.7–1.5 kpc orbit expectation is:

| generations | estimate | ESS |
|---|---|---:|
| 42–44 | 26.7k ± 13% | 59 |
| 45 | 20.0k ± 18% | 32 |
| 46 | 23.4k ± 15% | 45 |
| all | 24.1k ± 9% | 131 |
| real | 39.8k | |

The subsets agree at θ44 too (`gens_subset_estimators.png`). The top draw holds 3.8% of the 0.7–1.5 kpc expectation.

*How much is the mode.* On the same draws, the mock-only/real ratio at 0.7–1.5 kpc is:
- 0.81 at θ44;
- 0.75–0.79 in the profiles;
- 0.61 at θ46.

*Distance definitions are consistent.* The mock uses 1/ϖ from the cascade's fitted orbit solution, with truth = 1/BJ. The real side uses 1/(ϖ_NSS + zp(G)). For parent rows at G < 16 the BJ-vs-parallax bin migration is ≤ 1%, and the median d_ϖ/d_BJ is 1.000–1.002 (`diag.txt` parent section).

*What the residual deficit is made of (θ44, 0.7–1.5 kpc).*
- **Class.** Dwarfs are 0.67 (ESS 126) while evolved systems are 1.50 (ESS 41). The mock's evolved share is 0.325, against 0.176 real. Re-classifying the mock with the orbit-solution σ_ϖ gives 0.294, so the classification asymmetry is only about 10% of the gap.
- **Observables.** By M_G, 2–3 is 0.42 and 0–1 is 1.92. At every distance the mock lacks high-SNR, large-a0 orbits. For log a0/σ ≥ 1.5 the ratio is 0.74 / 0.65 / 0.65 at d < 1.5 kpc. For a0/σ > 100 it is 0.43–0.51.
- **Photocentre size.** The real sample has a tail at large photocentre size A = (a0/ϖ)/P^(2/3) that the luminous model cannot produce (`amrf_by_distance.png`). For A > 0.5 the counts are:

  | d (kpc) | real | mock |
  |---|---:|---:|
  | < 0.3 | 561 | 80 |
  | 0.3–0.7 | 2,239 | 300 |
  | 0.7–1.5 | 2,768 | 1,739 |

*Parent "looks-single" depletion, which has no 1/P(looks single) weight (§12.33).*
- Under the model, a companion makes D = 0.12–0.24 of systems look non-single at G 11–16 and d < 1.5 kpc. That falls to 0.06–0.08 beyond 1.5 kpc.
- This matches the observed parent attrition of 0.09–0.29 (`lsdeplete`).
- The effect is a normalization bias that depends on G and d, with the mock low nearby. It points the opposite way to the 0.7–1.5 kpc deficit.

**Q2. Why gens 45–46 move the fit.** Low-ESS cells pick the mode.
- On data 42–46, θ46 beats θ44 by 49.2 in −lnL:
  - orbits +42.2, of which +29.3 comes from 33 bins with ESS < 30;
  - accelerations +18.5, mostly evolved bins with ESS 8–48;
  - the N_s prior costs −11.5.
  - The only well-sampled cell with a large effect is G 13–16, d < 1 kpc, log P < 2.6, e ≥ 0.3, dwarfs: k = 8,894, θ44 predicts 16.7k, θ46 predicts 10.7k (Δll +10.1, ESS 54–61). That cell is the short-P e-shape.
- On data 42–44 the θ44 optimum was held in place by **two evolved cells with ESS = 1.0** (Δll −59.8 and −26.9 against θ46; `d44/diag.txt`).
- Gens 45 (distant batch) and 46 (evolved and distant rows) gave those evolved and d > 1 kpc cells their first coverage, at ESS 3–40. At θ44 they then show:
  - too many evolved orbits at d > 0.7 kpc (×1.4–1.5);
  - too many d > 1.5 kpc orbits (×1.27 mock-only).
- The fit corrects this through weakly constrained directions:
  - ln f_evo goes from +0.53 to −0.46;
  - the short-P e-shape changes (the profile's strongest constraint, Δ 31.6);
  - ln A rises to +0.87;
  - q tilt and twin fraction go up;
  - N_s rises.
- **The twins are not needed.** Holding them at θ44 costs only Δ 8.6, and N_s still pulls +4.7.
- **θ46 rests on a population with ESS 20.** Near-twins (q ≥ 0.95) make up 20.8% of θ46's mock orbits (ESS 20.5), against 2.3% at θ44. A single gen-42 draw (q 0.995, log f +0.10) carries a weight of 2,813 at θ46, against 226 at θ44.

**Q3. The nearby twin excess.** It comes from the twin model and the flux-ratio scatter. It is not caused by the ΔM cut or by resolved-companion handling.
- **Mock vs real twin/single at 0.1–0.2 kpc.**
  - θ46: 1.44 ± 36% (twin ESS 11.7).
  - θ44: 0.28 ± 32%.
  - Real: 0.36.
  - θ44 is within about 1.5σ of the real ratio at every distance below 1 kpc (`twin_band_by_distance.png`).
- **What fills the mock twin band.** The members are genuine companions: 74% have q ≥ 0.95, and 97% have a primary on the ridge (|ΔM| < 0.25).
- **Other causes ruled out.**
  - Resolved companions are 0%.
  - Only 32% of the primaries have C0 < 1.05, where the row ΔM cut is not applied.
- **Why twins make orbits at all.** The 0.1 dex log f scatter (MP-Q13/Q40) gives near-twins a photocentre wobble they would not otherwise have.
  - At the coeval MIST relation centre the median |q/(1+q) − f/(1+f)| is 0.025, which is undetectable: the detection rate for Δlog f ∈ [0, 0.1) is 0.0000.
  - 93% of detected q ≥ 0.95 orbits come from draws with |Δlog f| ≥ 0.1 (wobble 0.06–0.2).
  - This makes ln F_twin a lever on orbit counts.
  - The CMD twin occupancy, which is not in the likelihood, contradicts the high-twin mode.

**Q4. The F2 offset.** It comes from the twins.
- At G < 13 and θ46, q ≥ 0.95 systems are 34% of mock orbits (ESS 13). Their F2 median is 8.45, and 39% have F2 > 10.
- Without them the mock median is 4.79 (θ44: 4.81), against 4.55 real. The residual is +0.25.
- The excess is concentrated at M_G 4–5 (twin median 11.2, ESS 2.8).

## Ranked causes

1. **The luminous-only rung-3 model is missing high-A, high-SNR orbits.** These are dark (WD) companions or hierarchical triples, which rung 4 has not added yet. The fit absorbs them with N_s, and N_s pulls +4.7σ even with the twins fixed.
   - The fit grid has no a0/σ or A axis, so the spurious template, which sits at low a0 SNR by construction, cannot be told apart from them.
   - dγ_q → +1 is the only luminous lever that produces larger a0.
2. **Low-ESS cells choose the mode.** Evidence: the evolved cells at ESS 1 (data 42–44) and ESS 3–12 (data 42–46), and the twin population at ESS 20. The profile Δs (8.6–31.6) are of the same order as the Δll contributed by bins with ESS < 30 (29.3 + 18.5).
3. **The 0.1 dex flux-ratio scatter is applied to coeval MS pairs.** It creates wobble for near-twins, which makes the twin parameters an artificial degree of freedom and drives the Q3/Q4 symptoms.
4. **The evolved and distant population is too abundant at θ44** (×1.4–1.5 at d > 0.7 kpc), which forces ln f_evo down. The cause (high-M1 / giant companion frequency, or the far parent) is not isolated, and the ESS is too low to isolate it.
5. **Looks-single depletion** gives a G- and d-dependent normalization bias of 6–24%. It is absorbed by ln A on average, not by shape.
6. **Minor causes.**
   - The class asymmetry from σ_μ (row vs orbit-solution σ_ϖ) is about 10% of the evolved share at 0.7–1.5 kpc.
   - f_evo multiplies on the system class rather than the primary phase. MS primaries are 1–4% of system-evolved weight.
   - The distance definitions are fine.

## Proposed fixes (docs-first; nothing implemented)

| # | Fix | Draws |
|---|---|---|
| F1 | Add the compact or dark companion component (co_v1, #552) before trusting N_s, or freeze N_s at its prior and treat its pull as a misfit diagnostic. Add log a0/σ_a0 (or A) as a fit axis, or as a posterior-predictive check, so spurious (low SNR) and dark companions (high A) are separable. | Axis or check: reweight only. Dark companions: new draws (co_v1 generations, already planned). |
| F2 | ESS guard on the likelihood: merge or drop bins with ESS_b < 10 at the fit θ, or inflate their variance until the SAY term is ≤ 1 unit of Δll, and refit. Report every fit with its profile on the twin and N_s directions. | Reweight only. |
| F3 | Flux-ratio scatter for coeval MS companions: σ_f → 0, or a q-dependent σ_f that vanishes as q → 1, keeping 0.1 dex only where age/[Fe/H] differences are physical (non-coeval or evolved). | Narrowing to σ ≳ 0.03 dex is reweightable from stored log f at an ESS cost of ~σ_old/σ_new in the twin cells. σ → 0 needs re-simulation (§3.5), because the cascade used each draw's f. |
| F4 | Constrain ln F_twin with the nearby CMD twin-band occupancy (real twin/single by d), as a likelihood term or prior. | Reweight only (system CMD is stored). |
| F5 | Top up the evolved and d > 1 kpc cells (ESS 3–12 now) before interpreting ln f_evo; make f_evo act on the primary's phase (iso_phase ≥ 2), not the system class. | Top-up: new draws (same primary-model version). Phase switch: reweight only. |
| F6 | Row weight 1/(1 − D(row)) for looks-single depletion, with D from the mock's own outcomes (published, RUWE ≥ 1.25, or system ΔM < −0.4); validate against the parent attrition. | Reweight only. |
| F7 | Classify mock systems with observation-like σ_μ (orbit-solution σ_ϖ) and the fitted parallax. | Reweight only (post-hoc observable). |

## Files

- `d46/diag.txt` covers data 42–46 at θ46, with θ44 as the alternative: generation subsets, decomposition, cascade stages, cells, parent, twins, F2, f-scatter, AMRF, heavy draws, depletion.
- `d46_theta42_44/diag.txt` covers data 42–46 at θ44.
- `d44/diag.txt` gives the cells and twins on data 42–44.
- `profiles/*/report.txt` holds the four profile fits.
- Figures in `d46/`:
  - `gens_subset_estimators.png`
  - `decomp_by_distance.png`
  - `amrf_by_distance.png`
  - `twin_band_by_distance.png`
  - `cells_dll.png`
