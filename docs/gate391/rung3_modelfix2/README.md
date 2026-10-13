# #391 rung 3: Ryan's 2026-10-10 decisions implemented (f_evo, calibrated spurious template, near re-injection)

The spec is PR "§12.15–12.17" (f_evo, the calibrated template, the CMD-Malmquist ledger); the tracking issue is #471. **No proposal draws were made.** The one compute item was the approved near-distance re-injection.

## 1. MP-Q28 option A: free evolved-primary multiplier f_evo (§12.15)

- **Code.** `darkhunter_pop.rung3_model.evolved_log_modifier`, with tests. It is applied by reweighting on parent rows that the §10.2 classifier marks evolved; stored draws are untouched.
- **Likelihood axis.** A CMD-class axis (evolved vs dwarf + unclassified) on both the orbit and acceleration bins. The real side is classified with the same ridge and margin. Real evolved shares: 11.2% of C1 orbits and 11.6% of accelerations.
- **Fit** (interim grid, see 4): **ln f_evo = +0.27 ± 0.15 (noW) and +0.16 ± 0.10 (cmdW)**.
  - With the class axis in the likelihood, the evolved share is fitted directly: mock 11.8% against real 11.2%.
  - The 15% seen in the control fit came from parameters fitted without a class axis. Once the axis is present, the other parameters absorb most of it and f_evo is mildly above 1.

## 2. Spurious template calibrated on the re-injection samples (§12.16)

- **Data.**
  - The PR #455 sample: 300 real orbits at 0.7–1.5 kpc, G 13–16, plus 218 mock orbits.
  - The new near sample: 130 real orbits at d < 0.5 kpc, all G, plus 109 mock orbits.
  - Each orbit was re-injected 3 times, giving 1,288 real and 981 mock realizations.
- **Model.** k_i ~ Binomial(n_i, A_mock(x_i) · (1 − π(x_i))). Both A_mock and π are logistic in standardized covariates, with an N(0, 1) shrinkage prior on the slopes. Covariates enter by forward selection at ΔBIC ≥ 6.
- **What was reused.** The covariate definitions and the ΔBIC threshold come from the shared `spuriousness_model` (`config/spuriousness_model.yaml`). Its *fitted* model was not reused: it predicts whether a source is a usable compact-object candidate, trained on 293 high-mass-function candidates, whereas this template needs the self-consistency failure of the bulk orbit sample.
- **Covariates the data justify** (`spurious_calibration.txt`):

| model | selected covariates | coefficient per SD | ΔBIC when added | others tested (ΔBIC) |
|---|---|---|---|---|
| A_mock | log a0/σ_a0, then log ϖ/σ_ϖ | +1.04 ± 0.13, +0.42 ± 0.12 | +114.6, +6.6 | G and d (27.3, 21.0 alone) add nothing once significance is in (−2.1, −2.8) |
| **π** | **log a0/σ_a0 only** | **−0.58 ± 0.12** (intercept −1.53 ± 0.11) | **+16.9** | G +4.6 alone, then −6.1; **BP−RP −3.2**; d −0.2; log P −6.0; e −7.1; ϖ/σ_ϖ +8.6 alone, then −5.8; F2 −1.1; harmonic distance −3.6 |

- **Reading.**
  - Lower-significance orbits are more often spurious.
  - **Colour is not supported by these data** (Ryan's "likely depends on colour" remains untested beyond this ΔBIC).
  - The apparent G and near/far dependence is carried by significance: mean π is 0.21 in the far sample and 0.14 in the near sample. The near re-injection alone gives real 0.748 against mock 0.906, so π ≈ 0.17.
- **Template.** T_b ∝ Σ π̂ over the real C1 orbits in each bin. Σ π̂ = 22,106, which is 16.9% of the real C1 orbits. The relative calibration uncertainty is σ_cal = 0.086, so the prior is N_s ~ N(22,106, 1,900).

**MP-Q46: extrapolation outside the calibrated range. A decision for Ryan; not chosen.**
- The calibrated log a0/σ_a0 range is that of the 430 re-injected systems.
- Options:
  - (i) clamp, the provisional default;
  - (ii) linear-logit extrapolation;
  - (iii) π = 0 outside the range.
- Measured effect: (i) and (ii) give the same fit to < 1% (`interim_new` vs `interim_new_logit`: N_s 23.0k vs 22.9k, every PPC ratio within 0.01). Option (iii) was not run.
- **Recommendation: (i) clamp.** It is conservative, and with the current sample the choice does not matter.

## 3. Near re-injection (d < 0.5 kpc)

- **Run.** 110 real systems × 3 realizations at d < 0.5 kpc, all G (`near_reinjection/`), with 10 workers under `nice`, BLAS pinned, at 12.8 CPU-s per realization.
- **Cost: about 2.5 CPU-h,** a little over the approved ~2 CPU-h. A first launch at 220 systems would have cost about 4.6 CPU-h; I stopped it after about 0.2 CPU-h and relaunched at 110. The finished realizations from the first launch were kept: 20 extra systems, included in the calibration.
- **Result.** Real 0.748 against mock-symmetric 0.906, paired +0.158 ± 0.037. Distance is not selected in π once significance is in.

## 4. Interim evaluation on gens 23–30. INTERIM, not §12.14-compliant; not the rung-3 fit.

**Grid.** No grid with the CMD-class axis is fully ESS-feasible. The fallback, the coarsest grid (`config/population/rung3_refit5_interim.yaml`), is orbit G (1) × d [0, 0.7, 6] × log P [0, 2.6, 2.919] × e [0, 0.4, 1] × class (16 bins), with acceleration d × class.
- G is **not** an axis, so the G shape and α_lo are unconstrained on this grid.
- The same grid is used for the before and after fits.

**Before / after** (noW; cmdW in the reports):

| | before: control model (uniform-share template, no f_evo) | after: f_evo + calibrated template |
|---|---|---|
| starts agreeing (Δ < 5) | 10 / 12 (cmdW 10 / 12) | 10 / 12 (cmdW 4 / 12) |
| at bound | a1, b2 | **α_lo**, a1, b2 |
| N_s vs prior | 27.4k vs 27.5k ± 5.2k (−0.0σ) | **23.0k vs 22.1k ± 1.9k (+0.5σ)** |
| (mock + spurious) / real; accelerations | 0.988; 0.998 | **0.999; 1.000** |
| d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc | 1.32 / 0.85 / 0.91 / 1.59 | **0.94 / 1.04 / 0.98 / 1.09** |
| mock-only by d | 1.15 / 0.62 / 0.70 / 1.38 | 0.78 / 0.88 / 0.77 / 0.88 |
| orbit:accel (real 0.475 / 0.375 / 0.357 / 0.425) | 0.442 / 0.394 / 0.326 / 0.413 | 0.519 / 0.368 / 0.352 / 0.465 |
| evolved share (real 0.112) | 0.105 | 0.118 (ln f_evo +0.27) |
| e < 0.1 (real 14,230) | 18,086 | **16,672** |
| G 5–9 / 9–10 / 10–12 / 12–14 / 14–15 / 15–16 / 16–19.5 | 1.61 / 1.17 / 0.98 / 0.90 / 0.96 / 1.16 / 3.40 | 1.56 / 1.22 / 1.01 / 1.04 / 0.98 / **0.59 / 0.35** |
| ESS of orbits; big bins below 30 | 1,159; 2 of 10 (min 18) | 1,480; **0 of 10** (min 37) |

**Reading.**
- The new model closes the count-level checks: the totals, the near and far overshoots and the N_s prior.
- The **G shape is not constrained** on this grid, and it gets worse at the faint end (G 15–19.5 at 0.35–0.59). The calibrated π puts spurious orbits where significance is low, not specifically at faint G.
- So the faint-G deficit is **not** explained by spurious solutions under the calibrated model. Its remaining candidates are the faint-end detection model (the forward model) or the low-mass population (α_lo is at its bound here). It needs the G axis, i.e. the §12.14 grid.

**Malmquist-dependent quantities** (ledger §12.17; decision open):

| | noW | cmdW |
|---|---|---|
| α_lo / α_hi, after | +5.0 (bound) / −0.41 | +5.0 (bound) / −0.39 |
| α_lo / α_hi, before | −1.90 / +1.46 | −1.58 / +1.03 |
| near (< 0.3 kpc) residual, mock-only (after) | 0.78 | 0.82 |
| near (< 0.3 kpc) residual, with spurious (after) | 0.94 | 0.99 |

α_lo is unconstrained on the interim grid, which has no G axis.

**Closed loop.** None of the decisions adopted today changes the §11.5 closed loop, by construction:
- §10.6 sets W = 1 for giants, so f_evo does not interact with W, and the universe uses the MdS17 target;
- the spurious template exists only on the real side;
- the bin rule acts only on the rung-3 bins.

No re-run was needed. The worst 2-D pull stays at 7.6 (`docs/gate418/`).

## 5. Top-up cost under the updated model (`interim_new/feasibility.txt`)

These are the §12.14-compliant grids plus the class axis, evaluated at the new best fit (noW and cmdW) and MdS17:

| grid (G × d × log P × e × class) | bins | failing big bins (share of real) | min big ESS_b | cost |
|---|---|---|---|---|
| **2 × 3 × 3 × 3 × 2** (least-failing) | 108 | **5 (11.7%)** | 12.5 | **≈ 220 CPU-h** (total-ESS scaling; worst per-cell estimate 340) |
| 4 × 4 × 3 × 3 × 2 | 288 | 14 (24.9%) | 8.6 | ≈ 390 |
| others | 144–384 | 8–16 | 3.4–7.8 | 440–1,210 |

The cost for the smallest compliant grid falls **from ≈ 500 to ≈ 220–340 CPU-h** under the updated model. Not started; this is Ryan's call.
