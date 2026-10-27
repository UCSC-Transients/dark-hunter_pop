# Noise model v3: colour term, F2 calibration, single-star control (#572, #454, #574; #573 edge in `edge/`)

Spec: `docs/EPOCH_MODEL_SPEC.md` §8.12. Ryan's decisions of 2026-10-26, after the rung-3 v3 review (PR #571):
- add a colour term to the per-CCD noise model;
- refit F2 over the full G range;
- test the a0/σ_a0–RUWE edge against N (`edge/README.md`, #573);
- measure single-star false solutions (#574).

The model is behind `dr3.epoch_model.noise_colour.enabled` (**false**; version `colour_v1`). Production stays
`n2d_s1232`. No binaries were re-simulated.

## Data and runs

- **DR3 side.** The looks-single parent `20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj`, joined to the
  approved snapshot `20261027T054741Z_gaia_source_astrometric_quality_K1000000` (353,141 rows: gof_al, chi2_al, ε,
  n_good_obs_al, …; sha256 in its `meta.yaml`).
- **Rows.** Every §12.33 looks-single flag except RUWE < 1.25, G < 19, up to 6,000 rows per 0.5-mag G bin:
  62,209 rows.
- **Mock singles** (`scripts/single_star_noise_control.py simulate`). Each row is simulated at its own position, G
  and observed BP−RP, with true ϖ = 1/d_BJ.
  - It goes through the adopted §8 epoch model, gaiamock `predict_astrometry_single_source`, the per-CCD noise,
    `fit_5par_solution_only` and the full DR3 cascade with the publication cuts.
  - 16 noise realizations per row, 995,344 per run.
  - `before` (`n2d_s1232`): 8.8 CPU-h, 102 min on 6 workers. `after` (`colour_v1`): 5.2 CPU-h, 54 min.
- **Comparison cut.** RUWE < 1.25 on both sides.
- **Actual noise σ_act.**
  - DR3: σ_ϖ × UWE when ε = 0, else σ_ϖ.
  - Mock: the c-inflated σ_ϖ.
  - Single-star DR3 `gof_al` includes ε, so it is ≈ 0 by construction and is shown only. F2 is calibrated on orbits.
- **Orbits.** The 1,296 #390 published orbits re-injected with 2 realizations
  (`scripts/reinject_noise_colour_572.py`): production, colour model with e = 1 (calibration), and the final model
  (validation).

## 1. Colour term (fit: `fit/colour_fit_iter0.json`; figure `fit/colour_fit_from_n2d_s1232.png`)

**Model.** ln s(G, C) = a(G) + b(G) c + q c², with c = clip(BP−RP, 0, 3.5) − 1.2.
- a and b are piecewise linear in G; s multiplies the actual per-CCD noise and replaces the N2d excess.
- **Target:** the median paired ln(σ_act,DR3 / σ_act,mock), plus the run's own ln s, in 104+ G × colour cells.
  The errors are bootstrap errors with a 0.01 floor, scaled by √(χ²/dof).
- **Term choice by BIC:**

  | model | χ² / dof | BIC |
  |---|---|---|
  | G only | 387/103 | 459.0 |
  | G + c | 229/102 | 305.2 |
  | G + b(G)c | 169/98 | 264.6 |
  | **G + b(G)c + qc²** | **162/97** | **262.1** (adopted) |

**Colour coefficients.**

| knot G | 10.75 | 12.25 | 14.0 | 16.0 | 18.5 |
|---|---|---|---|---|---|
| b (per mag of BP−RP) | −0.154 ± 0.044 | −0.029 ± 0.019 | +0.043 ± 0.010 | +0.023 ± 0.009 | +0.018 ± 0.009 |

q = +0.011 ± 0.006. The a(G) knots are in `config/config.yaml`. b is constant outside its knots.

**Iteration check.** Refitting on the colour-model run (`fit/colour_fit_iter1_check.json`) returns the same
coefficients within 0.3σ (b at G = 14: 0.042 ± 0.011), so the fit has converged.

## 2. F2 calibration (#454): stated-error scale e(G)

e was solved per G bin so that the median accepted-orbit F2 of the re-injected #390 orbits equals DR3's
(`f2/f2_calibration_from_colour_iter0_e1.json`). F2 is a deterministic function of χ²_red/e². The errors are a
bootstrap over systems.

| G (median) | 9.89 | 11.03 | 12.09 | 12.74 | 13.51 | 14.40 | 15.46 |
|---|---|---|---|---|---|---|---|
| e | 0.936 ± 0.023 | 0.932 ± 0.024 | 0.847 ± 0.013 | 0.848 ± 0.011 | 0.972 ± 0.010 | 1.021 ± 0.011 | 1.001 ± 0.009 |

u0(G, BP−RP) (`config/epoch_model/dr3_ruwe_u0_colour_v1.csv`) is the 41st percentile of the mock single-star UWE ×
s/e. The direct-cell check is 0.997 (range 0.93–1.04 over 132 cells).

**Re-injected #390 orbits, mock/DR3 medians** (production: `f2/f2_before_n2d.json`; final model:
`f2/f2_after_colour_v1.json`):

| G bin | n (after) | F2 DR3 | F2 before | F2 after | a0/σ_a0 mock/DR3 before → after | σ_ϖ mock/DR3 before → after | RUWE mock/DR3 before → after |
|---|---|---|---|---|---|---|---|
| 6–10.5 | 174 | 4.71 | 4.44 | 4.92 | 0.866 → 0.870 | 1.010 → 0.963 | 1.023 → 1.018 |
| 10.5–11.5 | 172 | 5.30 | 5.34 | 6.00 | 0.763 → 0.806 | 1.092 → 0.999 | 1.014 → 1.046 |
| 11.5–12.5 | 381 | 5.21 | 5.25 | 4.73 | 0.853 → 1.013 | 1.127 → 0.948 | 0.950 → 1.039 |
| 12.5–13 | 209 | 4.52 | 4.10 | 4.25 | 0.974 → 1.106 | 1.012 → 0.884 | 0.977 → 1.084 |
| 13–14 | 471 | 0.04 | -0.07 | 0.27 | 0.965 → 0.996 | 1.016 → 0.994 | 1.013 → 0.993 |
| 14–15 | 363 | 0.16 | 0.21 | 0.41 | 1.001 → 0.971 | 0.992 → 1.014 | 1.037 → 0.983 |
| 15–17 | 104 | 0.44 | -0.17 | 0.39 | 0.965 → 0.951 | 0.991 → 1.006 | 1.025 → 0.991 |

Accepted fraction of re-injected realizations: before 0.696, after 0.723 (#390 rung 1: 0.742 Orbital). The after-run F2 medians agree with DR3 within the e bootstrap errors (±0.3–0.6 in F2); re-solving e on the after run returns the configured values within 1σ.

**Colour residual of F2 at G ≥ 13, BP−RP 2.5–3.5 (n = 31).** DR3 3.53 against production −0.10 and colour model
2.08. Bright red stars (G < 13, BP−RP 1.5–2) remain low: about 2.6 against 5.8 with the e-only calibration.
That is a documented residual, and e is not made colour-dependent: n = 44, and a bright colour term in e would
contradict the single-star σ_act.

## 3. Single-star comparison, before → after (`singles/`, `compare_before_after.json`)

**Paired σ_act DR3/mock by colour**, cells n-weighted (`sigma_act_ratio_before_after.png`):

| G | BP−RP: −0.5–0.6 | 0.6–0.8 | 0.8–1.0 | 1.0–1.2 | 1.2–1.5 | 1.5–2.0 | 2.0–2.5 | 2.5–3.5 |
|---|---|---|---|---|---|---|---|---|
| 9–13 before | 1.041 | 0.874 | 0.847 | 0.833 | 0.833 | 0.860 | 0.968 | – |
| 9–13 after | 1.167 | 1.003 | 0.988 | 0.977 | 0.983 | 1.043 | 1.168 | – |
| 13–16 before | 1.006 | 0.978 | 0.984 | 0.997 | 0.997 | 1.020 | 1.035 | 1.077 |
| 13–16 after | 1.023 | 0.995 | 0.994 | 1.002 | 0.996 | 1.005 | 0.997 | 0.997 |
| 16–19 before | – | 0.953 | 0.958 | 0.965 | 0.978 | 0.984 | 0.992 | 1.015 |
| 16–19 after | – | 0.990 | 0.994 | 0.998 | 1.008 | 1.004 | 0.994 | 0.992 |

- The rms of the cell ln ratios goes from 0.041 to 0.0135.
- The colour gradient at G ≥ 13 is removed: it was 0.95 → 1.08 blue to red, and is now 0.99–1.01.
- Remaining: the bluest and reddest bright cells (G < 13, few rows) are 4–17% high.

**Medians by G.** Bins are 6–9, 9–10, then 1-mag bins to 19.

| quantity | DR3 | before | after |
|---|---|---|---|
| RUWE | 0.986 0.986 0.977 0.98 0.999 1.00 1.01 1.01 1.01 1.01 1.01 | 1.07 1.06 1.07 1.06 1.06 1.05 1.05 1.03 1.02 1.02 1.01 | 1.03 1.02 1.00 1.02 1.03 1.01 1.02 1.02 1.01 1.01 1.01 |
| σ_act (mas) | 0.0193 0.0152 0.0152 0.0159 0.0140 0.0150 0.0212 0.0321 0.0524 0.0893 0.165 | 0.0184 0.0164 0.0183 0.0176 0.0166 0.0154 0.0212 0.0323 0.0542 0.0919 0.171 | 0.0201 0.0156 0.0158 0.0155 0.0139 0.0153 0.0216 0.0322 0.0528 0.0896 0.167 |

- **RUWE by colour.** DR3 is 1.001–1.010 at every colour. The mock was 1.007–1.048, with blue high; it is now
  0.997–1.022.
- **RUWE 84th percentile by G.** DR3 1.05–1.12. The mock was 1.05–1.17; it is now 1.05–1.15.
- **Against N_vis.** The paired ln ratio was +0.04/−0.02/−0.04/−0.04/−0.00/−0.03 and is now
  +0.06/−0.01/−0.02/−0.02/+0.02/−0.01. There is no N trend; the first bin is N_vis < 15.
- **UWE (AGIS-weighted).** DR3 single-star UWE is 1.25–1.57 at G < 13. That is the AGIS formal-error convention
  (u0_DR3), not the NSS stated errors that gaiamock emulates, so it is not a target.
- Figures: `singles_<ruwe|sigma_act|plx_snr|eps|f2|uwe|n_obs>_before_after.png`, each against G, BP−RP and N_vis.

## 4. Single stars misidentified as binaries (#574; `false_solution_rates_before_after.png`)

| run | realizations | reached orbit stage | accepted orbits | published accelerations |
|---|---|---|---|---|
| before (`n2d_s1232`) | 995,344 | 1,084 | 0 | 0 |
| after (`colour_v1`) | 995,344 | 770 | 0 | 0 |

- The per-star rate is < 3.0 × 10⁻⁶ (95%, if it does not depend on G).
- Scaled to the 3.79 × 10⁸ parent stars (G < 19, ϖ > 0.2 mas; × scale_to_full), white-noise single stars produce
  **< 1.1 × 10³** false orbits plus accelerations (95%). That is < 5% of the spurious template's N_s (prior
  21.4k ± 1.8k; v3 fit 41.7k).
- Per G bin (about 96k realizations each), the 95% upper limit is about 3 × 10⁻⁵.
- **The simulation can measure a rate of 10⁻³ per star**, and 10⁻⁵ per star overall. The rate is far below 10⁻³.
- So under the current Gaussian white-noise model, DR3's spurious orbits cannot come from single stars. They would
  need heavy-tailed or correlated errors, partially resolved pairs or similar, which are out of scope here.

## 5. What must be re-simulated if `colour_v1` is adopted

Every stored proposal-set draw, under MOCK_POPULATION_SPEC §3.5:
- luminous gens 42–46;
- co_v1 gens 1000–1003.

Their fragments carry the default `noise_model: n2d_s1232`. A colour-model generation sets
`noise_model: colour_v1`, and its primary-model version gains `:noise_colour_v1`, so `check_single_primary_model`
refuses any mix. The §12.16 spurious template must also be recalibrated: both re-injection samples need re-running,
because k_i and A_mock come from cascades with the mock noise.
