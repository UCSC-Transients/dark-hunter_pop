# A4 (#576): why the same orbit gets a lower a0/σ_a0 in the mock

Ryan (2026-10-27). This is light analysis of the existing same-orbit re-injections: the 1,296 #390 published orbits
with 2 realizations each.
- **before:** production `n2d_s1232` (`output/gate_noise_colour/reinject_n2d`);
- **after:** `colour_v1` with e(G) (#572, `reinject_colour_final`).

Script: `scripts/a4_same_orbit_576.py`. Numbers: `a4.json`. All ratios are log10 mock/DR3 on the *same* orbit and
are medians over accepted realizations: 1,803 before, 1,874 after.

## Decomposition of the same-orbit offset in z = (a0/σ_a0)/√(RUWE² − 1)

The split is exact per realization: log z = log a0 − log σ_a0 + log(1/√(RUWE² − 1)). σ_a0 itself splits into σ_ϖ (the
noise level) times σ_a0/σ_ϖ, a ratio that does not depend on the noise level ("geometry").

| term (log10 mock/DR3) | before | after |
|---|---|---|
| **z** | **−0.041** | **−0.019** |
| a0 | +0.003 | +0.002 |
| σ_a0 | +0.033 | +0.012 |
| · σ_ϖ (noise level) | +0.014 | −0.010 |
| · σ_a0/σ_ϖ (geometry, noise-free) | +0.020 | +0.021 |
| RUWE term −½ log((RUWE²−1)_m/(RUWE²−1)_d) | −0.004 | −0.007 |
| RUWE | +0.004 | +0.006 |

- **Before:** about 80% of the −0.041 dex comes from σ_a0 being too large. That splits into one third noise level
  (σ_ϖ +0.014) and two thirds geometry (+0.020). RUWE contributes about 10%.
- **After:** the colour/F2 model removes the noise-level part (σ_ϖ goes to −0.010), and the offset halves to
  −0.019 dex. What remains is the noise-free σ_a0/σ_ϖ excess (+0.021) and a small RUWE term.

**Dependence (after; `same_orbit_ratios_after.png`, `z_offset_decomposition.png`).**
- **σ_a0/σ_ϖ rises with A** = (a0/ϖ)/P^(2/3): −0.02 at A 0.05–0.1, then +0.013 / +0.016 / +0.024 / +0.040 up to
  A > 0.5. It also rises toward bright G: +0.06 at G < 11.5, +0.01 at G 13–15.
- It is flat in log P (0.016–0.040), e (0.015–0.031) and N_vis (0.01–0.03).
- The geometry term grows with orbit amplitude. That points to how σ_a0 is propagated from the Thiele–Innes
  covariance for large orbits, not to N or to the noise level.
- The bright z offset (−0.07 to −0.11 at G < 11.5) is mostly σ_a0: +0.054 / +0.093, with geometry +0.06.
- RUWE ratios are within ±0.035 dex everywhere. The largest is +0.035 at G 12.5–13 after the change.

## Are NSS orbit uncertainties scaled? Evidence

- **gaiamock (`vendor/overlays/gaiamock_mod.py`):**
  - `get_uncertainties_at_best_fit_binary_solution` forms cov = (JᵀJ)⁻¹ from the numerical Jacobian of the
    error-normalised residuals of the 12-parameter model.
  - It multiplies every uncertainty by c = √(χ²_red / (1 − 2/(9ν))³) with ν = N_obs − 12, and the A/B/F/G
    covariances by c².
  - σ_a0 then follows "Halbwachs+2023" (the code's own comment): first-order propagation of the Thiele–Innes
    covariance, including the cross terms.
  - The 5-, 7- and 9-parameter fits (`check_ruwe`, `check_7par`, `check_9par`) apply the same c.
- **DR3:** the published NSS astrometric-orbit uncertainties are inflated by the goodness-of-fit factor of Halbwachs
  et al. (2023, A&A 674, A9) Eq. 2. That is the c above, a function of F2 and ν. This repository already measures it
  from the published orbits (EPOCH_MODEL_SPEC §8.5: median c² − 1 = 0.54–0.96 at G < 13, about 0 above;
  ELBADRY2024_REPRODUCTION_SPEC §8: DR3 c is 1.22–1.34 × gaiamock's at G < 13 and 1.00 above).
- **Not checked:** the exact wording of the Gaia DR3 online documentation (NSS chapter, `nss_two_body_orbit` data
  model). Nothing was downloaded, as asked. The statement rests on Halbwachs et al. (2023) Eq. 2 as cited in the
  specs, and on the published `goodness_of_fit` / c relation measured here.
- **So both sides apply the same F2-based inflation.** The remaining noise-free σ_a0/σ_ϖ difference therefore lies
  elsewhere. Candidates:
  - ν and N: gaiamock fits per-CCD data with ν ≈ N_ccd − 12. With the epoch model, mock N_obs / DR3 `astrometric_n_good_obs_al` is a median 1.031 and a mean 1.049 on single stars (#581; the "13%" was the stale bare-gaiamock #398 number). This scales σ_ϖ and σ_a0 alike, so it cancels in the geometry ratio.
  - the a0/σ_a0 propagation at large A: DR3 `significance` against gaiamock's linearised a0 error;
  - the correlation of the Thiele–Innes elements with ϖ in DR3's full covariance.
  Separating these needs the DR3 per-row covariance (`corr_vec`), which is a follow-up.

## Paired before/after on the same orbits (`same_orbit_before_after.png`)

- **ϖ/σ_ϖ ratio against colour** (log10 mock/DR3):
  - before: −0.014 / −0.018 / −0.016 / −0.036 for BP−RP 0.6–0.8 / 0.8–1.0 / 1.0–1.2 / 1.2–1.5, i.e. too few high
    ϖ/σ_ϖ in the mock;
  - after: +0.017 / +0.011 / +0.011 / −0.010;
  - the bluest bin (< 0.6, few bright hot stars) goes from +0.038 to +0.071.
- **F2 against M_G** (G + 5 log ϖ/100, no extinction):
  - at M_G 1.5–2.5, before 3.8–4.3 against DR3 3.9–5.7; after 4.4–5.5;
  - at M_G > 4 both models are 0.3–1.5 above DR3.
- **a0/σ_a0 against RUWE:** both models are shown over DR3 for the same orbits.
- **RUWE ratio by G:**
  - before: −0.022 to +0.016;
  - after: −0.007 to +0.035, the largest at G 12.5–13.

## Secondary: fractional errors for the full populations (`population_fractional_errors.png`)

Weighted medians. Mock: rung-3 v3 luminous gens 42–46 accepted orbits at the default-target weights (production
noise). DR3: all Orbital + AstroSpectroSB1 orbits.

| | G 6–10.5 | 10.5–11.5 | 11.5–12.5 | 12.5–13 | 13–14 | 14–15 | 15–17 |
|---|---|---|---|---|---|---|---|
| σ_a0/a0 DR3 | 0.026 | 0.030 | 0.034 | 0.039 | 0.044 | 0.055 | 0.065 |
| σ_a0/a0 mock | 0.040 | 0.048 | 0.052 | 0.045 | 0.052 | 0.063 | 0.062 |
| σ_ϖ/ϖ DR3 | 0.006 | 0.008 | 0.009 | 0.011 | 0.013 | 0.018 | 0.022 |
| σ_ϖ/ϖ mock | 0.007 | 0.009 | 0.009 | 0.011 | 0.011 | 0.019 | 0.017 |

- The population σ_a0/a0 is 1.2–1.5× too large in the mock at G < 12.5, while σ_ϖ/ϖ agrees.
- Against A, mock σ_a0/a0 is 0.048–0.062 at A > 0.1 against DR3's 0.041–0.054.
- This is the same σ_a0 excess, made larger by the population: the mock's bright systems sit at lower a0.
- No colour axis for the mock population: the weights cache carries no colour.
