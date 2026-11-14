# RUNG5_SPEC.md — Rung 5: first DR3 WD / NS / BH companion mass functions (DRAFT, kickoff)

Tracking issue **#610**. Status: **accepted by Ryan 2026-11-06** (R5-Q1–Q12 decided as the recommended defaults; R5-Q6 superseded by the follow-up evidence layer). Docs-first: nothing here is implemented.
Ladder context: `docs/MOCK_POPULATION_SPEC.md` §5 (rung 5 = real inference), §12.37–§12.42 (rung 3), §13 (rung 4);
`docs/EPOCH_MODEL_SPEC.md` §8.12–§8.16 (adopted noise model `colour_v2.1+tr1+e2+st1`).

**Goal (Ryan, 2026-11-05).** "We should really start moving towards measuring the mass functions of each
population. DR4 is < 1 month away." Produce a first real measurement of the WD, NS and BH companion mass functions
from DR3, built so that DR4 is a config/data swap plus re-calibration, not a rewrite.

**Why a separate document.** `MOCK_POPULATION_SPEC.md` is 3,000+ lines and is being edited by open PRs (#608 adds
§12.43). Rung 5 is a different stage of the project (inference on real data, DR4 readiness), so it gets its own spec.
`MOCK_POPULATION_SPEC.md` §5 points here.

## 0. Settled decisions this spec respects (not reopened)

- **Classes.** WD, NS and BH are separate classes (ARCHITECTURE §4 `population_model`; MOCK_POPULATION_SPEC §13.0):
  WD hard-truncated at M_Ch; NS soft-truncated at M_TOV ≈ 2.2 M⊙ with a physical prior, marginalised; a dark companion
  with M > 3 M⊙ is a BH; NS vs BH near M_TOV is statistical (#551).
- **No NS mass-function functional form yet.** Non-parametric dN/dM per class (free-height bins; edges fixed before
  looking at real counts). Functional forms come after we look at the data (§13.7).
- **External compact-object populations are comparison-only**, never priors (pulsar MF, LIGO, Gaia 100 pc WDs,
  El-Badry 2024 astrometric WD masses).
- **Rung 5 owns** WD/NS contamination bookkeeping (§13.3 item 5) and the Phase-8 literature cut chains (R4-Q4).
- **WD target physics** (§12.39): non-interacting branch with the IFMR floor and MdS17 at the progenitor;
  post-mass-transfer He WDs on the Tauris & Savonije (1999) P–M relation; post-MT CO WDs at AU scales.
- **Per-object M2 likelihood** is a multivariate Student-t with ν set by a0/σ_a0 (#564, PR #566); gate 4b passes for
  dark companions.
- **Selection-function machinery** exists (rung 4): `compact_objects.build_selection_table`, co_v1 gens 1000–1003
  (old noise `n2d_s1232`), new-version CO gens 1006 (WD-focused, dark share 0.2) and 1007 (queued, #607).
- **Adopted noise model** `colour_v2.1+tr1+e2+st1`, with its known limitations (EPOCH_MODEL_SPEC §8.16).
- **Rung 3 is closed (Ryan, 2026-11-07; DR4 fast path) with one condition: rung 5 does not import the rung-3 spurious
  amplitude or WD rates** (#630; §2.6). Rung-3 v5: N_s = 40.7k against the four-sample π̂ prior 17.6k ± 1.5k (pull +15;
  the spurious term absorbs the distant luminous shortfall: spurious share 0.44 at 0.7–1.5 kpc, 0.79 beyond 1.5 kpc), and
  the WD component over-predicts A > 0.5 orbits at d < 0.3 kpc by ×4.3 with a WD orbit ESS of 32.
- Normalisation is per looks-single primary of the quality-cut parent sample, never an absolute Galactic density.

## 1. Gap analysis: what exists vs what rung 5 needs

### 1.1 Stage code (`src/darkhunter_pop/`)

| Module | What it does today | What rung 5 needs | Gap |
|---|---|---|---|
| `population_model.py` | 5-class type layer, free-height / GP dN/dM, `wd_hard_truncation_weight`, `ns_soft_truncation_marginalized`, two-tier dN/dM; consumes per-candidate **point** M2 and fixed companion-nature weights | Class rates per looks-single primary evaluated on **stored mock truth** (the rung-3 `joint_refit` pattern), with M2 the true mass and P/e/M1 laws per class | Large: no link to the proposal-set draws. Bin edges come from `bin_edge_policy: equal_log_m`, `n_mass_bins: 8` over one range; need per-class edges fixed from the mock (§2.3) |
| `inference.py` | 1-D Poisson in M2; intensity = MF × **scalar** astrometric SF × scalar follow-up SF × p_any; events are point masses with plug-in weights; dynesty + robustness suite + posterior-vs-prior overlap + zero-count upper limits | Λ(θ) from reweighted CO + luminous + spurious mock expectations; per-object or binned likelihood in observable space; MC-variance term | Large: the scalar SF (stand-in `scalar_selection_functions`, #252) and point-mass events must go. Keep the dynesty wrapper, robustness suite, overlap check and upper limits |
| `forward_model.py` (`selection_function_astrometric`) | Box-prior mock (`box_prior_mock_population` stand-in); six-panel plumbing check | The Step-1 chain: proposal-set draws + epoch model + cascade (#394) | #394 open. Rung 5 can bypass the stage at first (script level, §9) |
| `forward_model.py` (`selection_function_followup`) | Synthetic catalog vs synthetic twin (#351) | Not needed for the astrometric-only measurement (R5-Q6) | Declare `disabled_path`, SF_followup ≡ 1 |
| `sensitivity_analysis.py` | MC-noise convergence, dimensionality and per-class covariate BIC on a **synthetic fiducial catalog** (#350) | Run on the reweighted CO mock: which covariates (P, e, M1) each class rate needs; MC-noise gate per bin | Medium: replace the fiducial catalog with mock draws |
| `sbc.py` | `analytic_binned` Gamma-Poisson with SF = 1 (#356) and a `dynesty` backend over `run_inference` | Catalogs drawn from the **held-out** CO partition (and held-out luminous draws) through the rung-5 likelihood | Medium: new catalog generator; reuse rank/coverage code |
| `sample_selection.py` | Cut chains in `reproduction` / `forward_model` mode on real snapshot rows | The same chains applied to **mock** rows (posterior-predictive N per literature sample) | Medium: column adapter mock cascade → chain columns |
| `companion_nature.py` | ΔBIC channel wired; mostly `analytic_companion_nature_evidence` fallback; few Path-2 fits; #207 open | Not in the first result: WD vs dark separation comes from the forward model (WD light in the mock) | None for the minimal path; a later refinement |
| `compact_objects.py` | CO proposal/target, WD light, holdout partition, SF table, 12×12 covariance, M2 recovery with Student-t, gate 4b; `contamination_stand_in` | Response matrix per class (true-M2 bin → observable bin), contamination table, Student-t applied to **real** rows | Small–medium: extend, do not rewrite |
| `joint_refit.py` + `scripts/rung3_joint_391.py` | Binned observable-space Poisson over luminous + WD + spurious with per-family mixtures, F2–F6, MC variance fixed at reference θ | The rung-5 likelihood is this, plus dark (NS/BH) components with free-height M2 bins and an apparent-mass axis | Small–medium: the natural base (§3) |
| `run_validity.py` | Stand-in registry; science-validity block | Every rung-5 artifact lists the stand-ins it still uses | Small |

### 1.2 Declared stand-ins that rung 5 must replace or re-declare

| Stand-in | Where | Rung-5 action |
|---|---|---|
| `box_prior_mock_population` | `selection_function_astrometric` | Replaced by the Step-1 response (R5-T6; #394) |
| `scalar_selection_functions` (#252) | `inference` | Replaced by Λ(θ) from reweighted mock expectations |
| `synthetic_followup_catalog`, `synthetic_followup_calibration_twin` (#351) | `selection_function_followup` | Re-declared as `disabled_path` (SF_followup ≡ 1, astrometric-only MF) |
| fiducial catalog in `sensitivity_analysis` (#350) | `sensitivity_analysis` | Run on mock draws (R5-T8) |
| `analytic_binned` SBC (#356) | `sbc` | Replaced by held-out-partition SBC (R5-T9) |
| `per_sample_selection_weights`, `no_reproducing_literature_sample` | `inference` | Literature chains used as **comparison** (gate 5e), not in the likelihood |
| `analytic_companion_nature_evidence`, `analytic_wd_cooling_fallback` | `companion_nature` | Not used by the first result; declared as unused |
| `ci_scale_dynesty`, `fiducial_mass_function_heights` | dry-run | Replaced by the laptop recipe (§6) |
| `rung4_contamination_not_computed` | rung-4 table | Replaced by the contamination table (R5-T5) |

### 1.3 Rung-4 selection-function tables

- **co_v1 gens 1000–1003** are `n2d_s1232` (old noise). They are superseded and are never mixed with new-version draws
  (§12.40 guard). They remain useful only as a cross-check of how much the noise model moves the dark response.
- **Under `colour_v2.1+tr1+e2+st1`:** gen 1006 (WD 0.8 / dark 0.2) exists; gen 1007 (WD-focused, target-tabulated M2)
  is queued (#607). **There is no dark-focused generation under the adopted version.** The dark gen-1004 recipe was held
  at rung 3 (R_dark ≡ 0 there). Rung 5 needs it: **gen 1008, dark-focused** (gen-1004 recipe: re-centred rows, dark
  share 0.8, log P core 1.5–3.5, M2 log-uniform 1–40 M⊙), one 10-h / 90 CPU-h batch, possibly a second one if the
  ESS in the 3–40 M⊙ × A > 0.5 cells is below the gate (§5, gate 5d).
  Scale check: gens 1000–1003 gave a dark detected-draw ESS of 1,722 and 82/84 filled cells at ESS ≥ 30 with 0.3–0.6
  dark share. A 0.8-share batch should reach similar ESS by itself.
- **Gate 4b** must be re-checked on the new-version dark draws, and the Student-t ν re-calibrated on their held-out
  partition (`scripts/calibrate_student_t_564.py`). The noise model changed, so ν may change. **Done for ν (#637,
  MOCK_POPULATION_SPEC §13.13):** ν_core = 4.3 / 8.13 / 83 / 300 with 3.8 / 3.5 / 2.0 / 0 % catastrophic recoveries,
  mostly long-period interlopers carried by the response matrix.
  - **Gate 4b re-run with the adopted ν (2026-11-07; #639): PASSES, pointwise and by χ².**
    - Run: `scripts/build_co_selection_table.py` on the held-out partition of co_v1 gens 1006–1008, with the config ν
      `[4.30, 8.13, 83.08, 300]` for a0/σ_a0 bins 0–20 / 20–50 / 50–100 / > 100 (#638). Summary:
      `output/rung5/sf_1006_1008_nu638/summary.json`.
    - **The old-ν numbers are superseded** (ν `[2.21, 2.55, 8.18, 11.43]`, `output/rung5/sf_1006_1008/`). They failed pointwise in one bin: 68 % coverage 0.735 at 1.4–2.2 M⊙.
    - Dark companions, C1 domain. Errors are binomial with the Kish ESS:

      | M2 (M⊙) | n / ESS | median bias | 68 % coverage | 95 % coverage |
      |---|---|---|---|---|
      | 1.0–1.4 | 164 / 66 | +0.018 | 0.658 ± 0.058 | 0.924 ± 0.033 |
      | 1.4–2.2 | 221 / 99 | +0.007 | 0.710 ± 0.046 | 0.927 ± 0.026 |
      | 2.2–3.0 | 145 / 66 | −0.003 | 0.668 ± 0.058 | 0.934 ± 0.031 |
      | 3–5 | 277 / 121 | −0.003 | 0.653 ± 0.043 | 0.919 ± 0.025 |
      | 5–10 | 338 / 149 | +0.001 | 0.668 ± 0.039 | 0.943 ± 0.019 |
      | 10–20 | 386 / 174 | −0.005 | 0.667 ± 0.036 | 0.925 ± 0.020 |
      | 20–40 | 410 / 177 | −0.001 | 0.661 ± 0.036 | 0.916 ± 0.021 |

    - Bias passes: |median bias| ≤ 0.018. Coverage χ² is 1.52 on 7 dof (p = 0.98) at 68 % and 11.4 on 7 dof (p = 0.12) at 95 %, with 7/7 bins within ±0.05 at both levels.
    - **95 % coverage is 0.92–0.94, mild under-coverage.** The ~3 % catastrophic recoveries (outlier fractions in
      MOCK_POPULATION_SPEC §13.13) fall outside the 95 % interval and account for most of the gap. The explicit outlier
      component of R5-T12 should close the rest. **Re-check 95 % coverage once R5-T12 is built.**
- **Gate 4c (SBC)** has not been run with fitted rates. It is folded into gate 5c.
- **Gate 4d** is evaluated at rung 5 with fitted rates (§13.9).
- **Gate 4e (contamination closure)** has not been run. It is gate 5d.

## 2. Model

### 2.1 Data domain

- **Orbits: the C1 domain** of rung 3 (same parent cuts, G ≥ 10, P_fit ≤ 830.4 d, Orbital **and** AstroSpectroSB1).
  AstroSpectroSB1 is included: the st1 routing is adopted and DR3's bright C1 orbits are ~60 % AstroSpectroSB1.
  Excluding them would remove most bright NS/BH candidates (R5-Q2).
- **Accelerations (7/9-parameter)**: included as **binned counts** in the rung-3 cells (G, d, colour, significance).
  They carry no per-object mass, but they constrain the long-period rate of every class (the WD component predicted
  181k of 341k mock accelerations in v3). Minimal path: accelerations enter the luminous/WD fit as in rung 3, and the
  dark components contribute their predicted acceleration counts (R5-Q3).
- **Excluded:** SB1-only solutions (no inference entry point in v1), G < 10 (§12.36), fitted P > 830.4 d orbits (enter
  only through the response, gate-4b domain), DR3 `OrbitalTargetedSearch*` and other solution types.

### 2.2 Rate model

For each component k ∈ {lum, WD, NS, BH} plus the spurious term:

μ_b(θ) = μ_lum,b(θ_lum) + Σ_{k ∈ CO} Σ_j h_k,j R_k,bj + N_s T_b

- **Luminous** μ_lum,b(θ_lum): the rung-3 model (MdS17 θ, ln f_evo, v4 population terms). Default: θ_lum refit
  jointly with the CO heights, starting from the closed rung-3 fit (R5-Q4).
- **Compact objects:** h_k,j is the rate per looks-single primary in true-mass bin j of class k (free heights;
  dN/dlog M2 is piecewise constant inside the bin). R_k,bj is the **response matrix**: the expected number of DR3
  detections in observable bin b per unit height of true-mass bin j, computed once by reweighting the CO draws of one
  version (deterministic mixture, training partition) to the class's conditional target p_k(P, e, t_cool, M1 | M2).
  Because μ is linear in the heights, the likelihood is cheap and dynesty-friendly.
- **Class conditional laws (fixed in config; any other choice is a reweight):**
  - WD: the §12.39 physical P/e/cooling-age/M1 laws conditional on M_WD (non-interacting branch: IFMR floor and MdS17 at
    the progenitor; post-MT He on TS99; post-MT CO), **with the non-parametric heights replacing the §12.39 M_WD shape**.
    The WD rate parameters (A_ni, R_mt, f_He) are **free nuisance parameters fitted in the rung-5 cells** (§2.6); the
    rung-3 values are start points and centres of broad priors only. Hard truncation at M_Ch (`constants.M_CH`).
  - NS and BH (one dark population, two classes): P and e from the rung-4 target θ_CO (log P uniform 0.2–8; e uniform
    below MdS17 e_max(P), circular at P ≤ 2 d), every looks-single primary equally likely (R4-Q7). A kick-informed
    e law is a named model-comparison hypothesis (ARCHITECTURE §4), not the default.
  - **Tier 1** (classification-free): one dark dN/dM, free heights over 1–40 M⊙, no M_TOV assumption.
  - **Tier 2** (classified): NS = dark heights × S(M; M_TOV), BH = dark heights × (1 − S) below 3 M⊙ and all dark
    above 3 M⊙, with S the configured soft taper (`m_tov_soft_width_msun`) and M_TOV marginalised over its prior
    (`m_tov_prior_sigma_msun`, quadrature `m_tov_prior_n_quad`). This is #551: no hard 3 M⊙ cut below M_TOV.
  - WD/NS overlap (≈ 1.1–1.4 M⊙): both classes have heights there. They are separated only through the WD light the
    mock carries (hot WDs dilute the photocentre and change the colour) and the WD conditional laws. Responsibilities are
    reported per object, never forced; the separation is expected to be weak and is stated as such.
- **Covariates:** mass only (heights) unless `sensitivity_analysis` on the mock (R5-T8) shows a class needs P, e or
  M1. A covariate added later is a new config, never a silent change.
- **Spurious:** the §12.16 template T_b (π̂ recalibrated, #601) gives the **shape** only. Its amplitude is a free
  nuisance fitted in the rung-5 cells, with a separate free multiplier in the A > 0.5 / high-M2_app cells (§2.6). The
  rung-3 N_s (40.7k, pull +15 against the π̂ prior) is never imported.

### 2.3 Bins (fixed before looking at real counts)

- **True-mass bins** per class, edges fixed from the **mock** expected-detection density at the reference θ (ARCHITECTURE
  `bin_edge_policy`), roughly equal in log M:
  - WD: 0.15–0.45–0.6–0.8–1.0–1.2–M_Ch (the 0.45 edge separates He WDs);
  - dark: 1.0–1.4–1.8–2.2–2.6–3.0–5–10–20–40 (resolves M_Ch, M_TOV and the 2.5–5 gap).
  The config is committed (hash recorded in the run file) **before** any real-data likelihood is evaluated.
- **Observable bins:** the rung-3 v4 grid (G, d, P, e, A, colour) plus an **apparent-mass axis** M2_app (f = 0
  inversion with the isochrone M1 posterior, Student-t, identical estimator on mock and real rows): edges 0–0.6–1.0–1.4–
  2–3–5–10–40+. Accelerations keep the rung-3 cells.
- Merge rule (gate 5d): a cell used by the likelihood must have σ_MC/σ_Poisson < `physics.mc_noise_threshold` (0.1) at
  the reference θ, otherwise it is merged with its neighbour along M2_app, then P. The merge map is fixed before the
  real fit and committed with the edges.

### 2.4 Contamination

- Luminous + spurious expectations per observable bin come from the same fit (they are terms of μ_b), so contamination
  is bookkept, not subtracted.
- **Contamination table** (R5-T5; replaces `rung4_contamination_not_computed`): per (M2_app, A, P, G) cell, the expected
  luminous, spurious, WD, NS and BH counts at the fit with MC variance, and the inverse view: for each real orbit, the
  posterior class responsibilities (lum / spurious / WD / NS / BH).
- **WD → NS:** the expected number of WD companions with M2_app in the NS range, and how much of the NS posterior in the
  1.1–1.4 M⊙ bins they absorb (§13.3 item 5).
- **Robust region:** the AMRF class-III-like region (A above the luminous-companion envelope, M2_app > M1) is where
  luminous MS companions cannot reach; the dark measurement there is insensitive to the luminous model. Report results
  inside it separately as a robustness check.

### 2.5 Main scientific risk

At rung 3 the dark rate is degenerate with any luminous shortfall at A > 0.5 (§12.38: with R_dark free the fit put
1 % per star into dark companions). Rung 5 therefore requires, before the real fit:
(i) the rung-3 PPC in the A > 0.5 cells with R_dark ≡ 0 is acceptable **at M2_app < 1** (where dark companions are not
needed), and (ii) the closed-loop test (gate 5b) recovers injected dark rates of zero and of the El-Badry 2024 level
with the luminous model refit. If (i) fails, the first result is restricted to the robust region (§2.4).

### 2.6 Nuisance parameters fitted in rung 5, not imported from rung 3 (Ryan, 2026-11-07; #630)

Rung 3 was closed with a spurious amplitude and WD rates that absorb a luminous-model shortfall (§0). Rung 5 therefore
re-fits them in its own cells, A > 0.5 included.

| Nuisance | Rung-5 treatment | Prior (config; the rung-3 value is the start point only) |
|---|---|---|
| Spurious amplitude N_s | free; shape = the π̂ template T_b (#601) over all rung-5 cells | log-uniform on [0.25, 4] × Σπ̂ (the four-sample π̂ sum, 17.6k) |
| High-A spurious multiplier s_A | free; multiplies T_b in the A > 0.5 cells (and M2_app ≥ 1.4) | log-uniform on [0.1, 10] |
| WD rates ln A_ni, ln R_mt | free | normal, centred on the rung-3 v5 values, σ = 1.5 (e-folds) |
| WD He fraction f_He | free | uniform on [0, 1] |
| Luminous θ_lum | free (R5-Q4) | the rung-3 bounds; start at the rung-3 fit |

- **NS/BH–nuisance correlation.** In the A > 0.5 cells the dark heights, s_A, N_s and the WD rates trade against each
  other: an excess of large-A orbits can be dark companions, spurious orbits or massive WDs. Every rung-5 result reports
  the posterior correlation matrix of the dark heights with (N_s, s_A, ln A_ni, ln R_mt) and the dark heights with these
  nuisances held at ±1σ, and states that the dark rate is conditional on the astrometry alone.
- **What breaks the degeneracy is the §11 follow-up layer**: RV-confirmed/refuted labels fix s_A in the high-M2 cells
  (§11.3), and the UV excess plus the SED fits constrain the WD share (§11.2). The Gaia-only first result marginalises
  over the nuisances; the follow-up result replaces the broad priors with these measurements.
- **Gate 5b** (closed loop) injects spurious amplitudes and WD rates away from the rung-3 values (N_s at 0.5×, 1× and 2×
  the π̂ sum; s_A at 0.3 and 3; A_ni and R_mt at ±1 e-fold) and must recover the dark heights within their intervals.

### 2.7 Per-published-sample comparison (#671; ACCEPTED 2026-11-13)

**Status:** accepted 2026-11-13 (Ryan, PR #676). Nothing here is implemented. This section supersedes the first #671 draft (H1: a luminous-star mass split). Ryan's
2026-11-13 decisions on PR #676 are binding and are listed first. The evidence is the #671 diagnostic note.

**Terminology (Ryan, 2026-11-13).**
- **Luminous star:** the Gaia-detected star, mass M1.
- **Dark companion:** the unseen object implied by the orbit, mass M2, apparent mass M2_app.
- Every statement says whether it is about **real DR3** rows or **mock** draws.

#### 2.7.1 Ryan's decisions (2026-11-13)

- **Reproduction is paramount.** "We can only back out the underlying population if the simulation is identical to
  everything done with the real data." The rung-5 comparison is therefore made **per published sample**, and each
  sample's full cut chain is applied identically to real DR3 rows and to mock draws (`forward_model` mode).
- **Samples:**
  - Andrews et al. (2022) modified (`andrews2022_modified`, N = 25: published 24 plus Gaia BH1, Q14);
  - El-Badry et al. (2024) (`elbadry2024`);
  - El-Badry et al. (2026) astrometric subsamples (`elbadry2026`).
- **Overlap** is handled by the signed-off Q1 inclusion-indicator formulation (#112, PR #125; `sample_inclusion.py`):
  each unique source enters once.
- **The luminous-star mass split (former H1) is superseded.** The literature cuts already remove the A/F and evolved
  excess:
  - Andrews giant rejects (log g and CMD);
  - Shahaf AMRF class III;
  - El-Badry 2026 `main_sequence` and M̃2/M̃1.

  The orchestrator's real DR3 check: of 351 real C1 rows with M2_app ≥ 1.4, 183 have a main-sequence luminous star,
  47 of those have M2/M1 > 1.0, and 30 have M2/M1 > 1.2 (median luminous-star M1 = 1.00).
- **Kept from the first draft (approved):**
  - **H2:** cross-fitted spurious template.
  - **H3:** probabilistic bound f_s·s·max π̂ ≤ 1, now applied within each sample.
  - **H4:** a **power-law dark mass function** dN/dM2 ∝ M2^−α. α is free by default; exactly Salpeter (α = 2.35) is the
    fixed-slope comparison. The binned heights are kept as a check. Spill-over across the M2_app edges is then modelled
    by the forward model, not assumed.
- **H5 (decided):** a dark top-up is added only if gate 5d fails in the per-sample cells. Gate 5d itself is #675.
- **P1 (decided, #682):** reimplement Shahaf et al. (2023b) AMRF class III so the 177-candidate membership is
  reproduced exactly, so El-Badry 2024 can be forward-modelled.
- **P2 (decided, #680):** a GSP-Phot log g emulator for Andrews' giant reject.
- **P3 (decided):** El-Badry et al. (2023) Table E1 and the SB1 branch are comparison-only, never inference inputs.
- **Reproduction status is accepted as good enough to proceed** (Ryan, 2026-11-13, via the orchestrator). The current
  Andrews and El-Badry 2026 differences are not blockers. #335 and #133 are later follow-ups. §2.7.3 records the numbers
  re-measured on `main`; they gate nothing.

#### 2.7.2 Rate model per sample

For each published sample s with cell index b (observables after that sample's cuts):

  μ_s,b(θ) = Σ_k [population_k(θ) × SF_astrometric × SF_sample_s]_b + N_s,s · T_s,b

- **Terms.**
  - **population_k:** the luminous (rung 3), WD (§2.2) and dark components. The dark component is the H4 power law over
    1–40 M⊙ (normalisation and α).
  - **SF_astrometric:** the adopted cascade (`colour_v2.1+tr1+e2+st1`).
  - **SF_sample_s:** the sample's frozen cut chain (`config/selections/<s>.yaml`, `forward_model` mode), evaluated
    **per mock draw** on mock observables by the same `sample_selection` evaluator code path as the real rows.
- **The response for sample s** is the deterministic-mixture reweighting of the CO and luminous draws, restricted to the
  draws that pass the chain. No re-simulation is needed unless a cut needs an observable the draws do not store
  (§2.7.4).
- **Cells inside a sample:** M2_app (or the sample's own mass statistic, e.g. M̃2) × the observables the sample's cuts
  leave free (d, A, P). Bins are fixed from the mock before any real likelihood. The per-sample counts are small
  (tens), so the cells are coarse and the PPC gate (§2.7.5) is evaluated on them.
- **Overlap (Q1).** For each mock draw, the chains give a membership pattern m ∈ {0, 1}^S, which is exact per draw. The
  likelihood uses one point process whose cells are split by membership pattern; a real source with pattern m is
  counted once, in pattern m. This is the per-draw, deterministic form of `sample_inclusion.p_any`. No source is counted
  twice, and the nested samples (El-Badry 2026 `andrews2022_import` ⊂ Andrews; El-Badry 2024's A22 pool) are exact
  sub-patterns.
- **Spurious (H2, H3) per sample.**
  - **Template.** T_s,b = Σ π̂_i over the real rows that pass sample s, cross-fitted (K = 2 salted source_id folds), so
    no row is in its own template.
  - **Amplitude.** N_s,s = f_s · Σ_{i∈s} π̂_i, with f_s under the §12.16 calibrated prior N(1, σ_cal). The bound
    f_s·max_{i∈b} π̂_i ≤ 1 is enforced in every cell.
  - **One f_s across samples.** π̂ is sample-independent (§12.16, `spuriousness_model` philosophy), so f_s is shared.
  - **Validation targets only.** Each sample's published spurious rate (El-Badry 2024: 0.25 at G < 15; El-Badry 2026
    advisory, Q15) is a validation target, never an input.
- **Follow-up inclusion operators.** El-Badry 2024's `m2_joint_fit` / `not_spurious` / `orbit_coverage` and El-Badry
  2026's `not_spurious` / `followup_characterized` are catalog-level (`catalog_level_only`). The Gaia-only comparison
  uses the pre-follow-up candidate set: El-Badry 2024's union of 48, El-Badry 2026's pre-inclusion subsamples. The
  follow-up layer enters through SF_followup (§11) once it exists. Until then the post-follow-up N (21 COC; 227) are
  reported as comparisons.

#### 2.7.3 Reproduction status, re-measured on `main` @ c53c556 (2026-11-13)

**How this was measured.** `sample_selection.run_sample_selection` with the current `config/config.yaml`, on the
gate301 run's uncut-snapshot rows (`runs/20260930-022223-672b092.yaml`: data_acquisition + bulk enrichment), then
`sample_diagnostics.build_reproduction_comparisons`. Cost: 167 s, 6.6 GB peak, via `run_heavy.sh`.

| sample / subsample | mode | published / expected | ours | note |
|---|---|---:|---:|---|
| `andrews2022` | reproduction | 24 | 25 | extra row is still `6424213726885519744` (G = 15.15, pass-1 boundary p = 0.9545); **not** Gaia BH1, which is absent here as published |
| `andrews2022_modified` | forward_model | 25 (expected) | 24 | contains Gaia BH1; vs reproduction, drops `1749013354127453696` and `3649963989549165440` (forward-mode TAG10 M1) |
| `elbadry2024` union / COC | reproduction | 48 / 21 | 48 / 21 | match |
| `elbadry2026` union | reproduction | 227 | 225 | #335 |
| `elbadry2026` astrometric | reproduction | 76 | 74 | |
| `elbadry2026` spectroscopic | reproduction | 151 | 151 | N matches; ID sets differ (#335) |
| `primary_ns_bh` | reproduction | 47 | 46 | #133 |
| `andrews2022_import` | reproduction | 16 | 16 | match |
| `elbadry2023_table_e1` | reproduction | 5 | 5 | match |
| `sub_chandrasekhar` | reproduction | 22 | 22 | match |

Ryan accepted these as good enough to proceed (2026-11-13). They are recorded here and are **not** a gate. #335, #133
and the Andrews extra row are later follow-ups.

#### 2.7.4 Can each cut be evaluated on the mock?

**What the mock stores per draw.**
- Truth: luminous star, dark companion, orbit, `sys_bp_rp`, `sys_mg0`, `sys_evolved`, `atmosphere_logg`, position.
- Cascade fit: ϖ, A, B, F, G (Thiele–Innes) and their σ, P, e, a0, σ_a0, N_vis, N_obs, F2, RUWE.
- Fit covariance: `orbit_p0` and `orbit_cov`.
- Parent-row luminous-star M1: TAG10 (`proposal_set.tag10_m1_for_rows`) and MIST isochrone.

In the table, **Y** means computable from stored mock columns, **gap** means a column or emulator is missing, and
**N** means it cannot be forward-modelled.

| Sample / subsample | Cut | Real DR3 column(s) | Mock source | Mock? |
|---|---|---|---|---|
| Andrews 2022 (both) | parent: Orbital, C1 P range | `nss_solution_type`, period | `solution_type`, cascade P | Y |
| | `atf_covariance_valid`, `atf_pass1_root_found`, `atf_pass2_root_found` | ATF notebook on the NSS 12×12 | `orbit_p0` / `orbit_cov` through the same ATF code | Y (code path to wire: `build_andrews2022_atf_columns` on mock rows) |
| | `atf_m2_probability` P(M2 > 1.4) > threshold | ATF MC from the covariance + pass-2 M1 | same MC on `orbit_cov`; pass-2 M1 = TAG10 posterior (forward_model) | Y (wire) |
| | `atf_goodness_of_fit` | NSS `goodness_of_fit` | cascade F2 | Y. The F2 calibration is a known limitation (EPOCH_MODEL_SPEC §8.16) |
| | `atf_giant_reject_logg` (log g < dwarf min) | VizieR I/355 GSP-Phot log g | **none**: no GSP-Phot emulator | **gap**: GSP-Phot log g emulator (truth `atmosphere_logg` + calibrated noise / failure model) |
| | `atf_giant_reject_cmd` | BP−RP, M_G (ATF convention) | `sys_bp_rp`, G and ϖ_fit | Y (reddening convention must match ATF) |
| | `atf_m2_3sigma` | ATF pass-2 mean / std | same MC | Y (wire) |
| | modified: BH1 restored (scan-alias exclusion lifted) | n/a | n/a (the mock has no scan-alias rejection) | Y |
| El-Badry 2024 | `ns_candidate_pool` = Table 3 membership (A22 ∪ S23) | external table | A22 part = the Andrews chain above; S23 part = Shahaf 2023b class III | **A22 Y; S23 N** until an AMRF class-III reimplementation reproduces the Shahaf catalogue membership exactly (Q2) |
| | G < 15 | `phot_g_mean_mag` | `phot_g_mean_mag` | Y |
| | inclusion `m2_joint_fit`, `not_spurious`, `orbit_coverage` | follow-up labels | SF_followup (§11) | catalog-level: comparison only until §11 |
| El-Badry 2026 `primary_ns_bh` | `main_sequence` (dereddened CMD, Janssens segment) | M_G, BP−RP, E(B−V) from Green19 / Lallement19 at the position | `sys_bp_rp` / G / ϖ + the same dust-map E(B−V) at the mock (l, b, d) | Y (wire; the mock must carry **reddened** photometry and be dereddened by the same maps, not use truth `sys_colour0`) |
| | `m2_floor` M̃2 > 1.4; `m2_over_m1` M̃2/M̃1 > 1.2 | Janssens M̃1(M_G), M̃2 from a0, ϖ, P at fixed M̃1 | the same functions on cascade a0, ϖ, P and mock M_G | Y (wire) |
| | `goodness_of_fit` < 10, P < 1200 d, G < 15 | NSS columns | cascade F2, P; G | Y |
| El-Badry 2026 `andrews2022_import` | Andrews membership, G < 15 | Andrews chain | Andrews chain on the mock | Y, once the Andrews mock chain exists |
| El-Badry 2026 `elbadry2023_table_e1` | Table E1 membership | external table (5 rows) | none | **N**: no published selection function. Report as comparison; exclude from the forward-modelled likelihood |
| El-Badry 2026 `sub_chandrasekhar` (forward_model) | `main_sequence`, `m2_range` 1.05–1.40, M̃2/M̃1 > 1.0, `amrf` > threshold, `m2_error` σ_M̃2 ≤ 0.105 (analytic), P ≤ 900 d, G < 15 | Janssens chain; analytic σ_M̃2 from the NSS covariance | the same on `orbit_cov` | Y (wire). The reproduction mode (Shahaf class-III cross-match) is real-only by construction |
| El-Badry 2026 inclusion | `not_spurious`, `followup_characterized` | follow-up labels | SF_followup | catalog-level: comparison only |
| El-Badry 2026 spectroscopic branch | SB1 `k1_significance`, `mass_route` | SB1 | none (SB1 not in the mock cascade; Q6, v1 scope) | **N** (v1 scope boundary) |

#### 2.7.5 Kept gates

- **Gate 5d (#675)** in every used per-sample cell.
- **PPC per sample and per cell:** two-sided tail p ≥ 0.01 from ≥ 1,000 posterior draws, with gamma-Poisson MC noise.
  Every cell with M2_app ≥ 1.4 is reported by name.
- **Known real systems in a sample:** Gaia BH1 and the El-Badry good rows get their posterior responsibilities
  reported. None may exceed 0.5 spurious unless its π̂ alone (f_s = 1) gives that.
- **Power-law fit:** the free-α posterior is compared with the fixed Salpeter fit (Δ ln Z and PPC) and with the binned
  heights.

#### 2.7.6 Gap list → tickets (ordered; H = heavy, L = light)

1. **L — gate 5d** in `rung5_fit` (#675; already filed).
2. **L — mock → selection-row adapter.**
   - **Input:** proposal-set draws (truth, cascade, `orbit_p0` / `orbit_cov`, parent-row TAG10 and isochrone M1). **Output:** `sample_selection` rows with the real-row column names.
   - **Reddened photometry:** apply the same dust maps at the mock (l, b, d).
   - **Contract test:** the `sample_selection` evaluator runs unchanged on mock rows. Use fixture draws.
3. **L — Andrews ATF columns on mock rows.** Run the `build_andrews2022_atf_columns` code path on `orbit_cov`, with pass-2 M1 = TAG10 posterior (forward_model). Test it against the real-row sidecar on the published orbits.
4. **L — GSP-Phot log g emulator** for `atf_giant_reject_logg`, depending on P2.
   - Map truth `atmosphere_logg` to a GSP-Phot log g with noise and a missing-value model.
   - Calibrate it on real DR3 rows matched in G, BP−RP and M_G.
5. **L — El-Badry 2026 columns on mock rows:** `main_sequence` (same Janssens segment and dereddening), M̃1, M̃2, analytic σ_M̃2 from `orbit_cov`, AMRF and its threshold. Test against the real-row values on the published orbits.
6. **L — Shahaf 2023b AMRF class-III reimplementation (Q2, P1).** It must reproduce the published 177-candidate class-III membership exactly on real DR3 before it is applied to the mock. Without it, El-Badry 2024's S23 pool stays real-only.
7. **L — per-draw membership pattern and Q1 pattern cells.**
   - Per-sample cell definitions, frozen from the mock in a new `rung5_bins` schema version.
   - `rung5_build_inputs` per-sample mode: the response is restricted to chain-passing draws per pattern.
8. **L — spurious term per sample (H2, H3).** Cross-fitted (K = 2) template within each sample, f_s under N(1, σ_cal) shared across samples, and the bound f_s·max π̂ ≤ 1. Implemented in `rung5_fit` with tests.
9. **L — power-law dark mass function (H4).** α free plus a fixed-Salpeter variant, done by per-draw reweighting r_i ∝ M2^−α. The inputs therefore carry per-draw (cell, M2, w) rather than a fixed per-bin R. Keep the binned heights as a check.
10. **H — rebuild the rung-5 inputs per sample** (train and holdout) via `run_heavy.sh`, after items 2–9. This needs the operator's go-ahead.
11. **H — MC top-up if needed.** If gate 5d fails in the per-sample cells, run a dark generation restricted to main-sequence luminous stars that pass the sample cuts (R5-Q10 analogue). Operator go-ahead.
12. **H — per-sample fit:** dynesty ×3, gate 5b (closed loop per sample with the holdout), the per-sample PPC gate, and the known-object responsibilities.
13. **L — follow-ups, not blockers:**
    - #335 (El-Badry 2026 union 225 vs 227);
    - #133 (`primary_ns_bh` 46 vs 47);
    - the Andrews reproduction extra row (§2.7.3);
    - the mock F2 calibration (EPOCH_MODEL_SPEC §8.16, which feeds `atf_goodness_of_fit` and `goodness_of_fit`).

**Open questions for Ryan.**
- **P1.** El-Badry 2024: drop the S23 part of the pool from the forward-modelled likelihood until Q2 is resolved (only
  the A22 part forward-modelled), or reimplement Shahaf's AMRF triage first?
- **P2.** Andrews log g reject: build a GSP-Phot log g emulator, or (only if Ryan prefers) evaluate the Andrews chain
  on the mock without that cut and declare it as a stand-in?
- **P3.** Table E1 and the SB1 branch: comparison-only. Confirm?

## 3. Likelihood

- **Default (first result): binned inhomogeneous Poisson in observable space**, the rung-3 `joint_refit` likelihood
  extended with the CO components and the M2_app axis:
  ln L = Σ_b [k_b ln μ_b − μ_b] − ½ Σ_b (MC term), with the MC variance per bin fixed relative to the reference θ
  (`mc_variance_at: reference_relative`, as §12.39) and the F2 ESS floor. Plus the rung-3 extra terms (F4 twin
  binomial) and the §2.6 nuisance priors.
- **Per-object Student-t M2 likelihood** enters in two places:
  1. the M2_app estimator per row (posterior median under the isochrone M1 posterior and the `isotropic_log_m2` prior,
     Student-t tails with ν(a0/σ_a0)), applied identically to mock draws and real rows. Real rows use the DR3
     `significance` and the `nss_covariance` 12×12; no diagonal fallback;
  2. **the unbinned upgrade** (after the first result, R5-T12): in the dark region (M2_app > 1.4 or A above the luminous
     envelope, where the predicted luminous + spurious density is < 10 % of the total), each object contributes
     ln[(1/S) Σ_s λ(x_s | θ) / π(x_s)] over its S Student-t posterior samples with interim prior π, and Λ(θ) from the
     response (the hierarchical selection-corrected likelihood). The binned and unbinned results must agree within
     their errors (gate 5f).
- **MC-noise gate:** σ_MC/σ_Poisson < 0.1 per used cell at the fit (gate 5d), with the convergence diagnostic from
  `sensitivity_analysis`.
- **Multi-row sources** (#252): one row = one event, consistent with the response, which counts accepted rows.

## 4. Selection function / response

- R_k,bj is built from the training partition of the new-version CO gens (1006, 1007, 1008, ...) only; never from
  co_v1 1000–1003.
- The rung-4 table (`build_co_selection_table.py`, P(detected | M2, P, G)) remains the human-readable selection-function
  product and feeds `selection_function_astrometric` once #394 lands; the likelihood uses the response matrix, which
  carries the full observable distribution (not just the detection probability).
- Reweighting to θ_CO: the response is evaluated at the class conditional laws (§2.2); changing those laws is a
  reweight, never a re-simulation, as long as the proposal covers them (ESS reported per cell).

## 5. Validation gates (rung 5)

| Gate | Test | Pass |
|---|---|---|
| 5a | Known objects: Gaia BH1, BH2 analogues and the El-Badry 2024 NS candidates (published parameters) injected through the cascade; BH3 recorded as outside the C1 domain | **One-sided** (clarified 2026-11-08, #647): an object fails only if its C1 recovery is significantly *below* the response, z = (f − p)/√(p(1 − p)/n + σ²_MC) < −2, with p the local mock response (adopted-version dark draws in a kernel of ±0.2 dex M2, ±0.2 dex P, ±0.75 mag G, ±0.2 dex ϖ). Re-injecting published, detection-conditioned parameters biases recovery upward, so z > 0 is expected and is not a failure. Objects whose accepted realizations mostly have P_fit above the C1 ceiling are a separate outcome, **accepted outside C1** (a domain definition, never a failure). Each real object's posterior responsibility is reported |
| 5b | Closed loop: synthetic DR3 catalogs from **held-out** luminous + CO draws at known θ (dark rate 0, El-Badry-2024 level, ×10), through the full rung-5 fit | Injected heights inside the 68/95 % intervals at nominal rates |
| 5c | SBC (absorbs gate 4c): ≥ 200 held-out catalogs, θ from the prior | Rank statistics uniform, KS p > 0.01, for WD, dark tier 1 and NS/BH tier 2 (M_TOV marginalised) |
| 5d | MC noise + contamination closure (absorbs 4d, 4e) | σ_MC/σ_Poisson < 0.1 in every used cell; predicted contamination matches injected within errors |
| 5e | Literature comparison (comparison, never prior): forward_model-mode cut chains of Andrews 2022 modified, El-Badry 2024 and El-Badry 2026 astrometric applied to posterior-predictive mock catalogs | Predicted N vs observed survivor N, reported with Poisson + MC errors. Samples whose reproduction is not exact (#335, #133) are reported as advisory |
| 5f | Sampler: ≥ 3 dynesty runs (different seeds, nlive) and, for the unbinned upgrade, binned vs unbinned | Medians agree within 0.1σ; interval widths within 10 % |
| 5g | Small-N: posterior-vs-prior overlap per height; zero-count bins get Poisson upper limits | Reported, no pass threshold |

Plus rungs 1–4 green as in MOCK_POPULATION_SPEC §5, with rung 3 closed under the adopted version.

## 6. Sampler and compute (laptop only)

- dynesty over the CO heights (≈ 6 WD + 9 dark = 15) + the §2.6 nuisances (N_s, s_A, ln A_ni, ln R_mt, f_He) + the
  luminous θ (≈ 20). Default: θ_lum and the nuisances profiled
  at their joint optimum in a first pass, then a full dynesty run with θ_lum free (R5-Q4).
  The linear-in-heights structure makes one likelihood call ≈ the rung-3 call (ms), so dynesty with nlive 500 is
  < 1 h on the laptop. Three runs for gate 5f: a few hours.
- **Adopted sampler settings for the DR3 fit (#667).** dynesty 3.1 static sampler, `bound='multi'`,
  `sample='auto'` (rslice, ndim + 3 slices at ndim 23), `dlogz` 0.5, **base nlive 2000**; the three gate-5f runs use
  nlive 2000 / 3000 / 2000 (`rung5_fit_dr3.py fit --n-runs 3 --nlive 2000`, which alternates ×1.5) with distinct seeds.
  Reason: with nlive 500 / 750 / 500 (`fit_dr3_v2`) the posterior has a minor branch in which dark 1.0–1.4 M☉
  trades against WD 1.0–1.2 M☉ (log10 h_dark_0 down to the prior floor), whose weight differed ×6 between runs
  (< −5: 4.6 % / 1.9 % / 0.7 %); dark 1.4–40 medians spread 0.17–0.55σ. A tighter `dlogz` does not help (the branch
  sits at low L, explored early). The wall time scales ≈ linearly with nlive (v2: 19 min for 8 × 10⁶ calls), so
  the three runs take ≈ 1.5–2 h on the laptop.
- Heavy compute is only the new draws (gen 1008 dark, ≤ 2 × 10 h / 90 CPU-h) behind the #607 queue (≈ 30 h), and the
  known-object injections (minutes). SBC and closed loops are reweighting + fits: light-to-medium (≈ 200 fits × minutes).
- One heavy job at a time, `scripts/jobs/start_generation.sh` guards, `nice`, threadpoolctl.

## 7. Outputs

- Tier 1: dark dN/dlog M2 (1–40 M⊙), classification-free. Tier 2: NS and BH dN/dlog M2 with M_TOV marginalised.
  WD dN/dlog M2 (0.15–M_Ch). All per looks-single primary of the quality-cut parent, with posteriors.
- Derived: the NS fraction above M_Ch; the posterior occupancy of the 2.5–5 M⊙ gap (or a Poisson upper limit);
  the M_TOV posterior vs prior (overlap reported; never sharpened by external priors); the WD He fraction.
- Per-object responsibilities (lum / spurious / WD / NS / BH) for every C1 orbit with M2_app > 0.6.
- Posterior-predictive plots (the rung-3 set, plus M2_app and A × M2_app) and the gate 5a–5g reports in
  `docs/gate_rung5/`.
- Comparison-only overlays: Gaia 100 pc WD MF, El-Badry 2024 WD masses, pulsar MF, LIGO BH MF.
- Every artifact lists the stand-ins still in use (`run_validity`).

## 8. DR4 readiness

### 8.1 What changes in DR4

- **Epoch astrometry is published for all sources.** The per-object likelihood becomes exact: refit the epoch data
  with gaiamock's own fitter (ARCHITECTURE §6 option (b)) instead of the DR3 Gaussian + Student-t approximation. The
  Student-t ν calibration is then re-measured on DR4-style solutions (option (a), Gaia's NSS) or retired for option (b).
  Mock and real rows must use the same estimator (§13.12 framing).
- **66-month baseline:** C1 period ceiling (0.8 × baseline), longer-period orbits, fewer accelerations; σ scale with
  N_obs.
- **New NSS catalogue:** solution types, cuts and significance thresholds unpublished; the orbital/acceleration cuts
  in `dr4.selection_function_astrometric` are placeholders.
- **Epoch model and noise:** DR4 gaps, transit/CCD counts, per-CCD noise, the colour term, the G = 13 processing step
  (may move or vanish), u0, e2 and st1 must all be re-measured. `dr4.epoch_model` is `null` today.
- **Scanning law:** `dr4.scanning_law_id` pinned; gaiamock GOST/scan files for the DR4 window.
- **Epoch RVs** (ARCHITECTURE §6) may replace the empirical st1 AstroSpectroSB1 model by a physical RV term.
- **Parent and M1:** DR4 `gaia_source` parent snapshot, photometry, extinction; isochrone M1, giants classifier and
  Malmquist CMD weight re-built.

### 8.2 Re-calibration order (each step gates the next)

1. Snapshots: DR4 `gaia_source` parent, NSS table, astrometric-quality sample, epoch data (no live reads).
2. Epoch counts and gaps: `measure_epoch_counts_400.py`, `calibrate_epoch_model_400.py`, `calibrate_vp_tail_432.py`.
3. Transit / CCD-row: `epoch_transit_recal_581.py`.
4. Single-star noise: `single_star_noise_control.py`, `noise_colour_v2_579.py` (colour term, G-step), `ruwe_u0`.
5. Bright terms: `calibrate_source_scale_595.py` (if used), e2 + st1 (`calibrate_type_routing_597.py`,
   `calibrate_type_offsets_597.py`, `calibrate_st1_iter_597.py`), or their epoch-RV replacement.
6. Rung 1: re-inject published DR4 orbits (`injection_test.py`; #390 criteria).
7. Parent-side: isochrone M1 (`remeasure_bulk_isochrone_m1.py`), giants prior (`calibrate_giant_prior_418.py`), MIST
   low-mass (`calibrate_mist_lowmass_391.py`), Malmquist CMD weight.
8. Spurious template: `detection_vs_population_391.py` → `calibrate_spurious_391.py`.
9. Student-t ν: `calibrate_student_t_564.py` (option (a) only).
10. New version label; re-simulate luminous and CO generations; rung-3 refit; rung-4 response; rung-5 inference with the
    DR3 bin edges unchanged (shared population policy).

### 8.3 Config keys

- **Already path-specific (`dr3:` / `dr4:`):** `mission_baseline_months`, `scanning_law_id`, `zero_point_version`,
  `sed_filters`, `quality_cut_bins`, `nss_table`, `gaia_source_table`, `nss_enrichment_snapshot`,
  `allow_astrometric_epoch_outliers`, `selection_function_astrometric.{orbital_solution_cuts, acceleration_publication_cuts,
  d_min/max_pc, mock_g_mag_max}`, `rv_summary_root`, `vizier_apsis_snapshot_meta`, and **the whole noise/epoch model**
  under `<dr>.epoch_model` (gaps, transit loss, `ruwe_u0`, `transit_recal`, `noise_colour`, `bright_stated_refit`,
  `solution_type_model`, ...).
- **Shared (correctly):** M_TOV prior, M_Ch, IFMR, cooling tracks, IMF, mass-bin policy, MdS17 θ defaults.
- **Not yet path-specific (gap, R5-T13):** everything under `config/population/` — proposal-set fragments (parent
  snapshot, draw_index ranges), `rung3_joint_v*.yaml` (`c1_period_max_days`, spurious template and N_s prior, bin
  grid), `rung4_co_v1.yaml` (`gate_period_max_days`, `student_t_snr_edges`, `student_t_nu`), the decided-settings guard
  and version labels. Rung 5 must add a `data_release` key to each, and the version label must carry it, so DR3 and DR4
  draws can never be mixed.

## 9. Work plan

Sizes: S ≤ 1 day, M 2–3 days, L ≥ 4 days of agent work. **H** = needs heavy compute (one at a time on the laptop).

| Ticket | What | Size | Depends on | Can start now? |
|---|---|---|---|---|
| R5-T1 | Freeze rung-5 bin edges + merge map from the mock (WD, dark true-mass; M2_app axis) and commit | S | gen 1006 draws | **yes** |
| R5-T2 | Student-t M2_app for **real** DR3 C1 rows (nss_covariance, isochrone M1, `significance`) + the identical mock estimator; unit tests | M | — | **yes** |
| R5-T3 | Gen 1008 dark-focused CO fragment (gen-1004 recipe, adopted version, dark 0.8) appended to the #607 queue; 50-draw dry run | S + **H** (10 h) | #607 queue | fragment yes; run after #607 |
| R5-T4 | `rung5.py`: response matrix R_k,bj, class conditional laws, tier-1/tier-2 heights with M_TOV marginalisation, binned likelihood on top of `joint_refit` | L | T1, T2 | **yes** (on gen 1006 dark draws) |
| R5-T5 | Contamination table + per-object responsibilities (replaces `rung4_contamination_not_computed`) | M | T4 | after T4 |
| R5-T6 | Wire Step-1 response into `selection_function_astrometric` / `inference` stages (#394, #252); retire scalar SF | L | T4 | after T4 (not on the minimal path) |
| R5-T7 | Known-object injections (gate 5a): BH1, BH2, El-Badry 2024 NS candidates | S + light H | new-version cascade | **yes** |
| R5-T8 | `sensitivity_analysis` on mock draws (#350): per-class covariates, MC-noise convergence | M | T4 | after T4 |
| R5-T9 | Held-out-partition catalog generator; closed loop (5b) + SBC (5c) (#356) | M | T4, T3 | after T4 |
| R5-T10 | Mock → cut-chain column adapter; literature comparison (5e) | M | T4 | **yes** (adapter) |
| R5-T11 | First DR3 fit: dynesty ×3, gates 5d/5f/5g, plots, `docs/gate_rung5/` | M | rung 3 closed, T3–T5, T9 | blocked |
| R5-T12 | Unbinned per-object likelihood in the dark region + binned/unbinned agreement | M | T11 | blocked |
| R5-T13 | DR-path keys for `config/population/*` + version label carries the data release; DR3/DR4 audit extended | S | — | **yes** |
| R5-T14 | DR4 calibration runbook script (the §8.2 order as a dry-run driver; no DR4 data needed) | M | T13 | **yes** |

**Minimal path to a first DR3 result (~2–3 weeks):**
- Week 1 (parallel with rung-3 closure / #607 queue): T1, T2, T13, T7 now; T4 started on gen 1006's dark draws; T3
  fragment queued behind #607.
- Week 2: gen 1008 runs (H); rung-3 v5 refit on gens 53–59 + 1006/1007 (rung-3 owner); T4 finished, T5, T9 closed loop
  and SBC on held-out draws; gate-4b and ν re-check on gen 1008.
- Week 3: T11 first fit (binned, θ_lum profiled then free), gates 5a–5g, report to Ryan. T10 and T12 follow.
- What can slip without blocking the first result: T6 (stage wiring), T8 (covariates: default mass-only), T10, T12.
- What blocks it: rung-3 closure under the adopted version (the A > 0.5 PPC, N_s pull), and gen 1008.

## 10. Questions R5-Q1–Q12: **decided by Ryan 2026-11-06** (the recommended defaults below are adopted; R5-Q6 is superseded)

| Q | Question | Recommended default |
|---|---|---|
| R5-Q1 | New `docs/RUNG5_SPEC.md` vs MOCK_POPULATION_SPEC §14 | **Separate doc** (this one); §5 of the mock spec points here |
| R5-Q2 | Include AstroSpectroSB1 orbits | **Yes** (st1 routing adopted; most bright candidates) |
| R5-Q3 | Accelerations in the rung-5 likelihood | **Binned counts, no per-object mass**; dark components contribute predicted counts |
| R5-Q4 | Luminous θ fixed at rung 3 or refit jointly | **Refit jointly** (profiled first pass, then free in dynesty) |
| R5-Q5 | If the A > 0.5 PPC still fails at rung-3 closure (it did; rung 3 closed 2026-11-07, §0, §2.6) | **Restrict the first result to the robust (AMRF class-III-like) region** and state it |
| R5-Q6 | Follow-up SF | ~~Astrometric-only MF; SF_followup ≡ 1~~ **Superseded (Ryan 2026-11-06)**: the Gaia-only first result keeps SF_followup ≡ 1, but the follow-up evidence layer (SED incl. GALEX UV, RV) is built immediately after it, with a forward-modelled follow-up selection function — see the follow-up evidence layer section (§11, separate PR) |
| R5-Q7 | WD branch fractions (A_ni, R_mt, f_He) | ~~Fixed at the rung-3 fit~~ **Superseded (Ryan, 2026-11-07; #630): free nuisances with broad priors (§2.6)** |
| R5-Q8 | Binned vs unbinned for the first result | **Binned** (contamination self-consistent, MC-robust); unbinned as the gate-5f cross-check |
| R5-Q9 | Dark P/e law | **Rung-4 θ_CO** (log-uniform P, uniform e below e_max); kick-informed e as model comparison |
| R5-Q10 | Second dark generation | **Only if gate 5d fails** in the dark × A > 0.5 cells after gen 1008 |
| R5-Q11 | Companion-nature ΔBIC evidence in the first result | **No**; WD vs dark from the forward-modelled WD light; ΔBIC later (#207, SED throughput) |
| R5-Q12 | DR4 orbit source | **Both** (a) NSS and (b) epoch refit, compared on the overlap (ARCHITECTURE §6); (b) is the inference default once validated |

## 11. Follow-up evidence layer: SED (GALEX UV) and RV (Ryan, 2026-11-06; supersedes R5-Q6) — ACCEPTED 2026-11-06

**Direction (Ryan, 2026-11-06).** Get through all the Gaia data as fast as possible, then **immediately** add SED fitting
(especially GALEX UV) and RV fitting. Follow-up is the best way to measure (i) the true spurious-orbit rate in the mass
range we care about (NS/BH, M2 ≳ 1 M⊙) and (ii) the WD contamination of that range. The Gaia-only first result (§9,
R5-T11) is unchanged and stays on the critical path; this layer starts right after it, and its light parts are prepared
in parallel now.

### 11.1 What the layer measures

| Question | Evidence | Where it enters |
|---|---|---|
| Is the companion a (hot) WD, a luminous star, or dark? | SED: `dark-hunter_sed` Path-2 `1star` / `wd` / `2star` fits with GALEX FUV/NUV + optical/IR; a UV-excess statistic | Per-object class evidence (§11.4a) |
| Is the astrometric orbit real? | RV: Gaia DR3 RV statistics vs the K1 the astrometric orbit predicts; external survey RVs; our own spectra, fitted with The Joker; joint astrometry + RV (`rv_consistency`) | Spurious labels in the high-M2 range (§11.4c); per-object orbit likelihood |
| How many WDs sit in the NS range? | Both: hot WDs show a UV excess; RV confirms the mass function | WD/NS contamination (§2.4) with measured, not only modelled, rates |

### 11.2 SED channel

**Existing interface (CLAUDE.md "dark-hunter_sed → pop").**
- `phot_sed_adapter` reads `output/phot_sed/Gaia_DR3_<id>_<model>_summary.json` for `1star` (dark), `wd` (WD) and `2star`
  (coeval luminous → `other`), recovers χ² from BIC, and tags each candidate `phot_sed` or `analytic_fallback`.
  The WD leg is wired (#206 closed). **#207 is open**: no summary records the photometry-cleaning settings, so ΔBIC
  sensitivity to bad points cannot be measured.
- The console scripts (`darkhunter-sed-phot`, `-gather-phot`, `-batch`) are installed in pop's `.venv`.
- `dark-hunter_sed/photometry_gather.py` already queries GALEX FUV/NUV, but per star, live from MAST, by a cone that keeps the
  **first row** in the radius (no proper-motion propagation, no nearest-match or ambiguity check). For candidates that matter,
  a wrong UV match is a false WD. Fix upstream (FL-Q8) or feed pop-side matched photometry.
- Few Path-2 fits exist; throughput (uberMS SVI plus three dynesty runs per star) is the known blocker and is deferred to
  ziggy.

**Laptop-feasible scope.** Path-2 photometric fits only (no uberMS spectra), for the **candidate set**: real C1 orbits with
posterior-median M2_app ≥ 1.0 M⊙, plus those with A > 0.5 and M2_app ≥ 0.6 M⊙ (FL-Q2). The size is measured by FL-T1 once
R5-T2 gives M2_app for real rows (expected 10²–10³). FL-T5 measures the CPU cost per star on 5 stars before any batch. If
the candidate set × three models fits in one 10-h laptop batch it runs here, behind the generation queue; otherwise the
remainder waits for ziggy.

**Hot-WD UV-excess discriminant.**
- Statistic: the GALEX NUV (and FUV where detected) residual against the prediction for the primary alone (MIST
  isochrone at the row's M1 / age / [Fe/H], extinction from the configured map with R_NUV, R_FUV), in units of the
  combined error: Δ_UV = (m_obs − m_pred,1) / σ.
- It is **one-sided**. A significant excess identifies a hot WD; no excess only excludes WDs younger than a detectability
  age t_det(M_WD, primary, d, A_V). Most WD companions are old and UV-dark, so the class information comes from the
  forward model of P(excess | WD, M_WD, t_cool, x), not from the absence of a detection.
- The Path-2 ΔBIC (`wd` vs `1star`) is the per-object evidence for real candidates; Δ_UV is the cheap statistic computed
  identically on mocks and real rows.
- Known confounders, modelled or flagged: chromospheric UV of active M/K dwarfs; hot subdwarfs and blue stragglers;
  extinction near the plane; GALEX non-linearity and saturation of bright sources (NUV ≲ 15).

**GALEX catalogue choice (FL-Q1).**
- Photometry for real rows: the Gaia archive cross-match `gaiadr3.galex_ais_best_neighbour` → `gaiadr3.galex_ais_dr3`
  (already configured as `dr3.external_photometry_crossmatches[NUV]`, `enabled: false`). It gives epoch-propagated
  best-neighbour matches. FUV needs one more column (`fuv_mag`, `fuv_mag_err`) in that entry.
- Coverage and depth (needed by the forward model): GUVcat_AIS (Bianchi et al. 2017; unique AIS sources from GR6+7, with
  the footprint), and GALEX GR5 AIS/MIS (Bianchi et al. 2011) where the deeper MIS exposures exist.
  AIS covers most of the sky away from the Galactic plane and the Magellanic Clouds, to roughly NUV ≈ 20.8 and
  FUV ≈ 19.9 (AB); MIS is about 2 mag deeper over a much smaller area.
- Follow-up selection term for the forward model: SF_UV(x) = P(GALEX coverage | l, b) × P(detected | m_pred,UV, local
  depth, background), with the coverage from the footprint and the detection curve measured from GUVcat vs Gaia
  completeness per tile depth. This is a property of the sky and the predicted magnitudes, never of the class.

**Mock side.** The mock needs predicted GALEX FUV/NUV for the primary (MIST bolometric corrections for GALEX) and for the
WD companion (the Montreal synthetic photometry of the Bédard et al. 2020 sequences; FL-T2 checks that the snapshot carries
GALEX columns and adds them if not). This is a reweight-free post-processing of stored truth: no re-simulation.

### 11.3 RV channel

**Existing interface.** `dark-hunter_rv` (The Joker) writes `Gaia_DR3_<id>_summary.json` (schema v1), ingested by
`rv_adapter`; `rv_consistency` runs the χ²/dof gate and the joint astrometry + RV fit. 154 summaries are snapshotted under
`data/dr3/rv_summaries/`. These are targeted candidates (our own follow-up), so their selection is the targeting rule
(§11.4b).

**Gaia DR3 RV statistics.** `rv_amplitude_robust`, `rv_chisq_pvalue`, `rv_renormalised_gof`, `rv_nb_transits`,
`rv_template_teff` are published for bright RVS stars. A bright-parent + bright-host snapshot exists
(`data/external_catalogs/20261022T181049Z_dr3_rv_stats_bright`, `scripts/fetch_rv_stats_391.py`); it has to be extended to the
candidate set (download, FL-Q-data list).
- Test: the astrometric orbit predicts K1 = 29.78 (a0/ϖ) sin i / (P_yr √(1 − e²)) km/s (exact for a dark companion; a
  lower bound otherwise). Given the RVS transit times and the RV noise, it predicts the distribution of
  rv_amplitude_robust and of rv_chisq_pvalue.
  **Confirmed:** amplitude and variability consistent with K1. **Refuted:** no variability where K1 should be detected
  with high probability (spurious orbit or wrong a0). **Uninformative:** K1 below the RV detectability.
- This is the physical version of the st1 model that §12.41 deferred: RVS epochs on the same scanning law (a subset of
  the astrometric transits) and an RV noise model σ_RV(G_RVS, T_eff) calibrated on looks-single bright parent rows in the
  existing snapshot. Forward-modelled on mocks; a P-value threshold alone is not used (FL-Q6).

**External surveys.** APOGEE DR17 (allVisit epochs), LAMOST (LRS and MRS epochs), GALAH DR4, and others as available.
Multi-epoch RVs go through The Joker jointly with the astrometric orbit; single-epoch RVs contribute a systemic-velocity
residual term. Survey footprints and targeting functions are the availability selection (`config/target_lists/derived/
survey_sfs/`); every catalogue is snapshotted with a checksum, never read live.

**DR4.** DR4 publishes epoch RVs for the RVS sample. For bright candidates the RV term then becomes an exact per-object
likelihood in a joint epoch-astrometry + epoch-RV fit (ARCHITECTURE §6 "RV epochs"), on the same scanning law as the
mock, so availability is uniform and fully forward-modelled. External surveys remain the route for fainter stars. The
DR3 summary-statistic test is retired for DR4 rows.

**Spurious determination.** In the high-M2 range, each candidate with informative RV is labelled RV-confirmed or
RV-refuted. The labels:
- extend the shared `spuriousness_model` training set (alongside the 293 published labels), with unlabelled candidates
  **censored, not missing at random**: the censoring model is the RV-availability selection (§11.4b);
- calibrate the §12.16 spurious template's normalisation **in the high-M2 cells** (a separate N_s prior there, from the
  refuted fraction with its binomial error), replacing the extrapolation from the bulk π̂ fit.

### 11.4 How follow-up evidence enters inference

**(a) Per-object class evidence.** For a candidate with follow-up data D_i, its contribution to the likelihood gains the
factor Σ_k λ_k(x_i | θ) p(D_i | k, x_i) instead of Σ_k λ_k(x_i | θ). p(D_i | k, x) are **likelihoods**, not
posterior class weights (the population model supplies the class rates; using `companion_nature` posterior weights
would count the prior twice). `companion_nature`'s ΔBIC / photometry channel is the source of p(D | WD), p(D | other),
p(D | dark); the `phot_sed` vs `analytic_fallback` tag is carried per object. In the binned first-result likelihood
this becomes a split of each observable bin by follow-up outcome (UV excess yes/no/unobserved; RV confirmed/refuted/
uninformative/none), with the outcome probabilities per class from the mock.

**(b) Follow-up selection function (replaces "SF_followup ≡ 1").**
- **Availability** (GALEX coverage and detection, Gaia RVS coverage, survey footprints) depends only on observables
  (sky, magnitudes, colour). Conditioning on it is ignorable for the mass function **provided** the outcome
  probabilities are modelled per class. Default (FL-Q4): every object stays in the likelihood; the follow-up factor is
  applied only where data exist; availability enters the mock as SF_FU(x) so that the mock's outcome split matches the
  real one bin by bin.
- **Our own targeted spectroscopy** (dark-hunter_rv targets) is selected on Gaia-derived quantities (M2_app, a0, the
  target-list adoption dates). Its targeting rule must be recorded as a function of observables at adoption time
  (`config/target_lists/derived/`), and is then forward-modelled exactly like a survey footprint. A target with no
  recorded rule is used for validation only, never in the likelihood.
- `selection_function_followup` (#351) is rebuilt on this definition; its synthetic catalog and twin are retired.

**(c) Spurious calibration** in the mass range of interest: §11.3, via the shared model and the high-M2 N_s prior.

**Forward-modelled on mocks vs applied to real candidates only.**

| Forward-modelled on mocks (both sides identical) | Real candidates only |
|---|---|
| GALEX predicted FUV/NUV (primary + WD companion), coverage, detection, Δ_UV outcome per class | Path-2 SED fits and their ΔBIC; spectroscopic (uberMS) fits |
| Gaia DR3 RV statistics under the RVS epoch and noise model; RV detectability of K1 | The Joker fits of real RV epochs; joint astrometry + RV fits |
| Survey and targeting availability SF_FU(x) | Per-object confirmed/refuted labels |
| Outcome probabilities p(outcome | class, x) per bin | The ΔBIC → class reliability, calibrated on Path-2 fits of synthetic photometry of a mock subset (ziggy) |

### 11.5 Sequencing

1. **Critical path unchanged:** Gaia-only first DR3 result (R5-T1…T11).
2. **Now, in parallel (light, laptop, no downloads):**
   - FL-T1 candidate-set definition from the mock-fixed thresholds (needs R5-T2 for real M2_app);
   - FL-T2 GALEX predicted magnitudes on mock truth (MIST GALEX BCs, Bédard GALEX columns check);
   - FL-T3 K1-vs-RV-statistics forward model and the σ_RV(G_RVS, T_eff) calibration on the existing bright RV-stats snapshot;
   - FL-T4 targeting-rule audit of the 154 RV summaries against the adoption dates;
   - FL-T5 the data-request list below, plus a 5-star Path-2 cost measurement (no new data: stars already with photometry).
3. **Needs Ryan's approval to download** (listed, not fetched):
   - Gaia archive TAP: `gaiadr3.galex_ais_best_neighbour` + `galex_ais_dr3` (FUV, NUV, errors, flags) for the candidate set and
     for a looks-single control sample of the parent (small; one query each);
   - GUVcat_AIS (Bianchi et al. 2017) cone extracts for the candidates + the AIS footprint/coverage map; GALEX GR5 MIS
     (Bianchi et al. 2011) where MIS covers candidates;
   - Gaia DR3 RV statistics for the candidate set (extends the existing bright snapshot);
   - APOGEE DR17 allStar/allVisit, LAMOST LRS/MRS epoch tables, GALAH DR4 — candidate cross-matches only, plus their
     footprint/targeting files for the survey selection.
4. **Needs ziggy (deferred):** Path-2 fits beyond the laptop batch; uberMS spectra; the mock-subset ΔBIC calibration.
5. **Immediately after the first result:** FL-T6 wire per-object evidence into `companion_nature` → `inference` (§11.4a);
   FL-T7 high-M2 spurious calibration (§11.3); FL-T8 rebuild `selection_function_followup` (§11.4b); FL-T9 second DR3
   result with the follow-up layer on, compared with the Gaia-only result.

### 11.6 Questions FL-Q1–Q9: **decided by Ryan 2026-11-06** (the defaults in bold are adopted; the §11.5 downloads are approved)

| Q | Question | Recommended default |
|---|---|---|
| FL-Q1 | GALEX source | **Gaia-archive `galex_ais_best_neighbour` for photometry; GUVcat_AIS for coverage/depth; GR5 MIS where available** |
| FL-Q2 | Candidate set for SED/RV | **M2_app ≥ 1.0 M⊙, or A > 0.5 with M2_app ≥ 0.6 M⊙** (all C1 orbits get the cheap GALEX match) |
| FL-Q3 | Laptop Path-2 scope | **Candidate set only, if the measured cost fits one 10-h batch behind the generation queue; rest on ziggy** |
| FL-Q4 | Follow-up selection formulation | **Keep every object; follow-up factor where data exist; availability forward-modelled; own targeting by recorded rules only** |
| FL-Q5 | External RV surveys first | **APOGEE DR17, LAMOST LRS/MRS, GALAH DR4** |
| FL-Q6 | Gaia DR3 RV-statistic test | **Forward-modelled (RVS epochs + σ_RV calibrated on looks-single bright rows)**, not a fixed p-value cut |
| FL-Q7 | Where RV labels go | **Both: shared `spuriousness_model` (censored by RV availability) and a high-M2 N_s prior** |
| FL-Q8 | dark-hunter_sed GALEX matching | **Upstream fix (epoch-propagated nearest match with an ambiguity flag); hand-merged in that repo**; until then pop supplies the Gaia-archive match |
| FL-Q9 | DR4 epoch RVs | **Replace the DR3 summary-statistic test for bright candidates with a joint epoch-astrometry + epoch-RV fit** |

## References

Andrews et al. (2022); Cummings et al. (2018); El-Badry et al. (2023a, 2024, 2026); Janssens et al. (2022);
Kroupa (2001); Moe & Di Stefano (2017); Bianchi et al. (2011, 2017); Shahaf et al. (2023b); Tauris & Savonije (1999); Bédard et al. (2020).
