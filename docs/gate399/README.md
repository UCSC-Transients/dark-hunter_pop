# Gate 399: long-period acceleration capture (#399) and the σ deficit (#398)

Issues #399 and #398, both found by the #390 injection test (`docs/gate390/README.md`). This
report diagnoses them. **Nothing here changes the forward model.** The options at the end are
for Ryan to choose from; none has been chosen.

## Summary

| | finding | class |
|---|---|---|
| #399 decision rule | gaiamock's cascade applies DR3's documented acceleration-vs-orbit rule exactly | not a defect |
| #399 pop side | no bug: the replay reproduces 6,480/6,480 stored outcomes; our cuts act only on the orbit branch | (i) none |
| #399 noise level | correcting the #398 σ deficit lowers P > 600 d capture from 0.23 to 0.19-0.21 | (ii), small |
| #399 truth uncertainty | drawing the injected truth from the published covariance lowers it to 0.21-0.22 | (iii), small |
| #399 residual | ~0.19-0.21 remains; per-system capture is nearly deterministic for 74 of 515 systems | undecided: (ii) or (iii) |
| #398 epoch count | gaiamock has 13% more CCD observations than DR3 used (×√1.13 on σ) | (ii) |
| #398 bright-star excess | DR3's fit-residual inflation c is 1.22-1.34 × gaiamock's at G < 13, 1.00 above | (ii) |
| #398 after both terms | recovered/published σ = 1.02 (ϖ), 1.02 (a0), 0.99 (P), 1.02 (e) | explained |

## Method

`src/darkhunter_pop/cascade_replay.py` and `scripts/diagnose_cascade_399.py`:

- **Bit-for-bit replay.** Every #390 realization is seeded at the call boundary (#371). Calling
  `gaiamock_mod.predict_astrometry_binary_in_terms_of_a0` again with the same seeds gives the same
  epochs. A second call with the same seeds and `parallax = pmra = pmdec = a0 = 0` gives the same
  transits and noise draws with no signal, so the realized noise and the noise-free signal are
  separated without touching gaiamock. The 9/7-parameter checks (`check_9par`, `check_7par`)
  then reproduce the stored branch for **6,480/6,480** realizations. With the acceleration branch
  forced off (`skip_acceleration=True`), the 20 control realizations that reached the orbit in
  #390 reproduce the stored orbit **exactly** (P and a0 equal to the last bit).
- **Noise rescaling.** Every cascade statistic (s, F2, a0/σa0, ϖ/σϖ) is unchanged when the
  observations and their stated errors are scaled together (tested). So "noise × k, errors × k"
  is exactly the same truth observed with k times the noise, and k < 1 is higher S/N.
- **Tests run** (all on the stored seeds, Orbital and AstroSpectroSB1):

| test | what it changes | purpose |
|---|---|---|
| baseline | nothing | #390 numbers; records the discarded 9/7-par statistics |
| noise × 1.11, errors × 1.11 | per-epoch noise, both actual and stated | (a) the #398 σ deficit |
| noise × 1.11, errors × 1.07 | also matches #398's F2 offset | (a) |
| noise × 1.25, × 0.71, × 0.5 | bracket; × 0.5 is the (c) higher-S/N test | (a), (c) |
| DR3-matched | noise × k_N k_c, errors × k_N with k_N = √(N_mock / N_DR3) per system and k_c(G) from #398 | (a) with the measured #398 terms |
| N epochs only | noise and errors × k_N | (a) |
| orbit forced | `skip_acceleration=True` on the same data, 300 captured P > 600 d realizations (+20 controls) | (b) |
| truth drawn from published covariance | 25 truths per system from the full NSS covariance (never a diagonal), one noise draw each | (c) published orbit ≠ true orbit |
| FOV-correlated noise | the same per-CCD variance, a fraction f common to each FOV transit | noise *structure*, f = 0.1, 0.25, 0.5 |

Long-period systems (P > 600 d) also get 20 extra seeded realizations (25 in all) so that
per-system capture probabilities can be measured. All cheap tests cost ~20 ms per realization;
the forced orbit fits cost ~11 CPU s each. Every run used at most 2 worker processes.

## #399: why long-period published orbits come back as accelerations

### 1. The decision rule is DR3's

| | DR3 (Halbwachs et al. 2023) | El-Badry et al. (2024) | `gaiamock_mod` |
|---|---|---|---|
| order | 9-par first, then 7-par, then orbit (§2, §4.2: "the variable acceleration model was tried first") | §4.2, Fig. 2 | `check_9par` then `check_7par` then the orbit |
| direct acceptance | s > 12 and F2 < 25 (§2.2.2) | Eqs. 12-13 | same |
| parallax condition | ϖ/σϖ > 2.1 s^1.05 (9-par), 1.2 s^1.05 (7-par) (§4.3, item 1) | Eqs. 12-13 | same |
| after acceptance | "no other models were tried" (§4.2) | "none of the initially accepted acceleration solutions were fit with orbital solutions" (§4.2) | returns without fitting the orbit |
| σ correction | parameter σ × c, c from F2 (Eq. 2); s uses corrected σ (Eq. 3) | Eqs. 9-11 | same (`cc`) |
| basis | acceleration terms re-centred with ΔT (Eqs. 6-7) | footnote 5: no effect on s or F2 | plain t²/2, t³/6 |

The DR3 re-centring adds lower-order terms to the acceleration basis functions. The model's
column space is unchanged, so the top-order coefficients, their errors, s, F2 and the parallax
and its error are identical. DR3 also rejected outlier CCD transits in the linear fits
(Halbwachs §3.1). That lowers χ², which makes acceptance more likely, not less, so it cannot
explain an excess of captures in the mock. Halbwachs §4.2 calls capturing short-period binaries
with acceleration solutions an intended consequence. El-Badry §5.3 lists "less complete phase
coverage, which causes many long-period sources to receive acceleration solutions" as part of the
DR3 selection function.

**Conclusion (b):** the rule is not the difference. Forcing the orbit on the same data shows
what the rule costs: of the FORCED_N captured P > 600 d realizations, **FORCED_PASS** give an
orbit that passes every Eq. 18-22 cut (median a0/σa0 FORCED_S, F2 FORCED_F2, |ΔP|/P FORCED_DP;
`figures/forced_orbit.png`). Had the cascade tried the orbit, most would have been accepted
orbits, as DR3 published them.

### 2. What decides it: the parallax criterion

Orbital, P > 600 d, 5 stored realizations each (2,575; `figures/acceleration_statistics.png`):

- 23.1% captured; 64% of captures are 9-parameter.
- Captured 9-par: s9 = 16.0 (13.1-24.2), F2 = 9.3 (3.1-18.9), and
  (ϖ/σϖ) / (2.1 s9^1.05) = **1.19 (1.06-1.60)**, so just over the parallax threshold.
- Realizations that went on to the orbit failed the 9-par test on s (23%), on F2 with s passing
  (29%), or **on the parallax condition alone (48%)**.

The parallax ratio is (ϖ/σϖ) / s^1.05. Both terms scale as 1/σ, so the ratio scales as σ^0.05:
**the decision is almost independent of the noise level** and is set by the geometry (ϖ against
the photocentre orbit, P, e, phase coverage). That is why every noise rescaling below moves the
capture rate so little.

Capture against coverage (`figures/capture_vs_coverage.png`): it peaks where the ~957 d span
covers 1.0-1.3 orbits (0.27) and falls to 0.18 above 1.3 orbits. It rises with e (0.12 for
e < 0.1, 0.28 for 0.3-0.5). It is highest just above the orbit significance threshold (0.31-0.40
for published s / threshold < 2) and lowest at high significance (0.13 above 4). A periastron
falls inside the window in 99.8% of these realizations.

### 3. The tests

Accepted 7/9-parameter fraction, Orbital (`figures/capture_vs_period.png`):

| test | < 300 d | 300-600 d | 600-1000 d | > 1000 d |
|---|---|---|---|---|
| baseline (#390) | 0.006 | 0.053 | **0.230** | **0.240** |
| noise × 1.11, errors × 1.11 | 0.010 | 0.044 | 0.223 | 0.237 |
| noise × 1.11, errors × 1.07 | 0.008 | 0.042 | 0.217 | 0.234 |
| noise × 1.25 | 0.010 | 0.049 | 0.218 | 0.268 |
| N epochs only | 0.008 | 0.050 | 0.215 | 0.231 |
| **DR3-matched** (N + bright-star excess) | 0.008 | 0.046 | **0.193** | **0.212** |
| truth drawn from published covariance | | | 0.208 | 0.223 |
| noise × 0.71 (higher S/N) | 0.010 | 0.050 | 0.220 | 0.197 |
| noise × 0.5 (higher S/N) | 0.001 | 0.037 | 0.168 | 0.188 |
| FOV-correlated noise, f = 0.5 | | | 0.247 | 0.240 |

N: 785 / 1,770 / 2,250 / 325 realizations (truth draws: 11,100 / 1,575). For
AstroSpectroSB1 the DR3-matched noise matters more (0.120 → 0.061 at 600-1000 d, 0.243 → 0.165
above 1000 d), because that sample is brighter (G < 13 in most systems).

- **(a) Noise level, (ii):** the measured #398 terms remove **3.7 pp** (600-1000 d) and **2.8 pp**
  (> 1000 d) of the 23-24%; a flat 1.11 removes < 1 pp.
- **(c) Truth uncertainty, (iii):** the injected truth is the published orbit, not the true
  one. Drawing it from the full published covariance removes **2.2 pp** and **1.7 pp**.
- **(c) Higher S/N:** capture falls only to 0.17-0.19 at half the noise. It is not a near-threshold
  noise effect.

### 4. Is the residual winner's curse?

The published sample is conditioned on DR3 *not* capturing these systems. On a fresh noise draw
a correct forward model should capture each system i with its own probability p_i, so some
re-injection capture is expected. How much depends on how many similar binaries DR3 did capture,
and those are not in this sample.

Per-system capture probability over 25 realizations (`figures/capture_probability.png`; 515
Orbital systems, P > 600 d):

| | mean p | p ≤ 0.1 | p ≥ 0.9 | p = 1 |
|---|---|---|---|---|
| fixed published truth | 0.229 | 347 | **74** | 63 |
| truth drawn from published covariance | 0.210 | 329 | 49 | 27 |
| FOV-correlated noise f = 0.5 | 0.239 | 299 | 49 | 34 |

The distribution is bimodal, and **62% of all captures come from systems captured in ≥ 90% of
realizations**. Under gaiamock's noise, a Gaussian extrapolation of each condition's margin over
25 draws gives a median probability of ~4 × 10⁻⁴ that DR3 would have fitted an orbit for these
systems (58% below 10⁻³). If the forward model were right, each published system like this would
need on the order of 10³ similar binaries captured by DR3. That is more than DR3's whole
provisional-acceleration population can supply (Halbwachs §4.3: 0.81 M + 0.57 M solutions
including alternatives). So **winner's curse alone is unlikely for the near-deterministic
systems**. It is plausible for the other 38% of captures, from systems with intermediate p.

Two mechanisms widen the per-system spread and so make winner's curse more plausible:

- **Truth uncertainty** (above) brings the p ≥ 0.9 count from 74 to 49.
- **Correlated noise within a FOV transit** (not in gaiamock_mod, whose CCD measurements are
  independent) brings it to 49 at f = 0.5, while the mean capture stays at 0.24. El-Badry et al.
  (2024) Appendix C argue that the 9 CCDs of a transit are "nearly independent", because their
  single-star σϖ match DR3's, so a large f is not favoured.

**Classification of the residual: undecided between (ii) and (iii).** Re-injecting a selected
sample cannot settle it. The decisive test is population-level: the mock's long-period
orbit : acceleration ratio against DR3's, which needs the `nss_acceleration_astro` snapshot (spec
Q7 / Q7b, #341). The #391 full run already contains everything the mock side of that comparison
needs.

## #398: why recovered σ are ~11% below DR3's

Accepted Orbital realizations, 3,804 of them (`figures/sigma_ratio_decomposition.png`,
`figures/c_factor_vs_g.png`). A parameter σ is (stated per-epoch error / √N) × geometry × c, so
the ratio splits into an epoch-count term and an inflation-factor term:

| | σ_ϖ | σ_a0 | σ_P | σ_e |
|---|---|---|---|---|
| raw recovered / published | 0.892 | 0.893 | 0.862 | 0.882 |
| × √(N_mock / N_DR3) | 0.946 | 0.956 | 0.916 | 0.941 |
| × √(N_mock / N_DR3) × c_DR3 / c_mock | **1.023** | **1.024** | **0.988** | **1.023** |

- **Epoch count (ii).** gaiamock's realizations have a median **13%** more CCD observations than
  DR3's `astrometric_n_good_obs_al` for the same source (16th-84th 0.97-1.32). The excess is in FOV
  transits: GOST predicts ~11% more transits than DR3's `astrometric_matched_transits`, while the
  CCDs per transit agree (gaiamock 9.8 rows minus its 10% rejection ≈ 8.8; DR3 8.7). These are
  transits DR3 lost to gaps, dead time and rejection, which GOST does not know about. This is the
  same effect as #400's +2 visibility periods. El-Badry et al. (2024) §3.3.1 instead bin to an
  effective N_bin = 8 per transit, which absorbs part of it.
- **Bright-star excess noise (ii).** c = √(χ²/ν / (1 − 2/9ν)³) (Halbwachs Eq. 2), from the
  published F2 (ν from `astrometric_n_good_obs_al` − 12) and the recovered F2 (ν = N_mock − 12):

| G | N_mock/N_DR3 | c_DR3 | c_mock | c ratio | published F2 | recovered F2 | σ_ϖ ratio: raw → corrected |
|---|---|---|---|---|---|---|---|
| < 11 | 1.21 | 1.34 | 1.02 | 1.33 | 8.5 | 0.6 | 0.72 → 1.04 |
| 11-12 | 1.12 | 1.34 | 1.01 | 1.34 | 8.8 | 0.1 | 0.73 → 1.05 |
| 12-13 | 1.12 | 1.22 | 1.01 | 1.22 | 5.7 | 0.2 | 0.82 → 1.06 |
| 13-14 | 1.14 | 1.00 | 1.00 | 1.00 | 0.0 | 0.0 | 0.94 → 1.02 |
| 14-15 | 1.14 | 1.01 | 1.01 | 1.00 | 0.2 | 0.3 | 0.95 → 1.01 |
| > 15 | 1.11 | 1.02 | 1.00 | 1.02 | 0.4 | 0.1 | 0.92 → 0.97 |

  DR3's orbital fits at G < 13 have residuals ~1.2-1.35 × their stated errors; gaiamock's have
  ~1.0. This is the G < 13 F2 discontinuity El-Badry et al. (2024) model with their unmodeled
  per-FOV noise σ ~ U(0, 0.04) mas (§3.3.1). `gaiamock_mod` has neither that term nor the 0.5 mas
  marginally-resolved term (spec §3). As an equivalent per-FOV excess at G ≈ 11, a c ratio of 1.33
  needs ~0.038 mas per transit, against an rms of 0.023 mas for the paper's U(0, 0.04). The order
  of magnitude matches.
- **Other covariates:** after both terms there is no remaining trend with P, published a0/σa0 or
  |β| (`figures/sigma_ratio_decomposition.png`).
- **DR3's own inflation:** the only uncertainty inflation in DR3's NSS orbits is the c factor
  (Halbwachs §2.2.1). gaiamock applies the same factor (`get_uncertainties_at_best_fit_binary_solution`),
  and its per-CCD errors already use Holl et al. (2023a) "EDR3 adjusted", the inflated per-epoch
  errors the NSS pipeline used. So no separate DR3 inflation is missing beyond what c captures.
- **El-Badry et al. (2024) calibration:** they validated their per-FOV errors against DR3
  5-parameter σϖ for single stars (Appendix C, Fig. 19). Binary fits were not calibrated
  separately.

**Classification:** (ii), gaiamock_mod differs from DR3 in epoch count and bright-star excess
noise. Both are fully quantified, and together they close the σ deficit to ≤ 2.4% in median.

## Options for Ryan (none chosen)

**#399**

| option | what | #391 stored outputs |
|---|---|---|
| A. Accept as DR3's rule | keep the cascade; compare the mock to DR3 with accelerations included (Q7b snapshot of `nss_acceleration_astro`) | **reusable as is**: each draw stores the cascade vector (s, F2, ϖ, σϖ of an accepted acceleration) and `published_acceleration` |
| B. Bound it with a pop-side post-cascade rule | e.g. fit the orbit for accepted accelerations and keep it when it passes Eqs. 18-22. Not DR3's rule, so only as a sensitivity bound | **partial re-simulation**: only the acceleration-outcome draws need the orbit fit. They replay exactly from the stored seeds (`run_full_astrometric_cascade(..., skip_acceleration=True)` under `seeded_global_rng`; the same replay reproduced 20/20 orbits here) at ~11 CPU s each. Any rule that reads the 9-par statistics of a 7-par winner also needs the cheap linear replay, because the vector stores only the winner's statistics |
| C. Correct the noise first (see #398 options) | removes ~3-4 pp of capture | **full re-simulation** (every draw's epochs change) |
| D. Upstream gaiamock_mod change | thinning to DR3's transit count and the bright-star term, inside gaiamock_mod | **full re-simulation** |
| E. Decide after the population-level test | run Q7b first; it alone separates (ii) from (iii) | reusable (the test uses the stored outcomes) |

**#398**

| option | what | #391 stored outputs |
|---|---|---|
| 1. Leave as a documented systematic | σ ~11% optimistic; near-threshold detection slightly too easy | reusable as is |
| 2. Pop-side epoch thinning | drop simulated transits to DR3's count per source, wrapping gaiamock's `predict_*` and `fit_full_astrometric_cascade` (no reimplementation). Needs a per-position transit-loss model | full re-simulation |
| 3. Pop-side bright-star excess noise | add per-FOV noise for G < 13, as the paper does (§3.3.1), or calibrated to the c ratio above, again outside gaiamock | full re-simulation |
| 4. Upstream gaiamock_mod change | the same as 2-3 inside gaiamock_mod (new release, version triple bump) | full re-simulation |

Any option that changes the epoch data (B's orbit fits excepted) invalidates every stored #391
outcome, because each draw's cascade sees different epochs. Options A, E and 1 need no
re-simulation. Option B re-simulates only the acceleration-outcome draws, from their stored seeds.

## Reproduce

```bash
PY=.venv/bin/python   # PYTHONPATH=<worktree>/src when run from a worktree
$PY scripts/diagnose_cascade_399.py linear --workers 1          # variants + 20 extra long-P realizations
$PY scripts/diagnose_cascade_399.py matched --workers 1         # DR3-matched noise
$PY scripts/diagnose_cascade_399.py truth-draws --workers 1     # truths from the published covariance
$PY scripts/diagnose_cascade_399.py fov-correlated --workers 1  # correlated-noise diagnostic
$PY scripts/diagnose_cascade_399.py orbit --workers 2 --max 300 --n-control 20
$PY scripts/plot_cascade_399.py
```

## Artifacts (gitignored, primary checkout `output/gate399/`)

| file | contents |
|---|---|
| `linear_replay.h5` | 18,880 realizations × 6 noise variants: ruwe, s9, F2_9, ϖ/σϖ (9), s7, F2_7, ϖ/σϖ (7), branch, span, coverage, noise-free misfit |
| `linear_matched.h5` | baseline / DR3-matched / N-only |
| `truth_draws.h5` | 15,275 covariance-drawn truths (9 systems without a rebuildable covariance skipped, never diagonal) |
| `fov_correlated.h5` | f = 0, 0.1, 0.25, 0.5 |
| `forced_orbit.jsonl` | the skipped orbit fits (cascade vectors) + controls |

Input: `output/gate390/injection_test_full.h5`, sha256
`e585c9e69fba221bd00903185e0569001ffbdd62fabab262030310d8df10e6c2` (re-verified). Every number
above is in `figures/summary.json`.
