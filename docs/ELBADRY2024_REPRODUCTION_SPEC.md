# El-Badry et al. (2024) mock catalog and six-panel reproduction — spec

Issue: #339 (Phase 2). Status: **spec for Ryan's approval.** Only the parts of the method that the
paper or the authors' public code determines are marked for implementation. Every choice they do
not determine is listed in §9 as an open question. None of those has been picked.

Source: El-Badry, Lam, Holl, Halbwachs, Rix, Mazeh & Shahaf (2024), *A generative model for Gaia
astrometric orbit catalogs*, OJAp 7, 100, arXiv:2411.00088. The PDF is vendored at
`vendor/gaiamock/paper/generative_model_dr3.pdf` and section numbers below refer to it. Public code:
`kareemelbadry/gaiamock` (vendored submodule `vendor/gaiamock`, commit `dd30fdb`, plus our
`gaiamock_mod` overlay). The paper's population-synthesis code (Galaxia → COSMIC → MIST matching)
is **not** in that repository, and I found no public release of it or of the 46-million-binary
input catalog.

## 1. What the paper's Figure 5 is

- **Panels** (upper left to lower right, §5.1): orbital period (log axis, ~10–5000 d), apparent
  G (linear), distance plotted as 1/ϖ in kpc (linear, 0–2), eccentricity (0–1), astrometric mass
  function f_m,ast = (a0/ϖ)³ (P/yr)⁻² (Eq. 19; log axis, ~10⁻³–1 M⊙), and **signed cos i**
  (−1 to 1). The y axis is the count N.
- **Real sample** (§4, footnote 6): DR3 `nss_two_body_orbit` rows with
  `nss_solution_type = Orbital or AstroSpectroSB1`, which is 168,065 rows (134,598 + 33,467). No
  further cut. "We discard a random 19% of the observed catalog … such that the simulated
  comparison sample has the same number of binaries" (§5). That only rescales the counts, so it
  does not change a shape comparison.
- **Mock sample**: 137,000 simulated orbital solutions passing all cuts (§5).
- **Stated result**: P, G, distance, f_m,ast and inclination are "in reasonably good agreement".
  The mock is **more eccentric** than DR3, which the authors attribute to the input p(e) ∝ e^0.2.
  "The bias against edge-on orbits is reproduced well in the simulated catalog." The mock is 18%
  smaller than DR3.

Phase 1 of #339 makes our pipeline compute exactly this real sample, gates every mock panel on all
cuts, uses Eq. 19 for mock f_m, and plots these quantities on these axes.

## 2. Population model (paper §3) — what it specifies

| Item | El-Badry et al. (2024) | Determined? |
|---|---|---|
| Galactic model | Galaxia (Sharma et al. 2011), Besançon model, "modified version" of Lam et al. (2020); sources within **2 kpc** | Partly: the modified Galaxia's parameter file is not given (Q1) |
| Normalisation | Galaxia over-predicts by 1.4 within 50/100 pc → discard a random 1/1.4, independent of distance and magnitude | Yes, but shape-neutral |
| Binary population | COSMIC (Breivik et al. 2020) zero-age population, Moe & Di Stefano (2017) covariant, mass-dependent P, q, e; companions with **0.2 < log(P/d) < 8** | Partly: COSMIC version and sampler settings not given (Q2) |
| Primary IMF | Kroupa (2001), **not** COSMIC's default | Yes |
| Binary fraction | MdS17: ≈41% at solar type, 57% at 3 M⊙, 80% at 6 M⊙. Below 0.8 M⊙ it falls linearly in log M1 from 40% at 0.8 M⊙ to 0 at 0.08 M⊙ | Yes |
| Systems count | generate COSMIC systems until their number exceeds the Galaxia count by 10% (dead stars) | Yes |
| Sky position | each binary placed at the Galaxia star **closest in mass to its primary** | Yes |
| Orientation | i from a sin i distribution; Ω, ω ~ U(0, 2π); φ0 = 2πTp/P ~ U(0, 1) | Yes |
| Photometry | MIST (Choi et al. 2016) via `isochrones` (Morton 2015), age ~ U(0, 12 Gyr) | Partly: metallicity and isochrone grid version not given (Q3) |
| Evolution | drop binaries with initial mass > 8 M⊙ whose primary died; drop systems that fill or would have filled their Roche lobe | Partly: the Roche-lobe test is not specified (Q4) |
| WD companions | most WD binaries shrink and are undetectable; **10%** of WD + star binaries with initial a = 2–6 au survive at **50%** of initial a; WD masses from the Weidemann (2000) IFMR | Yes (prescription), but which "WD + luminous star" systems are eligible needs COSMIC output (Q2) |
| Extinction | `mwdust` Combined19; A_G = 2.8 E(B−V) | Yes; already ours |
| Resolved pairs | remove when ΔG < (1/25)(ρ/mas − 200) (Appendix B, Eq. B1, applied in G) | Yes |
| Magnitude limit | keep unresolved binaries with **G < 19** | Yes |
| Singles, triples, chance alignments | not simulated | Yes |
| N | **46 million** unresolved binaries, G < 19, within 2 kpc | Yes |

## 3. Epoch astrometry and noise (paper §3.3–3.4)

| Item | Paper | Our `gaiamock_mod` |
|---|---|---|
| Scanning law | GOST, healpix level-6 precompute | same function (`get_gost_one_position`) |
| DR3 window | JD 2456891–2457902 | same (`rescale_times_astrometry`) |
| Transit loss | reject 10% of FOV transits | same (`predict_astrometry_luminous_binary`) |
| Per-CCD noise | Holl et al. (2023a) "EDR3 adjusted"; measurements **binned** to FOV transits (÷√8) with χ² corrected by Eq. 5 | **unbinned CCD-level** plus an empirical sky-dependent rescaling (`get_realistic_epoch_astrometry_errors`) |
| Extra bright noise | G < 13: add σ ~ U(0, 0.04) mas | not in `gaiamock_mod` |
| Marginally resolved | add 0.5 mas noise to epochs with ξ > 0.5; blended photocenter via Eq. 1 with u = 90 mas | photocenter bias via `al_bias_binary`; **no 0.5 mas term** |

So the paper's Fig. 5 was made with the authors' **binned** model of the time. `gaiamock_mod` is a
later variant that the README describes as better for small orbits and single stars. Which one a
faithful reproduction should use is Q5.

## 4. Cascade and cuts (paper §4)

Our `gaiamock_mod.run_full_astrometric_cascade` implements the paper's Fig. 2 cascade:

1. ≥ 12 visibility periods, otherwise no solution.
2. 5-parameter fit; UWE < 1.4 → single-star solution.
3. 9-parameter, then 7-parameter acceleration acceptance (Eqs. 12, 13). An accepted acceleration
   solution is never fit with an orbit.
4. Otherwise a 12-parameter orbital fit.

The selection cuts are applied after the cascade:

- Eq. 18: s_orb = a0/σa0 > 5 **and F2 < 25**.
- Eq. 20: ϖ/σϖ > 20000 d / P.
- Eq. 21: σe < 0.079 ln(P/d) − 0.244.
- Eq. 22: a0/σa0 > 158/√(P/d).

Phase 1 now applies all of these, from `dr3.selection_function_astrometric.orbital_solution_cuts`.
**Discrepancy to note**: the authors' own helper `simulate_many_realizations_of_a_single_binary`
applies Eqs. 18 (s_orb only), 20, 21 and 22 and **omits F2 < 25**. The paper text includes it, and
Phase 1 follows the paper (Q6).

Published acceleration solutions (§5.2.1) need extra cuts: significance s > 20, and for
7-parameter solutions F2 < 22. The paper reports 2.4 × 10⁵ such mock solutions.

## 5. Solution-type fractions

The paper does not show a solution-type-fraction figure. It reports these counts: 46 M input
binaries; 3 × 10⁴ dropped for < 12 visibility periods; 5.3 × 10⁵ removed as provisional
acceleration solutions, 2.4 × 10⁵ of them publishable; 5.4 × 10⁵ orbits passing Eq. 18; 137,000
passing all cuts. A like-for-like real comparison needs the DR3 `nss_acceleration_astro` counts by
type. Those are not in our snapshot (it queries `nss_two_body_orbit` only). It also needs a real
denominator, which DR3 does not publish for "UWE < 1.4 binaries". Our current real fractions put
SB1 and EclipsingBinary in `insufficient_visibility` (#341). What this comparison should be is Q7.
The countable, determined targets are the ratios **orbital : publishable acceleration** (mock
137k : 240k in the paper) against DR3's published counts.

### 5.1 Real-side counts available now (#341, #330)

Uncut snapshot `20260826T234425Z_3d3f740b080c` (`nss_two_body_orbit`, 443,211 rows):

| nss_solution_type | rows |
|---|---|
| SB1 | 181,332 |
| Orbital | 134,598 |
| EclipsingBinary | 86,919 |
| AstroSpectroSB1 | 33,467 |
| SB2 / SB2C / SB1C | 4,630 / 746 / 202 |
| OrbitalAlternative(+Validated) | 619 + 10 |
| OrbitalTargetedSearch(+Validated) | 345 + 188 |
| EclipsingSpectro | 155 |

There are no acceleration rows: they live in `gaiadr3.nss_acceleration_astro`, which we have not
snapshotted. The current `solution_type_fractions` gate divides by all 443k rows and maps every
non-`Orbital*` type, SB1 and EclipsingBinary included, to `insufficient_visibility` (real 0.648).
The mock side meanwhile counts cascade outcomes per simulated binary. The two have **different
denominators and different category sets**, so the gate measures the mapping, not the physics.

**What the paper determines.** Only the orbital class has a like-for-like real sample: Orbital +
AstroSpectroSB1 (§4). The paper compares acceleration solutions only as counts, through the §5.2.1
publication cuts. It makes no comparison of 5-parameter, UWE < 1.4 or insufficient-visibility
fractions, because DR3 publishes no NSS catalog of them. SB1, EclipsingBinary and SB2 have no
gaiamock counterpart.

**Not determined, so questions for Ryan (Q7a–c below):** the real denominator; whether to snapshot
`nss_acceleration_astro`; and what to do with the types gaiamock cannot produce.

### 5.2 Validation statistics (#345, #346)

The paper validates Fig. 5 **by eye**. It reports no KS test, minimum N or tolerance. Our
`validation_gate` (KS p ≥ 0.01, |Δ fraction| ≤ 0.05, multi-solution ≤ 0.05) is our own
construction. Its thresholds, any minimum mock N or power criterion (#345), and whether the
multi-solution check is a gate or a unit test (#346) are **not determined by the paper**
(Q10, Q11). Measured fact for #345: the Phase 1 re-render had 12–20 accepted mocks out of 500, and
with n_mock ≈ 12 a two-sample KS test cannot exclude D ≈ 0.3 at p = 0.01.

## 6. What produces the face-on cos i excess

The paper does not model the inclination distribution explicitly. Inclinations are drawn
isotropically (sin i). The DR3 edge-on deficit appears in the mock purely as a **selection and fit
effect** of the cascade and cuts. The paper does not spell out the mechanism. The geometric
expectation is that an edge-on orbit projects to a line on the sky, so its 1D along-scan signal is
on average smaller, and it vanishes for scans perpendicular to the line. That lowers UWE (fewer
pass UWE > 1.4) and lowers s_orb relative to σa0 (fewer pass Eqs. 18 and 22). §8.2 measures this
directly with `gaiamock_mod`: near the detection threshold, face-on orbits are accepted twice as
often as edge-on ones.

## 7. Our `mock_population` vs the paper, item by item

| Item | Paper | Ours (`selection_function_astrometric.mock_population`, `elbadry_prior`) |
|---|---|---|
| Volume | sphere, d < 2 kpc, Galaxia positions. Our real sample has 2,319 of 168,065 (1.4%) beyond 1/ϖ = 2 kpc, consistent with the paper's "99% within 2 kpc" | `dr3.selection_function_astrometric` 50–1200 pc, `gaiamock.generate_coordinates_at_a_given_distance_exponential_disk` (h_z = 300 pc) |
| Primary mass | Kroupa IMF, matched to a Galaxia star | log-uniform 0.7–2.8 M⊙ |
| Binary fraction | MdS17, mass-dependent | 100% binaries |
| Period | MdS17, 0.2 < log P < 8 (includes wide pairs that make spurious solutions, §5.2) | uniform in 1/P, 60–8500 d |
| Eccentricity | MdS17 (e^0.2 at 100–1000 d) | U(0, 0.65) |
| Mass ratio | MdS17 | M2 log-uniform 0.08–3.5 M⊙, independent of M1 |
| Flux ratio | MIST, from the two masses and the age | log-uniform 10⁻⁴–0.15, independent of the masses |
| Absolute magnitude | MIST | M_G,tot ~ U(3, 6.5), independent of the masses |
| Faint and invisible sources | G < 19 cut; < 12 visibility periods from the scanning law | `faint_draw_fraction` 0.45 of draws use M_G 8–11.5. Before #344 they were tagged `insufficient_visibility` without simulation; since #344 they are simulated. No G cut |
| WD companions | 10% of 2–6 au WD binaries at 0.5 a | none |
| Dead or RLOF systems removed | yes | no |
| Resolved-pair cut (B1) | yes | no |
| Extinction | Combined19, A_G = 2.8 E(B−V) | same |
| Orientation, Tp | isotropic, uniform | same |
| Noise model | binned, + bright-star and ξ > 0.5 terms | `gaiamock_mod` unbinned, sky-dependent |
| Cuts | Eq. 18 (with F2) + Eqs. 20–22 | same since Phase 1 |
| N | 46 M binaries → 137k orbits | 500 → ~13 accepted (2.6%) |

The stand-in ranges were tuned to the synthetic placeholder fixture (see the comment in
`config/fragments/selection_function_astrometric.yaml`). Nothing about them is from the paper.

## 8. Measurements on this laptop

Measured on the #339 Phase 1 code (PR #359, merged). The scripts are in the session scratchpad and are reproduced in the PR description.

- **Per-realization cost** of `gaiamock_mod.run_full_astrometric_cascade`, single thread, by
  branch: see §8.1.
- **Inclination mechanism experiment**: see §8.2.

### 8.1 Cost

Measured on 2026-10-01 with `gaiamock_mod.run_full_astrometric_cascade` (epoch prediction plus
cascade), DR3, single process, **CPU time** (`time.process_time`). The laptop was shared, with a
load average of about 35 on 10 cores, so wall clock would be worse. Script:
`bench_cascade.py`, N = 30 per regime. RSS is 0.17 GB per process.

| Branch reached | Example binary | median CPU s | mean | p90 |
|---|---|---|---|---|
| 5-parameter only (UWE < 1.4) | P = 10 d, d = 1.5 kpc, G = 15 | 0.94 | 0.89 | 1.33 |
| mixed: acceleration / 5-par / orbit | P = 2×10⁴ d, d = 200 pc, G = 11 | 0.59 | 5.8 | 18.8 |
| 12-parameter orbital fit | P = 500 d, d = 300 pc, G = 11.5 | 13.3 | 13.7 | 20.4 |

The 800-realization inclination run (§8.2) gives the same orbital-fit cost: 9.1 s mean, 8.3 s
median, with BLAS pinned to one thread.

**Cost model for the paper's population.** About 1% of binaries reach the orbital fit (paper §4.4),
so the mean is ≈ 0.99 × 0.9 + 0.01 × 13 ≈ **1.0 CPU s per binary**. The paper's binned stock
gaiamock averaged 0.39 CPU s (5k CPU h / 46 M). The paper accepts 137k of 46 M binaries
(**0.30%**).

| Accepted orbits wanted | Binaries to simulate | CPU hours | Wall clock at 8 workers | Disk (≈320 B per realization) |
|---|---|---|---|---|
| 2,000 | 0.67 M | ≈ 190 | ≈ 24 h | ≈ 0.2 GB |
| 5,000 | 1.7 M | ≈ 470 | ≈ 2.5 d | ≈ 0.5 GB |
| 10,000 | 3.4 M | ≈ 940 | ≈ 5 d | ≈ 1.1 GB |

RSS stays far under the 6 GB budget (8 × 0.17 GB plus the parent). Disk is fine against the
3 GiB floor. **Wall clock is the constraint**, and it is shared with the other agents' load.
Which N to run is Q9.

### 8.2 Inclination experiment

Script: `inclination_experiment.py`. A fixed binary, a dark-companion variant of the paper's Fig. 14
fiducial: M1 = 1 M⊙, M2 = 0.7 M⊙, f = 0, e = 0.3, P = 500 d, M_G = 4.67. It is placed at random
sky positions in a ±10% distance shell with isotropic orientation (cos i uniform), run through the
`gaiamock_mod` cascade, and accepted on all six cuts. There is no population model, so this
isolates the selection and fit effect.

| d | N | accepted | P(accept) by \|cos i_true\| bin [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1] |
|---|---|---|---|
| 0.7 kpc (G ≈ 14) | 800 | 624 | 0.76, 0.74, 0.82, 0.83, 0.76 |
| 2.0 kpc (G ≈ 16) | 800 | 0 | all UWE < 1.4 (every realization stays a single star) |
| 1.2 kpc (G ≈ 15) | 800 | 62 | **0.051, 0.066, 0.086, 0.084, 0.102** |

**Finding.** A binary well above threshold (0.7 kpc) is accepted at nearly any inclination:
there is no strong edge-on deficit, and the fitted |cos i| histogram of the accepted systems is
roughly flat (121, 125, 128, 141, 109). **Near threshold (1.2 kpc) the acceptance doubles from
edge-on to face-on (0.051 → 0.102)**, and the fitted |cos i| of the accepted systems is skewed
face-on (6, 12, 10, 19, 15). The cascade and cuts alone therefore produce the face-on excess, but
only for marginal detections, where the smaller projected sky motion of edge-on orbits decides
UWE > 1.4 and s_orb. In the paper's population that is a large share of the detected sample
(the completeness is ≈ 2% within 2 kpc at the peak period, §5.3). So the excess is a
population-weighted threshold effect. A single-binary test can show it only near threshold, and
the full-sample excess needs the paper's distance, magnitude and period mix.

### 8.3 #344 measured (every realization through gaiamock)

With the faint short circuit removed (500 realizations, stand-in priors, real DA artifact of run
20260930-022223-672b092), the mock fractions are:

| Outcome | Fraction |
|---|---|
| insufficient_visibility | **0.000** |
| five_parameter | 0.830 |
| seven_parameter | 0.002 |
| nine_parameter | 0.010 |
| orbital passing cuts | 0.036 (18/500) |
| orbital failing cuts | 0.122 |

With the DR3 scanning law, gaiamock gives ≥ 12 visibility periods everywhere it was sampled. The
real 0.648 "insufficient_visibility" is the #341 mapping (SB1 / EB → that bin), not a physical
fraction. The gate now fails honestly (max |Δ| = 0.83) until Q7 is settled.

### 8.4 #399 / #398: long-period acceleration capture and the σ deficit (measured, not decided)

Full report: `docs/gate399/README.md`. Measured by replaying every #390 realization bit for bit
(6,480/6,480 cascade outcomes reproduced) and recomputing what the cascade discarded.

- **Decision rule.** `gaiamock_mod.fit_full_astrometric_cascade` applies DR3's documented rule
  exactly: 9-parameter model first, then 7-parameter, each accepted on s > 12, F2 < 25 and
  ϖ/σϖ > 2.1 s^1.05 (9-par) or 1.2 s^1.05 (7-par), and no orbit is fitted after an acceptance
  (Halbwachs et al. 2023 §2.2.2, §4.2, §4.3 item 1; this paper Eqs. 12-13 and §4.2). The DR3
  re-parameterisation of the acceleration terms (Halbwachs Eqs. 6-7) leaves s, F2 and ϖ/σϖ
  unchanged (same model column space). No pop-side bug was found.
- **What decides it.** For P > 600 d the deciding test is mostly the parallax criterion. Of the
  realizations that reached the orbit, 48% failed the 9-parameter test on ϖ/σϖ alone; captured
  9-parameter realizations sit at ϖ/σϖ / (2.1 s^1.05) = 1.19 (median). That ratio scales as
  σ^0.05, so the capture rate barely depends on the noise level.
- **Noise (#398) explains little.** Capture at 600-1000 / > 1000 d: baseline 0.230 / 0.240;
  noise × 1.11: 0.223 / 0.237; DR3-matched noise (epoch count + bright-star excess, below):
  0.193 / 0.212; truth drawn from the published covariance: 0.208 / 0.223; noise × 0.5:
  0.168 / 0.188.
- **Skipped orbit.** With the acceleration branch forced off on the same data, 85% (254/300) of
  captured P > 600 d realizations give an orbit that passes every Eq. 18-22 cut, so long-period
  acceptance would be ~0.85 instead of 0.66 without the acceleration branch. That is DR3's rule
  at work, not a gaiamock difference.
- **Residual.** About 0.19-0.21 survives every correction. Per-system capture is bimodal: 74 of
  515 long-period Orbital systems are captured in ≥ 90% of 25 realizations. A re-injection of a
  *selected* sample cannot by itself separate a forward-model defect from the expected
  re-injection capture of a sample conditioned on DR3 not capturing it; that needs the
  population-level orbit : acceleration comparison (Q7 / Q7b, #402).
- **σ deficit (#398) is explained.** Recovered/published σ (median 0.89 for ϖ and a0, 0.86 for P)
  becomes 1.02 / 1.02 / 0.99 after two terms: gaiamock has 13% more CCD observations than DR3's
  `astrometric_n_good_obs_al` (×√1.13), and DR3's goodness-of-fit inflation c (Halbwachs Eq. 2) is
  1.22-1.34 × gaiamock's at G < 13 and 1.00 at G ≥ 13 (the missing bright-star excess noise, §3).

Options are listed in the report and in Q13-Q14 below. None has been chosen.

## 9. Open questions for Ryan (nothing below has been chosen)

- **Q1 Galaxia.** Which Galaxia build and parameter file? Lam et al. (2020) used the PopSyCLE fork,
  but the "modified version" is not specified. Generating all 1.1 × 10⁹ stars within 2 kpc does not
  fit this laptop's ~8 GiB free disk, so it would have to be a random subsample (shape-neutral) or
  run elsewhere.
- **Q2 COSMIC.** Version, the `independent` sampler with `multidim` (MdS17) settings, how the
  Kroupa IMF override is applied, and the random seeds. This also adds a new dependency (`cosmic`)
  to the shared `.venv`.
- **Q3 MIST.** `isochrones` grid version and metallicity. The paper gives only age ~ U(0, 12 Gyr).
  This is also a new dependency, with grids of several GB.
- **Q4 Roche-lobe removal.** The paper does not specify the test.
- **Q5 Noise model.** Reproduce Fig. 5 with stock `gaiamock` (binned, the paper's model; forbidden
  on the science path by CLAUDE.md), or with `gaiamock_mod` (the project's science model, which the
  paper did not use)? Or both, as a validation step?
- **Q6 F2 < 25.** Paper Eq. 18 (applied since Phase 1) vs the authors' helper code, which omits it.
- **Q7 Solution-type fractions.** What is the real-side comparison? It needs an
  `nss_acceleration_astro` snapshot (an archive query) and a defined denominator (#341).
- **Q7a** What is the real denominator for solution-type fractions? Options: published astrometric
  NSS solutions only (orbital + acceleration); a `gaia_source` count in a volume; or drop
  fractions in favour of orbital : acceleration count ratios.
- **Q7b** Snapshot `gaiadr3.nss_acceleration_astro` (an archive query) for the acceleration
  counts?
- **Q7c** SB1 / EclipsingBinary / SB2 have no gaiamock counterpart. Exclude them from the
  solution-type comparison?
- **Q10 KS gate (#345).** The paper uses no statistical gate. Do we keep KS? If so, what minimum
  accepted-mock N or power criterion, and what p threshold?
- **Q11 Multi-solution check (#346).** The emission draws from the same rate table it is checked
  against. Demote it to a unit test, or redesign it with per-accepted-orbit vs per-real-Orbital
  denominators and binomial tolerances?
- **Q12 Resolved-pair cut (B1).** Is ρ the angular semi-major axis a/d, or the instantaneous
  projected separation? The paper says only "angular separation".
- **Q13 Long-period acceleration capture (#399).** Accept it as the DR3 rule (compare the mock to
  DR3 with the acceleration channel included, via Q7b); or bound it with a non-DR3 pop-side
  post-cascade rule; or wait for the population-level test. See `docs/gate399/README.md`.
- **Q14 σ deficit (#398).** Leave gaiamock_mod's noise as is (documented systematic); thin the
  simulated epochs to DR3's observation count and/or add bright-star excess noise pop-side (around
  gaiamock's predict + fit calls, not inside them); or request the change upstream in
  gaiamock_mod. See `docs/gate399/README.md`.
- **Q8 Input catalog.** Ask the authors for the 46 M-binary input catalog or the 137k mock catalog
  instead of regenerating them? That would remove Q1–Q4.
- **Q9 N and compute.** From §8.1, choose an N whose orbital count gives smooth histograms
  within the laptop budget, or move the run to ziggy or lux (out of scope today).

## 10. Implementation plan once approved (not started)

Only the determined parts, each using `gaiamock_mod` functions and never reimplementing them:

1. A `paper_population` input mode for `selection_function_astrometric` that reads a binary catalog
   (columns M1, M2, P, e, M_G1, M_G2, ra, dec, d, plus orientation if supplied) instead of drawing
   box priors, from a path under `paths.data_root`.
2. Apply the determined population-level steps to that catalog: A_G via Combined19
   (`mwdust`, as now), the B1 resolved-pair cut (ρ from `gaiamock.get_a_mas` and the orbital phase),
   the G < 19 cut, and isotropic orientation draws.
3. The cascade via `gaiamock.run_full_astrometric_cascade`, then the Phase 1 cuts. The flux ratio
   comes from M_G1 and M_G2 rather than an independent draw.
4. Publishable acceleration solutions (s > 20; F2 < 22 for 7-parameter) recorded per realization
   for the Q7 comparison.
5. The six-panel figure and KS statistics as in Phase 1, plus a counts-scaled version like the
   paper's.

Producing the input catalog (Galaxia + COSMIC + MIST) is blocked on Q1–Q4, or on Q8.
