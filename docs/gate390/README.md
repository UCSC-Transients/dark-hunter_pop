# Gate 390: Step 1a injection test

Issue #390. This test validates `gaiamock_mod` without any population model. Each published Gaia
DR3 `Orbital` / `AstroSpectroSB1` solution is re-injected at its real sky position, parallax,
proper motion and G, and the recovered solution is compared with what Gaia published.

## Method

- **Truth:** the published photocentre orbit. Campbell elements come from the published A/B/F/G via
  `gaiamock_mod.get_Campbell_elements`. The round trip rebuilds the published Thiele–Innes to
  2×10⁻¹⁴ mas, so the injected orbit *is* the published one. P, e, `t_periastron` (J2016.0, the
  same epoch as `rescale_times_astrometry`), parallax, proper motion, position and G are used as
  published.
- **Injection:** `gaiamock_mod.predict_astrometry_binary_in_terms_of_a0` takes the photocentre a0
  directly, so **no M1 / M2 / flux ratio is assumed**. For f = 0 the luminous-binary path
  (`al_bias_binary`) reduces to exactly the same linear photocentre shift. The fit is
  `gaiamock_mod.fit_full_astrometric_cascade` (`ruwe_min` 1.4, accelerations on), followed by the
  El-Badry et al. (2024) Eq. 18 + Eqs. 20–22 cuts (`forward_model.passes_orbital_solution_cuts`,
  thresholds from `dr3.selection_function_astrometric.orbital_solution_cuts`). Nothing is
  reimplemented.
- **Seeding (#371):** each (source_id, realization) gets numpy and libc `rand()` seeds from
  `SeedSequence(42, spawn_key=(2, source_id, realization))` via `forward_model.seeded_global_rng`.
  Every realization replays bit for bit (tested), whatever the worker count or order.
- **Published uncertainties:** σ_a0 = a0 / `significance`. A check that propagates the full NSS
  covariance gives exactly the same value (median ratio 1.0000000). σ_cos i is the published full
  A/B/F/G covariance propagated linearly; 12 systems whose covariance cannot be rebuilt get NaN,
  never a diagonal substitute. `visibility_periods_used` and `ecl_lat` come from a Gaia archive
  query of `gaiadr3.gaia_source` for the selected sources.
- **Sample:** 168,065 published solutions (134,598 Orbital + 33,467 AstroSpectroSB1, the
  El-Badry 2024 §4 comparison set; uncut snapshot `20260826T234425Z_3d3f740b080c`, joined to NSS
  enrichment on `(source_id, nss_solution_type)`, 0 duplicate keys). The test sample is
  stratified with equal allocation over 3×3×3 quantile cells of G, log P and log a0/σ_a0:
  **1,026 Orbital + 270 AstroSpectroSB1 = 1,296 systems × 5 realizations = 6,480**. The pilot
  (270 systems, `pilot/`) is a nested subset. Population-weighted acceptance uses the stratum
  weights.
- **Pulls:** (recovered − published) / σ_recovered over accepted realizations. gaiamock returns
  no 12-parameter covariance, so the cos i pull uses the published σ_cos i.

## Cost and compute

| | value |
|---|---|
| CPU per realization reaching the orbital fit | 13.9 s mean (17.2 s in the pilot under heavier load) |
| CPU per realization stopping earlier | 0.007 s |
| mean over all realizations | 11.9 CPU s; 16.9 wall s per worker |
| throughput, 8 workers, laptop load ~30 | 1.8–3.9 wall s per realization; 5,130 in 157 min, pilot 1,350 in 74 min |
| peak worker RSS | 0.26 GiB (about 2.2 GiB for 8 workers plus the parent) |

N was set to fit the 4–6 h budget. The cost is 5 realizations × ~14 CPU s ≈ 70 CPU s per system.

## Results

### Acceptance (orbit fit + all cuts)

| type | N realizations | accepted | stratum-weighted | reached orbit fit | 5-par | 7/9-par accel. | orbit failed cuts |
|---|---|---|---|---|---|---|---|
| Orbital | 5,130 | **0.742 ± 0.007** | 0.735 | 0.835 | 0.029 | 0.136 | 0.094 |
| AstroSpectroSB1 | 1,350 | **0.879 ± 0.009** | 0.874 | 0.926 | 0.010 | 0.064 | 0.047 |

For realizations that reached the orbital fit (Orbital), the per-cut pass fractions are
a0/σ > 5: 0.954, a0/σ > 158/√P: 0.930, ϖ/σ_ϖ > 20000/P: 0.947, σ_e: 0.980, F2 < 25: 1.000.

Acceptance against the published significance relative to its threshold, s / max(5, 158/√P):
1–1.5: 0.53; 1.5–2: 0.62; 2–4: 0.74; > 4: 0.89. Against period: < 300 d 0.84; 300–600 d 0.81;
600–1000 d 0.66; > 1000 d 0.65.

### Pulls (accepted realizations)

| parameter | Orbital median | Orbital σ_MAD | AstroSpectroSB1 median | AstroSpectroSB1 σ_MAD | \|pull\| ≥ 5 |
|---|---|---|---|---|---|
| P | −0.02 | 1.00 | +0.01 | 1.02 | ≤ 0.3% |
| e | +0.15 | 0.94 | +0.15 | 0.96 | ≤ 0.3% |
| a0 | +0.16 | 0.97 | +0.17 | 0.99 | ≤ 0.2% |
| ϖ | −0.01 | 1.00 | −0.03 | 0.97 | ≤ 0.3% |
| cos i (published σ) | +0.01 | 0.85 | −0.01 | 0.89 | ≤ 0.3% |

The pulls show no trend with G (G tertiles 7.2–12.5 / 12.5–13.8 / 13.8–17.1: σ_MAD 0.93–1.05 for P,
e, a0, ϖ) or with the published number of visibility periods (13–21 / 21–27 / 27–33: σ_MAD
0.91–1.02). See `figures/pulls_by_g.png` and `figures/pulls_by_nvis.png`.

### Recovered vs published uncertainty, significance, RUWE, F2

Median recovered / published, with 16th–84th percentiles:

| | Orbital | AstroSpectroSB1 (astrometry-only refit) |
|---|---|---|
| σ_P | 0.86 (0.61–1.14) | 0.98 (0.73–1.54) |
| σ_e | 0.88 (0.67–1.13) | 1.06 (0.78–1.66) |
| σ_a0 | 0.89 (0.65–1.22) | 1.08 (0.81–1.49) |
| σ_ϖ | 0.89 (0.71–1.09) | 0.92 (0.73–1.16) |
| a0/σ_a0 | 1.08 (0.75–1.49) | 0.92 (0.63–1.21) |
| RUWE | 0.99 (0.91–1.12) | 1.07 (0.95–1.26) |
| F2, recovered − published | −1.2 (−7.3 to +2.9) | −3.0 (−6.9 to +0.9) |

gaiamock's per-system visibility periods minus the published `visibility_periods_used`: median +2
(0 to +4).

## What agrees

- **P, ϖ, a0 pulls are N(0,1)** in width (σ_MAD 0.97–1.00), with < 0.3% outliers and no period
  aliasing. They do not depend on G or N_vis. gaiamock's orbit fit and error bars are internally
  consistent, and the published orbits are recoverable.
- **RUWE** is reproduced to 1% (median ratio 0.99) over 1.4–30, so the scanning law plus the
  sky-dependent noise give the right 5-parameter residual amplitude.
- **Inclination:** no bias in cos i (median +0.01σ).
- **Acceptance rises with published significance** as a detection threshold should, from about
  0.45–0.5 just above threshold to about 0.97 at a0/σ_a0 > 100.

## What does not agree, with likely causes

1. **Recovered uncertainties are about 11% below the published ones** for Orbital (σ ratio
   0.86–0.89, significance +8%, F2 −1.2). The cos i pulls normalised by the published σ are
   correspondingly narrow (σ_MAD 0.85). Likely causes: the unbinned CCD-level noise model vs DR3
   (spec Q5); the paper's G < 13 bright-star term and its 0.5 mas term for marginally resolved
   epochs, both missing from `gaiamock_mod`; and real excess (attitude/calibration) noise, which
   the F2 offset also points to. Filed as **#398**. For AstroSpectroSB1 the recovered σ exceeds
   the published one (σ_a0 ×1.08), **as expected**: the published solution also used RVs and this
   refit is astrometry-only.
2. **Acceleration capture at long period.** 23–24% of Orbital realizations with P > 600 d end as
   accepted 7- or 9-parameter acceleration solutions (0.6% below 300 d). That is the largest
   acceptance loss, 13.6% of all Orbital realizations. The captured solutions have median s = 16
   against the threshold of 12 and F2 = 11 against 25. Likely causes: the optimistic noise in
   (1) inflating s; a DR3 acceleration-vs-orbit decision that differs from the El-Badry Fig. 2
   cascade; and partly winner's curse (see 4). Filed as **#399**.
3. **Small positive biases in e and a0** (+0.15σ). The e bias grows toward low e (+0.44σ for
   e < 0.1, +0.05σ for e > 0.3), as expected for an estimator bounded at e ≥ 0 and for the
   positive-definite a0 at modest S/N. Expected statistical behaviour, not a forward-model defect.
4. **Acceptance below 1 near threshold** (0.53 for s / threshold < 1.5). **Expected:** published
   solutions are a detected sample, biased toward favourable noise (Eddington / winner's curse).
   Re-drawing the noise around the published, already up-scattered a0 fails the cuts about half
   the time. This is interpreted, not tuned away. The 11% optimistic σ partly offsets it, so the
   near-threshold acceptance in a corrected model would be somewhat lower still.
5. **Visibility periods:** gaiamock counts +2 more than DR3 `visibility_periods_used` (median),
   probably because of DR3 data gaps and rejections that GOST does not model. The pulls show no
   dependence on it. Filed as **#400**.

## Implication for Step 1 (#339)

The forward model recovers published orbits without bias in P, ϖ and cos i, and with correct RUWE.
Two systematic differences matter for the six-panel match: the ~11% optimistic uncertainties,
which make the mock detect binaries too easily near threshold, and the long-period acceleration
capture, which suppresses mock orbits at P > 600 d. Both should be understood (#398, #399) before
the mock period and significance distributions are compared to DR3.

## Reproduce

```bash
PY=.venv/bin/python; OUT=output/gate390
$PY scripts/run_injection_test_390.py --out $OUT build-published
$PY scripts/run_injection_test_390.py --out $OUT select --tag full --n-orbital 1000 --n-astrospectro 250
$PY scripts/run_injection_test_390.py --out $OUT fetch-vis --tag full        # Gaia archive (network)
$PY scripts/run_injection_test_390.py --out $OUT run --tag full --n-realizations 5 --workers 8
$PY scripts/run_injection_test_390.py --out $OUT assemble --tag full
$PY scripts/plot_injection_test_390.py $OUT/injection_test_full.h5 --fig-dir docs/gate390/figures
```

`run` appends to `realizations.jsonl` and skips finished (source_id, realization) pairs, so it
resumes after a kill.

## Artifacts (gitignored, primary checkout `output/gate390/`)

| file | sha256 |
|---|---|
| `injection_test_full.h5` (1,296 systems, 6,480 realizations: truth, recovered, σ, cut flags, acceptance, seeds, cost) | `e585c9e69fba221bd00903185e0569001ffbdd62fabab262030310d8df10e6c2` |
| `injection_test_pilot.h5` (270 systems, 1,350 realizations) | `d327a67b751136bd37527add93a2f9042c020829e82cd87b4c474e2f4036ddfd` |
| `published_orbits.h5` (168,065 published solutions + corr_vec) | `6ba9338015c8c03802f2c2adddabe28ca058425bdc694a3c1221877bf567d9d8` |

Provenance: gaiamock_mod `gaiamock-mod-v1`, sha256 `71454d76…f028`, submodule `dd30fdb`; config
`mock_population.random_seed` = 42. Every number above is in `figures/summary.json`.

## Figures

- `figures/pulls.png`: pull histograms with an N(0,1) overlay.
- `figures/sigma_recovered_vs_published.png`: recovered vs published σ, log–log with a 1:1 line.
- `figures/significance_ruwe_f2.png`: a0/σ_a0, RUWE and F2, recovered vs published.
- `figures/acceptance.png`: acceptance vs a0/σ_a0, G, P and |β|.
- `figures/pulls_by_g.png`, `figures/pulls_by_nvis.png`: the pulls split by G and by N_vis.
- `figures/nvis_and_outcomes.png`: gaiamock vs published N_vis, and the cascade outcome fractions.
