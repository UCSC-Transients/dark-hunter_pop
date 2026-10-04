# Gate 400: gaiamock's epoch excess over DR3, and a statistical epoch model (#400, #398)

Issues #400 (+2 visibility periods) and #398's epoch-count term (~11–13% more transits and CCD
observations than DR3; `docs/gate399/README.md`). The model and its calibration are specified
in `docs/EPOCH_MODEL_SPEC.md`. After Ryan's 2026-10-03 decisions, the **v2** model (spec §8)
is **on** (`dr3.epoch_model.enabled: true`). The bright-star N2 noise was tried and is **not
adopted**, because it breaks the bright NSS RUWE. See "v2" below. The v1 sections that follow
are the #412 record.

## Summary

| | finding |
|---|---|
| time window | gaiamock's DR3 window **is** the AGIS window (Lindegren et al. 2021 §2.2: 22 Aug 2014 – 28 May 2017) to 0.9 d; it explains 0.08% of transits. DR3's 25 Jul 2014 start includes the EPSL month, which AGIS did not use |
| (a) counting | visibility-period definition identical (≥ 4 d gap); DR3 `matched_transits` ≈ GOST (0.97–1.01); the excess is in which matched transits AGIS **used**; bad CCDs 0.3–0.5%; gaiamock's 10% drop is per CCD row in the unbinned overlay, not per transit as El-Badry et al. (2024) §3.3 describe, and with SM + AF rows it reproduces DR3's 8.7 CCDs per transit to 1–2% |
| (b) published gaps | ESA's 138-gap DR3 astrometry list removes 7.56% of GOST transits (0–18% by position); it takes the N_vis excess from +2 (median) to 0 for NSS stars and the transit ratio from 0.899 to 0.973 |
| (c) residual | per-transit keep fraction after the gaps 0.96–0.97 at G < 16, falling to 0.93 at G = 18.5–19; weak in density, none in \|b\| or \|β\|; scan-direction "trend" is the 3,072-position GOST grid, not a loss |
| model | gaps + per-transit loss p(G) = 0.028–0.073 (9 G bins), calibrated on 94,635 random stars; gaiamock's 10% row drop kept as the per-CCD term |
| on the data | NSS: mean transits 48.3 vs DR3 48.6 (GOST 54.0); mean N_vis 23.14 vs 23.15 (GOST 25.40) |
| validation (#390 set, 3,888 realizations) | Orbital σ ratio 0.89 → **0.98** (ϖ, a0), 0.87 → 0.97 (P); at G ≥ 13 **1.02–1.04**; CCD obs / DR3 1.129 → **1.004**; N_vis excess median +2 → **0** (mean +2.23 → −0.07); long-P acceleration capture **unchanged** (0.234 → 0.239); the El-Badry G < 13 U(0, 0.04) mas term is far too small to reproduce DR3's bright-star F2 |

## v2 (Ryan's decisions, 2026-10-03)

The v2 model is calibrated on NSS stars (E6), continuous in G (E7), uses a Galactic ℓ ≤ 2 sky
term (E3) and clustered faint-star losses (E4), and is on (E1). Spec §8. Fit:
`scripts/calibrate_epoch_model_400.py keep | noise` → `output/gate400/v2/{keep_fit,noise_fit}.json`.

| decision | result |
|---|---|
| E6 circularity | **Selection on N:** none measurable. The NSS keep fraction at fixed G does not rise toward threshold (13–15: 0.975 / 0.966 / 0.973 / 0.979 for s/threshold 1–1.5 / 1.5–2 / 2–4 / > 4). **Companion-induced loss:** keep falls with RUWE (13–15: 0.982 at 1.4–2 → 0.955 at > 5), c = −0.0157 ± 0.0033 per ln RUWE. The adopted model is the NSS keep **at RUWE = 1.4**: the star's own loss, without its companion's |
| E7 | cubic beats 8 G bins on NSS by ΔQAIC = 24 with half the parameters; jointly, poly4 beats 11 bins by ΔQAIC = 35. Observed/model per G bin is 0.98–1.02 at G 9–19. **Adopted: degree 4** |
| E3 | sky-blocked 5-fold CV picks **ℓmax = 2** (deviance 33186 vs 33215 at ℓ = 0 and 34481 at ℓ = 8; QAIC alone would keep adding harmonics, which is local overfitting). Sky rms 0.015 in log keep. **Not GOST-grid aliasing:** with an interpolated GOST denominator the sky term is unchanged (correlation 0.976) while φ falls from 2.05 to 1.46, and the grid-error proxy projects only 0.0028 rms onto ℓ ≤ 2 |
| E4 | with independent loss, random stars keep a N_vis excess of +0.09 / +0.24 / +0.29 / +0.40 (G 15–16 / 16–17 / 17–18 / 18–19). Episodes carrying f = 0.25 / 0.5 / 0.5 / 0.5 of the loss remove it (\|excess\| ≤ 0.03). Adopted ramp 0 → 0.5 between G = 14.5 and 16.5. **τ is not constrained by N_vis** (flat for 0.5–8 d). The epoch-time snapshot needed for the gap-length distribution could not be fetched: the Gaia archive returned HTTP 500, statement timeouts or socket timeouts on all 10 attempts, 10:13–16:18 PDT. **τ = 2 d is provisional** (option T1) |
| E1 | on |
| N2 | per-CCD white noise r²(G) = median(c²) − 1 of DR3 Orbital fits (0.38–0.96 at G < 13), RUWE renormalised by √(1 + r²). **Not adopted** (below) |

On the calibration data, one model draw per star with no per-star lookup
(`figures/model_counts_vs_dr3.png`): NSS mean transits are 1.011 × DR3 and mean N_vis 23.26 vs
23.15. For random stars it is 1.017 × and 20.32 vs 20.06. Both residuals come from the level
choices: RUWE = 1.4 for NSS, and the NSS level for random stars (offset δ = −1.3%).

### v2 validation (#390 set, 1,296 × 3, #390 seeds; single-star RUWE on 20,000 random stars)

`scripts/validate_epoch_model_v2_400.py inject | single` → `output/gate400/validation_v2/`
(`inject.jsonl` sha256 `fa7a839c…82a7`, `single.jsonl` `8adfae97…a06b`). There were 0 errors.
BLAS was pinned to 1 thread in every worker (threadpoolctl 3.7.0; each record carries
`blas_threads = [1]`), and the runs used `nice` and 6 workers. Orbital, medians:

| | #390 baseline | v1 (#412) | **v2 (adopted)** | v2 + N2 |
|---|---|---|---|---|
| σ_ϖ / σ_a0 / σ_P / σ_e | 0.89 / 0.90 / 0.87 / 0.88 | 0.98 / 0.98 / 0.97 / 0.97 | **0.97 / 0.96 / 0.95 / 0.95** | 1.02 / 1.03 / 1.03 / 1.01 |
| σ_ϖ, G < 13 / G ≥ 13 | 0.79 / 0.94 | 0.87 / 1.03 | **0.85 / 1.01** | 1.06 / 1.01 |
| a0/σ_a0 (all orbit fits) | 1.079 | 0.976 | **0.989** | 0.931 |
| F2 recovered − published | −1.15 | −1.20 | −1.21 | +0.25 |
| F2 at G < 11 / 11–12 / 12–13 (DR3 8.5 / 8.5 / 5.7) | 0.7 / 0.3 / 0.2 | 0.3 / 0.4 / 0.3 | 0.4 / 0.2 / 0.3 | **8.8 / 9.2 / 6.1** |
| RUWE rec/pub, all | 0.991 | | **0.988** | 0.952 |
| RUWE rec/pub at G < 11 / 11–12 / 12–13 | 1.10 / 1.14 / 1.02 | | 1.09 / 1.14 / 1.01 | **0.87 / 0.91 / 0.87** (not renormalised: 1.14 / 1.22 / 1.05) |
| N_vis − DR3, median / mean | +2 / +2.23 | 0 / −0.07 | **0 / +0.06** | same epochs |
| CCD obs / DR3 n_good | 1.129 | 1.004 | **1.020** | same |
| transits / DR3 | 1.114 | 1.000 | **1.000** | same |
| accepted | 0.741 | 0.679 | **0.682** | 0.661 |
| P > 600 d capture | 0.234 | 0.239 | **0.237** | 0.221 |

Single stars (`figures/single_star_ruwe.png`), RUWE median (16–84%) at G < 13:

| | median (16–84%) |
|---|---|
| DR3 | 1.053 (0.91–1.50); its tail includes real binaries |
| gaiamock | 0.997 (0.88–1.15) |
| v2 | 1.009 (0.88–1.14) |
| v2 + N2, renormalised | 1.001 (0.92–1.09) |
| v2 + N2, not renormalised | **1.28**, with 18.5% > 1.4 |

At G ≥ 13 the variants are identical (median 0.991–0.995 against DR3 1.015–1.017). The single-star
N_vis excess with v2 is +0.12 (13–17) and +0.25 (17–19), against +2.0 to +2.3 for gaiamock.

**N2 verdict (Ryan's condition): no single noise term satisfies everything.** Per-CCD noise
calibrated to DR3's c matches F2 at G < 13 (8.8 / 9.2 / 6.1 against 8.5 / 8.5 / 5.7) and closes
the bright σ deficit (0.85 → 1.06). Then:

- **With RUWE renormalised** (needed for single stars: median 1.00), bright NSS RUWE drops to
  0.87–0.91 × published. Significance falls to 0.93, acceptance to 0.661, and the
  astrometry-only AstroSpectroSB1 σ ratio rises to 1.3–1.5 at G < 13.
- **Without renormalisation**, single bright stars sit at RUWE 1.28.

DR3's bright stars carry a residual excess in the NSS fits (F2 ≈ 8) that is not visible as
extra RUWE for the same binaries. That points to a difference between the AGIS and NSS error
models, which a single noise term added to the data cannot reproduce.
`dr3.epoch_model.bright_excess_noise.enabled` is therefore **false**. The options are in spec
§8.6 (N2a–N2c).

### N2-u0: bright-star noise with DR3's RUWE = UWE / u0 (Ryan, 2026-10-03)

The N2 per-CCD noise is used with mock RUWE = UWE / u0_mock(G). u0_mock(G) is the 41st
percentile of mock single-star UWE per 0.25-mag bin (Lindegren 2018 TN LL-124), from 36,000
mock singles run through v2 and N2 (`output/gate400/u0/`; table
`config/epoch_model/dr3_ruwe_u0_mock.csv`, sha256 `8731b93b…3269`). The cascade's gate is
applied as UWE > 1.4 u0(G), which is exact (spec §8.8).

Validation runs:
- Inject: same suite and seeds as v2, `output/gate400/validation_u0/inject.jsonl`, 0 errors.
- Single stars: same 20,000 stars as before.
- Runs used `nice`, 6 workers, BLAS pinned to 1 thread.

| Orbital (medians) | DR3 / target | v2 | **N2-u0** | better? |
|---|---|---|---|---|
| F2 at G < 11 / 11–12 / 12–13 | 8.5 / 8.5 / 5.7 | 0.4 / 0.2 / 0.3 | **8.8 / 9.2 / 6.0** | yes |
| σ_ϖ/σ_a0/σ_P/σ_e at G < 13 | 1 | 0.85 / 0.84 / 0.84 / 0.84 | **1.05 / 1.04 / 1.03 / 1.04** | yes |
| same at G ≥ 13 | 1 | 1.01 / 1.02 / 1.02 / 1.00 | 1.01 / 1.02 / 1.02 / 1.00 | equal |
| a0/σ_a0 by G (< 11 / 11–12 / 12–13 / 13–15 / > 15) | 1 | 1.22 / 1.30 / 1.12 / 0.94 / 0.92 | **0.91 / 0.97 / 0.92** / 0.93 / 0.92 | yes (bright), equal (faint) |
| a0/σ_a0, all G | 1 | 0.989 | 0.930 | v2's all-G value comes from bright-high and faint-low cancelling |
| RUWE rec/pub, all | 1 | 0.988 | **1.003** | yes |
| RUWE at G < 11 / 11–12 / 12–13 / > 13 | 1 | 1.09 / 1.14 / **1.01** / 0.97 | 0.91 / **0.95** / 0.91 / **1.02** | mixed: **worse at 12–13** |
| single-star RUWE peak (41st percentile) by G | DR3 1.00–1.08 | 0.94–1.02 | **1.00–1.04** | yes |
| single-star RUWE median, G 13–17 | DR3 1.015 | 0.991 | **1.019** | yes |
| N_vis − DR3 | 0 | 0 / +0.06 | 0 / +0.06 | equal |
| accepted | (0.74 baseline) | 0.682 | 0.677 | equal |
| 5-parameter outcomes (#403) | — | 3.4% | **1.7%** | yes |
| P > 600 d capture | (undecided, #399) | 0.237 | 0.225 | no target |

AstroSpectroSB1 is an astrometry-only refit, so its published σ also used RVs. Its G < 13 σ
ratio rises from 1.0–1.2 to 1.28–1.52, significance falls from 0.82 to 0.63, and its RUWE
moves from 1.07 to 0.90.

**Verdict:** N2-u0 matches as well as or better than v2 on F2, the bright σ ratios,
significance by G, RUWE overall and at G > 13, the single-star RUWE peak at all G and the
#403 rate. It is **worse** on bright NSS RUWE at 12 < G < 13 (1.01 → 0.91) and on the
AstroSpectroSB1 astrometry-only refit. Under the rule "adopt only if everything matches as
well as or better", it is **not adopted** (`ruwe_u0.enabled: false`,
`bright_excess_noise.enabled: false`). It is offered as option **N2d** (spec §8.6).
Switching it on is two flags.

## Measurement (v1, #412)

`scripts/fetch_epoch_count_sample.py` → snapshot `20261003T063811Z_epoch_counts_400`
(94,635 random G < 19 stars, 16,930 NSS, the 1,296 #390 systems, a level-5 density map; ADQL,
date, checksums in its `meta.yaml`). `scripts/measure_epoch_counts_400.py gost | compare` →
`output/gate400/{gost_pixels.npz, epoch_compare.h5, decomposition.json}`.

DR3 `astrometric_matched_transits` / GOST transits at the same position (sum ratio):

| sample | GOST | + AGIS window | + 138 gaps | + 25 gaps (`scanninglaw`) |
|---|---|---|---|---|
| random (94,635) | 0.885 | 0.885 | 0.945 | 0.944 |
| NSS (16,930) | 0.899 | 0.900 | 0.973 | 0.972 |
| #390 set (1,296) | 0.906 | 0.907 | 0.982 | 0.981 |

Visibility periods, GOST − DR3: mean +2.26 / +2.25 / +2.29 (median +2, 16–84% 0 to +4) →
after the gaps +0.78 / +0.25 / +0.24 (median +1 / 0 / 0).

gaiamock CCDs / DR3 `n_good_obs_al` = gaps × residual × CCDs-per-transit:
random 1.068 × 1.059 × 1.017 = 1.150; NSS 1.082 × 1.028 × 1.012 = 1.126.

Residual keep fraction after the gaps vs G (random stars, 16–84% bootstrap over GOST positions):

| G | < 11 | 11–13 | 13–15 | 15–16 | 16–17 | 17–17.5 | 17.5–18 | 18–18.5 | 18.5–19 |
|---|---|---|---|---|---|---|---|---|---|
| keep | 0.962 | 0.972 | 0.966 | 0.965 | 0.956 | 0.947 | 0.948 | 0.940 | 0.927 |
| 16–84% | 0.949–0.980 | 0.965–0.981 | 0.962–0.971 | 0.960–0.970 | 0.952–0.961 | 0.943–0.952 | 0.944–0.954 | 0.935–0.946 | 0.922–0.933 |
| NSS | 0.958 | 0.977 | 0.974 | 0.968 | 0.978 | | | | |

NSS stars keep ~0.8% more than random stars at the same G. That is expected: detected binaries
are selected on significance ∝ √N. The mock reproduces this selection itself, so the model is
calibrated on the random stars.

Correlation: an independent per-transit loss at the calibrated rate reproduces DR3's mean N_vis
for NSS (−0.06 to +0.05 by G bin) and at G < 15 (+0.09 to +0.2), but leaves +0.34 to +0.45 at
G > 17. Losses for faint stars are clustered in time (option E4). Counts cannot measure
correlation inside a star's transit list, because the GOST grid adds per-star scatter (robust
σ 0.15 on the ratio, against a binomial 0.024). The mean ratio does not depend on the distance
from the GOST position (0.955–0.966), so the grid is unbiased.

## Validation

`scripts/validate_epoch_model_400.py` re-ran the 1,296 #390 systems × 3 realizations (3,888;
the #390 seeds; 0 errors; 179 min on 6 workers, BLAS pinned to 1 thread, see #408) through the
wrapper. Variant **epoch** is the model alone (gaps + p(G), with gaiamock's 10% row drop kept).
**epoch + noise** adds El-Badry et al. (2024) §3.3.1's G < 13 term to the same epochs:
σ ~ U(0, 0.04) mas per source, then N(0, σ) common to each transit's CCDs, with stated errors
unchanged. The 0.5 mas term for ξ > 0.5 needs the component separation and flux ratio, which a
photocentre-orbit injection does not have, so it was **not applied** (spec §5). The baseline
is the stored #390 result for the same (system, realization) pairs. Output
`output/gate400/validation/realizations.jsonl` (sha256 `288affb4…978c`). Every number is in
`figures/summary.json`.

Orbital, median over accepted realizations (16–84%), recovered / published:

| | #390 baseline | epoch | epoch + noise |
|---|---|---|---|
| accepted fraction | 0.741 | 0.679 | 0.679 |
| σ_ϖ | 0.892 (0.71–1.08) | **0.980** (0.78–1.19) | 0.983 |
| σ_a0 | 0.899 | **0.979** | 0.983 |
| σ_P | 0.865 | **0.969** | 0.970 |
| σ_e | 0.883 | **0.972** | 0.973 |
| σ_ϖ, G < 13 / G ≥ 13 | 0.79 / 0.94 | 0.87 / **1.03** | 0.88 / 1.03 |
| a0/σ_a0, all orbit fits (#390's definition) | 1.079 | **0.976** | 0.975 |
| F2 recovered − published, all orbit fits | −1.15 | −1.20 | −1.17 |
| c_recovered / c_published (Halbwachs Eq. 2) | 0.948 | 0.940 | 0.941 |
| RUWE | 0.999 | 0.992 | 0.993 |
| N_vis simulated − DR3, median (16–84%) / mean | +2 (0, +4) / +2.23 | **0 (−2, +2) / −0.07** | same epochs |
| CCD obs / DR3 `n_good_obs_al` | 1.129 (0.96–1.31) | **1.004** (0.86–1.17) | same |
| transits / DR3 `astrometric_matched_transits` | 1.114 (GOST) | **1.000** (0.85–1.16) | same |

σ_ϖ ratio by G (`figures/validation_sigma_ratio.png`): baseline 0.72 / 0.74 / 0.82 / 0.95 / 0.95
/ 0.92 for G < 11, 11–12, 12–13, 13–14, 14–15, > 15. With the epoch model: 0.79 / 0.80 / 0.90 /
1.04 / 1.02 / 1.00. **Above G = 13 the epoch model closes the σ deficit.** Below G = 13 the
remainder is the bright-star excess noise (#398 term (ii)).

**The El-Badry et al. (2024) term does not reproduce DR3's bright-star excess in gaiamock_mod.**
Median recovered F2 for orbit fits at G < 11 / 11–12 / 12–13 is 0.26 / 0.38 / 0.31 with the
epoch model and 0.56 / 0.60 / 0.67 with the term added, against **8.5 / 8.5 / 5.7** published.
The σ_ϖ ratio moves by only +0.005–0.014. U(0, 0.04) gives 0.023 mas rms per transit, while
#399 estimated ~0.038 mas from the c ratio. Because the overlay fits unbinned CCDs, a per-transit
offset raises χ²/ν by only σ_x²/σ_CCD² ≈ 0.03 at G ≈ 11 (σ_CCD = 0.135 mas). That is nine times
less than in the paper's binned (N_bin = 8–9) likelihood, although the extra scatter in the
parameters is the same. Reaching DR3's F2 ≈ 8.5 (χ²/ν ≈ 1.7 at ν ≈ 400) in the unbinned fit
would need ~0.1 mas per transit, or an independent per-CCD excess. This bears on #398 option 3:
the paper's value cannot be carried over to the unbinned overlay as is (option N2).

AstroSpectroSB1 (astrometry-only refit, so σ above published is expected, #390): σ_a0 1.077 →
1.197, σ_ϖ 0.917 → 1.013; N_vis median +2 → 0; CCD obs 1.128 → 1.006.

**#399 long-period acceleration capture** (Orbital, accepted 7/9-parameter fraction;
`figures/validation_capture_vs_period.png`):

| P (d) | < 300 | 300–600 | 600–1000 | > 1000 | > 600 |
|---|---|---|---|---|---|
| #390 baseline | 0.006 | 0.051 | 0.233 | 0.241 | **0.234** (362/1,545) |
| epoch | 0.021 | 0.054 | 0.236 | 0.256 | **0.239** (369/1,545) |
| epoch + noise | 0.023 | 0.053 | 0.240 | 0.251 | **0.241** (373/1,545) |

Correct epoch counts do **not** reduce capture. That fits #399's finding that the decision is
set by the parallax criterion, whose ratio scales as σ^0.05. The 3–4 pp drop #399 predicted
for its "DR3-matched" noise came mostly from the bright-star c-factor scaling, which neither
variant reproduces here. Short-period capture rises from 0.6% to 2.1% (10 of 471): with fewer
transits, a few short-period orbits are fit as accelerations.

**Acceptance falls 0.741 → 0.679** (Orbital) and 0.880 → 0.826 (AstroSpectroSB1). Realistic
epochs make published orbits harder to re-detect near threshold. The winner's-curse
interpretation of #390 (finding 4) applies more strongly.

## Options for Ryan

See `docs/EPOCH_MODEL_SPEC.md` §6 (E1–E7, N1–N2). None is chosen.

## Reproduce

```bash
PY=.venv/bin/python   # PYTHONPATH=<worktree>/src from a worktree; set OMP/OPENBLAS/VECLIB_MAXIMUM_THREADS=1 (#408)
$PY scripts/fetch_epoch_count_sample.py --data-root data --inj390 output/gate390/injection_test_full.h5
$PY scripts/measure_epoch_counts_400.py gost --out output/gate400 --workers 6
$PY scripts/measure_epoch_counts_400.py compare --out output/gate400 \
    --snapshot data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400
$PY scripts/plot_epoch_model_400.py measurement --out output/gate400
$PY scripts/validate_epoch_model_400.py run --inj390 output/gate390/injection_test_full.h5 \
    --out output/gate400/validation --n-realizations 3 --workers 6
$PY scripts/plot_epoch_model_400.py validation --inj390 output/gate390/injection_test_full.h5 \
    --log output/gate400/validation/realizations.jsonl \
    --snapshot data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400 \
    --compare output/gate400/epoch_compare.h5
```

## Figures

- `figures/transit_ratio.png`: DR3 / GOST transits per star, before and after the gaps.
- `figures/keep_vs_g.png`: residual keep fraction vs G with the calibrated step.
- `figures/keep_vs_covariates.png`: the same vs density, |b|, |β| and scan-direction strength.
- `figures/model_counts_vs_dr3.png`: transits and N_vis from one model draw per star vs DR3 and GOST.
- `figures/gap_fraction_sky.png`: fraction of GOST transits inside the published gaps.
- `figures/validation_*.png`: the #390 re-runs (v1 and v2; `validation_n2_f2_ruwe.png` is the N2 check).
- `figures/single_star_ruwe.png`, `figures/single_star_ruwe_peak_vs_g.png`: the single-star RUWE check (v2, N2, N2-u0).
- `figures/summary.json`: every number above.
