# Gate 400: gaiamock's epoch excess over DR3, and a statistical epoch model (#400, #398)

Issues #400 (+2 visibility periods) and #398's epoch-count term (~11–13% more transits and CCD
observations than DR3; `docs/gate399/README.md`). The model and its calibration are specified
in `docs/EPOCH_MODEL_SPEC.md`. **The model is implemented, calibrated and off**
(`dr3.epoch_model.enabled: false`). Every modelling choice the data and papers do not settle
is listed for Ryan (spec §6). None has been chosen.

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

## Measurement

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
- `figures/validation_*.png`: the #390 re-run.
- `figures/summary.json`: every number above.
