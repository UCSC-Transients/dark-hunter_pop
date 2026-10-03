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
| validation | see "Validation" below |

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

(filled in from `scripts/plot_epoch_model_400.py validation`)

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
