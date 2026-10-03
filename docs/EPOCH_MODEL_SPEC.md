# Epoch model: statistical transit loss around gaiamock's GOST list — spec

Issues: **#400** (gaiamock has more visibility periods and transits than DR3) and the epoch
part of **#398** (recovered σ ~11% below DR3). Status: **spec for Ryan's approval.** The model
is implemented and calibrated, and it is **off** (`dr3.epoch_model.enabled: false`) until Ryan
chooses among the §6 options. The measurements and the before/after validation are in
`docs/gate400/README.md`.

Related: `docs/gate399/README.md` (the #398 decomposition that found the ~11% transit excess),
`docs/gate390/README.md` (the injection set), `docs/GAIAMOCK_API.md` (what must not be
reimplemented), `docs/MOCK_POPULATION_SPEC.md` §4 (the mock's gaiamock interface).

## 0. Requirement (Ryan)

The epoch-loss model is **statistical**. The mock simulates stars we do not know
individually, and some never reach the catalogue, so the forward model has **no per-star
lookup** of real epoch counts. Only the model's parameters are calibrated on real data.

## 1. What gaiamock does today

- `get_gost_one_position(ra, dec, 'dr3')` returns the GOST-predicted rows of the nearest of
  3,072 HEALPix-16 positions (~3.7° apart) with JD 2456891.5–2457902. Each FoV transit is
  **10 rows**: one sky-mapper (SM) row and nine astrometric-field (AF1–AF9) rows, 4.9–5.8 s
  apart (measured: 84% of transits have 10 rows, 14% have 9 (CCD row 4, where AF9 is the
  wavefront sensor); mean 9.78).
- `predict_astrometry_*` then drops each **row** with probability 0.1
  (`t[np.random.uniform(0, 1, len(t)) > 0.1]`).
- El-Badry et al. (2024, §3.3) describe that 10% as a loss of **FoV transits**: "about 10% of
  FOV transits do not result in usable data … we reject FOV transits for each source with 10%
  probability", citing Lindegren et al. (2021). In the stock gaiamock the data are binned
  per transit, so the drop is per transit. In the unbinned `gaiamock_mod` the same line acts
  per CCD row, so it removes **no whole transits**. It removes ~1 of the 10 rows per transit.

## 2. Data

Snapshot `data/dr3/gaia_snapshots/20261003T063811Z_epoch_counts_400/` (`meta.yaml` records the
ADQL, the query date, the row counts and a SHA256 per table;
`scripts/fetch_epoch_count_sample.py`):

| table | selection | rows |
|---|---|---|
| `random.h5` | `gaia_source`, `random_index < 300000`, G < 19, 5/6-parameter solutions | 94,635 (88,558 with RUWE < 1.4) |
| `nss.h5` | Orbital + AstroSpectroSB1, `random_index < 181170977` (a uniform ~10% slice) | 16,930 |
| `inj390.h5` | the 1,296 #390 injection systems | 1,296 |
| `density.h5` | source counts per HEALPix level-5 pixel, `random_index < 10⁷`, all G | 12,288 |

Columns: `matched_transits`, `matched_transits_removed`, `astrometric_matched_transits`,
`astrometric_n_obs_al`, `astrometric_n_good_obs_al`, `astrometric_n_bad_obs_al`,
`visibility_periods_used`, `phot_g_n_obs`, position (`ra`, `dec`, `l`, `b`, `ecl_lon`,
`ecl_lat`), G, BP−RP, RUWE, `astrometric_params_solved`, `scan_direction_strength_k1..4`,
`scan_direction_mean_k1..4`. The density proxy is the level-5 count scaled by
1,811,709,771 / 10⁷ per deg². For each star the GOST side uses gaiamock's own lookup, without
the random 10% drop (`scripts/measure_epoch_counts_400.py gost`).

**Gap list.** ESA, "Gaps in Gaia (E)DR3 data" (https://www.cosmos.esa.int/web/gaia/dr3-data-gaps),
`Astrometry_EDR3.csv`: 138 gaps in OBMT revolutions, extending Lindegren et al. (2021) Table 1
(26 gaps > 1 rev) with internal sources. Downloaded 2026-10-03, sha256
`e3bb06ba3a2cffd93bdf8336152b95d7cbd7ee6f192277bd70f6cea293777161`, stored verbatim at
`config/gaia_gaps/dr3_astrometry_gaps_edr3.csv`. OBMT is converted with Lindegren et al.
(2021) Eq. 2, JD_TCB = 2457023.75 + (OBMT − 1717.6256)/4. The AGIS window is OBMT
1192.13–5230.09 (Lindegren et al. 2021 §2.2).

**`scanninglaw`** (Boubert & Everall; PyPI `scanninglaw` 1.1.1, 2021-06-03,
https://github.com/gaiaverse/scanninglaw) is installable from PyPI as a source tarball. It
was not installed here: the preamble forbids adding packages to the shared `.venv`, and its
data are fetched at run time. Its `dr3_nominal` version ships `data/Astrometry_dr3.csv`, a
**25-gap** list (Lindegren et al. 2021 Table 1 without one non-gap event; same OBMT
columns), plus the same AGIS window. All 25 coincide with ESA entries. The ESA list adds 113
short gaps (micro-meteoroid hits, gap tuning), which remove only 0.08% more transits (7.56%
vs 7.48%). The ESA list is used because it is the superset and the official source.

## 3. Measurement and decomposition

DR3 `astrometric_matched_transits` / GOST transits at the same positions (sum ratio over
stars; `docs/gate400/figures/transit_ratio.png`):

| sample | GOST as gaiamock reads it | after AGIS window | after window + gaps | residual loss |
|---|---|---|---|---|
| random, G < 19 (N = 94,635) | 0.885 | 0.885 | 0.945 | 5.5% |
| random, RUWE < 1.4 | 0.885 | 0.886 | 0.946 | 5.4% |
| NSS (N = 16,930) | 0.899 | 0.900 | 0.973 | 2.7% |
| #390 set (N = 1,296) | 0.906 | 0.907 | 0.982 | 1.8% |

gaiamock's CCD count over DR3 `astrometric_n_good_obs_al` (the #398 13%) factorises as follows.

| factor | random | NSS |
|---|---|---|
| (b) published gaps (transits) | 1.068 | 1.082 |
| (c) residual transit loss | 1.059 | 1.028 |
| (a) CCD rows per transit, gaiamock 0.9 × 9.78 = 8.80 against DR3 `n_good_obs_al` per transit (8.65 / 8.70) | 1.017 | 1.012 |
| product = gaiamock CCDs / DR3 `n_good_obs_al` | 1.150 | 1.126 |

**(a) Counting definitions.**

- *Time window:* gaiamock's JD 2456891.5–2457902 is the AGIS window (22 Aug 2014 21:00 UTC to
  28 May 2017; Lindegren et al. 2021 §2.2) to within 0.9 d. It removes 0.08% of transits.
  The issue's premise that gaiamock is ~3% *shorter* than DR3 holds only against the
  photometric window (which starts 25 Jul 2014 and includes the one-month EPSL). AGIS did not
  use the EPSL.
- *Visibility period:* identical. Lindegren et al. (2021, §4.4) define it as a group separated
  from others by ≥ 4 days; gaiamock counts `diff > 4` days.
- *Matched vs used:* DR3 `matched_transits` (all matched FoV transits, including the EPSL)
  is 0.97 (random) to 1.01 (NSS) × GOST. `astrometric_matched_transits` is 0.912 / 0.893 ×
  `matched_transits`. **The transit excess is entirely in which matched transits AGIS used.**
  `matched_transits_removed` is 0 for the median star.
- *`n_obs` vs `n_good_obs`:* the strongly down-weighted ("bad") CCDs are 0.3–0.5%.
- *Per CCD vs per transit:* DR3 uses AF CCDs only: 8.70 (random) and 8.72 (NSS) per transit
  in `n_obs_al`, against GOST's 8.78 AF rows. gaiamock keeps all 10 rows including SM and drops
  10%, giving 8.80. That reproduces DR3's per-transit CCD count to 1–2%, but **by coincidence
  of SM + 10% rows**, not by modelling the AF count.

**(b) Known spacecraft gaps.** The 138 published gaps cover 7.5% of the AGIS window in time.
They remove 7.56% of all GOST transits, with 5–95% of positions losing 0–18% (sky ribbons
around decontaminations; `figures/gap_fraction_sky.png`). El-Badry et al. (2024, §5.1.1) saw
the same ribbons as regions without good DR3 solutions near decontamination events. The gaps
take the median visibility-period excess from **+2 (0 to +4) to +1 (random) and 0 (NSS)**.
The mean excess falls from +2.26 to +0.78 (random) and from +2.25 to +0.25 (NSS).

**(c) Residual.** After the gaps (`figures/keep_vs_g.png`, `figures/keep_vs_covariates.png`;
bootstrap over GOST positions, because stars sharing a position share its count error):

- **G:** the keep fraction is 0.96–0.97 at G < 16 and falls to 0.927 ± 0.005 at 18.5 < G < 19.
  Below G = 11 it is 0.962 (+0.018 / −0.013). This is the only clear trend, and it is consistent
  with the magnitude-dependent on-board deletion and VPU window losses that ESA describes.
- **Density:** within a G bin, the keep fraction changes by +0.01 to −0.03 per dex of source
  density. The largest change is −0.028 per dex at 18.5 < G < 19. Over the full sample it falls
  from 0.976 to 0.94 between 10^2.5 and 10^5.5 deg⁻², which is partly the G mix.
- **|b|, |β|:** no monotonic trend beyond ±0.02.
- **Scan direction:** the keep fraction is 0.92–0.95 for `scan_direction_strength_k1` < 0.3 and
  1.09 above 0.4. This is **not a loss**. It is the nearest-of-3,072 GOST lookup: where scans
  pile up, the transit count changes faster than the 3.7° grid resolves. The mean ratio does
  not depend on a star's distance from its GOST position (0.955–0.966 from 0 to 2.5°), so the
  lookup is unbiased on average. The per-star scatter it adds (robust σ of the ratio 0.08 at
  < 0.25° and 0.15 overall, against a binomial 0.024) is a property of gaiamock's grid. The
  mock reproduces it, so it is not modelled. **It also means count data alone cannot measure
  how losses are correlated within a star.**
- **Time:** DR3 publishes no per-transit times for these stars. The visibility periods test it
  indirectly. After the gaps, *independent* per-transit loss at the calibrated rate
  reproduces DR3's mean N_vis for NSS (model 23.14 vs DR3 23.15) and at G < 15
  (+0.09 to +0.2). At G > 17 the model has +0.34 to +0.45 more visibility periods than DR3.
  That is the signature of **time-clustered loss for faint stars** (on-board deletion removes
  contiguous spans), which independent per-transit loss underestimates.
- **NSS vs random:** at the same G, NSS stars keep ~0.8% more transits (13 < G < 15: 0.974 vs
  0.966). Detected binaries are selected on significance ∝ √N, so they prefer stars with
  more transits. The mock reproduces that selection by itself, so the calibration uses the
  unselected random sample, and NSS is a validation set.

## 4. The model

Applied to gaiamock's GOST table **before** `predict_astrometry_*` reads it. gaiamock's own
code, including its 10% row drop, is unchanged.

1. **Deterministic:** remove rows outside the AGIS window (OBMT 1192.13–5230.09) or inside any
   of the 138 published gaps.
2. **Stochastic, per FoV transit:** drop each remaining transit, with all its rows, with
   probability p(G). Draws are independent between transits.

   | G | < 11 | 11–13 | 13–15 | 15–16 | 16–17 | 17–17.5 | 17.5–18 | 18–18.5 | 18.5–19 |
   |---|---|---|---|---|---|---|---|---|---|
   | p | 0.038 | 0.028 | 0.034 | 0.035 | 0.044 | 0.053 | 0.052 | 0.060 | 0.073 |

   Fit: maximum likelihood per bin for a binomial keep fraction, Σ DR3 transits / Σ GOST
   transits after the gaps, on the 94,635 random stars (`scripts/measure_epoch_counts_400.py
   compare`). The 16–84% bootstrap half-widths are 0.003–0.006 (0.016 at G < 11). Outside
   3 < G < 19 the nearest bin applies. An optional linear density term
   (`density_slope_per_dex`, default 0) is implemented but not switched on (§6, E3).
3. **Within a transit:** gaiamock's 10% row drop stays. It is the per-CCD term, and with
   10 GOST rows per transit it reproduces DR3's 8.70 CCDs per transit to 1.7%.

Result on the calibration and validation stars, one model draw per star with no per-star
lookup (`figures/model_counts_vs_dr3.png`):

| | DR3 | GOST (gaiamock today) | model |
|---|---|---|---|
| random: mean transits | 41.76 | 47.21 | 41.76 |
| random: mean N_vis | 20.06 | 22.32 | 20.42 |
| NSS: mean transits | 48.59 | 54.05 | 48.32 (ratio 0.994) |
| NSS: mean N_vis | 23.15 | 25.40 | 23.14 |
| NSS: N_vis model − DR3, 16/50/84% | | 0 / +2 / +4 | −2 / 0 / +2 |

### Implementation

`src/darkhunter_pop/epoch_model.py`:

- `gost_epoch_model(gaiamock, config, SourceEpochContext(g_mag), rng)` is a context manager.
  It replaces the module attribute `gaiamock.get_gost_one_position` with a wrapper that calls
  the original and returns the thinned table, and restores it on exit. Nothing in gaiamock is
  edited or reimplemented. `predict_*` looks the name up at call time, so `run_full_astrometric_cascade`,
  `predict_astrometry_luminous_binary` and `predict_astrometry_binary_in_terms_of_a0` all read
  the thinned list. With `enabled: false` it is a no-op.
- `epoch_model_rng(base_seed, *spawn_key)` gives a separate `numpy.random.Generator` from
  `SeedSequence(base_seed, spawn_key=(*key, 400))`. The thinning replays bit for bit, and with
  zero loss the predicted epochs are bit-identical to bare gaiamock (tested).
- Config: `dr3.epoch_model` (path-specific; `dr4.epoch_model: null`), including the gap-table
  checksum, which is verified on load.

**Hook point for #391 (not edited here):** `proposal_set.simulate_one` (and
`forward_model`'s cascade call) would wrap its `run_full_astrometric_cascade` call as

```python
with seeded_global_rng(seeds, c_funcs), gost_epoch_model(
    gaiamock, em_cfg, SourceEpochContext(g_mag=draw["phot_g_mean_mag"]),
    epoch_model_rng(cfg.base_seed, stream, draw_index),
):
    cascade = gaiamock.run_full_astrometric_cascade(...)
```

Any switch changes every draw's epochs, so the stored #391 outcomes would need a full
re-simulation (as for #398 options 2–4).

## 5. Validation (#390 injection set)

The 1,296 systems × 3 realizations were re-run with the #390 seeds through the wrapper
(`scripts/validate_epoch_model_400.py`). Variant `epoch` is the model alone. Variant
`epoch_noise` adds El-Badry et al. (2024) §3.3.1's unmodelled per-FoV-transit noise for G < 13:
σ ~ U(0, 0.04) mas drawn per source, then N(0, σ) common to the CCDs of each transit, with the
stated errors unchanged. The paper's 0.5 mas term for marginally resolved epochs (ξ > 0.5)
needs the component separation and flux ratio. The #390 truth is a photocentre orbit (a0
only), so that term is undefined for this test and is not applied. Results
(`docs/gate400/README.md`): with the model, Orbital recovered/published σ moves from 0.87–0.90 to
0.97–0.98 overall, and to 1.02–1.04 at G ≥ 13. CCD observations / DR3 go from 1.129 to 1.004, and
the visibility-period excess from +2 to 0 (median). Long-period acceleration capture is unchanged
(0.234 → 0.239). The G < 13 remainder (σ_ϖ ratio 0.79–0.90) is bright-star excess noise, and
the paper's U(0, 0.04) mas term raises the recovered F2 by only ~0.3 against a published
5.7–8.5.

## 6. Options for Ryan (none chosen)

| | option | note |
|---|---|---|
| E1 | switch the model on (gaps + p(G)) for the mock | full #391 re-simulation |
| E2 | gaps only, no stochastic loss | removes ~55% (random) to ~75% (NSS) of the transit excess |
| E3 | add the density term (fit per G bin, −0.03 to +0.02 per dex) | weak, largely degenerate with G and the GOST grid |
| E4 | add time-clustered loss for G > 17 (e.g. a fraction of p lost as whole visibility periods, calibrated to N_vis) | matters little for NSS (G < 16), more for a faint mock parent |
| E5 | per-CCD treatment: keep gaiamock's 10% row drop (now), or drop the SM row and use a ~1% per-CCD loss (needs `reject_10_percent=False`, which only `predict_astrometry_luminous_binary` exposes; the a0 path cannot) | numerically equivalent to 1–2% |
| E6 | calibrate on random stars (now), or on NSS | NSS is selected on N; random is the unselected parent |
| E7 | bin edges in G (now 9 bins from the calibration sample), or a smooth logistic | |
| N1 | El-Badry σ ~ U(0, 0.04) drawn per source (validation default) or per transit | the paper is ambiguous |
| N2 | add a G < 13 term to the mock (#398 option 3) | the paper's U(0, 0.04) mas per transit is too small for the unbinned overlay (F2 +0.3 vs +5.7–8.5 needed); a value calibrated to DR3's F2 / c ratio (#399: ~0.038 mas rms) would be a new choice |

## 7. Limitations

- GOST times are barycentric while the gaps are in spacecraft OBMT. The difference (< 8.5 min)
  is negligible against the gaps that matter. Micro-meteoroid gaps are sub-minute and remove
  0.08% of transits in total.
- The 3,072-position GOST grid adds per-star count scatter (robust σ ~ 0.15 on the ratio). It
  is a property of gaiamock that the mock inherits and the model does not correct.
- Losses within a star cannot be separated from the grid scatter with counts alone, so the
  per-transit independence is assumed. The N_vis test supports it for G < 16.
- The random sample excludes 2-parameter solutions (stars with < 9 visibility periods or poor
  astrometry), so the faintest bins are mildly biased high in keep fraction.

## References

- Boubert, D., Everall, A. & Holl, B. 2020, "Completeness of the Gaia-verse I"
  (arXiv:2004.14433), and the later papers in that series; their `scanninglaw` package 1.1.1,
  https://github.com/gaiaverse/scanninglaw (its `dr3_nominal` gap list is checked in §2).
- El-Badry, K. et al. 2024, OJAp 7, 100 (arXiv:2411.00088), §3.3, §3.3.1, §5.1.1.
- ESA, Gaps in Gaia (E)DR3 data, https://www.cosmos.esa.int/web/gaia/dr3-data-gaps.
- Halbwachs, J.-L. et al. 2023, A&A 674, A9.
- Lindegren, L. et al. 2021, A&A 649, A2 (arXiv:2012.03380), §2.2, Eq. 2, Table 1, §4.4.
