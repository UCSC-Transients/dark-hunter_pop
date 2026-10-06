# Epoch model: statistical transit loss around gaiamock's GOST list — spec

Issues: **#400** (gaiamock has more visibility periods and transits than DR3) and the epoch
part of **#398** (recovered σ ~11% below DR3). Status: **Ryan decided the §6 options on
2026-10-03** (https://github.com/UCSC-Transients/dark-hunter_pop/issues/400#issuecomment-5971280727):
E1 on, E3 Galactic sky dependence, E4 clustered faint losses, E6 NSS calibration, E7 continuous
in G, and N2 tried under the condition that nothing else breaks. §8 specifies and calibrates the
resulting **v2** model, which is **on** (`dr3.epoch_model.enabled: true`). §1–§7 document the
#412 v1 measurement it builds on. Measurements and validation: `docs/gate400/README.md`.

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

## 8. v2: Ryan's decisions (2026-10-03)

### 8.1 E6: calibrating on NSS stars without circularity

The keep probability is calibrated on the 16,930 NSS stars (Orbital + AstroSpectroSB1). Two
routes could make it circular:

1. **Selection on N.** An NSS star is published only when its orbit is significant, and
   significance ∝ √N, so NSS stars could be biased toward more transits. Test: at fixed G the
   keep fraction does **not** rise toward the detection threshold. By s / threshold
   (1–1.5, 1.5–2, 2–4, > 4), it is 0.975 / 0.966 / 0.973 / 0.979 at 13 < G < 15, with similar
   patterns in the other bins (`keep_fit.json` → `circularity`). Selection on N is not
   measurable, so no correction is applied.
2. **Losses caused by the companion.** At fixed G the keep fraction falls with RUWE
   (13 < G < 15: 0.982 at RUWE 1.4–2 → 0.955 at RUWE > 5; G < 11: 0.979 → 0.929). A large
   photocentre orbit makes transits look like outliers to the single-star AGIS solution, which
   then does not use them. That is a loss caused by the companion, not by the star's own
   observation, and the mock's binaries must not inherit it as an average. The fit therefore
   carries a nuisance term c · ln(RUWE / 1.4) for the NSS stars (c = −0.0157 ± 0.0033), and the
   **adopted model is the NSS keep at RUWE = 1.4**, the NSS input threshold (Halbwachs et al.
   2023 §1.2). That is the star's own loss. Companion-induced losses are option **C1** (§8.6).

Random stars (G < 19, `ipd_frac_multi_peak` ≤ 2, the part of the parent's Halbwachs (b) cut in
the snapshot) enter the same fit with a free offset δ = −0.0133 ± 0.0035 in log keep. They fix
only the **shape** at G > 16.5, where there are almost no NSS stars. The level is the NSS one.
The offset is not explained by the IPD cut (13 < G < 15: 0.966 all random stars, 0.969 with
IPD ≤ 2) and is recorded as is.

### 8.2 E7 and E3: continuous in G, smooth in Galactic (l, b)

Quasi-Poisson GLM with a log link, per star: E[k] = n · exp(η), where k is DR3
`astrometric_matched_transits` and n is the nearest-position GOST transits after the window
and gaps. The dispersion is φ ≈ 2.1, dominated by the GOST grid's per-star scatter, so the
comparisons use QAIC with a common φ.

η = Σ_{j≤d} a_j x^j + Σ_{1≤ℓ≤ℓmax} Σ_m b_{ℓm} Y_{ℓm}(l, b) [+ c ln(RUWE/1.4)] [+ δ·random],
with x = (clip(G, 6, 19) − 14)/4 and Y_{ℓm} the orthonormal real spherical harmonics in
Galactic coordinates.

- **E7, G basis.** On NSS alone, the cubic polynomial beats the 8-bin step model by ΔQAIC = 24
  with half the parameters (bins8 62470.9, poly1 62466.5, poly2 62451.5, **poly3 62446.7**,
  poly4 62448.6). In the joint NSS + random fit, **poly4** is best (363061.5 vs 363072.2 for
  poly3, 363096.2 for 11 bins). Residuals of observed/model per G bin are within ±1% from
  G = 9 to 19 for both samples (NSS 0.994–1.008; random 0.982–1.015), except G < 9 (+2.3%,
  261 NSS stars). **Adopted: degree 4.**
- **E3, sky order.** QAIC keeps improving up to ℓ = 8. That is local structure, not a smooth
  sky trend: 5-fold cross-validation over HEALPix-nside-4 sky blocks gives a deviance minimum at
  **ℓmax = 2** (33186 vs 33215 at ℓ = 0, rising to 34481 at ℓ = 8). **Adopted: ℓmax = 2**
  (8 coefficients). The sky term has an rms of 0.015 in log keep.
- **Aliasing check (GOST grid).** (i) Refitting with a bilinearly interpolated GOST count
  (4 nearest grid positions) as the denominator lowers φ from 2.05 to 1.46 but leaves the sky
  term unchanged (correlation 0.976, rms 0.0164 vs 0.0149). (ii) The grid-error proxy
  ln(n_nearest/n_interp) has an rms of 0.108 per star, but only 0.0028 projects onto ℓ ≤ 2.
  Even fully correlated, the grid could account for ≤ 20% of the fitted sky amplitude. The
  ℓ ≤ 2 term is not grid aliasing.
- **Density.** The sky term replaces the per-dex density slope (E3 asked for Galactic
  coordinates). Density is not used.

Adopted coefficients: `dr3.epoch_model.transit_loss.continuous` (from
`scripts/calibrate_epoch_model_400.py keep`; standard errors in `keep_fit.json`).

### 8.3 E4: time-clustered loss for faint stars

Loss after the gaps, q = 1 − p_keep, is split. A fraction f(G)·q is lost in **episodes**: a
Poisson process in time with durations ~ Exponential(τ), covering a long-run fraction
e = f q of the time, so every transit inside an episode is lost. The rest is lost
independently per transit with p_ind = 1 − (1 − q)/(1 − e), so the expected keep is unchanged.

- **f(G)** is fitted to the random stars' visibility-period excess (DR3 `visibility_periods_used`
  vs one model draw per star, 4,000 stars per bin, random offset applied). With f = 0 the excess
  is +0.09 (15–16), +0.24 (16–17), +0.29 (17–18), +0.40 (18–19). It vanishes for
  f ≈ 0.25 / 0.5 / 0.5 / 0.5 (|excess| ≤ 0.03), for any τ between 0.5 and 8 d. Adopted: a linear
  ramp from 0 at G = 14.5 to f_max = 0.5 at G = 16.5. Transit totals are unchanged
  (model/DR3 0.991–1.004).
- **τ** is not constrained by N_vis (the excess is flat for τ = 0.5–8 d). It needs per-transit
  times. DR3 publishes them only as epoch photometry, for variability candidates
  (`scripts/fetch_epoch_photometry_400.py`, matched-transit times). The archive returned HTTP 500
  and statement timeouts on 2026-10-03 for every query of that kind (§8.7). **τ = 2 d is
  provisional** until the gap-length distribution can be compared.

### 8.4 E1: on

`dr3.epoch_model.enabled: true`. Through `epoch_model.run_cascade` / `gost_epoch_model`, every
gaiamock call that is wrapped gets the v2 epochs. `proposal_set.simulate_one` (#391) is not
edited here; the hook is in §4.

### 8.5 N2: bright-star excess noise for the unbinned overlay

DR3's published Orbital c (Halbwachs Eq. 2) has a sharp step at G = 13 (median c² − 1:
0.54 at G 5–9, 0.68 / 0.79 / 0.96 / 0.77 / 0.53 / 0.38 from 9 to 13 in steps, then 0.00 at
13–15; 0.03 at 15–17). gaiamock's own c ≈ 1.0. The term:

- **white per-CCD** noise N(0, r²(G) σ_stated²), with r²(G) linearly interpolated through
  those medians for G < 13 and zero above. Stated errors are unchanged, so F2 and c rise as in
  DR3's NSS fits. Per-CCD white noise is the form that matches the measured c ratio
  (1.22–1.34) *and* the σ deficit (0.79–0.90) together. A per-transit common offset large
  enough to reach F2 ≈ 8 would inflate the true parameter scatter about 2.7× (9 CCDs share it)
  while c rises only 1.3×.
- **RUWE renormalisation.** RUWE and the cascade's RUWE > 1.4 test are divided by
  k = √(1 + r²), the analogue of DR3's RUWE normalisation u0(G, C) (Lindegren et al. 2021).
  This is done without reimplementing anything: `ruwe_min · k` is passed to
  `fit_full_astrometric_cascade`, and the RUWE entry of the returned vector is divided by k.
  RUWE scales as 1/error, so this is identical to computing RUWE with errors inflated by k.

Ryan's condition was that nothing else breaks. **Verdict: not adopted**
(`bright_excess_noise.enabled: false`; numbers in `docs/gate400/README.md`, "v2 validation").
N2 matches F2 at G < 13 (8.8 / 9.2 / 6.1 vs DR3 8.5 / 8.5 / 5.7) and the σ ratios (0.85 → 1.06).
But with RUWE renormalised, bright NSS RUWE falls to 0.87–0.91 × published, the orbit
significance to 0.93 and acceptance to 0.661. Without renormalisation, single bright stars have
RUWE 1.28 instead of ~1.0. No single data-noise term satisfies both.

### 8.6 Options remaining for Ryan (none chosen)

| | option | note |
|---|---|---|
| C1 | add a companion-induced transit loss as a function of the mock binary's own RUWE (two-pass: predict on GOST epochs, compute RUWE, thin again) | c = −0.0157 per ln RUWE: ~1.5% fewer transits at RUWE 3, 2.5% at RUWE 5 |
| T1 | τ (episode duration) once the epoch-photometry times can be fetched | N_vis is flat in τ |
| N2a | adopt N2 anyway (renormalised), accepting the −10% bright NSS RUWE and −6% significance | matches F2 and σ |
| N2b | N2 for the NSS fits only: gate the cascade with gaiamock's `check_ruwe` on the data *without* the extra noise (`ruwe_min = 0` inside `fit_full_astrometric_cascade`, pop-side RUWE gate first). This models an AGIS vs NSS error-model difference; not yet run | composition of gaiamock calls, no reimplementation |
| N2c | leave the bright σ deficit (v2: 0.85 at G < 13) as a documented systematic | superseded by N2d |
| N2d | **ADOPTED 2026-10-03 (Ryan: "Switch it on")**: N2-u0 (§8.8), per-CCD bright noise + RUWE = UWE / u0_mock(G); accepted regression: NSS RUWE 0.91 at 12 < G < 13 | better than v2 on F2, σ, significance by G, overall and faint RUWE, single-star RUWE peak, #403 rate; worse on bright NSS RUWE at 12–13 (1.01 → 0.91) and AstroSpectroSB1 astrometry-only σ |
| R1 | random-star offset δ: leave unexplained, or investigate (sky distribution, ≥ 12 visibility periods, IPD harmonic amplitude, C*) | 1.3% |

### 8.7 Data: epoch-time snapshot

`scripts/fetch_epoch_photometry_400.py` would snapshot `gaia_source` rows with
`has_epoch_photometry` in a `random_index` slice, plus the DataLink `EPOCH_PHOTOMETRY` (RAW)
G-band transit times for 1,500 faint (G > 17) and 1,000 brighter stars. `n_transits` equals
`matched_transits` (checked on 3 stars), so these are matched-transit times. The source query
failed on 2026-10-03 with HTTP 500 (statement timeout); even `SELECT TOP 3 source_id FROM
gaiadr3.vari_summary` timed out. A retry loop then ran from 11:28 to 16:18 PDT and failed on
all 10 attempts (HTTP 500, statement timeouts, socket timeouts). No epoch-time snapshot exists,
so τ stays provisional (option T1). The script is ready to rerun as is.

### 8.8 N2-u0: bright-star noise with DR3's RUWE definition (Ryan, 2026-10-03)

DR3's RUWE is UWE / u0(G, C). Lindegren (2018, GAIA-C3-TN-LU-LL-124 §4) defines u0 as the
**41st percentile** of UWE in magnitude–colour bins of the full sample, a proxy for the mode
of well-behaved single stars, smoothed in G and C. Lindegren et al. (2021) carried the same
empirical scaling into EDR3. The normalisation absorbs any excess noise that single stars of a
given G share, including the bright-star excess.

**Mock u0.**
- Sample: 36,000 mock single stars, 600 per 0.25-mag bin over 4 ≤ G ≤ 19, at random
  `gaia_source` snapshot positions (`validate_epoch_model_v2_400.py u0`).
- Processing: `predict_astrometry_single_source` with the v2 epoch model and the N2 per-CCD
  noise; UWE from gaiamock's `check_ruwe`.
- Statistic: u0_mock(G) is the 41st percentile per bin (`calibrate_epoch_model_400.py
  u0-table`).
- Colour axis: collapsed, because gaiamock's per-CCD noise has no colour term.
- Table: `config/epoch_model/dr3_ruwe_u0_mock.csv`, with a provenance header and its sha256 in
  config.
- Values: u0 = 1.19–1.32 at G < 12, 1.11 at 12.9, then 0.94–0.99 at G ≥ 13. gaiamock's own
  single-star UWE peaks slightly below 1 there.
- Calibrated once and statistical: there is no per-star lookup.

**Gate without editing gaiamock.**
- `fit_full_astrometric_cascade` tests its UWE against `ruwe_min` internally, and
  RUWE > 1.4 ⇔ UWE > 1.4 u0(G). So `run_cascade` passes `ruwe_min · u0(G)` and divides the
  returned RUWE by u0(G).
- The gate is therefore applied to the renormalised RUWE exactly, with no need to disable it
  and re-apply it pop-side.

**Relation to #403.** The mock's input gate is now on a DR3-style normalised RUWE, as DR3's NSS
input was (EDR3 RUWE > 1.4). The 5-parameter outcome rate of re-injected Orbital solutions
halves (v2 3.4% → 1.7%; at G > 15, 5.0% → 2.8%). The remainder is the expected winner's curse:
published sources are conditioned on RUWE > 1.4. DR3's later F2 ≤ 0 single-star re-acceptance
(Halbwachs et al. 2023 §3.3; 28 of 4.1 M sources) is a different rule and is still not
modelled. **#403 is resolved in definition, not completely.**

**Validation verdict:** under the strict rule it was not adopted at first. It matches as well
as or better than v2 on almost everything (`docs/gate400/README.md`, "N2-u0"), but bright
NSS RUWE at 12 < G < 13 gets worse (1.01 → 0.91).

**Decision: N2d adopted, 2026-10-03 (Ryan: "Switch it on").** Both `bright_excess_noise.enabled`
and `ruwe_u0.enabled` are true. The one accepted regression is NSS RUWE at 12 < G < 13 at
0.91 × published (v2: 1.01). The astrometry-only AstroSpectroSB1 refit's bright σ ratio
(1.28–1.52) is also noted; its published σ used RVs.

### 8.9 The insufficient-visibility channel (#428, measured; addressed by §8.10)

gaiamock's cascade returns flag 0 when a source has fewer than 12 visibility periods or fewer than
13 observations. DR3's NSS input has the same ≥ 12 visibility-period condition (Halbwachs et al.
2023 §1.2). The question was whether the v2 model restores that channel.

- **Script:** `scripts/measure_insufficient_visibility_428.py`, on branch
  `fix/epoch-insuf-vis-428` @ `51841ee`. Outputs are in `output/gate428/` (gitignored);
  `sim.jsonl` has sha256 `871632b0…44e9`.
- **Sample:** 6,000 single stars (1,000 per G bin) and 339 binaries at the real positions and G of
  a G-stratified subset of the `20261003T063811Z_epoch_counts_400` random slice.
- **Run:** through `epoch_model.run_cascade`, bare gaiamock vs `dr3.epoch_model`, with `main` @
  `4c6d509` code (`epoch_model` is unchanged at `2b2b30e`). Bright noise and u0 were off. They do
  not change epochs.
- **Compute:** 2 workers, `nice` 10, BLAS pinned to 1 thread.
- **DR3 side:** the new snapshot `20261004T164505Z_visibility_428`: `gaia_source`
  `random_index < 300000`, G < 19, **every** solution type, 96,114 rows. It was fetched with a sync
  job, because async returned HTTP 500 on 2026-10-04. `meta.yaml` holds the ADQL and sha256.

| | G < 13 | 13–15 | 15–16 | 16–17 | 17–18 | 18–19 | G < 19, G-weighted |
|---|---|---|---|---|---|---|---|
| DR3, all solutions: `visibility_periods_used` < 12 | 0.33% | 0.35% | 0.53% | 0.70% | 1.07% | 2.73% | **1.71%** (1,646 / 96,114) |
| DR3, 5/6-parameter only | 0.17% | 0.25% | 0.24% | 0.45% | 0.51% | 1.21% | 0.79% |
| mock singles, bare gaiamock (flag 0) | 0 / 1000 | 0 | 0 | 0 | 0 | 0 | **0** |
| mock singles, v2 epoch model (flag 0) | 3 / 1000 | 1 | 0 | 1 | 1 | 1 | **0.10%** |
| mock binaries, bare and v2 (flag 0) | 0 / 89 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 / 50 | 0 |

- **Paired, same 6,000 stars:**
  - DR3 has 31 stars below 12 visibility periods; v2 has 7, bare gaiamock has 0.
  - None of v2's 7 is one of DR3's 31. For DR3's 31, v2 gives N_vis = 13–23.
  - The model matches the *mean* (median N_vis − DR3 = 0, mean +0.25), as §8.2–8.3 found. It
    does **not** match the low tail, which is about 4× too thin among 5/6-parameter stars.
  - The mock never produces flag 0 from the < 13 observation condition.
- **Where DR3's tail is.** It concentrates at faint G, low ecliptic latitude (\|β\| < 15°: 3.9%;
  \|β\| > 45°: 0.3–0.4%) and low Galactic latitude (\|b\| < 5°: 2.5%). About 60% of DR3's
  2-parameter solutions (1.5% of the G < 19 slice) have < 12 visibility periods.
  - p(G, l, b) cannot carry this. Its ℓ ≤ 2 sky term is smooth, gaiamock's GOST grid is 3.7°, and
    the clustered-loss ramp was fitted to the mean N_vis excess only.
- **El-Badry et al. (2024)** report 3 × 10⁴ of 46 M mock binaries (0.07%) dropped for < 12
  visibility periods.

Consequences:

1. The test premise "mock `insufficient_visibility` > 10%" (`tests/test_forward_model.py::
   test_validation_gate_elbadry_prior_against_fixture`) has no support in DR3 for a G < 19
   parent. It came from the removed `faint_draw_fraction` short circuit (#344, #368).
2. With the epoch model, the expected mock fraction is about 0.1%. DR3's is 1.7% overall, or 0.8%
   among 5/6-parameter solutions. So the mock over-admits about 1–2% of the G < 19 parent to the
   NSS input, mostly faint, near the ecliptic and in the plane. Whether to model that tail is
   **#432** (Ryan's call).

### 8.10 v3: visibility-period loss calibrated on the N_vis distribution (#432, 2026-10-04/05)

Ryan (#432): "You should be able to model the visibility better … it just needs to have a
different correction to match the data." The changes:

1. **Grid resolution is not the cause.**
   - Setup: 8,000 random stars of the #428 snapshot (`scripts/measure_grid_resolution_432.py`
     → `output/gate432/grid_resolution.json`). The commanded DR3 scanning law was queried at
     each star's exact position with `gaiaunlimited` 0.3.3 (`GaiaScanningLaw('dr3_nominal')`,
     Cantat-Gaudin et al. 2023), after the same window and ESA gaps.
   - Fraction below 12: exact position 0.01%, the nearest grid centre 0%, gaiamock's GOST grid
     0%, against DR3 1.66%.
   - Mean N_vis: exact 20.97, grid centre 21.00, gaiamock GOST 20.73.
   - The grid smooths nothing that matters: DR3's tail is data loss, not geometry.
   - gaiamock's GOST tables have 3.8% fewer transits than the commanded law at the same
     positions.
2. **The mean transit keep has no extra β dependence.** |sin β| is 94% explained by the
   existing Galactic ℓ ≤ 2 harmonics (the ℓ = 2 subspace is rotation-invariant, so it contains
   the ecliptic sin²β). Its residual adds nothing to the keep GLM (coefficient
   0.00003 ± 0.011). So "we already include the ecliptic latitude" holds for the *mean*.
3. **The tail is whole visibility periods lost by a minority of stars.**
   - Per star, DR3 N_vis vs the exact-position count: the deficit has median 1 and a 99th
     percentile of 8–10.
   - Stars keeping < 70% of their visibility periods: 3.7% at |β| < 15° vs 1.5% at > 45°; 3.0%
     at G 18–19 vs 0.4% at G < 15; 2.9% at |b| < 5°.
   - The strong β dependence of the < 12 fraction comes mostly from the scanning law: stars at
     |β| < 15° start with 16.7 visibility periods, those at > 45° with 27.5.
4. **Model (`dr3.epoch_model.visibility_period_loss`, replaces the E4 episodes):**
   - Each star is degraded with probability
     π = expit(c0 + c1 x + c2 x² + c_β |sin β| + c_b e^{−|b|/10°}), where x = (G − 14)/4.
   - Its visibility periods (after the gaps) are dropped whole with q_bad = expit(e0) if
     degraded, else with q0 = expit(d0 + d1 x).
   - The rest of the loss is independent per transit, so the expected kept-transit fraction
     stays at the calibrated p_keep (§8.2).
   - No per-star lookup.
5. **Calibration on the distribution.**
   - A Poisson-binomial over each star's visibility periods, two-component mixture
     (`scripts/calibrate_vp_tail_432.py`).
   - Samples: the 96,114 random #428 stars (all solution types; Ryan's "random stars for
     shape"), plus the 16,930 NSS stars (E6) truncated at N ≥ 12, because DR3 selected the NSS
     input on it.
   - Conditioned on the exact position, the star-level fit ranks the variants by AIC as follows
     (lower is better): none 187528, G 175313, G+β 175153, G+β+|sin b| 175111, and
     **G+β+e^{−|b|/10} 175091**. β is the largest single gain after G.
   - But a mock built on gaiamock's grid sees 0.24 fewer visibility periods per star than the
     exact law. Applied there, the exact-fit parameters overshoot the tail (2.2% vs DR3 1.6%).
   - The adopted parameters are therefore fitted **conditional on the grid**, as the mock
     uses it. A star-level likelihood is ill-posed there (DR3 N_vis can exceed the grid count),
     so a multinomial likelihood of the N_vis histogram in G × |β| × |b| cells is used (`--likelihood cell`).
6. **Residual mismatch (documented, not tuned).**
   - The two-point mixture cannot match the extreme tail and the 9–11 shoulder at once.
     Cumulatively, N ≤ 5: model 0.05% vs DR3 0.15%; N ≤ 11: model 2.1% vs DR3 1.7%.
   - Overall the tail is ~1.3–1.4× too heavy (it was 20× too light).
   - The cells at 15° < |β| < 30° and 10° < |b| < 30° overshoot most.
   - A richer degraded-loss distribution, for example a second, catastrophic level or a beta
     distribution of q per star, is option **V1** (§8.6).

7. **Validation and verdict.** The before/after numbers are in `docs/gate400/README.md` ("v3").
   - The visibility tail improves by an order of magnitude: < 12 is 2.3% vs DR3 1.6%, where the
     production model gives 0.09%.
   - The single-star mean N_vis improves.
   - Every injection-suite metric stays within noise of production, except the NSS-set mean
     N_vis, which moves from +0.06 to −0.16.
   - Under "nothing else may get worse", `visibility_period_loss.enabled` is left **false**, and
     the E4 episodes stay on. **Adoption is Ryan's call** (option V0).

8. **V2 then V0 (Ryan, 2026-10-06): refit on 5/6-parameter stars and switch on.**
   - Why 5/6-parameter only: the mock parent requires ϖ > 0.2 mas, so DR3's 2-parameter stars
     can never be in it, and they carry about 60% of the tail.
   - Refit: the same cell likelihood conditioned on the grid, restricted to the 94.6k random
     stars with 5/6-parameter solutions plus the NSS stars
     (`--min-params 3`; `output/gate432/vp_tail_fit_gridcell_56p_G_beta_b.json`).
   - Single stars (20,000; `output/gate432/validation_c/`), fraction with < 12 visibility
     periods against DR3's 5/6-parameter stars:

     | | mock | DR3 5/6-param | production |
     |---|---|---|---|
     | all | 1.14% | 0.76% | 0.12% |
     | \|β\| < 15° / 15–30° / 30–45° / > 45° | 2.79 / 1.44 / 0.17 / 0.00% | 1.87 / 0.59 / 0.29 / 0.10% | |
     | G 18–19 | 1.66% | 1.13% | |
     | \|b\| < 5° | 1.58% | 1.15% | |

     Against all solution types the mock gives 1.16% vs DR3 1.61%. The single-star mean
     N_vis − DR3 is −0.03 (production +0.26).
   - #390 suite, 2 realizations (paired with production r < 2):
     - NSS-set mean N_vis −0.17 (production +0.06).
     - σ ratios 1.03–1.05 at G < 13 and 0.99–1.04 above (production 1.03–1.06 and 1.00–1.03).
     - Significance 0.917 (production 0.932); RUWE 1.003 (1.004).
     - F2 at G < 11 / 11–12 / 12–13: 8.3 / 9.3 / 5.6 (production 8.7 / 9.1 / 5.8; DR3 8.5 / 8.5 / 5.7).
     - Acceptance 0.664 (0.673); P > 600 d capture 0.241 (0.228); 5-parameter outcomes
       1.9% (1.9%).
   - **Enabled.** The tail goes from 6× too light to 1.5× too heavy, and the single-star mean
     improves.
   - The NSS-set mean shift (−0.17, < 1% of the ~23 visibility periods) is in the direction
     expected when re-injecting published orbits, which DR3 selected on significance ∝ √N. The
     truncated fit itself predicts −0.10 for NSS. Everything else changes by ≤ 0.015, the noise
     level of 2 realizations.
   - The E4 episodes (`clustered_loss`) are off: this model replaces them.

### 8.11 Options added by #432 (V2 and V0 adopted 2026-10-06)

| | option | note |
|---|---|---|
| V0 | **ADOPTED 2026-10-06 (Ryan)**, after V2: enable `visibility_period_loss` (replaces E4 episodes) | tail 0.09% → 2.3% (DR3 1.6%); NSS-set mean N_vis +0.06 → −0.16; everything else within noise |
| V1 | richer degraded-star loss distribution (two degraded levels or beta-distributed q) | would fix the N ≤ 5 deficit and the 9–11 overshoot; each fit costs hours on the laptop |
| V2 | **ADOPTED 2026-10-06 (Ryan)**: fit on 5/6-parameter stars only | DR3's 2-parameter stars carry ~60% of the tail |

## References

- Boubert, D., Everall, A. & Holl, B. 2020, "Completeness of the Gaia-verse I"
  (arXiv:2004.14433), and the later papers in that series; their `scanninglaw` package 1.1.1,
  https://github.com/gaiaverse/scanninglaw (its `dr3_nominal` gap list is checked in §2).
- El-Badry, K. et al. 2024, OJAp 7, 100 (arXiv:2411.00088), §3.3, §3.3.1, §5.1.1.
- ESA, Gaps in Gaia (E)DR3 data, https://www.cosmos.esa.int/web/gaia/dr3-data-gaps.
- Halbwachs, J.-L. et al. 2023, A&A 674, A9.
- Lindegren, L. et al. 2021, A&A 649, A2 (arXiv:2012.03380), §2.2, Eq. 2, Table 1, §4.4.
