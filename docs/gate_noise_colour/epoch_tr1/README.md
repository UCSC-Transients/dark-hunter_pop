# Transit and CCD-row recalibration (`tr1`) with the colour_v2 refit (`colour_v2.1`) (#581)

Spec: `docs/EPOCH_MODEL_SPEC.md` §8.14.

**Correction.** The "13% more CCD observations" quoted in #579 was the stale bare-gaiamock number. With the epoch
model the surplus was 3.1% (median) / 4.9% (mean) in paired CCD observations and 1.3% / 3.1% in transits, measured on
124k single-star pairs.

## Calibration (`calibration.json`)

Fitted on the colour_v2 single stars (995,344 realizations) against DR3 on the same rows:

| G bin | < 11 | 11–12 | 12–13 | 13–14 | 14–15 | 15–16 | 16–17 | 17–18 | 18–19 |
|---|---|---|---|---|---|---|---|---|---|
| κ_T (p_keep scale) | 1.000 | 0.985 | 0.988 | 0.988 | 0.993 | 0.991 | 0.990 | 0.999 | 1.004 |
| ρ (CCD-row keep) | 0.990 | 0.979 | 0.982 | 0.983 | 0.982 | 0.982 | 0.983 | 0.983 | 0.982 |

- **No E4 episodes.** The marginal transit-count quantiles at fixed G already matched DR3. DR3's 99th percentile is
  higher, not lower.
- **The paired upper tail is grid scatter.** The broad paired ratios (p10–p90 0.79–1.27) come from per-star count
  scatter in gaiamock's 3.7° GOST grid.
- **Refit of s, e and u0 in the tr1 frame** (`s_fit_tr1.json`, `e_fit_tr1.json`). s and e had absorbed the count
  surplus, so they were refit. Because the numbers changed, the model is relabelled `colour_v2.1`.
  - faint colour slopes b at G 13.25 / 14 / 16 / 18.5: +0.035 / +0.045 / +0.029 / +0.023 (±0.007–0.015);
  - q: bright +0.239, faint +0.006;
  - e above 13: 0.90 / 0.97 / 1.00 / 1.00 / 0.96.

## Validation

**Single stars** (`validate_counts_final.json`, `singles/compare_v2_v21tr1.json`, `validate.json`):

| | colour_v2 | colour_v2.1+tr1 | DR3 |
|---|---|---|---|
| transits, ratio of means | 1.0054 | 1.0011 | – |
| transits, paired median | 1.013 | 1.000 | – |
| transits, paired mean | 1.031 | 1.026 | – |
| CCD obs, ratio of means | 1.0233 | 1.0011 | – |
| CCD obs, paired median | 1.030 | 1.008 | – |
| CCD obs, paired mean | 1.049 | 1.026 | – |
| N_vis, mean ratio | 0.996 | 0.995 | – |
| CCD rows per transit | 8.807 | 8.653 | 8.653 |
| σ_act DR3/mock, G 9–13 / 13–16 / 16–19 | 0.992 / 0.999 / 1.001 | 0.997 / 1.001 / 1.001 | – |
| σ_act rms | 0.0078 | 0.0075 | – |
| σ_ϖ envelope jump at G = 13 | +22.3% | +23.0% | +24.6% |
| falloff 12.45/13.05 | 1.04 | 1.05 | 0.97 |
| faint envelope / DR3 | 0.968 | 0.968 | – |

**#390 same orbits** (mock/DR3; DR3 values in the last column):

| | colour_v2 | colour_v2.1+tr1 | DR3 |
|---|---|---|---|
| σ_ϖ ratio, G < 13 | 0.945 | 0.948 | – |
| σ_ϖ ratio, G ≥ 13 | 1.000 | 1.003 | – |
| F2 median, G < 13 | 4.91 | 5.07 | 4.90 |
| F2 median, G ≥ 13 | 0.23 | 0.25 | 0.16 |
| F2 > 3, G < 13 | 0.736 | 0.757 | 0.742 |
| F2 > 3, G ≥ 13 | 0.197 | 0.200 | 0.116 |
| F2 90th pct, G < 13 | 11.0 | 11.0 | 11.5 |
| F2 90th pct, G ≥ 13 | 4.74 | 4.83 | 3.44 |
| RUWE ratio, G < 13 | 1.043 | 1.057 | – |
| RUWE ratio, G ≥ 13 | 0.999 | 1.012 | – |
| envelope step at G = 13 | 1.07 | 1.07 | 1.19 |

#390 transits: paired median ratio 1.000; ratio of means 0.990 for these N-selected NSS stars (cf. §8.10).

**Verdict.** The observation counts are now right, and the single-star noise matches as well as or better than
colour_v2.
- The #390 shifts are inside the calibration errors: F2 by +0.16 bright, against the e bootstrap of ±0.2–0.6; RUWE
  by about +0.013.
- Unchanged residuals: the faint F2 > 3 tail, the #390 orbit step at G = 13, and the faint single-star envelope.
