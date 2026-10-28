# Noise model `colour_v2`: colour term on top of N2d, keeping the G = 13 break (#579)

Spec: `docs/EPOCH_MODEL_SPEC.md`, the note after §8.5 and §8.13. Ryan (2026-10-27) framed the G = 13 step as a DR3
bright/faint processing choice keyed on measured G. We keep it as a fixed discontinuity in the mock; it is not fitted
as smooth physics. `colour_v1` (§8.12, PR #575) had erased it and is superseded.

The model sits behind `dr3.epoch_model.noise_colour` (`enabled: false`, `version: colour_v2`). Production stays
`n2d_s1232`. No binaries were re-simulated.

- **Scripts:** `scripts/noise_colour_v2_579.py` (fit-s / fit-e / u0 / config / validate); the simulation scripts
  are those of #572.
- **Runs:**
  - singles: `n2d`, `v1` and `v2`, 995,344 star-realizations each (62,209 looks-single rows × 16);
  - #390 same-orbit re-injections: `n2d`, `v1` and `v2`, 2 realizations each;
  - wall time 54 min (singles v2) and 67 min (re-injection v2).

## Calibration (relative to production, exact)

**s(G, C).** ln s = a(G) + b(G) c + q c², with c = BP−RP − 1.2.
- Separate knots on each side of G = 13; colour clip [0.2, 2.6] bright and [0, 3.5] faint.
- s multiplies all actual noise, gaiamock's and N2d's. The fit is in `s_fit.json`.
- BIC: G-only 459.4; b(G)c 268.7; **b(G)c + qc² 240.2** (adopted).

| side | knot G | b (per mag BP−RP) |
|---|---|---|
| bright | 10.75 / 12.25 / 12.875 | −0.076 ± 0.038 / −0.019 ± 0.022 / −0.033 ± 0.023 |
| faint | 13.25 / 14 / 16 / 18.5 | +0.029 ± 0.014 / +0.043 ± 0.009 / +0.027 ± 0.007 / +0.022 ± 0.007 |

- q: bright +0.243 ± 0.041; faint +0.007 ± 0.004.

**e(G)**, the F2 stated-error scale (#454; `e_fit.json`): the #390 re-injection with s applied analytically through
χ²_red × (s/e)².

| side | bright | | | | | faint | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| G bin | 6–10.5 | 10.5–11.5 | 11.5–12 | 12–12.5 | 12.5–13 | 13–13.5 | 13.5–14 | 14–15 | 15–16 | 16–17.5 |
| e | 0.940 | 0.931 | 0.907 | 0.800 | 0.850 | 0.912 | 0.991 | 1.016 | 1.009 | 0.986 |

- Errors are ±0.008–0.024. e is held at 0.986 out to G = 19.

**u0.** `config/epoch_model/dr3_ruwe_u0_colour_v2.csv`, split at 13.

## Validation (`validate.json`; figures `sigma_plx_envelope_vs_G.png`, `f2_and_ruwe_vs_G.png`)

**σ_ϖ lower envelope (2nd percentile, 0.1-mag bins), single stars, RUWE < 1.25.**

| | jump 12.95 / 13.05 | falloff 12.45 / 13.05 | faint envelope / DR3 (G 13–19) | bright envelope / DR3 (G 11–13) |
|---|---|---|---|---|
| DR3 | **+24.6%** | **0.97** | – | – |
| production n2d | +19.0% | 1.17 | 0.980 | 1.143 |
| colour_v1 | +1.2% | 0.96 | 0.964 | 0.934 |
| **colour_v2** | **+22.3%** | **1.04** | **0.968** | **0.970** |

**Orbits.**
- All DR3 orbits: jump +26.8%, falloff 1.05.
- For the #390 orbits (0.2-mag bins, small numbers), the 12.5–13 / 13–13.5 envelope ratio is 1.19 for DR3, 1.15 for
  n2d, 1.04 for v1 and 1.07 for v2. v2's orbit step is weaker than DR3's; the bins are noisy.
- The faint singles envelope stays about 3% below DR3. e cannot change σ. **Correction (#581):** the earlier "13% surplus of CCD observations" was
  the stale bare-gaiamock #398 number. With the epoch model, mock N_obs / DR3 `astrometric_n_good_obs_al` is a median 1.031 and
  a mean 1.049 (124k pairs), from 1.2% too many transits and 1.4% too many CCD rows per transit (8.81 against 8.65–8.69). That
  is about 2% in σ, the size of the residual. #581 recalibrates it. This is a documented
  residual.

**F2 of the re-injected #390 orbits**, mock against DR3 on the same orbits:

| | median, G < 13 | median, G ≥ 13 | F2 > 3, G < 13 | F2 > 3, G ≥ 13 | 90th pct, G < 13 | 90th pct, G ≥ 13 |
|---|---|---|---|---|---|---|
| DR3 | 4.90 | 0.16 | 0.742 | 0.116 | 11.5 | 3.44 |
| n2d | 4.89 | 0.03 | 0.743 | 0.173 | 10.1 | 4.43 |
| v1 | 4.83 | 0.37 | 0.669 | 0.217 | 11.7 | 4.85 |
| **v2** | **4.91** | **0.23** | **0.736** | **0.197** | **11.0** | **4.74** |

- The F2 step at 13 is kept: DR3 all-orbit medians are 3.95 at G 12.8–13 and 0.25 at 13–13.2.
- The bright high-F2 tail improves (90th percentile 11.0 against DR3 11.5; n2d 10.1).
- The faint tail stays too heavy (F2 > 3: 0.20 against 0.12). That is a documented residual, possibly heavy-tailed
  errors, which are out of scope.

**σ_ϖ and RUWE ratios, mock/DR3 on the same orbits (v2).**
- σ_ϖ: 0.945 at G < 13 (n2d 1.069), 1.000 at G ≥ 13.
- RUWE: 1.043 at G < 13 (n2d 0.981), 0.999 at G ≥ 13.

**Single-star medians and tails at 0.1 mag** (bins G 9–12 / 12–13 / 13–14 / 14–16 / 16–19).

| | RUWE median (RUWE < 1.25) | RUWE > 1.4 fraction |
|---|---|---|
| DR3 | 0.984 / 0.996 / 1.001 / 1.005 / 1.007 | 0.121 / 0.091 / 0.064 / 0.032 / 0.004 |
| n2d | 1.063 / 1.059 / 1.050 / 1.043 / 1.017 | 0.003 / 0.006 / 0.011 / 0.001 / 0 |
| v2 | 1.016 / 1.017 / 1.021 / 1.017 / 1.010 | 0.001 / 0.002 / 0.005 / 0.000 / 0 |

The DR3 RUWE > 1.4 fraction includes real unresolved binaries: these rows are cut only on `non_single_star = 0`. It is
therefore an upper bound, not a target.

## Single-star before/after: paired σ_act DR3/mock by colour (`singles/compare_n2d_v2.json`, `singles/*.png`)

| G | BP−RP −0.5–0.6 | 0.6–0.8 | 0.8–1.0 | 1.0–1.2 | 1.2–1.5 | 1.5–2.0 | 2.0–2.5 | 2.5–3.5 |
|---|---|---|---|---|---|---|---|---|
| 9–13 n2d | 1.041 | 0.874 | 0.847 | 0.833 | 0.833 | 0.860 | 0.968 | – |
| 9–13 v2 | 1.036 | 0.985 | 0.989 | 1.008 | 1.002 | 0.990 | 0.960 | – |
| 13–16 n2d | 1.006 | 0.978 | 0.984 | 0.997 | 0.997 | 1.020 | 1.035 | 1.077 |
| 13–16 v2 | 1.034 | 1.000 | 0.995 | 1.002 | 0.997 | 1.005 | 0.998 | 0.999 |
| 16–19 n2d | – | 0.953 | 0.958 | 0.965 | 0.978 | 0.984 | 0.992 | 1.015 |
| 16–19 v2 | – | 0.992 | 0.995 | 0.998 | 1.006 | 1.004 | 0.994 | 0.996 |

- rms of the cell ln ratio: n2d 0.041, v1 0.0135, **v2 0.0078**.
- RUWE by colour (median): DR3 1.001–1.010; n2d 1.007–1.048; v2 0.997–1.022.
- False solutions: 0 / 995,344 accepted orbits or accelerations for v2, as for n2d and v1 (#574).

## If adopted

Re-simulate luminous gens 42–46 and co_v1 gens 1000–1003 with fragments carrying `noise_model: colour_v2`; their
primary-model version gains `:noise_colour_v2`. Recalibrate the §12.16 spurious template.
