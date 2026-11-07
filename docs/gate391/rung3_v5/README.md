# Rung 3 v5: refit on gens 53–59 + compact-object gens 1006/1007, and closure (#615, spec §12.44)

**Status: CLOSED by Ryan 2026-11-07 with known limitations (DR4 fast path).** The close criteria were fixed before the fit. They are recorded below as measured: C1 passes; C2, C3 and C4 fail.

**Condition:** rung 5 must **not** import the rung-3 N_s or WD rates. It refits them as free nuisance parameters in its own A > 0.5 cells.

## Inputs
- **Luminous:** gens 53–59, 1,502,691 draws, one mixture (`colour_v2.1+tr1+e2+st1`). Cache built by [build_cache_53_59.py](build_cache_53_59.py).
- **WD:** co_v1 gens 1006 + 1007, one mixture, including gen 1007's target-tabulated M2 (§12.43). R_dark ≡ 0.
- **Spurious:** the §12.16 template reselected on four samples (#609: far, near, gap, low-ϖ/σ_ϖ tail).
  - π covariates: log10 ϖ/σ_ϖ and BP−RP, coefficients [−1.648, −0.984, +0.513] ± [0.110, 0.146, 0.078].
  - N_s prior N(17,589, 1,538).
- **Model:** the §12.42 terms; the relative MC error fixed at the reference θ (the §12.37 v1 fit); start θ = v4.
- **Config and chains:** `config/population/rung3_joint_v5.yaml`; [chain.sh](chain.sh) and [chain3.sh](chain3.sh).

## Fit ([fit/report.txt](fit/report.txt), [fit/summary.json](fit/summary.json))
- **Convergence:** 12/12 starts within Δ(−lnL) < 5, offsets 0.00–0.98; −lnL 17,192.8.
- **At bounds:** a1, b2, Δln F_twin,sP and κ_cb.

| parameter | value |
|---|---|
| ln A | −0.34 ± 0.17 |
| α_lo / α_hi | +0.23 ± 1.66 / −0.37 ± 0.42 |
| γ_P / ln L_P | −0.73 ± 0.96 / +0.36 ± 0.79 |
| dγ_q / ln F_twin | +0.09 ± 0.16 / −0.15 ± 0.69 |
| a2, a3 / b1, b3 | 1.16 ± 0.46, 1.29 ± 0.25 / 16.8, 1.31 ± 0.46 |
| ln f_evo | −0.79 ± 0.16 |
| ln A_ni (R_wd) | +0.37 ± 0.10 (1.44) |
| ln R_mt / f_He | −6.25 ± 0.31 / 0.07 ± 0.28 |
| β_e | +1.95 ± 0.89 |
| Δln F_twin,sP | +2.0 (bound) |
| κ_cb | +5.0 (bound) ± 6.4 |
| N_s | 40,692 ± 937 (prior 17,589 ± 1,538; pull +15.0) |

- **Totals:** orbits 0.997 of DR3, accelerations 1.03.
- **By distance** (< 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc), total 0.99 / 0.96 / 1.05 / 1.16:
  - luminous 0.65 / 0.59 / 0.49 / 0.30;
  - WD 0.13 / 0.10 / 0.12 / 0.07;
  - spurious 0.21 / 0.27 / 0.44 / 0.79.
- **WD component:** 13.4k C1 orbits and 216k accelerations.
- **ESS:**
  - luminous orbits 631, luminous accelerations 610; WD orbits 32, WD accelerations 453;
  - cells (luminous / WD): d < 0.3: 338 / 15; G < 13: 298 / 13; G < 13 & log P < 2: 8 / 2; A > 0.5 by distance: 7/10, 11/8, 11/3, 13/1.

## Profiles ([fit/profile.json](fit/profile.json))
- dγ_q 0 / 0.2 / 0.4 / 0.6: 17,193.1 / 17,193.2 / 17,195.9 / 17,199.4.
- ln F_twin −1.5 / −0.75 / 0 / 0.75: 17,194.6 / 17,193.4 / 17,192.9 / 17,194.0.
- Δln F_twin,sP −4 / −2 / 0 / 2: 17,193.7 / 17,193.5 / 17,193.3 / 17,192.8.
- κ_cb −4 / −2 / 0 / 2 / 4.5: 17,194.5 / 17,194.0 / 17,193.6 / 17,193.3 / 17,192.9.
- N_s 17.6k / 26k / 35k / 45k: 17,592.7 / 17,336.4 / 17,212.8 / 17,202.0.

## Close criteria (§12.44, as measured)
| criterion | result |
|---|---|
| C1 convergence | **PASS**: 12/12 starts within 5; no profile point below the fit |
| C2 spurious amplitude | **FAIL**: pull +15.0; spurious share 0.44 / 0.79 at 0.7–1.5 / > 1.5 kpc (limit 0.35) |
| C3 totals | **FAIL (marginal)**: > 1.5 kpc at 1.16 (limit 1.15); the other bins and accelerations pass |
| C4 MC adequacy | **FAIL**: WD orbit ESS 32 (< 50); luminous 631 passes; cells below ESS 10 are listed above |
| C5 shape (reported) | orbit 1-D χ²/dof: G 2.6 ± 1.2, d 11.8 ± 2.2, P 1.6 ± 3.5, e 3.4 ± 1.4, (BP−RP)₀ 3.6 ± 5.0, M_G0 1.5 ± 1.2 |

## Known limitations (closure, Ryan 2026-11-07)
1. N_s runs to 2.3× the four-sample π̂ prior. The distant luminous shortfall is absorbed by the spurious term beyond 0.7 kpc.
2. A > 0.5 at d < 0.3 kpc is over-predicted ×4.3 by the WD component (2,594 against 600), and the WD orbit ESS is 32.
3. The short-period twin term and κ_cb are unconstrained; their cells have ESS 2–8.
4. The bright F2 / noise limitations of EPOCH_MODEL_SPEC §8.16 apply, and G < 10 is excluded.
5. R_dark ≡ 0 at rung 3.

## v3 / v4 / v5 on the flagged mismatches ([compare/](compare/); v3 edges)
DR3/mock with the pull in brackets; for medians, mock against DR3.

| mismatch | DR3 | v3 | v4 | v5 |
|---|---|---|---|---|
| N(F2 > 3), G < 13 | 35,964 | 0.62 | 0.88 | 0.92 (−1.5) |
| N(F2 > 8), G < 13 | 9,959 | 0.48 | 0.71 | 0.69 (−3.2) |
| F2 median, G < 13 | 4.55 | 4.89 | 5.40 | 5.55 |
| G × F2 / M_G0 × F2, χ²/dof (d ≥ 0.3) | | 12.0 / 10.6 | 9.5 / 6.9 | 16.3 ± 3.4 / 9.5 ± 2.3 |
| N(ϖ/σ_ϖ > 238), (BP−RP)₀ < 0.8 / 0.8–1.2 / 1.2–1.8 | 1,647 / 4,718 / 3,432 | 1.60 / 1.94 / 1.58 | 1.39 / 1.22 / 0.89 | 1.04 / 1.31 / 1.27 |
| N(ϖ/σ_ϖ > 238), log P 2–2.5 / 2.5–3 | 6,612 / 5,031 | 1.58 / 1.58 | 1.02 / 1.39 | 1.30 / 1.29 |
| colour × ϖ/σ_ϖ / P × ϖ/σ_ϖ, χ²/dof (d ≥ 0.3) | | 14.4 / 20.6 | 42.7 / 132.7 | 13.4 ± 8.8 / 63.5 ± 13.6 |
| a0/σ_a0 edge: DR3 share above mock p99, log RUWE 0.1–0.2 / 0.4–0.6 / 0.6–1 (expect 0.01) | | 0.034 / 0.042 / 0.040 | 0.008 / 0.012 / 0.024 | 0.008 / 0.014 / 0.023 |
| a0/σ_a0 × RUWE, χ²/dof (d ≥ 0.3) | | 46.4 | 34.7 | 29.8 ± 5.9 |
| N(P < 100 d), G < 13 | 677 | 1.30 | 0.88 | 1.09 (+0.3) |
| G × P, χ²/dof (d ≥ 0.3) | | 18.7 | 108.3 | 55.0 ± 6.0 |
| N(RUWE > 1.4) (orbit total) | 125,226 | 0.79 | 0.93 | 1.00 (+0.1) |
| G × RUWE, χ²/dof (d ≥ 0.3) | | 44.3 | 22.1 | 27.0 ± 6.1 |
| N(A > 0.5), d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 | 600 / 2,516 / 3,288 / 849 | 0.35 / 1.62 / 1.08 / 0.96 | 0.33 / 1.02 / 0.61 / 0.65 | 0.23 / 0.82 / 0.89 / 0.71 |
| twin/single, 0–0.2 / 0.2–0.5 / 0.5–1 kpc | 0.32 / 0.33 / 0.46 | 0.32 / 0.32 / 0.48 | 0.32 / 0.33 / 0.46 | 0.33 / 0.33 / 0.46 |
| median e, G 10–13 / 13–16 / 16–19.5 | 0.32 / 0.31 / 0.31 | 0.39 / 0.32 / 0.23 | 0.30 / 0.29 / 0.24 | 0.32 / 0.33 / 0.24 |

## Plots
- [plots/index.html](plots/index.html) and [plots/README.md](plots/README.md): about 200 figures.
  - Every 1-D (all, by distance, and orbits by Orbital / AstroSpectroSB1) and every 2-D pair, for orbits and accelerations.
  - χ²/dof ranking with 30-fold bootstrap MC errors.
- [compare/](compare/): v3 / v4 / v5 side by side, with [before_after.md](compare/before_after.md), [noise_metrics.md](compare/noise_metrics.md), [population_metrics.md](compare/population_metrics.md) and [chi2_2d_d_ge_0p3.md](compare/chi2_2d_d_ge_0p3.md).
