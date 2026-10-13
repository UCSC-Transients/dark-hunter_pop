# #391 rung 3: cmdW convergence, and the bright / near diagnosis (Ryan, 2026-10-13)

This is analysis only. No new draws were made, nothing was downloaded beyond reading the Green et al. (2019) paper text, and nothing was deleted.

- **Scripts:** `scripts/diagnose_near_bright_391.py`, and `scripts/rung3_refit5_391.py` (new `--wkeys`, `--init-from` and `--init-key` options).
- **Outputs:** `report.txt` (this directory) and `docs/gate391/rung3_topup/batch2_cmdW36/`.

## (a) cmdW re-run on the 108-bin grid

- **Setup.** 36 starts, seeded from the noW best fit (gens 23–32).
- **Result.** The best −lnL is **766.6**, against 770.9 from the earlier 12-start cmdW fit. 12 of 36 starts end within Δ < 5 of it, and the next cluster sits at +14.7. Only b1 is at a bound (P < 100 d Beta; few data).

| | noW (gens 23–32, 12 starts) | cmdW (36 starts) |
|---|---|---|
| ln A | −0.25 ± 0.08 | −0.08 ± 0.09 |
| α_lo | +0.36 ± 0.18 | +0.26 ± 0.18 |
| **α_hi** | **+0.42 ± 0.08** | **+0.18 ± 0.09** |
| γ_P | −0.34 ± 0.10 | −0.46 ± 0.11 |
| ln L_P | +1.00 ± 0.09 | +1.13 ± 0.10 |
| Δγ_q | +0.28 ± 0.20 | +0.36 ± 0.18 |
| Beta, log P 2–2.6 (a, b) | 1.21, 1.63 | 1.10, 1.49 |
| Beta, log P ≥ 2.6 (a, b) | 1.30, 2.46 | 1.23, 2.39 |
| ln f_evo | −0.01 ± 0.06 | −0.11 ± 0.06 |
| N_s (prior 22.1k ± 1.9k) | 25.2k (+1.6σ) | 25.4k (+1.7σ) |
| near (< 0.3 kpc): mock + spurious / mock only | 1.07 / 0.90 | 1.11 / 0.94 |
| min big-bin ESS_b | 54 | 51 |

**α_hi depends on the Malmquist weight:** Δ = 0.24 ± 0.12, about 2σ. The other parameters agree within 1σ.

## (b) Bright excess (G < 10) and near-distance residual

### Reddening

**Literature.**
- **Green et al. (2019), Bayestar19.** §4: "120 distance bins spaced logarithmically in distance from 63 pc to 63 kpc", so the map has no bin inside 63 pc. §2: the bins run from µ = 4 to 19 mag. The reliable distance range per pixel is "based on the inferred stellar distances" (`dustmaps` `BayestarQuery`, `min_reliable_distmod`).
- **mwdust `Combined19`.** It combines Marshall et al. (2006), Green et al. (2019) and Drimmel et al. (2003) on the Bayestar19 grid (µ = 4–18.875). It evaluates with `InterpolatedUnivariateSpline(k=1)`, which **extrapolates linearly below 63 pc**. Our `giants.combined19_ebv` does not clip it.

**Measured** (× 0.884, as the CMD uses it):

| distance | parent median E(B−V) | parent 90th percentile | real C1 orbits median | real 90th percentile |
|---|---|---|---|---|
| 0–40 pc | 0.000 | 0.005 | 0.000 | 0.004 |
| 40–63 pc | 0.000 | 0.009 | 0.000 | 0.012 |
| 63–100 pc | 0.000 | 0.018 | 0.000 | 0.019 |
| 100–150 pc | 0.000 | 0.030 | 0.000 | 0.033 |
| 200–300 pc | 0.035 | 0.14 | 0.033 | 0.12 |

The extrapolation never goes negative.

**If E(B−V) were set to 0 inside 100 pc:**
- 111 usable parent rows (0.07%) are affected; A_G is 0.000 at the median and ≤ 0.025 mag at the 90th percentile.
- **0 evolved-class flips.**
- The CMD Malmquist log-weight change on the 5,616 affected draws is +0.000 at the median (5–95%: −0.04 to +0.09).
- These rows carry 3–4% of the mock C1 weight, against 2.4% of the real orbits.

**Conclusion.** Combined19 is already about 0 inside 100 pc. The effect is far too small to drive a ×1.2–1.6 excess, or the near residual. No refit was needed.

### Local Milky Way structure

The parent is a **uniform random subsample of the real `gaia_source`** (random_index < K, scale_to_full a single constant). Local density structure is therefore inherited empirically, not modelled as a smooth or volume-scaled density.
- The Malmquist W is a CMD-completeness weight, not a density model.
- The targeted parent component cancels in the importance weights.
- There is no separate completeness correction.

The only finite-sampling effect is ESS, and the bright cells below have ESS 50–116. Local structure is **not** the driver.

### Where the bright excess is

Batch-2 fit, noW; (mock + spurious) / real, with the mock ESS in brackets:
- **G 8–10 dwarfs at 0.1–0.5 kpc: ×1.39–1.46** (ESS 50–116).
- **G 8–10 evolved at 0.2–0.5 kpc: ×1.82** (ESS 77).
- G 5–8 dwarfs within 0.1 kpc: ×1.34 (ESS 26; ×2.36 with cmdW).
- G 10–12: 0.99–1.13 everywhere.

These are **F/A dwarfs (M_G ≈ 1–3, M1 ≈ 1.3–2 M⊙)** and bright giants: a high-mass-primary population.

### Bright-star detection

- **Near re-injection self-consistency** (real vs mock-symmetric acceptance): G 5–10 0.90 vs 0.96 (n = 10 / 8); G 10–12 0.81 vs 0.95; G 12–14 0.77 vs 0.92. This shows no bright-specific detection problem, though G < 10 has only 10 systems.
- N2d (bright-star excess noise) is on in every generation. Window class and gating at G < 13 enter only through gaiamock's σ_AL(G) and N2d. This is **not tested at G < 10** with useful statistics.

## Options for Ryan (none chosen)

**Bright excess:**
1. **(recommended first)** A bright-G rung-1 re-injection: about 100–150 real orbits at G < 10, × 3 realizations, about 2 CPU-h. It would settle detection vs population.
2. If detection is consistent, a high-mass break in the frequency slope: a third α above about 1.4 M⊙, reweightable at no draw cost. Or check isochrone M1 for hot / rotating A stars.
3. Accept the excess and report G < 10 (about 4% of the C1 orbits) as a known limitation.

**Reddening taper (MP-Q47):**
- (i) no change (recommended): the map is already about 0 inside 100 pc;
- (ii) clamp the mwdust extrapolation below 63 pc to the 63 pc value;
- (iii) a Local-Bubble taper.

(ii) and (iii) change the parent CMD, and therefore the isochrone M1 drawn at generation time. That invalidates stored draws: re-simulating gens 23–32 costs about 480 CPU-h. They are not justified by the measured size of the effect.
