# #573: the a0/σ_a0–RUWE upper edge, mock vs DR3, as a function of N

Ryan (2026-10-26, rung-3 v3 plots, PR #571) saw a sharp upper edge of mock a0/σ_a0 against RUWE, which DR3 orbits
cross. The question here: is the mock/DR3 offset driven by the number of observations N? If it were, it would scale
as about √N.

Script: `scripts/edge_vs_n_573.py`. Numbers: `edge_vs_n.json`. Analysis only; nothing was simulated.

## Data

- **Mock.** Accepted orbits of the rung-3 v3 luminous generations 42–46 (primary model
  `looks_single_v2:ruwe1.25:dm-0.4:c1.05-2.75:g19`, noise `n2d_s1232`).
  - Weights are the deterministic-mixture importance weights at the default MdS17 target, from cache
    `weights_gen42_46.npz`.
  - 14,246 orbits with RUWE > 1.4 at 9 ≤ G < 19; Kish ESS 678.
  - N_vis and N_obs come from the cascade vector.
- **DR3.** All Orbital and AstroSpectroSB1 orbits: 165,362 with RUWE > 1.4 at 9 ≤ G < 19.
  - `significance` (= a0/σ_a0) comes from `nss_enrichment`; RUWE and `visibility_periods_used` from
    `20261003T182211Z_nss_orbit_input_columns`.
  - N_obs (`astrometric_n_good_obs_al`) is published only for the 16,930-orbit `epoch_counts_400` subset.
- **Statistic.** The normalised edge is the 99th percentile of z = (a0/σ_a0)/√(RUWE² − 1).
  - For one photocentre orbit, UWE² − 1 ∝ a0²/σ² and a0/σ_a0 ∝ √N a0/σ. So at fixed RUWE the largest a0/σ_a0 rises
    as √N, and z removes the RUWE dependence.
  - Errors are bootstrap errors.
  - N bins with mock ESS < 20 are masked: they have no measurable 99th percentile.

## Result: N is not the driver

| N_vis (median) | 16 | 19 | 22 | 26 | 28 |
|---|---:|---:|---:|---:|---:|
| log10 edge DR3/mock | 0.095 ± 0.030 | 0.126 ± 0.007 | 0.122 ± 0.016 | 0.085 ± 0.009 | 0.078 ± 0.013 |
| mock ESS | 84 | 173 | 115 | 268 | 190 |

- **Offset vs N_vis.** The DR3 edge is above the mock by 0.08–0.13 dex (a factor 1.2–1.35) in every N_vis bin.
  - The log–log slope of the offset against N_vis is **−0.25 ± 0.06**. √N scaling would give +0.5, which is
    excluded at > 10σ.
  - Each side's own edge rises with N: DR3 slope 0.30 ± 0.01, mock 0.54 ± 0.06. The mock follows √N, and DR3 is
    shallower.
- **Offset vs N_obs** (DR3 subset only; 3 bins pass the ESS mask): 0.00 ± 0.03, 0.083 ± 0.010, 0.098 ± 0.009, slope
  0.20 ± 0.09.
- **The N distributions are the same.** Median N_vis is 24–25 (DR3) against 22–25 (mock), and median N_obs is 373–403
  against 374–408, in every G bin.
- **By G.** The offset is 0.13 dex at all N_vis for 9 ≤ G < 13, and falls from 0.12 to 0.04 dex with N_vis at
  13 ≤ G < 15. Above G = 15 the mock ESS is too small.
- **By period.** The offset is the same at fixed log P: 0.080 / 0.098 / 0.083 / 0.077 dex for log P 2–2.5 / 2.5–2.75 /
  2.75–3 / 3–4. So the period mix does not explain it either.
- **By RUWE.** The offset is roughly constant (0.1–0.2 dex) from RUWE 1.4 to 30.

**Same orbits on both sides** (`paired_reinjection`). The #390 published orbits were re-injected through the epoch
model (`output/gate432/validation_c`, 1,779 accepted realizations).
- The mock z is lower than DR3's z for the *same* orbit by a median 0.040 dex: 0.056 at G 9–13, 0.028 at 13–15,
  0.032 at 15–17.
- There is no trend with N_vis (−0.03 to −0.07 dex).
- So about half of the population offset is per orbit: the mock gives a lower a0/σ_a0 at the same RUWE. The rest
  comes from which orbits populate the top 1%.

**Noise amplitude cannot move this edge.**
- With u0 defined from the mock's own single stars, any per-star multiplicative change of the noise, actual or stated,
  cancels in z:
  - σ_a0 ∝ actual noise;
  - RUWE² − 1 ∝ a0² / (u0 σ_stated)², and u0 σ_stated ∝ actual noise.
- The #572 colour term and the #454 F2 calibration therefore leave the edge unchanged by construction.
- What can move it:
  - a different relation between DR3's AGIS RUWE and the NSS orbit errors, for example AGIS down-weighting of
    transits that a large orbit makes look like outliers (spec §8.1: companion-induced transit loss, option C1);
  - how σ_a0 is formed from the orbit covariance;
  - heavy-tailed errors, which are out of scope for now.

## Figures

- `edge_scatter_by_nvis.png`: a0/σ_a0 against RUWE in N_vis bins, DR3 (black) and mock (orange; point size ∝ weight),
  with each side's 99th-percentile curve.
- `edge_norm_vs_nvis.png`, `edge_norm_vs_nobs.png`: the normalised edge z_99 against N on both sides, a √N reference,
  and the DR3/mock offset with its log–log slope.
- `edge_offset_vs_nvis_by_g.png`: the offset against N_vis in G bins.
