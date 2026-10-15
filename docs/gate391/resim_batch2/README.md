# #391 fresh re-simulation, gens 33 + 34 (39.4 CPU-h): v6a refit. Too sparse to be a result.

- Cache: `output/proposal_set/weights_gen33_34.npz`, built on corrected MIST.
- Fits ran alone; peak RSS 10.0 GB (noW) and 10.2 GB (cmdW).

| | corrected gens 23–32 (`../retruth_v6a`) | gens 33+34, noW | gens 33+34, cmdW |
|---|---|---|---|
| −lnL | 763.7 | 10831 | 10869 |
| converged starts | 12/12 | 10/12 | 8/12 |
| orbit ESS | — | 245 | 80 |
| big bins with ESS_b < 30 | 0/23 | 23/23 (median 14) | 23/23 (median 7) |
| α_lo / α_hi | +0.78 / +0.24 | +0.91 ± 0.60 / +0.22 ± 0.19 | +3.10 / +0.44 |
| γ_P / ln L_P | — | −1.60 / +1.82 | −2.82 / +2.50 |
| ln F_twin | — | −4.0 (at bound) | −4.0 (at bound) |
| ln f_evo | −0.18 | +0.69 | +0.30 |
| G 5–9 / 9–10 / 16–19.5 | 1.58 / 1.25 / 1.02 | 2.86 / 1.36 / 0.68 | 3.83 / 1.90 / 0.33 |
| d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 | 1.06 / 0.97 / 0.95 / 1.55 | 1.38 / 1.15 / 0.97 / 1.96 | 2.26 / 1.71 / 1.10 / 1.71 |
| orbit:accel at d > 1.5 (real 0.425) | 0.616 | 0.095 | 0.068 |

**Reading.** Monte Carlo noise dominates. Every large bin has ESS below 30, and 55 of the 108 orbit bins have no draws at all; they hold 1,842 real orbits. Parameters run to their bounds. So this fit says nothing about the d > 1.5 kpc bin either way.

**Weakest bins** (by row-variance relative error, among bins with ≥ 300 real orbits; relative errors 0.72–1.0):
- G 5–13, every distance: short P (log P < 2.6) dwarfs and evolved stars, and log P 2.6–2.92 evolved stars.
- Plus one faint cell: G 13–19.5, d 0.5–1 kpc, log P 2–2.6, e > 0.6.
- These bins hold only 3–27 parent rows, so they are limited by both the number of parent rows and the number of draws.
