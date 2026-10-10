# #391 rung 3, refit 4: staged top-ups (generations 29 and 30) and refits (spec §12.13)

Ryan approved this on 2026-10-10. The run used one ~50 CPU-h stage, plus a single extra stage because refit 4 found that sampling was still the limit. No further top-up has been started.

- **Script:** `scripts/rung3_refit4_391.py` (`choose` picks bins from ESS alone; `fit` runs the refit).
- **Configs:**
  - `config/population/proposal_set_restart_topup29.yaml` and `config/population/proposal_set_restart_topup30.yaml`;
  - `config/population/rung3_refit4_fine.yaml` (the grid chosen on gens 23–29);
  - `config/population/rung3_refit4_fine_gens23_30.yaml` (the grid chosen on gens 23–30).
- **Code:** a targeted parent component (`ParentProposalConfig.target_*`, with tests).

| subdirectory | draws | bins (orbit; acceleration) |
|---|---|---|
| `refit3_bins/` | gens 23–29 | refit-3 grid: d 2 × log P 2 × e 2; G 2 × d 2 |
| `fine_grid/` | gens 23–29 | ESS-chosen: d 3 × log P 2 × e 3 (18); G 4 × d 2 |
| `gens23_30_refit3_bins/` | gens 23–30 | refit-3 grid |
| `gens23_30_fine_grid/` | gens 23–30 | ESS-chosen again: d 3 × log P 2 × e 2 (12); G 4 × d 4 |
| `gens23_30_gen29_fine_grid/` | gens 23–30 | **control:** the gen-29 18-bin grid, so the effect of generation 30 can be seen at fixed bins |

## Generations

**Generation 29**
- **Proposal:** re-centred on ln L_P +1.7, γ_P −1.1 and α_hi +0.6. It has a 40% targeted parent component (G 12–16, d 0.7–1.5 kpc), a period core at log P 2.3–3.3, and a broad e proposal.
- **Run:** 10 workers under `nice`, BLAS pinned, `nohup`.
- **Cost:** draws cost 0.63 CPU-s each in practice, so the planned 330k would have exceeded the budget. `output/proposal_set/budget_stop.sh` stopped the run at **50.8 CPU-h**, and the contiguous prefix of **292,002 draws** (indices 867,000–1,159,001) was assembled with `scripts/assemble_partial_generation.py`.

**Generation 30**
- **Why it ran:** the gens 23–29 refit was still sampling-limited. No orbit grid with a G split was reachable, the log P < 2.6 cell had to stay merged, and a big bin was at ESS_b 20.
- **Proposal:** re-centred on γ_P −0.75, ln L_P +1.15 and Δγ_q +0.25. It has a 40% targeted component (G 5–16, d 0.7–5 kpc), a period core at log P 1.5–3.3, and e support 0.6.
- **Cost:** the full **330,000 draws** (indices 1,197,000–1,526,999) completed in **44.6 CPU-h**.
- **Resume commands:** in the logs and on #391.

## Per-bin and total ESS (noW, at each fit's own best θ)

| fit | ESS of C1 orbits, before → after | orbit bins below ESS_b 30 (big bins) |
|---|---|---|
| refit-3 grid, + gen 29 | 743 → 1,051 | 2 of 8 (1) |
| refit-3 grid, + gen 30 | 1,052 → 1,366 | 2 of 8 (1) |
| 18-bin grid, + gen 29 | 906 → 1,331 | 5 of 18 (1) |
| 18-bin grid, + gen 30 (control) | 752 → 964 | 7 of 18 (3) |
| 12-bin grid chosen on gens 23–30 | 360 → 447 | 4 of 12 (2) |

**Gain per generation.** Each top-up raises the total ESS by about 30–40%. Individual bins move both ways: Kish ESS falls where the new generation adds a few heavy draws.

**Grid choice.** On gens 23–30, 4 orbit candidates finally have every big bin at ESS_b ≥ 30 at both evaluation points (gens 23–29: none). **None of them has a G split.** The acceleration grid reaches G 4 × d 4.

## Refit results

The control grid best isolates the generation-30 effect, because the bins are fixed:

| | 18-bin grid, gens 23–29 (noW) | 18-bin grid, gens 23–30 (noW) | gens 23–30 cmdW | refit-3 grid, gens 23–30 (noW) |
|---|---|---|---|---|
| starts agreeing (Δ < 5) | 3 / 12 | **9 / 12** | 9 / 12 | 9 / 12 |
| at bound | none | none | a1 | none |
| ln A | +0.12 | +0.70 | +0.86 | +0.13 |
| α_lo / α_hi | +1.21 / +0.21 | +1.46 / −0.10 | +2.26 / −0.68 | −0.70 / +0.74 |
| γ_P | −0.75 | −1.05 | −0.95 | −0.85 |
| ln L_P | +1.13 | +1.37 | +1.29 | +1.39 |
| Δγ_q / ln F_twin | +0.25 / +0.38 | +0.90 / +1.29 | +0.28 / +1.29 | +0.70 / +1.05 |
| Beta a2, b2 (log P 2–2.6) | 1.04, 1.14 | **1.27, 1.37** | 1.04, 1.13 | unidentified (2 e bins) |
| Beta a3, b3 (log P ≥ 2.6) | 1.28, 2.36 | **1.24, 2.49** | 1.25, 2.41 | unidentified |
| Beta range 1 (log P < 2) | unidentified | unidentified | a1 at bound | unidentified |
| N_s (prior 27.5k ± 5.2k) | 13.6k (−2.7σ) | **13.7k (−2.6σ)** | 13.9k (−2.6σ) | 24.5k (−0.6σ) |
| (mock + spurious) / real; accelerations | 0.98; 0.99 | 1.00; 0.99 | 1.01; 0.99 | 0.99; 1.00 |
| counts vs d: < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc | 1.08 / 0.97 / 0.92 / 1.03 | 1.19 / 0.94 / 0.90 / **1.74** | | 1.27 / 0.89 / 0.88 / 1.72 |
| orbit:accel (real 0.475 / 0.375 / 0.357 / 0.425) | 0.541 / 0.370 / 0.321 / 0.466 | 0.543 / 0.369 / 0.328 / 0.570 | | 0.495 / 0.375 / 0.316 / 0.488 |
| e < 0.1 (real 14,230) | 16,038 | **16,432** | | 20,177 |

**G shape** in the six-panel, (mock + spurious) / real per magnitude: G 8–9 ≈ ×2.4, G 15–17 ≈ ×0.65, G 11–15 ≈ 0.9–1.06. The orbit likelihood never sees G, because no orbit grid with a G split is ESS-adequate.

**The re-chosen 12-bin grid on gens 23–30** gives a less stable fit: only 6 of 12 starts agree, a1 is at its bound, and noW and cmdW disagree. With only 2 e bins, the Beta parameters run to a2 ≈ 4.9, b2 ≈ 9.8. The fit then lands where the draws are thin (orbit ESS 447), and e < 0.1 drops to 9.5k. Use the control grid, which keeps 3 e bins, for comparisons.

## Where the limits are now

**Converged and robust** (fixed 18-bin grid, gens 23–30, noW ≈ cmdW):
- γ_P ≈ −1.0 and ln L_P ≈ +1.3 (more long-P companions than MdS17);
- Beta(e / e_max) with a ≈ 1.0–1.3 and b ≈ 1.1–1.4 for log P 2–2.6, and a ≈ 1.25, b ≈ 2.4–2.5 for log P ≥ 2.6;
- totals and accelerations at 1.0 ± 0.01;
- e < 0.1 at 1.15× real.

**Sampling limits** (the next targets for draws):
1. **The high-e, long-P cells.** e 0.6–1 and log P 2.6–2.92 at every distance sit at ESS_b ≈ 21–22. These are the three big bins below 30 on the control grid. The projection to bring every big bin to ESS_b ≥ 30 at this fit is ≈ 90 CPU-h for 90% of big bins and ≈ 490 CPU-h for all.
2. **A G split for orbits** is still not ESS-adequate. Bright, far systems (G < 12 at d > 0.7 kpc) are the weakest cells.

**Model or data limits** (more draws will not fix these):
1. **N_s sits at −2.6σ on the fine grid but −0.6σ on the coarse grids.** The template assumes contamination follows the real (G, d) counts and the (log P, e) shape of 300 systems in one bin; that does not match the finer structure. This is the template and prior form (MP-Q43).
2. **α_lo and α_hi depend on the Malmquist weight** (noW +1.46 / −0.10 against cmdW +2.26 / −0.68). They need the CMD-Malmquist closed-loop decision (§11.5), not more draws.
3. **G 8–9 is about ×2.4 too high.** Bright primaries at those distances are mostly evolved, so this points to the giant / evolved-primary treatment (MP-Q28), not to sampling.
4. **Beta range 1 (P < 100 d) is unidentifiable.** Real orbits there are few, and the ESS forces log P < 2.6 into one cell. Ryan declined fixing it to MdS17; it stays free and unconstrained.
5. **The far bin (d > 1.5 kpc, 3,059 real orbits) overshoots ×1.7** on the fixed grids, in a cell of the coarse d binning. It is ESS-limited and also sensitive to the spurious d-shape.

The extra stage is used, and no further top-up runs without Ryan.
