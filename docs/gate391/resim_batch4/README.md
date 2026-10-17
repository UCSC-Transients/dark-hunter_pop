# #391 fresh set, gens 33–36 (about 154 CPU-h): coarse-grid (§12.26) refits after batch 4

- Gen 36: 250k draws, 34.4 CPU-h, 1,052 orbits (30.5 per CPU-h).
- Gens 33–36 weights: noW orbit Kish ESS 847 (maximum weight share 0.0056); cmdW 434 (0.0225).
- Fits: `new_noW/`, `new_cmdW/`.

**The weak and empty cells mostly did not fill.**
- Coarse orbit bins with no mock draws: 8 → 7. The real orbits in them drop from 172 to 68, because one bin (G 13–19.5, d < 1, short P, e > 0.3, evolved) got ESS 1.
- The remaining empty bins are all **d 1–6 kpc at log P < 2.6**: the 4 G 5–13 cells and 3 G 13–19.5 cells.
- Big bins below ESS_b 30: 7/15 → 5/15 (noW).
- Per-cell ESS before (gens 33–35) → after (gens 33–36):
  - G 5–13 / d < 1 / long-P evolved: 23 → 34 and 28 → 38.
  - G 13–19.5 / d > 1 / long-P: dwarf 24 → 35; evolved 8 → 14 and 7 → 12.
  - The short-P evolved cells did not move (3.7 → 3.5, 11.5 → 10.3). Neither did the G 5–13 / d > 1 / long-P evolved cells (7.2 → 5.8, 7.0 → 6.9), even though their parent rows grew by 10–20%.
- These cells are limited by weight dispersion (a few heavy parent rows dominate), not by the number of draws.

**Fit results.**

| | noW 33–35 | noW 33–36 | cmdW 33–35 | cmdW 33–36 |
|---|---|---|---|---|
| α_lo | +0.57 ± 1.44 | +0.75 ± 1.19 | +5.0 (bound) | +5.0 (bound) |
| α_hi | +0.33 ± 0.25 | +0.36 ± 0.22 | +0.41 ± 0.30 | +0.36 ± 0.25 |
| γ_P | −0.77 ± 0.51 | −0.44 ± 0.36 | −1.59 ± 0.75 | −0.97 ± 1.11 |
| ln f_evo | +0.16 ± 0.12 | +0.28 ± 0.08 | +0.21 ± 0.15 | +0.25 ± 0.08 |
| ln F_twin | +0.46 | −4.0 (bound) | +0.09 | −4.0 (bound) |
| G 5–9 / 9–10 / 15–16 / 16–19.5 | 2.64 / 1.57 / 0.81 / 0.77 | 2.28 / 1.52 / 0.78 / 0.81 | 2.91 / 1.94 / 0.63 / 0.25 | 2.40 / 1.50 / 0.57 / 0.30 |
| d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 | 1.07 / 0.95 / 0.90 / 1.00 | 1.02 / 1.02 / 0.93 / 1.13 | 0.94 / 1.02 / 0.92 / 1.05 | 0.88 / 1.17 / 0.94 / 1.10 |
| **orbit:accel at d > 1.5 (real 0.425)** | 0.113 | **0.160** | 0.185 | **0.152** |

−lnL is not comparable across these fits: the set of empty, floored bins differs (8 vs 7).
