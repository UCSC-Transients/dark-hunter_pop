# #509: variant (a) v6a refit on the re-truthed gens 23–32 cache (deblended M1 for gens 29/31/32)

The cache is `output/proposal_set/weights_gen23_32_retruth.npz`, built on uncorrected MIST (as those generations ran). The fits use `--lowmass-correction-off`. Each fit ran alone.

| | old v6a (`../rung3_lowmass_a`) | corrected (`noW/`) |
|---|---|---|
| −lnL | 747.6 | 763.7 (not directly comparable: different cache) |
| α_lo | +0.763 ± 0.282 | +0.777 ± 0.290 |
| α_hi | +0.282 ± 0.106 | +0.239 ± 0.112 |
| ln f_evo | +0.015 | −0.182 ± 0.097 |
| G 5–9 / 9–10 / 10–12 / 15–16 / 16–19.5 | 1.46 / 1.20 / 1.01 / 0.84 / 1.11 | 1.58 / 1.25 / 1.02 / 0.88 / 1.02 |
| d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 kpc | 1.02 / 0.98 / 0.97 / 1.24 | 1.06 / 0.97 / 0.95 / **1.55** |
| orbit:accel at d > 1.5 kpc (real 0.425) | 0.457 | **0.616** |
| α_lo profile, −lnL at 0 / 0.68 / 1.5 / 3.0 | 752.5 / 747.6 / 750.3 / 763.1 | 768.5 / 763.7 / 766.1 / 779.0 |
| starts converged | 12/12 | 12/12 |

Peak RSS per fit (`/usr/bin/time -l`): 11.8 GB (12-start noW); 9.6 / 12.4 / 9.3 / 9.1 GB for the four profile fits.

**Earlier decisions:**
- **Variant (a): holds.** The profile minimum is still at α_lo ≈ 0.7.
- **Slope break: unchanged and still untestable.** The §12.14 grid has no G < 10 split, and the G 5–9 excess persists (1.58).
- **Clump / bright-evolved excess: persists**, slightly larger at G 5–9. ln f_evo moves to −0.18.
- **MP-Q27 ORB6: holds.** d < 0.3 kpc 1.06 and orbit:accel 0.468 against a real 0.475.
- **New: the distant bin (d > 1.5 kpc) gets worse.** (mock + spurious)/real goes 1.24 → 1.55, and orbit:accel 0.46 → 0.62 against a real 0.43. Gens 31/32 carry most of the distant draws, so the deblended M1 matters most there. This needs a look before the fresh set (gens 33+) is judged on the same bin.
