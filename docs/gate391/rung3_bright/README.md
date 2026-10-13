# #391 rung 3: bright-G re-injection (Ryan, 2026-10-13, option A)

- **Run.** 130 real C1 orbits at G 5–10 (all distances) × 3 realizations, plus 122 matched mock accepted orbits × 3, using the current forward model. 10 workers under `nice`, BLAS pinned, with a PID-only budget cap of 3 CPU-h.
- **Cost:** 2.89 CPU-h.
- **Files:** `reinjection/` (report, figures) and `split.txt`. The log is `output/detection_vs_population_391/reinjection_bright.jsonl`, untracked and regenerable from its seeds.

## Detection

- Real orbits re-detect at **0.844** (329/390) and mock orbits at **0.928**. The paired difference is +0.085 ± 0.029, so the excess failure is π ≈ 0.09.
- That is **lower** than at fainter G (0.17–0.21). It matches the calibrated π, which depends only on significance, and bright orbits are high-significance.

| split | real acceptance (n) | mock acceptance (n) | excess failure |
|---|---|---|---|
| G 5–8 | 0.90 (16) | 0.92 (12) | +0.02 |
| G 8–9 | 0.82 (26) | 0.98 (19) | +0.16 |
| G 9–10 | 0.84 (88) | 0.89 (64) | +0.06 |
| dwarf | 0.89 (85) | 0.94 (92) | +0.06 |
| evolved | 0.76 (45) | 0.88 (30) | +0.13 |
| d < 0.1 kpc | 0.90 (24) | 0.95 (22) | +0.05 |
| 0.1–0.2 kpc | 0.95 (29) | 0.93 (46) | −0.02 |
| 0.2–0.5 kpc | 0.82 (57) | 0.92 (42) | +0.10 |
| > 0.5 kpc | 0.67 (20) | 0.83 (12) | +0.20 |

**Noise level.** This compares recovered with published uncertainty for re-injected real orbits with orbital outcomes. A symmetric test cannot see a mis-set noise level, because both sides go through the same engine; this comparison can.

| sample | recovered / published significance | recovered / published parallax error |
|---|---|---|
| bright, G < 10 | 0.71 | 1.18 |
| near, G 5–10 | 0.74 | 1.10 |
| near, G 10–13 | 0.74 | 1.17 |
| near, G 13–20 | 0.93 | 1.10 |
| far, 0.7–1.5 kpc, G 13–16 | 0.90 | 1.07 |

**At G < 13 the forward model's noise is, if anything, higher than DR3's.** Part of the significance ratio below 1 is the selection bias in published significances (they are biased high). Higher mock noise would make the mock detect **fewer** bright orbits, so it cannot produce the ×1.2–1.6 bright *excess*. Detection is not the cause.

## Recommendation (not implemented; Ryan's call)

**Option B.** The excess sits in G 8–10 F/A dwarfs at 0.1–0.5 kpc (×1.4–1.5; primaries of about 1.3–2 M⊙) and in bright giants (×1.8). Detection is consistent and the reddening and local-density causes are excluded, so the remaining candidate is the population at high M1.
- A **break in the frequency slope above about 1.4 M⊙** (a third α) is reweightable, invalidates no draws and costs no draws.
- Check **isochrone M1 for A stars** first: rotation, and the MIST grid at hot temperatures. Cheap, and analysis only.

**Option C** (accept and document G < 10, about 4% of the C1 orbits) is the fallback if the break is not constrained or is not justified physically.
