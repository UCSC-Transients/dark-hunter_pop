# #391 fresh set, gens 33–38: batch 5 (gen 38) and coarse refits

- **Gen 38** (`config/population/proposal_set_batch38.yaml`):
  - 250k draws, 49.0 CPU-h; 2,527 orbits (51.6 per CPU-h).
  - Proposal: the §12.28 evolved proposal; a 35% parent target on G 5–13 at d > 0.5 kpc; evolved_weight 0.5; short-P core 0.6.
  - The decided-settings guard passed.
- **Gen 37** is the 4.1 CPU-h evolved-only pilot (`../evolved_pilot37/`). It is included in the set.
- **Evolved check, gen 38 alone** (`evolved_check_gen38.json`):
  - 2,487 evolved-detected draws, Kish ESS 77.6.
  - 3.0 per evolved-draw CPU-h, which is 7× gens 33–36 (10× on total CPU).
  - Lower than the pilot's 10.7, because gen 38's evolved draws are spread over the targeted and defensive components.
  - Flux-floor weight share ≈ 0. Top-5 share 0.16.

## Mixture ESS (`sanity_33_38.txt`; noW, cmdW in brackets)

| | gens 33–36 | gens 33–38 |
|---|---:|---:|
| detected, Kish ESS | 730 [512] | 2,092 [1,362] |
| orbits | 847 [435] | 1,033 [559] |
| orbits d > 1.5 kpc: ESS (Σw) | 18.2 (1,327) | 23.8 (2,358) |
| accelerations d > 1.5 kpc: ESS (Σw), top-1 share | 2.2 (11,763), 0.67 | 39.3 (4,355), 0.12 |

- The d > 1.5 kpc acceleration estimate was one evolved draw.
- With it diluted, mock-only orbit:accel at d > 1.5 is 0.54. The fit, with spurious orbits added, gives 0.397 (noW) and 0.430 (cmdW), against 0.425 real.
- That number is now measurable: about 24 orbit ESS and 39 acceleration ESS, so roughly ±25% relative.

## Per-cell ESS, stuck cells (noW; cmdW within about 1 of it except where noted)

| cell (G / d / log P / e / class) | real | 33–36 | 33–38 |
|---|---:|---:|---:|
| 5–13 / < 1 / short / < 0.3 / evolved | 912 | 3.5 | 10.1 |
| 5–13 / < 1 / short / > 0.3 / evolved | 998 | 10.3 | 13.4 |
| 5–13 / 1–6 / long / < 0.3 / evolved | 1,648 | 5.8 | 11.7 |
| 5–13 / 1–6 / long / > 0.3 / evolved | 1,400 | 6.9 | 19.0 |
| 13–19.5 / 1–6 / long / < 0.3 / evolved | 1,299 | 13.6 | 10.1 (cmdW 13.3 → 6.0) |
| 13–19.5 / 1–6 / long / > 0.3 / evolved | 1,470 | 12.2 | 13.2 |
| 13–19.5 / 1–6 / short / < 0.3 / dwarf | 35 | 1.0 | 1.0 |
| the other 7 d 1–6 / short cells | 68 total | empty | empty |

- **The distant short-P cells are a detectability limit, not a sampling gap.**
  - The 1.1M draws give only 6 mock C1 orbits at d > 1 kpc with log P < 2.6, against 1–3 per generation before gen 38. That is despite gen 38 putting 37% of its draws at d > 1 kpc with short P.
  - The minimum log P of mock d > 1 kpc orbits is 2.42; the 1st percentile is 2.64.
  - More draws will not fill these cells. The 68 + 35 real orbits there are most likely spurious, or have a0 outside the mock's support.
  - Recommendation: merge these cells into their long-P neighbours, or leave them to the spurious template. Do not top them up.

## Fits (coarse grid, §12.26)

| | noW 33–36 | noW 33–38 | cmdW 33–36 | cmdW 33–38 |
|---|---|---|---|---|
| α_lo | +0.75 ± 1.19 | −0.23 ± 0.73 | +5.0 (bound) | +2.6 ± 3.9 |
| α_hi | +0.36 ± 0.22 | +0.57 ± 0.20 | +0.36 ± 0.25 | +0.55 ± 0.25 |
| γ_P | −0.44 ± 0.36 | −0.54 ± 0.35 | −0.97 ± 1.11 | −1.39 ± 0.59 |
| ln L_P | +1.09 ± 0.30 | +1.17 ± 0.24 | | +1.86 ± 0.45 |
| dγ_q | −0.31 ± 0.27 | −0.47 ± 0.20 | | −0.52 ± 0.23 |
| ln F_twin | −4.0 (bound) | −4.0 (bound) | −4.0 (bound) | −4.0 (bound) |
| ln f_evo | +0.28 ± 0.08 | +0.15 ± 0.07 | +0.25 ± 0.08 | +0.13 ± 0.07 |
| G 5–9 / 9–10 / 15–16 / 16–19.5 | 2.28 / 1.52 / 0.78 / 0.81 | 2.00 / 1.38 / 1.00 / 1.03 | 2.40 / 1.50 / 0.57 / 0.30 | 2.08 / 1.24 / 0.59 / 0.35 |
| d < 0.3 / 0.3–0.7 / 0.7–1.5 / > 1.5 | 1.02 / 1.02 / 0.93 / 1.13 | 1.15 / 0.94 / 0.89 / 1.14 | 0.88 / 1.17 / 0.94 / 1.10 | 0.91 / 1.08 / 0.89 / 1.11 |
| orbit:accel at d > 1.5 (real 0.425) | 0.160 | **0.397** | 0.152 | **0.430** |
| orbit ESS; big bins below 30 | 847; 5/15 | 737; 5/15 | 434; 7/15 | 147; 8/15 |

- **cmdW is still weight-limited.** Its orbit ESS in the fit is 147, against 737 for noW. Its faint G deficit (0.35 at G 16–19.5) and its γ_P, ln L_P differ from noW by 2σ.
- noW is the one to read for now.
