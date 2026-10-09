# #391 detection vs population test: 0.7–1.5 kpc, G 13–16

This is Ryan's test from 2026-10-09. It uses analysis plus one small targeted re-injection; no proposal draws were added and no config was changed. C1 (P ≤ 0.8 × 1038 d = 830 d) is applied on both sides throughout.

- Script: `scripts/detection_vs_population_391.py`.
- Full numbers: `report.txt` and `summary.json`.
- Figures: `acceptance_mock_vs_reinjection.png` and `population_ratios.png`.
- The re-injection log is at `output/detection_vs_population_391/reinjection.jsonl` in the worktree. It is not tracked and can be regenerated from seeds.

## Setup

- **Real bin.**
  - Selection: mirror-filtered DR3 Orbital + AstroSpectroSB1 with ZP-corrected 1/ϖ in [0.7, 1.5] kpc, G in [13, 16] and published P ≤ 830 d. That gives **30,017 systems**.
  - Published a0, Ω, ω and i come from Thiele–Innes via `gaiamock.get_Campbell_elements`.
- **1. Matched mock truth.**
  - Each real system is matched by k-NN (k = 50) to gens 23–27 draws in standardized *true* space: (log a0, log P, e, G, d, |β|, |cos i|), with scales (0.1, 0.1, 0.1, 0.5 mag, 0.15 kpc, 20°, 0.1).
  - The mock's true photocentre a0 is `gaiamock.get_a0_mas`.
  - Acceptance is the unweighted fraction of neighbours that were accepted with recovered P ≤ 830 d.
  - A pooled local-linear correction for the neighbour offsets is also reported.
- **2. Targeted re-injection, current model.**
  - Sample: 300 real systems (a seeded random subset of the bin) × 3 realizations, 8 workers under `nice`, BLAS pinned.
  - Model: the published orbit goes through `predict_astrometry_binary_in_terms_of_a0` inside `epoch_model.run_cascade`, using `dr3.epoch_model` (v2 + N2d + VP loss, enabled as in the proposal runner). The visibility gate and `orbital_solution_cuts` are the same as the mock's.
  - Seeds: `injection_rng_seeds` and `epoch_model_rng(base_seed, INJECTION_RNG_STREAM, source_id, r)`.
- **Two added checks** (needed to read the result; same worker, same seeding scheme):
  - **Engine cross-check.** The 2 nearest matched mock draws per real system (506 draws) have their *truth* re-simulated through the rung-1 path above. The result is compared with their stored proposal-runner outcome.
  - **Symmetric rung 1 on the mock.**
    - Selection: for each real system, the nearest *accepted* mock draw on *recovered* quantities (218 unique draws).
    - Injection: each is re-injected at its *fitted* orbit, × 3 realizations. A, B, F, G are converted to Campbell elements, Tp = φ_p P / 2π (gaiamock's convention), and the fitted parallax is used.
    - This is the like-for-like counterpart of re-injecting real *published* (i.e. detected, fitted) orbits.

## Results

| quantity | value |
|---|---|
| mock acceptance, matched true orbits (all 30,017 / the 300) | 0.230 / 0.223 (offset-corrected 0.257 / 0.249) |
| **real re-injection acceptance**, current model | **0.599** (539/900; 1σ 0.582–0.615) |
| #390 acceptance in the same region (no epoch model; stratified set) | 0.633 (232 systems) |
| engine cross-check: matched mock truth via rung-1 path vs stored outcome | 0.285 vs 0.267 (per-draw agreement 0.86) |
| **symmetric: mock accepted orbits re-injected at their fitted orbit** | **0.754** (1σ 0.740–0.769); paired vs real **+0.156 ± 0.027** |

Re-detection by significance, with the 7/9-parameter fallback share in brackets. Real values use the published significance; mock values use the fitted one:

| a0/σ | real | mock (symmetric) |
|---|---|---|
| < 10 | 0.29 (0.24) | 0.44 (0.05) |
| 10–15 | 0.48 (0.21) | 0.63 (0.06) |
| 15–25 | 0.64 (0.15) | 0.81 (0.09) |
| > 25 | 0.78 (0.10) | 0.90 (0.05) |

By period, P 650–830 d gives real 0.49 (0.26) against mock 0.73 (0.10). The deficit in real re-detection is broad. It is worst at the longest P, at G > 14.5, at d ≈ 1.4 kpc, at |β| ≈ 15–30° and at published RUWE < 2.

## Verdict

**1. The literal test says "detection gap", but that comparison is not like-for-like.**
- Matched mock truth is accepted at 0.22–0.26, against 0.60 for real re-injection.
- The engine cross-check shows the mock and rung-1 paths give the same acceptance for the same truth (0.285 vs 0.267). So the forward model itself does not differ.
- The 0.25 → 0.60 step is selection. Real published orbits are *detected, fitted* values: the orientation, phase, noise and scan pattern happened to favour detection. Random-orientation true orbits at the same (a0, P, e, G, d, β, cos i) do not share that advantage. This holds for any forward model, including a perfect one.

**2. The like-for-like (symmetric) test shows a real but secondary detection-side difference, of opposite sign.**
- Re-injected at their fitted orbits, the mock's detected orbits are re-detected at 0.75; real detected orbits manage only 0.60.
- At fixed significance, real orbits fall back to a 7/9-parameter (acceleration) solution 2–4× as often (0.10–0.24 vs 0.05–0.09). The worst case is P 650–830 d (0.26 vs 0.10).
- So DR3 accepted orbital solutions that our cascade, faced with the same orbit, more often assigns to the acceleration branch. Possible causes:
  - the DR3 orbit/acceleration model selection is more permissive than gaiamock's acceleration-first rule at long P (cf. #399, P > 600 d capture);
  - and/or a spurious or non-Keplerian component of the real catalogue, which re-injects as marginal.
- **Factor that differs: cascade model selection** (7/9-par vs orbital). Noise is not the cause: median recovered significance is close (mock 18.4, real 17.3). Epoch counts are not the cause either: insufficient visibility is ≤ 0.2% on both sides.
- Scale: (0.75 − 0.60) / 0.75 ≈ 20% of real bin orbits sit beyond the mock's detection margin. That accounts for at most a factor ≈ 0.8 of the count ratio, not 0.45–0.5.

**3. Most of the deficit is therefore population.**
- Mock/real accepted counts in the bin are 0.51 (noW, ESS 108) and 0.44 (cmdW, ESS 94).
- After the detection-side ~0.8, roughly 0.6 remains. The mock under-produces relative to real here:
  - **High f_m:** log f_m > −1 is at 0.08 (2,625 real, 9% of the bin). This is the high-f_m component (#391 `high_fm/`), mostly spurious or class II.
  - **Primaries away from solar type:** dereddened M_G < 3 (subgiants and massive M1) is at 0.1–0.3, and M_G > 5 (M1 ≲ 0.8 M☉) is at 0.2–0.3. M_G 4–5 is at 0.78.
  - **Longest periods:** log P 2.65–2.92 (P ≈ 450–830 d) is at 0.49, against 0.7–0.9 at shorter P.
- Mock truth in the bin: detection keeps M1 ≈ 1.0–1.4 and q ≈ 0.3–0.8, and suppresses q < 0.2 and twins (q > 0.85). Almost all accepted orbits have P > 450 d.

**Bottom line.** Detection is consistent to within a factor of ≈ 0.8, and that factor comes from the long-P orbit vs acceleration model selection. The surviving 0.7–1.5 kpc deficit is mainly population:
- the mock's MdS17 companions around non-solar-type primaries;
- the high-f_m (spurious or class II) component it does not model;
- the P ≳ 450 d tail.

## Caveats

- k-NN matching uses neighbour distances of about 0.6 scale units per dimension. The pooled local-linear correction moves acceptance by only +0.03.
- The symmetric test matches one accepted draw per real system (218 unique). The mock accepted set in the box is sparse.
- Population ratios rest on a bin ESS of about 100.
- Mock F2 is uncalibrated (see `spurious_cut/`). The F2 cut never fails on either side here.
