# #391: the bright G < 10 excess and the distant short-P cells (Ryan, 2026-10-18; analysis only)

- **Inputs.** Best fits on gens 33–38, coarse grid (`../resim_batch5/new_{noW,cmdW}`), evaluated with `scripts/rung3_refit5_391.py fit --dump-from`, which does no optimisation.
- **Outputs.** `scripts/bright_excess_391.py` writes `summary.json` and `bright_excess.png`.
- All ratios are (mock + spurious) / real at the noW best fit unless stated. cmdW agrees.

## Bright excess: where it is (test a)

| G | 5–8 | 8–9 | 9–10 | 10–11 | 11–12 | 12–13 |
|---|---:|---:|---:|---:|---:|---:|
| orbits: ratio (mock ESS) | **2.59** (15) | **1.70** (33) | 1.38 (41) | 1.16 (170) | 0.97 | 0.90 |
| orbits: real count | 684 | 1,321 | 3,937 | 8,296 | 14,567 | 26,783 |
| accelerations: ratio (mock ESS) | 1.17 (30) | **1.82** (71) | 1.15 (125) | 1.08 | 0.93 | 0.90 |

- **The excess starts below G ≈ 10 and grows to about 2.6 at G < 8.** Accelerations share it at G 8–9 but are much weaker at G < 8.
- **Split of the G < 10 orbits** (mock ESS in brackets):
  - dwarfs 1.76 (51), evolved 1.35 (32);
  - log P < 2.0 → 2.56 (8); 2.0–2.4 → 1.88; 2.4–2.6 → 2.23; long P 1.2–1.3;
  - by distance: d 0–0.2 kpc 1.6–1.9, 0.2–0.3 kpc 0.86, 0.3–0.5 kpc 2.5 (14), 0.5–1 kpc 1.0.
- **The mock bright orbits are intermediate-mass stars.** At G < 10, 45% of the mock orbit weight has M1,ZAMS > 1.5 M⊙ (29% at 2–3 M⊙), against 14% at G 10–14.
- **Detected-orbit properties at G < 9 match** (mock weighted vs real):
  - a0, 10/50/90th percentiles: 0.62/1.79/4.57 mas against 0.64/1.60/4.45;
  - significance: 17/46/126 against 20/54/145.
- **The mock's extra near-threshold share is not specific to bright stars.** The share with significance < 20 is 0.17 vs 0.10 (mock vs real) at G < 9, and 0.40 vs 0.25 at G 11–13.
- So the excess is too many systems at every a0. It is not extra marginal detections from too little noise.

## Other tests

- **(b) Rung-1 re-detection** (#390 set re-injected with v2 + N2-u0; Orbital):
  - 0.94 at G < 9 (n 18), 0.80 at G 9–10 (51), 0.83 at G 10–11, 0.70 at 12–13, 0.66 at 13–15.
  - AstroSpectroSB1: 0.72–0.83, flat.
  - The mock cascade is about 1.2–1.3× more sensitive at the bright end than at G 12–15. That could contribute part of the 1.4× at G 9–10, but not 2.6×.
- **(c) Single-star RUWE tail.**
  - The mock's u0 set (600 stars per 0.25 mag) has RUWE > 1.4 in ≤ 0.3% of stars at every G, and its 90th percentile is 1.11–1.18, flat in G.
  - DR3's 90th percentile is 1.4–3.0 at G < 13. That reflects real binaries, not something the mock's single-star noise feeds through.
  - No bright-specific noise path into the RUWE > 1.4 gate. Ruled out.
- **(d) Parent sample and cuts.**
  - The parent is a `random_index` subsample, so its G < 10 rows (273) carry only Poisson error (≤ 12% per bin).
  - Pass rates of the Halbwachs IPD and C* cuts are flat in G for parent stars (IPD 0.90–0.95, C* 0.99–1.00). All real NSS orbits pass, because these cuts were DR3's NSS input selection.
  - **The parent's isochrone-M1 success rate falls at the bright end:** 0.58 at G < 8, 0.73 at 8–9, 0.76 at 9–10, against 0.94 for G > 13. That removes bright mock rows, which would make the excess underestimated, not cause it. Its symmetry with the real side (which needs only an atmosphere) is unchecked.

## Ranked candidates

1. **The population at M1 ≳ 1.5 M⊙ is too high.** These are the dwarf A/F primaries that dominate mock G < 10 orbits.
   - A single high-mass slope (α_hi +0.57 ± 0.20) has to serve both 0.8–1.5 M⊙ and 2–3 M⊙.
   - Excess is larger in dwarfs and at short P, where the MdS17 frequency of intermediate-mass stars is highest.
   - **Strongest candidate.**
2. **Bright-end cascade sensitivity** (b): 1.2–1.3×, a partial contributor at G 9–10.
3. **Ruled out:** single-star noise or the RUWE gate (c); parent sampling or cut pass rates (d); a noise floor (a0 and significance distributions match).

**Proposed next, analysis only, needing Ryan's OK before any model change:**
- Real-side M1 for G < 10 orbits and accelerations from the same isochrone machinery, compared with the mock M1 distribution. This directly tests candidate 1.
- A profile refit with the existing §12.19(b) slope break (`--m-break 1.4`) to see whether the G < 10 ratio falls toward 1 and what it does to ln L. This needs no draws.
- Check whether the bright isochrone-M1 failures are mirrored on the real side.

**Fix, if 1 is confirmed** (needs Ryan): adopt the §12.19(b) slope break, or a separate frequency term above about 1.5 M⊙, in the fit model.

## Distant short-P cells (d > 1 kpc, log P < 2.6), kept per §12.26

- **103 real C1 orbits:** 85 at G 13–19.5 and 18 at G 5–13; 86 Orbital, 17 AstroSpectroSB1 (bulk 80/20); 38% evolved.
- **They sit at the corner of both edges.** d 1.01–1.18 kpc and log P 2.39–2.48 (10th–90th percentiles); e median 0.27.
- **Clean, small orbits:**

| | these orbits | all real C1 orbits |
|---|---:|---:|
| significance, median (10th percentile) | 17 (12.7) | 25 |
| a0, median | 0.30 mas (about 0.34 AU) | 0.69 mas |
| ϖ/σ_ϖ, median | 79 | |
| RUWE, median | 1.57 | 2.36 |
| F2, median | −0.8 | +1.5 |
| spurious-model π̂, mean | 0.21 | 0.17 |

- **The full model accounts for about 23 of the 103:** 9.9 mock plus 13.3 spurious (template); Σπ̂ = 21.
- That leaves a shortfall of about 80 (4.5×) right at the detection-limit corner, which is exactly the test Ryan asked to keep.
- Their properties (low F2, RUWE near the gate, high ϖ/σ) do not look like the spurious population, which has high π̂ at low a0/σ. They look like real low-amplitude orbits that DR3 detects and the mock almost never does.
- Worth a dedicated rung-1-style check: re-inject these 103 published orbits and measure their re-detection fraction.
