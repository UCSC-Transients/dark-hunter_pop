# #391: symmetric spurious-cut diagnostic (analysis only)

The script is `scripts/spurious_cut_diagnostic_391.py`; every number is in `report.txt`.

Each cut is applied identically to two samples:
- **Mock:** the accepted orbits of generations 23–27. Unless stated, there is no Malmquist weight.
- **Real:** the 167,911 orbits after the mirror filters, using parallaxes corrected for the zero point.

Thresholds were fixed **from the real data only**, before any comparison with the mock. Each threshold is where the empirical CDFs of the high-f_m (> 0.1) and low-f_m orbits separate most (the KS point). It was measured on real orbits at d > 0.7 kpc with P ≤ 830 d.

The cuts:
- **C1:** drop P > 0.8 × 1038 d.
- **C2:** C1, plus drop F2 > 1.74 (high-f_m orbits are enriched above it).
- **C3:** C1, plus drop significance > 26.4 (high-f_m orbits are enriched above it).

## Results

The table uses no Malmquist weight. The KS results, with and without the 2-D CMD weight and for all orbits and for dwarfs, are in `report.txt`.

| Cut | Real kept | Mock kept (weight) | Total mock/real | mock/real at 0.7–1 / 1–1.5 kpc | Mean e, real / mock | ESS |
|---|---|---|---|---|---|---|
| C0 (none) | 1.000 | 1.000 | 0.65 | 0.49 / 0.40 | 0.339 / 0.440 | 881 |
| C1 | 0.781 | 0.789 | 0.66 | 0.52 / 0.41 | 0.331 / 0.420 | 704 |
| C2 | 0.408 | **0.285** | 0.45 | 0.44 / 0.31 | 0.323 / 0.392 | 373 |
| C3 | 0.413 | 0.517 | 0.81 | 0.62 / 0.43 | 0.377 / 0.464 | 401 |

**Sensitivity:** varying each threshold by ±20% changes the 0.7–1.5 kpc ratios by ≲ 0.1 for C1 and C3, and by < 0.05 for C2. Panel 2 of the figures shows this.

**Six-panel KS:** reported only on bins with ESS ≥ 30. Values are D and p, no Malmquist weight, all orbits.

| Panel | C0 | C1 |
|---|---|---|
| P | 0.045 (p 0.05) | 0.049 (p 0.06) |
| G | 0.076 (p 1e-4) | **0.050 (p 0.08)** |
| 1/ϖ | 0.091 (p 4e-7) | 0.079 (p 2e-4) |
| e | 0.155 | 0.131 |
| f_m | 0.059 | 0.054 |
| cos i | agrees | agrees |

The eccentricity mismatch stays highly significant (p ≈ 1e-12) after every cut.

## Verdict

- **C1 is symmetric and safe.** It removes about 22% on both sides and improves the G and P shapes. It does **not** reduce the distance deficit (0.49 → 0.52 at 0.7–1 kpc; 0.40 → 0.41 at 1–1.5 kpc). It cuts the eccentricity excess from +0.10 to +0.09.
- **C2 is not usable.** It removes far more mock weight than real (0.715 vs 0.592) and empties the mock at G < 12. The mock's F2 is not calibrated against DR3:
  - the #390 injection test found an offset of about −1.2 at fixed orbit;
  - the bright-star noise term raises the mock's F2 at the bright end.

  A cut on F2 therefore needs the mock's F2 distribution calibrated first.
- **C3 is partly symmetric** (real 0.41, mock 0.52 kept). It raises the total ratio to 0.81, mostly by removing nearby large-orbit systems, where the mock now *exceeds* the real count (1.4 at < 0.2 kpc). It leaves the 0.7–1.5 kpc deficit (0.62 / 0.43) and the eccentricity excess in place.
- **What survives every cut:**
  - a deficit concentrated at **0.7–1.5 kpc and G ≈ 13–16** (mock/real 0.4–0.6), flat in sky position, crowding and scan coverage (docs/gate391/distance_count);
  - a uniform eccentricity excess of +0.07 to +0.09.

  After C1 the real high-f_m share is still higher than the mock's at 0.4–1.5 kpc (real 0.075–0.131, mock 0.026–0.083). But the deficit is much larger than the high-f_m share, so it is **not class-II-shaped**. It is a broad shortfall in detectable orbits that grows with distance and faintness, in low-f_m orbits too.

## Recommendation

1. **Spuriousness term.** Adopt C1 (P ≤ 0.8 × baseline) on both sides as the minimal form; it is cheap and symmetric. A model-based P(spurious) weight is **not** the lever for the count deficit, and an F2-based term needs the mock F2 calibrated first (a new issue).
2. **Surviving distance/G deficit.** This is either a rung-3 population offset (MdS17 frequency or q at the M1 and P of distant G 13–16 stars) or a residual detection-model gap at faint G near threshold. Distinguish the two before rung 3 with a **targeted gaiamock check**. Rung 1 (#390) is already the right tool: compare its re-injection acceptance at G 14–16, d 0.7–1.5 kpc with the mock's acceptance for matched true orbits.
3. **Eccentricity excess:** population (MdS17 η). Fit it at rung 3.
