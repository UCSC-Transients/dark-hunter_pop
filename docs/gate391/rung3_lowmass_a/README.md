# #391: low-mass CMD conditioning fixes (§12.22 second amendment) and the MIST mass-scale checks (§12.24)

All fits use M1,ZAMS, parent-row variance, θ-dependent conditioning and MP-Q27 = ORB6, on the 108-bin grid. Contact sheet: `contact_sheet.png`. Variant (b) is in `../rung3_lowmass_b/`.

| fit | −lnL | α_lo | α_hi | G 5–9 / 9–10 / 16–19.5 | d < 0.3 kpc | orbit:accel at < 0.3 kpc (real 0.475) | converged starts |
|---|---|---|---|---|---|---|---|
| no conditioning, no ORB6 (`../rung3_rowvar`) | 754.1 | +0.68 | +0.28 | 1.53 / 1.21 / 1.13 | 1.06 | 0.477 | 12/12 |
| MIST locus + ORB6 (`../rung3_mpq27_orb6`) | 750.7 | +4.62 | +0.28 | 1.44 / 1.17 / 0.94 | 1.02 | 0.476 | 10/12 |
| **(a) M1 ≥ 0.8, noW** | **747.6** | +0.76 ± 0.28 | +0.28 ± 0.11 | 1.46 / 1.20 / 1.11 | 1.02 | 0.468 | 12/12 |
| (a), cmdW | 765.7 | +0.09 ± 0.37 | +0.53 ± 0.14 | 1.73 / 1.25 / 1.23 | 1.21 | 0.540 | 4/24 |
| (b) ridge locus, noW | 790.2 (not converged; the profile at α_lo = 0.68 reaches 762.8) | −0.17 | +0.05 | 1.44 / 1.16 / 1.75 | 1.08 | 0.453 | 2/12 |
| (b), cmdW | 767.4 | −0.36 | +0.70 | 1.85 / 1.29 / 1.44 | 1.21 | 0.538 | 2/24 |

**α_lo profiles.** Each uses 4 starts; values are −lnL at α_lo = 0 / 0.68 / 1.5 / 3.0.
- (a): 752.5 / 747.6 / 750.3 / 763.1. The minimum is at about 0.7, the same place as without conditioning.
- (b): 791.0 / 762.8 / 773.6 / 793.6.

**Conditioning factor vs M1** (noW mean, M1 bins 0.1–0.5 / 0.5–0.7 / 0.7–0.8 / 0.8–1.0 / 1.0–1.4 / 1.4–3 M⊙):
- MIST locus: 4.2 / 3.1 / 1.3 / 1.06 / 0.98 / 1.05
- (a): 1 / 1 / 1 / 1.06 / 0.98 / 1.06
- (b): 0.97 / 0.54 / 0.66 / 1.00 / 0.98 / 1.07. (b) overcorrects at 0.5–0.8 M⊙.

The cmdW fits show a larger mean factor at 0.8–1.0 M⊙ (×1.7–1.8). A few large Malmquist-weighted draws dominate it, so cmdW converges poorly.

**Blue evolved fraction** (real: 0.282 over all evolved orbits; 0.240 in the bright cell G 8–10, d 0.2–0.5 kpc):

| | all evolved | bright cell |
|---|---|---|
| no conditioning | 0.189 | 0.068 |
| (a) | 0.228 | 0.104 |
| (b) | 0.242 | 0.123 |

**Verdict.** (a) is the best fit so far (−lnL 747.6) with a physical α_lo. (b) fits worse at faint G and does not converge. Keep (a) as the working model until the MIST correction (§12.24) is in place.
