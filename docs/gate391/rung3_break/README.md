# #391 slope break (§12.19 b) and the PM13 mass scale (§12.19 c): diagnostics only, break OFF by default

Config: `config/population/rung3_compliant108_break.yaml` (M1,ZAMS on). cmdW is from the guarded re-run (24 starts). The noW M_br profile used 4 starts.

| fit | −lnL | α_hi | α_br | G 5–9 |
|---|---|---|---|---|
| noW, MIST, M_br 1.4 | 759.1 | +0.20 ± 0.26 | +0.79 ± 0.83 | 1.63 |
| noW, PM13, M_br 1.4 | 758.3 | +0.24 ± 0.25 | +1.21 ± 0.80 | 1.63 |
| cmdW, MIST | 765.6 | −0.15 ± 0.29 | +1.06 ± 0.86 | 1.62 |
| cmdW, PM13 | 763.7 | −0.08 ± 0.28 | +1.36 ± 0.83 | 1.63 |

**M_br profile (noW, MIST):** −lnL 758.9 / 759.1 / 759.2 / 759.4 at M_br = 1.2 / 1.4 / 1.7 / 2.0. It is flat (Δ < 0.5), and the break is unconstrained at every M_br.

**For reference without the break:**
- noW: 761.3 before M1,ZAMS; 759.5 with M1,ZAMS (`../rung3_rowvar`, draw variance).
- cmdW: 766.04, evaluated at the old best.
- So the break gains Δ ln L ≈ 0.4–1.

**Conclusion.** α_br is about 1σ from 0, positive (the wrong sign for the bright excess), and the G 5–9 ratio does not move. The §12.14 grid has no G < 10 resolution. Kept off until it is testable.
