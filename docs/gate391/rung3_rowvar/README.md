# #391 rung 3, 108-bin grid, gens 23–32: M1,ZAMS on, slope break off, MC variance draw vs parent row (#488)

Config: `config/population/rung3_compliant108_v2.yaml`. Fitter: `scripts/rung3_refit5_391.py`, with the optimizer guard.

| fit | −lnL | α_lo | α_hi | ln f_evo | G 5–9 / 9–10 | starts within 5 |
|---|---|---|---|---|---|---|
| noW, draw | 759.5 | +0.34 ± 0.18 | +0.43 ± 0.08 | −0.02 ± 0.06 | 1.55 / 1.21 | 12/12 |
| noW, parent row | 754.1 | +0.68 ± 0.25 | +0.28 ± 0.10 | −0.01 ± 0.07 | 1.53 / 1.21 | 12/12 |
| cmdW, draw | 766.0 | +0.26 ± 0.18 | +0.18 ± 0.09 | −0.11 ± 0.06 | 1.53 / 1.17 | 8/24 |
| cmdW, parent row | 759.6 | +0.62 ± 0.27 | +0.07 ± 0.11 | −0.08 ± 0.06 | 1.53 / 1.17 | 5/24 |

**Inflation from parent-row variance.**
- Median relative MC error per bin (row / draw):
  - G 5–13, d < 0.5 kpc: ×1.12–1.14;
  - G 5–13, d 0.5–1 kpc: ×1.12;
  - elsewhere: ×1.00–1.07.
- In every region the MC error is still about 4–7 × the Poisson error of the real counts. On this grid the draw count, not the number of parent rows, limits the fit.
- Parameter errors: α_lo ×1.44–1.49, α_hi ×1.16–1.22, the others ×0.85–1.13.

**Cut proposal (not applied).** On the 108-bin grid no cut is needed: the row term is a small, accounted systematic. The parent-row limit binds only in sub-cells such as G 8–10 at d 0.2–0.5 kpc (61 usable rows, 23 in the clump box). If a G < 10 bin is ever added, that cell should either carry the row variance or be cut at G < 8–9, not fitted on draw variance alone.
