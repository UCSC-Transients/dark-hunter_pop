# #391: θ-dependent CMD companion conditioning (§12.22 amendment) and MP-Q27 (§12.23), 108-bin grid, gens 23–32

All runs use M1,ZAMS, parent-row MC variance and no slope break. Contact sheet: `contact_sheet.png`.

Reports: `noW/`, `cmdW/`, the α_lo profiles `prof_alo*` (with conditioning) and `prof_nocond_alo*` (without), `../rung3_mpq27/` (the `none`/`hold` brackets) and `../rung3_mpq27_orb6/` (the ORB6-calibrated run). All runs finished cleanly.

Two operational notes:
- The superseded fixed-baseline v3 cmdW diagnostic was stopped at Ryan's request, so there is no `../rung3_cmdcond/cmdW`.
- `chain_v4` was relaunched as `chain_v4b` (cmdW_none starts from `rung3_rowvar/cmdW_row`).

| fit | −lnL | α_lo | α_hi | G 5–9 / 9–10 / 10–12 | d < 0.3 kpc (mock+spurious)/real | orbit:accel at d < 0.3 kpc (real 0.475) | converged starts |
|---|---|---|---|---|---|---|---|
| noW, no conditioning (`../rung3_rowvar`) | 754.1 | +0.68 ± 0.25 | +0.28 ± 0.10 | 1.53 / 1.21 / 0.98 | 1.06 | 0.477 | 12/12 |
| noW, θ-conditioning | 757.2 | **+4.86 ± 0.69** | +0.28 ± 0.11 | 1.45 / 1.20 / 1.02 | 1.07 | 0.502 | 12/12 |
| noW, + MP-Q27 `none` | 759.3 | +4.64 | +0.28 | 1.52 / 1.18 / 1.02 | 1.09 | 0.510 | 12/12 |
| noW, + MP-Q27 ORB6 | 750.7 | +4.62 | +0.28 | 1.44 / 1.17 / 1.02 | 1.02 | 0.476 | 10/12 |
| noW, + MP-Q27 `hold` | 750.3 | +4.64 | +0.28 | 1.50 / 1.19 / 1.03 | 1.02 | 0.476 | 12/12 |
| cmdW, θ-conditioning (= MP-Q27 `none`) | 766.9 | +3.36 ± 0.59 | +0.74 ± 0.14 | 1.79 / 1.27 / 1.28 | 1.22 | 0.572 | 24/24 |
| cmdW, + MP-Q27 ORB6 | 763.8 | +3.09 | +0.74 | 1.81 / 1.27 / 1.27 | 1.20 | 0.556 | 24/24 |

**The CMD conditioning is not usable as built: below 0.7 M⊙ it is dominated by isochrone systematics (the §12.22 stop rule).**
- The α_lo profile has a minimum near 0.68 without conditioning. With conditioning, −lnL keeps falling all the way to α_lo ≈ 4.9.
- For drawn orbits with M1 < 0.7 M⊙ the mean weight factor is ×3–4 (median LR(q) 1.3–1.9). For M1 ≥ 0.8 it is about 1.0–1.1 (noW).
- A large α_lo exists to cancel that factor.
- The likely cause: real K/M dwarfs sit above or redward of the MIST single-star main sequence, so the composite (binary) map fits them better.

**Blue evolved stars** (real fraction across all evolved C1 orbits: 0.282). The mock fraction rises from 0.189 to 0.229 with conditioning (noW). In the bright cell (G 8–10, d 0.2–0.5 kpc) it rises from 0.068 to 0.107, against a real 0.240.

**MP-Q27 effect.**
- Accelerations at d < 0.1 kpc pass at a mean rate of 0.976 (`none`), 0.847 (ORB6) and 0.751 (`hold`), MdS17-weighted. Everywhere else the rate is ≥ 0.9998.
- In noW the IPD weight improves −lnL by 8.6 (759.3 → 750.7).
- It also restores the near-distance orbit:acceleration ratio, 0.510 → 0.476 against a real 0.475.
- ORB6 and `hold` are equivalent in the fit (Δ−lnL 0.4). In cmdW the improvement is 3.1.

**noW vs cmdW.** With conditioning, cmdW is worse at bright G and near distance (G 10–12 at 1.28; d < 0.3 kpc at 1.22). α_hi differs (+0.28 vs +0.74).
