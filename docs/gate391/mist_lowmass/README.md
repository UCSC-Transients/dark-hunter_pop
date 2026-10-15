# #391 spec §12.24: MIST low-mass correction, calibrated but NOT enabled (part of the §12.25 re-simulation batch)

- Script: `scripts/calibrate_mist_lowmass_391.py`
- Calibrated nodes: `calibration.yaml` (`enabled: false`)
- Full output: `report.txt`
- Code: `darkhunter_pop/mist_correction.py`, config `isochrone_mass.lowmass_correction`

**Model.**
- Main-sequence points below the join mass get Δlog Teff(M) at fixed L.
- G/BP/RP are recomputed from MIST's own UBVRIplus BC tables, so any other filter follows the same way.
- A mass offset δ(M) is applied at fixed L.
- Both are piecewise linear on nodes 0.1–0.8 M⊙ and zero at and above the 0.9 M⊙ join.

**Calibrated values.**
- Δlog Teff = 0, 0, −0.059, −0.039, −0.034, −0.030, −0.016, −0.003 at 0.1–0.8 M⊙. The 0.1 and 0.2 nodes have too few prior points and stay at 0.
- δ = 0, −0.035, +0.025, −0.008, 0.000, +0.004, −0.018, 0 dex. This is noise level: with the Teff correction alone the Mann residuals are already ≤ 0.02 dex. The Teff-only option is the parsimonious choice.

**Validation.**

| check | before | after | verdict |
|---|---|---|---|
| ridge colour residual, M_G 5–10.4 | +0.02 … +0.50 mag | within ±0.025 | pass |
| ridge colour residual, M_G 10.6 (≈ 0.25 M⊙, below the calibrated nodes) | — | +0.155 | edge |
| Mann ratio, all bins (dex) | +0.053 / +0.043 / +0.030 | 0.000 | pass (calibration target) |
| Mann ratio, Teff only | — | ≤ 0.02 dex | pass |
| Hwang singles above 0.5 M⊙, raw photometry (dex) | +0.031 / +0.026 / −0.000 / −0.010 | −0.003 / +0.007 / −0.006 / −0.009 | pass |
| Hwang singles 0.3–0.5 M⊙ (dex) | −0.057 | −0.103 | Hwang map unreliable here |
| field-dwarf posterior [Fe/H] (prior mean −0.06) | −0.023 | −0.072 | pass |
| DEBCat ≥ 0.8 M⊙ (blended M1/M1_dyn) | 1.013 / 1.059 | 1.012 / 1.059 | pass, unchanged |

- The +0.26 quoted earlier was the nearby Mann subsample (red dwarfs), not field dwarfs.
- Hwang below 0.5 M⊙: their map there is the region with unresolved and excess-mass systems (Region A), and Hwang et al. say themselves that PARSEC fits it better. Mann is used there.

**Hwang photometry.** Hwang et al. use observed absg and bp_rp, with no extinction correction and |b| > 10° (their §2). Our ridge and isochrone fit use Combined19 × 0.884-dereddened photometry. The Hwang rows above put their observed values straight into our fit. Unaccounted reddening makes stars look redder and fainter, so their MIST masses come out slightly low. A dereddened rerun needs their positions and parallaxes; those were requested from the orchestrator and are pending.

**Clump: Hwang (−0.075 dex) vs APOKASC-3 (isochrone/seismic +0.026 dex for RC). Trust APOKASC.**
- Hwang's mass at a CMD position is the *total* mass within Gaia's resolution. For a 1.3 M⊙ primary, MdS17 companions out to log P ≈ 5.5–6 (unresolved at clump distances) add +0.055 to +0.077 dex on average.
- The box also mixes RGB stars in (APOKASC RGB −0.013 dex), and Hwang's photometry is not dereddened (giants are farther away).
- Together these account for the −0.075 dex. The seismic masses are per star and free of companion mass.
