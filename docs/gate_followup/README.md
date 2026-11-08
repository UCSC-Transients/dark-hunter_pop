# Follow-up evidence layer: measured first looks (RUNG5_SPEC §11)

## FL-T3 (#626): Gaia DR3 RV amplitude vs the astrometric K1

Script `scripts/fl_rv_amplitude_check_626.py`; inputs already on disk (the approved bright RV-stats snapshot
`20261022T181049Z_dr3_rv_stats_bright` and `output/gate390/published_orbits.h5`); numbers in `rv_amplitude_626.json`.
Report only: no threshold is adopted, and the RVS epoch sampling is not modelled (the forward model is FL-Q6).

- **Noise floor** (looks-single bright parent rows, `rv_amplitude_robust`): median / 90th percentile by G_RVS
  4–8, 8–9, 9–10, 10–11, 11–12 are in the JSON; the 90th percentile rises from ≈10 to ≈98 km/s. The parent sample can
  hide SB1s, so the 90th percentile is a conservative floor.
- **RV-detectable** real orbit hosts (2 K1 > 3 × floor p90 at their G_RVS): 387 of 28,805 (245 AstroSpectroSB1,
  142 Orbital; 175 with A > 0.5).
- **Ratio** r = amplitude / (2 K1) among them: 16/50/84 % = 0.65 / 0.97 / 1.13. The astrometric orbits predict the RV
  amplitude on average. AstroSpectroSB1 0.64 / 0.95 / 1.06; Orbital 0.70 / 1.03 / 1.29.
- **Flat** (amplitude below the floor's 90th percentile despite a detectable K1), candidate refutations: 5.9 % overall,
  7.3 % AstroSpectroSB1, 3.5 % Orbital, 6.9 % at A > 0.5.
- **Reading.** The DR3 RV statistics are informative only for the bright (G_RVS ≲ 12), large-K1 minority. Among them the
  astrometry and the RV amplitude agree at the median, and a few per cent look RV-flat. That fraction is an upper bound on
  spurious orbits there until the epoch sampling is forward-modelled; it is not yet a spurious rate.

## FL-T1 (#641): follow-up candidate set and survey coverage

Script `scripts/fl_candidates_641.py`; numbers in `fl_t1_candidates_summary.json`. Inputs: the real M2_app with the adopted ν (#638) for 122,841 C1 rows, the published orbits for A, and the #628 snapshots. Cuts (FL-Q2): M2_app ≥ 1.0 M⊙, or A > 0.5 and M2_app ≥ 0.6 M⊙. GALEX: nearest GUVcat_AIS source within 3" of the host position propagated to 2007.0.

**7,020 candidates.** Coverage fractions overall: GALEX NUV 32.1%, FUV 3.7%, ambiguous UV match 0.0%; Gaia DR3 RV amplitude 39.8%; any survey RV/spectrum 48.1%; multi-epoch RV (LAMOST MRS ≥ 2 nights or APOGEE ≥ 2 visits) 2.9%; GALAH 3.6%.

| M2_app (M⊙) | n | n with A > 0.5 | NUV | FUV | Gaia RV amplitude | any RV | multi-epoch RV | GALAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–0.6 | 0 | 0 | — | — | — | — | — | — |
| 0.6–1 | 4633 | 4633 | 35% | 2% | 33% | 42% | 3% | 3% |
| 1–1.4 | 2036 | 1874 | 28% | 6% | 51% | 57% | 3% | 4% |
| 1.4–2 | 326 | 325 | 19% | 6% | 63% | 69% | 4% | 6% |
| 2–3 | 22 | 22 | 41% | 18% | 73% | 73% | 4% | 0% |
| 3–5 | 1 | 1 | 100% | 0% | 100% | 100% | 0% | 0% |
| 5–10 | 0 | 0 | — | — | — | — | — | — |
| 10–1000 | 2 | 2 | 100% | 0% | 50% | 100% | 0% | 0% |

**Reading.**
- 7,020 candidates against the laptop scope of FL-Q3 (Path-2 SED fits for the candidate set): above 1.0 M⊙ there are 2,387, and above 1.4 M⊙ 351. The 0.6–1.0 M⊙ large-A bin (4,633) is the WD-dominated part; its SED fits are the ziggy-scale share.
- Multi-epoch survey RVs are rare (3 %); the Gaia DR3 RV amplitude covers 40 % (all G_RVS ≲ 12) and is the main RV handle in DR3 (FL-T3, #626). DR4 epoch RVs change this (§11.3).
- GALEX NUV detects a third of the candidates (FUV 4 %); a non-detection is informative only with the AIS footprint and depth (#633).
- The highest-M2_app candidates include Gaia BH1 (M2_app 12.5, q16–q84 10.2–15.5) and 6281177228434199296, which El-Badry et al. (2024, Table 3) label spurious: the follow-up labels are what separate these (§11.4).
