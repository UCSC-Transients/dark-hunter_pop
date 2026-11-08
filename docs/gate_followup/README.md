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
