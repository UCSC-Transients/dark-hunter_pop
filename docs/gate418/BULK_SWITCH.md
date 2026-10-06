# Data side: `mass_calibration.method: MIST_isochrone` (#418, off by default)

Specified in ARCHITECTURE.md `mass_derivation_bulk` (isochrone bullet) and MOCK_POPULATION_SPEC §11.7. Measured with `scripts/remeasure_bulk_isochrone_m1.py`:
- the `data_acquisition` artifact of run `20260930-022223-672b092`;
- a deterministic subsample `source_id % 10 == 0`, which gives 70,020 candidates;
- the known-truth BH systems, always included.

Full output is in `bulk_remeasure.txt` and `bulk_remeasure.json`. The switch stays **off**. These numbers are for Ryan's decision.

| funnel | TAG10 (default) | MIST_isochrone |
|---|---|---|
| input candidates | 70,020 | 70,020 |
| no atmosphere / no CMD | 265 | 419 |
| M1 off the isochrone grid | — | 7,784 |
| M1 ok | 69,755 | 61,817 |
| M2 computed (orbit + covariance) | 30,759 | 28,673 |
| no NSS covariance | 360 | 337 |
| after the M2 cut (M2 + 2σ ≥ 1.1 M⊙) | **3,811** | **3,957** (+3.8%) |
| M2 pre-cut 5/16/50/84/95% (M⊙) | 0.16 / 0.26 / 0.44 / 0.61 / 0.82 | 0.18 / 0.30 / 0.50 / 0.68 / 0.95 |
| wall time (s, laptop under load) | 388 | 435 |

The isochrone path loses 7% of the orbit candidates that reach M2: 2,086 off-grid or no-CMD rows, mostly with high E(B−V). The parent loses the same kind of rows, so the mock and the data stay symmetric (MOCK_POPULATION_SPEC §11.3). Whether 7% is acceptable, or the extinction-error fraction or the evidence floor should be loosened, is MP-Q34.

**Known truth** (published luminous-star masses now in `config/benchmarks/known_truth_gaia_bh.yaml`):

| system | published M1 | TAG10 M1 | isochrone M1 | M2: TAG10 / isochrone (published) |
|---|---|---|---|---|
| Gaia BH1 | 0.93 ± 0.05 | 0.82 ± 0.10 | **0.97 ± 0.07** | 12.6 ± 2.8 / 12.8 ± 2.8 (9.62 ± 0.18) |
| Gaia BH2 | 1.07 ± 0.19 | 0.77 ± 0.16 | **1.07 ± 0.07** | 8.06 ± 0.57 / 8.51 ± 0.53 (8.94 ± 0.34) |
| Gaia BH3 | 0.76 ± 0.05 | — | — | not in the DR3 NSS artifact |

Both BH M1 values move onto the published ones. BH1's M2 is limited by the astrometry-only σ, as in #374.

**Literature reproduction counts are not re-measured end to end here**, and neither is the Andrews forward-model pass-2 M1 under the switch. Both are in #425 and must be checked before flipping. The reproduction paths read their own M1 columns, but the `sample_selection` stage enriches rows from the bulk artifact.

Other issues:
- **#393**: does not apply to the switch path; it stays open for TAG10.
- **#380**: under the switch, σ_M1/M1 > 0.25 for 2.1% of parent rows.
- **#323**: out-of-range becomes `m1_off_grid`; close it when the switch flips.

## Flip to `MIST_isochrone` (2026-10-04, Ryan; #425 fixed)

#425 is fixed in two places:
- **`sample_selection`.** The bulk enrichment of uncut-snapshot rows now writes only `pipeline_m1_msun`, `pipeline_m1_sigma_msun`, `pipeline_m2_msun` and `pipeline_sigma_m2_msun`. The DA-fallback rows keep the generic names, since they carry no literature columns.
- **TAG10 by name.** The Andrews forward-model `pipeline_tag10_bulk` uses TAG10 under either method, and so does `derive_tag10_m1_r1`. The method key in the sidecar fingerprint is pinned, so existing sidecars still match.

The canonical config now sets `mass_calibration.method: MIST_isochrone`.

**End to end on the full `data_acquisition` artifact** of run `20260930-022223-672b092`: `scripts/remeasure_flip_418.py`, with outputs in `flip_remeasure_<method>.{json,txt}`. That run's bulk under each method feeds the stage-path selection rows, which come from the uncut snapshot plus the enrichment.

| | TAG10 | MIST_isochrone |
|---|---|---|
| input candidates | 351,268 | 351,268 |
| no atmosphere / no CMD | 1,268 | 2,280 |
| M1 off the isochrone grid | — | 38,451 |
| M2 computed | 154,046 | 143,929 |
| after the M2 cut | 18,916 | **19,476** |
| Gaia BH1 M1 (published 0.93 ± 0.05) | 0.82 ± 0.10 | **0.97 ± 0.07** |
| Gaia BH2 M1 (published 1.07 ± 0.19) | 0.77 ± 0.16 | **1.07 ± 0.07** |
| Gaia BH1 / BH2 M2 (published 9.62 / 8.94) | 12.58 / 8.06 | 12.84 / 8.51 |

**Literature selection, every enabled sample.** Parent N, survivor count and the survivor-ID hash are identical under both methods:

| sample (mode) | parent | survivors |
|---|---|---|
| `andrews2022` (reproduction) | 134,598 | 25 |
| `andrews2022_modified` (forward_model) | 134,598 | 24 |
| `elbadry2024` (reproduction) | 443,211 | 48 |
| `elbadry2026` (reproduction) | 443,211 | 240 (routes 132 / 30 / 11) |

The reproduction paths own their M1, and the Andrews forward model stays at TAG10 by name, so none of them moves. Whether the Andrews forward model should switch to the pipeline's isochrone M1 is a separate decision for Ryan.

**Off-grid rows.** 11% of all NSS candidates (38,451) and about 7% of the orbit candidates that reach M2 are off the isochrone grid. As MOCK_POPULATION_SPEC §11.9 shows for the parent, these are mostly high-E(B−V) rows, so this is MP-Q34 (extinction errors and the choice of 3-D map).
