# Multi-solution NSS source_id characterization

Measured on the uncut NSS parent snapshot `20260826T234425Z_3d3f740b080c` (issue #241). Descriptive counts only — no fitted model, no change to `data_acquisition.py`, `forward_model.py`, or any `config/selections/*.yaml` threshold. See `scripts/measure_multi_solution_rate.py` for the exact partitioning logic (`SourceGroup.is_fanout` / `.is_cross_type` / `.is_same_type_period_aliased`).

## Headline numbers

- Total NSS rows in the uncut snapshot: **443211**
- Total distinct `source_id`s: **437275**
- `source_id`s carrying more than one row: **5932**
- Of those, cross-match fan-out (same type + same period, #221/PR #240; excluded from the multi-solution counts below): **6**
- Genuine multi-solution `source_id`s (this ticket's subject): **5926**
  - Cross-type multiplicity (distinct `nss_solution_type` families co-occurring): **5926**
  - Same-type multiplicity / period aliasing (>1 Orbital-family solution): **0**
  - `source_id`s exhibiting both sub-cases at once: **0**

Cross-check against `docs/ARCHITECTURE.md` §4: documented as "5,926 of 5,932 duplicated source_ids are distinct NSS orbital solutions ... the remaining 6 groups (0.1%) are genuine cross-match fan-out" — measured here: 5932 duplicated source_ids total, 6 fan-out, 5926 genuine.

## Row-count multiplicity distribution (genuine multi-solution groups)

| rows per source_id | count of source_ids |
|---|---|
| 2 | 5922 |
| 3 | 4 |

## Top co-occurring `nss_solution_type` combinations (cross-type only)

| combination | source_id count |
|---|---|
| Orbital + SB1 | 5290 |
| EclipsingBinary + SB1 | 215 |
| AstroSpectroSB1 + OrbitalTargetedSearchValidated | 92 |
| Orbital + SB2 | 81 |
| EclipsingBinary + SB2 | 47 |
| OrbitalTargetedSearchValidated + SB1 | 46 |
| OrbitalTargetedSearch + SB1 | 40 |
| EclipsingBinary + Orbital | 33 |
| EclipsingBinary + SB2C | 17 |
| Orbital + SB2C | 17 |
| OrbitalAlternativeValidated + SB1 | 10 |
| OrbitalTargetedSearch + SB2 | 9 |
| OrbitalTargetedSearchValidated + SB2 | 8 |
| AstroSpectroSB1 + OrbitalTargetedSearch | 6 |
| Orbital + SB1C | 6 |
| EclipsingBinary + Orbital + SB2 | 2 |
| EclipsingSpectro + Orbital | 2 |
| EclipsingBinary + Orbital + SB2C | 1 |
| EclipsingBinary + SB1C | 1 |
| OrbitalTargetedSearch + SB2C | 1 |
| EclipsingBinary + Orbital + SB1 | 1 |
| EclipsingSpectro + OrbitalTargetedSearch | 1 |

## Covariate breakdown (descriptive, not a fitted model)

RV epoch count is not in this breakdown: the uncut snapshot's ADQL (`data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml`) never selects an RV-epoch-count column, so it is absent from the schema rather than merely unused.

| covariate | single-row sources | all genuine multi-solution rows | cross-type rows | same-type (period-aliased) rows |
|---|---|---|---|---|
| `period_day` | n=431343, median=252.5 (16/84%: 1.175/804.7) | n=11856, median=523 (16/84%: 10.15/951.4) | n=11856, median=523 (16/84%: 10.15/951.4) | n=0, median=nan (16/84%: nan/nan) |
| `ruwe` | n=431343, median=1.464 (16/84%: 1.002/3.203) | n=11856, median=3.113 (16/84%: 1.81/6.527) | n=11856, median=3.113 (16/84%: 1.81/6.527) | n=0, median=nan (16/84%: nan/nan) |
| `goodness_of_fit` | n=431343, median=1.239 (16/84%: -0.7313/10.01) | n=11856, median=2.483 (16/84%: -0.2363/10.78) | n=11856, median=2.483 (16/84%: -0.2363/10.78) | n=0, median=nan (16/84%: nan/nan) |
| `g_mag` | n=431343, median=12.72 (16/84%: 11.19/15.37) | n=11856, median=11.77 (16/84%: 10.26/12.52) | n=11856, median=11.77 (16/84%: 10.26/12.52) | n=0, median=nan (16/84%: nan/nan) |

## Worked examples (smallest `source_id` per sub-case, deterministic)

- Cross-type: source_id=2747576478880512, n_rows=2, types=['Orbital', 'SB1'], periods=[227.31967370212286, 0.6214878186210907]
- Same-type / period aliasing: (none)
- Cross-match fan-out (excluded, for contrast): source_id=1827929323783615360, n_rows=2, types=['EclipsingBinary', 'EclipsingBinary'], periods=[0.6438669562339783, 0.6438669562339783]

## Limitations

- Descriptive only; #242 (tag-and-keep) and the eventual `forward_model.py` multiplicity-emission model (blocked-by this ticket, per the issue) still need to translate this into pipeline behavior — not done here.
- RUWE, `goodness_of_fit`, and G magnitude are as reported per-row in the uncut snapshot; for multi-row sources these differ row-to-row (e.g. an `Orbital` row and an `SB1` row for the same source can carry different `goodness_of_fit`), so the covariate table above summarizes rows, not deduplicated sources — stated explicitly so #242 does not misread it as a per-source statistic.
- RV epoch count is absent from this snapshot's schema (see above) and is not reconstructable from the columns queried; a future measurement would need a new ADQL column, out of scope here.
- Measured **zero** same-type (period-aliasing) duplicates in this uncut snapshot under the strict definition (>1 row whose `nss_solution_type` *starts with* `Orbital` within one `source_id` group, at differing periods). Every genuine multi-solution `source_id` observed here co-occurs with at least one distinct solution-type family (cross-type). This is an empirical result for this snapshot, not evidence the phenomenon cannot occur — #242 should not assume it is structurally impossible, only that it was not observed in the ~443k rows measured.
