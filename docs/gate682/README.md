# Gate 682: Shahaf et al. (2023b) AMRF class III, reimplemented and compared with the published catalogue

Issue #682 (RUNG5_SPEC §2.7.4 and §2.7.6 item 6; CONTINUATION_PLAN §15 Q2). Ryan's decision P1 (2026-11-13): rebuild
the Shahaf et al. (2023b) class-III triage so that the El-Badry et al. (2024) S23 pool can be forward-modelled. On
real DR3 rows it must reproduce the 177-row catalogue exactly, or every difference must be explained.

**Result.** The parent selection, M1, the AMRF and the decision rule all reproduce exactly. Pr III reproduces within
Monte-Carlo noise. The membership is 177 vs 177, with **175 in common**. Four sources swap: two are lost and two are
gained. All four sit on the decision boundary, at 14–17 draws below `A_TR` out of 10⁵, and their pulls are below 2σ.
About 3.9 such swaps are expected from MC noise alone. The authors' random draws cannot be replayed, so an exact
match would need their RNG stream. **The membership is not exact, so this gate is escalated to Ryan** (below).

Terminology: luminous star (M1), dark companion (M2); "real DR3" is the archive, "mock" a forward-model draw.

## Inputs

| Input | Source | Note |
|---|---|---|
| Method | Shahaf et al. (2023), MNRAS 518, 2991 (arXiv:2209.00828v2): §2.3, §3.1, §3.3 and §4 | |
| Limiting curves `A_MS(M1)`, `A_TR(M1)` | MNRAS supplementary `AMRF_limiting_curves.fits` (SHA-256 `8c9cc008…`) | Copied verbatim into `config/selections/external/shahaf2023b_amrf_triage.yaml`: 91 rows, 0.20–2.00 M☉ |
| NSS orbits and covariance | `data/dr3/gaia_snapshots/nss_enrichment` (`corr_vec`, `bit_index`, `significance`) | Read-only |
| M1 | `gaiadr3.binary_masses`, all 195,315 rows | `scripts/fetch_shahaf2023b_triage_inputs.py` |
| Per-source truth | VizieR `J/MNRAS/518/2991/table1` (101,380 rows) and `table2` (177 rows) | Table 1 gives M1, A, e_A, M2min, Pr II and Pr III. Pr II and Pr III are fractions, despite the `%` unit |

Command (laptop, 6 workers with BLAS pinned to one thread, 352 s of Monte Carlo):

```
OMP_NUM_THREADS=1 nice .venv/bin/python scripts/reproduce_shahaf2023b_class3_682.py \
  --nss-ecsv data/dr3/gaia_snapshots/nss_enrichment/query.ecsv \
  --binary-masses-ecsv <binary_masses snapshot>/table.ecsv \
  --table1-ecsv <table1 snapshot>/table.ecsv --table2-ecsv <shahaf2023b_class3 snapshot>/table2.ecsv \
  --out-dir <out> --workers 6
```

The run is deterministic: two runs gave identical counts for every source. Definition SHA-256: `cb9a7641…`. The run
summary is `docs/gate682/summary.yaml`. The `binary_masses` and Table 1 snapshots used here were fetched into the session scratchpad, not `data/`. Before the
gate can be rerun from `data/`, the operator must run `scripts/fetch_shahaf2023b_triage_inputs.py`.

## Per-step agreement (real DR3)

| Step | Paper | Ours | Agreement |
|---|---|---|---|
| (i) Orbital + AstroSpectroSB1 | — | 168,065 | |
| (ii) `bit_index` ∈ {8191, 65535} | — | 166,086 | |
| (iii)+(iv) `binary_masses` IsocLum, `Orbital+M1` / `AstroSpectroSB1+M1` | 127,026 | 127,026 | exact |
| Halbwachs Δe, ϖ/σϖ, a0/σa0 | 126,624 | 126,624 | exact |
| Eq. 4, Σ(σ_X/X)² ≤ 36 | 110,401 | 110,401 | exact |
| P < 1000 d (clean) | 101,380 (84,771 Orbital + 16,609 AstroSpectroSB1) | 101,380 (84,771 + 16,609) | **exact by source ID** (0 / 0 one-sided) |
| M1 | Table 1 `M1` | `binary_masses.m1` | max \|Δ\| 5×10⁻⁷ (table rounding) |
| AMRF, Eq. 1 (Julian year) | Table 1 `A` | ours | max relative Δ 8×10⁻¹⁴ |
| `e_A` (reported, not used) | Table 1 `e_A` | first-order propagation, full NSS covariance, σ_M1 = max side | max relative Δ 7×10⁻⁷ |
| Pr III | Table 1 | MC, 10⁵ draws | interior rows (88,975): pull χ²/row 0.38, 99.8 % within 3σ; edge rows (12,405): χ²/row 17 |
| Pr II | Table 1 | MC | 0.4 < M1 < 1.7: χ²/row 0.99; all rows 28 (edge rows) |
| Decision alone: BH on the **published** Pr III | 177 | 177 | **identical ID set** |
| Membership: BH on **our** Pr III | 177 | 177 | 175 common, 2 + 2 swaps (MC noise) |
| Fixed cut, n_below ≤ 16 | (paper's stated equivalent) | 177 | the same set as BH on our Pr III |

The four boundary swaps. `n_below` is the number of draws (of 10⁵) at or below `A_TR`. Membership requires
n_below ≤ 16.

| source_id | M1 | A | n_below published | n_below ours | Table 2 | ours |
|---|---:|---:|---:|---:|---|---|
| 1683575679079854848 | 0.948 | 0.722 | 14 | 18 | yes | no |
| 4566039969296448128 | 0.514 | 0.672 | 16 | 27 | yes | no |
| 4384513795007410048 | 1.227 | 0.837 | 17 | 10 | no | yes |
| 4690739122056717184 | 0.530 | 0.700 | 17 | 10 | no | yes |

There are 40 sources with 5 ≤ n_below ≤ 40. If the published counts were the true rates, independent reruns would
swap 3.9 of them on average. We swap 4.

## Steps the paper does not state, and how each was settled

Each step below is settled by the reading that reproduces the authors' own per-source Table 1 values, never by the
Table 2 count. The alternatives are recorded in the definition file under `readings`.

1. **p-value for the FDR (§4).** The adopted p-value is `(n_below + 1)/(N + 1)`, the Monte-Carlo p-value matching
   Eq. 6. With it, BH at α = 0.1 over all 101,380 sources selects exactly Table 2. That also matches the paper's
   stated equivalent, "16 of 10⁵ below". The reading `1 − Pr III` selects 179: it adds 4384513795007410048 and
   4690739122056717184, both at 17 below.
2. **Covariance.** The draws use the full NSS covariance (column-major `corr_vec`) and reproduce Pr III.
   - Diagonal only: χ²/row ≈ 10⁴.
   - Row-major unpacking: ≈ 2×10⁴.
3. **Eccentricity.** Draws with e outside [0, 1) are rejected and redrawn, which keeps N = 10⁵ (Table 1 Pr III is
   always a multiple of 1/(10⁵+1)). Without rejection χ²/row is 68; with it, 3. The truncation matters because e is
   correlated with the Thiele–Innes constants.
4. **M1 draw.** Two half-normals of equal weight, built from `m1 − m1_lower` and `m1_upper − m1` (χ²/row 1.1).
   - Gaussian with σ = max side: 2.9.
   - Continuous split normal: 23.
   - The reported `e_A` instead uses σ = max side, which is exact for that column.
5. **M1 off the tabulated 0.2–2.0 M☉ grid.** M1 is clipped to the grid for the AMRF and the curve alike.
   - The tested readings were clip, clamp-curve-only, NaN, linear extrapolation, cubic extrapolation and rejection.
     None reproduces sources whose M1 interval reaches an edge.
   - Clip is the best on the high side, and on low-side sources whose nominal M1 is on the grid.
   - For the 88 sources with nominal M1 < 0.2, clamp-curve-only matches better.
   - The authors probably evaluated the MIST ensemble beyond the published table. The residual is a documented
     limitation, with no effect on membership: no source with n_below ≤ 40 has M1 near an edge (0.40–1.23 M☉).

## Mock rows

`mock_selection_rows.mock_draws_to_selection_rows(..., include_orbit=True, shahaf_triage=load_definition())` adds
`shahaf2023b_class3_reimpl`, plus `p_iii`, `n_below_atr`, `n_draws`, `amrf_reimpl`, `m1_source` and `parent_status`.
Mock rows have no real-DR3 sample to run BH over, so they use the paper's stated equivalent, n_below ≤ 16. On real
DR3 that rule selects the same set as BH. The Monte Carlo stops early, exactly, once 17 draws fall below `A_TR`.

Mock smoke, `co_gen1008` (st2), first 3,000 accepted orbits, 30 s:
- 1,576 class III and 1,423 not class III.
- 1 row is not applicable (no TAG10 M1).
- By companion: dark 1,542 / 2,676, WD 34 / 324.

### Known limitations on the mock side (escalated)

- **M1.** The mock carries TAG10 M1 (`pipeline_m1_msun` ± σ, symmetric). The real S23 used the `binary_masses`
  IsocLum M1 and its asymmetric 16/84 % interval.
- **`binary_masses` availability.** The parent cuts (iii)/(iv) cannot be applied to the mock: 22.5 % of real rows that
  pass every orbit cut (130,859 → 101,380) lack an IsocLum mass. Each mock row is marked
  `binary_masses_not_emulated`.
- **#704 pre-filter.** On `co_gen1008`, 397 of the 1,576 class-III rows fall outside the #704 luminous-covariance
  pre-filter. A luminous-generation row outside that filter therefore gets `mock_prefilter_excluded:<version>` for
  class III. The pre-filter was validated against the Andrews and El-Badry 2026 chains, not against Shahaf class III.
- **El-Badry 2024 file.** `config/selections/elbadry2024.yaml` still reads Table 3 membership in both modes. Pointing
  its forward-model S23 pool at `shahaf2023b_class3_reimpl` is an edit to a frozen file and needs Ryan.
