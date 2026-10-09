# #391 rung 2: MdS17-reweighted mock vs the full DR3 orbit sample

Issue #391. Spec: `docs/MOCK_POPULATION_SPEC.md`, §5 rung 2. The earlier paused diagnostic is in
`superseded_paused_2026-10-03/`.

**What this is.** This is MdS17 at its **published parameters**, with luminous companions only and the compact-object mixture off (MP-Q17). It is reweighted onto proposal-set draws simulated with every model decided through 2026-10-06:
- epoch model v2 (#441);
- isochrone M1, deblended and drawn from the posterior (#418, #445);
- MIST coeval flux ratios;
- the bounded eccentricity proposal.

Nothing was tuned toward the data. The 2-D CMD Malmquist weight still misses its closed-loop target (worst pull 7.6), so every result is shown **with and without** it. The weight is a reweightable target factor, so no draws had to be re-simulated.

> **Update 2026-10-09:** generations 25 and 26 were added with the efficiency proposal (PRs #447 and #448). The **dwarf-only** rung-2 result (ESS 879 without W, 656 with W) is in [`dwarfs/README.md`](dwarfs/README.md).
>
> Evolved rows are excluded until the evolved-row flux fix lands. They held 19% of the weight at ESS 4.4.

## Run

| | generation 23 (tuning) | generation 24 (full) |
|---|---|---|
| Config | `config/population/proposal_set_restart_tune.yaml` | `config/population/proposal_set_restart_full.yaml` |
| Draws | 25,000 | 430,000 |
| Accepted orbits | 488 | 8,646 |
| Wall time (8 workers, `nice`, BLAS at 1 thread) | 75.5 min | 497 min reported (the run spanned a laptop reboot at 187,500 draws, resumed by the coordinator; wall time is not additive) |

- Total compute: 129.5 CPU-h, a mean of 1.02 CPU s per draw.
- Parent: `20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj`, of which 159,116 rows are usable.
- The two generations are combined with deterministic-mixture weights. The only difference between them is the flux-proposal width: 0.15 dex in generation 23, 0.30 dex in generation 24.

## Headline numbers

| | no Malmquist weight | 2-D CMD Malmquist weight |
|---|---|---|
| Accepted-set MdS17 Kish ESS | **354.8** | **290.3** |
| Largest single weight share | 0.017 | 0.020 |
| Expected accepted orbits vs real 167,911 | 1.07 × 10⁵ (0.64×) | 9.13 × 10⁴ (0.54×) |

- **Real comparison sample:** DR3 Orbital + AstroSpectroSB1, 167,911 of 168,065 rows. The filters are ϖ > 0.2 mas, an atmosphere present, and the Halbwachs IPD/C* cuts (MP-Q24).
- **ESS shortfall:** the run reached ESS ≈ 290–355, against the 2,000 target. Efficiency was 2.7 ESS per CPU-h. It was flagged on #391 when the run was 2 h in.

## Six-panel (shapes, unit area)

Grey bands mark bins with ESS < 30 (MP-Q19). The KS test uses only bins with ESS ≥ 30; the KS columns give D and p.

| Panel | Bins with ESS ≥ 30 | Real sample in those bins | KS, no W | KS, with W | Verdict |
|---|---|---|---|---|---|
| P_orb | 5/20 | 77% | 0.062, 0.17 | 0.051, 0.48 | **agrees** |
| G | 7/20 | 89% (85% with W) | 0.061, 0.21 | 0.111, 0.006 | agrees without W; W makes it worse |
| 1/ϖ | 6/20 | 53% | 0.071, 0.031 | 0.080, 0.046 | marginal (full-range informational KS D = 0.155: the mock sits closer) |
| e | 6/20 (5/20 with W) | 48% (39%) | 0.146, 3 × 10⁻⁴ | 0.111, 0.032 | **disagrees** |
| f_m | 6/20 (5/20 with W) | 75% (73%) | 0.088, 0.003 | 0.106, 9 × 10⁻⁴ | **disagrees** |
| cos i | 2/20 | 12% | 0.078, 0.78 | 0.101, 0.52 | too few effective draws to test (full-range informational D = 0.047, p = 0.42) |

How each disagreement looks:
- **Eccentricity:** the mock is too eccentric, the same direction as El-Badry et al. (2024) Fig. 5. MdS17 η at published values makes too many orbits with e > 0.5.
- **f_m:** the mock cuts off near 0.06–0.1 M⊙. The real tail toward higher f_m needs dark or compact companions (MP-Q17) or companions outside q ≤ 1.
- **1/ϖ:** the mock sits closer than the real sample. The Malmquist weight does not close this gap.

## Solution-type mix (MdS17-weighted expected counts, scaled to the full G < 19 parent)

| Outcome | Mock, no W | Mock, with W | DR3 |
|---|---|---|---|
| Accepted orbits | 1.07 × 10⁵ | 9.1 × 10⁴ | 167,911 |
| Published acceleration (7/9-parameter; s > 20, F2 < 22 for 7-parameter) | 1.8 × 10⁵ | 1.6 × 10⁵ | **not compared**: the real `nss_acceleration_astro` counts are not snapshotted (MP-Q20 / ELBADRY2024 Q7). Flag only |
| 7-parameter / 9-parameter, all | 2.9 / 1.6 × 10⁵ | 2.6 / 1.4 × 10⁵ | — |
| Orbit fitted but failing the cuts | 1.0 × 10⁷ | 9.9 × 10⁶ | — |
| 5-parameter (binaries only) | 1.23 × 10⁸ | 1.36 × 10⁸ | 2.85 × 10⁸ stars with RUWE < 1.4 in the parent, singles included, so not like-for-like |
| Fewer than 12 visibility periods | 1.28 × 10⁶ | 1.32 × 10⁶ | 5.0 × 10⁵ (0.17% of the usable parent, from each star's own `visibility_periods_used`) |

The mock's insufficient-visibility count comes from binaries only, yet it is 2.6× the real count for *all* stars. The real number is per star and the mock number is per companion draw, so this is indicative, not a gate. It needs a look under #428 / #432.

## Giants: CMD-evolved primaries, Orbital only (`giants/`)

| | Value |
|---|---|
| Real Orbital, evolved fraction | 0.0989 ± 0.0008 (12,627 of 127,657 CMD-classified rows; real distances use the inverse NSS parallax) |
| Mock accepted, evolved, with W | 0.126 ± 0.020 |
| Mock accepted, evolved, no W | 0.107 ± 0.017 |
| Evolved-subset ESS | 12.7 (218 raw draws) |

- **The evolved fraction agrees within the mock error,** with or without the weight.
- **Evolved six-panel** (informational only, n_eff = 12.6): KS p = 0.03, 0.008, 0.21, 0.01, 0.02, 0.38 for P, G, 1/ϖ, e, f_m and cos i.
- **Evolved P medians agree:** 747 d mock vs 750 d DR3. The mock spreads wider: p10/p90 = 399/1,218 d against 472/1,040 d.
- **Undersampled:** evolved primaries need their own proposal top-up before any shape test.

## Agrees / disagrees (no tuning)

**Agrees:**
- the P shape;
- the G shape without W;
- cos i (full range, informational);
- the evolved fraction (≈ 10%);
- the evolved period median.

**Disagrees:**
- **e:** the mock is too eccentric;
- **f_m:** no high-f_m tail;
- **1/ϖ:** the mock is too close;
- **the total:** 0.54–0.64 of the real count;
- **G with W:** worse than without;
- **insufficient visibility:** 2.6× DR3.

**Not testable at this ESS:**
- most tail bins;
- the cos i core;
- the evolved six-panel.

## Caveats

- **The real-side atmosphere filter is still the TAG10 atmosphere-present test** from PR #397. The parent now drops rows by isochrone-M1 availability, which removes 198,734 → 159,116 rows after IPD/C*. The real-side mirror of MP-Q5 under isochrone M1 is therefore not exactly symmetric; it removes only 37 real rows.
- **The ESS is far below the 2,000 target.**
  - Per-factor ESS, measured mid-run on 456 accepted draws:

    | Factor | ESS |
    |---|---|
    | parent tilt | 158 |
    | M2–P | 150 |
    | flux | 135 |
    | eccentricity | 312 |
  - What would help: a MIST-centred flux proposal, M2 and P drawn from the MdS17 shape, a softer parent tilt, and an evolved-row top-up. Generation 24's draws stay usable through mixture weights.

## Reproduce

```bash
D=data/dr3/gaia_snapshots
python scripts/plot_proposal_pilot.py \
  --artifact output/proposal_set/restart_gen23_tune.h5 \
  --artifact output/proposal_set/restart_gen24_full.h5 \
  --parent-dir $D/20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj \
  --real-snapshot $D/20260826T234425Z_3d3f740b080c/query.ecsv \
  --real-input-columns $D/20261003T182211Z_nss_orbit_input_columns \
  [--cmd-malmquist config/population/malmquist_cmd.yaml] --out-dir <dir> --prefix rung2_{noW,cmdW}

python scripts/giant_population_diagnostics.py <same artifacts and parent> --real-types Orbital \
  --real-input-columns ... [--cmd-malmquist ...] --out-dir <dir>
```

The artifacts are gitignored and kept in the `proposal-restart-391` worktree.
