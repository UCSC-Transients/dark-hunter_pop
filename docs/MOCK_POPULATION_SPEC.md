# Mock population: Gaia-star primaries, Moe & Di Stefano companions, importance-reweighted proposal set — spec

Issue: #391 (Step 1 design, parent #339). Status: **spec for Ryan's approval.** Ryan's decisions
of 2026-10-02 (#391) are fixed inputs and are not reopened here. Everything they and the cited
papers determine is marked for implementation. Every choice they do not determine is listed in
§8 as a numbered open question. None of those has been picked. Where the pilot (§7) needs a value
for an open question to run at all, it uses a **provisional** setting that is named as such in the
config, the artifact and every figure caption, and which does not count as a decision.

Related specs: `docs/ELBADRY2024_REPRODUCTION_SPEC.md` (the six-panel gate, the cuts, the cost
measurements in its §8, open questions Q5–Q12 there), `docs/GAIAMOCK_API.md` (what must not be
reimplemented), `docs/ARCHITECTURE.md` §4 (`selection_function_astrometric`, `population_model`,
`sensitivity_analysis`, `inference`).

## 0. Decisions this spec builds on (Ryan, #391)

1. The mock that must reproduce the full DR3 NSS orbit sample (Step 1, #339) uses the **same
   parameterized population as inference**. We do not rebuild El-Badry et al. (2024)'s
   Galaxia → COSMIC → MIST population synthesis.
2. **Primaries are real Gaia stars** from `gaia_source`, selected like the real NSS parent, with
   their real positions, parallaxes, proper motions, G and colours. M1 is assigned exactly as on the
   data side.
3. **Companions** follow Moe & Di Stefano (2017, ApJS 230, 15; "MdS17") plus a companion-type
   mixture (luminous MS, WD, NS, BH).
4. **Importance reweighting**: one broad proposal set is simulated once through `gaiamock_mod`;
   truth, outcome and observed solution are stored per draw; any θ is a reweighting; under-covered
   regions get top-up draws.
5. Laptop compute, smaller test run first.
6. No catalog request to El-Badry et al. (old Q8 closed). Their Fig. 5 stays a by-eye cross-check.

### 0.1 Generation-time decisions (Ryan, 2026-10-02, #391 comment 5963152741)

Recorded at https://github.com/UCSC-Transients/dark-hunter_pop/issues/391#issuecomment-5963152741.
These replace the pilot's `provisional_*` settings for the questions below. They are fixed at
generation (§3.5).

| Question | Decision | Implementation |
|---|---|---|
| MP-Q1 parallax floor | ϖ > 0.2 mas, with the same floor on the real comparison sample | parent query cut on measured `parallax`; the real Orbital + AstroSpectroSB1 sample is filtered on its NSS `parallax` > 0.2 mas |
| MP-Q2 subsample | K = 10⁶ | `random_index < 1,000,000`, the pilot's slice |
| MP-Q3 Halbwachs (b)/(c) | applied, using each real star's own values | parent rows must have `ipd_frac_multi_peak` ≤ 2, `ipd_gof_harmonic_amplitude` < 0.1 and \|C\*\| < 1.645 σ_C\*. C\* = `phot_bp_rp_excess_factor` − polynomial(`bp_rp`) per Riello et al. (2021) Eq. 6 / Table 2; σ_C\*(G) = 0.0059898 + 8.817481×10⁻¹² G^7.618399 (Eq. 18). A row with undefined C\* (no BP/RP) fails, since the condition cannot hold. The per-cut pass flags are stored |
| MP-Q4 truth distance | Bailer-Jones et al. (2021) geometric | truth parallax = 1000 / `r_med_geo` (pc) from `external.gaiaedr3_distance`; `r_lo_geo` and `r_hi_geo` are stored; rows without a geometric distance are dropped and counted. The ϖ floor stays on the *measured* parallax (MP-Q1) |
| MP-Q5 M1 | TAG10 point value; no-atmosphere stars dropped on both sides; giants kept and flagged | parent as in the pilot. Real side: rows with neither an MSC nor a GSP-Phot atmosphere are dropped. `is_giant` = log g of the atmosphere TAG10 used < 3.6. The threshold is the dwarf/giant log g of the Andrews et al. (2022) ATF notebook cut, used here **only as a flag**: it changes no weight or selection |
| MP-Q6 light split | observed G is the total system light | gaiamock gets the star's observed `phot_g_mean_mag` |
| MP-Q13 mass–luminosity | Janssens et al. (2022) M_G(M), 0.1 dex log-normal scatter in f | target `p(log10 f) = N(log10 f_J(M1, M2), 0.1)`; it stays reweightable |

**Real-side check of the Halbwachs (b)/(c) cuts** (measured 2026-10-02, archive DR3 `gaia_source`
values for all 168,065 rows). The cuts are almost exactly implicit on the real side, as expected
for NSS input filters. The few failures probably come from the cuts being run on internal
pre-release values; this was not verified.

| Real sample | rows | IPD pass | C\* pass | both | RUWE > 1.4 | ≥ 12 vis. periods |
|---|---|---|---|---|---|---|
| Orbital | 134,598 | 99.984% | 99.972% | 99.956% | 100% | 100% |
| AstroSpectroSB1 | 33,467 | 99.866% | 99.970% | 99.836% | 100% | 100% |

So no IPD/C\* filter is applied to the real comparison sample. Whether to remove the 0.06% that
fail (114 rows) for exact symmetry is MP-Q24.

MP-Q7–Q12 and Q14–Q22 stay open. They can still be changed by reweighting.

## 1. Primary parent sample from `gaia_source`

### 1.1 What the real NSS astrometric pipeline processed

Halbwachs et al. (2023, A&A 674, A9, §1.2 "Selection of stars to be processed") give the input to
the DR3 astrometric-binary processing, in this order:

| Step | Criterion | Stars left |
|---|---|---|
| a | `phot_g_mean_mag` < 19, RUWE > 1.4, at least 12 visibility periods (from the EDR3 astrometric processing) | ≈ 36.5 M |
| b | `ipd_frac_multi_peak` ≤ 2 **and** `ipd_gof_harmonic_amplitude` < 0.1 (reject partially resolved pairs) | 10.9 M |
| c | corrected BP/RP flux excess \|C\*\| < 1.645 σ_C\* (Riello et al. 2021, Eqs. 6 and 18) | **4,115,743** |

The processing then ran the cascade (single star → acceleration → orbital → VIM) and the
post-processing filters (Halbwachs et al. 2023 §2; the Eq. 20–22 cuts we already apply). The
combined `AstroSpectroSB1` solutions (Gaia Collaboration, Arenou et al. 2023, A&A 674, A34) come
from sources that also entered the spectroscopic SB1 chain; El-Badry et al. (2024 §4) compare the
mock `Orbital` outcome with the union `Orbital` + `AstroSpectroSB1` (168,065 rows), and so do we
(`dr3.selection_function_astrometric.elbadry2024_comparison_nss_solution_types`).

**Measured check (2026-10-02, Gaia archive, `random_index` < 10⁶, i.e. a uniform 1/1812 slice of
`gaiadr3.gaia_source`):** 319,063 rows have G < 19 and 19,903 have G < 19, RUWE > 1.4 and
`visibility_periods_used` ≥ 12. Scaled to the 1,811,709,771-row table that is ≈ 578 M and
≈ **36.1 M**, which matches step (a)'s 36.5 M. So `gaia_source` reproduces the NSS input count.

### 1.2 Which of these become the mock parent, and which are simulated

RUWE > 1.4 and ≥ 12 visibility periods are **outcomes**. They depend on the companion, so they are
what gaiamock simulates (`ruwe_min = 1.4`, the cascade's visibility check). Selecting the parent on
the star's *real* RUWE would condition the mock on the answer. The mock parent is therefore:

- **G < 19** (step a's magnitude limit; a property of the system's light, which the mock keeps as
  the real star's G, see §1.5 and MP-Q6), and
- a **parallax floor** to keep the snapshot finite (MP-Q1). It is not in Halbwachs et al.; it is
  shape-neutral only if the same floor is applied to the real comparison sample. Measured on our
  uncut snapshot `20260826T234425Z_3d3f740b080c`: of the 168,065 Orbital + AstroSpectroSB1 rows,
  99.998% have ϖ > 0.2 mas, 99.90% > 0.33 mas, 98.6% > 0.5 mas, 80.4% > 1 mas.

The step (b) and (c) cuts are about resolved or blended pairs and contaminated photometry, which
gaiamock does not simulate. How to treat them is MP-Q3. The snapshot stores the columns needed for
any answer (`ipd_frac_multi_peak`, `ipd_gof_harmonic_amplitude`, `phot_bp_rp_excess_factor`,
`bp_rp`), so the choice can be applied later without re-querying.

### 1.3 Snapshot: query, size, disk

One archive query (async TAP), a **uniform random subsample** via `gaia_source.random_index`
(a random permutation of `0 … N−1` published for this purpose, so `random_index < K` is a
shape-neutral 1-in-(N/K) subsample):

```sql
SELECT gs.source_id, gs.random_index, gs.ra, gs.dec, gs.parallax, gs.parallax_error,
       gs.pmra, gs.pmdec, gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
       gs.bp_rp, gs.phot_bp_rp_excess_factor, gs.ruwe, gs.visibility_periods_used,
       gs.ipd_frac_multi_peak, gs.ipd_gof_harmonic_amplitude, gs.l, gs.b,
       ap.teff_msc1, ap.teff_msc1_upper, ap.teff_msc1_lower,
       ap.logg_msc1, ap.logg_msc1_upper, ap.logg_msc1_lower,
       ap.mh_msc, ap.mh_msc_upper, ap.mh_msc_lower,
       ap.teff_gspphot, ap.teff_gspphot_upper, ap.teff_gspphot_lower,
       ap.logg_gspphot, ap.logg_gspphot_upper, ap.logg_gspphot_lower,
       ap.mh_gspphot, ap.mh_gspphot_upper, ap.mh_gspphot_lower
FROM gaiadr3.gaia_source AS gs
LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON gs.source_id = ap.source_id
WHERE gs.random_index < :K AND gs.phot_g_mean_mag < 19 AND gs.parallax > :parallax_floor
```

Since §0.1 (MP-Q4), the query also selects `bj.r_med_geo, bj.r_lo_geo, bj.r_hi_geo` via
`LEFT JOIN external.gaiaedr3_distance AS bj ON gs.source_id = bj.source_id` (EDR3 and DR3
`source_id` are identical). That is a new snapshot with the same `K` and slice; the pilot snapshot
is kept unchanged.

The atmospheric columns are exactly the ones `data_acquisition` fetches for `mass_derivation`
(MSC preferred, GSP-Phot fallback), so the same TAG10 code path applies (§1.4). The real star's
RUWE etc. are stored for diagnostics only, never as a selection.

Size, from the measured slice above (≈ 170 B per row in HDF5, uncompressed):

| Parent | Full-table rows | Full disk | 1-in-181 (`K` = 10⁷) | 1-in-1812 (`K` = 10⁶) |
|---|---|---|---|---|
| G < 19, ϖ > 0.2 mas | ≈ 389 M | ≈ 66 GB | 2.1 M rows, ≈ 0.37 GB | 215 k rows, ≈ 37 MB |
| G < 19, ϖ > 0.5 mas | ≈ 176 M | ≈ 30 GB | 0.97 M rows, ≈ 0.17 GB | 97 k rows, ≈ 17 MB |

The full parent does not fit the laptop (≈ 13 GiB free, 3 GiB floor). A random subsample is
required and is enough: the proposal set (§3) draws ~10⁵ systems, so a 10⁶-row parent gives each
star ≲ 0.1 expected draws and no star dominates the weights. The full-run `K` is MP-Q2. The
snapshot is written once under `data/dr3/gaia_snapshots/<id>_gaia_source_parent/` with a
`meta.yaml` (ADQL, query date, row count, SHA256, `K`, the full-table row count used for scaling)
like the existing snapshots, and is never re-queried by a stage.

The scale factor to the full parent is `N_full / N_snap = 1,811,709,771 / K` (the expected value;
its binomial scatter is < 0.1% at these sizes and is recorded).

### 1.4 M1: the data-side path

`mass_derivation.resolve_atmosphere_from_extras` (MSC → GSP-Phot) followed by
`mass_derivation.derive_tag10_m1_r1` (TAG10, Santos switch from `mass_calibration`), called as
library functions, not reimplemented and not modified. Stars for which no atmosphere resolves get no
M1; on the real side 99.98% of the 168,065 rows have MSC parameters. Whether M1_true is the TAG10
point value or a draw from the TAG10 uncertainty, and what to do with unresolvable stars, is MP-Q5.
Either way M1 is fixed per draw at generation time (§3.5).

### 1.5 Stars that are already unresolved binaries

Every `gaia_source` row is a *system*: roughly half of solar-type stars have a companion
(MdS17 Table 13: F_n=0 = 0.60 single at q > 0.1). Its G, colours, astrometry and atmospheric
parameters already contain that companion's light and motion. The mock treats each parent row as
**one system** and draws at most one companion for it (triples off, §2.7). The real row's own
companion is not known and is not modelled separately. Two consequences:

- The data side applies TAG10 to the same blended light, so assigning M1 from the observed
  parameters is the **consistent** choice for comparing mock and real M2 inferences. It is not an
  unbiased truth for single-star physics, and this is a documented limitation, not a bug.
- How the observed light is split between primary and the drawn companion is MP-Q6: either the
  observed G is the total system light and the companion's share is carved out of it, or the
  observed star is the primary alone and the companion adds light. These give gaiamock different
  `phot_g_mean_mag` for the same draw, so the choice is fixed at generation time (§3.5).

The real star's own RUWE is never used to remove it, since that would select on an outcome.

## 2. Companions: Moe & Di Stefano (2017)

All equation and table numbers are MdS17 (arXiv:1606.05347). `log` is log₁₀, P in days,
q = M_comp/M1. These are the **published values** of θ_MdS for rung 2 (§5); at rung 3 they become
free parameters of `population_model`, so they live in config, not in `constants.py`: the published
coefficients in a frozen table `config/population/moe_distefano2017.yaml` (cited per equation), and
the switches in `config/population/proposal_set_pilot.yaml` (not under `config/fragments/`, which `load_config` merges into `PipelineConfig`), which merges into
`population_model.moe_distefano` when the stage integration lands (§3.8).

### 2.1 Domain

Table 1: 0.8 < M1/M⊙ < 40, 0.1 < q < 1.0, 0.2 < log P < 8.0, 0 < e < e_max(P). The model is a
**companion frequency** (mean number of companions per primary, which may exceed 1 for massive
primaries, §2 and Eq. 1), not a binary fraction. M1 < 0.8 M⊙ and q < 0.1 are outside it (MP-Q7,
MP-Q8).

### 2.2 Companion frequency f_logP;q>0.3(M1, P) (companions with q > 0.3 per decade of P)

Anchors (Eqs. 20–22), with x = log(M1/M⊙):

- f₁ ≡ f_logP<1;q>0.3 = 0.020 + 0.04 x + 0.07 x² (Eq. 20)
- f₂.₇ ≡ f_logP=2.7;q>0.3 = 0.039 + 0.07 x + 0.01 x² (Eq. 21)
- f₅.₅ ≡ f_logP=5.5;q>0.3 = 0.078 − 0.05 x + 0.04 x² (Eq. 22)

Piecewise form (Eq. 23), α = 0.018, ΔlogP = 0.7:

| log P range | f_logP;q>0.3 |
|---|---|
| 0.2 ≤ log P < 1.0 | f₁ |
| 1.0 ≤ log P < 2.7 − ΔlogP | f₁ + (log P − 1)/(1.7 − ΔlogP) × (f₂.₇ − f₁ − α ΔlogP) |
| 2.7 − ΔlogP ≤ log P < 2.7 + ΔlogP | f₂.₇ + α (log P − 2.7) |
| 2.7 + ΔlogP ≤ log P < 5.5 | f₂.₇ + α ΔlogP + (log P − 2.7 − ΔlogP)/(2.8 − ΔlogP) × (f₅.₅ − f₂.₇ − α ΔlogP) |
| 5.5 ≤ log P < 8.0 | f₅.₅ × exp[−0.3 (log P − 5.5)] |

Published 1σ uncertainties: Eq. 24 (0.8 < M1 < 2 M⊙), Eq. 25 (M1 > 6 M⊙), interpolated for
2–6 M⊙. They are a natural prior width for rung 3 (MP-Q22). Check value: integrating Eq. 23 at
M1 = 1 M⊙ gives f_mult;q>0.3 = 0.36 (§9.4), which the pilot's unit test reproduces.

### 2.3 Mass ratio p_q(q | M1, P), 0.1 < q < 1

Broken power law (Eq. 2): p_q ∝ q^γ_smallq on 0.1 < q < 0.3 and q^γ_largeq on 0.3 < q < 1,
**continuous at q = 0.3**, plus an excess twin fraction F_twin on 0.95 < q < 1 defined relative to
the power-law component (§2, Fig. 2). F_twin is an *excess* fraction of the q > 0.3 companions,
not a total. With I = ∫₀.₃¹ q^γ_largeq dq, the companion density per decade of P and per unit q is

- 0.3 ≤ q ≤ 1: f_logP;q>0.3 × [ (1 − F_twin) q^γ_largeq / I + F_twin × 1(q ≥ 0.95) / 0.05 ]
- 0.1 ≤ q < 0.3: f_logP;q>0.3 × (1 − F_twin) × 0.3^(γ_largeq − γ_smallq) q^γ_smallq / I

so that ∫₀.₃¹ (…) dq = f_logP;q>0.3 (the normalization in Fig. 2) and the power-law parts join at
q = 0.3.

**γ_largeq** (Eqs. 9–11):

- 0.8 < M1 < 1.2 M⊙ (Eq. 9): −0.5 for 0.2 ≤ log P < 5.0; −0.5 − 0.3 (log P − 5) for 5.0 ≤ log P < 8.0.
- M1 = 3.5 M⊙ (Eq. 10): −0.5 for 0.2 ≤ log P < 1.0; −0.5 − 0.2 (log P − 1) for 1.0 ≤ log P < 4.5;
  −1.2 − 0.4 (log P − 4.5) for 4.5 ≤ log P < 6.5; −2.0 for 6.5 ≤ log P < 8.0.
- M1 > 6.0 M⊙ (Eq. 11): −0.5 for 0.0 ≤ log P < 1.0; −0.5 − 0.9 (log P − 1) for 1.0 ≤ log P < 2.0;
  −1.4 − 0.3 (log P − 2) for 2.0 ≤ log P < 4.0; −2.0 for 4.0 ≤ log P < 8.0.
- Interpolate Eq. 9 ↔ Eq. 10 for 1.2–3.5 M⊙ and Eq. 10 ↔ Eq. 11 for 3.5–6.0 M⊙ "according to M1"
  (linear in M1 vs log M1 is MP-Q10). 1σ: δγ_largeq = 0.3 (Eq. 12).

**γ_smallq** (Eqs. 13–15):

- 0.8 < M1 < 1.2 M⊙ (Eq. 13): 0.3 for 0.2 < log P < 8.0.
- M1 = 3.5 M⊙ (Eq. 14): 0.2 for 0.2 ≤ log P < 2.5; 0.2 − 0.3 (log P − 2.5) for 2.5 ≤ log P < 5.5;
  −0.7 − 0.2 (log P − 5.5) for 5.5 ≤ log P < 8.0.
- M1 > 6.0 M⊙ (Eq. 15): 0.1 for 0.2 ≤ log P < 1.0; 0.1 − 0.15 (log P − 1) for 1.0 ≤ log P < 3.0;
  −0.2 − 0.50 (log P − 3) for 3.0 ≤ log P < 5.6; −1.5 for 5.6 ≤ log P < 8.0.
- Same interpolation in M1 as γ_largeq. 1σ: Eq. 16.

**Twin excess** (Eqs. 5–7), the P dependence:

- F_twin;logP<1 = 0.30 − 0.15 log(M1/M⊙) (Eq. 6)
- log P_twin = 8.0 − M1/M⊙ for M1 ≤ 6.5 M⊙; 1.5 for M1 > 6.5 M⊙ (Eq. 7)
- F_twin = F_twin;logP<1 for log P < 1; F_twin;logP<1 × [1 − (log P − 1)/(log P_twin − 1)] for
  1 ≤ log P < log P_twin; 0 for log P ≥ log P_twin (Eq. 5). 1σ: max{0.03, 0.3 F_twin} (Eq. 8).

Table 13 tabulates these at log P = 1, 3, 5, 7 per spectral type; the pilot's unit tests check the
formulas against it.

### 2.4 Eccentricity p_e(e | M1, P)

- p_e ∝ e^η on 0 ≤ e < e_max(P) (Table 1; §2).
- e_max(P) = 1 − (P / 2 d)^(−2/3) for P > 2 d (Eq. 3; Roche-lobe fill factor ≲ 70% at periastron).
- **Circularization**: binaries with P ≤ 2 d are circular, e = 0 (§2, text after Eq. 3).
- η (0.8 < M1 < 3 M⊙, 0.5 < log P < 6.0) = 0.6 − 0.7/(log P − 0.5) (Eq. 17).
- η (M1 > 7 M⊙, 0.5 < log P < 5.0) = 0.9 − 0.2/(log P − 0.5) (Eq. 18).
- Interpolate in M1 between Eqs. 17 and 18 for 3–7 M⊙. 1σ: δη = 0.3 for 1 < log P < 5 (Eq. 19).

Eq. 17 gives η < −1 (a non-normalizable e^η) for log P ≲ 0.94, and both fits stop at log P = 6
(late) or 5 (early). MdS17 notes η is "not well-defined" for log P ≲ 1 (§9.2). How to evaluate η
outside the fitted ranges is MP-Q11. It barely matters for orbits: gaiamock's cascade fits orbits
only for P ≥ 10 d (`P_min`), and the real orbits lie at 87–1,913 d (0.5–99.5 percentile).

### 2.5 Luminous-companion photometry (flux ratio in G)

`gaiamock_mod` has **no mass–luminosity relation** (checked: `vendor/overlays/gaiamock_mod.py`
takes the G-band flux ratio `f` as an input to `predict_astrometry_luminous_binary` and
`get_a0_mas`; `simulate_many_realizations_of_a_single_binary` takes `Mg_tot` and `f` as inputs).
Candidates (MP-Q13):

- **Janssens et al. (2022) Table 1** M_G(M) for dwarfs, 0.02–57.95 M⊙, already frozen in the repo
  (`config/selections/external/janssens2022_mass_magnitude.yaml`, `janssens_mass.py`), no new
  dependency, no age or metallicity term.
- **MIST** (Choi et al. 2016), as in El-Badry et al. (2024), via `isochrones` or a vendored grid:
  age and [Fe/H] dependence, a new dependency with a multi-GB grid.
- Pecaut & Mamajek (2013) mean dwarf sequence (mass and M_G columns).

The proposal set does **not** depend on this choice: f is a proposal dimension (§3.2) and the
mass–luminosity relation enters only the target density p(f | M1, M2, θ). That density needs a
finite width (age/metallicity spread or an intrinsic scatter σ_f, MP-Q13), because a deterministic
f(M2) cannot be reached by reweighting. Absolute magnitudes need A_G (MP-Q14).

### 2.6 Companion-type mixture

Types k ∈ {MS (luminous), WD, NS, BH}. gaiamock sees only (M2, f), so **the type is not a proposal
dimension**: the target companion density is a mixture

  λ(M2, f | M1, P, θ) = Σ_k π_k(θ) p_k(M2 | M1, θ) p_k(f | M1, M2, θ)

- MS: MdS17 q distribution (§2.3) and the luminous f density (§2.5).
- WD / NS / BH: dark or nearly dark. NS and BH have f = 0 exactly; WD flux is MP-Q15.
- The WD / NS / BH mass densities p_k(M2) and the fractions π_k are **free parameters of
  `population_model`** (ARCHITECTURE.md §4: free-height bins on log M2, WD hard-truncated at M_Ch,
  soft M_TOV for NS). There is no fixed prior from pulsar or LIGO mass functions; those are
  comparison-only (CLAUDE.md). Their P and e distributions are not MdS17 (which describes
  zero-age MS pairs) and are MP-Q16; their mixture at rung 2 is MP-Q17.

### 2.7 Triples off

`P(triple) = 0` in v1 (ARCHITECTURE.md §9). Each parent system gets **at most one** companion.
MdS17's f_mult counts companions in triples too; whether the mock uses the frequency as a Poisson
intensity, or caps at one companion and uses a binary fraction, is MP-Q9.

### 2.8 Orientation

Isotropic: cos i ~ U(−1, 1), Ω, ω ~ U(0, 2π), periastron phase ~ U(0, 1) (El-Badry et al. 2024
§3; the current `draw_mock_binary_params`). These are the same in proposal and target, so they
cancel in every weight.

## 3. Proposal set and importance reweighting

### 3.1 Notation

A draw is x = (s, M1, M2, f, log P, e, ω_geom), with s a parent-snapshot row and ω_geom the
orientation and phase. The target is a Poisson **intensity**: λ(x | θ) dx is the expected number of
companions in dx per parent system. The quantity every comparison needs is the expected number of
systems with outcome O (for example "accepted orbit in six-panel bin b"):

  Λ_O(θ) = Σ_{s ∈ full parent} ∫ λ(x | θ) 1[O(x, noise)] dx.

### 3.2 Proposal densities q(x)

Broad, analytic, evaluable at any x (needed for §3.6), and with q > 0 wherever any admissible θ
puts mass:

| Factor | Proposal |
|---|---|
| q(s) | mixture over snapshot rows: (1 − λ_ϖ) uniform + λ_ϖ ∝ min(ϖ, ϖ_cap)^β (oversamples nearby stars, which dominate detections; the uniform part is defensive, Hesterberg 1995) |
| M1 \| s | fixed by §1.4 (MP-Q5); not reweighted |
| q(log M2) | mixture of log-uniform components on [M2_min, M2_max] covering BDs to BHs, plus a component tied to M1 (log q uniform on [log q_min, 0]) |
| q(f) | point mass at f = 0 with probability ρ_dark (dark companions), else log-uniform on [f_min, f_max] |
| q(log P) | mixture: log-uniform on the detectable range plus a defensive log-uniform on MdS17's 0.2–8 |
| q(e \| P) | e = 0 for P ≤ 2 d (matching MdS17's circular class, §2.4), else U(0, e_cap) |
| geometry | isotropic, identical to the target (§2.8) |

All proposal settings are in config (`config/population/proposal_set_pilot.yaml`, merging into
`selection_function_astrometric.proposal_set` at stage integration). They change
efficiency, never the answer. The support condition is enforced by a test: q(x) > 0 on the union
of the MdS17 domain and the compact-object mass range.

### 3.3 What is stored per draw

One HDF5 row per draw, one artifact per proposal generation, keyed by a config fingerprint like
every stage artifact:

- **identity**: `draw_index` (global, never reused), `generation` (proposal id), parent
  `source_id` and snapshot row;
- **truth**: ra, dec, parallax, pmra, pmdec, phot_g_mean_mag fed to gaiamock, M1 (and its
  provenance), M2, f, P, e, inc, Ω, ω, Tp;
- **seeds** (#371): `(numpy_seed, c_rand_seed)` from
  `forward_model.mock_global_rng_seeds(base_seed, stream, draw_index)` with a stream per generation,
  and the scheme string, so any draw can be replayed alone;
- **cascade outcome**: `forward_model.SolutionType`, the raw 23-element cascade vector (fitted
  parallax, A, B, F, G, P, φ_p, e, inc, a0 and their σ, N_visibility_periods, N_obs, F2, RUWE);
- **flags**: `accepted_orbital` (all `orbital_solution_cuts`), `published_acceleration`
  (El-Badry et al. 2024 §5.2.1: s > 20, and F2 < 22 for 7-parameter solutions);
- **observed solution with σ**: the fitted values and σ above, plus the derived six-panel
  quantities (`astrometric_mass_function` f_m, cos i, 1/ϖ);
- **proposal bookkeeping**: log q(x) per proposal component, so weights are auditable;
- **cost**: CPU seconds per draw, for the compute model.

Truth and outcome are kept for **every** draw, not only accepted ones; denominators and top-up
diagnostics need them.

### 3.4 Weights

With N_draw draws from q (normalized over the snapshot and the continuous dimensions):

  w_i(θ) = (N_full / N_snap) × λ(x_i | θ) / [ N_draw × q(x_i) ]

and Λ_O(θ) ≈ Σ_i w_i(θ) 1[O_i]. This is an absolute expected count **relative to the G < 19
parent**, directly comparable with the 168,065 real orbits (CLAUDE.md: normalization relative to
the parent sample, never a Galactic space density). The orientation factors cancel. Weights are
recomputed from stored truth whenever θ changes; gaiamock is never rerun to change θ.

### 3.5 What cannot be reweighted (fixed at generation)

Reweighting changes densities over x; it cannot change the map from x to the gaiamock input.
These are therefore **generation-time** choices and must be settled before the full-size run:
the parent cuts (MP-Q1–Q3), the truth parallax / distance (MP-Q4), M1 (MP-Q5), the light split
that sets `phot_g_mean_mag` (MP-Q6), the gaiamock version triple and noise model (ELBADRY2024 Q5),
and the cascade settings (`ruwe_min`, `skip_acceleration`). Everything in §2 except these is
reweightable.

### 3.6 Effective sample size and the minimum per bin

Per bin b (observed six-panel bin, solution-type class, or mass bin):

  ESS_b = (Σ_{i∈b} w_i)² / Σ_{i∈b} w_i²   (Kish), with N_b = Σ_{i∈b} w_i.

The project's MC-noise criterion (`physics.mc_noise_threshold`, default 0.1; ARCHITECTURE.md §4
`sensitivity_analysis`) is σ_MC / σ_Poisson < threshold, with σ_MC² = Σ w_i² and
σ_Poisson² = N_b, which is equivalent to

  **ESS_b ≥ N_b / threshold²** (= 100 N_b at the default).

This is the trust criterion for rungs 3–5. It is cheap where N_b is small (compact-object bins) and
infeasible on the laptop for rung 2's luminous bins, where N_b reaches ~10⁴. The display minimum
for rung-2 figures is MP-Q19. Every reweighted histogram reports ESS_b and N_accepted per bin, and
bins below the threshold are marked, never silently drawn as trusted. Global diagnostics: ESS of the
whole accepted set, max_i w_i / Σ w_i, and the Pareto-k̂ of the weight tail (Vehtari et al. 2024)
as a warning when k̂ > 0.7.

### 3.7 Top-up and combining proposals

When a bin fails §3.6 under the θ of interest (MdS17 at rung 2, posterior draws at rung 3):

1. Define a new proposal q_j that concentrates on the under-covered region (for example the target
   λ(x | θ̂) itself, or the old q tilted toward the bin's truth region), always with a defensive
   share of the original q.
2. Simulate n_j new draws with new `draw_index` values and stream j (seeds stay unique).
3. Recompute **all** weights with the deterministic-mixture (balance-heuristic) denominator
   (Veach & Guibas 1995; Owen & Zhou 2000; Elvira et al. 2019):

   w_i(θ) = (N_full / N_snap) λ(x_i | θ) / Σ_j n_j q_j(x_i)

   over every generation j, for every draw i regardless of which q it came from. This needs each
   q_j evaluable at every x, hence analytic proposals recorded in config.
4. Re-check ESS. Never discard draws or regenerate selectively; that biases Λ.

All generations share one parent snapshot.

### 3.8 How this replaces `mock_population` and what it feeds

- `selection_function_astrometric.mock_population` (`elbadry_prior`, the box prior tuned to a
  placeholder fixture, ELBADRY2024 spec §7) is **replaced** by the proposal set as the mock input.
  The box prior is kept only until the new path is wired into the stage, then removed under its
  own issue.
- `selection_function_astrometric`: the stage artifact holds the proposal set (truth + outcome +
  seeds); the six-panel and solution-type gates (#339) are evaluated as weighted histograms at a
  stated θ with §3.6 diagnostics.
- `population_model`: owns λ(x | θ), i.e. the MdS17 densities (§2) and the compact-object mixture,
  as functions evaluated on stored truth.
- `sensitivity_analysis`: computes ESS_b and the MC-noise ratio from the weights instead of fresh
  injections, and triggers §3.7 top-ups.
- `inference`: the expected count in each bin is Σ w_i(θ) 1[i ∈ b] inside the sampler; gaiamock is
  never called inside the likelihood.
- SBC (rung 4) draws its mock "observed" catalogs from a **held-out partition** of the proposal set
  so the selection-function estimate and the mock data do not share noise.

Code placement (one file per concern, `src/` layout): MdS17 densities in
`src/darkhunter_pop/moe_distefano.py` (called by `population_model`), proposal sampling, storage,
weights and ESS in `src/darkhunter_pop/proposal_set.py` (called by `forward_model`), the parent
snapshot in `scripts/fetch_gaia_source_parent.py`.

## 4. Gaiamock interface

Per draw: `gaiamock.run_full_astrometric_cascade(ra, dec, parallax, pmra, pmdec, m1, m2, period,
Tp, ecc, omega, inc_deg, w, phot_g_mean_mag, f, data_release, c_funcs, ruwe_min,
skip_acceleration)` inside `forward_model.seeded_global_rng`, classified by
`forward_model.classify_cascade_result` and cut by `passes_orbital_solution_cuts`. No gaiamock
function is reimplemented (docs/GAIAMOCK_API.md). Real proper motions are passed (the box prior
passed 0).

**Epoch model (#400, not yet chosen).** `docs/EPOCH_MODEL_SPEC.md` calibrates a statistical
transit-loss model (published DR3 gaps + a G-dependent per-transit loss) that wraps gaiamock's
GOST list through `epoch_model.gost_epoch_model`. It is off (`dr3.epoch_model.enabled: false`).
If Ryan switches it on, the call above becomes
`with seeded_global_rng(seeds, c_funcs), gost_epoch_model(gaiamock, em_cfg,
SourceEpochContext(g_mag), epoch_model_rng(base_seed, stream, draw_index)):` in
`proposal_set.simulate_one`, and every stored draw needs re-simulation.

## 5. Validation ladder (each rung gates the next)

| Rung | What | Acceptance |
|---|---|---|
| 1 | **#390** injection test: published DR3 orbits re-injected through gaiamock_mod | #390's criteria: pulls ≈ N(0, 1), recovered σ vs published σ ≈ 1:1, acceptance pattern near threshold interpreted (not tuned). Owned by #390 |
| 2 | **MdS17 at published parameters** (§2 values, luminous companions; compact mixture per MP-Q17) reweighted onto the proposal set | (a) six-panel shape: weighted mock vs real 168,065 Orbital + AstroSpectroSB1 on the `diagnostics.elbadry_six_panel_axes` axes, by eye, with ESS_b and N_accepted shown per panel and §3.6 flags; (b) solution-type mix: mock orbital : published-acceleration counts against DR3's published counts (ELBADRY2024 Q7 for the real denominator); (c) the expected total Λ_orbit vs 168,065, reported, not tuned; (d) by-eye cross-check against El-Badry et al. (2024) Fig. 5, whose mock is known to be too eccentric. Any quantitative statistic and threshold is MP-Q19 |
| 3 | **Fit the full sample**: luminous-binary parameters (the MdS17 θ, MP-Q22) fit to the full NSS sample | Posterior-predictive six-panels and solution-type mix from posterior draws; §3.6 ESS_b ≥ N_b/threshold² in every bin used by the likelihood; PPC thresholds MP-Q22 |
| 4 | **Compact-object injection–recovery (SBC)** | Rank statistics uniform (`sbc.py`), coverage of injected WD / NS / BH dN/dM, mock catalogs from the held-out partition (§3.8) |
| 5 | **Real inference** | Rungs 1–4 green; MC-noise criterion met; dynesty multi-run agreement (ARCHITECTURE.md §4 `inference`) |

## 6. Compute

Measured `gaiamock_mod` cost (ELBADRY2024 spec §8.1, CPU seconds, single process, 0.17 GB RSS):
≈ 0.9 s for a draw ending at the 5-parameter solution, ≈ 13 s (8–14 s) for one reaching the
12-parameter orbital fit, and a mixed 0.6–19 s for acceleration cases. Cost per draw is therefore

  c ≈ 0.9 s + 12 s × P(orbital fit attempted | q).

Under the paper's population ~1% reach the orbital fit (≈ 1 CPU s per draw, 0.3% accepted). The
proposal q is built to oversample detectable systems, so P(orbital fit) is much larger and the cost
per draw higher (≈ 3–6 s expected); the pilot measures it.

| Run | Draws | CPU hours (at 3–6 s) | Wall at 4 workers | Disk (draws) | RSS |
|---|---|---|---|---|---|
| Pilot test | ≈ 2,000 | 2–3.5 | 0.5–1 h on an idle laptop; longer under shared load | < 5 MB | < 1.5 GB |
| Full laptop | ~10⁵ (sized from the pilot ESS) | 85–170 | 1–2 days | ≈ 60 MB | < 1.5 GB |

Parent snapshot: 17–370 MB (§1.3). The laptop is shared (load average 100+ on 10 cores measured on
2026-10-02), so wall clock is the constraint, not disk or memory. The full-size run is not started
without Ryan's approval of the pilot's measured projection.

## 7. Pilot (Phase B)

Implements only what §1–§4 determine: the parent snapshot (small `K`), the MdS17 densities, the
proposal sampler and per-draw storage, and the weights and ESS. A test run of ≈ 2,000 draws
measures cost and checks the plumbing, and produces two figures, both labelled **pilot** with
N_accepted shown: ESS per six-panel bin, and the MdS17-reweighted six-panel against the real
sample (rung 2). Where an open question must have a value to run, the pilot uses a provisional
setting that is named in the config key (`provisional_*`), the artifact and the caption:

Since §0.1, the generation-time rows below (MP-Q1, Q3–Q6, Q13) are superseded by the decisions.
The reweightable rows (MP-Q7, Q9–Q11, Q17) stay provisional.

| Open question | Pilot provisional setting (not a decision) |
|---|---|
| MP-Q1 parallax floor | ϖ > 0.2 mas (a superset; any higher floor is a later filter) |
| MP-Q3 IPD / C\* | not applied |
| MP-Q4 truth parallax | measured `parallax` |
| MP-Q5 M1 | TAG10 point value; stars with no atmosphere dropped |
| MP-Q6 light split | observed G is the total system light |
| MP-Q7 M1 < 0.8 M⊙ | MdS17 shapes at 0.8 M⊙, frequency scaled linearly in log M1 to 0 at 0.08 M⊙ (El-Badry et al. 2024 §3 binary-fraction prescription) |
| MP-Q9 frequency vs fraction | Poisson intensity |
| MP-Q10 interpolation in M1 | linear in M1 |
| MP-Q11 η outside range | clipped below at η = −0.9; formulas evaluated as written elsewhere |
| MP-Q13 mass–luminosity | Janssens et al. (2022) M_G(M) for both stars, log-normal f scatter of 0.1 dex |
| MP-Q14 extinction | not needed by the pilot (f from both masses; gaiamock gets the observed G) |
| MP-Q17 compact mixture at rung 2 | none (luminous MdS17 only) |

### 7.1 Pilot measurements (2026-10-02, laptop shared at load average 40–120 on 10 cores)

Parent snapshot `20261002T192654Z_gaia_source_parent_K1000000_plx0p2`: 214,666 rows (exactly the
count in §1.1's slice), 200,602 usable (TAG10 M1 resolved: 142,420 MSC, 58,182 GSP-Phot, 14,064
none). On disk 64 MB (`parent.h5`, all float64, ≈ 300 B per row, so the §1.3 sizes are ≈ 1.8×
low) plus 8.6 MB of cached M1. The TAG10 + Santos M1 has a hard floor at 0.597 M⊙, which hits
more than 25% of the parent (#393).

| Generation | Draws | Accepted orbits | CPU s per draw (mean) | Wall (4 workers) | ESS of its accepted draws, mixture weights |
|---|---|---|---|---|---|
| 0 (`proposal_set_pilot.yaml`) | 2,000 | 70 (3.5%) | 2.00 | 36.9 min | 2.18 |
| 1 top-up (`proposal_set_pilot_gen1.yaml`: σ_f 0.15 dex, ϖ¹ tilt) | 1,000 | 8 (0.8%) | 0.69 | 6.8 min | 3.04 |
| Combined | 3,000 | 78 | 1.57 | | **4.71** |

- Cost by outcome: a draw ending at a 5-, 7- or 9-parameter solution costs **0.01 CPU s**. One that
  reaches the 12-parameter fit costs **≈ 18 CPU s**, whether accepted or not. The ≈ 0.9 s figure for
  5-parameter draws from ELBADRY2024 §8.1 is not reproduced here. Its scripts timed a different
  harness, and the discrepancy is not investigated. So c ≈ 18 s × P(orbit fit). Peak RSS was
  0.30 GB for the parent process plus ≈ 0.22 GB per worker. Disk was ≈ 0.9 kB per draw.
- Weight diagnostics (generation 0): the flux-ratio factor and the parallax tilt of q(s) set the
  weight scatter. Only 10% of accepted draws lie within 0.2 dex of the provisional mass–luminosity
  relation, because accepted systems favour faint companions. The top-up was retuned on that
  basis.
- Expected accepted orbits under MdS17 (luminous, provisional settings), relative to the parent:
  7.6 × 10⁴, against 168,065 real. This is dominated by a few weights and is not a measurement.
- **Efficiency**: generation 0 gave 2.0 ESS per CPU hour; generation 1 gave 16. The figure for
  generation 1 rests on 8 accepted draws, so it is uncertain by a factor of about 3.

**Projection for the full laptop run (not started; MP-Q23).** At generation 1's efficiency, an
accepted-set ESS of 2,000 (≈ 100 per six-panel bin over 20 bins, a display-level target for
MP-Q19) needs ≈ 125 CPU h. That is ≈ 31 h wall at 4 workers, or ≈ 16 h at 8 (≈ 2 GB RSS), with
≈ 0.6 M draws and ≈ 0.5 GB of disk. The MC-noise rule at rung 2 (ESS_b ≥ 100 N_b with N_b ~ 10⁴)
stays out of reach on the laptop by a factor of ~10⁵. A further 2–3k-draw tuning generation
(≈ 1 h) would firm up the efficiency before the full run is sized.

## 8. Open questions for Ryan (MP-Q1–Q6 and Q13 decided, §0.1; none of the rest chosen)

- **MP-Q1**: decided 2026-10-02, see §0.1.
- **MP-Q2**: decided 2026-10-02, see §0.1.
- **MP-Q3**: decided 2026-10-02, see §0.1.
- **MP-Q4**: decided 2026-10-02, see §0.1.
- **MP-Q5**: decided 2026-10-02, see §0.1.
- **MP-Q6**: decided 2026-10-02, see §0.1.
- **MP-Q7 Primaries below 0.8 M⊙**, outside MdS17. Options: El-Badry et al. (2024)'s linear decline
  in log M1 to 0 at 0.08 M⊙; an M-dwarf multiplicity survey (e.g. Winters et al. 2019); MdS17's
  0.8 M⊙ values held flat; or free parameters at rung 3.
- **MP-Q8 Companions with q < 0.1** (brown dwarfs, and low-mass M2 around massive M1), outside
  MdS17. Include, and with what density?
- **MP-Q9 Frequency vs fraction** with triples off: use f_logP as a Poisson intensity (allowing
  more than one expected companion), or cap at one and rescale to a binary fraction
  (Table 13 F_n=1 or F_n≥1)?
- **MP-Q10 Interpolation in M1** between the anchor equations (9↔10, 10↔11, 13↔14, 14↔15,
  17↔18): linear in M1, which is the plain reading of "according to M1", or linear in log M1?
- **MP-Q11 η outside its fitted range**: Eq. 17 is < −1 for log P ≲ 0.94 and both fits stop at
  log P = 5–6. Clip η, hold it at the range edge, or extend the formula?
- **MP-Q12 e > e_max(P)**: the proposal covers e up to `e_cap`. Must every θ respect Eq. 3, or may
  alternative eccentricity models (El-Badry's e^0.2, thermal) exceed it?
- **MP-Q13**: decided 2026-10-02, see §0.1.
- **MP-Q14 Extinction** for absolute magnitudes: Combined19 (current) or `ag_gspphot`?
- **MP-Q15 WD flux**: f = 0, or a G-band flux from the Bédard et al. cooling tracks already used by
  `companion_nature`?
- **MP-Q16 P and e for compact companions**: the MdS17 forms, separate free families, or El-Badry
  et al. (2024)'s WD prescription (10% of 2–6 au WD binaries survive at 50% of initial a)?
- **MP-Q17 Compact mixture at rung 2**: none, or a WD share (MdS17 §8.4 estimates a few per cent of
  solar-type primaries have WD companions)?
- **MP-Q18 MdS17's C_evol** and the 5–30% of apparent primaries that are original secondaries
  (§2): ignored in v1?
- **MP-Q19 Rung-2 acceptance**: by eye only (as the paper), or a quantitative statistic (weighted KS
  or per-bin pulls with MC error) with a threshold; and the minimum ESS_b for drawing a rung-2 bin,
  given the 0.1 MC-noise rule needs ESS_b ≥ 100 N_b (infeasible for luminous bins on the laptop).
  Inherits ELBADRY2024 Q10.
- **MP-Q20 Solution-type mix denominator**: inherits ELBADRY2024 Q7, Q7a–c (needs an
  `nss_acceleration_astro` snapshot).
- **MP-Q21 AstroSpectroSB1**: compared together with Orbital (as El-Badry did), or is the RV-chain
  input (G_RVS) modelled for that subset?
- **MP-Q22 Rung 3 parameterization**: which MdS17 coefficients are free (all, or e.g. the
  f_logP anchors, γ_largeq, F_twin, η), their priors (the published 1σ of Eqs. 8, 12, 16, 19, 24,
  25 are available), and the posterior-predictive acceptance thresholds.
- **MP-Q23 Full laptop run**: approved 2026-10-02 (one ~1 h tuning generation, then the ~125 CPU-h run; §0.1 comment).
- **MP-Q24 Real-side Halbwachs (b)/(c) residue**: remove the 114 real rows (0.06%) that fail the IPD/C\* cuts on DR3 values, for exact symmetry with the parent, or keep El-Badry et al. (2024)'s "no further cut"? Measured in §0.1.

## References

- Bailer-Jones, C. A. L. et al. 2021, AJ 161, 147.
- Choi, J. et al. 2016, ApJ 823, 102 (MIST).
- El-Badry, K. et al. 2024, OJAp 7, 100 (arXiv:2411.00088).
- Elvira, V., Martino, L., Luengo, D. & Bugallo, M. 2019, Statistical Science 34, 129.
- Gaia Collaboration, Arenou, F. et al. 2023, A&A 674, A34.
- Halbwachs, J.-L. et al. 2023, A&A 674, A9 (arXiv:2206.05726).
- Hesterberg, T. 1995, Technometrics 37, 185.
- Janssens, S. et al. 2022, A&A 658, A129.
- Moe, M. & Di Stefano, R. 2017, ApJS 230, 15 (arXiv:1606.05347).
- Owen, A. & Zhou, Y. 2000, JASA 95, 135.
- Pecaut, M. J. & Mamajek, E. E. 2013, ApJS 208, 9.
- Riello, M. et al. 2021, A&A 649, A3.
- Veach, E. & Guibas, L. 1995, SIGGRAPH '95, 419.
- Vehtari, A. et al. 2024, JMLR 25, 72 (Pareto-smoothed importance sampling).
- Winters, J. G. et al. 2019, AJ 157, 216.
