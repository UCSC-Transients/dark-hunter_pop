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

The magnitude-limit (Malmquist / Öpik) conditioning that Ryan made a condition of accepting
Gaia-star primaries (#405) is derived in §9; its open choices are MP-Q25–Q32.

### 0.2 Decisions of 2026-10-03 (Ryan, #391 comment 5971280434)

Recorded at https://github.com/UCSC-Transients/dark-hunter_pop/issues/391#issuecomment-5971280434.

| Question | Decision | Implementation (PR for #391 / #405 / #409 / #410) |
|---|---|---|
| MP-Q24 | Drop the 114 real rows failing the IPD / C\* cuts (symmetry) | `scripts/fetch_real_nss_input_columns.py` snapshots the columns; `proposal_set.real_comparison_keep(..., input_columns)` applies `halbwachs_input_flags` joined by `source_id` (missing rows fail) |
| MP-Q19 | Minimum ESS per bin = 30, for display shading and the KS gate | `config/population/rung2_validation.yaml`. Bins with ESS_b < 30 are shaded grey. The weighted KS is computed only over the bins with ESS_b ≥ 30, and the report states the fraction of the real sample inside them. The §3.6 MC-noise rule stays the criterion for rungs 3–5 |
| MP-Q29 | Combined19 extinction for ΔM | `proposal_set.combined19_a_g`: mwdust Combined19 at (l, b, Bailer-Jones d), A_G = 2.8 E(B−V) (El-Badry et al. 2024 §3). σ_A = 0, because the extinction scatter is absorbed into the fitted σ_int |
| MP-Q25 | Fit σ_int and the M_G zero point on the parent's RUWE < 1.4 stars | `scripts/fit_malmquist_zero_point.py` → `proposal_set.fit_mg_zero_point` (Gaussian ML, bootstrap errors, giants excluded). **Result (docs/gate391/malmquist_zero_point_fit.json): zp = −2.357 ± 0.006 mag, σ_int = 2.082 ± 0.007 mag on 161,776 stars.** The statistical precision is far inside the closed loop's 0.05 mag tolerance, but the fit is dominated by systematics: median ΔM runs from +0.64 (d < 0.5 kpc) to −2.94 mag (2–5 kpc), and from −3.0 (M1 < 0.6) to +0.2 (M1 > 2). Causes are the TAG10 floor (#393) and evolved stars fitted by MSC as dwarfs (MP-Q28). Escalated as **#414**. The values are recorded in `config/population/malmquist_decided.yaml` with that warning |
| MP-Q30 | Gaussian distance marginalization | `gaussian_mu`, enforced by `proposal_set.malmquist_log_weight` |
| MP-Q26 | Ignore the TAG10 blended-light bias for now; document it | Not modelled. TAG10 run on an unresolved pair's blended atmosphere is biased; the conditioning uses M̂1 as if unbiased. Revisit with #393 / #414 |
| MP-Q32 | Install `gaiaunlimited` for the volume-limited diagnostic | Installed in the shared `.venv` (0.3.3, numpy kept at 1.26.4; it pulled in pandas 3.0.6, xarray, astromet, astropy_healpix 1.1.3). New `pyproject` extra `volume_diagnostic` |
| MP-Q27 | Later | `provisional_blending: all_unresolved` unchanged |
| MP-Q28 | Giants: match the real giant population | Owned by another agent. The hook is `malmquist` `provisional_giant_policy: unit_weight` plus `ParentSnapshot.is_giant` |
| #409 / #410 | Fix the eccentricity proposal coverage and the η floor | §3.2: new `eccentricity.shape: mds17_bounded` |
| #400 E1 | Epoch model on | `proposal.epoch_model: dr3_config` wraps each cascade call in `epoch_model.gost_epoch_model` (runner hook; `dr3.epoch_model` with `enabled` forced on) |
| #408 | `threadpoolctl` and `nice` | The runner pins every BLAS/OpenMP pool to one thread per worker, verifies it with `threadpool_info` (recorded per worker in the artifact), and applies `os.nice(10)` |

### 0.3 Decisions of 2026-10-03 on M1, the Malmquist weight and giants (Ryan, #418, #413)

Recorded in https://github.com/UCSC-Transients/dark-hunter_pop/issues/418 and the #413 thread.

| Question | Decision | Where |
|---|---|---|
| MP-Q5 / MP-Q28c M1 | **MIST isochrones** (v1.2, vvcrit 0.4, full isochrones, UBVRIplus Gaia bolometric corrections), on the dereddened CMD, for the mock parent **and** the data side. TAG10 is unusable for the parent: the 0.597 M⊙ floor (#393), giants ≈ 2× too low (§10.1), and the −2.36 mag / 2.08 mag MP-Q25 fit (#414) | §11.2, §11.7 |
| Malmquist weight | "Do the normal stellar-locus binary check, but a binary will not just shift M_G, but also BP−RP. Both have to be taken into account along with extinction." The weight works in the dereddened 2-D CMD against the measured single-star ridge, with MIST companion displacement vectors and extinction as its own vector | §11.4 |
| MP-Q28a | n_σ = 3 for the CMD evolved classifier | §10.2 |
| MP-Q28b | A free period floor for giants, fit at rung 3 | §10.5 |
| MP-Q28f | Giants are compared on Orbital only | §10.3 |

These replace the TAG10 route of §0.1 MP-Q5, the 1-D weight of §9.3 for the decided pipeline, and
the MP-Q25 fit of §0.2 (its procedure is superseded by the ridge calibration, §11.6). MP-Q26 (TAG10
blended-light bias ignored) is superseded by §11.3. Literature reproduction paths keep their own M1
(column ownership; CLAUDE.md).

### 0.4 Decisions of 2026-10-04 on the §11 options (Ryan, #418)

Recorded at https://github.com/UCSC-Transients/dark-hunter_pop/issues/418#issuecomment-5982025304. The implementation plan is §11.9.

| Question | Decision |
|---|---|
| MP-Q39 | `mist_density_ridge_anchored` |
| MP-Q37 | the companion and the primary are **coeval**: they share the same age and [Fe/H] draw; no fiducial isochrone |
| MP-Q36 | re-fit the deblended primary per draw. Given q and the shared age and [Fe/H], solve on the coeval MIST isochrone for the primary mass whose combined G and BP−RP match the observed system. Keep it cheap; an SED fit can come later |
| MP-Q35 | one posterior draw of the primary per proposal draw |
| MP-Q34 | try per-star BP/RP flux errors. Reddening errors are poorly known and 3-D extinction is needed: evaluate Bayestar19 posterior samples and Edenhofer et al. (2024) for an E(B−V) uncertainty |
| MP-Q33 | GSP-Phot [M/H] with the Andrae et al. (2023) calibration where it is reliable, else a solar-neighbourhood MDF |
| MP-Q38 | first determine what the blue rows are |
| MP-Q28d | extend the proposal's support for evolved rows |
| pipeline | use isochrone M1 in the pipeline: flip `mass_calibration.method: MIST_isochrone` once #425 is fixed |
| validation | download DEBCat and APOKASC-3 (approved by Ryan in the issue) |

### 0.5 Decisions of 2026-10-05 (Ryan, relayed by the orchestrator, #418)

| Question | Decision | Where |
|---|---|---|
| MP-Q40 | Take the MS companion's G-flux ratio from the **same coeval MIST isochrone** as the primary: log10 f = −0.4 [M_G(M2) − M_G(M1)], keeping the decided 0.1 dex scatter. This **replaces MP-Q13 (Janssens)** for MS companions. Evolved rows keep §10.4 | §11.9 |
| downloads | approved for DEBCat, APOKASC-3, the Culpan et al. (2022) hot subdwarfs, Bayestar19, the Gaia flux-error and GSP-Phot columns, and `gdr3apcal`. **Edenhofer et al. 2024 is not approved** | §11.9 |
| extinction map | Bayestar19 is evaluated against Combined19. The default map does not change without asking | §11.9 |

### 0.6 Decision of 2026-10-06: Combined19 E(B−V) units (Ryan, relayed by the orchestrator, #418)

"Yes to the 0.884 units fix."

- **What mwdust returns.** `mwdust` maps return E(B−V) on the SFD scale (mwdust README). In the
  north, Combined19 is Bayestar19 (Green et al. 2019), whose unit is SFD-like (E(g−r) = 0.901 E). We
  measured Bayestar19 / Combined19 = 0.884 exactly on the parent and on the real orbits, so in the
  north Combined19 returns Bayestar's native unit.
- **The conversion.** #295 already converts Bayestar19 to E(B−V) with 0.884 for
  `sample_selection`'s `green2019` map (Argonaut usage page: E(B−V) = 0.981 E(g−r)_P1, Schlafly &
  Finkbeiner 2011). That place is not converted again.
- **The decision.** Every pipeline consumer of `mwdust.Combined19` E(B−V) multiplies by the same
  0.884, set in config as `sample_selection.dust_maps.combined19_native_to_ebv`:
  - the dereddened CMD (`giants.cmd_for_rows`), and through it the isochrone M1 (mock parent and
    `mass_derivation_bulk`);
  - the evolved classifier;
  - the 2-D weight.
- **Southern and inner-Galaxy pixels.** Combined19 takes these from Marshall et al. (2006) and
  Drimmel et al. (2003). mwdust also puts them on the SFD scale, so the same factor is applied; this
  is recorded as an assumption.
- **What stays unchanged.**
  - The El-Badry et al. (2024) forward-model procedure (`selection_function_astrometric`
    `extinction_model: combined19` with A_G = 2.8 E(B−V), and `proposal_set.combined19_a_g` for the
    legacy 1-D weight), which follows that paper's own recipe.
  - The literature reproduction paths, which own their extinction (`sample_selection.dust_maps.maps`).

  Every literature count is re-measured to confirm nothing moves.

### 0.7 Tasks of 2026-10-07 (Ryan, relayed by the orchestrator, #418): calibrations and the distance test

- **The giant prior is calibrated against APOKASC-3.** The prior gains two knobs:
  `isochrone_mass.age.provisional_age_power`, an extra (age / 1 Gyr)^γ on the constant-SFR age
  weight, and `isochrone_mass.provisional_cheb_weight` ρ, a weight on core-He-burning points. They
  are fitted on half of the APOKASC-3 giants (even KIC) to zero the median log(M̂ / M_seis) of RGB
  and RC separately, and tested on the other half (odd KIC).
  - Result: γ = 0.5 is adopted. ρ has no measurable effect, so it stays 1. Over ρ = 0.4–1.6 the
    held-out medians move by less than 0.01, and the grid's formal optimum at ρ = 1.6 wins by a
    negligible margin at the grid edge.
  - Held-out M̂ / M_seis at (γ, ρ) = (0.5, 1): RGB 1.039 → 0.972 and RC 1.151 → 1.065. The RC − RGB
    offset of about 9% is not removed by ρ.
  - The residual trend with seismic mass (M̂ pulled toward about 1.2 M⊙) is what a posterior mean
    does when the CMD weakly constrains mass. A prior cannot remove it, so it is reported, not tuned.
  - DEBCat (deblended with the dynamical q) moves from 0.997 (0.043 dex) to 0.982 (0.041 dex).
    FLAME is re-checked in `docs/gate418`.
- **The calibrated [Fe/H] (gdr3apcal, MP-Q33) is tested on DEBCat.** The likelihood is switched on
  only if it improves deblended M1 / M_dyn.
  - Upper bound: DEBCat's own spectroscopic [M/H] with σ = 0.1 cuts the scatter from 0.046 to
    0.038 dex on 123 systems, and moves the bias from −0.009 to −0.014 dex.
  - The gdr3apcal test needs GSP-Phot columns for the DEBCat stars. Until it is run, the [Fe/H]
    likelihood stays off.
- **Distance test.** The closed loop now places its universe in the real map with
  `dust.kind: combined19`: Combined19 × 0.884, and the pipeline sees σ(E)/E = 0.085. Under G < 19
  and the parent cuts, the synthetic parent reproduces the real ridge-residual trend with distance:

  | d (kpc) | synthetic median (singles) | real median / mode |
  |---|---|---|
  | 1–2 | −0.15 | — |
  | 2–5 | −0.58 (−0.47) | −0.34 / −0.45 |
  | > 5 | −1.16 (−1.12) | −1.10 / −1.01 |

  **The trend is selection** (the magnitude limit acting on evolution), so no extinction change is
  needed. In the real-map closed loop the worst pull is 7.6 for the 2-D weight (8.8 for 1-D, 6.7
  for none), concentrated at the brightest log f and the top M2 bins. The ≤ 3 target is not yet met.

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
| q(e \| P) | e = 0 for P ≤ 2 d (matching MdS17's circular class, §2.4). Otherwise (since #409/#410, `shape: mds17_bounded`) a mixture of: a power law (η_q + 1) e^η_q / E^(η_q+1) with η_q = the target η floor; U(0, E), where E = e_max(P) from Eq. 3 (the target's own support); and a defensive U(0, 0.999) for alternative models (MP-Q12). The weights are bounded by (η+1)/((η_q+1) w_floor) near e = 0 and (η+1)/w_support elsewhere. The pilot and paused-run `U(0, e_cap)` truncated the target above 0.95 (~6% of companions, #409) and had infinite-variance weights for η < −0.5 (#410) |
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

## 8. Open questions for Ryan (MP-Q1–Q6, Q13 decided §0.1; MP-Q19, Q24–Q26, Q29, Q30, Q32 decided §0.2; MP-Q28a/b/c/f decided §0.3; MP-Q25–Q32 from §9; MP-Q33–Q39 from §11, decided §0.4; MP-Q40 open, §11.9; MP-Q41–Q45 from §12)

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
- **MP-Q19**: decided 2026-10-03, see §0.2.
- **MP-Q20 Solution-type mix denominator**: inherits ELBADRY2024 Q7, Q7a–c (needs an
  `nss_acceleration_astro` snapshot).
- **MP-Q21 AstroSpectroSB1**: compared together with Orbital (as El-Badry did), or is the RV-chain
  input (G_RVS) modelled for that subset?
- **MP-Q22 Rung 3 parameterization** (provisional defaults in §12, run 2026-10-09; the eccentricity part is resolved by §12.11, Beta per log P range): which MdS17 coefficients are free (all, or e.g. the
  f_logP anchors, γ_largeq, F_twin, η), their priors (the published 1σ of Eqs. 8, 12, 16, 19, 24,
  25 are available), and the posterior-predictive acceptance thresholds.
- **MP-Q23 Full laptop run**: approved 2026-10-02 (one ~1 h tuning generation, then the ~125 CPU-h run; §0.1 comment).
- **MP-Q24**: decided 2026-10-03, see §0.2.
- **MP-Q25**: decided 2026-10-03, see §0.2.
- **MP-Q26**: decided 2026-10-03, see §0.2.
- **MP-Q27 Resolved pairs and the IPD flags** (§9.1, §9.4 item 4). This decides which drawn
  companions blend into G and which are consistent with the row's own `ipd_frac_multi_peak` /
  `ipd_gof_harmonic_amplitude`. Options: (a) all blend and the IPD values carry no information
  (now); (b) blend only below an angular-separation threshold set from Fabricius et al. (2021)'s
  close-pair completeness (about 0.7–1.5″, contrast-dependent), with resolved companions given
  f_b = 0; (c) also add p(ipd_s | c) as a likelihood term (needs an IPD model; none is published
  for this purpose).
- **MP-Q28a, b, c, f**: decided 2026-10-03, see §0.3.
- **MP-Q33–Q39** (isochrone M1 and the 2-D weight, #418): options in §11.8.
- **MP-Q28 Giants**: decided 2026-10-03 (match the real giant population); treatment and the
  remaining options MP-Q28a–g are in §10 (#413). Original framing: (`is_giant`, §0.1). M_G^J is a dwarf relation, so ΔM is meaningless for giants.
  Options: (a) W = 1 for giants, the naive draw, flagged; (b) leave giants out of any statistic
  that relies on W; (c) use a giant M_G(M1, log g) relation.
- **MP-Q29**: decided 2026-10-03, see §0.2.
- **MP-Q30**: decided 2026-10-03, see §0.2.
- **MP-Q31 F > 1 under a Poisson intensity** (MP-Q9). Z_s needs a probability of no companion,
  1 − F ≥ 0. Options: cap at one companion and rescale to a binary fraction; or keep the
  intensity, with p(∅) = exp(−F) and a single-companion approximation, and record where F > 1.
- **MP-Q32**: decided 2026-10-03, see §0.2.

## 9. Magnitude-limit (Malmquist / Öpik) conditioning (#405)

Ryan accepts Gaia-star primaries (§0, §0.1) **only if the magnitude-limit bias is properly
accounted for** (#405). This section derives the importance weight that does that, says exactly
what is conditioned on what, and lists every choice the derivation and the cited papers do not
fix (MP-Q25–Q31, §8). Nothing here changes a §0.1 decision.

### 9.1 Model and notation

- **System**: primary ψ = (M1, ε), where ε is the primary's offset from the single-star
  mass–luminosity relation (age, [Fe/H]), so M_G,1 = M_G^J(M1) + ε with M_G^J the Janssens et al.
  (2022) relation (MP-Q13) and ε ~ N(0, σ_int); companion state c ∈ {∅} ∪ C with
  c = (M2, P, e, f, type, orientation); position r = (d, Ω).
- **Universe**: intensity n(ψ, c, r) = ρ_*(r) φ(ψ) π(c | M1, θ), with π(∅ | M1) = 1 − F(M1) and
  π(c | M1) = λ(c | M1, θ) on C, F = ∫ λ dc. At most one companion (triples off, §2.7), which
  needs F ≤ 1 (MP-Q9, MP-Q31). λ is MdS17 (§2) plus the compact mixture (§2.6). MdS17's
  frequencies are **volume-limited, Öpik-corrected** statistics: their §3.4 uses spectroscopic
  surveys that already removed distant twins or used fixed-distance clusters, §5.2 drops the two
  Sana et al. (2014) binaries that would fall below the H = 7.5 limit on the primary's light alone,
  §7.1 notes the De Rosa et al. (2014) sample is volume-limited within 75 pc, and the solar-type
  sample of §8 is the volume-limited Raghavan et al. (2010) sample. So π(c | M1) is the
  per-primary companion distribution **before** any magnitude selection.
- **Observables** of a `gaia_source` row: o = (G, ϖ_obs, a, Ω), with a the MSC / GSP-Phot
  atmosphere. The pipeline derives M̂1 = TAG10(a) (§1.4, MP-Q5) and d̂ = the Bailer-Jones et al.
  (2021) geometric distance from (ϖ_obs, Ω) (MP-Q4).
- **Blended light**: G = M_G,1 − 2.5 log10(1 + f_b) + μ(d) + A_G(r), where μ = 5 log10(d / 10 pc)
  and f_b = f when the pair is unresolved by Gaia (MP-Q27 for resolved pairs).
- **Parent selection** S: G < 19, ϖ_obs > 0.2 mas, the Halbwachs (b)/(c) flags (§0.1) and Gaia's
  source detection P_det(G, Ω) (crowding and scanning law; Cantat-Gaudin et al. 2023;
  Boubert & Everall 2020). Every factor is a function of the row's observables: P(S | o).

### 9.2 The identity that settles which factors enter

The real parent is a Poisson draw with intensity
n_par(o) = ∫ n(ψ, c, r) p(o | ψ, c, r) P(S | o) dψ dc dr. A mock draw keeps a row's o_s (real G,
real distance, real sky position and real M̂1, §0.1 MP-Q4–Q6) and replaces its unknown companion
state with a drawn c. The correct distribution for that c is

  p(c | o_s, S) = P(S | o_s) p(c, o_s) / [P(S | o_s) p(o_s)] = **p(c | o_s)**.

Because S depends on observables only and the mock conditions on **all** of them, **P(S | o)
cancels**: no detection-volume ratio and no completeness map enters the per-row weight. The
magnitude-limit bias does not go away; it lives entirely in the gap between p(c | o_s) and the
volume-limited π(c | M̂1_s). A row that is over-luminous for its M̂1 at its distance is more
likely to be a luminous binary, and the magnitude-limited parent contains more such rows near
G = 19 because they **are** the parent. Averaging p(c | o_s) over the real rows reproduces the Öpik
boost exactly (§9.5).

Two consequences:

1. **The naive mock is biased.** Drawing c ~ π(c | M̂1_s) and weighting by λ / q alone is
   the volume-limited conditional. It ignores what G says about c at the row's distance, so it
   under-predicts luminous companions in the parent wherever the magnitude limit binds.
2. **The "detection-volume ratio" form is the other branch of MP-Q6.** If gaiamock were fed the
   primary's own light plus the companion (G = G_1 − 2.5 log10(1 + f)), the row's G would not be
   conditioned on and the weight would be p(c | M1, d, Ω, S) ∝ π(c | M1) P(S | M1, c, d, Ω): the
   ratio of selection probabilities, which in a homogeneous Euclidean volume with no ϖ floor
   becomes (1 + f)^{3/2} (Öpik 1923). Under MP-Q6 as decided (G = observed total light) that form
   is **not** consistent; the companion must instead be made consistent with the observed G.

### 9.3 The weight

**Superseded for the decided pipeline by the 2-D weight of §11.4 (#418, §0.3).** The 1-D form below
is kept as the derivation and for the #405 tests; its M_G^J(M̂1) residual is what failed in #414.

Exactly, p(c | o) = ∫ dψ dr ρ_* φ π(c | ψ) p(o | ψ, c, r) / Z(o). Three approximations make it
computable. Each is named; where it is a choice, it is an open question.

- **(A1) M1 = M̂1.** The atmosphere fixes the primary's mass: p(M̂1 | M1, c) is narrow and
  unbiased compared with how fast φ and π vary. This is MP-Q5's decision (the TAG10 point value is
  the truth M1 fed to gaiamock). TAG10 run on a blended atmosphere is not unbiased (MP-Q26).
- **(A2) Distance.** p(μ | ϖ_obs, Ω) is the Bailer-Jones geometric posterior, taken as Gaussian in μ
  with σ_μ = (5 / ln 10) (r_hi − r_lo) / (2 r_med) from `r_lo_geo`, `r_med_geo`, `r_hi_geo`
  (16th / 50th / 84th percentiles). The **geometric** posterior uses no photometry
  (Bailer-Jones et al. 2021), so it carries no single-star colour–magnitude assumption; the
  photogeometric one does and must not be used here. Marginalizing d with the geometric posterior
  is exact if its prior matches ρ_* along the line of sight. It is not exact when ϖ/σ_ϖ is small and
  the posterior is skewed (MP-Q30).
- **(A3) Extinction** A_G is known to σ_A (MP-Q14, MP-Q29).

Then, for row s and companion c,

  **p(c | o_s) = π(c | M̂1_s) × W_s(c),  W_s(c) = L_s(f_b(c)) / Z_s**

  L_s(f) = N( ΔM_s + 2.5 log10(1 + f) ; 0, σ_s )

  ΔM_s = G_s − μ(d̂_s) − A_G,s − M_G^J(M̂1_s)   (the row's luminosity excess; < 0 is over-luminous)

  σ_s² = σ_int² + σ_μ,s² + σ_A,s² + (∂M_G^J / ∂log10 M1)² σ²_log M̂1,s

  Z_s = (1 − F(M̂1_s)) L_s(0) + ∫ λ(c | M̂1_s, θ) L_s(f_b(c)) dc
      = (1 − F_lum(M̂1_s)) L_s(0) + ∫ λ_f(log10 f | M̂1_s, θ) L_s(f) d log10 f.

λ_f is the luminous part of λ marginalized onto log10 f (over M2, P, e and the f scatter at fixed
M1); F_lum is its integral. Dark companions (WD with f = 0, NS, BH) have L_s(0), the same as a
single star, so **they get no magnitude-limit boost**, which is physically right: they add no light
(a luminous WD is MP-Q15). Z_s depends on θ through λ, so it is recomputed with the weights at every
θ. It is a one-dimensional integral per row, evaluated on a grid in M1 (§9.6).

With the proposal set (§3.4, §3.7) the full weight is

  **w_i(θ) = (N_full / N_snap) λ(x_i | θ) W_{s(i)}(x_i; θ) / Σ_j n_j q_j(x_i)**

and the per-row probability of having **no** companion is p(∅ | o_s) = (1 − F) L_s(0) / Z_s.
Statistics that count systems (a binary fraction) need it. Σ_i w_i 1[O_i] stays an expected count
relative to the G < 19 parent (CLAUDE.md normalization).

### 9.4 The four effects in #405, one by one

1. **Binary-fraction boost near the limit.** It is carried by L_s, with no separate factor. Rows
   near G = 19 at large d are, as a population, more over-luminous, and W raises their companion
   probability. The boost depends on the light a companion adds, so it is strong for twins
   (f ≈ 1, 0.75 mag) and negligible for f ≲ 0.05.
2. **M1 from blended light.** The data side and the mock apply TAG10 to the **same** row's
   atmosphere, so the M2 inferred from a0 and M̂1 is computed the same way on both sides; that is
   the consistency MP-Q5 asks for. What (A1) adds is that M̂1 is also the **truth** M1 behind the
   orbit and the conditioning. TAG10 uses T_eff, log g and [M/H] (Torres et al. 2010), but
   GSP-Phot's log g uses the parallax and G, so a companion's light leaks into log g and from there
   into M̂1. The orbit scales only as a ∝ (M1 + M2)^{1/3}, so a 10% error in M1 changes a0 by about 3%;
   the conditioning error enters through M_G^J(M̂1) and is absorbed in σ_s only if it is unbiased
   (MP-Q26).
3. **Distance prior and parallax-floor truncation.** The 0.2 mas floor is on the observed ϖ, so it
   cancels like every other factor of S. The luminosity-dependent truncation it causes (bright
   primaries reach ϖ = 0.2 mas before G = 19, so the Öpik boost switches off for them) is
   reproduced automatically, because the rows are the truncated parent. The distance prior enters
   only through (A2).
4. **Gaia detection completeness.** It cancels in W because it depends on (G, Ω) only. It is needed
   only to invert the parent back to a volume-limited population (§9.5), which is a diagnostic: the
   forward model never needs it. The published DR3 parent source-detection model is
   Cantat-Gaudin et al. (2023): S(G | M10) = 1 − ½ [tanh((x(M10) − G) / y(M10)) + 1]^{z(M10)}, with
   M10 per HEALPix pixel. Over most of the sky M10 ≈ 19–21.5, so G < 19 is nearly complete outside
   crowded fields. It is distributed in `gaiaunlimited`
   (`gaiaunlimited.selectionfunctions.DR3SelectionFunctionTCG`; version 0.3.3 on PyPI, **not
   installed**, needs `healpy` (already in `.venv`) and downloads `allsky_M10_hpx7.hdf5` on first
   use). The Everall & Boubert series covers DR2 source detection (Boubert & Everall 2020, Paper II)
   and the EDR3 **astrometry and RVS subsample** selection functions (Everall & Boubert 2022, Paper
   V). Paper V is a subsample model, not the parent. Neither model covers **companion-dependent**
   detection, meaning a close pair resolved into two sources (Fabricius et al. 2021: completeness
   for close pairs drops below about 1.5″ and falls fast below 0.7″) or flagged by the IPD cuts
   (MP-Q27). Installing it is MP-Q32.

### 9.5 Checks the formula must pass

- **Öpik limit.** In a homogeneous Euclidean volume, with σ_s → 0, a pure G limit and no floor,
  average W over the parent rows at fixed M1. The parent's companion density is then
  π(c) (1 + f)^{3/2} / ⟨(1 + f)^{3/2}⟩_π, the classical result: a system with flux ratio f is seen to
  (1 + f)^{1/2} times the distance.
- **No information, no correction.** As σ_s → ∞, W → 1 and the naive mock is recovered, correctly,
  since then G says nothing about c and the cut does not prefer binaries.
- **Volume-limited inversion (diagnostic).** n_vol(M1, c) ∝ Σ_i w_i / V_S(M̂1_i, c_i), with
  V_S(M1, c) = ∫ dΩ ∫ dr r² ρ_*(r, Ω) ∫ dε N(ε; 0, σ_int) P(S | G(M1, ε, c, r), ϖ(r), Ω). Singles
  enter with p(∅ | o_s) / V_S(M̂1_s, ∅). This needs ρ_*, the completeness and A_G. It is used only to
  compare with MdS17 directly; the pipeline's normalization stays relative to the parent.

### 9.6 Implementation and the closed-loop proof

- `src/darkhunter_pop/malmquist.py` (pop side, numpy): ΔM_s, σ_s, the λ_f(log10 f | M1) grid and
  Z_s for the MdS17 luminous target of `proposal_set.mds17_luminous_log_intensity`, log W per draw,
  p(∅ | o_s), and V_S for the volume-limited diagnostic. A test checks that the grid integral
  matches the target integrated directly, so the two cannot drift. `proposal_set.py` gets a small
  documented hook that adds log W to the target. The rest of its interface is unchanged.
- **Closed loop** (no gaiamock, numpy only): a synthetic exponential-disk universe with a Kroupa
  primary IMF, Janssens M_G(M) with scatter σ_int, and MdS17 companions drawn from the same table and
  provisional settings as the target. It is observed with G_total < 19, ϖ_obs > 0.2 mas and an
  optional synthetic completeness; distances come from a geometric posterior with the true density
  as prior, summarized as r_lo / r_med / r_hi exactly like Bailer-Jones. The synthetic parent then
  goes through `proposal_set.sample_proposal` (as if `gaia_source` rows) and the weights. It must
  recover (i) the parent's true companion statistics (binary fraction overall and versus G and M1;
  log P, q, e, log f) and, after §9.5's 1/V_S, the injected volume-limited MdS17; and (ii) the
  parent's distribution of observed system properties (the photocentre semi-major axis α0 and the
  count in an NSS-like (α0, P) window). It is run **with and without** W: without W it must fail
  visibly, or the test has no power. Results and figures are in `docs/gate405/`, with a small
  version in the required gate and a large one marked `slow`.

### 9.7 Closed-loop result (docs/gate405)

Large run: 336,543 synthetic parent rows. Without W the mock under-predicts parent twins by 12%
(−15σ) and misses the volume-limited binary fraction by up to 34σ. With W the twin deficit is 3%
and the total is within 0.65%. Handed the true distance and M1, W closes to MC noise (every
|pull| ≤ 2.2 at 10⁶ primaries). The remaining percent-level residual comes from approximations
A1 and A2. The M_G zero point is the sharpest open input: a −0.05 mag error undoes the correction
(MP-Q25). A split normal with its mode at r_med (MP-Q30 b) measured worse than the Gaussian. The
α0 / NSS-window counts barely move, because the Öpik boost lives in near-twins, which have small
photocentre orbits.

## 10. Giants and other evolved primaries (MP-Q28, #413)

Ryan's decision (2026-10-03, #391): **match the real giant population.** Identify giants by the
CMD, use a luminosity relation that fits giants, truncate their companions, and check the giant
fraction and the six-panel against DR3. This section measures where the mock stands today and
splits the treatment into what the data and the papers fix (implemented or specified here) and
what they do not (options MP-Q28a–g, §10.8, for Ryan). Measurements: `docs/gate_giants/`
(script `scripts/giant_population_diagnostics.py`). The mock numbers come from the paused #391
generation 10 + 11 artifacts, which are pre-noise-fix and pre-Malmquist, so they are diagnostic only.

### 10.1 Why the current treatment fails

- **The flag.** `is_giant` (log g of the atmosphere TAG10 used < 3.6, §0.1) marks 519 of 164,397
  usable parent rows (0.32%). MSC fits every source as two main-sequence stars, so `logg_msc1` is
  dwarf-like by construction. For real orbits that the CMD calls evolved, the median log g of the
  atmosphere TAG10 used is **4.51**, and MSC supplies all of them.
- **M1.** TAG10 (Torres et al. 2010) applied to that dwarf-like atmosphere gives, for the same
  real evolved orbits, a median TAG10 / FLAME mass ratio of **0.52** (16–84%: 0.26–0.74). For real
  dwarfs the ratio is 0.92 (0.76–1.03). FLAME giant masses are themselves coarse: there is a floor
  near 0.9 M⊙ and gridded stripes (`giants_m1_flame.png`). Even so, TAG10 underestimates giant masses
  by roughly a factor of two.
- **Companion light.** The target's f uses the Janssens et al. (2022) **dwarf** M_G for both stars.
  For the parent's evolved rows, the dwarf relation at the TAG10 M1 is fainter than the observed
  M_G0 by a median of **5.4 mag** (10–90%: 2.9–7.6). Under MdS17 weights, the companions drawn on
  evolved rows have median log10 f = −1.22 under the dwarf relation and −3.43 from the observed
  light (§10.4), about 2 dex too bright. Overestimating f shrinks the photocentre orbit by the
  factor (q/(1+q) − f/(1+f)).
- **Orbits.** MdS17 describes main-sequence pairs. An evolved primary has expanded, so close
  companions have been engulfed, have transferred mass or have been circularized (Verbunt & Phinney
  1995; Badenes et al. 2018; Price-Whelan & Goodman 2018). The mock does not truncate.
- **Malmquist.** §9's L_s uses M_G^J(M̂1). For a giant ΔM_dwarf ≈ −5 mag, so computing W with the
  dwarf relation would assign almost every giant a companion. `provisional_giant_policy:
  unit_weight` is what prevents that, but it reads the broken flag.

### 10.2 Identification (implemented: `darkhunter_pop.giants`, `config/population/giants.yaml`)

Dereddened CMD, applied identically to the parent and the real orbits:

  M_G0 = G − 5 log10(d / 10 pc) − A_G,   C0 = (BP − RP) − E(BP − RP)

E(B−V) is from Combined19 at d (MP-Q29). It is converted with the Gaia law (Babusiaux et al.
2018) and the `r_v` and coefficients already in `sample_selection.dust_maps` (#295). The parent
uses the Bailer-Jones geometric d (MP-Q4).

The **main-sequence ridge R(C0)** is measured from the parent itself. It is the smoothed mode of
M_G0 in 0.1 mag colour bins, using rows with ϖ/σ_ϖ ≥ 10. Binaries only brighten a star, so the
faint side of each bin holds singles plus noise; its clipped RMS is the ridge width σ_R(C0). This
is 0.24–0.31 mag over 0.85 < C0 < 1.8 and up to 0.6 mag at the blue and red ends. The table is in
`giants_report.txt`. A row is **evolved** (subgiant or giant) when

  ΔM = M_G0 − R(C0) < −2.5 log10 2 − n_σ σ_tot,   σ_tot² = σ_R² + σ_μ²

i.e. brighter than any main-sequence star with an equal-light companion. The 0.753 mag is
arithmetic (`constants.TWIN_BRIGHTENING_MAG`); σ_R is measured; n_σ is **provisional** (3; MP-Q28a).
Rows outside the ridge's colour range (0.35 ≤ C0 ≤ 2.75) or without a CMD are unclassified and
counted (8,626 usable parent rows, 5.2%: 7,368 bluer than C0 = 0.35, 830 redder than 2.75, 428 where the
extinction law did not converge). GSP-Phot log g is
reported only as a cross-check: it drops toward the dwarf prior at low parallax S/N (0.9% with
log g < 3.6 at ϖ/σ_ϖ < 5, against 4.2% CMD-evolved), so it is not used to classify.

**Hook.** `dataclasses.replace(parent, is_giant=giants.classify_parent(parent, cfg, gcfg)[0].evolved)`
swaps the flag that `malmquist.row_conditioning` and `log_weight_for_draws` read. Neither
`proposal_set.py` nor `malmquist.py` is edited (their owner is wiring §9). The replacement is the
owner's call, and it is reported in the PR.

### 10.3 Measured fractions (diagnostic, `docs/gate_giants/giants_report.txt`)

| Sample | Evolved (CMD, n_σ = 3) | n_σ = 2 / 4 / 5 |
|---|---|---|
| Parent, usable and classified (155,771) | **8.12% ± 0.07%** | 10.2 / 6.7 / 5.7% |
| Parent, ϖ/σ_ϖ ≥ 20 (23,065) | 14.2% ± 0.2% | |
| Real Orbital + AstroSpectroSB1, classified (158,921) | **14.37% ± 0.09%** | 16.5 / 13.3 / 12.4% |
| Real Orbital only (126,197) | 9.98% ± 0.08% | |
| Real AstroSpectroSB1 only (32,724) | 31.3% ± 0.3% | |
| Mock accepted orbits, raw draws (3,214) | 2.7% | |
| Mock accepted orbits, MdS17-weighted | **9.9% ± 1.4%** (ESS 31.6 for the evolved subset) | |
| Old flag: parent / mock accepted | 0.32% / 1.07% | |

Caveats:

- **Real distances.** The real side uses the inverse NSS parallax, with 16/84% from ϖ ± σ_ϖ. The
  Bailer-Jones join (`scripts/fetch_nss_bailer_jones.py`) timed out twice on the Gaia archive (HTTP
  500 statement timeout, the archive-side problem of #184). The real orbits are high-S/N (ϖ/σ_ϖ
  5th / 50th percentile = 29 / 81), and there 1/ϖ and the geometric distance agree to well under
  σ_R. Re-run with `--real-bj-dir` once the archive answers.
- **The mock's cascade produces Orbital solutions only.** The like-for-like real number is
  therefore the Orbital 10.0%, which the weighted mock's 9.9% ± 1.4% matches. The 31% giant share of
  AstroSpectroSB1 comes from the RV chain's G_RVS limit, which favours bright, i.e. luminous,
  stars. That is MP-Q21, not a giant-physics question.
- **The parent fraction depends on S/N.** Low-S/N rows have a wide σ_μ, so they are classified
  conservatively.

The weighted fraction agreeing with real Orbital does **not** mean the giants are right. The
six-panel (`giants_six_panel.png`) shows that the mock's evolved orbits sit at shorter P (median
517 d against 701 d in DR3), have smaller f_m (median 0.039 against 0.072 M⊙, with the mock tail
cut at ≈ 0.06), and are more eccentric (median e 0.49 against 0.32). The weighted KS gives p < 10⁻⁶
for P and f_m and p ≈ 0.007 for e, at n_eff = 32. The real evolved and dwarf e distributions are
almost identical, so the excess eccentricity is the generic #409/#410 issue and not a giant effect.
The small f_m is what a factor-of-two-low M1 plus a 2-dex-too-bright f produces (§10.1). The P
offset is consistent with missing truncation (§10.5), but the ESS is too low to say more.

### 10.4 Companion light ratio (determined; function implemented, wiring is the target owner's)

The companion of an evolved primary is less massive, so it is still on the main sequence, and the
Janssens relation (MP-Q13, decided) applies to **it**. MP-Q6 (decided) makes the row's observed G
the total system light. So the companion's share of the dereddened system light is
x = 10^{−0.4 (M_G^J(M2) − M_G0,sys)}, and

  f = x / (1 − x)   (`giants.evolved_log10_flux_ratio`)

with the decided 0.1 dex scatter around it and λ = 0 where x ≥ 1. No M1 → M_G relation for the
giant is needed. The exception is a near-twin (q ≳ 0.95 relative to the progenitor), whose
companion is itself evolving; that is a limitation of the dwarf relation for M2, recorded and not
fixed. This changes the **target only**, so it is reweightable (§3.5). The proposal's q(f) puts
90% of its mass around the dwarf relation (`proposal_set_decided_full.yaml`: `relation_weight`
0.9, σ 0.15 dex) and only a 10% log-uniform component on [−5, 0.5]. The evolved-row targets
(median log10 f = −3.4, 10–90%: −4.7 to −2.0) are therefore supported but inefficient, and 5.6%
(MdS17-weighted) fall below f_min = 10⁻⁵. Whether to centre the restart proposal on this relation for
evolved rows, and whether to lower f_min, is MP-Q28d (efficiency and support, not correctness).

### 10.5 Companion statistics for evolved primaries

- **Frequency, q, P and e at the progenitor mass.** MdS17 describes the zero-age pair. A low-mass
  red giant has lost little mass on the RGB. The asteroseismic estimate for NGC 6791 is
  ≈ 0.1 M⊙ (Miglio et al. 2012), so M1,ZAMS ≈ M1 for RGB stars and ≈ M1 + 0.1 M⊙ for clump stars.
  The q and period laws are evaluated at that mass. This is a small correction next to the
  factor-of-two M1 error (§10.1).
- **Engulfment / Roche-lobe overflow.** A companion inside the Roche lobe that the primary has
  filled **at any time** has interacted: common envelope, merger or stable transfer. Either way it
  is not an MdS17 pair now. The Roche-lobe radius is Eggleton (1983):
  r_L/a = 0.49 q^{2/3} / (0.6 q^{2/3} + ln(1 + q^{1/3})), q = M1/M2. Implemented as
  `giants.roche_period_floor_days`, at the CMD radius R1 from the Andrae et al. (2018) G-band
  bolometric correction and GSP-Phot Teff. Measured (`giants_period_radius.png`): for the current
  R1 the floor lies **below** the NSS period window for every R1 ≲ 30 R⊙. 0 of 49 accepted mock
  evolved draws violate it. So truncation at the *current* radius changes almost nothing that NSS
  sees. What does change things is the radius the star *has had*. A red-clump star has passed the
  RGB tip (R ~ 10² R⊙), so its floor is hundreds to thousands of days.
- **Tidal circularization and orbital decay** act before contact. The equilibrium-tide timescale
  scales as (R/a)⁻⁸ (Zahn 1977; Verbunt & Phinney 1995, who calibrate it on giants in clusters).
- **The real data show the effect.** The 5th percentile of P for real evolved orbits rises with R1:
  259 d at 3–6 R⊙, 314 d at 6–12 R⊙, 460 d at 12–30 R⊙ and 520 d above 30 R⊙. For real dwarfs it is
  226 d. The weighted mock evolved draws have P5 = 127–245 d in the same bins. APOGEE shows the same
  trend: the close-binary fraction falls and P_min grows as log g drops along the RGB (Badenes et
  al. 2018; Price-Whelan & Goodman 2018). Part of the real trend is selection (larger, more luminous
  giants are more distant, so a detectable photocentre orbit needs a longer P), and the mock
  reproduces that part automatically. The **form** of the truncation is MP-Q28b.
- **This is a target change** (λ = 0 or a survival factor below a floor), so it is reweightable. It
  needs R1 (and, for MP-Q28b's options, the past maximum radius) stored per parent row, which the
  CMD gives without regeneration.

### 10.6 Malmquist weight for giants (determined: W = 1)

§9.3's W_s(c) = L_s(f)/Z_s depends on c only through δ = 2.5 log10(1 + f_b). For an evolved
primary, ΔM relative to any single-star relation is dominated by the evolutionary state at fixed
M1. Its intrinsic spread σ_s is ~1 mag along the RGB, and §9.5's "no information" limit
(σ_s ≫ δ ⇒ W → 1) applies. Independently, δ is tiny. With the §10.4 flux ratio, the companions
drawn on evolved rows have weighted median, 90th and 99th percentile δ of **0.0004, 0.010 and
0.115 mag**. Since |ln W| ≲ δ |ΔM| / σ_s², W = 1 holds to ≲ 1% for 90% of draws. The 1%-tail
is subgiants with near-equal-mass companions, where it holds to ~10%. So **`unit_weight` is the
correct policy** for evolved rows, provided it reads the CMD flag (§10.2), **not** the TAG10-log g
flag. Computing W with the dwarf M_G^J for a giant (ΔM ≈ −5 mag) would be catastrophic:
ln W ≈ −ΔM δ / σ² ≈ +14 at σ = 0.15 mag for the median dwarf-relation companion (δ ≈ 0.065 mag).

### 10.7 Giant RUWE and epoch noise (measured: nothing needed)

The parent rows' own RUWE at fixed G gives these medians for evolved / dwarf rows:

| G (mag) | Evolved median RUWE | Dwarf median RUWE |
|---|---|---|
| 8–10 | 0.991 | 1.004 |
| 10–12 | 1.035 | 1.053 |
| 12–14 | 1.015 | 1.014 |
| 14–16 | 1.008 | 1.010 |
| 16–18 | 1.003 | 1.008 |

At no G is the evolved median higher. The RUWE > 1.4 fraction is *lower* for evolved rows (for
example 4.1% against 13.6% at G = 12–14), as expected for more distant systems with fewer close
luminous companions. So there is no sign of extra single-star astrometric jitter from giant
convection (predicted to matter only for supergiant / AGB-sized photospheres; Chiavassa et al.
2011) at the resolution of RUWE. **No giant-specific noise term is proposed.** The #398/#400 epoch
model depends on G and sky position only, and that is consistent with this.

### 10.8 Options for Ryan (MP-Q28a–g; none chosen)

- **MP-Q28a, classifier margin n_σ** (§10.2): 3 (provisional), 2, 4 or 5. The real Orbital +
  AstroSpectroSB1 evolved fraction moves 16.5 / 14.4 / 13.3 / 12.4%. Alternatively, use a
  probabilistic membership P(evolved) from σ_tot instead of a hard flag.
- **MP-Q28b, truncation form** (§10.5): (i) remove companions with P below the Roche floor at the
  **current** CMD radius (measured to change almost nothing); (ii) at the **maximum past** radius,
  which needs RGB vs clump membership (CMD clump box, or GSP-Phot / asteroseismic labels where they
  exist) and the RGB-tip radius from isochrones; (iii) an empirical survival factor in (P, log g) or
  (P, R1) calibrated on APOGEE (Price-Whelan & Goodman 2018; Badenes et al. 2018); (iv) a free
  P_min(R1) at rung 3, fit to the NSS giants themselves. For each: circularize (e = 0) below a
  Verbunt & Phinney (1995) tidal floor, or not.
- **MP-Q28c, M1 for evolved primaries** (§10.1). This is **generation-time** (§3.5), so it must be
  decided before the #391 restart. Options: (i) keep TAG10 on both sides (consistent with the data
  side, but the truth is wrong by ≈ ×2); (ii) Gaia FLAME `mass_flame` (Creevey et al. 2023) on both
  sides (available for 55% of real evolved orbits; coarse and floored near 0.9 M⊙ for giants; the
  parent needs a re-query of the same K = 10⁶ slice); (iii) an isochrone fit (MIST / PARSEC) to
  (M_G0, C0, [M/H]), a new dependency; (iv) asteroseismic scaling relations
  M ∝ (ν_max/ν_max,⊙)³ (Δν/Δν_⊙)⁻⁴ (T_eff/T_eff,⊙)^{3/2} (Kjeldsen & Bedding 1995). These exist only
  for Kepler / K2 / TESS fields, so they can calibrate (ii)/(iii) on an overlap sample (e.g.
  APOKASC-3; Pinsonneault et al. 2025) but cannot be a population-wide M1. Whatever is chosen must
  also replace TAG10 on the **data side** for evolved real orbits (MP-Q5 symmetry), or the inferred
  M2 of real giants stays biased by the same factor.
- **MP-Q28d, proposal support for evolved rows** (§10.4): at the restart, add a proposal component
  centred on the §10.4 relation for evolved rows and lower `log_f_min` from −5 to about −6 (5.6% of
  the evolved-row target lies below 10⁻⁵ now). Or leave the proposal as is and accept low ESS for
  giants.
- **MP-Q28e, MdS17 mass for evolved primaries** (§10.5): the current M1, or M1 + ΔM_RGB (≈ 0.1 M⊙,
  Miglio et al. 2012) for clump stars. It only matters once MP-Q28c fixes the factor-of-two error.
- **MP-Q28f, AstroSpectroSB1's giant excess**: compare the mock with Orbital only for giants, or
  model the RV-chain selection (MP-Q21).
- **MP-Q28g, real-side distances**: keep the inverse NSS parallax (high S/N), or block on a
  Bailer-Jones re-fetch once the archive answers (§10.3).

## 11. Isochrone M1 and the 2-D CMD Malmquist weight (#418)

Ryan's decisions are in §0.3. This section specifies what the decisions and the cited papers
determine, and lists the choices they do not determine as options MP-Q33–Q38 (§11.8). None of
those has been picked; each is a `provisional_*` config field. Measurements go in `docs/gate418/`.

### 11.1 Why TAG10 goes

TAG10 (Torres et al. 2010) maps an atmosphere (T_eff, log g, [Fe/H]) to a mass. For the parent and
the real orbits that atmosphere is MSC's two-dwarf fit or GSP-Phot, and three problems follow.
With the Santos et al. (2013) correction, M1 floors at 0.597 M⊙ (#393). MSC log g is dwarf-like by
construction, so giants come out ≈ 2× too light (§10.1). And the M_G(M̂1) residual that the 1-D
Malmquist weight needs runs over 3 mag with distance and M̂1 (#414). The CMD position itself
carries the mass information that is needed, for dwarfs, subgiants and giants alike, once age and
[Fe/H] are marginalized.

### 11.2 Isochrone M1

**Grid.** MIST v1.2 with v/v_crit = 0.4 (Choi et al. 2016; Dotter 2016), full isochrones,
[Fe/H] = −2.0 … +0.5 in 0.25 dex steps (11 files) and log10(age / yr) = 8.0 … 10.15 (44
isochrones). MIST phases 0 (MS), 2 (SGB + RGB), 3 (CHeB), 4 (EAGB) and 5 (TPAGB) are kept. PMS
(−1) and post-AGB / WD (6, 9) are dropped. G, BP and RP come from the MIST UBVRIplus
bolometric-correction tables at A_V = 0 (columns `Gaia_{G,BP,RP}_EDR3`; DR3 and EDR3
photometry are the same), with M_X = 4.74 − 2.5 log10 L − BC_X(T_eff, log g, [Fe/H]). The 4.74 is
the tables' convention (`constants.MIST_MBOL_SUN`). Points are sub-stepped by linear
interpolation at fixed EEP, the purpose MIST's EEPs are built for, to 0.05 dex in [Fe/H] and
0.0125 dex in age. That gives ≈ 8.7 × 10⁶ points.

The files are read in place from `isochrone_mass.mist_root`, a host-specific path set in
`config/host_profiles/` (laptop: `/Users/rfoley/.isochrones`; 21 GB, never copied). The parsed
native grid is cached under `<data_root>/isochrone_mist/` with a SHA256 that is checked on load.

**Inputs.** The dereddened CMD point y = (C0, M_G0) of §10.2, computed identically on both
sides: Combined19 E(B−V) at the Bailer-Jones geometric distance (MP-Q29, MP-Q4), the Babusiaux
et al. (2018) Gaia law, and σ_μ from the Bailer-Jones quantiles. The real orbits use the inverse
NSS parallax until the Bailer-Jones join answers (MP-Q28g). [Fe/H] measurements are **not** used
by default. GSP-Phot [M/H] has large systematics (Andrae et al. 2023), and MSC [M/H] comes from a
two-star fit. Whether to add a calibrated metallicity likelihood is MP-Q33.

**Likelihood.** A Gaussian that is diagonal in (C0, M_G0):

  σ_C² = σ_C,floor² + (ε E(B−V) k_E)²,   σ_M² = σ_μ² + σ_M,floor² + (ε E(B−V) k_A)²

Here k_A = A_G / E(B−V) and k_E = E(BP−RP) / E(B−V) are the per-star values of the Babusiaux law.
The floors stand in for photometric calibration, bolometric-correction and isochrone
systematics. ε is the fractional E(B−V) error. Defaults: σ_C,floor = 0.02, σ_M,floor = 0.05 mag
and ε = 0.1. All three are provisional (MP-Q34). ε = 0 (MP-Q29's σ_A = 0) left 11,306 parent rows
off the grid, mostly with high E(B−V). ε = 0.1 leaves 7,827 and moves the other rows' M1 by a median
0.0009 dex (ε = 0.2: 5,024).
Projecting the extinction vector onto each axis separately drops its (C0, M_G0) correlation; that
approximation is recorded, and it vanishes at ε = 0.

**Priors** (all config):

- IMF: Kroupa (2001) broken power law in initial mass.
- Age: a constant star-formation rate, i.e. uniform in linear age, over the grid's
  0.1–14.1 Gyr.
- [Fe/H]: a Gaussian N(−0.1, 0.25 dex), provisional (MP-Q33).

The weight of isochrone point j is

  w_j = ξ(M_init,j) ΔM_init,j × p(τ_j) Δτ_j × p([Fe/H]_j) Δ[Fe/H]_j

with ΔM_init the half-distance to the neighbouring EEPs along the isochrone. That is the standard
population weighting of an isochrone grid.

**Posterior and outputs.** p(j | y) ∝ w_j N(y; y_j, Σ). The moments come out per star:

- the current mass, as its mean and σ and as ⟨log10 M1⟩ and σ_log M1;
- the initial (progenitor) mass;
- ⟨log10 R⟩ and the **maximum past radius** R_max. R_max is the larger of the running maximum of R
  over lower EEPs on the same isochrone, which catches the RGB tip for clump stars, and of R at the
  same initial mass on earlier isochrones;
- log g, ⟨log age⟩ and ⟨[Fe/H]⟩;
- P(evolved), the posterior probability of MIST phase ≥ 2;
- the prior-predictive density at y.

Rows whose density is below 10⁻⁴ mag⁻² get no M1, and the reason `off_grid` is counted. These are
white dwarfs, hot subdwarfs and bad photometry. Rows with no CMD are counted as `no_cmd`.

The M1 that feeds point uses (the mock's truth M1 and the bulk M1) is the posterior mean. The
alternatives are 10^⟨log10 M1⟩, or a draw from the posterior per proposal draw (generation-time);
that choice is MP-Q35.

**Computation.** The weighted points are deposited once, cloud-in-cell, into a (C0, M_G0) map at
0.01 × 0.02 mag with one channel per moment. Each star's moments are then k_C^T Map_q k_M, with
k_C and k_M the cell-integrated 1-D Gaussian kernels. This is one matrix product per chunk of
colour-sorted stars, about 2 × 10⁵ stars per minute on one thread. The module is
`darkhunter_pop.isochrone_mass`.

**Giants.** Dwarfs, subgiants and giants go through one posterior. For giants it gives the
progenitor mass (MP-Q28e) and the current and maximum past radius. Those are covariates for the
rung-3 free period floor (MP-Q28b decided), not a hard truncation. The CMD evolved flag stays the
§10.2 classifier (n_σ = 3, MP-Q28a decided), and P(evolved) is reported beside it as a
cross-check.

### 11.3 Blended light: how M1 and the weight stay consistent

The isochrone fit assumes a single star. An unresolved luminous companion moves the system point
by Δ(c) = (ΔC0, ΔM_G0): brighter, and redder for M2 < M1. The fit then returns an M̂1 that is biased
(high for near-twins; the size is measured in `docs/gate418/`). Two rules keep the mock and the
data consistent:

1. **(A1′) The same estimator on both sides.** The data side computes M2 from a0 with M̂1. The mock
   assigns truth M1 = M̂1 to the row (§0.1 MP-Q5 symmetry, with the isochrone in place of TAG10).
   So the M̂1 → M2 mapping, including the blended-light bias, is the same on both sides. That is
   the property the forward model needs.
2. **The weight never uses M̂1 to judge the companion's light.** W (§11.4) subtracts the drawn
   companion's G, BP and RP flux from the row's observed (dereddened) photometry, which is the
   system total under MP-Q6. It then asks whether the remaining primary sits on the single-star
   ridge. M̂1 enters W only through π(c | M̂1), meaning MdS17's frequencies, q = M2 / M̂1 and the
   decided Janssens f relation. The light accounting is exact for the drawn c. Under the 1-D weight
   of §9.3 the bias of M̂1 went straight into ΔM. Here it does not enter at all.

The remaining inconsistency is that, for a binary, the truth M1 the mock hands gaiamock is the
blended-light M̂1 rather than the primary's own mass. Its effect on a0 ∝ (M1 + M2)^{1/3} is a few
percent for twins (measured). The exact alternative re-fits M1 to the subtracted primary point per
draw, which makes truth M1 depend on c. That is generation-time, and it is MP-Q36.

### 11.4 The 2-D weight

The notation is §9.1's, with the row's dereddened system point y_s = (C_s, M_s). For a luminous
companion c with G-band flux ratio f = F2/F1 and mass M2:

- The companion's share of the system G flux is x_G = f / (1 + f).
- The companion's intrinsic (BP − G)_2 and (G − RP)_2 come from the MIST main sequence at M2. The
  system's (BP − G)_s and (G − RP)_s come from the MIST single-star colour–colour relation at C_s.
  The (BP − G) vs (BP − RP) relation of MS + MS blends stays on the single-star relation to the
  precision this needs; that is measured, not assumed.
- The companion's share in each band is x_BP = x_G 10^{−0.4 [(BP − G)_2 − (BP − G)_s]} and
  x_RP = x_G 10^{+0.4 [(G − RP)_2 − (G − RP)_s]}.
- The primary is what remains:

  C_1 = C_s − 2.5 log10(1 − x_BP) + 2.5 log10(1 − x_RP),   M_1 = M_s + 2.5 log10(1 + f)

  If x_BP ≥ 1 or x_RP ≥ 1, the drawn companion would outshine the system in that band, and
  L = 0.
- The displacement vector is Δ(c) = y_s − (C_1, M_1). It is the MIST companion displacement Ryan
  specified, evaluated backwards from the observed system.

Then

  **L_s(c) = N(M_1 − R(C_1); 0, σ_s(C_1)),   L_s(∅) = N(M_s − R(C_s); 0, σ_s(C_s))**

  σ_s(C)² = σ_R(C)² + σ_μ,s² + σ_A,s² (1 − R′(C) / k_s)²

  **Z_s = (1 − F_lum(M̂1)) L_s(∅) + Σ_{log q, log f bins} λ_qf(M̂1) L_s(q, f),   W_s(c) = L_s(c) / Z_s**

- **R(C), σ_R(C)** are the measured single-star ridge and its faint-side width (§10.2). There is
  **one ridge**, shared by the evolved classifier and W. It is re-measured on usable parent rows
  with **RUWE < 1.4** and ϖ/σ_ϖ ≥ 10 (config `giants.ridge.ruwe_max`).
- **Extinction is its own vector.** An E(B−V) error moves y along (E(BP−RP), A_G), and its
  projection onto the ridge residual is σ_A |1 − R′ / k|, with k = A_G / E(BP−RP). The reddening
  vector is nearly parallel to the MS over much of it, so this is small. σ_A = 0 is the decided
  default (MP-Q29).
- **λ_qf(M1)** is the MdS17 luminous target (`proposal_set.mds17_luminous_log_intensity`)
  marginalized over P and e onto (log q, log f) bins. f is spread about the decided Janssens
  relation with σ_f = 0.1 dex (MP-Q13). A test checks the grid integral against the target
  integrated directly. Dark companions carry L_s(∅) as before.
- **Limits.** If the companion has the primary's colour, C_1 = C_s and L_s(c) reduces to §9.3's
  N(ΔM + 2.5 log10(1 + f); 0, σ), with ΔM = M_s − R(C_s) replacing M_s − M_G^J(M̂1) − δ_zp. **No
  M1 → M_G relation enters W at all.** That removes the source of the #414 systematics. As σ_s → ∞,
  W → 1. The Öpik limit of §9.5 holds unchanged, since it depends only on how the light adds up.
- **Rows where W = 1** (the naive draw), each counted:
  - CMD-evolved rows (§10.6);
  - rows outside the ridge's colour range;
  - rows without a CMD.

  A subtracted primary colour C_1 outside the ridge range uses the ridge end values, and those
  draws are counted. Extending the ridge with MIST offsets is MP-Q38.
- **Companion colours.** These use the MIST MS at a fiducial [Fe/H] and age, by default the
  [Fe/H] prior mean and log age 9.6. Using the row's own posterior (age, [Fe/H]) is MP-Q37. The
  closed loop gives the universe's companions the primary's true age and [Fe/H], so it measures the
  fiducial's cost.

**Single-star density (MP-Q39, found in the closed loop, `docs/gate418/`).** The Gaussian
N(M − R(C); 0, σ_R(C)) treats single stars as symmetric about the ridge, with σ_R taken from the
faint side. Real single stars at fixed colour are not symmetric. Turnoff stars, subgiants and the
[Fe/H] spread give them a bright-side tail. In the MIST closed loop the faint-side RMS is
0.46–0.66 mag and the bright-side RMS is 1.0–1.3 mag. The ridge form then leaves the brightest
companions (twins, log f > −0.5) about as under-predicted as no weight at all. The option
`mist_density_ridge_anchored` uses the full prior-predictive single-star CMD density
φ(C, M) of `isochrone_mass` (every kept phase) in place of the Gaussian. It is shifted in M per
colour so that its mode sits on the measured ridge, which is still the one ridge shared with the
§10.2 classifier. It is convolved with the row's σ_M, and it carries the colour Jacobian of the
light subtraction:

  L_s(c) = φ(C_1, M_1 | σ_M) / |J|,   J = ∂C_s/∂C_1 = (1 − x_BP) h′(C_1) − (1 − x_RP)(h′(C_1) − 1)

with h(C) = (BP − G)(C) on the MIST single-star relation. J = 1 without a companion, and
J ≈ 1 − x for a companion with the primary's colour. The colour density of single stars, which
the ridge form drops, then enters too. The extinction-error projection σ_A enters only the
Gaussian form, and MP-Q29 set σ_A = 0. Which form to use is MP-Q39 (§11.8).

Code: `darkhunter_pop.malmquist_cmd` (new), next to `malmquist` (the 1-D weight, kept for the
#405 tests and comparisons). The hook in `proposal_set` mirrors `malmquist_log_weight`.

### 11.5 Closed loop with CMD photometry (extends §9.6)

The synthetic universe of §9.6 is extended so that every star has MIST photometry:

- primaries are drawn from the prior-weighted MIST points (MS, the closed-loop IMF range);
- companions are MdS17 from the same table as the target, with G light from the target's f model;
- companions get BP and RP from MIST at M2, at the primary's own age and [Fe/H];
- there is a synthetic extinction vector, observed with a configurable fractional error, and a
  colour error.

The pipeline under test runs:

1. The dereddened CMD.
2. Isochrone M̂1, the real estimator.
3. The ridge, measured from the synthetic parent the same way as on real data. There is no RUWE in
   the universe, so the ridge uses all rows; that is recorded.
4. The weight.

It is run with no W, the 1-D W of §9.3 (now fed the isochrone M̂1) and the 2-D W. Pulls are as in
§9.6. Acceptance mirrors #405: in the small run (required gate) every 2-D-W parent pull is ≤ 3
and the no-W run fails visibly (some |pull| > 5). The large run and the figures go in
`docs/gate418/`.

**Result (small run: 10⁶ primaries → 40,151 usable parent rows; `docs/gate418/closed_loop_cmd_small_*.json`).**
The universe has MS, SGB/RGB and core-He-burning primaries. Evolved companions use the §10.4
flux relation, and the synthetic E(B−V) is known to 5%. The table gives max |pull| over the parent
statistics.

| weight | total binaries (truth 20,702) | twins (q > 0.95) | brightest log f bin | max \|pull\| |
|---|---|---|---|---|
| none | 20,024 (pull −2.3; CMD-dwarf rows −7.7) | −2.6 (dwarf rows −9.7) | −10.8 | 10.8 |
| 1-D (§9.3, isochrone M̂1) | 25,028 (+12.9) | +14.2 | +36.1 | 36.1 |
| 2-D, Gaussian ridge | 20,832 (+0.4) | −2.1 (dwarf rows −6.2) | −2.3 | 5.7 |
| 2-D, MIST density anchored | 20,528 (−0.6) | +0.6 (dwarf rows +1.6) | +2.0 | 7.4 |

The 1-D weight cannot be used with a CMD-derived M̂1. Its ΔM compares the CMD with the Janssens
relation at a mass inferred from the same CMD. Both 2-D forms remove most of the naive bias, and
the MIST density closes the twins. Neither form meets the ≤ 3 acceptance yet. Single bins remain
off at 4–7σ: log f in [−1, −0.5], q in 0.5–0.85 and the α0 tail. The suspected causes are the
fiducial companion colours (MP-Q37), M̂1 = blended-light mass (§11.3), and the pipeline's
σ_A = 0 against a 5% truth error. None is retuned.

### 11.6 What replaces the MP-Q25 fit

MP-Q25 fitted a zero point and σ_int for M_G^J(M̂1). W no longer uses that relation, so the fit
is retired. Its values in `config/population/malmquist_decided.yaml` are marked superseded, and
`fit_mg_zero_point` stays for the 1-D comparison only. The single-star calibration is now the
ridge R(C0) and σ_R(C0) on RUWE < 1.4 rows. The #414 acceptance tables are re-measured for
ΔM_2D = M_G0 − R(C0) on the same rows, binned by distance and by isochrone M̂1, and they must show
no multi-magnitude trend. The ridge's statistical precision is reported against the closed loop's
0.05 mag tolerance (§9.7).

### 11.7 Data side: `mass_derivation_bulk`

A pipeline design change, docs-first in `docs/ARCHITECTURE.md` (`mass_derivation_bulk`):

- **The switch.** A new value `mass_calibration.method: MIST_isochrone`. The default stays `TAG10`
  until Ryan flips it, after the re-measurements below. Under the switch, each candidate's M1
  ParameterSet holds the posterior M1 point (MP-Q35) and σ_M1, and R1 = 10^⟨log10 R⟩, with
  provenance `isochrone_mist_v1.2`. A row with no CMD or off the grid is skipped with its own
  counted reason, as TAG10's `no_atmosphere` / `m1_failed` are.
- **Inputs per candidate.** G, BP−RP and (l, b) from the candidate. The distance is the
  Bailer-Jones geometric one when it is present, otherwise 1 / ϖ_NSS (MP-Q28g). Combined19 E(B−V)
  and the Babusiaux law are computed exactly as for the parent (§11.2).
- **The TAG10 code is kept unchanged.** It remains the default and is still what the pre-switch
  run files used.
- **Literature reproduction paths keep their own M1** (column ownership): Andrews 2022
  Lick → FLAME → uniform; El-Badry 2024 IsocLum; El-Badry 2026 Janssens M̃1. Their reproduction
  counts must not change, and that is checked. The **forward-model** paths that read the pipeline's
  bulk M1, such as Andrews' forward_model pass 2, change with the switch, and their counts are
  reported.
- **#393** (Santos floor): does not apply under the switch. It stays open for the TAG10 path,
  which is still the default.
- **#380** (non-positive Gaussian M1 draws in the bulk σ_M2 MC): the isochrone σ_M1 / M1 is
  usually ≪ 1. The fraction above 0.25 is reported under the switch. The draw convention (Gaussian
  vs log-normal from ⟨log M1⟩, σ_log M1) is still #380's decision.
- **#323** (TAG10 outside its calibration range): the isochrone covers the CMD from the lower MS
  to the RGB / AGB. Out of range becomes `off_grid`, a counted reason, with no extrapolation. This
  is the "isochrone masses from MIST" candidate #323 lists. #323 closes for the switch path once
  Ryan flips it.
- **Re-measurements before flipping**, with the switch off by default:
  - the bulk funnel counts under both methods;
  - Gaia BH1 / BH2 / BH3 known-truth M1 and M2;
  - the literature reproduction counts (must be unchanged);
  - M1 against FLAME (dwarfs and giants separately);
  - M1 against eclipsing-binary dynamical masses (DEBCat, Southworth 2015);
  - the M1 bias from blended light.

### 11.8 Options for Ryan (MP-Q33–Q39; none chosen)

- **MP-Q33, [Fe/H] prior and metallicity data**:
  - (a) N(−0.1, 0.25 dex), the provisional setting;
  - (b) a solar-neighbourhood MDF, e.g. Casagrande et al. (2011) or Hayden et al. (2015), or a
    |z|-dependent thin/thick-disk mixture;
  - (c) add a GSP-Phot [M/H] likelihood with the Andrae et al. (2023) calibration where it is
    reliable.
- **MP-Q34, likelihood floors**: σ_C,floor = 0.02 mag, σ_M,floor = 0.05 mag and ε = 0.1 now (0 or 0.2 measured).
  Alternatively use the per-star BP/RP flux errors (needs a re-query of the flux_over_error
  columns), or a fractional E(B−V) error (Combined19 publishes no per-sightline error).
- **MP-Q35, the M1 point**: the posterior mean (now), 10^⟨log10 M1⟩, or a posterior draw per
  proposal draw (generation-time; it widens the truth M1 like a real population).
- **MP-Q36, deblended truth M1** (§11.3): keep M̂1 (now), or re-fit the subtracted primary per draw
  (generation-time).
- **MP-Q37, companion colours**: a fiducial MIST MS ([Fe/H] prior mean, log age 9.6; now), or the
  row's posterior (age, [Fe/H]).
- **MP-Q39, single-star density in W** (§11.4): `mist_density_ridge_anchored` (provisional
  default; it closes the twins) or `gaussian_ridge` (the first §11.4 form). An empirical
  asymmetric density deconvolved from RUWE < 1.4 rows is a third option, not implemented.
- **MP-Q38, rows outside the ridge's colour range** (about 5% of the parent, mostly bluer than
  C0 = 0.35): W = 1 (now), or extend R(C0) with the MIST single-star locus offset to match the
  measured ridge where they overlap.
- The age prior (constant SFR) and the IMF (Kroupa 2001) are standard choices and are config. They
  are listed so they can be revisited: Chabrier (2003), or a declining SFR.

### 11.9 Implementing the 2026-10-04 decisions (§0.4)

**Coeval companion colours (MP-Q37).** In W (§11.4), the companion's (BP − G, G − RP) at M2, and the system's single-star colour–colour relation h(C), come from the MIST MS of the row's own isochrone, not from a fiducial one.

- **Per row.** That isochrone sits at the row's posterior ⟨[Fe/H]⟩ (linear between the bracketing files) and its nearest grid age to ⟨log age⟩, from `isochrone_mass`. Z_s is per row, so the per-row isochrone is exact for Z_s.
- **Per draw.** Under MP-Q35, the per-draw L uses the same (age, [Fe/H]) that the draw's primary carries.
- **Closed loop.** The synthetic universe already gives the companion its primary's own age and [Fe/H].

**Posterior draw and deblending (MP-Q35, MP-Q36; generation-time).** Each proposal draw i on row s carries a posterior draw ψ_i = (age, [Fe/H], M̂1,i) of the single-star isochrone posterior p₁(ψ | o_s).

- **Sampling.** Sample a (C0, M_G0) cell of the prior map with probability proportional to its weight times the row's kernel; then sample a prior point within that cell by its weight. This is exact up to the map's cloud-in-cell binning.
- **Companion.** Given the drawn q, f and the coeval isochrone I(age, [Fe/H]), the companion is the MIST MS star at M2 = q M1. It contributes a fraction f of the primary's G light, with its BP − G and G − RP from I.
- **Deblending.** The truth M1 minimizes, along I's EEP sequence (linear between EEPs),

  χ²(M1) = [(G_comb − G_sys) / σ_M]² + [(C_comb − C_sys) / σ_C]²

  with G_comb = M_G(M1) − 2.5 log10(1 + f) and C_comb the flux-summed BP − RP. The search is cheap: interpolation only, no SED fit.
- **Truth and record.** gaiamock gets M1 = M1,deblended and M2 = q M1,deblended. Both M1,deblended and M̂1,i are stored, together with χ²_min.
- **Dark companions** (f = 0): the deblended mass is the single-star solution on I.
- **Measure.** The proposal draws q (not M2), with its density evaluated at the row's M̂1. Target and proposal are both per dex of q, so the change of truth M1 does not enter the weight's Jacobian.

The weight under posterior draws follows from writing the target over (ψ, c), t(ψ, c) ∝ φ(ψ) π(c | ψ) p(o_s | ψ, c), and the proposal as p₁(ψ | o_s) q(c). That gives

  w ∝ π(c | ψ) [p(o_s | ψ, c) / p(o_s | ψ, ∅)] / q(c),

normalized per row. In that ratio, ψ is evaluated with M1 profiled along the coeval isochrone, which is the deblending. **This profile approximation (A1″) replaces A1′ of §11.3, and the closed loop must validate it.**

- **MP-Q40 (decided 2026-10-05, §0.5): f comes from the coeval MIST main sequence.** The target is
  `mass_luminosity: mist_coeval`, and the caller passes log10 f_MIST per draw
  (`proposal_set.mist_relation_for_draws`). That uses the draw's own (age, [Fe/H]) after the posterior
  draw, else the row's posterior means.
  - In W, Z_s uses λ_q(M1), with P summed (`malmquist_cmd.build_q_grid`), times Gauss–Hermite nodes of
    N(log10 f_MIST(M1, q M1; the row's isochrone), 0.1 dex).
  - The flux *proposal* stays centred on Janssens, which only shapes efficiency and is still supported.
  - Companions below the Janssens mass range now count in F_lum.

**[Fe/H] (MP-Q33).**
- **Solar-neighbourhood MDF prior:** [Fe/H] ~ N(−0.06, 0.22). This is the Casagrande et al. (2011, Table 1, irfm sample of 5,976 stars; [M/H] has mean −0.02 and σ 0.19). It replaces the provisional N(−0.1, 0.25).
- **The calibration.** The Andrae et al. (2023, §3.5.3) GSP-Phot [M/H] calibration is a MARS model trained on LAMOST DR6. It is distributed as `gdr3apcal` (https://github.com/mpi-astronomy/gdr3apcal).
  - **Inputs:** `teff_gspphot`, `logg_gspphot`, `mh_gspphot`, `azero_gspphot`, `ebpminrp_gspphot`, `ag_gspphot`, `mg_gspphot`, `libname_gspphot` and the position.
  - **Validity, as the tool states it:** Teff 3800–8500 K and [Fe/H] −2.5 to +1, MARCS or PHOENIX libraries only, poor at high extinction. Andrae et al. validated it on cluster members with ϖ/σ_ϖ ≥ 10 and 4000 ≤ Teff ≤ 6500 K.
  - **The reliability cut is that intersection:** ϖ/σ_ϖ ≥ 10, 4000–6500 K, MARCS or PHOENIX, and A0 below a configured ceiling.
  - **Likelihood:** N([Fe/H]_cal; [Fe/H], σ_cal) per [Fe/H] layer of the map, with σ_cal measured on the calibration's own residuals.
  - **Two prerequisites, both needing approval.** (i) Installing `gdr3apcal` (a new dependency) in the shared `.venv`. (ii) A re-query of the five extra GSP-Phot columns for the parent and the real rows (a new snapshot). Until then, the MDF prior applies everywhere.

**Blue rows (MP-Q38): measured, the treatment is Ryan's.** Of 208,292 parent rows with a CMD, 10,367 have C0 < 0.35.

| | blue rows | other rows |
|---|---|---|
| \|b\| median | 2.0° | 8.5° |
| Combined19 E(B−V) median | 1.0 | 0.25 |
| E(BP−RP) median | 1.52 | — |
| distance median | 3.6 kpc | 2.3 kpc |
| observed BP−RP median | 1.61 | — |
| ϖ/σ_ϖ median | 3.1 | — |

- **Most are reddened Galactic-plane stars that the map over-corrects**, not intrinsically blue stars. Only 297 are bluer than 0.35 before dereddening. GSP-Phot gives a median T_eff of 4,980 K.
- About 2,180 have GSP-Phot T_eff > 7,000 K (161 above 10,000 K), so they are candidate genuine A/F or hotter stars.
- 184 sit in the sdB/sdO region of the CMD (4.5 ≤ M_G0 < 7) and 49 in the white-dwarf region (M_G0 ≥ 7).
- So MP-Q38 is mostly an extinction-error question (MP-Q34). No external catalogue was cross-matched.

**Proposal support for evolved rows (MP-Q28d).** For rows flagged evolved, the flux proposal's relation component is centred on the §10.4 evolved relation, `giants.evolved_log10_flux_ratio(M2, M_G0,sys)`, instead of the dwarf relation. `log_f_min` is lowered to −7. The proposal density is evaluated with the same per-row centre, so the weights stay exact. This is a proposal-only change: generation-time for coverage, with no target change.

**Pipeline (§11.7).** The switch flips to `MIST_isochrone` once #425 is fixed:

- **(a) Column ownership in `sample_selection`.** The stage's bulk enrichment writes only `pipeline_*` columns (`pipeline_m1_msun`, `pipeline_m2_msun`, `pipeline_sigma_m2_msun`). It never writes a column a literature sample owns (`m1_msun`, `m2_msun`, `m2_tilde_msun`, `sigma_m2_msun`), and the forward-model cut chains read the `pipeline_*` names. Reproduction counts must not change.
- **(b) Andrews forward-model pass-2 M1** (`pipeline_tag10_bulk`). It keeps TAG10 by name, calling TAG10 explicitly regardless of `mass_calibration.method`. A new method value, `pipeline_bulk`, reads the run's bulk M1 under either method. Which one the Andrews forward model uses is Ryan's call; until then it stays at TAG10, so its counts do not move.

After the flip, re-measure the reproduction counts end to end (they must be unchanged), the bulk funnel, and BH1 and BH2 known truth.

**Dust maps and per-star photometric errors (MP-Q34): evaluation only, no switch without asking.**
- **Flux errors.** The flux-error columns (`phot_bp_mean_flux_over_error`, `phot_rp_mean_flux_over_error`) need a parent re-query.
- **Candidate maps:**
  - Bayestar19 (Green et al. 2019): 3-D, covers dec ≳ −30°, posterior samples per sightline, through `dustmaps` or `mwdust`.
  - Edenhofer et al. (2024): 3-D, all-sky out to about 1.25 kpc (2 kpc in its extended version), posterior samples.

  Both are multi-GB downloads.
- **The trade.** Combined19 stitches Marshall et al. (2006), Green et al. (2019) and Drimmel et al. (2003), and publishes no per-sightline error. A posterior-sample map gives σ_E(B−V) directly but covers less of the parent: Bayestar19 misses the southern sky, and Edenhofer is distance-limited. Coverage fractions are to be measured on the parent once the maps are available.

## 12. Rung 3: fitting the luminous-binary population to the full DR3 sample (#391)

Ryan approved this on 2026-10-09. The luminous-binary parameters are fit by **reweighting the existing gens 23–27 draws** (§3.4): there is no new simulation and no top-up. Whether a top-up is needed, how big and where, is an output of the first fit (§12.8). This section answers MP-Q22 provisionally. Every choice the data or the papers do not settle is a numbered option in §12.9, and the provisional default used to run is labelled **[default]**. None of these is a decision.

### 12.1 Data and observed space

- **Orbits.**
  - *Real:* the rung-2 real sample, i.e. DR3 Orbital + AstroSpectroSB1 after the §0.2 mirror filters (167,911). C1 (P ≤ 0.8 × 1038 d = 830 d) leaves 131,169.
  - *Mock:* `accepted_orbital` draws with fitted P ≤ 830 d.
  - *Coordinates:* G; distance d; log₁₀ P; e.
    - Real: published P and e, and d = 1 / (ϖ + ZP(G)). The ZP is the +22–37 µas G-binned zero point (`distance_count/`).
    - Mock: fitted P and e, and d = 1 / ϖ_fit.
- **Accelerations.**
  - *Real:* the `nss_acceleration_astro` snapshot `20261009T205231Z` under the same mirror filters (337,960).
  - *Mock:* `published_acceleration` draws, with d = 1 / ϖ of the 7/9-parameter fit.
  - *Coordinates:* (G, d). Accelerations carry no period, but they are what constrains the long-P weight (§12.3).
- **Bin edges, fixed before fitting** (`config/population/rung3_fit.yaml`):
  - G: [5, 12, 14, 16, 19.5].
  - d (kpc): [0, 0.3, 0.7, 1.5, 6].
  - log₁₀ P (orbits): [0, 2.3, 2.6, 2.75, log₁₀ 830].
  - e (orbits): [0, 0.3, 0.6, 1].
  - That gives 4 × 4 × 4 × 3 = 192 orbit bins and 4 × 4 = 16 acceleration bins.
  - The edges are chosen for ESS (about 900 accepted-orbit ESS in total), not from the counts. Rows outside the edges are dropped on both sides.

### 12.2 Likelihood

Binned Poisson on all 208 bins. The expected count per bin is μ_b(θ) = Σ_{i∈b} w_i(θ) + s_b(θ): the importance-weighted mock sum plus the spurious component (§12.4).

MC noise enters through the **effective likelihood** of Argüelles, Schneider & Yuan (2019, JHEP 06, 030; "SAY"):
- per bin, with σ_b² = Σ w_i²: α = μ_b² / σ_b² + 1 and β = μ_b / σ_b²;
- L_b = β^α Γ(k_b + α) / [Γ(k_b + 1) (1 + β)^(k_b + α) Γ(α)];
- it reduces to Poisson as σ_b → 0.

The spurious term is analytic and noise-free, so it enters μ_b but not σ_b². A bin with μ_b = 0 and k_b > 0 contributes the Poisson limit at μ_b = s_b. **[default; option MP-Q41]**

### 12.3 Free population parameters (MdS17-anchored)

Each parameter is a multiplicative modifier of the published MdS17 intensity λ₀(x) (§2, `mds17_luminous_log_intensity`), so w_i(θ) = w_i⁰ · m(x_i; θ) / s_low(M1_i). λ₀ is evaluated **without** the MP-Q7 log-linear low-mass scale, which the low-mass slope replaces. All parameters are 0 at MdS17 except the amplitude.

| θ | modifier | MdS17 value |
|---|---|---|
| ln A | overall companion-frequency amplitude e^{ln A} | 0 |
| α_lo | (M1 / 0.8 M⊙)^{α_lo} for M1 < 0.8 M⊙, replacing MP-Q7 (whose log-linear-to-zero is ≈ α 0.5 near 0.4 M⊙) | free; start 0.5 |
| α_hi | (M1 / 0.8 M⊙)^{α_hi} for M1 ≥ 0.8 M⊙ (a tilt on MdS17's own M1 dependence) | 0 |
| γ_P | exp(γ_P (log P − 2.7)): a tilt of the log P distribution | 0 |
| ln L_P | long-P weight e^{ln L_P · σ((log P − 3.0) / 0.1)}, with σ the logistic; set mainly by accelerations | 0 |
| Δγ_q | q^{Δγ_q} (unnormalized; the amplitude absorbs the normalization **[default; MP-Q42]**) | 0 |
| ln F_tw | twin-excess multiplier e^{ln F_tw} for q ≥ 0.95 | 0 |
| Δη | η → η + Δη in p_e = (η + 1) e^η / e_max^{η+1} (normalized; the MP-Q11 floor still applies) | 0 |

Priors are uniform within wide bounds **[default]**. The published 1σ ranges of MdS17's Eqs. 8–25 are an option (MP-Q22). Off in the baseline:
- **Compact companions** (MP-Q17; about 1% of orbits).
- **The Malmquist weight.** The baseline uses the noW weights because the 2-D CMD closed loop is not at "every pull ≤ 3" (§11.5). cmdW is the sensitivity case.

### 12.4 Spurious-contamination component (orbits)

s_b = N_s · p_G(G) · p_d(d) · p_P(log P) · p_e(e), integrated over each bin.

- **p_P** ∝ exp(k_P (log P − 2.919)) and **p_d** ∝ exp(k_d d), each normalized over the bin edges; k_P and k_d are free. This concentrates the component at long P and large distance when k > 0.
- **p_G:** the real **acceleration** sample's G distribution **[default; MP-Q43]**. It is independent of the orbit counts being fit and follows the scan and brightness mix.
- **p_e:** flat in e **[default; MP-Q43]**.
- **Prior** from the symmetric rung-1 re-detection deficit (PR #455 / #457): the spurious share of real C1 orbits in the bin (0.7–1.5 kpc, G 13–16) is N(0.21, 0.04), truncated at 0.
- **No spurious component on accelerations** **[default; MP-Q43]**. DR3 accelerations are also contaminated, but nothing measures that yet.

Free: f_s (the spurious share of all real C1 orbits, the reported quantity), k_P and k_d. With §12.3 that is 11 parameters.

### 12.5 Fitter

- **Maximum likelihood** (scipy L-BFGS-B within bounds; several starts).
- **Uncertainties** from the inverse of a finite-difference Hessian (Laplace) **[default]**. A short dynesty run on the same likelihood is MP-Q44, for when the Laplace approximation is in doubt (bounded or skewed parameters).
- Weights are recomputed from cached per-draw truth and baseline weights only. Nothing is re-simulated.

### 12.6 ESS and MC-noise diagnostics

These are reported at the best fit:
- Kish ESS of the accepted orbits and of the published accelerations, against the MdS17 weights (881 / 696 at rung 2).
- ESS_b per likelihood bin; the bins with ESS_b < 30 (MP-Q19) and the share of the real counts in them.
- Σ σ_b² / μ_b² over the bins (the MC share of the variance).

If the ESS at the best fit falls below about 100 the fit is reported as **ESS-collapsed**: the best fit sits where the draws are sparse, and the result is a pointer for a top-up, not a measurement.

### 12.7 Posterior-predictive checks

All at the best fit, with weights × m(θ) plus the spurious component:
- the six-panel (as rung 2);
- counts vs distance and vs G (the `distance_count/` deficit);
- the e distribution vs distance (the +0.08–0.11 excess);
- the orbit : acceleration ratio vs G and d (`orbit_vs_accel/`).

Thresholds remain MP-Q22.

### 12.8 Top-up rule

A top-up is recommended where the best-fit weights put the likelihood bins below ESS_b 30 while they hold ≥ 1% of the real counts. It is sized for ESS_b ≥ 30 there at the measured ESS per CPU-h. The proposal is re-centred on the best-fit θ (deterministic mixture, §3.4), so no stored draw is invalidated.

### 12.9 Options for Ryan (MP-Q41–Q45; none chosen)

- **MP-Q41 MC-noise treatment.**
  - (a) SAY effective likelihood **[default]**;
  - (b) Barlow–Beeston-lite (one nuisance per bin);
  - (c) Poisson with bins of ESS_b < 30 dropped.
- **MP-Q42 q modifier normalization** (b adopted for the refit, Ryan 2026-10-09, §12.10).
  - (a) unnormalized, with A absorbing it **[default]**;
  - (b) renormalized over 0.3 ≤ q ≤ 1 per (M1, P), so A stays MdS17's f_logP;q>0.3.
- **MP-Q43 Spurious template** (b adopted for the refit, Ryan 2026-10-09, §12.10).
  - (a) p_G from real accelerations, p_e flat, no spurious accelerations **[default]**;
  - (b) p_G and p_e from the real orbits that the symmetric rung-1 run fails to re-detect (300 systems);
  - (c) add a spurious-acceleration fraction with its own prior.
- **MP-Q44 Fitter.**
  - (a) MLE + Laplace **[default]**;
  - (b) dynesty on the 11 parameters.
- **MP-Q45 Bins** (refit 3: chosen from ESS alone, §12.12).
  - (a) the §12.1 edges **[default]**;
  - (b) finer in log P and d once a top-up raises the ESS.

The parameter set and priors themselves stay MP-Q22.

### 12.10 Refit model (Ryan, 2026-10-09: MP-Q42b and MP-Q43b adopted for the refit)

These two changes are the refit's model. Everything else in §12.1–12.8 stands. The bins are unchanged, so the edges stay fixed before fitting.

**Proposal.** The refit adds the re-centred tuning generation 28 (`config/population/proposal_set_restart_recentred28.yaml`) to gens 23–27 in the deterministic mixture (§3.4).
- Its (log q, log P) shape component carries the first-fit modifiers (`M2PeriodShapeConfig` re-centring fields).
- Its eccentricity mixture puts more weight on the e^{−0.9} floor component.
- Its defensive parts are unchanged.

**MP-Q42b: normalized q modifiers.** The q tilt and the twin multiplier are normalized per draw over MdS17's own q density on 0.3 ≤ q ≤ 1:

m_q(q | M1, P) = g(q) / ∫_{0.3}^{1} p_q(q′ | M1, P) g(q′) dq′, with g(q) = q^{Δγ_q} e^{ln F_tw · [q ≥ 0.95]}.

So e^{ln A} is again a multiplier of MdS17's f_logP;q>0.3, which separates A from Δγ_q and F_tw. The integral uses a fixed 200-point q grid of `moe_distefano.q_density` per draw, precomputed once.

**MP-Q43b: spurious template from the non-re-detected real orbits.** In the PR #455 re-injection, 300 real orbits (0.7–1.5 kpc, G 13–16) were each re-injected 3 times at their own fitted orbits. The 218 matched mock accepted orbits were re-injected the same way.

- **Per-cell spurious probability.** Define π(log P, e) = max(0, (A_mock − A_real) / A_mock): the excess failure of real orbits over the mock's own self-consistency failure. This is evaluated on the §12.1 (log P, e) cells.
- **Sparsity.** The 300 systems are too sparse for the 4 × 3 joint cells: there is almost nothing at log P < 2.3, and only about 20–40 systems per cell elsewhere. So π is taken as **separable**, π(log P, e) = π̄ · r_P(log P) · r_e(e), from the two 1-D marginals.
  - Each marginal ratio is shrunk toward 1 by n / (n + 20), where n is the number of real systems in the marginal bin.
  - A log P bin with fewer than 10 real systems is set to r_P = 1.
- **Extrapolation beyond the measured bin.** The template assumes the spurious share depends on (log P, e) through π and on distance only through a free tilt exp(k_d (d − 1.1 kpc)). It has **no G dependence**.
  - The expected spurious count per bin is s_b = f_s · π(log P_b, e_b) · exp(k_d (d_b − 1.1)) · k_b. That is a share of the *observed* real count in the bin, so the template is data-anchored, not generative.
  - This assumption is untested outside 0.7–1.5 kpc and G 13–16: the 300 systems carry no information on G or d.
- **Amplitude prior.** Unchanged: the spurious share of real C1 orbits in the prior bin (d 0.7–1.5 kpc, G 12–16) is N(0.21, 0.04), truncated at 0.
- **Parameter changes.** f_s is now the multiplier of the template, about 1 when the measured deficit is all spurious. k_P is dropped, because π carries the P shape. That leaves 10 free parameters.

The first-fit options MP-Q41, Q44 and Q45 keep their defaults.

### 12.11 Refit 2 model (Ryan, 2026-10-09: additive spurious component and Beta eccentricities)

This replaces the §12.10 spurious term and the single η shift. The q normalization (MP-Q42b), the bins (§12.1), the likelihood (§12.2), the gens 23–28 mixture and the fitter (§12.5) are unchanged.

**Spurious component: additive and independent of θ.**
- **Expected counts.** μ_s,b = N_s · T_b, added to the mock expectation in each orbit bin. Accelerations carry no spurious term. T is a fixed, normalized shape, T_b = T_GD(G, d) · T_PE(log P, e), and it never depends on the population parameters or on the counts being fit.
- **T_PE.** The non-re-detection fraction π(log P, e) from the PR #455 re-injection, built exactly as in §12.10: π̄ times the shrunk, separable log P and e ratios, with the sparse bins set to 1. It is multiplied by the real C1 orbit counts in the same (log P, e) cells *inside the measured bin* (0.7–1.5 kpc, G 12–16), then normalized. So T_PE is the (log P, e) distribution of the orbits that fail to re-detect there.
- **T_GD.** Proportional to the real C1 orbits' marginal (G, d) distribution over all bins. **This is an extrapolation:** the 300 re-injected systems lie in one (G, d) bin and carry no information on how contamination varies with G or d. The template simply assumes it follows the orbit sample.
- **Prior on the amplitude.** The spurious share of real C1 orbits in the prior bin is N(0.21, 0.04), truncated at 0. It is converted once, using the fixed observed count k_prior and the template mass T_prior in that bin: N_s ~ N(0.21 · k_prior / T_prior, 0.04 · k_prior / T_prior). This prior is not revisited.
- **Sensitivity cases:**
  - (b) N_s fixed at the prior mean (f_s = 1);
  - (c) no spurious component (N_s = 0).

**Eccentricity: Beta distributions per log P range.**
- **Form.** p_e(e | P) = Beta(e / e_max(P); a_k, b_k) / e_max(P) on [0, e_max(P)), as used for exoplanet eccentricities by Kipping (2013). MdS17's e_max(P) truncation and the P ≤ 2 d circular class are kept.
- **Ranges, fixed up front:**
  - k = 1: log P < 2.0 (2–100 d);
  - k = 2: 2.0 ≤ log P < 2.6;
  - k = 3: log P ≥ 2.6.

  That is six parameters, replacing Δη. Range 3 has no orbit constraint above log P 2.92, so its (a, b) are set by the orbits at 400–830 d and only extrapolated beyond. The draws there enter only the acceleration bins, which carry no e information.
- **MP-Q22** (rung-3 eccentricity parameterization) is resolved by this choice.
- **Bounds and proposal coverage.** Every generation's e proposal has a floor component ∝ e^{−0.9} on [0, e_max) (weight 0.30, or 0.45 in generation 28) and U(0, e_max) (0.50, or 0.40). Hence:
  - p/q is bounded near e = 0 when a ≥ 0.1;
  - p/q is bounded near e_max when b ≥ 1;
  - the variance is finite near e = 0 when a > 0.05.

  So the bounds are a ∈ [0.1, 20] and b ∈ [1, 30]. The refit reports the maximum p/q of the fitted Beta against each generation's e proposal, and flags any range where coverage is poor. No new draws are generated for this.

Free parameters: ln A, α_lo, α_hi, γ_P, ln L_P, Δγ_q and ln F_tw (§12.3, q normalized), a_1–a_3 and b_1–b_3, and N_s. That is 14.

### 12.12 Refit 3: coarser bins chosen by ESS, and a floor on empty bins (Ryan, 2026-10-09)

The §12.11 model is unchanged: the same 14 free parameters, no new priors, and the P < 100 d Beta range left free (Ryan declined priors on the Beta parameters and fixing that range). Two things change.

**Bins chosen from ESS alone, before any fit residual is seen.**
- **Candidates.** Each axis has a short list of nested edge sets:
  - G: [5, 12, 14, 16, 19.5], [5, 13, 19.5] or one bin;
  - d (kpc): [0, 0.3, 0.7, 1.5, 6], [0, 0.5, 1.0, 6] or [0, 0.7, 6];
  - log P: [0, 2.3, 2.6, 2.75, 2.919], [0, 2.5, 2.75, 2.919] or [0, 2.6, 2.919];
  - e: [0, 0.3, 0.6, 1], [0, 0.4, 1] or one bin.

  The orbit candidates are every product of these. The acceleration candidates are every (G, d) product.
- **ESS evaluation.** ESS_b of the gens 23–28 noW mock weights is computed in every bin at two parameter points:
  - the refit-2 best fit (`docs/gate391/rung3_refit2/summary.json`);
  - MdS17: ln A = 0, α_lo = 0.5, all other population modifiers 0, and Beta a = 1.4, b = 1 in every range, which is MdS17's power law at η ≈ 0.4.
- **Rule** (ESS and real counts only; no fit residual is involved):
  - "Big" bins are those holding ≥ 1% of the real orbits.
  - Among the candidates where every big bin has min-over-both-points ESS_b ≥ 30, take the one with the most bins.
  - If none qualifies, take the candidate that maximizes the share of real orbits in bins with ESS_b ≥ 30 at both points; break ties by more bins.
- **Record.** The chosen edges are written to `config/population/rung3_refit3.yaml` before fitting.

**Spurious prior in a binning-independent form.**
- T_GD ∝ the real C1 orbits' (G, d) counts, and T_PE sums to one over (log P, e). So N_s · T gives the same spurious *share* of real orbits in every (G, d) region.
- The §12.11 prior (0.21 ± 0.04 of the real orbits in 0.7–1.5 kpc, G 13–16) is therefore exactly N_s ~ N(0.21 · N_o, 0.04 · N_o), where N_o is the number of real C1 orbits inside the edges. This is fixed once.
- T_PE is rebuilt on the chosen (log P, e) edges with the §12.10 rules: the PR #455 marginal ratios, shrinkage, and sparse bins set to 1. It is weighted by the real orbits in 0.7–1.5 kpc and G 13–16, taken unbinned in (G, d).

**Floor on empty bins.**
- An orbit or acceleration bin with **no mock draw** in gens 23–28 has a mock expectation of exactly 0 for every θ.
- Its expectation is set to a fixed floor φ = 1 count, attributed to the spurious component, with the N_s · T term removed from it. The bin then contributes a θ-independent constant to −lnL, so it cannot move any parameter or pull N_s.
- The value 1 (one expected orbit) only shifts the reported −lnL. It is the smallest count a real bin can hold.
- These bins are listed separately: count, real orbits and location.

Fits: noW (the baseline) and cmdW, each with ≥ 12 starts. The ESS rules, posterior-predictive checks and top-up rule of §12.6–12.8 apply. The top-up projection is re-centred on this fit only if the fit is stable and the checks close.

## References

- Andrae, R. et al. 2018, A&A 616, A8 (BC_G: Eq. 7, Table 4).
- Andrae, R. et al. 2023, A&A 674, A27 (GSP-Phot; [M/H] systematics).
- Argüelles, C. A., Schneider, A. & Yuan, T. 2019, JHEP 06, 030 (effective likelihood with MC-noise, §12.2).
- Babusiaux, C. et al. (Gaia Collaboration) 2018, A&A 616, A10 (Gaia-band extinction law).
- Badenes, C. et al. 2018, ApJ 854, 147 (APOGEE close-binary fraction vs log g).
- Bailer-Jones, C. A. L. et al. 2021, AJ 161, 147.
- Boubert, D. & Everall, A. 2020, MNRAS 497, 4246 (Completeness of the Gaia-verse II).
- Cantat-Gaudin, T. et al. 2023, A&A 669, A55 (empirical Gaia DR3 selection function; `gaiaunlimited`).
- Casagrande, L. et al. 2011, A&A 530, A138 (Geneva–Copenhagen re-analysis; solar-neighbourhood MDF, Table 1).
- Drimmel, R. et al. 2003, A&A 409, 205 (3-D dust model, part of Combined19).
- Edenhofer, G. et al. 2024, A&A 685, A82 (3-D dust map with posterior samples).
- Castro-Ginard, A. et al. 2024, A&A 688, A1 (RUWE detectability of unresolved binaries).
- Chiavassa, A. et al. 2011, A&A 528, A120 (convection-driven photocentre jitter).
- Choi, J. et al. 2016, ApJ 823, 102 (MIST).
- Chabrier, G. 2003, PASP 115, 763 (IMF).
- Creevey, O. L. et al. 2023, A&A 674, A26 (Gaia DR3 FLAME).
- Dotter, A. 2016, ApJS 222, 8 (MIST 0: EEPs).
- Eggleton, P. P. 1983, ApJ 268, 368 (Roche-lobe radius).
- El-Badry, K. et al. 2024, OJAp 7, 100 (arXiv:2411.00088).
- Elvira, V., Martino, L., Luengo, D. & Bugallo, M. 2019, Statistical Science 34, 129.
- Everall, A. & Boubert, D. 2022, MNRAS 509, 6205 (Completeness of the Gaia-verse V).
- Fabricius, C. et al. 2021, A&A 649, A5 (Gaia EDR3 catalogue validation).
- Gaia Collaboration, Arenou, F. et al. 2023, A&A 674, A34.
- Halbwachs, J.-L. et al. 2023, A&A 674, A9 (arXiv:2206.05726).
- Green, G. M. et al. 2019, ApJ 887, 93 (Bayestar19).
- Hayden, M. R. et al. 2015, ApJ 808, 132 (APOGEE MDFs across the disk).
- Hesterberg, T. 1995, Technometrics 37, 185.
- Janssens, S. et al. 2022, A&A 658, A129.
- Kipping, D. M. 2013, MNRAS 434, L51 (Beta distribution for orbital eccentricities, §12.11).
- Kjeldsen, H. & Bedding, T. R. 1995, A&A 293, 87 (asteroseismic scaling relations).
- Kroupa, P. 2001, MNRAS 322, 231 (IMF).
- Marshall, D. J. et al. 2006, A&A 453, 635 (3-D extinction in the inner Galaxy, part of Combined19).
- Miglio, A. et al. 2012, MNRAS 419, 2077 (RGB mass loss, NGC 6791 / 6819).
- Moe, M. & Di Stefano, R. 2017, ApJS 230, 15 (arXiv:1606.05347).
- Öpik, E. 1923, Publ. Tartu Obs. 25, 6.
- Owen, A. & Zhou, Y. 2000, JASA 95, 135.
- Pecaut, M. J. & Mamajek, E. E. 2013, ApJS 208, 9.
- Pinsonneault, M. H. et al. 2025, ApJS 276, 69 (APOKASC-3).
- Price-Whelan, A. M. & Goodman, J. 2018, ApJ 867, 5 (APOGEE binaries along the RGB).
- Raghavan, D. et al. 2010, ApJS 190, 1.
- Riello, M. et al. 2021, A&A 649, A3.
- Santos, N. C. et al. 2013, A&A 556, A150 (mass correction).
- Southworth, J. 2015, ASP Conf. Ser. 496, 164 (DEBCat).
- Torres, G., Andersen, J. & Giménez, A. 2010, A&ARv 18, 67.
- Veach, E. & Guibas, L. 1995, SIGGRAPH '95, 419.
- Vehtari, A. et al. 2024, JMLR 25, 72 (Pareto-smoothed importance sampling).
- Verbunt, F. & Phinney, E. S. 1995, A&A 296, 709 (tidal circularization of giants).
- Winters, J. G. et al. 2019, AJ 157, 216.
- Zahn, J.-P. 1977, A&A 57, 383 (equilibrium tide).
