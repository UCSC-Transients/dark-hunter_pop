# dark-hunter_pop — Architecture & Pipeline Specification

Status: Decisions locked (2026-08-25). Scaffold exists; Foundation implementation waits on
approval of the Foundation task list against this document.

## 0. Purpose and non-goals

**Goal:** a debiased mass function, dN/dM, for compact-object companions (WD / NS / BH) found in
Gaia astrometric binaries, via forward-modeling a population model through `gaiamock` and comparing
to the real Gaia NSS sample with an inhomogeneous Poisson point-process likelihood.

**Normalization:** shape/rate relative to the real, quality-cut parent NSS sample, not an absolute
Galactic space density. No stellar-population-synthesis / star-formation-history layer.

**Explicitly out of scope for v1** (full list in §9):
- Binary stellar evolution modeling. M1, P, e are drawn from phenomenological population priors;
  M2 (the compact-object mass) is the free, non-parametric target.
- Absolute Galactic rate normalization.
- The genuine-triple (unrelated third star) population component — module built, forced to
  P(triple)=0 in v1.
- The **fully joint** inference (population model + both selection functions + per-system
  companion-nature/outlier likelihoods all sampled together) — v1 uses a staged-but-connected
  approximation instead (§4, Stage 8); fully joint is documented v2 work, warm-started from the
  v1 result.
- SPHEREx (documentation only). Lam, El-Badry & Simon (2025)'s analytic selection function
  (documented as a future cross-check, not used).

## 1. Repository layout

Repo: `UCSC-Transients/dark-hunter_pop`. Local base directory:
`/Users/rfoley/darkhunter/pop/dark-hunter_pop/` (repo root itself). Python package:
`darkhunter_pop`.

```
dark-hunter_pop/
├── config/
│   ├── config.yaml             # single merged canonical config
│   └── fragments/              # tracked per-domain drafts; merged into config.yaml at checkpoints
├── runs/                       # per-run YAML run files (see §5); tracked in git
├── data/                       # gitignored raw data (Gaia snapshots, sheet dumps, cooling tracks)
│   ├── dr3/gaia_snapshots/
│   ├── dr4/gaia_snapshots/     # reserved; DR4 not runnable yet
│   └── target_lists/
│       └── snapshots/          # weekly sheet dumps (gitignored); derived dates tracked elsewhere
├── output/                     # gitignored stage HDF5 artifacts (paths.artifact_root)
├── src/darkhunter_pop/
│   ├── constants.py            # true physical constants only (astropy.constants + extras)
│   ├── run_management.py       # stage registry, caching/skip logic, run-file I/O (foundation)
│   ├── physics_utils.py        # unit handling, point-process primitives — NOT a Kepler solver
│   ├── schemas.py              # pydantic interface contracts (ParameterSet, CandidateRecord, ...)
│   ├── data_acquisition.py     # stage: data_acquisition
│   ├── mass_derivation.py      # stages: mass_derivation_bulk, mass_derivation_refined
│   ├── rv_consistency.py       # stages: rv_astrometry_gate, joint_orbit_fit
│   ├── companion_nature.py     # stage: companion_nature_likelihood
│   ├── triples/                # stage: triples (off by default)
│   │   ├── tess_variability.py
│   │   └── rotation_check.py
│   ├── forward_model.py        # stages: selection_function_astrometric, selection_function_followup
│   ├── population_model.py     # stage: population_model
│   ├── sensitivity_analysis.py # stage: sensitivity_analysis
│   ├── inference.py            # stage: inference
│   ├── plotting.py             # shared rendering primitives
│   ├── diagnostic_hooks.py     # low-level hooks shared by early stages + diagnostics (infra-only imports)
│   └── diagnostics.py          # stage: diagnostics
├── vendor/
│   ├── DATA_MANIFEST.md        # SHA256s for Release gaiamock-mod-v1 assets
│   ├── overlays/gaiamock_mod.py  # tracked mod source; install copies into submodule
│   └── gaiamock/               # git submodule @ upstream main + installed overlay (§1.1)
├── tests/
├── scripts/                    # entry point, install_gaiamock_mod, purge_run
├── notebooks/
└── docs/
```

Most modules are one file under `src/darkhunter_pop/`. `triples/` is the one exception.
Promote any other module to a subdirectory only once it has outgrown a single file.

The `src/` layout deliberately differs from `dark-hunter_rv` / `dark-hunter_sed` (flat packages);
sibling repos may align later. This document is authoritative for `dark-hunter_pop`.

### 1.1 Vendored `gaiamock` (modified-RUWE only)

Science paths import **`gaiamock_mod` only** (as `import gaiamock_mod as gaiamock`). Stock
`gaiamock` must not be used for RUWE / selection-function work.

Install (scripted by `scripts/install_gaiamock_mod.sh`):

1. Submodule `vendor/gaiamock` pinned to a commit on upstream `main` (`kareemelbadry/gaiamock`,
   MIT) via `git submodule update --init`.
2. Overlay from the modified-RUWE pack (upstream README Box link; mirrored on immutable GitHub
   Release `gaiamock-mod-v1` — **do not** host the default ~984 MB `healpix_scans.zip`):
   - `gaiamock_mod.py` — tracked at `vendor/overlays/gaiamock_mod.py`, copied into the submodule
     working tree; mirrored on the Release
   - `healpix_16_med_ruwe.npz` — Release / `mod_files/` → `vendor/gaiamock/`
   - `individual_ccds.zip` contents → `vendor/gaiamock/healpix_scans/`
3. Compile `kepler_solve_astrometry.so` inside `vendor/gaiamock/` (requires GSL + gcc).

Version triple recorded in config and every run/stage record (`darkhunter_pop.gaiamock_vendor`):
`gaiamock_mod_release`, `gaiamock_mod_sha256`, `gaiamock_git_commit`. Mismatch at a
gaiamock-using stage start → refuse (same class as config mismatch). Checksums live in
`vendor/DATA_MANIFEST.md`. Import science paths via `import_gaiamock_mod()` only.

Local staging drop: `mod_files/` (gitignored). Default 49 152-file healpix set is never hosted
or required for v1.

**Foundation-phase task**: audit `gaiamock_mod`'s public API before scoping `physics_utils.py`
(Kepler solver, RUWE prediction, 5/7/9/12-parameter cascade, epoch-astrometry fitter) so nothing
here reimplements it. Write `docs/GAIAMOCK_API.md`.

## 2. Foundation layer — built first, gates all parallel work

1. **`constants.py`** — true physical constants only, via `astropy.constants` where available
   (c, G, …) plus named extras that are not choosable (e.g. `M_Ch`; optional `Delta_M_Ch` lives
   in config). **Not** variables, thresholds, or method choices — those are config (including
   `M_MIN`, `M_TOV` prior, SN-kick velocity, TAG10 method flag, `sigma_logM`, Santos on/off,
   cooling-track choice). No physics constant hardcoded anywhere else.
2. **`physics_utils.py`** — unit conversions and Poisson point-process primitives only.
3. **`schemas.py`** (§3).
4. **`run_management.py`** (§5) — the stage-execution framework every subagent's stage code must
   conform to.
5. **Config schema** — one pydantic-validated section per domain, assembled into `config.yaml`;
   fragments under `config/fragments/` are tracked and merged at review checkpoints.

No subagent starts stage-implementation work until this layer's interfaces are frozen and
reviewed.

## 3. Interface contracts (schemas)

- **`ParameterSet`** — the standard output type for essentially every fit stage (MSC/gspphot
  Teff-logg-[Fe/H], the TAG10 mass+radius derivation, the RV/orbit fit, the final M1+M2
  derivation, uberMS's multi-parameter output): a named vector of quantities plus a **covariance
  matrix** (v1). Joint posterior samples are a documented alternate on-disk format (HDF5) for a
  future switch — stubbed in the schema / docs only in v1, not implemented. Most quantities here
  are jointly fit and correlated by construction, so joint storage is the default, not an opt-in.
  A plain single-value-with-uncertainty accessor is available as a marginal view. A bare scalar
  type is reserved only for genuinely standalone external inputs never jointly fit with anything
  else tracked here.
  Every `ParameterSet` carries a **provenance tag** (which method/tier produced it — TAG10 vs.
  gspphot-fallback vs. uberMS; `astrometry_only` vs. `joint_astrometry_rv` orbit tier) as metadata
  only — provenance does not by itself justify inflating the reported uncertainty.
  Stage outputs are stored as **one HDF5 file per stage** under `paths.artifact_root` (`output/`
  by default).
- **`CandidateRecord`** — one row per Gaia source: identifiers, NSS solution type, Thiele-Innes
  elements where available, reconstructed NSS fitted-parameter ``ParameterSet``
  (``nss_solution`` from ``corr_vec``/``bit_index``; absent on reconstruction failure — never a
  diagonal-only fallback); the full field set from the RV pipeline's per-star summary output
  (Gaia metadata block, NSS orbital parameters, external literature RV rows, internal pipeline RV
  epoch results, ideally ingested from a JSON version of `dark-hunter_rv`'s summary output); static
  (single/average, not epoch) photometry across all bands in use; a TESS block holding only
  derived products (period, amplitude, variability flag, implied-v_rot) plus a pointer to the full
  light curve stored externally.
- **`OutlierTestResult`** — RV/astrometry consistency gate output: chi2/dof, per-instrument fitted
  offset+jitter, pass/fail against the config threshold.
- **`OrbitTier`** — `astrometry_only` vs. `joint_astrometry_rv`.
- **`FitTier`** — `bulk_estimate` (TAG10, fast pass) vs. `full_uberMS` (queued, refined pass).
- **`FollowUpRecord`** — per-system target-list membership + adoption dates, N_observations, time
  span, brightness, declination, proper motion.
- **`RunManifest`** — see §5; this is now defined as the live YAML run file itself, not a one-shot
  end-of-run emission.

## 4. Pipeline stages

### `data_acquisition`

- Query `gaiadr3.nss_two_body_orbit`. Cross-match via Gaia's `*_best_neighbour` tables (GALEX AIS,
  PS1, 2MASS, AllWISE) plus SDSS and DECam-u (additions to `gather_phot` in `dark-hunter_sed`).
- **Duplicated `source_id`s are classified before a single HDF5 byte is written**
  (`assert_unique_source_ids` → `classify_duplicate_source_id_group`; issues #221, #237, #241,
  #242). Never an automatic merge, either way: on the uncut parent snapshot 5,926 of 5,932
  duplicated `source_id`s are *distinct NSS orbital solutions for one source* (differing
  `nss_solution_type` / `period`), not cross-match fan-out, and merging cells across them fabricates
  orbits that exist in no Gaia row (PR #238 reverted by PR #239). **Resolved and implemented
  (`CONTINUATION_PLAN.md` §15 Q17, 2026-09-24, Ryan Foley; issues #237/#241/#242): genuine
  multi-solution duplicates are kept and tagged, never merged or collapsed** — each duplicate group
  is classified as `cross_type` (distinct `nss_solution_type` families) or
  `same_type_period_aliased` (more than one period-aliased `Orbital`-family solution, possibly both
  at once), and every row in a kept group is written through unmodified, carrying only its own
  Gaia row's fields including its own `nss_solution_type`. The remaining 6 groups (0.1%) are genuine
  cross-match fan-out (identical `nss_solution_type` **and** identical `period`, differing only in
  2MASS `J/H/Ks` values and errors — one clean PSC/XSC oid in `gaiadr3.tmass_psc_xsc_join` maps to
  more than one `original_psc_source_id`; #290). Measured on `main` @ `c07897e` (#290): 5 of the 6
  survive `quality_cut_bins` (all `SB1`; the sixth, `EclipsingBinary` at G≈18.8, is cut), 0 are
  `unresolved`, and the post-cut `cross_type` count is 3,841.
- **Cross-match fan-out is collapsed with the conflicting bands masked** (#221 option B, Ryan Foley,
  2026-09-27). `collapse_crossmatch_fanout` runs after `apply_quality_cuts` and before the duplicate
  guard. A duplicated `source_id` group collapses to one row **only if** its rows are cell-for-cell
  identical (masked/NaN equal only to masked/NaN) in every column except the `<band>_mag` /
  `<band>_mag_err` columns of `dr3.crossmatch_fanout_maskable_bands` (`[J, H, Ks]`; must name
  enabled `external_photometry_crossmatches` bands; `[]` disables the collapse). The kept row is the
  group's first table row, so every surviving cell is that row's own — nothing is copied between
  rows and no measurement is chosen: a band whose value **or** error differs across the rows is
  masked in both value and error (read downstream as an absent band, exactly like a source with no
  2MASS match); a band that agrees is kept. Each collapsed candidate carries
  `extras.crossmatch_fanout_collapsed = True`, `crossmatch_fanout_n_matches` (rows collapsed) and
  `crossmatch_fanout_masked_bands` (possibly empty). A fan-out-shaped group that differs in any
  other column is not collapsed and is still refused by `DuplicateSourceIdError` (counted as
  fan-out that "could not be collapsed", separately from `unresolved`); `unresolved` shapes are
  still refused; genuine multi-solution groups never match the collapse (their orbit columns
  differ) and pass through untouched. Option A — joining 2MASS PSC on Gaia's own best-neighbour
  designation so no fan-out enters at the query — is deferred to the next snapshot. The funnel
  report records the per-sub-case kept multi-solution counts (`FunnelCounts.multi_solution`, a
  `MultiSolutionCounts`) and the collapse (`FunnelCounts.crossmatch_fanout`:
  `crossmatch_fanout_groups_collapsed` / `_rows_removed` / `_sources_masked` / `_bands_masked`).
  Limitation: `dark-hunter_sed` gathers its own photometry, so the mask does not reach an SED fit
  run upstream; it governs only pop's own `CandidateRecord.photometry`.

#### Multi-solution sources (real, not a bug — §15 Q17 resolved)

Gaia's NSS pipeline sometimes publishes more than one row for the same `source_id` because it runs
multiple independent solution families and more than one passes its internal quality gates. This is
distinct from, and far more common than, the 6-group cross-match fan-out above. Two sub-cases, both
real and both in scope:

1. **Cross-type multiplicity** — different `nss_solution_type` families for the same source (e.g.
   an `Orbital` astrometric row *and* a separate `SB1`/`AstroSpectroSB1` RV row). Different
   observable channels on the same physical system, not competing fits of the same thing.
2. **Same-type multiplicity (period aliasing)** — more than one `Orbital`-type solution with
   different (aliased) periods for the same source: genuine period-aliasing ambiguity in the
   astrometric-only solution, competing fits of the same channel.

**Ruling (verbatim, Ryan Foley, 2026-09-24):** "the simulations should match each sample, including
duplicates" and, confirming this covers period-aliasing too, "the multiple Orbital solutions should
also be reproduced." Binding consequences for downstream design:

- `data_acquisition.py` must **tag, never merge or collapse**, genuine multi-solution rows — each
  row is kept in the output and carries its own `nss_solution_type`; the 6 genuine cross-match
  fan-out cases are handled by their own collapse-and-mask logic (#221, above), unaffected by this
  ruling.
- `forward_model.py`'s `selection_function_astrometric` **and** `selection_function_followup` must
  reproduce the **empirical multi-solution emission rate** — both cross-type and same-type/period-
  aliasing — so a simulated system can emit more than one solution-type row at the rate real systems
  do.
- A source counts however many rows the real (and correspondingly the mock) selection process
  actually produces for it — never idealized to always-one or always-two — so real and simulated
  catalogs are counted the same way when compared in the likelihood.
- `population_model.py` / `inference.py` must count a multi-row system consistently on both sides of
  the real-vs-mock comparison so multiplicity is never double-weighted.

This domain decision is authoritative and not to be second-guessed in a coding session; see
`docs/CONTINUATION_PLAN.md` §15 Q17 for the full record.

**Implementation, `population_model.py` / `inference.py` (issue #244).** The chosen convention is
**each kept NSS row is its own independent observation in the unbinned Poisson point-process
likelihood** — not "group by `source_id` with an explicit multiplicity term." This was found to
already be the codebase's structure end to end, not a new design: `population_model.py`'s
`collect_system_weights` builds one `SystemPopulationWeight` per input `CandidateRecord` row (never
per distinct `source_id`), `inference.py`'s `collect_observed_events` correspondingly builds one
`ObservedEvent` per row, and the unbinned likelihood sums `Σ_i log λ(x_i)` with one term per event
(`physics_utils.poisson_log_likelihood_inhomogeneous`) — so a source with 2 kept rows contributes 2
terms, symmetric with a mock system that emits 2 rows under `forward_model.py`'s multi-solution
emission model (#243). The rejected alternative (group-by-`source_id` with a multiplicity term) was
ruled out because rows of one source can carry different `m2_msun` point estimates from different
`nss_solution_type` fits, so grouping would have to either discard mass information or invent a new
aggregate with none — per-row independence uses exactly what each row already carries.

Issue #244 audited both modules for implicit one-row-per-system assumptions and found the core
likelihood sum was already correct; the audit's actual findings were in the *reporting* layer:
`population_model`'s legacy `n_systems` payload key literally counts rows, not distinct sources
(kept, for backward compatibility, but now documented and paired with new `n_distinct_source_ids` /
`row_multiplicity_histogram` fields — `population_model.compute_row_multiplicity`), and
`inference.collect_observed_events`'s optional `eccentricities: Mapping[int, float]` parameter is a
source_id-keyed fallback that cannot distinguish rows of the same source (a per-row `eccentricity`
field on `system_weight_rows`, when present, already takes precedence — documented, not changed).

A genuine gap the audit *did* find, and deliberately did **not** fix in #244 (escalated to #252
rather than silently landed, per #244's own escalation instructions — folding row-multiplicity into
the Poisson normalization is a statistical-modeling decision, not mechanical bookkeeping):
`inference.read_astrometric_sf_scalar` / `read_followup_sf_scalar` — which set the SF factors in
`Λ = MF(M) × astro_sf × followup_sf × sample_sf` — currently estimate `P(≥1 row accepted)` per mock
system, not `E[rows accepted]`, so `Λ` does not yet reflect the mock's own multi-solution row
emission rate (#243). `inference.row_multiplicity_consistency_diagnostic` (backed by
`real_row_multiplicity_stats` and `estimate_mock_expected_row_multiplicity`, which reads
`forward_model.load_multi_solution_rate_table`) reports the real-vs-mock mean-rows-per-source
comparison as a new **diagnostic-only** field on the `inference` payload (`row_multiplicity_diagnostic`,
alongside `posterior_prior_overlap` / `zero_count_upper_limits`) so this asymmetry is visible without
being silently baked into the likelihood. See issue #252 for the follow-up.

**Reproduction-mode counting against a published N (issue #274).** A literature paper's published N
counts objects, not NSS rows. **Ruling (verbatim, Ryan Foley, 2026-09-27):** "In each solution type,
count distinct stars." So every count that `sample_diagnostics` compares against a published N
(`compare_to_published`, `build_reproduction_comparisons`, the attrition waterfall's `n_stars` line)
is the number of distinct `source_id`s **within each `nss_solution_type`, summed over types**
(`sample_selection.distinct_star_count`). Same-type duplicate rows of one star (period aliases,
cross-match fan-out) count once; a star with an `Orbital` row and an `SB1` row that **each passed
their own branch chain** counts once in each type. A multi-branch union is therefore the sum of
per-type distinct counts (El-Badry 2026: 227 = 76 astrometric + 151 spectroscopic). Per-type
survivors come from the rows that actually passed (`SampleEvaluationResult.*_by_solution_type`), not
from every row of a surviving star: the per-row `surviving_source_ids` emits all of a surviving
star's rows, including rows of other types that never passed any chain, and those do not count.
The per-row count (`recovered_n_rows`) and the distinct-across-types count
(`recovered_n_distinct_sources`) are still reported, as informational fields only. **This is
reporting only**: `data_acquisition` tag-and-keep, `forward_model` emission, `surviving_source_ids` /
`inference_source_ids`, and the per-row likelihood above are unchanged.
- Quality cut: goodness-of-fit vs. magnitude, configurable as **N separate (magnitude, threshold)
  bins** (El-Badry et al. 2023's <5/G>13, <10/G≤13 as the v1 default values; the mechanism
  supports arbitrary bin counts, since DR4 may need a different scheme).
- Snapshot: raw query result + checksum + literal ADQL text + query date. Full accounting
  standard, not bitwise-exact reproducibility.
- Reconstruct NSS fitted-parameter covariance from ``corr_vec`` + ``bit_index`` + per-parameter
  ``*_error`` columns; store as ``CandidateRecord.nss_solution`` and under
  ``data_acquisition/nss_covariance/`` in the stage HDF5. Failures are counted in the funnel
  (``covariance_health``); no diagonal-only fallback.
- **NSS enrichment join (#308).** The documented uncut snapshot `20260826T234425Z_3d3f740b080c`
  was queried with an ADQL that predates ``corr_vec`` / ``bit_index`` / the NSS-native
  ``ra/dec/pmra/pmdec_error`` / K1 columns, so replaying it alone reconstructs **0** covariances
  (every row ``missing_corr``). After the fan-out collapse and before rows become
  ``CandidateRecord``\ s, ``data_acquisition`` overlays the frozen ``nss_enrichment`` snapshot
  (``scripts/fetch_nss_enrichment.py``; never re-run) with the same
  ``merge_nss_enrichment_into_row`` / ``_enrichment_join_key`` join the literature-sample parent path
  uses, keyed on ``(source_id, nss_solution_type)`` so each row of a multi-solution source gets
  **its own** solution's covariance. The source is ``dr3.nss_enrichment_snapshot`` (directory under
  ``{data_root}/dr3/gaia_snapshots/``; ``dr4``: ``null``), a path-specific key in the stage
  fingerprint. (``sample_selection`` has a sibling key, ``dr3.shahaf2023b_class3_snapshot``. It names
  a frozen Shahaf et al. 2023b Table 2 snapshot under ``{data_root}/dr3/external_catalogs/``, written
  by ``scripts/fetch_shahaf2023b_class3.py`` with a ``meta.yaml`` SHA-256 that is verified on load.
  Only reproduction-mode subsamples declaring ``external_catalog: shahaf2023b_class3`` read it:
  El-Badry 2026 ``sub_chandrasekhar``, #315. ``dr4``: ``null``. Configured-but-missing raises.) Configured-but-missing and duplicate enrichment join keys raise. The funnel reports
  ``nss_enrichment_{enabled,rows_matched,rows_unmatched}``; the HDF5 ``meta`` records the enrichment
  ``snapshot_id`` and checksum. Measured on the laptop: 351,268 / 351,268 rows matched,
  covariance ``ok`` 338,141, ``unsupported_solution_type`` 11,287 (all ``EclipsingBinary``, Q4),
  ``unpack_failed`` 1,840 (1,703 ``Orbital``, 137 ``AstroSpectroSB1``).
- **Diagnostics:** funnel table (incl. covariance health), sky-coverage map, RUWE/period/eccentricity
  histograms.

### `mass_derivation_bulk`

- Primary mass from **MSC** (`teff_msc1`, `logg_msc1`, `mh_msc`) when available, **gspphot** as
  fallback (single-star-assuming; treated as a reasonable estimate for genuine candidates without
  added error inflation — see §3), transformed via the **Torres, Andersen & Giménez (2010,
  "TAG10")** calibration by default (replaces `mtgr` and Eker et al. entirely). The mass-calibration
  **method is a config choice** (`TAG10` in v1; other names may be reserved to raise
  "not implemented"). TAG10 Table 1 coefficients are named constants in `constants.py`; the
  published intrinsic scatter `sigma_logM` (0.027 dex) and the Santos correction on/off flag are
  **config** (Santos applied by default):

  ```
  log M = a1 + a2*X + a3*X^2 + a4*X^3 + a5*(log g)^2 + a6*(log g)^3 + a7*[Fe/H]
  log R = b1 + b2*X + b3*X^2 + b4*X^3 + b5*(log g)^2 + b6*(log g)^3 + b7*[Fe/H]
  ```
  (X = log Teff − 4.1; coefficients from Table 1 of Torres, Andersen & Giménez 2010, named
  constants, never inline). `log R` from this fit is the bulk-tier radius estimate everywhere
  radius is needed, until superseded by uberMS.
  Uncertainty: exact analytic partial derivatives of the polynomial, combined in quadrature with
  the config `sigma_logM`.
  **Santos et al. (2013) correction** (coefficients named constants; enable via config, default
  on): `M_corrected = 0.791·M_TAG10² − 0.575·M_TAG10 + 0.701`.
- **Isochrone M1 (#418; a design change, behind a switch that is off by default).** Ryan decided
  (2026-10-03) that M1 comes from MIST isochrones on the dereddened CMD, for the mock parent and
  the data side alike (`docs/MOCK_POPULATION_SPEC.md` §0.3, §11). The switch is a new value,
  `mass_calibration.method: MIST_isochrone`. The default stays `TAG10` until Ryan flips it after
  reviewing the re-measurements in `docs/gate418/`. Under the switch:
  - **Model.** MIST v1.2 (v/v_crit 0.4) full isochrones with UBVRIplus Gaia (EDR3 = DR3)
    bolometric corrections, from `isochrone_mass.mist_root`. That path is host-specific (host
    profiles) and is read in place; the parsed grid is cached under `<data_root>/isochrone_mist/`
    with a checksum. Priors are config: Kroupa (2001) IMF, constant SFR over 0.1–14.1 Gyr, and a
    Gaussian [Fe/H] prior (provisional). Spec §11.2.
  - **Inputs per candidate.** G, BP−RP and (l, b). The distance is Bailer-Jones geometric when
    present, else 1 / ϖ_NSS. Extinction is Combined19 E(B−V) with the Babusiaux et al. (2018)
    Gaia law, the same code as the parent (`giants.cmd_for_rows`).
  - **Output.** The `M1` ParameterSet holds the posterior M1 point and its posterior σ;
    `R1 = 10^⟨log10 R⟩`. Provenance is `isochrone_mist_v1.2`. The extras record ⟨log10 M1⟩,
    σ_log M1, the initial mass, the maximum past radius, P(evolved), ⟨log age⟩, ⟨[Fe/H]⟩ and the
    prior-predictive density.
  - **Skips.** A candidate with no CMD (`no_cmd`) or off the isochrone grid (`m1_off_grid`) is
    skipped and counted in the funnel. It is never extrapolated and never given a fallback mass.
  - **Flip (Ryan, 2026-10-04):** the default becomes `MIST_isochrone` once #425 is fixed. The fix
    has two parts. `sample_selection` enrichment writes only `pipeline_*` columns, so literature
    reproduction columns are never overwritten. The Andrews forward-model `pipeline_tag10_bulk`
    calls TAG10 explicitly. See MOCK_POPULATION_SPEC §11.9.
  - **TAG10 is untouched**, and it remains the default path. #393 (Santos floor) and #380 (Gaussian
    M1 draws) still apply to it. #323 (TAG10 outside its calibration range) is resolved for the
    switch path, where out-of-range is `m1_off_grid`.
  - **Literature reproduction paths keep their own M1** (column ownership). Their counts must be
    unchanged under the switch. Forward-model paths that read the pipeline's bulk M1 change, and
    their counts are reported.
  - **Blended light.** The fit assumes a single star, so a luminous companion biases M1. The mock
    applies the same estimator (MOCK_POPULATION_SPEC §11.3), so the M1 → M2 mapping matches between
    the mock and the data.
- **Companion mass and its uncertainty (#374).** The central `M2` is the `gaiamock_mod`
  `get_companion_mass_from_mass_function` inversion at the best-fit Thiele–Innes `a0`, period
  and parallax, with the TAG10 `M1` and `mass_derivation.dark_companion_flux_ratio`.
  **`sigma_M2` propagates the full NSS covariance plus the `M1` uncertainty.** The method is a
  Monte Carlo that reuses `mc_mass_function.propagate_nss_solution`, the same code the selection
  samples use, rather than a second implementation:
  - Draws come from the record's full `nss_solution` covariance: A/B/F/G, parallax, period and
    every cross term (12×12 for `Orbital`). The factorization is the shared
    Cholesky → nugget → eigen-clip sequence.
  - `M1` is drawn per draw from `N(M1, sigma_M1)` (the v1 Gaussian-`M1` convention). Its stream
    is decorrelated from the covariance draws by `mc_mass_function.M1_DRAW_SEED_SALT`.
  - Each draw goes through `a0 → f(m) → M2`. `sigma_M2` is the ensemble standard deviation over
    finite draws, which is the `mc_mass_function` convention (`MassFunctionDraws.m2_std`).
  - A draw with a non-physical input (`M1 <= 0`, parallax `<= 0`) inverts to NaN and is left out
    of that standard deviation. It is counted in `n_valid`, never replaced.
  - `n_draws`, `random_seed` and the eigen floors come from the shared `mc_mass_function` config
    section, which is part of this stage's fingerprint. The per-system seed is
    `random_seed XOR (source_id & 0x7FFFFFFF)`, the same derivation as the selection-sample MC.
  - `CandidateRecord.extras["m2_bulk_mc"]` records the ensemble summary: `n_draws`, `n_valid`,
    `random_seed`, `factorization`, `n_clipped_eigenvalues`, `n_m1_nonpositive`, the mean, and
    the 16/50/84 % quantiles.

  A first-order (Jacobian) propagation over (A, B, F, G, ϖ, P, M1) with the same covariance was
  measured and rejected. On a 1/100 sample of run `20260930-022223-672b092`, it agrees with the
  MC to about 2 % (median ratio) at `M2/sigma_M2 > 10`. Below `M2/sigma_M2 = 2` it
  underestimates the MC sigma by a median factor of about 9: `a0` is positive-definite and
  non-linear in A/B/F/G once the orbit is near zero significance. For Gaia BH1
  (DR3 4373465352415301632), the old M1-only term gave 0.168 M☉. The MC gives about 2.7 M☉
  (16–84 %: 10.8–15.9), consistent with El-Badry et al. (2023a)'s astrometry-only 12.8 ± 2.0.
- **Covariance-reconstruction failures are counted and excluded, never given a diagonal.**
  - A candidate with Thiele–Innes elements but no `nss_solution` has no defensible `sigma_M2`.
    That is an `nss_covariance` reconstruction failure in `data_acquisition`'s
    `covariance_health`. The candidate is dropped as `skipped_no_nss_covariance` and is never
    given an M1-only or diagonal sigma.
  - A candidate whose MC cannot produce a sigma is dropped as `skipped_m2_sigma_failed`. The
    causes are missing required parameters, a factorization or inversion error, or fewer than
    two finite draws.
  - Both counts are funnel rows. So are `m2_mc_cholesky_nugget` and `m2_mc_eigen_clip`: of the
    candidates that got a `sigma_M2`, how many covariances needed each fallback factorization.
- `mass_derivation_refined` passes the bulk `M2` ParameterSet through unchanged. It does not
  recompute `M2` from a refined `M1`.
- This stage's `M2` / `sigma_M2` is the pipeline's own posterior. It never writes the
  literature-reproduction columns. Those are Andrews' `p_m2_above` / `m2_msun*` and El-Badry
  2026's `m*_tilde` / `sigma_m2_astrometric`, which stay owned by each sample's own MC path.
- Cut: retain `M2 + n_sigma * sigma_M2 >= M_min` (`n_sigma=2`, `M_min=1.1 M_sun` as config
  defaults — choosable, not constants).
- **Diagnostics:** before/after counts (including the exclusion rows above), and the M2
  distribution before and after the cut.

### `mass_derivation_refined` (dark-hunter_sed integration)

- `dark-hunter_sed` (uberMS): refined M1 (age, Teff, logg, [Fe/H], extinction, v sin i, [α/Fe]
  where available) via an async queue — run once per star, cached, re-run only on new data.
  Prioritized by the information-gain diagnostic (`diagnostics`). `FitTier` metadata tracks
  `bulk_estimate` vs. `full_uberMS`; all candidates above the mass cut eventually target
  `full_uberMS` coverage.
- `dark-hunter_sed` extended for WISE and DECam-u photometry (new PRs against that repo).
- Watch-list diagnostic: uberMS's M1 prior is capped at 3 M☉ — flag any candidate approaching it.

### `rv_astrometry_gate`

`dark-hunter_rv` now has **The Joker** installed and integrated (RV fitting is no longer built
from scratch here — this stage consumes it). RV and astrometry are fit **separately**; The Joker
is not itself an MCMC but a rejection/importance sampler for the period problem, whose output can
seed further refinement. `dark-hunter_rv` is being extended for JSON summary output (replacing the
sectioned `summary.txt`) and WISE/DECam ingestion support as needed. Phase 1 / #5 conforms to the
Phase 0 `CandidateRecord` schema where possible; real breaks require asking before changing the
frozen contract.

- Orbital elements (P, e, T_periastron, K, ω) held fixed at the astrometric solution's values;
  only systemic velocity γ and jitter are free, **fit independently per RV instrument/source**.
  Test statistic: chi2/dof against the predicted RV curve, vs. a config-driven threshold.
- **SB2 handling**: if SB2 with an orbit consistent with the astrometric one, this is not a
  companion-type verdict by itself — it feeds the `companion_nature_likelihood` stage (spectral
  lines can look WD-like or ordinary-stellar) and additionally unlocks a **direct mass-ratio
  measurement that doesn't require an isochrone-based M1** — a third mass-determination channel
  alongside "astrometry + isochrone M1" and "astrometry + RV joint fit for dark companions." If
  the SB2 orbit is inconsistent with the astrometric one, the system routes to the outlier class,
  same as a failed gate.
- Documented v1 limitation: only whole-curve chi2/dof is implemented; per-RV-point outlier
  removal, and a more general robust/bad-data treatment (RV, photometry, and eventually
  astrometric epochs), are noted as future work. Astrometric epoch-level outliers require
  individual epoch measurements and are DR4-only (§6).

### `joint_orbit_fit`

Separate registered stage (same module `rv_consistency.py`), immediately after
`rv_astrometry_gate` in the default order.

- Gate passers only: orbital elements free, full simultaneous astrometry+RV fit (Joker-seeded);
  refined M1/M2 supersedes the astrometry-only value. `OrbitTier` records `joint_astrometry_rv`.
- Model (#347): one Keplerian orbit fit to the RV epochs (per-instrument γ + jitter, Gaussian
  likelihood with its normalization term) **and** the NSS astrometric solution vector (ϖ,
  Thiele–Innes A/B/F/G, plus C/H for `AstroSpectroSB1`, e, P, T_peri) through its full
  `nss_solution` covariance sub-block, with a Gaussian M1 prior from the upstream `m1`. Free:
  P, √e cos ω, √e sin ω, T_peri, Ω, i, M1, M2, ϖ. **K is derived** from a1 = a0/ϖ (the
  photocentre is taken as the primary's orbit — dark-companion hypothesis), never free.
  Candidates without an `nss_solution` covariance are skipped (`missing_nss_covariance`),
  never given a diagonal stand-in. RV epochs with `mjd < rv_consistency.rv_epoch_min_mjd`
  (upstream placeholder `mjd: 0.0` rows) are dropped and counted, in the gate as well.
- Convergence is a test on the final point, not an optimizer flag: Fisher information
  identified (eigenvalue ratio above `joint_fisher_min_eig_ratio`) and scoring decrement
  `gᵀF⁻¹g ≤ joint_fit_decrement_tol`. Covariance = F⁻¹ propagated to
  `extras["joint_orbit"]` = (P, e, T_periastron, K, ω, i, Ω, M1, M2, ϖ); `m2` is that M2
  marginal. An optimum within `joint_bound_tol` of a box edge (M1/M2 bounds, i ∈ [0, π],
  e ceiling, jitter ceiling) is a **bound hit**: flagged, counted, and the upstream
  `astrometry_only` mass is kept. Jitter at its floor is a legitimate boundary, held fixed.
  Unconverged fits likewise keep the upstream mass with `joint_orbit_fit_skip_reason`.
- Diagnostics: `joint_orbit_fit_report.txt` (converged / bound-hit / unconverged counts,
  bound-hit parameters, recovered-M2 range, priority calibrators) and figures of the recovered
  M2 distribution, joint vs upstream M2, and joint K vs the NSS-seed K.
- Gate failures: stage status `skipped` with `reason: rv_astrometry_gate_failed`. System keeps
  `astrometry_only` parameters and is scored by the population model's outlier class later — not
  silently excluded.

### `companion_nature_likelihood`

One unified module (absorbs what earlier drafts split across a separate vetting stage and a
separate WD-debiasing stage — the same underlying question through different instruments): for
every candidate, a continuous likelihood over "what is the true nature of M2" given whatever
evidence is available — broadband photometry residual (single-star vs. single+companion-of-mass-M
SED, via ΔBIC, config-driven threshold), Gaia XP spectral residual, and SB2 spectral
characteristics when present.
- **Joint** multi-band model (not independent per-band multiplication), correctly accounting for
  which evidence channels are actually available per system.
- Against Bédard et al. cooling tracks (default, config-swappable; tracks are **local files** under
  a config path, staged like other large data — not fetched at runtime in v1), 100% H (DA)
  atmosphere with He as a config option.
- Continuous, not step-function: a 5σ-expected non-detection still carries small non-zero
  probability of a real companion; a 2σ non-detection is still informative.
- Two-tier: fast approximate joint fit for the bulk pass, full joint fit queued for
  confirmed/critical systems.
- **Feeds directly into `population_model` as a per-system weight — not a pre-filter.** Nothing is
  discarded for "not looking like a compact object"; the underlying population is fit to match the
  observed, evidence-weighted population, contamination included.
- **Required diagnostic**: stratify by primary-star age bin to test the age-independence
  assumption.

### `triples` (off by default)

Distinct from the above — specifically an unrelated outer companion. Evidence: TESS photometric
variability plus a rotation-consistency check (implied v_rot vs. v sin i from uberMS). **Unbuilt
in v1**, forced to P(triple)=0 in `population_model`. Kept as its own module so it can be enabled
later without restructuring anything else.

### `selection_function_astrometric`

- `gaiamock`, vendored as a pinned commit (git submodule), modified-RUWE variant. Uses its
  automatic 5/7/9/12-parameter solution-type cascade as-is. Extinction (Bayestar default,
  swappable) feeds both SED fitting and gaiamock's simulated photometry/noise model.
- **Validation gate (must pass before trusted for science)**: reproduce El-Badry et al. (2024)'s
  six-panel comparison (P_orb, G, 1/parallax, eccentricity, astrometric mass function f_m, cos i)
  between real DR3 NSS and mock population, plus a new diagnostic comparing the fraction of mock
  sources landing in each Gaia solution-type bin against real fractions.
  - **Real side (#339)**: exactly the paper's §4 sample, `nss_solution_type` in
    `dr3.selection_function_astrometric.elbadry2024_comparison_nss_solution_types`
    (`Orbital`, `AstroSpectroSB1`; 168,065 rows), read from the **uncut** Gaia snapshot recorded
    on the `data_acquisition` artifact. No pipeline quality cut. One row set feeds all six panels.
  - **Mock side (#339)**: only realizations passing every
    `<dr>.selection_function_astrometric.orbital_solution_cuts` cut (paper Eq. 18, including
    `F2 < 25`, and Eqs. 20-22) contribute to any panel. Panel values are fitted (catalog-like),
    and mock `f_m` is `physics_utils.astrometric_mass_function(a0, parallax, P)`, the same
    function as the real side. The companion mass from
    `gaiamock.get_companion_mass_from_mass_function` is stored separately.
  - The stage artifact persists both samples (`six_panel_samples/{mock,real}`); `diagnostics`
    plots exactly those arrays. There is no reference-fixture fallback on the science path.
- **Mock population (#391)**: the mock input is the **importance-reweighted proposal set** of the
  Step 1 forward-model chain (next subsection), which replaces the `mock_population` box prior once
  #394 wires it into this stage. Until then the stage still draws from the box prior
  (`mock_population.sampling: elbadry_prior`) without the epoch model.
- No emulator in v1 — call `gaiamock` directly, profile, add an emulator only if profiling shows
  it's needed.
- **DR4 dual mode**: (a) fast — Gaia's own DR4 NSS catalog directly; (b) complete — `gaiamock`'s
  own epoch-astrometry fitting routine on raw DR4 epoch data, the identical code path used for
  mock data. Cross-validated against each other on the overlap sample.
- **Acceleration/jerk catalogs**: matched at the broad population/aggregate level only, using the
  same cascade to forward-model which systems land in which solution type. Doubles as the
  RV-follow-up target list. Tracked in a separate pending pool, promotable once resolved.
- **Multi-solution emission (§15 Q17 resolved)**: a simulated system is not restricted to emitting
  a single solution-type row. Both cross-type multiplicity (e.g. `Orbital` + `SB1` for the same
  mock source) and same-type/period-aliasing multiplicity (more than one mock `Orbital` solution
  at different periods) must be reproduced at the empirically measured real rate — see
  "Multi-solution sources" under `data_acquisition` (§4) — so real and mock catalogs are counted
  the same way when compared in the likelihood.

#### Step 1: the forward-model mock chain (#339, #391; enters this stage through #394, pending)

**Priority (Ryan).** Step 1 comes first: the mock must reproduce the **full** DR3 NSS orbit sample
(`Orbital` + `AstroSpectroSB1`, 168,065 rows, the El-Badry et al. 2024 §4 comparison set, no NS/BH
down-select) before any downstream inference is trusted (#339). Every diagnostic must be computed
from what the pipeline actually produced. No placeholder fixture, configured stand-in fraction or
self-compared benchmark may stand in for an output (#339, #344, #348, #352/#354).

This subsection is a map, not the specification. The authoritative documents are
`docs/MOCK_POPULATION_SPEC.md` (parent, companions, proposal set, weights, Malmquist, giants,
isochrone M1, ladder, every open MP-Q) and `docs/EPOCH_MODEL_SPEC.md` (epochs, transit loss,
bright-star noise, RUWE normalisation). Measurements are in the gate directories:
`docs/gate390/` (injection), `docs/gate399/` (cascade replay, σ deficit), `docs/gate400/` (epoch
model), `docs/gate405/` (1-D Malmquist closed loop), `docs/gate_giants/`, `docs/gate418/`
(isochrone M1, 2-D weight), `docs/gate391/` (rung 2, paused).

**Per draw, in order** (module → spec section):

1. **Parent** (spec §1, §0.1). Primaries are real `gaia_source` rows, snapshotted once by
   `scripts/fetch_gaia_source_parent.py`. The snapshot is a uniform `random_index < 10⁶` slice with
   G < 19 (the Halbwachs et al. 2023 §1.2 limit), measured ϖ > 0.2 mas and the Halbwachs (b)/(c)
   IPD and C\* flags. The truth parallax comes from the Bailer-Jones geometric distance, and the
   observed G is the total system light. RUWE > 1.4 and ≥ 12 visibility periods are **outcomes**
   that the cascade simulates; they are never pre-selected. The real comparison sample is
   filtered the same way (ϖ > 0.2 mas, the 114 IPD/C\* failures dropped).
2. **M1** (`isochrone_mass`, spec §11.2). MIST v1.2 isochrones are fitted to the dereddened CMD
   (Combined19 extinction). The same function serves the data side as `mass_derivation_bulk`
   under `mass_calibration.method: MIST_isochrone`, which is off until #425 (see
   `mass_derivation_bulk`). TAG10 is retired for the parent because of its 0.6 M⊙ floor (#393)
   and its factor-of-2-low giant masses.
3. **Giants** (`giants`, spec §10). A dereddened-CMD classifier flags evolved rows against the
   main-sequence ridge measured from the parent (n_σ = 3). Evolved rows get the §10.4 companion
   light relation, and giants are compared on `Orbital` only.
4. **Companions** (`moe_distefano`, spec §2). The Moe & Di Stefano (2017) densities come from the
   frozen table `config/population/moe_distefano2017.yaml`, plus `population_model`'s WD/NS/BH
   mixture. The flux ratio uses Janssens et al. (2022) (MP-Q13; MP-Q40 is open).
5. **Proposal and weights** (`proposal_set`, spec §3). One broad analytic proposal q(x) is
   simulated once. Every draw is stored with its truth, seeds (#371), cascade vector, outcome flags
   and fitted solution. Any θ is a reweighting
   w_i = (N_full/N_snap) λ(x_i|θ) W_i / Σ_j n_j q_j(x_i), with a deterministic-mixture denominator
   over every generation j. Trust is per-bin Kish ESS against the MC-noise rule
   ESS_b ≥ N_b / `mc_noise_threshold`². Anything that changes the gaiamock input is fixed at
   generation and needs re-simulation (spec §3.5): the parent cuts, the truth parallax, M1, the
   light split, the gaiamock triple, the epoch model and the cascade settings.
6. **Malmquist / Öpik weight W** (spec §9, §11.4). `malmquist` is the 1-D weight of #405, proven
   by the closed loop in `docs/gate405/`. For the decided pipeline it is superseded by
   `malmquist_cmd`, a 2-D CMD weight (#418). It subtracts the companion's light in G, BP and RP
   against the measured single-star ridge, treats extinction as its own vector, and uses the
   `mist_density_ridge_anchored` single-star density (MP-Q39). Its closed loop
   (`malmquist_cmd_closed_loop`) removes most of the bias, but the §11.5 "every pull ≤ 3"
   acceptance is **not met yet**.
7. **Epoch model** (`epoch_model.run_cascade`, `docs/EPOCH_MODEL_SPEC.md` §8). gaiamock_mod's
   GOST transit list is wrapped; gaiamock itself is never edited. The model has these parts:
   - every row inside ESA's 138 DR3 astrometry gaps is removed;
   - whole FoV transits are lost with p(G, l, b): degree 4 in G plus ℓ ≤ 2 Galactic harmonics,
     calibrated on NSS stars at RUWE = 1.4;
   - for faint stars part of that loss is time-clustered (ramp 0 → 0.5 between G = 14.5 and 16.5;
     τ = 2 d is provisional);
   - optionally, bright-star (G < 13) per-CCD excess noise with a DR3-style RUWE = UWE / u0_mock(G)
     (option N2d, calibrated and validated in #422; enabled by #426).
   `dr3.epoch_model.enabled` is true. Against #390 (`docs/gate400/`), the N_vis excess goes
   from +2 to 0, CCD observations / DR3 from 1.129 to 1.020, and the Orbital σ ratio from 0.89 to
   0.97. **#428:** the model does not restore DR3's insufficient-visibility channel. See
   `docs/EPOCH_MODEL_SPEC.md` for the measurement.
8. **Cascade** (`gaiamock_mod.fit_full_astrometric_cascade`, `forward_model.classify_cascade_result`).
   The El-Badry et al. (2024) Eq. 18 and Eqs. 20–22 cuts give `accepted_orbital`; §5.2.1 gives
   `published_acceleration`.

**Validation ladder** (spec §5; each rung gates the next):

| Rung | What | State |
|---|---|---|
| 1 | #390 injection: published DR3 orbits re-injected through gaiamock_mod (`injection_test`) | Done. Orbital acceptance 0.742, pulls σ_MAD ≈ 1 (`docs/gate390/`). #399/#398 diagnosed by bit-for-bit replay (`cascade_replay`, `docs/gate399/`), then re-validated with the epoch model (`docs/gate400/`) |
| 2 | MdS17 at published parameters, reweighted; six-panel and solution-type mix vs DR3 | **Paused** at 154,518 of 370,000 draws. The figures are diagnostic only: pre-noise-fix and pre-Malmquist (`docs/gate391/`). Restart waits for the #418 implementation (spec §11.9) and MP-Q28d |
| 3 | Fit the luminous-binary θ to the full NSS sample | not started |
| 4 | Compact-object injection–recovery (SBC) from a held-out partition | not started |
| 5 | Real inference | not started |

The closed loops (`docs/gate405/`, `docs/gate418/`) are not rungs. They gate the weight itself:
with W switched off, the mock must fail visibly.

**How it feeds `selection_function_astrometric` (#394, pending).** The stage artifact will hold
the proposal set (truth, seeds, outcome). The six-panel and solution-type gates become weighted
histograms at a stated θ, with ESS per bin. `sensitivity_analysis` reads ESS and MC noise from the
weights and triggers top-ups (spec §3.7). `inference` evaluates Σ_i w_i(θ) 1[i ∈ b] inside the
sampler, so gaiamock is never called in the likelihood. The `mock_population` box prior and its
keys are then removed. The proposal set runs today from `scripts/run_proposal_pilot.py` with
configs under `config/population/` (not `config/fragments/`, because `load_config` merges
fragments). Runs use at most the agreed worker count, under `nice`, with every BLAS/OpenMP pool
pinned to one thread through `threadpoolctl` (#408).

### `selection_function_followup`

A parametric approximation to who gets RV follow-up, driven by static factors — target-list
membership (Andrews et al., El-Badry et al., accel/jerk catalog) with real adoption dates,
declination/brightness limits, and the documented preference for cooler stars. Calibrated the same
way as the astrometric selection function: matching mock-vs-real histograms of N_observations and
follow-up time span, not by mechanistically modeling the adaptive stopping rule.
- **Data-source tiering, confirmed in v1 scope**: major spectroscopic surveys (APOGEE, RAVE,
  LAMOST, DESI) have documented, usable selection functions to look up and apply. Ad hoc
  literature RVs with unknown selection functions are approximated via brightness, declination,
  and proper motion. Other dedicated follow-up campaigns are handled per-source — well-documented
  ones (e.g. El-Badry et al., including dropped-vs.-still-observing systems) incorporated directly;
  harder-to-parse ones handled individually as feasible. Approximate selection functions for these
  are in v1 scope.
- **Target-list adoption dates**: reconstructed from the observing-log Google Sheet's revision
  history via the Drive API (`revisions.list` + per-revision export, diffed) — with the documented
  caveat that Google's own API docs note this listing can be incomplete for a long-lived,
  frequently-edited sheet; spot-check against the UI's version-history panel where it matters.
  Going forward, this stage also takes a **weekly snapshot** of the sheet's current state into this
  project's own archive, so future reconstruction doesn't depend on Google's revision retention.
- **Multi-solution emission (§15 Q17 resolved)**: applies here identically to
  `selection_function_astrometric` above — a mock system's follow-up-triggered rows must reproduce
  the real multi-solution emission rate (cross-type and same-type/period-aliasing) rather than
  assuming one row per system.

### `population_model`

Hierarchical multiplicity → type mixture:

1. **Multiplicity layer**: single / binary / triple. `P(single) = 0`, `P(triple) = 0` in v1 (both
   documented limitations, §9). v1 assumes everything in the NSS catalog is a binary. This is a
   genuine joint generative rate (a sum over multiplicity branches, each forward-modeled through
   `gaiamock`, feeding one Poisson likelihood) — not a hard pre-classification, since true
   multiplicity is latent. Kept generic enough that the triple branch can later carry its own
   internal type-mixture (topology (1+2)+3, any component combination, outer-period-dominated
   detection given Gaia's cadence) without a structural rewrite.
2. **Type layer, within binary**: five classes — **BH, NS, WD, other, outlier**.
   - Every class is a rate function `rate_k(M2, [covariates] | θ)`; per-object class
     *probabilities* are posterior responsibilities, not independently estimated then inverted.
     Mass is always included; additional covariates are included per class only where the unified
     `sensitivity_analysis` stage shows they matter (e.g. P(WD) may improve with RUWE included).
   - **"other"** = a genuinely luminous, non-degenerate secondary — a hot/big WD visible in the
     blue is still WD, resolved by `companion_nature_likelihood`, not "other."
   - **WD**: hard-truncated at `M_Ch` (rotational-support/composition edge cases documented,
     ignored in v1) — no single WD may exceed it.
   - **NS**: soft/marginalized truncation at `M_TOV` (real EOS uncertainty; nuisance parameter with
     its own prior). No NS above `M_TOV` — must be BH, or (once built) resolved by the triples
     module, since two WDs in a triple *can* exceed `M_Ch` in aggregate apparent mass. Real-data
     anchors: most massive Gaia NS-candidate ≈1.9 M☉, least massive Gaia BH-candidate ≈9 M☉.
   - **outlier**: catalog-level solutions dramatically inconsistent with independent data (the
     `rv_astrometry_gate`), regardless of underlying cause. Rate depends on mass (mandatory) plus
     whichever diagnostic covariates `sensitivity_analysis` shows matter (RUWE is the known
     candidate, not hand-picked). `M_Ch`/`M_TOV` truncation does **not** apply — its apparent M2
     doesn't describe a real star.
   - Two-tier output: (1) raw total compact-object dN/dM, classification-independent, no M_TOV
     assumption; (2) species-classified dN/dM with M_TOV marginalized.
   - **Hard rule**: pulsar mass function, LIGO BH mass function, or any other external
     compact-object population is never used as a prior — comparison-only, always.
- **Non-parametric compact-object mass function**: free-height bins, edges fixed before looking at
  real detections (from a fiducial population's expected-detection density), roughly log-mass
  scaled. GP-on-log(dN/dM) built and compared as a second, swappable model. Auxiliary distributions
  (M1, P, e) stay parametric, swappable families (Kroupa IMF / Moe & Di Stefano / flat-in-log-P /
  optional SN-kick-informed eccentricity — a named model-comparison hypothesis in `inference`).
  Since #391, M1 is not drawn from an IMF: it comes from the real parent star (TAG10). The
  luminous-companion (P, q, e) densities are Moe & Di Stefano (2017) Eqs. 2–23, with the published
  coefficients as config defaults (`population_model.moe_distefano`) that become free parameters
  at validation rung 3; `population_model` evaluates λ(x|θ) on the stored proposal-set truth
  (`docs/MOCK_POPULATION_SPEC.md` §2–§3). Open choices are MP-Q1–Q23 there.

### `sensitivity_analysis`

One shared module serving two purposes: (a) whether the overall population model needs the full
joint N-D treatment or collapses to 1D dN/dM; (b) whether a given type-mixture class benefits from
covariates beyond mass. Run as its own stage before finalizing either inference dimensionality or
per-class covariate sets. Mock injection volume enforces
`sigma_MC / sigma_Poisson < mc_noise_threshold` (config default `0.1`), verified by a required
convergence diagnostic.

### `inference`

- **v1 strategy: staged-but-connected**, not fully joint. `rv_astrometry_gate` and
  `companion_nature_likelihood` results are computed once, outside the sampler, used as fixed
  per-system weights — an empirical-Bayes-style plug-in, not a re-sampled joint treatment.
  **v2 (documented, not built now)**: fully joint, warm-started from the v1 result.
- **Likelihood**: inhomogeneous Poisson point process, `rate(θ) = population_model(θ) ×
  astrometric_selection_function × followup_selection_function`.
- Binned vs. unbinned decided by `sensitivity_analysis`; given the small-N regime, prefer
  unbinned/per-object over N-D histograms if a joint treatment is needed.
- **Sampler**: `dynesty`. Reproducibility = documented multi-run robustness protocol, not bitwise
  determinism.
- **Small-N handling, applied generically**: any stratified/binned sub-analysis gets an automatic
  posterior-vs-prior overlap check; zero-count bins get explicit Poisson upper limits.
- **Model comparison** (v1 deliverable): SN-kick-informed eccentricity/mass dependence vs. none;
  "circular implies WD" vs. no such dependence.

### `diagnostics`

- `plotting.py` provides shared rendering primitives; `diagnostics.py` decides what to check and
  calls them, while paper-ready product figures (dN/dM total with waterfall classification, dN/dM
  per class overplotted, WD contamination vs. mass, etc.) call the same primitives from a separate
  "what's a deliverable" path.
- Run manifest (§5) auto-emitted/updated every run.
- Simulation-based calibration: multiple distinct injected mass functions, full recovery,
  credible-interval coverage checked across repeated injections.
- Known-truth benchmarks: Gaia-BH1, Gaia-BH2 (clean detections); Gaia-BH3 (marginal/non-detection
  in DR3 mode, RUWE=3.4, would have appeared in the acceleration catalog for parts of its orbit).
  **Checked against this run's real outputs (#348)**: NSS membership, solution type and RUWE are read
  from the `data_acquisition` artifact, and M2 from every stage in `benchmarks.mass_check_stages`
  (bulk, refined, joint). Each M2 is compared with the published mass in the fixture
  (`published_m2_msun` ± `published_m2_sigma_msun`, fixture schema v2) within
  `benchmarks.mass_check_n_sigma` combined sigmas. Each check is `passed` / `failed` / `not_tested`.
  A check with no data is `not_tested`, never passed. A stage that **did not compute** a system's
  M2 is also `not_tested`, with the reason, not credited with the upstream value it passed through
  (#382). It is detected by the stage's own `extras["<stage>_skip_reason"]` (e.g. `joint_orbit_fit`
  with no RVs → `rv_astrometry_gate_failed`), or by an `m2` ParameterSet identical to the previous
  checked stage's for the same NSS row (`mass_derivation_refined` updates M1 only, so its M2 is
  always the bulk M2). `synthetic_observed_from_truth` is reachable
  from tests only. The comparison-catalog report says it is a fixture listing with no comparison
  computed, and marks each catalog `complete` / `incomplete (n of N)` / `empty`.
- **Empty is not a pass (#334)**: a diagnostic whose input is empty reports `not_tested` /
  `insufficient_data`, never `ok`. The age-stratified WD check needs at least two populated age bins.
  No upstream stage supplies primary ages yet (tracked separately).
- **Analytic / synthetic checks are labeled as such**: the MC-noise convergence check is an analytic
  identity on configured counts (#349). `sensitivity_analysis` runs on a synthetic fiducial catalog
  in stage runs (#350). Follow-up calibration without a real catalog is `not_calibrated` (#351). SBC
  with `recovery_backend: analytic_binned` is an analytic sanity check that does not validate
  inference (#356). Their reports, figure titles and the diagnostics-stage "check verdicts" block
  say so, and none of them is reported as a validated pass.
- Cross-validation catalogs (comparison only): El-Badry/Rix/Latham/Shahaf/Mazeh et al.'s 21-system
  NS-candidate catalog (itself reporting NS candidates more eccentric than typical WD+MS binaries
  — direct support for the SN-kick hypothesis); the 156-companions astrometry+RV validation paper;
  `gaiadr3.binary_masses` AMRF classification; Andrews et al.; Shahaf et al.
- Comparison-only, always-caveated: pulsar mass function (radio-selection-biased); LIGO BH mass
  function (different formation channel, though possibly similar to the first-formed object in a
  wide binary's history).
- Other required diagnostics: fit-tier coverage map, age-stratified WD-debiasing check,
  with/without-flagged-triples robustness comparison (once triples exists), information-gain/
  follow-up-priority report (per-system and population-level), sampler multi-run consistency
  check, mock-injection Poisson-negligibility convergence plot, gaiamock solution-type-fraction
  validation, RV chi2/dof gate pass-rate diagnostic.
- Diagnostic reports and plot captions are full-detail (exempted from caveman-mode compression per
  the project skill).

#### Per-stage reports (issue #331)

Every stage below writes a human-readable report beside its HDF5 artifact, under
`{artifact_stem}_diagnostics/{diagnostics.reports_subdir}/`, behind `diagnostics.write_reports`.
Before #331 these five formatters existed but were never called, so their results lived only in
HDF5 attrs (which is how a failed astrometric validation gate went unnoticed).

| Stage | Report | Notes |
|---|---|---|
| `selection_function_astrometric` | `validation_gate_report.txt` | Header states `VALIDATION GATE PASSED` / `FAILED`; written by the `pipeline` wrapper |
| `selection_function_followup` | `followup_calibration_report.txt` | Header states `calibration_status` (`not_calibrated` with no real follow-up catalog, #351) |
| `population_model` | `population_model_report.txt` | |
| `sensitivity_analysis` | `sensitivity_analysis_report.txt` | |
| `inference` | `inference_report.txt` | Leads with the science-validity verdict (§5 "Science validity") and the SF values used, with their source |

The `diagnostics` stage's own `diagnostics_stage.txt` leads with the same science-validity verdict
and lists the stand-ins the suite itself took.

## 5. Run management

Every stage above is registered under its canonical name (the names used in §4's headers —
`data_acquisition`, `mass_derivation_bulk`, `joint_orbit_fit`, etc. — are the actual identifiers
used in code and config, never a bare `stage1`/`stage2`). Registry keys are stage names; a stage→
module map is explicit (e.g. both mass-derivation stages → `mass_derivation.py`; both selection
stages → `forward_model.py`; `rv_astrometry_gate` and `joint_orbit_fit` → `rv_consistency.py`).
Each stage declares `inputs_from: [...]` for API-contract tests (§10).

**Active DR mode**: config default `dr3`. DR4 keys exist and must be independently configured, but
DR4 execution is not enabled yet.

**Caching**: each stage declares its expected output artifact path(s) — **one HDF5 per stage** —
parameterized by the config subset that actually affects that stage's result. Before running, the
main program checks whether that exact output already exists; if so, it skips. A different config
subset yields a different path (no silent overwrite / false cache hit). Cache paths and artifact
details are recorded in the run file so resume can resolve inputs.

**Cache validation (`run_management.plan_stage`)**: a `completed` / `cached` stage record whose
artifact exists on disk is honored as `SKIP_CACHED` **only** when both of the following still hold
against the *current* working tree and config:

1. the record's `source_hash` equals `compute_source_hash(spec)` for the stage's declared
   `dependency_modules` — a record carrying **no** `source_hash` fails this check, and
2. the record's artifact **file name** — the config-subset fingerprint — equals the one
   `stage_artifact_path` produces now.

Otherwise the plan entry is `StageAction.REFUSE_STALE`: the artifact is neither reused nor silently
re-run. The run plan still prints (so `--dry-run` reports exactly which stage is stale and why),
and `pipeline.execute_plan` validates the whole plan through `assert_plan_not_stale` **before any
stage executes**, raising `StaleStageCacheError`. This matches the refusal convention already used
for a config-checksum or gaiamock-version mismatch. The operator's escape hatch is an explicit
`--force-rerun <stage>`, which bypasses the cache check and (for a completed stage) starts a new
run file.

**Stage runners use `run_management.plan_and_guard`, never a bare `plan_stage`** (issues #167,
#172). A runner invoked directly — outside `pipeline.execute_plan`, as tests and recovery scripts
do — gets the identical contract: `REFUSE_STALE` raises `StaleStageCacheError` before any manifest
mutation or artifact write, `SKIP_CACHED` returns the manifest untouched, `SKIP_REASON` records
`running` then `skipped` (plan detail as the reason, no artifact) on the run file, and only `RUN`
proceeds into science. An unrecognized `StageAction` raises `ValueError` instead of defaulting to
"run", so a future action cannot silently reintroduce the fallthrough. `plan_and_guard` returns a
`StageGuardOutcome` (`plan`, `manifest`, `proceed`); the runner still owns `mark_stage_started`,
the artifact write, and `mark_stage_finished`. The plan-printing path (`build_stage_plan` /
`format_run_plan`, used by `--dry-run`) keeps calling `plan_stage` directly, so it reports a stale
stage without raising.

Only the artifact *file name* is compared, never the full path: `new_run_for_force_rerun` copies
prior stage records forward still pointing into the **parent** run's artifact directory, which is
correct, not stale.

**A record with no `source_hash` at all is refused, exactly like a mismatched hash** (issue #169,
Ryan's decision of 2026-09-14). It cannot be shown to match the current dependency modules, so
check 1 fails with `source_hash recorded=<missing> current=<hash>`, the plan entry is
`REFUSE_STALE`, and the remedy is the same explicit `--force-rerun <stage>`. Consequence, accepted
deliberately: run files written before `source_hash` recording are unresumable without that flag.
This population only shrinks — every freshly executed stage records a real hash
(`mark_stage_started` computes it at stage start; `mark_stage_finished` falls back to computing it
at completion), and `StageRecord` is never constructed anywhere else.

**Per-stage source hash**: each stage declares which package modules (and vendored pins) affect its
answers. At stage completion the run file records `source_hash` for that dependency set. Only
**that stage's** current hash is checked — never upstream stages — both on the cached path (above)
and, via `assert_stage_source_hash`, at stage start on the execute path. Reproducibility is the
tuple `(stage_name, source_hash_at_run, config_subset, artifact_path)` per stage. Docstring /
plotting / display-only modules are omitted from dependency lists so they do not spuriously
invalidate science stages. Apart from that cross-cutting infrastructure set (`run_management`,
`schemas`, `config_schema`, `config_loader`, `plotting`; `constants` is **not** exempt, since it
holds the TAG10/Santos tables and `M_Ch`), a stage's
`dependency_modules` must be **closed under first-party module-scope imports** and must also
declare every function-level (lazy) first-party import that executes in the stage, unless an
explicit, reasoned allowlist entry says otherwise; `tests/test_stage_dependency_modules.py`
enforces both (#183). No stage module imports the `diagnostics` stage at module scope — shared
early-stage hooks live in `diagnostic_hooks.py` (#182). Human judgment owns whether upstream stages must be force-re-run after
upstream code changes.

**Config checksum**: computed over the **active DR subtree + shared physics/population keys**
only (not the inactive DR subtree). Mismatch on resume/amend → hard refuse; start a new run.

**gaiamock version checks**: when a stage depends on gaiamock, recorded
`gaiamock_mod_release` / `gaiamock_mod_sha256` / `gaiamock_git_commit` must match config; else refuse.

**Force re-run** of an already-completed stage → **always a new run file**. Prior stages' completion
records and artifact paths are **copied** into the new file so later stages can still resolve
inputs. Mid-stage crash (partial outputs, no completion record) → wipe that stage's partial
artifacts and re-run it, **amending** the same run file. Two mechanisms enforce that rather than
leaving it to the operator (#221): `data_acquisition` writes its HDF5 through a `.partial` sibling
renamed into place only on success, so no truncated file ever occupies the artifact path; and
`plan_stage` treats a stage record left `running` or `failed` with a file at the artifact path as a
**re-run**, never a cache hit, whatever its recorded `source_hash`.

**Amend vs new run** (locked):

| Situation | Action |
|---|---|
| Stop between stages; resume at the next incomplete stage; config checksum OK; stage hash OK | **Amend** same run file |
| Force-re-run of a completed stage | **New** run file (copy prior stage records) |
| Config checksum mismatch | **Refuse**; new run required |
| Cached stage record whose `source_hash` or artifact fingerprint no longer matches current code/config | **Refuse** (`REFUSE_STALE` → `StaleStageCacheError`); require explicit `--force-rerun <stage>` |
| Cached stage record with **no recorded** `source_hash` (pre-dates hash recording) | **Refuse**, identically to a mismatch (#169); require explicit `--force-rerun <stage>` |
| gaiamock version mismatch (gaiamock-using stage) | **Refuse** |
| Mid-stage crash (record left `running` / `failed`, file present at the artifact path) | Wipe partials; **amend**; re-run that stage — `plan_stage` returns `RUN`, never `SKIP_CACHED` |
| Docstring / plotting-only edits | Ignored (not in stage dependency hash) |

**The run file** is a YAML document under `runs/`, filename `runs/{run_id}.yaml` where
`run_id = YYYYMMDD-HHMMSS-<shortgit>`. It is built incrementally and *is* the `RunManifest`.
Required fields (minimum): `run_id`, `created_at`, `parent_run_id` (nullable; set when copying
forward from a force-re-run), `config_checksum`, `active_dr_mode`, `artifact_root`, gaiamock
version triple; per stage: `status`, `started_at`, `finished_at`, `source_hash`, `config_subset`,
`artifact_path`, `code_commit`, `force_rerun`, optional `reason` (e.g. skipped).

**Dry-run labeling** (issue #201). Three further manifest fields exist so a run built on
substitutions can never be mistaken for a science result:

| Field | Meaning |
|---|---|
| `dry_run` | `true` when the run used documented substitutions instead of real inputs. Present in every run file, so a dry run is distinguishable by grepping rather than by reading. |
| `dry_run_label` | The banner carried by every report and figure caption built from the run. |
| `synthetic_stand_ins` | One `SyntheticStandIn` per substitution: `name`, `stage`, `kind`, `replaces`, `description`, the `config_keys` holding the placeholder values, and those values as resolved at run time. |

**Point-of-use stand-ins** (issue #354). The pre-execution `synthetic_stand_ins` list above is a
hand-maintained declaration (the run plan must name stand-ins before anything runs). It is **not**
the inventory. Each stage that takes a synthetic or placeholder path registers it in its own HDF5
artifact at the point of use, as a JSON list in the root attribute `stand_ins_json`
(`run_validity.write_stand_ins`), with the values it actually used. Registering stages
(`run_validity.STAND_IN_REGISTERING_STAGES`) always write the attribute, possibly empty;
an artifact of one of them without it predates registration and is reported as unregistered, never as
clean. After a dry run, `dry_run.resolve_run_stand_ins` collects every artifact's registrations
(`run_validity.collect_stand_ins`); they replace any same-named declaration, and the merged list is
written back to the run file, the dry-run report and the dN/dM caption.

**Science validity** (issue #352). Two further manifest fields, `science_valid` (`null` until
assessed) and `science_validity_reasons`, are set by `inference`. It reads every upstream
validation gate before consuming what that gate guards: `selection_function_astrometric`'s
`validation_gate_passed` and `selection_function_followup`'s `calibration_status` (`passed` /
`failed` / `not_calibrated`; a missing artifact is `not_run`, an unreadable one `unreadable`).
`inference.upstream_gate_policy` decides what a gate that did not pass does:

| Policy | Effect |
|---|---|
| `mark_not_science_valid` (default) | Inference runs; artifact attr `science_valid`, the report header and the run file all say `science_valid: False` with every reason. |
| `refuse` | Inference raises `UpstreamGateFailedError` before computing anything, records the stage `failed` with the reason, and sets `science_valid: False`. |

Neither policy, and no other configuration, yields `science_valid: True` on a failed, missing or
unreadable gate. `science_valid` is True only when every gate passed, **no** stand-in was
registered by any stage up to and including `inference`, and in-stage checks passed (the
posterior-vs-prior overlap must be `passed`: `prior_dominated` and `collapsed`, a posterior whose
width ratio falls below `inference.posterior_collapse_width_ratio_floor` on any parameter, are
failures). The Q1 sample-overlap matrix is built from each sample's `inference_source_ids` in the
`sample_selection` artifact; with none available it is reported `not_computed` — there is no toy
default.

`RunManifest` **refuses** `dry_run: true` without both a label and at least one declared
stand-in, so the label is applied at birth or not at all. `format_run_plan` leads with the banner
and prints every stand-in before the stage list. `new_run_for_force_rerun` carries all three
forward: a child of a dry run is still a dry run. `create_run_manifest` also stamps
`random_seeds` — accounting only, since `dynesty` is validated by multi-run posterior agreement
and never by bitwise seed replay.

**Per-stage cost** (optional; issue #201, EXECUTION_PLAN.md §5.6). `StageRecord` carries
`wall_clock_seconds`, `rss_high_water_bytes` (the `getrusage` process-lifetime high-water mark
at that stage's end — the figure a concurrency budget must hold) and `rss_increase_bytes` (the
increase in that mark across the stage). Because the mark is monotonic, a stage peaking below an
earlier stage's peak reports `0` added; that is information, not a failure, and these numbers
cannot rank the standalone cost of later stages. All three stay `null` for cached and skipped
stages and for any stage run without a monitor attached: measurement is applied by the caller via
`run_management.record_stage_resources`, so no stage runner knows it is measured.

The monitor is deliberately **fork-free**. An earlier version sampled `ps` from a background
thread; on macOS that is the fork-in-a-threaded-process hazard, and it deadlocked a real run
mid-stage. A measurement harness must not be able to hang the thing it measures.

**Dry-run entry point**: `scripts/run_dry_run.py` (module `darkhunter_pop.dry_run`) wraps
`run_pipeline` with the declarations, the measurement and the product figure. It replays the
only pristine local Gaia snapshot rather than querying the archive, refusing to guess when several are staged — derived `+enrich`
caches and the `nss_enrichment` working directory are excluded from discovery, since replaying
either as the parent query would under-count the parent.

**Run selection**:
- If ≥1 **incomplete** run exists under `runs/` and `--run-file` is omitted: print a table of
  incomplete runs (`run_id`, status, last completed stage, created_at, config checksum short,
  artifact_root) and exit nonzero. Require `--run-file`.
- If zero incomplete runs and `--run-file` omitted: create a new run.
- Selection among runs uses the **run_id timestamp inside the file**, never filesystem mtime.
- `scripts/purge_run.py`: default deletes the run YAML only; `--with-artifacts` also deletes
  the HDF5 artifacts **this run produced** — only recorded paths under
  `{artifact_root}/{this run_id}/` that no other run file in the runs directory references.
  Copied-forward (parent) artifacts, shared artifacts and paths outside `artifact_root` are
  kept and listed as "kept, owned by <run_id>" / "kept, also referenced by <run_id>" /
  "kept, outside artifact_root" (#376). The delete/keep plan is printed before anything is
  deleted; `--dry-run` prints it and deletes nothing. Refuse purging completed runs unless
  `--force`.

**Required screen output at run start**: before any stage executes, print a run plan — which run
file is used/created, and for every stage whether it will run or be skipped and why
("cached: output exists at `<path>`" / "running: output missing" / "running: force_rerun=True" /
"skipped: rv_astrometry_gate_failed" / "stale cache: … re-run with `--force-rerun <stage>`"), plus
which config values/variant each stage will use.
Per-stage start/end status is reported during execution. (Exempt from caveman compression.)

## 6. DR3 / DR4 mode matrix

| Component | DR3 | DR4 |
|---|---|---|
| Mission baseline | ~34 months | longer (independent config value) |
| Scanning law file | pinned DR3 set | pinned DR4 set |
| Orbit source | Gaia's published `Orbital`/acceleration solutions | (a) Gaia's DR4 NSS catalog **or** (b) direct epoch-astrometry refit via gaiamock's own fitter |
| RV epochs | not used | Gaia epoch RVs ingested as an RV-pipeline input source |
| Zero-points | independent pinned DR3 versions | independent pinned DR4 versions |
| SED filter selection | independent config | independent config (may exclude filters DR3 includes) |
| Quality-cut bins | independent config, N-bin | independent config, N-bin — may need different N/values |
| Astrometric epoch outliers | not possible | possible (new capability) |
| Cross-validation | — | (a) vs (b) compared on overlap sample |
| Query snapshots | `data/dr3/gaia_snapshots/` | `data/dr4/gaia_snapshots/` |

Default active mode: **`dr3`**. DR4 cannot be run yet; keys are reserved and must still be
independently present so the audit function can fire.

Physics/population parameters (M_TOV prior, IMF, cooling tracks, mass-function bin policy) are
**shared** across both paths, enforced by the audit function described alongside
`selection_function_astrometric`/`selection_function_followup`: it walks the full parameter set
and flags both unexpected divergence in shared physics parameters and (informationally) unexpected
identical values in parameters meant to be independently configured per path.

## 7. Config philosophy

Zero hardcoded physics constants, thresholds, or file paths outside `constants.py` and
`config.yaml` — including the ΔBIC threshold, the chi2/dof outlier threshold, the N-bin
goodness-of-fit cuts, and the mock-injection Poisson-noise threshold. True constants
(`astropy.constants`, `M_Ch`, TAG10 coefficient tables, Santos coefficients) live in
`constants.py`. Choosable numbers and method switches live in config.

Subagents draft modular fragments under `config/fragments/` (tracked in git) during development;
the review/integration subagent merges them into the single canonical `config.yaml` at each
checkpoint. Secrets (Gaia archive password, Google credentials) are never committed — env vars /
local files only; config holds key *names*, not values.

Target-list sheet: world-readable Google Sheet for current values; revision-history mining waits on
credentials supplied later. Weekly dumps → gitignored `data/target_lists/snapshots/`; derived
fields (e.g. `APF_added_date`) → tracked YAML/JSON under the repo.

## 8. Open items

None currently flagged. Further adjustments go through PRs that update this document first.

## 9. Documented v1 limitations (revisit list)

- Multiplicity layer: `P(single) = 0`, `P(triple) = 0` (module built, off).
- Fully joint inference deferred to v2 (staged-but-connected for v1).
- Literature RV selection functions: major surveys handled properly; ad hoc literature
  approximated via brightness/declination/proper-motion; some follow-up campaigns' true selection
  function may remain approximate despite being in v1 scope.
- RV per-point outlier removal, and a more general robust/bad-data treatment spanning RV,
  photometry, and (DR4-only) astrometric epochs: documented, not built.
- M1 measurement uncertainty treated as Gaussian even where the true posterior may be skewed.
- SPHEREx: documentation only. Lam, El-Badry & Simon (2025) analytic selection function: not used.
- No absolute Galactic rate normalization / stellar population synthesis / binary-evolution
  modeling.
- Acceleration/jerk-catalog systems matched at population level only, not individual masses.
- `ParameterSet` joint posterior samples: documented future HDF5 format only; v1 is covariance.
- Bédard cooling-track files: reserved config path; files added when needed.
- Google Drive revision-history mining: deferred until credentials provided; weekly snapshots from
  then on.

## 10. Testing and CI

| Marker | Required to merge? | Purpose |
|---|---|---|
| `unit` | yes | schemas, run_management, constants loader |
| `physics` | yes | analytic test problems (closed-form solutions) |
| `api` | yes | producer→consumer fixtures across stage I/O; registry `inputs_from` completeness |
| `gaiamock` | no | mod install + minimal RUWE smoke (needs Release assets) |
| `network` | no | Gaia, Sheet |
| `slow` | no | performance regression budgets |

Default GitHub Actions required check `tests` runs `pytest -m "unit or physics or api"` only,
target ≪ 20 minutes. Optional suites use `dorny/paths-filter` (run when relevant paths change)
and/or `workflow_dispatch` / nightly. Local full suite remains available.

Branch protection on `main`: PR required, `tests` status required, zero required reviews (sole
developer merges = approves), admin bypass allowed.
