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
│   └── diagnostics.py          # stage: diagnostics
├── vendor/gaiamock/            # git submodule @ upstream main + modified-RUWE overlay (§1.1)
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

1. Submodule `vendor/gaiamock` pinned to upstream `main` (`kareemelbadry/gaiamock`, MIT).
2. Overlay from the modified-RUWE pack (upstream README Box link; mirrored on an immutable
   GitHub Release tag, e.g. `gaiamock-mod-v1` — **do not** host the default ~984 MB
   `healpix_scans.zip`):
   - `gaiamock_mod.py` — tracked in this repo and mirrored on the Release
   - `healpix_16_med_ruwe.npz` — Release asset
   - `individual_ccds.zip` contents → `vendor/gaiamock/healpix_scans/` — Release asset
3. Compile `kepler_solve_astrometry.so` inside `vendor/gaiamock/`.

Version triple recorded in config and every run/stage record: `gaiamock_mod_release`,
`gaiamock_mod_sha256`, `gaiamock_git_commit`. Mismatch at a gaiamock-using stage start → refuse
(same class as config mismatch). Checksums live in `vendor/gaiamock/DATA_MANIFEST.md`.

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
  elements where available; the full field set from the RV pipeline's per-star summary output
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
- Quality cut: goodness-of-fit vs. magnitude, configurable as **N separate (magnitude, threshold)
  bins** (El-Badry et al. 2023's <5/G>13, <10/G≤13 as the v1 default values; the mechanism
  supports arbitrary bin counts, since DR4 may need a different scheme).
- Snapshot: raw query result + checksum + literal ADQL text + query date. Full accounting
  standard, not bitwise-exact reproducibility.
- **Diagnostics:** funnel table, sky-coverage map, RUWE/period/eccentricity histograms.

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
- Cut: retain `M2 + n_sigma * sigma_M2 >= M_min` (`n_sigma=2`, `M_min=1.1 M_sun` as config
  defaults — choosable, not constants).
- **Diagnostics:** before/after counts, M2 distribution pre/post cut.

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
- No emulator in v1 — call `gaiamock` directly, profile, add an emulator only if profiling shows
  it's needed.
- **DR4 dual mode**: (a) fast — Gaia's own DR4 NSS catalog directly; (b) complete — `gaiamock`'s
  own epoch-astrometry fitting routine on raw DR4 epoch data, the identical code path used for
  mock data. Cross-validated against each other on the overlap sample.
- **Acceleration/jerk catalogs**: matched at the broad population/aggregate level only, using the
  same cascade to forward-model which systems land in which solution type. Doubles as the
  RV-follow-up target list. Tracked in a separate pending pool, promotable once resolved.

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

**Per-stage source hash**: each stage declares which package modules (and vendored pins) affect its
answers. At stage completion the run file records `source_hash` for that dependency set. At stage
start, only **that stage's** current hash is checked — not upstream stages. Reproducibility is the
tuple `(stage_name, source_hash_at_run, config_subset, artifact_path)` per stage. Docstring /
plotting / display-only modules are omitted from dependency lists so they do not spuriously
invalidate science stages. Human judgment owns whether upstream stages must be force-re-run after
upstream code changes.

**Config checksum**: computed over the **active DR subtree + shared physics/population keys**
only (not the inactive DR subtree). Mismatch on resume/amend → hard refuse; start a new run.

**gaiamock version checks**: when a stage depends on gaiamock, recorded
`gaiamock_mod_release` / `gaiamock_mod_sha256` / `gaiamock_git_commit` must match config; else refuse.

**Force re-run** of an already-completed stage → **always a new run file**. Prior stages' completion
records and artifact paths are **copied** into the new file so later stages can still resolve
inputs. Mid-stage crash (partial outputs, no completion record) → wipe that stage's partial
artifacts and re-run it, **amending** the same run file.

**Amend vs new run** (locked):

| Situation | Action |
|---|---|
| Stop between stages; resume at the next incomplete stage; config checksum OK; stage hash OK | **Amend** same run file |
| Force-re-run of a completed stage | **New** run file (copy prior stage records) |
| Config checksum mismatch | **Refuse**; new run required |
| gaiamock version mismatch (gaiamock-using stage) | **Refuse** |
| Mid-stage crash | Wipe partials; **amend**; re-run that stage |
| Docstring / plotting-only edits | Ignored (not in stage dependency hash) |

**The run file** is a YAML document under `runs/`, filename `runs/{run_id}.yaml` where
`run_id = YYYYMMDD-HHMMSS-<shortgit>`. It is built incrementally and *is* the `RunManifest`.
Required fields (minimum): `run_id`, `created_at`, `parent_run_id` (nullable; set when copying
forward from a force-re-run), `config_checksum`, `active_dr_mode`, `artifact_root`, gaiamock
version triple; per stage: `status`, `started_at`, `finished_at`, `source_hash`, `config_subset`,
`artifact_path`, `code_commit`, `force_rerun`, optional `reason` (e.g. skipped).

**Run selection**:
- If ≥1 **incomplete** run exists under `runs/` and `--run-file` is omitted: print a table of
  incomplete runs (`run_id`, status, last completed stage, created_at, config checksum short,
  artifact_root) and exit nonzero. Require `--run-file`.
- If zero incomplete runs and `--run-file` omitted: create a new run.
- Selection among runs uses the **run_id timestamp inside the file**, never filesystem mtime.
- `scripts/purge_run.py`: default deletes the run YAML only; `--with-artifacts` also deletes
  recorded HDF5 paths; refuse purging completed runs unless `--force`.

**Required screen output at run start**: before any stage executes, print a run plan — which run
file is used/created, and for every stage whether it will run or be skipped and why
("cached: output exists at `<path>`" / "running: output missing" / "running: force_rerun=True" /
"skipped: rv_astrometry_gate_failed"), plus which config values/variant each stage will use.
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
