# Repository reference (moved from CLAUDE.md under #652)

> Detail moved verbatim from `CLAUDE.md` to keep the autoloaded brief small. `docs/ARCHITECTURE.md` and `docs/FOUNDATION_INTERFACE_FREEZE.md` remain authoritative where they overlap.

### Modules

| Module | Role |
|---|---|
| `constants.py` | True physical constants only (astropy, `M_CH`, TAG10/Santos coefficient tables) |
| `schemas.py` | `ParameterSet`, `CandidateRecord`, `RunManifest`/`StageRecord`, `OutlierTestResult`, `FollowUpRecord`, `OrbitTier`, `FitTier`, `ActiveDRMode`, `StageStatus` |
| `config_schema.py` / `config_loader.py` | `PipelineConfig`; `load_config`, `config_checksum`, `assert_config_checksum`, `SHARED_CHECKSUM_SECTIONS` |
| `run_management.py` | Stage registry, artifact paths, caching, run-file I/O, `plan_stage`, `format_run_plan`, `new_run_for_force_rerun`, `purge_run` |
| `pipeline.py` / `scripts/run_pipeline.py` | Orchestration entry point (plan-then-run) |
| `gaiamock_vendor.py` | **Sole** science import path: `import_gaiamock_mod()`, `read_versions()` |
| `physics_utils.py` | Units + Poisson point-process primitives **only** (no Kepler/RUWE — gaiamock owns those) |
| `data_acquisition.py` | Gaia DR3 NSS ADQL, cross-matches, N-bin quality cut, snapshots |
| `nss_covariance.py` | `corr_vec` / `bit_index` → 12×12 `ParameterSet` (no diagonal-only fallback) |
| `mass_derivation.py` | TAG10 (+Santos) bulk masses (MIST isochrone switch, off until #425); uberMS refined queue |
| `sample_selection.py` | Literature selection registry + `SampleSelection` cut-chain evaluator |
| `elbadry2024_selection.py`, `elbadry2026_selection.py`, `janssens_mass.py` | Per-sample logic and the Janssens `M_G`–mass relation |
| `mc_mass_function.py` | 10⁴-draw Monte Carlo `(m_f, M2)` posteriors from the full covariance |
| `physics_utils.spectroscopic_mass_function` | SB1/SB1C `f_m` closed form (config section `spectroscopic_mass_function`) — reproduction/validation only, no inference entry point |
| `spuriousness_model.py` | Shared, sample-independent `P(spurious \| covariates)` |
| `sample_inclusion.py`, `sample_diagnostics.py` | Multi-sample Poisson inclusion operator; attrition/reproduction reports |
| `rv_adapter.py`, `rv_consistency.py` | RV JSON summary ingestion; chi2/dof gate; joint astrometry+RV fit |
| `phot_sed_adapter.py` | `dark-hunter_sed` Path-2 per-model JSON summaries → `phot_chi2_*` extras (BIC→chi2, provenance tagging) |
| `companion_nature.py` | Joint multi-band WD/other/dark likelihood (ΔBIC, Bédard cooling tracks) |
| `forward_model.py` | Astrometric + follow-up selection functions via gaiamock |
| `population_model.py` | Multiplicity → 5-class type mixture; non-parametric `dN/dM` |
| `sensitivity_analysis.py` | Dimensionality + per-class covariate selection; MC-noise gate |
| `inference.py` | Poisson × SF × dynesty; `inference.multi_sample` |
| `sbc.py`, `benchmarks.py`, `diagnostics.py`, `plotting.py` | SBC recovery, known-truth/comparison catalogs, diagnostics, shared figure primitives |
| `diagnostic_hooks.py` | Infra-only diagnostic primitives (dirs, reports, `funnel_sky` / `gate_pass_rate` hooks, panel/solution-type labels) used by early stages; `diagnostics` re-exports them. Stage modules never import `diagnostics` itself (#182) |
| `triples/` | Stub subpackage (`tess_variability.py`, `rotation_check.py`), off by default |
| `run_validity.py` | Stage honesty flags: stand-ins registered in the stage's own artifact at the point of use, upstream gate status, science validity (#352, #354) |
| `proposal_set.py` | Step 1 importance-reweighted proposal set: parent sampling, q(x), per-draw truth/seeds/outcome storage, deterministic-mixture weights, Kish ESS; `simulate_one` goes through `epoch_model.run_cascade` (#391) |
| `moe_distefano.py` | MdS17 companion densities (frequency, q with twin excess, e) from the frozen `config/population/moe_distefano2017.yaml` |
| `malmquist.py`, `malmquist_cmd.py` | Magnitude-limit (Öpik) weight: 1-D (#405), superseded for the decided pipeline by the 2-D dereddened-CMD weight (#418); `*_closed_loop.py` are their synthetic-universe proofs |
| `epoch_model.py` | Statistical DR3 epoch model wrapped around gaiamock's GOST list (gaps, p(G, l, b), clustered faint loss, bright-star noise, RUWE u0); `run_cascade` (#400) |
| `giants.py` | Dereddened-CMD evolved-primary classifier, ridge, evolved companion light, diagnostics (#413) |
| `isochrone_mass.py` | MIST v1.2 isochrone M1 on the dereddened CMD, mock parent and data side (`mass_calibration.method: MIST_isochrone`, off until #425) (#418) |
| `compact_objects.py` | Rung 4 (§13): compact-object proposal/target (`co_v1`), WD light from the Bédard sequences, SBC partition, selection-function table, cascade covariance + mass recovery (#552) |
| `injection_test.py` | Rung 1: re-inject published DR3 orbits through gaiamock_mod (#390) |
| `cascade_replay.py` | Bit-for-bit replay of #390 realizations; recovers what the cascade discarded (#399, #398) |

### Data flow

Gaia DR3 NSS query (+ NSS covariance) → TAG10 bulk masses → literature cut chains → uberMS refined
masses → RV/astrometry consistency gate → joint orbit fit → companion-nature weights →
forward-modeled astrometric + follow-up + per-sample selection functions → hierarchical population
model → Poisson likelihood inference (dynesty) → diagnostics.

### Literature sample-selection layer (Phase 8)

Each published compact-object sample applied its own idiosyncratic cut chain, which is itself a
selection function and must be forward-modeled:

```
rate_s(θ) = population_model(θ) × SF_astrometric × SF_followup × SF_sample_s
```

- **Dual path, per sample**: `mode: reproduction` (paper's own mass assumption, must recover the
  published N and source-ID set exactly; a regression test with scientific content, never used for
  inference) vs `mode: forward_model` (our own mass posterior; this is what feeds `inference`).
- **Each selection file owns its own `primary_mass` block** — Andrews assumes fixed `M1 = 1.0 M☉`,
  El-Badry 2024 uses IsocLum masses, El-Badry 2026 uses the Janssens `M̃1`. There is deliberately
  no global primary-mass switch.
- **Spuriousness is one shared model**, not a per-sample constant: `P(spurious | x)` in
  `spuriousness_model.py`, trained on **293 labeled sources** across four staged published tables
  (El-Badry 2023a Table E1, 2024 Table 3, 2026 Tables 7 and 8; 91 good / 164 spurious / 38 unknown).
  Unknown labels are **censored, not missing at random** — modeled jointly (Heckman-style), never
  dropped. Each paper's quoted rate is a `validation_targets:` output, never an input.

## Sibling-repo interfaces

### `dark-hunter_rv` → pop (live and healthy)

`output/Gaia_DR3_<id>_summary.json` (schema v1) is the pop-facing contract, written by
`io_utils.write_star_summary` alongside the legacy `*_summary.txt`. Keys pop uses: `gaia_metadata`,
`nss_solution_type`, `nss_orbital`, `thiele_innes`, `pipeline_epochs`, `external_rvs`, `joker_fit`
(inclination, ω, `variants`) and `joker_fit_path` → `rv_fit_reports/<stem>_joker_fit.json`. Consumed
by `rv_adapter` into `CandidateRecord.rv_summary` and by `rv_consistency`. Upstream doc:
`dark-hunter_rv/docs/RV_SUMMARY_JSON.md`, which names pop issue #31 as its parent and requires a docs
PR here (against `FOUNDATION_INTERFACE_FREEZE.md`) before any breaking field rename.
`darkhunter_rv.rv_summary_json` and `summary_paths` are on that repo's `main`.

Note the coupling: `dark-hunter_sed`'s `push_m1` writes fitted M1 **back into the RV summary JSON**.
That file is the shared per-star record across all three repos, not an RV-only artifact.

### `dark-hunter_sed` → pop (two open gaps)

Local checkout: `/Users/rfoley/darkhunter/seds/dark-hunter_sed/`, installed editable into pop's
`.venv`. Console scripts are baked at install time — if `.venv/bin/darkhunter-sed-phot` is absent,
the checkout predates `phot_sed_cli` and there are no Path-2 outputs to adapt yet. Snapshot target
in pop is `data/phot_sed/`. Mapping: `1star` = dark, `wd` = luminous normal + WD, `2star` = a
**coeval** binary → `other` (pop's `other` is broader; non-coeval luminous secondaries have no
dedicated model — a documented limitation). Bad photometry points are **not fully resolved
upstream**, and since BIC depends on chi2 and n_data, the adapter records n_data, `n_free`, `bic`,
`lnZ`, the fitted `sigma_int` and whatever cleaning settings a summary carries — today none, which
is reported per model as `cleaning_settings_absent` (#207).

| Artifact | Pop consumer |
|---|---|
| `output/sed_summaries/Gaia_DR3_<id>_sed_summary.json` — medians, credible intervals, `m1_msun` | `mass_derivation._load_sed_summary_json` → `parameterset_from_sed_summary` |
| `output/samples/Gaia_DR3_<id>_ums.fits` / `_utp.fits` — full posterior chains | **nothing** |
| `output/phot_sed/Gaia_DR3_<id>_<model>_summary.json` — dynesty **BIC** + **lnZ** + max-L params for `--model 1star \| 2star \| wd` | `phot_sed_adapter` → `companion_nature.photometry_channel` (`1star` / `2star` only — see #206) |

- **Gap 1 — the ΔBIC channel is wired; the WD leg still has no input.** `phot_sed_adapter` (#197)
  reads `<mass_derivation.phot_sed_root>/Gaia_DR3_<id>_<model>_summary.json`, maps `1star` → dark,
  `wd` → WD, `2star` → other, recovers `chi2 = BIC − k ln n` (upstream defines
  `BIC = k ln n − 2 ln L_max`) and fills `phot_chi2_*_key` / `phot_n_data_key` **only** where real
  summaries exist. Every candidate is tagged `phot_sed` or `analytic_fallback`, and the funnel
  reports the split plus the WD/dark weights both ways. Two upstream gaps remain: `--model wd`
  writes `<id>/wd/wdstar_<atm>_<ifmr>_summary.json` carrying `logevidence` only — no
  BIC/`n_data`/`n_free`, and not at the pop-facing filename (**#206**) — and since the channel needs
  all three hypotheses, candidates still fall back to the analytic `*_mg_zero_point` /
  `*_mg_mass_slope` relations; and no summary records photometry-cleaning provenance (**#207**), so
  the adapter records `cleaning_settings_absent` rather than settings. Pop reads only the JSON files;
  it never imports or runs `darkhunter_sed`.
- **Gap 2 — `ParameterSet` is being fed marginals.** `sed_summary.json` carries medians and credible
  intervals; the joint information is in the `_ums.fits` chains. Either the SED summary gains a
  covariance block upstream (preferred) or pop reads the chains. **Never substitute a diagonal.**
- **Throughput** is the real WD-contamination blocker: uberMS SVI (UMS + UTP) plus a separate dynesty
  run per Path-2 model, three models per star. Prioritize the queue; run it on ziggy.

## Data and configuration

| Path | Contents |
|---|---|
| `config/config.yaml` | Single merged canonical config (top-level: `paths`, `active_dr_mode`, `gaiamock`, `mass_calibration`, `mass_derivation`, `rv_consistency`, `companion_nature`, `classification`, `physics`, `mc_mass_function`, `selection_function_astrometric`, `selection_function_followup`, `sample_selection`, `spuriousness_model`, `sensitivity_analysis`, `population_model`, `inference`, `diagnostics`, `plotting`, `benchmarks`, `triples`, `dr3`, `dr4`) |
| `config/fragments/*.yaml` | Tracked per-domain drafts; merged into `config.yaml` by Review/Integration at checkpoints (`summary_paths.md` documents RV/SED summary wiring). **`spectroscopic_mass_function.yaml` is not yet merged into `config.yaml`** — it runs on schema defaults |
| `config/selections/*.yaml` | **Frozen, versioned** literature cut chains: `andrews2022`, `andrews2022_modified`, `elbadry2024`, `elbadry2026`, `accel_jerk` (disabled) |
| `config/selections/external/` | User-supplied published tables: Janssens 2022 mass–magnitude, El-Badry 2023a E1, 2024 T3, 2026 T7/T8 (label fixtures at `schema_version: 3`) |
| `config/spuriousness_model.yaml` | Sample-**in**dependent; deliberately not under `selections/` |
| `config/benchmarks/` | Known-truth (Gaia BH1/BH2/BH3) + comparison-only catalogs |
| `config/target_lists/derived/` | Adoption dates (Andrews, El-Badry, accel/jerk); `survey_sfs/` holds APOGEE/RAVE/LAMOST/DESI selection functions |
| `runs/{run_id}.yaml` | Live YAML run manifest (`run_id = YYYYMMDD-HHMMSS-<shortgit>`); tracked in git; *is* the `RunManifest` |
| `data/` (gitignored) | `dr3/gaia_snapshots/` (incl. `nss_enrichment/`, `+enrich` and legacy `+enrich+mc10000` caches, the Step 1 `gaia_source` parent), `reproduction_columns/` (Andrews ATF sidecars), `dr3/rv_summaries/`, `sed_summaries/` |
| `output/` (gitignored) | Per-stage HDF5 under `paths.artifact_root/{run_id}/{stage}/{fingerprint}.h5` |
| `vendor/` | `gaiamock/` submodule, `overlays/gaiamock_mod.py`, `DATA_MANIFEST.md` SHA256s; staging drop in `mod_files/` |

## Dependencies

- **`gaiamock_mod`** — vendored fork of `kareemelbadry/gaiamock` with modified (unbinned, CCD-level,
  sky-dependent) RUWE. Import **only** via `import_gaiamock_mod()`; never stock `gaiamock`, and never
  reimplement anything in `docs/GAIAMOCK_API.md`.
- Gaia DR3 archive: `gaiadr3.nss_two_body_orbit`, `*_best_neighbour` cross-matches, `binary_masses`.
  DR4 keys reserved, not runnable.
- `dark-hunter_rv` (The Joker) → `Gaia_DR3_{source_id}_summary.json`;
  `dark-hunter_sed` (uberMS) → `Gaia_DR3_{source_id}_sed_summary.json`.
- `dynesty` (sampler), `mwdust` Combined19 (one-time map download), Bédard et al. WD cooling tracks
  (local files), optional `nsstools` (Q7, undecided).
- Google Sheets / Drive API for target-list adoption dates (revision-history mining).

## Gotchas (full text)


- Science code imports gaiamock **only** via `import_gaiamock_mod()`; a version-triple mismatch
  (`gaiamock_mod_release` / `sha256` / git commit) hard-refuses at stage start.
- **Never** add a diagonal-only covariance fallback — NSS reconstruction failures are counted in
  `covariance_health` and excluded from MC-dependent samples, never silently downgraded.
- Literature-sample parent queries must run against the **uncut** Gaia snapshot
  (`20260826T234425Z_3d3f740b080c`); the quality-cut snapshot under-counts the parent.
- Andrews evaluation, in **both** modes, needs only the ATF-notebook sidecar (below) on plain
  uncut-snapshot rows. Since `andrews2022.yaml` schema v4 (PR #320) no Andrews cut reads the legacy
  `p_m2_above` MC columns of the `+enrich+mc10000` cache, which stays on disk for history only.
  Pass `membership['andrews2022']` when evaluating El-Badry 2026 so `andrews2022_import` binds.
- Column ownership is strict: Andrews owns the `andrews_atf_*` columns (including
  `andrews_atf_p_m2_above`); El-Badry 2026 owns `m1_tilde_msun` / `m2_tilde_msun` / `sigma_m2_astrometric_msun` at
  fixed Janssens `M̃1`. **Never alias Andrews' σ into El-Badry's.** Both Andrews modes read only the
  Andrews-owned `andrews_atf_*` columns (ATF notebook, #296 / #306), built by
  `scripts/build_andrews2022_atf_columns.py` into one sidecar per mode,
  `data/reproduction_columns/dr3/atf_notebook_<mode>_<fp>.h5`; without it every `atf_*` cut is N/A.
  The modes differ only in pass-2 M1 (reproduction: Lick → FLAME → uniform; forward_model: pipeline
  TAG10 → uniform). The giant cut / FLAME M1 read the notebook's VizieR I/355/paramp values
  (`logg_vizier_apsis`, `mass_flame_vizier_apsis`, snapshot via `scripts/fetch_vizier_apsis.py` at
  `dr3.vizier_apsis_snapshot_meta`), kept beside the Gaia-archive `logg_gspphot` / `mass_flame`.
- `σ_M̃2` is computed lazily at the `m2_error` cut with **fixed** Janssens `M̃1`
  (`propagate_fit_uncertainty: false`, Q12). Since #284 the reproduction path uses **analytic**
  (first-order, nsstools-style) propagation (`sigma_m2_tilde.method` in `elbadry2026.yaml`; the
  full-covariance MC is the alternative). The provenance names the method, e.g.
  `elbadry2026_analytic_full_m1_tilde_fixed`; the legacy `elbadry2026_m1_tilde_fixed` is still read.
- El-Badry 2026 `sub_chandrasekhar` in **reproduction** mode (schema_version 3, #315) is the
  Shahaf et al. (2023b) Table 2 class-III cross-match. The Eq. 5 cuts are applied to Shahaf's
  `M2min` / `e_M2min`, which gives exactly the paper's 22. The snapshot is at
  `data/dr3/external_catalogs/<dr3.shahaf2023b_class3_snapshot>/` (fetch with
  `scripts/fetch_shahaf2023b_class3.py`; the checksum is verified on load). Configured but missing
  raises. `forward_model` keeps the Janssens chain and never reads the catalog (Q2).
- The NSS enrichment job is **COMPLETED** — do not re-run `--poll-job` unless
  `nss_enrichment/meta.yaml` is missing.
- `bins="auto"` on heavy-tailed NSS distributions (RUWE, period) produces thousands of sub-pixel
  bars; cap with `diagnostics.histogram_max_bins` and drop non-finite values before binning.
- `mwdust.Combined19()` triggers a one-time network download when
  `selection_function_astrometric.extinction_model: combined19`.
- Do not host or require the default ~984 MB `healpix_scans.zip`.
- `scripts/attach_mc_to_selection_cache.py` builds the legacy `+enrich+mc10000` cache (PR #150,
  `03a452a`). No current Andrews cut reads it (PR #320), so do not rebuild it for Andrews.
- Step 1 mock work: a change to the parent cuts, truth parallax, M1, light split, epoch model,
  gaiamock triple or cascade settings invalidates every stored proposal-set draw (spec §3.5).
  Reweighting cannot fix it; the draws must be re-simulated. Run gaiamock drivers under `nice` with
  `threadpoolctl` pinning (#408). Proposal-set configs live in `config/population/`, **not**
  `config/fragments/`, which `load_config` merges.
- gaiamock_mod gives ≥ 12 visibility periods at every DR3 position sampled (0 / 6,000). With the
  epoch model 0.10% of stars fall below 12, against DR3's 1.71% (#428, #432). A test or gate that
  expects a sizeable mock `insufficient_visibility` fraction is wrong.
- Never read `dark-hunter_rv` / `dark-hunter_sed` production output directories live — snapshot with
  a timestamp (and preserve mtimes, which the gate now orders by).

## Run management

- The run file **is** the manifest: config checksum, `active_dr_mode`, artifact root, gaiamock version
  triple, random seeds, and per stage `status`/`source_hash`/`config_subset`/`artifact_path`/
  `code_commit`/`force_rerun`/`reason`.
- **Amend vs new run**: resume between stages with matching checksum+hash → amend. Force-rerun of a
  completed stage → **always a new run file** (prior stage records copied forward). Config checksum
  or gaiamock version mismatch → **refuse**. Mid-stage crash → wipe partials, amend, re-run that
  stage. Docstring/plotting-only edits are outside dependency hashes and are ignored.
- Resume checksum covers the **active DR subtree + shared physics/population sections** only;
  `diagnostics` and `benchmarks` are deliberately excluded, as are the `inference` cluster-recipe
  comments.
- If incomplete runs exist and `--run-file` is omitted, the entry point prints a table and exits
  nonzero. Run selection uses the `run_id` timestamp inside the file, never filesystem mtime.

## Git / PR workflow

1. One worktree/branch per subagent; `[AI Checkpoint] <description>` micro-commits, **pushed** and
   kept in the PR's merge commit for regression tracing (bisect with `--first-parent` for per-PR
   granularity, then descend into a suspect PR).
2. **Full local `pytest -m "unit or physics or api"` pass before opening a PR.**
3. PR via `gh`: detailed description, `Closes #N` alone on a line, test checklist, CI `tests` green.
4. One issue = one thing; prefer fine-grained issues with optional umbrellas. Track work on GitHub,
   not only in chat.
5. Review/Integration merges config fragments into `config/config.yaml` at checkpoints.
6. **PRs may auto-merge on green CI.** Correctness after the merge is owned by the standing
   verification agent (roster #49): it re-runs the suite on `main`, confirms the code actually
   runs (`--dry-run` plan plus a small stage execution), hunts silent cross-PR conflicts
   (parameter/keyword name drift, duplicate config-key spellings, schema drift, stale
   `inputs_from` / `dependency_modules`, docs describing what the code stopped doing), and
   **owns branch and worktree teardown**, and **may revert a merge** that leaves `main` broken. It
   writes no code except tests, never lands a forward fix, and escalates to the orchestrator.
   Coding agents never delete a branch or worktree. Merges are **merge commits, not squashes**.
   Auto-merge is enabled on **`dark-hunter_pop` only** — `dark-hunter_rv` and `dark-hunter_sed`
   PRs are merged by hand.
7. Branch protection on `main`: PR required, `tests` required, zero required reviews, admin bypass.
8. Keep truly-simultaneous agent sessions to **2–3**.
9. **All coding is dispatched to subagents**, one per ticket, each opening with the standard session
   preamble in `docs/EXECUTION_PLAN.md` §5.2 and an explicit model/effort pair (§5.5, §5.5a). The
   orchestrator plans, files issues, reviews and runs the wave stops; it does not edit code.
   `caveman` is active in every session.
10. **The orchestrator owns the issues**, one per ticket, filed before dispatch — though filing is
   done by a Claude Code agent on the laptop, since the orchestrator has no `gh`. Subagents may *edit* their
   issue, and must **file a new issue** for anything they find outside its scope. **Nothing is
   silently fixed in a PR** — an untracked change is bounced by #49 even when it is an improvement.
   A PR that renames a parameter, config key, signature or schema field says so under
   "Interface changes". One issue is one thing, but **one PR may close several** — list each
   `Closes #N` on its own line.
11. Concurrency is bounded by **laptop memory**, not a session count: classify each session heavy
   (runs pytest / stages / HDF5 / MC) or light (docs, config, fixture-scale), cap the heavy ones,
   budget **40 GB** of the laptop's 64 GB for all agent work, assume ~6 GB per heavy session until
   #28 measures it, start at three and raise on evidence (`EXECUTION_PLAN.md` §5.6). Subagents
   report to the orchestrator on finish; the orchestrator pings #49 to verify that merge.
12. Every wave ends in a hard stop for operator review, with a written handoff note, before the next
   begins (§5.9).
