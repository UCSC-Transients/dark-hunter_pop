# CLAUDE.md — dark-hunter_pop

Guidance for Claude Code / Cursor agents working in this repository.

## Project

Derives a **debiased mass function `dN/dM` for compact-object companions (WD / NS / BH) in Gaia
astrometric binaries**. A population model is forward-modeled through a modified-RUWE `gaiamock`
overlay and compared to the real Gaia DR3 NSS sample with an **inhomogeneous Poisson point-process
likelihood**. Normalization is *shape/rate relative to the quality-cut NSS parent sample*, never an
absolute Galactic space density. No population-synthesis or binary-evolution layer.

Primary science interest: the **NS and BH mass functions**, including the mass gap and the
`M_Ch` / `M_TOV` boundaries.

## Documentation authority

Read these before making a design change. **Docs-first: update the spec via PR before implementing.**

| Doc | Role |
|---|---|
| `docs/ARCHITECTURE.md` | **Authoritative** pipeline specification (stages, schemas, run management, DR3/DR4 matrix, config philosophy, v1 limitations, CI) |
| `docs/FOUNDATION_INTERFACE_FREEZE.md` | Frozen import/stage/config contracts every stage builds against; per-phase landing status |
| `docs/ORCHESTRATION_PLAN.md` | Subagent roster, phase sequencing, git/PR/issue workflow |
| `docs/CONTINUATION_PLAN.md` | Phase 8+ spec: per-sample literature selection functions, spuriousness model, §14 kickoff prompts, §15 open questions |
| `docs/EXECUTION_PLAN.md` | **Forward-looking plan** — remaining work to a v1 science result and paper |
| `docs/ORCHESTRATOR_PROMPT.md` | Copy-paste prompts: the long-lived orchestrator (§A) and the verification agent (§B) |
| `docs/SELECTION_REPRODUCTION_STATUS.md` | Live hand-off: which literature reproduction numbers currently fail and why |
| `docs/GAIAMOCK_API.md` | `gaiamock_mod` public API; what must **not** be reimplemented |
| `docs/PLOTS.md` | Figure style guide + `plotting:` config defaults |
| `docs/PHASE1_KICKOFF.md` … `PHASE7_KICKOFF.md` | Historical per-phase paste prompts |

README prose is a summary, never a substitute for the locked decisions above.

## Repo

- GitHub: https://github.com/UCSC-Transients/dark-hunter_pop
- Local: `/Users/rfoley/darkhunter/pop/dark-hunter_pop/` (repo root is the base directory)
- Package: `darkhunter_pop`, `src/` layout, **one file per stage** under `src/darkhunter_pop/`
- Sibling repos: `UCSC-Transients/dark-hunter_rv` (RV / The Joker), `UCSC-Transients/dark-hunter_sed`
  (SED / uberMS / MIST / Payne). Both are flat packages; the `src/` divergence is deliberate.
  Interfaces below.
- Worktrees: one per active subagent branch, under `../dark-hunter_pop-worktrees/` and `.worktrees/`

## Architecture

Ordered, registered stages (`run_management.STAGE_ORDER`), each with `inputs_from`, one HDF5
artifact, and a `source_hash`:

`data_acquisition` → `mass_derivation_bulk` → `sample_selection` → `mass_derivation_refined` →
`rv_astrometry_gate` → `joint_orbit_fit` → `companion_nature_likelihood` → `triples` (off) →
`selection_function_astrometric` → `selection_function_followup` → `population_model` →
`sensitivity_analysis` → `inference` → `diagnostics`

Stage → module map is explicit: both mass-derivation stages → `mass_derivation.py`; both selection
functions → `forward_model.py`; `rv_astrometry_gate` + `joint_orbit_fit` → `rv_consistency.py`.

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
| `mass_derivation.py` | TAG10 (+Santos) bulk masses; uberMS refined queue |
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
| `triples/` | Stub subpackage (`tess_variability.py`, `rotation_check.py`), off by default |

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

## Compute environments

| Host | Role |
|---|---|
| **Laptop** (`/Users/rfoley/darkhunter/pop/dark-hunter_pop/`) | Workflow smoke-testing only — does the plan print, do stages wire up, does a tiny run finish. Holds a partial data set; never the production target. Sandbox HDF5/pytest segfaults are a laptop artifact, not a pipeline bug. |
| **ziggy** (`ziggy.ucolick.org`) | Holds the bulk of the data. SED production lives at `/data2/darkhunter/dark-hunter_sed/` (`scripts/activate_ziggy.sh`, daily `cron_update_sed.sh`, `cron_monthly_new_spectra.sh`). Natural home for acquisition, mass derivation, sample selection, and the RV/SED summary roots. |
| **lux** (supercomputer) | Production `inference` and the mock-injection volume for `sensitivity_analysis`. The December access deadline is about this machine. |

**Only the laptop is currently in scope**; ziggy and lux are deferred (`docs/EXECUTION_PLAN.md`
§2). The table records what they are for so the later move is cheap.

Host differences must be **config, never code** — `paths.data_root`, `paths.artifact_root`,
`mass_derivation.sed_summary_root`, `dr3.rv_summary_root` are all config keys, so a host profile is a
fragment. Record which profile a run used in its run file.

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
| `data/` (gitignored) | `dr3/gaia_snapshots/` (incl. `nss_enrichment/`, `+enrich` and `+enrich+mc10000` caches), `dr3/rv_summaries/`, `sed_summaries/` |
| `output/` (gitignored) | Per-stage HDF5 under `paths.artifact_root/{run_id}/{stage}/{fingerprint}.h5` |
| `vendor/` | `gaiamock/` submodule, `overlays/gaiamock_mod.py`, `DATA_MANIFEST.md` SHA256s; staging drop in `mod_files/` |

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot,inference]"
git submodule update --init
scripts/install_gaiamock_mod.sh            # optional overlay; needs GSL + gcc
# DOWNLOAD_FROM_RELEASE=1 scripts/install_gaiamock_mod.sh

pytest -m "unit or physics or api"         # REQUIRED merge gate (<< 20 min; CI check `tests`)
pytest -m gaiamock                         # optional; needs Release assets installed
pytest -m slow                             # optional; long suites (MC, diagnostics, perf)
pytest -m network                          # optional; Gaia archive, Sheets, catalogs

python scripts/run_pipeline.py --dry-run   # print the full plan, execute nothing
python scripts/run_pipeline.py [--config config/config.yaml] [--run-file runs/<id>.yaml]
python scripts/run_pipeline.py --force-rerun <stage> [--stages <stage> ...] [--list-stages]
python scripts/purge_run.py runs/<id>.yaml [--with-artifacts] [--force]

scripts/fetch_nss_enrichment.py            # NSS corr_vec / K1 / significance enrichment
scripts/resume_data_acquisition_from_snapshot.py
scripts/complete_stage_from_artifact.py
```

Use `.venv/bin/python` explicitly when the sandbox mangles the environment; HDF5/pytest may segfault
under restricted permissions.

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

## Conventions

- **Zero hardcoded physics constants, thresholds, or paths** outside `constants.py` and config. True
  constants (astropy, `M_Ch`, TAG10/Santos coefficients) → `constants.py`; every choosable number,
  prior, method switch, ΔBIC threshold, chi2/dof threshold, `M_MIN`, `M_TOV` prior, `sigma_logM`,
  Santos on/off, and the arbitrary-length `quality_cut_bins` list → config.
- **`ParameterSet` is the default fit-output type** — named vector + covariance + required provenance
  tag. Bare scalars only for genuinely standalone external inputs. Provenance never justifies
  inflating an uncertainty. Joint posterior samples are a documented future HDF5 format, not v1.
- **Every stage**: descriptive registered name (never `stage1`), declared `inputs_from`,
  `dependency_modules` → `source_hash`, one HDF5 artifact keyed by a config fingerprint, cache check
  before running, per-stage force-rerun override, and its own completion record appended to the run
  file. Every invocation prints the run plan before executing anything.
- **Frozen selection files**: editing a threshold in `config/selections/*.yaml` requires a
  `schema_version` bump, a `provenance` note, and human escalation. **Never retune to hit a target
  count.** Sample-level spurious rates are `validation_targets:`, never model inputs.
- **DR3/DR4 keys are always path-specific** for anything mission- or external-data-specific, even
  when the values coincide; genuine physics/population parameters (`M_TOV` prior, IMF, cooling
  tracks, bin-edge policy) are shared. Run the DR3/DR4 audit function before calling a config change
  complete.
- **Statistical guardrails**: mass-function bin edges fixed before looking at real counts;
  `sigma_MC / sigma_Poisson < physics.mc_noise_threshold` (default 0.1) with a convergence
  diagnostic; every population class is a rate function of mass plus only those covariates the
  sensitivity-analysis module justifies; `M_Ch` truncation applies to WD only, soft `M_TOV` to NS
  only, and the outlier class is exempt from both; external compact-object populations (pulsar MF,
  LIGO BH MF) are **comparison-only, never priors**; dynesty is validated by multi-run posterior
  agreement, not bitwise seed replay; small-N sub-analyses get posterior-vs-prior overlap checks and
  Poisson upper limits.
- **Plotting**: shared primitives in `plotting.py`; Okabe–Ito palette, serif fonts, thick lines,
  inward major+minor ticks on all four sides, units without slashes (`km s$^{-1}$`), finite-only
  histogram data with `diagnostics.histogram_max_bins`. See `docs/PLOTS.md`.
- **Diagnostics named in a task list are mandatory.** Unspecified extras are flagged as follow-ups,
  not silently added. Diagnostic reports, run-plan output, and plot captions are **exempt** from
  terse/caveman compression, as are convergence and gate failures.
- Skills active in every session: `strict-workflow`, `regression-hunter`, `caveman`,
  `dark-hunter-pop-workflow` (`.cursor/skills/`).

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
   preamble in `docs/EXECUTION_PLAN.md` §5.2 and a hand-picked model/effort pair (§5.5). The
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

## Status (as of 2026-09-14, `main` @ b2ce634)

- **Scope: the laptop only.** The current objective is the whole workflow running smoothly on
  `/Users/rfoley/darkhunter/pop/dark-hunter_pop/`. ziggy and lux are deferred and out of scope —
  no ticket depends on either, and the SED throughput campaign waits with them
  (`docs/EXECUTION_PLAN.md` §2). Authorized work is Waves −1 through D.
- **Wave −1 is complete** (umbrella #145, roster #50–#55). `main` now says what it is documented as
  saying:
  - **The reproduction-binding reconciliation landed** (#141 / PR #150 / `03a452a`) without reverting
    the Wave-2 RV/SED work. The three commits PR #134 never merged are on `main`.
  - **A chain of correctness fixes landed on top**, each found by #49's post-merge verification rather
    than by a test: stage-registry fingerprint (PR #155), the Andrews-σ landmine and the Janssens cache
    (PR #156), **cached stage artifacts were never checked against `source_hash`** (PR #159), missing
    transitive `dependency_modules` (PR #166), `StageAction` falling through silently in twelve stage
    runners (PRs #170, #173), and test-marker hygiene (PR #171). The `source_hash` one matters most —
    until it landed, a stale artifact could be reused as if fresh.
  - **The tree and environment are clean** (PR #179): run manifests tracked, ad-hoc logs gitignored,
    orphaned bytecode gone, `.venv` / `gh` / submodule / overlay all verified.
  - **A measured baseline exists** (#144), on `main` @ `c905575` in the primary checkout with the real
    `data/` tree and the overlay installed: required gate **467 passed, 0 skipped, 34m23s**, peak RSS
    4.56 GiB; `-m gaiamock` 4 passed; `-m slow` 6 passed; `-m network` **3 failed on Gaia TAP HTTP 500**
    (archive-side, #184 — those live parent-count checks are *not run*, not green). Peak RSS across all
    heavy workloads is **6.82 GiB** (the El-Badry 2026 `σ_M̃2` MC); the heavy-session concurrency cap is
    now **four**, set from measurement (`EXECUTION_PLAN.md` §5.6).
  - **Every `SELECTION_REPRODUCTION_STATUS.md` §3 number was re-measured on `main`** and the document
    rewritten with the SHA beside each number. No escalation signal tripped: all three exact-match
    equalities hold, nothing moved by an order of magnitude, nothing collapsed to or appeared from zero.
  - **Docs provenance corrected** (#187 / PR #188 / `b2ce634`): `scripts/attach_mc_to_selection_cache.py`
    is on `main` (it landed with the reconciliation, PR #150), not in an orphaned worktree as earlier
    docs claimed. A few more instances of the same "stale not-on-`main`" defect class remain, tracked
    under #189 rather than fixed piecemeal.
  - **Wave gate teardown**: 26 stale worktrees and 70 stale branches removed, each independently
    re-verified fully merged before deletion. One branch (`fix/selection-reproduction-binding`, fully
    superseded, patch-equivalent to what's on `main`) is left pending an explicit human
    `git branch -D` — its deletion was correctly refused by the permission system.
  - **Gate −1 awaits operator review.** Wave 0 is not dispatched until then.
- **Phases 0–8 complete on `main`.** The pipeline has run end-to-end on the real DR3 NSS catalog:
  `runs/20260904-152655-674c989.yaml` reaches `inference` completed, with `joint_orbit_fit` skipped
  (`rv_astrometry_gate_failed` for every system, pre-Wave-2), `triples` skipped by config, and
  `diagnostics` left stuck in `running` — no run has yet finished all fourteen stages cleanly. The
  most recent run (`20260908-183807-5a1c609`) is complete through `joint_orbit_fit` with live RV
  data. Every `inference` artifact so far is a CI-scale dynesty smoke, not a production posterior.
- **Wave 2 RV/SED live integration landed** (PRs #136, #137, #138): `sed_summary_root` and
  `rv_summary_root` wired, RV JSON re-attached at the gate, gate ordering by calibrators → public
  RVs → summary mtime, and gate `K` derived from Thiele–Innes inclination when NSS omits
  `Semi_Amp_Primary`. **154 `Gaia_DR3_*_summary.json` RV files are now staged** under
  `data/dr3/rv_summaries/`. The JSON writer is on `dark-hunter_rv` `main`. Only 2 SED summaries are
  staged so far.
- **Literature selection reproduction is the active blocker**, and is Wave A's work. Every number
  below was measured on `main` @ `c905575` (#144), not on a branch: Andrews N=**33** vs 24 (**352** vs
  106 after the `m2_probability` cut), El-Badry 2026 `primary_ns_bh` **42** vs 47, `sub_chandrasekhar`
  **861** vs 22, spectroscopic branch **123** vs 151, astrometric union **913** vs 76, spectro routes
  **98 / 30 / 5** vs 136 / 30 / 15, Q9 `G < 15` **19** vs 16. Green: Andrews' parent N (**134,598**),
  El-Badry 2024's catalog union (**48**), `elbadry2023_table_e1` (**5**), and `mode_divergence` (only
  Gaia BH1 differs). El-Badry 2024's published COC N of 21 was **not** re-measured — it needs a
  different path and stays recorded from the earlier measurement.
  Suspected root causes: the extinction / `a0` / AMRF / `M̃2` chain and NSS Monte Carlo binding. The
  `sub_chandrasekhar` mass window is already 1908 wide *before* the σ cut, so σ is second-order there.
  **Do not fix by retuning frozen thresholds.** Open issues #132 (NSS enrichment / K1, largely
  unblocked) and #133 (`primary_ns_bh`; extinction / `a0` / nsstools).
- **Companion-nature evidence is still analytic in practice.** The `phot_sed` adapter landed (#197):
  `companion_nature_likelihood` now uses real dynesty ΔBIC wherever all three Path-2 summaries exist,
  labels every candidate `phot_sed` vs `analytic_fallback`, and reports the split. In practice
  nothing reaches the real path yet — the `wd` model emits no readable BIC at the pop-facing path
  (#206) and only a handful of `1star`/`2star` fits exist. WD contamination stays unconstrained
  until #206 lands and the SED queue has coverage (ziggy throughput).
- A sample's reproduction path is **not working** until `sample_reproduction_report` matches the
  published N exactly; until then the sample must not be enabled in `forward_model` mode for
  inference. The spuriousness model is not working until one parameter set reproduces all three
  published rates.
- **`accel_jerk` (roster #22) is blocked** — no published selection function exists. Do **not** derive
  one from the target list. The registry entry stays disabled.
- Still local-only: `docs/ORCHESTRATOR_PROMPT.md` is untracked, and `README.md`,
  `docs/CONTINUATION_PLAN.md` and `docs/ORCHESTRATION_PLAN.md` carry uncommitted edits in the primary
  checkout (**#185**). `CLAUDE.md`, `docs/EXECUTION_PLAN.md` and
  `docs/SELECTION_REPRODUCTION_STATUS.md` landed with #144; `runs/*.yaml` landed with PR #179.
  Worktrees and finished branches are #49's to tear down — **coding sessions never delete either**.

## Open statistical questions (human sign-off)

| Q | Question | Blocks |
|---|---|---|
| Q1 | Multi-sample overlap: separate Poisson processes vs one inclusion-indicator formulation. Overlap is three-way (El-Badry 2024 drew from Andrews; El-Badry 2026 subsample 3 *is* Andrews restricted to `G < 15`), so naive summation double-counts. Signed off under #112 / PR #125 | `inference` |
| Q2 | Shahaf 2023b triage: cross-match the published 177-candidate catalog vs reimplement AMRF (a catalog cannot be applied to mocks) | El-Badry 2024 path, DR4 |
| Q4 | Whether `corr_vec`/`bit_index` suffice to reconstruct the full covariance for every solution type | NSS covariance |
| Q7 | Vendor `nsstools` (what El-Badry 2026 uses for `ã0`) vs our `thiele_innes_to_campbell`; the delta must be **measured** before choosing | El-Badry 2026 reproduction, #133 |
| Q9 | Confirm `G < 15` on our reproduced Andrews sample yields exactly 16; a mismatch is a failure of both reproductions, not a tuning opportunity | El-Badry 2026 |
| Q10 | The cut evaluator must distinguish "cut not applicable" (undefined `M̃1` for evolved sources) from "cut failed" in the attrition waterfall | framework, El-Badry 2026 |
| Q11 | Janssens Table 1 publishes no `a`–`b` covariance; zero correlation is the default — bound the effect on `σ_M̃1` and record it | El-Badry 2026 |
| Q15 | El-Badry 2026's ~50% SB1 spurious rate is not row-recoverable from Table 8; determine the denominator. Treat as **advisory**; gate acceptance on the two astrometric targets | spuriousness model |

Resolved and worth knowing: Q14 — source `4373465352415301632` **is Gaia BH1**; Andrews excluded it
as a scanning-law artifact and was wrong. `andrews2022.yaml` stays frozen as published (N=24); the
correction lives only in `andrews2022_modified.yaml` (N=25), which is the variant that feeds
inference. Q6 — the SB1 branch is reproduction/validation-only in v1.

## v1 scope boundaries (deliberate, not bugs)

`P(single) = P(triple) = 0` (triples module built, forced off) · fully joint inference deferred to
v2 (v1 is staged-but-connected: gate and companion-nature results are fixed plug-in weights) · SB1
branch has no inference entry point · DR4 configured but not runnable · no absolute Galactic rate
normalization, population synthesis, or binary-evolution modeling · accel/jerk systems matched at
aggregate solution-type occupancy only · RV per-point outlier rejection and general robust/bad-data
treatment not built · M1 uncertainty treated as Gaussian · SPHEREx documentation-only; Lam, El-Badry
& Simon (2025) analytic selection function documented but unused.

## Gotchas

- Science code imports gaiamock **only** via `import_gaiamock_mod()`; a version-triple mismatch
  (`gaiamock_mod_release` / `sha256` / git commit) hard-refuses at stage start.
- **Never** add a diagonal-only covariance fallback — NSS reconstruction failures are counted in
  `covariance_health` and excluded from MC-dependent samples, never silently downgraded.
- Literature-sample parent queries must run against the **uncut** Gaia snapshot
  (`20260826T234425Z_3d3f740b080c`); the quality-cut snapshot under-counts the parent.
- Andrews evaluation needs the `+enrich+mc10000` cache (`p_m2_above` is absent from `+enrich`).
  Pass `membership['andrews2022']` when evaluating El-Badry 2026 so `andrews2022_import` binds.
- Column ownership is strict: Andrews owns `p_m2_above` / `m2_msun` / `m2_msun_error` at fixed
  `M1 = 1.0`; El-Badry 2026 owns `m1_tilde_msun` / `m2_tilde_msun` / `sigma_m2_astrometric_msun` at
  fixed Janssens `M̃1`. **Never alias Andrews' σ into El-Badry's.**
- `σ_M̃2` is a lazy full-covariance NSS MC at the `m2_error` cut with **fixed** Janssens `M̃1`
  (`propagate_fit_uncertainty: false`, Q12), provenance `elbadry2026_m1_tilde_fixed`.
- The NSS enrichment job is **COMPLETED** — do not re-run `--poll-job` unless
  `nss_enrichment/meta.yaml` is missing.
- `bins="auto"` on heavy-tailed NSS distributions (RUWE, period) produces thousands of sub-pixel
  bars; cap with `diagnostics.histogram_max_bins` and drop non-finite values before binning.
- `mwdust.Combined19()` triggers a one-time network download when
  `selection_function_astrometric.extinction_model: combined19`.
- Do not host or require the default ~984 MB `healpix_scans.zip`.
- `scripts/attach_mc_to_selection_cache.py` (referenced by `docs/SELECTION_REPRODUCTION_STATUS.md`
  for rebuilding the Andrews `+enrich+mc10000` cache) **is on `main`**, landed via PR #150
  (issue #141, merge SHA `03a452a`). The cache itself is already built under
  `data/dr3/gaia_snapshots/`.
- Never read `dark-hunter_rv` / `dark-hunter_sed` production output directories live — snapshot with
  a timestamp (and preserve mtimes, which the gate now orders by).
