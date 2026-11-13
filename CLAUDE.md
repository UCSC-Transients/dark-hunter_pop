# CLAUDE.md — dark-hunter_pop

Guidance for Claude Code / Cursor agents working in this repository.

## Project

Derives a **debiased mass function `dN/dM` for compact-object companions (WD / NS / BH) in Gaia
astrometric binaries**: a population model forward-modeled through a modified-RUWE `gaiamock`
overlay, compared to the real Gaia DR3 NSS sample with an **inhomogeneous Poisson point-process
likelihood**. Normalization is shape/rate relative to the quality-cut NSS parent sample, never an
absolute Galactic density. No population-synthesis or binary-evolution layer. Primary interest: NS
and BH mass functions, the mass gap, `M_Ch` / `M_TOV`.

## Documentation authority

Read these before making a design change. **Docs-first: update the spec via PR before implementing.**
| Doc | Role |
|---|---|
| `docs/STATUS.md` | **Live status**: validation-ladder history, literature-reproduction table, run files (moved from CLAUDE.md, #652; stale, refreshed at wave checkpoints) |
| `docs/REPO_REFERENCE.md` | Full module table, data flow, literature-layer detail, sibling-repo interfaces, data/config paths, dependencies, full gotchas (moved from CLAUDE.md, #652) |
| `docs/ARCHITECTURE.md` | **Authoritative** pipeline specification (stages, schemas, run management, DR3/DR4 matrix, config philosophy, v1 limitations, CI) |
| `docs/FOUNDATION_INTERFACE_FREEZE.md` | Frozen import/stage/config contracts every stage builds against; per-phase landing status |
| `docs/ORCHESTRATION_PLAN.md` | Subagent roster, phase sequencing, git/PR/issue workflow |
| `docs/CONTINUATION_PLAN.md` | Phase 8+ spec: literature selection functions, spuriousness model, §15 open questions |
| `docs/EXECUTION_PLAN.md` | **Forward-looking plan** — remaining work to a v1 science result and paper |
| `docs/ORCHESTRATOR_PROMPT.md` | Copy-paste prompts: the long-lived orchestrator (§A) and the verification agent (§B) |
| `docs/SELECTION_REPRODUCTION_STATUS.md` | Live hand-off: which literature reproduction numbers currently fail and why |
| `docs/MOCK_POPULATION_SPEC.md` | Step 1 mock: parent, MdS17 companions, proposal set + weights, Malmquist, giants, isochrone M1, validation ladder (§5); Ryan's decisions §0.1–§0.4 |
| `docs/RUNG5_SPEC.md` | Rung 5 (DRAFT, #610): first DR3 WD/NS/BH mass functions, gates 5a–5g, R5-T1–T14, R5-Q1–Q12 |
| `docs/EPOCH_MODEL_SPEC.md` | Epoch model around gaiamock GOST list (§8 = adopted v2) |
| `docs/ELBADRY2024_REPRODUCTION_SPEC.md` | El-Badry et al. (2024) six-panel / solution-type gate: real comparison set, cuts, cost, Q5–Q12 |
| `docs/gate*/README.md` | Measured per-gate reports (`gate390`, `gate399`, `gate400`, `gate405`, `gate418`, `gate_giants`, `gate391`, `gate301`) |
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
artifact and a `source_hash`:

`data_acquisition` → `mass_derivation_bulk` → `sample_selection` → `mass_derivation_refined` →
`rv_astrometry_gate` → `joint_orbit_fit` → `companion_nature_likelihood` → `triples` (off) →
`selection_function_astrometric` → `selection_function_followup` → `population_model` →
`sensitivity_analysis` → `inference` → `diagnostics`

Stage → module: both mass-derivation stages → `mass_derivation.py`; both selection functions →
`forward_model.py`; `rv_astrometry_gate` + `joint_orbit_fit` → `rv_consistency.py`.

**Module map** (full table with roles: `docs/REPO_REFERENCE.md`). `src/darkhunter_pop/`:
`constants` (true constants only) · `schemas` (`ParameterSet`, `CandidateRecord`, `RunManifest`, …) ·
`config_schema`/`config_loader` · `run_management` · `pipeline` · `gaiamock_vendor` (**sole** science
import path) · `physics_utils` (units + Poisson primitives only) · `data_acquisition` ·
`nss_covariance` · `mass_derivation` · `sample_selection` + `elbadry2024_selection`/
`elbadry2026_selection`/`janssens_mass` · `mc_mass_function` · `spuriousness_model` ·
`sample_inclusion`/`sample_diagnostics` · `rv_adapter`/`rv_consistency` · `phot_sed_adapter` ·
`companion_nature` · `forward_model` · `population_model` · `sensitivity_analysis` · `inference` ·
`sbc`/`benchmarks`/`diagnostics`/`plotting`/`diagnostic_hooks` · `triples/` (off) · `run_validity`
(stand-in honesty flags) · Step 1 mock: `proposal_set`, `moe_distefano`, `malmquist`,
`malmquist_cmd`, `epoch_model`, `giants`, `isochrone_mass`, `compact_objects`, `injection_test`,
`cascade_replay`.

**Literature layer.** `rate_s(θ) = population_model(θ) × SF_astrometric × SF_followup × SF_sample_s`.
Each sample has `mode: reproduction` (paper's own masses; must recover published N and ID set; never
used for inference) and `mode: forward_model` (our mass posterior; feeds `inference`). Each
selection file owns its own `primary_mass` block (no global switch). Spuriousness is one shared
model `P(spurious | x)`; unknown labels are censored (Heckman-style), never dropped; quoted paper
rates are `validation_targets:`, never inputs. Detail: `docs/REPO_REFERENCE.md`.

## Compute environments

**Laptop** (this repo): workflow smoke-testing only, partial data, never the production target;
sandbox HDF5/pytest segfaults are a laptop artifact. **ziggy**: bulk data, SED production, natural
home for acquisition/mass derivation/selection. **lux**: production `inference` and mock-injection
volume (December access deadline). **Only the laptop is in scope**; ziggy and lux are deferred
(`docs/EXECUTION_PLAN.md` §2). Host differences are **config, never code** (`paths.data_root`,
`paths.artifact_root`, `mass_derivation.sed_summary_root`, `dr3.rv_summary_root`); a host profile
is a fragment, recorded in the run file.

## Sibling repos and data

- `dark-hunter_rv` → pop: `output/Gaia_DR3_<id>_summary.json` (schema v1), read by `rv_adapter`.
  `dark-hunter_sed` → pop: `sed_summaries/*_sed_summary.json` (→ `mass_derivation`) and
  `phot_sed/*_<model>_summary.json` (→ `phot_sed_adapter`). Pop reads JSON only; never imports or
  runs either package. `ParameterSet` from marginals: **never substitute a diagonal**. A breaking
  RV-summary field rename needs a docs PR against `FOUNDATION_INTERFACE_FREEZE.md` first. Open
  gaps #206 / #207. Full interface text: `docs/REPO_REFERENCE.md`.
- Config: single canonical `config/config.yaml`; tracked fragments `config/fragments/*.yaml`;
  frozen selections `config/selections/`; Step 1 proposal-set configs `config/population/`.
  Runs `runs/{run_id}.yaml` (tracked); artifacts `output/` (gitignored); `data/` gitignored;
  `vendor/` = gaiamock submodule + overlay. Path table: `docs/REPO_REFERENCE.md`.
- Deps: `gaiamock_mod` via `import_gaiamock_mod()` only; `dynesty`; `mwdust` Combined19; Bédard WD
  tracks (local); optional `nsstools` (Q7).

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
- Skills: `caveman` in every session; `strict-workflow` and `dark-hunter-pop-workflow`
  (`.cursor/skills/`) when opening a PR; `regression-hunter` for regression/bisect work.
- **Token budget** (`docs/EXECUTION_PLAN.md` §5.5a): this file autoloads — do not re-read it or
  README. Read specs by § / line range, never whole. pytest output `-q --tb=short | tail`. Every
  subagent dispatch sets `model` + `effort` explicitly (`haiku`/`low`, `sonnet`/`medium` default,
  `opus`/`high` for statistics); final reports are caveman-ultra in the §5.2 step-11 schema.

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

1. One worktree/branch per subagent; `[AI Checkpoint] <description>` micro-commits, **pushed**, kept
   in the PR merge commit (bisect with `--first-parent`, then descend into a suspect PR).
2. **Full local `pytest -m "unit or physics or api"` pass before opening a PR.**
3. PR via `gh`: detailed description, `Closes #N` alone on a line (one per issue), test checklist,
   CI `tests` green. A PR that renames a parameter, config key, signature or schema field says so
   under "Interface changes".
4. One issue = one thing; fine-grained issues, optional umbrellas. Track work on GitHub, not only in
   chat. Subagents may *edit* their issue and must **file a new issue** for anything out of scope.
   **Nothing is silently fixed in a PR**; #49 bounces untracked changes. Orchestrator owns issues
   (filed by a laptop Claude Code agent; the orchestrator has no `gh`).
5. Review/Integration merges config fragments into `config/config.yaml` at checkpoints.
6. **PRs may auto-merge on green CI** (merge commits, never squash; `dark-hunter_pop` only; rv/sed
   merged by hand). The standing verification agent (roster #49) re-runs the suite on `main`, runs
   `--dry-run` plus a small stage, hunts cross-PR drift (kwarg names, duplicate config keys, schema,
   stale `inputs_from`/`dependency_modules`, stale docs), **owns branch/worktree teardown** and may
   revert a broken merge. It writes only tests. Coding agents never delete a branch or worktree.
7. Branch protection on `main`: PR + `tests` required, zero reviews, admin bypass.
8. **All coding is dispatched to subagents**, one per ticket, with the session preamble in
   `docs/EXECUTION_PLAN.md` §5.2 and explicit model/effort (§5.5, §5.5a). The orchestrator plans,
   files issues, reviews, runs wave stops; it does not edit code.
9. Concurrency is bounded by **laptop memory**: classify sessions heavy (pytest/stages/HDF5/MC) or
   light, budget **40 GB** of 64 GB, ~6 GB per heavy session, start at three (`EXECUTION_PLAN.md`
   §5.6). Subagents report to the orchestrator, which pings #49.
10. Every wave ends in a hard stop for operator review with a written handoff note (§5.9).

## Status

Live status (validation ladder, reproduction table, run files) is in `docs/STATUS.md`. Standing
priority: Step 1 full-sample mock match first; diagnostics must be real (no undeclared stand-ins).
Scope: laptop only; ziggy/lux deferred.

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

(Full text with all paths/IDs: `docs/REPO_REFERENCE.md` § Gotchas.)

- gaiamock only via `import_gaiamock_mod()`; version-triple mismatch hard-refuses at stage start.
- **Never** a diagonal-only covariance fallback; NSS reconstruction failures are counted in
  `covariance_health` and excluded, never downgraded.
- Literature-sample parent queries run against the **uncut** snapshot (`20260826T234425Z_3d3f740b080c`).
- Andrews (both modes) needs only the ATF-notebook sidecar
  (`scripts/build_andrews2022_atf_columns.py`); without it every `atf_*` cut is N/A. Pass
  `membership['andrews2022']` when evaluating El-Badry 2026. No cut reads the legacy `+enrich+mc10000`
  cache; do not rebuild it.
- Column ownership is strict: Andrews owns `andrews_atf_*`; El-Badry 2026 owns `m1_tilde_msun` /
  `m2_tilde_msun` / `sigma_m2_astrometric_msun` at fixed Janssens `M̃1`. **Never alias Andrews' σ into
  El-Badry's.**
- `σ_M̃2` is lazy at the `m2_error` cut with fixed `M̃1` (Q12); reproduction uses analytic
  propagation (`sigma_m2_tilde.method`).
- El-Badry 2026 `sub_chandrasekhar` reproduction = Shahaf 2023b class-III cross-match (22); missing
  snapshot raises; `forward_model` never reads it (Q2).
- NSS enrichment job is COMPLETED; do not re-run `--poll-job` unless `nss_enrichment/meta.yaml` is
  missing.
- `bins="auto"` on heavy-tailed data: cap with `diagnostics.histogram_max_bins`, drop non-finite.
- `mwdust.Combined19()` downloads once; do not host the ~984 MB `healpix_scans.zip`.
- Step 1: any change to parent cuts, truth parallax, M1, light split, epoch model, gaiamock triple or
  cascade settings invalidates stored proposal-set draws (re-simulate; spec §3.5). Run gaiamock
  drivers under `nice` + `threadpoolctl` (#408). Proposal-set configs in `config/population/`, not
  `config/fragments/`.
- Mock `insufficient_visibility` fraction is ~0.10% vs DR3 1.71% (#428, #432); a test expecting a
  sizeable fraction is wrong.
- Never read `dark-hunter_rv` / `dark-hunter_sed` production output live; snapshot with timestamp,
  preserve mtimes.
