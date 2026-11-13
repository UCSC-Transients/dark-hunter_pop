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
| `docs/STATUS.md` | **Live status**: validation-ladder history, literature-reproduction table, run files (stale; refreshed at checkpoints) |
| `docs/REPO_REFERENCE.md` | Module table, data flow, literature layer, sibling interfaces, paths, deps, full gotchas, run management, git/PR workflow |
| `docs/ARCHITECTURE.md` | **Authoritative** pipeline spec |
| `docs/FOUNDATION_INTERFACE_FREEZE.md` | Frozen import/stage/config contracts every stage builds against; per-phase landing status |
| `docs/ORCHESTRATION_PLAN.md` | Subagent roster, phase sequencing, git/PR/issue workflow |
| `docs/CONTINUATION_PLAN.md` | Phase 8+ spec: literature selection functions, spuriousness model, §15 open questions |
| `docs/EXECUTION_PLAN.md` | **Forward-looking plan** — remaining work to a v1 science result and paper |
| `docs/ORCHESTRATOR_PROMPT.md` | Copy-paste prompts: the long-lived orchestrator (§A) and the verification agent (§B) |
| `docs/SELECTION_REPRODUCTION_STATUS.md` | Live hand-off: which literature reproduction numbers currently fail and why |
| `docs/MOCK_POPULATION_SPEC.md` | Step 1 mock: parent, MdS17 companions, proposal set + weights, Malmquist, giants, isochrone M1, validation ladder (§5); Ryan's decisions §0.1–§0.4 |
| `docs/RUNG5_SPEC.md` | Rung 5 (DRAFT, #610): first DR3 WD/NS/BH mass functions, gates 5a–5g, R5-T1–T14, R5-Q1–Q12 |
| `docs/EPOCH_MODEL_SPEC.md` | Epoch model around gaiamock GOST list (§8 = adopted v2) |
| `docs/ELBADRY2024_REPRODUCTION_SPEC.md` | El-Badry 2024 six-panel gate |
| `docs/gate*/README.md` | Measured per-gate reports (`gate390`, `gate399`, `gate400`, `gate405`, `gate418`, `gate_giants`, `gate391`, `gate301`) |
| `docs/GAIAMOCK_API.md` | `gaiamock_mod` public API; what must **not** be reimplemented |
| `docs/PLOTS.md` | Figure style guide + `plotting:` config defaults |

README prose is a summary, never a substitute for the locked decisions above.

## Repo

GitHub `UCSC-Transients/dark-hunter_pop`; local `/Users/rfoley/darkhunter/pop/dark-hunter_pop/`; package `darkhunter_pop`, `src/` layout, one file per stage. Worktrees under `../dark-hunter_pop-worktrees/` and `.worktrees/`.

## Architecture

Ordered, registered stages (`run_management.STAGE_ORDER`), each with `inputs_from`, one HDF5
artifact and a `source_hash`:

`data_acquisition` → `mass_derivation_bulk` → `sample_selection` → `mass_derivation_refined` →
`rv_astrometry_gate` → `joint_orbit_fit` → `companion_nature_likelihood` → `triples` (off) →
`selection_function_astrometric` → `selection_function_followup` → `population_model` →
`sensitivity_analysis` → `inference` → `diagnostics`

Stage → module: both mass-derivation stages → `mass_derivation.py`; both selection functions →
`forward_model.py`; `rv_astrometry_gate` + `joint_orbit_fit` → `rv_consistency.py`.

**Modules** (`src/darkhunter_pop/`, full table: `docs/REPO_REFERENCE.md`): one file per stage plus `constants` (true constants only), `schemas` (`ParameterSet`, `RunManifest`, ...), `config_loader`, `run_management`, `gaiamock_vendor` (**sole** science import path), `physics_utils` (units + Poisson only), `run_validity` (stand-in flags), and the Step 1 mock set (`proposal_set`, `moe_distefano`, `malmquist*`, `epoch_model`, `giants`, `isochrone_mass`, `compact_objects`, `injection_test`, `cascade_replay`).

**Literature layer.** `rate_s(θ) = population_model(θ) × SF_astrometric × SF_followup × SF_sample_s`. Per sample: `mode: reproduction` (paper's masses; must recover published N and IDs; never for inference) and `mode: forward_model` (feeds `inference`). Each selection file owns its `primary_mass` block. One shared spuriousness model; unknown labels censored, never dropped; paper rates are `validation_targets:`, never inputs.

## Compute environments

**Laptop** only is in scope (smoke-testing, partial data; sandbox HDF5/pytest segfaults are a laptop artifact). ziggy (bulk data, SED) and lux (production `inference`, mock volume; the December access deadline is about this machine) are deferred (`docs/EXECUTION_PLAN.md` §2). Host differences are **config, never code** (`paths.data_root`, `paths.artifact_root`, `mass_derivation.sed_summary_root`, `dr3.rv_summary_root`); record the profile in the run file.

## Sibling repos and data

`dark-hunter_rv` / `dark-hunter_sed` feed pop via JSON summaries only (`rv_adapter`, `mass_derivation`, `phot_sed_adapter`); pop never imports or runs them. **Never substitute a diagonal covariance** for marginals. Breaking RV-summary renames need a docs PR against `FOUNDATION_INTERFACE_FREEZE.md` first. Config: one `config/config.yaml`; fragments, frozen `config/selections/`, `config/population/`. Interfaces, paths, deps: `docs/REPO_REFERENCE.md`.

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
- **Plotting**: shared primitives in `plotting.py` (Okabe–Ito, serif, inward ticks, no slashes in units, finite-only histograms with `diagnostics.histogram_max_bins`); see `docs/PLOTS.md`.
- **Diagnostics named in a task list are mandatory.** Unspecified extras are flagged as follow-ups,
  not silently added. Diagnostic reports, run-plan output, and plot captions are **exempt** from
  terse/caveman compression, as are convergence and gate failures.
- Skills: `caveman` always; `strict-workflow` + `dark-hunter-pop-workflow` when opening a PR; `regression-hunter` for bisects.
- **Token budget** (`docs/EXECUTION_PLAN.md` §5.5a): CLAUDE.md autoloads, never re-read it or README; read specs by § / range; pytest `-q --tb=short | tail`; every dispatch sets model + effort; final reports caveman-ultra (§5.2 step 11).

## Run and git rules (full text: `docs/REPO_REFERENCE.md` § Run management, § Git / PR workflow)

Docs-first. One worktree + branch per agent; never delete a branch or worktree (#49 owns teardown). Full `pytest -m "unit or physics or api"` pass before a PR, then auto-merge (merge commit); nothing silently fixed, file a new issue.
The run file is the manifest; force-rerun of a completed stage gets a new run file; refuse on config-checksum or gaiamock mismatch.

## Status

Live status (ladder, reproduction table, run files): `docs/STATUS.md`. Priority: Step 1 full-sample mock match first; diagnostics must be real (no undeclared stand-ins). Scope: laptop only.

## Open statistical questions (human sign-off)

| Q | Question | Blocks |
|---|---|---|
| Q1 | Multi-sample overlap (El-Badry 2024 drew from Andrews; 2026 subsample 3 *is* Andrews at `G < 15`): separate Poisson processes vs inclusion indicator; naive sum double-counts. Signed off #112 / PR #125 | `inference` |
| Q2 | Shahaf 2023b: cross-match 177-candidate catalog vs reimplement AMRF (a catalog cannot be applied to mocks) | El-Badry 2024, DR4 |
| Q4 | Do `corr_vec`/`bit_index` reconstruct the full covariance for every solution type | NSS covariance |
| Q7 | Vendor `nsstools` vs our `thiele_innes_to_campbell`; **measure** the delta first | El-Badry 2026, #133 |
| Q9 | `G < 15` on reproduced Andrews must give exactly 16; mismatch is a failure, not a tuning opportunity | El-Badry 2026 |
| Q10 | Cut evaluator must distinguish "not applicable" (undefined `M̃1`) from "failed" | framework |
| Q11 | Janssens Table 1 has no `a`–`b` covariance; default zero, bound the effect on `σ_M̃1` | El-Badry 2026 |
| Q15 | El-Badry 2026 ~50% SB1 spurious rate not row-recoverable; **advisory**, gate on the two astrometric targets | spuriousness |

Resolved: Q14 — `4373465352415301632` **is Gaia BH1** (Andrews wrongly excluded it); `andrews2022.yaml` stays frozen (N=24), correction only in `andrews2022_modified.yaml` (N=25, feeds inference). Q6 — SB1 branch is reproduction/validation-only.

## v1 scope boundaries (deliberate, not bugs)

`P(single) = P(triple) = 0` (triples module built, forced off) · fully joint inference deferred to
v2 (v1 is staged-but-connected: gate and companion-nature results are fixed plug-in weights) · SB1
branch has no inference entry point · DR4 configured but not runnable · no absolute Galactic rate
normalization, population synthesis, or binary-evolution modeling · accel/jerk systems matched at
aggregate solution-type occupancy only · RV per-point outlier rejection and general robust/bad-data
treatment not built · M1 uncertainty treated as Gaussian · SPHEREx documentation-only; Lam, El-Badry
& Simon (2025) analytic selection function documented but unused.

## Gotchas (full text: `docs/REPO_REFERENCE.md` § Gotchas)

- gaiamock only via `import_gaiamock_mod()`; version-triple mismatch hard-refuses.
- **Never** a diagonal-only covariance fallback; failures counted in `covariance_health`, excluded.
- Literature parent queries use the **uncut** snapshot (`20260826T234425Z_3d3f740b080c`).
- Andrews (both modes) needs the ATF sidecar (`scripts/build_andrews2022_atf_columns.py`) or every `atf_*` cut is N/A; pass `membership['andrews2022']` for El-Badry 2026; legacy `+enrich+mc10000` cache is unused, do not rebuild.
- Column ownership strict: Andrews `andrews_atf_*`; El-Badry 2026 `m1_tilde_msun`/`m2_tilde_msun`/`sigma_m2_astrometric_msun` (fixed Janssens `M̃1`). **Never alias Andrews' σ into El-Badry's.**
- El-Badry 2026 `sub_chandrasekhar` reproduction = Shahaf 2023b class-III cross-match; `forward_model` never reads it (Q2).
- NSS enrichment job COMPLETED; no `--poll-job` unless `nss_enrichment/meta.yaml` missing.
- `bins="auto"` on heavy tails: cap with `diagnostics.histogram_max_bins`, drop non-finite.
- `mwdust.Combined19()` downloads once; do not host `healpix_scans.zip`.
- Step 1: changing parent cuts, truth parallax, M1, light split, epoch model, gaiamock triple or cascade invalidates stored proposal-set draws (re-simulate; spec §3.5). Run drivers under `nice` + `threadpoolctl` (#408). Proposal-set configs in `config/population/`, not `config/fragments/`.
- Mock `insufficient_visibility` ~0.10% vs DR3 1.71% (#428); tests expecting more are wrong.
- Never read rv/sed production output live; timestamped snapshots, preserve mtimes.
