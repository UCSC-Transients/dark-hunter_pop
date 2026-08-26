---
name: dark-hunter-pop-workflow
description: >
  Project-specific conventions for dark-hunter_pop (the compact-object mass-function pipeline):
  reproducibility manifests, run-file/staged-execution conventions, config-fragment-then-merge
  process, no-hardcoded-values enforcement, the ParameterSet interface pattern, DR3/DR4
  parameter-independence rules, and statistical guardrails specific to this forward-modeling
  pipeline. Use for all work in dark-hunter_pop, and for dark-hunter_rv/dark-hunter_sed changes
  made in service of this project. Layers on top of strict-workflow and regression-hunter; does
  not replace them.
---

# dark-hunter_pop Project Workflow

This skill is intentionally project-specific. `strict-workflow` and `regression-hunter` stay
general-purpose across all UCSC-Transients repos; this skill carries conventions particular to a
large, reproducibility-critical, statistics-heavy forward-modeling pipeline.

## 1. No hardcoded values, ever

Every true physical constant lives in `src/darkhunter_pop/constants.py` (prefer
`astropy.constants`; add named extras only when not choosable). Every threshold, statistical
criterion, method switch, prior, or file path lives in `config.yaml` — never inline. This
explicitly includes: the ΔBIC threshold for the companion-nature likelihood, the chi2/dof
threshold for the RV/astrometry outlier gate, the mock-injection Poisson-noise threshold
(default 0.1), `M_MIN`, `M_TOV` prior, TAG10 method / `sigma_logM` / Santos on-off, and the
goodness-of-fit quality cut, which must support an arbitrary, configurable number of
(magnitude, threshold) bins — never hardcode two. `M_Ch` is a constant (optional `Delta_M_Ch` in
config).

## 2. Every stage is named, cached, and reported — never a bare "stage1"

Stage identifiers are descriptive (`data_acquisition`, `mass_derivation_bulk`,
`rv_astrometry_gate`, ...) and match the module/file that implements them — see
`ARCHITECTURE.md` §4-5. When implementing a stage:
- Declare its expected output artifact path(s), parameterized by whatever config subset actually
  affects the result, so a different quality cut or option set produces a genuinely different
  output file rather than overwriting or falsely reusing a cached one.
- Check for that output before running; skip if it already exists and no force-re-run flag is set.
- Honor a per-stage force-re-run override.
- Append your stage's completion record (config subset used, output path, timestamp, commit hash,
  status) to the run file as you finish, rather than assuming some other layer will do it for you.
- Never silently swallow the "print what will run and why" requirement — every invocation of the
  main program prints its run plan (which stages run vs. are skipped, and why) before executing
  anything.

## 3. `ParameterSet` is the standard fit-output type

Almost everything this pipeline fits is a correlated, multi-quantity output — MSC/gspphot's
Teff+logg+[Fe/H], TAG10's simultaneous mass+radius, uberMS's full parameter vector, the RV/orbit
fit's P+e+T_periastron+K+ω — not an independent scalar. Use `ParameterSet` (named vector +
covariance/joint samples) as the default output type for any new fitting code, even if only one
quantity looks interesting right now. Reach for a bare scalar-with-uncertainty type only for
genuinely standalone external inputs never jointly fit with anything else tracked in this
pipeline. Tag every `ParameterSet` with its provenance (which method/tier produced it) as
metadata; provenance does not, by itself, justify inflating the reported uncertainty.

## 4. Reproducibility manifest = the run file

The run file (§2 above) *is* the run manifest — a live document built up as the pipeline
proceeds, not a one-shot end-of-run emission. It must end up containing: config snapshot, commit
hashes for this repo and every pinned external dependency (`gaiamock`, `uberMS`, `dark-hunter_rv`,
`dark-hunter_sed`), the Gaia query date and result checksum, package versions, random seeds. Full
accounting, not bitwise-exact replay — samplers like `dynesty` are validated by a multi-run
robustness protocol (§7), not by reproducing identical output from identical seeds.

## 5. Config: fragments during development, one file at delivery

Draft config additions in `config/fragments/<domain>.yaml` (tracked) against the frozen schema.
The review/integration subagent merges fragments into the single canonical `config.yaml` at each
integration checkpoint — the delivered repo always has exactly one merged config file.

## 6. DR3/DR4: independent by default, shared only when it's genuinely physics

Any parameter that is Gaia-mission-specific (scanning law, baseline, zero-point versions, the
goodness-of-fit quality-cut bins, epoch-data availability) or that governs how *external* data is
used (which photometric filters feed SED fitting, external quality cuts) is configured
**independently for DR3 and DR4, even if the values happen to match** — never share the config key.
Genuine physics/population parameters (M_TOV prior, IMF choice, cooling-track model, mass-function
bin-edge policy) **are shared**. Run the audit function (walks the full parameter set, flags
unexpected divergence in shared physics parameters and, informationally, unexpected identical
values in path-specific parameters) before any change to DR3/DR4 config is considered complete.

## 7. Statistical guardrails specific to this pipeline

- **Bin edges for the non-parametric mass function are fixed before looking at real detection
  counts** — from a fiducial population's expected-detection density, never data-adaptive.
- **Mock-injection Monte Carlo noise verified subdominant to real Poisson noise**:
  `sigma_MC / sigma_Poisson < mc_noise_threshold` (config default `0.1`) per bin, with a
  convergence diagnostic proving it.
- **External compact-object populations (pulsar mass function, LIGO BH mass function, any other
  literature compact-object catalog) are never used as an inference prior.** Comparison-only,
  always, no per-task exceptions.
- **Sampler robustness, not bitwise reproducibility.** `dynesty` runs are validated by multiple
  independent runs (different seeds/live-point counts) checked for posterior agreement.
- **Small-N sub-analyses report limits, not overconfident posteriors.** Any stratified or binned
  result gets an automatic posterior-vs-prior overlap check.
- **Every population class — BH, NS, WD, other, and outlier — is a rate function of mass at
  minimum, with additional covariates included only where the unified sensitivity-analysis module
  demonstrates they matter.** Don't hand-pick covariates without running that module.
- **`M_Ch` truncation applies only to the WD class. `M_TOV` truncation (soft/marginalized) applies
  only to NS.** The outlier class is explicitly exempt from both.
- **v1 inference is staged-but-connected, not fully joint**: per-system evidence is computed once
  and used as a fixed weight feeding the population inference, not re-sampled jointly. Do not
  silently upgrade a task to a fully joint treatment mid-implementation — that's a separately
  scoped v2 decision, not an incremental improvement to make casually.

## 8. Diagnostics are mandatory when specified, and only when specified

Every diagnostic and validation check named in a subagent's task list is mandatory — do not skip
for brevity, do not treat "lots of diagnostics" as scope creep. If you notice something else worth
investigating that wasn't specified, flag it as a suggested follow-up rather than silently doing
it.

## 9. Directory structure

Most modules are single files directly under `src/darkhunter_pop/` — don't create a subdirectory
for a one- or two-file module. Promote a module to its own subdirectory only once it's genuinely
outgrown a single file.

## 10. External repo changes (`dark-hunter_rv`, `dark-hunter_sed`)

Same rules as `dark-hunter_pop` itself: full local `pytest` pass before any PR, `strict-workflow`-
style issues/PRs, their own `[AI Checkpoint]`/`regression-hunter` conventions apply identically.
Never read their production output directories live — snapshot/copy with a timestamp instead.

## 11. Caveman-mode adjustments for this project

On top of the existing "Code/commits/PRs: write normal" exemption:
- **Diagnostic reports, validation summaries, run-plan screen output, and plot captions are also
  exempt** from compression — the run-plan output specifically must stay fully legible, since it's
  what a person checks before trusting what's about to execute.
- **Numerical/statistical convergence failures (sampler warnings, fit failures, non-convergence,
  RV/astrometry gate failures) get full diagnostic output, not the shortest decisive line.**
