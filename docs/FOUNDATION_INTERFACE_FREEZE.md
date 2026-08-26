# Foundation interface freeze

Status: **frozen** (landed with issue #15 / PR #27). Parent umbrella: #2.

Phase 1+ subagents build against these interfaces. Breaking changes require a docs PR
first and an explicit ask. Phase 2 (#35 / #36–#39) extended config sections and the
resume checksum payload; this document tracks the post–Phase-2 contract.

Authoritative design: `docs/ARCHITECTURE.md`, `docs/ORCHESTRATION_PLAN.md`,
`docs/GAIAMOCK_API.md`.

---

## 1. Package layout (import surface)

| Symbol / module | Role |
|---|---|
| `darkhunter_pop.constants` | True physical constants only (`astropy.constants`, `M_CH`, TAG10/Santos tables) |
| `darkhunter_pop.config_schema.PipelineConfig` | Validated merged config |
| `darkhunter_pop.config_loader.load_config` / `config_checksum` / `assert_config_checksum` | Load, merge fragments, resume checksum |
| `darkhunter_pop.schemas.*` | `ParameterSet`, `CandidateRecord`, `RunManifest`, enums, gate/follow-up types |
| `darkhunter_pop.run_management.*` | Stage registry, artifact paths, run files, plan/purge |
| `darkhunter_pop.gaiamock_vendor.import_gaiamock_mod` | Sole science import path for gaiamock |
| `darkhunter_pop.physics_utils` | Units + Poisson primitives only (no Kepler/RUWE) |

---

## 2. Schemas Phase 1 may rely on

### `ParameterSet` (v1)

- Fields: `names`, `values`, `covariance` (symmetric), `provenance` (required), optional `units`
- `marginal(name) -> MarginalView` from diagonal variance
- `get_posterior_samples()` → `NotImplementedError` (samples = future HDF5; see `SAMPLES_HDF5_FORMAT`)
- Provenance must not inflate uncertainties

### `CandidateRecord`

- Required: `source_id`
- Optional blocks: NSS / Thiele–Innes / `rv_summary` (dict; #5 conforms) / photometry / TESS /
  `m1`/`m2` ParameterSets / `orbit_tier` / `fit_tier` / `companion_nature_weights` / `extras`

### Enums

- `OrbitTier`: `astrometry_only` \| `joint_astrometry_rv`
- `FitTier`: `bulk_estimate` \| `full_uberMS`
- `ActiveDRMode`: `dr3` \| `dr4` (execution: **dr3 only** in v1)
- `StageStatus`: `pending` \| `running` \| `completed` \| `skipped` \| `failed` \| `cached`

### `RunManifest` / `StageRecord`

Live YAML under `runs/{run_id}.yaml`. Minimum fields as in ARCHITECTURE.md §5.
`is_incomplete()` is true until every recorded stage is completed/skipped/cached.

---

## 3. Config keys stages may rely on

Root: `PipelineConfig` (`config/config.yaml` + `config/fragments/*.yaml`).
Review/Integration materializes fragments into `config/config.yaml` at checkpoints;
`load_config` still deep-merges fragments first (canonical file wins on conflicts).

| Section | Notes |
|---|---|
| `paths.artifact_root`, `paths.data_root` | Artifacts under `{artifact_root}/{run_id}/{stage}/{fingerprint}.h5` |
| `active_dr_mode` | Default `dr3`; `require_dr3_active_for_v1()` |
| `gaiamock.mod_release` (+ optional sha/commit pins) | Version triple with installed overlay |
| `mass_calibration.*` | method, `sigma_logM`/`R`, Santos flag, `delta_M_Ch_msun` |
| `mass_derivation.*` | dark-companion flux ratio, uberMS prior/watch-list, SED queue caps |
| `classification.*` | `M_MIN_msun`, `n_sigma_mass_cut`, `M_TOV_msun` |
| `physics.*` | cooling tracks/atmosphere/path, IMF, `mc_noise_threshold` |
| `selection_function_astrometric.*` | shared mock/validation; distance windows under `dr3`/`dr4` |
| `selection_function_followup.*` | shared follow-up SF; accel/jerk catalog pins under `dr3`/`dr4` |
| `sensitivity_analysis.*` | N-D vs 1D / covariates; uses `physics.mc_noise_threshold` for MC gate |
| `diagnostics.*` | figure/report layout + hook enable flags (not science thresholds) |
| `dr3.*` / `dr4.*` | Independent path configs; `quality_cut_bins` is an arbitrary-length list |

Checksum for resume/amend (`config_schema.SHARED_CHECKSUM_SECTIONS` + active DR subtree):
**active DR subtree +** `mass_calibration`, `mass_derivation`, `classification`, `physics`,
`gaiamock`, `paths`, `selection_function_astrometric`, `selection_function_followup`,
`sensitivity_analysis` (`config_loader.config_checksum`). `diagnostics` is intentionally
excluded (layout-only). Inactive DR changes do not affect the checksum.

Constants precedence: `effective_M_Ch_msun(config) = constants.M_CH + delta_M_Ch_msun`.

---

## 4. Stage registry contract

Canonical names and order: `run_management.STAGE_ORDER` / `STAGE_REGISTRY`.

| Stage | Module | `inputs_from` (declared) |
|---|---|---|
| `data_acquisition` | `data_acquisition` | — |
| `mass_derivation_bulk` | `mass_derivation` | `data_acquisition` |
| `mass_derivation_refined` | `mass_derivation` | `mass_derivation_bulk` |
| `rv_astrometry_gate` | `rv_consistency` | `mass_derivation_refined` |
| `joint_orbit_fit` | `rv_consistency` | `rv_astrometry_gate` |
| `companion_nature_likelihood` | `companion_nature` | `joint_orbit_fit`, `mass_derivation_refined` |
| `triples` | `triples` | `companion_nature_likelihood` |
| `selection_function_astrometric` | `forward_model` | `data_acquisition` |
| `selection_function_followup` | `forward_model` | `selection_function_astrometric` |
| `population_model` | `population_model` | companion_nature + both selection functions |
| `sensitivity_analysis` | `sensitivity_analysis` | `population_model` |
| `inference` | `inference` | population + selections + sensitivity |
| `diagnostics` | `diagnostics` | `inference` |

Each stage: `dependency_modules` → `source_hash`; `config_fingerprint_keys` → artifact path;
optional `uses_gaiamock`.

Helpers: `resolve_run_file`, `plan_stage`, `format_run_plan`, `new_run_for_force_rerun`,
`purge_run`, `compute_source_hash`, `assert_stage_source_hash`.

---

## 5. gaiamock

- Submodule `vendor/gaiamock`; overlay source `vendor/overlays/gaiamock_mod.py`
- Install: `scripts/install_gaiamock_mod.sh` (Release `gaiamock-mod-v1` or `mod_files/`)
- Science: `import_gaiamock_mod()` only — see `docs/GAIAMOCK_API.md`

---

## 6. Tests / CI

| Marker | Merge-required? |
|---|---|
| `unit` | yes |
| `physics` | yes |
| `api` | yes |
| `gaiamock` | no (path-filter / `workflow_dispatch`) |
| `network` / `slow` | no |

---

## 7. Phase status (orchestrator)

- Phase 0 Foundation + Phase 1 (#28–#31) + Phase 2 (#35–#39) landed on `main`.
- Continuous Review/Integration: #40.
- Next: Phase 3 per `ORCHESTRATION_PLAN.md` §5 — roster #6 `rv_astrometry_gate` +
  `joint_orbit_fit`, #8 `triples` stub (Agents Window / worktree per §1 and §4).
