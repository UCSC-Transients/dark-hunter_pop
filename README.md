# dark-hunter_pop

A debiased mass function, dN/dM, for compact-object companions (WD / NS / BH) in Gaia
astrometric binaries. A population model is forward-modeled through a modified-RUWE
[`gaiamock`](https://github.com/kareemelbadry/gaiamock) overlay and compared to the real
Gaia NSS sample with an inhomogeneous Poisson point-process likelihood.

**Status:** Phases 0–8 complete on `main` (Foundation through literature sample-selection
layer). Entry point: `scripts/run_pipeline.py`. Design authority remains
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/FOUNDATION_INTERFACE_FREEZE.md`](docs/FOUNDATION_INTERFACE_FREEZE.md), and
[`docs/CONTINUATION_PLAN.md`](docs/CONTINUATION_PLAN.md) (Phase 8) — do not treat README
prose as a substitute for those locked decisions.

## Documentation index

| Doc | Role |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Authoritative pipeline specification |
| [`docs/ORCHESTRATION_PLAN.md`](docs/ORCHESTRATION_PLAN.md) | Subagent roster, phasing, git/PR workflow |
| [`docs/FOUNDATION_INTERFACE_FREEZE.md`](docs/FOUNDATION_INTERFACE_FREEZE.md) | Frozen import/stage contracts Phase 1+ build against |
| [`docs/PLOTS.md`](docs/PLOTS.md) | Figure style guide + `plotting:` config defaults |
| [`docs/GAIAMOCK_API.md`](docs/GAIAMOCK_API.md) | `gaiamock_mod` public API for science paths |
| [`docs/CONTINUATION_PLAN.md`](docs/CONTINUATION_PLAN.md) | Phase 8+ selection-function / spuriousness spec + §14 prompts |
| [`docs/PHASE1_KICKOFF.md`](docs/PHASE1_KICKOFF.md)–[`PHASE7_KICKOFF.md`](docs/PHASE7_KICKOFF.md) | Historical Agents Window paste prompts (per phase) |

## Related repositories

- [`dark-hunter_rv`](https://github.com/UCSC-Transients/dark-hunter_rv) — radial velocities (The Joker).
- [`dark-hunter_sed`](https://github.com/UCSC-Transients/dark-hunter_sed) — SED fitting (uberMS / MIST / Payne).

## Layout

```
config/                 merged config.yaml + tracked fragments/
runs/                   per-run YAML manifests (tracked)
src/darkhunter_pop/     package (constants.py + one file per stage)
vendor/gaiamock/        submodule; overlay installed from Release / mod_files/
vendor/overlays/        tracked gaiamock_mod.py source
vendor/DATA_MANIFEST.md SHA256s for gaiamock-mod-v1 assets
output/                 stage HDF5 artifacts (gitignored)
data/                   raw snapshots / sheet dumps (gitignored)
tests/ scripts/ notebooks/ docs/
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot,inference]"
git submodule update --init

# Optional — modified-RUWE gaiamock overlay (GSL + gcc):
#   pip install -e ".[gaiamock]"   # healpy, joblib, mwdust
#   scripts/install_gaiamock_mod.sh
#   # or: DOWNLOAD_FROM_RELEASE=1 scripts/install_gaiamock_mod.sh
```

`scripts/install_gaiamock_mod.sh` pulls immutable Release
[`gaiamock-mod-v1`](https://github.com/UCSC-Transients/dark-hunter_pop/releases/tag/gaiamock-mod-v1)
assets (or copies from local `mod_files/`), overlays `vendor/overlays/gaiamock_mod.py`, and
compiles `kepler_solve_astrometry.so`. Checksums: [`vendor/DATA_MANIFEST.md`](vendor/DATA_MANIFEST.md).
Do **not** host or require the default ~984 MB `healpix_scans.zip`.

**mwdust Combined19:** when `selection_function_astrometric.extinction_model: combined19`
(and gaiamock `do_dust=True`), the first `mwdust.Combined19()` call downloads Bayestar-based
dust maps into the local mwdust cache. Expect a one-time network fetch before dust-on mocks
run; subsequent calls reuse the cache.

## Test ladder

| Marker | Merge-required? | Purpose |
|---|---|---|
| `unit` | yes | schemas, run_management, constants, loaders |
| `physics` | yes | analytic / closed-form checks |
| `api` | yes | cross-module I/O contracts; registry `inputs_from` |
| `gaiamock` | no | overlay install + minimal RUWE smoke (needs Release assets) |
| `network` | no | Gaia archive, Sheets, catalog downloads |
| `slow` | no | long suites (diagnostics, robustness, performance) |

```bash
# Required merge gate (≪ 20 min; default GitHub Actions `tests` check):
pytest -m "unit or physics or api"

# Optional (after gaiamock overlay install):
pytest -m gaiamock

# Optional long / network suites (path-filtered or manual / workflow_dispatch):
pytest -m slow
pytest -m network
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §10.

## Usage

Print the full run plan (no stage execution; no new `runs/*.yaml` when `--run-file` is omitted):

```bash
python scripts/run_pipeline.py --dry-run
```

Execute `STAGE_ORDER` (creates or resumes a run file under `runs/`; prints the plan first):

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --config config/config.yaml
python scripts/run_pipeline.py --run-file runs/<run_id>.yaml
python scripts/run_pipeline.py --force-rerun data_acquisition
python scripts/run_pipeline.py --stages data_acquisition triples --dry-run
python scripts/purge_run.py runs/<run_id>.yaml
```

Flags: `--config`, `--run-file`, `--force-rerun STAGE [STAGE ...]`, `--stages STAGE ...`,
`--dry-run`, `--list-stages`. Incomplete runs without `--run-file` print a table and exit
nonzero ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5).

## Operator notes

**Run files (`runs/`).** Each run is a live YAML manifest `runs/{run_id}.yaml`
(`run_id = YYYYMMDD-HHMMSS-<shortgit>`). It *is* the `RunManifest`: config checksum, stage
status, artifact paths, source hashes. Amend vs new-run rules:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5. If incomplete runs exist and `--run-file` is
omitted, the entry point lists them and exits nonzero.

**Purge.** Delete a run YAML (optionally its HDF5 artifacts):

```bash
python scripts/purge_run.py runs/<run_id>.yaml
python scripts/purge_run.py runs/<run_id>.yaml --with-artifacts
# Completed runs require --force
python scripts/purge_run.py runs/<run_id>.yaml --force
```

**Config: fragments vs `config.yaml`.** Draft domain edits under tracked
`config/fragments/<domain>.yaml`. Review/Integration merges into the single canonical
`config/config.yaml` at checkpoints. Runtime and resume checksums use the merged file — do not
point production runs at a fragment alone.

**gaiamock-mod-v1.** Science paths import via `darkhunter_pop.gaiamock_vendor.import_gaiamock_mod()`
only. Version triple (`gaiamock_mod_release`, `gaiamock_mod_sha256`, `gaiamock_git_commit`) is
recorded in config and every run/stage record; mismatch refuses the stage. See
[`docs/GAIAMOCK_API.md`](docs/GAIAMOCK_API.md).

**December cluster / inference smoke.** CI keeps tiny dynesty settings. Full cluster recipe
(nlive / dlogz / maxcall / multi-run robustness) is documented in the header comments of
[`config/fragments/inference.yaml`](config/fragments/inference.yaml) — raise those values for
real sampling; require posterior agreement across independent seeds, not bitwise seed identity
([`docs/ORCHESTRATION_PLAN.md`](docs/ORCHESTRATION_PLAN.md) December note; workflow §7).
