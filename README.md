# dark-hunter_pop

A debiased mass function, dN/dM, for compact-object companions (WD / NS / BH) in Gaia astrometric
binaries. A population model is forward-modeled through a modified-RUWE
[`gaiamock`](https://github.com/kareemelbadry/gaiamock) overlay and compared to the real Gaia NSS
sample with an inhomogeneous Poisson point-process likelihood.

**Status:** Phases 0–6 on `main`; Phase 7 wires the main program (`scripts/run_pipeline.py`).

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — authoritative pipeline specification.
- [`docs/ORCHESTRATION_PLAN.md`](docs/ORCHESTRATION_PLAN.md) — subagent roster, phasing, git/PR workflow.
- [`docs/FOUNDATION_INTERFACE_FREEZE.md`](docs/FOUNDATION_INTERFACE_FREEZE.md) — frozen package interfaces.

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
# Overlay + compile (GSL + gcc; uses mod_files/ if present, else Release):
#   scripts/install_gaiamock_mod.sh
pytest -m "unit or physics or api"
# Optional: pytest -m gaiamock
```

Required CI check runs `unit` + `physics` + `api` only (≪ 20 min). Optional `gaiamock` /
`network` / `slow` suites are path-filtered or manual — see ARCHITECTURE.md §10.

## Usage

Print the full run plan (no stage execution; no new `runs/*.yaml` when `--run-file` is omitted):

```bash
python scripts/run_pipeline.py --dry-run
```

Execute `STAGE_ORDER` (creates or resumes a run file under `runs/`; prints the plan first):

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --run-file runs/<run_id>.yaml
python scripts/run_pipeline.py --force-rerun data_acquisition
python scripts/purge_run.py runs/<run_id>.yaml
```

Flags: `--config`, `--run-file`, `--force-rerun STAGE [STAGE ...]`, `--stages STAGE ...`,
`--dry-run`, `--list-stages`. Incomplete runs without `--run-file` print a table and exit
nonzero (ARCHITECTURE.md §5).
