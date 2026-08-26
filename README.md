# dark-hunter_pop

A debiased mass function, dN/dM, for compact-object companions (WD / NS / BH) in Gaia astrometric
binaries. A population model is forward-modeled through a modified-RUWE
[`gaiamock`](https://github.com/kareemelbadry/gaiamock) overlay and compared to the real Gaia NSS
sample with an inhomogeneous Poisson point-process likelihood.

**Status:** scaffold + locked architecture. Foundation implementation waits on approval of the
Foundation task list.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — authoritative pipeline specification.
- [`docs/ORCHESTRATION_PLAN.md`](docs/ORCHESTRATION_PLAN.md) — subagent roster, phasing, git/PR workflow.

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
