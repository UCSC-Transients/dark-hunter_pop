# dark-hunter_pop

A debiased mass function, dN/dM, for compact-object companions (WD / NS / BH) in Gaia astrometric
binaries. A population model is forward-modeled through [`gaiamock`](https://github.com/kareemelbadry/gaiamock)
and compared to the real Gaia NSS sample with an inhomogeneous Poisson point-process likelihood.

**Status: scaffold only.** No pipeline logic is implemented yet.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — authoritative pipeline specification: repository
  layout, foundation layer, interface contracts, per-stage design, run management, DR3/DR4 mode
  matrix, config philosophy, and documented v1 limitations.
- [`docs/ORCHESTRATION_PLAN.md`](docs/ORCHESTRATION_PLAN.md) — how the work is divided and
  sequenced: subagent roster, phasing, environment, and git/PR workflow.

## Related repositories

- [`dark-hunter_rv`](https://github.com/UCSC-Transients/dark-hunter_rv) — radial velocities (The Joker).
- [`dark-hunter_sed`](https://github.com/UCSC-Transients/dark-hunter_sed) — SED fitting (uberMS / MIST / Payne).

## Layout

```
config/        merged canonical config.yaml (modular fragments during development)
constants/     physical constants and classification-threshold defaults
runs/          per-run YAML run files (the live run manifest)
src/darkhunter_pop/   the package: one file per stage, plus the foundation layer
vendor/gaiamock/      pinned git submodule
tests/         pytest suite
scripts/       entry point that calls stages in order via run_management
notebooks/     gaiamock / El-Badry reproduction notebooks
docs/          specification and design documents
```

## Install

Environment management is `venv` (`ORCHESTRATION_PLAN.md` §2). Linux is the primary target;
macOS-friendly where reasonable.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot,inference]"
pytest
```
