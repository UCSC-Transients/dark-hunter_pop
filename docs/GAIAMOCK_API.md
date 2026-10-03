# gaiamock_mod public API audit

Parent issue: #4. Vendor install: #3 / PR #17. Audited against submodule commit
`dd30fdbf787eb96878734605ac077ac69bf28c84` and tracked overlay
`vendor/overlays/gaiamock_mod.py` (Release `gaiamock-mod-v1`).

**Hard rule:** science paths import only via
`darkhunter_pop.gaiamock_vendor.import_gaiamock_mod()` (`import gaiamock_mod as gaiamock`).
Do **not** reimplement anything listed under “Provided by gaiamock_mod” in
`physics_utils.py` or elsewhere.

## What gaiamock_mod already provides

### Kepler / photocenter geometry (C + Python)

| API | Role |
|---|---|
| `read_in_C_functions()` | Load `kepler_solve_astrometry.so` |
| `solve_kepler_eqn_on_array` | Kepler equation on arrays |
| `get_astrometric_chi2` | χ² for (P, φ_p, e) + linear params |
| `get_a_mas`, `get_a0_mas` | Angular semi-major / photocenter size |
| `al_bias_binary` | Along-scan photocenter bias for flux ratio |
| `get_Campbell_elements` | Thiele–Innes → Campbell |
| `photocenter_orbit_2d_from_thiele_innes` | Sky-plane photocenter track |
| `get_companion_mass_from_mass_function` | Invert astrometric mass function |

### Scanning law / epoch prediction

| API | Role |
|---|---|
| `get_gost_one_position` | Nearest healpix-16 scan times/angles (`healpix_scans/`) |
| `rescale_times_astrometry` | JD → years relative to DR3/DR4/DR5 reference epoch |
| `predict_astrometry_luminous_binary` | Simulate epoch AL observations for a luminous binary |
| `predict_astrometry_binary_in_terms_of_a0` | Same, parameterized by `a0_mas` |
| `predict_astrometry_single_source` | Single-star epoch astrometry |
| `get_realistic_epoch_astrometry_errors` | **Mod-only:** position-dependent RUWE rescaling via `healpix_16_med_ruwe.npz` |
| `al_uncertainty_per_ccd_interp` | Per-CCD AL uncertainty vs G |

### RUWE and 5/7/9/12-parameter cascade

| API | Role |
|---|---|
| `check_ruwe` | 5-parameter RUWE |
| `get_5par_solution_and_sigma_5d_max` | 5-par solution + `sigma5d_max` |
| `check_7par`, `check_9par` | Acceleration / jerk solutions |
| `fit_5par_solution_only` | 5-par fit only |
| `fit_full_astrometric_cascade` | Full cascade → solution-type assignment |
| `run_full_astrometric_cascade` | Predict epochs then run cascade (end-to-end mock) |
| `run_only_5par_solution` | End-to-end 5-par only |
| `fit_orbital_solution_nonlinear` | 12-par orbital nonlinear fit |
| `mcmc_fit_with_thiele_innes_elements` / `mcmc_fit_with_campbell_elements` | MCMC orbit refinement |

### Population / mock helpers (usable by selection_function_astrometric)

| API | Role |
|---|---|
| `draw_from_exponential_disk`, `generate_coordinates_at_a_given_distance_exponential_disk` | Sky sampling |
| `xyz_to_galactic`, `xyz_to_radec` | Coordinate transforms |
| `simulate_many_realizations_of_a_single_binary` | Monte Carlo detectability |
| `predict_radial_velocities`, `predict_astrometry_and_rvs_simultaneously` | Joint astrometry+RV prediction (RV gate may use elsewhere) |

### Plotting / diagnostics (optional)

`plot_residuals*`, `plot_2d_orbit_and_residuals` — prefer `darkhunter_pop.plotting` for product figures; these are fine for notebooks.

## Mod vs stock differences (do not “fix” by calling stock)

- **Unbinned CCD-level** measurements (no FOV binning) — slower, better RUWE variance.
- **Empirical sky-dependent** epoch error rescaling (`healpix_16_med_ruwe.npz`).
- Stock `gaiamock.py` APIs with `binned=` / parallax–eccentricity prior cascade helpers are **not** the science path; do not mix.

## Residual scope for `physics_utils.py` (#13)

Implement **only** what gaiamock_mod does not own:

1. **Unit conversions** used across the pipeline (mas ↔ AU, deg ↔ rad helpers not already trivial via astropy, Julian-date conventions *outside* gaiamock’s `rescale_times_astrometry` when talking to other codes).
2. **Inhomogeneous Poisson point-process primitives** for the population likelihood (intensity integrals, log-likelihood terms, empty-bin upper limits) — not present in gaiamock.
3. **Thin wrappers** that call `import_gaiamock_mod()` and adapt outputs into our schemas (`ParameterSet`, candidate tables) — adapters live with stages; shared numeric glue may live here only if truly shared.

Explicitly **out of** `physics_utils.py`:

- Kepler solver / orbital photocenter math
- RUWE / cascade / scanning-law / epoch simulation
- Astrometric mass-function inversion already in `get_companion_mass_from_mass_function`
- Dust / Galactic density sampling already in gaiamock helpers (unless we later replace the exponential-disk prior with our own population model draws — that replacement belongs in `population_model` / `forward_model`, not a reimplementation of Kepler)

## Import checklist for stage authors

```python
from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod, read_versions

gaiamock = import_gaiamock_mod()  # verifies sha256 + .so present
versions = read_versions()        # record into run file
```

## Random number generation (#371)

gaiamock_mod takes no seed or `Generator` argument. It draws from two process-global
RNGs our own `np.random.Generator` streams never reach:

| RNG | Draws in the mock path |
|---|---|
| numpy legacy global state (`np.random.*`) | sky positions (`generate_coordinates_at_a_given_distance_exponential_disk`, `draw_from_exponential_disk`); the random 10% transit loss and per-transit epoch noise (`predict_astrometry_luminous_binary`); per-source epoch-error jitter (`get_realistic_epoch_astrometry_errors`) |
| libc `rand()` in `kepler_solve_astrometry.so` | adaptive simulated-annealing proposals and acceptance in `run_astfit` / `run_astfit_reject_outlier` (the nonlinear orbit-fit search). `srand` is never called upstream, so the stream depends on how many fits ran earlier in the process |

`check_ruwe` / `check_7par` / `check_9par` / the Jacobian uncertainties are deterministic.
`simulate_many_realizations_of_a_single_binary` and `get_astrometric_likelihoods` use joblib
workers; pop does not call either.

Never edit gaiamock to take a seed. Seed at the call boundary: wrap every gaiamock call in
`forward_model.seeded_global_rng(mock_global_rng_seeds(base_seed, stream, index), c_funcs)`.
Seeds come from `SeedSequence(entropy=base_seed, spawn_key=(stream, index))`, so a realization
depends only on the seed and its index, not on call order or worker count; numpy's global
state is restored on exit. A worker process must enter the block itself. The scheme string
(`MOCK_RNG_SEED_SCHEME`) and per-realization seeds are written to the
`selection_function_astrometric` artifact (`mock_catalog/truth/rng_seed_*`) and the run
manifest (`random_seeds["selection_function_astrometric.mock_rng"]`).

## Epoch model around the GOST list (#400)

`get_gost_one_position` returns GOST rows for the nearest of 3,072 HEALPix-16 positions,
10 rows per FoV transit (SM + AF1–AF9; 9 in CCD row 4). `predict_astrometry_*` then drops
each **row** with probability 0.1. El-Badry et al. (2024) §3.3 meant a 10% loss of FoV
transits, but the unbinned overlay applies it per CCD row: no whole transits are lost, and
the result reproduces DR3's ~8.7 AF CCDs per transit to 1–2%. GOST also knows nothing about
DR3's data gaps. gaiamock therefore has ~11–13% more FoV transits and +2 visibility periods
compared with DR3 (#400, #398; `docs/gate400`).

`darkhunter_pop.epoch_model` corrects this **without editing or reimplementing gaiamock**.
`gost_epoch_model(gaiamock, config, SourceEpochContext(g_mag), rng)` temporarily replaces
the module attribute `gaiamock.get_gost_one_position` with a wrapper that calls the original
and then (1) removes rows outside the AGIS window and inside the 138 published DR3
astrometric gaps, and (2) drops whole FoV transits with a G-dependent probability calibrated on
DR3. The original is restored on exit. Everything downstream, including the 10% row drop,
noise and the cascade, is gaiamock's own code. The thinning uses its own seeded
`Generator` (`epoch_model_rng`), so the #371 seeding is untouched. Config:
`dr3.epoch_model` (off by default; `dr4.epoch_model: null`). Spec: `docs/EPOCH_MODEL_SPEC.md`.
The patch is a module attribute, so it is process-local and not thread-safe; every pop driver
already uses one process per worker.
