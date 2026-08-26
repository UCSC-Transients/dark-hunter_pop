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
