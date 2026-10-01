# Gate 301: full end-to-end run with every diagnostic (issue #301)

> **The sample-reproduction counts below are real measurements** on real Gaia DR3 NSS rows.
> **Everything downstream of `sample_selection` is still a plumbing check.** That covers the
> companion-nature weights, the selection functions, the population model, inference and the
> dN/dM figure. The run carries seven declared synthetic stand-ins (listed in
> `dry_run_report.txt`), so no number from those stages may be quoted in a figure, caption, talk or
> paper.

| | |
|---|---|
| Run file | `runs/20260930-022223-672b092.yaml` (tracked; `dry_run: true`) |
| Code | `main` @ **`672b092`** (primary checkout, real `data/`, gaiamock overlay installed) |
| Entry point | `.venv/bin/python scripts/run_dry_run.py --host-profile laptop --config docs/gate301/config_gate301.yaml --snapshot data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml` |
| Config | `docs/gate301/config_gate301.yaml` = `config/config.yaml` @ `672b092` with **one** change: `diagnostics.sbc.run_in_stage: true`, so that the SBC hook emits too. `diagnostics` is outside the resume checksum, so `config_checksum` is identical to `main`'s: `18a6427e6e603a2d0235403bee97c153bdc80775540f24feec65576710072a35` (verified). |
| Parent snapshot | uncut `20260826T234425Z_3d3f740b080c` (replayed offline) |
| Wall clock / peak RSS | 1730 s (28.8 min); `getrusage` max RSS **9.77 GiB** (10,491,707,392 B); peak memory footprint 10.70 GiB |
| Date | 2026-09-29 (run_id timestamps are UTC) |

Why `run_dry_run.py` rather than `run_pipeline.py`: it is the only entry point that records per-stage
RSS, and it replays the local snapshot. `run_pipeline.py` cannot replay a snapshot (#219). It runs
the same fourteen registered stage runners.

## Cache freshness

- **Andrews ATF-notebook sidecars rebuilt fresh for this run** (#327: their fingerprint covers no
  code). The old files were first moved aside, not deleted, to the session scratchpad. Then
  `.venv/bin/python scripts/build_andrews2022_atf_columns.py --workers 8` rebuilt both modes in one
  invocation: 589 s, max RSS 4.13 GiB. The fingerprints are unchanged, as expected, since they hash
  no code. The contents were rebuilt:

  | Sidecar | Old SHA-256 (moved aside) | New SHA-256 |
  |---|---|---|
  | `atf_notebook_reproduction_be3ee5f2648bd3bd.h5` | `9f3c52e1…ac08` | `bd5f7114…2dd3e` |
  | `atf_notebook_forward_model_f3162aea292c9f1e.h5` | `a589dc43…6246` | `5253a889…5fba` |
  | `atf_notebook_4181870cbce6ac8f.h5` (legacy, mode-less name) | `48822746…3f37` | not rebuilt, and moved back unchanged: no stage reads this name, but the VizieR Apsis snapshot's `id_source_sidecar` provenance points at it |

  Build log: pass-1 survivors (`atf_m2_probability`) = 115; 0 lacking a VizieR row; 112/115 with a
  pipeline TAG10 M1; covariance failures `singular` 2747, `nonfinite_input` 1839; pass-1 root
  failures 456.
- **E(B-V) cache: warm, not cold.** `sample_selection` read
  `data/dust_maps/ebv_cache/elbadry2026_30a69867aa069bca.h5` (written 2026-09-27). Its fingerprint
  (`dust_maps.extinction_fingerprint`) hashes `dust_maps.py`'s own source, the map md5s and the
  north/south split, so reusing it is code-safe. It is not a stale reuse. A cold rebuild was
  attempted by moving the cache aside, but the session permission system refused the move, so the
  cold ~12.5 GiB figure was **not** re-measured here (#276 holds the last measurement, 11.6 GB).
- **Reused as-is, for provenance:** the `+enrich+mc10000/selection_parent_rows.h5` selection-parent
  cache (mtime 2026-09-27 02:18; this cache records no MC provenance, #147) and the frozen
  `nss_enrichment` snapshot (2026-09-04). Every stage artifact is new under
  `output/20260930-022223-672b092/`. No stage reported "cached".

## Reproduction targets: ours vs published

Counting rule #274: distinct stars per `nss_solution_type`, summed over types. Every number was
measured at **`672b092`**. "In stage" means the `sample_selection` artifact of this run. "Out of stage" means
`docs/gate301/measure_flipped_modes.py`, which re-evaluates the *same parent rows* with each sample's
mode flipped, because the stage runs each sample in one mode only (#336). Its output is in
`flipped_modes.yaml` (525 s, max RSS 7.84 GiB).

| Target | Mode | Published | **Ours @ `672b092`** | Per-type distinct | Gate |
|---|---|---:|---:|---|---|
| Andrews 2022 | reproduction (in stage) | 24 | **25** | `Orbital` 25 | FAIL by 1: `6424213726885519744` extra (§3.1.5) |
| Andrews 2022 | forward_model (out of stage) | — | **23** | `Orbital` 23 | info |
| Andrews 2022 modified | reproduction (out of stage) | 25 | **26** | `Orbital` 26 | FAIL by 1 (same extra source) |
| Andrews 2022 modified | forward_model (in stage) | — | **24** | `Orbital` 24 | info; the in-stage report wrongly marks this `n_match: False` against 25 (#336) |
| `mode_divergence` andrews2022 vs modified | — | only_right = Gaia BH1 | only_right `4373465352415301632`; only_left `1749013354127453696`, `3649963989549165440` | — | **OK**, matches the configured expectation |
| El-Badry 2024 catalog union | reproduction | 48 | **48** | `Orbital` 42, `AstroSpectroSB1` 6 | **OK** |
| El-Badry 2024 published COC | reproduction | 21 | **21** | `Orbital` 18, `AstroSpectroSB1` 3 | **OK**, source-ID set exact (first re-measurement since `c905575`) |
| El-Badry 2024 union | forward_model (out of stage) | — | 48 | — | info |
| El-Badry 2026 published union | reproduction | 227 | **225** (240 rows) | `Orbital` 64, `AstroSpectroSB1` 10, `SB1` 150, `SB1C` 1 | FAIL by 2 (#335) |
| El-Badry 2026 astrometric branch | reproduction | 76 | **74** | `Orbital` 64, `AstroSpectroSB1` 10 | FAIL by 2; missing `6054379247042197504`, `6152333294796189568` (#335) |
| El-Badry 2026 spectroscopic branch | reproduction | 151 | **151** | `SB1` 150, `SB1C` 1 | N **OK**; source-ID set differs by 8 / 8 (#335) |
| `primary_ns_bh` | reproduction | 47 | **46** | `Orbital` 42, `AstroSpectroSB1` 4 | FAIL by 1 (#133) |
| `elbadry2023_table_e1` | reproduction | 5 | **5** | `Orbital` 3, `AstroSpectroSB1` 2 | **OK** |
| `andrews2022_import` (Andrews `G < 15`, Q9) | reproduction | 16 | **16** | `Orbital` 16 | **OK** |
| `sub_chandrasekhar` | reproduction | 22 | **22** | `Orbital` 18, `AstroSpectroSB1` 4 | **OK** (Shahaf 2023b class III + Eq. 5; exact IDs per §3.3.7) |
| `sub_chandrasekhar` | forward_model (out of stage) | — | **49** | — | info (v2 chain) |
| El-Badry 2026 forward-model union / astrometric / spectroscopic | forward_model (out of stage) | — | 253 / 102 / 151 | — | info; `primary_ns_bh` 46, `table_e1` 5, `andrews2022_import` 14 |
| Spectro routes: MS min / high f_m / both | reproduction | 136 / 30 / 15 | **132 / 30 / 11** | — | FAIL on the split (#335) |
| Simon 2026 exclusion breakdown | reproduction | 5 / 2 / 1 / 1 | **5 / 2 / 1 / 1**, 0 unclassified, `in_sample` 11 | — | **OK** |

El-Badry 2026 extinction outcomes: `ok` 318,964 · `beyond_map_limit` 30,446 · `invalid_parallax`
189. These are identical to `1058e29`.

**What moved since the last measurements:** Nothing moved an order of magnitude, collapsed to zero
or appeared from zero. Against the dispatch table (Andrews 25 / 23, modified 26 / 24,
`sub_chandrasekhar` 22 / 49), every number reproduces. The El-Badry 2026 union falls from 1507 to
225, and the astrometric union from 1356 to 74, because `sub_chandrasekhar` now reproduces exactly
(schema_version 3, #315).

## Stage status and peak RSS

`rss_high_water` is the `getrusage` process-lifetime mark at stage end (monotonic, so it cannot rank
later stages). `rss_added` is the increase across the stage.

| Stage | Status | Wall clock | rss_added | rss_high_water | Note |
|---|---|---:|---:|---:|---|
| `data_acquisition` | completed | 585.2 s | 9.63 GiB | **9.77 GiB** | 2.3 GB artifact. Higher than Wave 0's 7.36 GiB (the #308 enrichment join) |
| `mass_derivation_bulk` | completed | 59.8 s | 0 | 9.77 GiB | |
| `sample_selection` | completed | 198.9 s | 0 | 9.77 GiB | warm E(B-V) cache |
| `mass_derivation_refined` | completed | 4.0 s | 0 | 9.77 GiB | **DEGRADED**: 7816/7816 kept bulk M1 (see below) |
| `rv_astrometry_gate` | completed | 4.1 s | 0 | 9.77 GiB | 48 passed / 27 failed / 7741 skipped |
| `joint_orbit_fit` | completed | 5.6 s | 0 | 9.77 GiB | |
| `companion_nature_likelihood` | completed | 4.8 s | 0 | 9.77 GiB | analytic surrogate |
| `triples` | skipped | — | — | — | `triples.enabled=false` |
| `selection_function_astrometric` | completed | 720.0 s | 0 | 9.77 GiB | **`validation_gate_passed = False`** (#330) |
| `selection_function_followup` | completed | 0.1 s | 0 | 9.77 GiB | |
| `population_model` | completed | 2.6 s | 0 | 9.77 GiB | |
| `sensitivity_analysis` | completed | 0.2 s | 0 | 9.77 GiB | |
| `inference` | completed | 2.4 s | 0 | 9.77 GiB | CI-scale dynesty |
| `diagnostics` | completed | 137.1 s | 0 | 9.77 GiB | SBC in stage |

**All 14 stages are terminal** (13 completed, 1 skipped with a reason). No stage failed.
The out-of-stage runs peaked at 4.13 GiB (ATF sidecar build) and 7.84 GiB (flipped-mode
measurement). Disk: 13 GiB free before the run, 10 GiB after (`output/20260930-022223-672b092/` is
2.6 GB). The 3 GiB stop threshold was never approached.

## Environment state (reported, not changed)

- **`mass_derivation_refined` DEGRADED.** `darkhunter_sed.batch` does `from darkhunter_rv.summary_paths import …`
  (`dark-hunter_sed/darkhunter_sed/batch.py:11`), and **`darkhunter_rv` is not installed in pop's
  `.venv`** (`import darkhunter_rv` raises `ModuleNotFoundError`; no local checkout under
  `~/darkhunter/`). With `mass_derivation.require_sed_package: false` this is the documented degraded
  mode decided in #181. It did not block the stage. No new issue was filed; see also #180.
- **gaiamock triple prints `sha256=None git_commit=None`.** The manifest records the *config* values,
  and `config/config.yaml` leaves `gaiamock.mod_sha256` / `git_commit` null (#218, open). The
  installed overlay is fine: `read_versions()` gives `sha256=71454d76…f7ef028` (matching
  `vendor/DATA_MANIFEST.md`) and `git_commit=dd30fdbf…c84`. The `selection_function_astrometric`
  artifact records the installed triple. The consequence is that the version-mismatch refusal is
  inert (#218). It did not block anything.

## Diagnostic inventory

Every `diagnostics.hooks.*` flag is `true` in config, and `sbc.run_in_stage` was switched on for
this run. Results from `reports/diagnostics_stage.txt`:

| Hook (`diagnostics.py` / `sample_diagnostics.py`) | Emitted | Files |
|---|---|---|
| `funnel_sky` | yes | report + `funnel_bars.png` (the sky map is emitted by `data_acquisition`) |
| `elbadry_six_panel` | yes, real vs mock | report + 2 figures |
| `fit_tier_coverage` | yes | report + figure (7816 bulk, 0 uberMS) |
| `gate_pass_rate` | yes | report + 2 figures |
| `age_stratified_wd` | yes, but **degenerate**: every bin empty and still reports OK | #334 |
| `triples_robustness` | report only (stub-safe skip, triples off; by design) | |
| `info_gain_followup` | yes | report + 2 figures |
| `sampler_consistency` | yes (1 run, CI) | report + figure |
| `mc_noise_convergence` | yes | report + figure |
| `m2_posterior_convergence` | **NO**: skipped, "no M2PosteriorConvergenceDiagnostic provided" | #332 |
| `solution_type_fractions` (multi-solution real vs mock) | yes: **FAIL** (max \|Δ\| 0.392) | #330 |
| `known_truth_benchmarks` | report only (no plot primitive exists) | Gaia BH1 / BH2 checks pass |
| `comparison_catalogs` | report only (by design) | |
| `sbc_recovery` | report + `.h5` (no plot primitive exists) | coverage 0.695 vs 0.68, passed |
| `sample_attrition_waterfall` | yes, 4 samples | report + 4 figures |
| `sample_reproduction_report` | yes | report + figure; mode-mixing defect #336 |
| `simon2026_exclusion_breakdown` | yes | report + figure, 5 / 2 / 1 / 1 OK |
| `covariance_health` | yes | report + figure (1840 `unpack_failed`, #313) |
| `sample_selection_function` | yes, 12 figures | Andrews and El-Badry 2024 curves are identically 0 (#316; El-Badry 2024 is a catalog cross-match, Q2) |
| `mode_divergence` | yes | report + figure, matches expectation |
| `janssens_segment_occupancy` | **NO**: skipped, "no mg_0 values provided" | #332 |

Stage-owned diagnostics: `data_acquisition` (funnel, RUWE, period and eccentricity histograms, sky
map), `mass_derivation_bulk` (M2 pre-/post-cut), `mass_derivation_refined` (refined report, uberMS
watchlist), `rv_astrometry_gate` (funnel, pass rate, chi2/dof), `companion_nature_likelihood` (funnel,
ΔBIC WD − dark, a single spike at 0 as expected from the analytic surrogate), and the `dry_run`
harness (dN/dM by class, caption, report). All were emitted.

**Configured but not emitted: 2 hooks and 5 stage reports.**
- The `m2_posterior_convergence` and `janssens_segment_occupancy` hooks are never hydrated on a
  real run (#332).
- The report formatters `format_validation_gate_report`, `format_followup_calibration_report`,
  `format_population_model_report`, `format_sensitivity_report` and `format_inference_report` are
  never called, so the forward-model validation gates reach disk only as HDF5 attrs (#331).

### Forward-model validation gates (from HDF5 attrs, since no report is written: #331)

`selection_function_astrometric/7321305b6ab85f92.h5`, n_mock = 500, **`validation_gate_passed = False`**:
six-panel KS passes `P_orb` (p = 0.090), `G` (0.025), `e` (0.436) and `cos i` (0.151), and **fails
`f_m`** (D = 0.998, p = 1.3e-36) and **1/parallax** (D = 0.485, p = 2.5e-3). **Solution-type fractions
fail** (0.392 vs 0.05; the real-side label mapping sends every non-`Orbital*` type to
`insufficient_visibility`). **Multi-solution fails** (combo 0.973: mock combo rates are all 0). Filed
as #330.

### Figures vs `docs/PLOTS.md`

**Not met.** All 41 PNGs were reviewed by eye (the representative ones are copied into
`figures/`). Failures, detailed in **#333**: overlapping rotated category labels (the El-Badry 2026
waterfall is unreadable, and so are the reproduction bars, funnels and top-20 priority);
clipped titles; the six-panel `f_m` and 1/parallax panels are each a single spike in an empty frame;
heavy-tailed histograms without log axes; the sampler logZ bar; axes without units (the SF curves'
raw `m2_msun` / `period_day` / `g_mag`); the wrong `f_m` axis label ("companion mass fraction");
a sky map with no axis labels; and `M_Ch` / `M_TOV` drawn in the same style. The dN/dM product figure
otherwise meets the guide, and carries its stand-ins in the caption.

## Failures and issues filed

No stage failed. Diagnostic and gate failures, each filed on its own:

| Issue | What |
|---|---|
| [#330](https://github.com/UCSC-Transients/dark-hunter_pop/issues/330) | astrometric validation gate fails on real DR3 NSS; the stage completes silently |
| [#331](https://github.com/UCSC-Transients/dark-hunter_pop/issues/331) | 5 stage report formatters never called |
| [#332](https://github.com/UCSC-Transients/dark-hunter_pop/issues/332) | `m2_posterior_convergence` / `janssens_segment_occupancy` never hydrated |
| [#333](https://github.com/UCSC-Transients/dark-hunter_pop/issues/333) | figures fail `docs/PLOTS.md` |
| [#334](https://github.com/UCSC-Transients/dark-hunter_pop/issues/334) | `age_stratified_wd` false green on zero candidates |
| [#335](https://github.com/UCSC-Transients/dark-hunter_pop/issues/335) | El-Badry 2026 astrometric 74 vs 76; spectroscopic ID set 8 / 8 swap; routes |
| [#336](https://github.com/UCSC-Transients/dark-hunter_pop/issues/336) | reproduction report mixes modes; one mode per sample in stage |
| [#337](https://github.com/UCSC-Transients/dark-hunter_pop/issues/337) | `run_dry_run.py --plan-only` writes a run file despite printing "not written". The stray `runs/20260930-020817-672b092.yaml` (`stages: {}`) from this session's plan-only call was purged with `scripts/purge_run.py` (YAML only; no artifacts recorded) |
| [#133](https://github.com/UCSC-Transients/dark-hunter_pop/issues/133) (existing, commented) | `primary_ns_bh` 46 vs 47 |

Existing issues that this run re-confirms: #181 / #180 (SED degraded), #218 (null gaiamock triple),
#313 (1840 `unpack_failed`), #316 (Andrews SF on mocks), #219 (no snapshot replay in
`run_pipeline.py`), #276 (cold E(B-V) RSS; not re-measured).

## #301 acceptance status

- [x] Fresh run, all 14 stages terminal, peak RSS per stage recorded. The ATF sidecars were rebuilt.
      The E(B-V) cache was warm (code-fingerprinted), so the cold ~12.5 GiB figure was not
      re-measured.
- [ ] Every diagnostic enabled **and emitted**: 2 hooks skipped (#332) and 5 stage reports missing (#331).
- [ ] All figures meet `docs/PLOTS.md` (#333).
- [x] Summary table of every reproduction target, with SHA and per-type distinct counts (above).
- [x] Every failure filed as its own issue.

## Files here

| File | What |
|---|---|
| `config_gate301.yaml` | the run config (one-line diff from `main`) |
| `measure_flipped_modes.py`, `flipped_modes.yaml` | the out-of-stage opposite-mode measurement and its output |
| `dry_run_report.txt`, `dndm_by_class_caption.txt` | harness report (stand-ins, per-stage cost, run plan) and product caption |
| `reports/` | every `diagnostics/reports/*.txt` from the run |
| `figures/` | the product figure and the key diagnostic figures (the full set is under `output/20260930-022223-672b092/`, which is gitignored) |
