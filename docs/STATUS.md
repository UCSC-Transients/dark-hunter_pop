# Status (moved from CLAUDE.md under #652)

> Moved verbatim from the `CLAUDE.md` Status section (content dated 2026-10-04, `main` @ 2b2b30e; stale, refresh at each wave checkpoint). Moved under #652.

## Status (as of 2026-10-04, `main` @ 2b2b30e)

- **Scope: the laptop only.** The objective is the whole workflow running correctly on
  `/Users/rfoley/darkhunter/pop/dark-hunter_pop/`. ziggy and lux are deferred
  (`docs/EXECUTION_PLAN.md` §2), and so is the SED throughput campaign.
- **Ryan's priority order.**
  1. **Step 1 first (#339).** The forward-model mock must reproduce the **full** DR3 NSS orbit
     sample: `Orbital` + `AstroSpectroSB1`, 168,065 rows, the El-Badry et al. (2024) §4 set, no
     NS/BH down-select. Nothing downstream (`population_model`, `inference`, dN/dM figures) is
     trusted until it does.
  2. **Diagnostics must be real.** A figure or gate is computed from what the pipeline actually
     produced. No placeholder fixture (#339), configured stand-in fraction (#344), self-compared
     benchmark (#348) or undeclared synthetic path (#350, #352/#354). Stand-ins register
     themselves in their stage artifact (`run_validity`).
  The chain is mapped in `docs/ARCHITECTURE.md` (`selection_function_astrometric` → "Step 1"). The
  specs are `docs/MOCK_POPULATION_SPEC.md` and `docs/EPOCH_MODEL_SPEC.md`.
- **Step 1 validation ladder** (spec §5):
  - **Rung 1 done** (#390, PR #401 `fcb0d8f`, `docs/gate390/`). 1,296 published orbits × 5
    realizations re-injected. Orbital acceptance 0.742 ± 0.007, AstroSpectroSB1 0.879. Pulls on P,
    e, a0 and ϖ have σ_MAD 0.94–1.02.
  - **Rung 1 follow-ups.**
    - #399 / #398 diagnosed by bit-for-bit replay (PR #404 `c448e74`, `docs/gate399/`). The cascade
      applies DR3's acceleration-first rule exactly. P > 600 d capture is 0.23 and is still open
      (#399).
    - #400 epoch model v1 (PR #412 `bca1624`), then v2 **on** (PR #419 `83c319a`). Against #390:
      N_vis excess +2 → 0, CCD obs / DR3 1.129 → 1.020, Orbital σ ratio 0.89 → 0.97.
    - N2-u0 bright-star noise with RUWE = UWE / u0_mock(G) (PR #422 `686a100`): calibrated and
      off. Ryan adopted it as N2d on 2026-10-03; it is enabled by PR #426, which is open.
    - **#428:** the epoch model gives 0.10% of G < 19 stars below 12 visibility periods,
      against DR3's 1.71% (`docs/EPOCH_MODEL_SPEC.md` §8.9). Bare gaiamock gives 0. Whether to
      model the tail is #432.
  - **Rung 2 paused** (#391, `docs/gate391/`, PR #406 `bd08cab`). Generation 11 stopped at
    154,518 / 370,000 draws (accepted-set ESS 500, 12.3 ESS per CPU-h). Its figures are
    pre-noise-fix and pre-Malmquist: diagnostic only, never a result.
  - **Restart prerequisites since then.**
    - Bounded e proposal, 1-D Malmquist wiring, MP-Q19/Q24/Q25/Q29/Q30 (PR #415 `0ca0a73`).
    - 1-D Malmquist closed loop (PR #411 `d943676`, `docs/gate405/`). With W the twin deficit
      drops from 12% to 3%, and the total is within 0.65%.
    - Giants (PR #417 `b944513`, `docs/gate_giants/`). Weighted mock 9.9% ± 1.4% evolved vs DR3
      Orbital 10.0%, but the evolved six-panel mismatches.
    - MIST isochrone M1 + 2-D CMD weight (PRs #420 `f87f7f4`, #423 `55cd53f`, #424 `4c6d509`,
      `docs/gate418/`). M1 / FLAME on CMD dwarfs is 0.999 with 0.033 dex scatter, and the #393
      floor is gone. The 2-D closed loop is not yet at "every pull ≤ 3" (max 7.4σ).
    - Data-side switch (PR #427 `4e41a59`, off until #425). Ryan's 2026-10-04 decisions
      (MP-Q33–Q39, coeval companions, posterior M1 draws; PR #430 `2b2b30e`, spec §0.4, §11.9) are
      being implemented under #418.
  - **Rungs 3–5 not started.**
- **`selection_function_astrometric` does not use the Step 1 chain yet (#394).** It still draws the
  `mock_population` box prior with no epoch model, so its six-panel / solution-type gate is a
  plumbing check. #368 (`f252b1f`, #344) removed the configured faint-draw short circuit, so every
  mock goes through gaiamock.
- **End-to-end** (#301, PR #338 `4a3e034`, `docs/gate301/`). `runs/20260930-022223-672b092.yaml`
  reaches all 14 stages terminal (13 completed, `triples` skipped) at `main` @ `672b092`: 28.8 min,
  peak RSS 9.77 GiB. Everything after `sample_selection` carries seven declared stand-ins and is a
  plumbing check.
- **Literature reproduction** (measured at `672b092`, `docs/gate301/`; history in
  `docs/SELECTION_REPRODUCTION_STATUS.md`). Andrews moved to the ATF notebook in both modes
  (PRs #309 `d923c83`, #320 `3317336`), and `sub_chandrasekhar` reproduction is the Shahaf 2023b
  class-III cross-match (PR #321 `672b092`).

  | target | published | ours | gate |
  |---|---:|---:|---|
  | Andrews 2022, reproduction | 24 | 25 | FAIL by 1 (`6424213726885519744`) |
  | Andrews 2022 modified, forward model (feeds inference) | — | 24 | |
  | Q9: Andrews `G < 15` (`andrews2022_import`) | 16 | 16 | OK |
  | El-Badry 2024 catalog union / COC | 48 / 21 | 48 / 21 | OK, ID sets exact |
  | El-Badry 2026 union / astrometric | 227 / 76 | 225 / 74 | FAIL by 2 (#335) |
  | El-Badry 2026 spectroscopic | 151 | 151 | N OK; ID sets differ by 8/8 (#335) |
  | `primary_ns_bh` | 47 | 46 | FAIL by 1 (#133) |
  | `sub_chandrasekhar`, reproduction / forward model | 22 / — | 22 / 49 | OK, exact IDs |
  | `elbadry2023_table_e1`; Simon breakdown | 5; 5/2/1/1 | 5; 5/2/1/1 | OK |

  A sample is not enabled in `forward_model` mode for inference until its reproduction matches
  exactly. The spuriousness model is not working until one parameter set reproduces all three
  published rates.
- **Companion-nature evidence is still analytic in practice.** The `phot_sed` WD leg is wired
  (#206 closed), but few Path-2 fits exist, and #207 (cleaning provenance) is open.
- **#221 resolved:** 2MASS fan-out is collapsed with the conflicting bands masked (PR #305
  `a5ce757`, option B).
- **`accel_jerk` (roster #22) is blocked**: there is no published selection function. Do **not**
  derive one from the target list; the registry entry stays disabled.
- Tracked run files are `runs/20260920-033431-121d6de.yaml`, `runs/20260928-083639-f944c93.yaml`
  and `runs/20260930-022223-672b092.yaml`. The September runs `20260904-152655-674c989` and
  `20260908-183807-5a1c609` were purged (#318, PR #319 `5347254`; PR #326 `87e81b2`). Do not cite
  them. Worktrees and finished branches are #49's to tear down; **coding sessions never delete
  either**.

