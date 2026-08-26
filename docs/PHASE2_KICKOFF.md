# Phase 2 kickoff (Agents Window paste prompts)

Phase 1 (#28) complete. Freeze still: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#35](https://github.com/UCSC-Transients/dark-hunter_pop/issues/35).  
Review/Integration (continuous): [#40](https://github.com/UCSC-Transients/dark-hunter_pop/issues/40).

| Wave | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| 1 | [#36](https://github.com/UCSC-Transients/dark-hunter_pop/issues/36) | #3 followup SF + DR audit | Top | `phase2/selection-function-followup` |
| 1 | [#37](https://github.com/UCSC-Transients/dark-hunter_pop/issues/37) | #4 mass derivation | Top | `phase2/mass-derivation` |
| 1 | [#38](https://github.com/UCSC-Transients/dark-hunter_pop/issues/38) | #10 sensitivity_analysis | Top | `phase2/sensitivity-analysis` |
| 2 | [#39](https://github.com/UCSC-Transients/dark-hunter_pop/issues/39) | #12 plotting/diagnostics | Mid | `phase2/plotting-diagnostics` |

Keep ≤3 concurrent. Launch Slot D when a wave-1 slot frees. PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

---

## Slot A — selection_function_followup (#36)

```
You are a Phase 2 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/36
Branch from latest main: phase2/selection-function-followup (git worktree).
PR base: main. Closes #36 alone on a line. Labels: phase-2, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/ARCHITECTURE.md §4 selection_function_followup + DR3/DR4 audit notes
- docs/ORCHESTRATION_PLAN.md
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement selection_function_followup in forward_model.py (do not break selection_function_astrometric). Parametric follow-up SF + survey tiering + Sheets adoption-date mining/weekly snapshot hooks. Implement the DR3/DR4 parameter-independence audit function with tests. Config fragments only; register stage + inputs_from; one HDF5 via run_management. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot B — mass_derivation (#37)

```
You are a Phase 2 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/37
Branch: phase2/mass-derivation (worktree). PR → main. Closes #37 alone on a line. Labels: phase-2, enhancement.

Read and obey FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 mass_derivation_bulk + mass_derivation_refined, and the four project skills.

Implement both stages in mass_derivation.py: TAG10 from constants + config sigma_logM/Santos/method; ParameterSet + FitTier; M2 cut from config; refined path consumes dark-hunter_sed/uberMS (Phase 1 photometry/JSON already landed — wire, don't fork). Watch-list near 3 Msun prior cap. No hardcoded science numbers. Register stages; HDF5 artifacts; diagnostics. Full pytest before PR. Stop when PR open + CI green.
```

---

## Slot C — sensitivity_analysis (#38)

```
You are a Phase 2 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/38
Branch: phase2/sensitivity-analysis (worktree). PR → main. Closes #38 alone on a line. Labels: phase-2, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 sensitivity_analysis, four skills.

Implement sensitivity_analysis stage: (a) joint N-D vs 1D dN/dM recommendation; (b) per-class covariates beyond mass. Enforce mock-injection sigma_MC/sigma_Poisson < mc_noise_threshold (config default 0.1) with required convergence diagnostic. Artifacts consumable later by population_model/inference — do not silently rewrite those modules' defaults. Config fragment; register; HDF5. Full pytest before PR. Stop when PR open + CI green.
```

---

## Slot D — plotting + diagnostics infra (#39)

```
You are a Phase 2 subagent for UCSC-Transients/dark-hunter_pop (mid-tier). Launch when a concurrent slot is free.

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/39
Branch: phase2/plotting-diagnostics. PR → main. Closes #39 alone on a line. Labels: phase-2, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 diagnostics (infrastructure only), four skills.

Implement shared plotting.py primitives and diagnostics.py scaffolding used by stages. Hook points for existing funnel/sky/El-Badry-style outputs; no full SBC design (Phase 6). Diagnostic text stays full-detail. Config paths only. Full pytest before PR. Stop when PR open + CI green.
```

---

## Review/Integration (#40)

```
You are the continuous Review/Integration subagent (roster #15, top-tier) for dark-hunter_pop.

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/40

When Phase 2 PRs are open: merge config fragments into config/config.yaml at checkpoints; check cross-PR consistency vs FOUNDATION_INTERFACE_FREEZE; flag schema/run_management collisions. Prefer review comments + small integration PRs. Docs-first before any freeze break. Do not reimplement stage science.
```

---

## Orchestrator notes

After #36–#39 land: Phase 3 tickets — roster #6 `rv_astrometry_gate` + `joint_orbit_fit`, #8 `triples` stub.
