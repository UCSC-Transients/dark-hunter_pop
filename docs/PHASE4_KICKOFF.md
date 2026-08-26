# Phase 4 kickoff (Agents Window paste prompts)

Phase 3 (#47) complete (PRs #54, #53, freeze sync #52). Freeze: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#55](https://github.com/UCSC-Transients/dark-hunter_pop/issues/55).  
Review/Integration: [#58](https://github.com/UCSC-Transients/dark-hunter_pop/issues/58).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#56](https://github.com/UCSC-Transients/dark-hunter_pop/issues/56) | #7 companion_nature_likelihood | Top | `phase4/companion-nature` |
| B | [#57](https://github.com/UCSC-Transients/dark-hunter_pop/issues/57) | #9 population_model | Top | `phase4/population-model` |

Keep ≤3 concurrent. If weight-schema risk: land Slot A first or a tiny shared-contract PR. PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

**December note:** Phase 5 `inference` is next and time-critical — keep interfaces clean for a minimal dynesty path even against a simplified population.

---

## Slot A — companion_nature_likelihood (#56)

```
You are a Phase 4 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/56
Branch from latest main: phase4/companion-nature (git worktree).
PR base: main. Closes #56 alone on a line. Labels: phase-4, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/ARCHITECTURE.md §4 companion_nature_likelihood
- docs/ORCHESTRATION_PLAN.md
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement companion_nature_likelihood in companion_nature.py. Continuous joint multi-band likelihood over M2 nature (photometry ΔBIC, XP, SB2 when present). Bédard tracks from local config path (not runtime fetch). Config thresholds only. Weights for population_model — never pre-filter/discard. Two-tier fast/full. Age-bin diagnostic required. Register stage + inputs_from; HDF5 via run_management. Ask before freeze breaks. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot B — population_model (#57)

```
You are a Phase 4 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/57
Branch: phase4/population-model (worktree). PR → main. Closes #57 alone on a line. Labels: phase-4, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 population_model, four skills.

Implement population_model: multiplicity layer with P(single)=P(triple)=0 in v1; binary type mixture BH/NS/WD/other/outlier as rate functions; consume companion_nature weights + sensitivity_analysis covariate recommendations; M_Ch hard / M_TOV soft rules; two-tier dN/dM outputs; non-parametric free-height bins (+ optional GP-on-log swap); no external CO MF priors. Coordinate weight schema with #56 early. Config fragment; register; HDF5. Full pytest before PR. Stop when PR open + CI green.
```

---

## Review/Integration (#58)

```
You are the continuous Review/Integration subagent (roster #15, top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/58

When Phase 4 PRs open: merge config/fragments into config/config.yaml; check companion_nature ↔ population_model weight contract vs FOUNDATION_INTERFACE_FREEZE; watch config_schema / run_management collisions. Prefer review + small integration PRs. Docs-first before freeze breaks.
```

---

## Orchestrator notes

After #56–#57 (+ optional integration): **Phase 5 priority** — roster #11 `inference` (dynesty + Poisson point process), even against a placeholder/simplified population_model, to validate end-to-end on the December cluster target early.
