# Phase 3 kickoff (Agents Window paste prompts)

Phase 2 (#35) complete; config checkpoint merged (PR #46). Freeze: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#47](https://github.com/UCSC-Transients/dark-hunter_pop/issues/47).  
Review/Integration: [#50](https://github.com/UCSC-Transients/dark-hunter_pop/issues/50).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#48](https://github.com/UCSC-Transients/dark-hunter_pop/issues/48) | #6 rv gate + joint fit | Top | `phase3/rv-consistency` |
| B | [#49](https://github.com/UCSC-Transients/dark-hunter_pop/issues/49) | #8 triples stub | Mid | `phase3/triples-stub` |

Keep ≤3 concurrent. PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

---

## Slot A — rv_astrometry_gate + joint_orbit_fit (#48)

```
You are a Phase 3 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/48
Branch from latest main: phase3/rv-consistency (git worktree).
PR base: main. Closes #48 alone on a line. Labels: phase-3, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/ARCHITECTURE.md §4 rv_astrometry_gate + joint_orbit_fit
- docs/ORCHESTRATION_PLAN.md
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement both stages in rv_consistency.py. Gate: fix astrometric P,e,T,K,ω; fit γ+jitter per instrument; chi2/dof vs config threshold; SB2 routing per architecture. Joint fit: separate stage after gate; passers get free elements + Joker-seeded simultaneous fit + OrbitTier joint_astrometry_rv; failures skipped with reason rv_astrometry_gate_failed (keep astrometry_only). Consume dark-hunter_rv JSON; ParameterSet/CandidateRecord fields; config fragment; register + inputs_from; HDF5 via run_management. Ask before freeze breaks. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot B — triples stub (#49)

```
You are a Phase 3 subagent for UCSC-Transients/dark-hunter_pop (mid-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/49
Branch: phase3/triples-stub (worktree). PR → main. Closes #49 alone on a line. Labels: phase-3, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 triples, four skills.

Implement triples as an off-by-default stub stage/module: enable later without restructuring; interface hooks for TESS variability + rotation-consistency vs uberMS v sin i; config default off; clean skip/no-op in run_management when disabled. No real triples science. Full pytest before PR. Stop when PR open + CI green.
```

---

## Review/Integration (#50)

```
You are the continuous Review/Integration subagent (roster #15, top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/50

When Phase 3 PRs open: merge config/fragments into config/config.yaml; check consistency vs FOUNDATION_INTERFACE_FREEZE; watch collisions on config_schema / run_management / rv_consistency. Prefer review comments + small integration PRs. Docs-first before freeze breaks.
```

---

## Orchestrator notes

After #48–#49 (+ optional integration): Phase 4 — roster #7 `companion_nature_likelihood`, #9 `population_model` (both Top). Then prioritize Phase 5 `inference` toward December cluster cutoff.
