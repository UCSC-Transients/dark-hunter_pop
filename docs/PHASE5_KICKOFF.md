# Phase 5 kickoff (Agents Window paste prompts)

**Priority phase** — December supercomputer cutoff. Land a minimal Poisson + dynesty end-to-end path early.

Phase 4 (#55) complete (PRs #60, #61). Freeze: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#62](https://github.com/UCSC-Transients/dark-hunter_pop/issues/62).  
Review/Integration: [#64](https://github.com/UCSC-Transients/dark-hunter_pop/issues/64).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#63](https://github.com/UCSC-Transients/dark-hunter_pop/issues/63) | #11 inference | Top | `phase5/inference` |

PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

---

## Slot A — inference (#63)

```
You are a Phase 5 subagent for UCSC-Transients/dark-hunter_pop (top-tier). DECEMBER PRIORITY.

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/63
Branch from latest main: phase5/inference (git worktree).
PR base: main. Closes #63 alone on a line. Labels: phase-5, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/ARCHITECTURE.md §4 inference
- docs/ORCHESTRATION_PLAN.md Phase 5
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement inference in inference.py:
- v1 staged-but-connected: fixed weights from rv_astrometry_gate + companion_nature_likelihood (no joint re-sampling).
- Inhomogeneous Poisson point process: rate = population_model × astrometric_SF × followup_SF; use physics_utils Poisson primitives.
- Honor sensitivity_analysis dimensionality advice (prefer unbinned/small-N).
- Sampler: dynesty. Document multi-run robustness protocol (not bitwise seeds).
- Small-N: posterior-vs-prior overlap checks; Poisson upper limits on zero-count bins.
- Model-comparison config switches: SN-kick e/mass vs none; circular-implies-WD vs none.
- Config fragment; register stage + inputs_from; HDF5/products under config paths.
- CI: short dynesty smoke only; full cluster recipe in docs/comments, not default CI.
- First PR may use simplified population call path if needed — Poisson×SF×dynesty loop must be real and run_management-friendly.

Ask before freeze breaks. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Review/Integration (#64)

```
You are the continuous Review/Integration subagent (roster #15, top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/64

When the inference PR is open: merge config/fragments into config/config.yaml; check contracts vs FOUNDATION_INTERFACE_FREEZE and population_model/SF stages; keep CI dynesty smoke bounded. Prefer review + small integration PRs. Docs-first before freeze breaks.
```

---

## Orchestrator notes

After #63 (+ optional integration): Phase 6 — Validation/SBC + full diagnostic suite (roster #13). Then Phase 7 — main program wiring + documentation.
