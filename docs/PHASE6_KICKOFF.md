# Phase 6 kickoff (Agents Window paste prompts)

Phase 5 (#62) complete. Local required + gaiamock suites green. Freeze: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#68](https://github.com/UCSC-Transients/dark-hunter_pop/issues/68).  
Review/Integration: [#72](https://github.com/UCSC-Transients/dark-hunter_pop/issues/72).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#69](https://github.com/UCSC-Transients/dark-hunter_pop/issues/69) | #13 SBC recovery + coverage | Top | `phase6/sbc-recovery` |
| B | [#70](https://github.com/UCSC-Transients/dark-hunter_pop/issues/70) | #13 known-truth + comparison catalogs | Top | `phase6/benchmarks-catalogs` |
| C | [#71](https://github.com/UCSC-Transients/dark-hunter_pop/issues/71) | #13 full diagnostic suite + slow | Mid–Top | `phase6/diagnostic-suite` |

Keep ≤3 concurrent. PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

**CI rule:** required gate stays `unit|physics|api`. Put long multi-injection / large-fixture work under `@pytest.mark.slow` so `pytest -m slow` is non-empty.

Diagnostic reports/captions: **full detail** (caveman exemption).

---

## Slot A — SBC recovery (#69)

```
You are a Phase 6 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/69
Branch from latest main: phase6/sbc-recovery (git worktree).
PR base: main. Closes #69 alone on a line. Labels: phase-6, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 diagnostics (SBC), ORCHESTRATION_PLAN, and the four project skills.

Implement simulation-based calibration: multiple distinct injected mass functions, full recovery through staged inference path, credible-interval coverage across repeats. Config-driven tolerances. Small synthetic tests in unit/physics; fuller suite @pytest.mark.slow. Artifacts via diagnostics/run_management. Full-detail reports. Ask before freeze breaks. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot B — benchmarks + catalogs (#70)

```
You are a Phase 6 subagent for UCSC-Transients/dark-hunter_pop (top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/70
Branch: phase6/benchmarks-catalogs. PR → main. Closes #70 alone on a line. Labels: phase-6, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 diagnostics (known-truth + comparison catalogs), four skills.

Implement Gaia-BH1/BH2 clean-detection and Gaia-BH3 DR3 marginal/non-detection expectations (RUWE≈3.4 documented). Fixture/config truth tables with provenance. Comparison-only loaders/reports for NS-candidate / 156-companions / AMRF / Andrews / Shahaf; pulsar+LIGO MF comparison-only never as priors. Unit/api on fixtures; optional network/slow for live pulls. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot C — diagnostic suite (#71)

```
You are a Phase 6 subagent for UCSC-Transients/dark-hunter_pop (mid–top).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/71
Branch: phase6/diagnostic-suite. PR → main. Closes #71 alone on a line. Labels: phase-6, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §4 diagnostics (required list), extend Phase 2 scaffolding in diagnostics.py/plotting.py.

Complete: fit-tier coverage, age-stratified WD check, triples on/off robustness (safe when disabled), info-gain/follow-up priority, sampler multi-run consistency, mock Poisson-negligibility convergence, gaiamock solution-type fraction, RV gate pass-rate. Use @pytest.mark.slow for long suites. Keep validation gates green. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Review/Integration (#72)

```
You are the continuous Review/Integration subagent (roster #15, top-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/72

When Phase 6 PRs open: merge config/fragments into config/config.yaml; check vs FOUNDATION_INTERFACE_FREEZE; ensure slow suites stay out of default CI. Prefer review + small integration PRs. Docs-first before freeze breaks.
```

---

## Orchestrator notes

After #69–#71 (+ optional integration): **Phase 7** — roster #16 main program (`scripts/run_pipeline.py`) + #14 documentation + final review.
