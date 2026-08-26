# Phase 7 kickoff (Agents Window paste prompts)

Phase 6 (#68) complete (SBC, benchmarks, diagnostics; integration #77). Freeze: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#78](https://github.com/UCSC-Transients/dark-hunter_pop/issues/78).  
Review / final checkpoint: [#81](https://github.com/UCSC-Transients/dark-hunter_pop/issues/81).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#79](https://github.com/UCSC-Transients/dark-hunter_pop/issues/79) | #16 main program | Mid | `phase7/run-pipeline` |
| B | [#80](https://github.com/UCSC-Transients/dark-hunter_pop/issues/80) | #14 documentation | Light | `phase7/documentation` |

Keep ≤3 concurrent. PRs → **`main`**. `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

Run-plan / diagnostic operator text: **full detail** (caveman exemption).

---

## Slot A — run_pipeline (#79)

```
You are a Phase 7 subagent for UCSC-Transients/dark-hunter_pop (mid-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/79
Branch from latest main: phase7/run-pipeline (git worktree).
PR base: main. Closes #79 alone on a line. Labels: phase-7, enhancement.

Read FOUNDATION_INTERFACE_FREEZE, ARCHITECTURE.md §5 (run management / required screen output), ORCHESTRATION_PLAN roster #16, four skills.

Implement scripts/run_pipeline.py: load config; create/resume runs per frozen amend/force-rerun/--run-file rules; PRINT full run plan before any stage; execute STAGE_ORDER by calling existing stage runners only. CLI: config, run-file, force-rerun, dry-run. Unit/api tests for plan + dry-run (no live Gaia in default CI). Update README Usage with the entry command. Ask before freeze breaks. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot B — documentation (#80)

```
You are a Phase 7 subagent for UCSC-Transients/dark-hunter_pop (light-tier).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/80
Branch: phase7/documentation. PR → main. Closes #80 alone on a line. Labels: phase-7, documentation, enhancement.

Refresh README status/install/test ladder (unit|physics|api, gaiamock, slow, network); note mwdust Combined19 download; link docs index (ARCHITECTURE, ORCHESTRATION_PLAN, FREEZE, GAIAMOCK_API, kickoffs). Operator notes: runs/, purge_run, fragments vs config.yaml, gaiamock-mod-v1 Release. Do not rewrite locked architecture decisions. Sync Usage with run_pipeline once #79 exists (coordinate or follow). Full required pytest before PR if any code touched; docs-only OK. Stop when PR open + CI green.
```

---

## Review / final checkpoint (#81)

```
You are the Review/Integration subagent (roster #15, top-tier) for the Phase 7 final checkpoint.

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/81

After #79/#80: merge fragments if any; sync FOUNDATION_INTERFACE_FREEZE status to Phases 0–7 complete (accurate wording); verify run_pipeline against freeze §5 behaviors; small integration PR if needed. Docs-first before any freeze break.
```

---

## Orchestrator notes

Phase 7 closes the planned roster. After merge: optional human checklist (cluster dynesty recipe, real Gaia snapshot run) outside ticket scope unless you open follow-ups.
