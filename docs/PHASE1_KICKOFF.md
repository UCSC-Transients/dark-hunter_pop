# Phase 1 kickoff (Agents Window paste prompts)

Foundation Phase 0 is closed (#2). Interfaces: `docs/FOUNDATION_INTERFACE_FREEZE.md`.

Umbrella: [#28](https://github.com/UCSC-Transients/dark-hunter_pop/issues/28).

| Slot | Issue | Roster | Tier | Branch suggestion |
|------|-------|--------|------|-------------------|
| A | [#29](https://github.com/UCSC-Transients/dark-hunter_pop/issues/29) | #1 `data_acquisition` | Mid | `phase1/data-acquisition` |
| B | [#30](https://github.com/UCSC-Transients/dark-hunter_pop/issues/30) | #2 `selection_function_astrometric` | Top | `phase1/selection-function-astrometric` |
| C | [#31](https://github.com/UCSC-Transients/dark-hunter_pop/issues/31) | #5 rv/sed extensions | Mid | rv/sed repos + optional pop docs |

Keep ≤3 concurrent Agents Window sessions. Each session: own git worktree. PRs target **`main`** (or the sibling repo’s default branch for #31). Issue close keyword: `Closes #N` alone on a line.

Skills every response: `.cursor/skills/strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

---

## Slot A — data_acquisition (#29)

```
You are a Phase 1 subagent for UCSC-Transients/dark-hunter_pop.

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/29
Branch from latest main: phase1/data-acquisition (use a git worktree).
PR base: main. Body must include Closes #29 on its own line. Labels: phase-1, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/ARCHITECTURE.md §4 data_acquisition
- docs/ORCHESTRATION_PLAN.md
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement stage data_acquisition only. Register with run_management; one HDF5 under paths.artifact_root; CandidateRecord rows for fields this stage owns. Quality cuts from config.dr3.quality_cut_bins (N bins). Snapshot ADQL + checksum + date under data/dr3/gaia_snapshots/. No hardcoded science numbers. Full local pytest (unit|physics|api) before PR. Network tests mark @pytest.mark.network. Stop when PR is open and CI green; do not start other stages.
```

---

## Slot B — selection_function_astrometric (#30)

```
You are a Phase 1 subagent for UCSC-Transients/dark-hunter_pop (top-tier judgment).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/30
Branch from latest main: phase1/selection-function-astrometric (worktree).
PR base: main. Closes #30 alone on a line. Labels: phase-1, enhancement.

Read and obey:
- docs/FOUNDATION_INTERFACE_FREEZE.md
- docs/GAIAMOCK_API.md
- docs/ARCHITECTURE.md §4 selection_function_astrometric
- .cursor/skills/strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow

Implement selection_function_astrometric wrapping gaiamock_mod ONLY via darkhunter_pop.gaiamock_vendor.import_gaiamock_mod(). Record gaiamock version triple; refuse mismatch. Include El-Badry et al. (2024) six-panel mock-vs-real validation gate + solution-type fraction diagnostic. DR3 first; do not enable DR4 execution. Config fragments for new keys. @pytest.mark.gaiamock where overlay needed. Full required pytest before PR. Stop when PR open + CI green.
```

---

## Slot C — dark-hunter_rv / dark-hunter_sed (#31)

```
You are a Phase 1 subagent for roster #5 (sibling repos).

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/31
Work in UCSC-Transients/dark-hunter_rv and UCSC-Transients/dark-hunter_sed (separate PRs). Optional tiny docs/adapter PR in dark-hunter_pop only if needed — do not break FOUNDATION_INTERFACE_FREEZE without asking.

Conform CandidateRecord.rv_summary / JSON contract in pop schemas. Ask before schema breaks.

Deliver:
1. dark-hunter_rv: JSON summary replacing sectioned summary.txt (Joker already present — wire output, not a new sampler).
2. dark-hunter_sed: WISE + DECam-u in gather_phot.
3. Document JSON fields in the rv PR (and short pop doc only if useful).

Skills: strict-workflow, regression-hunter, caveman, dark-hunter-pop-workflow (+ each repo’s norms). Full local pytest per repo before each PR. Reference pop#31 in PR bodies (Refs UCSC-Transients/dark-hunter_pop#31). Close #31 only when both sibling deliverables land (or split follow-ups if you open finer issues first).
```

---

## Orchestrator notes

After all three PRs merge: open Phase 2 tickets (#3 followup SF, #4 mass derivation, #10 sensitivity, #12 plotting) per ORCHESTRATION_PLAN §5. Review/Integration (#15 roster) runs continuously from Phase 1 onward — create that ticket when the first Phase 1 PR is ready for cross-check.
