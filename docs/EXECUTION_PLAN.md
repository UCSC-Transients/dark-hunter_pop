# dark-hunter_pop — Execution Plan to v1 Completion

**Status:** active forward plan. Written 2026-09-11 against `main` @ `94dc390`; revised the same day
after reading `dark-hunter_sed` and `dark-hunter_rv`, and again after operator input on scope,
dispatch, and merge policy.

**Wave −1 is complete** as of 2026-09-13, `main` @ `c905575` — roster #50–#55 all landed, the
reproduction-binding reconciliation is on `main`, and the baseline (full suite, peak RSS, §3
re-measured on `main`) exists. See §7's Wave −1 section for what it measured, §5.6 for the concurrency
cap those measurements set, and the handoff note in umbrella issue **#145**, which the orchestrator
writes per §5.9. **Wave 0 is not dispatched until the operator approves Gate −1.**

**Current authorized scope: Waves −1 through D, on the laptop only.** Waves E–H are planned below but **not
authorized** — see §2.

Companion to `ARCHITECTURE.md` (authoritative technical spec), `ORCHESTRATION_PLAN.md` (Phases 0–7,
roster #0–#16, git/PR workflow) and `CONTINUATION_PLAN.md` (Phase 8, roster #17–#27). Those three
remain authoritative for everything they already cover and are now the **historical record**: their
planned rosters are closed. This document is the **only forward-looking plan**.

Nothing here changes a locked decision. Where this plan needs a decision that `ARCHITECTURE.md` or
`CONTINUATION_PLAN.md` owns, it names the open question and routes it to the escalation path rather
than resolving it inline.

---

## 1. Where the project actually stands

**Built and landed.** All fourteen stages are registered, implemented, and have run against the real
DR3 NSS catalog. `runs/20260904-152655-674c989.yaml` reaches `inference` completed — though
`joint_orbit_fit` was skipped there because no system passed the RV gate (pre-Wave-2), `triples` is
off by config, and `diagnostics` is stuck in `running`. No run has yet finished all fourteen stages
cleanly. The Foundation contracts are frozen, the literature selection layer exists, the spuriousness
model module exists, multi-sample Poisson inclusion is wired into `inference`, SBC and benchmark
infrastructure exist, and Wave-2 RV/SED wiring landed (PRs #136–#138) with 154 real
`Gaia_DR3_*_summary.json` RV summaries staged.

**What is not true yet.** The pipeline runs; it is not yet *trustworthy*. Five things stand between
here and a science result:

0. ~~**`main` does not contain the reproduction fixes it is documented as containing.**~~
   **Resolved in Wave −1.** `SELECTION_REPRODUCTION_STATUS.md` §2 listed six binding fixes as done while
   three of the four commits it cited (`e90f91d`, `78120a7`, `e5a911e`, plus `2e6c53f`) were not on
   `main` — PR #134 merged only `7d5b22e` — so every "Last measured" number in that document had been
   measured on a branch. #141 / PR #150 (`03a452a`) reconciled it without reverting the Wave-2 RV/SED
   work, and #144 re-measured every §3 number on `main` @ `c905575` and rewrote the document with the
   SHA beside each number. Kept here because the *lesson* outlives the defect: a merged PR is not
   evidence that its commits landed, and a number without a SHA beside it is not evidence of anything.
1. **The literature reproduction gates fail.** `docs/SELECTION_REPRODUCTION_STATUS.md` §3 is red in
   seven places. Until a sample reproduces its published N exactly, it must not be enabled in
   `forward_model` mode — so the multi-sample likelihood has nothing valid to integrate over.
2. **The spuriousness model has not demonstrated three-rate reproduction** from a single parameter
   set, which is its entire acceptance test; the censoring treatment and covariate selection are
   unproven.
3. **Companion-nature evidence is still synthetic.** `companion_nature_likelihood` computes ΔBIC from
   analytic magnitude–mass relations, not real SED model comparison. The real evidence exists
   upstream in `dark-hunter_sed` and pop has no adapter for it (§3).
4. **The selection functions have never been validated against real data at production settings.**
   The El-Badry 2024 six-panel gate and the solution-type-fraction diagnostic are implemented but
   never demonstrated green on a real mock population.
5. **No production inference run exists.** Every `inference` artifact is a CI-scale dynesty smoke.

## 2. Scope boundary and compute

### 2.1 The current boundary

**The laptop is the only host in scope.** The objective of the authorized work is to get the whole
workflow running *smoothly* on `/Users/rfoley/darkhunter/pop/dark-hunter_pop/` — correct, reproducible,
and boring to operate. ziggy and lux come after that, not alongside it.

| Host | Status | Role when it opens |
|---|---|---|
| **Laptop** | **In scope** | Everything in Waves 0–D. Partial data set, which is sufficient for all of it (§2.2). |
| **ziggy** (`ziggy.ucolick.org`) | **Deferred — out of scope** | Holds the bulk of the data; SED production at `/data2/darkhunter/dark-hunter_sed/` with `activate_ziggy.sh` and existing cron. Future home for the SED campaign and full-catalog stages. |
| **lux** (supercomputer) | **Deferred — out of scope** | Production `inference` and the mock-injection volume for `sensitivity_analysis`. |

Recorded so it is not lost when the boundary moves: host differences must be **config, never code**
(`paths.data_root`, `paths.artifact_root`, `mass_derivation.sed_summary_root`, `dr3.rv_summary_root`
are already config keys), so opening ziggy is a fragment, not a fork. Roster #29 builds that profile
machinery now, on the laptop, precisely so the later move is one flag.

No agent reaches ziggy or lux. No ticket in the authorized scope depends on either.

### 2.2 Why laptop-only is enough for Waves 0–D

This is worth stating because it is not obvious, and a later wave that assumes otherwise will stall.

- **Wave A (reproduction) is not data-volume-limited.** It runs against snapshot caches already on
  disk: `data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c` and its `+enrich` and
  `+enrich+mc10000` derivatives. The Andrews parent (134,598) already reproduces. These are
  binding-and-physics bugs, not scale problems.
- **Wave B (spuriousness) fits 293 labeled rows** staged in `config/selections/external/`. Trivially
  laptop-scale.
- **Wave C, as scoped here, is plumbing**: the `phot_sed` adapter, the coverage diagnostic, and the
  Gap-2 covariance decision. The SED *campaign* that fills the coverage is ziggy work and is deferred
  with it.
- **Wave D (forward-model activation)** needs mock volume large enough for the six-panel validation.
  This is the one authorized wave where laptop scale may bind; #41's first task is to measure what
  mock size the gate actually needs and report it. If the laptop cannot reach it, that is a finding
  that helps size the ziggy/lux ask — not a reason to weaken the gate.

Laptop-scale results are **plumbing and method validation, never final science numbers.** Reproduction
of a published N is an exception: it is exact and host-independent, which is precisely why it is the
first gate.

### 2.3 The December consequence

`ORCHESTRATION_PLAN.md` §5 built its phasing around supercomputer access ending in December. Deferring
lux work compresses that window. This is a deliberate trade — a fast campaign on untrustworthy inputs
produces an untrustworthy posterior — but it is a real cost and is tracked as risk **C6** (§10), not
quietly absorbed. Revisit at the Wave D stop: at that point the reproduction gates are green, the
selection functions are validated, and the size of the compute ask is known, which is the earliest
moment the ziggy/lux conversation can be had on facts.

## 3. Cross-repo interfaces

Read from `UCSC-Transients/dark-hunter_sed` and `UCSC-Transients/dark-hunter_rv` on 2026-09-11.

### 3.1 `dark-hunter_rv` → pop (live and healthy)

`output/Gaia_DR3_<id>_summary.json` (schema v1) is the pop-facing contract, written by
`io_utils.write_star_summary` alongside the legacy `*_summary.txt`. Keys pop uses: `gaia_metadata`,
`nss_solution_type`, `nss_orbital`, `thiele_innes`, `pipeline_epochs`, `external_rvs`, `joker_fit`
(inclination, ω, `variants`) and `joker_fit_path` → `rv_fit_reports/<stem>_joker_fit.json`. Consumed
by `rv_adapter` into `CandidateRecord.rv_summary` and by `rv_consistency`. Upstream doc:
`dark-hunter_rv/docs/RV_SUMMARY_JSON.md`, which names pop issue #31 as its parent and requires a docs
PR here (against `FOUNDATION_INTERFACE_FREEZE.md`) before any breaking field rename.
`darkhunter_rv.rv_summary_json` and `summary_paths` are on that repo's `main`, so
`docs/ESCALATE_RV_SUMMARY_JSON.md` is **obsolete** and should be retired.

Note the coupling: `dark-hunter_sed`'s `push_m1` writes fitted M1 **back into the RV summary JSON**.
That file is the shared per-star record across all three repos, not an RV-only artifact.

### 3.2 `dark-hunter_sed` → pop (two open gaps)

| Artifact | Pop consumer |
|---|---|
| `output/sed_summaries/Gaia_DR3_<id>_sed_summary.json` — medians, credible intervals, `m1_msun` | `mass_derivation._load_sed_summary_json` → `parameterset_from_sed_summary` |
| `output/samples/Gaia_DR3_<id>_ums.fits` / `_utp.fits` — full posterior chains | **nothing** |
| `output/phot_sed/Gaia_DR3_<id>_<model>_summary.json` — dynesty **BIC** + **lnZ** + max-L params for `--model 1star \| 2star \| wd` | **nothing** — snapshot target is `data/phot_sed/` |

**Model → hypothesis mapping (confirmed).** `1star` = **dark** companion (no luminous secondary);
`wd` = **WD** (luminous normal star + white dwarf); `2star` = **other**, and `2star` is specifically a
**coeval** binary. Pop's `other` class is formally broader — a luminous non-degenerate secondary of
any age — so a non-coeval luminous secondary has no dedicated evidence model. Record that as a
documented limitation of the channel rather than a mapping error.

**Known caveat — bad photometry points.** `dark-hunter_sed` has outlier/error-floor controls
(`--phot-err-floor`, `--gaia-phot-err-floor`, `--phot-outlier-sigma`) but bad photometric data points
are **not fully resolved upstream**. Since BIC depends on both chi2 and n_data, a contaminated
photometry set moves ΔBIC in a way pop's fixed `companion_nature.delta_bic_threshold: 10.0` was not
tuned for. The adapter must record the n_data and the cleaning settings each summary was produced
under, so the sensitivity of the weights to photometry cleaning is measurable rather than invisible.

**Gap 1 — the ΔBIC channel is not wired.** `dark-hunter_sed` already implements the model comparison
`companion_nature_likelihood` was specified to consume: `phot_sed_cli --model` supports `1star`,
`2star` (coeval binary) and `wd` (`wd_model.run_wd_plus_star_fit` on `bergeron_wd.BergeronGrid`),
each reporting BIC and lnZ via MISTy → PHOENIX × F99 (R_V = 3.1) → synphot. Pop has the hooks —
`phot_chi2_dark_key`, `phot_chi2_wd_key`, `phot_chi2_other_key`, `phot_n_data_key`, and four XP
equivalents — but nothing fills them, so `companion_nature` falls back to its analytic
`*_mg_zero_point` / `*_mg_mass_slope` relations. **The capability is not missing; the adapter is.**

**Gap 2 — `ParameterSet` is being fed marginals.** `sed_summary.json` carries medians and credible
intervals; the joint information is in the `_ums.fits` chains, so `parameterset_from_sed_summary`
builds an M1 `ParameterSet` whose covariance is not the fitted covariance. Per
`dark-hunter-pop-workflow` §3 that is the wrong shape. Either the SED summary gains a covariance
block upstream (preferred, small PR) or pop reads the chains. **Never substitute a diagonal.**

**Throughput** is why WD contamination is blocked: uberMS SVI (UMS + UTP) plus a separate dynesty run
per Path-2 model, three models per star, against 2 staged summaries today. That campaign is ziggy
work (§2.1) and is deferred; the adapter that will consume it is not.

## 4. Strategy

**Laptop-smooth before anything else.** The bar for the authorized scope is not "a result" but "the
workflow runs correctly and predictably end to end on one machine". Everything below serves that.

**Low-hanging fruit first, across areas.** Wave 0 deliberately cuts across the wave structure and
pulls every cheap, unblocking item to the front. A cheap item that unblocks a different area beats an
expensive item in the current one.

**Get a dry run through end-to-end early.** Prove the plumbing carries a `dN/dM` with all five classes
out the far end, substituting documented synthetic inputs where real ones are missing. A direction
check, not a result — §7 Wave 0 states exactly how it must be labeled.

**Reproduce before you forward-model.** No sample flips to `forward_model`, and none enters the
likelihood, until `sample_reproduction_report` matches its published N exactly. A cut chain that
recovers the wrong N is one we do not understand.

**Fix the physics, not the thresholds.** Frozen thresholds in `config/selections/*.yaml` stay frozen.
Any change requires a `schema_version` bump, a `provenance` note, and explicit human sign-off — and if
a threshold change is what it takes, that is a finding to write down, not a fix to apply quietly.

## 5. How work is dispatched

### 5.1 All coding happens in subagents

Every implementation ticket in §6 is dispatched to its own subagent session, in its own worktree, on
its own branch, with the model and effort hand-picked for that ticket. The **wave runner** (§5.8)
dispatches them and does not edit code itself; the plan of record (this file and `CLAUDE.md`) is
maintained separately and never dispatches.

### 5.2 Standard session preamble

Paste this at the start of every subagent session, with `<slug>`, `<issue>`, and `Task:` filled in.
Paths are verified against this machine.

**Standing note (added Wave −1, from #49's audit of PR #171):** a subagent's fresh
`git worktree add` checkout has no `data/` (gitignored) and usually no gaiamock overlay installed.
Required-gate wall-clock and pass/skip counts measured there are **not representative** — several
data-dependent tests silently skip or run near-instantly without `data/`, understating both the
gate's real cost and, in one case (PR #171), the size of a fix's actual improvement. Any claim
about gate timing, or about which tests ran vs. skipped, must be measured on a checkout with the
real `data/` tree and the overlay present (the primary checkout, or a worktree with them copied in)
before it's trusted — #49 re-measures this routinely; a coding subagent should flag its own numbers
as worktree-only rather than imply they're representative.

```
1. Read and follow:
   * /Users/rfoley/.agents/skills/strict-workflow/SKILL.md
   * /Users/rfoley/.agents/skills/caveman/SKILL.md
   * /Users/rfoley/.agents/skills/regression-hunter/SKILL.md
   * /Users/rfoley/darkhunter/pop/dark-hunter_pop/.cursor/skills/dark-hunter-pop-workflow/SKILL.md
     (project skill; repo-local, NOT in ~/.agents — read it by path, do not expect autoload)
   * /Users/rfoley/.cursor/skills-cursor/create-rule/SKILL.md  (only if adding rules)

2. Read README.md. Check CLAUDE.md. If it is a dummy file or incomplete, fill it with the
   normal information plus anything in README.md, docs/, or this project's conversations.
   Read docs/EXECUTION_PLAN.md §7 for the wave you are in — its Entry state, Do not assume,
   and Exit artifacts. docs/ARCHITECTURE.md is authoritative for anything it covers.

3. Use ONLY:
   * .venv/bin/python3
   * .venv/bin/pytest
   * /opt/local/bin/gh
   Work on this laptop only. ziggy and lux are out of scope.
   Exception: if YOUR issue explicitly authorizes a scratch environment (the Q7 nsstools measurement
   does), create it outside the repo, record exact versions, and never add its packages to
   pyproject.toml or install them into .venv — every other ticket shares that venv.

4. Your issue is #<issue>, already filed by the orchestrator. Read it first.
   You MAY edit it — sharpen the acceptance checklist, record findings, note what you ruled out.
   Anything you discover that is OUTSIDE this issue's scope becomes a NEW issue you file with
   /opt/local/bin/gh, in UCSC-Transients/dark-hunter_rv style (## Summary, ## Context,
   ## Acceptance checklist, ## Do not), linked to #<issue>. Paste every issue URL.
   NOTHING is silently fixed in a PR. An untracked change is a defect, not a favor.

5. Create an isolated worktree + branch from latest main:
   git fetch origin && git worktree add ../dark-hunter_pop-worktrees/<slug> -b feat/<slug> origin/main
   Work ONLY in that worktree. Do NOT delete branches or worktrees — #49 owns teardown.

6. Create consistent [AI Checkpoint] microcommits when code changes. These are PUSHED and land on
   main inside the PR's merge commit — write each one so its message and diff still make sense to
   someone bisecting six months from now.

7. Before every final commit before a PR: git fetch origin && git merge origin/main (or rebase);
   resolve so a merge to main would be conflict-free. Do not commit if conflicts remain.

8. Run .venv/bin/pytest and fix failures before commit/PR. Required gate:
   -m "unit or physics or api". gaiamock / slow / network are optional.

9. Open a verbose PR with /opt/local/bin/gh that Closes #<issue>, includes a Test plan, and
   notes pytest status. Then arm auto-merge as a MERGE COMMIT (not squash) so the checkpoint
   history survives:
       /opt/local/bin/gh pr merge --auto --merge
   Do not force-push main. No --no-verify.

10. Execute ONLY this issue's scope. No unrelated refactors. Everything else is a new issue (step 4).

11. Strict typing, vectorize numerics, document params/limits per function;
    log new deps in pyproject.toml in the same PR.

12. Zero hardcoded physics constants, thresholds, or paths outside constants.py and config.
    Never edit a frozen config/selections/*.yaml threshold — escalate instead.

13. If you rename a parameter, config key, function signature, or schema field, say so explicitly
    in the PR body under "Interface changes". #49 checks for silent drift; help it.

14. When you finish, report back to the orchestrator: issue(s), PR, merge status, interface changes,
    anything you filed, and anything you could not do. The orchestrator pings the verification agent
    (#49) from that report.

15. User domain decisions are absolute.

Model: <slug from §5.5>
Effort: <Deep | Standard | Light — see CONTINUATION_PLAN.md §2.2>

Task: <one paragraph, imperative>
```

**Harness.** Subagents run as **Claude Code sessions**, which read `~/.agents/skills/` natively and
pick up `CLAUDE.md` automatically — so steps 1 and 2 are cheap. The project skill is the exception:
it lives in the repo, not in `~/.agents/skills/`, so the preamble reads it by absolute path rather
than relying on autoload. If it starts getting skipped, copy it into `~/.agents/skills/` and update
this section.

Two deviations from earlier drafts, both deliberate: worktrees go under
`../dark-hunter_pop-worktrees/<slug>`, matching the ones already on disk rather than introducing a
second naming scheme; and `dark-hunter-pop-workflow` is read from the **repo**, since it is
project-specific and does not live in `~/.agents/skills/`.

### 5.3 Issues are the ledger

- **The orchestrator owns the issues**, one per ticket, in `dark-hunter_rv` style, filed before
  dispatch. The subagent receives its number and does not open a duplicate.
- **The wave runner files them.** Issue bodies are written into the plan ahead of time; the wave
  runner files them as its first act (§5.8), because only it has `gh`. Drafts live under `data/issues/<wave>/` with a
  `file_issues.sh` that creates the children and then the umbrella linking them; the directory is
  gitignored and disposable. A reviewed draft is filed **as written** — if it is wrong, file it
  and comment, never silently rewrite it. The same applies to anything else the orchestrator
  cannot reach from the cloud: it becomes a Claude Code ticket rather than something the
  orchestrator improvises around.
- **Subagents may edit their issue** — sharpen acceptance criteria, record what was measured, note
  dead ends. An issue at the end of a ticket should read like a lab notebook, not like the brief.
- **Subagents file new issues** for anything they find outside scope, linked to their own.
- **One issue is one thing, but one PR may close several.** Fine-grained issues so partial work closes
  independently (`ORCHESTRATION_PLAN.md` §6.7); a PR that naturally covers several related ones lists
  every `Closes #N` on its own line. Wave 0's hygiene issues are the obvious case.
- **Nothing is silently fixed in a PR.** A PR that changes something no issue covers is bounced by
  #49, regardless of whether the change was an improvement. The point is that the project's state is
  legible from GitHub alone, and a fix nobody filed is a fix nobody can find later.

### 5.4 Merge, and the verification agent (#49)

**PRs may auto-merge on green CI — in `dark-hunter_pop` only.** Auto-merge is enabled on this repo
and **not** on `dark-hunter_rv` or `dark-hunter_sed`; PRs against those are merged by hand, by Ryan.
Any ticket with an upstream component (§3.2 Gap 2 is the one the authorized scope expects) therefore
has a hand-merge step that does not resolve on the agent's schedule — plan the pop-side work so it
does not block on it.

Auto-merge is only safe because a standing agent owns correctness *after* the merge.

**#49 — Verification / integrity.** Continuous, like Review/Integration (#15). Top tier, Deep effort:
it is the only role whose job is catching cross-module inconsistency, and being wrong here is silent.

**How it is triggered.** #49 is not a daemon. Each subagent **reports back to the orchestrator when
it finishes**, and the orchestrator then pings #49 to verify that merge — **one ping per merge**,
never batched, because a verdict covering three merges cannot say which one broke something. #49 also runs at every wave
boundary, where it takes the optional markers as well. Because it is heavy (§5.6), the orchestrator
holds the ping until memory allows rather than stacking it on top of a saturated laptop.

It does:

- Re-run the suite on `main` for each merge it is pinged about; optional markers at wave boundaries.
- Check the merged code **actually runs**, not merely that tests pass: `run_pipeline.py --dry-run`
  plans cleanly, and a small real stage execution completes.
- Confirm the merge **landed what the PR claimed** — every commit, every file. PR #134 landed one of
  four commits and nobody noticed for a week (risk C0); that check is now part of the job.
- Hunt **silent conflicts** between independently merged PRs — the failure mode auto-merge invites:
  divergent parameter or keyword names for the same quantity, config keys added under two spellings,
  schema/`ParameterSet` field drift, stage `inputs_from` or `dependency_modules` gone stale, artifact
  fingerprint keys no longer covering what affects a result, docs describing something the code
  stopped doing, two PRs that each pass alone and disagree together.
- **Own branch and worktree teardown.** It deletes a branch and removes its worktree once it considers
  the work finished. Coding agents never delete either — this is why the preamble has no teardown
  step, and why ~25 stale worktrees exist today.

It does **not**:

- Write code, with one exception: **it may write tests**, including a regression test that pins the
  inconsistency it found.
- Fix what it finds. It **escalates to the orchestrator**, which assigns the fix to an existing agent
  or files a new issue and dispatches a new one.

**It may revert.** When a merge leaves `main` broken, #49 reverts that merge first and escalates
second. A revert is not a fix — it restores a known-good state and bounds how long `main` stays
broken, which is the price of auto-merge. The revert is recorded on the originating issue, which
reopens. #49 never lands a forward fix; a verifier that writes fixes stops being an independent
check.

**The merge gate stays `unit or physics or api`.** It is kept under twenty minutes on purpose so
merges do not queue; the optional `gaiamock` / `slow` / `network` markers run in #49's post-merge
sweep and at every wave boundary. The gate is deliberately not the safety net — #49 is.

Division with #15: Review/Integration still merges `config/fragments/` into `config/config.yaml` at
checkpoints and owns pre-merge config coherence. #49 owns everything after the merge button.

### 5.5 Model and effort matched to the task

`caveman` is active in **every** session — it is the token-reduction skill, and its exemptions
(diagnostic reports, validation summaries, run-plan output, plot captions, and all convergence and
gate failures) are exactly the outputs that must stay legible. Cheap sessions benefit most, because a
Light-tier model wastes the largest fraction of its budget on prose.

| Tier | Slug | Use for | Typical effort |
|---|---|---|---|
| Top | `claude-opus-5-thinking-medium` | Statistical formulation, selection-function correctness, likelihood changes, cross-module integrity — anywhere being wrong is silent | Deep |
| Top (alt) | `gpt-5.6-sol-medium` | Second opinion on a Top-tier statistical result | Deep |
| Mid | `claude-4-sonnet` | Well-specified implementation against a frozen interface; adapters; ETL; diagnostics | Standard |
| Light | `claude-haiku` | Transcription, doc sync, fixture generation, hygiene, mechanical sweeps with exact acceptance tests | Light |

Effort is a contract about how much the subagent may **decide** and how much verification it must
produce before a PR (`CONTINUATION_PLAN.md` §2.2). A **Light** session that discovers it needs a
structural decision must stop and escalate, not decide. Lower tier *and* effort together — a Haiku
session running a Deep contract is the worst of both.

Concretely: #28 and the doc-sync half of #36 are Light; #29, #30, #31, #35, #40 are Mid/Standard;
#32, #33, #34, #37, #38, #41, #42, #43 and #49 are Top/Deep.

### 5.6 Concurrency

Run as many concurrent sessions as makes sense; the binding constraint is **laptop memory**, not a
fixed session count.

Classify every dispatched session before starting it:

| Class | What it does | Cost |
|---|---|---|
| **Heavy** | Runs `pytest`, executes stages, loads HDF5 snapshots, runs MC draws or dynesty smokes | RAM-bound; several GB each, and the `+enrich+mc10000` cache is the usual spike |
| **Light** | Docs, config fragments, issue triage, small adapters with fixture-only tests | Negligible |

**Budget: 40 GB.** The laptop has 64 GB and must stay usable for other work, so all agent activity
together stays under 40 GB, and lower is better.

What the heavy sessions are actually carrying, measured on disk:

| Artifact | Size |
|---|---|
| `…+enrich+mc10000/selection_parent_rows.h5` (Wave A's working cache) | 734 MB |
| `…+enrich/selection_parent_rows.h5` | 691 MB |
| uncut parent `selection_parent_rows.h5` | 681 MB |
| largest `data_acquisition` artifact | 931 MB |

**Measured, not estimated.** Peak RSS for every representative heavy workload, taken on `main`
@ `c905575` in the primary checkout (real `data/` tree, gaiamock overlay installed, nothing else
running), via `/usr/bin/time -l` — whose `maximum resident set size` is reported in **bytes** on
macOS 14 — cross-checked in-process with `resource.getrusage(RUSAGE_SELF).ru_maxrss`:

| Workload | Wall clock | **Peak RSS** |
|---|---|---|
| `pytest -m "unit or physics or api"` (the required gate) | 34m23s | **4.56 GiB** |
| `pytest -m slow` | 6m30s | **3.26 GiB** |
| `pytest -m gaiamock` | 4m48s | **1.91 GiB** |
| `pytest -m network` | 9m11s | **0.20 GiB** |
| `+enrich+mc10000` cache read + `andrews2022` evaluate | 23 s | **3.01 GiB** |
| — of which the cache read alone (443,211 rows) | 17 s | 2.77 GiB |
| Full `SELECTION_REPRODUCTION_STATUS.md` §3 sweep, incl. the El-Badry 2026 lazy `σ_M̃2` MC | 22m52s | **6.82 GiB** |
| `data_acquisition` artifact (931 MB) read + `mass_derivation_bulk` over 7,816 records | 65 s | **4.61 GiB** |

Zero swaps in every run. The old ~6 GB guess was close, but for the wrong reason: a typical heavy
session peaks near **4.6 GiB**, and only the El-Badry 2026 Monte Carlo reaches **6.8 GiB**. That MC is
the real ceiling, and it is a Wave A workload — so Wave A is the wave that runs closest to the budget.

**Cap: four heavy sessions.** Four times the measured worst case is 27.3 GiB, inside the 40 GB budget
with ~12 GB of headroom — enough for a handful of light sessions and for the operator's own work. Five
is arithmetically possible (34.1 GiB) and is exactly the kind of no-headroom arithmetic this section
says not to trust: it leaves nothing for a workload nobody has measured yet, and the failure mode is
swapping, which does not degrade gracefully. Raise to five only after a wave has actually run four
without pressure. Remember that **#49 is itself heavy** and consumes one of the four whenever it is
verifying a merge.

These numbers replace the previous estimate; **#28's measurement is discharged** by issue #144
(Wave −1), which took them. Re-measure if `_read_selection_parent_cache` changes how rows are
materialized, if the MC draw count moves off 10,000, or if the parent cache grows.

Rule: **cap the heavy sessions, let the light ones run free** (light sessions are ~1 GB; a handful is
free). Never let it swap — swapping does not degrade a run gracefully, it turns a twenty-minute suite
into a lost afternoon. Wave 0 is mostly light and can run wide; Waves A and D are mostly heavy and
should run narrow. **#49 is itself a heavy session** — it runs the suite — so it counts against the
budget, and the orchestrator does not ping it while heavy coding sessions are saturating memory. The
orchestrator states the class when it dispatches.

One caveat the measurements make concrete: a heavy session's cost is **wall clock**, not only memory.
The required gate alone is 34 minutes (#177), so four concurrent heavy sessions each running the gate
before opening a PR is over two CPU-hours of the laptop. Memory sets the cap; gate duration is what
makes a wide heavy wave feel slow.

### 5.6b Merge mode and commit history

PRs land as **merge commits, keeping every `[AI Checkpoint]`**. The reason is regression-hunting: when
something breaks weeks later, the checkpoint granularity is what makes the introducing change findable.
This supersedes `ORCHESTRATION_PLAN.md` §6's "local use only, not pushed".

Two consequences to work with rather than around:

- **Bisect with `--first-parent`.** Plain `git bisect` will land on mid-PR states that were never
  meant to pass. `git bisect start --first-parent` walks merge commits only, giving per-PR
  granularity; drop the flag to descend into a suspect PR's checkpoints once it is identified.
  That is the two-level search the merge-commit history exists to support.
- **Checkpoint messages are public now.** They are no longer scratch. Each should name what changed
  and why in one line — `[AI Checkpoint] bind logg_apsis as None for giant cut` beats
  `[AI Checkpoint] wip`.

### 5.7 Compaction

Compact a subagent conversation when it crosses into a long repetitive loop — a fit re-run with
tweaked settings, a suite driven to green, a wide mechanical sweep — and always at a natural boundary:
after the issue is read, after the first passing pytest, and before the pre-PR merge. Carry forward
the ticket scope, the acceptance checklist, decisions made, and the current diff; drop the tool-output
history. **Do not compact across a decision the session has not yet written down** in the issue, a
checkpoint commit, or a doc PR — that is how a rationale is lost and then re-litigated.

### 5.8 Who orchestrates, and carrying state across sessions

Orchestration splits across two places, because no single session can both decide and execute.

**The orchestrator — one long-lived Claude Code session on the laptop — executes.** It is the only
thing with `gh`, the repo `.venv`, and the ability to spawn subagents, so it owns the whole run: it
files each wave's issues, dispatches one subagent per ticket with the §5.2 preamble, collects their
completion reports, **pings the verification agent (#49) on every merge**, assembles the gate
artifacts, and stops for approval at each wave boundary. It persists across waves; it never writes
production code. Its prompt is `docs/ORCHESTRATOR_PROMPT.md` §A, and the verification prompt it pastes
on every ping is §B of the same file.

It keeps a **ledger** in each wave's umbrella issue — ticket, issue, model/effort, class, PR, merge
SHA, #49 verdict — one row per ticket. A row is not done until its verdict column is filled, and
nothing reaches a gate unverified. The ledger is also how the orchestrator holds the concurrency line:
it tracks the heavy-session count against the 40 GB budget and holds a dispatch rather than exceed it.
When memory is saturated, a verification ping is **queued and recorded**, never skipped.

**The plan of record lives on disk.** `EXECUTION_PLAN.md` (this file) and `CLAUDE.md` are the
specification; a Cowork or chat session may maintain them, review what landed, and write handoff
notes, but it cannot reach `gh` or run the repo `.venv` and therefore never dispatches. Anything it
cannot reach becomes a ticket rather than something it improvises around.

Because every session ends, nothing load-bearing may live only in a conversation. State carries in
four durable places, and any successor rebuilds context from them alone:

- **This document** — plan, roster, current wave (update the status line at each checkpoint).
- **`CLAUDE.md`** — the standing brief: architecture, conventions, status, open questions.
- **GitHub issues and PRs** — one issue per ticket; decisions and escalations recorded there.
- **`runs/*.yaml` and the diagnostic artifacts** — what was executed and what it produced.

A decision that exists only in a chat transcript will be re-litigated. Write it to one of the four
before the session ends.

### 5.9 Stop at every wave gate — and the handoff note

**Each wave ends in a hard stop for operator review.** The next wave's tickets are not dispatched
until Ryan has examined the gate artifacts and approved. At each stop the orchestrator presents the
artifacts themselves (not a description of them), what changed on `main`, anything that failed or was
escalated, and any open question the wave surfaced. Approval is explicit, never inferred from silence.

This is why the gates are defined as **artifacts**: a stop where the operator approves a claim rather
than looks at a diagnostic is not a review.

**The handoff note.** Waves must not reach each other by paraphrase. At every stop the orchestrator
writes a note into the wave's umbrella issue and links it from this document's status line, in exactly
this shape:

```
## Wave <X> handoff

### What is now true
<facts, each with the artifact or command that demonstrates it>

### What changed on main
<PRs merged, interface changes, config keys added or renamed>

### Numbers measured
<every measured value, with the cache/run it came from — never a rounded restatement>

### What did NOT get done
<tickets deferred, gates missed, and why>

### Do not assume
<beliefs a later wave might wrongly inherit — see each wave's own list in §7>

### Open questions raised
<new escalations, with the issue number>
```

A later wave reads the note **and verifies its Entry state independently** (§7). The note says what
was believed; the Entry-state checks say what is true. Where they disagree, the checks win and the
disagreement is escalated.

### 5.10 Workflow interview before Wave 0

Before the first ticket is dispatched, the orchestrator interviews the operator about workflow issues
— tooling friction, prior sessions that went wrong and why, host and credential access, laptop
resource limits for §5.6, anything about the protocol above that does not match how work actually
happens. Answers are recorded **in this section**, not in a chat log.

## 6. Roster continuation

Numbering continues from `CONTINUATION_PLAN.md` §3 (#17–#27). Tier slugs and effort contracts are
defined in §5.5 and `CONTINUATION_PLAN.md` §2.2.

| # | Subagent | Wave | Depends on | Tier | Effort |
|---|---|---|---|---|---|
| 50 | **Bootstrap: file the issue sets** — file Wave −1 and Wave 0 from the drafted bodies; the orchestrator has no `gh` | −1 | — | Light | Light |
| 51 | **Branch / WIP inventory** — survey every unmerged branch, classify its contents, check for other dropped-commit merges. Read-only | −1 | #50 | Mid | Standard |
| 52 | **Reconcile `fix/selection-reproduction-binding`** — land the wanted commits (incl. `attach_mc_to_selection_cache.py`) without reverting the Wave-2 RV/SED work | −1 | #51 | Top | Deep |
| 53 | **Land or drop the remaining WIP** per the inventory, filing follow-ups for anything landed incomplete | −1 | #51, #52 | Top | Deep |
| 54 | **Tree and environment cleanup** — untracked run manifests, orphaned bytecode, stray patch, submodule pointer; venv, `gh` auth, gaiamock overlay, auto-merge setting | −1 | #53 | Mid | Standard |
| 55 | **Baseline** — full suite incl. optional markers, peak-RSS measurement, reproduction numbers re-measured on reconciled `main` | −1 | #54 | Mid | Standard |
| 28 | **Wave 0 docs/config quick wins** — retire `ESCALATE_RV_SUMMARY_JSON.md`; merge the orphaned `spectroscopic_mass_function` fragment into `config.yaml` | 0 | Wave −1 | Light | Light |
| 29 | **Host profiles** — `laptop` / `ziggy` / `lux` config fragments for `paths.*` and the summary roots, with the profile recorded in the run file. Only `laptop` is exercised now; the others are written and left unused | 0 | — | Mid | Standard |
| 30 | **`phot_sed` adapter** — read `Gaia_DR3_<id>_{1star,2star,wd}_summary.json` into the `phot_chi2_*` / `phot_n_data` extras so `companion_nature` uses real BIC/lnZ where available and the analytic fallback only where it is not, reporting the split | 0 | — | Mid | Standard |
| 31 | **Cheap measurements** — Q9 (`G<15` on reproduced Andrews → expect 16); the Q7 `nsstools` vs `thiele_innes_to_campbell` delta on `ã0`/`σ_ã0`; the #132 K1 bind; the single mis-bucketed Simon exclusion | 0 | — | Mid | Standard |
| 32 | **Dry-run harness** — one labeled end-to-end run to `dN/dM` with all five classes plotted, synthetic stand-ins enumerated in the run file | 0 | #28–#31 | Top | Deep |
| 33 | **Extinction / `a0` / AMRF chain audit** — fix `mg_0` / `a0` using #31's measured delta; re-measure `primary_ns_bh` and `sub_chandrasekhar` | A | #31 | Top | Deep |
| 34 | **Andrews MC attrition reconciliation** — 352 after `P(M2>1.4)≥0.95` against the paper's 106 (packing, floors, prefilters) | A | #33 | Top | Deep |
| 35 | **El-Badry 2026 spectroscopic-branch binding** — `mass_route` / MS / `m2_min` chain from 123 to 151 | A | #33 | Mid | Standard |
| 36 | **Reproduction closeout + status sync** | A | #33–#35 | Mid | Standard |
| 37 | **Spuriousness censoring + covariate selection** — joint Heckman-style fit, sensitivity-module covariate admission, `F2 × G` interaction | B | #36 | Top | Deep |
| 38 | **Spuriousness acceptance** — three-rate reproduction from one parameter set; Q15 denominator | B | #37 | Top | Deep |
| 39 | **SED campaign** — prioritized `phot_sed` throughput, uberMS coverage | C | #30 | — | — | **DEFERRED (ziggy)** |
| 40 | **Follow-up selection-function calibration** — Drive adoption-date mining, weekly sheet snapshots, survey SFs, N_obs / time-span histogram matching | C | #29 | Top | Deep |
| 41 | **Astrometric selection-function validation gate** — El-Badry 2024 six-panel + solution-type fractions; first task is measuring the mock size the gate needs | D | #33 | Top | Deep |
| 42 | **Forward-model activation** — flip reproduced samples, produce `sample_selection_function`, `mode_divergence`, `sample_overlap_matrix` | D | #36, #38, #41 | Top | Deep |
| 43 | **Q1 formulation verification** on the real three-way overlap | D | #42 | Top | Deep |
| 44–48 | **Cluster recipe, production campaign, validation, figures, paper** | E–H | — | — | — | **NOT AUTHORIZED (§2)** |
| 49 | **Verification / integrity** — post-merge correctness, silent-conflict hunting, may revert a bad merge, branch and worktree teardown; tests only, never forward fixes, always escalates (§5.4) | all | — | Top | Deep |
| 15 | **Review/Integration** — config fragments into `config.yaml` at checkpoints; pre-merge config coherence | all | — | Top | — |

`accel_jerk` (#22) stays **blocked and disabled**; not on the v1 critical path.

## 7. Waves and acceptance gates

Each wave below is written to be read **cold**, by a session that has not seen the earlier ones. The
`Entry state` items are facts to *verify*, not to trust; `Do not assume` lists the beliefs most likely
to arrive distorted. A gate is passed by a **diagnostic artifact**, not an assertion in a PR body.

---

### Wave −1 — clean the local environment and repo; make `main` the truth — **COMPLETE**

**Roster:** #50–#55, **all complete** (umbrella #145). **Precondition:** the §5.10 workflow interview.
**Class:** #52/#53 heavy, the rest light.

Nothing later in this plan is safe until `main` says what it is documented as saying. As of
`main` @ `c905575` it does.

**What the wave actually landed**

| Ticket | Issue | PR | Merge | Outcome |
|---|---|---|---|---|
| #50 bootstrap — file the issue sets | #139 | — | — | Done directly by the orchestrator |
| #51 branch / WIP inventory | #140 | — | — | Read-only survey; the orchestrator's first pass held — only three refs were ever ahead of `main` |
| #52 reconcile the reproduction branch | #141 | #150 | `03a452a` | The three missing commits are on `main`, with the Wave-2 RV/SED work intact |
| #53 land or drop the remaining WIP | #142 | — | — | Both remaining branches recorded as superseded; nothing to land |
| #54 tree / environment cleanup | #143 | #179 | `c905575` | Run manifests tracked, ad-hoc logs gitignored, environment verified |
| #55 full suite + RSS + re-measurement | #144 | this one | — | The baseline below |

Plus a chain of defects that #49's post-merge verification found and that were fixed rather than
carried into Wave A: #151/#155 (stage-registry fingerprint), #152/#156 (the σ landmine and the Janssens
cache), #157/#159 (cached artifacts never checked `source_hash`), #158/#166 (missing transitive deps),
#167/#170 and #172/#173 (`StageAction` falling through silently in twelve stage runners), #168/#171
(test-marker hygiene). Most of these were invisible correctness holes, not cosmetics — a stale cached
artifact being accepted without a `source_hash` check would have silently poisoned every later wave.

**The baseline this wave established** (all on `main` @ `c905575`, primary checkout, real `data/`,
gaiamock overlay installed — see §5.2's standing note on why the checkout matters):

- `pytest -m "unit or physics or api"` — **467 passed, 0 skipped, 10 deselected, 34m23s**, peak RSS
  4.56 GiB. Green, and still far over the `<< 20 min` target (#177).
- `pytest -m gaiamock` — 4 passed, 4m48s. `pytest -m slow` — 6 passed, 6m30s. Both green.
- `pytest -m network` — **3 failed, 1 passed**, all three failures `HTTPError: 500` from the Gaia TAP
  archive, which returns 500 for `SELECT TOP 1 source_id FROM gaiadr3.gaia_source` too. Archive-side,
  not ours; the live parent-count checks are therefore **not run**, not green (#184).
- Peak RSS for every heavy workload, and the concurrency cap set from it — §5.6.
- Every `SELECTION_REPRODUCTION_STATUS.md` §3 number re-measured on `main` and rewritten with its SHA.
  **No escalation signal tripped**: all three exact-match equalities hold (Andrews parent 134,598;
  El-Badry 2024 union 48; `elbadry2023_table_e1` 5), nothing moved by an order of magnitude, nothing
  collapsed to or appeared from zero. Every count is identical to the branch-measured value. The
  reconciliation lost nothing; the gates stay red for the reasons Wave A owns.

**Deliberately left for later, with issues:** #169 (no-`source_hash` policy — awaiting Ryan), #174,
#175, #176, #177, #178, #181, #182, #183 (#49's findings, deferred past this wave), #184 (the network
tests), #185 (four planning docs still only in the working tree).

**Entry state — verified at the start of the wave, kept for the record**

| Fact | How to check |
|---|---|
| PR #134 landed one of four commits | `git log --oneline main..fix/selection-reproduction-binding` lists `e90f91d`, `78120a7`, `e5a911e`, `2e6c53f` |
| The reconciliation is not a fast-forward | `git diff --stat main fix/selection-reproduction-binding` → 27 files, +778 / −1011, removing 277 lines of Wave-2 tests |
| Orphaned bytecode present | `ls src/darkhunter_pop/__pycache__/elbadry2026_m2_sigma*` with no matching `.py` |
| Untracked run manifests | `git status --short runs/` |
| ~25 prunable worktrees, ~40 branches — but only **three refs ahead of `main`** | `git worktree list`; `for b in $(git branch -a --format='%(refname:short)'); do echo $(git rev-list --count main..$b) $b; done` |
| `gh` is authenticated and auto-merge is on | `gh auth status`; `gh api repos/UCSC-Transients/dark-hunter_pop --jq '.allow_auto_merge'` |

**Tickets**

#50 files the issue sets (the orchestrator cannot reach `gh`). #51 surveys **every** unmerged branch
read-only and classifies what it holds, including an explicit check for other merges that dropped
commits the way #134 did.

A first pass has already been run by the orchestrator (recorded in #51's body, to be verified rather
than repeated): only three refs are ahead of `main` at all. `fix/selection-reproduction-binding` holds
the four real commits. `fix/mass-derivation-bulk-stream` is four Phase-8 *docs* commits already on
`main` under different SHAs — branch `cd690c3` is `5ca2954` on `main` — sitting 42 commits stale, so
its −16,777-line diff is age, not content. `origin/foundation/f0-gaiamock-vendor` is one Foundation-era
merge commit, likewise ancient. Both are superseded. **If that holds, #53 has nothing to do but record
the two dispositions, and #52 is the whole of the WIP work** — which makes Wave −1 considerably
smaller than its ticket count suggests. #52 reconciles the known branch. #53 dispositions everything else the
inventory flagged. #54 cleans the tree and verifies the local environment. #55 closes the wave with a
full-suite run, a peak-RSS measurement, and — critically — a **re-measurement of every
`SELECTION_REPRODUCTION_STATUS.md` §3 number on the reconciled `main`**.

**The WIP decision rule.** For every piece of unmerged work: **land it, or leave it out — never land it
half-finished and silent.** Land it when the remaining work is small and understood; if it is
incomplete but not breaking, land it *and file the follow-up issues in the same PR*, so the gap is
tracked rather than lurking. Leave it out when the remaining work is large, when it breaks something,
or when the shape of the remaining work is unclear — recording what it was, why, and which branch holds
it. **Never land anything that turns a green `main` red.** Judge by expected work, not by how nearly
done it looks.

**Reading the re-measured numbers.** Wave −1 records them; Wave A fixes them. The gates are red for
known reasons, so a number that is still wrong — or somewhat more wrong than the branch reported — is
recorded and left alone. Escalate on three signals only, each of which means the *reconciliation* lost
something rather than that the physics is unfinished: an **order-of-magnitude** change against the
branch-measured value; an **exact-match check that stops matching** (Andrews parent 134,598; El-Badry
2024's union of 48 and COC N of 21; `elbadry2023_table_e1` = 5 — these are equalities, so any change
counts); or a count **collapsing to zero or appearing from zero**, which is usually an unbound column.
Escalating means record it, tell the orchestrator, and stop — the orchestrator decides whether it
belongs to the reconciliation or to Wave A.

**Do not assume**

- That `SELECTION_REPRODUCTION_STATUS.md` §2 "What is already fixed" describes `main`. It describes the
  branch. Three of its six items are not in the code.
- That any "Last measured" number in that document was measured on `main`. None were.
- That the branch can be merged wholesale. It is *behind* `main` on the Wave-2 RV/SED work and would
  revert PRs #136–#138.
- That a merged PR landed what it claimed. #134 did not, and #51 checks whether others did the same.
- That a branch with no worktree is dead, or that a worktree marked prunable has nothing in it.

**Exit artifacts (Gate −1)** — status as of `main` @ `c905575`

| Artifact | Status |
|---|---|
| The branch inventory, as a table, with a disposition per branch | **Done** — #140 |
| A reconciled `main` with the reproduction-binding work and the Wave-2 work both intact | **Done** — #141 / PR #150 / `03a452a` |
| Every other WIP landed (with follow-up issues) or recorded as left out, with the reason | **Done** — #142; both remaining branches superseded |
| A clean `git status`; no untracked run manifests; no orphaned bytecode; submodule pointer resolved | **Mostly** — run manifests tracked and bytecode gone (PR #179), but four planning docs still live only in the working tree (#185) |
| A "known-good local state" note | **Done** — #143 |
| Full-suite results including the optional markers, every failure filed | **Done** — #144; the only failures are the three live-archive tests, filed as #184 |
| Peak-RSS measurements, and §5.6's concurrency cap set from them | **Done** — #144; cap set to four heavy sessions |
| `SELECTION_REPRODUCTION_STATUS.md` re-measured on `main`, with the commit each number came from, each checked against the three escalation signals | **Done** — #144; no signal tripped |
| Branches and worktrees this wave finished with, torn down by #49 | **Outstanding** — #49 owns it; no coding session may do it |

**Then stop for operator review (§5.9).** The handoff note goes in umbrella issue #145.

---

### Wave 0 — quick wins and the end-to-end dry run

**Roster:** #28–#32. **Precondition:** the §5.10 workflow interview. **Class:** mostly light (§5.6),
so this wave can run wide.

**Entry state — verify before starting**

| Fact | How to check |
|---|---|
| `main` is at or after `94dc390` | `git log --oneline -1` |
| The uncut snapshot and both caches exist | `ls data/dr3/gaia_snapshots/` → `20260826T234425Z_3d3f740b080c`, `+enrich`, `+enrich+mc10000` |
| 154 RV JSON summaries staged | `ls data/dr3/rv_summaries/*.json \| wc -l` |
| Only 2 SED summaries staged | `ls data/sed_summaries/` |
| Required gate passes | `.venv/bin/pytest -m "unit or physics or api"` |
| **The local `dark-hunter_sed` checkout is current enough for #30** | `ls .venv/bin \| grep darkhunter` — if `darkhunter-sed-phot` is missing, the editable install at `/Users/rfoley/darkhunter/seds/dark-hunter_sed/` predates `phot_sed_cli`; pull and re-run `pip install -e` before #30 assumes Path-2 outputs exist |

**Tickets**

*Quick wins (#28–#31).* Retire the obsolete RV escalation doc (the JSON writer is on
`dark-hunter_rv` `main`). Commit the untracked run files — a run file that exists only on a laptop is
not a manifest. Merge the orphaned `spectroscopic_mass_function` fragment into `config.yaml`. Land the
MC-cache script on `main`. Run the full suite including optional markers once and file what fails.
Build the host profiles (#29) so the later ziggy move is a flag, exercising only `laptop` now. Wire
the `phot_sed` adapter (#30) — pop gains real BIC/lnZ for whatever stars `dark-hunter_sed` has already
fitted and falls back to the analytic relations only where it must, with the funnel reporting the
split. Take the four cheap measurements (#31); each is a command and an answer, and Q7's answer is
what Wave A needs on day one.

*The dry run (#32).* Run all fourteen stages end to end and produce the product figure: total `dN/dM`
plus BH, NS, WD, other and outlier overplotted. Where a real input does not exist, substitute a
**documented** synthetic stand-in — a synthetic classification/companion-nature covariance, placeholder
per-sample selection weights, CI-scale dynesty settings. Every substitution is recorded in the run file
and printed in the run plan.

**Do not assume**

- That the dry run's `dN/dM` means anything about the real mass function. It does not. It tests that
  the plumbing carries a shape from one end to the other.
- That the `phot_sed` adapter gives real evidence for many stars. Two SED summaries exist; coverage is
  a Wave C/ziggy problem. The adapter's value now is that the *path* is proven and the split is
  measured.
- That `--force-rerun` amends a run. It always creates a new run file (`ARCHITECTURE.md` §5).

**Exit artifacts (Gate 0)**

- #28's full-suite failures fixed or filed as issues.
- One run reaching `diagnostics` **completed**, nothing left in `running`, every `skipped` carrying a
  stated `reason`.
- The `dN/dM`-by-class figure, to `docs/PLOTS.md` standards.
- A run file enumerating every synthetic stand-in used.
- The measured values from #31, each with the cache it came from.

> **Labeling requirement.** The dry-run figure and report must carry an explicit "synthetic inputs —
> plumbing check, not a science result" caption naming each stand-in, and the run must be tagged as
> such in `runs/`. The whole value is seeing whether the machinery's shape is right; the whole risk is
> a plausible-looking curve escaping into a talk. Label it at birth.

**Then stop for operator review (§5.9).** This gate in particular is the direction check the wave
exists for.

---

### Wave A — close the reproduction gates *(blocking)*

**Roster:** #33–#36. **Class:** heavy — run narrow.

**Entry state — verify before starting**

| Fact | How to check |
|---|---|
| Wave 0 approved, handoff note written | the umbrella issue |
| Q7 delta measured in #31 | the #31 issue — read the *number*, not a summary of it |
| Evaluation runs off `+enrich+mc10000` | `_read_selection_parent_cache(...+enrich+mc10000/selection_parent_rows.h5)` |
| Current red gates | `docs/SELECTION_REPRODUCTION_STATUS.md` §3, "Last measured" column — **re-measured on `main`** by the Wave 0 reconciliation ticket; pre-Wave-0 numbers came from a branch |
| No unmerged reproduction work remains | `git log --oneline main..fix/selection-reproduction-binding` is empty, or every remaining commit is classified obsolete |

**Tickets**

Audit the extinction chain end to end: `E(B−V)` source, `A_G` / `mg_0`, and how `a0` and the AMRF
propagate into `M̃1` / `M̃2`. `sub_chandrasekhar` at 861 against a target of 22 is an over-full mass
window — `m2_range` alone admits 1908 — so the **point estimate** of `M̃2`, not the σ cut, is where to
look. Reconcile the Andrews Monte Carlo (352 vs 106): covariance packing order, error floors, any
prefilter the paper applies before its MC. Fix the El-Badry 2026 spectroscopic binding (123 → 151)
downstream of the K1 bind, and the last Simon exclusion slot. Re-run every §3 check and update the
"Last measured" column in the same PR as the fix.

**Do not assume**

- That a failing number means a threshold is wrong. **Frozen thresholds stay frozen.** Every open
  failure has a suspected cause in binding or in the extinction/`a0`/AMRF chain.
- That Andrews' σ may be aliased into El-Badry's. Column ownership is strict: Andrews owns
  `p_m2_above` / `m2_msun` / `m2_msun_error` at fixed `M1 = 1.0`; El-Badry 2026 owns
  `m1_tilde_msun` / `m2_tilde_msun` / `sigma_m2_astrometric_msun` at fixed Janssens `M̃1`.
- That the literature parent is the quality-cut snapshot. It is the **uncut** one.
- That Andrews excluding Gaia BH1 was correct. It was not — Q14. `andrews2022.yaml` keeps the
  exclusion as published anyway; the correction lives only in `andrews2022_modified.yaml`.
- That the NSS enrichment job needs re-running. It is **COMPLETED**.
- That `σ_M̃2` propagates the Janssens fit uncertainty. It does not — Q12, `propagate_fit_uncertainty:
  false`, provenance `elbadry2026_m1_tilde_fixed`.

**Exit artifacts (Gate A)**

- `sample_reproduction_report` exact: Andrews N=24 (modified 25), El-Badry 2024 N=21, El-Badry 2026
  N=227 (76 astrometric + 151 SB1) with subsample and route breakdowns matching.
- `simon2026_exclusion_breakdown` = 5/2/1/1.
- Q9 confirmed (16).
- `m2_posterior_convergence`: MC noise subdominant, boundary count reported.
- `covariance_health`: zero silent drops.
- `SELECTION_REPRODUCTION_STATUS.md` updated, or retired.

No frozen threshold edited without a `schema_version` bump and a signed-off escalation note. **Stop.**

---

### Wave B — spuriousness model acceptance

**Roster:** #37–#38. **Class:** heavy.

**Entry state**

| Fact | How to check |
|---|---|
| Gate A passed | `sample_reproduction_report` artifact |
| Four label fixtures staged at `schema_version: 3` | `config/selections/external/elbadry{2023_table_e1,2024_table3,2026_table7,2026_table8}.yaml` |
| Totals | 293 rows: 91 good, 164 spurious, 38 `unknown_*` |

**Tickets**

Joint label/censoring fit over all 293 rows. Covariate admission driven by the sensitivity-analysis
module, not the candidate table, with `F2 × G` reported separately as the one literature-motivated
interaction. Stratify by branch and report by `reason` even though v1 predicts only the collapsed
state. Resolve the Q15 SB1 denominator.

**Do not assume**

- That the `unknown_*` rows may be dropped. They are **censored, not missing at random**, and the
  censoring depends on covariates the model uses (`G`, `F2`). Dropping them biases the model exactly
  where spuriousness is highest. If the joint model is unidentified, **escalate** — do not fall back.
- That a sample "has" a spurious rate. A rate is *produced* by pushing that sample's cuts through the
  shared propensity. Each paper's quoted rate is a `validation_targets:` output, never an input.
- That reproducing one rate is partial credit. It is the signature of a model that absorbed a
  per-sample normalization — the exact failure this design exists to prevent.
- That the SB1 ~50% is row-recoverable from Table 8. It is not (Q15); treat it as advisory.

**Exit artifacts (Gate B)**

`spuriousness_rate_reproduction` (both astrometric targets from one parameter set),
`spuriousness_covariate_sensitivity`, `spuriousness_censoring_report` (drop-the-rows bias quantified),
`spuriousness_labeled_set_performance` (all 293, Gaia BH1 called out). **Stop.**

---

### Wave C — companion-nature plumbing and follow-up calibration

**Roster:** #40 (+ #39 **deferred to ziggy**). **Class:** mixed. Runs parallel to A and B.

**Entry state**

| Fact | How to check |
|---|---|
| #30's `phot_sed` adapter merged | `git log`, and the coverage diagnostic exists |
| SED coverage is ~2 stars | `ls data/sed_summaries/` |
| Survey SFs staged | `config/target_lists/survey_sfs/{apogee,desi,lamost,rave}.yaml` |
| Adoption dates staged | `config/target_lists/derived/*.yaml` |

**Tickets**

Follow-up selection function (#40): Drive-API revision-history mining for target-list adoption dates
(with the documented incompleteness caveat spot-checked against the UI panel), the going-forward
weekly sheet snapshots, the staged survey selection functions, and calibration by matching
mock-vs-real `N_observations` and follow-up-time-span histograms. Resolve the Gap-2 covariance
question (§3.2) and, if the answer is an upstream `sed_summary.json` covariance block, file it against
`dark-hunter_sed`.

**Do not assume**

- That thin SED coverage blocks the wave. `companion_nature` treats nature as a **weight, not a
  filter**; thin coverage widens the posterior, it does not invalidate it. Report the real-evidence vs
  analytic-fallback split as a systematic.
- That the SED campaign can be run here. It is ziggy work and out of scope (§2.1).
- That a diagonal covariance is an acceptable Gap-2 stopgap. It is not.

**Exit artifacts (Gate C)**

Companion-nature coverage map (real `phot_sed` evidence vs analytic fallback, WD/dark split both
ways); RV gate pass-rate diagnostic on the real summary set with `skipped_no_rv` accounted for;
follow-up SF histogram match; adoption dates tracked under `config/target_lists/derived/`; the Gap-2
decision recorded. **Stop.**

---

### Wave D — forward-model activation

**Roster:** #41–#43. **Class:** heavy — run narrow. **Last authorized wave.**

**Entry state**

| Fact | How to check |
|---|---|
| Gates A, B, C passed | their artifacts |
| Every sample feeding inference reproduces exactly | `sample_reproduction_report` |
| Q1 signed off under #112 / PR #125 | `CONTINUATION_PLAN.md` §15 |

**Tickets**

#41's **first task is to measure what mock size the six-panel gate actually needs**, and report it —
this is the number that sizes the eventual ziggy/lux ask. Then demonstrate the astrometric validation
gate: the El-Badry 2024 six-panel comparison (P_orb, G, 1/parallax, eccentricity, `f_m`, cos i) plus
solution-type-fraction agreement. Flip each reproduced sample to `forward_model` and produce
`sample_selection_function` survival curves against M2, P_orb and G; produce `mode_divergence`
explained and quantified by the mass assumption; populate `sample_overlap_matrix` on the real
three-way overlap and verify the Q1 inclusion-indicator formulation does not double-count.

**Do not assume**

- That a sample may be enabled in `forward_model` because it is "close". Exact or omitted.
- That the overlap is pairwise. It is three-way: El-Badry 2024 drew from Andrews, and El-Badry 2026's
  subsample 3 *is* Andrews restricted to `G < 15`.
- That failing the six-panel gate is a calendar problem. It blocks science use of the astrometric
  selection function entirely (`ARCHITECTURE.md` §4); the calendar slips before the gate does.
- That laptop mock scale is the production scale. Measure and report the gap.

**Exit artifacts (Gate D)**

Six-panel and solution-type-fraction validation green; survival curves smooth and monotonic where
physically expected; `mode_divergence` explained; overlap matrix consistent with Q1; DR3/DR4 audit
clean; the **measured mock-size requirement**. **Stop — and hold the ziggy/lux scope conversation
here (§2.3).**

---

### Waves E–H — cluster readiness, production campaign, validation, paper

**Not authorized (§2).** Retained so the shape is not lost: E sizes and validates the dynesty recipe
on lux (nlive/dlogz/maxcall, wall-clock, checkpoint/restart, MC volume against
`physics.mc_noise_threshold`); F runs the production campaign under both mass-function models with
multi-seed robustness and the two v1 model comparisons; G validates at production settings (SBC
coverage, Gaia BH1/BH2/BH3 benchmarks, comparison-only catalogs, the age-stratified WD check); H
produces the science figures and the paper with its reproducibility appendix. Bin edges are fixed
before E produces any real posterior. These open when the laptop workflow is smooth and the Wave D
stop settles the compute ask.

## 8. Sequencing

```
  Wave -1     Wave 0      Wave A        Wave B      Wave C      Wave D      │  E–H
  ########    ########    ##########    ########    ########    ########    │  (not authorized)
  └─ STOP     └─ STOP     └─ STOP       └─ STOP     └─ STOP     └─ STOP     │
                          ◄──── C runs parallel to A and B ────►
```

There is no calendar here on purpose. Each wave ends in a hard stop for operator review and explicit
approval (§5.9); the next begins when approval lands. Wave −1 runs alone: it rewrites `main`, so nothing may build on top of it concurrently. Wave C runs
parallel to A and B — they were scoped to touch disjoint modules (`sample_selection` / `elbadry*_selection` / `janssens_mass` versus
`mass_derivation` / `companion_nature` / `rv_adapter` / follow-up `forward_model`).

Concurrency is bounded by laptop memory, not a fixed count (§5.6): Wave 0 runs wide, Waves A and D run narrow, and Wave −1's #52/#53 run one at a time.

**Checkpoints.** Review/Integration (#15) merges `config/fragments/` into `config/config.yaml` and
sweeps pre-merge config coherence at each wave boundary. Verification (#49) runs continuously and
re-runs the full suite at each boundary, and does the branch and worktree teardown for the wave's
merged work.

## 9. Cross-cutting work

**Documentation sync** (each wave): `ARCHITECTURE.md` and `CONTINUATION_PLAN.md` are updated by PR
**before** any design change lands. `CLAUDE.md` and `README.md` status sections refresh at each
checkpoint, and this document's status line names the current wave.

**Sibling-repo changes** (`dark-hunter_rv`, `dark-hunter_sed`): same rules — full local pytest before
any PR, `strict-workflow` issues/PRs. The `sed_summary.json` covariance block (§3.2 Gap 2) is the one
upstream change the authorized scope expects to need. Per `docs/RV_SUMMARY_JSON.md`, any breaking
field-name change upstream requires a docs PR in pop first, against `FOUNDATION_INTERFACE_FREEZE.md`.
Never read either repo's production `output/` live — snapshot with a timestamp, preserving mtimes,
which the RV gate now orders by.

**Open questions.** Q2, Q4, Q7, Q9, Q10, Q11 and Q15 attach to the waves that unblock them (§6). Each
closes with a decision recorded in `CONTINUATION_PLAN.md` §15's Resolved table, naming the measurement
that settled it. Q7 is a *measurement*, not a preference — Wave 0 takes it.

**v2 backlog** (explicitly not scheduled): fully joint inference warm-started from v1; triples enabled
with a real `P(triple)`; the DR4 path including epoch-level outlier rejection and the fast-vs-complete
cross-validation; the `accel_jerk` sample; SB1 as an inference entry point with `sin³i`
marginalization; RV per-point outlier rejection. Work drifting into any of these is scope creep — file
it as an issue (§5.3), do not do it.

## 10. Risks and contingencies

| # | Risk | Trigger | Response |
|---|---|---|---|
| C1 | Auto-merge lands two PRs that pass alone and conflict together | #49's post-merge checks | This is the failure mode auto-merge invites and the reason #49 exists. #49 pins it with a regression test and escalates; the orchestrator dispatches the fix. Do not let #49 fix it — a verifier that writes fixes stops being an independent check. |
| C2 | The dry-run figure gets mistaken for a result | any | The Wave 0 labeling requirement exists for this: caption, run-file tag, and report header each name the synthetic stand-ins. |
| C3 | A wave inherits a distorted belief from an earlier one | any | Every wave's Entry state is verified independently and every wave carries a `Do not assume` list (§7). Where a handoff note and a check disagree, the check wins and the disagreement is escalated. |
| C4 | The Andrews 352-vs-106 discrepancy is a genuine difference from the paper's method | #34 exhausts packing / floor / prefilter explanations | Document it as a finding, freeze `andrews2022.yaml`, carry the discrepancy as an explicit systematic. Do not retune to 24. |
| C5 | The joint censoring model is unidentified in practice | #37 | `CONTINUATION_PLAN.md` §4.8 is explicit: escalate rather than falling back to dropping rows. |
| C6 | Deferring ziggy/lux costs the December window | §2.3 | Accepted deliberately: a fast campaign on untrustworthy inputs produces an untrustworthy posterior. Revisit at the Wave D stop, when the gates are green and #41 has measured the compute ask. If the window closes first, the v1 result moves to whatever compute follows and the plan says so rather than rushing Wave A. |
| C7 | The laptop cannot reach the mock scale the six-panel gate needs | #41's measurement | That measurement **is** the finding — it sizes the ziggy/lux ask. Do not weaken the gate to fit the laptop. |
| C8 | Concurrency melts the laptop | memory pressure, swapping | Heavy/light classification and a hard cap on heavy sessions (§5.6). Start at two heavy, raise only while nothing swaps. |
| C0 | A merged PR silently drops commits, so documentation describes code that is not on `main` | observed: PR #134 landed 1 of 4 commits | The Wave 0 reconciliation ticket fixes this instance and reports whether other merges did the same. Going forward it is #49's job: verifying a merge includes checking that what the PR claimed to land is on `main`. |
| C10 | Bad photometry points shift ΔBIC in a way the fixed `delta_bic_threshold: 10.0` was not tuned for | unresolved upstream cleaning | The adapter records n_data and the cleaning settings per summary so the dependence is measurable; if the weights prove sensitive, the threshold is re-derived rather than nudged, and the re-derivation is its own issue. |
| C9 | Silent fixes accumulate outside the issue ledger | #49 review of PR diffs against issue scope | Bounce the PR regardless of whether the change was an improvement (§5.3). |

## 11. Definition of done

### 11.1 Authorized scope — "running smoothly on the laptop"

- Gates −1, 0, A, B, C and D passed, each with its artifacts, each approved at its stop.
- `main` contains everything it is documented as containing, and every reproduction number on
  record was measured on `main`.
- Every enabled sample reproduces its published N exactly; every sample feeding inference is in
  `forward_model` mode with a validated selection function.
- One shared spuriousness model reproduces both astrometric validation targets from a single
  parameter set.
- The astrometric selection-function validation gate and the solution-type-fraction diagnostic are
  green, and the mock size the gate requires is measured and recorded.
- A full fourteen-stage run completes cleanly and repeatably, resumes correctly, and its run file
  records config checksum, host profile, seeds, commit hashes, and the gaiamock version triple.
- No stale branches or worktrees; no untracked run files; the issue ledger accounts for every change
  on `main`.
- Docs reflect the delivered system; `docs/SELECTION_REPRODUCTION_STATUS.md` green or retired.

### 11.2 Full v1 — after the boundary moves

Adds: a production dynesty campaign with multi-seed posterior agreement under both mass-function
models, producing the two-tier `dN/dM` and the two v1 model comparisons; `companion_nature_likelihood`
on real `dark-hunter_sed` model comparison for a documented fraction of candidates, with the
analytic-fallback fraction reported as a systematic; SBC coverage and the Gaia BH1/BH2/BH3 benchmarks
passing, with comparison catalogs appearing only as comparisons; every figure regenerable from a
recorded run ID; a paper with a limitations section built from the recorded v1 scope boundaries.
