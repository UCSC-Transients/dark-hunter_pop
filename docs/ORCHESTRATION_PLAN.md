# dark-hunter_pop — Orchestration Plan

Companion to `ARCHITECTURE.md`. Decisions locked 2026-08-25; see that document for
authoritative technical detail.

## 1. Cursor mechanics

- **Primary mechanism: manually-launched Agents Window sessions, one per subagent role, each in
  its own `git worktree` on its own branch, with the model explicitly hand-picked per session.**
  Cursor's automatic subagent dispatch (`/multitask`, built-in Task-tool subagents) does not
  currently offer reliable control over which model a subagent uses, so it isn't suitable for the
  differentiated model-tier cost strategy below.
- `/multitask` is fine *inside* one subagent's own session for a large low-judgment chunk of that
  subagent's own task — not as the mechanism for assigning the roles themselves.
- Keep truly-simultaneous sessions to **2-3**; Cursor supports more but merge-conflict complexity
  scales non-linearly with concurrent agent count.
- Orchestrator session (top-tier model) holds this plan and `ARCHITECTURE.md`, breaks work into
  subagent tickets, and is periodically swept by you against the roster below — this human-
  mediated sweep is the actual orchestration loop.

## 2. Repo & environment

- `UCSC-Transients/dark-hunter_pop`, local base directory
  `/Users/rfoley/darkhunter/pop/dark-hunter_pop/` (repo root itself). Package: `darkhunter_pop`.
- `venv` (or agreed alternative); Linux primary target, macOS-friendly where reasonable. No conda
  requirement known for the December supercomputer target.
- `gh` CLI credentials available; used for all issue/PR creation, on this repo and on
  `dark-hunter_rv`/`dark-hunter_sed` when extended for this project.
- Git worktrees per active subagent branch.
- Large gaiamock-mod assets hosted on an immutable GitHub Release (e.g. `gaiamock-mod-v1`); see
  `ARCHITECTURE.md` §1.1. Default healpix pack is not hosted.

## 3. Skills active in every subagent session

`strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow` (project-specific).

## 4. Subagent roster

| # | Subagent | Depends on | Model tier | Rationale |
|---|---|---|---|---|
| 0 | **Foundation** (constants, schemas incl. `ParameterSet`, config schema, `run_management.py` stage registry/caching/run-file layer, gaiamock-overlap audit) | — (first, sequential, blocking) | Top | Everything else builds against these interfaces; the run-management layer in particular is a contract every stage subagent's code must conform to. |
| 1 | `data_acquisition` | Foundation | Mid | Query/ETL against a frozen schema; N-configurable quality-cut bins are the one piece worth real care. |
| 2 | `selection_function_astrometric` (gaiamock wrapper, DR3/DR4 modes, validation gate) | Foundation | Top | Physics-critical; the validation gate determines whether the selection function can be trusted at all. |
| 3 | `selection_function_followup` (+ DR3/DR4 audit function, Google Sheets adoption-date mining + going-forward weekly snapshot) | Foundation, target-list adoption-date access | Top | Genuinely subtle statistics (outcome-dependent selection risk); the audit function and the Sheets revision-history reconstruction are both correctness-critical, fiddly pieces. |
| 4 | `mass_derivation_bulk` + `mass_derivation_refined` (TAG10 + Santos correction, uberMS queue integration) | Foundation, #1 | Top | Directly determines the input sample; TAG10 implementation and uncertainty propagation need care. |
| 5 | `dark-hunter_rv`/`dark-hunter_sed` extensions (WISE/DECam, JSON output; Joker integration already done upstream — this is now wiring/consumption plus the remaining photometry work) | Foundation | Mid | Reduced scope now that The Joker is installed and integrated server-side; mostly adapter/integration work remains. Conform to Phase 0 schemas; ask before breaking them. |
| 6 | `rv_astrometry_gate` + `joint_orbit_fit` | #4, #5 | Top | The gate statistic (per-instrument nuisance terms, chi2/dof) and the gate→joint-fit→outlier-routing logic are all physics-critical. |
| 7 | `companion_nature_likelihood` | #4, #6 | Top | Core statistical model determining WD vs. other vs. dark; largest-uncertainty regime is exactly where precision matters most. |
| 8 | `triples` stub (off by default) | Foundation, #5 | Mid | Well-specified but unbuilt-for-real-use in v1; mostly interface/stub work. |
| 9 | `population_model` (multiplicity/type hierarchy, non-parametric mass function) | #7, #2, #3 | Top | Mixture-model architecture, hard/soft truncation rules, and the outlier-class rate all need careful judgment. |
| 10 | `sensitivity_analysis` | Foundation | Top | Serves both overall dimensionality and per-class covariate selection; statistical design matters more than code volume. |
| 11 | `inference` (staged-but-connected Poisson point process, dynesty) | #9, #2, #3, #10 | Top | The core statistical result; robustness protocol and model-comparison design are high-stakes. |
| 12 | `plotting` + diagnostics infrastructure (minus SBC design) | Foundation | Mid | Well-specified rendering/logging infrastructure, shared between diagnostics and product figures. |
| 13 | Validation/SBC design (recovery tests, benchmarks, cross-checks) | #11, #12 | Top | Statistical validity of the calibration tests matters as much as the inference itself. |
| 14 | Documentation | trails others | Light | Low-judgment, high-volume writing. |
| 15 | **Review/Integration** (merges config fragments, checks physics/interface consistency across PRs, manages queue) | runs continuously | Top | The only subagent whose job is catching cross-module inconsistency. |
| 16 | Main program (wires all stages in order via `run_management.py`, per config) | all of the above | Mid | Once interfaces are frozen, this is sequencing code. |

## 5. Phased sequencing

Supercomputer access ends **December** — the compute-heavy `inference` stage (#11) needs a minimal
end-to-end path working *early*, even against a simplified population model, so there's runway to
actually use the cluster before it's gone.

- **Phase 0** (sequential, blocking): #0 Foundation. Preceded by the docs/decisions PR that locks
  `ARCHITECTURE.md` / this plan.
- **Phase 1** (parallel, ≤3): #1 `data_acquisition`, #2 `selection_function_astrometric` +
  validation gate, #5 `dark-hunter_rv`/`_sed` extensions. Kickoff issues + paste prompts:
  `docs/PHASE1_KICKOFF.md` (umbrella #28; children #29–#31).
- **Phase 2** (parallel): #3 `selection_function_followup` + audit function, #4 mass derivation,
  #10 `sensitivity_analysis`, #12 plotting/diagnostics infrastructure. Kickoff issues + paste
  prompts: `docs/PHASE2_KICKOFF.md` (umbrella #35; children #36–#39; Review/Integration #40).
- **Phase 3** (parallel): #6 `rv_astrometry_gate` + `joint_orbit_fit`, #8 `triples` stub.
  Kickoff issues + paste prompts: `docs/PHASE3_KICKOFF.md` (umbrella #47; children #48–#49;
  Review/Integration #50).
- **Phase 4**: #7 `companion_nature_likelihood`, #9 `population_model`. Kickoff issues + paste
  prompts: `docs/PHASE4_KICKOFF.md` (umbrella #55; children #56–#57; Review/Integration #58).
- **Phase 5 (priority — get a minimal path here before the December cutoff)**: #11 `inference`,
  even against a placeholder population model at first, to validate the end-to-end
  Poisson-likelihood + dynesty machinery on the actual compute target early.
- **Phase 6**: #13 Validation/SBC completion, full diagnostic suite, all validation gates passing.
- **Phase 7**: #16 Main program wiring, #14 Documentation, final review.

\#15 Review/Integration runs continuously from Phase 1 onward.

## 6. Git / PR / issue workflow

Applies uniformly to `dark-hunter_pop`, `dark-hunter_rv`, `dark-hunter_sed`:

1. Subagent works in its own worktree/branch.
2. `[AI Checkpoint] <description>` micro-commits locally after each passing `pytest` run (local
   use only, for `regression-hunter`; not pushed).
3. **Full local `pytest` pass required before opening a PR** (at least the required marker set).
4. PR via `gh`: detailed description, links resolved issues ("Resolves #NN"), test checklist,
   requires GitHub CI green (`tests` = `unit or physics or api`).
5. Review/Integration subagent checks physics/interface consistency across open PRs, merges
   modular config fragments into the single `config.yaml`.
6. You do the final human merge to `main` (sole developer; merge = approval).
7. **GitHub Issues**: one issue = one thing. Prefer fine-grained issues so partial work can close
   independently; optional umbrella issues may track children. Track major changes, features, and
   bugs on GitHub — not only in chat.
8. **Docs-first**: update `ARCHITECTURE.md` / this plan (via PR) before implementing a design
   change. Adjust as we go, but every step needs a written plan.

Branch protection on `main`: PR required, required status check `tests`, zero required reviews,
admin bypass allowed.

## 7. Open items

None currently flagged. See `ARCHITECTURE.md` §8.
