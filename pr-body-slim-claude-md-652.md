Slim CLAUDE.md (41.4 KB -> ~20.6 KB). Status section moved verbatim to docs/STATUS.md; full module table, data flow, literature layer, sibling-repo interfaces, data/config paths, dependencies and full gotchas moved verbatim to docs/REPO_REFERENCE.md. CLAUDE.md keeps doc-authority table (+2 rows), condensed module map, conventions, commands, run management, git/PR rules, open questions, v1 scope, condensed gotchas, compute rule. ORCHESTRATOR_PROMPT 4.5 and two EXECUTION_PLAN lines repointed to docs/STATUS.md.

Closes #652

## Test plan
- `pytest -m "unit or physics or api"`: 1257 passed, 2 skipped, 1 failed (`test_gaiamock_vendor::test_submodule_present`, worktree has no gaiamock submodule; worktree-only).

## Interface changes
None (docs only).
