---
name: strict-workflow
description: Global standards for architecture, GitHub workflow, testing, checkpoints, and constraints. Use for all code generation, debugging, and features.
---
# Strict Workflow Protocol

## 1. Core Directives
* **Authority:** User domain knowledge is absolute. Contradict only with empirical evidence.
* **Verification:** Check live files/logs before forming hypotheses. Never assume codebase state.
* **Scope:** Execute exact instructions only. No unsolicited refactors, diagnoses, or next steps.
* **Freshness:** Uploaded files become obsolete after any code edit. Rely strictly on live files.
* **Constraints:** Negative constraints (e.g., "Do not...") are absolute hard stops.
* **Visuals:** State inability to verify images/plots. Rely on data arrays or request human verification.

## 2. Architecture & Planning
* **Plan & Halt:** Generate a structured Task List. STOP. Wait for explicit user approval before writing code.
* **Simplicity:** Maintain flat directory structures. Do not prematurely split files or nest folders.

## 3. GitHub Operations
* **Issues:** Create for bugs/features. Ignore typos/whitespace. Match format of `UCSC-Transients/dark-hunter_rv/issues`.
* **Issue Evolution:** Comment on existing issues if scope changes. Create and link new issues for major blockers.
* **Pull Requests:** Match `dark-hunter_rv/pulls` style.
  * *Docs:* Detail solutions and explicitly link issues (e.g., "Resolves #42").
  * *Tests:* Configure/require automated GitHub CI tests. Include test status in PR checklists.

## 4. Testing & State Management (Checkpoints)
* **Testing:** Execute `pytest` for unit, regression, smoke, and linting at each stage.
* **Micro-Commits:** Stage and commit locally immediately after `pytest` passes.
* **Commit Format:** `git commit -m "[AI Checkpoint] <precise description>"`.
* **Purpose:** For local `regression-hunter` use only. Do not push micro-commits to remote.

## 5. Code Standards
* **Documentation:** Detail all parameters, use cases, and limitations per function.
* **Typing:** Enforce strict Python type hints.
* **Performance:** Vectorize numerical/array operations. Eliminate unused imports.
* **Dependencies:** Log new libraries and update dependency files within the same PR.
