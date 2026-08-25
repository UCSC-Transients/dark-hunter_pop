---
name: regression-hunter
description: Strict Git checkpoint protocol to isolate and fix regressions. Use when working functionality breaks.
---
# Regression Resolution Protocol

1. **Identify:** State expected vs. failing behavior clearly before proceeding.
2. **Isolate (Diff Loop):** 
   - Ignore chat memory. 
   - Execute `git log --grep="\[AI Checkpoint\]"`.
   - Compare states using `git diff HEAD~1`. 
   - Iterate backwards (`HEAD~1 HEAD~2`) until the exact bug-introducing commit is found.
3. **Remediate:** Modify *only* lines related to the isolated diff. Do not touch surrounding logic or formatting.
4. **Verify (Hard Rule):** Execute tests (e.g., `pytest`) or load the local web instance to prove resolution. Reading code is insufficient.
5. **State Management:**
   - **Success:** Commit immediately: `[AI Checkpoint] Regression fixed: <desc>`.
   - **Failure:** Execute `git reset --hard`. Iterate further back in history.
