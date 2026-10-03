# gate339: corrected El-Badry six-panel (#339 Phase 1)

This directory keeps the run that produced the corrected six-panel figure for #339 Phase 1
(PR #359, merged as `6dd317b`). The worktree that held the run was torn down, so its outputs
are kept here.

| File | What it is |
|---|---|
| `run_20261001-075704-92caa8a.yaml` | The run manifest, copied verbatim from the `fix/elbadry-six-panel-339` worktree. The parent run is `20260930-022223-672b092` (gate301). Stages through `companion_nature_likelihood` are copied forward from the parent, at `code_commit` `672b092`. `selection_function_astrometric` onward ran at `92caa8a`, the branch tip merged by #359. |
| `figures/elbadry_six_panel.png` | Six-panel figure: the real NSS sample against the gaiamock mock, gated on every orbital solution cut. |
| `reports/elbadry_six_panel.txt` | The figure's report: sample definitions and per-panel KS values. |

Notes:

- This is a historical record, like `docs/gate301/`. Artifact paths in the manifest point to
  `output/` directories that no longer exist (a torn-down worktree, and the gate301 run in the
  primary checkout). To reproduce, check out `92caa8a` and re-run from the gate301 parent.
- The mock has 12 accepted orbital solutions out of 500 realizations, so the KS values are
  small-N. The `f_m` (p = 9e-8) and period (p = 1e-3) mismatches are open work under #339 and
  #344; nothing here is a validated selection function.
- Two earlier attempts from the same worktree (`20261001-072133-21e60a2`, crashed, and
  `20261001-072350-7bd54db`, superseded) were purged with `scripts/purge_run.py` and are not
  kept.
