# Wave 0 gate artifacts (issue #201)

> **SYNTHETIC INPUTS — PLUMBING CHECK, NOT A SCIENCE RESULT.**
>
> Nothing in this directory is a measurement of the compact-object mass function. No number,
> curve or axis value here may be quoted in a figure, caption, talk or paper.

These files are the Wave 0 exit artifacts (`docs/EXECUTION_PLAN.md` §7). They are tracked in git
because `output/` is gitignored and is deleted with the worktree that produced them, while the
gate they satisfy is reviewed by a person afterwards.

| File | What it is |
|---|---|
| `dndm_by_class.png` | The product figure: total `dN/dM` plus BH, NS, WD, other and outlier, to `docs/PLOTS.md` standards. The banner and every stand-in are burned into the caption. |
| `dndm_by_class_caption.txt` | The same caption as text, so it is greppable. |
| `dry_run_report.txt` | Full report: the stand-ins with their config keys and resolved values, per-stage status / wall clock / RSS, and the run plan exactly as printed before execution. |

The run itself is the manifest under `runs/`, tagged `dry_run: true` with a `synthetic_stand_ins`
block. Find it with:

```bash
grep -l '^dry_run: true' runs/*.yaml
```

Regenerate with `python scripts/run_dry_run.py --host-profile laptop --snapshot <meta.yaml>`;
see the README's "Labeled dry run" section.

## Artifact retention (2026-09-27)

The HDF5 artifacts of the Wave 0 reference run `runs/20260920-033431-121d6de.yaml` lived under
`output/` of the `wave0-dry-run-harness` worktree, which the manifest's `artifact_path` entries
point to by absolute path. On 2026-09-27 the operator (Ryan Foley) decided to delete that worktree
and its `output/` (~1.0 GB) to recover laptop disk. Once it is removed, those `artifact_path`
entries are dangling by design; the run file is kept unedited as the record. The run is
reproducible from what the manifest records — config checksum, `host_profile: laptop`, seeds,
gaiamock version triple, and the snapshot
`data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml` — via the regeneration command
above. The files in this directory are the retained gate evidence.

On 2026-09-29, by operator decision (Ryan Foley, #318), the early-September runs `20260904-170748-1a20ee9`, `20260904-184235-0d7c84b`, `20260906-063103-4a3b07a` and `20260908-183807-5a1c609` were purged, along with their primary-checkout `output/` directories (including the shared `output/20260904-152655-674c989/`, ~1.0 GB, which PR #304 had kept).
