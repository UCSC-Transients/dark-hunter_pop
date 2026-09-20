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
