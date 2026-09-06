# Escalate: dark-hunter_rv JSON summaries not on main

**Status:** escalate — do not invent a second summary format in pop.

## Gap

Pop `attach_rv_summaries` expects upstream files:

```
{rv_summary_root}/Gaia_DR3_{source_id}_summary.json
```

matching `darkhunter_rv.rv_summary_json` (`phase1/rv-summary-json` branch /
commit emitting `Gaia_DR3_*_summary.json`). On `dark-hunter_rv` **main** as of
this wiring PR, production `output/` still has `*_summary.txt` only — **0**
`*_summary.json` files. Fixture JSON under `tests/fixtures/rv_summaries/`
conforms to that upstream schema (`schema_version`, `pipeline_epochs`,
`nss_orbital`, …).

## Required upstream work

1. Land `phase1/rv-summary-json` (or equivalent) on `dark-hunter_rv` main.
2. Backfill / regenerate `Gaia_DR3_*_summary.json` beside existing `*_summary.txt`.
3. Snapshot/copy into pop `data/dr3/rv_summaries/` (see `summary_paths.md`).

Until then: fixture dry-path proves scoring + joint fit; live runs with empty
`data/dr3/rv_summaries/` complete the gate with `skipped_no_rv` reasons and
explicit attach diagnostics — never a silent drop.
