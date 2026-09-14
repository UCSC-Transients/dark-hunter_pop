# `phot_sed` adapter fixtures

Synthetic `dark-hunter_sed` Path-2 summaries for `tests/test_phot_sed_adapter.py`
(issue #197). Key set copied from a real
`output/phot_sed/Gaia_DR3_<id>_1star_summary.json` written by
`darkhunter_sed.phot_sed_fit`; values are invented.

| source_id | contents |
|---|---|
| `900001` | all three models (`1star`, `wd`, `2star`) |
| `900002` | `wd` only |
| `900003` | *no files* — the "none present" case |
| `900004` | malformed `1star` JSON |
| `900005` | `1star` whose `gaia_id` disagrees with its filename |
| `900006` | models disagree on `n_data` |
| `900007` | `bic` low enough that the recovered chi2 goes negative (offset case) |

**The `wd` fixtures assume the documented file contract** —
`Gaia_DR3_<id>_wd_summary.json` carrying `bic` / `n_free` / `n_data`. Upstream
`phot_sed_cli --model wd` does not write that today: it writes
`<out_dir>/<gaia_id>/wd/wdstar_<atm>_<ifmr>_summary.json` with `logevidence`
only. That gap is tracked separately; the adapter is deliberately not taught to
read the other layout.
