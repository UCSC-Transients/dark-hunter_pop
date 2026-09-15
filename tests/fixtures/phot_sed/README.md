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
| `900008` | all three models; `wd` uses the *real* upstream field set (issue #215) |

**The `900001`/`900002`/.../`900007` `wd` fixtures use a placeholder field set**
(`param_names`: `EEP`/`M`/`FeH`/`Av`/`parallax`/`sigma_int`, `n_free: 7`) that
predates the real upstream contract — see history below.

**`900008`'s `wd` fixture is copied field-for-field from the real
`dark-hunter_sed` upstream fix** (`darkhunter_sed.wd_model.write_wd_pop_summary`,
merged `dark-hunter_sed#67`, `feat/wd-pop-summary`, closing pop issue #206):
`param_names` is the real `WD_STAR_PARAM_NAMES` (`EEP1`, `M1`, `FeH`, `Av`,
`parallax`, `Teff_WD`, `logg_WD`, `sigma_int`; `n_free: 8`), and it carries the
two extra keys the real writer adds (`atm_type`, `ifmr` — the canonical
DA+MIST variant) that the earlier placeholder fixtures did not have. The
adapter ignores both extras (it only reads `gaia_id`/`source_id`, `bic`,
`n_free`, `n_data`, `logz`, `best_theta.sigma_int`, and
`CLEANING_SUMMARY_KEYS`), so this fixture is the regression proof that the
extra keys are harmless and every field the adapter *does* read lines up.

**History (issue #206, resolved).** Upstream `phot_sed_cli --model wd` used to
write only `<out_dir>/<gaia_id>/wd/wdstar_<atm>_<ifmr>_summary.json` with
`logevidence` only — no `bic`/`n_free`/`n_data`, and not at the pop-facing
path. That gap is now closed by `dark-hunter_sed#67`: `--model wd` also writes
`Gaia_DR3_<id>_wd_summary.json` at the pop-facing path, matching the
`1star`/`2star` contract field-for-field (extra `atm_type`/`ifmr` keys
ignored). No real Gaia source in either repo's local checkout yet has real
(non-placeholder) `1star`, `2star` *and* `wd` dynesty output at matching
`n_data`, so issue #215's verification is schema-level (the `900008` fixture
above), not an end-to-end real-candidate run.
