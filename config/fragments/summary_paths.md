# RV / SED summary on-disk layout

Pop consumes **existing** per-star JSON summaries from `dark-hunter_rv` and
`dark-hunter_sed`. Paths live only in config (`null` = attachment / snapshot
consume disabled). No absolute paths in Python.

## SED (`mass_derivation.sed_summary_root`)

Shared across DR modes (physics/queue knobs). Filename template must match
upstream `darkhunter_sed.posterior.sed_summary_path`:

```
{sed_summary_root}/Gaia_DR3_{source_id}_sed_summary.json
```

Default root: `data/sed_summaries` (under repo root; `data/` is gitignored).
Populate by snapshot/copy (never live-read production trees):

```bash
mkdir -p data/sed_summaries
# timestamped snapshot recommended; symlink OK for local dry runs
cp -a /path/to/dark-hunter_sed/output/sed_summaries/. data/sed_summaries/
```

When a file is present → refined stage sets `FitTier.full_uberMS`.
When missing → queue/`darkhunter_sed` fit path (watch-list first, then expand).
`uberms_m1_prior_max_msun` / `uberms_m1_watchlist_fraction` still drive the
watch-list diagnostic.

Fixtures: `tests/fixtures/sed_summaries/Gaia_DR3_*_sed_summary.json`.

## RV (`dr3.rv_summary_root` / `dr4.rv_summary_root`)

DR keys are **independent** even when values match. Upstream filename
(`darkhunter_rv.rv_summary_json.rv_summary_json_path`):

```
{rv_summary_root}/Gaia_DR3_{source_id}_summary.json
```

Defaults: `data/dr3/rv_summaries`, `data/dr4/rv_summaries`.

```bash
mkdir -p data/dr3/rv_summaries
cp -a /path/to/dark-hunter_rv/output/Gaia_DR3_*_summary.json data/dr3/rv_summaries/
```

`attach_rv_summaries` runs at `data_acquisition` and again at
`rv_astrometry_gate` so a later root fill does not require re-querying Gaia.
Systems with JSON epochs are scored; passers feed `joint_orbit_fit`
(`OrbitTier.joint_astrometry_rv`). Missing/failed stay skipped with an
explicit reason — never silent drop.

Gate ordering (config): `priority_source_ids` (e.g. Gaia BH1/BH2) first,
then public/`external_rvs`, then **newest on-disk summary mtime**. When
snapshotting JSON from `dark-hunter_rv/output`, preserve `*_summary.txt`
mtime onto the JSON (`os.utime`) so stale pre-pipeline summaries sort last.

**Escalate:** if `Gaia_DR3_*_summary.json` is absent upstream (only
`*_summary.txt`), do not invent a second format — land / backfill JSON in
`dark-hunter_rv` first. Many NSS Orbital solutions lack `semi_amp_primary`
(Thiele–Innes only); those stay `missing_astrometric_elements` until K is
derived (Joker / predicted-K + inclination).

Fixtures: `tests/fixtures/rv_summaries/Gaia_DR3_*_summary.json`.
