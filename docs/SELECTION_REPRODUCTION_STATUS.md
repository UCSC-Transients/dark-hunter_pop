# Selection reproduction status (agent hand-off)

**Purpose:** the live record of which literature sample-selection reproduction gates pass and which
fail, and of what each number actually was the last time someone ran it. Read this before claiming a
reproduction is done, and before trusting any count quoted elsewhere. Prefer binding and physics fixes
over editing frozen cut thresholds.

**Provenance — read this before using any number below.**

> **All §3 numbers were measured on `main` @ `c905575`**, in the primary checkout at
> `/Users/rfoley/darkhunter/pop/dark-hunter_pop`, against the real `+enrich+mc10000` parent cache
> (443,211 rows), on 2026-09-13, as the Wave −1 closing baseline (issue #144). Read-only: nothing was
> rebuilt, no enrichment job re-run, no frozen threshold touched.
>
> **Exception: §3.1.2's numbers were measured on `main` @ `9f70a53` (2026-09-26/27, issue #257)**,
> against a **rebuilt** `+enrich+mc10000` cache (new `mtime`, same 443,211-row count) that now uses
> the `flame_or_uniform_draw` M1 method instead of fixed `M1=1.0`. **This is the first valid
> measurement of that method** — a prior attempt on 2026-09-26 (`main` @ `dea93c3`, PR #266) reported
> numbers from a rebuild that turned out to be a silent no-op: `attach_mc_to_selection_cache.py`'s
> per-job MC payload never actually received the FLAME mass, so every source fell back to
> `Uniform(0.63, 1.0)` regardless of whether it had a real Gaia Apsis FLAME value. That defect and its
> fix are #267/PR #269; the invalid measurement and its write-up were reverted in PR #268/#266's
> merge commit `d2ad5ab`. **Treat everything PR #266 reported as if it never happened** — this
> section replaces it with a real measurement against the fixed code. Like the exception above, this
> is an **in-place rebuild of the same file path**: the fixed-`M1=1.0` version of `+enrich+mc10000`
> no longer exists on disk, so §3.1's original table, §3.1.1, and §3.2 (which does not depend on
> Andrews' MC output at all) still accurately describe what was true at `c905575`, but are no longer
> independently re-derivable from the cache currently on disk. **§3.3's `andrews2022_import` row
> (target 16) was re-measured against this rebuilt cache on `main` @ `a9397ba` (issue #270): 55,
> previously 19** (§3.3.0). The only other §3.3 counts that moved are the two unions downstream of
> it. Everything else in §3.3 is still at `c905575`.
>
> **§3.1.3's numbers (issue #280) were measured on `main` @ `2470fe5`** against that same rebuilt
> cache, read-only, using a paired targeted MC. Its production-method variant reproduces the cache
> exactly, and its fixed-M1 = 1.0 variant reproduces the `c905575` baseline exactly.
>
> **§3.3.3 (#275) was measured on `main` @ `2470fe5`** (#258's real maps landed), isolated worktree,
> astrometric-branch enrichment plus the window-only `σ_M̃2` MC. No count in the §3.3 table moved.
> The Simon row's status changed because of the #281 classifier/fixture fix, not because of a
> re-measured count.
>
> **§3.3.4 (#274) recounts the recorded membership sets** (`a9397ba` #270 and `1058e29` real maps)
> under the PI's per-solution-type distinct-star rule, using code off `main` @ `c07897e`. It does
> not re-evaluate anything. Only the El-Badry 2026 union moves: 1119 → 1069 and 1565 → 1507.

That provenance line is the point of this rewrite. Every "Last measured" number in the previous version
of this document came from the branch `fix/selection-reproduction-binding`, not from `main`, and three
of the four commits it cited were never on `main` at all — PR #134 merged only `7d5b22e`. The
reconciliation (#141, PR #150, merged as `03a452a`) put them there, and a further chain of Wave −1
fixes landed on top (PRs #155, #156, #159, #166, #170, #171, #173, #179). The numbers below are the
first ones in this document's history that describe `main`.

**Hard rule:** do **not** retune frozen thresholds in `config/selections/*.yaml` without a
`schema_version` bump and human escalation (`dark-hunter-pop-workflow` §1 / CONTINUATION_PLAN). The
gates below are red for physics and binding reasons. Closing them by moving a published cut would
produce a number that matches and means nothing.

**Who fixes these.** Wave −1 measured them; **Wave A owns closing them** (`EXECUTION_PLAN.md` §7).

Related issues: **#132** (NSS enrichment / K1 — largely unblocked), **#133** (`primary_ns_bh` 42 ≠ 47;
extinction / `a0` / nsstools, Q7), **#258** (extinction root cause was a dead `ExtinctionSpec`; now fixed with the
real Green2019/Lallement2019 maps — spectroscopic branch 151 exact, `primary_ns_bh` 46; see §3.3 /
§3.3.1). **#275** (the `M̃2` chain reproduces El-Badry 2026 Table 7; `sub_chandrasekhar` is a
selection-definition gap → **#284**; Simon last slot fixed as **#281**, stage wiring **#285**; §3.3.3). **#315** (landed: reproduction-mode `sub_chandrasekhar` is the Shahaf 2023b class-III cross-match with Eq. 5 on Shahaf's masses, which gives **exactly the paper's 22**; forward model unchanged; §3.3.7).

---

## 1. Local data artifacts (not in git)

| Artifact | Path | Notes |
|----------|------|--------|
| Photometry snapshot | `data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/` | Uncut literature parent |
| NSS enrichment | `data/dr3/gaia_snapshots/nss_enrichment/` | `corr_vec`, K1, significance; job **COMPLETED** — do not re-`--poll-job` |
| FLAME enrichment (#257) | `data/dr3/gaia_snapshots/flame_enrichment/` | 23 MB; `mass_flame`/`_upper`/`_lower` only, via sync fetch (`scripts/fetch_flame_enrichment.py --sync`); 241,196/443,205 rows carry a real value. Independent of `nss_enrichment` — never re-run that job to get this |
| Enrich-only parent cache | `…/20260826T234425Z_3d3f740b080c+enrich/selection_parent_rows.h5` | 691 MB; no Andrews `p_m2_above`; untouched by #257 |
| Enrich+FLAME cache (#257) | `…/20260826T234425Z_3d3f740b080c+enrich+flame/selection_parent_rows.h5` | 733 MB; `+enrich` plus `mass_flame`/`_upper`/`_lower` merged in (`scripts/merge_flame_enrichment_into_cache.py`), non-destructively — `+enrich` itself is never modified |
| Enrich+Andrews-MC cache | `…/20260826T234425Z_3d3f740b080c+enrich+mc10000/selection_parent_rows.h5` | 800 MB; 443,211 rows; rebuilt 2026-09-27 (~5h39m, 8 workers, `elapsed_s=20355.3`) against `+enrich+flame` under the `flame_or_uniform_draw` M1 method (#230/#257), with #269's per-job-payload fix in place — **superseded the earlier fixed-`M1=1.0` build and the invalid 2026-09-26 no-op build (#267/#268) this file used to hold**. **This is the one §3 is measured on now.** |
| Green2019 map (#258) | `data/dust_maps/green2019/bayestar2019.h5` | **Symlink** to the `dustmaps` package copy (727,902,836 bytes, md5 `ab815d2fd3068d1b81a1bd61fb18a722` = Dataverse `2EJ9TX/1CUGA1`); also linked at `~/.mwdust/green19/` for the `mwdust` cross-check test |
| Lallement2019 map (#258) | `data/dust_maps/lallement2019/map3D_GAIAdr2_feb2019.h5` | 809 MB; VizieR `J/A+A/625/A135`, downloaded 2026-09-27, md5 `e9262125307831a4a90b394e3b71f213` |
| El-Badry 2026 `E(B-V)` cache (#258) | `data/dust_maps/ebv_cache/elbadry2026_<fingerprint>.h5` | Native per-source map integrals for 349,594 keys, keyed `(source_id, ra, dec, parallax)`; built automatically on first evaluation (~45 s cold, ~11.6 GB peak for the full evaluation). Since #278 the fingerprint includes the SHA-256 of `dust_maps.py`, so **any** edit to that file rebuilds it cold; the #258-era `elbadry2026_9c8c2ef40427995c.h5` is orphaned by that change and is no longer read |

Rebuilding the Andrews MC cache is a multi-hour job and should not be done unless a landed change
actually invalidates it:

```bash
.venv/bin/python scripts/attach_mc_to_selection_cache.py --workers 10
```

That script **is on `main`** at `scripts/attach_mc_to_selection_cache.py`, landed via PR #150
(issue #141, merge SHA `03a452a`) — see §6. The cache it produces is already built, and is what
the numbers below were measured against.

Eval pattern: `_read_selection_parent_cache(…+enrich+mc10000/…)` then
`SampleSelectionRegistry(load_config()).selection(name).evaluate(rows, membership=…)`.
Pass `membership['andrews2022']` when evaluating El-Badry 2026 so `andrews2022_import` binds.

Cost, measured at `c905575` on the primary checkout: reading the cache takes ~18–26 s and ~2.8 GiB
resident; Andrews evaluates in ~4 s; El-Badry 2024 in ~6 s; **El-Badry 2026 takes ~22 min and ~6.8 GiB
peak**, because `σ_M̃2` is a lazy full-covariance NSS Monte Carlo evaluated at the `m2_error` cut.
Budget for that before starting a re-measurement.

---

## 2. What is already fixed (binding) — and this time it is on `main`

All six are present at `c905575`. Items 3, 4 and 5 are the ones that were previously documented as done
while living only on a branch.

1. Literature parent loads from the **uncut** Gaia snapshot (the quality-cut snapshot under-counts the
   parent).
2. NSS enrichment merge + Gaia `SOURCE_ID` → `source_id` normalize.
3. Andrews: `sigma_m2_msun` → `m2_msun_error`; missing `logg_apsis` binds as `None` (giant cut).
4. El-Badry 2026: **never** alias Andrews `sigma_m2_msun` → `sigma_m2_astrometric_msun`.
5. `σ_M̃2` is filled lazily at the `m2_error` cut with **fixed** Janssens `M̃1`
   (`propagate_fit_uncertainty: false`, CONTINUATION_PLAN §15 Q12). Since #284 the method is
   config-switched (`sigma_m2_tilde.method`: `analytic` default, `monte_carlo` alternative), and the
   provenance tag names it (`elbadry2026_{analytic_full|analytic_nsstools_blocks|mc}_m1_tilde_{fixed|janssens}`;
   legacy `elbadry2026_m1_tilde_fixed` still recognized).
6. Janssens YAML load is path-`lru_cache`'d and frozen against mutation (`janssens_mass.py`, PR #156).

The direct evidence that item 5 really landed is `sub_chandrasekhar` = **861**. On the raw branch,
before the σ fix, that count was **0**; a pre-reconciliation `main` gave **740** from the Andrews-σ
alias. 861 is the post-fix value, and it reproduces exactly at `c905575`.

---

## 3. Acceptance checklist — measured on `main` @ `c905575`

Targets come from the frozen YAML and CONTINUATION_PLAN. After changing extinction, `a0` / nsstools,
AMRF / `M̃2`, or MC packing, re-run the eval on `+enrich+mc10000` and rewrite this section **with the
commit the new numbers came from**.

The "Previous" column is the last value recorded before this baseline: measured on branch
`fix/selection-reproduction-binding` @ `5725e54`, whose content reached `main` as merge `03a452a`.

### 3.1 Andrews et al. (2022)

#### 3.1.4 #296 — reproduction mode is now the ATF selection notebook (current state; supersedes §3.1.1–§3.1.3 for reproduction)

**Provenance.** Measured 2026-09-28 on branch `feat/andrews-atf-notebook-296` (`main` @ `c07897e` + the
#296 commits, merged with `main` before the PR), in the isolated worktree
`../dark-hunter_pop-worktrees/andrews-atf-notebook-296` with `PYTHONPATH` → its `src/` and `data/`
symlinked to the primary checkout. The spec is the notebook that actually selected the published
sample. Ryan Foley (PI; co-author of Andrews, Taggart & Foley 2022) supplied it:
`data/reference/andrews2022_ATF_sample_selection.ipynb`, sha256 `343dc9e1…c5ccad`. It is gitignored
and not committed. The notebook functions used are `get_random_samples` / `find_massive` (cell 7),
`plot_system` (cell 17) and `calc_M2` (cell 34).

The columns come from `scripts/build_andrews2022_atf_columns.py --workers 8`, which ran in 217 s on a
warm file cache. It read the raw `nss_enrichment` NSS table, the `flame_enrichment` table and the
uncut snapshot `20260826T234425Z_3d3f740b080c`, all read-only. It wrote the sidecar
`data/reproduction_columns/dr3/atf_notebook_4181870cbce6ac8f.h5` (134,598 rows, 28 MB). Neither
`+enrich+mc*` cache was rewritten.

**Method.** `config/selections/andrews2022.yaml` is now at schema_version 3; `andrews2022_modified`
is at schema_version 2. The notebook is implemented exactly, including its quirks:

- **Pass 1.**
  - M1 is fixed at 1.0.
  - Each source gets 10⁴ draws from the 12×12 covariance, built from `*_error` and `corr_vec`
    **without `bit_index`**.
  - The source is **rejected** when `scipy.stats.multivariate_normal` refuses the matrix. Nothing is
    floored.
  - It is also rejected when `brentq` on [0, 1000] fails for **any** draw.
  - The source passes when `fraction(M2 > 1.4) > 0.95`, a strict inequality over **all** draws.
- **Pass 2.**
  - Rejected when GoF > 5.
  - Gaia BH1 is removed by hand. This is the existing exclusion.
  - The refined M1 is the UCO Lick mass, else FLAME, else U(0.63, 1.0).
    - Lick: `N(m, 0.1)` for 8 sources. The table is in the selection file with its provenance.
    - FLAME: `N(mass_flame, 0.1)`.
  - Rejected when `logg_gspphot < 3.6`, only for sources outside the Lick branch.
  - CMD cut: the line through (−0.5, −2) and (3, 9), i.e. slope **exactly** 11/3.5 and intercept
    9 − 3·slope (#255). Absolute magnitudes use the NSS parallax, with no extinction.
  - `brentq` on [0, 10⁴] per refined-M1 draw; any failure rejects.
  - Rejected when `mean(M2) < 3·std(M2)`.

These `atf_*` cuts apply **only in reproduction mode**. The schema_version-2 chain is kept verbatim
as `applies_to: [forward_model]`. As a check, forward_model `andrews2022_modified` still gives **64**
(1061 → 681 → 408 → 90 → 64), unchanged.

| Check | Target | fixed-M1 baseline (`c905575`) | FLAME at selection (`9f70a53`) | **ATF notebook (#296)** | Gate |
|-------|--------|------------------|------------------|--------------------------|------|
| Parent `Orbital` N | 134598 | 134598 | 134598 | **134598** | **OK** |
| After pass 1 (`atf_m2_probability`) | 106 | 352 | 1061 | **115** | FAIL by 9. All 24 published sources are in the 115; see the boundary note |
| Final reproduction N | 24 | 33 | 63 | **25** | FAIL by 1. It is a **strict superset** of the published 24 |
| Source-ID diff vs published 24 | — | +9 | +42 / −6 | **+1 / −0**: extra `6424213726885519744` | — |
| `andrews2022_modified` N (reproduction mode) | 25 | 34 | 64 | **26** | 25 + the same extra |
| Q9: `G < 15` among `andrews2022` | 16 | 19 | 55 | **16** | **OK** — exact |
| El-Badry 2026 `andrews2022_import` | 16 | 19 | 55 | **16** (see note) | **OK** |

Attrition (reproduction):

```
134598 → 130012 (atf_covariance_valid: 2747 singular, 1839 NaN inputs = bit_index 8179)
       → 129556 (atf_pass1_root_found: 456, every one m_f > g(1000))
       → 115 (atf_m2_probability) → 81 (atf_goodness_of_fit) → 49 (atf_giant_reject_logg)
       → 32 (atf_giant_reject_cmd) → 32 (atf_pass2_root_found) → 26 (atf_m2_3sigma)
       → 25 (explicit_exclusions, Gaia BH1)
```

**Implicit-rejection breakdown against our previous path.** The previous path uses
`propagate_nss_solution` at fixed M1 = 1.0, the cache's per-source seeds and the
`eig_rel_floor` / `eig_abs_floor` factorization. Re-run here, it reproduces the documented **352**
exactly. The table sorts those 352 by the reason the notebook removes each one:

| Reason | N of 352 | Note |
|---|---|---|
| SciPy covariance gate: "singular" | **235** | Our path factorized all 2747 such matrices by **plain Cholesky**, with no nugget and no clipping. SciPy's `allow_singular=False` check rejects any matrix with `min λ ≤ 1e6·eps·max|λ|` (≈ 2.2e-10). That check is relative, over a mixed-unit 12×12 (mas, days). All 235 sit below it (max ratio 1.3e-10). Only 6 lie within ×3 of the threshold, and 14 accepted sources lie within ×3 above it. **This is the 352-vs-106 driver.** |
| Any-draw root failure, `m_f > g(1000)` | 2 | No negative-parallax draw occurs in any of the 456 root failures parent-wide |
| `p` at or below 0.95 under the notebook's draws (MC noise) | 1 | — |
| Passes the notebook | 114 | Plus 1 source (`6092954989675820416`) that our path had at p = 0.9487 |
| NaN inputs (`bit_index` 8179) | 0 | Both paths reject the same 1839 sources, so ignoring `bit_index` changes nothing for `Orbital` |
| Valid-draw-only denominator | 0 | Our `p_m2_above` already used all draws. Only 2 root-failed sources would pass on a valid-draw denominator |

**Boundary sources.** The notebook is unseeded. At N = 10⁴ the binomial σ of p near 0.95 is ≈ 0.0022.

- The one extra final source, `6424213726885519744` (G = 15.15, FLAME-less, uniform M1), has
  **p = 0.9545**, about 2σ above threshold. The published `1749013354127453696` has **p = 0.9549**.
  Either could legitimately flip. The notebook run evidently kept the second and dropped the first.
- Other pass-1 sources with 0.95 < p ≤ 0.96: `1827768657631141248` 0.9564, `1982739530143208192`
  0.9528, `2010172929375186304` 0.9546, `2174796689377604480` 0.9560, `2231278532962626176` 0.9558,
  `3076065988396801280` 0.9557, `4509011462269157888` 0.9554, `5843399973710547584` 0.9550,
  `5866015312999519744` 0.9508, `6047116732165574016` 0.9576, `6092954989675820416` 0.9542. All of
  them fail a later cut, so they move the pass-1 count but not the final N.
- Just below threshold: `1987234539838699904` 0.9500 exactly (fails the strict `>`),
  `4783369128404125952` 0.9462, `276061123196785920` 0.9449, `5531157844139797632` 0.9447.
- A scratch run of the same pass 1 with an independent seed gave 116 against 115 here. The residual
  115 vs 106 is therefore not MC noise alone. It stays open and is not tuned.

**Q9 / `andrews2022_import`.** The survivors with `G < 15` are exactly the 16 published sources; the
extra source is at G = 15.15. The `andrews2022_import` subsample chain is `in_sample('andrews2022')`
AND `phot_g_mean_mag < 15`, so it evaluates to the same 16 by construction. Measured caveat: this was
computed from that chain's definition on the `andrews2022` survivor set, not from a full El-Badry 2026
`evaluate_all`. The full run was skipped because disk was critically low (≈ 4.6 GiB free) and a cold
E(B-V) run peaks near 12.5 GiB. The full `evaluate_all` should be re-run by #49.

**Ambiguity, flagged for Ryan.** The notebook's Apsis `logg` comes from VizieR `massive_apsis.vot`,
which has the columns `Teff, logg, [Fe/H], Dist, Mass-Flame`, i.e. I/355/paramp. It is taken to be
**`logg_gspphot`**, since VizieR paramp's `logg` is the GSP-Phot value. This is a config switch:
`reproduction_procedure.pass2.giant_logg_column`.

`mode_divergence` is not re-measured here. In the registry, `andrews2022` is reproduction mode, now
the ATF chain, while `andrews2022_modified` is forward_model mode, still the schema_version-2 chain.
The pair therefore no longer differs by BH1 alone. In reproduction mode the two differ by exactly BH1. Filed as #306.

| Check | Target | Previous (`5725e54`) | **`main` @ `c905575`** | Gate |
|-------|--------|----------------------|------------------------|------|
| Parent `Orbital` N | 134598 | 134598 | **134598** | **OK** — exact match holds |
| After `m2_probability` | 106 | 352 | **352** | FAIL — MC vs paper attrition; do not retune 0.95 / 1.4 |
| Final reproduction N | 24 | 33 | **33** | FAIL |
| `andrews2022_modified` N | 25 | 34 | **34** | Tracks the published number + Gaia BH1 |
| `mode_divergence` | only `4373465352415301632` in modified | matches | **matches** (only-modified `[4373465352415301632]`, only-published `[]`) | **OK** |
| Q9: Andrews survivors with `G < 15` | 16 | 19 | **19** | FAIL — follows the Andrews over-count, not an independent defect |

**Q9 re-confirmed at `main` @ `63c6f53`** (issue #198), against the same
`20260826T234425Z_3d3f740b080c+enrich+mc10000/selection_parent_rows.h5` cache: `andrews2022`
(frozen, N=33) restricted to `G < 15` gives **19**, unchanged from the `c905575` baseline — no drift
across the intervening commits (#191, #192, docs-only #188/#190). `andrews2022_modified` (N=34)
restricted to `G < 15` gives **20**: the one extra source is Gaia BH1
(`4373465352415301632`, G=13.77), present only in the modified variant per `mode_divergence`
above. The full source-ID sets for both are recorded in #198. Per CONTINUATION_PLAN §8.7,
El-Badry 2026's `andrews2022_import` resolves against the **frozen** `andrews2022`, so its target
of 16 is compared against 19, not 20 — the mismatch is inherited entirely from the Andrews
over-count (33 vs 24), not an independent Q9 defect.

Attrition, reproducing the documented shape exactly:

```
134598 → 352 (m2_probability, 1839 N/A) → 278 (goodness_of_fit) → 73 (m2_snr)
       → 50 (giant_reject_logg) → 34 (giant_reject_cmd) → 33 (explicit_exclusions)
```

**Re-confirmed unchanged at `main` @ `f9603a3`** (issue #233, 2026-09-25): parent `Orbital` N =
**134598**, `andrews2022` survivors = **33**, against the same
`20260826T234425Z_3d3f740b080c+enrich+mc10000/selection_parent_rows.h5` cache. No landed change
between `c905575` and `f9603a3` touches the Andrews cut chain, so the rest of the waterfall above is
carried forward unmeasured-but-unchanged.

#### 3.1.3 #280 localization — the 63-vs-24 over-count comes from moving FLAME M1 into the selection step, not from M1 scatter

**Provenance.** Measured 2026-09-27 on `main` @ `2470fe5`, in an isolated worktree
(`../dark-hunter_pop-worktrees/andrews-overcount-280`, `PYTHONPATH` pointed at its `src/`, `data/`
symlinked to the primary checkout's). The input was the §3.1.2 `+enrich+mc10000` cache, read-only.
FLAME masses came from `data/dr3/gaia_snapshots/flame_enrichment/query.ecsv`, and the NSS solutions
from `nss_enrichment/query.ecsv`. No cache was rebuilt, no production MC was run, and no config or
frozen threshold was touched. The paper text below is from arXiv:2207.00680 (Andrews et al. 2022),
§2 and the Table 1 footnote.

**Method: a paired MC over the whole usable parent, not a subset.** For each of the 132,759
Orbital sources that have a usable ensemble, the *same* 10⁴ full-covariance astrometric draws were
propagated under several M1 inputs. Each source used the same per-source seed as
`scripts/attach_mc_to_selection_cache.py`. Only the MC columns were substituted: `p_m2_above`,
`m2_msun` and `sigma_m2_msun`/`m2_msun_error`. The frozen `andrews2022` chain was then re-run
unchanged through `SampleSelectionRegistry(...).selection("andrews2022").evaluate`. The M1 inputs:

- **A**: fixed M1 = 1.0 M☉ (the pre-#257 convention).
- **B**: point M1 with no scatter. FLAME sources use their FLAME mass; fallback sources use 0.815 M☉,
  the midpoint of `Uniform(0.63, 1.0)`.
- **Bh**: as B, but fallback sources use 1.0 M☉.
- **C**: the production per-draw M1 from `_resolve_primary_mass_draws`.

A scratch-only vectorized Newton replaced `physics_utils.invert_astrometric_companion_mass`, which
calls `np.roots` once per draw. For F = 0 the cubic has a unique positive root. Against the
production solver, the maximum relative difference over 2×10⁴ random inputs was 5×10⁻¹⁶. This took
the run from about 7 h to 5 min. **Validation:**

- Variant C reproduces the cache exactly. The maximum `|p_C − p_cache|` is **0.0**, the maximum
  `|m2_C − m2_cache|` is 4×10⁻⁹, and the waterfall is identical (1061/63).
- Variant A reproduces the `c905575` fixed-M1 baseline exactly: 352 → 278 → 73 → 50 → 34 → **33**.

"pub24" below is the set of Andrews et al. (2022) Table 1 source IDs, taken from the `A22` rows of
`config/selections/external/elbadry2024_table3.yaml`. That set was cross-checked by eye against the
paper's Table 1: 23 of 24 matched directly, and the 24th differed only by a digit transposition in
the reading.

| M1 variant | m2_probability → GoF → m2_snr → logg → CMD → final | Final: FLAME / fallback | pub24 retained |
|---|---|---|---|
| **A** fixed 1.0 | 352 → 278 → 73 → 50 → 34 → **33** | 14 / 19 | **24 / 24** |
| **B** point M1, no scatter | 1092 → 701 → 424 → 101 → 72 → **71** | 58 / 13 | 20 / 24 |
| **Bh** point M1, fallback at 1.0 | 1120 → 719 → 442 → 118 → 78 → **77** | 58 / 19 | 22 / 24 |
| **C** production per-draw (= cache) | 1061 → 681 → 408 → 90 → 64 → **63** | 52 / 11 | 18 / 24 |
| FLAME point, fallback per-draw | 1087 → 697 → 420 → 97 → 70 → 69 | 58 / 11 | 18 / 24 |
| FLAME per-draw, fallback fixed 1.0 | 1094 → 703 → 430 → 111 → 72 → 71 | 52 / 19 | 22 / 24 |
| FLAME fixed 1.0, fallback per-draw | 319 → 256 → 51 → 29 → 26 → 25 | 14 / 11 | 20 / 24 |

**1. The widening hypothesis (§3.1.2) is refuted.** Compare the paired B and C runs, which use
identical astrometric draws. **No source** crosses `P(M2 > 1.4) ≥ 0.95` because of M1 scatter.
Scatter moves 31 sources the *other* way, from passing under B to failing under C (26 FLAME,
5 fallback). M1 scatter therefore slightly *reduces* the count, from 1092 to 1061 and from 71 to 63
final.

**2. The over-count is a mean shift toward massive FLAME primaries.** Going from A to C,
**747 sources gain** `p ≥ 0.95` and 38 lose it:

- **All 747 gainers have FLAME M1 > 1.0 M☉.** Their FLAME masses have median **2.80 M☉**, with a
  5–95% range of 1.66–4.70 M☉.
- The 38 losers are 5 FLAME sources with M1 < 1 (0.61–0.88 M☉) and 33 fallback sources.

At fixed m_f, M2 rises with M1, so a 2–3 M☉ primary clears 1.4 M☉ with a modest mass function.
Of the 52 FLAME final survivors, **50 have FLAME M1 > 1.0**. Their FLAME masses run from 0.91 to
3.29 M☉, median about 1.8.

**3. FLAME vs fallback split (production C).**

| After cut | N | FLAME | fallback |
|---|---|---|---|
| m2_probability | 1061 | 963 | 98 |
| goodness_of_fit | 681 | 602 | 79 |
| m2_snr | 408 | 395 | 13 |
| giant_reject_logg | 90 | 77 | 13 |
| giant_reject_cmd | 64 | 53 | 11 |
| explicit_exclusions (final) | 63 | 52 | 11 |

For comparison, 55.5% of the usable parent has FLAME. The **fallback range does not drive
survivors**. Its upper edge is 1.0 M☉, so relative to the old baseline it can only *lower* M2:

- Fallback sources gain nothing from A to C and lose 33.
- Putting the fallback sources back at a fixed 1.0 M☉ (the "FLAME per-draw, fallback fixed 1.0" row)
  moves the final count only from 63 to 71.

Of the 11 fallback survivors, 6 pass `giant_reject_logg` only because they have no Apsis logg
(`logg_apsis is None`). All 52 FLAME survivors have a logg, because FLAME needs GSP-Phot.

**4. Paper text vs implementation. The main finding.** Andrews et al. (2022) §2 states that the
106-candidate probability cut was made **assuming the luminous star is 1 M☉**, with a note that the
assumption is improved later. The FLAME-or-uniform rule appears only in the **Table 1 footnote**, as
the M1 used for the *reported* M2 of the final 24:

- Apsis or UCO Lick spectroscopic masses get a 0.1 M☉ uncertainty.
- Otherwise M1 is taken to lie between 0.63 and 1 M☉.

#230's "ATF" rule therefore matches the paper's post-selection mass refinement, not its selection
M1. #257 applied it to the selection cut itself. Consistent with that:

- Fixed M1 = 1.0 retains **all 24** published sources at every cut.
- The production FLAME/uniform method loses 6 of them:
  - `5681911574178198400` (FLAME 0.73): p 0.997 → 0.296
  - `747174436620510976` (FLAME 0.85): 1.000 → 0.638
  - `4271998639836225920` (fallback): 1.000 → 0.663
  - `1581117310088807552` (fallback): 1.000 → 0.903
  - `1058875159778407808` (fallback): 1.000 → 0.921
  - `5847919241396757888` (fallback): 0.972 → 0.901
- It also admits 42 sources the paper never selected.

The remaining line-by-line divergences are listed but not implemented:

| Item | #230 text | Paper (§2, eq. 6, Table 1) | Implementation | Divergence |
|---|---|---|---|---|
| M1 for the probability cut | FLAME ±0.1, else U(0.63, 1.0) | **1 M☉** for the 106-cut. FLAME/Lick/U(0.63, 1) only for the final Table 1 masses | `flame_or_uniform_draw` at the selection step (schema_version 2, #257) | **Major.** This alone accounts for 33 → 63 and 352 → 1061. **Needs Ryan's decision** (below) |
| Probability computation | not stated | 10⁴ MC draws of the full 12×12 covariance; P(M2 > 1.4) ≥ 95% | same | None. A point-estimate reading (MC-mean M2 > 1.4, no probability cut) is ruled out: 4760 → 248 final |
| "M2 determined within 3σ" | stated, order unspecified | `M2/σ_M2 > 3`, applied **after** the probability and GoF cuts | `m2_snr` (`m2/σ > 3`), after `goodness_of_fit` | None. Order does not affect the final N of an AND chain. The alternative reading `M2 − 3σ > 1.4` is ruled out: 255 → 8 final, 2/24 retained |
| Giant logg | "require log g > 3.6" | remove log g < 3.6 **only if Apsis measured log g** | `logg_apsis is None or logg_apsis >= 3.6` | None against the paper. #230's wording is a paraphrase. The source table (GSP-Phot only, no MSC) stays open per §3.1.1 |
| CMD line | slope 11/3.5, intercept 9 − 3·slope | printed as `G > 3.14(BP−RP) − 0.43`, no extinction | `3.14`, `−0.43`, `extinction_corrected: false` | None against the published equation. The frozen values are **exactly** the paper's printed coefficients, which bears on #255 |
| GoF, parent, exclusion | not mentioned | F2 < 5; 134,598 astrometric-only binaries; exclude `4373465352415301632` | same | None |

**What stays unexplained at fixed M1 = 1.0**, which is paper-faithful for the selection:

- `m2_probability` gives 352 against the published 106.
- The final 33 is a strict superset of the published 24. The 9 extras are `534854721213752960`,
  `875505688604417920`, `1100973226624010240`, `2092824730262458496`, `4390946762664086784`,
  `5449026524561550464`, `5802723506661520512`, `5810797564142964224` and `6424213726885519744`.
  Three of them pass the logg cut only as `logg_apsis is None`.

This residual gap is the next thing to localize, and it is independent of M1.

**Decision needed from Ryan (escalated, not made).** Ryan's #254 ruling put the FLAME/uniform M1
into the `m2_probability` step (`andrews2022.yaml` schema_version 2). The paper text says that
step used 1 M☉, and 1 M☉ retains all 24 published sources. Two choices:

1. Revert the *selection* M1 to fixed 1.0 in a schema_version 3 bump. The FLAME/uniform rule would
   then apply only to the post-selection refined M2, if at all.
2. Keep the ruling and accept that the reproduction cannot match.

No frozen file was edited here. Separately, the CMD evidence above suggests #255 can close with
"frozen values match the paper as printed". That call is Ryan's too.

#### 3.1.2 #257 landed — FLAME/uniform-draw M1 implemented, real measurement (supersedes reverted PR #266)

Ryan Foley's ruling on #254 (2026-09-25/26): "Yes, update the selection to the Apsis / uniform M1
mass. Adjust the query, etc." Landed across five PRs, all merged to `main`: **#260** (ADQL join +
`PrimaryMassSpec.flame_or_uniform_draw` + `MassFunctionDraws` per-draw `m1_msun` +
`andrews2022.yaml` `schema_version` 1→2), **#261** (supplemental
`fetch_flame_enrichment.py`/`merge_flame_enrichment_into_cache.py`, leaving the frozen
`nss_enrichment` job untouched), **#263** (client-side network timeout for the fetch), **#264**
(sync-mode fallback for a stalled async fetch), and **#265** (fixed a numpy-float-to-JSON bug in the
merge script).

**A first rebuild+measurement attempt (2026-09-26/27, `main` @ `dea93c3`) was invalid and reverted.**
`attach_mc_to_selection_cache.py` computed job-enumeration coverage and the printed FLAME-coverage
line from the FLAME-merged cache correctly, but built each individual MC job's payload from the
separate, older raw `nss_enrichment` ECSV — which predates #257 and has no `mass_flame` column at
all — and never folded the FLAME value back in before the payload reached `_resolve_primary_mass_draws`.
Every source's MC therefore silently fell back to `Uniform(0.63, 1.0)` regardless of whether it had a
real FLAME mass, for the full ~6h22m rebuild. This was caught (#267), fixed on `main` (PR #269,
`9f70a53`), and the invalid write-up reverted (PR #268/#266, merge `d2ad5ab`) — **the numbers below
are the first ones actually measured under the fix.**

**Real-data sanity check (required before trusting anything else in this section).** Before
re-measuring the reproduction gates, several real `source_id`s with distinct FLAME masses were
spot-checked between the `+enrich+flame` cache and the freshly rebuilt `+enrich+mc10000` cache
(random sample of 15 plus a min/q1/median/q3/max-spread sample of 5, covering FLAME masses from
0.50 to 6.99 M☉):

| `source_id` | FLAME `mass_flame` | rebuilt `m1_msun_mc_mean` | rebuilt `m1_msun_mc_sigma` |
|---|---|---|---|
| 5456158781612689536 | 1.5724 | 1.5742 | 0.1011 |
| 4040121150888032640 | 1.0015 | 1.0017 | 0.0994 |
| 5884900215454611200 | 2.8766 | 2.8769 | 0.0989 |
| 1825221226643060864 (min) | 0.5044 | 0.5040 | 0.1000 |
| 4561479469943220352 (max) | 6.9917 | 6.9923 | 0.0999 |

Full population check across all 73,644 Orbital rows with a real `mass_flame` and a produced MC
ensemble: `m1_msun_mc_mean` takes **15,359 distinct values** (rounded to 4 dp) ranging 0.504–6.992,
tracking each source's own FLAME mass to within ~0.002 M☉ — not a single shared constant. `m1_msun_mc_sigma`
has mean **0.10000**, std **0.00071**, range [0.0974, 0.1031] — consistent with the fixed
`flame_fixed_error_msun: 0.1` FLAME-branch error, not the ~0.1068 signature of the uniform-fallback
branch. **Sanity check passes**: the rebuild is genuinely using per-source FLAME data, not a repeat of
the #267 no-op.

**Full re-fetch + rebuild** (the FLAME fetch and merge were already done and independently verified
before this ticket started — only the MC rebuild was redone here):
- FLAME fetch (already staged, not redone): 443,205 NSS rows, 241,196 (54.4%) carry a real Gaia Apsis
  FLAME mass, in `…+enrich+flame/selection_parent_rows.h5` (733 MB).
- `attach_mc_to_selection_cache.py --workers 8` against `+enrich+flame` (10,000 draws, 134,598
  Orbital jobs), run in the primary checkout on `main` @ `9f70a53`: **20355.3 s (~5h39m)**
  wall-clock, `132759/134598` jobs produced a usable MC ensemble (same unusable-covariance remainder
  as every previous build), writing `…+enrich+mc10000/selection_parent_rows.h5` (800 MB, `mtime`
  2026-09-27 02:18).

| Check | Target | `c905575` (fixed `M1=1.0`) | `dea93c3` (invalid, #267 no-op) | **`9f70a53` (real FLAME/uniform-draw)** | Gate |
|-------|--------|----------------------------|-----------------------------------|-------------------------------------------|------|
| Parent `Orbital` N | 134598 | 134598 | 134598 | **134598** | **OK** — unaffected by the M1 change |
| After `m2_probability` | 106 | 352 | 273 (invalid) | **1061** | FAIL — far from target, and from both prior values |
| Final reproduction N | 24 | 33 | 14 (invalid) | **63** | FAIL |
| `andrews2022_modified` N | 25 | 34 | 15 (invalid) | **64** | FAIL |
| `mode_divergence` | only `4373465352415301632` in modified | matches | not re-checked (invalid run) | **matches** (only-modified `[4373465352415301632]`, only-published `[]`) | **OK** |
| Q9: Andrews (frozen) survivors with `G < 15` | 16 | 19 | 6 (invalid) | **55** | FAIL |

Attrition (real measurement):

```
134598 → 1061 (m2_probability, 1839 N/A) → 681 (goodness_of_fit) → 408 (m2_snr)
       → 90 (giant_reject_logg) → 64 (giant_reject_cmd) → 63 (explicit_exclusions)
```

`andrews2022_modified`: N=64 (frozen 63 + Gaia BH1), G<15 restricted count = 56.

**Read honestly, not spun.** This is worse than both previous baselines, not better, and moved in the
*opposite* direction from what the (invalid) #267-era run reported: that run's `m2_probability`
count fell (352→273); this real one **rises sharply** (352→1061), a ~3× increase, and every
downstream count now **over**-shoots its target by a wider margin than the original fixed-`M1=1.0`
baseline did. A plausible mechanism (not verified here, and not something this ticket's scope covers
investigating further): the fixed-`M1=1.0` baseline had **zero** M1 uncertainty, while every source
now carries either a FLAME-derived M1 with a fixed ±0.1 M☉ error or a much wider
`Uniform(0.63, 1.0)` draw (fallback for the 45.6% without a real FLAME value) — since
`m2_probability` is a right-tail probability `P(M2 > 1.4 M☉) ≥ 0.95`, adding substantial M1 variance
can inflate that tail probability for many marginal systems even where the mean M2 estimate is lower
than before. This is offered as a hypothesis for whoever picks up #254/#233 next, not as an
explanation that resolves the gate. **Tested in §3.1.3 (#280): refuted.** M1 scatter moves no source
across the threshold and removes 31. The rise comes from the FLAME mean shift toward M1 > 1 M☉.

**Escalating rather than retuning, per this ticket's explicit instruction and `CLAUDE.md`'s standing
rule**: `m2_probability_min` (0.95) and `m2_threshold_msun` (1.4) in `config/selections/andrews2022.yaml`
were **not** touched, and must not be retuned to chase 106. The gate is not closed. Whoever works
#254/#233 next should compare per-source `p_m2_above` distributions between the fixed-`M1=1.0`
baseline's implied shape and this FLAME/uniform-draw run's actual `p_m2_above` histogram (measured
here: mean 0.032, median 0.0, 95th percentile 0.180, 99th percentile 0.895 across 132,759 sources
with a usable MC ensemble) to localize whether the swing is expected physics or a remaining
implementation divergence from #230's exact procedure.

**Out of scope for this ticket, flagged rather than silently updated**: El-Badry 2026's
`andrews2022_import` row in §3.3 depends on `andrews2022`'s frozen survivor set, which changed from
N=33 to **N=63** in this rebuild — a re-measurement is needed, but El-Badry 2026 evaluation costs
~22 min and ~6.8 GiB peak (§1) and touches primary-mass handling this ticket's "Do not" section
explicitly places out of bounds. **Filed as a follow-up rather than measured here**: #270.
**#270 has now measured it**: `andrews2022_import` = **55**, the same source-ID set as the Q9
`G < 15` count above (§3.3.0).

#### 3.1.1 #230/#233 methodology reconciliation — findings (no code change landed)

Issue #230 records a colleague's confirmed Andrews et al. (2022) methodology ("ATF", independently
reproduces N=24). Compared field-by-field against `config/selections/andrews2022.yaml` and the code
that evaluates it (`sample_selection.py`, `mc_mass_function.py`,
`scripts/attach_mc_to_selection_cache.py` — `elbadry2026_m2_sigma.py` is not on the Andrews path):

| #230 field | Confirmed methodology | Current implementation | Divergence |
|---|---|---|---|
| M1 | Gaia Apsis **Flame** value with **fixed ±0.1 M☉ error**; else **draw uniform(0.63, 1.0) M☉** | `primary_mass.method: fixed`, `value_msun: 1.0` for **every** row, **zero** M1 uncertainty propagated in the MC (`mc_mass_function.MassFunctionDraws.m1_msun` is a bare `float`, not a per-draw array) | **Major** — see below |
| Giant exclusion | Apsis `log g > 3.6` | `giant_reject_logg`: `logg_apsis is None or logg_apsis >= 3.6`, `logg_apsis` sourced from `candidate.extras["logg_gspphot"]` only (`sample_selection.py:1554-1555`) — never falls back to `logg_msc1`, unlike `mass_derivation.py`'s stated MSC-preferred/gspphot-fallback order | Minor — cut logic and threshold match; source table choice (gspphot only, no MSC) is a narrower reading of "Apsis" than the rest of the pipeline uses elsewhere. Not obviously wrong, not obviously right — flagging, not fixing |
| CMD cut | slope = `11/3.5` = 3.142857…, intercept = `9 − 3×slope` = −0.428571… | `cmd_slope: 3.14`, `cmd_intercept: -0.43` (3-decimal rounding) | **Minor but real** — frozen-file value, not bit-exact to #230's formula |
| M2 within 3σ | Required as its own filter | `m2_snr` cut: `m2_msun / m2_msun_error > 3.0`, `applies_to` includes `reproduction` | **Matches** — already correctly implemented, no divergence found |

**Root-cause assessment for the 352-vs-106 gap at the `m2_probability` step:** the M1 divergence is
upstream of every other cut and is the only one large enough to plausibly explain a 3.3x overcount at
the very first filter. `scripts/attach_mc_to_selection_cache.py` reads
`andrews2022.yaml`'s `primary_mass.value_msun` (1.0) and MCs every one of the 134598 parent rows at
that single fixed mass with no M1 uncertainty term at all — so `p_m2_above` (`P(M2 > 1.4 M☉)`) for
every system reflects only astrometric/orbital covariance, never M1 spread or a physically appropriate
M1 per star. Using the actual Flame M1 (which is `<1.0 M☉` for most FGK dwarfs in this sample) plus a
real M1 uncertainty term would shift both the M2 mean and its spread for every system, plausibly
tightening `P(M2 > 1.4) ≥ 0.95` substantially.

**This is escalated, not fixed, for three independent reasons:**

1. **It contradicts a standing, tested, documented invariant.** `CLAUDE.md`'s "Column ownership is
   strict" gotcha and this document's §4 both state "Andrews owns `p_m2_above` / `m2_msun` /
   `m2_msun_error` at fixed `M1 = 1.0`" as an intentional design, not a placeholder — and
   `tests/test_andrews2022_selection.py:35-37,115-116` asserts `primary_mass.method == "fixed"` /
   `value_msun == 1.0` as the expected, tested behavior. Whether #230's Flame/uniform-draw rule
   **replaces** this fixed-M1=1.0 convention, or describes a *different* stage of Andrews' actual
   analysis (the frozen YAML's own comment reads "Andrews et al. (2022) first-cut assumption; refined
   later in their analysis" — implying the authors who wrote it believed M1=1.0 was deliberately a
   first pass, with refinement happening only via the giant/CMD cuts, not an M1 re-draw) is an
   ambiguous reading of the paper's actual method, exactly the kind of judgment call #233 says to flag
   rather than guess.
2. **The M1 data does not exist in the pipeline yet.** No Gaia Apsis Flame mass column
   (`mass_flame` et al.) is queried anywhere — `data_acquisition._AP_PARAM_STEMS` pulls
   `teff_msc1/logg_msc1/mh_msc` and `teff_gspphot/logg_gspphot/mh_gspphot` only. Implementing #230's
   M1 rule needs a new ADQL column, likely a re-run of the NSS enrichment job or a fresh Gaia archive
   query (network, out of this session's scope), a new `PrimaryMassSpec` method (fixed-Flame-with-
   fallback-draw), and restructuring `mc_mass_function.MassFunctionDraws` from a scalar `m1_msun` to a
   per-draw array so a per-draw M1 sample (Gaussian or uniform) actually propagates through the
   ensemble — not a same-file code-bug fix.
3. **`andrews2022.yaml` is frozen.** Per `CLAUDE.md`'s hard rule, any threshold edit — including
   adding the exact CMD-line fraction or a Flame-mass primary_mass method — needs a `schema_version`
   bump, a provenance note, and human escalation before landing, never a same-PR retune.

**Escalation issues filed:** #254 (M1 methodology), #255 (CMD line precision) — see issue #233 for
the full write-up; both block closing this gate.

### 3.2 El-Badry et al. (2024)

| Check | Target | Previous (`5725e54`) | **`main` @ `c905575`** | Gate |
|-------|--------|----------------------|------------------------|------|
| Catalog union (`table3` ∩ `G < 15`) | 48 | 48 | **48** | **OK** — exact match holds |
| Published COC N (inclusion / post-follow-up) | 21 | 21 | *not re-measured* | OK for the catalog-level wave — see note |

`elbadry2024` surviving N = 48; branch `astrometric` = 48; subsample `elbadry2024_table3` = 48.

**The COC N of 21 is the one §3 number not re-measured here**, and is deliberately not restated as though it had been. It
is an inclusion / post-follow-up report rather than the catalog-union count this eval produces, so it
needs a different measurement path. It stays recorded as green from the earlier measurement, and should
be re-measured by whoever next works the El-Badry 2024 path. The catalog path does **not** need Andrews
`p_m2`; extinction work should not break the 48.

### 3.3 El-Badry et al. (2026)

**Re-measured with the real frozen extinction policy (#258)** — Green2019 (Bayestar19) for
`dec_deg > -28`, Lallement 2019 for `dec_deg <= -28`, `A_G/E(B-V)=2.66`, `E(BP-RP)/E(B-V)=1.33`,
now actually applied (`dust_maps.py` + `elbadry2026_selection.deredden_elbadry2026_rows`; §3.3.1).
Provenance: branch `feat/elbadry2026-extinction-258` @ `1058e29` (= `main` @ `a9397ba` + the #258
commits), isolated worktree, `PYTHONPATH` → worktree `src/`, cache `…+enrich+mc10000` (443,211 rows,
the FLAME/uniform-draw rebuild §3.3.0 used), full `SampleSelectionRegistry.evaluate_all`. The
"undereddened" column is the **same code and cache** with `E(B-V)=0` forced on every row — it
reproduces the `c905575`/`a9397ba` (#270) numbers exactly, so every change in the last column is the
extinction correction and nothing else.

| Check | Target | `main` @ `c905575` | Undereddened @ `1058e29` | **Real maps @ `1058e29` (#258)** | Gate |
|-------|--------|--------------------|--------------------------|----------------------------------|------|
| Published union (per-row `n_surviving`; **#274 per-type distinct count in brackets**, §3.3.4) | 227 | 1085 | 1119 (1067 distinct) [**1069**] | **1565** (1504 distinct) [**1507**] | FAIL — driven by `sub_chandrasekhar` |
| Astrometric branch union | 76 | 913 | 946 | **1356** | FAIL |
| Spectroscopic branch | 151 | 123 | 123 | **151** | **OK** — exact match |
| `primary_ns_bh` | 47 | 42 | 42 | **46** | FAIL by 1 (was 5) |
| `elbadry2023_table_e1` | 5 | 5 | 5 | **5** | **OK** — unchanged |
| `andrews2022_import` | 16 | 19 | 55 | **55** | FAIL — Andrews over-count (§3.3.0), insensitive to extinction as expected |
| `sub_chandrasekhar` | 22 | 861 | 861 | **1265** | FAIL — **worse**; not the extinction lever (§3.3.1). **After #284 (schema_version 2: `M̃2/M̃1 > 1`, `A > 0.65`, analytic σ): 49, containing 19 of the paper's 22 — §3.3.6.** Still FAIL; no defensible combination reaches the paper's set. **Schema_version 3 (#315, §3.3.7): the reproduction mode is the Shahaf et al. (2023b) Table 2 cross-match with Eq. 5 on Shahaf's `M2min`/`e_M2min`, giving 22, the exact paper IDs — OK.** The forward model stays at 49 (the v2 chain) |
| Spectro routes (MS min / high `f_m` / both) | 136 / 30 / 15 | 98 / 30 / 5 | 98 / 30 / 5 | **132 / 30 / 11** | FAIL on the split; the branch total is exact |
| Simon exclusion breakdown | 5 / 2 / 1 / 1 | 5 / 2 / 1 / 0 (+1 unclassified) | 5 / 2 / 1 / 0 (+1) | **5 / 2 / 1 / 0** (+1 unclassified), `in_sample` 11 | **OK: 5 / 2 / 1 / 1, 0 unclassified, `in_sample` 11** from the real `sample_selection` stage + `diagnostics` hydration (#285, `main` @ `7c40198` + #285 branch, uncut snapshot; §3.3.3). Artifacts without the `tilde_masses` group still fall back to 5 / 2 / 1 / 0 (+1) |

Extinction outcomes over the 349,599 El-Badry 2026 rows that were enriched (both branches):
`ok` 318,964 · `beyond_map_limit` 30,446 (Lallement cube boundary or past the last Bayestar node;
integral to the limit is used and counted) · `invalid_parallax` 189 (now `NotApplicable
("extinction_invalid_parallax")` on `main_sequence`/`M̃1`, not silently undereddened). No source
fell outside a map footprint (Bayestar covers all of `dec > -28`).

`primary_ns_bh` attrition (real maps):

```
168065 → 147560 (main_sequence) → 1052 (m2_floor) → 91 (m2_over_m1)
       → 82 (goodness_of_fit) → 59 (period) → 46 (g_mag)        [target 47]
```

Undereddened for comparison: `168065 → 137696 → 499 → 85 → 77 → 55 → 42`. Versus the undereddened
set, 7 sources join (`465093354131112960, 1748901959855337472, 2010172929375186304,
2278534305772981504, 4099356347737522432, 5954343888087990656, 6037767138131854592`) and 3 leave
(`1998863902528338304, 2080945469200565248, 6742294434981881728`).

`sub_chandrasekhar` attrition (real maps):

```
168065 → 147560 (main_sequence) → 3043 (m2_range 1.05–1.40)
       → 1687 (m2_error σ ≤ 0.105; 1341 fail, 15 missing σ) → 1275 (period ≤ 900 d)
       → 1265 (G < 15)                                            [target 22]
```

Spectroscopic branch (real maps): `181534 → 133642 (k1_significance) → 151 (mass_route, 78839 N/A)`.

**Unit-conversion sensitivity (not the landed config).** El-Badry 2026 does not state how it converts
either map to `E(B-V)`. The landed config uses `E(B-V) = 0.884 × Bayestar19` and `E(B-V) = A0/3.1`,
which **the PI confirmed on 2026-09-27 (#295; §3.3.5)**. Re-running El-Badry 2026 alone with the
Bayestar factor set to 1.0 (raw Bayestar units as `E(B-V)`, the `mwdust` / #232-diagnostic convention),
same SHA and membership: `primary_ns_bh` 45, spectroscopic 150, routes 134 / 30 / 14,
`sub_chandrasekhar` 1313, astrometric union 1403, Simon unchanged. The conversion choice moves the
counts by a few; it does not change any conclusion.

The σ-cut diagnosis still stands: `sub_chandrasekhar`'s `M̃2` window is now **3043** wide before the
σ cut. Dereddening moves more sources onto the main sequence (137,696 → 147,560) and brightens
`MG,0`, which raises the Janssens `M̃1` and hence `M̃2`. Escalate the point-estimate `M̃2` / `a0` /
AMRF chain (#133 Q7 remainder, #237) — **not** the 0.105 threshold, and not the extinction policy.

Peak RSS for the full real-map `evaluate_all`: **11.6 GB** (`/usr/bin/time -l`, 2185 s). That is above
the ~8.7–8.8 GiB end-to-end figure in §5.6 of `EXECUTION_PLAN.md`: the two maps add about 3 GB
(Bayestar19 `best_fit` ≈ 2 GB, the Lallement cube ≈ 0.9 GB) on top of the row set. The cost is only
paid on a cache miss; once `data/dust_maps/ebv_cache/elbadry2026_<fingerprint>.h5` exists, no map is
loaded.

### 3.3.0 `andrews2022_import` re-measurement after the Andrews FLAME rebuild (#270)

**Scope: this subsection records only what #270 measured.** El-Badry 2026's extinction fix (#258) is
in progress and will move other §3.3 rows again (`primary_ns_bh`, the spectroscopic branch and
routes, `sub_chandrasekhar`, and the unions). The table's other rows are left at their `c905575`
values on purpose, for #258 to update; the union shifts caused by this membership change are
recorded below instead of in the table.

Provenance: `main` @ `a9397ba` (includes #269's FLAME-payload fix and PR #271's Andrews
re-measurement), isolated worktree, `PYTHONPATH` pointed at the worktree's `src/` (verified
`darkhunter_pop.__file__`), cache `…+enrich+mc10000/selection_parent_rows.h5` (823,999,664 bytes,
443,211 rows, `mtime` 2026-09-27 02:18:53). Read-only: no code, config, or cache changed. The
evaluation followed the §1 pattern: first `andrews2022`, then `elbadry2026` with
`membership={'andrews2022': frozenset(andrews.surviving_source_ids)}`. El-Badry 2026 took
1648 s, with **7.38 GiB** peak RSS (`/usr/bin/time -l`), a little above §1's 6.8 GiB.

- `andrews2022` N = **63** (matches §3.1.2). `andrews2022_import` attrition:
  `168065 → 63 (andrews2022_membership) → 55 (g_mag < 15)`. Target is **16**, so this row still
  **FAILS**, and by more than before (19 → 55). This is the honest number. No threshold was touched.
- The 55 IDs are **identical** to §3.1.2's Q9 set (`andrews2022` survivors with `G < 15`). The
  binding is therefore doing exactly what it should, and the whole overcount comes from Andrews
  (#254/#233), not from El-Badry 2026's import logic.
- Overlap with other subsamples: 13 of the 55 are also in `primary_ns_bh` and 2 in
  `sub_chandrasekhar`. None are in `elbadry2023_table_e1`.
- Source IDs (55):
  `181517829171152128, 421246788921590400, 430045252773901440, 809741149368202752,
  826299038567583744, 987751532149740672, 1002740757557931264, 1144019690966028928,
  1350295047363872512, 1695294922548180224, 1749013354127453696, 1854241667792418304,
  1871419337958702720, 2031373192892592000, 2080084929550950272, 2083618416332991616,
  2180088295239113600, 2195470016228007424, 2397135910639986304, 3113779241530432384,
  3649963989549165440, 4248199573219059584, 4287259238460324224, 5254694888140645504,
  5254757074929505792, 5255738530815159168, 5304995582275067136, 5307607682620027776,
  5311489130468652032, 5312506140052764544, 5322756753808947840, 5337671422919801344,
  5337911356971992832, 5340413055155981440, 5342871047750973312, 5351774995980250880,
  5355483511276428672, 5370081104356941184, 5405214250387742592, 5539729980386209536,
  5556237944880786816, 5580526947012630912, 5593444799901901696, 5602760132982139136,
  5617846158373940096, 5644387063402978304, 5703347859033266816, 5861845586994786432,
  5885622869493356288, 5932335341825006208, 5932802153199174912, 6001459821083925120,
  6052935107242950400, 6328149636482597888, 6593763230249162112`

**Side effects of the membership change on the other El-Badry 2026 counts:**

| Count | `c905575` (table) | `a9397ba` (#270) | Moved? |
|---|---:|---:|---|
| `primary_ns_bh` | 42 | 42 | no — attrition identical (`168065 → 137696 → 499 → 85 → 77 → 55 → 42`) |
| `elbadry2023_table_e1` | 5 | 5 | no |
| `sub_chandrasekhar` | 861 | 861 | no — attrition identical (`… → 1908 → 1161 (735 fail, 12 missing σ) → 868 → 861`) |
| Spectroscopic branch | 123 | 123 | no (`181534 → 133642 → 123`, 84090 N/A) |
| Spectro routes | 98 / 30 / 5 | 98 / 30 / 5 | no |
| Astrometric branch union | 913 | **946** | **yes, +33**. This is downstream of `andrews2022_import` (+36, minus overlap with the other subsamples). The 946 IDs are exactly the union of the four subsamples |
| Published union (`n_surviving`) | 1085 | **1119** | **yes, +34**. This is downstream of the astrometric union. See the caveat below |

Nothing moved that is independent of `andrews2022_import`. The cut chains are unaffected by the
membership input and by the rebuilt Andrews MC columns, as expected.

**Caveat, flagged but not fixed here:** the published-union figure is a count of **non-unique** IDs.
`surviving_source_ids` has 1119 entries but only **1067 distinct** source IDs. The astrometric and
spectroscopic branches overlap by 2, so 946 + 123 − 2 = 1067, and 52 IDs appear twice. The 1085 at
`c905575` presumably includes the same kind of duplication. This looks like the
cross-match / multi-solution duplicate-`source_id` question already tracked in #221/#237, not
something new from #270. It is noted so that nobody compares 1119 against the published 227 as if it
counted unique sources. **Resolved by #274 (§3.3.4):** all 52 are cross-type, and the counting rule
is now fixed.

### 3.3.4 Counting rule for published-N comparisons (#274)

**Ruling (verbatim, Ryan Foley, 2026-09-27):** "In each solution type, count distinct stars." Every
reproduction count compared against a published N is now the number of distinct `source_id`s
within each `nss_solution_type`, summed over types. It is computed from the rows that actually
passed their own branch chain (`sample_selection.distinct_star_count`,
`SampleEvaluationResult.*_by_solution_type`, `sample_diagnostics.compare_to_published`;
`docs/ARCHITECTURE.md` §4 "Multi-solution sources"). The per-row count and the count of distinct
IDs across all types are still reported, but only as information. This is reporting only: per-row
emission and the per-row likelihood (Q17 / #244) are unchanged.

**What the 52 duplicated IDs were (#270 membership, `a9397ba`).** Every one of the 52 is a
**cross-type** multi-solution star. None is a same-type duplicate and none is #221-style
cross-match fan-out. Row shapes in the parent cache: `Orbital`+`SB1` 49, `EclipsingBinary`+`SB1` 3.
By branch:

- **2** passed both branches: the `Orbital` row passed astrometric and the `SB1` row passed
  spectroscopic. These count once in each type.
- **39** passed astrometric only. Their `SB1` row failed the spectroscopic chain but is still
  emitted, because `_evaluate_branched` emits every row of a surviving star.
- **11** passed spectroscopic only. They carry an `Orbital` row (8) or an `EclipsingBinary` row (3)
  that is emitted the same way.

The 50 emitted rows that passed no chain do not count under the rule. The emission behavior itself
is tracked in **#299** and is not changed here.

For the real-map membership (`1058e29`, 1565 entries, 1504 distinct), 61 IDs are duplicated, again
all cross-type: 3 passed both branches, 46 are astrometric survivors with an emitted `SB1` row, and
12 are spectroscopic survivors with an emitted `Orbital` (9) or `EclipsingBinary` (3) row.

**Re-reported counts.** These are the membership sets recorded at the SHAs shown, recounted with the
landed rule (branch `feat/distinct-star-counts-274`, off `main` @ `c07897e`). Solution types come
from `…+enrich+mc10000/selection_parent_rows.h5` (443,211 rows). No selection was re-evaluated and
no threshold was touched.

| Count | Target | Per-row (old) | **Per-type distinct (#274)** | Distinct across types (info) | Per-type breakdown |
|---|---:|---:|---:|---:|---|
| El-Badry 2026 union, #270 (`a9397ba`, undereddened) | 227 | 1119 | **1069** = 946 + 123 | 1067 | `Orbital` 722, `AstroSpectroSB1` 224, `SB1` 122, `SB1C` 1 |
| El-Badry 2026 union, real maps (`1058e29`) | 227 | 1565 | **1507** = 1356 + 151 | 1504 | astrometric: `Orbital` 1068, `AstroSpectroSB1` 288; spectroscopic 151 |
| El-Badry 2026 astrometric branch (real maps) | 76 | 1356 | **1356** | 1356 | unchanged: no star has both `Orbital` and `AstroSpectroSB1` rows |
| El-Badry 2026 spectroscopic branch (real maps) | 151 | 151 | **151** | 151 | unchanged, still exact |
| El-Badry 2026 subsamples (real maps) | 47 / 5 / 16 / 22 | 46 / 5 / 55 / 1265 | **46 / 5 / 55 / 1265** | same | unchanged |
| Andrews 2022 (`andrews2022`) | 24 | 63 | **63** | 63 | `Orbital` only (Orbital-only parent) |
| Andrews 2022 modified | 25 | 64 | **64** | 64 | `Orbital` only |
| El-Badry 2024 catalog union | 48 | 48 | **48** | 48 | `Orbital` 42, `AstroSpectroSB1` 6 |

Same-type duplicates cannot move any of these numbers. The parent cache has only **6** same-type
`(source_id, type)` pairs (5 `SB1`, 1 `EclipsingBinary`), and none of them is in any sample's
membership. Only the El-Badry 2026 **union** moves. It is still a FAIL, driven by
`sub_chandrasekhar`, as before.

### 3.3.1 Extinction chain audit (#232, #258) — root cause confirmed, fix landed (#258)

**Update (#258): the frozen policy is now applied, with the real maps.** The PI authorized acquiring
the maps once disk space allowed. What was done:

- **Green2019 (north).** The Bayestar19 file was already on disk from the separate `dustmaps` package
  (`…/site-packages/dustmaps/data/bayestar/bayestar2019.h5`). Its md5 `ab815d2f…a722` equals the
  Harvard Dataverse checksum for `doi:10.7910/DVN/2EJ9TX/1CUGA1`, which is the exact URL
  `mwdust.Green19.download()` fetches. It is the same file, with the same HDF5 layout (`pixel_info`,
  `best_fit`, `samples`), so it was **symlinked**, not copied or re-downloaded, into both
  `data/dust_maps/green2019/bayestar2019.h5` (the pipeline's configured path) and
  `~/.mwdust/green19/bayestar2019.h5` (used only by the `slow` cross-check test against
  `mwdust.Green19`). No new dependency: `dust_maps.Bayestar2019Map` is a small vectorized reader that
  reuses `mwdust.util.healpix.ang2pix` and reproduces `mwdust.Green19` to `rtol 1e-5` inside the grid.
  It does not use `mwdust.Green19` directly because that class reads its path from the `DUST_DIR`
  environment variable at import time rather than from config, and it evaluates sources in a
  per-source Python loop.
- **Lallement2019 (south).** No maintained Python package ships a reader (`mwdust` has none, and
  `dustmaps` has Leike/Edenhofer/Chen but not Lallement 2019). The published cube *is* obtainable:
  VizieR `J/A+A/625/A135`, `map3D_GAIAdr2_feb2019.h5.gz` (772 MB compressed, 809 MB decompressed;
  the alternative `STILISM_cube.fits.gz` is 1.5 GB). It was downloaded to
  `data/dust_maps/lallement2019/` (md5 `e9262125…f213`, pinned in config; disk went from 12 GiB to
  14 GiB free over the session because of other cleanup — never below the 5 GiB floor). Layout:
  `stilism/cube_datas`, `(1201, 1201, 161)` float32 `dA0/ds` in mag pc⁻¹, 5 pc voxels, Sun at voxel
  centre `[600.5, 600.5, 80.5]` → ±3 kpc in X/Y and ±400 pc in Z. `dust_maps.Lallement2019Map`
  integrates `A0` along the sightline with trilinear interpolation. A source beyond the cube gets the
  integral to the boundary and is counted as `beyond_map_limit`. **No substitute map was used.**
  **Units (#279):** the CDS `STILISM_cube.fits` BINTABLE labels the density column `mag/kpc`, but
  the HDF5 attribute and the VizieR ReadMe say mag pc⁻¹ with identical values; **mag pc⁻¹ is
  correct** (vs `0.884 ×` Bayestar19, median ratio 0.86) — do not "fix" it by 1000×. The reader's
  axis order/frame is pinned against that FITS table's explicit X/Y/Z columns by
  `tests/test_dust_maps.py::test_real_lallement_frame_matches_stilism_xyz` (`slow`) and
  `::test_lallement_reader_frame_on_linear_density` (required gate).
- **Wiring.** `elbadry2026_selection.deredden_elbadry2026_rows` applies `mg_0 = abs_g_mag − 2.66·E(B−V)`
  and `bp_rp_0 = bp_rp − 1.33·E(B−V)`. Both coefficients are read from the frozen block. `E(B-V)` comes
  from the frozen hemisphere map. The call happens inside El-Badry 2026's enrichment, which is gated
  on the sample having both `extinction` **and** `main_sequence_cut`. El-Badry 2024 (whose file
  names Lallement 2022 at δ = −30°) and Andrews therefore never deredden; a test pins this.
  `candidate_to_selection_row` is unchanged. Its `mg_0 = abs_g_mag` alias is still what those samples
  see, and El-Badry 2026 overwrites it from the raw columns. Map locations and native→`E(B-V)`
  conversions are config (`sample_selection.dust_maps`), **not** the frozen file. The paper does not
  state them; **the PI settled them on 2026-09-27 (#295)**: `r_v: 3.1` (Gaia convention
  `A0 = 3.1 E(B-V)`, Babusiaux et al. 2018 §2; Lallement 2019 App. A also assumes `R = 3.1`), Lallement
  `A0 → E(B-V) = A0 / r_v` (derived from `r_v`, `native_quantity: a0`), Bayestar19 `0.884` (Green et al.
  2019 / Argonaut usage page Eq. 1, SF11 `R_V = 3.1`; the page's `E(r−z)` alternative is `0.996`). The
  `E(B-V) → A_G, E(BP−RP)` step is `gaia_band_extinction.law`: `paper_constant` (default, the frozen
  2.66 / 1.33) or `babusiaux2018` (the Gaia colour/`A0`-dependent law, coefficients in config). See §3.3.5.
- **Caching.** Native per-source integrals are cached at
  `data/dust_maps/ebv_cache/elbadry2026_<fingerprint>.h5`. The fingerprint covers the split, the map
  identities (md5), the `READER_VERSION` salt and — since #278 — the SHA-256 of `dust_maps.py`
  itself (same raw-bytes convention as the stage `source_hash`), so a reader change can no longer
  silently reuse stale integrals. It does not cover the unit conversions, so revising a conversion
  factor does not invalidate the cache. The cache is keyed per `(source_id, ra, dec, parallax)` and
  appended on miss. A full cold query of all 349,599 enriched rows (two batches, one per branch)
  took about 45 s including loading both maps, which is negligible next to the `σ_M̃2` MC. The slow
  part of the #232 Combined19 diagnostic was `mwdust`'s per-source loop, not the maps.

The rest of this subsection is the pre-fix record, kept for provenance.

**Root cause, confirmed by direct code inspection, not just symptom:** `config/selections/
elbadry2026.yaml`'s `extinction:` block (Green2019 north / Lallement2019 south, split at
`dec_deg > -28.0`, `a_g_over_e_bv=2.66`, `e_bp_rp_over_e_bv=1.33`) is parsed into
`config_schema.ExtinctionSpec` but has **zero consumers anywhere in `src/darkhunter_pop/`**
(`grep -rn "ExtinctionSpec" src/darkhunter_pop/*.py tests/` finds only the schema definition).
`sample_selection.candidate_to_selection_row` (~line 1578) sets `mg_0 = abs_g_mag` and
`bp_rp_0 = bp_rp` verbatim — the "if not already present" alias always fires for real candidates,
because nothing upstream ever computes a dereddened value. This is the exact mechanism §3.4 already
named for the Simon 2026 two-source swap: dead code, not a subtle numerical bug, confirmed at
`main` @ `f9603a3`.

**A real, paper-faithful fix is currently blocked, not just unimplemented.** The two maps the config
names are both unavailable in this environment: `mwdust` (this repo's only dust-map dependency) has
no Lallement 2019/2022 implementation at all (only Green15/17/19, Marshall06, Sale14, Combined15/19),
and Green2019's own data file (`bayestar2019.h5`, multi-GB) has never been downloaded here. **This
laptop's disk is at 100% capacity with only ~4.6 GiB free** (measured 2026-09-25; `~/.mwdust/
combined19` alone is 3.7 GiB) — the Green19 leg cannot even be attempted right now, separate from any
permission question, and is being addressed in a separate disk-cleanup effort. Escalated as #258
rather than guessed at.

**Diagnostic-only measurement (not a claimed fix), `main` @ `f9603a3`.** Using the already-downloaded
`mwdust.Combined19` (an all-sky, declination-unaware blend — not the frozen north/south split) purely
to bound the size of the extinction effect: recomputed `mg_0`/`bp_rp_0` for all 443,211 rows of the
real `+enrich+mc10000` cache and re-ran `SampleSelectionRegistry.evaluate_all`:

| Target | Published | Baseline (main, unextincted) | Combined19-dereddened (diagnostic) |
|---|---:|---:|---:|
| `primary_ns_bh` | 47 | 42 | **46** |
| `sub_chandrasekhar` | 22 | 861 | **1399** |

- `primary_ns_bh` closes ~80% of the gap (5-off → 1-off) — strong evidence the extinction chain is a
  major contributor to that target's mismatch.
- `sub_chandrasekhar` gets **worse**, not better (861 → 1399, further from 22) — a genuine, useful
  null-ish result. Dereddening the point estimate does not close this gap and is not the dominant
  lever for it; something else in the `M̃2` ∈ [1.05, 1.40] window (possibly still `a0`/AMRF,
  possibly the open multi-solution duplicate-row question in #237, since the cache used here predates
  #237's tag-and-keep resolution and was not rebuilt for this measurement) needs its own
  investigation. **Do not assume fixing extinction alone will fix `sub_chandrasekhar`.**
- `elbadry2023_table_e1` (5) and `andrews2022_import` (19) are unchanged in both runs, as expected —
  a sanity check that the diagnostic isolated the right mechanism and did not leak into unrelated
  subsamples.

**Disposition:** no code changed by #232. Substituting `Combined19` for the frozen `green2019`/
`lallement2019` split would be a real methodology deviation (and would make `sub_chandrasekhar`
worse), not a drop-in fix — #258 escalates the actual decision (acquire the real maps once disk
space allows vs. accept a documented, schema-bumped approximation vs. treat `sub_chandrasekhar` as
a separately-rooted problem) rather than guessing at it here.

### 3.3.2 Spectroscopic-branch binding audit (#234) — no code bug found, same root cause as #133/#258

**Update (#258):** with the real frozen maps applied (§3.3.1), the spectroscopic branch is **151,
an exact match**. Routes are 132 / 30 / 11 against 136 / 30 / 15: the total is right and the route
split is 4 off in each overlapping bucket. The `f_m > 3` route stays at 30, as it should. The
conclusion below stands: no spectroscopic-branch bug, the gap was extinction.

**Traced the full `mass_route`/main-sequence/`m2_min` chain at `main` @ `31ad1f2`, against the same
`+enrich+mc10000` cache, isolating just the spectroscopic branch's own cuts (bypassing the expensive
astrometric `σ_M̃2` MC entirely — `_evaluate_and_chain` on `spectroscopic`'s two cuts only).
Re-confirms the baseline exactly**: `181534 → 133642 (k1_significance) → 123 (mass_route, 84090 N/A)`,
routes `98 / 30 / 5`.

**Checked against `docs/CONTINUATION_PLAN.md` §8.4 line by line — implementation matches the documented
spec exactly, no divergence found:**

- `k1_significance`: cut expression `k1_significance > 10.0` where `k1_significance = K1 / σ_K1`
  (`elbadry2026_selection.enrich_elbadry2026_row`, `nss.semi_amplitude_primary` /
  `semi_amplitude_primary_error`) — this **is** the documented formula (`CONTINUATION_PLAN.md:1240`),
  not a shortcut around Gaia's own `nss.significance` column; ruled that alias out as a bug candidate.
- `mass_route`: `fm_msun > 3.0 or (main_sequence == True and m2_min_msun > 1.4 and m2_min_msun >
  m1_tilde_msun)` — bit-for-bit the documented disjunction (`CONTINUATION_PLAN.md:1241`). The
  `second_route_requires_main_sequence` cut parameter is unused dead config (the `main_sequence ==
  True` term inside the expression already enforces it) — harmless, not a correctness bug.
- The declarative AST evaluator's `or`/`and` short-circuiting (`sample_selection._eval_boolop`) was
  checked directly: a route that evaluates `True` short-circuits before the other route's
  `NotApplicable` (e.g. `m2_min_msun`) can propagate out, so an evolved source with `fm_msun > 3.0`
  correctly passes without spuriously going N/A. Confirmed by direct read of `_eval_boolop` /
  `_eval_node`, not just by the aggregate count matching.

**Root cause of the 123-vs-151 gap: entirely the same extinction mechanism #232/#258 already
diagnosed for `primary_ns_bh`, not an independent spectroscopic-branch bug.** `main_sequence` (and
therefore `m1_tilde_msun`, and therefore the whole `main_sequence_min_companion_mass` route) is
computed from `mg_0`/`bp_rp_0`, which `sample_selection.candidate_to_selection_row` aliases verbatim
from raw `abs_g_mag`/`bp_rp` with **zero** dereddening applied — the identical dead `ExtinctionSpec`
path #232 found for the astrometric branch. Per this issue's instructions, **the extinction mechanism
itself was not touched** (blocked on #258, a pending human decision) — only measured.

**Diagnostic-only measurement (not a claimed fix, not landed), `main` @ `31ad1f2`**, following #232's
`mwdust.Combined19` methodology exactly (all-sky, declination-unaware substitute for the frozen
Green2019/Lallement2019 split — **not paper-faithful**, same caveat as §3.3.1): dereddened
`mg_0 = mg_0_raw − 2.66·E(B−V)`, `bp_rp_0 = bp_rp_0_raw − 1.33·E(B−V)` (the frozen `elbadry2026.yaml`
coefficients) for all 181,342/181,534 SB1/SB1C rows with a positive parallax, using galactic
`(l, b, d)` from the real cache's `ra_deg`/`dec_deg`/`parallax_mas`, then re-ran the spectroscopic
branch's own two cuts (no other code path touched):

| Check | Target | Baseline (`main`, unextincted) | Combined19-dereddened (diagnostic) |
|---|---:|---:|---:|
| Spectroscopic branch survivors | 151 | 42 → **123** | **154** |
| Route: main-sequence min-companion-mass | 136 | 98 | **141** |
| Route: `f_m > 3 M☉` | 30 | 30 | **30** (unaffected — no `main_sequence` dependence) |
| Route: both | 15 | 5 | **17** |

Arithmetic check holds both times: `98+30−5=123`, `141+30−17=154`. The `f_m > 3 M☉` route count is
untouched by dereddening, exactly as expected since it has no `main_sequence`/`mg_0` dependence — a
useful sanity check that the diagnostic isolated the right mechanism, mirroring §3.3.1's
`elbadry2023_table_e1`/`andrews2022_import` null check for the astrometric branch.

**Disposition, per this issue's decision rule: dereddening closes essentially the entire gap** (123→154
against a target of 151 — it now *slightly overshoots*, consistent with Combined19 being an
approximate all-sky stand-in for the frozen north/south split rather than the real thing). This is
materially stronger evidence than `primary_ns_bh`'s ~80%-closure result in §3.3.1: for the
spectroscopic branch, the diagnostic dereddening alone is sufficient to reach (and slightly exceed) the
published count. **No spectroscopic-branch-specific code bug was found or fixed** — same disposition as
§3.3.1: no source code changed, real fix stays blocked on #258's still-pending map-acquisition decision.

**Simon 2026 exclusion breakdown re-check (§3.4) — attempted, informational only, incomplete.** A
first pass fed `simon2026_exclusion_breakdown` only the dereddened `primary_ns_bh` subsample's
survivors (reproducing `primary_ns_bh` = 46, consistent with §3.3.1) as `sample_ids`, which is **not**
the right membership set — the previous 5/2/1/0 baseline (§3.4) was measured against the full
`elbadry2026` sample union (astrometric ∪ spectroscopic, 1088 rows), not one subsample alone — and
predictably came out worse (`astrometric_f2_above_max` 3 vs target 2, `unclassified` rose to 5) because
several of the Simon table's 20 sources are only "in sample" via `sub_chandrasekhar` or the
spectroscopic branch. A corrected re-run unioning dereddened `primary_ns_bh` ∪ `sub_chandrasekhar` ∪
spectroscopic survivors (matching the original measurement's membership scope) was started but the
`sub_chandrasekhar` leg's per-row NSS Monte Carlo (`σ_M̃2`, lazy full-covariance) did not finish within
this session's time budget. **This diagnostic re-check is therefore left incomplete rather than
reported speculatively** — whoever next has budget for a ~5–10 min `sub_chandrasekhar`-only MC run
should redo the full-union version; the isolated-branch harness used here (bypassing `evaluate_all`'s
25-minute full sweep) is fast enough to make that cheap once resumed.

### 3.3.3 `M̃2` point-estimate chain audit (#275): the chain is right; `sub_chandrasekhar` is a selection-definition gap

**Provenance.** Branch `feat/elbadry2026-m2-chain-275`, cut from `main` @ `2470fe5` (#258 real maps
landed). Isolated worktree, `PYTHONPATH` pointed at the worktree `src/` (checked with
`darkhunter_pop.__file__`). Cache `…+enrich+mc10000` (the #257 rebuild) and the warm
`ebv_cache`. Measured 2026-09-27. To stay inside the memory budget, `evaluate_all` was not re-run.
Only the astrometric branch was enriched (168,065 `Orbital`/`AstroSpectroSB1` rows, through
`SampleSelection._enrich_rows_for_spec`), and the fixed-`M̃1` `σ_M̃2` MC was run on just the
3043-source `m2_range` window (`_attach_elbadry_m2_sigma_inplace`: 1268 s, 3.9 GB peak RSS). This
path reproduces the §3.3 attrition exactly: `168065 → 147560 → 3043 → 1687 → … → 1265`
(σ missing for 15). The σ run gives the same missing-σ count.

**1. The point-estimate chain reproduces the paper's own per-source values.** El-Badry 2026
Table 7 (`config/selections/external/elbadry2026_table7.yaml`, all 76 astrometric members)
publishes `A`, `M̃1`, `M̃2` and `E(B-V)` for each source. Given the paper's own `E(B-V)` (which
separates our chain from our map lookup), 73 of the 76 rows are main-sequence on our CMD cut. On
all 73, our `A` / `M̃1` / `M̃2` match Table 7 with median relative differences of **0.03 % / 0.06 % /
0.03 %**. The worst case is `2032579979951732736`, at 0.5 % / 1.7 % / 0.8 %, which is consistent with
the table's rounded `E(B-V)`. With our own map `E(B-V)` the median ratios stay within 0.4 % of unity (16–84 % range within ±2 % for `M̃1` and ±0.7 % for `M̃2`).
The chain being checked here is Janssens inversion → `photocenter_a0_from_thiele_innes` → AMRF
→ dark-companion inversion, on the NSS parallax. It is now pinned by
`tests/test_elbadry2026.py::test_point_chain_reproduces_elbadry2026_table7`, with real DR3 inputs in
`tests/fixtures/selections/elbadry2026_table7_dr3_inputs.yaml`. The other three rows
(`5870569352746779008` = Gaia BH2, `3664684869697065984`, `6152333294796189568`) are evolved on our
cut. They enter through subsamples 2 and 3, which need no MS cut.

**2. Input decomposition on the 3043-source `m2_range` window**, measured:

| Input | Effect on `M̃2` | Sources leaving the window |
|---|---|---|
| `a0` method (nsstools vs `photocenter_a0_from_thiele_innes`) | **0**, bit-identical (#199: \|Δa0\| ≤ 1.2e-13 mas) | 0 |
| `M̃1` via extinction (real maps vs `E(B-V)=0`) | median `ΔM̃1/M̃1` +8.8 % → median `ΔM̃2/M̃2` +4.5 % | 960 |
| Parallax ±1σ | median \|`ΔM̃2/M̃2`\| 2.7–2.8 % | 435 (+1σ) / 206 (−1σ) |
| Parallax zero point (0.017 mas) | median 2.2 % | 347 — **but the paper uses raw parallax in Eq. 2** (its ZP is §5.1.3, later), as do we |

Extinction is the only lever that moves the window materially, and it moves it the wrong way
(§3.3.1). No input in the chain can shrink 3043 → ~20.

**Correction (§3.3.7, #315).** The paper's actual subsample 4 is **not** the 22 Table 7 rows in the
`M̃2` window used below. `1581117310088807552` and `747174436620510976` are Andrews imports
(subsample 3). In their place the true 22 include `2080945469200565248` and `3494029910469026432`,
whose Table 7 `M̃2` is above 1.40. The paragraph below is kept as measured.

**3. Where the 1265 differ from the paper's subsample 4.** The Table 7 rows with
`1.05 ≤ M̃2 ≤ 1.40` number 22. Two of them (`1581117310088807552` at P = 927 d, `747174436620510976`
at P = 999 d) fail the paper's own `P ≤ 900 d`. They are El-Badry 2024 Table 3 (Andrews/Shahaf) NS
candidates and are most likely subsample-3 imports, so the paper's 22 cannot be recovered from
Table 7 exactly. `6037767138131854592` (paper `M̃2` 1.383; ours 1.4002 on our `E(B-V)` of 0.208 vs
the paper's 0.165) lands in our `primary_ns_bh` instead. Of the 22, **17 are in our 1265**. Two
more fail only on our MC `σ_M̃2` (`3389767036738482432` σ 0.140, `4466767229088016256` σ 0.119;
the analytic-vs-MC `σ` gap from #199).

**Every** paper member of this subsample has `M̃2/M̃1 ≥ 1.10` and `A ≥ 0.669` (max `M̃1` = 1.20).
Our 1265 have median `q = M̃2/M̃1 = 0.79` and median `A = 0.54`, and 56 % of the window has
`M̃1 > 1.5`. These are massive main-sequence primaries for which `M̃2 ∈ [1.05, 1.40]` is an
ordinary `q < 1` companion. The criteria in the paper's §2.1, as transcribed and re-read against
the arXiv HTML (2026-09-27), admit them: `MS`, `1.05 ≤ M̃2 ≤ 1.40`, `σ_M̃2 ≤ 0.105`,
`P ≤ 900 d`, `G < 15`. The paper states no ratio or AMRF threshold and no `σ_M̃2` method.
Counterfactuals on our 1265 (**diagnostic only, not proposed config**):

| Extra condition | N | paper members kept (of 17) |
|---|---:|---:|
| none (frozen) | 1265 | 17 |
| `M̃2 > M̃1` | 68 | 17 |
| `M̃2/M̃1 > 1.05` | 44 | 17 |
| `A > 0.65` | 44 | 17 |
| `M̃1 < 1.25` | 183 | 17 |
| Janssens fit σ (zero `a`–`b` correlation) added to `σ_M̃2` (the Q12 alternative) | 939 | 17 |

**Conclusion.** No code bug in the `M̃2` chain. `sub_chandrasekhar` needs a PI decision on an
unstated selection criterion and/or the `σ_M̃2` method: **#284**. Do not add a ratio or AMRF cut to
the frozen file without that decision and a `schema_version` bump.

**4. `primary_ns_bh` 46 vs 47, fully reconciled against Table 7.** Our 46 is a strict subset of
Table 7. Applying subsample 1's cuts to Table 7's own values (with our `F2`) gives exactly 47.
We miss `6054379247042197504` and `6152333294796189568`, which are MS in the paper and evolved on
our CMD. The first becomes MS with the paper's `E(B-V)` (0.201 vs our 0.166). The second stays
evolved even at the paper's 0.051 (`bp_rp_0` 0.973 vs the 0.934 needed at `MG,0` 3.16). We add
`6037767138131854592`, the `M̃2 = 1.4002` boundary case above. All three are `E(B-V)`/colour
boundary effects (#133), not the `a0`/AMRF chain.

**5. Simon 2026 last slot: a fixture defect, not the chain (#281).** `3263804373319076480` **is** in
El-Badry 2026's sample. Table 7 lists it with `A = 1.09`, `M̃1 = 1.16`, `M̃2 = 2.925`, and our chain
gives 1.088 / 1.166 / 2.931. The "M̃2 ≈ 4× the paper" in §3.4 compared our `M̃2` against Simon's
catalog `m2_lower` (0.73), which is a different estimator. `1864406790238257536` is **not** in
Table 7 or Table 8. It is the only excluded Simon source that is astrometric with `G < 15` and
`F2 < 10`, so it must be the paper's single `fails_m2_over_m1` exclusion, and our chain agrees
(`M̃2/M̃1` = 0.55 after #258). The real pipeline's membership for both sources therefore already
matches the paper. The breakdown read 5/2/1/0 (+1) because `classify_simon2026_row` judged the
ratio on Simon's catalog value (5.08), and because the unit-test fixture had the two sources
swapped. Fixed in #281: the classifier now takes El-Badry's own `M̃2/M̃1`, the fixture is derived
from Table 7 ∪ Table 8, and the breakdown is **5 / 2 / 1 / 1, `in_sample` 11, 0 unclassified**.
**#285 wired it through the stages.** `sample_selection` now persists El-Badry 2026's own
`(M̃1, M̃2)` per enriched row (`samples/elbadry2026/tilde_masses/`), and `diagnostics` hydrates
`simon_elbadry_m2_over_m1` from it. Measured with the real `run_sample_selection_stage` on the uncut
snapshot (443,211 rows; El-Badry 2026 N = 1565; 147,560 ratios persisted) plus
`_hydrate_diagnostics_from_manifest` and the Simon hook, on `main` @ `7c40198` + the #285 branch:
**5 / 2 / 1 / 1, `in_sample` 11, 0 unclassified**. `1864406790238257536` hydrates to `M̃2/M̃1` = 0.550.
Forcing the fallback on the same artifact gives 5 / 2 / 1 / 0 (+1). Peak RSS 12.5 GiB, 63 min
(E(B-V) cache cold). A full 14-stage run could not be used because the dry run currently stops at
`data_acquisition` on an unrecognized duplicate-`source_id` shape (**#290**).

### 3.3.5 PI decision on the `E(B-V)` conversions (#295) — re-measured, no count moves

PI decision (Ryan Foley, 2026-09-27), verbatim: *"Use the Gaia conversion. Use Rv = 3.1 unless it
says something else. Make sure these numbers are all in the config and easy to adjust."*

Sources read, for each step:

| Step | What the sources say | Adopted (config key) |
|---|---|---|
| `E(B-V) → A_G, E(BP−RP)` | El-Badry 2026 §2 states the constants `E(BP−RP) = 1.33 E(B−V)`, `A_G = 2.66 E(B−V)` ("appropriate for… Teff ≈ 6000 K"). §4 separately uses Cardelli R_V = 3.1 for SED priors | Frozen 2.66 / 1.33, unchanged: `sample_selection.dust_maps.gaia_band_extinction.law: paper_constant` |
| `A0 ↔ E(B-V)` | Gaia Collaboration, Babusiaux et al. 2018 (A&A 616, A10) §2: "We assume `A_0 = 3.1 E(B-V)`". Lallement et al. 2019 App. A converts Bayestar to `A0` with `0.88 × 3.1`, "assuming R = 3.1". No source states another R_V | `dust_maps.r_v: 3.1`; Lallement `native_quantity: a0` → `E(B-V) = A0 / r_v`, **derived** from `r_v` (no baked 0.3226) |
| Bayestar19 → `E(B-V)` | Green et al. 2019 / Argonaut usage page: `E(B-V) = 0.884 × Bayestar19` via SF11 Table 6 (R_V = 3.1, F99, 7000 K) `E(B-V) = 0.981 E(g−r)`; the same page gives `0.996` via `E(r−z)` | `dust_maps.maps.green2019.native_to_ebv: 0.884` (tabulated, **not** recomputed from `r_v`) |

**Reading of "the Gaia conversion".** Implemented as the Gaia collaboration's `A0 = 3.1 E(B-V)` (the
`A0`-to-`E(B-V)` conversion the question was about), which coincides with the R_V = 3.1 instruction.
A second reading is the Gaia collaboration's colour/`A0`-dependent band law (Babusiaux 2018 Eq. 1 /
Table 1) *instead of* the paper's constants. That contradicts what the paper says it did, so it is not
the reproduction default. It is implemented as a one-line switch
(`gaia_band_extinction.law: babusiaux2018`, coefficients in config). Note: the paper's 2.66 equals
`3.1 × k_G` of that law at `(BP−RP)_0 ≈ 0.72`, `A0 → 0`, but its 1.33 does not (the law gives ≈ 1.49 there).

Provenance: branch `feat/ebv-conversion-pi` @ `1d54daa` (= `main` @ `c07897e` + #295; the later merge of `main` @ `a5ce757` brings only #274 counting helpers and #221 data-acquisition collapse, neither of which touches cut evaluation or this cache), isolated
worktree, `PYTHONPATH` → worktree `src/` (verified), cache `…+enrich+mc10000` (443,211 rows, 823,999,664
bytes, mtime 2026-09-27 02:18:53), warm native `E(B-V)` cache `elbadry2026_30a69867aa069bca.h5` (unchanged:
`dust_maps.py` was not edited). `andrews2022` first (N = 63), then `elbadry2026` with that membership.
Peak RSS for both variants in one process: **7.3 GB** (`/usr/bin/time -l`; warm cache).

| Check | Target | `paper_constant` (**landed default**) | `babusiaux2018` (sensitivity) |
|---|---|---|---|
| Published union (`n_surviving`, non-unique) | 227 | **1565** (1504 distinct; #274 per-type distinct **1507**, same membership as §3.3.4's real-map set) | 1584 (1522 distinct) |
| Astrometric branch union | 76 | **1356** | 1372 |
| Spectroscopic branch | 151 | **151** | 153 |
| Spectro routes (MS min / high `f_m` / both) | 136 / 30 / 15 | **132 / 30 / 11** | 137 / 30 / 14 |
| `primary_ns_bh` | 47 | **46** | 45 (loses `6037767138131854592`) |
| `elbadry2023_table_e1` | 5 | **5** | 5 |
| `andrews2022_import` | 16 | **55** | 55 |
| `sub_chandrasekhar` | 22 | **1265** | 1282 |
| Simon breakdown | 5 / 2 / 1 / 1 | **5 / 2 / 1 / 1**, 0 unclassified, `in_sample` 11 | 5 / 2 / 1 / 1 |
| Extinction outcomes | — | `ok` 318,964 · `beyond_map_limit` 30,446 · `invalid_parallax` 189 | same + `gaia_law_nonconvergent` 60 |

The landed default is numerically identical to the #258 numbers in §3.3: the confirmed factors are the
ones #258 already used, and it is the same arithmetic. `primary_ns_bh` attrition
`168065 → 147560 → 1052 → 91 → 82 → 59 → 46`; `sub_chandrasekhar`
`168065 → 147560 → 3043 → 1687 → 1275 → 1265`; spectroscopic `181534 → 133642 → 151`. Under
`babusiaux2018`, 60 very red, highly extincted rows fail the law's fixed-point `(BP−RP)_0` solve (outside
its fitted 3500–10000 K range). They are `NotApplicable("extinction_gaia_law_nonconvergent")` and
counted, never passed through.

### 3.3.6 `sub_chandrasekhar` after the #284 decision: `M̃2 > M̃1`, AMRF cut, analytic `σ_M̃2`

**Provenance.** Branch `feat/elbadry2026-subchandra-284` @ `0cd19bc` (= `main` @ `3df5da1` + the #284
commits), isolated worktree, `PYTHONPATH` → worktree `src/` (checked with `darkhunter_pop.__file__`).
Same path as §3.3.3: the 168,065 astrometric-branch rows of the `…+enrich+mc10000` cache went through
`SampleSelection._enrich_rows_for_spec` with the real maps and the warm `ebv_cache`. That gives
`147,560` main-sequence rows, and **3043** in the `m2_range` window, the same as §3.3. It took 70 s at
2.3 GB peak RSS. `σ_M̃2` was then computed on that window only, and the combinations were evaluated
from the stored columns. The default configuration was cross-checked through the real
`SampleSelection.evaluate` on the window rows, and gives the same 49. Measured 2026-09-28. **Re-measured after
merging #307** (PI `E(B-V)` conversions, §3.3.5), at `6e1d6e1`. Every number in this section is
identical there: window 3043, every sweep cell, and the real-evaluator 49 / 19. As §3.3.5 records,
the landed conversions are the ones #258 already used.

**What landed** (`config/selections/elbadry2026.yaml` schema_version 2, sanctioned by Ryan Foley's
#284 decision):

- `sub_chandrasekhar` gains `m2_over_m1` (`M̃2/M̃1 > 1.0`) and `amrf`
  (`amrf > amrf_threshold_elbadry2026`), placed before `m2_error`.
- The `amrf_cut` block selects the threshold:
  - `flat` uses `flat_min`, default **0.65**, the #275 counterfactual.
  - `shahaf2019_class3` uses the Shahaf et al. (2019) class-II/III boundary `max_q A_triple(q; M̃1)`,
    for an equal-mass close MS pair that is fainter than the primary. It is available with three
    mass–luminosity relations:
    - `shahaf2019_hp`: their Table A1 Hipparcos relation, primaries 0.6–1.8 M☉. This reproduces their
      Fig. 3 to within 0.01 (unit-tested).
    - `janssens2022_g`: Gaia G, the same relation El-Badry 2026 uses for `M̃1`.
    - `power_law`: reproduces their published β = 5 maxima of 0.36 and 0.56.
- `sigma_m2_tilde.method: analytic` is a first-order Jacobian of `M̃2(A, B, F, G, ϖ, P; M̃1)`.
  `analytic_covariance` has two settings:
  - `full`: the 6×6 block of the NSS covariance, cross terms kept. This is the default.
  - `nsstools_blocks`: nsstools' `σ_a0` from the `(A, B, F, G)` block, then `ϖ` and `P` in quadrature.

  `monte_carlo` is still available.
- Janssens `M̃1` fit σ enters when `primary_mass.propagate_fit_uncertainty` is set. That remains
  `false` (Q12).
- `_sigma_m2_astrometric_provenance` names the method and the `M̃1` treatment, e.g.
  `elbadry2026_analytic_full_m1_tilde_fixed`.

Frozen window / σ / period / G thresholds are unchanged.

**Correction (§3.3.7, #315).** The "22" in this section is the set of Table 7 rows in the `M̃2`
window. The paper's actual subsample 4, found by removing subsamples 1–3 from Table 7, swaps the
two P > 900 d Andrews imports below for `2080945469200565248` and `3494029910469026432`. The same 19
are in both sets. Schema_version 3 replaces this chain in reproduction mode.

**(1) The two MC-σ failures are recovered.** With the analytic method:

- `3389767036738482432`: MC σ 0.140 → analytic 0.062.
- `4466767229088016256`: MC σ 0.119 → analytic 0.064.

Both pass every #284 cut in every analytic variant, so our set now holds **19 of the paper's 22**
(was 17). The other three:

- `1581117310088807552` (P = 927 d) and `747174436620510976` (P = 999 d) fail the paper's own
  `P ≤ 900 d`, so they cannot be subsample-4 members.
  - Both are Andrews et al. (2022) published members (El-Badry 2024 Table 3, reference "A22"), with
    G = 14.51 and 13.99. They therefore reach El-Badry's table through `andrews2022_import`
    (published Andrews ∩ `G < 15` = 16).
  - With #296 (ATF selection notebook, §3.1.4, merged into this branch at `d923c83`), the Andrews
    reproduction is the published 24 plus one extra at G = 15.15, and `andrews2022_import` is exactly
    the published 16. **Both sources are therefore now recovered through ATF.** Under the earlier
    FLAME-at-selection rule (#257) both were dropped (p 0.903 / 0.638, §3.1.3).
- `6037767138131854592`: our `M̃2` = 1.40020 against the paper's 1.383. The difference comes from our
  map `E(B-V)` of 0.208 against the paper's 0.165. It is a window-edge / `E(B-V)` boundary case, not
  a σ or cut effect.
  - It falls outside the window by 2 × 10⁻⁴ M☉, so it is never σ-tested here, and it lands in our
    `primary_ns_bh` instead.
  - Rounding `M̃2` to Table 7's 3 decimals (1.400) would admit it. That would be an unstated
    edge-treatment choice, so it is **not** adopted.

**(2) Sample size with the new cuts.** The default (`M̃2/M̃1 > 1`, `A > 0.65`, analytic-full σ,
`M̃1` fixed) gives **49**, containing 19 of the paper's 22 (all 19 of its 20 `P ≤ 900 d` members that
are in our window). Attrition:

```
3043 (m2_range) → 256 (m2_over_m1) → 182 (amrf > 0.65) → 101 (m2_error) → 58 (period) → 49 (G < 15)
```

With the Shahaf class-III boundary instead of 0.65, the size is 70 (Hp relation) or 77 (Janssens G).
Both contain the same 19.

For point estimates `A > 0.65` implies `q > 1.049`, so the flat cut makes `M̃2 > M̃1` redundant. The
class-III boundary (0.62–0.67 in Hp, 0.44–0.64 in G for these `M̃1`) is looser than 0.65 and much
looser than `q > 1` (`A(q=1) = 0.63`). With the Janssens-G relation it removes nothing beyond
`M̃2 > M̃1`.

**(3) Combination sweep.** All rows below include `M̃2/M̃1 > 1`, the frozen window, `σ ≤ 0.105`,
`P ≤ 900 d` and `G < 15`. Entries are N / paper members kept (of 22):

| `σ_M̃2` method | `M̃1` σ | no AMRF cut | `A > 0.65` | class III (Hp) | class III (Janssens G) |
|---|---|---:|---:|---:|---:|
| analytic, full cov | fixed | 77 / 19 | **49 / 19** | 70 / 19 | 77 / 19 |
| analytic, full cov | Janssens | 71 / 19 | 44 / 19 | 64 / 19 | 71 / 19 |
| analytic, nsstools blocks | fixed | 76 / 19 | 48 / 19 | 69 / 19 | 76 / 19 |
| analytic, nsstools blocks | Janssens | 68 / 19 | 43 / 19 | 61 / 19 | 68 / 19 |
| MC (10⁴ draws) | fixed | 68 / 17 | 44 / 17 | 63 / 17 | 68 / 17 |
| MC (10⁴ draws) | Janssens | 58 / 17 | 39 / 17 | 53 / 17 | 58 / 17 |

Without `M̃2 > M̃1` (analytic-full, `M̃1` fixed), the sizes are: σ alone 1605, `A > 0.65` 49, Hp class
III 134, G class III 541. Every analytic variant keeps the same 19 members.

**No combination reproduces the paper's set.** Against Ryan's 17 (our pre-#284 overlap) or against
19 (the 17 plus the two σ recoveries), the smallest analytic size is 43. That is 24 non-members
beyond the 19, from nsstools-blocks σ with the Janssens `M̃1` σ and `A > 0.65`.

The 30 extras in the default 49 look like the paper's members, which have `A` ≥ 0.669 and `q` ≥ 1.096
on our values:

- `q` 1.05–1.41 and `A` 0.652–0.784;
- analytic σ 0.012–0.097 and G 11.6–15.0;
- F2 spanning the same range as the members (one member has F2 = 11.3, so an F2 < 10 cut is not
  implied).

About 17 of the 30 have `A` < 0.669. A threshold near the members' minimum `A` would remove them, but
it would be tuned to the answer, so it is not proposed. The remaining gap is therefore not a σ-method
or AMRF-variant question. It is an unstated selection or vetting step, or the E(B-V) conversion.

**Adopted default:** `amrf_cut.criterion: flat`, `flat_min: 0.65`. This matches the "≤ 44 objects"
in Ryan's decision (the #275 counterfactual) and is the tightest of the tested variants. To use the
paper-formula boundary instead, set `amrf_cut.criterion: shahaf2019_class3` with a `mass_luminosity`
choice. The σ default is analytic, full covariance, `M̃1` fixed. Flipping any switch is a config
edit, with the counts above.

### 3.3.7 `sub_chandrasekhar` = Shahaf et al. (2023b) class III + Eq. 5 on Shahaf's masses: exactly the paper's 22 (#284, #315; schema_version 3 landed)

**Landed (schema_version 3).**

**PI decision (Ryan Foley, 2026-09-29), verbatim:** *"We should always try to match the papers when
trying to reproduce their samples. Let's use the Shahaf criterion for the El-Badry sample."* This
was resolved to the reading that matches the paper: a cross-match to Shahaf et al. (2023b) Table 2.

- **Snapshot.** VizieR `J/MNRAS/518/2991/table2`, 177 rows, queried 2026-09-29, SHA-256
  `7b6d287e…52ec0`. It is stored at
  `data/dr3/external_catalogs/shahaf2023b_class3_20260929T195047Z_7b6d287e/` (`table2.ecsv` +
  `meta.yaml`) and was written by `scripts/fetch_shahaf2023b_class3.py`.
  - Config key: `dr3.shahaf2023b_class3_snapshot`, with `dr4: null`. The key is path-specific, is in
    `PATH_SPECIFIC_LEAF_KEYS`, and is part of the `sample_selection` stage fingerprint. The DR3/DR4
    audit reports no violation.
  - The checksum is verified on load.
- **Reproduction chain.** `sub_chandrasekhar` declares `external_catalog: shahaf2023b_class3`. In
  reproduction mode its cuts are:
  - `in_sample('shahaf2023b_class3')`;
  - `main_sequence`;
  - `1.05 ≤ shahaf2023b_m2min_msun ≤ 1.40`;
  - `shahaf2023b_m2min_error_msun ≤ 0.105`;
  - `P ≤ 900 d`;
  - `G < 15`.

  These are the paper's published thresholds, and no value changed.
- **Which mass the cuts use.**
  - The text points both ways. Eq. 5's `M̃2` / `σ_M̃2` notation is defined in §2 through the
    Janssens `M̃1`, but the paper also says "These cuts yielded 22 sources, none of which were
    included in the three subsamples listed above". Only Shahaf's own `M2min` / `e_M2min`
    reproduce that stated result.
  - On the Janssens `M̃2`, two members (`2080945469200565248`, `3494029910469026432`) sit above 1.44
    at every `E(B-V)` ≥ 0.
  - Applying the Janssens chain to catalog members instead gives **19**, all of them paper members,
    missing those two and `6037767138131854592`.
  - Reproduction mode therefore uses Shahaf's columns. This is flagged for Ryan: if the notation
    reading is preferred, it is a two-cut edit (19 of 22).
- **Forward model.** Unchanged. The schema-v2 Janssens chain (`m2_range`, `m2_over_m1`, `amrf`,
  `m2_error`) now carries `applies_to: [forward_model]`, and the catalog is never read in that mode.
  A catalog cannot be applied to mocks; that is §15 **Q2**.

**Re-measurement.**

- Setup: branch `docs/subchandra-extras-284` @ `13b14b6` plus the schema-v3 changes, an isolated
  worktree with `PYTHONPATH` → worktree `src/` (verified), and the 168,065 astrometric-branch rows of
  the `…+enrich+mc10000` cache with the real maps and the warm `ebv_cache`.
- The run used the real `SampleSelectionRegistry(load_config()).selection("elbadry2026").evaluate`
  on the 147,560 main-sequence rows. It took 332 s at 5.1 GB peak RSS.

```
147560 (MS) → 177 (shahaf2023b_class3_membership) → 177 (main_sequence) → 33 (m2_range_shahaf2023b)
       → 29 (m2_error_shahaf2023b) → 25 (period ≤ 900 d) → 22 (G < 15)
```

- **N = 22, the same 22 source IDs as the paper's subsample 4**, with no misses and no extras. All
  177 Table 2 sources are in our parent and on our main sequence.
- Forward model on the 3043-row Janssens window: schema v2 gives 49 and schema v3 gives 49, the
  **identical set**.
- There are no extras left, so there is no extras table for the new sample. The investigation below
  is on the schema-v2 sample (49), which is still the forward-model set.
- Not re-measured here: the published union (227) and the astrometric union (76). They need the full
  `evaluate_all` (~12 GB).

---

The investigation that led to the decision (measured on schema v2, 2026-09-28) follows.

**Question (Ryan Foley, 2026-09-28, on #284).** *"We should still try to figure out the El-Badry
sub-Chandra sample — is there something else like the RUWE that is different? Period?"*

**Provenance.**

- Code: `main` @ `774abcd`, isolated worktree, `PYTHONPATH` → worktree `src/` (checked with
  `darkhunter_pop.__file__`).
- Rows: the 168,065 astrometric-branch rows of the `…+enrich+mc10000` cache, enriched with
  `SampleSelection._enrich_rows_for_spec` using the real maps and the warm `ebv_cache`.
- The default schema-v2 `sub_chandrasekhar` was then run through the real `SampleSelection.evaluate`
  on the 3043-row window: `3043 → 256 → 182 → 101 → 58 → 49`, the same as §3.3.6. That run took
  571 s at 2.95 GB peak RSS.
- NSS columns (`significance`, `bit_index`, errors) came from the `nss_enrichment` snapshot.
- Columns not in any local snapshot came from a **Gaia archive TAP query** on 2026-09-28, covering 238
  rows (the 182 post-AMRF rows plus the 76 Table 7 rows). These are `flags`, `efficiency`,
  `astrometric_n_good_obs_al`, `visibility_periods_used`, `ipd_*`, the RV columns,
  `phot_variable_flag`, and `l`/`b`.
- The Shahaf et al. (2023b) catalog (MNRAS 518, 2991) came from **VizieR `J/MNRAS/518/2991`** on
  2026-09-28. Table 1 has the 101,380-row clean sample with `PIII`; Table 2 has the 177 "highly
  probable class-III" systems.
- The paper text was re-read from arXiv:2608.06453v1: §2, §2.1, §5.1.2, §6.4, Fig. 8, and the Table 7
  note.
- Nothing was written to `config/` or `data/`.

**1. The paper's 22 are identified by elimination, and the identification corrects §3.3.3 / §3.3.6.**

- §2.1 says the four astrometric subsamples are 47 + 5 + 16 + 22, with overlaps of 2 (E1 ∩ primary)
  and 12 (Andrews ∩ the first two). Subsample 4 overlaps none of the others, which gives the union
  of 76.
- Applying subsamples 1–3 to Table 7's own values gives exactly 47 / 5 / 16:
  - subsample 1 uses our F2, as in §3.3.3 item 4;
  - subsample 2 is the E1 fixture with `G < 15`;
  - subsample 3 is the El-Badry 2024 Table 3 `A22` rows with `G < 15`.

  The overlaps are exactly 2 and 12. That leaves **exactly 22** rows, which must be subsample 4.
- Those 22 are **not** the 22 Table 7 rows with `1.05 ≤ M̃2 ≤ 1.40` that §3.3.3 and §3.3.6 counted.
  - `1581117310088807552` and `747174436620510976` (P > 900 d) are Andrews imports, as §3.3.6 already
    argued, so they are not subsample 4.
  - Their places are taken by `2080945469200565248` (Table 7 `M̃2` = 1.608, `q` = 1.08) and
    `3494029910469026432` (`M̃2` = 1.472, `q` = 1.14). These fail subsample 1's `M̃2/M̃1 > 1.2` and
    are in no other subsample.
- Both of those sources are outside the published window on Table 7's own `M̃2`. Our chain puts them
  above 1.40 at **every** `E(B-V)` ≥ 0: at `E(B-V)` = 0 it gives `M̃2` = 1.446 and 1.442. So they
  cannot have been selected by the Janssens-`M̃2` window in Eq. 5.
- Our schema-v2 49 contains **19 of these 22**. The 19 are the same sources as before. The three we
  miss are `6037767138131854592` (ours 1.4002), `2080945469200565248` (ours 1.621) and
  `3494029910469026432` (ours 1.471), all above the window on our values.

**2. No Gaia quality column separates the 30 extras from the 19 members.** The table gives the range
over each group. The last column counts extras outside the members' range. The members are
`sub_chandrasekhar` ∩ paper subsample 4 (19); the extras are the other 30.

| Quantity | Members (19): min / median / max | Extras (30): min / median / max | Extras outside members' range |
|---|---|---|---:|
| RUWE | 2.53 / 5.48 / 21.9 | 2.00 / 3.93 / 16.6 | 5 |
| Period (d) | 45 / 575 / 894 | 135 / 685 / 889 | 0 |
| `period_error` (d) | 0.005 / 2.0 / 12 | 0.33 / 6.2 / 56 | 8 |
| `e` / `eccentricity_error` | 0.05–0.75 / 0.004–0.082 | 0.007–0.72 / 0.008–0.082 | 1 / 0 |
| `significance` (a0/σa0) | 23.2 / 64.8 / 157 | 14.9 / 36.7 / 135 | 5 |
| F2 (`goodness_of_fit`) | −3.0 / 0.41 / 11.3 | −2.6 / 0.61 / 13.6 | 1 |
| `parallax_over_error` | 37 / 100 / 798 | 27 / 57 / 284 | 5 |
| `astrometric_n_good_obs_al` | 301 / 396 / 611 | 194 / 389 / 519 | 4 |
| `visibility_periods_used` | 17 / 23 / 29 | 14 / 22.5 / 31 | 3 |
| `flags` values | {0, 64, 192} | {0, 64, 8384} | 1 |
| `bit_index` | {8191, 65535} | {8191, 65535} | 0 |
| G / BP−RP / M_G,0 | 10.4–15.0 / 0.66–1.34 / 4.00–6.72 | 11.6–15.0 / 0.78–1.57 / 3.62–6.39 | 1 / 2 / 3 |
| distance 1/ϖ (pc) | 80 / 541 / **750** | 290 / 645 / 1250 | **13** |
| `E(B-V)` (ours) | 0 / 0.027 / 0.271 | 0 / 0.080 / 0.513 | 5 |
| A (AMRF) | **0.669** / 0.749 / 0.900 | 0.652 / 0.666 / 0.784 | **17** |
| `q = M̃2/M̃1` | 1.10 / 1.31 / 1.78 | 1.05 / 1.09 / 1.41 | 17 |
| analytic `σ_M̃2` | 0.015 / 0.040 / 0.090 | 0.012 / 0.059 / 0.097 | 6 |
| Dec (deg) / `l` / `b` | −66…+69 / 32…352 / −84…+78 | −71…+76 / 11…358 / −54…+72 | 2 / 2 / 0 |
| `ipd_frac_multi_peak` / `ipd_gof_harmonic_amplitude` | 0–2 / 0.003–0.074 | 0–2 / 0.005–0.035 | 0 / 0 |
| NSS type (Orbital / AstroSpectroSB1) | 16 / 3 | 27 / 3 | — |

Other checks on the same rows:

- **Gaia DR3 Orbital validity criteria.** These are the three criteria printed in cell 47 of the
  Andrews ATF notebook, from Halbwachs et al. 2023:
  - `ϖ/σϖ > 20000/P`;
  - `significance > 158/√P`;
  - `eccentricity_error < 0.079 ln P − 0.244`.

  All 49 pass all three. They are DPAC publication filters, so every published Orbital solution
  passes them.
- **Sample-level exclusions.** None of the 30 extras is in El-Badry 2023a E1, El-Badry 2024 Table 3,
  El-Badry 2026 Table 8, or Simon 2026. One extra, `131940776857845504`, also has an SB1 solution.
  Having an SB1 solution does not exclude a source.
- **RUWE.** Every one of the 49 has RUWE > 2, as expected for an astrometric orbit, so RUWE does not
  separate them.
- **Period.** The distributions overlap across 135–889 d.
- **Northern-map sky cut.** `dec > −28` would drop 4 of the 19 members.

Criterion counts. Each row gives the paper members kept (of 19) and the extras removed (of 30).
Rows marked *tuned* use a number read off the members' range. They are listed only to show how far
apart the two groups are, and none of them is proposed.

| Criterion | Source | Members kept | Extras removed |
|---|---|---:|---:|
| Halbwachs 2023 validity triplet | Gaia DR3 / ATF cell 47 | 19 | 0 |
| RUWE < 1.4 | common single-star cut | 0 | 30 |
| significance > 10 / > 20 | paper Fig. 8 discussion, SB1 cut analog | 19 / 19 | 0 / 4 |
| F2 < 10 / < 5 / < 3 | subsample-1 F2 cut and tighter | 18 / 18 / 15 | 1 / 5 / 7 |
| ϖ/σϖ > 10 / > 20 | common quality cuts | 19 / 19 | 0 / 0 |
| `flags == 0` | NSS flags | 1 | 27 |
| `ipd_frac_multi_peak ≤ 2`, `ipd_gof_harmonic_amplitude < 0.1` | common duplicity cuts | 19 | 0 |
| `dec > −28` (Bayestar hemisphere) | map boundary | 15 | 7 |
| P ≤ 700 d | — | 14 | 15 |
| *tuned* A ≥ 0.669 (members' minimum) | — | 19 | 17 |
| *tuned* d ≤ 750 pc (members' maximum) | — | 19 | 13 |
| *tuned* A ≥ 0.669 and d ≤ 750 pc and significance ≥ 23 | — | 19 | 23 |
| **Shahaf et al. (2023b) Table 2 membership** (class III, `PIII ≥ 0.99984`) | cited by the paper (§5.1.2, §6.4) | **19** | **30** |

Of the 30 extras, 27 are in Shahaf's clean Table 1 sample, with `PIII` from 0.586 to 0.9997. That is
below Table 2's floor, which is **exactly** `PIII ≥ 0.99984`: those 177 rows are Table 2. The other 3
extras (`131940776857845504`, `4129918029309568896`, `5728893740393030528`) are not in Shahaf's clean
sample at all. All 19 members have `PIII ≥ 0.99986`.

The extras sit just above our flat `A > 0.65` cut, at median `q` 1.09 against 1.31 for the members.
Shahaf's triage asks for `A` to exceed the class-II/III boundary with high probability given
`σ_A`, and our point-estimate AMRF cut does not reproduce that.

**3. The exact reproduction: Shahaf 2023b Table 2 with Eq. 5 applied to Shahaf's own masses.**
Eq. 5's cuts, with `M̃2` and `σ_M̃2` replaced by Shahaf's `M2min` and `e_M2min` (which come from the
`binary_masses` `M1`, not from Janssens), give:

```
177 (Shahaf 2023b Table 2) → 33 (1.05 ≤ M2min ≤ 1.40) → 29 (e_M2min ≤ 0.105) → 25 (P ≤ 900 d) → 22 (G < 15)
```

These are **exactly the paper's 22 subsample-4 source IDs**, with no misses and no extras.

- All 22 pass our `MS = true` cut, so that cut is redundant here.
- None of the 22 is in subsamples 1–3, as the paper states.
- The window edges are not knife-edge:
  - Inside the window, the members run from `M2min` 1.061 to 1.393.
  - The nearest Table 2 sources outside it are 1.042 and 1.429.
  - The one Table 2 source at 1.051 fails `G < 15` (15.11).
- The σ cut is binding, but only on sources that also fail G: the 4 it removes have G of 15.45–16.84.
- The two P > 900 d Andrews imports (`1581117310088807552`, `747174436620510976`) are in Table 2 and
  inside the window. The `P ≤ 900 d` cut removes them, so they are counted once, through subsample 3.

This also explains every earlier puzzle:

- The two members that no Janssens-`M̃2` window can reach have Shahaf `M2min` 1.381 and 1.318.
- `6037767138131854592` (our 1.4002) is at 1.393.
- The 30 extras are the Janssens-window sources that Shahaf's triage does not classify as class III.
- The Table 7 `M̃2` column was recomputed with the Janssens `M̃1` after selection, which is why it
  "cannot recover the paper's 22" (§3.3.3).

**4. What the paper states.**

- §2.1 item 4 does **not** name Shahaf 2023b as subsample 4's parent. It defines `M̃2` / `σ_M̃2`
  through the Janssens `M̃1` for every subsample.
- The paper does say that most of the astrometric sample are "class III" sources under the Shahaf et
  al. (2019, 2023) triage (§5.1.2).
- It also says El-Badry et al. (2024a) drew its NS candidates from "the AMRF sample of Shahaf et al.
  (2023)" (§6.4).
- Fig. 8's caption says there is "no simple set of cuts" that separates reliable from spurious
  orbits. That is consistent with no quality column separating the extras here.

Adopting Shahaf's catalog is therefore a cited-but-unstated criterion. It needs a PI decision.
**Ryan decided on 2026-09-29, and the change landed as schema_version 3 (top of this section).**

**5. Recommendation as written before the decision.**

- **Reproduction mode.** Redefine `sub_chandrasekhar` as membership in Shahaf 2023b Table 2, with
  `1.05 ≤ M2min ≤ 1.40`, `e_M2min ≤ 0.105`, `P ≤ 900 d` and `G < 15` applied to Shahaf's columns.
  - `MS` can stay; it is redundant.
  - This follows the pattern of `andrews2022_import` / `elbadry2023_table_e1`: a frozen external
    fixture under `config/selections/external/` plus a `schema_version` 3 bump.
  - It reproduces 22 exactly, and it would bring the astrometric union to the published structure.
- **Forward-model mode.** A catalog cannot be applied to mocks. A faithful forward model needs
  Shahaf's probabilistic triage: `PIII` from the AMRF and its uncertainty, with the `binary_masses`
  `M1` and a threshold ≈ 0.9998. This is §15 **Q2**, and it is the same parent the El-Badry 2024
  path uses.
- **Keep schema v2** until that decision (it was made; see above). Do not tune A, q, σ or distance to remove the extras.

Tracked as **#315**. The schema-v2 49 rows follow, with members first and each group sorted by significance.
Distances are 1/ϖ with no zero point. σ is analytic, full covariance, with `M̃1` fixed.

| source_id | in paper | type | P (d) | e | signif. | F2 | RUWE | ϖ/σϖ | d (pc) | E(B-V) | G | A | q | σ_M̃2 | S23 `PIII` | S23 Table 2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1522897482203494784` | yes | ASB1 | 46 | 0.05 | 156.6 | 1.9 | 6.4 | 798 | 80 | 0.000 | 11.05 | 0.900 | 1.779 | 0.015 | 1.00000 | yes |
| `4466767229088016256` | yes | Orb | 777 | 0.15 | 116.1 | -2.4 | 6.9 | 37 | 750 | 0.000 | 13.78 | 0.685 | 1.136 | 0.064 | 0.99993 | yes |
| `1694708646628402048` | yes | Orb | 632 | 0.26 | 114.9 | 0.8 | 8.0 | 85 | 550 | 0.027 | 13.20 | 0.729 | 1.254 | 0.031 | 1.00000 | yes |
| `1801110822095134848` | yes | ASB1 | 894 | 0.59 | 92.2 | 3.6 | 21.9 | 225 | 241 | 0.000 | 12.19 | 0.826 | 1.536 | 0.025 | 1.00000 | yes |
| `3184078579032750464` | yes | Orb | 615 | 0.15 | 87.4 | -0.0 | 5.1 | 81 | 699 | 0.062 | 13.73 | 0.682 | 1.129 | 0.034 | 0.99992 | yes |
| `4240540718818313984` | yes | Orb | 691 | 0.62 | 85.2 | 0.4 | 7.5 | 66 | 481 | 0.177 | 14.61 | 0.856 | 1.632 | 0.042 | 1.00000 | yes |
| `5039979680444075392` | yes | Orb | 553 | 0.18 | 83.8 | 3.2 | 9.3 | 135 | 375 | 0.000 | 12.72 | 0.720 | 1.229 | 0.028 | 1.00000 | yes |
| `1434445448240677376` | yes | Orb | 572 | 0.30 | 82.5 | -3.0 | 6.1 | 110 | 695 | 0.053 | 13.65 | 0.718 | 1.223 | 0.035 | 1.00000 | yes |
| `5355633933885075328` | yes | Orb | 575 | 0.11 | 76.8 | 0.0 | 6.0 | 100 | 711 | 0.060 | 13.73 | 0.669 | 1.096 | 0.029 | 0.99989 | yes |
| `1579254496872812032` | yes | Orb | 601 | 0.31 | 64.8 | -0.4 | 5.1 | 94 | 636 | 0.000 | 14.50 | 0.749 | 1.307 | 0.034 | 1.00000 | yes |
| `2032579979951732736` | yes | Orb | 215 | 0.15 | 63.1 | -1.0 | 3.3 | 133 | 546 | 0.080 | 14.21 | 0.789 | 1.424 | 0.038 | 1.00000 | yes |
| `3389767036738482432` | yes | Orb | 717 | 0.14 | 55.5 | 1.1 | 4.4 | 42 | 541 | 0.271 | 14.72 | 0.777 | 1.389 | 0.062 | 1.00000 | yes |
| `1683575679079854848` | yes | Orb | 796 | 0.62 | 49.6 | 0.1 | 5.5 | 79 | 728 | 0.000 | 14.30 | 0.724 | 1.239 | 0.042 | 0.99986 | yes |
| `6481502062263141504` | yes | Orb | 230 | 0.30 | 48.5 | 2.1 | 3.9 | 89 | 576 | 0.030 | 13.58 | 0.744 | 1.295 | 0.049 | 1.00000 | yes |
| `5446310318525312768` | yes | ASB1 | 867 | 0.25 | 47.0 | 4.3 | 11.2 | 226 | 187 | 0.004 | 10.37 | 0.692 | 1.155 | 0.040 | 0.99999 | yes |
| `2995961897685517312` | yes | Orb | 190 | 0.37 | 39.9 | 0.4 | 4.0 | 128 | 399 | 0.062 | 13.00 | 0.750 | 1.310 | 0.062 | 1.00000 | yes |
| `5283631903842076032` | yes | Orb | 91 | 0.31 | 31.3 | 11.3 | 5.0 | 243 | 296 | 0.009 | 13.31 | 0.774 | 1.380 | 0.056 | 1.00000 | yes |
| `220012968211559296` | yes | Orb | 196 | 0.41 | 25.3 | 0.1 | 2.5 | 113 | 313 | 0.150 | 13.93 | 0.783 | 1.407 | 0.077 | 1.00000 | yes |
| `2919995917769953408` | yes | Orb | 457 | 0.75 | 23.2 | -0.3 | 3.6 | 95 | 436 | 0.018 | 14.97 | 0.853 | 1.623 | 0.089 | 1.00000 | yes |
| `5728893740393030528` | no | ASB1 | 459 | 0.25 | 135.0 | 2.3 | 10.4 | 107 | 300 | 0.230 | 11.61 | 0.654 | 1.058 | 0.037 | — | no |
| `348572609373952768` | no | ASB1 | 775 | 0.28 | 115.7 | 2.9 | 12.5 | 108 | 328 | 0.053 | 11.77 | 0.662 | 1.077 | 0.031 | 0.99777 | no |
| `2102252286555969792` | no | Orb | 779 | 0.14 | 104.5 | 3.8 | 7.2 | 86 | 641 | 0.071 | 13.43 | 0.654 | 1.058 | 0.025 | 0.97326 | no |
| `2208943221256515712` | no | ASB1 | 889 | 0.05 | 99.5 | 2.6 | 13.9 | 284 | 290 | 0.151 | 12.69 | 0.680 | 1.124 | 0.012 | 0.99942 | no |
| `5929211736672889472` | no | Orb | 442 | 0.01 | 86.0 | -2.6 | 5.7 | 64 | 648 | 0.114 | 13.47 | 0.652 | 1.053 | 0.044 | 0.99972 | no |
| `471967157230679552` | no | Orb | 545 | 0.48 | 68.8 | 0.2 | 3.5 | 45 | 735 | 0.230 | 14.55 | 0.661 | 1.074 | 0.053 | 0.58630 | no |
| `426453663677608320` | no | Orb | 455 | 0.21 | 67.1 | 0.0 | 4.1 | 97 | 612 | 0.203 | 13.70 | 0.667 | 1.092 | 0.039 | 0.99553 | no |
| `611041256286259584` | no | Orb | 804 | 0.40 | 63.6 | -1.3 | 6.1 | 45 | 848 | 0.000 | 14.00 | 0.681 | 1.126 | 0.058 | 0.99563 | no |
| `1962476183678332672` | no | Orb | 559 | 0.42 | 62.7 | 0.1 | 5.4 | 106 | 573 | 0.221 | 14.44 | 0.702 | 1.180 | 0.034 | 0.99923 | no |
| `2191969549169214464` | no | Orb | 820 | 0.41 | 57.9 | 8.5 | 7.0 | 64 | 558 | 0.301 | 14.62 | 0.691 | 1.153 | 0.030 | 0.97231 | no |
| `5807146051670381312` | no | Orb | 854 | 0.22 | 54.7 | 0.8 | 3.2 | 32 | 1248 | 0.063 | 14.48 | 0.679 | 1.121 | 0.097 | 0.99669 | no |
| `4357622008333542912` | no | Orb | 668 | 0.36 | 54.0 | -0.2 | 4.0 | 47 | 579 | 0.221 | 14.15 | 0.676 | 1.114 | 0.057 | 0.96784 | no |
| `5528031588984234368` | no | Orb | 135 | 0.15 | 45.8 | 0.6 | 3.2 | 160 | 525 | 0.027 | 13.11 | 0.666 | 1.087 | 0.039 | 0.98815 | no |
| `131940776857845504` | no | Orb | 814 | 0.18 | 42.4 | 5.0 | 16.6 | 71 | 314 | 0.071 | 11.96 | 0.666 | 1.089 | 0.034 | — | no |
| `5341695601100252800` | no | Orb | 608 | 0.46 | 38.5 | -1.0 | 2.3 | 38 | 1052 | 0.097 | 14.97 | 0.655 | 1.060 | 0.057 | 0.87918 | no |
| `2145894170800250624` | no | Orb | 854 | 0.11 | 34.8 | -0.3 | 3.9 | 30 | 940 | 0.027 | 14.46 | 0.664 | 1.083 | 0.050 | 0.96798 | no |
| `2051443819881554304` | no | Orb | 503 | 0.39 | 33.4 | 5.6 | 3.7 | 55 | 844 | 0.062 | 13.82 | 0.657 | 1.066 | 0.090 | 0.93924 | no |
| `4845954532847810560` | no | Orb | 543 | 0.37 | 33.3 | -0.2 | 3.1 | 53 | 923 | 0.020 | 14.48 | 0.658 | 1.069 | 0.065 | 0.81640 | no |
| `2132670722412809856` | no | Orb | 398 | 0.48 | 33.3 | 5.3 | 4.8 | 61 | 387 | 0.013 | 12.86 | 0.686 | 1.140 | 0.060 | 0.89820 | no |
| `4518143627905286656` | no | Orb | 547 | 0.08 | 30.0 | -0.5 | 2.0 | 39 | 960 | 0.274 | 14.97 | 0.686 | 1.138 | 0.091 | 0.99002 | no |
| `1503707843203401728` | no | Orb | 643 | 0.56 | 29.3 | 0.5 | 3.5 | 80 | 824 | 0.000 | 13.94 | 0.654 | 1.057 | 0.063 | 0.86439 | no |
| `548950254119058816` | no | Orb | 702 | 0.50 | 27.1 | -0.4 | 3.0 | 45 | 905 | 0.327 | 14.98 | 0.684 | 1.134 | 0.096 | 0.97134 | no |
| `4485210127905490304` | no | Orb | 812 | 0.62 | 27.1 | 0.6 | 4.1 | 65 | 584 | 0.177 | 14.39 | 0.701 | 1.178 | 0.076 | 0.89839 | no |
| `1260782225197846144` | no | Orb | 797 | 0.72 | 25.2 | 3.0 | 4.3 | 55 | 601 | 0.000 | 14.17 | 0.751 | 1.313 | 0.059 | 0.99978 | no |
| `2131963839514727296` | no | Orb | 611 | 0.58 | 24.3 | 2.0 | 2.1 | 36 | 1250 | 0.053 | 14.81 | 0.660 | 1.073 | 0.096 | 0.90027 | no |
| `1825002320759122304` | no | Orb | 712 | 0.49 | 23.0 | -0.8 | 2.7 | 34 | 1025 | 0.340 | 14.87 | 0.654 | 1.059 | 0.083 | 0.97422 | no |
| `4129918029309568896` | no | Orb | 777 | 0.46 | 19.6 | 1.1 | 3.6 | 58 | 572 | 0.513 | 14.65 | 0.661 | 1.074 | 0.073 | — | no |
| `5396859095870109952` | no | Orb | 609 | 0.59 | 18.9 | -0.9 | 2.5 | 51 | 839 | 0.061 | 14.52 | 0.679 | 1.122 | 0.089 | 0.81758 | no |
| `5832387540143968512` | no | Orb | 774 | 0.49 | 17.0 | 1.3 | 2.5 | 27 | 946 | 0.090 | 14.78 | 0.655 | 1.061 | 0.073 | 0.83417 | no |
| `780381641989821312` | no | Orb | 821 | 0.71 | 14.9 | 13.6 | 7.4 | 66 | 333 | 0.000 | 14.00 | 0.784 | 1.410 | 0.074 | 0.99948 | no |

### 3.4 Simon 2026 exclusion breakdown

**Update (#275/#281): resolved as a fixture/classifier defect; see §3.3.3 item 5.** With El-Badry's
own `M̃2/M̃1` supplied, the breakdown is 5 / 2 / 1 / 1 with `in_sample` 11. The pipeline's
membership for `3263804373319076480` (in) and `1864406790238257536` (out) matches Table 7. The
historical record below attributes the slot to the `a0`/AMRF chain. That attribution was wrong:
it compared against Simon's catalog masses, not El-Badry's.

**Update (#258): re-measured with the full-union membership that #234 left incomplete**, using the
real frozen maps on `1058e29`. `sample_ids` was the whole El-Badry 2026 survivor set (1565 entries,
astrometric ∪ spectroscopic, §3.3). Result: **5 / 2 / 1 / 0 (+1 unclassified), `in_sample` 11** —
identical to the pre-fix breakdown and to the same run with `E(B-V)=0`. Per-source detail for the
two swap sources below, after dereddening:

- `1864406790238257536` (north, Green2019): `E(B-V)=0.411` moves it from `mg_0 = 1.077,
  bp_rp_0 = 0.844` (evolved) to `mg_0 = −0.018, bp_rp_0 = 0.297`. It is now `main_sequence = True`,
  but the Janssens inversion at `MG,0 ≈ 0` gives `M̃1 = 3.88`, so `M̃2 = 2.13` and
  `M̃2/M̃1 = 0.55 < 1.2`. It still drops out of `primary_ns_bh`, now via the `m2_over_m1` cut instead
  of the evolved gate. The paper has `m2_over_m1 = 5.08`, which implies a far smaller `M̃1`. Either
  the paper's reddening toward this source is much lower than Bayestar19's 0.41 (× 0.884), or it
  treats `M̃1` differently at the bright end. This is a real per-source discrepancy to chase; it is
  not a tuning knob.
- `3263804373319076480` (north): `E(B-V)=0.088`, `mg_0 = 4.104`, `M̃1 = 1.17`, `M̃2 = 2.93`. It is
  still in the sample. As §3.4 already said, this one is the `a0`/AMRF chain (`M̃2` ≈4× the paper),
  not extinction.

So extinction alone does not close the Simon last slot. The remaining swap is the `a0`/AMRF chain
(#133 Q7 remainder) plus the bright-end `M̃1` behaviour for `1864406790238257536`.

Re-measured at `eb009d8` (post Wave −1, ahead of Wave A) via
`sample_diagnostics.run_simon2026_diagnostic`/`elbadry2026_selection.simon2026_exclusion_breakdown`,
fed the real `elbadry2026` `sample_selection` stage output (`surviving_source_ids`, 1088 rows — the
orphaned artifact `output/20260912-165745-c4aa127/sample_selection/19b58e641119bbd8.h5`, the closest
on-disk artifact to the `main` @ `c905575` baseline). Acceptance target is 5 / 2 / 1 / 1 for the four
exclusion reasons, plus 11 overlapping sources in sample (CONTINUATION_PLAN §8.9):

```
in_sample:                11   expected=—   n/a
sb1_fails_significance:    5   expected=5   yes
astrometric_f2_above_max:  2   expected=2   yes
fainter_than_g_limit:      1   expected=1   yes
fails_m2_over_m1:          0   expected=1   NO
unclassified:              1   expected=—   n/a
```

Unchanged from the previously recorded 5 / 2 / 1 / 0 (+1 unclassified) — issue #200's root-cause
investigation.

**Root cause found and it is (c): downstream of the extinction/`a0`/mass-ratio chain (#133), not a bug
in `classify_simon2026_row` / `simon2026_exclusion_breakdown` themselves.** Per-source detail (Table 1
values transcribed in `config/selections/external/simon2026_orbital.yaml`, our own re-derived values
from `enrich_elbadry2026_row` on the real `+enrich+mc10000` parent-cache rows):

- The **unclassified** source is `1864406790238257536` (`AstroSpectroSB1`, paper `m2_over_m1 = 5.083`
  — per the paper's own numbers this source should be a real detection, not `fails_m2_over_m1`). Our
  pipeline classifies it `main_sequence = False` (`mg_0 = 1.077`, `bp_rp_0 = 0.844`), so
  `paper_m1_from_mg` returns `NotApplicable("evolved")` and it never reaches the `m2_over_m1` cut at
  all — it drops out of `primary_ns_bh` for a **cut-not-applicable** reason `classify_simon2026_row`
  has no bucket for (Q10-shaped symptom, checked first per the issue — but the *cause* is not a Q10
  boolean-logic bug: given its inputs, `is_main_sequence` classifies correctly).
- The reason it has those inputs: `sample_selection.candidate_to_selection_row` aliases `mg_0` /
  `bp_rp_0` straight from `abs_g_mag` / `bp_rp` (**no dereddening**) whenever a dereddened value isn't
  already present upstream (`src/darkhunter_pop/sample_selection.py:1578-1581`) — the exact `mg_0` /
  extinction gap #133 already names for `primary_ns_bh` 42 ≠ 47.
- The **compensating** source is `3263804373319076480` (`AstroSpectroSB1`, paper `m2_over_m1 = 0.753`,
  `m1 = 0.97`, `m2_lower = 0.73` — should be `fails_m2_over_m1`). Our pipeline puts it `main_sequence =
  True` and computes `m1_tilde_msun = 1.10`, `m2_tilde_msun = 2.87` (`amrf = 1.11` via
  `photocenter_a0_from_thiele_innes`) — an `M̃2` about 4x the paper's `m2_lower`, so it clears
  `m2_over_m1_min = 1.2` and survives into the sample instead of being excluded. Same `a0`/AMRF chain
  as Q7 (nsstools vs. `thiele_innes_to_campbell`), not a new bug.
- These two sources are a like-for-like swap: our pipeline includes the one the paper excludes and
  excludes the one the paper includes, net `in_sample = 11` (right count, wrong two members) and one
  orphaned bucket. **No threshold was or should be touched** — `simon2026_exclusion_breakdown`'s cut
  values (`g_mag_faint_limit=15`, `goodness_of_fit_max=10`, `k1_significance_min=10`,
  `m2_over_m1_min=1.2`) are exactly the frozen `elbadry2026.yaml` values, and
  `test_simon2026_exclusion_breakdown_5_2_1_1` (a fixed-`in_sample`-fixture unit test, decoupled from
  the real pipeline) already asserts and passes 5/2/1/1 given the paper's own membership — the
  classification logic is correct; only the real pipeline's re-derived masses for these two sources
  are wrong, and that is the extinction/`a0` chain, #133's territory.

Per issue #200's own decision rule: **do not fix here.** This is filed against #133 (comment added,
same root cause, not a new issue) rather than fixed under #200's mass-ratio-path-bug branch. Once
#133 lands a real dereddened `mg_0`/`bp_rp_0` and the nsstools-vs-Thiele–Innes `a0` delta is resolved,
re-run this diagnostic — it should self-correct without touching `elbadry2026_selection.py`.

Because the El-Badry 2026 sample this is computed over is itself inflated (1088 against a published
227), the breakdown sits downstream of an already-failing gate and is not independently diagnostic yet.
Close `primary_ns_bh` and the spectroscopic branch first; re-check this afterwards.

Note for whoever re-runs it: pass `sample_ids` and let `run_simon2026_diagnostic` load its own Simon
table rows. Handing it selection-parent-cache rows raises `KeyError: 'm2_over_m1'` — those rows are a
different shape.

---

## 4. Known root causes (for fix agents)

| Symptom | Likely cause | Do / don't |
|---------|--------------|------------|
| Andrews 0 survivors | Missing `p_m2_above` (enrich-only cache) | Use `+enrich+mc10000` |
| Andrews 352 after `P(M2)` vs 106 | Our full-cov MC ≠ the paper's attrition (packing / floors / prefilters) | Escalate; don't edit 0.95 |
| `sub_chandrasekhar` 0 | Pre-reconciliation branch state, before `σ_M̃2` bound | Historical; resolved |
| `sub_chandrasekhar` 740 | Andrews `σ` aliased into `sigma_m2_astrometric_msun` | Fixed; on `main` since `03a452a` |
| `sub_chandrasekhar` 1265 (current, real maps; 861 undereddened) | **Not the `M̃2` chain** (#275, §3.3.3): it reproduces Table 7 to ≤0.06 % median. Every paper member has `M̃2/M̃1 ≥ 1.10` and `A ≥ 0.669`; ours are mostly `q < 1` companions of massive MS primaries that the written criteria admit | PI decision on an unstated criterion and/or the `σ_M̃2` method: **#284**. Do not add a cut or retune without it |
| `primary_ns_bh` 46 ≠ 47 (current; 42 before #258) | Reconciled against Table 7 (#275, §3.3.3 item 4): −2 sources MS in the paper but evolved on our CMD, +1 at the `M̃2 = 1.4002` boundary. All three are `E(B-V)`/colour boundary effects | #133; not the `a0`/AMRF chain. Do not retune |
| Spectro 123 ≠ 151 | **Resolved by #258**: 151 exact with the real maps. The binding already matched spec (#234) | Routes 132/30/11 vs 136/30/15 remain — §3.3.2 |
| Simon `fails_m2_over_m1` 0 ≠ 1 | **Resolved (#281):** the classifier used Simon's catalog ratio and the unit fixture had `1864406790238257536` / `3263804373319076480` swapped relative to Table 7. The pipeline's membership already matched the paper | Pass El-Badry `M̃2/M̃1` (`elbadry_m2_over_m1_by_source`); stage wiring landed in #285 (real run 5 / 2 / 1 / 1) |

**Column ownership** (strict — violating this is what produced the 740):

- Andrews owns `p_m2_above`, `m2_msun`, `sigma_m2_msun` / `m2_msun_error` at fixed **M1 = 1.0**.
- El-Badry 2026 owns `m1_tilde_msun`, `m2_tilde_msun`, `sigma_m2_astrometric_msun` at fixed
  **Janssens M̃1**. Never alias Andrews' σ into it.

---

## 5. Minimal re-check script sketch

```python
from pathlib import Path
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.sample_selection import SampleSelectionRegistry, _read_selection_parent_cache

cache = Path(
    "data/dr3/gaia_snapshots/"
    "20260826T234425Z_3d3f740b080c+enrich+mc10000/selection_parent_rows.h5"
)
rows = _read_selection_parent_cache(cache)
reg = SampleSelectionRegistry(load_config())
orb = [r for r in rows if r.get("nss_solution_type") == "Orbital"]
a = reg.selection("andrews2022").evaluate(orb)
membership = {"andrews2022": frozenset(a.surviving_source_ids)}
# … andrews2022_modified, mode_divergence, elbadry2024, elbadry2026(rows, membership=membership) …
# Compare each CutAttrition's n_out against expected_n_after, and the branch / subsample
# survivor counts against the tables in §3.
```

Use `.venv/bin/python` only. Prefer full permissions if HDF5/pytest segfaults in the sandbox — that is
a laptop sandbox artifact, not a pipeline bug. Expect ~25 min wall clock and ~7 GiB resident for a full
sweep including El-Badry 2026.

---

## 6. Out of scope for this hand-off

- RV / SED live integration (landed in Wave 2, PRs #136–#138).
- Retuning frozen selection YAML numbers. Ever, without escalation.
- Re-fetching NSS enrichment. The job is COMPLETED unless `nss_enrichment/meta.yaml` is missing.
- Rebuilding the `+enrich+mc10000` cache. `scripts/attach_mc_to_selection_cache.py` is on `main`
  (landed via PR #150 / issue #141, merge SHA `03a452a`) and the cache it builds already exists.

When a §3 gate goes green — or an escalation is explicitly accepted — rewrite that row **together with
the commit SHA it was measured at**. A number in this document without a SHA beside it is not evidence.
