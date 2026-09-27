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
extinction / `a0` / nsstools, Q7), **#258** (extinction root cause confirmed — dead `ExtinctionSpec`,
never wired up — real fix blocked on missing Green2019/Lallement2019 map data and laptop disk space;
see §3.3.1).

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
5. `σ_M̃2` = lazy full-covariance NSS MC at the `m2_error` cut with **fixed** Janssens `M̃1`
   (`propagate_fit_uncertainty: false`, CONTINUATION_PLAN §15 Q12). Provenance tag:
   `_sigma_m2_astrometric_provenance == "elbadry2026_m1_tilde_fixed"`.
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
explanation that resolves the gate.

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

| Check | Target | Previous (`5725e54`) | **`main` @ `c905575`** | Gate |
|-------|--------|----------------------|------------------------|------|
| Published union | 227 | inflated (astro-dominated) | **1085** | FAIL |
| Astrometric branch union | 76 | 913 | **913** | FAIL |
| Spectroscopic branch | 151 | 123 | **123** | FAIL — no code bug found; binding matches spec exactly (§3.3.2). Same unfixed extinction mechanism as `primary_ns_bh` (#133/#258) — Combined19 diagnostic (non-landed) closes the gap: 123→154 |
| `primary_ns_bh` | 47 | 42 | **42** | FAIL — **#133**, extinction / `a0` (nsstools vs Thiele–Innes) |
| `elbadry2023_table_e1` | 5 | 5 | **5** | **OK** — exact match holds |
| `andrews2022_import` | 16 | 19 (`c905575`/`63c6f53`, fixed-`M1=1.0` Andrews membership, N=33) | **55** (re-measured `main` @ `a9397ba`, #270, against the FLAME/uniform-draw `+enrich+mc10000` cache, `mtime` 2026-09-27 02:18:53; Andrews membership N=63) — equals §3.1.2's Q9 count exactly (same source-ID set) — see §3.3.0 | FAIL until Andrews and Q9 close |
| `sub_chandrasekhar` | 22 | 861 | **861** | FAIL — see the waterfall below |
| Spectro routes (MS min / high `f_m` / both) | 136 / 30 / 15 | 98 / 30 / 5 | **98 / 30 / 5** | FAIL — Combined19 diagnostic (non-landed, §3.3.2): 141 / 30 / 17 |
| Simon exclusion breakdown | 5 / 2 / 1 / 1 | 5 / 2 / 1 / 0 (+1 unclassified) | **5 / 2 / 1 / 0** (+1 unclassified) | FAIL last slot — diagnosed (#133-linked), not fixed — §3.4. Re-check attempted under #234 with `primary_ns_bh`-only membership (incomplete methodology — see §3.3.2); full-union re-measurement not completed this session |

`primary_ns_bh` attrition:

```
168065 → 137696 (main_sequence) → 499 (m2_floor) → 85 (m2_over_m1)
       → 77 (goodness_of_fit) → 55 (period) → 42 (g_mag)        [target 47]
```

`sub_chandrasekhar` attrition:

```
168065 → 137696 (main_sequence) → 1908 (m2_range 1.05–1.40)
       → 1161 (m2_error σ ≤ 0.105; 735 fail, 12 missing σ) → 868 (period ≤ 900 d)
       → 861 (G < 15)                                            [target 22]
```

Spectroscopic branch: `181534 → 133642 (k1_significance) → 123 (mass_route, 84090 N/A)`.

The diagnosis is unchanged, and worth restating because it determines where Wave A should look: fixing
the σ binding removed the Andrews pollution but did **not** recover N = 22. The mass window is already
**1908** wide before the σ cut is applied at all, so the σ cut is second-order. Escalate the
point-estimate `M̃2` / `a0` / extinction / main-sequence-CMD chain — **not** the 0.105 threshold.

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
counted unique sources.

### 3.3.1 Extinction chain audit (#232, #258) — root cause confirmed, fix blocked

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

### 3.4 Simon 2026 exclusion breakdown

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
| `sub_chandrasekhar` 861 (current) | Too many systems with `M̃2` ∈ [1.05, 1.40]; the σ cut is secondary | **Not extinction** (#258: Combined19 diagnostic makes this worse, 861→1399) — look at `a0`/AMRF or #237's open multi-solution question instead |
| `primary_ns_bh` 42 ≠ 47 | **Confirmed**: `ExtinctionSpec` parsed but never applied — `mg_0`/`bp_rp_0` are raw, un-dereddened `abs_g_mag`/`bp_rp` (dead code, not the `a0` method — Q7 already ruled that out). Combined19 diagnostic closes ~80% of the gap (42→46) | #133, #258; real fix needs Green2019/Lallement2019 map data, blocked on disk space — §3.3.1 |
| Spectro 123 ≠ 151 | **Confirmed (#234)**: binding matches spec exactly (K1 sig + mass_route both bit-for-bit `CONTINUATION_PLAN.md` §8.4); the gap is the same undereddened `mg_0`/`bp_rp_0` → `main_sequence` chain as `primary_ns_bh`. Combined19 diagnostic closes it fully (123→154, target 151) | Same root cause as #133/#258; not a spectroscopic-branch-specific bug — §3.3.2 |
| Simon `fails_m2_over_m1` 0 ≠ 1 | Confirmed (#200): two sources swap via `mg_0`/`a0` chain — `1864406790238257536` wrongly `evolved` (unextincted `mg_0`), `3263804373319076480` wrongly clears `m2_over_m1` (AMRF `M̃2` ≈4x paper) | Same root cause as #133; fix there, not by editing `elbadry2026_selection.py` |

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
