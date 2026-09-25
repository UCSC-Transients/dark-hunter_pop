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
extinction / `a0` / nsstools, Q7).

---

## 1. Local data artifacts (not in git)

| Artifact | Path | Notes |
|----------|------|--------|
| Photometry snapshot | `data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/` | Uncut literature parent |
| NSS enrichment | `data/dr3/gaia_snapshots/nss_enrichment/` | `corr_vec`, K1, significance; job **COMPLETED** — do not re-`--poll-job` |
| Enrich-only parent cache | `…/20260826T234425Z_3d3f740b080c+enrich/selection_parent_rows.h5` | 691 MB; no Andrews `p_m2_above` |
| Enrich+Andrews-MC cache | `…/20260826T234425Z_3d3f740b080c+enrich+mc10000/selection_parent_rows.h5` | 734 MB; 443,211 rows; Orbital MC for Andrews `P(M2)` / SNR. **This is the one §3 is measured on.** |

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
| Spectroscopic branch | 151 | 123 | **123** | FAIL — after the K1 bind; `mass_route` / MS / `m2_min` |
| `primary_ns_bh` | 47 | 42 | **42** | FAIL — **#133**, extinction / `a0` (nsstools vs Thiele–Innes) |
| `elbadry2023_table_e1` | 5 | 5 | **5** | **OK** — exact match holds |
| `andrews2022_import` | 16 | 19 (with Andrews membership) | **19** (re-confirmed `63c6f53`, #198) | FAIL until Andrews and Q9 close |
| `sub_chandrasekhar` | 22 | 861 | **861** | FAIL — see the waterfall below |
| Spectro routes (MS min / high `f_m` / both) | 136 / 30 / 15 | 98 / 30 / 5 | **98 / 30 / 5** | FAIL |
| Simon exclusion breakdown | 5 / 2 / 1 / 1 | 5 / 2 / 1 / 0 (+1 unclassified) | **5 / 2 / 1 / 0** (+1 unclassified) | FAIL last slot — diagnosed (#133-linked), not fixed — §3.4 |

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
| `sub_chandrasekhar` 861 (current) | Too many systems with `M̃2` ∈ [1.05, 1.40]; the σ cut is secondary | Fix the `a0` / extinction / AMRF chain |
| `primary_ns_bh` 42 ≠ 47 | Extinction maps / `mg_0` / `a0` method (Q7 nsstools) | #133; **measure** the nsstools delta before choosing |
| Spectro 123 ≠ 151 | K1 is fine; downstream `mass_route` / MS / `m2_min` | Binding, not the Orbital MC |
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
