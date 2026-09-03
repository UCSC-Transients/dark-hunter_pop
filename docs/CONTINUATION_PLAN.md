# dark-hunter_pop — Continuation Plan (Phase 8+)

Companion to `ARCHITECTURE.md` (authoritative technical specification) and `ORCHESTRATION_PLAN.md`
(subagent roster, phasing, git/PR workflow). Those two documents remain authoritative for anything
they already cover; this document extends them and does not restate them.

**Status: specification only.** Nothing in §5–§12 is implemented. This document is the deliverable
of the spec pass; implementation happens in Phase 8 under the kickoff prompts in §14.

## 1. Purpose and current status

Phases 0–7 of `ORCHESTRATION_PLAN.md` are complete: the pipeline runs end-to-end from
`data_acquisition` through `inference` and `diagnostics` on the DR3 NSS catalog. What is missing is
the piece that makes the mass function *debiased with respect to the literature samples we compare
against*: each published compact-object candidate sample (Andrews et al. 2022, El-Badry et al. 2024,
El-Badry et al. 2026) applied its own idiosyncratic cut chain to the parent NSS catalog, and those
cut chains are themselves selection functions that must be forward-modeled, not ignored.

Phase 8 adds:

1. A **per-sample selection-function layer** — a registry of named literature samples, each with a
   frozen, versioned, reproducible cut chain (§4–§9).
2. The **12×12 astrometric covariance ingestion** that Andrews' Monte Carlo mass-function
   propagation requires (§10).
3. **Spectroscopic (SB1) mass-function support**, the pipeline's first non-astrometric mass path,
   required by El-Badry 2026's second parent branch (§8.4).
4. The **config/preset machinery** to enable or disable matching each published sample
   independently (§12).

## 2. Subagent dispatch: model tier × effort level

`ORCHESTRATION_PLAN.md` §1 established manually-launched Agents Window sessions with a hand-picked
model per session. That remains the primary mechanism for long, judgment-heavy roles. Phase 8 adds
an explicit **effort level** orthogonal to model tier, because several Phase 8 tasks are
high-precision but low-judgment (transcribing a published cut chain into a frozen config file) and
should not burn a top-tier reasoning budget.

### 2.1 Model tiers → slugs

| Tier | Slug | Use for |
|---|---|---|
| Top | `claude-opus-5-thinking-medium` | Statistical design, selection-function correctness, likelihood changes |
| Top (alt) | `gpt-5.6-sol-medium` | Second opinion on statistics; cross-checking a Top-tier result |
| Mid | `claude-4-sonnet` | Well-specified implementation against a frozen interface |
| Mid (fast) | `cursor-grok-4.6-high-fast` | Bulk mechanical implementation with clear acceptance tests |
| Light | `gpt-5-mini`, `gemini-3.5-flash` | Transcription, doc sync, fixture generation, table extraction |

Use `inherit` only when the parent session's model is already the intended tier.

### 2.2 Effort levels

Effort is a contract about **how much the subagent is allowed to decide** and **how much verification
it must produce** — not a model setting.

| Effort | Subagent may decide | Verification required before PR |
|---|---|---|
| **Deep** | Statistical formulation, interface shape, what diagnostics are needed | Full required pytest + a written justification of the statistical choice + at least one diagnostic figure/report |
| **Standard** | Implementation details within a frozen interface | Full required pytest + unit tests for every new branch |
| **Light** | Nothing structural; transcribes a spec into code/config | Full required pytest + exact-value round-trip test against the published numbers |

A **Light**-effort subagent that discovers it needs a structural decision must stop and escalate
rather than deciding — this is the same "flag it as a suggested follow-up" rule as
`dark-hunter-pop-workflow` §8.

### 2.3 Assignment convention

Every Phase 8 kickoff prompt in §14 carries an explicit `Model:` and `Effort:` line. When
dispatching via the Task tool rather than the Agents Window, pass the slug from §2.1 directly; when
dispatching via the Agents Window, hand-pick it in the session picker.

## 3. Phase 8 roster

Extends the `ORCHESTRATION_PLAN.md` §4 roster; numbering continues from #16.

| # | Subagent | Depends on | Model tier | Effort | Rationale |
|---|---|---|---|---|---|
| 17 | **Sample-selection framework** (`sample_selection.py`, registry, `SampleSelection` interface, dual-path mode switch, config schema) | — (first, blocking) | Top | Deep | Every per-sample module builds against this interface; the reproduction/forward-model split is a contract, not an implementation detail. |
| 18 | **NSS covariance ingestion** (`corr_vec`/`bit_index` → 12×12 matrices in `data_acquisition`) | — (parallel with #17) | Mid | Standard | Well-specified ETL against a documented Gaia data model; correctness is checkable against Gaia's published matrix layout. |
| 19 | **Monte Carlo mass-function propagation** (1e4 draws, per-system posterior over `m_f` and `M2`) | #17, #18 | Top | Deep | Sampling from a rank-deficient / near-singular covariance needs real care; downstream probability cuts depend on it being right. |
| 20 | **Andrews et al. (2022) selection** (`config/selections/andrews2022.yaml` + module) | #17, #19 | Mid | Light | Cut chain is fully specified in §6; success criterion is reproducing N = 24 exactly. |
| 21 | **El-Badry et al. (2024) selection** (`config/selections/elbadry2024.yaml` + module) | #17, #19 | Top | Standard | Outcome-dependent criteria (§7.3) are a genuine statistical hazard, not a transcription job. |
| 22 | **Acceleration/jerk sample selection** (`config/selections/accel_jerk.yaml` + module) | #17 | Mid | Standard | Aggregate-level matching only (`ARCHITECTURE.md` §4); no per-system orbital solution to reproduce. |
| 23 | **Selection-function integration into `inference`** (per-sample terms in the Poisson rate) | #20, #21, #22 | Top | Deep | Changes the core likelihood; the multi-sample overlap/double-counting question is the hard part. |
| 24 | **Sample-reproduction diagnostics** (published-N waterfall per sample, cut-by-cut attrition) | #20, #21, #22 | Mid | Standard | Well-specified reporting; this is the artifact that proves the reproduction path works. |
| 25 | **El-Badry et al. (2026) selection** (`config/selections/elbadry2026.yaml` + module; two parent branches) | #17, #19, #20, #26 | Top | Deep | Two parallel branches (astrometric + SB1) with different mass-function machinery; the SB1 branch is new to this pipeline and subsample 3 imports Andrews' evaluated membership (§8.7). |
| 26 | **SB1 spectroscopic mass-function support** (`SB1`/`SB1C` ingestion, `K1`/`σ_K1`/`e`, `f_m` primitive) | #18 | Mid | Standard | Well-specified ETL plus one closed-form statistic; scope decision in §15 Q6 gates whether it feeds inference or stays reproduction-only. |

Review/Integration (#15) continues, merging `config/fragments/` into `config/config.yaml` at each
checkpoint as before.

**Sequencing.** #17 and #18 first and in parallel (both blocking). Then #19 and #26 in parallel.
Then #20–#22 in parallel (≤3 concurrent per `ORCHESTRATION_PLAN.md` §1). Then #25, which needs
#20 merged because it imports Andrews' evaluated membership (§8.7). Then #23 and #24, which need
every sample landed.

## 4. Sample selection functions — architecture

### 4.1 Why each sample needs its own selection function

`ARCHITECTURE.md` §4 already models two selection effects: `selection_function_astrometric` (does
Gaia produce an acceptable orbital solution?) and `selection_function_followup` (does the system get
RV follow-up?). Neither captures a third effect: **a published sample is a specific, hand-built
subset of the NSS catalog, defined by a cut chain that is a strong and non-obvious function of
mass, period, magnitude, and goodness-of-fit.**

Andrews' 24 systems, El-Badry 2024's 21 systems, and El-Badry 2026's 227 candidates are not random
draws from the compact-object population; they are draws filtered through cuts like "95% probability
that M2 > 1.4 M☉", "G < 15", and "F2 < 10", which sculpt the recovered mass function directly.
Comparing our dN/dM to theirs, or using their systems as anchors, requires forward-modeling those
cut chains through the same mock pipeline that already handles the astrometric and follow-up
selection.

### 4.2 Dual-path requirement

**Every literature sample gets two execution paths**, selected per sample, not globally.

| Path | Purpose | Primary mass | Success criterion |
|---|---|---|---|
| **`reproduction`** | Verify we can reproduce the published sample exactly | Whatever the paper assumed (frozen in the per-sample file) | Recovered N and source-ID set match the published table |
| **`forward_model`** | Statistically correct treatment for new analysis and DR4 | Our own MSC/gspphot → TAG10 masses with full uncertainty (`ARCHITECTURE.md` §4 `mass_derivation_bulk`) | Cut chain applied to the mock population reproduces the real sample's summary statistics |

The reproduction path exists so that a change to our mass derivation, covariance handling, or
Thiele-Innes conversion that silently breaks agreement with a published sample fails loudly. It is
a regression test with scientific content. It is **not** used for inference.

The forward-model path is the one that feeds `inference`. It applies the *same geometric and
quality cuts* as the reproduction path but replaces the paper's mass assumption with our own
mass posterior, and is applied identically to real candidates and to mock realizations.

### 4.3 Per-sample mass assumption ownership

This is the specific reason the mode switch must be per-sample rather than global:

- **Andrews et al. (2022)** assumes a **fixed M1 = 1.0 M☉** for the first-cut mass function, then
  refines later in their analysis.
- **El-Badry et al. (2024)** uses **IsocLum masses from `gaiadr3.binary_masses`** (PARSEC
  isochrones with a near-solar-metallicity prior) inherited from Shahaf et al. (2023b), and refines
  them with SED fitting.

These are different assumptions producing different M2 for the same source. A single global
`primary_mass_method` key would make it impossible to reproduce both papers in one run. Therefore:

> **Rule.** Each per-sample selection file owns its own `primary_mass` block. There is no global
> primary-mass switch governing sample reproduction. The forward-model path ignores the per-sample
> `primary_mass` block and uses the pipeline's own `mass_derivation` output.

### 4.4 Registry and file layout

```
config/
├── config.yaml                       # merged canonical config; holds only the on/off switches
├── fragments/
│   └── sample_selection.yaml         # dev-time fragment for the registry block
└── selections/                       # NEW: one frozen file per published sample
    ├── andrews2022.yaml
    ├── elbadry2024.yaml
    ├── elbadry2026.yaml
    ├── accel_jerk.yaml
    └── external/                     # frozen published tables a selection depends on
        └── elbadry2023_table_e1.yaml
```

Each file under `config/selections/` is **versioned and frozen**: once a sample's reproduction path
matches the published N, its numeric thresholds are not edited without a `schema_version` bump and
a note in the file's `provenance` block. This is what keeps the reproduction path reproducible.

Each file carries, at minimum:

```yaml
schema_version: 1
name: <sample key>
provenance:
  reference: <author, year, bibcode/arXiv>
  section: <where in the paper the cut chain is described>
  published_n: <integer>
  data_release: dr3
  frozen_on: <ISO date>
mode: reproduction            # or forward_model; per-sample, see §4.2
primary_mass: {...}           # per-sample; see §4.3
parent_query:                 # the ADQL that defines the parent catalog (§4.5)
  adql: |
    ...
  expected_parent_n: <integer>
cuts: [...]                   # ordered post-query cuts (§4.6)
exclusions: [...]             # explicitly removed source_ids with a stated reason
monte_carlo: {...}            # see §11
```

### 4.5 Parent query as SQL

The parent catalog for each sample is defined by an **ADQL query against the Gaia archive**, stored
literally in the per-sample file. This matches how `data_acquisition` already snapshots its own
literal ADQL text (`ARCHITECTURE.md` §4), and makes the parent-sample definition auditable against
the paper without reading Python.

The query defines only what can be expressed in the archive (solution type, catalog membership,
column-level bounds). Everything requiring derived quantities — the astrometric mass function, the
Monte Carlo probability cut, the CMD cut on absolute magnitudes — is a **post-query cut** in the
`cuts` list, applied in order, each recording its own attrition count for the waterfall diagnostic
(§13).

### 4.6 Cut chain semantics

`cuts` is an **ordered list**; order is significant and must match the paper. Each entry:

```yaml
- id: <stable identifier used in the attrition waterfall>
  kind: <column | derived | probability | exclusion>
  expression: <declarative predicate; no inline Python>
  applies_to: [reproduction, forward_model]   # a cut may be reproduction-only
  expected_n_after: <integer|null>            # null when the paper does not report it
```

`expected_n_after` is what makes the reproduction path self-checking: the waterfall diagnostic
asserts each stage against the published number where one exists.

### 4.7 Where this enters the likelihood

`ARCHITECTURE.md` §4 `inference` defines
`rate(θ) = population_model(θ) × astrometric_selection_function × followup_selection_function`.

Phase 8 extends this to, for a sample `s`:

```
rate_s(θ) = population_model(θ)
          × astrometric_selection_function
          × followup_selection_function
          × sample_selection_function_s
```

`sample_selection_function_s` is the probability that a system with given true properties survives
sample `s`'s cut chain, estimated by pushing mock realizations through the identical chain.

**Open architectural question (§15, Q1):** whether the samples are analyzed as separate Poisson
processes with separate rate functions, or unified with an inclusion-indicator per sample. They
overlap (El-Badry drew candidates from Andrews' sample), so naive summation double-counts.

## 5. Shared astrometric formulae

These are shared primitives, not per-sample. They belong in `physics_utils.py` alongside the
existing `thiele_innes_to_campbell` and `astrometric_mass_function` helpers. **Not landed in this
pass** — Phase 8 #17/#19 implement them.

### 5.1 Photocenter semi-major axis from Thiele-Innes elements

Following Halbwachs et al. (2022), from the NSS `a_thiele_innes`, `b_thiele_innes`,
`f_thiele_innes`, `g_thiele_innes` (all in mas):

```
u  = (A^2 + B^2 + F^2 + G^2) / 2
v  = A*G - B*F
a0 = sqrt( u + sqrt(u^2 - v^2) )
```

`a0` is the angular semi-major axis of the photocenter orbit, in mas.

### 5.2 Astrometric mass function

Using the parallax ϖ to convert `a0` to a physical separation, and Kepler's third law with the
observed orbital period:

```
m_f = (a0 / 1 mas)^3 * (varpi / 1 mas)^-3 * (P_orb / 1 yr)^-2   [M_sun]
```

### 5.3 Dark-companion limit

Assuming the companion contributes no light, so the photocenter follows the luminous star:

```
m_f = M2^3 / (M1 + M2)^2
```

Index 1 is the observed (most luminous) star; index 2 the companion. Note this differs from the
spectroscopic mass function by a factor `sin^3 i`.

### 5.4 Luminous-companion generalization

When the companion contributes flux (Halbwachs et al. 2022):

```
m_f = (F1*M2 - F2*M1)^3 / [ (F1 + F2)^3 * (M1 + M2)^2 ]
```

With flux ratio `F = F2/F1` and mass ratio `q = M2/M1`:

```
m_f / M1 = (q - F)^3 / [ (1 + F)^3 * (1 + q)^2 ]
```

For fixed `m_f/M1`, the implied companion mass rises steeply as the companion's flux contribution
grows: more companion light means the observed photocenter wobble implies a larger true orbital
separation, hence a more massive companion at fixed period. **The dark-companion assumption is
therefore the conservative one** (it yields the minimum companion mass), which is why both Andrews
and El-Badry adopt it for selection.

> Transcription note: the source text for §5.4 renders the denominator of the `m_f/M1` expression
> as `(1 + F1)^3`; this is a typo for `(1 + F)^3`, as required for dimensional consistency with the
> line above. The corrected form is specified here.

## 6. Sample: Andrews et al. (2022)

**File:** `config/selections/andrews2022.yaml` · **Roster:** #20 · **Model:** Mid · **Effort:** Light
· **Published N: 24**

### 6.1 Parent query

Start from the DR3 non-single-star catalog restricted to binaries **detected using astrometry
only** — explicitly avoiding spectroscopic and photometric binaries, which are treated elsewhere in
the literature (El-Badry & Rix 2022; Gomel et al. 2022; Mazeh et al. 2022).

- **Expected parent N: 134,598.**
- The ADQL restricts `nss_solution_type` to the astrometry-only solution types. The exact set of
  type strings that reproduces 134,598 must be verified against the archive and then frozen in the
  file; `AstroSpectroSB1` is excluded because it is a joint astrometric+spectroscopic solution.

### 6.2 Derived quantities

1. Compute `a0` from the Thiele-Innes elements (§5.1).
2. Compute the astrometric mass function `m_f` (§5.2).
3. Assume the **dark-companion** relation (§5.3) and a **fixed primary mass M1 = 1.0 M☉** to solve
   for M2.

```yaml
primary_mass:
  method: fixed
  value_msun: 1.0
  rationale: "Andrews et al. (2022) first-cut assumption; refined later in their analysis."
```

Under `mode: forward_model`, this block is ignored and the pipeline's own TAG10/MSC mass posterior
is used instead (§4.3).

### 6.3 Uncertainty propagation

10⁴ Monte Carlo random draws of the **full 12×12 covariance matrix** of each binary's astrometric
solution, propagated through §5.1–§5.3 to give a per-system posterior on M2. See §11.

### 6.4 Cut chain (ordered)

| # | `id` | Cut | Expected N after |
|---|---|---|---|
| 0 | `parent` | Astrometry-only DR3 NSS solutions | 134,598 |
| 1 | `m2_probability` | ≥ 95% posterior probability that M2 > **1.4 M☉** | 106 |
| 2 | `goodness_of_fit` | `goodness_of_fit` (F2) **< 5** | — |
| 3 | `m2_snr` | **M2 / σ_M2 > 3** | — |
| 4 | `giant_reject_logg` | Reject `log g < 3.6` **when Apsis `log g` is available** | — |
| 5 | `giant_reject_cmd` | Reject unless `G > 3.14 (BP − RP) − 0.43` | — |
| 6 | `harmonic_exclusion` | Remove `Gaia DR3 4373465352415301632` | **24** |

Notes that must be carried in the file as comments:

- The 1.4 M☉ limiting mass is chosen specifically to avoid contamination by massive white dwarfs.
- Cuts 4–5 remove giant donors. This is acknowledged in the paper as arguably over-restrictive
  (giants with NS/BH companions do exist), but large giant luminosities can hide even massive
  non-degenerate companions.
- The CMD cut uses **absolute** magnitudes derived from apparent magnitudes and parallax, with **no
  extinction correction applied** — this must be reproduced as-published, including the omission.
- The excluded source has `m_f ≈ 11.6 M☉` and `P_orb = 186 d`, roughly three times Gaia's scanning
  period; Halbwachs et al. (2022) identify large-`m_f` systems at harmonics of the scanning law as
  probable contaminants.

### 6.5 Config sketch

```yaml
schema_version: 1
name: andrews2022
provenance:
  reference: "Andrews et al. (2022)"
  section: "Sample selection"
  published_n: 24
  data_release: dr3
mode: reproduction
primary_mass:
  method: fixed
  value_msun: 1.0
parent_query:
  adql: |
    SELECT ...
    FROM gaiadr3.nss_two_body_orbit AS nss
    WHERE nss.nss_solution_type IN ( ... astrometry-only types ... )
  expected_parent_n: 134598
monte_carlo:
  n_draws: 10000
  covariance: full_12x12
cuts:
  - id: m2_probability
    kind: probability
    expression: "P(M2 > m2_threshold_msun) >= m2_probability_min"
    m2_threshold_msun: 1.4
    m2_probability_min: 0.95
    expected_n_after: 106
  - id: goodness_of_fit
    kind: column
    expression: "goodness_of_fit < goodness_of_fit_max"
    goodness_of_fit_max: 5.0
    expected_n_after: null
  - id: m2_snr
    kind: derived
    expression: "m2_msun / m2_msun_error > m2_snr_min"
    m2_snr_min: 3.0
    expected_n_after: null
  - id: giant_reject_logg
    kind: derived
    expression: "logg_apsis is null or logg_apsis >= logg_dwarf_min"
    logg_dwarf_min: 3.6
    expected_n_after: null
  - id: giant_reject_cmd
    kind: derived
    expression: "abs_g_mag > cmd_slope * (bp_rp) + cmd_intercept"
    cmd_slope: 3.14
    cmd_intercept: -0.43
    extinction_corrected: false
    expected_n_after: null
exclusions:
  - source_id: 4373465352415301632
    reason: "m_f ~ 11.6 Msun at P_orb = 186 d ~ 3x Gaia scanning period; probable scanning-law harmonic contaminant (Halbwachs et al. 2022)."
    expected_n_after: 24
```

Every numeric threshold above lives in this file, never inline in Python
(`dark-hunter-pop-workflow` §1).

## 7. Sample: El-Badry et al. (2024)

**File:** `config/selections/elbadry2024.yaml` · **Roster:** #21 · **Model:** Top · **Effort:**
Standard · **Published N: 21** · **Source:** arXiv:2405.00089v2 §2

### 7.1 Parent query and triage

The parent is Gaia DR3 astrometric binary solutions of types **`Orbital`** and
**`AstroSpectroSB1`** — note this **differs from Andrews**, who excludes the joint
astrometric+spectroscopic type.

Selection follows the astrometric **"triage" algorithm** of Shahaf et al. (2019): identify sources
whose astrometric orbit is so large, given the orbital period, that it cannot be explained by any
luminous-star companion, nor by a companion that is itself a close binary of two luminous stars.
Shahaf et al. (2023b) applied this to DR3, producing a catalog of **177 candidates** whose secondary
must be a WD, NS, or BH.

Two additions beyond that catalog:

- Astrometrically-selected NS candidates whose orbital periods **slightly exceed 1000 days**, which
  fall outside the Shahaf et al. (2023b) period cut.
- Candidates listed in other DR3 compact-object samples, **including Andrews et al. (2022)**.

> **Implementation note.** The Shahaf triage is a published algorithm plus a published catalog. The
> per-sample file should support both routes and declare which is authoritative for the frozen
> reproduction: (a) cross-match against the ingested Shahaf et al. (2023b) catalog as an external
> comparison catalog, or (b) reimplement the AMRF triage. Route (a) is recommended for the
> reproduction path; route (b) is required eventually for the forward-model path and for DR4, since
> a published catalog cannot be applied to mock realizations. Flagged as §15 Q2.

### 7.2 Primary mass assumption

Shahaf et al. (2023b) — and therefore this sample's parent — use the **IsocLum** mass estimates from
`gaiadr3.binary_masses`, inferred by comparing extinction-corrected colors and absolute magnitudes
to a grid of PARSEC isochrones **with a prior that the metallicity is close to solar**.

```yaml
primary_mass:
  method: isoclum_binary_masses
  table: gaiadr3.binary_masses
  caveats:
    - "Overestimated for sub-solar [Fe/H]; underestimated for super-solar."
    - "Contingent on the extinction estimate and on single-star photometry."
```

The paper improves these later via SED fitting. This is a **different assumption from Andrews'
fixed M1 = 1.0 M☉**, and is precisely why §4.3 requires per-sample mass ownership.

### 7.3 Completeness criteria (§2.2 of the paper) — outcome-dependent

Among the Shahaf et al. (2023b) and Andrews et al. (2022) candidates, the published sample includes
all systems that:

| `id` | Criterion |
|---|---|
| `g_mag_limit` | (a) brighter than **G = 15** |
| `m2_joint_fit` | (b) best-fit **M2 > 1.25 M☉** from **joint fitting of astrometry and RVs** |
| `not_spurious` | (c) not found to have spurious solutions or significantly underestimated astrometric uncertainties through RV follow-up |
| `orbit_coverage` | (d) observed over **at least half an orbit** |

> **Statistical hazard — must be handled explicitly, not silently.** Criteria (b), (c), and (d) are
> **outcome-dependent**: they depend on the result of the RV follow-up, not on properties knowable
> before the follow-up decision was made. This is exactly the outcome-dependent selection risk
> flagged for roster #3 in `ORCHESTRATION_PLAN.md`.
>
> Consequences for the spec:
> - The **follow-up selection function** may only condition on pre-follow-up observables (G,
>   declination, target-list membership, parent-catalog membership).
> - Criteria (b)–(d) belong in the **inclusion operator of the Poisson likelihood** (§4.7), applied
>   identically to mock realizations, and must **not** be implemented as a data filter on real
>   systems that silently discards them.
> - Criterion (c) requires forward-modeling the **spurious-solution rate**: the paper reports that
>   **about a quarter of candidates with good astrometric quality flags turned out to be spurious**,
>   and notes this fraction is higher than for astrometric binaries generally, because spurious
>   solutions are over-represented where genuine binaries are rare. The 25% figure is a
>   candidate-population-specific rate and belongs in the per-sample file as a modeled parameter,
>   not as a global constant.

### 7.4 Additional context to encode

- Parent samples contain mainly sources **near the main sequence with M⋆ ≲ 1.3 M☉**. NS companions
  to evolved stars and more massive MS stars are not distinguishable from luminous stars or tight
  luminous binaries by astrometry alone, so they are absent by construction.
- Documented exclusions from the published sample: candidates whose RV coverage was insufficient to
  confirm the astrometric solution, and candidates **too faint (G > 15)** for the available
  instruments.
- The sample contains **no NSs below 1.25 M☉**, a regime where NSs do exist; distinguishing NSs from
  massive WDs becomes progressively harder at lower mass.
- Three of the 21 have `AstroSpectroSB1` solutions; the rest are `Orbital`.

### 7.5 Sample summary statistics (forward-model validation targets)

These are the published sample properties the forward-model path should reproduce, and are the
natural acceptance targets for §13:

- Luminous stars are solar-type main-sequence, **M1 ≈ (0.7 − 1.3) M☉**.
- Most binaries lie **within 1 kpc**.
- Orbital periods **100–1000 d**, distribution **peaking near 600 d**.
- **Deficit of systems with P_orb near 1 year**, from the degeneracy between such orbits and
  parallactic motion.
- Period distribution is fairly similar to that of all DR3 astrometric binaries — short-period
  orbits are smaller and resolvable only nearby; much longer orbits are poorly sampled within the
  ~1000-day DR3 observing window.

### 7.6 Extinction treatment

The paper's CMD uses extinction-corrected magnitudes, with a **declination-split dust map**:

```yaml
extinction:
  north:
    applies_when: "dec_deg > -30.0"
    map: green2019
  south:
    applies_when: "dec_deg <= -30.0"
    map: lallement2022
```

This differs from Andrews, who applies **no** extinction correction (§6.4). Each sample file owns
its own extinction policy; do not share one.

## 8. Sample: El-Badry et al. (2026)

**File:** `config/selections/elbadry2026.yaml` · **Roster:** #25 · **Model:** Top · **Effort:** Deep
· **Published N: 227** (76 astrometric + 151 spectroscopic) · **Source:** arXiv:2608.06453v1 §2

This sample is structurally unlike §6 and §7: it has **two parallel parent branches** — an
astrometric branch and a single-lined spectroscopic (SB1) branch — with different mass-function
machinery, different cut chains, and different contamination physics. Both are part of one named
sample because the paper selects, follows up, and reports them as one program.

**The initial 227-source sample is a purely catalog-level selection.** Every cut in §8.3 and §8.4 is
evaluable from DR3 catalog columns plus photometric priors, with no dependence on follow-up
outcomes. This is a materially better forward-modeling target than §7, whose completeness criteria
were outcome-dependent. The outcome-dependence in this paper enters only when modeling the
*follow-up subsets* rather than the initial sample (§8.5).

### 8.1 Parent samples and shared preprocessing

Both branches draw from `gaiadr3.nss_two_body_orbit`:

| Branch | `nss_solution_type` |
|---|---|
| Astrometric | `Orbital`, `AstroSpectroSB1` |
| Spectroscopic | `SB1`, `SB1C` |

The astrometric branch's solution types match §7 exactly. The SB1 branch is **new machinery for this
pipeline** — see §15 Q6.

Shared preprocessing applied before either branch's cuts:

**Extinction.** 3D dust maps as available in 2022, split by declination:

```yaml
extinction:
  north:
    applies_when: "dec_deg > -28.0"
    map: green2019
  south:
    applies_when: "dec_deg <= -28.0"
    map: lallement2019
  coefficients:
    e_bp_rp_over_e_bv: 1.33
    a_g_over_e_bv: 2.66
    valid_for_teff_k: 6000
```

> **Differs from §7.6.** El-Badry 2024 splits at **δ = −30°** and uses **Lallement et al. (2022)**
> in the south; this paper splits at **δ = −28°** and uses **Lallement et al. (2019)**. Both values
> are correct for their own paper. Each file owns its own extinction policy (§4.3); do not
> consolidate.

**Main-sequence classification.** A source is on the main sequence if either condition holds:

```
MG,0 > 4.5    or    MG,0 > -9.37 + 13.42 * (G_BP - G_RP)_0
```

where `(G_BP − G_RP)_0` and `MG,0` are the extinction-corrected color and absolute magnitude. This
`MS = true` flag gates several cuts in both branches and is the only route to a primary-mass
estimate (§8.2).

### 8.2 Primary-mass assumption — owned by this sample

For sources classified as main sequence, the primary mass `M̃1` comes from the **empirical
`MG,0`–mass relation compiled by Janssens et al. (2022, their Figure 3)**.

```yaml
primary_mass:
  method: janssens2022_mg_mass_relation
  reference: "Janssens et al. (2022), Figure 3"
  applies_only_when: "main_sequence == true"
  caveats:
    - "Photometric only; does not model each source's detailed evolutionary state."
    - "Does not model light contributions from possible luminous companions."
    - "Suitable for initial candidate selection, not for final masses."
```

The paper denotes these estimates `M̃1`, and companion masses derived from them `M̃2`, specifically
to mark them as selection-grade rather than science-grade; it replaces them with SED-fit masses in
its Section 4. **Our reproduction path must use `M̃1`, not our own masses**, or the cut chain will
not reproduce the published counts.

This is the **third distinct primary-mass assumption** across the three literature samples, and the
clearest justification for the §4.3 per-sample ownership rule:

| Sample | Primary mass | Defined for |
|---|---|---|
| Andrews et al. (2022) | Fixed `M1 = 1.0 M☉` | All sources |
| El-Badry et al. (2024) | IsocLum from `gaiadr3.binary_masses` (PARSEC, near-solar [Fe/H] prior) | Catalog coverage |
| **El-Badry et al. (2026)** | **Janssens et al. (2022) empirical `MG,0`–mass relation** | **Main-sequence sources only** |

The "main-sequence only" restriction is itself a selection effect: for evolved candidates no `M̃1`
exists, so every cut expressed in terms of `M̃1` or `M̃2` is undefined and the source can only enter
through a branch that does not require them (§8.3 subsample 2, or §8.4's `fm > 3 M☉` route).

Under `mode: forward_model` this whole block is ignored and the pipeline's TAG10/MSC mass posterior
is used, per §4.3.

### 8.3 Astrometric branch (paper §2.1)

**Derived quantities.** The photocenter semi-major axis `ã0` and its uncertainty `σ_ã0` are computed
from the best-fit Thiele-Innes parameters using the **`nsstools` package** (Halbwachs et al. 2023) —
see §15 Q7 on whether we vendor `nsstools` or use our own §5.1 conversion.

The selection statistic is the **astrometric mass ratio function** (AMRF; Shahaf et al. 2019), the
same quantity underlying the §7.1 triage:

```
A = (a0_tilde / varpi) * (M1_tilde / M_sun)^(-1/3) * (P_orb / yr)^(-2/3)
```

If the companion contributes negligible G-band light, this relates to the mass ratio `q = M2/M1` by

```
A = q / (1 + q)^(2/3)
```

which is **inverted numerically** for `q`, giving `M̃2 = q · M̃1`. For a given `M̃1` this is the
*minimum* companion mass consistent with the photocenter orbit; a luminous companion implies a
larger mass — the same conservatism argued in §5.4.

> **Identity worth encoding once.** The AMRF and the §5.2 astrometric mass function are the same
> quantity in different variables: `A^3 = m_f / M1`. Substituting the dark-companion relation
> `m_f = M2³/(M1+M2)²` recovers `A = q/(1+q)^(2/3)` exactly. Implement one primitive and derive the
> other; do not maintain two independent inversions that can drift apart.

**Subsample structure.** The astrometric branch is a **union of four subsamples**, each with its own
cut set. Order matters only for attrition accounting; membership is a union.

| Sub | `id` | Criteria | N | New |
|---|---|---|---|---|
| 1 | `primary_ns_bh` | `MS = true`; `M̃2 > 1.4 M☉`; `M̃2/M̃1 > 1.2`; `F2 < 10`; `P_orb < 1200 d`; `G < 15` | 47 | 47 |
| 2 | `elbadry2023_table_e1` | BH candidates from Table E1 of El-Badry et al. (2023a), restricted to `G < 15` | 5 | 3 |
| 3 | `andrews2022_import` | BH/NS candidates of Andrews et al. (2022) with `G < 15` | 16 | 4 |
| 4 | `sub_chandrasekhar` | `MS = true`; `1.05 ≤ M̃2/M☉ ≤ 1.40`; `σ_M̃2 ≤ 0.105 M☉`; `P_orb ≤ 900 d`; `G < 15` | 22 | 22 |
| — | `union` | Unique sources across subsamples 1–4 | — | **76** |

Reproduction arithmetic the waterfall must confirm: `47 + 3 + 4 + 22 = 76`.

Notes to carry in the file:

- The `G < 15` limit exists to make high-resolution spectroscopy feasible on 2m-class telescopes;
  **92% of sources with astrometric orbital solutions satisfy it.** That 92% figure is a useful
  sanity check on our own parent query.
- Subsample 2 applies **no goodness-of-fit cut and no main-sequence requirement**, targeting higher
  companion masses than Equation 4. Two of its five `G < 15` members are already in subsample 1.
- Subsample 3 imports Andrews' selection wholesale. The paper explicitly notes Andrews differs by
  neglecting extinction corrections, assuming `M̃1 = 1 M☉`, and imposing additional cuts on
  companion-mass uncertainty and `F2`. Twelve of its sixteen members are already covered above.
  See §8.7.
- Subsample 4 targets dark companions *below* the Chandrasekhar-adjacent threshold — massive WDs or
  low-mass NSs — and contributes 22 sources found by no other route. It is the reason this sample
  reaches down to `1.05 M☉` where §6 and §7 stop at `1.4` and `1.25 M☉` respectively.

### 8.4 Spectroscopic (SB1) branch (paper §2.2)

**Derived quantity.** The classical spectroscopic binary mass function — **not** the astrometric one
from §5.2:

```
f_m = P_orb * K1^3 * (1 - e^2)^(3/2) / (2 * pi * G)
```

where `K1` is the RV semi-amplitude and `e` the eccentricity. Assuming primary mass `M̃1`, solve
numerically for `M̃2,min`, the companion mass implied at edge-on inclination.

**Cut chain.** A significance gate, then a disjunction of two mass routes:

| # | `id` | Cut |
|---|---|---|
| 1 | `k1_significance` | `significance > 10`, where `significance = K1 / σ_K1` |
| 2 | `mass_route` | **either** `f_m > 3 M☉` **or** (`M̃2,min > 1.4 M☉` **and** `M̃2,min > M̃1`) |

Because `M̃1` is defined only for main-sequence sources (§8.2), the second route applies **only to
main-sequence sources**; the `f_m > 3 M☉` route has no such restriction and is the only way an
evolved source enters this branch.

Published breakdown the waterfall must reproduce:

| Route | N |
|---|---|
| Main-sequence minimum-companion-mass route | 136 |
| `f_m > 3 M☉` route | 30 |
| Both routes | 15 |
| **Union** | **151** |

Arithmetic check: `136 + 30 − 15 = 151`.

### 8.5 Outcome-dependent criteria — inclusion operator only

The initial 227-source selection is catalog-level and carries **no** outcome-dependent criteria.
Everything below describes what happened *after* selection, and per the §7.3 rule none of it may be
implemented as a filter that discards real systems. All of it belongs in the Poisson inclusion
operator (§4.7), applied identically to mock realizations.

**Astrometric branch follow-up outcomes** (of 76):

| Outcome | N |
|---|---|
| Characterized via own follow-up + archival spectra/light curves | 70 |
| Gaia solution shown spurious by follow-up/archival RVs before a many-epoch orbit | 16 |
| Secure short-period eclipsing binary implying a hierarchical triple | 1 |
| Additional possible eclipsing system | 1 |
| Many-epoch follow-up with a measured orbital solution | 51 |

**Spectroscopic branch follow-up outcomes** (of 151, all vetted):

| Outcome | N |
|---|---|
| Two-temperature SEDs | 52 |
| — of which Algol-type with strong ellipsoidal variability | 20 |
| Eclipsing binaries from TESS / ASAS-SN | 25 |
| — eclipse period ≪ Gaia SB1 period → hierarchical triple | 14 |
| — eclipsing binary may itself produce the Gaia solution | 11 |
| Obviously double-lined (SB2) | 15 |
| Rejected with RVs clearly inconsistent with the Gaia SB1 orbit | 23 |
| Multi-epoch follow-up obtained | 24 |
| Independent RV-only orbits measured | 19 |

**Explicitly outcome-dependent follow-up prioritization**, which must be modeled in the follow-up
selection function rather than treated as a property of the systems:

- Sources whose measured RVs were inconsistent with the Gaia solution were **deprioritized** for
  further follow-up — though complete orbits were still obtained for nine of them.
- Sources **closer to the ZAMS were prioritized**, because a luminous companion or inner binary is
  harder to rule out for evolved stars, and SED fitting suggested a majority of the SB1 sample is
  somewhat evolved.
- Follow-up of sources with detected eclipses was **generally avoided**, with outer orbits obtained
  only where eclipses were discovered late in the program.

These prioritization rules condition on measured RVs and SED-fit outcomes, i.e. on the result of the
observation being selected for. The follow-up selection function may condition only on
pre-follow-up observables; the rest is inclusion-operator territory.

### 8.6 Purity and contamination as per-sample parameters

Per the established rule, these are **modeled per-sample parameters in this file**, never global
constants — and note they differ sharply between the two branches, so each branch carries its own:

```yaml
contamination:
  astrometric:
    reliable_solution_and_compact_object_fraction: 0.60
  spectroscopic:
    spurious_solution_fraction: 0.50
    dominant_residual_contaminants:
      - post_mass_transfer_binaries
      - hierarchical_triples
```

For context, the paper's astrometric branch yields the two known Gaia BHs, 27 NS candidates, and a
dozen massive WDs; it also shows that **tight WD+WD binaries can masquerade as NSs** within this
sample — a contamination channel our `companion_nature_likelihood` stage should be checked against.
The SB1 branch's `M̃2,min > 3 M☉` candidates were **all** luminous binaries or spurious solutions,
and the astrometric branch's `M̃2 > 2 M☉` candidates other than the two confirmed BHs were refuted or
revised downward.

Compare §7.3's ~25% spurious rate among good-quality-flag candidates: that is a third, different
number for a third, different candidate population. Three samples, three rates, three files.

### 8.7 Cross-sample dependencies

This sample is **not independent of the others**, which has consequences for both the registry and
the likelihood:

- Astrometric subsample 3 **is** the Andrews et al. (2022) selection restricted to `G < 15`. The
  registry must therefore support one selection file depending on another's evaluated membership,
  and `elbadry2026` must declare `depends_on: [andrews2022]`.
- Astrometric subsample 2 depends on **Table E1 of El-Badry et al. (2023a)**, an external published
  table that must be ingested as a frozen fixture (§15 Q8).
- Two SB1 sources — `263578264603666560` and `5728328827639713792` — have astrometric solutions with
  similar periods but did **not** enter the astrometric sample; the paper attributes this to
  luminous companions diluting the photocenter. These are a useful test case for the forward-model
  path's flux-ratio handling (§5.4).

Combined with §7's dependence on Andrews, the overlap structure across our four samples is a
three-way tangle, not pairwise. This sharpens §15 Q1 rather than resolving it.

### 8.8 Forward-model validation targets

Published sample properties the forward-model path should reproduce:

- Most systems in both RV follow-up samples are **solar-type stars**; nearly all apparently massive
  or evolved primaries were rejected early for multiple luminous components or spurious orbits.
- **SB1 targets are on average more luminous** than astrometric targets.
- Many astrometric targets, particularly the lowest-luminosity ones, fall on the **blue edge of the
  main sequence**, likely from WD companion light.
- **SB1 orbital periods are on average shorter** than astrometric ones — astrometric wobble grows
  with period while RV amplitude grows at short period. Reproducing this opposing period response
  in the two branches is the sharpest available test of the forward model.
- Both branches are largely confined to **`P_orb ≲ 1000 d`** (the DR3 observing baseline) and
  **`d ≲ 2 kpc`**.

### 8.9 Independent cut-chain validation (paper §2.3.1)

The paper's comparison against Simon et al. (2026) is an unusually good acceptance test, because it
enumerates *why* each non-overlapping source failed. Simon et al. present 20 sources with DR3
orbital solutions (besides Gaia BH1, BH2, NS1); 11 are in this paper's samples (7 astrometric, 4
SB1); the 9 absent ones were excluded by specific quality cuts:

| Reason | N |
|---|---|
| SB1 sources failing `significance > 10` | 5 |
| Astrometric sources with `F2 > 10` | 2 |
| Fainter than `G = 15` | 1 |
| Fails `M̃2/M̃1 > 1.2` | 1 |

**Spec this as a required acceptance test**: running our implementation of the cut chain over the
Simon et al. (2026) source list must reproduce this exclusion breakdown exactly. It validates four
distinct thresholds independently of the aggregate N, which the headline count of 227 cannot do.

None of the 11 overlapping sources are in the final follow-up samples of 51 astrometric and 24 SB1
systems.

### 8.10 Config sketch

```yaml
schema_version: 1
name: elbadry2026
provenance:
  reference: "El-Badry et al. (2026)"
  arxiv: "2608.06453v1"
  section: "2. Sample selection"
  published_n: 227
  published_n_by_branch:
    astrometric: 76
    spectroscopic: 151
  data_release: dr3
mode: reproduction
depends_on: [andrews2022]
primary_mass:
  method: janssens2022_mg_mass_relation
  applies_only_when: "main_sequence == true"
extinction:
  north: {applies_when: "dec_deg > -28.0", map: green2019}
  south: {applies_when: "dec_deg <= -28.0", map: lallement2019}
  coefficients: {e_bp_rp_over_e_bv: 1.33, a_g_over_e_bv: 2.66}
main_sequence_cut:
  # MG_0 > mg_floor OR MG_0 > cmd_intercept + cmd_slope * (BP-RP)_0
  mg_floor: 4.5
  cmd_intercept: -9.37
  cmd_slope: 13.42
monte_carlo:
  n_draws: 10000
  covariance: full_12x12
branches:
  astrometric:
    parent_query:
      adql: |
        SELECT ...
        FROM gaiadr3.nss_two_body_orbit AS nss
        WHERE nss.nss_solution_type IN ('Orbital', 'AstroSpectroSB1')
    a0_method: nsstools            # see §15 Q7
    statistic: amrf                # A = (a0/varpi) * (M1/Msun)^(-1/3) * (P/yr)^(-2/3)
    subsamples:
      - id: primary_ns_bh
        require_main_sequence: true
        m2_msun_min: 1.4
        m2_over_m1_min: 1.2
        goodness_of_fit_max: 10.0
        period_days_max: 1200.0
        g_mag_faint_limit: 15.0
        expected_n: 47
      - id: elbadry2023_table_e1
        external_table: config/selections/external/elbadry2023_table_e1.yaml
        require_main_sequence: false
        apply_goodness_of_fit_cut: false
        g_mag_faint_limit: 15.0
        expected_n: 5
        expected_n_new: 3
      - id: andrews2022_import
        from_sample: andrews2022
        g_mag_faint_limit: 15.0
        expected_n: 16
        expected_n_new: 4
      - id: sub_chandrasekhar
        require_main_sequence: true
        m2_msun_min: 1.05
        m2_msun_max: 1.40
        m2_msun_error_max: 0.105
        period_days_max: 900.0
        g_mag_faint_limit: 15.0
        expected_n: 22
    expected_union_n: 76
  spectroscopic:
    parent_query:
      adql: |
        SELECT ...
        FROM gaiadr3.nss_two_body_orbit AS nss
        WHERE nss.nss_solution_type IN ('SB1', 'SB1C')
    statistic: spectroscopic_mass_function
    cuts:
      - id: k1_significance
        kind: derived
        expression: "k1 / k1_error > k1_significance_min"
        k1_significance_min: 10.0
      - id: mass_route
        kind: derived
        expression: "fm_msun > fm_msun_min or (m2_min_msun > m2_min_msun_floor and m2_min_msun > m1_msun)"
        fm_msun_min: 3.0
        m2_min_msun_floor: 1.4
        second_route_requires_main_sequence: true
    expected_n_by_route:
      main_sequence_min_companion_mass: 136
      high_mass_function: 30
      both: 15
    expected_union_n: 151
contamination:
  astrometric: {reliable_solution_and_compact_object_fraction: 0.60}
  spectroscopic: {spurious_solution_fraction: 0.50}
acceptance_tests:
  simon2026_exclusion_breakdown:
    sb1_fails_significance: 5
    astrometric_f2_above_max: 2
    fainter_than_g_limit: 1
    fails_m2_over_m1: 1
```

Every numeric threshold above lives in this file, never inline in Python.

### 8.11 Summary of differences from El-Badry et al. (2024)

| Aspect | El-Badry 2024 (§7) | El-Badry 2026 (§8) |
|---|---|---|
| Published N | 21 | 227 (76 astrometric + 151 SB1) |
| Parent solution types | `Orbital`, `AstroSpectroSB1` | Same **plus** `SB1`, `SB1C` |
| Candidate statistic | Shahaf AMRF triage via published catalog | AMRF computed directly + spectroscopic `f_m` |
| Primary mass | IsocLum, `gaiadr3.binary_masses` | Janssens et al. (2022) `MG,0`–mass, MS only |
| Extinction split | δ = −30°, Lallement 2022 | δ = −28°, Lallement 2019 |
| Companion-mass floor | 1.25 M☉ | 1.4 M☉ (primary), down to 1.05 M☉ (subsample 4) |
| Magnitude limit | G < 15 | G < 15 (both branches) |
| Goodness-of-fit cut | — | F2 < 10 (subsample 1 only) |
| Selection outcome-dependence | **Yes** — criteria (b)(c)(d) depend on follow-up | **No** — initial 227 is catalog-level only |
| Spurious rate | ~25% of good-quality-flag candidates | ~40% astrometric; ~50% SB1 |

The last two rows are the ones that matter architecturally. El-Badry 2026's initial sample is
directly forward-modelable in a way El-Badry 2024's published sample is not, which makes it the
better anchor for the DR4 selection design the paper itself is aimed at.

## 9. Sample: Acceleration/jerk stars

**File:** `config/selections/accel_jerk.yaml` · **Roster:** #22 · **Model:** Mid · **Effort:**
Standard

Unlike §6–§8, this is **not a published compact-object candidate list with a target N to reproduce**.
Per `ARCHITECTURE.md` §4, the acceleration/jerk catalogs are matched **at the broad
population/aggregate level only**, using the same `gaiamock` cascade to forward-model which systems
land in which solution type, and the pool doubles as the RV follow-up target list. Systems are
tracked in a separate pending pool, promotable once their orbits resolve.

Therefore:

- `mode` is **`forward_model` only**; there is no reproduction path. State this explicitly in the
  file so the dual-path machinery does not expect a published N.
- The selection function is the **solution-type occupancy** predicted by the cascade — the fraction
  of mock systems landing in the 7-parameter (acceleration) and 9-parameter (jerk) bins — validated
  against the real catalog's aggregate fractions, reusing the existing solution-type-fraction
  diagnostic from `selection_function_astrometric`.
- The DR3/DR4 catalog identifiers stay path-specific
  (`dr3.selection_function_followup.accel_jerk_catalog_id`), consistent with
  `dark-hunter-pop-workflow` §6.
- Adoption dates for the follow-up-target aspect continue to come from
  `config/target_lists/derived/accel_jerk_adoption_dates.yaml`; this file governs the *selection*
  aspect only, and must not duplicate those dates.

## 10. `data_acquisition` additions required

Andrews' Monte Carlo propagation (§6.3, §11) needs the **full 12×12 astrometric covariance matrix**
per NSS solution. The current DR3 query in `data_acquisition.build_nss_adql` selects the
Thiele-Innes elements and their 1σ errors, but **not** the correlation information — so the
covariance cannot currently be reconstructed.

Required additions (roster #18, Mid / Standard):

1. **Query columns.** Add `nss.corr_vec` and `nss.bit_index` to the NSS `SELECT` list. `corr_vec` is
   the packed lower-triangular correlation vector for the solution's fitted parameters; `bit_index`
   identifies which parameters are present for that solution type, and is required to unpack
   `corr_vec` into the correct matrix positions.
2. **Unpacking helper.** Reconstruct the full covariance from `corr_vec`, `bit_index`, and the
   per-parameter `*_error` columns, following the Gaia DR3 NSS data-model definition of the packing
   order. Parameter count varies by solution type — `Orbital` is 12-parameter, other types differ —
   so the unpacker must be driven by `bit_index`, never by a hardcoded size.
3. **Schema.** Extend `CandidateRecord` with the reconstructed covariance, respecting the
   `ParameterSet` convention (`dark-hunter-pop-workflow` §3): the astrometric solution is a
   correlated multi-quantity fit and should be stored as a named vector plus covariance, tagged with
   provenance, not as loose scalars.
4. **Persistence.** Store the matrix in the `data_acquisition` HDF5 artifact alongside the existing
   NSS panels, so downstream stages do not re-query.
5. **Validation.** Assert symmetry and positive-semi-definiteness on load; record the count of
   solutions that fail PSD, and expose it in the funnel diagnostic rather than silently dropping
   them.

**No diagonal-only fallback path is specified.** If a solution's covariance cannot be reconstructed,
the system is recorded as such in the funnel and excluded from samples that require the Monte Carlo
cut — it is not silently downgraded to independent errors, which would misstate the M2 posterior.

If the archive columns prove insufficient, the covariance matrices can be obtained separately and
staged as a local file under a config path, following the same pattern as the cooling tracks
(`ARCHITECTURE.md` §4 `companion_nature_likelihood`).

## 11. Monte Carlo uncertainty propagation

Matching Andrews et al. (2022) exactly:

```yaml
monte_carlo:
  n_draws: 10000
  covariance: full_12x12
  random_seed: <config; recorded in the run manifest>
```

- **10⁴ draws per system** from the full 12×12 astrometric covariance (§10).
- Each draw is propagated through §5.1 → §5.2 → §5.3 to yield one `(m_f, M2)` realization; the
  ensemble is the per-system posterior.
- The probability cut (`P(M2 > 1.4 M☉) ≥ 0.95` for Andrews) is evaluated on this ensemble.
- `M2 / σ_M2` for the SNR cut uses the ensemble mean and standard deviation, not a linearized
  propagation.
- Seeds are recorded in the run manifest per `dark-hunter-pop-workflow` §4. Reproducibility is full
  accounting, not bitwise replay.
- **Required convergence diagnostic:** verify that the Monte Carlo noise on the probability estimate
  is subdominant at 10⁴ draws — the same principle as the existing `mc_noise_threshold` guardrail
  (`dark-hunter-pop-workflow` §7). A cut at exactly 95% probability is sensitive to MC noise for
  systems near the boundary; the diagnostic must report how many systems sit within the MC
  uncertainty of the threshold.

## 12. Config schema additions

### 12.1 Registry block in `config.yaml`

`config.yaml` holds **only** the on/off switches and the pointer to each sample's frozen file. All
thresholds live in the per-sample files (§4.4).

```yaml
sample_selection:
  enabled: true
  samples:
    - name: andrews2022
      enabled: true
      path: config/selections/andrews2022.yaml
      mode: reproduction
    - name: elbadry2024
      enabled: true
      path: config/selections/elbadry2024.yaml
      mode: reproduction
    - name: elbadry2026
      enabled: true
      path: config/selections/elbadry2026.yaml
      mode: reproduction
    - name: accel_jerk
      enabled: true
      path: config/selections/accel_jerk.yaml
      mode: forward_model
```

- Each sample is independently enable/disable-able, as required.
- `mode` is per-sample (§4.2), overridable per run.
- Drafted as `config/fragments/sample_selection.yaml` during development and merged into
  `config.yaml` by Review/Integration at the checkpoint, per `dark-hunter-pop-workflow` §5.

### 12.2 Pydantic schema

Add to `config_schema.py`, following existing conventions (`model_config = ConfigDict(extra="forbid")`,
`Field` bounds, `model_validator` cross-checks):

- `SampleSelectionEntry` — `name`, `enabled`, `path`, `mode`.
- `SampleSelectionConfig` — `enabled`, `samples: list[SampleSelectionEntry]`, with a validator
  rejecting duplicate names and missing files for enabled entries.
- `SampleSelectionMode` — a `str, Enum` with `REPRODUCTION` and `FORWARD_MODEL`. Per the workspace
  TypeScript-exhaustiveness rule's Python analogue, any dispatch over this enum raises on an
  unhandled member rather than falling through to a default.
- The per-sample file bodies get their own models (`SampleSelectionFile`, `SampleCut`,
  `SampleExclusion`, `PrimaryMassSpec`, `MonteCarloSpec`), loaded and validated at stage start so a
  malformed selection file fails before any query runs.

### 12.3 Stage registration

Register `sample_selection` as a named stage in `run_management.py` (`dark-hunter-pop-workflow` §2),
between `mass_derivation_bulk` and `selection_function_followup`. Its artifact path must be
parameterized by the **content hash of every enabled selection file plus its mode**, so changing a
threshold in `andrews2022.yaml` produces a genuinely different artifact rather than falsely reusing
a cached one.

### 12.4 DR3/DR4 independence

Per `dark-hunter-pop-workflow` §6: parent queries, solution-type lists, magnitude limits, and
catalog identifiers are **DR-path-specific keys even when the values match**. The physics shared
across paths (the mass-function formulae in §5) is not duplicated. Run the DR3/DR4 audit function
before any selection-config change is considered complete.

## 13. Diagnostics and acceptance criteria

Mandatory per `dark-hunter-pop-workflow` §8; full-detail reports (caveman exemption).

| Diagnostic | Content | Acceptance |
|---|---|---|
| `sample_attrition_waterfall` | Per sample, per cut `id`: N in, N out, N removed, and `expected_n_after` where published | Every published checkpoint matches exactly |
| `sample_reproduction_report` | Recovered source-ID set vs. published table; symmetric difference itemized | Andrews: N = 24. El-Badry 2024: N = 21. El-Badry 2026: N = 227 (76 astrometric + 151 SB1), with the §8.3 subsample and §8.4 route breakdowns matching |
| `simon2026_exclusion_breakdown` | For El-Badry 2026: why each Simon et al. (2026) source is in or out of the sample (§8.9) | 5 / 2 / 1 / 1 exclusion split reproduced exactly |
| `m2_posterior_convergence` | MC noise on `P(M2 > threshold)` at 10⁴ draws; count of systems within MC uncertainty of the cut | MC noise subdominant; boundary count reported |
| `covariance_health` | Count of solutions with missing / non-PSD covariance, by solution type | Reported in the funnel; zero silent drops |
| `sample_selection_function` | Forward-model path: survival probability vs. M2, P_orb, G, for each sample | Smooth, monotonic where physically expected |
| `sample_overlap_matrix` | Pairwise overlap in source IDs between enabled samples | Feeds the §15 Q1 double-counting decision |
| `mode_divergence` | Reproduction vs. forward-model recovered sets for the same sample | Divergence explained by the mass assumption, quantified |

A sample's reproduction path is **not considered working** until
`sample_reproduction_report` matches the published N exactly. Until then the sample must not be
enabled in `forward_model` mode for inference.

## 14. Phase 8 kickoff prompts

Same conventions as `docs/PHASE*_KICKOFF.md`: one worktree/branch per subagent, PRs to `main`,
`Closes #N` alone on a line, ≤3 concurrent sessions. Skills active in every session:
`strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow`.

Required CI gate stays `unit|physics|api`. Long multi-draw Monte Carlo suites go under
`@pytest.mark.slow`.

Issue numbers are assigned when the umbrella issue is opened; fill them into the prompts below.

---

### Slot A — sample-selection framework (#17)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Deep

Issue: <umbrella child for roster #17>
Branch from latest main: phase8/sample-selection-framework (git worktree).
PR base: main. Closes #N alone on a line. Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §4, §12, docs/FOUNDATION_INTERFACE_FREEZE.md,
ARCHITECTURE.md §4 and §7, and the four project skills.

Implement src/darkhunter_pop/sample_selection.py: the SampleSelection interface, the named-sample
registry, the ordered cut-chain evaluator with per-cut attrition accounting, and the per-sample
reproduction/forward_model mode switch. Add the config schema models from §12.2 and register the
stage per §12.3 with artifact paths hashed over the enabled selection files and their modes.

Do NOT implement any individual sample's cut chain — that is roster #20-#22. Ship the framework
plus a synthetic fixture sample proving the evaluator, attrition accounting, and mode switch work.

Effort contract (Deep): you own the interface shape. Justify the reproduction/forward_model
boundary in the PR description. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot B — NSS covariance ingestion (#18)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-4-sonnet    Effort: Standard

Issue: <umbrella child for roster #18>
Branch: phase8/nss-covariance. PR → main. Closes #N alone on a line. Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §10, ARCHITECTURE.md §3 (ParameterSet) and §4 data_acquisition,
FOUNDATION_INTERFACE_FREEZE, and the four project skills.

Add nss.corr_vec and nss.bit_index to build_nss_adql. Implement a bit_index-driven unpacker that
reconstructs the full astrometric covariance from corr_vec plus the per-parameter *_error columns,
per the Gaia DR3 NSS data model. Parameter count varies by solution type — never hardcode 12.
Store as a ParameterSet-style named vector + covariance with provenance on CandidateRecord, persist
in the data_acquisition HDF5 artifact, and validate symmetry + PSD on load. Non-PSD / missing
covariance is COUNTED and REPORTED in the funnel, never silently dropped and never downgraded to a
diagonal fallback.

Effort contract (Standard): interface is frozen by #17 and ARCHITECTURE.md; implement within it.
Unit tests for every solution type's unpack path. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot C — Monte Carlo mass-function propagation (#19)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Deep

Issue: <umbrella child for roster #19>
Branch: phase8/mc-mass-function. PR → main. Closes #N alone on a line. Labels: phase-8, enhancement.
Depends on #17 and #18 being merged.

Read docs/CONTINUATION_PLAN.md §5, §10, §11, and the four project skills.

Implement the shared astrometric primitives in physics_utils.py per §5 (Thiele-Innes u/v/a0, the
astrometric mass function, the dark-companion inversion for M2, and the luminous-companion
generalization in terms of flux ratio F and mass ratio q). Then implement 1e4-draw Monte Carlo
propagation from the full covariance to a per-system (m_f, M2) posterior, with the probability and
SNR statistics §6.4 needs.

Handle near-singular covariance explicitly and document the choice. Emit the
m2_posterior_convergence diagnostic from §13, including the count of systems within MC uncertainty
of a probability threshold. Seeds recorded in the run manifest.

Effort contract (Deep): you own the sampling strategy; justify it in the PR. Long MC suites under
@pytest.mark.slow. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot D — Andrews et al. (2022) selection (#20)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-4-sonnet    Effort: Light

Issue: <umbrella child for roster #20>
Branch: phase8/selection-andrews2022. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #17 and #19 being merged.

Read docs/CONTINUATION_PLAN.md §4, §5, §6, §11, and the four project skills.

Transcribe the Andrews et al. (2022) cut chain into config/selections/andrews2022.yaml exactly as
specified in §6, including every numeric threshold, the fixed M1 = 1.0 Msun assumption, the
no-extinction CMD cut, and the excluded source 4373465352415301632 with its stated reason. Wire it
into the registry from #17. Verify the ADQL solution-type set actually returns the published parent
N of 134,598 and freeze it.

Success criterion: the reproduction path recovers exactly 24 systems, and the attrition waterfall
matches 134,598 -> 106 -> ... -> 24.

Effort contract (Light): transcription only. You may NOT invent thresholds, reinterpret a cut, or
change the framework. If a published number cannot be reproduced, STOP and escalate with the
attrition table rather than tuning anything. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot E — El-Badry et al. (2024) selection (#21)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Standard

Issue: <umbrella child for roster #21>
Branch: phase8/selection-elbadry2024. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #17 and #19 being merged.

Read docs/CONTINUATION_PLAN.md §4, §7, and the four project skills. Source paper: arXiv:2405.00089v2
Section 2.

Build config/selections/elbadry2024.yaml per §7: parent = DR3 Orbital + AstroSpectroSB1, Shahaf
et al. (2023b) triage (177 candidates), the P_orb > 1000 d additions, the Andrews-sourced
additions, IsocLum primary masses from gaiadr3.binary_masses, the declination-split extinction
policy, and the four completeness criteria (G < 15; M2 > 1.25 Msun from JOINT astrometry+RV fitting;
not spurious; >= half an orbit observed).

CRITICAL — §7.3: criteria (b), (c), (d) are outcome-dependent. They go in the likelihood inclusion
operator applied identically to mocks, NOT as a data filter that discards real systems. The
follow-up selection function may condition only on pre-follow-up observables. Model the ~25%
spurious rate among good-quality-flag candidates as a per-sample parameter, not a global constant.

Resolve §15 Q2 (Shahaf catalog cross-match vs. AMRF reimplementation) in the PR description; the
reproduction path may use the catalog, but state what the forward-model/DR4 path needs.

Effort contract (Standard): framework frozen by #17. Success criterion: reproduction path recovers
21 systems. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot F — acceleration/jerk selection (#22)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-4-sonnet    Effort: Standard

Issue: <umbrella child for roster #22>
Branch: phase8/selection-accel-jerk. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #17 being merged.

Read docs/CONTINUATION_PLAN.md §9, ARCHITECTURE.md §4 selection_function_astrometric and
selection_function_followup, and the four project skills.

Build config/selections/accel_jerk.yaml as forward_model-only — there is no published N to
reproduce, and the file must say so explicitly so the dual-path machinery does not expect one. The
selection function is solution-type occupancy (7-parameter acceleration, 9-parameter jerk) from the
existing gaiamock cascade, validated against the real catalog's aggregate fractions by reusing the
solution-type-fraction diagnostic.

Keep the DR3/DR4 catalog identifiers path-specific. Do NOT duplicate the adoption dates already in
config/target_lists/derived/accel_jerk_adoption_dates.yaml.

Effort contract (Standard). Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot I — SB1 spectroscopic mass-function support (#26)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-4-sonnet    Effort: Standard

Issue: <umbrella child for roster #26>
Branch: phase8/sb1-mass-function. PR → main. Closes #N alone on a line. Labels: phase-8, enhancement.
Depends on #18 being merged.

Read docs/CONTINUATION_PLAN.md §8.4, §8.1, §15 Q6, ARCHITECTURE.md §4 data_acquisition, four skills.

Extend data_acquisition to ingest DR3 SB1 and SB1C solutions with the columns the spectroscopic
mass function needs: K1 (semi-amplitude), its error, eccentricity, and period. Implement the
spectroscopic binary mass function f_m = P K1^3 (1-e^2)^(3/2) / (2 pi G) in physics_utils.py, plus
the numerical inversion for M2_min at edge-on inclination given a primary mass.

This is the FIRST non-astrometric mass path in the pipeline. Keep it clearly separated from the
astrometric mass function in §5.2 — same module, distinct names, no shared inversion code that
could silently apply the wrong relation.

Escalate §15 Q6 (does the SB1 branch feed inference, or is it reproduction/comparison-only?) rather
than deciding it; ship the ingestion and primitive either way.

Effort contract (Standard). Unit tests against closed-form values. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot J — El-Badry et al. (2026) selection (#25)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Deep

Issue: <umbrella child for roster #25>
Branch: phase8/selection-elbadry2026. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #17, #19, #20, and #26 being merged.

Read docs/CONTINUATION_PLAN.md §4, §5, §8 in full, and the four project skills.
Source paper: arXiv:2608.06453v1 Section 2.

Build config/selections/elbadry2026.yaml per §8.10. This sample has TWO parent branches:
- Astrometric (Orbital, AstroSpectroSB1): AMRF statistic, union of four subsamples
  (47 + 3 + 4 + 22 = 76 unique).
- Spectroscopic (SB1, SB1C): spectroscopic mass function, 136 + 30 - 15 = 151.
Total published N = 227.

Implement the AMRF exactly once and derive it from the §5.2 astrometric mass function via the
identity A^3 = m_f / M1 — do NOT maintain a second independent inversion.

This sample owns the Janssens et al. (2022) MG_0-mass relation (§8.2), which is defined ONLY for
main-sequence sources and differs from both Andrews (fixed 1.0 Msun) and El-Badry 2024 (IsocLum).
Its extinction split is delta = -28 deg with Lallement 2019, NOT the -30 deg / Lallement 2022 used
by El-Badry 2024. Do not consolidate either one.

Subsample 3 imports Andrews' evaluated membership, so declare depends_on: [andrews2022] and make
the registry resolve it. Subsample 2 needs El-Badry et al. (2023a) Table E1 staged as a frozen
fixture under config/selections/external/ (§15 Q8).

The initial 227-source selection is purely catalog-level: NO outcome-dependent criteria in the cut
chain. The follow-up outcomes and prioritization rules in §8.5 go in the likelihood inclusion
operator applied identically to mocks, never as filters on real systems. Branch-specific
contamination rates (§8.6) are per-sample parameters in this file.

Required acceptance test beyond the headline N: reproduce the Simon et al. (2026) exclusion
breakdown from §8.9 (5 / 2 / 1 / 1). It validates four thresholds independently.

Effort contract (Deep): you own the two-branch structure and how it maps onto the #17 framework.
Justify it in the PR. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot G — likelihood integration (#23)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Deep

Issue: <umbrella child for roster #23>
Branch: phase8/sample-selection-inference. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #20, #21, #22, #25 being merged.

Read docs/CONTINUATION_PLAN.md §4.7, §8.7, §13, §15, ARCHITECTURE.md §4 inference, and four skills.

Extend the Poisson rate to include sample_selection_function_s per §4.7. Resolve §15 Q1: the
samples OVERLAP three ways, not pairwise — El-Badry 2024 drew candidates from Andrews, and El-Badry
2026's astrometric subsample 3 IS the Andrews selection restricted to G < 15 (§8.7). Naive summation
double-counts. Decide between separate Poisson processes per sample and a unified
inclusion-indicator formulation, justify it in writing, and implement it. Emit the
sample_overlap_matrix diagnostic.

Also fold in the outcome-dependent inclusion terms from §7.3 and §8.5, applied identically to mocks.

Do not silently upgrade v1's staged-but-connected treatment to a fully joint one
(dark-hunter-pop-workflow §7) — that is a separately scoped v2 decision.

Effort contract (Deep): this changes the core likelihood. Written justification plus SBC-style
recovery evidence required in the PR. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot H — reproduction diagnostics (#24)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: cursor-grok-4.6-high-fast    Effort: Standard

Issue: <umbrella child for roster #24>
Branch: phase8/sample-diagnostics. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #20, #21, #22, #25 being merged.

Read docs/CONTINUATION_PLAN.md §13, §8.9, docs/PLOTS.md, ARCHITECTURE.md §4 diagnostics, four skills.

Implement the §13 diagnostics: sample_attrition_waterfall, sample_reproduction_report,
simon2026_exclusion_breakdown, covariance_health, sample_selection_function, and mode_divergence.
The attrition waterfall must handle El-Badry 2026's two-branch, four-subsample union structure
(§8.3) rather than assuming a single linear cut chain. Use the shared plotting
primitives in plotting.py; do not add new rendering code paths. Reports are full-detail (caveman
exemption). Wire them into the diagnostics hook registry with config on/off switches matching the
existing hook convention.

Effort contract (Standard). Full required pytest before PR. Stop when PR open + CI green.
```

---

### Review/Integration (continuous, roster #15)

```
You are the continuous Review/Integration subagent (roster #15).
Model: claude-opus-5-thinking-medium    Effort: Deep

When Phase 8 PRs open: merge config/fragments/sample_selection.yaml into config/config.yaml; verify
every numeric threshold lives in config/selections/*.yaml and none leaked inline
(dark-hunter-pop-workflow §1); check DR3/DR4 key independence per §12.4 and run the audit function;
confirm no selection file was edited after freeze without a schema_version bump; keep slow MC
suites out of the default CI gate.

Prefer review plus small integration PRs. Docs-first before any freeze break.
```

## 15. Open items

| Q | Question | Blocks | Owner |
|---|---|---|---|
| Q1 | Samples overlap **three ways, not pairwise**: El-Badry 2024 drew candidates from Andrews, and El-Badry 2026's astrometric subsample 3 *is* the Andrews selection restricted to `G < 15` (§8.7). Separate Poisson processes per sample, or one unified inclusion-indicator formulation? Naive summation double-counts. | #23 | #23 subagent, with human sign-off |
| Q2 | Shahaf et al. (2023b) triage: cross-match the published 177-candidate catalog, or reimplement the AMRF algorithm? Catalog is fine for reproduction; the forward-model and DR4 paths need the algorithm, since a catalog cannot be applied to mocks. **Partially eased by §8.3**, which computes the AMRF directly and so supplies the algorithm side; the remaining question is whether §7's triage can be re-expressed in those terms or genuinely needs the catalog. | #21, DR4 | #21 subagent, with human sign-off |
| Q4 | Whether `nss.corr_vec` / `nss.bit_index` as published are sufficient to reconstruct the full 12×12 covariance for every solution type, or whether matrices must be staged from a local file (§10). | #18 | #18 subagent |
| Q5 | Exact `nss_solution_type` string set that reproduces Andrews' parent N of 134,598. | #20 | #20 subagent |
| Q6 | **Scope:** does El-Badry 2026's SB1 branch feed the compact-object mass function, or is it reproduction/comparison-only? The pipeline's mass derivation is astrometric throughout (`ARCHITECTURE.md` §4), and an SB1 mass function constrains only `M2 sin³i`, so the inclination marginalization differs fundamentally from the astrometric path. Reproduction-only is the smaller, safer scope; feeding inference is a genuine v1-scope expansion that should not be made incrementally. | #25, #26, #23 | User, with #26 subagent input |
| Q7 | `nsstools` (Halbwachs et al. 2023) is what El-Badry 2026 uses for `ã0` and `σ_ã0`. Do we vendor it as a pinned dependency alongside `gaiamock`, or use our existing `thiele_innes_to_campbell` (§5.1)? Either is defensible, but the reproduction path must match the paper's numbers, so any difference between the two must be measured before choosing. | #25 | #25 subagent, with human sign-off |
| Q8 | Two external published tables are needed as frozen fixtures: **Table E1 of El-Badry et al. (2023a)** (§8.3 subsample 2) and the **Janssens et al. (2022) Figure 3** `MG,0`–mass relation (§8.2). The latter is published as a figure, not a table, so it must be obtained in tabular form or digitized with a stated uncertainty. | #25 | User to supply; #25 subagent to stage |
| Q9 | El-Badry 2026 §2.1 subsample 3 says the Andrews import "yields 16 sources", but Andrews' published sample is 24 (§6). The difference is presumably the `G < 15` cut, but the paper does not state it explicitly. Confirm that applying `G < 15` to our reproduced Andrews sample yields exactly 16, and treat a mismatch as a failure of *both* reproductions rather than tuning either. | #25 | #25 subagent |
| Q10 | El-Badry 2026's `M̃1` is defined only for main-sequence sources, so `M̃1`-dependent cuts are undefined for evolved candidates rather than false. Confirm the framework's cut evaluator distinguishes "cut not applicable" from "cut failed" in the attrition waterfall — otherwise the subsample counts will not reproduce. | #17, #25 | #17 subagent |

**Resolved.** Q3 (El-Badry 2026 paper not yet supplied) — paper received as arXiv:2608.06453v1; §8
is fully specified.
