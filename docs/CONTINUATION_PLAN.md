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
against*: each published compact-object candidate sample (Andrews et al., El-Badry et al.) applied
its own idiosyncratic cut chain to the parent NSS catalog, and those cut chains are themselves
selection functions that must be forward-modeled, not ignored.

Phase 8 adds:

1. A **per-sample selection-function layer** — a registry of named literature samples, each with a
   frozen, versioned, reproducible cut chain (§4–§9).
2. The **12×12 astrometric covariance ingestion** that Andrews' Monte Carlo mass-function
   propagation requires (§10).
3. The **config/preset machinery** to enable or disable matching each published sample
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
| 25 | El-Badry et al. (2026) selection | #17, paper | TBD | TBD | Blocked pending paper (§8). |

Review/Integration (#15) continues, merging `config/fragments/` into `config/config.yaml` at each
checkpoint as before.

**Sequencing.** #17 and #18 first and in parallel (both blocking). Then #19. Then #20–#22 in
parallel (≤3 concurrent per `ORCHESTRATION_PLAN.md` §1). Then #23 and #24. #25 whenever the paper
lands.

## 4. Sample selection functions — architecture

### 4.1 Why each sample needs its own selection function

`ARCHITECTURE.md` §4 already models two selection effects: `selection_function_astrometric` (does
Gaia produce an acceptable orbital solution?) and `selection_function_followup` (does the system get
RV follow-up?). Neither captures a third effect: **a published sample is a specific, hand-built
subset of the NSS catalog, defined by a cut chain that is a strong and non-obvious function of
mass, period, magnitude, and goodness-of-fit.**

Andrews' 24 systems and El-Badry's 21 systems are not random draws from the compact-object
population; they are draws filtered through cuts like "95% probability that M2 > 1.4 M☉" and
"G < 15", which sculpt the recovered mass function directly. Comparing our dN/dM to theirs, or
using their systems as anchors, requires forward-modeling those cut chains through the same mock
pipeline that already handles the astrometric and follow-up selection.

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
    ├── elbadry2026.yaml              # placeholder, disabled (§8)
    └── accel_jerk.yaml
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

## 8. Sample: El-Badry et al. (2026) — TODO pending paper

**File:** `config/selections/elbadry2026.yaml` · **Roster:** #25 · **Status: BLOCKED — paper not yet
supplied.**

Create the file **disabled** (`enabled: false` in the registry) with the same structure as §6 and
§7, and every field stubbed as `null` with a `TODO` marker. Do not guess values.

Sections to fill once the paper is available, mirroring §7:

1. Parent query and solution types.
2. Triage / candidate-identification algorithm and its published catalog, if any.
3. Primary-mass assumption (per §4.3 — must be owned by this file).
4. Cut chain, ordered, with `expected_n_after` at each published checkpoint.
5. Outcome-dependent criteria, called out explicitly as in §7.3.
6. Published N and the source-ID table for the reproduction check.
7. Extinction policy.
8. Forward-model validation targets.

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
      enabled: false
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
| `sample_reproduction_report` | Recovered source-ID set vs. published table; symmetric difference itemized | Andrews: N = 24. El-Badry 2024: N = 21 |
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

### Slot G — likelihood integration (#23)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop.
Model: claude-opus-5-thinking-medium    Effort: Deep

Issue: <umbrella child for roster #23>
Branch: phase8/sample-selection-inference. PR → main. Closes #N alone on a line. Labels: phase-8.
Depends on #20, #21, #22 being merged.

Read docs/CONTINUATION_PLAN.md §4.7, §13, §15, ARCHITECTURE.md §4 inference, and the four skills.

Extend the Poisson rate to include sample_selection_function_s per §4.7. Resolve §15 Q1: the
samples OVERLAP (El-Badry drew candidates from Andrews), so naive summation double-counts. Decide
between separate Poisson processes per sample and a unified inclusion-indicator formulation,
justify it in writing, and implement it. Emit the sample_overlap_matrix diagnostic.

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
Depends on #20, #21, #22 being merged.

Read docs/CONTINUATION_PLAN.md §13, docs/PLOTS.md, ARCHITECTURE.md §4 diagnostics, four skills.

Implement the §13 diagnostics: sample_attrition_waterfall, sample_reproduction_report,
covariance_health, sample_selection_function, and mode_divergence. Use the shared plotting
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
| Q1 | Samples overlap (El-Badry drew from Andrews). Separate Poisson processes per sample, or one unified inclusion-indicator formulation? Naive summation double-counts. | #23 | #23 subagent, with human sign-off |
| Q2 | Shahaf et al. (2023b) triage: cross-match the published 177-candidate catalog, or reimplement the AMRF algorithm? Catalog is fine for reproduction; the forward-model and DR4 paths need the algorithm, since a catalog cannot be applied to mocks. | #21, DR4 | #21 subagent, with human sign-off |
| Q3 | El-Badry et al. (2026) paper not yet supplied; §8 is a stub. | #25 | User |
| Q4 | Whether `nss.corr_vec` / `nss.bit_index` as published are sufficient to reconstruct the full 12×12 covariance for every solution type, or whether matrices must be staged from a local file (§10). | #18 | #18 subagent |
| Q5 | Exact `nss_solution_type` string set that reproduces Andrews' parent N of 134,598. | #20 | #20 subagent |
