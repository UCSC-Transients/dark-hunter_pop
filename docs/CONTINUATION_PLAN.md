# dark-hunter_pop — Continuation Plan (Phase 8+)

Companion to `ARCHITECTURE.md` (authoritative technical specification) and `ORCHESTRATION_PLAN.md`
(subagent roster, phasing, git/PR workflow). Those two documents remain authoritative for anything
they already cover; this document extends them and does not restate them.

**Status: Phase 8 implemented on `main`.** Spec + kickoff prompts remain authoritative; roster
#17–#21 and #23–#27 landed under umbrella [#103](https://github.com/UCSC-Transients/dark-hunter_pop/issues/103)
(children #104–#113). Roster #22 / accel_jerk stays blocked (§9) with the registry entry disabled.

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
4. A **shared, sample-independent spuriousness model** `P(spurious | ·)`, replacing the three
   per-sample contamination constants the literature quotes, fit against **293 labeled sources
   staged from four published tables** (§4.8).
5. The **config/preset machinery** to enable or disable matching each published sample
   independently (§12).

One correction to the literature is carried deliberately rather than silently: Andrews et al.
(2022) excluded Gaia BH1 as a scanning-law artifact, so the frozen reproduction path keeps that
exclusion and a **modified variant** restores it for inference (§6.6).

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

Every Phase 8 prompt in §14 names its tier and effort in the opening line — `(top-tier, Deep
effort)` — and repeats the slug on a `Model:` line immediately below it. When
dispatching via the Task tool rather than the Agents Window, pass the slug from §2.1 directly; when
dispatching via the Agents Window, hand-pick it in the session picker.

## 3. Phase 8 roster

Extends the `ORCHESTRATION_PLAN.md` §4 roster; numbering continues from #16.

| # | Subagent | Depends on | Model tier | Effort | Rationale |
|---|---|---|---|---|---|
| 17 | **Sample-selection framework** (`sample_selection.py`, registry, `SampleSelection` interface, dual-path mode switch, config schema) | — (first, blocking) | Top | Deep | Every per-sample module builds against this interface; the reproduction/forward-model split is a contract, not an implementation detail. |
| 18 | **NSS covariance ingestion** (`corr_vec`/`bit_index` → 12×12 matrices in `data_acquisition`) | — (parallel with #17) | Mid | Standard | Well-specified ETL against a documented Gaia data model; correctness is checkable against Gaia's published matrix layout. |
| 19 | **Monte Carlo mass-function propagation** (1e4 draws, per-system posterior over `m_f` and `M2`) | #17, #18 | Top | Deep | Sampling from a rank-deficient / near-singular covariance needs real care; downstream probability cuts depend on it being right. |
| 20 | **Andrews et al. (2022) selection** (`andrews2022.yaml` **and** `andrews2022_modified.yaml`) | #17, #19 | Mid | Light | Cut chain is fully specified in §6; success criteria are the blocking parent count of 134,598, then N = 24 frozen and N = 25 modified (§6.6). |
| 21 | **El-Badry et al. (2024) selection** (`config/selections/elbadry2024.yaml` + module) | #17, #19 | Top | Standard | Outcome-dependent criteria (§7.3) are a genuine statistical hazard, not a transcription job. |
| 22 | **Acceleration/jerk sample selection** (`config/selections/accel_jerk.yaml` + module) — **BLOCKED** | #17 **+ user-supplied selection function (§9)** | Mid | Standard | Aggregate-level matching only (`ARCHITECTURE.md` §4). The selection is unpublished; there is nothing to build until the user supplies it. Do not derive one from the target list. |
| 23 | **Selection-function integration into `inference`** (per-sample terms in the Poisson rate) | #20, #21, #25, #27 | Top | Deep | Changes the core likelihood; the multi-sample overlap/double-counting question is the hard part, and the inclusion operator needs #27's spuriousness model to evaluate "not spurious" on mocks. |
| 24 | **Sample-reproduction diagnostics** (published-N waterfall per sample, cut-by-cut attrition) | #20, #21, #25 | Mid | Standard | Well-specified reporting; this is the artifact that proves the reproduction path works. |
| 25 | **El-Badry et al. (2026) selection** (`config/selections/elbadry2026.yaml` + module; two parent branches) | #17, #19, #20, #26 | Top | Deep | Two parallel branches (astrometric + SB1) with different mass-function machinery; the SB1 branch is new to this pipeline and subsample 3 imports Andrews' evaluated membership (§8.7). |
| 26 | **SB1 spectroscopic mass-function support** (`SB1`/`SB1C` ingestion, `K1`/`σ_K1`/`e`, `f_m` primitive) | #18 | Mid | Standard | Well-specified ETL plus one closed-form statistic. Scope is now settled (§8.4.1): reproduction and validation only, no inference entry point in v1. |
| 27 | **Spuriousness model** (`spuriousness_model.py`, `config/spuriousness_model.yaml`; shared `P(spurious \| ·)`) | #18, #19 | Top | Deep | Sample-independent model replacing three per-sample constants (§4.8). 293 labeled sources are staged. Covariate set must come from the sensitivity-analysis module, not be hand-picked; censoring of the 33 `undetermined` rows must be modeled, not dropped. Blocks the #23 inclusion operator. |

Review/Integration (#15) continues, merging `config/fragments/` into `config/config.yaml` at each
checkpoint as before.

**Sequencing.** #17 and #18 first and in parallel (both blocking). Then #19 and #26 in parallel.
Then #20, #21, and #27 in parallel (≤3 concurrent per `ORCHESTRATION_PLAN.md` §1) — #22 is **not** in
this wave, being blocked on user input (§9). Then #25, which needs #20 merged because it imports
Andrews' evaluated membership (§8.7). Then #23 and #24, which need every unblocked sample landed;
#23 additionally needs #27.

**Blocked work.** #22 does not start until the user supplies the accel/jerk selection function.
#23 and #24 proceed without it rather than waiting; the registry (§12.1) simply carries `accel_jerk`
disabled until then.

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
├── spuriousness_model.yaml           # NEW: sample-INdependent; deliberately not under selections/
└── selections/                       # NEW: one frozen file per published sample
    ├── andrews2022.yaml              # frozen reproduction path, N = 24
    ├── andrews2022_modified.yaml     # restores Gaia BH1, N = 25; feeds inference (§6.6)
    ├── elbadry2024.yaml
    ├── elbadry2026.yaml
    ├── accel_jerk.yaml
    └── external/                     # frozen published tables, supplied by the user
        ├── janssens2022_mass_magnitude.yaml  # MG-mass fit parameters (§8.2)
        ├── elbadry2023_table_e1.yaml         # BH candidates: membership + labels (§4.8)
        ├── elbadry2024_table3.yaml           # all NS candidates: labels (§4.8)
        ├── elbadry2026_table7.yaml           # astrometric branch: membership + labels
        └── elbadry2026_table8.yaml           # SB1 branch: membership + labels
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

### 4.8 Spuriousness is one shared model, not a per-sample constant

> **This supersedes the per-sample contamination-constant design.** Earlier drafts of this document
> stored each paper's quoted spurious fraction as a *modeled parameter* in that sample's selection
> file. That was wrong, and the rest of this section explains why and what replaces it.

#### The problem with per-sample rates

The literature gives us three numbers: **~25%** of El-Badry 2024's good-quality-flag candidates were
spurious (§7.3); **~40%** for El-Badry 2026's astrometric branch and **~50%** for its SB1 branch
(§8.6). Treating each as a parameter attached to its sample confounds two different things:

1. the **underlying propensity** of a Gaia astrometric or spectroscopic solution to be spurious,
   which is a function of the source's physical and observational properties; and
2. the **particular cut chain** each paper applied, which reweights that propensity by preferentially
   admitting sources from regions of parameter space where it is high or low.

El-Badry 2024 says this explicitly: the spurious fraction among their candidates is *higher* than
among astrometric binaries generally, **because spurious solutions are over-represented where genuine
binaries are rare**. That is a statement about how a selection interacts with an underlying rate, not
about an intrinsic sample property. A sample does not *have* a spurious rate; a sample's spurious
rate is *produced* by pushing its cuts through the underlying propensity.

Two further consequences make the per-sample-constant design untenable:

- **A constant cannot be applied to a mock.** The inclusion operator (§4.7) has to evaluate, for each
  mock realization, whether that realization would have been flagged spurious. A scalar attached to a
  real sample has nothing to condition on. A parametric function of observables does.
- **Three samples give three numbers and no cross-checks.** Three constants fit three observations
  exactly and predict nothing. One shared model fit to the same information is heavily
  over-determined, and reproducing all three rates becomes a real test.

#### The replacement

Specify a single, **sample-independent** model:

```
P(spurious | physical and observational parameters)
```

Every sample's selection integrates over it. For sample `s` with cut chain `C_s`, the predicted
aggregate rate is

```
f_spurious(s) = E_{x ~ p(x | C_s)} [ P(spurious | x) ]
```

i.e. the expectation of the shared model over the parameter distribution of sources that survive that
sample's cuts. One model, three (and later, arbitrarily many) predicted rates.

#### Candidate covariates

The features below are the ones the literature and the labeled data (see "Labeled data" below, and
§8.3) actually implicate. They are a **candidate set to test, not a hand-picked final set** — see
the sensitivity requirement immediately after.

| Covariate | Why it plausibly drives spuriousness |
|---|---|
| `period_days`, and proximity to **Gaia scanning-law harmonics** | Andrews excluded a source at `P = 186 d ≈ 3 ×` the scanning period on exactly this basis (§6.4); Halbwachs et al. (2022) identify large-`m_f` solutions at scanning-law harmonics as probable contaminants. This should enter as a *distance to nearest harmonic*, not raw period. |
| `goodness_of_fit` (F2) | Both El-Badry papers cut on it; §8.9 shows two Simon et al. sources excluded by `F2 > 10` alone. |
| Parallax S/N `varpi / sigma_varpi` | Spurious solutions cluster at low astrometric S/N. |
| `a0 / sigma_a0` | Directly controls whether the photocenter wobble is a detection. |
| `phot_g_mean_mag` | Sets per-epoch astrometric precision; also the axis along which the samples' `G < 15` cuts bite. |
| Implied `m_f` or `M̃2` | The `122 ± 47 M☉` and `119 ± 71 M☉` rows of Table E1 are verdicted spurious essentially on implausibility of the implied mass. |
| `visibility_periods_used` | Few visibility periods → poorly constrained orbit → spurious solution. |
| **RV consistency**: observed vs. expected RV semi-amplitude | The single sharpest discriminant in the labeled data (below). Available only where follow-up RVs exist, so it must be modeled as *missing-at-selection*, not imputed. |

> **Do not hand-pick the covariate set.** Per `dark-hunter-pop-workflow`, every population class is a
> rate function of the relevant covariates, with covariates admitted **only where the sensitivity-
> analysis module shows they matter**. Spec running that module over the candidate set above and
> retaining what it justifies. A covariate that survives because it was in this table rather than
> because the sensitivity analysis kept it is a specification failure.

#### Labeled data (§15 Q13 resolved)

**Four published tables carry per-source spuriousness labels**, all user-supplied and all staged
under `config/selections/external/`. Together they are the training and validation set for this
model.

| Fixture | Source | Rows | genuine | spurious | undetermined |
|---|---|---|---|---|---|
| `elbadry2023_table_e1.yaml` | El-Badry 2023a Table E1 — **BH candidates** | 6 | 1 | 4 | 1 |
| `elbadry2024_table3.yaml` | El-Badry 2024 Table 3 — **all NS candidates** of Andrews 2022 + Shahaf 2023b | 60 | 24 | 12 | 24 |
| `elbadry2026_table7.yaml` | El-Badry 2026 Table 7 — astrometric branch | 76 | 46 | 23 | 7 |
| `elbadry2026_table8.yaml` | El-Badry 2026 Table 8 — SB1 branch | 151 | 126 | 24 | 1 |
| **Total** | | **293** | **197** | **63** | **33** |

Table E1 is **complete at six rows** — it covers only the BH candidates, which is why it is small.
The NS candidates are in El-Badry 2024 Table 3, and the 2026 program's full astrometric and SB1
candidate lists are Tables 7 and 8.

This spans **BH and NS candidates across three papers with three different selections**, which is
precisely what a sample-independent model needs. Each table carries the observables alongside the
label: Table E1 has `G`, `M̃`, `P_orb`, `a0 × d`, `GoF`, and observed vs. expected RV amplitude;
Table 3 has `P_orb`, `G`, and `GoF`; Tables 7 and 8 have `P_orb`, `e`, the AMRF or `f_m`, `M̃1`,
`M̃2`, **significance**, `G`, and `E(B−V)`.

> **Is 293 enough for a multi-covariate fit?** **Yes, with caveats** — this supersedes the earlier
> assessment that the label set was too small. 63 positives against roughly 8 candidate covariates
> is on the order of 8 events per covariate, which is workable for a regularized fit but not for an
> unregularized one, and it is the *positive* count that binds, not the total. Concretely: spec
> regularization, spec cross-validated covariate selection through the sensitivity module rather
> than stepwise addition, and report per-covariate effective sample size. Do not fit interaction
> terms beyond the one the literature explicitly motivates (`F2 × G`, below) without the module
> justifying them. The three-way rate reproduction remains the real acceptance test; in-sample fit
> quality on 293 rows is not.

#### Two label axes, not one

The tables label two different things, and collapsing them would corrupt the model:

- **`verdict`** — is the *Gaia solution* reliable? `genuine` / `spurious` / `undetermined`. **This
  is what the spuriousness model predicts.**
- **`nature`** — what is the companion? Compact object, massive WD, post-mass-transfer binary,
  hierarchical triple, SB2, Be star. **This belongs to `companion_nature_likelihood`, not here.**

An "ultramassive WD" row in Table 3 is a *genuine* astrometric solution whose companion is not an
NS. A "two-temperature SED" row in Table 8 is a genuine SB1 solution with a luminous companion.
Neither is a spurious solution. Both fixtures carry the axes as separate columns; keep them separate.

#### Three-valued labels and censoring

The `undetermined` rows are **censored, not missing at random**, and the censoring mechanism depends
on covariates the model uses. Table 3's breakdown makes this concrete:

| Censor reason | N | Depends on |
|---|---|---|
| `magnitude_limit` (fainter than G = 15) | 12 | **`G`, a covariate** |
| `campaign_incomplete` | 8 | Observing time, roughly exogenous |
| `deprioritized_by_goodness_of_fit` | 2 | **`F2`, a covariate** |
| `rv_precision_vsini` | 2 | Stellar rotation, correlated with mass and `G` |

El-Badry 2026 states the mechanism outright: *"Sources with spurious solutions were preferentially
dropped from follow-up, and thus a smaller fraction of them have complete orbits."* The censoring is
**correlated with the outcome being censored**.

> **Requirement.** §4.8 must model censoring explicitly. Do **not** drop `undetermined` rows.
> Dropping them biases the model against exactly the faint, high-`F2`, low-significance regime where
> spuriousness is highest — the 12 rows censored by `G > 15` and the 2 censored by poor `F2` are the
> most informative rows in Table 3 about where the model should predict high spuriousness, and
> discarding them would make the fitted rate an underestimate precisely where it matters for DR4.

The defensible treatment, and the one specified here: fit the label and the censoring **jointly**,
as a two-part model in which `P(observed verdict | x)` and `P(spurious | x)` share covariates. This
is the standard Heckman-style selection correction, and it is the right choice here because the
censoring indicator is *known* for every row and its drivers (`G`, `F2`) are already in the
covariate set — so the selection equation is identified by functional form plus the exogenous
`campaign_incomplete` variation, without needing an instrument we do not have.

Two alternatives were considered and rejected. Treating `undetermined` as a third outcome in a
multinomial model is simpler, but it conflates "we do not know" with a physical state and yields a
`P(spurious | x)` that cannot be evaluated on a mock, which defeats the purpose. Imputing the
missing verdicts from the fitted model is circular. If #27 finds the joint model unidentified in
practice, escalate rather than falling back to dropping rows.

#### What the literature already says about the covariates

El-Badry 2026 §5.1.1 makes several quantitative claims that the model should be checked against —
these are **prior evidence, not a substitute for the sensitivity analysis**:

- *"Most sources in our sample with `F2 > 6` and `G < 13` turned out to have spurious orbits, as did
  sources with `F2 > 4` and `G > 13`."* The `F2` threshold **moves with `G`**, with the break at the
  Gaia window-class change at `G = 13`. This is the one interaction term with explicit literature
  motivation.
- *"A majority of sources with significance > 20 have reliable orbits, while spurious sources
  preferentially have lower significance."* Confirmed in the staged fixture: of Table 7's
  adjudicated rows, **11.9% are spurious above significance 20 versus 66.7% below, and 73.3% below
  significance 10.** Significance is the strongest single covariate in the data we have.
- *"The fraction of spurious orbits does not depend strongly on `G` or `P_orb` within our sample."*
  Marginally weak — but note "within our sample", i.e. after selection. Both may still matter
  unconditionally, which is exactly the distinction §4.8 exists to make.
- *"There is no simple set of cuts that cleanly divides reliable and spurious orbits."* An explicit
  statement that a threshold-based treatment is inadequate and a probabilistic model is required.
- *"The fraction of spurious orbits in our astrometric sample is very likely higher than in the full
  DR3 astrometric orbit catalog: true solutions with high `f_m` are rare, so a small number of
  spurious solutions will represent a disproportionate fraction of the total sample at high `f_m`."*
  **This is the §4.8 thesis in the authors' own words**, and it identifies `f_m` as the covariate
  through which the selection dependence primarily operates.

#### The Gaia BH1 label conflict — resolved

Table E1 verdicts `4373465352415301632` genuine; Andrews et al. (2022) excluded it as a probable
scanning-law harmonic. **The source is Gaia BH1** (El-Badry 2026 Table 7 names it), so Andrews was
wrong and Table E1 was right (§6.6, §15 Q14 resolved). It stays in the training set labeled
**genuine**, with `label_conflict` recording the disagreement, and `distance to nearest scanning-law
harmonic` remains a candidate covariate — now known to be insufficient on its own.

A second, quieter resolution worth noting: Table E1 verdicts `5870569352746779008` *uncertain*, and
Table 7 identifies it as **Gaia BH2**. Two of Table E1's six rows are therefore now known to be
genuine confirmed black holes, and the remaining four are confirmed spurious by Table 7 and Simon
et al. (2026). The cross-table label agreement is exact where the tables overlap, which is a real
consistency check on the transcription and on the taxonomy above.

#### Validation targets, not inputs

Each paper's quoted rate becomes an **acceptance test**:

| Sample | Population the rate applies to | Target | Recovered from the staged fixtures |
|---|---|---|---|
| El-Badry 2024 (§7.3) | Candidates with good astrometric quality flags | ~25% | **25.0%** — 12 spurious of the 48 Table 3 rows with `G < 15` |
| El-Badry 2026 astrometric (§8.6) | The 76-source astrometric branch | ~60% reliable | **60.5%** — 46 genuine of 76 in Table 7 |
| El-Badry 2026 SB1 (§8.6) | The 151-source spectroscopic branch | ~50% spurious | **Not row-recoverable** — see §15 Q15 |

Integrating the one shared model over each sample's selection must reproduce these. **Reproducing
them from a single model is a strong check** — the selections differ in magnitude limit, mass floor,
goodness-of-fit treatment, and (for SB1) in the detection modality itself, so agreement is not
achievable by tuning an overall normalization.

The first two targets are now **exactly** recoverable from the staged label fixtures, which is a
strong signal that the label taxonomy above is the one the papers used: 12/48 reproduces the
published "about a quarter" to three digits, and 46/76 reproduces the published "∼60% … have
reliable orbital solutions and indeed host compact objects" to within half a point. The SB1 target
does not reduce this way and is treated separately under §15 Q15.

The rates stay recorded in each per-sample selection file, but **relabeled**: they live under a
`validation_targets:` block, explicitly as outputs to be reproduced, never read by the selection or
likelihood code. A rate that is read as an input is a bug.

#### Where it lives

- **Model:** `spuriousness_model.py`, config at `config/spuriousness_model.yaml`. It is
  sample-independent, so it does **not** live under `config/selections/`.
- **Consumers:** the Poisson inclusion operator (§4.7), which evaluates it per mock realization; and
  the §7.3 / §8.5 outcome-dependent criteria, whose "not spurious" conditions this model supplies.
- **Roster:** #27 (§3), a **dependency of #23**, the likelihood-integration work.

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
- The ADQL restricts `nss_solution_type` to the astrometry-only solution types.

> **Working hypothesis (user-supplied, unverified).** The parent is **all 12-parameter (orbital)
> astrometric solutions that are not SB1/SB2** — i.e. the purely astrometric orbital solution types,
> excluding the spectroscopic (`SB1`, `SB1C`, `SB2`, …) and astro-spectroscopic
> (`AstroSpectroSB1`) types. This matches the paper's own framing ("detected using astrometry only",
> avoiding spectroscopic and photometric binaries) and is consistent with §7, where El-Badry 2024's
> inclusion of `AstroSpectroSB1` is called out as a *difference* from Andrews.
>
> The user expressed this with explicit uncertainty, so **it is a starting point for #20, not a
> settled fact.** Write it into `andrews2022.yaml` as the initial `nss_solution_type` set, then
> verify.

> **Blocking acceptance test.** The parent query must return **exactly 134,598** rows against the
> Gaia archive before *any* downstream Andrews cut is trusted. This gates the whole sample: an
> attrition waterfall that starts from the wrong parent can still land on 24 by coincidence of
> compensating errors, and that would be worse than failing. Concretely, #20 must:
>
> 1. Start from the working hypothesis above.
> 2. Run the count against the archive. If it is not 134,598, iterate over the solution-type set —
>    and **only** the solution-type set; do not add ancillary quality cuts to reach the number.
> 3. Freeze the verified set in `andrews2022.yaml`, together with the **literal archive query that
>    confirmed it** and the returned count, under the file's `provenance` block.
> 4. If no solution-type set reproduces 134,598, **STOP and escalate** with the counts tried. Do not
>    proceed to the cut chain.

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
  # Working hypothesis: 12-parameter orbital astrometric solutions, excluding SB1/SB2 and
  # AstroSpectroSB1. MUST be verified to return exactly 134,598 before the cut chain is trusted.
  adql: |
    SELECT ...
    FROM gaiadr3.nss_two_body_orbit AS nss
    WHERE nss.nss_solution_type IN ( ... astrometry-only types ... )
  expected_parent_n: 134598
  verification:
    status: unverified          # -> verified once the archive returns 134598
    confirmed_count: null
    confirming_query: null      # literal ADQL text that returned the count
    verified_on: null
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

### 6.6 Modified variant: `andrews2022_modified` (N = 25)

**File:** `config/selections/andrews2022_modified.yaml` · **Roster:** #20 · **§15 Q14 resolved.**

Andrews' exclusion of `4373465352415301632` was **wrong**, and we now know this with unusual
certainty: **that source is Gaia BH1**, the first dormant black hole discovered by Gaia
(El-Badry et al. 2023a). El-Badry 2026 Table 7 lists it explicitly as `Gaia BH1`, and El-Badry
2023a Table E1 verdicts it genuine (`GoF = 0.3`, no discrepant RV). Andrews removed it because
`P = 186 d` sits near three times Gaia's scanning period and the `m_f ≈ 11.6 M☉` looked too large —
the period looked too perfect to be a coincidence. It was not a coincidence; it was a black hole.

This is the single best cautionary example in the whole document, and the architecture should
preserve it rather than quietly fix it.

**Two files, both first-class.**

| Sample | Excludes Gaia BH1? | N | Role |
|---|---|---|---|
| `andrews2022` | Yes, as published | **24** | Reproduction only. Frozen. Proves we can reproduce the paper. |
| `andrews2022_modified` | **No** | **25** | Feeds forward-modeling and inference. |

The frozen `andrews2022` file is **not** edited. Reproduction fidelity is the entire purpose of
that path, and a reproduction path that silently corrects the paper reproduces nothing. The
modified variant is a **separate named sample in the §12.1 registry**, not a flag buried inside the
frozen preset, so both can be enabled together and compared directly by the `mode_divergence` and
`sample_overlap_matrix` diagnostics.

```yaml
schema_version: 1
name: andrews2022_modified
provenance:
  reference: "Andrews et al. (2022), with one documented correction"
  derived_from: config/selections/andrews2022.yaml
  published_n: null          # no published N; this variant is ours
  expected_n: 25
inherits: andrews2022        # identical parent query and cuts 1-5
mode: forward_model          # the inference-facing variant
exclusions: []               # cut 6 (harmonic_exclusion) is NOT applied
correction:
  restores_source_id: 4373465352415301632
  identification: "Gaia BH1 (El-Badry et al. 2023a)"
  rationale: >-
    Andrews et al. (2022) removed this source as a probable Gaia scanning-law harmonic
    contaminant (P = 185.8 d, ~3x the scanning period, m_f ~ 11.6 Msun). It is in fact
    Gaia BH1, confirmed by El-Badry et al. (2023a) and listed as such in El-Badry et al.
    (2026) Table 7. The exclusion was a false positive.
```

**Why not the other option.** The alternative considered was to keep one Andrews sample and replace
the single-source exclusion with a general rule — reject any period near a multiple of one year.
That was **rejected**, for two reasons:

1. **It would discard genuine systems.** Gaia BH1 is the proof: a blanket harmonic cut removes
   real black holes. Andrews' one-source version of that rule already produced one false positive
   out of one application.
2. **It hard-codes the very thing the model is supposed to learn.** Whether proximity to a
   scanning-law harmonic predicts spuriousness, and how strongly, and in combination with what
   else, is a question for the §4.8 spuriousness model and the sensitivity analysis — not for a
   threshold written into a selection file. Encoding it as a cut would make the answer an
   assumption and remove it from the evidence.

**Consequences for §4.8.** Gaia BH1 stays in the labeled training set with verdict **genuine**, and
its `label_conflict` block records Andrews' disagreement. `distance to nearest scanning-law
harmonic` remains a **candidate** covariate for the sensitivity module to adjudicate — this episode
does not remove it from consideration, but it does establish that it is **not sufficient on its
own**, since the strongest single instance of the pattern is a true positive for the population and
a false positive for the cut.

**Consequence for §8.3.** El-Badry 2026's astrometric subsample 3 imports Andrews' *published*
membership, so it must resolve against the frozen **`andrews2022`**, never the modified variant.
Getting this backwards would break the `16`-source expectation of §15 Q9. Note that Gaia BH1 enters
El-Badry 2026 through subsample 2 (Table E1) regardless, so the union of 76 is unaffected.

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
>   solutions are over-represented where genuine binaries are rare.
>
>   Per **§4.8**, the 25% is **not** a parameter of this sample. It is what you get when this
>   sample's cut chain is integrated over the shared `P(spurious | ·)` model, and the paper's own
>   explanation for why it exceeds the general rate — over-representation where genuine binaries are
>   rare — is precisely a statement that the underlying rate depends on where in parameter space the
>   cuts land you. Record 25% in `elbadry2024.yaml` under `validation_targets:` as a number to be
>   reproduced, and satisfy criterion (c) by evaluating the shared model.

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

> **v1 scope boundary (decided; §15 Q6 resolved).** Both branches get a **reproduction** path and
> must match their published counts. **Only the astrometric branch (76) has an inference entry
> point in v1.** The SB1 branch (151) is reproduction-and-validation only: it proves our cut chain
> and catalog handling are correct, but it does not contribute to the population likelihood. See
> §8.4.1 for the reason and for exactly what a v2 would have to add.
>
> Per `dark-hunter-pop-workflow` §7, this is a **scope boundary decision, not a limitation to route
> around**. Do not silently upgrade the SB1 branch to an inference input mid-implementation; that
> is a separately scoped v2 decision.

### 8.1 Parent samples and shared preprocessing

Both branches draw from `gaiadr3.nss_two_body_orbit`:

| Branch | `nss_solution_type` | Published N | v1 paths |
|---|---|---|---|
| Astrometric | `Orbital`, `AstroSpectroSB1` | 76 | reproduction + forward_model + inference |
| Spectroscopic | `SB1`, `SB1C` | 151 | reproduction only (§8.4.1) |

The astrometric branch's solution types match §7 exactly. The SB1 branch is **new machinery for this
pipeline**; roster #26 builds the ingestion and the `f_m` primitive it needs.

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
`MG,0`–mass relation of Janssens et al. (2022)**. The relation is **user-supplied** (§15 Q8
resolved) as the paper's Table 1 fit parameters; do not fetch, digitize, or re-fit it.

#### Functional form (verified, not assumed)

The published relation gives **magnitude as a function of mass**, piecewise in eight mass regimes:

```
M_G = a * log10(M / M_sun) + b
```

This form was verified against the supplied Table 1 and Figure before being written down:

- **Solar anchor.** The `0.87–1.55 M☉` segment gives `M_G(1 M☉) = b = 4.73`, against an observed
  solar `M_G ≈ 4.67`.
- **Continuity.** Evaluating both adjoining segments at each of the seven internal mass boundaries
  agrees to **≤ 0.0094 mag** (worst case at 1.55 and 1.8 M☉; four of seven agree to < 0.005 mag).
  The piecewise fit is continuous to within its own precision.
- **Figure endpoints.** The form predicts `M_G = 11.90` at `0.2 M☉` and `M_G = −5.96` at
  `57.95 M☉`, matching the figure's span of roughly `+11.5` to `−6` over `0.2–60 M☉`.
- **Monotonicity.** Every `a < 0`, so `M_G` decreases strictly with mass across the full range,
  which makes the inversion single-valued with no branch ambiguity.

#### Fit parameters (Janssens et al. 2022, Table 1)

Staged at `config/selections/external/janssens2022_mass_magnitude.yaml`:

```yaml
schema_version: 1
name: janssens2022_mass_magnitude
provenance:
  reference: "Janssens et al. (2022), Table 1"
  description: "Fit parameters for the mass-magnitude relation of dwarfs in different mass regimes"
  supplied_by: user
  frozen_on: <ISO date>
form: "M_G = a * log10(M / M_sun) + b"
applies_to: dwarfs           # main-sequence only; see below
mass_range_msun: [0.02, 57.95]
segments:
  # m_low, m_up in M_sun; a, b with 1-sigma uncertainties
  - {m_low: 22.90, m_up: 57.95, a: -3.31,  a_err: 0.08, b: -0.12, b_err: 0.11}
  - {m_low: 15.55, m_up: 22.90, a: -4.22,  a_err: 0.43, b:  1.12, b_err: 0.59}
  - {m_low:  7.60, m_up: 15.55, a: -6.07,  a_err: 0.25, b:  3.33, b_err: 0.31}
  - {m_low:  1.80, m_up:  7.60, a: -6.83,  a_err: 0.06, b:  4.00, b_err: 0.12}
  - {m_low:  1.55, m_up:  1.80, a: -10.51, a_err: 5.66, b:  4.93, b_err: 1.45}
  - {m_low:  0.87, m_up:  1.55, a: -9.41,  a_err: 0.42, b:  4.73, b_err: 0.39}
  - {m_low:  0.60, m_up:  0.87, a: -19.58, a_err: 0.93, b:  4.11, b_err: 0.40}
  - {m_low:  0.02, m_up:  0.60, a: -7.23,  a_err: 0.21, b:  6.85, b_err: 0.43}
inversion:
  # M = 10 ** ((M_G - b) / a)
  segment_selection: by_mg_interval
  boundary_tolerance_mag: 0.01
  boundary_tie_break: lower_mass_segment
extrapolation: forbid
```

#### Inversion `M_G → M̃1`

We are given `M_G` and need mass, so the relation must be inverted:

```
M_tilde1 = 10 ** ( (M_G - b) / a )
```

**Segment selection.** Because the relation is monotonic in mass but we enter with magnitude, the
segment must be chosen by `M_G` interval, not by mass. Precompute each segment's magnitude range
from its mass bounds:

| Mass range [M☉] | `a` | `b` | `M_G` range |
|---|---|---|---|
| 0.02 – 0.60 | −7.23 | 6.85 | 19.134 … 8.454 |
| 0.60 – 0.87 | −19.58 | 4.11 | 8.454 … 5.294 |
| 0.87 – 1.55 | −9.41 | 4.73 | 5.299 … 2.939 |
| 1.55 – 1.80 | −10.51 | 4.93 | 2.930 … 2.247 |
| 1.80 – 7.60 | −6.83 | 4.00 | 2.256 … −2.016 |
| 7.60 – 15.55 | −6.07 | 3.33 | −2.017 … −3.904 |
| 15.55 – 22.90 | −4.22 | 1.12 | −3.909 … −4.619 |
| 22.90 – 57.95 | −3.31 | −0.12 | −4.621 … −5.956 |

**Boundary behavior.** The ≤ 0.0094 mag discontinuities mean adjacent `M_G` intervals overlap or
gap by up to ~0.01 mag. Resolve deterministically with `boundary_tolerance_mag: 0.01` and
`boundary_tie_break: lower_mass_segment`, and record boundary-resolved sources in the attrition
diagnostic so the count is visible rather than silent. The induced mass ambiguity at a boundary is
at most ~0.002 dex, far below the fit uncertainty, so the tie-break choice is not physically
consequential — but it must be *fixed* for the reproduction path to be deterministic.

**Extrapolation.** `M_G` outside `[−5.956, 19.134]` — i.e. implied mass outside `0.02–57.95 M☉` —
is **not applicable**, not clamped to an endpoint. Combined with the main-sequence restriction
below, this is a second route by which `M̃1` can be undefined, and it must propagate as
"cut not applicable" rather than "cut failed" (§15 Q10).

**Main-sequence only.** The relation is a *dwarf* mass-magnitude relation and, per §8.1, `M̃1` is
computed only for sources passing the main-sequence cut. For evolved candidates `M̃1` does not
exist, so every cut expressed through `M̃1` or `M̃2` is undefined rather than false.

#### Uncertainty propagation

Differentiating the inversion at fixed `M_G`:

```
sigma_log10M^2 = (sigma_b / a)^2 + (log10(M) * sigma_a / a)^2 + cross-term
```

The `a`–`b` covariance is not published, so the cross-term cannot be evaluated (§15 Q11). Spec the
implementation to accept a configurable correlation coefficient defaulting to zero, and to report
that the default was used.

> **The `1.55–1.80 M☉` segment is nearly uninformative and must be called out.** Its `a = −10.51 ±
> 5.66` is a 54% fractional uncertainty, and `b = 4.93 ± 1.45`. Propagated, a source landing in
> this segment carries `σ_log10(M) ≈ 0.19 dex` — roughly a 50% mass uncertainty, and **wider than
> the segment itself**, which spans only 0.065 dex in mass. Any candidate whose `M̃1` lands here
> should be flagged in the diagnostic output, and the forward-model path should not treat its
> `M̃1` as informative.

**Which path propagates what.** The **reproduction** path uses the central `a`, `b` values only —
the published counts were produced that way, and injecting extra scatter would not reproduce them.
The **forward-model** path ignores this block entirely and uses the pipeline's own TAG10/MSC mass
posterior (§4.3). The `a`/`b` uncertainties therefore matter in exactly one place: the
`σ_M̃2 ≤ 0.105 M☉` cut of astrometric subsample 4.

**Decided (§15 Q12 resolved):** that `σ_M̃2` uses the **astrometric covariance only** and
**excludes** the Janssens fit uncertainty. Running it both ways stays a *recommended* validation
when #25 executes — if the astrometric-only variance fails to recover 22 sources, that points to a
problem elsewhere in the chain rather than to this choice — but it is not an open decision.

```yaml
primary_mass:
  method: janssens2022_mass_magnitude
  table: config/selections/external/janssens2022_mass_magnitude.yaml
  propagate_fit_uncertainty: false     # confirmed correct; §15 Q12 resolved
  ab_correlation: 0.0                  # unpublished; see §15 Q11
  applies_only_when: "main_sequence == true"
  caveats:
    - "Photometric only; does not model each source's detailed evolutionary state."
    - "Does not model light contributions from possible luminous companions."
    - "Suitable for initial candidate selection, not for final masses."
    - "1.55-1.80 Msun segment is nearly uninformative (a = -10.51 +/- 5.66)."
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
- **Table E1 is user-supplied and dual-purpose (§15 Q8, Q13 resolved).** Staged verbatim at
  `config/selections/external/elbadry2023_table_e1.yaml`, preserving every quoted uncertainty; do
  not fetch or re-derive it. It is **complete at six rows** — it covers only the BH candidates.
  Beyond supplying this subsample's membership, its covariate columns and `Verdict` labels make it
  one of the four **labeled sets for the §4.8 spuriousness model**, so the file is referenced from
  two places and must not be trimmed to just the source IDs. Its `✓` verdict for
  `4373465352415301632` was **correct and Andrews' exclusion was wrong** — that source is Gaia BH1
  (§6.6, §15 Q14).
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

#### 8.4.1 v1 scope boundary — the SB1 branch is reproduction-only

**Decided (§15 Q6 resolved).** In v1 the spectroscopic branch has:

- a **reproduction path**, which must recover all **151** sources and the `136 / 30 / 15` route
  breakdown exactly;
- a **validation role**, since reproducing 151 proves our SB1 ingestion, `f_m` primitive, and
  main-sequence gating are correct, and the branch supplies one of the three §4.8 spuriousness
  validation targets;
- **no inference entry point.** It contributes nothing to the population likelihood. Only the
  astrometric branch's 76 sources do.

**Why the boundary is here.** The pipeline's mass derivation is astrometric throughout
(`ARCHITECTURE.md` §4). An astrometric mass function constrains `M2` directly (§5.3); a
spectroscopic one constrains only `M2 sin³i` (§8.4). Feeding SB1 systems into the mass function
therefore requires **marginalizing over inclination** — with an inclination prior that is itself
selection-dependent, because edge-on systems are preferentially detected as SB1s. That is a distinct
piece of statistical machinery, not a configuration change.

**What a v2 would have to add**, recorded so the reason for the boundary stays on file:

1. A `sin³i` marginalization for the spectroscopic mass function.
2. A selection-aware inclination prior consistent with the SB1 detection probability.
3. Joint treatment of the 2 sources appearing in both branches, whose inclination is constrained
   astrometrically and spectroscopically at once.

> Per `dark-hunter-pop-workflow` §7 this is a **scope boundary, not a limitation to route around**.
> The #25 and #26 subagents must not upgrade the SB1 branch to an inference input mid-implementation
> because it "seems to work"; that is a separately scoped v2 decision requiring human sign-off.

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

### 8.6 Purity and contamination — recorded validation targets

**Per §4.8, these are outputs to be reproduced, not parameters to be used.** The paper's two
branch-level rates are recorded here so the shared spuriousness model can be tested against them; no
selection or likelihood code may read them as inputs.

```yaml
validation_targets:
  spurious_fraction:
    astrometric: 0.40         # ~40%: reproduce by integrating the shared model over this branch
    spectroscopic: 0.50       # ~50%: same, over the SB1 branch
    source: "El-Badry et al. (2026)"
    role: acceptance_test     # NEVER an input; see CONTINUATION_PLAN.md 4.8
  purity:
    astrometric_reliable_and_compact_object_fraction: 0.60
  dominant_residual_contaminants:
    - post_mass_transfer_binaries
    - hierarchical_triples
```

That the two branches differ by 10 percentage points **within a single paper, single program, and
single follow-up campaign** is itself the argument for §4.8: nothing about "the sample" changed
between them except the detection modality and the cut chain. Together with §7.3's ~25%, we have
three rates spanning 25–50%, and the shared model has to produce all three from one set of
parameters.

Physical context the model will need to accommodate: the astrometric branch yields the two known
Gaia BHs, 27 NS candidates, and a dozen massive WDs, and shows that **tight WD+WD binaries can
masquerade as NSs** — a contamination channel distinct from spuriousness, and one our
`companion_nature_likelihood` stage should be checked against. The SB1 branch's `M̃2,min > 3 M☉`
candidates were **all** luminous binaries or spurious solutions, and the astrometric branch's
`M̃2 > 2 M☉` candidates other than the two confirmed BHs were refuted or revised downward. Both
observations say the same thing the labeled Table E1 rows say (§4.8): **implausibly large implied
mass is itself a strong spuriousness covariate.**

### 8.7 Cross-sample dependencies

This sample is **not independent of the others**, which has consequences for both the registry and
the likelihood:

- Astrometric subsample 3 **is** the Andrews et al. (2022) selection restricted to `G < 15`. The
  registry must therefore support one selection file depending on another's evaluated membership,
  and `elbadry2026` must declare `depends_on: [andrews2022]`.
- Astrometric subsample 2 depends on **Table E1 of El-Badry et al. (2023a)**, staged as a frozen
  fixture (§15 Q8 resolved).
- Subsample 3 resolves against the **frozen `andrews2022`** (N = 24), *not* the modified variant of
  §6.6 — the paper imported Andrews' published membership, so importing our corrected version would
  break the expected count of 16 (§15 Q9). Gaia BH1 still enters this sample, via subsample 2.
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
inference_branches: [astrometric]   # SB1 is reproduction-only in v1 (8.4.1)
primary_mass:
  method: janssens2022_mass_magnitude
  table: config/selections/external/janssens2022_mass_magnitude.yaml
  propagate_fit_uncertainty: false
  ab_correlation: 0.0
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
        # Same file is registered as spuriousness_model.labeled_sets[] (4.8); do not trim
        # it to source IDs only.
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
    inference: false            # reproduction + validation only in v1 (8.4.1)
validation_targets:
  spurious_fraction:
    astrometric: 0.40
    spectroscopic: 0.50
    role: acceptance_test       # reproduced by the shared spuriousness model; never read as input
  purity:
    astrometric_reliable_and_compact_object_fraction: 0.60
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
| Primary mass | IsocLum, `gaiadr3.binary_masses` | Janssens et al. (2022) `M_G = a log₁₀M + b`, MS only |
| Extinction split | δ = −30°, Lallement 2022 | δ = −28°, Lallement 2019 |
| Companion-mass floor | 1.25 M☉ | 1.4 M☉ (primary), down to 1.05 M☉ (subsample 4) |
| Magnitude limit | G < 15 | G < 15 (both branches) |
| Goodness-of-fit cut | — | F2 < 10 (subsample 1 only) |
| Selection outcome-dependence | **Yes** — criteria (b)(c)(d) depend on follow-up | **No** — initial 227 is catalog-level only |
| Spurious rate (validation target, §4.8) | ~25% of good-quality-flag candidates | ~40% astrometric; ~50% SB1 |
| Inference entry point | Yes | Astrometric branch only (§8.4.1) |

The outcome-dependence row is the one that matters architecturally. El-Badry 2026's initial sample is
directly forward-modelable in a way El-Badry 2024's published sample is not, which makes it the
better anchor for the DR4 selection design the paper itself is aimed at. The spurious-rate row is no
longer a difference between the samples in any modeled sense — under §4.8 the three numbers are three
outputs of one model, and their spread is the test.

## 9. Sample: Acceleration/jerk stars — BLOCKED, PENDING USER INPUT

**File:** `config/selections/accel_jerk.yaml` · **Roster:** #22 · **Status: BLOCKED — the selection
is not yet published; the user will supply the selection function.**

Unlike §6–§8, this sample has **no published cut chain to transcribe and no published N to
reproduce**. The selection function itself does not yet exist in the literature. Everything below is
the architectural frame the eventual selection must fit into; the cut chain, thresholds, and parent
query are deliberately absent.

**Do not invent a selection from the existing `accel_jerk` target list.** The target list governs
*which systems we observe*; it is not a candidate-selection cut chain, and treating it as one would
manufacture a selection function that no paper applied and that we would then be forward-modeling
against nothing.

What is already settled, and will carry over unchanged when the selection arrives:

- **`mode` is `forward_model` only.** There is no reproduction path, because there is no published
  sample to reproduce. State this explicitly in the file so the dual-path machinery (§4.2) does not
  expect a `published_n`.
- Per `ARCHITECTURE.md` §4, the acceleration/jerk catalogs are matched **at the broad
  population/aggregate level only**, using the same `gaiamock` cascade to forward-model which systems
  land in which solution type. The selection function is the **solution-type occupancy** predicted by
  the cascade — the fraction of mock systems landing in the 7-parameter (acceleration) and
  9-parameter (jerk) bins — validated against the real catalog's aggregate fractions, reusing the
  existing solution-type-fraction diagnostic from `selection_function_astrometric`.
- The pool doubles as the RV follow-up target list; systems are tracked in a separate pending pool,
  promotable once their orbits resolve.
- The DR3/DR4 catalog identifiers stay path-specific
  (`dr3.selection_function_followup.accel_jerk_catalog_id`), consistent with
  `dark-hunter-pop-workflow` §6.
- Adoption dates for the follow-up-target aspect continue to come from
  `config/target_lists/derived/accel_jerk_adoption_dates.yaml`; this file will govern the *selection*
  aspect only, and must not duplicate those dates.

**Unblocking condition.** The user supplies the selection function. This section is then rewritten to
the §6–§8 structure (parent query, ordered cut chain with a cut table, validation targets, config
sketch), roster #22 and Slot F are unblocked, and `accel_jerk` is enabled in the §12.1 registry.

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
      mode: reproduction             # frozen; N = 24; never feeds inference
    - name: andrews2022_modified
      enabled: true
      path: config/selections/andrews2022_modified.yaml
      mode: forward_model            # restores Gaia BH1; N = 25; this one feeds inference (§6.6)
    - name: elbadry2024
      enabled: true
      path: config/selections/elbadry2024.yaml
      mode: reproduction
    - name: elbadry2026
      enabled: true
      path: config/selections/elbadry2026.yaml
      mode: reproduction
    - name: accel_jerk
      enabled: false                 # BLOCKED: selection function not yet supplied (§9)
      path: config/selections/accel_jerk.yaml
      mode: forward_model

spuriousness_model:                  # sample-INdependent (§4.8); not under sample_selection
  enabled: true
  path: config/spuriousness_model.yaml
  labeled_sets:                      # 293 labeled sources across three papers
    - config/selections/external/elbadry2023_table_e1.yaml
    - config/selections/external/elbadry2024_table3.yaml
    - config/selections/external/elbadry2026_table7.yaml
    - config/selections/external/elbadry2026_table8.yaml
  censoring: joint                   # model undetermined labels; never drop them (§4.8)
```

- Each sample is independently enable/disable-able, as required.
- `mode` is per-sample (§4.2), overridable per run.
- `spuriousness_model` sits **outside** the `samples` list on purpose: it is one model shared by all
  of them, and nesting it under any sample would reintroduce exactly the per-sample-constant design
  §4.8 replaces.
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
| `spuriousness_rate_reproduction` | Shared §4.8 model integrated over each enabled sample's selection, against that sample's recorded `validation_targets` | Reproduced from **one** parameter set: El-Badry 2024 ~25% (fixture gives 12/48 = 25.0%), El-Badry 2026 astrometric ~60% reliable (46/76 = 60.5%), SB1 per §15 Q15 |
| `spuriousness_covariate_sensitivity` | Output of the sensitivity-analysis module over the §4.8 candidate covariates: which are retained, which are dropped, and the effect of each on the predicted rates | Every retained covariate justified by the module, not by the candidate table; the `F2 × G` interaction reported separately since it is the one with literature motivation |
| `spuriousness_labeled_set_performance` | Shared model scored against all 293 labeled sources across the four fixtures, itemized per source and broken out per table | Reported with per-covariate effective sample size; cross-table agreement on overlapping sources checked; Gaia BH1 (§6.6) scored explicitly |
| `spuriousness_censoring_report` | The 33 `undetermined` rows: censor reason, fitted `P(spurious)`, and the shift in each sample's predicted rate between the joint-censoring model and a drop-the-rows baseline | The drop-the-rows bias quantified and shown to be non-negligible, justifying the joint treatment (§4.8) |
| `janssens_segment_occupancy` | Count of El-Badry 2026 sources whose `M̃1` lands in each Janssens mass segment, plus counts resolved at a segment boundary and counts falling outside `0.02–57.95 M☉` | Boundary-resolved and out-of-range counts reported, never silent; the `1.55–1.80 M☉` occupancy flagged (§8.2) |

A sample's reproduction path is **not considered working** until
`sample_reproduction_report` matches the published N exactly. Until then the sample must not be
enabled in `forward_model` mode for inference.

The spuriousness model is **not considered working** until
`spuriousness_rate_reproduction` reproduces all three published rates from a single parameter set.
Reproducing one or two is not partial credit — it is the signature of a model that has absorbed a
sample-specific normalization, which is the failure mode §4.8 exists to prevent.

## 14. Phase 8 prompts — orchestrator and subagent roster

Copy-pasteable working prompts, one per roster item plus the orchestrator. House style follows the
Phase 2 prompts: terse, imperative, dense; the reading list carries the detail so the prompt does
not have to.

**GitHub issue map** (roster → issue). Umbrella: [#103](https://github.com/UCSC-Transients/dark-hunter_pop/issues/103).
#22 was not opened (blocked, §9). Slot F still carries `TBD-22` by design.

| Roster | Slot | Issue |
|---|---|---|
| 17 | A | [#104](https://github.com/UCSC-Transients/dark-hunter_pop/issues/104) |
| 18 | B | [#105](https://github.com/UCSC-Transients/dark-hunter_pop/issues/105) |
| 19 | C | [#106](https://github.com/UCSC-Transients/dark-hunter_pop/issues/106) |
| 20 | D | [#107](https://github.com/UCSC-Transients/dark-hunter_pop/issues/107) |
| 21 | E | [#108](https://github.com/UCSC-Transients/dark-hunter_pop/issues/108) |
| 23 | J | [#112](https://github.com/UCSC-Transients/dark-hunter_pop/issues/112) |
| 24 | K | [#113](https://github.com/UCSC-Transients/dark-hunter_pop/issues/113) |
| 25 | H | [#110](https://github.com/UCSC-Transients/dark-hunter_pop/issues/110) |
| 26 | G | [#109](https://github.com/UCSC-Transients/dark-hunter_pop/issues/109) |
| 27 | I | [#111](https://github.com/UCSC-Transients/dark-hunter_pop/issues/111) |

Placeholders `#TBD-<roster>` in the prompts below have been substituted with these numbers.

Conventions carried from `ORCHESTRATION_PLAN.md`: one worktree/branch per subagent, PRs to `main`,
`Closes #N` alone on a line, ≤3 concurrent sessions. Skills active in every session:
`strict-workflow`, `regression-hunter`, `caveman`, `dark-hunter-pop-workflow` — referred to below as
"four skills". Required CI gate is `unit|physics|api`; long multi-draw Monte Carlo suites go under
`@pytest.mark.slow`.

Model slugs are in §2.1, effort contracts in §2.2. Each prompt states both in its opening line.

**Do-not-dispatch:** Slot F (#22) only. See §9.

---

### Orchestrator

```
You are the Phase 8 orchestrator for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

You do not write pipeline code. You open issues, dispatch subagents, gate on dependencies, run the
phase acceptance tests, and merge config fragments.

Read docs/CONTINUATION_PLAN.md §1-§4 and §12-§15, docs/ORCHESTRATION_PLAN.md §1 and §4,
docs/ARCHITECTURE.md §4 inference, four skills.

SETUP. Open one umbrella issue "Phase 8: per-sample literature selection functions" with children
for roster #17-#21 and #23-#27. Skip #22: blocked, see §9. Substitute the real issue numbers into
every #TBD- placeholder in §14 before dispatching anything.

DISPATCH ORDER (§3). Wave 1: #17, #18 in parallel — both blocking, nothing else starts.
Wave 2: #19, #26 in parallel. Wave 3: #20, #21, #27 in parallel (3 concurrent, the cap).
Wave 4: #25 — requires #20 merged, it imports Andrews' evaluated membership.
Wave 5: #23, #24. #23 additionally requires #27.
Never exceed 3 concurrent sessions. Never dispatch a slot whose dependencies are unmerged; the
prompts say so, but you enforce it.

INTEGRATION CHECKPOINT after every merge. Merge config/fragments/sample_selection.yaml into
config/config.yaml. Verify no numeric threshold leaked inline (dark-hunter-pop-workflow §1). Verify
no code path READS a per-sample spurious or contamination rate — those live under
validation_targets: and are outputs only (§4.8); grep every selection PR for it. Verify
config/spuriousness_model.yaml stayed OUTSIDE config/selections/. Run the DR3/DR4 key-independence
audit (§12.4). Confirm no frozen selection file changed without a schema_version bump.

PHASE ACCEPTANCE TESTS — the phase is not done until all pass (§13):
1. Andrews parent query returns exactly 134,598. BLOCKING for #20 and everything downstream.
2. Reproduction N: andrews2022 = 24, andrews2022_modified = 25, elbadry2024 = 21,
   elbadry2026 = 227 (76 astrometric + 151 SB1), with the §8.3 subsample split 47/3/4/22 and the
   §8.4 route split 136/30/15.
3. Simon et al. (2026) exclusion breakdown reproduces 5 / 2 / 1 / 1 (§8.9).
4. One spuriousness parameter set reproduces El-Badry 2024 ~25% and El-Badry 2026 astrometric
   ~60% reliable. SB1 is advisory pending §15 Q15.
5. Covariance health: zero silent drops. MC convergence at 1e4 draws reported.

ESCALATE TO HUMAN, do not decide: §15 Q1 (multi-sample overlap formulation), Q7 (nsstools vs our
conversion), Q14 follow-through if a subagent proposes re-litigating it, Q15 (SB1 denominator), and
any request to give the SB1 branch an inference entry point (§8.4.1 is a scope boundary).

Stop when all five acceptance tests pass, the umbrella issue's children are closed, and
config.yaml is merged and green.
```

---

### Slot A — sample-selection framework (#17)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/104
Branch: phase8/sample-selection-framework (worktree). PR → main. Closes #104 alone on a line.
Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §4, §12, §13, docs/ARCHITECTURE.md §4 mass_derivation_bulk and
selection_function_astrometric, four skills. Blocking for all of Phase 8 — nothing else starts
until this merges.

Implement sample_selection.py: SampleSelection interface, named-sample registry, per-sample
reproduction/forward_model mode switch (§4.2), ordered cut chain with per-cut attrition (§4.6),
parent-query-as-ADQL loading (§4.5), and cross-sample depends_on resolution (§8.7). Pydantic models
per §12.2 with extra="forbid" and an exhaustive-dispatch enum for mode. Register the stage per
§12.3 with the artifact path keyed on the content hash of every enabled selection file plus mode.

The cut evaluator MUST distinguish "cut not applicable" from "cut failed" as distinct states (§15
Q10). El-Badry 2026's M1-tilde is undefined for evolved sources and for magnitudes outside the
Janssens range; collapsing the two states into false makes the subsample counts irreproducible.
Carry both through the attrition waterfall.

Registry must support two variants of one paper as independent named samples (andrews2022 and
andrews2022_modified, §6.6) — not a flag inside one file.

Effort contract (Deep): you own the interface shape; every other slot builds against it. Written
justification in the PR. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot B — NSS covariance ingestion (#18)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (mid-tier, Standard effort).
Model: claude-4-sonnet

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/105
Branch: phase8/nss-covariance (worktree). PR → main. Closes #105 alone on a line.
Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §10, §11, docs/ARCHITECTURE.md §4 data_acquisition, four skills.
Runs in parallel with #17; no dependency.

Add nss.corr_vec and nss.bit_index to build_nss_adql. Reconstruct the full astrometric covariance
from corr_vec + bit_index + the per-parameter *_error columns per the DR3 NSS data model. Parameter
count varies by solution type — drive the unpacker from bit_index, never a hardcoded size. Store on
CandidateRecord as a named vector plus covariance with provenance, per the ParameterSet convention
(dark-hunter-pop-workflow §3). Persist in the data_acquisition HDF5 alongside the NSS panels.

Assert symmetry and PSD on load. Count failures, expose in the funnel diagnostic. NO diagonal-only
fallback: a solution whose covariance cannot be reconstructed is recorded and excluded from
MC-dependent samples, never silently downgraded to independent errors.

Escalate §15 Q4 (whether the archive columns suffice for every solution type, or matrices must be
staged from a local file) rather than deciding it.

Effort contract (Standard). Round-trip test against Gaia's published matrix layout. Full required
pytest before PR. Stop when PR open + CI green.
```

---

### Slot C — Monte Carlo mass-function propagation (#19)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/106
Branch: phase8/mc-mass-function (worktree). PR → main. Closes #106 alone on a line.
Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §5, §11, §13, four skills. Depends on #17 and #18 merged.

Land the §5 primitives in physics_utils.py: Thiele-Innes → a0 (§5.1), astrometric mass function
(§5.2), dark-companion inversion (§5.3), luminous-companion generalization (§5.4). Note the §5.4
transcription correction — the denominator is (1+F)^3, not (1+F1)^3.

Implement 1e4-draw MC propagation from the full 12x12 covariance to a per-system posterior over
(m_f, M2). Sampling from a near-singular covariance needs real care; state your factorization and
its fallback. Probability cuts evaluate on the ensemble; M2/sigma_M2 uses ensemble mean and
standard deviation, not linearized propagation. Seeds recorded in the run manifest.

Required convergence diagnostic (§13 m2_posterior_convergence): MC noise on P(M2 > threshold) must
be subdominant at 1e4 draws, and you must report how many systems sit within MC uncertainty of the
95% boundary. Reuse the existing mc_noise_threshold guardrail.

Effort contract (Deep): downstream probability cuts depend on this being right. Written
justification plus the convergence figure in the PR. Slow suites under @pytest.mark.slow. Full
required pytest before PR. Stop when PR open + CI green.
```

---

### Slot D — Andrews et al. (2022) selection (#20)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (mid-tier, Light effort).
Model: claude-4-sonnet

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/107
Branch: phase8/selection-andrews2022 (worktree). PR → main. Closes #107 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §4, §5, §6 (all of it, including §6.6), §11, four skills.
Depends on #17 and #19 merged.

BLOCKING FIRST TASK. Working hypothesis for the parent, user-supplied with explicit uncertainty:
all 12-parameter orbital astrometric solutions that are NOT SB1/SB2, excluding AstroSpectroSB1.
Run it. It must return exactly 134,598. If yes, freeze the solution-type set plus the literal
confirming ADQL and count under provenance.verification. If no, iterate over the SOLUTION-TYPE SET
ONLY — never add quality cuts to reach the number — and if nothing reproduces it, STOP and escalate
with the counts tried. Run no downstream cut until the parent verifies: a wrong parent can still
land on 24 by compensating errors, which is worse than failing.

Then transcribe §6.4 into config/selections/andrews2022.yaml: every threshold, fixed M1 = 1.0 Msun,
the no-extinction CMD cut, the excluded source. Target N = 24.

ALSO produce config/selections/andrews2022_modified.yaml per §6.6: inherits andrews2022, drops the
harmonic_exclusion cut, restores 4373465352415301632 — which is Gaia BH1, so Andrews' exclusion was
a false positive. Target N = 25. Register both as independent named samples; the frozen file is
reproduction-only, the modified one is what feeds inference. Do NOT edit the frozen file, and do
NOT generalize the exclusion into a period-near-harmonic cut (§6.6 records why).

Effort contract (Light): transcription plus the parent verification. You may not invent thresholds
or reinterpret a cut. If a published number cannot be reproduced, STOP and escalate with the
attrition table rather than tuning. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot E — El-Badry et al. (2024) selection (#21)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Standard effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/108
Branch: phase8/selection-elbadry2024 (worktree). PR → main. Closes #108 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §4, §4.8, §7, four skills. Source: arXiv:2405.00089v2 §2.
Depends on #17 and #19 merged.

Build config/selections/elbadry2024.yaml per §7: parent = DR3 Orbital + AstroSpectroSB1, Shahaf
et al. (2023b) triage (177 candidates), the P_orb > 1000 d and Andrews-sourced additions, IsocLum
primary masses from gaiadr3.binary_masses, the declination-split extinction policy (delta = -30,
Lallement 2022 — NOT El-Badry 2026's -28/Lallement 2019), and the four completeness criteria.
Target N = 21.

CRITICAL §7.3: criteria (b) M2 > 1.25 Msun from joint fitting, (c) not spurious, (d) half an orbit
observed are OUTCOME-DEPENDENT. They go in the likelihood inclusion operator applied identically to
mocks, never as filters discarding real systems. The follow-up selection function may condition
only on pre-follow-up observables.

CRITICAL §4.8: the ~25% spurious rate is NOT a parameter of this sample. Put it under
validation_targets: as an output for the shared #27 model to reproduce; ensure nothing reads it.
Criterion (c) is satisfied by evaluating that model. Note the staged fixture
config/selections/external/elbadry2024_table3.yaml gives 12 spurious / 48 with G<15 = 25.0%, so the
target is exact, not approximate.

Resolve §15 Q2 (Shahaf catalog cross-match vs AMRF reimplementation) in the PR description; the
reproduction path may use the catalog, but state what forward-model/DR4 needs.

Effort contract (Standard): framework frozen by #17. Full required pytest before PR. Stop when PR
open + CI green.
```

---

### Slot F — acceleration/jerk selection (#22) — DO NOT DISPATCH

**Blocking condition: the accel/jerk selection function is unpublished and has not been supplied by
the user (§9).** There is no cut chain to build. Dispatching this slot would produce an invented
selection derived from the `accel_jerk` target list, which §9 prohibits. The draft below is
incomplete by design — parent query, cut chain, and thresholds are missing because they do not
exist. Unblock by rewriting §9 with the supplied selection, then dispatch.

```
[DRAFT — BLOCKED. Do not run as-is.]

You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (mid-tier, Standard effort).
Model: claude-4-sonnet

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/TBD-22
Branch: phase8/selection-accel-jerk (worktree). PR → main. Closes #TBD-22 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §9, docs/ARCHITECTURE.md §4 selection_function_astrometric and
selection_function_followup, four skills. Depends on #17 merged AND on §9 having been rewritten
with the user-supplied selection function.

If §9 still reads as a stub when you start, STOP immediately and report the slot is still blocked.

Build config/selections/accel_jerk.yaml as forward_model-only — no published N, and the file must
say so explicitly so the dual-path machinery does not expect one. Selection function is
solution-type occupancy (7-parameter acceleration, 9-parameter jerk) from the existing gaiamock
cascade, validated against the real catalog's aggregate fractions via the existing
solution-type-fraction diagnostic. Transcribe the cut chain from the rewritten §9.

You may NOT derive a selection from the accel_jerk target list — that list says which systems we
observe, not which candidates are selected. Keep DR3/DR4 catalog identifiers path-specific. Do not
duplicate the adoption dates in config/target_lists/derived/accel_jerk_adoption_dates.yaml. Flip
accel_jerk to enabled: true in the §12.1 registry in this PR.

Effort contract (Standard). Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot G — SB1 spectroscopic mass-function support (#26)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (mid-tier, Standard effort).
Model: claude-4-sonnet

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/109
Branch: phase8/sb1-mass-function (worktree). PR → main. Closes #109 alone on a line.
Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §8.1, §8.4, §8.4.1, docs/ARCHITECTURE.md §4 data_acquisition, four
skills. Depends on #18 merged.

Extend data_acquisition to ingest DR3 SB1 and SB1C solutions with K1, sigma_K1, eccentricity, and
period. Implement f_m = P K1^3 (1-e^2)^(3/2) / (2 pi G) in physics_utils.py plus the numerical
inversion for M2_min at edge-on inclination given a primary mass.

This is the FIRST non-astrometric mass path in the pipeline. Keep it clearly separated from the
§5.2 astrometric mass function: same module, distinct names, no shared inversion code that could
silently apply the wrong relation.

SCOPE IS SETTLED (§8.4.1) and it constrains what you build. The SB1 branch is reproduction and
validation only in v1: recover 151 sources and the 136/30/15 route split, and stop there. Do NOT
implement sin^3 i inclination marginalization. Do NOT wire SB1 systems into the population
likelihood even if it looks like a small step — that is a scoped v2 decision requiring human
sign-off (dark-hunter-pop-workflow §7). Mark the artifacts so the boundary reads as deliberate.

Effort contract (Standard). Unit tests against closed-form values. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot H — El-Badry et al. (2026) selection (#25)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/110
Branch: phase8/selection-elbadry2026 (worktree). PR → main. Closes #110 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §4, §4.8, §5, §8 in full, four skills. Source: arXiv:2608.06453v1 §2.
Depends on #17, #19, #20, #26 merged.

Build config/selections/elbadry2026.yaml per §8.10. TWO parent branches: astrometric (Orbital,
AstroSpectroSB1) via AMRF, union of four subsamples 47+3+4+22 = 76; spectroscopic (SB1, SB1C) via
f_m, 136+30-15 = 151. Total 227. Only the astrometric branch has an inference entry point (§8.4.1).

Implement the AMRF exactly once, derived from the §5.2 astrometric mass function via A^3 = m_f / M1.
Do not maintain a second independent inversion.

Primary mass is the Janssens et al. (2022) relation (§8.2), fully specified there and staged at
config/selections/external/janssens2022_mass_magnitude.yaml: piecewise M_G = a*log10(M) + b over
eight segments, inverted as M = 10**((M_G - b)/a), segment selection by M_G interval, extrapolation
FORBIDDEN outside 0.02-57.95 Msun, central a/b only. Do not re-fit, re-digitize, or "improve" it.
Defined for main-sequence sources ONLY — M1-tilde-dependent cuts are NOT APPLICABLE for evolved
sources, not false (§15 Q10). sigma_M2 for subsample 4 uses the astrometric covariance only, not the
Janssens fit uncertainty (§15 Q12 resolved); running it both ways is a recommended sanity check.

Extinction is delta = -28 with Lallement 2019, NOT El-Badry 2024's -30/Lallement 2022. Do not
consolidate.

Subsample 3 imports Andrews' PUBLISHED membership: declare depends_on: [andrews2022] and resolve
against the FROZEN andrews2022 (N=24), never andrews2022_modified (§6.6) — that would break the
expected 16. Subsample 2 uses the staged elbadry2023_table_e1.yaml; keep every column, it is also a
§4.8 labeled set.

The initial 227 is purely catalog-level: NO outcome-dependent criteria in the cut chain. §8.5
follow-up outcomes and prioritization go in the inclusion operator. The ~40%/~50% rates go under
validation_targets:, never read as inputs (§4.8).

Required acceptance test beyond the headline N: reproduce the Simon et al. (2026) exclusion
breakdown 5/2/1/1 (§8.9) — it validates four thresholds independently.

Escalate, do not decide: §15 Q7 (nsstools vs our thiele_innes_to_campbell) and Q9 (Andrews G<15
must yield exactly 16; a mismatch fails BOTH reproductions and is not to be tuned).

Effort contract (Deep): you own the two-branch structure and its mapping onto #17. Justify in the
PR. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot I — spuriousness model (#27)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/111
Branch: phase8/spuriousness-model (worktree). PR → main. Closes #111 alone on a line.
Labels: phase-8, enhancement.

Read docs/CONTINUATION_PLAN.md §4.8 in full, plus §6.4, §6.6, §7.3, §8.5, §8.6, §13, four skills.
Depends on #18 and #19 merged. Blocks #23.

Build ONE sample-independent model P(spurious | physical and observational parameters) in
spuriousness_model.py, configured by config/spuriousness_model.yaml. This REPLACES the per-sample
contamination constants earlier drafts specified; any code path reading a per-sample rate as an
input is a bug you are here to remove. Model must be evaluable on MOCK realizations — that is the
point, since it feeds the #23 inclusion operator and a constant cannot be.

Labeled data, already staged, 293 sources / 63 spurious across three papers:
config/selections/external/{elbadry2023_table_e1,elbadry2024_table3,elbadry2026_table7,
elbadry2026_table8}.yaml.

TWO LABEL AXES. `verdict` (genuine/spurious/undetermined) is solution reliability and is what you
predict. `nature` is companion type and belongs to companion_nature_likelihood. An ultramassive WD
is a genuine solution with a non-NS companion. Do not collapse the axes.

CENSORING. The 33 `undetermined` rows are censored, NOT missing at random: 12 by G > 15 and 2 by
poor F2 — both covariates — and El-Badry 2026 states outright that spurious sources were
preferentially dropped from follow-up. DO NOT DROP THEM; that biases the model against exactly the
faint, high-F2, low-significance regime where spuriousness is highest. Fit label and censoring
jointly, Heckman-style, sharing covariates (§4.8 gives the argument and the two rejected
alternatives). If the joint model is unidentified in practice, escalate — do not fall back to
dropping rows.

COVARIATES: §4.8 lists candidates — harmonic distance, F2, parallax S/N, a0 S/N, G, implied
m_f/M2, visibility_periods_used, RV consistency. DO NOT HAND-PICK. Run the sensitivity-analysis
module and retain only what it justifies; emit spuriousness_covariate_sensitivity showing kept,
dropped, and why. A covariate retained because it appeared in §4.8's table rather than because the
module kept it is a specification failure and will be treated as one in review. The one interaction
with literature motivation is F2 x G with a break at the G = 13 window-class change; do not add
others without module support. Regularize — 63 positives over ~8 covariates binds.

RV consistency exists only where follow-up happened: model as missing-at-selection, do not impute,
do not silently restrict to rows that have it.

ACCEPTANCE (§13): one parameter set reproduces El-Badry 2024 ~25% (fixture: 12/48 = 25.0%) and
El-Badry 2026 astrometric ~60% reliable (fixture: 46/76 = 60.5%). SB1 is ADVISORY pending §15 Q15 —
Table 8's Notes label nature, not spuriousness, and give only 15.9%; resolve what denominator the
paper's ~50% refers to and report it, do not force agreement. Reproducing one target but not the
other signals an absorbed per-sample normalization, which is the failure §4.8 exists to prevent —
STOP and escalate with both predicted values and your covariate set rather than adding a per-sample
term.

Gaia BH1 (4373465352415301632) is labeled genuine and Andrews' exclusion of it was wrong (§6.6, §15
Q14 resolved). Do not re-litigate. Score it explicitly in
spuriousness_labeled_set_performance.

Effort contract (Deep): you own functional form, link function, and missing-data treatment. Justify
each in the PR including what the sensitivity analysis rejected. Full required pytest before PR.
Stop when PR open + CI green.
```

---

### Slot J — likelihood integration (#23)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/112
Branch: phase8/sample-selection-inference (worktree). PR → main. Closes #112 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §4.7, §4.8, §6.6, §8.4.1, §8.7, §13, §15, docs/ARCHITECTURE.md §4
inference, four skills. Depends on #20, #21, #25, #27 merged. NOT on #22, which is blocked (§9) —
proceed with accel_jerk disabled.

Extend the Poisson rate with sample_selection_function_s per §4.7. Resolve §15 Q1: the samples
overlap THREE ways, not pairwise — El-Badry 2024 drew from Andrews, and El-Badry 2026's astrometric
subsample 3 IS the Andrews selection restricted to G < 15 (§8.7). Naive summation double-counts.
Choose between separate Poisson processes per sample and a unified inclusion-indicator formulation,
justify in writing, implement it, and emit sample_overlap_matrix.

Inference uses andrews2022_modified (N=25), not the frozen andrews2022 (§6.6), and only El-Badry
2026's astrometric branch, not its SB1 branch (§8.4.1).

Fold in the outcome-dependent inclusion terms from §7.3 and §8.5, applied identically to mocks. The
"not spurious" condition is evaluated by calling #27's shared P(spurious | ·) per mock realization.
Do NOT read any per-sample spurious rate — those are validation_targets: only.

Do not silently upgrade v1's staged-but-connected treatment to a fully joint one
(dark-hunter-pop-workflow §7); that is a separately scoped v2 decision.

Effort contract (Deep): this changes the core likelihood. Written justification plus SBC-style
recovery evidence in the PR. Full required pytest before PR. Stop when PR open + CI green.
```

---

### Slot K — reproduction diagnostics (#24)

```
You are a Phase 8 subagent for UCSC-Transients/dark-hunter_pop (mid-tier fast, Standard effort).
Model: cursor-grok-4.6-high-fast

Issue: https://github.com/UCSC-Transients/dark-hunter_pop/issues/113
Branch: phase8/sample-diagnostics (worktree). PR → main. Closes #113 alone on a line.
Labels: phase-8.

Read docs/CONTINUATION_PLAN.md §13, §8.2, §8.3, §8.9, docs/PLOTS.md, docs/ARCHITECTURE.md §4
diagnostics, four skills. Depends on #20, #21, #25 merged. NOT on #22, which is blocked (§9).

Implement the §13 diagnostics you own: sample_attrition_waterfall, sample_reproduction_report,
simon2026_exclusion_breakdown, covariance_health, sample_selection_function, mode_divergence,
janssens_segment_occupancy. The three spuriousness diagnostics belong to #27, not you.

The attrition waterfall must handle El-Badry 2026's two-branch, four-subsample UNION structure
(§8.3), not a single linear cut chain, and must render "cut not applicable" separately from "cut
failed" (§15 Q10) — collapsing them makes the subsample counts irreproducible.
janssens_segment_occupancy reports per-segment counts, boundary-resolved counts, out-of-range
counts, and flags the near-uninformative 1.55-1.80 Msun segment (§8.2).

mode_divergence must cover andrews2022 vs andrews2022_modified (§6.6), where the expected
divergence is exactly one source, Gaia BH1.

Use the shared primitives in plotting.py; add no new rendering code paths. Reports are full-detail
(caveman exemption). Wire into the diagnostics hook registry with config on/off switches matching
the existing hook convention.

Effort contract (Standard). Full required pytest before PR. Stop when PR open + CI green.
```

---

### Review/Integration (continuous, roster #15)

```
You are the continuous Review/Integration subagent for UCSC-Transients/dark-hunter_pop
(top-tier, Deep effort).
Model: claude-opus-5-thinking-medium

No issue of your own; you review and open small integration PRs against Phase 8 branches.

Read docs/CONTINUATION_PLAN.md §4.4, §4.8, §12, §13, docs/ORCHESTRATION_PLAN.md §4, four skills.

On every Phase 8 PR: merge config/fragments/sample_selection.yaml into config/config.yaml; verify
every numeric threshold lives in config/selections/*.yaml with none inline
(dark-hunter-pop-workflow §1); check DR3/DR4 key independence per §12.4 and run the audit function;
confirm no frozen selection file was edited without a schema_version bump; keep slow MC suites out
of the default CI gate.

Two Phase-8-specific checks, run on every selection PR:
- No code path READS a per-sample spurious or contamination rate. They live under
  validation_targets: and are outputs only (§4.8). Grep for it.
- config/spuriousness_model.yaml stays OUTSIDE config/selections/. Nesting it under a sample
  reintroduces the design §4.8 removed.

Reject edits to config/selections/andrews2022.yaml that "fix" the Gaia BH1 exclusion — the
correction belongs in andrews2022_modified.yaml and the frozen file must stay as published (§6.6).

Prefer review plus small integration PRs. Docs-first before any freeze break.
```

## 15. Open items

| Q | Question | Blocks | Owner |
|---|---|---|---|
| Q1 | Samples overlap **three ways, not pairwise**: El-Badry 2024 drew candidates from Andrews, and El-Badry 2026's astrometric subsample 3 *is* the Andrews selection restricted to `G < 15` (§8.7). Separate Poisson processes per sample, or one unified inclusion-indicator formulation? Naive summation double-counts. | #23 | #23 subagent, with human sign-off |
| Q2 | Shahaf et al. (2023b) triage: cross-match the published 177-candidate catalog, or reimplement the AMRF algorithm? Catalog is fine for reproduction; the forward-model and DR4 paths need the algorithm, since a catalog cannot be applied to mocks. **Partially eased by §8.3**, which computes the AMRF directly and so supplies the algorithm side; the remaining question is whether §7's triage can be re-expressed in those terms or genuinely needs the catalog. | #21, DR4 | #21 subagent, with human sign-off |
| Q4 | Whether `nss.corr_vec` / `nss.bit_index` as published are sufficient to reconstruct the full 12×12 covariance for every solution type, or whether matrices must be staged from a local file (§10). | #18 | #18 subagent |
| Q5 | **Partially answered.** The user's working hypothesis is that Andrews' parent is **all 12-parameter (orbital) astrometric solutions that are not SB1/SB2**, excluding `AstroSpectroSB1` — recorded in §6.1 and in `andrews2022.yaml`. Expressed with explicit uncertainty ("I think"), so archive verification still gates it: the query must return **exactly 134,598** as a *blocking* acceptance test before any downstream cut is trusted, and the verified set plus the confirming query must be frozen into the selection file. | #20 | #20 subagent |
| Q7 | `nsstools` (Halbwachs et al. 2023) is what El-Badry 2026 uses for `ã0` and `σ_ã0`. Do we vendor it as a pinned dependency alongside `gaiamock`, or use our existing `thiele_innes_to_campbell` (§5.1)? Either is defensible, but the reproduction path must match the paper's numbers, so any difference between the two must be measured before choosing. | #25 | #25 subagent, with human sign-off |
| Q9 | El-Badry 2026 §2.1 subsample 3 says the Andrews import "yields 16 sources", but Andrews' published sample is 24 (§6). The difference is presumably the `G < 15` cut, but the paper does not state it explicitly. Confirm that applying `G < 15` to our reproduced Andrews sample yields exactly 16, and treat a mismatch as a failure of *both* reproductions rather than tuning either. | #25 | #25 subagent |
| Q10 | El-Badry 2026's `M̃1` is defined only for main-sequence sources, so `M̃1`-dependent cuts are undefined for evolved candidates rather than false. The Janssens `extrapolation: forbid` policy (§8.2) creates a second route to the same state. Confirm the framework's cut evaluator distinguishes "cut not applicable" from "cut failed" in the attrition waterfall — otherwise the subsample counts will not reproduce. | #17, #25 | #17 subagent |
| Q11 | The Janssens et al. (2022) Table 1 fit does not publish the **`a`–`b` covariance** within each mass segment, so the cross-term in the §8.2 uncertainty propagation cannot be evaluated. Spec'd to default to zero correlation and report that the default was used. Determine whether the correlation is recoverable from the paper or its data; if not, bound the effect on `σ_M̃1` and record the bound. | #25 | #25 subagent |
| Q15 | **El-Badry 2026's ~50% SB1 spurious rate is not recoverable row-by-row from Table 8.** The paper states *"∼50% have spurious spectroscopic solutions, and a majority of the rest are post-mass-transfer binaries or hierarchical triples"*, but Table 8's `Notes` column labels **companion nature** (two-temperature SED, Algol-type, EB, SB2) for most rows and says `RVs inconsistent with orbit` for only 24 of 151 (15.9%). Either many of the nature-labeled rows *also* have spurious solutions and the Notes report the more specific finding, or the ~50% is drawn from the follow-up subsample rather than all 151. Resolve before using the SB1 branch as a validation target: determine the denominator the ~50% refers to, and whether the nature labels and spuriousness labels are exclusive or overlapping. Until resolved, treat the SB1 target as **advisory**, and gate the §13 acceptance test on the two astrometric targets, which are exactly recoverable. | #27 | #27 subagent, with human sign-off |

**Resolved.**

| Q | Decision |
|---|---|
| Q3 | El-Badry 2026 paper supplied as arXiv:2608.06453v1; §8 is fully specified. |
| Q6 | **SB1 branch is reproduction-only for v1.** All 151 spectroscopic sources must be reproduced to validate the selection, and the branch supplies one of §4.8's three spuriousness validation targets, but only the astrometric branch (76) feeds population inference. The `sin³i` inclination marginalization — plus a selection-aware inclination prior and joint treatment of the two dual-branch sources — is explicitly **not** v1 work, and is what a v2 would have to add. Recorded in §8.4.1 as a scope boundary, not a limitation to route around. |
| Q8 | **Both external tables supplied by the user.** The Janssens et al. (2022) mass-magnitude relation arrived as Table 1 fit parameters plus the corresponding figure; the functional form `M_G = a·log₁₀(M/M☉) + b` was verified against both (solar anchor, continuity at all seven internal boundaries to ≤ 0.0094 mag, figure endpoints) and is fully specified in §8.2 with its inversion, segment selection, extrapolation policy, and uncertainty propagation. El-Badry 2023a Table E1 is staged at `config/selections/external/elbadry2023_table_e1.yaml`. Neither is to be fetched, digitized, or re-fit. |
| Q12 | **`σ_M̃2` for El-Badry 2026 subsample 4 excludes the Janssens `a`/`b` fit uncertainty**, using the astrometric covariance only. `propagate_fit_uncertainty: false` in §8.2 is confirmed correct, not an assumption. The empirical both-ways check stays a **recommended validation** when #25 runs — if the astrometric-only variance fails to recover 22 sources, that is evidence of a problem elsewhere in the chain — but it is no longer an open decision. |
| Q13 | **Table E1 is complete at six rows**; it covers only the BH candidates, which is why it is small. Three further labeled tables were supplied and staged: El-Badry 2024 Table 3 (60 NS candidates), and El-Badry 2026 Tables 7 (76 astrometric) and 8 (151 SB1). The label set is now **293 sources with 63 spurious across three papers**, which §4.8 assesses as adequate for a regularized multi-covariate fit. The earlier "too small for a multi-covariate fit" language is withdrawn. |
| Q14 | **Andrews was wrong; Table E1 was right.** Source `4373465352415301632` is **Gaia BH1** — El-Badry 2026 Table 7 names it explicitly. Andrews et al. (2022) excluded it as a probable scanning-law harmonic (`P = 185.8 d ≈ 3 ×` the scanning period, `m_f ≈ 11.6 M☉`); the period looked too perfect to be a coincidence, and it was not a coincidence. Handled by carrying a **modified Andrews variant** that restores the source (§6.6, N = 25) alongside the frozen reproduction path (N = 24), rather than by generalizing the exclusion into a period-near-harmonic cut — that alternative was rejected because it would discard genuine systems and would hard-code the very question the §4.8 model exists to answer. The source stays in the training set labeled genuine; harmonic distance remains a candidate covariate, now known to be insufficient alone. |
| Q2 → superseded in part | Q2 remains open for §7's triage, but the AMRF is now computed directly in §8.3, so the algorithm side is covered. |

**Superseded.** The earlier design storing each sample's quoted spurious/contamination rate as a
*modeled per-sample parameter* is replaced by §4.8's single sample-independent
`P(spurious | ·)` model. Those rates are now recorded as `validation_targets:` — outputs to
reproduce, never inputs. Roster #27 owns the model; it blocks #23.
