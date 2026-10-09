# #391: what are the real high-f_m DR3 orbits? (analysis only)

The script is `scripts/characterize_high_fm_391.py`; the per-bin numbers are in `report.txt`.

**Sample:** real DR3 Orbital + AstroSpectroSB1 orbits with the #391 mirror filters, 167,911 rows.

**Parallax correction:** the zero point is added per G bin, +22 to +37 µas. It is the median of (Bailer-Jones truth − observed parallax) over the parent's stars with ϖ/σ > 20, which approximates Lindegren et al. (2021) without their full recipe.

**Corrected trend:** after the correction, the high-f_m share (f_m > 0.1) rises from **3.5% within 0.2 kpc to 38% at 1.5–3 kpc**.

## Methods

- **Primary mass M1:** MIST isochrone posterior mean on the dereddened CMD, the same machinery as the mock parent (`parent_cmd_isochrone`). It is defined for 93% of rows.
- **AMRF (Shahaf et al. 2019):** A = (a0/ϖ) M1^(−1/3) P_yr^(−2/3). Class limits are computed per M1 from the Janssens MS mass–luminosity relation:
  - **Class I:** A is at or below the largest AMRF any single luminous MS companion can give (q ≤ 1).
  - **Class II:** above that, but at or below the largest AMRF of an MS pair (two equal MS stars).
  - **Class III:** above both, so no luminous companion configuration can produce it. It needs a compact (dark) companion, or the orbit is spurious.
- **Dark-companion mass:** M2_dark is the mass-function mass with f = 0. "Beyond WD" means M2_dark > 1.5 M⊙.
- **Spuriousness model:** P(spurious | x) comes from `spuriousness_model`, refit from its four labelled tables. It retains `goodness_of_fit`, `a0_snr` (= `significance`), `log_implied_companion_mass` (we use log M2_dark) and `f2_x_g_break`.
  - It is **trained on compact-object candidates**, so applying it to the general orbit population is an extrapolation.
  - It has not yet passed its rate-reproduction acceptance (Wave B).
- **Labelled-table cross-match:** only 105 of the orbits appear in the labelled tables, so these counts are anecdotal.

## Findings by distance (f_m > 0.1 vs ≤ 0.1)

- **AMRF class III share falls with distance; class II rises.** Among high-f_m orbits:

  | Distance (kpc) | 0–0.2 | 0.2–0.4 | 0.4–0.7 | 0.7–1 | 1–1.5 | 1.5–3 |
  |---|---|---|---|---|---|---|
  | Class III | 0.46 | 0.35 | 0.22 | 0.15 | 0.15 | 0.17 |
  | Class II | 0.47 | 0.58 | 0.72 | 0.79 | 0.77 | 0.69 |

  Class I is about 0 in every bin, by construction of f_m > 0.1.
- **Spurious indicators grow with distance for high-f_m orbits**, faster than for low-f_m orbits:

  | Distance (kpc) | 0–0.2 | 0.7–1 | 1.5–3 |
  |---|---|---|---|
  | P > 0.8 × the 1038 d baseline, high-f_m | 0.34 | 0.41 | 0.66 |
  | P > 0.8 × baseline, low-f_m | 0.18 | 0.20 | 0.51 |
  | Mean P(spurious), high-f_m | 0.21 | 0.30 | 0.41 |
  | Mean P(spurious), low-f_m | 0.28 | 0.23 | 0.30 |

  - High-f_m orbits also have higher F2 at the same distance.
  - Their eccentricity rises with distance (0.16 → 0.38).
- **Beyond-WD dark masses are rare:** 1–5%. The IPD multi-peak fraction (about 0.09) and the IPD harmonic amplitude are the same for both groups. Resolved-pair contamination does not stand out.
- **Labelled sources** (good / spurious / unknown) in the high-f_m group, by distance:

  | Distance (kpc) | 0.2–0.4 | 0.4–0.7 | 0.7–1 | 1–1.5 | 1.5–3 |
  |---|---|---|---|---|---|
  | good / spurious / unknown | 8/7/3 | 19/6/12 | 13/8/3 | 5/4/2 | 0/7/1 |

  All 7 labelled sources beyond 1.5 kpc are spurious.

## Rough decomposition of high-f_m orbits

Fractions are of **all** orbits in each distance bin.

| Distance (kpc) | High-f_m total | Plausibly spurious | Plausibly compact (class III, M2_dark ≤ 1.5) | Class III beyond WD | Class II (triple or compact) |
|---|---|---|---|---|---|
| 0–0.2 | 0.035 | 0.015 | 0.009 | 0.000 | 0.009 |
| 0.2–0.4 | 0.069 | 0.028 | 0.015 | 0.000 | 0.023 |
| 0.4–0.7 | 0.096 | 0.044 | 0.011 | 0.000 | 0.039 |
| 0.7–1 | 0.130 | 0.066 | 0.007 | 0.001 | 0.052 |
| 1–1.5 | 0.172 | 0.102 | 0.008 | 0.001 | 0.057 |
| 1.5–3 | 0.380 | 0.274 | 0.010 | 0.003 | 0.083 |

**Category criteria:**
- **Spurious:** P(spurious) > 0.5, or P > 0.8 × 1038 d, or labelled spurious.
- **Compact:** not spurious, class III, and M2_dark ≤ 1.5 M⊙.
- **Class II:** not spurious, and the AMRF admits an MS pair. That is either a triple with a luminous inner pair, or a compact companion with a partly luminous system.

**Uncertainties:**
- The Poisson errors are below 0.006 in every cell. The systematic uncertainties dominate:
  - the spurious criteria are judgement calls (the P > 830 d cut alone flags 34–66% of high-f_m orbits);
  - the spuriousness model is extrapolated;
  - the class limits depend on M1 and on the Janssens relation;
  - for high-f_m orbits, f_m and AMRF are inflated by a0 noise near threshold.
- Treat the split as **order of magnitude**: ±50% per category.

## What this says

- The **plausibly compact share is about 1% of orbits, roughly flat with distance**. A WD/compact mixture would add about 1% everywhere. It **cannot** produce the high-f_m excess that grows with distance.
- The **growth is mostly plausibly spurious** (1.5% → 27%, long periods near the baseline, high P(spurious), labels spurious far out). The **class II** share also grows (0.9% → 8%); that is triples, or compact companions in luminous systems.
- The count deficit is **not only a high-f_m problem**. At 1–1.5 kpc high-f_m orbits are 17% of the real total, but the mock has 0.35–0.45× the real count (docs/gate391/distance_count). Most of that deficit is in **low-f_m** orbits.

## Recommendation (before re-checking counts by distance)

1. **Spuriousness term in the forward comparison, first.** Either weight the real orbits by 1 − P(spurious), or forward-model spurious solutions. As a minimum, compare with orbits P > 0.8 × baseline and high P(spurious) excluded on both sides. This touches the largest component, and it changes the real side, not the mock.
2. **Then the compact mixture (MP-Q17), WD at least.** It is small (about 1% of orbits) but real, and needed for the f_m tail at small distance. It is reweightable.
3. **Triples:** the class II excess is the second-largest physical component. Modelling it needs the triples module (off in v1, a scope decision), or an explicit "luminous-triple" class.
4. **Re-check the low-f_m distance deficit after (1)**, separately. It points to the MdS17 frequency and q at the M1 and P of distant stars (rung-3 fit), not to compact companions.
