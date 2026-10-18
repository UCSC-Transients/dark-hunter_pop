# #391 follow-ups to the bright-excess round (Ryan, 2026-10-18; analysis plus a small re-injection)

Contact sheet: `followups.png`. Numbers: `shortp_injection.json`, `real_m1.json`, `mbreak14_noW/`.

## 1. Re-injecting the distant short-P real orbits

- **Run.** The 103 real C1 orbits at d > 1 kpc with log P < 2.6, plus two controls matched in G:
  - same period, nearer: d 0.8–1.0 kpc, log P 2.3–2.6;
  - same distance, longer period: d 1.0–1.2 kpc, log P 2.6–2.75.
- Each was injected 3 times with the current epoch model (v2 + N2d + visibility-period loss, `validate_epoch_model_v2_400.py inject --variants v3`).
- The 2 CPU-h watcher stopped the run at 906 of 927 realizations. Only the longer-P control is incomplete (96 of 103 systems).

| group | re-detected | rejections: ϖ/σ cut / RUWE gate / acceleration-first | published ϖ/σ ÷ threshold, median | σ_ϖ recovered / published, median |
|---|---:|---|---:|---:|
| target | **0.20** (62/309) | 79% / 17% / 0 | 1.07 | 1.23 |
| control, nearer | 0.52 | 77% / 12% / < 1% | 1.13 | 1.12 |
| control, longer P | 0.68 | 58% / 12% / 18% | 1.26 | 1.06 |

- **The cut that rejects them is the Halbwachs parallax cut, ϖ/σ_ϖ > 20000 d / P.** It is applied after an orbital solution is found. The acceleration-first rule plays no role for the target.
- The published orbits sit only 7% above that cut. 65% are within 10% of it, against 36% and 24% for the controls.
- The re-fit σ_ϖ is 1.23× the published one. That is partly expected: orbits published close to a threshold were selected for favourable noise, so they regress on re-injection. The controls follow the same ordering (1.12 nearer the threshold, 1.06 further away).
- **Reading.** These cells are set by the ϖ/σ_ϖ > 20000/P cut, at P of about 250–300 d, where the orbit and the parallax are partly degenerate.
- The mock's deficit there (about 10 expected against 103 real) is consistent with a σ_ϖ in this regime that is about 10% too large, on top of the threshold selection.
- These injections cannot separate the two. That needs σ_ϖ compared against DR3 for non-threshold orbits at log P 2.3–2.5. Proposed as the next test.

## 2. Profile refit with the slope break at 1.4 M⊙ (noW, gens 33–38, coarse grid)

- α_br = −0.26 ± 1.14. −ln L is 387.8 against 388.2 without the break, Δ = 0.4 for one more parameter. α_hi moves from +0.57 ± 0.20 to +0.64 ± 0.64.
- G < 10 orbit ratios (5–8 / 8–9 / 9–10 mag): 2.59 / 1.70 / 1.38 without the break, 2.60 / 1.68 / 1.35 with it. Accelerations are unchanged.
- **The break is unconstrained and does nothing.** The coarse grid has a single G 5–13 bin, so the likelihood never sees the bright excess. cmdW was not run, since it would show the same.
- **Even α_hi → 0 does not remove the excess.** Reweighting the best fit to α_hi = 0 above 0.8 M⊙, without refitting, still leaves 2.3× at G < 8 and 1.45× at G 8–9 relative to G 11–16.

## 3. Real-side M1 at G < 10 (isochrone M̂1 on both sides; real with the fit's 1/(ϖ + zp) convention)

| row M̂1 (M⊙) | < 0.8 | 0.8–1 | 1–1.2 | 1.2–1.5 | 1.5–2 | **2–3** |
|---|---:|---:|---:|---:|---:|---:|
| orbits, mock ÷ real | 5.3 | 1.0 | 1.2 | 1.2 | 1.8 | **6.3** (mock ESS 4.7) |
| accelerations, mock ÷ real | 3.1 | 1.5 | 0.9 | 1.0 | 2.0 | **3.9** |

- At G 10–11 the 2–3 M⊙ orbit ratio is 2.4.
- **The excess is concentrated at M̂1 ≥ 1.5 M⊙, and most of all in 2–3 M⊙ dwarfs.** These are 22% of mock G < 10 dwarf-orbit weight but 3.6% of the real ones.
- Below 1.5 M⊙, mock and real agree to within about 20%.
- The excess is too large to be a frequency slope (see 2).

## 4. Bright isochrone-M1 failures

| G | < 8 | 8–9 | 9–10 | 10–11 | 13–15 |
|---|---:|---:|---:|---:|---:|
| parent, M1 success | 0.58 | 0.73 | 0.76 | 0.81 | 0.94 |
| real orbits, M1 success | 0.84 | 0.90 | 0.94 | 0.96 | 0.95 |
| real accelerations, M1 success | 0.83 | 0.90 | 0.94 | 0.96 | |

- The parent rate is the same with the BJ distances and with the real-side 1/ϖ convention, so the distance convention is not the cause.
- **The failures are not symmetric.** The parent loses 24–42% of its rows at G < 10. Real NSS rows lose 6–16%, and the fit keeps them anyway (it only requires an atmosphere).
- The bias runs against the excess: the mock has no stand-in for those parent rows. If the failed rows host orbits at the real rate, at most about 7% of the real G < 10 orbits (477 have no M̂1) have no mock counterpart. That is small next to a 1.4–2.6× excess.
- Why the parent fails more often at the bright end is not yet diagnosed. It is presumably rows the MIST posterior cannot place, but this is unchecked.

## Ranked causes and proposed fixes (none implemented; each needs Ryan)

1. **Too many mock orbits around 1.5–3 M⊙ dwarfs (A/early-F)**, beyond what any frequency slope gives. Candidates:
   - (a) `poisson_intensity` multiplicity. MdS17's total companion frequency is above 1 at A/B masses. The mock treats every companion as an independent, detectable astrometric binary, but in reality the extra companions (an inner binary, or a luminous tertiary) dilute or perturb the photocentre.
   - (b) A DR3 detection deficit specific to hot or fast-rotating primaries.
   - Test next, analysis only:
     - the mock's 2–3 M⊙ G < 10 weight split by companion count per parent row and by f;
     - the real 2–3 M⊙ NSS hosts' Teff / vsini against the parent's.
   - Fix options:
     - cap the multiplicity, or give it a per-row outer-companion veto, for M1 > 1.5;
     - a high-mass frequency term constrained by a bright or M1 axis in the fit grid. The grid edge would come from the model (1.5 M⊙), not the counts.
2. **σ_ϖ near P ≈ 250–300 d** (distant short-P cells): test the mock σ_ϖ against DR3 on orbits away from the threshold before any change.
3. **Isochrone-M1 failures:** a small asymmetric bias (≤ 7% of real G < 10 orbits). Fix: also require isochrone M̂1 on the real side, or report it as a systematic.
