# #391 rung 3: model problems before more draws (Ryan, 2026-10-10)

Ryan set this order: (a) the G shape and evolved primaries; (b) the spurious template; (c) the near and far distance overshoot; (d) a refit under the new bin rule (spec §12.14, PR #468). This is analysis only: no draws, no config changes, nothing deleted.

**Stopped at the stop conditions.** The rule-compliant refit (d) is blocked by an ESS conflict, and the fixes for (a) and (b) involve science choices for Ryan.

- **Scripts:** `scripts/diagnose_g_shape_391.py`, `scripts/binrule_feasibility_391.py`.
- **Outputs:**
  - `g_shape/` and `g_shape_cmdW/`: report and figure (G by CMD class, CMD ratio, period by class);
  - `binrule_feasibility.txt`;
  - `distance_and_template_check.txt`.
- **Reference fit:** the gens 23–30 control (18-bin grid) best fit, `docs/gate391/rung3_refit4/gens23_30_gen29_fine_grid/`.

## (a) G 8–9 excess and G 15–17 deficit; evolved primaries

**The G 8–9 excess (×2.45) is not an evolved-primary effect, and is mostly draw noise.**
- Dwarfs (×2.06) and evolved stars (×2.12) both show it. The evolved share at G 8–9 is 0.42 in the mock and 0.49 in the real data.
- The bin has ESS 19 (evolved ESS 5.5; its heaviest draw holds 12% of the bin's weight). The neighbouring bins swing: G 7–8 ×0.77 at ESS 10, G 9–10 ×1.17 at ESS 49.
- Combined, G 6–10 is ×1.3–1.4 (cmdW ×1.2).
- This is the bright, sparse end of the K = 10⁶ parent, so it is sampling, not a model to fix now.

**The G 15–17 deficit (×0.64–0.66) is real and dwarf-only.**
- ESS is 119 at G 15–16. The deficit is the same under cmdW (×0.66), so it is not a Malmquist effect. The median distances match (0.53 vs 0.51 kpc).
- It is fixed by (b), not by an evolved-primary correction (see the table there).

**Evolved primaries are over-produced overall:** 15.0% of mock C1 orbits against 11.2% real.
- The excess is ×1.2–1.6 at G 9–14 and ×1.62 in the M_G0 0–1.5 band (clump / subgiant).
- **It is flat in period:** ×1.11, 1.43, 1.10, 1.44, 1.16 across log P < 2.3 to 2.8–2.92, and the same under cmdW.
- So a Roche / tidal short-P truncation (MP-Q28b) would **not** remove it. It is a normalization of the evolved channel.

**Science decision for Ryan (MP-Q28; none chosen):**

| option | what | needs |
|---|---|---|
| **A (recommended)** | A free evolved-primary frequency multiplier f_evo at rung 3 (a reweightable target change; no stored draw is invalidated) | A CMD-class (evolved / dwarf) split, or a G split, in the likelihood under §12.14 |
| B | MP-Q28a: change the classifier margin n_σ (3 → 4 or 5). The real evolved fraction moves 16.5 / 14.4 / 13.3 / 12.4% | No draws; it changes both sides symmetrically, so it may not close a mock-only excess |
| C | MP-Q28b truncation | Not supported: the excess is P-independent |
| D | MP-Q28c / e: the evolved M1 or MdS17-mass choice | Generation-time for M1 (§3.5): **invalidates stored draws** and means a full re-simulation, about 280 CPU-h for gens 23–30 at the measured cost |

## (b) Spurious template form (MP-Q43)

**Diagnosis.** The PR #455 re-injection measures how the spurious probability depends on G:

| G | real systems | A_real | A_mock | π | shrunk ratio |
|---|---|---|---|---|---|
| 13–14 | 165 | 0.65 | 0.78 | 0.16 | 0.91 |
| 14–15 | 104 | 0.58 | 0.73 | 0.21 | 1.10 |
| 15–16 | 31 | 0.39 | 0.64 | **0.40** | **1.70** |

The current template (T_GD ∝ the real (G, d) counts, so a uniform spurious share) cannot put spurious orbits at faint G. The fit therefore lowers N_s (−2.6σ) rather than overfill the bright and middle bins.

**Proposed fix.** Multiply T_GD by r_G(G), the shrunk ratios above, taken flat outside G 13–16 (an extrapolation: the re-injection covers only G 13–16 at 0.7–1.5 kpc). Keep the 0.21 ± 0.04 prior on the measured bin. This changes the fit model only; no draws are invalidated.

**Before / after at the fixed control θ, noW** (no refit, see (d)), (mock + spurious) / real:

| G | uniform template (fit N_s 13.7k) | G-dependent template, same N_s | G-dependent, N_s at the prior share 0.21 |
|---|---|---|---|
| 12–14 | 1.03 | 1.02 | 1.12 |
| 14–15 | 0.93 | 0.94 | 1.05 |
| 15–16 | 0.64 | 0.71 | **0.89** |
| 16–18 | 0.63 | 0.71 | **0.88** |

With the G dependence, the prior-level amplitude closes most of the faint deficit while the middle rises about 10%. A refit would rebalance ln A, so it is expected to resolve both the faint deficit and the N_s tension together. This cannot be confirmed without a G-split likelihood (see (d)).

**Choice for Ryan:** the extrapolation outside G 13–16 (flat as proposed, or extended with a slope). The d dependence of spurious orbits is not measured either; see (c).

## (c) Near (< 0.3 kpc) and far (> 1.5 kpc) overshoot: model or sampling?

| d (kpc) | real | ESS | mock / real (noW / cmdW) | + uniform spurious |
|---|---|---|---|---|
| 0–0.3 | 28,855 | 316 | 1.09 / 1.01 | 1.19 / 1.12 |
| 0.3–0.7 | 58,982 | 355 | 0.84 / 0.87 | 0.94 / 0.98 |
| 0.7–1.5 | 40,272 | 331 | 0.80 / 0.83 | 0.90 / 0.93 |
| > 1.5 | 3,059 | **19** | 1.63 / 1.38 | 1.74 / 1.49 |

- **Near: model.** ESS is ample (316). The mock alone is at 1.09 (noW) or 1.01 (cmdW). Most of the overshoot is the uniform spurious share added at near distances, where spurious solutions are probably rare (bright, high S/N); the G-dependent template barely changes it (r_G 0.91). The remaining 0.09 depends on the Malmquist weight (noW vs cmdW), which is the **open 2-D CMD Malmquist closed-loop decision** (untouched).
  - A d-dependent spurious term would need a re-injection at < 0.5 kpc. A ~300-system × 3 re-injection, as in PR #455, costs about 2 CPU-h; it needs Ryan's go-ahead because it is compute.
- **Far: sampling.** ESS is 19–20 for 2.3% of the orbits, and that bin sits inside the fine grid's 1–6 kpc cell. It is not evidence for a model change.

## (d) Refit under the §12.14 bin rule: blocked (conflict)

- **Axis requirements** (from the model structure): a G split (M1 slopes; spurious G dependence; option A), log P edges at 2.0 and 2.6 (Beta ranges), ≥ 3 e bins, and ≥ 3 d bins.
- **Candidates:** 16 compliant grids, from 54 to 320 bins. ESS_b was evaluated at the control best fit (noW and cmdW) and at MdS17. Real counts were used only for the big-bin definition.
- **No compliant grid has every big bin at ESS_b ≥ 30** (`binrule_feasibility.txt`).
  - The least-failing grid is G [5, 13, 19.5] × d [0, 0.5, 1.0, 6] × log P [0, 2.0, 2.6, 2.919] × e [0, 0.3, 0.6, 1] (54 bins, 24 big). It fails in **11 big bins holding 26.8% of the real orbits** (min ESS_b 7.5).
  - The failing bins are mostly G < 13 at d < 1 kpc with log P 2.0–2.6, all e; G < 13 at d > 1 kpc with log P > 2.6; and two G ≥ 13 cells.
- **Draw cost to resolve: about 470–530 CPU-h** at the generation-30 rate (per-cell estimate 473 CPU-h; total-ESS scaling 534 CPU-h). Finer compliant grids cost 580–1,430 CPU-h.
- Per §12.14, neither requirement is relaxed, so no refit was run. A top-up needs Ryan.

## What closes rung 3 (recommendation)

1. Decide MP-Q28: option A, a free f_evo, needs a class or G split.
2. Adopt the G-dependent spurious template, with Ryan's call on its extrapolation. Optionally run a ~2 CPU-h near-distance re-injection for its d dependence.
3. Approve about 500 CPU-h of draws targeted at the 11 failing cells of the 54-bin compliant grid (mainly bright G < 13), so that the §12.14 refit can run.
4. The CMD-Malmquist closed loop stays open. α_lo and α_hi depend on it: the gens 23–30 noW and cmdW fits give +1.46 / −0.10 and +2.26 / −0.68, and the near-distance residual moves between 1.09 and 1.01.
