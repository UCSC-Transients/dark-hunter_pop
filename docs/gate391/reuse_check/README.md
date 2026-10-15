# #391 old-draw reuse: check result and proposal for Ryan (no code until approved)

Inputs: `report.txt` (per 108-grid cell), `ess_coarse.txt` (per coarse cell), and the per-draw deltas in `output/proposal_set/reuse_check.npz`. The check script is `scripts/reuse_check_391.py`.

**Check.** 14.3% of the 2.43 M old draws (gens 23–32) qualify, and 13.9% of the old accepted orbits (4,550 of 32,777).
- By old M1: 3% below 0.8 M⊙, 11% at 0.8–0.9, 24–31% above 0.9.
- Weak G 5–13 cells: 12–38%. Faint, near cells: 2–5%.
- Draws fail mainly through the companion. The Teff correction moves the coeval flux ratio for every companion below about 0.9 M⊙, with a median |Δ log f| of 0.03–0.06 dex.

**Proposal.**
1. **Region R is a deterministic function of the stored draw.** A draw is in R when, recomputed under the current model with the same seeds, it satisfies |Δ ln M1| < 0.01 (deblended M1), |Δ log10 f| < 0.02 (row-level MIST coeval relation), and the resolved flag is false. R is computed once per old generation and stored as a boolean truth column, versioned with the model key.
2. **Mixture weights.** Old generation j enters only with its proposal restricted to R: q_j^R(x) = q_j(x)·1_R(x). For x ∈ R the deterministic-mixture denominator is Σ_{j ∈ old} n_j q_j(x) + Σ_{k ∈ new} n_k q_k(x). Outside R it is new generations only, and old draws outside R are dropped.
   - Because R is a fixed function of x, 1_R(x) is evaluated identically in every term, so this is the standard restricted-support mixture and stays unbiased for any target over R.
   - The catch is that 1_R must be computable for the new draws too. It is, because R depends only on (row, q, P, e, f-noise) through the same recompute.
3. **Expected gain** (coarse grid, Kish ESS of noW weights, accepted orbits). Reusing old draws in R roughly doubles the ESS of the bright cells:
   - G 5–13 / d < 1 kpc: new-only 29–52, old-in-R 9–62;
   - G 5–13 / d > 1 kpc: new 2–6, old-R 7–17;
   - faint near cells: new 28–58, old-R 11–52.
   That is about 1.5–3× the effective sample in the weak bright cells, worth roughly 40–80 CPU-h of new draws.
4. **Residual bias and validation.** The tolerances leave each reused draw's generation-time inputs off by up to 1% in M1 and 0.02 dex in f. At significance 20–30 that is a ≲ 0.3% shift in a0 and a ≲ 1% change in the photocentre factor, well below gaiamock's per-draw noise.
   - Validation: in coarse cells covered by both, compare the new-only and the combined weighted counts and parameter fits, with differences required within 1σ (MC).
   - Also, tighten the tolerances by 2× and show the fitted parameters do not move beyond 0.5σ.
