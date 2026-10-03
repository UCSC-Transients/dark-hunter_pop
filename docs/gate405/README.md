# Gate 405: the Malmquist / Öpik weight for Gaia-star primaries, and its closed-loop test

Issue #405. The derivation is in `docs/MOCK_POPULATION_SPEC.md` §9 (docs PR #407, merged). This
report shows the weight working on a synthetic universe whose truth is known. **Every
`provisional_*` setting stays open for Ryan** (MP-Q25–Q32); the closed loop only measures what
each option costs.

## The weight (spec §9.3)

The mock keeps each real row's observables: G, the Bailer-Jones distance, the sky position and the
TAG10 M1 (MP-Q4–Q6). The parent selection (G < 19, ϖ_obs > 0.2 mas, Gaia detection) depends only
on observables, so **it cancels**. The correct companion density for row s is then

    p(c | o_s) = π(c | M̂1_s) · W_s(c),   W_s(c) = L_s(f) / Z_s
    L_s(f) = N(ΔM_s + 2.5 log10(1 + f); 0, σ_s),   ΔM_s = G − μ(d̂) − A_G − M_G^J(M̂1)
    Z_s    = (1 − F_lum) L_s(0) + ∫ λ_f(log f | M̂1) L_s(f) dlog f

π is MdS17, volume-limited by construction (MdS17 §3.4, §5.2, §7.1, §8). The weight is
w_i = (N_full/N_snap) λ W / Σ_j n_j q_j. Dark companions get L(0), the same as singles, so they
get no boost. Gaia completeness enters only the volume-limited diagnostic (1/V_S, §9.5).

Code: `src/darkhunter_pop/malmquist.py` (the weight; `log_weight_for_draws` is the hook for
`proposal_set` draws) and `src/darkhunter_pop/malmquist_closed_loop.py` (the test universe).
Configs: `config/population/malmquist.yaml` and `config/population/malmquist_closed_loop.yaml`.
Figures: `scripts/run_malmquist_closed_loop.py`.

## The closed loop (spec §9.6)

A synthetic universe with this setup:

- an exponential disk (R_d = 2.6 kpc, h_z = 300 pc, out to 8 kpc);
- a Kroupa IMF on 0.3–2.5 M⊙ and Janssens M_G(M) with σ_int = 0.15 mag;
- MdS17 luminous companions drawn from the same table and provisional settings as the rung-2 target (η floor −0.4 on both sides, #410), with at most one per primary;
- no extinction.

The universe is observed with G_total < 19, ϖ_obs > 0.2 mas (σ_ϖ(G) rising to 0.37 mas at G = 19) and a synthetic sky-dependent completeness that bites toward the plane. Each parent row gets a Bailer-Jones-like geometric distance (r_lo/r_med/r_hi, with the true density as prior) and M̂1 = M1 × 10^N(0, 0.02 dex).

The synthetic parent then goes through the pipeline exactly as `gaia_source` rows would: `proposal_set.sample_proposal` (10 draws per row), `mds17_luminous_log_intensity`, then `importance_weights` with and without `malmquist.log_weight_for_draws`. No gaiamock is involved. Pulls on parent counts are Σ_s d_s / √(N var d_s) with d_s = (mock − truth) per row, so both the truth's Bernoulli scatter and the MC noise are in the denominator.

### Headline (large run: 3 × 10⁶ primaries → 336,543 parent rows, 3.37 × 10⁶ draws; 181 s wall, 6.0 GiB peak RSS)

| statistic | truth | no W | with W |
|---|---|---|---|
| parent binaries (q > 0.1) | 180,236 | 177,147 (−1.7%, pull −9.3) | 179,064 (−0.65%, pull −3.1) |
| parent twins (q > 0.95) | 22,959 | 20,140 (−12.3%, pull −15.5) | 22,232 (−3.2%, pull −2.9) |
| parent log f > −0.5 (bright companions) | — | pull −17.1 | pull −6.4 |
| parent binary fraction, G < 15 | — | pull −15.9 | pull −6.6 |
| volume-limited binary fraction, 0.3–0.5 M⊙ (truth 0.333) | | 0.283 (−34σ) | 0.315 (−5σ) |
| volume-limited binary fraction, 0.5–0.8 M⊙ (truth 0.433) | | 0.394 (−37σ) | 0.439 (+3σ) |
| volume-limited twin share of companions (truth 0.118) | | 0.085 (−28σ) | 0.118 (0.1σ) |
| NSS-like window (α0 > 0.3 mas, 30 < P < 1600 d) | 1,056 | 994 (pull −1.9) | 983 (pull −2.2) |

**Without W the test fails visibly** (twins, bright companions, the binary fraction at bright G and low M1, and every volume-limited quantity, at 10–50σ). With W the bias drops by a factor of 3–30, to about 1% or less. A residual of 3–7σ survives at this N; see the next section for where it comes from.

The small run (3 × 10⁵ primaries, 33,419 rows, in the required gate) passes with every with-W pull below 3, while the no-W pulls reach −7 (log f) and −6 (twins).

Figures (`figures/`, large run):

- `closed_loop_parent_binary_fraction.png`: binary fraction vs G and vs M̂1, truth vs no-W vs with-W, with pulls.
- `closed_loop_parent_companions.png`: parent counts in log P, q, e and log f.
- `closed_loop_observed_alpha0.png`: photocentre α0 of the parent's binaries, the "observed system property".
- `closed_loop_volume_limited.png`: 1/V_S inversion vs the injected universe. "Parent truth / V_S" validates V_S itself.

`figures/closed_loop_results.json` has every number.

### Where the residual comes from: oracle runs (10⁶ primaries, 111,585 rows; `sensitivity_1e6.json`)

| run | total pull, with W | max \|pull\| with W, parent stats | twin-bin pull | max volume-limited z |
|---|---|---|---|---|
| production approximations (A1 + A2) | −1.6 | 4.0 | −0.4 | 5.0 |
| **true distance and true M1 given to the pipeline** | +0.3 | **2.2** | +1.2 | **1.8** |
| no W (reference) | −5.4 | 9.2 | −7.8 | 22 |

With the exact distance and M1 the weight closes to MC noise, so **the weight and the bookkeeping are right**. The residual comes from the two approximations of spec §9.3: A2, the Gaussian in μ from BJ quantiles (median σ_μ = 0.47 mag here, 0.94 mag at the 95th percentile), and A1, M1 = M̂1. A separate pair of runs showed that each one removes part of the G-dependent and q-dependent residual.

### Sensitivity to the open choices (same 10⁶ universe)

| pipeline setting (truth: σ_int = 0.15, zero point 0) | total pull | max \|pull\| | twin pull |
|---|---|---|---|
| matched | −1.6 | 4.0 | −0.4 |
| σ_int = 0.25 (too broad) | −3.6 | 8.0 | −4.6 |
| σ_int = 0.08 (too narrow) | −0.5 | 3.4 | +1.6 |
| M_G zero point +0.05 mag | +2.5 | 4.8 | +4.5 |
| M_G zero point −0.05 mag | **−5.8** | **10.5** | −5.3 |
| split-normal μ (MP-Q30 b, mode at r_med) | −5.9 | 11.0 | −6.9 |

- **The M_G zero point (MP-Q25) is the sharpest knob.** A −0.05 mag offset between the M_G(M1) relation and the real single-star sequence undoes the whole correction. On real data it must be calibrated to ≲ 0.02 mag, for example from the single-star ridge of the parent's own ΔM distribution (MP-Q25 a).
- An over-broad σ_int weakens W toward the naive draw, and that is the safe direction. An over-narrow one is mildly over-confident.
- The simple split normal (mode at r_med) is **worse** than the Gaussian, so it is not recommended. A quantile-matched split normal was not tested.

### What the correction does and does not move

The Öpik boost lives in **bright companions** (f ≳ 0.3: twins and near-twins). Their photocentre wobble is small, since α0 ∝ |q/(1+q) − f/(1+f)| → 0 for twins. So the parent's **α0 distribution and NSS-like window counts barely move** (no-W vs with-W differ by ≲ 1–3%, inside the noise). The correction matters for the luminous-binary population, its q and f shapes, and any inversion to volume-limited rates. It matters much less for the astrometric-orbit counts that #339 compares directly. Dark companions get no boost at all (spec §9.3).

## Findings outside #405's scope (filed)

- **#409**: the decided proposal's `e_cap = 0.95` truncates the MdS17 target. e_max(P) > 0.95 for P ≳ 180 d, so about 6% of luminous companions (2.2% at 100–1000 d) are never proposed, and the loss is fixed at generation.
- **#410**: under the uniform-e proposal, the provisional η floor of −0.9 makes the weights infinite-variance for log P ≲ 1.4. Finite-N IS totals come out low (Σw/N 0.488–0.494 vs 0.497 at 10 draws per row). The closed loop uses −0.4 on both sides.

## Options for Ryan (none chosen; spec §8)

- **MP-Q25** σ_int and the M_G zero point. The closed loop shows the zero point must be good to ≲ 0.02 mag. Options: (a) fit both from the parent's ΔM distribution, as singles + binaries per M̂1 bin, truncated at G = 19; (b) isochrone spreads; (c) constants.
- **MP-Q26** TAG10 on blended atmospheres (A1). Options: (a) ignore (now; the closed loop shows a percent-level residual at σ_logM1 = 0.02 dex, and real TAG10 scatter is likely larger); (b) model b(f).
- **MP-Q27** resolved pairs / IPD flags: blend everything (now) vs a separation threshold (Fabricius et al. 2021) vs an IPD likelihood.
- **MP-Q28** giants: W = 1 (now) / exclude / giant relation.
- **MP-Q29** extinction for ΔM: Combined19 vs `ag_gspphot` (biased for binaries). **This blocks applying W to real rows**: the parent snapshot has no A_G.
- **MP-Q30** distance marginalization: Gaussian (now; best measured) / quantile-matched split normal (untested) / apply W only above a ϖ/σ_ϖ threshold.
- **MP-Q31** F > 1 under a Poisson intensity: (1 − F) is clipped at 0 now. None of it occurs in the closed loop, where M1 ≤ 2.5 and F ≤ 0.7.
- **MP-Q32** `gaiaunlimited` for the volume-limited diagnostic: install it (a new dependency plus a map download) / skip / mask crowded fields.

## Reproduce

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 .venv/bin/python scripts/run_malmquist_closed_loop.py --size large   # ~3 min, ~6 GiB
.venv/bin/pytest tests/test_malmquist.py -m "unit or physics or api"   # incl. small closed loop, ~25 s
.venv/bin/pytest tests/test_malmquist.py -m slow                       # large oracle + production checks
```
