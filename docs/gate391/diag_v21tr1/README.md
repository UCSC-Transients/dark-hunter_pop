# Diagnostics on the colour_v2.1+tr1 generations (#589, #590, #591)

Ryan (2026-10-30) approved these as analyses of existing data: luminous gens 47–50 (`weights_gen47_50.npz`, default-target
MdS17 mixture weights unless noted) and DR3 data on disk. No simulation was run, and the queue stays paused.

Sources:
- same-orbit sample: the #390 published orbits re-injected under colour_v2.1+tr1 (`epoch-581/.../reinject_v21tr1`; 1,884 accepted realizations, 838 orbits with two);
- DR3 12×12 covariances: `corr_vec` / `bit_index` from `output/gate390/published_orbits.h5`, via `nss_covariance`; 166,086 of 168,065 rows rebuilt;
- DR3 C1 orbits and accelerations: as in the rung-3 fit (#587 dump).

Scripts: `scripts/diag_*_589.py`, `diag_*_590.py`, `diag_shortp_591.py`, and `diag_common_v21tr1.py` for the shared loaders.

## 1. The a0/σ_a0–RUWE edge (#589): [edge/](edge/), [width/](width/)

The statistic is z = (a0/σ_a0)/√(RUWE² − 1); its 99th percentile is the edge.

- **The σ_a0 propagation is not the cause.** Halbwachs first-order propagation of DR3's own Thiele–Innes covariance
  reproduces DR3's published σ_a0 (a0 / significance) exactly: median log ratio 1e-8, in every A bin. gaiamock_mod uses the
  same formula.
- **The coupling to (e, P, T_p) is real in DR3 but not the driver.** Its share f_nl of σ_a0² is 0.37 rising to 0.66 with A, and 0.94 at A > 1. The mock inflates σ_P and σ_e relative to σ_ϖ by only about 0.01 dex, so the predicted geometry
  excess is 0.001–0.006 dex. The observed same-orbit geometry excess (σ_a0/σ_ϖ, mock/DR3) is +0.007 to +0.029 dex, rising with A, and +0.06 at G < 11.5.
- **Faint stars (G ≥ 13): the offset is an eccentricity effect** ([width.json](width/width.json)).
  - z falls by about 0.35 dex from e < 0.2 to e > 0.6.
  - At fixed (G, P, e) the mock matches DR3 in median, σ68 and 99th percentile, all within about 0.05 dex.
  - The default-target mock is too eccentric (median e 0.39–0.42 vs 0.30–0.32). At the v3 θ the faint e distribution matches.
- **Bright stars (G < 13): two parts.**
  - **(i) The bright mock is too eccentric at the v3 θ:** median 0.42 vs DR3 0.29–0.32 at log P 2.5–3. DR3's bright and faint e
    distributions are the same.
    - This is not fitted-e bias (same-orbit Δe +0.004) and not acceptance (re-injection acceptance is flat in e).
    - It follows from MdS17's η steepening with M1: the bright primaries have median M1 1.06, the faint 0.83.
    - **Reweight-only:** an M1-dependent eccentricity-shape term.
  - **(ii) A residual at fixed (P, e):** DR3's median z is 0.03–0.09 dex higher and its 99th percentile 0.06–0.16 dex higher.
    - On the same orbits, DR3 scatters around the mock mean by more than the mock's realization noise, at G < 13:

      | quantity | extra scatter |
      |---|---|
      | log σ_a0 | 0.13 dex (DR3 0.148 vs mock 0.067) |
      | ½ log(RUWE² − 1) | 0.05 dex |
      | log σ_ϖ | 0.08 dex |
      | log z | 0.125 dex |

      At G ≥ 13 the extra scatter is 0.04–0.05 dex.
    - F2 inflation, (e, P, T_p) coupling and the orbit covariates explain almost none of it: the regression takes the residual MAD only from 0.093 to 0.089.
    - Smearing the population z by this amount brings the bright 99th percentile to DR3 (1.303 → 1.435 vs 1.439 at G 9–11.5; it overshoots slightly at 11.5–13). Population σ68 at fixed (G, P, A) is not larger in DR3, though, so only the tail and median matter.
- **Proposed fixes.**
  - (a) An M1-dependent eccentricity shape: reweight-only.
  - (b) For G < 13 only, a per-source astrometric error-scale term (lognormal, about 0.05–0.13 dex in σ_a0 and about 0.05 dex in the RUWE term). It can be applied post hoc to stored cascade outputs with the cuts re-applied, but routing changes need re-simulation for exactness.
  - **Pending:** a scan-pattern control (re-inject a subset at neighbouring GOST grid points) to subtract grid-point mismatch from the same-orbit dispersion. It is held for Ryan's approval: 1,200 cascades, about 3.2 CPU-h.

## 2. Bright high-F2 tail, F2 > 12 at G 11–12.5 (#590): [f2tail/](f2tail/)

- **Composition.** The DR3 tail is 78% dwarfs, 5% subgiants, 6% giants and 12% unclassified. The mock mix is the same when unweighted.
- **The deficit is in both classes and confined to G 10–12.** P(F2 > 12) at G 10–11 / 11–12, DR3 vs mock:
  - orbit dwarfs: 0.128 / 0.142 vs 0.040 / 0.085;
  - orbit evolved: 0.058 / 0.094 vs 0.016 / 0.013;
  - matched from G 12 down.
- **Accelerations.** The mock median F2 is low, more so for evolved stars: 10.8 vs 4.4 at G 10–11, against 11.6 vs 7.1 for dwarfs. Red stars ((BP−RP)_0 > 1) are worst.
- **Not convective photocentre jitter.**
  - Within a G bin, DR3's tail fraction does not depend on R★/d.
  - θ★ is 10–100 µas, so a jitter of a few per cent of R★ (Chiavassa et al. 2011) is ≤ 5 µas, far below σ_AL ~ 0.1 mas.
- **Same orbits.** DR3's per-orbit F2 scatter is about twice the mock's realization scatter at G < 12 (3.9–4.4 vs 2.0–2.4), with matched medians. That implies a per-source true-noise scale σ(log10 s) of 0.05–0.06 at G < 12, 0.04 at G 12–13 and about 0.02 fainter.
- **Verdict: a heavy-tailed bright error term**, i.e. a per-source noise scale s(G, colour) with e(G) kept for the medians.
  - **Calibration:** on the #390 same-orbit F2 dispersion and the 95th/99th percentiles per (G, colour), plus the acceleration
    re-injections. DR3 single-star gof_al cannot be used (it is about 0 by construction).
  - **Validation:** the bright P(F2 > 12) by class, the σ_ϖ envelope and the G = 13 step.
  - **Implementation:** post hoc on stored draws as an approximation; exact only with re-simulation.

## 3. Short-period bright systems, P < 100 d at G < 13 (#591): [shortp/](shortp/)

- **(a) No cascade-cut problem.** Re-injection acceptance at G < 13 is 0.83–0.88 at log P 2–2.92. Rejections are spread over the
  a0/σ_a0 √P, ϖ/σ_ϖ ∝ 1/P and σ_e cuts, with no short-P concentration. F2 never rejects, and periods are recovered.
  The #390 sample holds only 4 bright log P < 2 orbits.
- **(b) Population.**
  - 98% of DR3's 677 short-P bright orbits lie at d < 0.3 kpc (median 0.14). The "×2.8 at d ≥ 0.3" is 16 vs 6 objects.
  - The mock's short-P bright cell has ESS 13: gens 47–50 lack the nearby gen-46 recipe.
  - The shortfall sits in red K/M dwarfs: DR3/mock share 2.0–3.2 at (BP−RP)_0 > 0.8, while the mock has too many at 0.6–0.8.
  - Detection is M1-limited: the accepted fraction is 0.034 at M1 < 0.8 and 0 at M1 > 1.2.
  - Twins (q ≥ 0.95) are 40% of the short-P target weight but only 3% of accepted systems, so a P-dependent twin fraction is worth up to about 1.5×.
  - DR3 short-P e is 0.33 against 0.24 in the mock; the v3 fit has the short-P e-shape at its bound.
- **(c) DR3's short-P bright orbits are not spurious-like.**
  - F2, a0/σ_a0 and π̂ are the same as for long-P bright orbits (π̂ 0.114 vs 0.116).
  - Only 1.2% lie near 62.97 d / n, and the log P distribution is smooth.
- **Ranked causes.**
  1. Nearby low-mass MC coverage: re-simulation (gen 51 or a nearby low-mass target).
  2. Low-mass close-binary frequency (α_lo / an M1-dependent close-binary term): reweight-only once ESS allows.
  3. P-dependent twin fraction: reweight-only.
  4. Short-P e-shape: reweight-only.
  5. Cuts and spurious orbits: not supported.
