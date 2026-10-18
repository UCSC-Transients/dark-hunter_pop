# #391: Ryan's CMD / twin test, the bright-end M̂1 split and σ_ϖ (Ryan, 2026-10-18; analysis only)

- **Scripts.**
  - `scripts/cmd_twin_391.py` → `cmd_twin.json`, `cmd_points.npz`.
  - `scripts/mult_sigma_391.py` → `mult_sigma.json`.
  - `scripts/cmd_twin_sheet_391.py` → `cmd_twin.png`.
- **Mock side.** The gens 33–38 noW best fit (`../resim_batch5/new_noW`, via `--dump-from`).
- **ΔM** = M_G0 − MS ridge(C0), using the §10.2 parent ridge.
  - Twin band ΔM −0.95 to −0.55; single band ±0.25.
  - Both are counted only at C0 0.85–2.0, i.e. the MS below the subgiants, per Ryan's constraint.

## Headline: the common cause at G < 10 is parent sparsity, not the isochrone fit

- Only **181 usable parent rows** have G < 10 (17 at G < 8). Each row stands for about 1,800 gaia_source stars.
- **Both extreme M̂1 bins rest on 2–3 rows.**
  - M̂1 < 0.8 M⊙ (×5.3): **3 parent rows**, nearby K dwarfs (M_G0 ≈ 6.4, d 10–40 pc, ΔM ≈ 0). One row carries 40% of the weight.
  - M̂1 2–2.5 M⊙ orbits: 10 rows, row-level ESS 2.8, top row 57%.
  - M̂1 2.5–3 M⊙: 2 rows, row-level ESS 1.9.
- With parent-row noise the per-bin excesses are 1.5–1.9σ each: G 5–8 orbits 2.59 ± 0.82, G 8–9 1.70 ± 0.41, G 9–10 1.38 ± 0.24.
- **Pooled over G < 10:**
  - orbits 1.59 ± 0.21 (2.9σ; 167 rows, row-level ESS 49);
  - accelerations 1.31 ± 0.14 (2.2σ).
- So a real but modest excess, about 1.3–1.6×, remains. The ×6 values are shot noise from the parent subsample.

## 0. Twin sequence and CMD shape (parent / real hosts / mock-detected)

| d (kpc) | 0–0.1 | 0.1–0.2 | 0.2–0.5 | 0.5–1 |
|---|---:|---:|---:|---:|
| parent: twin / single | 0.11 | 0.008 | 0.046 | 0.078 |
| real orbit hosts | 0.38 | 0.36 | 0.32 | 0.44 |
| real acceleration hosts | 0.13 | 0.29 | 0.32 | 0.48 |
| mock-detected orbits | 0.13 | 0.015 | 0.080 | 0.90 (ESS low) |
| mock-detected accelerations | 0.26 | 0.008 | 0.064 | 0.39 |

The 1–3 kpc slice is excluded: at C0 > 0.85 it is mostly giants.

- **The ridge and its width have the same shape in every slice.** Real NSS hosts carry a 5–40× heavier twin band than the parent at d < 0.5 kpc, as expected for binaries with luminous companions.
- **Mock-detected systems sit where the parent sits** (twin/single 0.01–0.08). That follows from the convention that the mock's observed CMD is its parent row's CMD (MP-Q6/Q36), and most rows are single-like.
- **No double-counted light.** Each mock primary is deblended against the row's total light, so a mock binary on a single-like row gets a fainter, lower-mass primary. The twin-band rows (real binaries) are not preferentially chosen. The §12.22 CMD conditioning upweights them only for M1 ≥ 0.8.
- **Consequence.** The mock and real hosts are compared at the same G but at different places relative to the ridge. Real twin-band hosts get an inflated M̂1 (next point); mock hosts mostly do not.
- **The isochrone fit treats unresolved twins as heavier, metal-rich singles.**
  - On parent MS rows at G 13–16, the twin band gets M̂1 0.95 against 0.79 on the ridge (+20%), and [Fe/H] +0.23 against −0.07 (+0.3 dex). G 10–13 is the same.
  - This is Ryan's hypothesis, confirmed for the MS below the subgiants.
  - It cannot drive the G < 10 high-mass excess, which needs mock M̂1 above real.
- **Bright failures are not on the twin sequence.**
  - Success on twin-band rows is 0.95–1.0.
  - Parent G < 11 failures are `off_grid` (G < 10: 73 of 73) at M_G0 −2.7 to −0.1, ΔM −6 to −11. These are luminous red giants and upper-MS B/A stars outside the MIST prior grid (panel 2).
  - Real orbit hosts at G < 10 are rarely that luminous, so the asymmetry (parent 58–76% success vs real 84–94%) is a population difference.

## 1. The 2–3 M⊙ excess

- **Finer M̂1 bins at G < 10, mock ÷ real:**

| M̂1 (M⊙) | 1.35–1.5 | 1.5–1.75 | 1.75–2 | 2–2.5 | 2.5–3 |
|---|---:|---:|---:|---:|---:|
| orbits | 0.97 | 1.4 | 2.3 | 6.3 | 6.0 |
| accelerations | 0.76 | 1.2 | 3.3 | 3.6 | 6.1 |

  Above 2 M⊙ these rest on 2–11 parent rows (row-level ESS 2–9).
- **Multiplicity is not the cause.**
  - MdS17 companions per star (q > 0.1) in log P 1–4: 0.14 at 1 M⊙, 0.19 at 1.5, 0.23 at 2, 0.27 at 2.5, 0.30 at 3.
  - So P(another companion in the window) is 0.13 at 1 M⊙ and 0.23 at 2.5 M⊙.
  - Treating extra companions as independent can bias 2–3 M⊙ relative to 1 M⊙ by about 1.12× at most.
- **Flux ratio.** Mock 2–3 M⊙ G < 10 orbits are mostly luminous-companion systems: log f −2 to −1 holds 57%, above −1 holds 23%, q 0.4–0.6 holds 53%. Not dark.
- **Teff of 1.5–3 M⊙ G < 10 hosts.** GSP-Phot medians are 6,750 K for real orbit hosts and 6,890 K for the parent (56 rows). Above 7,500 K: 23% of real hosts against 39% of the parent.
  - A hint that DR3 finds fewer orbits for hot (A-type) primaries. The parent statistics are thin.
  - v sin i (gaia_source `vbroad`) is not in the local snapshots and would need a fetch.

## 3. σ_ϖ: gaiamock vs DR3

- **Data.** The #390 set, 3 realizations each, v2 + N2-u0. No visibility-period loss; that variant has not been re-run.
- **How σ_ϖ is derived.**
  - gaiamock takes the marginal σ_ϖ from the full 12-parameter Gauss–Newton covariance, (JᵀJ)⁻¹, scaled by its fit-quality factor cc = √(χ²_red / (1 − 2/(9ν))³), which is equivalent to an F2-based inflation.
  - DR3's `parallax_error` for NSS solutions is the marginal error from the solution covariance (corr_vec).
  - **Whether DR3 applied the same F2 inflation to Orbital solutions is not verified here.** That is the next check.

| log P | 2.0–2.3 | 2.3–2.5 | 2.5–2.7 | 2.7–3.0 |
|---|---:|---:|---:|---:|
| σ_ϖ recovered / published, away from the cut | 1.32 | **1.13** | 1.10 | 1.13 |
| systems away from the cut | 111 | 336 | 381 | 1,584 |
| part explained by mock F2 above DR3's (cc ratio) | 1.13 | 1.05 | 1.03 | 1.08 |

- Mock F2 is higher than DR3's at every log P, e.g. a median of 5.5 against 2.6 at log P 2.3–2.5.
- Published max |ρ(ϖ, other parameters)| from corr_vec is 0.31 at log P 2.3–2.5 and peaks at 0.52 at log P 2.5–2.7 (P ≈ 1 yr). So orbit–parallax degeneracy is moderate in the short-P window.
- **Result.** The mock σ_ϖ is about 10–30% above DR3 for orbits away from the cut. About half comes from the higher mock F2, through the error inflation; the rest is noise or epoch sampling. That fits the distant short-P deficit near the ϖ/σ_ϖ cut.

## Proposed fixes (none implemented; each needs Ryan)

1. **Bright-end parent densification** (data fetch; needs approval):
   - add a denser gaia_source subsample at G < 11, e.g. `random_index < 2×10⁷` restricted to G < 11, which gives about 20× the rows;
   - add a bright parent-target component.
   - This turns the G < 10 test from row-limited into a measurement. Until then, report G < 10 as a 1.3–1.6× excess at 2–3σ.
2. **σ_ϖ / F2:**
   - verify DR3's NSS error definition (whether an F2 inflation is applied);
   - find why mock F2 runs above DR3's (per-CCD noise model or epoch count);
   - then decide whether to match F2 or σ_ϖ.
3. **Twin-band asymmetry.** Compare mock and real hosts at matched ΔM, or extend the §12.22 conditioning below 0.8 M⊙. The isochrone M̂1/[Fe/H] bias on twin rows (+20%, +0.3 dex) is a documented systematic for the data-side M1.
4. **Hot primaries.** Fetch `vbroad` for the bright 1.5–3 M⊙ hosts and parent rows (needs approval) to test a DR3 detection deficit for A stars.
