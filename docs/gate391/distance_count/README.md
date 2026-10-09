# #391: why the mock is too close and too few (generations 23–27)

This is analysis only: no new draws and no configuration changes. The script is `scripts/diagnose_distance_count_391.py`; the numbers are in `report.txt`.

**Inputs**
- **Mock:** MdS17 at its published parameters (luminous companions only), reweighted onto 837k proposal draws from generations 23–27. Results are shown without the CMD Malmquist weight ("no W") and with it ("with W").
- **Real:** the 167,911 Orbital + AstroSpectroSB1 rows that survive the mirror filters.

**Headline:** the mock predicts 1.09 × 10⁵ accepted orbits without W and 9.7 × 10⁴ with W. That is 0.65× and 0.58× the real count.

## Verdicts

| Test | Result | Verdict |
|---|---|---|
| **A. Where is the deficit?** Absolute counts, mock/real (no W) | 1/ϖ: **1.03** at < 0.2 kpc, 0.76 at 0.3, 0.72 at 0.5, 0.52 at 0.7, 0.50 at 0.9, 0.45 at 1.1, 0.34–0.38 at 1.3–1.5 kpc. G: about 1 at G < 10, falling to 0.4–0.6 at G = 13–16. **Flat (0.55–0.8)** in l, b, ecliptic β, the crowding proxy (log parent density 2.9–4.6 deg⁻²) and visibility periods. Only one bin dips: 0.48 at β ≈ −7° (sky map `A_count_ratio_sky.png`) | **The deficit tracks distance, not position, crowding or scan frequency** |
| **B. Is the truth-distance scale off?** | Simulated vs observed parallax, all 837k draws. At ϖ/σ > 20, truth − observed is **+22 to +37 µas** at every G; that is the Gaia parallax zero point. Bailer-Jones r_med_geo uses zero-point-corrected parallaxes (Lindegren et al. 2021), but the observed `parallax` is uncorrected. The pull rises to +1.5σ at the highest ϖ/σ, and is −0.2 to −0.4σ at ϖ/σ < 4 (prior shrinkage, expected). Weak trend with \|β\| (+0.13 → +0.46σ), consistent with the zero point's position dependence | **Real but small.** The mock's 1/ϖ sits about 3% too close at 1 kpc (about 6% at 2 kpc) when compared with uncorrected NSS parallaxes. This does not explain a factor of 2 |
| **C. Orbit fraction vs distance** (N_orbit/N_parent, d = 1/observed ϖ) | Mock and real agree at d < 0.2 kpc in every G and colour bin. Beyond that, the mock fraction falls faster in **every** G bin (e.g. G < 12 at 0.9 kpc: real 1.0e-2, mock 3.2e-3; 14 ≤ G < 16 at 0.6 kpc: 3.7e-3 vs 1.8e-3) and every colour (M1-proxy) bin. **Real orbits with f_m > 0.1** make up 4% at < 0.2 kpc, rising to 7%, 11%, 15%, 21% and 43% with distance. The mock is at 2–4% between 0.2 and 1 kpc. The median photocentre a0 in AU at 0.7–1 kpc is 0.54 real vs 0.46 mock | **Mostly population, not detection.** At larger distance only large photocentre orbits are detected, and the mock lacks them. That points to the missing dark/WD companions (MP-Q17 compact mixture off) and the high-f_m tail. Rung 1 (#390) validated detection at fixed true orbits, so detection is not the prime suspect |
| **D. Eccentricity vs distance and G** | The mean-e excess is **uniform**: +0.08 to +0.11 in every distance bin (0.1–1.25 kpc) and every G bin. Real 0.32–0.39, mock 0.38–0.49 | **Population** (MdS17 η at published parameters), not selection or fit bias |

## Fixes

| Cause | Fix | Reweight or re-simulate? |
|---|---|---|
| Missing dark/WD companions and high-f_m tail (C) | Turn on a compact-companion mixture (MP-Q17; WD at least), then compare the count and 1/ϖ by distance again | **Reweightable** (no new draws). The proposal already draws f = 0 companions (`dark_fraction` 0.1) and M2 up to 50 M⊙. Coverage of WD-mass dark companions should be checked through ESS; a top-up may be needed if it is low |
| Remaining distance deficit after that | Fit the MdS17 frequency and q at the M1 of distant stars (rung 3) | Reweightable |
| Parallax zero point (B) | Compare against zero-point-corrected real parallaxes (Lindegren et al. 2021) in the 1/ϖ panel and in d-binned tests | **Analysis-level** (no new draws). The truth (zero-point-corrected Bailer-Jones) is physically right. The alternative, simulating the uncorrected observed parallax, would be **generation-time** |
| e excess (D) | MdS17 η / MP-Q11 floor, fitted at rung 3 | Reweightable |

**Not tested:** a targeted gaiamock detection check. B and C do not point at detection: the deficit is flat in sky, crowding and scan proxies, and nearby counts match. It can be run if Ryan wants to rule detection out explicitly.

Figures: `A_count_ratio_profiles.png`, `A_count_ratio_sky.png`, `B_parallax_pulls.png`, `C_orbit_fraction.png`, `D_eccentricity_vs_distance_G.png`.
