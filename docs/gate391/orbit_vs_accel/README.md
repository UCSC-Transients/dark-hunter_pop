# #391 / #402 orbit : acceleration ratio, mock vs real

Ryan (#402, 2026-10-09): settle the ≈ 0.8 detection factor from `../detection_vs_population/` (PR #455). This is analysis only: no config change and no new draws.

- Scripts:
  - `scripts/fetch_nss_acceleration_snapshot.py` builds the snapshot.
  - `scripts/orbit_vs_accel_391.py` runs the comparison.
- Outputs: `report.txt`, `summary.json`, `orbit_accel_profiles.png`, `mock_orbit_accel_by_true_period.png`.

## Data

- **Real accelerations.**
  - Snapshot `data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro/`: `table.h5` + `meta.yaml`, holding the ADQL, date and sha256 `9cdc2a35…`.
  - Contents: 338,215 rows (Acceleration7 246,947, Acceleration9 91,268), taken from `gaiadr3.nss_acceleration_astro` joined to `gaia_source` and `astrophysical_parameters`.
  - Parallax: the NSS solution's parallax.
- **Fetching.** astroquery's async job hung on the joined query, so the fetch uses direct TAP `sync` in 36 source_id-range chunks. Each chunk is retried and checked against its own `COUNT(*)`; the total is checked against the global count (338,215 = 338,215, all unique).
- **Mirror filters.** These are the same as for the orbits: `real_comparison_keep` (parallax floor, atmosphere, Halbwachs IPD/C*). They leave **337,960**. Distance is the zero-point-corrected 1/ϖ, as for the orbits.
- **Mock.** Gens 23–27 with noW / cmdW weights.
  - Orbits: `accepted_orbital` with fitted P ≤ 830 d (C1).
  - Accelerations: `published_acceleration`, i.e. a 7/9-parameter outcome passing the DR3 publication cuts.
  - Distance: 1/(fitted parallax).
- Real orbits are subject to C1 on the published P.

## Results

| sample | real orbit/accel | mock orbit/accel (noW / cmdW) | mock/real orbits | mock/real accelerations | ratio of ratios (mock/real) |
|---|---|---|---|---|---|
| all | 131,169 / 337,960 = 0.388 | 0.465 / 0.472 | 0.66 / 0.58 | 0.55 / 0.48 | 1.20 ± 0.05 / 1.22 ± 0.05 |
| **0.7–1.5 kpc, G 13–16** | 30,017 / 81,434 = 0.369 | 0.381 / 0.388 | 0.51 / 0.44 | 0.50 / 0.42 | **1.03 ± 0.12 / 1.05 ± 0.13** |

The ratio-of-ratios errors come from the Kish ESS (orbits 108 / 94, accelerations 206 / 163 in the bin) plus Poisson noise on the real counts.

- **The mock does not have too many accelerations.**
  - In the bin, the orbit/acceleration ratio matches.
  - Overall, the mock has *fewer* accelerations per orbit (×1.2). That comes from G < 11 and d < 0.3 kpc, where mock orbits are at about 1× real but mock accelerations stay at about 0.5×.
- **Mock/real accelerations are about 0.4–0.6 at every G and distance.** The long-P (P ≳ 10³ d) companions behind the accelerations are under-produced uniformly. This is a population statement: the deficit appears even nearby, where orbits match.
- **Mock, true P 450–830 d in the bin:** 19% of published solutions are accelerations (3,031 of 15,549). At P 830–2000 d it is 97%, and above 2000 d it is 100%. Accelerations from P 450–830 d are 7% of the mock's bin accelerations.

## Verdict

**The orbit-vs-acceleration decision rule is not the gap.**
- If gaiamock's acceleration-first rule sent 20% too many of the bin's orbits to accelerations, the mock's orbit/acceleration ratio would be about 0.8 × real. It is 1.03 ± 0.12 (noW) and 1.05 ± 0.13 (cmdW). That excludes 0.8 at about 1.9σ, and it bounds a rule-driven orbit loss to ≲ 10% (1σ).
- No pop-side or upstream rule change is indicated. For the record, the options would be:
  - (a) a pop-side post-cascade relabel of 7/9-par outcomes with P ≤ 830 d that pass the orbital cuts;
  - (b) an upstream gaiamock change to run an orbital fit when the acceleration solution is significant, as DR3 does.
  
  Neither is supported by this test.

**The ≈ 0.8 from the symmetric rung-1 test is therefore a property of the real orbits, not a mock detection loss.**
- About 21% ± 4% of the bin's real published orbits ((0.754 − 0.599) / 0.754) are not re-detected by a Keplerian forward model at their own fitted parameters. They fall back to accelerations 2–4× as often.
- These are most likely spurious or non-Keplerian solutions: about **6,300 ± 1,100 of 30,017** in the bin.
- The matched orbit/acceleration ratio then means the real accelerations carry a comparable fraction of the same kind of contamination. DR3 accelerations are known to have high spurious rates. So the ratio test alone cannot isolate it.

**Corrected detection factor to carry into rung 3: 1.0** (the orbit-vs-acceleration bound is ≈ ±0.12 at 1σ in the bin).
- Treat the ≈ 21% as contamination of the real sample, to be handled by the shared spuriousness model, not as a mock detection correction.
- With that, the bin's mock/real orbit ratio (0.51 noW / 0.44 cmdW) becomes about 0.65 / 0.56 against the non-spurious real orbits. That residual is population, and the accelerations show the same uniform ×0.5 for the long-P companions.

## Caveats

- The bin ESS is about 100 for orbits and about 200 for accelerations, which sets the ±0.12.
- Mock accelerations use gaiamock's 7/9-parameter publication cuts. The Acceleration9 share is lower in the mock: 22% of mock accelerations vs 27% of real.
- The spurious estimate rests on the 300-system symmetric re-injection of PR #455.
