# #390 Step 1a injection test: pilot results

Pilot run: 270 published DR3 systems (216 `Orbital`, 54 `AstroSpectroSB1`), stratified 3×3×3 by
G, log P and log(a0/σ_a0), with 5 noise realizations each, so 1,350 realizations. Seeds come from
`base_seed = 42` via `injection_rng_seeds`. Run with 8 workers in 74 min wall clock (laptop load
average ~30). There were 0 errors. The final write-up is in `../README.md`; these figures and
`summary.json` are the early checkpoint.

- Artifact: `output/gate390/injection_test_pilot.h5` (gitignored),
  sha256 `d327a67b751136bd37527add93a2f9042c020829e82cd87b4c474e2f4036ddfd`.
- Cost: 17.2 CPU s per realization reaching the orbital fit, 0.01 s otherwise, 14.4 s on average.
  Peak worker RSS is 0.22 GiB.
- Acceptance (orbit fit + all El-Badry 2024 Eq. 18/20–22 cuts): Orbital 0.735 ± 0.013,
  AstroSpectroSB1 0.856 ± 0.021.
- Pulls of the accepted realizations, σ_MAD: P 0.90, e 0.93, a0 0.97, ϖ 1.01, cos i 0.89 (Orbital).
  The medians are within ±0.12, except a0 (+0.12 Orbital, +0.27 AstroSpectroSB1).
- Recovered σ / published σ, median for Orbital: P 0.87, e 0.89, a0 0.89, ϖ 0.90.
