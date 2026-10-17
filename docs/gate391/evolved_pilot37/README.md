# #391 spec §12.28: evolved-proposal pilot (generation 37)

- **Run.** Fragment `config/population/proposal_set_evolved37.yaml`, evolved rows only. 32,000 draws, 4.12 CPU-h, 37.6 min wall on 8 workers.
  - Run alone at git `b87e3b9`, under the 6 CPU-h watcher and mem_guard2. Neither tripped.
  - Decided-settings pre-check: OK.
  - Accepted orbits: 153. Evolved-detected draws (orbit or published acceleration): 356.
- **Check.** `scripts/evolved_pilot_check_391.py` → `summary.json`.
  - Pilot weights are for gen 37 alone. The reference is the gens 33–36 deterministic mixture (`weights_gen33_36.npz`, 154.2 CPU-h, of which 56.8 CPU-h on evolved draws).
  - The 2-D CMD W is 1 on evolved rows by design, so noW = cmdW here.
  - Recomputed log q matches the generation-time value to 7e-15.

| evolved-detected | gens 33–36 | gen 37 pilot |
|---|---:|---:|
| draws | 5,223 | 356 |
| Kish ESS | 23.9 | 44.0 |
| ESS per CPU-h (all CPU) | 0.155 | 10.67 (**69×**) |
| ESS per CPU-h (evolved-draw CPU) | 0.421 | 10.67 (**25×**) |
| top-1 / top-5 / top-1% weight share | 0.20 / 0.24 / 0.39 | 0.05 / 0.21 / 0.17 |
| weight share at the flux defensive floor | 0.20 | 5e-52 (≈ 0) |
| Σw (expected evolved detections) | 39,040 | 40,022 (ratio 1.025) |

- **Result.** Pass on both criteria.
  - ESS per CPU-h is ≥ 10× better on either CPU basis.
  - The flux-floor weight share is ≈ 0. 26% of pilot evolved-detected draws lie at the floor, but they carry negligible weight.
  - The two independent estimates of Σw agree to 2.5%.
- **Remaining tail.** The top 5 draws still hold 21% of the weight (24% in the reference), against top-1 0.05.
- **Definition of the floor.** A draw is at the floor when its own generation's flux density is within 1% of the defensive-uniform level. For the reference this gives 20%; the earlier 30% figure used a different cut.
