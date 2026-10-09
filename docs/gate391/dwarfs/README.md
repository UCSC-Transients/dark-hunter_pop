# #391 rung 2, dwarf primaries only (generations 23–26)

The mock and the real sample are both cut to dwarf primaries with the same classifier: the CMD dwarf/evolved cut of spec §10.2 (n_σ = 3), using the main-sequence ridge measured on the parent. This excludes the evolved rows, whose weights still have a heavy tail until the evolved-row flux fix (#391 option (i)) lands.

**Inputs**
- **Mock:** 817k draws from generations 23–26, combined with deterministic-mixture weights; 10,609 accepted dwarf orbits. MdS17 at published parameters, luminous companions only.
- **Real:** 137,744 DR3 Orbital + AstroSpectroSB1 rows after the mirror filters and the CMD dwarf cut. Real distances are 1/ϖ from the NSS solution, as in `gate_giants`.

**Headline numbers**

| | No Malmquist weight | 2-D CMD Malmquist weight |
|---|---|---|
| Accepted-set ESS | **879** | **656** |
| Largest single-draw weight share | 0.011 | 0.012 |
| Expected accepted orbits vs 137,744 real | 8.8×10⁴ (0.64×) | 7.6×10⁴ (0.55×) |

**KS test.** Computed on the bins with ESS ≥ 30 (MP-Q19). Each cell gives D and p, then the share of the real sample that lies in those bins.

| Panel | No W | With W | Verdict |
|---|---|---|---|
| P | 0.061, 0.003 (98%) | 0.069, 0.005 (91%) | small but significant offset: the mock lacks some orbits near 10³ d |
| G | 0.084, 1×10⁻⁵ (98%) | 0.100, 1×10⁻⁵ (96%) | the mock is slightly brighter |
| 1/ϖ | 0.094, 6×10⁻⁷ (82%) | 0.108, 6×10⁻⁷ (75%) | the mock is too close |
| e | 0.166, 8×10⁻²² (99%) | 0.161, 1×10⁻¹² (81%) | the mock is too eccentric; largest disagreement |
| f_m | 0.078, 2×10⁻⁵ (83%) | 0.094, 1×10⁻⁵ (80%) | the mock has no tail above ≈ 0.1 M⊙ |
| cos i | 0.034, 0.26 (91%) | 0.048, 0.10 (81%) | **agrees** |

At this ESS (≈ 900), the KS test can detect offsets of D ≈ 0.05. Every panel except cos i now differs significantly, though P, G and f_m differ only by D < 0.1. The Malmquist weight does not improve any panel.

Reports: `rung2_dwarf_{noW,cmdW}_report.txt`.
