# Sources without a usable BP−RP (#585)

**Failure.** On 2026-10-28 the #583 queue step `spur_far` stopped at 09:32 with "noise_colour needs a finite observed
BP-RP for every source". It had written 902 of 2,013 realizations, and those are kept: the run resumes.

**Rule** (EPOCH_MODEL_SPEC §8.13 note). A missing or non-finite observed BP−RP takes the colour term at the pivot,
c = 0: a G-only s on its side of the G = 13 break, and u0 at BP−RP = 1.2. The rule is the same for mock and real rows.

**Rows affected:**

| sample | rows without a finite BP−RP |
|---|---|
| luminous recipes, gens 42–46 (in-cut draws) | 0 of 1,240,836 |
| compact-object recipes, gens 1000–1003 (in-cut draws) | 0 of 505,777 |
| queue fragments, gens 47–52 and 1004–1005 (50-draw smokes, all run to completion) | 0 of 349 |
| real DR3 orbits (Orbital + AstroSpectroSB1) | 2 of 168,065 |
| looks-single parent rows, G < 19 (the colour of the mock rows in the §12.16 re-injection) | 5,562 of 214,666 (2.6%) |

- Generation draws always carry a MIST system colour, so the rule matters only for the §12.16 re-injection
  samples. Those use observed row colours for real orbits and for the matched old-generation mock orbits.
