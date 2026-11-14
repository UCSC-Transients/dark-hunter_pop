# st2 routing gate (#696)

`routing_696.json` is written by `scripts/calibrate_st2_routing_696.py`. It is computed post hoc, exactly, from the stored
cascades of luminous gens 53–59 (rung-3 v5 weights) and compact-object gens 1006–1008 (co_v1 target weights), against
real DR3 Orbital + AstroSpectroSB1 published orbits. Domain: G 10–19.5, fitted P ≤ 830.4 d. Spec: EPOCH_MODEL_SPEC §8.16
(st2).

| weighted AstroSpectroSB1 share, luminous | G 10–11 | 11–12 | 12–13 | 10–13 | ≥ 13 |
|---|---|---|---|---|---|
| st1 (stored) | 0.570 ± 0.026 | 0.490 ± 0.054 | 0.199 ± 0.028 | 0.347 ± 0.029 | 0.003 |
| st2 (δ ≡ 0) | 0.646 ± 0.027 | 0.590 ± 0.053 | 0.325 ± 0.036 | 0.457 ± 0.031 | 0.004 |
| real DR3 | 0.629 | 0.574 | 0.342 | 0.458 | 0.004 |

The intercept re-solved on the weighted mock is −0.11 ± 0.14 / −0.13 ± 0.20 / +0.09 ± 0.27, and +0.00 ± 0.11 pooled. The
st1 intercept is −0.50 / −0.38 / −0.68.

Compact-object gens under st2, G 10–13: dark 0.70 ± 0.03, WD 0.51 ± 0.05. These are not like-for-like with DR3. Dark
draws sit beyond DR3's log K1 support (#699).

The overall luminous share is 0.224 vs 0.184. That comes from the weighted G < 13 fraction, 0.48 vs 0.40 (#698).

The replay tool is `scripts/replay_routing_696.py` (targeted mode). It was smoke-tested on gen 55 and co gen 1008: the
replayed changed draws matched the post-hoc type and cascade, and the unchanged draws were bit-identical.
