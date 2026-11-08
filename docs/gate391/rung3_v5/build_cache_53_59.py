"""#615: deterministic-mixture weight cache for the colour_v2.1+tr1+e2+st1 luminous generations 53-59 (one primary-model version).

Run from the repo root under scripts/jobs/run_heavy.sh. Refuses (check_single_primary_model inside build_cache) if the four
generations do not share one primary-model version; additionally requires the ...:noise_colour_v2.1+tr1+e2+st1 suffix, so
older-version gens (42-50) and every co_v1 generation can never enter this cache.
"""
import sys
from argparse import Namespace
from pathlib import Path

import h5py

sys.path.insert(0, "scripts")
from rung3_refit_391 import build_cache  # noqa: E402

from darkhunter_pop import proposal_set as ps  # noqa: E402
from darkhunter_pop.config_loader import load_config  # noqa: E402

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V = D / "output/proposal_set/v21tr1_e2st1"
arts = [V / f"single_gen{g}.h5" for g in (53, 54, 55, 56, 57, 58, 59)]
for a in arts:
    with h5py.File(a, "r") as h:
        g = ps.ProposalConfig.model_validate_json(h.attrs["proposal_config_json"])
    v = ps.primary_model_version(g)
    print(f"{a.name}: generation {g.generation}, n_draws {g.n_draws}, version {v}", flush=True)
    if not v.endswith(":noise_colour_v2.1+tr1+e2+st1"):
        raise SystemExit(f"{a.name}: version {v} is not the adopted noise model colour_v2.1+tr1+e2+st1")
cfg = load_config()
prop = ps.load_proposal_set_fragment("config/population/proposal_set_single59.yaml").proposal
parent = ps.load_parent_snapshot(D / "data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj", cfg, prop)
build_cache(Namespace(artifact=arts, cache=V / "weights_gen53_59.npz", cmd_malmquist=Path("config/population/malmquist_cmd.yaml")), cfg, parent)
import numpy as np  # noqa: E402

z = np.load(V / "weights_gen53_59.npz", allow_pickle=True)
cnt = {int(g): int(np.sum(np.asarray(z["generation"]) == g)) for g in z["gens"]}
print("n_per_gen", dict(zip([int(g) for g in z["gens"]], [int(n) for n in z["n_per_gen"]])), "stored", cnt, flush=True)
if [cnt[int(g)] for g in z["gens"]] != [int(n) for n in z["n_per_gen"]]:
    raise SystemExit("n_per_gen differs from the stored draw counts (partial generation not assembled as n_draws)")
