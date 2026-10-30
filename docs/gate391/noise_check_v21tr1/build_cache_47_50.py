"""#587: deterministic-mixture weight cache for the colour_v2.1+tr1 luminous generations 47-50 (one primary-model version).

Run from the repo root under scripts/jobs/run_heavy.sh. Refuses (check_single_primary_model inside build_cache) if the four
generations do not share one primary-model version; additionally requires the ...:noise_colour_v2.1+tr1 suffix, so the
old-noise gens 42-46 and the co_v1 gens 1000-1003 can never enter this cache.
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
V = D / "output/proposal_set/v21tr1"
arts = [V / f"single_gen{g}.h5" for g in (47, 48, 49, 50)]
for a in arts:
    with h5py.File(a, "r") as h:
        g = ps.ProposalConfig.model_validate_json(h.attrs["proposal_config_json"])
    v = ps.primary_model_version(g)
    print(f"{a.name}: generation {g.generation}, n_draws {g.n_draws}, version {v}", flush=True)
    if not v.endswith(":noise_colour_v2.1+tr1"):
        raise SystemExit(f"{a.name}: version {v} is not the adopted noise model colour_v2.1+tr1")
cfg = load_config()
prop = ps.load_proposal_set_fragment("config/population/proposal_set_single50.yaml").proposal
parent = ps.load_parent_snapshot(D / "data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj", cfg, prop)
build_cache(Namespace(artifact=arts, cache=V / "weights_gen47_50.npz", cmd_malmquist=Path("config/population/malmquist_cmd.yaml")), cfg, parent)
