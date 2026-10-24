#!/bin/zsh
# #554: profile fits on the EXISTING gens 42-46 draws (no new draws): hold parameter groups at the gens 42-44 values / the prior
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/proposal-restart-391
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop/; PAR=${D}data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
H=scripts/jobs/run_heavy.sh; O=output/proposal_set/v2_single; X=$O/diag46
C=(fit --model new --gate390 ${D}output/gate390 --parent-dir $PAR
 --real-snapshot ${D}data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv --real-input-columns ${D}data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot ${D}data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro --wkeys noW --fit-config config/population/rung3_coarse48_v12.yaml
 --new-gen 46 --fragment config/population/proposal_set_single46.yaml --cache $O/weights_gen42_46.npz
 --artifacts v2_single/single_gen42.h5 v2_single/single_gen43.h5 v2_single/single_gen44.h5 v2_single/single_gen45.h5 v2_single/single_gen46.h5
 --n-starts 4 --init-from docs/gate391/v2_single/gen46/v12_noW/summary.json)
$H d554_prof_twin scripts/rung3_refit5_391.py $C --fix dgamma_q=0.401 ln_F_twin=-0.768 --out-dir $X/prof_twin44 || exit 1
$H d554_prof_ns scripts/rung3_refit5_391.py $C --fix N_s=21434 --out-dir $X/prof_ns_prior || exit 1
$H d554_prof_ecc scripts/rung3_refit5_391.py $C --fix a1=6.445 a2=4.338 a3=0.726 b1=30.0 b2=3.550 b3=1.0 --out-dir $X/prof_ecc44 || exit 1
$H d554_prof_twin_ns scripts/rung3_refit5_391.py $C --fix dgamma_q=0.401 ln_F_twin=-0.768 N_s=21434 --out-dir $X/prof_twin44_ns || exit 1
echo DONE > $X/profiles.done
