#!/bin/zsh
# #554: cross-evaluation dumps at FIXED theta (no optimisation, no new draws): theta(42-44) on data 42-46, theta(42-46) on data 42-44
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/proposal-restart-391
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop/; PAR=${D}data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
H=scripts/jobs/run_heavy.sh; O=output/proposal_set/v2_single; X=$O/diag46
BASE=(--model new --gate390 ${D}output/gate390 --parent-dir $PAR
 --real-snapshot ${D}data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv --real-input-columns ${D}data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot ${D}data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro --wkeys noW --fit-config config/population/rung3_coarse48_v12.yaml)
D46=(--new-gen 46 --fragment config/population/proposal_set_single46.yaml --cache $O/weights_gen42_46.npz
 --artifacts v2_single/single_gen42.h5 v2_single/single_gen43.h5 v2_single/single_gen44.h5 v2_single/single_gen45.h5 v2_single/single_gen46.h5)
D44=(--new-gen 44 --fragment config/population/proposal_set_single44.yaml --cache $O/weights_gen42_44.npz
 --artifacts v2_single/single_gen42.h5 v2_single/single_gen43.h5 v2_single/single_gen44.h5)
$H d554_t44_d46 scripts/rung3_refit5_391.py fit $BASE $D46 --dump-from docs/gate391/v2_single/v12/v12_gens42_44_noW/summary.json --out-dir $X/dump_t44_d46 || exit 1
$H d554_t46_d44 scripts/rung3_refit5_391.py fit $BASE $D44 --dump-from docs/gate391/v2_single/gen46/v12_noW/summary.json --out-dir $X/dump_t46_d44 || exit 1
$H d554_t44_d44 scripts/rung3_refit5_391.py fit $BASE $D44 --dump-from docs/gate391/v2_single/v12/v12_gens42_44_noW/summary.json --out-dir $X/dump_t44_d44 || exit 1
echo DONE > $X/xdump.done
