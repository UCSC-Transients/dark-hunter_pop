#!/bin/zsh
# #561 §12.37: joint luminous (gens 42-46) + co_v1 (gens 1000-1003) refit, then profiles (twins at θ44; N_s at prior).
# Sequential via run_heavy.sh (memory guard, nice); stops at the first failure.
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/joint-refit-391
export REPO=$PWD LOG_DIR=output/joint561
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; R=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/proposal-restart-391/output/proposal_set; P=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/rung4-co-391/output/proposal_set/co_v1; Q=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/wd-cooling-556/output/proposal_set/co_v1; C=config/population
COMMON=(--fit-config $C/rung3_joint_v1.yaml --cache $R/v2_single/weights_gen42_46.npz --artifact-dir $R
 --artifacts v2_single/single_gen42.h5 v2_single/single_gen43.h5 v2_single/single_gen44.h5 v2_single/single_gen45.h5 v2_single/single_gen46.h5
 --co-artifacts $P/co_gen1000_smoke.h5 $P/co_gen1001_recentred.h5 $P/co_gen1002_batch.h5 $Q/co_gen1003_wd.h5
 --co-fragments $C/proposal_set_co_v1_smoke.yaml $C/proposal_set_co_v1_recentred1001.yaml $C/proposal_set_co_v1_batch1002.yaml $C/proposal_set_co_v1_wd1003.yaml
 --fragment $C/proposal_set_single46.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro)
H=scripts/jobs/run_heavy.sh; O=output/joint561
$H joint_eval scripts/rung3_joint_391.py $COMMON --n-starts 1 --init-from $O/fit/summary.json --out-dir $O/fit_eval || exit 1
echo DONE > $O/joint_eval.done
