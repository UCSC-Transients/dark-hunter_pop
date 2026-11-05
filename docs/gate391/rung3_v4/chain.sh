#!/bin/zsh
# #603 rung 3 v4 (spec §12.42): cache, routed-type check, joint fit. Strictly sequential via run_heavy.sh (nice, memory
# guard); stops at the first failure. Profiles and plots run from chain_profiles.sh / chain_plots.sh after the fit.
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/rung3-v4
export REPO=$PWD LOG_DIR=output/v4
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; V=$D/output/proposal_set/v21tr1_e2st1; C=config/population; G=docs/gate391/rung3_v4
COMMON=(--fit-config $C/rung3_joint_v4.yaml --cache $V/weights_gen53_57.npz --artifact-dir $V
 --artifacts single_gen53.h5 single_gen54.h5 single_gen55.h5 single_gen56.h5 single_gen57.h5
 --co-artifacts $V/co_gen1006.h5 --co-fragments $C/proposal_set_co_v1_wd1006.yaml
 --calibration $V/spur/spurious_calibration --fragment $C/proposal_set_single57.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro)
H=scripts/jobs/run_heavy.sh; O=output/v4
[ -f $V/weights_gen53_57.npz ] || { $H cache53_57 $G/build_cache_53_57.py || exit 1; }
$H routed_check57 scripts/check_routed_type_603.py --artifact $V/single_gen57.h5 --fragment $C/proposal_set_single57.yaml --n 20 --out $G/routed_type_check_gen57.json || exit 1
$H routed_check1006 scripts/check_routed_type_603.py --artifact $V/co_gen1006.h5 --fragment $C/proposal_set_co_v1_wd1006.yaml --n 10 --out $G/routed_type_check_gen1006.json || exit 1
$H joint_fit scripts/rung3_joint_391.py $COMMON --n-starts 12 --init-from docs/gate391/joint561/fit/summary.json --fix ln_R_dark=-50 --out-dir $O/fit || exit 1
echo DONE > $O/fit.done
