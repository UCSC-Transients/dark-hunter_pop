#!/bin/zsh
# #603: plots, v3 comparison, then profiles (one setup per parameter, start from the fit, same reference θ as the fit).
cd /Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/rung3-v4
export REPO=$PWD LOG_DIR=output/v4
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; V=$D/output/proposal_set/v21tr1_e2st1; C=config/population; G=docs/gate391/rung3_v4
NC=docs/gate391/noise_check_v21tr1; V3=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/rung3-v3/output/v3/fit
H=scripts/jobs/run_heavy.sh; O=output/v4
$H plots_v4 scripts/rung3_plots_391.py --dump $O/fit/plot_dump.npz --summary $O/fit/summary.json --out-dir $G/plots --boot-2d \
  --title "Rung 3 v4: DR3 against the weighted mock (luminous gens 53-57 + WD co 1006 + spurious)" \
  --intro "Spec §12.42 (#603). Mock = luminous (gens 53–57, colour_v2.1+tr1+e2+st1) + WD (co_v1 gen 1006, §12.39 physics-derived rate) + spurious (real orbits × N_s π̂/Σπ̂, #601 template); R_dark = 0. Orbit splits by solution type use the post-hoc routed st1 type for the mock and the DR3 type for real (and spurious) orbits. " || exit 1
$H rank_v4_common scripts/rung3_plots_391.py --dump $O/fit/plot_dump.npz --summary $O/fit/summary.json --out-dir $G/v4_rank_v3edges --edges-in $NC/v3_rank/edges.json --rank-only --boot-2d || exit 1
$H compare_v3 scripts/rung3_v4_compare_603.py --edges $NC/v3_rank/edges.json \
  --run "v3=$V3/plot_dump.npz:$NC/v3_rank:docs/gate391/rung3_v3/fit/summary.json" --run "v4=$O/fit/plot_dump.npz:$G/v4_rank_v3edges:$O/fit/summary.json" --out-dir $G/compare || exit 1
COMMON=(--fit-config $C/rung3_joint_v4.yaml --cache $V/weights_gen53_57.npz --artifact-dir $V
 --artifacts single_gen53.h5 single_gen54.h5 single_gen55.h5 single_gen56.h5 single_gen57.h5
 --co-artifacts $V/co_gen1006.h5 --co-fragments $C/proposal_set_co_v1_wd1006.yaml
 --calibration $V/spur/spurious_calibration --fragment $C/proposal_set_single57.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro)
$H profiles scripts/rung3_joint_391.py $COMMON --n-starts 1 --init-from $O/fit/summary.json --reference-from docs/gate391/joint561/fit/summary.json \
  --fix ln_R_dark=-50 --out-dir $O/profiles --profile dgamma_q=0.0,0.15,0.45,0.6 ln_F_twin=-1.5,-0.9,0.4,1.0 \
  dln_F_twin_sP=-4.0,-2.0,0.0,1.0 N_s=prior,20000,30000,38000 || exit 1
echo DONE > $O/post.done
