#!/bin/zsh
# #615 rung 3 v5 + coordinator queue (2026-11-06). Strictly one heavy job at a time via run_heavy.sh (nice, mem_guard).
# Science steps stop the chain on failure; suite steps record PASS/FAIL and continue. Lines go to queue.log.
# Order: cache53_59 -> v5 fit -> v5 profiles -> suites #617/#619/#622 -> gen 1008 (dry + gen) -> v5 plots + compare
#        -> suite #623 -> suite #625 -> gate5a inject + report. HELD (not here): #621 M1 table, #620 real M2_app.
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; W=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees
V5=$W/rung3-v5; cd $V5
V=$D/output/proposal_set/v21tr1_e2st1; Q=$V/queue.log; C=config/population; G=docs/gate391/rung3_v5
H=$D/scripts/jobs/run_heavy.sh; O=output/v5
log() { echo "$1 $(date '+%F %T')" >> $Q; }
fail() { log "FAIL $1"; log "QUEUE STOPPED at step $1 (#615 chain)"; exit 1; }
disk() { local f=$(df -g $V | awk 'NR==2 {print $4}'); [ $f -ge 10 ] || { log "STOP disk free ${f} GB < 10"; exit 1; }; }
suite() {  # suite <pr> <worktree>
  REPO=$2 LOG_DIR=$D/output/suite_prs $H suite_$1 -m pytest -m "unit or physics or api" -q -p no:cacheprovider
  local rc=$? res=$(grep -E "passed|failed" $D/output/suite_prs/suite_$1.log | tail -1)
  if [ $rc -eq 0 ]; then gh pr merge $1 --auto --merge >/dev/null 2>&1; log "SUITE PR $1 PASS ($res); auto-merge on"
  else log "SUITE PR $1 FAIL rc=$rc ($res); left open"; fi
}
export REPO=$V5 LOG_DIR=$O
COMMON=(--fit-config $C/rung3_joint_v5.yaml --cache $V/weights_gen53_59.npz --artifact-dir $V
 --artifacts single_gen53.h5 single_gen54.h5 single_gen55.h5 single_gen56.h5 single_gen57.h5 single_gen58.h5 single_gen59.h5
 --co-artifacts $V/co_gen1006.h5 $V/co_gen1007.h5 --co-fragments $C/proposal_set_co_v1_wd1006.yaml $C/proposal_set_co_v1_wd1007.yaml
 --calibration $V/spur/spurious_calibration_4s --fragment $C/proposal_set_single59.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro)
REF=(--reference-from docs/gate391/joint561/fit/summary.json)
disk
if [ ! -f $O/fit/summary.json ]; then
  log "START v5 cache+fit"
  [ -f $V/weights_gen53_59.npz ] || { $H cache53_59 $G/build_cache_53_59.py || fail cache53_59; }
  $H fit_v5 scripts/rung3_joint_391.py $COMMON --n-starts 12 --init-from docs/gate391/rung3_v4/fit/summary.json $REF --fix ln_R_dark=-50 --out-dir $O/fit || fail fit_v5
  log "STEP v5 fit OK"
fi
if [ ! -f $O/profiles/profile.json ]; then
  $H prof_v5 scripts/rung3_joint_391.py $COMMON --n-starts 1 --init-from $O/fit/summary.json $REF --fix ln_R_dark=-50 --out-dir $O/profiles \
    --profile dgamma_q=0.0,0.2,0.4,0.6 ln_F_twin=-1.5,-0.75,0.0,0.75 dln_F_twin_sP=-4.0,-2.0,0.0,2.0 kappa_cb=-4.0,-2.0,0.0,2.0,4.5 \
    N_s=prior,26000,35000,45000 || fail prof_v5
  log "STEP v5 profiles OK"
fi
suite 617 $D/.worktrees/rung5-t1
suite 619 $D/.worktrees/rung5-t13
suite 622 $D/.worktrees/rung5-t2
disk
REPO=$W/topup-607 $W/topup-607/scripts/jobs/gen1008_queue_616.sh || fail gen1008
export REPO=$V5 LOG_DIR=$O
NC=docs/gate391/noise_check_v21tr1; V4D=$W/rung3-v4/output/v4/fit; V3D=$W/rung3-v3/output/v3/fit
$H plots_v5 scripts/rung3_plots_391.py --dump $O/fit/plot_dump.npz --summary $O/fit/summary.json --out-dir $G/plots --boot-2d \
  --title "Rung 3 v5: DR3 against the weighted mock (luminous gens 53-59 + WD co 1006/1007 + spurious)" \
  --intro "Spec §12.44 (#615). Mock = luminous (gens 53–59, colour_v2.1+tr1+e2+st1) + WD (co_v1 gens 1006/1007, §12.39 physics-derived rate) + spurious (real orbits × N_s π̂/Σπ̂, #609 four-sample template); R_dark = 0. Orbit splits by solution type use the post-hoc routed st1 type for the mock and the DR3 type for real (and spurious) orbits. " || fail plots_v5
$H rank_v5 scripts/rung3_plots_391.py --dump $O/fit/plot_dump.npz --summary $O/fit/summary.json --out-dir $G/v5_rank_v3edges --edges-in $NC/v3_rank/edges.json --rank-only --boot-2d || fail rank_v5
$H compare_v5 scripts/rung3_v4_compare_603.py --edges $NC/v3_rank/edges.json \
  --run "v3=$V3D/plot_dump.npz:$NC/v3_rank:docs/gate391/rung3_v3/fit/summary.json" \
  --run "v4=$V4D/plot_dump.npz:docs/gate391/rung3_v4/v4_rank_v3edges:docs/gate391/rung3_v4/fit/summary.json" \
  --run "v5=$O/fit/plot_dump.npz:$G/v5_rank_v3edges:$O/fit/summary.json" --out-dir $G/compare || fail compare_v5
log "STEP v5 plots OK"
suite 623 $D/.worktrees/rung5-t2b
suite 625 $D/.worktrees/rung5-t7
REPO=$D/.worktrees/rung5-t7 LOG_DIR=$D/output/rung5 $H gate5a scripts/rung5_known_objects_5a.py inject --out $D/output/rung5/gate5a --workers 4 || fail gate5a
REPO=$D/.worktrees/rung5-t7 LOG_DIR=$D/output/rung5 $H gate5a_report scripts/rung5_known_objects_5a.py report --out $D/output/rung5/gate5a || fail gate5a_report
log "STEP gate5a OK"
log "PAUSE after gate5a (#615 chain): held steps #621 M1 table and #620 real M2_app wait for the coordinator"
