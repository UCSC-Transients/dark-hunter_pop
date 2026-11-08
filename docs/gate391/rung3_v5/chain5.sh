#!/bin/zsh
# Coordinator order (2026-11-08): #640 suite (running) -> #636 (+ #646 exports) -> #645 suite -> [#645 merged] R5-T11
# heavy steps B1 (train inputs), B2 (holdout inputs), F (first DR3 fit), C (gate-5b closed loop). One heavy job at a time.
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop; W=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees; V5=$W/rung3-v5
Q=$D/output/proposal_set/v21tr1_e2st1/queue.log; H=$D/scripts/jobs/run_heavy.sh; SP=$D/output/suite_prs; V=$D/output/proposal_set/v21tr1_e2st1
log() { echo "$1 $(date '+%F %T')" >> $Q; }
fail() { log "FAIL $1"; log "QUEUE STOPPED at step $1 (chain5)"; exit 1; }
merged() { [ "$(gh pr view $1 --json state -q .state 2>/dev/null)" = MERGED ]; }
waitmerged() { for i in {1..$2}; do merged $1 && return 0; sleep 120; done; return 1; }
suite() {  # suite <pr> <worktree> <logname>
  REPO=$2 LOG_DIR=$SP $H $3 -m pytest -m "unit or physics or api" -q -p no:cacheprovider
  local rc=$? res=$(grep -E "passed|failed" $SP/$3.log | tail -1)
  if [ $rc -eq 0 ]; then gh pr merge $1 --auto --merge >/dev/null 2>&1; log "SUITE PR $1 PASS ($res); auto-merge on"; return 0
  else log "SUITE PR $1 FAIL rc=$rc ($res); left open"; return 1; fi
}
# 1. #640
while pgrep -f "run_heavy.sh suite_640b" >/dev/null; do sleep 30; done
if ! grep -q "^exit 0" $SP/suite_640b.log 2>/dev/null || ! grep -q "SUITE PR 640 PASS" $Q; then
res=$(grep -E "passed|failed" $SP/suite_640b.log | tail -1)
if grep -q "^exit 0" $SP/suite_640b.log; then gh pr update-branch 640 >/dev/null 2>&1; gh pr merge 640 --auto --merge >/dev/null 2>&1; log "SUITE PR 640 PASS ($res); auto-merge on"
else log "SUITE PR 640 FAIL ($res); left open"; fi
fi
# 2. #636 + exports (#646)
cd $V5; export REPO=$V5 LOG_DIR=$V5/output/v5
C=config/population
COMMON=(--fit-config $C/rung3_joint_v5.yaml --cache $V/weights_gen53_59.npz --artifact-dir $V
 --artifacts single_gen53.h5 single_gen54.h5 single_gen55.h5 single_gen56.h5 single_gen57.h5 single_gen58.h5 single_gen59.h5
 --co-artifacts $V/co_gen1006.h5 $V/co_gen1007.h5 --co-fragments $C/proposal_set_co_v1_wd1006.yaml $C/proposal_set_co_v1_wd1007.yaml
 --calibration $V/spur/spurious_calibration_4s --fragment $C/proposal_set_single59.yaml --gate390 $D/output/gate390
 --parent-dir $D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --accel-snapshot $D/data/dr3/gaia_snapshots/20261009T205231Z_nss_acceleration_astro)
grep -q "STEP export_lum_v5 OK" $Q || { $H export_lum_v5 scripts/rung3_joint_391.py $COMMON --eval-only --init-from output/v5/fit/summary.json \
  --reference-from docs/gate391/joint561/fit/summary.json --fix ln_R_dark=-50 --out-dir output/v5/eval_export \
  --export-lum-weights $D/output/rung5/lum_weights_v5.npz || fail export_lum_v5
log "STEP export_lum_v5 OK (output/rung5/lum_weights_v5.npz; pihat_4s_dr3.npz exported earlier)"; }
grep -q "^exit 0" $SP/suite_636b.log 2>/dev/null || suite 636 $V5 suite_636b   # the Nov-7 pass predates the #646 commits
# 3. #645
W645=$D/.worktrees/rung5-t11; [ -d $W645 ] || W645=$(git -C $D worktree list | grep "\[feat/rung5-t11-fit\]" | awk '{print $1}')
gh pr update-branch 645 >/dev/null 2>&1; sleep 20; git -C $W645 fetch -q origin && git -C $W645 merge -q --ff-only origin/feat/rung5-t11-fit 2>/dev/null
suite 645 $W645 suite_645
waitmerged 645 180 || { log "chain5: #645 not merged in 6 h; R5-T11 steps not run"; exit 1; }
waitmerged 636 360 || { log "chain5: HOLD - #636 not merged (B1/B2 read rung3_joint_v5.yaml from main only)"; exit 1; }
git -C $D pull -q --ff-only origin main || fail pull_main
cd $D; export REPO=$D LOG_DIR=$D/output/rung5
WPC=config/population/rung3_joint_v5.yaml; [ -f $WPC ] || fail "rung3_joint_v5.yaml not on main"   # coordinator: inputs trace to main
IN=(--parent-dir data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
 --real-snapshot data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
 --real-input-columns data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns
 --co-artifact output/proposal_set/v21tr1_e2st1/co_gen1006.h5 output/proposal_set/v21tr1_e2st1/co_gen1007.h5 output/proposal_set/v21tr1_e2st1/co_gen1008.h5
 --co-fragment config/population/proposal_set_co_v1_wd1006.yaml config/population/proposal_set_co_v1_wd1007.yaml config/population/proposal_set_co_v1_dark1008.yaml
 --lum-artifact output/proposal_set/v21tr1_e2st1/single_gen53.h5 output/proposal_set/v21tr1_e2st1/single_gen54.h5 output/proposal_set/v21tr1_e2st1/single_gen55.h5 output/proposal_set/v21tr1_e2st1/single_gen56.h5 output/proposal_set/v21tr1_e2st1/single_gen57.h5 output/proposal_set/v21tr1_e2st1/single_gen58.h5 output/proposal_set/v21tr1_e2st1/single_gen59.h5
 --lum-weights output/rung5/lum_weights_v5.npz --pihat output/rung5/pihat_4s_dr3.npz --wd-physics-config $WPC --ln-r-center 6.62)
$H r5_inputs_train scripts/rung5_build_inputs.py --partition train $IN --out output/rung5/inputs_dr3_train.h5 || fail r5_inputs_train
log "STEP r5_inputs_train OK"
$H r5_inputs_holdout scripts/rung5_build_inputs.py --partition holdout $IN --out output/rung5/inputs_dr3_holdout.h5 || fail r5_inputs_holdout
log "STEP r5_inputs_holdout OK"
$H r5_fit scripts/rung5_fit_dr3.py fit --inputs output/rung5/inputs_dr3_train.h5 --out output/rung5/fit_dr3 --n-runs 3 --nlive 500 --map-starts 8 || fail r5_fit
log "STEP r5_fit OK"
$H r5_gate5b scripts/rung5_fit_dr3.py closed-loop --inputs output/rung5/inputs_dr3_holdout.h5 --fit-inputs output/rung5/inputs_dr3_train.h5 \
  --theta-ref output/rung5/fit_dr3/rung5_fit.json --out output/rung5/gate5b --n-real 20 || fail r5_gate5b
log "STEP r5_gate5b OK"
log "PAUSE after chain5"
