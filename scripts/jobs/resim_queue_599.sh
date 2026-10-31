#!/bin/zsh
# #599 (MOCK_POPULATION_SPEC §12.41; Ryan 2026-10-31): unattended re-simulation queue under the adopted noise/epoch model
# colour_v2.1+tr1+e2+st1 (modelled on resim_queue_583.sh). New OUT directory and new generation numbers / draw_index ranges
# (10,000,000-12,499,999 and 101,610,000-102,009,999; disjoint from everything used so far). Ends with a PAUSE line: the
# luminous top-up (gen-52 recipe) and the dark compact-object recipe (gen 1004) wait for a joint-refit check. Strictly one heavy job at a time; every step must exit 0 before the next starts (a generation counts as
# done only when its own "wrote" line is logged or its prefix assembly exits 0). Generations run through
# start_generation.sh (nice, BLAS pinned, CPU-h budget, absolute wall-clock stop, mem_guard); the small spurious steps run
# through run_heavy.sh (nice, mem_guard). Stops if free disk < MIN_FREE_GB. Each finished step appends one line to
# $OUT/queue.log ("STEP <name> OK ..."), which the operator reports.
# Usage: resim_queue_599.sh [first_step]   (steps: spur_far spur_near spur_fit g53 g54 g55 g56 g57 g1006)
set -u
HERE=${0:A:h}; REPO=${REPO:-${HERE:h:h}}; cd $REPO
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop
PAR=$D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
OLD=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/proposal-restart-391/output/proposal_set
OUT=${OUT:-$D/output/proposal_set/v21tr1_e2st1}; mkdir -p $OUT/spur
export LOG_DIR=$OUT REPO
H=$HERE/run_heavy.sh; G=$HERE/start_generation.sh; Q=$OUT/queue.log
MIN_FREE_GB=${MIN_FREE_GB:-10}; W=${WORKERS:-8}
STEPS=(spur_far spur_near spur_fit g53 g54 g55 g56 g57 g1006)
FIRST=${1:-spur_far}
REAL=(--real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
      --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns)
log() { echo "$1 $(date '+%F %T') (git $(git rev-parse --short HEAD))" >> $Q; }
fail() { log "FAIL $1"; log "QUEUE STOPPED at step $1 (see $OUT/$1*.log); later steps not started"; exit 1; }
disk_ok() { local f=$(df -g $OUT | awk 'NR==2 {print $4}'); [ $f -ge $MIN_FREE_GB ] || { log "STOP disk free ${f} GB < $MIN_FREE_GB"; exit 1; }; }

gen() {  # gen <name> <fragment> <offset> <cpu_h> <wall_h>
  local name=$1 fr=$2 off=$3 bud=$4 wh=$5 out=$OUT/$1.h5
  disk_ok
  if [ -f $out ]; then log "STEP $name SKIP (exists)"; return 0; fi
  log "START $name $fr offset $off budget ${bud} CPU-h ${wh} h"
  $G $fr $out $off $PAR $W $bud $(( $(date +%s) + wh * 3600 ))
  if ! grep -q "^wrote " $OUT/$name.log; then
    $H ${name}_assemble scripts/assemble_partial_generation.py --parent-dir $PAR --fragment $fr \
      --draw-index-offset $off --partial $OUT/$name.partial.jsonl --out $out || { fail "$name"; }
  fi
  [ -f $out ] || { fail "$name"; }
  log "STEP $name OK"
}

run=0
for s in $STEPS; do
  [ $s = $FIRST ] && run=1
  [ $run = 1 ] || continue
  case $s in
    spur_far)
      disk_ok; log "START spur_far"
      $H spur_far scripts/detection_vs_population_391.py --artifact-dir $OLD --cache $OLD/weights_gen23_27.npz \
        --parent-dir $PAR $REAL --gate390 $D/output/gate390 --n-inject 300 --n-real 3 --workers $W \
        --inject-log $OUT/spur/far.jsonl --out-dir $OUT/spur/far || { fail "spur_far"; }
      log "STEP spur_far OK";;
    spur_near)
      disk_ok; log "START spur_near"
      $H spur_near scripts/detection_vs_population_391.py --artifact-dir $OLD --cache $OLD/weights_gen23_27.npz \
        --parent-dir $PAR $REAL --gate390 $D/output/gate390 --n-inject 130 --n-real 3 --workers $W \
        --d-range 0.0 0.5 --g-range 0.0 30.0 --inject-log $OUT/spur/near.jsonl --out-dir $OUT/spur/near || { fail "spur_near"; }
      log "STEP spur_near OK";;
    spur_fit)
      log "START spur_fit"
      $H spur_fit scripts/calibrate_spurious_391.py --log far=$OUT/spur/far.jsonl --log near=$OUT/spur/near.jsonl \
        --old-cache $OLD/weights_gen23_27.npz --gate390 $D/output/gate390 --parent-dir $PAR \
        --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv \
        --out $OUT/spur/spurious_calibration || { fail "spur_fit"; }
      log "STEP spur_fit OK";;
    g53) gen single_gen53 config/population/proposal_set_single53.yaml 10000000 90 10;;   # gen-42 recipe
    g54) gen single_gen54 config/population/proposal_set_single54.yaml 10500000 90 10;;   # gen-43 recipe
    g55) gen single_gen55 config/population/proposal_set_single55.yaml 11000000 40 5;;    # gen-44 recipe
    g56) gen single_gen56 config/population/proposal_set_single56.yaml 11500000 90 10;;   # gen-45 recipe
    g57) gen single_gen57 config/population/proposal_set_single57.yaml 12000000 90 10;;   # nearby gen-46 recipe
    g1006) gen co_gen1006 config/population/proposal_set_co_v1_wd1006.yaml 101610000 90 10;;  # WD-focused (gen-1005 recipe)
  esac
done
log "PAUSE after co_gen1006 (Ryan 2026-10-31): gen-52 top-up and dark CO gen-1004 recipes wait for a joint-refit check; queue stopped"
