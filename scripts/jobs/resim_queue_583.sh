#!/bin/zsh
# #583 (MOCK_POPULATION_SPEC §12.40; Ryan 2026-10-28): unattended re-simulation queue under the adopted noise/epoch model
# colour_v2.1+tr1. Strictly one heavy job at a time; every step must exit 0 before the next starts (a generation counts as
# done only when its own "wrote" line is logged or its prefix assembly exits 0). Generations run through
# start_generation.sh (nice, BLAS pinned, CPU-h budget, absolute wall-clock stop, mem_guard); the small spurious steps run
# through run_heavy.sh (nice, mem_guard). Stops if free disk < MIN_FREE_GB. Each finished step appends one line to
# $OUT/queue.log ("STEP <name> OK ..."), which the operator reports.
# Usage: resim_queue_583.sh [first_step]   (steps: spur_far spur_near spur_fit g47 g48 g49 g50 g51 g52 g1004 g1005)
set -u
HERE=${0:A:h}; REPO=${REPO:-${HERE:h:h}}; cd $REPO
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop
PAR=$D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
OLD=/Users/rfoley/darkhunter/pop/dark-hunter_pop-worktrees/proposal-restart-391/output/proposal_set
OUT=${OUT:-$D/output/proposal_set/v21tr1}; mkdir -p $OUT/spur
export LOG_DIR=$OUT REPO
H=$HERE/run_heavy.sh; G=$HERE/start_generation.sh; Q=$OUT/queue.log
MIN_FREE_GB=${MIN_FREE_GB:-10}; W=${WORKERS:-8}
STEPS=(spur_far spur_near spur_fit g47 g48 g49 g50 g51 g52 g1004 g1005)
FIRST=${1:-spur_far}
REAL=(--real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv
      --real-input-columns $D/data/dr3/gaia_snapshots/20261003T182211Z_nss_orbit_input_columns)
log() { echo "$1 $(date '+%F %T') (git $(git rev-parse --short HEAD))" >> $Q; }
disk_ok() { local f=$(df -g $OUT | awk 'NR==2 {print $4}'); [ $f -ge $MIN_FREE_GB ] || { log "STOP disk free ${f} GB < $MIN_FREE_GB"; exit 1; }; }

gen() {  # gen <name> <fragment> <offset> <cpu_h> <wall_h>
  local name=$1 fr=$2 off=$3 bud=$4 wh=$5 out=$OUT/$1.h5
  disk_ok
  if [ -f $out ]; then log "STEP $name SKIP (exists)"; return 0; fi
  log "START $name $fr offset $off budget ${bud} CPU-h ${wh} h"
  $G $fr $out $off $PAR $W $bud $(( $(date +%s) + wh * 3600 ))
  if ! grep -q "^wrote " $OUT/$name.log; then
    $H ${name}_assemble scripts/assemble_partial_generation.py --parent-dir $PAR --fragment $fr \
      --draw-index-offset $off --partial $OUT/$name.partial.jsonl --out $out || { log "FAIL $name assemble"; exit 1; }
  fi
  [ -f $out ] || { log "FAIL $name no artifact"; exit 1; }
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
        --inject-log $OUT/spur/far.jsonl --out-dir $OUT/spur/far || { log "FAIL spur_far"; exit 1; }
      log "STEP spur_far OK";;
    spur_near)
      disk_ok; log "START spur_near"
      $H spur_near scripts/detection_vs_population_391.py --artifact-dir $OLD --cache $OLD/weights_gen23_27.npz \
        --parent-dir $PAR $REAL --gate390 $D/output/gate390 --n-inject 130 --n-real 3 --workers $W \
        --d-range 0.0 0.5 --g-range 0.0 30.0 --inject-log $OUT/spur/near.jsonl --out-dir $OUT/spur/near || { log "FAIL spur_near"; exit 1; }
      log "STEP spur_near OK";;
    spur_fit)
      log "START spur_fit"
      $H spur_fit scripts/calibrate_spurious_391.py --log far=$OUT/spur/far.jsonl --log near=$OUT/spur/near.jsonl \
        --old-cache $OLD/weights_gen23_27.npz --gate390 $D/output/gate390 --parent-dir $PAR \
        --real-snapshot $D/data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/query.ecsv \
        --out $OUT/spur/spurious_calibration || { log "FAIL spur_fit"; exit 1; }
      log "STEP spur_fit OK";;
    g47) gen single_gen47 config/population/proposal_set_single47.yaml 7000000 90 10;;
    g48) gen single_gen48 config/population/proposal_set_single48.yaml 7500000 90 10;;
    g49) gen single_gen49 config/population/proposal_set_single49.yaml 8000000 40 5;;
    g50) gen single_gen50 config/population/proposal_set_single50.yaml 8500000 90 10;;
    g51) gen single_gen51 config/population/proposal_set_single51.yaml 9000000 90 10;;
    g52) gen single_gen52 config/population/proposal_set_single52.yaml 9500000 90 10;;
    g1004) gen co_gen1004 config/population/proposal_set_co_v1_resim1004.yaml 100810000 90 10;;
    g1005) gen co_gen1005 config/population/proposal_set_co_v1_wd1005.yaml 101210000 90 10;;
  esac
done
log "QUEUE DONE"
