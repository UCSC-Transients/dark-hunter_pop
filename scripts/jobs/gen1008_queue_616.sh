#!/bin/zsh
# #616 (MOCK_POPULATION_SPEC §12.45; Ryan 2026-11-06): dark-focused compact-object gen 1008 under colour_v2.1+tr1+e2+st1.
# A 50-draw dry run, then gen 1008 (dark 0.8, gen-1004 recipe). Strictly one heavy job at a time; every step must
# exit 0 before the next starts. Generations run through start_generation.sh (nice, BLAS pinned, CPU-h budget, absolute
# wall-clock stop, mem_guard, <= 8 workers); a generation stopped by its guards is assembled from its prefix. Stops if
# free disk < MIN_FREE_GB. Ends with a PAUSE line.
# Usage: gen1008_queue_616.sh [first_step]   (steps: dry1008 g1008)
set -u
HERE=${0:A:h}; REPO=${REPO:-${HERE:h:h}}; cd $REPO
D=/Users/rfoley/darkhunter/pop/dark-hunter_pop
PAR=$D/data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj
OUT=${OUT:-$D/output/proposal_set/v21tr1_e2st1}; mkdir -p $OUT/dry607
export LOG_DIR=$OUT REPO
H=$HERE/run_heavy.sh; G=$HERE/start_generation.sh; Q=$OUT/queue.log
MIN_FREE_GB=${MIN_FREE_GB:-10}; W=${WORKERS:-8}
STEPS=(dry1008 g1008)
FIRST=${1:-dry1008}
C=config/population
log() { echo "$1 $(date '+%F %T') (git $(git rev-parse --short HEAD))" >> $Q; }
fail() { log "FAIL $1"; log "QUEUE STOPPED at step $1 (see $OUT/$1*.log); later steps not started"; exit 1; }
disk_ok() { local f=$(df -g $OUT | awk 'NR==2 {print $4}'); [ $f -ge $MIN_FREE_GB ] || { log "STOP disk free ${f} GB < $MIN_FREE_GB"; exit 1; }; }

dry() {  # dry <name> <fragment> <offset>: 50 draws into the scratch dir, all the generation code paths, no reserved output
  disk_ok; log "START $1"
  rm -f $OUT/dry607/$1.h5 $OUT/dry607/$1.partial.jsonl
  $H $1 scripts/run_proposal_pilot.py --parent-dir $PAR --fragment $2 --draw-index-offset $3 --out $OUT/dry607/$1.h5 \
    --workers 2 --nice 0 --n-draws 50 --progress-every 10 || { fail "$1"; }
  [ -f $OUT/dry607/$1.h5 ] || { fail "$1"; }
  log "STEP $1 OK"
}

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
    dry1008) dry dry1008 $C/proposal_set_co_v1_dark1008.yaml 102410000;;
    g1008) gen co_gen1008 $C/proposal_set_co_v1_dark1008.yaml 102410000 90 10;;  # dark-focused (gen-1004 recipe)
  esac
done
log "PAUSE after co_gen1008 (Ryan 2026-11-06, #616): gate 4b / nu re-check and the dark selection-function table follow; queue stopped"
