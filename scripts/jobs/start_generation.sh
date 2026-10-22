#!/bin/zsh
# #391: launch ONE proposal generation detached (nice, BLAS pinned, decided-settings check in the driver) with its three
# guards: CPU-h budget, absolute wall-clock stop (epoch seconds), and mem_guard.sh. PID-only stops.
# Usage: start_generation.sh <fragment> <out.h5> <draw_offset> <parent_dir> <workers> <cpu_h_budget> <stop_epoch>
HERE=${0:A:h}; REPO=${REPO:-${HERE:h:h}}; PY=${PY:-/Users/rfoley/darkhunter/pop/dark-hunter_pop/.venv/bin/python}
cd $REPO
FR=$1; OUT=$2; OFF=$3; PAR=$4; W=${5:-8}; BUD=$6; STOP=$7
B=${OUT%.h5}; L=$B.log; PF=$B.pid
echo "start $(date) (git $(git rev-parse --short HEAD))" >> $L
env PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
  $PY scripts/run_proposal_pilot.py --parent-dir $PAR --fragment $FR --draw-index-offset $OFF --out $OUT --workers $W --nice 0 --progress-every 20000 >> $L 2>&1 &
PID=$!; echo $PID > $PF; echo "python PID $PID (pidfile $PF)" >> $L
nohup $HERE/budget_stop.sh $B.partial.jsonl $PF run_proposal_pilot $BUD $L cpu_seconds > /dev/null 2>&1 &
nohup $HERE/wallclock_stop.sh $PF run_proposal_pilot $STOP $L > /dev/null 2>&1 &
nohup $HERE/mem_guard.sh $PF run_proposal_pilot $L > /dev/null 2>&1 &
caffeinate -i -w $PID
wait $PID; echo "exit $? at $(date)" >> $L
