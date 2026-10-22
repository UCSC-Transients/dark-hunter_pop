#!/bin/zsh
# #391: run ONE memory-heavy python job (nice, BLAS pinned, /usr/bin/time -l) under mem_guard.sh. Returns the job's exit
# code and kills leftover descendants, so a chain (use `run_heavy.sh ... || exit 1` per step) never overlaps jobs.
# Env: REPO (default: this repo), PY (default: $REPO/../../dark-hunter_pop/.venv/bin/python), LOG_DIR (default output/proposal_set).
# Usage: run_heavy.sh <name> <python args...>
HERE=${0:A:h}
REPO=${REPO:-${HERE:h:h}}
PY=${PY:-/Users/rfoley/darkhunter/pop/dark-hunter_pop/.venv/bin/python}
LOG_DIR=${LOG_DIR:-output/proposal_set}
cd $REPO
mkdir -p $LOG_DIR
N=$1; shift; L=$LOG_DIR/$N.log; echo "start $(date)" >> $L
env PYTHONPATH=$REPO/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 /usr/bin/time -l $PY "$@" >> $L 2>&1 &
PID=$!; echo $PID > $LOG_DIR/$N.pid
for i in {1..40}; do KIDS=($(pgrep -P $PID)); [ ${#KIDS} -gt 0 ] && break; kill -0 $PID 2>/dev/null || break; sleep 0.5; done
nohup $HERE/mem_guard.sh $LOG_DIR/$N.pid time $L > /dev/null 2>&1 &
wait $PID; RC=$?
for k in ${KIDS[@]}; do kill -0 $k 2>/dev/null && { case "$(ps -o command= -p $k)" in *python*) kill $k; echo "killed leftover $k" >> $L;; esac; }; done
echo "exit $RC at $(date)" >> $L
exit $RC
