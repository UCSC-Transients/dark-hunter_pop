#!/bin/bash
# #391: stop a job at BUDGET CPU-h summed from a JSONL key, by recorded PID only (verified command line; no pattern kills).
# Usage: budget_stop.sh <jsonl> <pidfile> <cmd-substring> <budget_cpu_h> <log> <cpu_key>
P=$1; PF=$2; SUB=$3; B=$4; L=$5; K=$6
PY=/Users/rfoley/darkhunter/pop/dark-hunter_pop/.venv/bin/python
until [ -s "$PF" ]; do sleep 5; done
PID=$(cat "$PF")
while kill -0 "$PID" 2>/dev/null; do
  H=$($PY -c "
import json
s=0.0
try:
    for l in open('$P'):
        try: s+=json.loads(l).get('$K',0.0)
        except Exception: pass
except FileNotFoundError: pass
print(s/3600)")
  if $PY -c "import sys; sys.exit(0 if $H >= $B else 1)"; then
    CMD=$(ps -o command= -p "$PID")
    case "$CMD" in
      *"$SUB"*) KIDS=$(pgrep -P "$PID"); echo "BUDGET STOP at $(date): $H CPU-h >= $B; killing PID $PID and children: $KIDS" >> $L
               for k in $KIDS; do case "$(ps -o command= -p $k)" in *python*) kill $k;; esac; done; kill "$PID";;
      *) echo "BUDGET STOP skipped: PID $PID command is not the job ($CMD)" >> $L;;
    esac
    exit 0
  fi
  sleep 120
done
