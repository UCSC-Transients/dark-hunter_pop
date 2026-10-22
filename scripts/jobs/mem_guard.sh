#!/bin/zsh
# #391 memory guard (v3): every 30 s, stop the job (recorded PID + ALL its descendants, PID-verified, no pattern kills) if
# swap use grows by more than SWAP_MB over its value when the guard started, or system free memory falls below FREE_PCT.
# Usage: mem_guard.sh <pidfile> <cmd-substring> <log> [SWAP_MB=4096] [FREE_PCT=10]
PF=$1; SUB=$2; L=$3; SW=${4:-4096}; FP=${5:-10}
until [ -s "$PF" ]; do sleep 5; done
PID=$(cat "$PF")
B=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); B=${B%.*}
echo "mem_guard watching PID $PID (swap growth > ${SW} MB over ${B} MB, or free < ${FP}%) from $(date)" >> $L
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
while kill -0 "$PID" 2>/dev/null; do
  U=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); U=${U%.*}
  F=$(memory_pressure -Q 2>/dev/null | sed -nE 's/.*percentage: ([0-9]+)%.*/\1/p')
  if [ $(( U - B )) -gt "$SW" ] || { [ -n "$F" ] && [ "$F" -lt "$FP" ]; }; then
    CMD=$(ps -o command= -p "$PID")
    case "$CMD" in
      *"$SUB"*) KIDS=($(desc $PID)); echo "MEM_GUARD STOP at $(date): swap ${U} MB (start ${B}), free ${F}%; killing PID $PID and descendants: ${KIDS[*]}" >> $L
               for k in ${KIDS[@]}; do kill $k 2>/dev/null; done; kill "$PID";;
      *) echo "MEM_GUARD skipped: PID $PID command is not the job ($CMD)" >> $L;;
    esac
    exit 0
  fi
  sleep 30
done
