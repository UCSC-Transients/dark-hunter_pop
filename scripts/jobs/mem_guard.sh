#!/bin/zsh
# #391 memory guard (v4, spec §12.35 amendment). Every INTERVAL s (default 30), stop the job (recorded PID + ALL its
# descendants; PID-verified; no pattern kills) when
#   (a) system free memory < FREE_PCT % or the kernel memory-pressure level is critical (4), OR
#   (b) swap use has grown by > SWAP_MB since start AND our own job tree is large: tree RSS > RSS_PCT % of RAM, or tree
#       RSS has grown by > RSS_GROW_MB since start.
# (b) no longer fires on swap growth caused by other applications alone. Every stop logs its reason.
# Usage: mem_guard.sh <pidfile> <cmd-substring> <log> [SWAP_MB=4096] [FREE_PCT=10]
# Test hooks (env): MG_SWAP_FILE / MG_RSS_FILE (read each poll), MG_SWAP_MB, MG_FREE_PCT, MG_PRESSURE, MG_TREE_RSS_MB, MG_RAM_MB override the system readings.
PF=$1; SUB=$2; L=$3; SW=${4:-4096}; FP=${5:-10}
INTERVAL=${INTERVAL:-30}; RSS_PCT=${RSS_PCT:-50}; RSS_GROW_MB=${RSS_GROW_MB:-4096}
until [ -s "$PF" ]; do sleep 1; done
PID=$(cat "$PF")
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
swap_mb() { [ -n "$MG_SWAP_FILE" ] && { cat "$MG_SWAP_FILE"; return; }; [ -n "$MG_SWAP_MB" ] && { echo $MG_SWAP_MB; return; }; local u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); echo ${u%.*}; }
free_pct() { [ -n "$MG_FREE_PCT" ] && { echo $MG_FREE_PCT; return; }; memory_pressure -Q 2>/dev/null | sed -nE 's/.*percentage: ([0-9]+)%.*/\1/p'; }
pressure() { [ -n "$MG_PRESSURE" ] && { echo $MG_PRESSURE; return; }; sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0; }
ram_mb() { [ -n "$MG_RAM_MB" ] && { echo $MG_RAM_MB; return; }; echo $(( $(sysctl -n hw.memsize) / 1048576 )); }
tree_rss_mb() { [ -n "$MG_RSS_FILE" ] && { cat "$MG_RSS_FILE"; return; }; [ -n "$MG_TREE_RSS_MB" ] && { echo $MG_TREE_RSS_MB; return; }; local s=0 p r; for p in $PID $(desc $PID); do r=$(ps -o rss= -p $p 2>/dev/null); s=$(( s + ${r:-0} )); done; echo $(( s / 1024 )); }
B=$(swap_mb); R0=$(tree_rss_mb); RAM=$(ram_mb)
echo "mem_guard v4 watching PID $PID from $(date): swap start ${B} MB, tree RSS start ${R0} MB, RAM ${RAM} MB" >> $L
while kill -0 "$PID" 2>/dev/null; do
  U=$(swap_mb); F=$(free_pct); LV=$(pressure); R=$(tree_rss_mb)
  REASON=""
  if [ -n "$F" ] && [ "$F" -lt "$FP" ]; then REASON="free memory ${F}% < ${FP}%"
  elif [ "$LV" -ge 4 ]; then REASON="kernel memory pressure critical (level ${LV})"
  elif [ $(( U - B )) -gt "$SW" ]; then
    if [ "$R" -gt $(( RAM * RSS_PCT / 100 )) ]; then
      REASON="swap +$(( U - B )) MB and job tree RSS ${R} MB > ${RSS_PCT}% of RAM"
    elif [ $(( R - R0 )) -gt "$RSS_GROW_MB" ]; then
      REASON="swap +$(( U - B )) MB and job tree RSS grew $(( R - R0 )) MB"
    fi
  fi
  if [ -n "$REASON" ]; then
    CMD=$(ps -o command= -p "$PID")
    case "$CMD" in
      *"$SUB"*) KIDS=($(desc $PID)); echo "MEM_GUARD STOP at $(date): ${REASON}; killing PID $PID and descendants: ${KIDS[*]}" >> $L
               for k in ${KIDS[@]}; do kill $k 2>/dev/null; done; kill "$PID";;
      *) echo "MEM_GUARD skipped: PID $PID command is not the job ($CMD)" >> $L;;
    esac
    exit 0
  fi
  sleep $INTERVAL
done
