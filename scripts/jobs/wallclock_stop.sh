#!/bin/zsh
# #391: stop the job in <pidfile> (and its children) at an absolute epoch second; PID-verified, no pattern kills.
PF=$1; SUB=$2; AT=$3; L=$4
until [ -s "$PF" ]; do sleep 5; done
PID=$(cat "$PF")
while kill -0 "$PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$AT" ]; then
    CMD=$(ps -o command= -p "$PID")
    case "$CMD" in *"$SUB"*)
      KIDS=$(pgrep -P "$PID" | tr '\n' ' ')
      echo "WALLCLOCK STOP at $(date): killing PID $PID and children: $KIDS" >> "$L"
      kill $KIDS 2>/dev/null; kill "$PID" ;;
    esac
    exit 0
  fi
  sleep 30
done
