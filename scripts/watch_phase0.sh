#!/bin/bash
# Event stream for the Phase 0 job copies: state changes (sacct), step starts and failures from their logs.
# Usage: bash scripts/watch_phase0.sh JOBID[,JOBID...]   (exits when every copy is in a terminal state)
cd /scratch/gpfs/GROUP/USER/project/frozen-critic
JOBS=$1
declare -A seen
prev=""
while true; do
  st=$(sacct -n -X -j "$JOBS" -o JobID,State%24 2>/dev/null | awk '{printf "%s:%s ", $1, $2}')
  if [ -n "$st" ] && [ "$st" != "$prev" ]; then echo "STATE $st"; prev="$st"; fi
  for f in logs/slurm/fc-phase0-*.out; do
    [ -f "$f" ] || continue
    n=${seen[$f]:-0}; total=$(wc -l < "$f")
    if [ "$total" -gt "$n" ]; then
      sed -n "$((n+1)),${total}p" "$f" | grep -E "^=== |FAILED|Traceback|another copy|OutOfMemory|CUDA error|Killed" | sed "s#^#$(basename $f): #"
      seen[$f]=$total
    fi
  done
  if [ -n "$st" ] && ! echo "$st" | grep -qE "PENDING|RUNNING|REQUEUED|CONFIGURING"; then echo "ALL TERMINAL $st"; exit 0; fi
  sleep 120
done
