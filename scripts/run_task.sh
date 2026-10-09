#!/bin/bash
# One frozen-critic task inside a (possibly packed) Slurm job: the critic's value server on the last GPU of this task's
# GPU list (shared with miles: the 1B actor leaves room), then miles in the image on the task's GPUs.
#   run_task.sh RUN_NAME GPU_LIST LANE     (train args from runs/RUN_NAME/train_args, one per line)
# LANE (0, 1) offsets the value-server port and Ray's port block so two tasks can share a node.
# FLA_DISABLE_BACKEND_DISPATCH=1: Gated DeltaNet runs on fla's own Triton kernels (Triton 3.7.1 in the image) instead
# of flash_qla. Job 15202658 (tests/gdn_backend_grads.py) found the two agree (max relative gradient difference 0.4-0.8%,
# deterministic across calls), so flash_qla is NOT the cause of the 1.8e12 step-0 grad norm seen in the smokes; the
# switch stays only so all three lambda arms run the same kernels.
# Retry: NCCL "Cuda failure 999 'unknown error'" in the first all_reduce after the colocate wake_up hit 3 of 62 miles runs
# since 2026-09-28 on 3 different nodes (14952940, 14986291, 15168368), so it is treated as transient: a main run that
# dies with it restarts in place from its latest checkpoint (--load = --save), at most MAX_TRY attempts in all. Any other
# failure, and any smoke copy, fails at once. NCCL_DEBUG=WARN prints the failing CUDA call; ECC/row-remap state and
# kernel Xid lines are saved to runs/RUN/gpu_diag.JOBID.TRY.txt after each failure.
# NCCL_NVLS_ENABLE=0: with NCCL_DEBUG=WARN the failure in job 15202994 (lambda 0.5, rollout 126) points at
# transport/nvls.cc:257. miles rebuilds its process groups at every colocate wake_up, so each rollout sets up NVLS
# (NVLink SHARP multicast) buffers again; the failures come after tens of rollouts, at the first collective after a
# rebuild. NVLS is off so all-reduce uses plain NVLink rings and trees (no measurable cost for a 1B model on 4 GPUs).
# Stall watchdog: with NVLS off the same spot can hang instead (another project's job 14984569, 2026-10-09: 30 min at
# 0% GPU after the first post-wake_up collective, no NCCL warning, distributed timeout of 10 min never fired). Health-check
# lines keep the log growing during such a hang, so the watchdog counts progress lines ("rollout|step|eval N: {"): none for
# STALL_MIN minutes (default 30; a healthy step takes under a minute) kills the attempt's process group, logs a watchdog
# line and retries from the checkpoint like the 999 failure.
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
RUN_NAME=$1 DEVS=$2 LANE=$3
RUN_DIR=$ROOT/runs/$RUN_NAME
mapfile -t ARGS < "$RUN_DIR/train_args"
IFS=, read -ra DEV <<< "$DEVS"; NG=${#DEV[@]}; CRIT=${DEV[$((NG - 1))]}
VP=$((40000 + (SLURM_JOB_ID % 2000) * 2 + LANE))          # above Ray's job-derived block (20000-32700)
RAY_BASE=$((20000 + ((SLURM_JOB_ID * 2 + LANE) % 1270) * 10))
# compute-node /tmp is small and shared (owner rule 2026-10-08): this task's temp space lives on scratch; the lane dir
# becomes the container's /tmp (HOME, Triton/Inductor caches, Ray), and TMPDIR points there for every child process
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID/lane$LANE
JOBTMP=$TMPDIR; mkdir -p "$JOBTMP" "$ROOT/cache/root"
export APPTAINERENV_TMPDIR=/tmp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
echo "[task] $(date -u +%FT%TZ) run=$RUN_NAME gpus=$DEVS critic_gpu=$CRIT value_port=$VP ray_base=$RAY_BASE"
(cd "$ROOT/src" && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CRIT exec "$ROOT/.venv-critic/bin/python" value_server.py --port $VP --tag "$SLURM_JOB_ID-$LANE") \
  > "$RUN_DIR/value_server.$SLURM_JOB_ID.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true; rm -rf "$JOBTMP"; rmdir "$(dirname "$JOBTMP")" 2>/dev/null || true' EXIT
mine() { curl -sf "http://127.0.0.1:$VP/health" | grep -q "\"tag\": \"$SLURM_JOB_ID-$LANE\""; }   # our server, not a stranger on the port
for i in $(seq 1 120); do
  mine && break
  kill -0 $VPID 2>/dev/null || { echo "[task] value server died"; tail -20 "$RUN_DIR/value_server.$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
mine || { echo "[task] value server not ready (or port $VP taken) after 10 min"; exit 1; }
TRANSIENT="Cuda failure 999|ncclUnhandledCudaError|watchdog: no progress"
STALL_MIN=${STALL_MIN:-30}; WATCH_SEC=${WATCH_SEC:-60}
PROGRESS="(rollout|step|eval) [0-9]+: \\{"
[[ $RUN_NAME == smokecopy_* ]] && MAX_TRY=1 || MAX_TRY=${MAX_TRY:-4}
LOG=$RUN_DIR/console.log
for TRY in $(seq 1 $MAX_TRY); do
  START=$(( $( [ -f "$LOG" ] && wc -l < "$LOG" || echo 0) + 1 ))
  echo "[task] $(date -u +%FT%TZ) attempt $TRY/$MAX_TRY ckpt=$(cat "$RUN_DIR/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
  set +e
  # own session and process group (setsid), so the watchdog can kill the whole attempt and nothing of the other lane
  setsid apptainer exec --nv --contain --cleanenv --writable-tmpfs \
    --bind /scratch/gpfs/GROUP/USER --bind "$JOBTMP":/tmp --bind /dev/shm --bind "$ROOT/cache/root":/root/.cache \
    --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="$DEVS" --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    --env RUN_NAME="$RUN_NAME" --env GPUS="$NG" --env RAY_PORT_BASE="$RAY_BASE" --env FC_VALUE_URL="http://127.0.0.1:$VP" \
    --env FLA_DISABLE_BACKEND_DISPATCH=1 --env NCCL_DEBUG=WARN --env NCCL_NVLS_ENABLE=0 \
    "$ROOT/images/miles-20260930.sif" bash "$ROOT/scripts/run_in_container.sh" "${ARGS[@]}" &
  APID=$!
  ( last=-1; since=$(date +%s)
    while kill -0 $APID 2>/dev/null; do
      sleep "$WATCH_SEC"
      n=$(tail -n +$START "$LOG" 2>/dev/null | grep -cE "$PROGRESS")
      if [ "$n" != "$last" ]; then last=$n; since=$(date +%s); continue; fi
      if [ $(( $(date +%s) - since )) -ge $((STALL_MIN * 60)) ]; then
        echo "[task] $(date -u +%FT%TZ) watchdog: no progress line for $STALL_MIN min ($n so far this attempt); killing the attempt" | tee -a "$LOG"
        kill -TERM -- -$APID 2>/dev/null; sleep 30; kill -KILL -- -$APID 2>/dev/null
        pkill -9 -u "$(id -u)" -f "ray_${SLURM_JOB_ID}_$RAY_BASE" 2>/dev/null
        break
      fi
    done ) &
  WPID=$!
  wait $APID; rc=$?
  kill $WPID 2>/dev/null; wait $WPID 2>/dev/null
  # a failed attempt may leave processes of its group behind (TERM ends apptainer first): kill what is left
  [ $rc = 0 ] || kill -KILL -- -$APID 2>/dev/null
  [ $rc = 0 ] && exit 0
  # failure handling runs without -e: a diagnostic that matches nothing must not end the task before the retry
  { echo "[diag] $(date -u +%FT%TZ) job=$SLURM_JOB_ID host=$(hostname) try=$TRY rc=$rc gpus=$DEVS"
    nvidia-smi --query-gpu=index,pci.bus_id,memory.used,ecc.errors.uncorrected.volatile.total,retired_pages.pending,remapped_rows.pending,remapped_rows.failure --format=csv 2>&1
    nvidia-smi -q -d ECC,ROW_REMAPPER 2>&1 | grep -iE "^GPU|uncorr|pending|failure" | head -60
    dmesg -T 2>&1 | grep -iE "xid|nvrm" | tail -20 || true
  } > "$RUN_DIR/gpu_diag.$SLURM_JOB_ID.$TRY.txt" 2>&1
  # count, not grep -q: -q exits at the first match, tail then dies of SIGPIPE and pipefail reads it as "no match"
  # (this hid the retry in job 15202994, lambda 0.5)
  if [ "$(tail -n +$START "$LOG" | grep -cE "$TRANSIENT")" = 0 ]; then echo "[task] attempt $TRY failed rc=$rc (not the transient NCCL error)"; exit $rc; fi
  [ $TRY -lt $MAX_TRY ] || { echo "[task] transient failure again on the last attempt"; exit $rc; }
  # the container's exit trap kills its Ray tree; wait until no process of this lane's Ray instance is left
  for i in $(seq 1 30); do pgrep -u "$(id -u)" -f "ray_${SLURM_JOB_ID}_$RAY_BASE" >/dev/null || break; sleep 10; done
  pkill -9 -u "$(id -u)" -f "ray_${SLURM_JOB_ID}_$RAY_BASE" 2>/dev/null || true
  echo "[task] $(date -u +%FT%TZ) attempt $TRY hit a transient failure (NCCL error or watchdog, rc=$rc); restarting from the latest checkpoint"
  sleep 30
  set -e
done
