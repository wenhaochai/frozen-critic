#!/bin/bash
# One frozen-critic task inside a (possibly packed) Slurm job: the critic's value server on the last GPU of this task's
# GPU list (shared with miles: the 1B actor leaves room), then miles in the image on the task's GPUs.
#   run_task.sh RUN_NAME GPU_LIST LANE     (train args from runs/RUN_NAME/train_args, one per line)
# LANE (0, 1) offsets the value-server port and Ray's port block so two tasks can share a node.
# FLA_DISABLE_BACKEND_DISPATCH=1: Gated DeltaNet runs on fla's own Triton kernels (Triton 3.7.1 in the image) instead
# of flash_qla. Job 15202658 (tests/gdn_backend_grads.py) found the two agree (max relative gradient difference 0.4-0.8%,
# deterministic across calls), so flash_qla is NOT the cause of the 1.8e12 step-0 grad norm seen in the smokes; the
# switch stays only so all three lambda arms run the same kernels.
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
apptainer exec --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP/USER --bind "$JOBTMP":/tmp --bind /dev/shm --bind "$ROOT/cache/root":/root/.cache \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="$DEVS" --env SLURM_JOB_ID="$SLURM_JOB_ID" \
  --env RUN_NAME="$RUN_NAME" --env GPUS="$NG" --env RAY_PORT_BASE="$RAY_BASE" --env FC_VALUE_URL="http://127.0.0.1:$VP" \
  --env FLA_DISABLE_BACKEND_DISPATCH=1 \
  "$ROOT/images/miles-20260930.sif" bash "$ROOT/scripts/run_in_container.sh" "${ARGS[@]}"
