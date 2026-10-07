#!/bin/bash
# Runs INSIDE images/miles-20260930.sif (radixark/miles:latest of 2026-09-30, /root/miles = 874b3e275). Adapted from
# miles-q38-build/run_miles_in_container.sh: miles' own launcher pkills sglang/ray on the node, unsafe on shared nodes,
# so this starts a job-private Ray head on job-derived ports, exports the env before `ray start`, then runs train.py.
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
MILES=/scratch/gpfs/GROUP/USER/project/miles-frozen-critic
: "${RUN_NAME:?}" "${SLURM_JOB_ID:?}" "${GPUS:?}" "${FC_VALUE_URL:?}"
RUN_DIR=$ROOT/runs/$RUN_NAME
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
export HOME=/tmp/home TRITON_CACHE_DIR=/tmp/home/.cache/triton; mkdir -p "$HOME"
MEGATRON=/root/Megatron-LM
# pyfix: opentelemetry-api 1.44.0 (the image's 1.45.0 breaks Ray's dashboard agent), see miles-q38-build/README.md
export PYTHONPATH=$ROOT/pyfix:$MEGATRON:$MILES:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_USAGE_STATS_ENABLED=0 WANDB_MODE=offline
echo "[fc-run] $(date -u +%FT%TZ) host=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME miles=$(git -C $MILES rev-parse --short HEAD) dirty=$(git -C $MILES status --porcelain | wc -l) repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l) value=$FC_VALUE_URL"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
P=${RAY_PORT_BASE:-$((20000 + (SLURM_JOB_ID % 1270) * 10))}
RAY_TMP=/tmp/ray_${SLURM_JOB_ID}_$P
ray start --head --node-ip-address 127.0.0.1 --num-gpus "$GPUS" --port=$P --dashboard-port=$((P + 1)) \
  --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
  --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
  --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMP"
export RAY_ADDRESS=127.0.0.1:$P
trap 'pkill -9 -u "$(id -u)" -f -- "$RAY_TMP" 2>/dev/null || true' EXIT
cd "$MILES"
MODEL_ARGS=$(python3 -c "from miles.utils.external_utils.model_args_utils import shell_safe_model_args; print(shell_safe_model_args('qwen3.5-1b-fc'))")
echo "[fc-run] model args: $MODEL_ARGS"
echo "[fc-run] train args: $*"
eval "python3 train.py $MODEL_ARGS \"\$@\""
