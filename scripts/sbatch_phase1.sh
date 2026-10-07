#!/bin/bash
# Host-side SLURM wrapper for a Phase 1 run: the frozen critic's value server on this job's last GPU (shared with miles:
# the 1B actor leaves room), then miles in the image on every GPU of the job. RUN_NAME and GPUS from the env.
#SBATCH --account=pli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
: "${RUN_NAME:?}" "${GPUS:?}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
RUN_DIR=$ROOT/runs/$RUN_NAME; mkdir -p "$RUN_DIR"
JOBTMP=${TMPDIR:-/tmp}/fc_$SLURM_JOB_ID; mkdir -p "$JOBTMP"
IFS=, read -ra DEV <<< "${CUDA_VISIBLE_DEVICES:?}"
CRIT=${DEV[$((${#DEV[@]} - 1))]}
VP=$((40000 + SLURM_JOB_ID % 2000))   # above Ray's job-derived block (20000-32700)
echo "[sbatch] $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME gpus=$CUDA_VISIBLE_DEVICES critic_gpu=$CRIT port=$VP"
(cd "$ROOT/src" && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$CRIT exec "$ROOT/.venv-critic/bin/python" value_server.py --port $VP) \
  > "$RUN_DIR/value_server.$SLURM_JOB_ID.log" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true; rm -rf "$JOBTMP"' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$VP/health" >/dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "[sbatch] value server died"; tail -20 "$RUN_DIR/value_server.$SLURM_JOB_ID.log"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$VP/health" || { echo "[sbatch] value server not ready after 10 min"; exit 1; }
echo
apptainer exec --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP/USER --bind "$JOBTMP":/tmp --bind /dev/shm --bind "$ROOT/cache/root":/root/.cache \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" --env RUN_NAME="$RUN_NAME" --env GPUS="$GPUS" \
  --env FC_VALUE_URL="http://127.0.0.1:$VP" \
  "$ROOT/images/miles-20260930.sif" bash "$ROOT/scripts/run_in_container.sh" "$@"
