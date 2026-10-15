#!/bin/bash
#SBATCH --job-name=fc-direct
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# A Qwen3.5 model answers the screen's 200 problems with and without steps (owner 2026-10-14/15; never Qwen2.5): src/direct_answer.py.
# Usage: sbatch --export=ALL,RUNS="Qwen/Qwen3.5-0.8B:direct Qwen/Qwen3.5-0.8B:steps" scripts/direct_answer.sbatch
#        smoke only (4 problems): add SMOKE_ONLY=1 and --qos=pli-cp. Default RUNS: Qwen3.5-0.8B, direct and steps.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
cd $ROOT/src
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l)"
rc=0
N=""; [ "${SMOKE_ONLY:-0}" = 1 ] && N="--n 4"
for r in ${RUNS:-Qwen/Qwen3.5-0.8B:direct Qwen/Qwen3.5-0.8B:steps}; do
  $ROOT/.venv-critic/bin/python direct_answer.py --model ${r%%:*} --mode ${r##*:} $N || rc=1
done
exit $rc
