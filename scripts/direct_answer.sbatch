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
# Qwen2.5 instruct models answer the screen's 200 problems with no steps (owner 2026-10-14): src/direct_answer.py.
# Usage: sbatch [--qos=...] scripts/direct_answer.sbatch
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
cd $ROOT/src
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l)"
rc=0
for m in 0.5B 1.5B 3B 7B; do
  $ROOT/.venv-critic/bin/python direct_answer.py --model Qwen/Qwen2.5-$m-Instruct || rc=1
done
exit $rc
