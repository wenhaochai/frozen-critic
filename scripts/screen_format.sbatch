#!/bin/bash
#SBATCH --job-name=fc-format
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# Which part of the ORZ answer format holds the critic's answer knowledge (owner 2026-10-14): src/screen_format.py.
# Smoke (3 problems, k < 50) then the full run (200 problems, k < 1000), in one job.
# Usage: sbatch --export=ALL,CRITIC=orz-critic-32B,CHUNK=100 scripts/screen_format.sbatch
#        smoke only: sbatch --qos=pli-cp --export=ALL,CRITIC=orz-critic-0.5B,SMOKE_ONLY=1 scripts/screen_format.sbatch
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
cd $ROOT/src
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l) critic=${CRITIC:?}"
PY=$ROOT/.venv-critic/bin/python
$PY screen_format.py --critic $CRITIC --n 3 --kmax 50 --chunk 50 --out format_${CRITIC}_smoke || exit 1
[ "${SMOKE_ONLY:-0}" = 1 ] && exit 0   # pli-cp smoke copy
$PY screen_format.py --critic $CRITIC --chunk ${CHUNK:-250}
