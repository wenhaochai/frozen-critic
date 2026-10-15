#!/bin/bash
#SBATCH --job-name=fc-answerpick
#SBATCH --account=pli
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# Offline diagnostic (owner, 2026-10-11): does the frozen critic rank the gold boxed answer above 20 distractors?
# A100 copy (owner chose A100, 2026-10-12): flash-attn in .venv-critic has H100 (sm90) kernels only, so the critic runs
# on the eager backend (packs of at most 4,096 tokens). Shares logs/answerpick.lock with the two H100 copies.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
if ! mkdir $ROOT/logs/answerpick.lock 2>/dev/null; then echo "another copy holds logs/answerpick.lock; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(date)" > $ROOT/logs/answerpick.lock/owner
cd $ROOT/src
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l)"
$ROOT/.venv-critic/bin/python diag_answer_pick.py --backend eager
rc=$?
if [ $rc -ne 0 ]; then rm -rf $ROOT/logs/answerpick.lock; fi   # a failed copy frees the lock for the others
exit $rc
