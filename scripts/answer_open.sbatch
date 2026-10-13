#!/bin/bash
#SBATCH --job-name=fc-answeropen
#SBATCH --account=pli
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# Open-set follow-up (2026-10-12): the critic scores boxed 0..999 for 200 GSM8K problems, under the true and a swapped prompt.
# Two copies wait in two queues (pli-short and pli-low); the first to start takes logs/answeropen.lock, the other exits.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
if ! mkdir $ROOT/logs/answeropen.lock 2>/dev/null; then echo "another copy holds logs/answeropen.lock; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(date)" > $ROOT/logs/answeropen.lock/owner
cd $ROOT/src
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l)"
$ROOT/.venv-critic/bin/python diag_answer_open.py
rc=$?
if [ $rc -ne 0 ]; then rm -rf $ROOT/logs/answeropen.lock; fi   # a failed copy frees the lock for the others
exit $rc
