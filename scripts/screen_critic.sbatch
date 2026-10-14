#!/bin/bash
#SBATCH --job-name=fc-screen
#SBATCH --account=pli
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
# Offline screen of one Open-Reasoner-Zero critic (owner 2026-10-13: is Limite just too weak?): src/screen_critic.py.
# Usage: sbatch --export=ALL,CRITIC=orz-critic-7B,CHUNK=250 [--qos=...] scripts/screen_critic.sbatch
# Two copies per critic wait in two queues (pli-short and pli-low); the first to start takes logs/screen_$CRITIC.lock.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
L=$ROOT/logs/screen_${CRITIC:?}.lock
if ! mkdir $L 2>/dev/null; then echo "another copy holds $L; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(date)" > $L/owner
cd $ROOT/src
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "repo=$(git -C $ROOT rev-parse --short HEAD) dirty=$(git -C $ROOT status --porcelain --untracked-files=no | wc -l) critic=$CRITIC"
$ROOT/.venv-critic/bin/python screen_critic.py --critic $CRITIC --chunk ${CHUNK:-250}
rc=$?
if [ $rc -ne 0 ]; then rm -rf $L; fi   # a failed copy frees the lock for the others
exit $rc
