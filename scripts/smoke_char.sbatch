#!/bin/bash
# Fast integration smoke of the character-level arm on pli-cp (smoke tests only): the 3-rollout smoke copy of
# ppo_char_lam0.95_s42 (64 x 8, GBS 256, eval on 64 test problems) under its own run dir runs/smoketest_char_lam0.95.
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=fc-smoke-char
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
rm -rf -- "$ROOT/runs/smoketest_char_lam0.95/ckpt" "$ROOT/runs/smoketest_char_lam0.95/rollouts"
MAX_TRY=1 "$ROOT/scripts/run_task.sh" smoketest_char_lam0.95 "${CUDA_VISIBLE_DEVICES:?}" 0
