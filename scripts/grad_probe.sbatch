#!/bin/bash
# Step-0 grad-norm diagnostic: one 1-GPU task (actor, SGLang and critic share the GPU), 2 rollouts of 8 x 8, with
# miles_ext.grad_probe recording per-parameter grad norms of the first steps. Train args: runs/gradprobe_lam0.95/train_args.
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=fc-gradprobe
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/frozen-critic/logs/slurm/%x-%j.out
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
rm -rf -- "$ROOT/runs/gradprobe_lam0.95/ckpt" "$ROOT/runs/gradprobe_lam0.95/grad_probe"
MAX_TRY=1 "$ROOT/scripts/run_task.sh" gradprobe_lam0.95 "${CUDA_VISIBLE_DEVICES:?}" 0
