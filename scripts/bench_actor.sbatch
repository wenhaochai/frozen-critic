#!/bin/bash
#SBATCH --job-name=fc-benchactor
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/%x-%j.out
# Actor-update throughput of the dense actor sizes (qwen3_5 proxies; flash-linear-attention with tilelang, since
# fla refuses its Triton backward on Hopper with Triton 3.6). Copies wait in pli-low and pli-short; one runs.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID APPTAINERENV_TMPDIR=/tmp; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
cd $ROOT/src
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if ! mkdir $ROOT/logs/benchactor.lock 2>/dev/null; then echo "another copy holds logs/benchactor.lock; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(date)" > $ROOT/logs/benchactor.lock/owner
nvidia-smi --query-gpu=name --format=csv,noheader
$ROOT/.venv-critic/bin/python bench_actor.py
echo "=== done $(date +%T) rc=$?"
