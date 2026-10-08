#!/bin/bash
#SBATCH --job-name=fc-phase0
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:45:00
#SBATCH --output=logs/slurm/%x-%j.out
# Phase 0 on one H100: random-actor samples, rollout and actor speed, then every critic check.
set -uo pipefail
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID APPTAINERENV_TMPDIR=/tmp; mkdir -p "$TMPDIR"   # no node /tmp (rule 2026-10-08)
trap 'rm -rf "$TMPDIR"' EXIT
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
cd $ROOT/src
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
# flashinfer kernels missing from flashinfer-jit-cache are compiled at run time and need nvcc.
module load cudatoolkit/13.0
GEN=$ROOT/.venv-gen/bin/python; CRI=$ROOT/.venv-critic/bin/python
# Copies of this job wait in several queues (pli-low, pli-short); the first one to start does the work.
if ! mkdir $ROOT/logs/phase0b.lock 2>/dev/null; then echo "another copy holds logs/phase0b.lock; exiting"; exit 0; fi
echo "$SLURM_JOB_ID $(date)" > $ROOT/logs/phase0b.lock/owner
nvidia-smi --query-gpu=name --format=csv,noheader
step() { echo "=== $(date +%T) $*"; "$@" || echo "FAILED: $*"; }
# Random-init actor: dense Qwen3.8-Flash-Next architecture, ~1B, critic vocabulary, vLLM random (dummy) weights.
step $GEN gen_gsm8k.py --model ../models/actor-q4dense-1b --name random-q4d1b --random-init --load-format dummy --n-problems 256 --n-samples 8 --max-tokens 1024
step $CRI phase0_critic.py repro
step $CRI phase0_critic.py rank
step $CRI phase0_critic.py probes
step $CRI phase0_critic.py speed
step $GEN bench_rollout.py
step $CRI bench_actor.py
step $CRI phase0_critic.py ascent --n-problems 8 --steps 64 --k 512
echo "=== done $(date +%T)"
