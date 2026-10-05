#!/bin/bash
#SBATCH --job-name=fc-gen
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x-%j.out
# Phase 0: GSM8K test samples from normal generators and random-init actors (vLLM, one H100).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/frozen-critic
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=/scratch/gpfs/GROUP/USER/project/frozen-critic/.venv-gen/bin/python
cd src
nvidia-smi --query-gpu=name --format=csv,noheader
run() { echo "=== $(date +%T) $*"; $PY gen_gsm8k.py "$@" || echo "FAILED: $*"; }
STAGE=${STAGE:-all}   # all | random | normal
# Random-init actors first: an RL-sized batch (256 prompts x 8 samples x 1024 tokens) also times the rollout.
if [ "$STAGE" != normal ]; then
run --model ../models/random-qwen2.5-0.5b --name random0.5b --random-init --n-problems 256 --n-samples 8 --max-tokens 1024
run --model ../models/random-qwen2.5-1.5b --name random1.5b --random-init --n-problems 256 --n-samples 8 --max-tokens 1024
fi
if [ "$STAGE" != random ]; then
run --model Qwen/Qwen2.5-0.5B-Instruct --name qwen0.5b-instruct --max-tokens 2048
run --model Qwen/Qwen2.5-1.5B-Instruct --name qwen1.5b-instruct --max-tokens 2048
run --model Qwen/Qwen2.5-Math-1.5B-Instruct --name qwen-math1.5b-instruct --max-tokens 2048
run --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --name r1distill1.5b --max-tokens 3584
fi
echo "=== done $(date +%T)"
