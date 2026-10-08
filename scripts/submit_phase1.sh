#!/bin/bash
# Phase 1: PPO + GAE in miles. Actor = random-init dense qwen3_5 1.05B (models/actor-q35d1b-*); values from the frozen
# Limite critic (value server, --external-critic-values); truncated responses bootstrap with V(full) (train_score).
#   LAMBDAS="0.95 0.5" bash scripts/submit_phase1.sh   -> one 8-GPU job, two 4-GPU lanes side by side (pli-short)
#   LAMBDAS="0"        bash scripts/submit_phase1.sh   -> one 4-GPU job (pli-low; owner rule: 1-4 GPU jobs there)
# Each lane runs its own 3-rollout smoke copy first and trains only if it passes (owner's packing rules, 2026-10-07).
# Main runs: --load = --save, so resubmitting the same command resumes after a wall-clock kill.
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
MILES=/scratch/gpfs/GROUP/USER/project/miles-frozen-critic
read -ra LAMS <<< "${LAMBDAS:?}"; SEED=${SEED:-42}
MG=$ROOT/models/actor-q35d1b-text; SG=$ROOT/models/actor-q35d1b-lm; D=$ROOT/data
write_args() {  # write_args RUN LAMBDA smoke|main
  local R=$1 LAM=$2 KIND=$3 NR RB NS GBS EVD EV SAVE WU extra
  if [ "$KIND" = smoke ]; then NR=3; RB=8; NS=4; GBS=16; EVD=$D/gsm8k_test64.parquet; EV=3; SAVE=3; WU=0
    extra=(--skip-eval-before-train)
  else NR=${NUM_ROLLOUT:-300}; RB=64; NS=8; GBS=256; EVD=$D/gsm8k_test.parquet; EV=20; SAVE=20; WU=10
    extra=(--use-wandb --wandb-mode offline --wandb-dir $ROOT/runs/$R --wandb-project frozen-critic
           --wandb-group phase1 --wandb-run-id $R --disable-wandb-random-suffix)
  fi
  local CK=$ROOT/runs/$R/ckpt
  local args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
    --ref-load $MG --load $CK --save $CK --save-interval $SAVE
    --custom-megatron-post-save-hook-path miles_ext.ckpt_rotate.keep_latest
    --prompt-data $D/gsm8k_train.parquet --input-key messages --label-key label --apply-chat-template --rollout-shuffle
    --custom-rm-path miles_ext.frozen_critic_rm.reward_func --reward-key train_score --eval-reward-key score
    --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt $NS --global-batch-size $GBS
    --rollout-max-response-len 1024 --rollout-temperature 1.0 --rollout-top-p 1.0
    --eval-prompt-data gsm8k $EVD --eval-interval $EV --n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-top-k 1
    --save-debug-rollout-data $ROOT/runs/$R/rollouts/{rollout_id}.pt
    --advantage-estimator ppo --external-critic-values --gamma 1.0 --lambd $LAM --normalize-advantages
    --eps-clip 0.2 --calculate-per-token-loss --kl-coef 0 --entropy-coef 0
    --optimizer adam --lr 1e-4 --lr-decay-style constant --lr-warmup-iters $WU --weight-decay 0.01
    --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --use-dynamic-batch-size --max-tokens-per-gpu 8192 --qkv-format thd
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 --sglang-dtype bfloat16
    --attention-dropout 0.0 --hidden-dropout 0.0
    --actor-num-nodes 1 --actor-num-gpus-per-node 4 --colocate --seed $SEED --rollout-seed $SEED "${extra[@]}")
  mkdir -p $ROOT/runs/$R
  printf '%s\n' "${args[@]}" > $ROOT/runs/$R/train_args
}
LANES=""
for LAM in "${LAMS[@]}"; do
  M=ppo_lam${LAM}_s$SEED; S=smokecopy_$M
  write_args $S $LAM smoke; write_args $M $LAM main
  LANES="$LANES $S:$M"
done
N=${#LAMS[@]}; [ $N -le 2 ] || { echo "at most two lanes per job" >&2; exit 1; }
if [ $N = 2 ]; then QOS=(--partition=pli-c --qos=pli-short); else QOS=(--partition=pli --qos=pli-low); fi
TIME=${TIME:-24:00:00}; MAIN_TIMEOUT=${MAIN_TIMEOUT:-22h}
echo "lanes:$LANES gpus=$((4 * N)) ${QOS[*]} time=$TIME miles=$(git -C $MILES rev-parse --short HEAD) repo=$(git -C $ROOT rev-parse --short HEAD)"
[ -n "${DRY:-}" ] && exit 0
mkdir -p $ROOT/logs/slurm $ROOT/logs/packed
jid=$(env LANES="${LANES# }" MAIN_TIMEOUT=$MAIN_TIMEOUT sbatch --parsable "${QOS[@]}" --time=$TIME --gres=gpu:$((4 * N)) \
  --cpus-per-task=$((16 * N)) --mem=$((200 * N))G --job-name="fc-ppo-$(echo ${LAMS[*]} | tr ' ' '-')" $ROOT/scripts/sbatch_lanes.sh)
echo "  submitted: job $jid"
