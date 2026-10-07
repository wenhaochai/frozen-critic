#!/bin/bash
# Phase 1: PPO + GAE in miles, actor = random-init dense qwen3_5 1.05B (models/actor-q35d1b-*), values from the frozen
# Limite critic (value server, --external-critic-values). LAMBDA=0.95|0.5|0 [SMOKE=1] bash scripts/submit_phase1.sh
#   4 GPUs per run on pli-low (owner rule: 1-4 GPU jobs there; 8-GPU jobs go to pli-short). The 1B actor trains with
#   Megatron DP4 colocated with 4 SGLang engines; the critic server shares the last GPU.
#   Full run: --load = --save, so the same command resumes after a wall-clock kill.
set -euo pipefail
ROOT=/scratch/gpfs/GROUP/USER/project/frozen-critic
LAMBDA=${LAMBDA:?}; SEED=${SEED:-42}; GPUS=4
MG=$ROOT/models/actor-q35d1b-text; SG=$ROOT/models/actor-q35d1b-lm; D=$ROOT/data
if [ -n "${SMOKE:-}" ]; then
  R=smoke_lam${LAMBDA}; NR=${NUM_ROLLOUT:-3}; RB=8; NS=4; GBS=16; EVD=$D/gsm8k_test64.parquet; EV=$NR; SAVE=$NR
  TIME=${TIME:-01:00:00}; WU=0; extra=(--skip-eval-before-train)
else
  R=ppo_lam${LAMBDA}_s$SEED; NR=${NUM_ROLLOUT:-300}; RB=64; NS=8; GBS=256; EVD=$D/gsm8k_test.parquet; EV=20; SAVE=20
  TIME=${TIME:-24:00:00}; WU=10
  extra=(--use-wandb --wandb-mode offline --wandb-dir $ROOT/runs/$R --wandb-project frozen-critic
         --wandb-group phase1 --wandb-run-id $R --disable-wandb-random-suffix)
fi
CK=$ROOT/runs/$R/ckpt
args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
  --ref-load $MG --load $CK --save $CK --save-interval $SAVE
  --prompt-data $D/gsm8k_train.parquet --input-key messages --label-key label --apply-chat-template --rollout-shuffle
  --custom-rm-path miles_ext.frozen_critic_rm.reward_func --reward-key train_score --eval-reward-key score
  --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt $NS --global-batch-size $GBS
  --rollout-max-response-len 1024 --rollout-temperature 1.0 --rollout-top-p 1.0
  --eval-prompt-data gsm8k $EVD --eval-interval $EV --n-samples-per-eval-prompt 1 --eval-max-response-len 1024 --eval-top-k 1
  --save-debug-rollout-data $ROOT/runs/$R/rollouts/{rollout_id}.pt
  # PPO with the frozen external critic: GAE over its values, no critic model, no KL (the reference would be the random init)
  --advantage-estimator ppo --external-critic-values --gamma 1.0 --lambd $LAMBDA --normalize-advantages
  --eps-clip 0.2 --calculate-per-token-loss --kl-coef 0 --entropy-coef 0
  --optimizer adam --lr 1e-4 --lr-decay-style constant --lr-warmup-iters $WU --weight-decay 0.01
  --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-dynamic-batch-size --max-tokens-per-gpu 16384 --qkv-format thd
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.5 --sglang-dtype bfloat16
  --attention-dropout 0.0 --hidden-dropout 0.0
  --actor-num-nodes 1 --actor-num-gpus-per-node $GPUS --colocate --seed $SEED --rollout-seed $SEED
  "${extra[@]}")
echo "$R: miles=$(git -C /scratch/gpfs/GROUP/USER/project/miles-frozen-critic rev-parse --short HEAD) repo=$(git -C $ROOT rev-parse --short HEAD) time=$TIME"
[ -n "${DRY:-}" ] && { printf '  %s\n' "${args[*]}"; exit 0; }
mkdir -p $ROOT/runs/$R $ROOT/logs/slurm
jid=$(env RUN_NAME=$R GPUS=$GPUS sbatch --parsable --partition=pli --qos=pli-low --time=$TIME --gres=gpu:$GPUS \
  --cpus-per-task=32 --mem=256G --job-name="fc-$R" $ROOT/scripts/sbatch_phase1.sh "${args[@]}")
echo "  submitted: job $jid"
