# Plugging the frozen Limite critic into miles (Phase 1 plan, not built yet)

Read on 2026-10-05 in `project/miles-easyppo`, branch `easyppo` at `8cbcddec7` (upstream `874b3e275` plus EasyPPO
commits). The older upstream clone `project/miles` (main `969c2a70`) runs its critic on separate GPUs over NCCL and
adds the PPO reward at the last token on CP rank 0 only; use the fork. Line numbers below are from the fork.

## What miles does today

- `--advantage-estimator ppo` sets `args.use_critic = True` (`miles/utils/arguments.py:3601`); there is no
  `--use-critic` flag. With a critic, `--kl-coef` must be 0 and `--offload-train` is forced on.
- `--gamma` and `--lambd` (both default 1.0) feed GAE: `compute_advantages_and_returns`
  (`backends/training_utils/loss.py:28`) reads `rollout_data["values"]` (`:60`), then `compute_advantages`
  (`loss_hub/advantages.py`) calls `get_advantages_and_returns_batch` (`loss_hub/math_utils.py:624`), which adds the
  scalar reward at the last trainable token and runs `chunked_gae` (`:818`); lambda = 0 has its own branch.
- Value alignment: miles slices `logits[start-1:end-1]` (`loss_hub/logit_processors.py:81,85`), so the value for
  response token t is V(prompt, response[:t]). Limite's `post[P-1:P+L-1]` is the same quantity: no shift needed.
- The Megatron critic computes values in `_train_critic` (`backends/megatron_utils/actor.py:618`) and sends them to
  the actor through the object store. Rollout samples cannot carry values today: `values` is not in the key list
  shipped to the trainer (`ray/rollout/train_data_conversion.py:420-446`).

## Proposed change (about 25 lines in miles plus about 100 of our own)

1. `arguments.py`: add `--external-critic-values`; set `use_critic = (estimator == "ppo") and not external`. The run
   then builds no critic trainer, skips the KL assert and the forced offload, and trains the actor only.
2. `train_data_conversion.py`: add `"values": "float32"`, fill `train_data["values"]` from each sample's
   `reward["values"]`, assert `len == response_length`, add `"values"` to the shipped keys (`:420`).
3. `training_utils/data.py:98`: move `"values"` to the GPU with the other per-token lists.
4. `actor.py`, before GAE (`:757`): assert values are present when the flag is on.
5. Ours: a reward function for `--custom-rm-path` (awaited per sample right after generation,
   `inference_rollout/inference_rollout_common.py:131`, so scoring overlaps generation) that grades GSM8K and POSTs
   `prompt_ids` and `response_ids` to a value server; returns `{"score": r, "values": [...]}` with `--reward-key score`.
   The server wraps `src/critic.py` (packed FlashAttention-2 scoring, batching requests from all rollout workers) on its
   own GPU outside the Ray pool.

## Settings the random actor needs

- Actor checkpoint: `AutoModelForCausalLM.from_config` + `save_pretrained`, passed as `--hf-checkpoint` with
  `--megatron-to-hf-mode bridge`; every lambda arm starts from the same weights.
- Vocabulary: Qwen2.5 has 151,936 embedding rows but 151,665 tokens; the critic accepts ids < 151,667. Mask rows
  >= 151,665 in sampling (sglang) or truncate the checkpoint's vocabulary as `src/make_random_actor.py` does.
- Prompt format for the critic: the critic's own chat template (`<|endoftext|>` + Qwen2.5 template); completed
  rollouts end with `<|endoftext|>` (151643). The value server rebuilds the critic prompt from the question, so the
  actor may use its own template.
- No `check_reward_nonzero_std` filter: with all-zero rewards it drops every group and rollouts stall.
- No KL to a reference (the reference would be the random init); `--entropy-coef 0`, log entropy.
- `--normalize-advantages`: the same setting for all arms, since whitening rescales the lambda = 0 advantages.
