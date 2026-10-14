# Frozen critic, random actor: final report (closed 2026-10-13)

**Result (negative).** A frozen 1B value model tells a randomly initialised actor only how long to write, never what
to write: with raw values the actor never stops, with values centred on the prompt it stops at once, and GSM8K test
accuracy is 0 in all six PPO runs (three GAE λ × two offsets, one seed each). Offline, the critic knows the size of the
answer but not the answer: among all numbers 0 to 999 its top-valued `\boxed{k}` is right on 4.5% [2.0, 8.0] of
problems, against 1.5% for always answering 4 on the same problems.

Status: failed (the hypothesis is not supported). The owner closed the project on 2026-10-13. No run is queued.

Page: `blogs/frozen-critic.html` on branch `pretrain-practice` of the website clone (index card marked "Failed");
private preview https://claude.ai/artifact/556X5h46FnCueDn4xRdQmf. Day-by-day notes: [STATUS.md](STATUS.md).
Original brief: [BRIEF.md](BRIEF.md). Per-experiment records: [experiments/](experiments/).

## Question

One view of pretraining holds that next-token prediction mostly teaches a model to judge partial solutions, and that
the policy still has to be learned. We tested the second half in its starkest form: the value function is given and
frozen, and the policy starts from random weights.

- Critic: `paradigma-inc/limite-1b-value-model` (1.0B, released 2026-09-21), frozen. One value per response token,
  regressed on final correctness.
- Actor: dense Qwen3.5-architecture model, 0.46B parameters, random init. Own 100-token vocabulary, each token one
  critic token (95 printable ASCII characters, newline, `boxed`, two chat markers, the stop token). It reads the
  question character by character and may write only 27 ids (digits, `+ - * / =`, brackets, `$`, comma, period, the
  pieces of `\boxed{}`, space, newline, stop).
- Algorithm: PPO + GAE in miles (fork `wenhaochai/miles-private`, branch `frozen-critic`), γ = 1, λ ∈ {0.95, 0.5, 0}.
- Reward: 1 if the last `\boxed{}` holds the answer, +0.1 if a finished response's last box holds one number; a
  response cut at 1,024 tokens takes the critic's value of the whole response.
- Task and metric: GSM8K, test accuracy under a strict last-`\boxed{}` grader, greedy decoding, all 1,319 problems
  every 20 PPO steps.

## Results

| Experiment | What we measured | Result |
|---|---|---|
| Phase 0: critic check (jobs 15011145, 15021750, 15132135) | Card example, ranking of other models' GSM8K samples | Card values reproduced exactly. Within-problem AUC 0.66 to 0.73 on Qwen2.5 generators; beats a length baseline on the two general models, ties it on the math model and R1-Distill. |
| Full-vocabulary actor, λ 0.95 / 0.5 / 0 (dropped 2026-10-09) | PPO with an actor that may write all 151,667 tokens | Never samples a boxed number; games the critic with token soup; accuracy 0. The owner dropped these runs from the page (a reward of about 1e-30 per position cannot be learned). |
| Character actor, raw values, λ 0.95 / 0.5 / 0 (jobs 15264957, 15375157) | Accuracy, finished boxed answers, critic value | The actor stops ending its responses within 15 PPO steps; from step 36 at the latest 99% run to 1,024 tokens of digits and brackets. Accuracy 0 at all 16 evals of every run. |
| Character actor, centred values V − V(prompt), λ 0.95 / 0.5 / 0 (jobs 15281139, 15281143) | The same | Every training response is the stop token alone from step 5 to 9 on. Accuracy 0. |
| Answer pick, 21 candidates (job 15449542, A100) | Is the gold `\boxed{k}` the critic's top value among 21 candidates? | 11.3% [9.6, 13.0] vs chance 4.8%; part is a preference for round numbers. |
| Answer, open set 0 to 999 (job 15453738) | Is the gold the critic's top value among all 1,000 numbers? | 4.5% [2.0, 8.0]; under another problem's question 0.5%. Spearman(top number, gold) 0.56 vs −0.02 under another question. All 9 hits have gold ≤ 20. Always answering 4 (most common train answer): 1.5% on the same 200, 2.7% on all of test. |

## Why it failed

1. **The stop token decides everything.** With γ = 1 the stop token's advantage is the reward minus the critic's value,
   about 0 − 0.54 for a random response, while every other token changes the value by much less. The actor learns
   never to stop.
2. **Centring flips the sign.** Subtracting V(prompt) = 0.71 leaves every other TD residual unchanged and adds 0.71 to
   the stop token's, so stopping now earns about +0.17. The critic rates any random text below the prompt (the first
   character lowers the value by 0.077 on average, 97% of samples), so the best policy is to write nothing.
3. **The critic does not hold the answer.** Even a template actor that only writes `\boxed{k}` and always finds the
   critic's best k would score about 4.5%, 3.0 points [0.5, 6.5] above a constant answer. We did not train it.

A critic fitted to a competent policy's states has no reliable values on a random policy's states; what it does
rank reliably there is length.

## Limits of the evidence

- One seed per setting; one actor size (0.46B); one critic; GSM8K only; responses capped at 1,024 tokens.
- The grader accepts any box with one number in it (audit #24/#54). In training it graded two responses correct: a
  truncated one-digit box and a box holding the right digit plus junk;
  no evaluation was affected (accuracy is 0 throughout).
- Step-0 gradient: the first two optimizer steps have grad norm 7e11 to 1.2e12, all in the input embedding (the
  padding row; see STATUS). All runs share this init, so the comparison between arms holds; a new model should
  re-initialise that row.
- The open-set diagnostic uses 200 problems; its intervals are wide.

## Compute

From sacct over every `fc-*` job since 2026-10-04 (split by the packing ledger `logs/packed/`): main runs
53.8 H100-hours in 4 jobs; other compute 39.7 H100-hours and 0.4 A100-hours in 16 jobs; total 93.9 GPU-hours.

## Reproduce

| Run | Code (this repo) | miles | W&B |
|---|---|---|---|
| `ppo_char_lam0.95_s42`, `ppo_char_lam0.5_s42` | `7f11065` | `adf376d4d` | [0.95](https://wandb.ai/reself/frozen-critic/runs/ppo_char_lam0.95_s42), [0.5](https://wandb.ai/reself/frozen-critic/runs/ppo_char_lam0.5_s42) |
| `ppo_char_lam0_s42` | `eda16e6` | `c497fc612` (adds masked entropy logging and request-path asserts) | [0](https://wandb.ai/reself/frozen-critic/runs/ppo_char_lam0_s42) |
| `ppo_charc_lam{0.95,0.5,0}_s42` | `a1059c9` | `adf376d4d` | [0.95](https://wandb.ai/reself/frozen-critic/runs/ppo_charc_lam0.95_s42), [0.5](https://wandb.ai/reself/frozen-critic/runs/ppo_charc_lam0.5_s42), [0](https://wandb.ai/reself/frozen-critic/runs/ppo_charc_lam0_s42) |

- Actor: `src/make_char_actor.py`. Data: `src/make_miles_data.py`. Launch: `scripts/submit_phase1.sh` (VOCAB=char,
  CENTER=1 for the centred arm) and `scripts/run_task.sh`; critic served by `src/value_server.py`.
- Diagnostics: `src/phase0_critic.py`, `src/diag_answer_pick.py`, `src/diag_answer_open.py`,
  `src/diag_answer_open_extra.py`. Outputs in `results/` (`answer_*.json`, `phase1_char*_curves.json`, `rank.json`).
- Page: `site/phase1_curves.py`, then `site/build.py` (writes the website data, record and `site/_artifact`).
- Environments and critic weights: see "How to run" in [STATUS.md](STATUS.md). Run directories (`runs/`), weights
  (`models/`) and logs (`logs/`) are not in git; they stay on Della scratch at
  `/scratch/gpfs/GROUP/USER/project/frozen-critic`.

## Not done

- The template-answer PPO arm (ceiling about 3 points above a constant answer; see above).
- More seeds, a larger actor, a stricter grader, and the remaining low-priority audit items listed in STATUS.
