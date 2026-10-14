# Frozen critic, random actor: status

**Reopened 2026-10-13 (owner): is Limite just too weak? Screening the Open-Reasoner-Zero critics (0.5B, 1.5B, 7B, 32B) offline before any new PPO run. Report of the Limite result: [FINAL.md](FINAL.md).**

**ORZ critic screen (jobs 15523267, 15523269, 15535201, 15540815; 2026-10-14; `src/screen_critic.py`, results/screen_orz-critic-*.json):** no critic passes the gate (open-set top-1 >= 20% and V before stop on random text <= 0.1). Larger ORZ critics rank finished solutions much better than Limite (within-problem AUC 0.5B 0.70-0.75, 1.5B 0.77-0.91, 7B 0.83-0.89, 32B 0.74-0.95; Limite 0.66-0.73), but they still do not hold the answer when the actor writes characters: top-1 over 0..999 with `\boxed{k}` as single characters right after `<think>` is 1.0%, 5.0%, 3.5%, 5.0% [2.0, 8.5] (Limite 4.5%, always-4 1.5%). In ORZ's own format (` </think> <answer> \boxed{k} </answer>`, BPE tokens) the 32B critic picks the gold on 19.0% [14.0, 24.5] of 200 problems (median rank 9.5), under another problem's question 1.5%; 7B 7.0%, 1.5B 3.5%, 0.5B 0.5%. All of them rate random text high: V before stop 0.065, 0.19, 0.43, 0.75 against V(prompt) 0.11, 0.41, 0.82, 0.88, so a free-form random actor would again learn never to stop (stop advantage about -V). The 32B checkpoint saves its value head as [8, 640] tensor-parallel shards; `reshape(1, -1)` (ORZ issue #63, commit 332432f) gives the order the AUC confirms. `.venv-critic` gained `accelerate` 1.15.0 for `device_map`.

**Phase 0 (2026-10-06):** the frozen Limite critic reproduces its card exactly and ranks other models' GSM8K solutions moderately (within-problem AUC 0.66-0.73 on Qwen2.5 generators, above a length baseline on the general ones), but it gives a random actor an almost flat signal (values 0.53-0.65, per-token change sd 0.033) that greedy token search cannot climb above a correct solution: hard to exploit, little to learn from.

**Result in one sentence (2026-10-11, lambda 0 control added 2026-10-13):** a frozen critic fitted to a competent policy tells a randomly initialised character-level actor only how long to write, never what to write: with raw values the stop token costs about V = 0.54, so the actor never stops; with values centred on V(prompt) = 0.71 every written character loses value and stopping earns +0.17, so the actor writes nothing; GSM8K accuracy is 0 in all six runs (one seed each).

**Phase 1 (2026-10-09), redesigned:** the full-vocabulary runs are dropped (owner: they do not make sense; a random actor over 151,667 tokens cannot sample a boxed answer, about 1e-30 per position). Character-level arm: 0.46B dense qwen3_5 actor, own 100-token vocabulary (1:1 to critic ids), ASCII prompts, output masked to 27 arithmetic ids, reward = strict grade + 0.1 format for a finished boxed number. Job 15264957 (lambda 0.95 and 0.5, 300 rollouts each, 3.0 h wall each): from rollout 13 (0.95) and 12 (0.5) at least 99% of responses are cut at 1,024 tokens, mostly digits and brackets (lambda 0.95: '4' 17%, '(' 11%, 'boxed' 8%; 14 distinct greedy outputs over 1,319 test problems at the last eval); test accuracy 0.0 at all 16 evals; last boxed-number finish at rollout 5 (0.95) and 10 (0.5); one truncated clean one-digit box ('9') and one box holding 4 plus junk graded correct (the grader takes any box with one number in it, audit #24/#54). Likely cause: at the stop token the advantage is r - V with V about 0.54 at rollout 0, while later tokens change V by much less.

**Centred test (owner 2026-10-09; jobs 15281139, 15281143, 2026-10-11):** every value and the truncation bootstrap minus V(prompt) = values[0]. This leaves every TD residual except the stop token's unchanged and adds V(prompt) = 0.71 to it, so the stop token earns about 0 - 0.54 + 0.71 = +0.17, not about 0 as planned. The critic rates any random text below the prompt (first character -0.077 on average, 97% of samples negative, same in the uncentred runs), so an immediate stop (advantage 0) beats writing. All three lambdas collapse to the bare stop token: every training response has length 1 from rollout 8 (0.95), 4 (0.5) and 5 (0) to 299; accuracy 0.0 at all 16 evals; no boxed finish. Owner 2026-10-11: take this as the conclusion, no further arms; run the missing uncentred lambda 0 control.

**Uncentred lambda 0 control (owner 2026-10-11; job 15375157, 2026-10-13):** same as lambda 0.95 and 0.5. At least 99% of training responses are cut at 1,024 tokens from rollout 35 (99.6% already at rollout 15); last boxed-number finish at rollout 4; accuracy 0.0 at all 16 evals; critic value before the last token 0.54 to 0.42. Figure 2 now shows all three lambdas.

**Does the critic know the answer? (answer_pick, owner plan 2026-10-11; A100 job 15449542, 2026-10-12):** on 1,317 test problems the critic scores \boxed{k} + stop for the gold k and 20 distractors. Gold is top-1 in 11.3% [9.6, 13.0] vs chance 4.8%; within-problem AUC 0.73; vs gold +-1/+-2 0.69; vs numbers copied from the question 0.90; the gold beats the mean distractor by only 0.008 in value. Part is a round-number preference: golds not ending in 0 or 5 give top-1 9.5% and near AUC 0.59. The four formats (char or BPE, with or without the think prefix) agree. logs/answer_pick.yaml has the numbers. Open-set follow-up (src/diag_answer_open.py, job 15453738, 2026-10-13): for 200 test problems the critic scores \boxed{k} + stop for every k in 0..999. The critic knows the size of the answer, not the answer. Under the problem's own prompt the argmax is the gold in 4.5% [2.0, 8.0] (chance 0.1%), the gold is in the top 10 in 26%, its median rank is 36.5, and Spearman(argmax, gold) is 0.56; under another problem's prompt these drop to 0.5%, 6.5%, 137.5 and -0.02. All 9 exact hits have gold <= 20. That ceiling is near a constant guess: always answering 4, the most common training answer (210/7,473), is correct on 2.7% of GSM8K test (35/1,319) and on 1.5% of these 200; on the same 200 the critic's argmax beats it by 3.0 points [0.5, 6.5] (paired bootstrap). A template-answer PPO arm that found the critic's argmax every time would gain at most about 3 points over a constant answer; not run (owner closed the project 2026-10-13). logs/answer_open.yaml has the numbers.


**Step-0 gradient (audit 2026-10-11, #13/#21):** the first rollout's two optimizer steps have grad norm 7e11-1.2e12, all in the input embedding. Likely cause: HF zeroes the padding_idx row and pad = <|endoftext|> (id 0), which starts every prompt through the chat template; Megatron's embedding has no padding_idx, so that zero row gets a huge gradient through the norms. After clipping, the other parameters barely move in those two steps. Not fixed in the finished runs (models unchanged so all runs share one init); src/make_char_actor.py should re-initialise that row before any new model is built.

Brief: [BRIEF.md](BRIEF.md). Progress page (private artifact): https://claude.ai/artifact/556X5h46FnCueDn4xRdQmf,
source `blogs/frozen-critic.html` on branch `pretrain-practice` of the website clone, generated by `site/build.py`.

## Owner decisions after the brief

- 2026-10-05 (superseded 2026-10-07 by a dense qwen3_5 1.05B actor, and 2026-10-09 by the 0.46B character-level actor): actor architecture = Qwen3.8-Flash-Next (`qwen4_exp`, the newest Qwen architecture miles supports;
  RL support merged 2026-09-24), made dense (no experts) at about 1B parameters; vocabulary = the critic's 151,667
  ids, so actor tokens are critic tokens. Config: `src/make_actor_config.py` -> `models/actor-q4dense-1b`
  (hidden 1280, 24 layers = 18 Gated DeltaNet + 6 gated attention with the QSA indexer, SwiGLU 3840,
  10 heads x 256, 4-stream hyper-connections r=160, n-gram PLE at layer 2 with 65,536 rows per head, tied
  embeddings, no MTP). Sizes 0.5B/2B/4B of the same family only for the speed table.
- 2026-10-05: no A100 for Phase 0; wait for H100.

## Phase 0: verify the critic (done 2026-10-06)

| Check | Script | State |
|---|---|---|
| Card example, eager CPU | `src/critic.py` (vis node) | done: 0.695918 / 0.680881 vs card 0.696658 / 0.680541 |
| Card example, eager and FA2 on H100; packed vs single | `phase0_critic.py repro` | done (job 15021750): exact card values for both backends; packed vs single max 2.4e-3 over 64 sequences |
| Ranking of graded GSM8K samples (AUC, curves, drop position) | `phase0_critic.py rank` | done, results/rank.json |
| Random-actor samples and constructed probes | `phase0_critic.py probes` | done, results/probes.json |
| Greedy value ascent | `phase0_critic.py ascent` | done, results/ascent.json |
| Critic scoring throughput | `phase0_critic.py speed` | done, results/speed.json |
| Rollout and actor throughput by actor size (dense Qwen3.8-Flash-Next; actor update on a same-shape qwen3_5 proxy) | done: rollout job 15021750, actor update job 15132135 (fla needs tilelang on Hopper with Triton 3.6); 1B step on one H100 = 113 s rollout + 17 s critic + 130 s update |

Generations (GSM8K test, 500 problems x 4 samples, temperature 1, strict last-`\boxed{}` grader), job 15011145:

| Generator | Accuracy | Finished | Problems with both outcomes |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.376 | 0.999 | 259 / 500 |
| Qwen2.5-1.5B-Instruct | 0.634 | 1.000 | 261 / 500 |
| Qwen2.5-Math-1.5B-Instruct | 0.814 | 0.994 | 132 / 500 |
| R1-Distill-Qwen-1.5B (3,584-token cap) | 0.749 | 0.776 | 168 / 500 |

## How to run

```bash
# vis node: envs, downloads, page build, screenshots
ssh della-vis1; cd /scratch/gpfs/GROUP/USER/project/frozen-critic
.venv-critic/bin/python site/phase1_curves.py && cd site && ../.venv-critic/bin/python build.py   # curves, record, page, site/_artifact
PLAYWRIGHT_BROWSERS_PATH=$PWD/.venv-shot/browsers .venv-shot/bin/python site/shot.py
# login node: jobs
sbatch scripts/phase0.sbatch                                  # pli-low copy (1 GPU)
sbatch --partition=pli-c --qos=pli-short scripts/phase0.sbatch  # pli-short copy; logs/phase0.lock picks one
```

Envs: `.venv-critic` (Python 3.12, torch 2.11.0+cu130, transformers 5.6.2, flash-attn 2.8.3.post1 built for sm90),
`.venv-gen` (vLLM 0.30.0, torch 2.13.0), `.venv-shot` (Playwright). Critic weights: `models/limite-1b-value-model`
(HF sha ce83740e). Random actors: `models/random-qwen2.5-{0.5b,1.5b}` (HF default init, seed 0, vocab cut to the
151,665 real ids).

## Notes

- The critic API scores one sequence per call. `src/critic.py` packs many sequences into one row: positions restart
  at 0 per sequence and the backbone's boundary function is swapped for one that reads the restarts. Every other
  GMPT op is per token, so packing is exact up to bf16 kernel noise (2.5e-3 max on CPU eager).
- Critic token ids: Qwen2.5 ids plus `<think>` 151665 and `</think>` 151666; valid ids < 151,667. A random Qwen2.5
  actor has 151,936 embedding rows; rows >= 151,665 must be masked (0.18% of samples per token otherwise).
- Completed responses end with `<|endoftext|>` (151643) for the critic, whatever the generator's stop token.
