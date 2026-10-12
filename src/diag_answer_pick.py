"""Does the frozen critic know the answer? Offline diagnostic before any template-answer RL (owner, 2026-10-11).

For each GSM8K test problem (the exact ASCII prompts of the char-run eval dump), the critic scores the response
"\\boxed{k}" followed by the stop token, for the gold answer k and 20 distractors, and we read V(prompt + "\\boxed{k}"),
the value before the stop token. If the critic ranks the gold answer first far more often than 1 in 21, the answer is
in the value and a template actor (environment writes the box, the actor writes the digits) could learn it from the
critic; if not, no RL design can extract it.

Response variants (critic token ids):
  char        the box written one character per token, as the char actor writes it (critic_ids.json mapping)
  think_char  "<think>\\n\\n</think>\\n\\n" (critic BPE) before it, the format the critic was trained on
  bpe         the critic's own BPE tokens for the box
  think_bpe   both
Distractors, fixed per problem (seeded by its index): gold +-1, +-2, +-10, 2 x gold, gold // 2, up to five numbers from
the question, then random numbers with the gold's digit count. Metrics per variant: top-1 accuracy (chance 1/21), mean
fraction of distractors scored below gold (within-problem AUC), the same against near distractors (+-1, +-2) and against
the question's numbers, each with a 95% bootstrap interval over problems.

  python diag_answer_pick.py [--limit N] [--device cpu --backend eager]   -> results/answer_pick.json, answer_pick_scores.jsonl
"""
import argparse
import json
import random
import re

import numpy as np
import torch
from transformers import AutoTokenizer

import critic
from common import ROOT

EVAL_DUMP = ROOT / "runs/ppo_char_lam0.95_s42/rollouts/eval_0.pt"
CHAR_TOK = ROOT / "models/actor-q35d05b-char-lm"
CRITIC_TOK = ROOT / "models/actor-q35d1b-lm"
EOS = 151643
VARIANTS = ["char", "think_char", "bpe", "think_bpe"]


def distractors(gold, question, idx, n=20):
    rng = random.Random(idx)
    near = [gold + d for d in (1, -1, 2, -2)]
    cands = near + [gold + 10, gold - 10, 2 * gold, gold // 2]
    qnums = [int(x) for x in re.findall(r"(?<![\d.])\d+(?![\d.])", question.replace(",", ""))][:5]
    cands += qnums
    out = []
    for c in cands:
        if c >= 0 and c != gold and c not in out:
            out.append(c)
    # random fill with the gold's digit count; a one-digit gold has only 9 such numbers, so it draws from 0..99
    lo, hi = (10 ** (len(str(gold)) - 1), 10 ** len(str(gold)) - 1) if gold >= 10 else (0, 99)
    while len(out) < n:
        c = rng.randint(lo, hi)
        if c != gold and c not in out:
            out.append(c)
    out = out[:n]
    kind = {c: ("near" if c in near else "question" if c in qnums else "other") for c in out}
    return out, kind


def boot(vals, reps=1000, seed=0):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)]
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="flash_attention_2")
    ap.add_argument("--out", default="answer_pick")
    a = ap.parse_args()
    samples = torch.load(EVAL_DUMP, weights_only=False)["samples"]
    if a.limit:
        samples = samples[:a.limit]
    ctok, btok = AutoTokenizer.from_pretrained(CHAR_TOK), AutoTokenizer.from_pretrained(CRITIC_TOK)
    cmap = json.load(open(CHAR_TOK / "critic_ids.json"))
    think = btok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    def resp(variant, k):
        box = "\\boxed{%d}" % k
        if "char" in variant:
            ids = [cmap[i] for i in ctok.encode(box, add_special_tokens=False)]
            assert len(ids) == len(box) - len("boxed") + 1, (box, ids)  # one token per character, 'boxed' is one token
        else:
            ids = btok.encode(box, add_special_tokens=False)
        return (think if variant.startswith("think") else []) + ids + [EOS]

    probs, items = [], []
    for s in samples:
        if not re.fullmatch(r"\d+", s["label"]):
            continue
        gold, i = int(s["label"]), s["index"]
        q = s["prompt"].split("<|im_start|>user\n", 1)[1]
        ds, kind = distractors(gold, q.split("\n\nPlease reason")[0], i)
        p_ids = btok.encode(s["prompt"], add_special_tokens=False)
        probs.append({"index": i, "gold": gold, "cands": [gold] + ds, "kind": kind})
        for v in VARIANTS:
            for k in [gold] + ds:
                items.append(((i, v, k), p_ids, resp(v, k)))
    m = critic.load(device=a.device, backend=a.backend)
    sc = critic.score_items(m, items, max_tokens=65536 if a.backend != "eager" else 4096)
    res, rows = {}, []
    for v in VARIANTS:
        top1, auc, near, qn, gap = [], [], [], [], []
        for p in probs:
            val = {k: float(sc[(p["index"], v, k)]["pre"][-1]) for k in p["cands"]}
            g = val[p["gold"]]
            ds = p["cands"][1:]
            top1.append(float(all(g > val[d] for d in ds)))
            beat = lambda S: np.mean([1.0 if g > val[d] else 0.5 if g == val[d] else 0.0 for d in S]) if S else None
            auc.append(beat(ds))
            n_, q_ = beat([d for d in ds if p["kind"][d] == "near"]), beat([d for d in ds if p["kind"][d] == "question"])
            if n_ is not None: near.append(n_)
            if q_ is not None: qn.append(q_)
            gap.append(g - float(np.mean([val[d] for d in ds])))
            rows.append({"index": p["index"], "variant": v, "gold": p["gold"], "values": val,
                         "v_prompt": float(sc[(p["index"], v, p["gold"])]["pre"][0])})
        res[v] = {"n_problems": len(top1), "chance_top1": 1 / 21, "top1": boot(top1), "auc": boot(auc),
                  "auc_near": boot(near), "auc_question_numbers": boot(qn), "gold_minus_mean_distractor": boot(gap)}
        print(v, {k: (np.round(x, 4).tolist() if isinstance(x, list) else x) for k, x in res[v].items()}, flush=True)
    (ROOT / "results" / f"{a.out}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / f"{a.out}_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
