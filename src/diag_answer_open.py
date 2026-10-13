"""Open-set follow-up to diag_answer_pick.py (2026-10-12): a template actor that maximises the critic's value picks
among all numbers, not among 21 candidates, so the ceiling of a template-answer PPO arm is how often the critic's
best-scored number is the gold answer.

For a seeded sample of GSM8K test problems with an integer gold answer in 0..999, the critic scores "\\boxed{k}" + stop
for every k in 0..999 (the char variant: one critic token per character, as the char actor writes it), and we read
V(prompt + "\\boxed{k}"), the value before the stop token. Control: the same 1,000 candidates under the prompt of
another problem of the sample (a seeded derangement), which keeps any preference for particular numbers (round
numbers, small numbers) and removes the question. Metrics per arm, with 95% bootstrap intervals over problems: top-1
(the argmax is the gold), top-10, and the gold's rank among the 1,000.

  python diag_answer_open.py [--n 200] [--device cpu --backend eager]  -> results/answer_open.json, answer_open_scores.jsonl
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
from diag_answer_pick import CHAR_TOK, CRITIC_TOK, EOS, EVAL_DUMP, boot

K = range(1000)  # candidates; --kmax shrinks it for a smoke test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="flash_attention_2")
    ap.add_argument("--out", default="answer_open")
    ap.add_argument("--kmax", type=int, default=1000)
    a = ap.parse_args()
    global K
    K = range(a.kmax)
    samples = [s for s in torch.load(EVAL_DUMP, weights_only=False)["samples"]
               if re.fullmatch(r"\d+", s["label"]) and int(s["label"]) < a.kmax]
    rng = random.Random(0)
    samples = rng.sample(samples, min(a.n, len(samples)))
    perm = list(range(len(samples)))
    while any(i == j for i, j in enumerate(perm)):  # derangement: every problem gets another problem's prompt
        rng.shuffle(perm)
    ctok, btok = AutoTokenizer.from_pretrained(CHAR_TOK), AutoTokenizer.from_pretrained(CRITIC_TOK)
    cmap = json.load(open(CHAR_TOK / "critic_ids.json"))
    resp = {k: [cmap[i] for i in ctok.encode("\\boxed{%d}" % k, add_special_tokens=False)] + [EOS] for k in K}
    p_ids = [btok.encode(s["prompt"], add_special_tokens=False) for s in samples]
    items = [((i, arm, k), p_ids[i if arm == "true" else perm[i]], resp[k])
             for i in range(len(samples)) for arm in ("true", "swapped") for k in K]
    m = critic.load(device=a.device, backend=a.backend)
    sc = critic.score_items(m, items, max_tokens=65536 if a.backend != "eager" else 4096)
    res, rows = {"n_problems": len(samples), "chance_top1": 1 / len(K)}, []
    for arm in ("true", "swapped"):
        top1, top10, rank = [], [], []
        for i, s in enumerate(samples):
            g = int(s["label"])
            v = np.array([float(sc[(i, arm, k)]["pre"][-1]) for k in K])
            r = 1 + int((v > v[g]).sum())
            top1.append(float(r == 1)); top10.append(float(r <= 10)); rank.append(r)
            best = np.argsort(-v)[:5]
            rows.append({"index": s["index"], "arm": arm, "gold": g, "prompt_of": samples[i if arm == "true" else perm[i]]["index"],
                         "gold_rank": r, "top5": [[int(k), round(float(v[k]), 5)] for k in best], "gold_value": float(v[g])})
        res[arm] = {"top1": boot(top1), "top10": boot(top10), "mean_rank": boot(rank), "median_rank": float(np.median(rank))}
        print(arm, res[arm], flush=True)
    (ROOT / "results" / f"{a.out}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / f"{a.out}_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
