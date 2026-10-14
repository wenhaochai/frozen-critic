"""Which part of the ORZ answer format holds the critic's answer knowledge? (owner 2026-10-14, after the ORZ screen)

The ORZ screen (screen_critic.py) found the 32B critic's top-valued k in 0..999 is the gold on 19.0% of 200 problems
with the native response " </think> <answer> \\boxed{k} </answer>", but on 5.0% with "\\boxed{k}" right after "<think>".
The digits and "\\boxed{...}" are the same Qwen ids in both (Qwen2.5 splits numbers into single digits), so the gap is
the wrapper. Each variant below adds or removes one piece of it; the value is read at the variant's last token.

  bare          \\boxed{k}                                 (= screen "char", 5.0%)
  space         " \\boxed{k}"                              the space-backslash token the native string uses
  think         " </think> \\boxed{k}"                     closes the reasoning only
  answer        " <answer> \\boxed{k} </answer>"           answer tags, reasoning never closed
  native_open   " </think> <answer> \\boxed{k}"            native, read before " </answer>"
  native        " </think> <answer> \\boxed{k} </answer>"  (= screen "native", 19.0%)
  native_char   native written one character per token     the tags as single characters

Same 200 problems and derangement as diag_answer_open.py and screen_critic.py; own question vs another question.
  python screen_format.py --critic orz-critic-32B [--n 200] [--kmax 1000] [--chunk 100] -> results/format_<critic>.json
"""
import argparse
import json
import random
import re

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from common import ROOT
from diag_answer_pick import CHAR_TOK, EVAL_DUMP, boot
from screen_critic import ORZ, Critic, question_of

FORMATS = {
    "bare": "\\boxed{%d}",
    "space": " \\boxed{%d}",
    "think": " </think> \\boxed{%d}",
    "answer": " <answer> \\boxed{%d} </answer>",
    "native_open": " </think> <answer> \\boxed{%d}",
    "native": " </think> <answer> \\boxed{%d} </answer>",
    "native_char": " </think> <answer> \\boxed{%d} </answer>",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = ROOT / "models" / a.critic
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(path)
    ctok = AutoTokenizer.from_pretrained(CHAR_TOK)
    cmap = json.load(open(CHAR_TOK / "critic_ids.json"))  # char-vocab id -> Qwen id
    char_ids = lambda s: [cmap[i] for i in ctok.encode(s, add_special_tokens=False)]
    cr = Critic(path, a.device, dtype)
    samples = [s for s in torch.load(EVAL_DUMP, weights_only=False)["samples"]
               if re.fullmatch(r"\d+", s["label"]) and int(s["label"]) < a.kmax]
    rng = random.Random(0)
    samples = rng.sample(samples, min(a.n, len(samples)))
    perm = list(range(len(samples)))
    while any(i == j for i, j in enumerate(perm)):
        rng.shuffle(perm)
    prefixes = [tok.encode(ORZ.replace("{q}", question_of(s["prompt"])), add_special_tokens=False) for s in samples]
    res, rows = {"critic": a.critic, "n_problems": len(samples), "chance_top1": 1 / a.kmax, "formats": {}}, []
    for fmt in a.formats.split(","):
        enc = char_ids if fmt == "native_char" else (lambda s: tok.encode(s, add_special_tokens=False))
        resp = [enc(FORMATS[fmt] % k) for k in range(a.kmax)]
        pv = [np.array(cr.after_prefix(p, resp, a.chunk)) for p in prefixes]
        res["formats"][fmt] = {"example_tokens": [tok.decode([i]) for i in resp[min(123, a.kmax - 1)]]}
        for arm in ("true", "swapped"):
            top1, rank, amax, gold = [], [], [], []
            for i, s in enumerate(samples):
                g, v = int(s["label"]), pv[i if arm == "true" else perm[i]]
                r = 1 + int((v > v[g]).sum())
                top1.append(float(r == 1)); rank.append(r); amax.append(int(v.argmax())); gold.append(g)
                rows.append({"fmt": fmt, "arm": arm, "index": s["index"], "gold": g, "gold_rank": r,
                             "top5": [[int(k), round(float(v[k]), 5)] for k in np.argsort(-v)[:5]]})
            res["formats"][fmt][arm] = {"top1": boot(top1), "median_rank": float(np.median(rank)),
                                        "spearman_argmax_gold": float(spearmanr(amax, gold)[0]),
                                        "value_range": [float(np.mean([x.min() for x in pv])), float(np.mean([x.max() for x in pv]))]}
            print(fmt, arm, res["formats"][fmt][arm], flush=True)
    name = a.out or f"format_{a.critic}"
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / f"{name}_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
