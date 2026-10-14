"""Offline screen of another frozen critic (owner 2026-10-13: is Limite just too weak?), before any new PPO run.

Critics: the Open-Reasoner-Zero PPO critics (Qwen2.5 base + a linear value head, OpenRLHF layout, sizes 0.5B to 32B),
fed the ORZ prompt template they were trained on. V(prefix) is the head on the last token's hidden state, as in
ORZ (values[:, :-1]); the head's mean/std buffers are not saved, so values are raw. Token ids are Qwen2.5 ids, the
same ids the character actor's tokens map to (models/actor-q35d05b-char-lm/critic_ids.json).

Three tests, the same as for Limite (results/answer_open.json, docs/STATUS.md):
  open    for the 200 problems of diag_answer_open.py, every k in 0..999: is the critic's top-valued response the
          gold? Own question vs another problem's question (the same derangement). Two response formats:
          char   "\\boxed{k}" written one character per token right after "<think>", as the char actor writes it
          native " </think> <answer> \\boxed{k} </answer>" in Qwen BPE, the format the critic was trained on
          The value is read at the last response token, the state before the stop token.
  random  rollout 0 of ppo_char_lam0.95_s42 (512 responses of the random-init char actor): V(prompt), the change
          after the first character, and the value before the stop token. Limite: 0.71, -0.077, 0.54.
  rank    the Phase 0 generations (results/raw/gen_*.jsonl, 500 problems x 4 samples per generator): AUC of the value
          before the last token, overall and within each problem, and the mean value of correct and wrong samples.

  python screen_critic.py --critic orz-critic-7B [--n 200] [--kmax 1000] [--device cpu] -> results/screen_<critic>.json
"""
import argparse
import copy
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from common import ROOT
from diag_answer_pick import CHAR_TOK, EOS, EVAL_DUMP, boot
from phase0_critic import auc, within_problem_auc

ORZ = ("A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant "
       "first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning "
       "process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
       "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer "
       "inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted "
       "automatically by the \\boxed{} tag.\nThis is the problem:\n{q}\nAssistant: <think>")
GENS = ["qwen0.5b-instruct", "qwen1.5b-instruct", "qwen-math1.5b-instruct", "r1distill1.5b"]


def question_of(chat_prompt):
    """The question inside the char run's chat prompt (ASCII, as the char actor reads it)."""
    return chat_prompt.split("<|im_start|>user\n", 1)[1].split("\n\nPlease reason", 1)[0]


class Critic:
    def __init__(self, path, device, dtype):
        self.dev = device
        self.model = AutoModel.from_pretrained(path, dtype=dtype, attn_implementation="sdpa", device_map=device).eval()
        idx = path / "model.safetensors.index.json"
        shard = json.load(open(idx))["weight_map"]["value_head.weight"] if idx.exists() else "model.safetensors"
        with safe_open(str(path / shard), "pt") as f:
            w = f.get_tensor("value_head.weight")
        self.head = torch.nn.Linear(w.shape[1], 1, bias=False, dtype=dtype, device=device)
        self.head.weight.data.copy_(w)

    @torch.no_grad()
    def full(self, seqs, budget=32768):
        """Per-position values of each sequence (right-padded batches; causal attention never sees the padding)."""
        out = [None] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        i = 0
        while i < len(order):
            L = len(seqs[order[i]]); b = max(1, budget // L)
            idx = order[i:i + b]; i += b
            ids = torch.zeros(len(idx), L, dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, :len(seqs[j])] = torch.tensor(seqs[j])
            h = self.model(input_ids=ids.to(self.dev)).last_hidden_state
            v = self.head(h).squeeze(-1).float().cpu()
            for r, j in enumerate(idx):
                out[j] = v[r, :len(seqs[j])].numpy()
        return out

    @torch.no_grad()
    def after_prefix(self, prefix, cands, chunk):
        """Value at the last token of prefix + c for each continuation c; the prefix is run once (KV cache)."""
        o = self.model(input_ids=torch.tensor([prefix], device=self.dev), use_cache=True)
        P, res = len(prefix), []
        for s in range(0, len(cands), chunk):
            cs = cands[s:s + chunk]; L = max(map(len, cs))
            ids = torch.zeros(len(cs), L, dtype=torch.long)
            for r, c in enumerate(cs):
                ids[r, :len(c)] = torch.tensor(c)
            cache = copy.deepcopy(o.past_key_values)  # the prefix once per row
            cache.batch_repeat_interleave(len(cs))
            pos = torch.arange(P, P + L, device=self.dev)[None].expand(len(cs), -1)
            h = self.model(input_ids=ids.to(self.dev), past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
            v = self.head(h).squeeze(-1).float().cpu()
            res += [float(v[r, len(c) - 1]) for r, c in enumerate(cs)]
        return res



def test_open(cr, tok, samples, perm, kmax, char_ids, chunk):
    res, rows = {}, []
    for fmt in ("char", "native"):
        if fmt == "char":
            resp = {k: char_ids("\\boxed{%d}" % k) for k in range(kmax)}
        else:
            resp = {k: tok.encode(" </think> <answer> \\boxed{%d} </answer>" % k, add_special_tokens=False) for k in range(kmax)}
        pv = {}
        for i, s in enumerate(samples):
            pv[i] = np.array(cr.after_prefix(tok.encode(ORZ.replace("{q}", question_of(s["prompt"])), add_special_tokens=False),
                                             [resp[k] for k in range(kmax)], chunk))
        for arm in ("true", "swapped"):
            top1, rank, amax, gold = [], [], [], []
            for i, s in enumerate(samples):
                g, v = int(s["label"]), pv[i if arm == "true" else perm[i]]
                r = 1 + int((v > v[g]).sum())
                top1.append(float(r == 1)); rank.append(r); amax.append(int(v.argmax())); gold.append(g)
                rows.append({"fmt": fmt, "arm": arm, "index": s["index"], "gold": g, "gold_rank": r,
                             "top5": [[int(k), round(float(v[k]), 5)] for k in np.argsort(-v)[:5]], "gold_value": float(v[g])})
            res[f"{fmt}_{arm}"] = {"top1": boot(top1), "median_rank": float(np.median(rank)),
                                   "spearman_argmax_gold": float(spearmanr(amax, gold)[0]),
                                   "value_range": [float(np.mean([pv[i].min() for i in pv])), float(np.mean([pv[i].max() for i in pv]))]}
            print(fmt, arm, res[f"{fmt}_{arm}"], flush=True)
    return res, rows


def test_random(cr, tok, char_map, n):
    d = torch.load(ROOT / "runs/ppo_char_lam0.95_s42/rollouts/0.pt", weights_only=False)["samples"][:n]
    seqs, P = [], []
    for s in d:
        p = tok.encode(ORZ.replace("{q}", question_of(s["prompt"])), add_special_tokens=False)
        r = [char_map[t] for t in s["tokens"][-s["response_length"]:]]
        seqs.append(p + r); P.append(len(p))
    vals = cr.full(seqs)
    vp = np.array([v[p - 1] for v, p in zip(vals, P)])          # V(prompt)
    v1 = np.array([v[p] for v, p in zip(vals, P)])              # after the first character
    vl = np.array([v[-2] if len(v) - p >= 2 else v[p - 1] for v, p in zip(vals, P)])  # before the last token
    out = {"n": len(vals), "v_prompt": float(vp.mean()), "first_char_change": float((v1 - vp).mean()),
           "first_char_negative": float(((v1 - vp) < 0).mean()), "v_before_stop": float(vl.mean()),
           "mean_response_tokens": float(np.mean([len(v) - p for v, p in zip(vals, P)]))}
    print("random", out, flush=True)
    return out


def test_rank(cr, tok, max_per_gen):
    qs = {int(r.idx): r.messages[0]["content"].split("\n\nPlease reason", 1)[0]
          for r in pd.read_parquet(ROOT / "data/gsm8k_test.parquet").itertuples()}
    out = {}
    for g in GENS:
        recs = [json.loads(l) for l in open(ROOT / f"results/raw/gen_{g}.jsonl")][:max_per_gen]
        seqs, P = [], []
        for r in recs:
            p = tok.encode(ORZ.replace("{q}", qs[r["idx"]]), add_special_tokens=False)
            seqs.append(p + r["response_ids"]); P.append(len(p))
        vals = cr.full(seqs, budget=16384)
        for r, v, p in zip(recs, vals, P):
            r["last"] = float(v[-2] if len(v) - p >= 2 else v[p - 1])
            r["prompt_value"] = float(v[p - 1])
        c = [r["last"] for r in recs if r["correct"]]; w = [r["last"] for r in recs if not r["correct"]]
        wa, n = within_problem_auc(recs, "last")
        out[g] = {"n": len(recs), "accuracy": len(c) / len(recs), "auc_last": auc(c, w), "within_auc_last": wa, "within_pairs": n,
                  "mean_last_correct": float(np.mean(c)) if c else None, "mean_last_wrong": float(np.mean(w)) if w else None,
                  "mean_prompt_value": float(np.mean([r["prompt_value"] for r in recs]))}
        print("rank", g, out[g], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True)  # a directory under models/
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=1000)
    ap.add_argument("--rank_per_gen", type=int, default=2000)
    ap.add_argument("--random_n", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=250)  # candidates per batch over the cached prompt (100 for 32B)
    ap.add_argument("--tests", default="open,random,rank")
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
    res = {"critic": a.critic}
    if "open" in a.tests:
        # the same 200 problems and derangement as diag_answer_open.py
        samples = [s for s in torch.load(EVAL_DUMP, weights_only=False)["samples"]
                   if re.fullmatch(r"\d+", s["label"]) and int(s["label"]) < a.kmax]
        rng = random.Random(0)
        samples = rng.sample(samples, min(a.n, len(samples)))
        perm = list(range(len(samples)))
        while any(i == j for i, j in enumerate(perm)):
            rng.shuffle(perm)
        res["open"], rows = test_open(cr, tok, samples, perm, a.kmax, char_ids, a.chunk)
        res["open"]["n_problems"], res["open"]["chance_top1"] = len(samples), 1 / a.kmax
    if "random" in a.tests:
        res["random"] = test_random(cr, tok, cmap, a.random_n)
    if "rank" in a.tests:
        res["rank"] = test_rank(cr, tok, a.rank_per_gen)
    name = a.out or f"screen_{a.critic}"
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
    if "open" in a.tests:
        with open(ROOT / "results" / f"{name}_open_scores.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
