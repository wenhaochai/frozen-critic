"""Can a generator answer the screen's 200 GSM8K problems without writing steps? (owner 2026-10-14: "even the smallest
Qwen can do math"; 2026-10-15: use the smallest Qwen3.5, not only Qwen2.5)

The critic screens ask for the answer with no reasoning before it (the 32B ORZ critic's top-valued k is the gold on
19.0% of these problems in its native format). Two modes on the same problems:
  direct  the assistant turn is prefilled with "\\boxed{" so the model writes only the number (greedy, up to 8 tokens)
  steps   the model reasons first ("Please reason step by step, ..."), greedy, up to 2,048 tokens; last \\boxed{} graded
          (the smoke run cut 1 of 4 at 1,024 tokens; "truncated" is the share that hit the cap without an end token)
Thinking mode stays off (Qwen3.5's template turns it on only when asked).

  python direct_answer.py --model Qwen/Qwen3.5-0.8B [--mode direct|steps] [--n 200] -> results/direct_<name>[_steps].json
"""
import argparse
import json
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ROOT
from diag_answer_pick import EVAL_DUMP, boot
from screen_critic import question_of

ASK = {"direct": "\nGive only the final number in \\boxed{}, no steps.",
       "steps": "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}


def grade(pred, gold):
    return pred is not None and re.fullmatch(r"-?\d+(\.0+)?", pred) is not None and int(float(pred)) == int(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="direct", choices=list(ASK))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=1000)  # the same filter as the screens, so the same 200 problems
    ap.add_argument("--batch", type=int, default=50)  # steps mode only
    ap.add_argument("--max_new", type=int, default=2048)  # steps mode only
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    samples = [s for s in torch.load(EVAL_DUMP, weights_only=False)["samples"]
               if re.fullmatch(r"\d+", s["label"]) and int(s["label"]) < a.kmax]
    samples = random.Random(0).sample(samples, min(a.n, len(samples)))
    prompts = [tok.apply_chat_template([{"role": "user", "content": question_of(s["prompt"]) + ASK[a.mode]}],
                                       tokenize=False, add_generation_prompt=True) for s in samples]
    texts, cut = [], []
    if a.mode == "direct":
        for p in prompts:
            ids = tok(p + "\\boxed{", return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False)
            texts.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    else:
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        for i in range(0, len(prompts), a.batch):
            enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]
            ends = {tok.pad_token_id, tok.eos_token_id, *([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int) else model.generation_config.eos_token_id or [])}
            cut += [gen.shape[1] == a.max_new and int(g[-1]) not in ends for g in gen]
            texts += tok.batch_decode(gen, skip_special_tokens=True)
    rows, hit = [], []
    cut = cut or [False] * len(texts)
    for s, text, c in zip(samples, texts, cut):
        if a.mode == "direct":
            m = re.match(r"\s*(-?[\d,]*\.?\d+)", text)
        else:
            m = ([None] + list(re.finditer(r"\\boxed\{\s*(-?[\d,]*\.?\d+)\s*\}", text)))[-1]
        pred = m.group(1).replace(",", "") if m else None
        ok = grade(pred, s["label"])
        hit.append(float(ok))
        rows.append({"index": s["index"], "gold": s["label"], "out": text, "pred": pred, "correct": ok,
                     "n_tokens": len(tok(text, add_special_tokens=False).input_ids), "truncated": c})
    name = a.model.split("/")[-1] + ("" if a.mode == "direct" else "_steps") + ("" if a.n == 200 else f"_n{a.n}")  # smoke runs never overwrite
    res = {"model": a.model, "mode": a.mode, "n": len(samples), "top1": boot(hit),
           "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
           "truncated": sum(r["truncated"] for r in rows) / len(rows), "examples": rows[:10]}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
    (ROOT / "results" / f"direct_{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / f"direct_{name}_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
