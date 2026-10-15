"""Can a generator answer the screen's 200 GSM8K problems without writing steps? (owner 2026-10-14: "even the smallest
Qwen can do math")

The critic screens ask for the answer with no reasoning before it (the 32B ORZ critic's top-valued k is the gold on
19.0% of these problems in its native format). This gives an instruct model the same problems and forces the same
setting: the assistant turn is prefilled with "\\boxed{" so the model writes only the number (greedy, up to 8 tokens).
Phase 0 measured the same models with free reasoning (results/raw/gen_*.jsonl, 500 other test problems).

  python direct_answer.py --model Qwen/Qwen2.5-0.5B-Instruct [--n 200] [--device cpu] -> results/direct_<name>.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=1000)  # the same filter as the screens, so the same 200 problems
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    samples = [s for s in torch.load(EVAL_DUMP, weights_only=False)["samples"]
               if re.fullmatch(r"\d+", s["label"]) and int(s["label"]) < a.kmax]
    samples = random.Random(0).sample(samples, min(a.n, len(samples)))
    rows, hit = [], []
    for s in samples:
        msgs = [{"role": "user", "content": question_of(s["prompt"]) + "\nGive only the final number in \\boxed{}, no steps."}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\\boxed{"
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.match(r"\s*(-?[\d,]*\.?\d+)", text)
        pred = m.group(1).replace(",", "") if m else None
        ok = pred is not None and re.fullmatch(r"-?\d+(\.0+)?", pred) is not None and int(float(pred)) == int(s["label"])
        hit.append(float(ok))
        rows.append({"index": s["index"], "gold": s["label"], "out": text, "pred": pred, "correct": ok})
    name = a.model.split("/")[-1]
    res = {"model": a.model, "n": len(samples), "top1": boot(hit), "examples": rows[:10]}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
    (ROOT / "results" / f"direct_{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / f"direct_{name}_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
