"""Phase 1 curves for page figure 2 -> results/phase1_char_curves.json (run on della-vis1 with .venv-critic).

Per character-level run (ppo_char_lam*_s42) and training rollout (from the debug dumps runs/RUN/rollouts/ID.pt): the
mean critic value before the last response token (reward['values'][-1]), the fraction of responses that the strict
grader marks correct (reward['score'] > 0) and the fraction of finished responses whose last box holds a number
(reward['format'], not truncated). Per eval point (from console.log, greedy decoding on the 1,319 GSM8K test problems): accuracy.
"""
import ast
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase1_char_curves.json"
LAMBDAS = ["0.95", "0.5", "0"]


def run_curves(run):
    d = ROOT / "runs" / run
    train = []
    if not (d / "rollouts").exists():
        return {"train": [], "eval": []}
    for f in sorted((p for p in (d / "rollouts").glob("*.pt") if p.stem.isdigit()), key=lambda p: int(p.stem)):
        s = torch.load(f, weights_only=False)["samples"]
        v = [x["reward"]["values"][-1] for x in s]
        boxed = sum(x["reward"].get("format", 0) > 0 and not x["reward"]["truncated"] for x in s)
        train.append([int(f.stem), sum(v) / len(v), sum(x["reward"]["score"] > 0 for x in s) / len(s), boxed / len(s)])
    ev = {}
    log = d / "console.log"
    if log.exists():
        for m in re.finditer(r"eval (\d+): (\{.*?\})\s*$", log.read_text(errors="replace"), re.M):
            ev[int(m.group(1))] = ast.literal_eval(m.group(2))["eval/gsm8k"]  # a resumed run repeats points: keep last
    return {"train": train, "eval": sorted([k, a] for k, a in ev.items())}


if __name__ == "__main__":
    res = {l: run_curves(f"ppo_char_lam{l}_s42") for l in LAMBDAS}
    OUT.write_text(json.dumps(res))
    for l, r in res.items():
        print(l, len(r["train"]), "rollouts", r["train"][-1:], "eval", r["eval"][-2:])
