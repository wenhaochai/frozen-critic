"""Readings of results/answer_open_scores.jsonl that diag_answer_open.py does not write (2026-10-13), for figure 4.

Constant baseline: the most common answer of GSM8K train ("4", 210 of 7,473), chosen without looking at test, on the same
200 problems and on all of test; paired bootstrap of (critic argmax is gold) - (gold is that constant) over the 200.
Size: Spearman of the critic's argmax with the gold, per arm.

  python diag_answer_open_extra.py  -> results/answer_open_extra.json
"""
import collections
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import ROOT
from diag_answer_pick import EVAL_DUMP, boot


def main():
    rows = [json.loads(l) for l in open(ROOT / "results/answer_open_scores.jsonl")]
    train = pd.read_parquet(ROOT / "data/gsm8k_train.parquet")
    const, n_const = collections.Counter(train.label).most_common(1)[0]
    test = torch.load(EVAL_DUMP, weights_only=False)["samples"]
    own = [r for r in rows if r["arm"] == "true"]
    res = {"const": const, "const_train_count": n_const, "n_train": len(train),
           "const_test": boot([float(s["label"] == const) for s in test]), "n_test": len(test),
           "const_200": boot([float(str(r["gold"]) == const) for r in own]),
           "critic_minus_const_200": boot([float(r["gold_rank"] == 1) - float(str(r["gold"]) == const) for r in own]),
           "hits_gold_max": max(r["gold"] for r in own if r["gold_rank"] == 1),
           "frac_gold_le_20": float(np.mean([r["gold"] <= 20 for r in own]))}
    for arm in ("true", "swapped"):
        R = [r for r in rows if r["arm"] == arm]
        res[f"spearman_{arm}"] = float(spearmanr([r["top5"][0][0] for r in R], [r["gold"] for r in R])[0])
    (ROOT / "results/answer_open_extra.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
