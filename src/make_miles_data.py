"""GSM8K in miles' prompt format: messages = one user turn in the critic's training format (question, blank line,
the boxed-answer instruction), label = the gold number. Written to data/gsm8k_{train,test}.parquet."""
import pandas as pd

from common import ROOT, load_gsm8k, user_prompt

out = ROOT / "data"
out.mkdir(exist_ok=True)
for split in ("train", "test"):
    rows = load_gsm8k(split)
    df = pd.DataFrame({"messages": [[{"role": "user", "content": user_prompt(r["question"])}] for r in rows],
                       "label": [r["gold"] for r in rows], "idx": [r["idx"] for r in rows]})
    df.to_parquet(out / f"gsm8k_{split}.parquet")
    print(split, len(df), df.iloc[0]["messages"][0]["content"][-80:], df.iloc[0]["label"])
