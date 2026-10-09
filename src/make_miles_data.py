"""GSM8K in miles' prompt format: messages = one user turn in the critic's training format (question, blank line,
the boxed-answer instruction), label = the gold number. Written to data/gsm8k_{train,test}.parquet, and an
ASCII-normalized copy data/gsm8k_{train,test}_ascii.parquet for the character-level actor (src/make_char_actor.py),
whose vocabulary has only printable ASCII: 368 of 8,792 questions have a non-ASCII character (mostly the apostrophe ’)."""
import pandas as pd

from common import ROOT, load_gsm8k, user_prompt

ASCII = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " ",
         "​": "", " ": "\n", "€": "EUR ", "\xa3": "GBP ", "\xa2": " cents", "\xb3": "^3", "\xbc": "1/4",
         "\xbe": "3/4", "\xe7": "c", "\xe8": "e", "\xe9": "e", "\xf1": "n", "А": "A"}


def to_ascii(s: str) -> str:
    s = "".join(ASCII.get(ch, ch) for ch in s)
    bad = sorted({ch for ch in s if not (ch == "\n" or 0x20 <= ord(ch) < 0x7F)})
    assert not bad, bad
    return s


out = ROOT / "data"
out.mkdir(exist_ok=True)
for split in ("train", "test"):
    rows = load_gsm8k(split)
    for suffix, f in (("", lambda s: s), ("_ascii", to_ascii)):
        df = pd.DataFrame({"messages": [[{"role": "user", "content": f(user_prompt(r["question"]))}] for r in rows],
                           "label": [r["gold"] for r in rows], "idx": [r["idx"] for r in rows]})
        df.to_parquet(out / f"gsm8k_{split}{suffix}.parquet")
        print(split + suffix, len(df), repr(df.iloc[0]["messages"][0]["content"][-60:]), df.iloc[0]["label"])
