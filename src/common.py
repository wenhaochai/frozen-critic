"""Shared paths, the GSM8K prompt and the answer grader."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITIC_DIR = ROOT / "models" / "limite-1b-value-model"
RESULTS = ROOT / "results"
GSM8K_DIR = Path("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
                 "740312add88f781978c0658806c59bc2815b9866/main")

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CRITIC_EOS = 151643          # <|endoftext|>: the critic's end of a completed rollout
CRITIC_VOCAB = 151667        # valid critic ids are [0, 151667)
QWEN25_VOCAB = 151665        # Qwen2.5 tokenizer size; ids >= this are padding rows of the actor's embedding


def user_prompt(question: str) -> str:
    """The critic's training prompt: the question, a blank line, the fixed instruction."""
    return f"{question}\n\n{INSTRUCTION}"


def load_gsm8k(split="test"):
    import pandas as pd
    df = pd.read_parquet(GSM8K_DIR / f"{split}-00000-of-00001.parquet")
    rows = []
    for i, r in df.iterrows():
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"idx": int(i), "question": r["question"], "gold": gold, "solution": r["answer"]})
    return rows


def last_boxed(text: str):
    """Content of the last \\boxed{...}, with nested braces; None if absent or unclosed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


_NUM = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")  # a comma left in place splits "3,4" into two numbers -> not an answer


def normalize_number(s):
    """The single number inside a boxed answer, or None. Handles $, %, thousands separators (, and \\,),
    \\text{...} wrappers and a plain \\frac{a}{b}; anything with more than one number is not an answer."""
    if s is None:
        return None
    s = s.replace("\u2212", "-").replace("{,}", "").replace("\\!", "").replace("\\,", "").replace("\\$", "").replace("$", "")
    # a comma is a thousands separator only before exactly three digits (18,000); otherwise it separates a list
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = s.replace("\\%", "").replace("%", "").replace("\\left", "").replace("\\right", "")
    frac = re.fullmatch(r"\s*(-?)\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}\s*", s)
    if frac:
        a, b = int(frac.group(2)), int(frac.group(3))
        return (-1 if frac.group(1) else 1) * a / b if b else None
    slash = re.fullmatch(r"\s*(-?\d+)\s*/\s*(\d+)\s*", s)
    if slash:
        a, b = int(slash.group(1)), int(slash.group(2))
        return a / b if b else None
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|textrm)\{([^}]*)\}", r" \1 ", s)
    s = s.replace("{", " ").replace("}", " ").strip()
    m = _NUM.findall(s)
    if len(m) != 1:
        return None
    return float(m[0])


def grade(response_text: str, gold: str) -> bool:
    """Strict GSM8K reward: the last \\boxed{} must hold exactly the gold number."""
    pred = normalize_number(last_boxed(response_text))
    return pred is not None and abs(pred - float(gold)) < 1e-6
