"""End-to-end check of the value path on CPU: value server (eager) + miles reward function with a fake Sample.
Run: (cd src; FC_VALUE_URL=http://127.0.0.1:PORT python test_value_path.py PORT) after starting value_server --device cpu."""
import asyncio
import sys
from types import SimpleNamespace

import critic
from common import CRITIC_DIR, CRITIC_EOS, load_gsm8k, user_prompt
from miles_ext.frozen_critic_rm import reward_func
from value_model import tokenize_pair
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(CRITIC_DIR))
row = load_gsm8k("test")[0]
# The prompt exactly as miles builds it: chat template of the actor dir (= critic's), generation prompt on.
msgs = [{"role": "user", "content": user_prompt(row["question"])}]
p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
assert p_ids == tokenize_pair(str(CRITIC_DIR), user_prompt(row["question"]), "a")[0], "prompt differs from the critic's"
resp_text = "Janet sells 16 - 3 - 4 = 9 eggs. 9 * 2 = 18 dollars. The answer is \\boxed{18}."
r_ids = tok.encode(resp_text, add_special_tokens=False) + [CRITIC_EOS]
sample = SimpleNamespace(tokens=p_ids + r_ids, response_length=len(r_ids), response=resp_text, label=row["gold"],
                         status="completed")
out = asyncio.run(reward_func(None, sample))
m = critic.load("cpu", "eager")
ref = m.score_tokens(p_ids, r_ids)["pre_action_values"]
diff = max(abs(a - b) for a, b in zip(out["values"], ref))
print("score", out["score"], "n values", len(out["values"]), "response_length", len(r_ids), "max|server - score_tokens|", round(diff, 5))
assert out["score"] == 1.0 and out["train_score"] == 1.0 and len(out["values"]) == len(r_ids) and diff < 5e-3
cut = SimpleNamespace(tokens=p_ids + r_ids[:-1], response_length=len(r_ids) - 1, response=resp_text, label=row["gold"],
                      status="truncated")
out2 = asyncio.run(reward_func(None, cut))
ref2 = m.score_tokens(p_ids, r_ids[:-1])["raw_value_after_complete_input"]
print("truncated: score", out2["score"], "train_score", round(out2["train_score"], 4), "V(full)", round(ref2, 4))
assert abs(out2["train_score"] - ref2) < 5e-3, "a cut response trains on V(full) alone"
print("OK")
