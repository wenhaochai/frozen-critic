"""Save a randomly initialised Qwen2.5 model (config only, HF default init) for sampling."""
import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
cfg = AutoConfig.from_pretrained(args.base)
model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
# Keep only the 151,665 real tokenizer ids: rows above are embedding padding, and a random head would sample them
# (271/151,936 = 0.18% per token). Truncation equals masking those logits; RL needs the same mask.
model.resize_token_embeddings(151665, mean_resizing=False)
out = Path(args.out)
model.save_pretrained(out)
AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
std = model.model.embed_tokens.weight.float().std().item()
print(f"saved {out}: {sum(p.numel() for p in model.parameters())/1e9:.3f}B params, embed std {std:.4f}")
