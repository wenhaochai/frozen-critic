"""Actor-side compute on one H100: forward+backward and forward-only token throughput of randomly initialised
Qwen2.5 models (bf16 weights, FlashAttention-2 when installed), on equal-length sequences without padding.
The optimizer step is left out (it is per parameter, not per token); 7B runs with bf16 weights and grads only."""
import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="0.5B,1.5B,3B,7B")
ap.add_argument("--seq", type=int, default=1152)       # ~130 prompt + 1024 response tokens
ap.add_argument("--tokens-per-step", type=int, default=16384)
args = ap.parse_args()

try:
    import flash_attn  # noqa: F401
    attn = "flash_attention_2"
except ImportError:
    attn = "sdpa"
out = {"attn": attn, "gpu": torch.cuda.get_device_name(0)}
H100_BF16 = 989e12
for size in args.sizes.split(","):
    cfg = AutoConfig.from_pretrained(f"Qwen/Qwen2.5-{size}-Instruct")
    cfg._attn_implementation = attn
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable() if size in ("3B", "7B") else None
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.model.embed_tokens.weight.numel()
    bs = max(1, args.tokens_per_step // args.seq)
    ids = torch.randint(0, 151665, (bs, args.seq), device="cuda")

    def fwd_bwd():
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def fwd():
        model(input_ids=ids, logits_to_keep=0)

    res = {"params_B": n_params / 1e9, "non_embedding_params_B": (n_params - n_embed) / 1e9, "batch": bs, "seq": args.seq,
           "grad_ckpt": size in ("3B", "7B")}
    for name, fn in (("fwd_bwd", fwd_bwd), ("fwd", fwd)):
        try:
            torch.cuda.reset_peak_memory_stats()
            fn(); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            tps = bs * args.seq / dt
            flops_per_tok = (6 if name == "fwd_bwd" else 2) * n_params
            res[name] = {"tok_per_s": tps, "mfu_6N": tps * flops_per_tok / H100_BF16,
                         "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        except torch.cuda.OutOfMemoryError:
            res[name] = "OOM"
            torch.cuda.empty_cache()
    out[size] = res
    print(size, json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
(RESULTS / "bench_actor.json").write_text(json.dumps(out, indent=1))
